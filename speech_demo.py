# -*- coding: utf-8 -*-
"""
MAGFORMER - Speech Commands Demo
================================
End-to-end knowledge distillation from a Tiny Transformer to a
Kuramoto Oscillator Network on real audio data (Speech Commands).
"""

import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import soundfile as sf
def safe_load(path, *args, **kwargs):
    waveform, sample_rate = sf.read(path)
    waveform = torch.tensor(waveform, dtype=torch.float32)
    if waveform.ndim == 1: waveform = waveform.unsqueeze(0)
    else: waveform = waveform.t()
    return waveform, sample_rate
torchaudio.load = safe_load

# ---------------------------------------------------------------------
# HYPERPARAMS
# ---------------------------------------------------------------------
WORDS        = ["yes", "no", "up", "down"]
N_CLASSES    = len(WORDS)
N_MFCC       = 16
SEQ_LEN      = 32  # Downsampled time steps
EMBED_DIM    = 16
N_HEADS      = 2
FFN_DIM      = 32
N_OSCILLATORS = 32
BATCH_SIZE   = 128
EPOCHS_T     = 30  # Transformer pretrain epochs
EPOCHS_O     = 100 # Oscillator distillation epochs
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------------------------------------------------------------------
# DATASET
# ---------------------------------------------------------------------
def get_dataloaders():
    print("Loading SpeechCommands dataset...")
    # This assumes dataset is already downloaded to ./SpeechCommands
    try:
        dataset = torchaudio.datasets.SPEECHCOMMANDS('./', download=False)
    except:
        dataset = torchaudio.datasets.SPEECHCOMMANDS('./', download=True)
    
    # Filter for our target words
    word_to_idx = {w: i for i, w in enumerate(WORDS)}
    filtered_data = []
    
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=16000, n_mfcc=N_MFCC, 
        melkwargs={"n_mels": 40, "n_fft": 400, "hop_length": 500} # 16000/500 = 32 steps for 1 sec
    )

    for idx, (waveform, sample_rate, label, speaker_id, utterance_number) in enumerate(dataset):
        if label in word_to_idx:
            # Pad or truncate to exactly 16000 samples (1 sec)
            if waveform.size(1) < 16000:
                waveform = F.pad(waveform, (0, 16000 - waveform.size(1)))
            elif waveform.size(1) > 16000:
                waveform = waveform[:, :16000]
                
            mfcc = mfcc_transform(waveform).squeeze(0).transpose(0, 1) # (time, n_mfcc)
            
            # Ensure exactly SEQ_LEN time steps
            if mfcc.size(0) > SEQ_LEN:
                mfcc = mfcc[:SEQ_LEN, :]
            elif mfcc.size(0) < SEQ_LEN:
                mfcc = F.pad(mfcc, (0, 0, 0, SEQ_LEN - mfcc.size(0)))
                
            filtered_data.append((mfcc, word_to_idx[label]))
            
            if len(filtered_data) >= 4000: # Limit size for quick demo
                break
                
    print(f"Loaded {len(filtered_data)} samples for words: {WORDS}")
    
    # Split train/val (80/20)
    train_size = int(0.8 * len(filtered_data))
    train_data = filtered_data[:train_size]
    val_data = filtered_data[train_size:]
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader

# ---------------------------------------------------------------------
# MODELS
# ---------------------------------------------------------------------
class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(N_MFCC, EMBED_DIM)
        pe = torch.zeros(SEQ_LEN, EMBED_DIM)
        pos = torch.arange(SEQ_LEN).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, EMBED_DIM, 2).float() * (-math.log(10000.0) / EMBED_DIM))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM, nhead=N_HEADS, dim_feedforward=FFN_DIM,
            batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.classifier = nn.Linear(EMBED_DIM, N_CLASSES)

    def forward(self, x):
        x = self.input_proj(x) + self.pe[:, :x.size(1)]
        x = self.transformer(x)
        x = x.mean(dim=1)  # Mean pooling over time
        return self.classifier(x)

class OscillatorNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Flatten sequence and project to oscillators (handles positional info by flattening)
        self.input_proj = nn.Sequential(
            nn.Linear(SEQ_LEN * N_MFCC, 128),
            nn.GELU(),
            nn.Linear(128, N_OSCILLATORS)
        )

        # Physics parameters
        self._omega_raw = nn.Parameter(torch.randn(N_OSCILLATORS) * 0.5)
        self._coupling_raw = nn.Parameter(torch.randn(N_OSCILLATORS, N_OSCILLATORS) * 0.1)
        self._damping_raw = nn.Parameter(torch.zeros(N_OSCILLATORS))

        self.readout = nn.Sequential(
            nn.Linear(N_OSCILLATORS, 32),
            nn.GELU(),
            nn.Linear(32, N_CLASSES)
        )

    @property
    def omega(self): return F.softplus(self._omega_raw)
    @property
    def coupling(self): return F.softplus(self._coupling_raw)
    @property
    def damping(self): return 0.01 + F.softplus(self._damping_raw)

    def forward(self, x):
        B = x.size(0)
        # Flatten time and feature dims
        x_flat = x.reshape(B, -1)
        u = self.input_proj(x_flat) # (B, N_OSC)

        omega = self.omega
        K = self.coupling
        delta_omega = (omega.unsqueeze(1) - omega.unsqueeze(0)).abs()
        sync_weights = torch.clamp(1.0 - delta_omega / (K + 1e-6), min=0.0)
        sync_weights = sync_weights * (1.0 - torch.eye(N_OSCILLATORS, device=x.device))

        coupled = torch.matmul(u, sync_weights.T)
        state = u + coupled
        state = torch.tanh(state)
        state = state * (1.0 / self.damping)
        state = torch.tanh(state)

        return self.readout(state)

# ---------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------
def train_transformer(model, train_loader, val_loader, epochs=30):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    for epoch in range(epochs):
        total_loss, total_correct, samples = 0, 0, 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.size(0)
            total_correct += (logits.argmax(dim=-1) == y).sum().item()
            samples += x.size(0)
            
        if (epoch + 1) % 5 == 0:
            val_acc = eval_model(model, val_loader)
            print(f"  [Transformer] Epoch {epoch+1:2d} | Train Acc: {total_correct/samples*100:.1f}% | Val Acc: {val_acc:.1f}%")

def eval_model(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            correct += (logits.argmax(dim=-1) == y).sum().item()
            total += x.size(0)
    model.train()
    return correct / total * 100

def compile_oscillator(osc_model, transformer, train_loader, val_loader, epochs=100):
    osc_model.train()
    transformer.eval()
    opt = torch.optim.AdamW(osc_model.parameters(), lr=3e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    temp = 4.0

    for epoch in range(epochs):
        total_loss, total_correct, total_agree, samples = 0, 0, 0, 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.no_grad():
                t_logits = transformer(x)
            
            s_logits = osc_model(x)
            loss_kd = F.kl_div(
                F.log_softmax(s_logits / temp, dim=-1),
                F.softmax(t_logits / temp, dim=-1),
                reduction='batchmean'
            ) * (temp ** 2)
            loss_ce = F.cross_entropy(s_logits, y)
            loss = 0.8 * loss_kd + 0.2 * loss_ce
            
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(osc_model.parameters(), 1.0)
            opt.step()
            
            samples += x.size(0)
            total_loss += loss.item() * x.size(0)
            total_correct += (s_logits.argmax(dim=-1) == y).sum().item()
            total_agree += (s_logits.argmax(dim=-1) == t_logits.argmax(dim=-1)).sum().item()
            
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            val_acc = eval_model(osc_model, val_loader)
            print(f"  [Oscillator] Epoch {epoch+1:3d} | Loss: {total_loss/samples:.4f} | "
                  f"Train Acc: {total_correct/samples*100:.1f}% | Val Acc: {val_acc:.1f}% | Agree: {total_agree/samples*100:.1f}%")

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
if __name__ == '__main__':
    import math # ensure it's loaded for Transformer
    print(f"\n{'=' * 60}")
    print("  MAGFORMER - Speech Commands Demo")
    print(f"  Device: {DEVICE}")
    print(f"{'=' * 60}\n")

    train_loader, val_loader = get_dataloaders()

    print("\n[1/3] Training Golden Transformer...")
    transformer = TinyTransformer().to(DEVICE)
    train_transformer(transformer, train_loader, val_loader, epochs=EPOCHS_T)

    print("\n[2/3] Compiling to Oscillator Network...")
    for p in transformer.parameters(): p.requires_grad_(False)
    osc_model = OscillatorNet().to(DEVICE)
    
    print(f"  Transformer params: {sum(p.numel() for p in transformer.parameters()):,}")
    print(f"  Oscillator params:  {sum(p.numel() for p in osc_model.parameters()):,}")
    compile_oscillator(osc_model, transformer, train_loader, val_loader, epochs=EPOCHS_O)
    
    print("\n[3/3] Done. Oscillator network is ready for SPICE export.")
