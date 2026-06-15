"""
MagFormer Hyperparameter Tuning - Iteration 2
Focus on: more oscillators, longer training
"""

import csv
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from itertools import product

VOCAB_SIZE = 32
SEQ_LEN = 8
EMBED_DIM = 16
N_HEADS = 2
FFN_DIM = 32
N_CLASSES = 4
N_SAMPLES = 2048
BATCH_SIZE = 64
DEVICE = 'cpu'
DTYPE = torch.float32


class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        pe = torch.zeros(SEQ_LEN, EMBED_DIM)
        pos = torch.arange(SEQ_LEN).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, EMBED_DIM, 2).float() * (-math.log(10000.0) / EMBED_DIM))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM, nhead=N_HEADS, dim_feedforward=FFN_DIM,
            batch_first=True, dropout=0.0
        )
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.classifier = nn.Linear(EMBED_DIM, N_CLASSES)

    def forward(self, tokens):
        x = self.embed(tokens) + self.pe[:, :tokens.size(1)]
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.classifier(x)


class OscillatorNet(nn.Module):
    def __init__(self, n_oscillators):
        super().__init__()
        self.n_osc = n_oscillators
        self.embed = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.input_proj = nn.Sequential(
            nn.Linear(SEQ_LEN * EMBED_DIM, FFN_DIM),
            nn.GELU(),
            nn.Linear(FFN_DIM, n_oscillators)
        )

        self._omega_raw = nn.Parameter(torch.randn(n_oscillators) * 0.5)
        self._coupling_raw = nn.Parameter(torch.randn(n_oscillators, n_oscillators) * 0.1)
        self._damping_raw = nn.Parameter(torch.zeros(n_oscillators))

        self.readout = nn.Sequential(
            nn.Linear(n_oscillators, FFN_DIM),
            nn.GELU(),
            nn.Linear(FFN_DIM, N_CLASSES)
        )

    @property
    def omega(self):
        return F.softplus(self._omega_raw)

    @property
    def coupling(self):
        return F.softplus(self._coupling_raw)

    @property
    def damping(self):
        return 0.01 + F.softplus(self._damping_raw)

    def forward(self, tokens):
        B = tokens.size(0)
        x = self.embed(tokens)
        x_flat = x.view(B, -1)
        u = self.input_proj(x_flat)

        omega = self.omega
        K = self.coupling
        delta_omega = (omega.unsqueeze(1) - omega.unsqueeze(0)).abs()
        sync_weights = torch.clamp(1.0 - delta_omega / (K + 1e-6), min=0.0)
        sync_weights = sync_weights * (1.0 - torch.eye(self.n_osc, device=tokens.device))

        coupled = torch.matmul(u, sync_weights.T)
        state = u + coupled
        state = torch.tanh(state)
        state = state * (1.0 / self.damping)
        return torch.tanh(state)


def train(osc_model, transformer, tokens, labels, epochs=300, lr=0.01):
    optimizer = torch.optim.Adam(osc_model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    history = {'loss': [], 'acc': []}

    for epoch in range(epochs):
        osc_model.train()
        optimizer.zero_grad()

        outputs = osc_model(tokens)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        acc = (predicted == labels).float().mean().item()

        history['loss'].append(loss.item())
        history['acc'].append(acc)

        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] Loss: {loss.item():.4f} Acc: {acc:.4f}")

    return history


def compile_oscillator(osc_model, transformer, tokens, labels, epochs=300, lr=0.01):
    return train(osc_model, transformer, tokens, labels, epochs, lr)


def make_data():
    tokens = torch.randint(0, VOCAB_SIZE, (N_SAMPLES, SEQ_LEN)).to(DEVICE)
    labels = torch.randint(0, N_CLASSES, (N_SAMPLES,)).to(DEVICE)
    return tokens, labels


if __name__ == '__main__':
    print(f"\n{'=' * 60}")
    print("  MAGFORMER - Hyperparameter Tuning - Iteration 2")
    print(f"{'=' * 60}\n")

    start = time.time()

    print("Generating synthetic data...")
    tokens, labels = make_data()

    print("\nTraining Transformer reference...")
    transformer = TinyTransformer().to(DEVICE)
    for p in transformer.parameters():
        p.requires_grad_(False)

    # Hyperparameter grid
    n_osc_options = [32, 48, 96]
    epoch_options = [500, 1000, 2000]
    lr_options = [1e-3, 3e-3, 1e-2]

    results = []
    best_acc = 0
    best_params = None

    print(f"\nTesting {len(n_osc_options) * len(epoch_options) * len(lr_options)} combinations...")

    for n_osc, epochs, lr in product(n_osc_options, epoch_options, lr_options):
        print(f"\n[{len(results)+1}] n_osc={n_osc}, epochs={epochs}, lr={lr}")

        osc_model = OscillatorNet(n_oscillators=n_osc).to(DEVICE)
        history = compile_oscillator(osc_model, transformer, tokens, labels, epochs=epochs, lr=lr)

        # Final eval
        osc_model.eval()
        with torch.no_grad():
            outputs = osc_model(tokens)
            _, predicted = torch.max(outputs, 1)
            acc = (predicted == labels).float().mean().item()

        print(f"  Accuracy: {acc:.1f}%")

        if acc > best_acc:
            best_acc = acc
            best_params = {'n_osc': n_osc, 'epochs': epochs, 'lr': lr}
            print(f"  ✅ NEW BEST: {acc:.1f}%")

        results.append({
            'n_osc': n_osc,
            'epochs': epochs,
            'lr': lr,
            'accuracy': acc
        })

        # Early stop if time budget exceeded
        if time.time() - start > 180:  # 3 minutes
            print("\n⏱️  Time budget reached, stopping...")
            break

    print(f"\n{'=' * 60}")
    print("TUNING COMPLETE - Iteration 2")
    print(f"{'=' * 60}")
    print(f"Best accuracy: {best_acc:.1f}%")
    print(f"Best params: {best_params}")

    # Save results
    with open('tuning_results_v2.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['n_osc', 'epochs', 'lr', 'accuracy'])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: tuning_results_v2.csv")
