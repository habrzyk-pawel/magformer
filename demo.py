# -*- coding: utf-8 -*-
"""
MAGFORMER DEMO - Magnetic Transformer Emulator
================================================
Single-file demo: train a tiny Transformer, then train a Kuramoto-style
oscillator network to match its behavior via knowledge distillation.

Run:  python demo.py
Deps: torch (that's it)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time

# ---------------------------------------------------------------------
# HYPERPARAMS - all hardcoded, all tiny
# ---------------------------------------------------------------------
VOCAB_SIZE   = 32
SEQ_LEN      = 8
EMBED_DIM    = 16
N_HEADS      = 2
FFN_DIM      = 32
N_CLASSES    = 4
N_OSCILLATORS = 16
N_SAMPLES    = 2048
BATCH_SIZE   = 64
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE        = torch.float32  # float32 is fine for this demo


# ---------------------------------------------------------------------
# BLOCK 1: Tiny Transformer (the golden reference)
# ---------------------------------------------------------------------
class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        # Sinusoidal positional encoding (fixed, not learned)
        pe = torch.zeros(SEQ_LEN, EMBED_DIM)
        pos = torch.arange(SEQ_LEN).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, EMBED_DIM, 2).float() * (-math.log(10000.0) / EMBED_DIM))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, SEQ_LEN, EMBED_DIM)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM, nhead=N_HEADS, dim_feedforward=FFN_DIM,
            batch_first=True, dropout=0.0
        )
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.classifier = nn.Linear(EMBED_DIM, N_CLASSES)

    def forward(self, tokens):
        x = self.embed(tokens) + self.pe[:, :tokens.size(1)]
        x = self.transformer(x)
        x = x.mean(dim=1)  # mean pool over sequence
        return self.classifier(x)


# ---------------------------------------------------------------------
# BLOCK 2: Oscillator Network (the physics model)
# ---------------------------------------------------------------------
class OscillatorNet(nn.Module):
    """
    Kuramoto-inspired oscillator network.
    
    Physics analogy:
      - Each of N_OSCILLATORS oscillators has a natural frequency omega_i
      - Coupling matrix K_ij determines how strongly oscillator j influences i
      - Two oscillators synchronize (= attend to each other) when their
        frequency difference |omega_i - omega_j| < K_ij (phase-locking condition)
      - tanh nonlinearity = magnetic core saturation
    
    NOT using an ODE solver - using the analytical Kuramoto steady-state.
    This is what a real chip would compute.
    """
    def __init__(self):
        super().__init__()
        # Input projection: tokens -> oscillator drive signals
        self.embed = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.input_proj = nn.Sequential(
            nn.Linear(EMBED_DIM, FFN_DIM),
            nn.GELU(),
            nn.Linear(FFN_DIM, N_OSCILLATORS)
        )

        # Physics parameters (learnable = tunable on chip)
        self._omega_raw = nn.Parameter(torch.randn(N_OSCILLATORS) * 0.5)   # natural frequencies
        self._coupling_raw = nn.Parameter(torch.randn(N_OSCILLATORS, N_OSCILLATORS) * 0.1)  # coupling
        self._damping_raw = nn.Parameter(torch.zeros(N_OSCILLATORS))        # damping coefficients

        # Output readout: oscillator states -> class logits
        self.readout = nn.Sequential(
            nn.Linear(N_OSCILLATORS, FFN_DIM),
            nn.GELU(),
            nn.Linear(FFN_DIM, N_CLASSES)
        )

    @property
    def omega(self):
        """Natural frequencies - must be positive (physical constraint)."""
        return F.softplus(self._omega_raw)

    @property
    def coupling(self):
        """Coupling strengths - must be positive (conductance)."""
        return F.softplus(self._coupling_raw)

    @property
    def damping(self):
        """Damping - must be positive (resistance)."""
        return 0.01 + F.softplus(self._damping_raw)  # minimum damping to prevent blowup

    def forward(self, tokens):
        B = tokens.size(0)

        # 1. Embed tokens and project to oscillator space
        x = self.embed(tokens)                          # (B, SEQ_LEN, EMBED_DIM)
        drive = self.input_proj(x)                      # (B, SEQ_LEN, N_OSC)
        # Aggregate drive signal: each oscillator gets sum of driven inputs
        u = drive.mean(dim=1)                           # (B, N_OSC)

        # 2. Compute Kuramoto synchronization weights
        omega = self.omega                              # (N_OSC,)
        K = self.coupling                               # (N_OSC, N_OSC)
        # Frequency difference matrix
        delta_omega = (omega.unsqueeze(1) - omega.unsqueeze(0)).abs()  # (N_OSC, N_OSC)
        # Phase-locking condition: sync when frequency gap < coupling strength
        # sync_weight in [0, 1]: 1 = perfectly locked, 0 = no interaction
        sync_weights = torch.clamp(1.0 - delta_omega / (K + 1e-6), min=0.0)  # (N_OSC, N_OSC)
        # Zero out self-connections (oscillator doesn't couple with itself)
        sync_weights = sync_weights * (1.0 - torch.eye(N_OSCILLATORS, device=tokens.device))

        # 3. Oscillator state update via synchronized coupling
        # Each oscillator's state = its drive + weighted sum of other oscillators' drives
        coupled = torch.matmul(u, sync_weights.T)       # (B, N_OSC)
        state = u + coupled                              # (B, N_OSC)

        # 4. Magnetic saturation nonlinearity (tanh = physical core saturation)
        state = torch.tanh(state)

        # 5. Apply damping (energy dissipation)
        state = state * (1.0 / self.damping)

        # 6. Second saturation pass (deeper nonlinear compression)
        state = torch.tanh(state)

        # 7. Readout -> classification
        logits = self.readout(state)                    # (B, N_CLASSES)
        return logits

    def print_physics(self):
        """Print the learned physical parameters."""
        omega = self.omega.detach().cpu()
        K = self.coupling.detach().cpu()
        damping = self.damping.detach().cpu()

        print(f"\n  Frequencies (w):  {omega.numpy().round(3)}")
        print(f"  Damping (g):      {damping.numpy().round(3)}")
        # Count near-zero coupling entries (sparsity)
        sync_mask = (K > 0.5).float()
        density = sync_mask.sum().item() / (N_OSCILLATORS * N_OSCILLATORS) * 100
        print(f"  Coupling density: {density:.1f}% ({int(sync_mask.sum().item())}/{N_OSCILLATORS**2} active)")
        # Detect frequency clusters
        sorted_omega, _ = omega.sort()
        clusters = 1
        for i in range(1, len(sorted_omega)):
            if sorted_omega[i] - sorted_omega[i-1] > 0.3:
                clusters += 1
        print(f"  Frequency clusters: {clusters} (tokens grouping by resonance)")


# ---------------------------------------------------------------------
# BLOCK 3: Synthetic Data
# ---------------------------------------------------------------------
def make_data():
    """
    Synthetic classification task: label depends on token statistics.
    Simple enough that a tiny Transformer can learn it perfectly.
    """
    torch.manual_seed(42)
    tokens = torch.randint(0, VOCAB_SIZE, (N_SAMPLES, SEQ_LEN))
    # Label = first token mod N_CLASSES
    # Simple enough for a tiny Transformer to learn perfectly
    labels = tokens[:, 0] % N_CLASSES
    return tokens, labels


# ---------------------------------------------------------------------
# BLOCK 4: Training
# ---------------------------------------------------------------------
def train_transformer(model, tokens, labels, epochs=100):
    """Pretrain the Transformer on the synthetic task."""
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        perm = torch.randperm(len(tokens), device=DEVICE)
        total_loss = 0.0
        total_correct = 0

        for i in range(0, len(tokens), BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            x, y = tokens[idx], labels[idx]
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
            total_correct += (logits.argmax(dim=-1) == y).sum().item()

        if (epoch + 1) % 20 == 0:
            acc = total_correct / len(tokens) * 100
            print(f"  [Transformer] Epoch {epoch+1:3d} | Loss: {total_loss/len(tokens):.4f} | Acc: {acc:.1f}%")

    model.eval()
    with torch.no_grad():
        logits = model(tokens)
        acc = (logits.argmax(dim=-1) == labels).sum().item() / len(tokens) * 100
    return acc


def compile_oscillator(osc_model, transformer, tokens, labels, epochs=300):
    """
    Knowledge distillation: train oscillator network to match
    the frozen Transformer's output distribution.
    """
    osc_model.train()
    transformer.eval()
    opt = torch.optim.Adam(osc_model.parameters(), lr=3e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    temperature = 4.0  # soft targets

    # Get all Transformer logits (frozen, compute once)
    with torch.no_grad():
        teacher_logits = transformer(tokens)  # (N_SAMPLES, N_CLASSES)

    best_loss = float('inf')
    history = []

    for epoch in range(epochs):
        perm = torch.randperm(len(tokens), device=DEVICE)
        total_loss = 0.0
        total_correct = 0
        total_agree = 0

        for i in range(0, len(tokens), BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            x, y = tokens[idx], labels[idx]
            t_logits = teacher_logits[idx]

            # Forward through oscillator
            s_logits = osc_model(x)

            # Knowledge distillation loss (soft targets)
            loss_kd = F.kl_div(
                F.log_softmax(s_logits / temperature, dim=-1),
                F.softmax(t_logits / temperature, dim=-1),
                reduction='batchmean'
            ) * (temperature ** 2)

            # Hard label loss (actual accuracy)
            loss_ce = F.cross_entropy(s_logits, y)

            # Combined
            loss = 0.7 * loss_kd + 0.3 * loss_ce

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(osc_model.parameters(), 1.0)
            opt.step()

            total_loss += loss.item() * len(idx)
            total_correct += (s_logits.argmax(dim=-1) == y).sum().item()
            total_agree += (s_logits.argmax(dim=-1) == t_logits.argmax(dim=-1)).sum().item()

        scheduler.step()
        avg_loss = total_loss / len(tokens)
        acc = total_correct / len(tokens) * 100
        agree = total_agree / len(tokens) * 100
        history.append((avg_loss, acc, agree))

        if avg_loss < best_loss:
            best_loss = avg_loss

        if (epoch + 1) % 30 == 0:
            print(f"  [Oscillator]  Epoch {epoch+1:3d} | Loss: {avg_loss:.4f} | "
                  f"Acc: {acc:.1f}% | Agrees w/ Transformer: {agree:.1f}%")

    return history


# ---------------------------------------------------------------------
# BLOCK 5: Results
# ---------------------------------------------------------------------
def print_results(transformer, osc_model, tokens, labels, history):
    """Print final comparison."""
    transformer.eval()
    osc_model.eval()

    with torch.no_grad():
        t_logits = transformer(tokens)
        s_logits = osc_model(tokens)

    t_acc = (t_logits.argmax(dim=-1) == labels).sum().item() / len(tokens) * 100
    s_acc = (s_logits.argmax(dim=-1) == labels).sum().item() / len(tokens) * 100
    agree = (t_logits.argmax(dim=-1) == s_logits.argmax(dim=-1)).sum().item() / len(tokens) * 100
    gap = t_acc - s_acc

    print("\n" + "=" * 60)
    print("       MAGFORMER DEMO RESULTS")
    print("=" * 60)
    print(f"  Transformer accuracy:    {t_acc:.1f}%")
    print(f"  Oscillator accuracy:     {s_acc:.1f}%")
    print(f"  Accuracy gap:            {gap:+.1f}%")
    print(f"  Agreement (same output): {agree:.1f}%")
    print(f"  Final distillation loss: {history[-1][0]:.4f}")
    print(f"  Best distillation loss:  {min(h[0] for h in history):.4f}")

    # Physics analysis
    print("\n  --- Physics Parameters ---")
    osc_model.print_physics()

    # Component mapping (what this would be on a chip)
    omega = osc_model.omega.detach().cpu()
    damping = osc_model.damping.detach().cpu()
    print("\n  --- Hardware Mapping ---")
    # omega = 1/sqrt(LC), so for C=1pF: L = 1/(omega^2 * C)
    C_pf = 1.0  # assume 1 pF capacitor
    L_values = 1.0 / (omega.numpy() ** 2 * C_pf + 1e-9)
    print(f"  Inductors (nH, at C=1pF): {L_values.round(2)}")
    R_values = damping.numpy()
    print(f"  Damping resistors (Ohm):  {R_values.round(3)}")
    n_coupling = (osc_model.coupling.detach().cpu() > 0.5).sum().item()
    print(f"  Coupling resistors:       {n_coupling} (MOSFET switches)")
    total_components = N_OSCILLATORS * 3 + n_coupling  # L + C + R per osc + coupling
    print(f"  Total component count:    {total_components}")
    print("=" * 60)

    # Convergence summary
    losses = [h[0] for h in history]
    if len(losses) > 1 and losses[-1] < losses[0] * 0.5:
        print("\n  [OK] Loss decreased significantly -- optimization is working")
    else:
        print("\n  [FAIL] Loss did not decrease enough -- check hyperparameters")

    if gap < 5.0:
        print("  [OK] Accuracy gap < 5% -- oscillator successfully clones Transformer")
    elif gap < 10.0:
        print("  [~] Accuracy gap < 10% -- partial success, may need more oscillators")
    else:
        print("  [FAIL] Accuracy gap > 10% -- oscillator cannot match Transformer")

    if agree > 90:
        print("  [OK] >90% agreement -- strong behavioral cloning")
    elif agree > 80:
        print("  [~] >80% agreement -- decent behavioral cloning")
    else:
        print("  [FAIL] <80% agreement -- weak behavioral cloning")

    print()


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
if __name__ == '__main__':
    print(f"\n{'=' * 60}")
    print("  MAGFORMER - Magnetic Transformer Emulator Demo")
    print(f"  Device: {DEVICE}")
    print(f"  Oscillators: {N_OSCILLATORS} | Embed: {EMBED_DIM} | Classes: {N_CLASSES}")
    print(f"{'=' * 60}\n")

    start = time.time()

    # Data
    print("[1/4] Generating synthetic data...")
    tokens, labels = make_data()
    tokens, labels = tokens.to(DEVICE), labels.to(DEVICE)
    print(f"  {N_SAMPLES} samples, seq_len={SEQ_LEN}, vocab={VOCAB_SIZE}, classes={N_CLASSES}")

    # Transformer
    print("\n[2/4] Training Transformer (golden reference)...")
    transformer = TinyTransformer().to(DEVICE)
    t_acc = train_transformer(transformer, tokens, labels, epochs=100)
    print(f"  -> Transformer trained. Final accuracy: {t_acc:.1f}%")
    for p in transformer.parameters():
        p.requires_grad_(False)  # freeze

    # Oscillator
    print("\n[3/4] Compiling Transformer -> Oscillator Network...")
    osc_model = OscillatorNet().to(DEVICE)
    n_params_t = sum(p.numel() for p in transformer.parameters())
    n_params_o = sum(p.numel() for p in osc_model.parameters())
    print(f"  Transformer params: {n_params_t:,}")
    print(f"  Oscillator params:  {n_params_o:,}")
    history = compile_oscillator(osc_model, transformer, tokens, labels, epochs=300)

    # Results
    print("\n[4/4] Final evaluation...")
    print_results(transformer, osc_model, tokens, labels, history)

    elapsed = time.time() - start
    print(f"  Total time: {elapsed:.1f}s")
