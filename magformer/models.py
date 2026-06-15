import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConfigurableTransformer(nn.Module):
    """
    A configurable Transformer encoder model supporting either token inputs (Embedding)
    or continuous inputs (Linear projection, e.g. for MFCC audio features).
    """
    def __init__(self, vocab_size=None, input_dim=None, seq_len=8, embed_dim=16, n_heads=2, ffn_dim=32, n_layers=1, n_classes=4, dropout=0.0):
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        
        if vocab_size is not None:
            self.input_layer = nn.Embedding(vocab_size, embed_dim)
        elif input_dim is not None:
            self.input_layer = nn.Linear(input_dim, embed_dim)
        else:
            raise ValueError("Either vocab_size or input_dim must be provided to ConfigurableTransformer.")
            
        # Sinusoidal positional encoding
        pe = torch.zeros(seq_len, embed_dim)
        pos = torch.arange(seq_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, seq_len, embed_dim)
        
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=ffn_dim,
            batch_first=True, dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=n_layers)
        self.classifier = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        # x shape: (B, seq_len) for tokens or (B, seq_len, input_dim) for continuous features
        out = self.input_layer(x) + self.pe[:, :x.size(1)]
        out = self.transformer(out)
        out = out.mean(dim=1)  # Mean pooling over sequence
        return self.classifier(out)


class ConfigurableOscillatorNet(nn.Module):
    """
    A configurable Kuramoto-inspired oscillator network (Megformer) supporting
    either token inputs or continuous inputs.
    """
    def __init__(self, n_oscillators, vocab_size=None, input_dim=None, seq_len=8, embed_dim=16, ffn_dim=32, n_classes=4):
        super().__init__()
        self.n_oscillators = n_oscillators
        self.seq_len = seq_len
        
        if vocab_size is not None:
            self.embed = nn.Embedding(vocab_size, embed_dim)
            proj_in_features = seq_len * embed_dim
        elif input_dim is not None:
            self.embed = None
            proj_in_features = seq_len * input_dim
        else:
            raise ValueError("Either vocab_size or input_dim must be provided to ConfigurableOscillatorNet.")
            
        self.input_proj = nn.Sequential(
            nn.Linear(proj_in_features, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, n_oscillators)
        )
        
        # Physics parameters (learnable)
        self._omega_raw = nn.Parameter(torch.randn(n_oscillators) * 0.5)
        self._coupling_raw = nn.Parameter(torch.randn(n_oscillators, n_oscillators) * 0.1)
        self._damping_raw = nn.Parameter(torch.zeros(n_oscillators))
        
        # Readout to class logits
        self.readout = nn.Sequential(
            nn.Linear(n_oscillators, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, n_classes)
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
        return 0.01 + F.softplus(self._damping_raw)

    def forward(self, x):
        B = x.size(0)
        if self.embed is not None:
            # Token inputs of shape (B, seq_len)
            x_emb = self.embed(x)
            x_flat = x_emb.view(B, -1)
        else:
            # Continuous feature inputs of shape (B, seq_len, input_dim)
            x_flat = x.view(B, -1)
            
        u = self.input_proj(x_flat)
        
        omega = self.omega
        K = self.coupling
        # Frequency difference matrix
        delta_omega = (omega.unsqueeze(1) - omega.unsqueeze(0)).abs()
        
        # Phase-locking condition: sync when frequency gap < coupling strength
        sync_weights = torch.clamp(1.0 - delta_omega / (K + 1e-6), min=0.0)
        # Zero out self-connections
        sync_weights = sync_weights * (1.0 - torch.eye(self.n_oscillators, device=x.device))
        
        # Oscillator state update
        coupled = torch.matmul(u, sync_weights.T)
        state = u + coupled
        
        # First magnetic saturation
        state = torch.tanh(state)
        # Apply damping
        state = state * (1.0 / self.damping)
        # Second saturation pass
        state = torch.tanh(state)
        
        return self.readout(state)

    def print_physics(self):
        """Print the learned physical parameters."""
        omega = self.omega.detach().cpu()
        K = self.coupling.detach().cpu()
        damping = self.damping.detach().cpu()

        print(f"\n  Frequencies (w):  {omega.numpy().round(3)}")
        print(f"  Damping (g):      {damping.numpy().round(3)}")
        sync_mask = (K > 0.5).float()
        density = sync_mask.sum().item() / (self.n_oscillators * self.n_oscillators) * 100
        print(f"  Coupling density: {density:.1f}% ({int(sync_mask.sum().item())}/{self.n_oscillators**2} active)")
        
        # Detect frequency clusters
        sorted_omega, _ = omega.sort()
        clusters = 1
        for i in range(1, len(sorted_omega)):
            if sorted_omega[i] - sorted_omega[i-1] > 0.3:
                clusters += 1
        print(f"  Frequency clusters: {clusters}")
