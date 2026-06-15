import os
import sys
import json
import time
import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add workspace root to python path to import magformer package
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from magformer.models import ConfigurableTransformer, ConfigurableOscillatorNet

# Configuration presets
PRESETS = {
    "tiny": {
        "vocab_size": 32,
        "seq_len": 8,
        "embed_dim": 16,
        "n_heads": 2,
        "ffn_dim": 32,
        "n_layers": 1,
        "n_classes": 4,
        "n_oscillators": 16,
        "epochs_t": 100,
        "epochs_o": 150,
        "batch_size": 64,
        "lr_t": 1e-3,
        "lr_o": 3e-3,
        "n_samples": 2048,
    },
    "medium": {
        "vocab_size": 64,
        "seq_len": 16,
        "embed_dim": 32,
        "n_heads": 4,
        "ffn_dim": 64,
        "n_layers": 2,
        "n_classes": 8,
        "n_oscillators": 32,
        "epochs_t": 100,
        "epochs_o": 200,
        "batch_size": 64,
        "lr_t": 1e-3,
        "lr_o": 3e-3,
        "n_samples": 2048,
    },
    "large": {
        "vocab_size": 128,
        "seq_len": 32,
        "embed_dim": 64,
        "n_heads": 8,
        "ffn_dim": 128,
        "n_layers": 4,
        "n_classes": 10,
        "n_oscillators": 64,
        "epochs_t": 120,
        "epochs_o": 250,
        "batch_size": 64,
        "lr_t": 1e-3,
        "lr_o": 3e-3,
        "n_samples": 4096,
    }
}

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def generate_synthetic_data(vocab_size, seq_len, n_classes, n_samples, seed=42):
    set_seed(seed)
    tokens = torch.randint(0, vocab_size, (n_samples, seq_len))
    # Label is based on first token mod n_classes (consistent with demo.py)
    labels = tokens[:, 0] % n_classes
    return tokens, labels

def train_transformer(model, tokens, labels, config, device):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr_t"])
    
    epochs = config["epochs_t"]
    batch_size = config["batch_size"]
    n_samples = len(tokens)
    
    for epoch in range(epochs):
        perm = torch.randperm(n_samples, device=device)
        total_loss = 0.0
        total_correct = 0
        
        for i in range(0, n_samples, batch_size):
            idx = perm[i:i+batch_size]
            x, y = tokens[idx], labels[idx]
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(idx)
            total_correct += (logits.argmax(dim=-1) == y).sum().item()
            
    model.eval()
    with torch.no_grad():
        logits = model(tokens)
        acc = (logits.argmax(dim=-1) == labels).sum().item() / n_samples * 100
    return acc

def train_oscillator(osc_model, transformer, tokens, labels, config, device):
    osc_model.train()
    transformer.eval()
    optimizer = torch.optim.Adam(osc_model.parameters(), lr=config["lr_o"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs_o"])
    temperature = 4.0
    
    epochs = config["epochs_o"]
    batch_size = config["batch_size"]
    n_samples = len(tokens)
    
    with torch.no_grad():
        teacher_logits = transformer(tokens)
        
    history = []
    for epoch in range(epochs):
        perm = torch.randperm(n_samples, device=device)
        total_loss = 0.0
        total_correct = 0
        total_agree = 0
        
        for i in range(0, n_samples, batch_size):
            idx = perm[i:i+batch_size]
            x, y = tokens[idx], labels[idx]
            t_logits = teacher_logits[idx]
            
            s_logits = osc_model(x)
            
            loss_kd = F.kl_div(
                F.log_softmax(s_logits / temperature, dim=-1),
                F.softmax(t_logits / temperature, dim=-1),
                reduction='batchmean'
            ) * (temperature ** 2)
            
            loss_ce = F.cross_entropy(s_logits, y)
            loss = 0.7 * loss_kd + 0.3 * loss_ce
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(osc_model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item() * len(idx)
            total_correct += (s_logits.argmax(dim=-1) == y).sum().item()
            total_agree += (s_logits.argmax(dim=-1) == t_logits.argmax(dim=-1)).sum().item()
            
        scheduler.step()
        avg_loss = total_loss / n_samples
        acc = total_correct / n_samples * 100
        agree = total_agree / n_samples * 100
        history.append((avg_loss, acc, agree))
        
    osc_model.eval()
    with torch.no_grad():
        final_logits = osc_model(tokens)
        final_t_logits = teacher_logits
        s_acc = (final_logits.argmax(dim=-1) == labels).sum().item() / n_samples * 100
        agree = (final_logits.argmax(dim=-1) == final_t_logits.argmax(dim=-1)).sum().item() / n_samples * 100
        
    return s_acc, agree, history[-1][0]

def main():
    parser = argparse.ArgumentParser(description="Verify Transformer vs Megformer performance in CI")
    parser.add_argument(
        "--config", type=str, default="tiny", choices=["tiny", "medium", "large"],
        help="Model preset configuration"
    )
    parser.add_argument(
        "--baseline-file", type=str, default="ci_baseline.json",
        help="Path to baseline performance file"
    )
    parser.add_argument(
        "--update-baseline", action="store_true",
        help="Update the baseline file with the current run results"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for data and training"
    )
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = PRESETS[args.config]
    
    print(f"============================================================")
    print(f"  VERIFYING PERFORMANCE: {args.config.upper()} CONFIG")
    print(f"  Device: {device} | Seed: {args.seed}")
    print(f"  Oscillators: {config['n_oscillators']} | Embed: {config['embed_dim']} | Heads: {config['n_heads']}")
    print(f"============================================================")
    
    print("[1/3] Generating synthetic data...")
    tokens, labels = generate_synthetic_data(
        vocab_size=config["vocab_size"],
        seq_len=config["seq_len"],
        n_classes=config["n_classes"],
        n_samples=config["n_samples"],
        seed=args.seed
    )
    tokens, labels = tokens.to(device), labels.to(device)
    
    print("[2/3] Training Transformer ideal model...")
    transformer = ConfigurableTransformer(
        vocab_size=config["vocab_size"],
        seq_len=config["seq_len"],
        embed_dim=config["embed_dim"],
        n_heads=config["n_heads"],
        ffn_dim=config["ffn_dim"],
        n_layers=config["n_layers"],
        n_classes=config["n_classes"]
    ).to(device)
    
    t_acc = train_transformer(transformer, tokens, labels, config, device)
    print(f"  -> Transformer trained. Final accuracy: {t_acc:.2f}%")
    for p in transformer.parameters():
        p.requires_grad_(False)
        
    print("[3/3] Compiling Megformer (Oscillator Network)...")
    osc_model = ConfigurableOscillatorNet(
        n_oscillators=config["n_oscillators"],
        vocab_size=config["vocab_size"],
        embed_dim=config["embed_dim"],
        seq_len=config["seq_len"],
        ffn_dim=config["ffn_dim"],
        n_classes=config["n_classes"]
    ).to(device)
    
    s_acc, agreement, dist_loss = train_oscillator(osc_model, transformer, tokens, labels, config, device)
    gap = t_acc - s_acc
    
    print("\n---------------------- Run Metrics ----------------------")
    print(f"  Transformer accuracy:    {t_acc:.2f}%")
    print(f"  Oscillator accuracy:     {s_acc:.2f}%")
    print(f"  Accuracy gap:            {gap:+.2f}% (Goal: closer to 0%)")
    print(f"  Agreement:               {agreement:.2f}%")
    print(f"  Distillation loss:       {dist_loss:.4f}")
    print("---------------------------------------------------------")
    
    metrics = {
        "transformer_accuracy": round(t_acc, 2),
        "oscillator_accuracy": round(s_acc, 2),
        "accuracy_gap": round(gap, 2),
        "agreement": round(agreement, 2),
        "distillation_loss": round(dist_loss, 4)
    }
    
    # Save parameters if we ran the tiny demo
    if args.config == "tiny":
        torch.save({
            'omega': osc_model.omega.detach().cpu(),
            'coupling': osc_model.coupling.detach().cpu(),
            'damping': osc_model.damping.detach().cpu()
        }, 'trained_parameters.pt')
        print("  Saved trained parameters to trained_parameters.pt")
        
    if args.update_baseline:
        baselines = {}
        if os.path.exists(args.baseline_file):
            try:
                with open(args.baseline_file, 'r') as f:
                    baselines = json.load(f)
            except Exception as e:
                print(f"  Warning: could not load existing baseline file: {e}")
                
        baselines[args.config] = metrics
        with open(args.baseline_file, 'w') as f:
            json.dump(baselines, f, indent=2)
        print(f"  Updated baseline file: {args.baseline_file}")
        sys.exit(0)
        
    # Verify against baseline
    if not os.path.exists(args.baseline_file):
        print(f"[FAIL] Baseline file {args.baseline_file} not found. Run with --update-baseline to establish baseline.")
        sys.exit(1)
        
    with open(args.baseline_file, 'r') as f:
        baselines = json.load(f)
        
    if args.config not in baselines:
        print(f"[FAIL] Baseline for config {args.config} not found in {args.baseline_file}. Run with --update-baseline.")
        sys.exit(1)
        
    baseline = baselines[args.config]
    print(f"\nComparing to baseline from {args.baseline_file}:")
    print(f"  Accuracy Gap:      Run = {gap:+.2f}%, Baseline = {baseline['accuracy_gap']:+.2f}%")
    print(f"  Distillation Loss: Run = {dist_loss:.4f}, Baseline = {baseline['distillation_loss']:.4f}")
    print(f"  Agreement:         Run = {agreement:.2f}%, Baseline = {baseline['agreement']:.2f}%")
    
    # Check for performance degradation:
    # 1. We don't want the accuracy gap to increase by more than 2% compared to baseline.
    # 2. We don't want the distillation loss to increase by more than 25% or absolute 0.05.
    failed = False
    
    # Check gap
    if gap > baseline["accuracy_gap"] + 2.0:
        print(f"[FAIL] DEGRADATION: Accuracy gap increased by more than 2.0% (Allowed: {baseline['accuracy_gap'] + 2.0:.2f}%)")
        failed = True
        
    # Check distillation loss
    if dist_loss > max(baseline["distillation_loss"] * 1.25, baseline["distillation_loss"] + 0.05):
        print(f"[FAIL] DEGRADATION: Distillation loss increased significantly.")
        failed = True
        
    # Check agreement
    if agreement < baseline["agreement"] - 3.0:
        print(f"[FAIL] DEGRADATION: Agreement decreased by more than 3.0% (Allowed: {baseline['agreement'] - 3.0:.2f}%)")
        failed = True
        
    if failed:
        print("\n[FAIL] CI Check FAILED: Megformer performance degraded relative to baseline.")
        sys.exit(1)
    else:
        print("\n[SUCCESS] CI Check PASSED: Megformer is matching or exceeding the baseline performance.")
        sys.exit(0)

if __name__ == "__main__":
    main()
