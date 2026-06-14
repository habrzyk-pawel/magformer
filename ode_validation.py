# -*- coding: utf-8 -*-
"""
MAGFORMER - ODE Validation Mode
===============================
Verifies that the closed-form algebraic Kuramoto synchronization
(SSA mode) used for fast training matches the true physical
continuous-time dynamics (ODE mode) computed via torchdiffeq.
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint

N_OSCILLATORS = 8
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class OscillatorPhysics(nn.Module):
    """The continuous-time derivative function dx/dt for the oscillators."""
    def __init__(self, omega, coupling, damping):
        super().__init__()
        self.omega = omega          # (N,)
        self.K = coupling           # (N, N)
        self.damping = damping      # (N,)

    def forward(self, t, x):
        # x is the state vector of shape (B, N)
        # In a real physical system, the phases evolve according to:
        # dx_i/dt = omega_i + sum_j K_ij * sin(x_j - x_i) - damping_i * x_i
        
        B = x.size(0)
        N = x.size(1)
        
        # Calculate phase differences: (x_j - x_i)
        # Shape: (B, N, N) where [b, i, j] is x_j - x_i
        x_j = x.unsqueeze(1).expand(B, N, N)
        x_i = x.unsqueeze(2).expand(B, N, N)
        phase_diff = x_j - x_i
        
        # sum_j K_ij * sin(x_j - x_i)
        # K is (N, N). We multiply elementwise and sum over j.
        interaction = (self.K.unsqueeze(0) * torch.sin(phase_diff)).sum(dim=2) # (B, N)
        
        # dx/dt
        dxdt = self.omega.unsqueeze(0) + interaction - self.damping.unsqueeze(0) * x
        return dxdt

def run_ode_mode(u, omega, K, damping, t_end=10.0, steps=100):
    """
    Simulates the oscillators over time using an ODE solver.
    Initial state is driven by input u.
    """
    # Initial state
    x0 = u.clone()
    
    # Time span to integrate over
    t = torch.linspace(0, t_end, steps).to(u.device)
    
    # Physics function
    physics = OscillatorPhysics(omega, K, damping)
    
    # Solve ODE
    trajectory = odeint(physics, x0, t, method='rk4')
    
    # Return the final steady state
    steady_state = trajectory[-1]
    
    # Apply magnetic saturation
    return torch.tanh(steady_state)

def run_ssa_mode(u, omega, K, damping):
    """
    The fast algebraic approximation used during training.
    """
    delta_omega = (omega.unsqueeze(1) - omega.unsqueeze(0)).abs()
    sync_weights = torch.clamp(1.0 - delta_omega / (K + 1e-6), min=0.0)
    sync_weights = sync_weights * (1.0 - torch.eye(N_OSCILLATORS, device=u.device))

    coupled = torch.matmul(u, sync_weights.T)
    state = u + coupled
    state = torch.tanh(state)
    state = state * (1.0 / damping)
    return torch.tanh(state)

def validate():
    print(f"\n{'=' * 60}")
    print("  MAGFORMER - ODE vs SSA Validation")
    print(f"{'=' * 60}\n")
    
    torch.manual_seed(42)
    B = 2
    
    # Generate random physical parameters
    omega_raw = torch.randn(N_OSCILLATORS) * 0.5
    coupling_raw = torch.randn(N_OSCILLATORS, N_OSCILLATORS) * 0.5
    damping_raw = torch.randn(N_OSCILLATORS) * 0.1
    
    omega = F.softplus(omega_raw).to(DEVICE)
    K = F.softplus(coupling_raw).to(DEVICE)
    damping = (0.01 + F.softplus(damping_raw)).to(DEVICE)
    
    # Generate random input drive
    u = torch.randn(B, N_OSCILLATORS).to(DEVICE)
    
    print("Running Fast SSA Mode (Algebraic)...")
    t0 = time.time()
    out_ssa = run_ssa_mode(u, omega, K, damping)
    ssa_time = time.time() - t0
    
    print("\nRunning True ODE Mode (Numerical Integration, t=10.0s)...")
    t0 = time.time()
    out_ode = run_ode_mode(u, omega, K, damping, t_end=10.0, steps=100)
    ode_time = time.time() - t0
    
    print(f"\n--- Results ---")
    print(f"SSA Time: {ssa_time*1000:.2f} ms")
    print(f"ODE Time: {ode_time*1000:.2f} ms")
    print(f"Speedup:  {ode_time/ssa_time:.0f}x")
    
    # Compare outputs
    diff = (out_ssa - out_ode).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"\nMax difference:  {max_diff:.4f}")
    print(f"Mean difference: {mean_diff:.4f}")
    
    if mean_diff < 0.1:
        print("\n[OK] Validation passed! SSA approximation accurately matches physical ODE.")
    else:
        print("\n[WARNING] Significant divergence. The algebraic approximation may need tuning.")

if __name__ == "__main__":
    validate()
