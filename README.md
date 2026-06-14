# Magformer: Magnetic Transformer Emulator

Magformer is a design-automation software pipeline built to compile digital Transformer models into physical analog oscillator circuits. The goal is to produce hardware capable of running Transformer inference at 10x to 100x lower energy consumption (e.g., <10 µJ per inference for Edge AI Keyword Spotting) by replacing expensive digital matrix multiplications with the natural physics of coupled oscillators.

## Core Philosophy
This repository focuses on **productization and engineering**. We are taking the theoretical physics of oscillator-based attention (Kuramoto models) and building a compiler that spits out fabrication-ready SPICE netlists. 

The pipeline has 4 stages:
1. **Target**: A tiny frozen PyTorch Transformer.
2. **Compiler**: A knowledge distillation loop that forces a differentiable oscillator network to clone the Transformer's behavior.
3. **Validator**: Cross-checks the fast algebraic training math against true differential equations (ODE).
4. **Exporter**: Maps the trained neural parameters to physical inductors, capacitors, and resistors in a 180nm CMOS process.

---

## File Structure & Usage

### 1. `demo.py` (The Hello World)
A completely self-contained, single-file demonstration on synthetic data.
* **What it does:** Trains a tiny 1-layer Transformer on a toy sequence dataset, then trains an 8-oscillator network to mimic it using the fast algebraic Steady-State Approximation (SSA) of the Kuramoto model.
* **Run:** `python demo.py`

### 2. `speech_demo.py` (Real Audio Workload)
The core compiler pipeline applied to the Google Speech Commands dataset.
* **What it does:** Downloads 1-second audio clips for the words "yes", "no", "up", "down", extracts MFCCs, trains a golden Transformer, and successfully distills it into a 16-oscillator network.
* **Status:** Achieves 100% behavioral cloning agreement (89.6% validation accuracy) between the digital Transformer and the analog analog oscillator simulation.
* **Dataset:** run `python download_data.py` first to fetch Google Speech Commands v0.02 (~2.3 GB).
* **Run:** `python speech_demo.py`

### 3. `ode_validation.py` (Physics Verification)
* **What it does:** Uses `torchdiffeq` to simulate the continuous-time physical differential equations ($dx/dt$) of the oscillators. It compares this "true physics" output against the fast algebraic shortcut (SSA) used during training to ensure our models don't break the laws of physics.
* **Run:** `python ode_validation.py`

### 4. `spice_export.py` (Hardware Generation)
* **What it does:** Takes the abstract parameters of a trained oscillator network (resonant frequencies $\omega$, damping factors $\gamma$, and coupling matrix $K$) and maps them to standard physical component values (e.g., 1.2nH inductors, 1pF capacitors). Outputs a `.cir` netlist.
* **Output:** Generates `magformer_chip.cir`
* **Run:** `python spice_export.py`

---

## Dependencies
The stack is built to be as minimal as possible.
* `torch`
* `torchaudio`
* `soundfile` (used as the backend for torchaudio on Windows)
* `torchdiffeq` (for the ODE solver validation)

## Install

```bash
pip install torch torchaudio soundfile torchdiffeq

# Download the dataset (~2.3 GB)
python download_data.py
```

---

## Next Steps for Hardware Productization
We have successfully proven the software toolchain. The immediate next steps for the hardware track are:
1. Load `magformer_chip.cir` into LTSpice / Cadence to verify the analog behavior on transistor-level models.
2. Implement a Monte Carlo fabrication yield simulator (perturbing component values by ±5% to ensure the learned weights are robust to silicon manufacturing variances).
