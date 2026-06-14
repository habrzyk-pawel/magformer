# Magformer: Magnetic Transformer Emulator

Magformer is a design-automation pipeline that compiles a frozen digital Transformer into a physical analog oscillator circuit. The goal is to push Edge AI inference energy down by one to two orders of magnitude by replacing expensive digital matrix multiplications with Kuramoto-style coupled LC oscillators.

This repository contains a weekend-scale proof-of-concept: a PyTorch compiler that distills a tiny Transformer into oscillator parameters, validates the result against true differential equations, and exports a SPICE netlist (`magformer_chip.cir`) ready for LTSpice or Cadence.

## Core idea

A self-attention block computes:

Attention(*Q*, *K*, *V*) = softmax(*QK*<sup>T</sup> / √*d<sub>k</sub>*) *V*

Loading *Q*, *K*, *V* and running the GEMM is expensive on a battery-powered sensor. A network of weakly coupled LC tanks naturally settles into phase relationships that behave like a kernel / similarity function. If we learn the right frequencies, damping terms, and coupling resistances, the oscillator array can approximate the Transformer output without ever executing a dense multiply.

## The 4-stage pipeline

| Stage | File | What it does |
|---|---|---|
| 1. Target | `demo.py`, `speech_demo.py` | Train a small golden Transformer on toy or audio data. |
| 2. Compiler | `speech_demo.py` | Distill the Transformer into oscillator parameters using a fast algebraic Steady-State Approximation (SSA). |
| 3. Validator | `ode_validation.py` | Integrate the true continuous-time ODEs with `torchdiffeq` and compare against SSA. |
| 4. Exporter | `spice_export.py` | Map parameters to 180 nm CMOS components and emit a `.cir` netlist. |

## Quick start

```bash
pip install torch torchaudio soundfile torchdiffeq

# Download Google Speech Commands v0.02 (~2.3 GB)
python download_data.py

# Run the end-to-end audio pipeline
python speech_demo.py

# Verify physics
python ode_validation.py

# Generate the chip netlist
python spice_export.py
```

## File guide

- `demo.py` — minimal sequence-to-sequence demo on synthetic data (no dataset download needed).
- `speech_demo.py` — keyword spotting on Google Speech Commands (`yes`, `no`, `up`, `down`); reaches ~89.6% validation accuracy and 100% behavioral-cloning agreement with the oscillator approximation.
- `ode_validation.py` — physics check: compares SSA predictions to explicit ODE integration.
- `spice_export.py` — converts abstract *ω*, *γ*, *K* into inductors, capacitors, and resistors.
- `download_data.py` — fetches the ~2.3 GB audio dataset from Google.
- `assets/images/magformer/` — diagrams and plots used in the blog post.

## Results (back-of-the-envelope)

| Platform | Estimated energy / inference |
|---|---|
| CPU / cloud | ~10 000 µJ |
| Edge MCU | ~100 µJ |
| Optimized analog ASIC (target) | **< 10 µJ** |

## Next steps

1. Open `magformer_chip.cir` in LTSpice / Cadence and verify analog behavior with real transistor models.
2. Add a Monte Carlo yield simulator (±5% component tolerance) to prove silicon robustness.

## Read more

- Blog post: https://habrzyk-pawel.github.io/2026/06/14/Magformer-compiling-transformers-into-analog-oscillator-circuits.html
- Author: Paweł Habrzyk
