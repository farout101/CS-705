# CS705 — Mixed Precision Performance Analysis

**Course:** CS705 — Accelerated Computing  
**Project Type:** Final Project  
**Team Size:** 9 members

## Overview

This project benchmarks the performance and numerical trade-offs of **FP32 vs FP16 vs BF16** floating-point precision formats across four workload categories using PyTorch and CUDA on an NVIDIA GPU.

The goal is to measure — not assume — when and by how much lower-precision formats accelerate computation, and what numerical costs they introduce.

## File Structure

```
CS-705/
├── CS705_Mixed_Precision_Performance_Analysis.ipynb   # Main notebook (your project)
├── CS705_CPU_vs_GPU_Performance_Analysis.ipynb        # Reference notebook (other team)
├── CS705_CPU_vs_GPU_Performance_Analysis.pdf          # Reference presentation (other team)
├── diagrams/                                          # Auto-generated plots (created on run)
├── data/                                              # CIFAR-10 dataset (downloaded on first run, ~170 MB)
└── README.md                                          # This file
```

## Experiments

| # | Experiment | What it measures |
|---|-----------|-----------------|
| 1 | Matrix Multiplication | Speedup + numerical error (MAE) across FP32/FP16/BF16 at 4 matrix sizes |
| 2 | Element-wise & Reductions | Bandwidth-driven speedup for multiply, sigmoid, sum, mean |
| 3 | CNN Training (CIFAR-10) | FP32 baseline vs AMP-FP16 vs AMP-BF16 training/inference speed |
| 4 | Memory & Throughput | VRAM usage and achieved TFLOPS per dtype |

## Requirements

- Python 3.10+
- PyTorch 2.x with CUDA
- NVIDIA GPU (FP16 Tensor Cores: Volta+, BF16 Tensor Cores: Ampere+)

Install dependencies:

```bash
pip install torch torchvision scipy numpy pandas matplotlib
```

## Running the Notebook

Open in Jupyter or Google Colab and run cells top to bottom. All configuration (sizes, repetitions, epoch count) is in **Section 12 — Configuration**.

The `diagrams/` folder is created automatically. CIFAR-10 (~170 MB) is downloaded automatically on first run of Experiment 3.

## Research Questions

- **RQ1:** How does FP16/BF16 speed compare to FP32 across workload types?
- **RQ2:** How does workload size affect the speedup?
- **RQ3:** Is the speedup consistent across operation types?
- **RQ4:** What numerical error does FP16/BF16 introduce vs FP32?

## Hypotheses

- **H1:** FP16/BF16 will be faster than FP32, especially for Tensor Core ops.
- **H2:** Speedup increases with workload size.
- **H3:** BF16 achieves similar speedup to FP16 with smaller numerical error.
- **H4:** AMP training is faster than FP32 with minimal impact on loss convergence.
