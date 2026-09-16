# CS705 — CPU vs GPU Performance Analysis: Full Project Explanation

This document explains everything in the notebook `CS705_CPU_vs_GPU_Performance_Analysis_Team.ipynb` from start to finish — what every section does, why it exists, and how the code works.

---

## Overview

The notebook answers one core question:

> **How much faster is a GPU than a CPU for different types of computational workloads, and does it depend on problem size?**

To answer this, we run the **same operations on both CPU and GPU**, measure the time each takes, compute the speedup, and analyse the results statistically. We do this across **5 different experiment types** to cover a broad range of workloads.

---

## Part 1: The Setup Sections (Cells 0–15)

### Cell 0 — Title and Team Table

Just the notebook title and a table listing all 9 team members with their assigned sections. Each member owns a part of the report.

---

### Cell 1 — Introduction

Explains the motivation. CPUs and GPUs are built differently:

- A **CPU** has a few very powerful cores (4–64) optimised for doing one thing very fast — good for complex, sequential logic.
- A **GPU** has thousands of simpler cores optimised for doing many things at the same time — good for repetitive, parallel math.

Because of this, GPUs can be much faster for certain workloads (like matrix multiplication) but CPUs remain competitive for others (like small operations or sequential code). The notebook measures this difference empirically — with real timings, not theory.

---

### Cell 2 — Background Theory (7 subsections)

This is the academic foundation of the project. It covers:

**2.1 CPU Architecture**

- CPUs use out-of-order execution, branch prediction, and SIMD (AVX-512) to extract speed from complex sequential code.
- Limited to ~4–64 cores, but each core is very powerful individually.

**2.2 GPU Architecture**

- GPUs have thousands of CUDA cores organised into Streaming Multiprocessors (SMs).
- Threads execute in groups of 32 called **warps** in lockstep (SIMT model).
- RTX 3050 Laptop GPU: ~2048 CUDA cores.

**2.3 CUDA Programming Model**

- CPU submits work to the GPU asynchronously. `torch.cuda.synchronize()` is needed to wait for GPU to finish before taking a time reading.

**2.4 Memory Hierarchy**
A comparison table showing that GPU VRAM bandwidth (~200–400 GB/s) is 4–8× higher than CPU DRAM bandwidth (~50–80 GB/s). This matters a lot for memory-bound workloads.

**2.5 Amdahl's Law**
The theoretical limit on parallel speedup. Only operations that are fully parallelisable get the full GPU benefit. The formula is:

```
Speedup = 1 / ((1 - p) + p/N)
```

Where `p` is the fraction of code that can be parallelised and `N` is the number of parallel processors.

**2.6 Workload Categories**
A table predicting expected GPU advantage per workload type before we run the experiments.

**2.7 Benchmarking Considerations**
Explains why you need warm-up runs, synchronisation, and multiple repetitions to get accurate GPU timings.

---

### Cell 3 — Problem Statement

States the gap: most people *assume* GPUs are always faster. This project *measures* whether that's true across different workload types and sizes on a real consumer GPU (RTX 3050 Laptop).

---

### Cell 4 — Objectives

6 concrete goals:

1. Measure execution time for CPU and GPU across 5 workload types
2. Quantify GPU speedup as a function of workload and problem size
3. Find the crossover point (minimum size where GPU becomes faster)
4. Measure consistency of results (coefficient of variation)
5. Test statistical significance (paired t-test)
6. Provide a cross-experiment comparison chart

---

### Cell 5 — Research Questions (RQ1–RQ4)

Four questions the experiments are designed to answer:

- **RQ1:** How much faster is the GPU for each workload category?
- **RQ2:** Does speedup grow with problem size?
- **RQ3:** Is GPU always faster, or does CPU win sometimes?
- **RQ4:** Are the differences statistically significant and consistent?

---

### Cell 6 — Hypotheses (H1–H4)

Four predictions made *before* running experiments:

- **H1:** GPU will be significantly faster for compute-intensive workloads (matmul, convolution), speedup > 10×.
- **H2:** Speedup grows with problem size as the GPU's cores get better utilised.
- **H3:** For small problem sizes, CPU will be competitive due to GPU kernel launch overhead.
- **H4:** Sorting will show the least speedup; element-wise ops will show moderate speedup.

These are evaluated at the end of the notebook with SUPPORTED / NOT SUPPORTED verdicts based on actual measurements.

---

### Cell 7 — Methodology

Explains how every experiment is run:

1. **Pre-allocate tensors** on each device separately before timing starts — this ensures data transfer overhead is NOT included in the timing.
2. **Warm-up phase:** Run the operation 5 times (untimed) to force CUDA JIT compilation and warm up caches.
3. **Timed phase:** Run 20 times, measuring each with `time.perf_counter()`.
4. **Synchronise:** Call `torch.cuda.synchronize()` before and after each GPU timing so we wait for the GPU to actually finish.
5. **Report:** Mean, std, min, max of the 20 repetitions.

**Speedup formula:**

```
Speedup = CPU_time / GPU_time
```

If speedup > 1 → GPU is faster. If speedup < 1 → CPU is faster.

---

### Cells 8–10 — Environment Detection

Code that prints:

- Python version, PyTorch version
- Whether CUDA is available
- GPU name, VRAM size, compute capability, number of multiprocessors
- CPU model, number of logical cores

This establishes the hardware baseline so results can be contextualised.

---

### Cells 11–13 — Core Utilities

Defines three functions used by every experiment:

```python
def sync():
    # Waits for GPU to finish all pending work
    if GPU_AVAILABLE:
        torch.cuda.synchronize()

def benchmark_fn(fn, warmup=5, reps=20):
    # Times a callable fn() with warm-up and repetitions
    # Returns: {mean, std, min, max, raw}

def speedup(cpu_time, gpu_time):
    # Returns cpu_time / gpu_time
```

These are the building blocks. Every experiment calls `benchmark_fn` with a lambda wrapping the operation to time.

---

### Cells 14–15 — Configuration

All tunable parameters in one place:

```python
MATRIX_SIZES     = [256, 512, 1024, 2048, 4096]   # Experiment 1
TENSOR_SIZES     = [256K, 1M, 4M, 16M elements]   # Experiments 2 & 3
CONV_BATCH_SIZES = [8, 16, 32, 64, 128]            # Experiment 4
GEMV_BATCH_SIZES = [64, 256, 512, 1024]            # Experiment 5
GEMV_DIMS        = [128, 256, 512]                 # Experiment 5
```

Also creates the `./diagrams/` folder automatically so plot saving doesn't fail.

---

## Part 2: The Experiments (Cells 16–35)

Each experiment follows the same structure:

1. **Markdown cell** — background, what we measure, expected outcome
2. **Code cell** — runs the benchmark loop, builds a DataFrame, displays a table
3. **Code cell** — generates and saves plots to `./diagrams/`
4. **Markdown cell** — interprets the results

---

### Experiment 1 — Matrix Multiplication (Cells 16–19)

**What it does:**
Runs `torch.matmul(A, A)` for square matrices of size N×N, where N ∈ {256, 512, 1024, 2048, 4096}.

**Why matrix multiplication?**
It's the foundational operation of deep learning — every Linear layer, every attention mechanism, every weight update is ultimately matrix multiplication. For an N×N matmul, 2N³ floating-point operations are needed. This makes it **compute-bound** for large N, which is exactly where GPUs shine.

**PyTorch backend:**

- CPU → Intel MKL / OpenBLAS (highly tuned BLAS library)
- GPU → cuBLAS + Tensor Cores (RTX 3050 has Ampere Tensor Cores)

**Code logic:**

```python
for size in MATRIX_SIZES:
    A_cpu = torch.randn(size, size, device=CPU_DEVICE)
    stats_cpu = benchmark_fn(lambda A=A_cpu: torch.matmul(A, A))

    A_gpu = A_cpu.to(GPU_DEVICE)
    stats_gpu = benchmark_fn(lambda A=A_gpu: torch.matmul(A, A))

    speedup = stats_cpu['mean'] / stats_gpu['mean']
```

**Plots generated:**

- Left: execution time (ms) vs matrix size N on log-log scale — one line for CPU (blue), one for GPU (red)
- Right: GPU speedup vs N, with speedup values annotated on each point

**Expected result:** GPU speedup grows from near-parity at 256×256 to very high at 4096×4096 (potentially 20–50×).

---

### Experiment 2 — Element-wise and Reduction Operations (Cells 20–23)

**What it does:**Benchmarks 4 operations on 1D float32 tensors of 4 sizes (256K to 16M elements):

- `x * x` — element-wise multiply
- `torch.sigmoid(x)` — element-wise sigmoid
- `x.sum()` — sum reduction
- `x.mean()` — mean reduction

**Why these operations?**
These are **memory-bandwidth-bound** — the compute per element is tiny, so the bottleneck is how fast data can be loaded from memory. GPU VRAM bandwidth (~200–400 GB/s) is 4–8× higher than CPU DRAM (~50–80 GB/s), so GPU should be moderately faster — but not as dramatically as matmul.

**Code logic:**

```python
ops_config = {
    'Elementwise Multiply': lambda x: x * x,
    'Sigmoid':              lambda x: torch.sigmoid(x),
    'Sum Reduction':        lambda x: x.sum(),
    'Mean Reduction':       lambda x: x.mean(),
}

for n_elem in TENSOR_SIZES:
    for op_name, op_fn in ops_config.items():
        x_cpu = torch.randn(n_elem, device=CPU_DEVICE)
        x_gpu = x_cpu.to(GPU_DEVICE)
        # benchmark both...
```

**Plots generated:**

- 2×2 grid: execution time vs tensor size for each operation (CPU blue, GPU red)
- Separate speedup chart: one line per operation showing how speedup changes with size

---

### Experiment 3 — Sorting (Cells 24–27)

**What it does:**
Benchmarks `torch.sort(x)` on 1D float32 tensors of sizes 256K to 16M elements.

**Why sorting?**
Sorting is fundamentally different from matmul or element-wise ops. It involves **data-dependent branching and irregular memory access** — the output position of each element depends on its value. This is harder to parallelise efficiently on SIMT GPUs.

**PyTorch backend:**

- CPU → introsort/merge sort (std library sort)
- GPU → CUB's radix sort (avoids comparison branching by processing fixed-width bit groups in passes)

**Expected result:** Lower GPU speedup than matmul. For small arrays, CPU may win due to cache advantage. For very large arrays, GPU radix sort may pull ahead.

**Plots generated:**

- Left: execution time vs array size (log-log)
- Right: GPU speedup vs array size with annotated values

---

### Experiment 4 — CNN Convolution (Cells 28–31)

**What it does:**
Runs a forward pass (inference only, no backward/gradient) through a 3-layer Conv2d network on batches of 32×32 synthetic images:

```
Conv2d(3→32, 3×3) + ReLU
Conv2d(32→64, 3×3) + ReLU
Conv2d(64→128, 3×3) + ReLU
```

Tests batch sizes: 8, 16, 32, 64, 128.

**Why CNN convolution?**
Convolution is the defining operation of image-processing neural networks. Internally, PyTorch converts convolution to matrix multiplication (via im2col) or uses Winograd/FFT algorithms. cuDNN auto-selects the fastest algorithm, making GPU convolution extremely optimised.

**Important:** The same model weights are loaded onto both CPU and GPU (using `load_state_dict`) so results are comparable.

**Expected result:** High GPU speedup, increasing with batch size as the GPU gets better utilised.

**Plots generated:**

- Left: execution time vs batch size
- Right: bar chart of GPU speedup per batch size with values annotated

---

### Experiment 5 — Batched GEMV (Cells 32–35)

**What it does:**Benchmarks `torch.bmm(A, x)` where:

- A has shape `(batch, dim, dim)` — a batch of square matrices
- x has shape `(batch, dim, 1)` — a batch of column vectors

This computes `batch` independent matrix-vector products simultaneously.

**Why batched GEMV?**
This simulates what happens inside a fully-connected (Linear) layer during neural network inference. GEMV has **lower arithmetic intensity** than GEMM — 2N² FLOPs for N² data (constant intensity), vs GEMM's intensity that grows with N. This means GEMV is more memory-bandwidth-bound than GEMM, and GPU speedup is more nuanced.

**VRAM-safe sizes (for 4 GB GPU):**

- Batch sizes: 64, 256, 512, 1024
- Dims: 128, 256, 512
- Max allocation: ~1 GB (well within 4 GB VRAM)

**Plots generated:**

- Left: execution time vs batch size, grouped by dimension (both CPU solid lines and GPU dashed lines)
- Right: grouped bar chart of GPU speedup per batch size, one bar group per dimension

**Practical insight:** For batch=1 inference (single-item serving), each Linear layer is a single GEMV. In this regime the CPU may be competitive with the GPU, which is why some production serving systems run small-batch inference on CPU to avoid PCIe transfer overhead.

---

## Part 3: Analysis and Conclusions (Cells 36–49)

### Cell 36–37 — Consolidated Results

Displays all 5 experiment DataFrames in one place — a unified table of CPU time, GPU time, and speedup for every (workload, size/batch) combination. This is the master results table.

---

### Cells 38–39 — Statistical Analysis

Two statistical tests:

**Paired t-test (scipy.stats.ttest_rel):**Takes the 20 CPU timings and 20 GPU timings from the largest problem size of each experiment, pairs them up, and tests whether the difference is statistically significant (p < 0.05).

- H₀: mean CPU time = mean GPU time
- H₁: they are different
- p < 0.05 → reject H₀ → difference is statistically significant

**Coefficient of Variation (CoV = std / mean):**
Measures how consistent/repeatable the measurements are. A CoV of 0.02 means measurements vary by ±2% — very stable. A CoV of 0.20 means ±20% variance — noisy.

Output table shows: CPU mean, GPU mean, CPU CoV, GPU CoV, t-statistic, p-value, and whether the result is significant.

---

### Cells 40–41 — Cross-Experiment Comparison

Aggregates all 5 experiments into a single comparison:

For each workload category, computes:

- Average speedup across all tested sizes
- Maximum speedup (best case)
- Minimum speedup (worst case / small sizes)

**Two plots:**

1. Grouped bar chart showing min/avg/max speedup per workload category
2. Horizontal bar chart ranking workloads by average GPU speedup (lowest to highest)

This is the most visually impactful chart in the notebook — it answers "which workload benefits most from GPU?" at a glance.

---

### Cells 42–43 — Answering Research Questions

Code that programmatically prints answers to RQ1–RQ4 using actual measured data:

- **RQ1:** Prints avg speedup for each workload category with "faster/slower" verdict
- **RQ2:** Computes Pearson correlation between matrix size N and speedup (r > 0 means speedup grows with size)
- **RQ3:** Counts how many (workload, size) combinations had CPU winning (speedup < 1.0)
- **RQ4:** Prints p-values and CoV from the statistical analysis

---

### Cells 44–45 — Hypothesis Evaluation

Code that evaluates H1–H4 with SUPPORTED / NOT SUPPORTED / PARTIALLY SUPPORTED verdicts:

- **H1:** Checks if max speedup for matmul or convolution exceeds 5× (>5× is treated as significant)
- **H2:** Checks Pearson r(N, speedup) > 0.5 for matrix multiplication
- **H3:** Checks if speedup at the smallest matrix size is < 2× (CPU still competitive)
- **H4:** Checks that sorting has the lowest avg speedup and element-wise has lower speedup than matmul

---

### Cell 46 — Discussion

A written interpretation of all results in context:

- Why matmul has the highest speedup (compute-bound, Tensor Cores)
- Why convolution has the second-highest speedup (cuDNN optimisation)
- Why element-wise ops have moderate speedup (bandwidth-bound)
- Why sorting has the lowest speedup (irregular memory access)
- A practical table: "Use GPU when... / Use CPU when..."
- Connects results back to Amdahl's Law

---

### Cell 47 — Limitations

Honest acknowledgement of what the results don't cover:

1. Single hardware — results are specific to RTX 3050 Laptop + this CPU
2. PCIe transfer overhead excluded — real applications may be slower
3. Thermal throttling on laptop GPU possible during long runs
4. Only float32 tested — float16/bfloat16 would show different profiles
5. PyTorch-specific backends — raw CUDA code may differ
6. Forward pass only for convolution — training (backward pass) adds more overhead

---

### Cell 48 — Conclusion

Four key conclusions:

1. GPU acceleration is workload-dependent — not universally faster
2. Compute-bound workloads (matmul, convolution) benefit most
3. Problem size matters — GPU advantage grows with larger inputs
4. Measurement rigour is essential — warm-up, sync, and repetition are necessary for reliable GPU benchmarks

---

### Cell 49 — References

10 academic and technical references including:

- NVIDIA CUDA Programming Guide
- PyTorch paper (NeurIPS 2019)
- Volkov & Demmel GPU benchmarking paper
- Amdahl's original 1967 paper
- cuDNN paper
- Harris's parallel reduction paper
- Merrill's GPU sorting paper
- Hennessy & Patterson computer architecture textbook

---

## Diagram Files Generated

When you run the notebook, 6 PNG files are saved to `./diagrams/`:

| File                                | What it shows                                          |
| ----------------------------------- | ------------------------------------------------------ |
| `exp1_matrix_cpu_vs_gpu.png`      | Matmul execution time + GPU speedup vs matrix size     |
| `exp2_elemwise_cpu_vs_gpu.png`    | 2×2 grid of element-wise op times                     |
| `exp2_elemwise_speedup.png`       | Speedup per operation vs tensor size                   |
| `exp3_sort_cpu_vs_gpu.png`        | Sort time + speedup vs array size                      |
| `exp4_conv_cpu_vs_gpu.png`        | Conv time + speedup bar chart vs batch size            |
| `exp5_gemv_cpu_vs_gpu.png`        | GEMV time + grouped speedup bars                       |
| `cross_experiment_comparison.png` | Master comparison: all workloads ranked by GPU speedup |

---

## Summary Table

| Section               | Cells  | Purpose                                 |
| --------------------- | ------ | --------------------------------------- |
| Title & Team          | 0      | Identity                                |
| Introduction          | 1      | Motivation                              |
| Background Theory     | 2      | Academic foundation                     |
| Problem Statement     | 3      | Research gap                            |
| Objectives            | 4      | What we measure                         |
| Research Questions    | 5      | Questions to answer                     |
| Hypotheses            | 6      | Predictions before experiments          |
| Methodology           | 7      | How experiments are run                 |
| Environment           | 8–10  | Hardware/software baseline              |
| Utilities             | 11–13 | `benchmark_fn`, `sync`, `speedup` |
| Configuration         | 14–15 | All tunable parameters                  |
| Experiment 1          | 16–19 | Matrix multiplication CPU vs GPU        |
| Experiment 2          | 20–23 | Element-wise & reductions CPU vs GPU    |
| Experiment 3          | 24–27 | Sorting CPU vs GPU                      |
| Experiment 4          | 28–31 | CNN convolution CPU vs GPU              |
| Experiment 5          | 32–35 | Batched GEMV CPU vs GPU                 |
| Consolidated Results  | 36–37 | Master results table                    |
| Statistical Analysis  | 38–39 | t-tests and CoV                         |
| Cross-Experiment      | 40–41 | Ranked speedup comparison chart         |
| RQ Answers            | 42–43 | Data-driven answers to RQ1–RQ4         |
| Hypothesis Evaluation | 44–45 | SUPPORTED/NOT SUPPORTED verdicts        |
| Discussion            | 46     | Interpretation and practical guidance   |
| Limitations           | 47     | Honest scope acknowledgement            |
| Conclusion            | 48     | 4 key takeaways                         |
| References            | 49     | 10 citations                            |
