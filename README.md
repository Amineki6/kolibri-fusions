# Kolibri-Fusions: Production In-Register CUDA Kernels for 78B MoE Inference

[![Platform](https://img.shields.io/badge/Platform-NVIDIA%20Ada%20Lovelace%20(RTX%204090)-76B900.svg)](https://www.nvidia.com)
[![CUDA](https://img.shields.io/badge/CUDA-12.8%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14%20%7C%202.6-EE4C2C.svg)](https://pytorch.org)
[![TensorRT-LLM](https://img.shields.io/badge/TensorRT--LLM-v1.3.0-76B900.svg)](https://github.com/NVIDIA/TensorRT-LLM)
[![Presentation Slides](https://img.shields.io/badge/Presentation-Slide%20Deck%20(PDF)-blue.svg)](assets/Kolibri%20Optimization%20--%20Presentation.pdf)

Production CUDA kernel fusions for the **Kolibri 1 (78B MoE)** decoder stack in **NVIDIA TensorRT-LLM**, eliminating non-GEMM memory bandwidth bottlenecks and launch bubbles across 50 decoder layers.

> **Presentation Slide Deck**: A high-level overview of the architectural challenges, kernel fusions, and empirical results is available in [`Kolibri Optimization -- Presentation.pdf`](assets/Kolibri%20Optimization%20--%20Presentation.pdf).

---

## 1. Executive Summary

While large matrix multiplications (GEMMs) dominate total FLOPs during large-batch prefill, **decode latency in sparse Mixture-of-Experts (MoE) architectures is bottlenecked by non-GEMM memory bandwidth and kernel launch latency**. In Kolibri 1 (50 layers, $E = 384$ experts, $\text{Top-}K = 6$, $D = 2560$), standard TensorRT-LLM dispatches up to 7 separate memory-bound kernels per layer for normalization, residual accumulation, router top-$K$ selection, dispatch mapping, and token expansion.

This repository provides **three self-contained, production CUDA fusions** that replace these 7 disjoint operations with 3 highly tuned, in-register kernels:

1. **Region 1: Sandwich Norm (`fusions/01_sandwich_norm`)**: Fused Post-Attention RMSNorm + Residual Add + Pre-MoE RMSNorm (**1.85× – 2.18×** speedup).
2. **Region 2: MoE Prologue (`fusions/03_moe_prologue`)**: Fused Router Top-6 + Dispatch Mapping + Vectorized Row Expansion (**1.63× – 2.90×** speedup).
3. **Region 3: Post-MoE Norm (`fusions/02_post_moe_norm`)**: Fused Routed + Shared Expert Add + Post-FFN RMSNorm (**1.60× – 2.40×** speedup).

Across the entire 50-layer decoder stack, these optimizations save **1.11 ms to 9.82 ms per forward pass**, accelerating non-GEMM decode by over **2.2×**.

---

## 2. Kolibri Architecture & Optimization Regions

![Kolibri Architecture and Optimization Regions](assets/img/Kolibri%20Architecture.jpg)

---

## 3. Empirical Results

All benchmarks were evaluated on an **NVIDIA GeForce RTX 4090** (Ada Lovelace, 128 SMs, 24 GB GDDR6X, sm_89) under **TensorRT-LLM 1.3.0** with locked clocks and real 2-block production weights (`checkpoints/kolibri_2blocks_real.pt`).

### Level 1: Fusion Regions Breakdown (Microbenchmarks)
*Latency measured in microseconds ($\mu\text{s}$), paired execution across 200 runs with interleaved clock-drift cancellation:*

![Region Latency Breakdown](assets/img/fig1_region_latency.png)

| Tokens ($N$) | R1 Baseline | R1 Fused | R1 Speedup | R2 Baseline | R2 Fused | R2 Speedup | R3 Baseline | R3 Fused | R3 Speedup | **Total Fusion Speedup** |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** (Decode) | 18.15 $\mu$s | 9.63 $\mu$s | **1.89×** | 51.29 $\mu$s | 20.16 $\mu$s | **2.54×** | 12.13 $\mu$s | 7.22 $\mu$s | **1.68×** | **2.20×** |
| **16** | 17.80 $\mu$s | 9.44 $\mu$s | **1.88×** | 55.68 $\mu$s | 20.95 $\mu$s | **2.66×** | 11.98 $\mu$s | 7.21 $\mu$s | **1.66×** | **2.27×** |
| **64** | 18.82 $\mu$s | 10.70 $\mu$s | **1.76×** | 63.08 $\mu$s | 22.37 $\mu$s | **2.82×** | 12.34 $\mu$s | 7.47 $\mu$s | **1.65×** | **2.33×** |
| **256** | 26.58 $\mu$s | 13.48 $\mu$s | **1.97×** | 79.77 $\mu$s | 27.88 $\mu$s | **2.86×** | 16.08 $\mu$s | 9.55 $\mu$s | **1.68×** | **2.40×** |
| **512** (Prefill) | 38.78 $\mu$s | 17.83 $\mu$s | **2.18×** | 102.73 $\mu$s | 37.70 $\mu$s | **2.72×** | 22.56 $\mu$s | 10.48 $\mu$s | **2.15×** | **2.49×** |
| **1024** | 67.15 $\mu$s | 31.36 $\mu$s | **2.14×** | 147.79 $\mu$s | 65.18 $\mu$s | **2.27×** | 34.68 $\mu$s | 14.47 $\mu$s | **2.40×** | **2.25×** |
| **4096** | 230.30 $\mu$s | 124.78 $\mu$s | **1.85×** | 560.50 $\mu$s | 344.88 $\mu$s | **1.63×** | 189.08 $\mu$s | 117.52 $\mu$s | **1.61×** | **1.67×** |

---

### Level 2: End-to-End 2-Block Forward Pass
*Measured with real CUTLASS Grouped MoE GEMMs, shared expert FFNs, and FlashAttention-2:*

| Tokens ($N$) | Baseline 2-Block ($\mu\text{s}$) | Fused 2-Block ($\mu\text{s}$) | $\Delta t_{\text{2-block}}$ Saved | E2E Speedup | Fusion Region Speedup |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | 470.1 $\pm$ 11.1 | 425.1 $\pm$ 2.3 | **+44.6 $\mu\text{s}$** | **1.106×** | **2.20×** |
| **16** | 628.0 $\pm$ 4.9 | 579.8 $\pm$ 1.2 | **+47.9 $\mu\text{s}$** | **1.083×** | **2.27×** |
| **64** | 803.4 $\pm$ 4.7 | 750.6 $\pm$ 3.3 | **+53.7 $\mu\text{s}$** | **1.070×** | **2.33×** |
| **256** | 1141.4 $\pm$ 4.4 | 1072.6 $\pm$ 1.7 | **+71.5 $\mu\text{s}$** | **1.064×** | **2.40×** |
| **512** | 1802.8 $\pm$ 5.1 | 1673.8 $\pm$ 2.9 | **+98.1 $\mu\text{s}$** | **1.077×** | **2.49×** |
| **1024** | 3483.2 $\pm$ 7.5 | 3341.6 $\pm$ 5.8 | **+138.6 $\mu\text{s}$** | **1.042×** | **2.25×** |
| **4096** | 15381.1 $\pm$ 112.9 | 15001.1 $\pm$ 100.4 | **+392.7 $\mu\text{s}$** | **1.025×** | **1.67×** |

---

### Level 3: 50-Layer Full Model Projection

Because Kolibri's 50 decoder layers share identical dimensions ($D=2560, E=384, K=6$) and execute sequentially, the measured latency reduction scales linearly across depth:
$$\Delta t_{\text{50-layer}} = 50 \times \left( \frac{\Delta t_{\text{2-block}}}{2} \right) = 25 \times \Delta t_{\text{2-block}}$$

![Full Model 50-Layer Latency Savings Projection](assets/img/fig5_full_model.png)

* **Single-Token Decode ($N = 1$)**: Saves **1.115 ms per token** forward pass.
* **Mid-Batch ($N = 512$)**: Saves **2.453 ms per forward pass**.
* **Long Prefill ($N = 4096$)**: Saves **9.818 ms per forward pass**.

---

## 4. Integration into TensorRT-LLM

In addition to the self-contained fusions in this repository, all optimized kernels are integrated directly into [TensorRT-LLM](https://github.com/Amineki6/TensorRT-LLM) on branch `perf/kolibri-optimizations`:

### 1. C++ & CUDA Core Subsystem
* **Kernels Directory** (`cpp/tensorrt_llm/kernels/moe/`):
  * `sandwich_norm/`: Implements `kolibriSandwichNormKernel.cu` & `.h` for in-register residual and norm fusion.
  * `routing/`: Implements `kolibriRoutingKernels.cu` & `.h` for Bitonic padded Top-6 routing, dynamic dispatch mapping, and 128-bit row expansion.
  * `post_norm/`: Implements `kolibriPostMoeNormKernel.cu` & `.h` for fused routed + shared expert accumulation with RMSNorm.
* **PyTorch C++ Extensions (THOP)** (`cpp/tensorrt_llm/thop/moe/`):
  * Registered as native `torch.ops.trtllm.*` operators (`kolibri_sandwich_norm_op`, `kolibri_router_op`, `kolibri_post_moe_norm_op`, `kolibri_build_expert_maps_op`, `kolibri_activation_expansion_op`).
  * Seamlessly discovered via CMake's `file(GLOB_RECURSE ...)` targets into `th_common`.

### 2. PyTorch Model Integration (`tensorrt_llm/_torch/models/modeling_kolibri.py`)
* **Sandwich Norm**: `Kolibri1DecoderLayer.forward` invokes `torch.ops.trtllm.kolibri_sandwich_norm_op` directly in the forward graph.
* **Router**: `Kolibri1RoutingMethod.apply` dispatches to `torch.ops.trtllm.kolibri_router_op`, bypassing disjoint PyTorch sorting and indexing passes.
* **Post-MoE Norm**: `Kolibri1MoE.forward` incorporates a single-GPU ($TP = 1$) fast-path calling `torch.ops.trtllm.kolibri_post_moe_norm_op`, while preserving standard AllReduce accumulation for multi-GPU Tensor Parallelism.
* **CUDA Graph & torch.compile Compatibility**: All operations register abstract fake tensor implementations in `tensorrt_llm/_torch/custom_ops/cpp_custom_ops.py`.

---

## 5. Key Microarchitectural Innovations

### 1. In-Register Residual Fusion (Region 1 & 3)
Standard TRT-LLM invokes separate kernels for residual elementwise additions and normalization passes, writing out full $N \times D$ matrices to DRAM and immediately reading them back. Our kernels:
* Hold intermediate accumulated values directly in **128-bit vector registers** (`Vec4B_t`, `ushort4`).
* Compute RMSNorm sums using **warp shuffle down intrinsics** (`__shfl_down_sync`), eliminating dynamic shared memory allocations and shared-memory synchronization barriers.
* Emit final results in a single coalesced 128-bit global store (`Vec4B_t::store_coalesced`).

### 2. Saturated 256-Thread Block Geometry (Region 3)
* **Baseline TRT-LLM `generalRmsNorm`**: Configured with 1024 threads (32 warps), allowing only 1 CTA per SM on Ada (occupancy capped at 66.7%). Schedulers spent **64.0% of cycles stalled on "No Eligible" warps** due to barrier synchronization.
* **Our Fused Kernel**: Configured with 256 threads (8 warps), hosting **6 active CTAs per SM (100% occupancy)**. Scheduler idle dropped to **8.6%**, saturating DRAM bandwidth at **95.1% (934.7 GB/s)**.

### 3. Dynamic Hybrid MoE Prologue ($N=512$ Tipping Point)
Routing in large-expert MoE models ($E=384$) incurs severe launch overhead if broken into separate scan passes. Our prologue implements an adaptive dispatch:
* **Decode Regime ($N < 512$)**: Employs a single thread block with 3.0 KB shared memory to construct expert histograms and prefix sums in-place. Total latency is an irreducible **~13 $\mu\text{s}$**, eliminating the multi-kernel launch penalty.
* **Prefill Regime ($N \ge 512$)**: Transitions to a multi-block 3-stage pipeline that distributes work across all 128 SMs, outperforming CUB radix scan by up to **68.9×** for large batches up to $N = 98,304$.

---

## 6. Repository Structure

```
kolibri-fusions/
├── README.md                          # Comprehensive publication overview & benchmark summary
├── fusions/                           # The 3 self-contained production fusions
│   ├── 01_sandwich_norm/              # Region 1: Post-Attention Sandwich Norm + Residual Add + Pre-MoE Norm
│   │   ├── fused/                     # Production in-register CUDA kernel & PyTorch C++ extension
│   │   │   ├── custom_sandwich_norm.cu
│   │   │   ├── custom_sandwich_norm.cuh
│   │   │   ├── sandwich_norm_binding.cpp
│   │   │   └── fused_sandwich_norm.py
│   │   ├── baseline/                  # Native TRT-LLM 3-kernel reference implementation
│   │   │   ├── trtllm_sandwich_norm_baseline.cu
│   │   │   ├── trtllm_sandwich_norm_baseline.cuh
│   │   │   ├── baseline_binding.cpp
│   │   │   └── trtllm_baseline.py
│   │   ├── benchmark_and_verify.py    # Microbenchmark & bit-exact parity verification
│   │   └── README.md                  # Microarchitectural specs, register analysis, DRAM savings
│   │
│   ├── 02_post_moe_norm/              # Region 3: Routed + Shared Expert Add + Post-FFN Norm
│   │   ├── fused/
│   │   ├── baseline/
│   │   ├── benchmark_and_verify.py
│   │   └── README.md
│   │
│   └── 03_moe_prologue/               # Region 2: MoE Router Top-6 + Dispatch Map + Activation Expand
│       ├── fused/
│       ├── baseline/
│       ├── benchmark_and_verify.py
│       └── README.md
│
├── benchmarks/                        # Finalized evaluation suite
│   ├── benchmark_2block_e2e.py        # Primary E2E 2-block forward pass benchmark harness
│   ├── weights_loader.py              # Production weight extractor & deterministic synthetic fallback
│   ├── figstyle.py                    # Publication typography, palette, and style definitions
│   ├── generate_e2e_figures.py        # Generates publication figures (PDF, PNG, SVG)
│   ├── generate_r2_figures.py         # Generates prologue stage breakdown figures
│   ├── cutlass_grouped_moe.py         # TRT-LLM CUTLASS Grouped MoE GEMM wrapper
│   └── cutlass_grouped_moe_binding.cpp
│
├── results/                           # Verified benchmark reports
│   ├── e2e_benchmark_report.csv       # Free routing macro data
│   ├── e2e_benchmark_report.md
│   ├── e2e_benchmark_report_meta.json
│   ├── e2e_benchmark_report_pinned.csv# Pinned routing control data
│   ├── e2e_benchmark_report_pinned.md
│   ├── e2e_benchmark_report_pinned_meta.json
│   └── r2_stage_breakdown.csv         # MoE prologue breakdown metrics
│
└── assets/                            # Publication-ready figures & presentation
    ├── Kolibri Optimization -- Presentation.pdf
    └── img/                           # Figures used throughout documentation
```

---

## 7. Quick Start & Reproduction

### Prerequisites
* NVIDIA GPU with Compute Capability $\ge 8.0$ (Optimized for Ada Lovelace `sm_89`)
* CUDA Toolkit 12.0+ & `g++` 11+
* PyTorch 2.1+ with CUDA support
* TensorRT-LLM 1.3.0+

```bash
# Clone the repository
git clone https://github.com/Amineki6/kolibri-fusions.git
cd kolibri-fusions
```

### 1. Verify Numerical Parity & Microbenchmarks
Verify bit-exact output equivalence against PyTorch eager and native TRT-LLM:

```bash
# Region 1: Sandwich Norm
python fusions/01_sandwich_norm/benchmark_and_verify.py

# Region 2: Dynamic Hybrid MoE Prologue
python fusions/03_moe_prologue/benchmark_and_verify.py

# Region 3: Post-MoE Norm
python fusions/02_post_moe_norm/benchmark_and_verify.py
```

### 2. Run End-to-End 2-Block Benchmark
Run the comprehensive E2E benchmark across token batch sizes $N \in [1, 4096]$:

```bash
# Free routing sweep
python benchmarks/benchmark_2block_e2e.py --runs 200 \
    --export-csv results/e2e_benchmark_report.csv \
    --export-markdown results/e2e_benchmark_report.md

# Pinned routing control (isolates kernel speedup from FP tie-break divergence)
python benchmarks/benchmark_2block_e2e.py --runs 200 --pin-routing \
    --export-csv results/e2e_benchmark_report_pinned.csv \
    --export-markdown results/e2e_benchmark_report_pinned.md
```

### 3. Generate Publication Figures
Render high-resolution PDF and PNG figures:

```bash
# Generate End-to-End Figures
python benchmarks/generate_e2e_figures.py --formats pdf png

# Generate Region 2 Stage Breakdown Figures
python benchmarks/generate_r2_figures.py --formats pdf png
```

---

## 8. Citation & Acknowledgments

This research was conducted as part of the Master's thesis on **High-Performance MoE Inference Optimization in TensorRT-LLM**. Special thanks to Aleph Alpha for open-sourcing the Kolibri architecture and weights, and NVIDIA for the TensorRT-LLM framework.
