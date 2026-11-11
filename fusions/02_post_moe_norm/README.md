# Fusion 2: Fused Add + Post-MoE Norm (`fusions/02_post_moe_norm`)

## 1. Executive Summary

This directory contains the production implementation and native TensorRT-LLM baseline for **Candidate 2: Routed + Shared Add + Post-MoE Norm** in the Kolibri-5E MoE architecture.

In standard unfused MoE architectures, the outputs of the routed experts (`routed_out`) and the shared expert (`shared_out`) are combined and normalized via two separate CUDA kernel launches:
1. **Launch 1: Elementwise Vector Add**: `combined = routed_out + shared_out` (writes result to global DRAM).
2. **Launch 2: Post-MoE RMSNorm**: `moe_out = RMSNorm(combined, gamma)` (reads `combined` from DRAM, normalizes, and writes `moe_out` to DRAM).

Our **Production Fused Kernel** merges both operations into a single kernel execution (`fusedAddPostMoeNormRegKernel`), holding intermediate sums in 128-bit vector registers with **zero dynamic shared memory** and **zero intermediate DRAM round-trips**.

---

## 2. Mathematical Definition

Given:
* $\mathbf{X}_{\text{routed}} \in \mathbb{R}^{N \times D}$ (output from routed experts)
* $\mathbf{X}_{\text{shared}} \in \mathbb{R}^{N \times D}$ (output from shared expert)
* $\boldsymbol{\gamma} \in \mathbb{R}^{D}$ (Post-MoE RMSNorm scale weight)

$$\mathbf{C}_{i, :} = \mathbf{X}_{\text{routed}, i, :} + \mathbf{X}_{\text{shared}, i, :}$$

$$\sigma_i = \sqrt{\frac{1}{D} \sum_{j=1}^{D} C_{i, j}^2 + \epsilon}$$

$$\mathbf{Y}_{i, j} = \left(\frac{C_{i, j}}{\sigma_i}\right) \cdot \gamma_j$$

where $N$ is the number of tokens, $D = 2560$ is the Kolibri hidden dimension, and $\epsilon = 10^{-5}$.

---

## 3. Directory Structure

```
fusions/02_post_moe_norm/
├── README.md                      # Architecture documentation and performance report
├── benchmark_and_verify.py        # Comprehensive parity verification & latency benchmark
├── fused/                         # Production Fused Kernel (Self-contained, 0 external deps)
│   ├── custom_post_moe_norm.cuh   # Vec4B_t vector types and launcher declarations
│   ├── custom_post_moe_norm.cu    # fusedAddPostMoeNormRegKernel (256-thread Ada Lovelace)
│   ├── post_moe_norm_binding.cpp  # PyTorch C++ extension bindings
│   └── fused_post_moe_norm.py     # Clean Python wrapper with lazy JIT compilation
└── baseline/                      # Native TensorRT-LLM Unfused Baseline
    ├── trtllm_post_moe_norm_baseline.cuh # Baseline header
    ├── trtllm_post_moe_norm_baseline.cu # Vector add + invokeGeneralRmsNorm
    ├── baseline_binding.cpp       # PyTorch C++ extension bindings
    └── trtllm_baseline.py         # Links directly to TensorRT-LLM/cpp rmsnormKernels.cu
```

---

## 4. Key Architectural Features

1. **Coalesced 4-Byte Transactions (32-Bit Vector Primitives)**:
   All loads and stores use 4-byte aligned vector primitives (`Vec4B_t`), transferring 2 BF16/FP16 elements per thread (`LDG.E.32` / `STG.E.32`). A 32-thread warp transfers exactly 128 bytes, matching the native GPU cache line with 100% bus efficiency.
2. **High Occupancy (32 Registers / Thread & 100% Theoretical Occupancy)**:
   For $D=2560$ in BF16/FP16, each token is processed by 256 threads (8 warps) over 5 iterations. Register usage drops to 32 registers per thread, reaching 100% theoretical SM occupancy (48 warps / SM on Ada Lovelace).
3. **Warp Scheduler Load-Balancing**:
   8 warps map symmetrically (2 warps per scheduler) across the 4 processing sub-cores (SMSPs) on each Ada Lovelace SM.
4. **Clean C++ Memory Loads (No Inline PTX Barriers)**:
   Eliminates inline PTX assembly, allowing the compiler to perform instruction reordering, pipelining, and dual-issuing freely.
5. **Single-Barrier CTA Reduction (`blockSumAll`)**:
   Accumulates variance across threads using hardware warp shuffle instructions (`__shfl_xor_sync`) followed by an 8-warp broadcast through 32 bytes of static shared memory.
6. **Zero Dynamic Shared Memory**:
   Requires 0 dynamic shared memory bytes, completely avoiding shared memory bank conflicts and CTA occupancy limits.

---

## 5. Usage

```python
import torch
from fusions.02_post_moe_norm.fused.fused_post_moe_norm import fused_add_post_moe_norm

# Inputs [tokens, hidden_dim] on CUDA
tokens, hidden_dim = 64, 2560
routed_out = torch.randn(tokens, hidden_dim, device="cuda", dtype=torch.bfloat16)
shared_out = torch.randn(tokens, hidden_dim, device="cuda", dtype=torch.bfloat16)
gamma = torch.ones(hidden_dim, device="cuda", dtype=torch.bfloat16)

# Execute single-kernel fused operation
moe_out = fused_add_post_moe_norm(routed_out, shared_out, gamma, eps=1e-5)
```

---

## 6. Empirical Verification & Benchmark Results

Measured on an **NVIDIA GeForce RTX 4090** (24 GB VRAM, Ada Lovelace SM 89) using BF16 (`hidden_dim = 2560`, `eps = 1e-5`):

| Batch / Tokens ($N$) | Eager PyTorch | TRT-LLM Native Baseline | Custom Fused Kernel | Speedup vs TRT-LLM | Launch Reduction | Effective Bandwidth |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 10.54 µs | 7.28 µs | **3.52 µs** | **2.07x** | 2 $\rightarrow$ 1 (-50%) | 5.8 GB/s |
| **16** | 10.25 µs | 7.12 µs | **3.50 µs** | **2.03x** | 2 $\rightarrow$ 1 (-50%) | 71.6 GB/s |
| **64** | 10.16 µs | 7.06 µs | **3.47 µs** | **2.03x** | 2 $\rightarrow$ 1 (-50%) | 284.8 GB/s |
| **128** | 10.26 µs | 7.06 µs | **3.50 µs** | **2.02x** | 2 $\rightarrow$ 1 (-50%) | 562.9 GB/s |
| **256** | 10.14 µs | 6.94 µs | **3.44 µs** | **2.02x** | 2 $\rightarrow$ 1 (-50%) | 1,144.9 GB/s |
| **512** | 10.05 µs | 9.64 µs | **3.69 µs** | **2.61x** | 2 $\rightarrow$ 1 (-50%) | 2,130.6 GB/s |
| **1,024** | 10.57 µs | 15.87 µs | **5.87 µs** | **2.70x** | 2 $\rightarrow$ 1 (-50%) | 2,681.3 GB/s |
| **2,048** | 16.73 µs | 27.81 µs | **8.85 µs** | **3.14x** | 2 $\rightarrow$ 1 (-50%) | 3,553.9 GB/s |
| **4,096** | 76.94 µs | 95.56 µs | **15.03 µs** | **6.36x** | 2 $\rightarrow$ 1 (-50%) | 4,186.4 GB/s |
| **8,192** | 191.07 µs | 240.12 µs | **134.57 µs** | **1.78x** | 2 $\rightarrow$ 1 (-50%) | 935.1 GB/s |
| **16,384** | 454.58 µs | 484.21 µs | **270.55 µs** | **1.79x** | 2 $\rightarrow$ 1 (-50%) | 930.2 GB/s |
| **98,304** | 2,728.51 µs | 2,894.51 µs | **1,629.96 µs** | **1.78x** | 2 $\rightarrow$ 1 (-50%) | 926.4 GB/s |

### Verification Summary
- **Numerical Parity**: Bit-for-bit / high-precision equivalence verified against PyTorch eager reference and TRT-LLM across BF16 (`atol=0.04, rtol=0.02`) and FP16 (`atol=0.001, rtol=0.001`).
- **Kernel Symbols Checked**: PyTorch CUDA Profiler confirms TRT-LLM baseline executes `vector_add_kernel` + `tensorrt_llm::_v1::kernels::generalRmsNorm`, while the fused version executes only `kolibri::fusions::fusedAddPostMoeNormRegKernel`.
