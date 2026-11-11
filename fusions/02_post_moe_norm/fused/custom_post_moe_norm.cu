#include "custom_post_moe_norm.cuh"

#include <stdexcept>
#include <string>

namespace kolibri
{
namespace fusions
{

// Single-barrier block reduction: warp shuffle followed by a NUM_WARPS-warp broadcast read
template <int NUM_WARPS>
__device__ __forceinline__ float blockSumAll(float v, float* red)
{
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
    {
        v += __shfl_xor_sync(0xffffffffu, v, o);
    }
    if ((threadIdx.x & 31) == 0)
    {
        red[threadIdx.x >> 5] = v;
    }
    __syncthreads();
    float t = 0.f;
    #pragma unroll
    for (int w = 0; w < NUM_WARPS; ++w)
    {
        t += red[w]; // broadcast read across CTA
    }
    return t; // valid across every thread in the CTA
}

// =================================================================================================
// Production Fused Add + Post-MoE Norm Kernel (SM 89 Ada Lovelace Optimized)
//
// High-Occupancy Coalesced 4-Byte Transactions: 2 BF16/FP16 elements per thread (LDG.E.32).
// A 32-thread warp requests 128 bytes, exactly matching the hardware L2/DRAM cache-line size.
// High Occupancy (32 Registers / Thread): 256 threads (8 warps) with 5 iterations drops register
// pressure to 32 registers, reaching theoretical 100% SM occupancy (48 warps / SM).
// Perfect Warp Scheduler Load Balancing: 8 warps map symmetrically (2 warps per scheduler) across
// the 4 processing sub-cores (SMSPs) on each Ada Lovelace SM.
// In-Register Fusion: Adds routed_out and shared_out in registers without DRAM or dynamic shared memory.
// Single-Barrier CTA Reduction: Fast blockSumAll using warp shuffles and 32-byte static shared memory.
// =================================================================================================
template <typename T, int NUM_WARPS, int ITERS, int HIDDEN_DIM>
__global__ void __launch_bounds__(NUM_WARPS * 32)
fusedAddPostMoeNormRegKernel(
    PackedFloat2<T> const* __restrict__ routed_out,
    PackedFloat2<T> const* __restrict__ shared_out,
    PackedFloat2<T> const* __restrict__ gamma,
    PackedFloat2<T>* __restrict__ moe_out,
    float const eps)
{
    using V = PackedFloat2<T>;
    constexpr int THREADS = NUM_WARPS * 32;
    constexpr int N_VEC   = THREADS * ITERS;
    constexpr float INV_HIDDEN_DIM = 1.0f / float(HIDDEN_DIM);

    __shared__ float red[NUM_WARPS];

    size_t const row = size_t(blockIdx.x) * N_VEC;
    int const tid = threadIdx.x;

    V c[ITERS];
    V g[ITERS];

    // Standard 4-byte loads: fully coalesced 128-byte warp memory requests
    #pragma unroll
    for (int k = 0; k < ITERS; ++k)
    {
        int const i = tid + k * THREADS;
        V routed = routed_out[row + i];
        V shared = shared_out[row + i];
        c[k] = routed.add(shared);
        g[k] = gamma[i];
    }

    // Accumulate sum of squares in registers
    float ss = 0.f;
    #pragma unroll
    for (int k = 0; k < ITERS; ++k)
    {
        ss += c[k].sum_of_squares();
    }

    // Intra-block reduction
    float const sum = blockSumAll<NUM_WARPS>(ss, red);
    float const inv = rsqrtf(sum * INV_HIDDEN_DIM + eps);

    // Apply RMS scale and gamma weight, then store directly to global DRAM
    #pragma unroll
    for (int k = 0; k < ITERS; ++k)
    {
        moe_out[row + tid + k * THREADS] = c[k].rmsnorm(inv, g[k]);
    }
}

// Host launcher function
template <typename T>
void invokeFusedAddPostMoeNorm(
    T const* routed_out,
    T const* shared_out,
    T const* gamma,
    T* moe_out,
    int const tokens,
    int const hidden_dim,
    float const eps,
    cudaStream_t stream)
{
    constexpr int HIDDEN_DIM = 2560;
    if (hidden_dim != HIDDEN_DIM)
    {
        throw std::runtime_error("Fused post-MoE norm requires hidden_dim=2560");
    }

    dim3 const grid(tokens);
    using V4B = PackedFloat2<T>;

    // High-occupancy Kolibri configuration: 256 threads (8 warps), 5 iterations per thread
    fusedAddPostMoeNormRegKernel<T, 8, 5, HIDDEN_DIM><<<grid, 256, 0, stream>>>(
        reinterpret_cast<V4B const*>(routed_out),
        reinterpret_cast<V4B const*>(shared_out),
        reinterpret_cast<V4B const*>(gamma),
        reinterpret_cast<V4B*>(moe_out),
        eps
    );
}

// Explicit template instantiations
template void invokeFusedAddPostMoeNorm<__nv_bfloat16>(
    __nv_bfloat16 const* routed_out,
    __nv_bfloat16 const* shared_out,
    __nv_bfloat16 const* gamma,
    __nv_bfloat16* moe_out,
    int const tokens,
    int const hidden_dim,
    float const eps,
    cudaStream_t stream);

template void invokeFusedAddPostMoeNorm<half>(
    half const* routed_out,
    half const* shared_out,
    half const* gamma,
    half* moe_out,
    int const tokens,
    int const hidden_dim,
    float const eps,
    cudaStream_t stream);

template void invokeFusedAddPostMoeNorm<float>(
    float const* routed_out,
    float const* shared_out,
    float const* gamma,
    float* moe_out,
    int const tokens,
    int const hidden_dim,
    float const eps,
    cudaStream_t stream);

} // namespace fusions
} // namespace kolibri
