#include "custom_sandwich_norm.cuh"

#include <stdexcept>
#include <string>

namespace kolibri
{
namespace fusions
{

// Single-barrier block reduction: warp shuffle followed by a broadcast sum across warps
template <int NUM_WARPS>
__device__ __forceinline__ float blockSumAll(float v, float* red)
{
    // Intra-warp reduction: butterfly exchange summing across all 32 lanes in registers
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
    {
        v += __shfl_xor_sync(0xffffffffu, v, o);
    }

    // First thread of each warp (lane 0) writes its warp-level sum to shared memory
    if ((threadIdx.x & 31) == 0)
    {
        red[threadIdx.x >> 5] = v;
    }
    __syncthreads();

    // Broadcast accumulation: every thread directly sums the small array of warp totals.
    // For small warp counts (such as 5 warps), this avoids a second tree reduction and barrier.
    float t = 0.f;
    #pragma unroll
    for (int w = 0; w < NUM_WARPS; ++w)
    {
        t += red[w];
    }
    return t; // The complete block sum is now available in registers across all threads
}

// =================================================================================================
// Fused Sandwich Norm Kernel (4-byte packed float vectorization for 128-byte coalesced warp transfers)
// =================================================================================================
template <typename T, int NUM_WARPS, int ITERS, int HIDDEN_DIM>
__global__ void __launch_bounds__(NUM_WARPS * 32)
fusedSandwichNormKernel(
    PackedFloat2<T> const* __restrict__ attn_out,
    PackedFloat2<T> const* __restrict__ residual_in,
    PackedFloat2<T> const* __restrict__ gamma1,
    PackedFloat2<T> const* __restrict__ gamma2,
    PackedFloat2<T>* __restrict__ updated_residual_out,
    PackedFloat2<T>* __restrict__ moe_in_out,
    float const eps)
{
    using P = PackedFloat2<T>;
    constexpr int THREADS = NUM_WARPS * 32;
    constexpr int N_PACKED = THREADS * ITERS; // 160 * 8 = 1280 packed pairs (2560 elements)
    constexpr float INV_HIDDEN_DIM = 1.0f / float(HIDDEN_DIM);

    // Shared memory holding warp sums for block-wide reduction
    __shared__ float red1[NUM_WARPS];
    __shared__ float red2[NUM_WARPS];

    // Each block processes a single token row across the hidden dimension
    size_t const row_packed = size_t(blockIdx.x) * N_PACKED;
    int const tid = threadIdx.x;

    // Registers to cache attention output and first norm scale weights (as 4-byte packed pairs)
    P a[ITERS];
    P g1[ITERS];

    // Load attention output and gamma1 into registers with packed instructions (128 bytes per warp)
    #pragma unroll
    for (int k = 0; k < ITERS; ++k)
    {
        int const i = tid + k * THREADS;
        a[k] = attn_out[row_packed + i];
        g1[k] = gamma1[i];
    }

    // Accumulate sum of squares for the first RMSNorm (post-attention sandwich norm)
    float ss = 0.f;
    #pragma unroll
    for (int k = 0; k < ITERS; ++k)
    {
        ss += a[k].sum_of_squares();
    }

    // Load input residual while the warp/block reduction runs
    P r[ITERS];
    #pragma unroll
    for (int k = 0; k < ITERS; ++k)
    {
        r[k] = residual_in[row_packed + tid + k * THREADS];
    }

    // Compute reciprocal RMS scaling factor for first norm across the block
    float const inv1 = rsqrtf(blockSumAll<NUM_WARPS>(ss, red1) * INV_HIDDEN_DIM + eps);

    // Apply first RMSNorm, accumulate residual, write updated residual, and compute second norm sum of squares
    ss = 0.f;
    #pragma unroll
    for (int k = 0; k < ITERS; ++k)
    {
        a[k] = a[k].rmsnorm(inv1, g1[k]).add(r[k]);
        updated_residual_out[row_packed + tid + k * THREADS] = a[k];
        ss += a[k].sum_of_squares();
    }

    // Load gamma2 weights for the second norm
    P g2[ITERS];
    #pragma unroll
    for (int k = 0; k < ITERS; ++k)
    {
        g2[k] = gamma2[tid + k * THREADS];
    }

    // Compute reciprocal RMS scaling factor for second norm across the block
    float const inv2 = rsqrtf(blockSumAll<NUM_WARPS>(ss, red2) * INV_HIDDEN_DIM + eps);

    // Apply second RMSNorm and write out MoE input
    #pragma unroll
    for (int k = 0; k < ITERS; ++k)
    {
        moe_in_out[row_packed + tid + k * THREADS] = a[k].rmsnorm(inv2, g2[k]);
    }
}

template <typename T>
void invokeFusedSandwichNormReg(
    T const* attn_out,
    T const* residual_in,
    T const* gamma1,
    T const* gamma2,
    T* updated_residual_out,
    T* moe_in_out,
    int const tokens,
    int const hidden_dim,
    float const eps,
    cudaStream_t stream)
{
    constexpr int NUM_WARPS = 5;
    constexpr int ITERS = 8; // 5 * 32 * 8 = 1280 pairs = 2560 elements
    constexpr int HIDDEN_DIM = 2560;

    if (hidden_dim != HIDDEN_DIM)
    {
        throw std::runtime_error("Fused sandwich norm requires hidden_dim=2560");
    }

    dim3 const grid(tokens);
    dim3 const block(NUM_WARPS * 32);
    using P = PackedFloat2<T>;

    fusedSandwichNormKernel<T, NUM_WARPS, ITERS, HIDDEN_DIM><<<grid, block, 0, stream>>>(
        reinterpret_cast<P const*>(attn_out),
        reinterpret_cast<P const*>(residual_in),
        reinterpret_cast<P const*>(gamma1),
        reinterpret_cast<P const*>(gamma2),
        reinterpret_cast<P*>(updated_residual_out),
        reinterpret_cast<P*>(moe_in_out),
        eps);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        throw std::runtime_error(std::string("FusedSandwichNorm launch failed: ") + cudaGetErrorString(err));
    }
}

// Template explicit instantiations
template void invokeFusedSandwichNormReg<half>(
    half const* attn_out,
    half const* residual_in,
    half const* gamma1,
    half const* gamma2,
    half* updated_residual_out,
    half* moe_in_out,
    int const tokens,
    int const hidden_dim,
    float const eps,
    cudaStream_t stream);

template void invokeFusedSandwichNormReg<__nv_bfloat16>(
    __nv_bfloat16 const* attn_out,
    __nv_bfloat16 const* residual_in,
    __nv_bfloat16 const* gamma1,
    __nv_bfloat16 const* gamma2,
    __nv_bfloat16* updated_residual_out,
    __nv_bfloat16* moe_in_out,
    int const tokens,
    int const hidden_dim,
    float const eps,
    cudaStream_t stream);

template void invokeFusedSandwichNormReg<float>(
    float const* attn_out,
    float const* residual_in,
    float const* gamma1,
    float const* gamma2,
    float* updated_residual_out,
    float* moe_in_out,
    int const tokens,
    int const hidden_dim,
    float const eps,
    cudaStream_t stream);

} // namespace fusions
} // namespace kolibri
