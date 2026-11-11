#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace kolibri
{
namespace fusions
{

// 4-byte packed float container for 32-bit coalesced memory transactions (LDG.E.32 / STG.E.32)
// For BF16/FP16: 2 elements = 4 bytes per thread (coalesces to native 128-byte cache-line per warp)
// For FP32: 1 element = 4 bytes per thread
template <typename T>
struct alignas(4) PackedFloat2;

// Vector specialization for BF16 (2 elements = 4 bytes)
template <>
struct alignas(4) PackedFloat2<__nv_bfloat16>
{
    static constexpr int size = 2;
    __nv_bfloat16 data[2];

    __device__ inline float sum_of_squares() const
    {
        float const f0 = __bfloat162float(data[0]);
        float const f1 = __bfloat162float(data[1]);
        return f0 * f0 + f1 * f1;
    }

    __device__ inline PackedFloat2<__nv_bfloat16> rmsnorm(float s_var, PackedFloat2<__nv_bfloat16> const& gamma) const
    {
        PackedFloat2<__nv_bfloat16> out;
        float const val0 = __bfloat162float(data[0]);
        float const val1 = __bfloat162float(data[1]);
        float const g0   = __bfloat162float(gamma.data[0]);
        float const g1   = __bfloat162float(gamma.data[1]);
        out.data[0]      = __float2bfloat16(val0 * s_var * g0);
        out.data[1]      = __float2bfloat16(val1 * s_var * g1);
        return out;
    }

    __device__ inline PackedFloat2<__nv_bfloat16> add(PackedFloat2<__nv_bfloat16> const& b) const
    {
        PackedFloat2<__nv_bfloat16> out;
        out.data[0] = __float2bfloat16(__bfloat162float(data[0]) + __bfloat162float(b.data[0]));
        out.data[1] = __float2bfloat16(__bfloat162float(data[1]) + __bfloat162float(b.data[1]));
        return out;
    }
};

// Vector specialization for FP16 (2 elements = 4 bytes)
template <>
struct alignas(4) PackedFloat2<half>
{
    static constexpr int size = 2;
    half data[2];

    __device__ inline float sum_of_squares() const
    {
        float const f0 = __half2float(data[0]);
        float const f1 = __half2float(data[1]);
        return f0 * f0 + f1 * f1;
    }

    __device__ inline PackedFloat2<half> rmsnorm(float s_var, PackedFloat2<half> const& gamma) const
    {
        PackedFloat2<half> out;
        float const val0 = __half2float(data[0]);
        float const val1 = __half2float(data[1]);
        float const g0   = __half2float(gamma.data[0]);
        float const g1   = __half2float(gamma.data[1]);
        out.data[0]      = __float2half(val0 * s_var * g0);
        out.data[1]      = __float2half(val1 * s_var * g1);
        return out;
    }

    __device__ inline PackedFloat2<half> add(PackedFloat2<half> const& b) const
    {
        PackedFloat2<half> out;
        out.data[0] = __float2half(__half2float(data[0]) + __half2float(b.data[0]));
        out.data[1] = __float2half(__half2float(data[1]) + __half2float(b.data[1]));
        return out;
    }
};

// Vector specialization for FP32 (1 element = 4 bytes)
template <>
struct alignas(4) PackedFloat2<float>
{
    static constexpr int size = 1;
    float data[1];

    __device__ inline float sum_of_squares() const
    {
        return data[0] * data[0];
    }

    __device__ inline PackedFloat2<float> rmsnorm(float s_var, PackedFloat2<float> const& gamma) const
    {
        PackedFloat2<float> out;
        out.data[0] = data[0] * s_var * gamma.data[0];
        return out;
    }

    __device__ inline PackedFloat2<float> add(PackedFloat2<float> const& b) const
    {
        PackedFloat2<float> out;
        out.data[0] = data[0] + b.data[0];
        return out;
    }
};

// Host launcher function declaration
template <typename T>
void invokeFusedAddPostMoeNorm(
    T const* routed_out,
    T const* shared_out,
    T const* gamma,
    T* moe_out,
    int const tokens,
    int const hidden_dim,
    float const eps,
    cudaStream_t stream);

} // namespace fusions
} // namespace kolibri
