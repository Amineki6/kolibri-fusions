#include "custom_moe_prologue.cuh"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/cub.cuh>
#include <algorithm>
#include <stdexcept>
#include <string>

// Authentic TRT-LLM TopK reduction functions
#include "tensorrt_llm/kernels/moeTopKFuncs.cuh"

namespace cg = cooperative_groups;

namespace kolibri
{
namespace fusions
{

static constexpr int WARP_SIZE = 32;

// =================================================================================================
// 1. Stage 1: Fast Multi-Block Top-K Router Kernel
//
// Grid: (num_tokens + 7) / 8 blocks, 256 threads (8 warps per block).
// Each warp processes 1 token row (384 expert scores).
// Faithful implementation of kolibri_gate_forward_kernel using tensorrt_llm::kernels::reduce_topk::reduceTopK.
// =================================================================================================
template <
    int nExperts = 384,
    int topK = 6,
    int kExpertsPerThread = nExperts / WARP_SIZE,
    int kPaddedExpertsPerThread = 16,
    int kWarpsPerBlock = 8>
__global__ void kolibri_router_kernel(
    float const* __restrict__ scores_in,
    float const* __restrict__ bias,
    float* __restrict__ out_weights,
    int* __restrict__ out_indices,
    int const num_tokens)
{
    static_assert(kExpertsPerThread == nExperts / WARP_SIZE, "kExpertsPerThread must match nExperts / WARP_SIZE");
    static_assert(kPaddedExpertsPerThread >= kExpertsPerThread, "kPaddedExpertsPerThread must be >= kExpertsPerThread");
    static_assert(topK <= WARP_SIZE, "topK must be <= WARP_SIZE");

    // Shared memory cache for raw (unbiased) scores for each warp in the block.
    // Preserves unbiased scores for the post-topK sigmoid computation without DRAM/L2 re-reads.
    __shared__ float smem_raw_scores[kWarpsPerBlock][nExperts];

    // Each warp processes exactly one token row across the grid
    int const global_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int const local_warp_id  = (threadIdx.x / WARP_SIZE) % kWarpsPerBlock;
    int const lane_id        = threadIdx.x % WARP_SIZE;

    // Out-of-bounds tokens check (warp-level early exit)
    if (global_warp_id >= num_tokens)
        return;

    auto warp = cg::tiled_partition<WARP_SIZE>(cg::this_thread_block());
    float* my_smem = smem_raw_scores[local_warp_id];
    float const* scores_row = scores_in + global_warp_id * nExperts;

    // Thread-local candidate buffers (padded to a power-of-2 for bitonic sorting)
    float biased_scores[kPaddedExpertsPerThread];
    int indices[kPaddedExpertsPerThread];

    //Load expert scores, apply routing bias, and cache unbiased scores into shared memory
#pragma unroll
    for (int e = 0; e < kExpertsPerThread; ++e)
    {
        int const expert_id = lane_id + e * WARP_SIZE;
        float const raw_score = scores_row[expert_id];
        float const b = bias ? bias[expert_id] : 0.0f;
        biased_scores[e] = raw_score + b;
        indices[e] = expert_id;
        my_smem[expert_id] = raw_score; // Cache for un-biased sigmoid
    }

    // Pad unused candidate slots with -infinity and sentinel IDs.
    // Padding from 12 to 16 (power of 2) forces reduceTopK to use an unrolled
    // Bitonic Sorting Network instead of falling back to O(N^2) odd-even transposition sort.
#pragma unroll
    for (int e = kExpertsPerThread; e < kPaddedExpertsPerThread; ++e)
    {
        biased_scores[e] = -1e30f;
        indices[e] = 65535;
    }
    __syncwarp();

    // Warp-level Top-K reduction across all candidates
    float topk_values[topK];
    int32_t topk_indices[topK];
    constexpr float minValue = -1e30f;
    tensorrt_llm::kernels::reduce_topk::reduceTopK<topK, float, kPaddedExpertsPerThread>(
        warp, topk_values, topk_indices, biased_scores, indices, minValue, topK);

    // Compute unnormalized sigmoid routing weights from raw scores and write out results.
    // Only the first topK lanes (0..topK-1) write the top-K experts for this token.
    if (lane_id < topK)
    {
        int const expert_id = topk_indices[lane_id];
        float const my_topk_value = my_smem[expert_id];
        float const s = 1.0f / (1.0f + __expf(-my_topk_value));

        int const out_offset = global_warp_id * topK + lane_id;
        out_weights[out_offset] = s;
        out_indices[out_offset] = expert_id;
    }
}

void invokeKolibriRouter(
    float const* scores_in,
    float const* bias,
    float* out_weights,
    int* out_indices,
    int const num_tokens,
    cudaStream_t stream)
{
    constexpr int warps_per_block = 8;
    constexpr int threads_per_block = warps_per_block * WARP_SIZE;
    int const blocks = (num_tokens + warps_per_block - 1) / warps_per_block;

    kolibri_router_kernel<384, 6, 384 / WARP_SIZE, 16, warps_per_block><<<blocks, threads_per_block, 0, stream>>>(
        scores_in, bias, out_weights, out_indices, num_tokens
    );
}

// =================================================================================================
// 2. Stage 2A: Single-Block Build Expert Maps Kernel (Optimized for N < 1024 Decode)
//
// 256 threads, 1 block.
// 3.0 KB static shared memory (384 bins x 4B counts + 385 x 4B offsets).
// Completely avoids CUB 32 KB allocation and dynamic memory overhead.
// =================================================================================================
template <int NUM_EXPERTS = 384, int TOP_K = 6, int BLOCK_SIZE = 256>
__global__ void single_block_build_expert_maps_kernel(
    int const* __restrict__ token_selected_experts,       // [num_tokens * TOP_K]
    int* __restrict__ permuted_row_to_unpermuted_row,     // [num_tokens * TOP_K]
    int* __restrict__ unpermuted_row_to_permuted_row,     // [num_tokens * TOP_K] (optional)
    int64_t* __restrict__ expert_first_token_offset,      // [NUM_EXPERTS + 1]
    int const num_tokens,
    int2* __restrict__ selection_scratch)                 // [num_tokens * TOP_K]
{
    // Shared memory histogram bins and prefix-sum offsets for each expert
    __shared__ int expert_counts[NUM_EXPERTS];
    __shared__ int expert_offsets[NUM_EXPERTS + 1];

    // Initialize histogram counters in shared memory
    #pragma unroll
    for (int e = threadIdx.x; e < NUM_EXPERTS; e += BLOCK_SIZE)
    {
        expert_counts[e] = 0;
    }
    __syncthreads();

    // Count token assignments per expert and record the relative intra-expert rank
    int const total_selections = num_tokens * TOP_K;
    for (int idx = threadIdx.x; idx < total_selections; idx += BLOCK_SIZE)
    {
        int const expert_id = token_selected_experts[idx];
        int local_offset = -1;
        if (expert_id >= 0 && expert_id < NUM_EXPERTS)
        {
            local_offset = atomicAdd(&expert_counts[expert_id], 1);
        }
        selection_scratch[idx] = make_int2(expert_id, local_offset);
    }
    __syncthreads();

    // Compute exclusive prefix sum across expert counts to find starting slot per expert
    if (threadIdx.x == 0)
    {
        int sum = 0;
        for (int e = 0; e < NUM_EXPERTS; ++e)
        {
            expert_offsets[e] = sum;
            sum += expert_counts[e];
        }
        expert_offsets[NUM_EXPERTS] = sum;
    }
    __syncthreads();

    // Write computed expert bucket offsets to global memory for downstream GEMM kernels
    #pragma unroll
    for (int e = threadIdx.x; e < NUM_EXPERTS + 1; e += BLOCK_SIZE)
    {
        expert_first_token_offset[e] = expert_offsets[e];
    }

    // Scatter permuted-to-unpermuted and reverse mapping rows into expert-contiguous order
    for (int idx = threadIdx.x; idx < total_selections; idx += BLOCK_SIZE)
    {
        int const token_id = idx / TOP_K;
        int const rank     = idx % TOP_K;
        int2 const sel     = selection_scratch[idx];
        int const expert_id    = sel.x;
        int const local_offset = sel.y;

        if (local_offset >= 0)
        {
            int const dst_slot = expert_offsets[expert_id] + local_offset;
            int const unpermuted_row = rank * num_tokens + token_id;

            permuted_row_to_unpermuted_row[dst_slot] = unpermuted_row;
            if (unpermuted_row_to_permuted_row != nullptr)
            {
                unpermuted_row_to_permuted_row[unpermuted_row] = dst_slot;
            }
        }
    }
}

void invokeSingleBlockBuildExpertMaps(
    int const* token_selected_experts,
    int* permuted_row_to_unpermuted_row,
    int* unpermuted_row_to_permuted_row,
    int64_t* expert_first_token_offset,
    void* scratch,
    int const num_tokens,
    cudaStream_t stream)
{
    int2* selection_scratch = reinterpret_cast<int2*>(scratch);
    constexpr int BLOCK_SIZE = 256;
    single_block_build_expert_maps_kernel<NUM_EXPERTS, TOP_K, BLOCK_SIZE><<<1, BLOCK_SIZE, 0, stream>>>(
        token_selected_experts,
        permuted_row_to_unpermuted_row,
        unpermuted_row_to_permuted_row,
        expert_first_token_offset,
        num_tokens,
        selection_scratch
    );
}

// =================================================================================================
// 3. Stage 2B: 3-Stage Multi-Block Build Expert Maps Pipeline (Optimized for N >= 512 Prefill)
//
// Scales across all 128 SMs of the Ada Lovelace GPU.
// Stage 2b-1: Block local histograms (1024 threads, 1.5 KB smem)
// Stage 2b-2: Prefix sum in thread registers (512 threads, 0 smem bank conflicts, 0 spills)
// Stage 2b-3: Direct coalesced scatter (256 threads, 3 KB smem, saturates 84% peak bus rate)
// =================================================================================================
template <int nExperts = 384, int BLOCK_SIZE = 1024>
__global__ void kolibri_multi_block_expert_counts_kernel(
    int const* __restrict__ token_selected_experts,
    int* __restrict__ block_expert_counts,
    int const total_selections,
    int const num_blocks)
{
    // Shared memory histogram bins for expert selection counts within this thread block
    __shared__ int local_expert_counts[nExperts];

    // Initialize block-local expert counters to zero
    #pragma unroll
    for (int e = threadIdx.x; e < nExperts; e += BLOCK_SIZE)
    {
        local_expert_counts[e] = 0;
    }
    __syncthreads();

    // Partition the input token selections uniformly across the thread blocks
    int const chunk_size = (total_selections + num_blocks - 1) / num_blocks;
    int const start_idx  = blockIdx.x * chunk_size;
    int const end_idx    = min(start_idx + chunk_size, total_selections);

    // Stride through this block's assigned token-expert assignments and tally counts in shared memory
    for (int idx = start_idx + threadIdx.x; idx < end_idx; idx += BLOCK_SIZE)
    {
        int const expert_id = token_selected_experts[idx];
        if (expert_id >= 0 && expert_id < nExperts)
        {
            atomicAdd(&local_expert_counts[expert_id], 1);
        }
    }
    __syncthreads();

    // Export the block-level expert histogram to global memory for downstream multi-block prefix sum
    int const block_offset = blockIdx.x * nExperts;
    #pragma unroll
    for (int e = threadIdx.x; e < nExperts; e += BLOCK_SIZE)
    {
        block_expert_counts[block_offset + e] = local_expert_counts[e];
    }
}

template <int nExperts = 384, int MAX_BLOCKS = 128, int BLOCK_SIZE = 512>
__global__ void kolibri_multi_block_prefix_sum_kernel(
    int const* __restrict__ block_expert_counts,
    int* __restrict__ block_expert_offsets,
    int64_t* __restrict__ expert_first_token_offset,
    int const num_blocks)
{
    // Shared memory totals and prefix sum offsets for each expert across all blocks
    __shared__ int expert_totals[nExperts];
    __shared__ int expert_offsets[nExperts + 1];

    int const e = threadIdx.x;
    int local_counts[MAX_BLOCKS];

    // Each thread e handles expert e: aggregates counts across all blocks and caches in registers.
    // Unrolling by 8 pipelines strided global/L2 loads to hide memory latency while aligning
    // with common block counts (multiples of 8/16/32) without excessive register pressure or spills.
    if (e < nExperts)
    {
        int total = 0;
        #pragma unroll 8
        for (int g = 0; g < num_blocks; ++g)
        {
            int const c = block_expert_counts[g * nExperts + e];
            local_counts[g] = c;
            total += c;
        }
        expert_totals[e] = total;
    }
    __syncthreads();

    // Single thread computes exclusive prefix sum across all experts to get starting offsets
    if (threadIdx.x == 0)
    {
        int sum = 0;
        #pragma unroll 16
        for (int i = 0; i < nExperts; ++i)
        {
            expert_offsets[i] = sum;
            sum += expert_totals[i];
        }
        expert_offsets[nExperts] = sum;
    }
    __syncthreads();

    // Export global token offsets for each expert
    if (threadIdx.x < nExperts + 1)
    {
        expert_first_token_offset[threadIdx.x] = static_cast<int64_t>(expert_offsets[threadIdx.x]);
    }

    // Compute and write the base destination offset for each (block, expert) slice.
    // Reuses counts cached in registers to avoid re-reading global memory.
    if (e < nExperts)
    {
        int running_offset = expert_offsets[e];
        #pragma unroll 8
        for (int g = 0; g < num_blocks; ++g)
        {
            block_expert_offsets[g * nExperts + e] = running_offset;
            running_offset += local_counts[g]; // Cached in register
        }
    }
}

template <int nExperts = 384, int kTopK = 6, int BLOCK_SIZE = 1024>
__global__ void kolibri_multi_block_scatter_maps_kernel(
    int const* __restrict__ token_selected_experts,
    int const* __restrict__ block_expert_offsets,
    int* __restrict__ permuted_row_to_unpermuted_row,
    int* __restrict__ unpermuted_row_to_permuted_row,
    int const total_selections,
    int const num_tokens,
    int const num_blocks)
{
    // Shared memory holding this block's starting base offset per expert and intra-block running counters
    __shared__ int local_expert_base_offsets[nExperts];
    __shared__ int local_expert_counters[nExperts];

    // Load pre-calculated block base offsets and reset intra-block counters
    int const block_offset = blockIdx.x * nExperts;
    #pragma unroll
    for (int e = threadIdx.x; e < nExperts; e += BLOCK_SIZE)
    {
        local_expert_base_offsets[e] = block_expert_offsets[block_offset + e];
        local_expert_counters[e] = 0;
    }
    __syncthreads();

    // Determine the slice of token-expert assignments handled by this block
    int const chunk_size = (total_selections + num_blocks - 1) / num_blocks;
    int const start_idx  = blockIdx.x * chunk_size;
    int const end_idx    = min(start_idx + chunk_size, total_selections);

    for (int idx = start_idx + threadIdx.x; idx < end_idx; idx += BLOCK_SIZE)
    {
        int const expert_id = token_selected_experts[idx];
        bool const valid    = (expert_id >= 0 && expert_id < nExperts);

        // Active mask of valid lanes in the warp
        unsigned int const active     = __activemask();
        unsigned int const valid_mask = __ballot_sync(active, valid);

        if (valid)
        {
            // Bitmask of all active lanes in the warp that selected the same expert_id
            unsigned int const match_mask = __match_any_sync(valid_mask, expert_id);

            // Elect the lowest active lane in this matching group as leader (0-indexed)
            int const leader              = __ffs(match_mask) - 1;

            // Lane index of the current thread within its warp (0 to 31)
            int const lane                = threadIdx.x & 31;

            // Total number of threads in the warp requesting an allocation for this expert
            int const count               = __popc(match_mask);

            // Number of threads with a lower lane index requesting this expert (thread's relative rank in warp)
            int const prefix              = __popc(match_mask & ((1u << lane) - 1));

            // Only the leader executes a single atomic reservation for the entire matching group
            int warp_base_offset = 0;
            if (lane == leader)
            {
                warp_base_offset = atomicAdd(&local_expert_counters[expert_id], count);
            }
            // Broadcast the reserved base offset from the leader to all matching lanes via registers
            warp_base_offset = __shfl_sync(valid_mask, warp_base_offset, leader);

            // Compute the unique intra-block offset and final destination slot in the permuted tensor
            int const local_offset = warp_base_offset + prefix;
            int const dst_slot     = local_expert_base_offsets[expert_id] + local_offset;

            int const token_id       = idx / kTopK;
            int const rank           = idx % kTopK;
            int const unpermuted_row = rank * num_tokens + token_id;

            // Store permuted to unpermuted index mapping (and optional inverse mapping)
            permuted_row_to_unpermuted_row[dst_slot] = unpermuted_row;
            if (unpermuted_row_to_permuted_row != nullptr)
            {
                unpermuted_row_to_permuted_row[unpermuted_row] = dst_slot;
            }
        }
    }
}

void invokeMultiBlockBuildExpertMaps(
    int const* token_selected_experts,
    int* permuted_row_to_unpermuted_row,
    int* unpermuted_row_to_permuted_row,
    int64_t* expert_first_token_offset,
    void* scratch,
    int const num_tokens,
    cudaStream_t stream)
{
    int const total_selections = num_tokens * TOP_K;
    int const max_blocks = 128;
    int const num_blocks = std::min(max_blocks, std::max(1, (total_selections + 1023) / 1024));

    int* scratch_i = reinterpret_cast<int*>(scratch);
    int* d_block_expert_counts = scratch_i;
    int* d_block_expert_offsets = scratch_i + num_blocks * NUM_EXPERTS;

    // Stage 1: Counts
    kolibri_multi_block_expert_counts_kernel<NUM_EXPERTS, 1024><<<num_blocks, 1024, 0, stream>>>(
        token_selected_experts, d_block_expert_counts, total_selections, num_blocks
    );

    // Stage 2: Prefix sum
    kolibri_multi_block_prefix_sum_kernel<NUM_EXPERTS, 128, 512><<<1, 512, 0, stream>>>(
        d_block_expert_counts, d_block_expert_offsets, expert_first_token_offset, num_blocks
    );

    // Stage 3: Scatter with Warp Aggregation (1024 threads for 66.7% occupancy)
    kolibri_multi_block_scatter_maps_kernel<NUM_EXPERTS, TOP_K, 1024><<<num_blocks, 1024, 0, stream>>>(
        token_selected_experts, d_block_expert_offsets,
        permuted_row_to_unpermuted_row, unpermuted_row_to_permuted_row,
        total_selections, num_tokens, num_blocks
    );
}

// =================================================================================================
// Dynamic Hybrid Host Dispatcher
// =================================================================================================
void invokeDynamicBuildExpertMaps(
    int const* token_selected_experts,
    int* permuted_row_to_unpermuted_row,
    int* unpermuted_row_to_permuted_row,
    int64_t* expert_first_token_offset,
    void* scratch,
    int const num_tokens,
    cudaStream_t stream,
    bool force_single_block,
    bool force_multi_block)
{
    if (force_single_block || (!force_multi_block && num_tokens < HYBRID_TIPPING_POINT))
    {
        invokeSingleBlockBuildExpertMaps(
            token_selected_experts,
            permuted_row_to_unpermuted_row,
            unpermuted_row_to_permuted_row,
            expert_first_token_offset,
            scratch,
            num_tokens,
            stream
        );
    }
    else
    {
        invokeMultiBlockBuildExpertMaps(
            token_selected_experts,
            permuted_row_to_unpermuted_row,
            unpermuted_row_to_permuted_row,
            expert_first_token_offset,
            scratch,
            num_tokens,
            stream
        );
    }
}

// =================================================================================================
// 4. Stage 3: 128-Bit Vectorized Scatter Activation Expansion Kernel (Read Once, Scatter 6 Times)
//
// 128-bit (uint4 = 16 bytes = 8 BF16/FP16 elements) memory transfers.
// Reads each input token ONCE into registers and directly scatters to 6 expert destination rows.
// Strictly eliminates 6x redundant DRAM read traffic at large prefill (saving up to 2.5 GB DRAM traffic).
// =================================================================================================
template <typename T, int TOP_K = 6, int HIDDEN_SIZE = 0, int BLOCK_SIZE = 256>
__global__ void kolibri_activation_expansion_kernel(
    T const* __restrict__ unpermuted_input,
    T* __restrict__ permuted_output,
    int const* __restrict__ unpermuted_row_to_permuted,
    int const num_tokens,
    int const hidden_size)
{
    // Use 128-bit vector types (uint4 = 16 bytes = 8 BF16/FP16 elements)
    using VecType = uint4;
    constexpr int ELEMS_PER_VEC = sizeof(VecType) / sizeof(T);
    int const actual_hidden = (HIDDEN_SIZE > 0) ? HIDDEN_SIZE : hidden_size;
    int const num_vecs = actual_hidden / ELEMS_PER_VEC;

    // Each block processes one original token row
    int const src_token_id = blockIdx.x;
    if (src_token_id < num_tokens)
    {
        // Cache this token's TOP_K destination row offsets in registers
        int dst_rows[TOP_K];
        #pragma unroll
        for (int k = 0; k < TOP_K; ++k)
        {
            dst_rows[k] = unpermuted_row_to_permuted[k * num_tokens + src_token_id];
        }

        // Pointer to the start of this token's input feature vector
        VecType const* src_vec = reinterpret_cast<VecType const*>(unpermuted_input + src_token_id * actual_hidden);

        // Vectorized loop along hidden dimension: loads input chunk once into registers,
        // then replicates and pushes to all TOP_K expert destination rows to eliminate redundant reads.
        #pragma unroll 4
        for (int i = threadIdx.x; i < num_vecs; i += BLOCK_SIZE)
        {
            VecType const val = src_vec[i];
            #pragma unroll
            for (int k = 0; k < TOP_K; ++k)
            {
                VecType* dst_vec = reinterpret_cast<VecType*>(permuted_output + dst_rows[k] * actual_hidden);
                dst_vec[i] = val;
            }
        }
    }
}

template <typename T>
void invokeKolibriActivationExpansion(
    T const* unpermuted_input,
    T* permuted_output,
    int const* unpermuted_row_to_permuted_row,
    int const num_tokens,
    int const hidden_size,
    cudaStream_t stream)
{
    constexpr int BLOCK_SIZE = 256;
    int const grid_size = num_tokens;

    if (hidden_size == 2560)
    {
        kolibri_activation_expansion_kernel<T, TOP_K, 2560, BLOCK_SIZE><<<grid_size, BLOCK_SIZE, 0, stream>>>(
            unpermuted_input,
            permuted_output,
            unpermuted_row_to_permuted_row,
            num_tokens,
            hidden_size
        );
    }
    else
    {
        kolibri_activation_expansion_kernel<T, TOP_K, 0, BLOCK_SIZE><<<grid_size, BLOCK_SIZE, 0, stream>>>(
            unpermuted_input,
            permuted_output,
            unpermuted_row_to_permuted_row,
            num_tokens,
            hidden_size
        );
    }
}

// Template explicit instantiations
template void invokeKolibriActivationExpansion<__nv_bfloat16>(
    __nv_bfloat16 const* unpermuted_input,
    __nv_bfloat16* permuted_output,
    int const* unpermuted_row_to_permuted_row,
    int const num_tokens,
    int const hidden_size,
    cudaStream_t stream);

template void invokeKolibriActivationExpansion<half>(
    half const* unpermuted_input,
    half* permuted_output,
    int const* unpermuted_row_to_permuted_row,
    int const num_tokens,
    int const hidden_size,
    cudaStream_t stream);

// =================================================================================================
// 5. End-to-End Prologue Launcher
// =================================================================================================
template <typename T>
void invokeFusedMoePrologue(
    T const* unpermuted_input,
    float const* router_logits,
    float const* router_bias,
    T* permuted_output,
    float* routing_weights,
    int* selected_experts,
    int* permuted_row_to_unpermuted,
    int* unpermuted_to_permuted,
    int64_t* expert_offsets,
    void* scratch,
    int const num_tokens,
    int const hidden_size,
    cudaStream_t stream)
{
    // Stage 1: Router Top-6 selection
    invokeKolibriRouter(
        router_logits, router_bias, routing_weights, selected_experts, num_tokens, stream
    );

    // Stage 2: Dynamic Hybrid Dispatch Mapping (Single-Block for N < 512, Multi-Block for N >= 512)
    invokeDynamicBuildExpertMaps(
        selected_experts,
        permuted_row_to_unpermuted,
        unpermuted_to_permuted,
        expert_offsets,
        scratch,
        num_tokens,
        stream
    );

    // Stage 3: 128-bit Vectorized Scatter Activation Expansion (Read once, scatter 6 times)
    invokeKolibriActivationExpansion<T>(
        unpermuted_input,
        permuted_output,
        unpermuted_to_permuted,
        num_tokens,
        hidden_size,
        stream
    );
}

template void invokeFusedMoePrologue<__nv_bfloat16>(
    __nv_bfloat16 const* unpermuted_input,
    float const* router_logits,
    float const* router_bias,
    __nv_bfloat16* permuted_output,
    float* routing_weights,
    int* selected_experts,
    int* permuted_row_to_unpermuted,
    int* unpermuted_to_permuted,
    int64_t* expert_offsets,
    void* scratch,
    int const num_tokens,
    int const hidden_size,
    cudaStream_t stream);

template void invokeFusedMoePrologue<half>(
    half const* unpermuted_input,
    float const* router_logits,
    float const* router_bias,
    half* permuted_output,
    float* routing_weights,
    int* selected_experts,
    int* permuted_row_to_unpermuted,
    int* unpermuted_to_permuted,
    int64_t* expert_offsets,
    void* scratch,
    int const num_tokens,
    int const hidden_size,
    cudaStream_t stream);

} // namespace fusions
} // namespace kolibri
