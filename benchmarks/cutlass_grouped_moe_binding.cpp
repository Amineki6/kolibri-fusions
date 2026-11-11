#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "tensorrt_llm/kernels/moe/cutlass/include/moe_gemm_kernels.h"

namespace {

using namespace tensorrt_llm::kernels::cutlass_kernels;
using namespace tensorrt_llm::cutlass_extensions;

MoeGemmRunner<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16>& get_moe_gemm_runner() {
    static MoeGemmRunner<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16> runner;
    return runner;
}

} // namespace

torch::Tensor cutlass_grouped_moe_gemm(
    torch::Tensor permuted_act,
    torch::Tensor expert_first_token_offset,
    torch::Tensor fc1_weights_col,
    torch::Tensor fc2_weights_col
) {
    TORCH_CHECK(permuted_act.is_cuda(), "permuted_act must be a CUDA tensor");
    TORCH_CHECK(expert_first_token_offset.is_cuda(), "expert_first_token_offset must be a CUDA tensor");
    TORCH_CHECK(fc1_weights_col.is_cuda(), "fc1_weights_col must be a CUDA tensor");
    TORCH_CHECK(fc2_weights_col.is_cuda(), "fc2_weights_col must be a CUDA tensor");

    TORCH_CHECK(permuted_act.scalar_type() == torch::kBFloat16, "permuted_act must be BFloat16");
    TORCH_CHECK(expert_first_token_offset.scalar_type() == torch::kInt64, "expert_first_token_offset must be Int64");
    TORCH_CHECK(fc1_weights_col.scalar_type() == torch::kBFloat16, "fc1_weights_col must be BFloat16");
    TORCH_CHECK(fc2_weights_col.scalar_type() == torch::kBFloat16, "fc2_weights_col must be BFloat16");

    TORCH_CHECK(permuted_act.is_contiguous(), "permuted_act must be contiguous");
    TORCH_CHECK(expert_first_token_offset.is_contiguous(), "expert_first_token_offset must be contiguous");
    TORCH_CHECK(fc1_weights_col.is_contiguous(), "fc1_weights_col must be contiguous");
    TORCH_CHECK(fc2_weights_col.is_contiguous(), "fc2_weights_col must be contiguous");

    const c10::cuda::OptionalCUDAGuard device_guard(device_of(permuted_act));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    const int64_t expanded_num_rows = permuted_act.size(0);
    const int64_t hidden_size = permuted_act.size(1);
    const int num_experts = static_cast<int>(fc2_weights_col.size(0));
    const int64_t inter_size = fc2_weights_col.size(2);

    auto fc1_act_out = torch::empty({expanded_num_rows, inter_size}, permuted_act.options());
    auto fc2_out = torch::empty({expanded_num_rows, hidden_size}, permuted_act.options());

    if (expanded_num_rows == 0) {
        return fc2_out;
    }

    auto& runner = get_moe_gemm_runner();
    const int64_t* total_tokens_including_expert = expert_first_token_offset.data_ptr<int64_t>() + 1;

    // SM80/SM89 CUTLASS tile configuration used by TensorRT-LLM for Kolibri MoE
    CutlassGemmConfig config(
        CutlassTileConfig::CtaShape32x128x64_WarpShape32x32x64,
        SplitKStyle::NO_SPLIT_K,
        /*split_k_factor=*/1,
        /*stages=*/3
    );
    TmaWarpSpecializedGroupedGemmInput tma_ws_input{};

    // 1. FC1 Grouped GEMM + Fused SwiGLU Activation (1 kernel launch: fused_moe_oss::run_global<Fused_Moe_Kernel_sm80>)
    GroupedGemmInput<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16> fc1_input{
        reinterpret_cast<const __nv_bfloat16*>(permuted_act.data_ptr()),
        total_tokens_including_expert,
        reinterpret_cast<const __nv_bfloat16*>(fc1_weights_col.data_ptr()),
        /*scales=*/nullptr,
        /*zeros=*/nullptr,
        /*biases=*/nullptr,
        reinterpret_cast<__nv_bfloat16*>(fc1_act_out.data_ptr()),
        /*alpha_scales=*/nullptr,
        /*occupancy=*/nullptr,
        ActivationType::Swiglu,
        expanded_num_rows,
        /*n=*/inter_size,
        /*k=*/hidden_size,
        num_experts,
        /*groupwise_quant_group_size=*/0,
        /*bias_is_broadcast=*/true,
        /*use_fused_moe=*/true,
        stream,
        config
    };
    runner.moeGemmBiasAct(fc1_input, tma_ws_input);

    // 2. FC2 Grouped GEMM (1 kernel launch: cutlass::Kernel<cutlass::gemm::kernel::MoeFCGemm>)
    GroupedGemmInput<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16> fc2_input{
        reinterpret_cast<const __nv_bfloat16*>(fc1_act_out.data_ptr()),
        total_tokens_including_expert,
        reinterpret_cast<const __nv_bfloat16*>(fc2_weights_col.data_ptr()),
        /*scales=*/nullptr,
        /*zeros=*/nullptr,
        /*biases=*/nullptr,
        reinterpret_cast<__nv_bfloat16*>(fc2_out.data_ptr()),
        /*alpha_scales=*/nullptr,
        /*occupancy=*/nullptr,
        ActivationType::Identity,
        expanded_num_rows,
        /*n=*/hidden_size,
        /*k=*/inter_size,
        num_experts,
        /*groupwise_quant_group_size=*/0,
        /*bias_is_broadcast=*/false,
        /*use_fused_moe=*/false,
        stream,
        config
    };
    runner.moeGemmBiasAct(fc2_input, tma_ws_input);

    return fc2_out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "cutlass_grouped_moe_gemm",
        &cutlass_grouped_moe_gemm,
        "TensorRT-LLM SM80/SM89 CUTLASS Grouped MoE GEMM (FC1+SwiGLU & FC2)"
    );
}

