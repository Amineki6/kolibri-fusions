import os
import sys
import torch
import tensorrt_llm
from torch.utils.cpp_extension import load

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

python_bin = os.path.dirname(sys.executable)
cuda_bin = "/usr/local/cuda/bin"
os.environ["PATH"] = f"{python_bin}:{cuda_bin}:{os.environ.get('PATH', '')}"

TRT_LLM_CPP = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "TensorRT-LLM", "cpp"))
TRT_LLM_PACKAGE_DIR = os.path.dirname(os.path.abspath(tensorrt_llm.__file__))
TRT_LLM_LIBS_DIR = os.path.join(TRT_LLM_PACKAGE_DIR, "libs")

# Locate CUTLASS headers bundled in the Python environment
SITE_PACKAGES_DIR = os.path.dirname(TRT_LLM_PACKAGE_DIR)
CUTLASS_INCLUDE_CANDIDATES = [
    os.path.join(SITE_PACKAGES_DIR, "flashinfer", "data", "cutlass", "include"),
    os.path.join(TRT_LLM_PACKAGE_DIR, "include", "trtllm_gen_kernels", "fmha", "cutlass", "include"),
    os.path.join(TRT_LLM_PACKAGE_DIR, "deep_gemm", "include"),
]
CUTLASS_INCLUDE = next((p for p in CUTLASS_INCLUDE_CANDIDATES if os.path.isdir(p)), CUTLASS_INCLUDE_CANDIDATES[0])

_cutlass_moe_ops = None


def get_cutlass_grouped_moe_ops():
    global _cutlass_moe_ops
    if _cutlass_moe_ops is None:
        _cutlass_moe_ops = load(
            name="kolibri_trtllm_cutlass_grouped_moe_ops",
            sources=[
                os.path.join(CURRENT_DIR, "cutlass_grouped_moe_binding.cpp"),
            ],
            extra_cflags=[
                "-O3",
                "-std=c++20",
                "-DENABLE_BF16=1",
                "-DENABLE_FP8=1",
                "-DENABLE_FP4=1",
            ],
            extra_include_paths=[
                CURRENT_DIR,
                TRT_LLM_CPP,
                os.path.join(TRT_LLM_CPP, "include"),
                CUTLASS_INCLUDE,
                "/usr/local/cuda/include",
            ],
            extra_ldflags=[
                f"-L{TRT_LLM_LIBS_DIR}",
                f"-Wl,-rpath,{TRT_LLM_LIBS_DIR}",
                "-lth_common",
            ],
            verbose=False,
        )
    return _cutlass_moe_ops


def prepare_cutlass_moe_weights_inplace(layer_dict: dict, inter_size: int = 512) -> dict:
    """
    Convert `w13` ([E, H, 2*I_MOE] RowMajor: [gate, up]) and `w2` ([E, I_MOE, H] RowMajor)
    in-place into the ColumnMajor weight layouts required by TensorRT-LLM's CUTLASS MoE kernels:
      - `w31_col`: [E, 2*I_MOE, H] where `[:, :I_MOE, :]` is `w3` (up_proj) and `[:, I_MOE:, :]` is `w1` (gate_proj)
      - `w2_col`:  [E, H, I_MOE]
    Frees the original `w13` and `w2` tensors so total VRAM remains unchanged (~6.08 GB for 2 blocks).
    """
    if "w31_col" in layer_dict and "w2_col" in layer_dict:
        return layer_dict

    w13 = layer_dict.pop("w13")  # [E, H, 2*I_MOE]
    w2 = layer_dict.pop("w2")    # [E, I_MOE, H]

    w1_col = w13[:, :, :inter_size].transpose(1, 2).contiguous()   # [E, I_MOE, H] (gate)
    w3_col = w13[:, :, inter_size:].transpose(1, 2).contiguous()   # [E, I_MOE, H] (up)
    del w13

    layer_dict["w31_col"] = torch.cat([w3_col, w1_col], dim=1).contiguous()  # [E, 2*I_MOE, H]
    del w1_col, w3_col

    layer_dict["w2_col"] = w2.transpose(1, 2).contiguous()  # [E, H, I_MOE]
    del w2

    return layer_dict


def cutlass_grouped_moe_gemm(
    permuted_act: torch.Tensor,
    expert_first_token_offset: torch.Tensor,
    w31_col: torch.Tensor,
    w2_col: torch.Tensor,
) -> torch.Tensor:
    """
    Execute TensorRT-LLM's SM80/SM89 CUTLASS Grouped MoE GEMM (FC1+SwiGLU and FC2)
    in 2 CUDA kernel launches with zero host-device synchronization.
    """
    ops = get_cutlass_grouped_moe_ops()
    return ops.cutlass_grouped_moe_gemm(permuted_act, expert_first_token_offset, w31_col, w2_col)
