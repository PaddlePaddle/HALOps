English | [简体中文](README_CN.md)

<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/halops-logo-dark.svg">
  <img alt="HALOps" src="docs/images/halops-logo-light.svg" width="50%">
</picture>

# **H**igh-performance **A**ccelerated **L**LM **Op**erator**s**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?logo=python&logoColor=white)](pyproject.toml)
[![CUDA](https://img.shields.io/badge/CUDA-13.0%2B-76B900.svg?logo=nvidia&logoColor=white)](docs/build_guide_EN.md)
[![GPU](https://img.shields.io/badge/GPU-NVIDIA%20Blackwell%20SM100-76B900.svg)](docs/build_guide_EN.md)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.13%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](pyproject.toml)

</div>

HALOps (High-performance Accelerated LLM Operators) is a high-performance operator library for large language model inference, developed by the ERNIE team. The operators are built to exploit NVIDIA GPU hardware features, and use PDL, multi-stream overlap, operator fusion and MegaKernels to remove bubbles between kernels, keeping compute, memory and communication as close to saturation as possible for high-throughput, low-latency inference.

[Quick Start](#quick-start) | [Requirements](#hardware-and-software-requirements) | [Build Guide](docs/build_guide_EN.md) | [Benchmark Guide](benchmark/README.md#english-quick-reference) | [Contributing](CONTRIBUTING.md)

## Highlights

The core CUDA kernels are built on CUTLASS and cover the Attention, MLA and MoE inference paths.
- **Attention**: variable-length BF16 prefill, BF16/FP8 paged-KV decode, GQA/MTP, and fused RoPE, RMSNorm and KV cache writes.
- **GEMM**: W8A8-MXFP8 FP8 and W4A4 FP4 (NVFP4 / MXFP4) dense, batched, and MoE masked/contiguous GEMM.
- **MLA**: an MLA MegaKernel that fuses the Q projection (including norm and RoPE), attention and the KV cache write into a single kernel.
- **Quantization**: group-wise quantization (FP8 / NVFP4 / MXFP4), SwiGLU + quantization.

## Supported Operators

### Attention

| Python API | Description |
| --- | --- |
| [`encoder_attention`](halops/attention/encoder_attention.py) | Variable-length BF16 prefill with GQA and an optional mask |
| [`decoder_attention`](halops/attention/decoder_attention.py) | BF16 or block-wise FP8 paged-KV decode with GQA/MTP |
| [`write_cache_kv_encoder`](halops/attention/write_cache_kv_encoder.py) | Prefill QKV processing and paged cache write |
| [`write_cache_kv_decoder`](halops/attention/write_cache_kv_decoder.py) | Decode Q generation and KV cache append |
| [`get_qk_token`](halops/attention/get_qk_token.py) | KV prefix sums, per-sample KV lengths, and max/total token counts |

### GEMM

| Python API | Description |
| --- | --- |
| [`dense_gemm`](halops/gemm/dense_gemm.py) | W8A8-MXFP8 FP8 / W4A4 FP4 (NVFP4, MXFP4) dense GEMM with BF16 output |
| [`batch_gemm`](halops/gemm/batch_gemm.py) | W8A8-MXFP8 / W4A4 (NVFP4, MXFP4) batched GEMM |
| [`group_masked_gemm`](halops/gemm/group_masked_gemm.py) | Padded MoE grouped GEMM, W8A8-MXFP8 / W4A4 (NVFP4, MXFP4) |
| [`group_contiguous_gemm`](halops/gemm/group_contiguous_gemm.py) | Contiguous-token MoE grouped GEMM, W8A8-MXFP8 / W4A4 (NVFP4, MXFP4) |
| [`prefill_permute_to_masked_gemm`](halops/gemm/prefill_permute_to_masked_gemm.py) | Permute tokens by expert and convert the scale layout to build masked GEMM inputs |

### MLA

| Python API | Description |
| --- | --- |
| [`mega_mla`](halops/mla/mega_mla.py) | Fused MLA MegaKernel with optional Q computation and KV cache write |
| [`mega_mla_plan`](halops/mla/mega_mla_plan.py) | Builds the schedule for `mega_mla`; called internally by `mega_mla` by default |

### Quantization

| Python API | Description |
| --- | --- |
| [`group_wise_quant`](halops/gemm/group_wise_quant.py) | BF16/FP16 to FP8 (UE8M0, group=32), NVFP4 (UE4M3, group=16) or MXFP4 (UE8M0, group=32) |
| [`fused_swiglu_quant`](halops/gemm/fused_swiglu_quant.py) | SwiGLU fused with FP8/FP4 quantization |

## Hardware and Software Requirements

| Component | Requirement |
| --- | --- |
| GPU | NVIDIA Blackwell SM100 data center GPU |
| CUDA Toolkit | 13.0 or later; the compiler must support `compute_100f/sm_100f` |
| NVIDIA Driver | R580 or later, compatible with CUDA 13.x |
| Python / PyTorch | Python 3.10+; PyTorch 2.13+ |

See the [NVIDIA CUDA compatibility matrix](https://docs.nvidia.com/deploy/cuda-compatibility/) for driver requirements, and the [Build Guide](docs/build_guide_EN.md) for environment checks, building and troubleshooting.

## Quick Start

Set up a virtual environment with a CUDA build of PyTorch that matches your local CUDA Toolkit, then run from the repository root:

```bash
git submodule update --init --recursive
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation --no-deps -v .
```

The build target is fixed to `compute_100f/sm_100f`. See the [Build Guide](docs/build_guide_EN.md) for wheel builds, CUDA path configuration and development builds.

After installation, the W8A8-MXFP8 dense GEMM example below checks the quantization and GEMM path. It should print `torch.Size([128, 256]) torch.bfloat16`.

```python
import torch

from halops import dense_gemm, group_wise_quant

M, N, K = 128, 256, 512
a = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
b = torch.randn((N, K), device="cuda", dtype=torch.bfloat16)

a_fp8, scale_a = group_wise_quant(a, mode="w8a8-mxfp8", use_swizzle=True)
b_fp8, scale_b = group_wise_quant(b, mode="w8a8-mxfp8", use_swizzle=True)
output = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)

dense_gemm(a_fp8, b_fp8, scale_a, scale_b, output, mode="w8a8-mxfp8")
print(output.shape, output.dtype)
```

W4A4 works the same way: pass `mode="w4a4-mxfp4"` to both `group_wise_quant` and the GEMM (FP4 E2M1 data with one UE8M0 scale per 32 elements), or `mode="w4a4-nvfp4"` (one UE4M3 scale per 16 elements), which additionally takes the per-expert global `alpha`.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development setup, code quality and testing requirements, and the pull request checklist.

## Acknowledgement

HALOps is built on [CUTLASS](https://github.com/NVIDIA/cutlass) and [CuTe](https://github.com/NVIDIA/cutlass/tree/main/include/cute). Thanks to the CUTLASS developers for their work.

## License

HALOps is released under the [Apache License 2.0](LICENSE).
