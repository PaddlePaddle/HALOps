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

[Highlights](#highlights) | [Performance](#performance) | [Hardware and Software Requirements](#hardware-and-software-requirements) | [Quick Start](#quick-start) | [Build Guide](docs/build_guide_EN.md) | [Benchmark Guide](benchmark/README.md#english-quick-reference) | [Contributing](CONTRIBUTING.md#english)

## Highlights

HALOps is designed from the ground up for NVIDIA Blackwell (SM100). Built on CUTLASS / CuTe, it uses tcgen05, TMEM and TMA directly and covers the Attention, MoE and MLA hot paths of LLM inference.

- **GQA attention**: variable-length prefill and paged-KV decode across BF16 / FP8 KV cache, GQA and MTP; split-KV parallelism, adaptive reduction and fused pre/post-processing make full use of SM100 compute and bandwidth in long-context and small-batch decode.
- **Multi-precision GEMM**: FP8 (MXFP8) and FP4 (NVFP4 / MXFP4) block-scaled GEMM for dense, batched and MoE grouped workloads; with ultra-lightweight flexible tile configurations and pre-mapped tile scheduling, it delivers industry-leading performance across all scenarios on SM100.
- **MegaKernel sparse MLA**: high-performance MegaKernel sparse attention for both HCA and SWA layers. It absorbs the many small surrounding ops (projections, normalization, RoPE, KV cache write and compression) into a single kernel, removing kernel scheduling and intermediate memory round trips, and orchestrates tile-level pipelines so the compute and memory traffic of different stages overlap, pushing GPU utilization further.
- **Low precision end to end**: quantization, fused SwiGLU + quantization and expert permutation write the scale layout the GEMMs read, so low-precision data flows between operators without layout conversion.

## Performance
Hardware architecture: SM100.

<table>
  <tr>
    <td width="50%"><img alt="HALOps vs DeepGEMM vs FlashInfer on dense W8A8 GEMM" src="docs/images/dense_w8a8_performance_halops_vs_backends.png"></td>
    <td width="50%"><img alt="HALOps MegaMLA vs multi-stream + PDL on the HCA attention layer" src="docs/images/megamla_hca_performance_halops_vs_baseline.png"></td>
  </tr>
  <tr>
    <td align="center">Figure 1: dense Mxfp8 W8A8 GEMM, HALOps against DeepGEMM and FlashInfer</td>
    <td align="center">Figure 2: HCA attention layer; for small-batch decode, MegaMLA fuses the whole layer into a MegaKernel and beats the multi-kernel implementation chained with multi-stream overlap and PDL</td>
  </tr>
</table>

## Hardware and Software Requirements

| Component | Requirement |
| --- | --- |
| GPU | NVIDIA Blackwell SM100 data center GPU |
| CUDA Toolkit | 13.0 or later, hardware architecture `compute_100f/sm_100f` |
| NVIDIA Driver | R580 or later, compatible with CUDA 13.x |
| Python / PyTorch | Python 3.10+; PyTorch 2.13+ |

See the [NVIDIA CUDA compatibility matrix](https://docs.nvidia.com/deploy/cuda-compatibility/) for driver requirements, and the [Build Guide](docs/build_guide_EN.md) for environment checks, building and troubleshooting.

## Quick Start

### Install from whl

With a CUDA build of PyTorch that matches your local CUDA, install the [prebuilt HALOps package](https://pypi.org/project/halops/):

```bash
python -m pip install halops
```

### Build from Source

To build it yourself or develop HALOps, run from the repository root:

```bash
git submodule update --init --recursive
python -m pip install -r requirements.txt
# install into the current environment
python -m pip install --no-build-isolation --no-deps -v .
# or install in development mode; after changing CUDA/C++ code, rebuild incrementally with bash build.sh
python -m pip install --no-build-isolation --no-deps -e . -v
```

The build target is fixed to `compute_100f/sm_100f`. See the [Build Guide](docs/build_guide_EN.md) for wheel builds, CUDA path configuration and development builds.

### Verify the Installation

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

W4A4 works the same way with `mode="w4a4-mxfp4"` or `mode="w4a4-nvfp4"`; `w4a4-nvfp4` also takes an fp32 scaling factor `alpha`, see the GEMM docstrings.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md#english) for the development setup, code quality and testing requirements, and the pull request checklist.

## Acknowledgement

HALOps is built on [CUTLASS](https://github.com/NVIDIA/cutlass) and [CuTe](https://github.com/NVIDIA/cutlass/tree/main/include/cute). Thanks to the CUTLASS developers for their work.

## License

HALOps is released under the [Apache License 2.0](LICENSE).
