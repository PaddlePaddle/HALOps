English | [简体中文](README_CN.md)

<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/halops-logo-dark.svg">
  <img alt="HALOps" src="docs/images/halops-logo-light.svg" width="50%">
</picture>

# **H**igh-performance **A**ccelerated **L**LM **Op**erator**s**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?logo=python&logoColor=white)](pyproject.toml)
[![CUDA](https://img.shields.io/badge/CUDA-13.0%2B-76B900.svg?logo=nvidia&logoColor=white)](docs/build_guide.md)
[![GPU](https://img.shields.io/badge/GPU-NVIDIA%20Blackwell%20SM100-76B900.svg)](docs/build_guide.md)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.13%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](pyproject.toml)

</div>

HALOps (High-performance Accelerated LLM Operators) is a high-performance operator library for large language model inference, developed by the ERNIE team. The operators are built to exploit NVIDIA GPU hardware features, and use PDL, multi-stream overlap, operator fusion and MegaKernels to remove bubbles between kernels, keeping compute, memory and communication as close to saturation as possible for high-throughput, low-latency inference.

[Highlights](#highlights) | [Performance](#performance) | [Hardware and Software Requirements](#hardware-and-software-requirements) | [Quick Start](#quick-start) | [Documentation](https://paddlepaddle.github.io/HALOps/) | [Build Guide](docs/build_guide.md) | [Benchmark Guide](benchmark/README.md#english-quick-reference) | [Contributing](CONTRIBUTING.md#english)

## Highlights

HALOps is designed from the ground up for NVIDIA Blackwell (SM100). Built on CUTLASS / CuTe, it uses tcgen05, TMEM and TMA directly and covers the Attention, MoE and MLA hot paths of LLM inference.

- **GQA attention**: variable-length prefill and paged-KV decode across BF16 / FP8 KV cache, GQA and speculative decoding; split-KV parallelism, adaptive reduction and fused pre/post-processing make full use of SM100 compute and bandwidth in long-context and small-batch decode.
- **Multi-precision GEMM**: FP8 (MXFP8) and FP4 (NVFP4 / MXFP4) block-scaled GEMM for dense, batched and MoE grouped workloads; with ultra-lightweight flexible tile configurations and pre-mapped tile scheduling, it delivers industry-leading performance in most scenarios on SM100.
- **MegaKernel sparse MLA**: high-performance MegaKernel sparse attention for HCA and SWA, removing kernel scheduling and intermediate memory round trips, with tile-level pipelining that hides compute and memory latency to push GPU utilization further.

## Performance
All results are measured on SM100: MXFP8 dense GEMM outperforms DeepGEMM and FlashInfer in some cases, and MegaMLA consistently outperforms the multi-stream + PDL implementation at small concurrency.

<table>
  <tr>
    <td width="50%"><img alt="HALOps vs DeepGEMM vs FlashInfer on dense W8A8 GEMM" src="docs/images/dense_w8a8_performance_halops_vs_backends.png"></td>
    <td width="50%"><img alt="HALOps MegaMLA vs multi-stream + PDL on the HCA attention layer" src="docs/images/megamla_hca_performance_halops_vs_baseline.png"></td>
  </tr>
  <tr>
    <td align="center"><sub>Figure 1: Dense Mxfp8 W8A8 GEMM performance comparison</sub></td>
    <td align="center"><sub>Figure 2: HCA attention layer performance comparison</sub></td>
  </tr>
</table>

## Hardware and Software Requirements

| Component | Requirement |
| --- | --- |
| GPU | NVIDIA Blackwell SM100 GPU |
| CUDA Toolkit | 13.0 or later, hardware architecture `compute_100f/sm_100f` |
| NVIDIA Driver | R580 or later, compatible with CUDA 13.x |
| Python / PyTorch | Python 3.10+; PyTorch 2.13+ |

See the [NVIDIA CUDA compatibility matrix](https://docs.nvidia.com/deploy/cuda-compatibility/) for driver requirements, and the [Build Guide](docs/build_guide.md) for the full source-build requirements.

## Quick Start

### Install from Wheel

Prebuilt wheels are available for Python 3.12 with CUDA 13.2 on Linux x86_64 and aarch64. With a CUDA 13.2 build of PyTorch installed, install the [prebuilt HALOps package](https://www.paddlepaddle.org.cn/packages/stable/cu132/halops/) directly:

```bash
python -m pip install halops==0.1.0 -f https://www.paddlepaddle.org.cn/packages/stable/cu132/halops/
```

For other Python or CUDA versions, [build from source](#build-from-source).

### Build from Source

To build it yourself, run from the repository root:

```bash
git submodule update --init --recursive
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation --no-deps -v .
```

See the [Build Guide](docs/build_guide.md) for development installs, wheel builds, CUDA path configuration and build parallelism.

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

## Roadmap

- **Complete GEMM coverage**: add W4A8, BF16 and other GEMM types to build a full quantization-type support matrix, so users can pick the trade-off between inference speed and accuracy they need.
- **Higher-performance MegaKernels**: cover more compute-communication fusion scenarios and further overlap compute with memory traffic.
- **High-performance kernels for small workloads**: kernels tuned for small problem sizes that deliver the lowest latency at low concurrency.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md#english) for the development setup, code quality and testing requirements, and the pull request checklist.

## Acknowledgement

HALOps is built on [CUTLASS](https://github.com/NVIDIA/cutlass) and [CuTe](https://github.com/NVIDIA/cutlass/tree/main/include/cute); we thank the CUTLASS developers for their work. We are also grateful to [FlashInfer](https://github.com/flashinfer-ai/flashinfer), [FlashMLA](https://github.com/deepseek-ai/FlashMLA) and [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM), whose prior work on LLM inference kernels inspired many of our designs and served as valuable references for validating them.

## License

HALOps is released under the [Apache License 2.0](LICENSE).
