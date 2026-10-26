[English](README_EN.md) | 简体中文

<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/halops-logo-dark.svg">
  <img alt="HALOps" src="docs/images/halops-logo-light.svg" width="50%">
</picture>

# **H**igh-performance **A**ccelerated **L**LM **Op**erator**s**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?logo=python&logoColor=white)](pyproject.toml)
[![CUDA](https://img.shields.io/badge/CUDA-13.0%2B-76B900.svg?logo=nvidia&logoColor=white)](docs/build_guide_CN.md)
[![GPU](https://img.shields.io/badge/GPU-NVIDIA%20Blackwell%20SM100-76B900.svg)](docs/build_guide_CN.md)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.13%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](pyproject.toml)

</div>

HALOps（High-performance Accelerated LLM Operators）是文心团队开发的大模型推理高性能算子库。算子充分利用 NVIDIA GPU 硬件特性，并配合 PDL、多流 overlap、算子融合与 MegaKernel 等手段压缩 kernel 间的空泡，让计算、访存与通信尽可能满载，服务高吞吐、低时延的推理场景。

[核心能力](#核心能力) | [性能](#性能) | [硬件与软件要求](#硬件与软件要求) | [快速开始](#快速开始) | [构建指南](docs/build_guide_CN.md) | [Benchmark 指南](benchmark/README.md) | [贡献指南](CONTRIBUTING.md)

## 核心能力

HALOps 面向 NVIDIA Blackwell（SM100）从零设计，基于 CUTLASS / CuTe 直接使用 tcgen05、TMEM 与 TMA 等新硬件特性，覆盖大模型推理的 Attention、MoE 与 MLA 关键路径。

- **GQA Attention**：支持变长 prefill 与 paged-KV decode，覆盖 BF16 / FP8 KV cache、GQA 与 MTP 等推理场景，结合 split-KV 并行、自适应规约与前后处理算子融合，在长上下文与小 batch decode 下充分释放 SM100 的算力与带宽。
- **多精度 GEMM**：支持 FP8（MXFP8）与 FP4（NVFP4 / MXFP4）block-scaled GEMM，覆盖 dense、batched 与 MoE grouped 场景，结合超轻量开销的灵活 tile 配置与 pre-mapping tile 调度等策略，在 SM100 架构上取得了全场景业界领先的性能。
- **巨内核稀疏 MLA**：支持 HCA 与 SWA 两类层的高性能巨内核（MegaKernel）稀疏 attention，将投影、归一化、RoPE、KV cache 写入与压缩等大量零散算子吸收进单个 kernel，消除 kernel 调度与中间结果的显存往返开销，并通过 tile 级流水编排让各阶段的计算与访存相互掩盖，进一步提升 GPU 算力利用率。
- **端到端低精度**：量化、SwiGLU 融合量化与 expert 重排等算子直接输出 GEMM 所需的 scale 布局，低精度数据在算子之间流转无需额外的布局转换。

## 性能
硬件架构：SM100。

<table>
  <tr>
    <td width="50%"><img alt="HALOps vs DeepGEMM vs FlashInfer on dense W8A8 GEMM" src="docs/images/dense_w8a8_performance_halops_vs_backends.png"></td>
    <td width="50%"><img alt="HALOps MegaMLA vs multi-stream + PDL on the HCA attention layer" src="docs/images/megamla_hca_performance_halops_vs_baseline.png"></td>
  </tr>
  <tr>
    <td align="center">图 1：Dense Mxfp8 W8A8 GEMM，HALOps 对比 DeepGEMM 与 FlashInfer</td>
    <td align="center">图 2：HCA attention 层，小 batch decode 下 MegaMLA 以 MegaKernel 融合整层，单层时延优于多流 + PDL 串联多个 kernel 的实现</td>
  </tr>
</table>

## 硬件与软件要求

| 组件 | 要求 |
| --- | --- |
| GPU | NVIDIA Blackwell SM100 GPU |
| CUDA Toolkit | 13.0 或更高版本，硬件架构 `compute_100f/sm_100f` |
| NVIDIA Driver | 与 CUDA 13.x 兼容的 R580 或更新版本 |
| Python / PyTorch | Python 3.10+；PyTorch 2.13+ |

驱动要求以 [NVIDIA CUDA compatibility matrix](https://docs.nvidia.com/deploy/cuda-compatibility/) 为准；环境检查、编译和故障排查见 [构建指南](docs/build_guide_CN.md)。

## 快速开始

### 从 whl 安装

准备好与本机 CUDA 兼容的 CUDA 版 PyTorch 后，直接安装 [HALOps 预编译包](https://pypi.org/project/halops/)：

```bash
python -m pip install halops
```

### 从源码构建

需要自行编译或参与开发时，在仓库根目录执行：

```bash
git submodule update --init --recursive
python -m pip install -r requirements.txt
# 安装到当前环境
python -m pip install --no-build-isolation --no-deps -v .
# 或以开发模式安装，修改 CUDA/C++ 代码后用 bash build.sh 增量编译
python -m pip install --no-build-isolation --no-deps -e . -v
```

构建目标固定为 `compute_100f/sm_100f`；wheel 构建、CUDA 路径配置和开发态编译见 [构建指南](docs/build_guide_CN.md)。

### 验证安装

安装后可通过一个 W8A8-MXFP8 dense GEMM 简单示例验证量化和计算链路，预期输出为 `torch.Size([128, 256]) torch.bfloat16`。

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

W4A4 的用法相同，将 `mode` 换成 `"w4a4-mxfp4"` 或 `"w4a4-nvfp4"` 即可；`w4a4-nvfp4` 还需额外传入 fp32 缩放因子 `alpha`，详见各 GEMM 接口的 docstring。

## 如何贡献

开发环境、代码质量、测试要求和提交检查项见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 致谢

HALOps 基于 [CUTLASS](https://github.com/NVIDIA/cutlass) 与 [CuTe](https://github.com/NVIDIA/cutlass/tree/main/include/cute) 构建，感谢 CUTLASS 开发者的工作。

## 许可证

HALOps 使用 [Apache License 2.0](LICENSE)。
