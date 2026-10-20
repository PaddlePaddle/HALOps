<div align="center">

<pre style="font-size: 90%; line-height: 1.1;">
██╗  ██╗ █████╗ ██╗      ██████╗ ██████╗ ███████╗
██║  ██║██╔══██╗██║     ██╔═══██╗██╔══██╗██╔════╝
███████║███████║██║     ██║   ██║██████╔╝███████╗
██╔══██║██╔══██║██║     ██║   ██║██╔═══╝ ╚════██║
██║  ██║██║  ██║███████╗╚██████╔╝██║     ███████║
╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚═╝     ╚══════╝
</pre>

# **H**igh-performance **A**ccelerated **L**LM **Op**erator**s**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?logo=python&logoColor=white)](pyproject.toml)
[![CUDA](https://img.shields.io/badge/CUDA-13.0%2B-76B900.svg?logo=nvidia&logoColor=white)](docs/build_guide.md)
[![GPU](https://img.shields.io/badge/GPU-NVIDIA%20Blackwell%20SM100-76B900.svg)](docs/build_guide.md)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.13%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](pyproject.toml)

</div>

HALOps（High-performance Accelerated LLM Operators）是文心团队开发的大模型推理高性能算子库。算子充分利用 NVIDIA GPU 硬件特性，并配合 PDL、多流 overlap、算子融合与 MegaKernel 等手段压缩 kernel 间的空泡，让计算、访存与通信尽可能满载，服务高吞吐、低时延的推理场景。

[快速开始](#快速开始) | [环境要求](#硬件与软件要求) | [构建指南](docs/build_guide.md) | [Benchmark 指南](benchmark/README.md) | [贡献指南](CONTRIBUTING.md)

## 核心能力

核心 CUDA kernel 基于 CUTLASS 实现，覆盖 Attention、MLA 与 MoE 推理路径。
- **Attention**：变长 BF16 prefill、BF16/FP8 paged-KV decode、GQA/MTP，以及融合 RoPE、RMSNorm 与 KV cache 写入。
- **GEMM**：W8A8 FP8 与 W4A4 FP4 的 dense、batched、MoE masked/contiguous GEMM。
- **MLA**：MLA MegaKernel，单 kernel 融合 Q 投影（含 norm 与 RoPE）、attention 与 KV cache 写入。
- **Quantization**：group-wise quantization、SwiGLU + quantization。

## 支持的算子

### Attention

| Python API | 功能 |
| --- | --- |
| [`encoder_attention`](halops/attention/encoder_attention.py) | 变长 BF16 prefill，支持 GQA 和可选 mask |
| [`decoder_attention`](halops/attention/decoder_attention.py) | BF16 或 block-wise FP8 paged-KV decode，支持 GQA/MTP |
| [`write_cache_kv_encoder`](halops/attention/write_cache_kv_encoder.py) | prefill QKV 处理与 paged cache 写入 |
| [`write_cache_kv_decoder`](halops/attention/write_cache_kv_decoder.py) | decode Q 生成与 KV cache 追加 |
| [`get_qk_token`](halops/attention/get_qk_token.py) | 计算 KV prefix-sum、per-sample KV 长度及 max/total token 统计 |

### GEMM

| Python API | 功能 |
| --- | --- |
| [`dense_gemm`](halops/gemm/dense_gemm.py) | W8A8 FP8 / W4A4 FP4 dense GEMM，BF16 输出 |
| [`batch_gemm`](halops/gemm/batch_gemm.py) | W8A8 / W4A4 batched GEMM |
| [`group_masked_gemm`](halops/gemm/group_masked_gemm.py) | padded MoE grouped GEMM |
| [`group_contiguous_gemm`](halops/gemm/group_contiguous_gemm.py) | contiguous-token MoE grouped GEMM |
| [`prefill_permute_to_masked_gemm`](halops/gemm/prefill_permute_to_masked_gemm.py) | 按 expert 重排 token 并转换 scale 布局，生成 masked GEMM 输入 |

### MLA

| Python API | 功能 |
| --- | --- |
| [`mega_mla`](halops/mla/mega_mla.py) | 融合 MLA MegaKernel，可选 Q 计算与 KV cache 写入 |
| [`mega_mla_plan`](halops/mla/mega_mla_plan.py) | 生成 `mega_mla` 的调度计划；默认由 `mega_mla` 内部调用 |

### Quantization

| Python API | 功能 |
| --- | --- |
| [`group_wise_quant`](halops/gemm/group_wise_quant.py) | BF16/FP16 到 FP8 或 packed FP4 |
| [`fused_swiglu_quant`](halops/gemm/fused_swiglu_quant.py) | SwiGLU 与 FP8/FP4 量化融合 |

## 硬件与软件要求

| 组件 | 要求 |
| --- | --- |
| GPU | NVIDIA Blackwell SM100 数据中心 GPU |
| CUDA Toolkit | 13.0 或更高版本，编译器必须支持 `compute_100f/sm_100f` |
| NVIDIA Driver | 与 CUDA 13.x 兼容的 R580 或更新版本 |
| Python / PyTorch | Python 3.10+；PyTorch 2.13+ |

驱动要求以 [NVIDIA CUDA compatibility matrix](https://docs.nvidia.com/deploy/cuda-compatibility/) 为准；环境检查、编译和故障排查见 [构建指南](docs/build_guide.md)。

## 快速开始

准备虚拟环境并安装与本机 CUDA Toolkit 兼容的 CUDA 版 PyTorch，然后在仓库根目录执行：

```bash
git submodule update --init --recursive
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation --no-deps -v .
```

构建目标固定为 `compute_100f/sm_100f`；wheel 构建、CUDA 路径配置和开发态编译见 [构建指南](docs/build_guide.md)。

安装后可通过一个 W8A8 dense GEMM 简单示例验证量化和计算链路，预期输出为 `torch.Size([128, 256]) torch.bfloat16`。

```python
import torch

from halops import dense_gemm, group_wise_quant

M, N, K = 128, 256, 512
a = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
b = torch.randn((N, K), device="cuda", dtype=torch.bfloat16)

a_fp8, scale_a = group_wise_quant(a, is_fp4=False, use_swizzle=True)
b_fp8, scale_b = group_wise_quant(b, is_fp4=False, use_swizzle=True)
output = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)

dense_gemm(a_fp8, b_fp8, scale_a, scale_b, output, mode="w8a8")
print(output.shape, output.dtype)
```

## 如何贡献

开发环境、代码质量、测试要求和提交检查项见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 许可证

HALOps 使用 [Apache License 2.0](LICENSE)。
