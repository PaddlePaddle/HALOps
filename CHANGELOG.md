# Changelog

## 0.1.0

First open-source release. / 首个开源版本。

- Attention: variable-length BF16 prefill, BF16/FP8 paged-KV decode (GQA/MTP), and KV cache writes.
  变长 BF16 prefill、BF16/FP8 paged-KV decode（GQA/MTP），以及 KV cache 写入。
- GEMM: W8A8 FP8 and W4A4 FP4 dense, batched, and MoE masked/contiguous GEMM.
  W8A8 FP8 与 W4A4 FP4 的 dense、batched、MoE masked/contiguous GEMM。
- MLA: fused MLA MegaKernel.
  融合 MLA MegaKernel。
- Quantization: group-wise quantization, SwiGLU + quantization.
  group-wise quantization、SwiGLU + quantization。
