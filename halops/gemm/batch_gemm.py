"""
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

from typing import Optional

import torch

import halops.halops_kernel as halops_kernel

__all__ = ["batch_gemm"]


def batch_gemm(
    input: torch.Tensor,
    weight: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    output: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    alpha: Optional[torch.Tensor] = None,
    sm_count: int = -1,
    mode: str = "w8a8-mxfp8",
) -> None:
    """Batch GEMM on Blackwell SM100 tensor cores.

    Computes ``output[b] = input[b] @ weight[b]^T`` for each batch item.
    The batch dimension ``B`` acts as the group (expert) index: batch item
    ``b`` is multiplied only by ``weight[b]``, i.e. there is a one-to-one
    mapping between input batch index and weight batch index (no separate
    ``expert_ids`` tensor is needed). Supports FP8 (W8A8-MXFP8) and both FP4 (W4A4) recipes
    (NVFP4 and MXFP4) via CUTLASS.

    Args:
        input: [B, M, K] CUDA tensor.
            w8a8-mxfp8: dtype float8_e4m3fn.
            w4a4-*: dtype uint8 (packed FP4, logical K is size(2)*2).
        weight: [B, N, K] CUDA tensor. Same dtype/packing rules as input.
        sfa: [B, M, sf_k] scale factor for A, dtype uint8, CUTLASS SfAtom 128x4 swizzled layout.
            sf_k = K // 32 (w8a8-mxfp8, w4a4-mxfp4) or K // 16 (w4a4-nvfp4).
            w8a8-mxfp8, w4a4-mxfp4: UE8M0 encoding.
            w4a4-nvfp4: UE4M3 encoding.
        sfb: [B, N, sf_k] scale factor for B, dtype uint8, same SfAtom layout as sfa.
        output: [B, M, N] pre-allocated output tensor, dtype bfloat16. Written in-place.
        bias: Optional [B, N] bfloat16 bias vector. Only supported in w8a8-mxfp8 mode.
        alpha: Optional float32 per-batch scaling factor with shape [B], one
            value per batch item. Required for ``w4a4-nvfp4``; not accepted for
            ``w4a4-mxfp4``.
        sm_count: Number of SMs to use (-1 for all available).
        mode: ``"w8a8-mxfp8"``, ``"w4a4-mxfp4"`` or ``"w4a4-nvfp4"``.

    Returns:
        None. The result is written in-place to output.

    Example:
        >>> from halops import batch_gemm, group_wise_quant
        >>>
        >>> b, m, n, k = 4, 256, 256, 256
        >>> x_bf16 = torch.randn(b, m, k, dtype=torch.bfloat16, device="cuda")
        >>> w_bf16 = torch.randn(b, n, k, dtype=torch.bfloat16, device="cuda")
        >>> x, sfa = group_wise_quant(x_bf16.view(b * m, k), mode="w8a8-mxfp8")
        >>> w, sfb = group_wise_quant(w_bf16.view(b * n, k), mode="w8a8-mxfp8")
        >>> x = x.view(b, m, k); sfa = sfa.view(b, m, -1)
        >>> w = w.view(b, n, k); sfb = sfb.view(b, n, -1)
        >>> out = torch.empty(b, m, n, dtype=torch.bfloat16, device="cuda")
        >>> batch_gemm(x, w, sfa, sfb, out, mode="w8a8-mxfp8")
    """
    halops_kernel.batch_gemm(
        input, weight, sfa, sfb, output, bias, alpha, sm_count, mode
    )
