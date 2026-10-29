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

__all__ = ["group_contiguous_gemm"]


def group_contiguous_gemm(
    input: torch.Tensor,
    weight: torch.Tensor,
    sfa: Optional[torch.Tensor],
    sfb: Optional[torch.Tensor],
    expert_ids: torch.Tensor,
    output: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    alpha: Optional[torch.Tensor] = None,
    sm_count: int = -1,
    mode: str = "w8a8-mxfp8",
) -> None:
    """Grouped contiguous GEMM for MoE on Blackwell.

    Unlike masked GEMM, tokens from different experts are packed contiguously
    (no per-expert padding). ``expert_ids`` maps each token row to its
    corresponding expert weight.

    Args:
        input: [M, K] CUDA tensor. All tokens packed contiguously.

            - ``w8a8-mxfp8``: dtype float8_e4m3fn.
            - ``w4a4-*``: dtype uint8 (packed FP4, logical K is size(1)*2).

        weight: [num_experts, N, K] 3D CUDA tensor. Per-expert weights.
        sfa: Scale factor for A, dtype uint8, CUTLASS SfAtom 128x4 swizzled
            layout. Shape: ``[ceil(M / 128) * 128, sf_k]``, where
            ``sf_k = K // 32`` (w8a8-mxfp8, w4a4-mxfp4) or ``K // 16`` (w4a4-nvfp4).

            - ``w8a8-mxfp8``, ``w4a4-mxfp4``: UE8M0 encoding.
            - ``w4a4-nvfp4``: UE4M3 encoding.

        sfb: Scale factor for B, dtype uint8, same SfAtom layout as sfa.
            shape: [num_experts, N, K // 32] (w8a8-mxfp8, w4a4-mxfp4) or
            [num_experts, N, K // 16] (w4a4-nvfp4).
        expert_ids: [M] int32. Maps each row to an expert index in
            [0, num_experts). Must be sorted in non-decreasing order.
        output: [M, N] bfloat16 output. Written in-place.
        bias: Optional [num_experts, N] bfloat16 bias. Only supported in
            w8a8-mxfp8 mode.
        alpha: Optional float32 per-expert scale with shape [num_experts].
            Required for ``w4a4-nvfp4``; not accepted for ``w4a4-mxfp4``.
        sm_count: Number of SMs to use (-1 for all available).
        mode: ``"w8a8-mxfp8"``, ``"w4a4-mxfp4"`` or ``"w4a4-nvfp4"`` for
            quantized GEMM, or ``"bf16"`` for native BF16 GEMM (no
            quantization; pass ``sfa=None, sfb=None``).

    Returns:
        None. The result is written in-place to output.

    Example:
        >>> from halops import group_contiguous_gemm, group_wise_quant
        >>>
        >>> m, n, k = 256, 256, 256
        >>> num_experts = 4
        >>> x_bf16 = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
        >>> w_bf16 = torch.randn(num_experts, n, k, dtype=torch.bfloat16, device="cuda")
        >>> x, sfa = group_wise_quant(x_bf16, mode="w8a8-mxfp8")
        >>> w, sfb = group_wise_quant(w_bf16.view(num_experts * n, k), mode="w8a8-mxfp8")
        >>> w = w.view(num_experts, n, k)
        >>> sfb = sfb.view(num_experts, n, -1)
        >>> # first 128 rows -> expert 0, next 128 -> expert 1
        >>> expert_ids = torch.arange(m, device="cuda", dtype=torch.int32) // 128
        >>> out = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
        >>> group_contiguous_gemm(x, w, sfa, sfb, expert_ids, out, mode="w8a8-mxfp8")
    """
    if sfa is None:
        sfa = torch.empty(0, dtype=torch.uint8, device=input.device)
    if sfb is None:
        sfb = torch.empty(0, dtype=torch.uint8, device=input.device)

    halops_kernel.group_contiguous_gemm(
        input,
        weight,
        sfa,
        sfb,
        expert_ids,
        output,
        bias,
        alpha,
        sm_count,
        mode,
    )
