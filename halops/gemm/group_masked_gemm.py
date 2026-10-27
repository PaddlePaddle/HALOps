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

__all__ = ["group_masked_gemm"]


def group_masked_gemm(
    input: torch.Tensor,
    weight: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    masked_m: torch.Tensor,
    output: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    alpha: Optional[torch.Tensor] = None,
    max_m_per_expert: int = 0,
    sm_count: int = -1,
    mode: str = "w8a8-mxfp8",
) -> None:
    """Grouped masked GEMM for MoE on Blackwell with per-expert token masking.

    Tokens are pre-arranged in a padded ``[num_experts * max_m_per_expert, K]``
    layout. ``masked_m`` specifies how many rows are valid per expert, enabling
    early-exit for padded rows.

    Args:
        input: [num_experts * max_m_per_expert, K] CUDA tensor.

            - ``w8a8-mxfp8``: dtype float8_e4m3fn.
            - ``w4a4-*``: dtype uint8 (packed FP4, logical K is size(1)*2).

        weight: [num_experts, N, K] 3D CUDA tensor. Per-expert weights.
        sfa: Scale factor for A, dtype uint8, CUTLASS SfAtom 128x4 swizzled
            layout. Shape: ``[num_experts * max_m_per_expert, sf_k]``, where
            ``sf_k = K // 32`` (w8a8-mxfp8, w4a4-mxfp4) or ``K // 16`` (w4a4-nvfp4).

            - ``w8a8-mxfp8``, ``w4a4-mxfp4``: UE8M0 encoding.
            - ``w4a4-nvfp4``: UE4M3 encoding.

        sfb: Scale factor for B, dtype uint8, same SfAtom layout as sfa,
            shape [num_experts, N, sf_k].
        masked_m: [num_experts] int32. Actual valid token count per expert.
        output: [num_experts * max_m_per_expert, N] bfloat16 output. Written in-place.
        bias: Optional [num_experts, N] bfloat16 bias. Only supported in w8a8-mxfp8 mode.
        alpha: Optional float32 per-expert scale with shape [num_experts].
            Required for ``w4a4-nvfp4``; not accepted for ``w4a4-mxfp4``.
        max_m_per_expert: Maximum tokens per expert (padding bound), must be > 0.
        sm_count: Number of SMs to use (-1 for all available).
        mode: ``"w8a8-mxfp8"``, ``"w4a4-mxfp4"`` or ``"w4a4-nvfp4"``.

    Returns:
        None. The result is written in-place to output.

    Example:
        >>> from halops import group_masked_gemm, group_wise_quant
        >>>
        >>> num_experts, max_m_per_expert, n, k = 4, 128, 256, 256
        >>> m = num_experts * max_m_per_expert
        >>> x_bf16 = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
        >>> w_bf16 = torch.randn(num_experts, n, k, dtype=torch.bfloat16, device="cuda")
        >>> x, sfa = group_wise_quant(x_bf16, mode="w8a8-mxfp8")
        >>> w, sfb = group_wise_quant(w_bf16.view(num_experts * n, k), mode="w8a8-mxfp8")
        >>> w = w.view(num_experts, n, k)
        >>> sfb = sfb.view(num_experts, n, -1)
        >>> # valid rows of each expert's 128-row slot
        >>> masked_m = torch.tensor([128, 64, 0, 32], dtype=torch.int32, device="cuda")
        >>> out = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
        >>> group_masked_gemm(
        ...     x, w, sfa, sfb, masked_m, out,
        ...     max_m_per_expert=max_m_per_expert, mode="w8a8-mxfp8",
        ... )
    """
    halops_kernel.group_masked_gemm(
        input,
        weight,
        sfa,
        sfb,
        masked_m,
        output,
        bias,
        alpha,
        max_m_per_expert,
        sm_count,
        mode,
    )
