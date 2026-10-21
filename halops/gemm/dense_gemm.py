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

__all__ = ["dense_gemm"]


def dense_gemm(
    input: torch.Tensor,
    weight: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    output: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    alpha: Optional[torch.Tensor] = None,
    sm_count: int = -1,
    mode: str = "w8a8",
) -> None:
    """Dense GEMM on Blackwell SM100 tensor cores.

    Computes ``output = input @ weight^T + bias``. Supports FP8 (W8A8) and
    FP4 (W4A4) quantized GEMM via CUTLASS.

    Args:
        input: [M, K] CUDA tensor.
            w8a8: dtype float8_e4m3fn.
            w4a4: dtype uint8 (packed FP4, physical shape [M, K // 2]).
        weight: [N, K] CUDA tensor. Same dtype and packing rules as input.
        sfa: Scale factor for A, dtype uint8, CUTLASS SfAtom 128x4 swizzled
            layout.  The layout covers whole 128-row atoms, so the shape is
            ``[ceil(M / 128) * 128, K // 32]`` (w8a8) or
            ``[ceil(M / 128) * 128, K // 16]`` (w4a4) -- exactly what
            ``group_wise_quant`` returns.  The rows past M are read but
            discarded, and are zero so the values stay finite.
            w8a8: UE8M0 encoding (block_size=32).
            w4a4: UE4M3 encoding (block_size=16).
        sfb: Scale factor for B, dtype uint8, in the same ordinary SfAtom
            layout as sfa. Shape [N, K // 32] (w8a8) or [N, K // 16] (w4a4).
        output: [M, N] pre-allocated output tensor, dtype bfloat16. Written in-place.
        bias: Optional [N] bfloat16 bias vector. Only supported in w8a8 mode.
        alpha: Optional float32 scaling factor with shape [1]. Required for w4a4 mode.
        sm_count: Number of SMs to use (-1 for all available).
        mode: ``"w8a8"`` for FP8 E4M3 GEMM; ``"w4a4"`` for FP4 E2M1 GEMM.

    Returns:
        None. The result is written in-place to output.

    Example:
        >>> from halops import dense_gemm, group_wise_quant
        >>>
        >>> m = n = k = 256
        >>> x_bf16 = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
        >>> w_bf16 = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
        >>> x, sfa = group_wise_quant(x_bf16, is_fp4=False)
        >>> w, sfb = group_wise_quant(w_bf16, is_fp4=False)
        >>> out = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
        >>> dense_gemm(x, w, sfa, sfb, out, mode="w8a8")
    """
    halops_kernel.dense_gemm(
        input,
        weight,
        sfa,
        sfb,
        output,
        bias,
        alpha,
        sm_count,
        mode,
    )
