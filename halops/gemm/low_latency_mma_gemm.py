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

__all__ = ["low_latency_mma_gemm"]


def low_latency_mma_gemm(
    input: torch.Tensor,
    weight: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    output: torch.Tensor,
    input2: Optional[torch.Tensor] = None,
    weight2: Optional[torch.Tensor] = None,
    sfa2: Optional[torch.Tensor] = None,
    sfb2: Optional[torch.Tensor] = None,
    output2: Optional[torch.Tensor] = None,
    sm_count: int = -1,
    use_pdl: bool = True,
    use_sigmoid: bool = False,
) -> None:
    """Low-latency W8A8-MXFP8 GEMM for decode-sized M (M <= 16).

    Computes ``output = input @ weight^T``. Two fused variants take a second
    weight:

    - concat-K, with ``input2``: ``output = input @ weight^T + input2 @
      weight2^T``.
    - dual output, with ``output2``: ``output2 = input @ weight2^T`` as well.

    Scales use the plain row-major layout from
    ``group_wise_quant(..., use_swizzle=False)``, not the SfAtom layout
    ``dense_gemm`` takes. Every scale row stride must be a multiple of 16
    bytes: this holds when its K is a multiple of 512; otherwise pass the
    [rows, K // 32] view of a [rows, round_up(K // 32, 16)] buffer. An SfAtom
    sfb is not detected when N % 128 == 0 and gives wrong results.

    Args:
        input: [M, K1] float8_e4m3fn. The whole input is kept in shared
            memory; a K (K1 + K2 with concat-K) too large for it raises an
            error. K must be a multiple of 256.
        weight: [N, K1] float8_e4m3fn, N % 8 == 0.
        sfa: [M, K1 // 32] uint8 UE8M0 scale of input.
        sfb: [N, K1 // 32] uint8 UE8M0 scale of weight. With ``use_pdl`` it is
            read before waiting for the previous kernel, so that kernel must
            not write it.
        output: [M, N] bfloat16, written in-place.
        input2: Optional [M, K2] float8_e4m3fn, enables concat-K. K1 must be
            a multiple of 512.
        weight2: [N, K2] with ``input2``, [N2, K1] with ``output2``;
            N2 % 8 == 0.
        sfa2: [M, K2 // 32] scale of input2.
        sfb2: Scale of weight2.
        output2: Optional [M, N2] bfloat16, enables dual output. Not
            combinable with ``input2``.
        sm_count: Number of SMs to use (-1 for all available).
        use_pdl: Enable Programmatic Dependent Launch (on by default).
        use_sigmoid: Write ``torch.sigmoid`` of the bf16 result instead.
            Not combinable with ``input2`` or ``output2``.

    Returns:
        None. The result is written in-place to output (and output2).

    Example:
        >>> from halops import group_wise_quant, low_latency_mma_gemm
        >>>
        >>> m, n, k = 8, 4096, 4096
        >>> x_bf16 = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
        >>> w_bf16 = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
        >>> x, sfa = group_wise_quant(x_bf16, mode="w8a8-mxfp8", use_swizzle=False)
        >>> w, sfb = group_wise_quant(w_bf16, mode="w8a8-mxfp8", use_swizzle=False)
        >>> out = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
        >>> low_latency_mma_gemm(x, w, sfa, sfb, out)
    """
    halops_kernel.low_latency_mma_gemm(
        input,
        weight,
        sfa,
        sfb,
        output,
        input2,
        weight2,
        sfa2,
        sfb2,
        output2,
        sm_count,
        use_pdl,
        use_sigmoid,
    )
