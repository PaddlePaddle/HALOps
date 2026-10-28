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
    sm_count: int = -1,
    output2: Optional[torch.Tensor] = None,
    n_split: int = 0,
    input2: Optional[torch.Tensor] = None,
    sfa2: Optional[torch.Tensor] = None,
    use_pdl: bool = True,
) -> None:
    """Low-latency W8A8-MXFP8 GEMM for decode-sized M, built on ``mma.sync``.

    Computes ``output = input @ weight^T`` for ``M <= 16``. The whole A operand
    stays resident in shared memory while each CTA streams B tiles; the tile
    shape and B ring depth are picked from (M, N, K, sm_count).

    On GB200 (cold L2, CUDA graph, PDL on) it is up to 2x faster than
    ``dense_gemm`` at M <= 8 with less than 32 MiB of weight (N * K), and at
    M = 16 with less than 8 MiB; on larger weights ``dense_gemm`` is as fast or
    faster.

    Scales use the plain row-major group-32 layout, not the CUTLASS SfAtom
    layout ``dense_gemm`` takes: quantize with
    ``group_wise_quant(..., use_swizzle=False)``. An SfAtom sfa is rejected by
    its padded row count, but an SfAtom sfb has the same shape as a plain one
    when N % 128 == 0 and gives wrong results without an error.

    Args:
        input: [M, K] contiguous CUDA tensor, dtype float8_e4m3fn. M <= 16,
            K % 256 == 0, and the resident [M, K] A must fit in shared memory
            (K up to 23808 at M <= 8, 12800 at M <= 16).
        weight: [N, K] contiguous CUDA tensor, dtype float8_e4m3fn.
            N % 8 == 0.
        sfa: [M, K // 32] uint8 UE8M0 scale of input, row-major. Each row
            must start on a 16-byte boundary with a pitch of at least
            ``round_up(K // 32, 16)`` bytes. A contiguous scale meets this
            when K % 512 == 0; for other K, pass the [M, K // 32] view of a
            [M, round_up(K // 32, 16)] buffer.
        sfb: [N, K // 32] uint8 UE8M0 scale of weight, same layout as sfa.
            With ``use_pdl`` it is read before the grid-dependency wait, so it
            must not be written by the kernel launched right before this one.
        output: [M, N] contiguous bfloat16 tensor, or [M, n_split] when
            ``output2`` is given. Written in-place.
        sm_count: Number of SMs to use (-1 for all available).
        output2: Optional [M, N - n_split] contiguous bfloat16 tensor receiving
            columns ``n_split:`` of the result, e.g. two projections stacked
            in one weight written to separate buffers. Not combinable with
            ``input2``.
        n_split: Column split for ``output2``, a multiple of 8.
        input2: Optional [M, K2] contiguous float8_e4m3fn tensor concatenated
            to input along K, so ``weight`` is [N, K1 + K2], e.g. two
            down-projections summed in one GEMM. K1 must be a multiple of 512
            and K2 of 128; for other splits concatenate the operands and their
            scales along K.
        sfa2: [M, K2 // 32] scale of input2, same dtype and layout as sfa.
            Required with ``input2``; ``sfa`` is then [M, K1 // 32].
        use_pdl: Enable Programmatic Dependent Launch (on by default).

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
        sm_count,
        output2,
        n_split,
        input2,
        sfa2,
        use_pdl,
    )
