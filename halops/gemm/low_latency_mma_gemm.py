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

__all__ = ["low_latency_mma_gemm", "prefer_low_latency_mma"]


def low_latency_mma_gemm(
    input: torch.Tensor,
    weight: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    output: torch.Tensor,
    block_n: int = -1,
    warps_k: int = -1,
    stage_smem: int = -1,
    persist: int = -1,
    sm_count: int = -1,
    output2: Optional[torch.Tensor] = None,
    n_split: int = 0,
    input2: Optional[torch.Tensor] = None,
    sfa2: Optional[torch.Tensor] = None,
    use_pdl: bool = True,
) -> None:
    """Low-latency W8A8-MXFP8 dense GEMM for decode-sized M on Blackwell SM100.

    Computes ``output = input @ weight^T`` with warp-level ``mma.sync``
    instead of tcgen05: one resident CTA per SM keeps A in shared memory and
    walks a list of ``block_n``-wide N tiles, streaming B through a TMA ring.
    It targets M <= 16, where the UMMA tiles of ``dense_gemm`` cannot fill the
    SMs. Shapes it cannot serve raise instead of falling back; route them to
    ``dense_gemm``.

    Args:
        input: [M, K] CUDA tensor, dtype float8_e4m3fn, row-major. M <= 16 and
            K divisible by 128.
        weight: [N, K] CUDA tensor, dtype float8_e4m3fn, row-major. N divisible
            by ``block_n``.
        sfa: Scale factor for A, dtype uint8 UE8M0, plain row-major layout
            with at least [M, K // 32] elements, i.e.
            ``group_wise_quant(input, mode="w8a8-mxfp8", use_swizzle=False)``.
            Must be K-contiguous; a row pitch that is not a multiple of 16
            bytes is repacked on the host.
        sfb: Scale factor for B, dtype uint8, same layout rules as ``sfa``
            with at least [N, K // 32] elements.
        output: [M, N] pre-allocated output tensor, dtype bfloat16. Written
            in-place. With ``output2`` it is the [M, n_split] first segment.
        block_n: N columns per CTA, one of 8 / 16 / 32 / 64; -1 selects by
            shape.
        warps_k: Warps splitting the K loop inside a CTA. 2 * warps_k must
            divide K / 128; -1 selects by shape.
        stage_smem: Upper bound on the B ring depth; -1 selects the deepest
            ring that fits in shared memory.
        persist: N-tile order per CTA: 2 walks a contiguous block of tiles,
            -1 / 1 walk them cyclically.
        sm_count: Number of SMs to use (-1 for all available).
        output2: Optional [M, N - n_split] bfloat16 tensor. When set, weight
            rows [0, n_split) are written to ``output`` and rows
            [n_split, N) to ``output2`` from the same launch.
        n_split: Column split between ``output`` and ``output2``; a multiple
            of ``block_n``. Only used with ``output2``.
        input2: Optional [M, K2] float8_e4m3fn activation stacked after
            ``input`` along K: the launch computes
            ``input @ weight[:, :K1]^T + input2 @ weight[:, K1:]^T`` with
            K1 + K2 == K, both multiples of 128. Not combinable with
            ``output2``.
        sfa2: Scale factor for ``input2``, required with it; same dtype and
            layout rules as ``sfa`` with at least [M, K2 // 32] elements.
        use_pdl: Enable Programmatic Dependent Launch (on by default). Turned
            off for a launch whose ``sfb`` needs the host repack.

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
        block_n,
        warps_k,
        stage_smem,
        persist,
        sm_count,
        output2,
        n_split,
        input2,
        sfa2,
        use_pdl,
    )


def prefer_low_latency_mma(m: int, k: int) -> bool:
    """Whether ``low_latency_mma_gemm`` has a tile config for an [M, K] input.

    Args:
        m: Number of rows of ``input``.
        k: Reduction dimension.

    Returns:
        ``True`` when M <= 16 and K is an even number of 128-wide k-blocks.
    """
    return m <= 16 and k % 128 == 0 and (k // 128) % 2 == 0
