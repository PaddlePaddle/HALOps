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
    """Low-latency W8A8-MXFP8 GEMM for decode-sized M, built on ``mma.sync``.

    Computes ``output = input @ weight^T`` for ``M <= 16``. The whole A operand
    stays resident in shared memory while each CTA streams B tiles, which beats
    ``dense_gemm`` when M is too small to fill a tcgen05 tile.

    Args:
        input: [M, K] CUDA tensor, dtype float8_e4m3fn. M <= 16, K % 128 == 0.
        weight: [N, K] CUDA tensor, dtype float8_e4m3fn. N % block_n == 0.
        sfa: Scale factor for A, dtype uint8 UE8M0, plain row-major
            ``[>= M, K // 32]`` (``group_wise_quant(..., use_swizzle=False)``).
            Rows whose pitch is not 16-byte granular are repacked on the host.
        sfb: Scale factor for B, same layout as sfa, shape ``[>= N, K // 32]``.
        output: [M, N] pre-allocated bfloat16 tensor, or ``[M, n_split]`` when
            ``output2`` is given. Written in-place.
        block_n: N tile per CTA, one of 8/16/32/64 (-1 picks from (N, K)).
        warps_k: Warps splitting K, with ``block_n * warps_k <= 128`` and
            ``(K / 128) % (2 * warps_k) == 0`` (-1 picks from (N, K)).
        stage_smem: Cap on B ring stages (-1 uses as many as fit in SMEM).
        persist: Tile order across CTAs: 1 cyclic, 2 blocked (-1 cyclic).
        sm_count: Number of SMs to use (-1 for all available).
        output2: Optional [M, N - n_split] bfloat16 tensor receiving columns
            ``n_split:`` of the result. Not combinable with ``input2``.
        n_split: Column split for ``output2``, divisible by block_n.
        input2: Optional [M, K2] float8_e4m3fn tensor concatenated to input
            along K, so ``weight`` is [N, K + K2]. Both K and K2 must be
            multiples of 128.
        sfa2: Scale factor for input2, same dtype/layout as sfa. Required with
            ``input2``.
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
