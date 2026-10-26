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

__all__ = ["mla_output_proj"]


def mla_output_proj(
    q_or_x: torch.Tensor,
    positions: torch.Tensor,
    freqs_real: torch.Tensor,
    wo_a_weight: torch.Tensor,
    wo_a_sf: torch.Tensor,
    wo_b_weight: torch.Tensor,
    wo_b_sf: torch.Tensor,
    out: torch.Tensor,
    workspace: torch.Tensor,
    attn_sink: Optional[torch.Tensor] = None,
    c128_attn_len: Optional[torch.Tensor] = None,
) -> None:
    """MLA output projection after an ``mla_attention`` launch, on SM100.

    Computes ``out = wo_b(quant(wo_a(quant(rope^-1(combine(O))))))``: the
    split-KV LSE combine and inverse RoPE of O, group-32 e4m3 / ue8m0
    quantization, the grouped wo_a GEMM, group-32 quantization and the wo_b
    GEMM, as two PDL-chained kernels.  It also zeroes the attention's barrier
    banks, so the preceding ``mla_attention`` may pass ``reset_barriers=False``.

    The wo_a weight stacks G output groups: group g maps O columns
    [g * K, (g + 1) * K) of the flat [T, 64 * 512] O to its 2 * H / G rows of
    o_lora, with K = 64 * 512 / G.  o_lora is twice the hidden size H.  The
    fixed work split needs H a multiple of 512 and at most 4096, K a multiple
    of 512 and 2 * H / G a multiple of 256.

    Args:
        q_or_x: [T, 64 * 512] bfloat16, contiguous, 1 <= T <= 640.  The
            ``q_or_x`` of the attention launch, which holds O on return of
            ``mla_attention``; read only.
        positions: [T] int32 token positions of the attention launch.
        freqs_real: [max_pos, 64] float32 RoPE table of the attention launch.
        wo_a_weight: [2 * H, 64 * 512 / G] float8_e4m3fn wo_a weight.
        wo_a_sf: [2 * H, 64 * 512 / G / 32] uint8 SfAtom ue8m0 scale of
            ``wo_a_weight``.
        wo_b_weight: [H, 2 * H] float8_e4m3fn wo_b weight.
        wo_b_sf: [H, 2 * H / 32] uint8 SfAtom ue8m0 scale of ``wo_b_weight``.
        out: [T, H] bfloat16, contiguous; written in place.
        workspace: uint8 workspace of the attention launch.
        attn_sink: [64] float32 attention sink of the attention launch, or None
            if it had none.
        c128_attn_len: ``c128_attn_len`` of the attention launch (None on SWA
            layers).  Only whether it is None is used: it selects the SWA or
            HCA plan slot of ``workspace``, so it must match the attention
            launch.

    Returns:
        None. The result is written in place to ``out``.
    """
    halops_kernel.combine_with_rope_quant(
        q_or_x,
        positions,
        freqs_real,
        attn_sink,
        c128_attn_len,
        wo_a_weight,
        wo_b_weight,
        workspace,
    )
    halops_kernel.mla_output_proj(
        wo_a_weight,
        wo_a_sf,
        wo_b_weight,
        wo_b_sf,
        out,
        workspace,
    )
