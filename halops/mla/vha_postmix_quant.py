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

from typing import Optional, Tuple

import torch

import halops.halops_kernel as halops_kernel

__all__ = ["vha_postmix_quant"]


def vha_postmix_quant(
    o: torch.Tensor,
    mix: torch.Tensor,
    freqs: torch.Tensor,
    positions: torch.Tensor,
    groups: int = 8,
    group_major: bool = True,
    use_pdl: bool = True,
    output_quant: Optional[torch.Tensor] = None,
    output_sf: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Inverse RoPE + VHA cross-head postmix + W8A8-MXFP8 quantization.

    For every token ``t``: the last 64 dims of each head of ``o`` are rotated
    back (interleaved pairs, ``e' = e cos + o sin``, ``o' = o cos - e sin`` in
    fp32, rounded to bf16), mixed across heads with
    ``out[t, h] = sum_h' mix[h, h'] o'[t, h']`` (fp32 accumulation, rounded to
    bf16), viewed as ``[groups, D = 64 / groups * 512]`` (group g = heads
    ``g * 64 / groups ...``) and quantized like
    ``group_wise_quant(..., "w8a8-mxfp8")``.

    With ``mix = I`` the result is bit-identical to inverse RoPE followed by
    ``group_wise_quant``; otherwise the mix's accumulation order differs from
    ``torch.matmul`` by at most 1 bf16 ulp per element.

    Args:
        o: [T, 64, 512] bfloat16 contiguous attention output.
        mix: [64, 64] bfloat16 postmix matrix.
        freqs: [max_pos, 64] float32 table, (cos, sin) of rope pair ``j`` at
            columns ``2j, 2j + 1``.
        positions: [T] int32 / int64 positions.
        groups: Head groups G (divides 64).
        group_major: ``True`` for a ``[G, T, D]`` output with scale rows
            ``g * T + t`` (the A operand of ``batch_gemm``); ``False`` for
            ``[T, G, D]`` with scale rows ``t * G + g``.
        use_pdl: Enable Programmatic Dependent Launch (on by default).
        output_quant: Optional pre-allocated contiguous float8_e4m3fn output.
        output_sf: Optional pre-allocated contiguous scale buffer.

    Returns:
        ``(output_quant, output_sf)``: float8_e4m3fn codes and
        [ceil(G * T / 128) * 128, D / 32] uint8 UE8M0 scales in the CUTLASS
        SfAtom layout (padding rows hold 0).

    Example:
        >>> from halops import vha_postmix_quant
        >>>
        >>> t = 128
        >>> o = torch.randn(t, 64, 512, dtype=torch.bfloat16, device="cuda")
        >>> mix = torch.eye(64, dtype=torch.bfloat16, device="cuda")
        >>> theta = torch.rand(4096, 32, device="cuda")
        >>> freqs = torch.stack([theta.cos(), theta.sin()], -1).flatten(-2)
        >>> pos = torch.arange(t, device="cuda")
        >>> o_fp8, sf = vha_postmix_quant(o, mix, freqs, pos, groups=8)
    """
    return halops_kernel.vha_postmix_quant(
        o,
        mix,
        freqs,
        positions,
        groups,
        group_major,
        use_pdl,
        output_quant,
        output_sf,
    )
