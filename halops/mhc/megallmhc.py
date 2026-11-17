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

__all__ = ["megallmhc"]


def megallmhc(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    norm_weight: torch.Tensor,
    *,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    norm_eps: float,
    mhc_align_fleet: bool = False,
    shared: Optional[torch.Tensor] = None,
    use_pdl: bool = True,
    use_swizzle: bool = False,
    token_tiles: int = 4,
) -> Tuple[torch.Tensor, ...]:
    """Fused mHC post + pre + RMSNorm + W8A8-MXFP8 quantization for decode.

    For every token: ``residual_out[j] = bf16(post[j] * (x + shared) +
    sum_i comb[i, j] * residual[i])``; the mix ``fn @ residual_out`` with its
    RMS gives ``pre``, ``post_out`` (sigmoid) and ``comb_out`` (sinkhorn);
    ``layer_out = RMSNorm(bf16(sum_j pre[j] * residual_out[j]))`` is quantized
    like ``group_wise_quant(layer_out, "w8a8-mxfp8", use_swizzle)``, bit for
    bit. Hidden size 4096 and 4 hyper-connection streams.

    Args:
        x: [T, 4096] bfloat16 layer output.
        residual: [T, 4, 4096] bfloat16 hyper-connection streams.
        post: [T, 4] float32 post weights of the previous junction.
        comb: [T, 4, 4] float32 comb of the previous junction, [source, dest].
        fn: [24, 16384] float32 mix weights, used at tf32 precision.
        hc_scale: [3] float32 mix scales.
        hc_base: [24] float32 mix biases.
        norm_weight: [4096] bfloat16 RMSNorm weight.
        rms_eps: Epsilon of the mix RMS.
        hc_pre_eps: Added to ``pre``.
        hc_sinkhorn_eps: Sinkhorn epsilon.
        hc_post_mult_value: Scale of ``post_out``.
        sinkhorn_repeat: Sinkhorn iterations.
        norm_eps: Epsilon of the RMSNorm.
        mhc_align_fleet: Use ``1 / (sqrt(ms) + eps)`` for the mix RMS and
            ``exp2`` in sinkhorn, as fleet; off by default.
        shared: Optional [T, 4096] bfloat16 term added to ``x``.
        use_pdl: Enable Programmatic Dependent Launch (on by default).
        use_swizzle: Scales in the CUTLASS SfAtom layout, rows padded to a
            multiple of 128.
        token_tiles: Max token tiles of the first kernel's grid; does not
            change the result.

    Returns:
        ``(residual_out, layer_out, post_out, comb_out, q, sf)``: bfloat16
        [T, 4, 4096] and [T, 4096], float32 [T, 4] and [T, 4, 4],
        float8_e4m3fn [T, 4096] codes and uint8 [T, 128] UE8M0 scales
        ([ceil(T / 128) * 128, 128] with ``use_swizzle``).
    """
    return tuple(
        halops_kernel.megallmhc(
            x,
            residual,
            post,
            comb,
            fn,
            hc_scale,
            hc_base,
            norm_weight,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            norm_eps,
            mhc_align_fleet,
            shared,
            use_pdl,
            use_swizzle,
            token_tiles,
        )
    )
