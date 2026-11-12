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

__all__ = ["indexer_prepare"]


def indexer_prepare(
    x: torch.Tensor,
    q_lora: torch.Tensor,
    wk_w: torch.Tensor,
    wq_b: torch.Tensor,
    wq_b_sf: torch.Tensor,
    k_norm_weight: torch.Tensor,
    k_norm_bias: Optional[torch.Tensor],
    k_norm_eps: float,
    rope_cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    paged_kv_cache_with_scale: torch.Tensor,
    kv_cache_loc: torch.Tensor,
    out_q: torch.Tensor,
    out_q_sf: Optional[torch.Tensor],
    out_weights: torch.Tensor,
    weight_scale: float,
    is_neox_style: bool = True,
    apply_hadamard: bool = True,
    use_pdl: bool = True,
) -> None:
    """Fuse indexer Q/K preparation into one CUDA launch.

    The operator projects ``x`` to K and raw per-head weights, quantizes
    ``q_lora`` before the Q projection, then applies normalization, RoPE and
    the optional Hadamard transform. It writes the selected FP8, MXFP8 or
    packed MXFP4 Q/K representation to the caller-owned output buffers and
    updates the paged K cache in place.

    The current CUDA specialization uses 64 heads, head dimension 128 and
    rotary dimension 64. ``hidden_size`` and ``q_lora_rank`` are runtime
    dimensions; the supported projection configurations include
    ``(4096, 1024)`` and ``(7168, 1536)``. The kernel requires an SM100-family
    GPU with at least 128 available SMs. It validates the remaining shape,
    dtype and layout requirements at the native boundary.

    Args:
        x: Contiguous [T, hidden_size] bfloat16 input. ``hidden_size`` must
            be positive and divisible by 128.
        q_lora: Contiguous [T, q_lora_rank] bfloat16 input. ``q_lora_rank``
            must be divisible by 256 and therefore by the group size 32.
        wk_w: Contiguous [head_dim + num_heads, hidden_size] bfloat16
            projection weights. The first ``head_dim`` rows project K and the
            last ``num_heads`` rows produce the raw per-head weights.
        wq_b: Contiguous [num_heads * head_dim, q_lora_rank]
            float8_e4m3fn Q projection weights.
        wq_b_sf: Contiguous [num_heads * head_dim, q_lora_rank // 32]
            uint8 group-32 UE8M0 scales in CUTLASS SfAtom layout, as returned
            by ``group_wise_quant(..., mode="w8a8-mxfp8", use_swizzle=True)``.
        k_norm_weight: Contiguous [head_dim] float32 normalization gain.
        k_norm_bias: Optional contiguous [head_dim] float32 bias. A tensor
            selects LayerNorm, while ``None`` selects RMSNorm.
        k_norm_eps: Positive normalization epsilon.
        rope_cos_sin_cache: Contiguous [max_position, rope_dim] float32
            table in ``[cos | sin]`` layout. For this build, ``rope_dim=64``
            and the first 64 Q/K values are rotated.
        positions: Contiguous [T] int32 or int64 RoPE positions. Values must
            fit int32 and index the supplied table.
        paged_kv_cache_with_scale: Row-contiguous uint8 page pool. In fp8 mode,
            ``[num_pages, page_size * 132]`` for FP8/MXFP8 or
            ``[num_pages, page_size * 68]`` for MXFP4. Data rows precede
            four scale bytes per row. The buffer is written in place.
        kv_cache_loc: Contiguous [T] int32 or int64 cache slots. Negative
            slots skip K writes. Nonnegative slots are independent of
            ``positions`` and must be valid for the allocated page pool.
        out_q: Contiguous preallocated output buffer. FP8 and MXFP8 use
            float8_e4m3fn [T, num_heads, head_dim]; MXFP4 uses int8
            [T, num_heads, head_dim // 2] with two packed E2M1 values per
            byte. ``out_q_sf`` selects MXFP8 or MXFP4.
        out_q_sf: Optional contiguous preallocated uint8
            [T, num_heads, head_dim // 32] UE8M0 scale output for MXFP8 or
            MXFP4. Each scale covers 32 values. It must be ``None`` for FP8,
            whose Q scale is folded into ``out_weights``.
        out_weights: Contiguous preallocated [T, num_heads] float32,
            bfloat16 or float16 output. Weights are computed in FP32 and
            rounded when stored.
        weight_scale: Scalar multiplied into the raw per-head weights. For
            FP8 output, the per-row Q scale is folded into the same weights.
        is_neox_style: True (default) pairs dimensions i and i+32 in the
            rotary prefix. False selects adjacent pairs (2*i, 2*i+1).
        apply_hadamard: Apply the normalized 128-point Hadamard transform to
            Q and K.
        use_pdl: Enable CUDA Programmatic Dependent Launch. Successors may
            initialize early; their dependency wait observes completed outputs.

    Returns:
        None. ``out_q``, ``out_q_sf``, ``out_weights`` and the selected cache
        slots are modified in place.
    """

    halops_kernel.indexer_prepare(
        x,
        q_lora,
        wk_w,
        wq_b,
        wq_b_sf,
        k_norm_weight,
        k_norm_bias,
        k_norm_eps,
        rope_cos_sin_cache,
        positions,
        paged_kv_cache_with_scale,
        kv_cache_loc,
        out_q,
        out_q_sf,
        out_weights,
        weight_scale,
        is_neox_style=is_neox_style,
        apply_hadamard=apply_hadamard,
        use_pdl=use_pdl,
    )
