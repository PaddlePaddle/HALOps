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
    """Project, normalize, rotate and quantize indexer Q/K in one CUDA launch.

    This entry point writes into caller-owned CUDA buffers and returns ``None``.
    All tensors must be contiguous where stated, reside on the same CUDA
    device, and satisfy the alignment requirements of the SM100 kernel. The
    build requires an SM100 GPU with at least 128 SMs.

    Args:
        x: Contiguous [T, 4096] bfloat16 input for K and Q projection. Row
            starts must meet the kernel's 16-byte alignment requirement.
        q_lora: Contiguous [T, 1024] bfloat16 input for Q projection. Row
            starts must meet the kernel's 8-byte alignment requirement.
        wk_w: Contiguous [192, 4096] bfloat16 weights for K and Q's weights.
        wq_b: Contiguous [8192, 1024] float8_e4m3fn Q projection weight.
        wq_b_sf: Contiguous [8192, 32] uint8 group-32 UE8M0 scales in
            CUTLASS SfAtom layout, as returned by
            ``group_wise_quant(..., mode="w8a8-mxfp8", use_swizzle=True)``.
        k_norm_weight: Contiguous [128] float32 normalization gain.
        k_norm_bias: Contiguous [128] float32 bias; a tensor selects
            LayerNorm, while None selects RMSNorm.
        k_norm_eps: Normalization epsilon.
        rope_cos_sin_cache: Contiguous [max_position, 64] float32, stored as
            [cos(32) | sin(32)]. RoPE rotates the first 64 Q/K elements.
        positions: Contiguous [T] int32/int64 RoPE positions. Values must fit
            int32 and index the supplied table.
        paged_kv_cache_with_scale: Row-contiguous uint8 page pool. In fp8 mode,
            [num_pages, page_size * 132], with all 128-byte FP8 rows followed
            by one float32 scale per row. In mxfp8 mode the shape is the same,
            with four uint8 UE8M0 group scales per row. In mxfp4 mode,
            [num_pages, page_size * 68], with all 64-byte packed E2M1 rows
            followed by four uint8 UE8M0 scales per row (group size 32).
            page_size must be a power of two. The buffer is written in place.
        kv_cache_loc: Contiguous [T] int32/int64 cache slots. Negative slots
            skip K writes. Nonnegative slots must be in range and unique within
            a launch; all values must fit int32. Slots are independent of
            positions.
        out_q: Contiguous output buffer whose dtype selects the Q/K format.
            float8_e4m3fn with ``out_q_sf=None`` is [T, 64, 128] FP8;
            float8_e4m3fn with ``out_q_sf`` is [T, 64, 128] MXFP8; and int8
            with ``out_q_sf`` is [T, 64, 64] packed MXFP4. Even elements
            occupy the low nibble. The buffer must be preallocated.
        out_q_sf: Optional contiguous [T, 64, 4] uint8 UE8M0 scale output for
            MXFP4/MXFP8, in linear group order. Each scale covers 32 Q
            elements. It must be None for FP8, whose Q scale is folded into
            weights.
        out_weights: Contiguous [T, 64] float32, bfloat16 or float16 output.
            Weights are computed in FP32 and rounded to its dtype. The buffer
            must be preallocated.
        weight_scale: Scalar folded into weights.
        is_neox_style: True (default) pairs dimensions i and i+32 in the
            rotary prefix. False selects adjacent pairs (2*i, 2*i+1).
        apply_hadamard: Apply the normalized 128-point Hadamard to Q and K.
        use_pdl: Enable Programmatic Dependent Launch. Successors may initialize
            early; their dependency wait observes complete outputs.

    Returns:
        None. ``out_q``, ``out_q_sf`` and ``out_weights`` are modified in
        place. The cache is also updated in place at nonnegative
        ``kv_cache_loc`` slots.
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
        is_neox_style,
        apply_hadamard,
        use_pdl,
    )
