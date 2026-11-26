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

__all__ = ["hca_decode_attention", "dsa_decode_attention"]


def hca_decode_attention(
    q: torch.Tensor,
    swa_k_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_lengths: torch.Tensor,
    c128_k_cache: torch.Tensor,
    c128_indices: torch.Tensor,
    c128_lengths: torch.Tensor,
    attn_sink: Optional[torch.Tensor],
    softmax_scale: float,
    tokens_per_req: int,
    out: Optional[torch.Tensor] = None,
    use_pdl: bool = True,
) -> torch.Tensor:
    """HCA sparse decode attention over fp8 KV caches, batched per request.

    Same semantics as FlashMLA's fp8 sparse decode with an extra cache
    (``flash_mla_with_kvcache(..., is_fp8_kvcache=True, indices=swa_indices,
    extra_k_cache=c128_k_cache, ...)``): token ``t`` attends the SWA rows
    ``swa_indices[t, :swa_lengths[t]]`` and the c128 rows
    ``c128_indices[t, :c128_lengths[t]]``; an index that is negative or past
    the end of its cache is skipped, and ``attn_sink`` joins the softmax
    denominator.  A row with no valid key yields 0.  (FlashMLA skips -1 too,
    but a past-the-end SWA index there acts as a key with K = V = 0.)

    Tokens ``2j`` and ``2j + 1`` of a request (paired within the request, so a
    pair never straddles a request) attend the union of their lists, gathered
    and dequantized once.  When ``tokens_per_req`` is odd, each request's last
    token is unpaired and runs on its own.  The union relies on the verify-step structure: the earlier token's c128
    list is a prefix of the later one's, and its SWA list (newest first) is
    the later one's shifted by one row.  Where both tokens have valid but
    different keys at the same place of the union, the earlier token's key is
    dropped; such rows (CUDA-graph padding requests) are read safely, but
    their outputs can differ from FlashMLA's.  Invalid or short rows are
    exact.

    Args:
        q: [T, 64, 512] (or [T, 1, 64, 512]) bfloat16 contiguous queries.
        swa_k_cache: uint8 [pages, page_size, 1, 584] SWA cache (page stride a
            multiple of 576 bytes).
        swa_indices: int32 [T, W] (or [T, 1, W]) contiguous pool token indices.
        swa_lengths: int32 [T].
        c128_k_cache: uint8 [pages, page_size, 1, 584] c128 cache.
        c128_indices: int32 [T, W] (or [T, 1, W]) contiguous pool token indices.
        c128_lengths: int32 [T].
        attn_sink: Optional float32 [64] per-head sink logits.
        softmax_scale: Scale of the QK^T logits.
        tokens_per_req: Tokens per request (>= 2, divides T); requests are
            consecutive ``[r * t, (r + 1) * t)``.
        out: Optional pre-allocated contiguous bfloat16 output shaped like q.
        use_pdl: Enable Programmatic Dependent Launch (on by default).

    Returns:
        bfloat16 output shaped like ``q``.

    Note:
        The first call on a device allocates a persistent split workspace
        (about 20 MB with 152 SMs); make it before CUDA graph capture.
    """
    return halops_kernel.hca_decode_attention(
        q,
        swa_k_cache,
        swa_indices,
        swa_lengths,
        c128_k_cache,
        c128_indices,
        c128_lengths,
        attn_sink,
        softmax_scale,
        tokens_per_req,
        out,
        use_pdl,
    )


def dsa_decode_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    indices: torch.Tensor,
    lengths: torch.Tensor,
    attn_sink: Optional[torch.Tensor],
    softmax_scale: float,
    tokens_per_req: int,
    out: Optional[torch.Tensor] = None,
    use_pdl: bool = True,
) -> torch.Tensor:
    """DSA (full-MLA) fp8 sparse decode attention over a single KV cache.

    Same semantics as FlashMLA's fp8 full-MLA sparse decode
    (``flash_mla_with_kvcache(q[:, None], k_cache, head_dim_v=512,
    is_fp8_kvcache=True, indices=indices[:, None], attn_sink=attn_sink,
    topk_length=lengths, causal=False)``): token ``t`` attends the rows
    ``indices[t, :lengths[t]]``, contracting QK over all 576 dims (512 fp8
    latent + 64 bf16 rope), and ``attn_sink`` joins the softmax denominator.
    An index that is negative or past the end of the cache is skipped; a row
    with no valid key yields 0.  There is no SWA segment.

    Args:
        q: [T, 64, 576] (or [T, 1, 64, 576]) bfloat16 contiguous queries, the
            first 512 dims latent and the last 64 rope.
        k_cache: uint8 [pages, page_size, 1, 656] inline rows — 512 fp8 latent,
            4 f32 group scales (one per 128 latent dims), 64 bf16 rope.
        indices: int32 [T, W] (or [T, 1, W]) contiguous pool token indices.
        lengths: int32 [T].
        attn_sink: Optional float32 [64] per-head sink logits.
        softmax_scale: Scale of the QK^T logits (typically 576 ** -0.5).
        tokens_per_req: Tokens per request (>= 1, divides T); requests are
            consecutive ``[r * t, (r + 1) * t)``.
        out: Optional pre-allocated contiguous bfloat16 [T, 64, 512] output.
        use_pdl: Enable Programmatic Dependent Launch (on by default).

    Returns:
        bfloat16 [T, 64, 512] output.

    Note:
        The first call on a device allocates a persistent split workspace;
        make it before CUDA graph capture.
    """
    return halops_kernel.dsa_decode_attention(
        q,
        k_cache,
        indices,
        lengths,
        attn_sink,
        softmax_scale,
        tokens_per_req,
        out,
        use_pdl,
    )
