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

__all__ = ["prefill_pair_mla"]


def prefill_pair_mla(
    q: torch.Tensor,
    swa_k_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_lengths: torch.Tensor,
    c128_k_cache: torch.Tensor,
    c128_indices: torch.Tensor,
    c128_lengths: torch.Tensor,
    attn_sink: Optional[torch.Tensor],
    softmax_scale: float,
    tokens_per_req: int = 0,
    out: Optional[torch.Tensor] = None,
    use_pdl: bool = True,
    cu_seqlens: Optional[torch.Tensor] = None,
    cu_pairs: Optional[torch.Tensor] = None,
    swa_shift: int = 1,
    q_positions: Optional[torch.Tensor] = None,
    q_freqs: Optional[torch.Tensor] = None,
    q_norm_eps: float = 1e-6,
) -> torch.Tensor:
    """Sparse fp8 MLA prefill attention over a SWA and a compressed segment.

    Each token attends its SWA rows plus its compressed (``c128_*``) rows with
    an optional per-head sink, like FlashMLA's sparse fp8
    ``flash_mla_with_kvcache`` with ``extra_k_cache``.  A 2-CTA cluster
    attends two consecutive tokens ``a = 2j``, ``b = 2j + 1`` of a request
    over the union of their lists, so each KV row is dequantized once for
    both.  The lists must therefore have the structure of a causal prefill
    chunk, or the result is wrong (no out-of-bounds reads):

    - SWA: newest first, ``a``'s list is ``b``'s shifted by ``swa_shift``.
    - compressed: ``a``'s list is a prefix of ``b``'s.  This holds for HCA
      and dense C4 but not for top-k lists (sparse C4); only HCA is tested.

    Under CP round-robin pass the local request boundaries and
    ``swa_shift = cp``; padding rows go in one extra trailing request.

    With ``q_positions`` and ``q_freqs``, ``q`` is the raw projection and the
    kernel applies DSv4's ``q_norm_rope`` (weightless RMS norm, RoPE on dims
    448..511).  It is more accurate than, but not bit-exact with,
    ``fused_q_norm_rope`` followed by this kernel.

    The first call on a device allocates a persistent workspace, grown to the
    largest call: run the largest call before capturing a CUDA graph, and do
    not run calls concurrently on one device.

    Args:
        q: [T, 64, 512] or [T, 1, 64, 512] bfloat16, contiguous; with
            ``q_positions`` any 16 B aligned token stride is allowed (each
            token's heads contiguous).
        swa_k_cache: [pages, page_size, 1, 584] uint8 V4 fp8 cache (448 e4m3
            nope + 64 bf16 rope + 8 B UE8M0 scales per row, scales after a
            page's rows); power-of-two ``page_size``, page stride a multiple
            of 576 B.
        swa_indices: [T, W] or [T, 1, W] int32 contiguous SWA row ids.
        swa_lengths: [T] int32 valid SWA rows per token.
        c128_k_cache: Compressed cache, same layout as ``swa_k_cache``.
        c128_indices: [T, Wc] or [T, 1, Wc] int32 contiguous row ids.
        c128_lengths: [T] int32 valid compressed rows per token.
        attn_sink: Optional [64] float32 per-head sink logits.
        softmax_scale: Scale of ``q k^T``.
        tokens_per_req: Tokens per request (``>= 2``, divides T) when
            ``cu_seqlens`` is not given.
        out: Optional contiguous bfloat16 output with ``q``'s element count.
        use_pdl: Enable Programmatic Dependent Launch.
        cu_seqlens: Optional [R + 1] int32 request token offsets.
        cu_pairs: [R + 1] int32 request pair offsets (``ceil(len / 2)`` per
            request), given with ``cu_seqlens``.
        swa_shift: Position gap of a pair's tokens, in [1, 128].
        q_positions: Optional [T] int32 / int64 token positions; enables the
            fused q_norm_rope.
        q_freqs: [P, 32] complex64 or [P, 64] float32 RoPE table,
            ``P > max(q_positions)``.
        q_norm_eps: Epsilon of the fused RMS norm.

    Returns:
        ``out``, or a new tensor shaped like ``q``.

    Example:
        >>> from halops import prefill_pair_mla
        >>>
        >>> lens = torch.tensor([1023, 1, 2048], dtype=torch.int32, device="cuda")
        >>> cu_seqlens = torch.nn.functional.pad(lens.cumsum(0), (1, 0)).int()
        >>> cu_pairs = torch.nn.functional.pad(
        ...     ((lens + 1) // 2).cumsum(0), (1, 0)).int()
        >>> o = prefill_pair_mla(
        ...     q, swa_cache, swa_idx, swa_len, c128_cache, c_idx, c_len,
        ...     sink, 512 ** -0.5, cu_seqlens=cu_seqlens, cu_pairs=cu_pairs)
    """
    return halops_kernel.prefill_pair_mla(
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
        cu_seqlens,
        cu_pairs,
        swa_shift,
        q_positions,
        q_freqs,
        q_norm_eps,
    )
