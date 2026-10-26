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

__all__ = ["mla_attention"]


def mla_attention(
    q_or_x: torch.Tensor,
    hidden_states: torch.Tensor,
    kv_cache: torch.Tensor,
    page_size: int,
    positions: torch.Tensor,
    freqs_real: torch.Tensor,
    sm_scale: float,
    wq_a_weight: torch.Tensor,
    wq_a_sf: torch.Tensor,
    q_norm_weight: torch.Tensor,
    wq_b_weight: torch.Tensor,
    wq_b_sf: torch.Tensor,
    wkv_weight: torch.Tensor,
    wkv_sf: torch.Tensor,
    kv_norm_weight: torch.Tensor,
    swa_loc: torch.Tensor,
    attn_sink: Optional[torch.Tensor],
    workspace: torch.Tensor,
    q_eps: float = 1e-6,
    kv_eps: float = 1e-6,
    wkv_gate_weight: Optional[torch.Tensor] = None,
    kv_score_ring: Optional[torch.Tensor] = None,
    kv_gate_ring_row: Optional[torch.Tensor] = None,
    c128_ape: Optional[torch.Tensor] = None,
    c128_norm_weight: Optional[torch.Tensor] = None,
    c128_freqs_real: Optional[torch.Tensor] = None,
    c128_out_loc: Optional[torch.Tensor] = None,
    c128_pool: Optional[torch.Tensor] = None,
    c128_stride_kv_block: int = 0,
    c128_page_block: int = 0,
    c128_page_table: Optional[torch.Tensor] = None,
    c128_attn_len: Optional[torch.Tensor] = None,
    draft_tokens: int = 0,
    reset_barriers: bool = True,
) -> None:
    """DSV4 MLA decode/verify megakernel up to the sparse attention, on SM100.

    One persistent launch (one CTA per SM) runs x -> wq_a / wkv GEMMs ->
    q / kv RMSNorm + RoPE + KV cache write -> wq_b GEMM, the kv_gate GEMM and
    c128 compressor of HCA layers, and the sparse attention over the SWA window
    (plus the compressed c128 segment on HCA layers).  ``mla_output_proj``
    finishes the layer.

    Only the DSV4 attention config is supported: 64 heads of head dim 512
    (448 NoPE + 64 RoPE), q_lora rank 1024, kv_lora rank 512, value dim 512,
    c128 compress ratio 128, fp8 KV rows with one ue8m0 scale per 64 NoPE
    elements.  The hidden size H is free but must be a multiple of 32.

    Each token's SWA window (its KV runs and length) comes from the plan slot
    of this layer kind in ``workspace``: call ``mla_attention_plan`` for the
    kind first, once per forward, on the same stream with the same T.

    Args:
        q_or_x: [T, 64 * 512] bfloat16, contiguous, T > 0.  q staging buffer;
            on return it holds the normalized attention output O (before the
            inverse RoPE) of every token whose KV was not split.  The other
            tokens' partials stay in ``workspace``; pass both to
            ``mla_output_proj``.
        hidden_states: [T, H] bfloat16 layer input, unit column stride, 16-byte
            aligned rows (may be a column slice of a wider buffer).
        kv_cache: [num_pages, page_bytes] uint8 paged SWA KV cache.  A page holds
            ``page_size`` rows of 576 bytes (448 float8_e4m3fn NoPE + 64
            bfloat16 RoPE), then ``page_size`` 8-byte ue8m0 scale slots;
            ``page_bytes`` is a multiple of 576.  Each token's new row is
            written in place.
        page_size: Rows per ``kv_cache`` page, a power of two, as passed to
            ``mla_attention_plan``.
        positions: [T] int32 position of each token.
        freqs_real: [max_pos, 64] float32 RoPE table, contiguous, 32 interleaved
            (cos, sin) pairs per position.
        sm_scale: Softmax scale of the QK scores.
        wq_a_weight: [1024, H] float8_e4m3fn wq_a weight.
        wq_a_sf: [1024, H / 32] uint8 SfAtom ue8m0 scale of ``wq_a_weight``.
        q_norm_weight: [1024] bfloat16 RMSNorm gain of q_lora.
        wq_b_weight: [64 * 512, 1024] float8_e4m3fn wq_b weight.
        wq_b_sf: [64 * 512, 1024 / 32] uint8 SfAtom ue8m0 scale of
            ``wq_b_weight``.
        wkv_weight: [512, H] float8_e4m3fn wkv weight.
        wkv_sf: [512, H / 32] uint8 SfAtom ue8m0 scale of ``wkv_weight``.
        kv_norm_weight: [512] bfloat16 RMSNorm gain of the new KV row.
        swa_loc: [T] int32 ``kv_cache`` slot of each token's new KV row.
        attn_sink: [64] float32 per-head sink logit added to the softmax
            denominator, or None for no sink.
        workspace: uint8 workspace from ``mla_workspace``, planned for this
            layer kind by ``mla_attention_plan``.
        q_eps: RMSNorm epsilon of q.
        kv_eps: RMSNorm epsilon of the KV rows (new and compressed).
        wkv_gate_weight: [2 * 512, H] bfloat16 kv_gate weight (compressor kv and
            score halves).  This and the ``kv_*`` / ``c128_*`` arguments below
            are given on HCA layers and left None on SWA layers;
            ``wkv_gate_weight`` and ``c128_attn_len`` must be given together.
        kv_score_ring: [num_requests * 128, 2 * 512] float32 ring of each
            request's last 128 kv_gate outputs ([kv | score]); updated in place.
        kv_gate_ring_row: [T] int32 ``kv_score_ring`` row of each token
            (request * 128 + position % 128).
        c128_ape: [128, 512] float32 absolute position bias added to the ring
            scores.
        c128_norm_weight: [512] bfloat16 RMSNorm gain of a compressed row.
        c128_freqs_real: [max_pos, 64] float32 RoPE table of compressed rows;
            a row fired at position p is roped at p - 127 (its window start).
        c128_out_loc: [T] int32 ``c128_pool`` slot written by a token whose
            position % 128 == 127 (the compressor fires there).
        c128_pool: uint8 paged compressed KV pool, same row and scale layout as
            ``kv_cache`` with ``c128_page_block`` rows per page.
        c128_stride_kv_block: Bytes per ``c128_pool`` page, a multiple of 576.
        c128_page_block: Rows per ``c128_pool`` page, a power of two.
        c128_page_table: [T, max_pages] int32 ``c128_pool`` pages of each
            token's compressed segment.
        c128_attn_len: [T] int32 compressed rows each token attends over; None
            on SWA layers.
        draft_tokens: Tokens per request in verify, laid out consecutively at
            consecutive positions; 0 if unknown.
        reset_barriers: True to zero the barrier banks at exit; pass False only
            if the next op on the stream (``mla_output_proj``) does it.

    Returns:
        None. Results are written in place.
    """
    halops_kernel.mla_attention(
        q_or_x,
        kv_cache,
        int(page_size),
        positions,
        freqs_real,
        float(sm_scale),
        wq_a_weight,
        q_norm_weight,
        wq_b_weight,
        float(q_eps),
        wkv_weight,
        kv_norm_weight,
        float(kv_eps),
        swa_loc,
        attn_sink,
        wq_a_sf,
        wq_b_sf,
        wkv_sf,
        workspace,
        wkv_gate_weight,
        kv_score_ring,
        kv_gate_ring_row,
        c128_ape,
        c128_norm_weight,
        c128_freqs_real,
        c128_out_loc,
        c128_pool,
        int(c128_stride_kv_block),
        int(c128_page_block),
        c128_page_table,
        c128_attn_len,
        int(draft_tokens),
        hidden_states,
        bool(reset_barriers),
    )
