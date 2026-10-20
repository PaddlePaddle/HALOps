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

from __future__ import annotations
import math
import torch
import halops


def fold_dense_q_to_halops(
    q: torch.Tensor,
    H_KV: int,
    kblockm: int = 64,
) -> torch.Tensor:
    """Fold dense Q into the layout consumed by HALOps decode.

    Input is standard attention layout [B, S_Q, H_Q, D]. HALOps groups query
    heads by KV head and stores the active rows in [B, H_KV, S_Q * GQA, D],
    padded to [B, H_KV, KBLOCKM, D]. The input dtype is preserved, so this works
    for both BF16 Q and already-quantized FP8 Q.
    """
    B, s_q, H_Q, D = q.shape
    gqa = H_Q // H_KV
    q_bw = torch.zeros(
        B,
        H_KV,
        kblockm,
        D,
        dtype=q.dtype,
        device=q.device,
    )
    q_bw[:, :, : s_q * gqa, :] = (
        q.view(B, s_q, H_KV, gqa, D)
        .permute(0, 2, 1, 3, 4)
        .reshape(B, H_KV, s_q * gqa, D)
    )
    return q_bw


def unfold_halops_q(
    q_bw: torch.Tensor,
    s_q: int,
    H_Q: int,
    H_KV: int,
    flatten: bool = False,
) -> torch.Tensor:
    """Unfold HALOps Q layout back to standard query-head order.

    The operation is the inverse of fold_dense_q_to_halops for the active
    S_Q * GQA rows. It does not dequantize or cast, so FP8 q_bw remains FP8 for
    A8C8 backends. When flatten=True, returns [B * S_Q, H_Q, D] for FlashInfer
    decode APIs; otherwise returns dense [B, S_Q, H_Q, D].
    """
    B, _, _, D = q_bw.shape
    gqa = H_Q // H_KV
    q = q_bw[:, :, : s_q * gqa, :].reshape(B, H_KV, s_q, gqa, D).permute(0, 2, 1, 3, 4)
    if flatten:
        return q.reshape(B * s_q, H_Q, D).contiguous()
    return q.reshape(B, s_q, H_Q, D).contiguous()


def recover_dense_q(
    q_bw: torch.Tensor,
    B: int,
    s_q: int,
    H_Q: int,
    H_KV: int,
    D: int,
    q_dequant_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Recover dense Q from the q tensor actually consumed by HALOps.

    BF16 q_bw only needs to be unfolded from HALOps's GQA layout. FP8 q_bw
    also needs dequantization; write_cache_kv_decoder folds scale_softmax_log2
    into q_dequant_scale so the FP8 attention kernel can feed exp2 directly.
    Dense reference backends apply softmax_scale themselves, so divide that
    factor out when reconstructing Q.
    """
    if q_dequant_scale is not None:
        scale_softmax_log2 = math.log2(math.e) / math.sqrt(D)
        q_scale = (q_dequant_scale / scale_softmax_log2).view(B, H_KV, 1, 1)
        q_bw = q_bw[:, :, : s_q * (H_Q // H_KV), :].float() * q_scale
    return unfold_halops_q(q_bw, s_q, H_Q, H_KV).to(torch.bfloat16)


def halops_v_reorder_indices(
    block_size: int,
    device: torch.device,
    is_fp8: bool,
) -> torch.Tensor:
    """Return the intra-block V row permutation used by HALOps cache layout.

    The decode kernel stores V rows in a swizzled order to match its vectorized
    load pattern. BF16 swaps bit0/bit1 within groups of 4 rows; FP8 rotates the
    low 3 bits within groups of 8 rows. Both permutations are self-inverse, so
    the same indices can be used when writing logical V rows into cache and when
    reading cached V rows back into logical order.
    """
    rows = torch.arange(block_size, device=device)
    if is_fp8:
        return (rows & ~7) | ((rows & 1) << 2) | ((rows & 6) >> 1)
    return (rows & ~3) | ((rows & 1) << 1) | ((rows & 2) >> 1)


def build_cache_kv(
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    block_tables: torch.Tensor,
    k_src: torch.Tensor,
    v_src: torch.Tensor,
    total_kv: int,
    is_fp8: bool,
    kv_dequant_scale: torch.Tensor | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
) -> None:
    """Build HALOps paged cache layout from dense K/V pages.

    BF16 and FP8 share the same page construction. They differ in cache dtype,
    V row reorder, and FP8 scale metadata.
    """
    block_size = cache_k.shape[2]
    B, _, H_KV, D = k_src.shape
    pages_per_seq = math.ceil(total_kv / block_size)
    padded_len = pages_per_seq * block_size
    if padded_len > total_kv:
        k_padded = torch.zeros(
            B,
            padded_len,
            H_KV,
            D,
            dtype=k_src.dtype,
            device=k_src.device,
        )
        v_padded = torch.zeros_like(k_padded)
        k_padded[:, :total_kv] = k_src
        v_padded[:, :total_kv] = v_src
        k_src, v_src = k_padded, v_padded

    page_ids = block_tables[:, :pages_per_seq].reshape(-1).long()
    k_pages = k_src.reshape(B, pages_per_seq, block_size, H_KV, D).reshape(
        -1, block_size, H_KV, D
    )
    v_pages = v_src.reshape(B, pages_per_seq, block_size, H_KV, D).reshape(
        -1, block_size, H_KV, D
    )

    k_cache_pages = k_pages.permute(0, 2, 1, 3).contiguous()
    if is_fp8:
        assert kv_dequant_scale is not None
        assert k_scale is not None and v_scale is not None
        cache_k[page_ids] = k_cache_pages.view(torch.uint8)
    else:
        cache_k[page_ids] = k_cache_pages

    v_store_rows = halops_v_reorder_indices(block_size, cache_v.device, is_fp8)

    v_nhd_pages = v_pages.permute(0, 2, 1, 3).contiguous()
    v_cache_pages = torch.empty_like(v_nhd_pages)
    v_cache_pages[:, :, v_store_rows, :] = v_nhd_pages
    if is_fp8:
        cache_v[page_ids] = v_cache_pages.contiguous().view(torch.uint8)
    else:
        cache_v[page_ids] = v_cache_pages

    if is_fp8:
        kv_dequant_scale[:, :, 0] = k_scale
        kv_dequant_scale[:, :, 1] = v_scale


def write_cache_kv(
    B: int,
    s_kv: int,
    s_q: int,
    H_Q: int,
    H_KV: int,
    D: int,
    cache_k,
    cache_v,
    block_tables,
    kv_dequant_scale,
    input_scale: float,
    quant_mode: str,
):
    """Run write_cache_kv_encoder + write_cache_kv_decoder.

    Allocates encoder/decoder-specific tensors internally; the caller only needs
    to provide the shared cache/block_tables that persist after this call.
    Returns (q_bw, q_dequant_scale).
    """
    total_kv = s_kv + s_q
    max_input_len = total_kv

    # Identity RoPE
    rotary_embs = torch.zeros(
        2, max_input_len, D // 2, dtype=torch.float32, device="cuda"
    )
    rotary_embs[0] = 1.0

    # --- Encoder: populate cache with s_kv tokens ---
    qkv_enc = (
        torch.randn(B * s_kv, H_Q + 2 * H_KV, D, device="cuda") * input_scale
    ).to(torch.bfloat16)
    cu_q_enc = (torch.arange(B + 1, device="cuda") * s_kv).to(torch.int32)
    seq_len_enc = torch.full((B,), s_kv, dtype=torch.int32, device="cuda")
    seq_len_dec_zero = torch.zeros(B, dtype=torch.int32, device="cuda")
    q_enc_out = torch.zeros(B * s_kv, H_Q, D, dtype=torch.bfloat16, device="cuda")
    k_enc_out = torch.zeros(B * s_kv, H_KV, D, dtype=torch.bfloat16, device="cuda")
    v_enc_out = torch.zeros_like(k_enc_out)

    halops.write_cache_kv_encoder(
        qkv_enc,
        cu_q_enc,
        cu_q_enc.clone(),
        rotary_embs,
        seq_len_enc,
        seq_len_dec_zero,
        cache_k,
        cache_v,
        block_tables,
        q_enc_out,
        k_enc_out,
        v_enc_out,
        kv_dequant_scale,
        None,
        None,
        H_Q,
        H_KV,
        D,
        s_kv,
        s_kv,
        max_input_len,
        quant_mode,
    )

    # --- Decoder: write s_q tokens, get q_bw ---
    qkv_dec = (torch.randn(B * s_q, H_Q + 2 * H_KV, D, device="cuda") * input_scale).to(
        torch.bfloat16
    )
    cu_q_dec = (torch.arange(B + 1, device="cuda") * s_q).to(torch.int32)
    seq_len_enc_zero = torch.zeros(B, dtype=torch.int32, device="cuda")
    seq_len_dec_prev = torch.full((B,), s_kv, dtype=torch.int32, device="cuda")

    q_bw, q_dequant_scale = halops.write_cache_kv_decoder(
        qkv_dec,
        cu_q_dec,
        seq_len_enc_zero,
        seq_len_dec_prev,
        rotary_embs,
        cache_k,
        cache_v,
        block_tables,
        kv_dequant_scale,
        None,
        None,
        H_Q,
        H_KV,
        D,
        max_input_len,
        quant_mode,
    )

    return q_bw, q_dequant_scale


def _read_and_dequantize_fp8_cache(
    cache, kv_dequant_scale, H_KV, B, block_tables, seq_len, D, is_v=False
):
    """Dequantize FP8 paged KV cache back to float32 using torch ops.

    Args:
        cache: [num_phys_blocks, H_KV, BLOCK_SIZE, D] uint8 on CPU.
        kv_dequant_scale: [num_phys_blocks, H_KV, 4] on CPU.
        H_KV: Number of KV heads.
        B: Batch size.
        block_tables: [B, max_blocks] int32 on CPU.
        seq_len: Sequence length.
        D: Head dim.
        is_v: If True, apply V-specific row reorder.

    Returns:
        Tensor [B, seq_len, H_KV, D] float32 on CPU.
    """
    BLOCK_SIZE = cache.shape[2]
    blocks_needed = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    page_ids = block_tables[:, :blocks_needed].reshape(-1).long()

    # Gather: [B*blocks_needed, H_KV, BLOCK_SIZE, D]
    gathered = cache.index_select(0, page_ids)

    if is_v:
        reordered = halops_v_reorder_indices(
            BLOCK_SIZE,
            cache.device,
            is_fp8=True,
        )
        gathered = gathered[:, :, reordered, :]

    # View as float8_e4m3fn then to float32
    fp8_data = gathered.view(torch.float8_e4m3fn).float()

    # Scale: [B*blocks_needed, H_KV, 1, 1]
    scale_idx = 1 if is_v else 0
    scales = (
        kv_dequant_scale.index_select(0, page_ids)[:, :, scale_idx]
        .unsqueeze(-1)
        .unsqueeze(-1)
    )
    result = fp8_data * scales

    # Reshape to [B, seq_len, H_KV, D]
    result = (
        result.reshape(B, blocks_needed, H_KV, BLOCK_SIZE, D)
        .permute(0, 1, 3, 2, 4)
        .reshape(B, -1, H_KV, D)[:, :seq_len, :, :]
    )
    return result


def read_cache_kv(
    cache_k,
    cache_v,
    block_tables,
    B: int,
    H_KV: int,
    D: int,
    seq_len: int,
    is_fp8: bool,
    kv_dequant_scale=None,
) -> tuple:
    """Extract dense K/V from paged cache. Returns (k_flat, v_flat)."""
    if is_fp8:
        k_orig_bf16 = _read_and_dequantize_fp8_cache(
            cache_k.cpu(),
            kv_dequant_scale.cpu(),
            H_KV,
            B,
            block_tables.cpu(),
            seq_len,
            D,
            is_v=False,
        )
        v_orig_bf16 = _read_and_dequantize_fp8_cache(
            cache_v.cpu(),
            kv_dequant_scale.cpu(),
            H_KV,
            B,
            block_tables.cpu(),
            seq_len,
            D,
            is_v=True,
        )
        k_orig_bf16 = k_orig_bf16.reshape(B, seq_len, H_KV, D).cuda().to(torch.bfloat16)
        v_orig_bf16 = v_orig_bf16.reshape(B, seq_len, H_KV, D).cuda().to(torch.bfloat16)
    else:
        BLOCK_SIZE = cache_k.shape[2]
        blocks_needed = math.ceil(seq_len / BLOCK_SIZE)
        page_ids = block_tables[:, :blocks_needed].reshape(-1).long()
        k_orig_bf16 = (
            cache_k.index_select(0, page_ids)
            .reshape(B, blocks_needed, H_KV, BLOCK_SIZE, D)
            .permute(0, 1, 3, 2, 4)
            .reshape(B, -1, H_KV, D)[:, :seq_len, :, :]
            .contiguous()
        )
        v_perm = halops_v_reorder_indices(
            BLOCK_SIZE,
            cache_v.device,
            is_fp8=False,
        )
        v_orig_bf16 = (
            cache_v.index_select(0, page_ids)[:, :, v_perm, :]
            .reshape(B, blocks_needed, H_KV, BLOCK_SIZE, D)
            .permute(0, 1, 3, 2, 4)
            .reshape(B, -1, H_KV, D)[:, :seq_len, :, :]
            .contiguous()
        )
    return k_orig_bf16, v_orig_bf16


def run_halops_prefill(inp: dict) -> torch.Tensor:
    """
    inp keys:
      q          [total_q_tokens, H_Q, D]   bf16, flattened by adapt_halops_prefill
      k          [total_k_tokens, H_KV, D]  bf16
      v          [total_k_tokens, H_KV, D]  bf16
      cu_seq_q   [B+1]  int32
      cu_seq_k   [B+1]  int32
      attn_out   [total_q_tokens, H_Q, D]   bf16, pre-allocated output
      mask       Optional torch.Tensor or None
    """
    halops.encoder_attention(
        inp["q"],
        inp["k"],
        inp["v"],
        inp["cu_seq_q"],
        inp["cu_seq_k"],
        inp["attn_out"],
        inp.get("mask", None),
    )
    return inp["attn_out"]


def run_halops_decode(inp: dict) -> torch.Tensor:
    """
    inp keys:
      q                [q_token_num, H_Q, D]  bf16  (BF16 path: q_token_num = B)
      cu_seq_q         [B+1]  int32
      seq_lens_kv      [B]    int32
      seq_len_encoder  [B]    int32
      seq_len_decoder  [B]    int32
      cache_k          paged KV cache  bf16
      cache_v          paged KV cache  bf16
      block_tables     [B, max_blocks_per_seq]  int32
      attn_out         [q_token_num, H_Q * D]   bf16, pre-allocated output
      q_dequant_scale  Optional[Tensor]
      kv_dequant_scale Optional[Tensor]
      head_num         int
      kv_head_num      int
      head_dim         int
      max_seq_k        int
      quant_mode       str ("none" or "block_wise_fp8")
    """
    halops.decoder_attention(
        inp["q"],
        inp["cu_seq_q"],
        inp["seq_lens_kv"],
        inp["seq_len_encoder"],
        inp["seq_len_decoder"],
        inp["cache_k"],
        inp["cache_v"],
        inp["block_tables"],
        inp["attn_out"],
        inp.get("q_dequant_scale"),
        inp.get("kv_dequant_scale"),
        inp["head_num"],
        inp["kv_head_num"],
        inp["head_dim"],
        inp["max_seq_k"],
        inp.get("mtp_steps", 0),
        inp.get("quant_mode", "none"),
    )
    return inp["attn_out"]


def run_write_cache_kv_encoder(
    inp: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run HALOps's write_cache_kv_encoder to populate paged KV cache and
    materialize dense Q/K/V buffers for downstream attention.

    inp keys:
      qkv_input          [total_q_tokens, H_Q + 2*H_KV, D]  bf16
      cu_seq_q           [B+1]  int32
      cu_seq_k           [B+1]  int32
      rotary_embs        rotary embedding tensor
      seq_len_encoder    [B]    int32
      seq_len_decoder    [B]    int32  (>0 triggers cache-read path)
      cache_k, cache_v   paged KV cache (bf16 or uint8 for fp8)
      block_tables       [B, max_blocks_per_seq]  int32
      q, k, v            pre-allocated output buffers
      kv_dequant_scale   Optional[Tensor]         (fp8 only)
      head_num, kv_head_num, head_dim   int
      max_seq_len_q, max_seq_len_k, max_input_len int
      quant_mode         str ("none" or "block_wise_fp8")
    """
    halops.write_cache_kv_encoder(
        inp["qkv_input"],
        inp["cu_seq_q"],
        inp["cu_seq_k"],
        inp["rotary_embs"],
        inp["seq_len_encoder"],
        inp["seq_len_decoder"],
        inp["cache_k"],
        inp["cache_v"],
        inp["block_tables"],
        inp["q"],
        inp["k"],
        inp["v"],
        inp.get("kv_dequant_scale"),
        None,
        None,
        int(inp["head_num"]),
        int(inp["kv_head_num"]),
        int(inp["head_dim"]),
        int(inp["max_seq_len_q"]),
        int(inp["max_seq_len_k"]),
        int(inp["max_input_len"]),
        inp["quant_mode"],
    )
    return inp["q"], inp["k"], inp["v"]
