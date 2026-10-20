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

__all__ = ["decoder_attention"]


def decoder_attention(
    q_input: torch.Tensor,
    cu_seq_q: torch.Tensor,
    seq_lens_kv: torch.Tensor,
    seq_len_encoder: torch.Tensor,
    seq_len_decoder: torch.Tensor,
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    block_tables: torch.Tensor,
    attn_out: torch.Tensor,
    q_dequant_scale: Optional[torch.Tensor] = None,
    kv_dequant_scale: Optional[torch.Tensor] = None,
    head_num: int = 0,
    kv_head_num: int = 0,
    head_dim: int = 128,
    max_seq_k: int = 0,
    mtp_steps: int = 0,
    cache_quant_type: str = "none",
) -> None:
    """Compute paged-KV attention for the decode stage.

    Reads the current-step query together with the paged K/V cache, attends over
    each sample's visible KV range located through block_tables, and writes
    the attention result into attn_out. Supports grouped-query attention
    (GQA), multi-token prediction (MTP), and both a BF16 cache and a block-wise
    FP8 cache.

    Args:
        q_input (Tensor): Query, shape [batch, kv_head_num, 64, head_dim].
            Dtype is bfloat16 when cache_quant_type is "none" and
            float8_e4m3fn when it is "block_wise_fp8". The 64 rows hold
            the GQA-folded query heads padded to a fixed width.
        cu_seq_q (Tensor): Prefix-sum of decode query tokens per sample, shape
            [batch + 1], dtype int32.
        seq_lens_kv (Tensor): Total visible KV length per sample, shape
            [batch], dtype int32.
        seq_len_encoder (Tensor): Encoder/prefill length per sample, shape
            [batch], dtype int32.
        seq_len_decoder (Tensor): Decode prefix length per sample, shape
            [batch], dtype int32.
        cache_k (Tensor): Paged K cache, shape
            [num_blocks, kv_head_num, block_size, head_dim]. Dtype is
            bfloat16 for the "none" cache and uint8 (packed FP8) for
            the "block_wise_fp8" cache.
        cache_v (Tensor): Paged V cache, same shape and dtype as cache_k.
        block_tables (Tensor): Mapping from each sample to its physical cache
            blocks, shape [batch, max_blocks_per_seq], dtype int32.
        attn_out (Tensor): Output buffer written in place, shape
            [q_token_num, head_num * head_dim], dtype bfloat16.
        q_dequant_scale (Tensor, optional): Query dequantization scale used only
            by the FP8 path, shape [batch, kv_head_num], dtype float32.
            Ignored when cache_quant_type is "none". Default: None.
        kv_dequant_scale (Tensor, optional): Block-wise K/V dequantization scale
            used only by the FP8 path, shape [num_blocks, kv_head_num, 4],
            dtype float32. Ignored when cache_quant_type is "none".
            Default: None.
        head_num (int): Number of query heads.
        kv_head_num (int): Number of key/value heads.
        head_dim (int): Per-head dimension. Default: 128.
        max_seq_k (int): Maximum visible KV length across the current batch.
        mtp_steps (int, optional): Number of extra multi-token-prediction tokens;
            0 means ordinary single-token decode. Default: 0.
        cache_quant_type (str, optional): KV cache quantization mode. Supported
            values are "none" (BF16 cache) and "block_wise_fp8" (block-wise
            FP8 cache). Default: "none".

    Returns:
        None. The attention result is written in place into attn_out.

    Example:
        >>> import torch
        >>> from halops import (
        ...     write_cache_kv_encoder,
        ...     write_cache_kv_decoder,
        ...     decoder_attention,
        ... )
        >>>
        >>> head_num, kv_head_num, head_dim = 14, 1, 128
        >>> batch, seq_len_enc, block_size = 1, 128, 128
        >>> max_seq_len = 4096
        >>> num_blocks = batch * max_seq_len // block_size
        >>> total_kv = seq_len_enc + 1  # prefill tokens + 1 decode token
        >>>
        >>> # Shared paged KV cache and block table.
        >>> cache_k = torch.zeros(num_blocks, kv_head_num, block_size, head_dim,
        ...                       dtype=torch.bfloat16, device="cuda")
        >>> cache_v = torch.zeros_like(cache_k)
        >>> block_tables = torch.arange(
        ...     num_blocks, dtype=torch.int32, device="cuda"
        ... ).reshape(batch, max_seq_len // block_size)
        >>> # Identity RoPE table (cos=1, sin=0) so Q/K pass through unchanged.
        >>> rotary_embs = torch.zeros(2, max_seq_len, head_dim // 2,
        ...                           dtype=torch.float32, device="cuda")
        >>> rotary_embs[0] = 1
        >>>
        >>> # Step 1: prefill. Write the encoder Q/K/V and populate the KV cache.
        >>> qkv_enc = torch.randn(batch * seq_len_enc, head_num + 2 * kv_head_num,
        ...                       head_dim, dtype=torch.bfloat16, device="cuda")
        >>> cu_seq_enc = torch.tensor([0, seq_len_enc], dtype=torch.int32,
        ...                           device="cuda")
        >>> seq_len_enc_t = torch.tensor([seq_len_enc], dtype=torch.int32,
        ...                              device="cuda")
        >>> seq_len_dec_t = torch.zeros(batch, dtype=torch.int32, device="cuda")
        >>> q_enc = torch.zeros(batch * seq_len_enc, head_num, head_dim,
        ...                     dtype=torch.bfloat16, device="cuda")
        >>> k_enc = torch.zeros(batch * seq_len_enc, kv_head_num, head_dim,
        ...                     dtype=torch.bfloat16, device="cuda")
        >>> v_enc = torch.zeros_like(k_enc)
        >>> write_cache_kv_encoder(
        ...     qkv_enc, cu_seq_enc, cu_seq_enc, rotary_embs, seq_len_enc_t,
        ...     seq_len_dec_t, cache_k, cache_v, block_tables, q_enc, k_enc, v_enc,
        ...     head_num=head_num, kv_head_num=kv_head_num, head_dim=head_dim,
        ...     max_seq_len_q=seq_len_enc, max_seq_len_k=seq_len_enc,
        ...     max_input_len=max_seq_len, cache_quant_type="none",
        ... )
        >>>
        >>> # Step 2: decode. Write the current token's K/V and build the folded
        >>> # q_output of shape [batch, kv_head_num, 64, head_dim].
        >>> qkv_dec = torch.randn(batch, head_num + 2 * kv_head_num, head_dim,
        ...                       dtype=torch.bfloat16, device="cuda")
        >>> cu_seq_q = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
        >>> seq_len_enc_t2 = torch.zeros(batch, dtype=torch.int32, device="cuda")
        >>> seq_len_dec_t2 = torch.tensor([seq_len_enc], dtype=torch.int32,
        ...                               device="cuda")
        >>> q_output, _ = write_cache_kv_decoder(
        ...     qkv_dec, cu_seq_q, seq_len_enc_t2, seq_len_dec_t2, rotary_embs,
        ...     cache_k, cache_v, block_tables,
        ...     head_num=head_num, kv_head_num=kv_head_num, head_dim=head_dim,
        ...     max_input_len=max_seq_len, cache_quant_type="none",
        ... )
        >>>
        >>> # Step 3: decode attention over the paged cache.
        >>> seq_lens_kv = torch.tensor([total_kv], dtype=torch.int32, device="cuda")
        >>> attn_out = torch.zeros(batch, head_num * head_dim,
        ...                        dtype=torch.bfloat16, device="cuda")
        >>> decoder_attention(
        ...     q_output, cu_seq_q, seq_lens_kv, seq_len_enc_t2, seq_len_dec_t2,
        ...     cache_k, cache_v, block_tables, attn_out,
        ...     head_num=head_num, kv_head_num=kv_head_num, head_dim=head_dim,
        ...     max_seq_k=total_kv, cache_quant_type="none",
        ... )
    """
    halops_kernel.decoder_attention(
        q_input,
        cu_seq_q,
        seq_lens_kv,
        seq_len_encoder,
        seq_len_decoder,
        cache_k,
        cache_v,
        block_tables,
        attn_out,
        q_dequant_scale,
        kv_dequant_scale,
        head_num,
        kv_head_num,
        head_dim,
        max_seq_k,
        mtp_steps,
        cache_quant_type,
    )
