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

__all__ = ["write_cache_kv_encoder"]


def write_cache_kv_encoder(
    qkv_input: torch.Tensor,
    cu_seq_q: torch.Tensor,
    cu_seq_k: torch.Tensor,
    rotary_embs: torch.Tensor,
    seq_len_encoder: torch.Tensor,
    seq_len_decoder: torch.Tensor,
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    block_tables: torch.Tensor,
    q_input: torch.Tensor,
    k_input: torch.Tensor,
    v_input: torch.Tensor,
    kv_dequant_scale: Optional[torch.Tensor] = None,
    q_norm_gamma: Optional[torch.Tensor] = None,
    k_norm_gamma: Optional[torch.Tensor] = None,
    head_num: int = 0,
    kv_head_num: int = 0,
    head_dim: int = 128,
    max_seq_len_q: int = 0,
    max_seq_len_k: int = 0,
    max_input_len: int = 0,
    cache_quant_type: str = "none",
) -> None:
    """Split fused QKV, apply RoPE and optional RMSNorm to Q/K, output contiguous
    Q/K/V, and simultaneously write K/V to paged KV cache.

    Supports BF16 cache and block-wise FP8 cache.

    Args:
        qkv_input: [total_q_tokens, head_num + 2*kv_head_num, head_dim] BF16.
        cu_seq_q: [batch_size+1] int32, Q token prefix-sum.
        cu_seq_k: [batch_size+1] int32, K/V token prefix-sum.
        rotary_embs: RoPE cos/sin table, first half is cos, second half is sin.
        seq_len_encoder: [batch_size] int32, encoder/prefill token count.
        seq_len_decoder: [batch_size] int32, existing decode prefix length.
        cache_k: paged K cache, BF16 or FP8/uint8.
        cache_v: paged V cache, BF16 or FP8/uint8.
        block_tables: [batch_size, max_blocks_per_seq] int32.
        q_input: [total_q_tokens, head_num, head_dim] BF16 output, Q after RoPE.
        k_input: [total_k_tokens, kv_head_num, head_dim] BF16 output.
        v_input: [total_k_tokens, kv_head_num, head_dim] BF16 output.
        kv_dequant_scale: Required for FP8 cache path, K/V block-wise dequant scale.
        q_norm_gamma: Optional Q RMSNorm gamma.
        k_norm_gamma: Optional K RMSNorm gamma.
        head_num: Number of Q heads.
        kv_head_num: Number of KV heads.
        head_dim: Per-head dimension (primarily targets 128).
        max_seq_len_q: Max Q length in current batch.
        max_seq_len_k: Max K length in current batch.
        max_input_len: Max RoPE table length.
        cache_quant_type: "none" for BF16 cache; "block_wise_fp8" for FP8 cache.

    Returns:
        None. Writes the RoPE-applied Q/K/V into q_input/k_input/v_input
        and appends K/V into the paged cache_k/cache_v in-place.

    Example:
        >>> import torch
        >>> from halops import write_cache_kv_encoder
        >>>
        >>> head_num, kv_head_num, head_dim = 14, 1, 128
        >>> batch, seq_len, block_size = 1, 128, 128
        >>> max_seq_len = 4096
        >>> num_blocks = batch * max_seq_len // block_size
        >>> total_q = batch * seq_len
        >>> qkv = torch.randn(total_q, head_num + 2 * kv_head_num, head_dim,
        ...                   dtype=torch.bfloat16, device="cuda")
        >>> cu_seq_q = torch.tensor([0, seq_len], dtype=torch.int32, device="cuda")
        >>> cu_seq_k = cu_seq_q.clone()
        >>> # identity RoPE table: cos=1, sin=0. The BF16 encoder applies
        >>> # partial NeoX RoPE with rope_dim = head_dim // 4.
        >>> rotary_embs = torch.zeros(2, max_seq_len, head_dim // 4,
        ...                           dtype=torch.float32, device="cuda")
        >>> rotary_embs[0] = 1
        >>> seq_len_encoder = torch.tensor([seq_len], dtype=torch.int32, device="cuda")
        >>> seq_len_decoder = torch.zeros(batch, dtype=torch.int32, device="cuda")
        >>> cache_k = torch.zeros(num_blocks, kv_head_num, block_size, head_dim,
        ...                       dtype=torch.bfloat16, device="cuda")
        >>> cache_v = torch.zeros_like(cache_k)
        >>> block_tables = torch.arange(
        ...     num_blocks, dtype=torch.int32, device="cuda"
        ... ).reshape(batch, max_seq_len // block_size)
        >>> q_out = torch.zeros(total_q, head_num, head_dim,
        ...                     dtype=torch.bfloat16, device="cuda")
        >>> k_out = torch.zeros(total_q, kv_head_num, head_dim,
        ...                     dtype=torch.bfloat16, device="cuda")
        >>> v_out = torch.zeros_like(k_out)
        >>> write_cache_kv_encoder(
        ...     qkv, cu_seq_q, cu_seq_k, rotary_embs, seq_len_encoder,
        ...     seq_len_decoder, cache_k, cache_v, block_tables,
        ...     q_out, k_out, v_out,
        ...     head_num=head_num, kv_head_num=kv_head_num, head_dim=head_dim,
        ...     max_seq_len_q=seq_len, max_seq_len_k=seq_len,
        ...     max_input_len=max_seq_len, cache_quant_type="none",
        ... )
    """
    halops_kernel.write_cache_kv_encoder(
        qkv_input,
        cu_seq_q,
        cu_seq_k,
        rotary_embs,
        seq_len_encoder,
        seq_len_decoder,
        cache_k,
        cache_v,
        block_tables,
        q_input,
        k_input,
        v_input,
        kv_dequant_scale,
        q_norm_gamma,
        k_norm_gamma,
        head_num,
        kv_head_num,
        head_dim,
        max_seq_len_q,
        max_seq_len_k,
        max_input_len,
        cache_quant_type,
    )
