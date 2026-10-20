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

__all__ = ["write_cache_kv_decoder"]


def write_cache_kv_decoder(
    qkv_input: torch.Tensor,
    cu_seq_q: torch.Tensor,
    seq_len_encoder: torch.Tensor,
    seq_len_decoder: torch.Tensor,
    rotary_embs: torch.Tensor,
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    block_tables: torch.Tensor,
    kv_dequant_scale: Optional[torch.Tensor] = None,
    q_norm_gamma: Optional[torch.Tensor] = None,
    k_norm_gamma: Optional[torch.Tensor] = None,
    head_num: int = 0,
    kv_head_num: int = 0,
    head_dim: int = 128,
    max_input_len: int = 0,
    cache_quant_type: str = "none",
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Process the current decode token fused QKV, generate Q output for decode
    attention, and append K/V to paged KV cache.

    Args:
        qkv_input: [q_token_num, head_num + 2*kv_head_num, head_dim] BF16.
        cu_seq_q: [batch_size+1] int32, decode Q token prefix-sum.
        seq_len_encoder: [batch_size] int32; samples with >0 skip decoder-only cache write.
        seq_len_decoder: [batch_size] int32, existing decode prefix length before write.
        rotary_embs: RoPE cos/sin table.
        cache_k: paged K cache, BF16 or FP8/uint8.
        cache_v: paged V cache, BF16 or FP8/uint8.
        block_tables: [batch_size, max_blocks_per_seq] int32.
        kv_dequant_scale: Required for FP8 cache path, K/V block-wise dequant scale.
        q_norm_gamma: Optional Q RMSNorm gamma.
        k_norm_gamma: Optional K RMSNorm gamma.
        head_num: Number of Q heads.
        kv_head_num: Number of KV heads.
        head_dim: Per-head dimension (primarily targets 128).
        max_input_len: Maximum length of the RoPE table.
        cache_quant_type: ``"none"`` for BF16 cache; ``"block_wise_fp8"`` for FP8 cache.

    Returns:
        A tuple ``(q_output, q_dequant_scale)`` where:
        - ``q_output``: [batch, kv_head_num, 64, head_dim], BF16 for BF16 path
          or float8_e4m3fn for FP8 path.
        - ``q_dequant_scale``: [batch, kv_head_num] float32. In the FP8 path it
          holds the Q dequantization scale. In the BF16 path it is still
          allocated (never ``None``) but left uninitialized and must not be used.

    Example:
        >>> import torch
        >>> from halops import write_cache_kv_decoder
        >>>
        >>> head_num, kv_head_num, head_dim = 14, 1, 128
        >>> batch, prefix_len, block_size = 1, 128, 128
        >>> max_input_len = 4096
        >>> num_blocks = batch * max_input_len // block_size
        >>> # one new token per sequence for the current decode step
        >>> qkv = torch.randn(batch, head_num + 2 * kv_head_num, head_dim,
        ...                   dtype=torch.bfloat16, device="cuda")
        >>> cu_seq_q = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
        >>> seq_len_encoder = torch.zeros(batch, dtype=torch.int32, device="cuda")
        >>> seq_len_decoder = torch.tensor(
        ...     [prefix_len], dtype=torch.int32, device="cuda"
        ... )
        >>> rotary_embs = torch.zeros(2, max_input_len, head_dim // 2,
        ...                           dtype=torch.float32, device="cuda")
        >>> rotary_embs[0] = 1  # identity RoPE
        >>> cache_k = torch.zeros(num_blocks, kv_head_num, block_size, head_dim,
        ...                       dtype=torch.bfloat16, device="cuda")
        >>> cache_v = torch.zeros_like(cache_k)
        >>> block_tables = torch.arange(
        ...     num_blocks, dtype=torch.int32, device="cuda"
        ... ).reshape(batch, max_input_len // block_size)
        >>> q_output, q_dequant_scale = write_cache_kv_decoder(
        ...     qkv, cu_seq_q, seq_len_encoder, seq_len_decoder, rotary_embs,
        ...     cache_k, cache_v, block_tables,
        ...     head_num=head_num, kv_head_num=kv_head_num, head_dim=head_dim,
        ...     max_input_len=max_input_len, cache_quant_type="none",
        ... )
        >>> q_output.shape
        torch.Size([1, 1, 64, 128])
    """
    return halops_kernel.write_cache_kv_decoder(
        qkv_input,
        cu_seq_q,
        seq_len_encoder,
        seq_len_decoder,
        rotary_embs,
        cache_k,
        cache_v,
        block_tables,
        kv_dequant_scale,
        q_norm_gamma,
        k_norm_gamma,
        head_num,
        kv_head_num,
        head_dim,
        max_input_len,
        cache_quant_type,
    )
