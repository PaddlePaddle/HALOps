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

__all__ = ["encoder_attention"]


def encoder_attention(
    q_input: torch.Tensor,
    k_input: torch.Tensor,
    v_input: torch.Tensor,
    cu_seq_q: torch.Tensor,
    cu_seq_k: torch.Tensor,
    attn_out: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> None:
    """Variable-length BF16 attention forward for prefill with GQA and
    optional causal/custom mask.

    Args:
        q_input: [total_q_tokens, head_num, head_dim] BF16.
        k_input: [total_k_tokens, kv_head_num, head_dim] BF16.
        v_input: [total_k_tokens, kv_head_num, head_dim] BF16.
        cu_seq_q: [batch_size+1] int32, Q token prefix-sum.
        cu_seq_k: [batch_size+1] int32, K token prefix-sum.
        attn_out: [total_q_tokens, head_num, head_dim] BF16, output tensor
            allocated by caller. Written in-place.
        mask: Optional int32 mask; when passed, uses CustomMask template branch.
            When ``None``, the kernel applies the default causal mask (bottom-right
            aligned) per sequence.

    Returns:
        None. The result is written in-place to ``attn_out``.

    Example:
        >>> import torch
        >>> from halops import encoder_attention
        >>>
        >>> head_num, kv_head_num, head_dim = 56, 4, 128
        >>> seq_len = 1024
        >>> cu_seq_q = torch.tensor([0, seq_len], dtype=torch.int32, device="cuda")
        >>> cu_seq_k = torch.tensor([0, seq_len], dtype=torch.int32, device="cuda")
        >>> q = torch.randn(seq_len, head_num, head_dim,
        ...                 dtype=torch.bfloat16, device="cuda")
        >>> k = torch.randn(seq_len, kv_head_num, head_dim,
        ...                 dtype=torch.bfloat16, device="cuda")
        >>> v = torch.randn(seq_len, kv_head_num, head_dim,
        ...                 dtype=torch.bfloat16, device="cuda")
        >>> attn_out = torch.empty(seq_len, head_num, head_dim,
        ...                        dtype=torch.bfloat16, device="cuda")
        >>> encoder_attention(q, k, v, cu_seq_q, cu_seq_k, attn_out)
    """
    halops_kernel.encoder_attention(
        q_input,
        k_input,
        v_input,
        cu_seq_q,
        cu_seq_k,
        attn_out,
        mask,
    )
