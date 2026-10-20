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

from typing import Tuple

import torch

import halops.halops_kernel as halops_kernel

__all__ = ["get_qk_token"]


def get_qk_token(
    seq_lens_encoder: torch.Tensor,
    seq_lens_decoder: torch.Tensor,
    cu_seq_q: torch.Tensor,
    cu_seq_k: torch.Tensor,
    actual_cu_seq_k: torch.Tensor,
    seq_lens_kv: torch.Tensor,
    kv_head_num: int = 0,
) -> Tuple[torch.Tensor, ...]:
    """Compute actual_cu_seq_k, seq_lens_kv, and return max/total token statistics
    for subsequent attention kernels.

    Args:
        seq_lens_encoder: [batch_size] int32, encoder token count per sample.
        seq_lens_decoder: [batch_size] int32, existing decode prefix length per sample.
        cu_seq_q: [batch_size+1] int32, Q token prefix-sum.
        cu_seq_k: [batch_size+1] int32, reserved for interface compatibility.
        actual_cu_seq_k: [batch_size+1] int32 output, actual K/V prefix-sum.
        seq_lens_kv: [max_batch_size] int32 output, per-sample KV length for decode.
        kv_head_num: Number of KV heads.

    Returns:
        A tuple containing a single CPU tensor with contents
        ``[max_enc, max_dec, total_q, total_k]``.

    Example:
        >>> import torch
        >>> from halops import get_qk_token
        >>>
        >>> kv_head_num = 1
        >>> batch, max_batch = 2, 8
        >>> seq_lens_encoder = torch.tensor(
        ...     [128, 0], dtype=torch.int32, device="cuda"
        ... )
        >>> seq_lens_decoder = torch.tensor(
        ...     [0, 512], dtype=torch.int32, device="cuda"
        ... )
        >>> cu_seq_q = torch.tensor([0, 128, 129], dtype=torch.int32, device="cuda")
        >>> cu_seq_k = torch.zeros(batch + 1, dtype=torch.int32, device="cuda")
        >>> actual_cu_seq_k = torch.zeros(
        ...     batch + 1, dtype=torch.int32, device="cuda"
        ... )
        >>> seq_lens_kv = torch.zeros(max_batch, dtype=torch.int32, device="cuda")
        >>> (max_token_num,) = get_qk_token(
        ...     seq_lens_encoder, seq_lens_decoder, cu_seq_q, cu_seq_k,
        ...     actual_cu_seq_k, seq_lens_kv, kv_head_num,
        ... )
        >>> max_token_num.tolist()  # [max_enc, max_dec, total_q, total_k]
        [128, 513, 129, 129]
    """
    return halops_kernel.get_qk_token(
        seq_lens_encoder,
        seq_lens_decoder,
        cu_seq_q,
        cu_seq_k,
        actual_cu_seq_k,
        seq_lens_kv,
        kv_head_num,
    )
