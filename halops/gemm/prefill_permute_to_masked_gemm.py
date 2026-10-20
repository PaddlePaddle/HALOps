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

__all__ = ["prefill_permute_to_masked_gemm"]


def prefill_permute_to_masked_gemm(
    x: torch.Tensor,
    scale: torch.Tensor,
    topk_ids: torch.Tensor,
    num_local_experts: int,
    max_token_num: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Permute tokens into padded expert layout for masked GEMM.

    Rearranges tokens from a sorted/topk-selected layout into
    ``[E, max_token_num, hidden]`` padded format suitable for group_masked_gemm,
    and simultaneously converts scale factors from linear row-major into CUTLASS
    SfAtom layout.

    Args:
        x: [num_tokens, hidden] CUDA tensor. dtype float8_e4m3fn or bfloat16.
        scale: [num_tokens, packed_hidden_scale] CUDA tensor.
            dtype int32 (broadcast mode): packed_hidden_scale = hidden/128/4,
            each int32 holds 4 per-group-128 scale bytes.
            dtype uint8 (direct mode): packed_hidden_scale = hidden/128*4,
            each 4 consecutive bytes form one uint32 store.
        topk_ids: [num_tokens, topk] int64. Expert assignment per token;
            -1 means unassigned/padding.
        num_local_experts: Number of local experts (E).
        max_token_num: Maximum tokens per expert. Must be >= 128 and
            divisible by 128. Must also be large enough to hold the tokens
            routed to a single expert; the kernel does not clamp, so an
            expert receiving more than ``max_token_num`` tokens writes out
            of bounds (illegal memory access). Size it with headroom above
            ``num_tokens * topk / num_local_experts``.

    Returns:
        A tuple ``(permute_x, output_scale, permuted_indice_map, token_nums_per_expert)``:
        - ``permute_x``: [E * max_token_num, hidden], same dtype as x.
        - ``output_scale``: [E, max_token_num, kb_dim] uint8, SfAtom layout (128-byte aligned).
            kb_dim = hidden / 32 (both scale modes).
        - ``permuted_indice_map``: [num_tokens, topk] int32, maps (token, slot) to expert-local offset.
        - ``token_nums_per_expert``: [E, 1] int32, count of tokens assigned to each expert.

    Example:
        >>> from halops import prefill_permute_to_masked_gemm
        >>>
        >>> num_tokens, hidden, topk = 1024, 2048, 8
        >>> num_local_experts = 16
        >>> # average tokens per expert is num_tokens * topk / E = 512, so
        >>> # max_token_num must leave headroom for routing imbalance
        >>> max_token_num = 1024
        >>> x = torch.randn(num_tokens, hidden,
        ...                 dtype=torch.bfloat16, device="cuda")
        >>> # broadcast mode: int32 scale, hidden / 128 / 4 columns
        >>> scale = torch.zeros(num_tokens, hidden // 128 // 4,
        ...                     dtype=torch.int32, device="cuda")
        >>> topk_ids = torch.randint(0, num_local_experts,
        ...                          (num_tokens, topk),
        ...                          dtype=torch.int64, device="cuda")
        >>> permute_x, out_scale, indice_map, token_nums = (
        ...     prefill_permute_to_masked_gemm(
        ...         x, scale, topk_ids, num_local_experts, max_token_num))
    """
    return halops_kernel.prefill_permute_to_masked_gemm(
        x,
        scale,
        topk_ids,
        num_local_experts,
        max_token_num,
    )
