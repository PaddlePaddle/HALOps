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
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def run_golden(inp: dict) -> torch.Tensor:
    """Reference attention via torch SDPA on the fused backends.

    Computes softmax(Q K^T / sqrt(d)) V in bf16 using torch's fused SDPA
    kernels (flash, falling back to cudnn), which stream over S_KV instead of
    materializing the full [B, H, S_Q, S_KV] score matrix. The MATH backend is
    intentionally excluded: it is a small-op composition that OOMs on large
    S_KV, whereas the kernels under test are all fused. The fused result is a
    neutral, environment-independent correctness baseline (golden).

    inp keys (same layout as the fa/cudnn adapters):
      q [B, S_Q, H_Q, D], k [B, S_KV, H_KV, D], v [B, S_KV, H_KV, D],
      softmax_scale, causal. GQA (H_Q != H_KV) handled via enable_gqa.
    """
    q = inp["q"].transpose(1, 2)  # [B, H_Q,  S_Q,  D]
    k = inp["k"].transpose(1, 2)  # [B, H_KV, S_KV, D]
    v = inp["v"].transpose(1, 2)  # [B, H_KV, S_KV, D]
    with sdpa_kernel(
        [
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.CUDNN_ATTENTION,
        ]
    ):
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            scale=inp["softmax_scale"],
            is_causal=inp["causal"],
            enable_gqa=True,
        )
    return out.transpose(1, 2)  # [B, S_Q, H_Q, D]


def run_golden_chunked(inp: dict) -> torch.Tensor:
    """Torch-SDPA reference for chunked prefill.

    Attention mask (Q pos i in chunk, KV pos j in prefix+chunk):
        allowed iff j <= P + i
    i.e. full prefix visible + causal within chunk. This is not vanilla causal,
    so we cannot reuse `run_golden` (which only exposes `is_causal`).

    Returns output in varlen layout [B*C, H_Q, D] to match fa/fi/halops.
    """
    B = inp["batch_size"]
    C = inp["chunk_len"]
    P = inp["prefix_len"]
    H_Q = inp["num_q_heads"]
    H_KV = inp["num_kv_heads"]
    D = inp["head_dim"]

    # Varlen -> batched.
    q = inp["q"].reshape(B, C, H_Q, D)  # [B, C,     H_Q,  D]
    k = inp["k"].reshape(B, C + P, H_KV, D)  # [B, C + P, H_KV, D]
    v = inp["v"].reshape(B, C + P, H_KV, D)  # [B, C + P, H_KV, D]

    # SDPA expects [B, H, S, D].
    q_t = q.transpose(1, 2)
    k_t = k.transpose(1, 2)
    v_t = v.transpose(1, 2)

    # [C, C+P] boolean mask: mask[i, j] = (j <= P + i).
    device = q.device
    i_idx = torch.arange(C, device=device).unsqueeze(1)  # [C, 1]
    j_idx = torch.arange(C + P, device=device).unsqueeze(0)  # [1, C+P]
    attn_mask = j_idx <= (i_idx + P)  # [C, C+P]

    # FLASH backend rejects arbitrary attn_mask; use EFFICIENT / CUDNN.
    with sdpa_kernel(
        [
            SDPBackend.EFFICIENT_ATTENTION,
            SDPBackend.CUDNN_ATTENTION,
        ]
    ):
        out = F.scaled_dot_product_attention(
            q_t,
            k_t,
            v_t,
            attn_mask=attn_mask,
            is_causal=False,
            scale=inp["softmax_scale"],
            enable_gqa=True,
        )
    # [B, H_Q, C, D] -> [B, C, H_Q, D] -> [B*C, H_Q, D]
    return out.transpose(1, 2).contiguous().reshape(B * C, H_Q, D)
