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


def run_cudnn(inp: dict) -> torch.Tensor:
    """
    inp keys: q [B,S_Q,H_Q,D], k [B,S_KV,H_KV,D], v, softmax_scale, causal
    cuDNN SDPA requires [B, H, S, D] layout; GQA is handled via enable_gqa.
    """
    q = inp["q"].transpose(1, 2)  # [B, H_Q,  S_Q,  D]
    k = inp["k"].transpose(1, 2)  # [B, H_KV, S_KV, D]
    v = inp["v"].transpose(1, 2)  # [B, H_KV, S_KV, D]
    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            scale=inp["softmax_scale"],
            is_causal=inp["causal"],
            enable_gqa=True,
        )
    return out.transpose(1, 2)  # [B, S_Q, H_Q, D]
