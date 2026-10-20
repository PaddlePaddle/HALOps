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
import os
import sys
import torch

_SGLANG_PATH = os.environ.get("SGLANG_PATH", "/opt/sglang")

try:
    _triton_ops = os.path.join(
        _SGLANG_PATH, "python/sglang/srt/layers/attention/triton_ops"
    )
    if _triton_ops not in sys.path:
        sys.path.insert(0, _triton_ops)
    from decode_attention import decode_attention_fwd
    from prefill_attention import context_attention_fwd

    _SGLANG_AVAILABLE = True
except (ImportError, RuntimeError, Exception):
    _SGLANG_AVAILABLE = False


def _check():
    if not _SGLANG_AVAILABLE:
        raise ImportError(
            f"SGLang not found at {_SGLANG_PATH}. "
            "Set SGLANG_PATH env var to your local sglang clone."
        )


def run_sglang_triton(inp: dict) -> torch.Tensor:
    """
    inp keys:
      q             [total_tokens, H_Q, D]
      k_buffer      [total_kv_tokens, H_KV, D]  flat buffer
      v_buffer      [total_kv_tokens, H_KV, D]
      req_to_token  [B, max_seq_len]
      b_req_idx     [B]
      b_seq_len     [B]
      softmax_scale float
    """
    _check()
    o = torch.empty_like(inp["q"])
    decode_attention_fwd(
        inp["q"],
        inp["k_buffer"],
        inp["v_buffer"],
        o,
        inp["kv_indptr"],
        inp["kv_indices"],
        inp["attn_logits"],
        inp["attn_lse"],
        inp["num_kv_splits"],
        inp["max_kv_splits"],
        inp["softmax_scale"],
        1.0,
        1.0,
    )
    return o


def run_sglang_triton_prefill(inp: dict) -> torch.Tensor:
    """
    inp keys:
      q             [total_tokens, H_Q, D]
      k             [total_tokens, H_KV, D]
      v             [total_tokens, H_KV, D]
      b_start_loc   [B]  start offset of each req in the flat buffer
      b_seq_len     [B]
      max_input_len int
      softmax_scale float
    """
    _check()
    o = torch.empty_like(inp["q"])
    context_attention_fwd(
        inp["q"],
        inp["k"],
        inp["v"],
        o,
        inp["b_start_loc"],
        inp["b_seq_len"],
        inp["max_input_len"],
        is_causal=inp.get("is_causal", True),
        sm_scale=inp["softmax_scale"],
    )
    return o
