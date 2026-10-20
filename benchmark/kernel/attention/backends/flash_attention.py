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

try:
    import flash_attn as _fa2_mod
    from flash_attn import flash_attn_func as _fa2_func

    _FA2_AVAILABLE = True
except ImportError:
    _fa2_mod = None
    _FA2_AVAILABLE = False

try:
    from flash_attn.cute.interface import flash_attn_func as _fa4_func

    _FA4_AVAILABLE = True
except ImportError:
    _FA4_AVAILABLE = False


def run_fa2_prefill(inp: dict) -> torch.Tensor:
    """inp keys: q [B,S_Q,H,D], k [B,S_KV,H,D], v, softmax_scale, causal"""
    if not _FA2_AVAILABLE:
        raise ImportError("flash_attn not installed. pip install flash-attn")
    return _fa2_func(
        inp["q"],
        inp["k"],
        inp["v"],
        softmax_scale=inp["softmax_scale"],
        causal=inp["causal"],
    )


def run_fa2_decode(inp: dict) -> torch.Tensor:
    """decode: s_q=1, reuses the prefill path."""
    return run_fa2_prefill(inp)


def run_fa2_prefill_varlen(inp: dict) -> torch.Tensor:
    """FA2 varlen prefill.

    inp keys: q [total_q, H_Q, D], k [total_k, H_KV, D], v, cu_seq_q [B+1],
              cu_seq_k [B+1], max_seq_len_q, max_seq_len_k, softmax_scale, causal.
    """
    if not _FA2_AVAILABLE:
        raise ImportError("flash_attn not installed. pip install flash-attn")
    from flash_attn import flash_attn_varlen_func

    return flash_attn_varlen_func(
        inp["q"],
        inp["k"],
        inp["v"],
        cu_seqlens_q=inp["cu_seq_q"],
        cu_seqlens_k=inp["cu_seq_k"],
        max_seqlen_q=inp["max_seq_len_q"],
        max_seqlen_k=inp["max_seq_len_k"],
        softmax_scale=inp["softmax_scale"],
        causal=inp["causal"],
    )


def run_fa4_prefill(inp: dict) -> torch.Tensor:
    """inp keys: q [B,S_Q,H,D], k [B,S_KV,H,D], v, softmax_scale, causal"""
    if not _FA4_AVAILABLE:
        raise ImportError(
            "flash_attn.cute not available. pip install nvidia-cutlass-dsl==4.1.0"
        )
    out = _fa4_func(
        inp["q"],
        inp["k"],
        inp["v"],
        softmax_scale=inp["softmax_scale"],
        causal=inp["causal"],
        num_splits=inp.get("num_splits", 0),
    )
    return out[0] if isinstance(out, tuple) else out


def run_fa4_decode(inp: dict) -> torch.Tensor:
    """decode: s_q=1, reuses the prefill path."""
    return run_fa4_prefill(inp)


def run_fa4_prefill_varlen(inp: dict) -> torch.Tensor:
    """FA4 varlen prefill.

    inp keys: q [total_q, H_Q, D], k [total_k, H_KV, D], v, cu_seq_q [B+1],
              cu_seq_k [B+1], max_seq_len_q, max_seq_len_k, softmax_scale, causal,
              num_splits (optional, default 1).
    """
    if not _FA4_AVAILABLE:
        raise ImportError(
            "flash_attn.cute not available. pip install nvidia-cutlass-dsl==4.1.0"
        )
    from flash_attn.cute.interface import flash_attn_varlen_func

    out = flash_attn_varlen_func(
        inp["q"],
        inp["k"],
        inp["v"],
        cu_seqlens_q=inp["cu_seq_q"],
        cu_seqlens_k=inp["cu_seq_k"],
        max_seqlen_q=inp["max_seq_len_q"],
        max_seqlen_k=inp["max_seq_len_k"],
        softmax_scale=inp["softmax_scale"],
        causal=inp["causal"],
        num_splits=inp.get("num_splits", 1),
    )
    return out[0] if isinstance(out, tuple) else out


def run_fa4_paged_decode(inp: dict) -> torch.Tensor:
    """FA4 paged KV decode via flash_attn_varlen_func with page_table."""
    if not _FA4_AVAILABLE:
        raise ImportError(
            "flash_attn.cute not available. pip install nvidia-cutlass-dsl==4.1.0"
        )
    from flash_attn.cute.interface import flash_attn_varlen_func

    out = flash_attn_varlen_func(
        inp["q"],
        inp["k_cache"],
        inp["v_cache"],
        cu_seqlens_q=inp["cu_seqlens_q"],
        max_seqlen_q=inp["max_seqlen_q"],
        max_seqlen_k=inp.get("max_seqlen_k"),
        seqused_k=inp["seqused_k"],
        page_table=inp["page_table"],
        softmax_scale=inp.get("softmax_scale"),
        causal=inp.get("causal", True),
        num_splits=inp.get("num_splits", 1),
        pack_gqa=inp.get("pack_gqa"),
    )
    return out[0] if isinstance(out, tuple) else out
