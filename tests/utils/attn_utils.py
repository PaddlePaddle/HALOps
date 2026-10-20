# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared numpy/torch reference helpers for attention tests.

Collects the quantization and reference-attention utilities that were previously
duplicated across the decoder/encoder attention test files.
"""

import math

import numpy as np
import torch

HEAD_DIM = 128
BLOCK_SIZE = 128
FP8_MAX = 448.0


def to_fp8(x):
    return x.clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)


def quant_fp8_per128_tokens(x_np, block_size=BLOCK_SIZE):
    """FP8 quantization with per-(128-token) block absmax along the token axis.

    Args:
        x_np: [token_num, num_heads, head_dim] float32 array.
        block_size: token block size for grouping absmax.

    Returns:
        Tuple ``(x_dequant, scale)`` where ``x_dequant`` has the same shape as
        ``x_np`` and ``scale`` has shape [num_groups, num_heads].
    """
    token_num, num_heads, head_dim = x_np.shape
    groups = math.ceil(token_num / block_size)
    pad = groups * block_size - token_num
    if pad:
        x_np = np.concatenate([x_np, np.zeros([pad, num_heads, head_dim], np.float32)])
    x_g = x_np.reshape(groups, block_size, num_heads, head_dim)
    absmax = np.abs(x_g).max(axis=(1, 3), keepdims=True).clip(min=1e-8)
    scale = absmax / FP8_MAX
    x_deq = (
        to_fp8(
            torch.from_numpy((x_g / scale).clip(-FP8_MAX, FP8_MAX).astype(np.float32))
        )
        .float()
        .numpy()
        * scale
    )
    return x_deq.reshape(groups * block_size, num_heads, head_dim)[
        :token_num
    ], scale.reshape(groups, num_heads)


def quant_fp8_q(q_4d, kv_head_num):
    """Quantize decode Q so that each (batch, kv_head) group shares one scale.

    Args:
        q_4d: [batch, q_per_seq, head_num, head_dim] float32 array.
        kv_head_num: number of KV heads; head_num must be divisible by it.

    Returns:
        Tuple ``(q_dequant, scale)`` where ``q_dequant`` matches ``q_4d`` shape and
        ``scale`` has shape [batch, kv_head_num].
    """
    bs, qps, hnum, hdim = q_4d.shape
    gqa = hnum // kv_head_num
    q_r = q_4d.reshape(bs, qps, kv_head_num, gqa, hdim)
    absmax = np.abs(q_r).max(axis=(1, 3, 4), keepdims=True).clip(min=1e-8)
    scale = absmax / FP8_MAX
    q_deq = (
        to_fp8(
            torch.from_numpy((q_r / scale).clip(-FP8_MAX, FP8_MAX).astype(np.float32))
        )
        .float()
        .numpy()
        * scale
    )
    return q_deq.reshape(bs, qps, hnum, hdim), scale.reshape(bs, kv_head_num)


def naive_attn(q, k, v, head_num, kv_head_num, head_dim=HEAD_DIM):
    """Numpy reference for single-sequence FP8 decode attention (supports GQA).

    K/V are first re-quantized per 128-token block to mirror the block-wise FP8
    cache path.

    Args:
        q: [head_num, head_dim] float32 query for a single decode token.
        k: [kv_len, kv_head_num, head_dim] float32 keys.
        v: [kv_len, kv_head_num, head_dim] float32 values.
        head_num: number of query heads.
        kv_head_num: number of KV heads.
        head_dim: per-head dimension.

    Returns:
        [head_num, head_dim] float32 attention output.
    """
    k, _ = quant_fp8_per128_tokens(k)
    v, _ = quant_fp8_per128_tokens(v)
    attn_scale = 1.0 / math.sqrt(head_dim)
    gqa = head_num // kv_head_num
    out = np.zeros([head_num, head_dim], np.float32)
    for h in range(head_num):
        kv_h = h // gqa
        scores = k[:, kv_h, :] @ q[h] * attn_scale
        scores -= scores.max()
        w = np.exp(scores)
        w /= w.sum()
        out[h] = w @ v[:, kv_h, :]
    return out


def reference_attention(q, k, v, head_num, kv_head_num, head_dim=HEAD_DIM):
    """Numpy reference for BF16 decode attention with implicit causal alignment.

    For query position ``qi`` the visible KV length is ``k.shape[0] - num_q + qi + 1``,
    matching bottom-right causal alignment used by the decode kernel.

    Args:
        q: [num_q, head_num, head_dim] array.
        k: [kv_len, kv_head_num, head_dim] array.
        v: [kv_len, kv_head_num, head_dim] array.
        head_num: number of query heads.
        kv_head_num: number of KV heads.
        head_dim: per-head dimension.

    Returns:
        [num_q, head_num, head_dim] array with the same dtype as ``q``.
    """
    gqa_group_size = head_num // kv_head_num
    scale = 1.0 / np.sqrt(head_dim)
    num_q = q.shape[0]
    out = np.zeros_like(q)
    for h in range(head_num):
        kv_h = h // gqa_group_size
        for qi in range(num_q):
            q_vec = q[qi, h, :]
            kv_len = k.shape[0] - num_q + qi + 1
            k_slice = k[:kv_len, kv_h, :]
            v_slice = v[:kv_len, kv_h, :]
            scores = q_vec @ k_slice.T * scale
            scores_max = np.max(scores)
            exp_scores = np.exp(scores - scores_max)
            softmax = exp_scores / np.sum(exp_scores)
            out[qi, h, :] = softmax @ v_slice
    return out


def build_kv_cache(
    k_list, v_list, kv_head_num, num_blocks, max_blk_per_seq, head_dim=HEAD_DIM
):
    """Scatter per-sequence K/V into a paged cache and build the block table.

    Args:
        k_list: list of [kv_len, kv_head_num, head_dim] arrays, one per sequence.
        v_list: list of [kv_len, kv_head_num, head_dim] arrays, one per sequence.
        kv_head_num: number of KV heads.
        num_blocks: total number of physical cache blocks.
        max_blk_per_seq: maximum blocks reserved per sequence.
        head_dim: per-head dimension.

    Returns:
        Tuple ``(cache_k, cache_v, block_tables)`` numpy arrays where the caches
        have shape [num_blocks, kv_head_num, BLOCK_SIZE, head_dim] and
        ``block_tables`` has shape [batch, max_blk_per_seq].
    """
    ck = np.zeros([num_blocks, kv_head_num, BLOCK_SIZE, head_dim], np.float32)
    cv = np.zeros([num_blocks, kv_head_num, BLOCK_SIZE, head_dim], np.float32)
    batch = len(k_list)
    bt = np.zeros([batch, max_blk_per_seq], np.int32)
    for b, (k_np, v_np) in enumerate(zip(k_list, v_list)):
        base = b * max_blk_per_seq
        for t in range(len(k_np)):
            blk = base + t // BLOCK_SIZE
            off = t % BLOCK_SIZE
            ck[blk, :, off, :] = k_np[t]
            cv[blk, :, off, :] = v_np[t]
        bt[b] = np.arange(base, base + max_blk_per_seq)
    return ck, cv, bt
