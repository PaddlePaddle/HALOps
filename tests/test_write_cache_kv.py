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

import numpy as np
import pytest
import torch

from halops import write_cache_kv_encoder, write_cache_kv_decoder
from tests.utils.attn_utils import quant_fp8_per128_tokens
from tests.utils.device import skip_reason

if skip_reason():
    pytest.skip(skip_reason(), allow_module_level=True)

torch.manual_seed(0)

HEAD_DIM = 128
BLOCK_SIZE = 128
MAX_INPUT_LEN = 8192
DTYPE = torch.bfloat16

# Identity RoPE table (cos=1, sin=0) so that Q/K pass through unchanged.
rotary_embs = torch.zeros(
    2, MAX_INPUT_LEN, HEAD_DIM // 2, dtype=torch.float32, device="cuda"
)
rotary_embs[0] = 1


def _alloc_cache(num_blocks, kv_head_num):
    cache_k = torch.zeros(
        num_blocks, kv_head_num, BLOCK_SIZE, HEAD_DIM, dtype=DTYPE, device="cuda"
    )
    cache_v = torch.zeros_like(cache_k)
    return cache_k, cache_v


@pytest.mark.parametrize("head_num,kv_head_num", [(14, 1), (56, 4)])
@pytest.mark.parametrize("seq_len", [128, 256])
def test_write_cache_kv_encoder_correctness(head_num, kv_head_num, seq_len):
    batch = 1
    # The BF16 encoder kernel applies partial NeoX RoPE with rope_dim =
    # HEAD_DIM // 4, so its cos/sin table stride differs from the decoder's
    # HEAD_DIM // 2. Build an encoder-specific identity table (cos=1, sin=0).
    rotary_embs_enc = torch.zeros(
        2, MAX_INPUT_LEN, HEAD_DIM // 4, dtype=torch.float32, device="cuda"
    )
    rotary_embs_enc[0] = 1
    num_blocks = MAX_INPUT_LEN // BLOCK_SIZE
    qkv = torch.randn(
        seq_len, head_num + 2 * kv_head_num, HEAD_DIM, dtype=DTYPE, device="cuda"
    )
    q_ref = qkv[:, :head_num, :].contiguous()
    k_ref = qkv[:, head_num : head_num + kv_head_num, :].contiguous()
    v_ref = qkv[:, head_num + kv_head_num :, :].contiguous()

    cu_seq_q = torch.tensor([0, seq_len], dtype=torch.int32, device="cuda")
    cu_seq_k = cu_seq_q.clone()
    seq_len_encoder = torch.tensor([seq_len], dtype=torch.int32, device="cuda")
    seq_len_decoder = torch.zeros(batch, dtype=torch.int32, device="cuda")

    cache_k, cache_v = _alloc_cache(num_blocks, kv_head_num)
    block_tables = torch.arange(num_blocks, dtype=torch.int32, device="cuda").reshape(
        batch, num_blocks
    )

    q_out = torch.zeros(seq_len, head_num, HEAD_DIM, dtype=DTYPE, device="cuda")
    k_out = torch.zeros(seq_len, kv_head_num, HEAD_DIM, dtype=DTYPE, device="cuda")
    v_out = torch.zeros_like(k_out)

    write_cache_kv_encoder(
        qkv,
        cu_seq_q,
        cu_seq_k,
        rotary_embs_enc,
        seq_len_encoder,
        seq_len_decoder,
        cache_k,
        cache_v,
        block_tables,
        q_out,
        k_out,
        v_out,
        None,
        None,
        None,
        int(head_num),
        int(kv_head_num),
        int(HEAD_DIM),
        int(seq_len),
        int(seq_len),
        int(MAX_INPUT_LEN),
        "none",
    )

    # Identity RoPE + no RMSNorm => the split Q/K/V pass through unchanged.
    assert torch.equal(q_out, q_ref)
    assert torch.equal(k_out, k_ref)
    assert torch.equal(v_out, v_ref)

    # Every token must be scattered into the paged cache at (t // 128, t % 128).
    # K is stored at that offset directly. The V cache additionally applies a
    # within-block row reorder (swap bits 0/1 of the low 2 bits: 0 1 2 3 ->
    # 0 2 1 3) to match the decode-attention V read layout, so v_ref[t] lands
    # at the reordered offset.
    for t in range(seq_len):
        blk, off = t // BLOCK_SIZE, t % BLOCK_SIZE
        v_off = (off & ~3) | ((off & 1) << 1) | ((off & 2) >> 1)
        assert torch.equal(cache_k[blk, :, off, :], k_ref[t])
        assert torch.equal(cache_v[blk, :, v_off, :], v_ref[t])


@pytest.mark.parametrize("head_num,kv_head_num", [(14, 1), (56, 4)])
@pytest.mark.parametrize("prefix_len", [1, 130])
def test_write_cache_kv_decoder_correctness(head_num, kv_head_num, prefix_len):
    batch = 1
    num_blocks = MAX_INPUT_LEN // BLOCK_SIZE
    qkv = torch.randn(
        batch, head_num + 2 * kv_head_num, HEAD_DIM, dtype=DTYPE, device="cuda"
    )
    v_ref = qkv[:, head_num + kv_head_num :, :].contiguous()

    cu_seq_q = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    seq_len_encoder = torch.zeros(batch, dtype=torch.int32, device="cuda")
    seq_len_decoder = torch.tensor([prefix_len], dtype=torch.int32, device="cuda")

    cache_k, cache_v = _alloc_cache(num_blocks, kv_head_num)
    block_tables = torch.arange(num_blocks, dtype=torch.int32, device="cuda").reshape(
        batch, num_blocks
    )

    q_output, q_dequant_scale = write_cache_kv_decoder(
        qkv,
        cu_seq_q,
        seq_len_encoder,
        seq_len_decoder,
        rotary_embs,
        cache_k,
        cache_v,
        block_tables,
        None,
        None,
        None,
        int(head_num),
        int(kv_head_num),
        int(HEAD_DIM),
        int(MAX_INPUT_LEN),
        "none",
    )

    assert q_output.shape == (batch, kv_head_num, 64, HEAD_DIM)
    # BF16 path still returns an allocated-but-unused scale placeholder (not None).
    assert q_dequant_scale.shape == (batch, kv_head_num)

    # The new token's V (not affected by RoPE) is appended at offset ==
    # prefix_len, then goes through the same within-block row reorder (swap
    # bits 0/1 of the low 2 bits) that the encoder V cache uses.
    blk, off = prefix_len // BLOCK_SIZE, prefix_len % BLOCK_SIZE
    v_off = (off & ~3) | ((off & 1) << 1) | ((off & 2) >> 1)
    assert torch.equal(cache_v[blk, :, v_off, :], v_ref[0])


@pytest.mark.parametrize("head_num,kv_head_num", [(14, 1), (56, 4)])
def test_write_cache_kv_decoder_determinism(head_num, kv_head_num):
    batch, prefix_len = 1, 130
    num_blocks = MAX_INPUT_LEN // BLOCK_SIZE
    qkv = torch.randn(
        batch, head_num + 2 * kv_head_num, HEAD_DIM, dtype=DTYPE, device="cuda"
    )
    cu_seq_q = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    seq_len_encoder = torch.zeros(batch, dtype=torch.int32, device="cuda")
    seq_len_decoder = torch.tensor([prefix_len], dtype=torch.int32, device="cuda")
    block_tables = torch.arange(num_blocks, dtype=torch.int32, device="cuda").reshape(
        batch, num_blocks
    )

    def _run():
        cache_k, cache_v = _alloc_cache(num_blocks, kv_head_num)
        q_output, _ = write_cache_kv_decoder(
            qkv,
            cu_seq_q,
            seq_len_encoder,
            seq_len_decoder,
            rotary_embs,
            cache_k,
            cache_v,
            block_tables,
            None,
            None,
            None,
            int(head_num),
            int(kv_head_num),
            int(HEAD_DIM),
            int(MAX_INPUT_LEN),
            "none",
        )
        return q_output, cache_k, cache_v

    q1, ck1, cv1 = _run()
    q2, ck2, cv2 = _run()
    assert torch.equal(q1, q2)
    assert torch.equal(ck1, ck2)
    assert torch.equal(cv1, cv2)


@pytest.mark.parametrize("head_num,kv_head_num", [(14, 1), (56, 4)])
@pytest.mark.parametrize("seq_len", [128, 256])
def test_write_cache_kv_encoder_determinism(head_num, kv_head_num, seq_len):
    # Same inputs must yield bit-identical Q/K/V outputs and paged cache
    # contents across runs. Build fixed inputs once, then invoke the kernel
    # twice into freshly-zeroed output and cache buffers.
    batch = 1
    rotary_embs_enc = torch.zeros(
        2, MAX_INPUT_LEN, HEAD_DIM // 4, dtype=torch.float32, device="cuda"
    )
    rotary_embs_enc[0] = 1
    num_blocks = MAX_INPUT_LEN // BLOCK_SIZE
    qkv = torch.randn(
        seq_len, head_num + 2 * kv_head_num, HEAD_DIM, dtype=DTYPE, device="cuda"
    )
    cu_seq_q = torch.tensor([0, seq_len], dtype=torch.int32, device="cuda")
    cu_seq_k = cu_seq_q.clone()
    seq_len_encoder = torch.tensor([seq_len], dtype=torch.int32, device="cuda")
    seq_len_decoder = torch.zeros(batch, dtype=torch.int32, device="cuda")
    block_tables = torch.arange(num_blocks, dtype=torch.int32, device="cuda").reshape(
        batch, num_blocks
    )

    def _run():
        cache_k, cache_v = _alloc_cache(num_blocks, kv_head_num)
        q_out = torch.zeros(seq_len, head_num, HEAD_DIM, dtype=DTYPE, device="cuda")
        k_out = torch.zeros(seq_len, kv_head_num, HEAD_DIM, dtype=DTYPE, device="cuda")
        v_out = torch.zeros_like(k_out)
        write_cache_kv_encoder(
            qkv,
            cu_seq_q,
            cu_seq_k,
            rotary_embs_enc,
            seq_len_encoder,
            seq_len_decoder,
            cache_k,
            cache_v,
            block_tables,
            q_out,
            k_out,
            v_out,
            None,
            None,
            None,
            int(head_num),
            int(kv_head_num),
            int(HEAD_DIM),
            int(seq_len),
            int(seq_len),
            int(MAX_INPUT_LEN),
            "none",
        )
        return q_out, k_out, v_out, cache_k, cache_v

    q1, k1, v1, ck1, cv1 = _run()
    q2, k2, v2, ck2, cv2 = _run()
    assert torch.equal(q1, q2)
    assert torch.equal(k1, k2)
    assert torch.equal(v1, v2)
    assert torch.equal(ck1, ck2)
    assert torch.equal(cv1, cv2)


def _equal_bits(a, b):
    """Bit-exact comparison that also works for float8_e4m3fn tensors."""
    if a.dtype == torch.float8_e4m3fn:
        return torch.equal(a.view(torch.uint8), b.view(torch.uint8))
    return torch.equal(a, b)


def _fp8_v_reorder(off):
    """Within-block V row reorder used by the FP8 (block_wise_fp8) cache.

    This is the 8-row-granularity permutation the FP8 write kernels apply so the
    stored V layout matches the FP8 decode-attention read order. It differs from
    the BF16 cache's low-2-bit swap.
    """
    return (off & ~7) | ((off & 1) << 2) | ((off & 6) >> 1)


def _dequant_fp8_cache(cache_uint8, kv_dequant_scale, slot):
    """Dequantize a packed-FP8 (uint8) paged cache back to float32 numpy.

    Args:
        cache_uint8: [num_blocks, kv_head_num, BLOCK_SIZE, HEAD_DIM] uint8 tensor
            holding packed float8_e4m3fn values.
        kv_dequant_scale: [num_blocks, kv_head_num, 4] float32 per-block scale.
        slot: 0 reads the K scale, 1 reads the V scale.

    Returns:
        numpy float32 array with the same shape as ``cache_uint8``.
    """
    fp8 = cache_uint8.view(torch.float8_e4m3fn).float()
    scale = kv_dequant_scale[:, :, slot][:, :, None, None]
    return (fp8 * scale).cpu().numpy()


@pytest.mark.parametrize("head_num,kv_head_num", [(14, 1), (56, 4)])
@pytest.mark.parametrize("seq_len", [128, 256])
def test_write_cache_kv_encoder_fp8_determinism(head_num, kv_head_num, seq_len):
    # FP8 (block_wise_fp8) cache write: the kernel block-wise quantizes K/V into
    # a uint8 (packed FP8) cache and writes the per-block dequant scale. Identical
    # inputs must yield bit-identical Q/K/V outputs, packed FP8 cache, and scales.
    batch = 1
    num_blocks = MAX_INPUT_LEN // BLOCK_SIZE
    qkv = torch.randn(
        seq_len, head_num + 2 * kv_head_num, HEAD_DIM, dtype=DTYPE, device="cuda"
    )
    cu_seq_q = torch.tensor([0, seq_len], dtype=torch.int32, device="cuda")
    cu_seq_k = cu_seq_q.clone()
    seq_len_encoder = torch.tensor([seq_len], dtype=torch.int32, device="cuda")
    seq_len_decoder = torch.zeros(batch, dtype=torch.int32, device="cuda")
    block_tables = torch.arange(num_blocks, dtype=torch.int32, device="cuda").reshape(
        batch, num_blocks
    )
    q_norm_gamma = torch.ones(HEAD_DIM, dtype=torch.float32, device="cuda")
    k_norm_gamma = torch.ones(HEAD_DIM, dtype=torch.float32, device="cuda")

    def _run():
        cache_k = torch.zeros(
            num_blocks,
            kv_head_num,
            BLOCK_SIZE,
            HEAD_DIM,
            dtype=torch.uint8,
            device="cuda",
        )
        cache_v = torch.zeros_like(cache_k)
        kv_dequant_scale = torch.zeros(
            num_blocks, kv_head_num, 4, dtype=torch.float32, device="cuda"
        )
        q_out = torch.zeros(seq_len, head_num, HEAD_DIM, dtype=DTYPE, device="cuda")
        k_out = torch.zeros(seq_len, kv_head_num, HEAD_DIM, dtype=DTYPE, device="cuda")
        v_out = torch.zeros_like(k_out)
        write_cache_kv_encoder(
            qkv,
            cu_seq_q,
            cu_seq_k,
            rotary_embs,
            seq_len_encoder,
            seq_len_decoder,
            cache_k,
            cache_v,
            block_tables,
            q_out,
            k_out,
            v_out,
            kv_dequant_scale,
            q_norm_gamma,
            k_norm_gamma,
            int(head_num),
            int(kv_head_num),
            int(HEAD_DIM),
            int(seq_len),
            int(seq_len),
            int(MAX_INPUT_LEN),
            "block_wise_fp8",
        )
        return q_out, k_out, v_out, cache_k, cache_v, kv_dequant_scale

    first = _run()
    second = _run()
    for a, b in zip(first, second):
        assert _equal_bits(a, b)


@pytest.mark.parametrize("head_num,kv_head_num", [(14, 1), (56, 4)])
def test_write_cache_kv_decoder_fp8_determinism(head_num, kv_head_num):
    # FP8 decode cache write: the returned q_output is packed FP8 and the current
    # token's K/V is block-wise quantized into the uint8 cache. Identical inputs
    # must produce bit-identical q_output, q/kv dequant scales, and packed cache.
    batch, prefix_len = 1, 130
    num_blocks = MAX_INPUT_LEN // BLOCK_SIZE
    # q_output is [batch, kv_head_num, 64, head_dim] but the kernel only writes
    # the GQA-folded valid rows; rows past this are uninitialized padding (the
    # buffer is torch::empty), so only compare the valid region.
    valid_q_rows = head_num // kv_head_num
    qkv = torch.randn(
        batch, head_num + 2 * kv_head_num, HEAD_DIM, dtype=DTYPE, device="cuda"
    )
    cu_seq_q = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    seq_len_encoder = torch.zeros(batch, dtype=torch.int32, device="cuda")
    seq_len_decoder = torch.tensor([prefix_len], dtype=torch.int32, device="cuda")
    block_tables = torch.arange(num_blocks, dtype=torch.int32, device="cuda").reshape(
        batch, num_blocks
    )
    q_norm_gamma = torch.ones(HEAD_DIM, dtype=torch.float32, device="cuda")
    k_norm_gamma = torch.ones(HEAD_DIM, dtype=torch.float32, device="cuda")

    def _run():
        cache_k = torch.zeros(
            num_blocks,
            kv_head_num,
            BLOCK_SIZE,
            HEAD_DIM,
            dtype=torch.uint8,
            device="cuda",
        )
        cache_v = torch.zeros_like(cache_k)
        kv_dequant_scale = torch.zeros(
            num_blocks, kv_head_num, 4, dtype=torch.float32, device="cuda"
        )
        q_output, q_dequant_scale = write_cache_kv_decoder(
            qkv,
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
            int(head_num),
            int(kv_head_num),
            int(HEAD_DIM),
            int(MAX_INPUT_LEN),
            "block_wise_fp8",
        )
        return (
            q_output[:, :, :valid_q_rows, :],
            q_dequant_scale,
            cache_k,
            cache_v,
            kv_dequant_scale,
        )

    first = _run()
    second = _run()
    for a, b in zip(first, second):
        assert _equal_bits(a, b)


@pytest.mark.parametrize("head_num,kv_head_num", [(14, 1), (56, 4)])
@pytest.mark.parametrize("seq_len", [128, 256])
def test_write_cache_kv_encoder_fp8_accuracy(head_num, kv_head_num, seq_len):
    # FP8 (block_wise_fp8) encoder cache write accuracy. With gamma=None the
    # kernel applies identity RoPE and no RMSNorm, so K/V pass through unchanged
    # and are only block-wise (per-128-token) abs-max quantized into the packed
    # uint8 cache. We compare the dequantized cache and the per-block scales
    # against tests.attn_utils.quant_fp8_per128_tokens, the same block-wise FP8
    # reference used by the decode-attention accuracy test.
    batch = 1
    num_blocks = MAX_INPUT_LEN // BLOCK_SIZE
    qkv = torch.randn(
        seq_len, head_num + 2 * kv_head_num, HEAD_DIM, dtype=DTYPE, device="cuda"
    )
    k_ref = qkv[:, head_num : head_num + kv_head_num, :].contiguous()
    v_ref = qkv[:, head_num + kv_head_num :, :].contiguous()

    cu_seq_q = torch.tensor([0, seq_len], dtype=torch.int32, device="cuda")
    cu_seq_k = cu_seq_q.clone()
    seq_len_encoder = torch.tensor([seq_len], dtype=torch.int32, device="cuda")
    seq_len_decoder = torch.zeros(batch, dtype=torch.int32, device="cuda")
    block_tables = torch.arange(num_blocks, dtype=torch.int32, device="cuda").reshape(
        batch, num_blocks
    )

    cache_k = torch.zeros(
        num_blocks, kv_head_num, BLOCK_SIZE, HEAD_DIM, dtype=torch.uint8, device="cuda"
    )
    cache_v = torch.zeros_like(cache_k)
    kv_dequant_scale = torch.zeros(
        num_blocks, kv_head_num, 4, dtype=torch.float32, device="cuda"
    )
    q_out = torch.zeros(seq_len, head_num, HEAD_DIM, dtype=DTYPE, device="cuda")
    k_out = torch.zeros(seq_len, kv_head_num, HEAD_DIM, dtype=DTYPE, device="cuda")
    v_out = torch.zeros_like(k_out)

    write_cache_kv_encoder(
        qkv,
        cu_seq_q,
        cu_seq_k,
        rotary_embs,
        seq_len_encoder,
        seq_len_decoder,
        cache_k,
        cache_v,
        block_tables,
        q_out,
        k_out,
        v_out,
        kv_dequant_scale,
        None,
        None,
        int(head_num),
        int(kv_head_num),
        int(HEAD_DIM),
        int(seq_len),
        int(seq_len),
        int(MAX_INPUT_LEN),
        "block_wise_fp8",
    )

    # Block-wise FP8 reference (per-128-token abs-max) for K and V.
    k_np = k_ref.float().cpu().numpy()
    v_np = v_ref.float().cpu().numpy()
    k_deq, k_scale = quant_fp8_per128_tokens(k_np)  # scale: [groups, kv_head_num]
    v_deq, v_scale = quant_fp8_per128_tokens(v_np)
    groups = seq_len // BLOCK_SIZE

    deq_k = _dequant_fp8_cache(cache_k, kv_dequant_scale, 0)
    deq_v = _dequant_fp8_cache(cache_v, kv_dequant_scale, 1)

    # Per-block K/V scales must match the reference abs-max / FP8_MAX.
    kv_scale_np = kv_dequant_scale.cpu().numpy()
    for g in range(groups):
        np.testing.assert_allclose(
            kv_scale_np[g, :, 0], k_scale[g], rtol=1e-3, atol=1e-6
        )
        np.testing.assert_allclose(
            kv_scale_np[g, :, 1], v_scale[g], rtol=1e-3, atol=1e-6
        )

    # Gather the dequantized cache back into logical [token, head, dim] order:
    # K sits at its natural within-block offset, V at the 8-row-reordered offset.
    kernel_k = np.empty_like(k_np)
    kernel_v = np.empty_like(v_np)
    for t in range(seq_len):
        blk, off = t // BLOCK_SIZE, t % BLOCK_SIZE
        kernel_k[t] = deq_k[blk, :, off, :]
        kernel_v[t] = deq_v[blk, :, _fp8_v_reorder(off), :]

    # The kernel quantizes with a BF16-rounded scale factor, so allow a rare
    # single-FP8-step boundary flip in the worst case (atol ~ absmax/8), while a
    # tight mean bound guards against a wrong layout/reorder/scale index (which
    # would shift essentially every element by the full signal magnitude).
    absmax = max(float(np.abs(k_np).max()), float(np.abs(v_np).max()))
    np.testing.assert_allclose(kernel_k, k_deq, rtol=0.0, atol=absmax / 8 + 1e-6)
    np.testing.assert_allclose(kernel_v, v_deq, rtol=0.0, atol=absmax / 8 + 1e-6)
    assert np.abs(kernel_k - k_deq).mean() < absmax * 1e-2
    assert np.abs(kernel_v - v_deq).mean() < absmax * 1e-2


@pytest.mark.parametrize("head_num,kv_head_num", [(14, 1), (56, 4)])
@pytest.mark.parametrize("prefix_len", [1, 130])
def test_write_cache_kv_decoder_fp8_accuracy(head_num, kv_head_num, prefix_len):
    # FP8 (block_wise_fp8) decode cache write accuracy. The single new token's
    # K/V is block-wise abs-max quantized into the packed uint8 cache at offset
    # == prefix_len (K at the natural offset, V at the 8-row-reordered offset),
    # and that block's K/V scale is written. With gamma=None the new token's K/V
    # pass through unchanged (identity RoPE, no RMSNorm), so a single-token
    # quant_fp8_per128_tokens is an exact block-wise reference.
    batch = 1
    num_blocks = MAX_INPUT_LEN // BLOCK_SIZE
    qkv = torch.randn(
        batch, head_num + 2 * kv_head_num, HEAD_DIM, dtype=DTYPE, device="cuda"
    )
    k_ref = qkv[:, head_num : head_num + kv_head_num, :].contiguous()
    v_ref = qkv[:, head_num + kv_head_num :, :].contiguous()

    cu_seq_q = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    seq_len_encoder = torch.zeros(batch, dtype=torch.int32, device="cuda")
    seq_len_decoder = torch.tensor([prefix_len], dtype=torch.int32, device="cuda")
    block_tables = torch.arange(num_blocks, dtype=torch.int32, device="cuda").reshape(
        batch, num_blocks
    )

    cache_k = torch.zeros(
        num_blocks, kv_head_num, BLOCK_SIZE, HEAD_DIM, dtype=torch.uint8, device="cuda"
    )
    cache_v = torch.zeros_like(cache_k)
    kv_dequant_scale = torch.zeros(
        num_blocks, kv_head_num, 4, dtype=torch.float32, device="cuda"
    )

    write_cache_kv_decoder(
        qkv,
        cu_seq_q,
        seq_len_encoder,
        seq_len_decoder,
        rotary_embs,
        cache_k,
        cache_v,
        block_tables,
        kv_dequant_scale,
        None,
        None,
        int(head_num),
        int(kv_head_num),
        int(HEAD_DIM),
        int(MAX_INPUT_LEN),
        "block_wise_fp8",
    )

    # Single-token block-wise FP8 reference (abs-max over head_dim per kv head).
    k_np = k_ref.float().cpu().numpy()
    v_np = v_ref.float().cpu().numpy()
    k_deq, k_scale = quant_fp8_per128_tokens(k_np)  # scale: [1, kv_head_num]
    v_deq, v_scale = quant_fp8_per128_tokens(v_np)

    blk, off = prefix_len // BLOCK_SIZE, prefix_len % BLOCK_SIZE
    deq_k = _dequant_fp8_cache(cache_k, kv_dequant_scale, 0)
    deq_v = _dequant_fp8_cache(cache_v, kv_dequant_scale, 1)

    kv_scale_np = kv_dequant_scale.cpu().numpy()
    np.testing.assert_allclose(kv_scale_np[blk, :, 0], k_scale[0], rtol=1e-3, atol=1e-6)
    np.testing.assert_allclose(kv_scale_np[blk, :, 1], v_scale[0], rtol=1e-3, atol=1e-6)

    absmax = max(float(np.abs(k_np).max()), float(np.abs(v_np).max()))
    np.testing.assert_allclose(
        deq_k[blk, :, off, :], k_deq[0], rtol=0.0, atol=absmax / 8 + 1e-6
    )
    np.testing.assert_allclose(
        deq_v[blk, :, _fp8_v_reorder(off), :],
        v_deq[0],
        rtol=0.0,
        atol=absmax / 8 + 1e-6,
    )


if __name__ == "__main__":
    pytest.main([__file__])
