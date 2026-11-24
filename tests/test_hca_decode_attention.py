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

import pytest
import torch

from halops import hca_decode_attention

HEADS, DIM, NOPE, ROPE = 64, 512, 448, 64
ROW = NOPE + 2 * ROPE  # data bytes of a cache row: fp8 nope, bf16 rope
SF = 8  # scale bytes of a cache row: 7 UE8M0 scales of 64 dims, 1 pad
WINDOW = 128
SCALE = DIM**-0.5
# Relative to the largest |output| of a token.  P is rounded to bf16 for the
# PV MMA, as in FlashMLA: an error of 2^-9 |v| per key, about 2^-7 of the
# output's scale here (FlashMLA's own error against this reference is the
# same, 7.8e-3), plus the bf16 output rounding.
RTOL = 2e-2


# =====================================================================
# Inputs and reference
# =====================================================================


def make_cache(pages, page_size, gen):
    """uint8 [pages, page_size, 1, 584] view of FlashMLA's fp8 cache: a page
    holds page_size data rows, then page_size scale rows; its stride is a
    multiple of 576 bytes."""
    stride = (page_size * (ROW + SF) + ROW - 1) // ROW * ROW
    buf = torch.zeros(pages, stride, dtype=torch.uint8)
    data = buf[:, : page_size * ROW].view(pages, page_size, ROW)
    nope = (torch.randn(pages, page_size, NOPE, generator=gen) * 64).clamp(-448, 448)
    data[..., :NOPE] = nope.to(torch.float8_e4m3fn).view(torch.uint8)
    rope = torch.randn(pages, page_size, ROPE, generator=gen).bfloat16()
    data[..., NOPE:] = rope.view(torch.uint8).view(pages, page_size, 2 * ROPE)
    sf = buf[:, page_size * ROW : page_size * (ROW + SF)].view(pages, page_size, SF)
    sf[..., : NOPE // 64] = torch.randint(
        118, 124, (pages, page_size, NOPE // 64), dtype=torch.uint8, generator=gen
    )
    return buf[:, : page_size * (ROW + SF)].view(pages, page_size, 1, ROW + SF)


def dequant(cache, idx):
    """bf16 [n, 512] keys of the cache's token indices idx (all valid)."""
    pages, page_size = cache.shape[:2]
    flat = torch.as_strided(cache, (pages, cache.stride(0)), (cache.stride(0), 1))
    page, row = (idx // page_size)[:, None], (idx % page_size)[:, None]
    data = flat[page, row * ROW + torch.arange(ROW)]
    sf = flat[page, page_size * ROW + row * SF + torch.arange(NOPE // 64)]
    scale = torch.exp2(sf.float() - 127).repeat_interleave(64, -1)
    nope = (data[:, :NOPE].view(torch.float8_e4m3fn).float() * scale).bfloat16()
    return torch.cat([nope, data[:, NOPE:].contiguous().view(torch.bfloat16)], -1)


def valid_keys(cache, idx, length):
    n = cache.shape[0] * cache.shape[1]
    keep = idx[: max(min(int(length), idx.numel()), 0)]
    return keep[(keep >= 0) & (keep < n)].long()


def reference(q, swa, swa_idx, swa_len, c128, c_idx, c_len, sink):
    """fp32 attention over the bf16-dequantized keys, sink in the denominator."""
    out = torch.zeros_like(q)
    for t in range(q.shape[0]):
        k = torch.cat(
            [
                dequant(swa, valid_keys(swa, swa_idx[t], swa_len[t])),
                dequant(c128, valid_keys(c128, c_idx[t], c_len[t])),
            ]
        ).float()
        if k.shape[0] == 0:
            continue
        s = q[t].float() @ k.T * SCALE
        m = torch.maximum(s.amax(-1), sink)
        p = torch.exp(s - m[:, None])
        o = p @ k / (p.sum(-1) + torch.exp(sink - m))[:, None]
        out[t] = o.bfloat16()
    return out


def make_batch(tokens, tpr, n_c128, seed):
    """Verify-step inputs: request r is tokens [r * tpr, (r + 1) * tpr); token
    j's SWA window is the 128 newest rows up to its own position (newest
    first, shifted by one per token); the request's c128 list is shared, its
    last token sees one more row (the pair's lengths differ by 1)."""
    gen = torch.Generator(device=torch.get_default_device()).manual_seed(seed)
    reqs = tokens // tpr
    r, j = torch.arange(tokens) // tpr, torch.arange(tokens) % tpr
    swa_pages = (WINDOW + tpr) // 128 + 2
    swa = make_cache(reqs * swa_pages, 128, gen)
    newest = r * swa_pages * 128 + WINDOW + tpr + j
    swa_idx = (newest[:, None] - torch.arange(WINDOW)).int()
    swa_len = torch.full((tokens,), WINDOW, dtype=torch.int32)
    c_rows = n_c128 + 1
    c_pages = c_rows // 2 + 1
    c128 = make_cache(reqs * c_pages, 2, gen)
    lists = torch.stack(
        [torch.randperm(c_pages * 2, generator=gen)[:c_rows] + q * c_pages * 2 for q in range(reqs)]
    )
    c_idx = torch.full((tokens, (c_rows + 63) // 64 * 64), -1, dtype=torch.int32)
    c_idx[:, :c_rows] = lists[r].int()
    c_len = torch.where(j == tpr - 1, c_rows, n_c128).int()
    q = torch.randn(tokens, HEADS, DIM, generator=gen).bfloat16()
    sink = torch.randn(HEADS, generator=gen)
    return q, swa, swa_idx, swa_len, c128, c_idx, c_len, sink


def run(args, tpr, **kw):
    q, swa, swa_idx, swa_len, c128, c_idx, c_len, sink = args
    return hca_decode_attention(
        q, swa, swa_idx, swa_len, c128, c_idx, c_len, sink, SCALE, tpr, **kw
    )


def assert_close(out, ref, rows=None):
    rows = torch.ones(out.shape[0], dtype=torch.bool) if rows is None else rows
    d = (out.float() - ref.float())[rows].abs().flatten(1).amax(-1)
    bound = RTOL * ref.float()[rows].abs().flatten(1).amax(-1).clamp_min(1e-3)
    assert torch.isfinite(out[rows]).all()
    assert (d <= bound).all(), f"max rel err {(d / bound).max().item() * RTOL:.3g}"


# =====================================================================
# Correctness tests
# =====================================================================


@pytest.mark.parametrize(
    "tokens, tpr",
    [(8, 2), (8, 4), (8, 8), (64, 4), (64, 8), (128, 4), (128, 8), (256, 8), (768, 4), (1536, 8),
     (6, 3), (24, 3), (30, 5), (768, 3), (765, 5)],
)
def test_hca_decode_attention_matches_reference(tokens, tpr):
    """Small batches give a pair one or two clusters, larger ones split the
    tile stream, so pairs are cut and merged at every size.  Odd tokens_per_req
    leaves each request's last token unpaired (the lone CTA-1 path)."""
    args = make_batch(tokens, tpr, 860, seed=tokens + tpr)
    assert_close(run(args, tpr), reference(*args))


def test_hca_decode_attention_nan_query_isolated_lone():
    """A NaN in a lone (odd-request) token's query stays local, and a NaN in
    another token does not reach the lone one."""
    args = make_batch(24, 3, 860, seed=4)
    args[0][2, 1, 9] = float("nan")   # the first request's lone token
    args[0][7, 0, 0] = float("nan")   # a paired token of the third request
    rows = torch.ones(24, dtype=torch.bool)
    rows[2] = rows[7] = False
    assert_close(run(args, 3), reference(*args), rows)



def test_hca_decode_attention_long_and_short_lists():
    """c128 lists of 2000 and 7000 rows (more tiles than a cluster's share at
    T = 8) and a list shorter than a tile."""
    for n_c128, tpr in [(2000, 2), (7000, 8), (40, 4)]:
        args = make_batch(8, tpr, n_c128, seed=n_c128)
        assert_close(run(args, tpr), reference(*args))


def test_hca_decode_attention_irregular_rows():
    """Invalid and short rows are exact: a padding request (one key at index
    0, the rest -1), all-invalid rows, a pair with no keys, an earlier token
    with the longer c128 list, indices past the end of a cache and a
    shortened SWA window."""
    tokens, tpr = 32, 4
    args = make_batch(tokens, tpr, 300, seed=7)
    q, swa, swa_idx, swa_len, c128, c_idx, c_len, sink = args
    pad = slice(tokens - tpr, tokens)
    swa_idx[pad] = -1
    swa_idx[pad, 0] = 0
    swa_len[pad] = 1
    c_idx[pad] = -1
    c_len[pad] = torch.tensor([0, 1, 1, 0], dtype=torch.int32)
    c_len[4:6] = 0
    swa_len[4:6] = 0
    c_len[8] = 301
    c_len[9] = 300
    swa_idx[10, ::3] = swa.shape[0] * swa.shape[1] + 5
    c_idx[11, 5:40] = c128.shape[0] * c128.shape[1]
    swa_len[12] = 37
    out = run(args, tpr)
    assert_close(out, reference(*args))
    assert torch.all(out[4:6] == 0)


def test_hca_decode_attention_nan_query_isolated():
    """A NaN in one token's query reaches no other token, not even its pair
    partner, which shares the KV tiles."""
    args = make_batch(64, 4, 860, seed=3)
    args[0][5, 3, 7] = float("nan")
    rows = torch.ones(64, dtype=torch.bool)
    rows[5] = False
    assert_close(run(args, 4), reference(*args), rows)


def test_hca_decode_attention_deterministic_and_inplace():
    """Split pairs are merged in a fixed order: reruns are bit-identical; the
    4-D shapes and a given out buffer are accepted."""
    tokens, tpr = 256, 8
    args = make_batch(tokens, tpr, 860, seed=11)
    ref = run(args, tpr)
    q, swa, swa_idx, swa_len, c128, c_idx, c_len, sink = args
    out = torch.full((tokens, 1, HEADS, DIM), float("nan"), dtype=torch.bfloat16)
    got = hca_decode_attention(
        q[:, None], swa, swa_idx[:, None], swa_len, c128, c_idx[:, None], c_len,
        sink, SCALE, tpr, out=out, use_pdl=False,
    )
    assert got.data_ptr() == out.data_ptr()
    assert torch.equal(got.view_as(ref), ref)


def test_hca_decode_attention_matches_flash_mla():
    """Same inputs through FlashMLA's fp8 sparse decode, when available."""
    fm = pytest.importorskip("sgl_kernel.flash_mla")
    tokens, tpr = 128, 4
    args = make_batch(tokens, tpr, 860, seed=5)
    q, swa, swa_idx, swa_len, c128, c_idx, c_len, sink = args
    ref = fm.flash_mla_with_kvcache(
        q=q[:, None], k_cache=swa, head_dim_v=DIM, block_table=None,
        cache_seqlens=None, tile_scheduler_metadata=fm.get_mla_metadata()[0],
        softmax_scale=SCALE, is_fp8_kvcache=True, indices=swa_idx[:, None],
        topk_length=swa_len, attn_sink=sink, extra_k_cache=c128,
        extra_indices_in_kvcache=c_idx[:, None], extra_topk_length=c_len,
    )[0][:, 0]
    assert_close(run(args, tpr), ref)


# =====================================================================
# Error input tests
# =====================================================================


@pytest.mark.parametrize(
    "tpr, expected",
    [
        pytest.param(1, "tokens_per_req must be >= 2", id="tpr1"),
        pytest.param(16, "tokens_per_req must be >= 2 and divide T", id="tpr_not_dividing"),
        pytest.param(3, "tokens_per_req must be >= 2 and divide T", id="tpr_not_dividing_odd"),
    ],
)
def test_hca_decode_attention_errors(tpr, expected):
    args = make_batch(8, 4, 60, seed=1)
    with pytest.raises(RuntimeError, match=expected):
        run(args, tpr)
