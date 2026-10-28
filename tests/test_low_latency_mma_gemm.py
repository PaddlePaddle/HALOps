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

import random

import pytest
import torch

from halops import group_wise_quant, low_latency_mma_gemm, prefer_low_latency_mma
from tests.utils.gemm_utils import W8A8_MXFP8_TOL, ue8m0_to_float32

SCALE_GROUP = 32
# Relative Frobenius error vs the unquantized bf16 matmul; catches a reference
# that misreads the scale layout the same way the kernel does.
BF16_REL_TOL = 0.08

# Shapes are drawn from a fixed-seed RNG so the parametrization is stable
# across runs while not pinning the tests to a handful of hand-picked sizes.
SHAPE_SEED = 2026
_rng = random.Random(SHAPE_SEED)

# Supported (block_n, warps_k) pairs: block_n * warps_k <= 128, warps_k <= 16.
TILE_CONFIGS = [
    (bn, wk) for bn in (8, 16, 32, 64) for wk in (1, 2, 4, 8, 16) if bn * wk <= 128
]


def rand_n(step, lo=512, hi=16384):
    return _rng.randrange(-(-lo // step) * step, hi + 1, step)


def rand_k(warps_k, lo=1024, hi=8192):
    """K a multiple of 128 with K / 128 divisible by 2 * warps_k; hi bounds
    the 16-row resident A in shared memory."""
    step = 256 * max(warps_k, 1)
    return _rng.randrange(-(-lo // step) * step, hi + 1, step)


def rand_tiled_shape(lo_k=1024, hi_k=8192):
    block_n, warps_k = _rng.choice(TILE_CONFIGS)
    return rand_n(block_n), rand_k(warps_k, lo_k, hi_k), block_n, warps_k


def rand_odd_group_k():
    """K / 128 even but not a power of two, so the auto chooser has to halve
    warps_k; K / 32 not a multiple of 16 also forces the scale repack."""
    while True:
        j = _rng.randrange(3, 33)
        if j & (j - 1):
            return 256 * j


# Auto-configured decode projections, M = 8.
DECODE_SHAPES = [(rand_n(64), rand_k(1)) for _ in range(8)]

# (n, k, block_n, warps_k): explicit tile configs.
GATE_SHAPES = [rand_tiled_shape() for _ in range(9)]

ODD_GROUP_SHAPES = [(rand_n(64), rand_odd_group_k()) for _ in range(4)]


def rand_dual_output_shape():
    n, k, block_n, warps_k = rand_tiled_shape(hi_k=4096)
    n_split = _rng.randrange(block_n, n, block_n)
    return n, k, n_split, block_n, warps_k


# (n, k, n_split, block_n, warps_k) for the segmented dual output.
DUAL_OUTPUT_SHAPES = [rand_dual_output_shape() for _ in range(5)]


def rand_dual_a_shape(auto):
    block_n, warps_k = (-1, -1) if auto else _rng.choice(TILE_CONFIGS)
    n = rand_n(64 if auto else block_n)
    k = rand_k(warps_k, lo=2048, hi=6144)
    k1 = _rng.randrange(128, k, 128)
    return n, k1, k - k1, block_n, warps_k


# (n, k1, k2, block_n, warps_k) for the dual-A concat-K path; random K1 often
# has K1 % 512 != 0 (host merge of sfa and sfa2) and non-16B scale rows.
DUAL_A_SHAPES = [rand_dual_a_shape(auto=i == 0) for i in range(5)]

# A CTA that walks several tiles, each many B phases deep: 4096 / 8 = 512
# tiles over the SMs, and warps_k = 1 gives 16 phases per tile, so ring wraps
# and tile order both matter.
WALK_N, WALK_K, WALK_BLOCK_N, WALK_WARPS_K = 4096, 4096, 8, 1


def spread_groups(t):
    """Scale every (row, 32-wide K group) by its own power of two in
    2^-2..2^2, so a scale read from the wrong row or k-group shows up as a
    factor-of-two error."""
    rows, k = t.shape
    e = torch.randint(-2, 3, (rows, k // SCALE_GROUP, 1)).float()
    t = t.float().view(rows, k // SCALE_GROUP, SCALE_GROUP) * torch.exp2(e)
    return t.view(rows, k).to(torch.bfloat16)


def make_bf16(rows, k, scale):
    return spread_groups(torch.randn(rows, k, dtype=torch.bfloat16) * scale)


def quant(x):
    return group_wise_quant(x, mode="w8a8-mxfp8", use_swizzle=False)


def dequant(q, sf):
    rows, k = q.shape
    scale = ue8m0_to_float32(sf[:rows, : k // SCALE_GROUP]).unsqueeze(-1)
    q = q.float().view(rows, k // SCALE_GROUP, SCALE_GROUP)
    return (q * scale).view(rows, k)


def make_operands(m, n, k):
    x = make_bf16(m, k, 0.1)
    w = make_bf16(n, k, k**-0.5)
    x_q, sfa = quant(x)
    w_q, sfb = quant(w)
    return x, w, x_q, sfa, w_q, sfb


def run(x_q, w_q, sfa, sfb, **kwargs):
    out = torch.empty(x_q.shape[0], w_q.shape[0], dtype=torch.bfloat16)
    low_latency_mma_gemm(x_q, w_q, sfa, sfb, out, **kwargs)
    return out


def check(out, x, w, x_ref, w_ref):
    torch.testing.assert_close(out.float(), x_ref @ w_ref.T, **W8A8_MXFP8_TOL)
    bf16_out = (x @ w.T).float()
    rel = ((out.float() - bf16_out).norm() / bf16_out.norm()).item()
    assert rel < BF16_REL_TOL


def check_gemm(m, n, k, **kwargs):
    x, w, x_q, sfa, w_q, sfb = make_operands(m, n, k)
    out = run(x_q, w_q, sfa, sfb, **kwargs)
    check(out, x, w, dequant(x_q, sfa), dequant(w_q, sfb))


def strided_rows(sf, pad):
    """View of ``sf`` with ``pad`` extra bytes per row: a row pitch that is
    not 16-byte granular, which the host repacks."""
    rows, cols = sf.shape
    buf = torch.zeros(rows, cols + pad, dtype=torch.uint8)
    buf[:, :cols].copy_(sf)
    return buf[:, :cols]


@pytest.mark.parametrize("n,k", DECODE_SHAPES)
def test_decode_shapes(n, k):
    check_gemm(8, n, k)


@pytest.mark.parametrize("m", [1, 2, 4, 8, 16])
def test_m_range(m):
    check_gemm(m, 512, 4096)


@pytest.mark.parametrize("n,k,block_n,warps_k", GATE_SHAPES)
@pytest.mark.parametrize("m", [1, 4, 8, 16])
def test_gate_configs(m, n, k, block_n, warps_k):
    check_gemm(m, n, k, block_n=block_n, warps_k=warps_k)


@pytest.mark.parametrize("block_n,warps_k", TILE_CONFIGS)
def test_tile_configs(block_n, warps_k):
    check_gemm(8, 1024, 4096, block_n=block_n, warps_k=warps_k)


@pytest.mark.parametrize("warps_k", [1, 2, 4])
def test_block_n8_k1024(warps_k):
    check_gemm(8, 8192, 1024, block_n=8, warps_k=warps_k)


@pytest.mark.parametrize("n,k", ODD_GROUP_SHAPES)
@pytest.mark.parametrize("m", [8, 16])
def test_auto_config_odd_k_groups(m, n, k):
    assert prefer_low_latency_mma(m, k)
    check_gemm(m, n, k)


def test_repacked_pitch():
    check_gemm(8, 2048, 1280)


@pytest.mark.parametrize("sfa_pad,sfb_pad", [(4, 0), (0, 4)])
@pytest.mark.parametrize("m", [8, 16])
def test_strided_scales_match_contiguous(m, sfa_pad, sfb_pad):
    _, _, x_q, sfa, w_q, sfb = make_operands(m, 512, 4096)
    sfa_s = strided_rows(sfa, sfa_pad) if sfa_pad else sfa
    sfb_s = strided_rows(sfb, sfb_pad) if sfb_pad else sfb
    assert torch.equal(run(x_q, w_q, sfa, sfb), run(x_q, w_q, sfa_s, sfb_s))


def test_ring_depths_agree():
    _, _, x_q, sfa, w_q, sfb = make_operands(8, WALK_N, WALK_K)
    tile = dict(block_n=WALK_BLOCK_N, warps_k=WALK_WARPS_K)
    deep = run(x_q, w_q, sfa, sfb, stage_smem=-1, **tile)
    shallow = run(x_q, w_q, sfa, sfb, stage_smem=1, **tile)
    assert torch.equal(deep, shallow)


def test_blocked_tile_order_agrees():
    _, _, x_q, sfa, w_q, sfb = make_operands(8, WALK_N, WALK_K)
    tile = dict(block_n=WALK_BLOCK_N, warps_k=WALK_WARPS_K)
    cyclic = run(x_q, w_q, sfa, sfb, persist=1, **tile)
    blocked = run(x_q, w_q, sfa, sfb, persist=2, **tile)
    assert torch.equal(cyclic, blocked)


def test_determinism():
    _, _, x_q, sfa, w_q, sfb = make_operands(8, 512, 4096)
    assert torch.equal(run(x_q, w_q, sfa, sfb), run(x_q, w_q, sfa, sfb))


@pytest.mark.parametrize("sm_count", [64, 88, 120])
def test_sm_count_matches_full_device(sm_count):
    _, _, x_q, sfa, w_q, sfb = make_operands(8, 4096, 4096)
    tile = dict(block_n=32, warps_k=4)
    full = run(x_q, w_q, sfa, sfb, **tile)
    capped = run(x_q, w_q, sfa, sfb, sm_count=sm_count, **tile)
    assert torch.equal(full, capped)


def test_pdl_matches_plain_launch():
    _, _, x_q, sfa, w_q, sfb = make_operands(8, 4096, 4096)
    with_pdl = run(x_q, w_q, sfa, sfb, use_pdl=True)
    without_pdl = run(x_q, w_q, sfa, sfb, use_pdl=False)
    assert torch.equal(with_pdl, without_pdl)


@pytest.mark.parametrize("use_pdl", [True, False])
@pytest.mark.parametrize("m", [5, 8])
def test_quant_chain_graph_replay(m, use_pdl):
    """group_wise_quant -> GEMM -> group_wise_quant -> GEMM captured in one
    CUDA graph: a missing or misplaced grid-dependency wait under PDL shows
    up as a stale operand on replay."""
    k1, n1, n2 = 4096, 2048, 4096
    w1_q, sfb1 = quant(torch.randn(n1, k1, dtype=torch.bfloat16) * k1**-0.5)
    w2_q, sfb2 = quant(torch.randn(n2, n1, dtype=torch.bfloat16) * n1**-0.5)
    inputs = torch.randn(16, m, k1, dtype=torch.bfloat16) * 0.1
    src = torch.empty(m, k1, dtype=torch.bfloat16)
    hid = torch.empty(m, n1, dtype=torch.bfloat16)
    out = torch.empty(m, n2, dtype=torch.bfloat16)
    tile = dict(block_n=16, warps_k=8, use_pdl=use_pdl)

    def step():
        x_q, sfa = group_wise_quant(
            src, mode="w8a8-mxfp8", use_swizzle=False, use_pdl=use_pdl
        )
        low_latency_mma_gemm(x_q, w1_q, sfa, sfb1, hid, **tile)
        h_q, sfh = group_wise_quant(
            hid, mode="w8a8-mxfp8", use_swizzle=False, use_pdl=use_pdl
        )
        low_latency_mma_gemm(h_q, w2_q, sfh, sfb2, out, **tile)

    expected = []
    for x in inputs:
        src.copy_(x)
        step()
        expected.append(out.clone())

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        step()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        step()

    for x, ref in zip(inputs, expected):
        src.copy_(x)
        out.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, ref)


@pytest.mark.parametrize("n,k,n_split,block_n,warps_k", DUAL_OUTPUT_SHAPES)
@pytest.mark.parametrize("m", [1, 8, 16])
def test_dual_output_matches_single(m, n, k, n_split, block_n, warps_k):
    _, _, x_q, sfa, w_q, sfb = make_operands(m, n, k)
    tile = dict(block_n=block_n, warps_k=warps_k)
    full = run(x_q, w_q, sfa, sfb, **tile)
    out0 = torch.empty(m, n_split, dtype=torch.bfloat16)
    out1 = torch.empty(m, n - n_split, dtype=torch.bfloat16)
    low_latency_mma_gemm(
        x_q, w_q, sfa, sfb, out0, output2=out1, n_split=n_split, **tile
    )
    assert torch.equal(out0, full[:, :n_split])
    assert torch.equal(out1, full[:, n_split:])


def test_dual_output_rejects_unaligned_split():
    m, n, k, n_split = 8, 10240, 1024, 8200
    _, _, x_q, sfa, w_q, sfb = make_operands(m, n, k)
    out0 = torch.empty(m, n_split, dtype=torch.bfloat16)
    out1 = torch.empty(m, n - n_split, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="n_split must be divisible by block_n"):
        low_latency_mma_gemm(
            x_q,
            w_q,
            sfa,
            sfb,
            out0,
            block_n=16,
            warps_k=4,
            output2=out1,
            n_split=n_split,
        )


@pytest.mark.parametrize("pad_sfa", [False, True])
@pytest.mark.parametrize("n,k1,k2,block_n,warps_k", DUAL_A_SHAPES)
@pytest.mark.parametrize("m", [1, 8, 16])
def test_dual_a_concat_k(m, n, k1, k2, block_n, warps_k, pad_sfa):
    x1, x2 = make_bf16(m, k1, 0.1), make_bf16(m, k2, 0.1)
    w = make_bf16(n, k1 + k2, (k1 + k2) ** -0.5)
    (x1_q, sfa1), (x2_q, sfa2), (w_q, sfb) = quant(x1), quant(x2), quant(w)
    a1 = strided_rows(sfa1, 4) if pad_sfa else sfa1
    a2 = strided_rows(sfa2, 4) if pad_sfa else sfa2
    tile = dict(block_n=block_n, warps_k=warps_k)
    out = run(x1_q, w_q, a1, sfb, input2=x2_q, sfa2=a2, **tile)

    x_ref = torch.cat([dequant(x1_q, sfa1), dequant(x2_q, sfa2)], dim=1)
    check(out, torch.cat([x1, x2], dim=1), w, x_ref, dequant(w_q, sfb))
    # Single A on the concatenated operands walks the same k-groups with the
    # same scales.
    xc_q = torch.cat([x1_q, x2_q], dim=1).contiguous()
    sfac = torch.cat([sfa1, sfa2], dim=1).contiguous()
    assert torch.equal(out, run(xc_q, w_q, sfac, sfb, **tile))


def test_dual_a_rejects_sfa2_dtype():
    m, n, k = 8, 4096, 2048
    _, _, x_q, sfa, w_q, sfb = make_operands(m, n, 2 * k)
    out = torch.empty(m, n, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="sfa2 must have sfa's dtype"):
        low_latency_mma_gemm(
            x_q[:, :k].contiguous(),
            w_q,
            sfa,
            sfb,
            out,
            input2=x_q[:, k:].contiguous(),
            sfa2=torch.zeros(m, k // 512, dtype=torch.int32),
        )


def test_dual_a_rejects_partial_k_block():
    m, n = 8, 4096
    _, _, _, _, w_q, sfb = make_operands(m, n, 4096)
    xa_q, sa = quant(torch.randn(m, 2016, dtype=torch.bfloat16))
    xb_q, sb = quant(torch.randn(m, 2080, dtype=torch.bfloat16))
    out = torch.empty(m, n, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="multiples of 128"):
        low_latency_mma_gemm(xa_q, w_q, sa, sfb, out, input2=xb_q, sfa2=sb)


@pytest.mark.parametrize("m", [17, 32])
def test_rejects_m_above_16(m):
    _, _, x_q, sfa, w_q, sfb = make_operands(m, 512, 4096)
    with pytest.raises(RuntimeError, match="M <= 16"):
        run(x_q, w_q, sfa, sfb)


def test_rejects_resident_a_over_smem():
    # A 16 x 16384 resident A (256 KiB) exceeds the shared memory budget.
    _, _, x_q, sfa, w_q, sfb = make_operands(16, 4096, 16384)
    with pytest.raises(RuntimeError, match="cannot serve"):
        run(x_q, w_q, sfa, sfb)


@pytest.mark.parametrize("k", [1152, 3200])
def test_rejects_odd_k_groups(k):
    assert not prefer_low_latency_mma(8, k)
    _, _, x_q, sfa, w_q, sfb = make_operands(8, 512, k)
    with pytest.raises(RuntimeError, match="cannot serve"):
        run(x_q, w_q, sfa, sfb)


def test_rejects_k_not_multiple_of_128():
    x = torch.empty(8, 64, dtype=torch.float8_e4m3fn)
    w = torch.empty(512, 64, dtype=torch.float8_e4m3fn)
    sfa = torch.empty(8, 2, dtype=torch.uint8)
    sfb = torch.empty(512, 2, dtype=torch.uint8)
    with pytest.raises(RuntimeError, match="must be divisible by 128"):
        run(x, w, sfa, sfb)


@pytest.mark.parametrize(
    "sfa_dtype,sfb_dtype",
    [(torch.int32, torch.uint8), (torch.uint8, torch.float32)],
)
def test_rejects_non_group32_scales(sfa_dtype, sfb_dtype):
    x = torch.empty(8, 128, dtype=torch.float8_e4m3fn)
    w = torch.empty(512, 128, dtype=torch.float8_e4m3fn)
    sfa = torch.empty(8, 4, dtype=sfa_dtype)
    sfb = torch.empty(512, 4, dtype=sfb_dtype)
    with pytest.raises(RuntimeError, match="supports group-32 scales only"):
        run(x, w, sfa, sfb)


def test_rejects_n_not_multiple_of_block_n():
    x = torch.empty(8, 128, dtype=torch.float8_e4m3fn)
    w = torch.empty(520, 128, dtype=torch.float8_e4m3fn)
    sfa = torch.empty(8, 4, dtype=torch.uint8)
    sfb = torch.empty(520, 4, dtype=torch.uint8)
    with pytest.raises(RuntimeError, match="N must be divisible by block_n"):
        run(x, w, sfa, sfb, block_n=16)
