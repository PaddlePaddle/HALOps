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

from halops import group_wise_quant, low_latency_mma_gemm
from tests.utils.gemm_utils import W8A8_MXFP8_TOL, ue8m0_to_float32

SCALE_GROUP = 32
# The kernel copies whole scale rows rounded up to 16 bytes.
SCALE_ROW_ALIGN = 16
SCALE_LAYOUT_ERROR = "sfa / sfb / sfa2 must be plain row-major"
# Relative Frobenius error vs the unquantized bf16 matmul; catches a reference
# that misreads the scale layout the same way the kernel does.
BF16_REL_TOL = 0.08

# Shapes are drawn from a fixed-seed RNG so the parametrization is stable
# across runs while not pinning the tests to a handful of hand-picked sizes.
SHAPE_SEED = 2026
_rng = random.Random(SHAPE_SEED)

# Mirrors kLlTiles: block_n * warps_k <= 128. N = 65 * block_n rules out
# wider tiles and K / 128 = 6 * warps_k (or 2 * warps_k at the deepest splits)
# rules out larger K splits, so on a small grid the ll_cost model lands on
# exactly this tile; retuning ll_cost needs these shapes rechecked.
TILES = [(bn, wk) for bn in (8, 16, 32) for wk in (1, 2, 4, 8, 16) if bn * wk <= 128]
TILE_SM_COUNT = 8


def tile_shape(block_n, warps_k):
    return 65 * block_n, 128 * warps_k * (2 if warps_k >= 8 else 6)


# (m, n, k) landing on each B ring depth tier (1, 2, 4 and 8 stages).
STAGE_TIER_SHAPES = [
    (1, 512, 23808),
    (1, 4096, 7168),
    (1, 4096, 1024),
    (1, 512, 256),
]


def rand_n(step=8, lo=512, hi=16384):
    return _rng.randrange(-(-lo // step) * step, hi + 1, step)


def rand_k(lo=1024, hi=8192, step=256):
    """K a multiple of 256: whole two-k-block B phases."""
    return _rng.randrange(-(-lo // step) * step, hi + 1, step)


def rand_odd_group_k():
    """K / 128 even but not a power of two, so the dispatcher has to shrink
    warps_k; K / 32 not a multiple of 16 also needs padded scale rows."""
    while True:
        j = _rng.randrange(3, 33)
        if j & (j - 1):
            return 256 * j


# Decode projections, M = 8.
DECODE_SHAPES = [(rand_n(64), rand_k()) for _ in range(8)]

# (n, k) with N only 8-aligned and K only 256-aligned.
RANDOM_SHAPES = [(rand_n(), rand_k()) for _ in range(9)]

ODD_GROUP_SHAPES = [(rand_n(64), rand_odd_group_k()) for _ in range(4)]


def rand_dual_output_shape():
    n, k = rand_n(), rand_k(hi=4096)
    return n, k, _rng.randrange(8, n, 8)


# (n, k, n_split) for the segmented dual output.
DUAL_OUTPUT_SHAPES = [rand_dual_output_shape() for _ in range(5)]


def rand_dual_a_shape():
    k1 = _rng.randrange(512, 4097, 512)
    return rand_n(), k1, _rng.randrange(256, 2049, 256)


# (n, k1, k2) for the dual-A concat-K path: K1 % 512 == 0, K1 + K2 % 256 == 0,
# and a random K2 often gives sfa2 rows that are not 16-byte granular.
DUAL_A_SHAPES = [rand_dual_a_shape() for _ in range(5)]


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


def aligned_pitch(cols):
    return -(-cols // SCALE_ROW_ALIGN) * SCALE_ROW_ALIGN


def strided_rows(sf, pad=0):
    """[rows, K / 32] view of ``sf`` copied into a buffer whose row pitch is
    the 16-byte-rounded row plus ``pad`` bytes."""
    rows, cols = sf.shape
    buf = torch.zeros(rows, aligned_pitch(cols) + pad, dtype=torch.uint8)
    buf[:, :cols].copy_(sf)
    return buf[:, :cols]


def aligned_scale(sf):
    """``sf`` when its rows are already 16-byte granular, else the view of a
    copy padded to whole 16-byte rows."""
    return sf if sf.shape[1] % SCALE_ROW_ALIGN == 0 else strided_rows(sf)


def quant(x):
    x_q, sf = group_wise_quant(x, mode="w8a8-mxfp8", use_swizzle=False)
    return x_q, aligned_scale(sf)


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


@pytest.mark.parametrize("n,k", DECODE_SHAPES)
def test_decode_shapes(n, k):
    check_gemm(8, n, k)


@pytest.mark.parametrize("m", [1, 2, 4, 8, 9, 16])
def test_m_range(m):
    check_gemm(m, 512, 4096)


@pytest.mark.parametrize("n,k", RANDOM_SHAPES)
@pytest.mark.parametrize("m", [1, 4, 8, 16])
def test_random_shapes(m, n, k):
    check_gemm(m, n, k)


@pytest.mark.parametrize("block_n,warps_k", TILES)
@pytest.mark.parametrize("m", [1, 16])
def test_tiles(m, block_n, warps_k):
    n, k = tile_shape(block_n, warps_k)
    check_gemm(m, n, k, sm_count=TILE_SM_COUNT)


@pytest.mark.parametrize("m,n,k", STAGE_TIER_SHAPES)
def test_stage_tiers(m, n, k):
    check_gemm(m, n, k)


@pytest.mark.parametrize("n,k", ODD_GROUP_SHAPES)
@pytest.mark.parametrize("m", [8, 16])
def test_odd_k_groups(m, n, k):
    check_gemm(m, n, k)


def test_padded_scale_rows():
    # K = 1280: 40-byte scale rows, padded to 48.
    check_gemm(8, 2048, 1280)


@pytest.mark.parametrize("sfa_pad,sfb_pad", [(16, 0), (0, 16)])
@pytest.mark.parametrize("m", [8, 16])
def test_strided_scales_match_contiguous(m, sfa_pad, sfb_pad):
    _, _, x_q, sfa, w_q, sfb = make_operands(m, 512, 4096)
    sfa_s = strided_rows(sfa, sfa_pad) if sfa_pad else sfa
    sfb_s = strided_rows(sfb, sfb_pad) if sfb_pad else sfb
    assert torch.equal(run(x_q, w_q, sfa, sfb), run(x_q, w_q, sfa_s, sfb_s))


def test_determinism():
    _, _, x_q, sfa, w_q, sfb = make_operands(8, 512, 4096)
    assert torch.equal(run(x_q, w_q, sfa, sfb), run(x_q, w_q, sfa, sfb))


@pytest.mark.parametrize("sm_count", [1, 64, 88, 120])
def test_sm_count_matches_full_device(sm_count):
    # 4096 x 1024 keeps warps_k = 4 at every grid size, so the K reduction
    # order is unchanged and only the tile walk differs.
    _, _, x_q, sfa, w_q, sfb = make_operands(8, 4096, 1024)
    full = run(x_q, w_q, sfa, sfb)
    capped = run(x_q, w_q, sfa, sfb, sm_count=sm_count)
    assert torch.equal(full, capped)


def test_more_sms_than_tiles():
    check_gemm(8, 16, 4096)


def test_pdl_matches_plain_launch():
    _, _, x_q, sfa, w_q, sfb = make_operands(8, 4096, 4096)
    with_pdl = run(x_q, w_q, sfa, sfb, use_pdl=True)
    without_pdl = run(x_q, w_q, sfa, sfb, use_pdl=False)
    assert torch.equal(with_pdl, without_pdl)


@pytest.mark.parametrize("use_pdl", [True, False])
@pytest.mark.parametrize("m", [5, 8])
def test_quant_chain_graph_replay(m, use_pdl):
    """group_wise_quant -> GEMM -> group_wise_quant -> GEMM captured in one
    CUDA graph: a missing or misplaced grid-dependency wait or trigger under
    PDL shows up as a stale operand on replay."""
    k1, n1, n2 = 4096, 2048, 4096
    w1_q, sfb1 = quant(torch.randn(n1, k1, dtype=torch.bfloat16) * k1**-0.5)
    w2_q, sfb2 = quant(torch.randn(n2, n1, dtype=torch.bfloat16) * n1**-0.5)
    inputs = torch.randn(16, m, k1, dtype=torch.bfloat16) * 0.1
    src = torch.empty(m, k1, dtype=torch.bfloat16)
    hid = torch.empty(m, n1, dtype=torch.bfloat16)
    out = torch.empty(m, n2, dtype=torch.bfloat16)

    def step():
        x_q, sfa = group_wise_quant(
            src, mode="w8a8-mxfp8", use_swizzle=False, use_pdl=use_pdl
        )
        low_latency_mma_gemm(x_q, w1_q, sfa, sfb1, hid, use_pdl=use_pdl)
        h_q, sfh = group_wise_quant(
            hid, mode="w8a8-mxfp8", use_swizzle=False, use_pdl=use_pdl
        )
        low_latency_mma_gemm(h_q, w2_q, sfh, sfb2, out, use_pdl=use_pdl)

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


@pytest.mark.parametrize("n,k,n_split", DUAL_OUTPUT_SHAPES)
@pytest.mark.parametrize("m", [1, 8, 16])
def test_dual_output_matches_single(m, n, k, n_split):
    _, _, x_q, sfa, w_q, sfb = make_operands(m, n, k)
    full = run(x_q, w_q, sfa, sfb)
    out0 = torch.empty(m, n_split, dtype=torch.bfloat16)
    out1 = torch.empty(m, n - n_split, dtype=torch.bfloat16)
    low_latency_mma_gemm(x_q, w_q, sfa, sfb, out0, output2=out1, n_split=n_split)
    assert torch.equal(out0, full[:, :n_split])
    assert torch.equal(out1, full[:, n_split:])


def test_dual_output_rejects_unaligned_split():
    m, n, k, n_split = 8, 10240, 1024, 8196
    _, _, x_q, sfa, w_q, sfb = make_operands(m, n, k)
    out0 = torch.empty(m, n_split, dtype=torch.bfloat16)
    out1 = torch.empty(m, n - n_split, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="n_split must be divisible by 8"):
        low_latency_mma_gemm(x_q, w_q, sfa, sfb, out0, output2=out1, n_split=n_split)


@pytest.mark.parametrize("pad_sfa", [False, True])
@pytest.mark.parametrize("n,k1,k2", DUAL_A_SHAPES)
@pytest.mark.parametrize("m", [1, 8, 16])
def test_dual_a_concat_k(m, n, k1, k2, pad_sfa):
    x1, x2 = make_bf16(m, k1, 0.1), make_bf16(m, k2, 0.1)
    w = make_bf16(n, k1 + k2, (k1 + k2) ** -0.5)
    (x1_q, sfa1), (x2_q, sfa2), (w_q, sfb) = quant(x1), quant(x2), quant(w)
    a1 = strided_rows(sfa1, SCALE_ROW_ALIGN) if pad_sfa else sfa1
    a2 = strided_rows(sfa2, SCALE_ROW_ALIGN) if pad_sfa else sfa2
    out = run(x1_q, w_q, a1, sfb, input2=x2_q, sfa2=a2)

    x_ref = torch.cat([dequant(x1_q, sfa1), dequant(x2_q, sfa2)], dim=1)
    check(out, torch.cat([x1, x2], dim=1), w, x_ref, dequant(w_q, sfb))
    # Single A on the concatenated operands walks the same k-groups with the
    # same scales.
    xc_q = torch.cat([x1_q, x2_q], dim=1).contiguous()
    sfac = aligned_scale(torch.cat([sfa1, sfa2], dim=1))
    assert torch.equal(out, run(xc_q, w_q, sfac, sfb))


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


def test_dual_a_rejects_k1_not_multiple_of_512():
    m, n, k1, k2 = 8, 4096, 1920, 2176
    (x1_q, sfa1), (x2_q, sfa2) = (
        quant(torch.randn(m, k1, dtype=torch.bfloat16)),
        quant(torch.randn(m, k2, dtype=torch.bfloat16)),
    )
    _, _, _, _, w_q, sfb = make_operands(m, n, k1 + k2)
    out = torch.empty(m, n, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="K1 must be a multiple of 512"):
        low_latency_mma_gemm(x1_q, w_q, sfa1, sfb, out, input2=x2_q, sfa2=sfa2)


@pytest.mark.parametrize("operand", ["sfa", "sfb"])
@pytest.mark.parametrize("layout", ["contiguous", "strided"])
def test_rejects_unaligned_scale_rows(operand, layout):
    # K = 1280 gives 40-byte contiguous rows; a 4-byte pad gives a 36-byte
    # pitch at K = 1024. Neither pitch is 16-byte granular.
    k = 1280 if layout == "contiguous" else 1024
    _, _, x_q, sfa, w_q, sfb = make_operands(8, 512, k)
    if layout == "contiguous":
        bad = (sfa if operand == "sfa" else sfb).contiguous()
    else:
        bad = strided_rows(sfa if operand == "sfa" else sfb, 4)
    if operand == "sfa":
        sfa = bad
    else:
        sfb = bad
    with pytest.raises(RuntimeError, match=SCALE_LAYOUT_ERROR):
        run(x_q, w_q, sfa, sfb)


@pytest.mark.parametrize("m", [17, 32])
def test_rejects_m_above_16(m):
    _, _, x_q, sfa, w_q, sfb = make_operands(m, 512, 4096)
    with pytest.raises(RuntimeError, match="M <= 16"):
        run(x_q, w_q, sfa, sfb)


def test_rejects_resident_a_over_smem():
    # A 16 x 16384 resident A (256 KiB) exceeds the shared memory budget.
    _, _, x_q, sfa, w_q, sfb = make_operands(16, 4096, 16384)
    with pytest.raises(RuntimeError, match="leaves no room for a B stage"):
        run(x_q, w_q, sfa, sfb)


@pytest.mark.parametrize("k", [1152, 3200])
def test_rejects_odd_k_groups(k):
    _, _, x_q, sfa, w_q, sfb = make_operands(8, 512, k)
    with pytest.raises(RuntimeError, match="K must be divisible by 256"):
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


def test_rejects_n_not_multiple_of_8():
    x = torch.empty(8, 256, dtype=torch.float8_e4m3fn)
    w = torch.empty(516, 256, dtype=torch.float8_e4m3fn)
    sfa = torch.empty(8, 8, dtype=torch.uint8)
    sfb = torch.empty(516, 8, dtype=torch.uint8)
    with pytest.raises(RuntimeError, match="N must be divisible by 8"):
        run(x, w, sfa, sfb)


def test_rejects_swizzled_sfa():
    m, n, k = 8, 4096, 4096
    x = torch.randn(m, k, dtype=torch.bfloat16)
    x_q, sfa = group_wise_quant(x, mode="w8a8-mxfp8", use_swizzle=True)
    _, _, _, _, w_q, sfb = make_operands(m, n, k)
    with pytest.raises(RuntimeError, match=SCALE_LAYOUT_ERROR):
        run(x_q, w_q, sfa, sfb)


def test_rejects_misaligned_weight():
    m, n, k = 8, 512, 2048
    _, _, x_q, sfa, w_q, sfb = make_operands(m, n, k)
    buf = torch.empty(n * k + 1, dtype=torch.float8_e4m3fn)
    w_off = buf[1:].view(n, k)
    w_off.copy_(w_q)
    with pytest.raises(RuntimeError, match="weight TMA descriptor was rejected"):
        run(x_q, w_off, sfa, sfb)
