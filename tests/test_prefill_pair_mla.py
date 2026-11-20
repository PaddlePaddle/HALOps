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

from halops import prefill_pair_mla
from tests.utils.prefill_pair_mla_utils import (
    DIM,
    HEADS,
    SCALE,
    Req,
    cp_local,
    cu_offsets,
    make_batch,
    make_cache,
    positions,
    q_norm_rope,
    references,
    rope_freqs,
    score,
)

# Both the kernel and FlashMLA round P and O to bf16: against the fp32
# reference they measure RMS ~2.0e-3 and a worst element ~7e-3.
RMS_TOL = 2.5e-3
MAX_TOL = 1.2e-2

# One request, the chunk being the last T of CTX positions (T, CTX).  Pairs
# of at most 5 tiles (t2, t64, t2048_from0) run the kQst kernels.  1024/64K,
# 256/1M (c128 width >= 512, T >= 256) and 4096/32K (width 256, T >= 4096)
# pre-dequantize c128 to bf16 first; the others read it as fp8.
UNIFORM_CASES = [
    pytest.param(2, 256, id="t2", marks=pytest.mark.quick),
    pytest.param(3, 65536, id="t3_odd"),
    pytest.param(64, 4096, id="t64", marks=pytest.mark.quick),
    pytest.param(2048, 2048, id="t2048_from0"),
    pytest.param(1024, 32768, id="t1024_ctx32k"),
    pytest.param(1024, 65536, id="t1024_ctx64k"),
    pytest.param(4096, 32768, id="t4096_ctx32k"),
    pytest.param(64, 1048576, id="t64_ctx1m"),
    pytest.param(256, 1048576, id="t256_ctx1m"),
]

# Odd chunks, a 1-token request, a chunk starting at position 0.
VARLEN_REQS = [
    Req(1023, 65536),
    Req(1, 70001),
    Req(2048, 131072),
    Req(517, 517),
    Req(2, 9000),
]

# SWA replay: a replay start inside and outside the 128-key window, and
# requests with the arena overlay but no clamp.
REPLAY_REQS = [
    Req(1023, 65536, 37),
    Req(1, 70001, 0),
    Req(517, 517, 0),
    Req(2, 9000, 1),
    Req(300, 4000, 127),
    Req(777, 50000, 60, clamp=False),
    Req(64, 8000, 3, clamp=False),
]


def run(batch, sink=True, **kwargs):
    return prefill_pair_mla(
        batch.q,
        batch.swa,
        batch.swa_idx,
        batch.swa_len,
        batch.c128,
        batch.c_idx,
        batch.c_len,
        batch.sink if sink else None,
        SCALE,
        **kwargs,
    )


def run_flashmla(batch):
    fm = pytest.importorskip("sgl_kernel.flash_mla")
    return fm.flash_mla_with_kvcache(
        q=batch.q,
        k_cache=batch.swa,
        head_dim_v=DIM,
        block_table=None,
        cache_seqlens=None,
        tile_scheduler_metadata=fm.get_mla_metadata()[0],
        softmax_scale=SCALE,
        is_fp8_kvcache=True,
        indices=batch.swa_idx,
        topk_length=batch.swa_len,
        attn_sink=batch.sink,
        extra_k_cache=batch.c128,
        extra_indices_in_kvcache=batch.c_idx,
        extra_topk_length=batch.c_len,
    )[0]


def check(o, batch, sink=True, rows=None):
    """Against the fp32 reference on rows (default every row)."""
    assert torch.isfinite(o).all(), "non-finite output"
    if rows is None:
        rows = list(range(batch.q.shape[0]))
    refs = references(batch, rows, sink)
    worst, rms = score(o, rows, refs)
    assert rms <= RMS_TOL and worst <= MAX_TOL, f"rms {rms:.3e} max {worst:.3e}"
    return rows, refs, worst, rms


def varlen_kwargs(lens, swa_shift=1):
    cu_seqlens, cu_pairs = cu_offsets(lens)
    return dict(cu_seqlens=cu_seqlens, cu_pairs=cu_pairs, swa_shift=swa_shift)


def cp_reqs(cp):
    """VARLEN_REQS with the token count padded to a multiple of cp."""
    reqs = list(VARLEN_REQS)
    pad = -sum(r.tokens for r in reqs) % cp
    if pad:  # the padding rows form one more request
        reqs.append(Req(pad, pad))
    return reqs


CP_RANKS = [
    pytest.param(cp, rank, id=f"cp{cp}_rank{rank}")
    for cp in (2, 4, 8)
    for rank in (0, cp - 1)
]


@pytest.mark.parametrize("t,ctx", UNIFORM_CASES)
def test_uniform(t, ctx):
    batch = make_batch([Req(t, ctx)])
    check(run(batch, tokens_per_req=t), batch)


def test_tokens_per_req_matches_varlen():
    """tokens_per_req = t is the varlen batch of T / t requests of t tokens."""
    batch = make_batch([Req(256, 8192)] * 4)
    o = run(batch, tokens_per_req=256)
    check(o, batch)
    assert torch.equal(o, run(batch, **varlen_kwargs([256] * 4)))


@pytest.mark.quick
def test_varlen():
    batch = make_batch(VARLEN_REQS)
    check(run(batch, **varlen_kwargs([r.tokens for r in VARLEN_REQS])), batch)


def test_replay():
    batch = make_batch(REPLAY_REQS)
    check(run(batch, **varlen_kwargs([r.tokens for r in REPLAY_REQS])), batch)


@pytest.mark.parametrize("cp,rank", CP_RANKS)
def test_cp(cp, rank):
    """CP round-robin: rank's tokens of a cp-way split."""
    local, lens = cp_local(make_batch(cp_reqs(cp)), cp, rank)
    check(run(local, **varlen_kwargs(lens, swa_shift=cp)), local)


def test_no_sink():
    batch = make_batch([Req(517, 517)])
    check(run(batch, sink=False, tokens_per_req=517), batch, sink=False)


@pytest.mark.parametrize("sink", [True, False], ids=["sink", "no_sink"])
@pytest.mark.parametrize(
    "req",
    [Req(2, 65536), Req(512, 262144)],
    ids=["combine", "in_kernel_merge"],
)
def test_no_keys(req, sink):
    """Tokens with no valid key output zeros, also when their pair is split:
    one pair is split by the combine kernel, 256 pairs are cut by the
    in-kernel merge's range boundaries."""
    batch = make_batch([req])
    swa_len, c_len = batch.swa_len.clone(), batch.c_len.clone()
    swa_len[0::2] = 0
    c_len[0::2] = 0
    batch = batch._replace(swa_len=swa_len, c_len=c_len)
    o = run(batch, sink, tokens_per_req=req.tokens).view(-1, HEADS, DIM)
    assert torch.isfinite(o).all(), "non-finite output"
    assert not o[0::2].any(), "a token with no keys has non-zero output"
    check(o, batch, sink, rows=list(range(1, req.tokens, 2)))


def skip_keys(idx, lens, rows):
    """idx with some row ids inside the lengths replaced, the same way in
    every token so pairs keep their structure: v % 7 == 3 by -1 and
    v % 11 == 5 by v + rows (past the pool)."""
    inside = torch.arange(idx.shape[-1], device=idx.device) < lens[:, None, None]
    far = torch.where(inside & (idx % 11 == 5), idx + rows, idx)
    return torch.where(inside & (idx % 7 == 3), -1, far)


@pytest.mark.parametrize(
    "req", [Req(1024, 32768), Req(1024, 65536)], ids=["fp8", "predq"]
)
def test_invalid_keys(req):
    """Indices inside the lengths that are negative or past the pool are
    skipped: by the kernel's key mask, and with predq also in the bf16 copy."""
    batch = make_batch([req])
    batch = batch._replace(
        swa_idx=skip_keys(
            batch.swa_idx, batch.swa_len, batch.swa.shape[0] * batch.swa.shape[1]
        ),
        c_idx=skip_keys(
            batch.c_idx, batch.c_len, batch.c128.shape[0] * batch.c128.shape[1]
        ),
    )
    check(run(batch, tokens_per_req=req.tokens), batch)


def test_c128_not_prefix():
    """A pair whose c128 lists are not a prefix of its request's last token's
    reads them as fp8 instead of the pre-dequantized copy: pair (100, 101)
    with both lists moved one row up, pair (600, 601) with only 601's rows
    after a's first 448 changed.  Read from the copy, their full tiles would
    hold the wrong rows."""
    req = Req(1024, 65536)
    batch = make_batch([req])
    c_idx, c_len = batch.c_idx.clone(), batch.c_len.clone()
    cols = torch.arange(c_idx.shape[-1], device="cuda")
    for t in (100, 101):
        c_idx[t, 0] = torch.where(cols < c_len[t], cols + 1, -1)
    c_len[600], c_len[601] = 448, 512
    c_idx[600, 0] = torch.where(cols < 448, cols, -1)
    c_idx[601, 0] = torch.where(cols < 448, cols, cols + 1)
    batch = batch._replace(c_idx=c_idx, c_len=c_len)
    check(run(batch, tokens_per_req=req.tokens), batch)


# A pair split by the combine kernel, the kQst kernels, the streaming kernels
# with predq.
OUT_CASES = [
    pytest.param([Req(2, 65536)], id="combine"),
    pytest.param([Req(2048, 2048)], id="qst"),
    pytest.param(VARLEN_REQS, id="streaming"),
]


@pytest.mark.parametrize("use_pdl", [True, False], ids=["pdl", "no_pdl"])
@pytest.mark.parametrize("reqs", OUT_CASES)
def test_out_buffer(reqs, use_pdl):
    """A given out is written in place and returned; repeated calls agree."""
    batch = make_batch(reqs)
    kwargs = varlen_kwargs([r.tokens for r in reqs])
    out = torch.empty(batch.q.shape[0], HEADS, DIM, dtype=torch.bfloat16)
    o = run(batch, out=out, use_pdl=use_pdl, **kwargs)
    assert o.data_ptr() == out.data_ptr()
    check(o, batch)
    assert torch.equal(run(batch, **kwargs).view_as(o), o)


@pytest.mark.parametrize("swa_page,c128_page", [(64, 1), (256, 64)])
def test_page_size(swa_page, c128_page):
    batch = make_batch(VARLEN_REQS, swa_page, c128_page)
    check(run(batch, **varlen_kwargs([r.tokens for r in VARLEN_REQS])), batch)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs")
@pytest.mark.parametrize("reqs", OUT_CASES[::2])
def test_other_device(reqs):
    """cuda:1 inputs, the current device being cuda:0, run on cuda:1 with its
    own workspaces and give cuda:0's bits."""
    batch = make_batch(reqs)
    kwargs = varlen_kwargs([r.tokens for r in reqs])
    o = run(batch, **kwargs)

    def to1(x):  # keeps the caches' padded page stride
        if not torch.is_tensor(x):
            return x
        y = torch.empty_strided(x.shape, x.stride(), dtype=x.dtype, device="cuda:1")
        return y.copy_(x)

    batch1 = batch._replace(**{k: to1(v) for k, v in batch._asdict().items()})
    o1 = run(batch1, **{k: to1(v) for k, v in kwargs.items()})
    assert o1.device == torch.device("cuda:1")
    assert torch.equal(o1.cpu(), o.cpu())
    assert torch.equal(run(batch, **kwargs), o)


@pytest.mark.parametrize("reqs", OUT_CASES[::2])
def test_cuda_graph(reqs):
    """A call captured after an eager one of the same shape replays on new q
    like an eager call."""
    batch = make_batch(reqs)
    kwargs = varlen_kwargs([r.tokens for r in reqs])
    run(batch, **kwargs)  # sizes the workspaces outside the capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        o = run(batch, **kwargs)
    for _ in range(2):
        batch.q.copy_(torch.randn_like(batch.q))
        graph.replay()
        assert torch.equal(o, run(batch, **kwargs))


def test_empty():
    batch = make_batch([Req(2, 256)])
    batch = batch._replace(
        q=batch.q[:0],
        swa_idx=batch.swa_idx[:0],
        swa_len=batch.swa_len[:0],
        c_idx=batch.c_idx[:0],
        c_len=batch.c_len[:0],
    )
    assert run(batch, tokens_per_req=2).numel() == 0


@pytest.mark.parametrize(
    "reqs",
    [[Req(1024, 65536)], VARLEN_REQS, REPLAY_REQS],
    ids=["uniform", "varlen", "replay"],
)
def test_vs_flashmla(reqs):
    """Not less accurate than FlashMLA on the same inputs."""
    batch = make_batch(reqs)
    o = run(batch, **varlen_kwargs([r.tokens for r in reqs]))
    rows, refs, worst, rms = check(o, batch)
    fm_worst, fm_rms = score(run_flashmla(batch), rows, refs)
    assert (
        rms <= fm_rms * 1.02 and worst <= fm_worst * 1.25
    ), f"rms {rms:.3e} vs flashmla {fm_rms:.3e}, max {worst:.3e} vs {fm_worst:.3e}"


# Fused q_norm_rope: q is a raw projection (per-head scales, a few outlier
# channels).  Req(64, 256) runs the kQst kernels, the others the streaming
# ones, Req(1024, 32768) with fp8 c128.
QNR_EPS = 1e-6
QNR_CASES = [
    pytest.param([Req(64, 256)], id="short", marks=pytest.mark.quick),
    pytest.param([Req(1024, 32768)], id="uniform_fp8"),
    pytest.param([Req(1024, 65536)], id="uniform"),
    pytest.param(VARLEN_REQS, id="varlen"),
    pytest.param(REPLAY_REQS, id="replay"),
]


def qnr_inputs(reqs):
    """A batch with a raw q, its token positions and RoPE frequencies."""
    batch = make_batch(reqs)
    t = batch.q.shape[0]
    q = torch.randn(t, HEADS, DIM, device="cuda")
    q *= torch.rand(1, HEADS, 1, device="cuda") * 4 + 0.5
    q[..., torch.randint(0, DIM, (8,), device="cuda")] *= 20
    batch = batch._replace(q=q.bfloat16())
    return batch, positions(reqs), rope_freqs(max(r.ctx for r in reqs))


def run_qnr(batch, pos, freqs, **kwargs):
    return run(batch, q_positions=pos, q_freqs=freqs, q_norm_eps=QNR_EPS, **kwargs)


def score_qnr(o, batch, q_ref):
    assert torch.isfinite(o).all(), "non-finite output"
    rows = list(range(batch.q.shape[0]))
    return score(o, rows, references(batch, rows, q=q_ref))


@pytest.mark.parametrize("reqs", QNR_CASES)
def test_qnr(reqs):
    """Against fp32 q_norm_rope and attention, and not less accurate than
    q_norm_rope rounded to bf16 first (the unfused path)."""
    batch, pos, freqs = qnr_inputs(reqs)
    kwargs = varlen_kwargs([r.tokens for r in reqs])
    q_ref = q_norm_rope(batch.q, pos, freqs, QNR_EPS)
    worst, rms = score_qnr(run_qnr(batch, pos, freqs, **kwargs), batch, q_ref)
    assert rms <= RMS_TOL and worst <= MAX_TOL, f"rms {rms:.3e} max {worst:.3e}"
    unfused = run(batch._replace(q=q_ref.bfloat16()), **kwargs)
    sep_worst, sep_rms = score_qnr(unfused, batch, q_ref)
    assert (
        rms <= sep_rms * 1.02 and worst <= sep_worst * 1.25
    ), f"rms {rms:.3e} vs unfused {sep_rms:.3e}, max {worst:.3e} vs {sep_worst:.3e}"


@pytest.mark.parametrize("layout", ["int32_pos_fp32_freqs", "q_4d", "q_strided"])
def test_qnr_layouts(layout):
    """int32 positions with float32 frequencies, a [T, 1, 64, 512] q, or a q
    with a token stride give the bits of the default layout."""
    batch, pos, freqs = qnr_inputs(VARLEN_REQS)
    kwargs = varlen_kwargs([r.tokens for r in VARLEN_REQS])
    o = run_qnr(batch, pos, freqs, **kwargs)
    if layout == "int32_pos_fp32_freqs":
        pos, freqs = pos.int(), torch.view_as_real(freqs).flatten(-2).contiguous()
    elif layout == "q_4d":
        batch = batch._replace(q=batch.q.unsqueeze(1))
    else:
        t = batch.q.shape[0]
        wide = torch.zeros(t, 3 * HEADS * DIM, dtype=torch.bfloat16, device="cuda")
        q_view = wide[:, HEADS * DIM : 2 * HEADS * DIM].view(t, HEADS, DIM)
        q_view.copy_(batch.q)
        batch = batch._replace(q=q_view)
    o2 = run_qnr(batch, pos, freqs, **kwargs)
    assert o2.shape == batch.q.shape
    assert torch.equal(o2.view_as(o), o)


@pytest.mark.parametrize("rank", [0, 3])
def test_qnr_cp(rank):
    """CP round-robin: rank's tokens and positions of a 4-way split."""
    cp = 4
    batch, pos, freqs = qnr_inputs(cp_reqs(cp))
    local, lens = cp_local(batch, cp, rank)
    lpos = pos[rank::cp].contiguous()
    o = run_qnr(local, lpos, freqs, **varlen_kwargs(lens, swa_shift=cp))
    worst, rms = score_qnr(o, local, q_norm_rope(local.q, lpos, freqs, QNR_EPS))
    assert rms <= RMS_TOL and worst <= MAX_TOL, f"rms {rms:.3e} max {worst:.3e}"


# Invalid inputs: each case replaces some arguments of a valid call (one
# 64-token request, tokens_per_req=64) and expects the host check that names
# the violated constraint.
ARG_REQ = Req(64, 256)


def valid_args():
    b = make_batch([ARG_REQ])
    return dict(
        q=b.q,
        swa_k_cache=b.swa,
        swa_indices=b.swa_idx,
        swa_lengths=b.swa_len,
        c128_k_cache=b.c128,
        c128_indices=b.c_idx,
        c128_lengths=b.c_len,
        attn_sink=b.sink,
        softmax_scale=SCALE,
        tokens_per_req=ARG_REQ.tokens,
    )


def qnr_args():
    return dict(q_positions=positions([ARG_REQ]), q_freqs=rope_freqs(ARG_REQ.ctx))


def cu_args(dtype=torch.int32, device="cuda"):
    t = ARG_REQ.tokens
    return dict(
        cu_seqlens=torch.tensor([0, t // 2, t], dtype=dtype, device=device),
        cu_pairs=torch.tensor([0, t // 4, t // 2], dtype=dtype, device=device),
    )


def strided(x):
    """x's values with a last-dim stride of 2 (not contiguous)."""
    return torch.stack([x, x], -1)[..., 0]


def raw_q(token_stride, offset=0):
    """[T, 64, 512] bf16 view of a [T, token_stride] buffer at offset."""
    buf = torch.zeros(ARG_REQ.tokens, token_stride, dtype=torch.bfloat16, device="cuda")
    return buf[:, offset : offset + HEADS * DIM].view(-1, HEADS, DIM)


def short_stride_cache(pages=2, page_size=128):
    """[pages, page_size, 1, 584] with a page stride of page_size * 576 B."""
    buf = torch.zeros(pages * page_size * 584, dtype=torch.uint8, device="cuda")
    return buf.as_strided((pages, page_size, 1, 584), (page_size * 576, 584, 584, 1))


def two_gpus(*args, **kwargs):
    need = pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs")
    return pytest.param(*args, **kwargs, marks=need)


VALID_ARGS = [
    pytest.param(lambda a: {}, id="uniform"),
    pytest.param(lambda a: {**cu_args(), "tokens_per_req": 0}, id="varlen"),
    pytest.param(lambda a: qnr_args(), id="qnr"),
]

QNR_Q_MSG = "view with contiguous heads"
Q_MSG = "q must be a contiguous bfloat16"
CU_MSG = "cu_seqlens / cu_pairs must be contiguous int32"
TPR_MSG = "tokens_per_req must be >= 2 and divide T"
OUT_MSG = "out must be a contiguous bfloat16"
DEV_MSG = "all tensors must be on q's device"
SINK_MSG = "attn_sink must be a contiguous float32"
CACHE_MSG = "k caches must be uint8"
IDX_MSG = "indices must be contiguous int32"
LEN_MSG = "lengths must be contiguous int32"
POS_MSG = "q_positions must be a contiguous int64 / int32"
FREQS_MSG = "q_freqs must be complex64"

INVALID_ARGS = [
    # swa_shift
    pytest.param(lambda a: dict(swa_shift=0), "swa_shift must be in", id="swa_shift_0"),
    pytest.param(
        lambda a: dict(swa_shift=129), "swa_shift must be in", id="swa_shift_129"
    ),
    # q, unfused and fused q_norm_rope
    pytest.param(lambda a: dict(q=a["q"].float()), Q_MSG, id="q_fp32"),
    pytest.param(lambda a: dict(q=strided(a["q"])), Q_MSG, id="q_strided"),
    pytest.param(
        lambda a: dict(q=a["q"][..., :32, :].contiguous()), Q_MSG, id="q_32_heads"
    ),
    pytest.param(
        lambda a: dict(q_positions=qnr_args()["q_positions"]),
        "q_positions and q_freqs go together",
        id="positions_only",
    ),
    pytest.param(
        lambda a: dict(q_freqs=qnr_args()["q_freqs"]),
        "q_positions and q_freqs go together",
        id="freqs_only",
    ),
    pytest.param(
        lambda a: {**qnr_args(), "q": a["q"].float()}, QNR_Q_MSG, id="qnr_q_fp32"
    ),
    pytest.param(
        lambda a: {**qnr_args(), "q": strided(a["q"])},
        QNR_Q_MSG,
        id="qnr_q_head_stride",
    ),
    pytest.param(
        lambda a: {**qnr_args(), "q": raw_q(HEADS * DIM + 4)},
        QNR_Q_MSG,
        id="qnr_q_token_stride_not_16B",
    ),
    pytest.param(
        lambda a: {**qnr_args(), "q": raw_q(HEADS * DIM + 8, offset=1)},
        QNR_Q_MSG,
        id="qnr_q_misaligned",
    ),
    # request layout
    pytest.param(
        lambda a: dict(cu_seqlens=cu_args()["cu_seqlens"]),
        "cu_seqlens and cu_pairs go together",
        id="cu_seqlens_only",
    ),
    pytest.param(
        lambda a: dict(cu_pairs=cu_args()["cu_pairs"]),
        "cu_seqlens and cu_pairs go together",
        id="cu_pairs_only",
    ),
    pytest.param(lambda a: cu_args(torch.int64), CU_MSG, id="cu_int64"),
    pytest.param(lambda a: cu_args(device="cpu"), CU_MSG, id="cu_cpu"),
    pytest.param(
        lambda a: {**cu_args(), "cu_pairs": cu_args()["cu_pairs"][:2]},
        CU_MSG,
        id="cu_size_mismatch",
    ),
    pytest.param(
        lambda a: {k: v[:1] for k, v in cu_args().items()}, CU_MSG, id="cu_no_request"
    ),
    pytest.param(lambda a: dict(tokens_per_req=0), TPR_MSG, id="tokens_per_req_0"),
    pytest.param(lambda a: dict(tokens_per_req=1), TPR_MSG, id="tokens_per_req_1"),
    pytest.param(lambda a: dict(tokens_per_req=48), TPR_MSG, id="tokens_per_req_48"),
    # out
    pytest.param(
        lambda a: dict(out=torch.empty_like(a["q"], dtype=torch.float)),
        OUT_MSG,
        id="out_fp32",
    ),
    pytest.param(
        lambda a: dict(out=strided(torch.empty_like(a["q"]))), OUT_MSG, id="out_strided"
    ),
    pytest.param(
        lambda a: dict(out=torch.empty_like(a["q"][:32])), OUT_MSG, id="out_32_tokens"
    ),
    # devices
    pytest.param(
        lambda a: dict(swa_indices=a["swa_indices"].cpu()), DEV_MSG, id="idx_cpu"
    ),
    pytest.param(
        lambda a: dict(c128_k_cache=a["c128_k_cache"].cpu()), DEV_MSG, id="cache_cpu"
    ),
    pytest.param(
        lambda a: dict(out=torch.empty_like(a["q"], device="cpu")),
        DEV_MSG,
        id="out_cpu",
    ),
    two_gpus(
        lambda a: dict(swa_lengths=a["swa_lengths"].to("cuda:1")),
        DEV_MSG,
        id="len_cuda1",
    ),
    # attn_sink
    pytest.param(
        lambda a: dict(attn_sink=a["attn_sink"].half()), SINK_MSG, id="sink_fp16"
    ),
    pytest.param(lambda a: dict(attn_sink=a["attn_sink"][:32]), SINK_MSG, id="sink_32"),
    pytest.param(
        lambda a: dict(attn_sink=a["attn_sink"].cpu()), SINK_MSG, id="sink_cpu"
    ),
    # k caches (c128 is checked first, so a swa case also passes the c128 checks)
    pytest.param(
        lambda a: dict(swa_k_cache=a["swa_k_cache"].view(torch.int8)),
        CACHE_MSG,
        id="cache_int8",
    ),
    pytest.param(
        lambda a: dict(swa_k_cache=a["swa_k_cache"][:, :, 0]), CACHE_MSG, id="cache_3d"
    ),
    pytest.param(
        lambda a: dict(swa_k_cache=a["swa_k_cache"][..., :576]),
        CACHE_MSG,
        id="cache_row_576",
    ),
    pytest.param(
        lambda a: dict(
            swa_k_cache=torch.zeros(2, 128, 1, 584, dtype=torch.uint8, device="cuda")
        ),
        CACHE_MSG,
        id="cache_page_stride_not_576x",
    ),
    pytest.param(
        lambda a: dict(swa_k_cache=short_stride_cache()),
        CACHE_MSG,
        id="cache_page_stride_short",
    ),
    pytest.param(
        lambda a: dict(c128_k_cache=make_cache(4, 3)),
        "page sizes must be powers of two, got 3",
        id="cache_page_3",
    ),
    # indices and lengths
    pytest.param(
        lambda a: dict(swa_indices=a["swa_indices"].long()), IDX_MSG, id="idx_int64"
    ),
    pytest.param(
        lambda a: dict(c128_indices=strided(a["c128_indices"])),
        IDX_MSG,
        id="idx_strided",
    ),
    pytest.param(
        lambda a: dict(swa_indices=a["swa_indices"][:32]), IDX_MSG, id="idx_32_tokens"
    ),
    pytest.param(
        lambda a: dict(swa_indices=a["swa_indices"].flatten()), IDX_MSG, id="idx_1d"
    ),
    pytest.param(
        lambda a: dict(c128_lengths=a["c128_lengths"].long()), LEN_MSG, id="len_int64"
    ),
    pytest.param(
        lambda a: dict(swa_lengths=strided(a["swa_lengths"])), LEN_MSG, id="len_strided"
    ),
    pytest.param(
        lambda a: dict(swa_lengths=a["swa_lengths"][:32]), LEN_MSG, id="len_32_tokens"
    ),
    # q_norm_rope tables
    pytest.param(
        lambda a: {**qnr_args(), "q_positions": qnr_args()["q_positions"].float()},
        POS_MSG,
        id="pos_fp32",
    ),
    pytest.param(
        lambda a: {**qnr_args(), "q_positions": qnr_args()["q_positions"][:32]},
        POS_MSG,
        id="pos_32_tokens",
    ),
    pytest.param(
        lambda a: {**qnr_args(), "q_positions": strided(qnr_args()["q_positions"])},
        POS_MSG,
        id="pos_strided",
    ),
    pytest.param(
        lambda a: {**qnr_args(), "q_positions": qnr_args()["q_positions"].cpu()},
        POS_MSG,
        id="pos_cpu",
    ),
    pytest.param(
        lambda a: {
            **qnr_args(),
            "q_freqs": torch.view_as_real(qnr_args()["q_freqs"]).flatten(-2).double(),
        },
        FREQS_MSG,
        id="freqs_fp64",
    ),
    pytest.param(
        lambda a: {**qnr_args(), "q_freqs": qnr_args()["q_freqs"][:, :16].contiguous()},
        FREQS_MSG,
        id="freqs_16",
    ),
    pytest.param(
        lambda a: {**qnr_args(), "q_freqs": qnr_args()["q_freqs"].cpu()},
        FREQS_MSG,
        id="freqs_cpu",
    ),
]


@pytest.mark.parametrize("change", VALID_ARGS)
def test_valid_args(change):
    """The calls the invalid-input cases start from run."""
    a = valid_args()
    a.update(change(a))
    assert torch.isfinite(prefill_pair_mla(**a)).all()


@pytest.mark.quick
@pytest.mark.parametrize("change,match", INVALID_ARGS)
def test_invalid_args(change, match):
    a = valid_args()
    a.update(change(a))
    with pytest.raises(RuntimeError, match=match):
        prefill_pair_mla(**a)
