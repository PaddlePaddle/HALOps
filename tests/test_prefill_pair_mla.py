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

from typing import NamedTuple

import pytest
import torch

from halops import prefill_pair_mla
from tests.utils.prefill_pair_mla_utils import (
    DIM,
    HEADS,
    SCALE,
    Batch,
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


class Case(NamedTuple):
    """One batch of requests; exercises names the kernel path it adds."""

    name: str
    reqs: list[Req]
    exercises: str
    quick: bool = False


def params(cases: list[Case]):
    """One param per case; `quick` rides along as a marker."""
    return [
        pytest.param(c, marks=[pytest.mark.quick] if c.quick else [], id=c.name)
        for c in cases
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

# One request of T tokens.  kQst: a pair has at most 5 key
# tiles; predq: c128 width >= 512 with T >= 256, or width 256 with T >= 4096.
UNIFORM_CASES = [
    Case("T2-ctx256", [Req(2, 256)], "kQst, one pair, combine", quick=True),
    Case("T64-ctx4K", [Req(64, 4096)], "kQst, 32 pairs, combine", quick=True),
    Case("T2048-ctx2K", [Req(2048, 2048)], "kQst, no combine, chunk from 0"),
    Case("T1024-ctx32K", [Req(1024, 32768)], "streaming, fp8 c128"),
    Case("T1024-ctx64K", [Req(1024, 65536)], "predq, c128 width 512"),
    Case("T4096-ctx32K", [Req(4096, 32768)], "predq, c128 width 256"),
    Case("T64-ctx1M", [Req(64, 1048576)], "streaming, 32 pairs, combine"),
    Case("T256-ctx1M", [Req(256, 1048576)], "predq, long key streams"),
]

COMBINE = Case("combine", [Req(2, 65536)], "one pair split by the combine kernel")
QST = Case("qst", [Req(2048, 2048)], "kQst kernels")
STREAMING = Case("streaming", VARLEN_REQS, "streaming kernels with predq")

CP_RANKS = [
    pytest.param(cp, rank, id=f"cp{cp}-rank{rank}")
    for cp, rank in ((2, 1), (4, 0), (8, 7))
]


def run(batch: Batch, sink: bool = True, **kwargs) -> torch.Tensor:
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


def run_flashmla(batch: Batch) -> torch.Tensor:
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


def check(o: torch.Tensor, batch: Batch, sink: bool = True, rows=None):
    """Against the fp32 reference on rows (default every row)."""
    assert torch.isfinite(o).all(), "non-finite output"
    if rows is None:
        rows = list(range(batch.q.shape[0]))
    refs = references(batch, rows, sink)
    worst, rms = score(o, rows, refs)
    assert rms <= RMS_TOL and worst <= MAX_TOL, f"rms {rms:.3e} max {worst:.3e}"
    return rows, refs, worst, rms


def varlen_kwargs(lens: list[int], swa_shift: int = 1) -> dict:
    cu_seqlens, cu_pairs = cu_offsets(lens)
    return dict(cu_seqlens=cu_seqlens, cu_pairs=cu_pairs, swa_shift=swa_shift)


def case_kwargs(case: Case) -> dict:
    return varlen_kwargs([r.tokens for r in case.reqs])


def cp_reqs(cp: int) -> list[Req]:
    """VARLEN_REQS with the token count padded to a multiple of cp."""
    reqs = list(VARLEN_REQS)
    pad = -sum(r.tokens for r in reqs) % cp
    if pad:  # the padding rows form one more request
        reqs.append(Req(pad, pad))
    return reqs


@pytest.mark.parametrize("case", params(UNIFORM_CASES))
def test_uniform(case: Case):
    """One request, against the fp32 reference."""
    batch = make_batch(case.reqs)
    check(run(batch, **case_kwargs(case)), batch)


@pytest.mark.quick
def test_varlen():
    """Requests of different lengths through cu_seqlens / cu_pairs."""
    batch = make_batch(VARLEN_REQS)
    check(run(batch, **case_kwargs(STREAMING)), batch)


def test_replay():
    """SWA keys partly read from the per-request replay arena."""
    batch = make_batch(REPLAY_REQS)
    check(run(batch, **varlen_kwargs([r.tokens for r in REPLAY_REQS])), batch)


@pytest.mark.parametrize("cp,rank", CP_RANKS)
def test_cp(cp: int, rank: int):
    """CP round-robin: rank's tokens of a cp-way split."""
    local, lens = cp_local(make_batch(cp_reqs(cp)), cp, rank)
    check(run(local, **varlen_kwargs(lens, swa_shift=cp)), local)


def test_no_sink():
    """attn_sink = None, with an odd request length."""
    batch = make_batch([Req(517, 517)])
    check(run(batch, sink=False, **varlen_kwargs([517])), batch, sink=False)


@pytest.mark.parametrize("sink", [True, False], ids=["sink", "no-sink"])
@pytest.mark.parametrize(
    "req", [Req(2, 65536), Req(512, 262144)], ids=["combine", "in-kernel-merge"]
)
def test_no_keys(req: Req, sink: bool):
    """Tokens with no valid key output zeros, also when their pair is split:
    one pair is split by the combine kernel, 256 pairs are cut by the
    in-kernel merge's range boundaries."""
    batch = make_batch([req])
    swa_len, c_len = batch.swa_len.clone(), batch.c_len.clone()
    swa_len[0::2] = 0
    c_len[0::2] = 0
    batch = batch._replace(swa_len=swa_len, c_len=c_len)
    o = run(batch, sink, **varlen_kwargs([req.tokens])).view(-1, HEADS, DIM)
    assert torch.isfinite(o).all(), "non-finite output"
    assert not o[0::2].any(), "a token with no keys has non-zero output"
    check(o, batch, sink, rows=list(range(1, req.tokens, 2)))


def skip_keys(idx: torch.Tensor, lens: torch.Tensor, rows: int) -> torch.Tensor:
    """idx with some row ids inside the lengths replaced, the same way in
    every token so pairs keep their structure: v % 7 == 3 by -1 and
    v % 11 == 5 by v + rows (past the pool)."""
    inside = torch.arange(idx.shape[-1], device=idx.device) < lens[:, None, None]
    far = torch.where(inside & (idx % 11 == 5), idx + rows, idx)
    return torch.where(inside & (idx % 7 == 3), -1, far)


@pytest.mark.parametrize(
    "req", [Req(1024, 32768), Req(1024, 65536)], ids=["fp8", "predq"]
)
def test_invalid_keys(req: Req):
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
    check(run(batch, **varlen_kwargs([req.tokens])), batch)


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
    check(run(batch, **varlen_kwargs([req.tokens])), batch)


@pytest.mark.parametrize("case", params([COMBINE, QST, STREAMING]))
def test_no_pdl(case: Case):
    """Without PDL; the default call (with PDL) agrees."""
    batch = make_batch(case.reqs)
    kwargs = case_kwargs(case)
    o = run(batch, use_pdl=False, **kwargs)
    check(o, batch)
    assert torch.equal(run(batch, **kwargs), o)


def test_page_size():
    """Pool page sizes other than the default 128 (SWA) and 2 (c128)."""
    batch = make_batch(VARLEN_REQS, 64, 1)
    check(run(batch, **case_kwargs(STREAMING)), batch)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs")
@pytest.mark.parametrize("case", params([COMBINE, STREAMING]))
def test_other_device(case: Case):
    """cuda:1 inputs, the current device being cuda:0, run on cuda:1 with its
    own workspaces and give cuda:0's bits."""
    batch = make_batch(case.reqs)
    kwargs = case_kwargs(case)
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


@pytest.mark.parametrize("case", params([COMBINE, STREAMING]))
def test_cuda_graph(case: Case):
    """A call captured after an eager one of the same shape replays on new q
    like an eager call."""
    batch = make_batch(case.reqs)
    kwargs = case_kwargs(case)
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
    """T = 0 returns an empty output without a launch."""
    batch = make_batch([Req(2, 256)])
    batch = batch._replace(
        q=batch.q[:0],
        swa_idx=batch.swa_idx[:0],
        swa_len=batch.swa_len[:0],
        c_idx=batch.c_idx[:0],
        c_len=batch.c_len[:0],
    )
    assert run(batch, **varlen_kwargs([0])).numel() == 0


FLASHMLA_CASES = [
    Case("uniform", [Req(1024, 65536)], "predq"),
    Case("replay", REPLAY_REQS, "varlen with the replay arena"),
]


@pytest.mark.parametrize("case", params(FLASHMLA_CASES))
def test_vs_flashmla(case: Case):
    """Not less accurate than FlashMLA on the same inputs."""
    batch = make_batch(case.reqs)
    o = run(batch, **case_kwargs(case))
    rows, refs, worst, rms = check(o, batch)
    fm_worst, fm_rms = score(run_flashmla(batch), rows, refs)
    assert (
        rms <= fm_rms * 1.02 and worst <= fm_worst * 1.25
    ), f"rms {rms:.3e} vs flashmla {fm_rms:.3e}, max {worst:.3e} vs {fm_worst:.3e}"


# Fused q_norm_rope: q is a raw projection (per-head scales, a few outlier
# channels).
QNR_EPS = 1e-6
QNR_CASES = [
    Case("short", [Req(64, 256)], "kQst kernels", quick=True),
    Case("uniform-fp8", [Req(1024, 32768)], "streaming, fp8 c128"),
    Case("uniform", [Req(1024, 65536)], "streaming, predq"),
    Case("varlen", VARLEN_REQS, "varlen, predq"),
    Case("replay", REPLAY_REQS, "varlen with the replay arena"),
]


def qnr_inputs(reqs: list[Req]):
    """A batch with a raw q, its token positions and RoPE frequencies."""
    batch = make_batch(reqs)
    t = batch.q.shape[0]
    q = torch.randn(t, HEADS, DIM, device="cuda")
    q *= torch.rand(1, HEADS, 1, device="cuda") * 4 + 0.5
    q[..., torch.randint(0, DIM, (8,), device="cuda")] *= 20
    batch = batch._replace(q=q.bfloat16())
    return batch, positions(reqs), rope_freqs(max(r.ctx for r in reqs))


def run_qnr(batch: Batch, pos, freqs, **kwargs) -> torch.Tensor:
    return run(batch, q_positions=pos, q_freqs=freqs, q_norm_eps=QNR_EPS, **kwargs)


def score_qnr(o: torch.Tensor, batch: Batch, q_ref: torch.Tensor):
    assert torch.isfinite(o).all(), "non-finite output"
    rows = list(range(batch.q.shape[0]))
    return score(o, rows, references(batch, rows, q=q_ref))


@pytest.mark.parametrize("case", params(QNR_CASES))
def test_qnr(case: Case):
    """Against fp32 q_norm_rope and attention, and not less accurate than
    q_norm_rope rounded to bf16 first (the unfused path)."""
    batch, pos, freqs = qnr_inputs(case.reqs)
    kwargs = case_kwargs(case)
    q_ref = q_norm_rope(batch.q, pos, freqs, QNR_EPS)
    worst, rms = score_qnr(run_qnr(batch, pos, freqs, **kwargs), batch, q_ref)
    assert rms <= RMS_TOL and worst <= MAX_TOL, f"rms {rms:.3e} max {worst:.3e}"
    unfused = run(batch._replace(q=q_ref.bfloat16()), **kwargs)
    sep_worst, sep_rms = score_qnr(unfused, batch, q_ref)
    assert (
        rms <= sep_rms * 1.02 and worst <= sep_worst * 1.25
    ), f"rms {rms:.3e} vs unfused {sep_rms:.3e}, max {worst:.3e} vs {sep_worst:.3e}"


@pytest.mark.parametrize("layout", ["int32-pos-fp32-freqs", "q-4d", "q-strided"])
def test_qnr_layouts(layout: str):
    """int32 positions with float32 frequencies, a [T, 1, 64, 512] q, or a q
    with a token stride give the bits of the default layout."""
    batch, pos, freqs = qnr_inputs(VARLEN_REQS)
    kwargs = case_kwargs(STREAMING)
    o = run_qnr(batch, pos, freqs, **kwargs)
    if layout == "int32-pos-fp32-freqs":
        pos, freqs = pos.int(), torch.view_as_real(freqs).flatten(-2).contiguous()
    elif layout == "q-4d":
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


def test_qnr_cp():
    """CP round-robin: the last rank's tokens and positions of a 4-way split."""
    cp, rank = 4, 3
    batch, pos, freqs = qnr_inputs(cp_reqs(cp))
    local, lens = cp_local(batch, cp, rank)
    lpos = pos[rank::cp].contiguous()
    o = run_qnr(local, lpos, freqs, **varlen_kwargs(lens, swa_shift=cp))
    worst, rms = score_qnr(o, local, q_norm_rope(local.q, lpos, freqs, QNR_EPS))
    assert rms <= RMS_TOL and worst <= MAX_TOL, f"rms {rms:.3e} max {worst:.3e}"


# Invalid inputs: each case changes some arguments of a valid call (the
# 64 tokens of ARG_REQ as two 32-token requests) and expects the host check
# that names the violated constraint.
ARG_REQ = Req(64, 256)


@pytest.fixture
def prefill_args() -> dict:
    """Valid `prefill_pair_mla` arguments for ARG_REQ."""
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
        **cu_args(),
    )


def qnr_args() -> dict:
    return dict(q_positions=positions([ARG_REQ]), q_freqs=rope_freqs(ARG_REQ.ctx))


def cu_args(dtype: torch.dtype = torch.int32) -> dict:
    t = ARG_REQ.tokens
    return dict(
        cu_seqlens=torch.tensor([0, t // 2, t], dtype=dtype, device="cuda"),
        cu_pairs=torch.tensor([0, t // 4, t // 2], dtype=dtype, device="cuda"),
    )


def strided(x: torch.Tensor) -> torch.Tensor:
    """x's values with a last-dim stride of 2 (not contiguous)."""
    return torch.stack([x, x], -1)[..., 0]


def raw_q(token_stride: int, offset: int = 0) -> torch.Tensor:
    """[T, 64, 512] bf16 view of a [T, token_stride] buffer at offset."""
    buf = torch.zeros(ARG_REQ.tokens, token_stride, dtype=torch.bfloat16, device="cuda")
    return buf[:, offset : offset + HEADS * DIM].view(-1, HEADS, DIM)


def short_stride_cache(pages: int = 2, page_size: int = 128) -> torch.Tensor:
    """[pages, page_size, 1, 584] with a page stride of page_size * 576 B."""
    buf = torch.zeros(pages * page_size * 584, dtype=torch.uint8, device="cuda")
    return buf.as_strided((pages, page_size, 1, 584), (page_size * 576, 584, 584, 1))


QNR_Q_MSG = "view with contiguous heads"
Q_MSG = "q must be a contiguous bfloat16"
CU_MSG = "cu_seqlens / cu_pairs must be contiguous int32"
CACHE_MSG = "k caches must be uint8"
IDX_MSG = "indices must be contiguous int32"
QNR_TAB_MSG = "q_positions must be contiguous int64 / int32"

INVALID_ARGS = [
    pytest.param(
        lambda a: a.update(swa_shift=0), "swa_shift must be in", id="swa_shift-0"
    ),
    pytest.param(
        lambda a: a.update(q_positions=qnr_args()["q_positions"]),
        "q_positions and q_freqs go together",
        id="positions-only",
    ),
    pytest.param(lambda a: a.update(q=a["q"].float()), Q_MSG, id="q-fp32"),
    pytest.param(
        lambda a: a.update(q=a["q"][..., :32, :].contiguous()),
        "unsupported q dims",
        id="q-32-heads",
    ),
    pytest.param(
        lambda a: a.update(qnr_args(), q=strided(a["q"])),
        QNR_Q_MSG,
        id="qnr-q-head-stride",
    ),
    pytest.param(
        lambda a: a.update(qnr_args(), q=raw_q(HEADS * DIM + 8, offset=1)),
        QNR_Q_MSG,
        id="qnr-q-misaligned",
    ),
    pytest.param(
        lambda a: a.update(qnr_args(), q_positions=positions([ARG_REQ]).float()),
        QNR_TAB_MSG,
        id="qnr-positions-fp32",
    ),
    pytest.param(
        lambda a: a.update(
            qnr_args(), q_freqs=rope_freqs(ARG_REQ.ctx)[:, :16].contiguous()
        ),
        QNR_TAB_MSG,
        id="qnr-freqs-16",
    ),
    pytest.param(lambda a: a.update(cu_args(torch.int64)), CU_MSG, id="cu-int64"),
    pytest.param(
        lambda a: a.update({k: v[:1] for k, v in cu_args().items()}),
        CU_MSG,
        id="cu-no-request",
    ),
    pytest.param(
        lambda a: a.update(cu_pairs=a["cu_pairs"][:2]), CU_MSG, id="cu-pairs-short"
    ),
    pytest.param(
        lambda a: a.update(cu_seqlens=strided(a["cu_seqlens"])),
        CU_MSG,
        id="cu-strided",
    ),
    pytest.param(
        lambda a: a.update(attn_sink=a["attn_sink"][:32]),
        "attn_sink must be a contiguous float32",
        id="sink-32",
    ),
    # c128 is checked first, so a swa case also passes the c128 checks.
    pytest.param(
        lambda a: a.update(swa_k_cache=a["swa_k_cache"][..., :576]),
        CACHE_MSG,
        id="cache-row-576",
    ),
    pytest.param(
        lambda a: a.update(swa_k_cache=short_stride_cache()),
        CACHE_MSG,
        id="cache-page-stride-short",
    ),
    pytest.param(
        lambda a: a.update(c128_k_cache=make_cache(4, 3)),
        CACHE_MSG,
        id="cache-page-3",
    ),
    pytest.param(
        lambda a: a.update(swa_indices=a["swa_indices"].long()),
        IDX_MSG,
        id="indices-int64",
    ),
    pytest.param(
        lambda a: a.update(swa_lengths=a["swa_lengths"][:32]),
        IDX_MSG,
        id="lengths-32-tokens",
    ),
]


@pytest.mark.quick
@pytest.mark.parametrize("mutate,match", INVALID_ARGS)
def test_invalid_args(prefill_args: dict, mutate, match: str):
    """Inputs outside the host checks are rejected before the launch."""
    mutate(prefill_args)
    with pytest.raises(RuntimeError, match=match):
        prefill_pair_mla(**prefill_args)
