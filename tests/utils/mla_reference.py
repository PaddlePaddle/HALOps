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

import math
from typing import NamedTuple

import pytest
import torch

from halops.gemm.dense_gemm import dense_gemm
from halops.gemm.group_wise_quant import group_wise_quant
from halops.mla.mla_attention import mla_attention
from halops.mla.mla_attention_plan import mla_attention_plan
from halops.mla.mla_output_proj import mla_output_proj
from halops.mla.mla_workspace import mla_workspace

# Max abs error relative to max |reference|.  Both sides quantize to fp8 with
# group-32 / per-64 ue8m0 scales but at different rounding points.
# the projection alone, both sides fed the kernel's O
OUTPUT_PROJ_REL_TOL = 0.01
# the whole layer: the ~1% O difference flips the e4m3 rounding of O and o_lora
# elements near a rounding tie, which the projection amplifies to ~4%
LAYER_REL_TOL = 0.05
# the fixtures' SWA windows hold ascending slots
KV_RUN_STEP = 1


class Dims(NamedTuple):
    """Model and pool geometry; mirrors ModelDims in csrc/kernel/mla/mla_traits.cuh."""

    b_h: int = 64
    d_q: int = 512
    d_nope: int = 448
    d_rope: int = 64
    hidden: int = 4096
    qlora_rank: int = 1024
    kvlora_rank: int = 512
    o_groups: int = 8
    o_lora_rank: int = 1024
    c128_ratio: int = 128
    b_topk: int = 64
    kv_sc: int = 8
    kv_quant_block: int = 64
    fp8_max: float = 448.0
    page_size: int = 128

    @property
    def kv_row(self) -> int:
        """Bytes per KV row: TMA_K_STRIDE = D_NOPE + 2 * D_ROPE."""
        return self.d_nope + 2 * self.d_rope

    @property
    def kv_page_stride(self) -> int:
        """Bytes per SWA page: page_size rows, then page_size scale slots, row-rounded."""
        return (
            -(-(self.page_size * (self.kv_row + self.kv_sc)) // self.kv_row)
            * self.kv_row
        )

    def validate(self) -> None:
        assert self.d_q == self.d_nope + self.d_rope, "D_Q must be D_NOPE + D_ROPE"
        assert (
            self.d_nope % self.kv_quant_block == 0
        ), "NoPE must split into scale groups"
        assert (
            self.d_nope // self.kv_quant_block < self.kv_sc
        ), "one scale byte per group"
        assert (
            self.b_h * self.d_q % self.o_groups == 0
        ), "wo_a's groups must tile flat_o"
        assert (
            self.kv_page_stride // self.kv_row >= self.page_size
        ), "rows_per_page < page"


def _e4m3(x: torch.Tensor, d: Dims) -> torch.Tensor:
    """fp32 -> e4m3, CLAMPED first, which is what the kernel's `satfinite` cvt does."""
    return x.clamp(-d.fp8_max, d.fp8_max).to(torch.float8_e4m3fn)


def _pack_kv_rows(latent: torch.Tensor, d: Dims) -> tuple[torch.Tensor, torch.Tensor]:
    """bf16 [n, 512] latent (already normed AND roped) -> (rows [n, 576], scales [n, 8])."""
    n = latent.shape[0]
    nope = (
        latent[:, : d.d_nope]
        .float()
        .reshape(n, d.d_nope // d.kv_quant_block, d.kv_quant_block)
    )
    amax = nope.abs().amax(dim=-1).clamp_min(1e-4)
    exp = torch.clamp(torch.ceil(torch.log2(amax / d.fp8_max)) + 127, 0, 255)
    q = _e4m3(nope * torch.pow(2.0, 127.0 - exp)[..., None], d)
    rows = torch.zeros((n, d.kv_row), dtype=torch.uint8)
    rows[:, : d.d_nope] = q.reshape(n, d.d_nope).view(torch.uint8)
    rows[:, d.d_nope :] = (
        latent[:, d.d_nope :].contiguous().view(torch.uint8).reshape(n, -1)
    )
    scales = torch.zeros((n, d.kv_sc), dtype=torch.uint8)
    scales[:, : d.d_nope // d.kv_quant_block] = exp.to(torch.uint8)
    return rows, scales


def _unpack_kv_rows(rows: torch.Tensor, scales: torch.Tensor, d: Dims) -> torch.Tensor:
    """(rows [n, 576], scales [n, 8]) -> fp32 [n, 512], what the kernel's attention sees."""
    n = rows.shape[0]
    nope = (
        rows[:, : d.d_nope]
        .contiguous()
        .view(torch.float8_e4m3fn)
        .float()
        .reshape(n, d.d_nope // d.kv_quant_block, d.kv_quant_block)
    )
    exp = scales[:, : d.d_nope // d.kv_quant_block].float()
    nope = (nope * torch.pow(2.0, exp - 127.0)[..., None]).reshape(n, d.d_nope)
    rope = (
        rows[:, d.d_nope :]
        .contiguous()
        .view(torch.bfloat16)
        .float()
        .reshape(n, d.d_rope)
    )
    return torch.cat([nope, rope], dim=1)


def _rope(
    v: torch.Tensor, freqs: torch.Tensor, d: Dims, *, inverse: bool
) -> torch.Tensor:
    """Rotate the last 64 dims of `v` in place-ish, adjacent-pair interleaved."""
    out = v.clone()
    tail = v[..., d.d_nope :].reshape(*v.shape[:-1], d.d_rope // 2, 2)
    c = freqs[..., 0::2].reshape(*freqs.shape[:-1], d.d_rope // 2)
    s = freqs[..., 1::2].reshape(*freqs.shape[:-1], d.d_rope // 2)
    while c.dim() < tail.dim() - 1:
        c, s = c.unsqueeze(-2), s.unsqueeze(-2)
    f0, f1 = tail[..., 0], tail[..., 1]
    if inverse:
        r0, r1 = f0 * c + f1 * s, -f0 * s + f1 * c
    else:
        r0, r1 = f0 * c - f1 * s, f0 * s + f1 * c
    out[..., d.d_nope :] = torch.stack([r0, r1], dim=-1).reshape(
        *v.shape[:-1], d.d_rope
    )
    return out


def _rms(v: torch.Tensor, eps: float, gain: torch.Tensor | None) -> torch.Tensor:
    """RMSNorm over the last dim in fp32, optional bf16 gain applied after."""
    x = v.float()
    x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return x if gain is None else x * gain.float()


class Weight(NamedTuple):
    """One projection: the fp8 bytes and the ONE scale both sides read."""

    q: torch.Tensor
    sf: torch.Tensor


def _make_weight(n: int, k: int) -> Weight:
    """Random bf16 weight, quantized by the package's own quantizer."""
    w = torch.randn((n, k), dtype=torch.bfloat16) * 0.02
    q, sf = group_wise_quant(w, mode="w8a8-mxfp8", use_swizzle=True)
    return Weight(q=q, sf=sf)


def _freqs_real(max_pos: int, d: Dims, theta: float = 10000.0) -> torch.Tensor:
    """fp32 [max_pos, 64] = 32 interleaved (cos, sin) pairs, the layout `freqs_real` is."""
    p = torch.arange(max_pos, dtype=torch.float32)[:, None]
    inv = theta ** (-torch.arange(0, d.d_rope, 2, dtype=torch.float32) / d.d_rope)
    ang = p * inv[None, :]
    out = torch.empty((max_pos, d.d_rope), dtype=torch.float32)
    out[:, 0::2], out[:, 1::2] = torch.cos(ang), torch.sin(ang)
    return out


class Aux(NamedTuple):
    """What the reference needs and the argument dict cannot carry."""

    x: torch.Tensor
    swa_slots: torch.Tensor
    topk_lengths: torch.Tensor
    c128_slots: torch.Tensor
    weights: dict
    swa_len: int
    c128_len: int
    dims: Dims


def build_inputs(
    *,
    bs: int,
    draft_tokens: int,
    swa_len: int = 128,
    c128_len: int = 0,
    fire_compressor: bool = False,
    attn_sink: bool = True,
    dims: Dims = Dims(),
):
    """Build the whole `mla_attention` argument dict (all but the workspace) on
    the default device and global RNG, plus the reference's `Aux`.  attn_sink:
    random per-head sink, else None."""
    assert swa_len > 0, "a token must have at least one KV row"
    d = dims
    d.validate()
    T = bs * draft_tokens

    w = {
        "wq_a": _make_weight(d.qlora_rank, d.hidden),
        "wq_b": _make_weight(d.b_h * d.d_q, d.qlora_rank),
        "wkv": _make_weight(d.kvlora_rank, d.hidden),
        "wo_b": _make_weight(d.hidden, d.o_groups * d.o_lora_rank),
        "wo_a": _make_weight(d.o_groups * d.o_lora_rank, d.b_h * d.d_q // d.o_groups),
    }

    base = torch.arange(bs, dtype=torch.int64) * 128 * 8 + 128 * 64
    if fire_compressor:
        base = base - 1
    positions = (base[:, None] + torch.arange(draft_tokens, dtype=torch.int64)).reshape(
        -1
    )
    freqs = _freqs_real(int(positions.max()) + 1, d)

    swa_slots = (
        torch.arange(T, dtype=torch.int64)[:, None] * swa_len
        + torch.arange(swa_len, dtype=torch.int64)[None, :]
    )
    n_pages = -(-int(swa_slots.max() + 1) // d.page_size)
    kv_cache = torch.zeros((n_pages, d.kv_page_stride), dtype=torch.uint8)

    hist = swa_slots[:, :-1].reshape(-1)
    lat = torch.randn((hist.numel(), d.d_q), dtype=torch.bfloat16)
    _scatter_rows(
        kv_cache, hist, d.page_size, d.kv_page_stride, *_pack_kv_rows(lat, d), d
    )
    swa_loc = swa_slots[:, -1].to(torch.int32).contiguous()
    topk_lengths = torch.full((T,), swa_len, dtype=torch.int32)

    c128 = {}
    c128_slots = torch.zeros((T, 0), dtype=torch.int64)
    if c128_len:
        # 2 compressed rows per pool page (page_size / c128_ratio), addressed
        # through a page table.  A token's pages are consecutive here, so its
        # slots come out ascending; the page-table path is still the one used.
        pb = 2
        stride = -(-(pb * (d.kv_row + d.kv_sc)) // d.kv_row) * d.kv_row
        ppt = -(-c128_len // pb)  # pages one token's window spans
        pool = torch.zeros((T * ppt + T, stride), dtype=torch.uint8)
        page_table = (
            torch.arange(T, dtype=torch.int32)[:, None] * ppt
            + torch.arange(ppt, dtype=torch.int32)[None, :]
        ).contiguous()
        c128_slots = (
            torch.arange(T, dtype=torch.int64)[:, None] * (ppt * pb)
            + torch.arange(c128_len, dtype=torch.int64)[None, :]
        )
        flat = c128_slots.reshape(-1)
        lat = torch.randn((flat.numel(), d.d_q), dtype=torch.bfloat16)
        _scatter_rows(pool, flat, pb, stride, *_pack_kv_rows(lat, d), d)
        c128 = dict(
            c128_pool=pool,
            c128_stride_kv_block=stride,
            c128_page_block=pb,
            c128_page_table=page_table,
            c128_attn_len=torch.full((T,), c128_len, dtype=torch.int32),
            wkv_gate_weight=torch.randn(
                (2 * d.kvlora_rank, d.hidden),
                dtype=torch.bfloat16,
            )
            * 0.02,
            kv_score_ring=torch.randn(
                (bs * d.c128_ratio, 2 * d.kvlora_rank),
                dtype=torch.float32,
            )
            * 0.1,
            kv_gate_ring_row=(
                (torch.arange(T, dtype=torch.int64) // draft_tokens) * d.c128_ratio
                + (positions % d.c128_ratio)
            ).to(torch.int32),
            c128_ape=torch.randn(
                (d.c128_ratio, d.kvlora_rank),
                dtype=torch.float32,
            )
            * 0.02,
            c128_norm_weight=torch.ones(d.kvlora_rank, dtype=torch.bfloat16),
            c128_freqs_real=freqs,
            c128_out_loc=((T * ppt + torch.arange(T, dtype=torch.int64)) * pb).to(
                torch.int32
            ),
        )

    x = torch.randn((T, d.hidden), dtype=torch.bfloat16)

    q_or_x = torch.zeros((T, d.b_h * d.d_q), dtype=torch.bfloat16)

    args = dict(
        q_or_x=q_or_x,
        hidden_states=x.clone(),
        kv_cache=kv_cache,
        page_size=d.page_size,
        positions=positions.to(torch.int32),
        freqs_real=freqs,
        sm_scale=1.0 / math.sqrt(d.d_q),
        wq_a_weight=w["wq_a"].q,
        wq_a_sf=w["wq_a"].sf,
        q_norm_weight=torch.ones(d.qlora_rank, dtype=torch.bfloat16),
        wq_b_weight=w["wq_b"].q,
        wq_b_sf=w["wq_b"].sf,
        q_eps=1e-6,
        wkv_weight=w["wkv"].q,
        wkv_sf=w["wkv"].sf,
        kv_norm_weight=torch.ones(d.kvlora_rank, dtype=torch.bfloat16),
        kv_eps=1e-6,
        swa_loc=swa_loc,
        attn_sink=(
            torch.randn(d.b_h, dtype=torch.float32) * 0.1 if attn_sink else None
        ),
        draft_tokens=draft_tokens,
        **c128,
    )
    aux = Aux(
        x=x.clone(),
        swa_slots=swa_slots,
        topk_lengths=topk_lengths,
        c128_slots=c128_slots,
        weights=w,
        swa_len=swa_len,
        c128_len=c128_len,
        dims=d,
    )
    return args, aux


def _row_bases(slots: torch.Tensor, page_block: int, stride: int, d: Dims):
    """Slot -> (row byte offset, scale byte offset) inside a flat pool."""
    page, off = slots // page_block, slots % page_block
    return (
        page * stride + off * d.kv_row,
        page * stride + page_block * d.kv_row + off * d.kv_sc,
    )


def _scatter_rows(pool, slots, page_block, stride, rows, scales, d):
    r0, s0 = _row_bases(slots, page_block, stride, d)
    br = torch.arange(d.kv_row, device=pool.device)
    bs = torch.arange(d.kv_sc, device=pool.device)
    pool.view(-1)[(r0[:, None] + br[None, :]).reshape(-1)] = rows.reshape(-1)
    pool.view(-1)[(s0[:, None] + bs[None, :]).reshape(-1)] = scales.reshape(-1)


def _gather_latents(pool, slots, page_block, stride, d) -> torch.Tensor:
    """The [*, 512] fp32 latents the kernel's attention sees for `slots`."""
    flat = slots.reshape(-1)
    r0, s0 = _row_bases(flat, page_block, stride, d)
    br = torch.arange(d.kv_row, device=pool.device)
    bs = torch.arange(d.kv_sc, device=pool.device)
    rows = pool.view(-1)[(r0[:, None] + br[None, :]).reshape(-1)].reshape(-1, d.kv_row)
    scales = pool.view(-1)[(s0[:, None] + bs[None, :]).reshape(-1)].reshape(-1, d.kv_sc)
    return _unpack_kv_rows(rows, scales, d).reshape(*slots.shape, d.d_q)


def _gemm(a: torch.Tensor, w: Weight) -> torch.Tensor:
    """bf16 [T, K] @ fp8 [N, K]^T -> bf16 [T, N], through this package's own GEMM."""
    t, k = a.shape
    tp = -(-t // 128) * 128
    a_pad = a
    if tp != t:
        a_pad = torch.zeros((tp, k), dtype=torch.bfloat16, device=a.device)
        a_pad[:t] = a
    aq, asf = group_wise_quant(
        a_pad.to(torch.bfloat16), mode="w8a8-mxfp8", use_swizzle=True
    )
    out = torch.zeros((tp, w.q.shape[0]), dtype=torch.bfloat16, device=a.device)
    dense_gemm(aq, w.q, asf, w.sf, out)
    return out[:t]


def _run_breaks(idx: torch.Tensor, lengths: torch.Tensor, page_size: int, step: int):
    """[T, topk - 1] bool: window column c + 1 starts a new run (its slot is not
    the previous one + step, or lies on another page)."""
    shift = page_size.bit_length() - 1
    prev, cur = idx[:, :-1], idx[:, 1:]
    brk = (cur != prev + step) | ((cur >> shift) != (prev >> shift))
    col = torch.arange(1, idx.shape[1], device=idx.device)
    return brk & (col[None, :] < lengths[:, None])


def _ascending_runs(first0, len0, first1, rows, step: int) -> torch.Tensor:
    """Runs scanned with `step` -> ascending (slot0, len0, slot1, rows)."""
    if step > 0:
        return torch.stack([first0, len0, first1, rows], dim=1)
    len1 = rows - len0
    two = len1 > 0
    slot0 = torch.where(two, first1 - (len1 - 1), first0 - (rows - 1))
    slot1 = torch.where(two, first0 - (len0 - 1), slot0)
    return torch.stack([slot0, torch.where(two, len1, rows), slot1, rows], dim=1)


def reference_kv_runs(
    kv_indices: torch.Tensor,
    topk_lengths: torch.Tensor | None,
    page_size: int,
    step: int,
) -> torch.Tensor:
    """`mla_attention_plan`'s per-token KV runs in torch, int32 [T, 4]: a
    window of more than two runs is cut to its first row, then its first run
    ends at the first break."""
    idx = kv_indices.to(torch.int64)
    T, topk = idx.shape
    lengths = torch.full((T,), topk, device=idx.device)
    if topk_lengths is not None:
        lengths = topk_lengths.to(torch.int64)
    cut = _run_breaks(idx, lengths, page_size, step).sum(1) > 1
    lengths = torch.where(cut, torch.ones_like(lengths), lengths)
    brk = _run_breaks(idx, lengths, page_size, step)
    len0 = torch.where(brk.any(1), brk.int().argmax(1) + 1, lengths)
    first1 = idx.gather(1, len0.clamp(max=topk - 1)[:, None])[:, 0]
    return _ascending_runs(idx[:, 0], len0, first1, lengths, step).to(torch.int32)


def reference_attention(args: dict, aux: Aux) -> torch.Tensor:
    """`mla_attention` in torch: attention O before the inverse RoPE, fp32
    [T, 64, 512]."""
    d = aux.dims
    w, x = aux.weights, aux.x
    T = x.shape[0]
    pos = args["positions"].to(torch.int64)
    fr = args["freqs_real"][pos]

    q_lora = _rms(_gemm(x, w["wq_a"]), args["q_eps"], args["q_norm_weight"])
    flat_q = _gemm(q_lora.to(torch.bfloat16), w["wq_b"])

    q = _rms(flat_q.reshape(T, d.b_h, d.d_q), args["q_eps"], None)
    q = _rope(q, fr[:, None, :], d, inverse=False)

    kv = _rms(_gemm(x, w["wkv"]), args["kv_eps"], args["kv_norm_weight"])
    kv = _rope(kv.to(torch.bfloat16).float(), fr, d, inverse=False).to(torch.bfloat16)
    own = _unpack_kv_rows(*_pack_kv_rows(kv, d), d)

    kv_seg = _gather_latents(
        args["kv_cache"], aux.swa_slots, d.page_size, args["kv_cache"].stride(0), d
    )
    kv_seg[:, -1] = own
    if aux.c128_len:
        kv_seg = torch.cat(
            [
                kv_seg,
                _gather_latents(
                    args["c128_pool"],
                    aux.c128_slots,
                    args["c128_page_block"],
                    args["c128_stride_kv_block"],
                    d,
                ),
            ],
            dim=1,
        )

    s = torch.einsum("thd,tld->thl", q, kv_seg) * args["sm_scale"]
    m = s.amax(dim=-1, keepdim=True)
    p = torch.exp(s - m)

    den = p.sum(dim=-1, keepdim=True)
    if args["attn_sink"] is not None:
        den = den + torch.exp(args["attn_sink"][None, :, None] - m)
    return torch.einsum("thl,tld->thd", p / den, kv_seg)


def reference_output_proj(o: torch.Tensor, args: dict, aux: Aux) -> torch.Tensor:
    """`mla_output_proj` in torch: fp32 attention O [T, 64, 512] -> inverse
    RoPE -> grouped wo_a -> wo_b, bf16 [T, hidden]."""
    d, w = aux.dims, aux.weights
    T = o.shape[0]
    fr = args["freqs_real"][args["positions"].to(torch.int64)]
    o = _rope(o, fr[:, None, :], d, inverse=True).to(torch.bfloat16)

    R = d.o_lora_rank
    o_flat = o.reshape(T, d.b_h * d.d_q)
    per_group = d.b_h * d.d_q // d.o_groups
    o_lora = torch.cat(
        [
            _gemm(
                o_flat[:, g * per_group : (g + 1) * per_group].contiguous(),
                Weight(
                    q=w["wo_a"].q[g * R : (g + 1) * R].contiguous(),
                    sf=w["wo_a"].sf[g * R : (g + 1) * R].contiguous(),
                ),
            )
            for g in range(d.o_groups)
        ],
        dim=1,
    )
    return _gemm(o_lora.to(torch.bfloat16), w["wo_b"])


def assert_rel_close(out: torch.Tensor, ref: torch.Tensor, rel_tol: float) -> None:
    """Max abs error within rel_tol of max |ref|, all values finite."""
    out, ref = out.float(), ref.float()
    assert torch.isfinite(out).all(), "non-finite output"
    atol = rel_tol * max(float(ref.abs().max()), 1e-6)
    torch.testing.assert_close(out, ref, rtol=0, atol=atol)


class Mode(NamedTuple):
    """One attention scenario: c128_len compressed rows per token (0: SWA
    layer), fire: the compressor writes a row this launch, swa_len SWA rows."""

    name: str
    c128_len: int
    fire: bool
    swa_len: int = 128


class Case(NamedTuple):
    """One (bs, draft_tokens) shape, T = bs * draft_tokens."""

    bs: int
    draft_tokens: int
    exercises: str
    quick: bool


# T ladder of the attention prelude: select_tile_t in mla_workspace.cuh picks the
# HCA token tile from T (SWA always uses 128); exercises names what each adds
CASES = [
    Case(1, 5, "token tile 16, distributed q norm", quick=True),
    Case(8, 5, "token tile 32", quick=False),
    Case(12, 5, "token tile 128 from T = 45 on", quick=False),
    Case(13, 5, "token tile 128, one group", quick=True),
    Case(24, 5, "token tile 128, one group nearly full", quick=False),
    Case(26, 5, "HCA token tile 64 (3 groups), SWA 2 groups", quick=True),
    Case(40, 5, "2 token groups of 128", quick=True),
]


def params(modes: list[Mode]):
    """The (mode, case) product for one suite; `quick` rides along as a marker."""
    return [
        pytest.param(
            mode,
            case,
            marks=[pytest.mark.quick] if case.quick else [],
            id=f"{mode.name}-T{case.bs * case.draft_tokens}",
        )
        for mode in modes
        for case in CASES
    ]


def layer_workspace(args: dict, aux: Aux, max_tokens: int) -> torch.Tensor:
    """`mla_workspace` for the weight shapes of one layer."""
    w = aux.weights
    return mla_workspace(
        max_tokens, args["wq_a_weight"], args["wkv_weight"], w["wo_a"].q, w["wo_b"].q
    )


def plan_layer(args: dict, aux: Aux, workspace: torch.Tensor) -> torch.Tensor:
    """`mla_attention_plan` of one layer's kind into `workspace`."""
    return mla_attention_plan(
        aux.swa_slots.to(torch.int32),
        aux.topk_lengths,
        args["page_size"],
        KV_RUN_STEP,
        args.get("c128_attn_len"),
        workspace,
    )


def run_layer(args: dict, aux: Aux, workspace: torch.Tensor | None = None):
    """A closure running `mla_attention` + `mla_output_proj` for one layer; it
    restores q_or_x first and returns the layer output.  Without `workspace`
    the layer gets its own, planned here; with one, the caller runs the layer
    kind's `mla_attention_plan` first."""
    w = aux.weights
    T = args["q_or_x"].shape[0]
    q0 = args["q_or_x"].clone()
    ws = workspace
    if ws is None:
        ws = layer_workspace(args, aux, T)
        plan_layer(args, aux, ws)

    def run() -> torch.Tensor:
        out = torch.empty((T, w["wo_b"].q.shape[0]), dtype=torch.bfloat16)
        args["q_or_x"].copy_(q0)
        mla_attention(**args, workspace=ws, reset_barriers=False)
        mla_output_proj(
            args["q_or_x"],
            args["positions"],
            args["freqs_real"],
            w["wo_a"].q,
            w["wo_a"].sf,
            w["wo_b"].q,
            w["wo_b"].sf,
            out,
            ws,
            attn_sink=args["attn_sink"],
            c128_attn_len=args.get("c128_attn_len"),
        )
        return out

    return run


def check_attention(mode: Mode, case: Case, *, attn_sink: bool = True) -> None:
    """Run the layer on one fixture and compare its output with the torch
    layer."""
    args, aux = build_inputs(
        bs=case.bs,
        draft_tokens=case.draft_tokens,
        swa_len=mode.swa_len,
        c128_len=mode.c128_len,
        fire_compressor=mode.fire,
        attn_sink=attn_sink,
    )
    out = run_layer(args, aux)()
    ref = reference_output_proj(reference_attention(args, aux), args, aux)
    assert_rel_close(out, ref, LAYER_REL_TOL)
