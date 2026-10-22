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

Inputs and a torch reference for the `mega_mla` fused MLA megakernel.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch

from halops.mla.mega_mla import build_kv_runs


class Dims(NamedTuple):
    """Model and pool geometry; mirrors Dsv4Dims in csrc/kernel/mega_mla/mla_traits.cuh."""

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
    dev = latent.device
    nope = (
        latent[:, : d.d_nope]
        .float()
        .reshape(n, d.d_nope // d.kv_quant_block, d.kv_quant_block)
    )
    amax = nope.abs().amax(dim=-1).clamp_min(1e-4)
    exp = torch.clamp(torch.round(torch.log2(amax / d.fp8_max)) + 127, 0, 255)
    q = _e4m3(nope * torch.pow(2.0, 127.0 - exp)[..., None], d)
    rows = torch.zeros((n, d.kv_row), dtype=torch.uint8, device=dev)
    rows[:, : d.d_nope] = q.reshape(n, d.d_nope).view(torch.uint8)
    rows[:, d.d_nope :] = (
        latent[:, d.d_nope :].contiguous().view(torch.uint8).reshape(n, -1)
    )
    scales = torch.zeros((n, d.kv_sc), dtype=torch.uint8, device=dev)
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


def _make_weight(n: int, k: int, gen: torch.Generator, dev: str) -> Weight:
    """Random bf16 weight, quantized by the package's own quantizer."""
    from halops.gemm.group_wise_quant import group_wise_quant

    w = torch.randn((n, k), generator=gen, device=dev, dtype=torch.bfloat16) * 0.02
    q, sf = group_wise_quant(w, mode="w8a8-mxfp8", use_swizzle=True)
    return Weight(q=q, sf=sf)


def _freqs_real(
    max_pos: int, dev: str, d: Dims, theta: float = 10000.0
) -> torch.Tensor:
    """fp32 [max_pos, 64] = 32 interleaved (cos, sin) pairs, the layout `freqs_real` is."""
    p = torch.arange(max_pos, device=dev, dtype=torch.float32)[:, None]
    inv = theta ** (
        -torch.arange(0, d.d_rope, 2, device=dev, dtype=torch.float32) / d.d_rope
    )
    ang = p * inv[None, :]
    out = torch.empty((max_pos, d.d_rope), device=dev, dtype=torch.float32)
    out[:, 0::2], out[:, 1::2] = torch.cos(ang), torch.sin(ang)
    return out


class Aux(NamedTuple):
    """What the reference needs and the argument dict cannot carry."""

    x: torch.Tensor
    swa_slots: torch.Tensor
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
    seed: int = 0,
    device: str = "cuda",
    dims: Dims = Dims(),
):
    """Build the whole `mega_mla` argument dict from scratch, plus the reference's `Aux`."""
    assert swa_len > 0, "a token must have at least one KV row"
    d = dims
    d.validate()
    dev = device
    gen = torch.Generator(device=dev).manual_seed(seed)
    T = bs * draft_tokens

    w = {
        "wq_a": _make_weight(d.qlora_rank, d.hidden, gen, dev),
        "wq_b": _make_weight(d.b_h * d.d_q, d.qlora_rank, gen, dev),
        "wkv": _make_weight(d.kvlora_rank, d.hidden, gen, dev),
        "wo_b": _make_weight(d.hidden, d.o_groups * d.o_lora_rank, gen, dev),
    }

    w["wo_a"] = _make_weight(
        d.o_groups * d.o_lora_rank, d.b_h * d.d_q // d.o_groups, gen, dev
    )

    base = torch.arange(bs, device=dev, dtype=torch.int64) * 128 * 8 + 128 * 64
    if fire_compressor:
        base = base - 1
    positions = (
        base[:, None] + torch.arange(draft_tokens, device=dev, dtype=torch.int64)
    ).reshape(-1)
    freqs = _freqs_real(int(positions.max()) + 1, dev, d)

    swa_slots = (
        torch.arange(T, device=dev, dtype=torch.int64)[:, None] * swa_len
        + torch.arange(swa_len, device=dev, dtype=torch.int64)[None, :]
    )
    n_pages = -(-int(swa_slots.max() + 1) // d.page_size)
    kv_cache = torch.zeros((n_pages, d.kv_page_stride), dtype=torch.uint8, device=dev)

    hist = swa_slots[:, :-1].reshape(-1)
    lat = torch.randn(
        (hist.numel(), d.d_q), generator=gen, device=dev, dtype=torch.bfloat16
    )
    _scatter_rows(
        kv_cache, hist, d.page_size, d.kv_page_stride, *_pack_kv_rows(lat, d), d
    )
    swa_loc = swa_slots[:, -1].to(torch.int32).contiguous()
    topk_lengths = torch.full((T,), swa_len, dtype=torch.int32, device=dev)
    kv_runs, kv_run_step = build_kv_runs(
        swa_slots.to(torch.int32), topk_lengths, d.page_size
    )

    c128 = {}
    c128_slots = torch.zeros((T, 0), dtype=torch.int64, device=dev)
    if c128_len:
        pb = max(d.b_topk, 1 << (c128_len - 1).bit_length())
        stride = -(-(pb * (d.kv_row + d.kv_sc)) // d.kv_row) * d.kv_row
        pool = torch.zeros((2 * T, stride), dtype=torch.uint8, device=dev)
        c128_slots = (
            torch.arange(T, device=dev, dtype=torch.int64)[:, None] * pb
            + torch.arange(c128_len, device=dev, dtype=torch.int64)[None, :]
        )
        flat = c128_slots.reshape(-1)
        lat = torch.randn(
            (flat.numel(), d.d_q), generator=gen, device=dev, dtype=torch.bfloat16
        )
        _scatter_rows(pool, flat, pb, stride, *_pack_kv_rows(lat, d), d)
        c128 = dict(
            c128_pool=pool,
            c128_stride_kv_block=stride,
            c128_page_block=pb,
            c128_attn_slot0=c128_slots[:, 0].to(torch.int32).contiguous(),
            c128_attn_len=torch.full((T,), c128_len, dtype=torch.int32, device=dev),
            wkv_gate_weight=torch.randn(
                (2 * d.kvlora_rank, d.hidden),
                generator=gen,
                device=dev,
                dtype=torch.bfloat16,
            )
            * 0.02,
            kv_score_ring=torch.randn(
                (bs * d.c128_ratio, 2 * d.kvlora_rank),
                generator=gen,
                device=dev,
                dtype=torch.float32,
            )
            * 0.1,
            kv_gate_ring_row=(
                (torch.arange(T, device=dev, dtype=torch.int64) // draft_tokens)
                * d.c128_ratio
                + (positions % d.c128_ratio)
            ).to(torch.int32),
            c128_ape=torch.randn(
                (d.c128_ratio, d.kvlora_rank),
                generator=gen,
                device=dev,
                dtype=torch.float32,
            )
            * 0.02,
            c128_norm_weight=torch.ones(
                d.kvlora_rank, dtype=torch.bfloat16, device=dev
            ),
            c128_freqs_real=freqs,
            c128_out_loc=((T + torch.arange(T, device=dev, dtype=torch.int64)) * pb).to(
                torch.int32
            ),
        )

    x = torch.randn((T, d.hidden), generator=gen, device=dev, dtype=torch.bfloat16)

    q_or_x = torch.zeros((T, d.b_h * d.d_q), dtype=torch.bfloat16, device=dev)
    q_or_x[:, : d.hidden] = x

    args = dict(
        q_or_x=q_or_x,
        kv_cache=kv_cache,
        page_size=d.page_size,
        kv_runs=kv_runs,
        topk=swa_len,
        kv_run_step=kv_run_step,
        topk_lengths=topk_lengths,
        positions=positions.to(torch.int32),
        freqs_real=freqs,
        wo_a_weight=w["wo_a"].q,
        wo_a_scale_inv=w["wo_a"].sf,
        wo_b_weight=w["wo_b"].q,
        wo_b_scale_inv=w["wo_b"].sf,
        out=torch.zeros((T, d.hidden), dtype=torch.bfloat16, device=dev),
        sm_scale=1.0 / math.sqrt(d.d_q),
        wq_a_weight=w["wq_a"].q,
        wq_a_scale_inv=w["wq_a"].sf,
        q_norm_weight=torch.ones(d.qlora_rank, dtype=torch.bfloat16, device=dev),
        wq_b_weight=w["wq_b"].q,
        wq_b_scale_inv=w["wq_b"].sf,
        q_eps=1e-6,
        use_phase1=True,
        wkv_weight=w["wkv"].q,
        wkv_scale_inv=w["wkv"].sf,
        kv_norm_weight=torch.ones(d.kvlora_rank, dtype=torch.bfloat16, device=dev),
        kv_eps=1e-6,
        use_kv_write=True,
        swa_loc=swa_loc,
        attn_sink=torch.randn(d.b_h, generator=gen, device=dev, dtype=torch.float32)
        * 0.1,
        draft_tokens=draft_tokens,
        **c128,
    )
    aux = Aux(
        x=x.clone(),
        swa_slots=swa_slots,
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
    from halops.gemm.dense_gemm import dense_gemm
    from halops.gemm.group_wise_quant import group_wise_quant

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


def reference_forward(args: dict, aux: Aux) -> torch.Tensor:
    """The whole fused layer in torch: bf16 [T, d.hidden], to compare against `args["out"]`."""
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

    den = p.sum(dim=-1, keepdim=True) + torch.exp(args["attn_sink"][None, :, None] - m)
    o = torch.einsum("thl,tld->thd", p / den, kv_seg)
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
