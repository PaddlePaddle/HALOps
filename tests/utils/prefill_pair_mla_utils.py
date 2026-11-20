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

from typing import NamedTuple, Optional

import torch
import torch.nn.functional as F

HEADS, DIM, NOPE, ROPE = 64, 512, 448, 64
ROW_BYTES, SF_BYTES = NOPE + 2 * ROPE, 8  # 576 B data + 8 B UE8M0 scales
SWA_WINDOW, SWA_PAGE, C128_PAGE = 128, 128, 2
SCALE = DIM**-0.5


def pad64(n):
    return (n + 63) // 64 * 64


def make_cache(pages, page_size):
    """Random V4 fp8 KV cache, uint8 [pages, page_size, 1, 584].

    A page holds its page_size data rows (448 e4m3 nope + 64 bf16 rope), then
    their scales; the page stride is padded to a multiple of 576 bytes.
    """
    stride = (
        (page_size * (ROW_BYTES + SF_BYTES) + ROW_BYTES - 1) // ROW_BYTES * ROW_BYTES
    )
    buf = torch.zeros(pages, stride, dtype=torch.uint8, device="cuda")
    data = buf[:, : page_size * ROW_BYTES].view(pages, page_size, ROW_BYTES)
    nope = (torch.randn(pages, page_size, NOPE, device="cuda") * 64).clamp(-448, 448)
    data[..., :NOPE] = nope.to(torch.float8_e4m3fn).view(torch.uint8)
    rope = torch.randn(pages, page_size, ROPE, device="cuda").bfloat16()
    data[..., NOPE:] = rope.view(torch.uint8).view(pages, page_size, 2 * ROPE)
    sf = buf[:, page_size * ROW_BYTES : page_size * (ROW_BYTES + SF_BYTES)]
    sf = sf.view(pages, page_size, SF_BYTES)
    sf[..., : NOPE // 64] = torch.randint(
        118, 124, (pages, page_size, NOPE // 64), dtype=torch.uint8, device="cuda"
    )
    return buf[:, : page_size * (ROW_BYTES + SF_BYTES)].view(
        pages, page_size, 1, ROW_BYTES + SF_BYTES
    )


def dequant(cache, idx):
    """bf16 [n, 512] rows idx of a make_cache cache, dequantized exactly."""
    pages, page_size = cache.shape[:2]
    flat = torch.as_strided(cache, (pages, cache.stride(0)), (cache.stride(0), 1))
    page, row = (idx // page_size)[:, None], (idx % page_size)[:, None]
    data = flat[page, row * ROW_BYTES + torch.arange(ROW_BYTES, device=idx.device)]
    sf = flat[
        page,
        page_size * ROW_BYTES
        + row * SF_BYTES
        + torch.arange(NOPE // 64, device=idx.device),
    ]
    scale = torch.exp2(sf.float() - 127).repeat_interleave(64, -1)
    nope = (data[:, :NOPE].view(torch.float8_e4m3fn).float() * scale).bfloat16()
    return torch.cat([nope, data[:, NOPE:].contiguous().view(torch.bfloat16)], -1)


class Req(NamedTuple):
    """A request whose chunk is the last ``tokens`` of ``ctx`` positions.

    With ``replay`` set, keys at positions >= ctx - tokens - replay are read
    from a per-request arena instead of the shared SWA pool, and with
    ``clamp`` the SWA window also stops at that position.
    """

    tokens: int
    ctx: int
    replay: Optional[int] = None
    clamp: bool = True


class Batch(NamedTuple):
    q: torch.Tensor  # [T, 1, 64, 512] bf16
    swa: torch.Tensor
    swa_idx: torch.Tensor  # [T, 1, W] int32
    swa_len: torch.Tensor  # [T] int32
    c128: torch.Tensor
    c_idx: torch.Tensor  # [T, 1, Wc] int32
    c_len: torch.Tensor  # [T] int32
    sink: torch.Tensor  # [64] fp32
    starts: list  # request token offsets, [R + 1]


def make_batch(reqs, swa_page=SWA_PAGE, c128_page=C128_PAGE):
    """Inputs of a prefill batch, laid out like the DSv4 backend builds them.

    Each request owns a contiguous slice of both pools.  SWA: token p sees
    min(p + 1, 128) keys, newest first.  c128: token p sees
    max((p + 1) // 128, 1) compressed rows, oldest first, the index width
    padded to a multiple of 64.
    """
    shared_pages = sum((r.ctx + swa_page - 1) // swa_page for r in reqs)
    arena_pages = sum(
        (r.tokens + r.replay + swa_page - 1) // swa_page
        for r in reqs
        if r.replay is not None
    )
    swa = make_cache(shared_pages + arena_pages, swa_page)
    wc = pad64(max(r.ctx for r in reqs) // 128)
    c128 = make_cache(
        sum((r.ctx // 128 + c128_page - 1) // c128_page + 1 for r in reqs), c128_page
    )
    cols = torch.arange(SWA_WINDOW, device="cuda")
    ccols = torch.arange(wc, device="cuda")
    swa_idx, swa_len, c_idx, c_len = [], [], [], []
    swa_base = c_base = 0
    arena = shared_pages * swa_page
    for r in reqs:
        pos = torch.arange(r.ctx - r.tokens, r.ctx, device="cuda")
        kp = pos[:, None] - cols  # key positions, newest first
        if r.replay is None:
            sl = torch.clamp(pos + 1, max=SWA_WINDOW)
            si = swa_base + kp
        else:
            rs = r.ctx - r.tokens - r.replay  # replay start
            sl = torch.clamp(pos + 1 - (rs if r.clamp else 0), max=SWA_WINDOW)
            si = torch.where(kp >= rs, arena + kp - rs, swa_base + kp)
            arena += (r.tokens + r.replay + swa_page - 1) // swa_page * swa_page
        swa_idx.append(torch.where(cols[None] < sl[:, None], si, -1))
        swa_len.append(sl)
        cl = torch.clamp((pos + 1) // 128, min=1)
        ci = (c_base + ccols).expand(r.tokens, wc)
        c_idx.append(torch.where(ccols[None] < cl[:, None], ci, -1))
        c_len.append(cl)
        swa_base += (r.ctx + swa_page - 1) // swa_page * swa_page
        c_base += ((r.ctx // 128 + c128_page - 1) // c128_page + 1) * c128_page
    t = sum(r.tokens for r in reqs)
    starts = [0]
    for r in reqs:
        starts.append(starts[-1] + r.tokens)
    return Batch(
        q=torch.randn(t, 1, HEADS, DIM, device="cuda").bfloat16(),
        swa=swa,
        swa_idx=torch.cat(swa_idx).int().unsqueeze(1).contiguous(),
        swa_len=torch.cat(swa_len).int(),
        c128=c128,
        c_idx=torch.cat(c_idx).int().unsqueeze(1).contiguous(),
        c_len=torch.cat(c_len).int(),
        sink=torch.randn(HEADS, device="cuda"),
        starts=starts,
    )


def cu_offsets(lens):
    """(cu_seqlens, cu_pairs) int32 [R + 1] of request token counts."""
    lens = torch.tensor(lens, dtype=torch.int32, device="cuda")
    cu_seqlens = F.pad(lens.cumsum(0), (1, 0)).int()
    cu_pairs = F.pad(((lens + 1) // 2).cumsum(0), (1, 0)).int()
    return cu_seqlens, cu_pairs


def cp_local(batch, cp, rank):
    """Rank's rows of a CP round-robin split, and its local request lengths."""

    def count(x):  # global tokens g < x with g % cp == rank
        return max(0, (x - rank + cp - 1) // cp)

    sel = slice(rank, None, cp)
    local = batch._replace(
        q=batch.q[sel].contiguous(),
        swa_idx=batch.swa_idx[sel].contiguous(),
        swa_len=batch.swa_len[sel].contiguous(),
        c_idx=batch.c_idx[sel].contiguous(),
        c_len=batch.c_len[sel].contiguous(),
        starts=[count(s) for s in batch.starts],
    )
    lens = [e - s for s, e in zip(local.starts, local.starts[1:])]
    return local, lens


def positions(reqs):
    """int64 [T] positions of make_batch(reqs)'s tokens."""
    return torch.cat(
        [torch.arange(r.ctx - r.tokens, r.ctx, device="cuda") for r in reqs]
    )


def rope_freqs(max_pos, theta=10000.0):
    """complex64 [max_pos, 32] RoPE frequencies of the 64 rope dims."""
    inv = theta ** -(torch.arange(0, ROPE, 2, device="cuda").float() / ROPE)
    angles = torch.outer(torch.arange(max_pos, device="cuda").float(), inv)
    return torch.polar(torch.ones_like(angles), angles)


def q_norm_rope(q, pos, freqs, eps=1e-6):
    """fp32 [T, 64, 512] DSv4 q_norm_rope of raw q [T, (1,) 64, 512].

    Each head is RMS-normalized over its 512 dims (no weight), then dims
    448..511 are rotated as adjacent (re, im) pairs by freqs[pos].
    """
    x = q.reshape(-1, HEADS, DIM).float()
    x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    r = torch.view_as_complex(x[..., NOPE:].reshape(-1, HEADS, ROPE // 2, 2))
    r = torch.view_as_real(r * freqs[pos][:, None]).reshape(-1, HEADS, ROPE)
    return torch.cat([x[..., :NOPE], r], -1)


def references(batch, rows, sink=True, q=None):
    """fp32 attention of tokens rows: [len(rows), 64, 512].

    q, if given, is an fp32 [T, 64, 512] query replacing batch.q.  Indices
    outside the pool (negative or past its last row) are skipped.  Every row
    needs a valid key.
    """
    width = batch.swa_idx.shape[-1] + batch.c_idx.shape[-1]
    chunk = max(1, (1 << 26) // (width * DIM))  # 256 MB of fp32 keys
    out = []
    for i in range(0, len(rows), chunk):
        t = torch.tensor(rows[i : i + chunk], device=batch.q.device)
        keys, ok = [], []
        for cache, idx, n in (
            (batch.swa, batch.swa_idx, batch.swa_len),
            (batch.c128, batch.c_idx, batch.c_len),
        ):
            ix = idx[t, 0].long()
            m = torch.arange(ix.shape[1], device=ix.device) < n[t, None]
            m &= (ix >= 0) & (ix < cache.shape[0] * cache.shape[1])
            k = dequant(cache, torch.where(m, ix, 0).flatten())
            keys.append(k.view(*ix.shape, DIM))
            ok.append(m)
        k, ok = torch.cat(keys, 1).float(), torch.cat(ok, 1)
        qt = batch.q[t, 0].float() if q is None else q[t]
        s = torch.einsum("nhd,nkd->nhk", qt, k) * SCALE
        s = s.masked_fill(~ok[:, None], float("-inf"))
        m = s.amax(-1)
        if sink:
            m = torch.maximum(m, batch.sink)
        p = torch.exp(s - m[..., None])
        den = p.sum(-1)
        if sink:
            den = den + torch.exp(batch.sink - m)
        out.append(torch.einsum("nhk,nkd->nhd", p, k) / den[..., None])
    return torch.cat(out)


def score(o, rows, refs):
    """(max over rows and heads of max abs err / max abs ref, RMS err / RMS ref)."""
    o = o.view(-1, HEADS, DIM)
    worst, se, sr = 0.0, 0.0, 0.0
    for t, ref in zip(rows, refs):
        e = o[t].float() - ref
        worst = max(
            worst, (e.abs().amax(-1) / ref.abs().amax(-1).clamp_min(1e-3)).max().item()
        )
        se += e.square().sum().item()
        sr += ref.square().sum().item()
    return worst, (se / sr) ** 0.5
