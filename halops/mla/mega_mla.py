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

"""Python wrapper for the fused MLA megakernel (Phase 1 + Phase 2 + KV cache write)."""

import torch
from torch.utils.weak import WeakTensorKeyDictionary

import halops.halops_kernel as halops_kernel

__all__ = ["mega_mla", "build_kv_runs"]


# Cache for converted scale factors, KEYED ON THE TENSOR ITSELF (weakly).
#
# ⚠️ IT USED TO BE KEYED ON `id(sf_fp32)`, AND THAT WAS A CORRECTNESS BUG.  CPython
# reuses the address of a freed object, so once a weight tensor is released -- reloading a
# model, or a test harness building one layer per case -- the NEXT scale tensor can land
# at the same address, the lookup HITS, and the kernel dequantizes with the PREVIOUS
# weight's scale factors.  The result is finite and plausible, just wrong: measured
# `rel max_diff` 8.4% against a 2-4% fp8 noise band.  Two of the four observed collisions
# were also across DIFFERENT SHAPES ((8,32) vs (32,64)), i.e. the kernel read a scale
# buffer smaller than the one it was told to expect -- an illegal memory access, which is
# how this surfaced.  Observed 4 stale hits in one 7-case sweep
# (test/manual/mla_megakernel/sf_cache_probe.py), so it is not a theoretical race.
#
# A WeakTensorKeyDictionary keys on tensor IDENTITY and holds only a weak reference, so an
# entry is dropped when its weight is freed and an address can never be inherited.  It
# cannot cache across two distinct tensors with equal contents; that is the right
# trade -- the conversion is once per weight per process, and silently reusing another
# tensor's conversion is exactly the failure above.
_sf_cache: WeakTensorKeyDictionary = WeakTensorKeyDictionary()
_wo_a_sf_cache: WeakTensorKeyDictionary = WeakTensorKeyDictionary()

# S13: gmem workspace cache, keyed by device.  Two things depend on it:
#
#   * the grid-barrier counter lives in the first 4 bytes and used to be zeroed by
#     the host on EVERY call.  Under a CUDA graph that memset is a graph node and
#     cost 2.71 us per replay to clear 4 bytes (measured: empty graph 1.35 us,
#     empty graph + one 128 B zero_() 4.06 us).  The kernel now resets the counter
#     itself on the way out, so the host only has to zero a FRESH buffer — which
#     happens here, outside any graph capture, on the first call per size.
#   * `torch.empty` of ~34 MB per call was itself host work the graph hid but the
#     eager path paid.
#
# Growing only (never shrinks) so a later smaller T reuses the big buffer; the
# kernel slices what it needs.  Keyed by device only, so a size change reallocates
# and re-zeroes, which is exactly what a fresh buffer needs.
_ws_cache: dict = {}


def _get_workspace(nbytes: int, device, barrier_bytes: int) -> torch.Tensor:
    ws = _ws_cache.get(device)
    if ws is None or ws.numel() < nbytes:
        ws = torch.empty(nbytes, dtype=torch.uint8, device=device)
        ws[:barrier_bytes].zero_()  # fresh buffer: the counter must start at 0
        _ws_cache[device] = ws
    return ws


def reorder_sf_to_cutlass_sfatom(sf_u8: torch.Tensor) -> torch.Tensor:
    """Reorder a logical [M, K/32] uint8 scale tensor into CUTLASS SfAtom layout.

    Mirrors the swizzle branch of group_wise_quant_kernel
    (csrc/kernel/gemm/group_wise_quant.cu:239-260), which is the layout the
    block-scaled UMMA expects to read directly from gmem.

    The atom is 128 rows x 4 K-groups = 512 bytes; within an atom the byte
    order is ``mn_i * 16 + mn_j * 4 + sf_l``.
    """
    M, kb_dim = sf_u8.shape
    assert M % 128 == 0, f"M must be a multiple of 128 for SfAtom layout, got {M}"
    assert kb_dim % 4 == 0, f"kb_dim must be a multiple of 4, got {kb_dim}"
    dev = sf_u8.device

    n_idx = torch.arange(M, device=dev)
    kb_idx = torch.arange(kb_dim, device=dev)
    n_grid, kb_grid = torch.meshgrid(n_idx, kb_idx, indexing="ij")

    n_tile = n_grid // 128
    n_local = n_grid % 128
    mn_i = n_local % 32
    mn_j = n_local // 32
    k_tile = kb_grid // 4
    sf_l = kb_grid % 4
    num_k_tiles = kb_dim // 4

    cutlass_byte = (n_tile * num_k_tiles + k_tile) * 512 + mn_i * 16 + mn_j * 4 + sf_l
    out = torch.empty_like(sf_u8)
    out.view(-1)[cutlass_byte.reshape(-1)] = sf_u8.reshape(-1)
    return out.contiguous()


def convert_sf_fp32_to_g32_ue8m0(sf_fp32: torch.Tensor, M: int, K: int) -> torch.Tensor:
    """Convert [M/128, K/128] fp32 blockwise scales → SfAtom-swizzled [M, K/32] ue8m0.

    Two steps:
      1. Expand the 128x128 blockwise fp32 scale to a logical per-group-32
         ue8m0 tensor (4 group-32 slots per 128 K-elements, 128 rows per block).
      2. Reorder into the CUTLASS SfAtom 128x4 interleaved layout so the
         block-scaled UMMA can consume it directly.
    """
    assert sf_fp32.shape == (
        M // 128,
        K // 128,
    ), f"sf_fp32 shape {sf_fp32.shape} != ({M//128}, {K//128})"
    abs_s = torch.clamp(torch.abs(sf_fp32), min=1e-38)
    exp = torch.ceil(torch.log2(abs_s)) + 127
    exp = torch.clamp(exp, 0, 254).to(torch.uint8)
    # Repeat along K: 128/32 = 4 group-32 slots per block
    exp_k = exp.repeat_interleave(4, dim=1)  # [M/128, K/32]
    # Repeat along M: 128 rows per block
    exp_mk = exp_k.repeat_interleave(128, dim=0)  # [M, K/32]
    return reorder_sf_to_cutlass_sfatom(exp_mk.contiguous())


def convert_wo_a_sf_packed_to_g32_ue8m0(
    sf_packed: torch.Tensor, G: int, R: int, K: int
) -> torch.Tensor:
    """Convert wo_a's packed-int32 MN-major scale → SfAtom-swizzled [G*R, K/32] ue8m0.

    wo_a's scale is stored as ``[G, R, ceil(K/128/4)]`` int32, MN-major (r has
    stride 1), with four ue8m0 exponent bytes packed per word. The scalar kernel
    decoded it as::

        word = sf[g, r, qg >> 2]
        exp  = (word >> ((qg & 3) * 8)) & 0xFF     # already biased by 127

    where ``qg`` is the 128-granularity K group. Here we unpack all of them, fan
    each 128-element group out to the four group-32 slots it covers, and reorder
    into the CUTLASS SfAtom layout the block-scaled UMMA reads.
    """
    assert (
        sf_packed.shape[0] == G and sf_packed.shape[1] == R
    ), f"sf_packed shape {tuple(sf_packed.shape)} != ({G}, {R}, *)"
    n_qg = K // 128  # 128-granularity groups along K
    packed_k = sf_packed.shape[2]
    assert packed_k * 4 >= n_qg, f"packed_k={packed_k} too small for n_qg={n_qg}"

    # int32 may be negative; widen before shifting so the top byte survives.
    words = sf_packed.to(torch.int64) & 0xFFFFFFFF
    shifts = torch.tensor([0, 8, 16, 24], device=sf_packed.device, dtype=torch.int64)
    exp = (words.unsqueeze(-1) >> shifts) & 0xFF  # [G, R, packed_k, 4]
    exp = exp.reshape(G, R, packed_k * 4)[:, :, :n_qg]  # [G, R, n_qg]
    # Each 128-element block covers 4 group-32 slots.
    exp = exp.repeat_interleave(128 // 32, dim=2)  # [G, R, K/32]

    logical = exp.reshape(G * R, K // 32).to(torch.uint8).contiguous()
    return reorder_sf_to_cutlass_sfatom(logical)


def _get_g32_sf(sf_fp32: torch.Tensor, M: int, K: int) -> torch.Tensor:
    """Get or create cached group-32 ue8m0 scale from fp32 blockwise scale."""
    hit = _sf_cache.get(sf_fp32)
    if hit is None:
        _sf_cache[sf_fp32] = ((M, K), convert_sf_fp32_to_g32_ue8m0(sf_fp32, M, K))
        return _sf_cache[sf_fp32][1]
    # The conversion depends on (M, K), so the same tensor asked about under different
    # dimensions would silently reuse the wrong layout.  Nothing does that today; assert
    # rather than leave it to be discovered as a wrong number.
    assert hit[0] == (
        M,
        K,
    ), f"scale tensor cached for M,K={hit[0]} but now requested as {(M, K)}"
    return hit[1]


def _get_wo_a_g32_sf(sf_packed: torch.Tensor, G: int, R: int, K: int) -> torch.Tensor:
    """Get or create cached SfAtom ue8m0 scale from wo_a's packed int32 scale."""
    # A SEPARATE cache, not a ("wo_a", tensor) key: a weak-tensor-keyed dict keys on the
    # tensor alone, and wo_a's packed int32 scale takes a different conversion from the
    # fp32 blockwise one, so the two must not share a namespace.
    hit = _wo_a_sf_cache.get(sf_packed)
    if hit is None:
        _wo_a_sf_cache[sf_packed] = (
            (G, R, K),
            convert_wo_a_sf_packed_to_g32_ue8m0(sf_packed, G, R, K),
        )
        return _wo_a_sf_cache[sf_packed][1]
    assert hit[0] == (
        G,
        R,
        K,
    ), f"wo_a scale cached for G,R,K={hit[0]} but now requested as {(G, R, K)}"
    return hit[1]


def build_kv_runs(
    kv_indices: torch.Tensor, topk_lengths: torch.Tensor, page_size: int
) -> tuple[torch.Tensor, int]:
    """Compress a per-token slot list into the kernel's `kv_runs` descriptor.

    The kernel no longer reads a [T, topk] slot list; it reads one int4 per token,
    ``(slot0, len0, slot1, _)``, meaning: window row g lands on cache slot
    ``slot0 + step*g`` for ``g < len0`` and ``slot1 + step*(g-len0)`` after that
    (``step`` = the returned ``kv_run_step``). That is exact for both supported
    modes -- SWA's window is one descending slot run, HCA's are runs of 2 -- and it
    is what let warp 7 (which used to materialize TMA coords, a valid mask and 64
    scattered scale reads into smem per KV tile) be deleted.

    Runs are cut at page boundaries as well as at slot discontinuities: a page holds
    ``stride_kv_block / 576`` TMA rows of which only the first ``page_size`` are
    tokens, so slot-contiguous is not row-contiguous across a page.

    This is a verification-path helper -- in production the descriptor should come
    straight out of the metadata kernel instead of being reverse-engineered from a
    slot list. Raises if the window needs more than 2 runs, which the kernel's
    MAX_KV_SEGS == 2 cannot express.
    """
    T, topk = kv_indices.shape
    dev = kv_indices.device
    lengths = topk_lengths.to(torch.int64)
    col = torch.arange(topk, device=dev)
    valid = col[None, :] < lengths[:, None]
    assert bool((kv_indices >= 0)[valid].all()), "padding must be a suffix of each row"

    idx = kv_indices.to(torch.int64)
    # Direction: read it off the first token that has >= 2 rows; a single-row window
    # is direction-agnostic.
    multi = (lengths >= 2).nonzero()
    step = 1 if multi.numel() == 0 else int(idx[multi[0, 0], 1] - idx[multi[0, 0], 0])
    assert step in (1, -1), f"KV slots must be consecutive, got step {step}"

    shift = page_size.bit_length() - 1
    assert 1 << shift == page_size, f"page_size must be a power of two, got {page_size}"
    prev, cur = idx[:, :-1], idx[:, 1:]
    brk = ((cur != prev + step) | ((cur >> shift) != (prev >> shift))) & valid[:, 1:]
    n_runs = 1 + brk.sum(1)
    assert int(n_runs.max()) <= 2, f"window needs {int(n_runs.max())} runs > 2"

    # len0 = first break column (== the run-0 length), or the whole window.
    first_brk = brk.int().argmax(1) + 1
    len0 = torch.where(brk.any(1), first_brk, lengths)
    slot1 = idx.gather(1, len0.clamp(max=topk - 1)[:, None])[:, 0]
    kv_runs = torch.stack([idx[:, 0], len0, slot1, torch.zeros_like(len0)], dim=1)
    return kv_runs.to(torch.int32).contiguous(), step


def _mega_mla_workspace(
    q_or_x: torch.Tensor,
    wo_a_weight: torch.Tensor,
    wo_b_weight: torch.Tensor,
    wq_a_weight: torch.Tensor = None,
    wkv_weight: torch.Tensor = None,
) -> torch.Tensor:
    """The gmem workspace: pre-quantized operand-B activations, the K-split
    partials, the barrier banks and the work plan.

    The size comes from the kernel itself (``mega_mla_workspace_bytes``) rather
    than from a copy of the layout here: every block scales with the token count
    or with a weight dimension, so a mirrored constant is both a drift hazard and
    a cap on which models can run.
    """
    total, zero_prefix = halops_kernel.mega_mla_workspace_bytes(
        q_or_x.shape[0], wo_a_weight, wo_b_weight, wq_a_weight, wkv_weight
    )
    return _get_workspace(total, q_or_x.device, zero_prefix)


def mega_mla(
    q_or_x: torch.Tensor,
    kv_cache: torch.Tensor,
    page_size: int,
    kv_runs: torch.Tensor,
    topk: int,
    kv_run_step: int,
    topk_lengths: torch.Tensor,
    positions: torch.Tensor,
    freqs_real: torch.Tensor,
    wo_a_weight: torch.Tensor,
    wo_a_scale_inv: torch.Tensor,
    wo_b_weight: torch.Tensor,
    wo_b_scale_inv: torch.Tensor,
    out: torch.Tensor,
    sm_scale: float,
    wq_a_weight: torch.Tensor = None,
    wq_a_scale_inv: torch.Tensor = None,
    q_norm_weight: torch.Tensor = None,
    wq_b_weight: torch.Tensor = None,
    wq_b_scale_inv: torch.Tensor = None,
    q_eps: float = 1e-6,
    use_phase1: bool = False,
    wkv_weight: torch.Tensor = None,
    wkv_scale_inv: torch.Tensor = None,
    kv_norm_weight: torch.Tensor = None,
    kv_eps: float = 1e-6,
    use_kv_write: bool = False,
    swa_loc: torch.Tensor = None,
    attn_sink: torch.Tensor = None,
    # HCA (compress_ratio=128): bf16 [1024, 4096], no scale factor.  None = HCA off.
    wkv_gate_weight: torch.Tensor = None,
    # The compressor's kv_score ring, fp32 [slots, 1024] = | kv 512 | score 512 |, and
    # this launch's slot per token, int32 [T].  The wkv_gate epilogue writes the ring
    # directly, so there is no separate `c128_write_decode` pass.
    #   ring_row[t] = req_pool_idx[t] * ring_size + positions[t] % ring_size
    #   ring_size   = 128 (no spec) / 256 (spec)          -- c_plan.cuh:359-371
    # ⚠️ The ring is BOTH input and output: the 128-slot weighted sum that consumes it
    # reads 127 slots written by earlier launches.  Reset it per iteration in any
    # accuracy probe, and pre-fill with a FINITE sentinel -- otherwise "wrote the right
    # value" and "wrote nothing, read last round's right value" are indistinguishable.
    kv_score_ring: torch.Tensor = None,
    kv_gate_ring_row: torch.Tensor = None,
    # HCA step 2 (c128): 128-slot weighted sum + c128 cache store.  Gated on
    # c128_ape being present.  c128_row_out is fp32 [T, 512], the raw weighted-sum
    # rows for the probe's Gate A; pass None in production.  c128_pool +
    # (c128_stride_kv_block, c128_page_block) define the c128 pool's pages the
    # same way the SWA cache does.
    c128_ape: torch.Tensor = None,
    c128_norm_weight: torch.Tensor = None,
    c128_freqs_real: torch.Tensor = None,
    c128_out_loc: torch.Tensor = None,
    c128_row_out: torch.Tensor = None,
    c128_pool: torch.Tensor = None,
    c128_stride_kv_block: int = 0,
    c128_page_block: int = 0,
    # §6.4: the c128 ATTENTION segment.  Each token's window becomes
    # SWA rows ++ c128 rows, the second part being compressed entries
    # `c128_attn_slot0[t] .. +c128_attn_len[t]-1` of the c128 pool -- ASCENDING and
    # consecutive, which is what a per-request c128 allocation buys over deriving the
    # slots from the full-pool page table.  Both None = SWA-only (zero extra tiles).
    # ⚠️ Requires c128_page_block >= 64 (the kernel's B_TOPK) so a KV tile crosses at
    # most one page boundary; the kernel TORCH_CHECKs it.
    c128_attn_slot0: torch.Tensor = None,
    c128_attn_len: torch.Tensor = None,
    # Max query tokens per REQUEST (the speculative draft length).  Used only to narrow
    # the two KV-cache rendezvous the kernel waits on at attention entry: a token's
    # top-k rows can name the cache slots of the other draft tokens of its OWN request
    # and nothing else, so it need only wait for the groups covering [t-(D-1), t+(D-1)]
    # -- at most two when D <= the N tile.
    # ⚠️ 0 MEANS "UNKNOWN" and makes the kernel wait for EVERY token group.  Correct but
    # slower; that is the safe direction, since too few groups is a wrong number.
    # ⚠️ AN UPPER BOUND, NOT A UNIFORM SHAPE: passing max(accepted lengths) is fine.
    draft_tokens: int = 0,
    # True == the caller already ran `mega_mla_plan` on this stream with the SAME
    # workspace, T, topk and KV lengths.  Only for benchmarks that want the plan
    # kernel's time reported separately; a stale plan is a hang or wrong output.
    plan_ready: bool = False,
) -> None:
    """Fused MLA megakernel.

    Phase 2 only (use_phase1=False): reads pre-prepared Q from gmem.
    Phase 1+2 (use_phase1=True): computes Q from x (wq_a + q_norm + wq_b + self-norm + RoPE).
    KV cache write (use_kv_write=True): computes KV and writes to paged cache at swa_loc.

    Weight scales (wq_a/wq_b/wkv/wo_b) are pre-converted from fp32 128-gran blockwise
    to ue8m0 group-32 format via convert_sf_fp32_to_g32_ue8m0().
    """
    e = torch.empty(0, dtype=torch.float8_e4m3fn, device=q_or_x.device)
    f = torch.empty(0, dtype=torch.float32, device=q_or_x.device)
    b = torch.empty(0, dtype=torch.bfloat16, device=q_or_x.device)
    u = torch.empty(0, dtype=torch.uint8, device=q_or_x.device)
    i = torch.empty(0, dtype=torch.int32, device=q_or_x.device)

    # Pre-convert weight scales to group-32 ue8m0 format (cached)
    # Shapes come from the weights themselves, so nothing here can drift from
    # the kernel's config; the C++ side then checks those weights against it.
    def _mk(w, sf):
        if sf is None:
            return u
        assert w is not None, "a weight's scale was passed without the weight"
        # Already in the layout the kernel reads -- pass it through.  `group_wise_quant`
        # produces exactly this (uint8 [M, K/32], SfAtom-swizzled), so a caller that
        # quantized its weights with the package's own quantizer hands the SAME bytes to
        # this kernel and to `dense_gemm` instead of deriving a second scale for each.
        # ⚠️ THE TWO PATHS ARE NOT THE SAME QUANTIZATION, and the difference is GRANULARITY,
        # not rounding: both ceil `log2(scale)` to a power of two (group_wise_quant.cu:190
        # and `convert_sf_fp32_to_g32_ue8m0`), but the fp32 path carries ONE exponent per
        # 128x128 block and fans it out to all 128 rows and all four group-32 slots, while
        # `group_wise_quant` gives every (row, 32 elements) its own.  The fp32 path also
        # assumes the fp8 bytes were produced against the ORIGINAL fp32 scale, which the
        # kernel then dequantizes with the rounded-up power of two -- consistent only when
        # those scales already are powers of two.
        if sf.dtype == torch.uint8:
            assert sf.shape == (w.shape[-2], w.shape[-1] // 32), (
                f"a pre-converted scale must be [M, K/32] = "
                f"{(w.shape[-2], w.shape[-1] // 32)}, got {tuple(sf.shape)}"
            )
            return sf
        return _get_g32_sf(sf, w.shape[-2], w.shape[-1])

    wq_a_sf_g32 = _mk(wq_a_weight, wq_a_scale_inv)
    wq_b_sf_g32 = _mk(wq_b_weight, wq_b_scale_inv)
    wkv_sf_g32 = _mk(wkv_weight, wkv_scale_inv)
    wo_b_sf_g32 = _mk(wo_b_weight, wo_b_scale_inv)
    # wo_a's scale may arrive either as the group-32 uint8 SfAtom tensor the other four use
    # (pass-through, `_mk`) or as production's 128-granularity packed int32 -- its own
    # converter.  C++ derives G from `wo_a_weight` itself, so neither form has to carry it.
    wo_a_sf_g32 = (
        _mk(wo_a_weight, wo_a_scale_inv)
        if wo_a_scale_inv is not None and wo_a_scale_inv.dtype == torch.uint8
        else (
            _get_wo_a_g32_sf(
                wo_a_scale_inv,
                wo_a_scale_inv.shape[0],
                wo_a_weight.numel() // wo_a_weight.shape[-1] // wo_a_scale_inv.shape[0],
                wo_a_weight.shape[-1],
            )
            if wo_a_scale_inv is not None
            else u
        )
    )

    workspace = _mega_mla_workspace(
        q_or_x, wo_a_weight, wo_b_weight, wq_a_weight, wkv_weight
    )

    # The plan pass.  Separate op on purpose: it is a different kernel with its own
    # launch, so a benchmark can time it apart from the megakernel (pass
    # plan_ready=True after calling mega_mla_plan yourself).  In production it
    # belongs in the metadata kernel that already runs ahead of this one.
    # The workspace is passed IN, not fetched again: the plan writes into it, and
    # a second _get_workspace call could hand back a freshly grown buffer.
    if not plan_ready:
        from halops.mla.mega_mla_plan import mega_mla_plan

        mega_mla_plan(
            q_or_x, kv_runs, topk, topk_lengths, c128_attn_len, wo_a_weight, workspace
        )

    halops_kernel.mega_mla(
        q_or_x,
        kv_cache,
        int(page_size),
        kv_runs,
        int(topk),
        int(kv_run_step),
        topk_lengths,
        positions,
        freqs_real,
        wo_a_weight,
        wo_a_scale_inv,
        wo_b_weight,
        out,
        float(sm_scale),
        wq_a_weight if wq_a_weight is not None else e,
        q_norm_weight if q_norm_weight is not None else b,
        wq_b_weight if wq_b_weight is not None else e,
        float(q_eps),
        bool(use_phase1),
        wkv_weight if wkv_weight is not None else e,
        kv_norm_weight if kv_norm_weight is not None else b,
        float(kv_eps),
        bool(use_kv_write),
        swa_loc if swa_loc is not None else i,
        attn_sink if attn_sink is not None else f,
        wq_a_sf_g32,
        wq_b_sf_g32,
        wkv_sf_g32,
        wo_b_sf_g32,
        wo_a_sf_g32,
        workspace,
        wkv_gate_weight if wkv_gate_weight is not None else b,
        kv_score_ring if kv_score_ring is not None else f,
        kv_gate_ring_row if kv_gate_ring_row is not None else i,
        c128_ape if c128_ape is not None else f,
        c128_norm_weight if c128_norm_weight is not None else b,
        c128_freqs_real if c128_freqs_real is not None else f,
        c128_out_loc if c128_out_loc is not None else i,
        c128_row_out if c128_row_out is not None else f,
        c128_pool if c128_pool is not None else u,
        int(c128_stride_kv_block),
        int(c128_page_block),
        c128_attn_slot0 if c128_attn_slot0 is not None else i,
        c128_attn_len if c128_attn_len is not None else i,
        int(draft_tokens),
    )
