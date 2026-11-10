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
# Implemented with reference to the PR of flashinfer (https://github.com/flashinfer-ai/flashinfer/pull/7325)
"""

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm

from .kernel_utils import (
    _topk_atomic_add,
    _topk_prefix_pick_warp0,
    _topk_red_add,
)


@cute.jit
def score_key(score: cutlass.Float32) -> cutlass.Uint32:
    """Encode an FP32 score as a uint32 key ordered from larger scores to smaller.

    Args:
        score: FP32 candidate score to encode.
    """
    bits = cutlass.Uint32(llvm.bitcast(cutlass.Uint32.mlir_type, score.ir_value()))
    key = (bits ^ cutlass.Uint32(0xFFFFFFFF)) & cutlass.Uint32(0x7FFFFFFF)
    if bits & 0x80000000:
        key = bits
    return key


@cute.jit
def key_score(key: cutlass.Uint32) -> cutlass.Float32:
    """Decode a descending uint32 score key back to FP32.

    Args:
        key: Descending key produced by ``score_key``.
    """
    bits = (key ^ cutlass.Uint32(0xFFFFFFFF)) & cutlass.Uint32(0x7FFFFFFF)
    if key & 0x80000000:
        bits = key
    return cutlass.Float32(llvm.bitcast(cutlass.Float32.mlir_type, bits.ir_value()))


@cute.jit
def load_candidate(entries, index_lo, position):
    """Reconstruct a full-key candidate from packed or split shared storage.

    Args:
        entries: Shared packed uint64 candidates, or uint32 score keys in split-index
            mode.
        index_lo: Optional shared uint16 index storage; None selects packed uint64
            entries.
        position: Flat element index in the candidate storage.
    """
    value = cutlass.Uint64(0)
    if cutlass.const_expr(index_lo is None):
        value = entries[position]
    else:
        value = (cutlass.Uint64(entries[position]) << 32) | cutlass.Uint64(
            index_lo[position]
        )
    return value


@cute.jit
def store_candidate(entries, index_lo, position, value):
    """Store the full score key and its packed or 16-bit candidate index.

    Args:
        entries: Shared packed uint64 candidates, or uint32 score keys in split-index
            mode.
        index_lo: Optional shared uint16 index storage; None selects packed uint64
            entries.
        position: Flat element index in the candidate storage.
        value: Uint64 ``(score_key << 32) | index`` candidate to store.
    """
    if cutlass.const_expr(index_lo is None):
        entries[position] = value
    else:
        entries[position] = cutlass.Uint32(value >> 32)
        index_lo[position] = cutlass.Uint16(value & 0xFFFF)


@cute.jit
def append(
    score,
    hit,
    index,
    slot: cutlass.Constexpr,
    cap: cutlass.Constexpr,
    entries: cute.Tensor,
    counts: cute.Tensor,
    index_lo: cute.Tensor | None = None,
    index_base=0,
):
    """Append ballot-selected candidates to a shared arena using one atomic per warp.

    Args:
        score: FP32 score contributed by the calling lane.
        hit: Whether the lane's candidate should be appended.
        index: Row-local key index before optional tile-pair encoding.
        slot: Query slot selecting the arena segment and shared count.
        cap: Number of entries per query slot; the caller reserves enough free space.
        entries: Shared packed uint64 candidates, or uint32 score keys in split-index
            mode.
        counts: Shared candidate counts incremented for every appended entry.
        index_lo: Optional shared uint16 index storage; None selects packed uint64
            entries.
        index_base: Absolute key start added before compact tile-pair encoding in
            split-index mode.
    """
    lane = cute.arch.lane_idx()
    mask = cute.arch.vote_ballot_sync(hit)
    total = cute.arch.popc(mask)
    base = cutlass.Int32(0)
    if total > 0:
        if lane == 0:
            base = _topk_atomic_add(counts.iterator + slot, total)
        base = cute.arch.shuffle_sync(base, 0)
        rank = cute.arch.popc(mask & ((cutlass.Int32(1) << lane) - 1))
        if hit:
            stored_index = cutlass.Uint32(index)
            if cutlass.const_expr(index_lo is not None):
                absolute = index + index_base
                stored_index = cutlass.Uint32(((absolute >> 8) << 7) | (absolute & 127))
            value = (cutlass.Uint64(score_key(score)) << 32) | cutlass.Uint64(
                stored_index
            )
            store_candidate(entries, index_lo, slot * cap + base + rank, value)


@cute.jit
def select_bound(
    values: cute.Tensor,
    valid: cute.Tensor | None,
    count,
    ltid,
    k: cutlass.Constexpr,
    limit: cutlass.Constexpr,
    wg: cutlass.Constexpr,
    bar: cutlass.Constexpr,
    hist: cute.Tensor,
    scalars: cute.Tensor,
    thread_count: cutlass.Constexpr = 128,
):
    """Return a conservative kth descending key and its inclusive survivor count.

    Args:
        values: Per-thread register tensor of uint64 packed candidates or uint32 score
            keys.
        valid: Per-element validity flags, or None to treat all-one keys as padding.
        count: Total valid candidate count across participating threads.
        ltid: Thread index within the participating 128- or 256-thread group.
        k: One-based target rank in descending score order.
        limit: Maximum survivor count before further radix refinement; use k for exact
            selection.
        wg: Group slot selecting 256 histogram bins and two shared scalars.
        bar: Named barrier ID used by all participating threads.
        hist: Shared int32 radix histogram storage.
        scalars: Shared int32 scratch for the chosen bin and output cursor.
        thread_count: Number of synchronizing threads, either 128 or 256.

    Notes:
        After two key bytes, stop only if the boundary bucket fits ``limit``; retain
        every survivor unless the remaining excess consists of exact-score ties.
    """
    pivot = cutlass.Uint32(0)
    fill = cutlass.Uint32(0xFFFFFFFF)
    before = cutlass.Int32(0)
    keep = count
    for level in cutlass.range_constexpr(4):
        shift = cutlass.const_expr(24 - 8 * level)
        threshold = k if cutlass.const_expr(level < 2) else limit
        if keep > threshold:
            hist[wg * 256 + ltid] = 0
            if cutlass.const_expr(thread_count == 128):
                hist[wg * 256 + 128 + ltid] = 0
            cute.arch.barrier(barrier_id=bar, number_of_threads=thread_count)
            for j in cutlass.range_constexpr(cute.size(values)):
                if cutlass.const_expr(values.element_type == cutlass.Uint32):
                    key = values[j]
                else:
                    key = cutlass.Uint32(values[j] >> 32)
                if cutlass.const_expr(valid is None):
                    active = key != cutlass.Uint32(0xFFFFFFFF)
                else:
                    active = valid[j]
                if cutlass.const_expr(level > 0):
                    active = active & ((key >> (shift + 8)) == (pivot >> (shift + 8)))
                if active:
                    bucket = cutlass.Int32((key >> shift) & 255)
                    _topk_red_add(hist.iterator + wg * 256 + bucket, cutlass.Int32(1))
            cute.arch.barrier(barrier_id=bar, number_of_threads=thread_count)
            _topk_prefix_pick_warp0(
                ltid, k - before, hist, wg * 256, scalars, wg * 2 + 1
            )
            cute.arch.barrier(barrier_id=bar, number_of_threads=thread_count)
            bucket = scalars[wg * 2 + 1]
            keep = before + hist[wg * 256 + bucket]
            below = cutlass.Int32(0)
            if bucket > 0:
                below = hist[wg * 256 + bucket - 1]
            before = before + below
            pivot = pivot | (cutlass.Uint32(bucket) << shift)
            fill = fill >> 8
            cute.arch.barrier(barrier_id=bar, number_of_threads=thread_count)
    return pivot | fill, keep


@cute.jit
def compact(
    ltid,
    old_tau,
    slot: cutlass.Constexpr,
    wg: cutlass.Constexpr,
    k: cutlass.Constexpr,
    cap: cutlass.Constexpr,
    trigger: cutlass.Constexpr,
    bar: cutlass.Constexpr,
    entries: cute.Tensor,
    counts: cute.Tensor,
    hist: cute.Tensor,
    scalars: cute.Tensor,
    low_registers: cutlass.Constexpr = False,
    index_lo: cute.Tensor | None = None,
):
    """Compact a streaming candidate arena and return its updated score cutoff.

    Args:
        ltid: Thread index within the participating 128-thread warpgroup.
        old_tau: Current FP32 score cutoff returned unchanged when compaction is
            unnecessary.
        slot: Query slot selecting the candidate segment and count.
        wg: Group slot selecting histogram and scalar scratch.
        k: Number of winners that must survive compaction.
        cap: Number of entries reserved per query slot.
        trigger: Count above which compaction runs, reserving room for subsequent
            appends.
        bar: Named barrier ID shared by all 128 participating threads.
        entries: Shared packed uint64 candidates, or uint32 score keys in split-index
            mode.
        counts: Shared per-slot candidate counts updated after compaction.
        hist: Shared radix histograms with 256 bins per group.
        scalars: Shared output-cursor and selected-bin scratch with two values per
            group.
        low_registers: Whether all-one padding keys replace a separate validity array.
        index_lo: Optional shared uint16 index storage; None selects packed uint64
            entries.
    """
    count = counts[slot]
    # Every warp must snapshot the count before any warp can skip compaction
    # and append the next tile. Otherwise a late reader can see that append,
    # take a different branch, and strand the WG at a conditional barrier.
    # The caller's pre-compact barrier only orders the preceding appends.
    cute.arch.barrier(barrier_id=bar, number_of_threads=128)
    tau = old_tau
    if count > trigger:
        values = cute.make_rmem_tensor((cap // 128,), cutlass.Uint64)
        # NaNs never pass the caller's score > tau predicate. Their all-one
        # descending key can mark padding without a live validity array.
        if cutlass.const_expr(low_registers):
            valid = None
        else:
            valid = cute.make_rmem_tensor((cap // 128,), cutlass.Boolean)
        for j in cutlass.range_constexpr(cap // 128):
            i = ltid + j * 128
            if cutlass.const_expr(low_registers):
                values[j] = cutlass.Uint64(0xFFFFFFFFFFFFFFFF)
            else:
                valid[j] = i < count
                values[j] = cutlass.Uint64(0)
            if i < count:
                values[j] = load_candidate(entries, index_lo, slot * cap + i)
        bound, keep = select_bound(
            values, valid, count, ltid, k, trigger, wg, bar, hist, scalars
        )
        # A coarse bucket may contain more than k distinct FP32 scores. Keep
        # all of them; truncating it would discard possible global winners.
        # Full refinement is needed only for buckets exceeding trigger, which
        # reserves period*128 free entries before returning to the mainloop.
        if ltid == 0:
            scalars[wg * 2] = 0
        cute.arch.barrier(barrier_id=bar, number_of_threads=128)
        if keep <= trigger:
            for j in cutlass.range_constexpr(cap // 128):
                key = cutlass.Uint32(values[j] >> 32)
                active = (
                    (key != cutlass.Uint32(0xFFFFFFFF))
                    if cutlass.const_expr(low_registers)
                    else valid[j]
                )
                if active & (key <= bound):
                    pos = _topk_atomic_add(scalars.iterator + wg * 2, cutlass.Int32(1))
                    store_candidate(entries, index_lo, slot * cap + pos, values[j])
            cute.arch.barrier(barrier_id=bar, number_of_threads=128)
        else:
            # An oversized exact-score tie can still exceed trigger. Reserve
            # strict winners first, then fill to k with interchangeable ties.
            for tie in cutlass.range_constexpr(2):
                for j in cutlass.range_constexpr(cap // 128):
                    key = cutlass.Uint32(values[j] >> 32)
                    hit = key < bound if cutlass.const_expr(tie == 0) else key == bound
                    active = (
                        (key != cutlass.Uint32(0xFFFFFFFF))
                        if cutlass.const_expr(low_registers)
                        else valid[j]
                    )
                    if active & hit:
                        pos = _topk_atomic_add(
                            scalars.iterator + wg * 2, cutlass.Int32(1)
                        )
                        if pos < k:
                            store_candidate(
                                entries, index_lo, slot * cap + pos, values[j]
                            )
                cute.arch.barrier(barrier_id=bar, number_of_threads=128)
            keep = cutlass.Int32(k)
        if ltid == 0:
            counts[slot] = keep
        cute.arch.barrier(barrier_id=bar, number_of_threads=128)
        tau = key_score(bound)
    return tau
