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

from . import topk
from .config import needs_repair
from .kernel_utils import (
    _topk_atomic_add,
    _topk_prefix_pick_warp0,
    _topk_red_add,
)


@cute.jit
def finish_shared(
    tid,
    row,
    query_group_id: cutlass.Constexpr,
    k: cutlass.Constexpr,
    candidate_capacity: cutlass.Constexpr,
    barrier_id: cutlass.Constexpr,
    packed_candidates,
    candidate_counts,
    score_histogram,
    shared_counters,
    topk_indices,
    topk_indices_offset,
):
    """Select and emit the exact top-k row from one group's shared candidates.

    Args:
        tid: Thread index within the participating 128-thread warpgroup.
        row: Live query row receiving the selected indices.
        query_group_id: Compile-time slot of this query within the CTA's repair group.
        k: Number of output indices selected for the row.
        candidate_capacity: Shared candidate capacity per query slot.
        barrier_id: Named barrier used by all 128 threads, including both tie-emission
            passes.
        packed_candidates: Shared uint64 ``(score_key << 32) | row_local_index``
            candidates.
        candidate_counts: Shared count of valid candidates for each query slot.
        score_histogram: Shared int32 radix histograms, with 256 bins per query slot.
        shared_counters: Shared cursor and selected-bin scratch, with two values per
            query slot.
        topk_indices: Int32 output ``[M, k]`` receiving selected indices and -1 padding.
        topk_indices_offset: Optional int32/int64 ``[M]`` offsets added only to valid
            output indices; None keeps row-local indices and padding remains -1.
    """
    count = candidate_counts[query_group_id]
    values = cute.make_rmem_tensor((candidate_capacity // 128,), cutlass.Uint32)
    for j in cutlass.range_constexpr(candidate_capacity // 128):
        i = tid + j * 128
        values[j] = cutlass.Uint32(0xFFFFFFFF)
        if i < count:
            values[j] = cutlass.Uint32(
                packed_candidates[query_group_id * candidate_capacity + i] >> 32
            )
    bound, _ = topk.select_bound(
        values,
        None,
        count,
        tid,
        k,
        k,
        query_group_id,
        barrier_id,
        score_histogram,
        shared_counters,
    )
    for j in cutlass.range(tid, k, 128):
        topk_indices[row, j] = -1
    if tid == 0:
        shared_counters[query_group_id * 2] = 0
    cute.arch.barrier(barrier_id=barrier_id, number_of_threads=128)
    for tie in cutlass.range_constexpr(2):
        for j in cutlass.range_constexpr(candidate_capacity // 128):
            key = values[j]
            hit = key < bound if cutlass.const_expr(tie == 0) else key == bound
            emit(
                (key != cutlass.Uint32(0xFFFFFFFF)) & hit,
                packed_candidates[query_group_id * candidate_capacity + tid + j * 128],
                row,
                k,
                shared_counters.iterator + query_group_id * 2,
                topk_indices,
                topk_indices_offset,
            )
        cute.arch.barrier(barrier_id=barrier_id, number_of_threads=128)


@cute.jit
def select_global(
    row,
    tid,
    k: cutlass.Constexpr,
    candidate_capacity: cutlass.Constexpr,
    candidates,
    candidate_counts,
    topk_indices,
    score_histogram,
    shared_counters,
    topk_indices_offset,
    partitions: cutlass.Constexpr = 0,
):
    """Select a successful row's exact top-k from the global filter arena.

    Args:
        row: Live query row to select.
        tid: Thread index within the participating 256-thread CTA.
        k: Number of output indices selected for the row.
        candidate_capacity: Entries per row in the global candidate arena, including any
            partition headers.
        candidates: Int64 ``[M, candidate_capacity]`` packed score-key and
            row-local-index arena.
        candidate_counts: Int32 row metadata whose ``[row, 0]`` entry is the valid
            candidate count.
        topk_indices: Int32 output ``[M, k]`` receiving selected indices and -1 padding.
        score_histogram: Shared int32 scratch containing 256 radix bins.
        shared_counters: Two shared int32 values for the output cursor and selected bin.
        topk_indices_offset: Optional int32/int64 ``[M]`` offsets added only to valid
            output indices; None keeps row-local indices and padding remains -1.
        partitions: Four for warp-partitioned candidates, or zero for compact candidate
            rows.
    """
    count = candidate_counts[row, 0]
    if cutlass.const_expr(partitions > 0):
        pitch = candidate_capacity // 4
        prefix_counts = cute.make_rmem_tensor((4,), cutlass.Int32)
        for warp in cutlass.range_constexpr(4):
            prefix_counts[warp] = cutlass.Int32(candidates[row, (warp + 1) * pitch - 1])
    keys = cute.make_rmem_tensor((candidate_capacity // 256,), cutlass.Uint32)
    for j in cutlass.range_constexpr(candidate_capacity // 256):
        i = tid + j * 256
        keys[j] = cutlass.Uint32(0xFFFFFFFF)
        valid = i < count
        if cutlass.const_expr(partitions > 0):
            valid = cutlass.Boolean(False)
            for warp in cutlass.range_constexpr(4):
                valid |= (i >= warp * pitch) & (
                    i < warp * pitch + min(prefix_counts[warp], pitch - 1)
                )
        if valid:
            keys[j] = cutlass.Uint32(cutlass.Uint64(candidates[row, i]) >> 32)
    bound, _ = topk.select_bound(
        keys,
        None,
        count,
        tid,
        k,
        k,
        0,
        1,
        score_histogram,
        shared_counters,
        thread_count=256,
    )
    for j in cutlass.range(tid, k, 256):
        topk_indices[row, j] = -1
    if tid == 0:
        shared_counters[0] = 0
    cute.arch.barrier(barrier_id=1, number_of_threads=256)
    for tie in cutlass.range_constexpr(2):
        for j in cutlass.range_constexpr(candidate_capacity // 256):
            key = keys[j]
            hit = key < bound if cutlass.const_expr(tie == 0) else key == bound
            value = cutlass.Uint64(0)
            if key != cutlass.Uint32(0xFFFFFFFF):
                value = cutlass.Uint64(candidates[row, tid + j * 256])
            emit(
                (key != cutlass.Uint32(0xFFFFFFFF)) & hit,
                value,
                row,
                k,
                shared_counters.iterator,
                topk_indices,
                topk_indices_offset,
            )
        cute.arch.barrier(barrier_id=1, number_of_threads=256)


@cute.jit
def partition_counts(
    row, cap: cutlass.Constexpr, partitions: cutlass.Constexpr, candidates
):
    """Read immutable partition headers to recover the stored count and repair marker.

    Args:
        row: Live query row whose partition headers are read.
        cap: Total entries per candidate row, including four partition headers.
        partitions: Partition-layout parameter; the current implementation always reads
            four partitions.
        candidates: Int64 global arena with each warp's count stored at its partition's
            last entry.

    Notes:
        Headers remain immutable while success CTAs run, so other CTAs can safely
        compute repair eligibility.
    """
    pitch = cap // 4
    total = cutlass.Int32(0)
    filled = cutlass.Int32(0)
    overflow = cutlass.Boolean(False)
    for part in cutlass.range_constexpr(4):
        count = cutlass.Int32(candidates[row, (part + 1) * pitch - 1])
        total += count
        filled += min(count, pitch - 1)
        overflow |= count > pitch - 1
    marker = -total - 1 if overflow else total
    return filled, marker


@cute.jit
def emit(
    hit,  # per-lane: does this candidate make the cut
    value,  # the packed (key, index) word; only the index is written
    row,
    k: cutlass.Constexpr,  # truncation bound, so ties cannot overflow the row
    cursor,  # shared output cursor, bumped one atomic per warp
    topk_indices,
    topk_indices_offset,  # None, or [M] added to the index
):
    """Emit ballot-selected candidate indices without exceeding the row's top-k width.

    Args:
        hit: Whether the calling lane's candidate is selected.
        value: Uint64 packed score key and row-local index; only the index is emitted.
        row: Output query row.
        k: Maximum number of indices that may be written to the row.
        cursor: Shared-memory Int32 output cursor advanced once per participating warp.
        topk_indices: Int32 output receiving valid indices at the reserved positions.
        topk_indices_offset: Optional int32/int64 ``[M]`` offsets added only to valid
            output indices; None keeps row-local indices and padding remains -1.
    """
    lane = cute.arch.lane_idx()
    mask = cute.arch.vote_ballot_sync(hit)
    count = cutlass.Int32(cute.arch.popc(mask))
    pos = cutlass.Int32(0)
    if count > 0:
        if lane == 0:
            pos = _topk_atomic_add(cursor, count)
        pos = cutlass.Int32(
            cute.arch.shuffle_sync(pos, 0)
            + cute.arch.popc(mask & ((cutlass.Uint32(1) << lane) - 1))
        )
        if hit & (pos < k):
            index = cutlass.Int32(value & 0xFFFFFFFF)
            if cutlass.const_expr(topk_indices_offset is not None):
                index += cutlass.Int32(topk_indices_offset[row])
            topk_indices[row, pos] = index


@cute.jit
def schedule(
    block,  # this CTA's blockIdx.z
    sms,  # runtime grid size; few failures can be split across CTAs
    candidate_counts,  # holds the failed-group queue SuccessTopK published
    key_start,  # [M] inclusive row bounds used by the repair contract
    key_end,  # [M] exclusive row bounds used by the repair contract
    nrows,
    k: cutlass.Constexpr,  # the success bar, re-tested here
    candidate_capacity: cutlass.Constexpr,
    query_block_size: cutlass.Constexpr,
):
    """Assign a failed query group and optional split-repair partition to a CTA.

    Args:
        block: This CTA's zero-based ``blockIdx.z``.
        sms: Runtime repair grid size available for distributing failed groups.
        candidate_counts: Global int32 row markers followed by the failed-group count
            and queue.
        key_start: Nonnegative int32 ``[M]`` inclusive key starts in full-KV
            coordinates.
        key_end: Int32 ``[M]`` exclusive absolute key ends in full-KV coordinates,
            between each row's start and the full KV row count.
        nrows: Number M of live query rows; the queue header is at
            ``candidate_counts[M, 0]``.
        k: Top-k count used to recheck row completeness.
        candidate_capacity: Maximum stored candidates per row used by the repair
            predicate.
        query_block_size: Number of query rows in each repair group.
    """
    groups = (nrows + query_block_size - 1) // query_block_size
    failed = candidate_counts[nrows, 0]
    parts = cutlass.Int32(1)
    part = cutlass.Int32(0)
    target = cutlass.Int32(groups)
    if (failed > 0) & (failed <= sms // 2):
        parts = sms // failed
        part = block // failed
        if part < parts:
            target = candidate_counts[nrows + 1 + block % failed, 0]
    elif failed > 0:
        target = next_group(
            block,
            sms,
            candidate_counts,
            key_start,
            key_end,
            nrows,
            k,
            candidate_capacity,
            True,
            query_block_size,
            parts,
        )
    return (target, part, parts, failed)


@cute.jit
def next_group(
    start,
    step,
    candidate_counts,
    key_start,
    key_end,
    nrows,
    k: cutlass.Constexpr,
    candidate_capacity: cutlass.Constexpr,
    enabled: cutlass.Constexpr,
    query_block_size: cutlass.Constexpr,
    parts,
):
    """Find the next query group containing at least one row that needs repair.

    Args:
        start: First query-group index to inspect.
        step: Query-group stride between assignments to the same persistent CTA.
        candidate_counts: Global int32 row markers followed by the failed-group count
            and queue.
        key_start: Nonnegative int32 ``[M]`` inclusive key starts in full-KV
            coordinates.
        key_end: Int32 ``[M]`` exclusive absolute key ends in full-KV coordinates,
            between each row's start and the full KV row count.
        nrows: Number M of live query rows; the queue header is at
            ``candidate_counts[M, 0]``.
        k: Top-k count used to recheck row completeness.
        candidate_capacity: Maximum stored candidates per row used by the repair
            predicate.
        enabled: Whether to scan repair markers; False returns the starting group.
        query_block_size: Number of query rows in each repair group.
        parts: Number of KV partitions per group; values above one end the persistent
            scan.
    """
    group = cutlass.Int32(start)
    groups = (nrows + query_block_size - 1) // query_block_size
    if cutlass.const_expr(enabled):
        if (parts > 1) | (
            (candidate_counts[nrows, 0] > 0) & (candidate_counts[nrows, 0] <= step // 2)
        ):
            group = groups
        found = cutlass.Boolean(False)
        while (group < groups) & ~found & enabled:
            for offset in cutlass.range_constexpr(query_block_size):
                row = group * query_block_size + offset
                if row < nrows:
                    mark = candidate_counts[row, 1]
                    found |= needs_repair(
                        mark,
                        key_end[row] - key_start[row],
                        k,
                        candidate_capacity,
                    )
            if not found:
                group += step
    return group


@cute.jit
def finish_split(
    tid,
    row,
    query_group_id: cutlass.Constexpr,
    k: cutlass.Constexpr,
    candidate_capacity: cutlass.Constexpr,
    barrier_id: cutlass.Constexpr,
    packed_candidates,
    candidate_counts,
    score_histogram,
    shared_counters,
    topk_indices,
    scratch,
    status,
    block,
    part,
    parts,
    failed,
    topk_indices_offset,
):
    """Merge split-CTA candidate lists and emit one exact repaired row.

    Args:
        tid: Thread index within the participating 128-thread warpgroup.
        row: Live query row receiving the selected indices.
        query_group_id: Compile-time slot of this query within the CTA's repair group.
        k: Number of output indices selected for the row.
        candidate_capacity: Shared candidate capacity per query slot.
        barrier_id: Named barrier used by all 128 threads, including both tie-emission
            passes.
        packed_candidates: Shared uint64 ``(score_key << 32) | row_local_index``
            candidates.
        candidate_counts: Shared count of valid candidates for each query slot.
        score_histogram: Shared int32 radix histograms, with 256 bins per query slot.
        shared_counters: Shared cursor and selected-bin scratch, with two values per
            query slot.
        topk_indices: Int32 output ``[M, k]`` receiving selected indices and -1 padding.
        scratch: Global int64 ``[repair_ctas, block_q, candidate_capacity + 1]`` split
            lists with a trailing count; unused when parts equals one.
        status: Global row metadata whose first column counts completed split CTAs.
        block: Zero-based repair CTA index selecting this CTA's scratch slot.
        part: Assigned KV partition index, retained in the interface but unused by the
            merge.
        parts: Number of CTAs contributing partitions to this query group.
        failed: Number of failed query groups, also the scratch stride between
            partitions.
        topk_indices_offset: Optional int32/int64 ``[M]`` offsets added only to valid
            output indices; None keeps row-local indices and padding remains -1.
    """
    if parts == 1:
        finish_shared(
            tid,
            row,
            query_group_id,
            k,
            candidate_capacity,
            barrier_id,
            packed_candidates,
            candidate_counts,
            score_histogram,
            shared_counters,
            topk_indices,
            topk_indices_offset,
        )
    else:
        count = candidate_counts[query_group_id]
        for j in cutlass.range(tid, count, 128):
            scratch[block, query_group_id, j] = cutlass.Int64(
                packed_candidates[query_group_id * candidate_capacity + j]
            )
        if tid == 0:
            scratch[block, query_group_id, candidate_capacity] = cutlass.Int64(count)
        # Publish every lane before the CTA signals completion.
        cute.arch.fence_acq_rel_gpu()
        cute.arch.barrier(barrier_id=barrier_id, number_of_threads=128)
        if tid == 0:
            shared_counters[query_group_id * 2] = cute.arch.atomic_add(
                (status.iterator + row * 2).llvm_ptr,
                cutlass.Int32(1),
                sem="acq_rel",
                scope="gpu",
            )
        cute.arch.barrier(barrier_id=barrier_id, number_of_threads=128)
        # The last CTA merges; other CTAs never wait for unscheduled work.
        if shared_counters[query_group_id * 2] == parts - 1:
            cute.arch.fence_acq_rel_gpu()
            rank = block % failed
            total = cutlass.Int32(0)
            for p in cutlass.range(parts):
                total += cutlass.Int32(
                    scratch[rank + p * failed, query_group_id, candidate_capacity]
                )
            # A short row may share a split repair group with a long row.
            # There is no kth bucket when total <= k; emit every candidate.
            pivot = cutlass.Uint32(0xFFFFFFFF)
            before = cutlass.Int32(0)
            if total > k:
                pivot = cutlass.Uint32(0)
                for level in cutlass.range_constexpr(4):
                    shift = cutlass.const_expr(24 - 8 * level)
                    score_histogram[query_group_id * 256 + tid] = 0
                    score_histogram[query_group_id * 256 + 128 + tid] = 0
                    cute.arch.barrier(barrier_id=barrier_id, number_of_threads=128)
                    for p in cutlass.range(parts):
                        src = rank + p * failed
                        n = cutlass.Int32(
                            scratch[src, query_group_id, candidate_capacity]
                        )
                        for base in cutlass.range(0, n, 128):
                            i = base + tid
                            key = cutlass.Uint32(0xFFFFFFFF)
                            if i < n:
                                key = cutlass.Uint32(
                                    cutlass.Uint64(scratch[src, query_group_id, i])
                                    >> 32
                                )
                            mask = cutlass.const_expr(
                                (0xFFFFFFFF << (shift + 8)) & 0xFFFFFFFF
                            )
                            bucket = cutlass.Int32(256)
                            if (i < n) & ((key & mask) == pivot):
                                bucket = cutlass.Int32(key >> shift & 255)
                            peers = cute.arch.match_sync(0xFFFFFFFF, bucket)
                            lane = cute.arch.lane_idx()
                            lower = (cutlass.Uint32(1) << lane) - 1
                            if (bucket < 256) & ((peers & lower) == 0):
                                _topk_red_add(
                                    score_histogram.iterator
                                    + query_group_id * 256
                                    + bucket,
                                    cutlass.Int32(cute.arch.popc(peers)),
                                )
                    cute.arch.barrier(barrier_id=barrier_id, number_of_threads=128)
                    _topk_prefix_pick_warp0(
                        tid,
                        k - before,
                        score_histogram,
                        query_group_id * 256,
                        shared_counters,
                        query_group_id * 2 + 1,
                    )
                    cute.arch.barrier(barrier_id=barrier_id, number_of_threads=128)
                    bucket = shared_counters[query_group_id * 2 + 1]
                    if bucket > 0:
                        before += score_histogram[query_group_id * 256 + bucket - 1]
                    pivot = pivot | (cutlass.Uint32(bucket) << shift)
                    cute.arch.barrier(barrier_id=barrier_id, number_of_threads=128)
            for i in cutlass.range(tid, k, 128):
                topk_indices[row, i] = -1
            if tid == 0:
                shared_counters[query_group_id * 2] = 0
            cute.arch.barrier(barrier_id=barrier_id, number_of_threads=128)
            for tie in cutlass.range_constexpr(2):
                for p in cutlass.range(parts):
                    src = rank + p * failed
                    n = cutlass.Int32(scratch[src, query_group_id, candidate_capacity])
                    for base in cutlass.range(0, n, 128):
                        i = base + tid
                        value = cutlass.Uint64(0xFFFFFFFFFFFFFFFF)
                        if i < n:
                            value = cutlass.Uint64(scratch[src, query_group_id, i])
                        key = cutlass.Uint32(value >> 32)
                        hit = (
                            key < pivot
                            if cutlass.const_expr(tie == 0)
                            else key == pivot
                        )
                        emit(
                            (i < n) & hit,
                            value,
                            row,
                            k,
                            shared_counters.iterator + query_group_id * 2,
                            topk_indices,
                            topk_indices_offset,
                        )
                cute.arch.barrier(barrier_id=barrier_id, number_of_threads=128)
