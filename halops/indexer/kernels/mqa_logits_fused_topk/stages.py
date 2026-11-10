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

import functools

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import testing

from .launch import KernelLaunch
from .repair import needs_repair, partition_counts, select_global


class EmptyPilot:
    """An empty direct sample has no cutoff: retain all keys for exact selection."""

    @cute.jit
    def __call__(self, threshold, stream: cuda.CUstream):
        """Launch threshold initialization for an empty direct sample.

        Args:
            threshold: Score-dtype ``[M, 1]`` output filled with negative infinity.
            stream: CUDA stream on which to enqueue the kernel.
        """
        self.kernel(threshold).launch(
            grid=(cute.ceil_div(threshold.shape[0], 128), 1, 1),
            block=(128, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(self, threshold):
        """Set every live row's cutoff to negative infinity.

        Args:
            threshold: Score-dtype ``[M, 1]`` output filled with negative infinity.
        """
        row = cute.arch.block_idx()[0] * 128 + cute.arch.thread_idx()[0]
        if row < threshold.shape[0]:
            threshold[row, 0] = cutlass.Float32(-float("inf")).to(
                threshold.element_type
            )


class SuccessTopK:
    def __init__(self, k, cap, bq, partitions=0):
        """Configure exact selection and repair-queue grouping.

        Args:
            k: Number of output indices per query row.
            cap: Candidate arena entries per row, including headers in partitioned mode.
            bq: Number of query rows in a repair group.
            partitions: Four for warp-partitioned candidates, or zero for a compact row
                arena.
        """
        self.k, self.cap, self.bq = (k, cap, bq)
        self.partitions = partitions

    @cute.jit
    def __call__(
        self,
        cand,  # [M, cap] i64 candidates the filter wrote
        counts,  # [filled, marker] per row, then the failed-group queue
        offsets,  # None, or [M] added to every emitted index
        out,  # [M, k] i32 output; its row count is also the grid
        ks,  # row bounds: ke-ks is the window the completeness test needs
        ke,
        stream: cuda.CUstream,
    ):
        """Launch selection of complete rows and publish the failed-group queue.

        Args:
            cand: Int64 ``[M, cap]`` packed score-key/index candidates written by the
                filter.
            counts: Int32 ``[filled, marker]`` row metadata followed by the failed-group
                queue.
            offsets: Optional int32/int64 ``[M]`` offsets added only to valid output
                indices; None keeps row-local indices and padding remains -1.
            out: Int32 ``[M, k]`` output; complete rows receive indices and -1 padding.
            ks: Nonnegative int32 ``[M]`` inclusive key starts in full-KV coordinates.
            ke: Int32 ``[M]`` exclusive absolute key ends in full-KV coordinates,
                between each row's start and the full KV row count. The window length
                determines whether short rows are complete.
            stream: CUDA stream on which to enqueue the kernel.
        """
        self.kernel(cand, counts, offsets, out, ks, ke).launch(
            grid=(out.shape[0], 1, 1), block=(256, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(self, cand, counts, offsets, out, ks, ke):
        """Select complete rows and queue query groups requiring exact repair.

        Args:
            cand: Int64 ``[M, cap]`` packed score-key/index candidates written by the
                filter.
            counts: Int32 ``[filled, marker]`` row metadata followed by the failed-group
                queue.
            offsets: Optional int32/int64 ``[M]`` offsets added only to valid output
                indices; None keeps row-local indices and padding remains -1.
            out: Int32 ``[M, k]`` output; complete rows receive indices and -1 padding.
            ks: Nonnegative int32 ``[M]`` inclusive key starts in full-KV coordinates.
            ke: Int32 ``[M]`` exclusive absolute key ends in full-KV coordinates,
                between each row's start and the full KV row count. The window length
                determines whether short rows are complete.

        Notes:
            The shared ``needs_repair`` marker contract keeps complete short rows out of
            the repair queue.
        """
        row = cute.arch.block_idx()[0]
        tid = cute.arch.thread_idx()[0]
        smem = cutlass.utils.SmemAllocator()
        hist = smem.allocate_tensor(
            cutlass.Int32, cute.make_layout((256,)), byte_alignment=128
        )
        scalars = smem.allocate_tensor(
            cutlass.Int32, cute.make_layout((2,)), byte_alignment=128
        )
        marker = cutlass.Int32(0)
        if cutlass.const_expr(self.partitions > 0):
            filled, marker = partition_counts(row, self.cap, self.partitions, cand)
            if tid == 0:
                counts[row, 0] = filled
                counts[row, 1] = marker
            cute.arch.barrier(barrier_id=1, number_of_threads=256)
        else:
            marker = counts[row, 1]
        # A complete short row is already exact even when marker < k.  The
        # shared predicate also catches threshold truncation and overflow.
        if ~needs_repair(marker, ke[row] - ks[row], self.k, self.cap):
            select_global(
                row,
                tid,
                self.k,
                self.cap,
                cand,
                counts,
                out,
                hist,
                scalars,
                offsets,
                self.partitions,
            )
        elif tid == 0:
            counts[row, 0] = 0
        # Only immutable filter flags are read while other CTAs write outputs.
        if row == 0:
            m = out.shape[0]
            if tid == 0:
                counts[m, 0] = 0
            cute.arch.barrier(barrier_id=1, number_of_threads=256)
            for group in cutlass.range(tid, (m + self.bq - 1) // self.bq, 256):
                hit = cutlass.Boolean(False)
                for offset in cutlass.range_constexpr(self.bq):
                    r = group * self.bq + offset
                    if r < m:
                        mark = cutlass.Int32(0)
                        if cutlass.const_expr(self.partitions > 0):
                            _, mark = partition_counts(
                                r, self.cap, self.partitions, cand
                            )
                        else:
                            mark = counts[r, 1]
                        hit |= needs_repair(mark, ke[r] - ks[r], self.k, self.cap)
                if hit:
                    pos = cute.arch.atomic_add(
                        (counts.iterator + m * 2).llvm_ptr,
                        cutlass.Int32(1),
                        sem="relaxed",
                        scope="gpu",
                    )
                    counts[m + 1 + pos, 0] = group


class SampleKV:
    """Gather the pilot's KV sample and its per-token scales.

    ``take`` consecutive rows out of every ``group``: slot ``p`` holds
    ``kv[(p // take) * group + (p % take)]`` (zero past the end), which is the
    same slot-to-key map ``QueryBlockBounds.coordinate`` inverts. The default (16, 1) is the
    uniform 1/16 sample. This stage exists only because a lone 4-byte scale is
    below TMA's 16-byte minimum; for ``take >= 4`` ``pipeline.prepare`` skips
    it and the pilot reads the sample in place.
    """

    def __init__(
        self,
        row_bytes,  # bytes per KV row; also the scale's slot in the flat index
        group=16,  # sample `take` rows out of every `group`
        take=1,
    ):
        """Configure dense gathering of sampled KV rows and their scales.

        Args:
            row_bytes: Packed byte width of one KV row.
            group: Number of source KV rows in one sampling group.
            take: Number of consecutive rows sampled from the start of each group.
        """
        self.row_bytes = row_bytes
        self.group = group
        self.take = take

    @cute.jit
    def __call__(
        self,
        kv,  # full KV, read strided
        scales,  # its per-token scales, same stride
        kv_out,  # dense sample the pilot reads; its rows size the grid
        scales_out,
        stream: cuda.CUstream,
    ):
        """Launch dense gathering of sampled KV rows and matching scales.

        Args:
            kv: Uint8 KV byte view ``[N, 128]`` for FP8/MXFP8 or ``[N, 64]`` for MXFP4.
            scales: Per-key ``[N]`` scales: FP32 for FP8 or packed int32 UE8M0 for
                MXFP4/MXFP8.
            kv_out: Dense sampled KV byte buffer, zero-filled where source rows are out
                of bounds.
            scales_out: Per-sample scale output in the input scale dtype, with
                zero-filled padding.
            stream: CUDA stream on which to enqueue the kernel.
        """
        work = kv_out.shape[0] * (self.row_bytes + 1)
        self.kernel(kv, scales, kv_out, scales_out).launch(
            grid=((work + 255) // 256, 1, 1), block=(256, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(self, kv, scales, kv_out, scales_out):
        """Gather each sampled row from its source group and zero-fill padding.

        Args:
            kv: Uint8 KV byte view ``[N, 128]`` for FP8/MXFP8 or ``[N, 64]`` for MXFP4.
            scales: Per-key ``[N]`` scales: FP32 for FP8 or packed int32 UE8M0 for
                MXFP4/MXFP8.
            kv_out: Dense sampled KV byte buffer, zero-filled where source rows are out
                of bounds.
            scales_out: Per-sample scale output in the input scale dtype, with
                zero-filled padding.
        """
        i = cute.arch.block_idx()[0] * 256 + cute.arch.thread_idx()[0]
        width = self.row_bytes + 1
        row = i // width
        col = i - row * width
        src = (row // self.take) * self.group + (row % self.take)
        if row < kv_out.shape[0]:
            if col < self.row_bytes:
                byte = cutlass.Uint8(0)
                if src < kv.shape[0]:
                    byte = kv[src, col]
                kv_out[row, col] = byte
            else:
                word = scales_out.element_type(0)
                if src < scales.shape[0]:
                    word = scales[src]
                scales_out[row] = word


_ROWS_PER_CTA = 8  # one warp per row, 256 threads


class WindowPrep:
    """Append the newest W keys after top-k and update the live selection bounds.

    Window columns contain local indices in ascending order, padded with -1.
    ``sync_ks`` copies starts when the caller supplies a separate bounds buffer.
    """

    def __init__(self, k, w, sync_ks):
        """Configure appended window columns and live bound refresh.

        Args:
            k: Number of top-k columns preceding the window output.
            w: Number of newest key indices to append after top-k.
            sync_ks: Whether to copy live starts into a separate start-bound buffer.
        """
        self.k, self.w, self.sync_ks = k, w, sync_ks

    @cute.jit
    def __call__(
        self,
        ks,
        ke,
        ks_buf,
        ke_buf,
        out,
        num_rows: cutlass.Int32,
        offsets,
        stream: cuda.CUstream,
    ):
        """Launch window-index emission and refresh the top-k bounds.

        Args:
            ks: Int32 ``[>=M]`` live inclusive key starts.
            ke: Int32 ``[>=M]`` live exclusive key ends before window exclusion.
            ks_buf: Int32 start bounds read by top-k; may alias ``ks`` or be refreshed
                from it.
            ke_buf: Separate int32 top-k end bounds updated to exclude the appended
                window.
            out: Int32 ``[M, k + w]`` storage whose final w columns receive window
                indices.
            num_rows: Number M of live query rows.
            offsets: Optional int32/int64 ``[M]`` offsets added only to valid output
                indices; None keeps row-local indices and padding remains -1.
            stream: CUDA stream on which to enqueue the kernel.
        """
        grid = (num_rows + _ROWS_PER_CTA - 1) // _ROWS_PER_CTA
        self.kernel(ks, ke, ks_buf, ke_buf, out, num_rows, offsets).launch(
            grid=(grid, 1, 1), block=(32 * _ROWS_PER_CTA, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(self, ks, ke, ks_buf, ke_buf, out, num_rows: cutlass.Int32, offsets):
        """Append ordered window indices and refresh top-k bounds from live inputs.

        Args:
            ks: Int32 ``[>=M]`` live inclusive key starts.
            ke: Int32 ``[>=M]`` live exclusive key ends before window exclusion.
            ks_buf: Int32 start bounds read by top-k; may alias ``ks`` or be refreshed
                from it.
            ke_buf: Separate int32 top-k end bounds updated to exclude the appended
                window.
            out: Int32 ``[M, k + w]`` storage whose final w columns receive window
                indices.
            num_rows: Number M of live query rows.
            offsets: Optional int32/int64 ``[M]`` offsets added only to valid output
                indices; None keeps row-local indices and padding remains -1.
        """
        tid = cute.arch.thread_idx()[0]
        lane = tid % 32
        row = cute.arch.block_idx()[0] * _ROWS_PER_CTA + tid // 32
        if row < num_rows:
            start = ks[row]
            length = cutlass.max(ke[row] - start, cutlass.Int32(0))
            first = cutlass.max(length - self.w, cutlass.Int32(0))
            if lane == 0:
                ke_buf[row] = start + first
                if cutlass.const_expr(self.sync_ks):
                    ks_buf[row] = start
            valid = length - first
            for j in cutlass.range(lane, self.w, 32):
                value = cutlass.Int32(-1)
                if j < valid:
                    value = first + j
                    if cutlass.const_expr(offsets is not None):
                        value += cutlass.Int32(offsets[row])
                out[row, self.k + j] = value


def _vec():
    """Build an int32 compilation tensor with a symbolic one-dimensional extent."""
    return cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (cute.sym_int(),), stride_order=(0,)
    )


@functools.cache
def _compile_prep(k, w, sync_ks, row_stride, offset_dtype):
    """Compile and cache window preparation for the output layout and offset type.

    Args:
        k: Number of top-k output columns.
        w: Number of appended window columns.
        sync_ks: Whether to copy the live starts into a separate bounds buffer.
        row_stride: Output row stride in int32 elements.
        offset_dtype: CuTe Int32/Int64 offset type, or None when offsets are absent.
    """
    return cute.compile(
        WindowPrep(k, w, sync_ks),
        _vec(),
        _vec(),
        _vec(),
        _vec(),
        cute.runtime.make_fake_tensor(
            cutlass.Int32, (cute.sym_int(), k + w), stride=(row_stride, 1)
        ),
        cutlass.Int32(1),
        (
            cute.runtime.make_fake_compact_tensor(
                offset_dtype, (cute.sym_int(),), stride_order=(0,)
            )
            if offset_dtype is not None
            else None
        ),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def topk_end(ks, ke, sliding_window):
    """Return the top-k prefix end ``ks + max(ke - ks - W, 0)``.

    Args:
        ks: Nonnegative int32 ``[M]`` inclusive key starts in full-KV coordinates.
        ke: Int32 ``[M]`` exclusive absolute key ends in full-KV coordinates, between
            each row's start and the full KV row count.
        sliding_window: Number W of newest keys excluded from top-k selection.
    """
    return torch.maximum(ke - sliding_window, ks)


def prep_launch(
    ks,
    ke,
    ks_buf,
    ke_buf,
    out,
    num_rows,
    top_k,
    sliding_window,
    *,
    topk_indices_offset=None,
):
    """Prepare the launch that appends window indices and updates top-k bounds.

    Args:
        ks: Int32 ``[>=M]`` live inclusive key starts.
        ke: Int32 ``[>=M]`` live exclusive key ends before window exclusion.
        ks_buf: Int32 start bounds read by top-k; may alias ``ks`` or be refreshed from
            it.
        ke_buf: Separate int32 top-k end bounds updated to exclude the appended window.
        out: Int32 ``[M, top_k + sliding_window]`` storage with contiguous columns.
        num_rows: Number M of live query rows.
        top_k: Number of top-k columns preceding the appended window.
        sliding_window: Number of newest key indices to append, padding missing keys
            with -1.
        topk_indices_offset: Optional int32/int64 ``[M]`` offsets added only to valid
            output indices; None keeps row-local indices and padding remains -1.
    """
    for name, t in (("ks", ks), ("ke", ke), ("ks_buf", ks_buf), ("ke_buf", ke_buf)):
        if t.dtype != torch.int32 or t.ndim != 1 or t.shape[0] < num_rows:
            raise ValueError(f"{name} must be int32 [>= {num_rows}]")
    if ke_buf is ke or ke_buf.data_ptr() == ke.data_ptr():
        raise ValueError("ke_buf must not alias the caller's ke")
    if (
        out.dtype != torch.int32
        or out.shape != (num_rows, top_k + sliding_window)
        or out.stride(1) != 1
        or out.stride(0) < top_k + sliding_window
    ):
        raise ValueError(
            "out must be int32 [M, K + W] with contiguous, non-overlapping rows"
        )
    sync_ks = ks_buf.data_ptr() != ks.data_ptr()
    offset_dtype = (
        None
        if topk_indices_offset is None
        else (
            cutlass.Int32 if topk_indices_offset.dtype == torch.int32 else cutlass.Int64
        )
    )
    fn = _compile_prep(top_k, sliding_window, sync_ks, out.stride(0), offset_dtype)
    return KernelLaunch(
        fn,
        testing.JitArguments(
            ks, ke, ks_buf, ke_buf, out, num_rows, topk_indices_offset
        ),
    )
