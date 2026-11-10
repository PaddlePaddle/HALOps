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
import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass import Float4E2M1FN, Float8E8M0FNU, testing
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from . import topk
from .config import (
    BLOCK_KV_ROWS,
    SPLIT_KV_ROWS,
    SUPPORTED_HEAD_DIM,
    LaunchArg,
    candidate_capacity,
    round_up,
    select_config,
    validate_topk,
)
from .epilogue import score_epilogue
from .geometry import MqaMmaGeometry, QueryBlockBounds
from .kernel_utils import (
    _apply_utccp_chunk_layout,
    _subview,
    utccp_required_smem_warp_transpose,
)
from .launch import (
    KernelLaunch,
    make_bounds_fakes,
    make_input_fakes,
    prepare_input_views,
)
from .repair import finish_split, needs_repair, next_group, schedule

_TK_BAR_WG0 = 9
_TK_BAR_FIN = 13


class MqaProducerKernel(MqaMmaGeometry):
    """Full KV producer kernel used by filter and exact repair launches."""

    def __init__(
        self,
        block_kv: int = BLOCK_KV_ROWS,
        num_heads: int = 64,
        head_dim: int = SUPPORTED_HEAD_DIM,
        num_sms: int = 148,
        epi_dtype=cutlass.Float32,
        remove_online_sf_transpose: bool = False,
        top_k: int = 1024,
        qkv_dtype: str = "mxfp4",
        *,
        config,
        repair=False,
        materialize_scores=False,
    ):
        """Configure filter or repair geometry and candidate storage.

        Args:
            block_kv: Number of key rows in one MMA tile, fixed at 128.
            num_heads: Logical query head count: 8, 16, 32, or 64.
            head_dim: Unpacked dimension of each query/key head, fixed at 128.
            num_sms: Number of SMs available to the persistent launch grid.
            epi_dtype: Score arithmetic type, ``cutlass.Float32`` or
                ``cutlass.BFloat16``.
            remove_online_sf_transpose: Whether MXFP4/MXFP8 K scales use the
                pre-transposed 128-row chunk layout.
            top_k: Number of indices selected per query, from 128 through 2048.
            qkv_dtype: Packed input format, ``"fp8"``, ``"mxfp4"`` or ``"mxfp8"``.
            config: Compile-time query tiling, histogram, accumulator and register
                policy.
            repair: Whether to rescan failed rows and select exact top-k indices.
            materialize_scores: Whether to emit full diagnostic scores; incompatible
                with repair.
        """
        super().__init__(
            block_kv,
            num_heads,
            head_dim,
            num_sms,
            epi_dtype,
            remove_online_sf_transpose,
            top_k,
            qkv_dtype,
            config=config,
        )
        self.stream_period = 4
        self.stream_cap = candidate_capacity(top_k, qkv_dtype)
        # Filter may emit diagnostic scores; repair always emits exact indices.
        assert not (materialize_scores and repair)
        self.materialize_scores = materialize_scores
        self.repair = repair
        # Wide query tiles share math warpgroups and write directly to the
        # global arena. Private warp partitions keep the hot loop atomic-free;
        # the partition headers let SuccessTopK detect every overflow.
        self.direct_candidates = self.queries_per_group > 1 and not materialize_scores
        assert not (self.direct_candidates and repair)
        self.bounds = QueryBlockBounds(
            self.block_q,
            self.block_kv,
            ordered=config.ordered_bounds,
        )

    @cute.jit
    def __call__(
        self,
        # Filter and repair share the candidate arena and failed-group queue.
        kv: cute.Tensor,  # A operand bytes
        kv_sf: cute.Tensor,  # per-token K scales
        b: cute.Tensor,  # B operand bytes (Q)
        q_sf: cute.Tensor,  # Q block scales (MXFP4/MXFP8)
        weights: cute.Tensor,  # head weights
        logits: cute.Tensor,  # [M,1]: pilot writes tau here, filter reads it
        topk_indices: cute.Tensor,  # final output; repair writes its rows
        topk_cand: cute.Tensor,  # [M,cap] i64 candidate arena (key<<32|index)
        topk_cand_hi: cute.Tensor,  # repair-only split-merge scratch
        topk_cand_cnt: cute.Tensor,  # [filled, marker] + the failed-group queue
        ks: cute.Tensor,  # ks; every read is clamped to M-1
        ke: cute.Tensor,  # ke
        num_q_tokens: cutlass.Int32,  # M: row masks and the task loop bound
        num_kv_tokens: cutlass.Int32,  # real KV row extent
        topk_indices_offset,  # None, or [M] added at the stores
        stream: cuda.CUstream,
    ):
        """Build TMA descriptors and launch filtering, repair or score diagnostics.

        Args:
            kv: Uint8 KV byte view ``[N, 128]`` for FP8/MXFP8 or ``[N, 64]`` for MXFP4.
            kv_sf: Per-key ``[N]`` scales: FP32 for FP8 or packed int32 UE8M0 for
                MXFP4/MXFP8.
            b: Flattened Q byte view ``[M * H, packed_width, 1]`` used as MMA operand B.
            q_sf: Flattened int32 ``[M * H]`` Q scales; unused for FP8.
            weights: Flattened ``[M * H]`` head weights in the score dtype.
            logits: Pilot cutoff input ``[M, 1]``, or full score output in diagnostic
                mode.
            topk_indices: Int32 ``[M, top_k]`` repair output, or a one-row placeholder
                in diagnostic mode.
            topk_cand: Global int64 candidate arena containing packed score keys and
                row-local indices.
            topk_cand_hi: Global int64 split-repair scratch, or a uint8 placeholder for
                filter launches.
            topk_cand_cnt: Int32 row counts and repair markers followed by the
                failed-group queue.
            ks: Nonnegative int32 ``[M]`` inclusive key starts in full-KV coordinates.
            ke: Int32 ``[M]`` exclusive absolute key ends in full-KV coordinates,
                between each row's start and the full KV row count.
            num_q_tokens: Number M of live query rows.
            num_kv_tokens: Number N of real, unpadded KV rows.
            topk_indices_offset: Optional int32/int64 ``[M]`` offsets added only to
                valid output indices; None keeps row-local indices and padding remains
                -1.
            stream: CUDA stream on which to enqueue the kernel.
        """
        head_dim_bytes = self.head_dim // 2 if self.is_fp4 else self.head_dim
        kv_data = cute.recast_tensor(
            kv, Float4E2M1FN if self.is_fp4 else cutlass.Float8E4M3FN
        )
        b = cute.recast_tensor(b, Float4E2M1FN if self.is_fp4 else cutlass.Float8E4M3FN)
        # Declare the real row extent instead of a tiled (128, blocks) shape, so
        # the TMA descriptor bound-checks rows itself: a tile that runs past
        # num_kv_tokens is zero-filled rather than read. That is what lets the
        # caller pass unpadded KV -- zero rows score 0 and the per-row
        # [ks, ke) mask drops them, so they never reach the candidate pool.
        kv_layout = cute.make_layout(
            (num_kv_tokens, self.head_dim), stride=(self.head_dim, 1)
        )
        a = cute.make_tensor(kv_data.iterator, kv_layout)
        kv_sf_layout = cute.make_layout((num_kv_tokens,), stride=(1,))
        kv_sf = cute.make_tensor(kv_sf.iterator, kv_sf_layout)
        a_dtype = a.element_type
        b_dtype = b.element_type
        a_major = cutlass.utils.LayoutEnum.from_tensor(a).mma_major_mode()
        b_major = cutlass.utils.LayoutEnum.ROW_MAJOR.mma_major_mode()
        tiled_mma = self._setup_mma(a_dtype, b_dtype, a_major, b_major)
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp()
        self.a_tma_smem_layout = sm100_utils.make_smem_layout(
            cute.nvgpu.OperandMajorMode.K,
            (self.block_kv, self.head_dim),
            a_dtype,
            self.num_kv_stages,
        )
        a_smem_per_stage = cute.slice_(self.a_tma_smem_layout, (None, None, 0))
        tma_atom_a, tma_tensor_a = cpasync.make_tiled_tma_atom(
            tma_load_op, a, a_smem_per_stage, (self.block_kv, self.head_dim)
        )
        self.kv_sf_smem_layout_staged = cute.make_layout(
            (
                self.block_kv,
                self.num_sc_slots if not self.is_mx_sf else self.num_kv_stages,
            )
        )
        kv_sf_smem_per_block = cute.make_layout((self.block_kv,))
        tma_atom_kv_sf, tma_tensor_kv_sf = cpasync.make_tiled_tma_atom(
            tma_load_op, kv_sf, kv_sf_smem_per_block, (self.block_kv,)
        )
        (
            tma_atom_b,
            tma_tensor_b,
            tma_atom_w,
            tma_tensor_w,
            tma_atom_q_sf,
            tma_tensor_q_sf,
        ) = self._setup_q_tma(b, q_sf, weights, tiled_mma)
        # Both input formats transfer one 4-byte K scale per token.
        self.num_kv_sf_tma_bytes = self.block_kv * (head_dim_bytes + 4)
        num_q_blocks = (num_q_tokens + self.block_q - 1) // self.block_q
        num_ctas = topk_cand_hi.shape[0] if self.repair else self.num_sms

        @cute.struct
        class SharedStorage:
            kv_mbar_0: cute.struct.MemRange[cutlass.Int64, self.num_kv_stages * 2]
            kv_mbar_1: cute.struct.MemRange[cutlass.Int64, self.num_kv_stages * 2]
            q_mbar: cute.struct.MemRange[cutlass.Int64, self.num_q_stages * 2]
            umma_mbar_0: cute.struct.MemRange[cutlass.Int64, self.num_umma_stages * 2]
            umma_mbar_1: cute.struct.MemRange[cutlass.Int64, self.num_umma_stages * 2]
            sfb_war_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            tmem_holding_buf: cutlass.Int32

        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_w,
            tma_tensor_w,
            tma_atom_kv_sf,
            tma_tensor_kv_sf,
            tma_atom_q_sf,
            tma_tensor_q_sf,
            logits,
            topk_indices,
            topk_cand,
            topk_cand_hi,
            topk_cand_cnt,
            ks,
            ke,
            num_q_tokens,
            num_q_blocks,
            num_kv_tokens,
            topk_indices_offset,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.a_tma_smem_layout,
            self.b_smem_layout_staged,
            self.w_smem_layout_staged,
            self.kv_sf_smem_layout_staged,
            self.q_sf_smem_layout_staged,
            self.q_sf_tma_smem_layout_staged,
            self.tmem_sfa_layout,
            self.tmem_sfb_layout,
            self.sfa_chunk_smem_layout,
            self.sfb_chunk_smem_layout,
            self.copy_atom_s2t,
            self.epi_tile,
            SharedStorage,
        ).launch(
            grid=(self.hist_splits, 1, num_ctas),
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            min_blocks_per_mp=1,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_w: cute.CopyAtom,
        mW_tma: cute.Tensor,
        tma_atom_kv_sf: cute.CopyAtom,
        mSF_KV_tma: cute.Tensor,
        tma_atom_q_sf: cute.CopyAtom,
        mSF_Q_tma: cute.Tensor,
        mLogits: cute.Tensor,
        mTopkIdx: cute.Tensor,
        mTopkCand: cute.Tensor,
        mTopkCandHi: cute.Tensor,
        mTopkCandCnt: cute.Tensor,
        mCuSeqLenKStart: cute.Tensor,
        mCuSeqLenKEnd: cute.Tensor,
        num_q_tokens: cutlass.Int32,
        num_q_blocks: cutlass.Int32,
        num_kv_tokens: cutlass.Int32,
        topk_indices_offset,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        a_tma_smem_layout: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        w_smem_layout_staged: cute.Layout,
        kv_sf_smem_layout_staged: cute.Layout,
        q_sf_smem_layout_staged: cute.Layout,
        q_sf_tma_smem_layout_staged: cute.Layout,
        tmem_sfa_layout: cute.Layout,
        tmem_sfb_layout: cute.Layout,
        sfa_chunk_smem_layout: cute.Layout,
        sfb_chunk_smem_layout: cute.Layout,
        copy_atom_s2t: cute.CopyAtom,
        epi_tile: cute.Tile,
        SharedStorage: cutlass.Constexpr,
    ):
        """Score legal keys and filter candidates, repair failed rows or emit
        diagnostics.

        Args:
            tiled_mma: Tiled FP8 or block-scaled MXFP4 MMA operation.
            tma_atom_a: TMA copy atom loading key operand A into shared memory.
            mA_mkl: Recast key operand with the layout used by its TMA descriptor.
            tma_atom_b: TMA copy atom loading query operand B into shared memory.
            mB_nkl: Recast query operand with the layout used by its TMA descriptor.
            tma_atom_w: TMA copy atom loading head weights.
            mW_tma: Global head-weight tensor addressed by the weight TMA descriptor.
            tma_atom_kv_sf: TMA copy atom loading per-key scales.
            mSF_KV_tma: Global FP32 or packed UE8M0 key-scale tensor.
            tma_atom_q_sf: TMA copy atom loading MXFP4/MXFP8 query scales; unused for
                FP8.
            mSF_Q_tma: Global packed UE8M0 query-scale tensor; unused for FP8.
            mLogits: Global pilot cutoff input ``[M, 1]``, or full score output in
                diagnostic mode.
            mTopkIdx: Int32 ``[M, top_k]`` repair output, or a one-row placeholder in
                diagnostic mode.
            mTopkCand: Global int64 candidate arena containing packed score keys and
                row-local indices.
            mTopkCandHi: Global int64 split-repair scratch, or a uint8 placeholder for
                filter launches.
            mTopkCandCnt: Int32 row counts and repair markers followed by the
                failed-group queue.
            mCuSeqLenKStart: Nonnegative int32 ``[M]`` inclusive key starts in full-KV
                coordinates.
            mCuSeqLenKEnd: Int32 ``[M]`` exclusive absolute key ends in full-KV
                coordinates, between each row's start and the full KV row count.
            num_q_tokens: Number M of live query rows.
            num_q_blocks: Number of query tiles, including a possible partial final
                tile.
            num_kv_tokens: Number N of real, unpadded KV rows.
            topk_indices_offset: Optional int32/int64 ``[M]`` offsets added only to
                valid output indices; None keeps row-local indices and padding remains
                -1.
            cluster_layout_vmnk: CTA cluster layout in MMA partition coordinates.
            a_smem_layout_staged: Staged shared-memory layout for MMA key fragments.
            a_tma_smem_layout: Staged shared-memory layout used by key TMA loads.
            b_smem_layout_staged: Staged shared-memory layout for query operands.
            w_smem_layout_staged: Staged head-weight layout including aligned stage
                strides.
            kv_sf_smem_layout_staged: Staged key-scale layout, including the FP8 scale
                ring.
            q_sf_smem_layout_staged: Padded shared-memory allocation layout for query
                scales.
            q_sf_tma_smem_layout_staged: Logical query-scale TMA layout excluding unused
                padded columns.
            tmem_sfa_layout: TMEM layout for MXFP4 key block scales.
            tmem_sfb_layout: TMEM layout for MXFP4 query block scales.
            sfa_chunk_smem_layout: Shared-memory chunk layout for copying key scales to
                TMEM.
            sfb_chunk_smem_layout: Shared-memory chunk layout for copying query scales
                to TMEM.
            copy_atom_s2t: Shared-memory-to-TMEM copy atom for MXFP4 block scales.
            epi_tile: M/N tile shape used to partition score accumulators.
            SharedStorage: Compile-time shared-storage struct for pipeline barriers and
                the TMEM handle.
        """
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2
        bidx, bidy, bidz = cute.arch.block_idx()
        grid_size = mTopkCandHi.shape[0] if self.repair else self.num_sms
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        tidx, _, _ = cute.arch.thread_idx()
        warpgroup_idx = warp_idx // 4
        is_math_warp = warp_idx < self.num_math_warps
        is_tma_warp_0 = warp_idx == self.num_math_warps + 0
        is_tma_warp_1 = warp_idx == self.num_math_warps + 1
        is_tma_warp = is_tma_warp_0 | is_tma_warp_1
        is_umma_warp_0 = warp_idx == self.num_math_warps + 2
        is_umma_warp_1 = warp_idx == self.num_math_warps + 3
        NUM_MATH_WG = 2
        block_kv_val = self.block_kv
        block_q_val = self.block_q
        repair_part = bidx
        repair_parts = self.hist_splits
        failed_groups = cutlass.Int32(0)
        sm_idx = bidz

        # Math-group IDs are static for shared-memory slots and named barriers.
        # The helper lives above every staged region: CuTe-DSL resolves names
        # used inside staged control flow only when they are bound before the
        # region, and it does not support closure captures, so every staged
        # input arrives as an explicit argument at the call site.
        def math_warpgroup(
            _TK_WG,
            has_work,
            prev_q_block_idx,
            next_q_block_idx,
            next_num_kv,
            sc_slot,
            q_block_idx,
            num_kv,
            kv_idx,
            self,
            NUM_MATH_WG,
            bidz,
            block_kv_val,
            block_q_val,
            cC,
            copy_atom_t2r,
            epi_sub_mn,
            failed_groups,
            grid_size,
            local_tidx,
            mCuSeqLenKEnd,
            mCuSeqLenKStart,
            mLogits,
            mTopkCand,
            mTopkCandCnt,
            mTopkCandHi,
            mTopkIdx,
            num_epi_subtiles,
            num_heads,
            num_q_blocks,
            num_q_tokens,
            q_cons_state,
            q_pipeline,
            repair_part,
            repair_parts,
            sSF_KV_0,
            sSF_KV_1,
            sW,
            s_score_hist,
            s_stream_counts,
            s_stream_entries,
            s_stream_scalars,
            tCtAcc_base_0,
            tCtAcc_base_1,
            topk_indices_offset,
            umma_cons_state_0,
            umma_cons_state_1,
            umma_pipeline_0,
            umma_pipeline_1,
        ):
            tAcc_ref = tCtAcc_base_0[None, None, None, 0][(None, None), 0, 0]
            tAcc_ref_epi = cute.flat_divide(tAcc_ref, epi_sub_mn)
            tiled_copy_ref = tcgen05.make_tmem_copy(
                copy_atom_t2r, tAcc_ref_epi[None, None, 0, 0]
            )
            thr_copy_ref = tiled_copy_ref.get_slice(local_tidx)
            tTR_cC = thr_copy_ref.partition_D(cC)
            m_coord = tTR_cC[0][0]
            tTR_rAcc = cute.make_fragment_like(tTR_cC, self.acc_dtype)
            MAX_NUM_W_IN_REG = (
                40 if block_q_val == 4 else 56 if block_q_val == 3 else 64
            )
            if cutlass.const_expr(self.topk_w_in_reg > 0):
                MAX_NUM_W_IN_REG = self.topk_w_in_reg
            NUM_W_IN_REG = min(MAX_NUM_W_IN_REG, num_heads)
            w_cache = cute.make_rmem_tensor(
                (NUM_W_IN_REG * block_q_val,), self.epi.dtype
            )
            result_arr = cute.make_rmem_tensor((block_q_val,), self.epi.dtype)
            tau_cache = cute.make_rmem_tensor((block_q_val,), cutlass.Float32)
            warp_counts = cute.make_rmem_tensor((block_q_val,), cutlass.Int32)
            kv_start_cache = [cutlass.Int32(0) for _ in range(block_q_val)]
            kv_end_cache = [cutlass.Int32(0) for _ in range(block_q_val)]
            q_stage_local = cutlass.Int32(0)
            kv_lo_cur = cutlass.Int32(0)
            while has_work:
                q_block_idx = next_q_block_idx
                num_kv = next_num_kv
                if q_block_idx != prev_q_block_idx:
                    if prev_q_block_idx < num_q_blocks:
                        q_pipeline.consumer_release(q_cons_state)
                        q_cons_state.advance()
                    q_pipeline.consumer_wait(q_cons_state)
                    q_stage_local = q_cons_state.index
                    q_base = q_block_idx * block_q_val
                    for t_i in cutlass.range_constexpr(block_q_val):
                        for w_j in cutlass.range_constexpr(NUM_W_IN_REG):
                            w_cache[t_i * NUM_W_IN_REG + w_j] = sW[
                                t_i * num_heads + w_j, q_stage_local
                            ]
                        token_idx = min(q_base + t_i, num_q_tokens - 1)
                        kv_start_cache[t_i] = mCuSeqLenKStart[token_idx]
                        kv_end_cache[t_i] = mCuSeqLenKEnd[token_idx]
                        if cutlass.const_expr(self.direct_candidates):
                            tau_cache[t_i] = cutlass.Float32(mLogits[token_idx, 0])
                    kv_lo_cur = self.bounds.tile_start(
                        mCuSeqLenKStart,
                        mCuSeqLenKEnd,
                        q_block_idx,
                        num_q_tokens,
                        repair_part,
                        repair_parts,
                        0,
                    )
                prev_q_block_idx = q_block_idx
                if cutlass.const_expr(self.direct_candidates):
                    for query in cutlass.range_constexpr(
                        _TK_WG * self.queries_per_group,
                        (_TK_WG + 1) * self.queries_per_group,
                    ):
                        warp_counts[query] = 0
                elif cutlass.const_expr(self.repair):
                    if local_tidx == 0:
                        if cutlass.const_expr(not self.materialize_scores):
                            s_stream_counts[_TK_WG] = 0
                elif local_tidx < 4:
                    if cutlass.const_expr(not self.materialize_scores):
                        s_stream_counts[_TK_WG * 4 + local_tidx] = 0
                if cutlass.const_expr(not self.materialize_scores):
                    stream_warp_count = cutlass.Int32(0)
                    stream_tau = (
                        cutlass.Float32(-float("inf"))
                        if cutlass.const_expr(self.repair)
                        else cutlass.Float32(
                            mLogits[
                                min(
                                    q_block_idx * block_q_val + _TK_WG,
                                    num_q_tokens - 1,
                                ),
                                0,
                            ]
                        )
                    )
                    stream_status_row = min(
                        q_block_idx * block_q_val + _TK_WG,
                        num_q_tokens - 1,
                    )
                    if cutlass.const_expr(self.repair):
                        stream_repair = needs_repair(
                            mTopkCandCnt[stream_status_row, 1],
                            kv_end_cache[_TK_WG] - kv_start_cache[_TK_WG],
                            self.top_k,
                            self.stream_cap,
                        )
                    else:
                        stream_repair = cutlass.Boolean(True)
                    stream_tiles = cutlass.Int32(0)
                    cute.arch.barrier(
                        barrier_id=_TK_BAR_WG0 + _TK_WG, number_of_threads=128
                    )
                kv_idx = kv_lo_cur + cutlass.Int32(0)
                kv_go = cutlass.Boolean(True)
                while kv_go:
                    q_half_start = cutlass.const_expr(_TK_WG * self.queries_per_group)
                    q_half_end = cutlass.const_expr(
                        (_TK_WG + 1) * self.queries_per_group
                    )
                    # Expand both tiles at trace time; each keeps its own pipeline state.
                    for kv_group in cutlass.range_constexpr(2):
                        tCtAcc_base = (tCtAcc_base_0, tCtAcc_base_1)[kv_group]
                        umma_pipeline = (umma_pipeline_0, umma_pipeline_1)[kv_group]
                        umma_cons_state = (
                            umma_cons_state_0,
                            umma_cons_state_1,
                        )[kv_group]
                        if cutlass.const_expr(not self.is_mx_sf):
                            sSF_KV = (sSF_KV_0, sSF_KV_1)[kv_group]
                        kv_pos = (kv_idx + kv_group) * block_kv_val + m_coord
                        umma_pipeline.consumer_wait(umma_cons_state)
                        tCtAcc_c0 = tCtAcc_base[None, None, None, umma_cons_state.index]
                        tAcc_c0 = tCtAcc_c0[(None, None), 0, 0]
                        tAcc_c0_epi = cute.flat_divide(tAcc_c0, epi_sub_mn)
                        tTR_0 = thr_copy_ref.partition_S(tAcc_c0_epi)
                        cute.copy(
                            tiled_copy_ref,
                            tTR_0[
                                None,
                                None,
                                None,
                                0,
                                q_half_start
                                // self.queries_per_group
                                * num_epi_subtiles,
                            ],
                            tTR_rAcc,
                        )
                        cute.arch.fence_view_async_tmem_load()
                        if cutlass.const_expr(self.queries_per_group > 1):
                            umma_pipeline.consumer_release(umma_cons_state)
                            umma_cons_state.advance()
                            if cutlass.const_expr(self.single_acc):
                                umma_cons_state.advance()
                        subtile_n = num_heads // num_epi_subtiles
                        for t in cutlass.range_constexpr(q_half_start, q_half_end):
                            s_acc = self.epi.zero()
                            for i in cutlass.range_constexpr(num_epi_subtiles):
                                if cutlass.const_expr(self.queries_per_group == 1) and (
                                    t > q_half_start or i > 0
                                ):
                                    cute.copy(
                                        tiled_copy_ref,
                                        tTR_0[
                                            None,
                                            None,
                                            None,
                                            0,
                                            t * num_epi_subtiles + i,
                                        ],
                                        tTR_rAcc,
                                    )
                                    cute.arch.fence_view_async_tmem_load()
                                if (
                                    self.queries_per_group == 1
                                    and t == q_half_end - 1
                                    and i == num_epi_subtiles - 1
                                ):
                                    umma_pipeline.consumer_release(umma_cons_state)
                                    umma_cons_state.advance()
                                    if cutlass.const_expr(self.single_acc):
                                        umma_cons_state.advance()
                                sw_col = sW[None, q_stage_local]
                                reg_h_end = min(
                                    subtile_n,
                                    max(0, NUM_W_IN_REG - i * subtile_n),
                                )
                                for h in cutlass.range_constexpr(0, reg_h_end, 4):
                                    n0 = h + (t - q_half_start) * num_heads
                                    h_g = i * subtile_n + h
                                    x4 = _subview(tTR_rAcc, n0, 4)
                                    a4 = self.epi.relu(x4)
                                    r0 = t * NUM_W_IN_REG + h_g
                                    w4 = _subview(w_cache, r0, 4)
                                    s_acc = self.epi.fma(a4, w4, s_acc)
                                smem_h_start = max(0, NUM_W_IN_REG - i * subtile_n)
                                for h in cutlass.range_constexpr(
                                    smem_h_start, subtile_n, 4
                                ):
                                    n0 = h + (t - q_half_start) * num_heads
                                    h_g = i * subtile_n + h
                                    x4 = _subview(tTR_rAcc, n0, 4)
                                    a4 = self.epi.relu(x4)
                                    w_i = t * num_heads + h_g
                                    w4 = _subview(sw_col, w_i, 4)
                                    s_acc = self.epi.fma(a4, w4, s_acc)
                            if cutlass.const_expr(not self.is_mx_sf):
                                result_arr[t] = self.epi.finish(
                                    s_acc, sSF_KV[m_coord, sc_slot]
                                )
                            else:
                                result_arr[t] = self.epi.finish(s_acc, None)
                        for t in cutlass.range_constexpr(q_half_start, q_half_end):
                            if cutlass.const_expr(self.repair):
                                topk.append(
                                    cutlass.Float32(result_arr[t]),
                                    (
                                        (
                                            q_block_idx * block_q_val + t < num_q_tokens
                                            and kv_pos >= kv_start_cache[t]
                                            and (kv_pos < kv_end_cache[t])
                                        )
                                        and kv_pos < num_kv * block_kv_val
                                    )
                                    & (cutlass.Float32(result_arr[t]) >= stream_tau)
                                    & stream_repair,
                                    kv_pos - kv_start_cache[t],
                                    t,
                                    self.stream_cap,
                                    s_stream_entries,
                                    s_stream_counts,
                                )
                            else:
                                if cutlass.const_expr(self.materialize_scores):
                                    if (
                                        q_block_idx * block_q_val + t < num_q_tokens
                                        and kv_pos >= kv_start_cache[t]
                                        and (kv_pos < kv_end_cache[t])
                                    ) and kv_pos < num_kv * block_kv_val:
                                        mLogits[
                                            q_block_idx * block_q_val + t,
                                            kv_pos,
                                        ] = result_arr[t]
                                elif cutlass.const_expr(self.direct_candidates):
                                    warp_counts[t] = global_append(
                                        cutlass.Float32(result_arr[t]),
                                        (
                                            q_block_idx * block_q_val + t < num_q_tokens
                                            and kv_pos >= kv_start_cache[t]
                                            and kv_pos < kv_end_cache[t]
                                            and kv_pos < num_kv * block_kv_val
                                        )
                                        & (
                                            cutlass.Float32(result_arr[t])
                                            >= tau_cache[t]
                                        ),
                                        kv_pos - kv_start_cache[t],
                                        q_block_idx * block_q_val + t,
                                        self.stream_cap,
                                        warp_counts[t],
                                        mTopkCand,
                                    )
                                else:
                                    stream_warp_count = bounded_append(
                                        cutlass.Float32(result_arr[t]),
                                        (
                                            (
                                                q_block_idx * block_q_val + t
                                                < num_q_tokens
                                                and kv_pos >= kv_start_cache[t]
                                                and (kv_pos < kv_end_cache[t])
                                            )
                                            and kv_pos < num_kv * block_kv_val
                                        )
                                        & (cutlass.Float32(result_arr[t]) >= stream_tau)
                                        & stream_repair,
                                        kv_pos - kv_start_cache[t],
                                        t,
                                        self.stream_cap,
                                        s_stream_entries,
                                        stream_warp_count,
                                    )
                        if cutlass.const_expr(kv_group == 0):
                            umma_cons_state_0 = umma_cons_state
                        else:
                            umma_cons_state_1 = umma_cons_state
                    if cutlass.const_expr(not self.is_mx_sf):
                        sc_slot = (sc_slot + 1) % self.num_sc_slots
                    kv_idx = kv_idx + NUM_MATH_WG
                    kv_go = kv_idx < num_kv
                    if cutlass.const_expr(not self.materialize_scores):
                        stream_tiles += 1
                    if cutlass.const_expr(self.repair):
                        if cutlass.const_expr(not self.materialize_scores):
                            if (stream_tiles == self.stream_period) | (not kv_go):
                                cute.arch.barrier(
                                    barrier_id=_TK_BAR_WG0 + _TK_WG,
                                    number_of_threads=128,
                                )
                                stream_tau = topk.compact(
                                    local_tidx,
                                    stream_tau,
                                    _TK_WG,
                                    _TK_WG,
                                    self.top_k,
                                    self.stream_cap,
                                    self.stream_cap - self.stream_period * 256,
                                    _TK_BAR_WG0 + _TK_WG,
                                    s_stream_entries,
                                    s_stream_counts,
                                    s_score_hist,
                                    s_stream_scalars,
                                    low_registers=True,
                                )
                                stream_tiles = 0
                if cutlass.const_expr(self.direct_candidates):
                    for query in cutlass.range_constexpr(
                        _TK_WG * self.queries_per_group,
                        (_TK_WG + 1) * self.queries_per_group,
                    ):
                        if local_tidx % 32 == 0:
                            row = q_block_idx * block_q_val + query
                            if row < num_q_tokens:
                                part = local_tidx // 32
                                mTopkCand[
                                    row, (part + 1) * (self.stream_cap // 4) - 1
                                ] = cutlass.Int64(warp_counts[query])
                elif cutlass.const_expr(not self.repair):
                    if local_tidx % 32 == 0:
                        if cutlass.const_expr(not self.materialize_scores):
                            s_stream_counts[_TK_WG * 4 + local_tidx // 32] = (
                                stream_warp_count
                            )
                if cutlass.const_expr(not self.materialize_scores):
                    cute.arch.barrier(
                        barrier_id=_TK_BAR_FIN,
                        number_of_threads=self.num_math_threads,
                    )
                    stream_row = q_block_idx * block_q_val + _TK_WG
                    if cutlass.const_expr(self.direct_candidates):
                        pass
                    elif (stream_row < num_q_tokens) & stream_repair:
                        if cutlass.const_expr(self.repair):
                            finish_split(
                                local_tidx,
                                stream_row,
                                _TK_WG,
                                self.top_k,
                                self.stream_cap,
                                _TK_BAR_WG0 + _TK_WG,
                                s_stream_entries,
                                s_stream_counts,
                                s_score_hist,
                                s_stream_scalars,
                                mTopkIdx,
                                mTopkCandHi,
                                mTopkCandCnt,
                                bidz,
                                repair_part,
                                repair_parts,
                                failed_groups,
                                topk_indices_offset,
                            )
                        else:
                            partition_dump(
                                local_tidx,
                                stream_row,
                                _TK_WG,
                                self.stream_cap,
                                s_stream_entries,
                                s_stream_counts,
                                mTopkCand,
                                mTopkCandCnt,
                            )
                    cute.arch.barrier(
                        barrier_id=_TK_BAR_FIN,
                        number_of_threads=self.num_math_threads,
                    )
                next_q_block_idx = next_group(
                    q_block_idx + grid_size,
                    grid_size,
                    mTopkCandCnt,
                    mCuSeqLenKStart,
                    mCuSeqLenKEnd,
                    num_q_tokens,
                    self.top_k,
                    self.stream_cap,
                    self.repair,
                    self.block_q,
                    repair_parts,
                )
                if next_q_block_idx < num_q_blocks:
                    next_num_kv = self.bounds.tile_end(
                        mCuSeqLenKEnd,
                        next_q_block_idx,
                        num_q_tokens,
                        repair_part,
                        repair_parts,
                        0,
                    )
                has_work = next_q_block_idx < num_q_blocks
            if q_block_idx < num_q_blocks:
                q_pipeline.consumer_release(q_cons_state)
                q_cons_state.advance()

        if cutlass.const_expr(self.repair):
            sm_idx, repair_part, repair_parts, failed_groups = schedule(
                bidz,
                grid_size,
                mTopkCandCnt,
                mCuSeqLenKStart,
                mCuSeqLenKEnd,
                num_q_tokens,
                self.top_k,
                self.stream_cap,
                self.block_q,
            )
            if (repair_parts > 1) & (sm_idx < num_q_blocks):
                end = self.bounds.end(mCuSeqLenKEnd, sm_idx, num_q_tokens)
                # Bound the last-CTA merge work relative to each KV slice.
                repair_parts = min(repair_parts, max(1, end // (8 * self.top_k)))
                if repair_part >= repair_parts:
                    sm_idx = num_q_blocks
        # Skip empty repair CTAs before initializing pipelines and TMEM.
        if cutlass.const_expr(not self.repair) or sm_idx < num_q_blocks:
            start_q_block = sm_idx
            start_q_block_clamped = min(start_q_block, num_q_blocks - 1)
            current_num_kv = self.bounds.tile_end(
                mCuSeqLenKEnd,
                start_q_block_clamped,
                num_q_tokens,
                repair_part,
                repair_parts,
                0,
            )

            current_kv_lo = self.bounds.tile_start(
                mCuSeqLenKStart,
                mCuSeqLenKEnd,
                start_q_block_clamped,
                num_q_tokens,
                repair_part,
                repair_parts,
                0,
            )
            if is_tma_warp:
                cpasync.prefetch_descriptor(tma_atom_a)
                cpasync.prefetch_descriptor(tma_atom_b)
                cpasync.prefetch_descriptor(tma_atom_w)
                cpasync.prefetch_descriptor(tma_atom_kv_sf)
                cpasync.prefetch_descriptor(tma_atom_q_sf)
            smem = cutlass.utils.SmemAllocator()
            storage = smem.allocate(SharedStorage)
            num_heads = self.num_heads
            num_epi_subtiles = 1
            prod_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
            q_cons_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.num_math_warps
            )
            q_pipeline = pipeline.PipelineTmaAsync.create(
                barrier_storage=storage.q_mbar.data_ptr(),
                num_stages=self.num_q_stages,
                producer_group=prod_group,
                consumer_group=q_cons_group,
                tx_count=self.num_q_tma_bytes,
                cta_layout_vmnk=cluster_layout_vmnk,
                tidx=tidx,
                defer_sync=True,
            )
            q_prod_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_q_stages
            )
            q_cons_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_q_stages
            )
            q_cons_state_umma_0 = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_q_stages
            )
            q_cons_state_umma_1 = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_q_stages
            )
            kv_cons_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
            kv_pipeline_0 = pipeline.PipelineTmaUmma.create(
                barrier_storage=storage.kv_mbar_0.data_ptr(),
                num_stages=self.num_kv_stages,
                producer_group=prod_group,
                consumer_group=kv_cons_group,
                tx_count=self.num_kv_sf_tma_bytes,
                cta_layout_vmnk=cluster_layout_vmnk,
                defer_sync=True,
            )
            kv_pipeline_1 = pipeline.PipelineTmaUmma.create(
                barrier_storage=storage.kv_mbar_1.data_ptr(),
                num_stages=self.num_kv_stages,
                producer_group=prod_group,
                consumer_group=kv_cons_group,
                tx_count=self.num_kv_sf_tma_bytes,
                cta_layout_vmnk=cluster_layout_vmnk,
                defer_sync=True,
            )
            kv_prod_state_0 = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_kv_stages
            )
            kv_prod_state_1 = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_kv_stages
            )
            kv_cons_state_umma_0 = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_kv_stages
            )
            kv_cons_state_umma_1 = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_kv_stages
            )
            math_threads_per_group = self.num_math_threads
            umma_pipeline_0 = pipeline.PipelineUmmaAsync.create(
                barrier_storage=storage.umma_mbar_0.data_ptr(),
                num_stages=self.num_umma_stages,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                consumer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, math_threads_per_group
                ),
                cta_layout_vmnk=cluster_layout_vmnk,
                defer_sync=True,
            )
            if cutlass.const_expr(self.single_acc):
                umma_pipeline_1 = umma_pipeline_0
            else:
                umma_pipeline_1 = pipeline.PipelineUmmaAsync.create(
                    barrier_storage=storage.umma_mbar_1.data_ptr(),
                    num_stages=self.num_umma_stages,
                    producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                    consumer_group=pipeline.CooperativeGroup(
                        pipeline.Agent.Thread, math_threads_per_group
                    ),
                    cta_layout_vmnk=cluster_layout_vmnk,
                    defer_sync=True,
                )
            sfb_war_pipeline = pipeline.PipelineUmmaAsync.create(
                barrier_storage=storage.sfb_war_mbar.data_ptr(),
                num_stages=1,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 2),
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
                cta_layout_vmnk=cluster_layout_vmnk,
                defer_sync=True,
            )
            sfb_war_prod_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, 1
            )
            sfb_war_cons_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, 1
            )
            umma_prod_state_0 = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_umma_stages
            )
            umma_prod_state_1 = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_umma_stages
            )
            umma_cons_state_0 = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_umma_stages
            )
            umma_cons_state_1 = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_umma_stages
            )
            if cutlass.const_expr(self.single_acc):
                assert self.num_umma_stages == 1
                umma_prod_state_1.advance()
                umma_cons_state_1.advance()
            tmem_alloc_num_threads = self.num_math_threads + 64
            tmem_alloc_barrier = pipeline.NamedBarrier(
                barrier_id=1, num_threads=tmem_alloc_num_threads
            )
            sfb_sync_barrier = pipeline.NamedBarrier(barrier_id=2, num_threads=64)
            tmem = cutlass.utils.TmemAllocator(
                storage.tmem_holding_buf,
                barrier_for_retrieve=tmem_alloc_barrier,
                allocator_warp_id=0,
                is_two_cta=False,
                arch=self.arch_str,
            )
            pipeline_init_arrive(
                cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True
            )
            sKV_0 = smem.allocate_tensor(
                element_type=self.a_dtype,
                layout=a_smem_layout_staged.outer,
                byte_alignment=128,
                swizzle=a_smem_layout_staged.inner,
            )
            sKV_1 = smem.allocate_tensor(
                element_type=self.a_dtype,
                layout=a_smem_layout_staged.outer,
                byte_alignment=128,
                swizzle=a_smem_layout_staged.inner,
            )
            sQ = smem.allocate_tensor(
                element_type=self.b_dtype,
                layout=b_smem_layout_staged.outer,
                byte_alignment=128,
                swizzle=b_smem_layout_staged.inner,
            )
            sSF_Q = smem.allocate_tensor(
                element_type=cutlass.Int32,
                layout=q_sf_smem_layout_staged,
                byte_alignment=128,
            )
            sW = smem.allocate_tensor(
                element_type=self.epi.dtype,
                layout=w_smem_layout_staged,
                byte_alignment=128,
            )
            sSF_KV_0 = smem.allocate_tensor(
                element_type=cutlass.Float32 if not self.is_mx_sf else cutlass.Int32,
                layout=kv_sf_smem_layout_staged,
                byte_alignment=128,
            )
            sSF_KV_1 = smem.allocate_tensor(
                element_type=cutlass.Float32 if not self.is_mx_sf else cutlass.Int32,
                layout=kv_sf_smem_layout_staged,
                byte_alignment=128,
            )
            s_score_hist = None
            s_stream_entries = None
            s_stream_counts = None
            s_stream_scalars = None
            if cutlass.const_expr(not self.materialize_scores):
                s_score_hist = smem.allocate_tensor(
                    element_type=cutlass.Int32,
                    layout=cute.make_ordered_layout(
                        (1 if self.direct_candidates else self.block_q * 256,),
                        order=(0,),
                    ),
                    byte_alignment=128,
                )
                s_stream_entries = smem.allocate_tensor(
                    element_type=cutlass.Uint64,
                    layout=cute.make_ordered_layout(
                        (
                            (
                                1
                                if self.direct_candidates
                                else self.block_q * self.stream_cap
                            ),
                        ),
                        order=(0,),
                    ),
                    byte_alignment=128,
                )
                s_stream_counts = smem.allocate_tensor(
                    element_type=cutlass.Int32,
                    layout=cute.make_ordered_layout((self.block_q * 4,), order=(0,)),
                    byte_alignment=128,
                )
                s_stream_scalars = smem.allocate_tensor(
                    element_type=cutlass.Int32,
                    layout=cute.make_ordered_layout((self.block_q * 2,), order=(0,)),
                    byte_alignment=128,
                )
            a_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            b_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )
            thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
            gA = cute.local_tile(
                mA_mkl, (self.block_kv, self.head_dim), coord=(None, None)
            )
            sKV_0_for_tma = cute.make_tensor(
                sKV_0.iterator, cute.group_modes(a_tma_smem_layout.outer, 0, 2)
            )
            sKV_1_for_tma = cute.make_tensor(
                sKV_1.iterator, cute.group_modes(a_tma_smem_layout.outer, 0, 2)
            )
            tAsA_0, tAgA_0 = cpasync.tma_partition(
                tma_atom_a,
                0,
                cute.make_layout(1),
                sKV_0_for_tma,
                cute.group_modes(gA, 0, 2),
            )
            tAsA_1, tAgA_1 = cpasync.tma_partition(
                tma_atom_a,
                0,
                cute.make_layout(1),
                sKV_1_for_tma,
                cute.group_modes(gA, 0, 2),
            )
            gB_nkl = cute.local_tile(
                mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
            )
            tCgB = thr_mma.partition_B(gB_nkl)
            b_cta_layout = cute.make_layout(
                cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
            )
            tBsB, tBgB = cpasync.tma_partition(
                tma_atom_b,
                block_in_cluster_coord_vmnk[1],
                b_cta_layout,
                cute.group_modes(sQ, 0, 3),
                cute.group_modes(tCgB, 0, 3),
            )
            tBgB = tBgB[None, None, None, 0]
            w_cta_layout = cute.make_layout((1,))
            tWsW, tWgW = cpasync.tma_partition(
                tma_atom_w,
                0,
                w_cta_layout,
                cute.group_modes(sW, 0, 1),
                cute.group_modes(
                    cute.local_tile(mW_tma, (self.UMMA_N,), (None,)), 0, 1
                ),
            )
            # Both formats share the TMA path. The allocation's full layout keeps
            # FP8's epilogue scale ring distinct from the MMA-owned KV stages.
            gSF_KV = cute.local_tile(mSF_KV_tma, (self.block_kv,), coord=(None,))
            tSsSF_KV_0, tSgSF_KV_0 = cpasync.tma_partition(
                tma_atom_kv_sf, 0, cute.make_layout(1), sSF_KV_0, gSF_KV
            )
            tSsSF_KV_1, tSgSF_KV_1 = cpasync.tma_partition(
                tma_atom_kv_sf, 0, cute.make_layout(1), sSF_KV_1, gSF_KV
            )
            sc_slot = cutlass.Int32(0)
            sSF_Q_for_tma = cute.make_tensor(
                sSF_Q.iterator, q_sf_tma_smem_layout_staged
            )
            q_sf_cta_layout = cute.make_layout((1,))
            tSF_Q_sSF_Q, tSF_Q_gSF_Q = cpasync.tma_partition(
                tma_atom_q_sf,
                0,
                q_sf_cta_layout,
                cute.group_modes(sSF_Q_for_tma, 0, 1),
                cute.group_modes(
                    cute.local_tile(mSF_Q_tma, (self.UMMA_N,), (None,)), 0, 1
                ),
            )
            tCrA_0 = tiled_mma.make_fragment_A(sKV_0)
            tCrA_1 = tiled_mma.make_fragment_A(sKV_1)
            tCrB = tiled_mma.make_fragment_B(sQ)
            acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
            acc_shape_staged = cute.append(acc_shape, self.num_umma_stages)
            tCtAcc_fake_staged = tiled_mma.make_fragment_C(acc_shape_staged)
            cols_per_group = (
                self.num_tmem_alloc_cols
                * self.num_umma_stages
                * (32 // self.acc_dtype.width)
            )

            c_layout = cutlass.utils.LayoutEnum.ROW_MAJOR
            epi_sub_mn = (
                epi_tile[0],
                num_heads * self.queries_per_group // num_epi_subtiles,
            )
            copy_atom_t2r = sm100_utils.get_tmem_load_op(
                self.cta_tile_shape_mnk,
                c_layout,
                self.acc_dtype,
                self.acc_dtype,
                epi_sub_mn,
                use_2cta_instrs,
            )
            q_block_idx = cutlass.Int32(sm_idx)
            kv_idx = cutlass.Int32(current_kv_lo)
            num_kv = current_num_kv
            prev_q_block_idx = num_q_blocks
            next_q_block_idx = q_block_idx
            next_kv_idx = kv_idx
            next_num_kv = num_kv
            has_work = q_block_idx < num_q_blocks
            pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)
            if is_tma_warp_0:
                cute.arch.warpgroup_reg_dealloc(
                    48 if self.queries_per_group > 1 else 24
                )
                if has_work:
                    q_pipeline.producer_acquire(q_prod_state)
                    q_bar = q_pipeline.producer_get_barrier(q_prod_state)
                    cute.copy(
                        tma_atom_b,
                        tBgB[None, next_q_block_idx, 0],
                        tBsB[None, q_prod_state.index],
                        tma_bar_ptr=q_bar,
                        mcast_mask=b_mcast_mask,
                    )
                    if cutlass.const_expr(self.is_mx_sf):
                        cute.copy(
                            tma_atom_q_sf,
                            tSF_Q_gSF_Q[None, next_q_block_idx],
                            tSF_Q_sSF_Q[None, q_prod_state.index],
                            tma_bar_ptr=q_bar,
                        )
                    cute.copy(
                        tma_atom_w,
                        tWgW[None, next_q_block_idx],
                        tWsW[None, q_prod_state.index],
                        tma_bar_ptr=q_bar,
                    )
                    q_prod_state.advance()
                while has_work:
                    q_block_idx_old = prev_q_block_idx
                    prev_q_block_idx = next_q_block_idx
                    q_block_idx = next_q_block_idx
                    kv_idx = next_kv_idx
                    num_kv = next_num_kv
                    if q_block_idx != q_block_idx_old:
                        prefetch_next_q = next_group(
                            q_block_idx + grid_size,
                            grid_size,
                            mTopkCandCnt,
                            mCuSeqLenKStart,
                            mCuSeqLenKEnd,
                            num_q_tokens,
                            self.top_k,
                            self.stream_cap,
                            self.repair,
                            self.block_q,
                            repair_parts,
                        )
                        if prefetch_next_q < num_q_blocks:
                            q_pipeline.producer_acquire(q_prod_state)
                            q_bar = q_pipeline.producer_get_barrier(q_prod_state)
                            cute.copy(
                                tma_atom_b,
                                tBgB[None, prefetch_next_q, 0],
                                tBsB[None, q_prod_state.index],
                                tma_bar_ptr=q_bar,
                                mcast_mask=b_mcast_mask,
                            )
                            if cutlass.const_expr(self.is_mx_sf):
                                cute.copy(
                                    tma_atom_q_sf,
                                    tSF_Q_gSF_Q[None, prefetch_next_q],
                                    tSF_Q_sSF_Q[None, q_prod_state.index],
                                    tma_bar_ptr=q_bar,
                                )
                            cute.copy(
                                tma_atom_w,
                                tWgW[None, prefetch_next_q],
                                tWsW[None, q_prod_state.index],
                                tma_bar_ptr=q_bar,
                            )
                            q_prod_state.advance()
                    kv_pipeline_0.producer_acquire(kv_prod_state_0)
                    bar = kv_pipeline_0.producer_get_barrier(kv_prod_state_0)
                    stage = kv_prod_state_0.index
                    cute.copy(
                        tma_atom_a,
                        tAgA_0[None, kv_idx, 0],
                        tAsA_0[None, stage],
                        tma_bar_ptr=bar,
                        mcast_mask=a_mcast_mask,
                    )
                    sf_slot = (
                        sc_slot if cutlass.const_expr(not self.is_mx_sf) else stage
                    )
                    cute.copy(
                        tma_atom_kv_sf,
                        tSgSF_KV_0[None, kv_idx],
                        tSsSF_KV_0[None, sf_slot],
                        tma_bar_ptr=bar,
                    )
                    if cutlass.const_expr(not self.is_mx_sf):
                        sc_slot = (sc_slot + 1) % self.num_sc_slots
                    kv_prod_state_0.advance()
                    next_kv_idx = kv_idx + NUM_MATH_WG
                    if next_kv_idx >= num_kv:
                        next_q_block_idx = next_group(
                            q_block_idx + grid_size,
                            grid_size,
                            mTopkCandCnt,
                            mCuSeqLenKStart,
                            mCuSeqLenKEnd,
                            num_q_tokens,
                            self.top_k,
                            self.stream_cap,
                            self.repair,
                            self.block_q,
                            repair_parts,
                        )
                        next_kv_idx = self.bounds.tile_start(
                            mCuSeqLenKStart,
                            mCuSeqLenKEnd,
                            next_q_block_idx,
                            num_q_tokens,
                            repair_part,
                            repair_parts,
                            0,
                        )
                        if next_q_block_idx < num_q_blocks:
                            next_num_kv = self.bounds.tile_end(
                                mCuSeqLenKEnd,
                                next_q_block_idx,
                                num_q_tokens,
                                repair_part,
                                repair_parts,
                                0,
                            )
                    has_work = next_q_block_idx < num_q_blocks
            elif is_tma_warp_1:
                cute.arch.warpgroup_reg_dealloc(
                    48 if self.queries_per_group > 1 else 24
                )
                while has_work:
                    q_block_idx = next_q_block_idx
                    kv_idx = next_kv_idx
                    num_kv = next_num_kv
                    kv_idx_1 = kv_idx + 1
                    kv_pipeline_1.producer_acquire(kv_prod_state_1)
                    bar = kv_pipeline_1.producer_get_barrier(kv_prod_state_1)
                    stage = kv_prod_state_1.index
                    cute.copy(
                        tma_atom_a,
                        tAgA_1[None, kv_idx_1, 0],
                        tAsA_1[None, stage],
                        tma_bar_ptr=bar,
                        mcast_mask=a_mcast_mask,
                    )
                    sf_slot = (
                        sc_slot if cutlass.const_expr(not self.is_mx_sf) else stage
                    )
                    cute.copy(
                        tma_atom_kv_sf,
                        tSgSF_KV_1[None, kv_idx_1],
                        tSsSF_KV_1[None, sf_slot],
                        tma_bar_ptr=bar,
                    )
                    if cutlass.const_expr(not self.is_mx_sf):
                        sc_slot = (sc_slot + 1) % self.num_sc_slots
                    kv_prod_state_1.advance()
                    next_kv_idx = kv_idx + NUM_MATH_WG
                    if next_kv_idx >= num_kv:
                        next_q_block_idx = next_group(
                            q_block_idx + grid_size,
                            grid_size,
                            mTopkCandCnt,
                            mCuSeqLenKStart,
                            mCuSeqLenKEnd,
                            num_q_tokens,
                            self.top_k,
                            self.stream_cap,
                            self.repair,
                            self.block_q,
                            repair_parts,
                        )
                        next_kv_idx = self.bounds.tile_start(
                            mCuSeqLenKStart,
                            mCuSeqLenKEnd,
                            next_q_block_idx,
                            num_q_tokens,
                            repair_part,
                            repair_parts,
                            0,
                        )
                        if next_q_block_idx < num_q_blocks:
                            next_num_kv = self.bounds.tile_end(
                                mCuSeqLenKEnd,
                                next_q_block_idx,
                                num_q_tokens,
                                repair_part,
                                repair_parts,
                                0,
                            )
                    has_work = next_q_block_idx < num_q_blocks
            elif is_umma_warp_0:
                cute.arch.warpgroup_reg_dealloc(
                    48 if self.queries_per_group > 1 else 24
                )
                tmem.wait_for_alloc()
                tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
                tCtAcc_base_0 = cute.make_tensor(tmem_ptr, tCtAcc_fake_staged.layout)
                tCtAcc_base_1 = cute.make_tensor(
                    tmem_ptr + (0 if self.single_acc else cols_per_group),
                    tCtAcc_fake_staged.layout,
                )
                if cutlass.const_expr(self.is_mx_sf):
                    sf_base_offset = (
                        self.num_tmem_alloc_cols
                        * self.num_acc_groups
                        * self.num_umma_stages
                    )
                    sfa_tmem_ptr = cute.recast_ptr(
                        tmem_ptr + sf_base_offset, dtype=self.sf_dtype
                    )
                    sfb_tmem_ptr = cute.recast_ptr(
                        tmem_ptr
                        + sf_base_offset
                        + self.num_sfa_tmem_cols * self.num_groups,
                        dtype=self.sf_dtype,
                    )
                    tCtSFA = cute.make_tensor(sfa_tmem_ptr, tmem_sfa_layout)
                    tCtSFB = cute.make_tensor(sfb_tmem_ptr, tmem_sfb_layout)
                if is_leader_cta:
                    num_k_blocks = cute.size(tCrA_0.shape[2])
                    q_stage_0 = cutlass.Int32(0)
                    while has_work:
                        q_block_idx_old = prev_q_block_idx
                        prev_q_block_idx = next_q_block_idx
                        q_block_idx = next_q_block_idx
                        kv_idx = next_kv_idx
                        num_kv = next_num_kv
                        if q_block_idx != q_block_idx_old:
                            if q_block_idx_old < num_q_blocks:
                                if cutlass.const_expr(self.is_mx_sf):
                                    sfb_war_pipeline.producer_commit(sfb_war_prod_state)
                                q_cons_state_umma_0.advance()
                            q_pipeline.consumer_wait(q_cons_state_umma_0)
                            q_stage_0 = q_cons_state_umma_0.index
                            if cutlass.const_expr(self.is_mx_sf):
                                q_sf_atoms = self.N_padded // 128
                                for atom_idx in cutlass.range_constexpr(q_sf_atoms):
                                    atom_offset = atom_idx * 128
                                    stage_offset = q_stage_0 * sSF_Q.layout.stride[1]
                                    utccp_required_smem_warp_transpose(
                                        sSF_Q.iterator + stage_offset + atom_offset
                                    )
                                cute.arch.fence_view_async_shared()
                                if q_block_idx_old < num_q_blocks:
                                    sfb_war_pipeline.consumer_wait(sfb_war_cons_state)
                                    sfb_war_cons_state.advance()
                                sSF_Q_ue8m0 = cute.recast_tensor(sSF_Q, Float8E8M0FNU)
                                stage_off_ue8m0 = (
                                    q_stage_0 * sSF_Q_ue8m0.layout.stride[1]
                                )
                                sSF_Q_chunk = cute.make_tensor(
                                    sSF_Q_ue8m0.iterator + stage_off_ue8m0,
                                    sfb_chunk_smem_layout,
                                )
                                tCsSFB_compact = cute.filter_zeros(sSF_Q_chunk)
                                tCtSFB_compact = cute.filter_zeros(tCtSFB)
                                tiled_copy_s2t_sfb = tcgen05.make_s2t_copy(
                                    copy_atom_s2t, tCtSFB_compact
                                )
                                thr_copy_s2t_sfb = tiled_copy_s2t_sfb.get_slice(0)
                                tCsSFB_s2t_ = thr_copy_s2t_sfb.partition_S(
                                    tCsSFB_compact
                                )
                                tCsSFB_s2t = tcgen05.get_s2t_smem_desc_tensor(
                                    tiled_copy_s2t_sfb, tCsSFB_s2t_
                                )
                                tCtSFB_s2t = thr_copy_s2t_sfb.partition_D(
                                    tCtSFB_compact
                                )
                                cute.copy(tiled_copy_s2t_sfb, tCsSFB_s2t, tCtSFB_s2t)
                                cute.arch.fence_view_async_tmem_store()
                                if cutlass.const_expr(self.is_mx_sf):
                                    sfb_sync_barrier.arrive_and_wait()
                        kv_pipeline_0.consumer_wait(kv_cons_state_umma_0)
                        umma_pipeline_0.producer_acquire(umma_prod_state_0)
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                        kv_stage = kv_cons_state_umma_0.index
                        if cutlass.const_expr(self.is_mx_sf):
                            if cutlass.const_expr(not self.remove_online_sf_transpose):
                                kv_sf_atoms = self.block_kv // BLOCK_KV_ROWS
                                for atom_idx in cutlass.range_constexpr(kv_sf_atoms):
                                    atom_offset = atom_idx * 128
                                    stage_offset = kv_stage * sSF_KV_0.layout.stride[1]
                                    utccp_required_smem_warp_transpose(
                                        sSF_KV_0.iterator + stage_offset + atom_offset
                                    )
                                cute.arch.fence_view_async_shared()
                            sSF_KV_0_ue8m0 = cute.recast_tensor(sSF_KV_0, Float8E8M0FNU)
                            stage_off_kv0_ue8m0 = (
                                kv_stage * sSF_KV_0_ue8m0.layout.stride[1]
                            )
                            sSF_KV_0_chunk = cute.make_tensor(
                                sSF_KV_0_ue8m0.iterator + stage_off_kv0_ue8m0,
                                sfa_chunk_smem_layout,
                            )
                            tCsSFA_compact = cute.filter_zeros(sSF_KV_0_chunk)
                            tCtSFA_compact = cute.filter_zeros(tCtSFA)
                            tiled_copy_s2t_sfa = tcgen05.make_s2t_copy(
                                copy_atom_s2t, tCtSFA_compact
                            )
                            thr_copy_s2t_sfa = tiled_copy_s2t_sfa.get_slice(0)
                            tCsSFA_s2t_ = thr_copy_s2t_sfa.partition_S(tCsSFA_compact)
                            tCsSFA_s2t = tcgen05.get_s2t_smem_desc_tensor(
                                tiled_copy_s2t_sfa, tCsSFA_s2t_
                            )
                            tCtSFA_s2t = thr_copy_s2t_sfa.partition_D(tCtSFA_compact)
                            cute.copy(tiled_copy_s2t_sfa, tCsSFA_s2t, tCtSFA_s2t)
                        tCtAcc_0 = tCtAcc_base_0[
                            None, None, None, umma_prod_state_0.index
                        ]
                        for k_block in cutlass.range_constexpr(num_k_blocks):
                            if cutlass.const_expr(self.is_mx_sf):
                                tCtSFA_k = cute.slice_(
                                    tCtSFA, (None, None, k_block * self.sf_k_step, None)
                                )
                                tCtSFB_k = cute.slice_(
                                    tCtSFB, (None, None, k_block * self.sf_k_step, None)
                                )
                                tiled_mma.set(tcgen05.Field.SFA, tCtSFA_k.iterator)
                                tiled_mma.set(tcgen05.Field.SFB, tCtSFB_k.iterator)
                            cute.gemm(
                                tiled_mma,
                                tCtAcc_0,
                                tCrA_0[None, None, k_block, kv_stage],
                                tCrB[None, None, k_block, q_stage_0],
                                tCtAcc_0,
                            )
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                        kv_pipeline_0.consumer_release(kv_cons_state_umma_0)
                        kv_cons_state_umma_0.advance()
                        umma_pipeline_0.producer_commit(umma_prod_state_0)
                        umma_prod_state_0.advance()
                        if cutlass.const_expr(self.single_acc):
                            umma_prod_state_0.advance()
                            sfb_sync_barrier.arrive_and_wait()
                        if cutlass.const_expr(self.is_mx_sf):
                            sfb_sync_barrier.arrive_and_wait()
                        next_kv_idx = kv_idx + NUM_MATH_WG
                        if next_kv_idx >= num_kv:
                            next_q_block_idx = next_group(
                                q_block_idx + grid_size,
                                grid_size,
                                mTopkCandCnt,
                                mCuSeqLenKStart,
                                mCuSeqLenKEnd,
                                num_q_tokens,
                                self.top_k,
                                self.stream_cap,
                                self.repair,
                                self.block_q,
                                repair_parts,
                            )
                            next_kv_idx = self.bounds.tile_start(
                                mCuSeqLenKStart,
                                mCuSeqLenKEnd,
                                next_q_block_idx,
                                num_q_tokens,
                                repair_part,
                                repair_parts,
                                0,
                            )
                            if next_q_block_idx < num_q_blocks:
                                next_num_kv = self.bounds.tile_end(
                                    mCuSeqLenKEnd,
                                    next_q_block_idx,
                                    num_q_tokens,
                                    repair_part,
                                    repair_parts,
                                    0,
                                )
                        has_work = next_q_block_idx < num_q_blocks
            elif is_umma_warp_1:
                cute.arch.warpgroup_reg_dealloc(
                    48 if self.queries_per_group > 1 else 24
                )
                tmem.wait_for_alloc()
                tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
                tCtAcc_base_0 = cute.make_tensor(tmem_ptr, tCtAcc_fake_staged.layout)
                tCtAcc_base_1 = cute.make_tensor(
                    tmem_ptr + (0 if self.single_acc else cols_per_group),
                    tCtAcc_fake_staged.layout,
                )
                if cutlass.const_expr(self.is_mx_sf):
                    sf_base_offset = (
                        self.num_tmem_alloc_cols
                        * self.num_acc_groups
                        * self.num_umma_stages
                    )
                    sfa_tmem_ptr = cute.recast_ptr(
                        tmem_ptr + sf_base_offset + self.num_sfa_tmem_cols,
                        dtype=self.sf_dtype,
                    )
                    sfb_tmem_ptr = cute.recast_ptr(
                        tmem_ptr
                        + sf_base_offset
                        + self.num_sfa_tmem_cols * self.num_groups,
                        dtype=self.sf_dtype,
                    )
                    tCtSFA = cute.make_tensor(sfa_tmem_ptr, tmem_sfa_layout)
                    tCtSFB = cute.make_tensor(sfb_tmem_ptr, tmem_sfb_layout)
                if is_leader_cta:
                    num_k_blocks_1 = cute.size(tCrA_1.shape[2])
                    q_stage_1 = cutlass.Int32(0)
                    while has_work:
                        q_block_idx_old = prev_q_block_idx
                        prev_q_block_idx = next_q_block_idx
                        q_block_idx = next_q_block_idx
                        kv_idx = next_kv_idx
                        num_kv = next_num_kv
                        if q_block_idx != q_block_idx_old:
                            if q_block_idx_old < num_q_blocks:
                                if cutlass.const_expr(self.is_mx_sf):
                                    sfb_war_pipeline.producer_commit(sfb_war_prod_state)
                                q_cons_state_umma_1.advance()
                            q_pipeline.consumer_wait(q_cons_state_umma_1)
                            q_stage_1 = q_cons_state_umma_1.index
                            if cutlass.const_expr(self.is_mx_sf):
                                sfb_sync_barrier.arrive_and_wait()
                        kv_pipeline_1.consumer_wait(kv_cons_state_umma_1)
                        if cutlass.const_expr(self.single_acc):
                            sfb_sync_barrier.arrive_and_wait()
                        umma_pipeline_1.producer_acquire(umma_prod_state_1)
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                        kv_stage_1 = kv_cons_state_umma_1.index
                        if cutlass.const_expr(self.is_mx_sf):
                            if cutlass.const_expr(not self.remove_online_sf_transpose):
                                kv_sf_atoms_1 = self.block_kv // BLOCK_KV_ROWS
                                for atom_idx in cutlass.range_constexpr(kv_sf_atoms_1):
                                    atom_offset = atom_idx * 128
                                    stage_offset = (
                                        kv_stage_1 * sSF_KV_1.layout.stride[1]
                                    )
                                    utccp_required_smem_warp_transpose(
                                        sSF_KV_1.iterator + stage_offset + atom_offset
                                    )
                                cute.arch.fence_view_async_shared()
                            sSF_KV_1_ue8m0 = cute.recast_tensor(sSF_KV_1, Float8E8M0FNU)
                            stage_off_kv1_ue8m0 = (
                                kv_stage_1 * sSF_KV_1_ue8m0.layout.stride[1]
                            )
                            sSF_KV_1_chunk = cute.make_tensor(
                                sSF_KV_1_ue8m0.iterator + stage_off_kv1_ue8m0,
                                sfa_chunk_smem_layout,
                            )
                            tCsSFA_compact_1 = cute.filter_zeros(sSF_KV_1_chunk)
                            tCtSFA_compact_1 = cute.filter_zeros(tCtSFA)
                            tiled_copy_s2t_sfa_1 = tcgen05.make_s2t_copy(
                                copy_atom_s2t, tCtSFA_compact_1
                            )
                            thr_copy_s2t_sfa_1 = tiled_copy_s2t_sfa_1.get_slice(0)
                            tCsSFA_s2t_1_ = thr_copy_s2t_sfa_1.partition_S(
                                tCsSFA_compact_1
                            )
                            tCsSFA_s2t_1 = tcgen05.get_s2t_smem_desc_tensor(
                                tiled_copy_s2t_sfa_1, tCsSFA_s2t_1_
                            )
                            tCtSFA_s2t_1 = thr_copy_s2t_sfa_1.partition_D(
                                tCtSFA_compact_1
                            )
                            cute.copy(tiled_copy_s2t_sfa_1, tCsSFA_s2t_1, tCtSFA_s2t_1)
                        tCtAcc_1 = tCtAcc_base_1[
                            None, None, None, umma_prod_state_1.index
                        ]
                        for k_block in cutlass.range_constexpr(num_k_blocks_1):
                            if cutlass.const_expr(self.is_mx_sf):
                                tCtSFA_k = cute.slice_(
                                    tCtSFA, (None, None, k_block * self.sf_k_step, None)
                                )
                                tCtSFB_k = cute.slice_(
                                    tCtSFB, (None, None, k_block * self.sf_k_step, None)
                                )
                                tiled_mma.set(tcgen05.Field.SFA, tCtSFA_k.iterator)
                                tiled_mma.set(tcgen05.Field.SFB, tCtSFB_k.iterator)
                            cute.gemm(
                                tiled_mma,
                                tCtAcc_1,
                                tCrA_1[None, None, k_block, kv_stage_1],
                                tCrB[None, None, k_block, q_stage_1],
                                tCtAcc_1,
                            )
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                        kv_pipeline_1.consumer_release(kv_cons_state_umma_1)
                        kv_cons_state_umma_1.advance()
                        umma_pipeline_1.producer_commit(umma_prod_state_1)
                        umma_prod_state_1.advance()
                        if cutlass.const_expr(self.single_acc):
                            umma_prod_state_1.advance()
                        if cutlass.const_expr(self.is_mx_sf):
                            sfb_sync_barrier.arrive_and_wait()
                        next_kv_idx = kv_idx + NUM_MATH_WG
                        if next_kv_idx >= num_kv:
                            next_q_block_idx = next_group(
                                q_block_idx + grid_size,
                                grid_size,
                                mTopkCandCnt,
                                mCuSeqLenKStart,
                                mCuSeqLenKEnd,
                                num_q_tokens,
                                self.top_k,
                                self.stream_cap,
                                self.repair,
                                self.block_q,
                                repair_parts,
                            )
                            next_kv_idx = self.bounds.tile_start(
                                mCuSeqLenKStart,
                                mCuSeqLenKEnd,
                                next_q_block_idx,
                                num_q_tokens,
                                repair_part,
                                repair_parts,
                                0,
                            )
                            if next_q_block_idx < num_q_blocks:
                                next_num_kv = self.bounds.tile_end(
                                    mCuSeqLenKEnd,
                                    next_q_block_idx,
                                    num_q_tokens,
                                    repair_part,
                                    repair_parts,
                                    0,
                                )
                        has_work = next_q_block_idx < num_q_blocks
            elif is_math_warp:
                cute.arch.warpgroup_reg_alloc(self.math_regs)
                tmem.allocate(self.num_tmem_alloc_cols_total)
                tmem.wait_for_alloc()
                tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
                tCtAcc_base_0 = cute.make_tensor(tmem_ptr, tCtAcc_fake_staged.layout)
                tCtAcc_base_1 = cute.make_tensor(
                    tmem_ptr + (0 if self.single_acc else cols_per_group),
                    tCtAcc_fake_staged.layout,
                )
                local_tidx = tidx % 128
                cC = cute.make_identity_tensor(epi_sub_mn)

                for _wg in cutlass.range_constexpr(self.math_groups):
                    if warpgroup_idx == _wg:
                        math_warpgroup(
                            _wg,
                            has_work,
                            prev_q_block_idx,
                            next_q_block_idx,
                            next_num_kv,
                            sc_slot,
                            q_block_idx,
                            num_kv,
                            kv_idx,
                            self=self,
                            NUM_MATH_WG=NUM_MATH_WG,
                            bidz=bidz,
                            block_kv_val=block_kv_val,
                            block_q_val=block_q_val,
                            cC=cC,
                            copy_atom_t2r=copy_atom_t2r,
                            epi_sub_mn=epi_sub_mn,
                            failed_groups=failed_groups,
                            grid_size=grid_size,
                            local_tidx=local_tidx,
                            mCuSeqLenKEnd=mCuSeqLenKEnd,
                            mCuSeqLenKStart=mCuSeqLenKStart,
                            mLogits=mLogits,
                            mTopkCand=mTopkCand,
                            mTopkCandCnt=mTopkCandCnt,
                            mTopkCandHi=mTopkCandHi,
                            mTopkIdx=mTopkIdx,
                            num_epi_subtiles=num_epi_subtiles,
                            num_heads=num_heads,
                            num_q_blocks=num_q_blocks,
                            num_q_tokens=num_q_tokens,
                            q_cons_state=q_cons_state,
                            q_pipeline=q_pipeline,
                            repair_part=repair_part,
                            repair_parts=repair_parts,
                            sSF_KV_0=sSF_KV_0,
                            sSF_KV_1=sSF_KV_1,
                            sW=sW,
                            s_score_hist=s_score_hist,
                            s_stream_counts=s_stream_counts,
                            s_stream_entries=s_stream_entries,
                            s_stream_scalars=s_stream_scalars,
                            tCtAcc_base_0=tCtAcc_base_0,
                            tCtAcc_base_1=tCtAcc_base_1,
                            topk_indices_offset=topk_indices_offset,
                            umma_cons_state_0=umma_cons_state_0,
                            umma_cons_state_1=umma_cons_state_1,
                            umma_pipeline_0=umma_pipeline_0,
                            umma_pipeline_1=umma_pipeline_1,
                        )
                tmem.relinquish_alloc_permit()
                tmem.free(tmem_ptr)
            else:
                cute.arch.warpgroup_reg_dealloc(
                    48 if self.queries_per_group > 1 else 24
                )


@functools.lru_cache(maxsize=None)
def _compile_kernel(
    block_kv: int,
    num_heads: int,
    head_dim: int,
    num_sms: int,
    epi_dtype,
    remove_online_sf_transpose: bool = False,
    top_k: int = 1024,
    qkv_dtype: str = "mxfp4",
    *,
    config,
    repair=False,
    materialize_scores=False,
    topk_row_stride=0,
    offset_dtype=None,
):
    """Compile and cache a producer specialization with dynamic row counts.

    Args:
        block_kv: Number of key rows in one MMA tile, fixed at 128.
        num_heads: Logical query head count: 8, 16, 32, or 64.
        head_dim: Unpacked dimension of each query/key head, fixed at 128.
        num_sms: Number of SMs available to the persistent launch grid.
        epi_dtype: Score arithmetic type, ``cutlass.Float32`` or ``cutlass.BFloat16``.
        remove_online_sf_transpose: Whether MXFP4/MXFP8 K scales use the pre-transposed
            128-row chunk layout.
        top_k: Number of indices selected per query, from 128 through 2048.
        qkv_dtype: Packed input format, ``"fp8"``, ``"mxfp4"`` or ``"mxfp8"``.
        config: Compile-time query tiling, histogram, accumulator and register policy.
        repair: Whether to rescan failed rows and select exact top-k indices.
        materialize_scores: Whether to emit full diagnostic scores; incompatible with
            repair.
        topk_row_stride: Output row stride in int32 elements, or zero for compact top-k
            rows.
        offset_dtype: CuTe Int32/Int64 index-offset type, or None when offsets are
            absent.
    """
    block_q = config.block_q
    sym_nq = cute.sym_int()
    inputs_fake = make_input_fakes(num_heads, head_dim, qkv_dtype, epi_dtype, sym_nq)
    epi = score_epilogue(epi_dtype)
    logits_fake = cute.runtime.make_fake_tensor(
        epi.dtype, (sym_nq, cute.sym_int()), stride=(cute.sym_int64(), 1)
    )
    if topk_row_stride and not materialize_scores:
        # Top-k columns of a wider [M, K + W] sliding-window output row.
        topk_fake = cute.runtime.make_fake_tensor(
            cutlass.Int32, (sym_nq, top_k), stride=(topk_row_stride, 1)
        )
    else:
        topk_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Int32,
            ((1 if materialize_scores else sym_nq), top_k),
            stride_order=(1, 0),
        )
    cand_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int64,
        (
            (1 if materialize_scores else sym_nq),
            candidate_capacity(top_k, qkv_dtype),
        ),
        stride_order=(1, 0),
    )
    cand_hi_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int64 if repair else cutlass.Uint8,
        (cute.sym_int(), block_q, cute.sym_int()) if repair else (1, 1, 1),
        stride_order=(2, 1, 0),
    )
    cand_cnt_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (cute.sym_int(), 2), stride_order=(1, 0)
    )
    fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    kernel = MqaProducerKernel(
        block_kv=block_kv,
        num_heads=num_heads,
        head_dim=head_dim,
        num_sms=num_sms,
        epi_dtype=epi_dtype,
        remove_online_sf_transpose=remove_online_sf_transpose,
        qkv_dtype=qkv_dtype,
        top_k=top_k,
        config=config,
        repair=repair,
        materialize_scores=materialize_scores,
    )
    compiled = cute.compile(
        kernel,
        *inputs_fake,
        logits_fake,
        topk_fake,
        cand_fake,
        cand_hi_fake,
        cand_cnt_fake,
        *make_bounds_fakes(),
        cutlass.Int32(1),
        cutlass.Int32(1),
        (
            cute.runtime.make_fake_compact_tensor(
                offset_dtype, (sym_nq,), stride_order=(0,)
            )
            if offset_dtype is not None
            else None
        ),
        fake_stream,
        options="--enable-tvm-ffi",
    )
    return compiled


def _prepare_workspace(
    *,
    q,
    max_kv_tokens,
    score_dtype,
    num_sms,
    top_k,
    qkv_dtype,
    config,
    repair,
    materialize_scores,
    workspace,
    topk_indices,
):
    """Allocate or reuse producer outputs and expose any repair scratch view.

    Args:
        q: Query tensor supplying the live row count, head count and allocation device.
        max_kv_tokens: Number of diagnostic score columns before allocation alignment.
        score_dtype: Torch FP32/BF16 dtype for the threshold or diagnostic score buffer.
        num_sms: Number of SMs available to the persistent launch grid.
        top_k: Output index count per query and candidate-capacity input.
        qkv_dtype: Packed input format, ``"fp8"``, ``"mxfp4"`` or ``"mxfp8"``.
        config: Selected producer geometry, including the repair query-block size.
        repair: Whether to reuse the consumed candidate arena as split-repair scratch.
        materialize_scores: Whether to allocate full scores with placeholder candidate
            outputs.
        workspace: Existing producer KernelLaunch whose output buffers are reused, or
            None to allocate.
        topk_indices: Optional int32 top-k output storage used only when no workspace is
            supplied.
    """
    rows = q.shape[0]
    if workspace is not None:
        logits, indices, candidates, scratch, counts = workspace.arguments.args[
            LaunchArg.LOGITS : LaunchArg.CANDIDATE_COUNTS + 1
        ]
    else:
        workspace_rows = 1 if materialize_scores else rows
        score_width = (
            round_up(max_kv_tokens, SPLIT_KV_ROWS) if materialize_scores else 1
        )
        logits = torch.zeros((rows, score_width), device=q.device, dtype=score_dtype)
        indices = (
            topk_indices
            if topk_indices is not None
            else torch.empty(
                (workspace_rows, top_k), device=q.device, dtype=torch.int32
            )
        )
        candidates = torch.empty(
            (workspace_rows, candidate_capacity(top_k, qkv_dtype)),
            device=q.device,
            dtype=torch.int64,
        )
        scratch = (
            None
            if repair
            else torch.empty((1, 1, 1), device=q.device, dtype=torch.uint8)
        )
        count_rows = 1
        if not materialize_scores:
            # Queue entries use repair tiles, independently of filter tile width.
            repair_q = select_config(
                rows, num_sms, num_heads=q.shape[1], repair=True
            ).block_q
            count_rows = rows + (rows + repair_q - 1) // repair_q + 1
        counts = torch.zeros((count_rows, 2), device=q.device, dtype=torch.int32)
    logits = logits[:rows, :max_kv_tokens] if materialize_scores else logits[:rows, :]
    if repair:
        # Success selection has consumed this arena before repair starts.
        slots = candidates.numel() // (config.block_q * (candidates.shape[1] + 1))
        repair_sms = max(1, min(num_sms, slots))
        shape = (
            repair_sms,
            config.block_q,
            candidates.shape[1] + 1 if repair_sms > 1 else 1,
        )
        scratch = candidates.view(-1)[: math.prod(shape)].view(shape)
    row_stride = (
        indices.stride(0)
        if not materialize_scores and indices.stride(0) != top_k
        else 0
    )
    return (logits, indices, candidates, scratch, counts), row_stride


def prepare_launch(
    q: torch.Tensor,
    q_sf: torch.Tensor,
    kv: torch.Tensor,
    kv_sf: torch.Tensor,
    weights: torch.Tensor,
    ks: torch.Tensor,
    ke: torch.Tensor,
    max_kv_tokens: int,
    scores_bf16: bool = False,
    num_sms: int = 148,
    remove_online_sf_transpose: bool = False,
    top_k: int = 1024,
    qkv_dtype: str = "mxfp4",
    *,
    repair=False,
    materialize_scores=False,
    workspace=None,
    topk_indices=None,
    topk_indices_offset=None,
    ordered_bounds=False,
) -> KernelLaunch:
    """Prepare a filter, exact-repair or full-score diagnostic launch.

    Args:
        q: Uint8 Q byte view ``[M, H, 128]`` for FP8/MXFP8 or ``[M, H, 64]`` for MXFP4.
        q_sf: Int32 ``[M, H]`` packed UE8M0 Q scales for MXFP4/MXFP8; an unused tensor
            placeholder for FP8.
        kv: Uint8 KV byte view ``[N, 128]`` for FP8/MXFP8 or ``[N, 64]`` for MXFP4.
        kv_sf: Per-key ``[N]`` scales: FP32 for FP8 or packed int32 UE8M0 for
            MXFP4/MXFP8.
        weights: Head weights ``[M, H]`` with softmax scaling folded in (and Q scaling
            for ordinary FP8); FP32 for FP32 scores, cast to BF16 for BF16 scores.
        ks: Nonnegative int32 ``[M]`` inclusive key starts in full-KV coordinates.
        ke: Int32 ``[M]`` exclusive absolute key ends in full-KV coordinates, between
            each row's start and the full KV row count.
        max_kv_tokens: Host key-extent bound, at least twice top-k for selection;
            diagnostic mode requires ``0 < max_kv_tokens <= N``.
        scores_bf16: Whether score arithmetic uses BF16 instead of FP32.
        num_sms: Number of SMs available to the persistent launch grid.
        remove_online_sf_transpose: Whether to pre-transpose MXFP4/MXFP8 K scale chunks
            during preparation and skip the device transpose.
        top_k: Number of indices selected per query, from 128 through 2048.
        qkv_dtype: Packed input format, ``"fp8"``, ``"mxfp4"`` or ``"mxfp8"``.
        repair: Whether to rescan failed rows for exact selection using the filter
            workspace.
        materialize_scores: Whether to emit full diagnostic scores instead of
            candidates; incompatible with repair.
        workspace: Prepared filter launch whose buffers are shared with repair, or None
            to allocate.
        topk_indices: Optional int32 ``[M, top_k]`` storage with contiguous columns,
            used when workspace is None.
        topk_indices_offset: Optional int32/int64 ``[M]`` offsets added only to valid
            output indices; None keeps row-local indices and padding remains -1.
        ordered_bounds: Whether end bounds are nondecreasing within every query tile
            after window trimming, including on every graph replay.
    """
    epi_dtype = cutlass.BFloat16 if scores_bf16 else cutlass.Float32
    if not materialize_scores:
        validate_topk(top_k)
    if not materialize_scores and max_kv_tokens < 2 * top_k:
        raise ValueError("max_kv_tokens must be at least twice top_k")
    q_3d, q_sf_flat, w_flat = prepare_input_views(
        q=q,
        q_sf=q_sf,
        kv=kv,
        kv_sf=kv_sf,
        weights=weights,
        ks=ks,
        ke=ke,
        qkv_dtype=qkv_dtype,
        epi_dtype=epi_dtype,
    )
    num_q_tokens, num_heads, packed_dim = q.shape
    head_dim = packed_dim * 2 if qkv_dtype == "mxfp4" else packed_dim
    num_kv_tokens = kv.shape[0]
    if materialize_scores and (
        num_q_tokens == 0 or not 0 < max_kv_tokens <= num_kv_tokens
    ):
        raise ValueError("Require nonempty Q and 0 < max_kv_tokens <= KV rows")
    block_kv = BLOCK_KV_ROWS
    config = select_config(
        num_q_tokens,
        num_sms,
        materialize_scores=materialize_scores,
        ordered_bounds=ordered_bounds,
        num_heads=num_heads,
        repair=repair,
    )
    # Real KV extents let TMA zero-fill the final partial tile.
    if remove_online_sf_transpose:
        num_kv_blocks = (num_kv_tokens + block_kv - 1) // block_kv
        kv_sf = (
            _apply_utccp_chunk_layout(kv_sf.view(num_kv_blocks, block_kv))
            .reshape(-1)
            .contiguous()
        )
    workspace_args, topk_row_stride = _prepare_workspace(
        q=q,
        max_kv_tokens=max_kv_tokens,
        score_dtype=score_epilogue(epi_dtype).torch_dtype,
        num_sms=num_sms,
        top_k=top_k,
        qkv_dtype=qkv_dtype,
        config=config,
        repair=repair,
        materialize_scores=materialize_scores,
        workspace=workspace,
        topk_indices=topk_indices,
    )
    compiled = _compile_kernel(
        block_kv,
        num_heads,
        head_dim,
        num_sms,
        epi_dtype,
        remove_online_sf_transpose=remove_online_sf_transpose,
        qkv_dtype=qkv_dtype,
        top_k=top_k,
        config=config,
        repair=repair,
        materialize_scores=materialize_scores,
        topk_row_stride=topk_row_stride,
        offset_dtype=(
            None
            if topk_indices_offset is None
            else (
                cutlass.Int32
                if topk_indices_offset.dtype == torch.int32
                else cutlass.Int64
            )
        ),
    )
    arguments = testing.JitArguments(
        kv,
        kv_sf,
        q_3d,
        q_sf_flat,
        w_flat,
        *workspace_args,
        ks,
        ke,
        num_q_tokens,
        num_kv_tokens,
        topk_indices_offset,
    )
    return KernelLaunch(compiled, arguments)


@cute.jit
def global_append(score, hit, index, row, cap: cutlass.Constexpr, count, out):
    """Append ballot-selected candidates to a bounded global warp partition.

    Args:
        score: FP32 score contributed by the calling lane.
        hit: Whether the lane's candidate should be retained.
        index: Row-local key index packed with the score key.
        row: Live query row selecting the global candidate arena.
        cap: Total candidate capacity divided into four warp partitions.
        count: Current count for this warp, advanced even when its partition overflows.
        out: Int64 global candidate arena; each partition reserves its last entry for a
            count.
    """
    lane = cute.arch.lane_idx()
    warp = cute.arch.thread_idx()[0] % 128 // 32
    mask = cute.arch.vote_ballot_sync(hit)
    total = cutlass.Int32(cute.arch.popc(mask))
    if total > 0:
        rank = cute.arch.popc(mask & ((cutlass.Uint32(1) << lane) - 1))
        pos = count + rank
        if hit & (pos < cap // 4 - 1):
            out[row, warp * (cap // 4) + pos] = cutlass.Int64(
                cutlass.Uint64(topk.score_key(score)) << 32
                | cutlass.Uint64(cutlass.Uint32(index))
            )
        count += total
    return count


@cute.jit
def bounded_append(
    score, hit, index, slot: cutlass.Constexpr, cap: cutlass.Constexpr, entries, count
):
    """Append ballot-selected candidates to a bounded shared warp partition.

    Args:
        score: FP32 score contributed by the calling lane.
        hit: Whether the lane's candidate should be retained.
        index: Row-local key index packed with the score key.
        slot: Query slot selecting four shared warp partitions.
        cap: Total candidate capacity divided into four warp partitions.
        entries: Shared uint64 packed-candidate storage with ``cap`` entries per slot.
        count: Current count for this warp, advanced even when its partition overflows.
    """
    lane = cute.arch.lane_idx()
    warp = cute.arch.thread_idx()[0] % 128 // 32
    mask = cute.arch.vote_ballot_sync(hit)
    total = cutlass.Int32(cute.arch.popc(mask))
    if total > 0:
        rank = cute.arch.popc(mask & (cutlass.Uint32(1) << lane) - 1)
        pos = count + rank
        if hit & (pos < cap // 4):
            entries[slot * cap + warp * (cap // 4) + pos] = cutlass.Uint64(
                topk.score_key(score)
            ) << 32 | cutlass.Uint64(cutlass.Uint32(index))
        count = count + total
    return count


@cute.jit
def partition_dump(
    tid,
    row,
    slot: cutlass.Constexpr,
    cap: cutlass.Constexpr,
    entries,
    counts,
    out,
    status,
):
    """Compact four shared warp partitions into a global candidate row.

    Args:
        tid: Thread index within the participating 128-thread warpgroup.
        row: Live query row receiving candidates and metadata.
        slot: Query slot selecting shared candidate and count segments.
        cap: Total candidate capacity divided into four equal warp partitions.
        entries: Shared uint64 packed score-key/index candidates.
        counts: Shared unbounded candidate counts, four per query slot.
        out: Global int64 candidate arena receiving the stored entries contiguously.
        status: Global int32 ``[filled, marker]`` metadata; overflow writes marker
            ``-total - 1``.
    """
    warp = tid // 32
    lane = tid % 32
    total = cutlass.Int32(0)
    filled = cutlass.Int32(0)
    prefix = cutlass.Int32(0)
    overflow = cutlass.Boolean(False)
    for w in cutlass.range_constexpr(4):
        count = counts[slot * 4 + w]
        total += count
        bounded = min(count, cap // 4)
        filled += bounded
        if w < warp:
            prefix += bounded
        overflow |= count > cap // 4
    own = min(counts[slot * 4 + warp], cap // 4)
    for j in cutlass.range(lane, own, 32):
        out[row, prefix + j] = cutlass.Int64(
            entries[slot * cap + warp * (cap // 4) + j]
        )
    if tid == 0:
        status[row, 0] = filled
        marker = total
        if overflow:
            # A full warp partition requires repair even if the total fits.
            marker = -total - 1
        status[row, 1] = marker
