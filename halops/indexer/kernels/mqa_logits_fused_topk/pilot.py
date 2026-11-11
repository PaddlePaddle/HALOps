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
import cutlass.pipeline as pipeline
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass import Float4E2M1FN, Float8E8M0FNU, testing
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from .config import (
    BLOCK_KV_ROWS,
    DEFAULT_KV_SAMPLE,
    SPLIT_KV_ROWS,
    SUPPORTED_HEAD_DIM,
    SamplingConfig,
    round_up,
    select_config,
)
from .epilogue import score_epilogue
from .geometry import MqaMmaGeometry, QueryBlockBounds
from .kernel_utils import (
    _apply_utccp_chunk_layout,
    _subview,
    _topk_red_add,
    coarse_pair,
    utccp_required_smem_warp_transpose,
    warp_scan,
)
from .launch import (
    KernelLaunch,
    make_bounds_fakes,
    make_input_fakes,
    prepare_input_views,
)

_TK_BAR_WG0 = 9
_TK_BAR_FIN = 13


class MqaPilotKernel(MqaMmaGeometry):
    """Pilot kernel that estimates a score threshold from sampled KV rows."""

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
        sample_group: int = DEFAULT_KV_SAMPLE[0],
        sample_take: int = DEFAULT_KV_SAMPLE[1],
        *,
        config,
    ):
        """Configure pilot MMA geometry, score arithmetic and KV sampling.

        Args:
            block_kv: Number of key rows in one MMA tile, fixed at 128.
            num_heads: Logical query head count: 8, 16, 32, or 64.
            head_dim: Unpacked dimension of each query/key head, fixed at 128.
            num_sms: Number of SMs available to the persistent launch grid.
            epi_dtype: Score arithmetic type, ``cutlass.Float32`` or
                ``cutlass.BFloat16``.
            remove_online_sf_transpose: Whether MXFP4/MXFP8 K scales use the
                pre-transposed 128-row chunk layout.
            top_k: One-based rank in the sampled scores used to choose the filter
                cutoff.
            qkv_dtype: Packed input format, ``"fp8"``, ``"mxfp4"`` or ``"mxfp8"``.
            sample_group: Number of full KV rows in each sampling group.
            sample_take: Number of consecutive keys sampled from the start of each
                group.
            config: Compile-time query tiling, histogram, accumulator and register
                policy.
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
        if not self.is_mx_sf and self.block_q >= 16 and self.pilot_bits > 10:
            self.num_kv_stages = 1
        # Sample `take` consecutive keys per group. Direct TMA requires at
        # least four 4-byte scales per run; smaller runs use SampleKV.
        sampling = SamplingConfig(sample_group, sample_take)
        self.bounds = QueryBlockBounds(
            self.block_q,
            self.block_kv,
            sampling.group,
            sampling.take,
            ordered=config.ordered_bounds,
        )
        self.sample_group = sample_group
        self.sample_take = sample_take
        self.sample_runs = self.block_kv // sample_take
        # 4 bytes per scale: int32 UE8M0 quad (MXFP4) or FP32 (FP8 K scales).
        self.sample_direct = sampling.uses_direct_sampling
        # TMA tile shapes: the sampled tile is still 128 rows, but in direct
        # mode those rows arrive as (sample_take, sample_runs) instead of a
        # single contiguous run.
        if self.sample_direct:
            self.a_tma_tiler = ((sample_take, self.sample_runs), self.head_dim)
            self.sf_tma_tiler = ((sample_take, self.sample_runs),)
        else:
            self.a_tma_tiler = (self.block_kv, self.head_dim)
            self.sf_tma_tiler = (self.block_kv,)

    @cute.jit
    def __call__(
        self,
        kv: cute.Tensor,  # A operand bytes
        kv_sf: cute.Tensor,  # per-token K scales
        b: cute.Tensor,  # B operand bytes (Q)
        q_sf: cute.Tensor,  # Q block scales (MXFP4/MXFP8)
        weights: cute.Tensor,  # head weights
        logits: cute.Tensor,  # [M,1]: pilot writes tau here, filter reads it
        ks: cute.Tensor,  # ks; every read is clamped to M-1
        ke: cute.Tensor,  # ke
        num_q_tokens: cutlass.Int32,  # M: row masks and the task loop bound
        num_kv_tokens: cutlass.Int32,  # padded N (pilot: *sampled* row count)
        stream: cuda.CUstream,
    ):
        """Build TMA descriptors and launch sampled-score threshold estimation.

        Args:
            kv: Uint8 KV byte view ``[N, 128]`` for FP8/MXFP8 or ``[N, 64]`` for MXFP4.
                Holds full KV for direct sampling or a dense sample for gathered mode.
            kv_sf: Per-key ``[N]`` scales: FP32 for FP8 or packed int32 UE8M0 for
                MXFP4/MXFP8. Uses the same full or gathered row order as ``kv``.
            b: Flattened Q byte view ``[M * H, packed_width, 1]`` used as MMA operand B.
            q_sf: Flattened int32 ``[M * H]`` Q scales; unused for FP8.
            weights: Flattened ``[M * H]`` head weights in the score dtype.
            logits: Score-dtype ``[M, 1]`` cutoff output consumed by the filter.
            ks: Nonnegative int32 ``[M]`` inclusive key starts in full-KV coordinates.
            ke: Int32 ``[M]`` exclusive absolute key ends in full-KV coordinates,
                between each row's start and the full KV row count.
            num_q_tokens: Number M of live query rows.
            num_kv_tokens: Declared sampled-row count, aligned to whole 128-row tiles.
            stream: CUDA stream on which to enqueue the kernel.
        """
        head_dim_bytes = self.head_dim // 2 if self.is_fp4 else self.head_dim
        kv_data = cute.recast_tensor(
            kv, Float4E2M1FN if self.is_fp4 else cutlass.Float8E4M3FN
        )
        b = cute.recast_tensor(b, Float4E2M1FN if self.is_fp4 else cutlass.Float8E4M3FN)
        num_kv_blocks = num_kv_tokens // self.block_kv
        if cutlass.const_expr(self.sample_direct):
            # `num_kv_tokens` counts *sampled* rows; the strides walk the
            # original KV, so TMA reads the sample in place. Mode 0 is
            # (row in run, run), i.e. sample_take contiguous rows every
            # sample_group rows, sample_runs runs per 128-row tile.
            kv_layout = cute.make_layout(
                ((self.sample_take, self.sample_runs), self.head_dim, num_kv_blocks),
                stride=(
                    (self.head_dim, self.sample_group * self.head_dim),
                    1,
                    self.sample_runs * self.sample_group * self.head_dim,
                ),
            )
            kv_sf_layout = cute.make_layout(
                ((self.sample_take, self.sample_runs), num_kv_blocks),
                stride=(
                    (1, self.sample_group),
                    self.sample_runs * self.sample_group,
                ),
            )
        else:
            kv_layout = cute.make_layout(
                (self.block_kv, self.head_dim, num_kv_blocks),
                stride=(self.head_dim, 1, self.block_kv * self.head_dim),
            )
            kv_sf_layout = cute.make_layout(
                (self.block_kv, num_kv_blocks), stride=(1, self.block_kv)
            )
        a = cute.make_tensor(kv_data.iterator, kv_layout)
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
            tma_load_op, a, a_smem_per_stage, self.a_tma_tiler
        )
        self.kv_sf_smem_layout_staged = cute.make_layout(
            (
                self.block_kv,
                self.num_sc_slots if not self.is_mx_sf else self.num_kv_stages,
            )
        )
        kv_sf_smem_per_block = cute.make_layout((self.block_kv,))
        tma_atom_kv_sf, tma_tensor_kv_sf = cpasync.make_tiled_tma_atom(
            tma_load_op, kv_sf, kv_sf_smem_per_block, self.sf_tma_tiler
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
        num_ctas = self.num_sms

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
            ks,
            ke,
            num_q_tokens,
            num_q_blocks,
            num_kv_tokens,
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
        mCuSeqLenKStart: cute.Tensor,
        mCuSeqLenKEnd: cute.Tensor,
        num_q_tokens: cutlass.Int32,
        num_q_blocks: cutlass.Int32,
        num_kv_tokens: cutlass.Int32,
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
        """Score sampled keys and write one histogram-derived cutoff per query.

        Args:
            tiled_mma: Tiled FP8 or block-scaled MXFP4 MMA operation.
            tma_atom_a: TMA copy atom loading key operand A into shared memory.
            mA_mkl: Recast sampled-key TMA view over the original KV buffer or a dense
                gathered sample.
            tma_atom_b: TMA copy atom loading query operand B into shared memory.
            mB_nkl: Recast query operand with the layout used by its TMA descriptor.
            tma_atom_w: TMA copy atom loading head weights.
            mW_tma: Global head-weight tensor addressed by the weight TMA descriptor.
            tma_atom_kv_sf: TMA copy atom loading per-key scales.
            mSF_KV_tma: Global FP32 or packed UE8M0 key-scale tensor.
            tma_atom_q_sf: TMA copy atom loading MXFP4/MXFP8 query scales; unused for
                FP8.
            mSF_Q_tma: Global packed UE8M0 query-scale tensor; unused for FP8.
            mLogits: Global score-dtype ``[M, 1]`` cutoff output.
            mCuSeqLenKStart: Nonnegative int32 ``[M]`` inclusive key starts in full-KV
                coordinates.
            mCuSeqLenKEnd: Int32 ``[M]`` exclusive absolute key ends in full-KV
                coordinates, between each row's start and the full KV row count.
            num_q_tokens: Number M of live query rows.
            num_q_blocks: Number of query tiles, including a possible partial final
                tile.
            num_kv_tokens: Declared sampled-row extent used to clamp transformed key
                bounds.
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
        sm_idx = bidz
        start_q_block = sm_idx

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
            bidx,
            block_kv_val,
            block_q_val,
            cC,
            copy_atom_t2r,
            epi_sub_mn,
            local_tidx,
            mCuSeqLenKEnd,
            mCuSeqLenKStart,
            mLogits,
            num_epi_subtiles,
            num_heads,
            num_q_blocks,
            num_q_tokens,
            num_kv_tokens,
            q_cons_state,
            q_pipeline,
            sSF_KV_0,
            sSF_KV_1,
            sW,
            s_score_hist,
            s_stream_scalars,
            tCtAcc_base_0,
            tCtAcc_base_1,
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
                        kv_start_cache[t_i] = self.bounds.coordinate(
                            mCuSeqLenKStart[token_idx], num_kv_tokens
                        )
                        kv_end_cache[t_i] = self.bounds.coordinate(
                            mCuSeqLenKEnd[token_idx], num_kv_tokens
                        )
                    kv_lo_cur = self.bounds.tile_start(
                        mCuSeqLenKStart,
                        mCuSeqLenKEnd,
                        q_block_idx,
                        num_q_tokens,
                        bidx,
                        self.hist_splits,
                        num_kv_tokens,
                    )
                prev_q_block_idx = q_block_idx
                for zero_i in range(
                    local_tidx, self.queries_per_group * (1 << self.pilot_bits), 128
                ):
                    s_score_hist[
                        _TK_WG * self.queries_per_group * (1 << self.pilot_bits)
                        + zero_i
                    ] = 0
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
                            pilot_add(
                                cutlass.Float32(result_arr[t]),
                                (
                                    q_block_idx * block_q_val + t < num_q_tokens
                                    and kv_pos >= kv_start_cache[t]
                                    and (kv_pos < kv_end_cache[t])
                                )
                                and kv_pos < num_kv * block_kv_val,
                                t,
                                s_score_hist,
                                self.pilot_bits,
                            )
                        if cutlass.const_expr(kv_group == 0):
                            umma_cons_state_0 = umma_cons_state
                        else:
                            umma_cons_state_1 = umma_cons_state
                    if cutlass.const_expr(not self.is_mx_sf):
                        sc_slot = (sc_slot + 1) % self.num_sc_slots
                    kv_idx = kv_idx + NUM_MATH_WG
                    kv_go = kv_idx < num_kv
                cute.arch.barrier(
                    barrier_id=_TK_BAR_FIN, number_of_threads=self.num_math_threads
                )
                for query in cutlass.range_constexpr(
                    _TK_WG * self.queries_per_group,
                    (_TK_WG + 1) * self.queries_per_group,
                ):
                    stream_row = q_block_idx * block_q_val + query
                    if stream_row < num_q_tokens:
                        pilot_cutoff(
                            local_tidx,
                            stream_row,
                            query,
                            self.top_k,
                            s_score_hist,
                            s_stream_scalars,
                            mLogits,
                            self.pilot_bits,
                            _TK_BAR_WG0 + _TK_WG,
                        )
                cute.arch.barrier(
                    barrier_id=_TK_BAR_FIN, number_of_threads=self.num_math_threads
                )
                next_q_block_idx = q_block_idx + self.num_sms
                if next_q_block_idx < num_q_blocks:
                    next_num_kv = self.bounds.tile_end(
                        mCuSeqLenKEnd,
                        next_q_block_idx,
                        num_q_tokens,
                        bidx,
                        self.hist_splits,
                        num_kv_tokens,
                    )
                has_work = next_q_block_idx < num_q_blocks
            if q_block_idx < num_q_blocks:
                q_pipeline.consumer_release(q_cons_state)
                q_cons_state.advance()

        start_q_block_clamped = min(start_q_block, num_q_blocks - 1)
        current_num_kv = self.bounds.tile_end(
            mCuSeqLenKEnd,
            start_q_block_clamped,
            num_q_tokens,
            bidx,
            self.hist_splits,
            num_kv_tokens,
        )

        current_kv_lo = self.bounds.tile_start(
            mCuSeqLenKStart,
            mCuSeqLenKEnd,
            start_q_block_clamped,
            num_q_tokens,
            bidx,
            self.hist_splits,
            num_kv_tokens,
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
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)
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
        s_score_hist = smem.allocate_tensor(
            cutlass.Int32,
            cute.make_layout((self.block_q * (1 << self.pilot_bits),)),
            byte_alignment=128,
        )
        s_stream_scalars = smem.allocate_tensor(
            cutlass.Int32,
            cute.make_layout((self.block_q * ((1 << self.pilot_bits) // 32 + 8),)),
            byte_alignment=128,
        )
        a_mcast_mask = cpasync.create_tma_multicast_mask(
            cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
        )
        b_mcast_mask = cpasync.create_tma_multicast_mask(
            cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
        )
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        gA = cute.local_tile(mA_mkl, self.a_tma_tiler, coord=(None, None, None))
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
        # mW_tma retains the true flat extent; local_tile adds the q-block mode.
        tWsW, tWgW = cpasync.tma_partition(
            tma_atom_w,
            0,
            w_cta_layout,
            cute.group_modes(sW, 0, 1),
            cute.group_modes(
                cute.local_tile(mW_tma, (self.UMMA_N,), (None,)),
                0,
                1,
            ),
        )
        # Both formats share the TMA path. The allocation's full layout keeps
        # FP8's epilogue scale ring distinct from the MMA-owned KV stages.
        gSF_KV = cute.local_tile(mSF_KV_tma, self.sf_tma_tiler, coord=(None, None))
        tSsSF_KV_0, tSgSF_KV_0 = cpasync.tma_partition(
            tma_atom_kv_sf, 0, cute.make_layout(1), sSF_KV_0, gSF_KV
        )
        tSsSF_KV_1, tSgSF_KV_1 = cpasync.tma_partition(
            tma_atom_kv_sf, 0, cute.make_layout(1), sSF_KV_1, gSF_KV
        )
        sc_slot = cutlass.Int32(0)
        sSF_Q_for_tma = cute.make_tensor(sSF_Q.iterator, q_sf_tma_smem_layout_staged)
        q_sf_cta_layout = cute.make_layout((1,))
        tSF_Q_sSF_Q, tSF_Q_gSF_Q = cpasync.tma_partition(
            tma_atom_q_sf,
            0,
            q_sf_cta_layout,
            cute.group_modes(sSF_Q_for_tma, 0, 1),
            cute.group_modes(
                cute.local_tile(mSF_Q_tma, (self.UMMA_N,), (None,)),
                0,
                1,
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
            cute.arch.warpgroup_reg_dealloc(48 if self.queries_per_group > 1 else 24)
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
                    prefetch_next_q = q_block_idx + self.num_sms
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
                    tAgA_0[None, 0, 0, kv_idx],
                    tAsA_0[None, stage],
                    tma_bar_ptr=bar,
                    mcast_mask=a_mcast_mask,
                )
                sf_slot = sc_slot if cutlass.const_expr(not self.is_mx_sf) else stage
                cute.copy(
                    tma_atom_kv_sf,
                    tSgSF_KV_0[None, 0, kv_idx],
                    tSsSF_KV_0[None, sf_slot],
                    tma_bar_ptr=bar,
                )
                if cutlass.const_expr(not self.is_mx_sf):
                    sc_slot = (sc_slot + 1) % self.num_sc_slots
                kv_prod_state_0.advance()
                next_kv_idx = kv_idx + NUM_MATH_WG
                if next_kv_idx >= num_kv:
                    next_q_block_idx = q_block_idx + self.num_sms
                    next_kv_idx = self.bounds.tile_start(
                        mCuSeqLenKStart,
                        mCuSeqLenKEnd,
                        next_q_block_idx,
                        num_q_tokens,
                        bidx,
                        self.hist_splits,
                        num_kv_tokens,
                    )
                    if next_q_block_idx < num_q_blocks:
                        next_num_kv = self.bounds.tile_end(
                            mCuSeqLenKEnd,
                            next_q_block_idx,
                            num_q_tokens,
                            bidx,
                            self.hist_splits,
                            num_kv_tokens,
                        )
                has_work = next_q_block_idx < num_q_blocks
        elif is_tma_warp_1:
            cute.arch.warpgroup_reg_dealloc(48 if self.queries_per_group > 1 else 24)
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
                    tAgA_1[None, 0, 0, kv_idx_1],
                    tAsA_1[None, stage],
                    tma_bar_ptr=bar,
                    mcast_mask=a_mcast_mask,
                )
                sf_slot = sc_slot if cutlass.const_expr(not self.is_mx_sf) else stage
                cute.copy(
                    tma_atom_kv_sf,
                    tSgSF_KV_1[None, 0, kv_idx_1],
                    tSsSF_KV_1[None, sf_slot],
                    tma_bar_ptr=bar,
                )
                if cutlass.const_expr(not self.is_mx_sf):
                    sc_slot = (sc_slot + 1) % self.num_sc_slots
                kv_prod_state_1.advance()
                next_kv_idx = kv_idx + NUM_MATH_WG
                if next_kv_idx >= num_kv:
                    next_q_block_idx = q_block_idx + self.num_sms
                    next_kv_idx = self.bounds.tile_start(
                        mCuSeqLenKStart,
                        mCuSeqLenKEnd,
                        next_q_block_idx,
                        num_q_tokens,
                        bidx,
                        self.hist_splits,
                        num_kv_tokens,
                    )
                    if next_q_block_idx < num_q_blocks:
                        next_num_kv = self.bounds.tile_end(
                            mCuSeqLenKEnd,
                            next_q_block_idx,
                            num_q_tokens,
                            bidx,
                            self.hist_splits,
                            num_kv_tokens,
                        )
                has_work = next_q_block_idx < num_q_blocks
        elif is_umma_warp_0:
            cute.arch.warpgroup_reg_dealloc(48 if self.queries_per_group > 1 else 24)
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
                            stage_off_ue8m0 = q_stage_0 * sSF_Q_ue8m0.layout.stride[1]
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
                            tCsSFB_s2t_ = thr_copy_s2t_sfb.partition_S(tCsSFB_compact)
                            tCsSFB_s2t = tcgen05.get_s2t_smem_desc_tensor(
                                tiled_copy_s2t_sfb, tCsSFB_s2t_
                            )
                            tCtSFB_s2t = thr_copy_s2t_sfb.partition_D(tCtSFB_compact)
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
                        stage_off_kv0_ue8m0 = kv_stage * sSF_KV_0_ue8m0.layout.stride[1]
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
                    tCtAcc_0 = tCtAcc_base_0[None, None, None, umma_prod_state_0.index]
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
                        next_q_block_idx = q_block_idx + self.num_sms
                        next_kv_idx = self.bounds.tile_start(
                            mCuSeqLenKStart,
                            mCuSeqLenKEnd,
                            next_q_block_idx,
                            num_q_tokens,
                            bidx,
                            self.hist_splits,
                            num_kv_tokens,
                        )
                        if next_q_block_idx < num_q_blocks:
                            next_num_kv = self.bounds.tile_end(
                                mCuSeqLenKEnd,
                                next_q_block_idx,
                                num_q_tokens,
                                bidx,
                                self.hist_splits,
                                num_kv_tokens,
                            )
                    has_work = next_q_block_idx < num_q_blocks
        elif is_umma_warp_1:
            cute.arch.warpgroup_reg_dealloc(48 if self.queries_per_group > 1 else 24)
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
                                stage_offset = kv_stage_1 * sSF_KV_1.layout.stride[1]
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
                        tCsSFA_s2t_1_ = thr_copy_s2t_sfa_1.partition_S(tCsSFA_compact_1)
                        tCsSFA_s2t_1 = tcgen05.get_s2t_smem_desc_tensor(
                            tiled_copy_s2t_sfa_1, tCsSFA_s2t_1_
                        )
                        tCtSFA_s2t_1 = thr_copy_s2t_sfa_1.partition_D(tCtSFA_compact_1)
                        cute.copy(tiled_copy_s2t_sfa_1, tCsSFA_s2t_1, tCtSFA_s2t_1)
                    tCtAcc_1 = tCtAcc_base_1[None, None, None, umma_prod_state_1.index]
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
                        next_q_block_idx = q_block_idx + self.num_sms
                        next_kv_idx = self.bounds.tile_start(
                            mCuSeqLenKStart,
                            mCuSeqLenKEnd,
                            next_q_block_idx,
                            num_q_tokens,
                            bidx,
                            self.hist_splits,
                            num_kv_tokens,
                        )
                        if next_q_block_idx < num_q_blocks:
                            next_num_kv = self.bounds.tile_end(
                                mCuSeqLenKEnd,
                                next_q_block_idx,
                                num_q_tokens,
                                bidx,
                                self.hist_splits,
                                num_kv_tokens,
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
                        bidx=bidx,
                        block_kv_val=block_kv_val,
                        block_q_val=block_q_val,
                        cC=cC,
                        copy_atom_t2r=copy_atom_t2r,
                        epi_sub_mn=epi_sub_mn,
                        local_tidx=local_tidx,
                        mCuSeqLenKEnd=mCuSeqLenKEnd,
                        mCuSeqLenKStart=mCuSeqLenKStart,
                        mLogits=mLogits,
                        num_epi_subtiles=num_epi_subtiles,
                        num_heads=num_heads,
                        num_q_blocks=num_q_blocks,
                        num_q_tokens=num_q_tokens,
                        num_kv_tokens=num_kv_tokens,
                        q_cons_state=q_cons_state,
                        q_pipeline=q_pipeline,
                        sSF_KV_0=sSF_KV_0,
                        sSF_KV_1=sSF_KV_1,
                        sW=sW,
                        s_score_hist=s_score_hist,
                        s_stream_scalars=s_stream_scalars,
                        tCtAcc_base_0=tCtAcc_base_0,
                        tCtAcc_base_1=tCtAcc_base_1,
                        umma_cons_state_0=umma_cons_state_0,
                        umma_cons_state_1=umma_cons_state_1,
                        umma_pipeline_0=umma_pipeline_0,
                        umma_pipeline_1=umma_pipeline_1,
                    )
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)
        else:
            cute.arch.warpgroup_reg_dealloc(48 if self.queries_per_group > 1 else 24)


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
    sample_group: int = DEFAULT_KV_SAMPLE[0],
    sample_take: int = DEFAULT_KV_SAMPLE[1],
    *,
    config,
):
    """Compile and cache a pilot specialization with dynamic tensor row counts.

    Args:
        block_kv: Number of key rows in one MMA tile, fixed at 128.
        num_heads: Logical query head count: 8, 16, 32, or 64.
        head_dim: Unpacked dimension of each query/key head, fixed at 128.
        num_sms: Number of SMs available to the persistent launch grid.
        epi_dtype: Score arithmetic type, ``cutlass.Float32`` or ``cutlass.BFloat16``.
        remove_online_sf_transpose: Whether MXFP4/MXFP8 K scales use the pre-transposed
            128-row chunk layout.
        top_k: One-based rank in the sampled scores used to choose the filter cutoff.
        qkv_dtype: Packed input format, ``"fp8"``, ``"mxfp4"`` or ``"mxfp8"``.
        sample_group: Number of full KV rows in each sampling group.
        sample_take: Number of consecutive keys sampled from the start of each group.
        config: Compile-time query tiling, histogram, accumulator and register policy.
    """
    sym_nq = cute.sym_int()
    inputs_fake = make_input_fakes(num_heads, head_dim, qkv_dtype, epi_dtype, sym_nq)
    epi = score_epilogue(epi_dtype)
    logits_fake = cute.runtime.make_fake_tensor(
        epi.dtype, (sym_nq, cute.sym_int()), stride=(cute.sym_int64(), 1)
    )
    fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    kernel = MqaPilotKernel(
        block_kv=block_kv,
        num_heads=num_heads,
        head_dim=head_dim,
        num_sms=num_sms,
        epi_dtype=epi_dtype,
        remove_online_sf_transpose=remove_online_sf_transpose,
        qkv_dtype=qkv_dtype,
        top_k=top_k,
        sample_group=sample_group,
        sample_take=sample_take,
        config=config,
    )
    compiled = cute.compile(
        kernel,
        *inputs_fake,
        logits_fake,
        *make_bounds_fakes(),
        cutlass.Int32(1),
        cutlass.Int32(1),
        fake_stream,
        options="--enable-tvm-ffi",
    )
    return compiled


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
    sample_group: int = DEFAULT_KV_SAMPLE[0],
    sample_take: int = DEFAULT_KV_SAMPLE[1],
    ordered_bounds=False,
    *,
    threshold=None,
) -> KernelLaunch:
    """Prepare a pilot launch that estimates one score cutoff per query row.

    Args:
        q: Uint8 Q byte view ``[M, H, 128]`` for FP8/MXFP8 or ``[M, H, 64]`` for MXFP4.
        q_sf: Int32 ``[M, H]`` packed UE8M0 Q scales for MXFP4/MXFP8; an unused tensor
            placeholder for FP8.
        kv: Uint8 KV byte view ``[N, 128]`` for FP8/MXFP8 or ``[N, 64]`` for MXFP4.
            Supply the full buffer for direct sampling or the dense gathered sample
            otherwise.
        kv_sf: Per-key ``[N]`` scales: FP32 for FP8 or packed int32 UE8M0 for
            MXFP4/MXFP8. Match the full or gathered KV row order.
        weights: Head weights ``[M, H]`` with softmax scaling folded in (and Q scaling
            for ordinary FP8); FP32 for FP32 scores, cast to BF16 for BF16 scores.
        ks: Nonnegative int32 ``[M]`` inclusive key starts in full-KV coordinates.
        ke: Int32 ``[M]`` exclusive absolute key ends in full-KV coordinates, between
            each row's start and the full KV row count.
        max_kv_tokens: Host bound on the full key extent, at least twice ``top_k``.
        scores_bf16: Whether score arithmetic uses BF16 instead of FP32.
        num_sms: Number of SMs available to the persistent launch grid.
        remove_online_sf_transpose: Whether to pre-transpose MXFP4/MXFP8 K scale chunks
            during preparation and skip the device transpose.
        top_k: One-based sampled-score rank used to estimate the cutoff, not the final
            output width.
        qkv_dtype: Packed input format, ``"fp8"``, ``"mxfp4"`` or ``"mxfp8"``.
        sample_group: Number of full KV rows per sampling group, also used to map row
            bounds.
        sample_take: Consecutive keys taken per group; values below four require
            pre-gathered inputs.
        ordered_bounds: Whether end bounds are nondecreasing within every query tile
            after window trimming, including on every graph replay.
        threshold: Optional ``[M, 1]`` cutoff output on Q's device in the score dtype;
            None allocates.
    """
    epi_dtype = cutlass.BFloat16 if scores_bf16 else cutlass.Float32
    if top_k < 1 or max_kv_tokens < 2 * top_k:
        raise ValueError("Pilot requires top_k >= 1 and max_kv_tokens >= 2 * top_k")
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
    block_kv = BLOCK_KV_ROWS
    config = select_config(
        num_q_tokens, num_sms, ordered_bounds=ordered_bounds, num_heads=num_heads
    )
    epi = score_epilogue(epi_dtype)
    if sample_take * 4 >= 16:
        # Direct mode: `kv` is the caller's KV, read in place, nothing padded or
        # copied. Declare only the whole 128-row sampled tiles whose runs lie
        # fully inside the real KV; TMA zero-fills anything past the descriptor
        # and the kernel's clamped sampling coordinate stops there, so the pilot
        # samples at most block_kv - 1 fewer tail keys.
        pilot_kv, pilot_kv_sf = kv, kv_sf
        declared_sampled_rows = SamplingConfig(
            sample_group, sample_take
        ).direct_sampled_rows(num_kv_tokens)
        num_kv_blocks = declared_sampled_rows // block_kv
    else:
        declared_sampled_rows = round_up(num_kv_tokens, SPLIT_KV_ROWS)
        num_kv_blocks = declared_sampled_rows // block_kv
        if declared_sampled_rows > num_kv_tokens:
            pilot_kv = torch.zeros(
                (declared_sampled_rows, packed_dim), dtype=kv.dtype, device=kv.device
            )
            pilot_kv[:num_kv_tokens].copy_(kv)
            pilot_kv_sf = torch.zeros(
                declared_sampled_rows, dtype=kv_sf.dtype, device=kv_sf.device
            )
            pilot_kv_sf[:num_kv_tokens].copy_(kv_sf)
        else:
            pilot_kv = kv
            pilot_kv_sf = kv_sf
    if remove_online_sf_transpose:
        pilot_kv_sf = (
            _apply_utccp_chunk_layout(pilot_kv_sf.view(num_kv_blocks, block_kv))
            .reshape(-1)
            .contiguous()
        )
    if threshold is None:
        threshold = torch.empty(
            (num_q_tokens, 1), device=q.device, dtype=epi.torch_dtype
        )
    elif (
        threshold.shape != (num_q_tokens, 1)
        or threshold.dtype != epi.torch_dtype
        or threshold.device != q.device
        or threshold.stride(1) != 1
        or threshold.stride(0) < 1
    ):
        raise ValueError("threshold must be [M, 1] scores on Q's device")
    compiled = _compile_kernel(
        block_kv,
        num_heads,
        head_dim,
        num_sms,
        epi_dtype,
        remove_online_sf_transpose=remove_online_sf_transpose,
        qkv_dtype=qkv_dtype,
        top_k=top_k,
        sample_group=sample_group,
        sample_take=sample_take,
        config=config,
    )
    arguments = testing.JitArguments(
        pilot_kv,
        pilot_kv_sf,
        q_3d,
        q_sf_flat,
        w_flat,
        threshold,
        ks,
        ke,
        num_q_tokens,
        declared_sampled_rows,
    )
    return KernelLaunch(compiled, arguments)


@cute.jit
def pilot_add(score, hit, slot: cutlass.Constexpr, hist, bits: cutlass.Constexpr = 13):
    """Add a valid sampled score to its query's coarse histogram.

    Args:
        score: FP32 sampled score converted to a descending FP16 key.
        hit: Whether the score belongs to the live query's legal sampled window.
        slot: Query slot selecting the histogram segment.
        hist: Shared int32 histogram with ``1 << bits`` bins per slot.
        bits: Number of high key bits used as the histogram bin, from 10 through 13.
    """
    if hit:
        bucket = cutlass.Int32(
            coarse_pair(score, score) >> (16 - bits) & ((1 << bits) - 1)
        )
        _topk_red_add(hist.iterator + slot * (1 << bits) + bucket, cutlass.Int32(1))


@cute.jit
def pilot_cutoff(
    tid,
    row,
    slot: cutlass.Constexpr,
    rank: cutlass.Constexpr,
    hist,
    scratch,
    tau,
    bits: cutlass.Constexpr = 13,
    barrier: cutlass.Constexpr = 9,
):
    """Select a sampled histogram cutoff and store it for the query row.

    Args:
        tid: Thread index within the participating 128-thread warpgroup.
        row: Live query row receiving the cutoff.
        slot: Query slot selecting histogram and scratch segments.
        rank: One-based sampled-score rank used to choose the cutoff.
        hist: Shared int32 histogram with ``1 << bits`` bins per query slot.
        scratch: Shared int32 prefix scratch with ``(1 << bits) // 32 + 8`` values per
            slot.
        tau: Global score-dtype ``[M, 1]`` cutoff output.
        bits: Number of high score-key bits retained by the histogram.
        barrier: Named barrier ID used by all 128 participating threads.
    """
    lane = cute.arch.lane_idx()
    warp = tid // 32
    if tid == 0:
        scratch[slot * ((1 << bits) // 32 + 8) + 4] = 0
        scratch[slot * ((1 << bits) // 32 + 8) + 5] = 0
    for j in cutlass.range_constexpr((1 << bits) // 128):
        value = hist[slot * (1 << bits) + j * 128 + tid]
        group_sum = pilot_sum(value)
        if lane == 0:
            scratch[slot * ((1 << bits) // 32 + 8) + 8 + j * 4 + warp] = group_sum
    cute.arch.barrier(barrier_id=barrier, number_of_threads=128)
    a = cutlass.Int32(0)
    b = cutlass.Int32(0)
    if tid * 2 < (1 << bits) // 32:
        a = scratch[slot * ((1 << bits) // 32 + 8) + 8 + tid * 2]
        b = scratch[slot * ((1 << bits) // 32 + 8) + 9 + tid * 2]
    total = a + b
    incl = warp_scan(total, lane, num_threads_per_warp=32)
    if lane == 31:
        scratch[slot * ((1 << bits) // 32 + 8) + warp] = incl
    cute.arch.barrier(barrier_id=barrier, number_of_threads=128)
    previous = incl - total
    for w in cutlass.range_constexpr(4):
        if w < warp:
            previous += scratch[slot * ((1 << bits) // 32 + 8) + w]
    if (previous < rank) & (previous + total >= rank):
        selected = tid * 2
        if previous + a < rank:
            selected += 1
            previous += a
        scratch[slot * ((1 << bits) // 32 + 8) + 4] = selected
        scratch[slot * ((1 << bits) // 32 + 8) + 5] = previous
    cute.arch.barrier(barrier_id=barrier, number_of_threads=128)
    if warp == 0:
        group = scratch[slot * ((1 << bits) // 32 + 8) + 4]
        preceding = scratch[slot * ((1 << bits) // 32 + 8) + 5]
        value = hist[slot * (1 << bits) + group * 32 + lane]
        incl_fine = warp_scan(value, lane, num_threads_per_warp=32)
        if (preceding + incl_fine - value < rank) & (preceding + incl_fine >= rank):
            scratch[slot * ((1 << bits) // 32 + 8) + 6] = group * 32 + lane
            scratch[slot * ((1 << bits) // 32 + 8) + 7] = preceding + incl_fine - value
    cute.arch.barrier(barrier_id=barrier, number_of_threads=128)
    if tid == 0:
        count = (
            scratch[slot * ((1 << bits) // 32 + 8)]
            + scratch[slot * ((1 << bits) // 32 + 8) + 1]
            + scratch[slot * ((1 << bits) // 32 + 8) + 2]
            + scratch[slot * ((1 << bits) // 32 + 8) + 3]
        )
        cutoff = cutlass.Float32(-float("inf"))
        if count >= rank:
            bucket = scratch[slot * ((1 << bits) // 32 + 8) + 6]
            low = cutlass.Int32((1 << (16 - bits)) - 1)
            if cutlass.const_expr(bits < 13):
                # Estimate the rank inside a coarse bucket. Taking its entire
                # lower edge can overflow the candidate pool at 10 bits. Any
                # under-estimate is detected and handled by exact repair.
                preceding = scratch[slot * ((1 << bits) // 32 + 8) + 7]
                low = min(
                    low,
                    (rank - preceding)
                    * (1 << (16 - bits))
                    // hist[slot * (1 << bits) + bucket],
                )
            key = cutlass.Uint32(bucket << (16 - bits) | low)
            half_bits = (key ^ cutlass.Uint32(65535)) & cutlass.Uint32(32767)
            if key & 32768:
                half_bits = key
            half = cutlass.Float16(
                llvm.bitcast(
                    cutlass.Float16.mlir_type, cutlass.Uint16(half_bits).ir_value()
                )
            )
            cutoff = cutlass.Float32(half)
        tau[row, 0] = cutoff.to(tau.element_type)


@cute.jit
def pilot_sum(value):
    """Sum an Int32 contribution across all 32 lanes of the current warp.

    Args:
        value: Int32 contribution from the calling lane; all lanes must participate.
    """
    return cutlass.Int32(
        llvm.inline_asm(
            cutlass.Int32.mlir_type,
            [value.ir_value()],
            "redux.sync.add.s32 $0, $1, -1;",
            "=r,r",
            has_side_effects=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )
