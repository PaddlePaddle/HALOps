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

import math

import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.arch import get_max_tmem_alloc_cols
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cutlass_dsl import BaseDSL

from .config import (
    BLOCK_KV_ROWS,
    SUPPORTED_HEAD_DIM,
    SUPPORTED_QKV_DTYPES,
    round_up,
    validate_num_heads,
)
from .epilogue import score_epilogue


class MqaMmaGeometry:
    """Host-only configuration and MMA layouts shared by pilot and producer.

    The subclasses own KV TMA descriptors, shared storage and barrier loops.
    No virtual dispatch or extra CUDA stage is added by this trace-time base.
    """

    def __init__(
        self,
        block_kv,
        num_heads,
        head_dim,
        num_sms,
        epi_dtype,
        remove_online_sf_transpose,
        top_k,
        qkv_dtype,
        *,
        config,
    ):
        """Initialize shared MMA geometry, pipeline sizes and score policy.

        Args:
            block_kv: Number of key rows in one MMA tile, fixed at 128.
            num_heads: Logical query head count: 8, 16, 32, or 64.
            head_dim: Unpacked dimension of each query/key head, fixed at 128.
            num_sms: Number of SMs available to the persistent launch grid.
            epi_dtype: Score arithmetic type, ``cutlass.Float32`` or
                ``cutlass.BFloat16``.
            remove_online_sf_transpose: Whether MXFP4/MXFP8 K scales use the
                pre-transposed 128-row chunk layout.
            top_k: Requested output width for the producer or sampled rank for the
                pilot.
            qkv_dtype: Packed input format, ``"fp8"``, ``"mxfp4"`` or ``"mxfp8"``.
            config: Compile-time query tiling, histogram, accumulator and register
                policy.
        """
        if qkv_dtype not in SUPPORTED_QKV_DTYPES:
            raise ValueError(f"Unsupported Q/KV dtype: {qkv_dtype}")
        if epi_dtype not in (cutlass.Float32, cutlass.BFloat16):
            raise ValueError("Histogram specialization requires FP32 or BF16 scores")
        validate_num_heads(num_heads)
        if head_dim != SUPPORTED_HEAD_DIM:
            raise ValueError(f"head_dim must be {SUPPORTED_HEAD_DIM}; got {head_dim}")
        if block_kv != BLOCK_KV_ROWS:
            raise ValueError(f"block_kv must be {BLOCK_KV_ROWS}; got {block_kv}")
        if (
            config.block_q < 2
            or config.math_groups not in (2, 3)
            or config.block_q % config.math_groups
            or config.block_q * num_heads > 256
        ):
            raise ValueError(
                "block_q must divide evenly over math groups and fit N<=256"
            )
        if not 10 <= config.pilot_bits <= 13:
            raise ValueError("pilot_bits must be between 10 and 13")
        self.epi = score_epilogue(epi_dtype)
        self.is_mx_sf = qkv_dtype != "fp8"
        self.is_fp4 = qkv_dtype == "mxfp4"
        self.block_kv = block_kv
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.block_q = config.block_q
        self.math_groups = config.math_groups
        self.queries_per_group = self.block_q // self.math_groups
        self.pilot_bits = config.pilot_bits
        self.math_regs = (
            {2: 224, 3: 152}[self.math_groups]
            if self.queries_per_group > 1
            else self.epi.math_regs(config.three_wg)
        )
        self.UMMA_N = self.block_q * num_heads
        self.hist_splits = config.hist_splits
        self.num_sms = max(1, num_sms // self.hist_splits)
        self.remove_online_sf_transpose = remove_online_sf_transpose
        self.three_wg = config.three_wg
        self.top_k = top_k
        self.topk_w_in_reg = config.weights
        self.epi_bytes = 2 if epi_dtype == cutlass.BFloat16 else 4
        self.smem_w_stage_stride = (
            round_up(self.UMMA_N * self.epi_bytes, 128) // self.epi_bytes
        )

        # Two alternating KV tiles; config controls how many accumulators they use.
        self.num_groups = 2
        self.single_acc = config.single_acc and self.is_mx_sf
        self.num_acc_groups = 1 if self.single_acc else self.num_groups
        self.num_umma_stages = 1
        # E4M3 operands need twice the tile storage of packed E2M1.
        self.num_kv_stages = 3 if self.is_fp4 else 2
        # FP8 scales live until the epilogue, beyond the KV stage's MMA release.
        # With two KV stages and one accumulator per group, TMA of pair p + 4
        # waits for MMA p + 2, hence math's release of accumulator p + 1.
        # Math has read pair p's scales by then, even though it releases each
        # accumulator before finishing that same pair's scale read.
        self.num_sc_slots = 4
        self.num_q_stages = 2
        self.num_math_threads = 128 * self.math_groups
        self.num_math_warps = self.num_math_threads // 32
        self.threads_per_cta = self.num_math_threads + 128
        self.acc_dtype = cutlass.Float32
        self.sf_dtype = cutlass.Float8E8M0FNU
        self.sf_vec_size = 32
        self.cta_group = tcgen05.CtaGroup.ONE
        self.cluster_shape_mn = (1, 1)
        self.mma_tiler_mn = (block_kv, self.UMMA_N)
        arch = BaseDSL._get_dsl().get_arch_enum()
        self.arch_str = f"sm_{arch.major}{arch.minor}"

    def _setup_q_tma(self, b, q_sf, weights, tiled_mma):
        """Build Q-side TMA descriptors and the shared Q transaction byte count.

        Args:
            b: Recast Q tensor ``[M * H, head_dim, 1]`` used as MMA operand B.
            q_sf: Flattened int32 Q scales; unused placeholder storage for FP8.
            weights: Flattened head weights in the selected score dtype.
            tiled_mma: Tiled MMA defining the Q operand and cluster partitioning.
        """
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp()
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)
        b_dtype = b.element_type
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        self.w_smem_layout_staged = cute.make_layout(
            (self.UMMA_N, self.num_q_stages), stride=(1, self.smem_w_stage_stride)
        )
        w_smem_per_stage = cute.select(self.w_smem_layout_staged, mode=[0])
        tma_atom_w, tma_tensor_w = cpasync.make_tiled_tma_atom(
            tma_load_op, weights, w_smem_per_stage, self.w_smem_layout_staged.shape[:1]
        )
        N_padded = (self.UMMA_N + 127) // 128 * 128
        self.N_padded = N_padded
        q_sf_stage_bytes = N_padded * 4
        q_sf_stage_stride_int32 = (q_sf_stage_bytes + 127) // 128 * 128 // 4
        self.q_sf_smem_layout_staged = cute.make_layout(
            (N_padded, self.num_q_stages), stride=(1, q_sf_stage_stride_int32)
        )
        self.q_sf_tma_smem_layout_staged = cute.make_layout(
            (self.UMMA_N, self.num_q_stages),
            stride=(1, q_sf_stage_stride_int32),
        )
        q_sf_tma_smem_per_stage = cute.select(
            self.q_sf_tma_smem_layout_staged, mode=[0]
        )
        tma_atom_q_sf, tma_tensor_q_sf = cpasync.make_tiled_tma_atom(
            tma_load_op,
            q_sf,
            q_sf_tma_smem_per_stage,
            (self.UMMA_N,),
        )
        b_copy_size = cute.size_in_bytes(b_dtype, b_smem_layout)
        w_copy_size = self.UMMA_N * self.epi_bytes
        # The barrier counts the scale TMA's transferred bytes, excluding
        # unused columns in the N_padded shared-memory allocation.
        q_sf_tma_bytes = 0 if not self.is_mx_sf else self.UMMA_N * 4
        self.num_q_tma_bytes = (
            b_copy_size * atom_thr_size + q_sf_tma_bytes + w_copy_size
        )
        return (
            tma_atom_b,
            tma_tensor_b,
            tma_atom_w,
            tma_tensor_w,
            tma_atom_q_sf,
            tma_tensor_q_sf,
        )

    def _setup_mma(self, a_dtype, b_dtype, a_major, b_major):
        """Build the tiled MMA and its staged shared-memory and TMEM layouts.

        Args:
            a_dtype: Recast key element type used by MMA operand A.
            b_dtype: Recast query element type used by MMA operand B.
            a_major: Operand-major mode describing A's MMA layout.
            b_major: Operand-major mode describing B's MMA layout.
        """
        self.a_dtype = a_dtype
        self.b_dtype = b_dtype
        self.a_major_mode = a_major
        self.b_major_mode = b_major
        self.mma_tiler = (*self.mma_tiler_mn, 1)
        if not self.is_mx_sf:
            tiled_mma = sm100_utils.make_trivial_tiled_mma(
                a_dtype,
                b_dtype,
                a_major,
                b_major,
                self.acc_dtype,
                self.cta_group,
                self.mma_tiler_mn,
            )
        else:
            tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
                a_dtype,
                b_dtype,
                a_major,
                b_major,
                self.sf_dtype,
                self.sf_vec_size,
                self.cta_group,
                self.mma_tiler_mn,
            )
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        self.umma_inst_k = mma_inst_shape_k
        mma_inst_tile_k = self.head_dim // mma_inst_shape_k
        full_k = mma_inst_shape_k * mma_inst_tile_k
        self.mma_tiler = (self.mma_tiler_mn[0], self.mma_tiler_mn[1], full_k)
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)), (tiled_mma.thr_id.shape,)
        )
        self.epi_tile = self.cta_tile_shape_mnk[:2]
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler, a_dtype, self.num_kv_stages
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, b_dtype, self.num_q_stages
        )
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)
        self.copy_atom_s2t = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(self.cta_group), self.sf_dtype
        )
        if not self.is_mx_sf:
            self.num_tmem_alloc_cols = cutlass.utils.get_num_tmem_alloc_cols(
                tCtAcc_fake
            )
            self.num_tmem_alloc_cols_total = (
                self.num_tmem_alloc_cols * self.num_groups * self.num_umma_stages
            )
            self.tmem_sfa_layout = cute.make_layout(1)
            self.tmem_sfb_layout = cute.make_layout(1)
            self.sfa_chunk_smem_layout = cute.make_layout(1)
            self.sfb_chunk_smem_layout = cute.make_layout(1)
            self.num_sfa_tmem_cols = 0
            self.num_sfb_tmem_cols = 0
            return tiled_mma
        # TMEM allocates in 32-column quanta.  Narrow logical head widths
        # (H=8, block_q=2/3) produce a sub-32-column accumulator fragment, so
        # request the allocator's minimum quantum for those variants.  Wider
        # variants retain the existing unrounded allocation policy.
        self.num_tmem_alloc_cols = cutlass.utils.get_num_tmem_alloc_cols(
            tCtAcc_fake, rounding=self.UMMA_N < 32
        )
        sfa_chunk_smem_for_inference = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma, self.mma_tiler, self.sf_vec_size, 1
        )
        sfb_chunk_smem_for_inference = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma, self.mma_tiler, self.sf_vec_size, 1
        )
        _tmem_sfa_rank3 = blockscaled_utils.make_tmem_layout_sfa(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            cute.slice_(sfa_chunk_smem_for_inference, (None, None, None, 0)),
        )
        _tmem_sfb_rank3 = blockscaled_utils.make_tmem_layout_sfb(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            cute.slice_(sfb_chunk_smem_for_inference, (None, None, None, 0)),
        )
        self.tmem_sfa_layout = cute.append(
            _tmem_sfa_rank3, cute.make_layout(1, stride=0)
        )
        self.tmem_sfb_layout = cute.append(
            _tmem_sfb_rank3, cute.make_layout(1, stride=0)
        )
        _sf_kmode_size = cute.size(_tmem_sfa_rank3, mode=[2])
        _mma_inst_tile_k_for_sf = self.head_dim // self.umma_inst_k
        assert (
            _sf_kmode_size % _mma_inst_tile_k_for_sf == 0
        ), f"SF K-mode ({_sf_kmode_size}) must be divisible by UMMA K-inst count ({_mma_inst_tile_k_for_sf})"
        self.sf_k_step = _sf_kmode_size // _mma_inst_tile_k_for_sf
        # Cp4x32x128b broadcasts each 4-byte scale pack over two TMEM columns.
        # The packed scale storage is identical for MXFP4 and MXFP8; only the
        # MMA K-step differs (64 and 32 elements respectively).
        sf_atom_mn = 32
        sf_cols_per_atom = self.head_dim // self.sf_vec_size // 2
        N_padded = (self.UMMA_N + 127) // 128 * 128
        self.num_sfa_tmem_cols = (
            self.cta_tile_shape_mnk[0] // sf_atom_mn * sf_cols_per_atom
        )
        self.num_sfb_tmem_cols = N_padded // sf_atom_mn * sf_cols_per_atom
        raw_total = (
            self.num_tmem_alloc_cols * self.num_acc_groups * self.num_umma_stages
            + self.num_sfa_tmem_cols * self.num_groups
            + self.num_sfb_tmem_cols
        )
        max_tmem_cols = get_max_tmem_alloc_cols(self.arch_str)
        if raw_total <= 512:
            self.num_tmem_alloc_cols_total = max(
                1 << math.ceil(math.log2(raw_total)), 32
            )
        else:
            self.num_tmem_alloc_cols_total = (raw_total + 31) // 32 * 32
        assert (
            self.num_tmem_alloc_cols_total <= max_tmem_cols
        ), f"MX TMEM {self.num_tmem_alloc_cols_total} cols (raw={raw_total}) exceeds {self.arch_str} capacity {max_tmem_cols} for block_q={self.block_q}: acc={self.num_tmem_alloc_cols * self.num_acc_groups * self.num_umma_stages}, sfa_per_wg={self.num_sfa_tmem_cols} x{self.num_groups}, sfb={self.num_sfb_tmem_cols}."
        self.sfa_chunk_smem_layout = sfa_chunk_smem_for_inference
        self.sfb_chunk_smem_layout = sfb_chunk_smem_for_inference
        return tiled_mma


class QueryBlockBounds:
    """Inline block min/max and sampling coordinates for all scanning warps.

    Row order is unrestricted. Partial Q tiles clamp to the final live row.
    Sampling coordinates are monotone, so transforming max(ke) preserves the
    exact block upper bound. Each score still has its own per-row mask.
    """

    def __init__(
        self,
        block_q,
        block_kv,
        sample_group=1,
        sample_take=1,
        *,
        ordered=False,
    ):
        """Configure query-block bounds and optional sampled-key coordinates.

        Args:
            block_q: Number of query rows in each block.
            block_kv: Number of full or sampled key rows in each scanned tile.
            sample_group: Number of full KV rows in one sampling group.
            sample_take: Number of consecutive rows taken from the start of each group.
            ordered: Whether end bounds are nondecreasing within every query tile after
                window trimming, including on every graph replay.
        """
        self.block_q = block_q
        self.block_kv = block_kv
        self.sample_group = sample_group
        self.sample_take = sample_take
        self.ordered = ordered

    @cute.jit
    def start(self, starts, block, rows):
        """Return the minimum absolute key start of a query block.

        Args:
            starts: Nonnegative int32 ``[M]`` inclusive key starts in full-KV
                coordinates.
            block: Zero-based query-block index.
            rows: Number of live query rows used to clamp partial final blocks.
        """
        value = starts[min(block * self.block_q, rows - 1)]
        for offset in cutlass.range_constexpr(1, self.block_q):
            value = min(value, starts[min(block * self.block_q + offset, rows - 1)])
        return value

    @cute.jit
    def end(self, ends, block, rows):
        """Return the maximum absolute key end of a query block.

        Args:
            ends: Int32 ``[M]`` exclusive absolute key ends in full-KV coordinates,
                between each row's start and the full KV row count.
            block: Zero-based query-block index.
            rows: Number of live query rows used to clamp partial final blocks.
        """
        if cutlass.const_expr(self.ordered):
            # Explicit caller contract: ends are nondecreasing inside each
            # query block. Standard packed SRT extend metadata guarantees this.
            return ends[min(block * self.block_q + self.block_q - 1, rows - 1)]
        value = ends[min(block * self.block_q, rows - 1)]
        for offset in cutlass.range_constexpr(1, self.block_q):
            value = max(value, ends[min(block * self.block_q + offset, rows - 1)])
        return value

    @cute.jit
    def coordinate(self, value, extent):
        """Map a full-KV boundary to sampled coordinates when sampling is enabled.

        Args:
            value: Absolute boundary in full KV rows.
            extent: Declared sampled-row extent used for clamping; ignored for unsampled
                scans.
        """
        if cutlass.const_expr(self.sample_group != 1 or self.sample_take != 1):
            value = min(
                value // self.sample_group * self.sample_take
                + min(value % self.sample_group, self.sample_take),
                extent,
            )
        return value

    @cute.jit
    def tile_end(self, ends, block, rows, part, parts, extent):
        """Return the exclusive final KV tile for a query-block partition.

        Args:
            ends: Int32 ``[M]`` exclusive absolute key ends in full-KV coordinates,
                between each row's start and the full KV row count.
            block: Zero-based query-block index.
            rows: Number of live query rows used to clamp partial final blocks.
            part: Zero-based KV partition assigned to this CTA.
            parts: Number of KV partitions sharing the query block.
            extent: Declared sampled-row extent, or an unused value for unsampled scans.
        """
        end = self.coordinate(self.end(ends, block, rows), extent)
        tiles = (end + self.block_kv - 1) // self.block_kv
        return min(tiles, (tiles + 2 * parts - 1) // (2 * parts) * 2 * (part + 1))

    @cute.jit
    def tile_start(self, starts, ends, block, rows, part, parts, extent):
        """Return the first KV tile aligned to a tile pair for a query-block partition.

        Args:
            starts: Nonnegative int32 ``[M]`` inclusive key starts in full-KV
                coordinates.
            ends: Int32 ``[M]`` exclusive absolute key ends in full-KV coordinates,
                between each row's start and the full KV row count.
            block: Zero-based query-block index.
            rows: Number of live query rows used to clamp partial final blocks.
            part: Zero-based KV partition assigned to this CTA.
            parts: Number of KV partitions sharing the query block.
            extent: Declared sampled-row extent, or an unused value for unsampled scans.
        """
        start = self.coordinate(self.start(starts, block, rows), extent)
        end = self.coordinate(self.end(ends, block, rows), extent)
        split = 2 * self.block_kv
        hi = (end + split - 1) // split
        return max(start // split * 2, (hi + parts - 1) // parts * part * 2)
