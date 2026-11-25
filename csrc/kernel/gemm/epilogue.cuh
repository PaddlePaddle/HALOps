// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "common/activation.cuh"
#include "common/mx_quant.cuh"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/numeric_conversion.h"
#include "kernel/gemm/gemm_utils.h"
#include "kernel/gemm/kernel_traits.cuh"
#include "kernel/gemm/tile_scheduler.cuh"

namespace halops {

template <typename KernelTraits>
class CollectiveEpilogue {
  public:
    using TypeAccumulate = typename KernelTraits::TypeAccumulate;
    using TypeC = typename KernelTraits::TypeC;
    using TypeD = typename KernelTraits::TypeD;
    using TiledMma = typename KernelTraits::TiledMma;
    using TileShape = typename KernelTraits::TileShape;
    using EpilogueTile = typename KernelTraits::EpilogueTile;
    using EpiComputeTile = typename KernelTraits::EpiComputeTile;
    using ClusterShape_MNK = typename KernelTraits::ClusterShape_MNK;
    using CtaShape_MNK = typename KernelTraits::CtaShape_MNK;
    static constexpr int KNumCTA = KernelTraits::KNumCTA;
    static constexpr bool AddBias = KernelTraits::AddBias;
    static constexpr bool SwapAB = KernelTraits::SwapAB;
    static constexpr bool HasAlpha = KernelTraits::HasAlpha;
    static_assert(cute::rank(EpilogueTile{}) == 2,
                  "EpilogueTile must be rank-2: [EPI_TILE_M, EPI_TILE_N]");

    // Bias C is (num_experts, n) row-major in GMEM; TMA sees it as
    // (M_virtual, N, num_experts) with M-stride 0 (row broadcast).
    using ShapeC = cute::Shape<int32_t, int32_t, int32_t>;
    using StrideC = cute::Shape<cute::_0, cute::_1, int32_t>;
    using LayoutC = cute::Layout<ShapeC, StrideC>;

    using CopyOpT2R = typename KernelTraits::CopyOpT2R;
    using CopyOpT2R_Sub = typename KernelTraits::CopyOpT2R_Sub;
    using OverlapComputeTile = typename KernelTraits::OverlapComputeTile;
    // One epilogue compute tile spans the whole per-CTA M.
    static_assert(cute::size<0>(EpilogueTile{}) ==
                  cute::size<0>(EpiComputeTile{}));
    // Bias uses plain TMA (no 2SM multicast); each CTA loads its own copy.
    using CopyOpG2S_Bias = cute::SM90_TMA_LOAD;
    using CopyOpS2R =
        decltype(cutlass::epilogue::collective::detail::sm100_get_smem_load_op<
                 StrideC, TypeC, TypeAccumulate, CopyOpT2R>());
    using SmemLayoutStageC = typename KernelTraits::SmemLayoutStageC;

    using SharedStorage = typename KernelTraits::SharedStorage;

    // make_tma_atom, not make_tma_copy: tma_partition needs a Copy_Atom.
    using TMALoad_C = decltype(cute::make_tma_atom(
        CopyOpG2S_Bias{},
        cute::make_tensor(
            cute::make_gmem_ptr(static_cast<TypeC const*>(nullptr)), ShapeC{},
            cute::append<3>(StrideC{}, cute::_0{})),
        SmemLayoutStageC{}, EpilogueTile{}, cute::_1{}));

    using AccumulatorPipeline = typename KernelTraits::AccumulatorPipeline;
    using AccumulatorPipelineState =
        typename KernelTraits::AccumulatorPipelineState;
    using BiasPipeline = typename KernelTraits::BiasPipeline;
    using BiasPipelineState = typename KernelTraits::BiasPipelineState;

    static constexpr uint32_t TmaTransactionBytesC =
        static_cast<uint32_t>(cute::cosize_v<SmemLayoutStageC> * sizeof(TypeC));

    static constexpr int KBiasStages = KernelTraits::KBiasStages;

    struct Arguments {
        TypeC const* ptr_c;
        TypeD* ptr_d;
        int m;
        int n;
        int num_experts;
        float const* ptr_alpha = nullptr;
        uint8_t* ptr_sfd = nullptr;
        TypeC const* ptr_gate = nullptr;
        int rows_per_batch = 0;
        SituCoef situ{};
        int64_t gate_ld = 0;
        // Batch D layout: false = [M, B * N] (default), true = [B, M, N].
        bool batch_out_bmn = false;
        // Stream-K fp32 partial workspace: [slot][CtaShape_M][KBlockN]; null
        // when stream-K is off.
        TypeAccumulate* ptr_partial = nullptr;
    };

    struct Params {
        TMALoad_C tma_load_c;
        int m;
        int n;
        int num_experts;
        TypeD* ptr_d;
        float const* ptr_alpha = nullptr;
        uint8_t* ptr_sfd = nullptr;
        TypeC const* ptr_gate = nullptr;
        cute::TmaDescriptor d_desc;
        SituCoef situ;
        TypeC const* ptr_c = nullptr;
        int64_t d_ld;
        int64_t gate_ld;
        bool batch_out_bmn = false;
        TypeAccumulate* ptr_partial = nullptr;
    };

    static Params to_underlying_arguments(Arguments const& args) {
        const int m = args.m;
        const int n = args.n;
        const int num_experts = args.num_experts;

        auto stride_c = cute::make_stride(cute::_0{}, cute::_1{}, n);
        cute::Layout layout_c =
            cute::make_layout(cute::make_shape(m, n, num_experts),
                              cute::append<3>(stride_c, cute::_0{}));
        cute::Tensor tensor_c =
            cute::make_tensor(cute::make_gmem_ptr(args.ptr_c), layout_c);

        TMALoad_C tma_load_c =
            cute::make_tma_atom(CopyOpG2S_Bias{}, tensor_c, SmemLayoutStageC{},
                                EpilogueTile{}, cute::_1{});

        const bool batch = KernelTraits::KernelType == KernelMode::Batch;
        const int64_t q_n = n / KAccColsPerOut;
        const int64_t rows = batch ? args.rows_per_batch : m;
        // Batch rows are B * N apart in the default layout, N when the caller
        // asked for [B, M, N].
        const int64_t d_ld =
            batch ? (args.batch_out_bmn ? q_n : num_experts * q_n) : q_n;
        cute::TmaDescriptor d_desc{};
        if constexpr (KQuantD && !SwapAB) {
            EncodeTmaDesc<3>(
                &d_desc, CU_TENSOR_MAP_DATA_TYPE_UINT8, args.ptr_d,
                {static_cast<cuuint64_t>(q_n), static_cast<cuuint64_t>(rows),
                 batch ? static_cast<cuuint64_t>(num_experts) : 1u},
                {static_cast<cuuint64_t>(d_ld),
                 static_cast<cuuint64_t>(batch ? q_n : rows * q_n)},
                {mxfp8::kGroup, static_cast<cuuint32_t>(KTileRows), 1},
                CU_TENSOR_MAP_SWIZZLE_32B, "quantized D",
                CU_TENSOR_MAP_L2_PROMOTION_NONE);
        }
        return {tma_load_c,
                m,
                n,
                num_experts,
                args.ptr_d,
                args.ptr_alpha,
                args.ptr_sfd,
                args.ptr_gate,
                d_desc,
                args.situ,
                args.ptr_c,
                d_ld,
                args.gate_ld,
                args.batch_out_bmn,
                args.ptr_partial};
    }

    // Producer warp: TMA-loads one bias C stage per scheduled tile.
    CUTLASS_DEVICE void load_c(Params const& epilogue_params,
                               BiasPipeline& bias_pipeline,
                               SharedStorage& shared_storage,
                               TileScheduler<KernelTraits>& scheduler) {
        auto tx_in_warp = threadIdx.x % 32;

        int block_rank_in_cluster = cute::block_rank_in_cluster();

        cute::Layout cluster_layout_vmnk =
            cute::tiled_divide(cute::make_layout(ClusterShape_MNK{}),
                               cute::make_tile(typename TiledMma::AtomThrID{}));
        auto cta_coord_vmnk =
            cluster_layout_vmnk.get_flat_coord(block_rank_in_cluster);
        int cta_v = cute::get<0>(cta_coord_vmnk);

        constexpr uint16_t mcast_mask_c = 0;

        // gC: (EpiM, EpiN, num_m_tiles, num_n_tiles, num_experts)
        cute::Tensor mC = epilogue_params.tma_load_c.get_tma_tensor(
            cute::make_shape(epilogue_params.m, epilogue_params.n,
                             epilogue_params.num_experts));
        cute::Tensor gC = cute::local_tile(
            mC, EpilogueTile{}, cute::make_coord(cute::_, cute::_, cute::_));

        cute::Tensor sC = shared_storage.tensor_sC();

        // Flatten (EpiM, EpiN) into one TMA mode on both sides.
        auto [tma_gC, tma_sC] = cute::tma_partition(
            epilogue_params.tma_load_c, cute::group_modes<0, 2>(sC),
            cute::group_modes<0, 2>(gC));

        BiasPipelineState bias_pipe_producer_state =
            cutlass::make_producer_start_state<BiasPipeline>();

        while (scheduler.is_valid()) {
            uint32_t m_idx = scheduler.m_idx();
            uint32_t n_idx = scheduler.n_idx();
            uint32_t expert_idx = scheduler.expert_idx();

            // Each CTA of the MMA pair owns one epilogue M tile.
            uint32_t epi_m_idx = m_idx * KNumCTA + cta_v;

            bias_pipeline.producer_acquire(bias_pipe_producer_state);

            if (tx_in_warp == 0) {
                bias_pipeline.producer_expect_transaction(
                    bias_pipe_producer_state);

                uint32_t stage_idx = bias_pipe_producer_state.index();
                cute::copy(epilogue_params.tma_load_c.with(
                               *bias_pipeline.producer_get_barrier(
                                   bias_pipe_producer_state),
                               mcast_mask_c),
                           tma_gC(cute::_, epi_m_idx, n_idx, expert_idx),
                           tma_sC(cute::_, stage_idx));

                // Must stay single-lane: producer_commit arrives
                // unconditionally and the full barrier expects one arrival.
                bias_pipeline.producer_commit(bias_pipe_producer_state);
            }

            ++bias_pipe_producer_state;
            scheduler.advance();
        }

        bias_pipeline.producer_tail(bias_pipe_producer_state);
    }

    // D rows of one CTA tile, unswapped (dim 0 normally, dim 1 for SwapAB).
    static constexpr int KTileRows = cute::size < SwapAB ? 1
                                                         : 0 > (CtaShape_MNK{});

    // Rows [lo, hi) of D the current tile may store: all of D, or for Batch
    // the rows of the tile's batch (a tile straddling two batches runs once
    // per batch).  full: the whole CTA tile lies inside them.
    struct TileRows {
        size_t lo, hi;
        size_t row0;
        bool full;
    };
    CUTLASS_DEVICE static TileRows tile_rows(
        TileScheduler<KernelTraits> const& scheduler, size_t out_m,
        uint32_t block_m_idx, uint32_t n_idx) {
        size_t lo = 0, hi = out_m;
        if constexpr (KernelTraits::KernelType == KernelMode::Batch) {
            lo = (size_t)scheduler.expert_idx() * scheduler.max_m_per_expert();
            hi = lo + scheduler.max_m_per_expert();
        }
        size_t row0 = (size_t)(SwapAB ? n_idx : block_m_idx) * KTileRows;
        if constexpr (KernelTraits::KActRowTma) row0 = scheduler.act_row0();
        return {lo, hi, row0, row0 >= lo && row0 + KTileRows <= hi};
    }
    // D, gate and D scales are row-major with d_ld columns; for Batch, D is
    // [rows_per_batch, B * N] and batch b owns columns [b * N, (b + 1) * N).
    CUTLASS_DEVICE static int64_t batch_col0(
        Params const& p, TileScheduler<KernelTraits> const& scheduler) {
        if constexpr (KernelTraits::KernelType == KernelMode::Batch) {
            return (int64_t)scheduler.expert_idx() * (p.n / KAccColsPerOut);
        } else {
            return 0;
        }
    }
    // Shift from the stacked-row D pointer to the tile's D. The default Batch
    // layout moves batch b's rows to row 0, columns b * N; [B, M, N] keeps
    // the stacked row.
    CUTLASS_DEVICE static int64_t d_shift(
        Params const& p, TileScheduler<KernelTraits> const& scheduler,
        TileRows const& rows) {
        if constexpr (KernelTraits::KernelType == KernelMode::Batch) {
            const int64_t shift =
                batch_col0(p, scheduler) - (int64_t)rows.lo * p.d_ld;
            if constexpr (KernelTraits::KIsBF16) {
                return p.batch_out_bmn ? 0 : shift;  // only BF16 takes it
            } else {
                return shift;
            }
        } else {
            return 0;
        }
    }
    CUTLASS_DEVICE static int64_t d_row(Params const& p, TileRows const& rows,
                                        int64_t row) {
        return (row - (int64_t)rows.lo) * p.d_ld;
    }
    CUTLASS_DEVICE static int64_t gate_row(Params const& p,
                                           TileRows const& rows, int64_t row) {
        return (row - (int64_t)rows.lo) * p.gate_ld;
    }
    // A loaded bf16x2 gate word; SigmoidGateMulQuant gates are logits, turned
    // into bf16 sigmoids while the tile's accumulator is still being computed.
    CUTLASS_DEVICE static void gate_act(uint32_t& g) {
        if constexpr (KernelTraits::KGateSigmoid) {
            const float2 f =
                __bfloat1622float2(reinterpret_cast<const __nv_bfloat162&>(g));
            const __nv_bfloat162 s =
                __floats2bfloat162_rn(sigmoid(f.x), sigmoid(f.y));
            g = reinterpret_cast<const uint32_t&>(s);
        }
    }
    CUTLASS_DEVICE static int64_t sf_offset(Params const& p,
                                            TileRows const& rows, int64_t row,
                                            int64_t col) {
        return mxfp8::sf_atom_offset(row - (int64_t)rows.lo,
                                     col / mxfp8::kGroup,
                                     p.d_ld / mxfp8::kGroup);
    }

    // R2G of one register tile, dropping elements outside rows.  A normal
    // store vector spans one M row, so the 256-bit store is kept.
    template <class TiledR2G, class Src, class Dst, class Coords>
    CUTLASS_DEVICE static void store_rows(TiledR2G const& tiled_r2g,
                                          Src const& src, Dst dst,
                                          Coords const& coords,
                                          TileRows const& rows) {
        if constexpr (SwapAB) {
            auto store_one = [&](uint32_t v_id) {
                auto* addr = (TypeD*)&dst(v_id);
                auto val = src[v_id];
                asm volatile("st.global.L1::no_allocate.b16 [%0], %1;"
                             :
                             : "l"(addr), "h"(*(uint16_t*)&val));
            };
            if (rows.full) {
                CUTLASS_PRAGMA_UNROLL
                for (uint32_t v_id = 0; v_id < cute::size(src); ++v_id) {
                    store_one(v_id);
                }
            } else {
                CUTLASS_PRAGMA_UNROLL
                for (uint32_t v_id = 0; v_id < cute::size(src); ++v_id) {
                    const size_t row = cute::get<1>(coords(v_id));
                    if (row >= rows.lo && row < rows.hi) {
                        store_one(v_id);
                    }
                }
            }
        } else if (rows.full) {
            cute::copy(tiled_r2g, src, dst);
        } else {
            auto pred = cute::make_tensor<bool>(cute::shape(src));
            CUTLASS_PRAGMA_UNROLL
            for (uint32_t v_id = 0; v_id < cute::size(src); ++v_id) {
                const size_t row = cute::get<0>(coords(v_id));
                pred(v_id) = row >= rows.lo && row < rows.hi;
            }
            cute::copy_if(tiled_r2g, pred, src, dst);
        }
    }

    CUTLASS_DEVICE static void overlap_arrive(SharedStorage& shared_storage,
                                              int cta_v, int warp_group_idx) {
        constexpr int overlap_bar_threads =
            cutlass::NumThreadsPerWarpGroup + cutlass::NumThreadsPerWarp;
        if (KNumCTA == 1 || cta_v == 0) {
            cutlass::arch::NamedBarrier::arrive(overlap_bar_threads, 0);
        } else {
            cutlass::arch::NamedBarrier wg_barrier(
                cutlass::NumThreadsPerWarpGroup, 1 /*barrier id*/);
            wg_barrier.arrive_and_wait();
            if (threadIdx.x ==
                warp_group_idx * cutlass::NumThreadsPerWarpGroup) {
                cutlass::arch::ClusterBarrier::arrive(
                    &shared_storage.overlap_cluster_bar,
                    cute::block_rank_in_cluster() - cta_v, true /*pred*/);
            }
        }
    }

    CUTLASS_DEVICE static float act(float x) {
        if constexpr (KernelTraits::KEpiOp == EpilogueOp::Sigmoid) {
            return sigmoid(x);
        } else {
            return x;
        }
    }

    CUTLASS_DEVICE static int pair_rank() {
        cute::Layout cluster_layout_vmnk =
            cute::tiled_divide(cute::make_layout(ClusterShape_MNK{}),
                               cute::make_tile(typename TiledMma::AtomThrID{}));
        return cute::get<0>(
            cluster_layout_vmnk.get_flat_coord(cute::block_rank_in_cluster()));
    }

    static constexpr uint32_t KEpiBarrier = 2;
    static constexpr int KStagedSmem = KernelTraits::KEpiStageOffset;
    static constexpr int KVec = sizeof(uint4);
    static constexpr bool KAct = KernelTraits::KEpiOp == EpilogueOp::Sigmoid;
    static constexpr bool KQuantD = KernelTraits::KQuantD;
    static constexpr bool KGateMul = KernelTraits::KGateMul;
    static constexpr bool KGlu = KernelTraits::KGlu;
    static constexpr int KAccColsPerOut = KernelTraits::KAccColsPerOut;
    static constexpr int KEpilogueWgs = KernelTraits::KEpilogueWarpGroupCount;

    template <class TmemStorage>
    CUTLASS_DEVICE void epilogue_swap(Params const& epilogue_params,
                                      AccumulatorPipeline& acc_pipeline,
                                      SharedStorage& shared_storage,
                                      TmemStorage& tmem_storage, int warp_idx,
                                      TileScheduler<KernelTraits>& scheduler) {
        using namespace mxfp8;
        constexpr int kChunk = KernelTraits::KEpiChunk;
        constexpr int kMyChunks = KTileRows / kChunk / KEpilogueWgs;
        constexpr int kRow = KernelTraits::KEpiRowBytes;
        constexpr int kRowVecs = kRow / KVec;
        constexpr int kWarpVecs = kRowVecs / cutlass::NumWarpsPerWarpGroup;
        constexpr int kStores =
            cute::min(kChunk * kRowVecs, int{cutlass::NumThreadsPerWarpGroup});
        constexpr int kVecs = kChunk * kRowVecs / kStores;
        constexpr int kLoads = kChunk / kMatRows;
        constexpr uint32_t kHalfLanes = cutlass::NumThreadsPerHalfWarp << 16;
        constexpr int kVals = 4 / KAccColsPerOut;
        constexpr int kBankRows = cute::max(1, kBankBytes / kRow);
        using SfRow = cute::conditional_t<KGlu, uint16_t, uint32_t>;

        const int wg = warp_idx / cutlass::NumWarpsPerWarpGroup,
                  w = warp_idx % cutlass::NumWarpsPerWarpGroup;
        const int lane = cutlass::canonical_lane_idx();
        const int tid = w * cutlass::NumThreadsPerWarp + lane;
        const int cta_v = pair_rank();
        const SituCoef situ = epilogue_params.situ;
        uint8_t* const stage = reinterpret_cast<uint8_t*>(&shared_storage) +
                               KStagedSmem +
                               wg * 2 * KernelTraits::KEpiBufBytes;
        const auto vec_off = [](int r, int v) {
            return r * kRow + (v ^ (r % kMatRows / kBankRows)) * KVec;
        };
        const int st_off = vec_off(lane % kMatRows,
                                   w * kWarpVecs + lane / kMatRows % kWarpVecs);
        const int ld_row = tid / kRowVecs, ld_vec = tid % kRowVecs;
        const bool storer = tid < kStores;
        struct Pending {
            uint4 row[kVecs];
            SfRow sf;
            uint8_t* row_dst[kVecs];
            uint8_t* sf_dst;
        } pend{};
        const auto store_pending = [&]() {
            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < kVecs; ++j) {
                if (pend.row_dst[j]) {
                    *reinterpret_cast<uint4*>(pend.row_dst[j]) = pend.row[j];
                }
            }
            if (KQuantD && pend.sf_dst) {
                *reinterpret_cast<SfRow*>(pend.sf_dst) = pend.sf;
            }
        };

        int buf_i = 0;
        AccumulatorPipelineState acc_pipe_consumer_state;
        while (scheduler.is_valid()) {
            const uint32_t block_m_idx = scheduler.m_idx() * KNumCTA + cta_v;
            const TileRows rows = tile_rows(scheduler, epilogue_params.m,
                                            block_m_idx, scheduler.n_idx());
            const int col0 =
                block_m_idx * cutlass::NumThreadsPerWarpGroup / KAccColsPerOut;
            const int64_t bcol0 = batch_col0(epilogue_params, scheduler);
            const auto row_off = [&](int64_t r) {
                return d_row(epilogue_params, rows, r) + bcol0;
            };
            const auto in_rows = [&](int64_t r) {
                return r >= (int64_t)rows.lo && r < (int64_t)rows.hi;
            };
            const auto chunk_row = [&](int i) {
                return (int64_t)rows.row0 + (wg + i * KEpilogueWgs) * kChunk;
            };
            float alpha = 1.0f;
            if constexpr (HasAlpha) {
                alpha = shared_storage.alpha[scheduler.expert_idx()];
            }
            uint32_t g[KGateMul ? kMyChunks : 1][kLoads][kQuad];
            if constexpr (KGateMul) {
                const TypeC* base = epilogue_params.ptr_gate + col0 +
                                    w * cutlass::NumThreadsPerWarp +
                                    kMatRows * (lane % kQuad);
                constexpr int kRound = 2;
                uint32_t dep = 0;
                CUTLASS_PRAGMA_UNROLL
                for (int i0 = 0; i0 < kMyChunks; i0 += kRound) {
                    uint4 x[kRound][kLoads];
                    CUTLASS_PRAGMA_UNROLL
                    for (int i = i0; i < i0 + kRound && i < kMyChunks; ++i) {
                        CUTLASS_PRAGMA_UNROLL
                        for (int l = 0; l < kLoads; ++l) {
                            const int64_t r =
                                chunk_row(i) + l * kMatRows + lane / kQuad;
                            x[i - i0][l] =
                                in_rows(r)
                                    ? __ldg(reinterpret_cast<const uint4*>(
                                          base +
                                          gate_row(epilogue_params, rows, r) +
                                          bcol0 + dep))
                                    : make_uint4(0u, 0u, 0u, 0u);
                        }
                    }
                    dep = __shfl_sync(~0u, x[0][0].x, lane) ^ x[0][0].x;
                    CUTLASS_PRAGMA_UNROLL
                    for (int i = i0; i < i0 + kRound && i < kMyChunks; ++i) {
                        CUTLASS_PRAGMA_UNROLL
                        for (int l = 0; l < kLoads; ++l) {
                            quad_transpose_4x4(x[i - i0][l], lane, g[i][l]);
                            for (uint32_t& e : g[i][l]) gate_act(e);
                        }
                    }
                }
            }
            float glu_bias[2 * kVals] = {};
            if (KGlu && epilogue_params.ptr_c) {
                CUTLASS_PRAGMA_UNROLL
                for (int j = 0; j < 2 * kVals; ++j) {
                    glu_bias[j] = static_cast<float>(
                        epilogue_params
                            .ptr_c[block_m_idx *
                                       cutlass::NumThreadsPerWarpGroup +
                                   w * cutlass::NumThreadsPerWarp +
                                   lane / kQuad + kMatRows * j]);
                }
            }
            uint8_t* const dst0 =
                reinterpret_cast<uint8_t*>(epilogue_params.ptr_d) +
                (row_off(chunk_row(0) + ld_row) + col0) * sizeof(TypeD) +
                ld_vec * KVec;
            const int64_t chunk_stride =
                KEpilogueWgs * kChunk * epilogue_params.d_ld * sizeof(TypeD);
            const int64_t vec_stride =
                kStores / kRowVecs * epilogue_params.d_ld * sizeof(TypeD);
            uint32_t row_ok = 0;
            int sf_off[kMyChunks];
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < kMyChunks; ++i) {
                CUTLASS_PRAGMA_UNROLL
                for (int j = 0; j < kVecs; ++j) {
                    const int64_t r =
                        chunk_row(i) + ld_row + j * (kStores / kRowVecs);
                    if (storer && in_rows(r)) row_ok |= 1u << (i * kVecs + j);
                }
                sf_off[i] =
                    KQuantD && tid < kChunk && in_rows(chunk_row(i) + tid)
                        ? (int)sf_offset(epilogue_params, rows,
                                         chunk_row(i) + tid, bcol0 + col0)
                        : -1;
            }
            acc_pipeline.consumer_wait(acc_pipe_consumer_state);
            const uint32_t acc =
                cute::raw_pointer_cast(
                    tmem_storage
                        .tmem_accumulators(cute::_, cute::_, cute::_,
                                           acc_pipe_consumer_state.index())
                        .data()) +
                ((uint32_t)(w * cutlass::NumThreadsPerWarp) << 16);
            using ChunkRaw = uint32_t[kLoads][2][4];
            using ChunkVals = __nv_bfloat162[kLoads][kVals];
            using ChunkMax = uint32_t[kLoads];
            const auto tmem_load_chunk = [&](int i, ChunkRaw& r) {
                CUTLASS_PRAGMA_UNROLL
                for (int l = 0; l < kLoads; ++l) {
                    const uint32_t a =
                        acc + (wg + i * KEpilogueWgs) * kChunk + l * kMatRows;
                    cute::SM100_TMEM_LOAD_16dp256b1x::copy(
                        a, r[l][0][0], r[l][0][1], r[l][0][2], r[l][0][3]);
                    cute::SM100_TMEM_LOAD_16dp256b1x::copy(
                        a + kHalfLanes, r[l][1][0], r[l][1][1], r[l][1][2],
                        r[l][1][3]);
                }
            };
            const auto to_vals = [&](int i, const ChunkRaw& r, ChunkVals& v) {
                CUTLASS_PRAGMA_UNROLL
                for (int l = 0; l < kLoads; ++l) {
                    const auto acc2 = [&](int j) {
                        const float2 x = make_float2(
                            __uint_as_float(r[l][j / 2][2 * (j % 2)]) * alpha,
                            __uint_as_float(r[l][j / 2][2 * (j % 2) + 1]) *
                                alpha);
                        if constexpr (KGlu) {
                            return __fadd2_rn(
                                x, make_float2(glu_bias[j], glu_bias[j]));
                        } else {
                            return x;
                        }
                    };
                    CUTLASS_PRAGMA_UNROLL
                    for (int j = 0; j < kVals; ++j) {
                        if constexpr (KGlu) {
                            v[l][j] = __float22bfloat162_rn(situ_glu(
                                acc2(j), acc2(j + kVals), situ.inv_beta,
                                situ.gate_up_scale, situ.inv_linear_beta));
                        } else {
                            const float2 x = acc2(j);
                            v[l][j] = __floats2bfloat162_rn(act(x.x), act(x.y));
                        }
                        if constexpr (KGateMul) {
                            uint32_t t;
                            cute::SM75_U32x1_MOVM_T::copy(g[i][l][j], t);
                            v[l][j] = __hmul2(
                                v[l][j],
                                reinterpret_cast<const __nv_bfloat162&>(t));
                        }
                    }
                }
            };
            const auto amax_buf = [&](int b) {
                return reinterpret_cast<uint32_t*>(
                    stage + b * KernelTraits::KEpiBufBytes +
                    kChunk * (kRow + KernelTraits::KSwapSfRowBytes));
            };
            const auto row_scales = [&](const ChunkVals& v, ChunkMax& m,
                                        int b) {
                CUTLASS_PRAGMA_UNROLL
                for (int l = 0; l < kLoads; ++l) {
                    m[l] = exp8_pair(abs_max<kVals>(v[l]));
                }
                CUTLASS_PRAGMA_UNROLL
                for (int x = kQuad; x < cutlass::NumThreadsPerWarp; x <<= 1) {
                    CUTLASS_PRAGMA_UNROLL
                    for (int l = 0; l < kLoads; ++l) {
                        m[l] = __vmaxu2(m[l], __shfl_xor_sync(~0u, m[l], x));
                    }
                }
                if (KGlu && lane < kQuad) {
                    CUTLASS_PRAGMA_UNROLL
                    for (int l = 0; l < kLoads; ++l) {
                        amax_buf(b)[(w * kLoads + l) * kQuad + lane] = m[l];
                    }
                }
            };
            const auto stage_chunk = [&](int i, const ChunkVals& v,
                                         ChunkMax& m) {
                uint8_t* const buf = stage + buf_i * KernelTraits::KEpiBufBytes;
                uint8_t* const sf_buf = buf + kChunk * kRow;
                CUTLASS_PRAGMA_UNROLL
                for (int l = 0; l < kLoads; ++l) {
                    if constexpr (KGlu) {
                        m[l] = __vmaxu2(
                            m[l],
                            amax_buf(buf_i)[((w ^ 1) * kLoads + l) * kQuad +
                                            lane % kQuad]);
                    }
                    auto& dst = *reinterpret_cast<cute::uint128_t*>(
                        buf + st_off + l * kMatRows * kRow);
                    if constexpr (KQuantD) {
                        const uint32_t e_lo = scale_of_exp8(m[l] & 0xffffu);
                        const uint32_t e_hi = scale_of_exp8(m[l] >> 16);
                        const uint32_t sc_bits =
                            multiplier_bits(e_lo) | multiplier_bits(e_hi) << 16;
                        const __nv_bfloat162 sc =
                            reinterpret_cast<const __nv_bfloat162&>(sc_bits);
                        if constexpr (KGlu) {
                            cute::SM100_U8x4_STSM_T::copy(
                                e4m3x4(v[l][0], v[l][1], sc), dst);
                        } else {
                            cute::SM100_U8x8_STSM_T::copy(
                                e4m3x4(v[l][0], v[l][1], sc),
                                e4m3x4(v[l][2], v[l][3], sc), dst);
                        }
                        if (lane < kMatRows && w % KAccColsPerOut == 0) {
                            sf_buf[(l * kMatRows + 2 * (lane % kQuad) +
                                    lane / kQuad) *
                                       sizeof(SfRow) +
                                   w / KAccColsPerOut] =
                                (uint8_t)(lane < kQuad ? e_lo : e_hi);
                        }
                    } else {
                        const auto& u =
                            reinterpret_cast<const uint32_t(&)[4]>(v[l]);
                        cute::SM90_U16x8_STSM_T::copy(u[0], u[1], u[2], u[3],
                                                      dst);
                    }
                }
                cutlass::arch::NamedBarrier::sync(
                    cutlass::NumThreadsPerWarpGroup, KEpiBarrier + wg);
                store_pending();
                CUTLASS_PRAGMA_UNROLL
                for (int j = 0; j < kVecs; ++j) {
                    const int r = ld_row + j * (kStores / kRowVecs);
                    pend.row[j] = *reinterpret_cast<const uint4*>(
                        buf + vec_off(r % kChunk, ld_vec));
                    pend.row_dst[j] =
                        row_ok >> (i * kVecs + j) & 1u
                            ? dst0 + i * chunk_stride + j * vec_stride
                            : nullptr;
                }
                if constexpr (KQuantD) {
                    pend.sf = *reinterpret_cast<const SfRow*>(
                        sf_buf + (tid % kChunk) * sizeof(SfRow));
                    pend.sf_dst = sf_off[i] >= 0
                                      ? epilogue_params.ptr_sfd + sf_off[i]
                                      : nullptr;
                }
                buf_i ^= 1;
            };
            ChunkRaw raw[2];
            ChunkVals cv[2];
            ChunkMax cm[2];
            tmem_load_chunk(0, raw[0]);
            if (kMyChunks > 1) tmem_load_chunk(1, raw[1]);
            cutlass::arch::fence_view_async_tmem_load();
            if (kMyChunks <= 2) {
                acc_pipeline.consumer_release(acc_pipe_consumer_state);
            }
            to_vals(0, raw[0], cv[0]);
            if constexpr (KQuantD) row_scales(cv[0], cm[0], buf_i);
            if constexpr (KGlu) {
                cutlass::arch::NamedBarrier::sync(
                    cutlass::NumThreadsPerWarpGroup, KEpiBarrier + wg);
            }
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < kMyChunks; ++i) {
                if (i + 1 < kMyChunks) {
                    to_vals(i + 1, raw[(i + 1) % 2], cv[(i + 1) % 2]);
                    if (i + 2 < kMyChunks) tmem_load_chunk(i + 2, raw[i % 2]);
                    if constexpr (KQuantD) {
                        row_scales(cv[(i + 1) % 2], cm[(i + 1) % 2], buf_i ^ 1);
                    }
                }
                stage_chunk(i, cv[i % 2], cm[i % 2]);
                if (i + 2 < kMyChunks) {
                    cutlass::arch::fence_view_async_tmem_load();
                    if (i + 3 == kMyChunks) {
                        acc_pipeline.consumer_release(acc_pipe_consumer_state);
                    }
                }
            }
            store_pending();
            pend = {};
            ++acc_pipe_consumer_state;
            scheduler.advance();
        }
    }

    // KStreamK: tail-only stream-K instantiation (BF16 only).
    template <bool KStreamK = false, class TmemStorage>
    CUTLASS_DEVICE void epilogue(Params const& epilogue_params,
                                 AccumulatorPipeline& acc_pipeline,
                                 BiasPipeline& bias_pipeline,
                                 SharedStorage& shared_storage,
                                 TmemStorage& tmem_storage, int warp_idx,
                                 TileScheduler<KernelTraits>& scheduler) {
        if constexpr (KernelTraits::KSwapStaged) {
            epilogue_swap(epilogue_params, acc_pipeline, shared_storage,
                          tmem_storage, warp_idx, scheduler);
        } else {
            using namespace mxfp8;
            constexpr int kCols = KernelTraits::KEpiComputeN;
            constexpr int kSubN = KernelTraits::KSubTileN;
            constexpr int kU = KQuantD ? kGroup : kSubN;
            constexpr int kUnits = kCols / kU;
            constexpr int kOvlpUnits =
                cute::ceil_div(KernelTraits::KTmemOverlapCols, kU);
            using UnitLoad =
                cute::conditional_t<KQuantD, cute::SM100_TMEM_LOAD_32dp32b32x,
                                    CopyOpT2R_Sub>;
            constexpr int kGroups =
                KQuantD ? kCols / kGroup / KAccColsPerOut : 1;
            using SfWord = cute::conditional_t<
                kGroups == 4, uint32_t,
                cute::conditional_t<kGroups == 2, uint16_t, uint8_t>>;

            const int wg = warp_idx / cutlass::NumWarpsPerWarpGroup;
            const int cta_v = pair_rank();
            const int tx = threadIdx.x % cutlass::NumThreadsPerWarpGroup;
            const size_t out_m = epilogue_params.m;
            const size_t out_n = epilogue_params.n;
            const int64_t d_ld = epilogue_params.d_ld;
            const SituCoef situ = epilogue_params.situ;

            cute::Tensor acc_sub_tile =
                cute::flat_divide(tmem_storage.tmem_accumulators(
                                      cute::make_coord(cute::_, cute::_),
                                      cute::_0{}, cute::_0{}, cute::_),
                                  OverlapComputeTile{});
            cute::TiledCopy tiled_t2r_sub = cute::make_tmem_copy(
                CopyOpT2R_Sub{}, acc_sub_tile(cute::_, cute::_, cute::_0{},
                                              cute::_0{}, cute::_0{}));
            cute::Tensor thr_rd_sub = cute::make_tensor<TypeD>(cute::shape(
                tiled_t2r_sub.get_slice(tx).partition_D(acc_sub_tile(
                    cute::_, cute::_, cute::_0{}, cute::_0{}, cute::_0{}))));
            constexpr int kN = decltype(cute::size(thr_rd_sub))::value;
            static_assert(decltype(cute::size(tiled_t2r_sub))::value ==
                          cutlass::NumThreadsPerWarpGroup);
            // The raw unit load assumes N in TMEM columns (32dp, 4x1); BF16
            // tiles also use the 16dp (64-row CTA) and 2SM "2x2" layouts, so
            // they load through the partitioned tiled copy instead.
            constexpr bool kPartitionedLoad = KernelTraits::KIsBF16;
            // BF16 tiles never straddle a batch (the host requires whole
            // tiles), so they clip against the stacked D; a per-batch bound
            // keeps the batch range live across the store loop and spills.
            constexpr bool kClipWholeD = KernelTraits::KIsBF16;
            cute::Tensor thr_tacc_src_sub =
                tiled_t2r_sub.get_slice(tx).partition_S(acc_sub_tile);
            cute::Tensor thr_racc_sub =
                cute::make_tensor<TypeAccumulate>(cute::shape(thr_rd_sub));
            const auto shape_d = SwapAB ? cute::make_shape(out_n, out_m, 1)
                                        : cute::make_shape(out_m, out_n, 1);
            cute::Tensor mD = cute::make_tensor(
                cute::make_gmem_ptr(epilogue_params.ptr_d),
                cute::make_layout(
                    shape_d,
                    cute::conditional_return<SwapAB>(
                        cute::make_stride(cute::Int<1>{}, d_ld, cute::_0{}),
                        cute::make_stride(d_ld, cute::Int<1>{}, cute::_0{}))));
            const auto sub_tiles = [&](auto t) {
                return cute::flat_divide(
                    cute::local_tile(
                        t, cute::take<0, 2>(CtaShape_MNK{}),
                        cute::make_coord(cute::_, cute::_, cute::Int<0>{})),
                    OverlapComputeTile{});
            };
            using StoreAtom = cute::conditional_t<
                SwapAB || KQuantD, cute::UniversalCopy<TypeD>,
                cute::SM100_STORE_256bit_CACHE_NOALLOCATION>;
            auto tiled_r2g = cute::make_tiled_copy_D(
                cute::Copy_Atom<StoreAtom, TypeD>{}, tiled_t2r_sub);
            auto thr_r2g = tiled_r2g.get_slice(tx);
            cute::Tensor src_sub = thr_r2g.retile_S(thr_rd_sub);
            cute::Tensor dst_sub = thr_r2g.partition_D(sub_tiles(mD));
            auto dst_coords_sub = thr_r2g.partition_D(
                sub_tiles(cute::make_identity_tensor(shape_d)));
            auto tiled_s2r_bias = cute::make_tiled_copy_D(
                cute::Copy_Atom<CopyOpS2R, TypeC>{}, tiled_t2r_sub);
            cute::Tensor thr_sC_sub =
                tiled_s2r_bias.get_slice(tx).partition_S(cute::flat_divide(
                    shared_storage.tensor_sC(), OverlapComputeTile{}));
            cute::Tensor thr_rC_sub =
                cute::make_tensor<TypeC>(cute::shape(thr_rd_sub));
            const TileRows whole_d{
                0, out_m, 0,
                out_m % (SwapAB ? KernelTraits::KBlockN
                                : KernelTraits::KBlockM) ==
                    0};
            const auto clip_rows = [&](const TileRows& rows) -> const TileRows& {
                if constexpr (kClipWholeD) {
                    return whole_d;
                } else {
                    return rows;
                }
            };

            // Stream-K partials go to a compact per-CTA [CtaShape_M][KBlockN]
            // slot, through the D store's thread layout. Built once: building
            // it per sub-tile spills. Empty when !KStreamK.
            constexpr int kPartialM = cute::size<0>(CtaShape_MNK{});
            constexpr int kPartialN = cute::size<1>(CtaShape_MNK{});
            auto partial_store = [&]() {
                if constexpr (KStreamK) {
                    auto tiled_p = cute::make_tiled_copy_D(
                        cute::Copy_Atom<
                            cute::AutoVectorizingCopyWithAssumedAlignment<128>,
                            TypeAccumulate>{},
                        tiled_t2r_sub);
                    auto thr_p = tiled_p.get_slice(tx);
                    return cute::make_tuple(
                        tiled_p, thr_p.retile_S(thr_racc_sub),
                        thr_p.partition_D(cute::flat_divide(
                            cute::make_tensor(
                                cute::make_gmem_ptr(epilogue_params.ptr_partial),
                                cute::make_layout(
                                    cute::make_shape(cute::Int<kPartialM>{},
                                                     cute::Int<kPartialN>{}),
                                    cute::make_stride(cute::Int<kPartialN>{},
                                                      cute::_1{}))),
                            OverlapComputeTile{})));
                } else {
                    return cute::make_tuple();
                }
            }();

            uint8_t* const stage = reinterpret_cast<uint8_t*>(&shared_storage) +
                                   KStagedSmem +
                                   wg * 2 * KernelTraits::KEpiBufBytes;
            const int st_off = tx * kGroup;
            const int st_x = (tx / (kBankBytes / kGroup)) % 2 * KVec;
            const bool issuer = KQuantD && tx == 0;
            if (issuer) {
                cute::prefetch_tma_descriptor(&epilogue_params.d_desc);
            }

            int buf_i = 0;
            // Quantized D: one UE8M0 scale per kGroup columns, e4m3 data via
            // the SMEM stage and TMA (direct stores for rows above the box),
            // then the row's scale word.
            const auto store_quant = [&](const __nv_bfloat162* vals,
                                         const TileRows& rows, int64_t row,
                                         bool row_ok, int64_t bcol0, int col0,
                                         uint32_t expert) {
                const int32_t box_row =
                    (int32_t)((int64_t)rows.row0 - (int64_t)rows.lo);
                SfWord sf_word = 0;
                CUTLASS_PRAGMA_UNROLL
                for (int g = 0; g < kGroups; ++g) {
                    const __nv_bfloat162* v = &vals[g * kGroup / 2];
                    const uint32_t m = exp8_pair(abs_max<kGroup / 2>(v));
                    const uint32_t e =
                        scale_of_exp8(::max(m & 0xffffu, m >> 16));
                    sf_word |= (SfWord)(e << (8 * g));
                    const uint32_t sc_bits = multiplier_bits(e) * 0x10001u;
                    const __nv_bfloat162 sc =
                        reinterpret_cast<const __nv_bfloat162&>(sc_bits);
                    uint32_t q[kGroup / 4];
                    CUTLASS_PRAGMA_UNROLL
                    for (int i = 0; i < kGroup / 4; ++i) {
                        q[i] = e4m3x4(v[2 * i], v[2 * i + 1], sc);
                    }
                    const uint4 lo = make_uint4(q[0], q[1], q[2], q[3]);
                    const uint4 hi = make_uint4(q[4], q[5], q[6], q[7]);
                    if (box_row < 0) {
                        if (row_ok) {
                            uint4* dst = reinterpret_cast<uint4*>(
                                reinterpret_cast<uint8_t*>(
                                    epilogue_params.ptr_d) +
                                d_row(epilogue_params, rows, row) + bcol0 +
                                col0 + g * kGroup);
                            dst[0] = lo;
                            dst[1] = hi;
                        }
                        continue;
                    }
                    uint8_t* buf =
                        stage + buf_i * KernelTraits::KEpiBufBytes;
                    *reinterpret_cast<uint4*>(buf + st_off + st_x) = lo;
                    *reinterpret_cast<uint4*>(buf + st_off +
                                              (st_x ^ KVec)) = hi;
                    cutlass::arch::fence_view_async_shared();
                    if (issuer) {
                        cute::tma_store_wait<0>();
                    }
                    cutlass::arch::NamedBarrier::sync(
                        cutlass::NumThreadsPerWarpGroup, KEpiBarrier + wg);
                    if (issuer) {
                        cute::SM90_TMA_STORE_3D::copy(
                            &epilogue_params.d_desc, buf, col0 + g * kGroup,
                            box_row,
                            KernelTraits::KernelType == KernelMode::Batch
                                ? (int32_t)expert
                                : 0);
                        cute::tma_store_arrive();
                    }
                    buf_i ^= 1;
                }
                if (row_ok) {
                    *reinterpret_cast<SfWord*>(
                        epilogue_params.ptr_sfd +
                        sf_offset(epilogue_params, rows, row,
                                  bcol0 + col0)) = sf_word;
                }
            };

            AccumulatorPipelineState acc_pipe_consumer_state;
            BiasPipelineState bias_pipe_consumer_state;
            while (scheduler.is_valid()) {
                const uint32_t n_idx = scheduler.n_idx();
                const uint32_t block_m_idx =
                    scheduler.m_idx() * KNumCTA + cta_v;
                const TileRows rows =
                    tile_rows(scheduler, out_m, block_m_idx, n_idx);
                float alpha = 1.0f;
                if constexpr (HasAlpha) {
                    alpha = shared_storage.alpha[scheduler.expert_idx()];
                }
                const int64_t bcol0 = batch_col0(epilogue_params, scheduler);
                const int64_t row_shift =
                    d_shift(epilogue_params, scheduler, rows);
                // < 0: this tile stores its final result.
                [[maybe_unused]] int partial_slot = -1;
                [[maybe_unused]] int64_t partial_off = 0;
                if constexpr (KStreamK) {
                    partial_slot = scheduler.partial_slot();
                    partial_off =
                        (static_cast<int64_t>(partial_slot) * KNumCTA + cta_v) *
                        kPartialM * kPartialN;
                }
                const int64_t row = (int64_t)rows.row0 + tx;
                const bool row_ok =
                    row >= (int64_t)rows.lo && row < (int64_t)rows.hi;
                const int col0 = (n_idx * KernelTraits::KBlockN + wg * kCols) /
                                 KAccColsPerOut;
                __nv_bfloat162 vals[KQuantD ? kCols / 2 / KAccColsPerOut : 1];
                if constexpr (KGateMul) {
                    if (row_ok) {
                        auto* g = reinterpret_cast<uint32_t*>(vals);
                        constexpr int kW =
                            sizeof(cute::uint256_t) / sizeof(uint32_t);
                        constexpr int kE =
                            sizeof(cute::uint256_t) / sizeof(TypeC);
                        CUTLASS_PRAGMA_UNROLL
                        for (int i = 0; i < kCols / kE; ++i) {
                            copy_to_regs<
                                cute::SM100_LOAD_256bit_CACHE_NOALLOCATION>(
                                *reinterpret_cast<const cute::uint256_t*>(
                                    epilogue_params.ptr_gate +
                                    gate_row(epilogue_params, rows, row) +
                                    bcol0 + col0 + i * kE),
                                reinterpret_cast<uint32_t(&)[kW]>(g[kW * i]));
                        }
                        for (__nv_bfloat162& v : vals) {
                            gate_act(reinterpret_cast<uint32_t&>(v));
                        }
                    }
                }
                [[maybe_unused]] uint32_t bias_stage = 0;
                if constexpr (AddBias) {
                    bias_pipeline.consumer_wait(bias_pipe_consumer_state);
                    bias_stage = bias_pipe_consumer_state.index();
                }
                acc_pipeline.consumer_wait(acc_pipe_consumer_state);
                const uint32_t stage_idx = acc_pipe_consumer_state.index();
                const uint32_t acc =
                    cute::raw_pointer_cast(
                        tmem_storage
                            .tmem_accumulators(cute::_, cute::_, cute::_,
                                               stage_idx)
                            .data()) +
                    ((uint32_t)((warp_idx % cutlass::NumWarpsPerWarpGroup) *
                                cutlass::NumThreadsPerWarp)
                     << 16) +
                    wg * kCols;
                const auto load_unit = [&](int u, uint32_t(&r)[kU]) {
                    if constexpr (kPartitionedLoad) {
                        cute::copy(tiled_t2r_sub,
                                   thr_tacc_src_sub(cute::_, cute::_, cute::_,
                                                    cute::_0{}, wg * kUnits + u,
                                                    stage_idx),
                                   thr_racc_sub);
                    } else {
                        copy_to_regs<UnitLoad>(acc + u * kU, r);
                    }
                };
                const auto consume = [&](int u, const uint32_t(&r)[kU]) {
                    if constexpr (KGlu) {
                        uint4 b[kU * sizeof(TypeC) / KVec] = {};
                        if (epilogue_params.ptr_c) {
                            const auto* p = reinterpret_cast<const uint4*>(
                                epilogue_params.ptr_c +
                                n_idx * KernelTraits::KBlockN + wg * kCols +
                                u * kU);
                            CUTLASS_PRAGMA_UNROLL
                            for (int i = 0; i < kU * sizeof(TypeC) / KVec;
                                 ++i) {
                                b[i] = __ldg(p + i);
                            }
                        }
                        const auto acc2 = [&](int i) {
                            return __fadd2_rn(
                                make_float2(__uint_as_float(r[i]),
                                            __uint_as_float(r[i + 1])),
                                __bfloat1622float2(
                                    reinterpret_cast<const __nv_bfloat162*>(
                                        b)[i / 2]));
                        };
                        CUTLASS_PRAGMA_UNROLL
                        for (int j = 0; j < kGluInterleave / 2; ++j) {
                            vals[u * kGluInterleave / 2 + j] =
                                __float22bfloat162_rn(situ_glu(
                                    acc2(2 * j), acc2(kGluInterleave + 2 * j),
                                    situ.inv_beta, situ.gate_up_scale,
                                    situ.inv_linear_beta));
                        }
                    } else if constexpr (KQuantD) {
                        CUTLASS_PRAGMA_UNROLL
                        for (int j = 0; j < kU / 2; ++j) {
                            const __nv_bfloat162 x = __floats2bfloat162_rn(
                                act(__uint_as_float(r[2 * j])),
                                act(__uint_as_float(r[2 * j + 1])));
                            __nv_bfloat162& v = vals[u * kU / 2 + j];
                            v = KGateMul ? __hmul2(x, v) : x;
                        }
                    } else {
                        const int sub_idx = wg * kUnits + u;
                        // The last N tile overhangs N (unaligned-SFB pair
                        // tiles, BF16 N-tails); a sub-tile is all in or all
                        // out because N is a multiple of the sub-tile width.
                        if constexpr ((KernelTraits::IsSfbUnaligned &&
                                       !SwapAB) ||
                                      KernelTraits::KEpiNeedsNPredicate) {
                            if (n_idx * cute::size<1>(TileShape{}) +
                                    sub_idx * kSubN >=
                                out_n) {
                                return;
                            }
                        }
                        // Stream-K tail chunk: store the raw fp32
                        // accumulator; the reduce kernel sums and casts.
                        if constexpr (KStreamK) {
                            if (partial_slot >= 0) {
                                auto dst_p = cute::get<2>(partial_store);
                                dst_p.data() = dst_p.data() + partial_off;
                                cute::copy(cute::get<0>(partial_store),
                                           cute::get<1>(partial_store),
                                           dst_p((cute::_, cute::_), cute::_,
                                                 cute::_, cute::Int<0>{},
                                                 sub_idx));
                                return;
                            }
                        }
                        if constexpr (AddBias) {
                            cute::copy(
                                tiled_s2r_bias,
                                thr_sC_sub(cute::_, cute::_, cute::_,
                                           cute::_0{}, sub_idx, bias_stage),
                                thr_rC_sub);
                        }
                        cutlass::Array<TypeAccumulate, kN> f;
                        CUTLASS_PRAGMA_UNROLL
                        for (int i = 0; i < kN; ++i) {
                            if constexpr (kPartitionedLoad) {
                                f[i] = thr_racc_sub(i) * alpha;
                            } else {
                                f[i] = __uint_as_float(r[i]) * alpha;
                            }
                            if constexpr (AddBias && KAct) {
                                f[i] += static_cast<float>(thr_rC_sub[i]);
                            }
                            f[i] = act(f[i]);
                        }
                        auto& d = reinterpret_cast<cutlass::Array<TypeD, kN>&>(
                            *thr_rd_sub.data());
                        d = cutlass::NumericArrayConverter<
                            TypeD, TypeAccumulate, kN>::convert(f);
                        if constexpr (AddBias && !KAct) {
                            CUTLASS_PRAGMA_UNROLL
                            for (int i = 0; i < kN; ++i) {
                                d[i] = d[i] + static_cast<TypeD>(thr_rC_sub[i]);
                            }
                        }
                        const auto dst_tile = cute::make_tensor(
                            dst_sub.data() + row_shift, dst_sub.layout());
                        store_rows(tiled_r2g, src_sub,
                                   dst_tile((cute::_, cute::_), cute::_,
                                            cute::_, cute::Int<0>{}, sub_idx,
                                            block_m_idx, n_idx),
                                   dst_coords_sub((cute::_, cute::_), cute::_,
                                                  cute::_, cute::Int<0>{},
                                                  sub_idx, block_m_idx, n_idx),
                                   clip_rows(rows));
                    }
                };
                const bool ovlp_wg =
                    kOvlpUnits > 0 && wg == 1 - (int)(stage_idx % 2);
                const auto run = [&](auto backwards) {
                    CUTLASS_PRAGMA_UNROLL
                    for (int i = 0; i < kUnits; ++i) {
                        const int u =
                            decltype(backwards)::value ? kUnits - 1 - i : i;
                        const bool last_ovlp = ovlp_wg && i == kOvlpUnits - 1;
                        uint32_t r[kU];
                        load_unit(u, r);
                        if (!KernelTraits::SkipTmemFence || last_ovlp) {
                            cutlass::arch::fence_view_async_tmem_load();
                        }
                        if (KernelTraits::KOverlapSingleLoad && last_ovlp) {
                            overlap_arrive(shared_storage, cta_v, wg);
                        }
                        consume(u, r);
                        if (!KernelTraits::KOverlapSingleLoad && last_ovlp) {
                            overlap_arrive(shared_storage, cta_v, wg);
                        }
                    }
                };
                if (ovlp_wg && stage_idx % 2 == 0) {
                    run(cute::true_type{});
                } else {
                    run(cute::false_type{});
                }
                if constexpr (KernelTraits::SkipTmemFence) {
                    cutlass::arch::fence_view_async_tmem_load();
                }
                acc_pipeline.consumer_release(acc_pipe_consumer_state);
                ++acc_pipe_consumer_state;
                if constexpr (AddBias) {
                    bias_pipeline.consumer_release(bias_pipe_consumer_state);
                    ++bias_pipe_consumer_state;
                }

                if constexpr (KQuantD) {
                    store_quant(vals, rows, row, row_ok, bcol0, col0,
                                scheduler.expert_idx());
                }
                scheduler.advance();
            }
            if (issuer) {
                bulk_store_wait_all();
            }
        }
    }
};

}  // namespace halops
