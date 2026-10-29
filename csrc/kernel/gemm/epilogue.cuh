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

#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/numeric_conversion.h"
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
    using CopyOpT2R_Overlap = typename KernelTraits::CopyOpT2R_Overlap;
    using OverlapComputeTile = typename KernelTraits::OverlapComputeTile;
    using OverlapLoadTile = typename KernelTraits::OverlapLoadTile;
    // Bias uses plain TMA (no 2SM multicast); each CTA loads its own copy.
    using CopyOpG2S_Bias = cute::SM90_TMA_LOAD;
    using CopyOpS2G = cute::SM90_TMA_STORE;
    using CopyOpS2R =
        decltype(cutlass::epilogue::collective::detail::sm100_get_smem_load_op<
                 StrideC, TypeC, TypeAccumulate, CopyOpT2R>());
    using SmemLayoutStageC = typename KernelTraits::SmemLayoutStageC;

    using SharedStorage = typename KernelTraits::SharedStorage;

    using ClusterLayout_VMNK = decltype(cute::tiled_divide(
        cute::make_layout(ClusterShape_MNK{}),
        cute::make_tile(typename TiledMma::AtomThrID{})));

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

    static constexpr uint32_t EpiMIter =
        cute::size<0>(EpilogueTile{}) / cute::size<0>(EpiComputeTile{});
    static constexpr uint32_t EpiNIter =
        cute::size<1>(EpilogueTile{}) / cute::size<1>(EpiComputeTile{});

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
        // Tail-only stream-K: fp32 partial workspace, laid out as
        //   [slot][CtaShape_M][KBlockN]  (slot = scheduler.partial_slot())
        // nullptr when stream-K is off.
        TypeAccumulate* ptr_partial = nullptr;
    };

    struct Params {
        TMALoad_C tma_load_c;
        int m;
        int n;
        int num_experts;
        TypeD* ptr_d;
        float const* ptr_alpha = nullptr;
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

        return {tma_load_c,      m, n, num_experts, args.ptr_d, args.ptr_alpha,
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

    // No-bias path: TMEM -> registers -> cast -> GMEM.
    // KStreamK: tail-only stream-K instantiation. Compile-time so the
    // data-parallel store path keeps its register allocation (a runtime branch
    // here causes heavy spills).
    template <bool KStreamK = false, class TmemStorage>
    CUTLASS_DEVICE void epilogue(Params const& epilogue_params,
                                 AccumulatorPipeline& acc_pipeline,
                                 SharedStorage& shared_storage,
                                 TmemStorage& tmem_storage, int warp_idx,
                                 TileScheduler<KernelTraits>& scheduler) {
        int warp_group_idx = warp_idx / 4;
        int block_rank_in_cluster = cute::block_rank_in_cluster();

        cute::Layout cluster_layout_vmnk =
            cute::tiled_divide(cute::make_layout(ClusterShape_MNK{}),
                               cute::make_tile(typename TiledMma::AtomThrID{}));
        auto cta_coord_vmnk =
            cluster_layout_vmnk.get_flat_coord(block_rank_in_cluster);
        int cta_v = cute::get<0>(cta_coord_vmnk);

        cute::Tensor tmem_accumulators = tmem_storage.tmem_accumulators;
        cute::Tensor acc = tmem_accumulators(cute::make_coord(cute::_, cute::_),
                                             cute::_0{}, cute::_0{}, cute::_);

        // TMEM is read in 16-column sub-tiles; KOverlapSubCount of them at the
        // stage boundary form the overlap region.
        static constexpr int SubTilesPerWG = KernelTraits::KSubTilesPerWG;
        static constexpr int OverlapSubCount = KernelTraits::KOverlapSubCount;

        cute::Tensor acc_sub_tile =
            cute::flat_divide(acc, OverlapComputeTile{});

        cute::TiledCopy tiled_t2r_sub = cute::make_tmem_copy(
            CopyOpT2R_Sub{},
            acc_sub_tile(cute::_, cute::_, cute::_0{}, cute::_0{}, cute::_0{}));
        auto tx = threadIdx.x % cute::size(tiled_t2r_sub);
        cute::ThrCopy thread_t2r_sub = tiled_t2r_sub.get_slice(tx);

        cute::Tensor thr_tacc_src_sub =
            thread_t2r_sub.partition_S(acc_sub_tile);
        cute::Tensor thr_demo_sub = thread_t2r_sub.partition_D(
            acc_sub_tile(cute::_, cute::_, cute::_0{}, cute::_0{}, cute::_0{}));
        cute::Tensor thr_racc_dst_sub =
            cute::make_tensor<TypeAccumulate>(cute::shape(thr_demo_sub));
        cute::Tensor thr_rd_sub =
            cute::make_tensor<TypeD>(cute::shape(thr_demo_sub));

        // Unswapped output dims.
        const size_t out_m = epilogue_params.m;  // activation M
        const size_t out_n = epilogue_params.n;  // weight N

        cute::Tensor mD = [&]() {
            if constexpr (SwapAB) {
                auto stride_d =
                    cute::make_stride(cute::Int<1>{}, out_n, out_m * out_n);
                auto layout_d =
                    cute::make_layout(cute::make_shape(out_n, out_m, 1),
                                      cute::append<3>(stride_d, cute::_0{}));
                return cute::make_tensor(
                    cute::make_gmem_ptr(epilogue_params.ptr_d), layout_d);
            } else {
                auto stride_d =
                    cute::make_stride(out_n, cute::_1{}, out_m * out_n);
                auto layout_d =
                    cute::make_layout(cute::make_shape(out_m, out_n, 1),
                                      cute::append<3>(stride_d, cute::_0{}));
                return cute::make_tensor(
                    cute::make_gmem_ptr(epilogue_params.ptr_d), layout_d);
            }
        }();

        cute::Tensor gD = cute::local_tile(
            mD, cute::take<0, 2>(CtaShape_MNK{}),
            cute::make_coord(cute::_, cute::_, cute::Int<0>{}));

        cute::Tensor gD_sub_tile = cute::flat_divide(gD, OverlapComputeTile{});
        auto tiled_r2g_sub = [&]() {
            if constexpr (SwapAB) {
                return cute::make_tiled_copy_D(
                    cute::Copy_Atom<cute::UniversalCopy<TypeD>, TypeD>{},
                    tiled_t2r_sub);
            } else {
                return cute::make_tiled_copy_D(
                    cute::Copy_Atom<cute::SM100_STORE_256bit_CACHE_NOALLOCATION,
                                    TypeD>{},
                    tiled_t2r_sub);
            }
        }();
        auto thr_r2g_sub = tiled_r2g_sub.get_slice(tx);
        cute::Tensor src_sub = thr_r2g_sub.retile_S(thr_rd_sub);
        cute::Tensor dst_sub = thr_r2g_sub.partition_D(gD_sub_tile);

        // Coordinates partitioned like dst: the stores below are not clipped,
        // so the M bound is checked per element.
        auto dst_coords_sub = [&]() {
            auto mD_coords = cute::make_identity_tensor(
                SwapAB ? cute::make_shape(out_n, out_m, 1)
                       : cute::make_shape(out_m, out_n, 1));
            auto gD_coords = cute::local_tile(
                mD_coords, cute::take<0, 2>(CtaShape_MNK{}),
                cute::make_coord(cute::_, cute::_, cute::Int<0>{}));
            auto gD_sub_tile_coords =
                cute::flat_divide(gD_coords, OverlapComputeTile{});
            return thr_r2g_sub.partition_D(gD_sub_tile_coords);
        }();

        // Tail-only stream-K: partials address a compact per-CTA
        // [CtaShape_M][KBlockN] fp32 slot instead of the global D grid.
        static constexpr int KPartialM =
            cute::size<0>(cute::take<0, 2>(CtaShape_MNK{}));
        static constexpr int KPartialN =
            cute::size<1>(cute::take<0, 2>(CtaShape_MNK{}));

        // Loop-invariant partial-store partition, built once against the
        // workspace base rather than in r2g_sub (inlined per sub-tile; doing it
        // there causes spills). Per work item only a pointer offset changes.
        // Same thread layout as the D store -> vectorized fp32 stores. Empty
        // tuple when !KStreamK so data-parallel codegen is untouched.
        auto partial_store = [&]() {
            if constexpr (KStreamK) {
                auto tiled_p = cute::make_tiled_copy_D(
                    cute::Copy_Atom<
                        cute::AutoVectorizingCopyWithAssumedAlignment<128>,
                        TypeAccumulate>{},
                    tiled_t2r_sub);
                auto thr_p = tiled_p.get_slice(tx);
                return cute::make_tuple(
                    tiled_p, thr_p.retile_S(thr_racc_dst_sub),
                    thr_p.partition_D(cute::flat_divide(
                        cute::make_tensor(
                            cute::make_gmem_ptr(epilogue_params.ptr_partial),
                            cute::make_layout(
                                cute::make_shape(cute::Int<KPartialM>{},
                                                 cute::Int<KPartialN>{}),
                                cute::make_stride(cute::Int<KPartialN>{},
                                                  cute::_1{}))),
                        OverlapComputeTile{})));
            } else {
                return cute::make_tuple();
            }
        }();

        AccumulatorPipelineState acc_pipe_consumer_state;

        uint32_t epilogue_stage_idx = 0;

        while (scheduler.is_valid()) {
            uint32_t m_idx = scheduler.m_idx();
            uint32_t n_idx = scheduler.n_idx();
            // Resolved once per work item: < 0 means it owns the whole K range
            // and stores the final bf16 result.
            int partial_slot = -1;
            int64_t partial_off = 0;
            if constexpr (KStreamK) {
                partial_slot = scheduler.partial_slot();
                partial_off =
                    (static_cast<int64_t>(partial_slot) * KNumCTA + cta_v) *
                    KPartialM * KPartialN;
            }
            float alpha_val = 1.0f;
            if constexpr (HasAlpha) {
                alpha_val = shared_storage.alpha[scheduler.expert_idx()];
            }
            acc_pipeline.consumer_wait(acc_pipe_consumer_state);
            epilogue_stage_idx = acc_pipe_consumer_state.index();

            auto t2r_sub = [&](uint32_t epi_m_idx, uint32_t sub_n_idx) {
                cute::copy(
                    tiled_t2r_sub,
                    thr_tacc_src_sub(cute::_, cute::_, cute::_, epi_m_idx,
                                     sub_n_idx, epilogue_stage_idx),
                    thr_racc_dst_sub);
                if constexpr (!KernelTraits::SkipTmemFence) {
                    cutlass::arch::fence_view_async_tmem_load();
                }
            };

            auto r2g_sub = [&](uint32_t epi_m_idx, uint32_t sub_n_idx) {
                // N-overhang predicate: a KBlockN that does not divide n
                // overhangs on the last tile. D is stored register -> GMEM
                // (SM100_STORE_256bit, not TMA; CopyOpS2G is unused), so
                // nothing clips OOB writes. The host enforces n % KSubTileN ==
                // 0, so each sub-tile is entirely in or out of bounds: one
                // scalar compare, vector store intact.
                if constexpr (KernelTraits::KEpiNeedsNPredicate) {
                    const uint32_t sub_n_begin =
                        n_idx * static_cast<uint32_t>(KernelTraits::KBlockN) +
                        sub_n_idx *
                            static_cast<uint32_t>(KernelTraits::KSubTileN);
                    if (sub_n_begin >= static_cast<uint32_t>(out_n)) {
                        return;
                    }
                }
                // Stream-K tail chunk (K-slice only): store the raw fp32
                // accumulator to its workspace slot; the reduce kernel sums and
                // casts. No alpha: stream-K is gated to no-alpha BF16 dense.
                if constexpr (KStreamK) {
                    if (partial_slot >= 0) {
                        auto dst_p = cute::get<2>(partial_store);
                        dst_p.data() = dst_p.data() + partial_off;
                        copy(cute::get<0>(partial_store),
                             cute::get<1>(partial_store),
                             dst_p((cute::_, cute::_), cute::_, cute::_,
                                   cute::Int<0>{}, sub_n_idx));
                        return;
                    }
                }

                const uint32_t block_m_idx = m_idx * KNumCTA + cta_v;
                auto dst_cur =
                    dst_sub((cute::_, cute::_), cute::_, cute::_,
                            cute::Int<0>{}, sub_n_idx, block_m_idx, n_idx);
                auto coord_cur = dst_coords_sub((cute::_, cute::_), cute::_,
                                                cute::_, cute::Int<0>{},
                                                sub_n_idx, block_m_idx, n_idx);

                {
                    constexpr int kN = decltype(cute::size(thr_rd_sub))::value;
                    using Converter =
                        cutlass::NumericArrayConverter<TypeD, TypeAccumulate,
                                                       kN>;
                    auto& src_arr =
                        reinterpret_cast<cutlass::Array<TypeAccumulate, kN>&>(
                            *thr_racc_dst_sub.data());
                    auto& dst_arr =
                        reinterpret_cast<cutlass::Array<TypeD, kN>&>(
                            *thr_rd_sub.data());
                    if constexpr (HasAlpha) {
                        CUTLASS_PRAGMA_UNROLL
                        for (int i = 0; i < kN; ++i) {
                            src_arr[i] *= alpha_val;
                        }
                    }
                    dst_arr = Converter::convert(src_arr);
                }

                if constexpr (SwapAB) {
                    auto store_one = [&](uint32_t v_id) {
                        auto* addr = (TypeD*)&dst_cur(v_id);
                        auto val = src_sub[v_id];
                        asm volatile("st.global.L1::no_allocate.b16 [%0], %1;"
                                     :
                                     : "l"(addr), "h"(*(uint16_t*)&val));
                    };
                    if (out_m % KernelTraits::KBlockN == 0) {
                        // Full M tiles: no bounds check needed.
                        CUTLASS_PRAGMA_UNROLL
                        for (uint32_t v_id = 0; v_id < src_sub.size(); ++v_id) {
                            store_one(v_id);
                        }
                    } else {
                        // Partial last M tile: skip rows past M.
                        CUTLASS_PRAGMA_UNROLL
                        for (uint32_t v_id = 0; v_id < src_sub.size(); ++v_id) {
                            if ((size_t)cute::get<1>(coord_cur(v_id)) >=
                                out_m) {
                                continue;
                            }
                            store_one(v_id);
                        }
                    }
                } else {
                    if (out_m % KernelTraits::KBlockM == 0) {
                        // Full M tiles: no bounds check needed.
                        cute::copy(tiled_r2g_sub, src_sub, dst_cur);
                    } else {
                        // Predicate per store vector (one vector spans one M
                        // row), which keeps the 256-bit store.
                        auto pred_cur =
                            cute::make_tensor<bool>(cute::shape(src_sub));
                        CUTLASS_PRAGMA_UNROLL
                        for (uint32_t v_id = 0; v_id < src_sub.size(); ++v_id) {
                            pred_cur(v_id) =
                                (size_t)cute::get<0>(coord_cur(v_id)) < out_m;
                        }
                        cute::copy_if(tiled_r2g_sub, pred_cur, src_sub,
                                      dst_cur);
                    }
                }
            };

            auto process_sub = [&](uint32_t epi_m_idx, uint32_t sub_n_idx) {
                if constexpr (KernelTraits::IsSfbUnaligned) {
                    // The last N tile overhangs N; a sub-tile is all in or all
                    // out because N is a multiple of the sub-tile width.
                    if (n_idx * cute::size<1>(TileShape{}) +
                            sub_n_idx * KernelTraits::KSubTileN >=
                        out_n) {
                        return;
                    }
                }
                t2r_sub(epi_m_idx, sub_n_idx);
                r2g_sub(epi_m_idx, sub_n_idx);
            };

            if constexpr (KernelTraits::KTmemOverlapCols > 0) {
                // Overlap WG alternates: wg1 on even stages, wg0 on odd.
                uint32_t overlap_wg = 1 - (epilogue_stage_idx % 2);

                uint32_t ovlp_sub_base =
                    (epilogue_stage_idx % 2 == 0)
                        ? (2 * SubTilesPerWG - OverlapSubCount)
                        : 0;

                uint32_t wg_sub_base = warp_group_idx * SubTilesPerWG;

                auto overlap_barrier_arrive = [&]() {
                    constexpr int overlap_bar_threads = 4 * 32 + 32;
                    if constexpr (KNumCTA == 1) {
                        cutlass::arch::NamedBarrier::arrive(overlap_bar_threads,
                                                            0);
                    } else {
                        if (cta_v == 0) {
                            cutlass::arch::NamedBarrier::arrive(
                                overlap_bar_threads, 0);
                        } else {
                            constexpr int wg_threads = 4 * 32;
                            cutlass::arch::NamedBarrier wg_barrier(
                                wg_threads, 1 /*barrier id*/);
                            wg_barrier.arrive_and_wait();
                            if (threadIdx.x == warp_group_idx * 128) {
                                // The pair's MMA leader: V is the low bits of
                                // the cluster rank.
                                cutlass::arch::ClusterBarrier::arrive(
                                    &shared_storage.overlap_cluster_bar,
                                    cute::block_rank_in_cluster() - cta_v,
                                    true /*pred*/);
                            }
                        }
                    }
                };

                // Remaining (non-overlap) sub-tiles of the overlap WG.
                auto process_non_overlap_subs = [&]() {
                    CUTLASS_PRAGMA_UNROLL
                    for (uint32_t epi_m_idx = 0; epi_m_idx < EpiMIter;
                         epi_m_idx++) {
                        CUTLASS_PRAGMA_UNROLL
                        for (uint32_t s = 0; s < SubTilesPerWG; s++) {
                            uint32_t sub_idx = wg_sub_base + s;
                            if (sub_idx < ovlp_sub_base ||
                                sub_idx >= ovlp_sub_base + OverlapSubCount) {
                                process_sub(epi_m_idx, sub_idx);
                            }
                        }
                    }
                };

                if constexpr (KernelTraits::KOverlapSingleLoad) {
                    // Overlap region fits one TMEM load: T2R, arrive, then
                    // R2G, so MMA resumes without waiting for the stores.
                    cute::Tensor acc_ovlp_tile =
                        cute::flat_divide(acc, OverlapLoadTile{});
                    cute::TiledCopy tiled_t2r_ovlp = cute::make_tmem_copy(
                        CopyOpT2R_Overlap{},
                        acc_ovlp_tile(cute::_, cute::_, cute::_0{}, cute::_0{},
                                      cute::_0{}));
                    auto tx_ovlp = threadIdx.x % cute::size(tiled_t2r_ovlp);
                    cute::ThrCopy thread_t2r_ovlp =
                        tiled_t2r_ovlp.get_slice(tx_ovlp);

                    cute::Tensor thr_tacc_src_ovlp =
                        thread_t2r_ovlp.partition_S(acc_ovlp_tile);
                    cute::Tensor thr_demo_ovlp = thread_t2r_ovlp.partition_D(
                        acc_ovlp_tile(cute::_, cute::_, cute::_0{}, cute::_0{},
                                      cute::_0{}));
                    cute::Tensor thr_racc_dst_ovlp =
                        cute::make_tensor<TypeAccumulate>(
                            cute::shape(thr_demo_ovlp));
                    cute::Tensor thr_rd_ovlp =
                        cute::make_tensor<TypeD>(cute::shape(thr_demo_ovlp));

                    cute::Tensor gD_ovlp_tile =
                        cute::flat_divide(gD, OverlapLoadTile{});
                    auto tiled_r2g_ovlp = [&]() {
                        if constexpr (SwapAB) {
                            return cute::make_tiled_copy_D(
                                cute::Copy_Atom<cute::UniversalCopy<TypeD>,
                                                TypeD>{},
                                tiled_t2r_ovlp);
                        } else {
                            return cute::make_tiled_copy_D(
                                cute::Copy_Atom<
                                    cute::SM100_STORE_256bit_CACHE_NOALLOCATION,
                                    TypeD>{},
                                tiled_t2r_ovlp);
                        }
                    }();
                    auto thr_r2g_ovlp = tiled_r2g_ovlp.get_slice(tx_ovlp);
                    cute::Tensor src_ovlp = thr_r2g_ovlp.retile_S(thr_rd_ovlp);
                    cute::Tensor dst_ovlp =
                        thr_r2g_ovlp.partition_D(gD_ovlp_tile);

                    static constexpr int WideTilesTotal =
                        KernelTraits::KWideTilesTotal;
                    uint32_t ovlp_wide_idx = (epilogue_stage_idx % 2 == 0)
                                                 ? (WideTilesTotal - 1)
                                                 : 0;

                    if (warp_group_idx == overlap_wg) {
                        CUTLASS_PRAGMA_UNROLL
                        for (uint32_t epi_m_idx = 0; epi_m_idx < EpiMIter;
                             epi_m_idx++) {
                            cute::copy(tiled_t2r_ovlp,
                                       thr_tacc_src_ovlp(
                                           cute::_, cute::_, cute::_, epi_m_idx,
                                           ovlp_wide_idx, epilogue_stage_idx),
                                       thr_racc_dst_ovlp);
                            cutlass::arch::fence_view_async_tmem_load();
                        }
                    }

                    // Overlap columns read; MMA may proceed.
                    if (warp_group_idx == overlap_wg) {
                        overlap_barrier_arrive();
                    }

                    if (warp_group_idx == overlap_wg) {
                        CUTLASS_PRAGMA_UNROLL
                        for (uint32_t epi_m_idx = 0; epi_m_idx < EpiMIter;
                             epi_m_idx++) {
                            {
                                constexpr int kN =
                                    decltype(cute::size(thr_rd_ovlp))::value;
                                using Converter =
                                    cutlass::NumericArrayConverter<
                                        TypeD, TypeAccumulate, kN>;
                                auto& src_arr = reinterpret_cast<
                                    cutlass::Array<TypeAccumulate, kN> const&>(
                                    *thr_racc_dst_ovlp.data());
                                auto& dst_arr = reinterpret_cast<
                                    cutlass::Array<TypeD, kN>&>(
                                    *thr_rd_ovlp.data());
                                if constexpr (HasAlpha) {
                                    CUTLASS_PRAGMA_UNROLL
                                    for (int i = 0; i < kN; ++i) {
                                        src_arr[i] *= alpha_val;
                                    }
                                }
                                dst_arr = Converter::convert(src_arr);
                            }

                            const uint32_t block_m_idx =
                                m_idx * KNumCTA + cta_v;
                            auto dst_cur =
                                dst_ovlp((cute::_, cute::_), cute::_, cute::_,
                                         cute::Int<0>{}, ovlp_wide_idx,
                                         block_m_idx, n_idx);

                            if constexpr (SwapAB) {
                                CUTLASS_PRAGMA_UNROLL
                                for (uint32_t v_id = 0; v_id < src_ovlp.size();
                                     ++v_id) {
                                    auto* addr = (TypeD*)&dst_cur(v_id);
                                    auto val = src_ovlp[v_id];
                                    asm volatile(
                                        "st.global.L1::no_allocate.b16 "
                                        "[%0], %1;"
                                        :
                                        : "l"(addr), "h"(*(uint16_t*)&val));
                                }
                            } else {
                                cute::copy(tiled_r2g_ovlp, src_ovlp, dst_cur);
                            }
                        }

                        process_non_overlap_subs();
                    }
                } else {
                    // No single TMEM load covers the overlap region: do
                    // T2R + R2G per sub-tile, then arrive after the last read.
                    if (warp_group_idx == overlap_wg) {
                        CUTLASS_PRAGMA_UNROLL
                        for (uint32_t epi_m_idx = 0; epi_m_idx < EpiMIter;
                             epi_m_idx++) {
                            CUTLASS_PRAGMA_UNROLL
                            for (uint32_t o = 0; o < OverlapSubCount; o++) {
                                process_sub(epi_m_idx, ovlp_sub_base + o);
                            }
                        }
                    }

                    if (warp_group_idx == overlap_wg) {
                        overlap_barrier_arrive();
                    }

                    if (warp_group_idx == overlap_wg) {
                        process_non_overlap_subs();
                    }
                }

                if (warp_group_idx != overlap_wg) {
                    CUTLASS_PRAGMA_UNROLL
                    for (uint32_t epi_m_idx = 0; epi_m_idx < EpiMIter;
                         epi_m_idx++) {
                        CUTLASS_PRAGMA_UNROLL
                        for (uint32_t s = 0; s < SubTilesPerWG; s++) {
                            process_sub(epi_m_idx, wg_sub_base + s);
                        }
                    }
                }
            } else {
                CUTLASS_PRAGMA_UNROLL
                for (uint32_t epi_m_idx = 0; epi_m_idx < EpiMIter;
                     epi_m_idx++) {
                    uint32_t wg_sub_base = warp_group_idx * SubTilesPerWG;
                    CUTLASS_PRAGMA_UNROLL
                    for (uint32_t s = 0; s < SubTilesPerWG; s++) {
                        process_sub(epi_m_idx, wg_sub_base + s);
                    }
                }
            }

            if constexpr (KernelTraits::SkipTmemFence) {
                // One fence after the last TMEM load, before releasing the
                // stage (instead of one per load).
                cutlass::arch::fence_view_async_tmem_load();
            }
            acc_pipeline.consumer_release(acc_pipe_consumer_state);
            ++acc_pipe_consumer_state;

            scheduler.advance();
        }
    }

    // Bias path: as epilogue(), plus bias C from SMEM added after the cast.
    template <class TmemStorage>
    CUTLASS_DEVICE void epilogue_with_bias(
        Params const& epilogue_params, AccumulatorPipeline& acc_pipeline,
        BiasPipeline& bias_pipeline, SharedStorage& shared_storage,
        TmemStorage& tmem_storage, int warp_idx,
        TileScheduler<KernelTraits>& scheduler) {
        int warp_group_idx = warp_idx / 4;
        int block_rank_in_cluster = cute::block_rank_in_cluster();

        cute::Layout cluster_layout_vmnk =
            cute::tiled_divide(cute::make_layout(ClusterShape_MNK{}),
                               cute::make_tile(typename TiledMma::AtomThrID{}));
        auto cta_coord_vmnk =
            cluster_layout_vmnk.get_flat_coord(block_rank_in_cluster);
        int cta_v = cute::get<0>(cta_coord_vmnk);

        cute::Tensor tmem_accumulators = tmem_storage.tmem_accumulators;
        cute::Tensor acc = tmem_accumulators(cute::make_coord(cute::_, cute::_),
                                             cute::_0{}, cute::_0{}, cute::_);

        static constexpr int SubTilesPerWG = KernelTraits::KSubTilesPerWG;
        static constexpr int OverlapSubCount = KernelTraits::KOverlapSubCount;

        cute::Tensor acc_sub_tile =
            cute::flat_divide(acc, OverlapComputeTile{});

        cute::TiledCopy tiled_t2r_sub = cute::make_tmem_copy(
            CopyOpT2R_Sub{},
            acc_sub_tile(cute::_, cute::_, cute::_0{}, cute::_0{}, cute::_0{}));
        auto tx = threadIdx.x % cute::size(tiled_t2r_sub);
        cute::ThrCopy thread_t2r_sub = tiled_t2r_sub.get_slice(tx);

        cute::Tensor thr_tacc_src_sub =
            thread_t2r_sub.partition_S(acc_sub_tile);
        cute::Tensor thr_demo_sub = thread_t2r_sub.partition_D(
            acc_sub_tile(cute::_, cute::_, cute::_0{}, cute::_0{}, cute::_0{}));
        cute::Tensor thr_racc_dst_sub =
            cute::make_tensor<TypeAccumulate>(cute::shape(thr_demo_sub));
        cute::Tensor thr_rd_sub =
            cute::make_tensor<TypeD>(cute::shape(thr_demo_sub));

        cute::Tensor sC = shared_storage.tensor_sC();
        cute::Tensor sC_sub = cute::flat_divide(sC, OverlapComputeTile{});

        cute::TiledCopy tiled_s2r_sub = cute::make_tiled_copy_D(
            cute::Copy_Atom<CopyOpS2R, TypeC>{}, tiled_t2r_sub);
        cute::ThrCopy thread_s2r_sub = tiled_s2r_sub.get_slice(tx);

        cute::Tensor thr_sC_src_sub = thread_s2r_sub.partition_S(sC_sub);
        cute::Tensor thr_rC_sub =
            cute::make_tensor<TypeC>(cute::shape(thr_demo_sub));

        const size_t out_m = epilogue_params.m;
        const size_t out_n = epilogue_params.n;

        cute::Tensor mD = [&]() {
            if constexpr (SwapAB) {
                auto stride_d =
                    cute::make_stride(cute::Int<1>{}, out_n, out_m * out_n);
                auto layout_d =
                    cute::make_layout(cute::make_shape(out_n, out_m, 1),
                                      cute::append<3>(stride_d, cute::_0{}));
                return cute::make_tensor(
                    cute::make_gmem_ptr(epilogue_params.ptr_d), layout_d);
            } else {
                auto stride_d =
                    cute::make_stride(out_n, cute::_1{}, out_m * out_n);
                auto layout_d =
                    cute::make_layout(cute::make_shape(out_m, out_n, 1),
                                      cute::append<3>(stride_d, cute::_0{}));
                return cute::make_tensor(
                    cute::make_gmem_ptr(epilogue_params.ptr_d), layout_d);
            }
        }();

        cute::Tensor gD = cute::local_tile(
            mD, cute::take<0, 2>(CtaShape_MNK{}),
            cute::make_coord(cute::_, cute::_, cute::Int<0>{}));

        cute::Tensor gD_sub_tile = cute::flat_divide(gD, OverlapComputeTile{});
        auto tiled_r2g_sub = [&]() {
            if constexpr (SwapAB) {
                return cute::make_tiled_copy_D(
                    cute::Copy_Atom<cute::UniversalCopy<TypeD>, TypeD>{},
                    tiled_t2r_sub);
            } else {
                return cute::make_tiled_copy_D(
                    cute::Copy_Atom<cute::SM100_STORE_256bit_CACHE_NOALLOCATION,
                                    TypeD>{},
                    tiled_t2r_sub);
            }
        }();
        auto thr_r2g_sub = tiled_r2g_sub.get_slice(tx);
        cute::Tensor src_sub = thr_r2g_sub.retile_S(thr_rd_sub);
        cute::Tensor dst_sub = thr_r2g_sub.partition_D(gD_sub_tile);

        AccumulatorPipelineState acc_pipe_consumer_state;

        BiasPipelineState bias_pipe_consumer_state;

        while (scheduler.is_valid()) {
            uint32_t m_idx = scheduler.m_idx();
            uint32_t n_idx = scheduler.n_idx();
            float alpha_val = 1.0f;
            if constexpr (HasAlpha) {
                alpha_val = shared_storage.alpha[scheduler.expert_idx()];
            }

            acc_pipeline.consumer_wait(acc_pipe_consumer_state);
            uint32_t epilogue_stage_idx = acc_pipe_consumer_state.index();

            bias_pipeline.consumer_wait(bias_pipe_consumer_state);
            uint32_t bias_stage_idx = bias_pipe_consumer_state.index();

            auto t2r_sub = [&](uint32_t epi_m_idx, uint32_t sub_n_idx) {
                cute::copy(
                    tiled_t2r_sub,
                    thr_tacc_src_sub(cute::_, cute::_, cute::_, epi_m_idx,
                                     sub_n_idx, epilogue_stage_idx),
                    thr_racc_dst_sub);
                cutlass::arch::fence_view_async_tmem_load();
            };

            auto r2g_sub_bias = [&](uint32_t epi_m_idx, uint32_t sub_n_idx) {
                cute::copy(tiled_s2r_sub,
                           thr_sC_src_sub(cute::_, cute::_, cute::_, epi_m_idx,
                                          sub_n_idx, bias_stage_idx),
                           thr_rC_sub);

                {
                    constexpr int kN = decltype(cute::size(thr_rd_sub))::value;
                    using Converter =
                        cutlass::NumericArrayConverter<TypeD, TypeAccumulate,
                                                       kN>;
                    auto& acc_arr =
                        reinterpret_cast<cutlass::Array<TypeAccumulate, kN>&>(
                            *thr_racc_dst_sub.data());
                    auto& dst_arr =
                        reinterpret_cast<cutlass::Array<TypeD, kN>&>(
                            *thr_rd_sub.data());
                    if constexpr (HasAlpha) {
                        CUTLASS_PRAGMA_UNROLL
                        for (int i = 0; i < kN; ++i) {
                            acc_arr[i] *= alpha_val;
                        }
                    }
                    dst_arr = Converter::convert(acc_arr);

                    CUTLASS_PRAGMA_UNROLL
                    for (uint32_t v_id = 0; v_id < kN; ++v_id) {
                        dst_arr[v_id] = dst_arr[v_id] +
                                        static_cast<TypeD>(thr_rC_sub[v_id]);
                    }
                }

                const uint32_t block_m_idx = m_idx * KNumCTA + cta_v;
                cute::copy(
                    tiled_r2g_sub, src_sub,
                    dst_sub((cute::_, cute::_), cute::_, cute::_,
                            cute::Int<0>{}, sub_n_idx, block_m_idx, n_idx));
            };

            auto process_sub_bias = [&](uint32_t epi_m_idx,
                                        uint32_t sub_n_idx) {
                t2r_sub(epi_m_idx, sub_n_idx);
                r2g_sub_bias(epi_m_idx, sub_n_idx);
            };

            if constexpr (KernelTraits::KTmemOverlapCols > 0) {
                uint32_t overlap_wg = 1 - (epilogue_stage_idx % 2);

                uint32_t ovlp_sub_base =
                    (epilogue_stage_idx % 2 == 0)
                        ? (2 * SubTilesPerWG - OverlapSubCount)
                        : 0;

                uint32_t wg_sub_base = warp_group_idx * SubTilesPerWG;

                auto overlap_barrier_arrive = [&]() {
                    constexpr int overlap_bar_threads = 4 * 32 + 32;
                    if constexpr (KNumCTA == 1) {
                        cutlass::arch::NamedBarrier::arrive(overlap_bar_threads,
                                                            0);
                    } else {
                        if (cta_v == 0) {
                            cutlass::arch::NamedBarrier::arrive(
                                overlap_bar_threads, 0);
                        } else {
                            constexpr int wg_threads = 4 * 32;
                            cutlass::arch::NamedBarrier wg_barrier(
                                wg_threads, 1 /*barrier id*/);
                            wg_barrier.arrive_and_wait();
                            if (threadIdx.x == warp_group_idx * 128) {
                                // The pair's MMA leader: V is the low bits of
                                // the cluster rank.
                                cutlass::arch::ClusterBarrier::arrive(
                                    &shared_storage.overlap_cluster_bar,
                                    cute::block_rank_in_cluster() - cta_v,
                                    true /*pred*/);
                            }
                        }
                    }
                };

                // Remaining (non-overlap) sub-tiles of the overlap WG.
                auto process_non_overlap_subs = [&]() {
                    CUTLASS_PRAGMA_UNROLL
                    for (uint32_t epi_m_idx = 0; epi_m_idx < EpiMIter;
                         epi_m_idx++) {
                        CUTLASS_PRAGMA_UNROLL
                        for (uint32_t s = 0; s < SubTilesPerWG; s++) {
                            uint32_t sub_idx = wg_sub_base + s;
                            if (sub_idx < ovlp_sub_base ||
                                sub_idx >= ovlp_sub_base + OverlapSubCount) {
                                process_sub_bias(epi_m_idx, sub_idx);
                            }
                        }
                    }
                };

                if constexpr (KernelTraits::KOverlapSingleLoad) {
                    // Single-load path; see epilogue().
                    cute::Tensor acc_ovlp_tile =
                        cute::flat_divide(acc, OverlapLoadTile{});
                    cute::TiledCopy tiled_t2r_ovlp = cute::make_tmem_copy(
                        CopyOpT2R_Overlap{},
                        acc_ovlp_tile(cute::_, cute::_, cute::_0{}, cute::_0{},
                                      cute::_0{}));
                    auto tx_ovlp = threadIdx.x % cute::size(tiled_t2r_ovlp);
                    cute::ThrCopy thread_t2r_ovlp =
                        tiled_t2r_ovlp.get_slice(tx_ovlp);

                    cute::Tensor thr_tacc_src_ovlp =
                        thread_t2r_ovlp.partition_S(acc_ovlp_tile);
                    cute::Tensor thr_demo_ovlp = thread_t2r_ovlp.partition_D(
                        acc_ovlp_tile(cute::_, cute::_, cute::_0{}, cute::_0{},
                                      cute::_0{}));
                    cute::Tensor thr_racc_dst_ovlp =
                        cute::make_tensor<TypeAccumulate>(
                            cute::shape(thr_demo_ovlp));
                    cute::Tensor thr_rd_ovlp =
                        cute::make_tensor<TypeD>(cute::shape(thr_demo_ovlp));

                    cute::Tensor sC_ovlp =
                        cute::flat_divide(sC, OverlapLoadTile{});
                    cute::TiledCopy tiled_s2r_ovlp = cute::make_tiled_copy_D(
                        cute::Copy_Atom<CopyOpS2R, TypeC>{}, tiled_t2r_ovlp);
                    cute::ThrCopy thread_s2r_ovlp =
                        tiled_s2r_ovlp.get_slice(tx_ovlp);
                    cute::Tensor thr_sC_src_ovlp =
                        thread_s2r_ovlp.partition_S(sC_ovlp);
                    cute::Tensor thr_rC_ovlp =
                        cute::make_tensor<TypeC>(cute::shape(thr_demo_ovlp));

                    cute::Tensor gD_ovlp_tile =
                        cute::flat_divide(gD, OverlapLoadTile{});
                    auto tiled_r2g_ovlp = [&]() {
                        if constexpr (SwapAB) {
                            return cute::make_tiled_copy_D(
                                cute::Copy_Atom<cute::UniversalCopy<TypeD>,
                                                TypeD>{},
                                tiled_t2r_ovlp);
                        } else {
                            return cute::make_tiled_copy_D(
                                cute::Copy_Atom<
                                    cute::SM100_STORE_256bit_CACHE_NOALLOCATION,
                                    TypeD>{},
                                tiled_t2r_ovlp);
                        }
                    }();
                    auto thr_r2g_ovlp = tiled_r2g_ovlp.get_slice(tx_ovlp);
                    cute::Tensor src_ovlp = thr_r2g_ovlp.retile_S(thr_rd_ovlp);
                    cute::Tensor dst_ovlp =
                        thr_r2g_ovlp.partition_D(gD_ovlp_tile);

                    static constexpr int WideTilesTotal =
                        KernelTraits::KWideTilesTotal;
                    uint32_t ovlp_wide_idx = (epilogue_stage_idx % 2 == 0)
                                                 ? (WideTilesTotal - 1)
                                                 : 0;

                    if (warp_group_idx == overlap_wg) {
                        CUTLASS_PRAGMA_UNROLL
                        for (uint32_t epi_m_idx = 0; epi_m_idx < EpiMIter;
                             epi_m_idx++) {
                            cute::copy(tiled_t2r_ovlp,
                                       thr_tacc_src_ovlp(
                                           cute::_, cute::_, cute::_, epi_m_idx,
                                           ovlp_wide_idx, epilogue_stage_idx),
                                       thr_racc_dst_ovlp);
                            cutlass::arch::fence_view_async_tmem_load();
                        }
                    }

                    // Overlap columns read; MMA may proceed.
                    if (warp_group_idx == overlap_wg) {
                        overlap_barrier_arrive();
                    }

                    if (warp_group_idx == overlap_wg) {
                        CUTLASS_PRAGMA_UNROLL
                        for (uint32_t epi_m_idx = 0; epi_m_idx < EpiMIter;
                             epi_m_idx++) {
                            cute::copy(tiled_s2r_ovlp,
                                       thr_sC_src_ovlp(
                                           cute::_, cute::_, cute::_, epi_m_idx,
                                           ovlp_wide_idx, bias_stage_idx),
                                       thr_rC_ovlp);

                            {
                                constexpr int kN =
                                    decltype(cute::size(thr_rd_ovlp))::value;
                                using Converter =
                                    cutlass::NumericArrayConverter<
                                        TypeD, TypeAccumulate, kN>;
                                auto& acc_arr = reinterpret_cast<
                                    cutlass::Array<TypeAccumulate, kN>&>(
                                    *thr_racc_dst_ovlp.data());
                                auto& dst_arr = reinterpret_cast<
                                    cutlass::Array<TypeD, kN>&>(
                                    *thr_rd_ovlp.data());
                                if constexpr (HasAlpha) {
                                    CUTLASS_PRAGMA_UNROLL
                                    for (int i = 0; i < kN; ++i) {
                                        acc_arr[i] *= alpha_val;
                                    }
                                }
                                dst_arr = Converter::convert(acc_arr);

                                CUTLASS_PRAGMA_UNROLL
                                for (uint32_t v_id = 0; v_id < kN; ++v_id) {
                                    dst_arr[v_id] =
                                        dst_arr[v_id] +
                                        static_cast<TypeD>(thr_rC_ovlp[v_id]);
                                }
                            }

                            const uint32_t block_m_idx =
                                m_idx * KNumCTA + cta_v;
                            cute::copy(
                                tiled_r2g_ovlp, src_ovlp,
                                dst_ovlp((cute::_, cute::_), cute::_, cute::_,
                                         cute::Int<0>{}, ovlp_wide_idx,
                                         block_m_idx, n_idx));
                        }

                        process_non_overlap_subs();
                    }
                } else {
                    // Interleaved path; see epilogue().
                    if (warp_group_idx == overlap_wg) {
                        CUTLASS_PRAGMA_UNROLL
                        for (uint32_t epi_m_idx = 0; epi_m_idx < EpiMIter;
                             epi_m_idx++) {
                            CUTLASS_PRAGMA_UNROLL
                            for (uint32_t o = 0; o < OverlapSubCount; o++) {
                                process_sub_bias(epi_m_idx, ovlp_sub_base + o);
                            }
                        }
                    }

                    if (warp_group_idx == overlap_wg) {
                        overlap_barrier_arrive();
                    }

                    if (warp_group_idx == overlap_wg) {
                        process_non_overlap_subs();
                    }
                }

                if (warp_group_idx != overlap_wg) {
                    CUTLASS_PRAGMA_UNROLL
                    for (uint32_t epi_m_idx = 0; epi_m_idx < EpiMIter;
                         epi_m_idx++) {
                        CUTLASS_PRAGMA_UNROLL
                        for (uint32_t s = 0; s < SubTilesPerWG; s++) {
                            process_sub_bias(epi_m_idx, wg_sub_base + s);
                        }
                    }
                }
            } else {
                CUTLASS_PRAGMA_UNROLL
                for (uint32_t epi_m_idx = 0; epi_m_idx < EpiMIter;
                     epi_m_idx++) {
                    uint32_t wg_sub_base = warp_group_idx * SubTilesPerWG;
                    CUTLASS_PRAGMA_UNROLL
                    for (uint32_t s = 0; s < SubTilesPerWG; s++) {
                        process_sub_bias(epi_m_idx, wg_sub_base + s);
                    }
                }
            }

            acc_pipeline.consumer_release(acc_pipe_consumer_state);
            ++acc_pipe_consumer_state;

            bias_pipeline.consumer_release(bias_pipe_consumer_state);
            ++bias_pipe_consumer_state;

            scheduler.advance();
        }
    }
};

}  // namespace halops
