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

#include "cutlass/arch/arch.h"
#include "cutlass/arch/reg_reconfig.h"
#include "kernel/gemm/epilogue.cuh"
#include "kernel/gemm/gemm_utils.h"
#include "kernel/gemm/mainloop_fwd.cuh"
#include "kernel/gemm/tile_scheduler.cuh"

namespace halops {

template <typename KernelTraits, typename MainloopParams,
          typename EpilogueParams, typename SchedulerParams>
__global__ void __launch_bounds__(KernelTraits::KThreads, 1)
    gemm_kernel(CUTE_GRID_CONSTANT const MainloopParams mainloop_params,
                CUTE_GRID_CONSTANT const EpilogueParams epilogue_params,
                CUTE_GRID_CONSTANT const SchedulerParams scheduler_params) {
    using ClusterShape = typename KernelTraits::ClusterShape_MNK;
    using TiledMma = typename KernelTraits::TiledMma;
    using SharedStorage = typename KernelTraits::SharedStorage;
    using CollectiveMainloop = CollectiveMainloop<KernelTraits>;
    using CollectiveEpilogue = CollectiveEpilogue<KernelTraits>;

    using MainloopPipeline = typename CollectiveMainloop::MainloopPipeline;
    using MainloopPipelineState =
        typename CollectiveMainloop::MainloopPipelineState;
    using AccumulatorPipeline =
        typename CollectiveEpilogue::AccumulatorPipeline;
    using AccumulatorPipelineState =
        typename CollectiveEpilogue::AccumulatorPipelineState;
    using BiasPipeline = typename KernelTraits::BiasPipeline;
    using BiasPipelineState = typename KernelTraits::BiasPipelineState;
    using AtomThrShapeMNK = typename CollectiveMainloop::AtomThrShapeMNK;
    static constexpr int KNumCTA = KernelTraits::KNumCTA;
    static constexpr int KClusterM = KernelTraits::KClusterM;
    static constexpr int KClusterN = KernelTraits::KClusterN;
    static constexpr int KClusterNum = KClusterM * KClusterN;
    static constexpr int KEpilogueWarpGroupCount =
        KernelTraits::KEpilogueWarpGroupCount;
    using TmemAllocator =
        cute::conditional_t<KNumCTA == 1, cute::TMEM::Allocator1Sm,
                            cute::TMEM::Allocator2Sm>;
    using X = cute::Underscore;

    auto tx = threadIdx.x;
    auto warp_idx = tx / 32;
    uint32_t lane_predicate = cute::elect_one_sync();
    auto tx_in_warp = tx % 32;

    auto cluster_shape = cutlass::detail::select_cluster_shape(ClusterShape{});
    dim3 block_id_in_cluster = cute::block_id_in_cluster();
    uint32_t cta_rank_in_cluster = cute::block_rank_in_cluster();
    int cta_coord_v =
        cta_rank_in_cluster % cute::size<0>(typename TiledMma::AtomThrID{});
    bool is_mma_leader_cta = cta_coord_v == 0;

    using Scheduler = TileScheduler<KernelTraits>;

    TiledMma tiled_mma{};
    CollectiveMainloop main_loop;
    CollectiveEpilogue epilogue_loop;
    extern __shared__ char shared_memory[];
    auto& shared_storage = *reinterpret_cast<SharedStorage*>(shared_memory);

    // Every warp role walks the same tile sequence with its own copy.
    Scheduler scheduler(scheduler_params);

    bool is_load_ab_warp = (warp_idx == KernelTraits::KLoadABWarpId);

    if (is_load_ab_warp && lane_predicate) {
        cute::prefetch_tma_descriptor(
            mainloop_params.tma_load_A.get_tma_descriptor());
        cute::prefetch_tma_descriptor(
            mainloop_params.tma_load_B.get_tma_descriptor());
        cute::prefetch_tma_descriptor(
            mainloop_params.tma_load_SFA.get_tma_descriptor());
        if constexpr (KernelTraits::IsSfbUnaligned) {
            cute::prefetch_tma_descriptor(&mainloop_params.sfb_run_desc);
        } else {
            cute::prefetch_tma_descriptor(
                mainloop_params.tma_load_SFB.get_tma_descriptor());
        }
    }
    if constexpr (KernelTraits::AddBias) {
        if (warp_idx == KernelTraits::KLoadCWarpId && lane_predicate) {
            cute::prefetch_tma_descriptor(
                epilogue_params.tma_load_c.get_tma_descriptor());
        }
    }

    // Pair-UMMA overlap barrier on the leader CTA; peer CTAs arrive remotely.
    if constexpr (KNumCTA > 1 && KernelTraits::KTmemOverlapCols > 0) {
        if (warp_idx == KernelTraits::KMmaWarpId && is_mma_leader_cta &&
            lane_predicate) {
            cutlass::arch::ClusterBarrier::init(
                &shared_storage.overlap_cluster_bar, KNumCTA - 1);
        }
        cutlass::arch::fence_barrier_init();
    }

    // Warp roles: epilogue WGs first, then TMA load (A/B/SF), MMA, and load_c
    // (bias only).
    typename MainloopPipeline::Params mainloop_pipeline_params;
    if (is_load_ab_warp) {
        mainloop_pipeline_params.role =
            MainloopPipeline::ThreadCategory::Producer;
    }
    if (warp_idx == KernelTraits::KMmaWarpId) {
        mainloop_pipeline_params.role =
            MainloopPipeline::ThreadCategory::Consumer;
    }
    // Only the leader CTA's load thread issues arrive_and_expect_tx.
    mainloop_pipeline_params.is_leader =
        tx_in_warp == 0 && is_mma_leader_cta && is_load_ab_warp;
    mainloop_pipeline_params.transaction_bytes =
        CollectiveMainloop::TmaTransactionBytesABFor(mainloop_params.m);

    mainloop_pipeline_params.initializing_warp = KernelTraits::KLoadABWarpId;

    if constexpr (KernelTraits::IsSfbRowShift) {
        if (warp_idx == KernelTraits::KSfbTransposeWarpId && lane_predicate) {
            CUTLASS_PRAGMA_UNROLL
            for (int s = 0; s < KernelTraits::KPrologueStages; ++s) {
                cutlass::arch::ClusterTransactionBarrier::init(
                    &shared_storage.sf_full[s], 1);
            }
        }
    }
    MainloopPipeline mainloop_pipeline(
        shared_storage.mainloop, mainloop_pipeline_params, cluster_shape,
        cute::true_type{},    // Perform barrier init
        cute::false_type{});  // Delay mask calculation
    mainloop_pipeline.init_masks(cluster_shape, block_id_in_cluster);

    typename AccumulatorPipeline::Params accumulator_pipeline_params;
    if (warp_idx == KernelTraits::KMmaWarpId) {
        accumulator_pipeline_params.role =
            AccumulatorPipeline::ThreadCategory::Producer;
    }
    if (warp_idx <= KernelTraits::KEpilogueWarpEndId) {
        accumulator_pipeline_params.role =
            AccumulatorPipeline::ThreadCategory::Consumer;
    }
    static constexpr int NumEpilogueWarps = KernelTraits::KEpilogueWarpEndId -
                                            KernelTraits::KEpilogueWarpStartId +
                                            1;
    accumulator_pipeline_params.producer_arv_count = 1;
    accumulator_pipeline_params.consumer_arv_count =
        cute::size(AtomThrShapeMNK{}) * NumEpilogueWarps * 32;
    accumulator_pipeline_params.initializing_warp = KernelTraits::KMmaWarpId;
    AccumulatorPipeline accumulator_pipeline(
        shared_storage.accumulator, accumulator_pipeline_params, cluster_shape,
        cute::true_type{},    // Perform barrier init
        cute::false_type{});  // Delay mask calculation
    accumulator_pipeline.init_masks(cluster_shape, block_id_in_cluster);

    // Bias pipeline: one load_c thread produces, all epilogue threads consume.
    typename BiasPipeline::Params bias_pipeline_params;
    if constexpr (KernelTraits::AddBias) {
        if (warp_idx == KernelTraits::KLoadCWarpId) {
            bias_pipeline_params.role = BiasPipeline::ThreadCategory::Producer;
        }
        if (warp_idx <= KernelTraits::KEpilogueWarpEndId) {
            bias_pipeline_params.role = BiasPipeline::ThreadCategory::Consumer;
        }
        bias_pipeline_params.producer_arv_count = 1;
        bias_pipeline_params.consumer_arv_count = NumEpilogueWarps * 32;
        bias_pipeline_params.transaction_bytes =
            CollectiveEpilogue::TmaTransactionBytesC;
        bias_pipeline_params.initializing_warp = KernelTraits::KLoadCWarpId;
    }
    BiasPipeline bias_pipeline(shared_storage.bias_pipeline,
                               bias_pipeline_params);

    TmemAllocator tmem_allocator{};
    if (warp_idx == KernelTraits::KMmaWarpId) {
        tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns,
                                &shared_storage.tmem_base_ptr);
    }

    // Per-expert alpha to SMEM; visible to all warps after the sync below.
    if constexpr (KernelTraits::HasAlpha) {
        int ne = epilogue_params.num_experts;
        if (ne > KernelTraits::KMaxAlphaExperts)
            ne = KernelTraits::KMaxAlphaExperts;
        for (int i = threadIdx.x; i < ne; i += blockDim.x) {
            shared_storage.alpha[i] = epilogue_params.ptr_alpha[i];
        }
    }

    if constexpr (KClusterNum > 1) {
        cute::cluster_sync();
    } else {
        __syncthreads();
    }
    TmemStorage tmem_storage = main_loop.init_tmem_tensors();

    main_loop.set_tmem_offsets(tmem_storage, shared_storage.tmem_base_ptr);

    if (is_load_ab_warp) {
        // Non-epilogue warps hand registers to the epilogue warps.
        if constexpr (KEpilogueWarpGroupCount != 1) {
            cutlass::arch::warpgroup_reg_dealloc<KernelTraits::KNumRegsOther>();
        }
        main_loop.tma_loadAB(mainloop_params, mainloop_pipeline, shared_storage,
                             tiled_mma, scheduler);
    } else if (warp_idx == KernelTraits::KMmaWarpId && is_mma_leader_cta) {
        if constexpr (KEpilogueWarpGroupCount != 1) {
            cutlass::arch::warpgroup_reg_dealloc<KernelTraits::KNumRegsOther>();
        }
        main_loop.mma(mainloop_params, mainloop_pipeline, accumulator_pipeline,
                      shared_storage, tmem_storage, tiled_mma, scheduler);
        // 224-column SFB rewriter, in every CTA.
    } else if (KernelTraits::IsSfbRowShift &&
               warp_idx >= KernelTraits::KSfbTransposeWarpId) {
        if constexpr (KEpilogueWarpGroupCount != 1) {
            cutlass::arch::warpgroup_reg_dealloc<KernelTraits::KNumRegsOther>();
        }
        if constexpr (KernelTraits::IsSfbRowShift) {
            main_loop.sfb_shift_rows(
                mainloop_params, shared_storage, scheduler,
                warp_idx - KernelTraits::KSfbTransposeWarpId);
        }
    } else if (warp_idx <= KernelTraits::KEpilogueWarpEndId) {
        if constexpr (KEpilogueWarpGroupCount != 1) {
            cutlass::arch::warpgroup_reg_alloc<
                KernelTraits::KNumRegsEpilogue>();
        }
        if constexpr (KernelTraits::AddBias) {
            epilogue_loop.epilogue_with_bias(
                epilogue_params, accumulator_pipeline, bias_pipeline,
                shared_storage, tmem_storage, warp_idx, scheduler);
        } else {
            epilogue_loop.epilogue(epilogue_params, accumulator_pipeline,
                                   shared_storage, tmem_storage, warp_idx,
                                   scheduler);
        }
    } else if (KernelTraits::AddBias &&
               warp_idx == KernelTraits::KLoadCWarpId) {
        if constexpr (KEpilogueWarpGroupCount != 1) {
            cutlass::arch::warpgroup_reg_dealloc<KernelTraits::KNumRegsOther>();
        }
        epilogue_loop.load_c(epilogue_params, bias_pipeline, shared_storage,
                             scheduler);
    } else {
        if constexpr (KEpilogueWarpGroupCount != 1) {
            cutlass::arch::warpgroup_reg_dealloc<KernelTraits::KNumRegsOther>();
        }
    }

    if constexpr (KClusterNum > 1) {
        // No peer-CTA SMEM is accessed after this point.
        cute::cluster_arrive_relaxed();
        cute::cluster_wait();
    } else {
        __syncthreads();
    }

    if (warp_idx == KernelTraits::KMmaWarpId) {
        tmem_allocator.release_allocation_lock();
        tmem_allocator.free(shared_storage.tmem_base_ptr,
                            TmemAllocator::Sm100TmemCapacityColumns);
    }
}

template <typename KernelTraits>
void CheckGemmKernelLaunchParams(const GemmParams& params,
                                 int runtime_sm_count) {
    TORCH_CHECK(params.m > 0, "gemm_impl requires M > 0, got ", params.m);
    TORCH_CHECK(params.n > 0, "gemm_impl requires N > 0, got ", params.n);
    TORCH_CHECK(params.k > 0, "gemm_impl requires K > 0, got ", params.k);
    TORCH_CHECK(params.num_experts > 0,
                "gemm_impl requires num_experts > 0, got ", params.num_experts);
    TORCH_CHECK(
        runtime_sm_count > 0,
        "gemm_impl requires runtime_sm_count > 0 after cluster alignment, got ",
        runtime_sm_count);

    TORCH_CHECK(params.ptr_a != nullptr,
                "gemm_impl requires ptr_a to be non-null");
    TORCH_CHECK(params.ptr_b != nullptr,
                "gemm_impl requires ptr_b to be non-null");
    TORCH_CHECK(params.ptr_d != nullptr,
                "gemm_impl requires ptr_d to be non-null");
    TORCH_CHECK(params.ptr_sfa != nullptr,
                "gemm_impl requires ptr_sfa to be non-null");
    TORCH_CHECK(params.ptr_sfb != nullptr,
                "gemm_impl requires ptr_sfb to be non-null");
    if constexpr (KernelTraits::AddBias) {
        TORCH_CHECK(
            params.ptr_c != nullptr,
            "gemm_impl requires ptr_c to be non-null when AddBias is true");
    }
    if constexpr (KernelTraits::HasAlpha) {
        TORCH_CHECK(params.ptr_alpha != nullptr,
                    "gemm_impl requires ptr_alpha to be non-null when HasAlpha "
                    "is true");
        TORCH_CHECK(params.num_experts <= KernelTraits::KMaxAlphaExperts,
                    "gemm_impl requires num_experts <= ",
                    KernelTraits::KMaxAlphaExperts,
                    " when HasAlpha is true, got ", params.num_experts);
    }

    if constexpr (KernelTraits::KernelType == KernelMode::Dense) {
        TORCH_CHECK(params.num_experts == 1,
                    "Dense gemm_impl requires num_experts == 1, got ",
                    params.num_experts);
        TORCH_CHECK(params.max_m_per_expert == params.m,
                    "Dense gemm_impl requires max_m_per_expert == M, got "
                    "max_m_per_expert=",
                    params.max_m_per_expert, ", M=", params.m);
        TORCH_CHECK(params.masked_m == nullptr,
                    "Dense gemm_impl requires masked_m to be null");
        TORCH_CHECK(params.expert_ids == nullptr,
                    "Dense gemm_impl requires expert_ids to be null");
    } else if constexpr (KernelTraits::KernelType == KernelMode::Masked) {
        TORCH_CHECK(params.max_m_per_expert > 0,
                    "Masked gemm_impl requires max_m_per_expert > 0, got ",
                    params.max_m_per_expert);
        const int64_t expected_m =
            static_cast<int64_t>(params.num_experts) *
            static_cast<int64_t>(params.max_m_per_expert);
        TORCH_CHECK(static_cast<int64_t>(params.m) == expected_m,
                    "Masked gemm_impl requires M == num_experts * "
                    "max_m_per_expert, got M=",
                    params.m, ", num_experts=", params.num_experts,
                    ", max_m_per_expert=", params.max_m_per_expert);
        TORCH_CHECK(params.masked_m != nullptr,
                    "Masked gemm_impl requires masked_m to be non-null");
        TORCH_CHECK(params.expert_ids == nullptr,
                    "Masked gemm_impl requires expert_ids to be null");
    } else if constexpr (KernelTraits::KernelType == KernelMode::Contiguous) {
        TORCH_CHECK(params.expert_ids != nullptr,
                    "Contiguous gemm_impl requires expert_ids to be non-null");
        TORCH_CHECK(params.masked_m == nullptr,
                    "Contiguous gemm_impl requires masked_m to be null");
    } else if constexpr (KernelTraits::KernelType == KernelMode::Batch) {
        TORCH_CHECK(params.expert_ids == nullptr,
                    "Batch gemm_impl requires expert_ids to be null");
        TORCH_CHECK(params.max_m_per_expert > 0,
                    "Batch gemm_impl requires max_m_per_expert > 0");
        TORCH_CHECK(params.masked_m == nullptr,
                    "Batch gemm_impl requires masked_m to be null");
    }

    if constexpr (KernelTraits::KernelType == KernelMode::Masked) {
        if constexpr (KernelTraits::SwapAB) {
            TORCH_CHECK(params.n % KernelTraits::KBlockM == 0,
                        "Masked SwapAB gemm_impl requires N to be divisible by "
                        "KBlockM=",
                        KernelTraits::KBlockM, ", got ", params.n);
            TORCH_CHECK(
                params.max_m_per_expert % KernelTraits::KBlockN == 0,
                "Masked SwapAB gemm_impl requires max_m_per_expert to be "
                "divisible by KBlockN=",
                KernelTraits::KBlockN, ", got ", params.max_m_per_expert);
        } else {
            TORCH_CHECK(
                params.max_m_per_expert % KernelTraits::KBlockM == 0,
                "Masked gemm_impl requires max_m_per_expert to be divisible "
                "by KBlockM=",
                KernelTraits::KBlockM, ", got ", params.max_m_per_expert);
            if constexpr (KernelTraits::IsSfbUnaligned) {
                TORCH_CHECK(params.n % 128 == 0 && params.n >= 384,
                            "Masked gemm_impl requires N to be a multiple of "
                            "128 and >= 384 for KBlockN=",
                            KernelTraits::KBlockN, ", got ", params.n);
            } else {
                TORCH_CHECK(
                    params.n % KernelTraits::KBlockN == 0,
                    "Masked gemm_impl requires N to be divisible by KBlockN=",
                    KernelTraits::KBlockN, ", got ", params.n);
            }
        }
    } else {
        if constexpr (KernelTraits::SwapAB) {
            TORCH_CHECK(
                params.n % KernelTraits::KBlockM == 0,
                "SwapAB gemm_impl requires N to be divisible by KBlockM=",
                KernelTraits::KBlockM, ", got ", params.n);
            TORCH_CHECK(
                params.m % KernelTraits::KBlockN == 0 ||
                    (KernelTraits::KIsDense &&
                     params.m <= KernelTraits::KBlockN),
                "SwapAB gemm_impl requires M to be divisible by KBlockN=",
                KernelTraits::KBlockN, ", got ", params.m);
        } else {
            TORCH_CHECK(params.m % KernelTraits::KBlockM == 0 ||
                            (KernelTraits::KDenseW8A8MXFP8 &&
                             params.m <= KernelTraits::KBlockM),
                        "gemm_impl requires M to be divisible by KBlockM=",
                        KernelTraits::KBlockM, ", got ", params.m);
            if constexpr (KernelTraits::IsSfbUnaligned) {
                TORCH_CHECK(params.n % 128 == 0 && params.n >= 384,
                            "gemm_impl requires N to be a multiple of 128 and "
                            ">= 384 for KBlockN=",
                            KernelTraits::KBlockN, ", got ", params.n);
            } else {
                TORCH_CHECK(params.n % KernelTraits::KBlockN == 0,
                            "gemm_impl requires N to be divisible by KBlockN=",
                            KernelTraits::KBlockN, ", got ", params.n);
            }
        }
    }
    TORCH_CHECK(params.k % KernelTraits::KBlockK == 0,
                "gemm_impl requires K to be divisible by KBlockK=",
                KernelTraits::KBlockK, ", got ", params.k);

    if constexpr (KernelTraits::KIsFP4) {
        TORCH_CHECK(params.k % 64 == 0,
                    "FP4 gemm_impl requires K to be divisible by 64, got ",
                    params.k);
    }
}

template <typename TypeA, typename TypeB, typename TypeC, typename TypeD,
          typename TypeSF, typename TypeAccumulate, typename KernelConfig,
          bool NarrowA>
void gemm_impl_body(GemmParams& params, const int sm_count,
                    cudaStream_t stream) {
    using KernelTraits =
        GemmKernelTraits<TypeA, TypeB, TypeC, TypeD, TypeSF, TypeAccumulate,
                         KernelConfig, NarrowA>;
    using MainloopParams = typename CollectiveMainloop<KernelTraits>::Params;
    using EpilogueParams = typename CollectiveEpilogue<KernelTraits>::Params;
    using Scheduler = TileScheduler<KernelTraits>;
    using SchedulerParams = typename Scheduler::Params;

    using ClusterShape_MNK = typename KernelTraits::ClusterShape_MNK;
    static constexpr int KCTAsPerCluster = cute::size(ClusterShape_MNK{});
    const int raw_sm_count = sm_count > 0 ? sm_count : GetSmCount();
    // Only launch full clusters.
    const int runtime_sm_count =
        (raw_sm_count / KCTAsPerCluster) * KCTAsPerCluster;
    int M = params.m;
    int N = params.n;
    int K = params.k;
    int num_experts = params.num_experts;

    CheckGemmKernelLaunchParams<KernelTraits>(params, runtime_sm_count);

    static constexpr bool SwapAB = KernelTraits::SwapAB;

    // SwapAB: weight (N x K x E) is the UMMA A operand and activation (M x K)
    // the B operand; the epilogue writes D transposed back to (M x N).
    int MMA_M = SwapAB ? N : M;
    int MMA_N = SwapAB ? M : N;

    constexpr int TileSF_N =
        cute::size<1>(typename KernelTraits::TileShape_SF{});
    int N_sfb = cutlass::ceil_div(N, KernelTraits::KBlockN) * TileSF_N;

    cute::Layout layout_A =
        SwapAB ? cute::make_layout(cute::make_shape(N, K, num_experts),
                                   cute::make_stride(K, cute::Int<1>{}, N * K))
               : cute::make_layout(cute::make_shape(M, K, 1),
                                   cute::make_stride(K, cute::Int<1>{}, 0));
    cute::Layout layout_B =
        SwapAB ? cute::make_layout(cute::make_shape(M, K, 1),
                                   cute::make_stride(K, cute::Int<1>{}, 0))
               : cute::make_layout(cute::make_shape(N, K, num_experts),
                                   cute::make_stride(K, cute::Int<1>{}, N * K));
    cute::Layout layout_C = cute::make_layout(
        cute::make_shape(M, N), cute::make_stride(N, cute::Int<1>{}));

    // Scale tensors cover whole 128-row SF atoms even when M is smaller.
    const int M_sf = (M + KernelTraits::KBlockM - 1) / KernelTraits::KBlockM *
                     KernelTraits::KBlockM;
    typename KernelTraits::LayoutSFA layout_SFA =
        SwapAB ? KernelTraits::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
                     cute::make_shape(MMA_M, M_sf, K, num_experts))
               : KernelTraits::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
                     cute::make_shape(M_sf, N, K, 1));
    typename KernelTraits::LayoutSFB layout_SFB =
        SwapAB ? KernelTraits::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
                     cute::make_shape(MMA_M, M_sf, K, 1))
               : KernelTraits::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
                     cute::make_shape(M_sf, N_sfb, K, num_experts));

    MainloopParams mainloop_params =
        CollectiveMainloop<KernelTraits>::to_underlying_arguments(
            {SwapAB ? static_cast<TypeA const*>(params.ptr_b)
                    : static_cast<TypeA const*>(params.ptr_a),
             SwapAB ? static_cast<TypeB const*>(params.ptr_a)
                    : static_cast<TypeB const*>(params.ptr_b),
             SwapAB ? static_cast<TypeSF const*>(params.ptr_sfb)
                    : static_cast<TypeSF const*>(params.ptr_sfa),
             SwapAB ? static_cast<TypeSF const*>(params.ptr_sfa)
                    : static_cast<TypeSF const*>(params.ptr_sfb),
             layout_A, layout_B, layout_SFA, layout_SFB, params.k, params.m,
             num_experts});

    EpilogueParams epilogue_params =
        CollectiveEpilogue<KernelTraits>::to_underlying_arguments(
            {static_cast<TypeC const*>(params.ptr_c),
             static_cast<TypeD*>(params.ptr_d), params.m, params.n,
             params.num_experts, params.ptr_alpha});

    int scheduler_m = SwapAB ? N : M;
    int scheduler_n = SwapAB ? M : N;
    SchedulerParams scheduler_params = Scheduler::to_underlying_arguments(
        scheduler_m, scheduler_n, params.num_experts, params.max_m_per_expert,
        params.masked_m, params.expert_ids);

    auto* kernel_ptr = &gemm_kernel<KernelTraits, MainloopParams,
                                    EpilogueParams, SchedulerParams>;
    int smem_size = sizeof(typename KernelTraits::SharedStorage);

    static_assert(sizeof(typename KernelTraits::SharedStorage) <=
                      cutlass::arch::sm100_smem_capacity_bytes,
                  "SMEM usage exceeded capacity");
    CUTE_CHECK_ERROR(cudaFuncSetAttribute(
        kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    dim3 dimBlock(KernelTraits::KThreads);
    auto cluster_shape = ClusterShape_MNK{};
    dim3 dimCluster(cute::size<0>(cluster_shape), cute::size<1>(cluster_shape),
                    cute::size<2>(cluster_shape));

    int num_clusters = runtime_sm_count / KCTAsPerCluster;
    dim3 dimGrid(dimCluster.x, num_clusters * dimCluster.y);

    cutlass::ClusterLaunchParams launch_params = {dimGrid, dimBlock, dimCluster,
                                                  smem_size, stream};
    cutlass::Status status = cutlass::launch_kernel_on_cluster(
        launch_params, (void const*)kernel_ptr, mainloop_params,
        epilogue_params, scheduler_params);
    cudaError_t cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        TORCH_CHECK(false, "CUDA error: ", cudaGetErrorString(cuda_status));
    }
}

template <typename TypeA, typename TypeB, typename TypeC, typename TypeD,
          typename TypeSF, typename TypeAccumulate, typename KernelConfig>
void gemm_impl(GemmParams& params, const int sm_count = -1,
               cudaStream_t stream = 0) {
    using Traits = GemmKernelTraits<TypeA, TypeB, TypeC, TypeD, TypeSF,
                                    TypeAccumulate, KernelConfig,
                                    /*NarrowA_=*/true>;
    if constexpr (Traits::KNarrowA) {
        if (params.m < KernelConfig::KBlockM) {
            gemm_impl_body<TypeA, TypeB, TypeC, TypeD, TypeSF, TypeAccumulate,
                           KernelConfig, /*NarrowA=*/true>(params, sm_count,
                                                           stream);
            return;
        }
    }
    gemm_impl_body<TypeA, TypeB, TypeC, TypeD, TypeSF, TypeAccumulate,
                   KernelConfig, /*NarrowA=*/false>(params, sm_count, stream);
}

}  // namespace halops
