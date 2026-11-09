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

#include <cuda_pipeline.h>

#include "common/ptx.cuh"
#include "indexer_epilogue.cuh"

namespace halops::indexer {

template <class KernelTraits, class Context>
CUTE_DEVICE void complete_indexer_grid(Context const& context) {
    const int warp_idx = KernelTraits::warp_idx();
    if (warp_idx == 0) {
        int previous = 0;
        if (KernelTraits::lane_idx() == 0)
            previous = halops::gmem_atom_add1_acq_rel(
                context.params.grid_completion_counter);
        if (__shfl_sync(0xffffffffu, previous, 0) == (int)gridDim.x - 1) {
            context.barriers.reset_active_warp();
            __syncwarp();
            if (KernelTraits::lane_idx() == 0)
                halops::gmem_st_zero_relaxed(
                    context.params.grid_completion_counter);
        }
    }
}

template <class KernelTraits, bool SINGLE_TOKEN, bool LOCAL_QUERY>
struct IndexerPrepareDispatch;

template <class KernelTraits>
struct IndexerPrepareDispatch<KernelTraits, false, false> {
    using Roles = typename KernelTraits::Roles;
    using MainLoop = CollectiveIndexer<KernelTraits>;
    using Epilogue = CollectiveIndexerEpilogue<KernelTraits>;

    template <class Context>
    static CUTE_DEVICE void run(Context const& context,
                                MainLoop const& main_loop,
                                Epilogue const& epilogue) {
        const int warp_idx = KernelTraits::warp_idx();
        if (Roles::is_mma(warp_idx)) {
            cutlass::arch::warpgroup_reg_dealloc<
                KernelTraits::Registers::Producer>();
            main_loop.run_matrix_products(context);
        } else if (Roles::is_weight_loader(warp_idx)) {
            cutlass::arch::warpgroup_reg_dealloc<
                KernelTraits::Registers::Producer>();
            main_loop.load_projection_weights(context);
        } else if (Roles::is_activation_loader(warp_idx)) {
            cutlass::arch::warpgroup_reg_dealloc<
                KernelTraits::Registers::Producer>();
            main_loop.load_projection_activations(context);
            if constexpr (KernelTraits::PrefetchRope &&
                          !KernelTraits::is_mx_sf) {
                if (context.params.raw_weight_prefetch_mode ==
                    RawWeightPrefetch::WeightLoaderWarp)
                    main_loop.prefetch_raw_weights(context);
            }
        } else if (Roles::is_prefetch(warp_idx)) {
            cutlass::arch::warpgroup_reg_dealloc<
                KernelTraits::Registers::Producer>();
            main_loop.prefetch_query_inputs(context);
        } else if (Roles::is_drain(warp_idx)) {
            if constexpr (KernelTraits::TokenTile == 128) {
                if (context.params.eight_warp_query &&
                    blockIdx.x >= context.params.key_weight_blocks) {
                    cutlass::arch::warpgroup_reg_alloc<
                        KernelTraits::Registers::EightWarpConsumer>();
                    drain_q_eight<KernelTraits>(context);
                } else {
                    cutlass::arch::warpgroup_reg_alloc<
                        KernelTraits::Registers::AccumulatorConsumer>();
                    if constexpr (KernelTraits::EightWarpEpilogue)
                        epilogue.template consume_accumulators<true>(context);
                    else
                        epilogue.consume_accumulators(context);
                }
            } else {
                cutlass::arch::warpgroup_reg_alloc<
                    KernelTraits::Registers::AccumulatorConsumer>();
                epilogue.consume_accumulators(context);
            }
        } else if (Roles::is_vector(warp_idx)) {
            if constexpr (KernelTraits::TokenTile == 128) {
                if (context.params.eight_warp_query &&
                    blockIdx.x >= context.params.key_weight_blocks) {
                    cutlass::arch::warpgroup_reg_alloc<
                        KernelTraits::Registers::EightWarpConsumer>();
                    epilogue.quantize_query_lora(context);
                    drain_q_eight<KernelTraits>(context);
                } else {
                    cutlass::arch::warpgroup_reg_alloc<
                        KernelTraits::Registers::VectorConsumer>();
                    epilogue.run_vector_operations(context);
                }
            } else {
                cutlass::arch::warpgroup_reg_alloc<
                    KernelTraits::Registers::VectorConsumer>();
                epilogue.run_vector_operations(context);
            }
        }
    }

    template <class Context>
    static CUTE_DEVICE void finish(Context const& context,
                                   Epilogue const& epilogue) {
        if constexpr (KernelTraits::TokenTile >= 64) {
            if (context.params.all_warps_process_keys &&
                blockIdx.x < context.params.key_weight_blocks)
                epilogue.template k_epilogue<0, KernelTraits::NumWarps>(
                    context);
        }
        sync_named_barrier<typename KernelTraits::Barriers::EveryoneSync>();
        if (KernelTraits::warp_idx() == 0) context.plan.release_tmem();
        complete_indexer_grid<KernelTraits>(context);
    }
};

template <class KernelTraits>
struct IndexerPrepareDispatch<KernelTraits, true, false> {
    using Roles = typename KernelTraits::Roles;
    using MainLoop = CollectiveIndexer<KernelTraits>;
    using Epilogue = CollectiveIndexerEpilogue<KernelTraits>;

    template <class Context>
    static CUTE_DEVICE void run(Context const& context,
                                MainLoop const& main_loop,
                                Epilogue const& epilogue) {
        const int warp_idx = KernelTraits::warp_idx();
        if (Roles::is_mma(warp_idx)) {
            main_loop.mma_decode(context);
            epilogue.template decode_key_consumer<true>(context);
        } else if (Roles::is_weight_loader(warp_idx)) {
            main_loop.weight_decode(context);
        } else if (Roles::is_activation_loader(warp_idx) ||
                   Roles::is_prefetch(warp_idx)) {
            main_loop.template quantize_local_q<2, true>(context);
        } else if (Roles::is_drain(warp_idx)) {
            epilogue.drain_decode(context);
        }
    }

    template <class Context>
    static CUTE_DEVICE void finish(Context const& context, Epilogue const&) {
        __syncthreads();
        if (KernelTraits::warp_idx() == 0) context.plan.release_tmem();
    }
};

template <class KernelTraits>
struct IndexerPrepareDispatch<KernelTraits, false, true> {
    using Roles = typename KernelTraits::Roles;
    using MainLoop = CollectiveIndexer<KernelTraits>;
    using Epilogue = CollectiveIndexerEpilogue<KernelTraits>;

    static CUTE_DEVICE void configure_registers() {
        const int warp_idx = KernelTraits::warp_idx();
        if (warp_idx < Roles::DrainWarpBegin)
            cutlass::arch::warpgroup_reg_dealloc<
                KernelTraits::Registers::LocalQueryProducer>();
        else if (Roles::is_drain(warp_idx))
            cutlass::arch::warpgroup_reg_alloc<
                KernelTraits::Registers::AccumulatorConsumer>();
        else
            cutlass::arch::warpgroup_reg_dealloc<
                KernelTraits::Registers::Producer>();
    }

    template <class Context>
    static CUTE_DEVICE void run(Context const& context,
                                MainLoop const& main_loop,
                                Epilogue const& epilogue) {
        configure_registers();
        const int warp_idx = KernelTraits::warp_idx();
        const bool has_key_weight_work =
            blockIdx.x < context.params.key_weight_work_items;
        if (Roles::is_mma(warp_idx)) {
            if (has_key_weight_work)
                main_loop.template run_matrix_products<true>(context);
            else
                main_loop.mma_decode(context);
        } else if (Roles::is_weight_loader(warp_idx)) {
            if (has_key_weight_work)
                main_loop.template load_projection_weights<true>(context);
            else
                main_loop.weight_decode(context);
        } else if (Roles::is_activation_loader(warp_idx)) {
            if (has_key_weight_work)
                main_loop.template load_projection_activations<true>(context);
            else if (context.params.num_tokens > 4)
                main_loop.activation_small(context);
        } else if (Roles::is_prefetch(warp_idx)) {
            if (has_key_weight_work)
                epilogue.small_key_consumer(context);
            else
                main_loop.prefetch_small_rope(context);
        } else if (Roles::is_drain(warp_idx)) {
            if (has_key_weight_work)
                epilogue.template consume_accumulators<true>(context);
            else
                epilogue.drain_small_q(context);
        } else if (Roles::is_vector(warp_idx)) {
            if (has_key_weight_work) {
                if (context.params.num_tokens > 4)
                    epilogue.quantize_query_lora(context);
            } else if (context.params.num_tokens <= 4) {
                main_loop.template quantize_local_q<4>(context);
            }
        }
    }

    template <class Context>
    static CUTE_DEVICE void finish(Context const& context, Epilogue const&) {
        __syncthreads();
        if (KernelTraits::warp_idx() == 0) context.plan.release_tmem();
        if (context.params.num_tokens > 4)
            complete_indexer_grid<KernelTraits>(context);
    }
};

template <class KernelTraits, class Parameters, bool SINGLE_TOKEN = false,
          bool LOCAL_QUERY = false>
__global__ void __launch_bounds__(SINGLE_TOKEN
                                      ? KernelTraits::SingleTokenThreads
                                      : KernelTraits::NumThreads,
                                  1, 1)
    indexer_prepare_kernel(__grid_constant__ const Parameters params) {
    using KeyWeightConfig = typename KernelTraits::KeyWeight;
    using QueryConfig = typename KernelTraits::Query;

    extern __shared__ char shared_memory[];
    IndexerSharedStorage<KernelTraits>& plan =
        *reinterpret_cast<IndexerSharedStorage<KernelTraits>*>(shared_memory);

    const int warp_idx = KernelTraits::warp_idx();
    if (warp_idx == 0) {
        if (cute::elect_one_sync()) {
            params.prefetch_descriptors();
            plan.initialize_barriers(params.eight_warp_query &&
                                             blockIdx.x >=
                                                 params.key_weight_blocks
                                         ? 256
                                         : 128);
        }
        plan.acquire_tmem();
    }
    if (params.use_pdl) cudaGridDependencySynchronize();
    __syncthreads();
    // Successors may initialize early but must wait for this grid's outputs.
    if (params.use_pdl && threadIdx.x == 0)
        cudaTriggerProgrammaticLaunchCompletion();
    if (params.prefetch_norm && warp_idx == 2 &&
        (params.uses_simt_projection()
             ? blockIdx.x == 0
             : blockIdx.x < min(params.vector_blocks, params.num_tokens))) {
        const int offset = KernelTraits::lane_idx() * 4;
        __pipeline_memcpy_async(plan.norm_weight + offset,
                                params.k_norm_weight + offset, 16);
        __pipeline_memcpy_async(plan.norm_bias + offset,
                                params.k_norm_bias + offset, 16);
        __pipeline_arrive_on(reinterpret_cast<uint64_t*>(&plan.norm_ready));
        plan.norm_ready.arrive();
    }

    char* gemm_shared_memory = plan.gemm_shared_memory();
    const GemmPipelineBarriers gemm_barriers = plan.gemm_barriers();

    const halops::indexer::KeyWeightResources<decltype(params.key_weight_tma),
                                              decltype(params.input_tma)>
        key_weight_resources{.tma_w = params.key_weight_tma,
                             .tma_act = params.input_tma,
                             .gemm_shared_memory = gemm_shared_memory,
                             .barriers = gemm_barriers,
                             .tmem_base = 0,
                             .activation_rows = params.padded_tokens,
                             .m_total = params.padded_key_weight_rows,
                             .k_dim = params.input_hidden_size,
                             .k_tiles = params.key_weight_tiles_per_split,
                             .partials = params.key_weight_partials,
                             .split_count = params.key_weight_splits};

    auto tmem_q = make_gemm_tmem<QueryConfig::BN, QueryConfig::TmaStages,
                                 KernelTraits::MaxAccStages>(0);
    const GemmResources<QueryConfig::BN, decltype(tmem_q)> query_resources{
        .weight = params.query_weight_descriptors,
        .activations = params.query_lora_descriptors,
        .gemm_shared_memory = gemm_shared_memory,
        .tmem = tmem_q,
        .barriers = gemm_barriers};

    const IndexerBarriers<KernelTraits, Parameters> barriers{params};

    using Context = halops::indexer::IndexerContext<
        KernelTraits, Parameters, decltype(key_weight_resources),
        decltype(query_resources), decltype(barriers)>;
    const Context context{params, plan, key_weight_resources, query_resources,
                          barriers};

    const halops::indexer::CollectiveIndexer<KernelTraits> main_loop;
    const halops::indexer::CollectiveIndexerEpilogue<KernelTraits> epilogue;

    using Dispatch =
        IndexerPrepareDispatch<KernelTraits, SINGLE_TOKEN, LOCAL_QUERY>;
    Dispatch::run(context, main_loop, epilogue);
    Dispatch::finish(context, epilogue);
}

}  // namespace halops::indexer
