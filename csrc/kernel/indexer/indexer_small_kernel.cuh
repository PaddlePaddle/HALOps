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

#include "common/ptx.cuh"
#include "indexer_prepare_kernel.cuh"

namespace halops::indexer {

// The last consumer resets the shared arrival counter after all reads.
CUTE_DEVICE void retire_small_barrier(PartialBarrier barrier, int consumers) {
    __syncwarp();
    if (cute::elect_one_sync()) {
        const int previous = halops::gmem_atom_add1_acq_rel(barrier.slot);
        if (previous == barrier.producer_count + consumers - 1)
            barrier.reset_relaxed();
    }
}

template <class KernelTraits, class Context>
CUTE_DEVICE void three_way_weight_consumer(Context const& context) {
    using Dimensions = typename KernelTraits::Dimensions;
    const auto& params = context.params;
    const int lane = KernelTraits::lane_idx();
    if ((int)blockIdx.x - params.key_weight_splits >= params.num_tokens) return;
    wait_group_ready(context.barriers.projection_ready());
    for (int token = (int)blockIdx.x - params.key_weight_splits;
         token < params.num_tokens; token += params.key_weight_splits) {
        float2 weight = make_float2(0.f, 0.f);
#pragma unroll 4
        for (int split = 0; split < params.key_weight_splits; ++split) {
            const float2 partial = *reinterpret_cast<const float2*>(
                params.key_weight_partials +
                ((int64_t)token * params.key_weight_splits + split) *
                    Dimensions::KEY_WEIGHT_ROWS +
                Dimensions::HEAD_DIM + lane * 2);
            weight = __fadd2_rn(weight, partial);
        }
        if constexpr (KernelTraits::is_mx_sf) {
            weight = __fmul2_rn(
                weight, make_float2(params.weight_scale, params.weight_scale));
            params.store_weights(
                (int64_t)token * Dimensions::NUM_HEADS + lane * 2, weight);
        }
        retire_small_barrier(context.barriers.projection_ready(),
                             params.num_tokens);
    }
}

template <class KernelTraits, class Context>
CUTE_DEVICE void three_way_key_consumer(Context const& context) {
    const auto& params = context.params;
    const int lane = KernelTraits::lane_idx();
    if (blockIdx.x >= params.num_tokens) return;
    wait_group_ready(context.barriers.key_ready());
    context.plan.norm_ready.wait(0);
    const float4 w =
        *reinterpret_cast<const float4*>(context.plan.norm_weight + lane * 4);
    const float4 b =
        *reinterpret_cast<const float4*>(context.plan.norm_bias + lane * 4);
    const float weights[4] = {w.x, w.y, w.z, w.w};
    const float biases[4] = {b.x, b.y, b.z, b.w};
    for (int token = blockIdx.x; token < params.num_tokens;
         token += params.key_weight_splits) {
        CollectiveIndexerEpilogue<KernelTraits>{}.k_token(context, token, lane,
                                                          weights, biases);
        retire_small_barrier(context.barriers.key_ready(), params.num_tokens);
    }
}

template <class KernelTraits, class Parameters, int QueryLanes>
__global__ void __launch_bounds__(KernelTraits::NumThreads, 1, 1)
    indexer_small_batch_kernel(__grid_constant__ const Parameters params) {
    using KeyWeightConfig = typename KernelTraits::KeyWeight;
    using QueryConfig = typename KernelTraits::Query;
    extern __shared__ char shared[];
    auto& plan = *reinterpret_cast<IndexerSharedStorage<KernelTraits>*>(shared);
    const int warp = KernelTraits::warp_idx();
    // Block ranges: K, weights, then one Q block per head.
    const bool projection = blockIdx.x < params.key_weight_work_items;
    if (warp == 0) {
        if (cute::elect_one_sync()) {
            params.prefetch_descriptors();
            plan.initialize_barriers();
        }
        plan.acquire_tmem();
    }
    if (params.use_pdl) cudaGridDependencySynchronize();
    __syncthreads();
    if (params.use_pdl && threadIdx.x == 0)
        cudaTriggerProgrammaticLaunchCompletion();
    if (warp == 2 &&
        blockIdx.x < min(params.key_weight_splits, params.num_tokens)) {
        const int offset = KernelTraits::lane_idx() * 4;
        __pipeline_memcpy_async(plan.norm_weight + offset,
                                params.k_norm_weight + offset, 16);
        __pipeline_memcpy_async(plan.norm_bias + offset,
                                params.k_norm_bias + offset, 16);
        __pipeline_arrive_on(reinterpret_cast<uint64_t*>(&plan.norm_ready));
        plan.norm_ready.arrive();
    }
    const KeyWeightResources<decltype(params.key_weight_tma),
                             decltype(params.input_tma)>
        key_weight{params.key_weight_tma,
                   params.input_tma,
                   plan.gemm_shared_memory(),
                   plan.gemm_barriers(),
                   0,
                   params.padded_tokens,
                   params.padded_key_weight_rows,
                   params.input_hidden_size,
                   params.key_weight_tiles_per_split,
                   params.key_weight_partials,
                   params.key_weight_splits};
    auto tmem = make_gemm_tmem<QueryConfig::BN, QueryConfig::TmaStages,
                               KernelTraits::MaxAccStages>(0);
    const GemmResources<QueryConfig::BN, decltype(tmem)> query{
        params.query_weight_descriptors, params.query_lora_descriptors,
        plan.gemm_shared_memory(), tmem, plan.gemm_barriers()};
    const IndexerBarriers<KernelTraits, Parameters> barriers{params};
    const IndexerContext<KernelTraits, Parameters, decltype(key_weight),
                         decltype(query), decltype(barriers)>
        context{params, plan, key_weight, query, barriers};
    const CollectiveIndexer<KernelTraits> mainloop;
    const CollectiveIndexerEpilogue<KernelTraits> epilogue;
    if (warp < 4)
        cutlass::arch::warpgroup_reg_dealloc<
            KernelTraits::Registers::LocalQueryProducer>();
    else if (warp < 8)
        cutlass::arch::warpgroup_reg_alloc<
            KernelTraits::Registers::AccumulatorConsumer>();
    else
        cutlass::arch::warpgroup_reg_dealloc<
            KernelTraits::Registers::Producer>();
    if (projection) {
        if (warp == 0)
            mainloop.template run_matrix_products<true>(context);
        else if (warp == 1)
            mainloop.template load_projection_weights<true>(context);
        else if (warp == 2)
            mainloop.template load_projection_activations<true>(context);
        else if (warp == 3) {
            if (blockIdx.x < params.key_weight_splits)
                three_way_key_consumer<KernelTraits>(context);
            else if constexpr (KernelTraits::is_mx_sf)
                three_way_weight_consumer<KernelTraits>(context);
        } else if (warp < 8) {
            auto drain =
                make_key_weight_consumer<KeyWeightConfig>(key_weight, 0);
            const int item = blockIdx.x;
            const auto work =
                CollectiveIndexer<KernelTraits>::key_weight_work_item(context,
                                                                      item);
            drain.wait_mma_and_load();
            drain.store(work);
            drain.end_item();
            sync_named_barrier<
                typename KernelTraits::Barriers::AccumulatorGroupSync>();
            if (warp == 4) {
                if (work.weight_tile == 0)
                    barriers.key_ready().arrive_warp();
                else
                    barriers.projection_ready().arrive_warp();
            }
        } else {
            epilogue.quantize_query_lora(context);
        }
    } else {
        if (warp == 0)
            mainloop.mma_decode(context);
        else if (warp == 1)
            mainloop.weight_decode(context);
        else if (warp == 2) {
            mainloop.activation_small(context);
            retire_small_barrier(barriers.query_lora_ready(0), 64);
        } else if (warp == 3) {
            mainloop.prefetch_small_rope(context);
        } else if (warp < 8) {
            auto drain = make_drain_op<QueryConfig>(query, 0);
            const int head = params.query_work_start();
            const auto item =
                CollectiveIndexer<KernelTraits>::query_work_item(context, head);
            drain.wait_mma_and_load();
            if constexpr (!KernelTraits::is_mx_sf)
                plan.local_weight_ready.wait(0);
            if (params.prefetch_small_rope) plan.rope_ready[0].wait(0);
            epilogue.template q_epilogue_transposed<QueryLanes>(context, drain,
                                                                item);
            drain.end_item();
        } else if constexpr (!KernelTraits::is_mx_sf) {
            wait_group_ready(barriers.projection_ready());
            const int lane = KernelTraits::lane_idx();
            for (int token = warp - 8; token < params.num_tokens; token += 4) {
                const float partial =
                    lane < params.key_weight_splits
                        ? params.key_weight_partials
                              [((int64_t)token * params.key_weight_splits +
                                lane) *
                                   KernelTraits::Dimensions::KEY_WEIGHT_ROWS +
                               KernelTraits::Dimensions::HEAD_DIM +
                               params.query_work_start()]
                        : 0.f;
                const float total = epilogue.warp_sum(partial);
                if (lane == 0) plan.local_weights[token] = total;
            }
            sync_named_barrier<
                typename KernelTraits::Barriers::VectorGroupSync>();
            if (warp == 8) {
                if (cute::elect_one_sync()) plan.local_weight_ready.arrive();
                retire_small_barrier(barriers.projection_ready(), 64);
            }
        }
    }
    __syncthreads();
    if (warp == 0) {
        plan.release_tmem();
    }
}
}  // namespace halops::indexer
