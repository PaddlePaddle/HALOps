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

#include "indexer_math.cuh"
#include "indexer_params.cuh"

namespace halops::indexer {

template <class KernelTraits>
struct CollectiveIndexer {
    using KeyWeightConfig = typename KernelTraits::KeyWeight;
    using QueryConfig = typename KernelTraits::Query;
    using Dimensions = typename KernelTraits::Dimensions;
    static constexpr int TokenTile = KernelTraits::TokenTile;

    template <class Context>
    static CUTE_DEVICE RangeBarrierLedger
    barriers_before_query(Context const& context) {
        RangeBarrierLedger ledger;
        ledger.contribute<KeyWeightConfig::TmaStages>(
            context.params.key_weight_tiles_per_split,
            context.params.key_weight_work_start() >=
                    context.params.key_weight_work_items
                ? 0
                : (context.params.key_weight_work_items - 1 -
                   context.params.key_weight_work_start()) /
                          context.params.key_weight_work_stride() +
                      1);
        return ledger;
    }

    template <class Context>
    static CUTE_DEVICE KeyWeightWorkItem
    key_weight_work_item(Context const& context, int item) {
        return key_weight_work_item_at<KeyWeightConfig>(
            item, context.params.token_groups, context.params.key_weight_splits,
            context.params.key_weight_tiles_per_split,
            context.params.num_tokens);
    }

    template <class Context>
    static CUTE_DEVICE GemmWorkItem query_work_item(Context const& context,
                                                    int item) {
        // Spread the final partial tile across Q blocks to balance their work.
        if (context.params.num_tokens > TokenTile &&
            context.params.num_tokens % TokenTile != 0)
            item =
                (item % Dimensions::NUM_HEADS) * context.params.token_groups +
                item / Dimensions::NUM_HEADS;
        const GemmWorkList work_list{context.params.token_groups,
                                     context.params.num_tokens};
        return gemm_item<QueryConfig>(work_list, item, TokenTile,
                                      context.params.query_rows,
                                      context.params.query_tiles_per_split);
    }

    template <class Context>
    CUTE_DEVICE void mma_decode(Context const& context) const {
        const int item = context.params.query_work_start();
        auto op =
            make_mma_op<QueryConfig>(context.query_resources, 0, 0,
                                     context.params.query_tiles_per_split);
        op.begin_item(query_work_item(context, item));
        CUTLASS_PRAGMA_NO_UNROLL
        for (int k = 0; k < context.params.query_tiles_per_split; ++k) {
            op.wait_tma(k);
            op.run_k_tile(k);
            op.commit(k);
        }
        op.end_item();
    }

    template <class Context>
    CUTE_DEVICE void weight_decode(Context const& context) const {
        const int item = context.params.query_work_start();
        auto op = make_tma_op_a<QueryConfig>(
            context.query_resources, 0, ~0u,
            context.params.query_tiles_per_split, NoGate{});
        op.begin_item(context.query_resources.weight,
                      query_work_item(context, item));
        CUTLASS_PRAGMA_NO_UNROLL
        for (int k = 0; k < context.params.query_tiles_per_split; ++k)
            op.issue_tile(k);
        op.end_item();
    }

    template <class Context>
    CUTE_DEVICE void activation_small(Context const& context) const {
        auto gate = [&](int group) {
            wait_group_ready(context.barriers.query_lora_ready(group));
        };
        auto op = make_tma_op_b<QueryConfig>(
            context.query_resources, 0, ~0u,
            context.params.query_tiles_per_split, gate);
        const int item = context.params.query_work_start();
        op.begin_item(context.query_resources.weight,
                      query_work_item(context, item));
        for (int k = 0; k < context.params.query_tiles_per_split; ++k)
            op.issue_tile(k);
        op.end_item();
    }

    template <bool KEY_WEIGHT_ONLY = false, class Context>
    CUTE_DEVICE void run_matrix_products(Context const& context) const {
        {
            auto op = make_bf16_mma_op<KeyWeightConfig>(
                context.key_weight_resources, 0, 0);
            for (int item = context.params.key_weight_work_start();
                 item < context.params.key_weight_work_items;
                 item += context.params.key_weight_work_stride()) {
                op.begin_item();
                CUTLASS_PRAGMA_NO_UNROLL
                for (int k = 0; k < context.params.key_weight_tiles_per_split;
                     ++k) {
                    op.wait_tma(k);
                    op.run_k_tile(k);
                    op.commit(k);
                }
                op.end_item();
            }
        }
        if (!KEY_WEIGHT_ONLY) {
            const RangeBarrierLedger ledger = barriers_before_query(context);
            auto op = make_mma_op<QueryConfig>(
                context.query_resources, ledger.previous_stage_parities(),
                ledger.previous_accumulators(),
                context.params.query_tiles_per_split);
            for (int item = context.params.query_work_start();
                 item < context.params.query_work_items;
                 item += context.params.query_work_stride()) {
                op.begin_item(query_work_item(context, item));
                CUTLASS_PRAGMA_NO_UNROLL
                for (int k = 0; k < context.params.query_tiles_per_split; ++k) {
                    op.wait_tma(k);
                    op.run_k_tile(k);
                    op.commit(k);
                }
                op.end_item();
            }
        }
    }

    template <bool KEY_WEIGHT_ONLY = false, class Context>
    CUTE_DEVICE void load_projection_weights(Context const& context) const {
        {
            auto op = make_key_weight_loader<KeyWeightConfig>(
                context.key_weight_resources, 0, ~0u);
            for (int item = context.params.key_weight_work_start();
                 item < context.params.key_weight_work_items;
                 item += context.params.key_weight_work_stride()) {
                op.begin_item(key_weight_work_item(context, item));
                CUTLASS_PRAGMA_NO_UNROLL
                for (int k = 0; k < context.params.key_weight_tiles_per_split;
                     ++k)
                    op.issue_tile(k);
                op.end_item();
            }
        }
        if (!KEY_WEIGHT_ONLY) {
            const RangeBarrierLedger ledger = barriers_before_query(context);
            auto query_ready = [&, ready_group = -1](int token_group) mutable {
                if (token_group != ready_group) {
                    wait_group_ready(
                        context.barriers.query_lora_ready(token_group));
                    ready_group = token_group;
                }
            };
            auto op = make_tma_op_a<QueryConfig>(
                context.query_resources, ledger.previous_stage_parities(),
                ledger.unused_stage_mask(),
                context.params.query_tiles_per_split, query_ready);
            for (int item = context.params.query_work_start();
                 item < context.params.query_work_items;
                 item += context.params.query_work_stride()) {
                op.begin_item(context.query_resources.weight,
                              query_work_item(context, item));
                CUTLASS_PRAGMA_NO_UNROLL
                for (int k = 0; k < context.params.query_tiles_per_split; ++k)
                    op.issue_tile(k);
                op.end_item();
            }
        }
    }

    template <bool KEY_WEIGHT_ONLY = false, class Context>
    CUTE_DEVICE void load_projection_activations(Context const& context) const {
        {
            auto op = make_key_weight_activation_loader<KeyWeightConfig>(
                context.key_weight_resources, 0, ~0u);
            for (int item = context.params.key_weight_work_start();
                 item < context.params.key_weight_work_items;
                 item += context.params.key_weight_work_stride()) {
                op.begin_item(key_weight_work_item(context, item));
                CUTLASS_PRAGMA_NO_UNROLL
                for (int k = 0; k < context.params.key_weight_tiles_per_split;
                     ++k)
                    op.issue_tile(k);
                op.end_item();
            }
        }
        if (context.params.uses_simt_projection()) return;
        if (!KEY_WEIGHT_ONLY) {
            const RangeBarrierLedger ledger = barriers_before_query(context);
            auto query_ready = [&, ready_group = -1](int token_group) mutable {
                if (token_group != ready_group) {
                    wait_group_ready(
                        context.barriers.query_lora_ready(token_group));
                    ready_group = token_group;
                }
            };
            auto op = make_tma_op_b<QueryConfig>(
                context.query_resources, ledger.previous_stage_parities(),
                ledger.unused_stage_mask(),
                context.params.query_tiles_per_split, query_ready);
            for (int item = context.params.query_work_start();
                 item < context.params.query_work_items;
                 item += context.params.query_work_stride()) {
                op.begin_item(context.query_resources.weight,
                              query_work_item(context, item));
                CUTLASS_PRAGMA_NO_UNROLL
                for (int k = 0; k < context.params.query_tiles_per_split; ++k)
                    op.issue_tile(k);
                op.end_item();
            }
        }
    }

    template <int WARPS, bool SPLIT_STAGES = false, class Context>
    CUTE_DEVICE void quantize_local_q(Context const& context) const {
        if constexpr (TokenTile == 16) {
            const auto& params = context.params;
            using Instance = GemmInstanceFor<QueryConfig>;
            constexpr int STAGES = QueryConfig::TmaStages;
            const int lane = KernelTraits::lane_idx();
            const int w = WARPS == 1
                              ? 0
                              : KernelTraits::warp_idx() - (WARPS == 4 ? 8 : 2);
            const bool elected = cute::elect_one_sync();
            const int group_lane = lane & 7;
            const unsigned mask = 0xffu << (lane & ~7);
            for (int k = SPLIT_STAGES ? w : (WARPS == 4 ? w / 2 : 0);
                 k < Dimensions::QUERY_LORA_RANK / QueryConfig::BK;
                 k += SPLIT_STAGES ? WARPS : (WARPS == 4 ? 2 : 1)) {
                wait_stage_free<STAGES>(context.query_resources.barriers, 0,
                                        ~0u, 0, params.query_tiles_per_split,
                                        k);
                const int stage = k % STAGES;
                auto tile = cute::make_tensor(
                    cute::make_smem_ptr(reinterpret_cast<e4m3*>(
                        context.query_resources.gemm_shared_memory +
                        Instance::OPERAND_B_OFFSET +
                        stage * Instance::OPERAND_B_STAGE_BYTES)),
                    halops::indexer::intrinsics::make_umma_k_major_layout<
                        QueryConfig::BN, QueryConfig::BK, e4m3>());
                auto* scales = reinterpret_cast<uint8_t*>(
                    context.query_resources.gemm_shared_memory +
                    Instance::SCALE_B_OFFSET +
                    stage * Instance::SCALE_B_STAGE_BYTES);
                for (int token = SPLIT_STAGES ? 0 : (WARPS == 4 ? w % 2 : w);
                     token <
                     ((SPLIT_STAGES || WARPS == 1) ? 1 : params.num_tokens);
                     token += WARPS == 4 ? 2 : WARPS) {
                    CUTLASS_PRAGMA_UNROLL
                    for (int pass = 0; pass < 2; ++pass) {
                        const int group = pass * 4 + lane / 8;
                        __nv_bfloat16 values[4];
                        *reinterpret_cast<uint2*>(values) =
                            *reinterpret_cast<const uint2*>(
                                params.q_lora +
                                (int64_t)token * params.query_lora_row_stride +
                                k * QueryConfig::BK + group * 32 +
                                group_lane * 4);
                        const auto* pairs =
                            reinterpret_cast<const __nv_bfloat162*>(values);
                        const __nv_bfloat162 m =
                            __hmax2(__habs2(pairs[0]), __habs2(pairs[1]));
                        float amax =
                            fmaxf(fmaxf(__bfloat162float(__low2bfloat16(m)),
                                        __bfloat162float(__high2bfloat16(m))),
                                  1e-10f);
                        CUTLASS_PRAGMA_UNROLL
                        for (int offset = 1; offset < 8; offset <<= 1)
                            amax = fmaxf(amax,
                                         __shfl_xor_sync(mask, amax, offset));
                        const uint8_t exponent =
                            quant_scale_exponent<8, 0x600000u>(amax);
                        const __nv_bfloat162 inverse =
                            __float2bfloat162_rn(quant_scale_inverse(exponent));
                        const __nv_bfloat162 a = __hmul2(pairs[0], inverse);
                        const __nv_bfloat162 b = __hmul2(pairs[1], inverse);
                        const uint32_t packed =
                            __nv_cvt_bfloat16raw2_to_fp8x2(
                                *reinterpret_cast<const __nv_bfloat162_raw*>(
                                    &a),
                                __NV_SATFINITE, __NV_E4M3) |
                            ((uint32_t)__nv_cvt_bfloat16raw2_to_fp8x2(
                                 *reinterpret_cast<const __nv_bfloat162_raw*>(
                                     &b),
                                 __NV_SATFINITE, __NV_E4M3)
                             << 16);
                        *reinterpret_cast<uint32_t*>(
                            &tile(token, group * 32 + group_lane * 4)) = packed;
                        if (group_lane == 0)
                            scales[(group / 4) * 512 + token * 16 + group % 4] =
                                exponent;
                    }
                }
                cutlass::arch::fence_view_async_shared();
                if constexpr (WARPS == 4) {
                    if (w < 2)
                        sync_named_barrier<typename KernelTraits::Barriers::
                                               QuantizerPair01Sync>();
                    else
                        sync_named_barrier<typename KernelTraits::Barriers::
                                               QuantizerPair23Sync>();
                } else if constexpr (WARPS == 2 && !SPLIT_STAGES)
                    sync_named_barrier<
                        typename KernelTraits::Barriers::QuantizerPair01Sync>();
                else
                    __syncwarp();
                if ((SPLIT_STAGES || (WARPS == 4 ? w % 2 == 0 : w == 0)) &&
                    elected)
                    context.query_resources.barriers.tma_b[stage].arrive();
            }
        }
    }

    template <class Context>
    CUTE_DEVICE void prefetch_small_rope(Context const& context) const {
        const auto& params = context.params;
        if (!params.prefetch_small_rope) return;
        for (int token = KernelTraits::lane_idx(); token < params.num_tokens;
             token += KernelTraits::WarpSize) {
            const float* source =
                params.rope +
                (int64_t)params.position_of(token) * Dimensions::ROPE_DIM;
            if ((reinterpret_cast<uintptr_t>(source) & 15) == 0) {
                CUTLASS_PRAGMA_UNROLL
                for (int d = 0; d < Dimensions::ROPE_DIM; d += 4)
                    __pipeline_memcpy_async(
                        context.plan.small_batch_rope[token] + d, source + d,
                        16);
            } else {
                CUTLASS_PRAGMA_UNROLL
                for (int d = 0; d < Dimensions::ROPE_DIM; d += 2)
                    __pipeline_memcpy_async(
                        context.plan.small_batch_rope[token] + d, source + d,
                        8);
            }
        }
        __pipeline_arrive_on(
            reinterpret_cast<uint64_t*>(&context.plan.rope_ready[0]));
        context.plan.rope_ready[0].arrive();
    }

    template <class Context>
    CUTE_DEVICE void prefetch_weight_item(Context const& context,
                                          const GemmWorkItem& work_item,
                                          int query_sequence,
                                          int& weights_group) const {
        const auto& params = context.params;
        const int stage = query_sequence & 1;
        if (query_sequence >= 2)
            context.plan.query_weights_free[stage].wait(
                ((query_sequence - 2) / 2) & 1);
        const int group = work_item.n_row / TokenTile;
        if (group != weights_group) {
            wait_group_ready(context.barriers.weights_ready(group));
            weights_group = group;
        }
        for (int token = KernelTraits::lane_idx();
             token < work_item.valid_tokens; token += KernelTraits::WarpSize)
            __pipeline_memcpy_async(
                context.plan.query_weights[stage] + token,
                params.raw_weights +
                    (int64_t)(work_item.n_row + token) * Dimensions::NUM_HEADS +
                    work_item.weight_tile,
                4);
        __pipeline_arrive_on(reinterpret_cast<uint64_t*>(
            &context.plan.query_weights_ready[stage]));
        context.plan.query_weights_ready[stage].arrive();
    }

    template <class Context>
    CUTE_DEVICE void prefetch_raw_weights(Context const& context) const {
        int sequence = 0, weights_group = -1;
        const auto& params = context.params;
        for (int item = params.query_work_start();
             item < params.query_work_items;
             item += params.query_work_stride(), ++sequence)
            prefetch_weight_item(context, query_work_item(context, item),
                                 sequence, weights_group);
    }

    template <bool RAW_WEIGHTS = false, class Context>
    CUTE_DEVICE void prefetch_rope(Context const& context) const {
        const auto& params = context.params;
        int sequence = 0, previous_row = -1;
        int cached_first = -1, cached_second = -1;
        int query_sequence = 0, weights_group = -1;
        for (int item = params.query_work_start();
             item < params.query_work_items;
             item += params.query_work_stride(), ++query_sequence) {
            const auto work_item = query_work_item(context, item);
            if (work_item.n_row != previous_row) {
                const int stage = sequence % KernelTraits::RopeStages;
                if (sequence >= KernelTraits::RopeStages)
                    context.plan.rope_free[stage].wait(
                        ((sequence - KernelTraits::RopeStages) /
                         KernelTraits::RopeStages) &
                        1);
                const int cached_row =
                    stage == 0 ? cached_first : cached_second;
                if (work_item.n_row != cached_row) {
                    for (int token = KernelTraits::lane_idx();
                         token < work_item.valid_tokens;
                         token += KernelTraits::WarpSize) {
                        const float* source =
                            params.rope + (int64_t)params.position_of(
                                              work_item.n_row + token) *
                                              Dimensions::ROPE_DIM;
                        if ((reinterpret_cast<uintptr_t>(source) & 15) == 0) {
                            CUTLASS_PRAGMA_UNROLL
                            for (int d = 0; d < Dimensions::ROPE_DIM; d += 4)
                                __pipeline_memcpy_async(
                                    context.plan.query.rope[stage][token] + d,
                                    source + d, 16);
                        } else {
                            CUTLASS_PRAGMA_UNROLL
                            for (int d = 0; d < Dimensions::ROPE_DIM; d += 2)
                                __pipeline_memcpy_async(
                                    context.plan.query.rope[stage][token] + d,
                                    source + d, 8);
                        }
                    }
                    __pipeline_arrive_on(reinterpret_cast<uint64_t*>(
                        &context.plan.rope_ready[stage]));
                    if (stage == 0)
                        cached_first = work_item.n_row;
                    else
                        cached_second = work_item.n_row;
                }
                context.plan.rope_ready[stage].arrive();
                previous_row = work_item.n_row;
                ++sequence;
            }
            if constexpr (RAW_WEIGHTS) {
                prefetch_weight_item(context, work_item, query_sequence,
                                     weights_group);
            }
        }
    }

    template <class Context>
    CUTE_DEVICE void prefetch_query_inputs(Context const& context) const {
        if constexpr (KernelTraits::PrefetchRope) {
            if constexpr (!KernelTraits::is_mx_sf) {
                if (context.params.raw_weight_prefetch_mode ==
                    RawWeightPrefetch::RopeWarp)
                    prefetch_rope<true>(context);
                else
                    prefetch_rope<false>(context);
            } else {
                prefetch_rope<false>(context);
            }
        } else {
            if (context.params.uses_simt_projection())
                quantize_local_q<1>(context);
        }
    }
};

}  // namespace halops::indexer
