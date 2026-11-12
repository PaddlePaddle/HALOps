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

#include <utility>

#include "common/ptx.cuh"
#include "indexer_mainloop.cuh"

namespace halops::indexer {

template <size_t... I>
CUTE_DEVICE void load_q_tmem_two_lanes(float (&values)[64], uint32_t address,
                                       std::index_sequence<I...>) {
    cute::SM100_TMEM_LOAD_16dp64b64x::copy(
        address, reinterpret_cast<uint32_t&>(values[I])...);
    cutlass::arch::fence_view_async_tmem_load();
}

CUTE_DEVICE uint32_t spread_nibbles(uint32_t value) {
    value &= 0xffff;
    value = (value | (value << 8)) & 0x00ff00ffu;
    return (value | (value << 4)) & 0x0f0f0f0fu;
}

template <class KernelTraits, class Context>
CUTE_DEVICE void q_two_lane_row(Context const& context, float (&values)[64],
                                int local_token, const GemmWorkItem& work_item,
                                int rope_slot, bool acquire_weights,
                                int query_sequence) {
    const auto& params = context.params;
    if (local_token >= work_item.valid_tokens) return;
    const int token = work_item.n_row + local_token;
    const int parity = (threadIdx.x >> 1) & 1;
    const unsigned mask = __activemask();
    const float* rope;
    if constexpr (KernelTraits::PrefetchRope)
        rope = context.plan.query.rope[rope_slot][local_token];
    else
        rope = params.rope +
               (int64_t)params.position_of(token) * IndexerDimensions::ROPE_DIM;
    if (params.is_neox_style) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < 16; i += 2) {
            const int d = i * 2 + parity;
            const float2 a = make_float2(values[i], values[i + 1]);
            const float2 b = make_float2(values[i + 16], values[i + 17]);
            const float2 cosine = make_float2(rope[d], rope[d + 2]);
            const float2 sine = make_float2(rope[32 + d], rope[34 + d]);
            const float2 left = __ffma2_rn(
                a, cosine, __fmul2_rn(make_float2(-b.x, -b.y), sine));
            const float2 right = __ffma2_rn(a, sine, __fmul2_rn(b, cosine));
            values[i] = left.x;
            values[i + 1] = left.y;
            values[i + 16] = right.x;
            values[i + 17] = right.y;
        }
    } else {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < 32; ++i) {
            const float peer = __shfl_xor_sync(mask, values[i], 2);
            values[i] = parity ? peer * rope[32 + i] + values[i] * rope[i]
                               : values[i] * rope[i] - peer * rope[32 + i];
        }
    }
    if (params.apply_hadamard) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < 64; i += 2) {
            const float2 peer =
                make_float2(__shfl_xor_sync(mask, values[i], 2),
                            __shfl_xor_sync(mask, values[i + 1], 2));
            const float sign = parity ? -1.f : 1.f;
            const float2 pair =
                __ffma2_rn(make_float2(sign, sign),
                           make_float2(values[i], values[i + 1]), peer);
            values[i] = pair.x;
            values[i + 1] = pair.y;
        }
        thread_hadamard<64>(values);
        scale_values<64>(values, IndexerDimensions::HADAMARD_SCALE);
    }
    const int64_t row =
        (int64_t)token * IndexerDimensions::NUM_HEADS + work_item.weight_tile;
    float scale = 1.f;
    if (params.is_fp4) {
        uint32_t scale_word = 0;
        CUTLASS_PRAGMA_UNROLL
        for (int group = 0; group < 4; ++group) {
            float maximum = thread_amax<16>(values, group * 16);
            maximum = fmaxf(maximum, __shfl_xor_sync(mask, maximum, 2));
            float inverse;
            const uint8_t exponent = mxfp4_scale(maximum, inverse);
            scale_word |= uint32_t(exponent) << (group * 8);
            uint32_t first = 0, second = 0;
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < 4; ++i) {
                first |= uint32_t(pack_scaled_e2m1x2(
                             values[group * 16 + 2 * i],
                             values[group * 16 + 2 * i + 1], inverse))
                         << (i * 8);
                second |= uint32_t(pack_scaled_e2m1x2(
                              values[group * 16 + 8 + 2 * i],
                              values[group * 16 + 8 + 2 * i + 1], inverse))
                          << (i * 8);
            }
            const uint32_t peer_first = __shfl_xor_sync(mask, first, 2);
            const uint32_t peer_second = __shfl_xor_sync(mask, second, 2);
            const uint32_t even = parity ? peer_second : first;
            const uint32_t odd = parity ? second : peer_first;
            *reinterpret_cast<uint2*>(params.q_output + row * 64 + group * 16 +
                                      parity * 8) =
                make_uint2(spread_nibbles(even) | (spread_nibbles(odd) << 4),
                           spread_nibbles(even >> 16) |
                               (spread_nibbles(odd >> 16) << 4));
        }
        if (parity == 0)
            store_output_scales(params.q_scale + row * 4, scale_word);
    } else {
        float inverse;
        if (!params.is_mx_sf) {
            float maximum = thread_amax<64>(values);
            maximum = fmaxf(maximum, __shfl_xor_sync(mask, maximum, 2));
            scale = fp8_row_scale(maximum);
            inverse = 1.f / scale;
        }
        uint32_t scale_word = 0;
        CUTLASS_PRAGMA_UNROLL
        for (int chunk = 0; chunk < 4; ++chunk) {
            if (params.is_mx_sf) {
                float maximum = thread_amax<16>(values, chunk * 16);
                maximum = fmaxf(maximum, __shfl_xor_sync(mask, maximum, 2));
                scale_word |= uint32_t(mxfp8_scale(maximum, inverse))
                              << (chunk * 8);
            }
            uint32_t words[4];
            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < 2; ++j) {
                const int begin = chunk * 16 + j * 4;
                const uint32_t first = mxfp8::pack_e4m3x4(
                    values[begin], values[begin + 1], values[begin + 2],
                    values[begin + 3], inverse);
                const uint32_t second = mxfp8::pack_e4m3x4(
                    values[begin + 8], values[begin + 9], values[begin + 10],
                    values[begin + 11], inverse);
                // Exchange only the half written by the other lane.
                const uint32_t local = parity ? second : first;
                const uint32_t peer =
                    __shfl_xor_sync(mask, parity ? first : second, 2);
                const uint32_t even = parity ? peer : local;
                const uint32_t odd = parity ? local : peer;
                words[j * 2] = __byte_perm(even, odd, 0x5140);
                words[j * 2 + 1] = __byte_perm(even, odd, 0x7362);
            }
            *reinterpret_cast<uint4*>(params.q_output + row * 128 + chunk * 32 +
                                      parity * 16) =
                make_uint4(words[0], words[1], words[2], words[3]);
        }
        if (params.is_mx_sf && parity == 0)
            store_output_scales(params.q_scale + row * 4, scale_word);
    }
    if (!params.independent_weights) {
        if (params.raw_weight_prefetch_mode)
            context.plan.query_weights_ready[query_sequence & 1].wait(
                (query_sequence / 2) & 1);
        else if (acquire_weights)
            wait_group_ready_active(context.barriers.weights_ready(
                work_item.n_row / KernelTraits::TokenTile));
        if (parity == 0) {
            const float raw =
                params.raw_weight_prefetch_mode
                    ? context.plan
                          .query_weights[query_sequence & 1][local_token]
                    : params.raw_weights[row];
            params.store_weights(row, raw * params.weight_scale * scale);
        }
    }
}

// Four drain warps cover the two 16-row halves of each TMEM quadrant in
// successive passes. Each lane holds only 64 values throughout the epilogue.
template <class KernelTraits, class Context>
CUTE_DEVICE void drain_q_four_split(Context const& context) {
    const auto& params = context.params;
    const auto ledger =
        CollectiveIndexer<KernelTraits>::barriers_before_query(context);
    const int warp = KernelTraits::warp_idx() - 4;
    const int lane = KernelTraits::lane_idx();
    int sequence = 0, previous_group = -1, rope_group = -1, rope_sequence = -1;
    for (int item = params.query_work_start(); item < params.query_work_items;
         item += params.query_work_stride(), ++sequence) {
        const auto work_item =
            CollectiveIndexer<KernelTraits>::query_work_item(context, item);
        const int group = work_item.n_row / KernelTraits::TokenTile;
        const int slot = wait_accumulator_slot<KernelTraits::Query::AccStages>(
            context.query_resources.barriers, ledger.previous_accumulators(),
            sequence);
        halops::tcgen05_after_thread_sync();
        if constexpr (KernelTraits::PrefetchRope) {
            if (work_item.n_row != rope_group) {
                ++rope_sequence;
                context.plan
                    .rope_ready[rope_sequence % KernelTraits::RopeStages]
                    .wait((rope_sequence / KernelTraits::RopeStages) & 1);
                rope_group = work_item.n_row;
            }
        }
        CUTLASS_PRAGMA_NO_UNROLL
        for (int half = 0; half < 2; ++half) {
            if (half * 16 >= work_item.valid_tokens) break;
            float values[64];
            const uint32_t address = (uint32_t(half * 16) << 16) + slot * 128;
            load_q_tmem_two_lanes(values, address,
                                  std::make_index_sequence<64>{});
            const int token = warp * 32 + half * 16 + lane / 4 + (lane & 1) * 8;
            q_two_lane_row<KernelTraits>(
                context, values, token, work_item,
                rope_sequence % KernelTraits::RopeStages,
                group != previous_group, sequence);
        }
        previous_group = group;
        if (params.raw_weight_prefetch_mode)
            context.plan.query_weights_free[sequence & 1].arrive();
        if constexpr (KernelTraits::PrefetchRope) {
            const int next = item + params.query_work_stride();
            if (next >= params.query_work_items ||
                CollectiveIndexer<KernelTraits>::query_work_item(context, next)
                        .n_row != work_item.n_row)
                context.plan.rope_free[rope_sequence % KernelTraits::RopeStages]
                    .arrive();
        }
        cutlass::arch::fence_view_async_tmem_load();
        context.query_resources.barriers.accumulator_free[slot].arrive();
    }
}

// Eight drain warps cover 128 tokens. Each pair of lanes owns alternating
// head dimensions, directly from the 16dp64b TMEM layout.
template <class KernelTraits, class Context>
CUTE_DEVICE void drain_q_eight(Context const& context) {
    const auto& params = context.params;
    const int warp = KernelTraits::warp_idx() - 4;
    const int lane = KernelTraits::lane_idx();
    int sequence = 0, previous_group = -1, rope_group = -1, rope_sequence = -1;
    for (int item = params.query_work_start(); item < params.query_work_items;
         item += params.query_work_stride(), ++sequence) {
        const auto work_item =
            CollectiveIndexer<KernelTraits>::query_work_item(context, item);
        const int group = work_item.n_row / KernelTraits::TokenTile;
        const int slot = wait_accumulator_slot<KernelTraits::Query::AccStages>(
            context.query_resources.barriers, 0, sequence);
        halops::tcgen05_after_thread_sync();
        if constexpr (KernelTraits::PrefetchRope) {
            if (work_item.n_row != rope_group) {
                ++rope_sequence;
                context.plan
                    .rope_ready[rope_sequence % KernelTraits::RopeStages]
                    .wait((rope_sequence / KernelTraits::RopeStages) & 1);
                rope_group = work_item.n_row;
            }
        }
        float values[64];
        const uint32_t address = (uint32_t((warp / 4) * 16) << 16) + slot * 128;
        load_q_tmem_two_lanes(values, address, std::make_index_sequence<64>{});
        const int token =
            (warp % 4) * 32 + (warp / 4) * 16 + lane / 4 + (lane & 1) * 8;
        q_two_lane_row<KernelTraits>(context, values, token, work_item,
                                     rope_sequence % KernelTraits::RopeStages,
                                     group != previous_group, sequence);
        previous_group = group;
        if (params.raw_weight_prefetch_mode)
            context.plan.query_weights_free[sequence & 1].arrive();
        if constexpr (KernelTraits::PrefetchRope) {
            const int next = item + params.query_work_stride();
            if (next >= params.query_work_items ||
                CollectiveIndexer<KernelTraits>::query_work_item(context, next)
                        .n_row != work_item.n_row)
                context.plan.rope_free[rope_sequence % KernelTraits::RopeStages]
                    .arrive();
        }
        cutlass::arch::fence_view_async_tmem_load();
        context.query_resources.barriers.accumulator_free[slot].arrive();
    }
}

template <class KernelTraits>
struct CollectiveIndexerEpilogue {
    using KeyWeightConfig = typename KernelTraits::KeyWeight;
    using QueryConfig = typename KernelTraits::Query;
    using Dimensions = typename KernelTraits::Dimensions;
    using Roles = typename KernelTraits::Roles;
    static constexpr int TokenTile = KernelTraits::TokenTile;

    static constexpr int VALUES_PER_LANE = Dimensions::HEAD_DIM / 32;
    static_assert(VALUES_PER_LANE == 4,
                  "the float4 partial reads assume 4 elements/lane");
    static_assert(Dimensions::NUM_HEADS % VALUES_PER_LANE == 0 &&
                      Dimensions::NUM_HEADS / VALUES_PER_LANE <= 32,
                  "the weights_raw fan-out gives the first N_HEADS/4 lanes "
                  "four heads each, so it must fit inside one warp");

    static CUTE_DEVICE float warp_sum(float x) {
        CUTLASS_PRAGMA_UNROLL
        for (int o = 16; o > 0; o >>= 1)
            x += __shfl_xor_sync(0xffffffffu, x, o);
        return x;
    }
    static CUTE_DEVICE float warp_max(float x) {
        CUTLASS_PRAGMA_UNROLL
        for (int o = 16; o > 0; o >>= 1)
            x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, o));
        return x;
    }

    template <class Context>
    CUTE_DEVICE void k_token(
        Context const& context, int token, int lane,
        const float (&norm_weights)[VALUES_PER_LANE],
        const float (&norm_biases)[VALUES_PER_LANE]) const {
        const auto& params = context.params;
        const int split_count = params.key_weight_splits;

        const int cache_slot = params.cache_loc_of(token);
        const float* rope = params.rope + (int64_t)params.position_of(token) *
                                              Dimensions::ROPE_DIM;

        const float* partials =
            params.key_weight_partials +
            (int64_t)token * (split_count * Dimensions::KEY_WEIGHT_ROWS);
        float values[VALUES_PER_LANE] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 4
        for (int s = 0; s < split_count; ++s) {
            const float4 q = *reinterpret_cast<const float4*>(
                partials + s * Dimensions::KEY_WEIGHT_ROWS +
                lane * VALUES_PER_LANE);
            values[0] += q.x;
            values[1] += q.y;
            values[2] += q.z;
            values[3] += q.w;
        }
        k_finish(context, lane, cache_slot, rope, norm_weights, norm_biases,
                 values);
    }

    template <int FIRST = Roles::VectorWarpBegin,
              int WARPS = Roles::VectorWarps, class Context>
    CUTE_DEVICE void k_epilogue(Context const& context) const {
        const auto& params = context.params;
        const int lane = KernelTraits::lane_idx();
        const int weight_values = KernelTraits::warp_idx() - FIRST;
        const int split_count = params.key_weight_splits;
        const int stride = split_count * Dimensions::KEY_WEIGHT_ROWS;
        const int step = params.vector_blocks * WARPS;
        const int first_token =
            (int)blockIdx.x + params.vector_blocks * weight_values;
        if (first_token >= params.num_tokens) return;
        const bool has_weight_row =
            lane * VALUES_PER_LANE < Dimensions::NUM_HEADS;

        if (params.prefetch_norm) context.plan.norm_ready.wait(0);
        const float4 weight_vector = *reinterpret_cast<const float4*>(
            (params.prefetch_norm ? context.plan.norm_weight
                                  : params.k_norm_weight) +
            lane * VALUES_PER_LANE);
        const float4 bias_vector =
            params.use_layernorm
                ? *reinterpret_cast<const float4*>((params.prefetch_norm
                                                        ? context.plan.norm_bias
                                                        : params.k_norm_bias) +
                                                   lane * VALUES_PER_LANE)
                : make_float4(0.f, 0.f, 0.f, 0.f);
        const float norm_weights[VALUES_PER_LANE] = {
            weight_vector.x, weight_vector.y, weight_vector.z, weight_vector.w};
        const float norm_biases[VALUES_PER_LANE] = {
            bias_vector.x, bias_vector.y, bias_vector.z, bias_vector.w};

        for (int token = first_token; token < params.num_tokens;
             token += step) {
            wait_group_ready(
                context.barriers.key_weight_ready(token / TokenTile));
            if (has_weight_row) {
                const float* weight_partials = params.key_weight_partials +
                                               (int64_t)token * stride +
                                               Dimensions::HEAD_DIM;
                float raw_weights[VALUES_PER_LANE] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 4
                for (int s = 0; s < split_count; ++s) {
                    const float4 q = *reinterpret_cast<const float4*>(
                        weight_partials + s * Dimensions::KEY_WEIGHT_ROWS +
                        lane * VALUES_PER_LANE);
                    raw_weights[0] += q.x;
                    raw_weights[1] += q.y;
                    raw_weights[2] += q.z;
                    raw_weights[3] += q.w;
                }
                if (params.independent_weights) {
                    params.store_weights(
                        (int64_t)token * Dimensions::NUM_HEADS +
                            lane * VALUES_PER_LANE,
                        make_float4(raw_weights[0] * params.weight_scale,
                                    raw_weights[1] * params.weight_scale,
                                    raw_weights[2] * params.weight_scale,
                                    raw_weights[3] * params.weight_scale));
                } else {
                    *reinterpret_cast<float4*>(params.raw_weights +
                                               (int64_t)token *
                                                   Dimensions::NUM_HEADS +
                                               lane * VALUES_PER_LANE) =
                        make_float4(raw_weights[0], raw_weights[1],
                                    raw_weights[2], raw_weights[3]);
                }
            }
            if (!params.independent_weights)
                context.barriers.weights_ready(token / TokenTile).arrive_warp();
        }
        for (int token = first_token; token < params.num_tokens; token += step)
            k_token(context, token, lane, norm_weights, norm_biases);
    }

    template <class Context>
    CUTE_DEVICE void k_finish(Context const& context, int lane, int cache_slot,
                              const float* rope,
                              const float (&norm_weights)[VALUES_PER_LANE],
                              const float (&norm_biases)[VALUES_PER_LANE],
                              float (&values)[VALUES_PER_LANE]) const {
        const auto& params = context.params;
        {
            if (params.use_layernorm) {
                float sum_values = 0.f;
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < VALUES_PER_LANE; ++i)
                    sum_values += values[i];
                const float mean = warp_sum(sum_values) / Dimensions::HEAD_DIM;
                float sum_squares = 0.f;
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < VALUES_PER_LANE; ++i) {
                    values[i] -= mean;
                    sum_squares += values[i] * values[i];
                }
                const float inverse_norm =
                    rsqrtf(warp_sum(sum_squares) / Dimensions::HEAD_DIM +
                           params.k_norm_eps);
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < VALUES_PER_LANE; ++i)
                    values[i] = values[i] * inverse_norm * norm_weights[i] +
                                norm_biases[i];
            } else {
                float sum_squares = 0.f;
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < VALUES_PER_LANE; ++i)
                    sum_squares += values[i] * values[i];
                const float inverse_norm =
                    rsqrtf(warp_sum(sum_squares) / Dimensions::HEAD_DIM +
                           params.k_norm_eps);
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < VALUES_PER_LANE; ++i)
                    values[i] = values[i] * inverse_norm * norm_weights[i];
            }

            group_rope<32>(values, lane, rope, params.is_neox_style);
            if (params.apply_hadamard) {
                warp_hadamard_128(values, lane);
                scale_values<VALUES_PER_LANE>(values,
                                              Dimensions::HADAMARD_SCALE);
            }

            if (cache_slot >= 0) {
                const KCacheSlot slot = k_cache_slot(
                    params.key_cache, params.cache_row_bytes, params.page_size,
                    params.page_shift, cache_slot, params.is_fp4);
                group_quantize_output<32>(values, lane, slot.data, slot.scale,
                                          params.is_mx_sf, params.is_fp4);
            }
        }
    }

    template <bool RESET, class Context>
    CUTE_DEVICE void decode_key_consumer(Context const& context) const {
        const auto& params = context.params;
        if (blockIdx.x != 0) return;
        const int lane = KernelTraits::lane_idx();
        wait_group_ready(context.barriers.key_weight_ready(0));
        const float4 key = *reinterpret_cast<const float4*>(
            params.key_weight_partials + lane * VALUES_PER_LANE);
        if (params.prefetch_norm) context.plan.norm_ready.wait(0);
        const float4 weight = *reinterpret_cast<const float4*>(
            (params.prefetch_norm ? context.plan.norm_weight
                                  : params.k_norm_weight) +
            lane * VALUES_PER_LANE);
        const float4 bias =
            params.use_layernorm
                ? *reinterpret_cast<const float4*>((params.prefetch_norm
                                                        ? context.plan.norm_bias
                                                        : params.k_norm_bias) +
                                                   lane * VALUES_PER_LANE)
                : make_float4(0.f, 0.f, 0.f, 0.f);
        float values[VALUES_PER_LANE] = {key.x, key.y, key.z, key.w};
        const float weight_values[VALUES_PER_LANE] = {weight.x, weight.y,
                                                      weight.z, weight.w};
        const float bias_values[VALUES_PER_LANE] = {bias.x, bias.y, bias.z,
                                                    bias.w};
        const float* rope =
            params.rope + (int64_t)params.position_of(0) * Dimensions::ROPE_DIM;
        k_finish(context, lane, params.cache_loc_of(0), rope, weight_values,
                 bias_values, values);
        if constexpr (RESET) {
            // This warp is the last reader of the active K/weight counters.
            __syncwarp();
            context.barriers.reset_active_warp();
        }
    }

    template <bool DRAIN = false, bool DEFAULT_HIDDEN = false, class Context>
    CUTE_DEVICE void compute_simt_key_weights(Context const& context) const {
        const auto& params = context.params;
        const int lane = KernelTraits::lane_idx();
        const int vector_warp =
            KernelTraits::warp_idx() -
            (DRAIN ? Roles::DrainWarpBegin : Roles::VectorWarpBegin);
        const int head = blockIdx.x;
        if (vector_warp < 3) {
            const int row = vector_warp < 2 ? 2 * head + vector_warp
                                            : Dimensions::HEAD_DIM + head;
            const __nv_bfloat16* w =
                params.key_weight_projection +
                (int64_t)row * (DEFAULT_HIDDEN ? Dimensions::DEFAULT_HIDDEN
                                               : params.input_hidden_size);
            float2 sum[4] = {};
            CUTLASS_PRAGMA_UNROLL
            for (int k = lane * 8;
                 k < (DEFAULT_HIDDEN ? Dimensions::DEFAULT_HIDDEN
                                     : params.input_hidden_size);
                 k += 32 * 8) {
                __nv_bfloat16 input_values[8], weight_values[8];
                *reinterpret_cast<uint4*>(input_values) =
                    *reinterpret_cast<const uint4*>(params.x + k);
                *reinterpret_cast<uint4*>(weight_values) =
                    *reinterpret_cast<const uint4*>(w + k);
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < 4; ++i) {
                    const float2 a = __bfloat1622float2(
                        reinterpret_cast<const __nv_bfloat162*>(
                            input_values)[i]);
                    const float2 b = __bfloat1622float2(
                        reinterpret_cast<const __nv_bfloat162*>(
                            weight_values)[i]);
                    sum[i] = __ffma2_rn(a, b, sum[i]);
                }
            }
            const float total =
                warp_sum((sum[0].x + sum[0].y) + (sum[1].x + sum[1].y) +
                         (sum[2].x + sum[2].y) + (sum[3].x + sum[3].y));
            if (lane == 0) {
                if (vector_warp < 2)
                    params.key_weight_partials[row] = total;
                else if (params.independent_weights) {
                    params.store_weights(head, total * params.weight_scale);
                } else {
                    context.plan.local_weight = total;
                    context.plan.local_weight_ready.arrive();
                }
            }
        }
        using Sync = cute::conditional_t<
            DRAIN, typename KernelTraits::Barriers::AccumulatorGroupSync,
            typename KernelTraits::Barriers::VectorGroupSync>;
        sync_named_barrier<Sync>();
        if (vector_warp == 0) {
            context.barriers.key_weight_ready(0).arrive_warp();
            if constexpr (!DRAIN) decode_key_consumer<false>(context);
        }
    }

    template <class Context>
    CUTE_DEVICE void drain_decode(Context const& context) const {
        if (context.params.input_hidden_size ==
            KernelTraits::Dimensions::DEFAULT_HIDDEN)
            compute_simt_key_weights<true, true>(context);
        else
            compute_simt_key_weights<true, false>(context);
        auto op = make_drain_op<QueryConfig>(context.query_resources, 0);
        const int item = blockIdx.x;
        const auto work_item =
            CollectiveIndexer<KernelTraits>::query_work_item(context, item);
        op.wait_mma_and_load();
        q_epilogue_transposed<32>(context, op, work_item);
        op.end_item();
    }

    template <class Context>
    CUTE_DEVICE void small_key_consumer(Context const& context) const {
        const auto& params = context.params;
        const int first_token = blockIdx.x;
        if (first_token >= params.num_tokens) return;
        const int lane = KernelTraits::lane_idx();
        wait_group_ready(context.barriers.key_weight_ready(0));
        if (params.prefetch_norm) context.plan.norm_ready.wait(0);
        const float4 weight = *reinterpret_cast<const float4*>(
            (params.prefetch_norm ? context.plan.norm_weight
                                  : params.k_norm_weight) +
            lane * VALUES_PER_LANE);
        const float4 bias =
            params.use_layernorm
                ? *reinterpret_cast<const float4*>((params.prefetch_norm
                                                        ? context.plan.norm_bias
                                                        : params.k_norm_bias) +
                                                   lane * VALUES_PER_LANE)
                : make_float4(0.f, 0.f, 0.f, 0.f);
        const float weight_values[VALUES_PER_LANE] = {weight.x, weight.y,
                                                      weight.z, weight.w};
        const float bias_values[VALUES_PER_LANE] = {bias.x, bias.y, bias.z,
                                                    bias.w};
        for (int token = first_token; token < params.num_tokens;
             token += params.key_weight_work_items) {
            if (params.independent_weights && params.num_tokens > 4) {
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
                if (params.independent_weights) {
                    weight = __fmul2_rn(
                        weight,
                        make_float2(params.weight_scale, params.weight_scale));
                    params.store_weights(
                        (int64_t)token * Dimensions::NUM_HEADS + lane * 2,
                        weight);
                } else {
                    *reinterpret_cast<float2*>(
                        params.raw_weights +
                        (int64_t)token * Dimensions::NUM_HEADS + lane * 2) =
                        weight;
                    context.barriers.weights_ready(0).arrive_warp();
                }
            }
            k_token(context, token, lane, weight_values, bias_values);
        }
        // Larger batches must wait for Q consumers before resetting counters.
        if (params.num_tokens > 4) return;
        __syncwarp();
        int prev = 0;
        if (lane == 0)
            prev =
                halops::gmem_atom_add1_acq_rel(params.grid_completion_counter);
        if (__shfl_sync(0xffffffffu, prev, 0) == params.num_tokens - 1) {
            context.barriers.reset_active_warp();
            __syncwarp();
            if (lane == 0)
                halops::gmem_st_zero_relaxed(params.grid_completion_counter);
        }
    }

    template <bool DEFAULT_HIDDEN = false, class Context>
    CUTE_DEVICE void drain_small_q(Context const& context) const {
        const auto& params = context.params;
        const int lane = KernelTraits::lane_idx();
        const int consumer_warp =
            Roles::drain_local_warp(KernelTraits::warp_idx());
        const int head = params.query_work_start();
        if (!params.independent_weights && params.num_tokens > 4) {
            wait_group_ready(context.barriers.key_weight_ready(0));
            if (params.num_tokens > 16) {
                const int tid = consumer_warp * 32 + lane;
                for (int token = tid / 4; token < params.num_tokens;
                     token += 32) {
                    float total = 0.f;
                    CUTLASS_PRAGMA_UNROLL
                    for (int split = 0; split < 16; split += 4)
                        if (split < params.key_weight_splits)
                            total += params.key_weight_partials
                                         [((int64_t)token *
                                               params.key_weight_splits +
                                           split + (lane & 3)) *
                                              Dimensions::KEY_WEIGHT_ROWS +
                                          Dimensions::HEAD_DIM + head];
                    const unsigned active = __activemask();
                    total += __shfl_xor_sync(active, total, 1, 4);
                    total += __shfl_xor_sync(active, total, 2, 4);
                    if ((lane & 3) == 0)
                        context.plan.local_weights[token] = total;
                }
            } else {
                for (int token = consumer_warp; token < params.num_tokens;
                     token += Roles::DrainWarps) {
                    const float partial =
                        lane < params.key_weight_splits
                            ? params.key_weight_partials
                                  [((int64_t)token * params.key_weight_splits +
                                    lane) *
                                       Dimensions::KEY_WEIGHT_ROWS +
                                   Dimensions::HEAD_DIM + head]
                            : 0.f;
                    const float total = warp_sum(partial);
                    if (lane == 0) context.plan.local_weights[token] = total;
                }
            }
        }
        if (params.num_tokens <= 4)
            for (int token = consumer_warp; token < params.num_tokens;
                 token += Roles::DrainWarps) {
                const auto* x = params.x + (int64_t)token * params.x_row_stride;
                const auto* w = params.key_weight_projection +
                                (int64_t)(Dimensions::HEAD_DIM + head) *
                                    (DEFAULT_HIDDEN ? Dimensions::DEFAULT_HIDDEN
                                                    : params.input_hidden_size);
                float2 sum[4] = {};
                CUTLASS_PRAGMA_UNROLL
                for (int k = lane * 8;
                     k < (DEFAULT_HIDDEN ? Dimensions::DEFAULT_HIDDEN
                                         : params.input_hidden_size);
                     k += 32 * 8) {
                    __nv_bfloat16 input_values[8], weight_values[8];
                    *reinterpret_cast<uint4*>(input_values) =
                        *reinterpret_cast<const uint4*>(x + k);
                    *reinterpret_cast<uint4*>(weight_values) =
                        *reinterpret_cast<const uint4*>(w + k);
                    CUTLASS_PRAGMA_UNROLL
                    for (int i = 0; i < 4; ++i) {
                        const float2 a = __bfloat1622float2(
                            reinterpret_cast<const __nv_bfloat162*>(
                                input_values)[i]);
                        const float2 b = __bfloat1622float2(
                            reinterpret_cast<const __nv_bfloat162*>(
                                weight_values)[i]);
                        sum[i] = __ffma2_rn(a, b, sum[i]);
                    }
                }
                const float total =
                    warp_sum((sum[0].x + sum[0].y) + (sum[1].x + sum[1].y) +
                             (sum[2].x + sum[2].y) + (sum[3].x + sum[3].y));
                if (lane == 0) {
                    if (params.independent_weights)
                        params.store_weights(
                            (int64_t)token * Dimensions::NUM_HEADS + head,
                            total * params.weight_scale);
                    else
                        context.plan.local_weights[token] = total;
                }
            }
        if (!params.independent_weights)
            sync_named_barrier<
                typename KernelTraits::Barriers::AccumulatorGroupSync>();
        auto op = make_drain_op<QueryConfig>(context.query_resources, 0);
        const auto work_item =
            CollectiveIndexer<KernelTraits>::query_work_item(context, head);
        op.wait_mma_and_load();
        if (params.prefetch_small_rope) context.plan.rope_ready[0].wait(0);
        const int query_lanes = query_epilogue_lanes(params.num_tokens, true);
        if (query_lanes == 32)
            q_epilogue_transposed<32>(context, op, work_item);
        else if (query_lanes == 16)
            q_epilogue_transposed<16>(context, op, work_item);
        else if (query_lanes == 4)
            q_epilogue_transposed<4>(context, op, work_item);
        else
            q_epilogue_transposed<8>(context, op, work_item);
        op.end_item();
    }

    template <class Context>
    CUTE_DEVICE void quantize_query_lora(Context const& context) const {
        const auto& params = context.params;
        constexpr int VEC_THREADS =
            Roles::VectorWarps * IndexerWarpTraits::WarpSize;
        constexpr int GROUPS_PER_PASS = VEC_THREADS / 8;
        const int vector_thread =
            (int)threadIdx.x -
            Roles::VectorWarpBegin * IndexerWarpTraits::WarpSize;
        const int first_scale_group = vector_thread / 8;
        const int lane_in_group = vector_thread & 7;
        const int token_groups =
            (params.num_tokens + TokenTile - 1) / TokenTile;
        const int scale_groups_per_row =
            params.query_lora_rank / GemmShapeDefaults::ScaleGroupSize;
        const int cta_stride = params.uses_local_query()
                                   ? params.key_weight_work_items
                                   : params.grid_blocks;
        const auto& layout_sfb = sf_layout(params.query_lora_descriptors);

        for (int token_group = 0; token_group < token_groups; ++token_group) {
            const int valid_tokens =
                min(TokenTile, params.num_tokens - token_group * TokenTile);
            for (int local_token = blockIdx.x; local_token < valid_tokens;
                 local_token += cta_stride) {
                const int token = token_group * TokenTile + local_token;
                const auto* input_row =
                    params.q_lora +
                    (int64_t)token * params.query_lora_row_stride;
                for (int group = first_scale_group;
                     group < scale_groups_per_row; group += GROUPS_PER_PASS) {
                    __nv_bfloat16 values[4];
                    *reinterpret_cast<uint2*>(values) =
                        *reinterpret_cast<const uint2*>(
                            input_row +
                            group * GemmShapeDefaults::ScaleGroupSize +
                            lane_in_group * 4);
                    quantize_group32_from_regs<TokenTile>(
                        values, params.quantized_query_lora,
                        params.query_lora_scales, layout_sfb, token,
                        params.query_lora_rank, group,
                        vector_thread & (IndexerWarpTraits::WarpSize - 1));
                }
                sync_named_barrier<
                    typename KernelTraits::Barriers::VectorGroupSync>();
                if (KernelTraits::warp_idx() == Roles::VectorWarpBegin)
                    context.barriers.query_lora_ready(token_group)
                        .arrive_warp();
            }
        }
    }

    template <class Context>
    CUTE_DEVICE void run_vector_operations(Context const& context) const {
        const auto& params = context.params;
        if (!params.uses_simt_projection()) {
            quantize_query_lora(context);
        }
        if (params.all_warps_process_keys || blockIdx.x >= params.vector_blocks)
            return;
        if (params.uses_simt_projection()) {
            if (params.input_hidden_size == Dimensions::DEFAULT_HIDDEN)
                compute_simt_key_weights<false, true>(context);
            else
                compute_simt_key_weights<false, false>(context);
        } else {
            k_epilogue(context);
        }
    }

    template <class Context, class Op>
    CUTE_DEVICE void q_epilogue(Context const& context, Op& op,
                                const GemmWorkItem& work_item, int rope_slot,
                                bool acquire_weights) const {
        const auto& params = context.params;
        using RegisterAccumulators =
            cute::remove_cvref_t<decltype(op.register_accumulators)>;
        static_assert(
            cute::size(RegisterAccumulators{}) == Dimensions::HEAD_DIM,
            "the Q epilogue needs a drain thread to own exactly one "
            "token's full head_dim row");
        const int local_token = cute::get<0>(op.coord(0));
        HALOPS_DEVICE_ASSERT(cute::get<0>(op.coord(Dimensions::HEAD_DIM - 1)) ==
                                 local_token &&
                             cute::get<1>(op.coord(0)) == 0 &&
                             cute::get<1>(op.coord(Dimensions::HEAD_DIM - 1)) ==
                                 Dimensions::HEAD_DIM - 1);
        if (local_token >= work_item.valid_tokens) return;
        const int token = work_item.n_row + local_token;
        const int h = work_item.weight_tile;

        auto& values = op.register_accumulators;
        const float* rope;
        if constexpr (KernelTraits::PrefetchRope)
            rope = context.plan.query.rope[rope_slot][local_token];
        else
            rope = params.rope +
                   (int64_t)params.position_of(token) * Dimensions::ROPE_DIM;
        thread_rope<KernelTraits::RopeAligned16>(values, rope,
                                                 params.is_neox_style);
        if (params.apply_hadamard) {
            thread_hadamard<Dimensions::HEAD_DIM>(values);
            scale_values<Dimensions::HEAD_DIM>(values,
                                               Dimensions::HADAMARD_SCALE);
        }

        const int64_t row = (int64_t)token * Dimensions::NUM_HEADS + h;
        const int row_bytes =
            params.is_fp4 ? Dimensions::HEAD_DIM / 2 : Dimensions::HEAD_DIM;
        const float scale = thread_quantize_output(
            values, params.q_output + row * row_bytes,
            params.is_mx_sf ? params.q_scale + row * Dimensions::MX_SCALE_BYTES
                            : nullptr,
            params.is_mx_sf, params.is_fp4, params.use_256bit_stores);

        const int64_t weight_offset =
            (int64_t)token * Dimensions::NUM_HEADS + h;
        if (!params.independent_weights) {
            float raw;
            if (params.raw_weight_prefetch_mode) {
                const int stage = op.accumulator_sequence & 1;
                context.plan.query_weights_ready[stage].wait(
                    (op.accumulator_sequence / 2) & 1);
                raw = context.plan.query_weights[stage][local_token];
            } else {
                if (acquire_weights)
                    wait_group_ready_active(context.barriers.weights_ready(
                        work_item.n_row / TokenTile));
                raw = params.raw_weights[weight_offset];
            }
            params.store_weights(weight_offset,
                                 raw * params.weight_scale * scale);
        }
    }

    template <int LANES, class Context, class Op>
    CUTE_DEVICE void q_epilogue_transposed(
        Context const& context, Op& op, const GemmWorkItem& work_item) const {
        const auto& params = context.params;
        const int lane = KernelTraits::lane_idx();
        const int consumer_warp =
            Roles::drain_local_warp(KernelTraits::warp_idx());
        constexpr int VALUES = Dimensions::HEAD_DIM / LANES;
        const int group_lane = lane % LANES;
        const int token0 = (consumer_warp * 32 + lane) / LANES;
        constexpr int TOKEN_STEP = Roles::DrainWarps * 32 / LANES;
        auto shared_column = [](int dim) {
            constexpr int SHIFT = LANES == 4 ? 3 : 2;
            return dim ^ ((dim >> 5) << SHIFT);
        };
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < cute::size(op.register_accumulators); ++i) {
            const int d = cute::get<0>(op.coord(i));
            const int token = cute::get<1>(op.coord(i));
            context.plan.query.transpose[token][shared_column(d)] =
                op.register_accumulators(i);
        }
        sync_named_barrier<
            typename KernelTraits::Barriers::AccumulatorGroupSync>();
        for (int local_token = token0; local_token < work_item.valid_tokens;
             local_token += TOKEN_STEP) {
            const int token = work_item.n_row + local_token;
            float values[VALUES];
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < VALUES; i += 4) {
                const float4 a = *reinterpret_cast<const float4*>(
                    context.plan.query.transpose[local_token] +
                    shared_column(group_lane * VALUES + i));
                values[i] = a.x;
                values[i + 1] = a.y;
                values[i + 2] = a.z;
                values[i + 3] = a.w;
            }
            const float* rope =
                params.rope +
                (int64_t)params.position_of(token) * Dimensions::ROPE_DIM;
            if constexpr (KernelTraits::TokenTile <= 32)
                if (params.prefetch_small_rope)
                    rope = context.plan.small_batch_rope[local_token];
            group_rope<LANES>(values, group_lane, rope, params.is_neox_style);
            const unsigned mask = __activemask();
            if (params.apply_hadamard) {
                thread_hadamard<VALUES>(values);
                CUTLASS_PRAGMA_UNROLL
                for (int offset = 1; offset < LANES; offset <<= 1) {
                    CUTLASS_PRAGMA_UNROLL
                    for (int i = 0; i < VALUES; ++i) {
                        const float other =
                            __shfl_xor_sync(mask, values[i], offset, LANES);
                        values[i] = (group_lane & offset) ? other - values[i]
                                                          : values[i] + other;
                    }
                }
                scale_values<VALUES>(values, Dimensions::HADAMARD_SCALE);
            }
            const int64_t row =
                (int64_t)token * Dimensions::NUM_HEADS + work_item.weight_tile;
            const int row_bytes =
                params.is_fp4 ? Dimensions::HEAD_DIM / 2 : Dimensions::HEAD_DIM;
            const float scale = group_quantize_output<LANES>(
                values, group_lane, params.q_output + row * row_bytes,
                params.is_mx_sf
                    ? params.q_scale + row * Dimensions::MX_SCALE_BYTES
                    : nullptr,
                params.is_mx_sf, params.is_fp4);
            if (group_lane == 0 && !params.independent_weights) {
                const int64_t output_offset =
                    (int64_t)token * Dimensions::NUM_HEADS +
                    work_item.weight_tile;
                const float raw = params.uses_local_query()
                                      ? context.plan.local_weights[token]
                                  : params.uses_simt_projection()
                                      ? context.plan.local_weight
                                      : params.raw_weights[output_offset];
                params.store_weights(output_offset,
                                     raw * params.weight_scale * scale);
            }
        }
        sync_named_barrier<
            typename KernelTraits::Barriers::AccumulatorGroupSync>();
    }

    template <class Context>
    CUTE_DEVICE void consume_query_accumulators(Context const& context) const {
        const auto& params = context.params;
        const RangeBarrierLedger ledger =
            CollectiveIndexer<KernelTraits>::barriers_before_query(context);
        auto op = make_drain_op<QueryConfig>(context.query_resources,
                                             ledger.previous_accumulators());
        int weights_group = -1, rope_group = -1, rope_sequence = -1;
        for (int item = params.query_work_start();
             item < params.query_work_items;
             item += params.query_work_stride()) {
            const GemmWorkItem work_item =
                CollectiveIndexer<KernelTraits>::query_work_item(context, item);
            const int token_group = work_item.n_row / TokenTile;
            if (params.uses_simt_projection())
                context.plan.local_weight_ready.wait(0);
            else if constexpr (QueryConfig::SwapAB) {
                if (!params.independent_weights &&
                    token_group != weights_group) {
                    wait_group_ready(
                        context.barriers.weights_ready(token_group));
                    weights_group = token_group;
                }
            }
            op.wait_mma_and_load();
            if constexpr (KernelTraits::PrefetchRope) {
                if (work_item.n_row != rope_group) {
                    ++rope_sequence;
                    context.plan
                        .rope_ready[rope_sequence % KernelTraits::RopeStages]
                        .wait((rope_sequence / KernelTraits::RopeStages) & 1);
                    rope_group = work_item.n_row;
                }
            }
            if constexpr (QueryConfig::SwapAB) {
                if (query_epilogue_lanes(params.num_tokens, false) == 32)
                    q_epilogue_transposed<32>(context, op, work_item);
                else
                    q_epilogue_transposed<8>(context, op, work_item);
            } else
                q_epilogue(context, op, work_item,
                           rope_sequence % KernelTraits::RopeStages,
                           token_group != weights_group);
            weights_group = token_group;
            if (params.raw_weight_prefetch_mode)
                context.plan.query_weights_free[op.accumulator_sequence & 1]
                    .arrive();
            if constexpr (KernelTraits::PrefetchRope) {
                const int next = item + params.query_work_stride();
                if (next >= params.query_work_items ||
                    CollectiveIndexer<KernelTraits>::query_work_item(context,
                                                                     next)
                            .n_row != work_item.n_row)
                    context.plan
                        .rope_free[rope_sequence % KernelTraits::RopeStages]
                        .arrive();
            }
            op.end_item();
        }
    }

    template <bool KEY_WEIGHT_ONLY = false, class Context>
    CUTE_DEVICE void consume_accumulators(Context const& context) const {
        const auto& params = context.params;
        {
            auto op = make_key_weight_consumer<KeyWeightConfig>(
                context.key_weight_resources, 0);
            for (int item = params.key_weight_work_start();
                 item < params.key_weight_work_items;
                 item += params.key_weight_work_stride()) {
                const KeyWeightWorkItem it =
                    CollectiveIndexer<KernelTraits>::key_weight_work_item(
                        context, item);
                op.wait_mma_and_load();
                op.store(it);
                op.end_item();
                sync_named_barrier<
                    typename KernelTraits::Barriers::AccumulatorGroupSync>();
                if (KernelTraits::warp_idx() == Roles::DrainWarpBegin)
                    context.barriers.key_weight_ready(it.token_group)
                        .arrive_warp();
            }
        }
        if (!KEY_WEIGHT_ONLY) {
            if constexpr (KernelTraits::SplitQueryDrain)
                drain_q_four_split<KernelTraits>(context);
            else
                consume_query_accumulators(context);
        }
    }
};

}  // namespace halops::indexer
