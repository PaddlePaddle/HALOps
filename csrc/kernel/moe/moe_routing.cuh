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

#include <cuda_bf16.h>
#include <cutlass/fast_math.h>

#include <cfloat>
#include <cstdint>
#include <cstring>

#include "moe_configs.cuh"
#include "moe_interface.h"
#include "moe_shared_storage.cuh"
#include "moe_traits.cuh"

namespace halops::moe {

// Load this lane's pair expert_ids; 0xFFFF marks a pair past the batch.
template <std::uint32_t kNumTokens, std::uint32_t kMaxTopk>
__device__ void load_token_eid_pairs(std::uint32_t n_pairs,
                                     cutlass::FastDivmod const& topk_divmod,
                                     const std::uint16_t* __restrict__ topk_ids,
                                     std::uint16_t (&expert_ids)[5]) {
    constexpr std::uint32_t kMaxRoutedPairs = kNumTokens * kMaxTopk;
    const std::uint32_t tid = WarpRole::get_lane_idx();
    const auto load = [&](std::uint32_t pair) -> std::uint16_t {
        if (pair >= n_pairs) return (std::uint16_t)0xFFFF;
        int token = 0;
        int k = 0;
        topk_divmod.fast_divmod(token, k, static_cast<int>(pair));
        return topk_ids[static_cast<std::uint32_t>(token) * kMaxTopk +
                        static_cast<std::uint32_t>(k)];
    };
    expert_ids[0] = load(tid);
    expert_ids[1] = load(tid + 32u);
    expert_ids[2] = (std::uint16_t)0xFFFF;
    expert_ids[3] = (std::uint16_t)0xFFFF;
    expert_ids[4] = (std::uint16_t)0xFFFF;
    // The wider chunks are compile-time gated; unreached ones keep 0xFFFF.
    if constexpr (kMaxRoutedPairs > 64u) {
        expert_ids[2] = load(tid + 64u);
    }
    if constexpr (kMaxRoutedPairs > 96u) {
        expert_ids[3] = load(tid + 96u);
    }
    if constexpr (kMaxRoutedPairs > 128u) {
        expert_ids[4] = load(tid + 128u);
    }
}

// Zero the per-expert token masks.
template <std::uint32_t kNumExperts>
__device__ void zero_expert_token_mask(
    std::uint32_t (&expert_token_mask)[kNumExperts]) {
    constexpr std::uint32_t kExpertsPerThread = kNumExperts / 32u;
    static_assert(kExpertsPerThread == 2u || kExpertsPerThread == 4u ||
                      kExpertsPerThread == 8u || kExpertsPerThread == 16u,
                  "kNumExperts must be one of {64,128,256,512} so one warp "
                  "covers the expert range exactly.");
    auto* base =
        &expert_token_mask[WarpRole::get_lane_idx() * kExpertsPerThread];
#pragma unroll
    for (std::uint32_t i = 0; i < kExpertsPerThread; ++i) {
        base[i] = 0u;
    }
    // The masks are then set by arbitrary lanes, so the zeroes must land
    // first.
    __syncwarp();
}

// Record which tokens route to each expert: bit `tok` set for every pair.
template <std::uint32_t kNumExperts>
__device__ void mark_routed_tokens_per_expert(
    std::uint32_t n_pairs, cutlass::FastDivmod const& topk_divmod,
    const std::uint16_t (&expert_ids)[5],
    std::uint32_t (&expert_token_mask)[kNumExperts]) {
    zero_expert_token_mask<kNumExperts>(expert_token_mask);

    const std::uint32_t tid = WarpRole::get_lane_idx();
    // The expert bound filters out the 0xFFFF sentinel chunks.
    constexpr std::uint32_t kChunks =
        sizeof(expert_ids) / sizeof(expert_ids[0]);
#pragma unroll
    for (std::uint32_t c = 0; c < kChunks; ++c) {
        const std::uint32_t pair = tid + 32u * c;
        const std::uint16_t expert_id = expert_ids[c];
        if (pair >= n_pairs || expert_id >= kNumExperts) continue;
        int tok = 0;
        [[maybe_unused]] int k = 0;
        topk_divmod.fast_divmod(tok, k, static_cast<int>(pair));
        atomicOr(&expert_token_mask[expert_id],
                 1u << static_cast<std::uint32_t>(tok));
    }
    __syncwarp();
}

// Prefix-sum the counts into CSR row offsets and list the active experts.
template <std::uint32_t kNumExperts, std::uint32_t kActiveExpertCapacity>
__device__ void build_expert_row_offsets(
    const std::uint32_t (&expert_token_mask)[kNumExperts],
    std::uint16_t (&expert_token_cumsum)[kActiveExpertCapacity + 1u],
    std::uint8_t (&expert_position)[kNumExperts],
    std::uint32_t (&active_experts)[kActiveExpertCapacity],
    std::uint32_t& active_expert_count) {
    constexpr std::uint32_t kExpertsPerThread = kNumExperts / 32;
    static_assert(kNumExperts % 32 == 0,
                  "kNumExperts must be a multiple of 32 for warp blocking of "
                  "expert_token_mask[] / expert_position[].");

    const std::uint32_t tid = WarpRole::get_lane_idx();

    // Per-lane sweep building both exclusive prefixes in one pass.
    uint32_t local_counts[kExpertsPerThread];
    uint32_t count_prefix[kExpertsPerThread];
    uint32_t active_prefix[kExpertsPerThread];
    uint32_t lane_total = 0;
    uint32_t lane_actives = 0;
    {
        const auto* cbase = &expert_token_mask[tid * kExpertsPerThread];
#pragma unroll
        for (uint32_t i = 0; i < kExpertsPerThread; ++i) {
            const uint32_t v = __popc(cbase[i]);
            local_counts[i] = v;
            count_prefix[i] = lane_total;
            active_prefix[i] = lane_actives;
            lane_total += v;
            lane_actives += (v > 0u) ? 1u : 0u;
        }
    }

    // Dual-value warp inclusive scan over both totals.
    uint32_t scan_total = lane_total;
    uint32_t scan_active = lane_actives;
#pragma unroll
    for (int off = 1; off <= 16; off *= 2) {
        const uint32_t t_total = __shfl_up_sync(FULL_MASK, scan_total, off, 32);
        const uint32_t t_active =
            __shfl_up_sync(FULL_MASK, scan_active, off, 32);
        if (static_cast<int>(tid) >= off) {
            scan_total += t_total;
            scan_active += t_active;
        }
    }
    const uint32_t lane_slot_offset = scan_total - lane_total;
    const uint32_t lane_active_offset = scan_active - lane_actives;
    // After the inclusive scan, lane 31 holds both totals.
    const uint32_t num_active_experts = __shfl_sync(FULL_MASK, scan_active, 31);
    const uint32_t num_rows = __shfl_sync(FULL_MASK, scan_total, 31);

    // Writeback in ascending expert_id: lanes and in-lane entries ascend.
    // Only active experts are written; no consumer reads a stale position.
#pragma unroll
    for (uint32_t i = 0; i < kExpertsPerThread; ++i) {
        if (local_counts[i] == 0u) continue;
        const uint32_t expert_id = tid * kExpertsPerThread + i;
        const uint32_t pos = lane_active_offset + active_prefix[i];
        active_experts[pos] = expert_id;
        expert_position[expert_id] = static_cast<uint8_t>(pos);
        expert_token_cumsum[pos] =
            static_cast<uint16_t>(lane_slot_offset + count_prefix[i]);
    }
    __syncwarp();

    if (tid == 0) {
        // Trailing row pointer: the total row count.
        expert_token_cumsum[num_active_experts] =
            static_cast<uint16_t>(num_rows);
        active_expert_count = num_active_experts;
    }
}

// Fills each routed pair's row with its combine slot and weight.
template <std::uint32_t kNumExperts, std::uint32_t kNumTokens,
          std::uint32_t kMaxTopk, std::uint32_t kActiveExpertCapacity>
__device__ void place_expert_rows(
    std::uint32_t n_pairs, cutlass::FastDivmod const& topk_divmod,
    const std::uint16_t (&expert_ids)[5],
    const DataType::ElementScale* __restrict__ topk_weights,
    const std::uint32_t (&expert_token_mask)[kNumExperts],
    const std::uint8_t (&expert_position)[kNumExperts],
    const std::uint16_t (&expert_token_cumsum)[kActiveExpertCapacity + 1u],
    std::uint8_t (&expert_row_slot)[kNumTokens * kMaxTopk],
    DataType::ElementScale (&expert_row_weight)[kNumTokens * kMaxTopk]) {
    const std::uint32_t tid = WarpRole::get_lane_idx();
    constexpr std::uint32_t kChunks =
        sizeof(expert_ids) / sizeof(expert_ids[0]);
#pragma unroll
    for (std::uint32_t c = 0; c < kChunks; ++c) {
        const std::uint32_t pair = tid + 32u * c;
        const std::uint16_t expert_id = expert_ids[c];
        if (pair >= n_pairs || expert_id >= kNumExperts) continue;
        int tok = 0;
        int k = 0;
        topk_divmod.fast_divmod(tok, k, static_cast<int>(pair));
        const std::uint32_t token = static_cast<std::uint32_t>(tok);
        // Slab position = tokens below `token` routing to this expert.
        const std::uint32_t slab_slot =
            __popc(expert_token_mask[expert_id] & ((1u << token) - 1u));
        const std::uint32_t row =
            expert_token_cumsum[expert_position[expert_id]] + slab_slot;
        // The pair's down combine slot, also its topk_weights index.
        const std::uint32_t combine_slot =
            token * kMaxTopk + static_cast<std::uint32_t>(k);
        expert_row_slot[row] = static_cast<std::uint8_t>(combine_slot);
        expert_row_weight[row] = topk_weights[combine_slot];
    }
    __syncwarp();
}

// Builds the routing tables from topk_ids; one warp runs the whole thing.
template <typename Dims>
__device__ void prepare_routing_tables(
    std::uint32_t batch_size, std::uint32_t top_k,
    cutlass::FastDivmod const& topk_divmod,
    MoESharedStorage<Dims>* __restrict__ smem) {
    constexpr std::uint32_t kNumExperts = Dims::kNumExperts;
    constexpr std::uint32_t kNumTokens = Dims::kNumTokens;
    constexpr std::uint32_t kMaxTopk = Dims::kMaxTopk;
    constexpr std::uint32_t kActiveCapacity =
        MoERoutingTables<Dims>::kMaxActiveExperts;

    constexpr std::uint32_t kMaxRoutedPairs = kNumTokens * kMaxTopk;
    static_assert(
        kMaxRoutedPairs <= 160,
        "caches up to 5 pair expert_ids per lane (BS * top_k <= 160).");

    auto& tables = smem->routing;
    auto& scratch = smem->routing_scratch;
    const std::uint32_t n_pairs = batch_size * top_k;

    std::uint16_t expert_ids[5];
    load_token_eid_pairs<kNumTokens, kMaxTopk>(n_pairs, topk_divmod,
                                               scratch.topk_ids, expert_ids);
    mark_routed_tokens_per_expert<kNumExperts>(n_pairs, topk_divmod, expert_ids,
                                               tables.expert_token_mask);
    build_expert_row_offsets<kNumExperts, kActiveCapacity>(
        tables.expert_token_mask, tables.expert_token_cumsum,
        scratch.expert_position, tables.active_experts,
        tables.active_expert_count);
    place_expert_rows<kNumExperts, kNumTokens, kMaxTopk, kActiveCapacity>(
        n_pairs, topk_divmod, expert_ids, scratch.topk_weights,
        tables.expert_token_mask, scratch.expert_position,
        tables.expert_token_cumsum, tables.expert_row_slot,
        tables.expert_row_weight);
}

}  // namespace halops::moe
