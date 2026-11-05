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

#include <cfloat>
#include <cstdint>

#include "kernel/moe/moe_configs.cuh"
#include "kernel/moe/moe_shared_storage.cuh"
#include "kernel/moe/moe_traits.cuh"

namespace halops::moe {

// Builds the routing tables from topk_ids on the calc warps; every calc thread
// calls it, and it returns with the tables complete.  Thread t takes slot t
// and experts 2t and 2t + 1, so the experts ascend with the threads.  Token
// t's picks sit at slots t * kMaxTopk + k, which are also their combine slots.
template <typename Dims>
__device__ void prepare_routing_tables(
    std::uint32_t batch_size, std::uint32_t top_k,
    MoESharedStorage<Dims>* __restrict__ smem) {
    constexpr std::uint32_t kNumExperts = Dims::kNumExperts;
    constexpr std::uint32_t kMaxTopk = Dims::kMaxTopk;
    constexpr std::uint32_t kNumSlots = Dims::kNumTokens * kMaxTopk;
    constexpr std::uint32_t kThreads =
        WarpRole::kCalcWarpCount * WarpRole::kThreadsPerWarp;
    static_assert(kNumExperts == 2u * kThreads && kNumSlots <= kThreads,
                  "a calc thread owns one slot and two experts");
    static_assert(MoERoutingTables<Dims>::kMaxActiveExperts < 65536u &&
                      kNumSlots < 65536u,
                  "the row and expert counts share one 32-bit scan");

    auto& tables = smem->routing;
    auto& scratch = smem->routing_scratch;
    const std::uint32_t thread = threadIdx.x;
    const std::uint32_t lane = WarpRole::get_lane_idx();
    const std::uint32_t warp = WarpRole::get_calc_warp_idx();
    std::uint32_t* const expert_token_mask = scratch.expert_token_mask;
    uint2* const own_masks =
        reinterpret_cast<uint2*>(expert_token_mask) + thread;

    *own_masks = make_uint2(0u, 0u);
    WarpRole::sync_calc_threads();
    // Each expert's routed tokens; any thread may set a bit of any mask.
    const std::uint32_t token = thread / kMaxTopk;
    const bool routed = thread < kNumSlots && token < batch_size &&
                        thread - token * kMaxTopk < top_k;
    const std::uint32_t expert =
        routed ? scratch.topk_ids[thread] : kNumExperts;
    if (expert < kNumExperts) {
        atomicOr(&expert_token_mask[expert], 1u << token);
    }
    WarpRole::sync_calc_threads();

    // The thread's experts' row counts and which of them are active, then one
    // scan of both over the calc threads: rows in the high half, experts in
    // the low one.
    const uint2 masks = *own_masks;
    const std::uint32_t own =
        ((__popc(masks.x) + __popc(masks.y)) << 16) |
        ((masks.x != 0u ? 1u : 0u) + (masks.y != 0u ? 1u : 0u));
    std::uint32_t scan = own;
#pragma unroll
    for (std::uint32_t off = 1u; off < 32u; off *= 2u) {
        const std::uint32_t t = __shfl_up_sync(FULL_MASK, scan, off);
        if (lane >= off) scan += t;
    }
    if (lane == 31u) scratch.warp_counts[warp] = scan;
    WarpRole::sync_calc_threads();
    std::uint32_t before = 0u;
    std::uint32_t total = 0u;
#pragma unroll
    for (std::uint32_t w = 0; w < WarpRole::kCalcWarpCount; ++w) {
        const std::uint32_t counts = scratch.warp_counts[w];
        before += w < warp ? counts : 0u;
        total += counts;
    }
    std::uint32_t row = (before + scan - own) >> 16;
    std::uint32_t pos = (before + scan - own) & 0xFFFFu;

    // Active experts in ascending id.
    const std::uint32_t mask_of[2] = {masks.x, masks.y};
#pragma unroll
    for (std::uint32_t j = 0; j < 2u; ++j) {
        if (mask_of[j] == 0u) continue;
        const std::uint32_t e = 2u * thread + j;
        tables.active_experts[pos] = e;
        tables.active_token_mask[pos] = mask_of[j];
        tables.expert_token_cumsum[pos] = static_cast<std::uint16_t>(row);
        scratch.expert_position[e] = static_cast<std::uint8_t>(pos);
        row += __popc(mask_of[j]);
        ++pos;
    }
    if (thread == 0u) {
        // Trailing row pointer: the total row count.
        tables.expert_token_cumsum[total & 0xFFFFu] =
            static_cast<std::uint16_t>(total >> 16);
        tables.active_expert_count = total & 0xFFFFu;
    }
    WarpRole::sync_calc_threads();

    // The slot's row: its expert's slab, then the tokens below it.
    if (expert < kNumExperts) {
        const std::uint32_t r =
            tables.expert_token_cumsum[scratch.expert_position[expert]] +
            __popc(expert_token_mask[expert] & ((1u << token) - 1u));
        tables.expert_row_slot[r] = static_cast<std::uint8_t>(thread);
        tables.expert_row_weight[r] = scratch.topk_weights[thread];
    }
    WarpRole::sync_calc_threads();
}

}  // namespace halops::moe
