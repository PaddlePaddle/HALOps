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

// Builds the routing tables from topk_ids; one warp runs the whole thing.
// Token t's picks sit at slots t * kMaxTopk + k, which are also their combine
// slots, so the passes walk the slots directly.  They are rolled loops: the
// build runs once per launch, so a launch whose instruction caches are cold
// fetches every one of its instructions, and on the routing's critical path.
template <typename Dims>
__device__ void prepare_routing_tables(
    std::uint32_t batch_size, std::uint32_t top_k,
    MoESharedStorage<Dims>* __restrict__ smem) {
    constexpr std::uint32_t kNumExperts = Dims::kNumExperts;
    constexpr std::uint32_t kMaxTopk = Dims::kMaxTopk;
    constexpr std::uint32_t kNumSlots = Dims::kNumTokens * kMaxTopk;
    constexpr std::uint32_t kExpertsPerLane = kNumExperts / 32u;
    static_assert(kNumExperts % 128u == 0u && kExpertsPerLane <= 32u,
                  "a lane's experts are whole 16-B words of masks and fit one "
                  "32-bit set");
    static_assert(MoERoutingTables<Dims>::kMaxActiveExperts < 65536u &&
                      kNumSlots < 65536u,
                  "the row and expert counts share one 32-bit scan");

    auto& tables = smem->routing;
    auto& scratch = smem->routing_scratch;
    const std::uint32_t lane = WarpRole::get_lane_idx();
    std::uint32_t* const expert_token_mask = scratch.expert_token_mask;
    uint4* const lane_masks =
        reinterpret_cast<uint4*>(&expert_token_mask[lane * kExpertsPerLane]);

    // Each expert's routed tokens; any lane may set a bit of any mask.
#pragma unroll
    for (std::uint32_t q = 0; q < kExpertsPerLane / 4u; ++q) {
        lane_masks[q] = make_uint4(0u, 0u, 0u, 0u);
    }
    __syncwarp();
#pragma unroll 1
    for (std::uint32_t s = lane; s < kNumSlots; s += 32u) {
        const std::uint32_t token = s / kMaxTopk;
        if (token >= batch_size || s - token * kMaxTopk >= top_k) continue;
        const std::uint32_t expert = scratch.topk_ids[s];
        if (expert < kNumExperts)
            atomicOr(&expert_token_mask[expert], 1u << token);
    }
    __syncwarp();

    // The lane's experts' row counts and which of them are active, then one
    // warp scan of both: rows in the high half, experts in the low one.
    std::uint32_t rows = 0u;
    std::uint32_t active = 0u;  // bit i: expert lane * kExpertsPerLane + i
#pragma unroll
    for (std::uint32_t q = 0; q < kExpertsPerLane / 4u; ++q) {
        const uint4 v = lane_masks[q];
        const std::uint32_t m[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
        for (std::uint32_t j = 0; j < 4u; ++j) {
            rows += __popc(m[j]);
            active |= (m[j] != 0u ? 1u : 0u) << (4u * q + j);
        }
    }
    const std::uint32_t own = (rows << 16) | __popc(active);
    std::uint32_t scan = own;
#pragma unroll
    for (std::uint32_t off = 1u; off < 32u; off *= 2u) {
        const std::uint32_t t = __shfl_up_sync(FULL_MASK, scan, off);
        if (lane >= off) scan += t;
    }
    const std::uint32_t total = __shfl_sync(FULL_MASK, scan, 31);
    std::uint32_t row = (scan - own) >> 16;
    std::uint32_t pos = (scan - own) & 0xFFFFu;

    // Active experts in ascending id: the lanes ascend, and so do a lane's
    // bits.
#pragma unroll 1
    for (std::uint32_t bits = active; bits != 0u; bits &= bits - 1u) {
        const std::uint32_t expert =
            lane * kExpertsPerLane +
            static_cast<std::uint32_t>(__ffs(bits) - 1);
        const std::uint32_t m = expert_token_mask[expert];
        tables.active_experts[pos] = expert;
        tables.active_token_mask[pos] = m;
        tables.expert_token_cumsum[pos] = static_cast<std::uint16_t>(row);
        scratch.expert_position[expert] = static_cast<std::uint8_t>(pos);
        row += __popc(m);
        ++pos;
    }
    if (lane == 0u) {
        // Trailing row pointer: the total row count.
        tables.expert_token_cumsum[total & 0xFFFFu] =
            static_cast<std::uint16_t>(total >> 16);
        tables.active_expert_count = total & 0xFFFFu;
    }
    __syncwarp();

    // Each routed pair's row: its expert's slab, then the tokens below it.
#pragma unroll 1
    for (std::uint32_t s = lane; s < kNumSlots; s += 32u) {
        const std::uint32_t token = s / kMaxTopk;
        if (token >= batch_size || s - token * kMaxTopk >= top_k) continue;
        const std::uint32_t expert = scratch.topk_ids[s];
        if (expert >= kNumExperts) continue;
        const std::uint32_t r =
            tables.expert_token_cumsum[scratch.expert_position[expert]] +
            __popc(expert_token_mask[expert] & ((1u << token) - 1u));
        tables.expert_row_slot[r] = static_cast<std::uint8_t>(s);
        tables.expert_row_weight[r] = scratch.topk_weights[s];
    }
    __syncwarp();
}

}  // namespace halops::moe
