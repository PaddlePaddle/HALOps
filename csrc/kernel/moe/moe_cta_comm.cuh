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

#include <cstdint>

#include "kernel/moe/moe_configs.cuh"
#include "kernel/moe/ptx_utils.cuh"

namespace halops::moe {

// Global handoff state, double-buffered by launch parity to separate
// consecutive launches.
template <typename Dims>
struct PackedTokenCounts {
    using Word = std::uint64_t;
    static constexpr std::uint32_t kWordBits = 64u;
    static constexpr std::uint32_t kNumFields = Dims::kNumTokens;
    static_assert(kWordBits % kNumFields == 0u,
                  "the word must hold one whole field per token");
    static constexpr std::uint32_t kBits = kWordBits / kNumFields;
    static constexpr Word kMask = (Word(1) << kBits) - 1u;
    static_assert(Dims::kMaxTopk <= kMask,
                  "a token's count reaches top_k, which must fit its field");

    // Packed completion increment for tokens selected by token_mask.
    __device__ __forceinline__ static Word increment(std::uint32_t token_mask) {
        Word inc = 0u;
#pragma unroll
        for (std::uint32_t t = 0; t < kNumFields; ++t) {
            inc |= static_cast<Word>((token_mask >> t) & 1u) << (t * kBits);
        }
        return inc;
    }
};

template <std::uint32_t kSpecMaxGrid, std::uint32_t kMaxRoutedRows,
          std::uint32_t kUpOutScaleCols, std::uint32_t kCtasPerExpert>
struct MoECtaComm {
    // Per-block launch counter; its parity selects the double-buffer half.
    std::uint32_t launch_flip[kSpecMaxGrid];

    // Per-row readiness bits, double-buffered by launch parity. Each up block
    // release-sets its bit after storing data and scales.
    static constexpr std::uint32_t kUpOutReadyFull =
        (kUpOutScaleCols >= 32u) ? 0xFFFFFFFFu : ((1u << kUpOutScaleCols) - 1u);
    std::uint32_t up_out_ready[2][kMaxRoutedRows];

    // Parity-buffered down-task cursor and packed per-token completion counts.
    // Each token completes after top_k tasks.
    std::uint32_t down_task_next[2];
    std::uint64_t down_tasks_done[2][kCtasPerExpert];

    // Published router-task count, double-buffered by launch parity.
    std::uint32_t router_tasks_done[2];
    // Published latent-task count, double-buffered by launch parity.
    std::uint32_t latent_tasks_done[2];
    // Shared expert blocks that have published their activation slice.
    std::uint32_t shared_up_out_done[2];
    std::uint32_t shared_down_done[2];

    // Routing tables are published in 32 lane-strided word slices.
    static constexpr std::uint32_t kRouterTableSlices = 32u;
    std::uint32_t router_tables_ready[2][kRouterTableSlices];

    // Advance this CTA's launch counter and return its parity.
    __device__ __forceinline__ std::uint32_t claim_parity() {
        const std::uint32_t flip = launch_flip[blockIdx.x];
        launch_flip[blockIdx.x] = flip + 1u;
        return flip & 1u;
    }

    // Clear readiness state for the next launch in the inactive buffer.
    __device__ __forceinline__ void reset_inactive(std::uint32_t parity,
                                                   std::uint32_t rank) {
        const std::uint32_t inactive = parity ^ 1u;
        if (blockIdx.x != 0u) return;  // block 0 owns the counters
        if (rank < kMaxRoutedRows) {
            up_out_ready[inactive][rank] = 0u;
        }
        if (rank < kCtasPerExpert) {
            down_tasks_done[inactive][rank] = 0;
        }
        if (rank < kRouterTableSlices) {
            router_tables_ready[inactive][rank] = 0u;
        }
        if (rank == 0u) {
            down_task_next[inactive] = 0u;
            router_tasks_done[inactive] = 0u;
            latent_tasks_done[inactive] = 0u;
            shared_up_out_done[inactive] = 0u;
            shared_down_done[inactive] = 0u;
        }
    }

    // Publish a routing-table slice with release ordering; readers acquire it.
    __device__ __forceinline__ void publish_router_table_slice(
        std::uint32_t parity, std::uint32_t slice) {
        global_store_release_u32(&router_tables_ready[parity][slice], 1u);
    }
    __device__ __forceinline__ void wait_router_table_slice(
        std::uint32_t parity, std::uint32_t slice) const {
        while (global_load_acquire_u32(&router_tables_ready[parity][slice]) ==
               0u) {
        }
    }

    // Release-set one (row, up_block) bit of `up_out_ready`.
    __device__ __forceinline__ void publish_up_out(std::uint32_t parity,
                                                   std::uint32_t row,
                                                   std::uint32_t up_block,
                                                   unsigned lane) {
        __syncwarp();
        if (lane == 0u) {
            global_red_or_release_u32(&up_out_ready[parity][row],
                                      1u << up_block);
        }
    }

    // Acquire the requested up-block bits for every routed row of this expert.
    __device__ __forceinline__ void wait_up_out(std::uint32_t parity,
                                                std::uint32_t expert_start,
                                                std::uint32_t routed_count,
                                                std::uint32_t need_mask) const {
        for (std::uint32_t r = 0u; r < routed_count; ++r) {
            const std::uint32_t* word = &up_out_ready[parity][expert_start + r];
            const std::uint64_t t0 = clock64();
            while ((global_load_acquire_u32(word) & need_mask) != need_mask) {
            }
        }
    }

    // Publish router tasks after their logit writers have joined.
    __device__ __forceinline__ void publish_router_tasks(
        std::uint32_t parity, std::uint32_t num_tiles) {
        global_red_add_release_u32(&router_tasks_done[parity], num_tiles);
    }

    // `num_tasks` latent down-projection tasks published.
    __device__ __forceinline__ void publish_latent_tasks(
        std::uint32_t parity, std::uint32_t num_tasks) {
        global_red_add_release_u32(&latent_tasks_done[parity], num_tasks);
    }

    // One shared expert block published its activation slice.
    __device__ __forceinline__ void publish_shared_up_out(
        std::uint32_t parity, std::uint32_t num_tasks) {
        global_red_add_release_u32(&shared_up_out_done[parity], num_tasks);
    }

    __device__ __forceinline__ void wait_shared_up_out(
        std::uint32_t parity, std::uint32_t num_tasks) const {
        while (global_load_acquire_u32(&shared_up_out_done[parity]) <
               num_tasks) {
        }
    }

    __device__ __forceinline__ void publish_shared_down_done(
        std::uint32_t parity, std::uint32_t num_tasks) {
        global_red_add_release_u32(&shared_down_done[parity], num_tasks);
    }

    __device__ __forceinline__ void wait_shared_down_done(
        std::uint32_t parity, std::uint32_t num_tasks) const {
        while (global_load_acquire_u32(&shared_down_done[parity]) < num_tasks) {
        }
    }

    // Waits until all `num_tasks` latent tasks are published.
    __device__ __forceinline__ void wait_latent_tasks(
        std::uint32_t parity, std::uint32_t num_tasks) const {
        while (global_load_acquire_u32(&latent_tasks_done[parity]) <
               num_tasks) {
        }
    }

    // Waits until all `num_tiles` router tasks are published.
    __device__ __forceinline__ void wait_router_tasks(
        std::uint32_t parity, std::uint32_t num_tiles) const {
        while (global_load_acquire_u32(&router_tasks_done[parity]) <
               num_tiles) {
        }
    }

    // Atomically claim a down task. Values beyond the task count end the
    // schedule.
    __device__ __forceinline__ std::uint32_t claim_down_task(
        std::uint32_t parity) {
        return atomicAdd(&down_task_next[parity], 1u);
    }

    // Publish a completed task after its output writers have joined.
    __device__ __forceinline__ void publish_down_task(std::uint32_t parity,
                                                      std::uint32_t m_block,
                                                      std::uint64_t count) {
        __threadfence();
        global_red_add_u64(&down_tasks_done[parity][m_block], count);
    }

    // Wait until the packed count at shift reaches need.
    __device__ __forceinline__ void wait_down_out_field(
        std::uint32_t parity, std::uint32_t m_block, std::uint32_t shift,
        std::uint64_t mask, std::uint64_t need) const {
        while (((global_load_acquire_u64(&down_tasks_done[parity][m_block]) >>
                 shift) &
                mask) < need) {
        }
    }
};

}  // namespace halops::moe
