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

// This single struct is both the gmem storage and the publish/wait behaviour.
// Every handoff between the blocks of a launch is double-buffered by launch
// parity: a launch publishes into the parity-selected half while phase 1
// refills the other half for the next launch, so no consumer observes a
// leftover from the previous launch.
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

    // The increment one finished task adds to a packed word, for the tokens in
    // `token_mask`.
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

    // Per-row up→down readiness bitmask (one bit per up-block), double-buffered
    // by launch parity. Bit b is set (release) once up has written this row's
    // up-block b into `routed_up_out.data` + `routed_up_out.scale`. A row is
    // fully published when all kUpOutScaleCols bits are set.
    static constexpr std::uint32_t kUpOutReadyFull =
        (kUpOutScaleCols >= 32u) ? 0xFFFFFFFFu : ((1u << kUpOutScaleCols) - 1u);
    std::uint32_t up_out_ready[2][kMaxRoutedRows];

    // Dynamic down schedule, both double-buffered by launch parity: the next
    // unclaimed (expert, m-block) task, and per output m-block the number
    // of its tasks finished so far. An m-block is complete once that count
    // reaches the launch's active expert count (the expert dimension length).
    std::uint32_t down_task_next[2];
    std::uint64_t down_tasks_done[2][kCtasPerExpert];

    // Router tasks (one M tile each) whose logits are published,
    // double-buffered by launch parity; the logits are complete once it
    // reaches the task count.
    std::uint32_t router_tasks_done[2];
    // Latent down-projection tasks whose fp8 output is published,
    // double-buffered by launch parity; the output is complete once it reaches
    // the task count.
    std::uint32_t latent_tasks_done[2];
    // Shared expert blocks that have published their activation slice.
    std::uint32_t shared_up_out_done[2];

    // Set once block 0's routing tables are in `router_tables`.  The router's
    // own blocks build theirs; the others copy block 0's.
    std::uint32_t router_tables_done[2];

    // Claims this launch's parity for `blockIdx.x`, advancing that block's
    // private counter.
    __device__ __forceinline__ std::uint32_t claim_parity() {
        const std::uint32_t flip = launch_flip[blockIdx.x];
        launch_flip[blockIdx.x] = flip + 1u;
        return flip & 1u;
    }

    // Zero-refills the readiness state the NEXT launch will use (its parity is
    // the currently-inactive one).
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
        if (rank == 0u) {
            down_task_next[inactive] = 0u;
            router_tasks_done[inactive] = 0u;
            latent_tasks_done[inactive] = 0u;
            shared_up_out_done[inactive] = 0u;
            router_tables_done[inactive] = 0u;
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

    // Per-expert up→down readiness wait (acquire), the counterpart to
    // publish_up_out: spins until every routed row [expert_start,
    // +routed_count) of this expert has the requested up-block bits.
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

    // `num_tiles` router tasks published.  The caller has joined the
    // writers of their logits.
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

    // Waits until all `num_tasks` latent tasks are published.
    __device__ __forceinline__ void wait_latent_tasks(
        std::uint32_t parity, std::uint32_t num_tasks) const {
        while (global_load_acquire_u32(&latent_tasks_done[parity]) <
               num_tasks) {
        }
    }

    // Block 0's routing tables are published, and their reader's wait.
    __device__ __forceinline__ void publish_router_tables(
        std::uint32_t parity) {
        global_red_add_release_u32(&router_tables_done[parity], 1u);
    }
    __device__ __forceinline__ void wait_router_tables(
        std::uint32_t parity) const {
        // Polled at full rate: the latent blocks arrive after their projection,
        // and a backoff would only delay seeing the flag.
        while (global_load_acquire_u32(&router_tables_done[parity]) < 1u) {
        }
    }

    // Waits until all `num_tiles` router tasks are published.
    __device__ __forceinline__ void wait_router_tasks(
        std::uint32_t parity, std::uint32_t num_tiles) const {
        while (global_load_acquire_u32(&router_tasks_done[parity]) <
               num_tiles) {
        }
    }

    // Claims the next down task of the grid-wide dynamic schedule (relaxed:
    // it only distributes work).  Returns values past the task count once the
    // schedule is exhausted.
    __device__ __forceinline__ std::uint32_t claim_down_task(
        std::uint32_t parity) {
        return atomicAdd(&down_task_next[parity], 1u);
    }

    // One finished task of `m_block`, adding `count` to its completion word.
    // The caller has joined every thread's output stores; the fence makes
    // them device-visible before the count.
    __device__ __forceinline__ void publish_down_task(std::uint32_t parity,
                                                      std::uint32_t m_block,
                                                      std::uint64_t count) {
        __threadfence();
        global_red_add_u64(&down_tasks_done[parity][m_block], count);
    }

    // Waits until all `num_tasks` tasks of `m_block` are published.
    // Polled at full rate on purpose: the satisfying value is what gates this
    // block's writeback, so any backoff here is added straight to the tail.
    __device__ __forceinline__ void wait_down_out(
        std::uint32_t parity, std::uint32_t m_block,
        std::uint32_t num_tasks) const {
        while (global_load_acquire_u64(&down_tasks_done[parity][m_block]) <
               num_tasks) {
        }
    }

    // Per-token form: waits until the count packed at `shift` reaches `need`.
    // Polled at full rate, like `wait_down_out`.
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
