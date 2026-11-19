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
#include <cutlass/arch/barrier.h>

#include <cstddef>
#include <cstdint>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/tensor.hpp>

#include "kernel/moe/moe_configs.cuh"
#include "kernel/moe/moe_smem_plan.cuh"
#include "kernel/moe/moe_traits.cuh"
#include "kernel/moe/ptx_utils.cuh"

namespace halops::moe {

// 16-B aligned: the tables travel between blocks in whole words.
template <typename Dims>
struct alignas(16) MoERoutingTables {
    using ExpertId = std::uint32_t;

    static constexpr std::uint32_t kNumExperts = Dims::kNumExperts;
    static constexpr std::uint32_t kNumTokens = Dims::kNumTokens;
    static constexpr std::uint32_t kMaxTopk = Dims::kMaxTopk;
    static constexpr std::uint32_t kMaxRoutedPairs = kNumTokens * kMaxTopk;
    static constexpr std::uint32_t kMaxActiveExperts =
        (kNumExperts < kMaxRoutedPairs) ? kNumExperts : kMaxRoutedPairs;

    ExpertId active_experts[kMaxActiveExperts];
    std::uint32_t active_expert_count;

    std::uint16_t expert_token_cumsum[kMaxActiveExperts + 1];

    std::uint8_t expert_row_slot[kMaxRoutedPairs];

    float expert_row_weight[kMaxRoutedPairs];

    // Active expert i's routed tokens, one bit per token.
    std::uint32_t active_token_mask[kMaxActiveExperts];

    __device__ __forceinline__ std::uint32_t slab_begin(std::uint32_t i) const {
        return expert_token_cumsum[i];
    }
    __device__ __forceinline__ std::uint32_t slab_count(std::uint32_t i) const {
        return expert_token_cumsum[i + 1u] - expert_token_cumsum[i];
    }

    __device__ __forceinline__ std::uint32_t token_mask(std::uint32_t i) const {
        return active_token_mask[i];
    }

    // Rank of `token` in active expert `i`'s slab; 0xFF if not routed.
    __device__ __forceinline__ std::uint8_t rank_of(std::uint32_t i,
                                                    std::uint32_t token) const {
        const std::uint32_t mask = token_mask(i);
        if (((mask >> token) & 1u) == 0u) return 0xFFu;
        return static_cast<std::uint8_t>(__popc(mask & ((1u << token) - 1u)));
    }
};

// A ring of `N` shared-memory mbarriers; slot and phase follow the iteration.
template <std::uint32_t N>
struct BarrierRingBase {
    alignas(16) std::uint64_t data[N];

    __device__ __forceinline__ void init(int arrivals = 1) {
#pragma unroll
        for (std::uint32_t i = 0; i < N; ++i) {
            cute::initialize_barrier(data[i], arrivals);
        }
    }

    __device__ static constexpr std::uint32_t slot(std::uint32_t iter) {
        return iter % N;
    }
    __device__ static constexpr std::uint32_t parity(std::uint32_t iter) {
        return (iter / N) & 1u;
    }

    __device__ static void wait_parity(std::uint64_t& mbar,
                                       std::uint32_t phase) {
        cute::wait_barrier(mbar, static_cast<int>(phase));
        asm volatile("" ::: "memory");
    }

    __device__ __forceinline__ void wait(std::uint32_t iter) {
        wait_parity(data[slot(iter)], parity(iter));
    }

    __device__ __forceinline__ void wait_slot(std::uint32_t slot_idx,
                                              std::uint32_t phase) {
        wait_parity(data[slot_idx % N], phase);
    }

    // Whether `iter`'s phase has completed; never blocks.
    __device__ __forceinline__ bool test(std::uint32_t iter) const {
        return cutlass::arch::ClusterBarrier::test_wait(&data[slot(iter)],
                                                        parity(iter), 1u);
    }

    __device__ __forceinline__ std::uint64_t* raw(std::uint32_t slot_idx) {
        return &data[slot_idx % N];
    }
};

// A ring whose producer signals without a byte count.
template <std::uint32_t N>
struct SimpleBarrier : BarrierRingBase<N> {
    using Base = BarrierRingBase<N>;
    using Base::slot;

    __device__ __forceinline__ void arrive(std::uint32_t iter) {
        cute::arrive_barrier(Base::data[slot(iter)]);
    }
};

// A TMA-arrival ring: the producer arms a slot with its exact byte count.
template <std::uint32_t N>
struct TransactionBarrier : BarrierRingBase<N> {
    using Base = BarrierRingBase<N>;
    using Base::slot;

    __device__ __forceinline__ void arrive(std::uint32_t iter,
                                           std::uint32_t bytes) {
        cute::set_barrier_transaction_bytes(Base::data[slot(iter)], bytes);
    }
};

template <std::uint32_t NumSlots, std::uint32_t NumAccSlots>
struct GemmPipelineBarrier {
    TransactionBarrier<NumSlots> input_ready;
    SimpleBarrier<NumSlots> input_empty;
    TransactionBarrier<NumSlots> weight_ready;
    SimpleBarrier<NumSlots> weight_empty;
    SimpleBarrier<NumAccSlots> accumulator_ready;
    SimpleBarrier<NumAccSlots> accumulator_empty;

    __device__ __forceinline__ void init(int accumulator_empty_arrivals = 1) {
        input_ready.init(1);
        input_empty.init(1);
        weight_ready.init(1);
        weight_empty.init(1);
        accumulator_ready.init(1);
        accumulator_empty.init(accumulator_empty_arrivals);
    }

    // A local quantizer publishes its MMA operand through input_empty.
    template <bool WaitInputEmpty = false, bool WaitAccumulatorEmpty = false>
    __device__ __forceinline__ void wait_mma_ready(std::uint32_t iter) {
        if constexpr (WaitAccumulatorEmpty) {
            wait_accumulator_empty(iter);
        }
        if constexpr (WaitInputEmpty) {
            input_empty.wait(iter);
        } else {
            input_ready.wait(iter);
        }
        weight_ready.wait(iter);
    }

    __device__ __forceinline__ void wait_input_empty(std::uint32_t iter) {
        if (iter >= NumSlots) {
            input_empty.wait(iter - NumSlots);
        }
    }

    __device__ __forceinline__ void wait_weight_empty(std::uint32_t iter) {
        if (iter >= NumSlots) {
            weight_empty.wait(iter - NumSlots);
        }
    }

    __device__ __forceinline__ void wait_accumulator_empty(std::uint32_t iter) {
        if (iter >= NumAccSlots) {
            accumulator_empty.wait(iter - NumAccSlots);
        }
    }
};

template <typename Dims, bool UseBlockscaled =
                             RoutedUpGemmTraits<Dims>::MmaCore::kUseBlockscaled>
struct RoutedBarrier;

template <typename Dims>
struct RoutedBarrier<Dims, false> {
    static constexpr std::uint32_t kNumUpSlots =
        GemmSlots<Dims, RoutedUpGemmTraits<Dims>>::kNumSlots;
    static constexpr std::uint32_t kNumDownSlots =
        GemmSlots<Dims, RoutedDownGemmTraits<Dims>>::kNumSlots;
    // Up and down share the accumulator ring; up's depth is the larger one.
    static constexpr std::uint32_t kTmemAccSlots =
        GemmSlots<Dims, RoutedUpGemmTraits<Dims>>::kTmemAccSlots;

    // Weight and input tiles share a slot: warp 9 and warp 10 each arrive.
    TransactionBarrier<kNumUpSlots> up_weight_ready;
    TransactionBarrier<kNumDownSlots> down_weight_ready;
    TransactionBarrier<kNumDownSlots> down_input_ready;

    SimpleBarrier<kNumUpSlots> input_scale_empty;

    // Each K tile's completion also releases its input and weight slots.
    SimpleBarrier<kTmemAccSlots> accumulator_ready;
    SimpleBarrier<kTmemAccSlots> accumulator_empty;

    __device__ __forceinline__ void init() {
        const int num_warps = static_cast<int>(WarpRole::kCalcWarpCount);
        up_weight_ready.init(2);
        down_weight_ready.init(1);
        down_input_ready.init(1);
        input_scale_empty.init(num_warps);
        accumulator_ready.init(1);
        accumulator_empty.init(num_warps);
    }

    __device__ __forceinline__ void wait_up_mma_ready(std::uint32_t iter) {
        if (iter >= kTmemAccSlots) {
            accumulator_empty.wait(iter - kTmemAccSlots);
        }
        up_weight_ready.wait(iter);
    }

    __device__ __forceinline__ void wait_down_mma_ready(
        std::uint32_t local_iter, std::uint32_t acc_iter) {
        if (acc_iter >= kTmemAccSlots) {
            accumulator_empty.wait(acc_iter - kTmemAccSlots);
        }
        down_weight_ready.wait(local_iter);
        down_input_ready.wait(local_iter);
    }
};

template <typename Dims>
struct RoutedBarrier<Dims, true> {
    static constexpr std::uint32_t kNumUpSlots =
        GemmSlots<Dims, RoutedUpGemmTraits<Dims>>::kNumSlots;
    static constexpr std::uint32_t kNumDownSlots =
        GemmSlots<Dims, RoutedDownGemmTraits<Dims>>::kNumSlots;
    // Up rotates its tasks over these slots; down continues the same ring.
    static constexpr std::uint32_t kTmemAccSlots =
        GemmSlots<Dims, RoutedUpGemmTraits<Dims>>::kTmemAccSlots;

    // Weight and input tiles share a slot: warp 9 and warp 10 each arrive.
    TransactionBarrier<kNumUpSlots> up_weight_ready;
    SimpleBarrier<kNumUpSlots> up_weight_empty;

    TransactionBarrier<kNumDownSlots> down_input_ready;
    TransactionBarrier<kNumDownSlots> down_weight_ready;
    SimpleBarrier<kNumDownSlots> down_weight_empty;

    SimpleBarrier<kTmemAccSlots> accumulator_ready;
    SimpleBarrier<kTmemAccSlots> accumulator_empty;

    __device__ __forceinline__ void init() {
        const int num_warps = static_cast<int>(WarpRole::kCalcWarpCount);
        up_weight_ready.init(2);
        up_weight_empty.init(1);
        down_weight_ready.init(1);
        down_input_ready.init(1);
        down_weight_empty.init(1);
        accumulator_ready.init(1);
        accumulator_empty.init(num_warps);
    }

    __device__ __forceinline__ void wait_up_mma_ready(std::uint32_t iter) {
        up_weight_ready.wait(iter);
    }

    __device__ __forceinline__ void wait_down_mma_ready(
        std::uint32_t local_iter) {
        down_weight_ready.wait(local_iter);
        down_input_ready.wait(local_iter);
    }
};

// Shared-memory mbarriers and the init/wait protocol that uses them.
template <typename Dims>
struct MoEBarriers {
    using RouterPipeline =
        GemmPipelineBarrier<GemmSlots<Dims, RouterGemmTraits<Dims>>::kNumSlots,
                            1u>;
    using LatentDownPipeline = GemmPipelineBarrier<
        GemmSlots<Dims, LatentDownGemmTraits<Dims>>::kNumSlots,
        GemmSlots<Dims, LatentDownGemmTraits<Dims>>::kTmemAccSlots>;

    RouterPipeline router;
    LatentDownPipeline latent_down;
    RoutedBarrier<Dims> routed;

    // SharedExpert
    static constexpr std::uint32_t kNumSharedUpSlots =
        GemmSlots<Dims, SharedUpGemmTraits<Dims>>::kNumSlots;
    static constexpr std::uint32_t kNumSharedDownSlots =
        GemmSlots<Dims, SharedDownGemmTraits<Dims>>::kNumSlots;
    using SharedUpPipeline = GemmPipelineBarrier<
        kNumSharedUpSlots,
        GemmSlots<Dims, SharedUpGemmTraits<Dims>>::kTmemAccSlots>;
    SharedUpPipeline shared_up;

    using SharedDownPipeline = GemmPipelineBarrier<kNumSharedDownSlots, 1u>;
    SharedDownPipeline shared_down;

    // The top-k's cluster rendezvous: both blocks of the pair arrive.
    struct RouterCoopBarrier {
        alignas(8) std::uint64_t mbar;

        __device__ __forceinline__ void init() {
            cute::initialize_barrier(mbar, 2);
        }

        __device__ __forceinline__ void arrive() { cute::arrive_barrier(mbar); }

        __device__ __forceinline__ std::uint32_t partner(std::uint32_t rank) {
            return cluster_map_shared_rank(&mbar, rank);
        }

        __device__ __forceinline__ void wait() {
            cute::wait_barrier(mbar, 0);
            asm volatile("" ::: "memory");
        }

        __device__ __forceinline__ static void arrive_remote(
            std::uint32_t mapped_addr) {
            asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];\n" ::"r"(
                             mapped_addr)
                         : "memory");
        }
    } router_coop;

    __device__ __forceinline__ void init_router_ring() {
        router.init();
        router_coop.init();
    }

    __device__ __forceinline__ void init_shared_ring() {
        shared_up.init();
        shared_down.init();
    }
};

// The routing phase's own scratch, dead once the tables are built.
template <typename Dims>
struct RoutingSmem {
    alignas(16)
        std::uint16_t topk_ids[(Dims::kNumTokens < 8u ? 8u : Dims::kNumTokens) *
                               Dims::kMaxTopk];
    alignas(
        16) float topk_weights[(Dims::kNumTokens < 8u ? 8u : Dims::kNumTokens) *
                               Dims::kMaxTopk];

    std::uint8_t expert_position[Dims::kNumExperts];

    // Token masks for all experts; routing tables retain active experts only.
    alignas(16) std::uint32_t expert_token_mask[Dims::kNumExperts];

    // Each calc warp's row and active-expert counts, for the block scan.
    std::uint32_t warp_counts[WarpRole::kCalcWarpCount];
};

template <typename Dims>
struct MoESharedStorage {
    using Traits = MoEKernelTraits<Dims>;

    static_assert(
        Dims::kNumExperts <= 65535,
        "Number of experts too high, cannot store as uint16 anymore.");

    union {
        RouterSmem<Dims, typename Traits::RouterGemm> router;
        LatentDownSmem<Dims, typename Traits::LatentDownGemm> latent_down;
        RoutingSmem<Dims> routing_scratch;
        SharedExpertSmem<Dims, typename Traits::SharedUpGemm,
                         typename Traits::SharedDownGemm>
            shared_expert;
        RoutedExpertsSmem<Dims, typename Traits::RoutedUpGemm,
                          typename Traits::RoutedDownGemm>
            routed_experts;
    };

    uint32_t tmem_ptr;

    MoERoutingTables<Dims> routing;

    MoEBarriers<Dims> barriers;

    uint32_t launch_parity;
    uint32_t tail_parity;

    static constexpr uint32_t kMaxDownTasks =
        MoERoutingTables<Dims>::kMaxActiveExperts *
        MoERuntime::kMaxCtasPerExpert;
    static constexpr uint32_t kNoDownTask = 0xFFFFu;
    static_assert(kMaxDownTasks < kNoDownTask,
                  "down grid task ids must fit in uint16 below the sentinel");
    uint16_t down_grid_task[kMaxDownTasks + 1];
    uint32_t down_tasks_claimed;
    uint32_t down_task_stores;
};

// Dynamic shared memory required by fused_moe_kernel<Dims>.
template <typename Dims>
__device__ __host__ constexpr size_t get_moe_smem_size() {
    using RoutedSmem = decltype(MoESharedStorage<Dims>::routed_experts);
    // The two weight rings are one union view, so they must start together.
    static_assert(offsetof(RoutedSmem, w_up) == offsetof(RoutedSmem, w_down),
                  "the two weight rings must share one region.");
    return sizeof(MoESharedStorage<Dims>);
}

}  // namespace halops::moe
