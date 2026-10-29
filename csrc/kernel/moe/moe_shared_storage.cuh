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

#include "moe_configs.cuh"
#include "moe_interface.h"
#include "moe_traits.cuh"
#include "ptx_utils.cuh"

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

    DataType::ElementScale expert_row_weight[kMaxRoutedPairs];

    std::uint32_t expert_token_mask[kNumExperts];

    __device__ __forceinline__ std::uint32_t slab_begin(std::uint32_t i) const {
        return expert_token_cumsum[i];
    }
    __device__ __forceinline__ std::uint32_t slab_count(std::uint32_t i) const {
        return expert_token_cumsum[i + 1u] - expert_token_cumsum[i];
    }

    __device__ __forceinline__ std::uint32_t token_mask(std::uint32_t i) const {
        return expert_token_mask[active_experts[i]];
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

// Shared-memory mbarriers and the init/wait protocol that uses them.
template <typename Dims>
struct MoEBarriers {
    static constexpr std::uint32_t kNumUpWeightSlots =
        NumUpWeightSlots<Dims>::value;
    static constexpr std::uint32_t kNumDownWeightSlots =
        NumDownWeightSlots<Dims>::value;
    static constexpr std::uint32_t kNumScaleSlots = 2;
    static constexpr std::uint32_t kNumActivationSlots = kNumDownWeightSlots;
    static constexpr std::uint32_t kNumAccumulatorSlots =
        NumAccSlots<Dims>::value;
    static constexpr std::uint32_t kNumRouterWeightSlots =
        RouterGemmTraits<Dims>::kNumWeightSlots;

    // LatentDown
    static constexpr std::uint32_t kNumInputSlots = NumInputSlots<Dims>::value;
    static constexpr std::uint32_t kNumLatentWeightSlots =
        LatentDownGemmTraits<Dims>::kNumWeightSlots;
    TransactionBarrier<MoEKernelTraits<Dims>::kLatentInputSlots>
        latent_down_input_ready;
    TransactionBarrier<kNumLatentWeightSlots> latent_down_weight_ready;
    SimpleBarrier<kNumLatentWeightSlots> latent_down_weight_empty;

    // The quantize's handshake: the issuer can lag the quantize by a whole K.
    static constexpr std::uint32_t kNumLatentInputEmptySlots =
        LatentDownGemmTraits<Dims>::kUseUmma
            ? LatentDownGemmTraits<Dims>::kNumTilesK
            : MoEKernelTraits<Dims>::kLatentInputSlots;
    SimpleBarrier<kNumLatentInputEmptySlots> latent_down_input_empty;

    // Router
    TransactionBarrier<MoEKernelTraits<Dims>::kRouterInputSlots>
        router_input_ready;
    SimpleBarrier<MoEKernelTraits<Dims>::kRouterInputSlots> router_input_empty;
    TransactionBarrier<kNumRouterWeightSlots> router_weight_ready;
    SimpleBarrier<kNumRouterWeightSlots> router_weight_empty;

    SimpleBarrier<1> router_accumulator_ready;

    // RoutedUp
    TransactionBarrier<kNumUpWeightSlots> up_weight_ready;

    SimpleBarrier<1> up_input_ready;
    // The up phase's fp8 staging, one slot per K tile.
    static constexpr std::uint32_t kNumUpStagingSlots =
        MoEKernelTraits<Dims>::UpGemm::kNumTilesK;
    SimpleBarrier<kNumUpStagingSlots> up_staging_ready;
    static constexpr std::uint32_t kNumUpInputSlots =
        MoEKernelTraits<Dims>::kUpInputSlots;
    // One bulk copy of the whole projection output, when the ring holds it.
    TransactionBarrier<1> latent_copy_ready;
    TransactionBarrier<kNumUpInputSlots> up_input_slot_ready;
    SimpleBarrier<kNumUpInputSlots> up_input_slot_empty;

    SimpleBarrier<kNumScaleSlots> weight_scale_ready;
    SimpleBarrier<kNumScaleSlots> weight_scale_empty;

    // RoutedDown
    TransactionBarrier<kNumDownWeightSlots> down_weight_ready;
    TransactionBarrier<kNumActivationSlots> activation_ready;

    // MMA-completion barriers for the TMEM accumulator ring.
    SimpleBarrier<kNumAccumulatorSlots> accumulator_ready;
    SimpleBarrier<kNumAccumulatorSlots> accumulator_empty;

    // SharedExpert
    static constexpr std::uint32_t kNumSharedUpWeightSlots =
        SharedUpGemmTraits<Dims>::kNumWeightSlots;
    static constexpr std::uint32_t kNumSharedDownWeightSlots =
        SharedDownGemmTraits<Dims>::kNumWeightSlots;
    TransactionBarrier<kNumSharedUpWeightSlots> shared_up_weight_ready;
    SimpleBarrier<kNumSharedUpWeightSlots> shared_up_weight_empty;
    SimpleBarrier<kNumSharedUpWeightSlots> shared_up_quantize_ready;
    TransactionBarrier<SharedUpGemmTraits<Dims>::kNumWeightSlots>
        shared_up_input_ready;
    SimpleBarrier<SharedUpGemmTraits<Dims>::kNumWeightSlots>
        shared_up_input_empty;
    TransactionBarrier<kNumSharedDownWeightSlots> shared_down_weight_ready;
    SimpleBarrier<kNumSharedDownWeightSlots> shared_down_weight_empty;
    TransactionBarrier<kNumSharedDownWeightSlots> shared_down_input_ready;
    static constexpr std::uint32_t kNumSharedUpAccSlots =
        MoEKernelTraits<Dims>::kNumAccSlots;
    SimpleBarrier<kNumSharedUpAccSlots> shared_up_accumulator_ready;
    SimpleBarrier<kNumSharedUpAccSlots> shared_up_accumulator_empty;
    SimpleBarrier<1> shared_down_accumulator_ready;

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
        router_input_ready.init(1);
        router_input_empty.init(1);
        router_weight_ready.init(1);
        // A slot frees when its previous occupant's UMMA completes.
        router_weight_empty.init(1);
        router_accumulator_ready.init(1);
        router_coop.init();
    }

    __device__ __forceinline__ void init_latent_ring() {
        latent_down_input_ready.init(1);
        latent_down_input_empty.init(1);
        latent_down_weight_ready.init(1);
        latent_down_weight_empty.init(1);
    }

    __device__ __forceinline__ void init_expert_ring() {
        const int num_warps = static_cast<int>(WarpRole::kCalcWarpCount);
        up_input_ready.init(num_warps);
        up_staging_ready.init(1);
        latent_copy_ready.init(1);
        up_input_slot_ready.init(1);
        up_input_slot_empty.init(num_warps);
        up_weight_ready.init(1);
        down_weight_ready.init(1);
        activation_ready.init(1);
        weight_scale_ready.init(1);
        weight_scale_empty.init(num_warps);
        accumulator_ready.init(1);
        accumulator_empty.init(num_warps);
    }

    __device__ __forceinline__ void init_shared_ring() {
        shared_up_weight_ready.init(1);
        shared_up_weight_empty.init(1);
        shared_up_quantize_ready.init(1);
        shared_up_input_ready.init(1);
        shared_up_input_empty.init(1);
        shared_down_weight_ready.init(1);
        shared_down_weight_empty.init(1);
        shared_down_input_ready.init(1);
        shared_up_accumulator_ready.init(1);
        shared_up_accumulator_empty.init(1);
        shared_down_accumulator_ready.init(1);
    }

    __device__ void wait_accumulator_empty(std::uint32_t ring_iter_id) {
        accumulator_empty.wait(ring_iter_id - kNumAccumulatorSlots);
    }

    __device__ __forceinline__ void wait_weight_scale_ready(
        std::uint32_t task_id, std::uint32_t slot) {
        weight_scale_ready.wait_slot(slot, (task_id / kNumScaleSlots) & 1u);
    }

    __device__ __forceinline__ void wait_weight_scale_empty(
        std::uint32_t task_id, std::uint32_t slot) {
        weight_scale_empty.wait_slot(slot, (task_id / kNumScaleSlots) & 1u);
    }

    __device__ void wait_router_input_empty(std::uint32_t tile) {
        if (tile < MoEKernelTraits<Dims>::kRouterInputSlots) return;
        const std::uint32_t prev =
            tile - MoEKernelTraits<Dims>::kRouterInputSlots;
        router_input_empty.wait(prev);
    }

    __device__ __forceinline__ void arrive_router_input_empty(
        std::uint32_t tile) {
        router_input_empty.arrive(tile);
    }

    __device__ void wait_router_weight_empty(std::uint32_t k_tile) {
        if (k_tile < kNumRouterWeightSlots) {
            return;
        }
        // The (prev / slots + 1)-th arrival on the slot releases `prev`.
        const std::uint32_t prev = k_tile - kNumRouterWeightSlots;
        router_weight_empty.wait(prev);
    }

    __device__ __forceinline__ void wait_latent_down_weight_empty(
        std::uint32_t k_tile) {
        if (k_tile < kNumLatentWeightSlots) return;
        const std::uint32_t prev = k_tile - kNumLatentWeightSlots;
        latent_down_weight_empty.wait(prev);
    }
};

// The routing phase's own scratch, dead once the tables are built.
template <typename Dims>
struct RoutingSmem {
    alignas(16)
        std::uint16_t topk_ids[(Dims::kNumTokens < 8u ? 8u : Dims::kNumTokens) *
                               Dims::kMaxTopk];
    alignas(16) DataType::ElementScale
        topk_weights[(Dims::kNumTokens < 8u ? 8u : Dims::kNumTokens) *
                     Dims::kMaxTopk];

    std::uint8_t expert_position[Dims::kNumExperts];
};

// A ring of streamed bf16 activation tiles: B operand and quantize source.
template <std::uint32_t NumSlots, std::uint32_t TileShapeN,
          std::uint32_t TileShapeK>
struct ActivationSmem {
    using Element = DataType::ElementA;

    static constexpr std::uint32_t kNumSlots = NumSlots;
    static constexpr std::uint32_t kSwizzleBoxK = SwizzleBoxK<Element>::value;
    static constexpr std::uint32_t kNumSwizzleBoxesK =
        TileShapeK / kSwizzleBoxK;

    static_assert(TileShapeK % kSwizzleBoxK == 0u,
                  "a tile must be whole swizzle boxes");

    alignas(1024)
        Element data[kNumSlots][kNumSwizzleBoxesK][TileShapeN][kSwizzleBoxK];
};

// The quantize's fp8 output, which is also the MMA's B operand.
template <std::uint32_t NumSlots, std::uint32_t TileShapeN,
          std::uint32_t TileShapeK, std::uint32_t GemmN, std::uint32_t GemmK,
          std::uint32_t QuantBlockK>
struct QuantizedActivationSmem {
    using Element = DataType::ElementQA;
    using ElementScale = DataType::ElementScale;

    static constexpr std::uint32_t kNumSlots = NumSlots;
    static constexpr std::uint32_t kLboChunkK = kFp8ActChunkK;
    static constexpr std::uint32_t kNumLboChunksK = TileShapeK / kLboChunkK;
    static constexpr std::uint32_t kNumQuantBlocks = GemmK / QuantBlockK;

    static_assert(TileShapeK % kLboChunkK == 0u,
                  "a tile must be whole 16-B chunks");

    alignas(1024)
        Element data[kNumSlots][kNumLboChunksK][TileShapeN + 1u][kLboChunkK];
    ElementScale scale[kNumQuantBlocks][GemmN];
};

template <typename Dims>
struct RouterSmem {
    using Traits = MoEKernelTraits<Dims>;
    using RouterGemm = typename Traits::RouterGemm;

    ActivationSmem<Traits::kRouterInputSlots, RouterGemm::kTileShapeN,
                   RouterGemm::kTileShapeK>
        input;

    alignas(1024) typename Traits::RouterGemm::MmaElement
        w_router[RouterGemm::kNumWeightSlots]
                [RouterGemm::MmaCore::kWeightTileElems];

    static constexpr std::uint32_t kNumMmaWarps =
        Select<RouterGemm::kUseUmma, 1u, WarpRole::kCalcWarpCount>::value;
    static constexpr std::uint32_t kMmaRows =
        Select<RouterGemm::kUseUmma, 1u, RouterGemm::kTileShapeN>::value;
    static constexpr std::uint32_t kMmaCols =
        Select<RouterGemm::kUseUmma, 1u, RouterGemm::kTileShapeM>::value;
    float mma_out[kNumMmaWarps][kMmaRows][kMmaCols];
};

template <typename Dims>
struct LatentDownSmem {
    using Traits = MoEKernelTraits<Dims>;
    using LatentDownGemm = typename Traits::LatentDownGemm;

    alignas(1024) DataType::ElementW
        w_latent_down[LatentDownGemm::kNumWeightSlots]
                     [LatentDownGemm::kQuantBlocksPerTile]
                     [LatentDownGemm::kTileShapeM][128u];

    ActivationSmem<Traits::kLatentInputSlots, LatentDownGemm::kTileShapeN,
                   LatentDownGemm::kTileShapeK>
        input;

    QuantizedActivationSmem<Traits::kLatentInputSlots,
                            LatentDownGemm::kTileShapeN,
                            LatentDownGemm::kTileShapeK, LatentDownGemm::kGemmN,
                            LatentDownGemm::kGemmK, Dims::kQuantBlockK>
        quantized_input;

    static constexpr std::uint32_t kNumMmaWarps =
        Select<LatentDownGemm::kUseUmma, 1u, WarpRole::kCalcWarpCount>::value;
    static constexpr std::uint32_t kMmaRows =
        Select<LatentDownGemm::kUseUmma, 1u,
               LatentDownGemm::kTileShapeM>::value;
    static constexpr std::uint32_t kMmaCols =
        Select<LatentDownGemm::kUseUmma, 1u,
               LatentDownGemm::kTileShapeN>::value;
    alignas(16) float mma_out[kNumMmaWarps][kMmaRows][kMmaCols];

    DataType::ElementScale
        w_latent_down_scale[LatentDownGemm::kNumScaleBlocksK];

    static constexpr std::uint32_t kNumOutChunks =
        Select<Traits::kLatentQuantizeOutput,
               LatentDownGemm::kTileShapeM / kFp8ActChunkK, 1u>::value;
    static constexpr std::uint32_t kNumOutRows =
        Select<Traits::kLatentQuantizeOutput, LatentDownGemm::kTileShapeN,
               1u>::value;
    alignas(16)
        DataType::ElementA out_bf16[kNumOutChunks][kNumOutRows][kFp8ActChunkK];
    alignas(16) float token_amax[kNumOutRows];

    alignas(16) float splited_mma_out[LatentDownGemm::kTileShapeM]
                                     [LatentDownGemm::kTileShapeN];
};

// The shared expert's own storage, in the union with the other phases.
template <typename Dims>
struct SharedExpertSmem {
    using SharedUpGemm = SharedUpGemmTraits<Dims>;
    using SharedDownGemm = SharedDownGemmTraits<Dims>;

    union {
        alignas(1024) DataType::ElementW w_up[SharedUpGemm::kNumWeightSlots][2u]
                                             [SharedUpGemm::kTileShapeM][128u];
        alignas(1024) DataType::ElementW
            w_down[SharedDownGemm::kNumWeightSlots][SharedDownGemm::kTileShapeM]
                  [SharedDownGemm::kTileShapeK];
        alignas(1024) float splited_mma_out[2u][SharedUpGemm::kActQuantBlock]
                                           [Dims::kNumTokens];
    };

    ActivationSmem<SharedUpGemm::kNumWeightSlots, Dims::kNumTokens,
                   SharedUpGemm::kTileShapeK>
        input;

    alignas(1024) QuantizedActivationSmem<
        SharedUpGemm::kNumWeightSlots, SharedUpGemm::kTileShapeN,
        SharedUpGemm::kTileShapeK, SharedUpGemm::kGemmN, SharedUpGemm::kGemmK,
        Dims::kQuantBlockK> quantized_input;

    alignas(1024) QuantizedActivationSmem<
        SharedDownGemm::kNumWeightSlots, Dims::kNumTokens,
        SharedDownGemm::kTileShapeK, Dims::kNumTokens, Dims::kIntermediateSize,
        SharedUpGemm::kActQuantBlock> act_down_input;

    alignas(16) float act_up_out[SharedUpGemm::kActQuantBlock]
                                [Dims::kNumTokens + 1u];
};

template <typename Dims>
struct RoutedExpertsSmem {
    using Traits = MoEKernelTraits<Dims>;
    using UpGemm = typename Traits::UpGemm;
    using DownGemm = typename Traits::DownGemm;

    union {
        alignas(1024) DataType::ElementW
            w_up[UpGemm::kNumWeightSlots][UpGemm::MmaCore::kNumSwizzleBoxKs]
                [UpGemm::kTileShapeM * UpGemm::MmaCore::kSwizzleBoxK];

        alignas(1024) DataType::ElementW
            w_down[DownGemm::kNumWeightSlots][DownGemm::kTileShapeM]
                  [DownGemm::kTileShapeK];
    };

    struct UpDownActivation {
        DataType::ElementAcc act_up_out[Traits::UpGemm::kActQuantBlock]
                                       [UpGemm::kTileShapeN + 1];

        static constexpr uint32_t kNumDownActElems =
            DownGemm::kTileShapeN * DownGemm::kTileShapeK;
        alignas(1024) DataType::ElementQA
            act_down_input[DownGemm::kNumWeightSlots][kNumDownActElems];
    };

    static constexpr std::uint32_t kUpInputSlots =
        Select<Traits::kLatentQuantizeOutput, 1u, Traits::kUpInputSlots>::value;
    union {
        ActivationSmem<kUpInputSlots, UpGemm::kTileShapeN, UpGemm::kTileShapeK>
            input;
        UpDownActivation up_down_act;
    };

    static constexpr std::uint32_t kNumStagingSlots = UpGemm::kNumTilesK;
    QuantizedActivationSmem<kNumStagingSlots, UpGemm::kTileShapeN,
                            UpGemm::kTileShapeK, UpGemm::kGemmN, UpGemm::kGemmK,
                            Dims::kQuantBlockK>
        quantized_input;

    static constexpr std::uint32_t kNumActQuantBlocks =
        Dims::kIntermediateSize / Traits::UpGemm::kActQuantBlock;
    DataType::ElementScale act_down_scale[kNumActQuantBlocks]
                                         [DownGemm::kTileShapeN];

    static constexpr uint32_t kNumUpScaleSlots = 2;
    DataType::ElementScale w_up_scale[kNumUpScaleSlots]
                                     [2 * UpGemm::kNumScaleBlocksK];

    DataType::ElementScale w_down_scale[DownGemm::MmaCore::kNumAtomMs]
                                       [DownGemm::kNumScaleBlocksK];
};

template <typename Dims>
struct MoESharedStorage {
    using Traits = MoEKernelTraits<Dims>;

    static_assert(
        Dims::kNumExperts <= 65535,
        "Number of experts too high, cannot store as uint16 anymore.");

    union {
        RouterSmem<Dims> router;
        LatentDownSmem<Dims> latent_down;
        RoutingSmem<Dims> routing_scratch;
        SharedExpertSmem<Dims> shared_expert;
        RoutedExpertsSmem<Dims> routed_experts;
    };

    uint32_t tmem_ptr;

    MoERoutingTables<Dims> routing;

    MoEBarriers<Dims> barriers;
    static_assert(
        MoEBarriers<Dims>::kNumScaleSlots ==
            RoutedExpertsSmem<Dims>::kNumUpScaleSlots,
        "the scale ring and the scale tile must agree on the slot "
        "count: the fetch warp fills slot e % kNumUpScaleSlots and the "
        "barrier parity is per slot");

    uint32_t launch_parity;

    static constexpr uint32_t kMaxDownTasks =
        MoERoutingTables<Dims>::kMaxActiveExperts *
        Traits::DownGemm::kCtasPerExpert;
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
    // The two weight rings are one union view, so they must start together.
    static_assert(offsetof(RoutedExpertsSmem<Dims>, w_up) ==
                      offsetof(RoutedExpertsSmem<Dims>, w_down),
                  "the two weight rings must share one region.");
    // H100/H200's per-block opt-in budget is 228 KB; 224 leaves driver margin.
    static_assert(
        sizeof(MoESharedStorage<Dims>) <= 224 * 1024,
        "MoESharedStorage layout exceeds the 224 KB per-block SHM budget.");
    return sizeof(MoESharedStorage<Dims>);
}

}  // namespace halops::moe
