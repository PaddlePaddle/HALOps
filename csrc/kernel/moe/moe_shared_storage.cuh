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

#include "common/device_timer.hpp"
#include "kernel/moe/moe_configs.cuh"
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
        Dims::KernelConfig::kNumUpWeightSlots;
    static constexpr std::uint32_t kNumDownWeightSlots =
        Dims::KernelConfig::kNumDownWeightSlots;
    static constexpr std::uint32_t kNumScaleSlots =
        MoEKernelTraits<Dims>::UpGemm::kNumWeightScaleSlots;
    static constexpr std::uint32_t kNumActivationSlots = kNumDownWeightSlots;
    static constexpr std::uint32_t kNumAccumulatorSlots =
        Dims::KernelConfig::kNumAccSlots;
    static constexpr std::uint32_t kNumRouterWeightSlots =
        RouterGemmTraits<Dims>::kNumWeightSlots;

    // LatentDown
    static constexpr std::uint32_t kNumLatentWeightSlots =
        LatentDownGemmTraits<Dims>::kNumWeightSlots;
    TransactionBarrier<MoEKernelTraits<Dims>::kLatentInputSlots>
        latent_down_input_ready;
    TransactionBarrier<kNumLatentWeightSlots> latent_down_weight_ready;
    SimpleBarrier<kNumLatentWeightSlots> latent_down_weight_empty;

    // The quantize's handshake: the issuer can lag the quantize by a whole K.
    static constexpr std::uint32_t kNumLatentInputEmptySlots =
        LatentDownGemmTraits<Dims>::kNumTilesK;
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
    // Blockscaled up reuses weights per K tile and accumulators per task.
    SimpleBarrier<kNumUpWeightSlots> up_weight_empty;
    // One fixed TMEM accumulator, handed off once per blockscaled up task.
    SimpleBarrier<1> up_accumulator_ready;
    SimpleBarrier<1> up_accumulator_empty;

    // The up phase's fp8 staging, one slot per K tile.
    static constexpr std::uint32_t kNumUpStagingSlots =
        MoEKernelTraits<Dims>::UpGemm::kNumTilesK;
    SimpleBarrier<kNumUpStagingSlots> up_staging_ready;
    // One bulk copy of the whole projection output.
    TransactionBarrier<1> latent_copy_ready;

    SimpleBarrier<kNumScaleSlots> weight_scale_ready;
    SimpleBarrier<kNumScaleSlots> weight_scale_empty;

    // RoutedDown
    TransactionBarrier<kNumDownWeightSlots> down_weight_ready;
    TransactionBarrier<kNumActivationSlots> activation_ready;
    // Blockscaled down releases both operand slots after each K tile's MMA.
    SimpleBarrier<kNumDownWeightSlots> down_weight_empty;
    // One fixed TMEM accumulator, handed off once per blockscaled down task.
    SimpleBarrier<1> down_accumulator_ready;
    SimpleBarrier<1> down_accumulator_empty;

    // TMEM accumulator ring for the blockwise up and down paths.
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
    // MX hands off one completed split; blockwise hands off each K tile.
    static constexpr std::uint32_t kNumSharedUpAccSlots =
        SharedUpGemmTraits<Dims>::MmaCore::kUseBlockscaled
            ? 1u
            : MoEKernelTraits<Dims>::kNumAccSlots;
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
        up_staging_ready.init(1);
        latent_copy_ready.init(1);
        up_weight_ready.init(1);
        if constexpr (MoEKernelTraits<Dims>::UpGemm::MmaCore::kUseBlockscaled) {
            up_weight_empty.init(1);
            up_accumulator_ready.init(1);
            up_accumulator_empty.init(num_warps);
        }
        down_weight_ready.init(1);
        activation_ready.init(1);
        if constexpr (MoEKernelTraits<
                          Dims>::DownGemm::MmaCore::kUseBlockscaled) {
            down_weight_empty.init(1);
            down_accumulator_ready.init(1);
            down_accumulator_empty.init(num_warps);
        }
        weight_scale_ready.init(1);
        weight_scale_empty.init(num_warps);
        init_accumulator_ring();
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

    // The expert phases' TMEM accumulator ring: the UMMA's completion arms a
    // slot's `ready`, and every calc warp releases it through `empty`.
    __device__ __forceinline__ void init_accumulator_ring() {
        accumulator_ready.init(1);
        accumulator_empty.init(static_cast<int>(WarpRole::kCalcWarpCount));
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
    alignas(
        16) float topk_weights[(Dims::kNumTokens < 8u ? 8u : Dims::kNumTokens) *
                               Dims::kMaxTopk];

    std::uint8_t expert_position[Dims::kNumExperts];

    // Every expert's routed tokens, one bit per token; the tables keep the
    // active experts' copies.
    alignas(16) std::uint32_t expert_token_mask[Dims::kNumExperts];
};

// A ring of streamed bf16 activation tiles: B operand and quantize source.
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

template <typename GemmTraits,
          bool UseBlockscaled = GemmTraits::MmaCore::kUseBlockscaled>
struct WeightScaleStorage;

// Blockwise scale tables follow the GEMM's prefetch granularity.
template <typename GemmTraits>
struct WeightScaleStorage<GemmTraits, false> {
    using Quant = typename GemmTraits::WeightQuant;
    static_assert(Quant::kMethod == QuantMethod::kBlockwiseFP8,
                  "the blockwise storage requires FP32 block scales");
    static constexpr std::uint32_t kNumSlots = GemmTraits::kNumWeightScaleSlots;
    static constexpr auto kScaleShape = GemmTraits::kWeightScaleShape;
    static constexpr std::uint32_t kScaleRows =
        static_cast<std::uint32_t>(cute::get<0>(kScaleShape));
    static constexpr std::uint32_t kScaleCols =
        static_cast<std::uint32_t>(cute::get<1>(kScaleShape));
    alignas(128)
        typename Quant::ScaleType scale[kNumSlots][kScaleRows * kScaleCols];

    __device__ auto& operator[](std::uint32_t slot) { return scale[slot]; }
    __device__ const auto& operator[](std::uint32_t slot) const {
        return scale[slot];
    }
};

// MX stages each weight tile's SfAtom slabs in the corresponding weight slot.
template <typename GemmTraits>
struct WeightScaleStorage<GemmTraits, true> {
    using Quant = typename GemmTraits::WeightQuant;
    static constexpr std::uint32_t kNumSlots = GemmTraits::kNumWeightSlots;
    static constexpr std::uint32_t kScaleCols =
        GemmTraits::kTileShapeM * GemmTraits::kTileShapeK / Quant::kBlockSize;
    alignas(128) std::uint8_t scale[kNumSlots][kScaleCols];

    __device__ auto& operator[](std::uint32_t slot) { return scale[slot]; }
    __device__ const auto& operator[](std::uint32_t slot) const {
        return scale[slot];
    }
};

template <typename GemmTraits,
          bool UseBlockscaled = GemmTraits::MmaCore::kUseBlockscaled>
struct ActivationScaleStorage;

// Blockwise down keeps the current task's full K scale table.
template <typename GemmTraits>
struct ActivationScaleStorage<GemmTraits, false> {
    using Quant = typename GemmTraits::ActQuant;
    static constexpr std::uint32_t kNumScaleBlocksK =
        GemmTraits::kGemmK / Quant::kBlockSize;
    alignas(128) typename Quant::ScaleType
        scale[kNumScaleBlocksK][GemmTraits::kTileShapeN];

    __device__ auto& operator[](std::uint32_t block) { return scale[block]; }
    __device__ const auto& operator[](std::uint32_t block) const {
        return scale[block];
    }
};

// MX stages each activation tile's SfAtom in the corresponding input slot.
template <typename GemmTraits>
struct ActivationScaleStorage<GemmTraits, true> {
    using Quant = typename GemmTraits::ActQuant;
    static constexpr std::uint32_t kNumSlots = GemmTraits::kNumWeightSlots;
    static constexpr std::uint32_t kSfAtomBytes =
        128u * (GemmTraits::kTileShapeK / Quant::kBlockSize);
    alignas(128) std::uint8_t scale[kNumSlots][kSfAtomBytes];

    __device__ auto& operator[](std::uint32_t slot) { return scale[slot]; }
    __device__ const auto& operator[](std::uint32_t slot) const {
        return scale[slot];
    }
};

// The quantize's fp8 output, which is also the MMA's B operand.
template <std::uint32_t NumSlots, std::uint32_t TileShapeN,
          std::uint32_t TileShapeK, std::uint32_t GemmN, std::uint32_t GemmK,
          std::uint32_t QuantBlock,
          QuantMethod Method = QuantMethod::kBlockwiseFP8>
struct QuantizedActivationSmem {
    using Quant = QuantSpec<Method, QuantBlock>;
    using ElementType = typename Quant::ElementType;
    using ElementScale = typename Quant::ScaleType;

    static constexpr std::uint32_t kNumSlots = NumSlots;
    static constexpr std::uint32_t kLboChunkK = kFp8ActChunkK;
    static constexpr std::uint32_t kNumLboChunksK = TileShapeK / kLboChunkK;
    static_assert(TileShapeK % kLboChunkK == 0u,
                  "a tile must be whole 16-B chunks");

    // MXFP8 scales are staged per slot in their SfAtom layout; the other
    // recipes keep one scale per (block, row).
    static constexpr std::uint32_t kSfAtomBytes =
        128u * (TileShapeK / Quant::kBlockSize);
    using ScaleArray =
        std::conditional_t<Method == QuantMethod::kMXFP8,
                           std::uint8_t[kNumSlots][kSfAtomBytes],
                           ElementScale[GemmK / Quant::kBlockSize][GemmN]>;

    alignas(1024) ElementType
        data[kNumSlots][kNumLboChunksK][TileShapeN + 1u][kLboChunkK];
    alignas(128) ScaleArray scale;
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
        RouterGemm::kUseUmma ? 1u : WarpRole::kCalcWarpCount;
    static constexpr std::uint32_t kMmaRows =
        RouterGemm::kUseUmma ? 1u : RouterGemm::kTileShapeN;
    static constexpr std::uint32_t kMmaCols =
        RouterGemm::kUseUmma ? 1u : RouterGemm::kTileShapeM;
    float mma_out[kNumMmaWarps][kMmaRows][kMmaCols];
};

template <typename Dims>
struct LatentDownSmem {
    using Traits = MoEKernelTraits<Dims>;
    using LatentDownGemm = typename Traits::LatentDownGemm;

    alignas(1024) LatentDownGemm::MmaElement
        w_latent_down[LatentDownGemm::kNumWeightSlots]
                     [LatentDownGemm::kQuantBlocksPerTile]
                     [LatentDownGemm::kTileShapeM][128u];

    ActivationSmem<Traits::kLatentInputSlots, LatentDownGemm::kTileShapeN,
                   LatentDownGemm::kTileShapeK>
        input;

    QuantizedActivationSmem<
        Traits::kLatentInputSlots, LatentDownGemm::kTileShapeN,
        LatentDownGemm::kTileShapeK, LatentDownGemm::kGemmN,
        LatentDownGemm::kGemmK, Dims::kQuantConfig.kBlockSize,
        Dims::kQuantConfig.kLatentQuant>
        quantized_input;

    // MX stages SfAtoms here; blockwise reads scales directly from gmem.
    WeightScaleStorage<LatentDownGemm> w_latent_down_scale;

    static constexpr std::uint32_t kNumOutChunks =
        LatentDownGemm::kTileShapeM / kFp8ActChunkK;
    static constexpr std::uint32_t kNumOutRows = LatentDownGemm::kTileShapeN;
    alignas(16)
        DataType::ElementA out_bf16[kNumOutChunks][kNumOutRows][kFp8ActChunkK];
    alignas(16) float token_amax[kNumOutRows];
    // MXFP8 scales each 32-feature group on its own; the block-wise recipe
    // reduces the whole block into `token_amax`.
    static constexpr std::uint32_t kNumOutGroups =
        LatentDownGemm::kTileShapeM / 32u;
    alignas(16) float token_group_amax[kNumOutRows][kNumOutGroups];

    alignas(16) float splited_mma_out[LatentDownGemm::kTileShapeM]
                                     [LatentDownGemm::kTileShapeN];
};

// The shared expert's own storage, in the union with the other phases.
template <typename Dims>
struct SharedExpertSmem {
    using SharedUpGemm = SharedUpGemmTraits<Dims>;
    using SharedDownGemm = SharedDownGemmTraits<Dims>;

    union {
        alignas(1024)
            SharedUpGemm::MmaElement w_up[SharedUpGemm::kNumWeightSlots][2u]
                                         [SharedUpGemm::kTileShapeM][128u];
        alignas(1024) SharedDownGemm::MmaElement
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
        Dims::kQuantConfig.kBlockSize,
        SharedUpGemm::ActQuant::kMethod> quantized_input;

    alignas(1024) QuantizedActivationSmem<
        SharedDownGemm::kNumWeightSlots, Dims::kNumTokens,
        SharedDownGemm::kTileShapeK, Dims::kNumTokens, Dims::kIntermediateSize,
        SharedUpGemm::kActQuantBlock,
        SharedDownGemm::ActQuant::kMethod> act_down_input;

    // The blockscaled recipe stages each weight slot's SfAtom slab here; the
    // blockwise recipe reads its fp32 table straight from gmem.
    alignas(1024) WeightScaleStorage<SharedUpGemm> w_up_scale;
    alignas(1024) WeightScaleStorage<SharedDownGemm> w_down_scale;

    alignas(16) float act_up_out[SharedUpGemm::kActQuantBlock]
                                [Dims::kNumTokens + 1u];
};

template <typename Dims>
struct RoutedExpertsSmem {
    using Traits = MoEKernelTraits<Dims>;
    using UpGemm = typename Traits::UpGemm;
    using DownGemm = typename Traits::DownGemm;

    union {
        alignas(1024) UpGemm::MmaElement
            w_up[UpGemm::kNumWeightSlots][UpGemm::MmaCore::kNumSwizzleBoxKs]
                [UpGemm::kTileShapeM * UpGemm::MmaCore::kSwizzleBoxK];

        alignas(1024) DownGemm::MmaElement
            w_down[DownGemm::kNumWeightSlots][DownGemm::kTileShapeM]
                  [DownGemm::kTileShapeK];
    };

    float act_up_out[Traits::UpGemm::kActQuantBlock][UpGemm::kTileShapeN + 1];

    alignas(1024) DataType::ElementQA
        act_down_input[DownGemm::kNumWeightSlots]
                      [DownGemm::kTileShapeN * DownGemm::kTileShapeK];

    static constexpr std::uint32_t kNumStagingSlots = UpGemm::kNumTilesK;
    QuantizedActivationSmem<kNumStagingSlots, UpGemm::kTileShapeN,
                            UpGemm::kTileShapeK, UpGemm::kGemmN, UpGemm::kGemmK,
                            Dims::kQuantConfig.kBlockSize,
                            UpGemm::ActQuant::kMethod>
        quantized_input;

    alignas(1024) WeightScaleStorage<UpGemm> w_up_scale;
    alignas(1024) WeightScaleStorage<DownGemm> w_down_scale;

    ActivationScaleStorage<DownGemm> act_down_scale;
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
            Traits::UpGemm::kNumWeightScaleSlots,
        "the scale ring and the scale tile must agree on the slot "
        "count: the fetch warp fills slot e % kNumWeightScaleSlots and the "
        "barrier parity is per slot");

    uint32_t launch_parity;
    uint32_t tail_parity;

    static constexpr uint32_t kMaxDownTasks =
        MoERoutingTables<Dims>::kMaxActiveExperts *
        Traits::DownGemm::kCtasPerExpert;
    static constexpr uint32_t kNoDownTask = 0xFFFFu;
    static_assert(kMaxDownTasks < kNoDownTask,
                  "down grid task ids must fit in uint16 below the sentinel");
    uint16_t down_grid_task[kMaxDownTasks + 1];
    uint32_t down_tasks_claimed;
    uint32_t down_task_stores;

#if defined(HALOPS_KERNEL_TIMER_ENABLED)
    // Per-warp slot write-cursors for the device event timer, zeroed in the
    // kernel prologue (before any probe) and published by its __syncthreads.
    // One entry per warp; the block must not exceed kTimerWarps warps.
    uint32_t timer_next[halops::kTimerWarps];
    static_assert(Dims::KernelConfig::kBlockSize / 32u <= halops::kTimerWarps,
                  "block has more warps than the timer's per-warp cursors");
    // The per-CTA record pool pointer (null = timer off), parked here by the
    // prologue so any phase can mint a writer via `timer()`
    // without threading it through its call chain.
    std::uint64_t* timer_base;

    __device__ __forceinline__ halops::DeviceTimer timer() {
        return halops::DeviceTimer::make(
            timer_base, timer_next, static_cast<int>(blockIdx.x),
            static_cast<int>(WarpRole::get_warp_idx()));
    }
#endif
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
