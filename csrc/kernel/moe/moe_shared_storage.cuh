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

template <std::uint32_t NumInputSlots, std::uint32_t NumWeightSlots,
          std::uint32_t NumAccSlots>
struct GemmPipelineBarrier {
    TransactionBarrier<NumInputSlots> input_ready;
    SimpleBarrier<NumInputSlots> input_empty;
    TransactionBarrier<NumWeightSlots> weight_ready;
    SimpleBarrier<NumWeightSlots> weight_empty;
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
        if (iter >= NumInputSlots) {
            input_empty.wait(iter - NumInputSlots);
        }
    }

    __device__ __forceinline__ void wait_weight_empty(std::uint32_t iter) {
        if (iter >= NumWeightSlots) {
            weight_empty.wait(iter - NumWeightSlots);
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
    static constexpr std::uint32_t kNumUpWeightSlots =
        RoutedUpGemmTraits<Dims>::kNumWeightSlots;
    static constexpr std::uint32_t kNumDownWeightSlots =
        RoutedDownGemmTraits<Dims>::kNumWeightSlots;
    static constexpr std::uint32_t kNumScaleSlots =
        RoutedUpGemmTraits<Dims>::kWeightScaleSlots;
    static constexpr std::uint32_t kNumAccSlots =
        Dims::KernelConfig::kNumAccSlots;

    // One bulk copy publishes all input tiles and scales for routed up.
    TransactionBarrier<1> up_input_ready;
    TransactionBarrier<kNumUpWeightSlots> up_weight_ready;
    TransactionBarrier<kNumDownWeightSlots> down_weight_ready;
    TransactionBarrier<kNumDownWeightSlots> down_input_ready;

    SimpleBarrier<kNumScaleSlots> weight_scale_ready;
    SimpleBarrier<kNumScaleSlots> weight_scale_empty;

    // Each K tile's completion also releases its input and weight slots.
    SimpleBarrier<kNumAccSlots> accumulator_ready;
    SimpleBarrier<kNumAccSlots> accumulator_empty;

    __device__ __forceinline__ void init() {
        const int num_warps = static_cast<int>(WarpRole::kCalcWarpCount);
        up_input_ready.init(1);
        up_weight_ready.init(1);
        down_weight_ready.init(1);
        down_input_ready.init(1);
        weight_scale_ready.init(1);
        weight_scale_empty.init(num_warps);
        accumulator_ready.init(1);
        accumulator_empty.init(num_warps);
    }

    __device__ __forceinline__ void wait_up_mma_ready(std::uint32_t iter) {
        if (iter >= kNumAccSlots) {
            accumulator_empty.wait(iter - kNumAccSlots);
        }
        up_weight_ready.wait(iter);
        if (iter == 0u) {
            up_input_ready.wait(0u);
        }
    }

    __device__ __forceinline__ void wait_down_mma_ready(
        std::uint32_t local_iter, std::uint32_t acc_iter) {
        if (acc_iter >= kNumAccSlots) {
            accumulator_empty.wait(acc_iter - kNumAccSlots);
        }
        down_weight_ready.wait(local_iter);
        down_input_ready.wait(local_iter);
    }

    __device__ __forceinline__ void wait_weight_scale_ready(
        std::uint32_t task_id, std::uint32_t slot) {
        weight_scale_ready.wait_slot(slot, (task_id / kNumScaleSlots) & 1u);
    }

    __device__ __forceinline__ void wait_weight_scale_empty(
        std::uint32_t task_id, std::uint32_t slot) {
        weight_scale_empty.wait_slot(slot, (task_id / kNumScaleSlots) & 1u);
    }
};

template <typename Dims>
struct RoutedBarrier<Dims, true> {
    static constexpr std::uint32_t kNumUpWeightSlots =
        RoutedUpGemmTraits<Dims>::kNumWeightSlots;
    static constexpr std::uint32_t kNumDownWeightSlots =
        RoutedDownGemmTraits<Dims>::kNumWeightSlots;
    // Up rotates its tasks over these slots; down continues the same ring.
    static constexpr std::uint32_t kNumAccSlots =
        RoutedUpGemmTraits<Dims>::kNumAccSlots;

    TransactionBarrier<1> up_input_ready;
    TransactionBarrier<kNumUpWeightSlots> up_weight_ready;
    SimpleBarrier<kNumUpWeightSlots> up_weight_empty;

    TransactionBarrier<kNumDownWeightSlots> down_input_ready;
    TransactionBarrier<kNumDownWeightSlots> down_weight_ready;
    SimpleBarrier<kNumDownWeightSlots> down_weight_empty;

    SimpleBarrier<kNumAccSlots> accumulator_ready;
    SimpleBarrier<kNumAccSlots> accumulator_empty;

    __device__ __forceinline__ void init() {
        const int num_warps = static_cast<int>(WarpRole::kCalcWarpCount);
        up_input_ready.init(1);
        up_weight_ready.init(1);
        up_weight_empty.init(1);
        down_weight_ready.init(1);
        down_input_ready.init(1);
        down_weight_empty.init(1);
        accumulator_ready.init(1);
        accumulator_empty.init(num_warps);
    }

    __device__ __forceinline__ void wait_up_mma_ready(std::uint32_t iter) {
        up_weight_ready.wait(iter);
        if (iter == 0u) {
            up_input_ready.wait(0u);
        }
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
        GemmPipelineBarrier<RouterGemmTraits<Dims>::kNumInputSlots,
                            RouterGemmTraits<Dims>::kNumWeightSlots, 1u>;
    using LatentDownPipeline =
        GemmPipelineBarrier<LatentDownGemmTraits<Dims>::kNumInputSlots,
                            LatentDownGemmTraits<Dims>::kNumWeightSlots,
                            LatentDownGemmTraits<Dims>::kNumAccSlots>;

    RouterPipeline router;
    LatentDownPipeline latent_down;
    RoutedBarrier<Dims> routed;

    // SharedExpert
    static constexpr std::uint32_t kNumSharedUpWeightSlots =
        SharedUpGemmTraits<Dims>::kNumWeightSlots;
    static constexpr std::uint32_t kNumSharedDownWeightSlots =
        SharedDownGemmTraits<Dims>::kNumWeightSlots;
    using SharedUpPipeline =
        GemmPipelineBarrier<kNumSharedUpWeightSlots, kNumSharedUpWeightSlots,
                            SharedUpGemmTraits<Dims>::kNumAccSlots>;
    SharedUpPipeline shared_up;

    using SharedDownPipeline =
        GemmPipelineBarrier<kNumSharedDownWeightSlots,
                            kNumSharedDownWeightSlots, 1u>;
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

// BF16 activation ring for quantization and MMA input.
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
    static constexpr std::uint32_t kNumSlots = GemmTraits::kWeightScaleSlots;
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
    static constexpr std::uint32_t kNumSlots = GemmTraits::kNumWeightSlots;
    // The slot must hold the largest layout a launch can select: a natural-row
    // slot stages the gate and up atoms side by side.
    static constexpr std::uint32_t kScaleCols =
        2u * GemmTraits::MmaCore::ScaleTraits::kSfaSmemBytes;
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
    using Quant = typename GemmTraits::InputQuant;
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
    static constexpr std::uint32_t kNumSlots = GemmTraits::kNumWeightSlots;
    static constexpr std::uint32_t kSfAtomBytes =
        GemmTraits::MmaCore::ScaleTraits::kSfbSmemBytes;
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
    static constexpr std::uint32_t kLboChunkK = LboChunkK<ElementType>::value;
    static constexpr std::uint32_t kNumLboChunksK = TileShapeK / kLboChunkK;
    static_assert(TileShapeK % kLboChunkK == 0u,
                  "a tile must be whole 16-B chunks");

    // MXFP8 uses per-slot SfAtoms; blockwise uses per-block, per-row scales.
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

template <typename GemmTraits>
struct RouterSmem {
    ActivationSmem<GemmTraits::kNumInputSlots, GemmTraits::kTileShapeN,
                   GemmTraits::kTileShapeK>
        input;

    alignas(1024) typename GemmTraits::MmaElement
        w_router[GemmTraits::kNumWeightSlots]
                [GemmTraits::MmaCore::kWeightTileElems];

    static constexpr std::uint32_t kNumMmaWarps =
        GemmTraits::kUseUmma ? 1u : WarpRole::kCalcWarpCount;
    static constexpr std::uint32_t kMmaRows =
        GemmTraits::kUseUmma ? 1u : GemmTraits::kTileShapeN;
    static constexpr std::uint32_t kMmaCols =
        GemmTraits::kUseUmma ? 1u : GemmTraits::kTileShapeM;
    float mma_out[kNumMmaWarps][kMmaRows][kMmaCols];
};

template <typename GemmTraits>
struct LatentDownSmem {
    alignas(1024) typename GemmTraits::MmaElement w_latent_down
        [GemmTraits::kNumWeightSlots][GemmTraits::MmaCore::kNumSwizzleBoxKs]
        [GemmTraits::kTileShapeM][GemmTraits::MmaCore::kSwizzleBoxK];

    ActivationSmem<GemmTraits::kNumInputSlots, GemmTraits::kTileShapeN,
                   GemmTraits::kTileShapeK>
        input;

    QuantizedActivationSmem<
        GemmTraits::kNumInputSlots, GemmTraits::kTileShapeN,
        GemmTraits::kTileShapeK, GemmTraits::kGemmN, GemmTraits::kGemmK,
        GemmTraits::InputQuant::kBlockSize, GemmTraits::InputQuant::kMethod>
        quantized_input;

    // MXFP8: this CTA's 128-row SfAtom scale slab.
    static constexpr std::uint32_t kSfaSlabBytes =
        GemmTraits::MmaCore::kUseBlockscaled ? 128u * (GemmTraits::kGemmK / 32u)
                                             : 0u;
    alignas(16) std::uint8_t sfa_slab[kSfaSlabBytes ? kSfaSlabBytes : 1u];

    // BF16 staging follows the quantized output's chunk boundaries.
    static constexpr std::uint32_t kOutChunkK =
        LboChunkK<typename GemmTraits::OutputQuant::ElementType>::value;
    static constexpr std::uint32_t kNumOutChunks =
        GemmTraits::kTileShapeM / kOutChunkK;
    static constexpr std::uint32_t kNumOutRows = GemmTraits::kTileShapeN;
    alignas(
        16) DataType::ElementA act_out[kNumOutChunks][kNumOutRows][kOutChunkK];
    alignas(16) float token_amax[kNumOutRows];

    static constexpr std::uint32_t kNumOutGroups =
        GemmTraits::kTileShapeM / 32u;
    alignas(16) float token_group_amax[kNumOutRows][kNumOutGroups];

    alignas(16) float splited_mma_out[GemmTraits::kTileShapeM]
                                     [GemmTraits::kTileShapeN];
};

// The shared expert's own storage, in the union with the other phases.
template <typename SharedUpGemm, typename SharedDownGemm>
struct SharedExpertSmem {
    union {
        alignas(1024) typename SharedUpGemm::MmaElement
            w_up[SharedUpGemm::kNumWeightSlots][2u][SharedUpGemm::kTileShapeM]
                [128u];
        alignas(1024) typename SharedDownGemm::MmaElement
            w_down[SharedDownGemm::kNumWeightSlots][SharedDownGemm::kTileShapeM]
                  [SharedDownGemm::kTileShapeK];
        alignas(1024) float splited_mma_out[2u][SharedUpGemm::kTileShapeM / 2u]
                                           [SharedUpGemm::kTileShapeN];
    };

    ActivationSmem<SharedUpGemm::kNumWeightSlots, SharedUpGemm::kTileShapeN,
                   SharedUpGemm::kTileShapeK>
        input;

    alignas(1024) QuantizedActivationSmem<
        SharedUpGemm::kNumWeightSlots, SharedUpGemm::kTileShapeN,
        SharedUpGemm::kTileShapeK, SharedUpGemm::kGemmN, SharedUpGemm::kGemmK,
        SharedUpGemm::InputQuant::kBlockSize,
        SharedUpGemm::InputQuant::kMethod> quantized_input;

    alignas(1024) QuantizedActivationSmem<
        SharedDownGemm::kNumWeightSlots, SharedDownGemm::kTileShapeN,
        SharedDownGemm::kTileShapeK, SharedDownGemm::kGemmN,
        SharedDownGemm::kGemmK, SharedDownGemm::InputQuant::kBlockSize,
        SharedDownGemm::InputQuant::kMethod> act_down_input;

    alignas(1024) WeightScaleStorage<SharedUpGemm> w_up_scale;
    alignas(1024) WeightScaleStorage<SharedDownGemm> w_down_scale;

    alignas(16) float act_up_out[SharedUpGemm::kTileShapeM / 2u]
                                [SharedUpGemm::kTileShapeN + 1u];
};

template <typename UpGemm, typename DownGemm>
struct RoutedExpertsSmem {
    union {
        alignas(1024) typename UpGemm::MmaElement
            w_up[UpGemm::kNumWeightSlots][UpGemm::MmaCore::kNumSwizzleBoxKs]
                [UpGemm::kTileShapeM * UpGemm::MmaCore::kSwizzleBoxK];

        alignas(1024) typename DownGemm::MmaElement
            w_down[DownGemm::kNumWeightSlots][DownGemm::kTileShapeM]
                  [DownGemm::kTileShapeK];
    };

    float act_up_out[UpGemm::kTileShapeM / 2u][UpGemm::kTileShapeN + 1];

    static constexpr std::uint32_t kNumStagingSlots = UpGemm::kNumTilesK;
    QuantizedActivationSmem<kNumStagingSlots, UpGemm::kTileShapeN,
                            UpGemm::kTileShapeK, UpGemm::kGemmN, UpGemm::kGemmK,
                            UpGemm::InputQuant::kBlockSize,
                            UpGemm::InputQuant::kMethod>
        quantized_input;

    alignas(1024) typename DownGemm::InputQuant::ElementType
        act_down_input[DownGemm::kNumWeightSlots]
                      [DownGemm::kTileShapeN * DownGemm::kTileShapeK];

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
        RouterSmem<typename Traits::RouterGemm> router;
        LatentDownSmem<typename Traits::LatentDownGemm> latent_down;
        RoutingSmem<Dims> routing_scratch;
        SharedExpertSmem<SharedUpGemmTraits<Dims>, SharedDownGemmTraits<Dims>>
            shared_expert;
        RoutedExpertsSmem<typename Traits::RoutedUpGemm,
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
        Traits::RoutedDownGemm::kCtasPerExpert;
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
    // Reserve shared-memory headroom for the driver.
    static_assert(
        sizeof(MoESharedStorage<Dims>) <= 224 * 1024,
        "MoESharedStorage layout exceeds the 224 KB per-block SHM budget.");
    return sizeof(MoESharedStorage<Dims>);
}

}  // namespace halops::moe
