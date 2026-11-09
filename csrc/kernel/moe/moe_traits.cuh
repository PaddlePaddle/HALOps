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

#include <cutlass/arch/barrier.h>

#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/atom/mma_traits_sm100.hpp>
#include <cute/tensor.hpp>
#include <type_traits>

#include "kernel/moe/mma_core.cuh"
#include "kernel/moe/moe_configs.cuh"
#include "kernel/moe/moe_tail.cuh"
#include "kernel/moe/ptx_utils.cuh"

namespace halops::moe {

// Tile shape shared by the router Linear and the latent down-projection.
template <std::uint32_t GemmM, std::uint32_t GemmN, std::uint32_t GemmK,
          std::uint32_t TileShapeN, std::uint32_t NumCtas, bool UseUmma>
struct GemmTileShape {
    static constexpr std::uint32_t kTileShapeN = TileShapeN;
    static constexpr std::uint32_t kNumTilesN = GemmN / TileShapeN;
    static_assert(GemmN % TileShapeN == 0u,
                  "the token width must be whole N tiles");
    static_assert(NumCtas % kNumTilesN == 0u,
                  "the CTAs must cover whole N tiles");

    static constexpr std::uint32_t kNumTilesM = NumCtas / kNumTilesN;
    static constexpr std::uint32_t kTileShapeM = GemmM / kNumTilesM;
    static_assert(GemmM % kNumTilesM == 0u,
                  "the rows must divide evenly across the projection's tiles");
    // Whole 8-row swizzle atoms keep every packed box 1 KB aligned.
    static_assert(kTileShapeM % 8u == 0u,
                  "the tile's M must be whole 8-row swizzle atoms");
    static_assert(!UseUmma || kTileShapeM % 64u == 0u,
                  "a UMMA tile must be whole 64-row instruction tiles");
    static_assert(UseUmma || kTileShapeM % 16u == 0u,
                  "an HMMA tile must be whole 16-row instructions");

    // K tile: two 128-K quantization blocks, one weight-ring slot.
    static constexpr std::uint32_t kTileShapeK = 256u;
    static_assert(GemmK % kTileShapeK == 0u,
                  "the reduction must be a whole number of K tiles");
    static constexpr std::uint32_t kNumTilesK = GemmK / kTileShapeK;

    using TileShape =
        cute::Shape<cute::Int<kTileShapeM>, cute::Int<kTileShapeN>,
                    cute::Int<kTileShapeK>>;
};

// Latent projection: M = latent width, N = tokens, K = hidden width.
template <typename Dims>
struct LatentDownGemmTraits {
    static constexpr std::uint32_t kGemmM = Dims::kLatentSize;
    static constexpr std::uint32_t kGemmN = Dims::kNumTokens;
    static constexpr std::uint32_t kGemmK = Dims::kHiddenSize;

    using WeightQuant = QuantSpec<Dims::kQuantConfig.kLatentQuant>;
    using InputQuant = QuantSpec<Dims::kQuantConfig.kLatentQuant>;
    using OutputQuant = QuantSpec<Dims::kQuantConfig.kExpertsActQuant,
                                  Dims::kQuantConfig.kBlockSize>;

    // The output's scale image: a blockscaled recipe stages SfAtom slabs.
    static constexpr bool kOutputUseBlockscaled =
        OutputQuant::kMethod != QuantMethod::kBlockwiseFP8;

    using MmaElement = typename WeightQuant::ElementType;
    using ElementScale = typename WeightQuant::ScaleType;

    static constexpr std::uint32_t kNumCtas =
        Dims::KernelConfig::kNumCtasForLatentDown;
    static constexpr std::uint32_t kTileShapeN = 8u;
    static constexpr std::uint32_t kNumTilesN = kGemmN / kTileShapeN;

    static constexpr bool kUseSplitK = Dims::KernelConfig::kLatentDownUseSplitK;

    static constexpr std::uint32_t kNumSplits =
        kUseSplitK ? kNumCtas / ((kGemmM / 128u) * kNumTilesN) : 1u;
    static_assert(!kUseSplitK ||
                      kNumCtas % ((kGemmM / 128u) * kNumTilesN) == 0u,
                  "the split groups must divide the CTAs");

    using GemmShape = GemmTileShape<kGemmM, kGemmN, kGemmK, kTileShapeN,
                                    kNumCtas / kNumSplits, /*UseUmma=*/true>;

    static constexpr std::uint32_t kTileShapeM =
        kUseSplitK ? 128u : GemmShape::kTileShapeM;
    static constexpr std::uint32_t kTileShapeK = GemmShape::kTileShapeK;
    using TileShape =
        cute::Shape<cute::Int<kTileShapeM>, cute::Int<kTileShapeN>,
                    cute::Int<kTileShapeK>>;

    static constexpr std::uint32_t kNumTilesM = kGemmM / kTileShapeM;
    static constexpr std::uint32_t kNumTilesK = GemmShape::kNumTilesK;

    // A K tile spans this many 128-K staging blocks.
    static constexpr std::uint32_t kNumStagingBlocks = kTileShapeK / 128u;

    static constexpr std::uint32_t kClusterSize = cute::max(
        (OutputQuant::kBlockSize + kTileShapeM - 1) / kTileShapeM, kNumSplits);

    using MmaCore = moe::UmmaCore<WeightQuant, InputQuant, float, TileShape>;

    static constexpr std::uint32_t kNumAccSlots =
        MmaCore::kUseBlockscaled ? 1 : Dims::KernelConfig::kNumAccSlots;
    static constexpr std::uint32_t kMaxWeightSlots =
        (128u * 1024u) / (kTileShapeM * kTileShapeK);
    static constexpr std::uint32_t kNumWeightSlots =
        Dims::KernelConfig::kNumLatentDownWeightSlots < kMaxWeightSlots
            ? Dims::KernelConfig::kNumLatentDownWeightSlots
            : kMaxWeightSlots;
    static constexpr std::uint32_t kNumInputSlots = kNumWeightSlots;
    static constexpr std::uint32_t kWeightScaleSlots = kNumWeightSlots;
    static constexpr auto kWeightScaleShape =
        WeightQuant::template ScaleShape<kTileShapeM, kTileShapeK>();
};

// Router Linear (logits = input x weight, bf16 in, fp32 accumulate, bf16 out).
template <typename Dims>
struct RouterGemmTraits {
    static constexpr std::uint32_t kGemmM = Dims::kNumExperts;
    static constexpr std::uint32_t kGemmN = Dims::kNumTokens;
    static constexpr std::uint32_t kGemmK = Dims::kHiddenSize;

    static constexpr bool kUseUmma =
        Dims::KernelConfig::kRouterMma == MmaKind::kUmma;
    static constexpr std::uint32_t kNumCtasForRouter =
        Dims::KernelConfig::kNumCtasForRouter;

    static constexpr std::uint32_t kNumRouterGates = Dims::kNumRouterGates;
    static_assert(kNumRouterGates == 1u || kNumRouterGates == 2u,
                  "the router runs one or two gates");
    static constexpr std::uint32_t kNumCtas =
        kNumRouterGates * kNumCtasForRouter;

    static constexpr std::uint32_t kTileShapeN = 8u;

    using Gemm = GemmTileShape<kGemmM, kGemmN, kGemmK, kTileShapeN,
                               kNumCtasForRouter, kUseUmma>;
    static constexpr std::uint32_t kTileShapeM = Gemm::kTileShapeM;
    static constexpr std::uint32_t kTileShapeK = Gemm::kTileShapeK;
    static constexpr std::uint32_t kNumTilesN = Gemm::kNumTilesN;
    static constexpr std::uint32_t kNumTilesM = Gemm::kNumTilesM;
    static constexpr std::uint32_t kNumTilesK = Gemm::kNumTilesK;
    using TileShape = typename Gemm::TileShape;

    using WeightQuant = QuantSpec<QuantMethod::kNone>;
    using InputQuant = QuantSpec<QuantMethod::kNone>;
    using MmaElement = typename WeightQuant::ElementType;

    using MmaCore = cute::conditional_t<
        kUseUmma, moe::UmmaCore<WeightQuant, InputQuant, float, TileShape>,
        moe::HmmaCore<TileShape>>;

    static constexpr std::uint32_t kNumWeightSlots =
        Dims::KernelConfig::kNumRouterWeightSlots < kNumTilesK
            ? Dims::KernelConfig::kNumRouterWeightSlots
            : kNumTilesK;
    static constexpr std::uint32_t kNumInputSlots =
        Dims::KernelConfig::kNumRouterInputSlots;
};

template <typename Dims>
struct RoutedUpGemmTraits {
    static constexpr std::uint32_t kGemmM = Dims::kIntermediateSize * 2;
    static constexpr std::uint32_t kGemmN = Dims::kNumTokens;
    static constexpr std::uint32_t kGemmK = MoEInputSize<Dims>::value;

    static constexpr std::uint32_t kCtasPerExpert =
        Dims::KernelConfig::kCtasPerExpert;
    static_assert(kGemmM % kCtasPerExpert == 0u,
                  "kCtasPerExpert must divide kGemmM (kIntermediateSize * 2).");

    static constexpr std::uint32_t kTileShapeM = kGemmM / kCtasPerExpert;
    static constexpr std::uint32_t kTileShapeN = kGemmN;
    static constexpr std::uint32_t kTileShapeK =
        Dims::KernelConfig::kUpTileShapeK;

    using WeightQuant = QuantSpec<Dims::kQuantConfig.kExpertsQuant>;
    using InputQuant = QuantSpec<Dims::kQuantConfig.kExpertsActQuant,
                                 Dims::kQuantConfig.kBlockSize>;
    using OutputQuant =
        QuantSpec<Dims::kQuantConfig.kExpertsActQuant, kTileShapeM / 2u>;

    using MmaElement = typename WeightQuant::ElementType;
    using ElementScale = typename WeightQuant::ScaleType;
    using TileShape =
        cute::Shape<cute::Int<kTileShapeM>, cute::Int<kTileShapeN>,
                    cute::Int<kTileShapeK>>;
    using MmaCore = moe::UmmaCore<WeightQuant, InputQuant, float, TileShape>;
    using TiledMma = typename MmaCore::TiledMma;
    using ScaleTraits = typename MmaCore::ScaleTraits;

    using WeightSmemElement = typename MmaCore::SmemElementA;
    using ActElement = typename InputQuant::ElementType;

    static constexpr std::uint32_t kNumTilesK = kGemmK / kTileShapeK;
    static constexpr std::uint32_t kNumSplits = 1u;
    static constexpr std::uint32_t kNumStagingBlocks = kTileShapeK / 128u;

    // The output's (the down's input) scale block.
    static constexpr std::uint32_t kOutputQuantBlock = OutputQuant::kBlockSize;

    static constexpr std::uint32_t kNumAccSlots =
        MmaCore::kUseBlockscaled ? 1 : Dims::KernelConfig::kNumAccSlots;
    static constexpr std::uint32_t kNumWeightSlots =
        Dims::KernelConfig::kNumUpWeightSlots;
    static constexpr std::uint32_t kWeightScaleSlots =
        MmaCore::kUseBlockscaled ? 1u : 2u;
    static constexpr auto kWeightScaleShape = [] {
        if constexpr (MmaCore::kUseBlockscaled) {
            return WeightQuant::template ScaleShape<kTileShapeM, kTileShapeK>();
        } else {
            return cute::make_shape(
                cute::_2{}, cute::Int<kGemmK / WeightQuant::kBlockSize>{});
        }
    }();
};

template <typename Dims>
struct RoutedDownGemmTraits {
    static constexpr std::uint32_t kGemmM = MoEInputSize<Dims>::value;
    static constexpr std::uint32_t kGemmN = Dims::kNumTokens;
    static constexpr std::uint32_t kGemmK = Dims::kIntermediateSize;

    static constexpr bool kUseLatent = Dims::kUseLatent;
    static constexpr std::uint32_t kCtasPerExpert =
        Dims::KernelConfig::kCtasPerExpert;
    static_assert(kGemmM % kCtasPerExpert == 0u,
                  "kCtasPerExpert must divide kGemmM (kHiddenSize).");

    static constexpr std::uint32_t kTileShapeM = kGemmM / kCtasPerExpert;
    static constexpr std::uint32_t kTileShapeN = kGemmN;
    static constexpr std::uint32_t kTileShapeK = 128;

    using WeightQuant = QuantSpec<Dims::kQuantConfig.kExpertsQuant>;
    using InputQuant = RoutedUpGemmTraits<Dims>::OutputQuant;

    using MmaElement = typename WeightQuant::ElementType;
    using ElementScale = typename WeightQuant::ScaleType;
    using TileShape =
        cute::Shape<cute::Int<kTileShapeM>, cute::Int<kTileShapeN>,
                    cute::Int<kTileShapeK>>;
    using MmaCore = moe::UmmaCore<WeightQuant, InputQuant, float, TileShape>;
    using TiledMma = typename MmaCore::TiledMma;
    using ScaleTraits = typename MmaCore::ScaleTraits;
    using WeightSmemElement = typename MmaCore::SmemElementA;
    using ActElement = typename InputQuant::ElementType;

    static constexpr std::uint32_t kNumTilesK = kGemmK / kTileShapeK;

    static constexpr std::uint32_t kNumAccSlots =
        MmaCore::kUseBlockscaled ? 1 : Dims::KernelConfig::kNumAccSlots;
    static constexpr std::uint32_t kNumWeightSlots =
        Dims::KernelConfig::kNumDownWeightSlots;
    static constexpr std::uint32_t kWeightScaleSlots =
        MmaCore::kUseBlockscaled ? kNumWeightSlots : 1u;
    static constexpr auto kWeightScaleShape =
        WeightQuant::template ScaleShape<kTileShapeM, kGemmK>();
};

// Shared-expert up projection over all tokens.
template <typename Dims>
struct SharedUpGemmTraits {
    static constexpr std::uint32_t kGemmM = Dims::kIntermediateSize * 2u;
    static constexpr std::uint32_t kGemmN = Dims::kNumTokens;
    static constexpr std::uint32_t kGemmK = Dims::kHiddenSize;

    // One block owns one M atom of the pair layout.
    static constexpr std::uint32_t kTileShapeM = 128u;
    static constexpr std::uint32_t kTileShapeN = kGemmN;
    static constexpr std::uint32_t kTileShapeK = 256u;

    static constexpr std::uint32_t kNumSplits =
        Dims::KernelConfig::kSharedUseSplitK
            ? Dims::KernelConfig::kNumCtasForShared / (kGemmM / kTileShapeM)
            : 1u;

    static constexpr std::uint32_t kNumCtas =
        kNumSplits * (kGemmM / kTileShapeM);
    static constexpr std::uint32_t kClusterSize = kNumSplits;

    using WeightQuant = QuantSpec<Dims::kQuantConfig.kSharedQuant>;
    using InputQuant = QuantSpec<Dims::kQuantConfig.kSharedQuant,
                                 Dims::kQuantConfig.kBlockSize>;
    using OutputQuant =
        QuantSpec<Dims::kQuantConfig.kSharedQuant, kTileShapeM / 2u>;

    using MmaElement = typename WeightQuant::ElementType;
    using ElementScale = typename WeightQuant::ScaleType;
    using TileShape =
        cute::Shape<cute::Int<kTileShapeM>, cute::Int<kTileShapeN>,
                    cute::Int<kTileShapeK>>;
    using MmaCore = moe::UmmaCore<WeightQuant, InputQuant, float, TileShape>;
    using TiledMma = typename MmaCore::TiledMma;
    using ScaleTraits = typename MmaCore::ScaleTraits;

    static constexpr std::uint32_t kNumTilesK = kGemmK / kTileShapeK;

    // A K tile spans this many 128-K staging blocks.
    static constexpr std::uint32_t kNumStagingBlocks = kTileShapeK / 128u;

    // The output's (the down's input) scale block.
    static constexpr std::uint32_t kOutputQuantBlock = OutputQuant::kBlockSize;

    static constexpr std::uint32_t kNumAccSlots =
        MmaCore::kUseBlockscaled ? 1 : Dims::KernelConfig::kNumAccSlots;
    static constexpr std::uint32_t kNumWeightSlots =
        Dims::KernelConfig::kNumSharedUpWeightSlots;
    static_assert(kNumWeightSlots <= kNumTilesK / kNumSplits,
                  "the weight ring must fit the block's K tiles");
    static constexpr std::uint32_t kWeightScaleSlots =
        MmaCore::kUseBlockscaled ? 1u : kNumWeightSlots;
    static constexpr auto kWeightScaleShape =
        WeightQuant::template ScaleShape<kTileShapeM, kTileShapeK>();
};

// The shared expert's down-projection: M hidden, N tokens, K intermediate.
template <typename Dims>
struct SharedDownGemmTraits {
    static constexpr std::uint32_t kGemmM = Dims::kHiddenSize;
    static constexpr std::uint32_t kGemmN = Dims::kNumTokens;
    static constexpr std::uint32_t kGemmK = Dims::kIntermediateSize;

    static constexpr std::uint32_t kNumCtas =
        Dims::KernelConfig::kNumCtasForShared;
    static_assert(kGemmM % kNumCtas == 0u,
                  "the blocks must tile the hidden width evenly");

    static constexpr std::uint32_t kTileShapeM = kGemmM / kNumCtas;
    static constexpr std::uint32_t kTileShapeN = kGemmN;
    static constexpr std::uint32_t kTileShapeK = 128u;

    using WeightQuant = QuantSpec<Dims::kQuantConfig.kSharedQuant>;
    using InputQuant = SharedUpGemmTraits<Dims>::OutputQuant;

    using MmaElement = typename WeightQuant::ElementType;
    using ElementScale = typename WeightQuant::ScaleType;
    using TileShape =
        cute::Shape<cute::Int<kTileShapeM>, cute::Int<kTileShapeN>,
                    cute::Int<kTileShapeK>>;
    using MmaCore = moe::UmmaCore<WeightQuant, InputQuant, float, TileShape>;
    using TiledMma = typename MmaCore::TiledMma;
    using ScaleTraits = typename MmaCore::ScaleTraits;

    static constexpr std::uint32_t kNumTilesK = kGemmK / kTileShapeK;

    static constexpr std::uint32_t kNumAccSlots =
        MmaCore::kUseBlockscaled ? 1u : kNumTilesK;
    static constexpr std::uint32_t kNumWeightSlots =
        Dims::KernelConfig::kNumSharedDownWeightSlots;
    static constexpr std::uint32_t kWeightScaleSlots =
        MmaCore::kUseBlockscaled ? 1u : kNumWeightSlots;
    static constexpr auto kWeightScaleShape =
        WeightQuant::template ScaleShape<kTileShapeM, kTileShapeK>();
};

// The accumulator columns one K step's tile needs in TMEM.
template <class GemmTraits>
constexpr std::uint32_t get_gemm_acc_columns() {
    return GemmTraits::MmaCore::kUseBlockscaled
               ? GemmTraits::MmaCore::kNumAtomMs * GemmTraits::kTileShapeN
               : GemmTraits::MmaCore::kNumAtomMs *
                     GemmTraits::MmaCore::kNumQuantBlocksB *
                     GemmTraits::kTileShapeN;
}

// One GEMM phase's TMEM columns.
template <class GemmTraits>
constexpr std::uint32_t get_gemm_tmem_columns() {
    return get_gemm_acc_columns<GemmTraits>() * GemmTraits::kNumAccSlots +
           GemmTraits::kWeightScaleSlots *
               (GemmTraits::ScaleTraits::kSfaTmemColumns +
                GemmTraits::ScaleTraits::kSfbTmemColumns);
}

// Round TMEM columns to an allocation size and check the limit.
template <std::uint32_t kNeededColumns>
constexpr std::uint32_t get_tmem_columns() {
    static_assert(kNeededColumns <= 512u,
                  "the TMEM allocation must fit 512 columns.");
    return kNeededColumns <= 32u
               ? 32u
               : (kNeededColumns <= 64u
                      ? 64u
                      : (kNeededColumns <= 128u
                             ? 128u
                             : (kNeededColumns <= 256u ? 256u : 512u)));
}

template <typename Dims>
struct MoEKernelTraits {
    using MoEDims = Dims;
    using RoutedUpGemm = RoutedUpGemmTraits<Dims>;
    using RoutedDownGemm = RoutedDownGemmTraits<Dims>;
    using RouterGemm = RouterGemmTraits<Dims>;
    using LatentDownGemm = LatentDownGemmTraits<Dims>;
    using SharedUpGemm = SharedUpGemmTraits<Dims>;
    using SharedDownGemm = SharedDownGemmTraits<Dims>;
    using TailGeometry = moe::TailGeometryT<Dims>;

    // Latent output and routed-up input must use the same quantization recipe.
    static_assert(std::is_same_v<typename LatentDownGemm::OutputQuant,
                                 typename RoutedUpGemm::InputQuant>,
                  "the latent output recipe must be the up's input recipe");

    static constexpr std::uint32_t kThreads = Dims::KernelConfig::kBlockSize;
    static constexpr std::uint32_t kCtasPerExpert =
        Dims::KernelConfig::kCtasPerExpert;

    static constexpr bool kUseLatent = Dims::kUseLatent;

    static constexpr std::uint32_t kLatentClusterSize =
        kUseLatent ? LatentDownGemm::kClusterSize : 1u;
    static constexpr std::uint32_t kClusterSize =
        cute::max(kLatentClusterSize, SharedUpGemmTraits<Dims>::kClusterSize);

    // A cluster may not mix blocks of different phases.
    static_assert(RouterGemm::kNumCtas % kClusterSize == 0u,
                  "the router's blocks must end on a cluster boundary");
    static_assert(LatentDownGemm::kNumCtas % kClusterSize == 0u,
                  "the projection's blocks must be whole clusters");
    static_assert(LatentDownGemm::kNumCtas % kLatentClusterSize == 0u,
                  "the projection's groups must be whole blocks");

    static constexpr std::uint32_t kMinGrid = cute::max(
        RouterGemm::kNumCtas + (kUseLatent ? LatentDownGemm::kNumCtas : 0u) +
            cute::max(SharedUpGemm::kNumCtas, SharedDownGemm::kNumCtas),
        TailGeometry::kNumCtas);

    static constexpr std::uint32_t kMaxGrid = kSpecMaxGrid;
    static_assert(kMinGrid <= kMaxGrid,
                  "Grid bounds must be ordered kCtasPerExpert <= kMinGrid <= "
                  "kMaxGrid.");

    // Gate/up rows are interleaved in groups of kPairLayoutGranularity.
    static constexpr std::uint32_t kPairLayoutGranularity = 8u;
    static_assert(RoutedUpGemm::kTileShapeM % (2u * kPairLayoutGranularity) ==
                      0u,
                  "The pair-layout interleave period must divide the A tile's "
                  "row count.");

    static constexpr std::uint32_t kUpSlotsPerDownSlot =
        (RoutedDownGemm::kTileShapeM * RoutedDownGemm::kTileShapeK) /
        (RoutedUpGemm::kTileShapeM * RoutedUpGemm::kTileShapeK);

    static constexpr std::uint32_t kUpTmemColumns =
        get_gemm_tmem_columns<RoutedUpGemm>();
    static constexpr std::uint32_t kDownTmemColumns =
        get_gemm_tmem_columns<RoutedDownGemm>();
    static constexpr std::uint32_t kSharedUpTmemColumns =
        get_gemm_tmem_columns<SharedUpGemmTraits<Dims>>();
    static constexpr std::uint32_t kSharedDownTmemColumns =
        get_gemm_tmem_columns<SharedDownGemmTraits<Dims>>();
    static constexpr std::uint32_t kMaxGemmTmemColumns =
        cute::max(cute::max(kUpTmemColumns, kDownTmemColumns),
                  cute::max(kSharedUpTmemColumns, kSharedDownTmemColumns));
    // tcgen05.alloc takes a power-of-two column count in [32, 512].
    static constexpr std::uint32_t kTmemColumns =
        get_tmem_columns<kMaxGemmTmemColumns>();
};

}  // namespace halops::moe
