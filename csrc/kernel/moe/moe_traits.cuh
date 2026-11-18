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

#include <cstdint>
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
template <std::uint32_t GemmM, std::uint32_t GemmN, std::uint32_t TileShapeN,
          std::uint32_t NumCtas, bool UseUmma>
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

    // K tile: two 128-K quantization blocks, one weight-ring slot. The
    // reduction width itself comes from the runtime problem size.
    static constexpr std::uint32_t kTileShapeK = 256u;

    using TileShape =
        cute::Shape<cute::Int<kTileShapeM>, cute::Int<kTileShapeN>,
                    cute::Int<kTileShapeK>>;
};

// Latent projection: N = tokens. Its M (the latent width) and K (the hidden
// width) come from the runtime problem size; one split owns a fixed 128-row
// M tile.
template <typename Dims>
struct LatentDownGemmTraits {
    static constexpr std::uint32_t kGemmN = Dims::kNumTokens;

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
    static constexpr std::uint32_t kTileShapeN = Dims::kNumTokens;

    static constexpr bool kUseSplitK = Dims::KernelConfig::kLatentDownUseSplitK;
    static_assert(kUseSplitK, "the latent down projection is split-K only");
    // The token width is one N tile, so the splits divide M only.
    static constexpr std::uint32_t kNumSplits =
        Dims::KernelConfig::kLatentDownSplits;

    static constexpr std::uint32_t kTileShapeM = 128u;
    static constexpr std::uint32_t kTileShapeK = 256u;
    using TileShape =
        cute::Shape<cute::Int<kTileShapeM>, cute::Int<kTileShapeN>,
                    cute::Int<kTileShapeK>>;
    // The CTA grid decides how many 128-row M tiles one split owns; the
    // runtime latent width must match it.
    static constexpr std::uint32_t kNumTilesM = kNumCtas / kNumSplits;

    static constexpr std::uint32_t kClusterSize = cute::max(
        (OutputQuant::kBlockSize + kTileShapeM - 1) / kTileShapeM, kNumSplits);

    using MmaCore = moe::UmmaCore<WeightQuant, InputQuant, float, TileShape>;

    static constexpr auto kWeightScaleShape =
        WeightQuant::template ScaleShape<kTileShapeM, kTileShapeK>();
};

// Router Linear (logits = input x weight, bf16 in, fp32 accumulate, bf16 out).
template <typename Dims>
struct RouterGemmTraits {
    static constexpr std::uint32_t kGemmM = Dims::kNumExperts;
    static constexpr std::uint32_t kGemmN = Dims::kNumTokens;

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

    using Gemm =
        GemmTileShape<kGemmM, kGemmN, kTileShapeN, kNumCtasForRouter, kUseUmma>;
    static constexpr std::uint32_t kTileShapeM = Gemm::kTileShapeM;
    static constexpr std::uint32_t kTileShapeK = Gemm::kTileShapeK;
    static constexpr std::uint32_t kNumTilesN = Gemm::kNumTilesN;
    using TileShape = typename Gemm::TileShape;

    using WeightQuant = QuantSpec<QuantMethod::kNone>;
    using InputQuant = QuantSpec<QuantMethod::kNone>;
    using MmaElement = typename WeightQuant::ElementType;

    using MmaCore = cute::conditional_t<
        kUseUmma, moe::UmmaCore<WeightQuant, InputQuant, float, TileShape>,
        moe::HmmaCore<TileShape>>;
};

template <typename Dims>
struct RoutedUpGemmTraits {
    static constexpr std::uint32_t kGemmN = Dims::kNumTokens;

    static constexpr std::uint32_t kCtasPerExpert =
        Dims::KernelConfig::kCtasPerExpert;

    static constexpr std::uint32_t kTileShapeM = 128u;
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

    static constexpr std::uint32_t kNumSplits = 1u;

    // The output's (the down's input) scale block.
    static constexpr std::uint32_t kOutputQuantBlock = OutputQuant::kBlockSize;

    static constexpr auto kWeightScaleShape =
        WeightQuant::template ScaleShape<kTileShapeM, kTileShapeK>();
};

template <typename Dims>
struct RoutedDownGemmTraits {
    static constexpr std::uint32_t kGemmN = Dims::kNumTokens;

    static constexpr bool kUseLatent = Dims::kUseLatent;
    static constexpr std::uint32_t kCtasPerExpert =
        Dims::KernelConfig::kCtasPerExpert;

    static constexpr std::uint32_t kTileShapeM = 256u;
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

    static constexpr auto kWeightScaleShape =
        WeightQuant::template ScaleShape<kTileShapeM, kTileShapeK>();
};

// Shared-expert up projection over all tokens.
template <typename Dims>
struct SharedUpGemmTraits {
    static constexpr std::uint32_t kGemmN = Dims::kNumTokens;

    // One block owns one 128-row M atom of the interleaved rows; the row count
    // comes from the runtime problem size.
    static constexpr std::uint32_t kTileShapeM = 128u;
    static constexpr std::uint32_t kTileShapeN = kGemmN;
    static constexpr std::uint32_t kTileShapeK = 256u;

    static constexpr std::uint32_t kNumSplits =
        Dims::KernelConfig::kSharedUpSplits;
    static constexpr std::uint32_t kNumCtas =
        Dims::KernelConfig::kNumCtasForShared;
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

    // The output's (the down's input) scale block.
    static constexpr std::uint32_t kOutputQuantBlock = OutputQuant::kBlockSize;

    static constexpr auto kWeightScaleShape =
        WeightQuant::template ScaleShape<kTileShapeM, kTileShapeK>();
};

// The shared expert's down-projection: M hidden, N tokens, K intermediate.
template <typename Dims>
struct SharedDownGemmTraits {
    static constexpr std::uint32_t kGemmN = Dims::kNumTokens;

    static constexpr std::uint32_t kNumCtas =
        Dims::KernelConfig::kNumCtasForShared;
    // One block owns one 128-row M tile; the runtime hidden width must cover
    // the blocks exactly (checked on the host).
    static constexpr std::uint32_t kTileShapeM = 128u;
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

    static constexpr std::uint32_t kMaxGrid = Dims::KernelConfig::kSpecMaxGrid;
    static_assert(kMinGrid <= kMaxGrid,
                  "Grid bounds must be ordered kCtasPerExpert <= kMinGrid <= "
                  "kMaxGrid.");

    // Gate/up rows are interleaved in groups of kInterleaveGranularity.
    static constexpr std::uint32_t kInterleaveGranularity = 8u;
    static_assert(RoutedUpGemm::kTileShapeM % (2u * kInterleaveGranularity) ==
                      0u,
                  "The interleave period must divide the A tile's row "
                  "count.");

    static constexpr std::uint32_t kUpSlotsPerDownSlot =
        (RoutedDownGemm::kTileShapeM * RoutedDownGemm::kTileShapeK) /
        (RoutedUpGemm::kTileShapeM * RoutedUpGemm::kTileShapeK);
};

}  // namespace halops::moe
