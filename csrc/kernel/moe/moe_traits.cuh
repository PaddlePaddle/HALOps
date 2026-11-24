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

// Latent projection: N = tokens, M = latent width and K = hidden width.
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

    static constexpr std::uint32_t kTileShapeN = kGemmN;
    static constexpr std::uint32_t kTileShapeM = 128u;
    static constexpr std::uint32_t kTileShapeK = 256u;
    using TileShape =
        cute::Shape<cute::Int<kTileShapeM>, cute::Int<kTileShapeN>,
                    cute::Int<kTileShapeK>>;

    using MmaCore = moe::UmmaCore<WeightQuant, InputQuant, float, TileShape>;

    static constexpr auto kWeightScaleShape =
        WeightQuant::template ScaleShape<kTileShapeM, kTileShapeK>();
};

// Router projection with BF16 inputs and FP32 accumulation.
template <typename Dims>
struct RouterGemmTraits {
    static constexpr std::uint32_t kGemmM = Dims::kNumExperts;
    static constexpr std::uint32_t kGemmN = Dims::kNumTokens;

    static constexpr bool kStoreScores = MoERuntime::kMaxRouterGates > 1u;
    using LogitType = std::conditional_t<kStoreScores, float, __nv_bfloat16>;

    static constexpr bool kUseUmma =
        Dims::KernelConfig::kRouterMma == MmaKind::kUmma;
    static constexpr std::uint32_t kNumCtasForRouter =
        Dims::KernelConfig::kNumCtasForRouter;

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

    static constexpr std::uint32_t kTileShapeM = 128u;
    static constexpr std::uint32_t kTileShapeN = kGemmN;
    static constexpr std::uint32_t kTileShapeK = 256u;

    using WeightQuant = QuantSpec<Dims::kQuantConfig.kExpertsQuant>;
    using InputQuant = QuantSpec<Dims::kQuantConfig.kExpertsActQuant,
                                 Dims::kQuantConfig.kBlockSize>;
    using OutputQuant =
        QuantSpec<Dims::kQuantConfig.kExpertsActQuant, kTileShapeM / 2u>;
    static constexpr std::uint32_t kOutputQuantBlock = OutputQuant::kBlockSize;

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

template <typename Dims>
struct RoutedDownGemmTraits {
    static constexpr std::uint32_t kGemmN = Dims::kNumTokens;

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

    static constexpr std::uint32_t kTileShapeM = 128u;
    static constexpr std::uint32_t kTileShapeN = kGemmN;
    static constexpr std::uint32_t kTileShapeK = 256u;

    using WeightQuant = QuantSpec<Dims::kQuantConfig.kSharedQuant>;
    using InputQuant = QuantSpec<Dims::kQuantConfig.kSharedQuant,
                                 Dims::kQuantConfig.kBlockSize>;
    using OutputQuant =
        QuantSpec<Dims::kQuantConfig.kSharedQuant, kTileShapeM / 2u>;
    static constexpr std::uint32_t kOutputQuantBlock = OutputQuant::kBlockSize;

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

// The shared expert's down-projection: M hidden, N tokens, K intermediate.
template <typename Dims>
struct SharedDownGemmTraits {
    static constexpr std::uint32_t kGemmN = Dims::kNumTokens;

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

    static constexpr std::uint32_t kMaxGrid = Dims::KernelConfig::kSpecMaxGrid;

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
