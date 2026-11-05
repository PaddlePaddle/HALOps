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

#include "kernel/moe/mma_core.cuh"
#include "kernel/moe/moe_configs.cuh"
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

// Latent down-projection: M = latent size, N = tokens, K = hidden, fp8 in
// and out, fp32 accumulate.
template <typename Dims>
struct LatentDownGemmTraits {
    static constexpr std::uint32_t kGemmM = Dims::kLatentSize;
    static constexpr std::uint32_t kGemmN = Dims::kNumTokens;
    static constexpr std::uint32_t kGemmK = Dims::kHiddenSize;

    // MXFP8 latent: UE8M0 scales, one per 32-K block.  The weight scales use
    // the SfAtom layout and the blockscaled UMMA instruction.
    using WeightQuant = QuantSpec<Dims::kQuantConfig.kLatentQuant>;
    using ElementScale = typename WeightQuant::ScaleType;

    static constexpr std::uint32_t kNumCtas =
        Dims::KernelConfig::kNumCtasForLatentDown;
    static constexpr std::uint32_t kTileShapeN = 8u;
    static constexpr std::uint32_t kNumTilesN = kGemmN / kTileShapeN;
    static_assert(kGemmN % kTileShapeN == 0u,
                  "the token width must be whole N tiles");

    // The reduction splits over a cluster when `kLatentDownUseSplitK` is on.
    static constexpr bool kUseSplitK = Dims::KernelConfig::kLatentDownUseSplitK;

    // Split-K pins the M tile at 128 rows, so one split group covers one M
    // tile per N tile; the shape helper is fed that group's CTA count so its
    // UMMA M asserts see the real tile.
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
    static constexpr std::uint32_t kTilesKPerCta = kNumTilesK / kNumSplits;
    static_assert(kNumTilesK % kNumSplits == 0u,
                  "the K tiles must split evenly over the groups");

    // A K tile spans this many 128-K quantization blocks.
    static constexpr std::uint32_t kQuantBlocksPerTile =
        kTileShapeK / Dims::kQuantConfig.kBlockSize;

    // The cluster is one 128-latent block's blocks of one token window.
    static constexpr std::uint32_t kClusterSize =
        (Dims::kQuantConfig.kBlockSize + kTileShapeM - 1) / kTileShapeM;
    static_assert(Dims::kQuantConfig.kBlockSize % kTileShapeM == 0u,
                  "a CTA's rows must sit inside one 128-row block");
    static_assert(kNumTilesM % kClusterSize == 0u,
                  "a cluster must not straddle a token window");
    static_assert(kClusterSize <= 8u,
                  "the max exchange's cluster must be a portable one");

    // The 128-latent weight-scale blocks.
    static constexpr std::uint32_t kNumScaleBlocksM =
        kGemmM / Dims::kQuantConfig.kBlockSize;
    static constexpr std::uint32_t kNumScaleBlocksK =
        kGemmK / Dims::kQuantConfig.kBlockSize;
    static_assert(kNumCtas % kNumScaleBlocksM == 0u,
                  "the CTAs must cover whole 128-row scale blocks");

    using ActQuant = QuantSpec<Dims::kQuantConfig.kLatentQuant>;
    using MmaElement = typename WeightQuant::ElementType;
    using MmaCore = moe::UmmaCore<WeightQuant, ActQuant, float, TileShape>;

    // The weight ring is sized in bytes, not slots.
    static constexpr std::uint32_t kWeightRingSlots =
        (128u * 1024u) / (kQuantBlocksPerTile * kTileShapeM * 128u);
    static constexpr std::uint32_t kNumWeightSlots =
        Dims::KernelConfig::kNumLatentDownWeightSlots < kWeightRingSlots
            ? Dims::KernelConfig::kNumLatentDownWeightSlots
            : kWeightRingSlots;
    static constexpr std::uint32_t kNumWeightScaleSlots = kNumWeightSlots;
    static constexpr auto kWeightScaleShape =
        WeightQuant::template ScaleShape<kTileShapeM, kTileShapeK>();
};

// Router Linear (logits = input x weight, bf16 in, fp32 accumulate, bf16 out).
template <typename Dims>
struct RouterGemmTraits {
    static constexpr std::uint32_t kGemmM = Dims::kNumExperts;
    static constexpr std::uint32_t kGemmN = Dims::kNumTokens;
    static constexpr std::uint32_t kGemmK = Dims::kHiddenSize;

    static constexpr std::uint32_t kNumCtasForRouter =
        Dims::KernelConfig::kNumCtasForRouter;
    static_assert(kNumCtasForRouter >= 1u,
                  "the router needs at least one CTA.");

    static constexpr bool kUseUmma =
        Dims::KernelConfig::kRouterMma == MmaKind::kUmma;

    static constexpr std::uint32_t kNumRouterGates = Dims::kNumRouterGates;
    static_assert(kNumRouterGates == 1u || kNumRouterGates == 2u,
                  "the router runs one or two gates");
    static constexpr std::uint32_t kNumCtas =
        kNumRouterGates * kNumCtasForRouter;

    static constexpr std::uint32_t kTileShapeN = 8u;
    static_assert(Dims::kNumTokens % 8u == 0u,
                  "the token width must be whole 8-token windows");

    using Gemm = GemmTileShape<kGemmM, kGemmN, kGemmK, kTileShapeN,
                               kNumCtasForRouter, kUseUmma>;
    static constexpr std::uint32_t kTileShapeM = Gemm::kTileShapeM;
    static constexpr std::uint32_t kTileShapeK = Gemm::kTileShapeK;
    static constexpr std::uint32_t kNumTilesN = Gemm::kNumTilesN;
    static constexpr std::uint32_t kNumTilesM = Gemm::kNumTilesM;
    static constexpr std::uint32_t kNumTilesK = Gemm::kNumTilesK;
    using TileShape = typename Gemm::TileShape;

    static constexpr std::uint32_t kNumWeightSlots =
        Dims::KernelConfig::kNumRouterWeightSlots < kNumTilesK
            ? Dims::KernelConfig::kNumRouterWeightSlots
            : kNumTilesK;

    using WeightQuant = QuantSpec<QuantMethod::kNone>;
    using ActQuant = QuantSpec<QuantMethod::kNone>;
    using MmaElement = typename WeightQuant::ElementType;
    static constexpr std::uint32_t kSwizzleBoxK =
        SwizzleBoxK<MmaElement>::value;

    using MmaCore = cute::conditional_t<
        kUseUmma, moe::UmmaCore<WeightQuant, ActQuant, float, TileShape>,
        moe::HmmaCore<TileShape>>;
    static_assert(MmaCore::kSwizzleBoxK == kSwizzleBoxK,
                  "the MMA must read the packed SW128 boxes");
    static_assert(kTileShapeK % kSwizzleBoxK == 0u,
                  "a router K tile must be whole SW128 boxes");
    static constexpr std::uint32_t kNumSwizzleBoxKs =
        kTileShapeK / kSwizzleBoxK;

    // A warp owns `kNumTilesKPerWarp` K tiles; the last warps may own fewer.
    static constexpr std::uint32_t kNumTilesKPerWarp =
        (kNumTilesK + WarpRole::kCalcWarpCount - 1u) / WarpRole::kCalcWarpCount;
};

template <typename Dims>
struct UpGemmTraits {
    static constexpr std::uint32_t kGemmM = Dims::kIntermediateSize * 2;
    static constexpr std::uint32_t kGemmN = Dims::kNumTokens;
    static constexpr std::uint32_t kGemmK = MoEInputSize<Dims>::value;

    static constexpr std::uint32_t kCtasPerExpert =
        Dims::KernelConfig::kCtasPerExpert;
    static constexpr bool kUseLatent = Dims::kUseLatent;
    static_assert(kGemmM % kCtasPerExpert == 0u,
                  "kCtasPerExpert must divide kGemmM (kIntermediateSize * 2).");

    static constexpr std::uint32_t kTileShapeM = kGemmM / kCtasPerExpert;
    static constexpr std::uint32_t kTileShapeN = kGemmN;
    static constexpr std::uint32_t kTileShapeK =
        Dims::KernelConfig::kUpTileShapeK;

    using WeightQuant = QuantSpec<Dims::kQuantConfig.kExpertsQuant>;
    using ActQuant = QuantSpec<Dims::kQuantConfig.kExpertsActQuant,
                               Dims::kQuantConfig.kBlockSize>;
    using MmaElement = typename WeightQuant::ElementType;
    using ElementScale = typename WeightQuant::ScaleType;
    using TileShape =
        cute::Shape<cute::Int<kTileShapeM>, cute::Int<kTileShapeN>,
                    cute::Int<kTileShapeK>>;
    using MmaCore = moe::UmmaCore<WeightQuant, ActQuant, float, TileShape>;
    using TiledMma = typename MmaCore::TiledMma;
    using ScaleTraits = typename MmaCore::ScaleTraits;
    // w4a8 stages a 4-bit weight one value per SMEM byte, and its activation
    // stays fp8: the two staging elements are no longer the same type.
    using WeightSmemElement = typename MmaCore::SmemElementA;
    using ActElement = typename ActQuant::ElementType;

    static_assert(kGemmK % kTileShapeK == 0,
                  "kGemmK (the MoE input width) must be a multiple of the "
                  "K tile.");
    static constexpr std::uint32_t kNumTilesK = kGemmK / kTileShapeK;

    // Weight-scale tile: one fp32 per 128x128 block of the [M, K] weight.
    static_assert(kGemmM % Dims::kQuantConfig.kBlockSize == 0u &&
                      kGemmK % Dims::kQuantConfig.kBlockSize == 0u,
                  "the up weight must be whole 128x128 scale blocks.");
    static constexpr std::uint32_t kNumScaleBlocksM =
        kGemmM / Dims::kQuantConfig.kBlockSize;
    static constexpr std::uint32_t kNumScaleBlocksK =
        kGemmK / Dims::kQuantConfig.kBlockSize;

    // The down's activation quantization block: half the up tile.
    static constexpr std::uint32_t kActQuantBlock = kTileShapeM / 2u;

    // Sub-accumulators per K step, one per input quantization block.
    static constexpr std::uint32_t kNumSubBlocks =
        kTileShapeK / Dims::kQuantConfig.kBlockSize;
    static_assert(kTileShapeK % Dims::kQuantConfig.kBlockSize == 0u,
                  "an input quantization block must divide the K tile");
    static constexpr std::uint32_t kAccColumns =
        MmaCore::kUseBlockscaled ? MmaCore::kNumAtomMs * kTileShapeN
                                 : kNumSubBlocks * kTileShapeN;

    static constexpr std::uint32_t kNumWeightSlots =
        Dims::KernelConfig::kNumUpWeightSlots;
    static_assert(
        kNumWeightSlots >= 2u,
        "kNumWeightSlots must be >= 2 (the up-proj weight pipeline needs at "
        "least a ping-pong double buffer).");

    // Blockwise up prefetches the full gate/up scale rows for each task.
    static constexpr std::uint32_t kNumWeightScaleSlots = 2u;
    static constexpr auto kWeightScaleShape =
        cute::make_shape(cute::_2{}, cute::Int<kNumScaleBlocksK>{});
};

template <typename Dims>
struct DownGemmTraits {
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
    using ActQuant = QuantSpec<Dims::kQuantConfig.kExpertsActQuant,
                               UpGemmTraits<Dims>::kActQuantBlock>;
    using MmaElement = typename WeightQuant::ElementType;
    using ElementScale = typename WeightQuant::ScaleType;
    using TileShape =
        cute::Shape<cute::Int<kTileShapeM>, cute::Int<kTileShapeN>,
                    cute::Int<kTileShapeK>>;
    using MmaCore = moe::UmmaCore<WeightQuant, ActQuant, float, TileShape>;
    using TiledMma = typename MmaCore::TiledMma;
    using ScaleTraits = typename MmaCore::ScaleTraits;
    using WeightSmemElement = typename MmaCore::SmemElementA;
    using ActElement = typename ActQuant::ElementType;

    static_assert(kGemmK % kTileShapeK == 0,
                  "kGemmK (kIntermediateSize) must be a multiple of 128.");
    static constexpr std::uint32_t kNumTilesK = kGemmK / kTileShapeK;

    // Weight-scale tile: one fp32 per 128x128 block of the [M, K] weight.
    static_assert(kGemmM % Dims::kQuantConfig.kBlockSize == 0u &&
                      kGemmK % Dims::kQuantConfig.kBlockSize == 0u,
                  "the down weight must be whole 128x128 scale blocks.");
    static constexpr std::uint32_t kNumScaleBlocksM =
        kGemmM / Dims::kQuantConfig.kBlockSize;
    static constexpr std::uint32_t kNumScaleBlocksK =
        kGemmK / Dims::kQuantConfig.kBlockSize;

    static constexpr std::uint32_t kNumWeightSlots =
        Dims::KernelConfig::kNumDownWeightSlots;
    // Blockwise down loads the current M tile's full K scale table per task.
    static constexpr std::uint32_t kNumWeightScaleSlots = 1u;
    static constexpr auto kWeightScaleShape =
        WeightQuant::template ScaleShape<kTileShapeM, kGemmK>();

    // Sub-accumulators per K step: one per activation quantization block.
    static constexpr std::uint32_t kNumSubBlocks =
        kTileShapeK / UpGemmTraits<Dims>::kActQuantBlock;
    static_assert(kTileShapeK % UpGemmTraits<Dims>::kActQuantBlock == 0u,
                  "the activation quantization block must divide the 128-K "
                  "step the down projection accumulates in one go");
    static constexpr std::uint32_t kAccColumns =
        MmaCore::kUseBlockscaled
            ? MmaCore::kNumAtomMs * kTileShapeN
            : MmaCore::kNumAtomMs * kNumSubBlocks * kTileShapeN;
};

// The shared expert's up-projection: one expert every token goes through, in
// the hidden space.
template <typename Dims>
struct SharedUpGemmTraits {
    static constexpr std::uint32_t kGemmM = Dims::kIntermediateSize * 2u;
    static constexpr std::uint32_t kGemmN = Dims::kNumTokens;
    static constexpr std::uint32_t kGemmK = Dims::kHiddenSize;

    // One block owns one M atom of the pair layout.
    static constexpr std::uint32_t kTileShapeM = 128u;
    static constexpr std::uint32_t kTileShapeN = kGemmN;
    static constexpr std::uint32_t kTileShapeK = 256u;
    static_assert(kGemmM % kTileShapeM == 0u,
                  "the slab must tile the pair layout evenly");
    static constexpr std::uint32_t kNumSplits =
        Dims::KernelConfig::kSharedUseSplitK
            ? Dims::KernelConfig::kNumCtasForShared / (kGemmM / kTileShapeM)
            : 1u;
    static_assert(kNumSplits >= 1u, "the reduction must have a share to run");
    static constexpr std::uint32_t kNumCtas =
        kNumSplits * (kGemmM / kTileShapeM);

    using WeightQuant = QuantSpec<Dims::kQuantConfig.kSharedQuant>;
    using ActQuant = QuantSpec<Dims::kQuantConfig.kSharedQuant,
                               Dims::kQuantConfig.kBlockSize>;
    using MmaElement = typename WeightQuant::ElementType;
    using ElementScale = typename WeightQuant::ScaleType;
    using TileShape =
        cute::Shape<cute::Int<kTileShapeM>, cute::Int<kTileShapeN>,
                    cute::Int<kTileShapeK>>;
    using MmaCore = moe::UmmaCore<WeightQuant, ActQuant, float, TileShape>;
    using TiledMma = typename MmaCore::TiledMma;
    using ScaleTraits = typename MmaCore::ScaleTraits;

    static_assert(kGemmK % kTileShapeK == 0u,
                  "kGemmK (kHiddenSize) must be a whole number of K tiles");
    static constexpr std::uint32_t kNumTilesK = kGemmK / kTileShapeK;

    static constexpr std::uint32_t kTilesKPerCta = kNumTilesK / kNumSplits;
    static constexpr std::uint32_t kQuantBlocksPerCta =
        kGemmK / kNumSplits / Dims::kQuantConfig.kBlockSize;
    static_assert(kNumTilesK % kNumSplits == 0u,
                  "the K tiles must split evenly over the groups");
    static_assert(kTilesKPerCta *
                          (kTileShapeK / Dims::kQuantConfig.kBlockSize) ==
                      kQuantBlocksPerCta,
                  "the block's K tiles must be its quantization blocks");

    // The activation's quantization block: the features one block owns.
    static constexpr std::uint32_t kActQuantBlock = kTileShapeM / 2u;

    // Sub-accumulators per K step, one per input quantization block.
    static constexpr std::uint32_t kNumSubBlocks =
        kTileShapeK / Dims::kQuantConfig.kBlockSize;
    static constexpr std::uint32_t kAccColumns =
        MmaCore::kUseBlockscaled ? MmaCore::kNumAtomMs * kTileShapeN
                                 : kNumSubBlocks * kTileShapeN;

    static constexpr std::uint32_t kNumWeightSlots =
        Dims::KernelConfig::kNumSharedUpWeightSlots;
    static_assert(kNumWeightSlots >= 1u,
                  "the shared up's weight ring needs at least one slot");
    static_assert(kNumWeightSlots <= kTilesKPerCta,
                  "the weight ring must fit the block's K tiles");
    static constexpr std::uint32_t kNumWeightScaleSlots = kNumWeightSlots;
    static constexpr auto kWeightScaleShape =
        WeightQuant::template ScaleShape<kTileShapeM, kTileShapeK>();
};

// The shared expert's down-projection: M hidden, N tokens, K intermediate.
template <typename Dims>
struct SharedDownGemmTraits {
    static constexpr std::uint32_t kGemmM = Dims::kHiddenSize;
    static constexpr std::uint32_t kGemmN = Dims::kNumTokens;
    static constexpr std::uint32_t kGemmK = Dims::kIntermediateSize;

    using WeightQuant = QuantSpec<Dims::kQuantConfig.kSharedQuant>;

    static constexpr std::uint32_t kNumCtas =
        Dims::KernelConfig::kNumCtasForShared;
    static_assert(kGemmM % kNumCtas == 0u,
                  "the blocks must tile the hidden width evenly");

    static constexpr std::uint32_t kTileShapeM = kGemmM / kNumCtas;
    static constexpr std::uint32_t kTileShapeN = kGemmN;
    static constexpr std::uint32_t kTileShapeK = 128u;

    using ActQuant = QuantSpec<Dims::kQuantConfig.kSharedQuant,
                               SharedUpGemmTraits<Dims>::kActQuantBlock>;
    using MmaElement = typename WeightQuant::ElementType;
    using ElementScale = typename WeightQuant::ScaleType;
    using TileShape =
        cute::Shape<cute::Int<kTileShapeM>, cute::Int<kTileShapeN>,
                    cute::Int<kTileShapeK>>;
    using MmaCore = moe::UmmaCore<WeightQuant, ActQuant, float, TileShape>;
    using TiledMma = typename MmaCore::TiledMma;
    using ScaleTraits = typename MmaCore::ScaleTraits;

    static_assert(kGemmK % kTileShapeK == 0u,
                  "kGemmK (kIntermediateSize) must be a whole number of K "
                  "tiles");
    static constexpr std::uint32_t kNumTilesK = kGemmK / kTileShapeK;

    // Sub-accumulators per K step, one per activation quantization block.
    static constexpr std::uint32_t kNumSubBlocks =
        kTileShapeK / SharedUpGemmTraits<Dims>::kActQuantBlock;
    static_assert(kTileShapeK % SharedUpGemmTraits<Dims>::kActQuantBlock == 0u,
                  "an activation quantization block must divide the K step the "
                  "down accumulates in one go");
    static constexpr std::uint32_t kAccColumns =
        MmaCore::kUseBlockscaled
            ? MmaCore::kNumAtomMs * kTileShapeN
            : MmaCore::kNumAtomMs * kNumSubBlocks * kTileShapeN;

    static constexpr std::uint32_t kNumWeightSlots =
        Dims::KernelConfig::kNumSharedDownWeightSlots;
    static_assert(kNumWeightSlots >= 1u,
                  "the shared down's weight ring needs at least one slot");
    static constexpr std::uint32_t kNumWeightScaleSlots = kNumWeightSlots;
    static constexpr auto kWeightScaleShape =
        WeightQuant::template ScaleShape<kTileShapeM, kTileShapeK>();
};

template <typename Dims>
struct MoEKernelTraits {
    using MoEDims = Dims;
    using UpGemm = UpGemmTraits<Dims>;
    using DownGemm = DownGemmTraits<Dims>;
    using RouterGemm = RouterGemmTraits<Dims>;
    using LatentDownGemm = LatentDownGemmTraits<Dims>;

    static constexpr std::uint32_t kThreads = Dims::KernelConfig::kBlockSize;
    static constexpr std::uint32_t kCtasPerExpert =
        Dims::KernelConfig::kCtasPerExpert;

    static constexpr bool kUseLatent = Dims::kUseLatent;

    // Whether the projection quantizes its own output.  The output is the up
    // phase's B operand, so it follows that GEMM's activation recipe: the
    // wholesale copy below expects the projection's image of both recipes.
    // The projection quantizes its output in a cluster, for its scale or
    // split reduction.
    static constexpr std::uint32_t kLatentScaleCluster =
        kUseLatent ? LatentDownGemm::kClusterSize : 1u;
    static constexpr std::uint32_t kLatentClusterSize =
        LatentDownGemm::kNumSplits > kLatentScaleCluster
            ? LatentDownGemm::kNumSplits
            : kLatentScaleCluster;

    // The shared expert: one expert every token goes through.
    static constexpr std::uint32_t kSharedClusterSize =
        SharedUpGemmTraits<Dims>::kNumSplits;
    static constexpr std::uint32_t kClusterSize =
        kLatentClusterSize > kSharedClusterSize ? kLatentClusterSize
                                                : kSharedClusterSize;

    // A cluster may not mix blocks of different phases.
    static_assert(RouterGemm::kNumCtas % kClusterSize == 0u,
                  "the router's blocks must end on a cluster boundary");
    static_assert(LatentDownGemm::kNumCtas % kClusterSize == 0u,
                  "the projection's blocks must be whole clusters");
    static_assert(LatentDownGemm::kNumCtas % LatentDownGemm::kClusterSize == 0u,
                  "the projection's groups must be whole blocks");

    static constexpr std::uint32_t kRouterInputSlots =
        Dims::KernelConfig::kNumRouterInputSlots;
    static constexpr std::uint32_t kLatentInputSlots =
        Dims::KernelConfig::kNumLatentInputSlots;

    static constexpr std::uint32_t kNumRouterWeightSlots =
        RouterGemm::kNumWeightSlots;
    static_assert(kNumRouterWeightSlots >= 1u,
                  "the router weight ring needs at least one slot");

    static constexpr std::uint32_t kMinGrid =
        RouterGemm::kNumCtas + (kUseLatent ? 0u : LatentDownGemm::kNumCtas);
    static constexpr std::uint32_t kMaxGrid = kSpecMaxGrid;
    static_assert(kMinGrid <= kMaxGrid,
                  "Grid bounds must be ordered kCtasPerExpert <= kMinGrid <= "
                  "kMaxGrid.");

    // Combine readiness granularity.
    static constexpr bool kCombineWaitPerToken =
        Dims::KernelConfig::kCombineWaitPerToken;

    static constexpr std::uint32_t kNumAccSlots =
        Dims::KernelConfig::kNumAccSlots;
    static_assert(kNumAccSlots >= 2u,
                  "kNumAccSlots must be >= 2 so the dequant warpgroup can lag "
                  "the MMA by one K-iteration.");

    // K per 16-B chunk, the granularity the MMA's SMEM descriptor walks.
    static constexpr std::uint32_t kLboChunkK = kFp8ActChunkK;
    static constexpr std::uint32_t kFp8ActChunksPerBlock =
        Dims::kQuantConfig.kBlockSize / kLboChunkK;
    static constexpr std::uint32_t kQuantBlocksPerTile =
        UpGemm::kTileShapeK / Dims::kQuantConfig.kBlockSize;

    // Gate/up pair layout: the Python repack interleaves gate and up in
    // groups of kPairLayoutGranularity rows.
    static constexpr std::uint32_t kPairLayoutGranularity = 8u;
    static_assert(UpGemm::kTileShapeM % (2u * kPairLayoutGranularity) == 0u,
                  "The pair-layout interleave period must divide the A tile's "
                  "row count.");

    static constexpr std::uint32_t kUpSlotsPerDownSlot =
        (DownGemm::kTileShapeM * DownGemm::kTileShapeK) /
        (UpGemm::kTileShapeM * UpGemm::kTileShapeK);

    // The down scale ring follows the weight slots: each K tile's MMA
    // completion releases its operands and their scale storage together.
    static constexpr std::uint32_t kDownScaleSlots = DownGemm::kNumWeightSlots;

    // SM100 TMEM allocation, shared by every phase of the block.
    // Blockscaled up and down each use one fixed accumulator. Down keeps one
    // SFA/SFB set per weight slot.
    static constexpr std::uint32_t kUpSfColumns =
        UpGemm::MmaCore::kUseBlockscaled
            ? UpGemm::ScaleTraits::kSfaTmemColumns +
                  UpGemm::ScaleTraits::kSfbTmemColumns
            : 0u;
    static constexpr std::uint32_t kDownSfColumns =
        DownGemm::MmaCore::kUseBlockscaled
            ? DownGemm::ScaleTraits::kSfaTmemColumns +
                  DownGemm::ScaleTraits::kSfbTmemColumns
            : 0u;
    static constexpr std::uint32_t kRingUpTmemColumns =
        UpGemm::kAccColumns *
            (UpGemm::MmaCore::kUseBlockscaled ? 1u : kNumAccSlots) +
        kUpSfColumns;
    static constexpr std::uint32_t kRingDownTmemColumns =
        DownGemm::kAccColumns *
            (DownGemm::MmaCore::kUseBlockscaled ? 1u : kNumAccSlots) +
        kDownScaleSlots * kDownSfColumns;
    static constexpr std::uint32_t kRingTmemColumns =
        kRingUpTmemColumns > kRingDownTmemColumns ? kRingUpTmemColumns
                                                  : kRingDownTmemColumns;
    // The blockscaled phases stage one SFA/SFB set beside their accumulators.
    static constexpr std::uint32_t kSharedUpSfColumns =
        SharedUpGemmTraits<Dims>::MmaCore::kUseBlockscaled
            ? SharedUpGemmTraits<Dims>::ScaleTraits::kSfaTmemColumns +
                  SharedUpGemmTraits<Dims>::ScaleTraits::kSfbTmemColumns
            : 0u;
    static constexpr std::uint32_t kSharedDownSfColumns =
        SharedDownGemmTraits<Dims>::MmaCore::kUseBlockscaled
            ? SharedDownGemmTraits<Dims>::ScaleTraits::kSfaTmemColumns +
                  SharedDownGemmTraits<Dims>::ScaleTraits::kSfbTmemColumns
            : 0u;
    static constexpr std::uint32_t kSharedUpTmemColumns =
        (SharedUpGemmTraits<Dims>::MmaCore::kUseBlockscaled ? 1u
                                                            : kNumAccSlots) *
            SharedUpGemmTraits<Dims>::kAccColumns +
        kSharedUpSfColumns;
    static constexpr std::uint32_t kSharedDownTmemColumns =
        (SharedDownGemmTraits<Dims>::MmaCore::kUseBlockscaled
             ? 1u
             : SharedDownGemmTraits<Dims>::kNumTilesK) *
            SharedDownGemmTraits<Dims>::kAccColumns +
        kSharedDownSfColumns;
    static constexpr std::uint32_t kTmemNeededColumns =
        kRingTmemColumns > kSharedUpTmemColumns
            ? (kRingTmemColumns > kSharedDownTmemColumns
                   ? kRingTmemColumns
                   : kSharedDownTmemColumns)
            : (kSharedUpTmemColumns > kSharedDownTmemColumns
                   ? kSharedUpTmemColumns
                   : kSharedDownTmemColumns);
    // tcgen05.alloc takes a power-of-two column count in [32, 512].
    static constexpr std::uint32_t kTmemColumns =
        kTmemNeededColumns <= 32u
            ? 32u
            : (kTmemNeededColumns <= 64u
                   ? 64u
                   : (kTmemNeededColumns <= 128u
                          ? 128u
                          : (kTmemNeededColumns <= 256u ? 256u : 512u)));
    static_assert(kTmemNeededColumns <= 512u,
                  "the TMEM allocation must fit 512 columns.");
    static_assert(RouterGemm::kTileShapeN <= kTmemColumns,
                  "the router accumulator must fit the TMEM allocation");
};

}  // namespace halops::moe
