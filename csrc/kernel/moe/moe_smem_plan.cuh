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
#include <cute/tensor.hpp>

#include "kernel/moe/moe_configs.cuh"
#include "kernel/moe/moe_traits.cuh"

namespace halops::moe {

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

// The quantize's fp8 output, which is also the MMA's B operand.
template <std::uint32_t NumSlots, std::uint32_t TileShapeN,
          std::uint32_t TileShapeK, std::uint32_t QuantBlock,
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

    // MXFP8 uses per-slot SfAtoms; blockwise keeps one tile's K blocks per
    // slot, each row as wide as the activation tile.
    static constexpr std::uint32_t kSfAtomBytes =
        128u * (TileShapeK / Quant::kBlockSize);
    static constexpr std::uint32_t kBlockRowsPerSlot =
        TileShapeK / Quant::kBlockSize;
    using ScaleArray = std::conditional_t<
        Method == QuantMethod::kMXFP8, std::uint8_t[NumSlots][kSfAtomBytes],
        ElementScale[NumSlots][kBlockRowsPerSlot * TileShapeN]>;

    alignas(1024) ElementType
        data[kNumSlots][kNumLboChunksK][TileShapeN + 1u][kLboChunkK];
    alignas(128) ScaleArray scale;

    __device__ ElementScale* scale_ptr() {
        return reinterpret_cast<ElementScale*>(&scale);
    }
};

template <class GemmTraits, std::uint32_t NumSlots,
          bool UseBlockscaled = GemmTraits::MmaCore::kUseBlockscaled>
struct WeightScaleStorageT;

// Blockwise scale tables follow the GEMM's prefetch granularity.
template <class GemmTraits, std::uint32_t NumSlots>
struct WeightScaleStorageT<GemmTraits, NumSlots, false> {
    using Quant = typename GemmTraits::WeightQuant;
    static_assert(Quant::kMethod == QuantMethod::kBlockwiseFP8,
                  "the blockwise storage requires FP32 block scales");
    static constexpr std::uint32_t kNumSlots = NumSlots;
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
template <class GemmTraits, std::uint32_t NumSlots>
struct WeightScaleStorageT<GemmTraits, NumSlots, true> {
    static constexpr std::uint32_t kNumSlots = NumSlots;
    static constexpr std::uint32_t kScaleCols =
        GemmTraits::MmaCore::ScaleTraits::kSfaSmemBytes;
    alignas(128) std::uint8_t scale[kNumSlots][kScaleCols];

    __device__ auto& operator[](std::uint32_t slot) { return scale[slot]; }
    __device__ const auto& operator[](std::uint32_t slot) const {
        return scale[slot];
    }
};

template <class GemmTraits, std::uint32_t NumSlots,
          bool UseBlockscaled = GemmTraits::MmaCore::kUseBlockscaled>
struct ActivationScaleStorageT;

// Blockwise down rings one K tile's activation scale rows per input slot.
template <class GemmTraits, std::uint32_t NumSlots>
struct ActivationScaleStorageT<GemmTraits, NumSlots, false> {
    using Quant = typename GemmTraits::InputQuant;
    static constexpr std::uint32_t kNumSlots = NumSlots;
    static constexpr std::uint32_t kBlocksPerTile =
        GemmTraits::kTileShapeK / Quant::kBlockSize;
    alignas(128) typename Quant::ScaleType
        scale[kNumSlots][kBlocksPerTile][GemmTraits::kTileShapeN];

    __device__ auto& operator[](std::uint32_t slot) { return scale[slot]; }
    __device__ const auto& operator[](std::uint32_t slot) const {
        return scale[slot];
    }
};

// MX stages each activation tile's SfAtom in the corresponding input slot.
template <class GemmTraits, std::uint32_t NumSlots>
struct ActivationScaleStorageT<GemmTraits, NumSlots, true> {
    static constexpr std::uint32_t kNumSlots = NumSlots;
    static constexpr std::uint32_t kSfAtomBytes =
        GemmTraits::MmaCore::ScaleTraits::kSfbSmemBytes;
    alignas(128) std::uint8_t scale[kNumSlots][kSfAtomBytes];

    __device__ auto& operator[](std::uint32_t slot) { return scale[slot]; }
    __device__ const auto& operator[](std::uint32_t slot) const {
        return scale[slot];
    }
};

template <class GemmTraits, std::uint32_t kNumSlots>
struct RouterSmemT {
    ActivationSmem<kNumSlots, GemmTraits::kTileShapeN, GemmTraits::kTileShapeK>
        input;

    alignas(1024) typename GemmTraits::MmaElement
        w_router[kNumSlots][GemmTraits::MmaCore::kWeightTileElems];

    static constexpr std::uint32_t kNumMmaWarps =
        GemmTraits::kUseUmma ? 1u : WarpRole::kCalcWarpCount;
    static constexpr std::uint32_t kMmaRows =
        GemmTraits::kUseUmma ? 1u : GemmTraits::kTileShapeN;
    static constexpr std::uint32_t kMmaCols =
        GemmTraits::kUseUmma ? 1u : GemmTraits::kTileShapeM;
    float mma_out[kNumMmaWarps][kMmaRows][kMmaCols];
};

template <class GemmTraits, std::uint32_t kNumSlots,
          std::uint32_t kWeightScaleSlots>
struct LatentDownSmemT {
    static constexpr bool kUseBlockscaled =
        GemmTraits::MmaCore::kUseBlockscaled;

    union {
        alignas(1024) typename GemmTraits::MmaElement
            w_latent_down[kNumSlots][GemmTraits::MmaCore::kNumSwizzleBoxKs]
                         [GemmTraits::kTileShapeM]
                         [GemmTraits::MmaCore::kSwizzleBoxK];
        alignas(16) float splited_mma_out[GemmTraits::kTileShapeM]
                                         [GemmTraits::kTileShapeN];
    };

    ActivationSmem<kNumSlots, GemmTraits::kTileShapeN, GemmTraits::kTileShapeK>
        input;
    QuantizedActivationSmem<
        kNumSlots, GemmTraits::kTileShapeN, GemmTraits::kTileShapeK,
        GemmTraits::InputQuant::kBlockSize, GemmTraits::InputQuant::kMethod>
        quantized_input;

    WeightScaleStorageT<GemmTraits, kWeightScaleSlots, kUseBlockscaled> w_scale;

    static constexpr std::uint32_t kOutLboChunkK =
        LboChunkK<typename GemmTraits::OutputQuant::ElementType>::value;
    static constexpr std::uint32_t kNumOutChunks =
        GemmTraits::kTileShapeM / kOutLboChunkK;
    alignas(16) DataType::ElementA
        act_out[kNumOutChunks][GemmTraits::kTileShapeN][kOutLboChunkK];
};

// The shared expert's own storage, in the union with the other phases.
template <class UpGemmTraits, class DownGemmTraits, std::uint32_t kNumSlots>
struct SharedExpertSmemT {
    static constexpr bool kUseBlockscaled =
        UpGemmTraits::MmaCore::kUseBlockscaled;
    static_assert(kUseBlockscaled == DownGemmTraits::MmaCore::kUseBlockscaled,
                  "the shared expert runs one quantization recipe");

    union {
        alignas(1024) typename UpGemmTraits::MmaElement
            w_up[kNumSlots][2u][UpGemmTraits::kTileShapeM][128u];
        alignas(1024) typename DownGemmTraits::MmaElement
            w_down[kNumSlots][DownGemmTraits::kTileShapeM]
                  [DownGemmTraits::kTileShapeK];
        alignas(1024) float splited_mma_out[2u][UpGemmTraits::kTileShapeM / 2u]
                                           [UpGemmTraits::kTileShapeN];
    };

    ActivationSmem<kNumSlots, UpGemmTraits::kTileShapeN,
                   UpGemmTraits::kTileShapeK>
        input;

    alignas(1024) QuantizedActivationSmem<
        kNumSlots, UpGemmTraits::kTileShapeN, UpGemmTraits::kTileShapeK,
        UpGemmTraits::InputQuant::kBlockSize,
        UpGemmTraits::InputQuant::kMethod> quantized_input;

    alignas(1024) QuantizedActivationSmem<
        kNumSlots, DownGemmTraits::kTileShapeN, DownGemmTraits::kTileShapeK,
        DownGemmTraits::InputQuant::kBlockSize,
        DownGemmTraits::InputQuant::kMethod> act_down_input;

    alignas(1024) WeightScaleStorageT<UpGemmTraits, kNumSlots,
                                      kUseBlockscaled> w_up_scale;
    alignas(1024) WeightScaleStorageT<DownGemmTraits, kNumSlots,
                                      kUseBlockscaled> w_down_scale;

    alignas(16) float act_up_out[UpGemmTraits::kTileShapeM / 2u]
                                [UpGemmTraits::kTileShapeN + 1u];
};

template <class UpGemmTraits, class DownGemmTraits, std::uint32_t kUpSlots,
          std::uint32_t kDownSlots>
struct RoutedExpertsSmemT {
    static constexpr bool kUseBlockscaled =
        UpGemmTraits::MmaCore::kUseBlockscaled;
    static_assert(kUseBlockscaled == DownGemmTraits::MmaCore::kUseBlockscaled,
                  "the routed experts run one quantization recipe");

    // A blockwise weight scale ring keeps the kernel config's prefetch depth.
    static constexpr std::uint32_t kUpScaleSlots =
        kUseBlockscaled ? kUpSlots : 2u;
    static constexpr std::uint32_t kDownScaleSlots =
        kUseBlockscaled ? kDownSlots : 1u;

    union {
        alignas(1024) typename UpGemmTraits::MmaElement
            w_up[kUpSlots][UpGemmTraits::MmaCore::kNumSwizzleBoxKs]
                [UpGemmTraits::kTileShapeM *
                 UpGemmTraits::MmaCore::kSwizzleBoxK];

        alignas(1024) typename DownGemmTraits::MmaElement
            w_down[kDownSlots][DownGemmTraits::kTileShapeM]
                  [DownGemmTraits::kTileShapeK];
    };

    float act_up_out[UpGemmTraits::kTileShapeM / 2u]
                    [UpGemmTraits::kTileShapeN + 1u];

    // The input and its MX scale slabs stream through the up weight ring's
    // slots.
    QuantizedActivationSmem<
        kUpSlots, UpGemmTraits::kTileShapeN, UpGemmTraits::kTileShapeK,
        UpGemmTraits::InputQuant::kBlockSize, UpGemmTraits::InputQuant::kMethod>
        quantized_input;

    alignas(1024) typename DownGemmTraits::InputQuant::ElementType
        act_down_input[kDownSlots][DownGemmTraits::kTileShapeN *
                                   DownGemmTraits::kTileShapeK];

    alignas(1024) WeightScaleStorageT<UpGemmTraits, kUpScaleSlots,
                                      kUseBlockscaled> w_up_scale;
    alignas(1024) WeightScaleStorageT<DownGemmTraits, kDownScaleSlots,
                                      kUseBlockscaled> w_down_scale;

    ActivationScaleStorageT<DownGemmTraits, kDownSlots, kUseBlockscaled>
        act_down_scale;
};

// Generic ring-depth plan: a phase's rings all share one depth, the largest
// power of two whose laid-out storage fits the phase budget.
template <template <std::uint32_t> class Layout, std::uint32_t kCap = 0u>
struct SlotsPlan {
    static constexpr std::uint32_t kTotalSmemBytes = 224u * 1024u;
    static constexpr std::uint32_t kReserveSmemBytes = 8u * 1024u;
    static constexpr std::uint32_t kPhaseSmemBytes =
        kTotalSmemBytes - kReserveSmemBytes;

    template <std::uint32_t kDepth = 2u>
    static constexpr std::uint32_t find_largest_depth() {
        static_assert(sizeof(Layout<2u>) <= kPhaseSmemBytes,
                      "the phase cannot afford double-buffered rings");
        if constexpr (kCap != 0u && kDepth >= kCap) {
            return kDepth;
        } else if constexpr (sizeof(Layout<kDepth + 1u>) <= kPhaseSmemBytes) {
            return find_largest_depth<kDepth + 1u>();
        } else {
            return kDepth;
        }
    }

    template <std::uint32_t kDepth>
    static constexpr std::uint32_t floor_power_of_two() {
        static_assert(kDepth >= 1u, "a ring depth is at least one slot");
        std::uint32_t depth = 1u;
        while (depth <= kDepth / 2u) {
            depth *= 2u;
        }
        return depth;
    }

    static constexpr std::uint32_t kNumSlots =
        floor_power_of_two<find_largest_depth()>();
};

// A down weight slot covers this many up weight slots.
template <typename Dims>
constexpr std::uint32_t up_slots_per_down_slot() {
    using Up = RoutedUpGemmTraits<Dims>;
    using Down = RoutedDownGemmTraits<Dims>;
    static_assert((Down::kTileShapeM * Down::kTileShapeK) >=
                          (Up::kTileShapeM * Up::kTileShapeK) &&
                      (Down::kTileShapeM * Down::kTileShapeK) %
                              (Up::kTileShapeM * Up::kTileShapeK) ==
                          0u,
                  "a down weight slot must cover whole up weight slots");
    return (Down::kTileShapeM * Down::kTileShapeK) /
           (Up::kTileShapeM * Up::kTileShapeK);
}

// A GEMM's ring depth, so the generic mainloop bases stay independent of
// which phase they serve.
template <typename Dims, class GemmTraits>
struct GemmSlots;

// The latent down phase's weight, input and scale rings share one depth.
template <typename Dims>
struct GemmSlots<Dims, LatentDownGemmTraits<Dims>> {
    template <std::uint32_t kNumSlots>
    using Layout =
        LatentDownSmemT<LatentDownGemmTraits<Dims>, kNumSlots, kNumSlots>;
    static constexpr std::uint32_t kNumSlots = SlotsPlan<Layout>::kNumSlots;
    static constexpr std::uint32_t kScaleSlots = kNumSlots;
    // The blockscaled latent phase refills one SFA/SFB set per K tile.
    static constexpr std::uint32_t kTmemScaleSlots =
        LatentDownGemmTraits<Dims>::MmaCore::kUseBlockscaled ? 1u : 0u;
    // A blockscaled phase's single accumulator is reused; the blockwise
    // pipeline keeps one per ring slot.
    static constexpr std::uint32_t kTmemAccSlots =
        LatentDownGemmTraits<Dims>::MmaCore::kUseBlockscaled ? 1u : kNumSlots;
};

// The router's weight and input rings share one depth; it keeps no scale
// ring and no TMEM accumulator ring.
template <typename Dims>
struct GemmSlots<Dims, RouterGemmTraits<Dims>> {
    template <std::uint32_t kNumSlots>
    using Layout = RouterSmemT<RouterGemmTraits<Dims>, kNumSlots>;
    static constexpr std::uint32_t kNumSlots = SlotsPlan<Layout>::kNumSlots;
    static constexpr std::uint32_t kScaleSlots = 0u;
    static constexpr std::uint32_t kTmemAccSlots = 0u;
    static constexpr std::uint32_t kTmemScaleSlots = 0u;
};

// The routed experts' up and down rings share one plan: down gets the depth
// `d` and up gets `up_slots_per_down_slot<Dims>() * d`, so the down ring
// aliases whole groups of up weight slots.
template <typename Dims>
struct GemmSlots<Dims, RoutedDownGemmTraits<Dims>> {
    using Up = RoutedUpGemmTraits<Dims>;
    using Down = RoutedDownGemmTraits<Dims>;
    static constexpr bool kUseBlockscaled = Down::MmaCore::kUseBlockscaled;

    template <std::uint32_t kDownSlots>
    using Layout = RoutedExpertsSmemT<
        Up, Down, up_slots_per_down_slot<Dims>() * kDownSlots, kDownSlots>;
    static constexpr std::uint32_t kNumSlots = SlotsPlan<Layout>::kNumSlots;
    static constexpr std::uint32_t kScaleSlots =
        kUseBlockscaled ? kNumSlots : 1u;
    // The blockscaled down dequant trails its MMA, so every in-flight weight
    // slot keeps its own SFA/SFB set in TMEM.
    static constexpr std::uint32_t kTmemScaleSlots =
        kUseBlockscaled ? kNumSlots : 0u;
    static constexpr std::uint32_t kTmemAccSlots =
        kUseBlockscaled ? 1u : kNumSlots;
};

template <typename Dims>
struct GemmSlots<Dims, RoutedUpGemmTraits<Dims>> {
    using Up = RoutedUpGemmTraits<Dims>;
    static constexpr std::uint32_t kNumSlots =
        up_slots_per_down_slot<Dims>() *
        GemmSlots<Dims, RoutedDownGemmTraits<Dims>>::kNumSlots;
    static constexpr std::uint32_t kScaleSlots =
        Up::MmaCore::kUseBlockscaled ? kNumSlots : 2u;
    static constexpr std::uint32_t kTmemScaleSlots =
        Up::MmaCore::kUseBlockscaled ? 2u : 0u;
    static constexpr std::uint32_t kTmemAccSlots =
        Up::MmaCore::kUseBlockscaled ? 2u : kNumSlots;
};

// The shared expert's up and down rings share one depth.
template <typename Dims>
struct GemmSlots<Dims, SharedUpGemmTraits<Dims>> {
    using Up = SharedUpGemmTraits<Dims>;
    using Down = SharedDownGemmTraits<Dims>;
    template <std::uint32_t kNumSlots>
    using Layout = SharedExpertSmemT<Up, Down, kNumSlots>;
    static constexpr std::uint32_t kNumSlots = SlotsPlan<Layout>::kNumSlots;
    static constexpr std::uint32_t kScaleSlots = kNumSlots;
    static constexpr std::uint32_t kTmemScaleSlots =
        Up::MmaCore::kUseBlockscaled ? 1u : 0u;
    static constexpr std::uint32_t kTmemAccSlots =
        Up::MmaCore::kUseBlockscaled ? 1u : kNumSlots;
};

template <typename Dims>
struct GemmSlots<Dims, SharedDownGemmTraits<Dims>> {
    using Up = SharedUpGemmTraits<Dims>;
    static constexpr std::uint32_t kNumSlots = GemmSlots<Dims, Up>::kNumSlots;
    static constexpr std::uint32_t kScaleSlots =
        GemmSlots<Dims, Up>::kScaleSlots;
    static constexpr std::uint32_t kTmemScaleSlots =
        GemmSlots<Dims, Up>::kTmemScaleSlots;
    static constexpr std::uint32_t kTmemAccSlots =
        GemmSlots<Dims, Up>::kTmemAccSlots;
};

// The phase storage, with the planned ring depths.
template <typename Dims, class GemmTraits>
using RouterSmem =
    RouterSmemT<GemmTraits, GemmSlots<Dims, GemmTraits>::kNumSlots>;

template <typename Dims, class GemmTraits>
using LatentDownSmem =
    LatentDownSmemT<GemmTraits, GemmSlots<Dims, GemmTraits>::kNumSlots,
                    GemmSlots<Dims, GemmTraits>::kScaleSlots>;

template <typename Dims, class UpGemmTraits, class DownGemmTraits>
using SharedExpertSmem =
    SharedExpertSmemT<UpGemmTraits, DownGemmTraits,
                      GemmSlots<Dims, UpGemmTraits>::kNumSlots>;

template <typename Dims, class UpGemmTraits, class DownGemmTraits>
using RoutedExpertsSmem =
    RoutedExpertsSmemT<UpGemmTraits, DownGemmTraits,
                       GemmSlots<Dims, UpGemmTraits>::kNumSlots,
                       GemmSlots<Dims, DownGemmTraits>::kNumSlots>;

// One GEMM phase's TMEM columns: its accumulators plus its SFA/SFB sets.
template <typename Dims, class GemmTraits>
constexpr std::uint32_t get_gemm_tmem_columns() {
    using ScaleTraits = typename GemmTraits::MmaCore::ScaleTraits;
    return get_gemm_acc_columns<GemmTraits>() *
               GemmSlots<Dims, GemmTraits>::kTmemAccSlots +
           GemmSlots<Dims, GemmTraits>::kTmemScaleSlots *
               (ScaleTraits::kSfaTmemColumns + ScaleTraits::kSfbTmemColumns);
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

// The block's resource plan: the rings above and the TMEM columns they feed.
template <typename Dims>
struct MoETmemPlan {
    using LatentDown = LatentDownGemmTraits<Dims>;
    using RoutedUp = RoutedUpGemmTraits<Dims>;
    using RoutedDown = RoutedDownGemmTraits<Dims>;
    using SharedUp = SharedUpGemmTraits<Dims>;
    using SharedDown = SharedDownGemmTraits<Dims>;

    static constexpr std::uint32_t kLatentTmemColumns =
        get_gemm_tmem_columns<Dims, LatentDown>();
    static constexpr std::uint32_t kUpTmemColumns =
        get_gemm_tmem_columns<Dims, RoutedUp>();
    static constexpr std::uint32_t kDownTmemColumns =
        get_gemm_tmem_columns<Dims, RoutedDown>();
    static constexpr std::uint32_t kSharedUpTmemColumns =
        get_gemm_tmem_columns<Dims, SharedUp>();
    static constexpr std::uint32_t kSharedDownTmemColumns =
        get_gemm_tmem_columns<Dims, SharedDown>();
    static constexpr std::uint32_t kMaxGemmTmemColumns =
        cute::max(cute::max(cute::max(kLatentTmemColumns, kUpTmemColumns),
                            kDownTmemColumns),
                  cute::max(kSharedUpTmemColumns, kSharedDownTmemColumns));
    // tcgen05.alloc takes a power-of-two column count in [32, 512].
    static constexpr std::uint32_t kTmemColumns =
        get_tmem_columns<kMaxGemmTmemColumns>();
};

}  // namespace halops::moe
