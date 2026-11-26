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

#include <cstdint>
#include <cute/config.hpp>
#include <cute/util/type_traits.hpp>

#include "common/sm100_arch.cuh"
#include "kernel/mla/common/intrinsics.cuh"
#include "kernel/mla/pair_mla/common/config.cuh"
#include "kernel/mla/pair_mla/decode/smem.cuh"

namespace halops {
using namespace mla;
}  // namespace halops

namespace halops::detail {

// Head geometry shared by the HCA and DSA formats and the combine kernel.
// kBlockDims is one SW128 row of bf16, 64 dims: the unit of the K-major smem
// layout, of a UMMA K block, and of the HCA UE8M0 scale, so the HCA format
// fixes its scale block to it instead of taking a separate parameter.
template <int kHeads_, int kDim_>
struct HeadLayout {
    static constexpr int kHeads = kHeads_;
    static constexpr int kDim = kDim_;
    static constexpr int kHalfDims = kDim / 2;
    static constexpr int kBlockDims =
        kSw128AtomRowBytes / sizeof(__nv_bfloat16);
    static constexpr int kLocalBlocks = kHalfDims / kBlockDims;
    static_assert(kHeads == 64,
                  "the 128 TMEM lanes of a CTA pair hold heads 0-63 for each "
                  "half of the key tile (UMMA M = 2 * kHeads = 128)");
    static_assert(kHalfDims % kBlockDims == 0,
                  "each CTA holds whole 64-dim blocks");
};

// PV MMA width and the fp32 split-pair partial planes over head layout L.
template <class L>
struct HcaPart {
    static constexpr int kPvN = 256;
    static constexpr int kPvMmas = L::kDim / kPvN;
    static constexpr int kPvBlocksPerMma = L::kLocalBlocks / kPvMmas;
    static constexpr int kOHalf = kPvN / 2;
    static constexpr int kPlaneCols = 4;
    static constexpr int kPlanes = L::kHalfDims / kPlaneCols;
    static constexpr int kPlaneBytes = kWarpGroup * kPlaneCols * sizeof(float);
    static constexpr int kPartBoxPlanes = 16;
    static constexpr int kPartBoxes = kPlanes / kPartBoxPlanes;
    static_assert(L::kDim % kPvN == 0, "PV covers kDim in N = 256 MMAs");
    static_assert(L::kLocalBlocks % kPvMmas == 0,
                  "every PV MMA reads whole 64-dim blocks");
};

// Combine-kernel config: the head layout plus its partial-plane layout.
template <class L>
struct HcaCombineCfg : L, HcaPart<L> {};

// HCA KV row: kNope fp8 dims, then the bf16 rope dims, then kSfBytes UE8M0
// scales per row stored after the page's rows; one scale per kBlockDims dims.
template <int kHeads_, int kDim_, int kNope_, int kRowBytes_, int kSfBytes_>
struct HcaFormat : HeadLayout<kHeads_, kDim_> {
    using Layout = HeadLayout<kHeads_, kDim_>;
    static constexpr bool kDsa = false;
    static constexpr int kNope = kNope_;
    static constexpr int kRowBytes = kRowBytes_;
    static constexpr int kSfBytes = kSfBytes_;
    static constexpr int kNopeBytes1 = kNope - Layout::kHalfDims;
    static_assert(kRowBytes == kNope + 2 * (kDim_ - kNope),
                  "a key row is kNope fp8 bytes followed by bf16 rope dims");
    static_assert(kNope % Layout::kBlockDims == 0 &&
                      kNope / Layout::kBlockDims <= kSfBytes,
                  "one UE8M0 scale byte per 64-dim block of the fp8 dims");
    static_assert(kNope > Layout::kHalfDims &&
                      kDim_ - kNope == Layout::kBlockDims,
                  "CTA 0 gathers fp8 dims only; CTA 1 gathers the rest of "
                  "the fp8 dims and the rope dims, which fill its last block");
};

// DSA KV row: kNope fp8 dims, kNope / kScaleGroup fp32 scales, then kRope
// bf16 rope dims; the value is the nope part, so kNope == kDim.
template <int kHeads_, int kDim_, int kNope_, int kRope_, int kScaleGroup_>
struct DsaFormat : HeadLayout<kHeads_, kDim_> {
    using Layout = HeadLayout<kHeads_, kDim_>;
    static constexpr bool kDsa = true;
    static constexpr int kDsaNope = kNope_;
    static constexpr int kDsaRope = kRope_;
    static constexpr int kDsaScaleGroup = kScaleGroup_;
    static constexpr int kDsaQk = kDsaNope + kDsaRope;
    static constexpr int kDsaScales = kDsaNope / kDsaScaleGroup;
    static constexpr int kDsaScaleBytes = kDsaScales * sizeof(float);
    static constexpr int kDsaRopeBytes = kDsaRope * sizeof(__nv_bfloat16);
    static constexpr int kDsaRowBytes =
        kDsaNope + kDsaScaleBytes + kDsaRopeBytes;
    static constexpr int kDsaNopeHalf = kDsaNope / 2;
    static constexpr int kDsaScaleOff = kDsaNope;
    static constexpr int kDsaRopeOff = kDsaNope + kDsaScaleBytes;
    static constexpr int kDsaQRopeBytes =
        kHeads_ * kDsaRope * sizeof(__nv_bfloat16);
    static constexpr int kDsaRopeSteps = kDsaRope / kUmmaK;
    static_assert(kDsaNope == kDim_,
                  "the value is the nope part; each CTA dequantizes "
                  "kDsaNope / 2 = kHalfDims dims");
    static_assert(kDsaNope % kDsaScaleGroup == 0 &&
                      kDsaScaleGroup % kChunk == 0,
                  "every 16-byte fp8 chunk takes a single scale");
    static_assert(kDsaScales / 2 * sizeof(float) == 8,
                  "each CTA's scales are one 8-byte cp.async");
    static_assert(kDsaRopeBytes == kSw128AtomRowBytes,
                  "the rope K and Q tiles are one SW128 row per key or head");
};

// Tile, pipeline, warp-role and TMEM layout of one decode kernel build over a
// KV format.  kFuse compiles in the fused merge; the launches that cannot use
// it run a build without it, whose main loop the merge code would slow down.
template <class Fmt, bool kFuseV>
struct HcaTraits : Fmt, HcaPart<typename Fmt::Layout> {
    static constexpr bool kFuse = kFuseV;
    static constexpr int kHcaTile = 64;
    static constexpr int kDsaTile = 64;
    static constexpr int kTile = Fmt::kDsa ? kDsaTile : kHcaTile;
    static constexpr int kHalf = kTile / 2;
    static constexpr int kBlockBytes = kTile * kSw128AtomRowBytes;
    static constexpr int kStageBytes = Fmt::kLocalBlocks * kBlockBytes;
    static constexpr int kKStepsPerBlock = Fmt::kBlockDims / kUmmaK;
    static constexpr int kQkSteps = Fmt::kHalfDims / kUmmaK;
    static constexpr int kPvSteps = kTile / kUmmaK;
    static constexpr int kKvStages = 5;
    static constexpr int kDsaKvStages = 4;
    static constexpr int kStages = Fmt::kDsa ? kDsaKvStages : kKvStages;
    static constexpr int kMergeBuf = 4;
    static constexpr int kMergeShared = 1 << 16;
    static constexpr int kTableUnits = 32;
    static constexpr int kIdxStages = 8;
    static constexpr int kIndexDepth = 4;
    static constexpr int kIndexWarps = kTile / kWarp;
    static constexpr int kGroupsPerWarp = kWarp / kTmaGatherRows;
    static constexpr int kThreads = 4 * kWarpGroup;
    static constexpr int kQkWarp = 12, kPvWarp = 13, kIndexWarp = 14,
                         kAllocWarp = 15;
    static constexpr int kSoftmaxRegs = 152, kQoRegs = 200, kDequantRegs = 104,
                         kAuxRegs = 56;
    static constexpr int kTmemO = 0;
    static constexpr int kTmemQ = Fmt::kHalfDims;
    static constexpr int kTmemS = kTmemQ + Fmt::kHalfDims / 2;
    static constexpr int kTmemSCols = kTile;
    static constexpr uint16_t kBothCtas = 0b11;
    static constexpr uint32_t kPairBar = 1, kDequantBar = 3, kQoBar = 4;
    static constexpr float kMaxInit = -1e30f;
    static constexpr float kRescaleLog2 = 6.0f;
    static constexpr int kQBoxDims = 64;
    static constexpr int kQBoxBytes =
        Fmt::kHeads * kQBoxDims * sizeof(__nv_bfloat16);
    static constexpr int kChunksPerBox =
        kQBoxDims * sizeof(__nv_bfloat16) / kChunk;
    static constexpr int kQoRounds = Fmt::kHalfDims / kQBoxDims;
    static constexpr int kQVecs =
        Fmt::kHalfDims * sizeof(__nv_bfloat16) / sizeof(uint4);
    static constexpr int kMergeCols = 32;
    static constexpr int kRoundWords = kQBoxDims / 2;
    static constexpr int kMergeWgs = 2;
    static constexpr uint32_t kMergeBar = 5;
    static constexpr int kMergeThreads = kMergeWgs * kWarpGroup;
    static constexpr int kKeep = 2;
    static constexpr int kLseRegs = 8;
    static constexpr int kFp8BlockChunks = Fmt::kBlockDims / kChunk;
    static constexpr int kMinMergeParts = 128;
    static constexpr int kMergeLoads = 16;
    static constexpr int kMergeKMid = 4;
    static constexpr int kMergeKWide = 8;
    static_assert(kMergeBuf <= kStages,
                  "the fused merge stages its part boxes in the KV stages");
    static_assert(kSoftmaxRegs + kQoRegs + kDequantRegs + kAuxRegs == 4 * 128,
                  "the four warp groups split the 64K registers of the SM: "
                  "the per-thread budgets must sum to 512");
    static_assert(kTmemO + Fmt::kHalfDims <= kTmemQ &&
                      kTmemS + 2 * kTmemSCols <= kTmemCols,
                  "O, Q and the two S buffers must fit the 512 TMEM columns");
};

// Compile-time variant selector.  An HCA format reproduces HCA byte-for-byte
// (every DSA-only statement lives under `if constexpr (C::kDsa)`); a DSA
// format selects the DSA smem and the inline-scale row path.  Every
// tile-derived constant is a member so the shared code reads C::kX.
template <class Fmt, bool kFuseV = !Fmt::kDsa>
struct Cfg : HcaTraits<Fmt, kFuseV> {
    using Format = Fmt;
    using Traits = HcaTraits<Fmt, kFuseV>;
    using Smem =
        cute::conditional_t<Fmt::kDsa, DsaSmem<Traits>, HcaSmem<Traits>>;
    static constexpr int kSmemBytes = sizeof(Smem) + kSw128AtomBytes;
    static_assert(kSmemBytes <= kSmemCapacity,
                  "shared memory must fit the 227 KiB opt-in limit of sm_100");
};

using HcaModelFormat = HcaFormat<64, 512, 448, 576, 8>;
using DsaModelFormat = DsaFormat<64, 512, 512, 64, 128>;
using HcaCfg = Cfg<HcaModelFormat>;
using DsaCfg = Cfg<DsaModelFormat>;

// Raw bytes per key row gathered by a CTA: the fp8 dims [0, kHalfDims) for
// CTA 0, the remaining fp8 dims and the bf16 rope dims for CTA 1.  The rows
// land at the start of the KV stage and are dequantized in place.
template <class C>
CUTE_HOST_DEVICE constexpr int raw_pitch(int cta) {
    return cta == 0 ? C::kHalfDims : C::kRowBytes - C::kHalfDims;
}

// Bit fields and widths of the PTX operands built by hca_ptx.cuh.
struct HcaIsa {
    static constexpr int kDescShift = 4;
    static constexpr int kInstrDescShift = 32;
    static constexpr int kTmemLaneShift = 16;
    static constexpr int kStoreWords = 8;
    static constexpr int kStoreElems = 16;
};

}  // namespace halops::detail
