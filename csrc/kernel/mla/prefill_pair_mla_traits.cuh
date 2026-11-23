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
#include <cute/arch/copy_sm90_desc.hpp>

#include "kernel/mla/mla_traits.cuh"

namespace halops::mla::prefill_pair {

// Kernel constants for model dims Dims (only 64 heads x 512 dims).
template <class Dims_ = ModelDims>
struct PairMlaTraits {
    using Dims = Dims_;

    static constexpr int kHeads = Dims::Heads;
    static constexpr int kDim = Dims::DimQ;
    static constexpr int kNope = Dims::DimNope;
    static constexpr int kRowBytes = Dims::KvRowBytes;
    static constexpr int kSfBytes = KvCacheLayout::ScaleSlotBytes;
    static constexpr int kBlockDims = KvCacheLayout::NopeQuantBlock;
    static constexpr int kBf16Mant = 7;
    static constexpr int kQRopeDims = kDim - kNope;
    // CTA c holds dims [kHalfDims * c, +kHalfDims) of every key.
    static constexpr int kHalfDims = kDim / 2;

    static constexpr int kTile = 64;
    static constexpr int kHalf = kTile / 2;
    static constexpr int kLocalBlocks = kHalfDims / kBlockDims;
    static constexpr int kRowPitch = Sm100::Sw128Bytes;
    static constexpr int kAtomBytes = Sm100::Sw128AtomBytes;
    static constexpr int kAtomRows = kAtomBytes / kRowPitch;
    static constexpr int kChunk = 16;
    static constexpr int kElemsPerChunk = kChunk / sizeof(__nv_bfloat16);
    static constexpr int kBlockBytes = kTile * kRowPitch;
    static constexpr int kStageBytes = kLocalBlocks * kBlockBytes;
    static constexpr int kGather = Sm100::Gather4Rows;

    static constexpr int kUmmaK = 16;
    static constexpr int kKStepBytes = kUmmaK * sizeof(__nv_bfloat16);
    static constexpr int kKStepsPerBlock = kBlockDims / kUmmaK;
    static constexpr int kQkSteps = kHalfDims / kUmmaK;
    static constexpr int kPvSteps = kTile / kUmmaK;
    static constexpr int kPvN = Sm100::UmmaMaxN;
    static constexpr int kPvMmas = kDim / kPvN;
    static constexpr int kPvBlocksPerMma = kLocalBlocks / kPvMmas;

    static constexpr int kKvStages = 4;
    // Merge ring: KV-stage boxes a split pair's parts stream through.
    static constexpr int kMergeBuf = 4;
    static constexpr int kIdxStages = 8;
    static constexpr int kIndexDepth = 4;

    static constexpr int kWarp = KernelBase::WarpSize;
    static constexpr int kWarpGroup = KernelBase::ThreadsPerWarpGroup;
    static constexpr int kWarpsPerGroup = KernelBase::WarpsPerWarpGroup;
    static constexpr int kIndexWarps = kTile / kWarp;
    static constexpr int kGroupsPerWarp = kWarp / kGather;
    static constexpr int kThreads = 4 * kWarpGroup;
    static constexpr int kQkWarp = 12, kPvWarp = 13, kIndexWarp = 14,
                         kAllocWarp = 15;
    static constexpr int kSoftmaxRegs = 152, kQoRegs = 200, kDequantRegs = 104,
                         kAuxRegs = 56;

    static constexpr int kTmemCols = Sm100::TmemColumns;
    static constexpr int kTmemO = 0;
    static constexpr int kTmemQ = kHalfDims;
    static constexpr int kTmemS = kTmemQ + kHalfDims / 2;
    static constexpr int kTmemSCols = kTile;
    static constexpr int kTmemLd = 32;

    static constexpr uint16_t kBothCtas = 0b11;
    // Named barriers: softmax partner warps, dequant and Q/O warp groups.
    static constexpr uint32_t kPairBar = 1, kDequantBar = 3, kQoBar = 4;

    static constexpr float kMaxInit = -1e30f;
    // Lazy rescale: O is rescaled when a row max grows by more than 2^6.
    static constexpr float kRescaleLog2 = 6.0f;

    // Raw bytes per key row gathered by CTA cta (CTA 1 has the bf16 rope).
    CUTE_HOST_DEVICE static constexpr int raw_pitch(int cta) {
        return cta == 0 ? kHalfDims : kRowBytes - kHalfDims;
    }
    static constexpr int kNopeBytes1 = kNope - kHalfDims;

    // Q and O boxes: 64 dims x the 64 heads of a token, 128-byte swizzled.
    static constexpr int kQBoxDims = 64;
    static constexpr int kQBoxBytes =
        kHeads * kQBoxDims * sizeof(__nv_bfloat16);
    static constexpr int kChunksPerBox =
        kQBoxDims * sizeof(__nv_bfloat16) / kChunk;
    static constexpr int kQoRounds = kHalfDims / kQBoxDims;
    static_assert(kQoRounds == 4, "fetch_q2: two Q buffers, rounds 0..3");
    // A Q/O thread's Q slice (one TMEM lane), in 16-byte vectors.
    static constexpr int kQVecs =
        kHalfDims * sizeof(__nv_bfloat16) / sizeof(uint4);
    // O leaves through qo_stage by TMA, a round of 64 dims per box.
    static constexpr int kRoundWords = kQBoxDims / 2;
    // kQnr: lanes 0..15 of warp 1 copy a unit's 64 RoPE floats into sm.qfreq.
    static constexpr int kFreqLanes = kQRopeDims * sizeof(float) / kChunk;
    static constexpr int kFreqT0 = kWarp;

    // o_part plane p holds columns [4p, 4p + 4) of every lane.
    static constexpr int kPlaneCols = 4;
    static constexpr int kPlanes = kHalfDims / kPlaneCols;
    static constexpr int kPlaneBytes = kWarpGroup * kPlaneCols * sizeof(float);
    static constexpr int kPartBoxPlanes = 16;  // planes per 32 KB bulk copy
    static constexpr int kPartBoxes = kPlanes / kPartBoxPlanes;
    // TMEM columns of O released to the next unit's first PV at a time.
    static constexpr int kOHalf = kPvN / 2;
    static constexpr int kMergeCols = 32;  // TMEM columns per step of the merge
    // Added by a merger that finds a part missing; the last part merges.
    static constexpr int kDeferred = 1 << 16;
    // Combine kernel: warp groups per block.
    static constexpr int kCombSplit = 4;

    static constexpr uint64_t kKvL2Hint =
        uint64_t(cute::TMA::CacheHintSm90::EVICT_FIRST);
    // The bf16 c128 rows are read by every pair of a request: evict-normal.
    static constexpr uint64_t kBf16L2Hint =
        uint64_t(cute::TMA::CacheHintSm90::EVICT_NORMAL);

    static_assert(kHeads == 64 && 2 * kHeads == kWarpGroup,
                  "a warp group's TMEM lanes hold 64 heads x 2 dims halves");
    static_assert(Dims::DimV == kDim && kDim == 512,
                  "TMEM holds O (256 columns), Q (128) and S (2 x 64)");
    static_assert(kTmemS + 2 * kTmemSCols == kTmemCols);
    static_assert(kQRopeDims == kBlockDims && kNope > kHalfDims,
                  "CTA 1 holds the rope dims in its last block");
    static_assert(kRowBytes == kNope + kQRopeDims * sizeof(__nv_bfloat16));
    static_assert(kMergeBuf <= kKvStages);
    static_assert(kSoftmaxRegs + kQoRegs + kDequantRegs + kAuxRegs ==
                  4 * kWarpGroup);
};

// Persistent schedule: tile stream split over the clusters.
struct PairSchedTraits {
    // Pair lengths of a range kept in shared memory.
    static constexpr int kTableUnits = 32;
    static constexpr int kMaxClusters = 128;  // scheduled clusters (SMs / 2)
    static constexpr int kDefaultTpc = 3;     // tiles per cluster when combine
    // Pairs of at most this many tiles run the kQst kernels.
    static constexpr int kQstMaxTiles = 5;
    // Schedule cost of an n-tile pair in quarter tiles: 4 n plus a fixed cost.
    static constexpr int kPairFixQst = 24;
    static constexpr int kPairFix = 24;
    static constexpr int kTileCost = 4;
};

// Pre-dequant (predq_kernel) limits by ld[0], token count and bytes.
struct PredqTraits {
    static constexpr int kThreads = 256;
    static constexpr int kWarps = kThreads / KernelBase::WarpSize;
    static constexpr int64_t kMaxBytes = int64_t(512) << 20;
    static constexpr int kMinLd = 512, kMinTokens = 256;
    static constexpr int kShortLd = 256, kShortTokens = 4096;
};

}  // namespace halops::mla::prefill_pair
