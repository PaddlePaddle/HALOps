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
#include <cute/arch/mma_sm100_desc.hpp>
#include <cute/arch/mma_sm100_umma.hpp>

#include "kernel/mla/common/traits.cuh"

namespace halops::mla {

// Kernel constants of the pair prefill kernels for model dims Dims (only
// 64 heads x 512 dims). CTA c of a 2-CTA cluster holds dims
// [kHalfDims * c, +kHalfDims) of every key and the Q / O of token c of a pair.
template <class Dims_ = ModelDims>
struct PrefillPairTraits {
    using Dims = Dims_;

    static constexpr int kHeads = Dims::Heads;
    static constexpr int kDim = Dims::DimQ;
    static constexpr int kNope = Dims::DimNope;
    static constexpr int kRowBytes = Dims::KvRowBytes;
    static constexpr int kSfBytes = KvCacheLayout::ScaleSlotBytes;
    static constexpr int kBlockDims = KvCacheLayout::NopeQuantBlock;
    static constexpr int kBf16Mant = 7;
    static constexpr int kQRopeDims = kDim - kNope;
    static constexpr int kHalfDims = kDim / 2;

    static constexpr int kTile = 64;
    static constexpr int kHalf = kTile / 2;
    static constexpr int kLocalBlocks = kHalfDims / kBlockDims;
    static constexpr int kRowPitch = Sm100::Sw128Bytes;
    static constexpr int kAtomBytes = Sm100::Sw128AtomBytes;
    static constexpr int kAtomRows = kAtomBytes / kRowPitch;
    static constexpr int kChunk = 16;
    static constexpr int kChunkMask = kChunk - 1;
    static constexpr int kElemsPerChunk = kChunk / sizeof(__nv_bfloat16);
    static constexpr int kFp8ChunksPerBlock = kBlockDims / kChunk;
    static constexpr int kVecLen = sizeof(float4) / sizeof(float);
    static constexpr int kVecW = kVecLen - 1;
    static constexpr int kIdxPerVec = sizeof(int4) / sizeof(int);
    static constexpr int kIdxVecMask = kIdxPerVec - 1;
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
    static constexpr int kTmemLaneShift = 16;

    static constexpr uint16_t kBothCtas = 0b11;
    static constexpr uint32_t kPairBar = 1, kDequantBar = 3, kQoBar = 4;

    static constexpr float kMaxInit = -1e30f;
    static constexpr float kRescaleLog2 = 6.0f;

    // Raw bytes per key row gathered by CTA cta (CTA 1 has the bf16 rope).
    CUTE_HOST_DEVICE static constexpr int raw_pitch(int cta) {
        return cta == 0 ? kHalfDims : kRowBytes - kHalfDims;
    }
    static constexpr int kNopeBytes1 = kNope - kHalfDims;

    static constexpr int kQBoxDims = 64;
    static constexpr int kQBoxBytes =
        kHeads * kQBoxDims * sizeof(__nv_bfloat16);
    static constexpr int kChunksPerBox =
        kQBoxDims * sizeof(__nv_bfloat16) / kChunk;
    static constexpr int kQoRounds = kHalfDims / kQBoxDims;
    static_assert(kQoRounds == 4, "fetch_q2: two Q buffers, rounds 0..3");
    static constexpr int kQVecs =
        kHalfDims * sizeof(__nv_bfloat16) / sizeof(uint4);
    static constexpr int kRoundWords = kQBoxDims / 2;
    static constexpr int kFreqLanes = kQRopeDims * sizeof(float) / kChunk;
    static constexpr int kFreqT0 = kWarp;

    static constexpr int kPlaneCols = 4;
    static constexpr int kPlanes = kHalfDims / kPlaneCols;
    static constexpr int kPlaneBytes = kWarpGroup * kPlaneCols * sizeof(float);
    static constexpr int kPartBoxPlanes = 16;
    static constexpr int kPartBoxes = kPlanes / kPartBoxPlanes;
    static constexpr int kOHalf = kPvN / 2;
    static constexpr int kMergeCols = 32;
    static constexpr int kGmemMergeCols = 16;
    static constexpr int kLseRegs = 8;
    static constexpr int kStoreWords = 8;
    static constexpr int kStoreCols = 2 * kStoreWords;
    static constexpr int kDeferred = 1 << 16;
    static constexpr int kCombSplit = 4;

    static constexpr uint64_t kKvL2Hint =
        uint64_t(cute::TMA::CacheHintSm90::EVICT_FIRST);
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
    static_assert(kVecLen == sizeof(uint4) / sizeof(uint32_t),
                  "a float4 and a uint4 have the same number of components");
    static_assert(kPlaneCols == kVecLen, "a plane column group is a float4");
};

// Persistent schedule: the tile stream is split over the clusters; kQst
// kernels run pairs of at most kQstMaxTiles tiles, and a pair of n tiles
// costs kTileCost * n plus a fixed cost in quarter tiles.
struct PrefillSchedTraits {
    static constexpr int kTableUnits = 32;
    static constexpr int kMaxClusters = 128;
    static constexpr int kDefaultTpc = 3;
    static constexpr int kQstMaxTiles = 5;
    static constexpr int kPairFixQst = 24;
    static constexpr int kPairFix = 24;
    static constexpr int kTileCost = 4;
    static constexpr int kKeepPairs = 2;
};

// Pre-dequant (prefill_predq_kernel) limits by ld[0], token count and bytes.
struct PrefillPredqTraits {
    static constexpr int kThreads = 256;
    static constexpr int kWarps = kThreads / KernelBase::WarpSize;
    static constexpr int64_t kMaxBytes = int64_t(512) << 20;
    static constexpr int kMinLd = 512, kMinTokens = 256;
    static constexpr int kShortLd = 256, kShortTokens = 4096;
};

// kQst: a unit's four Q rounds alternate between two buffers.
template <class T, bool kQst>
struct PrefillQBufs {};
template <class T>
struct PrefillQBufs<T, true> {
    alignas(T::kAtomBytes) uint8_t q_stage[2][2][T::kQBoxBytes];
    uint64_t qb_full[2];
};

// Shared memory of a pair prefill CTA.
template <class T, bool kQst>
struct PrefillSmem {
    alignas(T::kAtomBytes) uint8_t kv[T::kKvStages][T::kStageBytes];
    alignas(T::kAtomBytes) uint8_t p[2][T::kHeads * T::kRowPitch];
    alignas(T::kAtomBytes) uint8_t qo_stage[2][T::kQBoxBytes];
    PrefillQBufs<T, kQst> qb;
    alignas(T::kChunk) float xchg[T::kWarpGroup][T::kHalf];
    alignas(T::kChunk) float qfreq[T::kQRopeDims];
    float rs[2][T::kHeads];
    alignas(T::kChunk) uint8_t sf[T::kKvStages][T::kTile][T::kSfBytes];
    uint32_t valid[T::kIdxStages][T::kIndexWarps];
    uint32_t any[T::kIdxStages][T::kIndexWarps];
    int bf[T::kIdxStages];
    int idx_raw[T::kIndexWarps][T::kIndexDepth][T::kWarp][2];
    float red[T::kWarpGroup];
    float2 ml[2][T::kWarpGroup];
    int start[3];
    int4 unit_lens[PrefillSchedTraits::kTableUnits];
    int wstart;
    int sched;
    int dense;
    int bound[PrefillSchedTraits::kMaxClusters + 1];
    uint64_t q_full, q_empty, o_full, o_empty[2], merge_full[T::kMergeBuf];
    uint64_t qs_full;
    uint64_t ml_full[2], ml_empty[2];
    uint64_t rs_full[2];
    uint64_t idx_full[T::kIdxStages], idx_empty[T::kIdxStages];
    uint64_t raw_full[T::kKvStages], kv_full[T::kKvStages],
        kv_empty[T::kKvStages];
    uint64_t s_full[2], s_empty[2], p_full[2], pv_done[2];
    uint32_t tmem_base;
};

// kQst: Q buffers for short pairs; kFp8: no bf16 c128 copy; kQnr: fused
// q_norm_rope.
template <bool kQstV, bool kFp8V, bool kQnrV, class Dims = ModelDims>
struct PrefillCfg : PrefillPairTraits<Dims> {
    using Traits = PrefillPairTraits<Dims>;
    static constexpr bool kQst = kQstV;
    static constexpr bool kFp8 = kFp8V;
    static constexpr bool kQnr = kQnrV;
    using Smem = PrefillSmem<Traits, kQstV>;
    static constexpr int kSmemBytes = sizeof(Smem) + Traits::kAtomBytes;
    static_assert(kSmemBytes <= Sm100::SmemPerCtaMax);
};

// 2-CTA bf16 UMMA (M = 2 x 64 heads), A from TMEM (QK^T).
template <class C, int N, cute::UMMA::Major BMajor>
using PrefillUmmaTs =
    cute::SM100_MMA_F16BF16_2x1SM_TS<cutlass::bfloat16_t, cutlass::bfloat16_t,
                                     float, 2 * C::kHeads, N,
                                     cute::UMMA::Major::K, BMajor>;

// 2-CTA bf16 UMMA (M = 2 x 64 heads), A from shared memory (PV).
template <class C, int N, cute::UMMA::Major BMajor>
using PrefillUmmaSs =
    cute::SM100_MMA_F16BF16_2x1SM_SS<cutlass::bfloat16_t, cutlass::bfloat16_t,
                                     float, 2 * C::kHeads, N,
                                     cute::UMMA::Major::K, BMajor>;

// Dequantizes the KV stage of CTA kCta in place: all reads, a barrier, then
// the writes. Only CTA 1 holds rope chunks.
template <class C, int kCta>
struct PrefillDequantPlan {
    static constexpr int kFp8Chunks =
        (kCta == 0 ? C::kHalfDims : C::kNopeBytes1) / C::kChunk;
    static constexpr int kFp8Loads = C::kTile * kFp8Chunks / C::kWarpGroup;
    static constexpr int kRopeChunks =
        kCta == 0 ? 0 : (C::kRowBytes - C::kNope) / C::kChunk;
    static constexpr int kRopeLoads = C::kTile * kRopeChunks / C::kWarpGroup;
};

}  // namespace halops::mla
