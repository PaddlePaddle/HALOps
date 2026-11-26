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

#include <cute/arch/copy_sm75.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <type_traits>

#include "common/mx_quant.cuh"
#include "common/ptx.cuh"

namespace halops::mla {

/// Model shape of the attention output fed to the head-mix post-processing.
template <int kHeads_, int kHeadDim_, int kRopeDim_>
struct VhaPostmixFormat {
    static constexpr int kHeads = kHeads_;
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr int kRopeDim = kRopeDim_;
};

/// The V4 attention output: 64 heads of 512 dims, the last 64 of them rope.
using V4VhaPostmixFormat = VhaPostmixFormat<64, 512, 64>;

/// Tiling, MMA fragments and shared-memory staging of vha_postmix_quant.
template <class Fmt>
struct VhaPostmixCfg : Fmt {
    using Fmt::kHeadDim;
    using Fmt::kHeads;
    using Fmt::kRopeDim;
    using Mma = cute::SM80_16x8x16_F32BF16BF16F32_TN;
    using Ldsm = cute::SM75_U16x8_LDSM_T;

    static constexpr int kCols = 128;
    static constexpr int kQuarters = kHeadDim / kCols;
    static constexpr int kThreads = 128;
    static constexpr int kWarps = kThreads / cutlass::NumThreadsPerWarp;
    static constexpr int kElems = sizeof(uint4) / sizeof(__nv_bfloat16);
    static constexpr int kChunks = kCols / kElems;
    static constexpr int kRopeChunk = (kHeadDim - kRopeDim) % kCols / kElems;
    static constexpr int kLoads = kHeads * kChunks / kThreads;
    static constexpr int kRopeChunks = kRopeDim / kElems;
    static constexpr int kRopeLoads = kHeads * kRopeChunks / kThreads;
    static constexpr int kTileChunks = kHeads * kChunks;
    static constexpr int kMmaM = 16;
    static constexpr int kMmaN = 8;
    static constexpr int kMmaK = 16;
    static constexpr int kKSteps = kHeads / kMmaK;
    static constexpr int kNTiles = kCols / kMmaN;
    static constexpr int kBlocksPerSm = 4;
    static constexpr int kFreqChunks = kRopeDim * sizeof(float) / sizeof(uint4);
    static constexpr int kStageChunks = kTileChunks + kFreqChunks;
    static constexpr int kSmemBytes = 2 * kStageChunks * sizeof(uint4);
    static constexpr int kGroupTiles = mxfp8::kGroup / kMmaN;
    static constexpr int kNHalves = 2;
    static constexpr int kHalfTiles = kNTiles / kNHalves;
    static constexpr int kHalfGroups = kHalfTiles / kGroupTiles;
    static constexpr int kTileRowBytes = kCols * sizeof(__nv_bfloat16);

    static constexpr int kARegs = std::extent_v<typename Mma::ARegisters>;
    static constexpr int kBRegs = std::extent_v<typename Mma::BRegisters>;
    static constexpr int kCRegs = std::extent_v<typename Mma::CRegisters>;
    static constexpr int kLdsmRegs = std::extent_v<typename Ldsm::DRegisters>;
    static constexpr int kScaleBits = 8;
    static constexpr uint32_t kPermLo = 0x5410;
    static constexpr uint32_t kPermHi = 0x7632;

    static_assert(kHeads == kWarps * kMmaM, "one 16-head band per warp");
};

}  // namespace halops::mla
