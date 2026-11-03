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

#include "moe_configs.cuh"
#include "moe_tail_fused.cuh"

namespace halops::moe {

template <typename Dims>
struct TailGeometryT {
    using Config = typename Dims::KernelConfig;
    static constexpr int kWorld = Config::kTailWorld;
    static constexpr int kMaxM = static_cast<int>(Dims::kNumTokens);
    static constexpr int kTpg = Config::kTailTpg;
    static constexpr int kGsize = Config::kTailGsize;
    static constexpr int kThreads = Config::kTailThreads;
    static constexpr std::uint32_t kWarps = kThreads / 32;
    static constexpr int kStage = Config::kTailStage;
    static constexpr int kBlkB = Config::kTailBlkB;
    static constexpr int kWStagePad = Config::kTailWStagePad;
    static constexpr int kXPad = Config::kTailXPad;
    static constexpr int kXStride = static_cast<int>(Dims::kLatentSize) + kXPad;
    static constexpr int kShard = static_cast<int>(Dims::kHiddenSize) / kWorld;
    static constexpr int kRows = kShard / kGsize;
    static constexpr int kTiles = kRows / 16;
    static constexpr int kKSplit = kTiles >= static_cast<int>(kWarps)
                                       ? 1
                                       : static_cast<int>(kWarps) / kTiles;
    static constexpr int kItemsPerWarp =
        kTiles * kKSplit / static_cast<int>(kWarps);
    static_assert(kRows % 16 == 0,
                  "rows per CTA must be a multiple of the mma M tile");
    static_assert(kTiles * kKSplit % static_cast<int>(kWarps) == 0,
                  "work items must tile over the warps");
    static constexpr int kMaxStageSmemBytes = Config::kTailMaxStageSmemBytes;
    static constexpr std::uint32_t kPrefetchChunk = Config::kTailPrefetchChunk;
    static constexpr int kScaleBlock = static_cast<int>(Dims::kQuantBlockM);
    static_assert(Dims::kQuantBlockM == Dims::kQuantBlockK,
                  "fc2 scales are square blocks");
    static constexpr std::uint32_t kNumCtas = (kMaxM / kTpg) * kGsize;
    static_assert(kMaxM % kTpg == 0, "tail groups tile the token capacity");
    static_assert(kThreads == static_cast<int>(WarpRole::kCalcWarpCount *
                                               WarpRole::kThreadsPerWarp),
                  "the tail runs on exactly the calc warps");
    static_assert(
        kThreads * 8 == static_cast<int>(Dims::kLatentSize),
        "one CTA covers one latent row, one 16B vector (8 bf16) per thread");

    // Shared-memory layout of one tail CTA: byte offsets from the scratch base.
    static constexpr std::uint32_t align_up(std::uint32_t v, std::uint32_t a) {
        return (v + a - 1) / a * a;
    }
    static constexpr std::uint32_t kSmemWStage = 0;
    static constexpr std::uint32_t kSmemWStageBytes =
        kStage == 0 ? 0u
                    : static_cast<std::uint32_t>(
                          tail::stage_smem_bytes<kRows, Dims::kLatentSize,
                                                 kWStagePad>());
    static constexpr std::uint32_t kSmemWBar =
        align_up(kSmemWStage + kSmemWStageBytes, 16);
    static constexpr std::uint32_t kSmemX = align_up(kSmemWBar + 8, 16);
    static constexpr std::uint32_t kSmemWarp =
        align_up(kSmemX + std::uint32_t(kTpg * kXStride) * 2, 16);
    static constexpr std::uint32_t kSmemPart =
        align_up(kSmemWarp + kWarps * 4, 16);
    static constexpr std::uint32_t kSmemCol =
        align_up(kSmemPart + std::uint32_t(kKSplit * kTpg * kRows) * 4, 16);
    static constexpr std::uint32_t kSmemBytes =
        align_up(kSmemCol + std::uint32_t(kTpg * kRows) * 4, 128);

    static constexpr std::int64_t kSlotRegionBytes =
        std::int64_t(2) * 2 * kWorld * kMaxM * Dims::kHiddenSize * 2;
    static constexpr std::int64_t kXbufRegionBytes =
        std::int64_t(2) * kMaxM * Dims::kLatentSize * 2;
};

template <typename Dims>
constexpr std::int64_t tail_region_bytes() {
    return TailGeometryT<Dims>::kSlotRegionBytes +
           TailGeometryT<Dims>::kXbufRegionBytes;
}

struct CalcWarpSync {
    __device__ __forceinline__ void operator()() const {
        WarpRole::sync_calc_threads();
    }
};

}  // namespace halops::moe
