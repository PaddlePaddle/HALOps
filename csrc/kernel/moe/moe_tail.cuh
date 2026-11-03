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
    static constexpr int kStage = Config::kTailStage;
    static constexpr int kBlkB = Config::kTailBlkB;
    static constexpr int kWStagePad = Config::kTailWStagePad;
    static constexpr int kXPad = Config::kTailXPad;
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
};

template <typename Dims>
using TailParams =
    ::halops::moe::tail::Params<TailGeometryT<Dims>::kWorld, Dims::kLatentSize,
                                Dims::kHiddenSize, TailGeometryT<Dims>::kMaxM>;

template <typename Dims>
using TailSmemLayout = ::halops::moe::tail::TailSmem<
    TailGeometryT<Dims>::kWorld, Dims::kLatentSize, Dims::kHiddenSize,
    TailGeometryT<Dims>::kMaxM, TailGeometryT<Dims>>;

template <typename Dims>
constexpr std::int64_t tail_region_bytes() {
    return std::int64_t(2) * 2 * TailGeometryT<Dims>::kWorld *
               TailGeometryT<Dims>::kMaxM * Dims::kHiddenSize * 2 +
           std::int64_t(2) * TailGeometryT<Dims>::kMaxM * Dims::kLatentSize * 2;
}

struct CalcWarpSync {
    __device__ __forceinline__ void operator()() const {
        WarpRole::sync_calc_threads();
    }
};

}  // namespace halops::moe
