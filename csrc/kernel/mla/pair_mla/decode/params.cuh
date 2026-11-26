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

#include <cuda.h>
#include <cuda_bf16.h>
#include <cutlass/arch/barrier.h>

#include <cstdint>

#include "kernel/mla/pair_mla/common/schedule.cuh"

namespace halops::mla {

// Host schedule defaults and the combine-kernel launch shape.  kNoCap marks
// an uncapped split.
struct HcaPlan {
    static constexpr int kNoCap = 0x7fffffff;
    static constexpr int kDefaultSmax = 6;
    static constexpr int kCapFloor = 8;
    static constexpr int kDefaultTpc = 3;
    static constexpr int kCombSplit = 4;
    static constexpr int kCombJobs = 8;
    static constexpr int kEffSized = 1, kEffCost = 2, kEffAligned = 3;
};

// Measured cost model of find_range (eff == kEffCost) and the host's
// split-count fit: base + per-tile * tiles + per-pair * pairs.
struct HcaCost {
    static constexpr int kUnsplitBase = 1005, kUnsplitPerTile = 114,
                         kUnsplitPerPair = 3;
    static constexpr int kSplitBase = 1410, kSplitPerTile = 116,
                         kSplitPerPair = 11;
    static constexpr int kCombBase = 1540, kCombPerTile = 124,
                         kCombPerPair = 15;
    static constexpr float kFewClustersPerPair = 3.f;
    static constexpr int kFewSplitBase = 1362, kFewSplitPerTile = 115,
                         kFewSplitPerPair = 14;
    static constexpr int kFewCombBase = 1813, kFewCombPerTile = 116,
                         kFewCombPerPair = 14;
    static constexpr int kSpreadBase = 1417, kSpreadPerTile = 119,
                         kSpreadPerPair = 8;
    static constexpr float kSpreadPerCluster = 164.5f;
    static constexpr int kMinSplit = 3;
    static constexpr float kCeilEps = 0.01f;
    static constexpr float kTileCost = 1.17f;
    static constexpr float kFracEps = 1e-3f, kFracPenalty = 1.8f;
    static constexpr float kCombBias = 0.3f;
    static constexpr int kPlainTileWeight = 4, kPlainPairWeight = 3,
                         kPlainPairBias = 8;
    static constexpr int kAlignSlack = 6;
    static constexpr float kKsBase = 0.077f, kKsPerPair = 0.062f;
};

struct HcaMaps {
    CUtensorMap raw[2][2];
    CUtensorMap q;
    CUtensorMap out;
};

struct HcaParams {
    const int* idx[2];
    const int* len[2];
    int ld[2];
    const uint8_t* pool[2];
    int64_t pool_stride[2];
    int page_shift[2];
    int page_rows[2];
    int64_t pool_tokens[2];
    const __nv_bfloat16* q;
    const float* sink;
    __nv_bfloat16* out;
    float* o_part;
    float* lse_part;
    int* counter;
    int tokens;
    int tokens_per_req;
    int clusters;
    int cap;
    int smax;
    int nomerge;
    int pipe;
    int combine;
    int fuse;
    PairMergeJob* jobs;
    int* total_g;
    int eff;
    int tpc;
    int ks_kmax;
    float ks_b, ks_a, ks_rp;
    int* eff_g;
    float scale_log2;
};

using Bar = cutlass::arch::ClusterBarrier;
using NamedBar = cutlass::arch::NamedBarrier;

}  // namespace halops::mla
