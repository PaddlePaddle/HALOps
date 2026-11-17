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

#include <span>

#include "kernel/gemm/gemm_tile.cuh"

namespace halops {

struct TunedRow {
    int m;
    TileSpec tile;
    TileSpec alt = tile;
};

struct TunedShape {
    int n;
    int k;
    std::span<const TunedRow> rows;
};

inline constexpr TunedRow kDenseFp8TunedN1536K4096[] = {
    {1792, kFp8_256x160_pair_swap},
    {2944, kFp8_128x256},
    {3200, kFp8_128x256, kFp8_256x160_pair_swap},
    {3328, kFp8_256x160_pair_swap},
    {3584, kFp8_256x160_pair_swap},
    {3840, kFp8_256x160_pair_swap},
    {6016, kFp8_128x256, kFp8_256x160_pair_swap},
    {6272, kFp8_128x256},
    {9600, kFp8_256x160_pair_swap, kFp8_256x192_pair_swap},
    {9728, kFp8_256x256_pair, kFp8_256x160_pair_swap},
    {9856, kFp8_256x160_pair_swap, kFp8_256x224_pair_swap},
    {9984, kFp8_256x224_pair_reg160, kFp8_256x160_pair_swap},
    {12160, kFp8_128x256, kFp8_256x160_pair_swap},
    {12800, kFp8_256x256_pair, kFp8_256x224_pair_reg160},
    {13056, kFp8_256x224_pair_reg160},
    {13568, kFp8_256x192_pair_reg160, kFp8_256x224_pair_reg160},
    {13696, kFp8_256x160_pair_swap, kFp8_256x224_pair_swap},
    {13824, kFp8_256x192_pair_reg160, kFp8_256x224_pair_reg160},
    {14336, kFp8_256x192_pair_reg160, kFp8_256x128_pair},
};

inline constexpr TunedShape kDenseFp8Tuned[] = {
    {1536, 4096, kDenseFp8TunedN1536K4096},
};

}  // namespace halops
