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

#include "common/sm100_arch.cuh"

namespace halops::mla {
struct Sm100 {
    static constexpr int TmemColumns = 512;
    static constexpr int UmmaMaxN = 256;
    static constexpr int SmemPerCtaMax = kSmemCapacity;
    static constexpr int Gather4Rows = 4;
    static constexpr int Sw128Bytes = 128;
    static constexpr int Sw128AtomBytes = 8 * Sw128Bytes;
    static constexpr int UmmaIdescShift = 32;
};

struct SfLayout {
    static constexpr int Gran = 32;
    static constexpr int MnPerCol = 32;
    static constexpr int MnPerTile = 4 * MnPerCol;
    static constexpr int PackedMinTile = 64;
};
}  // namespace halops::mla
