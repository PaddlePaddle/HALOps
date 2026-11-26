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

// Per-CTA dequant loads: CTA 0 the low fp8 dims, CTA 1 the rest plus rope.
template <int kCta, class C>
struct PairDequantPlan {
    static constexpr int kFp8Chunks =
        (kCta == 0 ? C::kHalfDims : C::kNopeBytes1) / kChunk;
    static constexpr int kFp8Loads = C::kTile * kFp8Chunks / kWarpGroup;
    static constexpr int kRopeChunks =
        kCta == 0 ? 0 : (C::kRowBytes - C::kNope) / kChunk;
    static constexpr int kRopeLoads = C::kTile * kRopeChunks / kWarpGroup;
};

}  // namespace halops::mla
