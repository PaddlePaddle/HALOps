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

namespace halops::mla {
template <int kHeads_, int kDimQ_, int kDimV_, int kDimNope_, int kDimRope_,
          int kQLoraRank_, int kC128Ratio_>
struct CommonModelFormat {
    static constexpr int Heads = kHeads_;
    static constexpr int DimQ = kDimQ_;
    static constexpr int DimV = kDimV_;
    static constexpr int DimNope = kDimNope_;
    static constexpr int DimRope = kDimRope_;

    static constexpr int QLoraRank = kQLoraRank_;

    static constexpr int C128Ratio = kC128Ratio_;

    static constexpr int QTotal = Heads * DimQ;
    static constexpr int OTotal = Heads * DimV;
    static constexpr int KvRowBytes = DimNope + 2 * DimRope;
};

using ModelDims = CommonModelFormat<64, 512, 512, 448, 64, 1024, 128>;

template <class Dims, int kNopeQuantGroup_>
struct CommonKvCacheFormat {
    static constexpr int NopeQuantBlock = kNopeQuantGroup_;
    static constexpr int NopeGroups = Dims::DimNope / NopeQuantBlock;
    static constexpr int ScaleSlotBytes = sizeof(uint64_t);
    static_assert(NopeGroups <= ScaleSlotBytes, "one scale byte per group");
};

using KvCacheLayout = CommonKvCacheFormat<ModelDims, 64>;
}  // namespace halops::mla
