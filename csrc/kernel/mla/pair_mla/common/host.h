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
#include <torch/extension.h>

#include <cstdint>

#include "kernel/gemm/gemm_utils.h"

namespace halops::mla {

// A 2D TMA map over the KV pool rows (one row of C::kRowBytes per token slot).
template <class C>
CUtensorMap pair_make_pool_map(const void* ptr, CUtensorMapDataType dtype,
                               const torch::Tensor& pool, uint64_t cols,
                               uint32_t box, CUtensorMapSwizzle swizzle,
                               const char* op) {
    CUtensorMap map;
    EncodeTmaDesc<2>(
        &map, dtype, ptr,
        {cols, uint64_t(pool.size(0) * (pool.stride(0) / C::kRowBytes))},
        {C::kRowBytes}, {box, 1}, swizzle, op);
    return map;
}

}  // namespace halops::mla
