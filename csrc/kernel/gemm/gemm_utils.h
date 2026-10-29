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

#include <c10/util/BFloat16.h>
#include <c10/util/Float8_e4m3fn.h>
#include <c10/util/Half.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "common/cuda_utils.h"

namespace halops {

// SM count of the current device, cached for the host-side launch heuristics.
inline int GetSmCount() {
    static int cached = 0;
    if (cached == 0) {
        int device = 0;
        cudaGetDevice(&device);
        cudaDeviceGetAttribute(&cached, cudaDevAttrMultiProcessorCount, device);
    }
    return cached;
}

inline constexpr int kSfAtomM = 128;

constexpr int AlignSfM(int m) {
    return (m + kSfAtomM - 1) / kSfAtomM * kSfAtomM;
}

inline void CheckDivisible(int64_t value, int64_t divisor, const char* name) {
    TORCH_CHECK(value % divisor == 0, name, " must be divisible by ", divisor,
                ", got ", value);
}

// SM count for the tile-selection heuristics. `sm_count <= 0` falls back to
// the cached device count, so dispatch sees the same SM count gemm_impl
// launches onto (a hard-coded value would mis-pick tiles in the
// `tiles(...) < target_sm` ladders).
inline int ResolveSmCount(int sm_count) {
    return sm_count > 0 ? sm_count : GetSmCount();
}

inline void CheckGemmTileShape(int64_t m, int64_t n, int64_t k, int64_t tile_m,
                               int64_t tile_n, int64_t tile_k,
                               const char* kernel_name) {
    TORCH_CHECK(m % tile_m == 0, kernel_name, " requires M to be divisible by ",
                tile_m, ", got ", m);
    TORCH_CHECK(n % tile_n == 0, kernel_name, " requires N to be divisible by ",
                tile_n, ", got ", n);
    TORCH_CHECK(k % tile_k == 0, kernel_name, " requires K to be divisible by ",
                tile_k, ", got ", k);
}

}  // namespace halops
