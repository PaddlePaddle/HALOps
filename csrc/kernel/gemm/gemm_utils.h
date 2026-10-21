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

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Float8_e4m3fn.h>
#include <c10/util/Half.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace halops {

inline cudaStream_t GetCurrentCUDAStream(const torch::Tensor& tensor) {
    return at::cuda::getCurrentCUDAStream(tensor.device().index()).stream();
}

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

inline void CheckDivisible(int64_t value, int64_t divisor, const char* name) {
    TORCH_CHECK(value % divisor == 0, name, " must be divisible by ", divisor,
                ", got ", value);
}

}  // namespace halops
