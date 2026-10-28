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

#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>

extern "C" int32_t aoti_torch_get_current_cuda_stream(int32_t device_index,
                                                      void** ret_stream);

#define HALOPS_CUDA_CHECK(EXPR)                                            \
    do {                                                                   \
        const cudaError_t halops_cuda_err_ = (EXPR);                       \
        TORCH_CHECK(halops_cuda_err_ == cudaSuccess,                       \
                    "CUDA error: ", cudaGetErrorString(halops_cuda_err_)); \
    } while (0)

namespace halops {

class CudaDeviceGuard {
  public:
    explicit CudaDeviceGuard(c10::Device device) {
        if (!device.is_cuda() || device.index() < 0) return;
        HALOPS_CUDA_CHECK(cudaGetDevice(&prev_));
        if (prev_ != device.index()) {
            HALOPS_CUDA_CHECK(cudaSetDevice(device.index()));
            restore_ = true;
        }
    }
    ~CudaDeviceGuard() {
        if (restore_) cudaSetDevice(prev_);
    }
    CudaDeviceGuard(const CudaDeviceGuard&) = delete;
    CudaDeviceGuard& operator=(const CudaDeviceGuard&) = delete;

  private:
    int prev_ = -1;
    bool restore_ = false;
};

inline int CurrentDevice() {
    int device = 0;
    HALOPS_CUDA_CHECK(cudaGetDevice(&device));
    return device;
}

inline cudaStream_t GetCurrentCUDAStream(int device_index) {
    void* stream = nullptr;
    TORCH_CHECK(aoti_torch_get_current_cuda_stream(device_index, &stream) == 0,
                "failed to get the current CUDA stream of device ",
                device_index);
    return static_cast<cudaStream_t>(stream);
}

inline cudaStream_t GetCurrentCUDAStream(const torch::Tensor& tensor) {
    return GetCurrentCUDAStream(tensor.device().index());
}

inline cudaStream_t GetCurrentCUDAStream() {
    return GetCurrentCUDAStream(CurrentDevice());
}

inline int CurrentDeviceSmCount() {
    int sm_count = 0;
    HALOPS_CUDA_CHECK(cudaDeviceGetAttribute(
        &sm_count, cudaDevAttrMultiProcessorCount, CurrentDevice()));
    return sm_count;
}

}  // namespace halops
