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
#include <cuda_runtime.h>
#include <cutlass/float_subbyte.h>
#include <torch/extension.h>

#include <cstdint>
#include <type_traits>

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

// Rows of the 2D tensor t are contiguous and start 16-byte aligned, so a kernel
// can read them with 128-bit loads at the tensor's row stride.
inline void CheckRowsVecAligned(const torch::Tensor& t, const char* name) {
    constexpr int kVecBytes = sizeof(uint4);
    TORCH_CHECK(t.stride(1) == 1 &&
                    t.stride(0) * t.element_size() % kVecBytes == 0 &&
                    reinterpret_cast<uintptr_t>(t.data_ptr()) % kVecBytes == 0,
                name, " rows must be contiguous and 16-byte aligned");
}

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

template <typename T>
inline constexpr CUtensorMapDataType TmaDataType() {
    if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        return CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
    } else if constexpr (std::is_same_v<T, float>) {
        return CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
    } else if constexpr (std::is_same_v<T, std::uint32_t>) {
        return CU_TENSOR_MAP_DATA_TYPE_UINT32;
    } else if constexpr (std::is_same_v<T, cutlass::float_e2m1_t>) {
        return CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B;
    } else {
        static_assert(sizeof(T) == 1u, "unsupported TMA element type");
        return CU_TENSOR_MAP_DATA_TYPE_UINT8;
    }
}

template <int Rank, typename T>
inline CUtensorMap CreateTmaDesc(
    const char* debug_name, const T* global_ptr,
    const std::uint64_t (&shape)[Rank],
    const std::uint64_t (&strides)[Rank - 1], const std::uint32_t (&box)[Rank],
    bool use_swizzle = true,
    CUtensorMapL2promotion promotion = CU_TENSOR_MAP_L2_PROMOTION_L2_128B) {
    static_assert(Rank >= 2 && Rank <= 5, "TMA supports rank 2..5 here");
    static constexpr std::uint32_t kUnitStride[5] = {1u, 1u, 1u, 1u, 1u};
    CUtensorMap desc{};
    const CUresult res = cuTensorMapEncodeTiled(
        &desc, TmaDataType<T>(), Rank,
        const_cast<void*>(static_cast<const void*>(global_ptr)), shape, strides,
        box, kUnitStride, CU_TENSOR_MAP_INTERLEAVE_NONE,
        use_swizzle ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_NONE,
        promotion, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(res == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed for ",
                debug_name, ": CUresult=", static_cast<int>(res));
    return desc;
}

}  // namespace halops
