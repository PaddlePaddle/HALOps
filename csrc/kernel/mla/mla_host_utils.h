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

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cutlass/cuda_host_adapter.hpp>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "kernel/gemm/gemm_utils.h"

namespace halops::mla {
inline int grid_x() { return GetSmCount(); }

inline int page_shift_of(int64_t page) {
    int shift = 0;
    while ((1LL << shift) < page) ++shift;
    return shift;
}

inline CUtensorMap make_tensor_map(
    const std::vector<uint64_t>& size, const std::vector<uint64_t>& strides,
    const std::vector<uint32_t>& box_size, void* global_ptr,
    CUtensorMapDataType data_type, CUtensorMapSwizzle swizzle_mode,
    CUtensorMapL2promotion l2_promotion,
    CUtensorMapInterleave interleave_mode =
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
    CUtensorMapFloatOOBfill oob_fill =
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
    const std::vector<uint32_t>& element_strides_ = {}) {
    const int dim = (int)size.size();

    std::vector<uint32_t> element_strides;
    if (element_strides_.empty())
        element_strides.assign(dim, 1u);
    else
        element_strides = element_strides_;

    CUtensorMap result;
    const CUresult ret_code =
        CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
            &result, data_type, dim, global_ptr, size.data(), strides.data(),
            box_size.data(), element_strides.data(), interleave_mode,
            swizzle_mode, l2_promotion, oob_fill);
    if (ret_code != CUresult::CUDA_SUCCESS) {
        auto join = [](auto const& values) {
            std::string joined;
            for (auto entry : values) joined += std::to_string(entry) + " ";
            return joined;
        };
        TORCH_CHECK(false, "cuTensorMapEncodeTiled failed (", (int)ret_code,
                    "): dim=", dim, " size=[", join(size), "] strides=[",
                    join(strides), "] box=[", join(box_size),
                    "] elem_strides=[", join(element_strides),
                    "] ptr=", global_ptr, " dtype=", (int)data_type,
                    " swizzle=", (int)swizzle_mode, " l2=", (int)l2_promotion,
                    " interleave=", (int)interleave_mode,
                    " oob=", (int)oob_fill);
    }
    return result;
}

struct LaunchDims {
    int grid;
    int block;
    size_t smem = 0;
    int cluster = 1;
};

inline int setup_kernel_once(const void* kernel, const LaunchDims& dims,
                             const cudaLaunchConfig_t& cluster_query) {
    static std::mutex mutex;
    static std::map<std::pair<const void*, int>, int> max_clusters;
    int device = 0;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    const std::lock_guard<std::mutex> lock(mutex);
    const auto key = std::make_pair(kernel, device);
    const auto it = max_clusters.find(key);
    if (it != max_clusters.end()) return it->second;
    if (dims.smem > 0)
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dims.smem));
    int clusters = 0;
    if (dims.cluster > 1)
        C10_CUDA_CHECK(
            cudaOccupancyMaxActiveClusters(&clusters, kernel, &cluster_query));
    max_clusters.emplace(key, clusters);
    return clusters;
}

template <class Kernel, class... Args>
void launch_pdl(Kernel* kernel, const LaunchDims& dims, Args... args) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(dims.grid, 1, 1);
    cfg.blockDim = dim3(dims.block, 1, 1);
    cfg.dynamicSmemBytes = dims.smem;
    cfg.stream = c10::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute attr[2] = {};
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    attr[1].id = cudaLaunchAttributeClusterDimension;
    attr[1].val.clusterDim.x = dims.cluster;
    attr[1].val.clusterDim.y = 1;
    attr[1].val.clusterDim.z = 1;
    cfg.attrs = attr;
    cfg.numAttrs = dims.cluster > 1 ? 2 : 1;
    cudaLaunchConfig_t query = cfg;
    query.attrs = attr + 1;
    query.numAttrs = 1;
    const int max_clusters =
        setup_kernel_once((const void*)kernel, dims, query);
    TORCH_CHECK(dims.cluster == 1 || max_clusters >= dims.grid / dims.cluster,
                "mla: only ", max_clusters, " of ", dims.grid / dims.cluster,
                " clusters co-resident");
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, args...));
}
}  // namespace halops::mla
