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

#include <cstddef>
#include <cute/config.hpp>

namespace halops {

CUTE_DEVICE void pdl_wait() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

CUTE_DEVICE void pdl_trigger() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.launch_dependents;" :::);
#endif
}

inline cudaLaunchAttribute PdlLaunchAttribute() {
    cudaLaunchAttribute attr{};
    attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr.val.programmaticStreamSerializationAllowed = 1;
    return attr;
}

inline constexpr dim3 kNoCluster{1, 1, 1};

template <typename KernelT, typename... Args>
inline cudaError_t LaunchKernelWithPdl(KernelT kernel, dim3 grid_dim,
                                       dim3 block_dim, dim3 cluster_dim,
                                       std::size_t smem_bytes,
                                       cudaStream_t stream, bool use_pdl,
                                       Args&&... args) {
    const bool clustered = cluster_dim.x * cluster_dim.y * cluster_dim.z > 1;
    if (!use_pdl && !clustered) {
        kernel<<<grid_dim, block_dim, smem_bytes, stream>>>(args...);
        return cudaGetLastError();
    }
    cudaLaunchAttribute attrs[2];
    unsigned int num_attrs = 0;
    if (clustered) {
        attrs[num_attrs].id = cudaLaunchAttributeClusterDimension;
        attrs[num_attrs].val.clusterDim.x = cluster_dim.x;
        attrs[num_attrs].val.clusterDim.y = cluster_dim.y;
        attrs[num_attrs].val.clusterDim.z = cluster_dim.z;
        ++num_attrs;
    }
    if (use_pdl) attrs[num_attrs++] = PdlLaunchAttribute();
    cudaLaunchConfig_t config{};
    config.gridDim = grid_dim;
    config.blockDim = block_dim;
    config.dynamicSmemBytes = smem_bytes;
    config.stream = stream;
    config.attrs = attrs;
    config.numAttrs = num_attrs;
    return cudaLaunchKernelEx(&config, kernel, args...);
}

}  // namespace halops
