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

namespace halops {

// Programmatic Dependent Launch (PDL) helpers.

__device__ __forceinline__ void PdlWaitForDependents() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    cudaGridDependencySynchronize();
#endif
}

__device__ __forceinline__ void PdlTriggerAfterBlockWrites() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        cudaTriggerProgrammaticLaunchCompletion();
    }
#endif
}

template <typename KernelT, typename... Args>
inline void LaunchKernelWithPdl(KernelT kernel, dim3 grid_dim, dim3 block_dim,
                                std::size_t smem_bytes, cudaStream_t stream,
                                bool use_pdl, Args&&... args) {
    if (use_pdl) {
        cudaLaunchAttribute attr;
        attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr.val.programmaticStreamSerializationAllowed = 1;
        cudaLaunchConfig_t config{};
        config.gridDim = grid_dim;
        config.blockDim = block_dim;
        config.dynamicSmemBytes = smem_bytes;
        config.stream = stream;
        config.attrs = &attr;
        config.numAttrs = 1;
        cudaLaunchKernelEx(&config, kernel, args...);
        return;
    }
    kernel<<<grid_dim, block_dim, smem_bytes, stream>>>(args...);
}

}  // namespace halops
