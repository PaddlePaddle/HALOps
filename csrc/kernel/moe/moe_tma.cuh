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

#include <cstddef>
#include <cstdint>
#include <cute/arch/copy_sm90_tma.hpp>
#include <type_traits>

#include "kernel/moe/moe_configs.cuh"

#if defined(CUDA_VERSION) && (CUDA_VERSION < 12000)
#    error \
        "CUDA 12.0 or later is reqiured for CUtensorMap / cuTensorMapEncodeTiled."
#endif

namespace halops::moe {

// L2 eviction hint for the streamed TMA loads: a working set that fits in L2
// is worth keeping, otherwise evict first.
__device__ __forceinline__ constexpr std::uint64_t get_tma_cache_hint(
    bool evict_first) {
    return static_cast<std::uint64_t>(
        evict_first ? cute::TMA::CacheHintSm90::EVICT_FIRST
                    : cute::TMA::CacheHintSm90::EVICT_NORMAL);
}

// Issue one 1-D bulk copy (a contiguous byte range, no tensor map), tracked
// by an mbarrier.
__device__ static __forceinline__ void tma_load_1d(const void* src_gmem,
                                                   std::uint32_t bytes,
                                                   void* dst_smem,
                                                   std::uint64_t* bar_smem) {
    cute::SM90_BULK_COPY_G2S::copy(src_gmem, bar_smem, dst_smem,
                                   static_cast<std::int32_t>(bytes));
}

// Issue one 2D TMA bulk-tensor load, tracked by an mbarrier.
__device__ static __forceinline__ void tma_load_2d(
    CUtensorMap const& tma_desc, std::uint32_t coord0, std::uint32_t coord1,
    void* dst_smem, std::uint64_t* bar_smem,
    std::uint64_t cache_hint =
        static_cast<std::uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL)) {
    cute::SM90_TMA_LOAD_2D::copy(&tma_desc, bar_smem, cache_hint, dst_smem,
                                 static_cast<std::int32_t>(coord0),
                                 static_cast<std::int32_t>(coord1));
}

// Issue the full per-block BF16 input TMA load: one 64-k SW128 box per
// issue, box `kk` into `dest[kk]`.
template <std::size_t NumRows, std::size_t NumSwizzleBoxesK,
          std::size_t SwizzleBoxK>
__device__ __forceinline__ void load_input_tile(
    CUtensorMap const& tma_input_desc, std::uint32_t tile,
    std::uint32_t token_start,
    DataType::ElementA (&dest)[NumSwizzleBoxesK][NumRows][SwizzleBoxK],
    std::uint64_t* bar_smem_ptr) {
    // The box is one 8-row swizzle atom, so the ring's rows are whole atoms.
    constexpr std::uint32_t kRowGroups = NumRows / 8u;
    static_assert(NumRows % 8u == 0u, "the ring is whole swizzle atoms");
#pragma unroll
    for (std::uint32_t kk = 0u; kk < NumSwizzleBoxesK; ++kk) {
#pragma unroll
        for (std::uint32_t g = 0u; g < kRowGroups; ++g) {
            const std::uint32_t k_start =
                tile * (NumSwizzleBoxesK * SwizzleBoxK) + kk * SwizzleBoxK;
            tma_load_2d(tma_input_desc, /*coord0=*/k_start,
                        /*coord1=*/token_start + 8u * g,
                        /*dest_smem_ptr=*/&dest[kk][8u * g][0],
                        /*bar_smem_ptr=*/bar_smem_ptr);
        }
    }
}

// Create a 2-D TMA descriptor for a dense row-major `[rows, cols]` tensor of
// `T`; the box is the per-copy tile.
template <typename T>
inline CUtensorMap CreateTmaDesc2D(const char* debug_name, const T* global_ptr,
                                   std::uint32_t rows, std::uint32_t cols,
                                   std::uint32_t box_rows,
                                   std::uint32_t box_cols,
                                   bool use_swizzle = true) {
    CUtensorMapDataType data_type = CU_TENSOR_MAP_DATA_TYPE_UINT8;
    if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        data_type = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
    } else if constexpr (std::is_same_v<T, float>) {
        data_type = CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
    } else {
        static_assert(sizeof(T) == 1u, "unsupported TMA element type");
    }
    CUtensorMap desc{};
    uint64_t global_shape[2] = {cols, rows};
    uint64_t global_strides[1] = {static_cast<uint64_t>(cols) * sizeof(T)};
    uint32_t box_dim[2] = {box_cols, box_rows};
    uint32_t element_strides[2] = {1u, 1u};
    const CUresult res = cuTensorMapEncodeTiled(
        &desc, data_type, 2,
        const_cast<void*>(static_cast<const void*>(global_ptr)), global_shape,
        global_strides, box_dim, element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
        use_swizzle ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(res == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed for ",
                debug_name, ": CUresult=", static_cast<int>(res),
                " (rows=", rows, ", cols=", cols, ", box=(", box_rows, ", ",
                box_cols, "))");
    return desc;
}

}  // namespace halops::moe
