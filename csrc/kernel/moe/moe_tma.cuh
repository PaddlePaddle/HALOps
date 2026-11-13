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

// Retain weight working sets that fit in L2; otherwise evict first.
__device__ __forceinline__ constexpr std::uint64_t get_tma_cache_hint(
    bool evict_first) {
    return static_cast<std::uint64_t>(
        evict_first ? cute::TMA::CacheHintSm90::EVICT_FIRST
                    : cute::TMA::CacheHintSm90::EVICT_NORMAL);
}

// Prefetch a contiguous range into L2 without staging it in shared memory.
__device__ static __forceinline__ void tma_prefetch_l2_1d(const void* src_gmem,
                                                          std::uint32_t bytes) {
    cute::SM90_BULK_COPY_G2S::PREFETCH::copy(src_gmem,
                                             static_cast<std::int32_t>(bytes));
}

// Prefetch a 3D TMA box into L2 without staging it in shared memory.
__device__ static __forceinline__ void tma_prefetch_l2_3d(
    CUtensorMap const& tma_desc, std::uint32_t coord0, std::uint32_t coord1,
    std::uint32_t coord2) {
    cute::SM90_TMA_LOAD_3D::PREFETCH::copy(
        &tma_desc, static_cast<std::int32_t>(coord0),
        static_cast<std::int32_t>(coord1), static_cast<std::int32_t>(coord2));
}

// Issue an mbarrier-tracked contiguous bulk copy.
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

// Issue one 3D TMA bulk-tensor load, tracked by an mbarrier.
__device__ static __forceinline__ void tma_load_3d(
    CUtensorMap const& tma_desc, std::uint32_t coord0, std::uint32_t coord1,
    std::uint32_t coord2, void* dst_smem, std::uint64_t* bar_smem,
    std::uint64_t cache_hint =
        static_cast<std::uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL)) {
    cute::SM90_TMA_LOAD_3D::copy(&tma_desc, bar_smem, cache_hint, dst_smem,
                                 static_cast<std::int32_t>(coord0),
                                 static_cast<std::int32_t>(coord1),
                                 static_cast<std::int32_t>(coord2));
}

// Issue one 4D TMA bulk-tensor load, tracked by an mbarrier.
__device__ static __forceinline__ void tma_load_4d(
    CUtensorMap const& tma_desc, std::uint32_t coord0, std::uint32_t coord1,
    std::uint32_t coord2, std::uint32_t coord3, void* dst_smem,
    std::uint64_t* bar_smem,
    std::uint64_t cache_hint =
        static_cast<std::uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL)) {
    cute::SM90_TMA_LOAD_4D::copy(
        &tma_desc, bar_smem, cache_hint, dst_smem,
        static_cast<std::int32_t>(coord0), static_cast<std::int32_t>(coord1),
        static_cast<std::int32_t>(coord2), static_cast<std::int32_t>(coord3));
}

// Issue one 5D TMA bulk-tensor load, tracked by an mbarrier.
__device__ static __forceinline__ void tma_load_5d(
    CUtensorMap const& tma_desc, std::uint32_t coord0, std::uint32_t coord1,
    std::uint32_t coord2, std::uint32_t coord3, std::uint32_t coord4,
    void* dst_smem, std::uint64_t* bar_smem,
    std::uint64_t cache_hint =
        static_cast<std::uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL)) {
    cute::SM90_TMA_LOAD_5D::copy(
        &tma_desc, bar_smem, cache_hint, dst_smem,
        static_cast<std::int32_t>(coord0), static_cast<std::int32_t>(coord1),
        static_cast<std::int32_t>(coord2), static_cast<std::int32_t>(coord3),
        static_cast<std::int32_t>(coord4));
}

// Input tile box: four 128-byte K chunks.
inline constexpr std::uint32_t kInputChunksBoxChunks = 4u;

template <std::size_t NumRows, std::size_t NumSwizzleBoxesK,
          std::size_t SwizzleBoxK>
__device__ __forceinline__ void load_input_tile_chunks(
    CUtensorMap const& tma_input_desc, std::uint32_t tile,
    std::uint32_t token_start,
    DataType::ElementA (&dest)[NumSwizzleBoxesK][NumRows][SwizzleBoxK],
    std::uint64_t* bar_smem_ptr) {
    static_assert(NumSwizzleBoxesK == kInputChunksBoxChunks &&
                      SwizzleBoxK * sizeof(DataType::ElementA) == 128u,
                  "the tile must be the descriptor's box");
    tma_load_3d(tma_input_desc, /*coord0=*/0u, /*coord1=*/token_start,
                /*coord2=*/tile * NumSwizzleBoxesK, &dest[0][0][0],
                bar_smem_ptr);
}

// The CUtensorMap element type of T. FP4 dimensions count values.
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

// Create a rank-R row-major tiled TMA descriptor.
template <int Rank, typename T>
inline CUtensorMap CreateTmaDesc(const char* debug_name, const T* global_ptr,
                                 const std::uint64_t (&shape)[Rank],
                                 const std::uint64_t (&strides)[Rank - 1],
                                 const std::uint32_t (&box)[Rank],
                                 bool use_swizzle = true) {
    static_assert(Rank >= 2 && Rank <= 5, "TMA supports rank 2..5 here");
    static constexpr std::uint32_t kUnitStride[5] = {1u, 1u, 1u, 1u, 1u};
    CUtensorMap desc{};
    const CUresult res = cuTensorMapEncodeTiled(
        &desc, TmaDataType<T>(), Rank,
        const_cast<void*>(static_cast<const void*>(global_ptr)), shape, strides,
        box, kUnitStride, CU_TENSOR_MAP_INTERLEAVE_NONE,
        use_swizzle ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(res == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed for ",
                debug_name, ": CUresult=", static_cast<int>(res));
    return desc;
}

}  // namespace halops::moe
