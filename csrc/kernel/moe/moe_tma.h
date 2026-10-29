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

#include <cstddef>
#include <cstdint>
#include <cute/arch/copy_sm90_tma.hpp>

#include "moe_interface.h"

// TMA descriptor encoding requires CUDA toolkit 12.0+.
#if defined(CUDA_VERSION) && (CUDA_VERSION < 12000)
#    error \
        "moe_tma.h requires CUDA 12.0 or later for CUtensorMap / cuTensorMapEncodeTiled"
#endif

namespace halops::moe {

// Build a `CUtensorMap` for the fp8 up-projection weight `[E, 2*n, k]`.
CUtensorMap create_up_weight_tma_desc(const void* weights_ptr,
                                      uint32_t num_experts, uint32_t n,
                                      uint32_t k, uint32_t row_box = 128u);

// Build a `CUtensorMap` for the bf16 activation `[BS, K_hidden]`.
CUtensorMap create_activations_tma_desc(const void* activations_ptr,
                                        uint32_t batch_size_cap,
                                        uint32_t block_rows, uint32_t K_hidden);

// Build a `CUtensorMap` for the fp8 down-projection weight `[E, k, n]`.
CUtensorMap create_down_weight_tma_desc(const void* weights_ptr,
                                        uint32_t num_experts, uint32_t k,
                                        uint32_t n, uint32_t row_box = 128u);

// Build a `CUtensorMap` for the fp8 intermediate activation.
CUtensorMap create_down_activation_tma_desc(const void* activations_ptr,
                                            uint32_t temp_rows,
                                            uint32_t block_rows, uint32_t n);

// L2 eviction hint for the expert weights, streamed once per launch: an
// active set that fits in L2 is worth keeping, otherwise evict first.
__device__ __forceinline__ constexpr std::uint64_t expert_weight_cache_hint(
    bool evict_first) {
    return static_cast<std::uint64_t>(
        evict_first ? cute::TMA::CacheHintSm90::EVICT_FIRST
                    : cute::TMA::CacheHintSm90::EVICT_NORMAL);
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

// Issue one non-tensor TMA bulk copy, tracked by an mbarrier.
__device__ static __forceinline__ void tma_load_bulk(const void* src_gmem,
                                                     std::uint32_t bytes,
                                                     void* dst_smem,
                                                     std::uint64_t* bar_smem) {
    cute::SM90_BULK_COPY_G2S::copy(src_gmem, bar_smem, dst_smem,
                                   static_cast<std::int32_t>(bytes));
}

// Issue one TMA load for one 128-row up-projection gate/up pair tile.
__device__ __forceinline__ void tma_load_up_tcgen05_pair_tile(
    CUtensorMap const& tma_up_weights_desc, std::uint32_t expert_id,
    std::uint32_t n, std::uint32_t row_base, std::uint32_t k_start,
    void* dest_slot_smem_ptr, std::uint64_t* bar_smem_ptr,
    std::uint64_t cache_hint =
        static_cast<std::uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL)) {
    const std::uint32_t global_row = expert_id * 2u * n + row_base;
    tma_load_2d(tma_up_weights_desc, /*coord0=*/k_start, /*coord1=*/global_row,
                dest_slot_smem_ptr, bar_smem_ptr, cache_hint);
}

// Issue one TMA load for the bf16 activation tile used by streaming quantize.
__device__ __forceinline__ void tma_load_bf16_input_tile(
    CUtensorMap const& tma_input_desc, std::uint32_t k_start,
    std::uint32_t token_start, void* dest_smem_ptr,
    std::uint64_t* bar_smem_ptr) {
    // coord0 = k, coord1 = token; TMA zero-fills rows past `batch_size`.
    tma_load_2d(tma_input_desc, /*coord0=*/k_start, /*coord1=*/token_start,
                dest_smem_ptr, bar_smem_ptr);
}

// Issue one bulk load of router weight tile (`m_tile`, `k_tile`).
__device__ __forceinline__ void tma_load_router_weight_tile(
    const __nv_bfloat16* router_weight, std::uint32_t m_tile,
    std::uint32_t k_tile, std::uint32_t num_tiles_k, std::uint32_t tile_bytes,
    void* dest_smem_ptr, std::uint64_t* bar_smem_ptr) {
    const char* src =
        reinterpret_cast<const char*>(router_weight) +
        static_cast<std::size_t>(m_tile * num_tiles_k + k_tile) * tile_bytes;
    tma_load_bulk(src, tile_bytes, dest_smem_ptr, bar_smem_ptr);
}

// Issue one TMA load for the full 128×128 fp8 down-projection weight tile.
__device__ __forceinline__ void tma_load_down_umma_tile(
    CUtensorMap const& tma_down_weights_desc, std::uint32_t expert_id,
    std::uint32_t k, std::uint32_t base_col, std::uint32_t k_start,
    void* dest_smem_ptr, std::uint64_t* bar_smem_ptr,
    std::uint64_t cache_hint =
        static_cast<std::uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL)) {
    // coord0 = n, coord1 = row; the row is `expert_id * k + base_col`.
    const std::uint32_t global_row = expert_id * k + base_col;
    tma_load_2d(tma_down_weights_desc, /*coord0=*/k_start,
                /*coord1=*/global_row, dest_smem_ptr, bar_smem_ptr, cache_hint);
}

// Build a `CUtensorMap` for the shared latent down weight `[m, k]` (fp8).
CUtensorMap create_latent_down_weight_tma_desc(const void* weights_ptr,
                                               uint32_t m, uint32_t k,
                                               uint32_t row_box);

// Issue one 128x128 fp8 weight tile of the latent down-projection into SHM.
__device__ __forceinline__ void tma_load_latent_down_weight_tile(
    CUtensorMap const& tma_latent_down_weights_desc, std::uint32_t base_row,
    std::uint32_t k_start, void* dest_smem_ptr, std::uint64_t* bar_smem_ptr) {
    tma_load_2d(tma_latent_down_weights_desc, /*coord0=*/k_start,
                /*coord1=*/base_row, dest_smem_ptr, bar_smem_ptr);
}

// Issue one TMA load for up to 8 rows of the fp8 down-projection activation.
__device__ __forceinline__ void tma_load_down_umma_activation_bulk(
    CUtensorMap const& tma_down_activations_desc, std::uint32_t k_start,
    std::uint32_t first_row, void* dest_smem_ptr, std::uint64_t* bar_smem_ptr) {
    // Descriptor axis order (innermost first): coord0 = n, coord1 = row.
    tma_load_2d(tma_down_activations_desc, /*coord0=*/k_start,
                /*coord1=*/first_row, dest_smem_ptr, bar_smem_ptr);
}

}  // namespace halops::moe
