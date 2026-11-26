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
#include <cutlass/arch/barrier.h>

#include <cute/tensor.hpp>

namespace halops::mla {
template <typename TMA, typename Tensor0, typename Tensor1>
CUTE_DEVICE void launch_tma_copy(const TMA& tma_copy, Tensor0 src, Tensor1 dst,
                                 cutlass::arch::ClusterTransactionBarrier& bar,
                                 const cute::TMA::CacheHintSm90& cache_hint =
                                     cute::TMA::CacheHintSm90::EVICT_NORMAL) {
    auto thr_tma = tma_copy.get_slice(cute::_0{});
    cute::copy(
        tma_copy.with(
            reinterpret_cast<
                typename cutlass::arch::ClusterTransactionBarrier::ValueType&>(
                bar),
            0, cache_hint),
        thr_tma.partition_S(src), thr_tma.partition_D(dst));
}

CUTE_DEVICE
void tma_gather4(const void* desc_ptr,
                 cutlass::arch::ClusterTransactionBarrier& mbar_ptr,
                 void* smem_ptr, int col_idx, int4 row_idxs,
                 int64_t cache_hint) {
    uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
    uint32_t mbar_addr = cute::cast_smem_ptr_to_uint(&mbar_ptr);
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::"
        "complete_tx::bytes.cta_group::1.L2::cache_hint [%0], [%1, {%2, %3, "
        "%4, %5, %6}], [%7], %8;\n"
        :
        : "r"(smem_addr), "l"(desc_ptr), "r"(col_idx), "r"(row_idxs.x),
          "r"(row_idxs.y), "r"(row_idxs.z), "r"(row_idxs.w), "r"(mbar_addr),
          "l"(cache_hint)
        : "memory");
}

// tma_gather4 with a shared::cluster destination and no cta_group.
CUTE_DEVICE void tma_gather4_cluster(const void* desc_ptr, uint64_t* mbar_ptr,
                                     void* smem_ptr, int col_idx, int4 row_idxs,
                                     uint64_t cache_hint) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4.mbarrier:"
        ":complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, %3, %4, %5, %6}], "
        "[%7], %8;" ::"r"(cute::cast_smem_ptr_to_uint(smem_ptr)),
        "l"(desc_ptr), "r"(col_idx), "r"(row_idxs.x), "r"(row_idxs.y),
        "r"(row_idxs.z), "r"(row_idxs.w),
        "r"(cute::cast_smem_ptr_to_uint(mbar_ptr)), "l"(cache_hint)
        : "memory");
}

CUTE_DEVICE
void tma_load_2d(const void* desc_ptr,
                 cutlass::arch::ClusterTransactionBarrier& mbar, void* smem_ptr,
                 int crd0, int crd1, int64_t cache_hint) {
    uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
    uint32_t mbar_addr = cute::cast_smem_ptr_to_uint(&mbar);
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::"
        "bytes.L2::cache_hint"
        " [%0], [%1, {%3, %4}], [%2], %5;\n"
        :
        : "r"(smem_addr), "l"(desc_ptr), "r"(mbar_addr), "r"(crd0), "r"(crd1),
          "l"(cache_hint)
        : "memory");
}

// TMA tile loads without an L2 cache hint (cute::SM90_TMA_LOAD_2D/3D always
// pass one).
CUTE_DEVICE void tma_load_2d_no_hint(const void* desc_ptr, uint64_t* mbar_ptr,
                                     void* smem_ptr, int crd0, int crd1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::"
        "complete_tx::bytes [%0], [%1, {%2, %3}], [%4];" ::"r"(
            cute::cast_smem_ptr_to_uint(smem_ptr)),
        "l"(desc_ptr), "r"(crd0), "r"(crd1),
        "r"(cute::cast_smem_ptr_to_uint(mbar_ptr))
        : "memory");
}

CUTE_DEVICE
void tma_load_3d_no_hint(const void* desc_ptr, uint64_t* mbar_ptr,
                         void* smem_ptr, int crd0, int crd1, int crd2) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::"
        "complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];" ::"r"(
            cute::cast_smem_ptr_to_uint(smem_ptr)),
        "l"(desc_ptr), "r"(crd0), "r"(crd1), "r"(crd2),
        "r"(cute::cast_smem_ptr_to_uint(mbar_ptr))
        : "memory");
}

CUTE_DEVICE
void prefetch_l2_bulk(const void* ptr, uint32_t bytes) {
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(ptr),
                 "r"(bytes)
                 : "memory");
}

CUTE_DEVICE
void mbar_arrive_cluster(uint32_t bar_cluster) {
    asm volatile(
        "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" ::"r"(
            bar_cluster)
        : "memory");
}

CUTE_DEVICE
void mbar_arrive_cluster_relaxed(uint32_t bar_cluster) {
    asm volatile(
        "mbarrier.arrive.relaxed.cluster.shared::cluster.b64 _, [%0];" ::"r"(
            bar_cluster)
        : "memory");
}

CUTE_DEVICE
void bulk_copy_to_peer(uint32_t dst_cluster, uint32_t src, uint32_t bytes,
                       uint32_t bar_cluster) {
    asm volatile(
        "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::"
        "bytes [%0], [%1], %2, [%3];" ::"r"(dst_cluster),
        "r"(src), "r"(bytes), "r"(bar_cluster)
        : "memory");
}

CUTE_DEVICE void bulk_wait_all() { bulk_store_wait_all(); }
}  // namespace halops::mla
