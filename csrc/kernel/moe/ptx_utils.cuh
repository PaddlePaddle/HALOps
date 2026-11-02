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

#include <cstdint>
#include <cute/arch/copy_sm100.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>

namespace halops::moe {

// cp.async scalar helpers (sm_80+)
__device__ static inline void cp_async_4bytes(void* smem_dst,
                                              const void* gmem_src) {
    const std::uint32_t smem_addr =
        static_cast<std::uint32_t>(__cvta_generic_to_shared(smem_dst));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(smem_addr),
                 "l"(gmem_src));
}

__device__ static inline void cp_async_commit_group() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template <std::uint32_t N>
__device__ static inline void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
}

// Cluster-wide barrier: CTA-collective, release-arrive and acquire-wait.
__device__ static __forceinline__ void cluster_arrive_and_wait() {
    asm volatile("barrier.cluster.arrive.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
}

// The shared-memory address `ptr` has in cluster block `rank`.
__device__ static __forceinline__ std::uint32_t cluster_map_shared_rank(
    void const* ptr, std::uint32_t rank) {
    const std::uint32_t local =
        static_cast<std::uint32_t>(__cvta_generic_to_shared(ptr));
    std::uint32_t remote;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n"
                 : "=r"(remote)
                 : "r"(local), "r"(rank));
    return remote;
}

__device__ static __forceinline__ float cluster_load_f32(
    std::uint32_t mapped_addr) {
    float v;
    asm volatile("ld.shared::cluster.f32 %0, [%1];\n"
                 : "=f"(v)
                 : "r"(mapped_addr)
                 : "memory");
    return v;
}

__device__ static __forceinline__ uint4
cluster_load_v4(std::uint32_t mapped_addr) {
    uint4 v;
    asm volatile("ld.shared::cluster.v4.u32 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "r"(mapped_addr)
                 : "memory");
    return v;
}

// Device-scope acquire load of a 32-bit word (ld.acquire.gpu).
__device__ static __forceinline__ uint32_t
global_load_acquire_u32(const uint32_t* addr) {
    uint32_t v;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];"
                 : "=r"(v)
                 : "l"(addr)
                 : "memory");
    return v;
}

__device__ static __forceinline__ std::uint64_t global_load_acquire_u64(
    const std::uint64_t* addr) {
    std::uint64_t v;
    asm volatile("ld.acquire.gpu.global.u64 %0, [%1];"
                 : "=l"(v)
                 : "l"(addr)
                 : "memory");
    return v;
}

__device__ static __forceinline__ void global_red_add_u64(std::uint64_t* addr,
                                                          std::uint64_t val) {
    asm volatile("red.global.add.u64 [%0], %1;" ::"l"(addr), "l"(val)
                 : "memory");
}

// Device-scope release store of a 32-bit word (st.release.gpu).
__device__ static __forceinline__ void global_store_release_u32(uint32_t* addr,
                                                                uint32_t v) {
    asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(addr), "r"(v)
                 : "memory");
}

// Device-scope release OR-reduction into a 32-bit word.
__device__ static __forceinline__ void global_red_or_release_u32(
    uint32_t* addr, uint32_t bits) {
    asm volatile("red.release.gpu.global.or.b32 [%0], %1;" ::"l"(addr),
                 "r"(bits)
                 : "memory");
}

// CTA-scope release store / acquire load of a 32-bit SHM word.
__device__ static __forceinline__ void shared_store_release_u32(uint32_t* addr,
                                                                uint32_t v) {
    const std::uint32_t smem_addr =
        static_cast<std::uint32_t>(__cvta_generic_to_shared(addr));
    asm volatile("st.release.cta.shared::cta.u32 [%0], %1;" ::"r"(smem_addr),
                 "r"(v)
                 : "memory");
}

__device__ static __forceinline__ std::uint32_t shared_load_acquire_u32(
    const uint32_t* addr) {
    const std::uint32_t smem_addr = static_cast<std::uint32_t>(
        __cvta_generic_to_shared(const_cast<uint32_t*>(addr)));
    std::uint32_t v;
    asm volatile("ld.acquire.cta.shared::cta.u32 %0, [%1];"
                 : "=r"(v)
                 : "r"(smem_addr)
                 : "memory");
    return v;
}

// CTA-scope release ADD-reduction into a 32-bit SHM word.
__device__ static __forceinline__ void shared_red_add_release_u32(
    uint32_t* addr, uint32_t v) {
    const std::uint32_t smem_addr =
        static_cast<std::uint32_t>(__cvta_generic_to_shared(addr));
    asm volatile(
        "red.release.cta.shared::cta.add.u32 [%0], %1;" ::"r"(smem_addr), "r"(v)
        : "memory");
}

// Device-scope relaxed ADD-reduction into a 32-bit word.
__device__ static __forceinline__ void global_red_add_u32(uint32_t* addr,
                                                          uint32_t v) {
    asm volatile("red.relaxed.gpu.global.add.u32 [%0], %1;" ::"l"(addr), "r"(v)
                 : "memory");
}

// Device-scope release ADD-reduction into a 32-bit word.
__device__ static __forceinline__ void global_red_add_release_u32(
    uint32_t* addr, uint32_t v) {
    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" ::"l"(addr), "r"(v)
                 : "memory");
}

// Allocate `num_columns` TMEM columns; issued by ONE fully-active warp.
__device__ static __forceinline__ void tmem_alloc(
    std::uint32_t num_columns, std::uint32_t* dst_slot_smem) {
    cute::TMEM::Allocator1Sm alloc;
    alloc.allocate(static_cast<int>(num_columns), dst_slot_smem);
    alloc.release_allocation_lock();
}

__device__ static __forceinline__ void tmem_free(std::uint32_t tmem_addr,
                                                 std::uint32_t num_columns) {
    cute::TMEM::Allocator1Sm alloc;
    alloc.free(tmem_addr, static_cast<int>(num_columns));
}

// TMEM address: lane (the MMA M index) in the high 16 bits, column low.
__device__ static __forceinline__ std::uint32_t tmem_addr(
    std::uint32_t base, std::uint32_t lane_base, std::uint32_t column) {
    return base + (lane_base << 16) + column;
}

// Make prior `tcgen05` / TMEM accesses visible after a block-wide sync.
__device__ static __forceinline__ void tcgen05_fence_after_thread_sync() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

// Order prior `tcgen05` / TMEM accesses before a following thread sync.
__device__ static __forceinline__ void tcgen05_fence_before_thread_sync() {
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}

// Read 8 accumulator columns per lane from TMEM (tcgen05.ld.32x32b.x8).
__device__ static __forceinline__ void tmem_ld_x8(std::uint32_t addr,
                                                  float (&d)[8]) {
    std::uint32_t v[8];
    cute::SM100::TMEM::LOAD::SM100_TMEM_LOAD_32dp32b8x::copy(
        addr, v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);
#pragma unroll
    for (int i = 0; i < 8; ++i) d[i] = __int_as_float(v[i]);
}

// A lane's `N` accumulator columns as `N / 8` 8-wide reads.
template <std::uint32_t N>
__device__ static __forceinline__ void tmem_ld_tokens(std::uint32_t addr,
                                                      float (&d)[N]) {
    static_assert(N % 8u == 0u, "whole 8-column reads");
#pragma unroll
    for (std::uint32_t i = 0; i < N / 8u; ++i) {
        float v[8];
        tmem_ld_x8(addr + i * 8u, v);
#pragma unroll
        for (std::uint32_t j = 0; j < 8u; ++j) d[i * 8u + j] = v[j];
    }
}

// Read a 16-lane x 8-column accumulator window per warp (16x256b.x1).
__device__ static __forceinline__ void tmem_ld_16x256b1x(std::uint32_t addr,
                                                         float (&d)[4]) {
    std::uint32_t v[4];
    cute::SM100::TMEM::LOAD::SM100_TMEM_LOAD_16dp256b1x::copy(addr, v[0], v[1],
                                                              v[2], v[3]);
#pragma unroll
    for (int i = 0; i < 4; ++i) d[i] = __int_as_float(v[i]);
}

}  // namespace halops::moe
