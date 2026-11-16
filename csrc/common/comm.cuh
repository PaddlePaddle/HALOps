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

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>

#include <cstdint>
#include <cute/config.hpp>

namespace halops::comm {

inline constexpr int kMaxWorld = 8;
inline constexpr uint16_t kLamportEmpty = 0x0000;
inline constexpr uint16_t kLamportNegZero = 0x8000;

struct LamportPlane {
    uint8_t* peers[kMaxWorld];
    uint32_t* counters;
    int64_t slot_bytes;
    int world;
    int rank;
    int num_counters;
};

template <typename V>
CUTE_DEVICE void ld_global_16B(V& x, const void* addr, int64_t vec_offset) {
    static_assert(alignof(V) == 16 && sizeof(V) == 16);
    addr = static_cast<const uint8_t*>(addr) + vec_offset * sizeof(V);
    uint4 val;
    asm volatile("ld.global.v4.b32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w)
                 : "l"(addr));
    x = *reinterpret_cast<const V*>(&val);
}

template <typename V>
CUTE_DEVICE void st_global_16B(const V& x, void* addr, int64_t vec_offset) {
    static_assert(alignof(V) == 16 && sizeof(V) == 16);
    const uint4 val = *reinterpret_cast<const uint4*>(&x);
    addr = static_cast<uint8_t*>(addr) + vec_offset * sizeof(V);
    asm volatile("st.global.v4.b32 [%4], {%0, %1, %2, %3};"
                 :
                 : "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w), "l"(addr));
}

template <typename V>
CUTE_DEVICE void ld_relaxed_16B(V& x, const void* addr, int64_t vec_offset) {
    static_assert(alignof(V) == 16 && sizeof(V) == 16);
    addr = static_cast<const uint8_t*>(addr) + vec_offset * sizeof(V);
    uint4 val;
    asm volatile("ld.relaxed.sys.global.v4.b32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w)
                 : "l"(addr));
    x = *reinterpret_cast<const V*>(&val);
}

template <typename V>
CUTE_DEVICE void st_relaxed_16B(const V& x, void* addr, int64_t vec_offset) {
    static_assert(alignof(V) == 16 && sizeof(V) == 16);
    const uint4 val = *reinterpret_cast<const uint4*>(&x);
    addr = static_cast<uint8_t*>(addr) + vec_offset * sizeof(V);
    asm volatile("st.relaxed.sys.global.v4.b32 [%4], {%0, %1, %2, %3};"
                 :
                 : "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w), "l"(addr));
}

CUTE_DEVICE void st_async_v4(uint32_t addr, float4 v, uint32_t bar) {
    asm volatile(
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 "
        "[%0], {%1, %2, %3, %4}, [%5];" ::"r"(addr),
        "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w), "r"(bar)
        : "memory");
}

CUTE_DEVICE void mbar_wait_cluster(uint32_t bar, uint32_t phase) {
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P, "
        "[%0], %1;\n"
        "@!P bra WAIT_%=;\n"
        "}\n" ::"r"(bar),
        "r"(phase)
        : "memory");
}

CUTE_DEVICE void cluster_arrive_relaxed() {
    asm volatile("barrier.cluster.arrive.relaxed;" ::: "memory");
}

CUTE_DEVICE void cluster_wait() {
    asm volatile("barrier.cluster.wait.acquire;" ::: "memory");
}

CUTE_DEVICE void lamport_mark(uint4& v) {
    auto* atoms = reinterpret_cast<uint16_t*>(&v);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 8; ++i) {
        if (atoms[i] == kLamportEmpty) atoms[i] = kLamportNegZero;
    }
}

CUTE_DEVICE bool lamport_empty(const uint4& v) {
    const auto* atoms = reinterpret_cast<const uint16_t*>(&v);
    bool empty = false;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 8; ++i) empty |= atoms[i] == kLamportEmpty;
    return empty;
}

CUTE_DEVICE int64_t lamport_phase_offset(const LamportPlane& p, int counter) {
    return int64_t(p.counters[counter] & 1) * p.world * p.slot_bytes;
}

CUTE_DEVICE void lamport_advance(const LamportPlane& p, int counter) {
    atomicAdd(p.counters + counter, 1u);
}

CUTE_DEVICE void lamport_push(const LamportPlane& p, int64_t phase_off,
                              int64_t vid, uint4 v) {
    lamport_mark(v);
    const int64_t src_off = p.rank * p.slot_bytes + phase_off;
    CUTLASS_PRAGMA_UNROLL
    for (int d = 0; d < kMaxWorld; ++d) {
        if (d < p.world) st_relaxed_16B(v, p.peers[d] + src_off, vid);
    }
}

CUTE_DEVICE void lamport_poll(const LamportPlane& p, int64_t phase_off,
                              int64_t vid, uint4 (&v)[kMaxWorld]) {
    const uint8_t* base = p.peers[p.rank] + phase_off;
    bool pending;
    do {
        pending = false;
        CUTLASS_PRAGMA_UNROLL
        for (int s = 0; s < kMaxWorld; ++s) {
            if (s < p.world) ld_relaxed_16B(v[s], base + s * p.slot_bytes, vid);
        }
        CUTLASS_PRAGMA_UNROLL
        for (int s = 0; s < kMaxWorld; ++s) {
            if (s < p.world) pending |= lamport_empty(v[s]);
        }
    } while (pending);
}

CUTE_DEVICE void lamport_reset(const LamportPlane& p, int64_t phase_off,
                               int64_t vid) {
    const uint4 empty{};
    uint8_t* base = p.peers[p.rank] + phase_off;
    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < kMaxWorld; ++s) {
        if (s < p.world) st_global_16B(empty, base + s * p.slot_bytes, vid);
    }
}

CUTE_DEVICE uint4 sum_bf16x8(const LamportPlane& p,
                             const uint4 (&v)[kMaxWorld]) {
    constexpr int kPairs = sizeof(uint4) / sizeof(__nv_bfloat162);
    float2 acc[kPairs];
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < kPairs; ++j) {
        acc[j] = __bfloat1622float2(
            reinterpret_cast<const __nv_bfloat162*>(&v[0])[j]);
    }
    CUTLASS_PRAGMA_UNROLL
    for (int s = 1; s < kMaxWorld; ++s) {
        if (s >= p.world) break;
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < kPairs; ++j) {
            acc[j] = __fadd2_rn(
                acc[j], __bfloat1622float2(
                            reinterpret_cast<const __nv_bfloat162*>(&v[s])[j]));
        }
    }
    uint4 out;
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < kPairs; ++j) {
        reinterpret_cast<__nv_bfloat162*>(&out)[j] =
            __float22bfloat162_rn(acc[j]);
    }
    return out;
}

}  // namespace halops::comm
