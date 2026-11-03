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
#include <cuda_fp16.h>

#include <cstdint>

#include "ptx_utils.cuh"

namespace halops::moe {
namespace tail {

struct alignas(16) Vec16 {
    __nv_bfloat162 d[4];
    __device__ __forceinline__ __nv_bfloat162& operator[](int i) {
        return d[i];
    }
    __device__ __forceinline__ const __nv_bfloat162& operator[](int i) const {
        return d[i];
    }
    __device__ __forceinline__ void fill(const __nv_bfloat162& v) {
#pragma unroll
        for (int i = 0; i < 4; ++i) d[i] = v;
    }
};

// Conversions (round-to-nearest-even on the way down)
template <typename To, typename From>
__device__ __forceinline__ To cast(const From&);
template <>
__device__ __forceinline__ float2
cast<float2, __nv_bfloat162>(const __nv_bfloat162& v) {
    return __bfloat1622float2(v);
}
template <>
__device__ __forceinline__ __nv_bfloat162
cast<__nv_bfloat162, float2>(const float2& v) {
    return __float22bfloat162_rn(v);
}

// +0.0 of the element type: the "empty slot" value of the Lamport protocol
template <typename T>
__device__ __forceinline__ T get_pos_zero();
template <>
__device__ __forceinline__ __nv_bfloat16 get_pos_zero<__nv_bfloat16>() {
    const uint16_t z = 0x0000u;
    return *reinterpret_cast<const __nv_bfloat16*>(&z);
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, off);
    return v;
}

using vec_t = Vec16;

template <int WORLD, int L, int H, int MAXM>
struct Params {
    uint8_t* uc_base[WORLD];
    uint8_t* mc_base;
    uint32_t* epochs;
    const __nv_bfloat16* shared;
    const uint8_t* fc2_w8;
    const float* fc2_scale;
    const __nv_bfloat16* norm_weight;
    __nv_bfloat16* out;
    float eps;
    uint32_t m;
    uint32_t rank;
    const int32_t* valid_counts;
    uint32_t slot_rows;
};

template <int WORLD, int L, int H, int MAXM>
__device__ __forceinline__ bool row_valid(const Params<WORLD, L, H, MAXM>& p,
                                          uint32_t t) {
    if (t >= p.m) return false;
    if (p.valid_counts == nullptr) return true;
    return (t % p.slot_rows) <
           uint32_t(__ldg(p.valid_counts + t / p.slot_rows));
}

template <int WORLD, int H, int MAXM>
__device__ __forceinline__ int64_t slot_elems(int round, int parity, int src) {
    return (int64_t((round * 2 + parity) * WORLD + src)) * MAXM * H;
}

constexpr uint32_t kNegZeroPair =
    0x8000u;  // {-0.0,+0.0}: sum-neutral, non-zero

__device__ __forceinline__ void sanitize(vec_t& v) {
    auto& b = *reinterpret_cast<uint4*>(&v);
    if (b.x == 0) b.x = kNegZeroPair;
    if (b.y == 0) b.y = kNegZeroPair;
    if (b.z == 0) b.z = kNegZeroPair;
    if (b.w == 0) b.w = kNegZeroPair;
}

template <int WORLD, int L, int H, int MAXM>
__device__ __forceinline__ void push_only(const Params<WORLD, L, H, MAXM>& p,
                                          const vec_t& mine, int round,
                                          int parity, int t, int Wround,
                                          uint32_t vid) {
    const int64_t tok = int64_t(t) * Wround;
    vec_t v = mine;
    sanitize(v);
    const int64_t base =
        slot_elems<WORLD, H, MAXM>(round, parity, p.rank) + tok;
    st_multimem_16B(v, p.mc_base + base * 2, vid);
}

template <int WORLD, int L, int H, int MAXM>
__device__ __forceinline__ vec_t poll_reduce(const Params<WORLD, L, H, MAXM>& p,
                                             int round, int parity, int t,
                                             int Wround, uint32_t vid) {
    const int64_t tok = int64_t(t) * Wround;
    vec_t acc[WORLD];
    uint8_t* local = p.uc_base[p.rank];
    do {
        bool has_zero = false;
#pragma unroll
        for (int s = 0; s < WORLD; ++s) {
            const int64_t base =
                slot_elems<WORLD, H, MAXM>(round, parity, s) + tok;
            ld_relaxed_16B(acc[s], local + base * 2, vid);
            const auto b = *reinterpret_cast<const uint4*>(&acc[s]);
            has_zero |= (b.x == 0) | (b.y == 0) | (b.z == 0) | (b.w == 0);
        }
        if (!has_zero) break;
    } while (true);
    vec_t zero;
    zero.fill(__nv_bfloat162{get_pos_zero<__nv_bfloat16>(),
                             get_pos_zero<__nv_bfloat16>()});
#pragma unroll
    for (int s = 0; s < WORLD; ++s) {
        const int64_t base = slot_elems<WORLD, H, MAXM>(round, parity, s) + tok;
        st_global_16B(zero, local + base * 2, vid);
    }
    vec_t r;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float2 a{0.f, 0.f};
#pragma unroll
        for (int s = 0; s < WORLD; ++s) {
            const auto f = cast<float2>(acc[s][j]);
            a.x += f.x;
            a.y += f.y;
        }
        r[j] = cast<__nv_bfloat162>(a);
    }
    return r;
}

// One-shot push AllReduce
template <int WORLD, int L, int H, int MAXM>
__device__ __forceinline__ vec_t push_reduce(const Params<WORLD, L, H, MAXM>& p,
                                             const vec_t& mine, int round,
                                             int parity, int t, int Wround,
                                             uint32_t vid) {
    push_only<WORLD, L, H, MAXM>(p, mine, round, parity, t, Wround, vid);
    return poll_reduce<WORLD, L, H, MAXM>(p, round, parity, t, Wround, vid);
}

}  // namespace tail
}  // namespace halops::moe
