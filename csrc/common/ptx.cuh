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
#include <cutlass/cutlass.h>

#include <cute/config.hpp>
#include <utility>

namespace halops {

inline constexpr int kMatRows = 8;
inline constexpr int kQuad = 4;
inline constexpr int kBankBytes = 32 * sizeof(uint32_t);

CUTE_DEVICE uint32_t bf16x2_max_abs_xorsign(uint32_t a, uint32_t b) {
    uint32_t r;
    asm("max.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
    return r;
}

CUTE_DEVICE void quad_transpose_4x4(const uint4& in, int lane,
                                    uint32_t (&out)[4]) {
    const bool hi = lane & 2;
    const uint32_t a0 = __shfl_xor_sync(~0u, hi ? in.x : in.z, 2);
    const uint32_t a1 = __shfl_xor_sync(~0u, hi ? in.y : in.w, 2);
    const uint32_t x0 = hi ? a0 : in.x, x1 = hi ? a1 : in.y;
    const uint32_t x2 = hi ? in.z : a0, x3 = hi ? in.w : a1;
    const bool odd = lane & 1;
    const uint32_t b0 = __shfl_xor_sync(~0u, odd ? x0 : x1, 1);
    const uint32_t b1 = __shfl_xor_sync(~0u, odd ? x2 : x3, 1);
    out[0] = odd ? b0 : x0;
    out[1] = odd ? x1 : b0;
    out[2] = odd ? b1 : x2;
    out[3] = odd ? x3 : b1;
}

CUTE_DEVICE float tanh_approx(float x) {
    float t;
    asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(x));
    return t;
}

CUTE_DEVICE float2 tanh_approx(float2 x) {
    return make_float2(tanh_approx(x.x), tanh_approx(x.y));
}

CUTE_DEVICE void bulk_store_wait_all() {
    asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
}

namespace detail {
template <class Op, class Src, int N, size_t... I>
CUTE_DEVICE void copy_to_regs(const Src& src, uint32_t (&r)[N],
                              std::index_sequence<I...>) {
    Op::copy(src, r[I]...);
}
}  // namespace detail

template <class Op, class Src, int N>
CUTE_DEVICE void copy_to_regs(const Src& src, uint32_t (&r)[N]) {
    detail::copy_to_regs<Op>(src, r, std::make_index_sequence<N>{});
}

}  // namespace halops
