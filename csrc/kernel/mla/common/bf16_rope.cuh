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

namespace halops::mla {
template <int N>
CUTE_DEVICE void fma_bf16(float weight, const void* vals, float (&acc)[N]) {
    const auto* v = reinterpret_cast<const __nv_bfloat16*>(vals);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i)
        acc[i] = fmaf(weight, __bfloat162float(v[i]), acc[i]);
}

template <int N>
CUTE_DEVICE void pack_bf16(const float (&vals)[N], void* out) {
    auto* pairs = reinterpret_cast<__nv_bfloat162*>(out);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i)
        pairs[i] = __floats2bfloat162_rn(vals[2 * i], vals[2 * i + 1]);
}

template <int N>
CUTE_DEVICE void unpack_bf16(const void* in, float (&vals)[N]) {
    const auto* pairs = reinterpret_cast<const __nv_bfloat162*>(in);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
        const float2 v = __bfloat1622float2(pairs[i]);
        vals[2 * i] = v.x;
        vals[2 * i + 1] = v.y;
    }
}

template <int N = 8>
CUTE_DEVICE void rope_slice(const float* __restrict__ fr, int rb,
                            float (&cs)[N]) {
    constexpr int kVec = sizeof(float4) / sizeof(float);
    static_assert(N % kVec == 0, "whole float4 loads");
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < N / kVec; ++k)
        reinterpret_cast<float4*>(cs)[k] =
            reinterpret_cast<const float4*>(fr + rb)[k];
}

CUTE_DEVICE float2 rope_rotate(float x0, float x1, float cos_v, float sin_v) {
    return make_float2(x0 * cos_v - x1 * sin_v, x0 * sin_v + x1 * cos_v);
}
CUTE_DEVICE float2 rope_unrotate(float x0, float x1, float cos_v, float sin_v) {
    return make_float2(x0 * cos_v + x1 * sin_v, -x0 * sin_v + x1 * cos_v);
}

template <int N>
CUTE_DEVICE void rope_rotate(float (&vals)[N], const float (&cs)[N]) {
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < N / 2; ++k) {
        const float2 r =
            rope_rotate(vals[2 * k], vals[2 * k + 1], cs[2 * k], cs[2 * k + 1]);
        vals[2 * k] = r.x;
        vals[2 * k + 1] = r.y;
    }
}

template <int N>
CUTE_DEVICE void inverse_rope_bf16(float (&vals)[N], const float (&cs)[N]) {
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < N / 2; ++k) {
        const float2 r = rope_unrotate(vals[2 * k], vals[2 * k + 1], cs[2 * k],
                                       cs[2 * k + 1]);
        const float2 rounded =
            __bfloat1622float2(__floats2bfloat162_rn(r.x, r.y));
        vals[2 * k] = rounded.x;
        vals[2 * k + 1] = rounded.y;
    }
}

template <int N>
CUTE_DEVICE float rms_inv(float sq_sum, float eps) {
    return rsqrtf(sq_sum / N + eps);
}
}  // namespace halops::mla
