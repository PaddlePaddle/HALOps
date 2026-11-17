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
#include <cutlass/tfloat32.h>

#include <cute/config.hpp>

#include "common/ptx.cuh"

namespace halops::mhc {

CUTE_DEVICE float bf16_lo(uint32_t u) { return __uint_as_float(u << 16); }

CUTE_DEVICE float bf16_hi(uint32_t u) {
    return __uint_as_float(u & 0xffff0000u);
}

CUTE_DEVICE uint32_t pack_bf16_rn(float lo, float hi) {
    const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return reinterpret_cast<const uint32_t&>(v);
}

CUTE_DEVICE float2 bf16x2_to_float2(uint32_t u) {
    return make_float2(bf16_lo(u), bf16_hi(u));
}

CUTE_DEVICE float2 splat(float v) { return make_float2(v, v); }

CUTE_DEVICE void split_bf16x2(float2 f, uint32_t (&b)[2]) {
    const float r0 =
        static_cast<float>(cutlass::tfloat32_t::round_half_ulp_truncate(f.x));
    const float r1 =
        static_cast<float>(cutlass::tfloat32_t::round_half_ulp_truncate(f.y));
    b[0] = pack_bf16_rn(r0, r1);
    b[1] = pack_bf16_rn(__fsub_rn(r0, bf16_lo(b[0])),
                        __fsub_rn(r1, bf16_hi(b[0])));
}

CUTE_DEVICE float sigmoid(float z) { return 1.0f / (1.0f + expf(0.0f - z)); }

template <int kN, bool kFirst>
CUTE_DEVICE void sinkhorn_row_pass(float2 (&v)[kN * kN / 2], float eps) {
    constexpr int kH = kN / 2;
#pragma unroll
    for (int i = 0; i < kN; ++i) {
        float2 a[kH];
#pragma unroll
        for (int h = 0; h < kH; ++h) a[h] = v[kH * i + h];
#pragma unroll
        for (int w = kH / 2; w > 0; w >>= 1)
#pragma unroll
            for (int h = 0; h < w; ++h) a[h] = __fadd2_rn(a[h], a[h + w]);
        const float s = __fadd_rn(a[0].x, a[0].y);
        const float inv = rcp_approx(kFirst ? s : __fadd_rn(s, eps));
        const float2 inv2 = splat(inv);
#pragma unroll
        for (int h = 0; h < kH; ++h)
            v[kH * i + h] = kFirst ? __ffma2_rn(v[kH * i + h], inv2, splat(eps))
                                   : __fmul2_rn(v[kH * i + h], inv2);
    }
}

template <int kN>
CUTE_DEVICE void sinkhorn_col_pass(float2 (&v)[kN * kN / 2], float eps) {
    constexpr int kH = kN / 2;
#pragma unroll
    for (int h = 0; h < kH; ++h) {
        float2 a[kN];
#pragma unroll
        for (int i = 0; i < kN; ++i) a[i] = v[kH * i + h];
#pragma unroll
        for (int w = kN / 2; w > 0; w >>= 1)
#pragma unroll
            for (int i = 0; i < w; ++i) a[i] = __fadd2_rn(a[i], a[i + w]);
        const float2 se = __fadd2_rn(a[0], splat(eps));
        const float2 inv = make_float2(rcp_approx(se.x), rcp_approx(se.y));
#pragma unroll
        for (int i = 0; i < kN; ++i)
            v[kH * i + h] = __fmul2_rn(v[kH * i + h], inv);
    }
}

template <int kN, bool kAlignFleet>
CUTE_DEVICE void sinkhorn(float (&c)[kN * kN], float eps, int repeat) {
    static_assert(kN >= 2 && (kN & (kN - 1)) == 0);
    constexpr float kLog2e = 1.4426950408889634f;
#pragma unroll
    for (int i = 0; i < kN; ++i) {
        float m[kN];
#pragma unroll
        for (int j = 0; j < kN; ++j) m[j] = c[i * kN + j];
#pragma unroll
        for (int w = 1; w < kN; w <<= 1)
#pragma unroll
            for (int j = 0; j < kN; j += 2 * w) m[j] = fmaxf(m[j], m[j + w]);
        const float rmax = m[0];
#pragma unroll
        for (int j = 0; j < kN; ++j) {
            if constexpr (kAlignFleet)
                c[i * kN + j] = exp2f((c[i * kN + j] - rmax) * kLog2e);
            else
                c[i * kN + j] = expf(c[i * kN + j] - rmax);
        }
    }
    float2 v[kN * kN / 2];
#pragma unroll
    for (int k = 0; k < kN * kN / 2; ++k)
        v[k] = make_float2(c[2 * k], c[2 * k + 1]);
    sinkhorn_row_pass<kN, true>(v, eps);
    sinkhorn_col_pass<kN>(v, eps);
#pragma unroll 1
    for (int iter = 1; iter < repeat; ++iter) {
        sinkhorn_row_pass<kN, false>(v, eps);
        sinkhorn_col_pass<kN>(v, eps);
    }
#pragma unroll
    for (int k = 0; k < kN * kN / 2; ++k) {
        c[2 * k] = v[k].x;
        c[2 * k + 1] = v[k].y;
    }
}

}  // namespace halops::mhc
