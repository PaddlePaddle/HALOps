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
#include <cuda_fp8.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/numeric_types.h>

#include <cute/tensor.hpp>

#include "kernel/mla/common/bf16_rope.cuh"
#include "kernel/mla/common/cta_sync.cuh"
#include "kernel/mla/common/fp8_quant.cuh"
#include "kernel/mla/common/intrinsics.cuh"
#include "kernel/mla/common/kv_runs.cuh"
#include "kernel/mla/common/kv_span.cuh"
#include "kernel/mla/common/traits.cuh"

namespace halops::mla {
struct Ue8m0Scale {
    float biased_exp;
    float multiplier;
};
CUTE_DEVICE Ue8m0Scale ue8m0_scale(float amax) {
    const float biased_exp = ue8m0_exp_ceil(amax);
    return {biased_exp, ue8m0_quant_scale(biased_exp)};
}

template <int N>
CUTE_DEVICE float abs_max(const float (&vals)[N]) {
    float amax = Fp8Format::AmaxFloor;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) amax = fmaxf(amax, fabsf(vals[i]));
    return amax;
}

template <int Lanes>
CUTE_DEVICE float lane_group_max(float value, unsigned mask = 0xffffffffu) {
    CUTLASS_PRAGMA_UNROLL
    for (int offset = 1; offset < Lanes; offset <<= 1)
        value = fmaxf(value, __shfl_xor_sync(mask, value, offset));
    return value;
}

CUTE_DEVICE uint32_t pack_e4m3x4(float v0, float v1, float v2, float v3,
                                 float scale) {
    const float2 lo = make_float2(v0 * scale, v1 * scale);
    const float2 hi = make_float2(v2 * scale, v3 * scale);
    return (uint32_t)__nv_cvt_float2_to_fp8x2(lo, __NV_SATFINITE, __NV_E4M3) |
           ((uint32_t)__nv_cvt_float2_to_fp8x2(hi, __NV_SATFINITE, __NV_E4M3)
            << Fp8Format::PairBits);
}

template <int N>
CUTE_DEVICE void pack_e4m3(const float (&vals)[N], float scale,
                           uint32_t (&words)[N / 4]) {
    constexpr int kW = Fp8Format::PerWord;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / kW; ++i)
        words[i] = pack_e4m3x4(vals[kW * i], vals[kW * i + 1], vals[kW * i + 2],
                               vals[kW * i + kW - 1], scale);
}

template <int Lanes, int N>
CUTE_DEVICE Ue8m0Scale quantize_group(const float (&vals)[N],
                                      uint32_t (&words)[N / 4]) {
    const Ue8m0Scale scale = ue8m0_scale(lane_group_max<Lanes>(abs_max(vals)));
    pack_e4m3(vals, scale.multiplier, words);
    return scale;
}

template <int TileTokens, class LayoutSFB>
CUTE_DEVICE void quantize_group32_from_regs8(
    const __nv_bfloat16 (&vals)[8], cutlass::float_e4m3_t* __restrict__ ws_fp8,
    cutlass::float_ue8m0_t* __restrict__ ws_sf, LayoutSFB const& layout_sfb,
    int n_row, int k_dim, int group, int lane) {
    constexpr int kLastPair = QuantX::PerThread / 2 - 1;
    const auto* v2 = reinterpret_cast<const __nv_bfloat162*>(vals);
    const __nv_bfloat162 m2 =
        __hmax2(__hmax2(__habs2(v2[0]), __habs2(v2[1])),
                __hmax2(__habs2(v2[2]), __habs2(v2[kLastPair])));
    float amax = fmaxf(fmaxf(__bfloat162float(__low2bfloat16(m2)),
                             __bfloat162float(__high2bfloat16(m2))),
                       Fp8Format::AmaxFloor);

    constexpr int kLanes = QuantX::ThreadsPerGrp;
    amax = lane_group_max<kLanes>(amax, ((1u << kLanes) - 1)
                                            << (lane & ~(kLanes - 1)));

    const Ue8m0Scale sc = ue8m0_scale(amax);
    const float biased_exp = sc.biased_exp;
    const __nv_bfloat162 inv2 = __float2bfloat162_rn(sc.multiplier);

    const int base = group * SfLayout::Gran;
    const int j = lane & (kLanes - 1);
    uint32_t packed[2];
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 2; ++i) {
        const __nv_bfloat162 p0 = __hmul2(v2[i * 2], inv2);
        const __nv_bfloat162 p1 = __hmul2(v2[i * 2 + 1], inv2);
        packed[i] = (uint32_t)__nv_cvt_bfloat16raw2_to_fp8x2(
                        *reinterpret_cast<const __nv_bfloat162_raw*>(&p0),
                        __NV_SATFINITE, __NV_E4M3) |
                    ((uint32_t)__nv_cvt_bfloat16raw2_to_fp8x2(
                         *reinterpret_cast<const __nv_bfloat162_raw*>(&p1),
                         __NV_SATFINITE, __NV_E4M3)
                     << Fp8Format::PairBits);
    }

    *reinterpret_cast<uint2*>(ws_fp8 + (int64_t)n_row * k_dim + base +
                              j * QuantX::PerThread) =
        *reinterpret_cast<const uint2*>(packed);

    if (j == 0) {
        cute::Tensor mSFB = cute::make_tensor(
            cute::make_gmem_ptr(reinterpret_cast<uint8_t*>(ws_sf)), layout_sfb);
        mSFB(sf_row_of<TileTokens>(n_row), base, cute::_0{}) =
            static_cast<uint8_t>(biased_exp);
    }
}
}  // namespace halops::mla
