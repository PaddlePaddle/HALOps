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
#include <cutlass/float8.h>

#include "common/ptx.cuh"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"

namespace halops::mxfp8 {

using E4m3 = cutlass::float_e4m3_t;

constexpr int ceil_log2(double x) {
    int e = 0;
    for (; x > 1.0; x /= 2) ++e;
    for (; x <= 0.5; x *= 2) --e;
    return e;
}

inline constexpr int kGroup = 32;
inline constexpr uint32_t kBf16MantBits =
    E4m3::FP32_NUM_MANTISSA_BITS - cute::sizeof_bits_v<cutlass::bfloat16_t>;
inline constexpr uint32_t kUe8m0Bias = E4m3::FP32_EXPONENT_BIAS;
inline constexpr uint32_t kE4m3MaxExp =
    (E4m3::FP8_MAX_FLT >> E4m3::FP8_NUM_MANTISSA_BITS) -
    E4m3::FP8_EXPONENT_BIAS;
inline constexpr uint32_t kE4m3MaxMant =
    E4m3::FP8_MAX_FLT & E4m3::FP8_MANTISSA_MASK;
inline constexpr float kE4m3Max =
    (1.0f + float(kE4m3MaxMant) / (1u << E4m3::FP8_NUM_MANTISSA_BITS)) *
    float(1u << kE4m3MaxExp);
inline constexpr uint32_t kE4m3MaxMantCarry =
    (1u << kBf16MantBits) - 1 -
    (kE4m3MaxMant << (kBf16MantBits - E4m3::FP8_NUM_MANTISSA_BITS));
inline constexpr float kAmaxFloor = 1e-10f;
inline constexpr uint32_t kMinScale =
    int(kUe8m0Bias) + ceil_log2(kAmaxFloor / kE4m3Max);

using SfChunk = cutlass::detail::Sm1xxBlockScaledBasicChunk<kGroup>;
using SfAtom = SfChunk::SfAtom;
inline constexpr int kAtomRows = SfChunk::Blk_MN{};
inline constexpr int kAtomCols = SfChunk::Blk_SF{};
inline constexpr int kAtomColRows = cute::size<0, 0>(SfAtom{});

CUTE_DEVICE uint32_t exp8_pair(__nv_bfloat162 m) {
    return ((reinterpret_cast<const uint32_t&>(m) +
             kE4m3MaxMantCarry * 0x10001u) >>
            kBf16MantBits) &
           0x00ff00ffu;
}
CUTE_DEVICE uint32_t scale_of_exp8(uint32_t y) {
    return ::max(y, kMinScale + kE4m3MaxExp) - kE4m3MaxExp;
}
CUTE_DEVICE uint32_t multiplier_bits(uint32_t e) {
    return (2 * kUe8m0Bias - e) << kBf16MantBits;
}

template <int N>
CUTE_DEVICE __nv_bfloat162 abs_max(const __nv_bfloat162* v) {
    uint32_t w[N];
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) w[i] = reinterpret_cast<const uint32_t&>(v[i]);
    CUTLASS_PRAGMA_UNROLL
    for (int s = 1; s < N; s *= 2) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < N; i += 2 * s)
            w[i] = bf16x2_max_abs_xorsign(w[i], w[i + s]);
    }
    const uint32_t m = w[0] & 0x7fff7fffu;
    return __hmax2(reinterpret_cast<const __nv_bfloat162&>(m),
                   __float2bfloat162_rn(0.f));
}

CUTE_DEVICE uint32_t e4m3x4(__nv_bfloat162 v0, __nv_bfloat162 v1,
                            __nv_bfloat162 scale) {
    const auto cvt = [&](__nv_bfloat162 v) {
        const __nv_bfloat162 p = __hmul2(v, scale);
        return (uint32_t)__nv_cvt_bfloat16raw2_to_fp8x2(
            reinterpret_cast<const __nv_bfloat162_raw&>(p), __NV_SATFINITE,
            __NV_E4M3);
    };
    return cvt(v0) | (cvt(v1) << 16);
}

CUTE_DEVICE int64_t sf_atom_offset(int64_t row, int kb, int kb_dim) {
    const int r = row % kAtomRows;
    return ((row / kAtomRows) * (kb_dim / kAtomCols) + kb / kAtomCols) *
               cute::cosize_v<SfAtom> +
           SfAtom{}(cute::make_coord(
               cute::make_coord(r % kAtomColRows, r / kAtomColRows),
               cute::make_coord(0, kb % kAtomCols)));
}

}  // namespace halops::mxfp8
