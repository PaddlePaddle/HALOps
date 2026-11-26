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
#include <cutlass/cutlass.h>

#include <cute/config.hpp>

#include "common/sm100_arch.cuh"

namespace halops::mla {
CUTE_DEVICE
nv_bfloat162 fp8x2_to_bf16x2_with_scale(__nv_fp8x2_e4m3 data,
                                        nv_bfloat16 scale) {
    uint32_t bf;
    asm("cvt.rn.bf16x2.e4m3x2 %0, %1;"
        : "=r"(bf)
        : "h"(*reinterpret_cast<const unsigned short*>(&data)));
    return __hmul2(*reinterpret_cast<const nv_bfloat162*>(&bf),
                   __bfloat162bfloat162(scale));
}

CUTE_DEVICE
float2 add_rn_f32x2(const float2& lhs, const float2& rhs) {
    float2 result;
    asm volatile("add.rn.f32x2 %0, %1, %2;\n"
                 : "=l"(reinterpret_cast<uint64_t&>(result))
                 : "l"(reinterpret_cast<uint64_t const&>(lhs)),
                   "l"(reinterpret_cast<uint64_t const&>(rhs)));
    return result;
}
CUTE_DEVICE
float2 mul_rn_f32x2(const float2& lhs, const float2& rhs) {
    float2 result;
    asm volatile("mul.rn.f32x2 %0, %1, %2;\n"
                 : "=l"(reinterpret_cast<uint64_t&>(result))
                 : "l"(reinterpret_cast<uint64_t const&>(lhs)),
                   "l"(reinterpret_cast<uint64_t const&>(rhs)));
    return result;
}
CUTE_DEVICE
float2 fma_rn_f32x2(const float2& lhs, const float2& rhs,
                    const float2& addend) {
    float2 result;
    asm volatile("fma.rn.f32x2 %0, %1, %2, %3;\n"
                 : "=l"(reinterpret_cast<uint64_t&>(result))
                 : "l"(reinterpret_cast<uint64_t const&>(lhs)),
                   "l"(reinterpret_cast<uint64_t const&>(rhs)),
                   "l"(reinterpret_cast<uint64_t const&>(addend)));
    return result;
}

// Scalar .rn ops without .ftz, which __fmaf_rn and friends get under
// --use_fast_math.
CUTE_DEVICE float fma_rn_f32(float a, float b, float c) {
    float d;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(d) : "f"(a), "f"(b), "f"(c));
    return d;
}

CUTE_DEVICE float mul_rn_f32(float a, float b) {
    float d;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
    return d;
}

CUTE_DEVICE float add_rn_f32(float a, float b) {
    float d;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
    return d;
}

CUTE_DEVICE float warp_sum(float value) {
    CUTLASS_PRAGMA_UNROLL
    for (int offset = kWarp / 2; offset > 0; offset >>= 1)
        value += __shfl_xor_sync(~0u, value, offset);
    return value;
}
}  // namespace halops::mla
