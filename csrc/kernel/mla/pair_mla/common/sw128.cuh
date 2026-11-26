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

#include <cstdint>
#include <cute/arch/util.hpp>

#include "common/sm100_arch.cuh"
#include "kernel/mla/common/intrinsics.cuh"

namespace halops::mla {

// Row r and its swizzled 16-byte chunk c in a SW128 tile of 8-row atoms.
CUTE_DEVICE int pair_row_offset(int r) {
    return (r / kSw128AtomRows) * kSw128AtomBytes +
           (r % kSw128AtomRows) * kSw128AtomRowBytes;
}
CUTE_DEVICE int pair_sw128_offset(int r, int c) {
    return pair_row_offset(r) + (c ^ (r % kSw128AtomRows)) * kChunk;
}

// A lane's row in a 128-byte swizzled box; chunk Q is at row_x ^ 16Q.
CUTE_DEVICE uint32_t pair_swizzled_row(const void* box, int row) {
    return (cute::cast_smem_ptr_to_uint(box) + row * kSw128AtomRowBytes) |
           (row % kSw128AtomRows) * kChunk;
}

// Orders the group's shared loads before a later TMA write to the source.
CUTE_DEVICE uint32_t pair_xor_words(const uint4* v, int n) {
    uint32_t x = 0;
    for (int i = 0; i < n; ++i) x ^= v[i].x ^ v[i].w;
    return x;
}

// Eight fp8 e4m3 values times the e8 block scale, as bf16.
CUTE_DEVICE uint4 pair_dequant8(uint32_t lo, uint32_t hi, uint8_t e8) {
    const uint32_t raw[2] = {lo, hi};
    const auto* f8 = reinterpret_cast<const __nv_fp8x2_e4m3*>(raw);
    const nv_bfloat16 scale =
        __ushort_as_bfloat16(uint16_t(e8) << kBf16MantissaBits);
    uint4 out;
    auto* o = reinterpret_cast<nv_bfloat162*>(&out);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kElemsPerChunk / 2; ++i) {
        o[i] = fp8x2_to_bf16x2_with_scale(f8[i], scale);
    }
    return out;
}

}  // namespace halops::mla
