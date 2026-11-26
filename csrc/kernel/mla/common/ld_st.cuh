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

#include <cutlass/arch/barrier.h>

#include <cute/tensor.hpp>

namespace halops::mla {
CUTE_DEVICE
void ld_global_cg_v8(const void* ptr, uint4& lo, uint4& hi) {
    asm volatile("ld.global.cg.v8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                 : "=r"(lo.x), "=r"(lo.y), "=r"(lo.z), "=r"(lo.w), "=r"(hi.x),
                   "=r"(hi.y), "=r"(hi.z), "=r"(hi.w)
                 : "l"(ptr));
}

CUTE_DEVICE
void st_global_v8(void* ptr, const uint32_t (&words)[8]) {
    asm volatile("st.global.v8.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};" ::"l"(ptr),
                 "r"(words[0]), "r"(words[1]), "r"(words[2]), "r"(words[3]),
                 "r"(words[4]), "r"(words[5]), "r"(words[6]), "r"(words[7])
                 : "memory");
}

CUTE_DEVICE
void smem_st_128b(uint32_t smem_addr, __int128_t const& data) {
    asm volatile("st.weak.shared::cta.b128 [%0], %1;\n" ::"r"(smem_addr),
                 "q"(data));
}

// 16-byte shared load / store at smem_addr ^ kXor (an address swizzle).
template <uint32_t kXor>
CUTE_DEVICE uint4 lds_128b_xor(uint32_t smem_addr) {
    uint4 v;
    asm volatile(
        "{\n\t.reg .b32 a;\n\txor.b32 a, %4, %5;\n\t"
        "ld.shared.v4.u32 {%0, %1, %2, %3}, [a];\n\t}"
        : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
        : "r"(smem_addr), "n"(kXor)
        : "memory");
    return v;
}

template <uint32_t kXor>
CUTE_DEVICE void sts_128b_xor(uint32_t smem_addr, uint32_t a, uint32_t b,
                              uint32_t c, uint32_t d) {
    asm volatile(
        "{\n\t.reg .b32 a;\n\txor.b32 a, %0, %1;\n\t"
        "st.shared.v4.u32 [a], {%2, %3, %4, %5};\n\t}" ::"r"(smem_addr),
        "n"(kXor), "r"(a), "r"(b), "r"(c), "r"(d)
        : "memory");
}

CUTE_DEVICE void st_global_v8_na(void* ptr, const uint32_t (&words)[8]) {
    const uint4 lo = reinterpret_cast<const uint4*>(words)[0];
    const uint4 hi = reinterpret_cast<const uint4*>(words)[1];
    asm volatile(
        "st.global.L1::no_allocate.v8.b32 [%0], "
        "{%1,%2,%3,%4,%5,%6,%7,%8};\n" ::"l"(ptr),
        "r"(lo.x), "r"(lo.y), "r"(lo.z), "r"(lo.w), "r"(hi.x), "r"(hi.y),
        "r"(hi.z), "r"(hi.w)
        : "memory");
}
}  // namespace halops::mla
