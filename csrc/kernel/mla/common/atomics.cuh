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
void gmem_red_add1_release(int* ptr) {
    asm volatile("red.release.gpu.global.add.s32 [%0], 1;" ::"l"(ptr)
                 : "memory");
}

CUTE_DEVICE
int gmem_ld_acquire(const int* ptr) {
    int value;
    asm volatile("ld.acquire.gpu.global.s32 %0, [%1];"
                 : "=r"(value)
                 : "l"(ptr)
                 : "memory");
    return value;
}

CUTE_DEVICE
int gmem_ld_relaxed(const int* ptr) {
    int value;
    asm volatile("ld.relaxed.gpu.global.s32 %0, [%1];"
                 : "=r"(value)
                 : "l"(ptr)
                 : "memory");
    return value;
}

CUTE_DEVICE
int gmem_atom_add1_acq_rel(int* ptr) {
    int prev;
    asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], 1;"
                 : "=r"(prev)
                 : "l"(ptr)
                 : "memory");
    return prev;
}

CUTE_DEVICE
int gmem_atom_add_acq_rel(int* ptr, int v) {
    int prev;
    asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;"
                 : "=r"(prev)
                 : "l"(ptr), "r"(v)
                 : "memory");
    return prev;
}

CUTE_DEVICE
void gmem_red_add1_relaxed(int* ptr) {
    asm volatile("red.relaxed.gpu.global.add.s32 [%0], 1;" ::"l"(ptr)
                 : "memory");
}

CUTE_DEVICE
void fence_proxy_async_global() {
    asm volatile("fence.proxy.async.global;" ::: "memory");
}

CUTE_DEVICE
void gmem_st_zero_relaxed(int* ptr) {
    asm volatile("st.global.s32 [%0], 0;" ::"l"(ptr) : "memory");
}

// bar.red.or on named barrier kId (numbered as cutlass::arch::NamedBarrier),
// result discarded.
template <uint32_t kId, uint32_t kThreads>
CUTE_DEVICE void named_bar_red_or(uint32_t pred) {
    constexpr uint32_t kHwId =
        kId + uint32_t(cutlass::arch::ReservedNamedBarriers::FirstUserBarrier);
    asm volatile(
        "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %0, 0;\n\t"
        "bar.red.or.pred p, %1, %2, p;\n\t}" ::"r"(pred),
        "n"(kHwId), "n"(kThreads)
        : "memory");
}
}  // namespace halops::mla
