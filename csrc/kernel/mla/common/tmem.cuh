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
void umma_arrive_noelect(cutlass::arch::ClusterTransactionBarrier& bar) {
    uint32_t bar_intptr = cute::cast_smem_ptr_to_uint(&bar);
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 "
        "[%0];\n"
        :
        : "r"(bar_intptr)
        : "memory");
}

CUTE_DEVICE void tcgen05_before_thread_sync() {
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}
CUTE_DEVICE void tcgen05_after_thread_sync() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

template <int N>
using TmemLoad32dp32b = cute::conditional_t<
    N == 1, cute::SM100_TMEM_LOAD_32dp32b1x,
    cute::conditional_t<
        N == 2, cute::SM100_TMEM_LOAD_32dp32b2x,
        cute::conditional_t<
            N == 4, cute::SM100_TMEM_LOAD_32dp32b4x,
            cute::conditional_t<
                N == 8, cute::SM100_TMEM_LOAD_32dp32b8x,
                cute::conditional_t<
                    N == 16, cute::SM100_TMEM_LOAD_32dp32b16x,
                    cute::conditional_t<
                        N == 32, cute::SM100_TMEM_LOAD_32dp32b32x,
                        cute::conditional_t<
                            N == 64, cute::SM100_TMEM_LOAD_32dp32b64x,
                            cute::SM100_TMEM_LOAD_32dp32b128x>>>>>>>;

template <int N>
using CommonTmemStore32dp32b = cute::conditional_t<
    N == 1, cute::SM100_TMEM_STORE_32dp32b1x,
    cute::conditional_t<
        N == 2, cute::SM100_TMEM_STORE_32dp32b2x,
        cute::conditional_t<
            N == 4, cute::SM100_TMEM_STORE_32dp32b4x,
            cute::conditional_t<
                N == 8, cute::SM100_TMEM_STORE_32dp32b8x,
                cute::conditional_t<
                    N == 16, cute::SM100_TMEM_STORE_32dp32b16x,
                    cute::conditional_t<
                        N == 32, cute::SM100_TMEM_STORE_32dp32b32x,
                        cute::conditional_t<
                            N == 64, cute::SM100_TMEM_STORE_32dp32b64x,
                            cute::SM100_TMEM_STORE_32dp32b128x>>>>>>>;

template <int kNumElements>
CUTE_DEVICE void tmem_ld_32dp32bNx(uint32_t tmem_start, void* data_) {
    uint32_t* data = (uint32_t*)data_;
    [&]<size_t... Is>(cute::index_sequence<Is...>) {
        TmemLoad32dp32b<kNumElements>::copy(tmem_start, data[Is]...);
    }(cute::make_index_sequence<kNumElements>{});
}

// tmem_ld_32dp32bNx, then wait for the load.
template <int kNumElements>
CUTE_DEVICE void tmem_ld_32dp32bNx_wait(uint32_t tmem_start, void* data) {
    tmem_ld_32dp32bNx<kNumElements>(tmem_start, data);
    cutlass::arch::fence_view_async_tmem_load();
}

template <int kNumElements>
CUTE_DEVICE void tmem_st_32dp32bNx(uint32_t tmem_start, void const* data_) {
    uint32_t const* data = (uint32_t const*)data_;
    [&]<size_t... Is>(cute::index_sequence<Is...>) {
        CommonTmemStore32dp32b<kNumElements>::copy(data[Is]..., tmem_start);
    }(cute::make_index_sequence<kNumElements>{});
}
}  // namespace halops::mla
