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

#include "kernel/mla/common/intrinsics.cuh"
#include "kernel/mla/common/traits.cuh"

namespace halops::mla {
template <class Bar>
CUTE_DEVICE void bar_sync() {
    cutlass::arch::NamedBarrier::arrive_and_wait(Bar::width, Bar::id);
}

CUTE_DEVICE void cta_sync() {
    __syncwarp();
    bar_sync<KernelBase::Bars::EveryoneSync>();
}

CUTE_DEVICE void phase_end_sync() {
    cta_sync();
    cutlass::arch::fence_barrier_init();
    bar_sync<KernelBase::Bars::EveryoneSync>();
}

CUTE_DEVICE void bar_sync_warp_pair(int local_warp) {
    using Bars = KernelBase::Bars;
    static_assert(Bars::Wg1Warp13Sync::id == Bars::Wg1Warp02Sync::id + 1,
                  "pair barrier ids are adjacent");
    cutlass::arch::NamedBarrier::arrive_and_wait(
        Bars::Wg1Warp02Sync::width,
        Bars::Wg1Warp02Sync::id + (uint32_t)(local_warp & 1));
}

template <int Tokens>
CUTE_DEVICE int swizzled_token_index(int row, int token) {
    constexpr int kChunks = Tokens / kQuad;
    return row * Tokens + ((((token >> 2) ^ row) & (kChunks - 1)) << 2) +
           (token & (kQuad - 1));
}

CUTE_DEVICE int rev_bid() { return gridDim.x - 1 - blockIdx.x; }
CUTE_DEVICE int items_from(int first, int step, int count) {
    return count > first ? (count - first + step - 1) / step : 0;
}

struct PartialBarrier {
    int* slot;
    int n_producers;

    CUTE_DEVICE PartialBarrier(int* base, int idx, int n_prod)
        : slot(base + (int64_t)idx *
                          (WorkspaceLayout::RendezvousSlotBytes / sizeof(int))),
          n_producers(n_prod) {}

    CUTE_DEVICE void arrive_thread() const { gmem_red_add1_release(slot); }

    CUTE_DEVICE void wait(uint32_t epoch) const {
        const int target = (int)(epoch + 1) * n_producers;
        int seen;
        do {
            seen = gmem_ld_relaxed(slot);
        } while (seen < target);
        gmem_ld_acquire(slot);
    }

    CUTE_DEVICE void wait_warp(uint32_t epoch) const {
        if (cute::elect_one_sync()) wait(epoch);
        __syncwarp();
    }

    CUTE_DEVICE void wait_warp_for_tma(uint32_t epoch) const {
        wait_warp(epoch);
        fence_proxy_async_global();
    }

    CUTE_DEVICE void reset_relaxed() const { gmem_st_zero_relaxed(slot); }
};
}  // namespace halops::mla
