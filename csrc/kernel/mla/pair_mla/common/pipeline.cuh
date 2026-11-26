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
#include <cutlass/arch/barrier.h>

#include <cstdint>

#include "common/sm100_arch.cuh"
#include "kernel/mla/common/intrinsics.cuh"

namespace halops::mla {

// Long Q/O waits: one warp polls, the others sleep on a named barrier.
template <class C>
CUTE_DEVICE void pair_qo_wait(uint64_t* bar, int phase, int t) {
    if (t < kWarp) cutlass::arch::ClusterBarrier::wait(bar, phase);
    cutlass::arch::NamedBarrier::sync(kWarpGroup, C::kQoBar);
}

// Softmax thread t owns TMEM lane t (head t % 64, dims half t / 64).
template <class C>
CUTE_DEVICE void pair_lane_sync(int t) {
    cutlass::arch::NamedBarrier::sync(2 * kWarp, C::kPairBar + t / kWarp % 2);
}

// Waits for PV(j).
template <class C>
CUTE_DEVICE void pair_wait_pv(typename C::Smem& sm, int j) {
    cutlass::arch::ClusterBarrier::wait(&sm.pv_done[j % 2], (j / 2) & 1);
}

// Reads this lane's O columns [m * kOHalf, +kOHalf), freeing them for the
// next PV.
template <class C>
CUTE_DEVICE void pair_read_o_half(typename C::Smem& sm, uint32_t lane_base,
                                  int t, int m, float (&o)[C::kOHalf]) {
    CUTLASS_PRAGMA_UNROLL
    for (int c = 0; c < C::kOHalf; c += kTmemLd) {
        tmem_ld_32dp32bNx<kTmemLd>(lane_base + C::kTmemO + m * C::kOHalf + c,
                                   o + c);
    }
    cutlass::arch::fence_view_async_tmem_load();
    tcgen05_before_thread_sync();
    __syncwarp();
    if (t % kWarp == 0)
        cutlass::arch::ClusterBarrier::arrive(&sm.o_empty[m], 0, 1);
}

// Reads half mh of this lane's O as scaled bf16 pairs and frees it.
template <class C>
CUTE_DEVICE void pair_read_o_bf16(typename C::Smem& sm, uint32_t lane_base,
                                  int t, int mh, float scale,
                                  uint32_t (&w)[C::kOHalf / 2]) {
    CUTLASS_PRAGMA_UNROLL
    for (int c = 0; c < C::kOHalf; c += kTmemLd) {
        float o[kTmemLd];
        tmem_ld_32dp32bNx_wait<kTmemLd>(
            lane_base + C::kTmemO + mh * C::kOHalf + c, o);
        CUTLASS_PRAGMA_UNROLL
        for (int e = 0; e < kTmemLd / 2; ++e) {
            const nv_bfloat162 b2 =
                __floats2bfloat162_rn(o[2 * e] * scale, o[2 * e + 1] * scale);
            w[c / 2 + e] = reinterpret_cast<const uint32_t&>(b2);
        }
    }
    tcgen05_before_thread_sync();
    __syncwarp();
    if (t % kWarp == 0)
        cutlass::arch::ClusterBarrier::arrive(&sm.o_empty[mh], 0, 1);
}

}  // namespace halops::mla
