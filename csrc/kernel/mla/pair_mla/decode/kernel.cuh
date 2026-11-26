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

#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/tmem_allocator_sm100.hpp>

#include "kernel/mla/pair_mla/decode/kernel_combine.cuh"
#include "kernel/mla/pair_mla/decode/kernel_epilogue.cuh"
#include "kernel/mla/pair_mla/decode/kernel_mainloop.cuh"
#include "kernel/mla/pair_mla/decode/kernel_schedule.cuh"

namespace halops::mla {

template <class C>
CUTE_DEVICE void init_barriers(typename C::Smem& sm) {
    constexpr int kPairWarps = 2 * kWarpsPerGroup;
    Bar::init(&sm.q_full, kPairWarps);
    Bar::init(&sm.q_empty, 1);
    Bar::init(&sm.o_full, 1);
    for (int i = 0; i < C::kMergeBuf; ++i) Bar::init(&sm.merge_full[i], 1);
    Bar::init(&sm.qs_full, 1);
    for (int b = 0; b < 2; ++b) {
        Bar::init(&sm.o_empty[b], kPairWarps);
        Bar::init(&sm.ml_full[b], kWarpsPerGroup);
        Bar::init(&sm.ml_empty[b], kWarpsPerGroup);
        Bar::init(&sm.s_full[b], 1);
        Bar::init(&sm.s_empty[b], kPairWarps);
        Bar::init(&sm.p_full[b], kPairWarps);
        Bar::init(&sm.pv_done[b], 1);
    }
    for (int s = 0; s < C::kIdxStages; ++s) {
        Bar::init(&sm.idx_full[s], C::kIndexWarps);
        Bar::init(&sm.idx_empty[s], 2 * kWarpsPerGroup);
    }
    for (int s = 0; s < C::kStages; ++s) {
        Bar::init(&sm.raw_full[s], C::kIndexWarps * (1 + kWarp));
        Bar::init(&sm.kv_full[s], kPairWarps);
        Bar::init(&sm.kv_empty[s], 1);
    }
    cutlass::arch::fence_barrier_init();
}

// Persistent: cluster x walks its range of the tile stream.  Warp groups:
// 0 softmax, 1 Q/O, 2 dequant, 3 the QK^T and PV warps (CTA 0) and the two
// index warps (one also allocates TMEM).
template <class C>
CUTE_DEVICE void run_decode(const HcaParams& p, const HcaMaps& maps) {
    extern __shared__ uint8_t smem_raw[];
    typename C::Smem& sm = *reinterpret_cast<typename C::Smem*>(
        smem_raw + (kSw128AtomBytes - smem_u32(smem_raw) % kSw128AtomBytes) %
                       kSw128AtomBytes);
    const int cta = cute::block_rank_in_cluster();
    const int cluster = blockIdx.x / 2;
    const int warp = __shfl_sync(~0u, threadIdx.x / kWarp, 0);
    const int lane = threadIdx.x % kWarp;
    const int wg = warp / kWarpsPerGroup;
    const int t = threadIdx.x % kWarpGroup;

    if (warp == C::kAllocWarp) {
        cute::TMEM::Allocator2Sm().allocate(kTmemCols, &sm.tmem_base);
        cute::TMEM::Allocator2Sm().release_allocation_lock();
    } else if (threadIdx.x == 0) {
        init_barriers<C>(sm);
    }
    const float sink_log2 =
        p.sink ? __ldg(p.sink + t % C::kHeads) * CUDART_L2E_F : -CUDART_INF_F;
    tcgen05_before_thread_sync();
    cute::cluster_sync();
    tcgen05_after_thread_sync();
    const uint32_t tmem = sm.tmem_base;

    pdl_wait();
    find_range<C>(p, sm, cluster);
    const int n = sm.start[2];
    if (C::kFuse && wg < 2) {
        if (wg == 0) {
            cutlass::arch::warpgroup_reg_alloc<C::kSoftmaxRegs>();
            softmax_wg<C>(p, sm, tmem, t, cluster);
        } else {
            cutlass::arch::warpgroup_reg_alloc<C::kQoRegs>();
            qo_wg<C>(p, maps, sm, tmem, sink_log2, cta, t, cluster);
        }
        merge_tail<C>(p, sm, cta, cluster);
    } else if (wg == 0) {
        cutlass::arch::warpgroup_reg_alloc<C::kSoftmaxRegs>();
        softmax_wg<C>(p, sm, tmem, t, cluster);
    } else if (wg == 1) {
        cutlass::arch::warpgroup_reg_alloc<C::kQoRegs>();
        qo_wg<C>(p, maps, sm, tmem, sink_log2, cta, t, cluster);
    } else if (wg == 2) {
        cutlass::arch::warpgroup_reg_dealloc<C::kDequantRegs>();
        if (cta == 0) {
            dequant_wg<0, C>(sm, n, t);
        } else {
            dequant_wg<1, C>(sm, n, t);
        }
    } else {
        cutlass::arch::warpgroup_reg_dealloc<C::kAuxRegs>();
        if (warp == C::kQkWarp) {
            if (cta == 0) qk_warp<C>(p, sm, tmem, cluster);
        } else if (warp == C::kPvWarp) {
            if (cta == 0) pv_warp<C>(p, sm, tmem, cluster);
        } else if (warp - C::kIndexWarp < C::kIndexWarps) {
            index_warp<C>(p, maps, sm, cta, warp - C::kIndexWarp, lane,
                          cluster);
        }
    }
    pdl_trigger();
    tcgen05_before_thread_sync();
    cute::cluster_sync();
    if (warp == C::kAllocWarp) {
        tcgen05_after_thread_sync();
        cute::TMEM::Allocator2Sm().free(tmem, kTmemCols);
    }
}

template <class C>
__global__ void __launch_bounds__(C::kThreads, 1)
    hca_decode_attention_kernel(const __grid_constant__ HcaParams p,
                                const __grid_constant__ HcaMaps maps) {
    run_decode<C>(p, maps);
}

template <class C>
__global__ void __launch_bounds__(C::kThreads, 1)
    hca_plain_attention_kernel(const __grid_constant__ HcaParams p,
                               const __grid_constant__ HcaMaps maps) {
    run_decode<C>(p, maps);
}

template <class C>
__global__ void __launch_bounds__(C::kThreads, 1)
    dsa_decode_attention_kernel(const __grid_constant__ HcaParams p,
                                const __grid_constant__ HcaMaps maps) {
    run_decode<C>(p, maps);
}

}  // namespace halops::mla
