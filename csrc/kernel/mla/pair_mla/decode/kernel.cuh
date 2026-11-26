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
#include "kernel/mla/pair_mla/decode/scheduler_range.cuh"

namespace halops::mla {

template <class C>
CUTE_DEVICE void decode_init_barriers(typename C::Smem& sm) {
    constexpr int kPairWarps = 2 * kWarpsPerGroup;
    DecodeBar::init(&sm.q_full, kPairWarps);
    DecodeBar::init(&sm.q_empty, 1);
    DecodeBar::init(&sm.o_full, 1);
    for (int i = 0; i < C::kMergeBuf; ++i)
        DecodeBar::init(&sm.merge_full[i], 1);
    DecodeBar::init(&sm.qs_full, 1);
    for (int b = 0; b < 2; ++b) {
        DecodeBar::init(&sm.o_empty[b], kPairWarps);
        DecodeBar::init(&sm.ml_full[b], kWarpsPerGroup);
        DecodeBar::init(&sm.ml_empty[b], kWarpsPerGroup);
        DecodeBar::init(&sm.s_full[b], 1);
        DecodeBar::init(&sm.s_empty[b], kPairWarps);
        DecodeBar::init(&sm.p_full[b], kPairWarps);
        DecodeBar::init(&sm.pv_done[b], 1);
    }
    for (int s = 0; s < C::kIdxStages; ++s) {
        DecodeBar::init(&sm.idx_full[s], C::kIndexWarps);
        DecodeBar::init(&sm.idx_empty[s], 2 * kWarpsPerGroup);
    }
    for (int s = 0; s < C::kStages; ++s) {
        DecodeBar::init(&sm.raw_full[s], C::kIndexWarps * (1 + kWarp));
        DecodeBar::init(&sm.kv_full[s], kPairWarps);
        DecodeBar::init(&sm.kv_empty[s], 1);
    }
    cutlass::arch::fence_barrier_init();
}

// Persistent: cluster x walks its range of the tile stream.  Warp groups:
// 0 softmax, 1 Q/O, 2 dequant, 3 the QK^T and PV warps (CTA 0) and the two
// index warps (one also allocates TMEM).
template <class C>
CUTE_DEVICE void decode_run_attention(const DecodeParams& p,
                                      const DecodeMaps& maps) {
    extern __shared__ uint8_t smem_raw[];
    typename C::Smem& sm = *reinterpret_cast<typename C::Smem*>(
        smem_raw +
        (kSw128AtomBytes - decode_smem_u32(smem_raw) % kSw128AtomBytes) %
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
        decode_init_barriers<C>(sm);
    }
    const float sink_log2 =
        p.sink ? __ldg(p.sink + t % C::kHeads) * CUDART_L2E_F : -CUDART_INF_F;
    tcgen05_before_thread_sync();
    cute::cluster_sync();
    tcgen05_after_thread_sync();
    const uint32_t tmem = sm.tmem_base;

    pdl_wait();
    decode_find_range<C>(p, sm, cluster);
    const int n = sm.start[2];
    if (C::kFuse && wg < 2) {
        if (wg == 0) {
            cutlass::arch::warpgroup_reg_alloc<C::kSoftmaxRegs>();
            decode_softmax_wg<C>(p, sm, tmem, t, cluster);
        } else {
            cutlass::arch::warpgroup_reg_alloc<C::kQoRegs>();
            decode_qo_wg<C>(p, maps, sm, tmem, sink_log2, cta, t, cluster);
        }
        decode_merge_tail<C>(p, sm, cta, cluster);
    } else if (wg == 0) {
        cutlass::arch::warpgroup_reg_alloc<C::kSoftmaxRegs>();
        decode_softmax_wg<C>(p, sm, tmem, t, cluster);
    } else if (wg == 1) {
        cutlass::arch::warpgroup_reg_alloc<C::kQoRegs>();
        decode_qo_wg<C>(p, maps, sm, tmem, sink_log2, cta, t, cluster);
    } else if (wg == 2) {
        cutlass::arch::warpgroup_reg_dealloc<C::kDequantRegs>();
        if (cta == 0) {
            decode_dequant_wg<0, C>(sm, n, t);
        } else {
            decode_dequant_wg<1, C>(sm, n, t);
        }
    } else {
        cutlass::arch::warpgroup_reg_dealloc<C::kAuxRegs>();
        if (warp == C::kQkWarp) {
            if (cta == 0) decode_qk_warp<C>(p, sm, tmem, cluster);
        } else if (warp == C::kPvWarp) {
            if (cta == 0) decode_pv_warp<C>(p, sm, tmem, cluster);
        } else if (warp - C::kIndexWarp < C::kIndexWarps) {
            decode_index_warp<C>(p, maps, sm, cta, warp - C::kIndexWarp, lane,
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
    decode_pair_mla_hca_kernel(const __grid_constant__ DecodeParams p,
                               const __grid_constant__ DecodeMaps maps) {
    decode_run_attention<C>(p, maps);
}

template <class C>
__global__ void __launch_bounds__(C::kThreads, 1)
    decode_pair_mla_plain_kernel(const __grid_constant__ DecodeParams p,
                                 const __grid_constant__ DecodeMaps maps) {
    decode_run_attention<C>(p, maps);
}

template <class C>
__global__ void __launch_bounds__(C::kThreads, 1)
    decode_pair_mla_dsa_kernel(const __grid_constant__ DecodeParams p,
                               const __grid_constant__ DecodeMaps maps) {
    decode_run_attention<C>(p, maps);
}

}  // namespace halops::mla
