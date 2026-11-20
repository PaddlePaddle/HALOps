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

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>

#include "common/pdl.cuh"
#include "kernel/mla/prefill_pair_mla_epilogue.cuh"

namespace halops::mla::prefill_pair {

template <class C>
CUTE_DEVICE void init_barriers(typename C::Smem& sm) {
    constexpr int kPairWarps =
        2 * C::kWarpsPerGroup;  // a warp group of both CTAs
    cutlass::arch::ClusterBarrier::init(&sm.q_full, kPairWarps);
    cutlass::arch::ClusterBarrier::init(&sm.q_empty, 1);
    cutlass::arch::ClusterBarrier::init(&sm.o_full, 1);
    for (int i = 0; i < C::kMergeBuf; ++i)
        cutlass::arch::ClusterBarrier::init(&sm.merge_full[i], 1);
    cutlass::arch::ClusterBarrier::init(&sm.qs_full, 1);
    if constexpr (C::kQst) {
        for (int b = 0; b < 2; ++b)
            cutlass::arch::ClusterBarrier::init(&sm.qb.qb_full[b], 1);
    }
    for (int b = 0; b < 2; ++b) {
        cutlass::arch::ClusterBarrier::init(&sm.o_empty[b], kPairWarps);
        cutlass::arch::ClusterBarrier::init(&sm.ml_full[b], C::kWarpsPerGroup);
        cutlass::arch::ClusterBarrier::init(&sm.ml_empty[b], C::kWarpsPerGroup);
        cutlass::arch::ClusterBarrier::init(&sm.rs_full[b],
                                            C::kWarpsPerGroup / 2);
        cutlass::arch::ClusterBarrier::init(&sm.s_full[b], 1);
        cutlass::arch::ClusterBarrier::init(&sm.s_empty[b], kPairWarps);
        cutlass::arch::ClusterBarrier::init(&sm.p_full[b], kPairWarps);
        cutlass::arch::ClusterBarrier::init(&sm.pv_done[b], 1);
    }
    for (int s = 0; s < C::kIdxStages; ++s) {
        cutlass::arch::ClusterBarrier::init(&sm.idx_full[s], C::kIndexWarps);
        cutlass::arch::ClusterBarrier::init(
            &sm.idx_empty[s],
            2 * C::kWarpsPerGroup);  // softmax + dequant
    }
    for (int s = 0; s < C::kKvStages; ++s) {
        // per index warp: the expect-tx arrival and one cp.async per lane
        cutlass::arch::ClusterBarrier::init(&sm.raw_full[s],
                                            C::kIndexWarps * (1 + C::kWarp));
        cutlass::arch::ClusterBarrier::init(&sm.kv_full[s], kPairWarps);
        cutlass::arch::ClusterBarrier::init(&sm.kv_empty[s], 1);
    }
    cutlass::arch::fence_barrier_init();
}

// Warp groups: 0 softmax, 1 Q/O, 2 dequant, 3 QK^T, PV and index warps.
template <class C>
CUTE_DEVICE void run_attention(const PairMlaParams& p,
                               const PairMlaMaps& maps) {
    extern __shared__ uint8_t smem_raw[];
    typename C::Smem& sm = *reinterpret_cast<typename C::Smem*>(
        smem_raw + (C::kAtomBytes -
                    cute::cast_smem_ptr_to_uint(smem_raw) % C::kAtomBytes) %
                       C::kAtomBytes);
    const int cta = cute::block_rank_in_cluster();
    const int cluster = blockIdx.x / 2;
    const int warp = __shfl_sync(~0u, threadIdx.x / C::kWarp, 0);
    const int lane = threadIdx.x % C::kWarp;
    const int wg = warp / C::kWarpsPerGroup;
    const int t = threadIdx.x % C::kWarpGroup;

    if (warp == C::kAllocWarp) {
        cute::TMEM::Allocator2Sm().allocate(C::kTmemCols, &sm.tmem_base);
        cute::TMEM::Allocator2Sm().release_allocation_lock();
    } else if (threadIdx.x == 0) {
        init_barriers<C>(sm);
    }
    // The sink is a weight: read before pdl_wait.
    const float sink_log2 =
        p.sink ? __ldg(p.sink + t % C::kHeads) * CUDART_L2E_F : -CUDART_INF_F;
    tcgen05_before_thread_sync();
    cute::cluster_sync();
    tcgen05_after_thread_sync();
    const uint32_t tmem = sm.tmem_base;

    pdl_wait();
    find_range<C>(p, sm, cluster);
    const int n = sm.start[2];
    if (wg == 0) {
        cutlass::arch::warpgroup_reg_alloc<C::kSoftmaxRegs>();
        softmax_wg<C>(p, sm, tmem, t);
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
            if (cta == 0) qk_warp<C>(p, sm, tmem);
        } else if (warp == C::kPvWarp) {
            if (cta == 0) pv_warp<C>(p, sm, tmem);
        } else {
            static_assert(C::kIndexWarp + C::kIndexWarps ==
                          C::kThreads / C::kWarp);
            index_warp<C>(p, maps, sm, cta, warp - C::kIndexWarp, lane);
        }
    }
    pdl_trigger();
    tcgen05_before_thread_sync();
    cute::cluster_sync();
    if (warp == C::kAllocWarp) {
        tcgen05_after_thread_sync();
        cute::TMEM::Allocator2Sm().free(tmem, C::kTmemCols);
    }
}

// A 2-CTA cluster attends a token pair over the union of their KV lists;
// CTA c holds token c's Q and dims [256c, +256) of every key. A pair's keys
// stream as its c128 then SWA rows in 64-key tiles; a pair split over
// clusters is merged in store_o or by pair_mla_combine_kernel.
template <bool kQst, bool kFp8, bool kQnr, class Dims>
static __global__ void __launch_bounds__(PairMlaTraits<Dims>::kThreads, 1)
    pair_mla_kernel(const __grid_constant__ PairMlaParams p,
                    const __grid_constant__ PairMlaMaps maps) {
    run_attention<Cfg<kQst, kFp8, kQnr, Dims>>(p, maps);
}

// Combine kernel: block (plane, cta, c) merges the parts of the pair split
// from cluster c.
template <class C>
static __global__ void pair_mla_combine_kernel(
    const __grid_constant__ PairMlaParams p) {
    __shared__ float red[C::kCombSplit][C::kWarpGroup][C::kPlaneCols];
    __shared__ float redmd[C::kCombSplit][C::kWarpGroup]
                          [2];  // (running max, denom)
    pdl_wait();
    const MergeJob j = p.jobs[blockIdx.z];
    const int cta = blockIdx.y;
    if (j.wn == 0 || (cta == 1 && !j.has_b)) return;
    const int pl = blockIdx.x;                    // plane [0, kPlanes)
    const int grp = threadIdx.x / C::kWarpGroup;  // part-group [0, kCombSplit)
    const int tsi = threadIdx.x % C::kWarpGroup;  // storage thread index
    const int head = tsi % C::kHeads, half = tsi / C::kHeads;
    const int total = *p.total_g;
    const int sched = *p.sched_g;
    const int first = cluster_of(sched, total, j.w0);
    const int last = cluster_of(sched, total, j.w0 + j.wn - 1);
    const float sink_log2 =
        p.sink ? __ldg(p.sink + head) * CUDART_L2E_F : -CUDART_INF_F;
    // A cluster with a non-empty range has a part in slot c * 2 + (c == first).
    const auto empty = [&](int c) {
        return range_lo(c + 1, total, sched) <= range_lo(c, total, sched);
    };
    const auto sid = [&](int c) { return c * 2 + (c == first); };
    // One online-softmax pass over this group's parts.
    float m = -CUDART_INF_F, den = 0.0f;
    float acc[C::kPlaneCols] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int c = first + grp; c <= last; c += C::kCombSplit) {
        if (empty(c)) continue;
        const float lse = __ldg(p.lse_part + part_row<C>(sid(c), cta, head));
        if (!(lse > -CUDART_INF_F)) continue;
        const float nm = fmaxf(m, lse);
        const float sc = exp2f(m - nm), e = exp2f(lse - nm);
        den = den * sc + e;
        const float4 v = *reinterpret_cast<const float4*>(
            p.o_part +
            (int64_t((sid(c) * 2 + cta) * C::kPlanes + pl) * C::kWarpGroup +
             tsi) *
                C::kPlaneCols);
        acc[0] = acc[0] * sc + e * v.x;
        acc[1] = acc[1] * sc + e * v.y;
        acc[2] = acc[2] * sc + e * v.z;
        acc[3] = acc[3] * sc + e * v.w;
        m = nm;
    }
    CUTLASS_PRAGMA_UNROLL
    for (int e = 0; e < C::kPlaneCols; ++e) red[grp][tsi][e] = acc[e];
    redmd[grp][tsi][0] = m;
    redmd[grp][tsi][1] = den;
    __syncthreads();
    if (grp != 0) return;
    // Merge the group partials with the sink (denominator only), normalize.
    float mx = sink_log2;
    CUTLASS_PRAGMA_UNROLL
    for (int g = 0; g < C::kCombSplit; ++g) mx = fmaxf(mx, redmd[g][tsi][0]);
    float dtot = sink_log2 > -CUDART_INF_F ? exp2f(sink_log2 - mx) : 0.0f;
    CUTLASS_PRAGMA_UNROLL
    for (int e = 0; e < C::kPlaneCols; ++e) acc[e] = 0.0f;
    CUTLASS_PRAGMA_UNROLL
    for (int g = 0; g < C::kCombSplit; ++g) {
        const float mg = redmd[g][tsi][0];
        const float sc = mg > -CUDART_INF_F ? exp2f(mg - mx) : 0.0f;
        dtot += redmd[g][tsi][1] * sc;
        CUTLASS_PRAGMA_UNROLL
        for (int e = 0; e < C::kPlaneCols; ++e) acc[e] += red[g][tsi][e] * sc;
    }
    const float inv = dtot > 0.0f ? 1.0f / dtot : 0.0f;
    CUTLASS_PRAGMA_UNROLL
    for (int e = 0; e < C::kPlaneCols; ++e) acc[e] *= inv;
    const int mh = pl / (C::kOHalf / C::kPlaneCols);
    const int q = pl % (C::kOHalf / C::kPlaneCols);
    const int dim = half * C::kHalfDims + mh * C::kOHalf + q * C::kPlaneCols;
    __nv_bfloat16* const out =
        p.out + (int64_t(j.a + cta) * C::kHeads + head) * C::kDim + dim;
    reinterpret_cast<nv_bfloat162*>(out)[0] =
        __floats2bfloat162_rn(acc[0], acc[1]);
    reinterpret_cast<nv_bfloat162*>(out)[1] =
        __floats2bfloat162_rn(acc[2], acc[3]);
    pdl_trigger();
}

// Pre-dequant: the first blocks set c128_base per token, the others
// dequantize each request's reference c128 list into its copy rows.

CUTE_DEVICE int c128_len(const PairMlaParams& p, int t) {
    return max(min(__ldg(p.len[0] + t), p.ld[0]), 0);
}

CUTE_DEVICE int2 request_tokens(const PairMlaParams& p, int r) {
    if (!p.cu_seqlens)
        return {r * p.tokens_per_req, (r + 1) * p.tokens_per_req};
    return {__ldg(p.cu_seqlens + r), __ldg(p.cu_seqlens + r + 1)};
}

CUTE_DEVICE int request_of(const PairMlaParams& p, int t) {
    if (!p.cu_seqlens) return t / p.tokens_per_req;
    int lo = 0, hi = p.n_req;  // last request r with cu_seqlens[r] <= t
    while (hi - lo > 1) {
        const int mid = (lo + hi) / 2;
        if (__ldg(p.cu_seqlens + mid) <= t)
            lo = mid;
        else
            hi = mid;
    }
    return lo;
}

template <class C>
static __global__ void __launch_bounds__(PredqTraits::kThreads)
    predq_kernel(const __grid_constant__ PairMlaParams p) {
    pdl_wait();
    const int warp = threadIdx.x / C::kWarp, lane = threadIdx.x % C::kWarp;
    const int check_blocks =
        (p.tokens + PredqTraits::kWarps - 1) / PredqTraits::kWarps;
    if (int(blockIdx.x) < check_blocks) {
        const int t = blockIdx.x * PredqTraits::kWarps + warp;
        if (t >= p.tokens) return;
        const int r = request_of(p, t), ref = request_tokens(p, r).y - 1;
        const int len = c128_len(p, t);
        bool bad = len > c128_len(p, ref);
        if (!bad && t != ref) {
            const int* const mine = p.idx[0] + int64_t(t) * p.ld[0];
            const int* const theirs = p.idx[0] + int64_t(ref) * p.ld[0];
            int j = 0;
            if ((p.ld[0] & 3) == 0 &&
                (reinterpret_cast<uintptr_t>(p.idx[0]) & 15) == 0) {
                const int4* const m4 = reinterpret_cast<const int4*>(mine);
                const int4* const t4 = reinterpret_cast<const int4*>(theirs);
                const int n4 = len / 4;
#pragma unroll 4
                for (int k = lane; k < n4; k += C::kWarp) {
                    const int4 x = __ldg(m4 + k), y = __ldg(t4 + k);
                    bad |= x.x != y.x || x.y != y.y || x.z != y.z || x.w != y.w;
                }
                j = n4 * 4;
            }
            for (j += lane; j < len; j += C::kWarp) {
                bad |= __ldg(mine + j) != __ldg(theirs + j);
            }
        }
        bad = __any_sync(~0u, bad);
        if (lane == 0) p.c128_base[t] = bad ? -1 : r * p.c128_rows;
        return;
    }
    const int64_t g =
        int64_t(blockIdx.x - check_blocks) * PredqTraits::kWarps + warp;
    const int r = int(g / p.c128_rows), j = int(g % p.c128_rows);
    const int2 tok = request_tokens(p, r);
    if (tok.y <= tok.x || j >= c128_len(p, tok.y - 1)) return;
    const int key = __ldg(p.idx[0] + int64_t(tok.y - 1) * p.ld[0] + j);
    uint4* const dst = reinterpret_cast<uint4*>(p.c128_bf + g * C::kDim);
    if (key < 0 || key >= p.pool_tokens[0]) {
        dst[lane] = uint4{0, 0, 0, 0};
        dst[lane + C::kWarp] = uint4{0, 0, 0, 0};
        return;
    }
    const int64_t page = key >> p.page_shift[0];
    const int row = key & ((1 << p.page_shift[0]) - 1);
    const uint8_t* const base = p.pool[0] + page * p.pool_stride[0];
    const uint8_t* const src = base + int64_t(row) * C::kRowBytes;
    constexpr int kFp8Lanes = C::kNope / C::kChunk;
    if (lane < kFp8Lanes) {
        const uint4 raw = __ldg(reinterpret_cast<const uint4*>(src) + lane);
        const uint8_t e8 =
            __ldg(base + (int64_t(1) << p.page_shift[0]) * C::kRowBytes +
                  row * C::kSfBytes + lane * C::kChunk / C::kBlockDims);
        dst[2 * lane] = dequant8<C>(raw.x, raw.y, e8);
        dst[2 * lane + 1] = dequant8<C>(raw.z, raw.w, e8);
    } else {
        const uint4* const rope =
            reinterpret_cast<const uint4*>(src + C::kNope);
        const int c = 2 * (lane - kFp8Lanes);
        dst[2 * kFp8Lanes + c] = __ldg(rope + c);
        dst[2 * kFp8Lanes + c + 1] = __ldg(rope + c + 1);
    }
}

}  // namespace halops::mla::prefill_pair
