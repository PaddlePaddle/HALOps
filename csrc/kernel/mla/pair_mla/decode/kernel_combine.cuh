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

#include <math_constants.h>

#include "kernel/mla/pair_mla/decode/kernel_schedule.cuh"

namespace halops::detail {

template <class C>
CUTE_DEVICE void merge_sync() {
    NamedBar::sync(C::kMergeThreads, C::kMergeBar);
}

// acc[u] += sum over parts k in [k0, k1) of w[k][head] * part k's plane
// pl0 + u * pstep (planes >= phi are skipped).
template <class C, int UP, int UK>
CUTE_DEVICE void merge_block(const HcaParams& p, const float* w, int first,
                             int cta, int tsi, int head, int pl0, int pstep,
                             int phi, int k0, int k1, float4* acc) {
    for (int kb = k0; kb < k1; kb += UK) {
        float wk[UK];
        float4 v[UK][UP];
        CUTLASS_PRAGMA_UNROLL
        for (int b = 0; b < UK; ++b) {
            const int k = kb + b;
            wk[b] = k < k1 ? w[k * C::kHeads + head] : 0.0f;
            const int sid = (first + k) * 2 + (k == 0);
            const float4* const src = reinterpret_cast<const float4*>(
                p.o_part +
                (int64_t(sid * 2 + cta) * C::kPlanes * kWarpGroup + tsi) *
                    C::kPlaneCols);
            CUTLASS_PRAGMA_UNROLL
            for (int u = 0; u < UP; ++u) {
                const int pl = pl0 + u * pstep;
                v[b][u] = k < k1 && pl < phi
                              ? __ldcg(src + pl * kWarpGroup)
                              : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
        CUTLASS_PRAGMA_UNROLL
        for (int b = 0; b < UK; ++b) {
            if (wk[b] == 0.0f) continue;
            CUTLASS_PRAGMA_UNROLL
            for (int u = 0; u < UP; ++u) {
                acc[u].x += wk[b] * v[b][u].x;
                acc[u].y += wk[b] * v[b][u].y;
                acc[u].z += wk[b] * v[b][u].z;
                acc[u].w += wk[b] * v[b][u].w;
            }
        }
    }
}

template <class C, int UP, int UK>
CUTE_DEVICE void merge_slice(const HcaParams& p, const float* w, float4* red,
                             uint2* stage, int first, int n, int cta, int plo,
                             int phi, int token) {
    const int wg = threadIdx.x / kWarpGroup, tsi = threadIdx.x % kWarpGroup;
    const int head = tsi % C::kHeads, half = tsi / C::kHeads;
    const bool ksplit = phi - plo < C::kMergeWgs;
    const int pstep = ksplit ? 1 : C::kMergeWgs;
    const int k0 = ksplit ? n * wg / C::kMergeWgs : 0;
    const int k1 = ksplit ? n * (wg + 1) / C::kMergeWgs : n;
    const int ns = phi - plo, pitch = ns | 1;
    for (int pb = plo + (ksplit ? 0 : wg); pb < phi; pb += UP * pstep) {
        float4 acc[UP];
        CUTLASS_PRAGMA_UNROLL
        for (int u = 0; u < UP; ++u)
            acc[u] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        merge_block<C, UP, UK>(p, w, first, cta, tsi, head, pb, pstep, phi, k0,
                               k1, acc);
        if (ksplit) {
            if (wg == 1) red[tsi] = acc[0];
            merge_sync<C>();
            if (wg == 1) continue;
            const float4 o = red[tsi];
            acc[0].x += o.x, acc[0].y += o.y, acc[0].z += o.z, acc[0].w += o.w;
        }
        CUTLASS_PRAGMA_UNROLL
        for (int u = 0; u < UP; ++u) {
            const int pl = pb + u * pstep;
            if (pl >= phi) break;
            const nv_bfloat162 lo = __floats2bfloat162_rn(acc[u].x, acc[u].y);
            const nv_bfloat162 hi = __floats2bfloat162_rn(acc[u].z, acc[u].w);
            stage[tsi * pitch + pl - plo] =
                make_uint2(*reinterpret_cast<const uint32_t*>(&lo),
                           *reinterpret_cast<const uint32_t*>(&hi));
        }
    }
    merge_sync<C>();
    static_assert(C::kHalfDims == C::kPlanes * C::kPlaneCols);
    for (int e = threadIdx.x; e < kWarpGroup * ns; e += C::kMergeThreads) {
        const int row = e / ns, c = e - row * ns;
        const int dim =
            (row / C::kHeads) * C::kHalfDims + (plo + c) * C::kPlaneCols;
        *reinterpret_cast<uint2*>(
            p.out +
            (int64_t(token + cta) * C::kHeads + row % C::kHeads) * C::kDim +
            dim) = stage[row * pitch + c];
    }
}

template <class C>
CUTE_DEVICE void fused_merge(const HcaParams& p, typename C::Smem& sm, int cta,
                             int cluster) {
    constexpr int kMaxParts = sizeof(sm.kv[0]) / (C::kHeads * sizeof(float));
    static_assert(kMaxParts >= C::kMinMergeParts &&
                  sizeof(sm.kv[1]) >= kWarpGroup * sizeof(float4));
    float* const w = reinterpret_cast<float*>(sm.kv[0]);
    float4* const red = reinterpret_cast<float4*>(sm.kv[1]);
    uint2* const stage = reinterpret_cast<uint2*>(sm.kv[2]);
    static_assert(sizeof(sm.kv) - 2 * sizeof(sm.kv[0]) >=
                  kWarpGroup * (C::kPlanes + 1) * sizeof(uint2));
    merge_sync<C>();
    const int pairs =
        (p.tokens / p.tokens_per_req) * ((p.tokens_per_req + 1) / 2);
    const int n_help = p.clusters - sm.sched;
    for (int i = 0; i < sm.n_mj; ++i) {
        const int4 j = sm.mj[i];
        const int first = j.x, n = j.y - j.x + 1, q = j.w >> 1;
        int* const ctr = p.counter + first * 2 + cta;
        const int r = cluster - first;
        const bool helper = cluster >= sm.sched;
        const int hq = min(q < n_help ? (n_help - 1 - q) / pairs + 1 : 0,
                           max(C::kPlanes - n, 0));
        const int hi = (cluster - sm.sched) / max(pairs, 1);
        if (helper && hi >= hq) continue;
        if (r == 0 && i > 0) continue;
        __shared__ int shared_first;
        if (threadIdx.x == 0) {
            int c;
            while (((c = ld_acquire(ctr)) & (C::kMergeShared - 1)) < n) {
            }
            shared_first = c >= C::kMergeShared;
        }
        merge_sync<C>();
        const int sh = shared_first, nn = n - sh, nm = nn + hq;
        const int rr = helper ? nn + hi : r - sh;
        const int plo = C::kPlanes * rr / nm, phi = C::kPlanes * (rr + 1) / nm;
        if (phi > plo && n <= kMaxParts) {
            constexpr int kS = C::kMergeThreads / C::kHeads;
            float2* const md = reinterpret_cast<float2*>(red);
            const int h = threadIdx.x % C::kHeads, s = threadIdx.x / C::kHeads;
            float m = -CUDART_INF_F, den = 0.0f;
            for (int k = s; k < n; k += kS) {
                const float l = __ldcg(
                    p.lse_part +
                    pair_part_row<C>((first + k) * 2 + (k == 0), cta, h));
                w[k * C::kHeads + h] = l;
                if (l > -CUDART_INF_F) {
                    const float nm = fmaxf(m, l);
                    den = den * exp2f(m - nm) + exp2f(l - nm);
                    m = nm;
                }
            }
            md[s * C::kHeads + h] = make_float2(m, den);
            merge_sync<C>();
            const float sink =
                p.sink ? __ldg(p.sink + h) * CUDART_L2E_F : -CUDART_INF_F;
            float mx = sink;
            CUTLASS_PRAGMA_UNROLL
            for (int g = 0; g < kS; ++g)
                mx = fmaxf(mx, md[g * C::kHeads + h].x);
            float d = sink > -CUDART_INF_F ? exp2f(sink - mx) : 0.0f;
            CUTLASS_PRAGMA_UNROLL
            for (int g = 0; g < kS; ++g) {
                const float2 v = md[g * C::kHeads + h];
                if (v.y > 0.0f) d += v.y * exp2f(v.x - mx);
            }
            const float inv = d > 0.0f ? 1.0f / d : 0.0f;
            for (int k = s; k < n; k += kS) {
                const float l = w[k * C::kHeads + h];
                w[k * C::kHeads + h] =
                    l > -CUDART_INF_F ? exp2f(l - mx) * inv : 0.0f;
            }
            merge_sync<C>();
            if (n <= 2) {
                merge_slice<C, C::kMergeLoads / 2, 2>(p, w, red, stage, first,
                                                      n, cta, plo, phi, j.z);
            } else if (n <= C::kMergeKMid) {
                merge_slice<C, C::kMergeLoads / C::kMergeKMid, C::kMergeKMid>(
                    p, w, red, stage, first, n, cta, plo, phi, j.z);
            } else if (n <= C::kMergeKWide) {
                merge_slice<C, C::kMergeLoads / C::kMergeKWide, C::kMergeKWide>(
                    p, w, red, stage, first, n, cta, plo, phi, j.z);
            } else {
                merge_slice<C, 1, C::kMergeLoads>(p, w, red, stage, first, n,
                                                  cta, plo, phi, j.z);
            }
            merge_sync<C>();
        }
        if (threadIdx.x == 0 &&
            (atomicAdd(ctr, 1) & (C::kMergeShared - 1)) == n + nm - 1)
            *ctr = 0;
    }
}

// The softmax and Q/O warp groups (the ones with registers to spare) merge
// from inside their own branches, so the merge is compiled under their
// register budgets rather than the aux groups' 56; the others go straight
// to the teardown barrier.
template <class C>
CUTE_DEVICE void merge_tail(const HcaParams& p, typename C::Smem& sm, int cta,
                            int cluster) {
    if constexpr (C::kFuse) {
        if (sm.comb && p.fuse) {
            pdl_trigger();
            fused_merge<C>(p, sm, cta, cluster);
        }
    }
}

template <class C>
__global__ void hca_combine_kernel(const __grid_constant__ HcaParams p) {
    __shared__ float red[HcaPlan::kCombSplit][kWarpGroup][C::kPlaneCols];
    __shared__ float redmd[HcaPlan::kCombSplit][kWarpGroup][2];
    pdl_trigger();
    pdl_wait();
    const int sched = *p.eff_g;
    if (sched == 0) return;
    const int cta = blockIdx.y;
    const int pl = blockIdx.x;
    const int grp = threadIdx.x / kWarpGroup;
    const int tsi = threadIdx.x % kWarpGroup;
    const int head = tsi % C::kHeads, half = tsi / C::kHeads;
    const int total = *p.total_g;
    const float sink_log2 =
        p.sink ? __ldg(p.sink + head) * CUDART_L2E_F : -CUDART_INF_F;
    const int pairs =
        (p.tokens / p.tokens_per_req) * ((p.tokens_per_req + 1) / 2);
    for (int z = blockIdx.z; z < pairs; z += gridDim.z) {
        const PairMergeJob j = p.jobs[z];
        if (j.wn == 0 || (cta == 1 && !j.has_b)) continue;
        const int first = pair_cluster_of(sched, total, j.w0);
        const int last = pair_cluster_of(sched, total, j.w0 + j.wn - 1);
        const auto empty = [&](int c) {
            return range_lo(c + 1, total, sched) <= range_lo(c, total, sched);
        };
        const auto sid = [&](int c) { return c * 2 + (c == first); };
        float m = -CUDART_INF_F, den = 0.0f;
        float acc[C::kPlaneCols] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int c = first + grp; c <= last; c += HcaPlan::kCombSplit) {
            if (empty(c)) continue;
            const float lse =
                __ldg(p.lse_part + pair_part_row<C>(sid(c), cta, head));
            if (!(lse > -CUDART_INF_F)) continue;
            const float nm = fmaxf(m, lse);
            const float sc = exp2f(m - nm), e = exp2f(lse - nm);
            den = den * sc + e;
            const float4 v = *reinterpret_cast<const float4*>(
                p.o_part +
                (int64_t((sid(c) * 2 + cta) * C::kPlanes + pl) * kWarpGroup +
                 tsi) *
                    C::kPlaneCols);
            acc[0] = acc[0] * sc + e * v.x;
            acc[1] = acc[1] * sc + e * v.y;
            acc[2] = acc[2] * sc + e * v.z;
            acc[kQuad - 1] = acc[kQuad - 1] * sc + e * v.w;
            m = nm;
        }
        CUTLASS_PRAGMA_UNROLL
        for (int e = 0; e < C::kPlaneCols; ++e) red[grp][tsi][e] = acc[e];
        redmd[grp][tsi][0] = m;
        redmd[grp][tsi][1] = den;
        __syncthreads();
        if (grp == 0) {
            float mx = sink_log2;
            CUTLASS_PRAGMA_UNROLL
            for (int g = 0; g < HcaPlan::kCombSplit; ++g)
                mx = fmaxf(mx, redmd[g][tsi][0]);
            float dtot =
                sink_log2 > -CUDART_INF_F ? exp2f(sink_log2 - mx) : 0.0f;
            CUTLASS_PRAGMA_UNROLL
            for (int e = 0; e < C::kPlaneCols; ++e) acc[e] = 0.0f;
            CUTLASS_PRAGMA_UNROLL
            for (int g = 0; g < HcaPlan::kCombSplit; ++g) {
                const float sc = exp2f(redmd[g][tsi][0] - mx);
                dtot += redmd[g][tsi][1] * sc;
                CUTLASS_PRAGMA_UNROLL
                for (int e = 0; e < C::kPlaneCols; ++e)
                    acc[e] += red[g][tsi][e] * sc;
            }
            const float inv = dtot > 0.0f ? 1.0f / dtot : 0.0f;
            CUTLASS_PRAGMA_UNROLL
            for (int e = 0; e < C::kPlaneCols; ++e) acc[e] *= inv;
            const int mh = pl / (C::kOHalf / C::kPlaneCols);
            const int q = pl % (C::kOHalf / C::kPlaneCols);
            const int dim =
                half * C::kHalfDims + mh * C::kOHalf + q * C::kPlaneCols;
            __nv_bfloat16* const out =
                p.out + (int64_t(j.a + cta) * C::kHeads + head) * C::kDim + dim;
            reinterpret_cast<nv_bfloat162*>(out)[0] =
                __floats2bfloat162_rn(acc[0], acc[1]);
            reinterpret_cast<nv_bfloat162*>(out)[1] =
                __floats2bfloat162_rn(acc[2], acc[kQuad - 1]);
        }
        __syncthreads();
    }
}

}  // namespace halops::detail
