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

#include "kernel/mla/pair_mla/prefill/kernel_mainloop.cuh"

namespace halops::mla {

// Box col of Q row row: token * 64 (contiguous q), or the token (kQnr 3D map).
template <class C>
CUTE_DEVICE void prefill_load_q_box(const PrefillMaps& maps, uint64_t* bar,
                                    void* dst, int col, int row) {
    if constexpr (C::kQnr) {
        tma_load_3d_no_hint(&maps.q, bar, dst, col, 0, row);
    } else {
        tma_load_2d_no_hint(&maps.q, bar, dst, col, row);
    }
}

// Named-barrier sync of the Q/O warp group that consumes x.
template <class C>
CUTE_DEVICE void prefill_qo_sync_after(uint32_t x) {
    named_bar_red_or<C::kQoBar, C::kWarpGroup>(x);
}

// A unit's Q by TMA, one box per dims half.
template <class C>
CUTE_DEVICE void prefill_fetch_q(const PrefillMaps& maps, typename C::Smem& sm,
                                 int row, int t, uint4 (&v)[C::kQVecs]) {
    CUTLASS_PRAGMA_UNROLL
    for (int r = 0; r < C::kQoRounds; ++r) {
        if (t == 0) {
            cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(
                &sm.qs_full, 2 * C::kQBoxBytes);
            CUTLASS_PRAGMA_UNROLL
            for (int h = 0; h < 2; ++h) {
                prefill_load_q_box<C>(maps, &sm.qs_full, sm.qo_stage[h],
                                      (h * C::kQoRounds + r) * C::kQBoxDims,
                                      row);
            }
        }
        cutlass::arch::ClusterBarrier::wait(&sm.qs_full, r & 1);
        const uint32_t srow =
            pair_swizzled_row(sm.qo_stage[t / C::kHeads], t % C::kHeads);
        cute::for_each(
            cute::make_int_sequence<C::kChunksPerBox>{}, [&](auto q) {
                v[r * C::kChunksPerBox + q] =
                    lds_128b_xor<decltype(q)::value * C::kChunk>(srow);
            });
        prefill_qo_sync_after<C>(
            pair_xor_words(v + r * C::kChunksPerBox, C::kChunksPerBox));
    }
}

// kQst: round r of a token's Q (two 64-dim boxes) into Q buffer r % 2.
template <class C>
CUTE_DEVICE void prefill_issue_q(const PrefillMaps& maps, typename C::Smem& sm,
                                 int row, int r) {
    uint64_t* const bar = &sm.qb.qb_full[r % 2];
    cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(
        bar, 2 * C::kQBoxBytes);
    CUTLASS_PRAGMA_UNROLL
    for (int h = 0; h < 2; ++h) {
        prefill_load_q_box<C>(maps, bar, sm.qb.q_stage[r % 2][h],
                              (h * C::kQoRounds + r) * C::kQBoxDims, row);
    }
}

// fetch_q from the Q buffers: round r waits parity (r / 2) & 1.
template <class C>
CUTE_DEVICE void prefill_fetch_q2(const PrefillMaps& maps, typename C::Smem& sm,
                                  int row, int t, uint4 (&v)[C::kQVecs]) {
    CUTLASS_PRAGMA_UNROLL
    for (int r = 0; r < C::kQoRounds; ++r) {
        cutlass::arch::ClusterBarrier::wait(&sm.qb.qb_full[r % 2], (r / 2) & 1);
        const uint32_t srow = pair_swizzled_row(
            sm.qb.q_stage[r % 2][t / C::kHeads], t % C::kHeads);
        cute::for_each(
            cute::make_int_sequence<C::kChunksPerBox>{}, [&](auto q) {
                v[r * C::kChunksPerBox + q] =
                    lds_128b_xor<decltype(q)::value * C::kChunk>(srow);
            });
        prefill_qo_sync_after<C>(
            pair_xor_words(v + r * C::kChunksPerBox, C::kChunksPerBox));
        if (r + 2 < C::kQoRounds && t == 0)
            prefill_issue_q<C>(maps, sm, row, r + 2);
    }
}

// kQnr: lanes 0..15 of warp 1 copy a unit's 64 RoPE floats into sm.qfreq.
template <class C>
CUTE_DEVICE bool prefill_freq_lane(int t) {
    return t >= C::kFreqT0 && t < C::kFreqT0 + C::kFreqLanes;
}

// Position of a token, from int64 (pos64) or int32 q_pos.
CUTE_DEVICE int prefill_load_pos(const PrefillParams& p, int token) {
    return p.pos64 ? int(static_cast<const int64_t*>(p.q_pos)[token])
                   : static_cast<const int*>(p.q_pos)[token];
}

// This lane's 16 bytes of the RoPE frequencies at pos, by cp.async.
template <class C>
CUTE_DEVICE void prefill_load_freqs(const PrefillParams& p, float* qfreq,
                                    int pos, int t) {
    const int i = (t - C::kFreqT0) * C::kVecLen;
    cute::SM80_CP_ASYNC_CACHEGLOBAL<uint4>::copy(
        *reinterpret_cast<const uint4*>(p.q_freqs +
                                        int64_t(pos) * C::kQRopeDims + i),
        *reinterpret_cast<uint4*>(qfreq + i));
    cute::cp_async_fence();
}

// sglang fused_q_norm_rope with 1 / rms folded into the score scale (sm.rs):
// Q gets only RoPE; the squares are summed in the reference order.
template <class C>
CUTE_DEVICE void prefill_qo_norm_rope(const PrefillParams& p,
                                      typename C::Smem& sm, int t, int b,
                                      uint4 (&v)[C::kQVecs]) {
    const int head = t % C::kHeads;
    auto* const part = reinterpret_cast<float4(*)[C::kHeads]>(sm.qo_stage);
    static_assert(C::kQVecs / C::kVecLen * C::kHeads * sizeof(float4) <=
                  sizeof(sm.qo_stage));
    if (t < C::kHeads) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < C::kQVecs / C::kVecLen; ++i) {
            part[i][head] = make_float4(
                prefill_square_chain<C>(v[C::kVecLen * i], 0.f),
                prefill_square_chain<C>(v[C::kVecLen * i + 1], 0.f),
                prefill_square_chain<C>(v[C::kVecLen * i + 2], 0.f),
                prefill_square_chain<C>(v[C::kVecLen * i + C::kVecW], 0.f));
        }
    }
    if (prefill_freq_lane<C>(t))
        asm volatile("cp.async.wait_all;" ::: "memory");
    cutlass::arch::NamedBarrier::sync(C::kWarpGroup, C::kQoBar);
    if (t >= C::kHeads) {
        float c[C::kQVecs];
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < C::kQVecs / C::kVecLen; ++i) {
            const float4 a = part[i][head];
            const int vi = C::kVecLen * i;
            c[vi] = prefill_square_chain<C>(v[vi], a.x);
            c[vi + 1] = prefill_square_chain<C>(v[vi + 1], a.y);
            c[vi + 2] = prefill_square_chain<C>(v[vi + 2], a.z);
            c[vi + C::kVecW] = prefill_square_chain<C>(v[vi + C::kVecW], a.w);
        }
        CUTLASS_PRAGMA_UNROLL
        for (int d = C::kQVecs / 2; d >= 1; d /= 2) {
            CUTLASS_PRAGMA_UNROLL
            for (int l = 0; l < d; ++l) c[l] = add_rn_f32(c[l], c[l + d]);
        }
        const float n = rsqrt_approx(fma_rn_f32(c[0], 1.f / C::kDim, p.q_eps));
        sm.rs[b][head] = n * p.scale_log2;
        __syncwarp();
        if (t % C::kWarp == 0)
            cutlass::arch::ClusterBarrier::arrive(&sm.rs_full[b]);
        constexpr int kRope0 = C::kQVecs - C::kQRopeDims / C::kElemsPerChunk;
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < C::kQRopeDims / C::kElemsPerChunk; ++j) {
            const float4 f0 = reinterpret_cast<const float4*>(sm.qfreq)[2 * j];
            const float4 f1 =
                reinterpret_cast<const float4*>(sm.qfreq)[2 * j + 1];
            uint4& w = v[kRope0 + j];
            prefill_rope_pair(w.x, f0.x, f0.y);
            prefill_rope_pair(w.y, f0.z, f0.w);
            prefill_rope_pair(w.z, f1.x, f1.y);
            prefill_rope_pair(w.w, f1.z, f1.w);
        }
    }
    cutlass::arch::fence_view_async_shared();
    prefill_qo_sync_after<C>(pair_xor_words(v + C::kQVecs - 1, 1));
}

// A unit's Q into this lane's TMEM Q columns, then q_full.
template <class C>
CUTE_DEVICE void prefill_store_q(typename C::Smem& sm, uint32_t lane_base,
                                 int t, const uint4 (&v)[C::kQVecs]) {
    CUTLASS_PRAGMA_UNROLL
    for (int q = 0; q < C::kQVecs; ++q) {
        tmem_st_32dp32bNx<C::kVecLen>(lane_base + C::kTmemQ + C::kVecLen * q,
                                      &v[q]);
    }
    cutlass::arch::fence_view_async_tmem_store();
    tcgen05_before_thread_sync();
    __syncwarp();
    if (t % C::kWarp == 0)
        cutlass::arch::ClusterBarrier::arrive(&sm.q_full, 0, 1);
}

// Cluster of stream tile pos over the main kernel's bounds: the last c with
// bound[c] <= pos.
CUTE_DEVICE int prefill_cluster_of(const int* bound, int clusters, int pos) {
    int lo = 0, hi = clusters - 1;
    while (lo < hi) {
        const int mid = (lo + hi + 1) / 2;
        if (bound[mid] <= pos)
            lo = mid;
        else
            hi = mid - 1;
    }
    return lo;
}

// N floats of v as bf16 to global memory, kStoreCols at a time.
template <class C, int N>
CUTE_DEVICE void prefill_store_out(__nv_bfloat16* dst, const float* v) {
    CUTLASS_PRAGMA_UNROLL
    for (int q = 0; q < N; q += C::kStoreCols) {
        uint32_t w[C::kStoreWords];
        CUTLASS_PRAGMA_UNROLL
        for (int e = 0; e < C::kStoreWords; ++e) {
            const nv_bfloat162 b2 =
                __floats2bfloat162_rn(v[q + 2 * e], v[q + 2 * e + 1]);
            w[e] = reinterpret_cast<const uint32_t&>(b2);
        }
        st_global_v8_na(dst + q, w);
    }
}

// O leaves through qo_stage by TMA: round r writes dims [64r, 64r + 64).
template <class C>
CUTE_DEVICE void prefill_store_round(const PrefillMaps& maps,
                                     typename C::Smem& sm, int token, int t,
                                     int r, const uint32_t* w) {
    if (r > 0) {
        if (t == 0) cute::tma_store_wait<0>();
        cutlass::arch::NamedBarrier::sync(C::kWarpGroup, C::kQoBar);
    }
    const uint32_t row =
        pair_swizzled_row(sm.qo_stage[t / C::kHeads], t % C::kHeads);
    cute::for_each(cute::make_int_sequence<C::kChunksPerBox>{}, [&](auto q) {
        constexpr int Q = decltype(q)::value;
        sts_128b_xor<Q * C::kChunk>(
            row, w[C::kVecLen * Q], w[C::kVecLen * Q + 1],
            w[C::kVecLen * Q + 2], w[C::kVecLen * Q + C::kVecW]);
    });
    cutlass::arch::fence_view_async_shared();
    cutlass::arch::NamedBarrier::sync(C::kWarpGroup, C::kQoBar);
    if (t == 0) {
        CUTLASS_PRAGMA_UNROLL
        for (int h = 0; h < 2; ++h) {
            cute::SM90_TMA_STORE_2D::copy(&maps.out, sm.qo_stage[h],
                                          (h * C::kQoRounds + r) * C::kQBoxDims,
                                          token * C::kHeads);
        }
        cute::tma_store_arrive();
    }
}

// bf16 output of an unsplit unit; `write` is false for a lone token's CTA 1.
template <class C>
CUTE_DEVICE void prefill_store_out_rows(const PrefillMaps& maps,
                                        typename C::Smem& sm, int token,
                                        uint32_t lane_base, int t, float scale,
                                        bool write) {
    CUTLASS_PRAGMA_UNROLL
    for (int mh = 0; mh < 2; ++mh) {
        uint32_t w[C::kOHalf / 2];
        pair_read_o_bf16<C>(sm, lane_base, t, mh, scale, w);
        if (write) {
            prefill_store_round<C>(maps, sm, token, t, 2 * mh, w);
            prefill_store_round<C>(maps, sm, token, t, 2 * mh + 1,
                                   w + C::kRoundWords);
        }
    }
    if (write && t == 0) cute::tma_store_wait<0>();
}

// f(c) for each part of a split pair in cluster order, skipping empty ranges.
template <class F>
CUTE_DEVICE void prefill_for_parts(int clusters, const PrefillUnit& u,
                                   int first, const int* bound, F&& f) {
    const int end = u.w0 + u.wn;
    for (int c = first; c < clusters && bound[c] < end; ++c) {
        if (bound[c + 1] > bound[c]) f(c);
    }
}

// O of a finished unit to bf16 (sink in the denominator). A split pair is
// merged by its first tile's cluster, or by the last part to arrive. With
// combine, every part is written (owner in slot 1, continuations in slot 0)
// and CTA 0 records the job. A merger whose TMEM still holds O merges there
// (mode 1), streaming the other parts through idle KV stages; a deferred
// last part merges from global memory in the same order (mode 2).
template <class C>
CUTE_DEVICE void prefill_store_o(const PrefillParams& p,
                                 const PrefillMaps& maps, typename C::Smem& sm,
                                 const PrefillUnit& u, int unit, int cta,
                                 int cluster, uint32_t lane_base, int t,
                                 float sink_log2) {
    const int head = t % C::kHeads, half = t / C::kHeads;
    const int b = unit % 2;
    const int sched = sm.sched;
    const int first =
        u.split ? prefill_cluster_of(sm.bound, sched, u.w0) : cluster;
    const int last = u.split
                         ? prefill_cluster_of(sm.bound, sched, u.w0 + u.wn - 1)
                         : cluster;
    const bool dense = sm.dense;
    const auto part_cluster = [&](int j) {
        if (dense) return first + j;
        int c_j = first, n = 0;
        prefill_for_parts(sched, u, first, sm.bound, [&](int c) {
            if (n++ == j) c_j = c;
        });
        return c_j;
    };
    int n_parts = last - first + 1;
    if (u.split && !dense) {
        n_parts = 0;
        prefill_for_parts(sched, u, first, sm.bound, [&](int) { ++n_parts; });
    }
    const int others = n_parts - 1;
    if (u.split && first == cluster && t < C::kPartBoxes && !p.combine) {
        for (int k = 1; k <= others; ++k) {
            prefetch_l2_bulk(pair_part_box<C>(p, part_cluster(k), cta, t),
                             C::kPartBoxPlanes * C::kPlaneBytes);
        }
    }
    pair_qo_wait<C>(&sm.ml_full[b], (unit / 2) & 1, t);
    const float2 own = sm.ml[b][t], other = sm.ml[b][t ^ C::kHeads];
    __syncwarp();
    if (t % C::kWarp == 0)
        cutlass::arch::ClusterBarrier::arrive(&sm.ml_empty[b]);
    const float m = own.x, sum = own.y + other.y;
    const bool lone = cta == 1 && !u.g.has_b;
    if (!u.split) {
        const float scale =
            sum > 0.0f ? 1.0f / (sum + exp2f(sink_log2 - m)) : 0.0f;
        pair_qo_wait<C>(&sm.o_full, unit & 1, t);
        tcgen05_after_thread_sync();
        prefill_store_out_rows<C>(maps, sm, u.g.a + cta, lane_base, t, scale,
                                  !lone);
        return;
    }
    int* const ctr = p.counter + first * 2 + cta;
    __nv_bfloat16* const out =
        p.out + (int64_t(u.g.a + cta) * C::kHeads + head) * C::kDim +
        half * C::kHalfDims;
    const float own_scale = sum > 0.0f ? 1.0f / sum : 0.0f;
    const float own_lse = sum > 0.0f ? m + log2f(sum) : -CUDART_INF_F;
    pair_qo_wait<C>(&sm.o_full, unit & 1, t);
    tcgen05_after_thread_sync();
    if (lone) {
        if (first != cluster) {
            float o[C::kOHalf];
            pair_read_o_half<C>(sm, lane_base, t, 0, o);
            pair_read_o_half<C>(sm, lane_base, t, 1, o);
            cutlass::arch::NamedBarrier::sync(C::kWarpGroup, C::kQoBar);
        }
        return;
    }
    const auto write_part = [&](int sid) {
        if (half == 0) p.lse_part[pair_part_row<C>(sid, cta, head)] = own_lse;
        float* const dst =
            p.o_part +
            (int64_t(sid * 2 + cta) * C::kPlanes * C::kWarpGroup + t) *
                C::kPlaneCols;
        CUTLASS_PRAGMA_UNROLL
        for (int mh = 0; mh < 2; ++mh) {
            float o[C::kOHalf];
            pair_read_o_half<C>(sm, lane_base, t, mh, o);
            CUTLASS_PRAGMA_UNROLL
            for (int q = 0; q < C::kOHalf / C::kPlaneCols; ++q) {
                const int pl = mh * C::kOHalf / C::kPlaneCols + q;
                *reinterpret_cast<float4*>(dst +
                                           pl * C::kWarpGroup * C::kPlaneCols) =
                    make_float4(o[C::kVecLen * q] * own_scale,
                                o[C::kVecLen * q + 1] * own_scale,
                                o[C::kVecLen * q + 2] * own_scale,
                                o[C::kVecLen * q + C::kVecW] * own_scale);
            }
        }
        cutlass::arch::NamedBarrier::sync(C::kWarpGroup, C::kQoBar);
    };
    __shared__ int merge_mode;
    const auto broadcast = [&](int v) {
        if (t == 0) merge_mode = v;
        cutlass::arch::NamedBarrier::sync(C::kWarpGroup, C::kQoBar);
        return merge_mode;
    };
    const int merger_slot = p.clusters + first;
    int mode;
    if (first != cluster || p.combine) {
        const int sid = p.combine ? cluster * 2 + (first == cluster) : cluster;
        write_part(sid);
        if (p.combine) {
            if (first == cluster && cta == 0 && t == 0)
                p.jobs[cluster] = PairMergeJob{u.g.a, u.g.has_b, u.w0, u.wn};
            return;
        }
        int v = 0;
        if (t == 0)
            v = gmem_atom_add_acq_rel(ctr, 1) == C::kDeferred + others - 1 ? 2
                                                                           : 0;
        mode = broadcast(v);
    } else {
        int v = 0;
        if (t == 0) v = gmem_ld_acquire(ctr) == others ? 1 : 0;
        mode = broadcast(v);
        if (mode == 0) {
            write_part(merger_slot);
            if (t == 0)
                v = gmem_atom_add_acq_rel(ctr, C::kDeferred) == others ? 1 : 0;
            mode = broadcast(v);
        }
    }
    if (mode == 0) return;
    if (t == 0) {
        *ctr = 0;
        if (mode == 1) fence_proxy_async_global();
    }
    cutlass::arch::NamedBarrier::sync(C::kWarpGroup, C::kQoBar);
    const float lse0 =
        mode == 1
            ? own_lse
            : __ldcg(p.lse_part + pair_part_row<C>(merger_slot, cta, head));
    constexpr int kLseRegs = C::kLseRegs;
    float lse_r[kLseRegs];
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < kLseRegs; ++j) {
        lse_r[j] = j >= n_parts ? -CUDART_INF_F
                   : j == 0
                       ? lse0
                       : __ldcg(p.lse_part +
                                pair_part_row<C>(part_cluster(j), cta, head));
    }
    const auto lse_at = [&](int j) {
        if (j >= kLseRegs)
            return __ldcg(p.lse_part +
                          pair_part_row<C>(part_cluster(j), cta, head));
        float v = 0.0f;
        CUTLASS_PRAGMA_UNROLL
        for (int k = 0; k < kLseRegs; ++k) {
            if (k == j) v = lse_r[k];
        }
        return v;
    };
    float mx = sink_log2;
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < kLseRegs; ++j) mx = fmaxf(mx, lse_r[j]);
    for (int j = kLseRegs; j < n_parts; ++j) mx = fmaxf(mx, lse_at(j));
    float den = sink_log2 > -CUDART_INF_F ? exp2f(sink_log2 - mx) : 0.0f;
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < kLseRegs; ++j) {
        if (lse_r[j] > -CUDART_INF_F) den += exp2f(lse_r[j] - mx);
    }
    for (int j = kLseRegs; j < n_parts; ++j) {
        const float lse = lse_at(j);
        if (lse > -CUDART_INF_F) den += exp2f(lse - mx);
    }
    const float inv = den > 0.0f ? 1.0f / den : 0.0f;
    const float w_own = lse0 > -CUDART_INF_F ? exp2f(lse0 - mx) * inv : 0.0f;
    const uint32_t cols = lane_base + C::kTmemO;
    if (mode == 2) {
        constexpr int kCols = C::kGmemMergeCols;
        const auto plane = [&](int j, int pl) {
            const int slot = j == 0 ? merger_slot : part_cluster(j);
            return __ldcg(reinterpret_cast<const float4*>(
                p.o_part +
                ((int64_t(slot * 2 + cta) * C::kPlanes + pl) * C::kWarpGroup +
                 t) *
                    C::kPlaneCols));
        };
        for (int db = 0; db < C::kHalfDims / kCols; ++db) {
            float x[kCols];
            for (int j = 0; j < n_parts; ++j) {
                const float lse = j == 0 ? lse0 : lse_at(j);
                const float w =
                    lse > -CUDART_INF_F ? exp2f(lse - mx) * inv : 0.0f;
                float4 v[kCols / C::kPlaneCols];
                CUTLASS_PRAGMA_UNROLL
                for (int q = 0; q < kCols / C::kPlaneCols; ++q)
                    v[q] = plane(j, db * (kCols / C::kPlaneCols) + q);
                CUTLASS_PRAGMA_UNROLL
                for (int q = 0; q < kCols / C::kPlaneCols; ++q) {
                    const int xi = C::kVecLen * q;
                    if (j == 0) {
                        x[xi] = v[q].x * w;
                        x[xi + 1] = v[q].y * w;
                        x[xi + 2] = v[q].z * w;
                        x[xi + C::kVecW] = v[q].w * w;
                    } else {
                        x[xi] += v[q].x * w;
                        x[xi + 1] += v[q].y * w;
                        x[xi + 2] += v[q].z * w;
                        x[xi + C::kVecW] += v[q].w * w;
                    }
                }
            }
            prefill_store_out<C, kCols>(out + db * kCols, x);
        }
    } else {
        CUTLASS_PRAGMA_UNROLL
        for (int db = 0; db < C::kHalfDims / C::kMergeCols; ++db) {
            float x[C::kMergeCols];
            tmem_ld_32dp32bNx<C::kMergeCols>(cols + db * C::kMergeCols, x);
            cutlass::arch::fence_view_async_tmem_load();
            CUTLASS_PRAGMA_UNROLL
            for (int q = 0; q < C::kMergeCols; ++q)
                x[q] = x[q] * own_scale * w_own;
            tmem_st_32dp32bNx<C::kMergeCols>(cols + db * C::kMergeCols, x);
        }
        cutlass::arch::fence_view_async_tmem_store();
        cutlass::arch::NamedBarrier::sync(C::kWarpGroup, C::kQoBar);
        constexpr int kSubPerBox =
            C::kPartBoxPlanes / (C::kMergeCols / C::kPlaneCols);
        constexpr int kMergeBoxBytes = C::kPartBoxPlanes * C::kPlaneBytes;
        static_assert(C::kMergeBuf * kMergeBoxBytes <=
                      C::kKvStages * C::kStageBytes);
        const auto merge_buf = [&](int buf) {
            return sm.kv[0] + buf * kMergeBoxBytes;
        };
        const auto load_box = [&](int k, int bx, int buf) {
            if (t == 0) {
                cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(
                    &sm.merge_full[buf], C::kPartBoxPlanes * C::kPlaneBytes);
                cute::SM90_BULK_COPY_G2S::copy(
                    pair_part_box<C>(p, part_cluster(k), cta, bx),
                    &sm.merge_full[buf], merge_buf(buf),
                    C::kPartBoxPlanes * C::kPlaneBytes);
            }
        };
        const int nbox = others * C::kPartBoxes;
        for (int g = 0; g < C::kMergeBuf && g < nbox; ++g)
            load_box(1 + g / C::kPartBoxes, g % C::kPartBoxes, g);
        float w = 0.0f;
        for (int g = 0; g < nbox; ++g) {
            const int k = 1 + g / C::kPartBoxes, bx = g % C::kPartBoxes;
            const int buf = g % C::kMergeBuf;
            if (bx == 0) {
                const float lse = lse_at(k);
                w = lse > -CUDART_INF_F ? exp2f(lse - mx) * inv : 0.0f;
            }
            cutlass::arch::ClusterBarrier::wait(&sm.merge_full[buf],
                                                (g / C::kMergeBuf) & 1);
            CUTLASS_PRAGMA_UNROLL
            for (int sub = 0; sub < kSubPerBox; ++sub) {
                const int db = bx * kSubPerBox + sub;
                float x[C::kMergeCols];
                tmem_ld_32dp32bNx<C::kMergeCols>(cols + db * C::kMergeCols, x);
                cutlass::arch::fence_view_async_tmem_load();
                const uint8_t* const planes =
                    merge_buf(buf) +
                    sub * (C::kMergeCols / C::kPlaneCols) * C::kPlaneBytes +
                    t * C::kPlaneCols * sizeof(float);
                CUTLASS_PRAGMA_UNROLL
                for (int q = 0; q < C::kMergeCols / C::kPlaneCols; ++q) {
                    const float4 v = *reinterpret_cast<const float4*>(
                        planes + q * C::kPlaneBytes);
                    const int xi = C::kVecLen * q;
                    x[xi] += v.x * w;
                    x[xi + 1] += v.y * w;
                    x[xi + 2] += v.z * w;
                    x[xi + C::kVecW] += v.w * w;
                }
                if (k == others) {
                    prefill_store_out<C, C::kMergeCols>(
                        out + db * C::kMergeCols, x);
                } else {
                    tmem_st_32dp32bNx<C::kMergeCols>(cols + db * C::kMergeCols,
                                                     x);
                }
            }
            cutlass::arch::fence_view_async_tmem_store();
            cutlass::arch::NamedBarrier::sync(C::kWarpGroup, C::kQoBar);
            const int g2 = g + C::kMergeBuf;
            if (g2 < nbox)
                load_box(1 + g2 / C::kPartBoxes, g2 % C::kPartBoxes,
                         g2 % C::kMergeBuf);
        }
    }
}

// The Q/O warp group: loads each unit's Q (kQst: double-buffered, kQnr:
// with norm and RoPE) while the previous unit runs, then stores its O. A lone
// token's CTA 1 loads the valid token to balance the Q barriers.
template <class C>
CUTE_DEVICE void prefill_qo_wg(const PrefillParams& p, const PrefillMaps& maps,
                               typename C::Smem& sm, uint32_t tmem,
                               float sink_log2, int cta, int t, int cluster) {
    const uint32_t lane_base =
        tmem + ((t / C::kWarp * C::kWarp) << C::kTmemLaneShift);
    PrefillUnitCursor cur = prefill_start_cursor<C>(p, sm);
    PrefillUnit u, prev;
    bool more = cur.next<C>(p, sm.unit_lens, u);
    int unit = 0;
    constexpr bool kQst = C::kQst;
    const auto q_tok = [&](const PrefillUnit& v) {
        return v.g.a + (cta == 1 && !v.g.has_b ? 0 : cta);
    };
    const auto q_row = [&](const PrefillUnit& v) {
        return C::kQnr ? q_tok(v) : q_tok(v) * C::kHeads;
    };
    int qpos = 0;
    const auto next_pos = [&](bool known, const PrefillUnit& v) {
        if constexpr (C::kQnr) {
            if (known && prefill_freq_lane<C>(t))
                qpos = prefill_load_pos(p, q_tok(v));
        }
    };
    next_pos(more, u);
    if constexpr (kQst) {
        if (more && t == 0) {
            const int row = q_row(u);
            prefill_issue_q<C>(maps, sm, row, 0);
            prefill_issue_q<C>(maps, sm, row, 1);
        }
    }
    for (; more; ++unit) {
        uint4 qv[C::kQVecs];
        if constexpr (C::kQnr) {
            if (prefill_freq_lane<C>(t))
                prefill_load_freqs<C>(p, sm.qfreq, qpos, t);
        }
        PrefillUnit nxt;
        bool more_n = false;
        if constexpr (kQst) {
            prefill_fetch_q2<C>(maps, sm, t == 0 ? q_row(u) : 0, t, qv);
            more_n = cur.next<C>(p, sm.unit_lens, nxt);
            if (more_n && t == 0) {
                const int nrow = q_row(nxt);
                prefill_issue_q<C>(maps, sm, nrow, 0);
                prefill_issue_q<C>(maps, sm, nrow, 1);
            }
            next_pos(more_n, nxt);
        } else {
            prefill_fetch_q<C>(maps, sm, t == 0 ? q_row(u) : 0, t, qv);
        }
        if constexpr (C::kQnr) prefill_qo_norm_rope<C>(p, sm, t, unit % 2, qv);
        if (unit > 0) {
            pair_qo_wait<C>(&sm.q_empty, (unit - 1) & 1, t);
            tcgen05_after_thread_sync();
        }
        prefill_store_q<C>(sm, lane_base, t, qv);
        if (unit > 0) {
            prefill_store_o<C>(p, maps, sm, prev, unit - 1, cta, cluster,
                               lane_base, t, sink_log2);
        }
        prev = u;
        if constexpr (kQst) {
            more = more_n;
            if (more) u = nxt;
        } else {
            more = cur.next<C>(p, sm.unit_lens, u);
            next_pos(more, u);
        }
    }
    if (unit > 0) {
        prefill_store_o<C>(p, maps, sm, prev, unit - 1, cta, cluster, lane_base,
                           t, sink_log2);
    }
    if (t == 0) bulk_wait_all();
}

}  // namespace halops::mla
