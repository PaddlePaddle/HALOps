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

// A later unit's Q comes by TMA, one box per dims half at a time: a lane's
// head row is strided, and direct loads of it occupy the L1 a row per lane,
// stalling the softmax's shared memory accesses meanwhile.
template <class C>
CUTE_DEVICE void fetch_q(const HcaMaps& maps, typename C::Smem& sm, int token,
                         int t, uint4 (&v)[C::kQVecs]) {
    CUTLASS_PRAGMA_UNROLL
    for (int r = 0; r < C::kQoRounds; ++r) {
        if (t == 0) {
            cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(
                &sm.qs_full, 2 * C::kQBoxBytes);
            CUTLASS_PRAGMA_UNROLL
            for (int h = 0; h < 2; ++h) {
                tma_load_2d(&maps.q, &sm.qs_full, sm.qo_stage[h],
                            (h * C::kQoRounds + r) * C::kQBoxDims,
                            token * C::kHeads);
            }
        }
        Bar::wait(&sm.qs_full, r & 1);
        const uint32_t row =
            pair_swizzled_row(sm.qo_stage[t / C::kHeads], t % C::kHeads);
        cute::for_each(cute::make_int_sequence<C::kChunksPerBox>{},
                       [&](auto q) {
                           v[r * C::kChunksPerBox + q] =
                               lds_chunk<decltype(q)::value>(row);
                       });
        qo_sync_after<C>(
            pair_xor_words(v + r * C::kChunksPerBox, C::kChunksPerBox));
    }
}

template <class C>
CUTE_DEVICE void store_q(const HcaParams& p, typename C::Smem& sm,
                         uint32_t lane_base, int token, int cta, int t,
                         const uint4 (&v)[C::kQVecs]) {
    CUTLASS_PRAGMA_UNROLL
    for (int q = 0; q < C::kQVecs; ++q) {
        tmem_st_32dp32bNx<kQuad>(lane_base + C::kTmemQ + kQuad * q, &v[q]);
    }
    if constexpr (C::kDsa) {
        const int head = t % C::kHeads, g = t / C::kHeads;
        if (g == 0) {
            const __nv_bfloat16* src =
                p.q + (int64_t(token) * C::kHeads + head) * C::kDsaQk +
                C::kDsaNope;
            CUTLASS_PRAGMA_UNROLL
            for (int c = 0; c < C::kDsaRope / kElemsPerChunk; ++c) {
                *reinterpret_cast<uint4*>(sm.q_rope +
                                          pair_sw128_offset(head, c)) =
                    *reinterpret_cast<const uint4*>(src + c * kElemsPerChunk);
            }
        }
        cutlass::arch::fence_view_async_shared();
    }
    cutlass::arch::fence_view_async_tmem_store();
    tcgen05_before_thread_sync();
    __syncwarp();
    if (t % kWarp == 0) Bar::arrive(&sm.q_full, 0, 1);
}

template <class C>
CUTE_DEVICE void store_round(const HcaMaps& maps, typename C::Smem& sm,
                             int token, int t, int r, const uint32_t* w) {
    if (r > 0) {
        if (t == 0) bulk_wait_read();
        NamedBar::sync(kWarpGroup, C::kQoBar);
    }
    const uint32_t row =
        pair_swizzled_row(sm.qo_stage[t / C::kHeads], t % C::kHeads);
    cute::for_each(cute::make_int_sequence<C::kChunksPerBox>{}, [&](auto q) {
        constexpr int Q = decltype(q)::value;
        sts_chunk<Q>(row, w[kQuad * Q], w[kQuad * Q + 1], w[kQuad * Q + 2],
                     w[kQuad * Q + kQuad - 1]);
    });
    cutlass::arch::fence_view_async_shared();
    NamedBar::sync(kWarpGroup, C::kQoBar);
    if (t == 0) {
        CUTLASS_PRAGMA_UNROLL
        for (int h = 0; h < 2; ++h) {
            tma_store_2d(&maps.out, sm.qo_stage[h],
                         (h * C::kQoRounds + r) * C::kQBoxDims,
                         token * C::kHeads);
        }
        bulk_commit();
    }
}

// The bf16 output of an unsplit unit, a half at a time.  `write` is false for
// a lone token's CTA 1: it still drains O (releasing it to the next unit's PV)
// but stores nothing, as the row it would write belongs to the next request.
// The cluster's last unit stages all of O at once in the idle KV stages, so
// no round waits for the previous one's box to be read.
template <class C>
CUTE_DEVICE void store_out_rows(const HcaMaps& maps, typename C::Smem& sm,
                                int token, uint32_t lane_base, int t,
                                float scale, bool write, bool last) {
    static_assert(2 * C::kQoRounds * C::kQBoxBytes <= sizeof(sm.kv));
    if (!C::kDsa && last && write) {
        const auto box = [&](int h, int r) {
            return sm.kv[0] + (h * C::kQoRounds + r) * C::kQBoxBytes;
        };
        CUTLASS_PRAGMA_UNROLL
        for (int mh = 0; mh < 2; ++mh) {
            uint32_t w[C::kOHalf / 2];
            pair_read_o_bf16<C>(sm, lane_base, t, mh, scale, w);
            CUTLASS_PRAGMA_UNROLL
            for (int k = 0; k < 2; ++k) {
                const uint32_t row = pair_swizzled_row(
                    box(t / C::kHeads, 2 * mh + k), t % C::kHeads);
                const uint32_t* wr = w + k * C::kRoundWords;
                cute::for_each(
                    cute::make_int_sequence<C::kChunksPerBox>{}, [&](auto q) {
                        constexpr int Q = decltype(q)::value;
                        sts_chunk<Q>(row, wr[kQuad * Q], wr[kQuad * Q + 1],
                                     wr[kQuad * Q + 2],
                                     wr[kQuad * Q + kQuad - 1]);
                    });
            }
            cutlass::arch::fence_view_async_shared();
            NamedBar::sync(kWarpGroup, C::kQoBar);
            if (t == 0) {
                CUTLASS_PRAGMA_UNROLL
                for (int k = 0; k < 2; ++k) {
                    CUTLASS_PRAGMA_UNROLL
                    for (int h = 0; h < 2; ++h) {
                        const int r = 2 * mh + k;
                        tma_store_2d(&maps.out, box(h, r),
                                     (h * C::kQoRounds + r) * C::kQBoxDims,
                                     token * C::kHeads);
                    }
                }
                bulk_commit();
            }
        }
        if (t == 0) bulk_wait_read();
        return;
    }
    CUTLASS_PRAGMA_UNROLL
    for (int mh = 0; mh < 2; ++mh) {
        uint32_t w[C::kOHalf / 2];
        pair_read_o_bf16<C>(sm, lane_base, t, mh, scale, w);
        if (write) {
            store_round<C>(maps, sm, token, t, 2 * mh, w);
            store_round<C>(maps, sm, token, t, 2 * mh + 1, w + C::kRoundWords);
        }
    }
    if (write && t == 0) bulk_wait_read();
}

// Calls f(c) for the parts of a split pair in cluster order: clusters
// first.. up to the one holding the pair's last tile, skipping clusters with
// an empty range.
template <class F>
CUTE_DEVICE void for_parts(int clusters, const Unit& u, int first, int total,
                           F&& f) {
    const int end = u.w0 + u.wn;
    for (int c = first; c < clusters && range_lo(c, total, clusters) < end;
         ++c) {
        if (range_lo(c + 1, total, clusters) > range_lo(c, total, clusters)) {
            f(c);
        }
    }
}

// O of a finished unit: bf16 output with the sink in the denominator.  A
// split pair is merged by the cluster holding its first tile: that part is
// the cluster's last unit, so its KV stages are idle to stage the other
// parts, which are the first (or only) units of the following clusters and
// publish their normalized O and log2 sum by then.  The merge adds the parts
// in cluster order, in fp32 in TMEM, so it does not depend on timing.  Each
// half of O is released to the next unit's first PV once it is read.
template <class C>
CUTE_DEVICE void store_o(const HcaParams& p, const HcaMaps& maps,
                         typename C::Smem& sm, const Unit& u, int unit, int cta,
                         int cluster, uint32_t lane_base, int t,
                         float sink_log2, bool final_unit) {
    const int head = t % C::kHeads, half = t / C::kHeads;
    const int b = unit % 2;
    const int total = sm.total;
    const int sched = sm.sched;
    const int first = pair_cluster_of(sched, total, u.w0);
    const int last = pair_cluster_of(sched, total, u.w0 + u.wn - 1);
    const bool dense = total >= sched;
    const auto part_cluster = [&](int j) {
        if (dense) return first + j;
        int c_j = first, n = 0;
        for_parts(sched, u, first, total, [&](int c) {
            if (n++ == j) c_j = c;
        });
        return c_j;
    };
    int n_parts = last - first + 1;
    if (u.split && !dense) {
        n_parts = 0;
        for_parts(sched, u, first, total, [&](int) { ++n_parts; });
    }
    const int others = n_parts - 1;
    if (u.split && first == cluster && t < C::kPartBoxes && !sm.comb) {
        for (int k = 1; k <= others; ++k) {
            prefetch_l2_bulk(pair_part_box<C>(p, part_cluster(k), cta, t),
                             C::kPartBoxPlanes * C::kPlaneBytes);
        }
    }
    pair_qo_wait<C>(&sm.ml_full[b], (unit / 2) & 1, t);
    const float2 own = sm.ml[b][t], other = sm.ml[b][t ^ C::kHeads];
    __syncwarp();
    if (t % kWarp == 0) Bar::arrive(&sm.ml_empty[b]);
    const float m = own.x, sum = own.y + other.y;
    const bool lone = cta == 1 && !u.g.has_b;
    if (!u.split) {
        if (sm.comb && !p.fuse && cta == 0 && t == 0) p.jobs[u.pair].wn = 0;
        const float scale =
            sum > 0.0f ? 1.0f / (sum + exp2f(sink_log2 - m)) : 0.0f;
        pair_qo_wait<C>(&sm.o_full, unit & 1, t);
        tcgen05_after_thread_sync();
        store_out_rows<C>(maps, sm, u.g.a + cta, lane_base, t, scale, !lone,
                          final_unit);
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
            NamedBar::sync(kWarpGroup, C::kQoBar);
        }
        return;
    }
    if (first != cluster || p.nomerge || sm.comb) {
        const int sid = sm.comb ? cluster * 2 + (first == cluster) : cluster;
        if (half == 0) p.lse_part[pair_part_row<C>(sid, cta, head)] = own_lse;
        float* const dst =
            p.o_part + (int64_t(sid * 2 + cta) * C::kPlanes * kWarpGroup + t) *
                           C::kPlaneCols;
        CUTLASS_PRAGMA_UNROLL
        for (int mh = 0; mh < 2; ++mh) {
            float o[C::kOHalf];
            pair_read_o_half<C>(sm, lane_base, t, mh, o);
            CUTLASS_PRAGMA_UNROLL
            for (int q = 0; q < C::kOHalf / C::kPlaneCols; ++q) {
                const int pl = mh * C::kOHalf / C::kPlaneCols + q;
                *reinterpret_cast<float4*>(dst +
                                           pl * kWarpGroup * C::kPlaneCols) =
                    make_float4(o[kQuad * q] * own_scale,
                                o[kQuad * q + 1] * own_scale,
                                o[kQuad * q + 2] * own_scale,
                                o[kQuad * q + kQuad - 1] * own_scale);
            }
        }
        NamedBar::sync(kWarpGroup, C::kQoBar);
        if (C::kFuse && sm.comb && p.fuse) {
            if (t == 0) {
                const bool shared = first == cluster && sm.n_mj > 0;
                red_release_add(ctr, 1 + (shared ? C::kMergeShared : 0));
                sm.mj[sm.n_mj++] =
                    make_int4(first, last, u.g.a, u.pair * 2 + u.g.has_b);
            }
        } else if (sm.comb) {
            if (first == cluster && cta == 0 && t == 0) {
                p.jobs[u.pair] = PairMergeJob{u.g.a, u.g.has_b, u.w0, u.wn};
            }
        } else if (t == 0) {
            red_release_add(ctr, 1);
        }
        return;
    }
    if (t == 0) {
        while (ld_acquire(ctr) != others) {
        }
        *ctr = 0;
        fence_proxy_async_global();
    }
    NamedBar::sync(kWarpGroup, C::kQoBar);
    const auto stage_part = [&](int k) {
        if (t == 0) {
            cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(
                &sm.merge_full[0], C::kPlanes * C::kPlaneBytes);
        }
        NamedBar::sync(kWarpGroup, C::kQoBar);
        static_assert(C::kPartBoxes == kWarpsPerGroup);
        if (t % kWarp == 0) {
            const int bx = t / kWarp;
            bulk_load(sm.kv[0] + bx * C::kPartBoxPlanes * C::kPlaneBytes,
                      pair_part_box<C>(p, part_cluster(k + 1), cta, bx),
                      C::kPartBoxPlanes * C::kPlaneBytes, &sm.merge_full[0]);
        }
    };
    float lse_r[C::kLseRegs];
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < C::kLseRegs; ++j) {
        lse_r[j] = j >= n_parts ? -CUDART_INF_F
                   : j == 0
                       ? own_lse
                       : __ldcg(p.lse_part +
                                pair_part_row<C>(part_cluster(j), cta, head));
    }
    const auto lse_at = [&](int j) {
        if (j >= C::kLseRegs)
            return __ldcg(p.lse_part +
                          pair_part_row<C>(part_cluster(j), cta, head));
        float v = 0.0f;
        CUTLASS_PRAGMA_UNROLL
        for (int k = 0; k < C::kLseRegs; ++k) {
            if (k == j) v = lse_r[k];
        }
        return v;
    };
    float mx = sink_log2;
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < C::kLseRegs; ++j) mx = fmaxf(mx, lse_r[j]);
    for (int j = C::kLseRegs; j < n_parts; ++j) mx = fmaxf(mx, lse_at(j));
    float den = sink_log2 > -CUDART_INF_F ? exp2f(sink_log2 - mx) : 0.0f;
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < C::kLseRegs; ++j) {
        if (lse_r[j] > -CUDART_INF_F) den += exp2f(lse_r[j] - mx);
    }
    for (int j = C::kLseRegs; j < n_parts; ++j) {
        const float lse = lse_at(j);
        if (lse > -CUDART_INF_F) den += exp2f(lse - mx);
    }
    const float inv = den > 0.0f ? 1.0f / den : 0.0f;
    const float w_own =
        own_lse > -CUDART_INF_F ? exp2f(own_lse - mx) * inv : 0.0f;
    static_assert(C::kPlanes * C::kPlaneBytes <= C::kStages * C::kStageBytes);
    const uint32_t cols = lane_base + C::kTmemO;
    if (p.pipe) {
        CUTLASS_PRAGMA_UNROLL
        for (int db = 0; db < C::kHalfDims / C::kMergeCols; ++db) {
            float x[C::kMergeCols];
            tmem_ld_32dp32bNx<C::kMergeCols>(cols + db * C::kMergeCols, x);
            cutlass::arch::fence_view_async_tmem_load();
            CUTLASS_PRAGMA_UNROLL
            for (int q = 0; q < C::kMergeCols; ++q) x[q] *= own_scale * w_own;
            tmem_st_32dp32bNx<C::kMergeCols>(cols + db * C::kMergeCols, x);
        }
        cutlass::arch::fence_view_async_tmem_store();
        NamedBar::sync(kWarpGroup, C::kQoBar);
        constexpr int kSubPerBox =
            C::kPartBoxPlanes / (C::kMergeCols / C::kPlaneCols);
        constexpr int kMergeBoxBytes = C::kPartBoxPlanes * C::kPlaneBytes;
        static_assert(C::kMergeBuf * kMergeBoxBytes <=
                      C::kStages * C::kStageBytes);
        const auto merge_buf = [&](int buf) {
            return sm.kv[0] + buf * kMergeBoxBytes;
        };
        const auto load_box = [&](int k, int bx, int buf) {
            if (t == 0) {
                cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(
                    &sm.merge_full[buf], C::kPartBoxPlanes * C::kPlaneBytes);
                bulk_load(merge_buf(buf),
                          pair_part_box<C>(p, part_cluster(k), cta, bx),
                          C::kPartBoxPlanes * C::kPlaneBytes,
                          &sm.merge_full[buf]);
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
            Bar::wait(&sm.merge_full[buf], (g / C::kMergeBuf) & 1);
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
                    x[kQuad * q] += v.x * w;
                    x[kQuad * q + 1] += v.y * w;
                    x[kQuad * q + 2] += v.z * w;
                    x[kQuad * q + kQuad - 1] += v.w * w;
                }
                if (k == others) {
                    store_out<C::kMergeCols>(out + db * C::kMergeCols, x, 1.0f);
                } else {
                    tmem_st_32dp32bNx<C::kMergeCols>(cols + db * C::kMergeCols,
                                                     x);
                }
            }
            cutlass::arch::fence_view_async_tmem_store();
            NamedBar::sync(kWarpGroup, C::kQoBar);
            const int g2 = g + C::kMergeBuf;
            if (g2 < nbox)
                load_box(1 + g2 / C::kPartBoxes, g2 % C::kPartBoxes,
                         g2 % C::kMergeBuf);
        }
    } else {
        stage_part(0);
        for (int k = 0; k < others; ++k) {
            const float lse = lse_at(k + 1);
            const float w = lse > -CUDART_INF_F ? exp2f(lse - mx) * inv : 0.0f;
            if (k > 0) stage_part(k);
            Bar::wait(&sm.merge_full[0], k & 1);
            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < C::kHalfDims / C::kMergeCols; j += 2) {
                float x[2][C::kMergeCols];
                tmem_ld_32dp32bNx<C::kMergeCols>(cols + j * C::kMergeCols,
                                                 x[0]);
                tmem_ld_32dp32bNx<C::kMergeCols>(cols + (j + 1) * C::kMergeCols,
                                                 x[1]);
                cutlass::arch::fence_view_async_tmem_load();
                CUTLASS_PRAGMA_UNROLL
                for (int jj = 0; jj < 2; ++jj) {
                    if (k == 0) {
                        CUTLASS_PRAGMA_UNROLL
                        for (int q = 0; q < C::kMergeCols; ++q)
                            x[jj][q] = x[jj][q] * own_scale * w_own;
                    }
                    const uint8_t* const planes =
                        sm.kv[0] +
                        (j + jj) * C::kMergeCols / C::kPlaneCols *
                            C::kPlaneBytes +
                        t * C::kPlaneCols * sizeof(float);
                    CUTLASS_PRAGMA_UNROLL
                    for (int q = 0; q < C::kMergeCols / C::kPlaneCols; ++q) {
                        const float4 v = *reinterpret_cast<const float4*>(
                            planes + q * C::kPlaneBytes);
                        x[jj][kQuad * q] += v.x * w;
                        x[jj][kQuad * q + 1] += v.y * w;
                        x[jj][kQuad * q + 2] += v.z * w;
                        x[jj][kQuad * q + kQuad - 1] += v.w * w;
                    }
                    if (k == others - 1) {
                        store_out<C::kMergeCols>(out + (j + jj) * C::kMergeCols,
                                                 x[jj], 1.0f);
                    } else {
                        tmem_st_32dp32bNx<C::kMergeCols>(
                            cols + (j + jj) * C::kMergeCols, x[jj]);
                    }
                }
            }
            cutlass::arch::fence_view_async_tmem_store();
            NamedBar::sync(kWarpGroup, C::kQoBar);
        }
    }
}

template <class C>
CUTE_DEVICE void qo_wg(const HcaParams& p, const HcaMaps& maps,
                       typename C::Smem& sm, uint32_t tmem, float sink_log2,
                       int cta, int t, int cluster) {
    const uint32_t lane_base =
        tmem + ((t / kWarp * kWarp) << HcaIsa::kTmemLaneShift);
    UnitCursor<C> cur = start_cursor<C>(p, sm, cluster);
    Unit u, prev;
    bool more = cur.next(p, sm.unit_lens, u);
    int unit = 0;
    for (; more; ++unit) {
        uint4 qv[C::kQVecs];
        const int q_token = u.g.a + (cta == 1 && !u.g.has_b ? 0 : cta);
        fetch_q<C>(maps, sm, q_token, t, qv);
        if (unit > 0) {
            pair_qo_wait<C>(&sm.q_empty, (unit - 1) & 1, t);
            tcgen05_after_thread_sync();
        }
        store_q<C>(p, sm, lane_base, q_token, cta, t, qv);
        if (unit > 0) {
            store_o<C>(p, maps, sm, prev, unit - 1, cta, cluster, lane_base, t,
                       sink_log2, false);
        }
        prev = u;
        more = cur.next(p, sm.unit_lens, u);
    }
    if (unit > 0) {
        store_o<C>(p, maps, sm, prev, unit - 1, cta, cluster, lane_base, t,
                   sink_log2, true);
    }
    if (t == 0) bulk_store_wait_all();
}

}  // namespace halops::detail
