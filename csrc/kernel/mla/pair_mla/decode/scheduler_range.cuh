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

#include <cute/arch/cluster_sm90.hpp>

#include "kernel/mla/pair_mla/decode/config.cuh"
#include "kernel/mla/pair_mla/decode/params.cuh"
#include "kernel/mla/pair_mla/decode/ptx.cuh"
#include "kernel/mla/pair_mla/decode/scheduler.cuh"

namespace halops::mla {

// All threads count the tiles of every pair and scan them; the result is
// the first pair and tile of this cluster's range.
template <class C>
CUTE_DEVICE void decode_find_range(const DecodeParams& p, typename C::Smem& sm,
                                   int cluster) {
    __shared__ int part[C::kThreads / kWarp], part_max[C::kThreads / kWarp];
    __shared__ int part2[C::kThreads / kWarp];
    const int pairs =
        (p.tokens / p.tokens_per_req) * ((p.tokens_per_req + 1) / 2);
    const int per = (pairs + C::kThreads - 1) / C::kThreads;
    const int first = threadIdx.x * per;
    const int lane = threadIdx.x % kWarp, warp = threadIdx.x / kWarp;
    int4 keep[C::kKeep];
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < C::kKeep; ++j) keep[j] = make_int4(0, 0, 0, 0);
    const auto lens_of = [&](int q) {
        const int j = q - first;
        if (j >= 0 && j < min(per, C::kKeep)) {
            int4 v = keep[0];
            CUTLASS_PRAGMA_UNROLL
            for (int k = 1; k < C::kKeep; ++k) {
                if (j == k) v = keep[k];
            }
            return PairLens{{{v.x, v.y}, {v.z, v.w}}};
        }
        return decode_load_lens(p, q);
    };

    int sum_unc = 0, max_unc = 0;
    for (int q = first, j = 0; q < min(first + per, pairs); ++q, ++j) {
        const PairLens l = decode_load_lens(p, q);
        CUTLASS_PRAGMA_UNROLL
        for (int k = 0; k < C::kKeep; ++k) {
            if (j == k)
                keep[k] = make_int4(l.v[0][0], l.v[0][1], l.v[1][0], l.v[1][1]);
        }
        const int n = decode_pair_geom(p, q, l, C::kTile).n_tiles;
        sum_unc += n;
        max_unc = max(max_unc, n);
    }
    int incl_unc = sum_unc;
    {
        int m = max_unc;
        CUTLASS_PRAGMA_UNROLL
        for (int d = 1; d < kWarp; d *= 2) {
            const int v = __shfl_up_sync(~0u, incl_unc, d);
            if (lane >= d) incl_unc += v;
            m = max(m, __shfl_xor_sync(~0u, m, d));
        }
        if (lane == kWarp - 1) {
            part[warp] = incl_unc;
            part_max[warp] = m;
        }
    }
    __syncthreads();
    int total_unc = 0, n_max = 0, base_unc = 0;
    for (int w = 0; w < C::kThreads / kWarp; ++w) {
        base_unc += w < warp ? part[w] : 0;
        total_unc += part[w];
        n_max = max(n_max, part_max[w]);
    }
    int sched = decode_sched_clusters(p, total_unc);
    bool comb = p.combine;
    bool aligned = false;
    if (p.eff == DecodePlan::kEffCost) {
        bool spread = false;
        const int pp = max(pairs, 1);
        const int h0 = max(
            decode_ceil_div_f(total_unc, 2 * pp),
            (n_max + DecodePlan::kDefaultSmax - 1) / DecodePlan::kDefaultSmax);
        const int h1 = decode_ceil_div_f(total_unc, sched);
        if (p.fuse) {
            const int cu = DecodeCost::kUnsplitBase +
                           DecodeCost::kUnsplitPerTile * n_max +
                           DecodeCost::kUnsplitPerPair * pp;
            const int c0 = DecodeCost::kSplitBase +
                           DecodeCost::kSplitPerTile * h0 +
                           DecodeCost::kSplitPerPair * pp;
            const int c1 = DecodeCost::kCombBase +
                           DecodeCost::kCombPerTile * h1 +
                           DecodeCost::kCombPerPair * pp;
            comb = p.combine && c1 < min(c0, cu);
            aligned = !comb && cu <= c0;
            if (float(p.clusters) * p.ks_rp < DecodeCost::kFewClustersPerPair) {
                const int d0 = DecodeCost::kFewSplitBase +
                               DecodeCost::kFewSplitPerTile * h0 +
                               DecodeCost::kFewSplitPerPair * pp;
                const int d1 = DecodeCost::kFewCombBase +
                               DecodeCost::kFewCombPerTile * h1 +
                               DecodeCost::kFewCombPerPair * pp;
                const int d3 = DecodeCost::kSpreadBase +
                               DecodeCost::kSpreadPerTile * h1 +
                               DecodeCost::kSpreadPerPair * pp +
                               __float2int_rn(DecodeCost::kSpreadPerCluster *
                                              sched * p.ks_rp);
                const int dm = min(min(d0, d1), d3);
                aligned = cu <= dm;
                spread = !aligned && d3 == dm;
                comb = p.combine && !aligned && !spread && d1 < d0;
            }
            const int kmax = min(n_max, p.ks_kmax);
            if (C::kFuse && !aligned && kmax >= DecodeCost::kMinSplit &&
                n_max * pp == total_unc) {
                const float n = float(n_max), b = p.ks_b, fs = sched * p.ks_rp;
                const float hmin =
                    ceilf(__fdividef(n, kmax) - DecodeCost::kCeilEps);
                const float hs = fmaxf(floorf(n * rsqrtf(n * p.ks_a)), hmin);
                float bc =
                    comb ? DecodeCost::kTileCost * h1 + b * fs +
                               (fabsf(fs - rintf(fs)) > DecodeCost::kFracEps
                                    ? DecodeCost::kFracPenalty
                                    : 0.f)
                         : DecodeCost::kTileCost * ((n_max + 1) / 2) + 2.f * b;
                bc -= DecodeCost::kCombBias;
                int best = 0;
                CUTLASS_PRAGMA_UNROLL
                for (int j = 0; j < 2; ++j) {
                    const float h = hs + j;
                    const float k =
                        ceilf(__fdividef(n, h) - DecodeCost::kCeilEps);
                    const float c = DecodeCost::kTileCost * h + b * k;
                    if (k >= DecodeCost::kMinSplit && c < bc) {
                        bc = c;
                        best = int(k);
                    }
                }
                if (best) {
                    comb = true;
                    sched = best * pp;
                }
            }
        } else {
            comb = p.combine && DecodeCost::kPlainTileWeight * (h0 - h1) >
                                    DecodeCost::kPlainPairWeight * pp -
                                        DecodeCost::kPlainPairBias;
        }
        if (!comb && !spread) sched = min((aligned ? 1 : 2) * pp, p.clusters);
    } else if (p.eff == DecodePlan::kEffAligned) {
        const int wide = (total_unc + p.clusters - 1) / p.clusters;
        if (n_max - wide < DecodeCost::kAlignSlack) {
            sched = min(max(pairs, 1), p.clusters);
            aligned = pairs <= p.clusters;
        }
    }
    const int cap = (p.eff == DecodePlan::kEffCost && comb) || aligned
                        ? DecodePlan::kNoCap
                        : decode_effective_cap(p, total_unc, sched);
    if (threadIdx.x == 0) {
        sm.cap = cap;
        sm.sched = sched;
        sm.comb = comb;
        if (p.combine && cluster == 0) *p.eff_g = comb ? sched : 0;
    }

    int sum = sum_unc, incl = incl_unc, base = base_unc, total = total_unc;
    if (cap < n_max) {
        sum = 0;
        for (int q = first; q < min(first + per, pairs); ++q)
            sum +=
                min(decode_pair_geom(p, q, lens_of(q), C::kTile).n_tiles, cap);
        incl = sum;
        CUTLASS_PRAGMA_UNROLL
        for (int d = 1; d < kWarp; d *= 2) {
            const int v = __shfl_up_sync(~0u, incl, d);
            if (lane >= d) incl += v;
        }
        if (lane == kWarp - 1) part2[warp] = incl;
        __syncthreads();
        base = 0;
        total = 0;
        for (int w = 0; w < C::kThreads / kWarp; ++w) {
            base += w < warp ? part2[w] : 0;
            total += part2[w];
        }
    }
    const int lo = decode_range_lo(min(cluster, sched), total, sched);
    const int hi = decode_range_lo(min(cluster + 1, sched), total, sched);
    int pre = base + incl - sum;
    if (aligned) {
        if (first <= cluster && cluster < min(first + per, pairs)) {
            for (int q = first; q < cluster; ++q)
                pre += decode_pair_geom(p, q, lens_of(q), C::kTile).n_tiles;
            sm.start[0] = cluster;
            sm.wstart = pre;
            sm.start[1] = 0;
            sm.start[2] =
                decode_pair_geom(p, cluster, lens_of(cluster), C::kTile)
                    .n_tiles;
        } else if (threadIdx.x == 0 && cluster >= pairs) {
            sm.start[0] = 0;
            sm.start[2] = 0;
        }
    } else if (sum > 0 && pre <= lo && lo < pre + sum) {
        for (int q = first;; ++q) {
            const PairGeom g = decode_pair_geom(p, q, lens_of(q), C::kTile);
            const int w = min(g.n_tiles, cap);
            if (lo < pre + w) {
                sm.start[0] = q;
                sm.wstart = pre;
                if (cap == DecodePlan::kNoCap) {
                    sm.start[1] = lo - pre;
                    sm.start[2] = hi - lo;
                } else {
                    sm.start[1] = decode_w_to_tile(lo - pre, g.n_tiles, w);
                    int tiles =
                        decode_w_to_tile(min(hi - pre, w), g.n_tiles, w) -
                        sm.start[1];
                    int wpos = pre + w;
                    for (int qq = q + 1; wpos < hi; ++qq) {
                        const PairGeom g2 =
                            decode_pair_geom(p, qq, lens_of(qq), C::kTile);
                        const int w2 = min(g2.n_tiles, cap);
                        tiles += decode_w_to_tile(min(hi - wpos, w2),
                                                  g2.n_tiles, w2);
                        wpos += w2;
                    }
                    sm.start[2] = tiles;
                }
                break;
            }
            pre += w;
        }
    }
    if (threadIdx.x == 0 && lo == hi && !aligned) {
        sm.start[0] = 0;
        sm.start[2] = 0;
    }
    if (threadIdx.x == 0) {
        sm.total = total;
        sm.n_mj = 0;
    }
    if (sm.comb && cluster == 0 && threadIdx.x == 0) *p.total_g = total;
    __syncthreads();
    if constexpr (C::kFuse) {
        if (comb && p.fuse && cluster >= sched && pairs > 0) {
            const int q = (cluster - sched) % pairs;
            if (q >= first && q < min(first + per, pairs)) {
                int wq = base + incl - sum;
                for (int qq = first; qq < q; ++qq)
                    wq += min(
                        decode_pair_geom(p, qq, lens_of(qq), C::kTile).n_tiles,
                        cap);
                const PairGeom g = decode_pair_geom(p, q, lens_of(q), C::kTile);
                const int f = pair_cluster_of(sched, total, wq);
                const int l =
                    pair_cluster_of(sched, total, wq + min(g.n_tiles, cap) - 1);
                if (l > f && (cute::block_rank_in_cluster() == 0 || g.has_b)) {
                    sm.mj[0] = make_int4(f, l, g.a, q * 2 + g.has_b);
                    sm.n_mj = 1;
                }
            }
            __syncthreads();
        }
    }
    if (sm.start[2] > 0) {
        if (per <= C::kKeep) {
            CUTLASS_PRAGMA_UNROLL
            for (int k = 0; k < C::kKeep; ++k) {
                const int q = first + k, rel = q - sm.start[0];
                if (k < per && q < pairs && rel >= 0 && rel < C::kTableUnits) {
                    sm.unit_lens[rel] = keep[k];
                }
            }
        } else if (threadIdx.x < C::kTableUnits &&
                   sm.start[0] + int(threadIdx.x) < pairs) {
            const int q = sm.start[0] + threadIdx.x;
            const PairLens l = decode_load_lens(p, q);
            sm.unit_lens[threadIdx.x] =
                make_int4(l.v[0][0], l.v[0][1], l.v[1][0], l.v[1][1]);
        }
    }
    __syncthreads();
}

}  // namespace halops::mla
