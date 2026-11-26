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

#include "kernel/mla/pair_mla_prefill/scheduler.cuh"

namespace halops::mla {

// This cluster's first pair and tile (all threads scan every pair). The
// in-kernel merge splits the stream by pair cost, moving each boundary to a
// short pair's start; combine and all-long streams split the tiles evenly.
template <class C>
CUTE_DEVICE void prefill_find_range(const PrefillParams& p,
                                    typename C::Smem& sm, int cluster) {
    __shared__ int part[C::kThreads / C::kWarp], cpart[C::kThreads / C::kWarp];
    const int pairs = prefill_num_pairs(p);
    const int per = (pairs + C::kThreads - 1) / C::kThreads;
    const int first = threadIdx.x * per;
    const int lane = threadIdx.x % C::kWarp, warp = threadIdx.x / C::kWarp;
    constexpr int kKeep = PrefillSchedTraits::kKeepPairs;
    int4 keep[kKeep];
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < kKeep; ++j) keep[j] = make_int4(0, 0, 0, 0);

    bool by_cost = !p.combine;
    int sum = 0, csum = 0;
    for (int q = first, j = 0; q < min(first + per, pairs); ++q, ++j) {
        const PrefillPairLens l = prefill_load_lens<C>(p, q);
        CUTLASS_PRAGMA_UNROLL
        for (int k = 0; k < kKeep; ++k) {
            if (j == k)
                keep[k] = make_int4(l.v[0][0], l.v[0][1], l.v[1][0], l.v[1][1]);
        }
        const int n = prefill_pair_geom<C>(p, q, l).n_tiles;
        sum += n;
        csum += by_cost ? prefill_pair_cost<C>(n)
                        : PrefillSchedTraits::kTileCost * n;
    }
    {
        int v = sum, cv = csum;
        CUTLASS_PRAGMA_UNROLL
        for (int d = C::kWarp / 2; d > 0; d /= 2) {
            v += __shfl_xor_sync(~0u, v, d);
            cv += __shfl_xor_sync(~0u, cv, d);
        }
        if (lane == 0) {
            part[warp] = v;
            cpart[warp] = cv;
        }
    }
    __syncthreads();
    int tiles = 0, ctiles = 0;
    for (int w = 0; w < C::kThreads / C::kWarp; ++w) {
        tiles += part[w];
        ctiles += cpart[w];
    }
    if (ctiles == PrefillSchedTraits::kTileCost * tiles) by_cost = false;
    __syncthreads();
    const int sched = prefill_sched_clusters(p, tiles);
    if (threadIdx.x == 0) {
        sm.sched = sched;
        if (p.combine && cluster == 0) *p.sched_g = sched;
    }

    int incl = sum;
    CUTLASS_PRAGMA_UNROLL
    for (int d = 1; d < C::kWarp; d *= 2) {
        const int v = __shfl_up_sync(~0u, incl, d);
        if (lane >= d) incl += v;
    }
    if (lane == C::kWarp - 1) part[warp] = incl;
    __syncthreads();
    int base = 0, total = 0;
    for (int w = 0; w < C::kThreads / C::kWarp; ++w) {
        base += w < warp ? part[w] : 0;
        total += part[w];
    }
    int pre = base + incl - sum;
    if (!by_cost) {
        for (int c = threadIdx.x; c <= sched; c += C::kThreads) {
            sm.bound[c] = prefill_range_lo(c, total, sched);
        }
    } else {
        int cincl = csum;
        CUTLASS_PRAGMA_UNROLL
        for (int d = 1; d < C::kWarp; d *= 2) {
            const int v = __shfl_up_sync(~0u, cincl, d);
            if (lane >= d) cincl += v;
        }
        __syncthreads();
        if (lane == C::kWarp - 1) part[warp] = cincl;
        __syncthreads();
        int cbase = 0, ctotal = 0;
        for (int w = 0; w < C::kThreads / C::kWarp; ++w) {
            cbase += w < warp ? part[w] : 0;
            ctotal += part[w];
        }
        const auto x_of = [&](int c) {
            return int(uint64_t(c) * uint64_t(ctotal) / uint64_t(sched));
        };
        if (threadIdx.x == 0) {
            sm.bound[0] = 0;
            sm.bound[sched] = total;
        }
        int cs = cbase + cincl - csum, tp = pre;
        int c = max(int((uint64_t(cs) * sched + ctotal - 1) / ctotal), 1);
        if (csum > 0 && c < sched && x_of(c) < cs + csum) {
            for (int q = first; q < min(first + per, pairs) && c < sched; ++q) {
                const int n =
                    prefill_pair_geom<C>(p, q, prefill_load_lens<C>(p, q))
                        .n_tiles;
                const int ce = cs + prefill_pair_cost<C>(n);
                const bool lng = ce - cs == PrefillSchedTraits::kTileCost * n;
                for (; c < sched && x_of(c) < ce; ++c) {
                    sm.bound[c] = tp + (lng ? (x_of(c) - cs) /
                                                  PrefillSchedTraits::kTileCost
                                            : 0);
                }
                cs = ce;
                tp += n;
            }
        }
    }
    __syncthreads();
    const int lo = sm.bound[min(cluster, sched)];
    const int hi = sm.bound[min(cluster + 1, sched)];
    const int empty = int(threadIdx.x) < sched &&
                      sm.bound[threadIdx.x + 1] <= sm.bound[threadIdx.x];
    if (sum > 0 && pre <= lo && lo < pre + sum) {
        for (int q = first;; ++q) {
            const int w =
                prefill_pair_geom<C>(p, q, prefill_load_lens<C>(p, q)).n_tiles;
            if (lo < pre + w) {
                sm.start[0] = q;
                sm.wstart = pre;
                sm.start[1] = lo - pre;
                sm.start[2] = hi - lo;
                break;
            }
            pre += w;
        }
    }
    if (threadIdx.x == 0 && lo == hi) {
        sm.start[0] = 0;
        sm.start[2] = 0;
    }
    const int any_empty = __syncthreads_or(empty);
    if (threadIdx.x == 0) sm.dense = !any_empty;
    if (p.combine && threadIdx.x == 0) {
        if (cluster == 0) *p.total_g = total;
        if (blockIdx.x % 2 == 0) p.jobs[cluster].wn = 0;
    }
    __syncthreads();
    if (sm.start[2] > 0) {
        if (per <= kKeep) {
            CUTLASS_PRAGMA_UNROLL
            for (int k = 0; k < kKeep; ++k) {
                const int q = first + k, rel = q - sm.start[0];
                if (k < per && q < pairs && rel >= 0 &&
                    rel < PrefillSchedTraits::kTableUnits) {
                    sm.unit_lens[rel] = keep[k];
                }
            }
        } else if (threadIdx.x < PrefillSchedTraits::kTableUnits &&
                   sm.start[0] + int(threadIdx.x) < pairs) {
            const int q = sm.start[0] + threadIdx.x;
            const PrefillPairLens l = prefill_load_lens<C>(p, q);
            sm.unit_lens[threadIdx.x] =
                make_int4(l.v[0][0], l.v[0][1], l.v[1][0], l.v[1][1]);
        }
    }
    __syncthreads();
}

// A cursor at the start of this cluster's range (after prefill_find_range).
template <class C>
CUTE_DEVICE PrefillUnitCursor prefill_start_cursor(const PrefillParams& p,
                                                   const typename C::Smem& sm) {
    return {sm.start[0], sm.start[1], sm.start[2], sm.wstart, sm.start[0]};
}

}  // namespace halops::mla
