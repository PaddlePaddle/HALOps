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

#include "kernel/mla/prefill_pair_mla_params.cuh"

namespace halops::mla::prefill_pair {

// Key stream of pair (a, b): the c128 rows (a's list is a prefix of b's),
// then b's SWA window and the oldest swa_shift rows of a, masked per token.
struct PairGeom {
    int a, len[2][2], n_c4, n_s, n_tiles;  // len[seg][token of pair]
    bool has_b;  // false for a lone (odd-request) token
};

struct PairLens {
    int v[2][2];
};

// A pair's first token; a lone pair (odd request's last token) has no b.
struct PairMap {
    int a;
    bool has_b;
};

template <class C>
CUTE_DEVICE PairMap pair_map(const PairMlaParams& p, int pair) {
    PairMap m;
    // One request (cu_seqlens = [0, T]): no search. kQnr only.
    if (C::kQnr && p.n_req == 1) {
        m.a = 2 * pair;
        m.has_b = m.a + 1 < p.tokens;
        return m;
    }
    // Last request r with cu_pairs[r] <= pair (empty requests are skipped).
    int lo = 0, hi = p.n_req;
    while (hi - lo > 1) {
        const int mid = (lo + hi) / 2;
        if (__ldg(p.cu_pairs + mid) <= pair)
            lo = mid;
        else
            hi = mid;
    }
    m.a = __ldg(p.cu_seqlens + lo) + 2 * (pair - __ldg(p.cu_pairs + lo));
    m.has_b = m.a + 1 < __ldg(p.cu_seqlens + lo + 1);
    return m;
}

CUTE_DEVICE int num_pairs(const PairMlaParams& p) {
    // cu_pairs is never null; without this check ptxas spills more in the
    // kQnr = false kernels (52 B spill stores instead of 4).
    if (!p.cu_pairs) return 0;
    return __ldg(p.cu_pairs + p.n_req);
}

template <class C>
CUTE_DEVICE PairLens load_lens(const PairMlaParams& p, int pair) {
    const PairMap m = pair_map<C>(p, pair);
    PairLens l;
    CUTLASS_PRAGMA_UNROLL
    for (int seg = 0; seg < 2; ++seg) {
        l.v[seg][0] = __ldg(p.len[seg] + m.a);
        l.v[seg][1] = m.has_b ? __ldg(p.len[seg] + m.a + 1) : 0;
    }
    return l;
}

template <class C>
CUTE_DEVICE PairGeom pair_geom(const PairMlaParams& p, int pair,
                               const PairLens& l) {
    PairGeom g;
    const PairMap m = pair_map<C>(p, pair);
    g.a = m.a;
    g.has_b = m.has_b;
    CUTLASS_PRAGMA_UNROLL
    for (int seg = 0; seg < 2; ++seg) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < 2; ++i) {
            g.len[seg][i] = max(min(l.v[seg][i], p.ld[seg]), 0);
        }
    }
    const int n_c = max(g.len[0][0], g.len[0][1]);
    g.n_c4 = (n_c + C::kGather - 1) / C::kGather * C::kGather;
    g.n_s = max(g.len[1][1], g.len[1][0] > 0 ? g.len[1][0] + p.swa_shift : 0);
    g.n_tiles = max((g.n_c4 + g.n_s + C::kTile - 1) / C::kTile, 1);
    return g;
}

// Cluster c takes stream tiles [bound[c], bound[c + 1]) (find_range).
CUTE_DEVICE int range_lo(int c, int total, int clusters) {
    return int(uint64_t(c) * uint64_t(total) / uint64_t(clusters));
}

template <class C>
CUTE_DEVICE int pair_cost(int n) {
    return C::kQst
               ? PairSchedTraits::kTileCost * n + PairSchedTraits::kPairFixQst
               : max(PairSchedTraits::kTileCost * n,
                     PairSchedTraits::kPairFix + n);
}

// Clusters to schedule; the rest get an empty range and exit.
CUTE_DEVICE int sched_clusters(const PairMlaParams& p, int total) {
    if (!p.combine) return p.clusters;
    const int s = (total + PairSchedTraits::kDefaultTpc - 1) /
                  PairSchedTraits::kDefaultTpc;
    return max(1, min(s, p.clusters));
}

struct Unit {
    PairGeom g;
    int off, n;  // tiles [off, off + n) of the pair
    int w0, wn;  // the pair's first stream tile and its tile count
    bool split;
};

struct UnitCursor {
    int pair, off, left, wpair0, first_pair;
    template <class C>
    CUTE_DEVICE bool next(const PairMlaParams& p, const int4* table, Unit& u) {
        if (left == 0) return false;
        PairLens lens;
        const int rel = pair - first_pair;
        if (rel < PairSchedTraits::kTableUnits) {
            const int4 v = table[rel];
            lens = {{{v.x, v.y}, {v.z, v.w}}};
        } else {
            lens = load_lens<C>(p, pair);
        }
        u.g = pair_geom<C>(p, pair, lens);
        u.off = off;
        u.n = min(u.g.n_tiles - off, left);
        u.w0 = wpair0;
        u.wn = u.g.n_tiles;
        u.split = u.n != u.g.n_tiles;
        left -= u.n;
        off += u.n;
        if (off == u.g.n_tiles) {
            ++pair;
            off = 0;
            wpair0 += u.g.n_tiles;
        }
        return true;
    }
};

// This cluster's first pair and tile (all threads scan every pair).
template <class C>
CUTE_DEVICE void find_range(const PairMlaParams& p, typename C::Smem& sm,
                            int cluster) {
    __shared__ int part[C::kThreads / C::kWarp], cpart[C::kThreads / C::kWarp];
    const int pairs = num_pairs(p);
    const int per = (pairs + C::kThreads - 1) / C::kThreads;
    const int first = threadIdx.x * per;
    const int lane = threadIdx.x % C::kWarp, warp = threadIdx.x / C::kWarp;
    constexpr int kKeep = 2;
    int4 keep[kKeep];
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < kKeep; ++j) keep[j] = make_int4(0, 0, 0, 0);

    // pair_cost is for the in-kernel merge only; combine uses the even split.
    bool by_cost = !p.combine;
    // Pass 1: the tile total and the cost of this thread's pairs.
    int sum = 0, csum = 0;
    for (int q = first, j = 0; q < min(first + per, pairs); ++q, ++j) {
        const PairLens l = load_lens<C>(p, q);
        CUTLASS_PRAGMA_UNROLL
        for (int k = 0; k < kKeep; ++k) {
            if (j == k)
                keep[k] = make_int4(l.v[0][0], l.v[0][1], l.v[1][0], l.v[1][1]);
        }
        const int n = pair_geom<C>(p, q, l).n_tiles;
        sum += n;
        csum += by_cost ? pair_cost<C>(n) : PairSchedTraits::kTileCost * n;
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
    // Every pair long (long contexts): the cost is the tiles.
    if (ctiles == PairSchedTraits::kTileCost * tiles) by_cost = false;
    __syncthreads();
    const int sched = sched_clusters(p, tiles);
    if (threadIdx.x == 0) {
        sm.sched = sched;
        if (p.combine && cluster == 0) *p.sched_g = sched;
    }

    // Pass 2: scan this thread's pair tiles to their stream positions.
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
    // Boundary c at cost c * cost_total / sched, moved to a short pair's start.
    if (!by_cost) {
        for (int c = threadIdx.x; c <= sched; c += C::kThreads) {
            sm.bound[c] = range_lo(c, total, sched);
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
                const int n = pair_geom<C>(p, q, load_lens<C>(p, q)).n_tiles;
                const int ce = cs + pair_cost<C>(n);
                const bool lng = ce - cs == PairSchedTraits::kTileCost * n;
                for (; c < sched && x_of(c) < ce; ++c) {
                    sm.bound[c] =
                        tp +
                        (lng ? (x_of(c) - cs) / PairSchedTraits::kTileCost : 0);
                }
                cs = ce;
                tp += n;
            }
        }
    }
    __syncthreads();
    // Clusters past sched get an empty range (lo == hi == total) and exit.
    const int lo = sm.bound[min(cluster, sched)];
    const int hi = sm.bound[min(cluster + 1, sched)];
    const int empty = int(threadIdx.x) < sched &&
                      sm.bound[threadIdx.x + 1] <= sm.bound[threadIdx.x];
    if (sum > 0 && pre <= lo && lo < pre + sum) {
        for (int q = first;; ++q) {
            const int w = pair_geom<C>(p, q, load_lens<C>(p, q)).n_tiles;
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
        // No job unless store_o records one (CTA 0, after the barrier).
        if (blockIdx.x % 2 == 0) p.jobs[cluster].wn = 0;
    }
    __syncthreads();
    if (sm.start[2] > 0) {
        if (per <= kKeep) {
            CUTLASS_PRAGMA_UNROLL
            for (int k = 0; k < kKeep; ++k) {
                const int q = first + k, rel = q - sm.start[0];
                if (k < per && q < pairs && rel >= 0 &&
                    rel < PairSchedTraits::kTableUnits) {
                    sm.unit_lens[rel] = keep[k];
                }
            }
        } else if (threadIdx.x < PairSchedTraits::kTableUnits &&
                   sm.start[0] + int(threadIdx.x) < pairs) {
            const int q = sm.start[0] + threadIdx.x;
            const PairLens l = load_lens<C>(p, q);
            sm.unit_lens[threadIdx.x] =
                make_int4(l.v[0][0], l.v[0][1], l.v[1][0], l.v[1][1]);
        }
    }
    __syncthreads();
}

template <class C>
CUTE_DEVICE UnitCursor start_cursor(const PairMlaParams& p,
                                    const typename C::Smem& sm) {
    return {sm.start[0], sm.start[1], sm.start[2], sm.wstart, sm.start[0]};
}

}  // namespace halops::mla::prefill_pair
