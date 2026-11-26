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

#include "kernel/mla/pair_mla/prefill/params.cuh"

namespace halops::mla {

// The pair's first token: the last request r with cu_pairs[r] <= pair (empty
// requests are skipped); kQnr with one request needs no search.
template <class C>
CUTE_DEVICE PairMap prefill_pair_map(const PrefillParams& p, int pair) {
    PairMap m;
    if (C::kQnr && p.n_req == 1) {
        m.a = 2 * pair;
        m.has_b = m.a + 1 < p.tokens;
        return m;
    }
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

// cu_pairs is never null; the null check keeps ptxas register allocation of
// the kQnr = false kernels as it is.
CUTE_DEVICE int prefill_num_pairs(const PrefillParams& p) {
    if (!p.cu_pairs) return 0;
    return __ldg(p.cu_pairs + p.n_req);
}

// Raw c128 and SWA list lengths of both tokens of a pair.
template <class C>
CUTE_DEVICE PairLens prefill_load_lens(const PrefillParams& p, int pair) {
    return pair_load_lens(p, prefill_pair_map<C>(p, pair));
}

// The pair's clamped lengths and its key stream extents in rows and tiles.
template <class C>
CUTE_DEVICE PairGeom prefill_pair_geom(const PrefillParams& p, int pair,
                                       const PairLens& l) {
    return pair_make_geom(p, prefill_pair_map<C>(p, pair), l, p.swa_shift,
                          C::kTile);
}

// Cluster c takes stream tiles [bound[c], bound[c + 1]) of an even split.
CUTE_DEVICE int prefill_range_lo(int c, int total, int clusters) {
    return int(uint64_t(c) * uint64_t(total) / uint64_t(clusters));
}

// Schedule cost of an n-tile pair in quarter tiles.
template <class C>
CUTE_DEVICE int prefill_pair_cost(int n) {
    return C::kQst ? PrefillSchedTraits::kTileCost * n +
                         PrefillSchedTraits::kPairFixQst
                   : max(PrefillSchedTraits::kTileCost * n,
                         PrefillSchedTraits::kPairFix + n);
}

// Clusters to schedule; the rest get an empty range and exit.
CUTE_DEVICE int prefill_sched_clusters(const PrefillParams& p, int total) {
    if (!p.combine) return p.clusters;
    const int s = (total + PrefillSchedTraits::kDefaultTpc - 1) /
                  PrefillSchedTraits::kDefaultTpc;
    return max(1, min(s, p.clusters));
}

// Walks a cluster's tile range unit by unit; pair lengths come from the
// shared table for the first kTableUnits pairs of the range.
struct PrefillUnitCursor {
    int pair, off, left, wpair0, first_pair;
    template <class C>
    CUTE_DEVICE bool next(const PrefillParams& p, const int4* table,
                          PrefillUnit& u) {
        if (left == 0) return false;
        PairLens lens;
        const int rel = pair - first_pair;
        if (rel < PrefillSchedTraits::kTableUnits) {
            const int4 v = table[rel];
            lens = {{{v.x, v.y}, {v.z, v.w}}};
        } else {
            lens = prefill_load_lens<C>(p, pair);
        }
        u.g = prefill_pair_geom<C>(p, pair, lens);
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

}  // namespace halops::mla
