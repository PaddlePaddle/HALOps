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

#include "common/sm100_arch.cuh"
#include "kernel/mla/pair_mla/decode/params.cuh"

namespace halops::mla {

CUTE_DEVICE PairMap decode_pair_map(const DecodeParams& p, int pair) {
    const int ppr = (p.tokens_per_req + 1) / 2;
    const int r = pair / ppr, k = pair - r * ppr;
    PairMap m;
    m.a = r * p.tokens_per_req + 2 * k;
    m.has_b = 2 * k + 1 < p.tokens_per_req;
    return m;
}

CUTE_DEVICE PairLens decode_load_lens(const DecodeParams& p, int pair) {
    return pair_load_lens(p, decode_pair_map(p, pair));
}

CUTE_DEVICE PairGeom decode_pair_geom(const DecodeParams& p, int pair,
                                      const PairLens& l, int tile) {
    return pair_make_geom(p, decode_pair_map(p, pair), l, 1, tile);
}

struct DecodeUnit {
    PairGeom g;
    int off, n;
    int w0, wn;
    int pair;
    bool split;
};

template <class C>
struct DecodeUnitCursor {
    int pair, off, left, wpair0, cluster, first_pair, cap, tile;
    CUTE_DEVICE bool next(const DecodeParams& p, const int4* table,
                          DecodeUnit& u) {
        if (left == 0) return false;
        PairLens lens;
        const int rel = pair - first_pair;
        if (rel < C::kTableUnits) {
            const int4 v = table[rel];
            lens = {{{v.x, v.y}, {v.z, v.w}}};
        } else {
            lens = decode_load_lens(p, pair);
        }
        u.g = decode_pair_geom(p, pair, lens, tile);
        u.pair = pair;
        const int wn = min(u.g.n_tiles, cap);
        u.off = off;
        u.n = min(u.g.n_tiles - off, left);
        u.w0 = wpair0;
        u.wn = wn;
        u.split = u.n != u.g.n_tiles;
        left -= u.n;
        off += u.n;
        if (off == u.g.n_tiles) {
            ++pair;
            off = 0;
            wpair0 += wn;
        }
        return true;
    }
};

struct DecodeKeyRaw {
    bool in_a, in_b;
    int seg;
};

// Cluster c takes stream tiles [c * total / C, (c + 1) * total / C).
CUTE_DEVICE int decode_range_lo(int c, int total, int clusters) {
    return int(uint32_t(c) * uint32_t(total) / uint32_t(clusters));
}

// Real tile of weight offset wofs of a pair with weight w = min(n_tiles, cap).
CUTE_DEVICE int decode_w_to_tile(int wofs, int n_tiles, int w) {
    return int(int64_t(wofs) * n_tiles / w);
}

// Per-pair weight cap: p.cap if set, else smax * ceil(total / clusters).
CUTE_DEVICE int decode_effective_cap(const DecodeParams& p, int total_unc,
                                     int clusters) {
    if (p.cap != DecodePlan::kNoCap) return p.cap;
    if (p.smax <= 0) return DecodePlan::kNoCap;
    const int quota = (total_unc + clusters - 1) / clusters;
    return max(p.smax * quota, DecodePlan::kCapFloor);
}

// ceil(x / d) for 0 <= x < 2^20, d > 0, without integer division.
CUTE_DEVICE int decode_ceil_div_f(int x, int d) {
    int q = __float2int_rz(__fdividef(float(x), float(d)));
    q += (q + 1) * d <= x;
    q -= q * d > x;
    return q + (q * d < x);
}

// Clusters to schedule (about p.tpc tiles each); the rest exit.
CUTE_DEVICE int decode_sched_clusters(const DecodeParams& p, int total_unc) {
    if (!p.eff || p.eff == DecodePlan::kEffAligned) return p.clusters;
    const int tpc = p.tpc > 0 ? p.tpc : DecodePlan::kDefaultTpc;
    int s = decode_ceil_div_f(total_unc, tpc);
    return max(1, min(s, p.clusters));
}

// Cursor at the start of this cluster's range found by decode_find_range.
template <class C>
CUTE_DEVICE DecodeUnitCursor<C> decode_start_cursor(const DecodeParams& p,
                                                    const typename C::Smem& sm,
                                                    int cluster) {
    return {sm.start[0], sm.start[1], sm.start[2], sm.wstart,
            cluster,     sm.start[0], sm.cap,      C::kTile};
}

}  // namespace halops::mla
