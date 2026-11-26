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
#include "kernel/mla/pair_mla_decode/params.cuh"

namespace halops::detail {

// The key stream of tokens a = 2j and b = 2j+1: position u of the c128
// segment is b's row u (a's list is a prefix of it) or a's row u where b has
// none, then the SWA window of b followed by the oldest row of a (a's window
// is b's shifted by one).  A row counts for a token only if its own list has
// it at that place, so inputs without this structure are masked instead of
// read out of bounds.
struct PairGeom {
    int a, len[2][2], n_c4, n_s, n_tiles;
    bool has_b;
};

// The lengths of a pair's tokens, [seg][token].
struct PairLens {
    int v[2][2];
};

// A pair's first token and whether it has a second.  Tokens are paired within
// a request (t tokens each, laid out [r*t, (r+1)*t)): a pair never straddles a
// request, and an odd request leaves its last token unpaired ("lone").  CTA 1
// of a lone pair has no token of its own.
struct PairMap {
    int a;
    bool has_b;
};

CUTE_DEVICE PairMap pair_map(const HcaParams& p, int pair) {
    const int ppr = (p.tokens_per_req + 1) / 2;
    const int r = pair / ppr, k = pair - r * ppr;
    PairMap m;
    m.a = r * p.tokens_per_req + 2 * k;
    m.has_b = 2 * k + 1 < p.tokens_per_req;
    return m;
}

CUTE_DEVICE PairLens load_lens(const HcaParams& p, int pair) {
    const PairMap m = pair_map(p, pair);
    PairLens l;
    CUTLASS_PRAGMA_UNROLL
    for (int seg = 0; seg < 2; ++seg) {
        l.v[seg][0] = __ldg(p.len[seg] + m.a);
        l.v[seg][1] = m.has_b ? __ldg(p.len[seg] + m.a + 1) : 0;
    }
    return l;
}

CUTE_DEVICE PairGeom pair_geom(const HcaParams& p, int pair, const PairLens& l,
                               int tile) {
    PairGeom g;
    const PairMap m = pair_map(p, pair);
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
    g.n_c4 = (n_c + kTmaGatherRows - 1) / kTmaGatherRows * kTmaGatherRows;
    g.n_s = max(g.len[1][1], g.len[1][0] > 0 ? g.len[1][0] + 1 : 0);
    g.n_tiles = max((g.n_c4 + g.n_s + tile - 1) / tile, 1);
    return g;
}

struct Unit {
    PairGeom g;
    int off, n;
    int w0, wn;
    int pair;
    bool split;
};

template <class C>
struct UnitCursor {
    int pair, off, left, wpair0, cluster, first_pair, cap, tile;
    CUTE_DEVICE bool next(const HcaParams& p, const int4* table, Unit& u) {
        if (left == 0) return false;
        PairLens lens;
        const int rel = pair - first_pair;
        if (rel < C::kTableUnits) {
            const int4 v = table[rel];
            lens = {{{v.x, v.y}, {v.z, v.w}}};
        } else {
            lens = load_lens(p, pair);
        }
        u.g = pair_geom(p, pair, lens, tile);
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

struct KeyRaw {
    bool in_a, in_b;
    int seg;
};

template <class C>
CUTE_DEVICE int64_t part_row(int slot, int cta, int head) {
    return (int64_t(slot) * 2 + cta) * C::kHeads + head;
}

template <class C>
CUTE_DEVICE const float* part_box(const HcaParams& p, int slot, int cta,
                                  int bx) {
    return p.o_part +
           (int64_t(slot * 2 + cta) * C::kPlanes + bx * C::kPartBoxPlanes) *
               (C::kPlaneBytes / sizeof(float));
}

}  // namespace halops::detail
