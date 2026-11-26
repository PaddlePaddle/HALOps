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

CUTE_DEVICE PairMap pair_map(const HcaParams& p, int pair) {
    const int ppr = (p.tokens_per_req + 1) / 2;
    const int r = pair / ppr, k = pair - r * ppr;
    PairMap m;
    m.a = r * p.tokens_per_req + 2 * k;
    m.has_b = 2 * k + 1 < p.tokens_per_req;
    return m;
}

CUTE_DEVICE PairLens load_lens(const HcaParams& p, int pair) {
    return pair_load_lens(p, pair_map(p, pair));
}

CUTE_DEVICE PairGeom pair_geom(const HcaParams& p, int pair, const PairLens& l,
                               int tile) {
    return pair_make_geom(p, pair_map(p, pair), l, 1, tile);
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

}  // namespace halops::mla
