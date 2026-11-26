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

#include <cuda_runtime.h>
#include <cutlass/cutlass.h>

#include <cstdint>
#include <cute/config.hpp>

#include "common/sm100_arch.cuh"

namespace halops::mla {

// Key stream of pair (a, b); len is [seg][token], has_b false for a lone token.
struct PairGeom {
    int a, len[2][2], n_c4, n_s, n_tiles;
    bool has_b;
};

struct PairLens {
    int v[2][2];
};

struct PairMap {
    int a;
    bool has_b;
};

// A split pair for the combine kernel (wn 0: no job).
struct PairMergeJob {
    int a, has_b, w0, wn;
};

template <class P>
CUTE_DEVICE PairLens pair_load_lens(const P& p, const PairMap& m) {
    PairLens l;
    CUTLASS_PRAGMA_UNROLL
    for (int seg = 0; seg < 2; ++seg) {
        l.v[seg][0] = __ldg(p.len[seg] + m.a);
        l.v[seg][1] = m.has_b ? __ldg(p.len[seg] + m.a + 1) : 0;
    }
    return l;
}

// Clamped lengths and key stream extents; a's SWA window is b's + swa_shift.
template <class P>
CUTE_DEVICE PairGeom pair_make_geom(const P& p, const PairMap& m,
                                    const PairLens& l, int swa_shift,
                                    int tile) {
    PairGeom g;
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
    g.n_s = max(g.len[1][1], g.len[1][0] > 0 ? g.len[1][0] + swa_shift : 0);
    g.n_tiles = max((g.n_c4 + g.n_s + tile - 1) / tile, 1);
    return g;
}

// Cluster of stream tile pos: last c with c * total < (pos + 1) * clusters.
CUTE_DEVICE int pair_cluster_of(int clusters, int total, int pos) {
    return int((uint64_t(pos + 1) * clusters - 1) / uint64_t(total));
}

// Row of a part's log2 sum in lse_part.
template <class C>
CUTE_DEVICE int64_t pair_part_row(int slot, int cta, int head) {
    return (int64_t(slot) * 2 + cta) * C::kHeads + head;
}

// Box bx of the normalized partial O of slot and CTA cta.
template <class C, class P>
CUTE_DEVICE const float* pair_part_box(const P& p, int slot, int cta, int bx) {
    return p.o_part +
           (int64_t(slot * 2 + cta) * C::kPlanes + bx * C::kPartBoxPlanes) *
               (C::kPlaneBytes / sizeof(float));
}

}  // namespace halops::mla
