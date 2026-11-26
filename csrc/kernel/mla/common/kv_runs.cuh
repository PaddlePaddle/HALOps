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

#include <cute/tensor.hpp>

#include "kernel/mla/common/traits.cuh"

namespace halops::mla {
struct KvRuns {
    int row0, row1;
    int slot0, slot1;
    int len0;
    int total;
    int page_shift;
    const int* pt;
    int j0;
    int rows_per_page;
};

CUTE_DEVICE bool kv_run_breaks(int prev, int cur, int step, int page_shift) {
    return (int64_t)cur != (int64_t)prev + step ||
           (cur >> page_shift) != (prev >> page_shift);
}

CUTE_DEVICE int4 ascending_kv_runs(int first0, int len0, int first1, int rows,
                                   int step) {
    if (step > 0) return make_int4(first0, len0, first1, rows);
    const int len1 = rows - len0;
    if (len1 > 0)
        return make_int4(first1 - (len1 - 1), len1, first0 - (len0 - 1), rows);
    const int slot0 = first0 - (rows - 1);
    return make_int4(slot0, rows, slot0, rows);
}

CUTE_DEVICE KvRuns make_kv_runs(int4 ascending, int total, int page_shift,
                                int rows_per_page) {
    KvRuns runs;
    runs.pt = nullptr;
    runs.j0 = 0;
    runs.rows_per_page = rows_per_page;
    runs.total = total;
    runs.page_shift = page_shift;
    runs.slot0 = ascending.x;
    runs.len0 = ascending.y;
    runs.slot1 = ascending.z;
    const int mask = (1 << page_shift) - 1;
    runs.row0 =
        (runs.slot0 >> page_shift) * rows_per_page + (runs.slot0 & mask);
    runs.row1 =
        (runs.slot1 >> page_shift) * rows_per_page + (runs.slot1 & mask);
    return runs;
}

CUTE_DEVICE int kv_run_page_id(const KvRuns& runs, int j) {
    return __ldg(runs.pt + (j >> runs.page_shift));
}

CUTE_DEVICE int kv_run_slot(const KvRuns& runs, int row) {
    if (runs.pt) {
        const int j = runs.j0 + row;
        return (kv_run_page_id(runs, j) << runs.page_shift) +
               (j & ((1 << runs.page_shift) - 1));
    }
    return (row < runs.len0) ? runs.slot0 + row
                             : runs.slot1 + (row - runs.len0);
}

CUTE_DEVICE int kv_run_row(const KvRuns& runs, int row) {
    if (row >= runs.total) return 0;
    if (runs.pt) {
        const int j = runs.j0 + row;
        return kv_run_page_id(runs, j) * runs.rows_per_page +
               (j & ((1 << runs.page_shift) - 1));
    }
    return (row < runs.len0) ? runs.row0 + row : runs.row1 + (row - runs.len0);
}

CUTE_DEVICE int4 kv_run_row4(const KvRuns& runs, int g0) {
    constexpr int kLast = Sm100::Gather4Rows - 1;
    if (runs.pt)
        return make_int4(kv_run_row(runs, g0), kv_run_row(runs, g0 + 1),
                         kv_run_row(runs, g0 + 2),
                         kv_run_row(runs, g0 + kLast));
    const int lim = min(runs.len0, runs.total);
    int base = -1;
    if (g0 + kLast < lim)
        base = runs.row0 + g0;
    else if (g0 >= runs.len0 && g0 + kLast < runs.total)
        base = runs.row1 + (g0 - runs.len0);
    if (base >= 0) return make_int4(base, base + 1, base + 2, base + kLast);
    return make_int4(kv_run_row(runs, g0), kv_run_row(runs, g0 + 1),
                     kv_run_row(runs, g0 + 2), kv_run_row(runs, g0 + kLast));
}

CUTE_DEVICE bool kv_run_chunk(const KvRuns& runs, int g0, int count, int& row) {
    if (runs.pt) {
        const int spp = 1 << runs.page_shift;
        if (count > spp - ((runs.j0 + g0) & (spp - 1)) ||
            g0 + count > runs.total)
            return false;
        row = kv_run_row(runs, g0);
        return true;
    }
    const int lim = min(runs.len0, runs.total);
    if (g0 + count <= lim) {
        row = runs.row0 + g0;
        return true;
    }
    if (g0 >= runs.len0 && g0 + count <= runs.total) {
        row = runs.row1 + (g0 - runs.len0);
        return true;
    }
    return false;
}

CUTE_DEVICE int4 kv_pt_row4(const KvRuns& runs, int g0) {
    constexpr int kRows = Sm100::Gather4Rows;
    const int j = runs.j0 + g0;
    const int mask = (1 << runs.page_shift) - 1;
    const int i_max = max((runs.j0 + runs.total - 1) >> runs.page_shift, 0);
    const int i_lo = min(j >> runs.page_shift, i_max);
    const int i_hi = min((j + (kRows - 1)) >> runs.page_shift, i_max);
    const int base_lo = __ldg(runs.pt + i_lo) * runs.rows_per_page;
    const int base_hi = __ldg(runs.pt + i_hi) * runs.rows_per_page;
    int rows[kRows];
    CUTE_UNROLL
    for (int i = 0; i < kRows; ++i) {
        const int idx = j + i;
        const int base = ((idx >> runs.page_shift) == i_lo) ? base_lo : base_hi;
        rows[i] = (g0 + i < runs.total) ? base + (idx & mask) : 0;
    }
    return make_int4(rows[0], rows[1], rows[2], rows[kRows - 1]);
}
}  // namespace halops::mla
