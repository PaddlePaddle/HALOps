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

#include <cutlass/arch/barrier.h>

#include <cute/tensor.hpp>

#include "kernel/mla/common/intrinsics.cuh"
#include "kernel/mla/common/kv_runs.cuh"
#include "kernel/mla/common/traits.cuh"

namespace halops::mla {
template <int Span>
CUTE_DEVICE void issue_kv_span(const CUtensorMap* __restrict__ dense,
                               const void* gather_desc,
                               cutlass::arch::ClusterTransactionBarrier& bar,
                               uint8_t* smem_base, int row_bytes,
                               const KvRuns& runs, int g_base, int g_lo) {
    constexpr int64_t kHint = (int64_t)cute::TMA::CacheHintSm90::EVICT_NORMAL;
    constexpr int kTopLevel = AttnConfig::KvDenseLevels - 1;
    const int spp = runs.pt ? (1 << runs.page_shift) : 0;
    const bool try_dense = !runs.pt || spp >= Sm100::Gather4Rows;
    if (!try_dense) {
        constexpr int kQuads = Span / Sm100::Gather4Rows;
        const int quad = KernelBase::lane_idx();
        if (quad < kQuads) {
            const int row_off = g_lo + quad * Sm100::Gather4Rows;
            tma_gather4(gather_desc, bar,
                        smem_base + (size_t)row_bytes * row_off, 0,
                        kv_pt_row4(runs, g_base + row_off), kHint);
        }
        return;
    }
    if (!cute::elect_one_sync()) return;
    int row_off = g_lo;
    const int g_hi = g_lo + Span;
    CUTE_NO_UNROLL
    while (row_off < g_hi) {
        const int rem = g_hi - row_off;
        int found = -1, row = 0;
        {
            CUTE_UNROLL
            for (int k = kTopLevel; k >= 0; --k) {
                const int cand = Sm100::Gather4Rows << k;
                if (found < 0 && cand <= rem &&
                    kv_run_chunk(runs, g_base + row_off, cand, row))
                    found = k;
            }
        }
        uint8_t* dst = smem_base + (size_t)row_bytes * row_off;
        if (found >= 0) {
            tma_load_2d(&dense[found], bar, dst, 0, row, kHint);
            row_off += Sm100::Gather4Rows << found;
        } else {
            tma_gather4(gather_desc, bar, dst, 0,
                        kv_run_row4(runs, g_base + row_off), kHint);
            row_off += Sm100::Gather4Rows;
        }
    }
}

struct KvSegGeom {
    const uint8_t* pool;
    int stride_kv_block;
};

CUTE_DEVICE KvRuns kv_tile_slice(const KvRuns& runs, int g_base, int span) {
    const int valid = min(span, max(runs.total - g_base, 0));
    KvRuns out;
    out.pt = nullptr;
    out.j0 = 0;
    out.rows_per_page = runs.rows_per_page;
    out.page_shift = runs.page_shift;
    out.total = valid;
    if (g_base < runs.len0) {
        out.slot0 = runs.slot0 + g_base;
        out.row0 = runs.row0 + g_base;
        out.len0 = min(runs.len0 - g_base, valid);

        out.slot1 = runs.slot1;
        out.row1 = runs.row1;
    } else {
        const int offset = g_base - runs.len0;
        out.slot0 = runs.slot1 + offset;
        out.row0 = runs.row1 + offset;
        out.len0 = valid;
        out.slot1 = out.slot0;
        out.row1 = out.row0;
    }
    return out;
}

CUTE_DEVICE KvRuns kv_tile_page_table(const int* pt, int total, int g_base,
                                      int span, int page_shift,
                                      int rows_per_page) {
    KvRuns out;
    out.pt = pt;
    out.j0 = g_base;
    out.rows_per_page = rows_per_page;
    out.page_shift = page_shift;
    out.total = min(span, max(total - g_base, 0));
    out.row0 = out.row1 = out.slot0 = out.slot1 = out.len0 = 0;
    return out;
}

CUTE_DEVICE const uint64_t* kv_seg_scale_ptr(const KvRuns& runs,
                                             const KvSegGeom& seg, int row) {
    const int slot = (row < runs.total) ? kv_run_slot(runs, row) : 0;
    const int page = slot >> runs.page_shift;
    const int off = slot & ((1 << runs.page_shift) - 1);
    return (const uint64_t*)(seg.pool + (size_t)page * seg.stride_kv_block +
                             ((size_t)1 << runs.page_shift) *
                                 ModelDims::KvRowBytes +
                             (size_t)off * KvCacheLayout::ScaleSlotBytes);
}
}  // namespace halops::mla
