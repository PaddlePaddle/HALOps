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

#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/barrier.h>
#include <math_constants.h>
#include <torch/extension.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/tensor.hpp>

#include "kernel/gemm/gemm_utils.h"
#include "kernel/gemm/kernel_traits.cuh"
#include "kernel/gemm/mainloop_fwd.cuh"
#include "mla_intrinsics.cuh"
#include "mla_traits.cuh"
#include "umma_gemm.cuh"

namespace mega_mla {

// ===== TRACE, DETACHABLE =====================================================
// The whole feature is this struct plus the trailing site/item/k arguments on
// the wait helpers below.  Compiled out unless MEGA_MLA_TRACE is defined.
struct TraceProbe {
    enum Site : int {
        WS_NONE = 0,
        WS_QKV_TMA_W = 1,
        WS_QKV_TMA_ACT = 2,

        WS_QKV_BACKPRESSURE_W = 4,
        WS_QKV_BACKPRESSURE_ACT = 5,

        WS_WQB_TMA_W = 11,
        WS_WQB_TMA_ACT = 12,

        WS_WQB_BACKPRESSURE_W = 14,
        WS_WQB_BACKPRESSURE_ACT = 15,

        WS_QKV_ACC_FULL = 20,
        WS_QKV_ACC_EMPTY = 21,
        WS_WQB_ACC_FULL = 22,
        WS_WQB_ACC_EMPTY = 23,
        WS_WG2_PUBLISH = 24,
        WS_WG2_REUSE = 25,
        WS_PBAR_X = 26,
        WS_PBAR_QLORA = 27,
        WS_PBAR_QQ = 28,

        WS_KVG_TMA_W = 32,
        WS_KVG_TMA_ACT = 33,
        WS_KVG_BACKPRESSURE_W = 34,
        WS_KVG_BACKPRESSURE_ACT = 35,
        WS_KVG_ACC_FULL = 36,
        WS_KVG_ACC_EMPTY = 37,
        WS_PBAR_RING = 38,
        WS_PBAR_KV = 39,

        WS_WOA_TMA_W = 40,
        WS_WOA_TMA_ACT = 41,
        WS_WOA_BACKPRESSURE_W = 42,
        WS_WOA_BACKPRESSURE_ACT = 43,
        WS_WOA_ACC_FULL = 44,
        WS_WOA_ACC_EMPTY = 45,
        WS_WOB_TMA_W = 46,
        WS_WOB_TMA_ACT = 47,
        WS_WOB_BACKPRESSURE_W = 48,
        WS_WOB_BACKPRESSURE_ACT = 49,
        WS_WOB_ACC_FULL = 50,
        WS_WOB_ACC_EMPTY = 51,
        WS_PBAR_OLORA = 52,
        WS_PBAR_OQ = 53,

        WS_PBAR_WOB = 55,
        WS_PBAR_FLAT_O = 56,
        WS_PBAR_FLATQ = 57,
        WS_PBAR_O_STORED = 58,
        WS_PBAR_FLAT_Q = 59,
        WS_PBAR_KVCACHE = 60,
        WS_PBAR_C128 = 61,
        WS_PBAR_O_COEF = 62,
    };

#ifdef MEGA_MLA_TRACE
    static CUTE_DEVICE volatile int* slot() {
        const int idx = (int)blockIdx.x * KernelTrait::Trace::WarpsPerCta +
                        KernelTrait::warp_idx();
        return (volatile int*)mega_mla_trace +
               (int64_t)idx * KernelTrait::Trace::WordsPerWarp;
    }

    static CUTE_DEVICE void store(int site, int passed, int item, int k) {
        volatile int* s = slot();
        s[1] = item;
        s[2] = k;
        s[0] = site * 2 + passed;
        const bool acc_edge = (site >= TraceProbe::WS_QKV_ACC_FULL &&
                               site <= TraceProbe::WS_WQB_ACC_EMPTY) ||
                              site == TraceProbe::WS_KVG_ACC_FULL ||
                              site == TraceProbe::WS_KVG_ACC_EMPTY ||
                              site == TraceProbe::WS_WOA_ACC_FULL ||
                              site == TraceProbe::WS_WOA_ACC_EMPTY ||
                              site == TraceProbe::WS_WOB_ACC_FULL ||
                              site == TraceProbe::WS_WOB_ACC_EMPTY;
        if (passed && acc_edge) s[3] = s[3] + 1;
    }

    static CUTE_DEVICE void warp(int site, int passed, int item, int k) {
        __syncwarp();
        if (cute::elect_one_sync()) store(site, passed, item, k);
    }
    static CUTE_DEVICE void thread(int site, int passed, int item, int k) {
        store(site, passed, item, k);
    }
    static CUTE_DEVICE void warp_if(int site, int passed, int item) {
        if (site != TraceProbe::WS_NONE) warp(site, passed, item, -1);
    }
#else
    static CUTE_DEVICE void warp(int, int, int, int) {}
    static CUTE_DEVICE void thread(int, int, int, int) {}
    static CUTE_DEVICE void warp_if(int, int, int) {}
#endif
};

#ifdef MEGA_MLA_TRACE
__device__ int mega_mla_trace[KernelTrait::Trace::CtaSlots *
                              KernelTrait::Trace::WarpsPerCta *
                              KernelTrait::Trace::WordsPerWarp];
#endif
// ===== end TRACE ============================================================

template <class Bar>
CUTE_DEVICE void mbar_wait(Bar& bar, uint32_t phase,
                           [[maybe_unused]] int site = TraceProbe::WS_NONE,
                           [[maybe_unused]] int item = -1,
                           [[maybe_unused]] int k = -1) {
    TraceProbe::warp(site, 0, item, k);
    bar.wait(phase);
    TraceProbe::warp(site, 1, item, k);
}

template <class Bar>
CUTE_DEVICE void bar_sync([[maybe_unused]] int site = TraceProbe::WS_NONE,
                          [[maybe_unused]] int item = -1) {
    TraceProbe::warp_if(site, 0, item);
    cutlass::arch::NamedBarrier::arrive_and_wait(Bar::width, Bar::id);
    TraceProbe::warp_if(site, 1, item);
}

CUTE_DEVICE void bar_sync_warp_pair(int local_warp) {
    using A = KernelTrait::Bars::Wg1Warp02Sync;

    cutlass::arch::NamedBarrier::arrive_and_wait(
        A::width, A::id + (uint32_t)(local_warp & 1));
}

CUTE_DEVICE int n_token_groups(int num_tokens, int n_pad) {
    return (num_tokens + n_pad - 1) / n_pad;
}

CUTE_DEVICE int n_items(int count) {
    const int b = (int)blockIdx.x;
    return count > b ? (count - b + (int)gridDim.x - 1) / (int)gridDim.x : 0;
}

struct GemmPipelineBars {
    transac_bar_t* tma_a;
    transac_bar_t* tma_b;
    transac_bar_t* smem_free;
    transac_bar_t* acc_full;
    transac_bar_t* acc_empty;
};

template <int STAGES>
CUTE_DEVICE int stage_uses(int k_tps, int s) {
    return k_tps / STAGES + (s < k_tps % STAGES ? 1 : 0);
}

template <int STAGES>
CUTE_DEVICE void wait_stage_free(const GemmPipelineBars& bars,
                                 uint32_t base_bits, uint32_t base_zero_bits,
                                 int seq, int k_tps, int t, int site,
                                 int item) {
    const int s = t % STAGES;
    const bool stage_untouched = ((base_zero_bits >> s) & 1u) != 0u;
    const bool first_ever = stage_untouched && seq == 0 && t < STAGES;
    if (first_ever) return;
    const int need = (int)((base_bits >> s) & 1u) +
                     seq * stage_uses<STAGES>(k_tps, s) + t / STAGES;
    __syncwarp();
    mbar_wait(bars.smem_free[s], (uint32_t)((need - 1) & 1), site, item, t);
}

template <int STAGES>
CUTE_DEVICE void wait_stage_tma(const GemmPipelineBars& bars,
                                uint32_t base_bits, int seq, int k_tps, int k,
                                int site_a, int site_b, int item) {
    const int s = k % STAGES;
    const int need = (int)((base_bits >> s) & 1u) +
                     seq * stage_uses<STAGES>(k_tps, s) + k / STAGES + 1;
    const uint32_t phase = (uint32_t)((need - 1) & 1);
    __syncwarp();
    mbar_wait(bars.tma_a[s], phase, site_a, item, k);
    mbar_wait(bars.tma_b[s], phase, site_b, item, k);
}

template <int ACC_STAGES>
CUTE_DEVICE int reserve_acc_slot(const GemmPipelineBars& bars, int base_acc,
                                 int acc_seq, int site, int item) {
    const int total = base_acc + acc_seq;
    const int slot = total % ACC_STAGES;
    if (total >= ACC_STAGES) {
        const int lap = total / ACC_STAGES;
        __syncwarp();
        mbar_wait(bars.acc_empty[slot], (uint32_t)((lap - 1) & 1), site, item,
                  -1);
    }
    return slot;
}

template <int ACC_STAGES>
CUTE_DEVICE int claim_acc_slot(const GemmPipelineBars& bars, int base_acc,
                               int acc_seq, int site, int item) {
    const int total = base_acc + acc_seq;
    const int slot = total % ACC_STAGES;
    const int lap = total / ACC_STAGES;
    mbar_wait(bars.acc_full[slot], (uint32_t)(lap & 1), site, item, -1);
    return slot;
}

template <int STAGES, class F>
CUTE_DEVICE void dispatch_stage(int s, F&& f, int k) {
    static_assert(STAGES >= 1 && STAGES <= 4,
                  "dispatch_stage enumerates the stage index as a compile-time "
                  "value and only covers 1..4; a deeper pipeline that falls "
                  "off the end would run the WRONG stage's descriptor "
                  "silently -- the barrier index is computed elsewhere, so "
                  "nothing asserts and the TMA never completes");
    if (s == 0) {
        f(cute::Int<0>{}, k);
    } else if constexpr (STAGES > 2) {
        if (s == 1)
            f(cute::Int<1>{}, k);
        else if constexpr (STAGES > 3) {
            if (s == 2)
                f(cute::Int<2>{}, k);
            else
                f(cute::Int<STAGES - 1>{}, k);
        } else {
            f(cute::Int<STAGES - 1>{}, k);
        }
    } else {
        f(cute::Int<STAGES - 1>{}, k);
    }
}

struct KvRuns {
    int row0, row1;
    int slot0, slot1;
    int len0;
    int total;
    int page_shift;
};

CUTE_DEVICE KvRuns load_kv_runs(const int4* __restrict__ kv_runs, int token_idx,
                                int total, int step, int page_shift,
                                int rows_per_page) {
    static_assert(AttnConfig::MaxKvSegsPerToken == 2,
                  "the 16 B run-list load below assumes 2 runs");

    const int4 raw = __ldg(kv_runs + token_idx);
    const int d_slot0 = raw.x, d_len0 = raw.y, d_slot1 = raw.z;
    const int len1 = total - d_len0;
    KvRuns r;
    r.total = total;
    r.page_shift = page_shift;
    if (step > 0) {
        r.slot0 = d_slot0;
        r.len0 = min(d_len0, total);
        r.slot1 = d_slot1;
    } else if (len1 > 0) {
        r.slot0 = d_slot1 - (len1 - 1);
        r.len0 = len1;
        r.slot1 = d_slot0 - (d_len0 - 1);
    } else {
        r.slot0 = d_slot0 - (total - 1);
        r.len0 = total;
        r.slot1 = r.slot0;
    }
    const int mask = (1 << page_shift) - 1;
    r.row0 = (r.slot0 >> page_shift) * rows_per_page + (r.slot0 & mask);
    r.row1 = (r.slot1 >> page_shift) * rows_per_page + (r.slot1 & mask);
    return r;
}

CUTE_DEVICE int kv_run_slot(const KvRuns& r, int g) {
    return (g < r.len0) ? r.slot0 + g : r.slot1 + (g - r.len0);
}

CUTE_DEVICE int kv_run_row(const KvRuns& r, int g) {
    if (g >= r.total) return 0;
    return (g < r.len0) ? r.row0 + g : r.row1 + (g - r.len0);
}

CUTE_DEVICE int4 kv_run_row4(const KvRuns& r, int g0) {
    const int lim = min(r.len0, r.total);
    int base = -1;
    if (g0 + 3 < lim)
        base = r.row0 + g0;
    else if (g0 >= r.len0 && g0 + 3 < r.total)
        base = r.row1 + (g0 - r.len0);
    if (base >= 0) return make_int4(base, base + 1, base + 2, base + 3);
    return make_int4(kv_run_row(r, g0), kv_run_row(r, g0 + 1),
                     kv_run_row(r, g0 + 2), kv_run_row(r, g0 + 3));
}

CUTE_DEVICE bool kv_run_chunk(const KvRuns& r, int g0, int n, int& row) {
    const int lim = min(r.len0, r.total);
    if (g0 + n <= lim) {
        row = r.row0 + g0;
        return true;
    }
    if (g0 >= r.len0 && g0 + n <= r.total) {
        row = r.row1 + (g0 - r.len0);
        return true;
    }
    return false;
}

template <int SPAN>
CUTE_DEVICE void issue_kv_span(const CUtensorMap* __restrict__ dense,
                               const void* gather_desc, transac_bar_t& bar,
                               uint8_t* smem_base, int row_bytes,
                               const KvRuns& r, int g_base, int g_lo) {
    constexpr int64_t HINT = (int64_t)cute::TMA::CacheHintSm90::EVICT_LAST;
    constexpr int TOP = AttnConfig::KvDenseLevels - 1;
    int g = g_lo;
    const int g_hi = g_lo + SPAN;
    CUTE_NO_UNROLL
    while (g < g_hi) {
        const int rem = g_hi - g;
        int found = -1, row = 0;
        CUTE_UNROLL
        for (int k = TOP; k >= 0; --k) {
            const int cand = 4 << k;
            if (found < 0 && cand <= rem &&
                kv_run_chunk(r, g_base + g, cand, row))
                found = k;
        }
        uint8_t* dst = smem_base + (size_t)row_bytes * g;
        if (found >= 0) {
            ku::tma_load_2d(&dense[found], bar, dst, 0, row, HINT);
            g += 4 << found;
        } else {
            ku::tma_gather4(gather_desc, bar, dst, 0,
                            kv_run_row4(r, g_base + g), HINT);
            g += 4;
        }
    }
}

struct KvSegGeom {
    const uint8_t* pool;
    int stride_kv_block;
};

CUTE_DEVICE KvRuns kv_tile_slice(const KvRuns& r, int g_base, int span) {
    const int valid = min(span, max(r.total - g_base, 0));
    KvRuns o;
    o.page_shift = r.page_shift;
    o.total = valid;
    if (g_base < r.len0) {
        o.slot0 = r.slot0 + g_base;
        o.row0 = r.row0 + g_base;
        o.len0 = min(r.len0 - g_base, valid);

        o.slot1 = r.slot1;
        o.row1 = r.row1;
    } else {
        const int d = g_base - r.len0;
        o.slot0 = r.slot1 + d;
        o.row0 = r.row1 + d;
        o.len0 = valid;
        o.slot1 = o.slot0;
        o.row1 = o.row0;
    }
    return o;
}

CUTE_DEVICE KvRuns kv_tile_ascending(int slot0, int total, int g_base, int span,
                                     int page_shift, int rows_per_page) {
    const int mask = (1 << page_shift) - 1;
    const int valid = min(span, max(total - g_base, 0));
    const int s0 = slot0 + g_base;
    KvRuns o;
    o.page_shift = page_shift;
    o.total = valid;
    o.slot0 = s0;
    o.len0 = min(valid, (1 << page_shift) - (s0 & mask));
    o.slot1 = s0 + o.len0;
    o.row0 = (o.slot0 >> page_shift) * rows_per_page + (o.slot0 & mask);
    o.row1 = (o.slot1 >> page_shift) * rows_per_page + (o.slot1 & mask);
    return o;
}

CUTE_DEVICE const uint64_t* kv_seg_scale_ptr(const KvRuns& r,
                                             const KvSegGeom& seg, int g) {
    const int slot = (g < r.total) ? kv_run_slot(r, g) : 0;
    const int page = slot >> r.page_shift;
    const int off = slot & ((1 << r.page_shift) - 1);
    return (const uint64_t*)(seg.pool + (size_t)page * seg.stride_kv_block +
                             ((size_t)1 << r.page_shift) *
                                 ModelDims::TMA_K_STRIDE +
                             (size_t)off * 8);
}

struct PartialBarrier {
    int* slot;
    int n_producers;

    CUTE_DEVICE PartialBarrier(int* base, int idx, int n_prod)
        : slot(base + (int64_t)idx *
                          (WorkspaceLayout::RendezvousSlotBytes / sizeof(int))),
          n_producers(n_prod) {}

    CUTE_DEVICE void arrive_warp() const {
        __syncwarp();
        if (cute::elect_one_sync()) ku::gmem_red_add1_release(slot);
    }

    CUTE_DEVICE void wait(uint32_t epoch,
                          [[maybe_unused]] int site = TraceProbe::WS_NONE,
                          [[maybe_unused]] int g = -1) const {
        const int target = (int)(epoch + 1) * n_producers;
        int seen;
        TraceProbe::thread(site, 0, g, target);
        do {
            seen = ku::gmem_ld_relaxed(slot);
        } while (seen < target);
        seen = ku::gmem_ld_acquire(slot);
        TraceProbe::thread(site, 1, g, seen);
    }

    CUTE_DEVICE void wait_warp(uint32_t epoch, int site = TraceProbe::WS_NONE,
                               int g = -1) const {
        if (cute::elect_one_sync()) wait(epoch, site, g);
        __syncwarp();
    }

    CUTE_DEVICE bool try_wait(uint32_t epoch) const {
        return ku::gmem_ld_acquire(slot) >= (int)(epoch + 1) * n_producers;
    }

    CUTE_DEVICE bool try_wait_warp(uint32_t epoch) const {
        int ok = 0;
        if (cute::elect_one_sync()) ok = try_wait(epoch) ? 1 : 0;
        return __any_sync(0xffffffffu, ok) != 0;
    }

    CUTE_DEVICE void reset_relaxed() const { ku::gmem_st_zero_relaxed(slot); }
};

}  // namespace mega_mla
