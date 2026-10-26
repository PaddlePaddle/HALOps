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

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/numeric_types.h>

#include <cute/tensor.hpp>

#include "kernel/mla/mla_intrinsics.cuh"
#include "kernel/mla/mla_traits.cuh"

namespace halops::mla {
template <class Bar>
CUTE_DEVICE void bar_sync() {
    cutlass::arch::NamedBarrier::arrive_and_wait(Bar::width, Bar::id);
}

CUTE_DEVICE void cta_sync() {
    __syncwarp();
    bar_sync<KernelBase::Bars::EveryoneSync>();
}

CUTE_DEVICE void phase_end_sync() {
    cta_sync();
    cutlass::arch::fence_barrier_init();
    bar_sync<KernelBase::Bars::EveryoneSync>();
}

CUTE_DEVICE void bar_sync_warp_pair(int local_warp) {
    using Bars = KernelBase::Bars;
    static_assert(Bars::Wg1Warp13Sync::id == Bars::Wg1Warp02Sync::id + 1,
                  "pair barrier ids are adjacent");
    cutlass::arch::NamedBarrier::arrive_and_wait(
        Bars::Wg1Warp02Sync::width,
        Bars::Wg1Warp02Sync::id + (uint32_t)(local_warp & 1));
}

template <int Tokens>
CUTE_DEVICE int swizzled_token_index(int row, int token) {
    constexpr int kChunks = Tokens / 4;
    return row * Tokens + ((((token >> 2) ^ row) & (kChunks - 1)) << 2) +
           (token & 3);
}

CUTE_DEVICE int rev_bid() { return gridDim.x - 1 - blockIdx.x; }
CUTE_DEVICE int items_from(int first, int step, int count) {
    return count > first ? (count - first + step - 1) / step : 0;
}

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
    if (runs.pt)
        return make_int4(kv_run_row(runs, g0), kv_run_row(runs, g0 + 1),
                         kv_run_row(runs, g0 + 2), kv_run_row(runs, g0 + 3));
    const int lim = min(runs.len0, runs.total);
    int base = -1;
    if (g0 + 3 < lim)
        base = runs.row0 + g0;
    else if (g0 >= runs.len0 && g0 + 3 < runs.total)
        base = runs.row1 + (g0 - runs.len0);
    if (base >= 0) return make_int4(base, base + 1, base + 2, base + 3);
    return make_int4(kv_run_row(runs, g0), kv_run_row(runs, g0 + 1),
                     kv_run_row(runs, g0 + 2), kv_run_row(runs, g0 + 3));
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
    const int j = runs.j0 + g0;
    const int mask = (1 << runs.page_shift) - 1;
    const int i_max = max((runs.j0 + runs.total - 1) >> runs.page_shift, 0);
    const int i_lo = min(j >> runs.page_shift, i_max);
    const int i_hi = min((j + 3) >> runs.page_shift, i_max);
    const int base_lo = __ldg(runs.pt + i_lo) * runs.rows_per_page;
    const int base_hi = __ldg(runs.pt + i_hi) * runs.rows_per_page;
    int rows[4];
    CUTE_UNROLL
    for (int i = 0; i < 4; ++i) {
        const int idx = j + i;
        const int base = ((idx >> runs.page_shift) == i_lo) ? base_lo : base_hi;
        rows[i] = (g0 + i < runs.total) ? base + (idx & mask) : 0;
    }
    return make_int4(rows[0], rows[1], rows[2], rows[3]);
}

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

struct PartialBarrier {
    int* slot;
    int n_producers;

    CUTE_DEVICE PartialBarrier(int* base, int idx, int n_prod)
        : slot(base + (int64_t)idx *
                          (WorkspaceLayout::RendezvousSlotBytes / sizeof(int))),
          n_producers(n_prod) {}

    CUTE_DEVICE void arrive_thread() const { gmem_red_add1_release(slot); }

    CUTE_DEVICE void wait(uint32_t epoch) const {
        const int target = (int)(epoch + 1) * n_producers;
        int seen;
        do {
            seen = gmem_ld_relaxed(slot);
        } while (seen < target);
        gmem_ld_acquire(slot);
    }

    CUTE_DEVICE void wait_warp(uint32_t epoch) const {
        if (cute::elect_one_sync()) wait(epoch);
        __syncwarp();
    }

    CUTE_DEVICE void wait_warp_for_tma(uint32_t epoch) const {
        wait_warp(epoch);
        fence_proxy_async_global();
    }

    CUTE_DEVICE void reset_relaxed() const { gmem_st_zero_relaxed(slot); }
};

template <int N>
CUTE_DEVICE void fma_bf16(float weight, const void* vals, float (&acc)[N]) {
    const auto* v = reinterpret_cast<const __nv_bfloat16*>(vals);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i)
        acc[i] = fmaf(weight, __bfloat162float(v[i]), acc[i]);
}

template <int N>
CUTE_DEVICE void pack_bf16(const float (&vals)[N], void* out) {
    auto* pairs = reinterpret_cast<__nv_bfloat162*>(out);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i)
        pairs[i] = __floats2bfloat162_rn(vals[2 * i], vals[2 * i + 1]);
}

template <int N>
CUTE_DEVICE void unpack_bf16(const void* in, float (&vals)[N]) {
    const auto* pairs = reinterpret_cast<const __nv_bfloat162*>(in);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
        const float2 v = __bfloat1622float2(pairs[i]);
        vals[2 * i] = v.x;
        vals[2 * i + 1] = v.y;
    }
}

template <int N = 8>
CUTE_DEVICE void rope_slice(const float* __restrict__ fr, int rb,
                            float (&cs)[N]) {
    static_assert(N % 4 == 0, "whole float4 loads");
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < N / 4; ++k)
        reinterpret_cast<float4*>(cs)[k] =
            reinterpret_cast<const float4*>(fr + rb)[k];
}

CUTE_DEVICE float2 rope_rotate(float x0, float x1, float cos_v, float sin_v) {
    return make_float2(x0 * cos_v - x1 * sin_v, x0 * sin_v + x1 * cos_v);
}
CUTE_DEVICE float2 rope_unrotate(float x0, float x1, float cos_v, float sin_v) {
    return make_float2(x0 * cos_v + x1 * sin_v, -x0 * sin_v + x1 * cos_v);
}

template <int N>
CUTE_DEVICE void rope_rotate(float (&vals)[N], const float (&cs)[N]) {
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < N / 2; ++k) {
        const float2 r =
            rope_rotate(vals[2 * k], vals[2 * k + 1], cs[2 * k], cs[2 * k + 1]);
        vals[2 * k] = r.x;
        vals[2 * k + 1] = r.y;
    }
}

template <int N>
CUTE_DEVICE void inverse_rope_bf16(float (&vals)[N], const float (&cs)[N]) {
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < N / 2; ++k) {
        const float2 r = rope_unrotate(vals[2 * k], vals[2 * k + 1], cs[2 * k],
                                       cs[2 * k + 1]);
        const float2 rounded =
            __bfloat1622float2(__floats2bfloat162_rn(r.x, r.y));
        vals[2 * k] = rounded.x;
        vals[2 * k + 1] = rounded.y;
    }
}

template <int N>
CUTE_DEVICE float rms_inv(float sq_sum, float eps) {
    return rsqrtf(sq_sum / N + eps);
}

struct Fp8Format {
    static constexpr float E4m3Max = 448.0f;
    static constexpr int Ue8m0Bias =
        cutlass::detail::exponent_bias_cxx17<8, 0>();
    static constexpr float Ue8m0MaxExp = 2 * Ue8m0Bias;
    static constexpr uint64_t UnitScales = 0x0101010101010101ull * Ue8m0Bias;
    static constexpr float AmaxFloor = 1e-10f;
};

CUTE_DEVICE float ue8m0_exp_ceil(float amax) {
    return fminf(
        fmaxf(ceilf(log2f(amax / Fp8Format::E4m3Max)) + Fp8Format::Ue8m0Bias,
              0.0f),
        Fp8Format::Ue8m0MaxExp);
}

template <int TileTokens>
CUTE_HOST_DEVICE constexpr int sf_row_of(int token) {
    if constexpr (TileTokens < SfLayout::PackedMinTile)
        return (token / TileTokens) * SfLayout::MnPerTile +
               (token % TileTokens);
    else
        return token;
}

CUTE_HOST_DEVICE int sf_row_grouped(int token, int group_tokens, int n_groups) {
    if (n_groups > 1 && group_tokens % SfLayout::MnPerTile) {
        const int atom_rows =
            cute::round_up(group_tokens, int{SfLayout::MnPerTile});
        return (token / group_tokens) * atom_rows + token % group_tokens;
    }
    return token;
}

CUTE_DEVICE float ue8m0_quant_scale(float biased_exp) {
    constexpr int kF32MantissaBits = 23;
    const int exp_bits = (int)biased_exp;
    return exp_bits < Fp8Format::Ue8m0MaxExp
               ? __int_as_float((2 * Fp8Format::Ue8m0Bias - exp_bits)
                                << kF32MantissaBits)
               : exp2f(Fp8Format::Ue8m0Bias - biased_exp);
}

struct Ue8m0Scale {
    float biased_exp;
    float multiplier;
};
CUTE_DEVICE Ue8m0Scale ue8m0_scale(float amax) {
    const float biased_exp = ue8m0_exp_ceil(amax);
    return {biased_exp, ue8m0_quant_scale(biased_exp)};
}

template <int N>
CUTE_DEVICE float abs_max(const float (&vals)[N]) {
    float amax = Fp8Format::AmaxFloor;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) amax = fmaxf(amax, fabsf(vals[i]));
    return amax;
}

template <int Lanes>
CUTE_DEVICE float lane_group_max(float value, unsigned mask = 0xffffffffu) {
    CUTLASS_PRAGMA_UNROLL
    for (int offset = 1; offset < Lanes; offset <<= 1)
        value = fmaxf(value, __shfl_xor_sync(mask, value, offset));
    return value;
}

CUTE_DEVICE uint32_t pack_e4m3x4(float v0, float v1, float v2, float v3,
                                 float scale) {
    const float2 lo = make_float2(v0 * scale, v1 * scale);
    const float2 hi = make_float2(v2 * scale, v3 * scale);
    return (uint32_t)__nv_cvt_float2_to_fp8x2(lo, __NV_SATFINITE, __NV_E4M3) |
           ((uint32_t)__nv_cvt_float2_to_fp8x2(hi, __NV_SATFINITE, __NV_E4M3)
            << 16);
}

template <int N>
CUTE_DEVICE void pack_e4m3(const float (&vals)[N], float scale,
                           uint32_t (&words)[N / 4]) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 4; ++i)
        words[i] = pack_e4m3x4(vals[4 * i], vals[4 * i + 1], vals[4 * i + 2],
                               vals[4 * i + 3], scale);
}

template <int Lanes, int N>
CUTE_DEVICE Ue8m0Scale quantize_group(const float (&vals)[N],
                                      uint32_t (&words)[N / 4]) {
    const Ue8m0Scale scale = ue8m0_scale(lane_group_max<Lanes>(abs_max(vals)));
    pack_e4m3(vals, scale.multiplier, words);
    return scale;
}

template <int TileTokens, class LayoutSFB>
CUTE_DEVICE void quantize_group32_from_regs8(
    const __nv_bfloat16 (&vals)[8], cutlass::float_e4m3_t* __restrict__ ws_fp8,
    cutlass::float_ue8m0_t* __restrict__ ws_sf, LayoutSFB const& layout_sfb,
    int n_row, int k_dim, int group, int lane) {
    const auto* v2 = reinterpret_cast<const __nv_bfloat162*>(vals);
    const __nv_bfloat162 m2 = __hmax2(__hmax2(__habs2(v2[0]), __habs2(v2[1])),
                                      __hmax2(__habs2(v2[2]), __habs2(v2[3])));
    float amax = fmaxf(fmaxf(__bfloat162float(__low2bfloat16(m2)),
                             __bfloat162float(__high2bfloat16(m2))),
                       Fp8Format::AmaxFloor);

    constexpr int kLanes = SfLayout::Gran / 8;
    amax = lane_group_max<kLanes>(amax, ((1u << kLanes) - 1)
                                            << (lane & ~(kLanes - 1)));

    const Ue8m0Scale sc = ue8m0_scale(amax);
    const float biased_exp = sc.biased_exp;
    const __nv_bfloat162 inv2 = __float2bfloat162_rn(sc.multiplier);

    const int base = group * SfLayout::Gran;
    const int j = lane & (kLanes - 1);
    uint32_t packed[2];
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 2; ++i) {
        const __nv_bfloat162 p0 = __hmul2(v2[i * 2], inv2);
        const __nv_bfloat162 p1 = __hmul2(v2[i * 2 + 1], inv2);
        packed[i] = (uint32_t)__nv_cvt_bfloat16raw2_to_fp8x2(
                        *reinterpret_cast<const __nv_bfloat162_raw*>(&p0),
                        __NV_SATFINITE, __NV_E4M3) |
                    ((uint32_t)__nv_cvt_bfloat16raw2_to_fp8x2(
                         *reinterpret_cast<const __nv_bfloat162_raw*>(&p1),
                         __NV_SATFINITE, __NV_E4M3)
                     << 16);
    }

    *reinterpret_cast<uint2*>(ws_fp8 + (int64_t)n_row * k_dim + base + j * 8) =
        *reinterpret_cast<const uint2*>(packed);

    if (j == 0) {
        cute::Tensor mSFB = cute::make_tensor(
            cute::make_gmem_ptr(reinterpret_cast<uint8_t*>(ws_sf)), layout_sfb);
        mSFB(sf_row_of<TileTokens>(n_row), base, cute::_0{}) =
            static_cast<uint8_t>(biased_exp);
    }
}
}  // namespace halops::mla
