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
#include <math_constants.h>

#include <cfloat>
#include <cute/tensor.hpp>

#include "kernel/mla/mla_mainloop.cuh"

namespace halops::mla {
template <int TileTokens, class LayoutSFB, class Arrive>
__device__ void quantize_x(const __nv_bfloat16* __restrict__ x_input,
                           int hidden_row_stride,
                           cutlass::float_e4m3_t* __restrict__ ws_x_fp8,
                           cutlass::float_ue8m0_t* __restrict__ ws_x_sf,
                           LayoutSFB const& layout_sfb, int num_tokens,
                           int hidden_in, Arrive&& arrive) {
    using Roles = KernelBase::Roles;

    const int vec_tid =
        threadIdx.x - Roles::VecWarpBegin * KernelBase::WarpSize;
    const int grp0 = vec_tid / QuantX::ThreadsPerGrp,
              j = vec_tid % QuantX::ThreadsPerGrp;
    const int lane = vec_tid & (KernelBase::WarpSize - 1);
    const int n_groups = (num_tokens + TileTokens - 1) / TileTokens;
    const int n_grp_k = hidden_in / SfLayout::Gran;

    constexpr int kBatch = 4;
    const int stride = gridDim.x * QuantX::GrpPerPass;
    for (int g_tile = 0; g_tile < n_groups; ++g_tile) {
        const int nv = min(TileTokens, num_tokens - g_tile * TileTokens);
        const int n_items = nv * n_grp_k;
        for (int base0 = blockIdx.x * QuantX::GrpPerPass; base0 < n_items;
             base0 += kBatch * stride) {
            uint4 vals[kBatch];
            CUTLASS_PRAGMA_UNROLL
            for (int unit = 0; unit < kBatch; ++unit) {
                const int item = base0 + unit * stride + grp0;
                if (item < n_items) {
                    const int row = item / n_grp_k;
                    const int k_group = item - row * n_grp_k;
                    const int token = g_tile * TileTokens + row;
                    vals[unit] = __ldg(reinterpret_cast<const uint4*>(
                        x_input + (int64_t)token * hidden_row_stride +
                        k_group * SfLayout::Gran + j * QuantX::PerThread));
                }
            }
            const int nu = min(kBatch, (n_items - base0 + stride - 1) / stride);
            CUTLASS_PRAGMA_NO_UNROLL
            for (int unit = 0; unit < nu; ++unit) {
                const int item = base0 + unit * stride + grp0;
                uint4 cur = vals[0];
                CUTLASS_PRAGMA_UNROLL
                for (int k = 1; k < kBatch; ++k)
                    if (unit == k) cur = vals[k];
                if (item < n_items) {
                    const int row = item / n_grp_k;
                    const int k_group = item - row * n_grp_k;
                    const int token = g_tile * TileTokens + row;
                    quantize_group32_from_regs8<TileTokens>(
                        *reinterpret_cast<__nv_bfloat16(*)[8]>(&cur), ws_x_fp8,
                        ws_x_sf, layout_sfb, token, hidden_in, k_group, lane);
                }
            }
        }
        if (QuantX::active(blockIdx.x, n_items)) arrive(g_tile);
    }
}

template <int TileTokens, int QLoraRank, int KSplit, class PartT,
          class LayoutSFB>
__device__ void q_reduce_line(const PartT* __restrict__ ws_qlora_part,
                              cutlass::float_e4m3_t* __restrict__ ws_qlora_fp8,
                              cutlass::float_ue8m0_t* __restrict__ ws_qlora_sf,
                              LayoutSFB const& layout_sfb,
                              float* __restrict__ sq_buf, int token,
                              float q_eps, const uint4& w_pre) {
    using Roles = KernelBase::Roles;
    constexpr int kPasses = QLoraRank / SfLayout::Gran / QuantX::GrpPerPass;
    static_assert(kPasses == 1, "w_pre holds the norm weight of one pass");

    const int vec_tid =
        threadIdx.x - Roles::VecWarpBegin * KernelBase::WarpSize;
    const int grp0 = vec_tid / QuantX::ThreadsPerGrp,
              j = vec_tid & (QuantX::ThreadsPerGrp - 1);
    const int lane = vec_tid & (KernelBase::WarpSize - 1);
    const int wv = vec_tid / KernelBase::WarpSize;

    const PartT* part = ws_qlora_part + (int64_t)token * (KSplit * QLoraRank);
    constexpr int kVecLoads = (int)(QuantX::PerThread * sizeof(PartT)) / 16;

    float vals[kPasses][QuantX::PerThread];
    float sq[kPasses];
    CUTLASS_PRAGMA_UNROLL
    for (int pass = 0; pass < kPasses; ++pass) {
        const int d0 = (grp0 + pass * QuantX::GrpPerPass) * SfLayout::Gran +
                       j * QuantX::PerThread;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < QuantX::PerThread; ++i) vals[pass][i] = 0.f;
        CUTLASS_PRAGMA_UNROLL
        for (int split = 0; split < KSplit; ++split) {
            CUTE_ALIGNAS(16) PartT part_vals[QuantX::PerThread];
            CUTLASS_PRAGMA_UNROLL
            for (int chunk = 0; chunk < kVecLoads; ++chunk)
                reinterpret_cast<uint4*>(part_vals)[chunk] =
                    reinterpret_cast<const uint4*>(part + split * QLoraRank +
                                                   d0)[chunk];
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < QuantX::PerThread; ++i)
                vals[pass][i] += (float)part_vals[i];
        }

        sq[pass] = 0.f;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < QuantX::PerThread; ++i)
            sq[pass] += vals[pass][i] * vals[pass][i];
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < QuantX::PerThread; ++i)
            vals[pass][i] = __bfloat162float(__float2bfloat16(vals[pass][i]));
    }

    CUTLASS_PRAGMA_UNROLL
    for (int pass = 0; pass < kPasses; ++pass) {
        sq[pass] = warp_sum(sq[pass]);
        if (lane == 0) sq_buf[wv + pass * Roles::VecWarps] = sq[pass];
    }
    bar_sync<KernelBase::Bars::Wg2Sync>();
    float total_sq = 0.0f;
    CUTLASS_PRAGMA_UNROLL
    for (int slot = 0; slot < Roles::VecWarps * kPasses; ++slot)
        total_sq += sq_buf[slot];
    const float inv = rms_inv<QLoraRank>(total_sq, q_eps);

    {
        CUTLASS_PRAGMA_UNROLL
        for (int pass = 0; pass < kPasses; ++pass) {
            const int group = grp0 + pass * QuantX::GrpPerPass;
            __nv_bfloat16 w4[QuantX::PerThread];
            *reinterpret_cast<uint4*>(w4) = w_pre;
            __nv_bfloat16 q_bf16[QuantX::PerThread];
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < QuantX::PerThread; ++i)
                q_bf16[i] = __float2bfloat16(vals[pass][i] * inv *
                                             __bfloat162float(w4[i]));
            quantize_group32_from_regs8<TileTokens>(
                q_bf16, ws_qlora_fp8, ws_qlora_sf, layout_sfb, token, QLoraRank,
                group, lane);
        }
    }

    bar_sync<KernelBase::Bars::Wg2Sync>();
}

CUTE_DEVICE void pack_kv_row(int vec_tid,
                             const __nv_bfloat16* __restrict__ s_row,
                             uint8_t* __restrict__ data_ptr,
                             uint8_t* __restrict__ scale_ptr) {
    using Kv = KvCacheLayout;
    constexpr int kVec = 16 / sizeof(__nv_bfloat16);
    constexpr int kTeam = Kv::NopeQuantBlock / kVec;
    constexpr int kRopeThreads =
        ModelDims::DimRope * sizeof(__nv_bfloat16) / 16;
    if (vec_tid >= Kv::ScaleSlotBytes * kTeam) return;
    const int group = vec_tid / kTeam, j = vec_tid % kTeam;
    const int start = min(group, Kv::NopeGroups - 1) * Kv::NopeQuantBlock;
    __nv_bfloat16 vals[kVec];
    *reinterpret_cast<uint4*>(vals) =
        *reinterpret_cast<const uint4*>(s_row + start + j * kVec);
    float max_abs = 0.f;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kVec; ++i)
        max_abs = fmaxf(max_abs, fabsf(__bfloat162float(vals[i])));
    CUTLASS_PRAGMA_UNROLL
    for (int offset = 1; offset < kTeam; offset <<= 1)
        max_abs = fmaxf(max_abs, __shfl_xor_sync(0xffffffffu, max_abs, offset));
    constexpr float kKvAmaxFloor = 1e-4f;
    const float scale_raw = fmaxf(max_abs, kKvAmaxFloor) / Fp8Format::E4m3Max;
    const int bias = Fp8Format::Ue8m0Bias;
    const uint32_t scale_bits = __float_as_uint(scale_raw);
    const int exp_val =
        min((int)Fp8Format::Ue8m0MaxExp,
            (int)((scale_bits >> 23) & 0xFF) + ((scale_bits & 0x7FFFFF) != 0));
    const float inv = 1.0f / exp2f((float)(exp_val - bias));
    cutlass::float_e4m3_t out[kVec];
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kVec; ++i)
        out[i] = cutlass::float_e4m3_t(__bfloat162float(vals[i]) * inv);
    if (group < Kv::NopeGroups) {
        *reinterpret_cast<uint2*>(data_ptr + start + j * kVec) =
            *reinterpret_cast<const uint2*>(out);
        if (j == 0) scale_ptr[group] = (uint8_t)exp_val;
    }

    if (vec_tid < kRopeThreads)
        *reinterpret_cast<uint4*>(data_ptr + ModelDims::DimNope +
                                  vec_tid * 16) =
            *reinterpret_cast<const uint4*>(s_row + ModelDims::DimNope +
                                            vec_tid * kVec);
}

template <int KSplit, class PartT>
CUTE_DEVICE void sum_partials4(const PartT* __restrict__ part, int dim,
                               float (&vals)[4]) {
    constexpr int kRunBytes = (int)(4 * sizeof(PartT));
    CUTLASS_PRAGMA_UNROLL
    for (int split = 0; split < KSplit; ++split) {
        CUTE_ALIGNAS(16) PartT part_vals[4];
        if constexpr (kRunBytes == 16)
            *reinterpret_cast<uint4*>(part_vals) =
                *reinterpret_cast<const uint4*>(part + split * dim);
        else
            *reinterpret_cast<uint2*>(part_vals) =
                *reinterpret_cast<const uint2*>(part + split * dim);
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < 4; ++i) vals[i] += (float)part_vals[i];
    }
}

template <int KSplit, class PartT>
CUTE_DEVICE void kv_write_line(int vec_tid,
                               const PartT* __restrict__ ws_kv_part, int token,
                               const __nv_bfloat16* __restrict__ norm_weight,
                               const float* __restrict__ freqs_pos,
                               uint8_t* __restrict__ kv_pool,
                               int stride_kv_block, int page_block_size,
                               int loc, float eps, float* __restrict__ sq_buf,
                               __nv_bfloat16* __restrict__ s_row) {
    constexpr int kDim = ModelDims::DimV, kNope = ModelDims::DimNope,
                  ROW_STRIDE = ModelDims::KvRowBytes;
    using Roles = KernelBase::Roles;
    static_assert(Roles::VecWarps * KernelBase::WarpSize * 4 == kDim,
                  "4 kv dims per vec thread");

    const int lane = vec_tid & (KernelBase::WarpSize - 1);
    const int wv = vec_tid / KernelBase::WarpSize;

    const int e0 = vec_tid * 4;

    float vals[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    sum_partials4<KSplit>(ws_kv_part + (int64_t)token * (KSplit * kDim) + e0,
                          kDim, vals);
    CUTE_ALIGNAS(8) __nv_bfloat16 vb[4];
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) vb[i] = __float2bfloat16(vals[i]);

    float sq = 0.0f;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
        const float val = __bfloat162float(vb[i]);
        sq += val * val;
    }
    sq = warp_sum(sq);
    if (lane == 0) sq_buf[wv] = sq;
    bar_sync<KernelBase::Bars::Wg2Sync>();
    float total = 0.0f;
    CUTLASS_PRAGMA_UNROLL
    for (int slot = 0; slot < Roles::VecWarps; ++slot) total += sq_buf[slot];
    const float inv = rms_inv<kDim>(total, eps);

    {
        CUTE_ALIGNAS(8) __nv_bfloat16 nw[4], nb[4], ob[4];
        *reinterpret_cast<uint2*>(nw) =
            *reinterpret_cast<const uint2*>(norm_weight + e0);
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < 4; ++i)
            nb[i] = __float2bfloat16(__bfloat162float(vb[i]) * inv *
                                     __bfloat162float(nw[i]));
        if (e0 >= kNope) {
            const float4 fr =
                *reinterpret_cast<const float4*>(freqs_pos + (e0 - kNope));
            const float cs[4] = {fr.x, fr.y, fr.z, fr.w};
            CUTLASS_PRAGMA_UNROLL
            for (int k = 0; k < 2; ++k) {
                const float2 r = rope_rotate(__bfloat162float(nb[k * 2]),
                                             __bfloat162float(nb[k * 2 + 1]),
                                             cs[k * 2], cs[k * 2 + 1]);
                ob[k * 2] = __float2bfloat16(r.x);
                ob[k * 2 + 1] = __float2bfloat16(r.y);
            }
        } else {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < 4; ++i) ob[i] = nb[i];
        }
        *reinterpret_cast<uint2*>(s_row + e0) =
            *reinterpret_cast<const uint2*>(ob);
    }
    bar_sync<KernelBase::Bars::Wg2Sync>();

    const int kv_page = loc / page_block_size,
              kv_offset = loc % page_block_size;
    uint8_t* page_ptr = kv_pool + (int64_t)kv_page * stride_kv_block;
    pack_kv_row(vec_tid, s_row, page_ptr + kv_offset * ROW_STRIDE,
                page_ptr + (int64_t)page_block_size * ROW_STRIDE +
                    kv_offset * KvCacheLayout::ScaleSlotBytes);
    bar_sync<KernelBase::Bars::Wg2Sync>();
}

CUTE_DEVICE void c128_online(float (&run_max)[4], float (&es)[4],
                             float (&pr)[4], float4 kv4, float4 sc4,
                             float4 ap4) {
    const float* kva = reinterpret_cast<const float*>(&kv4);
    const float* sca = reinterpret_cast<const float*>(&sc4);
    const float* apa = reinterpret_cast<const float*>(&ap4);
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < 4; ++j) {
        const float sc = sca[j] + apa[j];
        const float diff = sc - run_max[j];
        const float decay = expf(-fabsf(diff));
        const float rs = (diff > 0.f) ? decay : 1.f;
        const float new_w = (diff > 0.f) ? 1.f : decay;
        es[j] = es[j] * rs + new_w;
        pr[j] = pr[j] * rs + kva[j] * new_w;
        run_max[j] = fmaxf(run_max[j], sc);
    }
}

template <class WaitRing>
CUTE_DEVICE void c128_slice_partial(int vec_tid, int slice, const float* ring,
                                    int kv_gate_k, int ring_row, int s0,
                                    const float* __restrict__ ape,
                                    float* row_ws, float* sq_ws,
                                    float* __restrict__ warp_sq,
                                    WaitRing&& wait_ring) {
    using Lay = C128Layout;
    constexpr int kRowsPerLane = ModelDims::C128Ratio / Lay::RowGroups;
    const int wv = vec_tid / KernelBase::WarpSize,
              lane = vec_tid % KernelBase::WarpSize;
    const int sg = lane / 2, dq = lane % 2;
    const int e0 =
        slice * Lay::SliceDim + wv * Lay::DimsPerWarp + dq * Lay::DimsPerLane;
    const int page_start = ring_row - (ModelDims::C128Ratio - 1);
    constexpr float kMinWeightSum = 1e-38f;
    constexpr float kLowest = -FLT_MAX;
    float run_max[4] = {kLowest, kLowest, kLowest, kLowest};
    float es[4] = {0.f, 0.f, 0.f, 0.f}, pr[4] = {0.f, 0.f, 0.f, 0.f};
    float4 kv[kRowsPerLane], sc[kRowsPerLane], ap[kRowsPerLane];
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kRowsPerLane; ++i) {
        const int slot = sg + i * Lay::RowGroups;
        ap[i] = __ldg(reinterpret_cast<const float4*>(
            ape + (int64_t)slot * ModelDims::DimV + e0));
        if (slot < s0) {
            const float* row = ring + (int64_t)(page_start + slot) * kv_gate_k;
            kv[i] = __ldcg(reinterpret_cast<const float4*>(row + e0));
            sc[i] = __ldcg(
                reinterpret_cast<const float4*>(row + ModelDims::DimV + e0));
        }
    }
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kRowsPerLane; ++i)
        if (sg + i * Lay::RowGroups < s0)
            c128_online(run_max, es, pr, kv[i], sc[i], ap[i]);
    wait_ring();
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kRowsPerLane; ++i) {
        const int slot = sg + i * Lay::RowGroups;
        if (slot >= s0) {
            const float* row = ring + (int64_t)(page_start + slot) * kv_gate_k;
            kv[i] = __ldcg(reinterpret_cast<const float4*>(row + e0));
            sc[i] = __ldcg(
                reinterpret_cast<const float4*>(row + ModelDims::DimV + e0));
            c128_online(run_max, es, pr, kv[i], sc[i], ap[i]);
        }
    }
    CUTLASS_PRAGMA_UNROLL
    for (int off = 2; off < KernelBase::WarpSize; off <<= 1) {
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < 4; ++j) {
            const float mo = __shfl_xor_sync(0xffffffffu, run_max[j], off);
            const float eo = __shfl_xor_sync(0xffffffffu, es[j], off);
            const float po = __shfl_xor_sync(0xffffffffu, pr[j], off);
            const float mm = fmaxf(run_max[j], mo);
            const float keep_w = expf(run_max[j] - mm), peer_w = expf(mo - mm);
            es[j] = es[j] * keep_w + eo * peer_w;
            pr[j] = pr[j] * keep_w + po * peer_w;
            run_max[j] = mm;
        }
    }
    float out[4], sq = 0.f;
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < 4; ++j) {
        out[j] = pr[j] / fmaxf(es[j], kMinWeightSum);
        sq += out[j] * out[j];
    }
    if (sg == 0)
        __stcg(reinterpret_cast<float4*>(row_ws + e0),
               make_float4(out[0], out[1], out[2], out[3]));
    sq += __shfl_xor_sync(0xffffffffu, sq, 1);
    if (lane == 0) warp_sq[wv] = sq;
    bar_sync<KernelBase::Bars::Wg2Sync>();
    if (vec_tid == 0) {
        float total = 0.f;
        CUTLASS_PRAGMA_UNROLL
        for (int w = 0; w < KernelBase::Roles::VecWarps; ++w)
            total += warp_sq[w];
        __stcg(sq_ws + slice, total);
    }
}

template <class Params>
CUTE_DEVICE void c128_finalize_line(int vec_tid, int token, const float* row_ws,
                                    const float* sq_ws, const Params& p,
                                    __nv_bfloat16* __restrict__ s_row) {
    const __nv_bfloat16* __restrict__ norm_weight =
        reinterpret_cast<const __nv_bfloat16*>(p.c128_norm_weight);
    const float* __restrict__ freqs_pos =
        p.c128_freqs_real +
        (int64_t)(p.positions[token] - (ModelDims::C128Ratio - 1)) *
            ModelDims::DimRope;
    const int loc = p.c128_out_loc[token];
    const int page_block = p.c128_page_block;
    constexpr int kDim = ModelDims::DimV, kNope = ModelDims::DimNope;
    const int e0 = vec_tid * 4;
    float total = 0.f;
    CUTLASS_PRAGMA_UNROLL
    for (int slice = 0; slice < C128Layout::Slices; ++slice)
        total += __ldcg(sq_ws + slice);
    const float inv = rms_inv<kDim>(total, p.kv_eps);
    const float4 row4 = __ldcg(reinterpret_cast<const float4*>(row_ws + e0));
    float out[4] = {row4.x, row4.y, row4.z, row4.w};
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < 4; ++j)
        out[j] *= inv * __bfloat162float(norm_weight[e0 + j]);
    if (e0 >= kNope) {
        const float* fr = freqs_pos + (e0 - kNope);
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < 4; j += 2) {
            const float2 r = rope_rotate(out[j], out[j + 1], fr[j], fr[j + 1]);
            out[j] = r.x;
            out[j + 1] = r.y;
        }
    }
    __nv_bfloat16* kv_n = s_row + e0;
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < 4; ++j) kv_n[j] = __float2bfloat16(out[j]);
    bar_sync<KernelBase::Bars::Wg2Sync>();
    uint8_t* page_ptr = p.c128_pool + (int64_t)(loc >> p.c128_page_shift) *
                                          p.c128_stride_kv_block;
    const int kv_off = loc & (page_block - 1);
    pack_kv_row(vec_tid, s_row, page_ptr + kv_off * ModelDims::KvRowBytes,
                page_ptr + (int64_t)page_block * ModelDims::KvRowBytes +
                    kv_off * KvCacheLayout::ScaleSlotBytes);
}

struct OStage {
    static constexpr int EpiCols = AttnConfig::OTmaBox;
    static constexpr int Chunks = AttnConfig::OMmaN / EpiCols;
    static constexpr int RunCols = ModelDims::DimV / 4;
    static constexpr int Vec = 16 / sizeof(cutlass::bfloat16_t);

    CUTE_DEVICE static int col_base(int i) {
        return (i * EpiCols >= RunCols ? ModelDims::DimV / 2 : 0) +
               (i * EpiCols % RunCols);
    }
    template <class SO>
    CUTE_DEVICE static void smem_bases(
        SO& sO, int idx_in_wg, cutlass::bfloat16_t* (&bases)[EpiCols / Vec]) {
        CUTE_UNROLL
        for (int i = 0; i < EpiCols / Vec; ++i)
            bases[i] = &sO(idx_in_wg % ModelDims::Heads,
                           (idx_in_wg / ModelDims::Heads) * RunCols + i * Vec);
    }
    CUTE_DEVICE static void stage_chunk(
        int i, cutlass::bfloat16_t* const (&bases)[EpiCols / Vec], float2 sc) {
        float2 acc[EpiCols / 2];
        __nv_bfloat162 o_bf16[EpiCols / 2];
        tmem_ld_32dp32bNx<EpiCols>(AttnConfig::TmemCols::O + i * EpiCols, acc);
        cutlass::arch::fence_view_async_tmem_load();
        CUTE_UNROLL
        for (int j = 0; j < EpiCols / 2; ++j) {
            acc[j] = mul_rn_f32x2(acc[j], sc);
            o_bf16[j] = __float22bfloat162_rn(acc[j]);
        }
        const int cb = col_base(i);
        CUTE_UNROLL
        for (int j = 0; j < EpiCols / Vec; ++j)
            *(__int128_t*)(bases[j] + cb * ModelDims::Heads) =
                *(__int128_t*)(&o_bf16[j * (Vec / 2)]);
    }
    template <class Atom, class Thr, class TS, class TG>
    CUTE_DEVICE static void store_chunk(Atom const& atom, Thr const& thr_tma,
                                        TS const& tma_sO, TG const& tma_gO,
                                        int i, int wv) {
        constexpr int kHalfBoxes = RunCols / EpiCols;
        const int cb = col_base(i) / EpiCols;
        if (wv == 0 && cute::elect_one_sync())
            cute::copy(atom, thr_tma.partition_S(tma_sO(cute::_, cute::_, cb)),
                       thr_tma.partition_D(tma_gO(cute::_, cute::_, cb)));
        if (wv == 1 && cute::elect_one_sync())
            cute::copy(
                atom,
                thr_tma.partition_S(tma_sO(cute::_, cute::_, cb + kHalfBoxes)),
                thr_tma.partition_D(tma_gO(cute::_, cute::_, cb + kHalfBoxes)));
    }
};

template <class KT>
struct AttentionEpilogue {
    using Dims = typename KT::Dims;
    using CfgQkv = typename KT::Qkv;
    using CfgWqB = typename KT::WqB;
    using CfgKvGate = typename KT::KvGate;
    using Roles = KernelBase::Roles;
    static constexpr int TileTokens = KT::TileT;
    static constexpr int KSplitQkv = CfgQkv::KSplit;
    static constexpr int kRolledKvGateMinTile = 64;

    template <class Op>
    static auto make_o_like_tma(cutlass::bfloat16_t* base, int64_t pages) {
        return cute::make_tma_copy(
            Op{},
            cute::make_tensor(
                cute::make_gmem_ptr(base),
                cute::make_layout(
                    cute::make_shape(Dims::Heads, Dims::DimV, (uint64_t)pages),
                    cute::make_stride(Dims::DimV, cute::_1{},
                                      (int64_t)(Dims::DimV * Dims::Heads)))),
            AttnConfig::SmemLayoutOBuf_TMA{});
    }

    static auto to_underlying_arguments(const MlaAttentionArguments& args,
                                        const MlaAttentionProblem<KT>& prob) {
        const int num_tokens = prob.num_tokens;
        const int slots = split_slots(num_tokens, prob.grid_x);

        auto shape_O =
            cute::make_shape(Dims::Heads, Dims::DimV, (uint64_t)num_tokens);
        auto shape_O_partial =
            cute::make_shape(Dims::Heads, Dims::DimV, (uint64_t)slots);
        auto tma_O =
            make_o_like_tma<cute::SM90_TMA_STORE>(prob.q_ptr, num_tokens);
        auto tma_O_partial =
            make_o_like_tma<cute::SM90_TMA_STORE>(prob.o_partial, slots);
        EpilogueParams<KT, decltype(tma_O)> params{};
        static_cast<CommonParams<KT>&>(params) =
            CommonParams<KT>::from(args, prob);
        params.shape_O = shape_O;
        params.tma_O = tma_O;
        params.shape_O_partial = shape_O_partial;
        params.tma_O_partial = tma_O_partial;
        params.ws_lse_partial =
            reinterpret_cast<float*>(prob.base + prob.wsl.lse_off);
        return params;
    }

    template <class WgSync, int FirstWarp, class Bar>
    CUTE_DEVICE static void arrive_wg(Bar const& bar) {
        bar_sync<WgSync>();
        if (threadIdx.x == FirstWarp * KT::WarpSize) bar.arrive_thread();
    }
    template <class Bar>
    CUTE_DEVICE void arrive_drain_wg(Bar const& bar) const {
        arrive_wg<typename KT::Bars::Wg1Sync, Roles::DrainWarpBegin>(bar);
    }
    template <class Bar>
    CUTE_DEVICE void arrive_vec_wg(Bar const& bar) const {
        arrive_wg<typename KT::Bars::Wg2Sync, Roles::VecWarpBegin>(bar);
    }

    template <class Ctx>
    CUTE_DEVICE void drain(Ctx const& ctx) const {
        {
            const PipelineOffsets led = ctx.sched.ledger_before_qkv();
            auto op =
                make_drain_op<CfgQkv>(ctx.prelude.res_qkv, led.base_acc());
            drain_items(
                op, blockIdx.x, ctx.sched.items_qkv, gridDim.x,
                [&](int item) {
                    const auto it = ctx.prelude.qkv_item(ctx.sched, item);
                    op.store(it.wi, it.out);
                    return it.wi.n_row / TileTokens;
                },
                [&](int item, int group) {
                    if (ctx.sched.qkv_is_q(item))
                        arrive_drain_wg(ctx.bars.qlora_ready(group));
                    else
                        arrive_drain_wg(ctx.bars.kv_ready(group));
                });
        }
        {
            const PipelineOffsets led = ctx.sched.ledger_before_wq_b();
            auto op =
                make_drain_op<CfgWqB>(ctx.prelude.res_wqb, led.base_acc());
            drain_items(
                op, ctx.sched.wqb_first(), ctx.sched.items_wq_b,
                ctx.sched.wqb_step(),
                [&](int item) {
                    const auto it = ctx.prelude.wqb_item(ctx.sched, item);
                    op.store(it.wi, it.out);
                    return it.wi.n_row / TileTokens;
                },
                [&](int, int group) {
                    arrive_drain_wg(ctx.bars.flat_q_ready(group));
                });
        }
        {
            const PipelineOffsets led = ctx.sched.ledger_before_kv_gate();
            auto op = make_bf16_drain_op<CfgKvGate>(ctx.prelude.res_kv_gate,
                                                    led.base_acc());
            drain_items(
                op, ctx.sched.kvg_first(), ctx.sched.items_kv_gate,
                ctx.sched.kvg_step(),
                [&](int item) {
                    const Bf16GemmItem it = bf16_gemm_item_at<CfgKvGate>(
                        item, ctx.sched.n_groups, ctx.sched.num_tokens);
                    if (TileTokens >= kRolledKvGateMinTile &&
                        ctx.sched.kvg_ctas > 0)
                        op.store_rolled(it);
                    else
                        op.store_unrolled(it);
                    return it.n_tile;
                },
                [&](int, int n_tile) {
                    arrive_drain_wg(ctx.bars.ring_ready(n_tile));
                });
        }

        phase_end_sync();
    }

    template <class Ctx>
    CUTE_DEVICE void vec(Ctx const& ctx) const {
        constexpr int kVecTid0 =
            KernelBase::Roles::VecWarpBegin * KernelBase::WarpSize;
        auto pub_x = [&](int group) { arrive_vec_wg(ctx.bars.x_ready(group)); };
        if (!ctx.params.prequant_x) {
            quantize_x<TileTokens>(
                reinterpret_cast<const __nv_bfloat16*>(ctx.params.x_input),
                ctx.params.hidden_row_stride, ctx.params.ws_x_fp8,
                ctx.params.ws_x_sf, sf_layout(ctx.params.act_x),
                ctx.sched.num_tokens, ctx.params.shape_wq_a.k_dim, pub_x);
        }

        {
            const auto& p = ctx.params;
            const int n_tok = ctx.sched.num_tokens;
            const int vec_tid = threadIdx.x - kVecTid0;
            const auto* q_nw =
                reinterpret_cast<const __nv_bfloat16*>(p.q_norm_weight);
            auto const& q_layout = sf_layout(p.act_qlora);

            uint4 q_nw_pre = make_uint4(0, 0, 0, 0);
            if (n_tok > blockIdx.x)
                q_nw_pre = *reinterpret_cast<const uint4*>(
                    q_nw + (vec_tid / QuantX::ThreadsPerGrp) * SfLayout::Gran +
                    (vec_tid % QuantX::ThreadsPerGrp) * QuantX::PerThread);
            auto do_q = [&](int token) {
                const int group = token / TileTokens;
                ctx.bars.qlora_ready(group).wait_warp(0);
                q_reduce_line<TileTokens, Dims::QLoraRank, KSplitQkv>(
                    p.ws_qlora_part, p.ws_qlora_fp8, p.ws_qlora_sf, q_layout,
                    ctx.plan.wg_exchange, token, p.q_eps, q_nw_pre);
                if (threadIdx.x == kVecTid0)
                    ctx.bars.qq_ready(group).arrive_thread();
            };
            auto do_kv = [&](int token) {
                const int group = token / TileTokens;
                ctx.bars.kv_ready(group).wait_warp(0);
                kv_write_line<KSplitQkv>(
                    vec_tid, p.ws_kv_part, token,
                    reinterpret_cast<const __nv_bfloat16*>(p.kv_norm_weight),
                    p.freqs_real + (int64_t)p.positions[token] * Dims::DimRope,
                    reinterpret_cast<uint8_t*>(p.kv), p.stride_kv_block,
                    p.page_block_size, p.swa_loc[token], p.kv_eps,
                    ctx.plan.kv_write_sq, ctx.plan.kv_write_row);
                if (threadIdx.x == kVecTid0)
                    ctx.bars.kvcache_ready(group).arrive_thread();
            };

            for (int i = blockIdx.x; i < 2 * n_tok; i += gridDim.x) {
                if (i < n_tok)
                    do_q(i);
                else
                    do_kv(i - n_tok);
            }
        }
        if (ctx.params.c128_compress()) c128_block(ctx);

        phase_end_sync();
    }

    template <class Ctx>
    CUTE_DEVICE void c128_block(Ctx const& ctx) const {
        constexpr int kVecTid0 =
            KernelBase::Roles::VecWarpBegin * KernelBase::WarpSize;
        const auto& p = ctx.params;
        const int vec_tid = threadIdx.x - kVecTid0;
        const int n_units = ctx.sched.num_tokens * C128Layout::Slices;
        for (int unit = rev_bid(); unit < n_units; unit += gridDim.x) {
            const int token = unit / C128Layout::Slices,
                      slice = unit - token * C128Layout::Slices;
            if ((__ldg(p.positions + token) & (ModelDims::C128Ratio - 1)) !=
                ModelDims::C128Ratio - 1)
                continue;
            char* tok = ctx.params.ws_c128_tok +
                        (int64_t)token * MlaWorkspace::C128TokenBytes;
            float* row_ws = reinterpret_cast<float*>(tok);
            float* sq_ws = row_ws + ModelDims::DimV;
            const int pos_t = __ldg(p.positions + token);
            const int dmax = ctx.params.draft_tokens > 0
                                 ? ctx.params.draft_tokens
                                 : ModelDims::C128Ratio;
            int k = 0;
            while (k + 1 < dmax && token - (k + 1) >= 0 &&
                   __ldg(p.positions + token - (k + 1)) == pos_t - (k + 1))
                ++k;
            const int s0 = ModelDims::C128Ratio - 1 - k;
            c128_slice_partial(vec_tid, slice, p.kv_score_ring,
                               ctx.params.shape_kv_gate.w_out,
                               p.kv_gate_ring_row[token], s0, p.c128_ape,
                               row_ws, sq_ws, ctx.plan.c128_warp_sq, [&] {
                                   for (int group = (token - k) / TileTokens;
                                        group <= token / TileTokens; ++group)
                                       ctx.bars.ring_ready(group).wait_warp(0);
                               });
            int* const flags =
                ctx.params.c128_flags + C128Layout::FlagsPerToken * token;
            if (vec_tid == 0) {
                __threadfence();
                const int prev =
                    gmem_atom_add1_acq_rel(flags + C128Layout::FlagSlicesDone);
                ctx.plan.c128_last_slice = prev == C128Layout::Slices - 1;
            }
            bar_sync<KernelBase::Bars::Wg2Sync>();
            if (ctx.plan.c128_last_slice) {
                c128_finalize_line(vec_tid, token, row_ws, sq_ws, p,
                                   ctx.plan.c128_row);
                bar_sync<KernelBase::Bars::Wg2Sync>();
                if (vec_tid == 0) {
                    __threadfence();
                    gmem_red_add1_release(flags + C128Layout::FlagRowWritten);
                }
            }
            bar_sync<KernelBase::Bars::Wg2Sync>();
        }
    }

    template <class Ctx>
    CUTE_DEVICE void drain_load_scores(
        Ctx const& ctx, int wv,
        float (&scores)[AttnConfig::KvBlock / 2]) const {
        const int tmem_p = AttnConfig::TmemCols::P +
                           ctx.rs.buf_idx * AttnConfig::TmemCols::PStride;
        float p_peer[AttnConfig::KvBlock / 2];
        constexpr int kHalf = AttnConfig::KvBlock / 2;
        if (wv < 2) {
            tmem_ld_32dp32bNx<kHalf>(tmem_p, scores);
            tmem_ld_32dp32bNx<kHalf>(tmem_p + kHalf, p_peer);
        } else {
            tmem_ld_32dp32bNx<kHalf>(tmem_p, p_peer);
            tmem_ld_32dp32bNx<kHalf>(tmem_p + kHalf, scores);
        }
        cutlass::arch::fence_view_async_tmem_load();
        tcgen05_before_thread_sync();

        const int px_slot = wv & 1;
        const bool px_lo = wv < 2;
        auto px_write = [&] {
            CUTE_UNROLL
            for (int i = 0; i < (AttnConfig::KvBlock / 2) / 4; ++i)
                ctx.plan.bufs.kv
                    .p_exchange_buf[px_slot]
                                   [i * KernelBase::WarpSize + ctx.lane_idx] =
                    *(float4*)(p_peer + i * 4);
        };
        auto px_read_add = [&] {
            CUTE_UNROLL
            for (int i = 0; i < (AttnConfig::KvBlock / 2) / 4; ++i) {
                float2 peer[2];
                *(float4*)peer =
                    ctx.plan.bufs.kv
                        .p_exchange_buf[px_slot][i * KernelBase::WarpSize +
                                                 ctx.lane_idx];
                float2* cur_p = (float2*)(scores + i * 4);
                cur_p[0] = add_rn_f32x2(cur_p[0], peer[0]);
                cur_p[1] = add_rn_f32x2(cur_p[1], peer[1]);
            }
        };
        if (px_lo) px_write();
        bar_sync_warp_pair(wv);
        if (!px_lo) px_read_add();
        bar_sync_warp_pair(wv);
        if (!px_lo) px_write();
        bar_sync_warp_pair(wv);
        if (px_lo) px_read_add();
    }

    CUTE_DEVICE static void drain_rescale_o(float scale) {
        constexpr int kChunk = 64;
        const float2 sf = {scale, scale};
        tcgen05_after_thread_sync();
        float2 acc[kChunk / 2];
        CUTE_UNROLL
        for (int ci = 0; ci < (ModelDims::DimV / 2) / kChunk; ++ci) {
            tmem_ld_32dp32bNx<kChunk>(AttnConfig::TmemCols::O + ci * kChunk,
                                      acc);
            cutlass::arch::fence_view_async_tmem_load();
            CUTE_UNROLL
            for (int i = 0; i < kChunk / 2; ++i)
                acc[i] = mul_rn_f32x2(acc[i], sf);
            tmem_st_32dp32bNx<kChunk>(AttnConfig::TmemCols::O + ci * kChunk,
                                      acc);
            cutlass::arch::fence_view_async_tmem_store();
        }
        tcgen05_before_thread_sync();
    }

    struct SoftmaxState {
        float mi;
        float li;
        float real_mi;
    };
    static constexpr int kHalfKeys = AttnConfig::KvBlock / 2;

    template <class Ctx>
    CUTE_DEVICE static void mask_scores(Ctx const& ctx, int block_idx,
                                        float (&scores)[kHalfKeys]) {
        const bool high_keys = ctx.idx_in_warpgroup >= Dims::Heads;
        const int valid_in_half = min(
            max(ctx.kv_tile_valid(block_idx) - (high_keys ? kHalfKeys : 0), 0),
            kHalfKeys);
        CUTE_UNROLL
        for (int i = 0; i < kHalfKeys; ++i)
            if (i >= valid_in_half) scores[i] = -CUDART_INF_F;
    }

    template <class Ctx>
    CUTE_DEVICE static bool update_max(Ctx const& ctx,
                                       const float (&scores)[kHalfKeys],
                                       SoftmaxState& st, float& scale_for_old) {
        float cur_pi_max = -CUDART_INF_F;
        CUTE_UNROLL
        for (int i = 0; i < kHalfKeys; ++i)
            cur_pi_max = max(cur_pi_max, scores[i]);
        cur_pi_max *= ctx.params.sm_scale_div_log2;
        ctx.plan.wg_exchange[ctx.idx_in_warpgroup] = cur_pi_max;
        bar_sync<KernelBase::Bars::Wg1Sync>();
        cur_pi_max =
            max(cur_pi_max,
                ctx.plan.wg_exchange[ctx.idx_in_warpgroup ^ Dims::Heads]);
        st.real_mi = max(st.real_mi, cur_pi_max);
        const bool should_scale_o = __any_sync(
            0xffffffff, cur_pi_max - st.mi > AttnConfig::RescaleThreshold);
        float new_max;
        if (!should_scale_o) {
            scale_for_old = 1.0f;
            new_max = st.mi;
        } else {
            new_max = max(cur_pi_max, st.mi);
            scale_for_old = exp2f(st.mi - new_max);
        }
        st.mi = new_max;
        return should_scale_o;
    }

    template <class Ctx>
    CUTE_DEVICE static void write_probs(Ctx const& ctx, int wv,
                                        const float (&scores)[kHalfKeys],
                                        float scale_for_old, SoftmaxState& st) {
        const float2 scale = {ctx.params.sm_scale_div_log2,
                              ctx.params.sm_scale_div_log2};
        const int sS_off = ctx.lane_idx * 8 + (wv & 1) * (Dims::Heads / 2) * 8 +
                           (wv / 2) * Dims::Heads * kHalfKeys;
        __nv_bfloat162 probs[kHalfKeys / 2];
        float2 neg_new_max = {-st.mi, -st.mi};
        float2 cur_sum = {0.0f, 0.0f};
        CUTE_UNROLL
        for (int i = 0; i < kHalfKeys / 2; ++i) {
            float2 logit = fma_rn_f32x2(
                float2{scores[i * 2], scores[i * 2 + 1]}, scale, neg_new_max);
            logit.x = exp2f(logit.x);
            logit.y = exp2f(logit.y);
            cur_sum = add_rn_f32x2(cur_sum, logit);
            probs[i] = __float22bfloat162_rn(logit);
        }
        st.li = fma(st.li, scale_for_old, (cur_sum.x + cur_sum.y));

        cutlass::bfloat16_t* sS_base =
            ctx.plan.bufs.kv.s[ctx.rs.buf_idx].data() + sS_off;
        CUTE_UNROLL
        for (int i = 0; i < kHalfKeys / 8; ++i)
            *(__int128_t*)(sS_base + Dims::Heads * 8 * i) =
                *(__int128_t*)(probs + i * 4);
    }

    template <class Ctx>
    CUTE_DEVICE static float finalize_softmax(Ctx const& ctx, SoftmaxState& st,
                                              float attn_sink) {
        if (st.real_mi == -CUDART_INF_F) {
            st.li = 0.0f;
            st.mi = -CUDART_INF_F;
        }
        ctx.plan.wg_exchange[ctx.idx_in_warpgroup] = st.li;
        bar_sync<KernelBase::Bars::Wg1Sync>();
        st.li += ctx.plan.wg_exchange[ctx.idx_in_warpgroup ^ Dims::Heads];

        ctx.plan.bar_sv_done[ctx.rs.buf_idx].wait(ctx.rs.bar_phase);
        ctx.rs.update();
        tcgen05_after_thread_sync();

        const bool split_pub = ctx.slot >= 0;
        const float o_scale =
            split_pub
                ? 1.0f
                : (st.li == 0.0f
                       ? 0.0f
                       : __fdividef(1.0f, st.li + exp2f(attn_sink - st.mi)));
        if (split_pub && ctx.idx_in_warpgroup < Dims::Heads) {
            float* lse = ctx.ep.ws_lse_partial +
                         (int64_t)ctx.slot * (Dims::Heads * 2) +
                         ctx.idx_in_warpgroup * 2;
            lse[0] = st.mi;
            lse[1] = st.li;
            __threadfence();
        }
        return o_scale;
    }

    template <class Ctx>
    CUTE_DEVICE static void store_o(Ctx const& ctx, int wv, float o_scale) {
        const float2 o_scale_f2 = {o_scale, o_scale};
        cute::Tensor sO = cute::make_tensor(
            cute::make_smem_ptr(ctx.plan.bufs.qo.o_buf.data()),
            AttnConfig::SmemLayoutOBuf{});
        cutlass::bfloat16_t* sO_bases[OStage::EpiCols / OStage::Vec];
        OStage::smem_bases(sO, ctx.idx_in_warpgroup, sO_bases);
        using OBox =
            cute::Shape<cute::Int<Dims::Heads>, cute::Int<AttnConfig::OTmaBox>>;
        cute::Tensor tma_sO = cute::flat_divide(sO, OBox{})(
            cute::_, cute::_, cute::_0{}, cute::_);

        auto store = [&](auto const& atom, auto const& gO_full) {
            auto thr_tma = atom.get_slice(cute::_0{});
            cute::Tensor tma_gO = cute::flat_divide(gO_full, OBox{})(
                cute::_, cute::_, cute::_0{}, cute::_);
            CUTE_UNROLL
            for (int i = 0; i < OStage::Chunks; ++i) {
                OStage::stage_chunk(i, sO_bases, o_scale_f2);
                cutlass::arch::fence_view_async_shared();
                bar_sync<KernelBase::Bars::Wg1Sync>();
                OStage::store_chunk(atom, thr_tma, tma_sO, tma_gO, i, wv);
            }
        };
        if (ctx.slot >= 0)
            store(ctx.ep.tma_O_partial,
                  ctx.ep.tma_O_partial.get_tma_tensor(ctx.ep.shape_O_partial)(
                      cute::_, cute::_, ctx.slot));
        else
            store(ctx.ep.tma_O, ctx.ep.tma_O.get_tma_tensor(ctx.ep.shape_O)(
                                    cute::_, cute::_, ctx.token_idx));
        cute::tma_store_arrive();
        cute::tma_store_wait<0>();
    }

    template <class Ctx>
    CUTE_DEVICE void attn_drain(Ctx const& ctx) const {
        const int wv = Roles::drain_local_warp(ctx.warp_idx);
        const int head = ctx.idx_in_warpgroup % Dims::Heads;
        const float attn_sink =
            ctx.params.attn_sink == nullptr
                ? -CUDART_INF_F
                : __ldg((float*)ctx.params.attn_sink + head) * CUDART_L2E_F;

        SoftmaxState st{AttnConfig::MaxInit, 0.0f, -CUDART_INF_F};
        CUTE_NO_UNROLL
        for (int block_idx = ctx.blk_lo; block_idx < ctx.blk_hi; ++block_idx) {
            bar_sync<KernelBase::Bars::Wg1Sync>();
            ctx.plan.bar_qk_done[ctx.rs.buf_idx].wait(ctx.rs.bar_phase);
            tcgen05_after_thread_sync();

            float scores[kHalfKeys];
            drain_load_scores(ctx, wv, scores);
            mask_scores(ctx, block_idx, scores);
            float scale_for_old;
            const bool should_scale_o =
                update_max(ctx, scores, st, scale_for_old);
            write_probs(ctx, wv, scores, scale_for_old, st);
            if (block_idx != ctx.blk_lo && should_scale_o) {
                const int prev_buf =
                    (ctx.rs.buf_idx + AttnConfig::NumBufs - 1) %
                    AttnConfig::NumBufs;
                const bool prev_phase =
                    ctx.rs.bar_phase ^ (ctx.rs.buf_idx == 0);
                ctx.plan.bar_sv_done[prev_buf].wait(prev_phase);
                drain_rescale_o(scale_for_old);
            }

            cutlass::arch::fence_view_async_shared();
            ctx.plan.bar_so_ready[ctx.rs.buf_idx].arrive();
            if (block_idx != ctx.blk_hi - 1) ctx.rs.update();
        }
        store_o(ctx, wv, finalize_softmax(ctx, st, attn_sink));
        bar_sync<KernelBase::Bars::Wg1Sync>();
    }

    CUTE_DEVICE static void dequant_row(const cutlass::float_e4m3_t* raw,
                                        uint32_t smem_dst,
                                        uint64_t row_scales) {
        using Kv = KvCacheLayout;
        constexpr int kBlockBytes = (int)sizeof(cutlass::bfloat16_t) *
                                    AttnConfig::KvBlock * AttnConfig::Sw128Cols;
        cutlass::bfloat16_t scales[Kv::ScaleSlotBytes];
        __nv_fp8_e8m0 scales_e8m0[Kv::ScaleSlotBytes];
        *(uint64_t*)scales_e8m0 = row_scales;
        CUTE_UNROLL
        for (int k = 0; k < Kv::ScaleSlotBytes; k += 2)
            *(__nv_bfloat162_raw*)(scales + k) = __nv_cvt_e8m0x2_to_bf162raw(
                *(unsigned short*)(scales_e8m0 + k));

        uint64_t cur_fp8x8 = *(const uint64_t*)raw;
        CUTE_UNROLL
        for (int lc = 0; lc < Kv::NopeGroups; ++lc) {
            __nv_fp8x2_e4m3 data_fp8[4];
            __nv_bfloat162 data_bf16[4];
            *(uint64_t*)data_fp8 = cur_fp8x8;
            if (lc + 1 < Kv::NopeGroups)
                cur_fp8x8 =
                    *(const uint64_t*)(raw + (lc + 1) * Kv::NopeQuantBlock);
            cutlass::bfloat16_t scale = scales[lc];
            CUTE_UNROLL
            for (int i = 0; i < 4; ++i)
                data_bf16[i] = fp8x2_to_bf16x2_with_scale(
                    data_fp8[i], *(__nv_bfloat16*)(&scale));
            smem_st_128b(smem_dst + lc * kBlockBytes, *(__int128_t*)data_bf16);
        }
    }

    template <class Ctx>
    CUTE_DEVICE void attn_vec(Ctx const& ctx) const {
        constexpr int kVec = 8;
        constexpr int kGroupSize = KvCacheLayout::NopeQuantBlock / kVec,
                      NUM_GROUPS = KT::ThreadsPerWarpGroup / kGroupSize,
                      ROWS_PER_GROUP = AttnConfig::KvBlock / NUM_GROUPS;
        int group_idx = ctx.idx_in_warpgroup / kGroupSize,
            idx_in_group = ctx.idx_in_warpgroup % kGroupSize;
        cute::Tensor nope0 = cute::make_tensor(
            cute::make_smem_ptr(ctx.plan.bufs.kv.dequant[0].nope.data()),
            AttnConfig::SmemLayoutKTiles_SW128<Dims::DimNope /
                                               AttnConfig::Sw128Cols>{});
        cutlass::bfloat16_t* nope0_base =
            &nope0(group_idx, idx_in_group * kVec);
        cutlass::bfloat16_t* nope1_base =
            nope0_base + (ctx.plan.bufs.kv.dequant[1].nope.data() -
                          ctx.plan.bufs.kv.dequant[0].nope.data());

        cutlass::float_e4m3_t* raw_nope0_base =
            ctx.plan.bufs.kv.raw_nope[0].data() + group_idx * Dims::DimNope +
            idx_in_group * kVec;
        cutlass::float_e4m3_t* raw_nope1_base =
            raw_nope0_base + (ctx.plan.bufs.kv.raw_nope[1].data() -
                              ctx.plan.bufs.kv.raw_nope[0].data());

        auto load_scales = [&](int blk, uint64_t(&dst)[ROWS_PER_GROUP]) {
            const KvRuns kv_tile = ctx.kv_tile_at(blk);
            const KvSegGeom& seg = ctx.kv_seg_at(blk);
            const bool has_inline_scales =
                seg.stride_kv_block > Dims::KvRowBytes;
            CUTE_UNROLL
            for (int lr = 0; lr < ROWS_PER_GROUP; ++lr)
                dst[lr] = has_inline_scales
                              ? __ldg(kv_seg_scale_ptr(
                                    kv_tile, seg, lr * NUM_GROUPS + group_idx))
                              : Fp8Format::UnitScales;
        };
        uint64_t scales_ahead[ROWS_PER_GROUP] = {};
        int scales_next = ctx.blk_lo;

        auto wait_q = [&] {
            ctx.plan.bar_q_utccp.wait(ctx.q_utccp_phase);
            ctx.q_utccp_phase ^= 1;
        };
        CUTE_NO_UNROLL
        for (int block_idx = ctx.blk_lo; block_idx < ctx.blk_hi; ++block_idx) {
            uint64_t scales_raw[ROWS_PER_GROUP];
            CUTE_NO_UNROLL
            for (; scales_next <= block_idx + 1; ++scales_next) {
                CUTE_UNROLL
                for (int lr = 0; lr < ROWS_PER_GROUP; ++lr)
                    scales_raw[lr] = scales_ahead[lr];
                if (scales_next < ctx.blk_hi)
                    load_scales(scales_next, scales_ahead);
            }
            ctx.plan.bar_raw_ready[ctx.rs.buf_idx].wait(ctx.rs.bar_phase);
            ctx.plan.bar_raw_ready2[ctx.rs.buf_idx].wait(ctx.rs.bar_phase);
            ctx.plan.bar_sv_done[ctx.rs.buf_idx].wait(ctx.rs.bar_phase ^ 1);
            AttnSchedule::wait_q_at(ctx, block_idx, wait_q);
            uint32_t cur_nope_base = cute::cast_smem_ptr_to_uint(
                ctx.rs.buf_idx == 0 ? nope0_base : nope1_base);
            cutlass::float_e4m3_t* raw_nope_base =
                ctx.rs.buf_idx == 0 ? raw_nope0_base : raw_nope1_base;

            CUTE_NO_UNROLL
            for (int lr = 0; lr < ROWS_PER_GROUP; ++lr) {
                const uint64_t row_scales = scales_raw[0];
                CUTE_UNROLL
                for (int k = 0; k + 1 < ROWS_PER_GROUP; ++k)
                    scales_raw[k] = scales_raw[k + 1];
                dequant_row(raw_nope_base + lr * NUM_GROUPS * Dims::DimNope,
                            cur_nope_base + (int)sizeof(cutlass::bfloat16_t) *
                                                lr * NUM_GROUPS *
                                                AttnConfig::Sw128Cols,
                            row_scales);
            }
            cutlass::arch::fence_view_async_shared();
            ctx.plan.bar_nope_ready[ctx.rs.buf_idx].arrive();
            ctx.plan.bar_raw_free[ctx.rs.buf_idx].arrive();
            ctx.rs.update();
        }
        AttnSchedule::wait_q_after(ctx, wait_q);
    }

    template <class Ctx>
    CUTE_DEVICE void token_loop_drain(Ctx const& ctx) const {
        phase1_qnorm_slice(ctx);
        for_each_token(ctx, [&](auto const& tok) {
            if (!tok.token_active) return;
            phase1_prepare_q(ctx, tok);
            attn_drain(tok);
        });
    }

    template <class Ctx>
    CUTE_DEVICE void role_drain(Ctx const& ctx) const {
        drain(ctx);
        token_loop_drain(ctx);
    }

    template <class Ctx>
    CUTE_DEVICE void token_loop_vec(Ctx const& ctx) const {
        for_each_token(ctx, [&](auto const& tok) {
            if (tok.token_active) attn_vec(tok);
            if (ctx.params.qnorm_dist) cutlass::arch::fence_view_async_shared();
        });
    }

    template <class Ctx>
    CUTE_DEVICE void role_vec(Ctx const& ctx) const {
        vec(ctx);
        token_loop_vec(ctx);
    }
};
}  // namespace halops::mla
