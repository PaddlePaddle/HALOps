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

#include <cfloat>
#include <cute/tensor.hpp>

#include "common/sm100_arch.cuh"
#include "kernel/mla/mega_mla/attention/config.cuh"
#include "kernel/mla/mega_mla/mla_utils.cuh"

namespace halops::mla {
/// Quantises the bf16 hidden rows of every token group to fp8 + ue8m0.
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

    constexpr int kBatch = MlaAttnConfig::kXQuantBatch;
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
                        *reinterpret_cast<__nv_bfloat16(*)[QuantX::PerThread]>(
                            &cur),
                        ws_x_fp8, ws_x_sf, layout_sfb, token, hidden_in,
                        k_group, lane);
                }
            }
        }
        if (QuantX::active(blockIdx.x, n_items)) arrive(g_tile);
    }
}

/// Reduces the q-lora split-K partials of one token, RMS-normalises and
/// quantises them.
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
    constexpr int kVecLoads = (int)(QuantX::PerThread * sizeof(PartT)) / kChunk;

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
            CUTE_ALIGNAS(kChunk) PartT part_vals[QuantX::PerThread];
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

/// Packs one normalised kv row into the fp8 nope + bf16 rope cache layout.
CUTE_DEVICE void pack_kv_row(int vec_tid,
                             const __nv_bfloat16* __restrict__ s_row,
                             uint8_t* __restrict__ data_ptr,
                             uint8_t* __restrict__ scale_ptr) {
    using Kv = KvCacheLayout;
    constexpr int kVec = kElemsPerChunk;
    constexpr int kTeam = Kv::NopeQuantBlock / kVec;
    constexpr int kRopeThreads =
        ModelDims::DimRope * sizeof(__nv_bfloat16) / kChunk;
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
        max_abs = fmaxf(max_abs, __shfl_xor_sync(MlaAttnConfig::kFullMask,
                                                 max_abs, offset));
    const float scale_raw =
        fmaxf(max_abs, MlaAttnConfig::kKvAmaxFloor) / Fp8Format::E4m3Max;
    const int bias = Fp8Format::Ue8m0Bias;
    const uint32_t scale_bits = __float_as_uint(scale_raw);
    const int exp_val =
        min((int)Fp8Format::Ue8m0MaxExp,
            (int)((scale_bits >> MlaAttnConfig::kF32MantBits) &
                  MlaAttnConfig::kF32ExpMask) +
                ((scale_bits & MlaAttnConfig::kF32MantMask) != 0));
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
                                  vec_tid * kChunk) =
            *reinterpret_cast<const uint4*>(s_row + ModelDims::DimNope +
                                            vec_tid * kVec);
}

/// Accumulates the split-K partials of this thread's kv dims.
template <int KSplit, class PartT>
CUTE_DEVICE void sum_partials4(const PartT* __restrict__ part, int dim,
                               float (&vals)[MlaAttnConfig::kKvPerThread]) {
    constexpr int kRunBytes =
        (int)(MlaAttnConfig::kKvPerThread * sizeof(PartT));
    CUTLASS_PRAGMA_UNROLL
    for (int split = 0; split < KSplit; ++split) {
        CUTE_ALIGNAS(kChunk) PartT part_vals[MlaAttnConfig::kKvPerThread];
        if constexpr (kRunBytes == kChunk)
            *reinterpret_cast<uint4*>(part_vals) =
                *reinterpret_cast<const uint4*>(part + split * dim);
        else
            *reinterpret_cast<uint2*>(part_vals) =
                *reinterpret_cast<const uint2*>(part + split * dim);
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < MlaAttnConfig::kKvPerThread; ++i)
            vals[i] += (float)part_vals[i];
    }
}

/// Normalises, ropes and writes the kv row of one token into the cache.
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
    static_assert(
        Roles::VecWarps * KernelBase::WarpSize * MlaAttnConfig::kKvPerThread ==
            kDim,
        "4 kv dims per vec thread");

    const int lane = vec_tid & (KernelBase::WarpSize - 1);
    const int wv = vec_tid / KernelBase::WarpSize;

    const int e0 = vec_tid * MlaAttnConfig::kKvPerThread;

    float vals[MlaAttnConfig::kKvPerThread] = {0.0f, 0.0f, 0.0f, 0.0f};
    sum_partials4<KSplit>(ws_kv_part + (int64_t)token * (KSplit * kDim) + e0,
                          kDim, vals);
    CUTE_ALIGNAS(sizeof(uint2)) __nv_bfloat16 vb[MlaAttnConfig::kKvPerThread];
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < MlaAttnConfig::kKvPerThread; ++i)
        vb[i] = __float2bfloat16(vals[i]);

    float sq = 0.0f;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < MlaAttnConfig::kKvPerThread; ++i) {
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
        CUTE_ALIGNAS(sizeof(uint2))
        __nv_bfloat16 nw[MlaAttnConfig::kKvPerThread],
            nb[MlaAttnConfig::kKvPerThread], ob[MlaAttnConfig::kKvPerThread];
        *reinterpret_cast<uint2*>(nw) =
            *reinterpret_cast<const uint2*>(norm_weight + e0);
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < MlaAttnConfig::kKvPerThread; ++i)
            nb[i] = __float2bfloat16(__bfloat162float(vb[i]) * inv *
                                     __bfloat162float(nw[i]));
        if (e0 >= kNope) {
            const float4 fr =
                *reinterpret_cast<const float4*>(freqs_pos + (e0 - kNope));
            const float cs[MlaAttnConfig::kF32PerVec4] = {fr.x, fr.y, fr.z,
                                                          fr.w};
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
            for (int i = 0; i < MlaAttnConfig::kKvPerThread; ++i) ob[i] = nb[i];
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

/// Folds one compressed-kv row into the per-dim online softmax state.
CUTE_DEVICE void c128_online(float (&run_max)[C128Layout::DimsPerLane],
                             float (&es)[C128Layout::DimsPerLane],
                             float (&pr)[C128Layout::DimsPerLane], float4 kv4,
                             float4 sc4, float4 ap4) {
    const float* kva = reinterpret_cast<const float*>(&kv4);
    const float* sca = reinterpret_cast<const float*>(&sc4);
    const float* apa = reinterpret_cast<const float*>(&ap4);
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < C128Layout::DimsPerLane; ++j) {
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

/// Gate-weighted pooling of one c128 slice over the compression window.
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
    constexpr float kLowest = -FLT_MAX;
    float run_max[Lay::DimsPerLane] = {kLowest, kLowest, kLowest, kLowest};
    float es[Lay::DimsPerLane] = {0.f, 0.f, 0.f, 0.f},
          pr[Lay::DimsPerLane] = {0.f, 0.f, 0.f, 0.f};
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
        for (int j = 0; j < Lay::DimsPerLane; ++j) {
            const float mo =
                __shfl_xor_sync(MlaAttnConfig::kFullMask, run_max[j], off);
            const float eo =
                __shfl_xor_sync(MlaAttnConfig::kFullMask, es[j], off);
            const float po =
                __shfl_xor_sync(MlaAttnConfig::kFullMask, pr[j], off);
            const float mm = fmaxf(run_max[j], mo);
            const float keep_w = expf(run_max[j] - mm), peer_w = expf(mo - mm);
            es[j] = es[j] * keep_w + eo * peer_w;
            pr[j] = pr[j] * keep_w + po * peer_w;
            run_max[j] = mm;
        }
    }
    float out[Lay::DimsPerLane], sq = 0.f;
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Lay::DimsPerLane; ++j) {
        out[j] = pr[j] / fmaxf(es[j], MlaAttnConfig::kMinWeightSum);
        sq += out[j] * out[j];
    }
    if (sg == 0)
        __stcg(reinterpret_cast<float4*>(row_ws + e0),
               make_float4(out[0], out[1], out[2], out[Lay::DimsPerLane - 1]));
    sq += __shfl_xor_sync(MlaAttnConfig::kFullMask, sq, 1);
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

/// Normalises, ropes and writes a finished c128 row into its kv pool.
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
    const int e0 = vec_tid * MlaAttnConfig::kKvPerThread;
    float total = 0.f;
    CUTLASS_PRAGMA_UNROLL
    for (int slice = 0; slice < C128Layout::Slices; ++slice)
        total += __ldcg(sq_ws + slice);
    const float inv = rms_inv<kDim>(total, p.kv_eps);
    const float4 row4 = __ldcg(reinterpret_cast<const float4*>(row_ws + e0));
    float out[MlaAttnConfig::kKvPerThread] = {row4.x, row4.y, row4.z, row4.w};
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < MlaAttnConfig::kKvPerThread; ++j)
        out[j] *= inv * __bfloat162float(norm_weight[e0 + j]);
    if (e0 >= kNope) {
        const float* fr = freqs_pos + (e0 - kNope);
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < MlaAttnConfig::kKvPerThread; j += 2) {
            const float2 r = rope_rotate(out[j], out[j + 1], fr[j], fr[j + 1]);
            out[j] = r.x;
            out[j + 1] = r.y;
        }
    }
    __nv_bfloat16* kv_n = s_row + e0;
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < MlaAttnConfig::kKvPerThread; ++j)
        kv_n[j] = __float2bfloat16(out[j]);
    bar_sync<KernelBase::Bars::Wg2Sync>();
    uint8_t* page_ptr = p.c128_pool + (int64_t)(loc >> p.c128_page_shift) *
                                          p.c128_stride_kv_block;
    const int kv_off = loc & (page_block - 1);
    pack_kv_row(vec_tid, s_row, page_ptr + kv_off * ModelDims::KvRowBytes,
                page_ptr + (int64_t)page_block * ModelDims::KvRowBytes +
                    kv_off * KvCacheLayout::ScaleSlotBytes);
}
}  // namespace halops::mla
