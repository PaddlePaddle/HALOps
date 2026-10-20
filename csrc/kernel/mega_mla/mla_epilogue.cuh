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

#include <cute/tensor.hpp>

#include "mla_mainloop.cuh"
#include "mla_params.cuh"

namespace mega_mla {

template <int N_PAD, class LayoutSFB, class Arrive>
__device__ void quantize_x(const __nv_bfloat16* __restrict__ x_input,
                           int x_row_stride, e4m3* __restrict__ ws_x_fp8,
                           ue8m0* __restrict__ ws_x_sf,
                           LayoutSFB const& layout_sfb, int num_tokens,
                           int hidden_in, Arrive&& arrive) {
    using Roles = KernelTrait::Roles;
    constexpr int VEC_THREADS = Roles::VecWarps * KernelTrait::WarpSize;

    constexpr int GRP_PER_PASS = VEC_THREADS / 8;

    const int vec_tid =
        (int)threadIdx.x - Roles::VecWarpBegin * KernelTrait::WarpSize;
    const int grp0 = vec_tid / 8, j = vec_tid & 7;
    const int n_groups = (num_tokens + N_PAD - 1) / N_PAD;
    const int n_grp_k = hidden_in / GemmShapeDefaults::SfGran;

    for (int g_tile = 0; g_tile < n_groups; ++g_tile) {
        const int t = g_tile * N_PAD + (int)blockIdx.x;
        const int nv = min(N_PAD, num_tokens - g_tile * N_PAD);
        if ((int)blockIdx.x < nv) {
            const auto* xin = x_input + (int64_t)t * x_row_stride;
            for (int g = grp0; g < n_grp_k; g += GRP_PER_PASS) {
                __nv_bfloat16 v[4];
                *reinterpret_cast<uint2*>(v) = *reinterpret_cast<const uint2*>(
                    xin + g * GemmShapeDefaults::SfGran + j * 4);
                quantize_group32_from_regs<N_PAD>(
                    v, ws_x_fp8, ws_x_sf, layout_sfb, t, hidden_in, g,
                    vec_tid & (KernelTrait::WarpSize - 1));
            }
            arrive(g_tile);
        }
    }
}

template <int N_PAD, int QLoraRank, int KS, class LayoutSFB, class Wait,
          class Arrive>
__device__ void q_reduce_norm_quant(
    const float* __restrict__ ws_qlora_part,
    const __nv_bfloat16* __restrict__ q_norm_weight,
    e4m3* __restrict__ ws_qlora_fp8, ue8m0* __restrict__ ws_qlora_sf,
    LayoutSFB const& layout_sfb, float* __restrict__ rowwise_max_buf,
    int num_tokens, float q_eps, bool do_quant, Wait&& wait, Arrive&& arrive) {
    using Roles = KernelTrait::Roles;
    using T = KernelTrait;
    constexpr int VEC_THREADS = Roles::VecWarps * T::WarpSize;
    constexpr int N_GRP = QLoraRank / GemmShapeDefaults::SfGran;
    constexpr int GRP_PER_PASS = VEC_THREADS / 8;
    constexpr int PASSES = N_GRP / GRP_PER_PASS;
    static_assert(
        N_GRP % GRP_PER_PASS == 0,
        "every pass must cover a whole number of groups, or a group is "
        "split across two passes and the 8-lane amax shuffle breaks");
    static_assert(
        Roles::VecWarps * PASSES == 8,
        "the slot index `wv + VecWarps*p` reproduces old warps 0..7; if "
        "that product is not 8 the cross-warp sum order changes and this "
        "stage is no longer bit-identical to the 384-thread form");

    const int vec_tid = (int)threadIdx.x - Roles::VecWarpBegin * T::WarpSize;
    const int grp0 = vec_tid / 8, j = vec_tid & 7;
    const int lane = vec_tid & (T::WarpSize - 1);
    const int wv = vec_tid / T::WarpSize;
    const int n_groups = (num_tokens + N_PAD - 1) / N_PAD;

    for (int g_tile = 0; g_tile < n_groups; ++g_tile) {
        const int nv = min(N_PAD, num_tokens - g_tile * N_PAD);
        if ((int)blockIdx.x >= nv) continue;
        wait(g_tile, TraceProbe::WS_PBAR_QLORA);

        const int t = g_tile * N_PAD + (int)blockIdx.x;
        const float* part = ws_qlora_part + (int64_t)t * (KS * QLoraRank);

        float v[PASSES][4];
        float sq[PASSES];
        CUTLASS_PRAGMA_UNROLL
        for (int p = 0; p < PASSES; ++p) {
            const int d0 =
                (grp0 + p * GRP_PER_PASS) * GemmShapeDefaults::SfGran + j * 4;
            v[p][0] = v[p][1] = v[p][2] = v[p][3] = 0.f;
            CUTLASS_PRAGMA_UNROLL
            for (int s = 0; s < KS; ++s) {
                const float4 q4 =
                    *reinterpret_cast<const float4*>(part + s * QLoraRank + d0);
                v[p][0] += q4.x;
                v[p][1] += q4.y;
                v[p][2] += q4.z;
                v[p][3] += q4.w;
            }

            sq[p] = 0.f;
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < 4; ++i) sq[p] += v[p][i] * v[p][i];
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < 4; ++i)
                v[p][i] = __bfloat162float(__float2bfloat16(v[p][i]));
        }

        CUTLASS_PRAGMA_UNROLL
        for (int p = 0; p < PASSES; ++p) {
            CUTLASS_PRAGMA_UNROLL
            for (int o = 16; o > 0; o >>= 1)
                sq[p] += __shfl_xor_sync(0xffffffff, sq[p], o);
            if (lane == 0) rowwise_max_buf[wv + p * Roles::VecWarps] = sq[p];
        }
        bar_sync<KernelTrait::Bars::Wg2Sync>(TraceProbe::WS_WG2_PUBLISH,
                                             g_tile);
        float total_sq = 0.0f;
        CUTLASS_PRAGMA_UNROLL

        for (int slot = 0; slot < Roles::VecWarps * PASSES; ++slot)
            total_sq += rowwise_max_buf[slot];
        const float rms_inv = rsqrtf(total_sq / QLoraRank + q_eps);

        if (do_quant) {
            CUTLASS_PRAGMA_UNROLL
            for (int p = 0; p < PASSES; ++p) {
                const int g = grp0 + p * GRP_PER_PASS;
                const int d0 = g * GemmShapeDefaults::SfGran + j * 4;
                __nv_bfloat16 w4[4];
                *reinterpret_cast<uint2*>(w4) =
                    *reinterpret_cast<const uint2*>(q_norm_weight + d0);
                __nv_bfloat16 q[4];
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < 4; ++i)
                    q[i] = __float2bfloat16(v[p][i] * rms_inv *
                                            __bfloat162float(w4[i]));
                quantize_group32_from_regs<N_PAD>(q, ws_qlora_fp8, ws_qlora_sf,
                                                  layout_sfb, t, QLoraRank, g,
                                                  lane);
            }

            arrive(g_tile);
        }

        bar_sync<KernelTrait::Bars::Wg2Sync>(TraceProbe::WS_WG2_REUSE, g_tile);
    }
}

CUTE_DEVICE void pack_kv_row_576(int vec_tid,
                                 const __nv_bfloat16* __restrict__ s_row,
                                 uint8_t* __restrict__ data_ptr,
                                 uint8_t* __restrict__ scale_ptr) {
    constexpr int NOPE = ModelDims::D_NOPE, QUANT_BLOCK = 64,
                  N_GROUPS = NOPE / QUANT_BLOCK;
    if (vec_tid >= 64) return;
    const int g = vec_tid >> 3, j = vec_tid & 7;
    const int start = min(g, N_GROUPS - 1) * QUANT_BLOCK;
    __nv_bfloat16 v[8];
    *reinterpret_cast<uint4*>(v) =
        *reinterpret_cast<const uint4*>(s_row + start + j * 8);
    float max_abs = 0.f;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 8; ++i)
        max_abs = fmaxf(max_abs, fabsf(__bfloat162float(v[i])));
    CUTLASS_PRAGMA_UNROLL
    for (int o = 1; o < 8; o <<= 1)
        max_abs = fmaxf(max_abs, __shfl_xor_sync(0xffffffffu, max_abs, o));
    const float scale_raw = fmaxf(max_abs, 1e-4f) / 448.0f;
    const int exp_val = max(0, min(255, (int)roundf(log2f(scale_raw)) + 127));
    const float inv = 1.0f / exp2f((float)(exp_val - 127));
    cutlass::float_e4m3_t out[8];
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 8; ++i)
        out[i] = cutlass::float_e4m3_t(__bfloat162float(v[i]) * inv);
    if (g < N_GROUPS) {
        *reinterpret_cast<uint2*>(data_ptr + start + j * 8) =
            *reinterpret_cast<const uint2*>(out);
        if (j == 0) scale_ptr[g] = (uint8_t)exp_val;
    }

    if (vec_tid < 8)
        *reinterpret_cast<uint4*>(data_ptr + NOPE + vec_tid * 16) =
            *reinterpret_cast<const uint4*>(s_row + NOPE + vec_tid * 8);
}

template <int KS, int NUM_THREADS_OLD>
CUTE_DEVICE void kv_write_line(int vec_tid,
                               const float* __restrict__ ws_kv_part, int t,
                               const __nv_bfloat16* __restrict__ norm_weight,
                               const float* __restrict__ freqs_pos,

                               uint8_t* __restrict__ kv_pool,
                               int stride_kv_block, int page_block_size,
                               int loc, float eps, float* __restrict__ sq_buf,
                               __nv_bfloat16* __restrict__ s_row) {
    constexpr int DIM = ModelDims::D_V, NOPE = ModelDims::D_NOPE,
                  ROPE = ModelDims::D_ROPE,
                  ROW_STRIDE = ModelDims::TMA_K_STRIDE;
    using Roles = KernelTrait::Roles;
    using T_ = KernelTrait;
    constexpr int VEC_THREADS = Roles::VecWarps * T_::WarpSize;
    constexpr int OLD_WARPS = NUM_THREADS_OLD / T_::WarpSize;
    constexpr int PASSES = NUM_THREADS_OLD / VEC_THREADS;
    static_assert(
        NUM_THREADS_OLD % VEC_THREADS == 0,
        "each pass must stand in for a whole number of old warps, or the "
        "slot index `wv + VecWarps*p` does not reproduce old warps 0..N");
    static_assert(
        Roles::VecWarps * PASSES == OLD_WARPS,
        "the slot mapping must cover exactly the old warp count, or the "
        "cross-warp sum order changes and this stage stops being "
        "bit-identical");

    const int lane = vec_tid & (T_::WarpSize - 1);
    const int wv = vec_tid / T_::WarpSize;

    static_assert(DIM == VEC_THREADS * 4,
                  "the float4/uint2 blocks below assume each of the 128 vec "
                  "threads owns exactly 4 consecutive elements of the row; "
                  "with any other ratio the widened accesses either overrun "
                  "the row or leave a tail unwritten");
    const int e0 = vec_tid * 4;

    {
        const float* part = ws_kv_part + (int64_t)t * (KS * DIM) + e0;
        float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        CUTLASS_PRAGMA_UNROLL
        for (int s = 0; s < KS; ++s) {
            const float4 p4 = *reinterpret_cast<const float4*>(part + s * DIM);
            v[0] += p4.x;
            v[1] += p4.y;
            v[2] += p4.z;
            v[3] += p4.w;
        }

        CUTE_ALIGNAS(8) __nv_bfloat16 vb[4];
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < 4; ++i) vb[i] = __float2bfloat16(v[i]);
        *reinterpret_cast<uint2*>(s_row + e0) =
            *reinterpret_cast<const uint2*>(vb);
    }
    bar_sync<KernelTrait::Bars::Wg2Sync>();

    CUTLASS_PRAGMA_UNROLL
    for (int p = 0; p < PASSES; ++p) {
        const int old_tid = vec_tid + VEC_THREADS * p;
        float sq = 0.0f;
        const float v0 = __bfloat162float(s_row[old_tid]);
        sq += v0 * v0;
        if (old_tid + NUM_THREADS_OLD < DIM) {
            const float v1 = __bfloat162float(s_row[old_tid + NUM_THREADS_OLD]);
            sq += v1 * v1;
        }
        CUTLASS_PRAGMA_UNROLL
        for (int o = 16; o > 0; o >>= 1)
            sq += __shfl_xor_sync(0xffffffffu, sq, o);
        if (lane == 0) sq_buf[wv + Roles::VecWarps * p] = sq;
    }
    bar_sync<KernelTrait::Bars::Wg2Sync>();
    if (vec_tid == 0) {
        float total = 0.0f;
        for (int slot = 0; slot < OLD_WARPS; ++slot) total += sq_buf[slot];
        sq_buf[0] = rsqrtf(total / DIM + eps);
    }
    bar_sync<KernelTrait::Bars::Wg2Sync>();
    const float rms_inv = sq_buf[0];

    {
        CUTE_ALIGNAS(8) __nv_bfloat16 sv[4], nw[4], ob[4];
        *reinterpret_cast<uint2*>(sv) =
            *reinterpret_cast<const uint2*>(s_row + e0);
        *reinterpret_cast<uint2*>(nw) =
            *reinterpret_cast<const uint2*>(norm_weight + e0);
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < 4; ++i)
            ob[i] = __float2bfloat16(__bfloat162float(sv[i]) * rms_inv *
                                     __bfloat162float(nw[i]));
        *reinterpret_cast<uint2*>(s_row + e0) =
            *reinterpret_cast<const uint2*>(ob);
    }
    bar_sync<KernelTrait::Bars::Wg2Sync>();

    for (int pi = vec_tid; pi < ROPE / 2; pi += VEC_THREADS) {
        const int d = NOPE + pi * 2;
        const float c = freqs_pos[pi * 2], s_v = freqs_pos[pi * 2 + 1];
        const float v0 = __bfloat162float(s_row[d]);
        const float v1 = __bfloat162float(s_row[d + 1]);
        s_row[d] = __float2bfloat16(v0 * c - v1 * s_v);
        s_row[d + 1] = __float2bfloat16(v0 * s_v + v1 * c);
    }
    bar_sync<KernelTrait::Bars::Wg2Sync>();

    const int kv_page = loc / page_block_size,
              kv_offset = loc % page_block_size;
    uint8_t* page_ptr = kv_pool + (int64_t)kv_page * stride_kv_block;
    pack_kv_row_576(
        vec_tid, s_row, page_ptr + kv_offset * ROW_STRIDE,
        page_ptr + (int64_t)page_block_size * ROW_STRIDE + kv_offset * 8);
    bar_sync<KernelTrait::Bars::Wg2Sync>();
}

#ifndef MEGA_MLA_C128_GPP
#    define MEGA_MLA_C128_GPP 2
#endif

CUTE_DEVICE void c128_compress_line(
    int vec_tid, int t, const float* __restrict__ ring, int kv_gate_k,

    int ring_row, const float* __restrict__ ape,
    const __nv_bfloat16* __restrict__ norm_weight,
    const float* __restrict__ freqs_pos, float* __restrict__ row_out,

    uint8_t* __restrict__ pool, int stride_kv_block, int page_block,
    int page_shift, int loc, float eps, float* __restrict__ s_stats,
    __nv_bfloat16* __restrict__ s_row) {
    constexpr int DIM = ModelDims::D_V, NOPE = ModelDims::D_NOPE;
    constexpr float NEG_INF = -3.402823466e38f;

    const int wv = vec_tid / 32, l = vec_tid & 31;
    const int e0 = vec_tid * 4;
    const int page_start = ring_row - (ModelDims::C128_RATIO - 1);

    constexpr int GPP = MEGA_MLA_C128_GPP;
    static_assert(
        ModelDims::C128_RATIO % (2 * GPP) == 0,
        "the ModelDims::C128_RATIO ring slots must split into whole double-"
        "buffered steps of 2*GPP, or the tail is dropped");
    float m[4] = {NEG_INF, NEG_INF, NEG_INF, NEG_INF};
    float es[4] = {0.f, 0.f, 0.f, 0.f};
    float pr[4] = {0.f, 0.f, 0.f, 0.f};
    float4 kvA[GPP], scA[GPP], apA[GPP];
    float4 kvB[GPP], scB[GPP], apB[GPP];

    auto issue = [&](auto& kv4, auto& sc4, auto& ap4, int s0) {
        CUTLASS_PRAGMA_UNROLL
        for (int u = 0; u < GPP; ++u) {
            const float* row =
                ring + (int64_t)(page_start + s0 + u) * kv_gate_k;
            kv4[u] = *reinterpret_cast<const float4*>(row + e0);
            sc4[u] = *reinterpret_cast<const float4*>(row + DIM + e0);
            ap4[u] = *reinterpret_cast<const float4*>(
                ape + (int64_t)(s0 + u) * DIM + e0);
        }
    };
    auto consume = [&](const auto& kv4, const auto& sc4, const auto& ap4) {
        CUTLASS_PRAGMA_UNROLL
        for (int u = 0; u < GPP; ++u) {
            const float* kva = reinterpret_cast<const float*>(&kv4[u]);
            const float* sca = reinterpret_cast<const float*>(&sc4[u]);
            const float* apa = reinterpret_cast<const float*>(&ap4[u]);
            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < 4; ++j) {
                const float sc = sca[j] + apa[j];

                const float d = sc - m[j];
                const float p = expf(-fabsf(d));
                const float rs = (d > 0.f) ? p : 1.f;
                const float e = (d > 0.f) ? 1.f : p;
                es[j] = es[j] * rs + e;
                pr[j] = pr[j] * rs + kva[j] * e;
                m[j] = fmaxf(m[j], sc);
            }
        }
    };
    issue(kvA, scA, apA, 0);
    CUTLASS_PRAGMA_NO_UNROLL
    for (int s0 = 0; s0 < ModelDims::C128_RATIO; s0 += 2 * GPP) {
        issue(kvB, scB, apB, s0 + GPP);
        consume(kvA, scA, apA);
        if (s0 + 2 * GPP < ModelDims::C128_RATIO)
            issue(kvA, scA, apA, s0 + 2 * GPP);
        consume(kvB, scB, apB);
    }
    float out[4];
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < 4; ++j) out[j] = pr[j] / fmaxf(es[j], 1e-38f);

    if (row_out)
        *reinterpret_cast<float4*>(row_out + (int64_t)t * DIM + e0) =
            make_float4(out[0], out[1], out[2], out[3]);

    float sq = 0.f;
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < 4; ++j) sq += out[j] * out[j];
    CUTLASS_PRAGMA_UNROLL
    for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
    if (l == 0) s_stats[wv] = sq;
    bar_sync<KernelTrait::Bars::Wg2Sync>();
    if (vec_tid == 0)
        s_stats[4] = rsqrtf(
            (s_stats[0] + s_stats[1] + s_stats[2] + s_stats[3]) / DIM + eps);
    bar_sync<KernelTrait::Bars::Wg2Sync>();
    const float rms_inv = s_stats[4];
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < 4; ++j)
        out[j] *= rms_inv * __bfloat162float(norm_weight[e0 + j]);
    if (e0 >= NOPE) {
        const float* fr = freqs_pos + ((e0 - NOPE) / 2) * 2;
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < 4; j += 2) {
            const float c = fr[j], s_v = fr[j + 1];
            const float v0 = out[j], v1 = out[j + 1];
            out[j] = v0 * c - v1 * s_v;
            out[j + 1] = v0 * s_v + v1 * c;
        }
    }

    __nv_bfloat16* kv_n = s_row + e0;
    kv_n[0] = __float2bfloat16(out[0]);
    kv_n[1] = __float2bfloat16(out[1]);
    kv_n[2] = __float2bfloat16(out[2]);
    kv_n[3] = __float2bfloat16(out[3]);
    bar_sync<KernelTrait::Bars::Wg2Sync>();

    uint8_t* page_ptr = pool + (int64_t)(loc >> page_shift) * stride_kv_block;
    const int kv_off = loc & (page_block - 1);
    pack_kv_row_576(
        vec_tid, s_row, page_ptr + kv_off * ModelDims::TMA_K_STRIDE,
        page_ptr + (int64_t)page_block * ModelDims::TMA_K_STRIDE + kv_off * 8);
}

template <class KT>
struct CollectiveEpilogue {
    using Dims = typename KT::Dims;
    using CfgQkv = typename KT::Qkv;
    using CfgWqB = typename KT::WqB;
    using CfgKvGate = typename KT::KvGate;
    using CfgWoA = typename KT::WoA;
    using CfgWoB = typename KT::WoB;
    using Roles = KernelTrait::Roles;
    static constexpr int BN = CfgQkv::BN;
    static constexpr int N_PAD = KT::TileT;
    static constexpr int KS_QKV = CfgQkv::KSplit;
    static constexpr int C128_D_ROPE = ModelDims::D_ROPE;
    static constexpr int KV_WRITE_OLD_THREADS = KernelTrait::NumThreads;

    template <class Op>
    static auto make_o_like_tma(bf16* base, int64_t pages) {
        using D = ModelDims;
        return cute::make_tma_copy(
            Op{},
            cute::make_tensor(
                cute::make_gmem_ptr(base),
                cute::make_layout(
                    cute::make_shape(D::B_H, D::D_V, (uint64_t)pages),
                    cute::make_stride(D::D_V, cute::_1{},
                                      (int64_t)(D::D_V * D::B_H)))),
            AttnConfig::SmemLayoutOBuf_TMA{});
    }

    static auto to_underlying_arguments(
        const halops::MegaMlaArguments& args,
        const halops::MegaMlaProblem<KT>& prob) {
        using D = ModelDims;
        using WS = halops::MegaMlaWorkspace<KT>;
        using CfgA = typename KT::Qkv;
        using CfgOA = typename KT::WoA;
        using CfgOB = typename KT::WoB;
        constexpr int TileT = KT::TileT;
        const int T = prob.num_tokens;

        auto shape_O = cute::make_shape(D::B_H, D::D_V, (uint64_t)T);
        auto shape_O_partial =
            cute::make_shape(D::B_H, D::D_V, (uint64_t)prob.grid_x);
        auto tma_O = make_o_like_tma<cute::SM90_TMA_STORE>(prob.q_ptr, T);
        auto tma_O_partial =
            make_o_like_tma<cute::SM90_TMA_STORE>(prob.o_partial, prob.grid_x);
        halops::EpilogueParams<KT, decltype(tma_O)> p{};
        static_cast<halops::CommonParams<KT>&>(p) =
            halops::CommonParams<KT>::from(args, prob);
        p.shape_wo_a =
            mega_mla::make_gemm_shape<CfgOA::BM, CfgOA::BK, CfgOA::KSplit,
                                      CfgOA::TmaStages>(
                prob.o_groups * prob.o_lora_rank, prob.d_per_group);
        p.shape_wo_b = prob.ks_wo_b == 4
                           ? mega_mla::make_gemm_shape<CfgOB::BM, CfgOB::BK, 4,
                                                       CfgOB::TmaStages>(
                                 prob.hidden, prob.olora_total)
                       : prob.ks_wo_b == 2
                           ? mega_mla::make_gemm_shape<CfgOB::BM, CfgOB::BK, 2,
                                                       CfgOB::TmaStages>(
                                 prob.hidden, prob.olora_total)
                           : mega_mla::make_gemm_shape<CfgOB::BM, CfgOB::BK, 1,
                                                       CfgOB::TmaStages>(
                                 prob.hidden, prob.olora_total);
        p.opdims = prob.opdims;
        p.final_out = reinterpret_cast<mega_mla::bf16*>(args.out.data_ptr());
        p.opost_flat_max =
            std::max(0, mega_mla::env_int("MEGA_MLA_OPOST_MAX", 64));
        p.shape_O = shape_O;
        p.tma_O = tma_O;
        p.shape_O_partial = shape_O_partial;
        p.tma_O_partial = tma_O_partial;
        p.ws_o_partial = prob.o_partial;
        p.ws_lse_partial =
            reinterpret_cast<float*>(prob.base + prob.wsg.kv_split_off +
                                     (int64_t)prob.grid_x * D::O_TOTAL * 2);
        p.gemm_wo_a = mega_mla::make_weight_atoms<TileT>(
            args.wo_a_weight.data_ptr(), args.wo_a_sf_g32.data_ptr(),
            p.shape_wo_a.m_total, p.shape_wo_a.k_dim);
        p.gemm_wo_b = mega_mla::make_weight_atoms<TileT>(
            args.wo_b_weight.data_ptr(), args.wo_b_sf_g32.data_ptr(),
            p.shape_wo_b.m_total, p.shape_wo_b.k_dim);
        p.ws_flat_o_fp8 = prob.ws_flat_o_fp8;
        p.ws_flat_o_sf = prob.ws_flat_o_sf;
        p.ws_olora_bf16 = prob.ws_olora_bf16;
        p.ws_olora_fp8 = prob.ws_olora_fp8;
        p.ws_olora_sf = prob.ws_olora_sf;
        p.act_flat_o = mega_mla::make_act_atoms<TileT>(
            p.ws_flat_o_fp8, p.ws_flat_o_sf, p.shape_wo_a.m_total, prob.t_pad,
            WS::FLAT_O_K, prob.t_pad_sf);
        p.act_olora = mega_mla::make_act_atoms<TileT>(
            p.ws_olora_fp8, p.ws_olora_sf, p.shape_wo_b.m_total, prob.t_pad,
            prob.wsd.olora_k, prob.t_pad_sf);
        return p;
    }

    template <class Ctx>
    CUTE_DEVICE void drain(Ctx const& c) const {
        {
            const RangeBarrierLedger led = c.sched.ledger_before_kv_gate();
            auto op =
                make_bf16_drain_op<CfgKvGate>(c.r.res_kv_gate, led.base_acc());
            for (int item = (int)blockIdx.x; item < c.sched.gemm.items_kv_gate;
                 item += gridDim.x) {
                const Bf16GemmItem it = bf16_gemm_item_at<CfgKvGate>(
                    item, c.sched.gemm.n_groups, c.sched.num_tokens());
                op.wait_mma_and_load(TraceProbe::WS_KVG_ACC_FULL, item);
                op.store(it);
                op.end_item();
                c.bars.ring_ready(it.n_tile).arrive_warp();
            }
        }
        {
            const RangeBarrierLedger led = c.sched.ledger_before_qkv();
            auto op = make_drain_op<CfgQkv>(c.r.res_qkv, led.base_acc());
            for (int item = (int)blockIdx.x; item < c.sched.gemm.items_qkv;
                 item += gridDim.x) {
                const QkvItem<BN> it = c.r.qkv_item(c.sched, item);
                op.wait_mma_and_load(TraceProbe::WS_QKV_ACC_FULL, item);
                op.store(it.wi, it.out);
                op.end_item();
                if (item < c.sched.gemm.items_wq_a)
                    c.bars.qlora_ready(it.wi.n_row / N_PAD).arrive_warp();
                else
                    c.bars.kv_ready(it.wi.n_row / N_PAD).arrive_warp();
            }
        }
        {
            const RangeBarrierLedger led = c.sched.ledger_before_wq_b();
            auto op = make_drain_op<CfgWqB>(c.r.res_wqb, led.base_acc());
            for (int item = (int)blockIdx.x; item < c.sched.gemm.items_wq_b;
                 item += gridDim.x) {
                const GemmWorkItem wi = c.sched.wqb_item(item);
                op.wait_mma_and_load(TraceProbe::WS_WQB_ACC_FULL, item);
                op.store(wi, c.r.wqb_out(c.sched, item));
                op.end_item();
                c.bars.flat_q_ready(wi.n_row / N_PAD).arrive_warp();
            }
        }

        __syncwarp();
        mega_mla::bar_sync<mega_mla::KernelTrait::Bars::EveryoneSync>();
        cutlass::arch::fence_barrier_init();
        mega_mla::bar_sync<mega_mla::KernelTrait::Bars::EveryoneSync>();
    }

    template <class Ctx>
    CUTE_DEVICE void vec(Ctx const& c) const {
        auto pub_x = [&](int g) { c.bars.x_ready(g).arrive_warp(); };
        auto wait_qlora = [&](int g, int site) {
            c.bars.qlora_ready(g).wait_warp(0, site, g);
        };
        auto pub_qq = [&](int g) { c.bars.qq_ready(g).arrive_warp(); };
        if (c.params.use_phase1) {
            quantize_x<N_PAD>(
                reinterpret_cast<const __nv_bfloat16*>(c.params.x_input),
                Dims::Q_TOTAL, c.params.ws_x_fp8, c.params.ws_x_sf,
                sf_layout(c.params.act_x), c.sched.num_tokens(),
                c.params.shape_wq_a.k_dim, pub_x);
            q_reduce_norm_quant<N_PAD, Dims::QLORA_RANK, KS_QKV>(
                c.params.ws_qlora_part,
                reinterpret_cast<const __nv_bfloat16*>(c.params.q_norm_weight),
                c.params.ws_qlora_fp8, c.params.ws_qlora_sf,
                sf_layout(c.params.act_qlora), c.plan.rowwise_max_buf,
                c.sched.num_tokens(), c.params.q_eps, true, wait_qlora, pub_qq);
        }
        if (c.params.use_kv_write) kv_write_block(c);
        if (c.params.use_kv_gate && c.params.use_c128) c128_block(c);

        __syncwarp();
        mega_mla::bar_sync<mega_mla::KernelTrait::Bars::EveryoneSync>();
        cutlass::arch::fence_barrier_init();
        mega_mla::bar_sync<mega_mla::KernelTrait::Bars::EveryoneSync>();
    }

    template <class Ctx>
    CUTE_DEVICE void c128_block(Ctx const& c) const {
        constexpr int VEC_TID0 =
            KernelTrait::Roles::VecWarpBegin * KernelTrait::WarpSize;
        static_assert(KernelTrait::Roles::VecWarps == 4,
                      "c128_compress_line's element tiling and its 4-partial "
                      "RMS reduction "
                      "are written for exactly the four vec warps");
        const mega_mla::C128Inputs c128{
            .ring = c.params.kv_score_ring,
            .ring_row = c.params.kv_gate_ring_row,
            .positions = c.params.positions,
            .ape = c.params.c128_ape,
            .norm_weight = reinterpret_cast<const __nv_bfloat16*>(
                c.params.c128_norm_weight),
            .freqs_real = c.params.c128_freqs_real,
            .out_loc = c.params.c128_out_loc,
            .row_out = c.params.c128_row_out,
            .pool = c.params.c128_pool,
            .stride_kv_block = c.params.c128_stride_kv_block,
            .page_block = c.params.c128_page_block,
            .page_shift = c.params.c128_page_shift,
            .eps = c.params.kv_eps,
            .stats = c.plan.c128_stats,
            .row = c.plan.c128_row};
        for (int t = (int)blockIdx.x; t < c.sched.num_tokens();
             t += gridDim.x) {
            if ((c128.positions[t] & (ModelDims::C128_RATIO - 1)) ==
                ModelDims::C128_RATIO - 1) {
                c.bars.ring_ready(t / N_PAD).wait_warp(
                    0, TraceProbe::WS_PBAR_RING, t / N_PAD);
                c128_compress_line(
                    (int)threadIdx.x - VEC_TID0, t, c128.ring,
                    c.params.shape_kv_gate.m_total, c128.ring_row[t], c128.ape,
                    c128.norm_weight,
                    c128.freqs_real + (int64_t)c128.positions[t] * C128_D_ROPE,
                    c128.row_out, c128.pool, c128.stride_kv_block,
                    c128.page_block, c128.page_shift, c128.out_loc[t], c128.eps,
                    c128.stats, c128.row);
            }

            c.bars.c128_ready(t / N_PAD).arrive_warp();
        }
    }

    template <class Ctx>
    CUTE_DEVICE void kv_write_block(Ctx const& c) const {
        constexpr int VEC_TID0 =
            KernelTrait::Roles::VecWarpBegin * KernelTrait::WarpSize;
        const mega_mla::KvWriteInputs kv_write{
            .ws_kv_part = c.params.ws_kv_part,
            .norm_weight =
                reinterpret_cast<const __nv_bfloat16*>(c.params.kv_norm_weight),
            .freqs_real = c.params.freqs_real,
            .positions = c.params.positions,
            .loc = c.params.swa_loc,
            .pool = reinterpret_cast<uint8_t*>(c.params.kv),
            .stride_kv_block = c.params.stride_kv_block,
            .page_block_size = c.params.page_block_size,
            .eps = c.params.kv_eps,
            .sq_buf = c.plan.kv_write_sq,
            .row = c.plan.kv_write_row};
        for (int t = (int)blockIdx.x; t < c.sched.num_tokens();
             t += gridDim.x) {
            c.bars.kv_ready(t / N_PAD).wait_warp(0, TraceProbe::WS_PBAR_KV,
                                                 t / N_PAD);
            kv_write_line<KS_QKV, KV_WRITE_OLD_THREADS>(
                (int)threadIdx.x - VEC_TID0, kv_write.ws_kv_part, t,
                kv_write.norm_weight,
                kv_write.freqs_real +
                    (int64_t)kv_write.positions[t] * C128_D_ROPE,
                kv_write.pool, kv_write.stride_kv_block,
                kv_write.page_block_size, kv_write.loc[t], kv_write.eps,
                kv_write.sq_buf, kv_write.row);

            c.bars.kvcache_ready(t / N_PAD).arrive_warp();
        }
    }

    template <class Ctx>
    CUTE_DEVICE void drain_wo_a(Ctx const& c) const {
        const RangeBarrierLedger led = c.sched.ledger_before_wo_a();
        auto op = make_drain_op<CfgWoA>(c.ro.res_wo_a, led.base_acc());
        for (int item = (int)blockIdx.x; item < c.sched.out_proj.items_wo_a;
             item += gridDim.x) {
            op.wait_mma_and_load(TraceProbe::WS_WOA_ACC_FULL, item);
            op.store(c.sched.wo_a_item(item), c.ro.wo_a_out(c.sched, item));
            op.end_item();

            c.bars.olora_ready(c.sched.wo_a_round(item), c.sched.wo_a_tg(item))
                .arrive_warp();
        }
    }
    template <int KS, class Ctx>
    CUTE_DEVICE void drain_wo_b(Ctx const& c) const {
        const RangeBarrierLedger led = c.sched.ledger_before_wo_b();
        auto op = make_drain_op<WithKSplit<CfgWoB, KS>>(c.ro.res_wo_b,
                                                        led.base_acc());
        for (int item = (int)blockIdx.x; item < c.sched.out_proj.items_wo_b;
             item += gridDim.x) {
            op.wait_mma_and_load(TraceProbe::WS_WOB_ACC_FULL, item);

            if constexpr (KS > 1)
                c.bars.flatq_free(c.sched.wo_b_tg(item))
                    .wait_warp(0, TraceProbe::WS_PBAR_FLATQ,
                               c.sched.wo_b_tg(item));
            op.store(c.sched.template wo_b_item<KS>(item),
                     c.ro.template wo_b_out<KS>(c.sched, item));
            op.end_item();

            if constexpr (KS > 1)
                c.bars.wob_ready(c.sched.wo_b_tg(item)).arrive_warp();
        }
    }

    template <class Ctx>
    CUTE_DEVICE void vec_wo_a(Ctx const& c) const {
        constexpr int KS = CfgWoA::KSplit;
        const int OL = c.sched.d.m_total_woa;
        constexpr int N_PAD = KT::TileT;
        constexpr int GRP_PER_PASS = Roles::VecWarps * 32 / 4;

        constexpr int GPP = KT::WoA::OloraGroupsPerPass;
        const int ROUNDS = c.sched.d.olora_rounds;

        const int tv = (int)threadIdx.x - Roles::VecWarpBegin * 32;
        const int gl = tv >> 2;
        const int j = tv & 3;

        for (int g_tile = 0; g_tile < c.sched.out_proj.n_groups; ++g_tile) {
            const int nv =
                min((int)N_PAD, c.sched.out_proj.num_tokens - g_tile * N_PAD);

            if ((int)blockIdx.x >= nv) continue;
            const int t = g_tile * N_PAD + (int)blockIdx.x;
            const auto* part = c.ro.out_olora + (int64_t)t * (KS * OL);

            uint4 raw[GPP][KS];
            auto issue_round = [&](int rnd) {
                CUTLASS_PRAGMA_UNROLL
                for (int u = 0; u < GPP; ++u) {
                    const int d0 = ((rnd * GPP + u) * GRP_PER_PASS + gl) *
                                       GemmShapeDefaults::SfGran +
                                   j * 8;
                    CUTLASS_PRAGMA_UNROLL
                    for (int s = 0; s < KS; ++s)
                        raw[u][s] =
                            *reinterpret_cast<const uint4*>(part + s * OL + d0);
                }
            };
            c.bars.olora_ready(0, g_tile).wait_warp(
                0, TraceProbe::WS_PBAR_OLORA, 0);
            issue_round(0);

            CUTLASS_PRAGMA_NO_UNROLL
            for (int rnd = 0; rnd < ROUNDS; ++rnd) {
                bool nxt_ready = true;
                if (rnd + 1 < ROUNDS)
                    nxt_ready =
                        c.bars.olora_ready(rnd + 1, g_tile).try_wait_warp(0);
                CUTLASS_PRAGMA_UNROLL
                for (int u = 0; u < GPP; ++u) {
                    const int g = (rnd * GPP + u) * GRP_PER_PASS + gl;
                    __nv_bfloat16 q[8];
                    olora_sum_partials<KS>(raw[u], q);
                    quantize_group32_from_regs8<N_PAD>(
                        q, c.ep.ws_olora_fp8, c.ep.ws_olora_sf,
                        sf_layout(c.ep.act_olora), t, OL, g,
                        (int)(threadIdx.x & 31));
                }

                if (rnd + 1 < ROUNDS) {
                    if (!nxt_ready)
                        c.bars.olora_ready(rnd + 1, g_tile)
                            .wait_warp(0, TraceProbe::WS_PBAR_OLORA, rnd + 1);
                    issue_round(rnd + 1);
                }
            }

            c.bars.oq_ready(g_tile).arrive_warp();
        }
    }

    template <int KS, class Ctx>
    CUTE_DEVICE void vec_wo_b(Ctx const& c) const {
        if constexpr (KS == 1) {
            return;
        } else {
            const int H = c.sched.d.m_total_wob;
            constexpr int N_PAD = KT::TileT;
            constexpr int VEC = 8;
            constexpr int LANES = Roles::VecWarps * 32;
            const int UNITS = c.sched.d.wob_units;
            const int tv = (int)threadIdx.x - Roles::VecWarpBegin * 32;
            const int total_units = c.sched.out_proj.num_tokens * UNITS;
            int waited_g = -1;
            for (int u = (int)blockIdx.x; u < total_units;
                 u += (int)gridDim.x) {
                const int t = u / UNITS;
                const int unit = u - t * UNITS;
                const int g_tile = t / N_PAD;
                if (g_tile != waited_g) {
                    c.bars.wob_ready(g_tile).wait_warp(
                        0, TraceProbe::WS_PBAR_WOB, g_tile);
                    waited_g = g_tile;
                }
                const int d0 = (unit * LANES + tv) * VEC;

                const auto* part =
                    reinterpret_cast<const bf16*>(c.ro.out_wo_b) +
                    (int64_t)t * (KS * H) + d0;
                uint4 raw[KS];
                CUTLASS_PRAGMA_UNROLL
                for (int s = 0; s < KS; ++s)
                    raw[s] = *reinterpret_cast<const uint4*>(part + s * H);
                bf16 q[VEC];

                float acc[VEC] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
                CUTLASS_PRAGMA_UNROLL
                for (int s = 0; s < KS; ++s) {
                    const auto* pv = reinterpret_cast<const bf16*>(&raw[s]);
                    CUTLASS_PRAGMA_UNROLL
                    for (int i = 0; i < VEC; ++i) acc[i] += float(pv[i]);
                }
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < VEC; ++i) q[i] = bf16(acc[i]);
                *reinterpret_cast<uint4*>(c.ro.final_out + (int64_t)t * H +
                                          d0) =
                    *reinterpret_cast<const uint4*>(q);
            }
        }
    }

    template <class Ctx>
    CUTE_DEVICE void attn_drain(Ctx const& c) const {
        const int wv = Roles::drain_local_warp(c.warp_idx);

        constexpr int B_EPI = 64;
        const float2 scale = {c.params.sm_scale_div_log2,
                              c.params.sm_scale_div_log2};
        const int sS_off = c.lane_idx * 8 +
                           (wv & 1) * (ModelDims::B_H / 2) * 8 +
                           (wv / 2) * ModelDims::B_H * (AttnConfig::B_TOPK / 2);
        float attn_sink = c.params.attn_sink == nullptr
                              ? -CUDART_INF_F
                              : __ldg((float*)c.params.attn_sink +
                                      (c.idx_in_warpgroup % 64)) *
                                    CUDART_L2E_F;

        c.plan.bar_last_store_done.arrive();

        float mi = AttnConfig::MAX_INIT_VAL;
        float li = 0.0f;
        float real_mi = -CUDART_INF_F;

        CUTE_NO_UNROLL
        for (int block_idx = c.blk_lo; block_idx < c.blk_hi; ++block_idx) {
            mega_mla::bar_sync<mega_mla::KernelTrait::Bars::Wg1Sync>();
            c.plan.bar_qk_done[c.rs.buf_idx].wait(c.rs.bar_phase);
            ku::tcgen05_after_thread_sync();

            const int tmem_p = AttnConfig::tmem_cols::P +
                               c.rs.buf_idx * AttnConfig::tmem_cols::P_STRIDE;
            float p[AttnConfig::B_TOPK / 2], p_peer[AttnConfig::B_TOPK / 2];
            if (wv < 2) {
                ku::tmem_ld_32dp32bNx<AttnConfig::B_TOPK / 2>(tmem_p, p);
                ku::tmem_ld_32dp32bNx<AttnConfig::B_TOPK / 2>(tmem_p + 32,
                                                              p_peer);
            } else {
                ku::tmem_ld_32dp32bNx<AttnConfig::B_TOPK / 2>(tmem_p, p_peer);
                ku::tmem_ld_32dp32bNx<AttnConfig::B_TOPK / 2>(tmem_p + 32, p);
            }
            cutlass::arch::fence_view_async_tmem_load();
            ku::tcgen05_before_thread_sync();

            CUTE_UNROLL
            for (int i = 0; i < (AttnConfig::B_TOPK / 2) / 4; ++i)
                c.plan.u.kv.p_exchange_buf[wv ^ 2][i * 32 + c.lane_idx] =
                    *(float4*)(p_peer + i * 4);
            mega_mla::bar_sync_warp_pair(wv);
            CUTE_UNROLL
            for (int i = 0; i < (AttnConfig::B_TOPK / 2) / 4; ++i) {
                float2 t[2];
                *(float4*)t =
                    c.plan.u.kv.p_exchange_buf[wv][i * 32 + c.lane_idx];
                float2* cur_p = (float2*)(p + i * 4);
                cur_p[0] = ku::float2_add(cur_p[0], t[0]);
                cur_p[1] = ku::float2_add(cur_p[1], t[1]);
            }

            const int valid_in_half = min(
                max(c.kv_tile_valid(block_idx) -
                        (c.idx_in_warpgroup >= 64 ? AttnConfig::B_TOPK / 2 : 0),
                    0),
                AttnConfig::B_TOPK / 2);
            CUTE_UNROLL
            for (int i = 0; i < AttnConfig::B_TOPK / 2; ++i)
                if (i >= valid_in_half) p[i] = -CUDART_INF_F;

            float cur_pi_max = -CUDART_INF_F;
            CUTE_UNROLL
            for (int i = 0; i < AttnConfig::B_TOPK / 2; ++i)
                cur_pi_max = max(cur_pi_max, p[i]);
            cur_pi_max *= c.params.sm_scale_div_log2;
            c.plan.rowwise_max_buf[c.idx_in_warpgroup] = cur_pi_max;
            mega_mla::bar_sync<mega_mla::KernelTrait::Bars::Wg1Sync>();
            cur_pi_max = max(cur_pi_max,
                             c.plan.rowwise_max_buf[c.idx_in_warpgroup ^ 64]);
            real_mi = max(real_mi, cur_pi_max);
            bool should_scale_o =
                __any_sync(0xffffffff, cur_pi_max - mi > 6.0f);

            float new_max, scale_for_old;
            if (!should_scale_o) {
                scale_for_old = 1.0f;
                new_max = mi;
            } else {
                new_max = max(cur_pi_max, mi);
                scale_for_old = exp2f(mi - new_max);
            }
            mi = new_max;

            __nv_bfloat162 s[(AttnConfig::B_TOPK / 2) / 2];
            float2 neg_new_max = {-new_max, -new_max};
            float2 cur_sum = {0.0f, 0.0f};
            CUTE_UNROLL
            for (int i = 0; i < (AttnConfig::B_TOPK / 2) / 2; ++i) {
                float2 d = ku::float2_fma(float2{p[i * 2], p[i * 2 + 1]}, scale,
                                          neg_new_max);
                d.x = exp2f(d.x);
                d.y = exp2f(d.y);
                cur_sum = ku::float2_add(cur_sum, d);
                s[i] = __float22bfloat162_rn(d);
            }
            li = fma(li, scale_for_old, (cur_sum.x + cur_sum.y));

            bf16* sS_base = c.plan.u.kv.s[c.rs.buf_idx].data() + sS_off;
            CUTE_UNROLL
            for (int i = 0; i < AttnConfig::B_TOPK / 2 / 8; ++i)
                *(uint128_t*)(sS_base + ModelDims::B_H * 8 * i) =
                    *(uint128_t*)(s + i * 4);

            if (block_idx != c.blk_lo && should_scale_o) {
                float2 sf = {scale_for_old, scale_for_old};
                ku::tcgen05_after_thread_sync();
                static constexpr int CHUNK = 64;
                float2 o[CHUNK / 2];
                CUTE_UNROLL
                for (int ci = 0; ci < (ModelDims::D_V / 2) / CHUNK; ++ci) {
                    ku::tmem_ld_32dp32bNx<CHUNK>(
                        AttnConfig::tmem_cols::O + ci * CHUNK, o);
                    cutlass::arch::fence_view_async_tmem_load();
                    CUTE_UNROLL
                    for (int i = 0; i < CHUNK / 2; ++i)
                        o[i] = ku::float2_mul(o[i], sf);
                    ku::tmem_st_32dp32bNx<CHUNK>(
                        AttnConfig::tmem_cols::O + ci * CHUNK, o);
                    cutlass::arch::fence_view_async_tmem_store();
                }
                ku::tcgen05_before_thread_sync();
            }

            cutlass::arch::fence_view_async_shared();
            c.plan.bar_so_ready[c.rs.buf_idx].arrive();
            if (block_idx != c.blk_hi - 1) c.rs.update();
        }

        if (real_mi == -CUDART_INF_F) {
            li = 0.0f;
            mi = -CUDART_INF_F;
        }

        c.plan.rowwise_max_buf[c.idx_in_warpgroup] = li;
        mega_mla::bar_sync<mega_mla::KernelTrait::Bars::Wg1Sync>();
        li += c.plan.rowwise_max_buf[c.idx_in_warpgroup ^ 64];

        c.plan.bar_sv_done[c.rs.buf_idx].wait(c.rs.bar_phase);
        c.rs.update();
        ku::tcgen05_after_thread_sync();

        c.plan.bar_last_store_done.arrive();

        const bool split_pub = c.slot >= 0;
        float o_scale =
            split_pub
                ? 1.0f
                : (li == 0.0f ? 0.0f
                              : __fdividef(1.0f, li + exp2f(attn_sink - mi)));
        if (split_pub && c.idx_in_warpgroup < ModelDims::B_H) {
            float* lse = c.ep.ws_lse_partial +
                         (int64_t)c.slot * (ModelDims::B_H * 2) +
                         c.idx_in_warpgroup * 2;
            lse[0] = mi;
            lse[1] = li;
            __threadfence();
        }
        float2 o_scale_f2 = {o_scale, o_scale};

        Tensor sO = make_tensor(make_smem_ptr(c.plan.u.qo.o.o_buf.data()),
                                AttnConfig::SmemLayoutOBuf{});
        bf16* sO_bases[B_EPI / 8];
        CUTE_UNROLL
        for (int i = 0; i < B_EPI / 8; ++i)
            sO_bases[i] = &sO(c.idx_in_warpgroup % 64,
                              (c.idx_in_warpgroup / 64) * 128 + i * 8);

        Tensor tma_sO = flat_divide(sO, Shape<Int<ModelDims::B_H>, Int<64>>{})(
            _, _, _0{}, _);

        auto store_o = [&](auto const& atom, auto const& gO_full) {
            auto thr_tma = atom.get_slice(_0{});
            Tensor tma_gO = flat_divide(
                gO_full, Shape<Int<ModelDims::B_H>, Int<64>>{})(_, _, _0{}, _);
            CUTE_UNROLL
            for (int i = 0; i < (ModelDims::D_V / 2) / B_EPI; ++i) {
                float2 o[B_EPI / 2];
                __nv_bfloat162 o_bf16[B_EPI / 2];
                ku::tmem_ld_32dp32bNx<B_EPI>(
                    AttnConfig::tmem_cols::O + i * B_EPI, o);
                cutlass::arch::fence_view_async_tmem_load();
                CUTE_UNROLL
                for (int j = 0; j < B_EPI / 2; ++j) {
                    o[j] = ku::float2_mul(o[j], o_scale_f2);
                    o_bf16[j] = __float22bfloat162_rn(o[j]);
                }
                int col_base =
                    (i * B_EPI >= ModelDims::D_V / 4 ? ModelDims::D_V / 2 : 0) +
                    (i * B_EPI % (ModelDims::D_V / 4));
                CUTE_UNROLL
                for (int j = 0; j < B_EPI / 8; ++j)
                    *(__int128_t*)(sO_bases[j] + col_base * ModelDims::B_H) =
                        *(__int128_t*)(&o_bf16[j * 4]);
                cutlass::arch::fence_view_async_shared();
                mega_mla::bar_sync<mega_mla::KernelTrait::Bars::Wg1Sync>();

                if (wv == 0 && elect_one_sync()) {
                    cute::copy(
                        atom, thr_tma.partition_S(tma_sO(_, _, col_base / 64)),
                        thr_tma.partition_D(tma_gO(_, _, col_base / 64)));
                }
                if (wv == 1 && elect_one_sync()) {
                    cute::copy(
                        atom,
                        thr_tma.partition_S(tma_sO(
                            _, _, col_base / 64 + (ModelDims::D_V / 4) / 64)),
                        thr_tma.partition_D(tma_gO(
                            _, _, col_base / 64 + (ModelDims::D_V / 4) / 64)));
                }
            }
        };
        if (split_pub)
            store_o(c.ep.tma_O_partial,
                    c.ep.tma_O_partial.get_tma_tensor(c.ep.shape_O_partial)(
                        _, _, c.slot));
        else
            store_o(c.ep.tma_O,
                    c.ep.tma_O.get_tma_tensor(c.ep.shape_O)(_, _, c.token_idx));
        cute::tma_store_arrive();
        cute::tma_store_wait<0>();

        mega_mla::bar_sync<mega_mla::KernelTrait::Bars::Wg1Sync>();
    }

    template <class Ctx>
    CUTE_DEVICE void attn_vec(Ctx const& c) const {
        constexpr int GROUP_SIZE = 8, NUM_GROUPS = 128 / 8,
                      ROWS_PER_GROUP = AttnConfig::B_TOPK / NUM_GROUPS,
                      COLS_PER_GROUP = ModelDims::D_NOPE / (GROUP_SIZE * 8);
        int group_idx = c.idx_in_warpgroup / GROUP_SIZE,
            idx_in_group = c.idx_in_warpgroup % GROUP_SIZE;
        Tensor nope0 = make_tensor(
            make_smem_ptr(c.plan.u.kv.dequant[0].nope.data()),
            AttnConfig::SmemLayoutKTiles_SW128<ModelDims::D_NOPE / 64>{});
        bf16* nope0_base = &nope0(group_idx, idx_in_group * 8);
        bf16* nope1_base = nope0_base + (c.plan.u.kv.dequant[1].nope.data() -
                                         c.plan.u.kv.dequant[0].nope.data());

        e4m3* raw_nope0_base = c.plan.u.kv.raw_nope[0].data() +
                               group_idx * ModelDims::D_NOPE + idx_in_group * 8;
        e4m3* raw_nope1_base =
            raw_nope0_base +
            (c.plan.u.kv.raw_nope[1].data() - c.plan.u.kv.raw_nope[0].data());

        c.plan.bar_q_utccp.wait(c.q_utccp_phase);
        c.q_utccp_phase ^= 1;
        CUTE_NO_UNROLL
        for (int block_idx = c.blk_lo; block_idx < c.blk_hi; ++block_idx) {
            uint64_t scales_raw[ROWS_PER_GROUP];
            const mega_mla::KvRuns kv_tile = c.kv_tile_at(block_idx);
            const mega_mla::KvSegGeom& seg = c.kv_seg_at(block_idx);

            const bool has_inline_scales =
                seg.stride_kv_block > ModelDims::TMA_K_STRIDE;
            CUTE_UNROLL
            for (int lr = 0; lr < ROWS_PER_GROUP; ++lr)
                scales_raw[lr] =
                    has_inline_scales
                        ? __ldg(mega_mla::kv_seg_scale_ptr(
                              kv_tile, seg, lr * NUM_GROUPS + group_idx))
                        : 0x7F7F7F7F7F7F7F7Full;
            c.plan.bar_raw_ready[c.rs.buf_idx].wait(c.rs.bar_phase);

            c.plan.bar_raw_ready2[c.rs.buf_idx].wait(c.rs.bar_phase);
            c.plan.bar_sv_done[c.rs.buf_idx].wait(c.rs.bar_phase ^ 1);
            uint32_t cur_nope_base = cute::cast_smem_ptr_to_uint(
                c.rs.buf_idx == 0 ? nope0_base : nope1_base);
            e4m3* raw_nope_base =
                c.rs.buf_idx == 0 ? raw_nope0_base : raw_nope1_base;

            auto st_128b = [&](int lr, int lc, __int128_t& data) {
                ku::smem_st_128b(
                    cur_nope_base + 2 * (lr * NUM_GROUPS * 64 +
                                         lc * AttnConfig::B_TOPK * 64),
                    data);
            };
            auto get_raw_fp8 = [&](int lr, int lc) -> uint64_t {
                return *(uint64_t*)(raw_nope_base +
                                    lr * NUM_GROUPS * ModelDims::D_NOPE +
                                    lc * (GROUP_SIZE * 8));
            };

            CUTE_UNROLL
            for (int lr = 0; lr < ROWS_PER_GROUP; ++lr) {
                bf16 scales[8];
                e8m0 scales_e8m0[8];
                *(uint64_t*)scales_e8m0 = scales_raw[lr];
                *(__nv_bfloat162_raw*)(scales + 0) =
                    __nv_cvt_e8m0x2_to_bf162raw(
                        *(unsigned short*)(scales_e8m0 + 0));
                *(__nv_bfloat162_raw*)(scales + 2) =
                    __nv_cvt_e8m0x2_to_bf162raw(
                        *(unsigned short*)(scales_e8m0 + 2));
                *(__nv_bfloat162_raw*)(scales + 4) =
                    __nv_cvt_e8m0x2_to_bf162raw(
                        *(unsigned short*)(scales_e8m0 + 4));
                *(__nv_bfloat162_raw*)(scales + 6) =
                    __nv_cvt_e8m0x2_to_bf162raw(
                        *(unsigned short*)(scales_e8m0 + 6));

                uint64_t cur_fp8x8 = get_raw_fp8(lr, 0);
                CUTE_UNROLL
                for (int lc = 0; lc < COLS_PER_GROUP; ++lc) {
                    ku::nve4m3x2 data_fp8[4];
                    ku::nvbf16x2 data_bf16[4];
                    *(uint64_t*)data_fp8 = cur_fp8x8;
                    if (lc + 1 < COLS_PER_GROUP)
                        cur_fp8x8 = get_raw_fp8(lr, lc + 1);
                    bf16 scale = scales[lc];
                    CUTE_UNROLL
                    for (int i = 0; i < 4; ++i)
                        data_bf16[i] = sm100::fp8x2_to_bf16x2_with_scale(
                            data_fp8[i], *(ku::nvbf16*)(&scale));
                    st_128b(lr, lc, *(__int128_t*)data_bf16);
                }
            }
            cutlass::arch::fence_view_async_shared();
            c.plan.bar_nope_ready[c.rs.buf_idx].arrive();
            c.plan.bar_raw_free[c.rs.buf_idx].arrive();
            c.rs.update();
        }
    }

    template <int BATCH, class Ctx>
    CUTE_DEVICE void merge_o_batch(Ctx const& c, __nv_bfloat16 (&v)[BATCH][8],
                                   int base, int stride, int n_items,
                                   int t0) const {
        using RolesV = mega_mla::KernelTrait::Roles;
        constexpr int N_GRP =
            ModelDims::O_TOTAL / mega_mla::GemmShapeDefaults::SfGran;
        constexpr int GRP_PER_ROW =
            ModelDims::D_V / mega_mla::GemmShapeDefaults::SfGran;
        const int j = ((int)threadIdx.x - RolesV::VecWarpBegin * 32) & 3;
        const float* w_all = c.ep.ws_lse_partial;
        const auto* p_all =
            reinterpret_cast<const __nv_bfloat16*>(c.ep.ws_o_partial);
        const auto* o_all = reinterpret_cast<const __nv_bfloat16*>(c.params.q);
        const int slot_step = c.sched.slot_step;
        const int w_step = slot_step * (ModelDims::B_H * 2);
        const int p_step = slot_step * (ModelDims::B_H * ModelDims::D_V);

        int w_off[BATCH], p_off[BATCH], se[BATCH];
        float acc[BATCH][8];
        int se_max = 0;
        CUTLASS_PRAGMA_UNROLL
        for (int u = 0; u < BATCH; ++u) {
            se[u] = 1;
            w_off[u] = 0;
            p_off[u] = 0;
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < 8; ++i) acc[u][i] = 0.0f;
            const int item = base + u * stride;
            if (item >= n_items) continue;
            const int g = item & (N_GRP - 1);
            const int gd =
                (g % GRP_PER_ROW) * mega_mla::GemmShapeDefaults::SfGran + j * 8;
            const int h = g / GRP_PER_ROW;
            const int tt = t0 + item / N_GRP;
            const int2 sp = c.plan.o_split[item / N_GRP];
            if (sp.x > 1) {
                se[u] = sp.x;
                w_off[u] = sp.y * (ModelDims::B_H * 2) + h * 2;
                p_off[u] = sp.y * (ModelDims::B_H * ModelDims::D_V) +
                           h * ModelDims::D_V + gd;
                se_max = max(se_max, sp.x);
            } else {
                *reinterpret_cast<uint4*>(v[u]) =
                    *reinterpret_cast<const uint4*>(
                        o_all + (int64_t)tt * ModelDims::O_TOTAL +
                        h * ModelDims::D_V + gd);
            }
        }
        for (int s = 0; s < se_max; ++s) {
            float w[BATCH];
            uint4 p[BATCH];
            CUTLASS_PRAGMA_UNROLL
            for (int u = 0; u < BATCH; ++u) {
                const int ss = min(s, se[u] - 1);
                w[u] = s < se[u] ? w_all[w_off[u] + ss * w_step] : 0.0f;
                p[u] = *reinterpret_cast<const uint4*>(p_all + p_off[u] +
                                                       ss * p_step);
            }
            CUTLASS_PRAGMA_UNROLL
            for (int u = 0; u < BATCH; ++u) {
                if (se[u] <= 1 || w[u] == 0.0f) continue;
                const auto* q = reinterpret_cast<const __nv_bfloat16*>(&p[u]);
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < 8; ++i)
                    acc[u][i] = fmaf(w[u], __bfloat162float(q[i]), acc[u][i]);
            }
        }
        CUTLASS_PRAGMA_UNROLL
        for (int u = 0; u < BATCH; ++u) {
            if (se[u] <= 1) continue;
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < 8; ++i) v[u][i] = __float2bfloat16(acc[u][i]);
        }
    }

    template <class Ctx>
    CUTE_DEVICE void merge_coefs(Ctx const& c, int t0, int nv) const {
        using RolesV = mega_mla::KernelTrait::Roles;
        constexpr int VEC_THREADS = RolesV::VecWarps * 32;
        const int tv = (int)threadIdx.x - RolesV::VecWarpBegin * 32;
        const int slot_step = c.sched.slot_step;
        for (int pair = (int)blockIdx.x * VEC_THREADS + tv;
             pair < nv * ModelDims::B_H; pair += (int)gridDim.x * VEC_THREADS) {
            const int tt = t0 + pair / ModelDims::B_H,
                      hh = pair % ModelDims::B_H;
            const int2 sp = c.plan.o_split[tt - t0];
            const int se = sp.x;
            if (se <= 1) continue;
            float* base = c.ep.ws_lse_partial + hh * 2;
            auto slot = [&](int s) {
                return base + (sp.y + s * slot_step) * (ModelDims::B_H * 2);
            };
            float m = -CUDART_INF_F;
            for (int s = 0; s < se; ++s) {
                const float2 e = *reinterpret_cast<const float2*>(slot(s));
                if (e.y != 0.0f) m = max(m, e.x);
            }
            float l = 0.0f;
            for (int s = 0; s < se; ++s) {
                const float2 e = *reinterpret_cast<const float2*>(slot(s));
                if (e.y != 0.0f) l = fmaf(exp2f(e.x - m), e.y, l);
            }
            if (m != -CUDART_INF_F && c.params.attn_sink != nullptr)
                l += exp2f(__ldg(c.params.attn_sink + hh) * CUDART_L2E_F - m);
            const float inv =
                (l == 0.0f || m == -CUDART_INF_F) ? 0.0f : __fdividef(1.0f, l);
            for (int s = 0; s < se; ++s) {
                float* p = slot(s);
                const float2 e = *reinterpret_cast<const float2*>(p);
                *p = e.y != 0.0f ? exp2f(e.x - m) * inv : 0.0f;
            }
        }
    }

    template <class Ctx>
    CUTE_DEVICE void flat_o_post(Ctx const& c, int n_groups_flat) const {
        constexpr int N_GRP =
            ModelDims::O_TOTAL / mega_mla::GemmShapeDefaults::SfGran;
        constexpr int GRP_PER_ROW =
            ModelDims::D_V / mega_mla::GemmShapeDefaults::SfGran;
        using RolesV = mega_mla::KernelTrait::Roles;
        constexpr int GRP_PER_PASS = RolesV::VecWarps * 32 / 4;
        constexpr int BATCH = MEGA_MLA_OPOST_BATCH;

        constexpr int N_PAD_V = Ctx::tile_t;
        const auto* o_all = reinterpret_cast<const __nv_bfloat16*>(c.params.q);

        const int tv = (int)threadIdx.x - RolesV::VecWarpBegin * 32;
        const int grp0 = tv / 4, j = tv & 3;
        const int stride = (int)gridDim.x * GRP_PER_PASS;
        const int start = (int)blockIdx.x * GRP_PER_PASS + grp0;
        for (int g_tile = 0; g_tile < n_groups_flat; ++g_tile) {
            const int t0 = g_tile * N_PAD_V;
            const int nv = min((int)N_PAD_V, c.params.num_tokens - t0);
            const int n_items = nv * N_GRP;

            if (c.warp_idx == RolesV::VecWarpBegin)
                c.bars.o_stored(g_tile).wait_warp(
                    0, mega_mla::TraceProbe::WS_PBAR_O_STORED, g_tile);
            mega_mla::bar_sync<mega_mla::KernelTrait::Bars::Wg2Sync>();
            auto run_items = [&](auto SPLIT_C) {
                constexpr bool SPLIT = decltype(SPLIT_C)::value;
                if constexpr (SPLIT) {
                    if (tv < nv) {
                        const mega_mla::TokenSplit ts =
                            c.params.wplan.tok[t0 + tv];
                        c.plan.o_split[tv] = {ts.nseg, ts.slot0};
                    }
                    mega_mla::bar_sync<mega_mla::KernelTrait::Bars::Wg2Sync>();
                    merge_coefs(c, t0, nv);
                    __threadfence();
                    __syncwarp();
                    c.bars.o_coef(g_tile).arrive_warp();
                    if (c.warp_idx == RolesV::VecWarpBegin)
                        c.bars.o_coef(g_tile).wait_warp(
                            0, mega_mla::TraceProbe::WS_PBAR_O_COEF, g_tile);
                    mega_mla::bar_sync<mega_mla::KernelTrait::Bars::Wg2Sync>();
                }
                CUTLASS_PRAGMA_NO_UNROLL
                for (int base = start; base < n_items; base += stride * BATCH) {
                    __nv_bfloat16 v[BATCH][8];
                    if constexpr (SPLIT) {
                        merge_o_batch<BATCH>(c, v, base, stride, n_items, t0);
                    } else {
                        CUTLASS_PRAGMA_UNROLL
                        for (int u = 0; u < BATCH; ++u) {
                            const int item = base + u * stride;
                            if (item >= n_items) continue;
                            const int g = item & (N_GRP - 1);
                            const int gd =
                                (g % GRP_PER_ROW) *
                                    mega_mla::GemmShapeDefaults::SfGran +
                                j * 8;
                            const int tt = t0 + item / N_GRP;
                            *reinterpret_cast<uint4*>(v[u]) =
                                *reinterpret_cast<const uint4*>(
                                    o_all + (int64_t)tt * ModelDims::O_TOTAL +
                                    (g / GRP_PER_ROW) * ModelDims::D_V + gd);
                        }
                    }
                    CUTLASS_PRAGMA_UNROLL
                    for (int u = 0; u < BATCH; ++u) {
                        const int item = base + u * stride;
                        if (item >= n_items) continue;
                        const int t = t0 + item / N_GRP;
                        const int g = item & (N_GRP - 1);
                        const int gd = (g % GRP_PER_ROW) *
                                           mega_mla::GemmShapeDefaults::SfGran +
                                       j * 8;
                        if (gd >= ModelDims::D_NOPE) {
                            const float* fr =
                                c.params.freqs_real +
                                c.params.positions[t] * ModelDims::D_ROPE;
                            CUTLASS_PRAGMA_UNROLL
                            for (int k = 0; k < 4; ++k) {
                                const int pi = (gd - ModelDims::D_NOPE) / 2 + k;
                                const float cs = fr[pi * 2],
                                            s_val = fr[pi * 2 + 1];
                                const float v0 = __bfloat162float(v[u][k * 2]);
                                const float v1 =
                                    __bfloat162float(v[u][k * 2 + 1]);
                                v[u][k * 2] =
                                    __float2bfloat16(v0 * cs + v1 * s_val);
                                v[u][k * 2 + 1] =
                                    __float2bfloat16(-v0 * s_val + v1 * cs);
                            }
                        }
                        mega_mla::quantize_group32_from_regs8<N_PAD_V>(
                            v[u], c.ep.ws_flat_o_fp8, c.ep.ws_flat_o_sf,
                            sf_layout(c.ep.act_flat_o), t, ModelDims::O_TOTAL,
                            g, (int)(threadIdx.x & 31));
                    }
                }
            };
            if (c.sched.any_split)
                run_items(cute::true_type{});
            else
                run_items(cute::false_type{});

            c.bars.flat_o(g_tile).arrive_warp();
        }
    }

    template <class Ctx>
    CUTE_DEVICE void token_loop_drain(Ctx const& c) const {
        for (int token_it = 0; token_it < c.token_iters; ++token_it) {
            const PerToken<Ctx> t = make_per_token(c, token_it);

            if (t.token_active) {
                phase1_prepare_q(c, t);

                attn_drain(t);
            }

            __syncwarp();
            mega_mla::bar_sync<mega_mla::KernelTrait::Bars::EveryoneSync>();
            if (token_it + 1 == c.token_iters) {
                __syncwarp();
                mega_mla::bar_sync<mega_mla::KernelTrait::Bars::EveryoneSync>();
            }
        }
    }

    template <class Ctx>
    CUTE_DEVICE void out_proj_drain(Ctx const& c) const {
        drain_wo_a(c);
        __threadfence();
        auto run_wo_b = [&](auto KS_C) {
            constexpr int KS = decltype(KS_C)::value;
            drain_wo_b<KS>(c);
            __threadfence();
        };
        if (c.sched.ks_wo_b == 4)
            run_wo_b(cute::Int<4>{});
        else if (c.sched.ks_wo_b == 2)
            run_wo_b(cute::Int<2>{});
        else
            run_wo_b(cute::Int<1>{});
    }

    template <class Ctx>
    CUTE_DEVICE void role_drain(Ctx const& c) const {
        drain(c);
        token_loop_drain(c);
        out_proj_drain(c);
    }

    template <class Ctx>
    CUTE_DEVICE void token_loop_vec(Ctx const& c) const {
        for (int token_it = 0; token_it < c.token_iters; ++token_it) {
            const PerToken<Ctx> t = make_per_token(c, token_it);

            if (t.token_active) {
                attn_vec(t);
            }

            __syncwarp();
            mega_mla::bar_sync<mega_mla::KernelTrait::Bars::EveryoneSync>();
            if (!c.sched.o_post_flat && t.token_active &&
                Roles::is_vec(c.warp_idx)) {
                const auto* o_gmem =
                    reinterpret_cast<const __nv_bfloat16*>(c.params.q) +
                    (int64_t)t.token_idx * (ModelDims::B_H * ModelDims::D_V);
                const float* fr =
                    c.params.freqs_real +
                    c.params.positions[t.token_idx] * ModelDims::D_ROPE;
                constexpr int N_GRP =
                    ModelDims::O_TOTAL / mega_mla::GemmShapeDefaults::SfGran;
                constexpr int GRP_PER_ROW =
                    ModelDims::D_V / mega_mla::GemmShapeDefaults::SfGran;
                constexpr int GRP_PER_PASS = Roles::VecWarps * 32 / 4;
                constexpr int BATCH = MEGA_MLA_OPOST_BATCH;
                static_assert(
                    N_GRP % (GRP_PER_PASS * BATCH) == 0,
                    "MEGA_MLA_OPOST_BATCH must divide the pass count");

                const int vec_tid = (int)threadIdx.x - Roles::VecWarpBegin * 32;
                const int grp0 = vec_tid / 4, j = vec_tid & 3;
                __nv_bfloat16 v[BATCH][8];
                CUTLASS_PRAGMA_NO_UNROLL
                for (int g0 = grp0; g0 < N_GRP; g0 += GRP_PER_PASS * BATCH) {
                    CUTLASS_PRAGMA_UNROLL
                    for (int u = 0; u < BATCH; ++u) {
                        const int g = g0 + u * GRP_PER_PASS;
                        const int gd = (g % GRP_PER_ROW) *
                                           mega_mla::GemmShapeDefaults::SfGran +
                                       j * 8;
                        *reinterpret_cast<uint4*>(v[u]) =
                            *reinterpret_cast<const uint4*>(
                                o_gmem + (g / GRP_PER_ROW) * ModelDims::D_V +
                                gd);
                    }
                    CUTLASS_PRAGMA_UNROLL
                    for (int u = 0; u < BATCH; ++u) {
                        const int g = g0 + u * GRP_PER_PASS;
                        const int gd = (g % GRP_PER_ROW) *
                                           mega_mla::GemmShapeDefaults::SfGran +
                                       j * 8;
                        if (gd >= ModelDims::D_NOPE) {
                            CUTLASS_PRAGMA_UNROLL
                            for (int k = 0; k < 4; ++k) {
                                const int pi = (gd - ModelDims::D_NOPE) / 2 + k;
                                const float cs = fr[pi * 2],
                                            s_val = fr[pi * 2 + 1];
                                const float v0 = __bfloat162float(v[u][k * 2]);
                                const float v1 =
                                    __bfloat162float(v[u][k * 2 + 1]);
                                v[u][k * 2] =
                                    __float2bfloat16(v0 * cs + v1 * s_val);
                                v[u][k * 2 + 1] =
                                    __float2bfloat16(-v0 * s_val + v1 * cs);
                            }
                        }
                        mega_mla::quantize_group32_from_regs8<Ctx::tile_t>(
                            v[u], c.ep.ws_flat_o_fp8, c.ep.ws_flat_o_sf,
                            sf_layout(c.ep.act_flat_o), t.token_idx,
                            ModelDims::O_TOTAL, g, (int)(threadIdx.x & 31));
                    }
                }

                c.bars.flat_o(t.token_idx / c.tile_t).arrive_warp();
            }
            if (token_it + 1 == c.token_iters) {
                __syncwarp();
                mega_mla::bar_sync<mega_mla::KernelTrait::Bars::EveryoneSync>();
            }
        }
    }

    template <class Ctx>
    CUTE_DEVICE void out_proj_vec(Ctx const& c) const {
        if (c.sched.o_post_flat) flat_o_post(c, c.sched.n_groups());
        vec_wo_a(c);
        __threadfence();
        auto run_wo_b = [&](auto KS_C) {
            constexpr int KS = decltype(KS_C)::value;
            vec_wo_b<KS>(c);
            __threadfence();
        };
        if (c.sched.ks_wo_b == 4)
            run_wo_b(cute::Int<4>{});
        else if (c.sched.ks_wo_b == 2)
            run_wo_b(cute::Int<2>{});
        else
            run_wo_b(cute::Int<1>{});
    }

    template <class Ctx>
    CUTE_DEVICE void role_vec(Ctx const& c) const {
        vec(c);
        token_loop_vec(c);
        out_proj_vec(c);
    }
};

}  // namespace mega_mla

namespace halops {

// What the roles cannot read from a launch parameter or from smem: the two GEMM
// range descriptions, the rendezvous set, the attention ring, and the scalars
// that depend on the plan the grid just read.  A field that is only one role's
// business belongs in that role, not here.
template <class KT_, class MP, class EP, class Plan, class Bars, class Sched,
          class R, class RO>
struct KernelPipelines {
    using KT = KT_;
    using MpT = MP;
    using EpT = EP;
    enum : int { tile_t = KT::TileT };

    const MP& params;
    const EP& ep;
    Plan& plan;
    mega_mla::AttnRing& rs;
    uint32_t& q_tma_phase;
    uint32_t& q_utccp_phase;
    int warp_idx;
    int lane_idx;
    int idx_in_warpgroup;
    int warpgroup_idx;

    int token_iters;
    const Sched& sched;
    R r;
    RO ro;
    Bars bars;
};

}  // namespace halops
