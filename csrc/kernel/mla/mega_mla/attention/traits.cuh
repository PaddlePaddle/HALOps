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

#include "common/sm100_arch.cuh"
#include "kernel/mla/mega_mla/attention/config.cuh"
#include "kernel/mla/mega_mla/mla_utils.cuh"

namespace halops::mla {
/// One warp per q head: RMS-normalises a q row and rotates its rope tail.
struct QHeadRow {
    static constexpr int Chunk = kElemsPerChunk;
    static constexpr int PerLane = 2 * Chunk;
    static_assert(KernelBase::WarpSize * PerLane == ModelDims::DimQ,
                  "one warp per q head");
    CUTE_DEVICE static int base(int half, int lane) {
        return (half * KernelBase::WarpSize + lane) * Chunk;
    }
    CUTE_DEVICE static int rope_offset(int lane) {
        return base(1, lane) - ModelDims::DimNope;
    }
    template <class Store>
    CUTE_DEVICE static void normalize(const __nv_bfloat16 (&vals)[PerLane],
                                      int lane, const float (&rope_cs)[Chunk],
                                      float eps, Store&& store) {
        float sq = 0.0f;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < PerLane; ++i) {
            const float val = __bfloat162float(vals[i]);
            sq += val * val;
        }
        sq = warp_sum(sq);
        const float inv = rsqrtf(sq / ModelDims::DimQ + eps);
        CUTLASS_PRAGMA_UNROLL
        for (int half = 0; half < 2; ++half) {
            const int b = base(half, lane);
            __nv_bfloat16 o8[Chunk];
            float vals_f[Chunk];
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < Chunk; ++i)
                vals_f[i] = __bfloat162float(vals[half * Chunk + i]) * inv;
            if (b >= ModelDims::DimNope) {
                CUTLASS_PRAGMA_UNROLL
                for (int k = 0; k < Chunk / 2; ++k) {
                    const float cos_v = rope_cs[k * 2],
                                sin_v = rope_cs[k * 2 + 1];
                    const float x0 = vals_f[k * 2], x1 = vals_f[k * 2 + 1];
                    vals_f[k * 2] = x0 * cos_v - x1 * sin_v;
                    vals_f[k * 2 + 1] = x0 * sin_v + x1 * cos_v;
                }
            }
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < Chunk; ++i) o8[i] = __float2bfloat16(vals_f[i]);
            store(b, *reinterpret_cast<const uint4*>(o8));
        }
    }
};

/// Buffer selection and q-wait points shared by the attention warp roles.
struct AttnSchedule {
    static constexpr int kRopeDepth = 2;

    template <class Tok>
    CUTE_DEVICE static cutlass::bfloat16_t* q_smem(Tok const& tok) {
        return tok.plan.bufs.kv.dequant[tok.q_buf].nope.data();
    }
    template <class Ctx>
    CUTE_DEVICE static uint8_t* rope_stage(Ctx const& ctx, int buf) {
        return buf == 0 ? (uint8_t*)ctx.plan.bufs.kv.rope_stage.data()
                        : (uint8_t*)ctx.plan.rope_stage2;
    }
    template <class Tok>
    CUTE_DEVICE static int kv_start(Tok const& tok) {
        return tok.blk_lo + min(AttnConfig::KvPause, tok.blk_hi - tok.blk_lo);
    }
    template <class Tok, class Wait>
    CUTE_DEVICE static void wait_q_at(Tok const& tok, int block, Wait&& wait) {
        if (block == tok.blk_lo + 1) wait();
    }
    template <class Tok, class Wait>
    CUTE_DEVICE static void wait_q_after(Tok const& tok, Wait&& wait) {
        if (tok.blk_lo + 1 >= tok.blk_hi) wait();
    }
    template <class Ctx>
    CUTE_DEVICE static int next_head(Ctx const& ctx) {
        int claimed = 0;
        if (KernelBase::lane_idx() == 0)
            claimed = atomicAdd(&ctx.plan.q_head_ctr, 1);
        return __shfl_sync(MlaAttnConfig::kFullMask, claimed, 0);
    }
    template <class Ctx>
    CUTE_DEVICE static void heads_done(Ctx const& ctx) {
        if (threadIdx.x == 0) ctx.plan.q_head_ctr = 0;
    }
};

/// Stages the O accumulator from TMEM to smem in TMA-box chunks and stores it.
struct OStage {
    static constexpr int EpiCols = AttnConfig::OTmaBox;
    static constexpr int Chunks = AttnConfig::OMmaN / EpiCols;
    static constexpr int RunCols = ModelDims::DimV / 4;
    static constexpr int Vec = kElemsPerChunk;

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
}  // namespace halops::mla
