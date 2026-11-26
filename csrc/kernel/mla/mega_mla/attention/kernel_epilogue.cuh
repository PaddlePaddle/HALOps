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

#include <cute/tensor.hpp>

#include "common/sm100_arch.cuh"
#include "kernel/mla/mega_mla/attention/config.cuh"
#include "kernel/mla/mega_mla/attention/context.cuh"
#include "kernel/mla/mega_mla/attention/kernel_mainloop.cuh"
#include "kernel/mla/mega_mla/attention/kernel_prelude.cuh"
#include "kernel/mla/mega_mla/attention/traits.cuh"

namespace halops::mla {
/// Warpgroup barrier, then one thread arrives on the cross-CTA flag.
template <class KT, class WgSync, int FirstWarp, class Bar>
CUTE_DEVICE void mla_attn_arrive_wg(Bar const& bar) {
    bar_sync<WgSync>();
    if (threadIdx.x == FirstWarp * KT::WarpSize) bar.arrive_thread();
}
template <class KT, class Bar>
CUTE_DEVICE void mla_attn_arrive_drain_wg(Bar const& bar) {
    mla_attn_arrive_wg<KT, typename KT::Bars::Wg1Sync,
                       KernelBase::Roles::DrainWarpBegin>(bar);
}
template <class KT, class Bar>
CUTE_DEVICE void mla_attn_arrive_vec_wg(Bar const& bar) {
    mla_attn_arrive_wg<KT, typename KT::Bars::Wg2Sync,
                       KernelBase::Roles::VecWarpBegin>(bar);
}

/// Drains the prelude GEMM accumulators and publishes their groups.
template <class Ctx>
CUTE_DEVICE void mla_attn_gemm_drain(Ctx const& ctx) {
    using KT = typename Ctx::KT;
    using CfgQkv = typename KT::Qkv;
    using CfgWqB = typename KT::WqB;
    using CfgKvGate = typename KT::KvGate;
    constexpr int TileTokens = KT::TileT;
    {
        const PipelineOffsets led = ctx.sched.ledger_before_qkv();
        auto op = make_drain_op<CfgQkv>(ctx.prelude.res_qkv, led.base_acc());
        drain_items(
            op, blockIdx.x, ctx.sched.items_qkv, gridDim.x,
            [&](int item) {
                const auto it = ctx.prelude.qkv_item(ctx.sched, item);
                op.store(it.wi, it.out);
                return it.wi.n_row / TileTokens;
            },
            [&](int item, int group) {
                if (ctx.sched.qkv_is_q(item))
                    mla_attn_arrive_drain_wg<KT>(ctx.bars.qlora_ready(group));
                else
                    mla_attn_arrive_drain_wg<KT>(ctx.bars.kv_ready(group));
            });
    }
    {
        const PipelineOffsets led = ctx.sched.ledger_before_wq_b();
        auto op = make_drain_op<CfgWqB>(ctx.prelude.res_wqb, led.base_acc());
        drain_items(
            op, ctx.sched.wqb_first(), ctx.sched.items_wq_b,
            ctx.sched.wqb_step(),
            [&](int item) {
                const auto it = ctx.prelude.wqb_item(ctx.sched, item);
                op.store(it.wi, it.out);
                return it.wi.n_row / TileTokens;
            },
            [&](int, int group) {
                mla_attn_arrive_drain_wg<KT>(ctx.bars.flat_q_ready(group));
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
                if (TileTokens >= MlaAttnConfig::kRolledKvGateMinTile &&
                    ctx.sched.kvg_ctas > 0)
                    op.store_rolled(it);
                else
                    op.store_unrolled(it);
                return it.n_tile;
            },
            [&](int, int n_tile) {
                mla_attn_arrive_drain_wg<KT>(ctx.bars.ring_ready(n_tile));
            });
    }

    phase_end_sync();
}

/// Vector warpgroup prelude: x quantisation, q/kv reduction and c128.
template <class Ctx>
CUTE_DEVICE void mla_attn_gemm_vec(Ctx const& ctx) {
    using KT = typename Ctx::KT;
    using Dims = typename KT::Dims;
    constexpr int TileTokens = KT::TileT;
    constexpr int KSplitQkv = KT::Qkv::KSplit;
    constexpr int kVecTid0 =
        KernelBase::Roles::VecWarpBegin * KernelBase::WarpSize;
    auto pub_x = [&](int group) {
        mla_attn_arrive_vec_wg<KT>(ctx.bars.x_ready(group));
    };
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
    if (ctx.params.c128_compress()) mla_attn_c128_block(ctx);

    phase_end_sync();
}

/// Pools every completed c128 window slice owned by this CTA.
template <class Ctx>
CUTE_DEVICE void mla_attn_c128_block(Ctx const& ctx) {
    using KT = typename Ctx::KT;
    constexpr int TileTokens = KT::TileT;
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
        const int dmax = ctx.params.draft_tokens > 0 ? ctx.params.draft_tokens
                                                     : ModelDims::C128Ratio;
        int k = 0;
        while (k + 1 < dmax && token - (k + 1) >= 0 &&
               __ldg(p.positions + token - (k + 1)) == pos_t - (k + 1))
            ++k;
        const int s0 = ModelDims::C128Ratio - 1 - k;
        c128_slice_partial(vec_tid, slice, p.kv_score_ring,
                           ctx.params.shape_kv_gate.w_out,
                           p.kv_gate_ring_row[token], s0, p.c128_ape, row_ws,
                           sq_ws, ctx.plan.c128_warp_sq, [&] {
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
CUTE_DEVICE void mla_attn_load_scores(
    Ctx const& ctx, int wv, float (&scores)[AttnConfig::KvBlock / 2]) {
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
        for (int i = 0;
             i < (AttnConfig::KvBlock / 2) / MlaAttnConfig::kF32PerVec4; ++i)
            ctx.plan.bufs.kv.p_exchange_buf[px_slot][i * KernelBase::WarpSize +
                                                     ctx.lane_idx] =
                *(float4*)(p_peer + i * MlaAttnConfig::kF32PerVec4);
    };
    auto px_read_add = [&] {
        CUTE_UNROLL
        for (int i = 0;
             i < (AttnConfig::KvBlock / 2) / MlaAttnConfig::kF32PerVec4; ++i) {
            float2 peer[2];
            *(float4*)peer =
                ctx.plan.bufs.kv
                    .p_exchange_buf[px_slot]
                                   [i * KernelBase::WarpSize + ctx.lane_idx];
            float2* cur_p = (float2*)(scores + i * MlaAttnConfig::kF32PerVec4);
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

CUTE_DEVICE void mla_attn_rescale_o(float scale) {
    constexpr int kCols = MlaAttnConfig::kRescaleCols;
    const float2 sf = {scale, scale};
    tcgen05_after_thread_sync();
    float2 acc[kCols / 2];
    CUTE_UNROLL
    for (int ci = 0; ci < (ModelDims::DimV / 2) / kCols; ++ci) {
        tmem_ld_32dp32bNx<kCols>(AttnConfig::TmemCols::O + ci * kCols, acc);
        cutlass::arch::fence_view_async_tmem_load();
        CUTE_UNROLL
        for (int i = 0; i < kCols / 2; ++i) acc[i] = mul_rn_f32x2(acc[i], sf);
        tmem_st_32dp32bNx<kCols>(AttnConfig::TmemCols::O + ci * kCols, acc);
        cutlass::arch::fence_view_async_tmem_store();
    }
    tcgen05_before_thread_sync();
}

template <class Ctx>
CUTE_DEVICE void mla_attn_mask_scores(
    Ctx const& ctx, int block_idx, float (&scores)[MlaAttnConfig::kHalfKeys]) {
    using KT = typename Ctx::KT;
    using Dims = typename KT::Dims;
    const bool high_keys = ctx.idx_in_warpgroup >= Dims::Heads;
    const int valid_in_half =
        min(max(ctx.kv_tile_valid(block_idx) -
                    (high_keys ? MlaAttnConfig::kHalfKeys : 0),
                0),
            MlaAttnConfig::kHalfKeys);
    CUTE_UNROLL
    for (int i = 0; i < MlaAttnConfig::kHalfKeys; ++i)
        if (i >= valid_in_half) scores[i] = -CUDART_INF_F;
}

template <class Ctx>
CUTE_DEVICE bool mla_attn_update_max(
    Ctx const& ctx, const float (&scores)[MlaAttnConfig::kHalfKeys],
    MlaAttnSoftmaxState& st, float& scale_for_old) {
    using KT = typename Ctx::KT;
    using Dims = typename KT::Dims;
    float cur_pi_max = -CUDART_INF_F;
    CUTE_UNROLL
    for (int i = 0; i < MlaAttnConfig::kHalfKeys; ++i)
        cur_pi_max = max(cur_pi_max, scores[i]);
    cur_pi_max *= ctx.params.sm_scale_div_log2;
    ctx.plan.wg_exchange[ctx.idx_in_warpgroup] = cur_pi_max;
    bar_sync<KernelBase::Bars::Wg1Sync>();
    cur_pi_max = max(cur_pi_max,
                     ctx.plan.wg_exchange[ctx.idx_in_warpgroup ^ Dims::Heads]);
    st.real_mi = max(st.real_mi, cur_pi_max);
    const bool should_scale_o =
        __any_sync(MlaAttnConfig::kFullMask,
                   cur_pi_max - st.mi > AttnConfig::RescaleThreshold);
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
CUTE_DEVICE void mla_attn_write_probs(
    Ctx const& ctx, int wv, const float (&scores)[MlaAttnConfig::kHalfKeys],
    float scale_for_old, MlaAttnSoftmaxState& st) {
    using KT = typename Ctx::KT;
    using Dims = typename KT::Dims;
    const float2 scale = {ctx.params.sm_scale_div_log2,
                          ctx.params.sm_scale_div_log2};
    const int sS_off = ctx.lane_idx * kElemsPerChunk +
                       (wv & 1) * (Dims::Heads / 2) * kElemsPerChunk +
                       (wv / 2) * Dims::Heads * MlaAttnConfig::kHalfKeys;
    __nv_bfloat162 probs[MlaAttnConfig::kHalfKeys / 2];
    float2 neg_new_max = {-st.mi, -st.mi};
    float2 cur_sum = {0.0f, 0.0f};
    CUTE_UNROLL
    for (int i = 0; i < MlaAttnConfig::kHalfKeys / 2; ++i) {
        float2 logit = fma_rn_f32x2(float2{scores[i * 2], scores[i * 2 + 1]},
                                    scale, neg_new_max);
        logit.x = exp2f(logit.x);
        logit.y = exp2f(logit.y);
        cur_sum = add_rn_f32x2(cur_sum, logit);
        probs[i] = __float22bfloat162_rn(logit);
    }
    st.li = fma(st.li, scale_for_old, (cur_sum.x + cur_sum.y));

    cutlass::bfloat16_t* sS_base =
        ctx.plan.bufs.kv.s[ctx.rs.buf_idx].data() + sS_off;
    CUTE_UNROLL
    for (int i = 0; i < MlaAttnConfig::kHalfKeys / kElemsPerChunk; ++i)
        *(__int128_t*)(sS_base + Dims::Heads * kElemsPerChunk * i) =
            *(__int128_t*)(probs + i * (kElemsPerChunk / 2));
}

template <class Ctx>
CUTE_DEVICE float mla_attn_finalize_softmax(Ctx const& ctx,
                                            MlaAttnSoftmaxState& st,
                                            float attn_sink) {
    using KT = typename Ctx::KT;
    using Dims = typename KT::Dims;
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
        split_pub ? 1.0f
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
CUTE_DEVICE void mla_attn_store_o(Ctx const& ctx, int wv, float o_scale) {
    using KT = typename Ctx::KT;
    using Dims = typename KT::Dims;
    const float2 o_scale_f2 = {o_scale, o_scale};
    cute::Tensor sO =
        cute::make_tensor(cute::make_smem_ptr(ctx.plan.bufs.qo.o_buf.data()),
                          AttnConfig::SmemLayoutOBuf{});
    cutlass::bfloat16_t* sO_bases[OStage::EpiCols / OStage::Vec];
    OStage::smem_bases(sO, ctx.idx_in_warpgroup, sO_bases);
    using OBox =
        cute::Shape<cute::Int<Dims::Heads>, cute::Int<AttnConfig::OTmaBox>>;
    cute::Tensor tma_sO =
        cute::flat_divide(sO, OBox{})(cute::_, cute::_, cute::_0{}, cute::_);

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

/// Online softmax over the KV blocks of one token, then the O store.
template <class Ctx>
CUTE_DEVICE void mla_attn_drain(Ctx const& ctx) {
    using KT = typename Ctx::KT;
    using Dims = typename KT::Dims;
    using Roles = KernelBase::Roles;
    const int wv = Roles::drain_local_warp(ctx.warp_idx);
    const int head = ctx.idx_in_warpgroup % Dims::Heads;
    const float attn_sink =
        ctx.params.attn_sink == nullptr
            ? -CUDART_INF_F
            : __ldg((float*)ctx.params.attn_sink + head) * CUDART_L2E_F;

    MlaAttnSoftmaxState st{AttnConfig::MaxInit, 0.0f, -CUDART_INF_F};
    CUTE_NO_UNROLL
    for (int block_idx = ctx.blk_lo; block_idx < ctx.blk_hi; ++block_idx) {
        bar_sync<KernelBase::Bars::Wg1Sync>();
        ctx.plan.bar_qk_done[ctx.rs.buf_idx].wait(ctx.rs.bar_phase);
        tcgen05_after_thread_sync();

        float scores[MlaAttnConfig::kHalfKeys];
        mla_attn_load_scores(ctx, wv, scores);
        mla_attn_mask_scores(ctx, block_idx, scores);
        float scale_for_old;
        const bool should_scale_o =
            mla_attn_update_max(ctx, scores, st, scale_for_old);
        mla_attn_write_probs(ctx, wv, scores, scale_for_old, st);
        if (block_idx != ctx.blk_lo && should_scale_o) {
            const int prev_buf = (ctx.rs.buf_idx + AttnConfig::NumBufs - 1) %
                                 AttnConfig::NumBufs;
            const bool prev_phase = ctx.rs.bar_phase ^ (ctx.rs.buf_idx == 0);
            ctx.plan.bar_sv_done[prev_buf].wait(prev_phase);
            mla_attn_rescale_o(scale_for_old);
        }

        cutlass::arch::fence_view_async_shared();
        ctx.plan.bar_so_ready[ctx.rs.buf_idx].arrive();
        if (block_idx != ctx.blk_hi - 1) ctx.rs.update();
    }
    mla_attn_store_o(ctx, wv, mla_attn_finalize_softmax(ctx, st, attn_sink));
    bar_sync<KernelBase::Bars::Wg1Sync>();
}

CUTE_DEVICE void mla_attn_dequant_row(const cutlass::float_e4m3_t* raw,
                                      uint32_t smem_dst, uint64_t row_scales) {
    using Kv = KvCacheLayout;
    constexpr int kBlockBytes = (int)sizeof(cutlass::bfloat16_t) *
                                AttnConfig::KvBlock * AttnConfig::Sw128Cols;
    cutlass::bfloat16_t scales[Kv::ScaleSlotBytes];
    __nv_fp8_e8m0 scales_e8m0[Kv::ScaleSlotBytes];
    *(uint64_t*)scales_e8m0 = row_scales;
    CUTE_UNROLL
    for (int k = 0; k < Kv::ScaleSlotBytes; k += 2)
        *(__nv_bfloat162_raw*)(scales + k) =
            __nv_cvt_e8m0x2_to_bf162raw(*(unsigned short*)(scales_e8m0 + k));

    uint64_t cur_fp8x8 = *(const uint64_t*)raw;
    CUTE_UNROLL
    for (int lc = 0; lc < Kv::NopeGroups; ++lc) {
        __nv_fp8x2_e4m3 data_fp8[MlaAttnConfig::kFp8x2PerWord];
        __nv_bfloat162 data_bf16[MlaAttnConfig::kFp8x2PerWord];
        *(uint64_t*)data_fp8 = cur_fp8x8;
        if (lc + 1 < Kv::NopeGroups)
            cur_fp8x8 = *(const uint64_t*)(raw + (lc + 1) * Kv::NopeQuantBlock);
        cutlass::bfloat16_t scale = scales[lc];
        CUTE_UNROLL
        for (int i = 0; i < MlaAttnConfig::kFp8x2PerWord; ++i)
            data_bf16[i] = fp8x2_to_bf16x2_with_scale(
                data_fp8[i], *(__nv_bfloat16*)(&scale));
        smem_st_128b(smem_dst + lc * kBlockBytes, *(__int128_t*)data_bf16);
    }
}

/// Dequantises the raw fp8 nope KV tiles into the bf16 UMMA layout.
template <class Ctx>
CUTE_DEVICE void mla_attn_vec(Ctx const& ctx) {
    using KT = typename Ctx::KT;
    using Dims = typename KT::Dims;
    constexpr int kVec = kElemsPerChunk;
    constexpr int kGroupSize = KvCacheLayout::NopeQuantBlock / kVec,
                  NUM_GROUPS = KT::ThreadsPerWarpGroup / kGroupSize,
                  ROWS_PER_GROUP = AttnConfig::KvBlock / NUM_GROUPS;
    int group_idx = ctx.idx_in_warpgroup / kGroupSize,
        idx_in_group = ctx.idx_in_warpgroup % kGroupSize;
    cute::Tensor nope0 = cute::make_tensor(
        cute::make_smem_ptr(ctx.plan.bufs.kv.dequant[0].nope.data()),
        AttnConfig::SmemLayoutKTiles_SW128<Dims::DimNope /
                                           AttnConfig::Sw128Cols>{});
    cutlass::bfloat16_t* nope0_base = &nope0(group_idx, idx_in_group * kVec);
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
        const bool has_inline_scales = seg.stride_kv_block > Dims::KvRowBytes;
        CUTE_UNROLL
        for (int lr = 0; lr < ROWS_PER_GROUP; ++lr)
            dst[lr] = has_inline_scales
                          ? __ldg(kv_seg_scale_ptr(kv_tile, seg,
                                                   lr * NUM_GROUPS + group_idx))
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
            mla_attn_dequant_row(
                raw_nope_base + lr * NUM_GROUPS * Dims::DimNope,
                cur_nope_base + (int)sizeof(cutlass::bfloat16_t) * lr *
                                    NUM_GROUPS * AttnConfig::Sw128Cols,
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
CUTE_DEVICE void mla_attn_token_loop_drain(Ctx const& ctx) {
    phase1_qnorm_slice(ctx);
    for_each_token(ctx, [&](auto const& tok) {
        if (!tok.token_active) return;
        phase1_prepare_q(ctx, tok);
        mla_attn_drain(tok);
    });
}

/// Drain warpgroup: prelude GEMM drains, then per-token softmax.
template <class Ctx>
CUTE_DEVICE void mla_attn_role_drain(Ctx const& ctx) {
    mla_attn_gemm_drain(ctx);
    mla_attn_token_loop_drain(ctx);
}

template <class Ctx>
CUTE_DEVICE void mla_attn_token_loop_vec(Ctx const& ctx) {
    for_each_token(ctx, [&](auto const& tok) {
        if (tok.token_active) mla_attn_vec(tok);
        if (ctx.params.qnorm_dist) cutlass::arch::fence_view_async_shared();
    });
}

/// Vector warpgroup: prelude reductions, then per-token KV dequant.
template <class Ctx>
CUTE_DEVICE void mla_attn_role_vec(Ctx const& ctx) {
    mla_attn_gemm_vec(ctx);
    mla_attn_token_loop_vec(ctx);
}
}  // namespace halops::mla
