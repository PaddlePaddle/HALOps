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
#include <math_constants.h>

#include <cute/tensor.hpp>

#include "kernel/mla/mla_gemm.cuh"
#include "kernel/mla/mla_params.cuh"
#include "kernel/mla/mla_scheduler.cuh"
#include "kernel/mla/mla_utils.cuh"

namespace halops::mla {

template <class Ctx>
struct PerToken {
    decltype(Ctx::params)& params;
    decltype(Ctx::ep)& ep;
    decltype(Ctx::plan)& plan;
    decltype(Ctx::rs)& rs;
    decltype(Ctx::q_tma_phase)& q_tma_phase;
    decltype(Ctx::q_utccp_phase)& q_utccp_phase;
    int warp_idx;
    int lane_idx;
    int idx_in_warpgroup;

    int token_idx;
    bool token_active;
    int topk_length;
    int swa_blocks;
    int c128_len;
    const int* c128_pt;
    KvRuns kv_runs;
    KvSegGeom kv_seg_swa;
    KvSegGeom kv_seg_c128;
    int blk_lo;
    int blk_hi;
    int slot;
    int q_buf;

    CUTE_DEVICE KvRuns kv_tile_at(int block_idx) const {
        if (block_idx < swa_blocks)
            return kv_tile_slice(kv_runs, block_idx * AttnConfig::KvBlock,
                                 AttnConfig::KvBlock);
        return kv_tile_page_table(
            c128_pt, c128_len, (block_idx - swa_blocks) * AttnConfig::KvBlock,
            AttnConfig::KvBlock, params.c128_page_shift,
            params.c128_rows_per_page);
    }
    CUTE_DEVICE const KvSegGeom& kv_seg_at(int block_idx) const {
        return block_idx < swa_blocks ? kv_seg_swa : kv_seg_c128;
    }

    CUTE_DEVICE int kv_tile_valid(int block_idx) const {
        return block_idx < swa_blocks
                   ? topk_length - block_idx * AttnConfig::KvBlock
                   : c128_len - (block_idx - swa_blocks) * AttnConfig::KvBlock;
    }

    CUTE_DEVICE auto kv_nope_desc(int block_idx) const {
        return block_idx < swa_blocks ? params.kv_nope_dense
                                      : params.c128_nope_dense;
    }
    CUTE_DEVICE const CUtensorMap* kv_nope_gather(int block_idx) const {
        return block_idx < swa_blocks ? &params.tensor_map_kv_nope
                                      : &params.c128_nope;
    }
    CUTE_DEVICE auto kv_rope_desc(int block_idx) const {
        return block_idx < swa_blocks ? params.kv_rope_dense
                                      : params.c128_rope_dense;
    }
    CUTE_DEVICE const CUtensorMap* kv_rope_gather(int block_idx) const {
        return block_idx < swa_blocks ? &params.tensor_map_kv_rope
                                      : &params.c128_rope;
    }
};

template <class Ctx>
CUTE_DEVICE PerToken<Ctx> make_per_token(Ctx const& k, int token_it) {
    const CtaWork work = k.params.wplan.work[token_it * gridDim.x + blockIdx.x];
    const bool token_active = work.blk_hi > work.blk_lo;
    const int token_idx = work.token;

    const int topk_length = token_active
                                ? __ldg(&k.params.wplan.kv_runs[token_idx].w)
                                : AttnConfig::KvBlock;
    const int total_topk_padded =
        max(cute::round_up(topk_length, int{AttnConfig::KvBlock}),
            int{AttnConfig::KvBlock});
    const int swa_blocks = total_topk_padded / AttnConfig::KvBlock;
    const int c128_len = (k.params.c128_attn_len && token_active)
                             ? __ldg(k.params.c128_attn_len + token_idx)
                             : 0;
    const int* c128_pt =
        (k.params.c128_page_table && token_active)
            ? k.params.c128_page_table +
                  (int64_t)token_idx * k.params.c128_page_table_stride
            : nullptr;

    const KvRuns kv_runs = make_kv_runs(
        __ldg(k.params.wplan.kv_runs + (token_active ? token_idx : 0)),
        topk_length, k.params.page_shift, k.params.rows_per_page);

    const KvSegGeom kv_seg_swa{reinterpret_cast<const uint8_t*>(k.params.kv),
                               k.params.stride_kv_block};
    const KvSegGeom kv_seg_c128{k.params.c128_pool,
                                k.params.c128_stride_kv_block};

    return PerToken<Ctx>{k.params,
                         k.ep,
                         k.plan,
                         k.rs,
                         k.q_tma_phase,
                         k.q_utccp_phase,
                         k.warp_idx,
                         k.lane_idx,
                         k.idx_in_warpgroup,
                         token_idx,
                         token_active,
                         topk_length,
                         swa_blocks,
                         c128_len,
                         c128_pt,
                         kv_runs,
                         kv_seg_swa,
                         kv_seg_c128,
                         work.blk_lo,
                         work.blk_hi,
                         work.slot,
                         k.rs.buf_idx ^ 1};
}

struct QHeadRow {
    static constexpr int Chunk = 16 / sizeof(__nv_bfloat16);
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

template <class Ctx>
CUTE_DEVICE void phase1_qnorm_slice(Ctx const& ctx) {
    if (!(ctx.params.qnorm_dist && ctx.warpgroup_idx < 2)) return;
    constexpr int kNumWarps = 2 * KernelBase::WarpsPerWarpGroup;
    const int lane = KernelBase::lane_idx(), warp = KernelBase::warp_idx();
    const int num_tokens = ctx.params.num_tokens;
    const int heads_per_cta = decltype(ctx.bars)::qn_slice(num_tokens);
    const int h0 = blockIdx.x * heads_per_cta;
    const int h1 = min(h0 + heads_per_cta, num_tokens * ModelDims::Heads);
    auto* gq = reinterpret_cast<__nv_bfloat16*>(ctx.params.ws_flat_q_bf16);
    const int rope_rb = QHeadRow::rope_offset(lane);
    int waited_g = -1;
    CUTE_NO_UNROLL
    for (int hg = h0 + warp; hg < h1; hg += kNumWarps) {
        const int tok = hg / ModelDims::Heads, head = hg % ModelDims::Heads;
        const int group = tok / ctx.TileT;
        if (group != waited_g) {
            ctx.bars.flat_q_ready(group).wait_warp(0);
            waited_g = group;
        }
        __nv_bfloat16* grow =
            gq + (int64_t)tok * ModelDims::QTotal + head * ModelDims::DimQ;
        float rope_cs[QHeadRow::Chunk];
        if (rope_rb >= 0)
            rope_slice(ctx.params.freqs_real +
                           ctx.params.positions[tok] * ModelDims::DimRope,
                       rope_rb, rope_cs);
        __nv_bfloat16 vals[QHeadRow::PerLane];
        *reinterpret_cast<uint4*>(vals) = __ldcg(
            reinterpret_cast<const uint4*>(grow + QHeadRow::base(0, lane)));
        *reinterpret_cast<uint4*>(vals + QHeadRow::Chunk) = __ldcg(
            reinterpret_cast<const uint4*>(grow + QHeadRow::base(1, lane)));
        QHeadRow::normalize(vals, lane, rope_cs, ctx.params.q_eps,
                            [&](int base, const uint4& out) {
                                *reinterpret_cast<uint4*>(grow + base) = out;
                            });
    }
    __threadfence();
    bar_sync<KernelBase::Bars::Wg01Sync>();
    if (threadIdx.x == 0 && h0 < h1) {
        const int g_lo = (h0 / ModelDims::Heads) / ctx.TileT;
        const int g_hi = ((h1 - 1) / ModelDims::Heads) / ctx.TileT;
        for (int group = g_lo; group <= g_hi; ++group)
            ctx.bars.qn_ready(group).arrive_thread();
    }
}

template <class Ctx, class Tok>
CUTE_DEVICE void kv_gate_wait_warp(Ctx const& ctx, Tok const& tok) {
    const int draft = ctx.params.draft_tokens;
    const int g_lo =
        draft > 0 ? max(0, tok.token_idx - (draft - 1)) / ctx.TileT : 0;
    const int g_hi = draft > 0 ? min(ctx.params.num_tokens - 1,
                                     tok.token_idx + (draft - 1)) /
                                     ctx.TileT
                               : ctx.sched.n_groups - 1;
    for (int group = g_lo; group <= g_hi; ++group)
        ctx.bars.kvcache_ready(group).wait_warp(0);
    if (ctx.params.c128_compress()) {
        const int tq = tok.token_idx;
        const int pos_t = __ldg(ctx.params.positions + tq);
        const int lo = draft > 0 ? max(0, tq - (draft - 1)) : 0;
        for (int s0 = lo; s0 <= tq; s0 += KernelBase::WarpSize) {
            const int src_token = s0 + ctx.lane_idx;
            if (src_token <= tq) {
                const int ps = __ldg(ctx.params.positions + src_token);
                const bool fire = (ps & (ModelDims::C128Ratio - 1)) ==
                                  ModelDims::C128Ratio - 1;
                if (fire && (draft == 0 || pos_t - ps == tq - src_token))
                    while (
                        gmem_ld_acquire(ctx.params.c128_flags +
                                        C128Layout::FlagsPerToken * src_token +
                                        C128Layout::FlagRowWritten) == 0) {
                    }
            }
        }
        __syncwarp();
    }
}

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
        return __shfl_sync(0xffffffffu, claimed, 0);
    }
    template <class Ctx>
    CUTE_DEVICE static void heads_done(Ctx const& ctx) {
        if (threadIdx.x == 0) ctx.plan.q_head_ctr = 0;
    }
};

template <class Ctx, class Body>
CUTE_DEVICE void for_each_token(Ctx const& ctx, Body&& body) {
    for (int token_it = 0; token_it < ctx.token_iters; ++token_it) {
        body(make_per_token(ctx, token_it));
        cta_sync();
        if (token_it + 1 == ctx.token_iters) cta_sync();
    }
}

template <class Ctx, class Tok>
CUTE_DEVICE void phase1_prepare_q(Ctx const& ctx, Tok const& tok) {
    if (ctx.warpgroup_idx < 2) {
        if (ctx.params.qnorm_dist) {
            cutlass::arch::fence_view_async_shared();
            if (ctx.warp_idx == 0)
                ctx.bars.qn_ready(tok.token_idx / ctx.TileT).wait_warp(0);
            bar_sync<KernelBase::Bars::Wg01Sync>();
            if (threadIdx.x == 0) {
                fence_proxy_async_global();
                cute::Tensor sQt = cute::make_tensor(
                    cute::make_smem_ptr(AttnSchedule::q_smem(tok)),
                    AttnConfig::SmemLayoutQ_SW128{});
                cute::Tensor gQ = ctx.params.tma_Q.get_tma_tensor(
                    ctx.params.shape_Q)(cute::_, cute::_, tok.token_idx);
                launch_tma_copy(ctx.params.tma_Q, gQ, sQt, ctx.plan.bar_q_tma,
                                cute::TMA::CacheHintSm90::EVICT_FIRST);
                ctx.plan.bar_q_tma.arrive_and_expect_tx(
                    ModelDims::Heads * ModelDims::DimQ *
                    sizeof(cutlass::bfloat16_t));
            }
            ctx.plan.bar_q_tma.wait(ctx.q_tma_phase);
            ctx.q_tma_phase ^= 1;
            bar_sync<KernelBase::Bars::Wg01Sync>();
            return;
        }
        const int lane = KernelBase::lane_idx();
        const int rope_rb = QHeadRow::rope_offset(lane);
        float rope_cs[QHeadRow::Chunk];
        if (rope_rb >= 0)
            rope_slice(
                ctx.params.freqs_real +
                    ctx.params.positions[tok.token_idx] * ModelDims::DimRope,
                rope_rb, rope_cs);
        ctx.bars.flat_q_ready(tok.token_idx / ctx.TileT).wait_warp(0);

        cute::Tensor sQ =
            cute::make_tensor(cute::make_smem_ptr(AttnSchedule::q_smem(tok)),
                              AttnConfig::SmemLayoutQ_SW128{});

        const auto* gq =
            reinterpret_cast<const __nv_bfloat16*>(ctx.params.ws_flat_q_bf16) +
            (int64_t)tok.token_idx * ModelDims::QTotal;
        auto load_q16 = [&](int head, __nv_bfloat16(&dst)[QHeadRow::PerLane]) {
            const __nv_bfloat16* grow = gq + head * ModelDims::DimQ;
            *reinterpret_cast<uint4*>(dst) =
                *reinterpret_cast<const uint4*>(grow + QHeadRow::base(0, lane));
            *reinterpret_cast<uint4*>(dst + QHeadRow::Chunk) =
                *reinterpret_cast<const uint4*>(grow + QHeadRow::base(1, lane));
        };
        __nv_bfloat16 v_ahead[QHeadRow::PerLane];
        int head = AttnSchedule::next_head(ctx);
        if (head < ModelDims::Heads) load_q16(head, v_ahead);
        while (head < ModelDims::Heads) {
            __nv_bfloat16 vals[QHeadRow::PerLane];
            *reinterpret_cast<uint4*>(vals) =
                *reinterpret_cast<const uint4*>(v_ahead);
            *reinterpret_cast<uint4*>(vals + QHeadRow::Chunk) =
                *reinterpret_cast<const uint4*>(v_ahead + QHeadRow::Chunk);
            const int h_next = AttnSchedule::next_head(ctx);
            if (h_next < ModelDims::Heads) load_q16(h_next, v_ahead);
            QHeadRow::normalize(vals, lane, rope_cs, ctx.params.q_eps,
                                [&](int base, const uint4& out) {
                                    *reinterpret_cast<uint4*>(&sQ(head, base)) =
                                        out;
                                });
            head = h_next;
        }

        cutlass::arch::fence_view_async_shared();

        bar_sync<KernelBase::Bars::Wg01Sync>();
        AttnSchedule::heads_done(ctx);
    }
}

template <class KT>
struct AttentionMainloop {
    using Dims = ModelDims;
    using CfgQkv = typename KT::Qkv;
    using CfgWqB = typename KT::WqB;
    using CfgKvGate = typename KT::KvGate;

    template <class Params>
    static void set_c128_page_geometry(Params& params,
                                       const torch::Tensor& c128_pool,
                                       int64_t stride_kv_block,
                                       int64_t page_block) {
        params.c128_pool = reinterpret_cast<uint8_t*>(c128_pool.data_ptr());
        params.c128_stride_kv_block = (int)stride_kv_block;
        params.c128_page_block = (int)page_block;
        params.c128_page_shift = page_shift_of(page_block);
        params.c128_rows_per_page =
            params.c128_stride_kv_block / Dims::KvRowBytes;
    }

    template <int Levels>
    static void make_kv_maps(void* pool, int64_t rows, CUtensorMap& nope,
                             CUtensorMap (&nope_dense)[Levels],
                             CUtensorMap& rope,
                             CUtensorMap (&rope_dense)[Levels]) {
        constexpr int kNopeWords = Dims::DimNope / sizeof(int64_t);
        auto nope_map = [&](uint32_t box_rows) {
            return make_tensor_map(
                {kNopeWords, (uint64_t)rows}, {Dims::KvRowBytes},
                {kNopeWords, box_rows}, pool,
                CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_INT64,
                CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
                CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_128B);
        };
        auto rope_map = [&](uint32_t box_rows) {
            return make_tensor_map(
                {Dims::DimRope, (uint64_t)rows}, {Dims::KvRowBytes},
                {Dims::DimRope, box_rows}, (uint8_t*)pool + Dims::DimNope,
                CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
                CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_128B);
        };
        nope = nope_map(1);
        rope = rope_map(1);
        for (int k = 0; k < Levels; ++k) {
            nope_dense[k] = nope_map((uint32_t)(Sm100::Gather4Rows << k));
            rope_dense[k] = rope_map((uint32_t)(Sm100::Gather4Rows << k));
        }
    }

    static auto to_underlying_arguments(const MlaAttentionArguments& args,
                                        const MlaAttentionProblem<KT>& prob) {
        const int num_tokens = prob.num_tokens;

        auto shape_Q =
            cute::make_shape(Dims::Heads, Dims::DimQ, (uint64_t)num_tokens);
        auto tma_Q = cute::make_tma_copy(
            cute::SM90_TMA_LOAD{},
            cute::make_tensor(
                cute::make_gmem_ptr(prob.ws_flat_q_bf16),
                cute::make_layout(shape_Q,
                                  cute::make_stride(Dims::DimQ, cute::_1{},
                                                    (int64_t)Dims::QTotal))),
            AttnConfig::SmemLayoutQ_SW128{});
        const GemmShape shape_kv_gate =
            make_gemm_shape<CfgKvGate::WeightTile, CfgKvGate::BK,
                            CfgKvGate::KSplit>(2 * prob.kvlora_rank,
                                               prob.hidden);
        const bool use_kv_gate = args.wkv_gate_weight.has_value();
        auto tma_kv_gate_w = make_bf16_gemm_tma_w<CfgKvGate>(
            use_kv_gate ? args.wkv_gate_weight->data_ptr() : (void*)prob.q_ptr,
            shape_kv_gate.w_out, shape_kv_gate.k_dim);
        auto tma_kv_gate_act = make_bf16_gemm_tma_act<CfgKvGate>(
            (const void*)prob.hidden_ptr, num_tokens, prob.hidden_row_stride,
            shape_kv_gate.k_dim);
        MainloopParams<KT, decltype(tma_Q), decltype(tma_kv_gate_w),
                       decltype(tma_kv_gate_act)>
            params{};
        static_cast<CommonParams<KT>&>(params) =
            CommonParams<KT>::from(args, prob);
        params.shape_Q = shape_Q;
        params.tma_Q = tma_Q;
        params.tma_kv_gate_w = tma_kv_gate_w;
        params.tma_kv_gate_act = tma_kv_gate_act;
        params.kv_gate_act_rows = num_tokens;
        params.shape_kv_gate = shape_kv_gate;
        set_attention_args(params, args, prob);
        set_prelude_args(params, args, prob);
        return params;
    }

    template <class Params>
    static void set_attention_args(Params& params,
                                   const MlaAttentionArguments& args,
                                   const MlaAttentionProblem<KT>& prob) {
        params.sm_scale_div_log2 = (float)args.sm_scale * CUDART_L2E_F;
        params.page_block_size = (int)args.page_size;
        params.page_shift = page_shift_of(args.page_size);
        params.attn_sink = optional_ptr<float>(args.attn_sink);
        params.kv =
            reinterpret_cast<cutlass::bfloat16_t*>(args.kv_cache.data_ptr());
        params.stride_kv_block = (int)args.kv_cache.stride(0);
        params.rows_per_page = params.stride_kv_block / Dims::KvRowBytes;
        params.freqs_real = args.freqs_real.data_ptr<float>();
        params.positions = args.positions.data_ptr<int>();
        params.use_c128 = args.c128_ape.has_value() ? 1 : 0;
        if (params.use_c128) {
            params.c128_ape = args.c128_ape->data_ptr<float>();
            params.c128_norm_weight =
                reinterpret_cast<const cutlass::bfloat16_t*>(
                    args.c128_norm_weight->data_ptr());
            params.c128_freqs_real = args.c128_freqs_real->data_ptr<float>();
            params.c128_out_loc = args.c128_out_loc->data_ptr<int>();
            set_c128_page_geometry(params, *args.c128_pool,
                                   args.c128_stride_kv_block,
                                   args.c128_page_block);
        }
        params.use_c128_attn =
            (args.c128_page_table.has_value() && args.c128_attn_len.has_value())
                ? 1
                : 0;
        if (params.use_c128_attn) {
            if (!params.use_c128)
                set_c128_page_geometry(params, *args.c128_pool,
                                       args.c128_stride_kv_block,
                                       args.c128_page_block);
            params.c128_page_table_stride =
                (int)args.c128_page_table->stride(0);
            params.c128_page_table = args.c128_page_table->data_ptr<int>();
            params.c128_attn_len = args.c128_attn_len->data_ptr<int>();
        }
        const int64_t kv_tma_rows = (int64_t)args.kv_cache.size(0) *
                                    (params.stride_kv_block / Dims::KvRowBytes);
        make_kv_maps<KT::Attn::KvDenseLevels>(
            args.kv_cache.data_ptr(), kv_tma_rows, params.tensor_map_kv_nope,
            params.kv_nope_dense, params.tensor_map_kv_rope,
            params.kv_rope_dense);
        const int64_t c128_page_bytes = params.use_c128_attn
                                            ? params.c128_stride_kv_block
                                            : params.stride_kv_block;
        make_kv_maps<KT::Attn::KvDenseLevels>(
            params.use_c128_attn ? args.c128_pool->data_ptr()
                                 : args.kv_cache.data_ptr(),
            params.use_c128_attn
                ? (int64_t)(args.c128_pool->numel() / c128_page_bytes) *
                      (c128_page_bytes / Dims::KvRowBytes)
                : kv_tma_rows,
            params.c128_nope, params.c128_nope_dense, params.c128_rope,
            params.c128_rope_dense);
    }

    template <class Params>
    static void set_prelude_args(Params& params,
                                 const MlaAttentionArguments& args,
                                 const MlaAttentionProblem<KT>& prob) {
        using GemmGeomT = typename KT::GemmBase;
        const bool use_kv_gate = args.wkv_gate_weight.has_value();
        params.q_eps = (float)args.q_eps;
        params.kv_eps = (float)args.kv_eps;
        params.shape_wq_a =
            make_gemm_shape<CfgQkv::WeightTile, CfgQkv::BK, CfgQkv::KSplit>(
                prob.qlora_rank, prob.hidden);
        params.shape_wkv =
            make_gemm_shape<CfgQkv::WeightTile, CfgQkv::BK, CfgQkv::KSplit>(
                prob.kvlora_rank, prob.hidden);
        params.shape_wq_b =
            make_gemm_shape<CfgWqB::WeightTile, CfgWqB::BK, CfgWqB::KSplit>(
                Dims::QTotal, prob.qlora_rank);
        params.x_input = prob.hidden_ptr;
        params.hidden_row_stride = prob.hidden_row_stride;
        params.q_norm_weight = reinterpret_cast<const cutlass::bfloat16_t*>(
            args.q_norm_weight.data_ptr());
        params.kv_norm_weight = reinterpret_cast<const cutlass::bfloat16_t*>(
            args.kv_norm_weight.data_ptr());
        params.swa_loc = args.swa_loc.data_ptr<int>();
        params.use_kv_gate = use_kv_gate ? 1 : 0;
        if (use_kv_gate) {
            params.kv_score_ring =
                reinterpret_cast<float*>(args.kv_score_ring->data_ptr());
            params.kv_gate_ring_row =
                reinterpret_cast<const int*>(args.kv_gate_ring_row->data_ptr());
        }
        params.gemm_wq_a = make_weight_atoms<typename KT::Qkv>(
            args.wq_a_weight.data_ptr(), args.wq_a_sf.data_ptr(),
            params.shape_wq_a.w_out, params.shape_wq_a.k_dim);
        params.gemm_wq_b = make_weight_atoms<GemmGeomT>(
            args.wq_b_weight.data_ptr(), args.wq_b_sf.data_ptr(),
            params.shape_wq_b.w_out, params.shape_wq_b.k_dim);
        params.gemm_wkv = make_weight_atoms<typename KT::Qkv>(
            args.wkv_weight.data_ptr(), args.wkv_sf.data_ptr(),
            params.shape_wkv.w_out, params.shape_wkv.k_dim);
        params.ws_x_fp8 = prob.ws_x_fp8;
        params.ws_x_sf = prob.ws_x_sf;
        params.ws_qlora_part = prob.ws_qlora_part;
        params.ws_qlora_fp8 = prob.ws_qlora_fp8;
        params.ws_qlora_sf = prob.ws_qlora_sf;
        params.ws_kv_part = prob.ws_kv_part;
        params.ws_c128_tok = prob.ws_c128_tok;
        params.c128_flags = prob.c128_flags;
        params.act_x = make_act_atoms<GemmGeomT>(
            params.ws_x_fp8, params.ws_x_sf, params.shape_wq_a.w_out,
            prob.t_pad, prob.hidden, prob.t_pad_sf);
        params.act_qlora = make_act_atoms<GemmGeomT>(
            params.ws_qlora_fp8, params.ws_qlora_sf, params.shape_wq_b.w_out,
            prob.t_pad, ModelDims::QLoraRank, prob.t_pad_sf);
    }

    template <class Ctx>
    CUTE_DEVICE void mma(Ctx const& ctx) const {
        {
            const PipelineOffsets led = ctx.sched.ledger_before_qkv();
            auto op = make_mma_op<CfgQkv>(ctx.prelude.res_qkv, led.base_tile(),
                                          led.base_acc(), ctx.sched.k_tps_qkv);
            mma_items(
                op, blockIdx.x, ctx.sched.items_qkv, gridDim.x,
                ctx.sched.k_tps_qkv, [&](int item) {
                    op.begin_item(ctx.prelude.qkv_item(ctx.sched, item).wi);
                });
        }
        {
            const PipelineOffsets led = ctx.sched.ledger_before_wq_b();
            auto op = make_mma_op<CfgWqB>(ctx.prelude.res_wqb, led.base_tile(),
                                          led.base_acc(), ctx.sched.k_tps_wqb);
            mma_items(
                op, ctx.sched.wqb_first(), ctx.sched.items_wq_b,
                ctx.sched.wqb_step(), ctx.sched.k_tps_wqb, [&](int item) {
                    op.begin_item(ctx.prelude.wqb_item(ctx.sched, item).wi);
                });
        }
        {
            const PipelineOffsets led = ctx.sched.ledger_before_kv_gate();
            auto op = make_bf16_mma_op<CfgKvGate>(
                ctx.prelude.res_kv_gate, led.base_tile(), led.base_acc());
            mma_items(op, ctx.sched.kvg_first(), ctx.sched.items_kv_gate,
                      ctx.sched.kvg_step(), ctx.sched.k_tiles_kvg,
                      [&](int) { op.begin_item(); });
        }

        cta_sync();

        if (cute::elect_one_sync()) ctx.plan.init_gemm_bars();
        cutlass::arch::fence_barrier_init();
        bar_sync<KernelBase::Bars::EveryoneSync>();
    }

    template <bool OpA, class Cfg, class Res, class Gate>
    CUTE_DEVICE static auto make_tma_op_ab(Res const& res, int base_tile,
                                           int k_tps, Gate gate) {
        if constexpr (OpA)
            return make_tma_op_a<Cfg>(res, base_tile, k_tps, gate);
        else
            return make_tma_op_b<Cfg>(res, base_tile, k_tps, gate);
    }

    template <bool OpA, class Ctx>
    CUTE_DEVICE void tma_prelude(Ctx const& ctx) const {
        {
            const PipelineOffsets led = ctx.sched.ledger_before_qkv();
            auto gx = [&](int tg) {
                if (!ctx.params.prequant_x)
                    ctx.bars.x_ready(tg).wait_warp_for_tma(0);
            };
            auto op = make_tma_op_ab<OpA, CfgQkv>(
                ctx.prelude.res_qkv, led.base_tile(), ctx.sched.k_tps_qkv, gx);
            issue_items(op, blockIdx.x, ctx.sched.items_qkv, gridDim.x,
                        ctx.sched.k_tps_qkv, [&](int item) {
                            const auto it =
                                ctx.prelude.qkv_item(ctx.sched, item);
                            op.begin_item(*it.weight, it.wi);
                        });
        }
        {
            const PipelineOffsets led = ctx.sched.ledger_before_wq_b();
            auto gq = [&](int tg) {
                ctx.bars.qq_ready(tg).wait_warp_for_tma(0);
            };
            auto op = make_tma_op_ab<OpA, CfgWqB>(
                ctx.prelude.res_wqb, led.base_tile(), ctx.sched.k_tps_wqb, gq);
            issue_items(
                op, ctx.sched.wqb_first(), ctx.sched.items_wq_b,
                ctx.sched.wqb_step(), ctx.sched.k_tps_wqb, [&](int item) {
                    const auto it = ctx.prelude.wqb_item(ctx.sched, item);
                    op.begin_item(*it.weight, it.wi);
                });
        }
        {
            const PipelineOffsets led = ctx.sched.ledger_before_kv_gate();
            auto op = [&] {
                if constexpr (OpA)
                    return make_bf16_weight_tma_op<CfgKvGate>(
                        ctx.prelude.res_kv_gate, led.base_tile());
                else
                    return make_bf16_act_tma_op<CfgKvGate>(
                        ctx.prelude.res_kv_gate, led.base_tile());
            }();
            issue_items(
                op, ctx.sched.kvg_first(), ctx.sched.items_kv_gate,
                ctx.sched.kvg_step(), ctx.sched.k_tiles_kvg, [&](int item) {
                    op.begin_item(bf16_gemm_item_at<CfgKvGate>(
                        item, ctx.sched.n_groups, ctx.sched.num_tokens));
                });
        }
        phase_end_sync();
    }

    template <class Tok>
    CUTE_DEVICE void attn_q_to_tmem(Tok const& tok) const {
        constexpr int kCopyCols =
            256 / cute::sizeof_bits_v<cutlass::bfloat16_t>;
        constexpr int kCopyTmemCols = 256 / 32;
        constexpr int kDescUnit = 16;
        constexpr int kSlabElems = 2 * Dims::Heads * AttnConfig::Sw128Cols;
        cute::UMMA::SmemDescriptor sQ_desc =
            cute::UMMA::make_umma_desc<cute::UMMA::Major::K>(cute::make_tensor(
                cute::make_smem_ptr(AttnSchedule::q_smem(tok)),
                cute::tile_to_shape(
                    cute::UMMA::Layout_K_SW128_Atom<cutlass::bfloat16_t>{},
                    cute::Shape<cute::Int<2 * Dims::Heads>,
                                cute::Int<AttnConfig::Sw128Cols>>{})));
        CUTE_UNROLL
        for (int slab = 0; slab < Dims::DimQ / (2 * AttnConfig::Sw128Cols);
             ++slab) {
            CUTE_UNROLL
            for (int sub = 0; sub < AttnConfig::Sw128Cols / kCopyCols; ++sub) {
                cute::SM100_UTCCP_128dp256bit_1cta::copy(
                    sQ_desc + (slab * kSlabElems + sub * kCopyCols) *
                                  (int)sizeof(cutlass::bfloat16_t) / kDescUnit,
                    AttnConfig::TmemCols::Q +
                        (slab * (AttnConfig::Sw128Cols / kCopyCols) + sub) *
                            kCopyTmemCols);
            }
        }
    }

    template <class Ctx>
    CUTE_DEVICE void attn_mma(Ctx const& ctx) const {
        attn_q_to_tmem(ctx);
        umma_arrive_noelect(ctx.plan.bar_q_utccp);

        cute::TiledMMA tiled_mma_P = AttnConfig::TiledMMA_P{};
        cute::TiledMMA tiled_mma_O = AttnConfig::TiledMMA_O{};
        cute::Tensor tP = cute::partition_fragment_C(
            tiled_mma_P, cute::Shape<cute::Int<ModelDims::Heads>,
                                     cute::Int<AttnConfig::KvBlock * 2>>{});
        cute::Tensor tO = cute::partition_fragment_C(
            tiled_mma_O, cute::Shape<cute::Int<ModelDims::Heads>,
                                     cute::Int<ModelDims::DimV>>{});
        tP.data().get() = AttnConfig::TmemCols::P;
        tO.data().get() = AttnConfig::TmemCols::O;

        ctx.plan.bar_q_utccp.wait(ctx.q_utccp_phase);
        ctx.q_utccp_phase ^= 1;
        tcgen05_after_thread_sync();

        auto rq = ctx.rs;
        auto issue_qk = [&]() {
            ctx.plan.bar_rope_ready[rq.buf_idx].wait(rq.bar_phase);
            ctx.plan.bar_nope_ready[rq.buf_idx].wait(rq.bar_phase);
            tcgen05_after_thread_sync();

            cute::Tensor tQ =
                tiled_mma_P.get_slice(cute::_0{})
                    .make_fragment_A(cute::partition_shape_A(
                        tiled_mma_P,
                        cute::Shape<cute::Int<ModelDims::Heads>,
                                    cute::Int<ModelDims::DimQ / 2>>{}));
            tQ.data().get() = AttnConfig::TmemCols::Q;
            cute::Tensor sK = cute::make_tensor(
                cute::make_smem_ptr(
                    ctx.plan.bufs.kv.dequant[rq.buf_idx].nope.data()),
                AttnConfig::SmemLayoutKTiles_DualGemm_SW128<
                    Dims::DimQ / AttnConfig::Sw128Cols / 2>{});
            tP.data().get() = AttnConfig::TmemCols::P +
                              rq.buf_idx * AttnConfig::TmemCols::PStride;
            umma_ts(tiled_mma_P, tQ, sK, tP, true);
            umma_arrive_noelect(ctx.plan.bar_qk_done[rq.buf_idx]);
            rq.update();
        };

        issue_qk();
        CUTE_NO_UNROLL
        for (int block_idx = ctx.blk_lo; block_idx < ctx.blk_hi; ++block_idx) {
            bool qk_pending = block_idx + 1 < ctx.blk_hi;
            if (qk_pending) {
                CUTE_NO_UNROLL
                while (true) {
                    if (ctx.plan.bar_rope_ready[rq.buf_idx].try_wait(
                            rq.bar_phase) &&
                        ctx.plan.bar_nope_ready[rq.buf_idx].try_wait(
                            rq.bar_phase)) {
                        issue_qk();
                        qk_pending = false;
                        break;
                    }
                    if (ctx.plan.bar_so_ready[ctx.rs.buf_idx].try_wait(
                            ctx.rs.bar_phase))
                        break;
                }
            }

            ctx.plan.bar_so_ready[ctx.rs.buf_idx].wait(ctx.rs.bar_phase);
            tcgen05_after_thread_sync();
            cute::Tensor sS = cute::make_tensor(
                cute::make_smem_ptr(ctx.plan.bufs.kv.s[ctx.rs.buf_idx].data()),
                AttnConfig::SmemLayoutS{});
            cute::Tensor sV = cute::make_tensor(
                cute::make_smem_ptr(
                    ctx.plan.bufs.kv.dequant[ctx.rs.buf_idx].nope.data()),
                AttnConfig::SmemLayoutKTilesTransposed_SW128<
                    Dims::DimV / AttnConfig::Sw128Cols>{});
            umma_ss(tiled_mma_O, sS, sV, tO, block_idx == ctx.blk_lo);
            umma_arrive_noelect(ctx.plan.bar_sv_done[ctx.rs.buf_idx]);
            if (qk_pending) issue_qk();

            ctx.rs.update();
        }
    }

    template <class Ctx, class Mid>
    CUTE_DEVICE void attn_tma12(Ctx const& ctx, int half, Mid&& mid) const {
        const int kv_start = AttnSchedule::kv_start(ctx);
        CUTE_NO_UNROLL
        for (int block_idx = ctx.blk_lo;; ++block_idx) {
            if (block_idx == kv_start) {
                mid();
                ctx.q_utccp_phase ^= 1;
            }
            if (block_idx >= ctx.blk_hi) break;
            ctx.plan.bar_raw_free[ctx.rs.buf_idx].wait(ctx.rs.bar_phase ^ 1);
            auto& bar = half == 0 ? ctx.plan.bar_raw_ready[ctx.rs.buf_idx]
                                  : ctx.plan.bar_raw_ready2[ctx.rs.buf_idx];
            issue_kv_span<AttnConfig::KvBlock / 2>(
                ctx.kv_nope_desc(block_idx), ctx.kv_nope_gather(block_idx), bar,
                (uint8_t*)ctx.plan.bufs.kv.raw_nope[ctx.rs.buf_idx].data(),
                ModelDims::DimNope * sizeof(cutlass::float_e4m3_t),
                ctx.kv_tile_at(block_idx), 0, half * (AttnConfig::KvBlock / 2));

            __syncwarp();
            if (cute::elect_one_sync())
                bar.arrive_and_expect_tx(AttnConfig::KvBlock / 2 *
                                         ModelDims::DimNope *
                                         sizeof(cutlass::float_e4m3_t));
            ctx.rs.update();
        }
    }

    template <class Ctx>
    CUTE_DEVICE void attn_rope_gather_at(Ctx const& ctx, int block_idx,
                                         int buf) const {
        issue_kv_span<AttnConfig::KvBlock>(
            ctx.kv_rope_desc(block_idx), ctx.kv_rope_gather(block_idx),
            ctx.plan.bar_rope_staged[buf], AttnSchedule::rope_stage(ctx, buf),
            ModelDims::DimRope * sizeof(cutlass::bfloat16_t),
            ctx.kv_tile_at(block_idx), 0, 0);
        __syncwarp();
        if (cute::elect_one_sync())
            ctx.plan.bar_rope_staged[buf].arrive_and_expect_tx(
                AttnConfig::KvBlock * ModelDims::DimRope *
                sizeof(cutlass::bfloat16_t));
    }

    template <class Ctx>
    CUTE_DEVICE void attn_rope_copy(Ctx const& ctx) const {
        static constexpr int kVecs = AttnConfig::KvBlock * ModelDims::DimRope *
                                     sizeof(cutlass::bfloat16_t) /
                                     sizeof(uint4);
        static constexpr int kPerLane = kVecs / KernelBase::WarpSize;
        const uint4* __restrict__ src = reinterpret_cast<const uint4*>(
            AttnSchedule::rope_stage(ctx, ctx.rs.buf_idx));
        uint4* __restrict__ dst = reinterpret_cast<uint4*>(
            ctx.plan.bufs.kv.dequant[ctx.rs.buf_idx].rope.data());
        const int lane = ctx.lane_idx;
        CUTE_UNROLL
        for (int blk = 0; blk < kPerLane / 4; ++blk) {
            uint4 vecs[4];
            CUTE_UNROLL
            for (int j = 0; j < 4; ++j)
                vecs[j] = src[(blk * 4 + j) * KernelBase::WarpSize + lane];
            CUTE_UNROLL
            for (int j = 0; j < 4; ++j)
                dst[(blk * 4 + j) * KernelBase::WarpSize + lane] = vecs[j];
        }
        cutlass::arch::fence_view_async_shared();
        __syncwarp();
        if (cute::elect_one_sync())
            ctx.plan.bar_rope_ready[ctx.rs.buf_idx].arrive();
    }

    template <class Ctx, class Mid>
    CUTE_DEVICE void attn_tma3(Ctx const& ctx, Mid&& mid) const {
        auto prefetch = [&] {
            const int n_blocks =
                min(AttnSchedule::kRopeDepth, ctx.blk_hi - ctx.blk_lo);
            CUTE_NO_UNROLL
            for (int k = 0; k < n_blocks; ++k)
                attn_rope_gather_at(ctx, ctx.blk_lo + k,
                                    (ctx.rs.buf_idx + k) % AttnConfig::NumBufs);
        };
        auto wait_q = [&] {
            if (cute::elect_one_sync())
                ctx.plan.bar_q_utccp.wait(ctx.q_utccp_phase);
            __syncwarp();
            ctx.q_utccp_phase ^= 1;
        };
        prefetch();
        mid();
        CUTE_NO_UNROLL
        for (int block_idx = ctx.blk_lo; block_idx < ctx.blk_hi; ++block_idx) {
            AttnSchedule::wait_q_at(ctx, block_idx, wait_q);
            if (cute::elect_one_sync()) {
                ctx.plan.bar_rope_staged[ctx.rs.buf_idx].wait(ctx.rs.bar_phase);
                ctx.plan.bar_sv_done[ctx.rs.buf_idx].wait(ctx.rs.bar_phase ^ 1);
            }
            __syncwarp();
            attn_rope_copy(ctx);
            if (block_idx + AttnSchedule::kRopeDepth < ctx.blk_hi)
                attn_rope_gather_at(
                    ctx, block_idx + AttnSchedule::kRopeDepth,
                    (ctx.rs.buf_idx + AttnSchedule::kRopeDepth) %
                        AttnConfig::NumBufs);
            ctx.rs.update();
        }
        AttnSchedule::wait_q_after(ctx, [&] { ctx.q_utccp_phase ^= 1; });
    }

    template <class Ctx, class Tok>
    CUTE_DEVICE void kv_cache_gate(Ctx const& ctx, Tok const& tok) const {
        if (ctx.warp_idx == 0) kv_gate_wait_warp(ctx, tok);
        bar_sync<KernelBase::Bars::Wg0Sync>();
        fence_proxy_async_global();
    }

    template <class Ctx, class Tok, class Attn>
    CUTE_DEVICE void token_attention(Ctx const& ctx, Tok const& tok,
                                     Attn&& attn) const {
        kv_cache_gate(ctx, tok);
        attn([&] { phase1_prepare_q(ctx, tok); });
        bar_sync<KernelBase::Bars::Wg0Sync>();
    }

    template <class Ctx>
    CUTE_DEVICE void token_loop_mma(Ctx const& ctx) const {
        phase1_qnorm_slice(ctx);
        for_each_token(ctx, [&](auto const& tok) {
            if (!tok.token_active) return;
            auto attn = [&](auto&& mid) {
                mid();
                const bool leader = cute::elect_one_sync();
                if (leader) attn_mma(tok);
                __syncwarp();
                const int src = __ffs(__ballot_sync(0xffffffffu, leader)) - 1;
                ctx.rs.buf_idx = __shfl_sync(0xffffffffu, ctx.rs.buf_idx, src);
                ctx.rs.bar_phase =
                    __shfl_sync(0xffffffffu, (int)ctx.rs.bar_phase, src);
                ctx.q_utccp_phase =
                    __shfl_sync(0xffffffffu, ctx.q_utccp_phase, src);
            };
            token_attention(ctx, tok, attn);
        });
    }

    template <class Ctx>
    CUTE_DEVICE void token_loop_tma12(Ctx const& ctx, int half) const {
        phase1_qnorm_slice(ctx);
        for_each_token(ctx, [&](auto const& tok) {
            if (!tok.token_active) return;
            token_attention(ctx, tok,
                            [&](auto&& mid) { attn_tma12(tok, half, mid); });
        });
    }

    template <class Ctx>
    CUTE_DEVICE void token_loop_tma3(Ctx const& ctx) const {
        phase1_qnorm_slice(ctx);
        for_each_token(ctx, [&](auto const& tok) {
            if (!tok.token_active) return;
            token_attention(ctx, tok, [&](auto&& mid) { attn_tma3(tok, mid); });
        });
    }

    template <class Ctx>
    CUTE_DEVICE void role_mma(Ctx const& ctx) const {
        mma(ctx);
        token_loop_mma(ctx);
    }

    template <class Ctx>
    CUTE_DEVICE void role_tma12(Ctx const& ctx, int half) const {
        if (half == 0)
            tma_prelude<true>(ctx);
        else
            tma_prelude<false>(ctx);
        token_loop_tma12(ctx, half);
    }

    template <class Ctx>
    CUTE_DEVICE void role_tma3(Ctx const& ctx) const {
        phase_end_sync();
        token_loop_tma3(ctx);
        __threadfence();
    }
};
}  // namespace halops::mla
