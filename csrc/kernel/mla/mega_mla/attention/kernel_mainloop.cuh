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

#include "kernel/mla/mega_mla/attention/config.cuh"
#include "kernel/mla/mega_mla/attention/context.cuh"
#include "kernel/mla/mega_mla/attention/scheduler.cuh"
#include "kernel/mla/mega_mla/attention/traits.cuh"
#include "kernel/mla/mega_mla/mla_gemm.cuh"
#include "kernel/mla/mega_mla/mla_utils.cuh"

namespace halops::mla {
/// Distributed q RMS-norm + rope of this CTA's slice of q heads.
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

/// Waits until every kv row the token attends to has been written.
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

template <class Ctx, class Body>
CUTE_DEVICE void for_each_token(Ctx const& ctx, Body&& body) {
    for (int token_it = 0; token_it < ctx.token_iters; ++token_it) {
        body(make_per_token(ctx, token_it));
        cta_sync();
        if (token_it + 1 == ctx.token_iters) cta_sync();
    }
}

/// Stages the normalised q of one token into smem for the QK^T UMMA.
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

/// Issues the UMMAs of the qkv, wq_b and kv-gate prelude GEMMs.
template <class Ctx>
CUTE_DEVICE void mla_attn_gemm_mma(Ctx const& ctx) {
    using KT = typename Ctx::KT;
    using CfgQkv = typename KT::Qkv;
    using CfgWqB = typename KT::WqB;
    using CfgKvGate = typename KT::KvGate;
    {
        const PipelineOffsets led = ctx.sched.ledger_before_qkv();
        auto op = make_mma_op<CfgQkv>(ctx.prelude.res_qkv, led.base_tile(),
                                      led.base_acc(), ctx.sched.k_tps_qkv);
        mma_items(op, blockIdx.x, ctx.sched.items_qkv, gridDim.x,
                  ctx.sched.k_tps_qkv, [&](int item) {
                      op.begin_item(ctx.prelude.qkv_item(ctx.sched, item).wi);
                  });
    }
    {
        const PipelineOffsets led = ctx.sched.ledger_before_wq_b();
        auto op = make_mma_op<CfgWqB>(ctx.prelude.res_wqb, led.base_tile(),
                                      led.base_acc(), ctx.sched.k_tps_wqb);
        mma_items(op, ctx.sched.wqb_first(), ctx.sched.items_wq_b,
                  ctx.sched.wqb_step(), ctx.sched.k_tps_wqb, [&](int item) {
                      op.begin_item(ctx.prelude.wqb_item(ctx.sched, item).wi);
                  });
    }
    {
        const PipelineOffsets led = ctx.sched.ledger_before_kv_gate();
        auto op = make_bf16_mma_op<CfgKvGate>(ctx.prelude.res_kv_gate,
                                              led.base_tile(), led.base_acc());
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
CUTE_DEVICE auto mla_attn_make_tma_op_ab(Res const& res, int base_tile,
                                         int k_tps, Gate gate) {
    if constexpr (OpA)
        return make_tma_op_a<Cfg>(res, base_tile, k_tps, gate);
    else
        return make_tma_op_b<Cfg>(res, base_tile, k_tps, gate);
}

/// Issues the operand-A or operand-B TMA loads of the prelude GEMMs.
template <bool OpA, class Ctx>
CUTE_DEVICE void mla_attn_gemm_tma(Ctx const& ctx) {
    using KT = typename Ctx::KT;
    using CfgQkv = typename KT::Qkv;
    using CfgWqB = typename KT::WqB;
    using CfgKvGate = typename KT::KvGate;
    {
        const PipelineOffsets led = ctx.sched.ledger_before_qkv();
        auto gx = [&](int tg) {
            if (!ctx.params.prequant_x)
                ctx.bars.x_ready(tg).wait_warp_for_tma(0);
        };
        auto op = mla_attn_make_tma_op_ab<OpA, CfgQkv>(
            ctx.prelude.res_qkv, led.base_tile(), ctx.sched.k_tps_qkv, gx);
        issue_items(op, blockIdx.x, ctx.sched.items_qkv, gridDim.x,
                    ctx.sched.k_tps_qkv, [&](int item) {
                        const auto it = ctx.prelude.qkv_item(ctx.sched, item);
                        op.begin_item(*it.weight, it.wi);
                    });
    }
    {
        const PipelineOffsets led = ctx.sched.ledger_before_wq_b();
        auto gq = [&](int tg) { ctx.bars.qq_ready(tg).wait_warp_for_tma(0); };
        auto op = mla_attn_make_tma_op_ab<OpA, CfgWqB>(
            ctx.prelude.res_wqb, led.base_tile(), ctx.sched.k_tps_wqb, gq);
        issue_items(op, ctx.sched.wqb_first(), ctx.sched.items_wq_b,
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
                return make_bf16_act_tma_op<CfgKvGate>(ctx.prelude.res_kv_gate,
                                                       led.base_tile());
        }();
        issue_items(op, ctx.sched.kvg_first(), ctx.sched.items_kv_gate,
                    ctx.sched.kvg_step(), ctx.sched.k_tiles_kvg, [&](int item) {
                        op.begin_item(bf16_gemm_item_at<CfgKvGate>(
                            item, ctx.sched.n_groups, ctx.sched.num_tokens));
                    });
    }
    phase_end_sync();
}

template <class Tok>
CUTE_DEVICE void mla_attn_q_to_tmem(Tok const& tok) {
    using Dims = ModelDims;
    constexpr int kCopyCols =
        MlaAttnConfig::kUtccpBits / cute::sizeof_bits_v<cutlass::bfloat16_t>;
    constexpr int kCopyTmemCols =
        MlaAttnConfig::kUtccpBits / MlaAttnConfig::kTmemColBits;
    constexpr int kDescUnit = MlaAttnConfig::kSmemDescUnit;
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

/// Leader-thread QK^T and SV UMMA loop over the KV blocks of one token.
template <class Ctx>
CUTE_DEVICE void mla_attn_mma(Ctx const& ctx) {
    using Dims = ModelDims;
    mla_attn_q_to_tmem(ctx);
    umma_arrive_noelect(ctx.plan.bar_q_utccp);

    cute::TiledMMA tiled_mma_P = AttnConfig::TiledMMA_P{};
    cute::TiledMMA tiled_mma_O = AttnConfig::TiledMMA_O{};
    cute::Tensor tP = cute::partition_fragment_C(
        tiled_mma_P, cute::Shape<cute::Int<ModelDims::Heads>,
                                 cute::Int<AttnConfig::KvBlock * 2>>{});
    cute::Tensor tO = cute::partition_fragment_C(
        tiled_mma_O,
        cute::Shape<cute::Int<ModelDims::Heads>, cute::Int<ModelDims::DimV>>{});
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
CUTE_DEVICE void mla_attn_tma12(Ctx const& ctx, int half, Mid&& mid) {
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
CUTE_DEVICE void mla_attn_rope_gather_at(Ctx const& ctx, int block_idx,
                                         int buf) {
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
CUTE_DEVICE void mla_attn_rope_copy(Ctx const& ctx) {
    constexpr int kBatch = MlaAttnConfig::kRopeCopyBatch;
    static constexpr int kVecs = AttnConfig::KvBlock * ModelDims::DimRope *
                                 sizeof(cutlass::bfloat16_t) / sizeof(uint4);
    static constexpr int kPerLane = kVecs / KernelBase::WarpSize;
    const uint4* __restrict__ src = reinterpret_cast<const uint4*>(
        AttnSchedule::rope_stage(ctx, ctx.rs.buf_idx));
    uint4* __restrict__ dst = reinterpret_cast<uint4*>(
        ctx.plan.bufs.kv.dequant[ctx.rs.buf_idx].rope.data());
    const int lane = ctx.lane_idx;
    CUTE_UNROLL
    for (int blk = 0; blk < kPerLane / kBatch; ++blk) {
        uint4 vecs[kBatch];
        CUTE_UNROLL
        for (int j = 0; j < kBatch; ++j)
            vecs[j] = src[(blk * kBatch + j) * KernelBase::WarpSize + lane];
        CUTE_UNROLL
        for (int j = 0; j < kBatch; ++j)
            dst[(blk * kBatch + j) * KernelBase::WarpSize + lane] = vecs[j];
    }
    cutlass::arch::fence_view_async_shared();
    __syncwarp();
    if (cute::elect_one_sync())
        ctx.plan.bar_rope_ready[ctx.rs.buf_idx].arrive();
}

template <class Ctx, class Mid>
CUTE_DEVICE void mla_attn_tma3(Ctx const& ctx, Mid&& mid) {
    auto prefetch = [&] {
        const int n_blocks =
            min(AttnSchedule::kRopeDepth, ctx.blk_hi - ctx.blk_lo);
        CUTE_NO_UNROLL
        for (int k = 0; k < n_blocks; ++k)
            mla_attn_rope_gather_at(ctx, ctx.blk_lo + k,
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
        mla_attn_rope_copy(ctx);
        if (block_idx + AttnSchedule::kRopeDepth < ctx.blk_hi)
            mla_attn_rope_gather_at(
                ctx, block_idx + AttnSchedule::kRopeDepth,
                (ctx.rs.buf_idx + AttnSchedule::kRopeDepth) %
                    AttnConfig::NumBufs);
        ctx.rs.update();
    }
    AttnSchedule::wait_q_after(ctx, [&] { ctx.q_utccp_phase ^= 1; });
}

template <class Ctx, class Tok>
CUTE_DEVICE void mla_attn_kv_cache_gate(Ctx const& ctx, Tok const& tok) {
    if (ctx.warp_idx == 0) kv_gate_wait_warp(ctx, tok);
    bar_sync<KernelBase::Bars::Wg0Sync>();
    fence_proxy_async_global();
}

template <class Ctx, class Tok, class Attn>
CUTE_DEVICE void mla_attn_token(Ctx const& ctx, Tok const& tok, Attn&& attn) {
    mla_attn_kv_cache_gate(ctx, tok);
    attn([&] { phase1_prepare_q(ctx, tok); });
    bar_sync<KernelBase::Bars::Wg0Sync>();
}

template <class Ctx>
CUTE_DEVICE void mla_attn_token_loop_mma(Ctx const& ctx) {
    phase1_qnorm_slice(ctx);
    for_each_token(ctx, [&](auto const& tok) {
        if (!tok.token_active) return;
        auto attn = [&](auto&& mid) {
            mid();
            const bool leader = cute::elect_one_sync();
            if (leader) mla_attn_mma(tok);
            __syncwarp();
            const int src =
                __ffs(__ballot_sync(MlaAttnConfig::kFullMask, leader)) - 1;
            ctx.rs.buf_idx =
                __shfl_sync(MlaAttnConfig::kFullMask, ctx.rs.buf_idx, src);
            ctx.rs.bar_phase = __shfl_sync(MlaAttnConfig::kFullMask,
                                           (int)ctx.rs.bar_phase, src);
            ctx.q_utccp_phase =
                __shfl_sync(MlaAttnConfig::kFullMask, ctx.q_utccp_phase, src);
        };
        mla_attn_token(ctx, tok, attn);
    });
}

template <class Ctx>
CUTE_DEVICE void mla_attn_token_loop_tma12(Ctx const& ctx, int half) {
    phase1_qnorm_slice(ctx);
    for_each_token(ctx, [&](auto const& tok) {
        if (!tok.token_active) return;
        mla_attn_token(ctx, tok,
                       [&](auto&& mid) { mla_attn_tma12(tok, half, mid); });
    });
}

template <class Ctx>
CUTE_DEVICE void mla_attn_token_loop_tma3(Ctx const& ctx) {
    phase1_qnorm_slice(ctx);
    for_each_token(ctx, [&](auto const& tok) {
        if (!tok.token_active) return;
        mla_attn_token(ctx, tok, [&](auto&& mid) { mla_attn_tma3(tok, mid); });
    });
}

/// MMA warp: prelude GEMM UMMAs, then the per-token attention UMMAs.
template <class Ctx>
CUTE_DEVICE void mla_attn_role_mma(Ctx const& ctx) {
    mla_attn_gemm_mma(ctx);
    mla_attn_token_loop_mma(ctx);
}

/// TMA warps 1 and 2: prelude GEMM operand loads, then raw nope KV loads.
template <class Ctx>
CUTE_DEVICE void mla_attn_role_tma12(Ctx const& ctx, int half) {
    if (half == 0)
        mla_attn_gemm_tma<true>(ctx);
    else
        mla_attn_gemm_tma<false>(ctx);
    mla_attn_token_loop_tma12(ctx, half);
}

/// TMA warp 3: rope KV gathers and their smem staging copies.
template <class Ctx>
CUTE_DEVICE void mla_attn_role_tma3(Ctx const& ctx) {
    phase_end_sync();
    mla_attn_token_loop_tma3(ctx);
    __threadfence();
}
}  // namespace halops::mla
