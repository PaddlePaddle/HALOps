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

#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/tmem_allocator_sm100.hpp>

#include "kernel/mla/mla_epilogue.cuh"
#include "kernel/mla/mla_scheduler.cuh"

namespace halops::mla {
template <class KT>
struct SharedMemoryPlan {
    using Attn = typename KT::Attn;
    using Dims = typename KT::Dims;
    using Arena = GemmArena;

    union {
        struct {
            cute::array_aligned<
                cutlass::bfloat16_t,
                cute::cosize_v<typename Attn::SmemLayoutQ_SW128>>
                head;
            cute::array_aligned<cutlass::bfloat16_t,
                                cute::cosize_v<typename Attn::SmemLayoutOBuf>>
                o_buf;
        } qo;
        struct {
            struct {
                cute::array_aligned<cutlass::bfloat16_t,
                                    Dims::Heads * Dims::DimNope>
                    nope;
                cute::array_aligned<cutlass::bfloat16_t,
                                    Dims::Heads * Dims::DimRope>
                    rope;
            } dequant[Attn::NumBufs];
            cute::array_aligned<cutlass::float_e4m3_t,
                                Dims::Heads * Dims::DimNope>
                raw_nope[Attn::NumBufs];

            float4 p_exchange_buf[2][KT::WarpSize * (Attn::KvBlock / 2) / 4];
            cute::array_aligned<cutlass::bfloat16_t,
                                cute::cosize_v<typename Attn::SmemLayoutS>>
                s[Attn::NumBufs];

            cute::array_aligned<cutlass::bfloat16_t,
                                Dims::Heads * Dims::DimRope>
                rope_stage;
        } kv;

        char gemm[Arena::Total];
    } bufs;

    CUTE_ALIGNAS(16)
    cutlass::bfloat16_t rope_stage2[Dims::Heads * Dims::DimRope];

    CUTE_ALIGNAS(16) float wg_exchange[KT::ThreadsPerWarpGroup];

    CUTE_ALIGNAS(16) float c128_warp_sq[KT::Roles::VecWarps];
    int c128_last_slice;
    CUTE_ALIGNAS(16) float kv_write_sq[KT::Roles::VecWarps];

    union {
        CUTE_ALIGNAS(16) __nv_bfloat16 c128_row[Dims::DimV];
        CUTE_ALIGNAS(16) __nv_bfloat16 kv_write_row[Dims::DimV];
    };

    cute::array_aligned<uint32_t, 1> tmem_start_addr;
    int q_head_ctr;
    cutlass::arch::ClusterTransactionBarrier bar_q_tma, bar_q_utccp;
    cutlass::arch::ClusterTransactionBarrier bar_rope_ready[Attn::NumBufs];
    cutlass::arch::ClusterTransactionBarrier bar_rope_staged[Attn::NumBufs];
    cutlass::arch::ClusterTransactionBarrier bar_nope_ready[Attn::NumBufs];
    cutlass::arch::ClusterTransactionBarrier bar_raw_ready[Attn::NumBufs],
        bar_raw_free[Attn::NumBufs];
    cutlass::arch::ClusterTransactionBarrier bar_raw_ready2[Attn::NumBufs];
    cutlass::arch::ClusterTransactionBarrier bar_qk_done[Attn::NumBufs],
        bar_so_ready[Attn::NumBufs], bar_sv_done[Attn::NumBufs];

    cutlass::arch::ClusterTransactionBarrier
        bar_gemm_tma_a[Arena::MaxTmaStages];
    cutlass::arch::ClusterTransactionBarrier
        bar_gemm_tma_b[Arena::MaxTmaStages];
    cutlass::arch::ClusterTransactionBarrier
        bar_gemm_smem_free[Arena::MaxTmaStages];
    cutlass::arch::ClusterTransactionBarrier
        bar_gemm_acc_full[Arena::MaxAccStages];
    cutlass::arch::ClusterTransactionBarrier
        bar_gemm_acc_empty[Arena::MaxAccStages];
    cutlass::arch::ClusterTransactionBarrier
        bar_gemm_acc_win_lo[Arena::MaxAccStages];
    cutlass::arch::ClusterTransactionBarrier
        bar_gemm_acc_win_hi[Arena::MaxAccStages];

    CUTE_DEVICE GemmPipelineBars gemm_bars() {
        return {bar_gemm_tma_a,     bar_gemm_tma_b,     bar_gemm_smem_free,
                bar_gemm_acc_full,  bar_gemm_acc_empty, bar_gemm_acc_win_lo,
                bar_gemm_acc_win_hi};
    }

    CUTE_DEVICE char* gemm_smem() {
        return reinterpret_cast<char*>(bufs.kv.dequant[0].nope.data());
    }

    CUTE_DEVICE void init_gemm_bars() {
        CUTE_UNROLL
        for (int i = 0; i < Arena::MaxTmaStages; ++i) {
            bar_gemm_tma_a[i].init(1);
            bar_gemm_tma_b[i].init(1);
            bar_gemm_smem_free[i].init(1);
        }
        CUTE_UNROLL
        for (int i = 0; i < Arena::MaxAccStages; ++i) {
            bar_gemm_acc_full[i].init(1);
            bar_gemm_acc_empty[i].init(KT::Roles::DrainWarps * KT::WarpSize);
            bar_gemm_acc_win_lo[i].init(KT::Roles::DrainWarps * KT::WarpSize);
            bar_gemm_acc_win_hi[i].init(KT::Roles::DrainWarps * KT::WarpSize);
        }
    }

    CUTE_DEVICE void init_bars() {
        q_head_ctr = 0;
        bar_q_tma.init(1);
        bar_q_utccp.init(1);
        CUTE_UNROLL
        for (int i = 0; i < Attn::NumBufs; ++i) {
            bar_rope_ready[i].init(1);
            bar_rope_staged[i].init(1);
            bar_nope_ready[i].init(KT::ThreadsPerWarpGroup);
            bar_raw_ready[i].init(1);
            bar_raw_ready2[i].init(1);
            bar_raw_free[i].init(KT::ThreadsPerWarpGroup);
            bar_qk_done[i].init(1);
            bar_so_ready[i].init(KT::ThreadsPerWarpGroup);
            bar_sv_done[i].init(1);
        }
        init_gemm_bars();
        cutlass::arch::fence_barrier_init();
    }

    CUTE_DEVICE void acquire_tmem() {
        cute::TMEM::Allocator1Sm().allocate(Attn::TmemCols::Total,
                                            tmem_start_addr.data());
        cute::TMEM::Allocator1Sm().release_allocation_lock();
    }

    CUTE_DEVICE void release_tmem() {
        cute::TMEM::Allocator1Sm().free(0, Attn::TmemCols::Total);
    }
};

template <class KT_, class MP, class EP, class Plan, class Bars, class Sched,
          class PreludeRanges>
struct KernelPipelines {
    using KT = KT_;
    static constexpr int TileT = KT::TileT;

    const MP& params;
    const EP& ep;
    Plan& plan;
    AttnRing& rs;
    uint32_t& q_tma_phase;
    uint32_t& q_utccp_phase;
    int warp_idx;
    int lane_idx;
    int idx_in_warpgroup;
    int warpgroup_idx;

    int token_iters;
    const Sched& sched;
    PreludeRanges prelude;
    Bars bars;
};

template <class KT, class Ctx>
CUTE_DEVICE void run_role(const AttentionMainloop<KT>& main_loop,
                          const AttentionEpilogue<KT>& epilogue_loop,
                          const Ctx& ctx) {
    using Roles = typename KT::Roles;
    const int warp_idx = ctx.warp_idx;
    if (Roles::is_mma(warp_idx)) {
        cutlass::arch::warpgroup_reg_dealloc<Roles::RegsWg0>();
        main_loop.role_mma(ctx);
    } else if (Roles::is_tma1(warp_idx) || Roles::is_tma2(warp_idx)) {
        cutlass::arch::warpgroup_reg_dealloc<Roles::RegsWg0>();
        main_loop.role_tma12(ctx, warp_idx - Roles::Tma1Warp);
    } else if (Roles::is_tma3(warp_idx)) {
        cutlass::arch::warpgroup_reg_dealloc<Roles::RegsWg0>();
        main_loop.role_tma3(ctx);
    } else if (Roles::is_drain(warp_idx)) {
        cutlass::arch::warpgroup_reg_alloc<Roles::RegsDrain>();
        epilogue_loop.role_drain(ctx);
    } else if (Roles::is_vec(warp_idx)) {
        cutlass::arch::warpgroup_reg_dealloc<Roles::RegsVec>();
        epilogue_loop.role_vec(ctx);
    }
}

template <class KT, class MP, class EP, class SP>
__global__ void __launch_bounds__(KT::NumThreads, 1, 1)
    mla_attention_kernel(__grid_constant__ const MP mainloop_params,
                         __grid_constant__ const EP epilogue_params,
                         __grid_constant__ const SP scheduler_params) {
    using Attn = typename KT::Attn;
    using CfgQkv = typename KT::Qkv;
    using CfgWqB = typename KT::WqB;
    using CfgKvGate = typename KT::KvGate;

    pdl_trigger();

    const AttentionMainloop<KT> main_loop;
    const AttentionEpilogue<KT> epilogue_loop;

    extern __shared__ char smem_buf[];
    SharedMemoryPlan<KT>& plan =
        *reinterpret_cast<SharedMemoryPlan<KT>*>(smem_buf);

    const int warp_idx = KT::warp_idx();
    if (warp_idx == 0) {
        if (cute::elect_one_sync()) {
            mainloop_params.prefetch_descriptors();
            plan.init_bars();
        }
        plan.acquire_tmem();
    }
    __syncthreads();

    AttnRing rs;
    uint32_t q_tma_phase = 0, q_utccp_phase = 0;

    pdl_wait();

    const auto scheduler = MlaScheduler<KT>::make(scheduler_params);

    char* gemm_smem = plan.gemm_smem();
    const GemmPipelineBars gemm_bars = plan.gemm_bars();

    auto tmem_qkv = make_gemm_tmem_for<CfgQkv>(Attn::TmemCols::O);
    auto tmem_wqb = make_gemm_tmem_for<CfgWqB>(Attn::TmemCols::O);
    GemmResources<typename CfgQkv::Geom, decltype(tmem_qkv), CfgQkv> res_qkv{
        .weight = mainloop_params.gemm_wq_a,
        .act = mainloop_params.act_x,
        .gemm_smem = gemm_smem,
        .tmem = tmem_qkv,
        .bars = gemm_bars};
    GemmResources<typename CfgWqB::Geom, decltype(tmem_wqb)> res_wqb{
        .weight = mainloop_params.gemm_wq_b,
        .act = mainloop_params.act_qlora,
        .gemm_smem = gemm_smem,
        .tmem = tmem_wqb,
        .bars = gemm_bars};
    Bf16GemmResources<CfgKvGate, decltype(mainloop_params.tma_kv_gate_w),
                      decltype(mainloop_params.tma_kv_gate_act)>
        res_kv_gate{.tma_w = mainloop_params.tma_kv_gate_w,
                    .tma_act = mainloop_params.tma_kv_gate_act,
                    .gemm_smem = gemm_smem,
                    .bars = gemm_bars,
                    .tmem_base = Attn::TmemCols::O,
                    .act_rows = mainloop_params.kv_gate_act_rows,
                    .w_out = mainloop_params.shape_kv_gate.w_out,
                    .k_dim = mainloop_params.shape_kv_gate.k_dim,
                    .k_tiles = mainloop_params.shape_kv_gate.k_tiles,
                    .ring = mainloop_params.kv_score_ring,
                    .ring_rows = mainloop_params.kv_gate_ring_row};
    auto ranges = make_mla_gemm_ranges<CfgQkv, CfgWqB>(
        res_qkv, &mainloop_params.gemm_wkv, res_wqb, res_kv_gate,
        mainloop_params.ws_qlora_part, mainloop_params.ws_kv_part,
        mainloop_params.ws_flat_q_bf16);

    const KernelBars<KT, MP> bars{mainloop_params};

    using Ctx =
        KernelPipelines<KT, MP, EP, SharedMemoryPlan<KT>, decltype(bars),
                        decltype(scheduler), decltype(ranges)>;
    const Ctx ctx{
        .params = mainloop_params,
        .ep = epilogue_params,
        .plan = plan,
        .rs = rs,
        .q_tma_phase = q_tma_phase,
        .q_utccp_phase = q_utccp_phase,
        .warp_idx = warp_idx,
        .lane_idx = (int)(threadIdx.x % KT::WarpSize),
        .idx_in_warpgroup = (int)(threadIdx.x % KT::ThreadsPerWarpGroup),
        .warpgroup_idx = (int)cutlass::canonical_warp_group_idx(),
        .token_iters = scheduler.token_iters(),
        .sched = scheduler,
        .prelude = ranges,
        .bars = bars};
    run_role(main_loop, epilogue_loop, ctx);

    if (warp_idx == 0) plan.release_tmem();
    if (mainloop_params.reset_barriers) bars.reset_after_grid();
}
}  // namespace halops::mla
