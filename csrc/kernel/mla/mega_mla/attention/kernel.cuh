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

#include "kernel/mla/mega_mla/attention/config.cuh"
#include "kernel/mla/mega_mla/attention/context.cuh"
#include "kernel/mla/mega_mla/attention/kernel_epilogue.cuh"
#include "kernel/mla/mega_mla/attention/kernel_mainloop.cuh"
#include "kernel/mla/mega_mla/attention/scheduler.cuh"

namespace halops::mla {
/// Dispatches each warp to its role after rebalancing its register budget.
template <class Ctx>
CUTE_DEVICE void mla_attn_run_role(const Ctx& ctx) {
    using KT = typename Ctx::KT;
    using Roles = typename KT::Roles;
    const int warp_idx = ctx.warp_idx;
    if (Roles::is_mma(warp_idx)) {
        cutlass::arch::warpgroup_reg_dealloc<Roles::RegsWg0>();
        mla_attn_role_mma(ctx);
    } else if (Roles::is_tma1(warp_idx) || Roles::is_tma2(warp_idx)) {
        cutlass::arch::warpgroup_reg_dealloc<Roles::RegsWg0>();
        mla_attn_role_tma12(ctx, warp_idx - Roles::Tma1Warp);
    } else if (Roles::is_tma3(warp_idx)) {
        cutlass::arch::warpgroup_reg_dealloc<Roles::RegsWg0>();
        mla_attn_role_tma3(ctx);
    } else if (Roles::is_drain(warp_idx)) {
        cutlass::arch::warpgroup_reg_alloc<Roles::RegsDrain>();
        mla_attn_role_drain(ctx);
    } else if (Roles::is_vec(warp_idx)) {
        cutlass::arch::warpgroup_reg_dealloc<Roles::RegsVec>();
        mla_attn_role_vec(ctx);
    }
}

/// Mega MLA attention kernel: prelude GEMMs, kv write and sparse attention.
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
    mla_attn_run_role(ctx);

    if (warp_idx == 0) plan.release_tmem();
    if (mainloop_params.reset_barriers) bars.reset_after_grid();
}
}  // namespace halops::mla
