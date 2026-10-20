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

#include "mla_epilogue.cuh"
#include "mla_scheduler.cuh"

namespace halops {

template <class KT, class MP, class EP, class SP>
__global__ void __launch_bounds__(KT::NumThreads, 1, 1)
    mega_mla_kernel(__grid_constant__ const MP mainloop_params,
                    __grid_constant__ const EP epilogue_params,
                    __grid_constant__ const SP scheduler_params) {
    using Attn = typename KT::Attn;
    using CfgQkv = typename KT::Qkv;
    using CfgWqB = typename KT::WqB;
    using CfgKvGate = typename KT::KvGate;
    using CfgWoA = typename KT::WoA;
    using CfgWoB = typename KT::WoB;

    static_assert(
        CfgWoA::BN == CfgWoB::BN && CfgWoA::TmaStages == CfgWoB::TmaStages,
        "both output-projection ranges overlay one TMEM accumulator "
        "and one smem operand union, so their N tile and TMA stage "
        "count must match");
    static_assert(CfgWoB::KSplitMax >= 4,
                  "the dispatch picks the wo_b K-split from {4,2,1}");
    static_assert(CfgQkv::TmaStages >= CfgWqB::TmaStages,
                  "bar_gemm_tma_a is sized from the qkv range; a deeper "
                  "segment would index past it (a hang, not a fault)");
    static_assert(mega_mla::Bf16GemmTypes<CfgKvGate>::SMEM_TOTAL <=
                      (int)sizeof(decltype(SharedMemoryPlan<KT>::u)),
                  "the bf16 kv_gate operands must fit the `u` union");

    using Roles = typename KT::Roles;
    const mega_mla::CollectiveMainloop<KT> main_loop;
    const mega_mla::CollectiveEpilogue<KT> epilogue_loop;

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

    mega_mla::AttnRing rs;
    uint32_t q_tma_phase = 0, q_utccp_phase = 0;

    const auto scheduler = mega_mla::MlaScheduler<KT>::make(scheduler_params);

    char* gemm_smem = plan.gemm_smem();
    const mega_mla::GemmPipelineBars gemm_bars = plan.gemm_bars();

    auto tmem_o =
        mega_mla::make_gemm_tmem<CfgWoA::BN, CfgWoA::TmaStages,
                                 KT::MaxAccStages>(Attn::tmem_cols::O);
    mega_mla::GemmResources<CfgWoA::BN, decltype(tmem_o)> res_wo_a{
        .weight = epilogue_params.gemm_wo_a,
        .act = epilogue_params.act_flat_o,
        .gemm_smem = gemm_smem,
        .tmem = tmem_o,
        .bars = gemm_bars};
    mega_mla::GemmResources<CfgWoB::BN, decltype(tmem_o)> res_wo_b{
        .weight = epilogue_params.gemm_wo_b,
        .act = epilogue_params.act_olora,
        .gemm_smem = gemm_smem,
        .tmem = tmem_o,
        .bars = gemm_bars};
    auto op_ranges = mega_mla::make_out_proj_ranges<CfgWoA, CfgWoB>(
        res_wo_a, res_wo_b, epilogue_params.ws_olora_bf16,
        scheduler.ks_wo_b == 1 ? epilogue_params.final_out
                               : epilogue_params.ws_flat_q_bf16,
        epilogue_params.final_out);

    auto tmem_ph1 =
        mega_mla::make_gemm_tmem<KT::TileT, CfgQkv::TmaStages,
                                 KT::MaxAccStages>(Attn::tmem_cols::O);
    mega_mla::GemmResources<CfgQkv::BN, decltype(tmem_ph1)> res_qkv{
        .weight = mainloop_params.gemm_wq_a,
        .act = mainloop_params.act_x,
        .gemm_smem = gemm_smem,
        .tmem = tmem_ph1,
        .bars = gemm_bars};
    mega_mla::GemmResources<CfgWqB::BN, decltype(tmem_ph1)> res_wqb{
        .weight = mainloop_params.gemm_wq_b,
        .act = mainloop_params.act_qlora,
        .gemm_smem = gemm_smem,
        .tmem = tmem_ph1,
        .bars = gemm_bars};
    mega_mla::Bf16GemmResources<CfgKvGate,
                                decltype(mainloop_params.tma_kv_gate_w),
                                decltype(mainloop_params.tma_kv_gate_act)>
        res_kv_gate{.tma_w = mainloop_params.tma_kv_gate_w,
                    .tma_act = mainloop_params.tma_kv_gate_act,
                    .gemm_smem = gemm_smem,
                    .bars = gemm_bars,
                    .tmem_base = Attn::tmem_cols::O,
                    .act_rows = mainloop_params.kv_gate_act_rows,
                    .m_total = mainloop_params.shape_kv_gate.m_total,
                    .k_dim = mainloop_params.shape_kv_gate.k_dim,
                    .k_tiles = mainloop_params.shape_kv_gate.k_tiles,
                    .ring = mainloop_params.kv_score_ring,
                    .ring_rows = mainloop_params.kv_gate_ring_row};
    auto ranges = mega_mla::make_mla_gemm_ranges<CfgQkv, CfgWqB, CfgKvGate>(
        res_qkv, &mainloop_params.gemm_wkv, res_wqb, res_kv_gate,
        mainloop_params.ws_qlora_part, mainloop_params.ws_kv_part,
        mainloop_params.ws_flat_q_bf16);

    const KernelBars<KT, MP, EP> bars{
        mainloop_params,       epilogue_params,      scheduler.any_split,
        scheduler.o_post_flat, scheduler.n_groups(), scheduler.ks_wo_b};

    using Ctx = KernelPipelines<KT, MP, EP, SharedMemoryPlan<KT>,
                                decltype(bars), decltype(scheduler),
                                decltype(ranges), decltype(op_ranges)>;
    const Ctx c{mainloop_params,
                epilogue_params,
                plan,
                rs,
                q_tma_phase,
                q_utccp_phase,
                warp_idx,
                (int)(threadIdx.x % KT::WarpSize),
                (int)(threadIdx.x % KT::ThreadsPerWarpGroup),
                (int)cutlass::canonical_warp_group_idx(),
                scheduler.token_iters(),
                scheduler,
                ranges,
                op_ranges,
                bars};

    if (Roles::is_mma(warp_idx)) {
        // cutlass::arch::warpgroup_reg_dealloc<96>();
        main_loop.role_mma(c);
    } else if (Roles::is_tma1(warp_idx)) {
        // cutlass::arch::warpgroup_reg_dealloc<96>();
        main_loop.role_tma1(c);
    } else if (Roles::is_tma2(warp_idx)) {
        // cutlass::arch::warpgroup_reg_dealloc<96>();
        main_loop.role_tma2(c);
    } else if (Roles::is_tma3(warp_idx)) {
        // cutlass::arch::warpgroup_reg_dealloc<96>();
        main_loop.role_tma3(c);
    } else if (Roles::is_drain(warp_idx)) {
        // cutlass::arch::warpgroup_reg_alloc<208>();
        epilogue_loop.role_drain(c);
    } else if (Roles::is_vec(warp_idx)) {
        // cutlass::arch::warpgroup_reg_alloc<200>();
        epilogue_loop.role_vec(c);
    }

    if (warp_idx == 0) plan.release_tmem();

    int prev = 0;
    __syncwarp();
    mega_mla::bar_sync<typename KT::Bars::EveryoneSync>();
    if (threadIdx.x == 0)
        prev = ku::gmem_atom_add1_acq_rel(mainloop_params.grid_bar);
    if (__syncthreads_or(threadIdx.x == 0 && prev == (int)gridDim.x - 1)) {
        bars.reset_all();
        if (threadIdx.x == 0)
            ku::gmem_st_zero_relaxed(mainloop_params.grid_bar);
    }
}

}  // namespace halops
