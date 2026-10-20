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

#include "mla_gemm.cuh"
#include "mla_params.cuh"

namespace mega_mla {

// Which item a CTA is on and how many items exist.  Every number here is host
// arithmetic over the problem shape and the grid the host itself chose, so the
// kernel reads it instead of having all 12 warps of all 152 CTAs recompute it.
template <class KT>
struct MlaScheduler {
    static constexpr int TileT = KT::TileT;
    using CfgQkv = typename KT::Qkv;
    using CfgWqB = typename KT::WqB;
    using CfgKvGate = typename KT::KvGate;
    using CfgWoA = typename KT::WoA;
    using CfgWoB = typename KT::WoB;
    static constexpr int KS_QKV = CfgQkv::KSplit;
    static constexpr int KS_WOA = CfgWoA::KSplit;
    static constexpr int OLORA_GPP = TraitsN128::WoA::OloraGroupsPerPass;

    struct Params {
        GemmWorkList gemm;
        OutProjWorkList out_proj;
        OutProjDims d;
        int ks_wo_b;
        int m_total_wq_a;
        int m_total_wkv;
        int m_total_wq_b;
        int k_tps_qkv;
        int k_tps_wqb;
        int k_tiles_kvg;
        int k_tps_woa;
        int k_tiles_per_group;
        int k_tps_wob;
        // The plan kernel wrote these; the host never sees the bytes, so the
        // split state is the one thing resolved on the device.
        PlanView wplan;
        int opost_flat_max;
    };

    template <class MP, class EP>
    static Params to_underlying_arguments(
        const halops::MegaMlaProblem<KT>& prob, const MP& mp, const EP& ep) {
        const int n_groups = prob.n_groups;
        const int wo_b_items = n_groups * ep.opdims.m_tiles_all_wob;
        const int ks_wo_b = prob.ks_wo_b;
        const int items_wq_a =
            mp.use_phase1 ? n_groups * mp.shape_wq_a.m_tiles * CfgQkv::KSplit
                          : 0;
        const int items_wkv =
            mp.use_kv_write ? n_groups * mp.shape_wkv.m_tiles * CfgQkv::KSplit
                            : 0;
        Params p{};
        p.gemm = GemmWorkList{
            .items_kv_gate =
                mp.use_kv_gate ? n_groups * mp.shape_kv_gate.m_tiles : 0,
            .items_wq_a = items_wq_a,
            .items_qkv = items_wq_a + items_wkv,
            .items_wq_b = mp.use_phase1 ? n_groups * mp.shape_wq_b.m_tiles : 0,
            .n_groups = n_groups,
            .num_tokens = prob.num_tokens};
        p.out_proj = OutProjWorkList{
            .items_wo_a = n_groups * ep.opdims.m_tiles_all * CfgWoA::KSplit,
            .items_wo_b = wo_b_items * ks_wo_b,
            .n_groups = n_groups,
            .num_tokens = prob.num_tokens};
        p.d = ep.opdims;
        p.ks_wo_b = ks_wo_b;
        p.m_total_wq_a = mp.shape_wq_a.m_total;
        p.m_total_wkv = mp.shape_wkv.m_total;
        p.m_total_wq_b = mp.shape_wq_b.m_total;
        p.k_tps_qkv = mp.shape_wq_a.k_tps;
        p.k_tps_wqb = mp.shape_wq_b.k_tps;
        p.k_tiles_kvg = mp.shape_kv_gate.k_tiles;
        p.k_tps_woa = ep.shape_wo_a.k_tps;
        p.k_tiles_per_group = ep.shape_wo_a.k_tiles;
        p.k_tps_wob = ep.shape_wo_b.k_tps;
        p.wplan = mp.wplan;
        p.opost_flat_max = ep.opost_flat_max;
        return p;
    }

    GemmWorkList gemm;
    OutProjWorkList out_proj;
    OutProjDims d;
    int ks_wo_b;
    int m_total_wq_a;
    int m_total_wkv;
    int m_total_wq_b;
    int k_tps_qkv;
    int k_tps_wqb;
    int k_tiles_kvg;
    int k_tps_woa;
    int k_tiles_per_group;
    int k_tps_wob;
    bool any_split;
    bool o_post_flat;
    int slot_step;

    // A FACTORY, NOT A CONSTRUCTOR.  A user-declared constructor here made nvcc
    // materialise the whole kernel context in local memory, which puts a TMA
    // descriptor in an address space cp.async.bulk.tensor may not read -- a
    // hang with no fault.
    CUTE_DEVICE static MlaScheduler make(Params const& params) {
        MlaScheduler s{params.gemm,
                       params.out_proj,
                       params.d,
                       params.ks_wo_b,
                       params.m_total_wq_a,
                       params.m_total_wkv,
                       params.m_total_wq_b,
                       params.k_tps_qkv,
                       params.k_tps_wqb,
                       params.k_tiles_kvg,
                       params.k_tps_woa,
                       params.k_tiles_per_group,
                       params.k_tps_wob,
                       false,
                       false,
                       1};
        s.any_split = __ldg(&params.wplan.hdr->any_split) != 0;
        s.o_post_flat =
            params.gemm.num_tokens <= params.opost_flat_max || s.any_split;
        s.slot_step = s.any_split ? __ldg(&params.wplan.hdr->slot_step) : 1;
        return s;
    }

    CUTE_DEVICE int num_tokens() const { return gemm.num_tokens; }
    CUTE_DEVICE int n_groups() const { return gemm.n_groups; }
    CUTE_DEVICE int token_iters() const {
        return (gemm.num_tokens + (int)gridDim.x - 1) / (int)gridDim.x;
    }

    CUTE_DEVICE bool qkv_is_q(int item) const { return item < gemm.items_wq_a; }
    CUTE_DEVICE GemmWorkItem qkv_item(int item) const {
        const bool is_q = qkv_is_q(item);
        const int it = is_q ? item : item - gemm.items_wq_a;
        const int m_total = is_q ? m_total_wq_a : m_total_wkv;
        const int tg = it % gemm.n_groups;
        const int rest = it / gemm.n_groups;
        const int ks = rest % KS_QKV;
        const int mt = rest / KS_QKV;
        const int n_base = tg * TileT;
        return GemmWorkItem{
            /*n_row=*/n_base,
            /*m_tile=*/mt,
            /*m_total=*/m_total,
            /*k_tile_begin=*/ks * k_tps_qkv,
            /*a_k_tile_begin=*/ks * k_tps_qkv,
            /*out_row_stride=*/KS_QKV * m_total,
            /*n_valid=*/min((int)TileT, gemm.num_tokens - n_base)};
    }
    CUTE_DEVICE int64_t qkv_out_off(int item) const {
        const bool is_q = qkv_is_q(item);
        const int it = is_q ? item : item - gemm.items_wq_a;
        const int m_total = is_q ? m_total_wq_a : m_total_wkv;
        const int tg = it % gemm.n_groups;
        const int rest = it / gemm.n_groups;
        const int ks = rest % KS_QKV;
        const int mt = rest / KS_QKV;
        return (int64_t)(tg * TileT) * (KS_QKV * m_total) + ks * m_total +
               mt * GemmShapeDefaults::BM;
    }
    CUTE_DEVICE GemmWorkItem wqb_item(int item) const {
        return gemm_item<CfgWqB>(gemm, item, TileT, m_total_wq_b, k_tps_wqb);
    }

    CUTE_DEVICE RangeBarrierLedger ledger_before_kv_gate() const {
        return RangeBarrierLedger{};
    }
    CUTE_DEVICE RangeBarrierLedger ledger_before_qkv() const {
        RangeBarrierLedger led;
        led.contribute<CfgKvGate::TmaStages>(k_tiles_kvg,
                                             n_items(gemm.items_kv_gate));
        return led;
    }
    CUTE_DEVICE RangeBarrierLedger ledger_before_wq_b() const {
        RangeBarrierLedger led = ledger_before_qkv();
        led.contribute<GemmGeom<CfgQkv>::STAGES>(k_tps_qkv,
                                                 n_items(gemm.items_qkv));
        return led;
    }

    CUTE_DEVICE int wo_a_tg(int item) const { return item / d.items_per_tg; }
    CUTE_DEVICE int wo_a_rest(int item) const { return item % d.items_per_tg; }
    CUTE_DEVICE int wo_a_group(int item) const {
        return (wo_a_rest(item) / KS_WOA) / d.m_tiles_per_group;
    }
    CUTE_DEVICE int wo_a_round(int item) const {
        return wo_a_group(item) / OLORA_GPP;
    }
    CUTE_DEVICE int wo_b_tg(int item) const { return item % out_proj.n_groups; }

    CUTE_DEVICE GemmWorkItem wo_a_item(int item) const {
        const int tg = wo_a_tg(item);
        const int rest = wo_a_rest(item);
        const int ks = rest % KS_WOA;
        const int m_tile = rest / KS_WOA;
        const int g = m_tile / d.m_tiles_per_group;
        const int n_base = tg * TileT;
        return GemmWorkItem{
            /*n_row=*/n_base,
            /*m_tile=*/m_tile,
            /*m_total=*/d.m_total_woa,
            /*k_tile_begin=*/g * k_tiles_per_group + ks * k_tps_woa,
            /*a_k_tile_begin=*/ks * k_tps_woa,
            /*out_row_stride=*/KS_WOA * d.m_total_woa,
            /*n_valid=*/min((int)TileT, out_proj.num_tokens - n_base)};
    }
    CUTE_DEVICE int64_t wo_a_out_off(int item) const {
        const int tg = wo_a_tg(item);
        const int rest = wo_a_rest(item);
        const int ks = rest % KS_WOA;
        const int m_tile = rest / KS_WOA;
        return (int64_t)(tg * TileT) * (KS_WOA * d.m_total_woa) +
               ks * d.m_total_woa + m_tile * GemmShapeDefaults::BM;
    }

    template <int KS>
    CUTE_DEVICE GemmWorkItem wo_b_item(int item) const {
        const int tg = item % out_proj.n_groups;
        const int rest = item / out_proj.n_groups;
        const int ks = rest % KS;
        const int m_tile = rest / KS;
        const int n_base = tg * TileT;
        return GemmWorkItem{
            /*n_row=*/n_base,
            /*m_tile=*/m_tile,
            /*m_total=*/d.m_total_wob,
            /*k_tile_begin=*/ks * k_tps_wob,
            /*a_k_tile_begin=*/ks * k_tps_wob,
            /*out_row_stride=*/KS * d.m_total_wob,
            /*n_valid=*/min((int)TileT, out_proj.num_tokens - n_base)};
    }
    template <int KS>
    CUTE_DEVICE int64_t wo_b_out_off(int item) const {
        const int tg = item % out_proj.n_groups;
        const int rest = item / out_proj.n_groups;
        const int ks = rest % KS;
        const int m_tile = rest / KS;
        return (int64_t)(tg * TileT) * (KS * d.m_total_wob) +
               ks * d.m_total_wob + m_tile * GemmShapeDefaults::BM;
    }

    CUTE_DEVICE RangeBarrierLedger ledger_before_wo_a() const {
        return RangeBarrierLedger{};
    }
    CUTE_DEVICE RangeBarrierLedger ledger_before_wo_b() const {
        RangeBarrierLedger led = ledger_before_wo_a();
        led.contribute<GemmGeom<CfgWoA>::STAGES>(k_tps_woa,
                                                 n_items(out_proj.items_wo_a));
        return led;
    }
};

}  // namespace mega_mla
