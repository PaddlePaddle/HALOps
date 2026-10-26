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

#include "kernel/mla/mla_params.cuh"

namespace halops::mla {
struct PipelineOffsets {
    int tiles = 0;
    int acc_items = 0;

    CUTE_DEVICE void contribute(int k_tps, int n_items) {
        tiles += k_tps * n_items;
        acc_items += n_items;
    }

    CUTE_DEVICE int base_tile() const { return tiles; }
    CUTE_DEVICE int base_acc() const { return acc_items; }
};

template <class Geom, class PartT>
struct GemmTask {
    GemmWorkItem wi;
    const WeightAtoms<Geom>* weight;
    PartT* __restrict__ out;
};

template <class CfgQkv, class CfgWqB, class ResQkv, class ResWqB,
          class ResKvGate>
struct MlaGemmRanges {
    using Geom = typename CfgQkv::Geom;

    const ResQkv& res_qkv;
    const WeightAtoms<typename ResQkv::AtomKeyT>* atoms_wkv;
    const ResWqB& res_wqb;
    const ResKvGate& res_kv_gate;
    typename CfgQkv::PartialT* __restrict__ out_qlora_part;
    typename CfgQkv::PartialT* __restrict__ out_kv_part;
    typename CfgWqB::PartialT* __restrict__ out_flat_q;

    using GemmTaskT =
        GemmTask<typename ResQkv::AtomKeyT, typename CfgQkv::PartialT>;

    template <class Sched>
    CUTE_DEVICE GemmTaskT qkv_item(const Sched& sched, int item) const {
        const bool is_q = sched.qkv_is_q(item);
        return GemmTaskT{
            sched.qkv_item(item), is_q ? &res_qkv.weight : atoms_wkv,
            (is_q ? out_qlora_part : out_kv_part) + sched.qkv_out_off(item)};
    }

    template <class Sched>
    CUTE_DEVICE GemmTask<Geom, typename CfgWqB::PartialT> wqb_item(
        const Sched& sched, int item) const {
        return GemmTask<Geom, typename CfgWqB::PartialT>{
            sched.wqb_item(item), &res_wqb.weight,
            out_flat_q + sched.wqb_out_off(item)};
    }
};

template <class CfgQkv, class CfgWqB, class ResQkv, class ResWqB,
          class ResKvGate>
CUTE_DEVICE auto make_mla_gemm_ranges(
    const ResQkv& res_qkv,
    const WeightAtoms<typename ResQkv::AtomKeyT>* atoms_wkv,
    const ResWqB& res_wqb, const ResKvGate& res_kv_gate,
    typename CfgQkv::PartialT* __restrict__ out_qlora_part,
    typename CfgQkv::PartialT* __restrict__ out_kv_part,
    typename CfgWqB::PartialT* __restrict__ out_flat_q) {
    return MlaGemmRanges<CfgQkv, CfgWqB, ResQkv, ResWqB, ResKvGate>{
        .res_qkv = res_qkv,
        .atoms_wkv = atoms_wkv,
        .res_wqb = res_wqb,
        .res_kv_gate = res_kv_gate,
        .out_qlora_part = out_qlora_part,
        .out_kv_part = out_kv_part,
        .out_flat_q = out_flat_q};
}

struct PreludeSchedule {
    int items_kv_gate;
    int items_wkv;
    int items_qkv;
    int items_wq_b;
    int n_groups;
    int num_tokens;
    int w_out_wq_a;
    int w_out_wkv;
    int w_out_wq_b;
    int k_tps_qkv;
    int k_tps_wqb;
    int k_tiles_kvg;
    PlanView wplan;
    int kvg_ctas;
};

template <class KT>
struct MlaScheduler : PreludeSchedule {
    static constexpr int TileT = KT::TileT;
    using CfgQkv = typename KT::Qkv;
    using CfgWqB = typename KT::WqB;
    using Params = PreludeSchedule;
    static constexpr int KSplitQkv = CfgQkv::KSplit;
    static constexpr int kNoItem = 0x3fffffff;

    template <class MP>
    static Params to_underlying_arguments(const MlaAttentionProblem<KT>& prob,
                                          const MP& mp) {
        const int n_groups = prob.n_groups;
        const int items_wq_a =
            n_groups * mp.shape_wq_a.w_tiles * CfgQkv::KSplit;
        const int items_wkv = n_groups * mp.shape_wkv.w_tiles * CfgQkv::KSplit;
        Params params{};
        params.items_kv_gate = kv_gate_items(n_groups, mp.use_kv_gate);
        params.items_wkv = items_wkv;
        params.items_qkv = items_wq_a + items_wkv;
        params.items_wq_b = n_groups * mp.shape_wq_b.w_tiles;
        params.n_groups = n_groups;
        params.num_tokens = prob.num_tokens;
        params.w_out_wq_a = mp.shape_wq_a.w_out;
        params.w_out_wkv = mp.shape_wkv.w_out;
        params.w_out_wq_b = mp.shape_wq_b.w_out;
        params.k_tps_qkv = mp.shape_wq_a.k_tps;
        params.k_tps_wqb = mp.shape_wq_b.k_tps;
        params.k_tiles_kvg = mp.shape_kv_gate.k_tiles;
        params.wplan = mp.wplan;
        params.kvg_ctas =
            kv_gate_ctas(prob.num_tokens, prob.grid_x, params.items_kv_gate);
        return params;
    }

    int t_iters;

    CUTE_DEVICE static MlaScheduler make(Params const& params) {
        return MlaScheduler{params, __ldg(&params.wplan.hdr->iters)};
    }

    CUTE_DEVICE int token_iters() const { return t_iters; }

    CUTE_DEVICE int wqb_ctas() const { return gridDim.x - kvg_ctas; }
    CUTE_DEVICE int wqb_first() const {
        return blockIdx.x < wqb_ctas() ? blockIdx.x : kNoItem;
    }
    CUTE_DEVICE int wqb_step() const { return wqb_ctas(); }
    CUTE_DEVICE int kvg_first() const {
        if (kvg_ctas == 0) return rev_bid();
        const int cta = blockIdx.x - wqb_ctas();
        return cta >= 0 ? cta : kNoItem;
    }
    CUTE_DEVICE int kvg_step() const {
        return kvg_ctas == 0 ? gridDim.x : kvg_ctas;
    }
    struct ItemCoord {
        int tg, ks, mt, w_out;
    };

    CUTE_DEVICE bool qkv_is_q(int item) const { return item >= items_wkv; }
    CUTE_DEVICE ItemCoord qkv_coord(int item) const {
        const bool is_q = qkv_is_q(item);
        const int it = is_q ? item - items_wkv : item;
        const int rest = it / n_groups;
        return {it % n_groups, rest % KSplitQkv, rest / KSplitQkv,
                is_q ? w_out_wq_a : w_out_wkv};
    }
    CUTE_DEVICE GemmWorkItem qkv_item(int item) const {
        const auto [tg, ks, mt, w_out] = qkv_coord(item);
        const int n_base = tg * TileT;
        return GemmWorkItem{.n_row = n_base,
                            .w_tile = mt,
                            .w_out = w_out,
                            .k_tile_begin = ks * k_tps_qkv,
                            .a_k_tile_begin = ks * k_tps_qkv,
                            .out_row_stride = KSplitQkv * w_out,
                            .n_valid = min(TileT, num_tokens - n_base),
                            .w_valid = min(CfgQkv::WeightTile,
                                           w_out - mt * CfgQkv::WeightTile)};
    }
    CUTE_DEVICE int64_t qkv_out_off(int item) const {
        const auto [tg, ks, mt, w_out] = qkv_coord(item);
        return (int64_t)(tg * TileT) * (KSplitQkv * w_out) + ks * w_out +
               mt * CfgQkv::WeightTile;
    }
    CUTE_DEVICE ItemCoord wqb_coord(int item) const {
        return {item % n_groups, 0, item / n_groups, w_out_wq_b};
    }
    CUTE_DEVICE GemmWorkItem wqb_item(int item) const {
        const auto [tg, ks, mt, w_out] = wqb_coord(item);
        const int n_base = tg * TileT;
        return GemmWorkItem{.n_row = n_base,
                            .w_tile = mt,
                            .w_out = w_out,
                            .k_tile_begin = 0,
                            .a_k_tile_begin = 0,
                            .out_row_stride = w_out,
                            .n_valid = min(TileT, num_tokens - n_base),
                            .w_valid = min(CfgWqB::WeightTile,
                                           w_out - mt * CfgWqB::WeightTile)};
    }
    CUTE_DEVICE int64_t wqb_out_off(int item) const {
        const auto [tg, ks, mt, w_out] = wqb_coord(item);
        return (int64_t)(tg * TileT) * w_out + mt * CfgWqB::WeightTile;
    }

    CUTE_DEVICE PipelineOffsets ledger_before_qkv() const {
        return PipelineOffsets{};
    }
    CUTE_DEVICE PipelineOffsets ledger_before_wq_b() const {
        PipelineOffsets led = ledger_before_qkv();
        led.contribute(k_tps_qkv, items_from(blockIdx.x, gridDim.x, items_qkv));
        return led;
    }
    CUTE_DEVICE PipelineOffsets ledger_before_kv_gate() const {
        PipelineOffsets led = ledger_before_wq_b();
        led.contribute(k_tps_wqb,
                       items_from(wqb_first(), wqb_step(), items_wq_b));
        return led;
    }
};
}  // namespace halops::mla
