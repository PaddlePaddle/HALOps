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

#include "kernel/mla/mega_mla/gemm/pipeline.cuh"

namespace halops::mla {
template <class Cfg>
struct GemmGeom {
    using Inst = GemmInst<Cfg>;
    using Traits = typename Inst::Traits;
    using Mainloop = typename Inst::Mainloop;
    using TileShape = typename Traits::TileShape;
    using UtccpOp = typename Mainloop::UtccpOp;

    static constexpr int Stages = Cfg::TmaStages;

    static constexpr int AccStages = Cfg::AccStages;
    static constexpr int TileT = Cfg::TokenTile;
    static constexpr bool SfTilePerGroup = Inst::SfTilePerGroup;
    static constexpr int VsPerCtaN = Inst::VsPerCtaN;
    static constexpr int SfbColsPerSubSlot = Inst::SfbColsPerSubSlot;

    static constexpr int DrainN = Cfg::DrainNOverride
                                      ? Cfg::DrainNOverride
                                      : (Cfg::BN <= 128 ? Cfg::BN : 64);
    static constexpr int DrainSlices = Cfg::BN / DrainN;
    using EpiTile = cute::Shape<cute::Int<Cfg::BM>, cute::Int<DrainN>>;
    using CopyOpT2R = TmemLoad32dp32b<DrainN>;
    static constexpr int AccSharedCols = Cfg::AccSharedCols;
    static constexpr bool AccOverlap = AccSharedCols > 0;
    static constexpr int WinLoSlices =
        AccOverlap ? (AccSharedCols + DrainN - 1) / DrainN : 0;
    static constexpr int OwnHighCols =
        Cfg::BN > Cfg::AccStride ? Cfg::BN - Cfg::AccStride : 0;
    static constexpr int WinHiSlices =
        AccOverlap && OwnHighCols ? (OwnHighCols + DrainN - 1) / DrainN : 0;

    CUTE_DEVICE static int drain_slice(int step, int acc_stage) {
        if constexpr (!AccOverlap) {
            return step;
        } else {
            const int lo = acc_stage > 0 ? WinLoSlices : 0;
            const int hi = acc_stage < AccStages - 1 ? WinHiSlices : 0;
            if (lo == 0) return step < hi ? DrainSlices - hi + step : step - hi;
            if (step < lo) return step;
            if (step < lo + hi) return DrainSlices - hi + (step - lo);
            return step - hi;
        }
    }

    CUTE_DEVICE static int win_lo_step(int acc_stage) {
        return acc_stage > 0 ? WinLoSlices - 1 : 0;
    }
    CUTE_DEVICE static int win_hi_step(int acc_stage) {
        if (WinHiSlices == 0 || acc_stage >= AccStages - 1) return 0;
        return (acc_stage > 0 ? WinLoSlices : 0) + WinHiSlices - 1;
    }
};

template <class Geom, class TmemStorage, class AtomKey = Geom>
struct GemmResources {
    using GeomT = Geom;
    using AtomKeyT = AtomKey;

    const WeightAtoms<AtomKey>& weight;
    const ActAtoms<Geom>& act;
    char* __restrict__ gemm_smem;
    TmemStorage& tmem;

    GemmPipelineBars bars;
};

template <class Cfg, class Res>
CUTE_DEVICE auto const& a_side(const Res& res) {
    if constexpr (Cfg::WeightInA)
        return res.weight;
    else
        return res.act;
}

template <class Cfg, class Res>
CUTE_DEVICE auto const& b_side(const Res& res) {
    if constexpr (Cfg::WeightInA)
        return res.act;
    else
        return res.weight;
}

struct NoGate {
    CUTE_DEVICE void operator()(int) const {}
};

template <class Cfg, class Gate>
CUTE_DEVICE auto gate_on_a(const Gate& gate) {
    if constexpr (Cfg::WeightInA)
        return NoGate{};
    else
        return gate;
}

template <class Cfg, class Gate>
CUTE_DEVICE auto gate_on_b(const Gate& gate) {
    if constexpr (Cfg::WeightInA)
        return gate;
    else
        return NoGate{};
}

struct GemmWorkItem {
    int n_row;
    int w_tile;
    int w_out;
    int k_tile_begin;
    int a_k_tile_begin;
    int out_row_stride;
    int n_valid;
    int w_valid;
};

template <class Derived, int Stages>
struct TmaStageWalk {
    const GemmPipelineBars& bars;
    bool elected;
    int base_tile;
    int seq;
    int k_tiles;

    CUTE_DEVICE void issue_tile(int k_tile) {
        const int chain_idx = chain_tile(base_tile, seq, k_tiles, k_tile);
        wait_smem_stage_free<Stages>(bars, chain_idx);
        if (elected)
            dispatch_stage<Stages>(chain_idx % Stages,
                                   static_cast<Derived*>(this)->issuer(),
                                   k_tile);
    }
    CUTE_DEVICE void end_item() { ++seq; }
};

template <int Stages, int AccStages, bool Overlap>
struct MmaStageWalk {
    const GemmPipelineBars& bars;
    bool elected;
    int base_tile;
    int seq;
    int k_tiles;
    int base_acc;
    int acc_seq;
    int acc_stage;

    CUTE_DEVICE void acquire_acc() {
        acc_stage = wait_acc_stage_free<AccStages>(bars, base_acc, acc_seq);
        wait_acc_windows_free<AccStages, Overlap>(bars, base_acc, acc_seq);
    }
    CUTE_DEVICE int smem_stage(int k) const {
        return chain_tile(base_tile, seq, k_tiles, k) % Stages;
    }
    CUTE_DEVICE void wait_tma(int k) {
        wait_smem_stage_filled<Stages>(bars,
                                       chain_tile(base_tile, seq, k_tiles, k));
    }
    CUTE_DEVICE void commit(int k) {
        __syncwarp();
        if (elected) umma_arrive_noelect(bars.smem_free[smem_stage(k)]);
    }
    CUTE_DEVICE void end_item() {
        if (elected) umma_arrive_noelect(bars.acc_full[acc_stage]);
        ++seq;
        ++acc_seq;
    }
};

template <int AccStages>
struct DrainStageWalk {
    const GemmPipelineBars& bars;
    int base_acc;
    int acc_seq;
    int acc_stage;

    CUTE_DEVICE void wait_acc_full() {
        acc_stage = wait_acc_stage_filled<AccStages>(bars, base_acc, acc_seq);
        tcgen05_after_thread_sync();
    }
    CUTE_DEVICE void end_item() {
        cutlass::arch::fence_view_async_tmem_load();
        bars.acc_empty[acc_stage].arrive();
        ++acc_seq;
    }
};

template <class Cfg>
struct GemmParts {
    using Inst = GemmInst<Cfg>;

    template <class T, int Off, class Layout>
    static CUTE_DEVICE auto stage_tensor(char* smem) {
        return cute::make_tensor(
            cute::make_smem_ptr(reinterpret_cast<T*>(smem + Off)), Layout{});
    }
    static CUTE_DEVICE auto smem_a(char* smem) {
        return stage_tensor<cutlass::float_e4m3_t, Inst::OffA,
                            typename Inst::SmemLayoutAStaged>(smem);
    }
    static CUTE_DEVICE auto smem_b(char* smem) {
        return stage_tensor<cutlass::float_e4m3_t, Inst::OffB,
                            typename Inst::SmemLayoutBStaged>(smem);
    }
    static CUTE_DEVICE auto smem_sfa(char* smem) {
        return stage_tensor<cutlass::float_ue8m0_t, Inst::OffSfa,
                            typename Inst::SmemLayoutSFAStaged>(smem);
    }
    static CUTE_DEVICE auto smem_sfb(char* smem) {
        return stage_tensor<cutlass::float_ue8m0_t, Inst::OffSfb,
                            typename Inst::SmemLayoutSFBStaged>(smem);
    }
};
}  // namespace halops::mla
