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
#include <cutlass/numeric_types.h>

#include <cute/tensor.hpp>

#include "kernel/gemm/kernel_traits.cuh"
#include "kernel/gemm/mainloop_fwd.cuh"
#include "kernel/mla/mla_intrinsics.cuh"
#include "kernel/mla/mla_traits.cuh"
#include "kernel/mla/mla_utils.cuh"

namespace halops::mla {
template <int KBlockN_, int KPrologueStages_ = 1, int KEpilogueStages_ = 1,
          int KBlockM_ = 128, int KBlockK_ = 256, int KTmemOverlapCols_ = 0,
          bool PairUmma_ = false>
struct MlaGemmConfigT : GemmKernelConfig {
    static constexpr int KBlockM = KBlockM_;
    static constexpr int KBlockN = KBlockN_;
    static constexpr int KBlockK = KBlockK_;
    static constexpr bool IsPairUmma = PairUmma_;
    static constexpr int KClusterM = PairUmma_ ? 2 : 1;
    static constexpr int KClusterN = 1;
    static constexpr int KPrologueStages = KPrologueStages_;
    static constexpr int KEpilogueStages = KEpilogueStages_;
    static constexpr int KBiasStages = 1;
    static constexpr int KTmemOverlapCols = KTmemOverlapCols_;
    static constexpr bool UseTileSwizzle = false;
    static constexpr KernelMode KernelType = KernelMode::Dense;
    static constexpr bool AddBias = false;
    static constexpr bool SwapAB = true;
    static constexpr bool KIsFP4 = false;
    static constexpr bool HasAlpha = false;
};

template <class Cfg, int Stages = Cfg::TmaStages,
          int AccStages = Cfg::AccStages>
struct GemmInst {
    static constexpr int KBlockM = Cfg::BM;
    static constexpr int KBlockN = Cfg::BN;
    static constexpr int KBlockK = Cfg::BK;

    static constexpr int OverlapCols =
        Cfg::AccStride == KBlockN ? 0 : Sm100::UmmaMaxN - Cfg::AccStride;

    using Config = MlaGemmConfigT<KBlockN, Stages, AccStages, KBlockM, KBlockK,
                                  OverlapCols>;
    using Traits =
        GemmKernelTraits<cutlass::float_e4m3_t, cutlass::float_e4m3_t,
                         cutlass::bfloat16_t, cutlass::bfloat16_t,
                         cutlass::float_ue8m0_t, float, Config>;
    using Mainloop = CollectiveMainloop<Traits>;

    static constexpr bool SfTilePerGroup = Cfg::SfTilePerGroup;

    static constexpr int VsPerCtaN = Traits::CTA_N_SF / KBlockN;

    static constexpr int SfbColsPerSubSlot = KBlockN / SfLayout::MnPerCol;

    using SmemLayoutAStaged = typename Traits::SmemLayoutA;
    using SmemLayoutBStaged = typename Traits::SmemLayoutB;
    using SmemLayoutSFAStaged = typename Traits::SmemLayoutSFA;
    using SmemLayoutSFBStaged = typename Traits::SmemLayoutSFB;

    static constexpr int SmemA = cute::cosize_v<typename Traits::SmemLayoutA> *
                                 sizeof(cutlass::float_e4m3_t) / Stages;
    static constexpr int SmemSfa =
        cute::cosize_v<typename Traits::SmemLayoutSFA> *
        sizeof(cutlass::float_ue8m0_t) / Stages;

    static constexpr int SlotB = Cfg::ChainBN * KBlockK;
    static constexpr int SlotSfb = Cfg::ChainCtaNSf * KBlockK / SfLayout::Gran;

    static constexpr int ArenaStages = Cfg::OwnerTraits::MaxTmaStages;

    static constexpr int SmemTotal =
        ArenaStages * (SmemA + SlotB + SmemSfa + SlotSfb);

    static constexpr int OffA = 0;
    static constexpr int OffB = OffA + ArenaStages * SmemA;
    static constexpr int OffSfa = OffB + ArenaStages * SlotB;
    static constexpr int OffSfb = OffSfa + ArenaStages * SmemSfa;

    static constexpr uint32_t TmaBytesA =
        Mainloop::TmaTransactionBytesA + Mainloop::TmaTransactionBytesSFA;
    static constexpr uint32_t TmaBytesB =
        Mainloop::TmaTransactionBytesB + Mainloop::TmaTransactionBytesSFB;
};

template <class KT>
CUTE_HOST_DEVICE constexpr int traits_smem_max() {
    return cute::max(GemmInst<typename KT::Qkv>::SmemTotal,
                     GemmInst<typename KT::WqB>::SmemTotal);
}

struct GemmArena {
    static constexpr int Total =
        cute::max(traits_smem_max<TraitsN64>(), traits_smem_max<TraitsN128>(),
                  traits_smem_max<TraitsN32>(), traits_smem_max<TraitsN16>());
    static constexpr int MaxTmaStages =
        cute::max(TraitsN64::MaxTmaStages, TraitsN128::MaxTmaStages,
                  TraitsN32::MaxTmaStages, TraitsN16::MaxTmaStages);
    static constexpr int MaxAccStages =
        cute::max(TraitsN64::MaxAccStages, TraitsN128::MaxAccStages,
                  TraitsN32::MaxAccStages, TraitsN16::MaxAccStages);
};

struct GemmShape {
    int w_out;
    int k_dim;
    int w_tiles;
    int k_tiles;
    int k_tps;
};

template <int WeightTile, int BK, int KSplit>
CUTE_HOST GemmShape make_gemm_shape(int w_out, int k_dim) {
    TORCH_CHECK(k_dim % (BK * KSplit) == 0, "mla: GEMM K = ", k_dim,
                " must be a multiple of ", BK * KSplit);
    const int k_tiles = k_dim / BK;
    const int k_tps = k_tiles / KSplit;
    GemmShape shape{};
    shape.w_out = w_out;
    shape.k_dim = k_dim;
    shape.w_tiles = (w_out + WeightTile - 1) / WeightTile;
    shape.k_tiles = k_tiles;
    shape.k_tps = k_tps;
    return shape;
}

template <class Geom>
struct GemmAtomsA {
    using Inst = GemmInst<Geom>;
    typename Inst::Mainloop::TMALoad_A tma_a;
    typename Inst::Mainloop::TMALoad_SFA tma_sfa;
    typename Inst::Traits::LayoutSFA layout_sfa;
    int w_out;
    int k_dim;
};

template <class Geom>
CUTE_HOST GemmAtomsA<Geom> make_atoms_a(const void* weight,
                                        const void* weight_sf, int w_out,
                                        int k_dim) {
    using Traits = typename GemmInst<Geom>::Traits;
    using Mainloop = typename GemmInst<Geom>::Mainloop;

    auto layout_a =
        cute::make_layout(cute::make_shape(w_out, k_dim, 1),
                          cute::make_stride(k_dim, cute::Int<1>{}, 0));
    auto layout_sfa = Traits::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
        cute::make_shape(w_out, Geom::BN, k_dim, 1));

    cute::Tensor tensor_a = cute::make_tensor(
        cute::make_gmem_ptr(static_cast<const cutlass::float_e4m3_t*>(weight)),
        layout_a);
    cute::Tensor tensor_sfa = cute::make_tensor(
        static_cast<const cutlass::float_ue8m0_t*>(weight_sf), layout_sfa);

    auto cluster_layout_vmnk = cute::tiled_divide(
        cute::make_layout(typename Traits::ClusterShape_MNK{}),
        cute::make_tile(typename Traits::TiledMma::AtomThrID{}));

    GemmAtomsA<Geom> atoms{
        cute::make_tma_atom_A_sm100(
            typename Mainloop::GmemTiledLoad{}, tensor_a,
            typename Traits::SmemLayoutA{}(cute::_, cute::_, cute::_,
                                           cute::Int<0>{}),
            typename Traits::TileShape{}, typename Traits::TiledMma{},
            cluster_layout_vmnk),
        cute::make_tma_atom_A_sm100<uint16_t>(
            typename Mainloop::GmemTiledLoad{}, tensor_sfa,
            typename Traits::SmemLayoutSFA{}(cute::_, cute::_, cute::_,
                                             cute::Int<0>{}),
            typename Traits::TileShape{}, typename Traits::TiledMma{},
            cluster_layout_vmnk),
        layout_sfa,
        w_out,
        k_dim,
    };
    return atoms;
}

template <class Geom>
struct GemmAtomsB {
    using Inst = GemmInst<Geom>;
    typename Inst::Mainloop::TMALoad_B tma_b;
    typename Inst::Mainloop::TMALoad_SFB tma_sfb;
    typename Inst::Traits::LayoutSFB layout_sfb;
    int n_pad;
    int k_dim;
};

template <class Geom>
CUTE_HOST GemmAtomsB<Geom> make_atoms_b(const void* act_fp8, const void* act_sf,
                                        int w_out, int n_pad, int k_dim,
                                        int n_pad_sf) {
    using Traits = typename GemmInst<Geom>::Traits;
    using Mainloop = typename GemmInst<Geom>::Mainloop;

    auto layout_b =
        cute::make_layout(cute::make_shape(n_pad, k_dim, 1),
                          cute::make_stride(k_dim, cute::Int<1>{}, 0));
    auto layout_sfb = Traits::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
        cute::make_shape(w_out, n_pad_sf, k_dim, 1));

    cute::Tensor tensor_b = cute::make_tensor(
        cute::make_gmem_ptr(static_cast<const cutlass::float_e4m3_t*>(act_fp8)),
        layout_b);
    cute::Tensor tensor_sfb = cute::make_tensor(
        static_cast<const cutlass::float_ue8m0_t*>(act_sf), layout_sfb);

    auto cluster_layout_vmnk = cute::tiled_divide(
        cute::make_layout(typename Traits::ClusterShape_MNK{}),
        cute::make_tile(typename Traits::TiledMma::AtomThrID{}));
    auto cluster_layout_sfb_vmnk = cute::tiled_divide(
        cute::make_layout(typename Traits::ClusterShape_MNK{}),
        cute::make_tile(typename Traits::TiledMma_SF::AtomThrID{}));

    GemmAtomsB<Geom> atoms{
        cute::make_tma_atom_B_sm100(
            typename Mainloop::GmemTiledLoad{}, tensor_b,
            typename Traits::SmemLayoutB{}(cute::_, cute::_, cute::_,
                                           cute::Int<0>{}),
            typename Traits::TileShape{}, typename Traits::TiledMma{},
            cluster_layout_vmnk),
        cute::make_tma_atom_B_sm100<uint16_t>(
            typename Mainloop::GmemTiledLoad{}, tensor_sfb,
            typename Traits::SmemLayoutSFB{}(cute::_, cute::_, cute::_,
                                             cute::Int<0>{}),
            typename Traits::TileShape_SF{}, typename Traits::TiledMma_SF{},
            cluster_layout_sfb_vmnk),
        layout_sfb,
        n_pad,
        k_dim,
    };
    return atoms;
}

template <class Geom>
using WeightAtoms =
    cute::conditional_t<Geom::WeightInA, GemmAtomsA<Geom>, GemmAtomsB<Geom>>;
template <class Geom>
using ActAtoms =
    cute::conditional_t<Geom::WeightInA, GemmAtomsB<Geom>, GemmAtomsA<Geom>>;

template <class Geom>
CUTE_HOST WeightAtoms<Geom> make_weight_atoms(const void* weight,
                                              const void* w_sf, int w_out,
                                              int k) {
    if constexpr (Geom::WeightInA)
        return make_atoms_a<Geom>(weight, w_sf, w_out, k);
    else {
        const int w_tiles = (w_out + Geom::WeightTile - 1) / Geom::WeightTile;
        return make_atoms_b<Geom>(weight, w_sf, w_out, w_out, k,
                                  w_tiles * Geom::CtaNSf);
    }
}

template <class Geom>
CUTE_HOST ActAtoms<Geom> make_act_atoms(const void* act, const void* act_sf,
                                        int w_out, int t_pad, int k,
                                        int t_pad_sf) {
    if constexpr (Geom::WeightInA)
        return make_atoms_b<Geom>(act, act_sf, w_out, t_pad, k, t_pad_sf);
    else
        return make_atoms_a<Geom>(act, act_sf, t_pad, k);
}

template <class Geom>
CUTE_HOST_DEVICE auto const& sf_layout(const GemmAtomsA<Geom>& at) {
    return at.layout_sfa;
}
template <class Geom>
CUTE_HOST_DEVICE auto const& sf_layout(const GemmAtomsB<Geom>& at) {
    return at.layout_sfb;
}

template <class Acc, class SFA, class SFB>
struct GemmTmemStorage {
    Acc tmem_accumulators;
    SFA tmem_SFA;
    SFB tmem_SFB;
};

template <class Cfg, int AccStages = Cfg::AccStages>
CUTE_DEVICE auto make_gemm_tmem_for(uint32_t tmem_base) {
    using Inst = GemmInst<Cfg, Cfg::TmaStages, AccStages>;
    using Mainloop = typename Inst::Mainloop;
    auto tmem_storage = Mainloop::init_tmem_tensors();
    Mainloop::set_tmem_offsets(tmem_storage, tmem_base);
    tmem_storage.tmem_SFA.data() = tmem_base + (uint32_t)Cfg::SfBaseCols;
    tmem_storage.tmem_SFB.data() =
        tmem_storage.tmem_SFA.data().get() +
        cutlass::detail::find_tmem_tensor_col_offset(tmem_storage.tmem_SFA);
    if constexpr (Inst::OverlapCols == 0) {
        return GemmTmemStorage<decltype(tmem_storage.tmem_accumulators),
                               decltype(tmem_storage.tmem_SFA),
                               decltype(tmem_storage.tmem_SFB)>{
            tmem_storage.tmem_accumulators, tmem_storage.tmem_SFA,
            tmem_storage.tmem_SFB};
    } else {
        auto const& acc = tmem_storage.tmem_accumulators;
        auto acc_ring = cute::make_tensor(
            acc.data(),
            cute::make_layout(
                cute::replace<3>(acc.layout().shape(), cute::Int<AccStages>{}),
                acc.layout().stride()));
        return GemmTmemStorage<decltype(acc_ring),
                               decltype(tmem_storage.tmem_SFA),
                               decltype(tmem_storage.tmem_SFB)>{
            acc_ring, tmem_storage.tmem_SFA, tmem_storage.tmem_SFB};
    }
}

struct GemmPipelineBars {
    cutlass::arch::ClusterTransactionBarrier* tma_a;
    cutlass::arch::ClusterTransactionBarrier* tma_b;
    cutlass::arch::ClusterTransactionBarrier* smem_free;
    cutlass::arch::ClusterTransactionBarrier* acc_full;
    cutlass::arch::ClusterTransactionBarrier* acc_empty;
    cutlass::arch::ClusterTransactionBarrier* acc_win_lo;
    cutlass::arch::ClusterTransactionBarrier* acc_win_hi;
};

CUTE_DEVICE int chain_tile(int base_tile, int seq, int k_tps, int k_tile) {
    return base_tile + seq * k_tps + k_tile;
}

template <int Stages>
CUTE_DEVICE void wait_smem_stage_free(const GemmPipelineBars& bars, int seq) {
    if (seq < Stages) return;
    __syncwarp();
    bars.smem_free[seq % Stages].wait((uint32_t)((seq / Stages - 1) & 1));
}

template <int Stages>
CUTE_DEVICE void wait_smem_stage_filled(const GemmPipelineBars& bars, int seq) {
    const int smem_stage = seq % Stages;
    const uint32_t phase = (uint32_t)((seq / Stages) & 1);
    __syncwarp();
    bars.tma_a[smem_stage].wait(phase);
    bars.tma_b[smem_stage].wait(phase);
}

template <int AccStages>
CUTE_DEVICE int wait_acc_stage_free(const GemmPipelineBars& bars, int base_acc,
                                    int acc_seq) {
    const int acc_global = base_acc + acc_seq;
    const int acc_stage = acc_global % AccStages;
    if (acc_global >= AccStages) {
        __syncwarp();
        bars.acc_empty[acc_stage].wait(
            (uint32_t)((acc_global / AccStages - 1) & 1));
    }
    return acc_stage;
}

template <int AccStages>
CUTE_DEVICE int wait_acc_stage_filled(const GemmPipelineBars& bars,
                                      int base_acc, int acc_seq) {
    const int acc_global = base_acc + acc_seq;
    const int acc_stage = acc_global % AccStages;
    bars.acc_full[acc_stage].wait((uint32_t)((acc_global / AccStages) & 1));
    return acc_stage;
}

template <int AccStages, bool Overlap>
CUTE_DEVICE void wait_acc_windows_free(const GemmPipelineBars& bars,
                                       int base_acc, int acc_seq) {
    if constexpr (Overlap) {
        const int acc_global = base_acc + acc_seq;
        const int stage = acc_global % AccStages;
        const int prev_acc = acc_global - 1;
        if (stage > 0 && prev_acc >= 0) {
            __syncwarp();
            bars.acc_win_hi[prev_acc % AccStages].wait(
                (uint32_t)((prev_acc / AccStages) & 1));
        }
        const int next_acc = acc_global - (AccStages - 1);
        if (stage < AccStages - 1 && next_acc >= 0) {
            __syncwarp();
            bars.acc_win_lo[next_acc % AccStages].wait(
                (uint32_t)((next_acc / AccStages) & 1));
        }
    }
}

template <bool Overlap>
CUTE_DEVICE void arrive_acc_window_lo(const GemmPipelineBars& bars,
                                      int acc_stage) {
    if constexpr (Overlap) bars.acc_win_lo[acc_stage].arrive();
}

template <bool Overlap>
CUTE_DEVICE void arrive_acc_window_hi(const GemmPipelineBars& bars,
                                      int acc_stage) {
    if constexpr (Overlap) bars.acc_win_hi[acc_stage].arrive();
}

template <int Stages, int Stage = 0, class Fn>
CUTE_DEVICE void dispatch_stage(int stage, Fn&& fn, int k) {
    if constexpr (Stage + 1 == Stages)
        fn(cute::Int<Stage>{}, k);
    else if (stage == Stage)
        fn(cute::Int<Stage>{}, k);
    else
        dispatch_stage<Stages, Stage + 1>(stage, fn, k);
}

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

template <class Cfg, class At>
CUTE_DEVICE auto tile_gA(const At& at, int w_out) {
    cute::Tensor mA = at.tma_a.get_tma_tensor(
        cute::make_shape(w_out, at.k_dim, 1))(cute::_, cute::_, cute::Int<0>{});
    return cute::local_tile(mA, typename GemmGeom<Cfg>::TileShape{},
                            cute::make_coord(cute::_, cute::_, cute::_),
                            cute::Step<cute::_1, cute::X, cute::_1>{});
}
template <class Cfg, class At>
CUTE_DEVICE auto tile_gSFA(const At& at) {
    cute::Tensor mSFA = at.tma_sfa.get_tma_tensor(at.layout_sfa.shape())(
        cute::_, cute::_, cute::Int<0>{});
    return cute::local_tile(mSFA, typename GemmGeom<Cfg>::TileShape{},
                            cute::make_coord(cute::_, cute::_, cute::_),
                            cute::Step<cute::_1, cute::X, cute::_1>{});
}

template <class Cfg, class Res, class SA, class SSFA, class CtaMma, class GA,
          class GSFA, class Gate>
struct TmaOpA : TmaStageWalk<TmaOpA<Cfg, Res, SA, SSFA, CtaMma, GA, GSFA, Gate>,
                             GemmGeom<Cfg>::Stages> {
    using CfgGeom = GemmGeom<Cfg>;
    using Inst = typename CfgGeom::Inst;

    SA sA;
    SSFA sSFA;
    CtaMma cta_mma;

    using AtA =
        cute::conditional_t<Cfg::WeightInA, WeightAtoms<typename Res::AtomKeyT>,
                            ActAtoms<typename Res::GeomT>>;
    const AtA* at;
    GA gA;
    GSFA gSFA;
    int w_tile;
    int a_k_begin;
    Gate gate;

    CUTE_DEVICE void begin_item(
        const WeightAtoms<typename Res::AtomKeyT>& atoms,
        const GemmWorkItem& wi) {
        gate(wi.n_row / CfgGeom::TileT);
        if constexpr (Cfg::WeightInA) {
            at = &atoms;
            gA = tile_gA<Cfg>(atoms, wi.w_out);
            gSFA = tile_gSFA<Cfg>(atoms);
            w_tile = wi.w_tile;
            a_k_begin = wi.a_k_tile_begin;
        } else {
            w_tile = wi.n_row / CfgGeom::TileT;
            a_k_begin = wi.k_tile_begin;
        }
    }

    CUTE_DEVICE auto issuer() {
        return [this](auto St, int k) {
            const int ka = k + a_k_begin;
            constexpr int stage = decltype(St)::value;
            this->bars.tma_a[stage].arrive_and_expect_tx(Inst::TmaBytesA);
            uint64_t& mbar =
                reinterpret_cast<uint64_t&>(this->bars.tma_a[stage]);
            auto [tma_gA, tma_sA] = cute::tma_partition(
                at->tma_a, cute::Int<0>{}, cute::make_layout(cute::Int<1>{}),
                cute::group_modes<0, 3>(sA(cute::_, cute::_, cute::_, St)),
                cute::group_modes<0, 3>(cta_mma.partition_A(gA)));
            auto [tma_gSFA, tma_sSFA] = cute::tma_partition(
                at->tma_sfa, cute::Int<0>{}, cute::make_layout(cute::Int<1>{}),
                cute::group_modes<0, 3>(sSFA(cute::_, cute::_, cute::_, St)),
                cute::group_modes<0, 3>(cta_mma.partition_A(gSFA)));
            cute::copy(at->tma_a.with(mbar, 0), tma_gA(cute::_, w_tile, ka),
                       tma_sA);
            cute::copy(at->tma_sfa.with(mbar, 0), tma_gSFA(cute::_, w_tile, ka),
                       tma_sSFA);
        };
    }
};

template <class Cfg, class Res, class Gate>
CUTE_DEVICE auto make_tma_op_a(const Res& res, int base_tile, int k_tps,
                               const Gate& act_gate) {
    using Parts = GemmParts<Cfg>;
    typename GemmGeom<Cfg>::Traits::TiledMma tiled_mma;
    auto sA = Parts::smem_a(res.gemm_smem);
    auto sSFA = Parts::smem_sfa(res.gemm_smem);
    auto cta_mma = tiled_mma.get_slice(0);
    auto gA = tile_gA<Cfg>(a_side<Cfg>(res), a_side<Cfg>(res).w_out);
    auto gSFA = tile_gSFA<Cfg>(a_side<Cfg>(res));
    auto gate = gate_on_a<Cfg>(act_gate);
    return TmaOpA<Cfg, Res, decltype(sA), decltype(sSFA), decltype(cta_mma),
                  decltype(gA), decltype(gSFA), decltype(gate)>{
        {res.bars, cute::elect_one_sync() != 0, base_tile, 0, k_tps},
        sA,
        sSFA,
        cta_mma,
        &a_side<Cfg>(res),
        gA,
        gSFA,
        0,
        0,
        gate};
}

template <class Cfg, class At>
CUTE_DEVICE auto tile_gB(const At& at) {
    const int k_dim = at.k_dim;
    cute::Tensor mB = at.tma_b.get_tma_tensor(
        cute::make_shape(at.n_pad, k_dim, 1))(cute::_, cute::_, cute::Int<0>{});
    return cute::local_tile(mB, typename GemmGeom<Cfg>::TileShape{},
                            cute::make_coord(cute::_, cute::_, cute::_),
                            cute::Step<cute::X, cute::_1, cute::_1>{});
}

template <class Cfg, class At>
CUTE_DEVICE auto tile_gSFB(const At& at) {
    cute::Tensor mSFBg = at.tma_sfb.get_tma_tensor(at.layout_sfb.shape())(
        cute::_, cute::_, cute::Int<0>{});
    return cute::local_tile(mSFBg,
                            typename GemmGeom<Cfg>::Traits::TileShape_SF{},
                            cute::make_coord(cute::_, cute::_, cute::_),
                            cute::Step<cute::X, cute::_1, cute::_1>{});
}

template <class Cfg, class Res, class SB, class SSFB, class CtaMma,
          class CtaMmaSf, class GB, class GSFB, class Gate>
struct TmaOpB
    : TmaStageWalk<TmaOpB<Cfg, Res, SB, SSFB, CtaMma, CtaMmaSf, GB, GSFB, Gate>,
                   GemmGeom<Cfg>::Stages> {
    using CfgGeom = GemmGeom<Cfg>;
    using Inst = typename CfgGeom::Inst;

    SB sB;
    SSFB sSFB;
    CtaMma cta_mma;
    CtaMmaSf cta_mma_sfb;
    GB gB;
    GSFB gSFB;

    using AtB =
        cute::conditional_t<Cfg::WeightInA, ActAtoms<typename Res::GeomT>,
                            WeightAtoms<typename Res::AtomKeyT>>;
    const AtB* at;
    int n_tile;
    int nb_sf;
    int k_begin;
    Gate gate;

    CUTE_DEVICE void begin_item(
        const WeightAtoms<typename Res::AtomKeyT>& w_atoms,
        const GemmWorkItem& wi) {
        gate(wi.n_row / CfgGeom::TileT);
        if constexpr (Cfg::WeightInA) {
            n_tile = wi.n_row / CfgGeom::TileT;
            k_begin = wi.k_tile_begin;
        } else {
            at = &w_atoms;
            gB = tile_gB<Cfg>(w_atoms);
            gSFB = tile_gSFB<Cfg>(w_atoms);
            n_tile = wi.w_tile;
            k_begin = wi.a_k_tile_begin;
        }
        if constexpr (CfgGeom::SfTilePerGroup)
            nb_sf = n_tile;
        else
            nb_sf = n_tile / CfgGeom::VsPerCtaN;
    }

    CUTE_DEVICE auto issuer() {
        return [this](auto St, int k) {
            const int kb = k + k_begin;
            constexpr int stage = decltype(St)::value;
            this->bars.tma_b[stage].arrive_and_expect_tx(Inst::TmaBytesB);
            uint64_t& mbar =
                reinterpret_cast<uint64_t&>(this->bars.tma_b[stage]);
            auto [tma_gB, tma_sB] = cute::tma_partition(
                at->tma_b, cute::Int<0>{}, cute::make_layout(cute::Int<1>{}),
                cute::group_modes<0, 3>(sB(cute::_, cute::_, cute::_, St)),
                cute::group_modes<0, 3>(cta_mma.partition_B(gB)));
            auto [tma_gSFB, tma_sSFB] = cute::tma_partition(
                at->tma_sfb, cute::Int<0>{}, cute::make_layout(cute::Int<1>{}),
                cute::group_modes<0, 3>(sSFB(cute::_, cute::_, cute::_, St)),
                cute::group_modes<0, 3>(cta_mma_sfb.partition_B(gSFB)));
            cute::copy(at->tma_b.with(mbar, 0), tma_gB(cute::_, n_tile, kb),
                       tma_sB);
            cute::copy(at->tma_sfb.with(mbar, 0), tma_gSFB(cute::_, nb_sf, kb),
                       tma_sSFB);
        };
    }
};

template <class Cfg, class Res, class Gate>
CUTE_DEVICE auto make_tma_op_b(const Res& res, int base_tile, int k_tps,
                               const Gate& act_gate) {
    using Parts = GemmParts<Cfg>;
    using Traits = typename GemmGeom<Cfg>::Traits;
    typename Traits::TiledMma tiled_mma;
    typename Traits::TiledMma_SF tiled_mma_sf;
    auto sB = Parts::smem_b(res.gemm_smem);
    auto sSFB = Parts::smem_sfb(res.gemm_smem);
    auto cta_mma = tiled_mma.get_slice(0);
    auto cta_mma_sfb = tiled_mma_sf.get_slice(0);
    auto gB = tile_gB<Cfg>(b_side<Cfg>(res));
    auto gSFB = tile_gSFB<Cfg>(b_side<Cfg>(res));
    auto gate = gate_on_b<Cfg>(act_gate);
    return TmaOpB<Cfg, Res, decltype(sB), decltype(sSFB), decltype(cta_mma),
                  decltype(cta_mma_sfb), decltype(gB), decltype(gSFB),
                  decltype(gate)>{
        {res.bars, cute::elect_one_sync() != 0, base_tile, 0, k_tps},
        sB,
        sSFB,
        cta_mma,
        cta_mma_sfb,
        gB,
        gSFB,
        &b_side<Cfg>(res),
        0,
        0,
        0,
        gate};
}

template <class Cfg, class Res, class TiledMmaT, class RA, class RB,
          class CopySFA, class CopySFB, class SrcSFA, class DstSFA,
          class SrcSFB, class DstSFB, class TmemSFA, class TmemSFB,
          class TmemAcc>
struct MmaOp : MmaStageWalk<GemmGeom<Cfg>::Stages, GemmGeom<Cfg>::AccStages,
                            GemmGeom<Cfg>::AccOverlap> {
    using CfgGeom = GemmGeom<Cfg>;
    using Walk =
        MmaStageWalk<CfgGeom::Stages, CfgGeom::AccStages, CfgGeom::AccOverlap>;

    TiledMmaT tiled_mma;
    RA tile_rA;
    RB tile_rB;
    CopySFA copy_sfa;
    CopySFB copy_sfb;
    SrcSFA src_sfa;
    DstSFA dst_sfa;
    SrcSFB src_sfb;
    DstSFB dst_sfb;
    TmemSFA tmem_SFA;
    TmemSFB tmem_SFB;
    TmemAcc tmem_acc;
    TmemSFB tmem_SFB_mma;

    CUTE_DEVICE void begin_item(const GemmWorkItem& wi) {
        tmem_SFB_mma = tmem_SFB;
        if constexpr (!CfgGeom::SfTilePerGroup && CfgGeom::VsPerCtaN > 1) {
            const int n_tile = wi.n_row / CfgGeom::TileT;
            tmem_SFB_mma.data() =
                tmem_SFB_mma.data().get() +
                (n_tile % CfgGeom::VsPerCtaN) * CfgGeom::SfbColsPerSubSlot;
        }
        this->acquire_acc();
    }

    CUTE_DEVICE void run_k_tile(int k) {
        if (k == 0) tiled_mma.accumulate_ = cute::UMMA::ScaleOut::Zero;
        tcgen05_after_thread_sync();
        dispatch_stage<CfgGeom::Stages>(this->smem_stage(k), stager(), k);
    }

    CUTE_DEVICE void commit(int k) {
        tiled_mma.accumulate_ = cute::UMMA::ScaleOut::One;
        Walk::commit(k);
    }

    CUTE_DEVICE auto stager() {
        return [this](auto St, int) {
            if (this->elected) {
                cute::copy(copy_sfa,
                           src_sfa(cute::_, cute::_, cute::_, cute::_, St),
                           dst_sfa(cute::_, cute::_, cute::_, cute::_));
                cute::copy(copy_sfb,
                           src_sfb(cute::_, cute::_, cute::_, cute::_, St),
                           dst_sfb(cute::_, cute::_, cute::_, cute::_));
            }
            CUTLASS_PRAGMA_UNROLL
            for (int kb = 0; kb < Cfg::KItersPerBlock; ++kb) {
                cute::gemm(
                    tiled_mma.with(tiled_mma.accumulate_,
                                   tmem_SFA(cute::_, cute::_, kb),
                                   tmem_SFB_mma(cute::_, cute::_, kb)),
                    tile_rA(cute::_, cute::_, kb, St),
                    tile_rB(cute::_, cute::_, kb, St),
                    tmem_acc(cute::_, cute::_, cute::_, this->acc_stage));
                tiled_mma.accumulate_ = cute::UMMA::ScaleOut::One;
            }
        };
    }
};

template <class Cfg, class Res>
CUTE_DEVICE auto make_mma_op(const Res& res, int base_tile, int base_acc,
                             int k_tps) {
    using Parts = GemmParts<Cfg>;
    using CfgGeom = GemmGeom<Cfg>;
    using Traits = typename CfgGeom::Traits;
    auto sA = Parts::smem_a(res.gemm_smem);
    auto sB = Parts::smem_b(res.gemm_smem);
    auto sSFA = Parts::smem_sfa(res.gemm_smem);
    auto sSFB = Parts::smem_sfb(res.gemm_smem);

    typename Traits::TiledMma tiled_mma;
    auto cta_mma = tiled_mma.get_slice(0);

    auto tile_rA = cta_mma.make_fragment_A(sA);
    auto tile_rB = cta_mma.make_fragment_B(sB);

    auto tmem_SFA = res.tmem.tmem_SFA;
    auto tmem_SFB = res.tmem.tmem_SFB;
    auto sm_sfa =
        cute::make_tensor(sSFA.data(), cute::filter_zeros(sSFA.layout()));
    auto sm_sfb =
        cute::make_tensor(sSFB.data(), cute::filter_zeros(sSFB.layout()));
    auto tm_sfa = cute::make_tensor(tmem_SFA.data(),
                                    cute::filter_zeros(tmem_SFA.layout()));
    auto tm_sfb = cute::make_tensor(tmem_SFB.data(),
                                    cute::filter_zeros(tmem_SFB.layout()));

    auto copy_sfa = cute::make_utccp_copy(typename CfgGeom::UtccpOp{}, tm_sfa);
    auto copy_sfb = cute::make_utccp_copy(typename CfgGeom::UtccpOp{}, tm_sfb);
    auto thr_sfa = copy_sfa.get_slice(0);
    auto thr_sfb = copy_sfb.get_slice(0);
    auto src_sfa = cute::get_utccp_smem_desc_tensor<typename CfgGeom::UtccpOp>(
        thr_sfa.partition_S(sm_sfa));
    auto dst_sfa = thr_sfa.partition_D(tm_sfa);
    auto src_sfb = cute::get_utccp_smem_desc_tensor<typename CfgGeom::UtccpOp>(
        thr_sfb.partition_S(sm_sfb));
    auto dst_sfb = thr_sfb.partition_D(tm_sfb);

    auto tmem_acc = res.tmem.tmem_accumulators;
    using Op = MmaOp<Cfg, Res, decltype(tiled_mma), decltype(tile_rA),
                     decltype(tile_rB), decltype(copy_sfa), decltype(copy_sfb),
                     decltype(src_sfa), decltype(dst_sfa), decltype(src_sfb),
                     decltype(dst_sfb), decltype(tmem_SFA), decltype(tmem_SFB),
                     decltype(tmem_acc)>;
    return Op{{res.bars, cute::elect_one_sync() != 0, base_tile, 0, k_tps,
               base_acc, 0, 0},
              tiled_mma,
              tile_rA,
              tile_rB,
              copy_sfa,
              copy_sfb,
              src_sfa,
              dst_sfa,
              src_sfb,
              dst_sfb,
              tmem_SFA,
              tmem_SFB,
              tmem_acc,
              tmem_SFB};
}

template <class Cfg, class Res, class TiledT2R, class ThrTacc, class Racc,
          class Coord>
struct DrainOp : DrainStageWalk<GemmGeom<Cfg>::AccStages> {
    using CfgGeom = GemmGeom<Cfg>;
    using PartialT = typename Cfg::PartialT;
    static constexpr bool Overlap = CfgGeom::AccOverlap;

    TiledT2R tiled_t2r;
    ThrTacc thr_tacc;
    Racc thr_racc;
    Coord coord;

    static constexpr int DrainSlices = CfgGeom::DrainSlices;
    static constexpr int DrainN = CfgGeom::DrainN;

    CUTE_DEVICE void load_slice(int nt) {
        cute::copy(tiled_t2r,
                   thr_tacc(cute::_, cute::_, cute::_, cute::Int<0>{}, nt,
                            this->acc_stage),
                   thr_racc);
        cutlass::arch::fence_view_async_tmem_load();
    }

    CUTE_DEVICE void arrive_windows(int step) const {
        if constexpr (Overlap) {
            if (step == CfgGeom::win_lo_step(this->acc_stage))
                arrive_acc_window_lo<Overlap>(this->bars, this->acc_stage);
            if (step == CfgGeom::win_hi_step(this->acc_stage))
                arrive_acc_window_hi<Overlap>(this->bars, this->acc_stage);
        }
    }

    CUTE_DEVICE void store(const GemmWorkItem& wi,
                           PartialT* __restrict__ output) {
        CUTLASS_PRAGMA_UNROLL
        for (int step = 0; step < DrainSlices; ++step) {
            const int nt = CfgGeom::drain_slice(step, this->acc_stage);
            load_slice(nt);
            arrive_windows(step);
            store_slice(wi, output, nt);
        }
    }

    CUTE_DEVICE void store_slice(const GemmWorkItem& wi,
                                 PartialT* __restrict__ output, int nt) const {
        const int stride = wi.out_row_stride;
        const int n_valid = wi.n_valid;
        const int a_row = cute::get<0>(coord(0));
        const int b_row = cute::get<1>(coord(0));
        if constexpr (Cfg::WeightInA) {
            PartialT* out_ptr =
                output + (int64_t)(b_row + nt * DrainN) * stride + a_row;
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < cute::size(thr_racc); ++i) {
                if (b_row + i < n_valid) *out_ptr = PartialT(thr_racc(i));
                out_ptr += stride;
            }
        } else {
            if (a_row >= n_valid) return;
            if (nt * DrainN >= wi.w_valid) return;
            constexpr int kPer = 32 / (int)sizeof(PartialT);
            PartialT* out_ptr =
                output + (int64_t)a_row * stride + b_row + nt * DrainN;
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < cute::size(thr_racc); i += kPer) {
                uint32_t words[8];
                PartialT* packed = reinterpret_cast<PartialT*>(words);
                CUTLASS_PRAGMA_UNROLL
                for (int j = 0; j < kPer; ++j)
                    packed[j] = PartialT(thr_racc(i + j));
                st_global_v8_na(out_ptr + i, words);
            }
        }
    }
};

template <class Cfg, class Res>
CUTE_DEVICE auto make_drain_op(const Res& res, int base_acc) {
    using CfgGeom = GemmGeom<Cfg>;
    using Traits = typename CfgGeom::Traits;
    using Roles = KernelBase::Roles;

    cute::Tensor tmem_acc = res.tmem.tmem_accumulators;
    cute::Tensor acc = tmem_acc(cute::make_coord(cute::_, cute::_), cute::_0{},
                                cute::_0{}, cute::_);
    cute::Tensor acc_tile = cute::flat_divide(acc, typename CfgGeom::EpiTile{});
    cute::TiledCopy tiled_t2r = cute::make_tmem_copy(
        typename CfgGeom::CopyOpT2R{},
        acc_tile(cute::_, cute::_, cute::_0{}, cute::_0{}, cute::_0{}));

    const int warp = KernelBase::warp_idx();
    const int lane = KernelBase::lane_idx();

    cute::ThrCopy thr_t2r =
        tiled_t2r.get_slice((warp % Roles::DrainWarps) * 32 + lane);
    cute::Tensor thr_tacc = thr_t2r.partition_S(acc_tile);
    cute::Tensor t2r_dst = thr_t2r.partition_D(
        acc_tile(cute::_, cute::_, cute::_0{}, cute::_0{}, cute::_0{}));
    cute::Tensor thr_racc = cute::make_tensor<float>(cute::shape(t2r_dst));
    cute::Tensor coord =
        thr_t2r.partition_D(cute::make_identity_tensor(cute::shape(
            acc_tile(cute::_, cute::_, cute::_0{}, cute::_0{}, cute::_0{}))));

    return DrainOp<Cfg, Res, decltype(tiled_t2r), decltype(thr_tacc),
                   decltype(thr_racc), decltype(coord)>{
        {res.bars, base_acc, 0, 0}, tiled_t2r, thr_tacc, thr_racc, coord};
}

template <class Cfg>
struct Bf16GemmTypes {
    static constexpr int BM = Cfg::BM;
    static constexpr int BN = Cfg::BN;
    static constexpr int BK = Cfg::BK;
    static constexpr int Stages = Cfg::TmaStages;

    using SmemLayoutA1 =
        decltype(make_umma_canonical_k_major_layout<BM, BK, 128,
                                                    cutlass::bfloat16_t>());
    using SmemLayoutB1 =
        decltype(make_umma_canonical_k_major_layout<BN, BK, 128,
                                                    cutlass::bfloat16_t>());

    static constexpr int SmemA =
        cute::cosize_v<SmemLayoutA1> * (int)sizeof(cutlass::bfloat16_t);
    static constexpr int SmemB =
        cute::cosize_v<SmemLayoutB1> * (int)sizeof(cutlass::bfloat16_t);
    static constexpr int ArenaStages = Cfg::OwnerTraits::MaxTmaStages;

    using Fp8Slot = GemmInst<typename Cfg::OwnerTraits::Qkv, Stages>;
    static constexpr int SlotA = Fp8Slot::SmemA;
    static constexpr int SlotB = Fp8Slot::SlotB;
    static constexpr bool Overlap = Cfg::AccSharedCols > 0;
    static constexpr int OffA = 0;
    static constexpr int OffB = ArenaStages * SlotA;

    using MmaAtom =
        SM100_MMA_F16BF16_SS_NOELECT<cutlass::bfloat16_t, cutlass::bfloat16_t,
                                     float, BM, BN, cute::UMMA::Major::K,
                                     cute::UMMA::Major::K>;
    using TiledMma = decltype(cute::make_tiled_mma(MmaAtom{}));

    using TileA = cute::Shape<cute::Int<BM>, cute::Int<BK>>;
    using TileB = cute::Shape<cute::Int<BN>, cute::Int<BK>>;

    using CopyOpT2R = TmemLoad32dp32b<BN>;
};

template <class Cfg>
CUTE_HOST auto make_bf16_gemm_tma_w(const void* weight, int w_out, int k_dim) {
    using Types = Bf16GemmTypes<Cfg>;
    auto layout = cute::make_layout(cute::make_shape(w_out, k_dim),
                                    cute::make_stride(k_dim, cute::_1{}));
    return cute::make_tma_copy(
        cute::SM90_TMA_LOAD{},
        cute::make_tensor(cute::make_gmem_ptr(
                              static_cast<const cutlass::bfloat16_t*>(weight)),
                          layout),
        typename Types::SmemLayoutA1{});
}

template <class Cfg>
CUTE_HOST auto make_bf16_gemm_tma_act(const void* act, int num_rows,
                                      int row_stride, int k_dim) {
    using Types = Bf16GemmTypes<Cfg>;
    auto layout = cute::make_layout(cute::make_shape(num_rows, k_dim),
                                    cute::make_stride(row_stride, cute::_1{}));
    return cute::make_tma_copy(
        cute::SM90_TMA_LOAD{},
        cute::make_tensor(
            cute::make_gmem_ptr(static_cast<const cutlass::bfloat16_t*>(act)),
            layout),
        typename Types::SmemLayoutB1{});
}

struct Bf16GemmItem {
    int w_tile;
    int n_tile;
    int n_valid;
};

template <class Cfg>
CUTE_DEVICE Bf16GemmItem bf16_gemm_item_at(int it, int n_groups,
                                           int num_tokens) {
    const int tg = it % n_groups;
    const int mt = it / n_groups;
    return Bf16GemmItem{mt, tg, min(Cfg::BN, num_tokens - tg * Cfg::BN)};
}

template <class Cfg, class TmaW, class TmaAct>
struct Bf16GemmResources {
    const TmaW& tma_w;
    const TmaAct& tma_act;
    char* __restrict__ gemm_smem;
    GemmPipelineBars bars;
    uint32_t tmem_base;
    int act_rows;
    int w_out;
    int k_dim;
    int k_tiles;
    float* __restrict__ ring;
    const int* __restrict__ ring_rows;
};

template <class Cfg, int St>
CUTE_DEVICE auto bf16_smem_a(char* smem) {
    using Types = Bf16GemmTypes<Cfg>;
    return cute::make_tensor(
        cute::make_smem_ptr(reinterpret_cast<cutlass::bfloat16_t*>(
            smem + Types::OffA + St * Types::SlotA)),
        typename Types::SmemLayoutA1{});
}
template <class Cfg, int St>
CUTE_DEVICE auto bf16_smem_b(char* smem) {
    using Types = Bf16GemmTypes<Cfg>;
    return cute::make_tensor(
        cute::make_smem_ptr(reinterpret_cast<cutlass::bfloat16_t*>(
            smem + Types::OffB + St * Types::SlotB)),
        typename Types::SmemLayoutB1{});
}

template <class Cfg, class Res>
struct Bf16TmaOpA : TmaStageWalk<Bf16TmaOpA<Cfg, Res>, Cfg::TmaStages> {
    using Types = Bf16GemmTypes<Cfg>;

    const Res& res;
    int w_tile;

    CUTE_DEVICE void begin_item(const Bf16GemmItem& it) { w_tile = it.w_tile; }

    CUTE_DEVICE auto issuer() {
        return [this](auto St, int k) {
            constexpr int stage = decltype(St)::value;
            res.bars.tma_a[stage].arrive_and_expect_tx(Types::SmemA);
            uint64_t& mbar = reinterpret_cast<uint64_t&>(res.bars.tma_a[stage]);
            auto mA = res.tma_w.get_tma_tensor(
                cute::make_shape(res.w_out, res.k_dim));
            auto thr_w = res.tma_w.get_slice(cute::_0{});
            auto tA = cute::local_tile(mA, typename Types::TileA{},
                                       cute::make_coord(w_tile, k));
            cute::copy(
                res.tma_w.with(mbar, 0), thr_w.partition_S(tA),
                thr_w.partition_D(bf16_smem_a<Cfg, stage>(res.gemm_smem)));
        };
    }
};

template <class Cfg, class Res>
struct Bf16TmaOpB : TmaStageWalk<Bf16TmaOpB<Cfg, Res>, Cfg::TmaStages> {
    using Types = Bf16GemmTypes<Cfg>;

    const Res& res;
    int n_tile;

    CUTE_DEVICE void begin_item(const Bf16GemmItem& it) { n_tile = it.n_tile; }

    CUTE_DEVICE auto issuer() {
        return [this](auto St, int k) {
            constexpr int stage = decltype(St)::value;
            res.bars.tma_b[stage].arrive_and_expect_tx(Types::SmemB);
            uint64_t& mbar = reinterpret_cast<uint64_t&>(res.bars.tma_b[stage]);
            auto mB = res.tma_act.get_tma_tensor(
                cute::make_shape(res.act_rows, res.k_dim));
            auto thr_act = res.tma_act.get_slice(cute::_0{});
            auto tB = cute::local_tile(mB, typename Types::TileB{},
                                       cute::make_coord(n_tile, k));
            cute::copy(
                res.tma_act.with(mbar, 0), thr_act.partition_S(tB),
                thr_act.partition_D(bf16_smem_b<Cfg, stage>(res.gemm_smem)));
        };
    }
};

template <class Cfg, class Res>
struct Bf16MmaOp : MmaStageWalk<Cfg::TmaStages, Cfg::AccStages,
                                Bf16GemmTypes<Cfg>::Overlap> {
    using Types = Bf16GemmTypes<Cfg>;

    const Res& res;
    typename Types::TiledMma tiled_mma;

    CUTE_DEVICE void begin_item() { this->acquire_acc(); }

    CUTE_DEVICE void run_k_tile(int k) {
        tcgen05_after_thread_sync();
        if (!this->elected) return;
        dispatch_stage<Cfg::TmaStages>(this->smem_stage(k), stager(), k);
    }

    CUTE_DEVICE auto stager() {
        return [this](auto St, int k) {
            constexpr int stage = decltype(St)::value;
            cute::Tensor tmem_frag = cute::partition_fragment_C(
                tiled_mma,
                cute::Shape<cute::Int<Cfg::BM>, cute::Int<Cfg::BN>>{});
            tmem_frag.data().get() =
                res.tmem_base + (uint32_t)(this->acc_stage * Cfg::AccStride);
            umma_ss(tiled_mma, bf16_smem_a<Cfg, stage>(res.gemm_smem),
                    bf16_smem_b<Cfg, stage>(res.gemm_smem), tmem_frag, k == 0);
        };
    }
};

template <class Cfg, class Res, class TiledT2R, class ThrTacc, class Racc,
          class Coord>
struct Bf16DrainOp : DrainStageWalk<Cfg::AccStages> {
    using Types = Bf16GemmTypes<Cfg>;
    using Walk = DrainStageWalk<Cfg::AccStages>;

    const Res& res;
    TiledT2R tiled_t2r;
    ThrTacc thr_tacc;
    Racc thr_racc;
    Coord coord;
    uint32_t acc_addr;

    CUTE_DEVICE void wait_acc_full() {
        Walk::wait_acc_full();
        acc_addr = thr_tacc.data().get() +
                   (uint32_t)(this->acc_stage * Cfg::AccStride);
    }

    CUTE_DEVICE void store_ring(const Bf16GemmItem& it, int i,
                                float value) const {
        const int row = cute::get<0>(coord(i));
        const int act_row = cute::get<1>(coord(i));
        if (act_row < it.n_valid) {
            const int ring_row = res.ring_rows[it.n_tile * Cfg::BN + act_row];
            res.ring[(int64_t)ring_row * res.w_out + it.w_tile * Cfg::BM +
                     row] = value;
        }
    }

    CUTE_DEVICE void release_windows() const {
        arrive_acc_window_lo<Types::Overlap>(this->bars, this->acc_stage);
        arrive_acc_window_hi<Types::Overlap>(this->bars, this->acc_stage);
    }

    CUTE_DEVICE void store_unrolled(const Bf16GemmItem& it) {
        ThrTacc src = thr_tacc;
        src.data() = acc_addr;
        cute::copy(tiled_t2r, src, thr_racc);
        cutlass::arch::fence_view_async_tmem_load();
        release_windows();
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < cute::size(thr_racc); ++i)
            store_ring(it, i, thr_racc(i));
    }

    CUTE_DEVICE void store_rolled(const Bf16GemmItem& it) const {
        constexpr int kChunk =
            cute::size(Racc{}) < 32 ? (int)cute::size(Racc{}) : 32;
        constexpr int kNumChunks = cute::size(Racc{}) / kChunk;
        float cur[kChunk], nxt[kChunk];
        tmem_ld_32dp32bNx<kChunk>(acc_addr, nxt);
        cutlass::arch::fence_view_async_tmem_load();
        CUTE_NO_UNROLL
        for (int ch = 0; ch < kNumChunks; ++ch) {
            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < kChunk; ++j) cur[j] = nxt[j];
            if (ch + 1 < kNumChunks)
                tmem_ld_32dp32bNx<kChunk>(acc_addr + (ch + 1) * kChunk, nxt);
            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < kChunk; ++j)
                store_ring(it, ch * kChunk + j, cur[j]);
            cutlass::arch::fence_view_async_tmem_load();
        }
        release_windows();
    }
};

template <class Cfg, class Res>
CUTE_DEVICE auto make_bf16_weight_tma_op(const Res& res, int base_tile) {
    return Bf16TmaOpA<Cfg, Res>{
        {res.bars, cute::elect_one_sync() != 0, base_tile, 0, res.k_tiles},
        res,
        0};
}

template <class Cfg, class Res>
CUTE_DEVICE auto make_bf16_act_tma_op(const Res& res, int base_tile) {
    return Bf16TmaOpB<Cfg, Res>{
        {res.bars, cute::elect_one_sync() != 0, base_tile, 0, res.k_tiles},
        res,
        0};
}

template <class Cfg, class Res>
CUTE_DEVICE auto make_bf16_mma_op(const Res& res, int base_tile, int base_acc) {
    return Bf16MmaOp<Cfg, Res>{{res.bars, cute::elect_one_sync() != 0,
                                base_tile, 0, res.k_tiles, base_acc, 0, 0},
                               res,
                               typename Bf16GemmTypes<Cfg>::TiledMma{}};
}

template <class Cfg, class Res>
CUTE_DEVICE auto make_bf16_drain_op(const Res& res, int base_acc) {
    using Types = Bf16GemmTypes<Cfg>;
    using Roles = KernelBase::Roles;

    typename Types::TiledMma tiled_mma;

    cute::Tensor frag0 = cute::partition_fragment_C(
        tiled_mma, cute::Shape<cute::Int<Cfg::BM>, cute::Int<Cfg::BN>>{});
    frag0.data().get() = res.tmem_base;

    cute::Tensor acc0 =
        frag0(cute::make_coord(cute::_, cute::_), cute::_0{}, cute::_0{});
    cute::TiledCopy tiled_t2r =
        cute::make_tmem_copy(typename Types::CopyOpT2R{}, acc0);

    const int warp = KernelBase::warp_idx();
    const int lane = KernelBase::lane_idx();
    cute::ThrCopy thr_t2r =
        tiled_t2r.get_slice((warp % Roles::DrainWarps) * 32 + lane);
    cute::Tensor t2r_dst = thr_t2r.partition_D(acc0);
    cute::Tensor thr_racc = cute::make_tensor<float>(cute::shape(t2r_dst));
    cute::Tensor coord =
        thr_t2r.partition_D(cute::make_identity_tensor(cute::shape(acc0)));

    using Op = Bf16DrainOp<Cfg, Res, decltype(tiled_t2r),
                           decltype(thr_t2r.partition_S(acc0)),
                           decltype(thr_racc), decltype(coord)>;

    return Op{{res.bars, base_acc, 0, 0},
              res,
              tiled_t2r,
              thr_t2r.partition_S(acc0),
              thr_racc,
              coord,
              0};
}

template <class Op, class Begin>
CUTE_DEVICE void issue_items(Op& op, int first, int end, int step, int k_tiles,
                             Begin&& begin) {
    for (int item = first; item < end; item += step) {
        begin(item);
        CUTLASS_PRAGMA_NO_UNROLL
        for (int k = 0; k < k_tiles; ++k) op.issue_tile(k);
        op.end_item();
    }
}

template <class Op, class Begin>
CUTE_DEVICE void mma_items(Op& op, int first, int end, int step, int k_tiles,
                           Begin&& begin) {
    for (int item = first; item < end; item += step) {
        begin(item);
        CUTLASS_PRAGMA_NO_UNROLL
        for (int k = 0; k < k_tiles; ++k) {
            op.wait_tma(k);
            op.run_k_tile(k);
            op.commit(k);
        }
        op.end_item();
    }
}

template <class Op, class Store, class Publish>
CUTE_DEVICE void drain_items(Op& op, int first, int end, int step,
                             Store&& store, Publish&& publish) {
    for (int item = first; item < end; item += step) {
        op.wait_acc_full();
        const auto stored = store(item);
        op.end_item();
        publish(item, stored);
    }
}
}  // namespace halops::mla
