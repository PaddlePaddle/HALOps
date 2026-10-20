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

#include <cutlass/numeric_types.h>

#include <cute/tensor.hpp>

#include "mla_intrinsics.cuh"
#include "mla_traits.cuh"
#include "mla_utils.cuh"
#include "umma_gemm.cuh"

namespace mega_mla {

template <class Cfg>
struct GemmGeom {
    using Inst = GemmInstFor<Cfg>;
    using Traits = typename Inst::Traits;
    using Mainloop = typename Inst::Mainloop;
    using TileShape = typename Traits::TileShape;
    using UtccpOp = typename Mainloop::UtccpOp;

    static constexpr int STAGES = Cfg::TmaStages;

    static constexpr int ACC_STAGES = Cfg::AccStages;
    static constexpr int TileT = Inst::TileT;
    static constexpr bool SF_TILE_PER_GROUP = Inst::SF_TILE_PER_GROUP;
    static constexpr int VS_PER_CTA_N = Inst::VS_PER_CTA_N;
    static constexpr int SFB_COLS_PER_SUB_SLOT = Inst::SFB_COLS_PER_SUB_SLOT;
    static_assert(STAGES > 0,
                  "a waiter multiplies stage_uses by the items already walked "
                  "to get this barrier's running arrive count, so producer "
                  "back-pressure must be the per-stage bar_gemm_smem_free, "
                  "never bar_mma (phase, not count)");
};

template <int BN, class TmemStorage>
struct GemmResources {
    const WeightAtoms<BN>& weight;
    const ActAtoms<BN>& act;
    char* __restrict__ gemm_smem;
    TmemStorage& tmem;

    GemmPipelineBars bars;
};

template <class Cfg, class TmemStorage>
CUTE_DEVICE const GemmAtomsA<Cfg::BN>& a_side(
    const GemmResources<Cfg::BN, TmemStorage>& res) {
    if constexpr (Cfg::SwapAB)
        return res.weight;
    else
        return res.act;
}

template <class Cfg, class TmemStorage>
CUTE_DEVICE const GemmAtomsB<Cfg::BN>& b_side(
    const GemmResources<Cfg::BN, TmemStorage>& res) {
    if constexpr (Cfg::SwapAB)
        return res.act;
    else
        return res.weight;
}

struct NoGate {
    CUTE_DEVICE void operator()(int) const {}
};

template <class Cfg, class Gate>
CUTE_DEVICE auto gate_on_a(const Gate& g) {
    static_assert(
        Cfg::BM > 0,
        "The activation is produced by this kernel's own vec warps, so its TMA "
        "must first wait on a per-token-group readiness gate; the weight is a "
        "host tensor and needs none.  The gate therefore belongs to the "
        "ACTIVATION, not to a warp -- these two selectors are the only place "
        "that turns that into 'which slot waits', and they are why no item "
        "loop "
        "mentions SwapAB.  A gate left on the wrong slot does not fail loudly: "
        "the TMA outruns the quantizer and reads workspace that was never "
        "written, so the SAME .so returns a DIFFERENT wrong answer each run.");
    if constexpr (Cfg::SwapAB)
        return NoGate{};
    else
        return g;
}

template <class Cfg, class Gate>
CUTE_DEVICE auto gate_on_b(const Gate& g) {
    if constexpr (Cfg::SwapAB)
        return g;
    else
        return NoGate{};
}

struct GemmWorkItem {
    int n_row;
    int m_tile;
    int m_total;

    int k_tile_begin;

    int a_k_tile_begin;
    int out_row_stride;
    int n_valid;
};

template <class Cfg>
struct GemmParts {
    using Inst = GemmInstFor<Cfg>;

    static CUTE_DEVICE auto smem_a(char* p) {
        return cute::make_tensor(
            cute::make_smem_ptr(reinterpret_cast<e4m3*>(p + Inst::OFF_A)),
            typename Inst::SmemLayoutAStaged{});
    }
    static CUTE_DEVICE auto smem_b(char* p) {
        return cute::make_tensor(
            cute::make_smem_ptr(reinterpret_cast<e4m3*>(p + Inst::OFF_B)),
            typename Inst::SmemLayoutBStaged{});
    }
    static CUTE_DEVICE auto smem_sfa(char* p) {
        return cute::make_tensor(
            cute::make_smem_ptr(reinterpret_cast<ue8m0*>(p + Inst::OFF_SFA)),
            typename Inst::SmemLayoutSFAStaged{});
    }
    static CUTE_DEVICE auto smem_sfb(char* p) {
        return cute::make_tensor(
            cute::make_smem_ptr(reinterpret_cast<ue8m0*>(p + Inst::OFF_SFB)),
            typename Inst::SmemLayoutSFBStaged{});
    }
};

template <class Cfg>
CUTE_DEVICE auto tile_gA(const GemmAtomsA<Cfg::BN>& at, int m_total) {
    cute::Tensor mA = at.tma_a.get_tma_tensor(cute::make_shape(
        m_total, at.k_dim, 1))(cute::_, cute::_, cute::Int<0>{});
    return cute::local_tile(mA, typename GemmGeom<Cfg>::TileShape{},
                            cute::make_coord(cute::_, cute::_, cute::_),
                            cute::Step<cute::_1, cute::X, cute::_1>{});
}
template <class Cfg>
CUTE_DEVICE auto tile_gSFA(const GemmAtomsA<Cfg::BN>& at) {
    cute::Tensor mSFA = at.tma_sfa.get_tma_tensor(at.layout_sfa.shape())(
        cute::_, cute::_, cute::Int<0>{});
    return cute::local_tile(mSFA, typename GemmGeom<Cfg>::TileShape{},
                            cute::make_coord(cute::_, cute::_, cute::_),
                            cute::Step<cute::_1, cute::X, cute::_1>{});
}

template <class Cfg, class TmemStorage, class SA, class SSFA, class CtaMma,
          class GA, class GSFA, class Gate>
struct TmaOpA {
    using G = GemmGeom<Cfg>;
    using Inst = typename G::Inst;
    static constexpr int STAGES = G::STAGES;

    const GemmResources<Cfg::BN, TmemStorage>& res;
    SA sA;
    SSFA sSFA;
    CtaMma cta_mma;
    bool elected;

    uint32_t base_bits;
    uint32_t base_zero_bits;
    int seq;
    int k_tps;

    const GemmAtomsA<Cfg::BN>* at;
    GA gA;
    GSFA gSFA;
    int m_tile;
    int a_k_begin;
    Gate gate;

    CUTE_DEVICE void begin_item(const WeightAtoms<Cfg::BN>& atoms,
                                const GemmWorkItem& wi) {
        gate(wi.n_row / G::TileT);
        if constexpr (Cfg::SwapAB) {
            at = &atoms;
            gA = tile_gA<Cfg>(atoms, wi.m_total);
            gSFA = tile_gSFA<Cfg>(atoms);
            m_tile = wi.m_tile;
            a_k_begin = wi.a_k_tile_begin;
        } else {
            m_tile = wi.n_row / G::TileT;
            a_k_begin = wi.k_tile_begin;
        }
    }

    CUTE_DEVICE void issue_tile(int t, int site, int item) {
        wait_stage_free<STAGES>(res.bars, base_bits, base_zero_bits, seq, k_tps,
                                t, site, item);
        if (elected) dispatch_stage<STAGES>(t % STAGES, issuer(), t);
    }

    CUTE_DEVICE void end_item() { ++seq; }

    CUTE_DEVICE auto issuer() {
        return [this](auto St, int k) {
            const int ka = k + a_k_begin;
            constexpr int s = decltype(St)::value;
            res.bars.tma_a[s].arrive_and_expect_tx(Inst::TMA_BYTES_A);
            uint64_t& mbar = reinterpret_cast<uint64_t&>(res.bars.tma_a[s]);
            auto [tma_gA, tma_sA] = cute::tma_partition(
                at->tma_a, cute::Int<0>{}, cute::make_layout(cute::Int<1>{}),
                cute::group_modes<0, 3>(sA(cute::_, cute::_, cute::_, St)),
                cute::group_modes<0, 3>(cta_mma.partition_A(gA)));
            auto [tma_gSFA, tma_sSFA] = cute::tma_partition(
                at->tma_sfa, cute::Int<0>{}, cute::make_layout(cute::Int<1>{}),
                cute::group_modes<0, 3>(sSFA(cute::_, cute::_, cute::_, St)),
                cute::group_modes<0, 3>(cta_mma.partition_A(gSFA)));
            cute::copy(at->tma_a.with(mbar, 0), tma_gA(cute::_, m_tile, ka),
                       tma_sA);
            cute::copy(at->tma_sfa.with(mbar, 0), tma_gSFA(cute::_, m_tile, ka),
                       tma_sSFA);
        };
    }
};

template <class Cfg, class TmemStorage, class Gate>
CUTE_DEVICE auto make_tma_op_a(const GemmResources<Cfg::BN, TmemStorage>& res,
                               uint32_t base_bits, uint32_t base_zero_bits,
                               int k_tps, const Gate& act_gate) {
    using P = GemmParts<Cfg>;
    typename GemmGeom<Cfg>::Traits::TiledMma tiled_mma;
    auto sA = P::smem_a(res.gemm_smem);
    auto sSFA = P::smem_sfa(res.gemm_smem);
    auto cta_mma = tiled_mma.get_slice(0);
    auto gA = tile_gA<Cfg>(a_side<Cfg>(res), a_side<Cfg>(res).m_total);
    auto gSFA = tile_gSFA<Cfg>(a_side<Cfg>(res));
    auto gate = gate_on_a<Cfg>(act_gate);
    return TmaOpA<Cfg, TmemStorage, decltype(sA), decltype(sSFA),
                  decltype(cta_mma), decltype(gA), decltype(gSFA),
                  decltype(gate)>{res,
                                  sA,
                                  sSFA,
                                  cta_mma,
                                  cute::elect_one_sync() != 0,
                                  base_bits,
                                  base_zero_bits,
                                  0,
                                  k_tps,
                                  &a_side<Cfg>(res),
                                  gA,
                                  gSFA,
                                  0,
                                  0,
                                  gate};
}

template <class Cfg>
CUTE_DEVICE auto tile_gB(const GemmAtomsB<Cfg::BN>& at) {
    const int k_dim = at.k_dim;
    cute::Tensor mB = at.tma_b.get_tma_tensor(
        cute::make_shape(at.n_pad, k_dim, 1))(cute::_, cute::_, cute::Int<0>{});
    return cute::local_tile(mB, typename GemmGeom<Cfg>::TileShape{},
                            cute::make_coord(cute::_, cute::_, cute::_),
                            cute::Step<cute::X, cute::_1, cute::_1>{});
}

template <class Cfg>
CUTE_DEVICE auto tile_gSFB(const GemmAtomsB<Cfg::BN>& at) {
    cute::Tensor mSFBg = at.tma_sfb.get_tma_tensor(at.layout_sfb.shape())(
        cute::_, cute::_, cute::Int<0>{});
    return cute::local_tile(mSFBg,
                            typename GemmGeom<Cfg>::Traits::TileShape_SF{},
                            cute::make_coord(cute::_, cute::_, cute::_),
                            cute::Step<cute::X, cute::_1, cute::_1>{});
}

template <class Cfg, class TmemStorage, class SB, class SSFB, class CtaMma,
          class CtaMmaSf, class GB, class GSFB, class Gate>
struct TmaOpB {
    using G = GemmGeom<Cfg>;
    using Inst = typename G::Inst;
    static constexpr int STAGES = G::STAGES;

    const GemmResources<Cfg::BN, TmemStorage>& res;
    SB sB;
    SSFB sSFB;
    CtaMma cta_mma;
    CtaMmaSf cta_mma_sfb;
    GB gB;
    GSFB gSFB;
    bool elected;

    uint32_t base_bits;
    uint32_t base_zero_bits;
    int seq;
    int k_tps;

    const GemmAtomsB<Cfg::BN>* at;
    int n_tile;
    int nb_sf;
    int k_begin;
    Gate gate;

    CUTE_DEVICE void begin_item(const WeightAtoms<Cfg::BN>& w_atoms,
                                const GemmWorkItem& wi) {
        gate(wi.n_row / G::TileT);
        if constexpr (Cfg::SwapAB) {
            n_tile = wi.n_row / G::TileT;
            k_begin = wi.k_tile_begin;
        } else {
            at = &w_atoms;
            gB = tile_gB<Cfg>(w_atoms);
            gSFB = tile_gSFB<Cfg>(w_atoms);
            n_tile = wi.m_tile;
            k_begin = wi.a_k_tile_begin;
        }
        if constexpr (G::SF_TILE_PER_GROUP)
            nb_sf = n_tile;
        else
            nb_sf = n_tile / G::VS_PER_CTA_N;
    }

    CUTE_DEVICE void issue_tile(int t, int site, int item) {
        wait_stage_free<STAGES>(res.bars, base_bits, base_zero_bits, seq, k_tps,
                                t, site, item);
        if (elected) dispatch_stage<STAGES>(t % STAGES, issuer(), t);
    }

    CUTE_DEVICE void end_item() { ++seq; }

    CUTE_DEVICE auto issuer() {
        return [this](auto St, int k) {
            const int kb = k + k_begin;
            constexpr int s = decltype(St)::value;
            res.bars.tma_b[s].arrive_and_expect_tx(Inst::TMA_BYTES_B);
            uint64_t& mbar = reinterpret_cast<uint64_t&>(res.bars.tma_b[s]);
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

template <class Cfg, class TmemStorage, class Gate>
CUTE_DEVICE auto make_tma_op_b(const GemmResources<Cfg::BN, TmemStorage>& res,
                               uint32_t base_bits, uint32_t base_zero_bits,
                               int k_tps, const Gate& act_gate) {
    using P = GemmParts<Cfg>;
    using Traits = typename GemmGeom<Cfg>::Traits;
    typename Traits::TiledMma tiled_mma;
    typename Traits::TiledMma_SF tiled_mma_sf;
    auto sB = P::smem_b(res.gemm_smem);
    auto sSFB = P::smem_sfb(res.gemm_smem);
    auto cta_mma = tiled_mma.get_slice(0);
    auto cta_mma_sfb = tiled_mma_sf.get_slice(0);
    auto gB = tile_gB<Cfg>(b_side<Cfg>(res));
    auto gSFB = tile_gSFB<Cfg>(b_side<Cfg>(res));
    auto gate = gate_on_b<Cfg>(act_gate);
    return TmaOpB<Cfg, TmemStorage, decltype(sB), decltype(sSFB),
                  decltype(cta_mma), decltype(cta_mma_sfb), decltype(gB),
                  decltype(gSFB), decltype(gate)>{res,
                                                  sB,
                                                  sSFB,
                                                  cta_mma,
                                                  cta_mma_sfb,
                                                  gB,
                                                  gSFB,
                                                  cute::elect_one_sync() != 0,
                                                  base_bits,
                                                  base_zero_bits,
                                                  0,
                                                  k_tps,
                                                  &b_side<Cfg>(res),
                                                  0,
                                                  0,
                                                  0,
                                                  gate};
}

template <class Cfg, class TmemStorage, class TiledMmaT, class RA, class RB,
          class CopySFA, class CopySFB, class SrcSFA, class DstSFA,
          class SrcSFB, class DstSFB, class TmemSFA, class TmemSFB,
          class TmemAcc>
struct MmaOp {
    using G = GemmGeom<Cfg>;
    static constexpr int STAGES = G::STAGES;

    static constexpr int ACC_STAGES = G::ACC_STAGES;

    const GemmResources<Cfg::BN, TmemStorage>& res;
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
    bool elected;

    uint32_t base_bits;
    int seq;
    int k_tps;

    int base_acc;
    int acc_seq;

    TmemSFB tmem_SFB_mma;
    int acc_slot;

    static CUTE_DEVICE void assert_stage_modes(RA const& tile_rA,
                                               SrcSFA const& src_sfa) {
        static_assert(
            cute::rank<4>(SrcSFA{}.layout()) == 1 &&
                cute::size<4>(SrcSFA{}.layout()) == STAGES,
            "the UTCCP smem descriptor's last mode must be the A-side stage, "
            "and it must have exactly TmaStages of them");
        static_assert(cute::size<3>(RA{}.layout()) == STAGES,
                      "tile_rA mode 3 must be the A-side stage");
    }

    CUTE_DEVICE void begin_item(const GemmWorkItem& wi, int site, int item) {
        tmem_SFB_mma = tmem_SFB;
        if constexpr (!G::SF_TILE_PER_GROUP && G::VS_PER_CTA_N > 1) {
            const int n_tile = wi.n_row / G::TileT;
            tmem_SFB_mma.data() =
                tmem_SFB_mma.data().get() +
                (n_tile % G::VS_PER_CTA_N) * G::SFB_COLS_PER_SUB_SLOT;
        }
        acc_slot = reserve_acc_slot<ACC_STAGES>(res.bars, base_acc, acc_seq,
                                                site, item);
    }

    CUTE_DEVICE void wait_tma(int k, int site_a, int site_b, int item) {
        wait_stage_tma<STAGES>(res.bars, base_bits, seq, k_tps, k, site_a,
                               site_b, item);
    }

    CUTE_DEVICE void run_k_tile(int k) {
        if (k == 0) tiled_mma.accumulate_ = cute::UMMA::ScaleOut::Zero;
        ku::tcgen05_after_thread_sync();
        dispatch_stage<STAGES>(k % STAGES, stager(), k);
    }

    CUTE_DEVICE void commit(int k) {
        tiled_mma.accumulate_ = cute::UMMA::ScaleOut::One;
        __syncwarp();
        if (elected) ku::umma_arrive_noelect(res.bars.smem_free[k % STAGES]);
    }

    CUTE_DEVICE void end_item() {
        if (elected) ku::umma_arrive_noelect(res.bars.acc_full[acc_slot]);
        ++seq;
        ++acc_seq;
    }

    CUTE_DEVICE auto stager() {
        return [this](auto St, int) {
            if (elected) {
                cute::copy(copy_sfa,
                           src_sfa(cute::_, cute::_, cute::_, cute::_, St),
                           dst_sfa(cute::_, cute::_, cute::_, cute::_));
                cute::copy(copy_sfb,
                           src_sfb(cute::_, cute::_, cute::_, cute::_, St),
                           dst_sfb(cute::_, cute::_, cute::_, cute::_));
            }
            CUTLASS_PRAGMA_UNROLL
            for (int kb = 0; kb < GemmShapeDefaults::KItersPerBlock; ++kb) {
                cute::gemm(tiled_mma.with(tiled_mma.accumulate_,
                                          tmem_SFA(cute::_, cute::_, kb),
                                          tmem_SFB_mma(cute::_, cute::_, kb)),
                           tile_rA(cute::_, cute::_, kb, St),
                           tile_rB(cute::_, cute::_, kb, St),
                           tmem_acc(cute::_, cute::_, cute::_, acc_slot));
                tiled_mma.accumulate_ = cute::UMMA::ScaleOut::One;
            }
        };
    }
};

template <class Cfg, class TmemStorage>
CUTE_DEVICE auto make_mma_op(const GemmResources<Cfg::BN, TmemStorage>& res,
                             uint32_t base_bits, int base_acc, int k_tps) {
    using P = GemmParts<Cfg>;
    using G = GemmGeom<Cfg>;
    using Traits = typename G::Traits;
    auto sA = P::smem_a(res.gemm_smem);
    auto sB = P::smem_b(res.gemm_smem);
    auto sSFA = P::smem_sfa(res.gemm_smem);
    auto sSFB = P::smem_sfb(res.gemm_smem);

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

    auto copy_sfa = cute::make_utccp_copy(typename G::UtccpOp{}, tm_sfa);
    auto copy_sfb = cute::make_utccp_copy(typename G::UtccpOp{}, tm_sfb);
    auto thr_sfa = copy_sfa.get_slice(0);
    auto thr_sfb = copy_sfb.get_slice(0);
    auto src_sfa = cute::get_utccp_smem_desc_tensor<typename G::UtccpOp>(
        thr_sfa.partition_S(sm_sfa));
    auto dst_sfa = thr_sfa.partition_D(tm_sfa);
    auto src_sfb = cute::get_utccp_smem_desc_tensor<typename G::UtccpOp>(
        thr_sfb.partition_S(sm_sfb));
    auto dst_sfb = thr_sfb.partition_D(tm_sfb);

    auto tmem_acc = res.tmem.tmem_accumulators;
    using Op = MmaOp<Cfg, TmemStorage, decltype(tiled_mma), decltype(tile_rA),
                     decltype(tile_rB), decltype(copy_sfa), decltype(copy_sfb),
                     decltype(src_sfa), decltype(dst_sfa), decltype(src_sfb),
                     decltype(dst_sfb), decltype(tmem_SFA), decltype(tmem_SFB),
                     decltype(tmem_acc)>;
    Op::assert_stage_modes(tile_rA, src_sfa);

    return Op{res,       tiled_mma,
              tile_rA,   tile_rB,
              copy_sfa,  copy_sfb,
              src_sfa,   dst_sfa,
              src_sfb,   dst_sfb,
              tmem_SFA,  tmem_SFB,
              tmem_acc,  cute::elect_one_sync() != 0,
              base_bits, 0,
              k_tps,     base_acc,
              0,         tmem_SFB,
              0};
}

template <class Cfg, class TmemStorage, class TiledT2R, class ThrTacc,
          class Racc, class Coord>
struct DrainOp {
    using G = GemmGeom<Cfg>;
    using PartialT = typename Cfg::PartialT;

    static constexpr int ACC_STAGES = G::ACC_STAGES;

    const GemmResources<Cfg::BN, TmemStorage>& res;
    TiledT2R tiled_t2r;
    ThrTacc thr_tacc;
    Racc thr_racc;
    Coord coord;

    int base_acc;
    int acc_seq;
    int acc_slot;

    CUTE_DEVICE void wait_mma_and_load(int site, int item) {
        acc_slot =
            claim_acc_slot<ACC_STAGES>(res.bars, base_acc, acc_seq, site, item);
        ku::tcgen05_after_thread_sync();
        cute::copy(tiled_t2r,
                   thr_tacc(cute::_, cute::_, cute::_, cute::Int<0>{},
                            cute::Int<0>{}, acc_slot),
                   thr_racc);
        cutlass::arch::fence_view_async_tmem_load();
    }

    CUTE_DEVICE void store(const GemmWorkItem& wi,
                           PartialT* __restrict__ output) const {
        const int stride = wi.out_row_stride;
        const int n_valid = wi.n_valid;
        const int a_row = cute::get<0>(coord(0));
        const int b_row = cute::get<1>(coord(0));
        const int last = cute::size(thr_racc) - 1;
        KU_TRAP_ONLY_DEVICE_ASSERT(cute::get<0>(coord(last)) == a_row &&
                                   cute::get<1>(coord(last)) == b_row + last);
        if constexpr (Cfg::SwapAB) {
            PartialT* p = output + (int64_t)b_row * stride + a_row;
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < cute::size(thr_racc); ++i) {
                if (b_row + i < n_valid) *p = PartialT(thr_racc(i));
                p += stride;
            }
        } else {
            if (a_row >= n_valid) return;
            static_assert(
                cute::size(Racc{}) % (32 / (int)sizeof(PartialT)) == 0,
                "the non-swap drain writes one 32 B stg.256 per "
                "chunk, so a thread's element run must be a whole "
                "number of them");
            constexpr int PER = 32 / (int)sizeof(PartialT);
            PartialT* p = output + (int64_t)a_row * stride + b_row;
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < cute::size(thr_racc); i += PER) {
                uint32_t w[8];
                PartialT* v = reinterpret_cast<PartialT*>(w);
                CUTLASS_PRAGMA_UNROLL
                for (int j = 0; j < PER; ++j) v[j] = PartialT(thr_racc(i + j));
                asm volatile(
                    "st.global.L1::no_allocate.v8.f32 [%0], "
                    "{%1,%2,%3,%4,%5,%6,%7,%8};\n" ::"l"(p + i),
                    "r"(w[0]), "r"(w[1]), "r"(w[2]), "r"(w[3]), "r"(w[4]),
                    "r"(w[5]), "r"(w[6]), "r"(w[7]));
            }
        }
    }

    CUTE_DEVICE void end_item() {
        cutlass::arch::fence_view_async_tmem_load();
        res.bars.acc_empty[acc_slot].arrive();
        ++acc_seq;
    }
};

template <class Cfg, class TmemStorage>
CUTE_DEVICE auto make_drain_op(const GemmResources<Cfg::BN, TmemStorage>& res,
                               int base_acc) {
    using G = GemmGeom<Cfg>;
    using Traits = typename G::Traits;
    using Roles = KernelTrait::Roles;

    cute::Tensor tmem_acc = res.tmem.tmem_accumulators;
    cute::Tensor acc = tmem_acc(cute::make_coord(cute::_, cute::_), cute::_0{},
                                cute::_0{}, cute::_);
    cute::Tensor acc_tile =
        cute::flat_divide(acc, typename Traits::EpiComputeTile{});
    cute::TiledCopy tiled_t2r = cute::make_tmem_copy(
        typename Traits::CopyOpT2R{},
        acc_tile(cute::_, cute::_, cute::_0{}, cute::_0{}, cute::_0{}));
    static_assert(
        (int)cute::size(decltype(tiled_t2r){}) / 32 == Roles::DrainWarps,
        "the T2R copy needs one warp per TMEM datapath quadrant; DrainWarps "
        "must equal that count");

    const int warp = KernelTrait::warp_idx();
    const int lane = KernelTrait::lane_idx();

    cute::ThrCopy thr_t2r =
        tiled_t2r.get_slice((warp % Roles::DrainWarps) * 32 + lane);
    cute::Tensor thr_tacc = thr_t2r.partition_S(acc_tile);
    cute::Tensor thr_demo = thr_t2r.partition_D(
        acc_tile(cute::_, cute::_, cute::_0{}, cute::_0{}, cute::_0{}));
    cute::Tensor thr_racc = cute::make_tensor<float>(cute::shape(thr_demo));
    cute::Tensor coord =
        thr_t2r.partition_D(cute::make_identity_tensor(cute::shape(
            acc_tile(cute::_, cute::_, cute::_0{}, cute::_0{}, cute::_0{}))));

    return DrainOp<Cfg, TmemStorage, decltype(tiled_t2r), decltype(thr_tacc),
                   decltype(thr_racc), decltype(coord)>{
        res, tiled_t2r, thr_tacc, thr_racc, coord, base_acc, 0, 0};
}

template <class Cfg>
struct Bf16GemmTypes {
    static constexpr int BM = Cfg::BM;
    static constexpr int BN = Cfg::BN;
    static constexpr int BK = Cfg::BK;
    static constexpr int STAGES = Cfg::TmaStages;

    static_assert(BK % 64 == 0,
                  "SW128 at bf16 needs BK to be a multiple of 64");
    using SmemLayoutA1 =
        decltype(ku::make_umma_canonical_k_major_layout<BM, BK, 128, bf16>());
    using SmemLayoutB1 =
        decltype(ku::make_umma_canonical_k_major_layout<BN, BK, 128, bf16>());

    static constexpr int SMEM_A =
        cute::cosize_v<SmemLayoutA1> * (int)sizeof(bf16);
    static constexpr int SMEM_B =
        cute::cosize_v<SmemLayoutB1> * (int)sizeof(bf16);
    static constexpr int OFF_STAGES = KernelTraits<ModelDims, BN>::MaxTmaStages;
    static_assert(
        SMEM_A == GemmInst<BN, STAGES>::SMEM_A &&
            SMEM_B == GemmInst<BN, STAGES>::SMEM_B,
        "the bf16 kv_gate range and the fp8 ranges share one smem base, so "
        "their per-stage A/B bytes must match or slot s names different bytes "
        "in each.  They do today only because bf16 BK=128 x 2 B equals fp8 "
        "BK=256 x 1 B -- if either BK moves, this scheme breaks");
    static constexpr int SMEM_TOTAL = OFF_STAGES * (SMEM_A + SMEM_B);
    static constexpr int OFF_A = 0;
    static constexpr int OFF_B = OFF_STAGES * SMEM_A;

    using MmaAtom = cute::SM100_MMA_F16BF16_SS_NOELECT<
        bf16, bf16, float, BM, BN, cute::UMMA::Major::K, cute::UMMA::Major::K>;
    using TiledMma = decltype(cute::make_tiled_mma(MmaAtom{}));

    using TileA = cute::Shape<cute::Int<BM>, cute::Int<BK>>;
    using TileB = cute::Shape<cute::Int<BN>, cute::Int<BK>>;

    static constexpr uint32_t TMA_BYTES = (uint32_t)(SMEM_A + SMEM_B);

    using CopyOpT2R = cute::conditional_t<
        BN == 128, cute::SM100_TMEM_LOAD_32dp32b128x,
        cute::conditional_t<
            BN == 64, cute::SM100_TMEM_LOAD_32dp32b64x,
            cute::conditional_t<
                BN == 32, cute::SM100_TMEM_LOAD_32dp32b32x,
                cute::conditional_t<BN == 16, cute::SM100_TMEM_LOAD_32dp32b16x,
                                    cute::SM100_TMEM_LOAD_32dp32b8x>>>>;
    static_assert(
        BN >= 16 && BN <= 128 && (BN & (BN - 1)) == 0,
        "BN must be a power of two in 16..128 -- this GEMM's own UMMA "
        "needs N % 16 == 0 at M=128 (mla_intrinsics.cuh:707), which is "
        "what sets the floor for the whole profile.  BN < 64 also "
        "requires the single-token-group precondition MegaMlaImpl "
        "checks");
};

template <class Cfg>
CUTE_HOST auto make_bf16_gemm_tma_w(const void* weight, int m_total,
                                    int k_dim) {
    using Ty = Bf16GemmTypes<Cfg>;
    auto layout = cute::make_layout(cute::make_shape(m_total, k_dim),
                                    cute::make_stride(k_dim, cute::_1{}));
    return cute::make_tma_copy(
        cute::SM90_TMA_LOAD{},
        cute::make_tensor(cute::make_gmem_ptr(static_cast<const bf16*>(weight)),
                          layout),
        typename Ty::SmemLayoutA1{});
}

template <class Cfg>
CUTE_HOST auto make_bf16_gemm_tma_act(const void* act, int T, int row_stride,
                                      int k_dim) {
    using Ty = Bf16GemmTypes<Cfg>;
    auto layout = cute::make_layout(cute::make_shape(T, k_dim),
                                    cute::make_stride(row_stride, cute::_1{}));
    return cute::make_tma_copy(
        cute::SM90_TMA_LOAD{},
        cute::make_tensor(cute::make_gmem_ptr(static_cast<const bf16*>(act)),
                          layout),
        typename Ty::SmemLayoutB1{});
}

struct Bf16GemmAcc {
    static constexpr int Stages = TraitsN128::WoA::AccStages;
};
static_assert(
    Bf16GemmAcc::Stages <= GemmArena::MaxAccStages,
    "this range shares `bar_gemm_acc_{full,empty}`, which are sized from the "
    "profiles' max; a deeper depth would index past them (a hang, not a "
    "fault)");

template <class Cfg>
struct Bf16GemmBytes {
    using Ty = Bf16GemmTypes<Cfg>;
    static constexpr uint32_t W = (uint32_t)Ty::SMEM_A;
    static constexpr uint32_t ACT = (uint32_t)Ty::SMEM_B;
    static_assert(
        W + ACT == Ty::TMA_BYTES,
        "the split halves must account for exactly what the fused form did");
};

struct Bf16GemmItem {
    int m_tile;
    int n_tile;
    int n_valid;
};

template <class Cfg>
CUTE_DEVICE Bf16GemmItem bf16_gemm_item_at(int it, int n_groups,
                                           int num_tokens) {
    const int tg = it % n_groups;
    const int mt = it / n_groups;
    return Bf16GemmItem{mt, tg, min((int)Cfg::BN, num_tokens - tg * Cfg::BN)};
}

template <class Cfg, class TmaW, class TmaAct>
struct Bf16GemmResources {
    const TmaW& tma_w;
    const TmaAct& tma_act;
    char* __restrict__ gemm_smem;
    GemmPipelineBars bars;
    uint32_t tmem_base;
    int act_rows;
    int m_total;
    int k_dim;
    int k_tiles;
    float* __restrict__ ring;
    const int* __restrict__ ring_rows;
};

template <class Cfg, int St>
CUTE_DEVICE auto bf16_smem_a(char* p) {
    using Ty = Bf16GemmTypes<Cfg>;
    return cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<bf16*>(
                                 p + Ty::OFF_A + St * Ty::SMEM_A)),
                             typename Ty::SmemLayoutA1{});
}
template <class Cfg, int St>
CUTE_DEVICE auto bf16_smem_b(char* p) {
    using Ty = Bf16GemmTypes<Cfg>;
    return cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<bf16*>(
                                 p + Ty::OFF_B + St * Ty::SMEM_B)),
                             typename Ty::SmemLayoutB1{});
}

template <class Cfg, class Res>
struct Bf16TmaOpA {
    using Ty = Bf16GemmTypes<Cfg>;
    static constexpr int STAGES = Cfg::TmaStages;

    const Res& res;
    bool elected;
    uint32_t base_bits;
    uint32_t base_zero_bits;
    int seq;
    int m_tile;

    CUTE_DEVICE void begin_item(const Bf16GemmItem& it) { m_tile = it.m_tile; }
    CUTE_DEVICE void end_item() { ++seq; }

    CUTE_DEVICE void issue_tile(int t, int site, int item) {
        wait_stage_free<STAGES>(res.bars, base_bits, base_zero_bits, seq,
                                res.k_tiles, t, site, item);
        if (!elected) return;
        dispatch_stage<STAGES>(t % STAGES, issuer(), t);
    }

    CUTE_DEVICE auto issuer() {
        return [this](auto St, int k) {
            constexpr int s = decltype(St)::value;
            res.bars.tma_a[s].arrive_and_expect_tx(Bf16GemmBytes<Cfg>::W);
            uint64_t& mbar = reinterpret_cast<uint64_t&>(res.bars.tma_a[s]);
            auto mA = res.tma_w.get_tma_tensor(
                cute::make_shape(res.m_total, res.k_dim));
            auto thr_w = res.tma_w.get_slice(cute::_0{});
            auto tA = cute::local_tile(mA, typename Ty::TileA{},
                                       cute::make_coord(m_tile, k));
            cute::copy(res.tma_w.with(mbar, 0), thr_w.partition_S(tA),
                       thr_w.partition_D(bf16_smem_a<Cfg, s>(res.gemm_smem)));
        };
    }
};

template <class Cfg, class Res>
struct Bf16TmaOpB {
    using Ty = Bf16GemmTypes<Cfg>;
    static constexpr int STAGES = Cfg::TmaStages;

    const Res& res;
    bool elected;
    uint32_t base_bits;
    uint32_t base_zero_bits;
    int seq;
    int n_tile;

    CUTE_DEVICE void begin_item(const Bf16GemmItem& it) { n_tile = it.n_tile; }
    CUTE_DEVICE void end_item() { ++seq; }

    CUTE_DEVICE void issue_tile(int t, int site, int item) {
        wait_stage_free<STAGES>(res.bars, base_bits, base_zero_bits, seq,
                                res.k_tiles, t, site, item);
        if (!elected) return;
        dispatch_stage<STAGES>(t % STAGES, issuer(), t);
    }

    CUTE_DEVICE auto issuer() {
        return [this](auto St, int k) {
            constexpr int s = decltype(St)::value;
            res.bars.tma_b[s].arrive_and_expect_tx(Bf16GemmBytes<Cfg>::ACT);
            uint64_t& mbar = reinterpret_cast<uint64_t&>(res.bars.tma_b[s]);
            auto mB = res.tma_act.get_tma_tensor(
                cute::make_shape(res.act_rows, res.k_dim));
            auto thr_act = res.tma_act.get_slice(cute::_0{});
            auto tB = cute::local_tile(mB, typename Ty::TileB{},
                                       cute::make_coord(n_tile, k));
            cute::copy(res.tma_act.with(mbar, 0), thr_act.partition_S(tB),
                       thr_act.partition_D(bf16_smem_b<Cfg, s>(res.gemm_smem)));
        };
    }
};

template <class Cfg, class Res>
struct Bf16MmaOp {
    using Ty = Bf16GemmTypes<Cfg>;
    static constexpr int STAGES = Cfg::TmaStages;
    static constexpr int ACC_STAGES = Bf16GemmAcc::Stages;

    const Res& res;
    typename Ty::TiledMma tiled_mma;
    bool elected;
    uint32_t base_bits;

    int seq;
    int base_acc;
    int acc_seq;
    int acc_slot;

    CUTE_DEVICE void begin_item(int site, int item) {
        acc_slot = reserve_acc_slot<ACC_STAGES>(res.bars, base_acc, acc_seq,
                                                site, item);
    }

    CUTE_DEVICE void wait_tma(int k, int site_a, int site_b, int item) {
        wait_stage_tma<STAGES>(res.bars, base_bits, seq, res.k_tiles, k, site_a,
                               site_b, item);
    }

    CUTE_DEVICE void run_k_tile(int k) {
        ku::tcgen05_after_thread_sync();
        if (!elected) return;
        dispatch_stage<STAGES>(k % STAGES, stager(), k);
    }

    CUTE_DEVICE auto stager() {
        return [this](auto St, int k) {
            constexpr int s = decltype(St)::value;

            cute::Tensor tmem_frag = cute::partition_fragment_C(
                tiled_mma,
                cute::Shape<cute::Int<Cfg::BM>, cute::Int<Cfg::BN>>{});
            tmem_frag.data().get() =
                res.tmem_base + (uint32_t)(acc_slot * Cfg::BN);
            ku::utcmma_ss(tiled_mma, bf16_smem_a<Cfg, s>(res.gemm_smem),
                          bf16_smem_b<Cfg, s>(res.gemm_smem), tmem_frag,
                          /*clear_accum=*/k == 0);
        };
    }

    CUTE_DEVICE void commit(int k) {
        __syncwarp();
        if (elected) ku::umma_arrive_noelect(res.bars.smem_free[k % STAGES]);
    }

    CUTE_DEVICE void end_item() {
        if (elected) ku::umma_arrive_noelect(res.bars.acc_full[acc_slot]);
        ++seq;
        ++acc_seq;
    }
};

template <class Cfg, class Res, class TiledT2R, class ThrTacc, class Racc,
          class Coord>
struct Bf16DrainOp {
    static constexpr int ACC_STAGES = Bf16GemmAcc::Stages;

    const Res& res;
    TiledT2R tiled_t2r;

    ThrTacc thr_tacc;
    Racc thr_racc;
    Coord coord;
    int base_acc;
    int acc_seq;
    int acc_slot;

    CUTE_DEVICE void wait_mma_and_load(int site, int item) {
        acc_slot =
            claim_acc_slot<ACC_STAGES>(res.bars, base_acc, acc_seq, site, item);
        ku::tcgen05_after_thread_sync();
        ThrTacc src = thr_tacc;
        src.data() = src.data().get() + (uint32_t)(acc_slot * Cfg::BN);
        cute::copy(tiled_t2r, src, thr_racc);
        cutlass::arch::fence_view_async_tmem_load();
    }

    CUTE_DEVICE void store(const Bf16GemmItem& it) const {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < cute::size(thr_racc); ++i) {
            const int row = cute::get<0>(coord(i));
            const int act_row = cute::get<1>(coord(i));
            if (act_row < it.n_valid) {
                const int ring_row =
                    res.ring_rows[it.n_tile * Cfg::BN + act_row];
                res.ring[(int64_t)ring_row * res.m_total + it.m_tile * Cfg::BM +
                         row] = thr_racc(i);
            }
        }
    }

    CUTE_DEVICE void end_item() {
        cutlass::arch::fence_view_async_tmem_load();
        res.bars.acc_empty[acc_slot].arrive();
        ++acc_seq;
    }
};

template <class Cfg, class Res>
CUTE_DEVICE auto make_bf16_weight_tma_op(const Res& res, uint32_t base_bits,
                                         uint32_t base_zero_bits) {
    return Bf16TmaOpA<Cfg, Res>{
        res, cute::elect_one_sync() != 0, base_bits, base_zero_bits, 0, 0};
}

template <class Cfg, class Res>
CUTE_DEVICE auto make_bf16_act_tma_op(const Res& res, uint32_t base_bits,
                                      uint32_t base_zero_bits) {
    return Bf16TmaOpB<Cfg, Res>{
        res, cute::elect_one_sync() != 0, base_bits, base_zero_bits, 0, 0};
}

template <class Cfg, class Res>
CUTE_DEVICE auto make_bf16_mma_op(const Res& res, uint32_t base_bits,
                                  int base_acc) {
    return Bf16MmaOp<Cfg, Res>{res,
                               typename Bf16GemmTypes<Cfg>::TiledMma{},
                               cute::elect_one_sync() != 0,
                               base_bits,
                               0,
                               base_acc,
                               0,
                               0};
}

template <class Cfg, class Res>
CUTE_DEVICE auto make_bf16_drain_op(const Res& res, int base_acc) {
    using Ty = Bf16GemmTypes<Cfg>;
    using Roles = KernelTrait::Roles;
    static_assert(
        Roles::DrainWarps == 4 && Roles::DrainWarpBegin % 4 == 0,
        "the T2R copy needs one warp per TMEM quadrant, based at a multiple of "
        "4");

    typename Ty::TiledMma tiled_mma;

    cute::Tensor frag0 = cute::partition_fragment_C(
        tiled_mma, cute::Shape<cute::Int<Cfg::BM>, cute::Int<Cfg::BN>>{});
    frag0.data().get() = res.tmem_base;

    cute::Tensor acc0 =
        frag0(cute::make_coord(cute::_, cute::_), cute::_0{}, cute::_0{});
    cute::TiledCopy tiled_t2r =
        cute::make_tmem_copy(typename Ty::CopyOpT2R{}, acc0);
    static_assert(
        (int)cute::size(decltype(tiled_t2r){}) / 32 == Roles::DrainWarps,
        "the T2R copy must need exactly DrainWarps warps");

    const int warp = KernelTrait::warp_idx();
    const int lane = KernelTrait::lane_idx();
    cute::ThrCopy thr_t2r =
        tiled_t2r.get_slice((warp % Roles::DrainWarps) * 32 + lane);
    cute::Tensor thr_demo = thr_t2r.partition_D(acc0);
    cute::Tensor thr_racc = cute::make_tensor<float>(cute::shape(thr_demo));
    cute::Tensor coord =
        thr_t2r.partition_D(cute::make_identity_tensor(cute::shape(acc0)));

    using Op = Bf16DrainOp<Cfg, Res, decltype(tiled_t2r),
                           decltype(thr_t2r.partition_S(acc0)),
                           decltype(thr_racc), decltype(coord)>;

    return Op{res,      tiled_t2r, thr_t2r.partition_S(acc0),
              thr_racc, coord,     base_acc,
              0,        0};
}

struct GemmWorkList {
    int items_kv_gate;
    int items_wq_a;
    int items_qkv;
    int items_wq_b;
    int n_groups;
    int num_tokens;
};

struct RangeBarrierLedger {
    uint32_t parity = 0;
    uint32_t touched = 0;
    int acc_items = 0;

    template <int STAGES>
    CUTE_DEVICE void contribute(int k_tps, int n) {
        if (n == 0) return;
        CUTLASS_PRAGMA_UNROLL
        for (int s = 0; s < STAGES; ++s) {
            const int arrives = n * stage_uses<STAGES>(k_tps, s);
            parity ^= (uint32_t)(arrives & 1) << s;
            touched |= 1u << s;
        }
        acc_items += n;
    }

    CUTE_DEVICE uint32_t base_bits() const { return parity; }
    CUTE_DEVICE uint32_t base_zero_bits() const { return ~touched; }
    CUTE_DEVICE int base_acc() const { return acc_items; }
};

template <class Cfg>
CUTE_DEVICE GemmWorkItem gemm_item(const GemmWorkList& wl, int it, int n_pad,
                                   int m_total, int k_tps) {
    const int tg = it % wl.n_groups;
    const int rest = it / wl.n_groups;
    const int ks = rest % Cfg::KSplit;
    const int mt = rest / Cfg::KSplit;
    const int n_base = tg * n_pad;
    return GemmWorkItem{/*n_row=*/n_base,
                        /*m_tile=*/mt,
                        /*m_total=*/m_total,

                        /*k_tile_begin=*/ks * k_tps,
                        /*a_k_tile_begin=*/ks * k_tps,
                        /*out_row_stride=*/Cfg::KSplit * m_total,
                        /*n_valid=*/min(n_pad, wl.num_tokens - n_base)};
}

template <class Cfg>
CUTE_DEVICE typename Cfg::PartialT* gemm_out(
    typename Cfg::PartialT* __restrict__ base, const GemmWorkList& wl, int it,
    int n_pad, int m_total) {
    const int tg = it % wl.n_groups;
    const int rest = it / wl.n_groups;
    const int ks = rest % Cfg::KSplit;
    const int mt = rest / Cfg::KSplit;
    return base + (int64_t)(tg * n_pad) * (Cfg::KSplit * m_total) +
           ks * m_total + mt * GemmShapeDefaults::BM;
}

template <int BN>
struct QkvItem {
    GemmWorkItem wi;
    const WeightAtoms<BN>* weight;
    float* __restrict__ out;
};

struct C128Inputs {
    const float* __restrict__ ring;
    const int* __restrict__ ring_row;
    const int* __restrict__ positions;
    const float* __restrict__ ape;
    const __nv_bfloat16* __restrict__ norm_weight;
    const float* __restrict__ freqs_real;
    const int* __restrict__ out_loc;
    float* __restrict__ row_out;
    uint8_t* __restrict__ pool;
    int stride_kv_block;
    int page_block;
    int page_shift;
    float eps;
    float* __restrict__ stats;
    __nv_bfloat16* __restrict__ row;
};

struct KvWriteInputs {
    const float* __restrict__ ws_kv_part;
    const __nv_bfloat16* __restrict__ norm_weight;
    const float* __restrict__ freqs_real;
    const int* __restrict__ positions;
    const int* __restrict__ loc;
    uint8_t* __restrict__ pool;
    int stride_kv_block;
    int page_block_size;
    float eps;
    float* __restrict__ sq_buf;
    __nv_bfloat16* __restrict__ row;
};

template <class CfgQkv, class CfgWqB, class CfgKvGate, class TmemStorage,
          class ResKvGate>
struct MlaGemmRanges {
    using CfgQkvT = CfgQkv;
    using CfgWqBT = CfgWqB;
    using CfgKvGateT = CfgKvGate;
    using TmemStorageT = TmemStorage;
    using ResKvGateT = ResKvGate;

    static constexpr int BN = CfgQkv::BN;
    static constexpr int N_PAD = GemmGeom<CfgQkv>::TileT;
    static constexpr int KS_QKV = CfgQkv::KSplit;

    static_assert(CfgQkv::BN == CfgWqB::BN,
                  "both ranges drain through the same TMEM accumulator tile");
    static_assert(
        N_PAD == GemmGeom<CfgWqB>::TileT,
        "both ranges pack tokens at the same N granularity, which is what lets "
        "one token-group index address all of their activations");
    static_assert(
        CfgKvGate::BN == BN,
        "the kv_gate range packs tokens at the same N granularity as the "
        "others, "
        "which is what makes one token-group index address all three");

    static_assert(
        CfgKvGate::TmaStages <= GemmArena::MaxTmaStages,
        "the kv_gate range reuses the shared per-stage barrier arrays");

    const GemmResources<BN, TmemStorage>& res_qkv;
    const WeightAtoms<BN>* atoms_wkv;
    const GemmResources<BN, TmemStorage>& res_wqb;

    const ResKvGate& res_kv_gate;
    float* __restrict__ out_qlora_part;
    float* __restrict__ out_kv_part;
    typename CfgWqB::PartialT* __restrict__ out_flat_q;

    template <class Sched>
    CUTE_DEVICE QkvItem<BN> qkv_item(const Sched& s, int item) const {
        const bool is_q = s.qkv_is_q(item);
        return QkvItem<BN>{
            s.qkv_item(item), is_q ? &res_qkv.weight : atoms_wkv,
            (is_q ? out_qlora_part : out_kv_part) + s.qkv_out_off(item)};
    }

    template <class Sched>
    CUTE_DEVICE typename CfgWqB::PartialT* wqb_out(const Sched& s,
                                                   int item) const {
        return gemm_out<CfgWqB>(out_flat_q, s.gemm, item, N_PAD,
                                s.m_total_wq_b);
    }
};

template <class CfgQkv, class CfgWqB, class CfgKvGate, class TmemStorage,
          class ResKvGate>
CUTE_DEVICE auto make_mla_gemm_ranges(
    const GemmResources<CfgQkv::BN, TmemStorage>& res_qkv,
    const WeightAtoms<CfgQkv::BN>* atoms_wkv,
    const GemmResources<CfgWqB::BN, TmemStorage>& res_wqb,
    const ResKvGate& res_kv_gate, float* __restrict__ out_qlora_part,
    float* __restrict__ out_kv_part,
    typename CfgWqB::PartialT* __restrict__ out_flat_q) {
    return MlaGemmRanges<CfgQkv, CfgWqB, CfgKvGate, TmemStorage, ResKvGate>{
        .res_qkv = res_qkv,
        .atoms_wkv = atoms_wkv,
        .res_wqb = res_wqb,
        .res_kv_gate = res_kv_gate,
        .out_qlora_part = out_qlora_part,
        .out_kv_part = out_kv_part,
        .out_flat_q = out_flat_q};
}

struct OutProjWorkList {
    int items_wo_a;
    int items_wo_b;
    int n_groups;
    int num_tokens;
};

inline int olora_rounds_for(int o_groups) {
    return o_groups / TraitsN128::WoA::OloraGroupsPerPass;
}

struct OutProjDims {
    int m_total_woa;
    int m_tiles_all;
    int m_tiles_per_group;
    int items_per_tg;
    int olora_rounds;
    int olora_items_per_slot;
    int m_total_wob;
    int m_tiles_all_wob;
    int wob_units;
};

CUTE_HOST OutProjDims make_out_proj_dims(int g, int r, int hidden, int ks_woa) {
    TORCH_CHECK(
        g > 0 && r > 0 && hidden > 0,
        "the output projection needs positive G/R/ModelDims::HIDDEN, got ", g,
        "/", r, "/", hidden);
    TORCH_CHECK(r % GemmShapeDefaults::BM == 0,
                "the G groups must be equal runs of whole m-tiles in the "
                "stacked wo_a weight; R=",
                r, " is not a multiple of ", GemmShapeDefaults::BM);
    TORCH_CHECK(
        r % GemmShapeDefaults::SfGran == 0,
        "an output group must be a whole number of scale groups, or the "
        "quantizer would straddle the edge it just waited on; R=",
        r);
    TORCH_CHECK(g % TraitsN128::WoA::OloraGroupsPerPass == 0,
                "the reduce rounds must partition the G output groups evenly, "
                "or the last round waits on a key no drain arrives on -- a "
                "hang, not a wrong number; G=",
                g, " GPP=", TraitsN128::WoA::OloraGroupsPerPass);
    constexpr int GRP_PER_PASS = KernelTrait::Roles::VecWarps * 32 / 4;
    TORCH_CHECK(r == GRP_PER_PASS * GemmShapeDefaults::SfGran,
                "one vec sub-pass must cover exactly one output group, or the "
                "round no longer spans a whole number of groups and the wait "
                "guards the wrong columns; R must be ",
                GRP_PER_PASS * GemmShapeDefaults::SfGran, ", got ", r);
    constexpr int WOB_UNIT = 8 * KernelTrait::Roles::VecWarps * 32;
    TORCH_CHECK(hidden % WOB_UNIT == 0,
                "ModelDims::HIDDEN must split into whole 128-lane uint4 units, "
                "or a unit "
                "straddles two tokens and `t` -- hence the wait key -- stops "
                "being warpgroup-uniform; ModelDims::HIDDEN=",
                hidden, " unit=", WOB_UNIT);
    OutProjDims d{};
    d.m_total_woa = g * r;
    d.m_tiles_all = d.m_total_woa / GemmShapeDefaults::BM;
    d.m_tiles_per_group = r / GemmShapeDefaults::BM;
    d.items_per_tg = d.m_tiles_all * ks_woa;
    d.olora_rounds = g / TraitsN128::WoA::OloraGroupsPerPass;
    d.olora_items_per_slot =
        d.m_tiles_per_group * ks_woa * TraitsN128::WoA::OloraGroupsPerPass;
    d.m_total_wob = hidden;
    d.m_tiles_all_wob = hidden / GemmShapeDefaults::BM;
    d.wob_units = hidden / WOB_UNIT;
    return d;
}

template <class CfgWoA_, class CfgWoB_, class TmemStorage>
struct MlaOutProjRanges {
    using CfgWoAT = CfgWoA_;
    using CfgWoBT = CfgWoB_;
    using TmemStorageT = TmemStorage;
    using GWoA = GemmGeom<CfgWoA_>;
    using GWoB = GemmGeom<CfgWoB_>;

    static constexpr int BN = CfgWoA_::BN;
    static constexpr int N_PAD = GWoA::TileT;
    static constexpr int KS_WOA = CfgWoA_::KSplit;

    template <int KS>
    using CfgWoBAt = WithKSplit<CfgWoB_, KS>;

    static constexpr int OLORA_GPP = TraitsN128::WoA::OloraGroupsPerPass;

    static_assert(CfgWoB_::BN == BN && GWoB::TileT == N_PAD,
                  "both output-projection ranges pack tokens at the same N "
                  "granularity, which is what lets one token-group index "
                  "address both of their activations");

    static_assert(CfgWoA_::AccStages == CfgWoB_::AccStages,
                  "the two ranges hand over `base_acc` as an item count; a "
                  "different slot count makes that hand-over lie");

    static_assert(CfgWoA_::TmaStages <= GemmArena::MaxTmaStages &&
                      CfgWoB_::TmaStages <= GemmArena::MaxTmaStages,
                  "these ranges reuse the shared per-stage barrier arrays");

    const GemmResources<BN, TmemStorage>& res_wo_a;
    const GemmResources<BN, TmemStorage>& res_wo_b;
    typename CfgWoA_::PartialT* __restrict__ out_olora;
    typename CfgWoB_::PartialT* __restrict__ out_wo_b;

    bf16* __restrict__ final_out;

    template <class Sched>
    CUTE_DEVICE typename CfgWoA_::PartialT* wo_a_out(const Sched& s,
                                                     int item) const {
        return out_olora + s.wo_a_out_off(item);
    }

    template <int KS, class Sched>
    CUTE_DEVICE typename CfgWoB_::PartialT* wo_b_out(const Sched& s,
                                                     int item) const {
        return out_wo_b + s.template wo_b_out_off<KS>(item);
    }
};

template <class CfgWoA, class CfgWoB, class TmemStorage>
CUTE_DEVICE auto make_out_proj_ranges(
    const GemmResources<CfgWoA::BN, TmemStorage>& res_wo_a,
    const GemmResources<CfgWoB::BN, TmemStorage>& res_wo_b,
    typename CfgWoA::PartialT* __restrict__ out_olora,
    typename CfgWoB::PartialT* __restrict__ out_wo_b,
    bf16* __restrict__ final_out) {
    return MlaOutProjRanges<CfgWoA, CfgWoB, TmemStorage>{
        .res_wo_a = res_wo_a,
        .res_wo_b = res_wo_b,
        .out_olora = out_olora,
        .out_wo_b = out_wo_b,
        .final_out = final_out};
}

template <int KS>
CUTE_DEVICE void olora_sum_partials(const uint4 (&raw)[KS],
                                    __nv_bfloat16 (&q)[8]) {
    if constexpr (KS == 2) {
        const auto* a2 = reinterpret_cast<const __nv_bfloat162*>(&raw[0]);
        const auto* b2 = reinterpret_cast<const __nv_bfloat162*>(&raw[1]);
        auto* q2 = reinterpret_cast<__nv_bfloat162*>(q);
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < 4; ++i) q2[i] = __hadd2(a2[i], b2[i]);
    } else {
        float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        CUTLASS_PRAGMA_UNROLL
        for (int s = 0; s < KS; ++s) {
            const auto* pv = reinterpret_cast<const __nv_bfloat16*>(&raw[s]);
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < 8; ++i) acc[i] += __bfloat162float(pv[i]);
        }
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < 8; ++i) q[i] = __float2bfloat16(acc[i]);
    }
}

}  // namespace mega_mla
