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

#include "kernel/mla/mega_mla/gemm/geom.cuh"

namespace halops::mla {
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
                cute::group_modes<0, CommonGemmModes::StageMode>(
                    sA(cute::_, cute::_, cute::_, St)),
                cute::group_modes<0, CommonGemmModes::StageMode>(
                    cta_mma.partition_A(gA)));
            auto [tma_gSFA, tma_sSFA] = cute::tma_partition(
                at->tma_sfa, cute::Int<0>{}, cute::make_layout(cute::Int<1>{}),
                cute::group_modes<0, CommonGemmModes::StageMode>(
                    sSFA(cute::_, cute::_, cute::_, St)),
                cute::group_modes<0, CommonGemmModes::StageMode>(
                    cta_mma.partition_A(gSFA)));
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
                cute::group_modes<0, CommonGemmModes::StageMode>(
                    sB(cute::_, cute::_, cute::_, St)),
                cute::group_modes<0, CommonGemmModes::StageMode>(
                    cta_mma.partition_B(gB)));
            auto [tma_gSFB, tma_sSFB] = cute::tma_partition(
                at->tma_sfb, cute::Int<0>{}, cute::make_layout(cute::Int<1>{}),
                cute::group_modes<0, CommonGemmModes::StageMode>(
                    sSFB(cute::_, cute::_, cute::_, St)),
                cute::group_modes<0, CommonGemmModes::StageMode>(
                    cta_mma_sfb.partition_B(gSFB)));
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
}  // namespace halops::mla
