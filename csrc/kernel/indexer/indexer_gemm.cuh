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

#include "common/ptx.cuh"
#include "indexer_config.cuh"
#include "indexer_utils.cuh"

namespace halops::indexer {

template <class GemmConfig>
struct GemmGeometry {
    using Instance = GemmInstanceFor<GemmConfig>;
    using Traits = Instance;
    static_assert(GemmConfig::BM == GemmShapeDefaults::BM &&
                  GemmConfig::BK == GemmShapeDefaults::BK);
    using TileShape = typename Traits::TileShape;
    using UtccpOp = cute::SM100_UTCCP_4x32dp128bit_1cta;

    static constexpr int STAGES = GemmConfig::TmaStages;

    static constexpr int ACC_STAGES = GemmConfig::AccStages;
    static constexpr int TokenTile = Instance::TokenTile;
    static constexpr bool SEPARATE_SCALE_TILE_PER_GROUP =
        Instance::SEPARATE_SCALE_TILE_PER_GROUP;
    static constexpr int TOKEN_TILES_PER_SCALE_TILE =
        Instance::TOKEN_TILES_PER_SCALE_TILE;
    static constexpr int SCALE_B_COLUMNS_PER_TOKEN_TILE =
        Instance::SCALE_B_COLUMNS_PER_TOKEN_TILE;
    static_assert(STAGES > 0,
                  "The shared-memory pipeline needs at least one stage");
};

template <int BN, class TmemStorage>
struct GemmResources {
    const WeightDescriptors<BN>& weight;
    const ActivationDescriptors<BN>& activations;
    char* __restrict__ gemm_shared_memory;
    TmemStorage& tmem;

    GemmPipelineBarriers barriers;
};

template <class GemmConfig, class TmemStorage>
CUTE_DEVICE const GemmOperandADescriptors<GemmConfig::BN>& a_side(
    const GemmResources<GemmConfig::BN, TmemStorage>& resources) {
    if constexpr (GemmConfig::SwapAB)
        return resources.weight;
    else
        return resources.activations;
}

template <class GemmConfig, class TmemStorage>
CUTE_DEVICE const GemmOperandBDescriptors<GemmConfig::BN>& b_side(
    const GemmResources<GemmConfig::BN, TmemStorage>& resources) {
    if constexpr (GemmConfig::SwapAB)
        return resources.activations;
    else
        return resources.weight;
}

struct NoGate {
    CUTE_DEVICE void operator()(int) const {}
};

template <class GemmConfig, class Gate>
CUTE_DEVICE auto gate_on_a(const Gate& g) {
    // The readiness gate follows the activation operand when A/B are swapped.
    if constexpr (GemmConfig::SwapAB)
        return NoGate{};
    else
        return g;
}

template <class GemmConfig, class Gate>
CUTE_DEVICE auto gate_on_b(const Gate& g) {
    if constexpr (GemmConfig::SwapAB)
        return g;
    else
        return NoGate{};
}

struct GemmWorkItem {
    int n_row;
    int weight_tile;
    int m_total;

    int k_tile_begin;

    int a_k_tile_begin;
    int out_row_stride;
    int valid_tokens;
};

template <class GemmConfig>
struct GemmParts {
    using Instance = GemmInstanceFor<GemmConfig>;

    static CUTE_DEVICE auto smem_a(char* p) {
        return cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<e4m3*>(
                                     p + Instance::OPERAND_A_OFFSET)),
                                 typename Instance::SmemLayoutAStaged{});
    }
    static CUTE_DEVICE auto smem_b(char* p) {
        return cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<e4m3*>(
                                     p + Instance::OPERAND_B_OFFSET)),
                                 typename Instance::SmemLayoutBStaged{});
    }
    static CUTE_DEVICE auto smem_sfa(char* p) {
        return cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<ue8m0*>(
                                     p + Instance::SCALE_A_OFFSET)),
                                 typename Instance::SmemLayoutSFAStaged{});
    }
    static CUTE_DEVICE auto smem_sfb(char* p) {
        return cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<ue8m0*>(
                                     p + Instance::SCALE_B_OFFSET)),
                                 typename Instance::SmemLayoutSFBStaged{});
    }
};

template <class GemmConfig>
CUTE_DEVICE auto tile_gA(const GemmOperandADescriptors<GemmConfig::BN>& at,
                         int m_total) {
    cute::Tensor mA = at.tma_a.get_tma_tensor(cute::make_shape(
        m_total, at.k_dim, 1))(cute::_, cute::_, cute::Int<0>{});
    return cute::local_tile(mA, typename GemmGeometry<GemmConfig>::TileShape{},
                            cute::make_coord(cute::_, cute::_, cute::_),
                            cute::Step<cute::_1, cute::X, cute::_1>{});
}
template <class GemmConfig>
CUTE_DEVICE auto tile_gSFA(const GemmOperandADescriptors<GemmConfig::BN>& at) {
    cute::Tensor mSFA = at.tma_sfa.get_tma_tensor(at.layout_sfa.shape())(
        cute::_, cute::_, cute::Int<0>{});
    return cute::local_tile(mSFA,
                            typename GemmGeometry<GemmConfig>::TileShape{},
                            cute::make_coord(cute::_, cute::_, cute::_),
                            cute::Step<cute::_1, cute::X, cute::_1>{});
}

template <class GemmConfig, class TmemStorage, class SA, class SSFA,
          class CtaMma, class GA, class GSFA, class Gate>
struct TmaOpA {
    using Geometry = GemmGeometry<GemmConfig>;
    using Instance = typename Geometry::Instance;
    static constexpr int STAGES = Geometry::STAGES;

    const GemmResources<GemmConfig::BN, TmemStorage>& resources;
    SA sA;
    SSFA sSFA;
    CtaMma cta_mma;
    bool elected;

    uint32_t previous_stage_parities;
    uint32_t unused_stage_mask;
    int item_sequence;
    int tiles_per_split;

    const GemmOperandADescriptors<GemmConfig::BN>* at;
    GA gA;
    GSFA gSFA;
    int weight_tile;
    int a_k_begin;
    Gate gate;

    CUTE_DEVICE void begin_item(const WeightDescriptors<GemmConfig::BN>& atoms,
                                const GemmWorkItem& work_item) {
        gate(work_item.n_row / Geometry::TokenTile);
        if constexpr (GemmConfig::SwapAB) {
            at = &atoms;
            gA = tile_gA<GemmConfig>(atoms, work_item.m_total);
            gSFA = tile_gSFA<GemmConfig>(atoms);
            weight_tile = work_item.weight_tile;
            a_k_begin = work_item.a_k_tile_begin;
        } else {
            weight_tile = work_item.n_row / Geometry::TokenTile;
            a_k_begin = work_item.k_tile_begin;
        }
    }

    CUTE_DEVICE void issue_tile(int t) {
        wait_stage_free<STAGES>(resources.barriers, previous_stage_parities,
                                unused_stage_mask, item_sequence,
                                tiles_per_split, t);
        if (elected) dispatch_stage<STAGES>(t % STAGES, issuer(), t);
    }

    CUTE_DEVICE void end_item() { ++item_sequence; }

    CUTE_DEVICE auto issuer() {
        return [this](auto St, int k) {
            const int ka = k + a_k_begin;
            constexpr int s = decltype(St)::value;
            resources.barriers.tma_a[s].arrive_and_expect_tx(
                Instance::TMA_BYTES_A);
            uint64_t& mbar =
                reinterpret_cast<uint64_t&>(resources.barriers.tma_a[s]);
            auto [tma_gA, tma_sA] = cute::tma_partition(
                at->tma_a, cute::Int<0>{}, cute::make_layout(cute::Int<1>{}),
                cute::group_modes<0, 3>(sA(cute::_, cute::_, cute::_, St)),
                cute::group_modes<0, 3>(cta_mma.partition_A(gA)));
            auto [tma_gSFA, tma_sSFA] = cute::tma_partition(
                at->tma_sfa, cute::Int<0>{}, cute::make_layout(cute::Int<1>{}),
                cute::group_modes<0, 3>(sSFA(cute::_, cute::_, cute::_, St)),
                cute::group_modes<0, 3>(cta_mma.partition_A(gSFA)));
            cute::copy(at->tma_a.with(mbar, 0),
                       tma_gA(cute::_, weight_tile, ka), tma_sA);
            cute::copy(at->tma_sfa.with(mbar, 0),
                       tma_gSFA(cute::_, weight_tile, ka), tma_sSFA);
        };
    }
};

template <class GemmConfig, class TmemStorage, class Gate>
CUTE_DEVICE auto make_tma_op_a(
    const GemmResources<GemmConfig::BN, TmemStorage>& resources,
    uint32_t previous_stage_parities, uint32_t unused_stage_mask,
    int tiles_per_split, const Gate& act_gate) {
    using Parameters = GemmParts<GemmConfig>;
    typename GemmGeometry<GemmConfig>::Traits::TiledMma tiled_mma;
    auto sA = Parameters::smem_a(resources.gemm_shared_memory);
    auto sSFA = Parameters::smem_sfa(resources.gemm_shared_memory);
    auto cta_mma = tiled_mma.get_slice(0);
    auto gA = tile_gA<GemmConfig>(a_side<GemmConfig>(resources),
                                  a_side<GemmConfig>(resources).m_total);
    auto gSFA = tile_gSFA<GemmConfig>(a_side<GemmConfig>(resources));
    auto gate = gate_on_a<GemmConfig>(act_gate);
    return TmaOpA<GemmConfig, TmemStorage, decltype(sA), decltype(sSFA),
                  decltype(cta_mma), decltype(gA), decltype(gSFA),
                  decltype(gate)>{resources,
                                  sA,
                                  sSFA,
                                  cta_mma,
                                  cute::elect_one_sync() != 0,
                                  previous_stage_parities,
                                  unused_stage_mask,
                                  0,
                                  tiles_per_split,
                                  &a_side<GemmConfig>(resources),
                                  gA,
                                  gSFA,
                                  0,
                                  0,
                                  gate};
}

template <class GemmConfig>
CUTE_DEVICE auto tile_gB(const GemmOperandBDescriptors<GemmConfig::BN>& at) {
    const int k_dim = at.k_dim;
    cute::Tensor mB = at.tma_b.get_tma_tensor(
        cute::make_shape(at.n_pad, k_dim, 1))(cute::_, cute::_, cute::Int<0>{});
    return cute::local_tile(mB, typename GemmGeometry<GemmConfig>::TileShape{},
                            cute::make_coord(cute::_, cute::_, cute::_),
                            cute::Step<cute::X, cute::_1, cute::_1>{});
}

template <class GemmConfig>
CUTE_DEVICE auto tile_gSFB(const GemmOperandBDescriptors<GemmConfig::BN>& at) {
    cute::Tensor mSFBg = at.tma_sfb.get_tma_tensor(at.layout_sfb.shape())(
        cute::_, cute::_, cute::Int<0>{});
    return cute::local_tile(
        mSFBg, typename GemmGeometry<GemmConfig>::Traits::TileShape_SF{},
        cute::make_coord(cute::_, cute::_, cute::_),
        cute::Step<cute::X, cute::_1, cute::_1>{});
}

template <class GemmConfig, class TmemStorage, class SB, class SSFB,
          class CtaMma, class CtaMmaSf, class GB, class GSFB, class Gate>
struct TmaOpB {
    using Geometry = GemmGeometry<GemmConfig>;
    using Instance = typename Geometry::Instance;
    static constexpr int STAGES = Geometry::STAGES;

    const GemmResources<GemmConfig::BN, TmemStorage>& resources;
    SB sB;
    SSFB sSFB;
    CtaMma cta_mma;
    CtaMmaSf cta_mma_sfb;
    GB gB;
    GSFB gSFB;
    bool elected;

    uint32_t previous_stage_parities;
    uint32_t unused_stage_mask;
    int item_sequence;
    int tiles_per_split;

    const GemmOperandBDescriptors<GemmConfig::BN>* at;
    int n_tile;
    int scale_tile_index;
    int first_k_tile;
    Gate gate;

    CUTE_DEVICE void begin_item(
        const WeightDescriptors<GemmConfig::BN>& weight_descriptors,
        const GemmWorkItem& work_item) {
        gate(work_item.n_row / Geometry::TokenTile);
        if constexpr (GemmConfig::SwapAB) {
            n_tile = work_item.n_row / Geometry::TokenTile;
            first_k_tile = work_item.k_tile_begin;
        } else {
            at = &weight_descriptors;
            gB = tile_gB<GemmConfig>(weight_descriptors);
            gSFB = tile_gSFB<GemmConfig>(weight_descriptors);
            n_tile = work_item.weight_tile;
            first_k_tile = work_item.a_k_tile_begin;
        }
        if constexpr (Geometry::SEPARATE_SCALE_TILE_PER_GROUP)
            scale_tile_index = n_tile;
        else
            scale_tile_index = n_tile / Geometry::TOKEN_TILES_PER_SCALE_TILE;
    }

    CUTE_DEVICE void issue_tile(int t) {
        wait_stage_free<STAGES>(resources.barriers, previous_stage_parities,
                                unused_stage_mask, item_sequence,
                                tiles_per_split, t);
        if (elected) dispatch_stage<STAGES>(t % STAGES, issuer(), t);
    }

    CUTE_DEVICE void end_item() { ++item_sequence; }

    CUTE_DEVICE auto issuer() {
        return [this](auto St, int k) {
            const int kb = k + first_k_tile;
            constexpr int s = decltype(St)::value;
            resources.barriers.tma_b[s].arrive_and_expect_tx(
                Instance::TMA_BYTES_B);
            uint64_t& mbar =
                reinterpret_cast<uint64_t&>(resources.barriers.tma_b[s]);
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
            cute::copy(at->tma_sfb.with(mbar, 0),
                       tma_gSFB(cute::_, scale_tile_index, kb), tma_sSFB);
        };
    }
};

template <class GemmConfig, class TmemStorage, class Gate>
CUTE_DEVICE auto make_tma_op_b(
    const GemmResources<GemmConfig::BN, TmemStorage>& resources,
    uint32_t previous_stage_parities, uint32_t unused_stage_mask,
    int tiles_per_split, const Gate& act_gate) {
    using Parameters = GemmParts<GemmConfig>;
    using Traits = typename GemmGeometry<GemmConfig>::Traits;
    typename Traits::TiledMma tiled_mma;
    typename Traits::TiledMma_SF tiled_mma_sf;
    auto sB = Parameters::smem_b(resources.gemm_shared_memory);
    auto sSFB = Parameters::smem_sfb(resources.gemm_shared_memory);
    auto cta_mma = tiled_mma.get_slice(0);
    auto cta_mma_sfb = tiled_mma_sf.get_slice(0);
    auto gB = tile_gB<GemmConfig>(b_side<GemmConfig>(resources));
    auto gSFB = tile_gSFB<GemmConfig>(b_side<GemmConfig>(resources));
    auto gate = gate_on_b<GemmConfig>(act_gate);
    return TmaOpB<GemmConfig, TmemStorage, decltype(sB), decltype(sSFB),
                  decltype(cta_mma), decltype(cta_mma_sfb), decltype(gB),
                  decltype(gSFB), decltype(gate)>{
        resources,
        sB,
        sSFB,
        cta_mma,
        cta_mma_sfb,
        gB,
        gSFB,
        cute::elect_one_sync() != 0,
        previous_stage_parities,
        unused_stage_mask,
        0,
        tiles_per_split,
        &b_side<GemmConfig>(resources),
        0,
        0,
        0,
        gate};
}

template <class GemmConfig, class TmemStorage, class TiledMmaT, class RA,
          class RB, class CopySFA, class CopySFB, class SrcSFA, class DstSFA,
          class SrcSFB, class DstSFB, class TmemSFA, class TmemSFB,
          class TmemAcc>
struct MmaOp {
    using Geometry = GemmGeometry<GemmConfig>;
    static constexpr int STAGES = Geometry::STAGES;

    static constexpr int ACC_STAGES = Geometry::ACC_STAGES;

    const GemmResources<GemmConfig::BN, TmemStorage>& resources;
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

    uint32_t previous_stage_parities;
    int item_sequence;
    int tiles_per_split;

    int previous_accumulators;
    int accumulator_sequence;

    TmemSFB tmem_SFB_mma;
    int accumulator_slot;

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

    CUTE_DEVICE void begin_item(const GemmWorkItem& work_item) {
        tmem_SFB_mma = tmem_SFB;
        if constexpr (!Geometry::SEPARATE_SCALE_TILE_PER_GROUP &&
                      Geometry::TOKEN_TILES_PER_SCALE_TILE > 1) {
            const int n_tile = work_item.n_row / Geometry::TokenTile;
            tmem_SFB_mma.data() =
                tmem_SFB_mma.data().get() +
                (n_tile % Geometry::TOKEN_TILES_PER_SCALE_TILE) *
                    Geometry::SCALE_B_COLUMNS_PER_TOKEN_TILE;
        }
        accumulator_slot = reserve_accumulator_slot<ACC_STAGES>(
            resources.barriers, previous_accumulators, accumulator_sequence);
    }

    CUTE_DEVICE void wait_tma(int k) {
        wait_stage_tma<STAGES>(resources.barriers, previous_stage_parities,
                               item_sequence, tiles_per_split, k);
    }

    CUTE_DEVICE void run_k_tile(int k) {
        if (k == 0) tiled_mma.accumulate_ = cute::UMMA::ScaleOut::Zero;
        halops::tcgen05_after_thread_sync();
        dispatch_stage<STAGES>(k % STAGES, stager(), k);
    }

    CUTE_DEVICE void commit(int k) {
        tiled_mma.accumulate_ = cute::UMMA::ScaleOut::One;
        __syncwarp();
        cutlass::arch::umma_arrive(reinterpret_cast<uint64_t const*>(
            &resources.barriers.buffer_free[k % STAGES]));
    }

    CUTE_DEVICE void end_item() {
        cutlass::arch::umma_arrive(reinterpret_cast<uint64_t const*>(
            &resources.barriers.accumulator_ready[accumulator_slot]));
        ++item_sequence;
        ++accumulator_sequence;
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
                cute::gemm(
                    tiled_mma.with(tiled_mma.accumulate_,
                                   tmem_SFA(cute::_, cute::_, kb),
                                   tmem_SFB_mma(cute::_, cute::_, kb)),
                    tile_rA(cute::_, cute::_, kb, St),
                    tile_rB(cute::_, cute::_, kb, St),
                    tmem_acc(cute::_, cute::_, cute::_, accumulator_slot));
                tiled_mma.accumulate_ = cute::UMMA::ScaleOut::One;
            }
        };
    }
};

template <class GemmConfig, class TmemStorage>
CUTE_DEVICE auto make_mma_op(
    const GemmResources<GemmConfig::BN, TmemStorage>& resources,
    uint32_t previous_stage_parities, int previous_accumulators,
    int tiles_per_split) {
    using Parameters = GemmParts<GemmConfig>;
    using Geometry = GemmGeometry<GemmConfig>;
    using Traits = typename Geometry::Traits;
    auto sA = Parameters::smem_a(resources.gemm_shared_memory);
    auto sB = Parameters::smem_b(resources.gemm_shared_memory);
    auto sSFA = Parameters::smem_sfa(resources.gemm_shared_memory);
    auto sSFB = Parameters::smem_sfb(resources.gemm_shared_memory);

    typename Traits::TiledMma tiled_mma;
    auto cta_mma = tiled_mma.get_slice(0);

    auto tile_rA = cta_mma.make_fragment_A(sA);
    auto tile_rB = cta_mma.make_fragment_B(sB);

    auto tmem_SFA = resources.tmem.tmem_SFA;
    auto tmem_SFB = resources.tmem.tmem_SFB;
    auto sm_sfa =
        cute::make_tensor(sSFA.data(), cute::filter_zeros(sSFA.layout()));
    auto sm_sfb =
        cute::make_tensor(sSFB.data(), cute::filter_zeros(sSFB.layout()));
    auto tm_sfa = cute::make_tensor(tmem_SFA.data(),
                                    cute::filter_zeros(tmem_SFA.layout()));
    auto tm_sfb = cute::make_tensor(tmem_SFB.data(),
                                    cute::filter_zeros(tmem_SFB.layout()));

    auto copy_sfa = cute::make_utccp_copy(typename Geometry::UtccpOp{}, tm_sfa);
    auto copy_sfb = cute::make_utccp_copy(typename Geometry::UtccpOp{}, tm_sfb);
    auto thr_sfa = copy_sfa.get_slice(0);
    auto thr_sfb = copy_sfb.get_slice(0);
    auto src_sfa = cute::get_utccp_smem_desc_tensor<typename Geometry::UtccpOp>(
        thr_sfa.partition_S(sm_sfa));
    auto dst_sfa = thr_sfa.partition_D(tm_sfa);
    auto src_sfb = cute::get_utccp_smem_desc_tensor<typename Geometry::UtccpOp>(
        thr_sfb.partition_S(sm_sfb));
    auto dst_sfb = thr_sfb.partition_D(tm_sfb);

    auto tmem_acc = resources.tmem.tmem_accumulators;
    using Op = MmaOp<GemmConfig, TmemStorage, decltype(tiled_mma),
                     decltype(tile_rA), decltype(tile_rB), decltype(copy_sfa),
                     decltype(copy_sfb), decltype(src_sfa), decltype(dst_sfa),
                     decltype(src_sfb), decltype(dst_sfb), decltype(tmem_SFA),
                     decltype(tmem_SFB), decltype(tmem_acc)>;
    Op::assert_stage_modes(tile_rA, src_sfa);

    return Op{resources,
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
              cute::elect_one_sync() != 0,
              previous_stage_parities,
              0,
              tiles_per_split,
              previous_accumulators,
              0,
              tmem_SFB,
              0};
}

template <class GemmConfig, class TmemStorage, class TmemCopy,
          class ThreadTmemAccumulators, class RegisterAccumulators, class Coord>
struct DrainOp {
    using Geometry = GemmGeometry<GemmConfig>;

    static constexpr int ACC_STAGES = Geometry::ACC_STAGES;

    const GemmResources<GemmConfig::BN, TmemStorage>& resources;
    TmemCopy tmem_copy;
    ThreadTmemAccumulators thread_tmem_accumulators;
    RegisterAccumulators register_accumulators;
    Coord coord;

    int previous_accumulators;
    int accumulator_sequence;
    int accumulator_slot;

    CUTE_DEVICE void wait_mma_and_load() {
        accumulator_slot = wait_accumulator_slot<ACC_STAGES>(
            resources.barriers, previous_accumulators, accumulator_sequence);
        halops::tcgen05_after_thread_sync();
        cute::copy(
            tmem_copy,
            thread_tmem_accumulators(cute::_, cute::_, cute::_, cute::Int<0>{},
                                     cute::Int<0>{}, accumulator_slot),
            register_accumulators);
        cutlass::arch::fence_view_async_tmem_load();
    }

    CUTE_DEVICE void end_item() {
        cutlass::arch::fence_view_async_tmem_load();
        resources.barriers.accumulator_free[accumulator_slot].arrive();
        ++accumulator_sequence;
    }
};

template <class GemmConfig, class TmemStorage>
CUTE_DEVICE auto make_drain_op(
    const GemmResources<GemmConfig::BN, TmemStorage>& resources,
    int previous_accumulators) {
    using Geometry = GemmGeometry<GemmConfig>;
    using Traits = typename Geometry::Traits;
    using Roles = IndexerWarpTraits::Roles;

    cute::Tensor tmem_acc = resources.tmem.tmem_accumulators;
    cute::Tensor acc = tmem_acc(cute::make_coord(cute::_, cute::_), cute::_0{},
                                cute::_0{}, cute::_);
    cute::Tensor acc_tile =
        cute::flat_divide(acc, typename Traits::EpiComputeTile{});
    cute::TiledCopy tmem_copy = cute::make_tmem_copy(
        typename Traits::CopyOpT2R{},
        acc_tile(cute::_, cute::_, cute::_0{}, cute::_0{}, cute::_0{}));
    static_assert(
        cute::size(decltype(tmem_copy){}) / 32 == Roles::DrainWarps,
        "the T2R copy needs one warp per TMEM datapath quadrant; DrainWarps "
        "must equal that count");

    const int warp = IndexerWarpTraits::warp_idx();
    const int lane = IndexerWarpTraits::lane_idx();

    cute::ThrCopy thread_tmem_copy =
        tmem_copy.get_slice((warp % Roles::DrainWarps) * 32 + lane);
    cute::Tensor thread_tmem_accumulators =
        thread_tmem_copy.partition_S(acc_tile);
    cute::Tensor thr_demo = thread_tmem_copy.partition_D(
        acc_tile(cute::_, cute::_, cute::_0{}, cute::_0{}, cute::_0{}));
    cute::Tensor register_accumulators =
        cute::make_tensor<float>(cute::shape(thr_demo));
    cute::Tensor coord =
        thread_tmem_copy.partition_D(cute::make_identity_tensor(cute::shape(
            acc_tile(cute::_, cute::_, cute::_0{}, cute::_0{}, cute::_0{}))));

    return DrainOp<GemmConfig, TmemStorage, decltype(tmem_copy),
                   decltype(thread_tmem_accumulators),
                   decltype(register_accumulators), decltype(coord)>{
        resources,
        tmem_copy,
        thread_tmem_accumulators,
        register_accumulators,
        coord,
        previous_accumulators,
        0,
        0};
}

template <class GemmConfig>
struct Bf16GemmTypes {
    static constexpr int BM = GemmConfig::BM;
    static constexpr int BN = GemmConfig::BN;
    static constexpr int BK = GemmConfig::BK;
    static constexpr int STAGES = GemmConfig::TmaStages;

    static_assert(BK % 64 == 0,
                  "SW128 at bf16 needs BK to be a multiple of 64");
    using SmemLayoutA1 =
        decltype(halops::indexer::intrinsics::make_umma_k_major_layout<BM, BK,
                                                                       bf16>());
    using SmemLayoutB1 =
        decltype(halops::indexer::intrinsics::make_umma_k_major_layout<BN, BK,
                                                                       bf16>());

    static constexpr int OPERAND_A_STAGE_BYTES =
        cute::cosize_v<SmemLayoutA1> * sizeof(bf16);
    static constexpr int OPERAND_B_STAGE_BYTES =
        cute::cosize_v<SmemLayoutB1> * sizeof(bf16);
    static constexpr int BUFFER_STAGE_COUNT = IndexerGemmConfig<BN>::TmaStages;
    static_assert(OPERAND_A_STAGE_BYTES ==
                          GemmInstance<BN, STAGES>::OPERAND_A_STAGE_BYTES &&
                      OPERAND_B_STAGE_BYTES ==
                          GemmInstance<BN, STAGES>::OPERAND_B_STAGE_BYTES,
                  "BF16 and FP8 stages must share the same A/B buffer sizes");
    static constexpr int SHARED_MEMORY_BYTES =
        BUFFER_STAGE_COUNT * (OPERAND_A_STAGE_BYTES + OPERAND_B_STAGE_BYTES);
    static constexpr int OPERAND_A_OFFSET = 0;
    static constexpr int OPERAND_B_OFFSET =
        BUFFER_STAGE_COUNT * OPERAND_A_STAGE_BYTES;

    using MmaAtom =
        cute::SM100_MMA_F16BF16_SS<bf16, bf16, float, BM, BN,
                                   cute::UMMA::Major::K, cute::UMMA::Major::K>;
    using TiledMma = decltype(cute::make_tiled_mma(MmaAtom{}));

    using TileA = cute::Shape<cute::Int<BM>, cute::Int<BK>>;
    using TileB = cute::Shape<cute::Int<BN>, cute::Int<BK>>;

    static constexpr uint32_t TMA_BYTES =
        (uint32_t)(OPERAND_A_STAGE_BYTES + OPERAND_B_STAGE_BYTES);

    using CopyOpT2R = cute::conditional_t<
        BN == 128, cute::SM100_TMEM_LOAD_32dp32b128x,
        cute::conditional_t<
            BN == 64, cute::SM100_TMEM_LOAD_32dp32b64x,
            cute::conditional_t<
                BN == 32, cute::SM100_TMEM_LOAD_32dp32b32x,
                cute::conditional_t<BN == 16, cute::SM100_TMEM_LOAD_32dp32b16x,
                                    cute::SM100_TMEM_LOAD_32dp32b8x>>>>;
    static_assert(BN >= 16 && BN <= 128 && (BN & (BN - 1)) == 0,
                  "N must be a power of two in [16, 128]");
};

template <class GemmConfig>
CUTE_HOST auto make_bf16_weight_tma(const void* weight, int m_total,
                                    int k_dim) {
    using GemmTypes = Bf16GemmTypes<GemmConfig>;
    auto layout = cute::make_layout(cute::make_shape(m_total, k_dim),
                                    cute::make_stride(k_dim, cute::_1{}));
    return cute::make_tma_copy(
        cute::SM90_TMA_LOAD{},
        cute::make_tensor(cute::make_gmem_ptr(static_cast<const bf16*>(weight)),
                          layout),
        typename GemmTypes::SmemLayoutA1{});
}

template <class GemmConfig>
CUTE_HOST auto make_bf16_activation_tma(const void* activations, int T,
                                        int row_stride, int k_dim) {
    using GemmTypes = Bf16GemmTypes<GemmConfig>;
    auto layout = cute::make_layout(cute::make_shape(T, k_dim),
                                    cute::make_stride(row_stride, cute::_1{}));
    return cute::make_tma_copy(
        cute::SM90_TMA_LOAD{},
        cute::make_tensor(
            cute::make_gmem_ptr(static_cast<const bf16*>(activations)), layout),
        typename GemmTypes::SmemLayoutB1{});
}

template <class GemmConfig, int St>
CUTE_DEVICE auto bf16_smem_a(char* p) {
    using GemmTypes = Bf16GemmTypes<GemmConfig>;
    return cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<bf16*>(
                                 p + GemmTypes::OPERAND_A_OFFSET +
                                 St * GemmTypes::OPERAND_A_STAGE_BYTES)),
                             typename GemmTypes::SmemLayoutA1{});
}
template <class GemmConfig, int St>
CUTE_DEVICE auto bf16_smem_b(char* p) {
    using GemmTypes = Bf16GemmTypes<GemmConfig>;
    return cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<bf16*>(
                                 p + GemmTypes::OPERAND_B_OFFSET +
                                 St * GemmTypes::OPERAND_B_STAGE_BYTES)),
                             typename GemmTypes::SmemLayoutB1{});
}

template <class GemmConfig, class Resources>
struct Bf16MmaOp {
    using GemmTypes = Bf16GemmTypes<GemmConfig>;
    static constexpr int STAGES = GemmConfig::TmaStages;
    static constexpr int ACC_STAGES = GemmConfig::AccStages;

    const Resources& resources;
    typename GemmTypes::TiledMma tiled_mma;
    uint32_t previous_stage_parities;

    int item_sequence;
    int previous_accumulators;
    int accumulator_sequence;
    int accumulator_slot;

    CUTE_DEVICE void begin_item() {
        accumulator_slot = reserve_accumulator_slot<ACC_STAGES>(
            resources.barriers, previous_accumulators, accumulator_sequence);
    }

    CUTE_DEVICE void wait_tma(int k) {
        wait_stage_tma<STAGES>(resources.barriers, previous_stage_parities,
                               item_sequence, resources.k_tiles, k);
    }

    CUTE_DEVICE void run_k_tile(int k) {
        halops::tcgen05_after_thread_sync();
        dispatch_stage<STAGES>(k % STAGES, stager(), k);
    }

    CUTE_DEVICE auto stager() {
        return [this](auto St, int k) {
            constexpr int s = decltype(St)::value;

            cute::Tensor tmem_frag = cute::partition_fragment_C(
                tiled_mma, cute::Shape<cute::Int<GemmConfig::BM>,
                                       cute::Int<GemmConfig::BN>>{});
            tmem_frag.data().get() =
                resources.tmem_base + (accumulator_slot * GemmConfig::BN);
            tiled_mma.accumulate_ =
                k == 0 ? cute::UMMA::ScaleOut::Zero : cute::UMMA::ScaleOut::One;
            auto thr_mma = tiled_mma.get_slice(cute::_0{});
            auto sA_frag = thr_mma.partition_fragment_A(
                bf16_smem_a<GemmConfig, s>(resources.gemm_shared_memory));
            auto sB_frag = thr_mma.partition_fragment_B(
                bf16_smem_b<GemmConfig, s>(resources.gemm_shared_memory));
            static_assert(cute::size<2>(sA_frag) == cute::size<2>(sB_frag));
            static_assert(cute::size<1>(sA_frag) == cute::size<1>(tmem_frag));
            static_assert(cute::size<1>(sB_frag) == cute::size<2>(tmem_frag));
            CUTE_UNROLL
            for (int k_inner = 0; k_inner < cute::size<2>(sA_frag); ++k_inner) {
                cute::gemm(tiled_mma, sA_frag(cute::_, cute::_, k_inner),
                           sB_frag(cute::_, cute::_, k_inner), tmem_frag);
                tiled_mma.accumulate_ = cute::UMMA::ScaleOut::One;
            }
        };
    }

    CUTE_DEVICE void commit(int k) {
        __syncwarp();
        cutlass::arch::umma_arrive(reinterpret_cast<uint64_t const*>(
            &resources.barriers.buffer_free[k % STAGES]));
    }

    CUTE_DEVICE void end_item() {
        cutlass::arch::umma_arrive(reinterpret_cast<uint64_t const*>(
            &resources.barriers.accumulator_ready[accumulator_slot]));
        ++item_sequence;
        ++accumulator_sequence;
    }
};

template <class GemmConfig, class Resources>
CUTE_DEVICE auto make_bf16_mma_op(const Resources& resources,
                                  uint32_t previous_stage_parities,
                                  int previous_accumulators) {
    return Bf16MmaOp<GemmConfig, Resources>{
        resources,
        typename Bf16GemmTypes<GemmConfig>::TiledMma{},
        previous_stage_parities,
        0,
        previous_accumulators,
        0,
        0};
}

struct GemmWorkList {
    int token_groups;
    int num_tokens;
};

struct RangeBarrierLedger {
    uint32_t parity = 0;
    uint32_t touched = 0;
    int accumulator_items = 0;

    template <int STAGES>
    CUTE_DEVICE void contribute(int tiles_per_split, int n) {
        if (n == 0) return;
        CUTLASS_PRAGMA_UNROLL
        for (int s = 0; s < STAGES; ++s) {
            const int arrives = n * stage_use_count<STAGES>(tiles_per_split, s);
            parity ^= (uint32_t)(arrives & 1) << s;
            touched |= 1u << s;
        }
        accumulator_items += n;
    }

    CUTE_DEVICE uint32_t previous_stage_parities() const { return parity; }
    CUTE_DEVICE uint32_t unused_stage_mask() const { return ~touched; }
    CUTE_DEVICE int previous_accumulators() const { return accumulator_items; }
};

template <class GemmConfig>
CUTE_DEVICE GemmWorkItem gemm_item(const GemmWorkList& work_list,
                                   int work_index, int n_pad, int m_total,
                                   int tiles_per_split) {
    const int token_group = work_index % work_list.token_groups;
    const int rest = work_index / work_list.token_groups;
    const int split_index = rest % GemmConfig::KSplit;
    const int weight_tile = rest / GemmConfig::KSplit;
    const int n_base = token_group * n_pad;
    return GemmWorkItem{
        .n_row = n_base,
        .weight_tile = weight_tile,
        .m_total = m_total,
        .k_tile_begin = split_index * tiles_per_split,
        .a_k_tile_begin = split_index * tiles_per_split,
        .out_row_stride = GemmConfig::KSplit * m_total,
        .valid_tokens = min(n_pad, work_list.num_tokens - n_base)};
}

struct KeyWeightWorkItem {
    int weight_tile;  // 0: key rows; 1: head-weight rows.
    int token_group;
    int split_index;
    int first_k_tile;
    int valid_tokens;
};

template <class GemmConfig>
CUTE_DEVICE KeyWeightWorkItem key_weight_work_item_at(int work_index,
                                                      int token_groups,
                                                      int key_weight_splits,
                                                      int tiles_per_split,
                                                      int num_tokens) {
    const int token_group = work_index % token_groups;
    const int rest = work_index / token_groups;
    const int split_index = rest % key_weight_splits;
    const int weight_tile = rest / key_weight_splits;
    return KeyWeightWorkItem{
        weight_tile, token_group, split_index, split_index * tiles_per_split,
        min(GemmConfig::BN, num_tokens - token_group * GemmConfig::BN)};
}

template <class WeightTma, class ActivationTma>
struct KeyWeightResources {
    const WeightTma& tma_w;
    const ActivationTma& tma_act;
    char* __restrict__ gemm_shared_memory;
    GemmPipelineBarriers barriers;
    uint32_t tmem_base;
    int activation_rows;
    int m_total;
    int k_dim;
    int k_tiles;
    float* __restrict__ partials;
    int split_count;
};

template <class GemmConfig, class Resources>
struct KeyWeightWeightLoader {
    using GemmTypes = Bf16GemmTypes<GemmConfig>;
    static constexpr int STAGES = GemmConfig::TmaStages;

    const Resources& resources;
    bool elected;
    uint32_t previous_stage_parities;
    uint32_t unused_stage_mask;
    int item_sequence;
    int weight_tile;
    int first_k_tile;

    CUTE_DEVICE void begin_item(const KeyWeightWorkItem& work_item) {
        weight_tile = work_item.weight_tile;
        first_k_tile = work_item.first_k_tile;
    }
    CUTE_DEVICE void end_item() { ++item_sequence; }

    CUTE_DEVICE void issue_tile(int t) {
        wait_stage_free<STAGES>(resources.barriers, previous_stage_parities,
                                unused_stage_mask, item_sequence,
                                resources.k_tiles, t);
        if (!elected) return;
        dispatch_stage<STAGES>(t % STAGES, issuer(), t);
    }

    CUTE_DEVICE auto issuer() {
        return [this](auto St, int k) {
            constexpr int s = decltype(St)::value;
            resources.barriers.tma_a[s].arrive_and_expect_tx(
                GemmTypes::OPERAND_A_STAGE_BYTES);
            uint64_t& mbar =
                reinterpret_cast<uint64_t&>(resources.barriers.tma_a[s]);
            auto mA = resources.tma_w.get_tma_tensor(
                cute::make_shape(resources.m_total, resources.k_dim));
            auto thr_w = resources.tma_w.get_slice(cute::_0{});
            auto tA = cute::local_tile(
                mA, typename GemmTypes::TileA{},
                cute::make_coord(weight_tile, k + first_k_tile));
            cute::copy(resources.tma_w.with(mbar, 0), thr_w.partition_S(tA),
                       thr_w.partition_D(bf16_smem_a<GemmConfig, s>(
                           resources.gemm_shared_memory)));
        };
    }
};

template <class GemmConfig, class Resources>
struct KeyWeightActivationLoader {
    using GemmTypes = Bf16GemmTypes<GemmConfig>;
    static constexpr int STAGES = GemmConfig::TmaStages;

    const Resources& resources;
    bool elected;
    uint32_t previous_stage_parities;
    uint32_t unused_stage_mask;
    int item_sequence;
    int n_tile;
    int first_k_tile;

    CUTE_DEVICE void begin_item(const KeyWeightWorkItem& work_item) {
        n_tile = work_item.token_group;
        first_k_tile = work_item.first_k_tile;
    }
    CUTE_DEVICE void end_item() { ++item_sequence; }

    CUTE_DEVICE void issue_tile(int t) {
        wait_stage_free<STAGES>(resources.barriers, previous_stage_parities,
                                unused_stage_mask, item_sequence,
                                resources.k_tiles, t);
        if (!elected) return;
        dispatch_stage<STAGES>(t % STAGES, issuer(), t);
    }

    CUTE_DEVICE auto issuer() {
        return [this](auto St, int k) {
            constexpr int s = decltype(St)::value;
            resources.barriers.tma_b[s].arrive_and_expect_tx(
                GemmTypes::OPERAND_B_STAGE_BYTES);
            uint64_t& mbar =
                reinterpret_cast<uint64_t&>(resources.barriers.tma_b[s]);
            auto mB = resources.tma_act.get_tma_tensor(
                cute::make_shape(resources.activation_rows, resources.k_dim));
            auto thr_act = resources.tma_act.get_slice(cute::_0{});
            auto tB =
                cute::local_tile(mB, typename GemmTypes::TileB{},
                                 cute::make_coord(n_tile, k + first_k_tile));
            cute::copy(resources.tma_act.with(mbar, 0), thr_act.partition_S(tB),
                       thr_act.partition_D(bf16_smem_b<GemmConfig, s>(
                           resources.gemm_shared_memory)));
        };
    }
};

template <class GemmConfig, class Resources, class TmemCopy,
          class ThreadTmemAccumulators, class RegisterAccumulators, class Coord>
struct KeyWeightAccumulatorConsumer {
    static constexpr int ACC_STAGES = GemmConfig::AccStages;

    const Resources& resources;
    TmemCopy tmem_copy;
    ThreadTmemAccumulators thread_tmem_accumulators;
    RegisterAccumulators register_accumulators;
    Coord coord;
    int previous_accumulators;
    int accumulator_sequence;
    int accumulator_slot;

    CUTE_DEVICE void wait_mma_and_load() {
        accumulator_slot = wait_accumulator_slot<ACC_STAGES>(
            resources.barriers, previous_accumulators, accumulator_sequence);
        halops::tcgen05_after_thread_sync();
        ThreadTmemAccumulators src = thread_tmem_accumulators;
        src.data() = src.data().get() + (accumulator_slot * GemmConfig::BN);
        cute::copy(tmem_copy, src, register_accumulators);
        cutlass::arch::fence_view_async_tmem_load();
    }

    CUTE_DEVICE void store(const KeyWeightWorkItem& work_item) const {
        const int64_t stride =
            (int64_t)resources.split_count * IndexerDimensions::KEY_WEIGHT_ROWS;
        float* base =
            resources.partials +
            (int64_t)(work_item.token_group * GemmConfig::BN) * stride +
            (int64_t)work_item.split_index * IndexerDimensions::KEY_WEIGHT_ROWS;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < cute::size(register_accumulators); ++i) {
            const int row =
                cute::get<0>(coord(i)) + work_item.weight_tile * GemmConfig::BM;
            const int local_token = cute::get<1>(coord(i));
            if (row < IndexerDimensions::KEY_WEIGHT_ROWS &&
                local_token < work_item.valid_tokens)
                base[local_token * stride + row] = register_accumulators(i);
        }
    }

    CUTE_DEVICE void end_item() {
        cutlass::arch::fence_view_async_tmem_load();
        resources.barriers.accumulator_free[accumulator_slot].arrive();
        ++accumulator_sequence;
    }
};

template <class GemmConfig, class Resources>
CUTE_DEVICE auto make_key_weight_consumer(const Resources& resources,
                                          int previous_accumulators) {
    using GemmTypes = Bf16GemmTypes<GemmConfig>;
    using Roles = IndexerTraits::Roles;
    static_assert(Roles::DrainWarps == 4 && Roles::DrainWarpBegin % 4 == 0,
                  "the T2R copy needs one warp per TMEM datapath quadrant, "
                  "based at a multiple of 4");

    typename GemmTypes::TiledMma tiled_mma;
    cute::Tensor frag0 = cute::partition_fragment_C(
        tiled_mma,
        cute::Shape<cute::Int<GemmConfig::BM>, cute::Int<GemmConfig::BN>>{});
    frag0.data().get() = resources.tmem_base;
    cute::Tensor acc0 =
        frag0(cute::make_coord(cute::_, cute::_), cute::_0{}, cute::_0{});
    cute::TiledCopy tmem_copy =
        cute::make_tmem_copy(typename GemmTypes::CopyOpT2R{}, acc0);
    static_assert(cute::size(decltype(tmem_copy){}) / 32 == Roles::DrainWarps,
                  "the T2R copy must need exactly DrainWarps warps");

    const int warp = IndexerTraits::warp_idx();
    const int lane = IndexerTraits::lane_idx();
    cute::ThrCopy thread_tmem_copy =
        tmem_copy.get_slice((warp % Roles::DrainWarps) * 32 + lane);
    cute::Tensor thr_demo = thread_tmem_copy.partition_D(acc0);
    cute::Tensor register_accumulators =
        cute::make_tensor<float>(cute::shape(thr_demo));
    cute::Tensor coord = thread_tmem_copy.partition_D(
        cute::make_identity_tensor(cute::shape(acc0)));

    using Op = KeyWeightAccumulatorConsumer<
        GemmConfig, Resources, decltype(tmem_copy),
        decltype(thread_tmem_copy.partition_S(acc0)),
        decltype(register_accumulators), decltype(coord)>;
    return Op{resources,
              tmem_copy,
              thread_tmem_copy.partition_S(acc0),
              register_accumulators,
              coord,
              previous_accumulators,
              0,
              0};
}

template <class GemmConfig, class Resources>
CUTE_DEVICE auto make_key_weight_loader(const Resources& resources,
                                        uint32_t previous_stage_parities,
                                        uint32_t unused_stage_mask) {
    return KeyWeightWeightLoader<GemmConfig, Resources>{
        resources,
        cute::elect_one_sync() != 0,
        previous_stage_parities,
        unused_stage_mask,
        0,
        0,
        0};
}

template <class GemmConfig, class Resources>
CUTE_DEVICE auto make_key_weight_activation_loader(
    const Resources& resources, uint32_t previous_stage_parities,
    uint32_t unused_stage_mask) {
    return KeyWeightActivationLoader<GemmConfig, Resources>{
        resources,
        cute::elect_one_sync() != 0,
        previous_stage_parities,
        unused_stage_mask,
        0,
        0,
        0};
}

}  // namespace halops::indexer
