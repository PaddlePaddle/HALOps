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

#include "kernel/mla/mega_mla/output_proj/params.cuh"

namespace halops::mla {

template <class C>
CUTE_DEVICE int oproj_pair_wo_a_k_tiles(const OutputProjShape& shape) {
    return shape.group_k / C::kBlockK;
}
template <class C>
CUTE_DEVICE int oproj_pair_wo_b_k_tiles(const OutputProjShape& shape) {
    return shape.olora / C::kBlockK / 2;
}

template <class C>
CUTE_DEVICE auto oproj_pair_smem_views(typename C::Smem& smem) {
    using Tr = typename C::Tr;
    using TrB = typename C::TrB;
    return cute::make_tuple(
        cute::make_tensor(cute::make_smem_ptr(smem.A.begin()),
                          typename Tr::SmemLayoutA{}),
        cute::make_tensor(cute::make_smem_ptr(smem.B.begin()),
                          typename TrB::SmemLayoutB{}),
        cute::make_tensor(cute::make_smem_ptr(smem.SFA.begin()),
                          typename Tr::SmemLayoutSFA{}),
        cute::make_tensor(cute::make_smem_ptr(smem.SFB.begin()),
                          typename TrB::SmemLayoutSFB{}));
}

template <class C, class MP>
CUTE_DEVICE auto oproj_pair_tma_parts(const MP& mp, typename C::Smem& smem,
                                      int half) {
    using Tr = typename C::Tr;
    auto cluster_layout =
        cute::tiled_divide(cute::make_layout(typename Tr::ClusterShape_MNK{}),
                           cute::make_tile(typename Tr::TiledMma::AtomThrID{}));
    auto cluster_layout_sf = cute::tiled_divide(
        cute::make_layout(typename Tr::ClusterShape_MNK{}),
        cute::make_tile(typename Tr::TiledMma_SF::AtomThrID{}));
    const auto cta = cluster_layout.get_flat_coord(half);
    const auto cta_sf = cluster_layout_sf.get_flat_coord(half);
    auto [sA, sB, sSFA, sSFB] = oproj_pair_smem_views<C>(smem);
    typename Tr::TiledMma tiled_mma;
    auto mma = tiled_mma.get_slice(half);
    auto mA = mp.tma_load_A.get_tma_tensor(mp.layout_A.shape());
    auto mB = mp.tma_load_B.get_tma_tensor(mp.layout_B.shape())(
        cute::_, cute::_, cute::Int<0>{});
    auto gA = cute::local_tile(mA, typename Tr::TileShape{},
                               cute::make_coord(cute::_, cute::_, cute::_),
                               cute::Step<cute::_1, cute::X, cute::_1>{});
    auto gB = cute::local_tile(mB, typename Tr::TileShape{},
                               cute::make_coord(cute::_, cute::_, cute::_),
                               cute::Step<cute::X, cute::_1, cute::_1>{});
    auto [tgA, tsA] = cute::tma_partition(
        mp.tma_load_A, cute::get<2>(cta),
        cute::make_layout(cute::size<2>(cluster_layout)),
        cute::group_modes<0, C::kTmaModes>(sA),
        cute::group_modes<0, C::kTmaModes>(mma.partition_A(gA)));
    auto [tgB, tsB] = cute::tma_partition(
        mp.tma_load_B, cute::get<1>(cta),
        cute::make_layout(cute::size<1>(cluster_layout)),
        cute::group_modes<0, C::kTmaModes>(sB),
        cute::group_modes<0, C::kTmaModes>(mma.partition_B(gB)));
    auto mSFA = mp.tma_load_SFA.get_tma_tensor(mp.layout_SFA.shape());
    auto mSFB = mp.tma_load_SFB.get_tma_tensor(mp.layout_SFB.shape())(
        cute::_, cute::_, cute::Int<0>{});
    auto mma_sf = typename Tr::TiledMma_SF{}.get_slice(
        half % cute::size(typename Tr::TiledMma_SF::AtomThrID{}));
    auto gSFA = cute::local_tile(mSFA, typename Tr::TileShape{},
                                 cute::make_coord(cute::_, cute::_, cute::_),
                                 cute::Step<cute::_1, cute::X, cute::_1>{});
    auto gSFB = cute::local_tile(mSFB, typename Tr::TileShape_SF{},
                                 cute::make_coord(cute::_, cute::_, cute::_),
                                 cute::Step<cute::X, cute::_1, cute::_1>{});
    auto [tgSFA, tsSFA] = cute::tma_partition(
        mp.tma_load_SFA, cute::get<2>(cta),
        cute::make_layout(cute::size<2>(cluster_layout)),
        cute::group_modes<0, C::kTmaModes>(sSFA),
        cute::group_modes<0, C::kTmaModes>(mma.partition_A(gSFA)));
    auto [tgSFB, tsSFB] = cute::tma_partition(
        mp.tma_load_SFB, cute::get<1>(cta_sf),
        cute::make_layout(cute::size<1>(cluster_layout_sf)),
        cute::group_modes<0, C::kTmaModes>(sSFB),
        cute::group_modes<0, C::kTmaModes>(mma_sf.partition_B(gSFB)));
    return cute::make_tuple(tgA, tsA, tgB, tsB, tgSFA, tsSFA, tgSFB, tsSFB);
}

template <class C, int Mode, class AtomThr>
CUTE_DEVICE uint16_t oproj_pair_mcast_mask(int rank) {
    auto cluster_layout = cute::tiled_divide(
        cute::make_layout(typename C::Tr::ClusterShape_MNK{}),
        cute::make_tile(AtomThr{}));
    return (
        uint16_t)(cute::create_tma_multicast_mask<Mode>(
                      cluster_layout, cluster_layout.get_flat_coord(rank & 1))
                  << (rank & ~1));
}
template <class C>
CUTE_DEVICE uint16_t oproj_pair_mask_a(int rank) {
    return oproj_pair_mcast_mask<C, 2, typename C::Tr::TiledMma::AtomThrID>(
        rank);
}
template <class C>
CUTE_DEVICE uint16_t oproj_pair_mask_b(int rank) {
    return oproj_pair_mcast_mask<C, 1, typename C::Tr::TiledMma::AtomThrID>(
        rank);
}
template <class C>
CUTE_DEVICE uint16_t oproj_pair_mask_sfb(int rank) {
    return oproj_pair_mcast_mask<C, 1, typename C::Tr::TiledMma_SF::AtomThrID>(
        rank);
}

template <class C>
CUTE_DEVICE void oproj_pair_wait_slot_free(typename C::Smem& smem, int seq) {
    if (seq >= C::Stages)
        cutlass::arch::ClusterBarrier::wait(
            &smem.smem_free[seq % C::Stages],
            (uint32_t)(((seq / C::Stages) - 1) & 1));
}

template <class C, class Prm>
CUTE_DEVICE void oproj_pair_weight_role(const Prm& params,
                                        typename C::Smem& smem,
                                        const OprojPairCoord<C>& coord,
                                        int lane) {
    using ML = typename C::ML;
    constexpr uint32_t kBytes =
        ML::TmaTransactionBytesA + ML::TmaTransactionBytesSFA;
    auto parts_a = oproj_pair_tma_parts<C>(params.wo_a, smem, coord.half);
    auto parts_b = oproj_pair_tma_parts<C>(params.wo_b, smem, coord.half);
    const uint16_t mask = oproj_pair_mask_a<C>(coord.rank);
    auto issue = [&](auto const& mp, auto& parts, int seq, int tile,
                     int k_tile) {
        const int stage = seq % C::Stages;
        oproj_pair_wait_slot_free<C>(smem, seq);
        if (lane == 0) {
            if (coord.half == 0)
                cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(
                    &smem.tma_a[stage], kBytes);
            cute::copy(
                mp.tma_load_A.with(smem.tma_a[stage], mask),
                cute::get<C::kPartGA>(parts)(cute::_, tile, k_tile, 0),
                cute::get<C::kPartSA>(parts)(cute::_, (cute::Int<0>{}, stage)));
            cute::copy(mp.tma_load_SFA.with(smem.tma_a[stage], mask),
                       cute::get<C::kPartGSFA>(parts)(cute::_, tile, k_tile, 0),
                       cute::get<C::kPartSSFA>(parts)(cute::_,
                                                      (cute::Int<0>{}, stage)));
        }
        __syncwarp();
    };
    const int k_a = oproj_pair_wo_a_k_tiles<C>(params.shape),
              k_b = oproj_pair_wo_b_k_tiles<C>(params.shape);
#pragma unroll 1
    for (int k = 0; k < k_a; ++k)
        issue(params.wo_a, parts_a, k, coord.tile_a, k);
#pragma unroll 1
    for (int k = 0; k < k_b; ++k)
        issue(params.wo_b, parts_b, k_a + k, coord.tile_b,
              k_b * coord.pair + k);
}

template <class C, class Prm>
CUTE_DEVICE void oproj_pair_act_role(const Prm& params, typename C::Smem& smem,
                                     const OprojPairCoord<C>& coord, int lane) {
    using ML = typename C::ML;
    constexpr uint32_t kBytes =
        C::NH * (ML::TmaTransactionBytesB + ML::TmaTransactionBytesSFB);
    constexpr int kTilesPerK = C::kBlockK / OutputProjLayout::RowTile;
    auto parts_a = oproj_pair_tma_parts<C>(params.wo_a, smem, coord.half);
    auto parts_b = oproj_pair_tma_parts<C>(params.wo_b, smem, coord.half);
    const uint16_t mask = oproj_pair_mask_b<C>(coord.rank);
    const uint16_t mask_sf = oproj_pair_mask_sfb<C>(coord.rank);
    pdl_wait();
    auto issue = [&](auto const& mp, auto& parts, int seq, int group,
                     int k_tile) {
        const int stage = seq % C::Stages;
        oproj_pair_wait_slot_free<C>(smem, seq);
        if (lane == 0) {
            if (coord.half == 0)
                cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(
                    &smem.tma_b[stage], kBytes);
            CUTLASS_PRAGMA_UNROLL
            for (int part = 0; part < C::NH; ++part) {
                cute::copy(
                    mp.tma_load_B.with(smem.tma_b[stage], mask),
                    cute::get<C::kPartGB>(parts)(cute::_, C::NH * group + part,
                                                 k_tile),
                    cute::get<C::kPartSB>(parts)(
                        cute::_, (cute::Int<0>{}, C::NH * stage + part)));
                cute::copy(
                    mp.tma_load_SFB.with(smem.tma_b[stage], mask_sf),
                    cute::get<C::kPartGSFB>(parts)(
                        cute::_, C::NH * group + part, k_tile),
                    cute::get<C::kPartSSFB>(parts)(
                        cute::_, (cute::Int<0>{}, C::NH * stage + part)));
            }
        }
        __syncwarp();
    };
    const int k_a = oproj_pair_wo_a_k_tiles<C>(params.shape),
              k_b = oproj_pair_wo_b_k_tiles<C>(params.shape);
    const int out_group =
        coord.tile_a / (params.shape.group_rows / C::kPairRows);
#pragma unroll 1
    for (int k = 0; k < k_a; ++k)
        issue(params.wo_a, parts_a, k, coord.group_a, out_group * k_a + k);
    const int tiles = params.shape.wo_a_tiles(), half_tiles = tiles / 2;
    const int* flag =
        params.flags + tiles * coord.group_b + half_tiles * coord.pair + lane;
    uint32_t ready = 0;
#pragma unroll 1
    for (int k = 0; k < k_b; ++k) {
        const uint32_t need = ((1u << kTilesPerK) - 1) << (kTilesPerK * k);
        while ((ready & need) != need)
            ready = __ballot_sync(
                ~0u, lane < half_tiles && gmem_ld_acquire(flag) >= 1);
        __syncwarp();
        fence_proxy_async_global();
        issue(params.wo_b, parts_b, k_a + k, coord.group_b,
              k_b * coord.pair + k);
    }
}

template <class UtccpOp, class SmemSf, class TmemSf>
CUTE_DEVICE auto oproj_pair_make_sf_copy(const SmemSf& smem_sf,
                                         const TmemSf& tmem_sf) {
    auto src =
        cute::make_tensor(smem_sf.data(), cute::filter_zeros(smem_sf.layout()));
    auto dst =
        cute::make_tensor(tmem_sf.data(), cute::filter_zeros(tmem_sf.layout()));
    auto copy = cute::make_utccp_copy(UtccpOp{}, dst);
    auto thr = copy.get_slice(0);
    return cute::make_tuple(
        copy, cute::get_utccp_smem_desc_tensor<UtccpOp>(thr.partition_S(src)),
        thr.partition_D(dst));
}

template <class SfCopy>
CUTE_DEVICE void oproj_pair_copy_sf(const SfCopy& sf, int stage) {
    cute::copy(cute::get<0>(sf),
               cute::get<1>(sf)(cute::_, cute::_, cute::_, cute::_, stage),
               cute::get<2>(sf)(cute::_, cute::_, cute::_, cute::_));
}

template <int KIters, class TiledMma, class Tmem, class FragA, class FragB>
CUTE_DEVICE void oproj_pair_mma_k_tile(TiledMma& tiled_mma, const Tmem& tmem,
                                       const FragA& rA, const FragB& rB,
                                       int stage_a, int stage_b, int acc_col,
                                       bool first) {
    auto acc = tmem.tmem_accumulators;
    acc.data() = acc.data().get() + acc_col;
    CUTLASS_PRAGMA_UNROLL
    for (int kb = 0; kb < KIters; ++kb) {
        tiled_mma.accumulate_ = (first && kb == 0) ? cute::UMMA::ScaleOut::Zero
                                                   : cute::UMMA::ScaleOut::One;
        cute::gemm(tiled_mma.with(tiled_mma.accumulate_,
                                  tmem.tmem_SFA(cute::_, cute::_, kb),
                                  tmem.tmem_SFB(cute::_, cute::_, kb)),
                   rA(cute::_, cute::_, kb, stage_a),
                   rB(cute::_, cute::_, kb, stage_b),
                   acc(cute::_, cute::_, cute::_, 0));
    }
}

template <class C, class Prm>
CUTE_DEVICE void oproj_pair_mma_role(const Prm& params, typename C::Smem& smem,
                                     const OprojPairCoord<C>& coord) {
    using Tr = typename C::Tr;
    using ML = typename C::ML;
    const uint16_t pair_mask = (uint16_t)(C::kPairCtaMask << (coord.rank & ~1));
    const auto views = oproj_pair_smem_views<C>(smem);
    typename Tr::TiledMma tiled_mma;
    auto mma = tiled_mma.get_slice(0);
    auto rA = mma.make_fragment_A(cute::get<C::kViewA>(views));
    auto rB = mma.make_fragment_B(cute::get<C::kViewB>(views));
    auto tmem = ML::init_tmem_tensors();
    ML::set_tmem_offsets(tmem, smem.tmem);
    tmem.tmem_SFA.data() = tmem.tmem_SFA.data().get() + (C::NH - 1) * C::TN;
    tmem.tmem_SFB.data() = tmem.tmem_SFB.data().get() + (C::NH - 1) * C::TN;
    const auto sfa = oproj_pair_make_sf_copy<typename ML::UtccpOp>(
        cute::get<C::kViewSFA>(views), tmem.tmem_SFA);
    const auto sfb = oproj_pair_make_sf_copy<typename ML::UtccpOp>(
        cute::get<C::kViewSFB>(views), tmem.tmem_SFB);
    const bool elected = cute::elect_one_sync();
    auto run = [&](int seq0, int k_tiles) {
#pragma unroll 1
        for (int k = 0; k < k_tiles; ++k) {
            const int seq = seq0 + k, stage = seq % C::Stages;
            const uint32_t phase = (uint32_t)((seq / C::Stages) & 1);
            cutlass::arch::ClusterBarrier::wait(&smem.tma_a[stage], phase);
            cutlass::arch::ClusterBarrier::wait(&smem.tma_b[stage], phase);
            tcgen05_after_thread_sync();
            if (elected) oproj_pair_copy_sf(sfa, stage);
            CUTLASS_PRAGMA_UNROLL
            for (int part = 0; part < C::NH; ++part) {
                if (elected) oproj_pair_copy_sf(sfb, C::NH * stage + part);
                oproj_pair_mma_k_tile<Tr::kItersK>(tiled_mma, tmem, rA, rB,
                                                   stage, C::NH * stage + part,
                                                   part * C::TN, k == 0);
            }
            cutlass::arch::umma_arrive_multicast_2x1SM(&smem.smem_free[stage],
                                                       pair_mask);
        }
        cutlass::arch::umma_arrive_multicast_2x1SM(&smem.acc_full, pair_mask);
    };
    const int k_a = oproj_pair_wo_a_k_tiles<C>(params.shape);
    run(0, k_a);
    cutlass::arch::ClusterBarrier::wait(&smem.acc_empty, 0u);
    tcgen05_after_thread_sync();
    run(k_a, oproj_pair_wo_b_k_tiles<C>(params.shape));
}

template <class C>
CUTE_DEVICE void oproj_pair_tmem_load_slice(uint32_t taddr,
                                            float (&vals)[C::kSlice]) {
    tmem_ld_32dp32bNx<C::kSlice>(taddr, vals);
    cutlass::arch::fence_view_async_tmem_load();
}

template <class C>
CUTE_DEVICE void oproj_pair_wg_sync(const OprojPairDrain<C>& d) {
    using Bars = KernelBase::Bars;
    cutlass::arch::NamedBarrier::arrive_and_wait(
        Bars::PairWg0Sync::width,
        d.wg == 0 ? Bars::PairWg0Sync::id : Bars::PairWg1Sync::id);
}
CUTE_DEVICE void oproj_pair_both_sync() {
    bar_sync<KernelBase::Bars::PairBothSync>();
}
template <class C>
CUTE_DEVICE bool oproj_pair_leader(const OprojPairDrain<C>& d) {
    return d.tid == 0 && d.wg == 0;
}
CUTE_DEVICE uint32_t oproj_pair_peer(const void* ptr, int rank) {
    return cute::set_block_rank(cute::cast_smem_ptr_to_uint(ptr),
                                (uint32_t)rank);
}
template <class C, class Prm>
CUTE_DEVICE int oproj_pair_token_end(const Prm& params, int group) {
    return min(params.num_tokens, group * C::TT + C::TT);
}
template <class C>
CUTE_DEVICE void oproj_pair_load_acc(const OprojPairDrain<C>& d, int slice,
                                     bool dry, float (&acc)[C::kSlice]) {
    if (!dry) {
        oproj_pair_tmem_load_slice<C>(d.tmem_base + C::kSlice * slice, acc);
    } else {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < C::kSlice; ++i) acc[i] = 0.0f;
    }
}

template <class C>
CUTE_DEVICE int oproj_pair_quant_token(const OprojPairDrain<C>& d) {
    return d.tid / C::GroupsPerToken;
}
template <class C>
CUTE_DEVICE int oproj_pair_quant_quarter(const OprojPairDrain<C>& d) {
    return d.tid % C::GroupsPerToken;
}

template <class C>
CUTE_DEVICE void oproj_pair_transpose_slice(const OprojPairDrain<C>& d,
                                            const float (&acc)[C::kSlice],
                                            bool first) {
    if (!first) oproj_pair_wg_sync(d);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < C::kSlice; ++i)
        d.transpose[i * C::TransposeRow + d.tid] = __float2bfloat16(acc[i]);
    oproj_pair_wg_sync(d);
}

template <class C>
CUTE_DEVICE void oproj_pair_load_group(const OprojPairDrain<C>& d,
                                       float (&vals)[SfLayout::Gran]) {
    const __nv_bfloat16* group = d.transpose +
                                 oproj_pair_quant_token(d) * C::TransposeRow +
                                 SfLayout::Gran * oproj_pair_quant_quarter(d);
    CUTLASS_PRAGMA_UNROLL
    for (int vec = 0; vec < SfLayout::Gran / C::BfPerVec; ++vec) {
        const uint4 word =
            *reinterpret_cast<const uint4*>(group + C::BfPerVec * vec);
        float part[C::BfPerVec];
        unpack_bf16(&word, part);
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < C::BfPerVec; ++i)
            vals[C::BfPerVec * vec + i] = part[i];
    }
}

template <class C, class Prm, class SfTensor>
CUTE_DEVICE void oproj_pair_store_group(const OprojPairDrain<C>& d,
                                        const Prm& params, SfTensor& sf,
                                        int row,
                                        const uint32_t (&packed)[C::GroupWords],
                                        float biased_exp) {
    const int col = d.coord.quant_tile() * OutputProjLayout::RowTile +
                    SfLayout::Gran * oproj_pair_quant_quarter(d);
    uint4* dst =
        reinterpret_cast<uint4*>(reinterpret_cast<uint8_t*>(params.olora_fp8) +
                                 (int64_t)row * params.shape.olora + col);
    dst[0] =
        make_uint4(packed[0], packed[1], packed[2], packed[C::kVecWords - 1]);
    dst[1] = make_uint4(packed[C::kVecWords], packed[C::kVecWords + 1],
                        packed[C::kVecWords + 2], packed[C::GroupWords - 1]);
    sf(sf_row_grouped(row, C::TN, 2 * C::NH), col, cute::_0{}) =
        (uint8_t)biased_exp;
}

template <class C, class Prm>
CUTE_DEVICE void oproj_pair_quant_wo_a(const OprojPairDrain<C>& d,
                                       const Prm& params) {
    const int token_lo = d.coord.group_a * C::TT;
    const int token_hi = oproj_pair_token_end<C>(params, d.coord.group_a);
    cute::Tensor sf = cute::make_tensor(cute::make_gmem_ptr(params.olora_sf),
                                        params.wo_b.layout_SFB);
#pragma unroll 1
    for (int it = -1; it < C::Slices / 2; ++it) {
        const bool dry = it < 0;
        const int slice = dry ? 0 : 2 * it + d.wg;
        float acc[C::kSlice];
        oproj_pair_load_acc(d, slice, dry, acc);
        if (it == C::Slices / 2 - 1) {
            tcgen05_before_thread_sync();
            mbar_arrive_cluster_relaxed(
                oproj_pair_peer(&d.smem.acc_empty, d.coord.rank & ~1));
        }
        oproj_pair_transpose_slice(d, acc, it <= 0);
        float vals[SfLayout::Gran];
        oproj_pair_load_group(d, vals);
        const Ue8m0Scale scale = ue8m0_scale(abs_max(vals));
        uint32_t packed[C::GroupWords];
        pack_e4m3(vals, scale.multiplier, packed);
        const int row =
            token_lo + C::kSlice * slice + oproj_pair_quant_token(d);
        if (row < token_hi && !dry)
            oproj_pair_store_group(d, params, sf, row, packed,
                                   scale.biased_exp);
        if (dry) {
            oproj_pair_wg_sync(d);
            cutlass::arch::ClusterBarrier::wait(&d.smem.acc_full, 0u);
            tcgen05_after_thread_sync();
        }
    }
    __threadfence();
    fence_proxy_async_global();
    oproj_pair_both_sync();
    if (oproj_pair_leader(d))
        gmem_red_add1_relaxed(params.flags +
                              params.shape.wo_a_tiles() * d.coord.group_a +
                              d.coord.quant_tile());
}

template <class C>
CUTE_DEVICE float* oproj_pair_ring(const OprojPairDrain<C>& d) {
    return reinterpret_cast<float*>(d.smem.A.begin());
}
template <class C>
CUTE_DEVICE uint32_t oproj_pair_ring_smem(const OprojPairDrain<C>& d,
                                          int unit) {
    return cute::cast_smem_ptr_to_uint(oproj_pair_ring(d)) +
           (uint32_t)(unit * C::ExchangeBytes);
}
template <class C>
CUTE_DEVICE int oproj_pair_partner(const OprojPairDrain<C>& d) {
    return d.coord.rank ^ 2;
}

template <class C>
CUTE_DEVICE void oproj_pair_send_slice(const OprojPairDrain<C>& d, int slice,
                                       int unit, bool dry) {
    constexpr int kV = C::kVecFloats;
    float acc[C::kSlice];
    oproj_pair_load_acc(d, slice, dry, acc);
    float* staging =
        dry ? reinterpret_cast<float*>(d.transpose)
            : oproj_pair_ring(d) + (C::Units + unit) * C::UnitFloats;
    CUTLASS_PRAGMA_UNROLL
    for (int c4 = 0; c4 < C::kSlice / kV; ++c4)
        if (!dry || d.tid < C::DryRows)
            *reinterpret_cast<float4*>(
                staging + swizzled_token_index<C::kSlice>(d.tid, kV * c4)) =
                make_float4(acc[kV * c4], acc[kV * c4 + 1], acc[kV * c4 + 2],
                            acc[kV * c4 + kV - 1]);
    cutlass::arch::fence_view_async_shared();
    oproj_pair_wg_sync(d);
    if (d.tid == 0 && !dry) {
        bulk_copy_to_peer(
            cute::set_block_rank(oproj_pair_ring_smem(d, unit),
                                 (uint32_t)oproj_pair_partner(d)),
            oproj_pair_ring_smem(d, C::Units + unit), C::ExchangeBytes,
            oproj_pair_peer(&d.smem.rx_full[unit], oproj_pair_partner(d)));
        cute::tma_store_arrive();
    }
}

template <class C>
CUTE_DEVICE void oproj_pair_add_partner(const OprojPairDrain<C>& d, int unit,
                                        float (&acc)[C::kSlice]) {
    constexpr int kV = C::kVecFloats;
    const float* received = oproj_pair_ring(d) + unit * C::UnitFloats;
    CUTLASS_PRAGMA_UNROLL
    for (int c4 = 0; c4 < C::kSlice / kV; ++c4) {
        const float4 other = *reinterpret_cast<const float4*>(
            received + swizzled_token_index<C::kSlice>(d.tid, kV * c4));
        acc[kV * c4] += other.x;
        acc[kV * c4 + 1] += other.y;
        acc[kV * c4 + 2] += other.z;
        acc[kV * c4 + kV - 1] += other.w;
    }
}

template <class C, class Prm>
CUTE_DEVICE void oproj_pair_own_slice(const OprojPairDrain<C>& d,
                                      const Prm& params, int slice, int unit,
                                      bool dry) {
    float acc[C::kSlice];
    oproj_pair_load_acc(d, slice, dry, acc);
    if (!dry) cutlass::arch::ClusterBarrier::wait(&d.smem.rx_full[unit], 0u);
    oproj_pair_add_partner(d, unit, acc);
    const int token_lo = d.coord.group_b * C::TT;
    const int token_hi = oproj_pair_token_end<C>(params, d.coord.group_b);
    const int col = d.coord.tile_b * C::kPairRows +
                    d.coord.half * OutputProjLayout::RowTile + d.tid;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < C::kSlice; ++i) {
        const int row = token_lo + C::kSlice * slice + i;
        if (row < token_hi && !dry)
            reinterpret_cast<__nv_bfloat16*>(
                params.out)[(int64_t)row * params.shape.hidden + col] =
                __float2bfloat16(acc[i]);
    }
}

template <class C>
CUTE_DEVICE void oproj_pair_ring_handoff(const OprojPairDrain<C>& d) {
    if (oproj_pair_leader(d))
        for (int unit = 0; unit < C::Units; ++unit)
            cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(
                &d.smem.rx_full[unit], C::ExchangeBytes);
    oproj_pair_both_sync();
    cutlass::arch::ClusterBarrier::wait(&d.smem.acc_full, 1u);
    tcgen05_after_thread_sync();
    if (oproj_pair_leader(d))
        mbar_arrive_cluster(
            oproj_pair_peer(&d.smem.rx_free, oproj_pair_partner(d)));
    if (oproj_pair_leader(d))
        cutlass::arch::ClusterBarrier::wait(&d.smem.rx_free, 0u);
    oproj_pair_both_sync();
}

template <class C, class Prm>
CUTE_DEVICE void oproj_pair_reduce_wo_b(const OprojPairDrain<C>& d,
                                        const Prm& params) {
    const int n_units = (C::Units - d.wg + 1) / 2;
#pragma unroll 1
    for (int it = -1; it < 2 * n_units; ++it) {
        const bool dry = it < 0;
        const int unit_send = dry ? 0 : 2 * it + d.wg;
        const int unit_own = dry ? 0 : 2 * (it - n_units) + d.wg;
        if (dry || it < n_units)
            oproj_pair_send_slice(d, 2 * unit_send + (1 - d.coord.pair),
                                  unit_send, dry);
        if (dry || it >= n_units)
            oproj_pair_own_slice(d, params, 2 * unit_own + d.coord.pair,
                                 unit_own, dry);
        if (dry) oproj_pair_ring_handoff(d);
    }
    if (d.tid == 0) bulk_wait_all();
}

/// Pair UMMA output projection for more than 128 tokens: each CTA pair runs
/// the wo_a GEMM of one 256-row tile with 2-SM UMMA and quantizes it to fp8,
/// then the wo_b GEMM of one K half, and the two pairs of a cluster add
/// their wo_b partials through DSMEM.
template <class C, class Prm>
__global__ void __launch_bounds__(KernelBase::NumThreads, 1)
    mla_output_proj_pair_kernel(__grid_constant__ const Prm params) {
    using Roles = KernelBase::Roles;
    extern __shared__ __align__(Sm100::Sw128AtomBytes) char smem_raw[];
    auto& smem = *reinterpret_cast<typename C::Smem*>(smem_raw);
    const int warp = KernelBase::warp_idx();
    const int lane = KernelBase::lane_idx();
    const OprojPairCoord<C> coord(params.shape);

    if (warp == 0) {
        if (cute::elect_one_sync()) {
            for (int i = 0; i < C::Stages; ++i) {
                cutlass::arch::ClusterTransactionBarrier::init(&smem.tma_a[i],
                                                               1);
                cutlass::arch::ClusterTransactionBarrier::init(&smem.tma_b[i],
                                                               1);
                cutlass::arch::ClusterBarrier::init(&smem.smem_free[i], 1);
            }
            cutlass::arch::ClusterBarrier::init(&smem.acc_full, 1);
            cutlass::arch::ClusterBarrier::init(
                &smem.acc_empty, 2 * 2 * KernelBase::ThreadsPerWarpGroup);
            for (int i = 0; i < C::Slices / 2; ++i)
                cutlass::arch::ClusterTransactionBarrier::init(&smem.rx_full[i],
                                                               1);
            cutlass::arch::ClusterBarrier::init(&smem.rx_free, 1);
            cutlass::arch::fence_barrier_init();
        }
        __syncwarp();
        cute::TMEM::Allocator2Sm().allocate(Sm100::TmemColumns, &smem.tmem);
    }
    cute::cluster_sync();
    pdl_trigger();

    if (warp == Roles::Tma1Warp) {
        oproj_pair_weight_role<C>(params, smem, coord, lane);
    } else if (warp == Roles::Tma2Warp) {
        oproj_pair_act_role<C>(params, smem, coord, lane);
    } else if (warp == Roles::MmaWarp && coord.half == 0) {
        oproj_pair_mma_role<C>(params, smem, coord);
    } else if (warp >= Roles::DrainWarpBegin) {
        const int wg = (warp - Roles::DrainWarpBegin) / Roles::DrainWarps;
        const int tid = threadIdx.x -
                        Roles::DrainWarpBegin * KernelBase::WarpSize -
                        KernelBase::ThreadsPerWarpGroup * wg;
        pdl_wait();
        const OprojPairDrain<C> drain{
            .smem = smem,
            .coord = coord,
            .wg = wg,
            .tid = tid,
            .tmem_base =
                smem.tmem +
                ((uint32_t)(KernelBase::WarpSize * (tid / KernelBase::WarpSize))
                 << C::kTmemLaneShift),
            .transpose = smem.transpose[wg]};
        oproj_pair_quant_wo_a(drain, params);
        oproj_pair_reduce_wo_b(drain, params);
    }

    __syncwarp();
    cute::cluster_sync();
    if (warp == 0) {
        cute::TMEM::Allocator2Sm().release_allocation_lock();
        cute::TMEM::Allocator2Sm().free(smem.tmem, Sm100::TmemColumns);
    }
}

}  // namespace halops::mla
