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

#include "kernel/mla/mla_gemm.cuh"

namespace halops::mla::pair_umma {

constexpr int kBlockK = Sm100::Sw128Bytes / sizeof(cutlass::float_e4m3_t);
constexpr int kPairRows = 2 * OutputProjLayout::RowTile;
constexpr int kSlice = 32;

template <int TN, int Stages>
using Cfg = MlaGemmConfigT<TN, Stages, 1, kPairRows, kBlockK, 0, true>;

template <int TT, int NH>
struct Shape {
    static constexpr int TN = TT / NH;
    static constexpr int Slices = (TT / kSlice + 1) & ~1;
    static constexpr int TransposeRow =
        OutputProjLayout::RowTile + sizeof(uint4) / sizeof(__nv_bfloat16);
    static constexpr int ExchangeBytes =
        OutputProjLayout::RowTile * kSlice * (int)sizeof(float);
    static constexpr int Reserve =
        2 * kSlice * TransposeRow * (int)sizeof(__nv_bfloat16) + 1024;
    static constexpr int SfRows = cute::round_up(TN, SfLayout::MnPerTile);
    static constexpr int StageBytes =
        OutputProjLayout::RowTile * kBlockK + NH * (TN / 2) * kBlockK +
        (OutputProjLayout::RowTile + NH * SfRows) * kBlockK / SfLayout::Gran;
    static constexpr int StageSlackBytes = 512;
    static constexpr int MaxStages = 8;
    static constexpr int FitStages =
        (Sm100::SmemPerCtaMax - Reserve) / (StageBytes + StageSlackBytes);
    static constexpr int Stages = FitStages > MaxStages ? MaxStages : FitStages;
    using Tr = GemmKernelTraits<cutlass::float_e4m3_t, cutlass::float_e4m3_t,
                                cutlass::bfloat16_t, cutlass::bfloat16_t,
                                cutlass::float_ue8m0_t, float, Cfg<TN, Stages>>;
    using TrB =
        GemmKernelTraits<cutlass::float_e4m3_t, cutlass::float_e4m3_t,
                         cutlass::bfloat16_t, cutlass::bfloat16_t,
                         cutlass::float_ue8m0_t, float, Cfg<TN, Stages * NH>>;
    using ML = CollectiveMainloop<Tr>;
};

template <int TT, int NH>
CUTE_HOST auto act_sf_layout(int rows, int t_pad_sf, int k) {
    return Shape<TT, NH>::Tr::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
        cute::make_shape(rows, t_pad_sf, k, 1));
}

template <int TT, int NH>
struct Smem {
    using Sh = Shape<TT, NH>;
    using Tr = typename Sh::Tr;
    using TrB = typename Sh::TrB;
    alignas(Sm100::Sw128AtomBytes) cute::ArrayEngine<
        cutlass::float_e4m3_t, cute::cosize_v<typename Tr::SmemLayoutA>> A;
    alignas(Sm100::Sw128AtomBytes) cute::ArrayEngine<
        cutlass::float_e4m3_t, cute::cosize_v<typename TrB::SmemLayoutB>> B;
    alignas(128) cute::ArrayEngine<
        cutlass::float_ue8m0_t, cute::cosize_v<typename Tr::SmemLayoutSFA>> SFA;
    alignas(
        128) cute::ArrayEngine<cutlass::float_ue8m0_t,
                               cute::cosize_v<typename TrB::SmemLayoutSFB>> SFB;
    alignas(128) __nv_bfloat16 transpose[2][kSlice * Sh::TransposeRow];
    alignas(8) uint64_t tma_a[Sh::Stages], tma_b[Sh::Stages],
        smem_free[Sh::Stages];
    alignas(8) uint64_t acc_full, acc_empty, rx_full[Sh::Slices / 2], rx_free;
    uint32_t tmem;
};

template <int TT, int NH>
constexpr size_t kSmemLaunchBytes =
    sizeof(Smem<TT, NH>) + Sm100::Sw128AtomBytes;

struct Coord {
    int rank, cluster, pair, half;
    int group_a, tile_a;
    int group_b, tile_b;
    CUTE_DEVICE explicit Coord(const OutputProjShape& shape) {
        rank = cute::block_rank_in_cluster();
        cluster = blockIdx.x / OutputProjLayout::ClusterSize;
        pair = rank >> 1;
        half = rank & 1;
        const int item = 2 * cluster + pair;
        const int tiles_a = shape.olora / kPairRows;
        const int tiles_b = shape.hidden / kPairRows;
        group_a = item / tiles_a;
        tile_a = item % tiles_a;
        group_b = cluster / tiles_b;
        tile_b = cluster % tiles_b;
    }
    CUTE_DEVICE int quant_tile() const { return 2 * tile_a + half; }
};

CUTE_HOST_DEVICE int wo_a_k_tiles(const OutputProjShape& shape) {
    return shape.group_k / kBlockK;
}
CUTE_HOST_DEVICE int wo_b_k_tiles(const OutputProjShape& shape) {
    return shape.olora / kBlockK / 2;
}

template <int TT, int NH>
struct Tma {
    using Tr = typename Shape<TT, NH>::Tr;
    using TrB = typename Shape<TT, NH>::TrB;
    CUTE_DEVICE static auto smem_views(Smem<TT, NH>& smem) {
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
    template <class MP>
    CUTE_DEVICE static auto parts(const MP& mp, Smem<TT, NH>& smem, int half) {
        auto cluster_layout = cute::tiled_divide(
            cute::make_layout(typename Tr::ClusterShape_MNK{}),
            cute::make_tile(typename Tr::TiledMma::AtomThrID{}));
        auto cluster_layout_sf = cute::tiled_divide(
            cute::make_layout(typename Tr::ClusterShape_MNK{}),
            cute::make_tile(typename Tr::TiledMma_SF::AtomThrID{}));
        const auto cta = cluster_layout.get_flat_coord(half);
        const auto cta_sf = cluster_layout_sf.get_flat_coord(half);
        auto [sA, sB, sSFA, sSFB] = smem_views(smem);
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
            cute::group_modes<0, 3>(sA),
            cute::group_modes<0, 3>(mma.partition_A(gA)));
        auto [tgB, tsB] = cute::tma_partition(
            mp.tma_load_B, cute::get<1>(cta),
            cute::make_layout(cute::size<1>(cluster_layout)),
            cute::group_modes<0, 3>(sB),
            cute::group_modes<0, 3>(mma.partition_B(gB)));
        auto mSFA = mp.tma_load_SFA.get_tma_tensor(mp.layout_SFA.shape());
        auto mSFB = mp.tma_load_SFB.get_tma_tensor(mp.layout_SFB.shape())(
            cute::_, cute::_, cute::Int<0>{});
        auto mma_sf = typename Tr::TiledMma_SF{}.get_slice(
            half % cute::size(typename Tr::TiledMma_SF::AtomThrID{}));
        auto gSFA =
            cute::local_tile(mSFA, typename Tr::TileShape{},
                             cute::make_coord(cute::_, cute::_, cute::_),
                             cute::Step<cute::_1, cute::X, cute::_1>{});
        auto gSFB =
            cute::local_tile(mSFB, typename Tr::TileShape_SF{},
                             cute::make_coord(cute::_, cute::_, cute::_),
                             cute::Step<cute::X, cute::_1, cute::_1>{});
        auto [tgSFA, tsSFA] = cute::tma_partition(
            mp.tma_load_SFA, cute::get<2>(cta),
            cute::make_layout(cute::size<2>(cluster_layout)),
            cute::group_modes<0, 3>(sSFA),
            cute::group_modes<0, 3>(mma.partition_A(gSFA)));
        auto [tgSFB, tsSFB] = cute::tma_partition(
            mp.tma_load_SFB, cute::get<1>(cta_sf),
            cute::make_layout(cute::size<1>(cluster_layout_sf)),
            cute::group_modes<0, 3>(sSFB),
            cute::group_modes<0, 3>(mma_sf.partition_B(gSFB)));
        return cute::make_tuple(tgA, tsA, tgB, tsB, tgSFA, tsSFA, tgSFB, tsSFB);
    }
    template <int Mode, class AtomThr>
    CUTE_DEVICE static uint16_t pair_mask(int rank) {
        auto cluster_layout = cute::tiled_divide(
            cute::make_layout(typename Tr::ClusterShape_MNK{}),
            cute::make_tile(AtomThr{}));
        return (uint16_t)(cute::create_tma_multicast_mask<Mode>(
                              cluster_layout,
                              cluster_layout.get_flat_coord(rank & 1))
                          << (rank & ~1));
    }
    CUTE_DEVICE static uint16_t mask_a(int rank) {
        return pair_mask<2, typename Tr::TiledMma::AtomThrID>(rank);
    }
    CUTE_DEVICE static uint16_t mask_b(int rank) {
        return pair_mask<1, typename Tr::TiledMma::AtomThrID>(rank);
    }
    CUTE_DEVICE static uint16_t mask_sfb(int rank) {
        return pair_mask<1, typename Tr::TiledMma_SF::AtomThrID>(rank);
    }
};

template <int TT, int NH>
CUTE_DEVICE void wait_slot_free(Smem<TT, NH>& smem, int seq) {
    using Sh = Shape<TT, NH>;
    if (seq >= Sh::Stages)
        cutlass::arch::ClusterBarrier::wait(
            &smem.smem_free[seq % Sh::Stages],
            (uint32_t)(((seq / Sh::Stages) - 1) & 1));
}

template <int TT, int NH, class Prm>
CUTE_DEVICE void weight_role(const Prm& params, Smem<TT, NH>& smem,
                             const Coord& coord, int lane) {
    using Sh = Shape<TT, NH>;
    using ML = typename Sh::ML;
    constexpr uint32_t kBytes =
        ML::TmaTransactionBytesA + ML::TmaTransactionBytesSFA;
    auto parts_a = Tma<TT, NH>::parts(params.wo_a, smem, coord.half);
    auto parts_b = Tma<TT, NH>::parts(params.wo_b, smem, coord.half);
    const uint16_t mask = Tma<TT, NH>::mask_a(coord.rank);
    auto issue = [&](auto const& mp, auto& parts, int seq, int tile,
                     int k_tile) {
        const int stage = seq % Sh::Stages;
        wait_slot_free(smem, seq);
        if (lane == 0) {
            if (coord.half == 0)
                cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(
                    &smem.tma_a[stage], kBytes);
            cute::copy(mp.tma_load_A.with(smem.tma_a[stage], mask),
                       cute::get<0>(parts)(cute::_, tile, k_tile, 0),
                       cute::get<1>(parts)(cute::_, (cute::Int<0>{}, stage)));
            cute::copy(mp.tma_load_SFA.with(smem.tma_a[stage], mask),
                       cute::get<4>(parts)(cute::_, tile, k_tile, 0),
                       cute::get<5>(parts)(cute::_, (cute::Int<0>{}, stage)));
        }
        __syncwarp();
    };
    const int k_a = wo_a_k_tiles(params.shape),
              k_b = wo_b_k_tiles(params.shape);
#pragma unroll 1
    for (int k = 0; k < k_a; ++k)
        issue(params.wo_a, parts_a, k, coord.tile_a, k);
#pragma unroll 1
    for (int k = 0; k < k_b; ++k)
        issue(params.wo_b, parts_b, k_a + k, coord.tile_b,
              k_b * coord.pair + k);
}

template <int TT, int NH, class Prm>
CUTE_DEVICE void act_role(const Prm& params, Smem<TT, NH>& smem,
                          const Coord& coord, int lane) {
    using Sh = Shape<TT, NH>;
    using ML = typename Sh::ML;
    constexpr uint32_t kBytes =
        NH * (ML::TmaTransactionBytesB + ML::TmaTransactionBytesSFB);
    constexpr int kTilesPerK = kBlockK / OutputProjLayout::RowTile;
    auto parts_a = Tma<TT, NH>::parts(params.wo_a, smem, coord.half);
    auto parts_b = Tma<TT, NH>::parts(params.wo_b, smem, coord.half);
    const uint16_t mask = Tma<TT, NH>::mask_b(coord.rank);
    const uint16_t mask_sf = Tma<TT, NH>::mask_sfb(coord.rank);
    pdl_wait();
    auto issue = [&](auto const& mp, auto& parts, int seq, int group,
                     int k_tile) {
        const int stage = seq % Sh::Stages;
        wait_slot_free(smem, seq);
        if (lane == 0) {
            if (coord.half == 0)
                cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(
                    &smem.tma_b[stage], kBytes);
            CUTLASS_PRAGMA_UNROLL
            for (int part = 0; part < NH; ++part) {
                cute::copy(
                    mp.tma_load_B.with(smem.tma_b[stage], mask),
                    cute::get<2>(parts)(cute::_, NH * group + part, k_tile),
                    cute::get<3>(parts)(cute::_,
                                        (cute::Int<0>{}, NH * stage + part)));
                cute::copy(
                    mp.tma_load_SFB.with(smem.tma_b[stage], mask_sf),
                    cute::get<6>(parts)(cute::_, NH * group + part, k_tile),
                    cute::get<7>(parts)(cute::_,
                                        (cute::Int<0>{}, NH * stage + part)));
            }
        }
        __syncwarp();
    };
    const int k_a = wo_a_k_tiles(params.shape),
              k_b = wo_b_k_tiles(params.shape);
    const int out_group = coord.tile_a / (params.shape.group_rows / kPairRows);
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
                0xffffffffu, lane < half_tiles && gmem_ld_acquire(flag) >= 1);
        __syncwarp();
        fence_proxy_async_global();
        issue(params.wo_b, parts_b, k_a + k, coord.group_b,
              k_b * coord.pair + k);
    }
}

template <class UtccpOp, class SmemSf, class TmemSf>
CUTE_DEVICE auto make_sf_copy(const SmemSf& smem_sf, const TmemSf& tmem_sf) {
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
CUTE_DEVICE void copy_sf(const SfCopy& sf, int stage) {
    cute::copy(cute::get<0>(sf),
               cute::get<1>(sf)(cute::_, cute::_, cute::_, cute::_, stage),
               cute::get<2>(sf)(cute::_, cute::_, cute::_, cute::_));
}

template <int KIters, class TiledMma, class Tmem, class FragA, class FragB>
CUTE_DEVICE void mma_k_tile(TiledMma& tiled_mma, const Tmem& tmem,
                            const FragA& rA, const FragB& rB, int stage_a,
                            int stage_b, int acc_col, bool first) {
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

template <int TT, int NH, class Prm>
CUTE_DEVICE void mma_role(const Prm& params, Smem<TT, NH>& smem,
                          const Coord& coord) {
    using Sh = Shape<TT, NH>;
    using Tr = typename Sh::Tr;
    using ML = typename Sh::ML;
    const uint16_t pair_mask = (uint16_t)(3u << (coord.rank & ~1));
    const auto views = Tma<TT, NH>::smem_views(smem);
    typename Tr::TiledMma tiled_mma;
    auto mma = tiled_mma.get_slice(0);
    auto rA = mma.make_fragment_A(cute::get<0>(views));
    auto rB = mma.make_fragment_B(cute::get<1>(views));
    auto tmem = ML::init_tmem_tensors();
    ML::set_tmem_offsets(tmem, smem.tmem);
    tmem.tmem_SFA.data() = tmem.tmem_SFA.data().get() + (NH - 1) * Sh::TN;
    tmem.tmem_SFB.data() = tmem.tmem_SFB.data().get() + (NH - 1) * Sh::TN;
    const auto sfa =
        make_sf_copy<typename ML::UtccpOp>(cute::get<2>(views), tmem.tmem_SFA);
    const auto sfb =
        make_sf_copy<typename ML::UtccpOp>(cute::get<3>(views), tmem.tmem_SFB);
    const bool elected = cute::elect_one_sync();
    auto run = [&](int seq0, int k_tiles) {
#pragma unroll 1
        for (int k = 0; k < k_tiles; ++k) {
            const int seq = seq0 + k, stage = seq % Sh::Stages;
            const uint32_t phase = (uint32_t)((seq / Sh::Stages) & 1);
            cutlass::arch::ClusterBarrier::wait(&smem.tma_a[stage], phase);
            cutlass::arch::ClusterBarrier::wait(&smem.tma_b[stage], phase);
            tcgen05_after_thread_sync();
            if (elected) copy_sf(sfa, stage);
            CUTLASS_PRAGMA_UNROLL
            for (int part = 0; part < NH; ++part) {
                if (elected) copy_sf(sfb, NH * stage + part);
                mma_k_tile<Tr::kItersK>(tiled_mma, tmem, rA, rB, stage,
                                        NH * stage + part, part * Sh::TN,
                                        k == 0);
            }
            cutlass::arch::umma_arrive_multicast_2x1SM(&smem.smem_free[stage],
                                                       pair_mask);
        }
        cutlass::arch::umma_arrive_multicast_2x1SM(&smem.acc_full, pair_mask);
    };
    const int k_a = wo_a_k_tiles(params.shape);
    run(0, k_a);
    cutlass::arch::ClusterBarrier::wait(&smem.acc_empty, 0u);
    tcgen05_after_thread_sync();
    run(k_a, wo_b_k_tiles(params.shape));
}

CUTE_DEVICE void tmem_load_slice(uint32_t taddr, float (&vals)[kSlice]) {
    tmem_ld_32dp32bNx<kSlice>(taddr, vals);
    cutlass::arch::fence_view_async_tmem_load();
}

template <int TT, int NH>
struct Drain {
    using Sh = Shape<TT, NH>;
    static_assert(OutputProjLayout::RowTile == KernelBase::ThreadsPerWarpGroup,
                  "one weight row per thread");
    static constexpr int GroupsPerToken =
        OutputProjLayout::RowTile / SfLayout::Gran;
    static constexpr int BfPerVec = sizeof(uint4) / sizeof(__nv_bfloat16);
    static constexpr int GroupWords = SfLayout::Gran / sizeof(uint32_t);
    static_assert(kSlice * GroupsPerToken == KernelBase::ThreadsPerWarpGroup,
                  "one task per thread");
    static constexpr int Units = Sh::Slices / 2;
    static constexpr int UnitFloats = OutputProjLayout::RowTile * kSlice;
    static constexpr int DryRows = 8;
    static_assert(DryRows * kSlice * sizeof(float) <=
                      sizeof(Smem<TT, NH>::transpose[0]),
                  "dry staging fits the transpose");
    Smem<TT, NH>& smem;
    const Coord& coord;
    int wg, tid;
    uint32_t tmem_base;
    __nv_bfloat16* transpose;

    CUTE_DEVICE void wg_sync() const {
        using Bars = KernelBase::Bars;
        static_assert(Bars::PairWg0Sync::width == Bars::PairWg1Sync::width,
                      "both drain WGs are one warp group");
        cutlass::arch::NamedBarrier::arrive_and_wait(
            Bars::PairWg0Sync::width,
            wg == 0 ? Bars::PairWg0Sync::id : Bars::PairWg1Sync::id);
    }
    CUTE_DEVICE static void both_sync() {
        bar_sync<KernelBase::Bars::PairBothSync>();
    }
    CUTE_DEVICE bool leader() const { return tid == 0 && wg == 0; }
    CUTE_DEVICE uint32_t peer(const void* ptr, int rank) const {
        return cute::set_block_rank(cute::cast_smem_ptr_to_uint(ptr),
                                    (uint32_t)rank);
    }
    template <class Prm>
    CUTE_DEVICE static int token_end(const Prm& params, int group) {
        return min(params.num_tokens, group * TT + TT);
    }
    CUTE_DEVICE void load_acc(int slice, bool dry, float (&acc)[kSlice]) const {
        if (!dry) {
            tmem_load_slice(tmem_base + kSlice * slice, acc);
        } else {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < kSlice; ++i) acc[i] = 0.0f;
        }
    }

    CUTE_DEVICE int quant_token() const { return tid / GroupsPerToken; }
    CUTE_DEVICE int quant_quarter() const { return tid % GroupsPerToken; }

    CUTE_DEVICE void transpose_slice(const float (&acc)[kSlice],
                                     bool first) const {
        if (!first) wg_sync();
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kSlice; ++i)
            transpose[i * Sh::TransposeRow + tid] = __float2bfloat16(acc[i]);
        wg_sync();
    }

    CUTE_DEVICE void load_group(float (&vals)[SfLayout::Gran]) const {
        const __nv_bfloat16* group = transpose +
                                     quant_token() * Sh::TransposeRow +
                                     SfLayout::Gran * quant_quarter();
        CUTLASS_PRAGMA_UNROLL
        for (int vec = 0; vec < SfLayout::Gran / BfPerVec; ++vec) {
            const uint4 word =
                *reinterpret_cast<const uint4*>(group + BfPerVec * vec);
            float part[BfPerVec];
            unpack_bf16(&word, part);
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < BfPerVec; ++i)
                vals[BfPerVec * vec + i] = part[i];
        }
    }

    template <class Prm, class SfTensor>
    CUTE_DEVICE void store_group(const Prm& params, SfTensor& sf, int row,
                                 const uint32_t (&packed)[GroupWords],
                                 float biased_exp) const {
        const int col = coord.quant_tile() * OutputProjLayout::RowTile +
                        SfLayout::Gran * quant_quarter();
        uint4* dst = reinterpret_cast<uint4*>(
            reinterpret_cast<uint8_t*>(params.olora_fp8) +
            (int64_t)row * params.shape.olora + col);
        dst[0] = make_uint4(packed[0], packed[1], packed[2], packed[3]);
        dst[1] = make_uint4(packed[4], packed[5], packed[6], packed[7]);
        sf(sf_row_grouped(row, Sh::TN, 2 * NH), col, cute::_0{}) =
            (uint8_t)biased_exp;
    }

    template <class Prm>
    CUTE_DEVICE void quant_wo_a(const Prm& params) {
        const int token_lo = coord.group_a * TT;
        const int token_hi = token_end(params, coord.group_a);
        cute::Tensor sf = cute::make_tensor(
            cute::make_gmem_ptr(params.olora_sf), params.wo_b.layout_SFB);
#pragma unroll 1
        for (int it = -1; it < Sh::Slices / 2; ++it) {
            const bool dry = it < 0;
            const int slice = dry ? 0 : 2 * it + wg;
            float acc[kSlice];
            load_acc(slice, dry, acc);
            if (it == Sh::Slices / 2 - 1) {
                tcgen05_before_thread_sync();
                mbar_arrive_cluster_relaxed(
                    peer(&smem.acc_empty, coord.rank & ~1));
            }
            transpose_slice(acc, it <= 0);
            float vals[SfLayout::Gran];
            load_group(vals);
            const Ue8m0Scale scale = ue8m0_scale(abs_max(vals));
            uint32_t packed[GroupWords];
            pack_e4m3(vals, scale.multiplier, packed);
            const int row = token_lo + kSlice * slice + quant_token();
            if (row < token_hi && !dry)
                store_group(params, sf, row, packed, scale.biased_exp);
            if (dry) {
                wg_sync();
                cutlass::arch::ClusterBarrier::wait(&smem.acc_full, 0u);
                tcgen05_after_thread_sync();
            }
        }
        __threadfence();
        fence_proxy_async_global();
        both_sync();
        if (leader())
            gmem_red_add1_relaxed(params.flags +
                                  params.shape.wo_a_tiles() * coord.group_a +
                                  coord.quant_tile());
    }

    CUTE_DEVICE float* ring() const {
        return reinterpret_cast<float*>(smem.A.begin());
    }
    CUTE_DEVICE uint32_t ring_smem(int unit) const {
        return cute::cast_smem_ptr_to_uint(ring()) +
               (uint32_t)(unit * Sh::ExchangeBytes);
    }
    CUTE_DEVICE int partner() const { return coord.rank ^ 2; }

    CUTE_DEVICE void send_slice(int slice, int unit, bool dry) {
        float acc[kSlice];
        load_acc(slice, dry, acc);
        float* staging = dry ? reinterpret_cast<float*>(transpose)
                             : ring() + (Units + unit) * UnitFloats;
        CUTLASS_PRAGMA_UNROLL
        for (int c4 = 0; c4 < kSlice / 4; ++c4)
            if (!dry || tid < DryRows)
                *reinterpret_cast<float4*>(
                    staging + swizzled_token_index<kSlice>(tid, 4 * c4)) =
                    make_float4(acc[4 * c4], acc[4 * c4 + 1], acc[4 * c4 + 2],
                                acc[4 * c4 + 3]);
        cutlass::arch::fence_view_async_shared();
        wg_sync();
        if (tid == 0 && !dry) {
            bulk_copy_to_peer(
                cute::set_block_rank(ring_smem(unit), (uint32_t)partner()),
                ring_smem(Units + unit), Sh::ExchangeBytes,
                peer(&smem.rx_full[unit], partner()));
            cute::tma_store_arrive();
        }
    }

    CUTE_DEVICE void add_partner(int unit, float (&acc)[kSlice]) const {
        const float* received = ring() + unit * UnitFloats;
        CUTLASS_PRAGMA_UNROLL
        for (int c4 = 0; c4 < kSlice / 4; ++c4) {
            const float4 other = *reinterpret_cast<const float4*>(
                received + swizzled_token_index<kSlice>(tid, 4 * c4));
            acc[4 * c4] += other.x;
            acc[4 * c4 + 1] += other.y;
            acc[4 * c4 + 2] += other.z;
            acc[4 * c4 + 3] += other.w;
        }
    }

    template <class Prm>
    CUTE_DEVICE void own_slice(const Prm& params, int slice, int unit,
                               bool dry) {
        float acc[kSlice];
        load_acc(slice, dry, acc);
        if (!dry) cutlass::arch::ClusterBarrier::wait(&smem.rx_full[unit], 0u);
        add_partner(unit, acc);
        const int token_lo = coord.group_b * TT;
        const int token_hi = token_end(params, coord.group_b);
        const int col = coord.tile_b * kPairRows +
                        coord.half * OutputProjLayout::RowTile + tid;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kSlice; ++i) {
            const int row = token_lo + kSlice * slice + i;
            if (row < token_hi && !dry)
                reinterpret_cast<__nv_bfloat16*>(
                    params.out)[(int64_t)row * params.shape.hidden + col] =
                    __float2bfloat16(acc[i]);
        }
    }

    CUTE_DEVICE void ring_handoff() {
        if (leader())
            for (int unit = 0; unit < Units; ++unit)
                cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(
                    &smem.rx_full[unit], Sh::ExchangeBytes);
        both_sync();
        cutlass::arch::ClusterBarrier::wait(&smem.acc_full, 1u);
        tcgen05_after_thread_sync();
        if (leader()) mbar_arrive_cluster(peer(&smem.rx_free, partner()));
        if (leader()) cutlass::arch::ClusterBarrier::wait(&smem.rx_free, 0u);
        both_sync();
    }

    template <class Prm>
    CUTE_DEVICE void reduce_wo_b(const Prm& params) {
        const int n_units = (Units - wg + 1) / 2;
#pragma unroll 1
        for (int it = -1; it < 2 * n_units; ++it) {
            const bool dry = it < 0;
            const int unit_send = dry ? 0 : 2 * it + wg;
            const int unit_own = dry ? 0 : 2 * (it - n_units) + wg;
            if (dry || it < n_units)
                send_slice(2 * unit_send + (1 - coord.pair), unit_send, dry);
            if (dry || it >= n_units)
                own_slice(params, 2 * unit_own + coord.pair, unit_own, dry);
            if (dry) ring_handoff();
        }
        if (tid == 0) bulk_wait_all();
    }
};

template <int TT, int NH, class Prm>
__global__ void __launch_bounds__(KernelBase::NumThreads, 1)
    mla_output_proj_pair_kernel(__grid_constant__ const Prm params) {
    using Sh = Shape<TT, NH>;
    using Roles = KernelBase::Roles;
    extern __shared__ __align__(1024) char smem_raw[];
    auto& smem = *reinterpret_cast<Smem<TT, NH>*>(smem_raw);
    const int warp = KernelBase::warp_idx();
    const int lane = KernelBase::lane_idx();
    const Coord coord(params.shape);

    if (warp == 0) {
        if (cute::elect_one_sync()) {
            for (int i = 0; i < Sh::Stages; ++i) {
                cutlass::arch::ClusterTransactionBarrier::init(&smem.tma_a[i],
                                                               1);
                cutlass::arch::ClusterTransactionBarrier::init(&smem.tma_b[i],
                                                               1);
                cutlass::arch::ClusterBarrier::init(&smem.smem_free[i], 1);
            }
            cutlass::arch::ClusterBarrier::init(&smem.acc_full, 1);
            cutlass::arch::ClusterBarrier::init(
                &smem.acc_empty, 2 * 2 * KernelBase::ThreadsPerWarpGroup);
            for (int i = 0; i < Sh::Slices / 2; ++i)
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
        weight_role<TT, NH>(params, smem, coord, lane);
    } else if (warp == Roles::Tma2Warp) {
        act_role<TT, NH>(params, smem, coord, lane);
    } else if (warp == Roles::MmaWarp && coord.half == 0) {
        mma_role<TT, NH>(params, smem, coord);
    } else if (warp >= Roles::DrainWarpBegin) {
        const int wg = (warp - Roles::DrainWarpBegin) / Roles::DrainWarps;
        const int tid = threadIdx.x -
                        Roles::DrainWarpBegin * KernelBase::WarpSize -
                        KernelBase::ThreadsPerWarpGroup * wg;
        pdl_wait();
        Drain<TT, NH> drain{
            .smem = smem,
            .coord = coord,
            .wg = wg,
            .tid = tid,
            .tmem_base =
                smem.tmem +
                ((uint32_t)(KernelBase::WarpSize * (tid / KernelBase::WarpSize))
                 << 16),
            .transpose = smem.transpose[wg]};
        drain.quant_wo_a(params);
        drain.reduce_wo_b(params);
    }

    __syncwarp();
    cute::cluster_sync();
    if (warp == 0) {
        cute::TMEM::Allocator2Sm().release_allocation_lock();
        cute::TMEM::Allocator2Sm().free(smem.tmem, Sm100::TmemColumns);
    }
}

}  // namespace halops::mla::pair_umma
