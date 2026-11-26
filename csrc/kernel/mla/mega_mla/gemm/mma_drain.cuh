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

#include "kernel/mla/mega_mla/gemm/tma_ops.cuh"

namespace halops::mla {
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
    static constexpr int StoreVecWords = 8;
    static constexpr int StoreVecBytes = StoreVecWords * (int)sizeof(uint32_t);

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
            constexpr int kPer = StoreVecBytes / (int)sizeof(PartialT);
            PartialT* out_ptr =
                output + (int64_t)a_row * stride + b_row + nt * DrainN;
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < cute::size(thr_racc); i += kPer) {
                uint32_t words[StoreVecWords];
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

    cute::ThrCopy thr_t2r = tiled_t2r.get_slice(
        (warp % Roles::DrainWarps) * KernelBase::WarpSize + lane);
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
}  // namespace halops::mla
