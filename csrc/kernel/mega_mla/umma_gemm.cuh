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

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/numeric_types.h>

#include <cute/tensor.hpp>

#include "kernel/gemm/kernel_traits.cuh"
#include "kernel/gemm/mainloop_fwd.cuh"
#include "mla_intrinsics.cuh"
#include "mla_traits.cuh"

namespace mega_mla {

template <int KBlockN_, int KPrologueStages_ = 1, int KEpilogueStages_ = 1,
          int KBlockM_ = 128, int KBlockK_ = 256>
struct MegaMlaGemmConfigT : halops::GemmKernelConfig {
    static constexpr int KBlockM = KBlockM_;
    static constexpr int KBlockN = KBlockN_;
    static constexpr int KBlockK = KBlockK_;
    static constexpr bool IsPairUmma = false;
    static constexpr int KClusterM = 1;
    static constexpr int KClusterN = 1;

    static constexpr int KPrologueStages = KPrologueStages_;

    static constexpr int KEpilogueStages = KEpilogueStages_;
    static constexpr int KBiasStages = 1;
    static constexpr int KTmemOverlapCols = 0;
    static constexpr bool UseTileSwizzle = false;
    static constexpr halops::KernelMode KernelType = halops::KernelMode::Naive;
    static constexpr bool AddBias = false;
    static constexpr bool SwapAB = true;
    static constexpr bool KIsFP4 = false;
    static constexpr bool HasAlpha = false;
};

template <int N_PAD>
CUTE_HOST_DEVICE constexpr int sf_row_of(int t) {
    static_assert(
        N_PAD >= KernelTrait::Tile::Smallest &&
            N_PAD <= GemmShapeDefaults::SfMnPerTile &&
            (N_PAD & (N_PAD - 1)) == 0,
        "below GemmShapeDefaults::SfPackedMinTile every token group gets its "
        "OWN 128-MN "
        "scale-factor tile, so a group's SF TMEM base is always column "
        "0.  Packing several groups into one tile needs a base of "
        "(g % (128/BN)) * (BN/32) columns, and that is only EVEN -- the "
        "hardware's requirement -- for BN >= 64; BN=32 faulted with "
        "misaligned address and BN=16 truncated to 0 and read group 0's "
        "scale factors.  MEASURED 2026-08-24, knowledge S5.10.  At and "
        "above 64 the packed layout is KEPT: one shared tile per two "
        "groups is L2-hot, and spreading it cost 1-2 us per case");
    if constexpr (N_PAD < GemmShapeDefaults::SfPackedMinTile)
        return (t / N_PAD) * GemmShapeDefaults::SfMnPerTile + (t % N_PAD);
    else
        return t;
}

template <int TileT_, int STAGES = GemmShapeDefaults::TmaStages,
          int ACC_STAGES = 1, int KBlockM = 128, int KBlockK = 256>
struct GemmInst {
    using Config =
        MegaMlaGemmConfigT<TileT_, STAGES, ACC_STAGES, KBlockM, KBlockK>;
    using Traits =
        halops::GemmKernelTraits<e4m3, e4m3, bf16, bf16, ue8m0, float, Config>;
    using Mainloop = halops::CollectiveMainloop<Traits>;

    static constexpr int TileT = TileT_;
    static_assert(Traits::KBlockM == GemmShapeDefaults::BM, "M tile mismatch");
    static_assert(
        Config::SwapAB,
        "MegaMlaGemmConfigT::SwapAB is NOT mega_mla's orientation knob -- that "
        "is GemmTileCfg::SwapAB, read only by mega_mla's own role ops.  Inside "
        "GemmKernelTraits this flag selects nothing but "
        "KEpilogueWarpGroupCount (1 vs 2), which divides KEpiComputeN and so "
        "reshapes EpiComputeTile -- the tile make_drain_op builds its T2R copy "
        "from.  Flipping it here would halve that tile and trip the "
        "one-warp-per-TMEM-quadrant assert.  Everything else mega_mla reads "
        "from the traits is orientation-symmetric at BM == BN.");
    static_assert(Traits::KBlockK == GemmShapeDefaults::BK, "K block mismatch");
    static_assert(Traits::kItersK == GemmShapeDefaults::KItersPerBlock,
                  "K iter count mismatch");

    static constexpr bool SF_TILE_PER_GROUP =
        TileT_ < GemmShapeDefaults::SfPackedMinTile;

    static constexpr int VS_PER_CTA_N = Traits::CTA_N_SF / TileT_;

    static constexpr int SFB_COLS_PER_SUB_SLOT =
        TileT_ / GemmShapeDefaults::SfMnPerCol;

    static_assert(
        Traits::CTA_N_SF == GemmShapeDefaults::SfMnPerTile,
        "one scale-factor tile per token group is the whole reason nb_sf can "
        "be "
        "the group index itself and the SF TMEM base can stay at column 0; "
        "that "
        "identity needs the SF tile to be exactly one 128-MN atom wide");
    static_assert(
        SF_TILE_PER_GROUP || SFB_COLS_PER_SUB_SLOT % 2 == 0,
        "the packed SF layout is only kept where the sub-slot base lands on an "
        "EVEN TMEM column; an odd one is a misaligned address at run time");

    using SmemLayoutAStaged = typename Traits::SmemLayoutA;
    using SmemLayoutBStaged = typename Traits::SmemLayoutB;
    using SmemLayoutSFAStaged = typename Traits::SmemLayoutSFA;
    using SmemLayoutSFBStaged = typename Traits::SmemLayoutSFB;

    static constexpr int SMEM_A =
        cute::cosize_v<typename Traits::SmemLayoutA> * sizeof(e4m3) / STAGES;
    static constexpr int SMEM_B =
        cute::cosize_v<typename Traits::SmemLayoutB> * sizeof(e4m3) / STAGES;
    static constexpr int SMEM_SFA =
        cute::cosize_v<typename Traits::SmemLayoutSFA> * sizeof(ue8m0) / STAGES;
    static constexpr int SMEM_SFB =
        cute::cosize_v<typename Traits::SmemLayoutSFB> * sizeof(ue8m0) / STAGES;

    static constexpr int OFF_STAGES =
        KernelTraits<ModelDims, TileT_>::MaxTmaStages;
    static_assert(
        OFF_STAGES >= STAGES,
        "every range that shares one smem base must place B/SFA/SFB at the "
        "SAME address, or barrier slot s stops naming the same bytes in "
        "consecutive ranges: the next range waits on smem_free[s] and then "
        "overwrites a buffer the previous range still owns (nan, no assert)");

    static constexpr int SMEM_TOTAL =
        OFF_STAGES * (SMEM_A + SMEM_B + SMEM_SFA + SMEM_SFB);

    static constexpr int OFF_A = 0;
    static constexpr int OFF_B = OFF_A + OFF_STAGES * SMEM_A;
    static constexpr int OFF_SFA = OFF_B + OFF_STAGES * SMEM_B;
    static constexpr int OFF_SFB = OFF_SFA + OFF_STAGES * SMEM_SFA;

    static constexpr uint32_t TMA_BYTES_A =
        Mainloop::TmaTransactionBytesA + Mainloop::TmaTransactionBytesSFA;
    static constexpr uint32_t TMA_BYTES_B =
        Mainloop::TmaTransactionBytesB + Mainloop::TmaTransactionBytesSFB;

    static_assert(cute::cosize_v<SmemLayoutAStaged> * (int)sizeof(e4m3) %
                          STAGES ==
                      0,
                  "A staged layout is not dense");
    static_assert(cute::cosize_v<SmemLayoutBStaged> * (int)sizeof(e4m3) %
                          STAGES ==
                      0,
                  "B staged layout is not dense");
    static_assert(cute::cosize_v<SmemLayoutSFAStaged> * (int)sizeof(ue8m0) %
                          STAGES ==
                      0,
                  "SFA staged layout is not dense");
    static_assert(cute::cosize_v<SmemLayoutSFBStaged> * (int)sizeof(ue8m0) %
                          STAGES ==
                      0,
                  "SFB staged layout is not dense");
};

template <class Cfg>
using GemmInstFor =
    GemmInst<Cfg::BN, Cfg::TmaStages, Cfg::AccStages, Cfg::BM, Cfg::BK>;

template <class KT>
CUTE_HOST_DEVICE constexpr int traits_smem_max() {
    return cfg_max4(GemmInstFor<typename KT::Qkv>::SMEM_TOTAL,
                    GemmInstFor<typename KT::WqB>::SMEM_TOTAL,
                    GemmInstFor<typename KT::WoA>::SMEM_TOTAL,
                    GemmInstFor<typename KT::WoB>::SMEM_TOTAL);
}

struct GemmArena {
    static constexpr int Total =
        cfg_max4(traits_smem_max<TraitsN64>(), traits_smem_max<TraitsN128>(),
                 traits_smem_max<TraitsN32>(), traits_smem_max<TraitsN16>());
    static constexpr int MaxTmaStages =
        cfg_max4(TraitsN64::MaxTmaStages, TraitsN128::MaxTmaStages,
                 TraitsN32::MaxTmaStages, TraitsN16::MaxTmaStages);
    static constexpr int MaxAccStages =
        cfg_max4(TraitsN64::MaxAccStages, TraitsN128::MaxAccStages,
                 TraitsN32::MaxAccStages, TraitsN16::MaxAccStages);
    static_assert(MaxTmaStages >= TraitsN128::WoA::TmaStages,
                  "bar_gemm_tma_a is sized from MaxTmaStages; a short array "
                  "hangs, it does not fault");
};

struct GemmShape {
    int m_total;
    int k_dim;
    int m_tiles;
    int k_tiles;
    int k_tps;
};

template <int BM, int BK, int KSplit, int Stages>
CUTE_HOST GemmShape make_gemm_shape(int m_total, int k_dim) {
    TORCH_CHECK(m_total > 0 && m_total % BM == 0,
                "a GEMM's M must be a positive multiple of ", BM, ", got ",
                m_total);
    TORCH_CHECK(k_dim > 0 && k_dim % BK == 0,
                "a GEMM's K must be a positive multiple of ", BK, ", got ",
                k_dim);
    const int k_tiles = k_dim / BK;
    TORCH_CHECK(k_tiles % KSplit == 0,
                "the K-split must divide the k-tile count; got ", k_tiles,
                " tiles for KSplit=", KSplit);
    const int k_tps = k_tiles / KSplit;
    TORCH_CHECK(k_tps >= Stages,
                "an item needs at least one k-tile per pipeline stage, else a "
                "stage is armed zero times and the prologue prefetches a tile "
                "the item never uses; got k_tps=",
                k_tps, " stages=", Stages);
    GemmShape s{};
    s.m_total = m_total;
    s.k_dim = k_dim;
    s.m_tiles = m_total / BM;
    s.k_tiles = k_tiles;
    s.k_tps = k_tps;
    return s;
}

static_assert(
    GemmArena::MaxAccStages * GemmShapeDefaults::BM <=
        TraitsN64::Attn::tmem_cols::P,
    "the staged accumulators must leave room for the SF columns inside TMEM's "
    "512; deepening AccStages past this needs a TMEM re-carve, not just a "
    "config "
    "bump -- an overrun is silent");

template <int TileT_>
struct GemmAtomsA {
    using Inst = GemmInst<TileT_>;
    typename Inst::Mainloop::TMALoad_A tma_a;
    typename Inst::Mainloop::TMALoad_SFA tma_sfa;
    typename Inst::Traits::LayoutSFA layout_sfa;
    int m_total;
    int k_dim;
};

template <int TileT_>
CUTE_HOST GemmAtomsA<TileT_> make_atoms_a(const void* weight,
                                          const void* weight_sf, int M, int K,
                                          int N_PAD = GemmInst<TileT_>::TileT) {
    using Traits = typename GemmInst<TileT_>::Traits;
    using Mainloop = typename GemmInst<TileT_>::Mainloop;

    const int MMA_M = M;
    const int n_pad = N_PAD;

    auto layout_a = cute::make_layout(cute::make_shape(M, K, 1),
                                      cute::make_stride(K, cute::Int<1>{}, 0));
    auto layout_sfa = Traits::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
        cute::make_shape(MMA_M, n_pad, K, 1));

    cute::Tensor tensor_a = cute::make_tensor(
        cute::make_gmem_ptr(static_cast<const e4m3*>(weight)), layout_a);
    cute::Tensor tensor_sfa =
        cute::make_tensor(static_cast<const ue8m0*>(weight_sf), layout_sfa);

    auto cluster_layout_vmnk = cute::tiled_divide(
        cute::make_layout(typename Traits::ClusterShape_MNK{}),
        cute::make_tile(typename Traits::TiledMma::AtomThrID{}));

    GemmAtomsA<TileT_> atoms{
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
        M,
        K,
    };
    return atoms;
}

template <int TileT_>
struct GemmAtomsB {
    using Inst = GemmInst<TileT_>;
    typename Inst::Mainloop::TMALoad_B tma_b;
    typename Inst::Mainloop::TMALoad_SFB tma_sfb;
    typename Inst::Traits::LayoutSFB layout_sfb;
    int n_pad;
    int k_dim;
};

template <int TileT_>
CUTE_HOST GemmAtomsB<TileT_> make_atoms_b(const void* act_fp8,
                                          const void* act_sf, int M, int N_PAD,
                                          int K, int N_PAD_SF) {
    using Traits = typename GemmInst<TileT_>::Traits;
    using Mainloop = typename GemmInst<TileT_>::Mainloop;

    auto layout_b = cute::make_layout(cute::make_shape(N_PAD, K, 1),
                                      cute::make_stride(K, cute::Int<1>{}, 0));
    auto layout_sfb = Traits::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
        cute::make_shape(M, N_PAD_SF, K, 1));

    cute::Tensor tensor_b = cute::make_tensor(
        cute::make_gmem_ptr(static_cast<const e4m3*>(act_fp8)), layout_b);
    cute::Tensor tensor_sfb =
        cute::make_tensor(static_cast<const ue8m0*>(act_sf), layout_sfb);

    auto cluster_layout_vmnk = cute::tiled_divide(
        cute::make_layout(typename Traits::ClusterShape_MNK{}),
        cute::make_tile(typename Traits::TiledMma::AtomThrID{}));
    auto cluster_layout_sfb_vmnk = cute::tiled_divide(
        cute::make_layout(typename Traits::ClusterShape_MNK{}),
        cute::make_tile(typename Traits::TiledMma_SF::AtomThrID{}));

    GemmAtomsB<TileT_> atoms{
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
        N_PAD,
        K,
    };
    return atoms;
}

template <int BN>
CUTE_HOST_DEVICE constexpr bool bn_swap_ab() {
    return KernelTraits<ModelDims, BN>::Qkv::SwapAB;
}

template <int BN>
using WeightAtoms =
    cute::conditional_t<bn_swap_ab<BN>(), GemmAtomsA<BN>, GemmAtomsB<BN>>;
template <int BN>
using ActAtoms =
    cute::conditional_t<bn_swap_ab<BN>(), GemmAtomsB<BN>, GemmAtomsA<BN>>;

template <int BN>
CUTE_HOST WeightAtoms<BN> make_weight_atoms(const void* w, const void* w_sf,
                                            int m_total, int k) {
    if constexpr (bn_swap_ab<BN>())
        return make_atoms_a<BN>(w, w_sf, m_total, k);
    else
        return make_atoms_b<BN>(w, w_sf, m_total, m_total, k, m_total);
}

template <int BN>
CUTE_HOST ActAtoms<BN> make_act_atoms(const void* a, const void* a_sf,
                                      int m_total, int t_pad, int k,
                                      int t_pad_sf) {
    if constexpr (bn_swap_ab<BN>())
        return make_atoms_b<BN>(a, a_sf, m_total, t_pad, k, t_pad_sf);
    else
        return make_atoms_a<BN>(a, a_sf, t_pad, k);
}

template <int BN>
CUTE_HOST_DEVICE auto const& sf_layout(const GemmAtomsA<BN>& at) {
    return at.layout_sfa;
}
template <int BN>
CUTE_HOST_DEVICE auto const& sf_layout(const GemmAtomsB<BN>& at) {
    return at.layout_sfb;
}

template <int N_PAD, class LayoutSFB>
CUTE_DEVICE void quantize_group32_from_regs(const __nv_bfloat16 (&v)[4],
                                            e4m3* __restrict__ ws_fp8,
                                            ue8m0* __restrict__ ws_sf,
                                            LayoutSFB const& layout_sfb,
                                            int n_row, int k_dim, int group,
                                            int lane) {
    const auto* v2 = reinterpret_cast<const __nv_bfloat162*>(v);
    const __nv_bfloat162 m2 = __hmax2(__habs2(v2[0]), __habs2(v2[1]));
    float amax = fmaxf(fmaxf(__bfloat162float(__low2bfloat16(m2)),
                             __bfloat162float(__high2bfloat16(m2))),
                       1e-10f);
    const unsigned mask = 0xffu << (lane & ~7);
    CUTLASS_PRAGMA_UNROLL
    for (int o = 1; o < 8; o <<= 1)
        amax = fmaxf(amax, __shfl_xor_sync(mask, amax, o));

    const float scale = amax / 448.0f;
    const float biased_exp =
        fminf(fmaxf(ceilf(log2f(scale)) + 127.0f, 0.0f), 254.0f);
    const __nv_bfloat162 inv2 =
        __float2bfloat162_rn(exp2f(127.0f - biased_exp));

    const int base = group * GemmShapeDefaults::SfGran;
    const int j = lane & 7;
    const __nv_bfloat162 p0 = __hmul2(v2[0], inv2);
    const __nv_bfloat162 p1 = __hmul2(v2[1], inv2);
    const uint32_t packed =
        (uint32_t)__nv_cvt_bfloat16raw2_to_fp8x2(
            *reinterpret_cast<const __nv_bfloat162_raw*>(&p0), __NV_SATFINITE,
            __NV_E4M3) |
        ((uint32_t)__nv_cvt_bfloat16raw2_to_fp8x2(
             *reinterpret_cast<const __nv_bfloat162_raw*>(&p1), __NV_SATFINITE,
             __NV_E4M3)
         << 16);
    *reinterpret_cast<uint32_t*>(ws_fp8 + (int64_t)n_row * k_dim + base +
                                 j * 4) = packed;

    if (j == 0) {
        cute::Tensor mSFB = cute::make_tensor(
            cute::make_gmem_ptr(reinterpret_cast<uint8_t*>(ws_sf)), layout_sfb);
        mSFB(sf_row_of<N_PAD>(n_row), base, cute::_0{}) =
            static_cast<uint8_t>(biased_exp);
    }
}

template <int N_PAD, class LayoutSFB>
CUTE_DEVICE void quantize_group32_from_regs8(const __nv_bfloat16 (&v)[8],
                                             e4m3* __restrict__ ws_fp8,
                                             ue8m0* __restrict__ ws_sf,
                                             LayoutSFB const& layout_sfb,
                                             int n_row, int k_dim, int group,
                                             int lane) {
    const auto* v2 = reinterpret_cast<const __nv_bfloat162*>(v);
    const __nv_bfloat162 m2 = __hmax2(__hmax2(__habs2(v2[0]), __habs2(v2[1])),
                                      __hmax2(__habs2(v2[2]), __habs2(v2[3])));
    float amax = fmaxf(fmaxf(__bfloat162float(__low2bfloat16(m2)),
                             __bfloat162float(__high2bfloat16(m2))),
                       1e-10f);

    const unsigned mask = 0xfu << (lane & ~3);

    CUTLASS_PRAGMA_UNROLL
    for (int o = 1; o < 4; o <<= 1)
        amax = fmaxf(amax, __shfl_xor_sync(mask, amax, o));

    const float scale = amax / 448.0f;
    const float biased_exp =
        fminf(fmaxf(ceilf(log2f(scale)) + 127.0f, 0.0f), 254.0f);
    const __nv_bfloat162 inv2 =
        __float2bfloat162_rn(exp2f(127.0f - biased_exp));

    const int base = group * GemmShapeDefaults::SfGran;
    const int j = lane & 3;
    uint32_t packed[2];
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 2; ++i) {
        const __nv_bfloat162 p0 = __hmul2(v2[i * 2], inv2);
        const __nv_bfloat162 p1 = __hmul2(v2[i * 2 + 1], inv2);
        packed[i] = (uint32_t)__nv_cvt_bfloat16raw2_to_fp8x2(
                        *reinterpret_cast<const __nv_bfloat162_raw*>(&p0),
                        __NV_SATFINITE, __NV_E4M3) |
                    ((uint32_t)__nv_cvt_bfloat16raw2_to_fp8x2(
                         *reinterpret_cast<const __nv_bfloat162_raw*>(&p1),
                         __NV_SATFINITE, __NV_E4M3)
                     << 16);
    }

    *reinterpret_cast<uint2*>(ws_fp8 + (int64_t)n_row * k_dim + base + j * 8) =
        *reinterpret_cast<const uint2*>(packed);

    if (j == 0) {
        cute::Tensor mSFB = cute::make_tensor(
            cute::make_gmem_ptr(reinterpret_cast<uint8_t*>(ws_sf)), layout_sfb);
        mSFB(sf_row_of<N_PAD>(n_row), base, cute::_0{}) =
            static_cast<uint8_t>(biased_exp);
    }
}

template <int TileT_, int STAGES = GemmShapeDefaults::TmaStages,
          int ACC_STAGES = 1>
CUTE_DEVICE auto make_gemm_tmem(uint32_t tmem_base) {
    using Mainloop = typename GemmInst<TileT_, STAGES, ACC_STAGES>::Mainloop;
    auto tmem_storage = Mainloop::init_tmem_tensors();
    Mainloop::set_tmem_offsets(tmem_storage, tmem_base);
    return tmem_storage;
}

}  // namespace mega_mla
