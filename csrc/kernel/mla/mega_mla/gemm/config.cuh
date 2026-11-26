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
#include "kernel/mla/common/intrinsics.cuh"
#include "kernel/mla/common/traits.cuh"
#include "kernel/mla/mega_mla/mla_utils.cuh"

namespace halops::mla {
struct CommonGemmModes {
    static constexpr int StageMode = 3;
};

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
}  // namespace halops::mla
