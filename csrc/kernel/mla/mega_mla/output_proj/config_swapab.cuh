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

#include "kernel/mla/mega_mla/mla_gemm.cuh"

namespace halops::mla {

/// K block and exchange width of the swap-AB output projection for a TT-token
/// tile: small tiles take 256-deep K blocks, and the cluster reduction moves
/// the accumulator in 32-token (16 for odd-sized tiles) exchanges.
template <int TT>
struct OprojSwapabTile {
    static constexpr int BK = TT <= 64 ? 256 : 128;
    static constexpr int Exchange = TT > 64 ? 32 : TT % 32 == 0 ? 32 : 16;
};

/// GEMM geometry handed to the shared mla_gemm pipeline: weight in A, the
/// TT-token tile in B, one or two TMEM accumulator stages.
template <int TT, int Stages>
struct OprojSwapabGeom {
    using Geom = OprojSwapabGeom;
    /// Stage count of the shared TMA arena.
    struct OwnerTraits {
        static constexpr int MaxTmaStages = Stages;
    };
    static constexpr int TokenTile = TT;
    static constexpr bool SfTilePerGroup = TT < SfLayout::PackedMinTile;
    static constexpr bool WeightInA = true;
    static constexpr int BK = OprojSwapabTile<TT>::BK;
    static constexpr int KItersPerBlock = BK / SfLayout::Gran;
    static constexpr int WeightTile = OutputProjLayout::RowTile;
    static constexpr int BM = OutputProjLayout::RowTile;
    static constexpr int BN = TT;
    static constexpr int CtaNSf = cute::round_up(BN, SfLayout::MnPerTile);
    static constexpr int TmaStages = Stages;
    static constexpr int SfCols =
        BK / SfLayout::Gran + (CtaNSf / SfLayout::MnPerCol) * (BK / 128);
    static constexpr int AccStages =
        2 * TT + SfCols <= Sm100::TmemColumns ? 2 : 1;
    using PartialT = cutlass::bfloat16_t;
    static constexpr int DrainNOverride =
        OprojSwapabTile<TT>::Exchange < TT ? OprojSwapabTile<TT>::Exchange : 0;
    static constexpr int ChainBN = TT;
    static constexpr int ChainCtaNSf = CtaNSf;
    static constexpr int AccStride = TT;
    static constexpr int AccSharedCols = 0;
    static constexpr int SfBaseCols = AccStages * TT;
};

/// Pipeline depth, shared-memory map and cluster reduction layout of the
/// swap-AB output projection for a TT-token tile.
template <int TT_>
struct OprojSwapabCfg : OprojSwapabTile<TT_> {
    using OprojSwapabTile<TT_>::BK;
    using OprojSwapabTile<TT_>::Exchange;
    static constexpr int TT = TT_;
    static constexpr int Slots = TT / Exchange;
    static constexpr int Ranks = OutputProjLayout::ClusterSize;
    static constexpr int RankRows = OutputProjLayout::RowTile / Ranks;
    static constexpr int SfRows = cute::round_up(TT, SfLayout::MnPerTile);
    static constexpr int StageBytes =
        OutputProjLayout::RowTile * BK + TT * BK +
        (OutputProjLayout::RowTile + SfRows) * BK / SfLayout::Gran;
    static constexpr int RxFloats = 2 * OutputProjLayout::RowTile * Exchange;
    static constexpr int ExchangeBytes = 2 * RxFloats * (int)sizeof(float);
    static constexpr int BarrierReserve = 3 * 1024;
    static constexpr int MaxStages = 6;
    static constexpr int FitStages =
        (Sm100::SmemPerCtaMax - ExchangeBytes - BarrierReserve) / StageBytes;
    static constexpr int Stages = FitStages < MaxStages ? FitStages : MaxStages;
    using Gemm = OprojSwapabGeom<TT, Stages>;
    static constexpr int TilesPerKBlock = BK / OutputProjLayout::RowTile;
    static constexpr int TileArrivals = 2;

    using Inst = GemmInst<Gemm>;
    static constexpr int Arena =
        cute::round_up(Inst::SmemTotal, Sm100::Sw128AtomBytes);
    static constexpr int RxOff = Arena;
    static constexpr int StagingOff = RxOff + RxFloats * (int)sizeof(float);
    static constexpr int BarOff = StagingOff + RxFloats * (int)sizeof(float);
    /// Pipeline and exchange barriers, the TMEM slot and the amax exchange.
    struct Bars {
        cutlass::arch::ClusterTransactionBarrier tma_a[Stages], tma_b[Stages],
            smem_free[Stages];
        cutlass::arch::ClusterTransactionBarrier acc_full[2], acc_empty[2];
        uint64_t rx_full[2], rx_free[2];
        uint32_t tmem_slot;
        float amax[KernelBase::ThreadsPerWarpGroup];
    };
    static constexpr int kSmemBytes = BarOff + (int)sizeof(Bars);

    static constexpr int kPrefetchSkip = Ranks * Stages * BK;
    static constexpr int kHalfRows = OutputProjLayout::RowTile / 2;
    static constexpr int kTaskRows = SfLayout::Gran / 2;
    static constexpr int kTasks = OutputProjLayout::RowTile / Exchange;
    static constexpr int kRowsPerTask = RankRows / kTasks;
    static constexpr int kBufFloats = OutputProjLayout::RowTile * Exchange;
    static constexpr uint32_t kRowBytes = Exchange * sizeof(float);
    static constexpr uint32_t kSlotBytes =
        OutputProjLayout::RowTile * kRowBytes;

    static constexpr int kVecFloats = sizeof(float4) / sizeof(float);
    static constexpr int kVecWords = sizeof(uint4) / sizeof(uint32_t);
    static constexpr int kBfPerVec = sizeof(uint4) / sizeof(__nv_bfloat16);
    static constexpr int kCodesPerWord =
        sizeof(uint32_t) / sizeof(cutlass::float_e4m3_t);

    static_assert(RankRows == KernelBase::WarpSize, "one wo_b row per lane");
    static_assert(kTaskRows / kCodesPerWord == kVecWords,
                  "a wo_a task stores one 16-B fp8 vector");
};

}  // namespace halops::mla
