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

#include "kernel/mla/mla_intrinsics.cuh"

namespace halops::mla {
struct Sm100 {
    static constexpr int TmemColumns = 512;
    static constexpr int UmmaMaxN = 256;
    static constexpr int SmemPerCtaMax = 227 * 1024;
    static constexpr int Gather4Rows = 4;
    static constexpr int Sw128Bytes = 128;
    static constexpr int Sw128AtomBytes = 8 * Sw128Bytes;
};

struct SfLayout {
    static constexpr int Gran = 32;
    static constexpr int MnPerCol = 32;
    static constexpr int MnPerTile = 4 * MnPerCol;
    static constexpr int PackedMinTile = 64;
};

struct ModelDims {
    static constexpr int Heads = 64;
    static constexpr int DimQ = 512;
    static constexpr int DimV = 512;
    static constexpr int DimNope = 448;
    static constexpr int DimRope = 64;

    static constexpr int QLoraRank = 1024;

    static constexpr int C128Ratio = 128;

    static constexpr int QTotal = Heads * DimQ;
    static constexpr int OTotal = Heads * DimV;
    static constexpr int KvRowBytes = DimNope + 2 * DimRope;
};

struct KvCacheLayout {
    static constexpr int NopeQuantBlock = 64;
    static constexpr int NopeGroups = ModelDims::DimNope / NopeQuantBlock;
    static constexpr int ScaleSlotBytes = sizeof(uint64_t);
    static_assert(NopeGroups <= ScaleSlotBytes, "one scale byte per group");
};

struct KernelBase {
    static constexpr int NumThreads = 384;
    static constexpr int WarpSize = 32;
    static constexpr int ThreadsPerWarpGroup = 128;
    static constexpr int WarpsPerWarpGroup = ThreadsPerWarpGroup / WarpSize;

    struct Tile {
        static constexpr int Widths[4] = {16, 32, 64, 128};
        static constexpr int Limits[3] = {32, 128, 192};
        static constexpr int Smallest = Widths[0];
        static constexpr int Widest = Widths[3];
        static constexpr int WidestFrom = 44;
    };

    static CUTE_DEVICE int warp_idx() { return threadIdx.x / WarpSize; }
    static CUTE_DEVICE int lane_idx() { return threadIdx.x & (WarpSize - 1); }

    struct Bars {
        template <uint32_t Id, int Width>
        struct Bar {
            static constexpr uint32_t id = Id;
            static constexpr int width = Width;
        };

        using Wg0Sync = Bar<1, ThreadsPerWarpGroup>;
        using Wg1Warp02Sync = Bar<2, 2 * WarpSize>;
        using Wg1Warp13Sync = Bar<3, 2 * WarpSize>;
        using EveryoneSync = Bar<4, NumThreads>;
        using Wg1Sync = Bar<5, ThreadsPerWarpGroup>;
        using Wg2Sync = Bar<6, ThreadsPerWarpGroup>;
        using Wg01Sync = Bar<7, 2 * ThreadsPerWarpGroup>;
        using PairWg0Sync = Bar<1, ThreadsPerWarpGroup>;
        using PairBothSync = Bar<2, 2 * ThreadsPerWarpGroup>;
        using PairWg1Sync = Bar<3, ThreadsPerWarpGroup>;
    };

    struct Roles {
        static constexpr int MmaWarp = 0;
        static constexpr int Tma1Warp = 1;
        static constexpr int Tma2Warp = 2;
        static constexpr int Tma3Warp = 3;
        static constexpr int DrainWarpBegin = 4;
        static constexpr int DrainWarps = WarpsPerWarpGroup;
        static constexpr int VecWarpBegin = 8;
        static constexpr int VecWarps = WarpsPerWarpGroup;
        static constexpr int RegsWg0 = 144;
        static constexpr int RegsDrain = 200;
        static constexpr int RegsVec = 160;

        __device__ static constexpr bool is_mma(int warp) {
            return warp == MmaWarp;
        }
        __device__ static constexpr bool is_tma1(int warp) {
            return warp == Tma1Warp;
        }
        __device__ static constexpr bool is_tma2(int warp) {
            return warp == Tma2Warp;
        }
        __device__ static constexpr bool is_tma3(int warp) {
            return warp == Tma3Warp;
        }
        __device__ static constexpr bool is_drain(int warp) {
            return warp >= DrainWarpBegin && warp < DrainWarpBegin + DrainWarps;
        }
        __device__ static constexpr bool is_vec(int warp) {
            return warp >= VecWarpBegin && warp < VecWarpBegin + VecWarps;
        }
        __device__ static constexpr int drain_local_warp(int warp) {
            return warp - DrainWarpBegin;
        }
    };

    template <class Dims>
    struct AttnTrait {
        static constexpr int KvBlock = 64;
        static constexpr int NumBufs = 2;
        static constexpr int KvDenseLevels = 4;

        struct KvSplit {
            static constexpr int SegmentCap = 4;
            static constexpr int MinBlocksToCut = 8;
            static constexpr int Gain = 64;
            static constexpr int TailCap = 2;
        };
        static constexpr int KvGateGridDiv = 6;
        static constexpr float RescaleThreshold = 6.0f;
        static constexpr int KvPause = 1;
        static constexpr int PrequantXMaxT = 256;
        static constexpr int QnormDistMaxT = 16;

        static constexpr float MaxInit = -1e30f;
        static constexpr int Sw128Cols =
            Sm100::Sw128Bytes / sizeof(cutlass::bfloat16_t);
        static constexpr int OTmaBox = Sw128Cols;
        static constexpr int OMmaN = Dims::DimV / 2;

        struct TmemCols {
            static constexpr int O = 0;
            static constexpr int Q = O + OMmaN;
            static constexpr int P = Q + Dims::DimQ / 4;
            static constexpr int PStride = KvBlock;
            static constexpr int Total = Sm100::TmemColumns;
            static_assert(P + 2 * PStride == Total, "TMEM column map");
        };

        using SmemLayoutQ_SW128 = decltype(cute::coalesce(
            cute::tile_to_shape(
                cute::UMMA::Layout_K_SW128_Atom<cutlass::bfloat16_t>{},
                cute::Shape<cute::Int<Dims::Heads>, cute::Int<Dims::DimQ>>{},
                cute::Step<cute::_1, cute::_2>{}),
            cute::Shape<cute::_1, cute::_1>{}));

        using SmemLayoutOBuf = decltype(cute::tile_to_shape(
            cute::UMMA::Layout_K_SW128_Atom<cutlass::bfloat16_t>{},
            cute::Shape<cute::Int<Dims::Heads>, cute::Int<Dims::DimV>>{}));

        using SmemLayoutOBuf_TMA = decltype(cute::tile_to_shape(
            cute::UMMA::Layout_K_SW128_Atom<cutlass::bfloat16_t>{},
            cute::Shape<cute::Int<Dims::Heads>, cute::Int<OTmaBox>>{}));

        using SmemLayoutS = decltype(cute::tile_to_shape(
            cute::UMMA::Layout_K_INTER_Atom<cutlass::bfloat16_t>{},
            cute::Shape<cute::Int<Dims::Heads>, cute::Int<KvBlock>>{},
            cute::Step<cute::_1, cute::_2>{}));

        template <int NumTiles>
        using SmemLayoutKTiles_SW128 = decltype(cute::coalesce(
            cute::tile_to_shape(
                cute::UMMA::Layout_K_SW128_Atom<cutlass::bfloat16_t>{},
                cute::Shape<cute::Int<KvBlock>,
                            cute::Int<Sw128Cols * NumTiles>>{},
                cute::Step<cute::_1, cute::_2>{}),
            cute::Shape<cute::_1, cute::_1>{}));

        template <int NumTiles>
        using SmemLayoutKTiles_DualGemm_SW128 = decltype(cute::coalesce(
            cute::tile_to_shape(
                cute::UMMA::Layout_K_SW128_Atom<cutlass::bfloat16_t>{},
                cute::Shape<cute::Int<KvBlock * 2>,
                            cute::Int<Sw128Cols * NumTiles>>{},
                cute::Step<cute::_1, cute::_2>{}),
            cute::Shape<cute::_1, cute::_1>{}));

        template <int NumTiles>
        using SmemLayoutKTilesTransposed_SW128 = decltype(cute::composition(
            SmemLayoutKTiles_SW128<NumTiles>{},
            cute::Layout<cute::Shape<cute::Int<Sw128Cols * NumTiles>,
                                     cute::Int<KvBlock>>,
                         cute::Stride<cute::Int<KvBlock>, cute::_1>>{}));

        using TiledMMA_P = decltype(cute::make_tiled_mma(
            SM100_MMA_F16BF16_WS_TS_NOELECT<
                cutlass::bfloat16_t, cutlass::bfloat16_t, float, Dims::Heads,
                KvBlock * 2, cute::UMMA::Major::K, cute::UMMA::Major::K>{}));

        using TiledMMA_O = decltype(cute::make_tiled_mma(
            SM100_MMA_F16BF16_WS_SS_NOELECT<
                cutlass::bfloat16_t, cutlass::bfloat16_t, float, Dims::Heads,
                OMmaN, cute::UMMA::Major::K, cute::UMMA::Major::MN>{}));
    };
};

template <class Dims_, int TileT_>
struct KernelTraits : KernelBase {
    using Dims = Dims_;
    static constexpr int TileT = TileT_;
    using Attn = KernelBase::AttnTrait<Dims_>;

    struct GemmBase {
        using OwnerTraits = KernelTraits;
        using Geom = GemmBase;

        static constexpr int TokenTile = TileT_;

        static constexpr bool SfTilePerGroup =
            TokenTile < SfLayout::PackedMinTile;

        static constexpr int NoSwapMinTile = 128;
        static constexpr bool WeightInA = TokenTile < NoSwapMinTile;

        static constexpr int BK = WeightInA ? 256 : 128;
        static constexpr int KItersPerBlock = BK / SfLayout::Gran;

        static constexpr int WeightTile = WeightInA ? 128 : 256;
        static constexpr int BM = WeightInA ? WeightTile : TokenTile;
        static constexpr int BN = WeightInA ? TokenTile : WeightTile;
        static constexpr int CtaNSf = cute::round_up(BN, SfLayout::MnPerTile);

        static constexpr int SmemPerStage = BM * BK + BN * BK +
                                            BM * BK / SfLayout::Gran +
                                            CtaNSf * BK / SfLayout::Gran;
        static constexpr int SmemStageBudget = 216 * 1024;
        static constexpr int SmemMaxStages = SmemStageBudget / SmemPerStage;

        static constexpr int StageCap = WeightInA ? 4 : 3;
        static constexpr int TmaStages =
            SmemMaxStages < StageCap ? SmemMaxStages : StageCap;

        static constexpr int AccStages = 1;
        static constexpr int KSplit = 1;
        using PartialT = cutlass::bfloat16_t;
        static constexpr int DrainNOverride = 0;

        static constexpr int ChainBN = WeightInA ? TokenTile : 256;
        static constexpr int ChainAccStages = 2;
        static constexpr int ChainCtaNSf =
            cute::round_up(ChainBN, SfLayout::MnPerTile);
        static constexpr int ChainSfCols =
            BK / SfLayout::Gran +
            (ChainCtaNSf / SfLayout::MnPerCol) * (BK / (4 * SfLayout::Gran));

        static constexpr int chain_acc_stride() {
            int result = -1;
            if (ChainAccStages * ChainBN + ChainSfCols <= Sm100::TmemColumns)
                result = ChainBN;
            for (int ov : {16, 32, 64, 128}) {
                const int stride = Sm100::UmmaMaxN - ov;
                if (result < 0 &&
                    (ChainAccStages - 1) * stride + ChainBN + ChainSfCols <=
                        Sm100::TmemColumns &&
                    2 * stride >= ChainBN)
                    result = stride;
            }
            return result;
        }
        static constexpr int AccStride = chain_acc_stride();
        static_assert(AccStride > 0, "chain accumulators exceed TMEM");
        static constexpr int AccSharedCols =
            ChainBN > AccStride ? ChainBN - AccStride : 0;
        static constexpr int SfBaseCols =
            (ChainAccStages - 1) * AccStride + ChainBN;
    };

    struct Qkv : GemmBase {
        static constexpr int AccStages = GemmBase::ChainAccStages;
        static constexpr int KSplit = 4;
        using PartialT = float;
        static constexpr int DrainNOverride = GemmBase::WeightInA ? 0 : 32;
    };

    struct WqB : GemmBase {
        static constexpr int AccStages = GemmBase::ChainAccStages;
    };

    struct KvGate : GemmBase {
        using Geom = KvGate;
        static constexpr int BK = GemmBase::BK / 2;
        static constexpr int AccStages = GemmBase::ChainAccStages;
        using PartialT = float;

        static constexpr bool WeightInA = true;
        static constexpr int WeightTile = 128;
        static constexpr int BM = WeightTile;
        static constexpr int BN = GemmBase::TokenTile;
        static constexpr int CtaNSf = cute::round_up(BN, SfLayout::MnPerTile);
    };

    static constexpr int MaxTmaStages =
        cute::max(Qkv::TmaStages, WqB::TmaStages);
    static constexpr int MaxAccStages =
        cute::max(Qkv::AccStages, WqB::AccStages);
};

using AttnConfig = KernelBase::AttnTrait<ModelDims>;

struct AttnRing {
    int buf_idx = 0;
    bool bar_phase = 0;
    CUTE_DEVICE void update() {
        bar_phase ^= (buf_idx == AttnConfig::NumBufs - 1);
        buf_idx = (buf_idx + 1) % AttnConfig::NumBufs;
    }
};

using TraitsN16 = KernelTraits<ModelDims, 16>;
using TraitsN32 = KernelTraits<ModelDims, 32>;
using TraitsN64 = KernelTraits<ModelDims, 64>;
using TraitsN128 = KernelTraits<ModelDims, 128>;

struct WorkspaceLayout {
    static constexpr int GridBarrierBytes = 128;
    static constexpr int RendezvousSlotBytes = 128;
    static constexpr int BlockAlignBytes = 128;
    static constexpr int OutputProjAlignBytes = Sm100::Sw128AtomBytes;
    static constexpr int TokenPadRows = SfLayout::MnPerTile;
    static constexpr int SfPadRows = TokenPadRows / KernelBase::Tile::Smallest;
};

struct C128Layout {
    static constexpr int DimsPerLane = 4;
    static constexpr int DimsPerWarp = 2 * DimsPerLane;
    static constexpr int RowGroups = KernelBase::WarpSize / 2;
    static constexpr int SliceDim = KernelBase::Roles::VecWarps * DimsPerWarp;
    static constexpr int Slices = ModelDims::DimV / SliceDim;
    static constexpr int FlagSlicesDone = 0;
    static constexpr int FlagRowWritten = 1;
    static constexpr int FlagsPerToken = 2;
};

struct OutputProjLayout {
    static constexpr int ClusterSize = 4;
    static constexpr int RowTile = 128;
    static constexpr int MaxTokenGroups = 2;
    static constexpr int MaxTokens = 640;
};

struct OutputProjShape {
    int hidden;
    int olora;
    int group_k;
    int group_rows;

    CUTE_HOST_DEVICE int clusters() const {
        return hidden / OutputProjLayout::RowTile;
    }
    CUTE_HOST_DEVICE int grid() const {
        return clusters() * OutputProjLayout::ClusterSize;
    }
    CUTE_HOST_DEVICE int wo_a_tiles() const {
        return olora / OutputProjLayout::RowTile;
    }
    CUTE_HOST_DEVICE int n_flags() const {
        return OutputProjLayout::MaxTokenGroups * wo_a_tiles();
    }
};

struct QuantX {
    static constexpr int PerThread = 8;
    static constexpr int ThreadsPerGrp = SfLayout::Gran / PerThread;
    static constexpr int GrpPerPass =
        KernelBase::Roles::VecWarps * KernelBase::WarpSize / ThreadsPerGrp;

    CUTE_HOST_DEVICE static bool active(int cta, int items) {
        return cta * GrpPerPass < items;
    }
    CUTE_HOST_DEVICE static int ctas(int items, int grid) {
        const int need = (items + GrpPerPass - 1) / GrpPerPass;
        return need < grid ? need : grid;
    }
};
}  // namespace halops::mla
