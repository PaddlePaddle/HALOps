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

#include "kernel/mla/common/arch.cuh"
#include "kernel/mla/common/umma_traits.cuh"

namespace halops::mla {
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
}  // namespace halops::mla
