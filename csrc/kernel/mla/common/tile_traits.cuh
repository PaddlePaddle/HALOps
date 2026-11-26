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
#include "kernel/mla/common/base_traits.cuh"
#include "kernel/mla/common/model.cuh"

namespace halops::mla {
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
}  // namespace halops::mla
