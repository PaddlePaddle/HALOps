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

/// Tile-independent shape of the 2-SM pair UMMA output projection: one K block
/// is a 128-byte swizzle row, a CTA pair owns 256 weight rows and the drain
/// moves the accumulator in 32-token slices.
struct OprojPairBase {
    static constexpr int kBlockK =
        Sm100::Sw128Bytes / sizeof(cutlass::float_e4m3_t);
    static constexpr int kPairRows = 2 * OutputProjLayout::RowTile;
    static constexpr int kSlice = 32;
};

/// Pipeline depth, UMMA traits, shared memory and drain layout of the pair
/// UMMA output projection for TT tokens per group in NH UMMA N parts.
template <int TT_, int NH_>
struct OprojPairCfg : OprojPairBase {
    static constexpr int TT = TT_;
    static constexpr int NH = NH_;
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

    using GemmA = MlaGemmConfigT<TN, Stages, 1, kPairRows, kBlockK, 0, true>;
    using GemmB =
        MlaGemmConfigT<TN, Stages * NH, 1, kPairRows, kBlockK, 0, true>;
    using Tr = GemmKernelTraits<cutlass::float_e4m3_t, cutlass::float_e4m3_t,
                                cutlass::bfloat16_t, cutlass::bfloat16_t,
                                cutlass::float_ue8m0_t, float, GemmA>;
    using TrB = GemmKernelTraits<cutlass::float_e4m3_t, cutlass::float_e4m3_t,
                                 cutlass::bfloat16_t, cutlass::bfloat16_t,
                                 cutlass::float_ue8m0_t, float, GemmB>;
    using ML = CollectiveMainloop<Tr>;

    /// Shared memory of one CTA: the TMA stages, the drain transpose buffers
    /// and the pipeline barriers.
    struct Smem {
        alignas(Sm100::Sw128AtomBytes) cute::ArrayEngine<
            cutlass::float_e4m3_t, cute::cosize_v<typename Tr::SmemLayoutA>> A;
        alignas(Sm100::Sw128AtomBytes) cute::ArrayEngine<
            cutlass::float_e4m3_t, cute::cosize_v<typename TrB::SmemLayoutB>> B;
        alignas(128)
            cute::ArrayEngine<cutlass::float_ue8m0_t,
                              cute::cosize_v<typename Tr::SmemLayoutSFA>> SFA;
        alignas(128)
            cute::ArrayEngine<cutlass::float_ue8m0_t,
                              cute::cosize_v<typename TrB::SmemLayoutSFB>> SFB;
        alignas(128) __nv_bfloat16 transpose[2][kSlice * TransposeRow];
        alignas(8) uint64_t tma_a[Stages], tma_b[Stages], smem_free[Stages];
        alignas(8) uint64_t acc_full, acc_empty, rx_full[Slices / 2], rx_free;
        uint32_t tmem;
    };
    static constexpr size_t kSmemLaunchBytes =
        sizeof(Smem) + Sm100::Sw128AtomBytes;

    static constexpr int GroupsPerToken =
        OutputProjLayout::RowTile / SfLayout::Gran;
    static constexpr int BfPerVec = sizeof(uint4) / sizeof(__nv_bfloat16);
    static constexpr int GroupWords = SfLayout::Gran / sizeof(uint32_t);
    static constexpr int Units = Slices / 2;
    static constexpr int UnitFloats = OutputProjLayout::RowTile * kSlice;
    static constexpr int DryRows = 8;

    static constexpr int kVecFloats = sizeof(float4) / sizeof(float);
    static constexpr int kVecWords = sizeof(uint4) / sizeof(uint32_t);
    static constexpr int kTmaModes = 3;
    static constexpr int kViewA = 0;
    static constexpr int kViewB = 1;
    static constexpr int kViewSFA = 2;
    static constexpr int kViewSFB = 3;
    static constexpr int kPartGA = 0;
    static constexpr int kPartSA = 1;
    static constexpr int kPartGB = 2;
    static constexpr int kPartSB = 3;
    static constexpr int kPartGSFA = 4;
    static constexpr int kPartSSFA = 5;
    static constexpr int kPartGSFB = 6;
    static constexpr int kPartSSFB = 7;
    static constexpr uint32_t kPairCtaMask = 3u;
    static constexpr int kTmemLaneShift = 16;

    static_assert(OutputProjLayout::RowTile == KernelBase::ThreadsPerWarpGroup,
                  "one weight row per thread");
    static_assert(kSlice * GroupsPerToken == KernelBase::ThreadsPerWarpGroup,
                  "one task per thread");
    static_assert(GroupWords == 2 * kVecWords,
                  "a quant group stores two 16-B fp8 vectors");
    static_assert(DryRows * kSlice * sizeof(float) <=
                      sizeof(Smem::transpose[0]),
                  "dry staging fits the transpose");
    static_assert(KernelBase::Bars::PairWg0Sync::width ==
                      KernelBase::Bars::PairWg1Sync::width,
                  "both drain WGs are one warp group");
};

}  // namespace halops::mla
