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

// GEMM tile configs: `GemmTile` parameter lists named by `GemmKernelConfig...`
// aliases. Dispatch catalogs and selectors live in dense_cfgs.cuh.

#pragma once

#include "kernel/gemm/utils.cuh"

namespace halops {

//   BM, BN, BK    KBlockM/N/K; for SwapAB, BM is the weight's M and BN the
//                 activation extent.
//   PST, EST      prologue (mainloop) / epilogue pipeline depth.
//   CM, CN        cluster shape; IsPairUmma = CM*CN > 1.
//   OVLP          KTmemOverlapCols.
//   REG           KNumRegsEpilogue; 160 is the Reg160 occupancy variant.
//   SWZ, PSFB     UseTileSwizzle, IsPackedSfb.
//   FENCE         SkipTmemFence.
//   SWAP, FP4     SwapAB, KIsFP4.
//   SFVS          KSfVecSize: 16 = NVFP4, 32 = MXFP4.
//   ALPHA, BIAS   HasAlpha, AddBias.
//   MODE          KernelType.
template <int BM, int BN, int BK, int PST, int EST, int CM = 1, int CN = 1,
          int OVLP = 0, int REG = 208, bool SWZ = false, bool PSFB = false,
          bool FENCE = false, bool SWAP = false, bool FP4 = false,
          int SFVS = 16, bool ALPHA = false, bool BIAS = false,
          KernelMode MODE = KernelMode::Naive>
struct GemmTile : GemmKernelConfig {
    static constexpr int KBlockM = BM;
    static constexpr int KBlockN = BN;
    static constexpr int KBlockK = BK;
    static constexpr int KPrologueStages = PST;
    static constexpr int KEpilogueStages = EST;
    static constexpr int KClusterM = CM;
    static constexpr int KClusterN = CN;
    static constexpr int KTmemOverlapCols = OVLP;
    static constexpr int KNumRegsEpilogue = REG;
    static constexpr bool SkipTmemFence = FENCE;
    static constexpr bool UseTileSwizzle = SWZ;
    static constexpr bool IsPackedSfb = PSFB;
    static constexpr bool SwapAB = SWAP;
    static constexpr bool KIsFP4 = FP4;
    static constexpr int KSfVecSize = SFVS;
    static constexpr bool HasAlpha = ALPHA;
    static constexpr bool AddBias = BIAS;
    static constexpr KernelMode KernelType = MODE;
    static constexpr bool IsPairUmma = (CM * CN) > 1;
};

using GemmKernelConfigNaiveNoBias256x256x128_2CTA =
    GemmTile<256, 256, 128, 6, 2, 2, 1, 16>;

using GemmKernelConfigNaiveNoBias256x192x128_2CTA =
    GemmTile<256, 192, 128, 7, 2, 2, 1, 0, 208, false, false, true>;

struct GemmKernelConfigNaiveNoBias256x192x128_2CTA_Reg160
    : GemmKernelConfigNaiveNoBias256x192x128_2CTA {
    static constexpr int KNumRegsEpilogue = 160;
};

using GemmKernelConfigNaiveNoBias256x224x128_2CTA =
    GemmTile<256, 224, 128, 6, 2, 2>;

// SFB prepacked per 224-column tile as two aligned 128-row atoms (no remap).
struct GemmKernelConfigNaiveNoBias256x224x128_2CTA_PackedSfb
    : GemmKernelConfigNaiveNoBias256x224x128_2CTA {
    static constexpr bool IsPackedSfb = true;
};

struct GemmKernelConfigNaiveNoBias256x224x128_2CTA_PackedSfb_Reg160
    : GemmKernelConfigNaiveNoBias256x224x128_2CTA_PackedSfb {
    static constexpr int KNumRegsEpilogue = 160;
    static constexpr bool SkipTmemFence = true;
};

// Packed SFB frees enough SMEM for a 7th prologue stage.
struct GemmKernelConfigNaiveNoBias256x224x128_2CTA_PackedSfb_Reg160_Stage7
    : GemmKernelConfigNaiveNoBias256x224x128_2CTA_PackedSfb_Reg160 {
    static constexpr int KPrologueStages = 7;
};

using GemmKernelConfigNaiveNoBias128x256x128_1CTA =
    GemmTile<128, 256, 128, 4, 2, 1, 1, 16>;

// Half-N variant for problems whose 128x256 grid underfills the GPU.
using GemmKernelConfigNaiveNoBias128x128x128_1CTA =
    GemmTile<128, 128, 128, 6, 2>;

// Half-N pair-UMMA tile: twice the tiles and room for 8 mainloop stages.
using GemmKernelConfigNaiveNoBias256x128x128_2CTA =
    GemmTile<256, 128, 128, 8, 2, 2>;

using GemmKernelConfigNaiveNoBias128x64x256_SwapAB_1CTA =
    GemmTile<128, 64, 256, 4, 7, 1, 1, 0, 208, false, false, false, true>;

// SwapAB with a 16-row activation tile for M <= 16.
using GemmKernelConfigNaiveNoBias128x16x256_SwapAB_1CTA =
    GemmTile<128, 16, 256, 5, 7, 1, 1, 0, 208, false, false, false, true>;

struct
    GemmKernelConfigNaiveNoBias256x224x128_2CTA_PackedSfb_Reg160_Stage7_Swizzle
    : GemmKernelConfigNaiveNoBias256x224x128_2CTA_PackedSfb_Reg160_Stage7 {
    static constexpr bool UseTileSwizzle = true;
};

using GemmKernelConfigNaiveNoBias128x64x128_1CTA = GemmTile<128, 64, 128, 9, 2>;

using GemmKernelConfigNaiveNoBias256x64x128_2CTA =
    GemmTile<256, 64, 128, 8, 2, 2>;

// One wave (8 x 19 tiles) on 152 SMs for 1024 x 4096; packed SFB.
using GemmKernelConfig128x224x128_1CTA_PackedSfb =
    GemmTile<128, 224, 128, 4, 2, 1, 1, 0, 160, false, true>;

using GemmKernelConfigNaiveBias256x128x128_2CTA =
    GemmTile<256, 128, 128, 6, 3, 2, 1, 0, 208, false, false, false, false,
             false, 16, false, true>;

using GemmKernelConfigContiguousNoBias256x256x128_2CTA =
    GemmTile<256, 256, 128, 6, 2, 2, 1, 16, 208, false, false, false, false,
             false, 16, false, false, KernelMode::Contiguous>;

using GemmKernelConfigContiguousNoBias128x256x128_1CTA =
    GemmTile<128, 256, 128, 4, 2, 1, 1, 16, 208, false, false, false, false,
             false, 16, false, false, KernelMode::Contiguous>;

using GemmKernelConfigContiguousNoBias128x64x256_SwapAB_1CTA =
    GemmTile<128, 64, 256, 4, 7, 1, 1, 0, 208, false, false, false, true, false,
             16, false, false, KernelMode::Contiguous>;

using GemmKernelConfigContiguousBias256x128x128_2CTA =
    GemmTile<256, 128, 128, 6, 3, 2, 1, 0, 208, false, false, false, false,
             false, 16, false, true, KernelMode::Contiguous>;

using GemmKernelConfigMaskedNoBias128x64x256_SwapAB_1CTA =
    GemmTile<128, 64, 256, 4, 7, 1, 1, 0, 208, false, false, false, true, false,
             16, false, false, KernelMode::Masked>;

using GemmKernelConfigMaskedNoBias128x256x128_1CTA =
    GemmTile<128, 256, 128, 4, 2, 1, 1, 16, 208, false, false, false, false,
             false, 16, false, false, KernelMode::Masked>;

using GemmKernelConfigMaskedBias256x128x128_2CTA =
    GemmTile<256, 128, 128, 6, 3, 2, 1, 0, 208, false, false, false, false,
             false, 16, false, true, KernelMode::Masked>;

using GemmKernelConfigMaskedNoBias256x256x128_2CTA =
    GemmTile<256, 256, 128, 6, 2, 2, 1, 16, 208, false, false, false, false,
             false, 16, false, false, KernelMode::Masked>;

// W4A4 (NVFP4): e2m1 inputs, SF vector size 16, MMA K = 64.

using GemmKernelConfigW4A4NaiveNoBias256x256x256_2CTA =
    GemmTile<256, 256, 256, 5, 2, 2, 1, 48, 208, false, false, false, false,
             true, 16, true>;

using GemmKernelConfigW4A4NaiveNoBias128x256x256_1CTA =
    GemmTile<128, 256, 256, 4, 2, 1, 1, 48, 208, false, false, false, false,
             true, 16, true>;

using GemmKernelConfigW4A4NaiveNoBias128x64x512_1CTA =
    GemmTile<128, 64, 512, 4, 7, 1, 1, 0, 208, false, false, false, true, true,
             16, true>;

using GemmKernelConfigW4A4ContiguousNoBias256x256x256_2CTA =
    GemmTile<256, 256, 256, 5, 2, 2, 1, 48, 208, false, false, false, false,
             true, 16, true, false, KernelMode::Contiguous>;

using GemmKernelConfigW4A4ContiguousNoBias128x256x256_1CTA =
    GemmTile<128, 256, 256, 4, 2, 1, 1, 48, 208, false, false, false, false,
             true, 16, true, false, KernelMode::Contiguous>;

using GemmKernelConfigW4A4ContiguousNoBias128x64x512_1CTA =
    GemmTile<128, 64, 512, 4, 7, 1, 1, 0, 208, false, false, false, true, true,
             16, true, false, KernelMode::Contiguous>;

// Batch mode: Contiguous tiles, expert index derived arithmetically (no
// expert_ids).

using GemmKernelConfigBatchNoBias256x256x128_2CTA =
    GemmTile<256, 256, 128, 6, 2, 2, 1, 16, 208, false, false, false, false,
             false, 16, false, false, KernelMode::Batch>;

using GemmKernelConfigBatchNoBias128x256x128_1CTA =
    GemmTile<128, 256, 128, 4, 2, 1, 1, 16, 208, false, false, false, false,
             false, 16, false, false, KernelMode::Batch>;

using GemmKernelConfigBatchNoBias128x64x256_SwapAB_1CTA =
    GemmTile<128, 64, 256, 4, 7, 1, 1, 0, 208, false, false, false, true, false,
             16, false, false, KernelMode::Batch>;

using GemmKernelConfigBatchBias256x128x128_2CTA =
    GemmTile<256, 128, 128, 6, 3, 2, 1, 0, 208, false, false, false, false,
             false, 16, false, true, KernelMode::Batch>;

using GemmKernelConfigW4A4BatchNoBias256x256x256_2CTA =
    GemmTile<256, 256, 256, 5, 2, 2, 1, 48, 208, false, false, false, false,
             true, 16, true, false, KernelMode::Batch>;

using GemmKernelConfigW4A4BatchNoBias128x256x256_1CTA =
    GemmTile<128, 256, 256, 4, 2, 1, 1, 48, 208, false, false, false, false,
             true, 16, true, false, KernelMode::Batch>;

using GemmKernelConfigW4A4BatchNoBias128x64x512_1CTA =
    GemmTile<128, 64, 512, 4, 7, 1, 1, 0, 208, false, false, false, true, true,
             16, true, false, KernelMode::Batch>;

using GemmKernelConfigW4A4MaskedNoBias256x256x256_2CTA =
    GemmTile<256, 256, 256, 5, 2, 2, 1, 48, 208, false, false, false, false,
             true, 16, true, false, KernelMode::Masked>;

using GemmKernelConfigW4A4MaskedNoBias128x256x256_1CTA =
    GemmTile<128, 256, 256, 4, 2, 1, 1, 48, 208, false, false, false, false,
             true, 16, true, false, KernelMode::Masked>;

using GemmKernelConfigW4A4MaskedNoBias128x64x512_1CTA =
    GemmTile<128, 64, 512, 4, 7, 1, 1, 0, 208, false, false, false, true, true,
             16, true, false, KernelMode::Masked>;

// W4A4 tile variants derived from the base aliases above.

#define IR_W4A4_SWAPAB(NAME, ACT, KB, PST, EST)                             \
    using NAME =                                                            \
        GemmTile<128, ACT, KB, PST, EST, 1, 1, 0, 208, false, false, false, \
                 /*SWAP=*/true, /*FP4=*/true, 16, /*ALPHA=*/true>;
#define IR_W4A4_PLAIN(NAME, BM, BN, KB, PAIR, PST, EST, OVLP)               \
    using NAME =                                                            \
        GemmTile<BM, BN, KB, PST, EST, (PAIR) ? 2 : 1, 1, OVLP, 208, false, \
                 false, false, false, /*FP4=*/true, 16, /*ALPHA=*/true>;
#define IR_W4A4_REG160(NAME, BASE)                   \
    struct NAME : BASE {                             \
        static constexpr int KNumRegsEpilogue = 160; \
        static constexpr bool SkipTmemFence = true;  \
    };
#define IR_W4A4_EPI(NAME, BASE, EST)                \
    struct NAME : BASE {                            \
        static constexpr int KEpilogueStages = EST; \
    };

IR_W4A4_REG160(GemmKernelConfigW4A4NaiveNoBias256x256x256_2CTA_Reg160,
               GemmKernelConfigW4A4NaiveNoBias256x256x256_2CTA)
IR_W4A4_REG160(GemmKernelConfigW4A4NaiveNoBias128x256x256_1CTA_Reg160,
               GemmKernelConfigW4A4NaiveNoBias128x256x256_1CTA)
IR_W4A4_REG160(GemmKernelConfigW4A4NaiveNoBias128x64x512_1CTA_Reg160,
               GemmKernelConfigW4A4NaiveNoBias128x64x512_1CTA)
IR_W4A4_SWAPAB(GemmKernelConfigW4A4NaiveNoBiasSwA128x32x256_S9, 32, 256, 9, 7)
IR_W4A4_SWAPAB(GemmKernelConfigW4A4NaiveNoBiasSwA128x128x256_S6, 128, 256, 6, 3)
IR_W4A4_SWAPAB(GemmKernelConfigW4A4NaiveNoBiasSwA128x16x256_S10, 16, 256, 10, 7)
IR_W4A4_REG160(GemmKernelConfigW4A4NaiveNoBiasSwA128x16x256_S10_Reg160,
               GemmKernelConfigW4A4NaiveNoBiasSwA128x16x256_S10)
IR_W4A4_PLAIN(GemmKernelConfigW4A4NaiveNoBias256x128x256_2CTA_S8, 256, 128, 256,
              true, 8, 2, 0)
IR_W4A4_EPI(GemmKernelConfigW4A4NaiveNoBias256x128x256_2CTA_S8_E3,
            GemmKernelConfigW4A4NaiveNoBias256x128x256_2CTA_S8, 3)
#undef IR_W4A4_SWAPAB
#undef IR_W4A4_PLAIN
#undef IR_W4A4_REG160
#undef IR_W4A4_EPI

}  // namespace halops
