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

#include <array>
#include <cstdint>

#include "kernel/gemm/utils.cuh"

namespace halops {

// ---------------------------------------------------------------------------
// Dense GEMM tile parameters; this is the only tile definition in the tree.
//
//   BM, BN, BK    KBlockM/N/K; for SwapAB, BM is the weight's M and BN the
//                 activation extent.
//   PST, EST      prologue (mainloop) / epilogue pipeline depth.
//   CM, CN        cluster shape; IsPairUmma = CM*CN > 1.
//   OVLP          KTmemOverlapCols.
//   REG           KNumRegsEpilogue; 160 is the Reg160 occupancy variant.
//   SWZ           UseTileSwizzle.
//   FENCE         SkipTmemFence.
//   SWAP, FP4     SwapAB, KIsFP4.
//   SFVS          KSfVecSize: 16 = NVFP4, 32 = MXFP4.
//   ALPHA, BIAS   HasAlpha, AddBias.
//   MODE          KernelType.
//   EWGO          KEpilogueWarpGroupsOverride; 0 = auto (SwapAB ? 1 : 2).
//   NTAIL         KForceEpiNPredicate (predicated store on power-of-2 N).
//   BF16          KIsBF16 (native BF16 GEMM, no scale factors).
//   ATOM          CTAs per MMA atom: 0 = auto (CM*CN > 1 ? 2 : 1), 1 = 1SM,
//                 2 = 2x1SM. Decoupled from the cluster shape so a 1SM atom
//                 can still sit in a multi-CTA cluster and share loads.
// ---------------------------------------------------------------------------
template <int BM, int BN, int BK, int PST, int EST, int CM = 1, int CN = 1,
          int OVLP = 0, int REG = 208, bool SWZ = false, bool FENCE = false,
          bool SWAP = false, bool FP4 = false, int SFVS = 16,
          bool ALPHA = false, bool BIAS = false,
          KernelMode MODE = KernelMode::Dense, int EWGO = 0, bool NTAIL = false,
          bool BF16 = false, int ATOM = 0>
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
    static constexpr bool SwapAB = SWAP;
    static constexpr bool KIsFP4 = FP4;
    static constexpr int KSfVecSize = SFVS;
    static constexpr bool HasAlpha = ALPHA;
    static constexpr bool AddBias = BIAS;
    static constexpr KernelMode KernelType = MODE;
    // CTAs per MMA atom (0 = legacy: any cluster > 1 implies pair-UMMA).
    static constexpr int KAtomCtas = ATOM == 0 ? ((CM * CN) > 1 ? 2 : 1) : ATOM;
    static constexpr bool IsPairUmma = KAtomCtas == 2;
    static constexpr int KEpilogueWarpGroupsOverride = EWGO;
    static constexpr bool KForceEpiNPredicate = NTAIL;
    static constexpr bool KIsBF16 = BF16;
};

struct TileSpec {
    int block_m = 0;          // KBlockM
    int block_n = 0;          // KBlockN
    int block_k = 0;          // KBlockK
    int main_stages = 2;      // KPrologueStages
    int epi_stages = 2;       // KEpilogueStages
    int cluster_m = 1;        // KClusterM
    int cluster_n = 1;        // KClusterN
    int tmem_overlap = 0;     // KTmemOverlapCols
    int epi_regs = 208;       // KNumRegsEpilogue; 160 trades regs for occ.
    bool swap_ab = false;     // SwapAB: the activation moves to the MMA's N
    bool skip_fence = false;  // SkipTmemFence
    bool swizzle = false;     // UseTileSwizzle
    bool bias = false;        // AddBias

    // Dispatch policy, not kernel parameters.
    bool dense_only = false;  // only tuned/allowed for the dense launcher
    bool searchable = true;   // may the cost model pick it?

    constexpr bool pair() const { return cluster_m * cluster_n > 1; }
    // Activation-tile rows/cols; SwapAB swaps the two MMA dimensions.
    constexpr int act_m() const { return swap_ab ? block_n : block_m; }
    constexpr int act_n() const { return swap_ab ? block_m : block_n; }

    friend constexpr bool operator==(const TileSpec&,
                                     const TileSpec&) = default;
};

enum class MmaKind : int32_t {
    BF16 = 0,
    MXFP8 = 1,
    MXFP4 = 2,
    NVFP4 = 3,
};

template <MmaKind Kind>
struct MmaTraits;

template <>
struct MmaTraits<MmaKind::MXFP8> {
    using Operand = cutlass::float_e4m3_t;
    using Output = cutlass::bfloat16_t;
    using Scale = cutlass::float_ue8m0_t;
    static constexpr bool kFp4 = false;
    static constexpr int kSfVecSize = 16;
    static constexpr bool kHasAlpha = false;
    static constexpr const char* kName = "w8a8-mxfp8";
};

template <>
struct MmaTraits<MmaKind::NVFP4> {
    using Operand = cutlass::float_e2m1_t;
    using Output = cutlass::bfloat16_t;
    using Scale = cutlass::float_ue4m3_t;
    static constexpr bool kFp4 = true;
    static constexpr int kSfVecSize = 16;
    static constexpr bool kHasAlpha = true;
    static constexpr const char* kName = "w4a4-nvfp4";
};

template <>
struct MmaTraits<MmaKind::MXFP4> {
    using Operand = cutlass::float_e2m1_t;
    using Output = cutlass::bfloat16_t;
    using Scale = cutlass::float_ue8m0_t;
    static constexpr bool kFp4 = true;
    static constexpr int kSfVecSize = 32;
    static constexpr bool kHasAlpha = false;
    static constexpr const char* kName = "w4a4-mxfp4";
};

template <MmaKind Kind, KernelMode Mode, TileSpec S>
using MakeTile =
    GemmTile<S.block_m, S.block_n, S.block_k, S.main_stages, S.epi_stages,
             S.cluster_m, S.cluster_n, S.tmem_overlap, S.epi_regs, S.swizzle,
             S.skip_fence, S.swap_ab, MmaTraits<Kind>::kFp4,
             MmaTraits<Kind>::kSfVecSize, MmaTraits<Kind>::kHasAlpha, S.bias,
             Mode>;

// Fp8
constexpr TileSpec kFp8_128x256 = {.block_m = 128,
                                   .block_n = 256,
                                   .block_k = 128,
                                   .main_stages = 4,
                                   .tmem_overlap = 16};
constexpr TileSpec kFp8_256x256_pair = {.block_m = 256,
                                        .block_n = 256,
                                        .block_k = 128,
                                        .main_stages = 6,
                                        .cluster_m = 2,
                                        .tmem_overlap = 16};
constexpr TileSpec kFp8_256x128_pair = {.block_m = 256,
                                        .block_n = 128,
                                        .block_k = 128,
                                        .main_stages = 8,
                                        .cluster_m = 2,
                                        .dense_only = true};
constexpr TileSpec kFp8_128x128 = {.block_m = 128,
                                   .block_n = 128,
                                   .block_k = 128,
                                   .main_stages = 6,
                                   .dense_only = true};
constexpr TileSpec kFp8_256x192_pair_reg160 = {.block_m = 256,
                                               .block_n = 192,
                                               .block_k = 128,
                                               .main_stages = 7,
                                               .cluster_m = 2,
                                               .epi_regs = 160,
                                               .skip_fence = true,
                                               .dense_only = true};
constexpr TileSpec kFp8_128x64x256_swap = {.block_m = 128,
                                           .block_n = 64,
                                           .block_k = 256,
                                           .main_stages = 4,
                                           .epi_stages = 7,
                                           .swap_ab = true};
constexpr TileSpec kFp8_128x16x256_swap = {.block_m = 128,
                                           .block_n = 16,
                                           .block_k = 256,
                                           .main_stages = 5,
                                           .epi_stages = 7,
                                           .swap_ab = true,
                                           .dense_only = true,
                                           .searchable = false};
constexpr TileSpec kFp8_128x64 = {.block_m = 128,
                                  .block_n = 64,
                                  .block_k = 128,
                                  .main_stages = 9,
                                  .dense_only = true};
constexpr TileSpec kFp8_256x64_pair = {.block_m = 256,
                                       .block_n = 64,
                                       .block_k = 128,
                                       .main_stages = 8,
                                       .cluster_m = 2,
                                       .dense_only = true};
constexpr TileSpec kFp8_128x224 = {.block_m = 128,
                                   .block_n = 224,
                                   .block_k = 128,
                                   .main_stages = 4,
                                   .epi_regs = 160};
constexpr TileSpec kFp8_256x224_pair_reg160 = {.block_m = 256,
                                               .block_n = 224,
                                               .block_k = 128,
                                               .main_stages = 7,
                                               .cluster_m = 2,
                                               .epi_regs = 160,
                                               .skip_fence = true};
constexpr TileSpec kFp8_256x224_pair_reg160_swizzle = {.block_m = 256,
                                                       .block_n = 224,
                                                       .block_k = 128,
                                                       .main_stages = 7,
                                                       .cluster_m = 2,
                                                       .epi_regs = 160,
                                                       .skip_fence = true,
                                                       .swizzle = true,
                                                       .dense_only = true};
constexpr TileSpec kFp8_256x128_bias = {.block_m = 256,
                                        .block_n = 128,
                                        .block_k = 128,
                                        .main_stages = 6,
                                        .epi_stages = 3,
                                        .cluster_m = 2,
                                        .bias = true,
                                        .searchable = false};

// Fp4 (shared by NVFP4 and MXFP4)
constexpr TileSpec kFp4_128x256 = {.block_m = 128,
                                   .block_n = 256,
                                   .block_k = 256,
                                   .main_stages = 4,
                                   .tmem_overlap = 48};
constexpr TileSpec kFp4_128x256_reg160 = {.block_m = 128,
                                          .block_n = 256,
                                          .block_k = 256,
                                          .main_stages = 4,
                                          .tmem_overlap = 48,
                                          .epi_regs = 160,
                                          .skip_fence = true,
                                          .dense_only = true};
constexpr TileSpec kFp4_128x64x512_swap = {.block_m = 128,
                                           .block_n = 64,
                                           .block_k = 512,
                                           .main_stages = 4,
                                           .epi_stages = 7,
                                           .swap_ab = true};
constexpr TileSpec kFp4_128x64x512_swap_reg160 = {.block_m = 128,
                                                  .block_n = 64,
                                                  .block_k = 512,
                                                  .main_stages = 4,
                                                  .epi_stages = 7,
                                                  .epi_regs = 160,
                                                  .swap_ab = true,
                                                  .skip_fence = true,
                                                  .dense_only = true};
constexpr TileSpec kFp4_128x32x256_swap = {.block_m = 128,
                                           .block_n = 32,
                                           .block_k = 256,
                                           .main_stages = 9,
                                           .epi_stages = 7,
                                           .swap_ab = true,
                                           .dense_only = true};
constexpr TileSpec kFp4_128x16x256_swap_reg160 = {.block_m = 128,
                                                  .block_n = 16,
                                                  .block_k = 256,
                                                  .main_stages = 10,
                                                  .epi_stages = 7,
                                                  .epi_regs = 160,
                                                  .swap_ab = true,
                                                  .skip_fence = true,
                                                  .dense_only = true};
constexpr TileSpec kFp4_128x128x256_swap = {.block_m = 128,
                                            .block_n = 128,
                                            .block_k = 256,
                                            .main_stages = 6,
                                            .epi_stages = 3,
                                            .swap_ab = true,
                                            .dense_only = true};
constexpr TileSpec kFp4_256x128_pair = {.block_m = 256,
                                        .block_n = 128,
                                        .block_k = 256,
                                        .main_stages = 8,
                                        .epi_stages = 3,
                                        .cluster_m = 2,
                                        .dense_only = true};
constexpr TileSpec kFp4_256x256_pair = {.block_m = 256,
                                        .block_n = 256,
                                        .block_k = 256,
                                        .main_stages = 5,
                                        .cluster_m = 2,
                                        .tmem_overlap = 48};
constexpr TileSpec kFp4_256x256_pair_reg160 = {.block_m = 256,
                                               .block_n = 256,
                                               .block_k = 256,
                                               .main_stages = 5,
                                               .cluster_m = 2,
                                               .tmem_overlap = 48,
                                               .epi_regs = 160,
                                               .skip_fence = true,
                                               .dense_only = true};

// Candidate tiles per MmaKind.
inline constexpr std::array<TileSpec, 13> kFp8Tiles = {
    kFp8_128x256,
    kFp8_256x256_pair,
    kFp8_256x128_pair,
    kFp8_128x128,
    kFp8_256x192_pair_reg160,
    kFp8_128x64x256_swap,
    kFp8_128x16x256_swap,
    kFp8_128x64,
    kFp8_256x64_pair,
    kFp8_128x224,
    kFp8_256x224_pair_reg160,
    kFp8_256x224_pair_reg160_swizzle,
    kFp8_256x128_bias,
};

inline constexpr std::array<TileSpec, 10> kFp4Tiles = {
    kFp4_128x256,          kFp4_128x256_reg160,
    kFp4_128x64x512_swap,  kFp4_128x64x512_swap_reg160,
    kFp4_128x32x256_swap,  kFp4_128x16x256_swap_reg160,
    kFp4_128x128x256_swap, kFp4_256x128_pair,
    kFp4_256x256_pair,     kFp4_256x256_pair_reg160,
};

template <MmaKind Kind>
struct CandidateTiles;
template <>
struct CandidateTiles<MmaKind::MXFP8> {
    static constexpr const auto& tiles = kFp8Tiles;
};
template <>
struct CandidateTiles<MmaKind::NVFP4> {
    static constexpr const auto& tiles = kFp4Tiles;
};
template <>
struct CandidateTiles<MmaKind::MXFP4> {
    static constexpr const auto& tiles = kFp4Tiles;
};

// ===========================================================================
// BF16 (kind::f16) configs — no block-scale, no alpha. KBlockK=64 gives the
// same per-stage A/B smem as FP8@KBlockK=128 and there is no SFA/SFB smem, so
// stage counts can mirror the FP8 configs (< 228 KB). The tail three template
// parameters are BF16-relevant: EWGO (KEpilogueWarpGroupsOverride, 1 for the
// 2SM "2x2" TMEM layout and narrow tiles), NTAIL (KForceEpiNPredicate, the
// predicated store for N % KBlockN != 0), BF16 (KIsBF16).
// ---------------------------------------------------------------------------
// Dense (Naive)
using GemmKernelConfigBF16NaiveNoBias256x256x64_2CTA =
    GemmTile<256, 256, 64, 6, 2, 2, 1, 0, 208, false, false, false, false, 16,
             false, false, KernelMode::Dense, 0, false, true>;
using GemmKernelConfigBF16NaiveNoBias128x256x64_1CTA =
    GemmTile<128, 256, 64, 4, 2, 1, 1, 0, 208, false, false, false, false, 16,
             false, false, KernelMode::Dense, 0, false, true>;
using GemmKernelConfigBF16NaiveNoBias128x128x64_1CTA =
    GemmTile<128, 128, 64, 4, 2, 1, 1, 0, 208, false, false, false, false, 16,
             false, false, KernelMode::Dense, 0, false, true>;
using GemmKernelConfigBF16NaiveNoBias128x32x64_1CTA =
    GemmTile<128, 32, 64, 10, 2, 1, 1, 0, 208, false, false, false, false, 16,
             false, false, KernelMode::Dense, 0, false, true>;
using GemmKernelConfigBF16NaiveNoBias128x64x64_2x2CTA =
    GemmTile<128, 64, 64, 8, 2, 2, 2, 0, 208, false, false, false, false, 16,
             false, false, KernelMode::Dense, 1, false, true>;
using GemmKernelConfigBF16NaiveNoBias128x64x64_2x2CTA_P16 =
    GemmTile<128, 64, 64, 16, 2, 2, 2, 0, 208, false, false, false, false, 16,
             false, false, KernelMode::Dense, 1, false, true>;
using GemmKernelConfigBF16NaiveNoBias128x64x64_2x2CTA_P18 =
    GemmTile<128, 64, 64, 18, 2, 2, 2, 0, 208, false, false, false, false, 16,
             false, false, KernelMode::Dense, 1, false, true>;
using GemmKernelConfigBF16NaiveNoBias128x64x64_2CTA_P18 =
    GemmTile<128, 64, 64, 18, 2, 2, 1, 0, 208, false, false, false, false, 16,
             false, false, KernelMode::Dense, 1, false, true>;
using GemmKernelConfigBF16NaiveNoBias64x96x64_1CTA_P11 =
    GemmTile<64, 96, 64, 11, 2, 1, 1, 0, 208, false, false, false, false, 16,
             false, false, KernelMode::Dense, 1, false, true>;
using GemmKernelConfigBF16NaiveNoBias64x224x64_1CTA_P6 =
    GemmTile<64, 224, 64, 6, 2, 1, 1, 0, 208, false, false, false, false, 16,
             false, false, KernelMode::Dense, 1, false, true>;
using GemmKernelConfigBF16NaiveNoBias128x64x64_2x1CTA =
    GemmTile<128, 64, 64, 8, 2, 2, 1, 0, 208, false, false, false, false, 16,
             false, false, KernelMode::Dense, 1, false, true>;
using GemmKernelConfigBF16NaiveNoBias64x64x64_1CTA =
    GemmTile<64, 64, 64, 10, 2, 1, 1, 0, 208, false, false, false, false, 16,
             false, false, KernelMode::Dense, 0, false, true>;
using GemmKernelConfigBF16NaiveNoBias64x128x64_1CTA =
    GemmTile<64, 128, 64, 8, 2, 1, 1, 0, 208, false, false, false, false, 16,
             false, false, KernelMode::Dense, 0, false, true>;
using GemmKernelConfigBF16NaiveNoBias64x32x64_1WG_P14_1CTA =
    GemmTile<64, 32, 64, 14, 2, 1, 1, 0, 208, false, false, false, false, 16,
             false, false, KernelMode::Dense, 1, false, true>;
using GemmKernelConfigBF16NaiveNoBias64x32x64_1WG_P18_1CTA =
    GemmTile<64, 32, 64, 18, 2, 1, 1, 0, 208, false, false, false, false, 16,
             false, false, KernelMode::Dense, 1, false, true>;
using GemmKernelConfigBF16NaiveNoBias64x32x64_1WG_1CTA =
    GemmTile<64, 32, 64, 10, 2, 1, 1, 0, 208, false, false, false, false, 16,
             false, false, KernelMode::Dense, 1, false, true>;
using GemmKernelConfigBF16NaiveNoBias64x64x64_1WG_1CTA =
    GemmTile<64, 64, 64, 10, 2, 1, 1, 0, 208, false, false, false, false, 16,
             false, false, KernelMode::Dense, 1, false, true>;
using GemmKernelConfigBF16NaiveNoBias64x64x64_P12_1CTA =
    GemmTile<64, 64, 64, 12, 2, 1, 1, 0, 208, false, false, false, false, 16,
             false, false, KernelMode::Dense, 0, false, true>;
using GemmKernelConfigBF16NaiveNoBias64x64x64_P14_1CTA =
    GemmTile<64, 64, 64, 14, 2, 1, 1, 0, 208, false, false, false, false, 16,
             false, false, KernelMode::Dense, 0, false, true>;
using GemmKernelConfigBF16NaiveNoBias64x64x64_1WG_P14_1CTA =
    GemmTile<64, 64, 64, 14, 2, 1, 1, 0, 208, false, false, false, false, 16,
             false, false, KernelMode::Dense, 1, false, true>;
using GemmKernelConfigBF16NaiveNoBias128x32x64_P4_1CTA =
    GemmTile<128, 32, 64, 4, 2, 1, 1, 0, 208, false, false, false, false, 16,
             false, false, KernelMode::Dense, 0, false, true>;
using GemmKernelConfigBF16NaiveNoBias128x32x64_P6_1CTA =
    GemmTile<128, 32, 64, 6, 2, 1, 1, 0, 208, false, false, false, false, 16,
             false, false, KernelMode::Dense, 0, false, true>;
using GemmKernelConfigBF16NaiveNoBias128x32x32_1CTA =
    GemmTile<128, 32, 32, 16, 2, 1, 1, 0, 208, false, false, false, false, 16,
             false, false, KernelMode::Dense, 0, false, true>;
using GemmKernelConfigBF16NaiveNoBias128x64x64_1CTA =
    GemmTile<128, 64, 64, 8, 2, 1, 1, 0, 208, false, false, false, false, 16,
             false, false, KernelMode::Dense, 0, false, true>;
using GemmKernelConfigBF16NaiveNoBias256x128x64_2CTA =
    GemmTile<256, 128, 64, 8, 2, 2, 1, 0, 208, false, false, false, false, 16,
             false, false, KernelMode::Dense, 0, false, true>;
using GemmKernelConfigBF16NaiveNoBias256x128x64_2CTA_Ntail =
    GemmTile<256, 128, 64, 8, 2, 2, 1, 0, 208, false, false, false, false, 16,
             false, false, KernelMode::Dense, 0, true, true>;
using GemmKernelConfigBF16NaiveNoBias256x64x64_2CTA =
    GemmTile<256, 64, 64, 10, 2, 2, 1, 0, 208, false, false, false, false, 16,
             false, false, KernelMode::Dense, 0, false, true>;
using GemmKernelConfigBF16NaiveNoBias256x256x64_2CTA_P7 =
    GemmTile<256, 256, 64, 7, 2, 2, 1, 0, 208, false, false, false, false, 16,
             false, false, KernelMode::Dense, 0, false, true>;
using GemmKernelConfigBF16NaiveNoBias256x128x64_2CTA_P9 =
    GemmTile<256, 128, 64, 9, 2, 2, 1, 0, 208, false, false, false, false, 16,
             false, false, KernelMode::Dense, 0, false, true>;
using GemmKernelConfigBF16NaiveNoBias256x128x64_2CTA_P9_Ntail =
    GemmTile<256, 128, 64, 9, 2, 2, 1, 0, 208, false, false, false, false, 16,
             false, false, KernelMode::Dense, 0, true, true>;
using GemmKernelConfigBF16NaiveNoBias256x192x64_2CTA =
    GemmTile<256, 192, 64, 8, 2, 2, 1, 0, 208, false, false, false, false, 16,
             false, false, KernelMode::Dense, 0, false, true>;
using GemmKernelConfigBF16NaiveNoBias128x64x64_SwapAB_1CTA =
    GemmTile<128, 64, 64, 8, 2, 1, 1, 0, 208, false, false, true, false, 16,
             false, false, KernelMode::Dense, 0, false, true>;
using GemmKernelConfigBF16NaiveNoBias128x32x64_SwapAB_1CTA =
    GemmTile<128, 32, 64, 8, 2, 1, 1, 0, 208, false, false, true, false, 16,
             false, false, KernelMode::Dense, 0, false, true>;
using GemmKernelConfigBF16NaiveNoBias256x64x64_SwapAB_2CTA =
    GemmTile<256, 64, 64, 9, 2, 2, 1, 0, 208, false, false, true, false, 16,
             false, false, KernelMode::Dense, 0, false, true>;
using GemmKernelConfigBF16NaiveNoBias128x64x128_SwapAB_1CTA =
    GemmTile<128, 64, 128, 4, 2, 1, 1, 0, 208, false, false, true, false, 16,
             false, false, KernelMode::Dense, 0, false, true>;

// Batch
using GemmKernelConfigBF16BatchNoBias256x256x64_2CTA =
    GemmTile<256, 256, 64, 6, 2, 2, 1, 16, 208, false, false, false, false, 16,
             false, false, KernelMode::Batch, 0, false, true>;

using GemmKernelConfigBF16BatchNoBias128x256x64_1CTA =
    GemmTile<128, 256, 64, 4, 2, 1, 1, 16, 208, false, false, false, false, 16,
             false, false, KernelMode::Batch, 0, false, true>;

using GemmKernelConfigBF16BatchNoBias128x64x64_SwapAB_1CTA =
    GemmTile<128, 64, 64, 8, 2, 1, 1, 0, 208, false, false, true, false, 16,
             false, false, KernelMode::Batch, 0, false, true>;

// Group contiguous
using GemmKernelConfigBF16ContiguousNoBias256x256x64_2CTA =
    GemmTile<256, 256, 64, 7, 2, 2, 1, 0, 208, false, false, false, false, 16,
             false, false, KernelMode::Contiguous, 0, false, true>;

using GemmKernelConfigBF16ContiguousNoBias256x128x64_2CTA =
    GemmTile<256, 128, 64, 9, 2, 2, 1, 0, 208, false, false, false, false, 16,
             false, false, KernelMode::Contiguous, 0, false, true>;

using GemmKernelConfigBF16ContiguousNoBias128x256x64_1CTA =
    GemmTile<128, 256, 64, 4, 2, 1, 1, 16, 208, false, false, false, false, 16,
             false, false, KernelMode::Contiguous, 0, false, true>;

using GemmKernelConfigBF16ContiguousNoBias128x128x64_1CTA =
    GemmTile<128, 128, 64, 6, 2, 1, 1, 0, 208, false, false, false, false, 16,
             false, false, KernelMode::Contiguous, 0, false, true>;

using GemmKernelConfigBF16ContiguousNoBias128x64x64_1CTA =
    GemmTile<128, 64, 64, 8, 2, 1, 1, 16, 208, false, false, false, false, 16,
             false, false, KernelMode::Contiguous, 0, false, true>;

using GemmKernelConfigBF16ContiguousNoBias128x64x64_SwapAB_1CTA =
    GemmTile<128, 64, 64, 8, 2, 1, 1, 0, 208, false, false, true, false, 16,
             false, false, KernelMode::Contiguous, 0, false, true>;

// Group masked
using GemmKernelConfigBF16MaskedNoBias128x256x64_1CTA =
    GemmTile<128, 256, 64, 4, 2, 1, 1, 16, 208, false, false, false, false, 16,
             false, false, KernelMode::Masked, 0, false, true>;

using GemmKernelConfigBF16MaskedNoBias128x64x64_SwapAB_1CTA =
    GemmTile<128, 64, 64, 8, 2, 1, 1, 0, 208, false, false, true, false, 16,
             false, false, KernelMode::Masked, 0, false, true>;

using GemmKernelConfigBF16MaskedNoBias128x128x64_1CTA =
    GemmTile<128, 128, 64, 6, 2, 1, 1, 0, 208, false, false, false, false, 16,
             false, false, KernelMode::Masked, 0, false, true>;

using GemmKernelConfigBF16MaskedNoBias128x64x64_1CTA =
    GemmTile<128, 64, 64, 8, 2, 1, 1, 16, 208, false, false, false, false, 16,
             false, false, KernelMode::Masked, 0, false, true>;

using GemmKernelConfigBF16MaskedNoBias256x256x64_2CTA =
    GemmTile<256, 256, 64, 7, 2, 2, 1, 0, 208, false, false, false, false, 16,
             false, false, KernelMode::Masked, 0, false, true>;

using GemmKernelConfigBF16MaskedNoBias256x128x64_2CTA =
    GemmTile<256, 128, 64, 9, 2, 2, 1, 0, 208, false, false, false, false, 16,
             false, false, KernelMode::Masked, 0, false, true>;

// Field-lock asserts: guard against positional-argument drift in the
// 21-parameter alias lists above (migration from hand-written structs).
static_assert(GemmKernelConfigBF16NaiveNoBias256x256x64_2CTA::KBlockM == 256 &&
              GemmKernelConfigBF16NaiveNoBias256x256x64_2CTA::KBlockN == 256 &&
              GemmKernelConfigBF16NaiveNoBias256x256x64_2CTA::KBlockK == 64 &&
              GemmKernelConfigBF16NaiveNoBias256x256x64_2CTA::KPrologueStages ==
                  6 &&
              GemmKernelConfigBF16NaiveNoBias256x256x64_2CTA::IsPairUmma &&
              GemmKernelConfigBF16NaiveNoBias256x256x64_2CTA::KIsBF16);
static_assert(
    GemmKernelConfigBF16NaiveNoBias256x128x64_2CTA_Ntail::KForceEpiNPredicate &&
    GemmKernelConfigBF16NaiveNoBias256x128x64_2CTA_Ntail::KPrologueStages ==
        8 &&
    GemmKernelConfigBF16NaiveNoBias256x128x64_2CTA_P9_Ntail::KPrologueStages ==
        9 &&
    GemmKernelConfigBF16NaiveNoBias256x128x64_2CTA_P9_Ntail::
        KForceEpiNPredicate);
static_assert(
    GemmKernelConfigBF16NaiveNoBias128x64x64_2x2CTA_P18::
            KEpilogueWarpGroupsOverride == 1 &&
    GemmKernelConfigBF16NaiveNoBias128x64x64_2x2CTA_P18::KPrologueStages ==
        18 &&
    GemmKernelConfigBF16NaiveNoBias128x64x64_2x2CTA_P18::KClusterN == 2);
static_assert(GemmKernelConfigBF16NaiveNoBias128x32x32_1CTA::KBlockK == 32 &&
              GemmKernelConfigBF16NaiveNoBias128x64x128_SwapAB_1CTA::KBlockK ==
                  128 &&
              GemmKernelConfigBF16NaiveNoBias128x64x128_SwapAB_1CTA::SwapAB);
static_assert(
    GemmKernelConfigBF16ContiguousNoBias128x256x64_1CTA::KTmemOverlapCols ==
        16 &&
    GemmKernelConfigBF16MaskedNoBias256x128x64_2CTA::KPrologueStages == 9 &&
    GemmKernelConfigBF16BatchNoBias128x64x64_SwapAB_1CTA::SwapAB);

}  // namespace halops
