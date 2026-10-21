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

template <int BM, int BN, int BK, int PST, int EST, int CM = 1, int CN = 1,
          int OVLP = 0, int REG = 208, bool SWZ = false, bool FENCE = false,
          bool SWAP = false, bool FP4 = false, int SFVS = 16,
          bool ALPHA = false, bool BIAS = false,
          KernelMode MODE = KernelMode::Dense>
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
    static constexpr bool IsPairUmma = (CM * CN) > 1;
};

struct TileSpec {
    int block_m = 0;           // KBlockM
    int block_n = 0;           // KBlockN
    int block_k = 0;           // KBlockK
    int main_stages = 2;       // KPrologueStages
    int epi_stages = 2;        // KEpilogueStages
    int cluster_m = 1;         // KClusterM
    int cluster_n = 1;         // KClusterN
    int tmem_overlap = 0;      // KTmemOverlapCols
    int epi_regs = 208;        // KNumRegsEpilogue; 160 trades regs for occ.
    bool swap_ab = false;      // SwapAB: the activation moves to the MMA's N
    bool skip_fence = false;   // SkipTmemFence
    bool swizzle = false;      // UseTileSwizzle
    bool bias = false;         // AddBias

    // Dispatch policy, not kernel parameters.
    bool dense_only = false;   // only tuned/allowed for the dense launcher
    bool searchable = true;    // may the cost model pick it?

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
    static constexpr const char* kName = "w8a8";
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
using MakeTile = GemmTile<S.block_m, S.block_n, S.block_k, S.main_stages,
                          S.epi_stages, S.cluster_m, S.cluster_n,
                          S.tmem_overlap, S.epi_regs, S.swizzle, S.skip_fence,
                          S.swap_ab, MmaTraits<Kind>::kFp4,
                          MmaTraits<Kind>::kSfVecSize,
                          MmaTraits<Kind>::kHasAlpha, S.bias, Mode>;

// Fp8
constexpr TileSpec kFp8_128x256 = {
    .block_m = 128, .block_n = 256, .block_k = 128, .main_stages = 4,
    .tmem_overlap = 16};
constexpr TileSpec kFp8_256x256_pair = {
    .block_m = 256, .block_n = 256, .block_k = 128, .main_stages = 6,
    .cluster_m = 2, .tmem_overlap = 16};
constexpr TileSpec kFp8_256x128_pair = {
    .block_m = 256, .block_n = 128, .block_k = 128, .main_stages = 8,
    .cluster_m = 2, .dense_only = true};
constexpr TileSpec kFp8_128x128 = {
    .block_m = 128, .block_n = 128, .block_k = 128, .main_stages = 6,
    .dense_only = true};
constexpr TileSpec kFp8_256x192_pair_reg160 = {
    .block_m = 256, .block_n = 192, .block_k = 128, .main_stages = 7,
    .cluster_m = 2, .epi_regs = 160, .skip_fence = true, .dense_only = true};
constexpr TileSpec kFp8_128x64x256_swap = {
    .block_m = 128, .block_n = 64, .block_k = 256, .main_stages = 4,
    .epi_stages = 7, .swap_ab = true};
constexpr TileSpec kFp8_128x16x256_swap = {
    .block_m = 128, .block_n = 16, .block_k = 256, .main_stages = 5,
    .epi_stages = 7, .swap_ab = true, .dense_only = true, .searchable = false};
constexpr TileSpec kFp8_128x64 = {
    .block_m = 128, .block_n = 64, .block_k = 128, .main_stages = 9,
    .dense_only = true};
constexpr TileSpec kFp8_256x64_pair = {
    .block_m = 256, .block_n = 64, .block_k = 128, .main_stages = 8,
    .cluster_m = 2, .dense_only = true};
constexpr TileSpec kFp8_128x224 = {
    .block_m = 128, .block_n = 224, .block_k = 128, .main_stages = 4,
    .epi_regs = 160};
constexpr TileSpec kFp8_256x224_pair_reg160 = {
    .block_m = 256, .block_n = 224, .block_k = 128, .main_stages = 7,
    .cluster_m = 2, .epi_regs = 160, .skip_fence = true};
constexpr TileSpec kFp8_256x224_pair_reg160_swizzle = {
    .block_m = 256, .block_n = 224, .block_k = 128, .main_stages = 7,
    .cluster_m = 2, .epi_regs = 160, .skip_fence = true, .swizzle = true,
    .dense_only = true};
constexpr TileSpec kFp8_256x128_bias = {
    .block_m = 256, .block_n = 128, .block_k = 128, .main_stages = 6,
    .epi_stages = 3, .cluster_m = 2, .bias = true, .searchable = false};

// Fp4 (shared by NVFP4 and MXFP4)
constexpr TileSpec kFp4_128x256 = {
    .block_m = 128, .block_n = 256, .block_k = 256, .main_stages = 4,
    .tmem_overlap = 48};
constexpr TileSpec kFp4_128x256_reg160 = {
    .block_m = 128, .block_n = 256, .block_k = 256, .main_stages = 4,
    .tmem_overlap = 48, .epi_regs = 160, .skip_fence = true,
    .dense_only = true};
constexpr TileSpec kFp4_128x64x512_swap = {
    .block_m = 128, .block_n = 64, .block_k = 512, .main_stages = 4,
    .epi_stages = 7, .swap_ab = true};
constexpr TileSpec kFp4_128x64x512_swap_reg160 = {
    .block_m = 128, .block_n = 64, .block_k = 512, .main_stages = 4,
    .epi_stages = 7, .epi_regs = 160, .swap_ab = true, .skip_fence = true,
    .dense_only = true};
constexpr TileSpec kFp4_128x32x256_swap = {
    .block_m = 128, .block_n = 32, .block_k = 256, .main_stages = 9,
    .epi_stages = 7, .swap_ab = true, .dense_only = true};
constexpr TileSpec kFp4_128x16x256_swap_reg160 = {
    .block_m = 128, .block_n = 16, .block_k = 256, .main_stages = 10,
    .epi_stages = 7, .epi_regs = 160, .swap_ab = true, .skip_fence = true,
    .dense_only = true};
constexpr TileSpec kFp4_128x128x256_swap = {
    .block_m = 128, .block_n = 128, .block_k = 256, .main_stages = 6,
    .epi_stages = 3, .swap_ab = true, .dense_only = true};
constexpr TileSpec kFp4_256x128_pair = {
    .block_m = 256, .block_n = 128, .block_k = 256, .main_stages = 8,
    .epi_stages = 3, .cluster_m = 2, .dense_only = true};
constexpr TileSpec kFp4_256x256_pair = {
    .block_m = 256, .block_n = 256, .block_k = 256, .main_stages = 5,
    .cluster_m = 2, .tmem_overlap = 48};
constexpr TileSpec kFp4_256x256_pair_reg160 = {
    .block_m = 256, .block_n = 256, .block_k = 256, .main_stages = 5,
    .cluster_m = 2, .tmem_overlap = 48, .epi_regs = 160, .skip_fence = true,
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
    kFp4_128x256,
    kFp4_128x256_reg160,
    kFp4_128x64x512_swap,
    kFp4_128x64x512_swap_reg160,
    kFp4_128x32x256_swap,
    kFp4_128x16x256_swap_reg160,
    kFp4_128x128x256_swap,
    kFp4_256x128_pair,
    kFp4_256x256_pair,
    kFp4_256x256_pair_reg160,
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

}  // namespace halops
