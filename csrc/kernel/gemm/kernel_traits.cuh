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

#include <numeric>

#include "common/mx_quant.cuh"
#include "kernel/gemm/gemm_tile.cuh"
#include "kernel/gemm/utils.cuh"

namespace halops {

struct GemmParams {
    void *__restrict__ ptr_a = nullptr;
    void *__restrict__ ptr_b = nullptr;
    void *__restrict__ ptr_c = nullptr;
    void *__restrict__ ptr_d = nullptr;
    void *__restrict__ ptr_sfa = nullptr;
    void *__restrict__ ptr_sfb = nullptr;
    float const *ptr_alpha = nullptr;
    int m = 0;
    int n = 0;
    int k = 0;
    int num_experts = 1;
    int max_m_per_expert = 0;
    int *__restrict__ masked_m = nullptr;
    int *__restrict__ expert_ids = nullptr;
    bool use_pdl = false;
    void *__restrict__ ptr_sfd = nullptr;
    void const *__restrict__ ptr_gate = nullptr;
    int64_t gate_ld = 0;
    SituCoef situ;
};

template <class AccTensor, class SfaTensor, class SfbTensor>
struct TmemStorage {
    AccTensor tmem_accumulators;
    SfaTensor tmem_SFA;
    SfbTensor tmem_SFB;
};

template <typename TypeA_ = cutlass::float_e4m3_t,
          typename TypeB_ = cutlass::float_e4m3_t,
          typename TypeC_ = cutlass::bfloat16_t,
          typename TypeD_ = cutlass::bfloat16_t,
          typename TypeSF_ = cutlass::float_ue4m3_t,
          typename TypeAccumulate_ = float,
          typename KernelConfig_ = GemmKernelConfig, bool NarrowA_ = false,
          bool UsePdl_ = false>
struct GemmKernelTraits {
    using TypeA = TypeA_;
    using TypeB = TypeB_;
    using TypeC = TypeC_;
    using TypeD = TypeD_;
    using TypeSF = TypeSF_;
    using TypeAccumulate = TypeAccumulate_;

    static constexpr KernelMode KernelType = KernelConfig_::KernelType;
    static constexpr int KPrologueStages = KernelConfig_::KPrologueStages;
    static constexpr int KEpilogueStages = KernelConfig_::KEpilogueStages;
    static constexpr int KTmemOverlapCols = KernelConfig_::KTmemOverlapCols;
    static constexpr bool UseTileSwizzle = KernelConfig_::UseTileSwizzle;
    static constexpr int KBlockM = KernelConfig_::KBlockM;
    static constexpr int KBlockN = KernelConfig_::KBlockN;
    static constexpr int KBlockK = KernelConfig_::KBlockK;
    static constexpr bool AddBias = KernelConfig_::AddBias;
    static constexpr bool SwapAB = KernelConfig_::SwapAB;
    static constexpr bool HasAlpha = KernelConfig_::HasAlpha;
    static constexpr bool KDenseW8A8MXFP8 =
        KernelType == KernelMode::Dense && !AddBias && !KernelConfig_::KIsFP4;
    static constexpr bool KIsDense =
        KernelType == KernelMode::Dense && !AddBias;
    static constexpr int KMaxAlphaExperts = 256;
    static constexpr bool IsPairUmma = KernelConfig_::IsPairUmma;
    static constexpr int KNumCTA = IsPairUmma ? 2 : 1;
    static constexpr int KClusterM = KernelConfig_::KClusterM;
    static constexpr int KClusterN = KernelConfig_::KClusterN;
    // KNumCTA 1: cta_group::1 UMMA; 2: pair-UMMA (2x1SM, cta_group::2).
    // MMA K step: 64 for FP4, 32 for FP8, 16 for BF16.
    static constexpr bool KIsFP4 = KernelConfig_::KIsFP4;
    static constexpr bool KIsBF16 = KernelConfig_::KIsBF16;
    // Narrow the A TMA box to the real M below KBlockM (the UMMA still reads
    // KBlockM rows; the epilogue drops D past M).
    static constexpr bool KNarrowA =
        NarrowA_ &&
        (KDenseW8A8MXFP8 || (KernelType == KernelMode::Batch && !AddBias)) &&
        !KIsBF16 &&
        !SwapAB && !IsPairUmma && !KernelConfig_::KIsFP4 && KBlockK == 128;
    static constexpr bool KUsePdl = UsePdl_;
    static constexpr int KMmaK = KIsFP4 ? 64 : (KIsBF16 ? 16 : 32);
    // Tail-only stream-K: only the BF16 dense no-bias no-alpha epilogue can
    // deposit an fp32 partial. SwapAB is excluded (the tail fold-back assumes
    // non-swap traversal; m=64 n=19584 k=1024 gave 121% rel error), as are
    // N-predicated tiles (their partial slot leaves holes the reduce sums).
    static constexpr bool KSupportsStreamK =
        KernelType == KernelMode::Dense && KIsBF16 && !AddBias && !HasAlpha &&
        !SwapAB && (KBlockN % 128 == 0 || KBlockN == 32 || KBlockN == 64) &&
        !KernelConfig_::KForceEpiNPredicate;
    static constexpr int kItersK = KBlockK / KMmaK;
    static constexpr bool KBFp4Unpack =
        !KIsFP4 && cute::sizeof_bits_v<TypeB> == 4;
    using TypeB_Mma =
        cute::conditional_t<KBFp4Unpack,
                            cutlass::detail::float_e2m1_unpacksmem_t, TypeB>;
    using TypeB_Smem = cute::conditional_t<KBFp4Unpack, uint8_t, TypeB>;
    static constexpr bool KAFp4Unpack =
        !KIsFP4 && cute::sizeof_bits_v<TypeA> == 4;
    using TypeA_Mma =
        cute::conditional_t<KAFp4Unpack,
                            cutlass::detail::float_e2m1_unpacksmem_t, TypeA>;
    using TypeA_Smem = cute::conditional_t<KAFp4Unpack, uint8_t, TypeA>;
    static constexpr EpilogueOp KEpiOp = KernelConfig_::KEpiOp;
    static constexpr bool KQuantD = IsQuantOp(KEpiOp);
    static constexpr bool KGateSigmoid =
        KEpiOp == EpilogueOp::SigmoidGateMulQuant;
    static constexpr bool KGateMul =
        KEpiOp == EpilogueOp::GateMulQuant || KGateSigmoid;
    static constexpr bool KGlu = KEpiOp == EpilogueOp::SituQuant;
    static constexpr int KAccColsPerOut = KGlu ? 2 : 1;
    static constexpr int KQuantAccCols = KAccColsPerOut * mxfp8::kGroup;
    // Epilogue warp groups split N: 1 for bf16 SwapAB (small N) and for a
    // quantized tile narrower than two quant groups, 2 otherwise.
    // KEpilogueWarpGroupsOverride forces 1, e.g. to make KBlockN=32 legal at
    // CtaM==64 (KEpiComputeN=32 meets the T2R atom floor).
    static constexpr int KEpilogueWarpGroupCount =
        KernelConfig_::KEpilogueWarpGroupsOverride != 0
            ? KernelConfig_::KEpilogueWarpGroupsOverride
            : (((SwapAB && !KQuantD) ||
                (KQuantD && !SwapAB && KBlockN < 2 * KQuantAccCols))
                   ? 1
                   : 2);
    static constexpr int KEpilogueWarpStartId = 0;
    static constexpr int KEpilogueWarpEndId = KEpilogueWarpGroupCount * 4 - 1;
    static constexpr int KLoadABWarpId = KEpilogueWarpEndId + 1;
    static constexpr int KMmaWarpId = KLoadABWarpId + 1;
    static constexpr int KLoadCWarpId = KMmaWarpId + 1;
    static constexpr int KNumWarp = get_num_warps(KLoadCWarpId, AddBias);
    static constexpr int KThreads = 32 * KNumWarp;
    static constexpr int KNumRegsEpilogue = KernelConfig_::KNumRegsEpilogue;
    static constexpr bool SkipTmemFence = KernelConfig_::SkipTmemFence;
    static constexpr int KNumRegsOther = 88;
    // setmaxnreg redistribution (208 + 88 split) for BF16 needs enough warps;
    // without it ptxas gives every warp one flat budget (54-87 regs), starving
    // the epilogue. Quantized configs keep the warp-group-count rule.
    static constexpr bool KRedistributeRegs =
        KIsBF16 ? (KNumWarp >= 8 && (!SwapAB || KernelType == KernelMode::Dense))
                : KEpilogueWarpGroupCount != 1;

    using TileShape =
        cute::Shape<cute::Int<KBlockM>, cute::Int<KBlockN>, cute::Int<KBlockK>>;
    using ClusterShape_MNK =
        cute::Shape<cute::Int<KClusterM>, cute::Int<KClusterN>, cute::Int<1>>;
    // Per-CTA output tile: each CTA of a pair-UMMA owns KBlockM / 2 rows.
    using EpilogueTile =
        cute::Shape<cute::Int<KBlockM / KNumCTA>, cute::Int<KBlockN>>;
    static constexpr int KEpiComputeN = KBlockN / KEpilogueWarpGroupCount;
    using EpiComputeTile =
        cute::Shape<cute::Int<KBlockM / KNumCTA>, cute::Int<KEpiComputeN>>;
    // Overlap TMEM loads use 16-wide sub-tiles (the TMEM load atom width);
    // KTmemOverlapCols need not divide KEpiComputeN, only both be multiples of
    // 16. Without overlap this is EpiComputeTile.
    //
    // Non-power-of-2 KBlockN (96/112/224) has no T2R atom for KEpiComputeN and
    // splits into 32-wide sub-tiles (16 gives only 8 fp32/thread, too few for
    // SM100_STORE_256bit). Not legal on the 2SM "2x2" layout: a 32-wide slice
    // straddles the N-half boundary and writes wrong elements.
    static constexpr bool KEpiNIsSingleAtom =
        (KEpiComputeN == 16 || KEpiComputeN == 32 || KEpiComputeN == 64 ||
         KEpiComputeN == 128);
    // Quantized dtypes keep main's rule: 16-wide unless the per-WG width has
    // its own atom (8 included).
    static constexpr bool KEpiComputeNHasAtom =
        KEpiNIsSingleAtom || KEpiComputeN == 8;
    static constexpr int KSubTileN =
        KIsBF16 ? (KTmemOverlapCols != 0
                       ? 16
                       : (KEpiNIsSingleAtom
                              ? KEpiComputeN
                              : (KEpiComputeN % 32 == 0 ? 32 : 16)))
                : ((KTmemOverlapCols == 0 && KEpiComputeNHasAtom) ? KEpiComputeN
                                                                  : 16);
    static_assert(!KIsBF16 || KEpiNIsSingleAtom ||
                      !(KNumCTA == 2 && KBlockM == 128),
                  "a non-power-of-2 KBlockN cannot use the 2SM 128-row \"2x2\" "
                  "TMEM layout: its N dimension is (KBlockN/2, 2) over two "
                  "datapath halves, so the 32-wide epilogue sub-tiles straddle "
                  "the half boundary and the store lands on wrong elements "
                  "(measured 67.85% mismatch). Use a 1-CTA KBlockM=64 config, "
                  "whose 16-datapath layout maps N straight to columns.");
    using OverlapComputeTile = cute::conditional_t<
        KSubTileN == KEpiComputeN, EpiComputeTile,
        cute::Shape<cute::Int<KBlockM / KNumCTA>, cute::Int<KSubTileN>>>;
    static constexpr int KSubTilesPerWG = KEpiComputeN / KSubTileN;
    // Overlap sub-tiles sit at the boundary between adjacent stages.
    static constexpr int KOverlapSubCount = KTmemOverlapCols / KSubTileN;
    // Overlap region fits one TMEM load (widths with a hardware op), so MMA can
    // resume right after T2R; other widths (e.g. 48) interleave per sub-tile.
    static constexpr bool KOverlapSingleLoad =
        (KTmemOverlapCols > 0) &&
        (KTmemOverlapCols == 16 || KTmemOverlapCols == 32 ||
         KTmemOverlapCols == 64 || KTmemOverlapCols == 128);

    // TMEM budget: tcgen05.alloc takes a power-of-2 column count in [32, 512];
    // accumulators need KEpilogueStages * KBlockN. Only BF16 without overlap
    // right-sizes; block-scaled dtypes also hold SFA/SFB in TMEM and overlap
    // shares columns between stages.
    static constexpr int KAccTmemColumns = KEpilogueStages * KBlockN;
    static constexpr int round_up_tmem_columns(int v) {
        int r = 32;
        while (r < v) {
            r *= 2;
        }
        return r > 512 ? 512 : r;
    }
    static constexpr int KTmemColumns =
        (KIsBF16 && KTmemOverlapCols == 0)
            ? round_up_tmem_columns(KAccTmemColumns)
            : 512;

    // T2R atom selection depends on per-CTA M and N. NNdp is the number of
    // datapath lanes an atom addresses: a 1SM 64-row accumulator fills only 16
    // lanes per subpartition, so it needs a 16dp atom (32dp fails
    // "Non-injective Layout") with the N repeat halved. Both use 128 threads.
    static constexpr int KEpiCtaM = KBlockM / KNumCTA;
    static_assert(KEpiCtaM == 64 || KEpiCtaM == 128,
                  "UMMA accumulators support 64 or 128 rows per CTA only "
                  "(mma_traits_sm100.hpp:502).");
    // 16dp32b8x is too narrow for the 256-bit store, so a 64-row CTA tile
    // needs KEpiComputeN >= 32.
    static_assert(KEpiCtaM == 128 || KEpiComputeN >= 32,
                  "CtaM==64 needs a 16dp TMEM load atom, whose N repeat count "
                  "is half the 32dp one; KEpiComputeN < 32 would land on "
                  "16dp32b8x, which cannot fill SM100_STORE_256bit.");

    // Accumulator layout being read:
    //   1CTA M=64  -> half subpartition, 16 lanes, N in columns (16dp atom);
    //   2CTA M=128 -> "2x2", second N half on datapaths 64-127, N/2 columns;
    //   2CTA M=256 -> "4x1", all 128 lanes, N in columns.
    // "2x2" needs a 32dp atom and a full-KBlockN epilogue tile
    // (KEpilogueWarpGroupsOverride=1), else warps 2/3 address outside it.
    static constexpr bool KIs2SmHalfM = (KNumCTA == 2) && (KEpiCtaM == 64);
    static constexpr int KEpiTmemDp =
        (KNumCTA == 1 && KEpiCtaM == 64) ? 16 : 32;
    static_assert(!KIs2SmHalfM || KEpiComputeN == KBlockN,
                  "2SM accumulators with KBlockM=128 use the 2x2 TMEM layout, "
                  "whose second N half sits on datapaths 64-127; the epilogue "
                  "tile must cover the full KBlockN (set "
                  "KEpilogueWarpGroupsOverride = 1) or warps 2/3 address "
                  "outside the tile and the output is silently wrong.");

    // CopyOp for the whole overlap region when KOverlapSingleLoad; width
    // matched to KTmemOverlapCols, else 16x.
    using CopyOpT2R_Overlap = cute::conditional_t<
        KEpiTmemDp == 16,
        // 16dp: N repeat halved relative to the 32dp column below.
        cute::conditional_t<
            KTmemOverlapCols == 16, cute::SM100_TMEM_LOAD_16dp32b8x,
            cute::conditional_t<
                KTmemOverlapCols == 32, cute::SM100_TMEM_LOAD_16dp32b16x,
                cute::conditional_t<KTmemOverlapCols == 64,
                                    cute::SM100_TMEM_LOAD_16dp32b32x,
                                    cute::SM100_TMEM_LOAD_16dp32b64x>>>,
        cute::conditional_t<
            KTmemOverlapCols == 16, cute::SM100_TMEM_LOAD_32dp32b16x,
            cute::conditional_t<
                KTmemOverlapCols == 32, cute::SM100_TMEM_LOAD_32dp32b32x,
                cute::conditional_t<KTmemOverlapCols == 64,
                                    cute::SM100_TMEM_LOAD_32dp32b64x,
                                    cute::SM100_TMEM_LOAD_32dp32b128x>>>>;

    static constexpr int KOverlapLoadN =
        KOverlapSingleLoad ? KTmemOverlapCols
                           : (KTmemOverlapCols == 0 ? 16 : KSubTileN);
    using OverlapLoadTile =
        cute::Shape<cute::Int<KBlockM / KNumCTA>, cute::Int<KOverlapLoadN>>;

    static constexpr int KWideTilesTotal = KBlockN / KOverlapLoadN;

    // T2R op for EpiComputeTile N (and CtaM). `Nx` is the column count; a
    // mismatched width moves the wrong columns, so every reachable width needs
    // its own arm. The 16dp series (CtaM=64) uses half-width atoms.
    using CopyOpT2R = cute::conditional_t<
        KEpiTmemDp == 16,
        cute::conditional_t<
            KEpiComputeN == 128, cute::SM100_TMEM_LOAD_16dp32b64x,
            cute::conditional_t<KEpiComputeN == 64,
                                cute::SM100_TMEM_LOAD_16dp32b32x,
                                cute::SM100_TMEM_LOAD_16dp32b16x>>,
        cute::conditional_t<
            KIs2SmHalfM,
            // 2SM "2x2": N/2 columns are used, so the repeat is halved.
            cute::conditional_t<
                KEpiComputeN == 128, cute::SM100_TMEM_LOAD_32dp32b64x,
                cute::conditional_t<KEpiComputeN == 64,
                                    cute::SM100_TMEM_LOAD_32dp32b32x,
                                    cute::SM100_TMEM_LOAD_32dp32b16x>>,
            cute::conditional_t<
                KEpiComputeN == 128, cute::SM100_TMEM_LOAD_32dp32b128x,
                cute::conditional_t<
                    KEpiComputeN == 64, cute::SM100_TMEM_LOAD_32dp32b64x,
                    cute::conditional_t<
                        KEpiComputeN == 32, cute::SM100_TMEM_LOAD_32dp32b32x,
                        cute::conditional_t<
                            KEpiComputeN == 8, cute::SM100_TMEM_LOAD_32dp32b8x,
                            cute::SM100_TMEM_LOAD_32dp32b16x>>>>>>;
    // Overlap: 16-wide sub-tile load (16 fp32/thread fills SM100_STORE_256bit;
    // 8x is too few). No overlap: width-matched CopyOpT2R. CtaM=64 halves the
    // repeat. Columns per sub-tile: N/2 on 16dp and "2x2", N on 4x1.
    static constexpr int KSubTileTmemCols =
        (KEpiTmemDp == 16 || KIs2SmHalfM) ? KSubTileN / 2 : KSubTileN;
    // Width-matched atom for one sub-tile; only for non-power-of-2 KBlockN.
    using CopyOpT2R_SubWide = cute::conditional_t<
        KEpiTmemDp == 16,
        cute::conditional_t<
            KSubTileTmemCols == 64, cute::SM100_TMEM_LOAD_16dp32b64x,
            cute::conditional_t<
                KSubTileTmemCols == 32, cute::SM100_TMEM_LOAD_16dp32b32x,
                cute::conditional_t<KSubTileTmemCols == 16,
                                    cute::SM100_TMEM_LOAD_16dp32b16x,
                                    cute::SM100_TMEM_LOAD_16dp32b8x>>>,
        cute::conditional_t<
            KSubTileTmemCols == 128, cute::SM100_TMEM_LOAD_32dp32b128x,
            cute::conditional_t<
                KSubTileTmemCols == 64, cute::SM100_TMEM_LOAD_32dp32b64x,
                cute::conditional_t<
                    KSubTileTmemCols == 32, cute::SM100_TMEM_LOAD_32dp32b32x,
                    cute::conditional_t<KSubTileTmemCols == 16,
                                        cute::SM100_TMEM_LOAD_32dp32b16x,
                                        cute::SM100_TMEM_LOAD_32dp32b8x>>>>>;
    using CopyOpT2R_Sub = cute::conditional_t<
        !KIsBF16,
        cute::conditional_t<KSubTileN == KEpiComputeN, CopyOpT2R,
                            cute::SM100_TMEM_LOAD_32dp32b16x>,
        cute::conditional_t<
            KTmemOverlapCols == 0,
            cute::conditional_t<KEpiNIsSingleAtom, CopyOpT2R,
                                CopyOpT2R_SubWide>,
            cute::conditional_t<KEpiTmemDp == 16,
                                cute::SM100_TMEM_LOAD_16dp32b8x,
                                cute::SM100_TMEM_LOAD_32dp32b16x>>>;

    // Each chain above ends in the narrowest atom; make_tmem_copy replicates
    // it over the tile, which is only correct when the atom width divides it.
    static_assert(!KIsBF16 || KEpiComputeN % KSubTileN == 0,
                  "the epilogue sub-tile loop must cover KEpiComputeN exactly");
    static_assert(!KIsBF16 || KEpiNIsSingleAtom || KTmemOverlapCols != 0 ||
                      KSubTileTmemCols % 16 == 0,
                  "KEpiComputeN outside {16,32,64,128} falls through to the "
                  "16-column atom; make_tmem_copy can only replicate that over "
                  "the tile when the tile's TMEM column count is a multiple of "
                  "16. Widen the table before using such a width.");

    // Non-power-of-2 KBlockN may overhang N, so the store is predicated at
    // KSubTileN granularity (n must be a multiple of it). KForceEpiNPredicate
    // forces this for a power-of-2 KBlockN (N=1600 on a 128-wide tile).
    static constexpr bool KEpiNeedsNPredicate =
        KIsBF16 && (((KBlockN & (KBlockN - 1)) != 0) ||
                    KernelConfig_::KForceEpiNPredicate);

    // Scale-factor vector size: 16 = NVFP4, 32 = MXFP4.
    static constexpr int KSFVecSize = KernelConfig_::KSfVecSize;

    // Real MMA atom used for the actual A@B computation.
    //   FP4  -> SM100_MMA_MXF4_SS / _2x1SM_SS   (block-scaled, has TypeSF)
    //   FP8  -> SM100_MMA_MXF8F6F4_SS / _2x1SM_SS (block-scaled, has TypeSF)
    //   BF16 -> SM100_MMA_F16BF16_SS / _2x1SM_SS  (NON block-scaled, no SF)
    using MMA_T = cute::conditional_t<
        KIsBF16,
        cute::conditional_t<
            KNumCTA == 1,
            cute::SM100_MMA_F16BF16_SS<
                TypeA_Mma, TypeB_Mma, TypeAccumulate_,
                cute::get<0>(TileShape{}), cute::get<1>(TileShape{}),
                cute::UMMA::Major::K, cute::UMMA::Major::K>,
            cute::SM100_MMA_F16BF16_2x1SM_SS<
                TypeA_Mma, TypeB_Mma, TypeAccumulate_,
                cute::get<0>(TileShape{}), cute::get<1>(TileShape{}),
                cute::UMMA::Major::K, cute::UMMA::Major::K>>,
        cute::conditional_t<
            KIsFP4,
            cute::conditional_t<
                KNumCTA == 1,
                cute::SM100_MMA_MXF4_SS<
                    TypeA_Mma, TypeB_Mma, TypeAccumulate_, TypeSF,
                    cute::get<0>(TileShape{}), cute::get<1>(TileShape{}),
                    KSFVecSize, cute::UMMA::Major::K, cute::UMMA::Major::K>,
                cute::SM100_MMA_MXF4_2x1SM_SS<
                    TypeA_Mma, TypeB_Mma, TypeAccumulate_, TypeSF,
                    cute::get<0>(TileShape{}), cute::get<1>(TileShape{}),
                    KSFVecSize, cute::UMMA::Major::K, cute::UMMA::Major::K>>,
            cute::conditional_t<
                KNumCTA == 1,
                cute::SM100_MMA_MXF8F6F4_SS<
                    TypeA_Mma, TypeB_Mma, TypeAccumulate_, TypeSF,
                    cute::get<0>(TileShape{}), cute::get<1>(TileShape{}),
                    cute::UMMA::Major::K, cute::UMMA::Major::K>,
                cute::SM100_MMA_MXF8F6F4_2x1SM_SS<
                    TypeA_Mma, TypeB_Mma, TypeAccumulate_, TypeSF,
                    cute::get<0>(TileShape{}), cute::get<1>(TileShape{}),
                    cute::UMMA::Major::K, cute::UMMA::Major::K>>>>;
    using TiledMma = decltype(cute::make_tiled_mma(MMA_T{}));

    CUTE_STATIC_ASSERT_V(cute::evenly_divides(TileShape{},
                                              cute::tile_shape(TiledMma{})),
                         "Static cluster shape used: TileShape should be "
                         "evenly divided by TiledMma");

    // F16BF16 has no SF members, so BF16 derives the SF types below from a
    // dummy MXF4 MMA; they are unused at runtime. MXF4 (SFVecSize=16) rather
    // than MXF8F6F4 (32): 64/16 = 4 matches the SfAtom's 4-in-K granularity.
    // The dummy keeps a fixed SFTileShape so the block-scaled constraints
    // (1-CTA M==128, KBlockK==64) do not leak into BF16 tiles.
    static constexpr int KSFDummyM = KNumCTA == 1 ? 128 : 256;
    using SFTileShape = cute::conditional_t<
        KIsBF16,
        cute::Shape<cute::Int<KSFDummyM>, cute::Int<128>, cute::Int<64>>,
        TileShape>;
    using DummySFMma_T = cute::conditional_t<
        KNumCTA == 1,
        cute::SM100_MMA_MXF4_SS<cutlass::float_e2m1_t, cutlass::float_e2m1_t,
                                TypeAccumulate_, cutlass::float_ue4m3_t,
                                cute::get<0>(SFTileShape{}),
                                cute::get<1>(SFTileShape{}), 16,
                                cute::UMMA::Major::K, cute::UMMA::Major::K>,
        cute::SM100_MMA_MXF4_2x1SM_SS<
            cutlass::float_e2m1_t, cutlass::float_e2m1_t, TypeAccumulate_,
            cutlass::float_ue4m3_t, cute::get<0>(SFTileShape{}),
            cute::get<1>(SFTileShape{}), 16, cute::UMMA::Major::K,
            cute::UMMA::Major::K>>;
    using TiledMmaForSF =
        cute::conditional_t<KIsBF16, decltype(make_tiled_mma(DummySFMma_T{})),
                            TiledMma>;

    // Unaligned-N pair tiles (IsSfbUnaligned) load SFB as a run of 128-row
    // atoms, so the SF MMA atom is 128 wide rather than round_up(N). BF16
    // never takes this arm.
    using MMA_SF_T = cute::conditional_t<
        KIsBF16 || SwapAB || KBlockN % 128 == 0,
        typename TiledMmaForSF::MMA_ScaleFactor,
        cute::conditional_t<
            KIsFP4,
            cute::SM100_MMA_MXF4_SS<TypeA_Mma, TypeB_Mma, TypeAccumulate_,
                                    TypeSF, KBlockM / KNumCTA, 128, KSFVecSize,
                                    cute::UMMA::Major::K, cute::UMMA::Major::K>,
            cute::SM100_MMA_MXF8F6F4_SS<TypeA_Mma, TypeB_Mma, TypeAccumulate_,
                                        TypeSF, KBlockM / KNumCTA, 128,
                                        cute::UMMA::Major::K,
                                        cute::UMMA::Major::K>>>;
    using TiledMma_SF = cute::TiledMMA<
        cute::MMA_Atom<MMA_SF_T>,
        cute::Layout<cute::Shape<cute::_1, cute::_1, cute::_1>>,
        cute::Tile<cute::Underscore, cute::Underscore, cute::Underscore>>;

    // SF fragment types for the mainloop's TMEM SF tensors.
    using FrgTypeSFA = typename TiledMmaForSF::FrgTypeSFA;
    using FrgTypeSFB = typename TiledMmaForSF::FrgTypeSFB;

    using AtomThrShapeMNK =
        cute::Shape<decltype(cute::shape<0>(
                        typename TiledMma::ThrLayoutVMNK{})),
                    cute::_1, cute::_1>;
    using CtaShape_MNK =
        decltype(cute::shape_div(TileShape{}, AtomThrShapeMNK{}));
    // Per-CTA SFTileShape; equals CtaShape_MNK for FP8/FP4.
    using SFCtaShape_MNK =
        decltype(cute::shape_div(SFTileShape{}, AtomThrShapeMNK{}));

    static constexpr uint32_t SFVectorSize = TiledMmaForSF::SFVecSize;
    using Sm1xxBlkScaledConfig =
        cutlass::detail::Sm1xxBlockScaledConfig<SFVectorSize>;
    using Blk_MN = typename Sm1xxBlkScaledConfig::Blk_MN;
    static constexpr bool IsCtaNSmall =
        cute::size<1>(CtaShape_MNK{}) < Blk_MN{};
    static constexpr int IsCtaN64 = cute::size<1>(CtaShape_MNK{}) == 64;
    // A pair tile whose N is not a multiple of the 128-row SF atom (e.g. 224)
    // starts at a 32-row-aligned offset inside an atom; SFB is then the
    // covering run of whole atoms and the MMA's SFB address moves forward by
    // that offset (one TMEM column holds 32 atom rows). BF16 skips this.
    static constexpr bool IsSfbUnaligned =
        !KIsBF16 && !KIsFP4 && (!SwapAB || KernelConfig_::KSwapSfbRun) &&
        KBlockN > 128 && KBlockN % 128 != 0;
    static_assert(!IsSfbUnaligned || KBlockN % 32 == 0,
                  "unaligned SFB tiles need KBlockN % 32 == 0");
    // Used when BLOCK_N=224.
    static constexpr bool IsSfbRowShift =
        IsSfbUnaligned && KBlockN % kSfbRowShiftAlign != 0;
    static constexpr int KSfbTransposeWarpId = KLoadCWarpId;
    static constexpr int KSfbTransposeWarps = KNumWarp - KSfbTransposeWarpId;
    static_assert(!IsSfbRowShift || (!AddBias && KSfbTransposeWarps >= 1),
                  "the 224-column SFB rewrite needs the idle bias-loader warp");
    static constexpr int KSfbMaxOffset =
        !IsSfbUnaligned ? 0 : (KBlockN % 64 == 0 ? 64 : 96);
    static constexpr int KSfbLoadN =
        (KBlockN + KSfbMaxOffset + 127) / 128 * 128;
    static constexpr int KSfbAtoms = KSfbLoadN / 128;
    static_assert(!IsSfbRowShift || KSfbLoadN >= KBlockN + 32,
                  "the 224-column window must have room for the 32-row shift");
    static int constexpr CTA_N_SF =
        IsSfbUnaligned
            ? KSfbLoadN
            : int(cutlass::ceil_div(cute::size<1>(SFCtaShape_MNK{}), Blk_MN{}) *
                  Blk_MN{});

    using TileShape_SF = decltype(cute::make_shape(
        cute::get<0>(SFCtaShape_MNK{}),
        cute::Int<CTA_N_SF>{} *
            cute::shape<2>(typename TiledMmaForSF::ThrLayoutVMNK()),
        cute::get<2>(SFTileShape{})));
    using MmaShapeA_MK = decltype(cute::partition_shape_A(
        TiledMma{}, cute::make_shape(cute::size<0>(TileShape{}),
                                     cute::size<2>(TileShape{}))));
    using MmaShapeB_NK = decltype(cute::partition_shape_B(
        TiledMma{}, cute::make_shape(cute::size<1>(TileShape{}),
                                     cute::size<2>(TileShape{}))));
    using BlockTileA_M = decltype(cute::size<0, 0>(MmaShapeA_MK{}) *
                                  cute::size<1>(MmaShapeA_MK{}));
    using BlockTileA_K = decltype(cute::size<0, 1>(MmaShapeA_MK{}) *
                                  cute::size<2>(MmaShapeA_MK{}));
    using BlockTileB_N = decltype(cute::size<0, 0>(MmaShapeB_NK{}) *
                                  cute::size<1>(MmaShapeB_NK{}));
    using BlockTileB_K = decltype(cute::size<0, 1>(MmaShapeB_NK{}) *
                                  cute::size<2>(MmaShapeB_NK{}));

    using SmemLayoutAtomA =
        decltype(cutlass::gemm::collective::detail::sm100_smem_selector<
                 cute::UMMA::Major::K, TypeA_Smem, BlockTileA_M,
                 BlockTileA_K>());
    using SmemLayoutAtomB =
        decltype(cutlass::gemm::collective::detail::sm100_smem_selector<
                 cute::UMMA::Major::K, TypeB_Smem, BlockTileB_N,
                 BlockTileB_K>());

    using SmemLayoutA = decltype(cute::UMMA::tile_to_mma_shape(
        SmemLayoutAtomA{},
        cute::append(MmaShapeA_MK{}, cute::Int<KPrologueStages>{}),
        cute::Step<cute::_1, cute::_2, cute::_3>{}));
    using SmemLayoutB = decltype(cute::UMMA::tile_to_mma_shape(
        SmemLayoutAtomB{},
        cute::append(MmaShapeB_NK{}, cute::Int<KPrologueStages>{}),
        cute::Step<cute::_1, cute::_2, cute::_3>{}));

    using LayoutSFA = decltype(Sm1xxBlkScaledConfig::deduce_layoutSFA());
    using LayoutSFB = decltype(Sm1xxBlkScaledConfig::deduce_layoutSFB());

    using SmemLayoutAtomSFA =
        decltype(Sm1xxBlkScaledConfig::deduce_smem_layoutSFA(TiledMmaForSF{},
                                                             SFTileShape{}));
    // Unaligned-N SFB loads KSfbAtoms whole atoms, so the SMEM atom covers
    // CTA_N_SF rows instead of SFTileShape's N. Every other tile uses the
    // SFTileShape form.
    using SmemLayoutAtomSFB = cute::conditional_t<
        IsSfbUnaligned,
        decltype(Sm1xxBlkScaledConfig::deduce_smem_layoutSFB(
            TiledMmaForSF{},
            cute::Shape<cute::Int<KBlockM>, cute::Int<CTA_N_SF>,
                        cute::Int<KBlockK>>{})),
        decltype(Sm1xxBlkScaledConfig::deduce_smem_layoutSFB(TiledMmaForSF{},
                                                             SFTileShape{}))>;

    using SmemLayoutSFA = decltype(cute::make_layout(
        cute::append(cute::shape(SmemLayoutAtomSFA{}),
                     cute::Int<KPrologueStages>{}),
        cute::append(cute::stride(SmemLayoutAtomSFA{}),
                     cute::size(cute::filter_zeros(SmemLayoutAtomSFA{})))));

    using SmemLayoutSFB = decltype(cute::make_layout(
        cute::append(cute::shape(SmemLayoutAtomSFB{}),
                     cute::Int<KPrologueStages>{}),
        cute::append(cute::stride(SmemLayoutAtomSFB{}),
                     cute::size(cute::filter_zeros(SmemLayoutAtomSFB{})))));
    using StrideC = cute::Shape<int32_t, cute::_1, int32_t>;

    using EpilogueTile_M = decltype(cute::size<0>(EpilogueTile{}));
    using EpilogueTile_N = decltype(cute::size<1>(EpilogueTile{}));
    using SmemLayoutAtomC =
        decltype(sm100_get_epilogue_smem_swizzle_layout_atom<StrideC, TypeC,
                                                             EpilogueTile>());

    using SmemLayoutStageC = decltype(cute::tile_to_shape(
        SmemLayoutAtomC{}, cute::product_each(cute::shape(EpilogueTile{})),
        cute::Step<cute::_1, cute::_2>{}));
    constexpr static int StrideStageC = cute::cosize_v<SmemLayoutStageC>;
    using SmemLayoutC = decltype(cute::append<3>(
        SmemLayoutStageC{},
        cute::Layout<cute::Int<KEpilogueStages>, cute::Int<StrideStageC>>{}));

    using MainloopPipeline =
        cutlass::PipelineTmaUmmaAsync<KPrologueStages, ClusterShape_MNK,
                                      AtomThrShapeMNK>;
    using MainloopPipelineState = typename MainloopPipeline::PipelineState;
    using AccumulatorPipeline =
        cutlass::PipelineUmmaAsync<KEpilogueStages, AtomThrShapeMNK>;
    using AccumulatorPipelineState =
        typename AccumulatorPipeline::PipelineState;

    // Multi-stage bias buffer so load_c overlaps the epilogue.
    static constexpr int KBiasStages = KernelConfig_::KBiasStages;
    using SmemLayoutBiasC = decltype(cute::append<3>(
        SmemLayoutStageC{},
        cute::Layout<cute::Int<KBiasStages>, cute::Int<StrideStageC>>{}));

    using BiasPipeline = cutlass::PipelineTransactionAsync<KBiasStages>;
    using BiasPipelineState = typename BiasPipeline::PipelineState;

    struct SharedStorage {
        alignas(
            128) cute::ArrayEngine<TypeA_Smem, cute::cosize_v<SmemLayoutA>> A;
        alignas(
            128) cute::ArrayEngine<TypeB_Smem, cute::cosize_v<SmemLayoutB>> B;
        // SFA/SFB SMEM: size 0 for BF16 (no scale factors).
        alignas(128) cute::ArrayEngine<
            TypeSF, KIsBF16 ? 0 : cute::cosize_v<SmemLayoutSFA>> SFA;
        alignas(128) cute::ArrayEngine<
            TypeSF, KIsBF16 ? 0 : cute::cosize_v<SmemLayoutSFB>> SFB;
        // Zero-sized unless AddBias.
        alignas(128) cute::ArrayEngine<
            TypeC, AddBias ? cute::cosize_v<SmemLayoutBiasC> : 0> C;

        // Per-expert alpha, prefetched at kernel start; zero-sized unless
        // HasAlpha.
        alignas(16) float alpha[HasAlpha ? KMaxAlphaExperts : 0];

        alignas(16) cute::uint32_t tmem_base_ptr;

        CUTE_DEVICE constexpr auto tensor_sA() {
            return cute::make_tensor(cute::make_smem_ptr(A.begin()),
                                     SmemLayoutA{});
        }
        CUTE_DEVICE constexpr auto tensor_sB() {
            return cute::make_tensor(cute::make_smem_ptr(B.begin()),
                                     SmemLayoutB{});
        }
        CUTE_DEVICE constexpr auto tensor_sSFA() {
            return cute::make_tensor(cute::make_smem_ptr(SFA.begin()),
                                     SmemLayoutSFA{});
        }
        CUTE_DEVICE constexpr auto tensor_sSFB() {
            return cute::make_tensor(cute::make_smem_ptr(SFB.begin()),
                                     SmemLayoutSFB{});
        }
        CUTE_DEVICE constexpr auto tensor_sC() {
            return cute::make_tensor(cute::make_smem_ptr(C.begin()),
                                     SmemLayoutBiasC{});
        }

        typename BiasPipeline::SharedStorage bias_pipeline;

        using MainloopPipelineStorage =
            typename MainloopPipeline::SharedStorage;
        alignas(16) MainloopPipelineStorage mainloop;
        using AccumulatorPipelineStorage =
            typename AccumulatorPipeline::SharedStorage;
        alignas(16) AccumulatorPipelineStorage accumulator;

        // Pair-UMMA overlap: the peer's overlap WG must finish T2R before the
        // leader issues the next UMMA, so it arrives on this cluster barrier
        // (the leader's NamedBarrier cannot see it).
        alignas(8) typename cutlass::arch::ClusterBarrier::ValueType
            overlap_cluster_bar;

        static constexpr int KSfFullStages =
            IsSfbRowShift ? KPrologueStages : 1;
        alignas(8) typename cutlass::arch::ClusterTransactionBarrier::ValueType
            sf_full[KSfFullStages];
    };

    static constexpr bool KActRowTma =
        KernelType == KernelMode::Batch && SwapAB && IsPairUmma && !KIsFP4 &&
        !KIsBF16;
    static constexpr int KActRowAlign = std::gcd(KBlockN, mxfp8::kAtomRows);

    static constexpr int KCtaRows = KBlockM / KNumCTA;
    static constexpr int KEpiChunk = !SwapAB ? KCtaRows
                                     : KBlockN % (2 * kSwapEpiChunk) == 0
                                         ? kSwapEpiChunk
                                         : kMatRows;
    static constexpr int KEpiRowBytes =
        SwapAB ? KCtaRows * sizeof(TypeD) / KAccColsPerOut
               : mxfp8::kGroup * sizeof(TypeD);
    static constexpr int KSwapSfRowBytes = KCtaRows / mxfp8::kGroup;
    static constexpr int KSwapAmaxBytes = cutlass::NumWarpsPerWarpGroup *
                                          (KEpiChunk / kMatRows) * kQuad *
                                          sizeof(uint32_t);
    static constexpr int KEpiBufBytes =
        KEpiChunk * KEpiRowBytes +
        (SwapAB && KQuantD ? KEpiChunk * KSwapSfRowBytes : 0) +
        (SwapAB && KGlu ? KSwapAmaxBytes : 0);
    static constexpr int KSmemSwizzleAlign = 1024;
    static constexpr int KEpiStageOffset =
        cute::round_up((int)sizeof(SharedStorage), KSmemSwizzleAlign);
    static constexpr int KEpiStageBytes =
        KEpilogueWarpGroupCount * 2 * KEpiBufBytes;
    static constexpr bool KEpiStageFits =
        KEpiStageOffset + KEpiStageBytes <=
        cutlass::arch::sm100_smem_capacity_bytes;
    static constexpr bool KSwapStaged = SwapAB && !KIsBF16 && KEpiStageFits;
    static constexpr int KSmemBytes = KSwapStaged || KQuantD
                                          ? KEpiStageOffset + KEpiStageBytes
                                          : (int)sizeof(SharedStorage);

    static void print_args() {}
};

}  // namespace halops
