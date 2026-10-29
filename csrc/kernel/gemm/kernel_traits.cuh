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
    // KNumCTA == 1: single-CTA UMMA (SM100_MMA_..._SS, cta_group::1)
    // KNumCTA == 2: pair-UMMA       (SM100_MMA_..._2x1SM_SS, cta_group::2)
    // FP4: MMA K-increment is 64 elements; FP8: MMA K-increment is 32 elements.
    // BF16: non-block-scaled kind::f16 MMA, K-increment is 16 elements.
    static constexpr bool KIsFP4 = KernelConfig_::KIsFP4;
    static constexpr bool KIsBF16 = KernelConfig_::KIsBF16;
    // Narrow the A TMA box to the real M below KBlockM (the UMMA
    // still reads KBlockM rows and the epilogue drops D past M);
    static constexpr bool KNarrowA =
        NarrowA_ && KDenseW8A8MXFP8 && !SwapAB && !IsPairUmma && !KIsFP4;
    static constexpr bool KUsePdl = UsePdl_;
    static constexpr int KMmaK = KIsFP4 ? 64 : (KIsBF16 ? 16 : 32);
    // Tail-only stream-K eligibility; single source of truth for the
    // scheduler's split decision and for emitting the KStreamK=true kernel.
    // Only the BF16 dense no-bias no-alpha epilogue can deposit a raw fp32
    // partial (the bias epilogue has no stream-K path; the reduce kernel does
    // not apply alpha). Also excluded, each a silent wrong-result otherwise:
    //   * SwapAB: the tail fold-back math assumes non-swap traversal
    //     (m=64 n=19584 k=1024 gave 121% rel error; HALOPS_STREAMK=0
    //     disables stream-K);
    //   * N-predicated tiles: the compact per-tile fp32 partial slot would
    //     leave holes that the reduce kernel sums.
    static constexpr bool KSupportsStreamK =
        KernelType == KernelMode::Dense && KIsBF16 && !AddBias && !HasAlpha &&
        !SwapAB && (KBlockN % 128 == 0 || KBlockN == 32 || KBlockN == 64) &&
        !KernelConfig_::KForceEpiNPredicate;
    static constexpr int kItersK = KBlockK / KMmaK;
    // SwapAB: 1 warp group (warps 0-3) suffices for the small N-dimension.
    // Normal: 2 warp groups (warps 0-7) split the N-dimension work.
    // KEpilogueWarpGroupsOverride can force 1, e.g. to make KBlockN=32 legal
    // at CtaM==64 (KEpiComputeN = 32 meets the T2R atom floor).
    static constexpr int KEpilogueWarpGroupCount =
        KernelConfig_::KEpilogueWarpGroupsOverride != 0
            ? KernelConfig_::KEpilogueWarpGroupsOverride
            : (SwapAB ? 1 : 2);
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
    // Whether to do runtime register redistribution (setmaxnreg): needs enough
    // warps for the 208+88 split. Not keyed on the epilogue warp-group count;
    // without it ptxas gives every warp one flat budget (54-87 regs, starving
    // the epilogue). SwapAB is enabled only for dense BF16; quantized SwapAB
    // configs keep their existing (unmeasured) behaviour.
    static constexpr bool KRedistributeRegs =
        KNumWarp >= 8 &&
        (!SwapAB || (KIsBF16 && KernelType == KernelMode::Dense));

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
    // Sub-tile for overlap TMEM load splitting.
    // When overlap is enabled, always use 16-wide sub-tiles (TMEM load atom
    // width). This is fully general: KTmemOverlapCols need not divide
    // KEpiComputeN evenly, as long as both are multiples of 16. Falls back to
    // EpiComputeTile when overlap is disabled (KTmemOverlapCols==0).
    //
    // Non-power-of-2 KBlockN (e.g. 96/112/224, for one-wave occupancy): the
    // T2R tables only have atoms for KEpiComputeN in {16, 32, 64, 128}, so
    // such widths are decomposed into uniform KSubTileN=32 sub-tiles (16 would
    // give only 8 fp32/thread, too few for SM100_STORE_256bit). Not legal on
    // the 2SM "2x2" layout, whose second N half sits on datapaths 64-127: a
    // 32-wide slice straddles the half boundary and writes wrong elements.
    // A per-warp-group width of 8 (224-pair, KEpiComputeN=112 falls back to
    // 32-wide sub-tiles here) has no single atom either.
    static constexpr bool KEpiNIsSingleAtom =
        (KEpiComputeN == 16 || KEpiComputeN == 32 || KEpiComputeN == 64 ||
         KEpiComputeN == 128);
    static constexpr int KSubTileN =
        KTmemOverlapCols != 0
            ? 16
            : (KEpiNIsSingleAtom ? KEpiComputeN
                                 : (KEpiComputeN % 32 == 0 ? 32 : 16));
    // KIs2SmHalfM spelled out (it is declared further down).
    static_assert(KEpiNIsSingleAtom || !(KNumCTA == 2 && KBlockM == 128),
                  "a non-power-of-2 KBlockN cannot use the 2SM 128-row \"2x2\" "
                  "TMEM layout: its N dimension is (KBlockN/2, 2) over two "
                  "datapath halves, so the 32-wide epilogue sub-tiles straddle "
                  "the half boundary and the store lands on wrong elements "
                  "(measured 67.85% mismatch). Use a 1-CTA KBlockM=64 config, "
                  "whose 16-datapath layout maps N straight to columns.");
    using OverlapComputeTile = cute::conditional_t<
        KTmemOverlapCols == 0 && KEpiNIsSingleAtom, EpiComputeTile,
        cute::Shape<cute::Int<KBlockM / KNumCTA>, cute::Int<KSubTileN>>>;
    static constexpr int KSubTilesPerWG = KEpiComputeN / KSubTileN;
    // Overlap sub-tiles sit at the boundary between adjacent stages.
    static constexpr int KOverlapSubCount = KTmemOverlapCols / KSubTileN;
    static constexpr int KNonOverlapSubCount =
        KSubTilesPerWG - KOverlapSubCount;

    // Overlap region fits one TMEM load (widths with a hardware op), so MMA can
    // resume right after T2R; other widths (e.g. 48) interleave per sub-tile.
    static constexpr bool KOverlapSingleLoad =
        (KTmemOverlapCols > 0) &&
        (KTmemOverlapCols == 16 || KTmemOverlapCols == 32 ||
         KTmemOverlapCols == 64 || KTmemOverlapCols == 128);

    // ---- TMEM column budget ----------------------------------------------
    // tcgen05.alloc takes a power-of-2 column count in [32, 512]; accumulators
    // need KEpilogueStages * KBlockN columns. Right-sizing is perf-neutral but
    // gives a truthful footprint for small tiles. BF16 without overlap only:
    // block-scaled dtypes also keep SFA/SFB in TMEM, and overlap shares
    // columns between stages, so both keep the full 512.
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

    // ---- TMEM load (T2R) atom selection: depends on BOTH per-CTA M and N ----
    // `NNdp` is the number of TMEM datapath lanes an atom addresses. A 1SM
    // 64-row accumulator only occupies the first 16 lanes of each of the 4
    // subpartitions, so it needs a 16dp atom (32dp fails "Non-injective
    // Layout"), with the N repeat count HALVED relative to 32dp. The tiled
    // copy stays 128 threads either way.
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

    // ---- Which TMEM accumulator layout are we actually reading? ------------
    //   1CTA, KBlockM=64  -> 1SM half-subpartition: 16 lanes/subpartition, N
    //                        in TMEM columns (16dp atom).
    //   2CTA, KBlockM=128 -> 2SM "2x2": second half of N on datapaths 64-127,
    //                        only N/2 columns used (32dp atom).
    //   2CTA, KBlockM=256 -> 2SM "4x1": all 128 lanes, N in columns.
    // The "2x2" case needs a 32dp atom AND an epilogue tile spanning the full
    // KBlockN (KEpilogueWarpGroupsOverride = 1); otherwise warps 2/3 address
    // outside the tile and the output is silently wrong (no assert fires).
    static constexpr bool KIs2SmHalfM = (KNumCTA == 2) && (KEpiCtaM == 64);
    static constexpr int KEpiTmemDp =
        (KNumCTA == 1 && KEpiCtaM == 64) ? 16 : 32;
    static_assert(!KIs2SmHalfM || KEpiComputeN == KBlockN,
                  "2SM accumulators with KBlockM=128 use the 2x2 TMEM layout, "
                  "whose second N half sits on datapaths 64-127; the epilogue "
                  "tile must cover the full KBlockN (set "
                  "KEpilogueWarpGroupsOverride = 1) or warps 2/3 address "
                  "outside the tile and the output is silently wrong.");

    // CopyOp for loading the full overlap region in one shot (when
    // KOverlapSingleLoad). Width-matched to KTmemOverlapCols. Fallback to 16x
    // when not single-load.
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

    // Select TMEM load op based on EpiComputeTile N dimension (and CtaM).
    // The `Nx` suffix is the column count; a mismatched width silently moves
    // the wrong columns, so every width a config can ask for needs its own
    // arm. The 16dp series (CtaM == 64) uses HALVED atom width (N/2).
    using CopyOpT2R = cute::conditional_t<
        KEpiTmemDp == 16,
        cute::conditional_t<
            KEpiComputeN == 128, cute::SM100_TMEM_LOAD_16dp32b64x,
            cute::conditional_t<KEpiComputeN == 64,
                                cute::SM100_TMEM_LOAD_16dp32b32x,
                                cute::SM100_TMEM_LOAD_16dp32b16x>>,
        cute::conditional_t<
            KIs2SmHalfM,
            // 2SM "2x2": only KEpiComputeN/2 TMEM columns are used, so the
            // column repeat is halved relative to the 4x1 case below.
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
    // Overlap path: 16-wide sub-tile load (16 fp32/thread fills
    // SM100_STORE_256bit; 8x is too few). No overlap: width-matched CopyOpT2R.
    // CtaM==64 halves the repeat count, as above.
    //
    // TMEM columns per KSubTileN-wide sub-tile: N/2 on the 16dp and 2SM "2x2"
    // layouts (upper N half on other datapaths), N on 4x1.
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
        KTmemOverlapCols == 0,
        cute::conditional_t<KEpiNIsSingleAtom, CopyOpT2R, CopyOpT2R_SubWide>,
        cute::conditional_t<KEpiTmemDp == 16, cute::SM100_TMEM_LOAD_16dp32b8x,
                            cute::SM100_TMEM_LOAD_32dp32b16x>>;

    // ---- Close the CopyOpT2R fall-through ---------------------------------
    // Each chain above ends in a default (narrowest) atom, which make_tmem_copy
    // replicates over the tile. That is only correct when the atom width
    // divides the tile; otherwise it compiles clean and writes wrong elements.
    static_assert(KEpiComputeN % KSubTileN == 0,
                  "the epilogue sub-tile loop must cover KEpiComputeN exactly");
    static_assert(KEpiNIsSingleAtom || KTmemOverlapCols != 0 ||
                      KSubTileTmemCols % 16 == 0,
                  "KEpiComputeN outside {16,32,64,128} falls through to the "
                  "16-column atom; make_tmem_copy can only replicate that over "
                  "the tile when the tile's TMEM column count is a multiple of "
                  "16. Widen the table before using such a width.");

    // Non-power-of-2 KBlockN tiles may overhang N, so the store is predicated
    // at KSubTileN granularity (n must be a multiple of it). A config can
    // force this for a power-of-2 KBlockN via KForceEpiNPredicate (e.g.
    // N=1600 on a 128-wide tile, KSubTileN=64).
    static constexpr bool KEpiNeedsNPredicate =
        ((KBlockN & (KBlockN - 1)) != 0) || KernelConfig_::KForceEpiNPredicate;

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
                TypeA, TypeB, TypeAccumulate_, cute::get<0>(TileShape{}),
                cute::get<1>(TileShape{}), cute::UMMA::Major::K,
                cute::UMMA::Major::K>,
            cute::SM100_MMA_F16BF16_2x1SM_SS<
                TypeA, TypeB, TypeAccumulate_, cute::get<0>(TileShape{}),
                cute::get<1>(TileShape{}), cute::UMMA::Major::K,
                cute::UMMA::Major::K>>,
        cute::conditional_t<
            KIsFP4,
            cute::conditional_t<
                KNumCTA == 1,
                cute::SM100_MMA_MXF4_SS<
                    TypeA, TypeB, TypeAccumulate_, TypeSF,
                    cute::get<0>(TileShape{}), cute::get<1>(TileShape{}),
                    KSFVecSize, cute::UMMA::Major::K, cute::UMMA::Major::K>,
                cute::SM100_MMA_MXF4_2x1SM_SS<
                    TypeA, TypeB, TypeAccumulate_, TypeSF,
                    cute::get<0>(TileShape{}), cute::get<1>(TileShape{}),
                    KSFVecSize, cute::UMMA::Major::K, cute::UMMA::Major::K>>,
            cute::conditional_t<
                KNumCTA == 1,
                cute::SM100_MMA_MXF8F6F4_SS<
                    TypeA, TypeB, TypeAccumulate_, TypeSF,
                    cute::get<0>(TileShape{}), cute::get<1>(TileShape{}),
                    cute::UMMA::Major::K, cute::UMMA::Major::K>,
                cute::SM100_MMA_MXF8F6F4_2x1SM_SS<
                    TypeA, TypeB, TypeAccumulate_, TypeSF,
                    cute::get<0>(TileShape{}), cute::get<1>(TileShape{}),
                    cute::UMMA::Major::K, cute::UMMA::Major::K>>>>;
    using TiledMma = decltype(make_tiled_mma(MMA_T{}));

    CUTE_STATIC_ASSERT_V(cute::evenly_divides(TileShape{},
                                              cute::tile_shape(TiledMma{})),
                         "Static cluster shape used: TileShape should be "
                         "evenly divided by TiledMma");

    // The F16BF16 atom has NO scale-factor members (SFVecSize, MMA_ScaleFactor,
    // FrgTypeSFA/SFB). For BF16 the SF types below are derived from a dummy
    // MXF4 MMA so they stay well-formed (never used at runtime; the mainloop
    // guards them with if constexpr(!KIsBF16)). MXF4 (SFVecSize=16) rather
    // than MXF8F6F4 (32): 64 / 16 = 4 matches the SfAtom's 4-in-K granularity.
    //
    // The dummy uses a fixed SFTileShape rather than TileShape, so the
    // block-scaled constraints (1-CTA M == 128, KBlockK == 64) do not leak
    // into BF16 tile choices (e.g. M=64, KBlockK=32). For FP8/FP4 SFTileShape
    // IS TileShape and TiledMmaForSF IS TiledMma: codegen unchanged.
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

    // For unaligned-N pair tiles (see IsSfbUnaligned) the SFB tile is a run of
    // 128-row atoms, so the SF MMA atom is 128 wide rather than round_up(N).
    // BF16 never takes this arm (KIsBF16 -> TiledMmaForSF is the dummy, no
    // SFB is loaded) and neither does the aligned-N majority.
    using MMA_SF_T = cute::conditional_t<
        (!KIsBF16 && !KIsFP4 && !SwapAB && KBlockN > 128 && KBlockN % 128 != 0),
        cute::SM100_MMA_MXF8F6F4_SS<TypeA, TypeB, TypeAccumulate_, TypeSF,
                                    KBlockM / KNumCTA, 128,
                                    cute::UMMA::Major::K, cute::UMMA::Major::K>,
        typename TiledMmaForSF::MMA_ScaleFactor>;
    using TiledMma_SF = cute::TiledMMA<
        cute::MMA_Atom<MMA_SF_T>,
        cute::Layout<cute::Shape<cute::_1, cute::_1, cute::_1>>,
        cute::Tile<cute::Underscore, cute::Underscore, cute::Underscore>>;

    // SF fragment types for the mainloop's TMEM SF tensors (from TiledMmaForSF,
    // since the BF16 TiledMma lacks them).
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
    // starts at a 32-row-aligned offset inside an atom.  SFB is then loaded as
    // the covering run of whole atoms, and the MMA's SFB TMEM address is moved
    // forward by the offset (one TMEM column holds 32 rows of an atom).
    // BF16 has no scale factors and never takes this path.
    static constexpr bool IsSfbUnaligned =
        !KIsBF16 && !KIsFP4 && !SwapAB && KBlockN > 128 && KBlockN % 128 != 0;
    static_assert(!IsSfbUnaligned || KBlockN % 32 == 0,
                  "unaligned SFB tiles need KBlockN % 32 == 0");
    // Used when BLOCK_N=224.
    static constexpr bool IsSfbRowShift = IsSfbUnaligned && KBlockN % 64 != 0;
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
                 cute::UMMA::Major::K, TypeA, BlockTileA_M, BlockTileA_K>());
    using SmemLayoutAtomB =
        decltype(cutlass::gemm::collective::detail::sm100_smem_selector<
                 cute::UMMA::Major::K, TypeB, BlockTileB_N, BlockTileB_K>());

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
    // Unaligned-N SFB loads a run of KSfbAtoms whole 128-row atoms (see
    // IsSfbUnaligned), so the SMEM atom covers CTA_N_SF rows rather than
    // SFTileShape's N. For every other tile (incl. the BF16 dummy, whose SF
    // layouts are never used at runtime) the SFTileShape form applies.
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
        alignas(128) cute::ArrayEngine<TypeA, cute::cosize_v<SmemLayoutA>> A;
        alignas(128) cute::ArrayEngine<TypeB, cute::cosize_v<SmemLayoutB>> B;
        // SFA/SFB SMEM: size 0 for BF16 (no scale factors).
        alignas(128) cute::ArrayEngine<
            TypeSF, KIsBF16 ? 0 : cute::cosize_v<SmemLayoutSFA>> SFA;
        alignas(128) cute::ArrayEngine<
            TypeSF, KIsBF16 ? 0 : cute::cosize_v<SmemLayoutSFB>> SFB;
        // Bias C SMEM buffer (only allocated when AddBias is true at compile
        // time) Uses empty array trick: size 0 when AddBias=false, no SMEM
        // overhead.
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

        // Pair-UMMA overlap: the peer CTA's overlap WG must finish T2R before
        // the leader issues the next UMMA; the leader's NamedBarrier cannot see
        // it, so the peer arrives on this cluster barrier.
        alignas(8) typename cutlass::arch::ClusterBarrier::ValueType
            overlap_cluster_bar;

        static constexpr int KSfFullStages =
            IsSfbRowShift ? KPrologueStages : 1;
        alignas(8) typename cutlass::arch::ClusterTransactionBarrier::ValueType
            sf_full[KSfFullStages];
    };

    static void print_args() {}
};

}  // namespace halops
