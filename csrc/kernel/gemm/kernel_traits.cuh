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
          typename KernelConfig_ = GemmKernelConfig, bool NarrowA_ = false>
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
    static constexpr bool KDenseW8A8 =
        KernelType == KernelMode::Dense && !AddBias && !KernelConfig_::KIsFP4;
    static constexpr bool KIsDense =
        KernelType == KernelMode::Dense && !AddBias;
    static constexpr int KMaxAlphaExperts = 256;
    static constexpr bool IsPairUmma = KernelConfig_::IsPairUmma;
    static constexpr int KNumCTA = IsPairUmma ? 2 : 1;
    static constexpr int KClusterM = KernelConfig_::KClusterM;
    static constexpr int KClusterN = KernelConfig_::KClusterN;
    // KNumCTA 1: cta_group::1 UMMA; 2: pair-UMMA (2x1SM, cta_group::2).
    // MMA K step: 64 elements for FP4, 32 for FP8.
    static constexpr bool KIsFP4 = KernelConfig_::KIsFP4;
    // Narrow the A TMA box to the real M below KBlockM (the UMMA still reads
    // KBlockM rows; the epilogue drops D past M).
    static constexpr bool KNarrowA =
        NarrowA_ && KDenseW8A8 && !SwapAB && !IsPairUmma && !KIsFP4;
    static constexpr int KMmaK = KIsFP4 ? 64 : 32;
    static constexpr int kItersK = KBlockK / KMmaK;
    // Epilogue warp groups split N: 1 for SwapAB (small N), 2 otherwise.
    static constexpr int KEpilogueWarpGroupCount = SwapAB ? 1 : 2;
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
    // TMEM is read in 16-wide sub-tiles when overlap is on or the per-WG width
    // has no TMEM load atom (e.g. 112); KTmemOverlapCols and KEpiComputeN only
    // need to be multiples of 16.
    static constexpr bool KEpiComputeNHasAtom =
        KEpiComputeN == 128 || KEpiComputeN == 64 || KEpiComputeN == 32 ||
        KEpiComputeN == 16 || KEpiComputeN == 8;
    static constexpr int KSubTileN =
        (KTmemOverlapCols == 0 && KEpiComputeNHasAtom) ? KEpiComputeN : 16;
    using OverlapComputeTile = cute::conditional_t<
        KSubTileN == KEpiComputeN, EpiComputeTile,
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

    using CopyOpT2R_Overlap = cute::conditional_t<
        KTmemOverlapCols == 16, cute::SM100_TMEM_LOAD_32dp32b16x,
        cute::conditional_t<
            KTmemOverlapCols == 32, cute::SM100_TMEM_LOAD_32dp32b32x,
            cute::conditional_t<KTmemOverlapCols == 64,
                                cute::SM100_TMEM_LOAD_32dp32b64x,
                                cute::SM100_TMEM_LOAD_32dp32b128x>>>;

    static constexpr int KOverlapLoadN =
        KOverlapSingleLoad ? KTmemOverlapCols
                           : (KTmemOverlapCols == 0 ? 16 : KSubTileN);
    using OverlapLoadTile =
        cute::Shape<cute::Int<KBlockM / KNumCTA>, cute::Int<KOverlapLoadN>>;

    static constexpr int KWideTilesTotal = KBlockN / KOverlapLoadN;

    // The op width (`Nx` = columns per datapath) must equal KEpiComputeN: a
    // narrower op that fits moves only part of the tile. Widths with no atom
    // (e.g. 48) use the 16-wide sub-tile path (CopyOpT2R_Sub).
    using CopyOpT2R = cute::conditional_t<
        KEpiComputeN == 128, cute::SM100_TMEM_LOAD_32dp32b128x,
        cute::conditional_t<
            KEpiComputeN == 64, cute::SM100_TMEM_LOAD_32dp32b64x,
            cute::conditional_t<
                KEpiComputeN == 32, cute::SM100_TMEM_LOAD_32dp32b32x,
                cute::conditional_t<KEpiComputeN == 8,
                                    cute::SM100_TMEM_LOAD_32dp32b8x,
                                    cute::SM100_TMEM_LOAD_32dp32b16x>>>>;
    // 16x yields 16 bf16 values per thread, exactly one 256-bit store; 8x is
    // too narrow for the store atom.
    using CopyOpT2R_Sub =
        cute::conditional_t<KSubTileN == KEpiComputeN, CopyOpT2R,
                            cute::SM100_TMEM_LOAD_32dp32b16x>;

    // Scale-factor vector size: 16 = NVFP4, 32 = MXFP4.
    static constexpr int KSFVecSize = KernelConfig_::KSfVecSize;

    using MMA_T = cute::conditional_t<
        KIsFP4,
        cute::conditional_t<
            KNumCTA == 1,
            cute::SM100_MMA_MXF4_SS<TypeA, TypeB, TypeAccumulate_, TypeSF,
                                    cute::get<0>(TileShape{}),
                                    cute::get<1>(TileShape{}), KSFVecSize,
                                    cute::UMMA::Major::K, cute::UMMA::Major::K>,
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
                cute::UMMA::Major::K, cute::UMMA::Major::K>>>;
    using TiledMma = decltype(cute::make_tiled_mma(MMA_T{}));

    CUTE_STATIC_ASSERT_V(cute::evenly_divides(TileShape{},
                                              cute::tile_shape(TiledMma{})),
                         "Static cluster shape used: TileShape should be "
                         "evenly divided by TiledMma");

    // An unaligned-N tile (see IsSfbUnaligned) reads its scale as a run of
    // 128-row atoms, so its SF MMA atom is 128 wide, in the tile's own recipe.
    using MMA_SF_T = cute::conditional_t<
        SwapAB || KBlockN % 128 == 0, typename TiledMma::MMA_ScaleFactor,
        cute::conditional_t<
            KIsFP4,
            cute::SM100_MMA_MXF4_SS<TypeA, TypeB, TypeAccumulate_, TypeSF,
                                    KBlockM / KNumCTA, 128, KSFVecSize,
                                    cute::UMMA::Major::K, cute::UMMA::Major::K>,
            cute::SM100_MMA_MXF8F6F4_SS<
                TypeA, TypeB, TypeAccumulate_, TypeSF, KBlockM / KNumCTA, 128,
                cute::UMMA::Major::K, cute::UMMA::Major::K>>>;
    using TiledMma_SF = cute::TiledMMA<
        cute::MMA_Atom<MMA_SF_T>,
        cute::Layout<cute::Shape<cute::_1, cute::_1, cute::_1>>,
        cute::Tile<cute::Underscore, cute::Underscore, cute::Underscore>>;

    using AtomThrShapeMNK =
        cute::Shape<decltype(cute::shape<0>(
                        typename TiledMma::ThrLayoutVMNK{})),
                    cute::_1, cute::_1>;
    using CtaShape_MNK =
        decltype(cute::shape_div(TileShape{}, AtomThrShapeMNK{}));

    static constexpr uint32_t SFVectorSize = TiledMma::SFVecSize;
    using Sm1xxBlkScaledConfig =
        cutlass::detail::Sm1xxBlockScaledConfig<SFVectorSize>;
    using Blk_MN = typename Sm1xxBlkScaledConfig::Blk_MN;
    static constexpr bool IsCtaNSmall =
        cute::size<1>(CtaShape_MNK{}) < Blk_MN{};
    static constexpr int IsCtaN64 = cute::size<1>(CtaShape_MNK{}) == 64;
    static constexpr bool IsSfbUnaligned =
        !KIsFP4 && !SwapAB && KBlockN > 128 && KBlockN % 128 != 0;
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
            : int(cutlass::ceil_div(cute::size<1>(CtaShape_MNK{}), Blk_MN{}) *
                  Blk_MN{});

    using TileShape_SF = decltype(cute::make_shape(
        cute::get<0>(CtaShape_MNK{}),
        cute::Int<CTA_N_SF>{} *
            cute::shape<2>(typename TiledMma::ThrLayoutVMNK()),
        cute::get<2>(TileShape{})));
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
        decltype(Sm1xxBlkScaledConfig::deduce_smem_layoutSFA(TiledMma{},
                                                             TileShape{}));
    using SmemLayoutAtomSFB =
        decltype(Sm1xxBlkScaledConfig::deduce_smem_layoutSFB(
            TiledMma{}, cute::Shape<cute::Int<KBlockM>, cute::Int<CTA_N_SF>,
                                    cute::Int<KBlockK>>{}));

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
        alignas(
            128) cute::ArrayEngine<TypeSF, cute::cosize_v<SmemLayoutSFA>> SFA;
        alignas(
            128) cute::ArrayEngine<TypeSF, cute::cosize_v<SmemLayoutSFB>> SFB;
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
