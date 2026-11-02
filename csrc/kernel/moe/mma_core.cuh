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

#include <cstdint>
#include <type_traits>

// SM100 keeps the warp-level fp8 MMA (ptxas lowers it to two HMMA.16816
// ops); CUTLASS gates it on its SM89 arch macro, opened here.
#if !defined(CUTE_ARCH_MMA_F32_SM89_ENABLED)
#    define CUTE_ARCH_MMA_F32_SM89_ENABLED 1
#endif

#include <cute/algorithm/tuple_algorithms.hpp>
#include <cute/arch/copy_sm75.hpp>
#include <cute/arch/mma_sm100.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cute/arch/mma_sm89.hpp>
#include <cute/atom/copy_traits_sm75.hpp>
#include <cute/atom/mma_traits_sm100.hpp>
#include <cute/atom/mma_traits_sm80.hpp>
#include <cute/atom/mma_traits_sm89.hpp>
#include <cute/tensor.hpp>
#include <cutlass/detail/sm100_blockscaled_layout.hpp>
#include <cutlass/detail/sm100_tmem_helper.hpp>

#include "kernel/moe/moe_configs.cuh"

namespace halops::moe {

// K width of one SW128 operand box.
template <typename T>
struct SwizzleBoxK {
    static constexpr std::uint32_t value = 1024u / cute::sizeof_bits_v<T>;
};

// UE8M0: the byte is a biased power-of-two exponent, i.e. the fp32 exponent
// field.  A zero group has a zero scale, which contributes nothing.
__device__ __forceinline__ float ue8m0_to_float(std::uint8_t biased_exp) {
    return __uint_as_float(static_cast<std::uint32_t>(biased_exp) << 23);
}

// The SM100 UMMA instruction for an operand pair.  `Method` separates the
// blockscaled MXFP8 recipe from the plain fp8 one, which share element types.
template <class ElementA, class ElementB, class ElementAcc, class AtomShape,
          cute::UMMA::Major MajorA = cute::UMMA::Major::K,
          cute::UMMA::Major MajorB = cute::UMMA::Major::K,
          QuantMethod Method = QuantMethod::kBlockwiseFP8>
struct UmmaAtom;

template <class ElementAcc, class AtomShape, cute::UMMA::Major MajorA,
          cute::UMMA::Major MajorB>
struct UmmaAtom<cute::float_e4m3_t, cute::float_e4m3_t, ElementAcc, AtomShape,
                MajorA, MajorB, QuantMethod::kBlockwiseFP8> {
    using type = cute::MMA_Traits<
        cute::SM100_MMA_F8F6F4_SS, cute::float_e4m3_t, cute::float_e4m3_t,
        ElementAcc, decltype(cute::get<0>(AtomShape{})),
        decltype(cute::get<1>(AtomShape{})),
        cute::integral_constant<cute::UMMA::Major, MajorA>,
        cute::integral_constant<cute::UMMA::Major, MajorB>,
        cute::integral_constant<cute::UMMA::ScaleIn, cute::UMMA::ScaleIn::One>,
        cute::integral_constant<cute::UMMA::ScaleIn, cute::UMMA::ScaleIn::One>>;
};

// Blockscaled MXFP8: UE8M0 factors come from TMEM, per 32-K group.
template <class ElementAcc, class AtomShape, cute::UMMA::Major MajorA,
          cute::UMMA::Major MajorB>
struct UmmaAtom<cute::float_e4m3_t, cute::float_e4m3_t, ElementAcc, AtomShape,
                MajorA, MajorB, QuantMethod::kMXFP8> {
    using type = cute::SM100_MMA_MXF8F6F4_SS<
        cute::float_e4m3_t, cute::float_e4m3_t, ElementAcc, cute::float_ue8m0_t,
        cute::get<0>(AtomShape{}), cute::get<1>(AtomShape{}), MajorA, MajorB>;
};

template <class ElementAcc, class AtomShape, cute::UMMA::Major MajorA,
          cute::UMMA::Major MajorB, QuantMethod Method>
struct UmmaAtom<__nv_fp8_e4m3, __nv_fp8_e4m3, ElementAcc, AtomShape, MajorA,
                MajorB, Method>
    : UmmaAtom<cute::float_e4m3_t, cute::float_e4m3_t, ElementAcc, AtomShape,
               MajorA, MajorB, Method> {};

template <class ElementAcc, class AtomShape, cute::UMMA::Major MajorA,
          cute::UMMA::Major MajorB>
struct UmmaAtom<cute::bfloat16_t, cute::bfloat16_t, ElementAcc, AtomShape,
                MajorA, MajorB, QuantMethod::kNone> {
    using type =
        cute::SM100_MMA_F16BF16_SS<cute::bfloat16_t, cute::bfloat16_t,
                                   ElementAcc, cute::get<0>(AtomShape{}),
                                   cute::get<1>(AtomShape{}), MajorA, MajorB>;
};

template <class ElementAcc, class AtomShape, cute::UMMA::Major MajorA,
          cute::UMMA::Major MajorB>
struct UmmaAtom<__nv_bfloat16, __nv_bfloat16, ElementAcc, AtomShape, MajorA,
                MajorB, QuantMethod::kNone>
    : UmmaAtom<cute::bfloat16_t, cute::bfloat16_t, ElementAcc, AtomShape,
               MajorA, MajorB, QuantMethod::kNone> {};

// MXFP8 scale-factor staging for the blockscaled instruction: one tile's
// UE8M0 scales are packed into CUTLASS's SfAtom SMEM layout, copied to TMEM
// with UTCCP, and read by the MMA per 32-K group.
template <class TiledMma, class TileShape, bool Blockscaled>
struct UmmaScaleTraits {
    static constexpr std::uint32_t kNumSF = 0u;
    static constexpr std::uint32_t kSfaSmemBytes = 1u;
    static constexpr std::uint32_t kSfbSmemBytes = 1u;
    static constexpr std::uint32_t kSfaTmemColumns = 0u;
    static constexpr std::uint32_t kSfbTmemColumns = 0u;
};

template <class TiledMma, class TileShape>
struct UmmaScaleTraits<TiledMma, TileShape, true> {
    static constexpr std::uint32_t kSFVecSize = 32u;
    static constexpr std::uint32_t kTileShapeM = cute::get<0>(TileShape{});
    static constexpr std::uint32_t kTileShapeN = cute::get<1>(TileShape{});
    static constexpr std::uint32_t kTileShapeK = cute::get<2>(TileShape{});
    static constexpr std::uint32_t kNumSF = kTileShapeK / kSFVecSize;
    static_assert(kTileShapeK % (4u * kSFVecSize) == 0u,
                  "an SfAtom holds four 32-K groups");

    using BlkScaledConfig = cutlass::detail::Sm1xxBlockScaledConfig<kSFVecSize>;
    // One SfAtom covers 128 M rows, and the instruction pairs each M atom
    // with its own scale slab, so the staging and TMEM layouts are per atom.
    static constexpr std::uint32_t kNumSfaAtoms = kTileShapeM / 128u;
    static constexpr std::uint32_t kNumSfbAtoms = (kTileShapeN + 127u) / 128u;
    static_assert(kTileShapeM % 128u == 0u,
                  "a blockscaled tile's M is whole 128-row atoms");
    using SfaAtomLayout = decltype(BlkScaledConfig::deduce_smem_layoutSFA(
        TiledMma{}, cute::make_shape(cute::Int<128>{}, cute::Int<kTileShapeN>{},
                                     cute::Int<kTileShapeK>{})));
    using SfbAtomLayout = decltype(BlkScaledConfig::deduce_smem_layoutSFB(
        TiledMma{}, TileShape{}));
    static constexpr std::uint32_t kSfaAtomBytes =
        cute::cosize_v<SfaAtomLayout>;
    static constexpr std::uint32_t kSfaSmemBytes = kNumSfaAtoms * kSfaAtomBytes;
    static constexpr std::uint32_t kSfbSmemBytes =
        cute::cosize_v<SfbAtomLayout>;
    static_assert(kSfaAtomBytes == 128u * kNumSF &&
                      kSfbSmemBytes == kNumSfbAtoms * 128u * kNumSF,
                  "the SfAtom layouts pack 128 rows of K/32 scales");

    using TmemSfa = decltype(cute::make_tensor<typename TiledMma::FrgTypeSFA>(
        cute::shape(SfaAtomLayout{})));
    using TmemSfb = decltype(cute::make_tensor<typename TiledMma::FrgTypeSFB>(
        cute::shape(SfbAtomLayout{})));
    static constexpr std::uint32_t kSfaAtomColumns =
        cutlass::detail::find_tmem_tensor_col_offset(TmemSfa{});
    static constexpr std::uint32_t kSfaTmemColumns =
        kNumSfaAtoms * kSfaAtomColumns;
    static constexpr std::uint32_t kSfbTmemColumns =
        cutlass::detail::find_tmem_tensor_col_offset(TmemSfb{});

    __device__ static TmemSfa make_tmem_sfa(std::uint32_t addr) {
        TmemSfa tensor{};
        tensor.data() = addr;
        return tensor;
    }

    __device__ static TmemSfb make_tmem_sfb(std::uint32_t addr) {
        TmemSfb tensor{};
        tensor.data() = addr;
        return tensor;
    }

    // UTCCP one slot's scales into their TMEM staging.
    __device__ static void copy_to_tmem(std::uint8_t* __restrict__ sfa,
                                        std::uint8_t* __restrict__ sfb,
                                        std::uint32_t sfa_addr,
                                        std::uint32_t sfb_addr) {
        using UtccpOp = cute::SM100_UTCCP_4x32dp128bit_1cta;
        TmemSfa tmem_sfa = make_tmem_sfa(sfa_addr);
        TmemSfb tmem_sfb = make_tmem_sfb(sfb_addr);
        auto sfa_compact = cute::make_tensor(
            cute::make_smem_ptr(reinterpret_cast<cute::float_ue8m0_t*>(sfa)),
            cute::filter_zeros(SfaAtomLayout{}));
        auto sfb_compact = cute::make_tensor(
            cute::make_smem_ptr(reinterpret_cast<cute::float_ue8m0_t*>(sfb)),
            cute::filter_zeros(SfbAtomLayout{}));
        auto tmem_sfa_compact = cute::make_tensor(
            tmem_sfa.data(), cute::filter_zeros(tmem_sfa.layout()));
        auto tmem_sfb_compact = cute::make_tensor(
            tmem_sfb.data(), cute::filter_zeros(tmem_sfb.layout()));
        auto copy_sfa = cute::make_utccp_copy(UtccpOp{}, tmem_sfa_compact);
        auto copy_sfb = cute::make_utccp_copy(UtccpOp{}, tmem_sfb_compact);
        auto thr_sfa = copy_sfa.get_slice(0);
        auto thr_sfb = copy_sfb.get_slice(0);
        auto s_sfa = cute::get_utccp_smem_desc_tensor<UtccpOp>(
            thr_sfa.partition_S(sfa_compact));
        auto s_sfb = cute::get_utccp_smem_desc_tensor<UtccpOp>(
            thr_sfb.partition_S(sfb_compact));
        cute::copy(copy_sfa, s_sfa, thr_sfa.partition_D(tmem_sfa_compact));
        cute::copy(copy_sfb, s_sfb, thr_sfb.partition_D(tmem_sfb_compact));
    }
};

// One SM100 MMA kind: instruction M is 128 (64 for a shorter tile), N is 8,
// and K is 256 bits of the operand type; the caller's tile may repeat it.
// MXFP8 uses the blockscaled instruction, which applies UE8M0 scale factors
// per 32-element group from TMEM.
template <class QuantA, class QuantB, class ElementAcc, class TileShape,
          cute::UMMA::Major MajorA = cute::UMMA::Major::K,
          cute::UMMA::Major MajorB = cute::UMMA::Major::K>
struct UmmaCore {
    using ElementA = typename QuantA::ElementType;
    using ElementB = typename QuantB::ElementType;

    static_assert(cute::sizeof_bits_v<ElementA> ==
                      cute::sizeof_bits_v<ElementB>,
                  "both operands must have the same element size");
    static_assert((QuantA::kMethod == QuantMethod::kMXFP8) ==
                      (QuantB::kMethod == QuantMethod::kMXFP8),
                  "mixed blockscaled/plain operand recipes are not wired");

    static constexpr bool kUseBlockscaled =
        QuantA::kMethod == QuantMethod::kMXFP8;

    static constexpr std::uint32_t kTileShapeM = cute::get<0>(TileShape{});
    static constexpr std::uint32_t kTileShapeN = cute::get<1>(TileShape{});
    static constexpr std::uint32_t kTileShapeK = cute::get<2>(TileShape{});

    // 256 bits of K per issue: 32 for fp8, 16 for bf16.  The blockscaled
    // instruction has no 64-row form, so its tile M is whole 128-row atoms.
    static constexpr std::uint32_t kAtomShapeM =
        kUseBlockscaled ? 128u : (kTileShapeM <= 64u ? 64u : 128u);
    static_assert(!kUseBlockscaled || kTileShapeM % 128u == 0u,
                  "a blockscaled tile's M must be whole 128-row atoms");
    static constexpr std::uint32_t kAtomShapeN = kTileShapeN;
    static constexpr std::uint32_t kAtomShapeK =
        256u / cute::sizeof_bits_v<ElementA>;
    using AtomShape =
        cute::Shape<cute::Int<kAtomShapeM>, cute::Int<kAtomShapeN>,
                    cute::Int<kAtomShapeK>>;

    static_assert(kTileShapeN % 8u == 0u && kTileShapeN <= 256u,
                  "the instruction's N is a multiple of 8, at most 256");
    static_assert(kTileShapeM % kAtomShapeM == 0u,
                  "the tile's M must be a whole number of instructions");
    static_assert(kTileShapeK % kAtomShapeK == 0u,
                  "the tile's K must be a whole number of instructions");

    static constexpr std::uint32_t kNumAtomMs = kTileShapeM / kAtomShapeM;

    using AtomOp = typename UmmaAtom<ElementA, ElementB, ElementAcc, AtomShape,
                                     MajorA, MajorB, QuantA::kMethod>::type;
    using TiledMma = decltype(cute::make_tiled_mma(AtomOp{}));
    using ScaleTraits = UmmaScaleTraits<TiledMma, TileShape, kUseBlockscaled>;

    static constexpr std::uint32_t kNumAtomKs = kTileShapeK / kAtomShapeK;

    // K width of one SW128 operand box; a wider tile repeats it along K.
    static constexpr std::uint32_t kSwizzleBoxK = SwizzleBoxK<ElementA>::value;
    static_assert(kTileShapeK % kSwizzleBoxK == 0u,
                  "the tile's K must be a whole number of SW128 boxes");
    static constexpr std::uint32_t kNumSwizzleBoxKs =
        kTileShapeK / kSwizzleBoxK;

    // Bytes of one tile's A operand: the mbarrier's byte count.
    static constexpr std::uint32_t kWeightTileElems = kTileShapeM * kTileShapeK;
    static constexpr std::uint32_t kWeightTileBytes =
        kWeightTileElems * (cute::sizeof_bits_v<ElementA> / 8u);

    // SMEM layout of one instruction's A operand: a kAtomShapeM x kTileShapeK
    // SWZ128 region; a wider tile repeats it at kAtomShapeM-row strides.
    using WeightTileLayout = decltype(cute::UMMA::tile_to_mma_shape(
        cute::UMMA::Layout_K_SW128_Atom<ElementA>{},
        cute::partition_shape_A(TiledMma{},
                                cute::make_shape(cute::Int<kAtomShapeM>{},
                                                 cute::Int<kTileShapeK>{}))));

    static constexpr std::uint64_t kWeightTileLbo = 16ULL;  // one LBO chunk
    static constexpr std::uint64_t kWeightTileSbo = 1024ULL;

    static_assert(
        cute::stride<0, 1>(cute::get_nonswizzle_portion(WeightTileLayout{})) *
                16 ==
            kWeightTileLbo,
        "A: K must be contiguous; 16 B = one LBO chunk.");
    static_assert(
        cute::stride<0, 0>(cute::get_nonswizzle_portion(WeightTileLayout{})) *
                8 * (cute::sizeof_bits_v<ElementA> / 8) ==
            kWeightTileSbo,
        "A: SBO mismatch.");

    // The tile's gemm; `sA` is the tile's A operand in SMEM and `tmem_col`
    // the TMEM address of its first accumulator column.
    template <std::uint32_t NumSubBlocks = 1u, class CtaMma, class TensorB,
              class TensorC>
    __device__ static void run(TiledMma& tiled_mma, CtaMma& cta_mma,
                               ElementA* sA, TensorB const& tCrB, TensorC& tCrC,
                               std::uint32_t tmem_col,
                               bool accumulate = false) {
        static_assert(!kUseBlockscaled,
                      "MXFP8 takes the scale-factor run overload");
        static_assert(
            kNumAtomKs % NumSubBlocks == 0u,
            "the K steps must divide evenly over the sub-accumulators");
        constexpr std::uint32_t kItersPerSubBlock = kNumAtomKs / NumSubBlocks;
        constexpr std::uint32_t kAtomOperandElems = kAtomShapeM * kTileShapeK;
        cute::for_each(cute::make_seq<int(kNumAtomMs)>{}, [&](auto m_block) {
            constexpr std::uint32_t m = decltype(m_block)::value;
            auto tCsA = cute::make_tensor(
                cute::make_smem_ptr(
                    reinterpret_cast<typename TiledMma::ValTypeA*>(
                        sA + m * kAtomOperandElems)),
                WeightTileLayout{});
            cute::Tensor tCrA = cta_mma.make_fragment_A(tCsA);
            cute::for_each(cute::make_seq<int(kNumAtomKs)>{}, [&](auto kb) {
                constexpr std::uint32_t k = decltype(kb)::value;
                constexpr std::uint32_t sub = k / kItersPerSubBlock;
                tCrC.data() = tmem_col + m * NumSubBlocks * kAtomShapeN +
                              sub * kAtomShapeN;
                tiled_mma.accumulate_ =
                    (k % kItersPerSubBlock == 0u && !accumulate)
                        ? cute::UMMA::ScaleOut::Zero
                        : cute::UMMA::ScaleOut::One;
                cute::gemm(tiled_mma, tCrA(cute::_, cute::_, kb),
                           tCrB(cute::_, cute::_, kb), tCrC);
            });
        });
    }

    // The blockscaled tile's gemm.  `sfa_addr`/`sfb_addr` are the TMEM
    // staging of this tile's UE8M0 scale factors, one 32-K group per K mode;
    // the instruction applies them, so the accumulator needs no sub-blocks.
    template <class CtaMma, class TensorB, class TensorC>
    __device__ static void run(TiledMma& tiled_mma, CtaMma& cta_mma,
                               ElementA* sA, TensorB const& tCrB, TensorC& tCrC,
                               std::uint32_t tmem_col, std::uint32_t sfa_addr,
                               std::uint32_t sfb_addr,
                               bool accumulate = false) {
        static_assert(kUseBlockscaled,
                      "the scale-factor run overload is the MXFP8 one");
        // Each M atom is its own instruction with its own SFA slab, so both
        // the A tile and the accumulator step atom by atom.
        constexpr std::uint32_t kAtomOperandElems = kAtomShapeM * kTileShapeK;
        const auto tCrSFB = ScaleTraits::make_tmem_sfb(sfb_addr);
        cute::for_each(cute::make_seq<int(kNumAtomMs)>{}, [&](auto m_block) {
            constexpr std::uint32_t m = decltype(m_block)::value;
            auto tCsA = cute::make_tensor(
                cute::make_smem_ptr(
                    reinterpret_cast<typename TiledMma::ValTypeA*>(
                        sA + m * kAtomOperandElems)),
                WeightTileLayout{});
            cute::Tensor tCrA = cta_mma.make_fragment_A(tCsA);
            const auto tCrSFA = ScaleTraits::make_tmem_sfa(
                sfa_addr + m * ScaleTraits::kSfaAtomColumns);
            cute::for_each(cute::make_seq<int(kNumAtomKs)>{}, [&](auto kb) {
                constexpr std::uint32_t k = decltype(kb)::value;
                tCrC.data() = tmem_col + m * kAtomShapeN;
                tiled_mma.accumulate_ = (k == 0u && !accumulate)
                                            ? cute::UMMA::ScaleOut::Zero
                                            : cute::UMMA::ScaleOut::One;
                cute::gemm(tiled_mma.with(tiled_mma.accumulate_,
                                          tCrSFA(cute::_, cute::_, kb),
                                          tCrSFB(cute::_, cute::_, kb)),
                           tCrA(cute::_, cute::_, kb),
                           tCrB(cute::_, cute::_, kb), tCrC);
            });
        });
    }
};

// The warp-level `mma.sync` core: one tile's gemm for a pair of quantized
// operands.  The fp8 kind is the primary path; unquantized (kNone) operands
// use the bf16 specialization below.
template <class QuantA, class QuantB, class ElementAcc, class TileShape,
          std::uint32_t StagedRows = 0u>
struct HmmaCore {
    using ElementA = typename QuantA::ElementType;
    using ElementB = typename QuantB::ElementType;
    static constexpr bool kUseBlockscaled = false;
    static_assert(cute::is_same_v<ElementA, __nv_fp8_e4m3> &&
                      cute::is_same_v<ElementB, __nv_fp8_e4m3>,
                  "the fp8 instruction takes two e4m3 operands");
    static_assert(cute::is_same_v<ElementAcc, float>,
                  "the instruction is the fp8 one accumulating in fp32");
    static_assert((QuantA::kMethod == QuantMethod::kMXFP8) ==
                      (QuantB::kMethod == QuantMethod::kMXFP8),
                  "mixed block-wise/MX operand recipes are not wired yet");

    static constexpr std::uint32_t kAtomShapeM = 16u;
    static constexpr std::uint32_t kAtomShapeN = 8u;
    static constexpr std::uint32_t kAtomShapeK = 32u;

    static constexpr std::uint32_t kTileShapeM = cute::get<0>(TileShape{});
    static constexpr std::uint32_t kTileShapeN = cute::get<1>(TileShape{});
    static constexpr std::uint32_t kTileShapeK = cute::get<2>(TileShape{});

    static_assert(kTileShapeM % kAtomShapeM == 0u,
                  "the tile's M must be a whole number of instructions");
    static_assert(kTileShapeN == kAtomShapeN,
                  "the tile's N is exactly one instruction wide");
    static_assert(kTileShapeK % kAtomShapeK == 0u,
                  "the tile's K must be a whole number of instructions");

    static constexpr std::uint32_t kNumAtomMs = kTileShapeM / kAtomShapeM;
    static constexpr std::uint32_t kNumAtomKs = kTileShapeK / kAtomShapeK;

    static constexpr std::uint32_t kAtomOperandsA =
        kAtomShapeM * kAtomShapeK / 32u / 4u;  // a0..a3
    static constexpr std::uint32_t kAtomOperandsB =
        kAtomShapeN * kAtomShapeK / 32u / 4u;  // b0, b1
    static constexpr std::uint32_t kAtomElemsC =
        kAtomShapeM * kAtomShapeN / 32u;
    static_assert(kAtomOperandsA == 4u && kAtomOperandsB == 2u &&
                      kAtomElemsC == 4u,
                  "the SM89 fp8 atom's register signature");

    using AtomOp = cute::SM89_16x8x32_F32E4M3E4M3F32_TN;
    // The atom tiled over the tile's M: one atom per 16 rows.
    using TiledMma = decltype(cute::make_tiled_mma(
        cute::MMA_Atom<AtomOp>{},
        cute::Layout<
            cute::Shape<cute::Int<kNumAtomMs>, cute::_1, cute::_1>>{}));

    // A is a K-major SW128 tile: one box carries one 128-K staging block.
    static constexpr std::uint32_t kSwizzleBoxK = SwizzleBoxK<ElementA>::value;
    static_assert(kTileShapeK % kSwizzleBoxK == 0u,
                  "the tile's K must be a whole number of staging blocks");

    // A 16-B swizzle chunk is the instruction's 4-K lane group.
    static constexpr std::uint32_t kChunkK = 16u;
    // The staging's N rows per 16-B chunk.
    static constexpr std::uint32_t kStagedRows =
        StagedRows != 0u ? StagedRows : kTileShapeN + 1u;
    static constexpr std::uint32_t kWeightBlockElems =
        kTileShapeM * kSwizzleBoxK;
    static constexpr std::uint32_t kWeightTileElems = kTileShapeM * kTileShapeK;
    static constexpr std::uint32_t kWeightTileBytes =
        kWeightTileElems * (cute::sizeof_bits_v<ElementA> / 8u);
    // One block of the activation staging: the N rows and 16-K chunks.
    static constexpr std::uint32_t kStagedBlockElems =
        kSwizzleBoxK * kStagedRows;

    // The SW128 weight box's word; the TMA XORs the chunk with the row's bits.
    __device__ static std::uint32_t const* weight_word(
        ElementA const* __restrict__ a, std::uint32_t row, std::uint32_t chunk,
        std::uint32_t inner4) {
        const std::uint32_t swizzled = chunk ^ (row & 7u);
        return reinterpret_cast<std::uint32_t const*>(
            a + row * kSwizzleBoxK + swizzled * kChunkK + inner4 * 4u);
    }

    // The activation staging's word of (N, 16-K chunk, inner four).
    __device__ static std::uint32_t const* staged_word(
        ElementB const* __restrict__ b, std::uint32_t chunk, std::uint32_t n,
        std::uint32_t inner4) {
        return reinterpret_cast<std::uint32_t const*>(
            b + (chunk * kStagedRows + n) * kChunkK + inner4 * 4u);
    }

    // One 16x8x32 atom: its weight box word and its staged activation.
    __device__ static void atom_mma(ElementA const* __restrict__ a,
                                    ElementB const* __restrict__ b,
                                    std::uint32_t row0, std::uint32_t row1,
                                    std::uint32_t k, std::uint32_t local,
                                    std::uint32_t inner4,
                                    float (&cur)[kAtomElemsC]) {
        std::uint32_t a_reg[kAtomOperandsA];
        a_reg[0] = *weight_word(a, row0, 2u * k, inner4);
        a_reg[1] = *weight_word(a, row1, 2u * k, inner4);
        a_reg[2] = *weight_word(a, row0, 2u * k + 1u, inner4);
        a_reg[3] = *weight_word(a, row1, 2u * k + 1u, inner4);
        std::uint32_t b_reg[kAtomOperandsB];
        b_reg[0] = *staged_word(b, 2u * k, local, inner4);
        b_reg[1] = *staged_word(b, 2u * k + 1u, local, inner4);
        // One SM89 warp-level fp8 MMA (m16n8k32, e4m3 in, fp32 accumulate):
        //   D = A @ B^T + D, A: 16x32, B: 8x32, K-major (TN).
        cute::SM89_16x8x32_F32E4M3E4M3F32_TN::fma(
            cur[0], cur[1], cur[2], cur[3], a_reg[0], a_reg[1], a_reg[2],
            a_reg[3], b_reg[0], b_reg[1], cur[0], cur[1], cur[2], cur[3]);
    }

    // The tile's gemm accumulated into `acc`, this warp's per-atom registers.
    // Both recipes walk block -> m -> k.  Block-wise scales are constant for
    // the whole 128-K block, so its four atoms accumulate before one scaling;
    // MX scales change per 32-K atom, so each atom is scaled on its own.
    template <std::uint32_t ScaleRows = kTileShapeN, class WeightScale,
              class ActScale>
    __device__ static void run(ElementA const* __restrict__ a,
                               ElementB const* __restrict__ b,
                               float (&acc)[kNumAtomMs][kAtomElemsC],
                               WeightScale const* __restrict__ w_scale,
                               ActScale const* __restrict__ act_scale,
                               std::uint32_t w_scale_stride = 0u,
                               std::uint32_t atom_base = 0u) {
        static_assert(QuantA::kMethod == QuantB::kMethod,
                      "both operands must use the same recipe");
        static_assert((QuantA::kMethod == QuantMethod::kBlockwiseFP8 &&
                       QuantA::kBlockSize == kSwizzleBoxK &&
                       QuantB::kBlockSize == kSwizzleBoxK) ||
                          (QuantA::kMethod == QuantMethod::kMXFP8 &&
                           QuantA::kBlockSize == kAtomShapeK &&
                           QuantB::kBlockSize == kAtomShapeK),
                      "the scale type and group must match the recipe");
        constexpr std::uint32_t kAtoms = kSwizzleBoxK / kAtomShapeK;
        const std::uint32_t lane_idx = threadIdx.x % 32u;
        const std::uint32_t local = lane_idx / 4u;
        const std::uint32_t col = 2u * (lane_idx & 3u);
        const std::uint32_t inner4 = lane_idx & 3u;
#pragma unroll
        for (std::uint32_t block = 0; block < kTileShapeK / kSwizzleBoxK;
             ++block) {
            ElementA const* const a_block = a + block * kWeightBlockElems;
            ElementB const* const b_block = b + block * kStagedBlockElems;
            std::uint32_t atom0 = 0u;
            float block_ws = 0.f;
            float block_as0 = 0.f;
            float block_as1 = 0.f;
            float atom_as0[kAtoms];
            float atom_as1[kAtoms];
            if constexpr (QuantA::kMethod == QuantMethod::kBlockwiseFP8) {
                block_ws = w_scale[block];
                block_as0 = act_scale[block * ScaleRows + col];
                block_as1 = act_scale[block * ScaleRows + col + 1u];
            } else {
                atom0 = atom_base + block * kAtoms;
#pragma unroll
                for (std::uint32_t k = 0; k < kAtoms; ++k) {
                    atom_as0[k] = ue8m0_to_float(
                        act_scale[(atom0 + k) * ScaleRows + col]);
                    atom_as1[k] = ue8m0_to_float(
                        act_scale[(atom0 + k) * ScaleRows + col + 1u]);
                }
            }
#pragma unroll
            for (std::uint32_t m = 0; m < kNumAtomMs; ++m) {
                const std::uint32_t row0 = m * kAtomShapeM + local;
                const std::uint32_t row1 = row0 + kAtomShapeM / 2u;
                if constexpr (QuantA::kMethod == QuantMethod::kBlockwiseFP8) {
                    float cur[kAtomElemsC] = {};
#pragma unroll
                    for (std::uint32_t k = 0; k < kAtoms; ++k) {
                        atom_mma(a_block, b_block, row0, row1, k, local, inner4,
                                 cur);
                    }
                    acc[m][0] += cur[0] * block_ws * block_as0;
                    acc[m][1] += cur[1] * block_ws * block_as1;
                    acc[m][2] += cur[2] * block_ws * block_as0;
                    acc[m][3] += cur[3] * block_ws * block_as1;
                } else {
                    // The block's four weight scales are one uint32 per row.
                    const std::uint32_t w0 =
                        *reinterpret_cast<const std::uint32_t*>(
                            w_scale + row0 * w_scale_stride + atom0);
                    const std::uint32_t w1 =
                        *reinterpret_cast<const std::uint32_t*>(
                            w_scale + row1 * w_scale_stride + atom0);
#pragma unroll
                    for (std::uint32_t k = 0; k < kAtoms; ++k) {
                        const float ws0 =
                            ue8m0_to_float((w0 >> (8u * k)) & 0xFFu);
                        const float ws1 =
                            ue8m0_to_float((w1 >> (8u * k)) & 0xFFu);
                        float cur[kAtomElemsC] = {};
                        atom_mma(a_block, b_block, row0, row1, k, local, inner4,
                                 cur);
                        acc[m][0] += cur[0] * ws0 * atom_as0[k];
                        acc[m][1] += cur[1] * ws0 * atom_as1[k];
                        acc[m][2] += cur[2] * ws1 * atom_as0[k];
                        acc[m][3] += cur[3] * ws1 * atom_as1[k];
                    }
                }
            }
        }
    }
};

// The SM80 warp-level kind: m16n8k16, bf16 in, fp32 accumulated in
// registers, operands loaded from SMEM with `ldmatrix`.
template <class ElementAcc, class TileShape, std::uint32_t StagedRows>
struct HmmaCore<QuantSpec<QuantMethod::kNone>, QuantSpec<QuantMethod::kNone>,
                ElementAcc, TileShape, StagedRows> {
    using Element = typename QuantSpec<QuantMethod::kNone>::ElementType;
    static constexpr bool kUseBlockscaled = false;
    static_assert(cute::is_same_v<Element, __nv_bfloat16> &&
                      cute::is_same_v<ElementAcc, float>,
                  "the instruction is the bf16 one accumulating in fp32");

    using AtomOp = cute::SM80_16x8x16_F32BF16BF16F32_TN;
    static constexpr std::uint32_t kAtomShapeM = 16u;
    static constexpr std::uint32_t kAtomShapeN = 8u;
    static constexpr std::uint32_t kAtomShapeK = 16u;

    static constexpr std::uint32_t kTileShapeM = cute::get<0>(TileShape{});
    static constexpr std::uint32_t kTileShapeN = cute::get<1>(TileShape{});
    static constexpr std::uint32_t kTileShapeK = cute::get<2>(TileShape{});

    static_assert(kTileShapeM % kAtomShapeM == 0u,
                  "the tile's M must be a whole number of instructions");
    static constexpr std::uint32_t kNumAtomMs = kTileShapeM / kAtomShapeM;
    // Values of the m16n8 C fragment one lane holds.
    static constexpr std::uint32_t kAtomElemsC = 4u;
    static_assert(kTileShapeN == kAtomShapeN,
                  "the tile's N is exactly one instruction wide");
    static_assert(kTileShapeK % kAtomShapeK == 0u,
                  "the tile's K must be a whole number of instructions");
    static constexpr std::uint32_t kNumAtomKs = kTileShapeK / kAtomShapeK;

    using TiledMma = decltype(cute::make_tiled_mma(AtomOp{}));

    // A is a K-major SW128 tile: 128 B of K per swizzle box.
    static constexpr std::uint32_t kSwizzleBoxK = SwizzleBoxK<Element>::value;
    static_assert(kTileShapeK % kSwizzleBoxK == 0u,
                  "the tile's K must be a whole number of SW128 boxes");
    static constexpr std::uint32_t kNumSwizzleBoxKs =
        kTileShapeK / kSwizzleBoxK;
    using WeightTileLayout = decltype(cute::tile_to_shape(
        cute::composition(
            cute::Swizzle<3, 3, 3>{},
            cute::Layout<cute::Shape<cute::_8, cute::Int<kSwizzleBoxK>>,
                         cute::Stride<cute::Int<kSwizzleBoxK>, cute::_1>>{}),
        cute::Shape<cute::Int<kTileShapeM>, cute::Int<kTileShapeK>>{}));

    static constexpr std::uint32_t kWeightTileElems = kTileShapeM * kTileShapeK;
    static constexpr std::uint32_t kWeightTileBytes =
        kWeightTileElems * (cute::sizeof_bits_v<Element> / 8u);

    using CopyAtomA = cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, Element>;
    using CopyAtomB = cute::Copy_Atom<cute::SM75_U32x2_LDSM_N, Element>;

    // The tile's gemm: one warp walks the tile's M atoms, each into its own
    // register accumulator.
    template <class TensorA, class TensorB>
    __device__ static void run(TensorA const& sA, TensorB const& sB,
                               float (&acc)[kNumAtomMs][kAtomElemsC]) {
        const std::uint32_t lane_idx = threadIdx.x % 32u;
        TiledMma tiled_mma;
        auto thr_mma = tiled_mma.get_slice(lane_idx);
        cute::Tensor tCrB = thr_mma.partition_fragment_B(sB);

        auto s2r_a = cute::make_tiled_copy_A(CopyAtomA{}, tiled_mma);
        auto thr_s2r_a = s2r_a.get_slice(lane_idx);
        auto s2r_b = cute::make_tiled_copy_B(CopyAtomB{}, tiled_mma);
        auto thr_s2r_b = s2r_b.get_slice(lane_idx);
        cute::Tensor tBsB = thr_s2r_b.partition_S(sB);
        cute::Tensor tBrB = thr_s2r_b.retile_D(tCrB);
        // Shape-only proxy: only the C fragment's layout is read from it.
        auto c_frag_layout =
            thr_mma
                .partition_fragment_C(cute::make_tensor(
                    cute::make_gmem_ptr(static_cast<ElementAcc*>(nullptr)),
                    cute::make_layout(cute::Shape<cute::Int<kAtomShapeM>,
                                                  cute::Int<kTileShapeN>>{},
                                      cute::GenRowMajor{})))
                .layout();
        cute::for_each(cute::make_seq<int(kNumAtomMs)>{}, [&](auto atom) {
            constexpr std::uint32_t m = decltype(atom)::value;
            // This atom's rows, in the parent tile's own strides.
            auto sA_m =
                cute::local_tile(sA,
                                 cute::make_shape(cute::Int<kAtomShapeM>{},
                                                  cute::Int<kTileShapeK>{}),
                                 cute::make_coord(m, cute::_0{}));
            cute::Tensor tCrA = thr_mma.partition_fragment_A(sA_m);
            cute::Tensor tAsA = thr_s2r_a.partition_S(sA_m);
            cute::Tensor tArA = thr_s2r_a.retile_D(tCrA);
            // The caller's registers, in the C fragment's layout.
            cute::Tensor tCrC = cute::make_tensor(
                cute::make_rmem_ptr(&acc[m][0]), c_frag_layout);
#pragma unroll
            for (std::uint32_t k = 0; k < kNumAtomKs; ++k) {
                cute::copy(s2r_a, tAsA(cute::_, cute::_, k),
                           tArA(cute::_, cute::_, k));
                cute::copy(s2r_b, tBsB(cute::_, cute::_, k),
                           tBrB(cute::_, cute::_, k));
                cute::gemm(tiled_mma, tCrA(cute::_, cute::_, k),
                           tCrB(cute::_, cute::_, k), tCrC);
            }
        });
    }
};

}  // namespace halops::moe
