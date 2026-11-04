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
#include <cute/algorithm/tuple_algorithms.hpp>
#include <cute/arch/copy_sm75.hpp>
#include <cute/arch/mma_sm100.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cute/atom/copy_traits_sm75.hpp>
#include <cute/atom/mma_traits_sm100.hpp>
#include <cute/atom/mma_traits_sm80.hpp>
#include <cute/tensor.hpp>
#include <cutlass/detail/sm100_blockscaled_layout.hpp>
#include <cutlass/detail/sm100_tmem_helper.hpp>
#include <type_traits>

#include "kernel/moe/moe_configs.cuh"

namespace halops::moe {

// K width of one SW128 operand box.
template <typename T>
struct SwizzleBoxK {
    static constexpr std::uint32_t value = 1024u / cute::sizeof_bits_v<T>;
};

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

// Blockscaled w4a8: 4-bit weights paired with fp8 activations, UE8M0 factors
// per 32-K group from TMEM.  Same instruction as MXFP8 (mxf8f6f4 takes any
// pair of operands at most 8 bits wide), but its format field is keyed off the
// *unpacked*-in-SMEM E2M1 marker: plain `float_e2m1_t` encodes the standalone
// MXF4 format instead (cute/arch/mma_sm100_desc.hpp).
template <class ElementAcc, class AtomShape, cute::UMMA::Major MajorA,
          cute::UMMA::Major MajorB>
struct UmmaAtom<cutlass::float_e2m1_t, __nv_fp8_e4m3, ElementAcc, AtomShape,
                MajorA, MajorB, QuantMethod::kMXFP4> {
    using type = cute::SM100_MMA_MXF8F6F4_SS<
        cute::detail::float_e2m1_unpacksmem_t, cute::float_e4m3_t, ElementAcc,
        cute::float_ue8m0_t, cute::get<0>(AtomShape{}),
        cute::get<1>(AtomShape{}), MajorA, MajorB>;
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
        copy_sfa_to_tmem(sfa, sfa_addr);
        copy_sfb_to_tmem(sfb, sfb_addr);
    }

    // One operand's half of the pair, for the paths that stage each from the
    // warp that loads it.
    __device__ static void copy_sfa_to_tmem(std::uint8_t* __restrict__ sfa,
                                            std::uint32_t sfa_addr) {
        using UtccpOp = cute::SM100_UTCCP_4x32dp128bit_1cta;
        auto sfa_compact = cute::make_tensor(
            cute::make_smem_ptr(reinterpret_cast<cute::float_ue8m0_t*>(sfa)),
            cute::filter_zeros(SfaAtomLayout{}));
        TmemSfa tmem_sfa = make_tmem_sfa(sfa_addr);
        auto tmem_sfa_compact = cute::make_tensor(
            tmem_sfa.data(), cute::filter_zeros(tmem_sfa.layout()));
        auto copy_sfa = cute::make_utccp_copy(UtccpOp{}, tmem_sfa_compact);
        auto thr_sfa = copy_sfa.get_slice(0);
        auto s_sfa = cute::get_utccp_smem_desc_tensor<UtccpOp>(
            thr_sfa.partition_S(sfa_compact));
        cute::copy(copy_sfa, s_sfa, thr_sfa.partition_D(tmem_sfa_compact));
    }

    __device__ static void copy_sfb_to_tmem(std::uint8_t* __restrict__ sfb,
                                            std::uint32_t sfb_addr) {
        using UtccpOp = cute::SM100_UTCCP_4x32dp128bit_1cta;
        auto sfb_compact = cute::make_tensor(
            cute::make_smem_ptr(reinterpret_cast<cute::float_ue8m0_t*>(sfb)),
            cute::filter_zeros(SfbAtomLayout{}));
        TmemSfb tmem_sfb = make_tmem_sfb(sfb_addr);
        auto tmem_sfb_compact = cute::make_tensor(
            tmem_sfb.data(), cute::filter_zeros(tmem_sfb.layout()));
        auto copy_sfb = cute::make_utccp_copy(UtccpOp{}, tmem_sfb_compact);
        auto thr_sfb = copy_sfb.get_slice(0);
        auto s_sfb = cute::get_utccp_smem_desc_tensor<UtccpOp>(
            thr_sfb.partition_S(sfb_compact));
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

    static constexpr bool kBlockscaledA =
        QuantA::kMethod == QuantMethod::kMXFP8 ||
        QuantA::kMethod == QuantMethod::kMXFP4;
    static constexpr bool kBlockscaledB =
        QuantB::kMethod == QuantMethod::kMXFP8;
    static_assert(kBlockscaledA == kBlockscaledB,
                  "mixed blockscaled/plain operand recipes are not wired");
    static_assert(!kBlockscaledA || QuantA::kMethod == QuantMethod::kMXFP4 ||
                      cute::sizeof_bits_v<ElementA> ==
                          cute::sizeof_bits_v<ElementB>,
                  "only the w4a8 recipe pairs different element sizes");

    static constexpr bool kUseBlockscaled = kBlockscaledA;

    // A sub-byte (4-bit) operand is staged one value per byte: the gmem tensor
    // is packed and the TMA's 16U4_ALIGN16B expansion fills the gap.  So every
    // SMEM/LBO/SBO quantity below is byte-sized for both fp8 and fp4.
    static constexpr std::uint32_t kElementABits =
        cute::sizeof_bits_v<ElementA>;
    static constexpr std::uint32_t kElementBBits =
        cute::sizeof_bits_v<ElementB>;
    static constexpr std::uint32_t kSmemABits =
        kElementABits < 8u ? 8u : kElementABits;
    static constexpr std::uint32_t kMmaBits =
        kElementABits > kElementBBits ? kElementABits : kElementBBits;

    static constexpr std::uint32_t kTileShapeM = cute::get<0>(TileShape{});
    static constexpr std::uint32_t kTileShapeN = cute::get<1>(TileShape{});
    static constexpr std::uint32_t kTileShapeK = cute::get<2>(TileShape{});

    // 256 bits of the widest operand per issue: 32 elements for fp8/fp4 mixed
    // (mxf8f6f4), 16 for bf16.  The blockscaled instruction has no 64-row form,
    // so its tile M is whole 128-row atoms.
    static constexpr std::uint32_t kAtomShapeM =
        kUseBlockscaled ? 128u : (kTileShapeM <= 64u ? 64u : 128u);
    static_assert(!kUseBlockscaled || kTileShapeM % 128u == 0u,
                  "a blockscaled tile's M must be whole 128-row atoms");
    static constexpr std::uint32_t kAtomShapeN = kTileShapeN;
    static constexpr std::uint32_t kAtomShapeK = 256u / kMmaBits;
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

    // K width of one SW128 A-operand box (128 B); a wider tile repeats it.
    static constexpr std::uint32_t kSwizzleBoxK = 1024u / kSmemABits;
    static_assert(kTileShapeK % kSwizzleBoxK == 0u,
                  "the tile's K must be a whole number of SW128 boxes");
    static constexpr std::uint32_t kNumSwizzleBoxKs =
        kTileShapeK / kSwizzleBoxK;

    // The A operand's SMEM element: one byte per value, whatever the weight's
    // packed width is.
    using SmemElementA =
        std::conditional_t<(kElementABits < 8u), std::uint8_t, ElementA>;

    // Bytes of one tile's A operand in SMEM.  A 4-bit weight lands expanded
    // to one byte per value (16 values per 16 B: 8 B of nibbles, then 8 B the
    // copy leaves as is), so its SW128 image is laid out like the fp8 one.
    static constexpr std::uint32_t kWeightTileElems = kTileShapeM * kTileShapeK;
    static constexpr std::uint32_t kWeightTileBytes =
        kWeightTileElems * (kSmemABits / 8u);
    // What the copy credits the mbarrier with: the gmem bytes it reads, so a
    // 4-bit weight (two values per byte) counts half its SMEM footprint.
    static constexpr std::uint32_t kWeightTileTxBytes =
        kWeightTileElems * kElementABits / 8u;

    // SMEM layout of one instruction's A operand: a kAtomShapeM x kTileShapeK
    // SWZ128 region; a wider tile repeats it at kAtomShapeM-row strides.
    using WeightTileLayout = decltype(cute::UMMA::tile_to_mma_shape(
        cute::UMMA::Layout_K_SW128_Atom<SmemElementA>{},
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
                8 * (kSmemABits / 8) ==
            kWeightTileSbo,
        "A: SBO mismatch.");

    // The tile's gemm; `sA` is the tile's A operand in SMEM and `tmem_col`
    // the TMEM address of its first accumulator column.
    template <std::uint32_t NumSubBlocks = 1u, class CtaMma, class TensorB,
              class TensorC>
    __device__ static void run(TiledMma& tiled_mma, CtaMma& cta_mma,
                               SmemElementA* sA, TensorB const& tCrB,
                               TensorC& tCrC, std::uint32_t tmem_col,
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
                cute::make_smem_ptr(sA + m * kAtomOperandElems),
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
                               SmemElementA* sA, TensorB const& tCrB,
                               TensorC& tCrC, std::uint32_t tmem_col,
                               std::uint32_t sfa_addr, std::uint32_t sfb_addr,
                               bool accumulate = false) {
        static_assert(kUseBlockscaled,
                      "the scale-factor run overload is the MXFP8 one");
        // An SF address's top two bits are its scale-factor ID, which the
        // instruction descriptor carries.  The staging sits at lane 0, ID 0,
        // so only the column field is set; saying so lets each K mode's ID
        // fold into a constant descriptor instead of being rebuilt per MMA.
        sfa_addr &= 0xFFFFu;
        sfb_addr &= 0xFFFFu;
        // Each M atom is its own instruction with its own SFA slab, so both
        // the A tile and the accumulator step atom by atom.
        constexpr std::uint32_t kAtomOperandElems = kAtomShapeM * kTileShapeK;
        const auto tCrSFB = ScaleTraits::make_tmem_sfb(sfb_addr);
        cute::for_each(cute::make_seq<int(kNumAtomMs)>{}, [&](auto m_block) {
            constexpr std::uint32_t m = decltype(m_block)::value;
            auto tCsA = cute::make_tensor(
                cute::make_smem_ptr(sA + m * kAtomOperandElems),
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

// The warp-level `mma.sync` core, the unquantized walk: m16n8k16, bf16 in,
// fp32 accumulated in registers, operands loaded from SMEM with `ldmatrix`.
// Every quantized group takes the UMMA cores above.
template <class TileShape>
struct HmmaCore {
    using Element = __nv_bfloat16;
    using ElementAcc = float;
    static constexpr bool kUseBlockscaled = false;

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
