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

// SM100_MMA_MXF8F6F4_SS without the election in fma, so one elected lane can
// issue a whole tile.
template <class a_type, class b_type, class c_type, class sf_type, int M, int N,
          cute::UMMA::Major a_major, cute::UMMA::Major b_major>
struct SM100_MMA_MXF8F6F4_SS_NOELECT
    : cute::SM100_MMA_MXF8F6F4_SS<a_type, b_type, c_type, sf_type, M, N,
                                  a_major, b_major> {
    CUTE_HOST_DEVICE static void fma(std::uint64_t const& desc_a,
                                     std::uint64_t const& desc_b,
                                     std::uint32_t const& tmem_c,
                                     std::uint32_t const& scaleC,
                                     std::uint64_t const& idescE,
                                     std::uint32_t const& tsfa_addr,
                                     std::uint32_t const& tsfb_addr) {
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %4, 0;\n\t"
            "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [%0], %1, "
            "%2, %3, [%5], [%6], p; \n\t"
            "}\n"
            :
            : "r"(tmem_c), "l"(desc_a), "l"(desc_b),
              "r"(static_cast<std::uint32_t>(idescE >> 32)), "r"(scaleC),
              "r"(tsfa_addr), "r"(tsfb_addr));
    }
};

template <class Op>
struct IsNoElectUmma : std::false_type {};

template <class a_type, class b_type, class c_type, class sf_type, int M, int N,
          cute::UMMA::Major a_major, cute::UMMA::Major b_major>
struct IsNoElectUmma<SM100_MMA_MXF8F6F4_SS_NOELECT<
    a_type, b_type, c_type, sf_type, M, N, a_major, b_major>> : std::true_type {
};

}  // namespace halops::moe

namespace cute {
// cute's traits, with with() and mma_unpack dispatching to the NOELECT fma.
template <class a_type, class b_type, class c_type, class sf_type, int M, int N,
          UMMA::Major a_major, UMMA::Major b_major>
struct MMA_Traits<halops::moe::SM100_MMA_MXF8F6F4_SS_NOELECT<
    a_type, b_type, c_type, sf_type, M, N, a_major, b_major>>
    : MMA_Traits<SM100_MMA_MXF8F6F4_SS<a_type, b_type, c_type, sf_type, M, N,
                                       a_major, b_major>> {
    using Base =
        MMA_Traits<SM100_MMA_MXF8F6F4_SS<a_type, b_type, c_type, sf_type, M, N,
                                         a_major, b_major>>;

    template <class TSFA, class TSFALayout, class TSFB, class TSFBLayout>
    CUTE_HOST_DEVICE constexpr MMA_Traits with(
        UMMA::ScaleOut accumulate, Tensor<TSFA, TSFALayout> const& SFA,
        Tensor<TSFB, TSFBLayout> const& SFB) const {
        return {Base::with(accumulate, SFA, SFB)};
    }

    template <class TD, class DLayout, class TA, class ALayout_, class TB,
              class BLayout_, class TC, class CLayout_>
    CUTE_HOST_DEVICE constexpr friend void mma_unpack(
        MMA_Traits const& traits, Tensor<TD, DLayout>& tensor_d,
        Tensor<TA, ALayout_> const& tensor_a,
        Tensor<TB, BLayout_> const& tensor_b, Tensor<TC, CLayout_> const&) {
        std::uint64_t desc_a = tensor_a[0];
        std::uint64_t desc_b = tensor_b[0];
        std::uint32_t tmem_c = raw_pointer_cast(tensor_d.data());
        std::uint64_t idesc = UMMA::make_runtime_instr_desc_block_scaled<>(
            traits.idesc_, traits.tsfa_addr_, traits.tsfb_addr_);
        halops::moe::SM100_MMA_MXF8F6F4_SS_NOELECT<
            a_type, b_type, c_type, sf_type, M, N, a_major,
            b_major>::fma(desc_a, desc_b, tmem_c,
                          static_cast<std::uint32_t>(traits.accumulate_), idesc,
                          traits.tsfa_addr_, traits.tsfb_addr_);
    }
};
}  // namespace cute

namespace halops::moe {

// K-major SW128 tile: 128 B of K per swizzle box.
template <typename T>
struct SwizzleBoxK {
    static constexpr std::uint32_t value = 1024u / cute::sizeof_bits_v<T>;
};

// K-major SMEM descriptor: elements in one contiguous 16-B LBO chunk.
template <typename T>
struct LboChunkK {
    static constexpr std::uint32_t value = 128u / cute::sizeof_bits_v<T>;
};

// The SM100 UMMA instruction for an operand pair.
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
    using type = SM100_MMA_MXF8F6F4_SS_NOELECT<
        cute::float_e4m3_t, cute::float_e4m3_t, ElementAcc, cute::float_ue8m0_t,
        cute::get<0>(AtomShape{}), cute::get<1>(AtomShape{}), MajorA, MajorB>;
};

// Blockscaled W4A8 with UE8M0 scales per 32 K elements.
template <class ElementAcc, class AtomShape, cute::UMMA::Major MajorA,
          cute::UMMA::Major MajorB>
struct UmmaAtom<cutlass::float_e2m1_t, __nv_fp8_e4m3, ElementAcc, AtomShape,
                MajorA, MajorB, QuantMethod::kMXFP4> {
    using type = SM100_MMA_MXF8F6F4_SS_NOELECT<
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
    // Each 128-row M atom has its own SMEM and TMEM scale slab.
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

    // Copy one operand's scales using its SfAtom layout.
    template <class SfLayout, class TmemSf>
    __device__ static void copy_sf_to_tmem(std::uint8_t* __restrict__ sf,
                                           TmemSf tmem_sf) {
        using UtccpOp = cute::SM100_UTCCP_4x32dp128bit_1cta;
        auto sf_compact = cute::make_tensor(
            cute::make_smem_ptr(reinterpret_cast<cute::float_ue8m0_t*>(sf)),
            cute::filter_zeros(SfLayout{}));
        auto tmem_sf_compact = cute::make_tensor(
            tmem_sf.data(), cute::filter_zeros(tmem_sf.layout()));
        auto copy_sf = cute::make_utccp_copy(UtccpOp{}, tmem_sf_compact);
        auto thr_sf = copy_sf.get_slice(0);
        auto s_sf = cute::get_utccp_smem_desc_tensor<UtccpOp>(
            thr_sf.partition_S(sf_compact));
        cute::copy(copy_sf, s_sf, thr_sf.partition_D(tmem_sf_compact));
    }

    __device__ static void copy_sfa_to_tmem(std::uint8_t* __restrict__ sfa,
                                            std::uint32_t sfa_addr) {
        copy_sf_to_tmem<SfaAtomLayout>(sfa, make_tmem_sfa(sfa_addr));
    }

    __device__ static void copy_sfb_to_tmem(std::uint8_t* __restrict__ sfb,
                                            std::uint32_t sfb_addr) {
        copy_sf_to_tmem<SfbAtomLayout>(sfb, make_tmem_sfb(sfb_addr));
    }
};

// SM100 UMMA core; blockscaled variants consume UE8M0 scales from TMEM.
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

    static constexpr std::uint32_t kElementABits =
        cute::sizeof_bits_v<ElementA>;
    static constexpr std::uint32_t kElementBBits =
        cute::sizeof_bits_v<ElementB>;
    static constexpr std::uint32_t kSmemABits =
        kElementABits < 8u ? 8u : kElementABits;
    static constexpr std::uint32_t kMmaBits =
        kElementABits > kElementBBits ? kElementABits : kElementBBits;

    // The A operand's SMEM element.
    using SmemElementA =
        std::conditional_t<(kElementABits < 8u), std::uint8_t, ElementA>;

    static constexpr std::uint32_t kTileShapeM = cute::get<0>(TileShape{});
    static constexpr std::uint32_t kTileShapeN = cute::get<1>(TileShape{});
    static constexpr std::uint32_t kTileShapeK = cute::get<2>(TileShape{});
    static constexpr std::uint32_t kNumQuantBlocksB =
        QuantB::kMethod == QuantMethod::kNone
            ? 1u
            : kTileShapeK / QuantB::kBlockSize;

    static constexpr std::uint32_t kAtomShapeM =
        kUseBlockscaled ? 128u : (kTileShapeM <= 64u ? 64u : 128u);
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
    static constexpr std::uint32_t kNumAtomKs = kTileShapeK / kAtomShapeK;

    using AtomOp = typename UmmaAtom<ElementA, ElementB, ElementAcc, AtomShape,
                                     MajorA, MajorB, QuantA::kMethod>::type;
    using TiledMma = decltype(cute::make_tiled_mma(AtomOp{}));
    using ScaleTraits = UmmaScaleTraits<TiledMma, TileShape, kUseBlockscaled>;

    static constexpr std::uint32_t kSwizzleBoxK =
        SwizzleBoxK<SmemElementA>::value;
    static_assert(kTileShapeK % kSwizzleBoxK == 0u,
                  "the tile's K must be a whole number of SW128 boxes");
    static constexpr std::uint32_t kNumSwizzleBoxKs =
        kTileShapeK / kSwizzleBoxK;

    // Bytes of one tile's A operand in SMEM.
    static constexpr std::uint32_t kWeightTileElems = kTileShapeM * kTileShapeK;
    static constexpr std::uint32_t kWeightTileBytes =
        kWeightTileElems * (kSmemABits / 8u);
    // Bytes of one tile's A operand for TMA.
    static constexpr std::uint32_t kWeightTileTxBytes =
        kWeightTileElems * kElementABits / 8u;

    // SW128 layout for one M atom, repeated at kAtomShapeM-row strides.
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

    // Run a tile with SMEM operand A and a TMEM accumulator.
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
            // CUDA and CUTLASS bf16 share storage but use distinct types.
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

    // Run blockscaled GEMM with SFA/SFB scale addresses in TMEM. One elected
    // lane issues the whole tile.
    template <class CtaMma, class TensorB, class TensorC>
    __device__ static void run(TiledMma& tiled_mma, CtaMma& cta_mma,
                               SmemElementA* sA, TensorB const& tCrB,
                               TensorC& tCrC, std::uint32_t tmem_col,
                               std::uint32_t sfa_addr, std::uint32_t sfb_addr,
                               bool accumulate = false) {
        static_assert(kUseBlockscaled,
                      "the scale-factor run overload is the MXFP8 one");
        static_assert(IsNoElectUmma<AtomOp>::value,
                      "an atom that elects inside fma cannot run in an "
                      "elected lane");

        sfa_addr &= 0xFFFFu;
        sfb_addr &= 0xFFFFu;

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

// Warp-level BF16 m16n8k16 HMMA with SMEM operands and FP32 accumulators.
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

    // Accumulate each M atom into its warp register fragment.
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
        // Only the C fragment layout is needed.
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
