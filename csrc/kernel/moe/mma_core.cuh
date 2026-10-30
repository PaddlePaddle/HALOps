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

namespace halops::moe {

// K width of one SW128 operand box.
template <typename T>
struct SwizzleBoxK {
    static constexpr std::uint32_t value = 1024u / cute::sizeof_bits_v<T>;
};

// The SM100 UMMA instruction for an operand pair.
template <class ElementA, class ElementB, class ElementAcc, class AtomShape,
          cute::UMMA::Major MajorA = cute::UMMA::Major::K,
          cute::UMMA::Major MajorB = cute::UMMA::Major::K>
struct UmmaAtom;

template <class ElementAcc, class AtomShape, cute::UMMA::Major MajorA,
          cute::UMMA::Major MajorB>
struct UmmaAtom<cute::float_e4m3_t, cute::float_e4m3_t, ElementAcc, AtomShape,
                MajorA, MajorB> {
    using type = cute::MMA_Traits<
        cute::SM100_MMA_F8F6F4_SS, cute::float_e4m3_t, cute::float_e4m3_t,
        ElementAcc, decltype(cute::get<0>(AtomShape{})),
        decltype(cute::get<1>(AtomShape{})),
        cute::integral_constant<cute::UMMA::Major, MajorA>,
        cute::integral_constant<cute::UMMA::Major, MajorB>,
        cute::integral_constant<cute::UMMA::ScaleIn, cute::UMMA::ScaleIn::One>,
        cute::integral_constant<cute::UMMA::ScaleIn, cute::UMMA::ScaleIn::One>>;
};

template <class ElementAcc, class AtomShape, cute::UMMA::Major MajorA,
          cute::UMMA::Major MajorB>
struct UmmaAtom<__nv_fp8_e4m3, __nv_fp8_e4m3, ElementAcc, AtomShape, MajorA,
                MajorB> : UmmaAtom<cute::float_e4m3_t, cute::float_e4m3_t,
                                   ElementAcc, AtomShape, MajorA, MajorB> {};

template <class ElementAcc, class AtomShape, cute::UMMA::Major MajorA,
          cute::UMMA::Major MajorB>
struct UmmaAtom<cute::bfloat16_t, cute::bfloat16_t, ElementAcc, AtomShape,
                MajorA, MajorB> {
    using type =
        cute::SM100_MMA_F16BF16_SS<cute::bfloat16_t, cute::bfloat16_t,
                                   ElementAcc, cute::get<0>(AtomShape{}),
                                   cute::get<1>(AtomShape{}), MajorA, MajorB>;
};

template <class ElementAcc, class AtomShape, cute::UMMA::Major MajorA,
          cute::UMMA::Major MajorB>
struct UmmaAtom<__nv_bfloat16, __nv_bfloat16, ElementAcc, AtomShape, MajorA,
                MajorB> : UmmaAtom<cute::bfloat16_t, cute::bfloat16_t,
                                   ElementAcc, AtomShape, MajorA, MajorB> {};

// One SM100 MMA kind: instruction M is 128 (64 for a shorter tile), N is 8,
// and K is 256 bits of the operand type; the caller's tile may repeat it.
template <class ElementA, class ElementB, class ElementAcc, class TileShape,
          cute::UMMA::Major MajorA = cute::UMMA::Major::K,
          cute::UMMA::Major MajorB = cute::UMMA::Major::K>
struct UmmaCore {
    static_assert(cute::sizeof_bits_v<ElementA> ==
                      cute::sizeof_bits_v<ElementB>,
                  "both operands must have the same element size");

    static constexpr std::uint32_t kTileShapeM = cute::get<0>(TileShape{});
    static constexpr std::uint32_t kTileShapeN = cute::get<1>(TileShape{});
    static constexpr std::uint32_t kTileShapeK = cute::get<2>(TileShape{});

    // 256 bits of K per issue: 32 for fp8, 16 for bf16.
    static constexpr std::uint32_t kAtomShapeM =
        kTileShapeM <= 64u ? 64u : 128u;
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
                                     MajorA, MajorB>::type;
    using TiledMma = decltype(cute::make_tiled_mma(cute::MMA_Atom<AtomOp>{}));

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
};

// The SM80 warp-level `mma.sync` kind: m16n8k16, bf16 in, fp32 accumulated
// in registers, operands loaded from SMEM with `ldmatrix`.
template <class Element, class ElementAcc, class TileShape,
          std::uint32_t StagedRows = 0u>
struct HmmaCore {
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

// One SM89 warp-level fp8 MMA (m16n8k32, e4m3 in, fp32 accumulate):
//   D = A @ B^T + D, A: 16x32, B: 8x32, K-major (TN).
__device__ __forceinline__ void fp8_mma_16x8x32(float (&d)[4],
                                                uint32_t const (&a)[4],
                                                uint32_t const (&b)[2]) {
    cute::SM89_16x8x32_F32E4M3E4M3F32_TN::fma(d[0], d[1], d[2], d[3], a[0],
                                              a[1], a[2], a[3], b[0], b[1],
                                              d[0], d[1], d[2], d[3]);
}

// The fp8 specialization of the warp-level core: the SM89 MMA
// (m16n8k32, e4m3 in, fp32 accumulate).
template <class ElementAcc, class TileShape, std::uint32_t StagedRows>
struct HmmaCore<__nv_fp8_e4m3, ElementAcc, TileShape, StagedRows> {
    static_assert(cute::is_same_v<ElementAcc, float>,
                  "the instruction is the fp8 one accumulating in fp32");

    using Element = __nv_fp8_e4m3;

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
                  "fp8_mma_16x8x32's register signature");

    using AtomOp = cute::SM89_16x8x32_F32E4M3E4M3F32_TN;
    // The atom tiled over the tile's M: one atom per 16 rows.
    using TiledMma = decltype(cute::make_tiled_mma(
        cute::MMA_Atom<AtomOp>{},
        cute::Layout<
            cute::Shape<cute::Int<kNumAtomMs>, cute::_1, cute::_1>>{}));

    // A is a K-major SW128 tile: one box carries one quantization block.
    static constexpr std::uint32_t kSwizzleBoxK = SwizzleBoxK<Element>::value;
    static constexpr std::uint32_t kQuantBlockK = 128u;
    static_assert(kQuantBlockK == kSwizzleBoxK,
                  "one quantization block must be one SW128 box of K");
    static_assert(kTileShapeK % kQuantBlockK == 0u,
                  "the tile's K must be a whole number of quantization blocks");
    static constexpr std::uint32_t kQuantBlocksPerTile =
        kTileShapeK / kQuantBlockK;
    static constexpr std::uint32_t kAtomsPerQuantBlock =
        kQuantBlockK / kAtomShapeK;

    // A 16-B swizzle chunk is the instruction's 4-K lane group.
    static constexpr std::uint32_t kChunkK = 16u;
    static constexpr std::uint32_t kChunksPerQuantBlock =
        kQuantBlockK / kChunkK;
    // The staging's token rows per 16-B chunk.
    static constexpr std::uint32_t kStagedRows =
        StagedRows != 0u ? StagedRows : kTileShapeN + 1u;
    static constexpr std::uint32_t kWeightBlockElems =
        kTileShapeM * kSwizzleBoxK;
    static constexpr std::uint32_t kWeightTileElems = kTileShapeM * kTileShapeK;
    static constexpr std::uint32_t kWeightTileBytes =
        kWeightTileElems * (cute::sizeof_bits_v<Element> / 8u);
    // One block of the activation staging: the token rows and 16-K chunks.
    static constexpr std::uint32_t kStagedBlockElems =
        kChunksPerQuantBlock * kStagedRows * kChunkK;

    // The SW128 weight box's word; the TMA XORs the chunk with the row's bits.
    __device__ static std::uint32_t const* weight_word(
        Element const* __restrict__ a, std::uint32_t row, std::uint32_t chunk,
        std::uint32_t inner4) {
        const std::uint32_t swizzled = chunk ^ (row & 7u);
        return reinterpret_cast<std::uint32_t const*>(
            a + row * kSwizzleBoxK + swizzled * kChunkK + inner4 * 4u);
    }

    // The activation staging's word of (token, 16-K chunk, inner four).
    __device__ static std::uint32_t const* staged_word(
        Element const* __restrict__ b, std::uint32_t chunk, std::uint32_t token,
        std::uint32_t inner4) {
        return reinterpret_cast<std::uint32_t const*>(
            b + (chunk * kStagedRows + token) * kChunkK + inner4 * 4u);
    }

    // The tile's gemm accumulated into `acc`, this warp's per-atom registers.
    // The operand scales change every quantization block, so each block
    // accumulates and is scaled on its own before it is merged into `acc`.
    template <std::uint32_t ScaleRows = kTileShapeN>
    __device__ static void run(Element const* __restrict__ a,
                               Element const* __restrict__ b,
                               float (&acc)[kNumAtomMs][kAtomElemsC],
                               float const* __restrict__ w_scale,
                               float const* __restrict__ act_scale) {
        const std::uint32_t lane_idx = threadIdx.x % 32u;
        const std::uint32_t local = lane_idx / 4u;
        const std::uint32_t col = 2u * (lane_idx & 3u);
        const std::uint32_t inner4 = lane_idx & 3u;
#pragma unroll
        for (std::uint32_t block = 0; block < kQuantBlocksPerTile; ++block) {
            Element const* const a_block = a + block * kWeightBlockElems;
            Element const* const b_block = b + block * kStagedBlockElems;
            const float ws = w_scale[block];
            const float as0 = act_scale[block * ScaleRows + col];
            const float as1 = act_scale[block * ScaleRows + col + 1u];
#pragma unroll
            for (std::uint32_t m = 0; m < kNumAtomMs; ++m) {
                const std::uint32_t row0 = m * kAtomShapeM + local;
                const std::uint32_t row1 = row0 + kAtomShapeM / 2u;
                float cur[kAtomElemsC] = {};
#pragma unroll
                for (std::uint32_t k = 0; k < kAtomsPerQuantBlock; ++k) {
                    std::uint32_t a_reg[kAtomOperandsA];
                    a_reg[0] = *weight_word(a_block, row0, 2u * k, inner4);
                    a_reg[1] = *weight_word(a_block, row1, 2u * k, inner4);
                    a_reg[2] = *weight_word(a_block, row0, 2u * k + 1u, inner4);
                    a_reg[3] = *weight_word(a_block, row1, 2u * k + 1u, inner4);
                    std::uint32_t b_reg[kAtomOperandsB];
                    b_reg[0] = *staged_word(b_block, 2u * k, local, inner4);
                    b_reg[1] =
                        *staged_word(b_block, 2u * k + 1u, local, inner4);
                    fp8_mma_16x8x32(cur, a_reg, b_reg);
                }
                acc[m][0] += cur[0] * ws * as0;
                acc[m][1] += cur[1] * ws * as1;
                acc[m][2] += cur[2] * ws * as0;
                acc[m][3] += cur[3] * ws * as1;
            }
        }
    }
};

}  // namespace halops::moe
