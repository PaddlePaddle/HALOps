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

#include <cutlass/arch/barrier.h>

#include <cute/tensor.hpp>

#include "kernel/mla/common/arch.cuh"

namespace halops::mla {
template <typename TiledMMA, typename TensorA, typename TensorB,
          typename TensorFragC>
CUTE_DEVICE void umma_ss(TiledMMA& tiled_mma, TensorA sA, TensorB sB,
                         TensorFragC tC_frag, bool clear_accum) {
    using namespace cute;
    tiled_mma.accumulate_ =
        clear_accum ? UMMA::ScaleOut::Zero : UMMA::ScaleOut::One;

    ThrMMA thr_mma = tiled_mma.get_slice(_0{});
    auto sA_frag = thr_mma.partition_fragment_A(sA);
    auto sB_frag = thr_mma.partition_fragment_B(sB);
    CUTE_UNROLL
    for (int k = 0; k < size<2>(sA_frag); ++k) {
        cute::gemm(tiled_mma, sA_frag(_, _, k), sB_frag(_, _, k), tC_frag);
        tiled_mma.accumulate_ = UMMA::ScaleOut::One;
    }
}

template <typename TiledMMA, typename TensorA, typename TensorB,
          typename TensorFragC>
CUTE_DEVICE void umma_ts(TiledMMA& tiled_mma, TensorA tA_frag, TensorB sB,
                         TensorFragC tC_frag, bool clear_accum) {
    using namespace cute;
    tiled_mma.accumulate_ =
        clear_accum ? UMMA::ScaleOut::Zero : UMMA::ScaleOut::One;
    ThrMMA thr_mma = tiled_mma.get_slice(_0{});
    auto sB_frag = thr_mma.partition_fragment_B(sB);
    CUTE_UNROLL
    for (int k = 0; k < size<2>(tA_frag); ++k) {
        cute::gemm(tiled_mma, tA_frag(_, _, k), sB_frag(_, _, k), tC_frag);
        tiled_mma.accumulate_ = UMMA::ScaleOut::One;
    }
}

template <int MN, int K, int Swizzle, typename Elem>
constexpr auto make_umma_canonical_k_major_layout() {
    using namespace cute;
    using base_atom_type = std::conditional_t<
        Swizzle == 0 || Swizzle == 16, UMMA::Layout_K_INTER_Atom<Elem>,
        std::conditional_t<
            Swizzle == 32, UMMA::Layout_K_SW32_Atom<Elem>,
            std::conditional_t<
                Swizzle == 64, UMMA::Layout_K_SW64_Atom<Elem>,
                std::conditional_t<Swizzle == 128,
                                   UMMA::Layout_K_SW128_Atom<Elem>, void>>>>;
    return coalesce(tile_to_shape(base_atom_type{}, Shape<Int<MN>, Int<K>>{},
                                  Step<_1, _2>{}),
                    Shape<_1, _1>{});
}
template <class a_type, class b_type, class c_type, int M, int N>
struct SM100_MMA_F16BF16_NOELECT_TraitsBase {
    using ValTypeD = c_type;
    using ValTypeA = a_type;
    using ValTypeB = b_type;
    using ValTypeC = c_type;

    static constexpr int K = 256 / cute::sizeof_bits<ValTypeA>::value;

    using Shape_MNK = cute::Shape<cute::Int<M>, cute::Int<N>, cute::Int<K>>;
    using ThrID = cute::Layout<cute::_1>;
    using ALayout = cute::Layout<
        cute::Shape<cute::_1, cute::Shape<cute::Int<M>, cute::Int<K>>>,
        cute::Stride<cute::_0, cute::Stride<cute::_1, cute::Int<M>>>>;
    using BLayout = cute::Layout<
        cute::Shape<cute::_1, cute::Shape<cute::Int<N>, cute::Int<K>>>,
        cute::Stride<cute::_0, cute::Stride<cute::_1, cute::Int<N>>>>;
    using CLayout = cute::Layout<
        cute::Shape<cute::_1, cute::Shape<cute::Int<M>, cute::Int<N>>>,
        cute::Stride<cute::_0, cute::Stride<cute::_1, cute::Int<M>>>>;
};

template <class a_type, class b_type, class c_type, int M, int N,
          cute::UMMA::Major a_major, cute::UMMA::Major b_major,
          cute::UMMA::ScaleIn a_neg = cute::UMMA::ScaleIn::One,
          cute::UMMA::ScaleIn b_neg = cute::UMMA::ScaleIn::One>
struct SM100_MMA_F16BF16_WS_TS_NOELECT {
    using DRegisters = void;
    using ARegisters = uint64_t[1];
    using BRegisters = uint64_t[1];
    using CRegisters = uint32_t[1];

    CUTE_HOST_DEVICE static void fma(uint32_t const& tmem_a,
                                     uint64_t const& desc_b,
                                     uint32_t const& tmem_c,
                                     uint32_t const& scaleC,
                                     uint64_t const& idescE) {
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %4, 0;\n\t"
            "tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [%1], %2, %3, p, 0; "
            "\n\t"
            "}\n"
            :
            : "r"(tmem_c), "r"(tmem_a), "l"(desc_b),
              "r"(uint32_t(idescE >> Sm100::UmmaIdescShift)), "r"(scaleC));
    }
};

template <class a_type, class b_type, class c_type, int M, int N,
          cute::UMMA::Major a_major, cute::UMMA::Major b_major,
          cute::UMMA::ScaleIn a_neg = cute::UMMA::ScaleIn::One,
          cute::UMMA::ScaleIn b_neg = cute::UMMA::ScaleIn::One>
struct SM100_MMA_F16BF16_WS_SS_NOELECT {
    using DRegisters = void;
    using ARegisters = uint64_t[1];
    using BRegisters = uint64_t[1];
    using CRegisters = uint32_t[1];

    CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                     uint64_t const& desc_b,
                                     uint32_t const& tmem_c,
                                     uint32_t const& scaleC,
                                     uint64_t const& idescE) {
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %4, 0;\n\t"
            "tcgen05.mma.ws.cta_group::1.kind::f16 [%0], %1, %2, %3, p, 0; \n\t"
            "}\n"
            :
            : "r"(tmem_c), "l"(desc_a), "l"(desc_b),
              "r"(uint32_t(idescE >> Sm100::UmmaIdescShift)), "r"(scaleC));
    }
};

template <class a_type, class b_type, class c_type, int M, int N,
          cute::UMMA::Major a_major, cute::UMMA::Major b_major,
          cute::UMMA::ScaleIn a_neg = cute::UMMA::ScaleIn::One,
          cute::UMMA::ScaleIn b_neg = cute::UMMA::ScaleIn::One>
struct SM100_MMA_F16BF16_SS_NOELECT {
    using DRegisters = void;
    using ARegisters = uint64_t[1];
    using BRegisters = uint64_t[1];
    using CRegisters = uint32_t[1];

    CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                     uint64_t const& desc_b,
                                     uint32_t const& tmem_c,
                                     uint32_t const& scaleC,
                                     uint64_t const& idescE) {
        const uint4 mask = make_uint4(0, 0, 0, 0);
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %4, 0;\n\t"
            "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, {%5, %6, %7, "
            "%8}, p; \n\t"
            "}\n"
            :
            : "r"(tmem_c), "l"(desc_a), "l"(desc_b),
              "r"(uint32_t(idescE >> Sm100::UmmaIdescShift)), "r"(scaleC),
              "r"(mask.x), "r"(mask.y), "r"(mask.z), "r"(mask.w));
    }
};
}  // namespace halops::mla
