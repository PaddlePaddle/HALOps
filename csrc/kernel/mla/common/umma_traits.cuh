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

#include <cute/tensor.hpp>

#include "kernel/mla/common/umma.cuh"

namespace cute {
template <class a_type, class b_type, class c_type, int M, int N,
          UMMA::Major a_major, UMMA::Major b_major, UMMA::ScaleIn a_neg,
          UMMA::ScaleIn b_neg>
struct MMA_Traits<halops::mla::SM100_MMA_F16BF16_WS_TS_NOELECT<
    a_type, b_type, c_type, M, N, a_major, b_major, a_neg, b_neg>>
    : halops::mla::SM100_MMA_F16BF16_NOELECT_TraitsBase<a_type, b_type, c_type,
                                                        M, N> {
    using FrgTypeA =
        UMMA::tmem_frg_1sm<a_type, a_type, UMMA::TmemAllocMode::NonInterleaved>;
    using FrgTypeB = UMMA::smem_desc<b_major>;
    using FrgTypeC = UMMA::tmem_frg_ws_1sm<c_type>;

    UMMA::ScaleOut accumulate_ = UMMA::ScaleOut::One;
    UMMA::InstrDescriptor idesc_ =
        UMMA::make_instr_desc<a_type, b_type, c_type, M, N, a_major, b_major,
                              a_neg, b_neg>();

    template <class TD, class DLayout, class TA, class ALayout_, class TB,
              class BLayout_, class TC, class CLayout_>
    CUTE_HOST_DEVICE constexpr friend void mma_unpack(
        MMA_Traits const& traits, Tensor<TD, DLayout>& tensor_d,
        Tensor<TA, ALayout_> const& tensor_a,
        Tensor<TB, BLayout_> const& tensor_b,
        Tensor<TC, CLayout_> const& tensor_c) {
        uint32_t tmem_a = raw_pointer_cast(tensor_a.data());
        uint64_t desc_b = tensor_b[0];
        uint32_t tmem_c = raw_pointer_cast(tensor_d.data());
        uint64_t idesc = UMMA::make_runtime_instr_desc<>(traits.idesc_);
        halops::mla::SM100_MMA_F16BF16_WS_TS_NOELECT<
            a_type, b_type, c_type, M, N, a_major, b_major, a_neg,
            b_neg>::fma(tmem_a, desc_b, tmem_c, uint32_t(traits.accumulate_),
                        idesc);
    }
};

template <class a_type, class b_type, class c_type, int M, int N,
          UMMA::Major a_major, UMMA::Major b_major, UMMA::ScaleIn a_neg,
          UMMA::ScaleIn b_neg>
struct MMA_Traits<halops::mla::SM100_MMA_F16BF16_WS_SS_NOELECT<
    a_type, b_type, c_type, M, N, a_major, b_major, a_neg, b_neg>>
    : halops::mla::SM100_MMA_F16BF16_NOELECT_TraitsBase<a_type, b_type, c_type,
                                                        M, N> {
    using FrgTypeA = UMMA::smem_desc<a_major>;
    using FrgTypeB = UMMA::smem_desc<b_major>;
    using FrgTypeC = UMMA::tmem_frg_ws_1sm<c_type>;

    UMMA::InstrDescriptor idesc_ =
        UMMA::make_instr_desc<a_type, b_type, c_type, M, N, a_major, b_major,
                              a_neg, b_neg>();
    UMMA::ScaleOut accumulate_ = UMMA::ScaleOut::One;

    template <class TD, class DLayout, class TA, class ALayout_, class TB,
              class BLayout_, class TC, class CLayout_>
    CUTE_HOST_DEVICE constexpr friend void mma_unpack(
        MMA_Traits const& traits, Tensor<TD, DLayout>& tensor_d,
        Tensor<TA, ALayout_> const& tensor_a,
        Tensor<TB, BLayout_> const& tensor_b,
        Tensor<TC, CLayout_> const& tensor_c) {
        uint64_t desc_a = tensor_a[0];
        uint64_t desc_b = tensor_b[0];
        uint32_t tmem_c = raw_pointer_cast(tensor_d.data());
        uint64_t idesc = UMMA::make_runtime_instr_desc<>(traits.idesc_);
        halops::mla::SM100_MMA_F16BF16_WS_SS_NOELECT<
            a_type, b_type, c_type, M, N, a_major, b_major, a_neg,
            b_neg>::fma(desc_a, desc_b, tmem_c, uint32_t(traits.accumulate_),
                        idesc);
    }
};

template <class a_type, class b_type, class c_type, int M, int N,
          UMMA::Major a_major, UMMA::Major b_major, UMMA::ScaleIn a_neg,
          UMMA::ScaleIn b_neg>
struct MMA_Traits<halops::mla::SM100_MMA_F16BF16_SS_NOELECT<
    a_type, b_type, c_type, M, N, a_major, b_major, a_neg, b_neg>>
    : halops::mla::SM100_MMA_F16BF16_NOELECT_TraitsBase<a_type, b_type, c_type,
                                                        M, N> {
    using FrgTypeA = UMMA::smem_desc<a_major>;
    using FrgTypeB = UMMA::smem_desc<b_major>;
    using FrgTypeC = UMMA::tmem_frg_1sm<c_type>;

    UMMA::InstrDescriptor idesc_ =
        UMMA::make_instr_desc<a_type, b_type, c_type, M, N, a_major, b_major,
                              a_neg, b_neg>();

    UMMA::ScaleOut accumulate_ = UMMA::ScaleOut::One;

    template <class TD, class DLayout, class TA, class ALayout_, class TB,
              class BLayout_, class TC, class CLayout_>
    CUTE_HOST_DEVICE constexpr friend void mma_unpack(
        MMA_Traits const& traits, Tensor<TD, DLayout>& tensor_d,
        Tensor<TA, ALayout_> const& tensor_a,
        Tensor<TB, BLayout_> const& tensor_b,
        Tensor<TC, CLayout_> const& tensor_c) {
        uint64_t desc_a = tensor_a[0];
        uint64_t desc_b = tensor_b[0];
        uint32_t tmem_c = raw_pointer_cast(tensor_d.data());
        uint64_t idesc = UMMA::make_runtime_instr_desc<>(traits.idesc_);
        halops::mla::SM100_MMA_F16BF16_SS_NOELECT<
            a_type, b_type, c_type, M, N, a_major, b_major, a_neg,
            b_neg>::fma(desc_a, desc_b, tmem_c, uint32_t(traits.accumulate_),
                        idesc);
    }
};
}  // namespace cute
