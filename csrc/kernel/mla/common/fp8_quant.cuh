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

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cutlass/numeric_types.h>

#include <cute/tensor.hpp>

#include "kernel/mla/common/traits.cuh"

namespace halops::mla {
struct Fp8Format {
    static constexpr float E4m3Max = 448.0f;
    static constexpr int Ue8m0Bias =
        cutlass::detail::exponent_bias_cxx17<8, 0>();
    static constexpr float Ue8m0MaxExp = 2 * Ue8m0Bias;
    static constexpr uint64_t UnitScales = 0x0101010101010101ull * Ue8m0Bias;
    static constexpr float AmaxFloor = 1e-10f;
    static constexpr int F32MantissaBits = 23;
    static constexpr int PairBits = 16;
    static constexpr int PerWord = 4;
};

CUTE_DEVICE float ue8m0_exp_ceil(float amax) {
    return fminf(
        fmaxf(ceilf(log2f(amax / Fp8Format::E4m3Max)) + Fp8Format::Ue8m0Bias,
              0.0f),
        Fp8Format::Ue8m0MaxExp);
}

template <int TileTokens>
CUTE_HOST_DEVICE constexpr int sf_row_of(int token) {
    if constexpr (TileTokens < SfLayout::PackedMinTile)
        return (token / TileTokens) * SfLayout::MnPerTile +
               (token % TileTokens);
    else
        return token;
}

CUTE_HOST_DEVICE int sf_row_grouped(int token, int group_tokens, int n_groups) {
    if (n_groups > 1 && group_tokens % SfLayout::MnPerTile) {
        const int atom_rows =
            cute::round_up(group_tokens, int{SfLayout::MnPerTile});
        return (token / group_tokens) * atom_rows + token % group_tokens;
    }
    return token;
}

CUTE_DEVICE float ue8m0_quant_scale(float biased_exp) {
    const int exp_bits = (int)biased_exp;
    return exp_bits < Fp8Format::Ue8m0MaxExp
               ? __int_as_float((2 * Fp8Format::Ue8m0Bias - exp_bits)
                                << Fp8Format::F32MantissaBits)
               : exp2f(Fp8Format::Ue8m0Bias - biased_exp);
}
}  // namespace halops::mla
