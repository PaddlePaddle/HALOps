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

#include <iterator>

#include "common/ptx.cuh"

namespace halops {

CUTE_DEVICE float sigmoid(float x) {
    return fmaf(0.5f, tanh_approx(0.5f * x), 0.5f);
}

CUTE_DEVICE float2 situ_glu(float2 gate, float2 up, float inv_beta,
                            float gate_up_scale, float inv_linear_beta) {
    constexpr float kSituPolyBeta = 4.0f;
    constexpr float kHalfRcpPoly[] = {0.0761225f, -0.26642883f, 0.43294686f,
                                      -0.49241751f, 0.49985132f};
    const float2 t =
        tanh_approx(__fmul2_rn(gate, make_float2(inv_beta, inv_beta)));
    float2 g;
    if (inv_beta == 1.0f / kSituPolyBeta) {
        const float2 u = __fmul2_rn(t, t);
        const float2 s = __fadd2_rn(t, make_float2(1.0f, 1.0f));
        constexpr int kTerms = std::size(kHalfRcpPoly);
        float2 p = make_float2(kHalfRcpPoly[0], kHalfRcpPoly[0]);
        CUTLASS_PRAGMA_UNROLL
        for (int i = 1; i < kTerms; ++i) {
            p = __ffma2_rn(p, u, make_float2(kHalfRcpPoly[i], kHalfRcpPoly[i]));
        }
        g = __fmul2_rn(__fmul2_rn(__fmul2_rn(t, s), s), p);
    } else {
        g = __fmul2_rn(t,
                       make_float2(__fdividef(1.0f, 1.0f + __expf(-gate.x)),
                                   __fdividef(1.0f, 1.0f + __expf(-gate.y))));
    }
    return __fmul2_rn(
        __fmul2_rn(g, make_float2(gate_up_scale, gate_up_scale)),
        inv_linear_beta != 0.0f
            ? tanh_approx(
                  __fmul2_rn(up, make_float2(inv_linear_beta, inv_linear_beta)))
            : up);
}

}  // namespace halops
