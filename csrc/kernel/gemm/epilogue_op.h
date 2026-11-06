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

namespace halops {

enum class EpilogueOp : int32_t {
    Identity = 0,
    Quant = 1,
    GateMulQuant = 2,
    Sigmoid = 3,
    SituQuant = 4,
    SigmoidGateMulQuant = 5
};

constexpr bool IsQuantOp(EpilogueOp op) {
    return op == EpilogueOp::Quant || op == EpilogueOp::GateMulQuant ||
           op == EpilogueOp::SituQuant || op == EpilogueOp::SigmoidGateMulQuant;
}

inline constexpr int kGluInterleave = 16;

struct SituCoef {
    float inv_beta = 0.0f;
    float gate_up_scale = 0.0f;
    float inv_linear_beta = 0.0f;
};

}  // namespace halops
