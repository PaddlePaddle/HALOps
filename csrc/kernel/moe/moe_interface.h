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

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>

namespace halops::moe {

// Tensor-core instruction family of an MMA.
enum class MmaKind : uint32_t {
    kUmma = 0,  // tcgen05 UMMA on the MMA issuer warp; calc warps stay free
    kHmma = 1,  // warp-level mma.sync on the calc warps
};

template <bool UseLatent = true>
struct Dims_BS8_E512_N512_K2048_SM100 {
    static constexpr uint32_t kHiddenSize = 4096;
    static constexpr uint32_t kIntermediateSize = 512;
    static constexpr std::uint32_t kLatentSize = 2048;
    static constexpr uint32_t kNumTokens = 8;
    static constexpr uint32_t kNumExperts = 512;
    static constexpr uint32_t kMaxTopk = 10;
    static constexpr uint32_t kQuantBlockM = 128;
    static constexpr uint32_t kQuantBlockK = 128;
    static constexpr bool kUseLatent = UseLatent;
    static constexpr std::uint32_t kNumRouterGates = 2;
    struct KernelConfig {
        static constexpr std::uint32_t kBlockSize = 384;
        static constexpr std::uint32_t kNumInputSlots = 8;
        // Router configs.
        static constexpr std::uint32_t kNumRouterWeightSlots = 16;
        static constexpr std::uint32_t kNumCtasForRouter = 32;
        static constexpr MmaKind kRouterMma = MmaKind::kHmma;
        // LatentDown configs.
        static constexpr std::uint32_t kNumLatentDownWeightSlots = 16;
        static constexpr std::uint32_t kNumCtasForLatentDown = 64;
        static constexpr MmaKind kLatentDownMma = MmaKind::kHmma;
        static constexpr bool kLatentDownUseSplitK = false;
        static constexpr bool kLatentQuantizeOutput = true;
        // SharedExpert configs.
        static constexpr bool kUseSharedExpert = true;
        static constexpr std::uint32_t kNumCtasForShared = 16;
        static constexpr bool kSharedUseSplitK = false;
        static constexpr std::uint32_t kNumSharedUpWeightSlots = 4;
        static constexpr std::uint32_t kNumSharedDownWeightSlots = 4;
        // MoE configs.
        static constexpr std::uint32_t kCtasPerExpert = 8;
        static constexpr std::uint32_t kUpTileShapeK = 256;
        static constexpr std::uint32_t kNumUpWeightSlots = 4;
        static constexpr std::uint32_t kNumDownWeightSlots = 4;
        static constexpr std::uint32_t kNumAccSlots = 4;
        static constexpr bool kCombineWaitPerToken = true;
    };
};

template <bool UseLatent = true>
struct Dims_BS16_E512_N512_K2048_SM100
    : Dims_BS8_E512_N512_K2048_SM100<UseLatent> {
    static constexpr uint32_t kNumTokens = 16;
    struct KernelConfig
        : Dims_BS8_E512_N512_K2048_SM100<UseLatent>::KernelConfig {
        static constexpr std::uint32_t kNumInputSlots = 8;
        static constexpr std::uint32_t kNumUpInputSlots = 4;
        // Router configs.
        static constexpr std::uint32_t kNumCtasForRouter = 32;
        static constexpr std::uint32_t kNumRouterWeightSlots = 8;
        static constexpr std::uint32_t kNumRouterInputSlots = 16;
        static constexpr MmaKind kRouterMma = MmaKind::kHmma;
        // LatentDown configs.
        static constexpr std::uint32_t kNumCtasForLatentDown = 64;
        static constexpr std::uint32_t kNumLatentDownWeightSlots = 8;
        static constexpr MmaKind kLatentDownMma = MmaKind::kUmma;
        static constexpr bool kLatentDownUseSplitK = true;
        // SharedExpert configs.
        static constexpr bool kSharedUseSplitK = true;
    };
};

// Scoring function for the router.
enum class ScoringFunc : uint32_t {
    kSigmoid = 0,
    kSoftmax = 1,
};

// Expert activation of the up-projection epilogue: how the gate and up halves
// of the expert's fc1 output are combined before re-quantization.
enum class Activation : uint32_t {
    kSilu = 0,  // silu(gate) * up -- unbounded
    kSitu = 1,  // SiTU-GLU -- output bounded by beta * linear_beta
};

// The MMA's B operand walks K in 16-B chunks, its descriptor's leading byte
// offset.
inline constexpr std::uint32_t kFp8ActChunkK = 16u;

struct DataType {
    using ElementW = __nv_fp8_e4m3;    // expert weights
    using ElementA = __nv_bfloat16;    // activations as they go into the GEMM
    using ElementQA = __nv_fp8_e4m3;   // activations after quantization
    using ElementScale = float;        // scaling factors
    using ElementOut = __nv_bfloat16;  // MoE output
    using ElementAcc = float;          // fp32 accumulators
};

}  // namespace halops::moe
