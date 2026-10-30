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
#include <cutlass/cutlass.h>

#include <cstdint>

#define FULL_MASK 0xFFFFFFFFU

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
        static constexpr std::uint32_t kNumRouterInputSlots = 8;
        static constexpr std::uint32_t kNumLatentInputSlots = 8;
        static constexpr std::uint32_t kNumUpInputSlots = 8;
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

template <bool kCondition, std::uint32_t kTrue, std::uint32_t kFalse>
struct Select {
    static constexpr std::uint32_t value = kCondition ? kTrue : kFalse;
};

// Width the experts reduce over (`up`) and produce (`down`).
template <typename Dims>
struct MoEInputSize {
    static constexpr std::uint32_t value =
        Dims::kUseLatent ? Dims::kLatentSize : Dims::kHiddenSize;
};

// The router's token axis is cut into instruction-wide windows, one CTA each.
template <typename Dims>
struct RouterTokenWindows {
    static constexpr std::uint32_t kWindow = 8u;
    static_assert(Dims::kNumTokens % kWindow == 0u,
                  "the token width must be whole 8-token windows");
    static constexpr std::uint32_t value = Dims::kNumTokens / kWindow;
};

struct WarpRole {
    static constexpr std::uint32_t kThreadsPerWarp = cutlass::NumThreadsPerWarp;

    static constexpr std::uint32_t kCalcWarpCount = 8;

    static constexpr std::uint32_t kMmaIssuerWarpId = 8;
    static constexpr std::uint32_t kWeightLoaderWarpId = 9;
    static constexpr std::uint32_t kOtherLoaderWarpId = 10;
    static constexpr std::uint32_t kPublishWarpId = 11;
    static constexpr std::uint32_t kTmemAllocWarpId = 0;

    static constexpr bool is_calc_warp(std::uint32_t warp_idx) {
        return warp_idx < kCalcWarpCount;
    }

    static constexpr bool is_weight_loader_warp(std::uint32_t warp_idx) {
        return warp_idx == kWeightLoaderWarpId;
    }

    static constexpr bool is_mma_issuer_warp(std::uint32_t warp_idx) {
        return warp_idx == kMmaIssuerWarpId;
    }

    static constexpr bool is_tmem_warp(std::uint32_t warp_idx) {
        return warp_idx == kTmemAllocWarpId;
    }

    static constexpr bool is_other_loader_warp(std::uint32_t warp_idx) {
        return warp_idx == kOtherLoaderWarpId;
    }

    static constexpr bool is_publish_warp(std::uint32_t warp_idx) {
        return warp_idx == kPublishWarpId;
    }

    static inline __device__ std::uint32_t get_lane_idx() {
        return cutlass::canonical_lane_idx();
    }

    static inline __device__ std::uint32_t get_warp_idx() {
        return cutlass::canonical_warp_idx();
    }

    static inline __device__ bool is_calc_thread() {
        return is_calc_warp(get_warp_idx());
    }

    static inline __device__ std::uint32_t get_calc_warp_idx() {
        assert(is_calc_thread());
        return get_warp_idx();
    }

    static inline __device__ bool is_weight_loader_thread() {
        return is_weight_loader_warp(get_warp_idx()) && get_lane_idx() == 0u;
    }

    static inline __device__ bool is_other_loader_thread() {
        return is_other_loader_warp(get_warp_idx()) && get_lane_idx() == 0u;
    }

    // Synchronizes the first 256 threads (the calc warps) of the block.
    static inline __device__ void sync_calc_threads() {
        static_assert(kCalcWarpCount * kThreadsPerWarp == 256,
                      "Adapt the thread number in sync_calc_threads");
        __asm volatile("bar.sync  15, 256;\n");
    }

    // Named-barrier (14) sync of the whole warps [warp_idx_begin, +num_warps).
    template <int warp_idx_begin, int num_warps>
    static inline __device__ void sync_warps() {
        static_assert(num_warps >= 1, "a named barrier needs an arriver.");
        static_assert(warp_idx_begin >= 0 && warp_idx_begin + num_warps <= 32,
                      "a block has at most 32 warps.");
        constexpr std::uint32_t kThreads =
            static_cast<std::uint32_t>(num_warps) * kThreadsPerWarp;
        asm volatile("bar.sync 14, %0;\n" ::"n"(kThreads) : "memory");
    }
};

// Upper bound on the launch grid.
constexpr std::uint32_t kSpecMaxGrid = 256u;

}  // namespace halops::moe
