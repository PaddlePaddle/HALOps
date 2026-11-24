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
#include <cutlass/fast_math.h>

#include <cstdint>
#include <cute/tensor.hpp>

namespace halops::moe {

enum class MmaKind : uint32_t {
    kUmma = 0,
    kHmma = 1,
};

enum class QuantMethod : uint32_t {
    kBlockwiseFP8 = 0,
    kMXFP8 = 1,
    kMXFP4 = 2,
    kNone = 3,
};

enum class ScaleLayout : std::uint32_t {
    kSfAtom,  // natural-row SfAtom from group_wise_quant(use_swizzle=True)
    kPlain,   // plain row-major from group_wise_quant(use_swizzle=False)
};

struct MoELayouts {
    ScaleLayout routed_down_scale = ScaleLayout::kSfAtom;
    ScaleLayout shared_down_scale = ScaleLayout::kPlain;
    ScaleLayout latent_down_scale = ScaleLayout::kPlain;
    ScaleLayout tail_fc2_scale = ScaleLayout::kPlain;

    constexpr MoELayouts() = default;

    constexpr MoELayouts(bool routed_use_swizzle, bool shared_use_swizzle,
                         bool latent_use_swizzle)
        : routed_down_scale(routed_use_swizzle ? ScaleLayout::kSfAtom
                                               : ScaleLayout::kPlain),
          shared_down_scale(shared_use_swizzle ? ScaleLayout::kSfAtom
                                               : ScaleLayout::kPlain),
          latent_down_scale(latent_use_swizzle ? ScaleLayout::kSfAtom
                                               : ScaleLayout::kPlain),
          tail_fc2_scale(latent_use_swizzle ? ScaleLayout::kSfAtom
                                            : ScaleLayout::kPlain) {}
};

// Operand quantization types and group width. MXFP8/MXFP4 use fixed 32-element
// groups.
template <QuantMethod Method,
          std::uint32_t BlockSize =
              (Method == QuantMethod::kBlockwiseFP8 ? 128u : 32u)>
struct QuantSpec;

template <std::uint32_t BlockSize>
struct QuantSpec<QuantMethod::kNone, BlockSize> {
    static constexpr QuantMethod kMethod = QuantMethod::kNone;
    static constexpr std::uint32_t kBlockSize = 0u;
    using ElementType = __nv_bfloat16;
    using ScaleType = void;
};

template <std::uint32_t BlockSize>
struct QuantSpec<QuantMethod::kBlockwiseFP8, BlockSize> {
    static constexpr QuantMethod kMethod = QuantMethod::kBlockwiseFP8;
    static constexpr std::uint32_t kBlockSize = BlockSize;
    using ElementType = __nv_fp8_e4m3;
    using ScaleType = float;

    template <std::uint32_t M, std::uint32_t N>
    __host__ __device__ static constexpr auto ScaleShape() {
        return cute::make_shape(cute::Int<(M + BlockSize - 1u) / BlockSize>{},
                                cute::Int<(N + BlockSize - 1u) / BlockSize>{});
    }
};

template <std::uint32_t BlockSize>
struct QuantSpec<QuantMethod::kMXFP8, BlockSize> {
    static constexpr QuantMethod kMethod = QuantMethod::kMXFP8;
    static constexpr std::uint32_t kBlockSize = 32u;
    using ElementType = __nv_fp8_e4m3;
    using ScaleType = std::uint8_t;  // one UE8M0 byte per 32-K group

    template <std::uint32_t M, std::uint32_t N>
    __host__ __device__ static constexpr auto ScaleShape() {
        static_assert(N % kBlockSize == 0u,
                      "K must be whole 32-element scale groups");
        return cute::make_shape(cute::Int<M>{}, cute::Int<N / kBlockSize>{});
    }
};

template <std::uint32_t BlockSize>
struct QuantSpec<QuantMethod::kMXFP4, BlockSize> {
    static constexpr QuantMethod kMethod = QuantMethod::kMXFP4;
    static constexpr std::uint32_t kBlockSize = 32u;

    // TMA expands each FP4 weight into one SMEM byte; scales use E8M0.
    using ElementType = cutlass::float_e2m1_t;
    using ScaleType = std::uint8_t;  // one UE8M0 byte per 32-K group

    template <std::uint32_t M, std::uint32_t N>
    __host__ __device__ static constexpr auto ScaleShape() {
        static_assert(N % kBlockSize == 0u,
                      "K must be whole 32-element scale groups");
        return cute::make_shape(cute::Int<M>{}, cute::Int<N / kBlockSize>{});
    }
};

template <QuantMethod LatentQuant = QuantMethod::kBlockwiseFP8,
          QuantMethod SharedQuant = QuantMethod::kBlockwiseFP8,
          QuantMethod ExpertsQuant = QuantMethod::kBlockwiseFP8>
struct QuantConfig {
    static constexpr QuantMethod kLatentQuant = LatentQuant;
    static constexpr QuantMethod kSharedQuant = SharedQuant;
    static constexpr QuantMethod kExpertsQuant = ExpertsQuant;
    static constexpr QuantMethod kExpertsActQuant =
        ExpertsQuant == QuantMethod::kMXFP4 ? QuantMethod::kMXFP8
                                            : ExpertsQuant;
    static constexpr QuantMethod kTailQuant =
        LatentQuant == QuantMethod::kBlockwiseFP8 ? QuantMethod::kBlockwiseFP8
                                                  : QuantMethod::kMXFP8;
    static constexpr std::uint32_t kBlockSize = 128;
};

template <QuantConfig Config, std::uint32_t NumTokens, int TailWorld = 4>
struct MoEDims_E512_N512_K2048 {
    static constexpr uint32_t kHiddenSize = 4096;
    static constexpr std::uint32_t kLatentSize = 2048;
    static constexpr uint32_t kNumTokens = NumTokens;
    static constexpr uint32_t kNumExperts = 512;
    static constexpr uint32_t kMaxTopk = 10;
    static constexpr QuantConfig kQuantConfig = Config;

    // The schedule serves a narrow (8-token) or a wide (16-token) batch.
    static_assert(NumTokens == 8u || NumTokens == 16u,
                  "the schedule serves 8 or 16 tokens");

    struct KernelConfig {
        static constexpr std::uint32_t kBlockSize = 384;
        static constexpr std::uint32_t kSpecMaxGrid = 256u;
        // Router configs.
        static constexpr std::uint32_t kNumCtasForRouter = 32;
        static constexpr MmaKind kRouterMma = MmaKind::kHmma;
        // LatentDown configs.
        static constexpr std::uint32_t kNumCtasForLatentDown = 32;
        // SharedExpert configs.
        static constexpr std::uint32_t kNumCtasForShared = 32;
        // MoE configs.
        static constexpr std::uint32_t kCtasPerExpert = 8;
        // Tail configs.
        static constexpr int kTailWorld = TailWorld;
        static constexpr int kTailTpg = NumTokens <= 8u ? 2 : 4;
        static constexpr int kTailGsize = 32;
        static constexpr int kTailThreads = 256;
        static constexpr int kTailStage = 1;
        static constexpr int kTailBlockRows = 32;
        static constexpr int kTailBlkB = 4;
        static constexpr int kTailWStagePad = 64;
        static constexpr int kTailXPad = 8;
        static constexpr int kTailMaxStageSmemBytes = 160 * 1024;
        static constexpr std::uint32_t kTailPrefetchChunk = 16384;
    };
};

// Scoring function for the router.
enum class ScoringFunc : uint32_t {
    kSigmoid = 0,
    kSoftmax = 1,
};

// Gated activation applied before output quantization.
enum class Activation : uint32_t {
    kSilu = 0,  // silu(gate) * up -- unbounded
    kSitu = 1,  // SiTU-GLU -- output bounded by beta * linear_beta
};

struct DataType {
    using ElementA = __nv_bfloat16;    // activations as they go into the GEMM
    using ElementQA = __nv_fp8_e4m3;   // activations after quantization
    using ElementOut = __nv_bfloat16;  // MoE output
};

struct MoEProblemSize {
    constexpr MoEProblemSize() = default;
    __host__ __device__ constexpr MoEProblemSize(
        std::uint32_t num_tokens, std::uint32_t hidden_size,
        std::uint32_t intermediate_size, std::uint32_t latent_size,
        std::uint32_t num_experts, std::uint32_t num_router_gates)
        : num_tokens(num_tokens),
          hidden_size(hidden_size),
          intermediate_size(intermediate_size),
          latent_size(latent_size),
          num_experts(num_experts),
          num_router_gates(num_router_gates),
          use_latent(latent_size != 0u) {}

    std::uint32_t num_tokens = 0u;
    std::uint32_t hidden_size = 0u;
    std::uint32_t intermediate_size = 0u;
    std::uint32_t latent_size = 0u;
    std::uint32_t num_experts = 0u;
    std::uint32_t num_router_gates = 0u;
    // The latent projection runs whenever a latent width is given.
    bool use_latent = false;

    __host__ __device__ constexpr std::uint32_t input_size() const {
        return latent_size != 0u ? latent_size : hidden_size;
    }
};

// The launch-time kernel layout: split groups, the phase CTA ranges, the
// launch cluster and the router's gate count. The host fills it per call.
struct MoERuntime {
    static constexpr std::uint32_t kMaxClusterSize = 4u;
    static constexpr std::uint32_t kMaxRouterGates = 2u;
    // The fixed per-expert buffers (SMEM task grid, workspace counters) keep
    // this capacity; the live count below may be smaller.
    static constexpr std::uint32_t kMaxCtasPerExpert = 8u;

    // Split-K groups: one GEMM reduction's consecutive CTAs.
    std::uint32_t num_latent_splits = 1u;
    std::uint32_t num_shared_splits = 1u;
    // The launch cluster; a split group never crosses a cluster boundary.
    std::uint32_t cluster_size = 1u;

    // CTAs per router gate; the phase spans gates x this many blocks.
    std::uint32_t num_router_gates = 0u;
    std::uint32_t num_router_ctas = 0u;
    // Blocks that share one routed-expert task; the divider must be
    // initialised on the host from `num_ctas_per_expert`.
    std::uint32_t num_ctas_per_expert = 1u;
    cutlass::FastDivmod ctas_per_expert_divmod{};

    // Host-precomputed dividers (see the helpers below).
    cutlass::FastDivmod latent_splits_divmod{};
    cutlass::FastDivmod shared_splits_divmod{};
    cutlass::FastDivmod latent_tiles_divmod{};
    cutlass::FastDivmod cluster_divmod{};
    std::uint32_t num_latent_ctas = 0u;
    std::uint32_t first_shared_cta = 0u;
    std::uint32_t num_shared_ctas = 0u;
    std::uint32_t num_tail_ctas = 0u;

    constexpr MoERuntime() = default;

    __host__ __device__ constexpr MoERuntime(
        MoEProblemSize const& problem_size, const std::uint32_t num_router_ctas,
        const std::uint32_t num_latent_ctas,
        const std::uint32_t num_shared_ctas)
        : num_router_gates(problem_size.num_router_gates),
          num_router_ctas(num_router_ctas),
          num_latent_ctas(num_latent_ctas),
          first_shared_cta(num_router_ctas * problem_size.num_router_gates +
                           num_latent_ctas),
          num_shared_ctas(num_shared_ctas) {}

    // The smallest grid in which every phase has its CTAs resident.
    __host__ __device__ constexpr std::uint32_t get_min_grid() const {
        return (first_shared_cta + num_shared_ctas) > num_tail_ctas
                   ? (first_shared_cta + num_shared_ctas)
                   : num_tail_ctas;
    }

    __device__ bool is_router_cta() const {
        return blockIdx.x < num_router_ctas * num_router_gates;
    }

    __device__ bool is_latent_cta() const {
        const std::uint32_t first_latent_cta =
            num_router_ctas * num_router_gates;
        return blockIdx.x >= first_latent_cta &&
               blockIdx.x < first_latent_cta + num_latent_ctas;
    }

    __device__ bool is_shared_cta() const {
        return blockIdx.x >= first_shared_cta &&
               blockIdx.x < first_shared_cta + num_shared_ctas;
    }
};

// Byte offset of a scale in a 128-row, four-group SfAtom.
__host__ __device__ static constexpr std::uint32_t sf_atom_offset(
    std::uint32_t mn, std::uint32_t k) {
    return (mn % 32u) * 16u + ((mn / 32u) % 4u) * 4u + (k % 4u) +
           (k / 4u) * 512u;
}

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

}  // namespace halops::moe
