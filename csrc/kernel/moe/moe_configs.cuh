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

#include <cutlass/cutlass.h>

#include <cstdint>

#include "moe_interface.h"

#define FULL_MASK 0xFFFFFFFFU

namespace halops::moe {

// A register fragment of `N` elements of `T`, left uninitialized.
template <typename T, std::uint32_t N>
struct Fragment {
    T data[N];
    __device__ __forceinline__ T& operator[](std::uint32_t i) {
        return data[i];
    }
    __device__ __forceinline__ T operator[](std::uint32_t i) const {
        return data[i];
    }
};

// A register fragment of floats, zero-initialized on construction.
template <std::uint32_t R, std::uint32_t C = 1u>
struct ZeroedFragment {
    float data[R][C];
    __device__ __forceinline__ ZeroedFragment() {
#pragma unroll
        for (std::uint32_t i = 0; i < R; ++i) {
#pragma unroll
            for (std::uint32_t j = 0; j < C; ++j) {
                data[i][j] = 0.f;
            }
        }
    }
    __device__ __forceinline__ float (&operator[](std::uint32_t i))[C] {
        return data[i];
    }
    __device__ __forceinline__ const float (&operator[](std::uint32_t i)
                                                const)[C] {
        return data[i];
    }
};

template <std::uint32_t N>
struct ZeroedFragment<N, 1u> {
    float data[N];
    __device__ __forceinline__ ZeroedFragment() {
#pragma unroll
        for (std::uint32_t i = 0; i < N; ++i) {
            data[i] = 0.f;
        }
    }
    __device__ __forceinline__ float& operator[](std::uint32_t i) {
        return data[i];
    }
    __device__ __forceinline__ float operator[](std::uint32_t i) const {
        return data[i];
    }
};

// Down-proj weight/activation TMA pipeline depth (`kNumDownWeightSlots`).
template <typename Dims>
struct NumDownWeightSlots {
  private:
    template <typename D>
    static constexpr auto test(int) -> decltype(static_cast<std::uint32_t>(
        D::KernelConfig::kNumDownWeightSlots)) {
        return static_cast<std::uint32_t>(D::KernelConfig::kNumDownWeightSlots);
    }
    template <typename D>
    static constexpr std::uint32_t test(...) {
        return D::kIntermediateSize / 128u;  // the down tile's K
    }

  public:
    static constexpr std::uint32_t value = test<Dims>(0);
};

// CTAs sharing one expert.
template <typename Dims>
struct CtasPerExpert {
  private:
    template <typename D>
    static constexpr auto test(int) -> decltype(static_cast<std::uint32_t>(
        D::KernelConfig::kCtasPerExpert)) {
        return static_cast<std::uint32_t>(D::KernelConfig::kCtasPerExpert);
    }
    template <typename>
    static constexpr std::uint32_t test(...) {
        return (Dims::kIntermediateSize * 2) / 128u;
    }

  public:
    static constexpr std::uint32_t value = test<Dims>(0);
};

template <bool kCondition, std::uint32_t kTrue, std::uint32_t kFalse>
struct Select {
    static constexpr std::uint32_t value = kCondition ? kTrue : kFalse;
};

// CTAs that split the router Linear's expert dimension, one M tile each.
template <typename Dims>
struct NumCtasForRouter {
  private:
    template <typename D>
    static constexpr auto test(int) -> decltype(static_cast<std::uint32_t>(
        D::KernelConfig::kNumCtasForRouter)) {
        return static_cast<std::uint32_t>(D::KernelConfig::kNumCtasForRouter);
    }
    template <typename>
    static constexpr std::uint32_t test(...) {
        return Dims::kNumExperts / 16u;  // the shipped 16-expert tile
    }

  public:
    static constexpr std::uint32_t value = test<Dims>(0);
};

// Depth of the router weight ring (`kNumRouterWeightSlots`).
template <typename Dims>
struct NumRouterWeightSlots {
  private:
    template <typename D>
    static constexpr auto test(int) -> decltype(static_cast<std::uint32_t>(
        D::KernelConfig::kNumRouterWeightSlots)) {
        return static_cast<std::uint32_t>(
            D::KernelConfig::kNumRouterWeightSlots);
    }
    template <typename>
    static constexpr std::uint32_t test(...) {
        return Dims::kHiddenSize / 256u;  // kNumTilesK at the shipped K tile
    }

  public:
    static constexpr std::uint32_t value = test<Dims>(0);
};
// Depth of the streamed-input ring (`kNumInputSlots`).
template <typename Dims>
struct NumInputSlots {
  private:
    template <typename D>
    static constexpr auto test(int) -> decltype(static_cast<std::uint32_t>(
        D::KernelConfig::kNumInputSlots)) {
        return static_cast<std::uint32_t>(D::KernelConfig::kNumInputSlots);
    }
    template <typename D>
    static constexpr std::uint32_t test(...) {
        // One slot per router K tile (256 K) of the latent output.
        return D::kLatentSize / 256u;
    }

  public:
    static constexpr std::uint32_t value = test<Dims>(0);
};

// Streamed-input ring depth of the router (`kNumInputSlots`).
template <typename Dims>
struct NumRouterInputSlots {
  private:
    template <typename D>
    static constexpr auto test(int) -> decltype(static_cast<std::uint32_t>(
        D::KernelConfig::kNumRouterInputSlots)) {
        return static_cast<std::uint32_t>(
            D::KernelConfig::kNumRouterInputSlots);
    }
    template <typename>
    static constexpr std::uint32_t test(...) {
        return NumInputSlots<Dims>::value;
    }

  public:
    static constexpr std::uint32_t value = test<Dims>(0);
};

template <typename Dims>
struct NumLatentInputSlots {
  private:
    template <typename D>
    static constexpr auto test(int) -> decltype(static_cast<std::uint32_t>(
        D::KernelConfig::kNumLatentInputSlots)) {
        return static_cast<std::uint32_t>(
            D::KernelConfig::kNumLatentInputSlots);
    }
    template <typename>
    static constexpr std::uint32_t test(...) {
        return NumInputSlots<Dims>::value;
    }

  public:
    static constexpr std::uint32_t value = test<Dims>(0);
};

template <typename Dims>
struct NumUpInputSlots {
  private:
    template <typename D>
    static constexpr auto test(int) -> decltype(static_cast<std::uint32_t>(
        D::KernelConfig::kNumUpInputSlots)) {
        return static_cast<std::uint32_t>(D::KernelConfig::kNumUpInputSlots);
    }
    template <typename>
    static constexpr std::uint32_t test(...) {
        return NumInputSlots<Dims>::value;
    }

  public:
    static constexpr std::uint32_t value = test<Dims>(0);
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

template <typename Dims>
struct UpTileShapeK {
  private:
    template <typename D>
    static constexpr auto test(int) -> decltype(static_cast<std::uint32_t>(
        D::KernelConfig::kUpTileShapeK)) {
        return static_cast<std::uint32_t>(D::KernelConfig::kUpTileShapeK);
    }
    template <typename>
    static constexpr std::uint32_t test(...) {
        return 128u;
    }

  public:
    static constexpr std::uint32_t value = test<Dims>(0);
};

// Number of physical slots in the up-proj weight-TMA pipeline.
template <typename Dims>
struct NumUpWeightSlots {
  private:
    template <typename D>
    static constexpr auto test(int) -> decltype(static_cast<std::uint32_t>(
        D::KernelConfig::kNumUpWeightSlots)) {
        return static_cast<std::uint32_t>(D::KernelConfig::kNumUpWeightSlots);
    }
    template <typename D>
    static constexpr std::uint32_t test(...) {
        // A 128 KiB ring of up tiles, M x K fp8 bytes each.
        constexpr std::uint32_t kSlotBytes =
            (2u * D::kIntermediateSize / CtasPerExpert<D>::value) *
            UpTileShapeK<D>::value;
        return (128u * 1024u) / kSlotBytes;
    }

  public:
    static constexpr std::uint32_t value = test<Dims>(0);
};

// Latent down-projection weight-TMA pipeline depth
// (`kNumLatentDownWeightSlots`).
template <typename Dims>
struct NumLatentDownWeightSlots {
  private:
    template <typename D>
    static constexpr auto test(int) -> decltype(static_cast<std::uint32_t>(
        D::KernelConfig::kNumLatentDownWeightSlots)) {
        return static_cast<std::uint32_t>(
            D::KernelConfig::kNumLatentDownWeightSlots);
    }
    template <typename D>
    static constexpr std::uint32_t test(...) {
        return (128u * 1024u) / (D::kQuantBlockM * D::kQuantBlockK);
    }

  public:
    static constexpr std::uint32_t value = test<Dims>(0);
};

// TMEM accumulator ring depth (`kNumAccSlots`).
template <typename Dims>
struct NumAccSlots {
  private:
    template <typename D>
    static constexpr auto test(int)
        -> decltype(static_cast<std::uint32_t>(D::KernelConfig::kNumAccSlots)) {
        return static_cast<std::uint32_t>(D::KernelConfig::kNumAccSlots);
    }
    template <typename D>
    static constexpr std::uint32_t test(...) {
        constexpr std::uint32_t kUp = NumUpWeightSlots<D>::value;
        constexpr std::uint32_t kDown = D::kIntermediateSize / 128u;
        return (kUp > kDown) ? kUp : kDown;
    }

  public:
    static constexpr std::uint32_t value = test<Dims>(0);
};

// Router Linear instruction family (`kRouterMma`), UMMA by default.
template <typename Dims>
struct RouterMmaKind {
  private:
    template <typename D>
    static constexpr auto test(int)
        -> decltype(static_cast<MmaKind>(D::KernelConfig::kRouterMma)) {
        return static_cast<MmaKind>(D::KernelConfig::kRouterMma);
    }
    template <typename>
    static constexpr MmaKind test(...) {
        return MmaKind::kUmma;
    }

  public:
    static constexpr MmaKind value = test<Dims>(0);
};

// Latent down-projection instruction family (`kLatentDownMma`).
template <typename Dims>
struct LatentDownMmaKind {
  private:
    template <typename D>
    static constexpr auto test(int)
        -> decltype(static_cast<MmaKind>(D::KernelConfig::kLatentDownMma)) {
        return static_cast<MmaKind>(D::KernelConfig::kLatentDownMma);
    }
    template <typename>
    static constexpr MmaKind test(...) {
        return MmaKind::kUmma;
    }

  public:
    static constexpr MmaKind value = test<Dims>(0);
};

// Latent down-projection CTA count (`kNumCtasForLatentDown`).
template <typename Dims>
struct NumCtasForLatentDown {
  private:
    template <typename D>
    static constexpr auto test(int) -> decltype(static_cast<std::uint32_t>(
        D::KernelConfig::kNumCtasForLatentDown)) {
        return static_cast<std::uint32_t>(
            D::KernelConfig::kNumCtasForLatentDown);
    }
    template <typename D>
    static constexpr std::uint32_t test(...) {
        return D::kLatentSize / 128u;
    }

  public:
    static constexpr std::uint32_t value = test<Dims>(0);
};

// Which side quantizes the latent projection's output
// (`kLatentQuantizeOutput`).
template <typename Dims>
struct LatentQuantizeOutput {
  private:
    template <typename D>
    static constexpr auto test(int)
        -> decltype(static_cast<bool>(D::KernelConfig::kLatentQuantizeOutput)) {
        return static_cast<bool>(D::KernelConfig::kLatentQuantizeOutput);
    }
    template <typename>
    static constexpr bool test(...) {
        return false;
    }

  public:
    static constexpr bool value = test<Dims>(0);
};

// The shared expert's CTA count (`kNumCtasForShared`).
template <typename Dims>
struct NumCtasForShared {
  private:
    template <typename D>
    static constexpr auto test(int) -> decltype(static_cast<std::uint32_t>(
        D::KernelConfig::kNumCtasForShared)) {
        return static_cast<std::uint32_t>(D::KernelConfig::kNumCtasForShared);
    }
    template <typename D>
    static constexpr std::uint32_t test(...) {
        return D::kHiddenSize / D::kIntermediateSize;
    }

  public:
    static constexpr std::uint32_t value = test<Dims>(0);
};

// Whether the shared up splits its reduction over a cluster group
// (`kSharedUseSplitK`).
template <typename Dims>
struct SharedUseSplitK {
  private:
    template <typename D>
    static constexpr auto test(int)
        -> decltype(static_cast<bool>(D::KernelConfig::kSharedUseSplitK)) {
        return static_cast<bool>(D::KernelConfig::kSharedUseSplitK);
    }
    template <typename D>
    static constexpr bool test(...) {
        return true;
    }

  public:
    static constexpr bool value = test<Dims>(0);
};

// Whether the latent projection splits its reduction over a cluster group
// (`kLatentDownUseSplitK`).
template <typename Dims>
struct LatentDownUseSplitK {
  private:
    template <typename D>
    static constexpr auto test(int)
        -> decltype(static_cast<bool>(D::KernelConfig::kLatentDownUseSplitK)) {
        return static_cast<bool>(D::KernelConfig::kLatentDownUseSplitK);
    }
    template <typename D>
    static constexpr bool test(...) {
        return false;
    }

  public:
    static constexpr bool value = test<Dims>(0);
};

// Physical slot counts of the shared expert's two weight rings
// (`kNumSharedUpWeightSlots` / `kNumSharedDownWeightSlots`).
template <typename Dims>
struct NumSharedUpWeightSlots {
  private:
    template <typename D>
    static constexpr auto test(int) -> decltype(static_cast<std::uint32_t>(
        D::KernelConfig::kNumSharedUpWeightSlots)) {
        return static_cast<std::uint32_t>(
            D::KernelConfig::kNumSharedUpWeightSlots);
    }
    template <typename>
    static constexpr std::uint32_t test(...) {
        return 2u;
    }

  public:
    static constexpr std::uint32_t value = test<Dims>(0);
};

template <typename Dims>
struct NumSharedDownWeightSlots {
  private:
    template <typename D>
    static constexpr auto test(int) -> decltype(static_cast<std::uint32_t>(
        D::KernelConfig::kNumSharedDownWeightSlots)) {
        return static_cast<std::uint32_t>(
            D::KernelConfig::kNumSharedDownWeightSlots);
    }
    template <typename>
    static constexpr std::uint32_t test(...) {
        return 2u;
    }

  public:
    static constexpr std::uint32_t value = test<Dims>(0);
};

// Combine readiness granularity (`kCombineWaitPerToken`).
template <typename Dims>
struct CombineWaitPerToken {
  private:
    template <typename D>
    static constexpr auto test(int)
        -> decltype(static_cast<bool>(D::KernelConfig::kCombineWaitPerToken)) {
        return static_cast<bool>(D::KernelConfig::kCombineWaitPerToken);
    }
    template <typename>
    static constexpr bool test(...) {
        return false;
    }

  public:
    static constexpr bool value = test<Dims>(0);
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

// Warp-wide max reduction; the result is returned on all lanes.
__device__ static inline float warp_reduce_max_float(float value) {
#if defined(__CUDA_ARCH__) && defined(__CUDA_ARCH_FEAT_SM100_ALL)
    float result;
    asm("redux.sync.max.f32 %0, %1, 0xffffffff;" : "=f"(result) : "f"(value));
    return result;
#else
    for (int i = 16; i >= 1; i /= 2) {
        value = fmaxf(__shfl_xor_sync(0xffffffff, value, i, 32), value);
    }
    return value;
#endif
}

// Upper bound on the launch grid.
constexpr std::uint32_t kSpecMaxGrid = 256u;

}  // namespace halops::moe
