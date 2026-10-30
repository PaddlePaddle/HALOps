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

#include <cstddef>
#include <cstdint>

#include "kernel/moe/mma_core.cuh"
#include "kernel/moe/moe_configs.cuh"
#include "kernel/moe/moe_cta_comm.cuh"
#include "kernel/moe/moe_shared_storage.cuh"
#include "kernel/moe/moe_traits.cuh"

namespace halops::moe {

// The router's per-(gate, token) row: bf16 logits or fp32 scores.
template <typename Dims>
struct RouterLogits {
    using RouterGemm = RouterGemmTraits<Dims>;

    static constexpr std::uint32_t kNumGates = RouterGemm::kNumRouterGates;
    static constexpr bool kStoreScores = kNumGates > 1u;

    using ResultType = std::conditional_t<kStoreScores, float, __nv_bfloat16>;
    static constexpr std::uint32_t kNumRows = RouterGemm::kGemmN;
    static constexpr std::uint32_t kNumCols = RouterGemm::kGemmM;

    alignas(16) ResultType data[kNumGates * kNumRows * kNumCols];

    __device__ __forceinline__ ResultType* row(std::uint32_t gate,
                                               std::uint32_t row_idx) {
        return data + (gate * kNumRows + row_idx) * kNumCols;
    }
    __device__ __forceinline__ const ResultType* row(
        std::uint32_t gate, std::uint32_t row_idx) const {
        return data + (gate * kNumRows + row_idx) * kNumCols;
    }
};

// The latent down-projection's output, as the up phase consumes it.
template <typename Dims, bool IsQuantized>
struct LatentDownOutput;

template <typename Dims>
struct LatentDownOutput<Dims, false> {
    using LatentDownGemm = LatentDownGemmTraits<Dims>;
    using Element = DataType::ElementA;

    static constexpr std::uint32_t kNumRows = LatentDownGemm::kGemmN;
    static constexpr std::uint32_t kNumCols = LatentDownGemm::kGemmM;

    alignas(16) Element data[kNumRows][kNumCols];

    __device__ __forceinline__ void store(std::uint32_t tok, std::uint32_t col,
                                          Element value) {
        data[tok][col] = value;
    }

    __host__ __device__ __forceinline__ Element* data_ptr() {
        return &data[0][0];
    }
};

template <typename Dims>
struct LatentDownOutput<Dims, true> {
    using LatentDownGemm = LatentDownGemmTraits<Dims>;
    using Element = DataType::ElementQA;
    using ElementScale = DataType::ElementScale;

    static constexpr std::uint32_t kNumRows = LatentDownGemm::kGemmN;
    static constexpr std::uint32_t kNumCols = LatentDownGemm::kGemmM;

    static constexpr std::uint32_t kLboChunk =
        MoEKernelTraits<Dims>::kLboChunkK;
    static constexpr std::uint32_t kNumChunks = kNumCols / kLboChunk;
    static constexpr std::uint32_t kNumScaleBlocks =
        kNumCols / Dims::kQuantBlockM;

    alignas(16) Element data[kNumChunks][kNumRows + 1u][kLboChunk];
    alignas(16) ElementScale scale[kNumScaleBlocks][kNumRows];

    static constexpr std::uint32_t kDataBytes = sizeof(data);
    static constexpr std::uint32_t kScaleBytes = sizeof(scale);

    __device__ __forceinline__ void store_cell(std::uint32_t chunk,
                                               std::uint32_t row,
                                               uint4 const& bytes) {
        *reinterpret_cast<uint4*>(&data[chunk][row][0]) = bytes;
    }

    __device__ __forceinline__ void clear_pad_row(std::uint32_t chunk) {
        *reinterpret_cast<uint4*>(&data[chunk][kNumRows][0]) =
            make_uint4(0u, 0u, 0u, 0u);
    }

    __device__ __forceinline__ ElementScale* scale_row(std::uint32_t block) {
        return scale[block];
    }

    __host__ __device__ __forceinline__ Element* data_ptr() {
        return &data[0][0][0];
    }
    __host__ __device__ __forceinline__ ElementScale* scale_ptr() {
        return &scale[0][0];
    }
};

// The shared expert's up output, in the down's staging layout.
template <typename Dims>
struct SharedUpOutput {
    using ElementType = DataType::ElementQA;
    using ScaleType = DataType::ElementScale;
    using SharedUpGemm = SharedUpGemmTraits<Dims>;

    static constexpr std::uint32_t kQuantBlock = SharedUpGemm::kActQuantBlock;

    static constexpr std::uint32_t kNumRows = SharedUpGemm::kGemmN;
    static constexpr std::uint32_t kNumCols = SharedUpGemm::kGemmM / 2u;

    static constexpr std::uint32_t kLboChunk = kFp8ActChunkK;
    static constexpr std::uint32_t kNumChunks = kNumCols / kLboChunk;
    static constexpr std::uint32_t kNumScaleCols = kNumCols / kQuantBlock;

    alignas(16) ElementType data[kNumChunks][kNumRows + 1u][kLboChunk];
    ScaleType scale[kNumScaleCols][kNumRows];

    __device__ __forceinline__ ElementType* cell(std::uint32_t chunk,
                                                 std::uint32_t row) {
        return &data[chunk][row][0];
    }

    __device__ __forceinline__ void clear_pad_row(std::uint32_t chunk) {
        *reinterpret_cast<uint4*>(&data[chunk][Dims::kNumTokens][0]) =
            make_uint4(0u, 0u, 0u, 0u);
    }

    __device__ __forceinline__ ScaleType* scale_row(std::uint32_t block) {
        return scale[block];
    }

    __host__ __device__ __forceinline__ ElementType* data_ptr() {
        return &data[0][0][0];
    }
    __host__ __device__ __forceinline__ ScaleType* scale_ptr() {
        return &scale[0][0];
    }
};

// The up projection's output: the gated activation in expert-sorted rows.
template <typename Dims>
struct RoutedUpOutput {
    using ElementType = DataType::ElementQA;
    using ScaleType = DataType::ElementScale;
    using UpGemm = UpGemmTraits<Dims>;

    static constexpr std::uint32_t kNumRows = UpGemm::kGemmN * Dims::kMaxTopk;
    static constexpr std::uint32_t kNumCols = UpGemm::kGemmM / 2u;

    static constexpr std::uint32_t kQuantBlock = UpGemm::kTileShapeM / 2u;
    static constexpr std::uint32_t kNumScaleCols = kNumCols / kQuantBlock;

    alignas(16) ElementType data[(kNumRows + 8u) * kNumCols];
    ScaleType scale[kNumRows * kNumScaleCols];

    __device__ __forceinline__ ElementType* row(std::uint32_t row_idx) {
        return data + row_idx * kNumCols;
    }

    __device__ __forceinline__ const ElementType* row(
        std::uint32_t row_idx) const {
        return data + row_idx * kNumCols;
    }

    __device__ __forceinline__ ScaleType* scale_row(std::uint32_t row_idx) {
        return scale + row_idx * kNumScaleCols;
    }

    __device__ __forceinline__ const ScaleType* scale_row(
        std::uint32_t row_idx) const {
        return scale + row_idx * kNumScaleCols;
    }
};

// The down projection's output: one bf16 slot per routed (token, expert) pair.
template <typename Dims>
struct RoutedDownOutput {
    using ElementType = DataType::ElementOut;
    using DownGemm = DownGemmTraits<Dims>;

    static constexpr std::uint32_t kNumRows = DownGemm::kGemmN * Dims::kMaxTopk;
    static constexpr std::uint32_t kNumCols = DownGemm::kGemmM;

    alignas(16) ElementType data[kNumRows * kNumCols];

    __device__ __forceinline__ ElementType* row(std::uint32_t row_idx) {
        return data + row_idx * kNumCols;
    }

    __device__ __forceinline__ const ElementType* row(
        std::uint32_t row_idx) const {
        return data + row_idx * kNumCols;
    }
};

template <typename Dims>
struct MoEWorkspace {
    using DownGemm = DownGemmTraits<Dims>;
    using LatentDownGemm = LatentDownGemmTraits<Dims>;
    using RoutedUpOut = RoutedUpOutput<Dims>;
    using RoutedDownOut = RoutedDownOutput<Dims>;
    using SharedUpOut = SharedUpOutput<Dims>;

    static constexpr uint32_t kMaxTopk = Dims::kMaxTopk;

    static constexpr uint32_t kMaxRoutedRowsPadding = RoutedUpOut::kNumRows;
    static constexpr uint32_t kMaxRoutedRows = RoutedDownOut::kNumRows;

    static constexpr uint32_t kSpecMaxGrid = moe::kSpecMaxGrid;

    SharedUpOut shared_up_out;

    using LatentOut =
        LatentDownOutput<Dims, MoEKernelTraits<Dims>::kLatentQuantizeOutput>;
    LatentOut latent_down_out;

    RouterLogits<Dims> router_logits;
    static constexpr bool kRouterStoreScores = RouterLogits<Dims>::kStoreScores;

    RoutedUpOut routed_up_out;

    // Byte offset of the fp8 values, for the host TMA wrapper.
    static constexpr size_t kUpOutFp8Offset =
        offsetof(MoEWorkspace<Dims>, routed_up_out.data);

    RoutedDownOut routed_down_out;

    MoERoutingTables<Dims> router_tables;

    static constexpr uint32_t kCtasPerExpert =
        Dims::KernelConfig::kCtasPerExpert;

    using CtaComm = MoECtaComm<kSpecMaxGrid, kMaxRoutedRows,
                               RoutedUpOut::kNumScaleCols, kCtasPerExpert>;
    static constexpr uint32_t kUpOutReadyFull = CtaComm::kUpOutReadyFull;

    CtaComm cta_comm;
};

}  // namespace halops::moe
