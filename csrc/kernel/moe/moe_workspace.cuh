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

// Latent output scales in routed-up staging layout: FP32 blocks or MXFP8
// SfAtoms.
template <typename Dims, bool IsMx>
struct LatentScaleImage;

template <typename Dims>
struct LatentScaleImage<Dims, false> {
    using LatentDownGemm = LatentDownGemmTraits<Dims>;
    using Element = float;

    static constexpr std::uint32_t kNumBlocks = LatentDownGemm::kGemmM / 128u;
    static constexpr std::uint32_t kNumRows = LatentDownGemm::kGemmN;

    alignas(16) Element data[kNumBlocks][kNumRows];

    __device__ __forceinline__ Element* row(std::uint32_t block) {
        return data[block];
    }
};

template <typename Dims>
struct LatentScaleImage<Dims, true> {
    using UpGemm = RoutedUpGemmTraits<Dims>;
    using Element = std::uint8_t;

    // One 128-token SfAtom slab per up K tile holds that tile's 32-K groups.
    static constexpr std::uint32_t kNumSlots = UpGemm::kNumTilesK;
    static constexpr std::uint32_t kSfAtomBytes =
        128u * (UpGemm::kTileShapeK / 32u);

    alignas(16) Element data[kNumSlots][kSfAtomBytes];

    __device__ __forceinline__ Element* slab(std::uint32_t slot) {
        return data[slot];
    }
};

// Latent output data and scales in routed-up staging layout.
template <typename Dims>
struct LatentDownOutput {
    using LatentDownGemm = LatentDownGemmTraits<Dims>;
    using Element = DataType::ElementQA;
    using ScaleImage =
        LatentScaleImage<Dims, LatentDownGemm::kOutputUseBlockscaled>;

    static constexpr std::uint32_t kNumRows = LatentDownGemm::kGemmN;
    static constexpr std::uint32_t kNumCols = LatentDownGemm::kGemmM;

    static constexpr std::uint32_t kLboChunk = LboChunkK<Element>::value;
    static constexpr std::uint32_t kNumChunks = kNumCols / kLboChunk;

    alignas(16) Element data[kNumChunks][kNumRows + 1u][kLboChunk];
    alignas(16) ScaleImage scale;

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

    __device__ __forceinline__ auto* scale_row(std::uint32_t block) {
        return scale.row(block);
    }

    __device__ __forceinline__ auto* sf_slab(std::uint32_t slot) {
        return scale.slab(slot);
    }

    __host__ __device__ __forceinline__ Element* data_ptr() {
        return &data[0][0][0];
    }
    __host__ __device__ __forceinline__ void* scale_ptr() { return &scale; }
};

// The shared expert's up output, in the down's staging layout.
template <typename Dims>
struct SharedUpOutput {
    using ElementType = DataType::ElementQA;
    using ScaleType = float;
    using SharedUpGemm = SharedUpGemmTraits<Dims>;
    using SharedDownGemm = SharedDownGemmTraits<Dims>;

    static constexpr std::uint32_t kQuantBlock = SharedUpGemm::kTileShapeM / 2u;

    static constexpr std::uint32_t kNumRows = SharedUpGemm::kGemmN;
    static constexpr std::uint32_t kNumCols = SharedUpGemm::kGemmM / 2u;

    static constexpr std::uint32_t kLboChunk = LboChunkK<ElementType>::value;
    static constexpr std::uint32_t kNumChunks = kNumCols / kLboChunk;
    static constexpr std::uint32_t kNumScaleCols = kNumCols / kQuantBlock;

    // The union keeps workspace sizing identical across quantization recipes.
    static constexpr std::uint32_t kDownTilesK = SharedDownGemm::kNumTilesK;
    static constexpr std::uint32_t kSfAtomBytes =
        128u * (SharedDownGemm::kTileShapeK / 32u);

    alignas(16) ElementType data[kNumChunks][kNumRows + 1u][kLboChunk];
    union alignas(16) ScaleImage {
        ScaleType blockwise[kNumScaleCols][kNumRows];
        std::uint8_t mxfp8[kDownTilesK][kSfAtomBytes];
    } scale;

    __device__ __forceinline__ ElementType* cell(std::uint32_t chunk,
                                                 std::uint32_t row) {
        return &data[chunk][row][0];
    }

    __device__ __forceinline__ void clear_pad_row(std::uint32_t chunk) {
        *reinterpret_cast<uint4*>(&data[chunk][Dims::kNumTokens][0]) =
            make_uint4(0u, 0u, 0u, 0u);
    }

    __device__ __forceinline__ ScaleType* scale_row(std::uint32_t block) {
        return scale.blockwise[block];
    }

    // UE8M0 scale slab for one down-input K tile.
    __device__ __forceinline__ std::uint8_t* sf_slab(std::uint32_t tile) {
        return scale.mxfp8[tile];
    }

    __host__ __device__ __forceinline__ ElementType* data_ptr() {
        return &data[0][0][0];
    }
    __host__ __device__ __forceinline__ ScaleType* scale_ptr() {
        return &scale.blockwise[0][0];
    }
    __host__ __device__ __forceinline__ std::uint8_t* scale_bytes() {
        return &scale.mxfp8[0][0];
    }
};

template <typename Dims>
struct SharedDownOutput {
    using ElementType = DataType::ElementOut;

    alignas(16) ElementType data[Dims::kNumTokens * Dims::kHiddenSize];
};

// The up projection's output: the gated activation in expert-sorted rows.
template <typename Dims>
struct RoutedUpOutput {
    using ElementType = DataType::ElementQA;
    using ScaleType = float;
    using UpGemm = RoutedUpGemmTraits<Dims>;

    static constexpr std::uint32_t kNumRows = UpGemm::kGemmN * Dims::kMaxTopk;
    static constexpr std::uint32_t kNumCols = UpGemm::kGemmM / 2u;

    static constexpr std::uint32_t kQuantBlock = UpGemm::kTileShapeM / 2u;
    static constexpr std::uint32_t kNumScaleCols = kNumCols / kQuantBlock;
    // MX uses one UE8M0 byte per row and 32 features, repacked into SfAtoms
    // during down loading.
    static constexpr std::uint32_t kNumMxScaleCols = kNumCols / 32u;

    alignas(16) ElementType data[(kNumRows + 8u) * kNumCols];
    // The union keeps workspace sizing identical across quantization recipes.
    union alignas(16) ScaleImage {
        ScaleType blockwise[kNumRows * kNumScaleCols];
        std::uint8_t mxfp8[kNumRows * kNumMxScaleCols];
    } scale;

    __device__ __forceinline__ ElementType* row(std::uint32_t row_idx) {
        return data + row_idx * kNumCols;
    }

    __device__ __forceinline__ const ElementType* row(
        std::uint32_t row_idx) const {
        return data + row_idx * kNumCols;
    }

    __device__ __forceinline__ ScaleType* scale_row(std::uint32_t row_idx) {
        return scale.blockwise + row_idx * kNumScaleCols;
    }

    __device__ __forceinline__ const ScaleType* scale_row(
        std::uint32_t row_idx) const {
        return scale.blockwise + row_idx * kNumScaleCols;
    }

    __device__ __forceinline__ std::uint8_t* mx_scale_row(
        std::uint32_t row_idx) {
        return scale.mxfp8 + row_idx * kNumMxScaleCols;
    }
};

// The down projection's output: one bf16 slot per routed (token, expert) pair.
template <typename Dims>
struct RoutedDownOutput {
    using ElementType = DataType::ElementOut;
    using DownGemm = RoutedDownGemmTraits<Dims>;

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
    using DownGemm = RoutedDownGemmTraits<Dims>;
    using LatentDownGemm = LatentDownGemmTraits<Dims>;
    using RoutedUpOut = RoutedUpOutput<Dims>;
    using RoutedDownOut = RoutedDownOutput<Dims>;
    using SharedUpOut = SharedUpOutput<Dims>;

    static constexpr uint32_t kMaxTopk = Dims::kMaxTopk;

    static constexpr uint32_t kMaxRoutedRows = RoutedDownOut::kNumRows;

    static constexpr uint32_t kSpecMaxGrid = moe::kSpecMaxGrid;

    SharedUpOut shared_up_out;
    SharedDownOutput<Dims> shared_down_out;

    using LatentOut = LatentDownOutput<Dims>;
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
