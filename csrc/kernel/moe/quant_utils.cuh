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
#include <cutlass/array.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "kernel/moe/moe_configs.cuh"
#include "kernel/moe/moe_shared_storage.cuh"
#include "kernel/moe/moe_traits.cuh"

namespace halops::moe {

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

// Map all NaN bit patterns to zero.
__device__ static __forceinline__ __nv_bfloat162
mask_nans_to_zero(__nv_bfloat162 xs) {
    return cutlass::platform::bit_cast<__nv_bfloat162>(
        cutlass::platform::bit_cast<uint32_t>(xs) & __heq2_mask(xs, xs));
}

// Load four BF16 values as FP32 and map NaNs to zero.
__device__ __forceinline__ float4
load_quantize_pair(const DataType::ElementA* __restrict__ bf16_4) {
    const __nv_bfloat162 bf_01 =
        mask_nans_to_zero(*reinterpret_cast<const __nv_bfloat162*>(bf16_4));
    const __nv_bfloat162 bf_23 =
        mask_nans_to_zero(*reinterpret_cast<const __nv_bfloat162*>(bf16_4 + 2));
    const float2 f01 = __bfloat1622float2(bf_01);
    const float2 f23 = __bfloat1622float2(bf_23);
    return make_float4(f01.x, f01.y, f23.x, f23.y);
}

// Scale and store an FP8 pair; lane 0 records its scale. row is local to
// staging, while global_row determines source padding.
template <std::size_t Fp8NumChunks, std::size_t Fp8Rows, std::size_t Fp8KInner>
__device__ __forceinline__ void store_quantized_pair(
    float4 v, std::uint32_t row, std::uint32_t chunk_at,
    std::uint32_t global_row,
    DataType::ElementQA (&fp8_act)[Fp8NumChunks][Fp8Rows][Fp8KInner],
    std::uint32_t actual_num_rows,
    float* __restrict__ quantized_input_scale_for_step) {
    constexpr float kFp8Max = 448.f;
    constexpr float kFp8MaxInv = 1.0f / 448.f;
    const std::uint32_t lane = WarpRole::get_lane_idx();
    const std::uint32_t col = lane * 4u;
    const std::uint32_t kc = chunk_at + col / 16u;
    const std::uint32_t ki = col % 16u;
    if (global_row >= actual_num_rows) {
        // Zero padded rows before UMMA reads them.
        *reinterpret_cast<std::uint32_t*>(&fp8_act[kc][row][ki]) = 0u;
        if (lane == 0u) *quantized_input_scale_for_step = 1.0f;
        return;
    }
    const float local_max =
        fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
    float blk_max = warp_reduce_max_float(local_max);
    // Clamp tiny maxima to keep the inverse scale finite.
    blk_max = fmaxf(blk_max, 1e-10f);
    const float blk_inv_scale = kFp8Max / blk_max;

    fp8_act[kc][row][ki + 0] = (DataType::ElementQA)(v.x * blk_inv_scale);
    fp8_act[kc][row][ki + 1] = (DataType::ElementQA)(v.y * blk_inv_scale);
    fp8_act[kc][row][ki + 2] = (DataType::ElementQA)(v.z * blk_inv_scale);
    fp8_act[kc][row][ki + 3] = (DataType::ElementQA)(v.w * blk_inv_scale);

    if (lane == 0u) {
        *quantized_input_scale_for_step = blk_max * kFp8MaxInv;
    }
}

// Encode ceil(log2(amax / 448)) + 127 as UE8M0 and return its inverse scale.
__device__ __forceinline__ float mxfp8_ue8m0_scale(float amax,
                                                   std::uint32_t& biased) {
    constexpr float kFp8MaxInv = 1.0f / 448.0f;
    // A max near FLT_MIN would invert to +inf and NaN the group.
    const std::uint32_t scaled_bits =
        __float_as_uint(fmaxf(amax, 1e-10f) * kFp8MaxInv);
    const std::uint32_t exp_field = (scaled_bits >> 23) & 0xFFu;
    const std::uint32_t round_up = (scaled_bits & 0x7FFFFFu) != 0u;
    const std::uint32_t raw_biased = exp_field + round_up;
    biased = raw_biased < 254u ? raw_biased : 254u;
    return __uint_as_float((254u - biased) << 23);
}

// A full warp quantizes N values, grouped by QuantSpec::kBlockSize.
// Each lane owns values[i] at column i * 32 + lane.
template <typename QuantSpec, std::uint32_t N>
struct GroupWiseQuantizer;

template <std::uint32_t BlockSize, std::uint32_t N>
struct GroupWiseQuantizer<QuantSpec<QuantMethod::kBlockwiseFP8, BlockSize>, N> {
    using Quant = QuantSpec<QuantMethod::kBlockwiseFP8, BlockSize>;
    using ElementType = typename Quant::ElementType;
    using ScaleType = typename Quant::ScaleType;

    static_assert(Quant::kBlockSize > 0u && Quant::kBlockSize % 32u == 0u,
                  "each scale group must contain whole warps of values");
    static_assert(N % Quant::kBlockSize == 0u,
                  "N must contain whole scale groups");

    static constexpr std::uint32_t kValuesPerThread = N / 32u;
    static constexpr std::uint32_t kNumGroups = N / Quant::kBlockSize;

    // The warp's inputs, and one block scale per group.
    using Values = cutlass::Array<float, kValuesPerThread>;
    using ScaleArray = cutlass::Array<ScaleType, kNumGroups>;

    __device__ static __forceinline__ void quantize(
        const Values& values, ElementType (&quantized_values)[kValuesPerThread],
        ScaleArray& quantized_scale) {
        constexpr std::uint32_t kGroupValuesPerThread = Quant::kBlockSize / 32u;
#pragma unroll
        for (std::uint32_t group = 0; group < kNumGroups; ++group) {
            const std::uint32_t begin = group * kGroupValuesPerThread;
            float local_max = 0.0f;
#pragma unroll
            for (std::uint32_t i = 0; i < kGroupValuesPerThread; ++i) {
                local_max = fmaxf(local_max, fabsf(values[begin + i]));
            }
            // Nonnegative finite floats order like their bit patterns.
            float block_max = __uint_as_float(
                __reduce_max_sync(0xffffffffu, __float_as_uint(local_max)));
            // Tiny maxima would overflow the inverse scale.
            block_max = fmaxf(block_max, 1e-10f);
            constexpr float kFp8Max = 448.0f;
            const float inv_scale = kFp8Max / block_max;
#pragma unroll
            for (std::uint32_t i = 0; i < kGroupValuesPerThread; ++i) {
                quantized_values[begin + i] =
                    static_cast<ElementType>(values[begin + i] * inv_scale);
            }
            quantized_scale[group] = block_max * (1.0f / kFp8Max);
        }
    }
};

template <std::uint32_t BlockSize, std::uint32_t N>
struct GroupWiseQuantizer<QuantSpec<QuantMethod::kMXFP8, BlockSize>, N> {
    using Quant = QuantSpec<QuantMethod::kMXFP8, BlockSize>;
    using ElementType = typename Quant::ElementType;
    using ScaleType = typename Quant::ScaleType;

    static_assert(N % Quant::kBlockSize == 0u,
                  "N must contain whole scale groups");

    static constexpr std::uint32_t kValuesPerThread = N / 32u;
    static constexpr std::uint32_t kNumGroups = N / Quant::kBlockSize;

    // Use a plain scale array to preserve byte-addressable UE8M0 entries.
    using Values = cutlass::Array<float, kValuesPerThread>;
    using ScaleArray = ScaleType[kNumGroups];

    __device__ static __forceinline__ void quantize(
        const Values& values, ElementType (&quantized_values)[kValuesPerThread],
        ScaleArray& quantized_scale) {
#pragma unroll
        for (std::uint32_t i = 0; i < kNumGroups; ++i) {
            const float amax = __uint_as_float(__reduce_max_sync(
                0xffffffffu, __float_as_uint(fabsf(values[i]))));
            std::uint32_t biased;
            const float inv_scale = mxfp8_ue8m0_scale(amax, biased);
            quantized_values[i] =
                static_cast<ElementType>(values[i] * inv_scale);
            quantized_scale[i] = static_cast<ScaleType>(biased);
        }
    }
};

// Staging slot, source block and row window for a quantized tile. Rows beyond
// actual_num_rows are padding.
struct QuantizeTile {
    std::uint32_t staging_slot_base;
    std::uint32_t block_base;
    std::uint32_t row_start;
    std::uint32_t actual_num_rows;

    __host__ __device__ QuantizeTile(std::uint32_t staging_slot_base_in,
                                     std::uint32_t block_base_in,
                                     std::uint32_t row_start_in,
                                     std::uint32_t actual_num_rows_in) {
        staging_slot_base = staging_slot_base_in;
        block_base = block_base_in;
        row_start = row_start_in;
        actual_num_rows = actual_num_rows_in;
    }
};

// Quantize a NumRows x NumCols tile into per-slot SfAtoms.
template <bool Staged, std::size_t NumRows, std::size_t NumCols,
          std::size_t QuantBlock, std::size_t SwizzleBoxK,
          std::size_t BlocksPerStage>
__device__ inline void quantize_mxfp8_groupwise_1x32(
    const DataType::ElementA* __restrict__ data, std::uint32_t k_base,
    DataType::ElementQA (*__restrict__ quantized_data)[QuantBlock *
                                                       BlocksPerStage / 16u]
                                                      [NumRows + 1u][16u],
    std::uint8_t* __restrict__ quantized_scale, const QuantizeTile& tile,
    std::uint32_t warp_idx, std::uint32_t num_quant_warps) {
    static constexpr std::uint32_t NumQuantBlocks =
        Staged ? BlocksPerStage : (NumCols / QuantBlock);
    static_assert(NumCols % QuantBlock == 0u,
                  "the tile must span whole quantization blocks");
    static_assert(NumQuantBlocks % BlocksPerStage == 0u,
                  "the tile's blocks must fill whole staging slots");
    static_assert(QuantBlock == 128u,
                  "one 128-K block is four 32-K sub-blocks");
    constexpr std::uint32_t kChunksPerBlock = QuantBlock / 16u;
    constexpr std::uint32_t kSubBlocks = QuantBlock / 32u;
    const std::uint32_t lane = WarpRole::get_lane_idx();
    const std::uint32_t col = lane * 4u;
    const std::uint32_t sub = lane / 8u;
    const bool leader = (lane % 8u) == 0u;

#pragma unroll 1
    for (std::uint32_t i = warp_idx; i < NumRows * NumQuantBlocks;
         i += num_quant_warps) {
        const std::uint32_t row = i / NumQuantBlocks;
        const std::uint32_t global_row = tile.row_start + row;
        const std::uint32_t blk = i % NumQuantBlocks;
        const std::uint32_t slot =
            tile.staging_slot_base + blk / BlocksPerStage;
        const std::uint32_t chunk_at = (blk % BlocksPerStage) * kChunksPerBlock;
        const std::uint32_t k = (tile.block_base + blk) * QuantBlock + col;
        const std::uint32_t src_row =
            global_row < tile.actual_num_rows ? row : 0u;
        const DataType::ElementA* bf16_4;
        if constexpr (Staged) {
            const std::uint32_t rel = k - k_base;
            const std::uint32_t chunk = (rel % SwizzleBoxK) / 8u;
            bf16_4 = data + (rel / SwizzleBoxK) * (NumRows * SwizzleBoxK) +
                     src_row * SwizzleBoxK + (chunk ^ (src_row % 8u)) * 8u +
                     rel % 8u;
        } else {
            bf16_4 = data +
                     (global_row < tile.actual_num_rows ? global_row : 0u) *
                         NumCols +
                     k;
        }

        const std::uint32_t kc = chunk_at + col / 16u;
        const std::uint32_t ki = col % 16u;
        // Each slot stages BlocksPerStage quantization blocks.
        constexpr std::uint32_t kSfAtomBytes =
            128u * (QuantBlock * BlocksPerStage / 32u);
        std::uint8_t* const scale =
            quantized_scale + slot * kSfAtomBytes +
            sf_atom_offset(row, (blk % BlocksPerStage) * kSubBlocks + sub);
        if (global_row >= tile.actual_num_rows) {
            // Zero padded data and store neutral scale exponents.
            *reinterpret_cast<std::uint32_t*>(
                &quantized_data[slot][kc][row][ki]) = 0u;
            if (leader) *scale = 127u;
            continue;
        }

        const float4 v = load_quantize_pair(bf16_4);
        float local_max =
            fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
        local_max =
            fmaxf(local_max, __shfl_xor_sync(0xffffffffu, local_max, 1, 8));
        local_max =
            fmaxf(local_max, __shfl_xor_sync(0xffffffffu, local_max, 2, 8));
        local_max =
            fmaxf(local_max, __shfl_xor_sync(0xffffffffu, local_max, 4, 8));
        std::uint32_t biased;
        const float inv_scale = mxfp8_ue8m0_scale(local_max, biased);

        const float4 q = make_float4(v.x * inv_scale, v.y * inv_scale,
                                     v.z * inv_scale, v.w * inv_scale);
        const __nv_fp8x2_storage_t lo = __nv_cvt_float2_to_fp8x2(
            make_float2(q.x, q.y), __NV_SATFINITE, __NV_E4M3);
        const __nv_fp8x2_storage_t hi = __nv_cvt_float2_to_fp8x2(
            make_float2(q.z, q.w), __NV_SATFINITE, __NV_E4M3);
        *reinterpret_cast<std::uint32_t*>(&quantized_data[slot][kc][row][ki]) =
            static_cast<std::uint32_t>(lo) |
            (static_cast<std::uint32_t>(hi) << 16);
        if (leader) {
            *scale = static_cast<std::uint8_t>(biased);
        }
    }
}

// Quantize a NumRows x NumCols tile; ScaleCols is the scale-table row stride.
template <bool Staged, std::size_t NumRows, std::size_t NumCols,
          std::size_t ScaleCols, std::size_t QuantBlock,
          std::size_t SwizzleBoxK, std::size_t BlocksPerStage>
__device__ inline void quantize_fp8_groupwise_128(
    const DataType::ElementA* __restrict__ data, std::uint32_t k_base,
    DataType::ElementQA (*__restrict__ quantized_data)[QuantBlock *
                                                       BlocksPerStage / 16u]
                                                      [NumRows + 1u][16u],
    float* __restrict__ quantized_scale, const QuantizeTile& tile,
    std::uint32_t warp_idx, std::uint32_t num_quant_warps) {
    static constexpr std::uint32_t NumQuantBlocks =
        Staged ? BlocksPerStage : (NumCols / QuantBlock);
    static_assert(NumRows <= 16u, "FP8 quantize supports BS<=16");
    static_assert(NumCols % QuantBlock == 0u,
                  "the tile must span whole quantization blocks");
    static_assert(NumQuantBlocks % BlocksPerStage == 0u,
                  "the tile's blocks must fill whole staging slots");
    constexpr std::uint32_t kChunksPerBlock = QuantBlock / 16u;
    const std::uint32_t col = WarpRole::get_lane_idx() * 4u;

#pragma unroll 1
    for (std::uint32_t i = warp_idx; i < NumRows * NumQuantBlocks;
         i += num_quant_warps) {
        // The window's row, and the row it holds in the source tensor.
        const std::uint32_t row = i / NumQuantBlocks;
        const std::uint32_t global_row = tile.row_start + row;
        const std::uint32_t blk = i % NumQuantBlocks;
        const std::uint32_t slot =
            tile.staging_slot_base + blk / BlocksPerStage;
        const std::uint32_t chunk_at = (blk % BlocksPerStage) * kChunksPerBlock;
        const std::uint32_t sblk = tile.block_base + blk;
        const std::uint32_t k = (tile.block_base + blk) * QuantBlock + col;
        // Padding rows read the first valid row and are zeroed on store.
        const std::uint32_t src_row =
            global_row < tile.actual_num_rows ? row : 0u;
        const DataType::ElementA* bf16_4;
        if constexpr (Staged) {
            // The box's 16-B chunk index is XOR-ed with the row.
            const std::uint32_t rel = k - k_base;
            const std::uint32_t chunk = (rel % SwizzleBoxK) / 8u;
            bf16_4 = data + (rel / SwizzleBoxK) * (NumRows * SwizzleBoxK) +
                     src_row * SwizzleBoxK + (chunk ^ (src_row % 8u)) * 8u +
                     rel % 8u;
        } else {
            // Row-major input uses the global source-row index.
            bf16_4 = data +
                     (global_row < tile.actual_num_rows ? global_row : 0u) *
                         NumCols +
                     k;
        }
        store_quantized_pair(load_quantize_pair(bf16_4), row, chunk_at,
                             global_row, quantized_data[slot],
                             tile.actual_num_rows,
                             &quantized_scale[sblk * ScaleCols + global_row]);
    }
}

}  // namespace halops::moe
