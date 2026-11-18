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

template <typename Quant>
struct QuantScale;

template <std::uint32_t BlockSize>
struct QuantScale<QuantSpec<QuantMethod::kMXFP8, BlockSize>> {
    using Quant = QuantSpec<QuantMethod::kMXFP8, BlockSize>;
    using ScaleType = typename Quant::ScaleType;
    // The scale a padding token carries.
    static constexpr ScaleType kPadding = 127u;

    // ceil(log2(amax / 448)) + 127, clamped to the UE8M0 exponent range.
    __device__ static __forceinline__ ScaleType encode(float amax) {
        constexpr float kFp8MaxInv = 1.0f / 448.0f;
        // A max near FLT_MIN would invert to +inf and NaN the group.
        const std::uint32_t scaled_bits =
            __float_as_uint(fmaxf(amax, 1e-10f) * kFp8MaxInv);
        const std::uint32_t exp_field = (scaled_bits >> 23) & 0xFFu;
        const std::uint32_t round_up = (scaled_bits & 0x7FFFFFu) != 0u;
        const std::uint32_t biased = exp_field + round_up;
        return static_cast<ScaleType>(biased < 254u ? biased : 254u);
    }

    // The inverse scale that exponent applies to the data.
    __device__ static __forceinline__ float decode(ScaleType scale) {
        return __uint_as_float((254u - static_cast<std::uint32_t>(scale))
                               << 23);
    }
};

template <std::uint32_t BlockSize>
struct QuantScale<QuantSpec<QuantMethod::kBlockwiseFP8, BlockSize>> {
    using Quant = QuantSpec<QuantMethod::kBlockwiseFP8, BlockSize>;
    using ScaleType = typename Quant::ScaleType;
    // The scale a padding token carries.
    static constexpr ScaleType kPadding = 1.0f;

    // The block scale one group's amax maps to.
    __device__ static __forceinline__ ScaleType encode(float amax) {
        constexpr float kFp8Max = 448.0f;
        // A max near FLT_MIN would overflow the inverse scale.
        amax = fmaxf(amax, 1e-10f);
        return amax * (1.0f / kFp8Max);
    }

    // The inverse scale that block scale applies to the data.
    __device__ static __forceinline__ float decode(ScaleType scale) {
        return 1.0f / scale;
    }
};

// Warp-wide max reduction; the result is returned on all lanes.
__device__ static inline float warp_reduce_max_float(float value) {
#if defined(__CUDA_ARCH__) && (defined(__CUDA_ARCH_FEAT_SM100_ALL) ||     \
                               (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && \
                                __CUDA_ARCH_FAMILY_SPECIFIC__ == 1000))
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

// A full warp quantizes N values, grouped by QuantSpec::kBlockSize.
// Each lane owns values[i] at column i * 32 + lane.
template <typename QuantSpec, std::uint32_t N>
struct RegisterQuantizer;

template <std::uint32_t BlockSize, std::uint32_t N>
struct RegisterQuantizer<QuantSpec<QuantMethod::kBlockwiseFP8, BlockSize>, N> {
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
            const float block_max = warp_reduce_max_float(local_max);
            quantized_scale[group] = QuantScale<Quant>::encode(block_max);
            const float inv_scale =
                QuantScale<Quant>::decode(quantized_scale[group]);
#pragma unroll
            for (std::uint32_t i = 0; i < kGroupValuesPerThread; ++i) {
                quantized_values[begin + i] =
                    static_cast<ElementType>(values[begin + i] * inv_scale);
            }
        }
    }
};

template <std::uint32_t BlockSize, std::uint32_t N>
struct RegisterQuantizer<QuantSpec<QuantMethod::kMXFP8, BlockSize>, N> {
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
            const float amax = warp_reduce_max_float(fabsf(values[i]));
            quantized_scale[i] = QuantScale<Quant>::encode(amax);
            const float inv_scale =
                QuantScale<Quant>::decode(quantized_scale[i]);
            quantized_values[i] =
                static_cast<ElementType>(values[i] * inv_scale);
        }
    }
};

// One staged activation tile: the window it covers, and the quantize that
// turns the staged bf16 into the slot's fp8 and scale entries.
template <typename Quant, std::size_t NumRows, std::size_t QuantBlock,
          std::size_t SwizzleBoxK, std::size_t BlocksPerStage>
struct SmemTileQuantizer {
    using Element = DataType::ElementA;
    using ElementQA = DataType::ElementQA;
    using ScaleType = typename Quant::ScaleType;
    // The staged fp8 tile: [slot][16-B chunk][row][16 bytes].
    using QuantizedTile =
        ElementQA (*)[QuantBlock * BlocksPerStage / 16u][NumRows + 1u][16u];

    std::uint32_t staging_slot_base;
    std::uint32_t block_base;
    std::uint32_t scale_row_base;
    std::uint32_t row_start;
    std::uint32_t actual_num_rows;

    __host__ __device__ SmemTileQuantizer(std::uint32_t staging_slot_base_in,
                                          std::uint32_t block_base_in,
                                          std::uint32_t scale_row_base_in,
                                          std::uint32_t row_start_in,
                                          std::uint32_t actual_num_rows_in) {
        staging_slot_base = staging_slot_base_in;
        block_base = block_base_in;
        scale_row_base = scale_row_base_in;
        row_start = row_start_in;
        actual_num_rows = actual_num_rows_in;
    }

    // The lanes that record a scale: one per 32-K sub-group for MXFP8,
    // lane 0 for a single block scale.
    __device__ static bool is_leader_thread(std::uint32_t lane) {
        return Quant::kMethod == QuantMethod::kMXFP8 ? lane % 8u == 0u
                                                     : lane == 0u;
    }

    // One group's amax over this lane's 4 values: 32 K elements across 8
    // lanes for MXFP8, the whole 128-K block across the warp for blockwise.
    __device__ static float group_amax(float4 v) {
        float amax =
            fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
        if constexpr (Quant::kMethod == QuantMethod::kMXFP8) {
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1, 8));
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2, 8));
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 4, 8));
        } else {
            amax = warp_reduce_max_float(amax);
        }
        return amax;
    }

    // One (row, block) unit: the warp's 32 lanes each hold 4 bf16 values.
    __device__ void quantize(std::uint32_t row, std::uint32_t blk,
                             const Element* __restrict__ data,
                             QuantizedTile quantized_data,
                             ScaleType* __restrict__ quantized_scale,
                             std::uint32_t k_base) const {
        constexpr std::uint32_t kChunksPerBlock = QuantBlock / 16u;
        static_assert(
            Quant::kMethod != QuantMethod::kMXFP8 || QuantBlock == 128u,
            "one 128-K block is four 32-K sub-blocks");
        static_assert(Quant::kMethod == QuantMethod::kMXFP8 || NumRows <= 16u,
                      "FP8 quantize supports BS<=16");
        const std::uint32_t lane_idx = WarpRole::get_lane_idx();
        const std::uint32_t col = lane_idx * 4u;
        const std::uint32_t global_row = row_start + row;
        const std::uint32_t slot = staging_slot_base + blk / BlocksPerStage;
        const std::uint32_t chunk_at = (blk % BlocksPerStage) * kChunksPerBlock;
        const std::uint32_t k = (block_base + blk) * QuantBlock + col;
        // Padding rows read the first valid row and are zeroed on store.
        const std::uint32_t src_row = global_row < actual_num_rows ? row : 0u;
        // The box's 16-B chunk index is XOR-ed with the row.
        const std::uint32_t rel = k - k_base;
        const std::uint32_t chunk = (rel % SwizzleBoxK) / 8u;
        const Element* const bf16_4 =
            data + (rel / SwizzleBoxK) * (NumRows * SwizzleBoxK) +
            src_row * SwizzleBoxK + (chunk ^ (src_row % 8u)) * 8u + rel % 8u;

        // The fp8 bytes this lane owns, and the scale entry it may record.
        ElementQA* const dest =
            &quantized_data[slot][chunk_at + col / 16u][row][col % 16u];
        ScaleType* scale;
        if constexpr (Quant::kMethod == QuantMethod::kMXFP8) {
            // Each slot stages BlocksPerStage quantization blocks.
            constexpr std::uint32_t kSfAtomBytes =
                128u * (QuantBlock * BlocksPerStage / 32u);
            scale = quantized_scale + slot * kSfAtomBytes +
                    sf_atom_offset(row,
                                   (blk % BlocksPerStage) * (QuantBlock / 32u) +
                                       lane_idx / 8u);
        } else {
            scale =
                quantized_scale + (scale_row_base + blk) * NumRows + global_row;
        }
        if (global_row >= actual_num_rows) {
            // Zero padded rows before UMMA reads them.
            *reinterpret_cast<std::uint32_t*>(dest) = 0u;
            if (is_leader_thread(lane_idx)) {
                *scale = QuantScale<Quant>::kPadding;
            }
            return;
        }

        const float4 v = load_quantize_pair(bf16_4);
        const ScaleType group_scale = QuantScale<Quant>::encode(group_amax(v));
        const float inv_scale = QuantScale<Quant>::decode(group_scale);

        const float4 q = make_float4(v.x * inv_scale, v.y * inv_scale,
                                     v.z * inv_scale, v.w * inv_scale);
        const __nv_fp8x2_storage_t lo = __nv_cvt_float2_to_fp8x2(
            make_float2(q.x, q.y), __NV_SATFINITE, __NV_E4M3);
        const __nv_fp8x2_storage_t hi = __nv_cvt_float2_to_fp8x2(
            make_float2(q.z, q.w), __NV_SATFINITE, __NV_E4M3);
        *reinterpret_cast<std::uint32_t*>(dest) =
            static_cast<std::uint32_t>(lo) |
            (static_cast<std::uint32_t>(hi) << 16);
        if (is_leader_thread(lane_idx)) {
            *scale = group_scale;
        }
    }

  private:
    // Map all NaN bit patterns to zero.
    __device__ static __forceinline__ __nv_bfloat162
    mask_nans_to_zero(__nv_bfloat162 xs) {
        return cutlass::platform::bit_cast<__nv_bfloat162>(
            cutlass::platform::bit_cast<uint32_t>(xs) & __heq2_mask(xs, xs));
    }

    // Load four BF16 values as FP32 and map NaNs to zero.
    __device__ static __forceinline__ float4
    load_quantize_pair(const Element* __restrict__ bf16_4) {
        const __nv_bfloat162 bf_01 =
            mask_nans_to_zero(*reinterpret_cast<const __nv_bfloat162*>(bf16_4));
        const __nv_bfloat162 bf_23 = mask_nans_to_zero(
            *reinterpret_cast<const __nv_bfloat162*>(bf16_4 + 2));
        const float2 f01 = __bfloat1622float2(bf_01);
        const float2 f23 = __bfloat1622float2(bf_23);
        return make_float4(f01.x, f01.y, f23.x, f23.y);
    }
};

}  // namespace halops::moe
