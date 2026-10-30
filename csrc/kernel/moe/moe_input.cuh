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

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "kernel/moe/moe_configs.cuh"
#include "kernel/moe/moe_shared_storage.cuh"
#include "kernel/moe/moe_tma.cuh"
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

// Issue the full per-block BF16 input TMA load: one 64-k SW128 box per
// issue, box `kk` into `dest[kk]`.
template <std::size_t kRingRows, std::size_t NumSwizzleBoxesK,
          std::size_t SwizzleBoxK>
__device__ __forceinline__ void load_input_tile(
    CUtensorMap const& tma_input_desc, std::uint32_t tile,
    std::uint32_t token_start,
    DataType::ElementA (&dest)[NumSwizzleBoxesK][kRingRows][SwizzleBoxK],
    std::uint64_t* bar_smem_ptr) {
    // The box is one 8-row swizzle atom, so the ring's rows are whole atoms.
    constexpr std::uint32_t kRowGroups = kRingRows / 8u;
    static_assert(kRingRows % 8u == 0u, "the ring is whole swizzle atoms");
#pragma unroll
    for (std::uint32_t kk = 0u; kk < NumSwizzleBoxesK; ++kk) {
#pragma unroll
        for (std::uint32_t g = 0u; g < kRowGroups; ++g) {
            const std::uint32_t k_start =
                tile * (NumSwizzleBoxesK * SwizzleBoxK) + kk * SwizzleBoxK;
            tma_load_bf16_input_tile(tma_input_desc, k_start,
                                     token_start + 8u * g,
                                     /*dest_smem_ptr=*/&dest[kk][8u * g][0],
                                     /*bar_smem_ptr=*/bar_smem_ptr);
        }
    }
}

// Reinterprets the bit-pattern of `x` to type `To`
template <typename To, typename From>
__device__ static __forceinline__ To bit_cast(From x) {
    static_assert(sizeof(To) == sizeof(From), "Types of different size");
    To y;
    memcpy(&y, &x, sizeof(From));
    return y;
}

// Sets NaNs (positive or negative, any payload) to 0.0 (bit-pattern all
// 0). Values other than NaN remain unchanged.
__device__ static __forceinline__ __nv_bfloat162
mask_nans_to_zero(__nv_bfloat162 xs) {
    return bit_cast<__nv_bfloat162>(bit_cast<uint32_t>(xs) &
                                    __heq2_mask(xs, xs));
}

// Per-128-K-block bf16 → fp8 quantization for one token.
__device__ __forceinline__ float4
moe_load_quantize_pair(const DataType::ElementA* __restrict__ bf16_4) {
    const __nv_bfloat162 bf_01 =
        mask_nans_to_zero(*reinterpret_cast<const __nv_bfloat162*>(bf16_4));
    const __nv_bfloat162 bf_23 =
        mask_nans_to_zero(*reinterpret_cast<const __nv_bfloat162*>(bf16_4 + 2));
    const float2 f01 = __bfloat1622float2(bf_01);
    const float2 f23 = __bfloat1622float2(bf_23);
    return make_float4(f01.x, f01.y, f23.x, f23.y);
}

// `row` is the staging row (the CTA's own token window), `chunk_at` the first
// 16-B chunk of this pair inside the slot, and `tok` the launch-wide token it
// belongs to (which decides whether the row is padding).
template <std::size_t Fp8NumChunks, std::size_t Fp8Tok, std::size_t Fp8KInner>
__device__ __forceinline__ void moe_store_quantized_pair(
    float4 v, std::uint32_t row, std::uint32_t chunk_at, std::uint32_t tok,
    DataType::ElementQA (&fp8_act)[Fp8NumChunks][Fp8Tok][Fp8KInner],
    std::uint32_t batch_size,
    float* __restrict__ quantized_input_scale_for_step) {
    constexpr float kFp8Max = 448.f;
    constexpr float kFp8MaxInv = 1.0f / 448.f;
    const std::uint32_t lane = WarpRole::get_lane_idx();
    const std::uint32_t col = lane * 4u;
    const std::uint32_t kc = chunk_at + col / 16u;
    const std::uint32_t ki = col % 16u;
    if (tok >= batch_size) {
        // One word per lane zeroes the padded slot's 128 B row, so the UMMA's
        // B operand sees no stray fp8 NaN bit patterns.
        *reinterpret_cast<std::uint32_t*>(&fp8_act[kc][row][ki]) = 0u;
        if (lane == 0u) *quantized_input_scale_for_step = 1.0f;
        return;
    }
    const float local_max =
        fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
    float blk_max = warp_reduce_max_float(local_max);
    // Tiny maxima would overflow the inverse scale; the eps matches vLLM's
    // group-quant clamp.
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

template <std::size_t NumRows, std::size_t ChunksPerSlot,
          std::size_t Fp8NumChunks, std::size_t Fp8Tok, std::size_t Fp8KInner>
__device__ __forceinline__ void moe_streaming_quantize_k128(
    const DataType::ElementA* __restrict__ bf16_4, std::uint32_t row,
    std::uint32_t chunk_at, std::uint32_t tok, std::uint32_t batch_size,
    DataType::ElementQA (&fp8_act)[Fp8NumChunks][Fp8Tok][Fp8KInner],
    float* __restrict__ quantized_input_scale_for_step) {
    static_assert(NumRows <= 16u, "Streaming quantize supports BS<=16");
    static_assert(Fp8NumChunks == ChunksPerSlot,
                  "fp8_act must hold the slot as its 16-K chunks");
    static_assert(Fp8Tok == NumRows || Fp8Tok == NumRows + 1u,
                  "fp8_act must carry the block's token rows, padded or not; "
                  "only the first ones are written");
    static_assert(Fp8KInner == 16, "fp8_act inner dim must be 16");

    moe_store_quantized_pair(moe_load_quantize_pair(bf16_4), row, chunk_at, tok,
                             fp8_act, batch_size,
                             quantized_input_scale_for_step);
}

template <bool Staged, std::size_t kRingRows, std::size_t kScaleCols,
          std::size_t NumCols, std::size_t SwizzleBoxK, std::size_t QuantBlock,
          std::size_t NumQuantBlocks, std::size_t kBlocksPerSlot>
__device__ inline void quantize_fp8_blockwise_128(
    const DataType::ElementA* __restrict__ src, std::uint32_t k_base,
    DataType::ElementQA (*__restrict__ quantized_input)[QuantBlock *
                                                        kBlocksPerSlot / 16u]
                                                       [kRingRows + 1u][16u],
    float* __restrict__ quantized_input_scale, std::uint32_t staging_slot_base,
    std::uint32_t scale_block_base, std::uint32_t src_block_base,
    std::uint32_t batch_size, std::uint32_t token_begin, std::uint32_t warp_idx,
    std::uint32_t num_quant_warps) {
    static_assert(kRingRows <= 16u,
                  "quantize_fp8_blockwise_128 supports BS<=16");
    static_assert(NumCols % QuantBlock == 0u,
                  "the tile must span whole quantization blocks");
    static_assert(NumQuantBlocks % kBlocksPerSlot == 0u,
                  "the tile's blocks must fill whole staging slots");
    constexpr std::uint32_t kChunksPerBlock = QuantBlock / 16u;
    const std::uint32_t col = WarpRole::get_lane_idx() * 4u;

#pragma unroll 1
    for (std::uint32_t i = warp_idx; i < kRingRows * NumQuantBlocks;
         i += num_quant_warps) {
        // The window's row, and the launch-wide token it holds.
        const std::uint32_t row = i / NumQuantBlocks;
        const std::uint32_t token = token_begin + row;
        const std::uint32_t blk = i % NumQuantBlocks;
        const std::uint32_t slot = staging_slot_base + blk / kBlocksPerSlot;
        const std::uint32_t chunk_at = (blk % kBlocksPerSlot) * kChunksPerBlock;
        const std::uint32_t sblk = scale_block_base + blk;
        const std::uint32_t k = (src_block_base + blk) * QuantBlock + col;
        // A padded token re-reads the ring's first row: the store zeroes it
        // anyway, and an unconditional read is what lets several pairs' loads
        // overlap.
        const std::uint32_t src_row = token < batch_size ? row : 0u;
        const DataType::ElementA* bf16_4;
        if constexpr (Staged) {
            // The box's 16-B chunk index is XOR-ed with the row.
            const std::uint32_t rel = k - k_base;
            const std::uint32_t chunk = (rel % SwizzleBoxK) / 8u;
            bf16_4 = src + (rel / SwizzleBoxK) * (kRingRows * SwizzleBoxK) +
                     src_row * SwizzleBoxK + (chunk ^ (src_row % 8u)) * 8u +
                     rel % 8u;
        } else {
            // A row-major source is the whole tensor, so it is addressed by the
            // launch-wide token.
            bf16_4 = src + (token < batch_size ? token : 0u) * NumCols + k;
        }
        moe_streaming_quantize_k128<kRingRows,
                                    QuantBlock / 16u * kBlocksPerSlot>(
            bf16_4, row, chunk_at, token, batch_size, quantized_input[slot],
            &quantized_input_scale[sblk * kScaleCols + token]);
    }
}

}  // namespace halops::moe
