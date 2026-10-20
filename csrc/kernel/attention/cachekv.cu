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

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include "kernel/attention/utils.cuh"

namespace halops {

// In-place RoPE on consecutive (2k, 2k+1) pairs; kPackSize must be even.
template <int kPackSize>
__device__ __forceinline__ void apply_rope(float* __restrict__ src_float,
                                           const float* __restrict__ cos,
                                           const float* __restrict__ sin) {
#pragma unroll
    for (int k = 0; k < kPackSize / 2; k++) {
        const float cos_val = cos[k];
        const float sin_val = sin[k];
        const float x = src_float[2 * k];
        const float y = src_float[2 * k + 1];
        src_float[2 * k] = cos_val * x - sin_val * y;
        src_float[2 * k + 1] = sin_val * x + cos_val * y;
    }
}

// In-place RMSNorm over one head. A head (kHeadDim = 128) spans 16 lanes, so
// the sum of squares is reduced over a half-warp and rms is broadcast from
// src_lane.
template <int kPackSize, int kHeadDim>
__device__ __forceinline__ void apply_rms_norm(
    float* __restrict__ src_float, const float* __restrict__ norm_gamma,
    const int gamma_idx, const int src_lane) {
    float sum_sq = 0.0f;
#pragma unroll
    for (int k = 0; k < kPackSize; k++) {
        sum_sq += src_float[k] * src_float[k];
    }
    unsigned mask = __activemask();
#pragma unroll
    for (int offset = 8; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down_sync(mask, sum_sq, offset, 16);
    }
    constexpr float head_dim_inv = 1.0f / static_cast<float>(kHeadDim);
    float rms = rsqrt(sum_sq * head_dim_inv + 1e-6f);
    rms = __shfl_sync(mask, rms, src_lane, 16);

#pragma unroll
    for (int k = 0; k < kPackSize; k++) {
        src_float[k] = src_float[k] * norm_gamma[gamma_idx + k] * rms;
    }
}

// Unpacks 8 BF16 from a float4, applies RoPE and, if norm_gamma is non-null,
// RMSNorm, then repacks in place.
template <int kPackSize, int kHeadDim>
__device__ __forceinline__ void apply_rope_and_norm(
    float4& src, float* __restrict__ src_float, const float* __restrict__ cos,
    const float* __restrict__ sin, const float* __restrict__ norm_gamma,
    const int gamma_idx, const int src_lane) {
    nv_bfloat162* src_bf16 = reinterpret_cast<nv_bfloat162*>(&src);
#pragma unroll
    for (int k = 0; k < kPackSize / 2; k++) {
        float2 xy = __bfloat1622float2(src_bf16[k]);
        src_float[2 * k] = xy.x;
        src_float[2 * k + 1] = xy.y;
    }

    apply_rope<kPackSize>(src_float, cos, sin);

    if (norm_gamma != nullptr) {
        apply_rms_norm<kPackSize, kHeadDim>(src_float, norm_gamma, gamma_idx,
                                            src_lane);
    }

#pragma unroll
    for (int k = 0; k < kPackSize / 2; k++) {
        src_bf16[k] =
            __float22bfloat162_rn({src_float[2 * k], src_float[2 * k + 1]});
    }
}

// Neox-style RoPE: left[k] pairs with right[k] of the rotary region. paired is
// the counterpart fragment; kIsRight selects the sign:
//   left'  = left * cos - right * sin
//   right' = right * cos + left * sin
template <int kPackSize, bool kIsRight>
__device__ __forceinline__ void apply_partial_rope(
    float4& src, const float4& paired, float* __restrict__ src_float,
    float* __restrict__ paired_float, const float* __restrict__ cos,
    const float* __restrict__ sin) {
    nv_bfloat162* src_bf16 = reinterpret_cast<nv_bfloat162*>(&src);
#pragma unroll
    for (int k = 0; k < kPackSize / 2; k++) {
        float2 xy = __bfloat1622float2(src_bf16[k]);
        src_float[2 * k] = xy.x;
        src_float[2 * k + 1] = xy.y;
    }
    const nv_bfloat162* paired_bf16 =
        reinterpret_cast<const nv_bfloat162*>(&paired);
#pragma unroll
    for (int k = 0; k < kPackSize / 2; k++) {
        float2 xy = __bfloat1622float2(paired_bf16[k]);
        paired_float[2 * k] = xy.x;
        paired_float[2 * k + 1] = xy.y;
    }

    constexpr float flip = kIsRight ? 1.0f : -1.0f;
#pragma unroll
    for (int k = 0; k < kPackSize; k++) {
        src_float[k] = src_float[k] * cos[k] + flip * paired_float[k] * sin[k];
    }

#pragma unroll
    for (int k = 0; k < kPackSize / 2; k++) {
        src_bf16[k] =
            __float22bfloat162_rn({src_float[2 * k], src_float[2 * k + 1]});
    }
}

// Advances cache to the next physical block (delta elements away) once
// store_row crosses kBlockSize.
template <typename T>
__device__ __forceinline__ void advance_cache_if_needed(T*& cache,
                                                        int& store_row,
                                                        const int kBlockSize,
                                                        const int64_t delta) {
    if (store_row >= kBlockSize) {
        store_row -= kBlockSize;
        cache += delta;
    }
}

// Copies cached prefix K/V from the BF16 paged cache into contiguous
// k/v_output. grid = (cache block, 2 * kv_head_num: K heads then V heads,
// batch). V rows are stored with the 0,1,2,3 -> 0,2,1,3 reorder and are
// un-permuted here.
template <typename T, int kBlockSize, int kThreads, int kHeadDim>
__global__ void read_cachekv_encoder(
    T* __restrict__ k_output, T* __restrict__ v_output,
    const int* __restrict__ cu_seq_k, const int* __restrict__ seq_len_encoders,
    const int* __restrict__ seq_len_decoders, const int kv_head_num,
    const int max_blocks_per_seq, const T* __restrict__ cache_k,
    const T* __restrict__ cache_v, const int* __restrict__ block_tables) {
    constexpr int kPackSize = 16 / sizeof(T);
    const int block_idx = blockIdx.x * kBlockSize;
    const int bidh = blockIdx.y;
    const int bidb = blockIdx.z;
    const int tidx = threadIdx.x;

    if (seq_len_encoders[bidb] == 0) return;
    const int seq_len_decoder = seq_len_decoders[bidb];
    const int remain_tokens = seq_len_decoder - block_idx;
    if (remain_tokens <= 0) return;

    const int row_idx = tidx / (kHeadDim / kPackSize);
    const int col_idx = tidx % (kHeadDim / kPackSize);

    static_assert(kBlockSize * kHeadDim % (kThreads * kPackSize) == 0);

    float4 src;

    const int64_t token_offset =
        static_cast<int64_t>(cu_seq_k[bidb] + block_idx);

    const int32_t* block_table_now = block_tables + bidb * max_blocks_per_seq;
    const uint32_t physical_block_number = block_table_now[blockIdx.x];

    const int64_t cache_offset = static_cast<int64_t>(physical_block_number) *
                                     kv_head_num * kBlockSize * kHeadDim +
                                 bidh * kBlockSize * kHeadDim +
                                 col_idx * kPackSize;
    const T* cache =
        (bidh < kv_head_num)
            ? (cache_k + cache_offset)
            : (cache_v + cache_offset - kv_head_num * kBlockSize * kHeadDim);

    const int kv_hidden = kv_head_num * kHeadDim;
    T* output_this_block =
        (bidh < kv_head_num)
            ? (k_output + token_offset * kv_hidden + bidh * kHeadDim +
               col_idx * kPackSize)
            : (v_output + token_offset * kv_hidden +
               (bidh - kv_head_num) * kHeadDim + col_idx * kPackSize);

    for (int i = row_idx; i < kBlockSize;
         i += kThreads / (kHeadDim / kPackSize)) {
        if (i >= remain_tokens) break;
        int read_row = i;
        if (bidh >= kv_head_num) {
            read_row =
                (read_row & ~3) | ((read_row & 1) << 1) | ((read_row & 2) >> 1);
        }
        src = *reinterpret_cast<const float4*>(cache + read_row * kHeadDim);
        *reinterpret_cast<float4*>(output_this_block + i * kv_hidden) = src;
    }
}

// Copies cached prefix K/V from the FP8 paged cache into BF16 k/v_output,
// dequantizing with the per-block per-head scale. Scale index:
// (phys_blk * kv_head_num + kv_h) * kDequantScalePackSize + {0: K, 1: V}.
template <typename T, int kBlockSize, int kThreads, int kHeadDim>
__global__ void read_fp8_cachekv_encoder(
    T* __restrict__ k_output, T* __restrict__ v_output,
    const int* __restrict__ cu_seq_k, const int* __restrict__ seq_len_encoders,
    const int* __restrict__ seq_len_decoders, const int kv_head_num,
    const int max_blocks_per_seq,
    const cutlass::float_e4m3_t* __restrict__ cache_k,
    const cutlass::float_e4m3_t* __restrict__ cache_v,
    const float* __restrict__ kv_dequant_scales,
    const int* __restrict__ block_tables) {
    constexpr int kFP8PackSize = 8;  // FP8 values per uint2 load.
    constexpr int kFP8VecsPerRow = kHeadDim / kFP8PackSize;

    const int block_idx = blockIdx.x * kBlockSize;
    const int bidh = blockIdx.y;  // [0, kv_head_num*2)
    const int bidb = blockIdx.z;
    const int tidx = threadIdx.x;

    if (seq_len_encoders[bidb] == 0) return;
    const int seq_len_decoder = seq_len_decoders[bidb];
    const int remain_tokens = seq_len_decoder - block_idx;
    if (remain_tokens <= 0) return;

    const int row_idx = tidx / kFP8VecsPerRow;
    const int col_idx = tidx % kFP8VecsPerRow;

    static_assert(kBlockSize * kHeadDim % (kThreads * kFP8PackSize) == 0);

    const int64_t token_offset =
        static_cast<int64_t>(cu_seq_k[bidb] + block_idx);

    const int32_t* block_table_now = block_tables + bidb * max_blocks_per_seq;
    const uint32_t physical_block_number = block_table_now[blockIdx.x];

    const int64_t cache_offset = static_cast<int64_t>(physical_block_number) *
                                     kv_head_num * kBlockSize * kHeadDim +
                                 bidh * kBlockSize * kHeadDim;
    const cutlass::float_e4m3_t* cache =
        (bidh < kv_head_num) ? (cache_k + cache_offset)
                             : (cache_v + cache_offset -
                                (int64_t)kv_head_num * kBlockSize * kHeadDim);

    // Stored scale is abs_max / kFP8Max, so bf16 = fp8 * scale.
    const int kv_h = (bidh < kv_head_num) ? bidh : (bidh - kv_head_num);
    const float dequant_factor =
        kv_dequant_scales[(physical_block_number * kv_head_num + kv_h) *
                              kDequantScalePackSize +
                          (bidh < kv_head_num ? 0 : 1)];

    const int kv_hidden = kv_head_num * kHeadDim;
    T* output_this_block =
        (bidh < kv_head_num)
            ? (k_output + token_offset * kv_hidden + bidh * kHeadDim)
            : (v_output + token_offset * kv_hidden +
               (bidh - kv_head_num) * kHeadDim);

    for (int i = row_idx; i < kBlockSize; i += kThreads / kFP8VecsPerRow) {
        if (i >= remain_tokens) break;

        int cache_row = i;
        if (bidh >= kv_head_num) {
            // V rows use the 8-row reorder written by write_cachekv_encoder_c8.
            cache_row = (cache_row & ~7) | ((cache_row & 1) << 2) |
                        ((cache_row & 6) >> 1);
        }

        const uint2 fp8_pack = reinterpret_cast<const uint2*>(
            cache)[cache_row * kFP8VecsPerRow + col_idx];

        const cutlass::float_e4m3_t* fp8_vals =
            reinterpret_cast<const cutlass::float_e4m3_t*>(&fp8_pack);
        nv_bfloat162 bf16_out[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            float a = float(fp8_vals[2 * k]) * dequant_factor;
            float b = float(fp8_vals[2 * k + 1]) * dequant_factor;
            bf16_out[k] = __floats2bfloat162_rn(a, b);
        }

        *reinterpret_cast<float4*>(output_this_block + i * kv_hidden +
                                   col_idx * kFP8PackSize) =
            *reinterpret_cast<float4*>(bf16_out);
    }
}

// Splits fused qkv_input [tokens, head_num + 2 * kv_head_num, head_dim] (all Q,
// then K, then V heads), applies Neox partial RoPE to Q/K, writes contiguous
// q/k/v_output and appends K/V to the BF16 paged cache.
// grid.y: [0, head_num) Q, [head_num, head_num + kv_head_num) K, the rest V.
// New tokens start at row seq_len_decoder % kBlockSize and may spill into the
// next physical block. V rows use the 0,1,2,3 -> 0,2,1,3 reorder expected by
// the attention kernel.
template <typename T, int kBlockSize, int kThreads, int kHeadDim>
__global__ void write_cachekv_encoder(
    const T* __restrict__ qkv_input, T* __restrict__ q_output,
    T* __restrict__ k_output, T* __restrict__ v_output,
    const int* __restrict__ cu_seq_q, const int* __restrict__ cu_seq_k,
    const float* __restrict__ rotary_embs,
    const int* __restrict__ seq_len_encoders,
    const int* __restrict__ seq_len_decoders, const int head_num,
    const int kv_head_num, const int max_blocks_per_seq,
    const int max_input_length, const int gqa_group_size,
    T* __restrict__ cache_k, T* __restrict__ cache_v,
    const int* __restrict__ block_tables,
    const float* __restrict__ q_norm_gamma,
    const float* __restrict__ k_norm_gamma) {
    constexpr int kPackSize = 16 / sizeof(T);
    const int block_idx = blockIdx.x * kBlockSize;
    const int bidh = blockIdx.y;
    const int bidb = blockIdx.z;
    const int tidx = threadIdx.x;

    const int seq_len = seq_len_encoders[bidb];
    const int seq_len_decoder = seq_len_decoders[bidb];
    const int remain_tokens = seq_len - block_idx;
    if (seq_len == 0 || remain_tokens <= 0) return;

    const int all_head_num = head_num + 2 * kv_head_num;
    const int hidden = all_head_num * kHeadDim;
    const int q_hidden = head_num * kHeadDim;
    const int kv_hidden = kv_head_num * kHeadDim;
    const int row_idx = tidx / (kHeadDim / kPackSize);
    const int col_idx = tidx % (kHeadDim / kPackSize);

    static_assert(kBlockSize * kHeadDim % (kThreads * kPackSize) == 0);

    float4 src;
    float src_float[kPackSize];
    float paired_float[kPackSize];

    constexpr int kVecsPerTok = kHeadDim / kPackSize;
    constexpr int kHalfVecs = kVecsPerTok / 4;
    constexpr int kRopeDim = kHeadDim / 4;

    const int64_t token_offset =
        static_cast<int64_t>(cu_seq_q[bidb] + block_idx);
    const int64_t token_offset_kv =
        static_cast<int64_t>(cu_seq_k[bidb] + block_idx + seq_len_decoder);
    const T* qkv_input_this_block = qkv_input + token_offset * hidden +
                                    bidh * kHeadDim + col_idx * kPackSize;

    const int32_t* block_table_now = block_tables + bidb * max_blocks_per_seq;
    const uint32_t physical_block_number =
        block_table_now[blockIdx.x + seq_len_decoder / kBlockSize];
    const uint32_t physical_block_number_next =
        block_table_now[blockIdx.x + seq_len_decoder / kBlockSize + 1];
    const int64_t cache_offset = static_cast<int64_t>(physical_block_number) *
                                     kv_head_num * kBlockSize * kHeadDim +
                                 (bidh - head_num) * kBlockSize * kHeadDim +
                                 col_idx * kPackSize;
    T* cache = cache_k + cache_offset;
    if (bidh >= head_num + kv_head_num) {
        cache = cache_v + cache_offset - kv_head_num * kBlockSize * kHeadDim;
    }

    const int row_idx_prev = seq_len_decoder % kBlockSize;
    const int64_t cache_block_stride =
        static_cast<int64_t>(physical_block_number_next -
                             physical_block_number) *
        kv_head_num * kBlockSize * kHeadDim;

    if (bidh < head_num) {
        // Q: partial RoPE -> q_output.
        T* q_output_this_block = q_output + token_offset * q_hidden +
                                 bidh * kHeadDim + col_idx * kPackSize;
        for (int i = row_idx; i < kBlockSize;
             i += kThreads / (kHeadDim / kPackSize)) {
            if (i >= remain_tokens) break;
            src = *reinterpret_cast<const float4*>(qkv_input_this_block +
                                                   i * hidden);
            const int token_idx = seq_len_decoder + block_idx + i;
            if (col_idx < kVecsPerTok / 2) {
                const int pair_col = col_idx ^ kHalfVecs;
                const float4 paired = *reinterpret_cast<const float4*>(
                    qkv_input + (int64_t)(token_offset + i) * hidden +
                    bidh * kHeadDim + pair_col * kPackSize);
                const float* cos = rotary_embs + token_idx * kRopeDim +
                                   (col_idx & (kHalfVecs - 1)) * kPackSize;
                const float* sin = cos + max_input_length * kRopeDim;
                if (col_idx < kHalfVecs) {
                    apply_partial_rope<kPackSize, false>(
                        src, paired, src_float, paired_float, cos, sin);
                } else {
                    apply_partial_rope<kPackSize, true>(src, paired, src_float,
                                                        paired_float, cos, sin);
                }
            }
            *reinterpret_cast<float4*>(q_output_this_block + i * q_hidden) =
                src;
        }

    } else if (bidh < head_num + kv_head_num) {
        // K: partial RoPE -> k_output and cache_k.
        T* const base_cache = cache;
        T* k_output_this_block = k_output + token_offset_kv * kv_hidden +
                                 (bidh - head_num) * kHeadDim +
                                 col_idx * kPackSize;
        for (int i = row_idx; i < kBlockSize;
             i += kThreads / (kHeadDim / kPackSize)) {
            if (i >= remain_tokens) break;
            const int abs_row = i + row_idx_prev;
            const int store_row = abs_row % kBlockSize;
            T* const effective_cache = (abs_row >= kBlockSize)
                                           ? (base_cache + cache_block_stride)
                                           : base_cache;
            src = *reinterpret_cast<const float4*>(qkv_input_this_block +
                                                   i * hidden);
            const int token_idx = seq_len_decoder + block_idx + i;
            if (col_idx < kVecsPerTok / 2) {
                const int pair_col = col_idx ^ kHalfVecs;
                const float4 paired = *reinterpret_cast<const float4*>(
                    qkv_input + (int64_t)(token_offset + i) * hidden +
                    bidh * kHeadDim + pair_col * kPackSize);
                const float* cos = rotary_embs + token_idx * kRopeDim +
                                   (col_idx & (kHalfVecs - 1)) * kPackSize;
                const float* sin = cos + max_input_length * kRopeDim;
                if (col_idx < kHalfVecs) {
                    apply_partial_rope<kPackSize, false>(
                        src, paired, src_float, paired_float, cos, sin);
                } else {
                    apply_partial_rope<kPackSize, true>(src, paired, src_float,
                                                        paired_float, cos, sin);
                }
            }
            *reinterpret_cast<float4*>(effective_cache + store_row * kHeadDim) =
                src;
            *reinterpret_cast<float4*>(k_output_this_block + i * kv_hidden) =
                src;
        }

    } else {
        // V: v_output and cache_v (rows reordered 0 1 2 3 -> 0 2 1 3).
        T* const base_cache = cache;
        T* v_output_this_block = v_output + token_offset_kv * kv_hidden +
                                 (bidh - head_num - kv_head_num) * kHeadDim +
                                 col_idx * kPackSize;
        for (int i = row_idx; i < kBlockSize;
             i += kThreads / (kHeadDim / kPackSize)) {
            if (i >= remain_tokens) break;
            const int abs_row = i + row_idx_prev;
            int store_row = abs_row % kBlockSize;
            T* const effective_cache = (abs_row >= kBlockSize)
                                           ? (base_cache + cache_block_stride)
                                           : base_cache;
            src = *reinterpret_cast<const float4*>(qkv_input_this_block +
                                                   i * hidden);
            *reinterpret_cast<float4*>(v_output_this_block + i * kv_hidden) =
                src;
            store_row = (store_row & ~3) | ((store_row & 1) << 1) |
                        ((store_row & 2) >> 1);
            *reinterpret_cast<float4*>(effective_cache + store_row * kHeadDim) =
                src;
        }
    }
}

// FP8 (C8) encoder cache write. Q/K get RoPE + optional RMSNorm and are
// written to BF16 q/k_output; V goes to BF16 v_output. K/V are also quantized
// to FP8 into the paged cache with a per-block per-head abs-max scale (stored
// as abs_max / kFP8Max). Each CTA covers one physical cache block. A first
// block partially filled by the decode prefix reuses its existing scale so old
// and new tokens share one scale. V uses the 8-row reorder expected by FP8
// decode attention.
template <typename T, int kHeadDim, int kBlockSize>
__global__ void write_cachekv_encoder_c8(
    const T* __restrict__ qkv_input, T* __restrict__ q_output,
    T* __restrict__ k_output, T* __restrict__ v_output,
    cutlass::float_e4m3_t* cache_k, cutlass::float_e4m3_t* cache_v,
    float* kv_dequant_scales, const int* __restrict__ cu_seq_q,
    const int* __restrict__ cu_seq_k, const float* __restrict__ rotary_embs,
    const int* __restrict__ seq_len_encoders,
    const int* __restrict__ seq_len_decoders,
    const int* __restrict__ block_tables,
    const float* __restrict__ q_norm_gamma,
    const float* __restrict__ k_norm_gamma, const int max_blocks_per_seq,
    const int head_num, const int kv_head_num, const int max_input_length) {
    constexpr int kPackSize = 16 / sizeof(T);
    constexpr int kVecsPerTok = kHeadDim / kPackSize;
    constexpr int kToksPerWarp = 32 / kVecsPerTok;
    constexpr int kNumWarps = kBlockSize / 32;
    constexpr int kIters = kBlockSize / (kNumWarps * kToksPerWarp);

    const int blk_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int batch = blockIdx.z;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int lane_tok = lane_id >> 4;
    const int lane_vec = lane_id & 15;

    const int seq_len_decoder = seq_len_decoders[batch];
    const int row_idx_prev = seq_len_decoder % kBlockSize;
    // blk_idx 0 fills the tail [row_idx_prev, kBlockSize) of the first physical
    // block after the decode prefix; blk_idx >= 1 each cover a full new block.
    const int token_offset_this_block =
        (blk_idx == 0)
            ? 0
            : (kBlockSize - row_idx_prev) + (blk_idx - 1) * kBlockSize;
    const int tokens_this_block =
        (blk_idx == 0) ? (kBlockSize - row_idx_prev) : kBlockSize;
    const int cache_row_base = (blk_idx == 0) ? row_idx_prev : 0;
    const int tile_start = cu_seq_q[batch] + token_offset_this_block;
    const int tile_start_kv =
        cu_seq_k[batch] + token_offset_this_block + seq_len_decoder;
    const int tile_end =
        min(tile_start + tokens_this_block, cu_seq_q[batch + 1]);
    const int valid_toks = tile_end - tile_start;
    if (valid_toks <= 0) return;
    if (seq_len_encoders[batch] == 0) return;

    const int stride = head_num + 2 * kv_head_num;
    const int stride_vecs = stride * kVecsPerTok;
    const int warp_base = warp_id * kToksPerWarp;

    float src_float[kPackSize];

    alignas(16) __shared__ float4 smem_kv[kHeadDim * kBlockSize / 8];
    const int gamma_idx = lane_vec * kPackSize;
    const int src_lane = lane_id & ~0xF;

    const int32_t* block_table_now = block_tables + batch * max_blocks_per_seq;
    const uint32_t physical_block_number =
        block_table_now[blk_idx + seq_len_decoder / kBlockSize];
    const int64_t cache_offset = static_cast<int64_t>(physical_block_number) *
                                     kv_head_num * kBlockSize * kHeadDim +
                                 (head_idx - head_num) * kBlockSize * kHeadDim;
    cutlass::float_e4m3_t* cache = cache_k + cache_offset;
    float* dequant_scales =
        kv_dequant_scales +
        physical_block_number * kv_head_num * kDequantScalePackSize +
        (head_idx - head_num) * kDequantScalePackSize;
    if (head_idx >= head_num + kv_head_num) {
        cache = cache_v + cache_offset - kv_head_num * kBlockSize * kHeadDim;
        dequant_scales -= kv_head_num * kDequantScalePackSize;
    }

    if (head_idx < head_num) {
        const int dst_stride_vecs = head_num * kVecsPerTok;
        const float4* src = reinterpret_cast<const float4*>(
            qkv_input + (int64_t)tile_start * stride * kHeadDim +
            head_idx * kHeadDim);
        float4* dst = reinterpret_cast<float4*>(
            q_output + (int64_t)tile_start * head_num * kHeadDim +
            head_idx * kHeadDim);

#pragma unroll
        for (int it = 0; it < kIters; ++it) {
            const int tok =
                warp_base + it * kNumWarps * kToksPerWarp + lane_tok;
            const int token_idx =
                seq_len_decoder + tok + token_offset_this_block;
            const float* cos = rotary_embs + token_idx * (kHeadDim / 2) +
                               lane_vec * (kPackSize / 2);
            const float* sin = cos + max_input_length * (kHeadDim / 2);
            if (tok < valid_toks) {
                float4 q = src[tok * stride_vecs + lane_vec];
                apply_rope_and_norm<kPackSize, kHeadDim>(
                    q, src_float, cos, sin, q_norm_gamma, gamma_idx, src_lane);
                dst[tok * dst_stride_vecs + lane_vec] = q;
            }
        }
    } else if (head_idx < head_num + kv_head_num) {
        const int k_h = head_idx - head_num;
        const int k_off = head_num + k_h;
        const int dst_stride_vecs = kv_head_num * kVecsPerTok;
        const float4* k_vecs = reinterpret_cast<const float4*>(
            qkv_input + (int64_t)tile_start * stride * kHeadDim +
            k_off * kHeadDim);
        float4* dst = reinterpret_cast<float4*>(
            k_output + (int64_t)tile_start_kv * kv_head_num * kHeadDim +
            k_h * kHeadDim);
        nv_bfloat162 k_max = {0.0f, 0.0f};
#pragma unroll
        for (int it = 0; it < kIters; ++it) {
            const int tok =
                warp_base + it * kNumWarps * kToksPerWarp + lane_tok;
            const int token_idx =
                seq_len_decoder + tok + token_offset_this_block;
            const float* cos = rotary_embs + token_idx * (kHeadDim / 2) +
                               lane_vec * (kPackSize / 2);
            const float* sin = cos + max_input_length * (kHeadDim / 2);
            if (tok < valid_toks) {
                float4 k_tmp = k_vecs[tok * stride_vecs + lane_vec];
                apply_rope_and_norm<kPackSize, kHeadDim>(k_tmp, src_float, cos,
                                                         sin, k_norm_gamma,
                                                         gamma_idx, src_lane);
                dst[tok * dst_stride_vecs + lane_vec] = k_tmp;
                bf16_abs_max(k_tmp, k_max);
                smem_kv[tok * kVecsPerTok + lane_vec] = k_tmp;
            }
        }
        float ks_f;
        if (blk_idx == 0 && row_idx_prev != 0) {
            // Block already holds decode-prefix tokens: reuse its scale.
            ks_f = 1.0f / dequant_scales[0];
        } else {
            k_max = bf16_abs(k_max);
            float k_max_f = max(float(k_max.x), float(k_max.y));
            k_max_f =
                fmaxf(BlockAllReduce<float, MaxOp<float>, kBlockSize>(k_max_f),
                      1e-6f);
            ks_f = kFP8Max / k_max_f;
            if (tid == 0) {
                dequant_scales[0] = k_max_f * kFP8MaxRcp;
            }
        }
        const nv_bfloat162 ks = __floats2bfloat162_rn(ks_f, ks_f);
        uint2 kfp8;
#pragma unroll
        for (int it = 0; it < kIters; ++it) {
            const int tok =
                warp_base + it * kNumWarps * kToksPerWarp + lane_tok;
            if (tok < valid_toks) {
                float4 k_tmp = smem_kv[tok * kVecsPerTok + lane_vec];
                quant_to_fp8(k_tmp, ks, kfp8);
                const int store_row = tok + cache_row_base;
                reinterpret_cast<uint2*>(
                    cache)[store_row * kVecsPerTok + lane_vec] = kfp8;
            }
        }
    } else {
        const int v_h = head_idx - head_num - kv_head_num;
        const int v_off = head_num + kv_head_num + v_h;
        const int dst_stride_vecs = kv_head_num * kVecsPerTok;
        const float4* v_vecs = reinterpret_cast<const float4*>(
            qkv_input + (int64_t)tile_start * stride * kHeadDim +
            v_off * kHeadDim);
        float4* dst = reinterpret_cast<float4*>(
            v_output + (int64_t)tile_start_kv * kv_head_num * kHeadDim +
            v_h * kHeadDim);
        nv_bfloat162 v_max = {0.0f, 0.0f};
#pragma unroll
        for (int it = 0; it < kIters; ++it) {
            const int tok =
                warp_base + it * kNumWarps * kToksPerWarp + lane_tok;
            if (tok < valid_toks) {
                float4 v_tmp = v_vecs[tok * stride_vecs + lane_vec];
                dst[tok * dst_stride_vecs + lane_vec] = v_tmp;
                bf16_abs_max(v_tmp, v_max);
                smem_kv[tok * kVecsPerTok + lane_vec] = v_tmp;
            }
        }
        float vs_f;
        if (blk_idx == 0 && row_idx_prev != 0) {
            // Block already holds decode-prefix tokens: reuse its scale.
            vs_f = 1.0f / dequant_scales[1];
        } else {
            v_max = bf16_abs(v_max);
            float v_max_f = max(float(v_max.x), float(v_max.y));
            v_max_f =
                fmaxf(BlockAllReduce<float, MaxOp<float>, kBlockSize>(v_max_f),
                      1e-6f);
            vs_f = kFP8Max / v_max_f;
            if (tid == 0) {
                dequant_scales[1] = v_max_f * kFP8MaxRcp;
            }
        }
        const nv_bfloat162 vs = __floats2bfloat162_rn(vs_f, vs_f);
        uint2 vfp8;
#pragma unroll
        for (int it = 0; it < kIters; ++it) {
            const int tok =
                warp_base + it * kNumWarps * kToksPerWarp + lane_tok;
            if (tok < valid_toks) {
                float4 v_tmp = smem_kv[tok * kVecsPerTok + lane_vec];
                quant_to_fp8(v_tmp, vs, vfp8);
                int store_row = tok + cache_row_base;
                store_row = (store_row & ~7) | ((store_row & 1) << 2) |
                            ((store_row & 6) >> 1);
                reinterpret_cast<uint2*>(
                    cache)[store_row * kVecsPerTok + lane_vec] = vfp8;
            }
        }
    }
}

// BF16 decode cache write; batches still in prefill are skipped.
// Q gets RoPE + optional RMSNorm and is written to q_output
// [batch, kv_head_num, kBlockM, head_dim], with tokens and the GQA group folded
// into kBlockM. K (RoPE + optional RMSNorm) and V are appended to the paged
// cache at seq_len_decoder; V uses the 0,2,1,3 row reorder.
template <typename T, int kBlockSize, int kThreads, int kHeadDim>
__global__ void __launch_bounds__(kThreads, 8) write_cachekv_decoder(
    const T* __restrict__ qkv_input, T* __restrict__ q_output,
    const int* __restrict__ cu_seq_q, const int* __restrict__ seq_len_encoders,
    const int* __restrict__ seq_len_decoders,
    const float* __restrict__ rotary_embs, const int max_input_length,
    T* __restrict__ cache_k, T* __restrict__ cache_v,
    const int* __restrict__ block_tables, const int head_num,
    const int kv_head_num, const int max_blocks_per_seq,
    const int gqa_group_size, const float* __restrict__ q_norm_gamma,
    const float* __restrict__ k_norm_gamma) {
    constexpr int kPackSize = 16 / sizeof(T);
    const int block_idx = blockIdx.x * kBlockSize;
    const int bidh = blockIdx.y;
    const int bidb = blockIdx.z;
    const int tidx = threadIdx.x;

    const int seq_len_decoder = seq_len_decoders[bidb];
    const int remain_tokens = cu_seq_q[bidb + 1] - cu_seq_q[bidb];

    if (seq_len_encoders[bidb] > 0 || seq_len_decoder == 0) return;

    const int all_head_num = head_num + 2 * kv_head_num;
    const int hidden = all_head_num * kHeadDim;
    const int row_idx = tidx / (kHeadDim / kPackSize);
    const int col_idx = tidx % (kHeadDim / kPackSize);

    static_assert(kBlockSize * kHeadDim % (kThreads * kPackSize) == 0);

    float4 src;
    float src_float[kPackSize];

    const int64_t token_offset =
        static_cast<int64_t>(cu_seq_q[bidb] + block_idx);
    const T* __restrict__ qkv_input_this_block =
        qkv_input + token_offset * hidden + bidh * kHeadDim +
        col_idx * kPackSize;

    const int32_t* __restrict__ block_table_now =
        block_tables + bidb * max_blocks_per_seq;
    const uint32_t physical_block_number =
        block_table_now[blockIdx.x + (seq_len_decoder + row_idx) / kBlockSize];

    const int64_t cache_offset = static_cast<int64_t>(physical_block_number) *
                                     kv_head_num * kBlockSize * kHeadDim +
                                 (bidh - head_num) * kBlockSize * kHeadDim +
                                 col_idx * kPackSize;
    T* cache = cache_k + cache_offset;
    if (bidh >= head_num + kv_head_num) {
        cache = cache_v + cache_offset - kv_head_num * kBlockSize * kHeadDim;
    }

    const int row_idx_prev = seq_len_decoder % kBlockSize;
    const int gamma_idx = col_idx * kPackSize;
    const int src_lane = tidx % 32 & ~0xF;

    constexpr int kBlockM = 64;

    if (bidh < head_num) {
        // Q: RoPE + optional QNorm -> q_output (GQA-folded).
        const int bidh_kv = bidh / gqa_group_size;
        for (int i = row_idx; i < kBlockSize;
             i += kThreads / (kHeadDim / kPackSize)) {
            src = (i < remain_tokens) ? *reinterpret_cast<const float4*>(
                                            qkv_input_this_block + i * hidden)
                                      : float4{0.0f, 0.0f, 0.0f, 0.0f};

            const int token_idx = seq_len_decoders[bidb] + i;
            const float* cos = rotary_embs + token_idx * (kHeadDim / 2) +
                               col_idx * (kPackSize / 2);
            const float* sin = cos + max_input_length * (kHeadDim / 2);
            apply_rope_and_norm<kPackSize, kHeadDim>(
                src, src_float, cos, sin, q_norm_gamma, gamma_idx, src_lane);

            const int bidh_kv_mod = bidh % gqa_group_size;
            const int store_token = i * gqa_group_size + bidh_kv_mod;
            const uint32_t store_idx =
                (bidb * kv_head_num + bidh_kv) * (kBlockM * kHeadDim) +
                store_token * kHeadDim + col_idx * kPackSize;
            if (store_token < kBlockM) {
                *reinterpret_cast<float4*>(q_output + store_idx) = src;
            }
        }

    } else if (bidh < head_num + kv_head_num) {
        // K: RoPE + optional KNorm -> cache_k.
        for (int i = row_idx; i < kBlockSize;
             i += kThreads / (kHeadDim / kPackSize)) {
            int store_row = (i + row_idx_prev) % kBlockSize;
            if (i >= remain_tokens) break;
            src = *reinterpret_cast<const float4*>(qkv_input_this_block +
                                                   i * hidden);
            const int token_idx = seq_len_decoders[bidb] + i;
            const float* cos = rotary_embs + token_idx * (kHeadDim / 2) +
                               col_idx * (kPackSize / 2);
            const float* sin = cos + max_input_length * (kHeadDim / 2);
            apply_rope_and_norm<kPackSize, kHeadDim>(
                src, src_float, cos, sin, k_norm_gamma, gamma_idx, src_lane);
            *reinterpret_cast<float4*>(cache + store_row * kHeadDim) = src;
        }

    } else {
        // V: cache_v (rows reordered 0 1 2 3 -> 0 2 1 3).
        for (int i = row_idx; i < kBlockSize;
             i += kThreads / (kHeadDim / kPackSize)) {
            int store_row = (i + row_idx_prev) % kBlockSize;
            if (i >= remain_tokens) break;
            src = *reinterpret_cast<const float4*>(qkv_input_this_block +
                                                   i * hidden);
            store_row = (store_row & ~3) | ((store_row & 1) << 1) |
                        ((store_row & 2) >> 1);
            *reinterpret_cast<float4*>(cache + store_row * kHeadDim) = src;
        }
    }
}

// Prefill entry. Splits fused QKV into q/k/v_input for encoder attention and
// writes new K/V to the paged cache ("none": BF16, "block_wise_fp8": FP8,
// requires kv_dequant_scale). If a sample has a cached decode prefix, its K/V
// are read back from the cache into k/v_input ahead of the new tokens.
void WriteCacheKVEncoder(
    const torch::Tensor& qkv_input, const torch::Tensor& cu_seq_q,
    const torch::Tensor& cu_seq_k, const torch::Tensor& rotary_embs,
    const torch::Tensor& seq_len_encoder, const torch::Tensor& seq_len_decoder,
    const torch::Tensor& cache_k, const torch::Tensor& cache_v,
    const torch::Tensor& block_tables, const torch::Tensor& q_input,
    const torch::Tensor& k_input, const torch::Tensor& v_input,
    const std::optional<torch::Tensor>& kv_dequant_scale,
    const std::optional<torch::Tensor>& q_norm_gamma,
    const std::optional<torch::Tensor>& k_norm_gamma, const int head_num,
    const int kv_head_num, const int head_dim, const int max_seq_len_q,
    const int max_seq_len_k, const int max_input_len,
    const std::string& cache_quant_type_str) {
    constexpr int kThreads = 128;
    constexpr int kHeadDim = 128;
    constexpr int kBlockSize = 128;
    assert(kHeadDim == head_dim);

    const int batch_size = cu_seq_q.size(0) - 1;
    const int64_t total_tokens = qkv_input.size(0);
    const int max_blocks_per_seq = block_tables.size(1);
    const int gqa_group_size = head_num / kv_head_num;

    using T = c10::BFloat16;
    using fp8 = c10::Float8_e4m3fn;
    dim3 grid_dims;

    if (cache_quant_type_str == "block_wise_fp8") {
        if (!kv_dequant_scale) {
            TORCH_CHECK(
                false,
                "kv_dequant_scale must not be null when cache_quant_type_str "
                "is block_wise_fp8");
        }

        const float* kv_dequant_scales =
            kv_dequant_scale.value().data_ptr<float>();

        grid_dims.x = (max_seq_len_q + kBlockSize - 1) / kBlockSize + 1;
        grid_dims.y = head_num + 2 * kv_head_num;
        grid_dims.z = batch_size;
        write_cachekv_encoder_c8<T, kHeadDim, kBlockSize>
            <<<grid_dims, kThreads, 0,
               at::cuda::getCurrentCUDAStream(qkv_input.device().index())
                   .stream()>>>(
                const_cast<T*>(qkv_input.data_ptr<T>()),
                const_cast<T*>(q_input.data_ptr<T>()),
                const_cast<T*>(k_input.data_ptr<T>()),
                const_cast<T*>(v_input.data_ptr<T>()),
                reinterpret_cast<cutlass::float_e4m3_t*>(
                    const_cast<uint8_t*>(cache_k.data_ptr<uint8_t>())),
                reinterpret_cast<cutlass::float_e4m3_t*>(
                    const_cast<uint8_t*>(cache_v.data_ptr<uint8_t>())),
                const_cast<float*>(kv_dequant_scales), cu_seq_q.data_ptr<int>(),
                cu_seq_k.data_ptr<int>(), rotary_embs.data_ptr<float>(),
                seq_len_encoder.data_ptr<int>(),
                seq_len_decoder.data_ptr<int>(), block_tables.data_ptr<int>(),
                q_norm_gamma ? q_norm_gamma.value().data_ptr<float>() : nullptr,
                k_norm_gamma ? k_norm_gamma.value().data_ptr<float>() : nullptr,
                max_blocks_per_seq, head_num, kv_head_num, max_input_len);

        // Mixed batch: dequantize cached prefix K/V into k_input/v_input.
        if (q_input.size(0) < k_input.size(0)) {
            grid_dims.x = (max_seq_len_k + kBlockSize - 1) / kBlockSize;
            grid_dims.y = kv_head_num * 2;
            grid_dims.z = batch_size;
            read_fp8_cachekv_encoder<T, kBlockSize, kThreads, kHeadDim>
                <<<grid_dims, kThreads, 0,
                   at::cuda::getCurrentCUDAStream(qkv_input.device().index())
                       .stream()>>>(
                    const_cast<T*>(k_input.data_ptr<T>()),
                    const_cast<T*>(v_input.data_ptr<T>()),
                    cu_seq_k.data_ptr<int>(), seq_len_encoder.data_ptr<int>(),
                    seq_len_decoder.data_ptr<int>(), kv_head_num,
                    max_blocks_per_seq,
                    reinterpret_cast<const cutlass::float_e4m3_t*>(
                        cache_k.data_ptr<uint8_t>()),
                    reinterpret_cast<const cutlass::float_e4m3_t*>(
                        cache_v.data_ptr<uint8_t>()),
                    kv_dequant_scales, block_tables.data_ptr<int>());
        }
        return;
    }

    if (q_input.size(0) < k_input.size(0)) {
        grid_dims.x = (max_seq_len_k + kBlockSize - 1) / kBlockSize;
        grid_dims.y = kv_head_num * 2;
        grid_dims.z = batch_size;
        read_cachekv_encoder<T, kBlockSize, kThreads, kHeadDim>
            <<<grid_dims, kThreads, 0,
               at::cuda::getCurrentCUDAStream(qkv_input.device().index())
                   .stream()>>>(
                const_cast<T*>(k_input.data_ptr<T>()),
                const_cast<T*>(v_input.data_ptr<T>()), cu_seq_k.data_ptr<int>(),
                seq_len_encoder.data_ptr<int>(),
                seq_len_decoder.data_ptr<int>(), kv_head_num,
                max_blocks_per_seq, const_cast<T*>(cache_k.data_ptr<T>()),
                const_cast<T*>(cache_v.data_ptr<T>()),
                block_tables.data_ptr<int>());
    }

    grid_dims.x = (max_seq_len_q + kBlockSize - 1) / kBlockSize;
    grid_dims.y = head_num + kv_head_num * 2;
    grid_dims.z = batch_size;

    write_cachekv_encoder<T, kBlockSize, kThreads, kHeadDim><<<
        grid_dims, kThreads, 0,
        at::cuda::getCurrentCUDAStream(qkv_input.device().index()).stream()>>>(
        const_cast<T*>(qkv_input.data_ptr<T>()),
        const_cast<T*>(q_input.data_ptr<T>()),
        const_cast<T*>(k_input.data_ptr<T>()),
        const_cast<T*>(v_input.data_ptr<T>()), cu_seq_q.data_ptr<int>(),
        cu_seq_k.data_ptr<int>(), rotary_embs.data_ptr<float>(),
        seq_len_encoder.data_ptr<int>(), seq_len_decoder.data_ptr<int>(),
        head_num, kv_head_num, max_blocks_per_seq, max_input_len,
        gqa_group_size, const_cast<T*>(cache_k.data_ptr<T>()),
        const_cast<T*>(cache_v.data_ptr<T>()), block_tables.data_ptr<int>(),
        q_norm_gamma ? q_norm_gamma.value().data_ptr<float>() : nullptr,
        k_norm_gamma ? k_norm_gamma.value().data_ptr<float>() : nullptr);
}

// Scale layout: [phys_blk, kv_head, kDequantScalePackSize]; slot 0 = K, 1 = V.
__device__ __forceinline__ float kv_scale_k(const float* __restrict__ s,
                                            int phys_blk, int kv_head_num,
                                            int kv_h) {
    return s[(phys_blk * kv_head_num + kv_h) * kDequantScalePackSize + 0];
}

__device__ __forceinline__ float kv_scale_v(const float* __restrict__ s,
                                            int phys_blk, int kv_head_num,
                                            int kv_h) {
    return s[(phys_blk * kv_head_num + kv_h) * kDequantScalePackSize + 1];
}

__device__ __forceinline__ int phys_block(const int* __restrict__ bt,
                                          int bt_stride, int batch,
                                          int logical_blk) {
    return bt[batch * bt_stride + logical_blk];
}

// FP8 cache tile of (phys_blk, kv_head_idx), laid out [kBlockSize, kHeadDim].
template <int kBlockSize, int kHeadDim>
__device__ __forceinline__ cutlass::float_e4m3_t* cache_tile_ptr(
    cutlass::float_e4m3_t* cache, int phys_blk, int kv_head_num,
    int kv_head_idx) {
    constexpr int kTileElems = kBlockSize * kHeadDim;
    return cache + ((int64_t)phys_blk * kv_head_num + kv_head_idx) * kTileElems;
}

// Rescales 8 packed FP8 values by ratio; used to requantize existing tokens of
// a partial block when its block-wise scale grows.
__device__ __forceinline__ uint2 rescale_fp8_u2(uint2 src, float ratio) {
    const cutlass::float_e4m3_t* s =
        reinterpret_cast<const cutlass::float_e4m3_t*>(&src);
    uint2 dst;
    cutlass::float_e4m3_t* d = reinterpret_cast<cutlass::float_e4m3_t*>(&dst);
#pragma unroll
    for (int i = 0; i < 8; ++i)
        d[i] = cutlass::float_e4m3_t(float(s[i]) * ratio);
    return dst;
}

__device__ __forceinline__ float* kv_scale_k_ptr(float* s, int phys_blk,
                                                 int kv_head_num, int kv_h) {
    return s + (phys_blk * kv_head_num + kv_h) * 4 + 0;
}

__device__ __forceinline__ float* kv_scale_v_ptr(float* s, int phys_blk,
                                                 int kv_head_num, int kv_h) {
    return s + (phys_blk * kv_head_num + kv_h) * 4 + 1;
}

// FP8 (C8) decode cache write; one CTA of 256 threads per (kv head, batch).
// Warps 0-3 (Q): RoPE + optional QNorm, FP8 quantize into q_output
//   [batch, kv_head_num, 64, head_dim] (tokens x GQA group folded into 64
//   rows). q_dequant_scale also folds in the softmax log2 scale.
// Warps 4-7 (K/V): RoPE + optional KNorm on K, FP8 quantize into the cache.
// When new tokens land in a partially filled block and exceed its abs-max, the
// existing FP8 data is rescaled by old/new so the block keeps a single scale.
// V uses the 8-row reorder expected by FP8 attention.
template <typename T, int kHeadDim, int kBlockSize>
__global__ void write_cachekv_decoder_c8(
    const T* qkv_input, cutlass::float_e4m3_t* q_output, float* q_dequant_scale,
    cutlass::float_e4m3_t* cache_k, cutlass::float_e4m3_t* cache_v,
    float* kv_dequant_scale, const int* cu_seq_q, const int* seq_len_encoder,
    const int* seq_len_decoder, const float* rotary_embs,
    const int* block_tables, const float* q_norm_gamma,
    const float* k_norm_gamma, const int max_blocks_per_seq, const int head_num,
    const int kv_head_num, const int gqa_ratio, const int max_input_length) {
    constexpr int kQThreads = 128;
    constexpr int kKVThreads = 128;
    constexpr int kQWarps = kQThreads / 32;
    constexpr int kKVWarps = kKVThreads / 32;
    constexpr int kVecSize = 16 / sizeof(T);
    constexpr int kVecsPerRow = kHeadDim / kVecSize;

    // Per-warp abs-max partials: Q (kQWarps entries), then K-partial,
    // V-partial, K-fresh, V-fresh (kKVWarps entries each).
    __shared__ float smem[kQWarps + 4 * kKVWarps];
    float* const q_smem = smem;
    float* const kp_smem = smem + kQWarps;
    float* const vp_smem = smem + kQWarps + kKVWarps;
    float* const kf_smem = smem + kQWarps + 2 * kKVWarps;
    float* const vf_smem = smem + kQWarps + 3 * kKVWarps;

    const int kv_h = blockIdx.x;
    const int batch = blockIdx.y;
    const int tid = threadIdx.x;
    // Thread id within the KV workers; valid only when tid >= kQThreads.
    const int kv_tid = tid - kQThreads;

    const int q_start = cu_seq_q[batch];
    const int tok_num = cu_seq_q[batch + 1] - q_start;  // <= 4
    if (tok_num <= 0) return;

    if (seq_len_encoder[batch] > 0 || seq_len_decoder[batch] == 0) return;

    const int k_start = seq_len_decoder[batch];
    const int stride = head_num + 2 * kv_head_num;

    // Q phase: tok_num * gqa_ratio <= 64 rows, i.e. <= kMaxQ vecs per Q worker.
    constexpr int kMaxQ = 8;
    float4 q_data[kMaxQ];
    uint8_t q_row[kMaxQ];
    uint8_t q_vec[kMaxQ];
    int q_n = 0;
    float src_float[kVecSize];
    nv_bfloat162 q_max = {0.0f, 0.0f};

    const int gamma_idx = (tid % kVecsPerRow) * kVecSize;
    const int src_lane = tid % 32 & ~0xF;

    if (tid < kQThreads) {
        const float* cos_rope = rotary_embs + k_start * (kHeadDim / 2);
        const float* sin_rope = cos_rope + max_input_length * (kHeadDim / 2);
        for (int vi = tid; vi < tok_num * gqa_ratio * kVecsPerRow;
             vi += kQThreads) {
            // Guards the register arrays if tok_num * gqa_ratio exceeds 64.
            if (q_n >= kMaxQ) break;

            const int row = vi >> 4;
            const int vec = vi & 15;
            const int tok_idx = row / gqa_ratio;
            const int g = row % gqa_ratio;

            float4 val = reinterpret_cast<const float4*>(
                qkv_input + (q_start + tok_idx) * stride * kHeadDim +
                (kv_h * gqa_ratio + g) * kHeadDim)[vec];

            const int rope_offset =
                tok_idx * (kHeadDim / 2) + vec * (kVecSize / 2);

            apply_rope_and_norm<kVecSize, kHeadDim>(
                val, src_float, cos_rope + rope_offset, sin_rope + rope_offset,
                q_norm_gamma, gamma_idx, src_lane);

            bf16_abs_max(val, q_max);
            q_data[q_n] = val;
            q_row[q_n] = static_cast<uint8_t>(row);
            q_vec[q_n] = static_cast<uint8_t>(vec);
            ++q_n;
        }
    }

    // KV phase: tok_num * kVecsPerRow <= 64 vecs.
    const int k_off = head_num + kv_h;
    const int v_off = head_num + kv_head_num + kv_h;
    const int k_vecs = tok_num * kVecsPerRow;
    const int first_blk = k_start / kBlockSize;
    const bool partial_blk = (k_start % kBlockSize) != 0;

    float enc_k_abs = 0.f, enc_v_abs = 0.f;
    int phys_partial = -1;
    if (partial_blk) {
        phys_partial =
            phys_block(block_tables, max_blocks_per_seq, batch, first_blk);
        enc_k_abs =
            kv_scale_k(kv_dequant_scale, phys_partial, kv_head_num, kv_h) *
            kFP8Max;
        enc_v_abs =
            kv_scale_v(kv_dequant_scale, phys_partial, kv_head_num, kv_h) *
            kFP8Max;
    }

    float4 k_val = {}, v_val = {};
    int kv_tok_local = 0, kv_vec_local = 0;
    bool has_kv = false;
    // abs-max is tracked separately for new tokens landing in the partially
    // filled first block and for those in fresh blocks.
    nv_bfloat162 k_max_p = {0.0f, 0.0f}, k_max_f = {0.0f, 0.0f};
    nv_bfloat162 v_max_p = {0.0f, 0.0f}, v_max_f = {0.0f, 0.0f};

    if (tid >= kQThreads) {
        has_kv = (kv_tid < k_vecs);
        if (has_kv) {
            kv_tok_local = kv_tid / kVecsPerRow;
            kv_vec_local = kv_tid % kVecsPerRow;
            const T* base =
                qkv_input + (q_start + kv_tok_local) * stride * kHeadDim;
            k_val = reinterpret_cast<const float4*>(
                base + k_off * kHeadDim)[kv_vec_local];

            const float* cos_rope = rotary_embs +
                                    (k_start + kv_tok_local) * (kHeadDim / 2) +
                                    kv_vec_local * (kVecSize / 2);
            const float* sin_rope =
                cos_rope + max_input_length * (kHeadDim / 2);
            apply_rope_and_norm<kVecSize, kHeadDim>(k_val, src_float, cos_rope,
                                                    sin_rope, k_norm_gamma,
                                                    gamma_idx, src_lane);

            v_val = reinterpret_cast<const float4*>(
                base + v_off * kHeadDim)[kv_vec_local];

            const bool in_partial =
                partial_blk &&
                ((k_start + kv_tok_local) / kBlockSize == first_blk);
            if (in_partial) {
                bf16_abs_max(k_val, k_max_p);
                bf16_abs_max(v_val, v_max_p);
            } else {
                bf16_abs_max(k_val, k_max_f);
                bf16_abs_max(v_val, v_max_f);
            }
        }
    }

    // Five-way abs-max reduction (Q, K/V partial, K/V fresh) with one barrier.
    auto warp_reduce_max = [](float v) -> float {
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1)
            v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, mask));
        return v;
    };

    const int global_wid = tid >> 5;
    const int lid = tid & 31;

    {
        float v = fmaxf(__bfloat162float(bf16_abs(q_max).x),
                        __bfloat162float(bf16_abs(q_max).y));
        v = warp_reduce_max(v);
        if (lid == 0 && tid < kQThreads) q_smem[global_wid] = v;
    }

    {
        float kp = fmaxf(__bfloat162float(bf16_abs(k_max_p).x),
                         __bfloat162float(bf16_abs(k_max_p).y));
        float vp = fmaxf(__bfloat162float(bf16_abs(v_max_p).x),
                         __bfloat162float(bf16_abs(v_max_p).y));
        float kf = fmaxf(__bfloat162float(bf16_abs(k_max_f).x),
                         __bfloat162float(bf16_abs(k_max_f).y));
        float vf = fmaxf(__bfloat162float(bf16_abs(v_max_f).x),
                         __bfloat162float(bf16_abs(v_max_f).y));
        kp = warp_reduce_max(kp);
        vp = warp_reduce_max(vp);
        kf = warp_reduce_max(kf);
        vf = warp_reduce_max(vf);
        if (lid == 0 && tid >= kQThreads) {
            kp_smem[global_wid - kQWarps] = kp;
            vp_smem[global_wid - kQWarps] = vp;
            kf_smem[global_wid - kQWarps] = kf;
            vf_smem[global_wid - kQWarps] = vf;
        }
    }

    __syncthreads();

    float q_max_f;
    {
        float v = (lid < kQWarps) ? q_smem[lid] : 0.f;
        v = warp_reduce_max(v);
        q_max_f = fmaxf(__shfl_sync(0xffffffff, v, 0), 1e-6f);
    }

    float k_partial_f, k_fresh_f;
    {
        float vp = (lid < kKVWarps) ? kp_smem[lid] : 0.f;
        float vf = (lid < kKVWarps) ? kf_smem[lid] : 0.f;
        vp = warp_reduce_max(vp);
        vf = warp_reduce_max(vf);
        k_partial_f = fmaxf(__shfl_sync(0xffffffff, vp, 0), 1e-6f);
        k_fresh_f = fmaxf(__shfl_sync(0xffffffff, vf, 0), 1e-6f);
    }

    float v_partial_f, v_fresh_f;
    {
        float vp = (lid < kKVWarps) ? vp_smem[lid] : 0.f;
        float vf = (lid < kKVWarps) ? vf_smem[lid] : 0.f;
        vp = warp_reduce_max(vp);
        vf = warp_reduce_max(vf);
        v_partial_f = fmaxf(__shfl_sync(0xffffffff, vp, 0), 1e-6f);
        v_fresh_f = fmaxf(__shfl_sync(0xffffffff, vf, 0), 1e-6f);
    }

    constexpr float kLog2e = 1.4426950408889634074f;
    constexpr float kLog2kHeadDim = 11.313708498984f;
    constexpr float scale_softmax_log2 = 1.0f / kLog2kHeadDim * kLog2e;
    if (tid < kQThreads) {
        if (tid == 0)
            q_dequant_scale[batch * kv_head_num + kv_h] =
                q_max_f * (kFP8MaxRcp * scale_softmax_log2);

        const nv_bfloat162 qs =
            __floats2bfloat162_rn(kFP8Max / q_max_f, kFP8Max / q_max_f);
        cutlass::float_e4m3_t* q_tile =
            q_output + (batch * kv_head_num + kv_h) * 64 * kHeadDim;

#pragma unroll
        for (int i = 0; i < kMaxQ; ++i) {
            if (i >= q_n) break;
            uint2 fp8;
            quant_to_fp8(q_data[i], qs, fp8);
            *reinterpret_cast<uint2*>(q_tile + q_row[i] * kHeadDim +
                                      q_vec[i] * kVecSize) = fp8;
        }
    }

    // Rescale the partial block's existing FP8 tokens (all threads), then
    // update its scale. k/v_partial_f cover only new tokens in the partial
    // block, k/v_fresh_f only those in fresh blocks.
    const bool k_rescale = partial_blk && (k_partial_f > enc_k_abs);
    const bool v_rescale = partial_blk && (v_partial_f > enc_v_abs);
    const float k_part_abs =
        partial_blk ? (k_rescale ? k_partial_f : enc_k_abs) : k_fresh_f;
    const float v_part_abs =
        partial_blk ? (v_rescale ? v_partial_f : enc_v_abs) : v_fresh_f;

    if (k_rescale || v_rescale) {
        const int enc_pos = k_start % kBlockSize;
        const int rescale_vecs = enc_pos * kVecsPerRow;
        cutlass::float_e4m3_t* k_tile = cache_tile_ptr<kBlockSize, kHeadDim>(
            cache_k, phys_partial, kv_head_num, kv_h);
        cutlass::float_e4m3_t* v_tile = cache_tile_ptr<kBlockSize, kHeadDim>(
            cache_v, phys_partial, kv_head_num, kv_h);

        // kVecsPerRow == 16: vi >> 4 is the row and vi & 15 the vec.
        if (k_rescale) {
            const float k_ratio = enc_k_abs / k_partial_f;
            for (int vi = tid; vi < rescale_vecs; vi += blockDim.x) {
                uint2* slot = reinterpret_cast<uint2*>(
                    k_tile + (vi >> 4) * kHeadDim + (vi & 15) * kVecSize);
                *slot = rescale_fp8_u2(*slot, k_ratio);
            }
        }
        if (v_rescale) {
            const float v_ratio = enc_v_abs / v_partial_f;
            for (int vi = tid; vi < rescale_vecs; vi += blockDim.x) {
                uint2* slot = reinterpret_cast<uint2*>(
                    v_tile + (vi >> 4) * kHeadDim + (vi & 15) * kVecSize);
                *slot = rescale_fp8_u2(*slot, v_ratio);
            }
        }
        __syncthreads();

        if (tid == kQThreads) {
            if (k_rescale)
                *kv_scale_k_ptr(kv_dequant_scale, phys_partial, kv_head_num,
                                kv_h) = k_partial_f * kFP8MaxRcp;
            if (v_rescale)
                *kv_scale_v_ptr(kv_dequant_scale, phys_partial, kv_head_num,
                                kv_h) = v_partial_f * kFP8MaxRcp;
        }
    }

    // Write K/V scales of fresh (non-partial) blocks.
    if (tid == kQThreads) {
        int prev = -1;
        for (int t = 0; t < tok_num; ++t) {
            const int blk = (k_start + t) / kBlockSize;
            if (blk != prev && (!partial_blk || blk != first_blk)) {
                prev = blk;
                const int phys =
                    phys_block(block_tables, max_blocks_per_seq, batch, blk);
                *kv_scale_k_ptr(kv_dequant_scale, phys, kv_head_num, kv_h) =
                    k_fresh_f * kFP8MaxRcp;
                *kv_scale_v_ptr(kv_dequant_scale, phys, kv_head_num, kv_h) =
                    v_fresh_f * kFP8MaxRcp;
            }
        }
    }

    // Quantize and write new tokens.
    if (tid >= kQThreads && has_kv) {
        const int kpos = k_start + kv_tok_local;
        const int this_blk = kpos / kBlockSize;
        const int phys =
            phys_block(block_tables, max_blocks_per_seq, batch, this_blk);
        const int koff = kpos % kBlockSize;
        const int voff = (koff & ~7) | ((koff & 1) << 2) | ((koff & 6) >> 1);
        const float k_use =
            (partial_blk && this_blk == first_blk) ? k_part_abs : k_fresh_f;
        const float v_use =
            (partial_blk && this_blk == first_blk) ? v_part_abs : v_fresh_f;

        const nv_bfloat162 ks =
            __floats2bfloat162_rn(kFP8Max / k_use, kFP8Max / k_use);
        const nv_bfloat162 vs =
            __floats2bfloat162_rn(kFP8Max / v_use, kFP8Max / v_use);
        uint2 kfp8, vfp8;
        quant_to_fp8(k_val, ks, kfp8);
        quant_to_fp8(v_val, vs, vfp8);
        *reinterpret_cast<uint2*>(cache_tile_ptr<kBlockSize, kHeadDim>(
                                      cache_k, phys, kv_head_num, kv_h) +
                                  koff * kHeadDim + kv_vec_local * kVecSize) =
            kfp8;
        *reinterpret_cast<uint2*>(cache_tile_ptr<kBlockSize, kHeadDim>(
                                      cache_v, phys, kv_head_num, kv_h) +
                                  voff * kHeadDim + kv_vec_local * kVecSize) =
            vfp8;
    }
}

// Decode entry. Writes q_output [batch, kv_head_num, 64, head_dim] for decoder
// attention and appends K/V to the paged cache. q_output is BF16 for "none" and
// FP8 for "block_wise_fp8"; q_dequant_scale_output [batch, kv_head_num] is only
// meaningful on the FP8 path.
std::vector<torch::Tensor> WriteCacheKVDecoder(
    const torch::Tensor& qkv_input, const torch::Tensor& cu_seq_q,
    const torch::Tensor& seq_len_encoder, const torch::Tensor& seq_len_decoder,
    const torch::Tensor& rotary_embs, const torch::Tensor& cache_k,
    const torch::Tensor& cache_v, const torch::Tensor& block_tables,
    const std::optional<torch::Tensor>& kv_dequant_scale,
    const std::optional<torch::Tensor>& q_norm_gamma,
    const std::optional<torch::Tensor>& k_norm_gamma, const int head_num,
    const int kv_head_num, const int head_dim, const int max_input_len,
    const std::string& cache_quant_type_str) {
    constexpr int kThreads = 128;
    constexpr int kHeadDim = 128;
    constexpr int kBlockSize = 128;
    constexpr int kBlockM = 64;
    assert(kHeadDim == head_dim);

    const int batch_size = cu_seq_q.size(0) - 1;
    const int max_blocks_per_seq = block_tables.size(1);
    const int gqa_group_size = head_num / kv_head_num;

    using T = c10::BFloat16;
    using fp8 = c10::Float8_e4m3fn;

    dim3 grid_dims;

    torch::Tensor q_dequant_scale_output = torch::empty(
        {batch_size, kv_head_num}, torch::TensorOptions()
                                       .dtype(torch::kFloat32)
                                       .device(qkv_input.device()));

    if (cache_quant_type_str == "block_wise_fp8") {
        if (!kv_dequant_scale) {
            TORCH_CHECK(
                false,
                "kv_dequant_scale must not be null when cache_quant_type_str "
                "is block_wise_fp8");
        }

        torch::Tensor q_output =
            torch::empty({batch_size, kv_head_num, kBlockM, head_dim},
                         torch::TensorOptions()
                             .dtype(torch::kFloat8_e4m3fn)
                             .device(qkv_input.device()));

        grid_dims.x = kv_head_num;
        grid_dims.y = batch_size;

        write_cachekv_decoder_c8<T, kHeadDim, kBlockSize>
            <<<grid_dims, 256, 0,
               at::cuda::getCurrentCUDAStream(qkv_input.device().index())
                   .stream()>>>(
                const_cast<T*>(qkv_input.data_ptr<T>()),
                reinterpret_cast<cutlass::float_e4m3_t*>(
                    q_output.data_ptr<fp8>()),
                q_dequant_scale_output.data_ptr<float>(),
                reinterpret_cast<cutlass::float_e4m3_t*>(
                    const_cast<uint8_t*>(cache_k.data_ptr<uint8_t>())),
                reinterpret_cast<cutlass::float_e4m3_t*>(
                    const_cast<uint8_t*>(cache_v.data_ptr<uint8_t>())),
                const_cast<float*>(kv_dequant_scale.value().data_ptr<float>()),
                cu_seq_q.data_ptr<int>(), seq_len_encoder.data_ptr<int>(),
                seq_len_decoder.data_ptr<int>(), rotary_embs.data_ptr<float>(),
                block_tables.data_ptr<int>(),
                q_norm_gamma ? q_norm_gamma.value().data_ptr<float>() : nullptr,
                k_norm_gamma ? k_norm_gamma.value().data_ptr<float>() : nullptr,
                max_blocks_per_seq, head_num, kv_head_num, gqa_group_size,
                max_input_len);
        return {q_output, q_dequant_scale_output};
    }

    grid_dims.x = 1;
    grid_dims.y = head_num + kv_head_num * 2;
    grid_dims.z = batch_size;

    torch::Tensor q_output =
        torch::empty({batch_size, kv_head_num, kBlockM, head_dim},
                     torch::TensorOptions()
                         .dtype(qkv_input.scalar_type())
                         .device(qkv_input.device()));

    write_cachekv_decoder<T, kBlockSize, kThreads, kHeadDim><<<
        grid_dims, kThreads, 0,
        at::cuda::getCurrentCUDAStream(qkv_input.device().index()).stream()>>>(
        const_cast<T*>(qkv_input.data_ptr<T>()), q_output.data_ptr<T>(),
        cu_seq_q.data_ptr<int>(), seq_len_encoder.data_ptr<int>(),
        seq_len_decoder.data_ptr<int>(), rotary_embs.data_ptr<float>(),
        max_input_len, const_cast<T*>(cache_k.data_ptr<T>()),
        const_cast<T*>(cache_v.data_ptr<T>()), block_tables.data_ptr<int>(),
        head_num, kv_head_num, max_blocks_per_seq, gqa_group_size,
        q_norm_gamma ? q_norm_gamma.value().data_ptr<float>() : nullptr,
        k_norm_gamma ? k_norm_gamma.value().data_ptr<float>() : nullptr);

    return {q_output, q_dequant_scale_output};
}

// Computes per-batch token metadata in a single CTA:
//   actual_cu_seq_k: K prefix-sum; prefill samples count decode prefix + new
//                    tokens, decode-only samples count only new tokens.
//   seq_lens_kv:     KV length for decode attention, 0 for non-decode samples.
//   max_token_num:   [max encoder tokens, max decoder KV tokens, total Q,
//                    total K].
template <int kNThreads>
__global__ void get_qk_token_kernel(const int* seq_lens_encoder,
                                    const int* seq_lens_decoder,
                                    const int* cu_seq_q, const int* cu_seq_k,
                                    int* actual_cu_seq_k, int* seq_lens_kv,
                                    int* max_token_num, const int batch_size,
                                    const int kv_head_num,
                                    const int max_batch_size) {
    const int tidx = threadIdx.x;
    int max_enc_token = 0;
    int max_dec_token = 0;
    for (int i = tidx; i < batch_size; i += blockDim.x) {
        max_enc_token = max(max_enc_token, seq_lens_encoder[i]);
        const int q_token = cu_seq_q[i + 1] - cu_seq_q[i];
        if (seq_lens_encoder[i] > 0 || seq_lens_decoder[i] == 0 ||
            q_token <= 0) {
            seq_lens_kv[i] = 0;
        } else {
            seq_lens_kv[i] = seq_lens_decoder[i] + q_token;
        }
        max_dec_token = max(max_dec_token, seq_lens_kv[i]);
    }

    for (int i = tidx + batch_size; i < max_batch_size; i += blockDim.x) {
        seq_lens_kv[i] = 0;
    }

    max_enc_token = BlockAllReduce<int, MaxOp<int>, kNThreads>(max_enc_token);
    max_dec_token = BlockAllReduce<int, MaxOp<int>, kNThreads>(max_dec_token);
    if (tidx == 0) {
        int k_tokens = 0;
        int q_tokens = 0;
        const int kblock_size = 128 * 8;
        const int max_n_block = (max_dec_token + kblock_size - 1) / kblock_size;
        cutlass::FastDivmod kblock_divmod(max_n_block);
        if (max_enc_token > 0 || max_dec_token > 0) {
            for (int i = 0; i < batch_size; i++) {
                actual_cu_seq_k[i] = k_tokens;
                int q_token = cu_seq_q[i + 1] - cu_seq_q[i];
                if (seq_lens_encoder[i] > 0) {
                    q_token = cu_seq_q[i + 1] - cu_seq_q[i];
                    k_tokens += seq_lens_decoder[i] + q_token;
                } else {
                    k_tokens += q_token;
                }
                q_tokens += q_token;
            }
            actual_cu_seq_k[batch_size] = k_tokens;
        }
        max_token_num[0] = max_enc_token;
        max_token_num[1] = max_dec_token;
        max_token_num[2] = q_tokens;
        max_token_num[3] = k_tokens;
    }
}

// Host wrapper for get_qk_token_kernel; cu_seq_k is not read. Returns
// max_token_num copied to CPU, since the host sizes outputs and grids from it.
std::vector<torch::Tensor> GetQKToken(const torch::Tensor& seq_lens_encoder,
                                      const torch::Tensor& seq_lens_decoder,
                                      const torch::Tensor& cu_seq_q,
                                      const torch::Tensor& cu_seq_k,
                                      const torch::Tensor& actual_cu_seq_k,
                                      const torch::Tensor& seq_lens_kv,
                                      const int kv_head_num) {
    const int batch_size = cu_seq_q.size(0) - 1;
    const int max_batch_size = seq_lens_kv.size(0);
    torch::Tensor max_token_num =
        torch::empty({4}, torch::TensorOptions()
                              .dtype(cu_seq_q.scalar_type())
                              .device(cu_seq_q.device()));
    constexpr int kNThreads = 128;
    get_qk_token_kernel<kNThreads>
        <<<1, kNThreads, 0,
           at::cuda::getCurrentCUDAStream(seq_lens_encoder.device().index())
               .stream()>>>(seq_lens_encoder.data_ptr<int>(),
                            seq_lens_decoder.data_ptr<int>(),
                            cu_seq_q.data_ptr<int>(), cu_seq_k.data_ptr<int>(),
                            const_cast<int*>(actual_cu_seq_k.data_ptr<int>()),
                            const_cast<int*>(seq_lens_kv.data_ptr<int>()),
                            max_token_num.data_ptr<int>(), batch_size,
                            kv_head_num, max_batch_size);
    auto max_token_num_cpu = max_token_num.cpu();
    return {max_token_num_cpu};
}

}  // namespace halops
