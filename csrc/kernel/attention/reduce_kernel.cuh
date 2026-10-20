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

#include <algorithm>

#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "kernel/attention/kernel_traits.cuh"
#include "kernel/attention/utils.cuh"

namespace halops {

// Number of row-groups that split a partition list of this length. More groups
// shorten the load chain but cost a larger fold and barrier.
template <int kNumThreadGroups>
__host__ __device__ __forceinline__ constexpr int
attention_reduce_select_groups(const int num_partitions) {
    static_assert(kNumThreadGroups >= 8,
                  "the step sizes below assume the CTA is at least 8 groups "
                  "wide");
    if (num_partitions <= 1) return 1;
    if (num_partitions <= 24) return 4;
    if (num_partitions <= 48) return 8;
    return kNumThreadGroups;
}

// Warps holding active groups (two groups share a warp).
template <int kThreadsPerGroup>
__host__ __device__ __forceinline__ constexpr int attention_reduce_active_warps(
    const int active_groups) {
    constexpr int kGroupsPerWarp = 32 / kThreadsPerGroup;
    return (active_groups + kGroupsPerWarp - 1) / kGroupsPerWarp;
}

// Smem floats for the cross-warp merge: one partial plus one sum per active
// warp past warp 0.
template <int kThreadsPerGroup, int kPairsPerThread>
__host__ __device__ __forceinline__ constexpr int
attention_reduce_partial_smem_floats(const int active_groups) {
    const int active_warps =
        attention_reduce_active_warps<kThreadsPerGroup>(active_groups);
    return (active_warps - 1) * (kThreadsPerGroup * kPairsPerThread * 2 + 1);
}

// Merge and stats staging floats past the 2 * num_partitions scale slots,
// sized for the full CTA width so any per-CTA group count fits.
template <int kThreadsPerGroup, int kNumThreadGroups, int kPairsPerThread>
__host__ __device__ __forceinline__ constexpr int
attention_reduce_max_staging_floats() {
    return attention_reduce_partial_smem_floats<kThreadsPerGroup,
                                                kPairsPerThread>(
               kNumThreadGroups) +
           attention_reduce_active_warps<kThreadsPerGroup>(kNumThreadGroups);
}

// Must match the decoder's per-batch partition count. The BF16 decoder derives
// it from the batch's KV length; FP8 uses a fixed partition width.
template <int kBlockN, int kNumKvBlocksPerPartition,
          bool kUseFixedPartitionWidth>
__device__ __forceinline__ int attention_reduce_num_partitions(
    const int seq_len_kv, const int max_num_partitions) {
    if constexpr (!kUseFixedPartitionWidth) {
        const int num_kv_blocks = (seq_len_kv + kBlockN - 1) / kBlockN;
        const int num_kv_blocks_per_cta =
            max(kNumKvBlocksPerPartition,
                (num_kv_blocks + max_num_partitions - 1) / max_num_partitions);
        return (num_kv_blocks + num_kv_blocks_per_cta - 1) /
               num_kv_blocks_per_cta;
    } else {
        static_assert(kNumKvBlocksPerPartition > 0);
        return (seq_len_kv + kNumKvBlocksPerPartition * kBlockN - 1) /
               (kNumKvBlocksPerPartition * kBlockN);
    }
}

__device__ __forceinline__ int get_max_sum_row_from_logical_row(
    int logical_row) {
    const int row_in_warp = logical_row % 16;
    return logical_row / 16 * 16 + row_in_warp % 8 * 2 + row_in_warp / 8;
}

// partition_o slot of (logical_row, parity) in the row-interleaved layout,
// where rows r and r + 8 alternate: slots 0..7 hold (row, parity) =
// (0,0) (8,0) (0,1) (8,1) (1,0) (9,0) (1,1) (9,1). Bijective onto [0, 128).
__device__ __forceinline__ int get_row_interleaved_slot(int logical_row,
                                                        int parity) {
    const int warp_idx = logical_row / 16;
    const int row_in_warp = logical_row % 16;
    return warp_idx * 32 + (row_in_warp % 8) * 4 + parity * 2 + row_in_warp / 8;
}

// Lane mask of this thread's row group. Scoping shuffles to the group keeps
// them well defined when other groups in the warp have diverged or exited.
template <int kThreadsPerGroup>
__device__ __forceinline__ unsigned int get_group_lane_mask() {
    static_assert(kThreadsPerGroup <= 32 && 32 % kThreadsPerGroup == 0,
                  "a row's group must tile a warp");
    constexpr unsigned int kMask =
        kThreadsPerGroup == 32 ? 0xffffffffU : (1U << kThreadsPerGroup) - 1U;
    return kMask << (threadIdx.x & (32 - kThreadsPerGroup));
}

// A thread holds one head-dim parity of kValsPerThread elements; its partner at
// slot_lane + kThreadsPerSlot holds the other parity. Each thread stores half
// of their contiguous columns, fully coalesced.
template <int kThreadsPerSlot, int kValsPerThread>
__device__ __forceinline__ void store_interleaved_bf16_output(
    __nv_bfloat16 *output, const float2 *values, const int slot_lane,
    const int row_parity, const unsigned int group_mask) {
    constexpr int kPairsPerThread = kValsPerThread / 2;
    constexpr int kHalfPairs = kPairsPerThread / 2;
    static_assert(kPairsPerThread % 2 == 0);

    unsigned int own[kPairsPerThread];
#pragma unroll
    for (int i = 0; i < kPairsPerThread; ++i) {
        const __nv_bfloat162 pair =
            __floats2bfloat162_rn(values[i].x, values[i].y);
        own[i] = static_cast<unsigned int>(__bfloat16_as_ushort(pair.x)) |
                 static_cast<unsigned int>(__bfloat16_as_ushort(pair.y)) << 16;
    }

    unsigned int peer[kPairsPerThread];
#pragma unroll
    for (int i = 0; i < kPairsPerThread; ++i) {
        peer[i] = __shfl_xor_sync(group_mask, own[i], kThreadsPerSlot);
    }

    // Parity 0 stores the low half of the run, parity 1 the high half.
    const unsigned int *even_cols = row_parity == 0 ? own : peer;
    const unsigned int *odd_cols = row_parity == 0 ? peer : own;
    const int pair_base = row_parity * kHalfPairs;

    unsigned int packed[kPairsPerThread];
#pragma unroll
    for (int i = 0; i < kHalfPairs; ++i) {
        const unsigned int even_bits = even_cols[pair_base + i];
        const unsigned int odd_bits = odd_cols[pair_base + i];
        packed[2 * i] = __byte_perm(even_bits, odd_bits, 0x5410);
        packed[2 * i + 1] = __byte_perm(even_bits, odd_bits, 0x7632);
    }

    *reinterpret_cast<uint4 *>(output + slot_lane * 2 * kValsPerThread +
                               row_parity * kValsPerThread) =
        *reinterpret_cast<const uint4 *>(packed);
}

// Reduces per-partition maxima to the row max, broadcast to the block. Short
// lists are scanned serially from smem; longer ones fold within each active
// warp, then across the active_warps per-warp results.
template <int kMaxWarps, int kSerialMaxPartitions>
__device__ __forceinline__ float reduce_partition_max(
    float local_max, const float *smem_partition_max, float *smem_warp_max,
    const int num_partitions, const int lane_idx, const int warp_idx,
    const bool active, const int active_warps) {
    constexpr int kWarpSize = 32;

    float global_max = -FLT_MAX;

    if (num_partitions <= kSerialMaxPartitions) {
        __syncthreads();
        if (lane_idx == 0) {
            int partition_idx = 0;
            for (; partition_idx + 1 < num_partitions; partition_idx += 2) {
                global_max =
                    fmax_3_input(global_max, smem_partition_max[partition_idx],
                                 smem_partition_max[partition_idx + 1]);
            }
            if (partition_idx < num_partitions) {
                global_max =
                    fmaxf(global_max, smem_partition_max[partition_idx]);
            }
        }
        return __shfl_sync(0xffffffffU, global_max, 0);
    }

    if (active) {
#pragma unroll
        for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
            local_max = fmaxf(local_max,
                              __shfl_xor_sync(0xffffffffU, local_max, offset));
        }
        if (lane_idx == 0) {
            smem_warp_max[warp_idx] = local_max;
        }
    }
    __syncthreads();

    if (lane_idx == 0) {
#pragma unroll
        for (int i = 0; i < kMaxWarps; ++i) {
            if (i >= active_warps) break;
            global_max = fmaxf(global_max, smem_warp_max[i]);
        }
    }
    return __shfl_sync(0xffffffffU, global_max, 0);
}

// Computes smem_scale[p] = exp2f(max[p] - global_max) and
// smem_scaled_sum[p] = sum[p] * smem_scale[p]. stats_threads must be a
// block-uniform multiple of 32 so `active` never splits a warp.
template <int kBlockM, int kMaxThreads>
__device__ __forceinline__ void load_partition_stats_and_prepare_scales(
    const float2 *__restrict__ partition_max_sum, float *smem_scale,
    float *smem_scaled_sum, float *smem_warp_max, const int num_partitions,
    const int max_num_partitions, const int batch, const int kv_head_idx,
    const int kv_head_num, const int logical_row, const int stats_threads) {
    constexpr int kWarpSize = 32;
    constexpr int kMaxWarps = kMaxThreads / kWarpSize;

    const int tidx = threadIdx.x;
    const int lane_idx = tidx % kWarpSize;
    const bool active = tidx < stats_threads;
    const int max_sum_row = get_max_sum_row_from_logical_row(logical_row);
    const int stats_partition_stride = kv_head_num * kBlockM;
    const int stats_partition_base =
        (batch * max_num_partitions * kv_head_num + kv_head_idx) * kBlockM +
        max_sum_row;

    float local_max = -FLT_MAX;
    if (active) {
        for (int partition_idx = tidx; partition_idx < num_partitions;
             partition_idx += stats_threads) {
            const int stats_offset =
                stats_partition_base + partition_idx * stats_partition_stride;
            const float2 stats = partition_max_sum[stats_offset];
            smem_scale[partition_idx] = stats.x;
            smem_scaled_sum[partition_idx] = stats.y;
            local_max = fmaxf(local_max, stats.x);
        }
    }

    const float global_max = reduce_partition_max<kMaxWarps, 8>(
        local_max, smem_scale, smem_warp_max, num_partitions, lane_idx,
        tidx / kWarpSize, active, stats_threads / kWarpSize);
    __syncthreads();

    // Read-modify-write: each partition needs exactly one owner, so reuse the
    // guard and stride of the load loop.
    if (active) {
        for (int partition_idx = tidx; partition_idx < num_partitions;
             partition_idx += stats_threads) {
            const float scale = exp2f(smem_scale[partition_idx] - global_max);
            smem_scale[partition_idx] = scale;
            smem_scaled_sum[partition_idx] *= scale;
        }
    }
    __syncthreads();
}

// Group-local variant for when one kThreadsPerGroup-lane group owns the row:
// the max reduction is a shuffle chain within the group, with no block barrier.
template <int kBlockM, int kThreadsPerGroup>
__device__ __forceinline__ void load_partition_stats_and_prepare_scales_local(
    const float2 *__restrict__ partition_max_sum, float *smem_scale,
    float *smem_scaled_sum, const int num_partitions,
    const int max_num_partitions, const int batch, const int kv_head_idx,
    const int kv_head_num, const int logical_row,
    const unsigned int group_mask) {
    const int group_tidx = threadIdx.x % kThreadsPerGroup;
    const int max_sum_row = get_max_sum_row_from_logical_row(logical_row);
    const int stats_partition_stride = kv_head_num * kBlockM;
    const int stats_partition_base =
        (batch * max_num_partitions * kv_head_num + kv_head_idx) * kBlockM +
        max_sum_row;

    float local_max = -FLT_MAX;
    for (int partition_idx = group_tidx; partition_idx < num_partitions;
         partition_idx += kThreadsPerGroup) {
        const int stats_offset =
            stats_partition_base + partition_idx * stats_partition_stride;
        const float2 stats = partition_max_sum[stats_offset];
        smem_scale[partition_idx] = stats.x;
        smem_scaled_sum[partition_idx] = stats.y;
        local_max = fmaxf(local_max, stats.x);
    }

#pragma unroll
    for (int offset = kThreadsPerGroup / 2; offset > 0; offset >>= 1) {
        local_max =
            fmaxf(local_max, __shfl_xor_sync(group_mask, local_max, offset));
    }

    for (int partition_idx = group_tidx; partition_idx < num_partitions;
         partition_idx += kThreadsPerGroup) {
        const float scale = exp2f(smem_scale[partition_idx] - local_max);
        smem_scale[partition_idx] = scale;
        smem_scaled_sum[partition_idx] *= scale;
    }
    __syncwarp(group_mask);
}

// Accumulates scaled partition_o for one row; each thread loads one 16 B vector
// per partition. Groups stride the partition list by active_groups, so the
// partials need merging when active_groups > 1.
template <int kBlockM, int kHeadDim, int kThreadsPerGroup,
          bool kRowInterleavedPartitionO = false>
__device__ __forceinline__ void accumulate_partition_output(
    const __nv_bfloat16 *__restrict__ partition_o,
    const float *__restrict__ smem_scale,
    const float *__restrict__ smem_scaled_sum, const int num_partitions,
    const int max_num_partitions, const int batch, const int kv_head_idx,
    const int kv_head_num, const int logical_row, const int partition_group,
    const int active_groups, float &partial_sum, float2 *partial_acc2) {
    constexpr int kValsPerRow = kHeadDim / 2;
    constexpr int kThreadsPerSlot = kThreadsPerGroup / 2;
    constexpr int kValsPerThread = kValsPerRow / kThreadsPerSlot;
    constexpr int kPairsPerThread = kValsPerThread / 2;
    constexpr int kBlockElements = kBlockM * kHeadDim;
    static_assert(kValsPerThread == 8, "one 16 B vector load per thread");

    if (partition_group >= active_groups) return;

    const int group_tidx = threadIdx.x % kThreadsPerGroup;
    const int row_parity = group_tidx / kThreadsPerSlot;
    const int slot_lane = group_tidx % kThreadsPerSlot;
    // partition_o holds 2 * kBlockM half-row slots; slot element m is head-dim
    // column (parity + 2 * m). Producers differ only in slot order.
    const int partition_o_slot =
        kRowInterleavedPartitionO
            ? get_row_interleaved_slot(logical_row, row_parity)
            : logical_row * 2 + row_parity;
    const int partition_o_stride = kv_head_num * kBlockElements;
    const int partition_o_row_base =
        (batch * max_num_partitions * kv_head_num + kv_head_idx) *
            kBlockElements +
        partition_o_slot * kValsPerRow + slot_lane * kValsPerThread;
    const __nv_bfloat16 *row_ptr = partition_o + partition_o_row_base;

    for (int partition_idx = partition_group; partition_idx < num_partitions;
         partition_idx += active_groups) {
        const float scale = smem_scale[partition_idx];
        const float2 scale2 = {scale, scale};

        partial_sum += smem_scaled_sum[partition_idx];

        const float4 vector_load = *reinterpret_cast<const float4 *>(
            row_ptr + partition_idx * partition_o_stride);
        const __nv_bfloat162 *values =
            reinterpret_cast<const __nv_bfloat162 *>(&vector_load);
#pragma unroll
        for (int i = 0; i < kPairsPerThread; ++i) {
            const float2 values_f32 = __bfloat1622float2(values[i]);
            fma_packed_f32x2(values_f32, scale2, partial_acc2[i],
                             partial_acc2[i]);
        }
    }
}

// Folds active_groups partials into group 0 and returns true on that group.
// Level 1 shuffles each warp's odd group into the even one; all lanes must
// participate, so inactive groups carry zeros. Level 2 stages one partial per
// active warp in smem; active_groups must be block-uniform for its barrier.
template <int kThreadsPerGroup, int kNumThreadGroups, int kPairsPerThread>
__device__ __forceinline__ bool merge_partition_groups(
    float2 *acc2, float &global_sum, float *smem_stats,
    const int num_partitions, const int active_groups) {
    constexpr int kWarpSize = 32;
    constexpr int kGroupsPerWarp = kWarpSize / kThreadsPerGroup;
    constexpr int kMaxWarps = kNumThreadGroups / kGroupsPerWarp;
    constexpr int kPartialsPerWarp = kThreadsPerGroup * kPairsPerThread;
    static_assert(kGroupsPerWarp == 2, "level 1 assumes 2 groups per warp");

    const int tidx = threadIdx.x;
    const int group_tidx = tidx % kThreadsPerGroup;
    const int partition_group = tidx / kThreadsPerGroup;
    const int active_warps =
        attention_reduce_active_warps<kThreadsPerGroup>(active_groups);

#pragma unroll
    for (int i = 0; i < kPairsPerThread; ++i) {
        acc2[i].x += __shfl_down_sync(0xffffffffU, acc2[i].x, kThreadsPerGroup);
        acc2[i].y += __shfl_down_sync(0xffffffffU, acc2[i].y, kThreadsPerGroup);
    }
    global_sum += __shfl_down_sync(0xffffffffU, global_sum, kThreadsPerGroup);

    if (active_warps == 1) {
        return partition_group == 0;
    }

    const int warp_idx = tidx / kWarpSize;
    float2 *smem_partial_o =
        reinterpret_cast<float2 *>(smem_stats + 2 * num_partitions);
    float *smem_partial_sum = reinterpret_cast<float *>(
        smem_partial_o + (active_warps - 1) * kPartialsPerWarp);

    if (warp_idx > 0 && warp_idx < active_warps &&
        partition_group % kGroupsPerWarp == 0) {
        float2 *dst = smem_partial_o + (warp_idx - 1) * kPartialsPerWarp +
                      group_tidx * kPairsPerThread;
#pragma unroll
        for (int i = 0; i < kPairsPerThread; ++i) dst[i] = acc2[i];
        if (group_tidx == 0) {
            smem_partial_sum[warp_idx - 1] = global_sum;
        }
    }
    __syncthreads();

    if (partition_group != 0) return false;

#pragma unroll
    for (int w = 0; w < kMaxWarps - 1; ++w) {
        if (w >= active_warps - 1) break;
        const float2 *out_ptr = smem_partial_o + w * kPartialsPerWarp +
                                group_tidx * kPairsPerThread;
#pragma unroll
        for (int i = 0; i < kPairsPerThread; ++i) {
            acc2[i] = add_packed_f32x2(acc2[i], out_ptr[i]);
        }
        global_sum += smem_partial_sum[w];
    }
    return true;
}

template <int kHeadDim, int kThreadsPerGroup>
__device__ __forceinline__ void normalize_and_store_output(
    const float global_sum, const float2 *acc2, __nv_bfloat16 *output,
    const int q_start, const int logical_row,
    cutlass::FastDivmod const &gqa_divmod, const int kv_head_idx,
    const int head_num, const unsigned int group_mask) {
    constexpr int kValsPerRow = kHeadDim / 2;
    constexpr int kThreadsPerSlot = kThreadsPerGroup / 2;
    constexpr int kValsPerThread = kValsPerRow / kThreadsPerSlot;
    constexpr int kPairsPerThread = kValsPerThread / 2;

    // The store shuffles across the group, so exit on a group-uniform
    // predicate.
    if (__all_sync(group_mask, global_sum <= 0.0f)) return;

    const int group_tidx = threadIdx.x % kThreadsPerGroup;
    const int row_parity = group_tidx / kThreadsPerSlot;
    const int slot_lane = group_tidx % kThreadsPerSlot;

    const int gqa_ratio = gqa_divmod.divisor;
    const int token_idx = gqa_divmod.divide(logical_row);
    const int head_idx =
        kv_head_idx * gqa_ratio + logical_row - token_idx * gqa_ratio;
    const float inv_sum = 1.0f / (global_sum + 0.0000001f);
    const float2 inv_sum2 = {inv_sum, inv_sum};

    float2 normalized[kPairsPerThread];
#pragma unroll
    for (int i = 0; i < kPairsPerThread; ++i) {
        mul_packed_f32x2(acc2[i], inv_sum2, normalized[i]);
    }

    __nv_bfloat16 *out =
        output + ((q_start + token_idx) * head_num + head_idx) * kHeadDim;
    store_interleaved_bf16_output<kThreadsPerSlot, kValsPerThread>(
        out, normalized, slot_lane, row_parity, group_mask);
}

// One output row per CTA, for few rows with many partitions: active_groups
// groups split the row's partition list and merge the partials.
// grid = {kv_head_num, batch_size, s_q * gqa_ratio}.
// kRowInterleavedPartitionO: false = row-major slots (BF16 decoder),
// true = row-interleaved slots (FP8 decoder).
template <int kBlockM, int kHeadDim, int kThreadsPerGroup, int kNumThreadGroups,
          int kBlockN, int kNumKvBlocksPerPartition,
          bool kRowInterleavedPartitionO, bool kUseFixedPartitionWidth = false>
__global__ void __launch_bounds__(kNumThreadGroups *kThreadsPerGroup,
                                  kNumThreadGroups *kThreadsPerGroup == 64 ? 1
                                                                           : 2)
    attention_reduce_single_row_per_cta_kernel(
        const __nv_bfloat16 *__restrict__ partition_o,
        const float2 *__restrict__ partition_max_sum,
        __nv_bfloat16 *__restrict__ output, const int *__restrict__ cu_seq_q,
        const int *__restrict__ seq_lens_kv,
        const int *__restrict__ seq_len_encoder,
        const int *__restrict__ seq_len_decoder, const int max_num_partitions,
        const int kv_head_num, const int head_num,
        const cutlass::FastDivmod gqa_divmod) {
    constexpr int kThreads = kNumThreadGroups * kThreadsPerGroup;
    constexpr int kValsPerRow = kHeadDim / 2;
    constexpr int kThreadsPerSlot = kThreadsPerGroup / 2;
    constexpr int kPairsPerThread = kValsPerRow / kThreadsPerSlot / 2;

    static_assert(kBlockM == 64);
    static_assert(kThreadsPerGroup * 8 == kHeadDim,
                  "a row's group must cover the row in 16 B slices");
    static_assert(kThreads % 32 == 0 && kThreadsPerGroup <= 32,
                  "the CTA must be a whole number of warps of whole groups");

    const int kv_head_idx = blockIdx.x;
    const int batch = blockIdx.y;
    const int logical_row = blockIdx.z;

    const int gqa_ratio = gqa_divmod.divisor;
    const int q_start = cu_seq_q[batch];
    const int num_valid_rows = (cu_seq_q[batch + 1] - q_start) * gqa_ratio;
    const int seq_len_kv = seq_lens_kv[batch];

    // Prefilling samples in a mixed batch have no partitions to merge.
    if (!is_decoder_batch(seq_len_encoder, seq_len_decoder, batch) ||
        logical_row >= num_valid_rows || seq_len_kv <= 0) {
        return;
    }

    const int num_partitions =
        attention_reduce_num_partitions<kBlockN, kNumKvBlocksPerPartition,
                                        kUseFixedPartitionWidth>(
            seq_len_kv, max_num_partitions);

    const int active_groups =
        attention_reduce_select_groups<kNumThreadGroups>(num_partitions);
    // Rounded up to whole warps.
    const int stats_threads =
        attention_reduce_active_warps<kThreadsPerGroup>(active_groups) * 32;

    // Smem: scales, scaled sums, merge staging, per-warp maxima.
    extern __shared__ float smem_stats[];
    float *smem_scale = smem_stats;
    float *smem_scaled_sum = smem_stats + num_partitions;
    float *smem_warp_max =
        smem_stats + 2 * num_partitions +
        attention_reduce_partial_smem_floats<kThreadsPerGroup, kPairsPerThread>(
            active_groups);

    // partition_max_sum: [batch_size, max_num_partitions, kv_head_num, kBlockM]
    // of float2 (max, sum).
    load_partition_stats_and_prepare_scales<kBlockM, kThreads>(
        partition_max_sum, smem_scale, smem_scaled_sum, smem_warp_max,
        num_partitions, max_num_partitions, batch, kv_head_idx, kv_head_num,
        logical_row, stats_threads);

    // partition_o: [batch_size, max_num_partitions, kv_head_num, kBlockM,
    // kHeadDim].
    float global_sum = 0.0f;
    float2 acc2[kPairsPerThread] = {};

    accumulate_partition_output<kBlockM, kHeadDim, kThreadsPerGroup,
                                kRowInterleavedPartitionO>(
        partition_o, smem_scale, smem_scaled_sum, num_partitions,
        max_num_partitions, batch, kv_head_idx, kv_head_num, logical_row,
        /*partition_group=*/threadIdx.x / kThreadsPerGroup, active_groups,
        global_sum, acc2);

    if (!merge_partition_groups<kThreadsPerGroup, kNumThreadGroups,
                                kPairsPerThread>(
            acc2, global_sum, smem_stats, num_partitions, active_groups)) {
        return;
    }

    normalize_and_store_output<kHeadDim, kThreadsPerGroup>(
        global_sum, acc2, output, q_start, logical_row, gqa_divmod, kv_head_idx,
        head_num, get_group_lane_mask<kThreadsPerGroup>());
}

// kRowsPerCta rows per CTA, one per group, for many rows with short partition
// lists. Smem is sliced per row and all syncs are group-scoped.
// grid = {kv_head_num, batch_size, ceil(s_q * gqa_ratio / kRowsPerCta)}.
template <int kBlockM, int kRowsPerCta, int kHeadDim, int kThreadsPerGroup,
          int kBlockN, int kNumKvBlocksPerPartition,
          bool kRowInterleavedPartitionO = false,
          bool kUseFixedPartitionWidth = false>
__global__ void __launch_bounds__(kRowsPerCta *kThreadsPerGroup,
                                  kRowsPerCta *kThreadsPerGroup == 64 ? 1 : 2)
    attention_reduce_multi_rows_per_cta_kernel(
        const __nv_bfloat16 *__restrict__ partition_o,
        const float2 *__restrict__ partition_max_sum,
        __nv_bfloat16 *__restrict__ output, const int *__restrict__ cu_seq_q,
        const int *__restrict__ seq_lens_kv,
        const int *__restrict__ seq_len_encoder,
        const int *__restrict__ seq_len_decoder, const int max_num_partitions,
        const int kv_head_num, const int head_num,
        const cutlass::FastDivmod gqa_divmod) {
    static_assert(kBlockM == 64);
    static_assert(kRowsPerCta > 1);
    static_assert(32 % kThreadsPerGroup == 0,
                  "a row's group must fit inside one warp: the groups of a "
                  "packed CTA walk independent rows with no barrier between "
                  "them, so __syncwarp is all a group ever gets");

    const int kv_head_idx = blockIdx.x;
    const int batch = blockIdx.y;

    // The row test is group-uniform and all syncs are group-scoped, so an early
    // exit never stalls another group.
    const int gqa_ratio = gqa_divmod.divisor;
    const int q_start = cu_seq_q[batch];
    const int num_valid_rows = (cu_seq_q[batch + 1] - q_start) * gqa_ratio;
    const int seq_len_kv = seq_lens_kv[batch];
    const int row_in_cta = threadIdx.x / kThreadsPerGroup;
    const int logical_row = blockIdx.z * kRowsPerCta + row_in_cta;

    // Prefilling samples in a mixed batch have no partitions to merge.
    if (!is_decoder_batch(seq_len_encoder, seq_len_decoder, batch) ||
        logical_row >= num_valid_rows || seq_len_kv <= 0) {
        return;
    }

    const int num_partitions =
        attention_reduce_num_partitions<kBlockN, kNumKvBlocksPerPartition,
                                        kUseFixedPartitionWidth>(
            seq_len_kv, max_num_partitions);

    extern __shared__ float smem_stats[];
    float *smem_row = smem_stats + row_in_cta * 2 * num_partitions;
    float *smem_scale = smem_row;
    float *smem_scaled_sum = smem_row + num_partitions;

    const unsigned int group_mask = get_group_lane_mask<kThreadsPerGroup>();

    load_partition_stats_and_prepare_scales_local<kBlockM, kThreadsPerGroup>(
        partition_max_sum, smem_scale, smem_scaled_sum, num_partitions,
        max_num_partitions, batch, kv_head_idx, kv_head_num, logical_row,
        group_mask);

    // The group owns the whole row, so no merge is needed.
    constexpr int kPairsPerThread = kHeadDim / 2 / (kThreadsPerGroup / 2) / 2;
    float global_sum = 0.0f;
    float2 acc2[kPairsPerThread] = {};

    accumulate_partition_output<kBlockM, kHeadDim, kThreadsPerGroup,
                                kRowInterleavedPartitionO>(
        partition_o, smem_scale, smem_scaled_sum, num_partitions,
        max_num_partitions, batch, kv_head_idx, kv_head_num, logical_row,
        /*partition_group=*/0, /*active_groups=*/1, global_sum, acc2);

    normalize_and_store_output<kHeadDim, kThreadsPerGroup>(
        global_sum, acc2, output, q_start, logical_row, gqa_divmod, kv_head_idx,
        head_num, group_mask);
}

template <int kBlockM, int kHeadDim, int kThreadsPerGroup, int kBlockN,
          int kNumKvBlocksPerPartition, bool kRowInterleavedPartitionO,
          bool kUseFixedPartitionWidth = false>
void LaunchAttentionReduceSingleRowPerCta(AttentionParams &params,
                                          const int mtp_steps,
                                          cudaStream_t stream) {
    // CTA width in row-groups; upper bound for the device-side group count.
    constexpr int kNumThreadGroups = 16;
    constexpr int kThreads = kNumThreadGroups * kThreadsPerGroup;
    constexpr int kPairsPerThread = kHeadDim / 2 / (kThreadsPerGroup / 2) / 2;
    const int active_rows =
        (mtp_steps + 1) * (params.head_num / params.kv_head_num);

    dim3 grid(params.kv_head_num, params.batch_size, active_rows);
    dim3 block(kThreads);

    // Sized for the widest group count, independent of KV length, so the launch
    // stays identical across shapes for CUDA graph capture.
    const int stats_smem_bytes = params.max_num_partitions * 2 * sizeof(float);
    const int staging_smem_bytes =
        attention_reduce_max_staging_floats<kThreadsPerGroup, kNumThreadGroups,
                                            kPairsPerThread>() *
        sizeof(float);

    attention_reduce_single_row_per_cta_kernel<
        kBlockM, kHeadDim, kThreadsPerGroup, kNumThreadGroups, kBlockN,
        kNumKvBlocksPerPartition, kRowInterleavedPartitionO,
        kUseFixedPartitionWidth>
        <<<grid, block, stats_smem_bytes + staging_smem_bytes, stream>>>(
            reinterpret_cast<const __nv_bfloat16 *>(params.partition_o_ptr),
            params.max_sum, reinterpret_cast<__nv_bfloat16 *>(params.o_ptr),
            params.cu_seq_q, params.cu_seq_k, params.seq_len_encoder,
            params.seq_len_decoder, params.max_num_partitions,
            params.kv_head_num, params.head_num,
            cutlass::FastDivmod(params.head_num / params.kv_head_num));
}

template <int kBlockM, int kRowsPerCta, int kHeadDim, int kThreadsPerGroup,
          int kBlockN, int kNumKvBlocksPerPartition,
          bool kRowInterleavedPartitionO = false,
          bool kUseFixedPartitionWidth = false>
void LaunchAttentionReduceMultiRowsPerCta(AttentionParams &params,
                                          const int mtp_steps,
                                          cudaStream_t stream) {
    const int active_rows =
        (mtp_steps + 1) * (params.head_num / params.kv_head_num);

    dim3 grid(params.kv_head_num, params.batch_size,
              (active_rows + kRowsPerCta - 1) / kRowsPerCta);
    dim3 block(kRowsPerCta * kThreadsPerGroup);

    // One (scale, scaled_sum) slice per row, bounded by max_num_partitions.
    const int stats_smem_bytes =
        kRowsPerCta * params.max_num_partitions * 2 * sizeof(float);

    attention_reduce_multi_rows_per_cta_kernel<
        kBlockM, kRowsPerCta, kHeadDim, kThreadsPerGroup, kBlockN,
        kNumKvBlocksPerPartition, kRowInterleavedPartitionO,
        kUseFixedPartitionWidth><<<grid, block, stats_smem_bytes, stream>>>(
        reinterpret_cast<const __nv_bfloat16 *>(params.partition_o_ptr),
        params.max_sum, reinterpret_cast<__nv_bfloat16 *>(params.o_ptr),
        params.cu_seq_q, params.cu_seq_k, params.seq_len_encoder,
        params.seq_len_decoder, params.max_num_partitions, params.kv_head_num,
        params.head_num,
        cutlass::FastDivmod(params.head_num / params.kv_head_num));
}

static int GetAttentionReduceSmCount() {
    static int cached = 0;
    if (cached == 0) {
        int dev;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&cached, cudaDevAttrMultiProcessorCount, dev);
    }
    return cached;
}

// Rows packed per CTA (8, 4 or 2; 1 selects one-row-per-CTA). Pack only while
// at least 2 CTAs per SM remain, rows fit the blockIdx.z extent, and stats
// smem stays within 32 KB (48 KB default dynamic smem limit).
static int SelectAttentionReduceRowsPerCta(const int total_rows,
                                           const int active_rows,
                                           const int max_num_partitions) {
    constexpr int kMinCtasPerSm = 2;
    constexpr int kMaxStatsSmemBytes = 32 * 1024;
    const int sm_count = GetAttentionReduceSmCount();
    const int row_stats_bytes = max_num_partitions * 2 * sizeof(float);

    for (int rows_per_cta = 8; rows_per_cta >= 2; rows_per_cta >>= 1) {
        if (rows_per_cta > active_rows) continue;
        if (total_rows < kMinCtasPerSm * sm_count * rows_per_cta) continue;
        if (rows_per_cta * row_stats_bytes > kMaxStatsSmemBytes) continue;
        return rows_per_cta;
    }
    return 1;
}

template <int kBlockM, int kHeadDim, int kBlockN, int kNumKvBlocksPerPartition,
          bool kRowInterleavedPartitionO = false,
          bool kUseFixedPartitionWidth = false>
void AttentionReduce(AttentionParams &params, const int mtp_steps,
                     cudaStream_t stream) {
    // 8 threads per head-dim parity, each moving 8 elements as one 16 B vector.
    constexpr int kThreadsPerGroup = 16;

    if (params.batch_size == 0) return;

    const int active_rows =
        (mtp_steps + 1) * (params.head_num / params.kv_head_num);
    const int total_rows = params.kv_head_num * params.batch_size * active_rows;
    const int rows_per_cta = SelectAttentionReduceRowsPerCta(
        total_rows, active_rows, params.max_num_partitions);

    if (rows_per_cta >= 8) {
        LaunchAttentionReduceMultiRowsPerCta<
            kBlockM, /*kRowsPerCta=*/8, kHeadDim, kThreadsPerGroup, kBlockN,
            kNumKvBlocksPerPartition, kRowInterleavedPartitionO,
            kUseFixedPartitionWidth>(params, mtp_steps, stream);
        return;
    }
    if (rows_per_cta >= 4) {
        LaunchAttentionReduceMultiRowsPerCta<
            kBlockM, /*kRowsPerCta=*/4, kHeadDim, kThreadsPerGroup, kBlockN,
            kNumKvBlocksPerPartition, kRowInterleavedPartitionO,
            kUseFixedPartitionWidth>(params, mtp_steps, stream);
        return;
    }
    if (rows_per_cta >= 2) {
        LaunchAttentionReduceMultiRowsPerCta<
            kBlockM, /*kRowsPerCta=*/2, kHeadDim, kThreadsPerGroup, kBlockN,
            kNumKvBlocksPerPartition, kRowInterleavedPartitionO,
            kUseFixedPartitionWidth>(params, mtp_steps, stream);
        return;
    }

    LaunchAttentionReduceSingleRowPerCta<
        kBlockM, kHeadDim, kThreadsPerGroup, kBlockN, kNumKvBlocksPerPartition,
        kRowInterleavedPartitionO, kUseFixedPartitionWidth>(params, mtp_steps,
                                                            stream);
}

}  // namespace halops
