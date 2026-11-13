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

#include <cfloat>
#include <cstdint>

#include "kernel/moe/moe_configs.cuh"
#include "kernel/moe/moe_shared_storage.cuh"
#include "kernel/moe/moe_traits.cuh"
#include "kernel/moe/moe_workspace.cuh"
#include "kernel/moe/ptx_utils.cuh"
#include "kernel/moe/quant_utils.cuh"

namespace halops::moe {

// One warp warms the selection's routing vectors in L2.
template <typename Dims>
__device__ static void prefetch_routing_vectors(
    const float* __restrict__ correction_bias,
    const float* __restrict__ per_expert_scale) {
    constexpr std::uint32_t kLineBytes = 128u;
    constexpr std::uint32_t kLines =
        (Dims::kNumExperts * sizeof(float) + kLineBytes - 1u) / kLineBytes;
    static_assert(2u * kLines <= WarpRole::kThreadsPerWarp,
                  "one lane per L2 line of the two vectors");
    const std::uint32_t lane_idx = WarpRole::get_lane_idx();
    const float* const vec =
        lane_idx < kLines ? correction_bias : per_expert_scale;
    if (vec != nullptr && lane_idx < 2u * kLines) {
        prefetch_l2_line(reinterpret_cast<const char*>(vec) +
                         (lane_idx % kLines) * kLineBytes);
    }
}

// The lane's slice of `correction_bias`, or zeros without one.
template <std::uint32_t kExpertsPerThread>
__device__ static cutlass::Array<float, kExpertsPerThread> load_correction_bias(
    const float* __restrict__ correction_bias) {
    cutlass::Array<float, kExpertsPerThread> bias;
    const std::uint32_t lane_base =
        WarpRole::get_lane_idx() * kExpertsPerThread;
#pragma unroll
    for (std::uint32_t i = 0; i < kExpertsPerThread; ++i) {
        bias[i] = correction_bias != nullptr
                      ? __ldg(correction_bias + lane_base + i)
                      : 0.0f;
    }
    return bias;
}

// Warp `token` reads the token's logits row, lane l its expert slice.
template <typename Dims, std::uint32_t kExpertsPerThread>
__device__ static cutlass::Array<float, kExpertsPerThread>
load_logits_one_token(const MoEWorkspace<Dims>* __restrict__ workspace,
                      std::uint32_t gate, std::uint32_t token_idx) {
    using RouterGemm = RouterGemmTraits<Dims>;
    static_assert(kExpertsPerThread % 8u == 0u,
                  "a lane's logits slice must be whole 16-B words");

    cutlass::Array<float, kExpertsPerThread> slice;
    const std::uint32_t lane_base =
        WarpRole::get_lane_idx() * kExpertsPerThread;

    if constexpr (MoEWorkspace<Dims>::kRouterStoreScores) {
        const float4* src = reinterpret_cast<const float4*>(
            workspace->router_logits.row(gate, token_idx) + lane_base);
#pragma unroll
        for (std::uint32_t v = 0; v < kExpertsPerThread / 4u; ++v) {
            const float4 raw = src[v];
            slice[4u * v + 0u] = raw.x;
            slice[4u * v + 1u] = raw.y;
            slice[4u * v + 2u] = raw.z;
            slice[4u * v + 3u] = raw.w;
        }
    } else {
        const uint4* src = reinterpret_cast<const uint4*>(
            workspace->router_logits.row(gate, token_idx) + lane_base);
#pragma unroll
        for (std::uint32_t v = 0; v < kExpertsPerThread / 8u; ++v) {
            const uint4 raw = src[v];
            const std::uint32_t word[4] = {raw.x, raw.y, raw.z, raw.w};
#pragma unroll
            for (std::uint32_t j = 0; j < 4u; ++j) {
                slice[8u * v + 2u * j + 0u] = __bfloat162float(
                    __ushort_as_bfloat16(static_cast<unsigned short>(word[j])));
                slice[8u * v + 2u * j + 1u] =
                    __bfloat162float(__ushort_as_bfloat16(
                        static_cast<unsigned short>(word[j] >> 16)));
            }
        }
    }
    return slice;
}

// Warp-cooperative softmax over logits distributed across threads.
template <uint32_t Count>
__device__ static void warp_softmax_inplace(float (&logits)[Count]) {
    float local_max = -FLT_MAX;
#pragma unroll
    for (uint32_t i = 0; i < Count; i++)
        local_max = fmaxf(local_max, logits[i]);
    float global_max = warp_reduce_max_float(local_max);

    float local_sum = 0.0f;
#pragma unroll
    for (uint32_t i = 0; i < Count; i++) {
        logits[i] = __expf(logits[i] - global_max);
        local_sum += logits[i];
    }
    for (int off = 16; off >= 1; off /= 2)
        local_sum += __shfl_xor_sync(0xFFFFFFFFu, local_sum, off, 32);

    float inv_sum = 1.0f / local_sum;
#pragma unroll
    for (uint32_t i = 0; i < Count; i++) {
        logits[i] *= inv_sum;
    }
}

// Warp-wide "max score, lowest expert id on ties", on ALL lanes.
__device__ static __forceinline__ void warp_argmax_expert(
    float my_score, uint32_t my_expert, float& warp_score,
    uint32_t& warp_expert) {
    const uint32_t bits = __float_as_uint(my_score);
    const uint32_t moe = (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
    const uint32_t best = __reduce_max_sync(0xFFFFFFFFu, moe);
    warp_expert =
        __reduce_min_sync(0xFFFFFFFFu, (moe == best) ? my_expert : 0xFFFFFFFFu);
    const uint32_t best_bits =
        (best & 0x80000000u) ? (best & 0x7FFFFFFFu) : ~best;
    warp_score = __uint_as_float(best_bits);
}

// Turns the lane's logits into fp32 candidates; NaN/Inf become -FLT_MAX.
template <uint32_t kExpertsPerThread>
__device__ static void fp32_candidates(
    uint32_t lane_idx, const cutlass::Array<float, kExpertsPerThread>& logits,
    float (&scores)[kExpertsPerThread],
    uint32_t (&expert_id)[kExpertsPerThread]) {
    const uint32_t lane_base = lane_idx * kExpertsPerThread;
#pragma unroll
    for (uint32_t i = 0; i < kExpertsPerThread; ++i) {
        const float logit = logits[i];
        scores[i] = (isnan(logit) || isinf(logit)) ? -FLT_MAX : logit;
        expert_id[i] = lane_base + i;
    }
}

// Pack a BF16 logit into an ordered score.
__device__ static __forceinline__ uint32_t pack_logit_bits(uint32_t bits,
                                                           uint32_t expert_id) {
    // NaN/Inf -> the lowest finite score.
    if ((bits & 0x7F80u) == 0x7F80u) bits = 0xFF7Fu;
    // Normalize signed zero before ordering.
    if ((bits & 0x7FFFu) == 0u) bits = 0u;
    const uint32_t moe = (bits & 0x8000u) ? (bits ^ 0xFFFFu) : (bits | 0x8000u);
    return (moe << 16) | (expert_id ^ 0xFFFFu);
}

// Turns the lane's logits into packed scores, one per owned expert.
template <uint32_t kExpertsPerThread>
__device__ static __forceinline__ void packed_candidates(
    uint32_t lane_idx, const cutlass::Array<float, kExpertsPerThread>& logits,
    uint32_t (&packed_scores)[kExpertsPerThread]) {
    const uint32_t lane_base = lane_idx * kExpertsPerThread;
#pragma unroll
    for (uint32_t i = 0; i < kExpertsPerThread; ++i) {
        const uint32_t bits = static_cast<uint32_t>(
            __bfloat16_as_ushort(__float2bfloat16_rn(logits[i])));
        packed_scores[i] = pack_logit_bits(bits, lane_base + i);
    }
}

// Picks the best remaining candidate, broadcasts it, and masks the winner.
template <uint32_t kExpertsPerThread>
__device__ static void select_next_candidate(
    float (&scores)[kExpertsPerThread],
    const uint32_t (&expert_id)[kExpertsPerThread], float& selected_score,
    uint32_t& selected_expert) {
    constexpr float kExcludedScore = -INFINITY;

    float max_val = kExcludedScore;
    uint32_t max_expert = 0xFFFFFFFFu;
#pragma unroll
    for (uint32_t i = 0; i < kExpertsPerThread; i++) {
        if (scores[i] > max_val ||
            (scores[i] == max_val && expert_id[i] < max_expert)) {
            max_val = scores[i];
            max_expert = expert_id[i];
        }
    }
    warp_argmax_expert(max_val, max_expert, selected_score, selected_expert);
#pragma unroll
    for (uint32_t i = 0; i < kExpertsPerThread; i++) {
        if (expert_id[i] == selected_expert) {
            scores[i] = kExcludedScore;
        }
    }
}

// Batcher's odd-even merge sort as compare-exchange pairs, in network order.
template <uint32_t N>
constexpr uint32_t merge_sort_pairs(uint8_t* lo, uint8_t* hi) {
    uint32_t n = 0;
    for (uint32_t p = 1; p < N; p <<= 1) {
        for (uint32_t k = p; k >= 1; k >>= 1) {
            for (uint32_t j = k % p; j + k < N; j += 2 * k) {
                for (uint32_t i = 0; i < k && i + j + k < N; ++i) {
                    if ((i + j) / (2 * p) == (i + j + k) / (2 * p)) {
                        if (lo != nullptr) {
                            lo[n] = static_cast<uint8_t>(i + j);
                            hi[n] = static_cast<uint8_t>(i + j + k);
                        }
                        ++n;
                    }
                }
            }
        }
    }
    return n;
}

template <uint32_t N>
inline constexpr uint32_t kMergeSortSize =
    merge_sort_pairs<N>(nullptr, nullptr);

template <uint32_t N>
struct MergeSortPairs {
    uint8_t lo[kMergeSortSize<N> + 1u];
    uint8_t hi[kMergeSortSize<N> + 1u];
};

template <uint32_t N>
constexpr MergeSortPairs<N> make_merge_sort_pairs() {
    MergeSortPairs<N> pairs = {};
    merge_sort_pairs<N>(pairs.lo, pairs.hi);
    return pairs;
}

template <uint32_t N>
inline constexpr MergeSortPairs<N> kMergeSortPairs = make_merge_sort_pairs<N>();

// Sorts the N keys into descending order with a fixed min/max network.
template <uint32_t N, uint32_t C = 0>
__device__ static void sort_descending(uint32_t (&keys)[N]) {
    static_assert((N & (N - 1u)) == 0u, "the network is built for 2^m keys");
    if constexpr (C < kMergeSortSize<N>) {
        constexpr uint32_t lo = kMergeSortPairs<N>.lo[C];
        constexpr uint32_t hi = kMergeSortPairs<N>.hi[C];
        const uint32_t a = keys[lo];
        const uint32_t b = keys[hi];
        keys[lo] = max(a, b);
        keys[hi] = min(a, b);
        sort_descending<N, C + 1u>(keys);
    }
}

// Select the highest packed score and remove it from the winner's list.
template <uint32_t N, uint32_t kKeep>
__device__ static __forceinline__ void select_next_packed_score(
    uint32_t (&best)[N], uint32_t& selected_packed_score) {
    selected_packed_score = __reduce_max_sync(0xFFFFFFFFu, best[0]);
    const bool won = best[0] == selected_packed_score;
#pragma unroll
    for (uint32_t i = 0; i + 1u < kKeep; i++)
        best[i] = won ? best[i + 1u] : best[i];
    best[kKeep - 1u] = won ? 0u : best[kKeep - 1u];
}

// Splits a packed score back into its logit and expert id.
__device__ static __forceinline__ void unpack_packed_score(
    uint32_t packed_score, float& logit, uint32_t& expert_id) {
    const uint32_t moe = packed_score >> 16;
    const uint32_t bits = (moe & 0x8000u) ? (moe & 0x7FFFu) : (moe ^ 0xFFFFu);
    logit = __bfloat162float(__ushort_as_bfloat16((uint16_t)bits));
    expert_id = (packed_score ^ 0xFFFFu) & 0xFFFFu;
}

// Unnormalized softmax weights use full-row probabilities.
template <uint32_t kExpertsPerThread, uint32_t kMaxTopk>
__device__ static void topk_weights_from_softmax_probs(
    uint32_t token, uint32_t top_k, float (&scores)[kExpertsPerThread],
    const uint32_t (&expert_id)[kExpertsPerThread], uint16_t* topk_ids,
    float* topk_weights) {
    const uint32_t lane_idx = WarpRole::get_lane_idx();

    warp_softmax_inplace<kExpertsPerThread>(scores);
    for (uint32_t k = 0; k < top_k; k++) {
        float selected_score;
        uint32_t selected_expert;
        select_next_candidate<kExpertsPerThread>(
            scores, expert_id, selected_score, selected_expert);
        if (lane_idx == 0u) {
            topk_ids[token * kMaxTopk + k] = (uint16_t)selected_expert;
            topk_weights[token * kMaxTopk + k] = selected_score;
        }
    }
}

// Sum lanes [0, top_k) in rank order.
template <uint32_t kMaxTopk>
__device__ static __forceinline__ float sum_over_ranks(float v,
                                                       uint32_t top_k) {
    float sum = 0.0f;
#pragma unroll
    for (uint32_t k = 0; k < kMaxTopk; k++) {
        const float rank_v = __shfl_sync(0xFFFFFFFFu, v, k);
        if (k < top_k) sum += rank_v;
    }
    return sum;
}

// Select packed scores for renormalized softmax or sigmoid. Lane k holds rank
// k.
template <uint32_t kExpertsPerThread, uint32_t kMaxTopk>
__device__ static void topk_weights_from_logits(
    uint32_t token, uint32_t top_k, ScoringFunc scoring_func, bool renormalize,
    uint32_t (&packed_scores)[kExpertsPerThread], uint16_t* topk_ids,
    float* topk_weights) {
    static_assert(kMaxTopk <= 32u, "one lane per rank");
    const uint32_t lane_idx = WarpRole::get_lane_idx();

    // A lane wins at most top_k rounds, so kKeep best candidates suffice.
    constexpr uint32_t kKeep =
        kExpertsPerThread < kMaxTopk ? kExpertsPerThread : kMaxTopk;
    sort_descending<kExpertsPerThread>(packed_scores);

    uint32_t my_packed = 0u;
    uint32_t best_packed = 0u;
    for (uint32_t k = 0; k < top_k; k++) {
        uint32_t selected_packed_score;
        select_next_packed_score<kExpertsPerThread, kKeep>(
            packed_scores, selected_packed_score);
        if (lane_idx == k) my_packed = selected_packed_score;
        if (k == 0u) best_packed = selected_packed_score;
    }

    float logit;
    uint32_t expert;
    unpack_packed_score(my_packed, logit, expert);

    // Apply the activation; the renormalizer folds into the write below.
    float weight;
    float inv = 1.0f;
    if (scoring_func == ScoringFunc::kSoftmax) {
        float row_max;
        uint32_t row_max_expert;
        unpack_packed_score(best_packed, row_max, row_max_expert);
        weight = __expf(logit - row_max);
        const float sum_exp = sum_over_ranks<kMaxTopk>(weight, top_k);
        inv = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 1.0f;
    } else {
        weight = 1.0f / (1.0f + __expf(-logit));
        if (renormalize) {
            const float sum_sig = sum_over_ranks<kMaxTopk>(weight, top_k);
            inv = (sum_sig > 0.0f) ? (1.0f / sum_sig) : 1.0f;
        }
    }

    if (lane_idx < top_k) {
        topk_ids[token * kMaxTopk + lane_idx] = (uint16_t)expert;
        topk_weights[token * kMaxTopk + lane_idx] = weight * inv;
    }
}

// An order-preserving uint32 image of a float: a larger float, a larger key.
__device__ static __forceinline__ uint32_t score_key(float score) {
    const uint32_t bits = __float_as_uint(score);
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

__device__ static __forceinline__ float key_score(uint32_t key) {
    return __uint_as_float((key & 0x80000000u) ? (key & 0x7FFFFFFFu) : ~key);
}

// The routed scaling of `expert`: 1 without a per-expert scale.
__device__ static __forceinline__ float expert_scale(
    const float* __restrict__ per_expert_scale, uint32_t expert) {
    return per_expert_scale != nullptr ? __ldg(per_expert_scale + expert)
                                       : 1.0f;
}

// Scan each lane's remaining scores; ties select the lowest expert ID.
template <uint32_t kExpertsPerThread, uint32_t kMaxTopk>
__device__ static void topk_weights_from_scores_fp32_scan(
    uint32_t token, uint32_t top_k, bool renormalize,
    const float* __restrict__ per_expert_scale,
    const float (&scores)[kExpertsPerThread],
    const float (&weights)[kExpertsPerThread], uint16_t* topk_ids,
    float* topk_weights) {
    static_assert((kExpertsPerThread & (kExpertsPerThread - 1u)) == 0u,
                  "a lane's best is the root of a binary tree");
    const uint32_t lane_idx = WarpRole::get_lane_idx();
    // Key 0 is below every score's key: it marks a taken expert.
    uint32_t key[kExpertsPerThread];
#pragma unroll
    for (uint32_t i = 0; i < kExpertsPerThread; ++i) {
        key[i] = score_key(scores[i]);
    }

    uint16_t* const ids = topk_ids + token * kMaxTopk;
    float my_score = 0.0f;
    uint32_t my_expert = 0u;
#pragma unroll 1
    for (uint32_t k = 0; k < top_k; k++) {
        // The lane's best, ties to the lower index: a tree whose ties go left.
        uint32_t best[kExpertsPerThread];
        uint32_t at[kExpertsPerThread];
        float best_weight[kExpertsPerThread];
#pragma unroll
        for (uint32_t i = 0; i < kExpertsPerThread; ++i) {
            best[i] = key[i];
            at[i] = i;
            best_weight[i] = weights[i];
        }
#pragma unroll
        for (uint32_t w = 1; w < kExpertsPerThread; w *= 2) {
#pragma unroll
            for (uint32_t i = 0; i < kExpertsPerThread; i += 2 * w) {
                const bool right = best[i + w] > best[i];
                best[i] = right ? best[i + w] : best[i];
                at[i] = right ? at[i + w] : at[i];
                best_weight[i] = right ? best_weight[i + w] : best_weight[i];
            }
        }
        const uint32_t top = __reduce_max_sync(0xFFFFFFFFu, best[0]);
        const uint32_t winner =
            __ffs(__ballot_sync(0xFFFFFFFFu, best[0] == top)) - 1u;
        const uint32_t expert = lane_idx * kExpertsPerThread + at[0];
        const float winner_weight =
            __shfl_sync(0xFFFFFFFFu, best_weight[0], winner);
        const uint32_t winner_expert = __shfl_sync(0xFFFFFFFFu, expert, winner);
        if (lane_idx == k) {
            my_score = winner_weight;
            my_expert = winner_expert;
        }
        // The winning lane records and removes its expert.
        const bool won = lane_idx == winner;
        if (won) {
            ids[k] =
                static_cast<uint16_t>(lane_idx * kExpertsPerThread + at[0]);
        }
#pragma unroll
        for (uint32_t i = 0; i < kExpertsPerThread; ++i) {
            key[i] = (won && i == at[0]) ? 0u : key[i];
        }
    }

    // The ranks sit one per lane, so the sum runs in rank order.
    float inv = 1.0f;
    if (renormalize) {
        const float sum = sum_over_ranks<kMaxTopk>(my_score, top_k);
        inv = (sum > 0.0f) ? (1.0f / sum) : 1.0f;
    }
    if (lane_idx < top_k) {
        topk_weights[token * kMaxTopk + lane_idx] =
            my_score * inv * expert_scale(per_expert_scale, my_expert);
    }
}

// Select sorted FP32 scores, breaking ties by expert ID. Repeated lane-local
// keys use the scan variant; renormalization uses the selected sum.
template <uint32_t kExpertsPerThread, uint32_t kMaxTopk>
__device__ static void topk_weights_from_scores_fp32(
    uint32_t token, uint32_t top_k, bool renormalize,
    const float* __restrict__ per_expert_scale,
    const float (&scores)[kExpertsPerThread],
    const float (&weights)[kExpertsPerThread], uint16_t* topk_ids,
    float* topk_weights) {
    // A lane wins at most top_k rounds, so kKeep heads suffice.
    constexpr uint32_t kKeep =
        kExpertsPerThread < kMaxTopk ? kExpertsPerThread : kMaxTopk;
    const uint32_t lane_idx = WarpRole::get_lane_idx();
    uint32_t key[kExpertsPerThread];
    uint32_t head[kExpertsPerThread];
#pragma unroll
    for (uint32_t i = 0; i < kExpertsPerThread; ++i) {
        key[i] = score_key(scores[i]);
        head[i] = key[i];
    }
    sort_descending<kExpertsPerThread>(head);
    bool repeated = false;
#pragma unroll
    for (uint32_t i = 0; i + 1u < kExpertsPerThread; ++i) {
        repeated |= head[i] == head[i + 1u];
    }
    if (__any_sync(0xFFFFFFFFu, repeated)) {
        topk_weights_from_scores_fp32_scan<kExpertsPerThread, kMaxTopk>(
            token, top_k, renormalize, per_expert_scale, scores, weights,
            topk_ids, topk_weights);
        return;
    }

    // Lane k keeps rank k's key and lane.
    uint32_t lanes_below;
    asm("mov.u32 %0, %%lanemask_lt;" : "=r"(lanes_below));
    uint32_t my_key = 0u;
    uint32_t my_lane = 0u;
#pragma unroll 1
    for (uint32_t k = 0; k < top_k; k++) {
        const uint32_t top = __reduce_max_sync(0xFFFFFFFFu, head[0]);
        const bool tied = head[0] == top;
        const uint32_t ties = __ballot_sync(0xFFFFFFFFu, tied);
        // The lowest tied lane wins.
        const bool won = tied && (ties & lanes_below) == 0u;
        if (lane_idx == k) {
            my_key = top;
            my_lane = __ffs(ties) - 1u;
        }
#pragma unroll
        for (uint32_t i = 0; i + 1u < kKeep; ++i) {
            head[i] = won ? head[i + 1u] : head[i];
        }
        head[kKeep - 1u] = won ? 0u : head[kKeep - 1u];
    }

    // Rank k's expert: the one slot of its key in its lane.
    uint32_t at = 0u;
#pragma unroll
    for (uint32_t i = 0; i < kExpertsPerThread; ++i) {
        at = __shfl_sync(0xFFFFFFFFu, key[i], my_lane) == my_key ? i : at;
    }

    float weight = 0.0f;
#pragma unroll
    for (uint32_t i = 0; i < kExpertsPerThread; ++i) {
        const float w = __shfl_sync(0xFFFFFFFFu, weights[i], my_lane);
        weight = at == i ? w : weight;
    }
    const float my_score = lane_idx < top_k ? weight : 0.0f;
    const uint32_t my_expert = my_lane * kExpertsPerThread + at;
    float inv = 1.0f;
    if (renormalize) {
        const float sum = sum_over_ranks<kMaxTopk>(my_score, top_k);
        inv = (sum > 0.0f) ? (1.0f / sum) : 1.0f;
    }
    if (lane_idx < top_k) {
        topk_ids[token * kMaxTopk + lane_idx] =
            static_cast<uint16_t>(my_expert);
        topk_weights[token * kMaxTopk + lane_idx] =
            my_score * inv * expert_scale(per_expert_scale, my_expert);
    }
}

// Top-K expert selection for one token, one warp per token.
template <typename Dims, uint32_t kExpertsPerThread>
__device__ static void topk_one_token(
    uint32_t token_idx, uint32_t top_k, ScoringFunc scoring_func,
    bool renormalize,
    const cutlass::Array<float, kExpertsPerThread>& correction_bias,
    const float* __restrict__ per_expert_scale,
    const MoEWorkspace<Dims>* __restrict__ workspace,
    MoESharedStorage<Dims>* smem) {
    static_assert(
        Dims::kNumTokens * Dims::kNumExperts < UINT32_MAX,
        "Batch size or number of experts too high for uint32 indices.");
    static_assert(Dims::kNumExperts % 32u == 0u,
                  "topK requires kNumExperts to be a multiple of 32 so "
                  "every thread owns exactly kNumExperts/32 experts (keeps the "
                  "per-thread arrays register-resident).");

    constexpr uint32_t kMaxTopk = Dims::kMaxTopk;
    constexpr uint32_t kNumGates = RouterGemmTraits<Dims>::kNumRouterGates;
    const uint32_t lane_idx = WarpRole::get_lane_idx();

    const cutlass::Array<float, kExpertsPerThread> logits =
        load_logits_one_token<Dims, kExpertsPerThread>(workspace, 0u,
                                                       token_idx);
    if constexpr (kNumGates > 1u) {
        // DualRouter: select on the sum of the two gates' scores.
        const cutlass::Array<float, kExpertsPerThread> other =
            load_logits_one_token<Dims, kExpertsPerThread>(workspace, 1u,
                                                           token_idx);
        float hi[kExpertsPerThread];
        float lo[kExpertsPerThread];
        uint32_t expert_id[kExpertsPerThread];
        fp32_candidates<kExpertsPerThread>(lane_idx, logits, hi, expert_id);
        fp32_candidates<kExpertsPerThread>(lane_idx, other, lo, expert_id);
        if (scoring_func == ScoringFunc::kSoftmax) {
            // Each gate's row is normalized on its own, then the scores add.
            warp_softmax_inplace<kExpertsPerThread>(hi);
            warp_softmax_inplace<kExpertsPerThread>(lo);
        }
        // Sigmoid scores were applied where the rows were published.
#pragma unroll
        for (uint32_t i = 0; i < kExpertsPerThread; ++i) {
            hi[i] += lo[i];
        }

        float choice[kExpertsPerThread];
#pragma unroll
        for (uint32_t i = 0; i < kExpertsPerThread; ++i) {
            choice[i] = hi[i] + correction_bias[i];
        }
        topk_weights_from_scores_fp32<kExpertsPerThread, kMaxTopk>(
            token_idx, top_k, renormalize, per_expert_scale, choice, hi,
            smem->routing_scratch.topk_ids, smem->routing_scratch.topk_weights);
        return;
    }

    if (scoring_func == ScoringFunc::kSoftmax && !renormalize) {
        float scores[kExpertsPerThread];
        uint32_t expert_id[kExpertsPerThread];
        fp32_candidates<kExpertsPerThread>(lane_idx, logits, scores, expert_id);
        topk_weights_from_softmax_probs<kExpertsPerThread, kMaxTopk>(
            token_idx, top_k, scores, expert_id, smem->routing_scratch.topk_ids,
            smem->routing_scratch.topk_weights);
    } else {
        uint32_t packed_scores[kExpertsPerThread];
        packed_candidates<kExpertsPerThread>(lane_idx, logits, packed_scores);
        topk_weights_from_logits<kExpertsPerThread, kMaxTopk>(
            token_idx, top_k, scoring_func, renormalize, packed_scores,
            smem->routing_scratch.topk_ids, smem->routing_scratch.topk_weights);
    }
}

}  // namespace halops::moe
