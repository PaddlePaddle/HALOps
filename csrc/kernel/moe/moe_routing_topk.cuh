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

#include <cfloat>
#include <cstdint>

#include "moe_configs.cuh"
#include "moe_interface.h"
#include "moe_shared_storage.cuh"
#include "moe_traits.cuh"
#include "moe_workspace.cuh"

namespace halops::moe {

// Warp `token` reads the token's logits row, lane l its expert slice.
template <typename Dims, std::uint32_t kExpertsPerThread>
__device__ static Fragment<float, kExpertsPerThread> load_logits_one_token(
    const MoEWorkspace<Dims>* __restrict__ workspace, std::uint32_t gate,
    std::uint32_t token_idx) {
    using RouterGemm = RouterGemmTraits<Dims>;
    static_assert(kExpertsPerThread % 8u == 0u,
                  "a lane's logits slice must be whole 16-B words");

    Fragment<float, kExpertsPerThread> slice;
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

// Tie-break-by-lowest-expert-index argmax winner.
__device__ static inline uint32_t warp_min_expert_with_max(float my_max,
                                                           float warp_max,
                                                           uint32_t my_expert) {
    return __reduce_min_sync(FULL_MASK,
                             (my_max == warp_max) ? my_expert : 0xFFFFFFFFu);
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
        local_sum += __shfl_xor_sync(FULL_MASK, local_sum, off, 32);

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
    const uint32_t best = __reduce_max_sync(FULL_MASK, moe);
    warp_expert =
        __reduce_min_sync(FULL_MASK, (moe == best) ? my_expert : 0xFFFFFFFFu);
    const uint32_t best_bits =
        (best & 0x80000000u) ? (best & 0x7FFFFFFFu) : ~best;
    warp_score = __uint_as_float(best_bits);
}

// Turns the lane's logits into fp32 candidates; NaN/Inf become -FLT_MAX.
template <uint32_t kExpertsPerThread>
__device__ static void fp32_candidates(
    uint32_t lane_idx, const float (&logits)[kExpertsPerThread],
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

// Packs one bf16 logit pattern into a packed score (see packed_candidates).
__device__ static __forceinline__ uint32_t pack_logit_bits(uint32_t bits,
                                                           uint32_t expert_id) {
    // NaN/Inf -> the lowest finite score.
    if ((bits & 0x7F80u) == 0x7F80u) bits = 0xFF7Fu;
    // -0.0 -> +0.0: equal as floats, so they must not order here either.
    if ((bits & 0x7FFFu) == 0u) bits = 0u;
    const uint32_t moe = (bits & 0x8000u) ? (bits ^ 0xFFFFu) : (bits | 0x8000u);
    return (moe << 16) | (expert_id ^ 0xFFFFu);
}

// Turns the lane's logits into packed scores, one per owned expert.
template <uint32_t kExpertsPerThread>
__device__ static __forceinline__ void packed_candidates(
    uint32_t lane_idx, const float (&logits)[kExpertsPerThread],
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

// Sorts (score, expert_id) pairs by descending score, ties to the lower id.
template <uint32_t N, uint32_t C = 0>
__device__ static void sort_descending_pairs(float (&score)[N],
                                             uint32_t (&expert_id)[N]) {
    static_assert((N & (N - 1u)) == 0u, "the network is built for 2^m keys");
    if constexpr (C < kMergeSortSize<N>) {
        constexpr uint32_t lo = kMergeSortPairs<N>.lo[C];
        constexpr uint32_t hi = kMergeSortPairs<N>.hi[C];
        const float a = score[lo];
        const float b = score[hi];
        const uint32_t ia = expert_id[lo];
        const uint32_t ib = expert_id[hi];
        const bool swap = (a < b) || (a == b && ia > ib);
        // Equal scores are equal bits here, so only the ids need the tie-break.
        score[lo] = fmaxf(a, b);
        score[hi] = fminf(a, b);
        expert_id[lo] = swap ? ib : ia;
        expert_id[hi] = swap ? ia : ib;
        sort_descending_pairs<N, C + 1u>(score, expert_id);
    }
}

// Pops the pair lists' head: warp max, tie-break, then a shift and refill.
template <uint32_t N, uint32_t kKeep>
__device__ static __forceinline__ void select_next_scored_pair(
    float (&score)[N], uint32_t (&expert_id)[N], float& selected_score,
    uint32_t& selected_expert) {
    const float warp_max = warp_reduce_max_float(score[0]);
    selected_expert =
        warp_min_expert_with_max(score[0], warp_max, expert_id[0]);
    selected_score = warp_max;
    const bool won = (score[0] == warp_max && expert_id[0] == selected_expert);
#pragma unroll
    for (uint32_t i = 0; i + 1u < kKeep; i++) {
        score[i] = won ? score[i + 1u] : score[i];
        expert_id[i] = won ? expert_id[i + 1u] : expert_id[i];
    }
    score[kKeep - 1u] = won ? -FLT_MAX : score[kKeep - 1u];
    expert_id[kKeep - 1u] = won ? 0xFFFFFFFFu : expert_id[kKeep - 1u];
}

// Picks the highest remaining packed score (score, then inverted expert id),
// then shifts the winner's list and refills its tail with 0.
template <uint32_t N, uint32_t kKeep>
__device__ static __forceinline__ void select_next_packed_score(
    uint32_t (&best)[N], uint32_t& selected_packed_score) {
    selected_packed_score = __reduce_max_sync(FULL_MASK, best[0]);
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

// Slow path: softmax + renormalize=False; weights are row probabilities.
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

// Sums `v` over lanes [0, top_k) in rank order, like a serial loop.
template <uint32_t kMaxTopk>
__device__ static __forceinline__ float sum_over_ranks(float v,
                                                       uint32_t top_k) {
    float sum = 0.0f;
#pragma unroll
    for (uint32_t k = 0; k < kMaxTopk; k++) {
        const float rank_v = __shfl_sync(FULL_MASK, v, k);
        if (k < top_k) sum += rank_v;
    }
    return sum;
}

// Fast path: softmax + renormalize=True, or sigmoid (either setting).
// Selection runs on packed scores; lane k keeps the rank-k winner.
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

// Selection on fp32 scores; `renormalize` divides by the selected sum.
template <uint32_t kExpertsPerThread, uint32_t kMaxTopk>
__device__ static void topk_weights_from_scores_fp32(
    uint32_t token, uint32_t top_k, bool renormalize,
    float (&scores)[kExpertsPerThread],
    uint32_t (&expert_id)[kExpertsPerThread], uint16_t* topk_ids,
    float* topk_weights) {
    const uint32_t lane_idx = WarpRole::get_lane_idx();
    constexpr uint32_t kKeep =
        kExpertsPerThread < kMaxTopk ? kExpertsPerThread : kMaxTopk;

    // One sort per lane, then each round is a warp max over the list heads.
    sort_descending_pairs<kExpertsPerThread>(scores, expert_id);
    float my_score = 0.0f;
    uint32_t my_expert = 0u;
    for (uint32_t k = 0; k < top_k; k++) {
        float selected_score;
        uint32_t selected_expert;
        select_next_scored_pair<kExpertsPerThread, kKeep>(
            scores, expert_id, selected_score, selected_expert);
        if (lane_idx == k) {
            my_score = selected_score;
            my_expert = selected_expert;
        }
    }

    // The ranks sit one per lane, so the sum runs in rank order.
    float inv = 1.0f;
    if (renormalize) {
        const float sum = sum_over_ranks<kMaxTopk>(my_score, top_k);
        inv = (sum > 0.0f) ? (1.0f / sum) : 1.0f;
    }
    if (lane_idx < top_k) {
        topk_ids[token * kMaxTopk + lane_idx] = (uint16_t)my_expert;
        topk_weights[token * kMaxTopk + lane_idx] = my_score * inv;
    }
}

// Top-K expert selection for one token, one warp per token.
template <typename Dims, uint32_t kExpertsPerThread>
__device__ static void topk_one_token(
    uint32_t token_idx, uint32_t top_k, ScoringFunc scoring_func,
    bool renormalize, const MoEWorkspace<Dims>* __restrict__ workspace,
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

    const Fragment<float, kExpertsPerThread> logits =
        load_logits_one_token<Dims, kExpertsPerThread>(workspace, 0u,
                                                       token_idx);
    if constexpr (kNumGates > 1u) {
        // DualRouter: select on the sum of the two gates' scores.
        const Fragment<float, kExpertsPerThread> other =
            load_logits_one_token<Dims, kExpertsPerThread>(workspace, 1u,
                                                           token_idx);
        float hi[kExpertsPerThread];
        float lo[kExpertsPerThread];
        uint32_t expert_id[kExpertsPerThread];
        fp32_candidates<kExpertsPerThread>(lane_idx, logits.data, hi,
                                           expert_id);
        fp32_candidates<kExpertsPerThread>(lane_idx, other.data, lo, expert_id);
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
        topk_weights_from_scores_fp32<kExpertsPerThread, kMaxTopk>(
            token_idx, top_k, renormalize, hi, expert_id,
            smem->routing_scratch.topk_ids, smem->routing_scratch.topk_weights);
        return;
    }

    if (scoring_func == ScoringFunc::kSoftmax && !renormalize) {
        float scores[kExpertsPerThread];
        uint32_t expert_id[kExpertsPerThread];
        fp32_candidates<kExpertsPerThread>(lane_idx, logits.data, scores,
                                           expert_id);
        topk_weights_from_softmax_probs<kExpertsPerThread, kMaxTopk>(
            token_idx, top_k, scores, expert_id, smem->routing_scratch.topk_ids,
            smem->routing_scratch.topk_weights);
    } else {
        uint32_t packed_scores[kExpertsPerThread];
        packed_candidates<kExpertsPerThread>(lane_idx, logits.data,
                                             packed_scores);
        topk_weights_from_logits<kExpertsPerThread, kMaxTopk>(
            token_idx, top_k, scoring_func, renormalize, packed_scores,
            smem->routing_scratch.topk_ids, smem->routing_scratch.topk_weights);
    }
}

}  // namespace halops::moe
