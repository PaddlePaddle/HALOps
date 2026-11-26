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

#include <initializer_list>
#include <mutex>
#include <optional>
#include <tuple>
#include <unordered_map>

#include "kernel/gemm/gemm_utils.h"
#include "kernel/mla/common/host_utils.h"
#include "kernel/mla/pair_mla_prefill/params.cuh"

namespace halops::mla {

// The q, RoPE and swa_shift layouts the kernels and their TMA maps rely on;
// with q_positions, q is the raw projection with contiguous heads.
template <class T>
void prefill_check_q_inputs(const torch::Tensor& q, const int64_t swa_shift,
                            const std::optional<torch::Tensor>& q_positions,
                            const std::optional<torch::Tensor>& q_freqs) {
    const int64_t tokens = q.size(0);
    TORCH_CHECK(swa_shift >= 1 && swa_shift <= 128,
                "swa_shift must be in [1, 128]");
    TORCH_CHECK(q_positions.has_value() == q_freqs.has_value(),
                "q_positions and q_freqs go together");
    if (q_positions.has_value()) {
        TORCH_CHECK(
            q.scalar_type() == torch::kBFloat16 &&
                (q.dim() == 3 || (q.dim() == 4 && q.size(1) == 1)) &&
                q.stride(-1) == 1 && q.stride(-2) == T::kDim &&
                q.stride(0) % T::kElemsPerChunk == 0 &&
                reinterpret_cast<uintptr_t>(q.data_ptr()) % T::kChunk == 0,
            "q must be a bfloat16 [T, 64, 512] view with contiguous heads");
        const torch::Tensor& pos = *q_positions;
        const torch::Tensor& freqs = *q_freqs;
        const bool cplx = freqs.scalar_type() == torch::kComplexFloat;
        TORCH_CHECK((pos.scalar_type() == torch::kLong ||
                     pos.scalar_type() == torch::kInt) &&
                        pos.is_contiguous() && pos.numel() == tokens &&
                        (cplx || freqs.scalar_type() == torch::kFloat) &&
                        freqs.is_contiguous() && freqs.dim() == 2 &&
                        freqs.size(1) * (cplx ? 2 : 1) == T::kQRopeDims,
                    "q_positions must be contiguous int64 / int32 [T], q_freqs "
                    "contiguous complex64 [P, 32] or float32 [P, 64]");
    } else {
        TORCH_CHECK(q.scalar_type() == torch::kBFloat16 && q.is_contiguous() &&
                        q.numel() == tokens * T::kHeads * T::kDim,
                    "q must be a contiguous bfloat16 [T, 64, 512] tensor");
    }
}

// The request offsets, sink and KV cache layouts (c128 first, then SWA).
template <class T>
void prefill_check_kv_inputs(
    const torch::Tensor& q, const torch::Tensor& swa_k_cache,
    const torch::Tensor& swa_indices, const torch::Tensor& swa_lengths,
    const torch::Tensor& c128_k_cache, const torch::Tensor& c128_indices,
    const torch::Tensor& c128_lengths,
    const std::optional<torch::Tensor>& attn_sink,
    const torch::Tensor& cu_seqlens, const torch::Tensor& cu_pairs) {
    const int64_t tokens = q.size(0);
    for (const torch::Tensor* x : {&cu_seqlens, &cu_pairs}) {
        TORCH_CHECK(x->scalar_type() == torch::kInt && x->is_contiguous() &&
                        x->dim() == 1 && x->numel() == cu_seqlens.numel() &&
                        x->numel() >= 2,
                    "cu_seqlens / cu_pairs must be contiguous int32 [R + 1]");
    }
    TORCH_CHECK(
        !attn_sink.has_value() ||
            (attn_sink->scalar_type() == torch::kFloat &&
             attn_sink->is_contiguous() && attn_sink->numel() == T::kHeads),
        "attn_sink must be a contiguous float32 [64] tensor");
    for (const auto& [pool, idx, len] :
         {std::tie(c128_k_cache, c128_indices, c128_lengths),
          std::tie(swa_k_cache, swa_indices, swa_lengths)}) {
        const int64_t page = pool.dim() == 4 ? pool.size(1) : 0;
        TORCH_CHECK(
            pool.dim() == 4 && pool.scalar_type() == torch::kByte &&
                pool.size(2) == 1 &&
                pool.size(3) == T::kRowBytes + T::kSfBytes &&
                (page & (page - 1)) == 0 &&
                pool.stride(0) % T::kRowBytes == 0 &&
                pool.stride(0) >= page * (T::kRowBytes + T::kSfBytes),
            "k caches must be uint8 [pages, page_size, 1, 584] with a "
            "power-of-two page_size and a page stride that is a multiple of "
            "576 and at least page_size * 584");
        TORCH_CHECK(
            idx.scalar_type() == torch::kInt && idx.is_contiguous() &&
                idx.dim() >= 2 && idx.numel() == tokens * idx.size(-1) &&
                len.scalar_type() == torch::kInt && len.is_contiguous() &&
                len.numel() == tokens,
            "indices must be contiguous int32 [T, W] or [T, 1, W], lengths "
            "contiguous int32 [T]");
    }
}

// A 2D TMA map over the KV pool rows (one row per token slot).
template <class T>
CUtensorMap prefill_make_map(const void* ptr, CUtensorMapDataType dtype,
                             const torch::Tensor& pool, uint64_t cols,
                             uint32_t box, CUtensorMapSwizzle swizzle) {
    CUtensorMap map;
    EncodeTmaDesc<2>(
        &map, dtype, ptr,
        {cols, uint64_t(pool.size(0) * (pool.stride(0) / T::kRowBytes))},
        {T::kRowBytes}, {box, 1}, swizzle, "prefill_pair_mla");
    return map;
}

// Segment seg's pool, lists and lengths, and the per-CTA raw row maps.
template <class T>
void prefill_set_pool(PrefillParams& p, PrefillMaps& maps, int seg,
                      const torch::Tensor& pool, const torch::Tensor& idx,
                      const torch::Tensor& len) {
    p.idx[seg] = idx.data_ptr<int>();
    p.len[seg] = len.data_ptr<int>();
    p.ld[seg] = idx.size(-1);
    p.pool[seg] = pool.data_ptr<uint8_t>();
    p.pool_stride[seg] = pool.stride(0);
    p.page_shift[seg] = __builtin_ctz(pool.size(1));
    p.page_rows[seg] = pool.stride(0) / T::kRowBytes;
    p.pool_tokens[seg] = pool.size(0) * pool.size(1);
    const uint8_t* base = pool.data_ptr<uint8_t>();
    for (int c = 0; c < 2; ++c) {
        maps.raw[c][seg] = prefill_make_map<T>(
            base, CU_TENSOR_MAP_DATA_TYPE_UINT16, pool, T::kRowBytes / 2,
            T::raw_pitch(c) / 2, CU_TENSOR_MAP_SWIZZLE_NONE);
    }
}

// Per device: split-pair partials (two slots per cluster, a continuation and
// an owner) and counters, the bf16 c128 copy and its per-token bases.
struct PrefillWorkspace {
    torch::Tensor o_part, lse_part, counter, jobs, total_g, sched_g, c128_rows,
        c128_base;
};

// The device's workspace, grown as needed; jobs holds
// PrefillMergeJob[clusters].
template <class T>
PrefillWorkspace& prefill_workspace(const torch::Tensor& q, int clusters,
                                    int64_t c128_rows, int64_t tokens) {
    static std::mutex mu;
    static std::unordered_map<int, PrefillWorkspace> cache;
    const std::lock_guard<std::mutex> lock(mu);
    PrefillWorkspace& ws = cache[q.get_device()];
    const auto i32 = q.options().dtype(torch::kInt);
    if (!ws.counter.defined() || ws.counter.size(0) != clusters) {
        const auto f32 = q.options().dtype(torch::kFloat);
        ws.o_part = torch::empty(
            {2 * clusters, 2, T::kPlanes, T::kWarpGroup, T::kPlaneCols}, f32);
        ws.lse_part = torch::empty({2 * clusters, 2, T::kHeads}, f32);
        ws.counter = torch::zeros({clusters, 2}, i32);
        ws.jobs = torch::empty({clusters, 4}, i32);
        ws.total_g = torch::zeros({1}, i32);
        ws.sched_g = torch::zeros({1}, i32);
    }
    if (c128_rows > 0) {
        if (!ws.c128_rows.defined() || ws.c128_rows.size(0) < c128_rows) {
            ws.c128_rows = torch::zeros({c128_rows, T::kDim}, q.options());
        }
        if (!ws.c128_base.defined() || ws.c128_base.size(0) < tokens) {
            ws.c128_base = torch::empty({tokens}, i32);
        }
    }
    return ws;
}

// Rows of the bf16 c128 copy, or 0 when pre-dequant does not pay off.
template <class T>
int64_t prefill_predq_rows(const PrefillParams& p) {
    const int64_t per_req =
        (int64_t(p.ld[0]) + T::kTile - 1) / T::kTile * T::kTile;
    const int64_t rows = int64_t(p.n_req) * per_req;
    const bool worth = (p.ld[0] >= PrefillPredqTraits::kMinLd &&
                        p.tokens >= PrefillPredqTraits::kMinTokens) ||
                       (p.ld[0] >= PrefillPredqTraits::kShortLd &&
                        p.tokens >= PrefillPredqTraits::kShortTokens);
    if (p.pool_tokens[0] == 0 || p.n_req == 0 || !worth ||
        rows * T::kDim * int64_t(sizeof(__nv_bfloat16)) >
            PrefillPredqTraits::kMaxBytes) {
        return 0;
    }
    return rows;
}

}  // namespace halops::mla
