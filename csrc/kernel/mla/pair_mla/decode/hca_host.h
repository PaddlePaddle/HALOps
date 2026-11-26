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

#include <mutex>
#include <unordered_map>

#include "common/sm100_arch.cuh"
#include "kernel/gemm/gemm_utils.h"
#include "kernel/mla/pair_mla/common/host.h"
#include "kernel/mla/pair_mla/decode/config.cuh"
#include "kernel/mla/pair_mla/decode/params.cuh"

namespace halops::detail {

template <class C>
void set_pool(HcaParams& p, HcaMaps& maps, int seg, const torch::Tensor& pool,
              const torch::Tensor& idx, const torch::Tensor& len,
              int64_t tokens) {
    TORCH_CHECK(pool.dim() == 4 && pool.scalar_type() == torch::kByte &&
                    pool.size(2) == 1 &&
                    pool.size(3) == C::kRowBytes + C::kSfBytes &&
                    pool.stride(0) % C::kRowBytes == 0,
                "k caches must be uint8 [pages, page_size, 1, ",
                C::kRowBytes + C::kSfBytes,
                "] with a page stride that is a multiple of ", C::kRowBytes);
    TORCH_CHECK(idx.scalar_type() == torch::kInt && idx.is_contiguous() &&
                    idx.dim() >= 2 && idx.numel() == tokens * idx.size(-1),
                "indices must be contiguous int32 [T, W] or [T, 1, W]");
    TORCH_CHECK(len.scalar_type() == torch::kInt && len.is_contiguous() &&
                    len.numel() == tokens,
                "lengths must be contiguous int32 [T]");
    p.idx[seg] = idx.data_ptr<int>();
    p.len[seg] = len.data_ptr<int>();
    p.ld[seg] = idx.size(-1);
    p.pool[seg] = pool.data_ptr<uint8_t>();
    p.pool_stride[seg] = pool.stride(0);
    const int page = pool.size(1);
    TORCH_CHECK((page & (page - 1)) == 0,
                "k cache page sizes must be powers of two, got ", page);
    p.page_shift[seg] = __builtin_ctz(page);
    p.page_rows[seg] = pool.stride(0) / C::kRowBytes;
    p.pool_tokens[seg] = pool.size(0) * pool.size(1);
    const uint8_t* base = pool.data_ptr<uint8_t>();
    for (int c = 0; c < 2; ++c) {
        maps.raw[c][seg] = pair_make_pool_map<C>(
            base, CU_TENSOR_MAP_DATA_TYPE_UINT16, pool, C::kRowBytes / 2,
            raw_pitch<C>(c) / 2, CU_TENSOR_MAP_SWIZZLE_NONE,
            "hca_decode_attention");
    }
}

// DSA tensor map: UINT16 view of the 656 B inline rows, box = this CTA's 256 B
// fp8 nope half (128 uint16), gathered per key row like HCA's raw map.
template <class C>
CUtensorMap make_map_dsa(const void* ptr, const torch::Tensor& pool,
                         uint64_t cols, uint32_t box) {
    CUtensorMap map;
    EncodeTmaDesc<2>(
        &map, CU_TENSOR_MAP_DATA_TYPE_UINT16, ptr,
        {cols, uint64_t(pool.size(0) * (pool.stride(0) / C::kDsaRowBytes))},
        {C::kDsaRowBytes}, {box, 1}, CU_TENSOR_MAP_SWIZZLE_NONE,
        "dsa_decode_attention");
    return map;
}

// DSA uses a single c128-style pool (seg 0) of 656 B inline rows and has no SWA
// segment.  Seg 1 is cloned from seg 0 with ld == 0 so the shared scheduler's
// unconditional length reads touch valid memory yet yield zero SWA keys.
template <class C>
void set_pool_dsa(HcaParams& p, HcaMaps& maps, const torch::Tensor& pool,
                  const torch::Tensor& idx, const torch::Tensor& len,
                  int64_t tokens) {
    TORCH_CHECK(pool.dim() == 4 && pool.scalar_type() == torch::kByte &&
                    pool.size(2) == 1 && pool.size(3) == C::kDsaRowBytes &&
                    pool.stride(0) % C::kDsaRowBytes == 0,
                "DSA k cache must be uint8 [pages, page_size, 1, ",
                C::kDsaRowBytes, "] with a page stride that is a multiple of ",
                C::kDsaRowBytes);
    TORCH_CHECK(idx.scalar_type() == torch::kInt && idx.is_contiguous() &&
                    idx.dim() >= 2 && idx.numel() == tokens * idx.size(-1),
                "indices must be contiguous int32 [T, W] or [T, 1, W]");
    TORCH_CHECK(len.scalar_type() == torch::kInt && len.is_contiguous() &&
                    len.numel() == tokens,
                "lengths must be contiguous int32 [T]");
    p.idx[0] = idx.data_ptr<int>();
    p.len[0] = len.data_ptr<int>();
    p.ld[0] = idx.size(-1);
    p.pool[0] = pool.data_ptr<uint8_t>();
    p.pool_stride[0] = pool.stride(0);
    const int page = pool.size(1);
    TORCH_CHECK((page & (page - 1)) == 0,
                "k cache page sizes must be powers of two, got ", page);
    p.page_shift[0] = __builtin_ctz(page);
    p.page_rows[0] = pool.stride(0) / C::kDsaRowBytes;
    p.pool_tokens[0] = pool.size(0) * pool.size(1);
    const uint8_t* base = pool.data_ptr<uint8_t>();
    for (int c = 0; c < 2; ++c) {
        maps.raw[c][0] = make_map_dsa<C>(base, pool, C::kDsaRowBytes / 2,
                                         C::kDsaNopeHalf / 2);
    }
    p.idx[1] = p.idx[0];
    p.len[1] = p.len[0];
    p.ld[1] = 0;
    p.pool[1] = p.pool[0];
    p.pool_stride[1] = p.pool_stride[0];
    p.page_shift[1] = p.page_shift[0];
    p.page_rows[1] = p.page_rows[0];
    p.pool_tokens[1] = p.pool_tokens[0];
    maps.raw[0][1] = maps.raw[0][0];
    maps.raw[1][1] = maps.raw[1][0];
}

// Split-pair partials and counters, kept per device: the counters start at
// zero and the last part of each split pair resets its own.
struct Workspace {
    torch::Tensor o_part, lse_part, counter, jobs, total_g, eff_g;
};

template <class C>
Workspace& workspace(const torch::Tensor& q, int clusters) {
    static std::mutex mu;
    static std::unordered_map<int, Workspace> cache;
    const std::lock_guard<std::mutex> lock(mu);
    Workspace& ws = cache[q.get_device()];
    if (!ws.counter.defined() || ws.counter.size(0) != clusters) {
        const auto f32 = q.options().dtype(torch::kFloat);
        const auto i32 = q.options().dtype(torch::kInt);
        ws.o_part = torch::empty(
            {2 * clusters, 2, C::kPlanes, kWarpGroup, C::kPlaneCols}, f32);
        ws.lse_part = torch::empty({2 * clusters, 2, C::kHeads}, f32);
        ws.counter = torch::zeros({clusters, 2}, i32);
        ws.jobs = torch::empty({clusters, 4}, i32);
        ws.total_g = torch::zeros({1}, i32);
        ws.eff_g = torch::zeros({1}, i32);
    }
    return ws;
}

}  // namespace halops::detail
