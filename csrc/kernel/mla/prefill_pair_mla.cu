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

#include <cuda.h>
#include <cuda_bf16.h>
#include <math_constants.h>

#include <algorithm>
#include <initializer_list>
#include <mutex>
#include <optional>
#include <tuple>
#include <unordered_map>

#include "common/pdl.cuh"
#include "kernel/gemm/gemm_utils.h"
#include "kernel/mla/mla_host_utils.h"
#include "kernel/mla/prefill_pair_mla_kernel.cuh"

namespace halops::mla::prefill_pair {

// The layouts the kernels and their TMA maps rely on.
template <class T>
void check_inputs(
    const torch::Tensor& q, const torch::Tensor& swa_k_cache,
    const torch::Tensor& swa_indices, const torch::Tensor& swa_lengths,
    const torch::Tensor& c128_k_cache, const torch::Tensor& c128_indices,
    const torch::Tensor& c128_lengths,
    const std::optional<torch::Tensor>& attn_sink, const int64_t tokens_per_req,
    const std::optional<torch::Tensor>& out_opt,
    const std::optional<torch::Tensor>& cu_seqlens,
    const std::optional<torch::Tensor>& cu_pairs, const int64_t swa_shift,
    const std::optional<torch::Tensor>& q_positions,
    const std::optional<torch::Tensor>& q_freqs) {
    const int64_t tokens = q.size(0);
    TORCH_CHECK(swa_shift >= 1 && swa_shift <= 128,
                "swa_shift must be in [1, 128]");
    TORCH_CHECK(q_positions.has_value() == q_freqs.has_value(),
                "q_positions and q_freqs go together");
    TORCH_CHECK(cu_seqlens.has_value() == cu_pairs.has_value(),
                "cu_seqlens and cu_pairs go together");
    if (q_positions.has_value()) {
        // Fused q_norm_rope: q is the raw projection, heads contiguous.
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
    if (cu_seqlens.has_value()) {
        for (const torch::Tensor* x : {&*cu_seqlens, &*cu_pairs}) {
            TORCH_CHECK(
                x->scalar_type() == torch::kInt && x->is_contiguous() &&
                    x->dim() == 1 && x->numel() == cu_seqlens->numel() &&
                    x->numel() >= 2,
                "cu_seqlens / cu_pairs must be contiguous int32 [R + 1]");
        }
    } else {
        TORCH_CHECK(tokens_per_req >= 2 && tokens % tokens_per_req == 0,
                    "tokens_per_req must be >= 2 and divide T");
    }
    TORCH_CHECK(!out_opt.has_value() ||
                    (out_opt->scalar_type() == torch::kBFloat16 &&
                     out_opt->is_contiguous() && out_opt->numel() == q.numel()),
                "out must be a contiguous bfloat16 [T, 64, 512] tensor");
    TORCH_CHECK(
        !attn_sink.has_value() ||
            (attn_sink->scalar_type() == torch::kFloat &&
             attn_sink->is_contiguous() && attn_sink->numel() == T::kHeads),
        "attn_sink must be a contiguous float32 [64] tensor");
    // c128 first, then swa.
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

template <class T>
CUtensorMap make_map(const void* ptr, CUtensorMapDataType dtype,
                     const torch::Tensor& pool, uint64_t cols, uint32_t box,
                     CUtensorMapSwizzle swizzle) {
    CUtensorMap map;
    EncodeTmaDesc<2>(
        &map, dtype, ptr,
        {cols, uint64_t(pool.size(0) * (pool.stride(0) / T::kRowBytes))},
        {T::kRowBytes}, {box, 1}, swizzle, "prefill_pair_mla");
    return map;
}

template <class T>
void set_pool(PairMlaParams& p, PairMlaMaps& maps, int seg,
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
        maps.raw[c][seg] = make_map<T>(
            base, CU_TENSOR_MAP_DATA_TYPE_UINT16, pool, T::kRowBytes / 2,
            T::raw_pitch(c) / 2, CU_TENSOR_MAP_SWIZZLE_NONE);
    }
}

// Per device: split-pair partials and counters, the bf16 c128 copy and its
// per-token bases (grown as needed).
struct Workspace {
    torch::Tensor o_part, lse_part, counter, jobs, total_g, sched_g, c128_rows,
        c128_base;
};

template <class T>
Workspace& workspace(const torch::Tensor& q, int clusters, int64_t c128_rows,
                     int64_t tokens) {
    static std::mutex mu;
    static std::unordered_map<int, Workspace> cache;
    const std::lock_guard<std::mutex> lock(mu);
    Workspace& ws = cache[q.get_device()];
    const auto i32 = q.options().dtype(torch::kInt);
    if (!ws.counter.defined() || ws.counter.size(0) != clusters) {
        const auto f32 = q.options().dtype(torch::kFloat);
        // Two slots per cluster: a continuation (0) and an owner (1).
        ws.o_part = torch::empty(
            {2 * clusters, 2, T::kPlanes, T::kWarpGroup, T::kPlaneCols}, f32);
        ws.lse_part = torch::empty({2 * clusters, 2, T::kHeads}, f32);
        ws.counter = torch::zeros({clusters, 2}, i32);
        ws.jobs = torch::empty({clusters, 4}, i32);  // MergeJob[clusters]
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
int64_t predq_rows(const PairMlaParams& p) {
    const int n_req = p.cu_seqlens ? p.n_req : p.tokens / p.tokens_per_req;
    const int64_t per_req =
        (int64_t(p.ld[0]) + T::kTile - 1) / T::kTile * T::kTile;
    const int64_t rows = int64_t(n_req) * per_req;
    const bool worth = (p.ld[0] >= PredqTraits::kMinLd &&
                        p.tokens >= PredqTraits::kMinTokens) ||
                       (p.ld[0] >= PredqTraits::kShortLd &&
                        p.tokens >= PredqTraits::kShortTokens);
    if (p.pool_tokens[0] == 0 || n_req == 0 || !worth ||
        rows * T::kDim * int64_t(sizeof(__nv_bfloat16)) >
            PredqTraits::kMaxBytes) {
        return 0;
    }
    return rows;
}

template <class T>
void predequant_c128(PairMlaParams& p, PairMlaMaps& maps, const Workspace& ws,
                     int64_t rows, cudaStream_t stream, bool use_pdl) {
    p.c128_bf = reinterpret_cast<__nv_bfloat16*>(ws.c128_rows.data_ptr());
    p.c128_base = ws.c128_base.data_ptr<int>();
    p.c128_rows = (p.ld[0] + T::kTile - 1) / T::kTile * T::kTile;
    EncodeTmaDesc<2>(&maps.bf16, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, p.c128_bf,
                     {uint64_t(T::kDim), uint64_t(rows)},
                     {uint64_t(T::kDim) * sizeof(__nv_bfloat16)},
                     {uint32_t(T::kBlockDims), uint32_t(T::kWarp)},
                     CU_TENSOR_MAP_SWIZZLE_128B, "prefill_pair_mla c128 bf16");
    cudaLaunchAttribute attr = PdlLaunchAttribute();
    cudaLaunchConfig_t cfg{};
    cfg.stream = stream;
    cfg.attrs = &attr;
    cfg.numAttrs = use_pdl ? 1 : 0;
    cfg.blockDim = dim3(PredqTraits::kThreads);
    cfg.gridDim =
        dim3((p.tokens + PredqTraits::kWarps - 1) / PredqTraits::kWarps +
             (rows + PredqTraits::kWarps - 1) / PredqTraits::kWarps);
    HALOPS_CUDA_CHECK(cudaLaunchKernelEx(&cfg, predq_kernel<T>, p));
}

// Main kernel, then the combine kernel for few pairs (p.combine).
template <class T>
void launch_attention(PairMlaParams& p, const PairMlaMaps& maps,
                      const Workspace& ws, int max_clusters,
                      cudaStream_t stream,
                      void (*kernel)(PairMlaParams, PairMlaMaps),
                      int smem_bytes, bool use_pdl) {
    // Varlen: (T + R) / 2 bounds the pair count.
    const int64_t pairs = p.cu_pairs ? (int64_t(p.tokens) + p.n_req) / 2
                                     : int64_t(p.tokens) / p.tokens_per_req *
                                           ((p.tokens_per_req + 1) / 2);
    p.combine = 2 * pairs <= max_clusters ? 1 : 0;
    p.clusters =
        p.combine ? max_clusters : int(std::min<int64_t>(pairs, max_clusters));
    p.o_part = ws.o_part.data_ptr<float>();
    p.lse_part = ws.lse_part.data_ptr<float>();
    p.counter = ws.counter.data_ptr<int>();
    p.jobs = reinterpret_cast<MergeJob*>(ws.jobs.data_ptr<int>());
    p.total_g = ws.total_g.data_ptr<int>();
    p.sched_g = ws.sched_g.data_ptr<int>();
    cudaLaunchAttribute attrs[2];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim = {2, 1, 1};
    attrs[1] = PdlLaunchAttribute();
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(2 * p.clusters);
    cfg.blockDim = dim3(T::kThreads);
    cfg.dynamicSmemBytes = smem_bytes;
    cfg.stream = stream;
    cfg.attrs = attrs;
    cfg.numAttrs = use_pdl ? 2 : 1;
    setup_kernel_once(reinterpret_cast<const void*>(kernel),
                      {0, T::kThreads, size_t(smem_bytes)}, cfg);
    HALOPS_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, p, maps));
    if (p.combine) {
        // A PDL dependent of the main kernel on the same stream.
        cudaLaunchAttribute cattr = PdlLaunchAttribute();
        cudaLaunchConfig_t ccfg{};
        ccfg.gridDim = dim3(T::kPlanes, 2, p.clusters);
        ccfg.blockDim = dim3(T::kCombSplit * T::kWarpGroup);
        ccfg.dynamicSmemBytes = 0;
        ccfg.stream = stream;
        ccfg.attrs = &cattr;
        ccfg.numAttrs = use_pdl ? 1 : 0;
        HALOPS_CUDA_CHECK(
            cudaLaunchKernelEx(&ccfg, pair_mla_combine_kernel<T>, p));
    }
}

template <class Dims>
torch::Tensor prefill_pair_mla(
    const torch::Tensor& q, const torch::Tensor& swa_k_cache,
    const torch::Tensor& swa_indices, const torch::Tensor& swa_lengths,
    const torch::Tensor& c128_k_cache, const torch::Tensor& c128_indices,
    const torch::Tensor& c128_lengths,
    const std::optional<torch::Tensor>& attn_sink, const double softmax_scale,
    const int64_t tokens_per_req, const std::optional<torch::Tensor>& out_opt,
    const bool use_pdl, const std::optional<torch::Tensor>& cu_seqlens,
    const std::optional<torch::Tensor>& cu_pairs, const int64_t swa_shift,
    const std::optional<torch::Tensor>& q_positions,
    const std::optional<torch::Tensor>& q_freqs, const double q_norm_eps) {
    using T = PairMlaTraits<Dims>;
    const CudaDeviceGuard device_guard(q.device());
    check_inputs<T>(q, swa_k_cache, swa_indices, swa_lengths, c128_k_cache,
                    c128_indices, c128_lengths, attn_sink, tokens_per_req,
                    out_opt, cu_seqlens, cu_pairs, swa_shift, q_positions,
                    q_freqs);
    torch::Tensor out =
        out_opt.has_value() ? *out_opt : torch::empty(q.sizes(), q.options());
    const int64_t tokens = q.size(0);
    if (tokens == 0) return out;
    const bool qnr = q_positions.has_value();
    const bool varlen = cu_seqlens.has_value();

    PairMlaParams p{};
    PairMlaMaps maps{};
    set_pool<T>(p, maps, 0, c128_k_cache, c128_indices, c128_lengths);
    set_pool<T>(p, maps, 1, swa_k_cache, swa_indices, swa_lengths);
    p.q = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr());
    p.sink = attn_sink.has_value() ? attn_sink->data_ptr<float>() : nullptr;
    p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
    p.tokens = tokens;
    p.tokens_per_req = varlen ? 0 : int(tokens_per_req);
    p.swa_shift = int(swa_shift);
    if (varlen) {
        p.cu_seqlens = cu_seqlens->data_ptr<int>();
        p.cu_pairs = cu_pairs->data_ptr<int>();
        p.n_req = int(cu_seqlens->numel() - 1);
    }
    p.scale_log2 = float(softmax_scale) * CUDART_L2E_F;
    if (qnr) {
        p.q_stride = q.stride(0);
        p.q_pos = q_positions->data_ptr();
        p.pos64 = q_positions->scalar_type() == torch::kLong;
        // complex64 [P, 32] is float32 [P, 64], (re, im) interleaved.
        p.q_freqs = reinterpret_cast<const float*>(q_freqs->data_ptr());
        p.q_eps = float(q_norm_eps);
        // Same boxes as below, one token's 64 heads apart by q_stride.
        EncodeTmaDesc<3>(
            &maps.q, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, p.q,
            {uint64_t(T::kDim), uint64_t(T::kHeads), uint64_t(tokens)},
            {uint64_t(T::kDim) * sizeof(__nv_bfloat16),
             uint64_t(p.q_stride) * sizeof(__nv_bfloat16)},
            {uint32_t(T::kQBoxDims), uint32_t(T::kHeads), 1u},
            CU_TENSOR_MAP_SWIZZLE_128B, "prefill_pair_mla raw q");
    } else {
        EncodeTmaDesc<2>(&maps.q, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, p.q,
                         {uint64_t(T::kDim), uint64_t(tokens * T::kHeads)},
                         {uint64_t(T::kDim) * sizeof(__nv_bfloat16)},
                         {uint32_t(T::kQBoxDims), uint32_t(T::kHeads)},
                         CU_TENSOR_MAP_SWIZZLE_128B, "prefill_pair_mla q");
    }
    EncodeTmaDesc<2>(&maps.out, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, p.out,
                     {uint64_t(T::kDim), uint64_t(tokens * T::kHeads)},
                     {uint64_t(T::kDim) * sizeof(__nv_bfloat16)},
                     {uint32_t(T::kQBoxDims), uint32_t(T::kHeads)},
                     CU_TENSOR_MAP_SWIZZLE_128B, "prefill_pair_mla out");
    const int max_clusters = GetSmCount() / 2;
    TORCH_CHECK(max_clusters <= PairSchedTraits::kMaxClusters,
                "more SMs than the pair kernel schedules");
    const int64_t rows = predq_rows<T>(p);
    const Workspace& ws = workspace<T>(q, max_clusters, rows, tokens);
    const cudaStream_t stream = GetCurrentCUDAStream(q);
    if (rows > 0) predequant_c128<T>(p, maps, ws, rows, stream, use_pdl);
    // kQst for short pairs; ld[0] + ld[1] bounds a pair's keys.
    const bool qst = (p.ld[0] + p.ld[1] + T::kTile - 1) / T::kTile <=
                     PairSchedTraits::kQstMaxTiles;
    const bool fp8 = p.c128_base == nullptr;
    using Kernel = void (*)(PairMlaParams, PairMlaMaps);
    constexpr Kernel kKernels[2][2][2] = {
        {{pair_mla_kernel<false, false, false, Dims>,
          pair_mla_kernel<false, false, true, Dims>},
         {pair_mla_kernel<false, true, false, Dims>,
          pair_mla_kernel<false, true, true, Dims>}},
        {{pair_mla_kernel<true, false, false, Dims>,
          pair_mla_kernel<true, false, true, Dims>},
         {pair_mla_kernel<true, true, false, Dims>,
          pair_mla_kernel<true, true, true, Dims>}}};
    const Kernel kernel = kKernels[qst][fp8][qnr];
    const int smem_bytes = qst ? Cfg<true, false, false, Dims>::kSmemBytes
                               : Cfg<false, false, false, Dims>::kSmemBytes;
    launch_attention<T>(p, maps, ws, max_clusters, stream, kernel, smem_bytes,
                        use_pdl);
    return out;
}

}  // namespace halops::mla::prefill_pair

namespace halops {

torch::Tensor PrefillPairMla(
    const torch::Tensor& q, const torch::Tensor& swa_k_cache,
    const torch::Tensor& swa_indices, const torch::Tensor& swa_lengths,
    const torch::Tensor& c128_k_cache, const torch::Tensor& c128_indices,
    const torch::Tensor& c128_lengths,
    const std::optional<torch::Tensor>& attn_sink, const double softmax_scale,
    const int64_t tokens_per_req, const std::optional<torch::Tensor>& out_opt,
    const bool use_pdl, const std::optional<torch::Tensor>& cu_seqlens,
    const std::optional<torch::Tensor>& cu_pairs, const int64_t swa_shift,
    const std::optional<torch::Tensor>& q_positions,
    const std::optional<torch::Tensor>& q_freqs, const double q_norm_eps) {
    // Model dims from q; only [64, 512] is instantiated.
    const int64_t heads = q.dim() >= 3 ? q.size(-2) : 0;
    const int64_t dim = q.dim() >= 3 ? q.size(-1) : 0;
    TORCH_CHECK(heads == mla::ModelDims::Heads && dim == mla::ModelDims::DimQ,
                "prefill_pair_mla: unsupported q dims [heads, dim] = [", heads,
                ", ", dim, "], supported: [64, 512]");
    return mla::prefill_pair::prefill_pair_mla<mla::ModelDims>(
        q, swa_k_cache, swa_indices, swa_lengths, c128_k_cache, c128_indices,
        c128_lengths, attn_sink, softmax_scale, tokens_per_req, out_opt,
        use_pdl, cu_seqlens, cu_pairs, swa_shift, q_positions, q_freqs,
        q_norm_eps);
}

}  // namespace halops
