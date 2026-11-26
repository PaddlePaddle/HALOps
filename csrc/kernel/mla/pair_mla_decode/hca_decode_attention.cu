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

#include <math_constants.h>

#include <initializer_list>
#include <optional>

#include "common/pdl.cuh"
#include "common/sm100_arch.cuh"
#include "kernel/gemm/gemm_utils.h"
#include "kernel/mla/pair_mla_decode/config.cuh"
#include "kernel/mla/pair_mla_decode/hca_host.h"
#include "kernel/mla/pair_mla_decode/kernel.cuh"
#include "kernel/mla/pair_mla_decode/params.cuh"

namespace halops {

namespace detail {

// Scheduling, workspace sizing and the two kernel launches, shared by the HCA
// and DSA entry points.  The callers differ only in the K-cache pool layout
// (set_pool vs set_pool_dsa) and the Q tensor-map column count, all already in
// p / maps by the time they call here, so the launch tail itself is identical.
template <class C>
void launch_decode(HcaParams& p, const HcaMaps& maps, const torch::Tensor& q,
                   bool use_pdl) {
    const int max_clusters = GetSmCount() / 2;
    const int64_t pairs =
        int64_t(p.tokens) / p.tokens_per_req * ((p.tokens_per_req + 1) / 2);
    const int heuristic = int(2 * pairs <= max_clusters ? 2 * pairs
                              : pairs <= max_clusters   ? pairs
                                                        : max_clusters);
    const bool is_dsa = C::kDsa;
    const int eff_mul = is_dsa ? 8 : 2;
    p.eff = eff_mul * pairs <= max_clusters
                ? (is_dsa ? HcaPlan::kEffSized : HcaPlan::kEffCost)
                : 0;
    if (!is_dsa && p.eff == 0 && pairs <= max_clusters)
        p.eff = HcaPlan::kEffAligned;
    p.tpc = HcaPlan::kDefaultTpc;
    p.clusters = p.eff ? max_clusters : heuristic;
    p.smax = p.eff == HcaPlan::kEffSized ? 0 : HcaPlan::kDefaultSmax;
    p.cap = HcaPlan::kNoCap;
    p.nomerge = 0;
    p.pipe = 1;
    p.combine =
        p.eff == HcaPlan::kEffSized || p.eff == HcaPlan::kEffCost ? 1 : 0;
    if (pairs > max_clusters) p.combine = 0;
    p.fuse = is_dsa ? 0 : 1;
    p.ks_kmax = p.eff == HcaPlan::kEffCost && p.combine && p.fuse && pairs > 0
                    ? p.clusters / int(pairs)
                    : 0;
    p.ks_b = HcaCost::kKsBase + HcaCost::kKsPerPair * float(pairs);
    p.ks_a = HcaCost::kTileCost / p.ks_b;
    p.ks_rp = pairs > 0 ? 1.f / float(pairs) : 0.f;
    void (*kernel)(HcaParams, HcaMaps);
    if constexpr (C::kDsa)
        kernel = dsa_decode_attention_kernel<C>;
    else if (p.combine && p.fuse)
        kernel = hca_decode_attention_kernel<C>;
    else
        kernel = hca_plain_attention_kernel<Cfg<typename C::Format, false>>;
    const int smem_bytes = C::kSmemBytes;
    Workspace& ws =
        workspace<HcaCombineCfg<typename C::Layout>>(q, max_clusters);
    p.o_part = ws.o_part.data_ptr<float>();
    p.lse_part = ws.lse_part.data_ptr<float>();
    p.counter = ws.counter.data_ptr<int>();
    p.jobs = reinterpret_cast<MergeJob*>(ws.jobs.data_ptr<int>());
    p.total_g = ws.total_g.data_ptr<int>();
    p.eff_g = ws.eff_g.data_ptr<int>();
    const cudaStream_t stream = GetCurrentCUDAStream(q);
    HALOPS_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
    HALOPS_CUDA_CHECK(LaunchKernelWithPdl(
        kernel, dim3(2 * p.clusters), dim3(C::kThreads), dim3(2, 1, 1),
        smem_bytes, stream, use_pdl, p, maps));
    if (p.combine && !p.fuse) {
        HALOPS_CUDA_CHECK(LaunchKernelWithPdl(
            hca_combine_kernel<HcaCombineCfg<typename C::Layout>>,
            dim3(C::kPlanes, 2,
                 int(std::min<int64_t>(pairs, HcaPlan::kCombJobs))),
            dim3(HcaPlan::kCombSplit * kWarpGroup), kNoCluster, 0, stream,
            use_pdl, p));
    }
}

}  // namespace detail

using namespace detail;

template <class C>
torch::Tensor HcaDecodeAttention(
    const torch::Tensor& q, const torch::Tensor& swa_k_cache,
    const torch::Tensor& swa_indices, const torch::Tensor& swa_lengths,
    const torch::Tensor& c128_k_cache, const torch::Tensor& c128_indices,
    const torch::Tensor& c128_lengths,
    const std::optional<torch::Tensor>& attn_sink, const double softmax_scale,
    const int64_t tokens_per_req, const std::optional<torch::Tensor>& out_opt,
    const bool use_pdl) {
    const CudaDeviceGuard device_guard(q.device());
    const int64_t tokens = q.size(0);
    TORCH_CHECK(q.is_cuda() && q.scalar_type() == torch::kBFloat16 &&
                    q.is_contiguous() &&
                    q.numel() == tokens * C::kHeads * C::kDim,
                "q must be a contiguous bfloat16 [T, ", C::kHeads, ", ",
                C::kDim, "] tensor");
    TORCH_CHECK(tokens_per_req >= 2 && tokens % tokens_per_req == 0,
                "tokens_per_req must be >= 2 and divide T");
    torch::Tensor out = out_opt.has_value() ? *out_opt : torch::empty_like(q);
    TORCH_CHECK(out.scalar_type() == torch::kBFloat16 && out.is_contiguous() &&
                    out.numel() == q.numel(),
                "out must be a contiguous bfloat16 [T, ", C::kHeads, ", ",
                C::kDim, "] tensor");
    for (const torch::Tensor* x : std::initializer_list<const torch::Tensor*>{
             &swa_k_cache, &swa_indices, &swa_lengths, &c128_k_cache,
             &c128_indices, &c128_lengths, &out}) {
        TORCH_CHECK(x->device() == q.device(),
                    "all tensors must be on q's device");
    }
    TORCH_CHECK(
        !attn_sink.has_value() ||
            (attn_sink->device() == q.device() &&
             attn_sink->scalar_type() == torch::kFloat &&
             attn_sink->is_contiguous() && attn_sink->numel() == C::kHeads),
        "attn_sink must be a contiguous float32 [", C::kHeads, "] tensor");
    if (tokens == 0) return out;

    HcaParams p{};
    HcaMaps maps{};
    set_pool<C>(p, maps, 0, c128_k_cache, c128_indices, c128_lengths, tokens);
    set_pool<C>(p, maps, 1, swa_k_cache, swa_indices, swa_lengths, tokens);
    p.q = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr());
    p.sink = attn_sink.has_value() ? attn_sink->data_ptr<float>() : nullptr;
    p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
    p.tokens = tokens;
    p.tokens_per_req = int(tokens_per_req);
    p.scale_log2 = float(softmax_scale) * CUDART_L2E_F;
    EncodeTmaDesc<2>(&maps.q, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, p.q,
                     {uint64_t(C::kDim), uint64_t(tokens * C::kHeads)},
                     {uint64_t(C::kDim) * sizeof(__nv_bfloat16)},
                     {uint32_t(C::kQBoxDims), uint32_t(C::kHeads)},
                     CU_TENSOR_MAP_SWIZZLE_128B, "hca_decode_attention q");
    EncodeTmaDesc<2>(&maps.out, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, p.out,
                     {uint64_t(C::kDim), uint64_t(tokens * C::kHeads)},
                     {uint64_t(C::kDim) * sizeof(__nv_bfloat16)},
                     {uint32_t(C::kQBoxDims), uint32_t(C::kHeads)},
                     CU_TENSOR_MAP_SWIZZLE_128B, "hca_decode_attention out");
    launch_decode<C>(p, maps, q, use_pdl);
    return out;
}

// DSA (full-MLA) decode: q is [T, 64, 576] bf16 (512 latent + 64 rope), the
// single K cache holds 656 B inline fp8 rows, and the output is [T, 64, 512].
// There is no SWA segment.  Everything past the pool layout and the 576-wide Q
// tensor map is the shared launch / scheduling / combine path.
template <class C>
torch::Tensor DsaDecodeAttention(
    const torch::Tensor& q, const torch::Tensor& k_cache,
    const torch::Tensor& indices, const torch::Tensor& lengths,
    const std::optional<torch::Tensor>& attn_sink, const double softmax_scale,
    const int64_t tokens_per_req, const std::optional<torch::Tensor>& out_opt,
    const bool use_pdl) {
    const CudaDeviceGuard device_guard(q.device());
    const int64_t tokens = q.size(0);
    TORCH_CHECK(q.is_cuda() && q.scalar_type() == torch::kBFloat16 &&
                    q.is_contiguous() &&
                    q.numel() == tokens * C::kHeads * C::kDsaQk,
                "q must be a contiguous bfloat16 [T, ", C::kHeads, ", ",
                C::kDsaQk, "] tensor");
    TORCH_CHECK(tokens_per_req >= 1 && tokens % tokens_per_req == 0,
                "tokens_per_req must be >= 1 and divide T");
    torch::Tensor out =
        out_opt.has_value()
            ? *out_opt
            : torch::empty({tokens, C::kHeads, C::kDim}, q.options());
    TORCH_CHECK(out.scalar_type() == torch::kBFloat16 && out.is_contiguous() &&
                    out.numel() == tokens * C::kHeads * C::kDim,
                "out must be a contiguous bfloat16 [T, ", C::kHeads, ", ",
                C::kDim, "] tensor");
    for (const torch::Tensor* x : std::initializer_list<const torch::Tensor*>{
             &k_cache, &indices, &lengths, &out}) {
        TORCH_CHECK(x->device() == q.device(),
                    "all tensors must be on q's device");
    }
    TORCH_CHECK(
        !attn_sink.has_value() ||
            (attn_sink->device() == q.device() &&
             attn_sink->scalar_type() == torch::kFloat &&
             attn_sink->is_contiguous() && attn_sink->numel() == C::kHeads),
        "attn_sink must be a contiguous float32 [", C::kHeads, "] tensor");
    if (tokens == 0) return out;

    HcaParams p{};
    HcaMaps maps{};
    set_pool_dsa<C>(p, maps, k_cache, indices, lengths, tokens);
    p.q = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr());
    p.sink = attn_sink.has_value() ? attn_sink->data_ptr<float>() : nullptr;
    p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
    p.tokens = tokens;
    p.tokens_per_req = int(tokens_per_req);
    p.scale_log2 = float(softmax_scale) * CUDART_L2E_F;
    EncodeTmaDesc<2>(&maps.q, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, p.q,
                     {uint64_t(C::kDsaQk), uint64_t(tokens * C::kHeads)},
                     {uint64_t(C::kDsaQk) * sizeof(__nv_bfloat16)},
                     {uint32_t(C::kQBoxDims), uint32_t(C::kHeads)},
                     CU_TENSOR_MAP_SWIZZLE_128B, "dsa_decode_attention q");
    EncodeTmaDesc<2>(&maps.out, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, p.out,
                     {uint64_t(C::kDim), uint64_t(tokens * C::kHeads)},
                     {uint64_t(C::kDim) * sizeof(__nv_bfloat16)},
                     {uint32_t(C::kQBoxDims), uint32_t(C::kHeads)},
                     CU_TENSOR_MAP_SWIZZLE_128B, "dsa_decode_attention out");
    launch_decode<C>(p, maps, q, use_pdl);
    return out;
}

torch::Tensor HcaDecodeAttention(
    const torch::Tensor& q, const torch::Tensor& swa_k_cache,
    const torch::Tensor& swa_indices, const torch::Tensor& swa_lengths,
    const torch::Tensor& c128_k_cache, const torch::Tensor& c128_indices,
    const torch::Tensor& c128_lengths,
    const std::optional<torch::Tensor>& attn_sink, const double softmax_scale,
    const int64_t tokens_per_req, const std::optional<torch::Tensor>& out_opt,
    const bool use_pdl) {
    const int64_t heads = q.dim() >= 3 ? q.size(-2) : 0;
    const int64_t dim = q.dim() >= 3 ? q.size(-1) : 0;
    const int64_t row = c128_k_cache.dim() == 4 ? c128_k_cache.size(3) : 0;
    TORCH_CHECK(heads == HcaCfg::kHeads && dim == HcaCfg::kDim &&
                    row == HcaCfg::kRowBytes + HcaCfg::kSfBytes,
                "hca_decode_attention has no kernel for heads=", heads,
                ", dim=", dim, ", k cache row bytes=", row,
                "; supported: heads=", HcaCfg::kHeads, ", dim=", HcaCfg::kDim,
                ", k cache row bytes=", HcaCfg::kRowBytes + HcaCfg::kSfBytes);
    return HcaDecodeAttention<HcaCfg>(q, swa_k_cache, swa_indices, swa_lengths,
                                      c128_k_cache, c128_indices, c128_lengths,
                                      attn_sink, softmax_scale, tokens_per_req,
                                      out_opt, use_pdl);
}

torch::Tensor DsaDecodeAttention(
    const torch::Tensor& q, const torch::Tensor& k_cache,
    const torch::Tensor& indices, const torch::Tensor& lengths,
    const std::optional<torch::Tensor>& attn_sink, const double softmax_scale,
    const int64_t tokens_per_req, const std::optional<torch::Tensor>& out_opt,
    const bool use_pdl) {
    const int64_t heads = q.dim() >= 3 ? q.size(-2) : 0;
    const int64_t dim = q.dim() >= 3 ? q.size(-1) : 0;
    const int64_t row = k_cache.dim() == 4 ? k_cache.size(3) : 0;
    TORCH_CHECK(heads == DsaCfg::kHeads && dim == DsaCfg::kDsaQk &&
                    row == DsaCfg::kDsaRowBytes,
                "dsa_decode_attention has no kernel for heads=", heads,
                ", qk dim=", dim, ", k cache row bytes=", row,
                "; supported: heads=", DsaCfg::kHeads,
                ", qk dim=", DsaCfg::kDsaQk,
                ", k cache row bytes=", DsaCfg::kDsaRowBytes);
    return DsaDecodeAttention<DsaCfg>(q, k_cache, indices, lengths, attn_sink,
                                      softmax_scale, tokens_per_req, out_opt,
                                      use_pdl);
}

}  // namespace halops
