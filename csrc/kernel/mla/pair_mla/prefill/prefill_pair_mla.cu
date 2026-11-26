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
#include <optional>

#include "common/pdl.cuh"
#include "kernel/mla/pair_mla/prefill/kernel.cuh"
#include "kernel/mla/pair_mla/prefill/launcher.h"

namespace halops::mla {

// Builds the bf16 c128 copy (prefill_predq_kernel) and its TMA map.
template <class T>
void prefill_predequant_c128(PrefillParams& p, PrefillMaps& maps,
                             const PrefillWorkspace& ws, int64_t rows,
                             cudaStream_t stream, bool use_pdl) {
    p.c128_bf = reinterpret_cast<__nv_bfloat16*>(ws.c128_rows.data_ptr());
    p.c128_base = ws.c128_base.data_ptr<int>();
    p.c128_rows = (p.ld[0] + T::kTile - 1) / T::kTile * T::kTile;
    EncodeTmaDesc<2>(&maps.bf16, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, p.c128_bf,
                     {uint64_t(T::kDim), uint64_t(rows)},
                     {uint64_t(T::kDim) * sizeof(__nv_bfloat16)},
                     {uint32_t(T::kBlockDims), uint32_t(T::kWarp)},
                     CU_TENSOR_MAP_SWIZZLE_128B, "prefill_pair_mla c128 bf16");
    constexpr int kWarps = PrefillPredqTraits::kWarps;
    const dim3 grid((p.tokens + kWarps - 1) / kWarps +
                    (rows + kWarps - 1) / kWarps);
    HALOPS_CUDA_CHECK(LaunchKernelWithPdl(prefill_predq_kernel<T>, grid,
                                          dim3(PrefillPredqTraits::kThreads),
                                          kNoCluster, 0, stream, use_pdl, p));
}

// Main kernel, then the combine kernel for few pairs (p.combine).
template <class T>
void prefill_launch_attention(PrefillParams& p, const PrefillMaps& maps,
                              const PrefillWorkspace& ws, int max_clusters,
                              cudaStream_t stream,
                              void (*kernel)(PrefillParams, PrefillMaps),
                              int smem_bytes, bool use_pdl) {
    const int64_t pairs = (int64_t(p.tokens) + p.n_req) / 2;
    p.combine = 2 * pairs <= max_clusters ? 1 : 0;
    p.clusters =
        p.combine ? max_clusters : int(std::min<int64_t>(pairs, max_clusters));
    p.o_part = ws.o_part.data_ptr<float>();
    p.lse_part = ws.lse_part.data_ptr<float>();
    p.counter = ws.counter.data_ptr<int>();
    p.jobs = reinterpret_cast<PairMergeJob*>(ws.jobs.data_ptr<int>());
    p.total_g = ws.total_g.data_ptr<int>();
    p.sched_g = ws.sched_g.data_ptr<int>();
    setup_kernel_once(reinterpret_cast<const void*>(kernel),
                      {0, T::kThreads, size_t(smem_bytes)},
                      cudaLaunchConfig_t{});
    HALOPS_CUDA_CHECK(LaunchKernelWithPdl(
        kernel, dim3(2 * p.clusters), dim3(T::kThreads), dim3(2, 1, 1),
        size_t(smem_bytes), stream, use_pdl, p, maps));
    if (p.combine) {
        HALOPS_CUDA_CHECK(LaunchKernelWithPdl(
            prefill_pair_mla_combine_kernel<T>, dim3(T::kPlanes, 2, p.clusters),
            dim3(T::kCombSplit * T::kWarpGroup), kNoCluster, 0, stream, use_pdl,
            p));
    }
}

// Pair prefill attention for model dims Dims. With q_positions, q is the raw
// projection and q_freqs is complex64 [P, 32] or float32 [P, 64], (re, im)
// interleaved.
template <class Dims>
torch::Tensor prefill_pair_mla(
    const torch::Tensor& q, const torch::Tensor& swa_k_cache,
    const torch::Tensor& swa_indices, const torch::Tensor& swa_lengths,
    const torch::Tensor& c128_k_cache, const torch::Tensor& c128_indices,
    const torch::Tensor& c128_lengths,
    const std::optional<torch::Tensor>& attn_sink, const double softmax_scale,
    const torch::Tensor& cu_seqlens, const torch::Tensor& cu_pairs,
    const bool use_pdl, const int64_t swa_shift,
    const std::optional<torch::Tensor>& q_positions,
    const std::optional<torch::Tensor>& q_freqs, const double q_norm_eps) {
    using T = PrefillPairTraits<Dims>;
    const CudaDeviceGuard device_guard(q.device());
    prefill_check_q_inputs<T>(q, swa_shift, q_positions, q_freqs);
    prefill_check_kv_inputs<T>(q, swa_k_cache, swa_indices, swa_lengths,
                               c128_k_cache, c128_indices, c128_lengths,
                               attn_sink, cu_seqlens, cu_pairs);
    torch::Tensor out = torch::empty(q.sizes(), q.options());
    const int64_t tokens = q.size(0);
    if (tokens == 0) return out;
    const bool qnr = q_positions.has_value();

    PrefillParams p{};
    PrefillMaps maps{};
    prefill_set_pool<T>(p, maps, 0, c128_k_cache, c128_indices, c128_lengths);
    prefill_set_pool<T>(p, maps, 1, swa_k_cache, swa_indices, swa_lengths);
    p.q = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr());
    p.sink = attn_sink.has_value() ? attn_sink->data_ptr<float>() : nullptr;
    p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
    p.tokens = tokens;
    p.swa_shift = int(swa_shift);
    p.cu_seqlens = cu_seqlens.data_ptr<int>();
    p.cu_pairs = cu_pairs.data_ptr<int>();
    p.n_req = int(cu_seqlens.numel() - 1);
    p.scale_log2 = float(softmax_scale) * CUDART_L2E_F;
    if (qnr) {
        p.q_stride = q.stride(0);
        p.q_pos = q_positions->data_ptr();
        p.pos64 = q_positions->scalar_type() == torch::kLong;
        p.q_freqs = reinterpret_cast<const float*>(q_freqs->data_ptr());
        p.q_eps = float(q_norm_eps);
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
    TORCH_CHECK(max_clusters <= PrefillSchedTraits::kMaxClusters,
                "more SMs than the pair kernel schedules");
    const int64_t rows = prefill_predq_rows<T>(p);
    const PrefillWorkspace& ws =
        prefill_workspace<T>(q, max_clusters, rows, tokens);
    const cudaStream_t stream = GetCurrentCUDAStream(q);
    if (rows > 0)
        prefill_predequant_c128<T>(p, maps, ws, rows, stream, use_pdl);
    const bool qst = (p.ld[0] + p.ld[1] + T::kTile - 1) / T::kTile <=
                     PrefillSchedTraits::kQstMaxTiles;
    const bool fp8 = p.c128_base == nullptr;
    using Kernel = void (*)(PrefillParams, PrefillMaps);
    constexpr Kernel kKernels[2][2][2] = {
        {{prefill_pair_mla_kernel<false, false, false, Dims>,
          prefill_pair_mla_kernel<false, false, true, Dims>},
         {prefill_pair_mla_kernel<false, true, false, Dims>,
          prefill_pair_mla_kernel<false, true, true, Dims>}},
        {{prefill_pair_mla_kernel<true, false, false, Dims>,
          prefill_pair_mla_kernel<true, false, true, Dims>},
         {prefill_pair_mla_kernel<true, true, false, Dims>,
          prefill_pair_mla_kernel<true, true, true, Dims>}}};
    const Kernel kernel = kKernels[qst][fp8][qnr];
    const int smem_bytes =
        qst ? PrefillCfg<true, false, false, Dims>::kSmemBytes
            : PrefillCfg<false, false, false, Dims>::kSmemBytes;
    prefill_launch_attention<T>(p, maps, ws, max_clusters, stream, kernel,
                                smem_bytes, use_pdl);
    return out;
}

}  // namespace halops::mla

namespace halops {

// Model dims from q; only [64, 512] is instantiated.
torch::Tensor PrefillPairMla(
    const torch::Tensor& q, const torch::Tensor& swa_k_cache,
    const torch::Tensor& swa_indices, const torch::Tensor& swa_lengths,
    const torch::Tensor& c128_k_cache, const torch::Tensor& c128_indices,
    const torch::Tensor& c128_lengths,
    const std::optional<torch::Tensor>& attn_sink, const double softmax_scale,
    const torch::Tensor& cu_seqlens, const torch::Tensor& cu_pairs,
    const bool use_pdl, const int64_t swa_shift,
    const std::optional<torch::Tensor>& q_positions,
    const std::optional<torch::Tensor>& q_freqs, const double q_norm_eps) {
    const int64_t heads = q.dim() >= 3 ? q.size(-2) : 0;
    const int64_t dim = q.dim() >= 3 ? q.size(-1) : 0;
    TORCH_CHECK(heads == mla::ModelDims::Heads && dim == mla::ModelDims::DimQ,
                "prefill_pair_mla: unsupported q dims [heads, dim] = [", heads,
                ", ", dim, "], supported: [64, 512]");
    return mla::prefill_pair_mla<mla::ModelDims>(
        q, swa_k_cache, swa_indices, swa_lengths, c128_k_cache, c128_indices,
        c128_lengths, attn_sink, softmax_scale, cu_seqlens, cu_pairs, use_pdl,
        swa_shift, q_positions, q_freqs, q_norm_eps);
}

}  // namespace halops
