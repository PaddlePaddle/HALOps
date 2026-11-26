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

#include <optional>
#include <vector>

#include "kernel/mla/mega_mla/vha_postmix/kernel.cuh"

namespace halops {
namespace mla {

template <class C>
void vha_postmix_check_inputs(const torch::Tensor& o, const torch::Tensor& mix,
                              const torch::Tensor& freqs,
                              const torch::Tensor& positions, int64_t groups) {
    TORCH_CHECK(o.dim() == 3 && o.size(1) == C::kHeads &&
                    o.size(2) == C::kHeadDim &&
                    o.scalar_type() == torch::kBFloat16 && o.is_contiguous(),
                "o must be a contiguous bfloat16 [T, ", C::kHeads, ", ",
                C::kHeadDim, "] tensor");
    TORCH_CHECK(mix.sizes() == torch::IntArrayRef({C::kHeads, C::kHeads}) &&
                    mix.scalar_type() == torch::kBFloat16 &&
                    mix.is_contiguous(),
                "mix must be a contiguous bfloat16 [", C::kHeads, ", ",
                C::kHeads, "] tensor");
    TORCH_CHECK(freqs.dim() == 2 && freqs.size(1) == C::kRopeDim &&
                    freqs.scalar_type() == torch::kFloat &&
                    freqs.is_contiguous(),
                "freqs must be a contiguous float32 [max_pos, ", C::kRopeDim,
                "] (cos, sin) table");
    const int tokens = o.size(0);
    TORCH_CHECK(positions.dim() == 1 && positions.size(0) == tokens &&
                    (positions.scalar_type() == torch::kInt ||
                     positions.scalar_type() == torch::kLong),
                "positions must be an int32 / int64 [T] tensor");
    TORCH_CHECK(groups > 0 && C::kHeads % groups == 0, "groups must divide ",
                C::kHeads, ", got ", groups);
}

template <class C, typename Pos>
void vha_postmix_launch(const torch::Tensor& o, const torch::Tensor& mix,
                        const torch::Tensor& freqs,
                        const torch::Tensor& positions, int groups,
                        bool group_major, torch::Tensor& q, torch::Tensor& sf,
                        bool use_pdl) {
    const int tokens = o.size(0);
    const int grid = std::max(
        std::min(tokens * C::kQuarters, GetSmCount() * C::kBlocksPerSm), 1);
    const auto kernel = vha_postmix_quant_kernel<C, Pos>;
    HALOPS_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, C::kSmemBytes));
    HALOPS_CUDA_CHECK(LaunchKernelWithPdl(
        kernel, grid, C::kThreads, kNoCluster, C::kSmemBytes,
        GetCurrentCUDAStream(o), use_pdl,
        reinterpret_cast<const __nv_bfloat16*>(o.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(mix.data_ptr()),
        freqs.data_ptr<float>(), positions.data_ptr<Pos>(), tokens, groups,
        group_major, reinterpret_cast<uint8_t*>(q.data_ptr()),
        sf.data_ptr<uint8_t>()));
}

}  // namespace mla

using namespace mla;

std::vector<torch::Tensor> VhaPostmixQuant(
    const torch::Tensor& o, const torch::Tensor& mix,
    const torch::Tensor& freqs, const torch::Tensor& positions,
    const int64_t groups, const bool group_major, const bool use_pdl,
    const std::optional<torch::Tensor>& output_quant_opt,
    const std::optional<torch::Tensor>& output_sf_opt) {
    const CudaDeviceGuard device_guard(o.device());
    using C = VhaPostmixCfg<V4VhaPostmixFormat>;
    vha_postmix_check_inputs<C>(o, mix, freqs, positions, groups);
    const int tokens = o.size(0);
    const int d = C::kHeads / groups * C::kHeadDim;
    const int rows = groups * tokens;
    const int m_sf = AlignSfM(rows);
    const std::vector<int64_t> q_shape =
        group_major ? std::vector<int64_t>{groups, tokens, d}
                    : std::vector<int64_t>{tokens, groups, d};
    torch::Tensor q =
        output_quant_opt.has_value()
            ? *output_quant_opt
            : torch::empty(q_shape, o.options().dtype(torch::kFloat8_e4m3fn));
    torch::Tensor sf = output_sf_opt.has_value()
                           ? *output_sf_opt
                           : torch::empty({m_sf, d / mxfp8::kGroup},
                                          o.options().dtype(torch::kByte));
    TORCH_CHECK(q.is_contiguous() && q.scalar_type() == torch::kFloat8_e4m3fn &&
                    q.numel() == int64_t(rows) * d,
                "output_quant must be a contiguous float8_e4m3fn tensor of ",
                rows, " x ", d, " elements");
    TORCH_CHECK(sf.is_contiguous() && sf.scalar_type() == torch::kByte &&
                    sf.numel() == int64_t(m_sf) * (d / mxfp8::kGroup),
                "output_sf must be a contiguous uint8 tensor of ", m_sf, " x ",
                d / mxfp8::kGroup, " elements");
    if (m_sf == 0) {
        return {q, sf};
    }
    if (positions.scalar_type() == torch::kInt) {
        vha_postmix_launch<C, int32_t>(o, mix, freqs, positions, groups,
                                       group_major, q, sf, use_pdl);
    } else {
        vha_postmix_launch<C, int64_t>(o, mix, freqs, positions, groups,
                                       group_major, q, sf, use_pdl);
    }
    return {q, sf};
}

}  // namespace halops
