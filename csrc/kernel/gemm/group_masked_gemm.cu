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
#include <string>

#include "kernel/gemm/gemm_dispatch.cuh"
#include "kernel/gemm/gemm_utils.h"

namespace halops {

void GroupMaskedGemm(const torch::Tensor& input, const torch::Tensor& weight,
                     const torch::Tensor& sfa, const torch::Tensor& sfb,
                     const torch::Tensor& masked_m, torch::Tensor& output,
                     const std::optional<torch::Tensor>& bias,
                     const std::optional<torch::Tensor>& alpha,
                     const int max_m_per_expert, const int sm_count,
                     const std::string& mode = "w8a8") {
    const at::cuda::OptionalCUDAGuard device_guard(input.device());
    TORCH_CHECK(mode == "w8a8" || mode == "w4a4-nvfp4" || mode == "w4a4-mxfp4",
                "Unsupported mode: ", mode,
                ". Expected 'w8a8', 'w4a4-nvfp4' or 'w4a4-mxfp4'.");
    TORCH_CHECK(input.dim() == 2,
                "input must be a 2D tensor of shape [num_experts * "
                "max_m_per_expert, K]");
    TORCH_CHECK(weight.dim() == 3,
                "weight must be a 3D tensor of shape [num_experts, N, K]");
    TORCH_CHECK(masked_m.dim() == 1,
                "masked_m must be a 1D tensor of shape [num_experts]");
    TORCH_CHECK(masked_m.scalar_type() == torch::kInt,
                "masked_m must have dtype torch.int32");
    TORCH_CHECK(output.dim() == 2,
                "output must be a 2D tensor of shape [M, N]");
    TORCH_CHECK(max_m_per_expert > 0,
                "max_m_per_expert must be greater than 0, got ",
                max_m_per_expert);

    const int m = input.size(0);
    const int n = weight.size(1);
    const int num_experts = weight.size(0);
    TORCH_CHECK(masked_m.size(0) == num_experts,
                "masked_m shape must be [num_experts], got ", masked_m.size(0),
                ", expected ", num_experts);
    TORCH_CHECK(m == num_experts * max_m_per_expert,
                "input M must equal num_experts * max_m_per_expert, got ", m,
                ", expected ", num_experts * max_m_per_expert);
    TORCH_CHECK(output.size(0) == m && output.size(1) == n,
                "output shape must be [M, N], got [", output.size(0), ", ",
                output.size(1), "], expected [", m, ", ", n, "]");
    const cudaStream_t stream = GetCurrentCUDAStream(input);
    const int sm = sm_count > 0 ? sm_count : GetSmCount();

    GemmParams params;
    params.masked_m = masked_m.mutable_data_ptr<int>();
    params.ptr_d = output.data_ptr();
    params.m = m;
    params.n = n;
    params.num_experts = num_experts;
    params.max_m_per_expert = max_m_per_expert;

    if (mode == "w8a8") {
        const int k = weight.size(2);
        TORCH_CHECK(input.size(1) == k,
                    "input K must match weight K, got input K=", input.size(1),
                    ", weight K=", k);
        params.k = k;
        params.ptr_a = input.data_ptr();
        params.ptr_b = weight.data_ptr();
        params.ptr_sfa = sfa.data_ptr();
        params.ptr_sfb = sfb.data_ptr();
        if (bias.has_value()) {
            params.ptr_c = bias->data_ptr();
        }
        LaunchCfg<KernelMode::Masked, MmaKind::MXFP8>(
            params,
            {.m = max_m_per_expert,
             .n = n,
             .k = k,
             .m_grid = m,
             .sm_count = sm,
             .has_bias = bias.has_value()},
            stream);
    } else {
        const int k = weight.size(2) * 2;  // 2 FP4 per packed byte
        TORCH_CHECK(
            input.size(1) == weight.size(2),
            "packed input K/2 must match packed weight K/2, got input K/2=",
            input.size(1), ", weight K/2=", weight.size(2));
        params.k = k;
        params.ptr_a = input.data_ptr();
        params.ptr_b = weight.data_ptr();
        params.ptr_sfa = sfa.data_ptr();
        params.ptr_sfb = sfb.data_ptr();
        TORCH_CHECK(!bias.has_value(), "W4A4 mode does not support bias yet.");

        const bool mxfp4 = mode == "w4a4-mxfp4";
        const int sf_k = k / (mxfp4 ? 32 : 16);
        TORCH_CHECK(sfa.size(-1) == sf_k && sfb.size(-1) == sf_k,
                    "W4A4 scale-factor shape does not match mode ",
                    mxfp4 ? "'w4a4-mxfp4' (UE8M0 per 32)"
                          : "'w4a4-nvfp4' (UE4M3 per 16)",
                    " requires SF K = ", sf_k, ", got sfa K=", sfa.size(-1),
                    " and sfb K=", sfb.size(-1), ".");

        TORCH_CHECK(!mxfp4 || !alpha.has_value(),
                    "'w4a4-mxfp4' does not take alpha; the per-expert scaling "
                    "factor is only used by 'w4a4-nvfp4'.");
        if (!mxfp4) {
            TORCH_CHECK(alpha.has_value(),
                        "W4A4-nvfp4 mode requires alpha input (per-expert "
                        "scaling factor).");
            params.ptr_alpha = alpha->data_ptr<float>();
        }
        const GemmShape shape = {.m = max_m_per_expert,
                                 .n = n,
                                 .k = k,
                                 .m_grid = m,
                                 .sm_count = sm,
                                 .has_bias = false};
        if (mxfp4) {
            LaunchCfg<KernelMode::Masked, MmaKind::MXFP4>(params, shape,
                                                          stream);
        } else {
            LaunchCfg<KernelMode::Masked, MmaKind::NVFP4>(params, shape,
                                                          stream);
        }
    }
}

}  // namespace halops
