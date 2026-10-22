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

void BatchGemm(const torch::Tensor& input, const torch::Tensor& weight,
               const torch::Tensor& sfa, const torch::Tensor& sfb,
               torch::Tensor& output, const std::optional<torch::Tensor>& bias,
               const std::optional<torch::Tensor>& alpha, const int sm_count,
               const std::string& mode = "w8a8") {
    const at::cuda::OptionalCUDAGuard device_guard(input.device());
    TORCH_CHECK(mode == "w8a8" || mode == "w4a4-nvfp4" || mode == "w4a4-mxfp4",
                "Unsupported mode: ", mode,
                ". Expected 'w8a8', 'w4a4-nvfp4' or 'w4a4-mxfp4'.");
    TORCH_CHECK(input.dim() == 3,
                "input must be a 3D tensor of shape [B, M, K]");
    TORCH_CHECK(weight.dim() == 3,
                "weight must be a 3D tensor of shape [B, N, K]");
    TORCH_CHECK(output.dim() == 3,
                "output must be a 3D tensor of shape [B, M, N]");

    const int batches = input.size(0);
    const int m = input.size(1);
    const int n = weight.size(1);
    TORCH_CHECK(weight.size(0) == batches,
                "weight batch dim must match input batch dim, got ",
                weight.size(0));
    TORCH_CHECK(
        output.size(0) == batches && output.size(1) == m && output.size(2) == n,
        "output shape must be [B, M, N]");

    // sfa/sfb: [B*M, sf_k] or [B, M, sf_k] -- same underlying memory.

    const cudaStream_t stream = GetCurrentCUDAStream(input);
    const int sm = sm_count > 0 ? sm_count : GetSmCount();

    // expert_ids stays null: the scheduler derives experts from
    // max_m_per_expert arithmetic.
    GemmParams params;
    params.ptr_d = output.data_ptr();
    params.m = batches * m;
    params.n = n;
    params.num_experts = batches;
    params.max_m_per_expert = m;

    if (mode == "w8a8") {
        const int k = weight.size(2);
        TORCH_CHECK(input.size(2) == k,
                    "input K must match weight K, got input K=", input.size(2),
                    ", weight K=", k);
        params.k = k;
        params.ptr_a = input.data_ptr();
        params.ptr_b = weight.data_ptr();
        params.ptr_sfa = sfa.data_ptr();
        params.ptr_sfb = sfb.data_ptr();
        if (bias.has_value()) {
            params.ptr_c = bias->data_ptr();
        }
        LaunchCfg<KernelMode::Batch, MmaKind::MXFP8>(
            params,
            {.m = m,
             .n = n,
             .k = k,
             .m_grid = m * batches,
             .sm_count = sm,
             .has_bias = bias.has_value()},
            stream);
    } else {
        const int k = weight.size(2) * 2;  // 2 FP4 per packed byte
        TORCH_CHECK(input.size(2) == weight.size(2),
                    "packed input K/2 must match packed weight K/2, got ",
                    input.size(2), ", weight K/2=", weight.size(2));
        params.k = k;
        params.ptr_a = input.data_ptr();
        params.ptr_b = weight.data_ptr();
        params.ptr_sfa = sfa.data_ptr();
        params.ptr_sfb = sfb.data_ptr();
        TORCH_CHECK(!bias.has_value(),
                    "W4A4 batch GEMM does not support bias yet.");

        const bool mxfp4 = mode == "w4a4-mxfp4";
        const int sf_k = k / (mxfp4 ? 32 : 16);
        TORCH_CHECK(sfa.size(-1) == sf_k && sfb.size(-1) == sf_k,
                    "W4A4 scale-factor shape does not match mode ",
                    mxfp4 ? "'w4a4-mxfp4' (UE8M0 per 32)"
                          : "'w4a4-nvfp4' (UE4M3 per 16)",
                    " requires SF K = ", sf_k, ", got sfa K=", sfa.size(-1),
                    " and sfb K=", sfb.size(-1), ".");

        TORCH_CHECK(!mxfp4 || !alpha.has_value(),
                    "'w4a4-mxfp4' does not take alpha; the per-batch scaling "
                    "factor is only used by 'w4a4-nvfp4'.");
        if (!mxfp4) {
            TORCH_CHECK(alpha.has_value(),
                        "W4A4-nvfp4 mode requires alpha input (per-batch scaling).");
            params.ptr_alpha = alpha->data_ptr<float>();
        }
        const GemmShape shape = {.m = m,
                                 .n = n,
                                 .k = k,
                                 .m_grid = m * batches,
                                 .sm_count = sm,
                                 .has_bias = false};
        if (mxfp4) {
            LaunchCfg<KernelMode::Batch, MmaKind::MXFP4>(params, shape, stream);
        } else {
            LaunchCfg<KernelMode::Batch, MmaKind::NVFP4>(params, shape, stream);
        }
    }
}

}  // namespace halops
