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

#include "kernel/gemm/low_latency_mma_gemm.cuh"

namespace halops {

void LowLatencyMmaGemm(const torch::Tensor& input, const torch::Tensor& weight,
                       const torch::Tensor& sfa, const torch::Tensor& sfb,
                       torch::Tensor& output, const int sm_count,
                       const std::optional<torch::Tensor>& output2,
                       const int n_split,
                       const std::optional<torch::Tensor>& input2,
                       const std::optional<torch::Tensor>& sfa2,
                       const bool use_pdl) {
    const CudaDeviceGuard device_guard(input.device());

    ResolveLlShapes(input, weight, sfa, sfb, output, output2, n_split, input2,
                    sfa2);
    const int m = input.size(0);
    const int n = weight.size(0);
    const int k = weight.size(1);

    const int sms = sm_count > 0 ? sm_count : GetSmCount();
    const int rows_tier = m <= kLlRowsTiers[0] ? 0 : 1;
    const int launcher =
        SelectLlTile(n, k, rows_tier, output2.has_value() ? n_split : 0, sms);
    TORCH_CHECK(launcher >= 0, "low_latency_mma_gemm: the resident A (",
                kLlRowsTiers[rows_tier], " x ", k,
                " fp8) leaves no room for a B stage in shared memory; use "
                "dense_gemm for this shape");

    LowLatencyMmaGemmParams params;
    params.ptr_a = input.const_data_ptr<c10::Float8_e4m3fn>();
    params.ptr_b = weight.const_data_ptr<c10::Float8_e4m3fn>();
    params.ptr_sfa = sfa.const_data_ptr<uint8_t>();
    params.ptr_sfb = sfb.const_data_ptr<uint8_t>();
    params.ptr_d = output.mutable_data_ptr<c10::BFloat16>();
    params.m = m;
    params.n = n;
    params.k = k;
    params.sfa_stride_mn = sfa.stride(0);
    params.sfb_stride_mn = sfb.stride(0);
    if (input2.has_value()) {
        params.ptr_a2 = input2->const_data_ptr<c10::Float8_e4m3fn>();
        params.ptr_sfa2 = sfa2->const_data_ptr<uint8_t>();
        params.k1 = input.size(1);
        params.sfa2_stride_mn = sfa2->stride(0);
    }
    if (output2.has_value()) {
        params.ptr_d2 = output2->mutable_data_ptr<c10::BFloat16>();
        params.n_split = n_split;
    }

    kLlLaunchers[launcher](params, sms, use_pdl, GetCurrentCUDAStream(input));
    HALOPS_CUDA_CHECK(cudaGetLastError());
}

}  // namespace halops
