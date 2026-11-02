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
                       torch::Tensor& output,
                       const std::optional<torch::Tensor>& input2,
                       const std::optional<torch::Tensor>& weight2,
                       const std::optional<torch::Tensor>& sfa2,
                       const std::optional<torch::Tensor>& sfb2,
                       const std::optional<torch::Tensor>& output2,
                       const int sm_count, const bool use_pdl) {
    const CudaDeviceGuard device_guard(input.device());
    ResolveLlShapes(input, weight, sfa, sfb, input2, weight2, sfa2, sfb2,
                    output2);

    LlGemmParams params;
    params.ptr_a = input.const_data_ptr<c10::Float8_e4m3fn>();
    params.ptr_b = weight.const_data_ptr<c10::Float8_e4m3fn>();
    params.ptr_sfa = sfa.const_data_ptr<uint8_t>();
    params.ptr_sfb = sfb.const_data_ptr<uint8_t>();
    params.ptr_d = output.mutable_data_ptr<c10::BFloat16>();
    params.m = input.size(0);
    params.n_split = weight.size(0);
    params.n = params.n_split + (output2.has_value() ? weight2->size(0) : 0);
    params.k1 = weight.size(1);
    params.k = params.k1 + (input2.has_value() ? weight2->size(1) : 0);
    params.sfa_stride_mn = sfa.stride(0);
    params.sfb_stride_mn = sfb.stride(0);
    if (weight2.has_value()) {
        params.ptr_b2 = weight2->const_data_ptr<c10::Float8_e4m3fn>();
        params.ptr_sfb2 = sfb2->const_data_ptr<uint8_t>();
        params.sfb2_stride_mn = sfb2->stride(0);
    }
    if (input2.has_value()) {
        params.ptr_a2 = input2->const_data_ptr<c10::Float8_e4m3fn>();
        params.ptr_sfa2 = sfa2->const_data_ptr<uint8_t>();
        params.sfa2_stride_mn = sfa2->stride(0);
    }
    if (output2.has_value()) {
        params.ptr_d2 = output2->mutable_data_ptr<c10::BFloat16>();
    }

    const int num_sms = sm_count > 0 ? sm_count : GetSmCount();
    const int rows_a = params.m > LlGemmConfig::KAtomRows
                           ? LlGemmConfig::KMaxM
                           : LlGemmConfig::KAtomRows;
    const LlTileSpec tile = SelectLlTile(params.n, params.k, params.k1, rows_a,
                                         params.n_split, num_sms);
    TORCH_CHECK(tile.stages > 0, "low_latency_mma_gemm: K=", params.k,
                " is too large to keep A resident in shared memory");

    LaunchLlKernel(params, tile, rows_a, num_sms, use_pdl,
                   GetCurrentCUDAStream(input));
    HALOPS_CUDA_CHECK(cudaGetLastError());
}

}  // namespace halops
