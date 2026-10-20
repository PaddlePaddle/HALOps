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

#include "kernel/gemm/gemm_kernel.cuh"
#include "kernel/gemm/gemm_utils.h"
#include "kernel/gemm/kernel_traits.cuh"

namespace halops {

void BatchGemm(const torch::Tensor& input, const torch::Tensor& weight,
               const torch::Tensor& sfa, const torch::Tensor& sfb,
               torch::Tensor& output, const std::optional<torch::Tensor>& bias,
               const std::optional<torch::Tensor>& alpha, const int sm_count,
               const std::string& mode = "w8a8") {
    TORCH_CHECK(mode == "w8a8" || mode == "w4a4", "Unsupported mode: ", mode,
                ". Expected 'w8a8' or 'w4a4'.");
    TORCH_CHECK(input.dim() == 3,
                "input must be a 3D tensor of shape [B, M, K]");
    TORCH_CHECK(weight.dim() == 3,
                "weight must be a 3D tensor of shape [B, N, K]");
    TORCH_CHECK(output.dim() == 3,
                "output must be a 3D tensor of shape [B, M, N]");

    const int B = input.size(0);
    const int M = input.size(1);
    const int N = weight.size(1);
    TORCH_CHECK(weight.size(0) == B,
                "weight batch dim must match input batch dim, got ",
                weight.size(0));
    TORCH_CHECK(
        output.size(0) == B && output.size(1) == M && output.size(2) == N,
        "output shape must be [B, M, N]");

    // sfa/sfb may be [B*M, sf_k] or [B, M, sf_k]; the memory is the same.
    TORCH_CHECK(sfa.is_contiguous(), "sfa must be contiguous");
    TORCH_CHECK(sfb.is_contiguous(), "sfb must be contiguous");

    const int total_m = B * M;

    using OutT = c10::BFloat16;
    using TypeD = cutlass::bfloat16_t;
    const cudaStream_t stream = GetCurrentCUDAStream(input);

    // No expert_ids: expert index is global_m_idx / max_m_per_expert.
    GemmParams params;
    params.ptr_d = reinterpret_cast<TypeD*>(output.mutable_data_ptr<OutT>());
    params.m = total_m;
    params.n = N;
    params.num_experts = B;
    params.max_m_per_expert = M;
    params.expert_ids = nullptr;

    if (mode == "w8a8") {
        using T = c10::Float8_e4m3fn;
        using TypeA = cutlass::float_e4m3_t;
        using TypeB = cutlass::float_e4m3_t;
        using TypeC = cutlass::bfloat16_t;
        using TypeSF = cutlass::float_ue8m0_t;
        using TypeAccumulate = float;

        const int k = weight.size(2);
        TORCH_CHECK(input.size(2) == k,
                    "input K must match weight K, got input K=", input.size(2),
                    ", weight K=", k);
        params.k = k;
        params.ptr_a = reinterpret_cast<TypeA*>(input.mutable_data_ptr<T>());
        params.ptr_b = reinterpret_cast<TypeB*>(weight.mutable_data_ptr<T>());
        params.ptr_sfa =
            reinterpret_cast<TypeSF*>(sfa.mutable_data_ptr<uint8_t>());
        params.ptr_sfb =
            reinterpret_cast<TypeSF*>(sfb.mutable_data_ptr<uint8_t>());

        if (bias.has_value()) {
            CheckGemmTileShape(total_m, N, k, 256, 128, 128,
                               "W8A8 batch GEMM bias");
            using KernelConfig = GemmKernelConfigBatchBias256x128x128_2CTA;
            params.ptr_c =
                reinterpret_cast<TypeC*>(bias.value().mutable_data_ptr<OutT>());
            gemm_impl<TypeA, TypeB, TypeC, TypeD, TypeSF, TypeAccumulate,
                      KernelConfig>(params, sm_count, stream);
        } else {
            if (total_m >= 4096) {
                CheckGemmTileShape(total_m, N, k, 256, 256, 128,
                                   "W8A8 batch GEMM large M");
                using KernelConfig =
                    GemmKernelConfigBatchNoBias256x256x128_2CTA;
                gemm_impl<TypeA, TypeB, TypeC, TypeD, TypeSF, TypeAccumulate,
                          KernelConfig>(params, sm_count, stream);
            } else if (total_m >= 128) {
                CheckGemmTileShape(total_m, N, k, 128, 256, 128,
                                   "W8A8 batch GEMM mid M");
                using KernelConfig =
                    GemmKernelConfigBatchNoBias128x256x128_1CTA;
                gemm_impl<TypeA, TypeB, TypeC, TypeD, TypeSF, TypeAccumulate,
                          KernelConfig>(params, sm_count, stream);
            } else {
                CheckDivisible(N, 128, "W8A8 batch GEMM SwapAB N");
                CheckDivisible(total_m, 64, "W8A8 batch GEMM SwapAB M");
                CheckDivisible(k, 256, "W8A8 batch GEMM SwapAB K");
                using KernelConfig =
                    GemmKernelConfigBatchNoBias128x64x256_SwapAB_1CTA;
                gemm_impl<TypeA, TypeB, TypeC, TypeD, TypeSF, TypeAccumulate,
                          KernelConfig>(params, sm_count, stream);
            }
        }
    } else {  // w4a4
        using TypeA = cutlass::float_e2m1_t;
        using TypeB = cutlass::float_e2m1_t;
        using TypeC = cutlass::bfloat16_t;
        using TypeSF = cutlass::float_ue4m3_t;
        using TypeAccumulate = float;
        using PackedT = uint8_t;

        const int k = weight.size(2) * 2;  // packed FP4
        TORCH_CHECK(input.size(2) == weight.size(2),
                    "packed input K/2 must match packed weight K/2, got ",
                    input.size(2), ", weight K/2=", weight.size(2));
        CheckDivisible(k, 64, "W4A4 batch GEMM K");
        params.k = k;
        params.ptr_a =
            reinterpret_cast<TypeA*>(input.mutable_data_ptr<PackedT>());
        params.ptr_b =
            reinterpret_cast<TypeB*>(weight.mutable_data_ptr<PackedT>());
        params.ptr_sfa =
            reinterpret_cast<TypeSF*>(sfa.mutable_data_ptr<uint8_t>());
        params.ptr_sfb =
            reinterpret_cast<TypeSF*>(sfb.mutable_data_ptr<uint8_t>());

        TORCH_CHECK(alpha.has_value(),
                    "W4A4 mode requires alpha input (per-batch scaling).");
        params.ptr_alpha = alpha.value().data_ptr<float>();

        if (!bias.has_value()) {
            if (total_m >= 4096) {
                CheckGemmTileShape(total_m, N, k, 256, 256, 256,
                                   "W4A4 batch GEMM large M");
                using KernelConfig =
                    GemmKernelConfigW4A4BatchNoBias256x256x256_2CTA;
                gemm_impl<TypeA, TypeB, TypeC, TypeD, TypeSF, TypeAccumulate,
                          KernelConfig>(params, sm_count, stream);
            } else if (total_m >= 128) {
                CheckGemmTileShape(total_m, N, k, 128, 256, 256,
                                   "W4A4 batch GEMM mid M");
                using KernelConfig =
                    GemmKernelConfigW4A4BatchNoBias128x256x256_1CTA;
                gemm_impl<TypeA, TypeB, TypeC, TypeD, TypeSF, TypeAccumulate,
                          KernelConfig>(params, sm_count, stream);
            } else {
                CheckDivisible(N, 128, "W4A4 batch GEMM SwapAB N");
                CheckDivisible(total_m, 64, "W4A4 batch GEMM SwapAB M");
                CheckDivisible(k, 512, "W4A4 batch GEMM SwapAB K");
                using KernelConfig =
                    GemmKernelConfigW4A4BatchNoBias128x64x512_1CTA;
                gemm_impl<TypeA, TypeB, TypeC, TypeD, TypeSF, TypeAccumulate,
                          KernelConfig>(params, sm_count, stream);
            }
        } else {
            TORCH_CHECK(false, "W4A4 batch GEMM does not support bias yet.");
        }
    }
}

}  // namespace halops
