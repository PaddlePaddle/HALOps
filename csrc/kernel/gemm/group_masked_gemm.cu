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

void GroupMaskedGemm(const torch::Tensor& input, const torch::Tensor& weight,
                     const torch::Tensor& sfa, const torch::Tensor& sfb,
                     const torch::Tensor& masked_m, torch::Tensor& output,
                     const std::optional<torch::Tensor>& bias,
                     const std::optional<torch::Tensor>& alpha,
                     const int max_m_per_expert, const int sm_count,
                     const std::string& mode = "w8a8") {
    TORCH_CHECK(mode == "w8a8" || mode == "w4a4", "Unsupported mode: ", mode,
                ". Expected 'w8a8' or 'w4a4'.");

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

    using OutT = c10::BFloat16;
    using TypeD = cutlass::bfloat16_t;
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

    GemmParams params;
    params.masked_m = masked_m.mutable_data_ptr<int>();
    params.ptr_d = reinterpret_cast<TypeD*>(output.mutable_data_ptr<OutT>());
    params.m = m;
    params.n = n;
    params.num_experts = num_experts;
    params.max_m_per_expert = max_m_per_expert;

    if (mode == "w8a8") {
        using T = c10::Float8_e4m3fn;
        using TypeA = cutlass::float_e4m3_t;
        using TypeB = cutlass::float_e4m3_t;
        using TypeC = cutlass::bfloat16_t;
        using TypeSF = cutlass::float_ue8m0_t;
        using TypeAccumulate = float;

        const int k = weight.size(2);
        TORCH_CHECK(input.size(1) == k,
                    "input K must match weight K, got input K=", input.size(1),
                    ", weight K=", k);
        params.k = k;
        params.ptr_a = reinterpret_cast<TypeA*>(input.mutable_data_ptr<T>());
        params.ptr_b = reinterpret_cast<TypeB*>(weight.mutable_data_ptr<T>());
        params.ptr_sfa =
            reinterpret_cast<TypeSF*>(sfa.mutable_data_ptr<uint8_t>());
        params.ptr_sfb =
            reinterpret_cast<TypeSF*>(sfb.mutable_data_ptr<uint8_t>());

        if (bias.has_value()) {
            CheckGemmTileShape(max_m_per_expert, n, k, 256, 128, 128,
                               "W8A8 masked GEMM bias");
            using KernelConfig = GemmKernelConfigMaskedBias256x128x128_2CTA;
            params.ptr_c =
                reinterpret_cast<TypeC*>(bias.value().mutable_data_ptr<OutT>());
            gemm_impl<TypeA, TypeB, TypeC, TypeD, TypeSF, TypeAccumulate,
                      KernelConfig>(params, sm_count, stream);
        } else {
            if (max_m_per_expert >= 4096) {
                CheckGemmTileShape(max_m_per_expert, n, k, 256, 256, 128,
                                   "W8A8 masked GEMM large M");
                using KernelConfig =
                    GemmKernelConfigMaskedNoBias256x256x128_2CTA;
                gemm_impl<TypeA, TypeB, TypeC, TypeD, TypeSF, TypeAccumulate,
                          KernelConfig>(params, sm_count, stream);
            } else if (max_m_per_expert >= 128) {
                CheckGemmTileShape(max_m_per_expert, n, k, 128, 256, 128,
                                   "W8A8 masked GEMM mid M");
                using KernelConfig =
                    GemmKernelConfigMaskedNoBias128x256x128_1CTA;
                gemm_impl<TypeA, TypeB, TypeC, TypeD, TypeSF, TypeAccumulate,
                          KernelConfig>(params, sm_count, stream);
            } else {
                CheckDivisible(n, 128, "W8A8 masked GEMM SwapAB N");
                CheckDivisible(max_m_per_expert, 64,
                               "W8A8 masked GEMM SwapAB max_m_per_expert");
                CheckDivisible(k, 256, "W8A8 masked GEMM SwapAB K");
                using KernelConfig =
                    GemmKernelConfigMaskedNoBias128x64x256_SwapAB_1CTA;
                gemm_impl<TypeA, TypeB, TypeC, TypeD, TypeSF, TypeAccumulate,
                          KernelConfig>(params, sm_count, stream);
            }
        }
    } else {
        using TypeA = cutlass::float_e2m1_t;
        using TypeB = cutlass::float_e2m1_t;
        using TypeC = cutlass::bfloat16_t;
        using TypeSF = cutlass::float_ue4m3_t;
        using TypeAccumulate = float;
        using PackedT = uint8_t;

        // K in elements = packed_bytes * 2 (2 FP4 per byte)
        const int k = weight.size(2) * 2;
        TORCH_CHECK(
            input.size(1) == weight.size(2),
            "packed input K/2 must match packed weight K/2, got input K/2=",
            input.size(1), ", weight K/2=", weight.size(2));
        CheckDivisible(k, 64, "W4A4 masked GEMM K");
        params.k = k;
        params.ptr_a =
            reinterpret_cast<TypeA*>(input.mutable_data_ptr<PackedT>());
        params.ptr_b =
            reinterpret_cast<TypeB*>(weight.mutable_data_ptr<PackedT>());
        params.ptr_sfa =
            reinterpret_cast<TypeSF*>(sfa.mutable_data_ptr<uint8_t>());
        params.ptr_sfb =
            reinterpret_cast<TypeSF*>(sfb.mutable_data_ptr<uint8_t>());
        TORCH_CHECK(
            alpha.has_value(),
            "W4A4 mode requires alpha input (per-expert scaling factor).");
        params.ptr_alpha = alpha.value().data_ptr<float>();

        if (!bias.has_value()) {
            if (max_m_per_expert >= 4096) {
                CheckGemmTileShape(max_m_per_expert, n, k, 256, 256, 256,
                                   "W4A4 masked GEMM large M");
                using KernelConfig =
                    GemmKernelConfigW4A4MaskedNoBias256x256x256_2CTA;
                gemm_impl<TypeA, TypeB, TypeC, TypeD, TypeSF, TypeAccumulate,
                          KernelConfig>(params, sm_count, stream);
            } else if (max_m_per_expert > 128) {
                CheckGemmTileShape(max_m_per_expert, n, k, 128, 256, 256,
                                   "W4A4 masked GEMM mid M");
                using KernelConfig =
                    GemmKernelConfigW4A4MaskedNoBias128x256x256_1CTA;
                gemm_impl<TypeA, TypeB, TypeC, TypeD, TypeSF, TypeAccumulate,
                          KernelConfig>(params, sm_count, stream);
            } else {
                CheckDivisible(n, 128, "W4A4 masked GEMM SwapAB N");
                CheckDivisible(max_m_per_expert, 64,
                               "W4A4 masked GEMM SwapAB max_m_per_expert");
                CheckDivisible(k, 512, "W4A4 masked GEMM SwapAB K");
                using KernelConfig =
                    GemmKernelConfigW4A4MaskedNoBias128x64x512_1CTA;
                gemm_impl<TypeA, TypeB, TypeC, TypeD, TypeSF, TypeAccumulate,
                          KernelConfig>(params, sm_count, stream);
            }
        } else {
            TORCH_CHECK(false, "W4A4 mode does not support bias yet.");
        }
    }
}

}  // namespace halops
