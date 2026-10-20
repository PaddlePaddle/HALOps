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

#include <algorithm>
#include <optional>
#include <string>

#include "kernel/gemm/dense_cfgs.cuh"
#include "kernel/gemm/gemm_kernel.cuh"
#include "kernel/gemm/gemm_utils.h"
#include "kernel/gemm/kernel_traits.cuh"

namespace halops {

namespace {

void LaunchW8A8Cfg(int cfg, GemmParams& params, int sm_count,
                   cudaStream_t stream) {
    using TypeA = cutlass::float_e4m3_t;
    using TypeB = cutlass::float_e4m3_t;
    using TypeC = cutlass::bfloat16_t;
    using TypeD = cutlass::bfloat16_t;
    using TypeSF = cutlass::float_ue8m0_t;
    using TypeAccumulate = float;
    DispatchW8A8Cfg(cfg, [&]<class Tile>() {
        gemm_impl<TypeA, TypeB, TypeC, TypeD, TypeSF, TypeAccumulate, Tile>(
            params, sm_count, stream);
    });
}

void LaunchW4A4Cfg(int cfg, GemmParams& params, int sm_count,
                   cudaStream_t stream) {
    using TypeA = cutlass::float_e2m1_t;
    using TypeB = cutlass::float_e2m1_t;
    using TypeC = cutlass::bfloat16_t;
    using TypeD = cutlass::bfloat16_t;
    using TypeAccumulate = float;
    DispatchW4A4Cfg(cfg, [&]<class Tile>() {
        using TypeSF =
            cute::conditional_t<Tile::KSfVecSize == 32, cutlass::float_ue8m0_t,
                                cutlass::float_ue4m3_t>;
        gemm_impl<TypeA, TypeB, TypeC, TypeD, TypeSF, TypeAccumulate, Tile>(
            params, sm_count, stream);
    });
}

}  // namespace

void DenseGemm(const torch::Tensor& input, const torch::Tensor& weight,
               const torch::Tensor& sfa, const torch::Tensor& sfb,
               torch::Tensor& output, const std::optional<torch::Tensor>& bias,
               const std::optional<torch::Tensor>& alpha,
               const int sm_count_arg, const std::string& mode,
               const std::optional<torch::Tensor>& sfb_packed) {
    TORCH_CHECK(mode == "w8a8" || mode == "w4a4", "Unsupported mode: ", mode,
                ". Expected 'w8a8' or 'w4a4'.");
    TORCH_CHECK(input.dim() == 2, "input must be a 2D tensor of shape [M, K]");
    TORCH_CHECK(weight.dim() == 2,
                "weight must be a 2D tensor of shape [N, K]");
    TORCH_CHECK(output.dim() == 2,
                "output must be a 2D tensor of shape [M, N]");

    using OutT = c10::BFloat16;
    using TypeD = cutlass::bfloat16_t;

    const int m = input.size(0);
    const int n = weight.size(0);
    const int sm_count = sm_count_arg > 0 ? sm_count_arg : GetSmCount();
    const cudaStream_t stream = GetCurrentCUDAStream(input);

    GemmParams params;
    params.ptr_d = reinterpret_cast<TypeD*>(output.mutable_data_ptr<OutT>());
    params.m = m;
    params.n = n;
    params.num_experts = 1;
    params.max_m_per_expert = m;

    if (mode == "w8a8") {
        using T = c10::Float8_e4m3fn;
        using TypeA = cutlass::float_e4m3_t;
        using TypeB = cutlass::float_e4m3_t;
        using TypeC = cutlass::bfloat16_t;
        using TypeSF = cutlass::float_ue8m0_t;
        using TypeAccumulate = float;

        const int k = weight.size(1);
        TORCH_CHECK(input.size(1) == k,
                    "input K must match weight K, got input K=", input.size(1),
                    ", weight K=", k);
        TORCH_CHECK(output.size(0) == m && output.size(1) == n,
                    "output shape must be [M, N], got [", output.size(0), ", ",
                    output.size(1), "], expected [", m, ", ", n, "]");
        params.k = k;
        params.ptr_a = reinterpret_cast<TypeA*>(input.mutable_data_ptr<T>());
        params.ptr_b = reinterpret_cast<TypeB*>(weight.mutable_data_ptr<T>());
        params.ptr_sfa =
            reinterpret_cast<TypeSF*>(sfa.mutable_data_ptr<uint8_t>());
        params.ptr_sfb =
            reinterpret_cast<TypeSF*>(sfb.mutable_data_ptr<uint8_t>());

        // A prepacked weight-SFB tensor has two 128-row SfAtom atoms per
        // logical 224-column tile.
        const int packed_sfb_rows = ((n + 223) / 224) * 256;
        auto is_packed_sfb = [&](const torch::Tensor& t) {
            return t.dim() == 2 && t.size(0) == packed_sfb_rows &&
                   t.size(1) == k / 32 && t.scalar_type() == torch::kUInt8 &&
                   t.is_contiguous();
        };
        const bool has_packed_sfb =
            sfb_packed.has_value() && sfb_packed->defined();
        if (has_packed_sfb) {
            TORCH_CHECK(is_packed_sfb(*sfb_packed),
                        "sfb_packed must be pack_sfb_224(sfb, N, K)");
        }

        if (bias.has_value()) {
            CheckGemmTileShape(m, n, k, 256, 128, 128, "W8A8 GEMM bias");
            using KernelConfig = GemmKernelConfigNaiveBias256x128x128_2CTA;
            params.ptr_c =
                reinterpret_cast<TypeC*>(bias.value().mutable_data_ptr<OutT>());
            gemm_impl<TypeA, TypeB, TypeC, TypeD, TypeSF, TypeAccumulate,
                      KernelConfig>(params, sm_count, stream);
        } else if (m <= 64 &&
                   (n >= 16384 || (m <= 16 && n >= 8192 && k >= 4096))) {
            // Small-M SwapAB tiles
            if (m <= 16) {
                CheckDivisible(k, 256, "W8A8 GEMM SwapAB K");
                using KernelConfig =
                    GemmKernelConfigNaiveNoBias128x16x256_SwapAB_1CTA;
                gemm_impl<TypeA, TypeB, TypeC, TypeD, TypeSF, TypeAccumulate,
                          KernelConfig>(params, sm_count, stream);
            } else {
                CheckDivisible(k, 256, "W8A8 GEMM SwapAB K");
                using KernelConfig =
                    GemmKernelConfigNaiveNoBias128x64x256_SwapAB_1CTA;
                gemm_impl<TypeA, TypeB, TypeC, TypeD, TypeSF, TypeAccumulate,
                          KernelConfig>(params, sm_count, stream);
            }
        } else {
            // The cost model runs at the tile's own M
            const int m_cost = std::max(m, 128);
            const int cfg =
                (has_packed_sfb && sm_count == 152 && m == 1024 && n == 4096 &&
                 k >= 1024 && k <= 2048 && k % 128 == 0)
                    ? W8A8TileIndex<
                          GemmKernelConfig128x224x128_1CTA_PackedSfb>()
                : (has_packed_sfb && m == 8192 && n == 4096 && k == 1536)
                    ? W8A8TileIndex<
                          GemmKernelConfigNaiveNoBias256x224x128_2CTA_PackedSfb_Reg160_Stage7>()
                : (m == 4096 && n == 4096 && k == 4096 && !has_packed_sfb)
                    ? W8A8TileIndex<
                          GemmKernelConfigNaiveNoBias256x192x128_2CTA_Reg160>()
                    : SelectDenseCfg(m_cost, n, k, sm_count, has_packed_sfb);
            TORCH_CHECK(cfg >= 0,
                        "No W8A8 dense GEMM tile shape supports M=", m,
                        ", N=", n, ", K=", k);
            TORCH_CHECK(CfgDivides(kDenseCfgs[cfg], m, n, k),
                        "W8A8 GEMM requires a shape supported by cfg ", cfg);

            const int launch_cfg =
                cfg == W8A8TileIndex<
                           GemmKernelConfigNaiveNoBias256x192x128_2CTA>()
                    ? W8A8TileIndex<
                          GemmKernelConfigNaiveNoBias256x192x128_2CTA_Reg160>()
                    : cfg;
            if (kDenseCfgHeuristicPacked[launch_cfg]) {
                TORCH_CHECK(has_packed_sfb, "W8A8 GEMM cfg ", launch_cfg,
                            " requires sfb_packed");
                params.ptr_sfb = reinterpret_cast<TypeSF*>(
                    sfb_packed->mutable_data_ptr<uint8_t>());
            }
            LaunchW8A8Cfg(launch_cfg, params, sm_count, stream);
        }
    } else {
        using TypeA = cutlass::float_e2m1_t;
        using TypeB = cutlass::float_e2m1_t;
        using TypeC = cutlass::bfloat16_t;
        using TypeSF = cutlass::float_ue4m3_t;
        using TypeAccumulate = float;
        using PackedT = uint8_t;

        // K in elements = packed_bytes * 2 (2 FP4 per byte)
        const int k = weight.size(1) * 2;
        TORCH_CHECK(
            input.size(1) == weight.size(1),
            "packed input K/2 must match packed weight K/2, got input K/2=",
            input.size(1), ", weight K/2=", weight.size(1));
        TORCH_CHECK(output.size(0) == m && output.size(1) == n,
                    "output shape must be [M, N], got [", output.size(0), ", ",
                    output.size(1), "], expected [", m, ", ", n, "]");
        CheckDivisible(k, 64, "W4A4 GEMM K");
        params.k = k;
        params.ptr_a =
            reinterpret_cast<TypeA*>(input.mutable_data_ptr<PackedT>());
        params.ptr_b =
            reinterpret_cast<TypeB*>(weight.mutable_data_ptr<PackedT>());
        params.ptr_sfa =
            reinterpret_cast<TypeSF*>(sfa.mutable_data_ptr<uint8_t>());
        params.ptr_sfb =
            reinterpret_cast<TypeSF*>(sfb.mutable_data_ptr<uint8_t>());

        params.ptr_alpha =
            alpha.has_value() ? alpha.value().data_ptr<float>() : nullptr;

        TORCH_CHECK(!bias.has_value(), "W4A4 mode does not support bias yet.");

        const bool mxfp4 = !alpha.has_value();
        const int sf_k_expected = k / (mxfp4 ? 32 : 16);
        TORCH_CHECK(
            sfa.size(1) == sf_k_expected && sfb.size(1) == sf_k_expected,
            "W4A4 scale-factor shape does not match the recipe selected "
            "by alpha: ",
            mxfp4 ? "alpha absent -> MXFP4 (UE8M0 per 32)"
                  : "alpha present -> NVFP4 (UE4M3 per 16)",
            " requires SF K = ", sf_k_expected, ", got sfa K=", sfa.size(1),
            " and sfb K=", sfb.size(1),
            ". Pass alpha for NVFP4, omit it for MXFP4.");

        const int w4a4_cfg = SelectW4A4Cfg(m, n, k, sm_count, mxfp4);
        TORCH_CHECK(
            w4a4_cfg >= 0 && W4A4CfgDivides(kW4A4Cfgs[w4a4_cfg], m, n, k),
            "No W4A4 dense GEMM tile shape supports M=", m, ", N=", n,
            ", K=", k);
        LaunchW4A4Cfg(w4a4_cfg, params, sm_count, stream);
    }
}

}  // namespace halops
