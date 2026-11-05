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

void DenseGemm(const torch::Tensor& input, const torch::Tensor& weight,
               const std::optional<torch::Tensor>& sfa,
               const std::optional<torch::Tensor>& sfb, torch::Tensor& output,
               const std::optional<torch::Tensor>& bias,
               const std::optional<torch::Tensor>& alpha,
               const int sm_count_arg, const std::string& mode,
               const bool use_pdl, const EpilogueOp epilogue,
               const std::optional<torch::Tensor>& output_sf,
               const double situ_beta,
               const std::optional<double> situ_linear_beta) {
    const CudaDeviceGuard device_guard(input.device());
    TORCH_CHECK(
        mode == "bf16" || mode == "w8a8-mxfp8" || mode == "w4a4-mxfp4" ||
            mode == "w4a4-nvfp4",
        "Unsupported mode: ", mode,
        ". Expected 'bf16', 'w8a8-mxfp8', 'w4a4-mxfp4' or 'w4a4-nvfp4'.");
    TORCH_CHECK(input.dim() == 2, "input must be a 2D tensor of shape [M, K]");
    TORCH_CHECK(weight.dim() == 2,
                "weight must be a 2D tensor of shape [N, K]");
    TORCH_CHECK(output.dim() == 2,
                "output must be a 2D tensor of shape [M, N]");

    const int m = input.size(0);
    const int n = weight.size(0);
    const int out_n = epilogue == EpilogueOp::SituQuant ? n / 2 : n;
    TORCH_CHECK(output.size(0) == m && output.size(1) == out_n,
                "output shape must be [", m, ", ", out_n, "], got [",
                output.size(0), ", ", output.size(1), "]");
    const int sm = sm_count_arg > 0 ? sm_count_arg : GetSmCount();
    const cudaStream_t stream = GetCurrentCUDAStream(input);

    const int m_sf = AlignSfM(m);
    // still need to pass a rounded-up M to the kernel when using the bias tile.
    const bool bias_tile =
        bias.has_value() && epilogue != EpilogueOp::SituQuant;
    const int m_shape = (m < kSfAtomM || bias_tile) ? m : m_sf;

    GemmParams params;
    params.ptr_d = output.data_ptr();
    params.m = m;
    params.n = n;
    params.num_experts = 1;
    params.max_m_per_expert = m;
    params.use_pdl = use_pdl;
    params.situ = {
        .inv_beta = static_cast<float>(1.0 / situ_beta),
        .gate_up_scale =
            static_cast<float>(situ_beta * situ_linear_beta.value_or(1.0)),
        .inv_linear_beta = situ_linear_beta
                               ? static_cast<float>(1.0 / *situ_linear_beta)
                               : 0.0f};

    if (mode == "bf16") {
        // Native BF16: sfa/sfb/bias/alpha are ignored; the tile ladder,
        // validation and launch live in gemm_dispatch.cuh (bf16::LaunchDense).
        TORCH_CHECK(input.dtype() == torch::kBFloat16,
                    "bf16 dense GEMM: input must be bfloat16");
        TORCH_CHECK(weight.dtype() == torch::kBFloat16,
                    "bf16 dense GEMM: weight must be bfloat16");
        TORCH_CHECK(output.dtype() == torch::kBFloat16,
                    "bf16 dense GEMM: output must be bfloat16");
        TORCH_CHECK(epilogue == EpilogueOp::Identity && !output_sf.has_value(),
                    "bf16 dense GEMM supports EpilogueOp.Identity only");

        const int k = weight.size(1);
        TORCH_CHECK(input.size(1) == k,
                    "input K must match weight K, got input K=", input.size(1),
                    ", weight K=", k);
        params.k = k;
        params.ptr_a = input.data_ptr();
        params.ptr_b = weight.data_ptr();
        params.ptr_sfa = nullptr;
        params.ptr_sfb = nullptr;
        params.ptr_c = nullptr;
        params.ptr_alpha = nullptr;

        // Entry guards per tier (friendlier messages than the per-tile checks).
        if (m >= 128 && m < 4096) {
            CheckDivisible(m, 128, "BF16 GEMM mid M M");
            CheckDivisible(n, 32, "BF16 GEMM mid M N");
            CheckDivisible(k, 64, "BF16 GEMM mid M K");
        } else if (m < 128 && !bf16::SkipSwapForThinM(m, n, k, sm)) {
            CheckDivisible(n, 128, "BF16 GEMM SwapAB N");
            CheckDivisible(m, 64, "BF16 GEMM SwapAB M");
            CheckDivisible(k, 64, "BF16 GEMM SwapAB K");
        }
        bf16::LaunchDense(params, sm, stream);
    } else if (mode == "w8a8-mxfp8") {
        TORCH_CHECK(sfa.has_value() && sfb.has_value(),
                    "w8a8-mxfp8 dense GEMM requires sfa/sfb scale factors "
                    "(only mode='bf16' accepts sfa=None, sfb=None)");
        const int k = weight.size(1);
        TORCH_CHECK(input.size(1) == k,
                    "input K must match weight K, got input K=", input.size(1),
                    ", weight K=", k);
        params.k = k;
        params.ptr_a = input.data_ptr();
        params.ptr_b = weight.data_ptr();
        params.ptr_sfa = sfa->data_ptr();
        // sfb is the quantizer's uint8 SfAtom weight scale, used as-is: a
        // 224-wide tile reads it as a run of atoms plus a 32-row rewrite.
        TORCH_CHECK(sfb->scalar_type() == torch::kUInt8,
                    "sfb must be the uint8 SfAtom weight scale that "
                    "group_wise_quant returns");
        params.ptr_sfb = sfb->data_ptr();
        if (bias.has_value()) {
            params.ptr_c = bias->data_ptr();
        }
        TORCH_CHECK(
            sfa->is_contiguous() && sfa->numel() >= int64_t(m_sf) * (k / 32),
            "sfa must hold whole 128-row SfAtoms: expected at least ",
            int64_t(m_sf) * (k / 32), " elements (", m_sf, " x ", k / 32,
            ") for M=", m, ", got ", sfa->numel());
        TORCH_CHECK(!bias_tile || m % 256 == 0,
                    "the bias tile has no per-row store predicate yet, so M "
                    "must be a multiple of 256 when bias is given, got ",
                    m);
        const GemmShape shape = {.m = m_shape,
                                 .n = n,
                                 .k = k,
                                 .m_grid = m_shape,
                                 .sm_count = sm,
                                 .has_bias = bias_tile};
        if (output_sf.has_value()) {
            params.ptr_sfd = output_sf->data_ptr();
        }
        switch (epilogue) {
            case EpilogueOp::Identity:
                LaunchCfg<KernelMode::Dense, MmaKind::MXFP8>(params, shape,
                                                             stream);
                break;
            case EpilogueOp::Sigmoid:
                LaunchCfg<KernelMode::Dense, MmaKind::MXFP8,
                          EpilogueOp::Sigmoid>(params, shape, stream);
                break;
            case EpilogueOp::SituQuant:
                LaunchCfg<KernelMode::Dense, MmaKind::MXFP8,
                          EpilogueOp::SituQuant>(params, shape, stream);
                break;
            default:
                TORCH_CHECK(false, "Unsupported epilogue ",
                            static_cast<int>(epilogue), " for ", mode);
        }
    } else {
        TORCH_CHECK(sfa.has_value() && sfb.has_value(),
                    "W4A4 dense GEMM requires sfa/sfb scale factors "
                    "(only mode='bf16' accepts sfa=None, sfb=None)");
        const int k = weight.size(1) * 2;  // 2 FP4 per packed byte
        TORCH_CHECK(
            input.size(1) == weight.size(1),
            "packed input K/2 must match packed weight K/2, got input K/2=",
            input.size(1), ", weight K/2=", weight.size(1));
        params.k = k;
        params.ptr_a = input.data_ptr();
        params.ptr_b = weight.data_ptr();
        params.ptr_sfa = sfa->data_ptr();
        params.ptr_sfb = sfb->data_ptr();
        TORCH_CHECK(!bias.has_value(), "W4A4 mode does not support bias yet.");

        const bool mxfp4 = mode == "w4a4-mxfp4";
        TORCH_CHECK(!mxfp4 || !alpha.has_value(),
                    "'w4a4-mxfp4' does not take alpha; the global scaling "
                    "factor is only used by 'w4a4-nvfp4'.");
        params.ptr_alpha =
            alpha.has_value() ? alpha->data_ptr<float>() : nullptr;
        const int sf_k = k / (mxfp4 ? 32 : 16);
        TORCH_CHECK(sfa->size(-1) == sf_k && sfb->size(-1) == sf_k,
                    "W4A4 scale-factor shape does not match mode ",
                    mxfp4 ? "'w4a4-mxfp4' (UE8M0 per 32)"
                          : "'w4a4-nvfp4' (UE4M3 per 16)",
                    " requires SF K = ", sf_k, ", got sfa K=", sfa->size(-1),
                    " and sfb K=", sfb->size(-1), ".");
        TORCH_CHECK(
            sfa->is_contiguous() && sfa->numel() >= int64_t(m_sf) * sf_k,
            "sfa must hold whole 128-row SfAtoms: expected at least ",
            int64_t(m_sf) * sf_k, " elements (", m_sf, " x ", sf_k,
            ") for M=", m, ", got ", sfa->numel());
        TORCH_CHECK(!mxfp4 || !alpha.has_value(),
                    "'w4a4-mxfp4' does not take alpha; the per-batch scaling "
                    "factor is only used by 'w4a4-nvfp4'.");
        if (!mxfp4) {
            TORCH_CHECK(
                alpha.has_value(),
                "W4A4-nvfp4 mode requires alpha input (per-batch scaling).");
            params.ptr_alpha = alpha->data_ptr<float>();
        }
        const GemmShape shape = {.m = m_shape,
                                 .n = n,
                                 .k = k,
                                 .m_grid = m_shape,
                                 .sm_count = sm,
                                 .has_bias = false};
        switch (epilogue) {
            case EpilogueOp::Identity:
                mxfp4 ? LaunchCfg<KernelMode::Dense, MmaKind::MXFP4>(
                            params, shape, stream)
                      : LaunchCfg<KernelMode::Dense, MmaKind::NVFP4>(
                            params, shape, stream);
                break;
            case EpilogueOp::Sigmoid:
                mxfp4 ? LaunchCfg<KernelMode::Dense, MmaKind::MXFP4,
                                  EpilogueOp::Sigmoid>(params, shape, stream)
                      : LaunchCfg<KernelMode::Dense, MmaKind::NVFP4,
                                  EpilogueOp::Sigmoid>(params, shape, stream);
                break;
            default:
                TORCH_CHECK(false, "Unsupported epilogue ",
                            static_cast<int>(epilogue), " for ", mode);
        }
    }
}

}  // namespace halops
