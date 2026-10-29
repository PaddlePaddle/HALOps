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

#include "kernel/gemm/bf16_cfgs.h"
#include "kernel/gemm/gemm_dispatch.cuh"
#include "kernel/gemm/gemm_utils.h"

namespace halops {

void DenseGemm(const torch::Tensor& input, const torch::Tensor& weight,
               const torch::Tensor& sfa, const torch::Tensor& sfb,
               torch::Tensor& output, const std::optional<torch::Tensor>& bias,
               const std::optional<torch::Tensor>& alpha,
               const int sm_count_arg, const std::string& mode,
               const bool use_pdl) {
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
    TORCH_CHECK(output.size(0) == m && output.size(1) == n,
                "output shape must be [M, N], got [", output.size(0), ", ",
                output.size(1), "], expected [", m, ", ", n, "]");
    const int sm = sm_count_arg > 0 ? sm_count_arg : GetSmCount();
    const cudaStream_t stream = GetCurrentCUDAStream(input);

    const int m_sf = AlignSfM(m);
    // still need to pass a rounded-up M to the kernel when using the bias path.
    const bool bias_path = bias.has_value();
    const int m_shape = (m < kSfAtomM || bias_path) ? m : m_sf;

    GemmParams params;
    params.ptr_d = output.data_ptr();
    params.m = m;
    params.n = n;
    params.num_experts = 1;
    params.max_m_per_expert = m;
    params.use_pdl = use_pdl;

    if (mode == "bf16") {
        // Native BF16: sfa/sfb/bias/alpha are ignored.
        using TypeA = cutlass::bfloat16_t;
        using TypeB = cutlass::bfloat16_t;
        using TypeC = cutlass::bfloat16_t;
        using TypeD = cutlass::bfloat16_t;
        // Placeholder, never dereferenced.
        using TypeSF = cutlass::float_ue8m0_t;
        using TypeAccumulate = float;

        TORCH_CHECK(input.dtype() == torch::kBFloat16,
                    "bf16 dense GEMM: input must be bfloat16");
        TORCH_CHECK(weight.dtype() == torch::kBFloat16,
                    "bf16 dense GEMM: weight must be bfloat16");
        TORCH_CHECK(output.dtype() == torch::kBFloat16,
                    "bf16 dense GEMM: output must be bfloat16");

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

        // Debug only: HALOPS_FORCE_CFG=256x128x64_2CTA (any name in kKnown,
        // i.e. GemmKernelConfigBF16NaiveNoBias<name>) skips the dispatch below
        // and runs that config on every shape it supports; other shapes are
        // dispatched normally.
        static const std::string forced_cfg = [] {
            const char* e = std::getenv("HALOPS_FORCE_CFG");
            std::string name(e == nullptr ? "" : e);
            static const char* kKnown[] = {
                "",
                "256x256x64_2CTA",
                "256x256x64_2CTA_P7",
                "256x128x64_2CTA",
                "256x128x64_2CTA_Ntail",
                "256x128x64_2CTA_P9",
                "256x128x64_2CTA_P9_Ntail",
                "256x192x64_2CTA",
                "256x64x64_2CTA",
                "128x256x64_1CTA",
                "128x128x64_1CTA",
                "128x64x64_1CTA",
                "128x32x64_1CTA",
                "128x64x64_2CTA_P18",
                "64x224x64_1CTA_P6",
                "64x128x64_1CTA",
                "64x96x64_1CTA_P11",
                "64x64x64_1CTA",
                "64x64x64_1WG_1CTA",
                "64x64x64_P12_1CTA",
                "64x64x64_P14_1CTA",
                "64x32x64_1WG_P14_1CTA",
                "64x32x64_1WG_P18_1CTA",
                "64x64x64_1WG_P14_1CTA",
            };
            bool known = false;
            for (const char* k : kKnown) {
                known = known || name == k;
            }
            TORCH_CHECK(known, "HALOPS_FORCE_CFG=", name,
                        " is not a config name this hatch knows; see the "
                        "kKnown list in dense_gemm.cu.");
            return name;
        }();
        if (!forced_cfg.empty()) {
            // Required n alignment: 32 for non-power-of-2 KBlockN, 64 for
            // _Ntail configs, KBlockN otherwise.
#define HALOPS_FORCE_TRY(NAME, CFG, EXTRA)                                    \
    if (forced_cfg == NAME) {                                                 \
        using KernelConfig = CFG;                                             \
        constexpr int kNMod =                                                 \
            KernelConfig::KForceEpiNPredicate                                 \
                ? 64                                                          \
                : ((KernelConfig::KBlockN & (KernelConfig::KBlockN - 1)) != 0 \
                       ? 32                                                   \
                       : KernelConfig::KBlockN);                              \
        constexpr int kNModCluster = kNMod * KernelConfig::KClusterN;         \
        if (m % KernelConfig::KBlockM == 0 && n % kNModCluster == 0 &&        \
            k % KernelConfig::KBlockK == 0 && (EXTRA)) {                      \
            gemm_impl<TypeA, TypeB, TypeC, TypeD, TypeSF, TypeAccumulate,     \
                      KernelConfig>(params, sm, stream);                      \
            return;                                                           \
        }                                                                     \
    }
            HALOPS_FORCE_TRY("256x256x64_2CTA",
                             GemmKernelConfigBF16NaiveNoBias256x256x64_2CTA,
                             true)
            HALOPS_FORCE_TRY("256x256x64_2CTA_P7",
                             GemmKernelConfigBF16NaiveNoBias256x256x64_2CTA_P7,
                             true)
            HALOPS_FORCE_TRY("256x128x64_2CTA",
                             GemmKernelConfigBF16NaiveNoBias256x128x64_2CTA,
                             true)
            HALOPS_FORCE_TRY(
                "256x128x64_2CTA_Ntail",
                GemmKernelConfigBF16NaiveNoBias256x128x64_2CTA_Ntail, true)
            HALOPS_FORCE_TRY("256x128x64_2CTA_P9",
                             GemmKernelConfigBF16NaiveNoBias256x128x64_2CTA_P9,
                             true)
            HALOPS_FORCE_TRY(
                "256x128x64_2CTA_P9_Ntail",
                GemmKernelConfigBF16NaiveNoBias256x128x64_2CTA_P9_Ntail, true)
            HALOPS_FORCE_TRY("256x192x64_2CTA",
                             GemmKernelConfigBF16NaiveNoBias256x192x64_2CTA,
                             true)
            HALOPS_FORCE_TRY("256x64x64_2CTA",
                             GemmKernelConfigBF16NaiveNoBias256x64x64_2CTA,
                             true)
            HALOPS_FORCE_TRY("128x256x64_1CTA",
                             GemmKernelConfigBF16NaiveNoBias128x256x64_1CTA,
                             true)
            HALOPS_FORCE_TRY("128x128x64_1CTA",
                             GemmKernelConfigBF16NaiveNoBias128x128x64_1CTA,
                             true)
            HALOPS_FORCE_TRY("128x64x64_1CTA",
                             GemmKernelConfigBF16NaiveNoBias128x64x64_1CTA,
                             true)
            HALOPS_FORCE_TRY("128x32x64_1CTA",
                             GemmKernelConfigBF16NaiveNoBias128x32x64_1CTA,
                             true)
            // Only within the correct CTA band.
            HALOPS_FORCE_TRY("128x64x64_2CTA_P18",
                             GemmKernelConfigBF16NaiveNoBias128x64x64_2CTA_P18,
                             bf16_dispatch::PairDeepCtaCount(m, n) >=
                                     bf16_dispatch::kPairDeepMinCtas &&
                                 bf16_dispatch::PairDeepCtaCount(m, n) <=
                                     bf16_dispatch::kPairDeepMaxCtas)
            HALOPS_FORCE_TRY("64x224x64_1CTA_P6",
                             GemmKernelConfigBF16NaiveNoBias64x224x64_1CTA_P6,
                             true)
            HALOPS_FORCE_TRY("64x128x64_1CTA",
                             GemmKernelConfigBF16NaiveNoBias64x128x64_1CTA,
                             true)
            HALOPS_FORCE_TRY("64x96x64_1CTA_P11",
                             GemmKernelConfigBF16NaiveNoBias64x96x64_1CTA_P11,
                             true)
            HALOPS_FORCE_TRY("64x64x64_1CTA",
                             GemmKernelConfigBF16NaiveNoBias64x64x64_1CTA, true)
            HALOPS_FORCE_TRY("64x64x64_1WG_1CTA",
                             GemmKernelConfigBF16NaiveNoBias64x64x64_1WG_1CTA,
                             true)
            HALOPS_FORCE_TRY("64x64x64_P12_1CTA",
                             GemmKernelConfigBF16NaiveNoBias64x64x64_P12_1CTA,
                             true)
            HALOPS_FORCE_TRY("64x64x64_P14_1CTA",
                             GemmKernelConfigBF16NaiveNoBias64x64x64_P14_1CTA,
                             true)
            HALOPS_FORCE_TRY(
                "64x32x64_1WG_P14_1CTA",
                GemmKernelConfigBF16NaiveNoBias64x32x64_1WG_P14_1CTA, true)
            HALOPS_FORCE_TRY(
                "64x32x64_1WG_P18_1CTA",
                GemmKernelConfigBF16NaiveNoBias64x32x64_1WG_P18_1CTA, true)
            HALOPS_FORCE_TRY(
                "64x64x64_1WG_P14_1CTA",
                GemmKernelConfigBF16NaiveNoBias64x64x64_1WG_P14_1CTA, true)
#undef HALOPS_FORCE_TRY
        }

        // Entry guards per tier (same messages the per-rung checks used).
        if (m >= 128 && m < 4096) {
            CheckDivisible(m, 128, "BF16 GEMM mid M M");
            CheckDivisible(n, 32, "BF16 GEMM mid M N");
            CheckDivisible(k, 64, "BF16 GEMM mid M K");
        } else if (m < 128 && !bf16_dispatch::ThinMNoSwapOk(m, n, k, sm)) {
            CheckDivisible(n, 128, "BF16 GEMM SwapAB N");
            CheckDivisible(m, 64, "BF16 GEMM SwapAB M");
            CheckDivisible(k, 64, "BF16 GEMM SwapAB K");
        }
        const int cfg = bf16_dispatch::dense::SelectBF16Cfg(m, n, k, sm);
        TORCH_CHECK(cfg >= 0, "No BF16 dense GEMM tile supports M=", m,
                    ", N=", n, ", K=", k);
        const auto& desc = bf16_dispatch::kBF16Cfgs[cfg];
        // Same divisibility contract the per-rung checks enforced: the
        // output M/N tile multiples (N may relax to %32 for non-power-of-2
        // widths or %64 for predicated-tail tiles), K a KBlockK multiple.
        // SwapAB swaps the roles: KBlockM tiles the weight (output N) and
        // KBlockN the activation (output M).
        const int n_mod =
            desc.ntail ? 64 : (((desc.bn & (desc.bn - 1)) != 0) ? 32 : desc.bn);
        const int m_div = desc.swap ? desc.bn : desc.bm;
        const int n_div = desc.swap ? desc.bm : n_mod;
        TORCH_CHECK(m % m_div == 0,
                    "BF16 dense GEMM requires M to be divisible by ", m_div,
                    ", got ", m);
        TORCH_CHECK(n % n_div == 0,
                    "BF16 dense GEMM requires N to be divisible by ", n_div,
                    ", got ", n);
        TORCH_CHECK(k % desc.bk == 0,
                    "BF16 dense GEMM requires K to be divisible by ", desc.bk,
                    ", got ", k);
        bf16_dispatch::DispatchBF16Cfg(cfg, [&]<class KernelConfig>() {
            gemm_impl<TypeA, TypeB, TypeC, TypeD, TypeSF, TypeAccumulate,
                      KernelConfig>(params, sm, stream);
        });
    } else if (mode == "w8a8-mxfp8") {
        const int k = weight.size(1);
        TORCH_CHECK(input.size(1) == k,
                    "input K must match weight K, got input K=", input.size(1),
                    ", weight K=", k);
        params.k = k;
        params.ptr_a = input.data_ptr();
        params.ptr_b = weight.data_ptr();
        params.ptr_sfa = sfa.data_ptr();
        // sfb is the quantizer's uint8 SfAtom weight scale, used as-is: a
        // 224-wide tile reads it as a run of atoms plus a 32-row rewrite.
        TORCH_CHECK(sfb.scalar_type() == torch::kUInt8,
                    "sfb must be the uint8 SfAtom weight scale that "
                    "group_wise_quant returns");
        params.ptr_sfb = sfb.data_ptr();
        if (bias.has_value()) {
            params.ptr_c = bias->data_ptr();
        }
        TORCH_CHECK(
            sfa.is_contiguous() && sfa.numel() >= int64_t(m_sf) * (k / 32),
            "sfa must hold whole 128-row SfAtoms: expected at least ",
            int64_t(m_sf) * (k / 32), " elements (", m_sf, " x ", k / 32,
            ") for M=", m, ", got ", sfa.numel());
        TORCH_CHECK(
            !bias_path || m % 256 == 0,
            "the bias epilogue has no per-row store predicate yet, so M "
            "must be a multiple of 256 when bias is given, got ",
            m);
        const GemmShape shape = {.m = m_shape,
                                 .n = n,
                                 .k = k,
                                 .m_grid = m_shape,
                                 .sm_count = sm,
                                 .has_bias = bias_path};
        LaunchCfg<KernelMode::Dense, MmaKind::MXFP8>(params, shape, stream);
    } else {
        const int k = weight.size(1) * 2;  // 2 FP4 per packed byte
        TORCH_CHECK(
            input.size(1) == weight.size(1),
            "packed input K/2 must match packed weight K/2, got input K/2=",
            input.size(1), ", weight K/2=", weight.size(1));
        params.k = k;
        params.ptr_a = input.data_ptr();
        params.ptr_b = weight.data_ptr();
        params.ptr_sfa = sfa.data_ptr();
        params.ptr_sfb = sfb.data_ptr();
        TORCH_CHECK(!bias.has_value(), "W4A4 mode does not support bias yet.");

        const bool mxfp4 = mode == "w4a4-mxfp4";
        TORCH_CHECK(!mxfp4 || !alpha.has_value(),
                    "'w4a4-mxfp4' does not take alpha; the global scaling "
                    "factor is only used by 'w4a4-nvfp4'.");
        params.ptr_alpha =
            alpha.has_value() ? alpha->data_ptr<float>() : nullptr;
        const int sf_k = k / (mxfp4 ? 32 : 16);
        TORCH_CHECK(sfa.size(-1) == sf_k && sfb.size(-1) == sf_k,
                    "W4A4 scale-factor shape does not match mode ",
                    mxfp4 ? "'w4a4-mxfp4' (UE8M0 per 32)"
                          : "'w4a4-nvfp4' (UE4M3 per 16)",
                    " requires SF K = ", sf_k, ", got sfa K=", sfa.size(-1),
                    " and sfb K=", sfb.size(-1), ".");
        TORCH_CHECK(sfa.is_contiguous() && sfa.numel() >= int64_t(m_sf) * sf_k,
                    "sfa must hold whole 128-row SfAtoms: expected at least ",
                    int64_t(m_sf) * sf_k, " elements (", m_sf, " x ", sf_k,
                    ") for M=", m, ", got ", sfa.numel());
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
        if (mxfp4) {
            LaunchCfg<KernelMode::Dense, MmaKind::MXFP4>(params, shape, stream);
        } else {
            LaunchCfg<KernelMode::Dense, MmaKind::NVFP4>(params, shape, stream);
        }
    }
}

}  // namespace halops
