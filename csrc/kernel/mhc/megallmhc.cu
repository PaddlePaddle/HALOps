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

#include <torch/extension.h>

#include <algorithm>
#include <optional>
#include <vector>

#include "common/cuda_utils.h"
#include "common/pdl.cuh"
#include "kernel/gemm/gemm_utils.h"
#include "kernel/mhc/megallmhc.cuh"
#include "kernel/moe/moe_tma.cuh"

namespace halops {

namespace mhc {

using Traits = MhcKernelTraits<4096>;

void CheckInput(const torch::Tensor& t, const char* name,
                torch::ScalarType dtype, int64_t numel,
                const torch::Device& device) {
    TORCH_CHECK(t.device() == device && t.scalar_type() == dtype &&
                    t.is_contiguous() && t.numel() == numel,
                name, " must be a contiguous ", dtype, " tensor of ", numel,
                " elements on ", device, ", got ", t.scalar_type(), " ",
                t.sizes(), " on ", t.device());
}

void LaunchMegaLlMhc(const Params& p, float* part, bool has_shared,
                     bool align_fleet, bool use_pdl, int64_t token_tiles,
                     cudaStream_t stream) {
    const uint64_t t = p.num_tokens, h = Traits::kHidden, hc = Traits::kHc;
    const uint64_t row = h * sizeof(__nv_bfloat16);
    const uint32_t cols = Traits::kCtaCols, rows = Traits::kMTile;
    TmaMaps maps;
    maps.x = moe::CreateTmaDesc<2>("x", p.x, {h, t}, {row}, {cols, rows});
    maps.shared = has_shared ? moe::CreateTmaDesc<2>("shared", p.shared, {h, t},
                                                     {row}, {cols, rows})
                             : maps.x;
    maps.res = moe::CreateTmaDesc<3>("residual", p.residual, {h, hc, t},
                                     {row, row * hc}, {cols, 1, rows});
    maps.out = moe::CreateTmaDesc<3>("residual_out", p.residual_out, {h, hc, t},
                                     {row, row * hc}, {cols, 1, rows});

    const int m_tiles = (p.num_tokens + Traits::kMTile - 1) / Traits::kMTile;
    const int tiles = static_cast<int>(std::min<int64_t>(token_tiles, m_tiles));
    const int per_tile = (m_tiles + tiles - 1) / tiles * Traits::kMTile;
    const auto partial = has_shared ? megallmhc_partial_kernel<Traits, true>
                                    : megallmhc_partial_kernel<Traits, false>;
    HALOPS_CUDA_CHECK(cudaFuncSetAttribute(
        partial, cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(Traits::kPartSmem)));
    HALOPS_CUDA_CHECK(LaunchKernelWithPdl(
        partial,
        dim3(Traits::kChunks, (p.num_tokens + per_tile - 1) / per_tile),
        Traits::kPartThreads, kNoCluster, Traits::kPartSmem, stream,
        /*use_pdl=*/false, p, part, per_tile, maps));
    const auto finish = align_fleet ? megallmhc_finish_kernel<Traits, true>
                                    : megallmhc_finish_kernel<Traits, false>;
    HALOPS_CUDA_CHECK(LaunchKernelWithPdl(
        finish, p.num_tokens, Traits::kFinThreads, kNoCluster, 0, stream,
        use_pdl, p, static_cast<const float*>(part)));
}

}  // namespace mhc

std::vector<torch::Tensor> MegaLlMhc(
    const torch::Tensor& x, const torch::Tensor& residual,
    const torch::Tensor& post, const torch::Tensor& comb,
    const torch::Tensor& fn, const torch::Tensor& hc_scale,
    const torch::Tensor& hc_base, const torch::Tensor& norm_weight,
    double rms_eps, double hc_pre_eps, double hc_sinkhorn_eps,
    double hc_post_mult_value, int64_t sinkhorn_repeat, double norm_eps,
    bool mhc_align_fleet, const std::optional<torch::Tensor>& shared,
    bool use_pdl, bool use_swizzle, int64_t token_tiles) {
    using mhc::CheckInput;
    using Traits = mhc::Traits;
    constexpr int64_t kHidden = Traits::kHidden, kHc = Traits::kHc;
    TORCH_CHECK(x.is_cuda() && x.dim() == 2,
                "x must be a [T, hidden] CUDA tensor, got ", x.sizes(), " on ",
                x.device());
    TORCH_CHECK(x.size(1) == kHidden, "megallmhc: unsupported hidden size ",
                x.size(1));
    TORCH_CHECK(token_tiles >= 1, "token_tiles must be >= 1, got ",
                token_tiles);
    const CudaDeviceGuard device_guard(x.device());
    const int64_t t = x.size(0);
    const auto dev = x.device();
    const auto kBf16 = torch::kBFloat16, kF32 = torch::kFloat32;
    CheckInput(x, "x", kBf16, t * kHidden, dev);
    if (shared) CheckInput(*shared, "shared", kBf16, t * kHidden, dev);
    CheckInput(residual, "residual", kBf16, t * kHc * kHidden, dev);
    CheckInput(post, "post", kF32, t * kHc, dev);
    CheckInput(comb, "comb", kF32, t * kHc * kHc, dev);
    CheckInput(fn, "fn", kF32, Traits::kMixDim * kHc * kHidden, dev);
    CheckInput(hc_scale, "hc_scale", kF32, Traits::kScaleGroups, dev);
    CheckInput(hc_base, "hc_base", kF32, Traits::kMixDim, dev);
    CheckInput(norm_weight, "norm_weight", kBf16, kHidden, dev);

    const auto opts = x.options();
    auto residual_out = torch::empty({t, kHc, kHidden}, opts);
    auto layer_out = torch::empty({t, kHidden}, opts);
    auto post_out = torch::empty({t, kHc}, opts.dtype(kF32));
    auto comb_out = torch::empty({t, kHc, kHc}, opts.dtype(kF32));
    auto q_out = torch::empty({t, kHidden}, opts.dtype(torch::kFloat8_e4m3fn));
    const int64_t sf_rows = use_swizzle ? AlignSfM(static_cast<int>(t)) : t;
    auto sf_out =
        torch::empty({sf_rows, Traits::kSfCols}, opts.dtype(torch::kByte));
    std::vector<torch::Tensor> outs = {residual_out, layer_out, post_out,
                                       comb_out,     q_out,     sf_out};
    if (t == 0) return outs;

    const auto bf = [](const torch::Tensor& v) {
        return reinterpret_cast<__nv_bfloat16*>(v.data_ptr());
    };
    mhc::Params p{};
    p.x = bf(x);
    p.shared = shared ? bf(*shared) : nullptr;
    p.residual = bf(residual);
    p.post = post.data_ptr<float>();
    p.comb = comb.data_ptr<float>();
    p.fn = fn.data_ptr<float>();
    p.hc_scale = hc_scale.data_ptr<float>();
    p.hc_base = hc_base.data_ptr<float>();
    p.norm_weight = bf(norm_weight);
    p.residual_out = bf(residual_out);
    p.layer_out = bf(layer_out);
    p.post_out = post_out.data_ptr<float>();
    p.comb_out = comb_out.data_ptr<float>();
    p.q_out = reinterpret_cast<uint8_t*>(q_out.data_ptr());
    p.sf_out = sf_out.data_ptr<uint8_t>();
    p.num_tokens = static_cast<int>(t);
    p.sf_swizzle = use_swizzle;
    p.sinkhorn_repeat = static_cast<int>(sinkhorn_repeat);
    p.rms_eps = static_cast<float>(rms_eps);
    p.hc_pre_eps = static_cast<float>(hc_pre_eps);
    p.hc_sinkhorn_eps = static_cast<float>(hc_sinkhorn_eps);
    p.hc_post_mult = static_cast<float>(hc_post_mult_value);
    p.norm_eps = static_cast<float>(norm_eps);

    auto part =
        torch::empty({t, Traits::kChunks, Traits::kPartW}, opts.dtype(kF32));
    mhc::LaunchMegaLlMhc(p, static_cast<float*>(part.data_ptr()),
                         shared.has_value(), mhc_align_fleet, use_pdl,
                         token_tiles, GetCurrentCUDAStream(x));
    return outs;
}

}  // namespace halops
