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

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <torch/extension.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <type_traits>

#include "common/cuda_utils.h"
#include "kernel/moe/moe_kernel.cuh"

namespace halops {
namespace moe {

template <typename MoEDims>
void CheckRouterWeightShape(const torch::Tensor& router_weight) {
    TORCH_CHECK(router_weight.scalar_type() == at::kBFloat16,
                "router_weight must be bfloat16.");
    TORCH_CHECK(router_weight.dim() == 2 &&
                    router_weight.size(0) ==
                        MoEDims::kNumRouterGates * MoEDims::kNumExperts &&
                    router_weight.size(1) == MoEDims::kHiddenSize,
                "router_weight must be [G * E, H] = [",
                MoEDims::kNumRouterGates * MoEDims::kNumExperts, ", ",
                MoEDims::kHiddenSize, "].");
}

template <typename MoEDims>
void CheckLatentDownShape(const torch::Tensor& latent_down_weight) {
    TORCH_CHECK(latent_down_weight.dim() == 2 &&
                    latent_down_weight.size(0) == MoEDims::kLatentSize &&
                    latent_down_weight.size(1) == MoEDims::kHiddenSize,
                "latent_down_weight must be [kLatentSize, kHiddenSize].");
}

template <typename MoEDims>
void CheckRoutedExpertShape(const torch::Tensor& expert_up_weights,
                            const torch::Tensor& expert_down_weights) {
    constexpr std::int64_t kExperts = MoEDims::kNumExperts;
    constexpr std::int64_t kIntermediate = MoEDims::kIntermediateSize;
    constexpr std::int64_t kExpertK = moe::MoEInputSize<MoEDims>::value;
    constexpr std::int64_t kUpRows = 2 * kIntermediate;
    // A 4-bit weight packs two values per byte along K.
    constexpr std::int64_t kWeightBytes =
        MoEDims::kQuantConfig.kExpertsQuant == moe::QuantMethod::kMXFP4 ? 2 : 1;
    TORCH_CHECK(expert_up_weights.dim() == 3 &&
                    expert_up_weights.size(0) == kExperts &&
                    expert_up_weights.size(1) == kUpRows &&
                    expert_up_weights.size(2) == kExpertK / kWeightBytes,
                "expert_up_weights must be [E, 2N, K / ", kWeightBytes, "].");
    TORCH_CHECK(expert_down_weights.dim() == 3 &&
                    expert_down_weights.size(0) == kExperts &&
                    expert_down_weights.size(1) == kExpertK &&
                    expert_down_weights.size(2) == kIntermediate / kWeightBytes,
                "expert_down_weights must be [E, K, N / ", kWeightBytes, "].");
}

template <typename MoEDims>
void CheckSharedExpertShape(const torch::Tensor& shared_up_weight,
                            const torch::Tensor& shared_down_weight) {
    constexpr std::int64_t kUpRows = 2 * MoEDims::kIntermediateSize;
    TORCH_CHECK(shared_up_weight.dim() == 2 &&
                    shared_up_weight.size(0) == kUpRows &&
                    shared_up_weight.size(1) == MoEDims::kHiddenSize,
                "shared_up_weight must be [2N, H].");
    TORCH_CHECK(shared_down_weight.dim() == 2 &&
                    shared_down_weight.size(0) == MoEDims::kHiddenSize &&
                    shared_down_weight.size(1) == MoEDims::kIntermediateSize,
                "shared_down_weight must be [H, N].");
}

template <typename MoEDims>
void CheckRoutingVectors(const std::optional<torch::Tensor>& correction_bias,
                         const std::optional<torch::Tensor>& per_expert_scale) {
    for (const auto* v : {&correction_bias, &per_expert_scale}) {
        TORCH_CHECK(!v->has_value() || (*v)->numel() == MoEDims::kNumExperts,
                    "correction_bias and per_expert_scale must have "
                    "num_experts elements, or be None.");
    }
}

template <typename MoEDims>
void CheckTailFc2Shape(const torch::Tensor& tail_fc2_weight,
                       const torch::Tensor& tail_fc2_scale) {
    constexpr std::int64_t kBlock =
        moe::TailGeometryT<MoEDims>::WeightQuant::kBlockSize;
    constexpr std::int64_t kScaleRows = moe::TailGeometryT<MoEDims>::kMxfp8
                                            ? MoEDims::kHiddenSize
                                            : MoEDims::kHiddenSize / kBlock;
    constexpr std::int64_t kScaleCols = MoEDims::kLatentSize / kBlock;
    TORCH_CHECK(
        tail_fc2_weight.sizes() ==
                at::IntArrayRef({MoEDims::kHiddenSize, MoEDims::kLatentSize}) &&
            tail_fc2_scale.sizes() == at::IntArrayRef({kScaleRows, kScaleCols}),
        "fc2_weight must be [H, L] and fc2_scale must be [", kScaleRows, ", ",
        kScaleCols, "].");
}

// All CTAs must remain resident for the kernel lifetime.
template <typename MoEDims>
uint32_t CalculateGridSize(int device, cudaStream_t stream,
                           std::uint32_t grid_cap = 0u) {
    using Traits = moe::MoEKernelTraits<MoEDims>;
    using Mainloop = moe::CollectiveMainloopMoE<Traits>;

    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);

    uint32_t grid_size =
        std::min(static_cast<std::uint32_t>(sm_count), Traits::kMaxGrid);
    if (grid_cap != 0u) grid_size = std::min(grid_size, grid_cap);
    constexpr std::uint32_t kClusterSize = Traits::kClusterSize;
    if constexpr (kClusterSize > 1u) {
        // Limit the grid to clusters that can be resident together.
        grid_size -= grid_size % kClusterSize;
        cudaLaunchAttribute cluster_attr{};
        cluster_attr.id = cudaLaunchAttributeClusterDimension;
        cluster_attr.val.clusterDim.x = kClusterSize;
        cluster_attr.val.clusterDim.y = 1u;
        cluster_attr.val.clusterDim.z = 1u;
        constexpr size_t smem_size = moe::get_moe_smem_size<MoEDims>();
        auto* kernel_ptr = &moe::fused_moe_kernel<Traits, Mainloop>;
        // Enable the shared-memory carveout before querying occupancy.
        if (smem_size >= 48 * 1024) {
            HALOPS_CUDA_CHECK(cudaFuncSetAttribute(
                kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(smem_size)));
        }
        cudaLaunchConfig_t query{};
        query.gridDim = dim3(grid_size, 1, 1);
        query.blockDim = dim3(Traits::kThreads, 1, 1);
        query.dynamicSmemBytes = smem_size;
        query.stream = stream;
        query.attrs = &cluster_attr;
        query.numAttrs = 1;

        int max_clusters = 0;
        const cudaError_t occupancy =
            cudaOccupancyMaxActiveClusters(&max_clusters, kernel_ptr, &query);
        if (occupancy == cudaSuccess) {
            grid_size =
                std::min(grid_size, static_cast<std::uint32_t>(max_clusters) *
                                        kClusterSize);
        } else {
            cudaGetLastError();
        }
    }
    TORCH_CHECK(grid_size >= Traits::kMinGrid, "The kernel needs at least ",
                Traits::kMinGrid,
                " co-resident blocks (router, latent and shared expert), but "
                "the launch grid has only ",
                grid_size,
                grid_cap > 0u ? " (grid_cap is set)" : " (the device limit).");
    return grid_size;
}

template <typename MoEDims>
void MoEImpl(const torch::Tensor& input, const torch::Tensor& router_weight,
             const torch::Tensor& expert_up_weights,
             const torch::Tensor& expert_up_weights_scale,
             const torch::Tensor& expert_down_weights,
             const torch::Tensor& expert_down_weights_scale,
             const torch::Tensor& shared_up_weight,
             const torch::Tensor& shared_up_weight_scale,
             const torch::Tensor& shared_down_weight,
             const torch::Tensor& shared_down_weight_scale,
             const torch::Tensor& latent_down_weight,
             const torch::Tensor& latent_down_weight_scale,
             const torch::Tensor& tail_fc2_weight,
             const torch::Tensor& tail_fc2_scale,
             const torch::Tensor& tail_norm_weight,
             const std::optional<torch::Tensor>& correction_bias,
             const std::optional<torch::Tensor>& per_expert_scale,
             const torch::Tensor& tail_epochs,
             const torch::Tensor& tail_uc_bases, const torch::Tensor& workspace,
             const torch::Tensor& tail_out, int64_t top_k, int64_t scoring_func,
             int64_t activation, int64_t tp_rank, int64_t tail_mc_base,
             int64_t tail_grid_cap, bool renormalize, double situ_beta,
             double situ_linear_beta, double tail_eps, bool use_pdl,
             bool routed_use_interleaved, bool shared_use_interleaved,
             bool routed_use_swizzle, bool shared_use_swizzle,
             bool latent_use_swizzle) {
    const uint32_t num_tokens = static_cast<uint32_t>(input.size(0));
    TORCH_CHECK(num_tokens >= 1 && num_tokens <= MoEDims::kNumTokens,
                "token count M=", num_tokens,
                " out of range (this schedule serves 1 <= M <= ",
                MoEDims::kNumTokens, ").");

    constexpr std::int64_t kMaxTopk =
        static_cast<std::int64_t>(MoEDims::kMaxTopk);
    TORCH_CHECK(top_k >= 1 && top_k <= kMaxTopk, "top_k must be between 1 and ",
                kMaxTopk, ".");

    CheckRouterWeightShape<MoEDims>(router_weight);
    CheckRoutedExpertShape<MoEDims>(expert_up_weights, expert_down_weights);
    if constexpr (MoEDims::kUseLatent) {
        CheckLatentDownShape<MoEDims>(latent_down_weight);
    }
    CheckSharedExpertShape<MoEDims>(shared_up_weight, shared_down_weight);
    CheckTailFc2Shape<MoEDims>(tail_fc2_weight, tail_fc2_scale);
    CheckRoutingVectors<MoEDims>(correction_bias, per_expert_scale);

    using Traits = moe::MoEKernelTraits<MoEDims>;
    using Mainloop = moe::CollectiveMainloopMoE<Traits>;
    using TailGeometry = moe::TailGeometryT<MoEDims>;

    typename Mainloop::Arguments args{};
    args.num_tokens = num_tokens;
    args.hidden_size = static_cast<uint32_t>(input.size(1));
    args.latent_size = MoEDims::kUseLatent
                           ? static_cast<uint32_t>(latent_down_weight.size(0))
                           : 0u;
    args.intermediate_size = static_cast<uint32_t>(shared_down_weight.size(1));
    args.layouts = moe::make_moe_layouts(
        routed_use_interleaved, routed_use_swizzle, shared_use_interleaved,
        shared_use_swizzle, latent_use_swizzle);
    args.top_k = static_cast<uint32_t>(top_k);
    args.scoring_func = static_cast<ScoringFunc>(scoring_func);
    args.renormalize = renormalize;
    args.correction_bias =
        correction_bias ? correction_bias->const_data_ptr<float>() : nullptr;
    args.per_expert_scale =
        per_expert_scale ? per_expert_scale->const_data_ptr<float>() : nullptr;
    args.activation = static_cast<Activation>(activation);
    args.situ_beta = static_cast<float>(situ_beta);
    args.situ_linear_beta = static_cast<float>(situ_linear_beta);

    args.input = static_cast<const DataType::ElementA*>(input.const_data_ptr());
    args.router_weight =
        static_cast<const DataType::ElementA*>(router_weight.const_data_ptr());

    args.expert_up_weights =
        static_cast<const moe::RoutedUpGemmTraits<MoEDims>::MmaElement*>(
            expert_up_weights.const_data_ptr());
    args.expert_up_weights_scale =
        static_cast<const moe::RoutedUpGemmTraits<MoEDims>::ElementScale*>(
            expert_up_weights_scale.const_data_ptr());
    args.expert_down_weights =
        static_cast<const moe::RoutedDownGemmTraits<MoEDims>::MmaElement*>(
            expert_down_weights.const_data_ptr());
    args.expert_down_weights_scale =
        static_cast<const moe::RoutedDownGemmTraits<MoEDims>::ElementScale*>(
            expert_down_weights_scale.const_data_ptr());

    args.shared_up_weight =
        static_cast<const moe::SharedUpGemmTraits<MoEDims>::MmaElement*>(
            shared_up_weight.const_data_ptr());
    args.shared_up_weight_scale =
        static_cast<const moe::SharedUpGemmTraits<MoEDims>::ElementScale*>(
            shared_up_weight_scale.const_data_ptr());
    args.shared_down_weight =
        static_cast<const moe::SharedDownGemmTraits<MoEDims>::MmaElement*>(
            shared_down_weight.const_data_ptr());
    args.shared_down_weight_scale =
        static_cast<const moe::SharedDownGemmTraits<MoEDims>::ElementScale*>(
            shared_down_weight_scale.const_data_ptr());

    args.latent_down_weight =
        MoEDims::kUseLatent
            ? static_cast<
                  const moe::LatentDownGemmTraits<MoEDims>::MmaElement*>(
                  latent_down_weight.const_data_ptr())
            : nullptr;
    using LatentScale =
        typename moe::LatentDownGemmTraits<MoEDims>::ElementScale;
    args.latent_down_weight_scale =
        MoEDims::kUseLatent ? static_cast<const LatentScale*>(
                                  latent_down_weight_scale.const_data_ptr())
                            : nullptr;

    args.workspace = static_cast<char*>(workspace.data_ptr());

    const int device = input.get_device();
    int l2_cached_bytes = 0;
    cudaDeviceGetAttribute(&l2_cached_bytes, cudaDevAttrL2CacheSize, device);
    args.l2_cached_bytes =
        l2_cached_bytes > 0 ? static_cast<uint32_t>(l2_cached_bytes) : 0u;

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(device).stream();
    const std::uint32_t grid_cap = static_cast<std::uint32_t>(tail_grid_cap);
    const int64_t* uc_bases = tail_uc_bases.const_data_ptr<int64_t>();
    for (int r = 0; r < MoEDims::KernelConfig::kTailWorld; ++r) {
        args.tail_uc_base[r] = reinterpret_cast<std::uint8_t*>(uc_bases[r]);
    }
    args.tail_mc_base = reinterpret_cast<std::uint8_t*>(
        static_cast<std::uintptr_t>(tail_mc_base));
    args.tail_epochs = static_cast<std::uint32_t*>(tail_epochs.data_ptr());
    args.tail_fc2_w8 = static_cast<const typename TailGeometry::ElementWeight*>(
        tail_fc2_weight.const_data_ptr());
    args.tail_fc2_scale =
        static_cast<const typename TailGeometry::ElementScale*>(
            tail_fc2_scale.const_data_ptr());
    args.tail_norm_weight = static_cast<const DataType::ElementA*>(
        tail_norm_weight.const_data_ptr());
    args.tail_out = static_cast<DataType::ElementOut*>(tail_out.data_ptr());
    args.tail_eps = static_cast<float>(tail_eps);
    args.tail_rank = static_cast<std::uint32_t>(tp_rank);

    const uint32_t grid_size =
        CalculateGridSize<MoEDims>(device, stream, grid_cap);

    const cudaError_t launch_status =
        moe::LaunchMoEKernel<MoEDims>(args, grid_size, stream, use_pdl);
    TORCH_CHECK(launch_status == cudaSuccess,
                "kernel launch failed: ", cudaGetErrorString(launch_status));
}

}  // namespace moe

template <class Dims>
constexpr int64_t WorkspaceBytesFor() {
    return static_cast<int64_t>(sizeof(moe::MoEWorkspace<Dims>));
}

int64_t GetWorkspaceSize(const std::optional<torch::Tensor>& input) {
    constexpr auto kBWFP8 = moe::QuantMethod::kBlockwiseFP8;
    constexpr auto kMXFP8 = moe::QuantMethod::kMXFP8;
    constexpr auto kMXFP4 = moe::QuantMethod::kMXFP4;
    using QuantBWFP8 = moe::QuantConfig<kBWFP8, kBWFP8, kBWFP8>;
    using QuantMXFP8 = moe::QuantConfig<kMXFP8, kMXFP8, kMXFP8>;
    using QuantMXFP4 = moe::QuantConfig<kMXFP8, kMXFP8, kMXFP4>;

    constexpr int64_t kBs8Bytes = std::max(
        {WorkspaceBytesFor<moe::Dims_BS8_E512_N512_K2048_SM100<QuantBWFP8{}>>(),
         WorkspaceBytesFor<moe::Dims_BS8_E512_N512_K2048_SM100<QuantMXFP8{}>>(),
         WorkspaceBytesFor<
             moe::Dims_BS8_E512_N512_K2048_SM100<QuantMXFP4{}>>()});
    constexpr int64_t kBs16Bytes =
        std::max({WorkspaceBytesFor<
                      moe::Dims_BS16_E512_N512_K2048_SM100<QuantBWFP8{}>>(),
                  WorkspaceBytesFor<
                      moe::Dims_BS16_E512_N512_K2048_SM100<QuantMXFP8{}>>(),
                  WorkspaceBytesFor<
                      moe::Dims_BS16_E512_N512_K2048_SM100<QuantMXFP4{}>>()});
    if (!input.has_value()) {
        return kBs8Bytes > kBs16Bytes ? kBs8Bytes : kBs16Bytes;
    }
    return input->size(0) <= 8 ? kBs8Bytes : kBs16Bytes;
}

void MegaLLMoE(const torch::Tensor& input, const torch::Tensor& router_weight,
               const torch::Tensor& expert_up_weights,
               const torch::Tensor& expert_up_weights_scale,
               const torch::Tensor& expert_down_weights,
               const torch::Tensor& expert_down_weights_scale,
               const torch::Tensor& shared_up_weight,
               const torch::Tensor& shared_up_weight_scale,
               const torch::Tensor& shared_down_weight,
               const torch::Tensor& shared_down_weight_scale,
               const torch::Tensor& latent_down_weight,
               const torch::Tensor& latent_down_weight_scale,
               const torch::Tensor& tail_fc2_weight,
               const torch::Tensor& tail_fc2_scale,
               const torch::Tensor& tail_norm_weight,
               const std::optional<torch::Tensor>& correction_bias,
               const std::optional<torch::Tensor>& per_expert_scale,
               const torch::Tensor& tail_epochs,
               const torch::Tensor& tail_uc_bases,
               const torch::Tensor& workspace, const torch::Tensor& tail_out,
               int64_t top_k, int64_t num_router_gates, int64_t tp_rank,
               int64_t tail_mc_base, int64_t tail_grid_cap,
               const std::string& scoring_func, const std::string& activation,
               bool renormalize, double situ_beta, double situ_linear_beta,
               double tail_eps, const std::string& quant_mode, int64_t tp_size,
               bool use_pdl, bool routed_use_interleaved,
               bool shared_use_interleaved, bool routed_use_swizzle,
               bool shared_use_swizzle, bool latent_use_swizzle) {
    TORCH_CHECK(
        quant_mode == "w8a8-mxfp8" || quant_mode == "w4a8-mxfp4" ||
            quant_mode == "w8a8-blockwise",
        "quant_mode must be w8a8-blockwise, w8a8-mxfp8 or w4a8-mxfp4, got ",
        quant_mode, ".");
    TORCH_CHECK(num_router_gates == 2, "Only support 2 router gates, got ",
                num_router_gates, ".");
    TORCH_CHECK(input.dim() == 2, "The input' shape must be [tokens, hidden].");

    const int64_t num_tokens = input.size(0);
    TORCH_CHECK(num_tokens >= 1 && num_tokens <= 16,
                "num_tokens must be in [1, 16].");

    TORCH_CHECK(scoring_func == "sigmoid" || scoring_func == "softmax",
                "scoring_func must be sigmoid or softmax.");
    TORCH_CHECK(activation == "silu" || activation == "situ",
                "activation must be silu or situ.");
    if (activation == "situ") {
        // Positive caps must be finite; nonpositive beta disables the cap.
        TORCH_CHECK(std::isfinite(situ_beta) && std::isfinite(situ_linear_beta),
                    "situ_beta / situ_linear_beta must be finite (<= 0 "
                    "means no cap).");
    }
    TORCH_CHECK(tp_size == 1 || tp_size == 4, "tp_size must be 1 or 4, got ",
                tp_size, ".");

    const int64_t scoring_type = scoring_func == "sigmoid" ? 0 : 1;
    const int64_t activation_type = activation == "silu" ? 0 : 1;

    const auto launch_func = [&](auto dims_tag) {
        using MoEDims = decltype(dims_tag);
        moe::MoEImpl<MoEDims>(
            input, router_weight, expert_up_weights, expert_up_weights_scale,
            expert_down_weights, expert_down_weights_scale, shared_up_weight,
            shared_up_weight_scale, shared_down_weight,
            shared_down_weight_scale, latent_down_weight,
            latent_down_weight_scale, tail_fc2_weight, tail_fc2_scale,
            tail_norm_weight, correction_bias, per_expert_scale, tail_epochs,
            tail_uc_bases, workspace, tail_out, top_k, scoring_type,
            activation_type, tp_rank, tail_mc_base, tail_grid_cap, renormalize,
            situ_beta, situ_linear_beta, tail_eps, use_pdl,
            routed_use_interleaved, shared_use_interleaved, routed_use_swizzle,
            shared_use_swizzle, latent_use_swizzle);
    };

    constexpr auto kBWFP8 = moe::QuantMethod::kBlockwiseFP8;
    constexpr auto kMXFP8 = moe::QuantMethod::kMXFP8;
    constexpr auto kMXFP4 = moe::QuantMethod::kMXFP4;
    using QuantBWFP8 = moe::QuantConfig<kBWFP8, kBWFP8, kBWFP8>;
    using QuantMXFP8 = moe::QuantConfig<kMXFP8, kMXFP8, kMXFP8>;
    using QuantMXFP4 = moe::QuantConfig<kMXFP8, kMXFP8, kMXFP4>;

    const auto launch_for_tp_size = [&](auto quant_tag) {
        if (tp_size == 1) {
            if (num_tokens <= 8) {
                launch_func(
                    moe::Dims_BS8_E512_N512_K2048_SM100<decltype(quant_tag){},
                                                        true, 1>{});
            } else {
                launch_func(
                    moe::Dims_BS16_E512_N512_K2048_SM100<decltype(quant_tag){},
                                                         true, 1>{});
            }
        } else {
            if (num_tokens <= 8) {
                launch_func(
                    moe::Dims_BS8_E512_N512_K2048_SM100<decltype(quant_tag){},
                                                        true, 4>{});
            } else {
                launch_func(
                    moe::Dims_BS16_E512_N512_K2048_SM100<decltype(quant_tag){},
                                                         true, 4>{});
            }
        }
    };

    if (quant_mode == "w4a8-mxfp4") {
        launch_for_tp_size(QuantMXFP4{});
    } else if (quant_mode == "w8a8-mxfp8") {
        launch_for_tp_size(QuantMXFP8{});
    } else {
        launch_for_tp_size(QuantBWFP8{});
    }
}

int64_t MoETailRegionBytes(bool wide, int64_t tp_size) {
    using QuantBWFP8 = moe::QuantConfig<>;
    TORCH_CHECK(tp_size == 1 || tp_size == 4, "tp_size must be 1 or 4.");
    if (tp_size == 1) {
        return wide
                   ? moe::tail_region_bytes<
                         moe::Dims_BS16_E512_N512_K2048_SM100<QuantBWFP8{},
                                                              true, 1>>()
                   : moe::tail_region_bytes<moe::Dims_BS8_E512_N512_K2048_SM100<
                         QuantBWFP8{}, true, 1>>();
    }
    return wide ? moe::tail_region_bytes<moe::Dims_BS16_E512_N512_K2048_SM100<
                      QuantBWFP8{}, true, 4>>()
                : moe::tail_region_bytes<moe::Dims_BS8_E512_N512_K2048_SM100<
                      QuantBWFP8{}, true, 4>>();
}

}  // namespace halops
