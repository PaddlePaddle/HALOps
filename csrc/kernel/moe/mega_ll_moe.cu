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
#include <cstdio>
#include <optional>
#include <string>
#include <type_traits>

#include "common/cuda_utils.h"
#include "kernel/moe/moe_kernel.cuh"

namespace halops {
namespace moe {

moe::MoEProblemSize BuildProblem(const torch::Tensor& input,
                                 const torch::Tensor& router_weight,
                                 const torch::Tensor& shared_down_weight,
                                 const torch::Tensor& latent_down_weight,
                                 std::int64_t num_router_gates) {
    TORCH_CHECK(num_router_gates >= 1 &&
                    num_router_gates <= MoERuntime::kMaxRouterGates,
                "num_router_gates must be 1 or 2, got ", num_router_gates, ".");
    TORCH_CHECK(input.dim() == 2,
                "The input's shape must be [tokens, hidden].");

    const std::uint32_t num_tokens = static_cast<std::uint32_t>(input.size(0));
    TORCH_CHECK(num_tokens >= 1 && num_tokens <= 16,
                "num_tokens must be in [1, 16].");

    const std::uint32_t hidden_size = static_cast<std::uint32_t>(input.size(1));
    const std::uint32_t intermediate_size =
        static_cast<std::uint32_t>(shared_down_weight.size(1));
    const std::uint32_t latent_size =
        latent_down_weight.defined()
            ? static_cast<std::uint32_t>(latent_down_weight.size(0))
            : 0u;
    const std::uint32_t num_experts = static_cast<std::uint32_t>(
        router_weight.size(0) / static_cast<std::int64_t>(num_router_gates));
    return moe::MoEProblemSize(num_tokens, hidden_size, intermediate_size,
                               latent_size, num_experts, num_router_gates);
}

void CheckRouterWeightShape(const torch::Tensor& router_weight,
                            moe::MoEProblemSize const& problem_size) {
    TORCH_CHECK(router_weight.scalar_type() == at::kBFloat16,
                "router_weight must be bfloat16.");
    TORCH_CHECK(
        router_weight.dim() == 2 &&
            router_weight.size(0) ==
                static_cast<std::int64_t>(problem_size.num_router_gates) *
                    problem_size.num_experts &&
            router_weight.size(1) == problem_size.hidden_size,
        "router_weight must be [G * E, H] = [",
        problem_size.num_router_gates * problem_size.num_experts, ", ",
        problem_size.hidden_size, "].");
}

void CheckLatentDownShape(const torch::Tensor& latent_down_weight,
                          moe::MoEProblemSize const& problem_size) {
    TORCH_CHECK(latent_down_weight.dim() == 2 &&
                    latent_down_weight.size(0) == problem_size.latent_size &&
                    latent_down_weight.size(1) == problem_size.hidden_size,
                "latent_down_weight must be [latent_size, hidden_size] = [",
                problem_size.latent_size, ", ", problem_size.hidden_size, "].");
}

template <typename MoEDims>
void CheckRoutedExpertShape(const torch::Tensor& expert_up_weights,
                            const torch::Tensor& expert_down_weights,
                            moe::MoEProblemSize const& problem_size) {
    std::int64_t num_experts = problem_size.num_experts;
    std::int64_t input_size = problem_size.input_size();
    std::int64_t intermediate_size = problem_size.intermediate_size;
    // A 4-bit weight packs two values per byte along K.
    constexpr std::int64_t kWeightBytes =
        MoEDims::kQuantConfig.kExpertsQuant == moe::QuantMethod::kMXFP4 ? 2 : 1;
    TORCH_CHECK(expert_up_weights.dim() == 3 &&
                    expert_up_weights.size(0) == num_experts &&
                    expert_up_weights.size(1) == 2 * intermediate_size &&
                    expert_up_weights.size(2) == input_size / kWeightBytes,
                "expert_up_weights must be [E, 2N, K / ", kWeightBytes, "].");
    TORCH_CHECK(
        expert_down_weights.dim() == 3 &&
            expert_down_weights.size(0) == num_experts &&
            expert_down_weights.size(1) == input_size &&
            expert_down_weights.size(2) == intermediate_size / kWeightBytes,
        "expert_down_weights must be [E, K, N / ", kWeightBytes, "].");
}

void CheckSharedExpertShape(const torch::Tensor& shared_up_weight,
                            const torch::Tensor& shared_down_weight,
                            moe::MoEProblemSize const& problem_size) {
    std::int64_t hidden_size = problem_size.hidden_size;
    std::int64_t intermediate_size = problem_size.intermediate_size;
    TORCH_CHECK(shared_up_weight.dim() == 2 &&
                    shared_up_weight.size(0) == 2 * intermediate_size &&
                    shared_up_weight.size(1) == hidden_size,
                "shared_up_weight must be [2N, H] = [", 2 * intermediate_size,
                ", ", hidden_size, "].");
    TORCH_CHECK(shared_down_weight.dim() == 2 &&
                    shared_down_weight.size(0) == hidden_size &&
                    shared_down_weight.size(1) == intermediate_size,
                "shared_down_weight must be [H, N] = [", hidden_size, ", ",
                intermediate_size, "].");
}

void CheckRoutingVectors(const std::optional<torch::Tensor>& correction_bias,
                         const std::optional<torch::Tensor>& per_expert_scale,
                         moe::MoEProblemSize const& problem_size) {
    for (const auto* v : {&correction_bias, &per_expert_scale}) {
        TORCH_CHECK(
            !v->has_value() || (*v)->numel() == problem_size.num_experts,
            "correction_bias and per_expert_scale must have "
            "num_experts elements, or be None.");
    }
}

template <typename MoEDims>
void CheckTailFc2Shape(const torch::Tensor& tail_fc2_weight,
                       const torch::Tensor& tail_fc2_scale,
                       moe::MoEProblemSize const& problem_size) {
    constexpr std::int64_t kBlock =
        moe::TailGeometryT<MoEDims>::WeightQuant::kBlockSize;
    const std::int64_t scale_rows = moe::TailGeometryT<MoEDims>::kMxfp8
                                        ? problem_size.hidden_size
                                        : problem_size.hidden_size / kBlock;
    const std::int64_t scale_cols = problem_size.input_size() / kBlock;
    TORCH_CHECK(
        tail_fc2_weight.sizes() ==
                at::IntArrayRef(
                    {problem_size.hidden_size, problem_size.input_size()}) &&
            tail_fc2_scale.sizes() == at::IntArrayRef({scale_rows, scale_cols}),
        "fc2_weight must be [H, input_size] and fc2_scale must be [",
        scale_rows, ", ", scale_cols, "].");
}

template <typename MoEDims>
moe::MoERuntime BuildRuntimeConfig(const MoEProblemSize& problem) {
    using Traits = moe::MoEKernelTraits<MoEDims>;
    using UpGemm = typename Traits::RoutedUpGemm;
    using DownGemm = typename Traits::RoutedDownGemm;
    using LatentDownGemm = typename Traits::LatentDownGemm;
    using SharedUpGemm = typename Traits::SharedUpGemm;
    using SharedDownGemm = typename Traits::SharedDownGemm;
    using TailGeometry = moe::TailGeometryT<MoEDims>;

    // The schedule's CTA budget per phase; the split counts follow from it.
    const std::uint32_t num_router_ctas =
        MoEDims::KernelConfig::kNumCtasForRouter;
    const std::uint32_t num_latent_ctas =
        problem.use_latent ? MoEDims::KernelConfig::kNumCtasForLatentDown : 0u;
    const std::uint32_t num_shared_ctas =
        MoEDims::KernelConfig::kNumCtasForShared;

    // Every phase's CTA budget must cover whole tiles.
    const std::uint32_t num_latent_tiles =
        problem.latent_size / LatentDownGemm::kTileShapeM;
    const std::uint32_t num_up_tiles =
        2u * problem.intermediate_size / SharedUpGemm::kTileShapeM;
    const std::uint32_t num_down_tiles =
        problem.hidden_size / SharedDownGemm::kTileShapeM;
    if (problem.use_latent) {
        TORCH_CHECK(num_latent_ctas % num_latent_tiles == 0u,
                    "the latent CTA budget (", num_latent_ctas,
                    ") must cover whole 128-row M tiles of latent_size=",
                    problem.latent_size, ".");
    }
    TORCH_CHECK(num_shared_ctas % num_up_tiles == 0u, "the shared CTA budget (",
                num_shared_ctas,
                ") must cover whole up-projection slabs of 2*intermediate=",
                2u * problem.intermediate_size, ".");
    TORCH_CHECK(
        problem.hidden_size == num_shared_ctas * SharedDownGemm::kTileShapeM,
        "the shared up and down passes need the same CTA count (",
        num_shared_ctas, " vs ", num_down_tiles, ").");
    const std::uint32_t num_ctas_per_expert =
        MoEDims::KernelConfig::kCtasPerExpert;
    TORCH_CHECK(num_ctas_per_expert <= moe::MoERuntime::kMaxCtasPerExpert,
                "the routed experts run on 1..",
                moe::MoERuntime::kMaxCtasPerExpert, " blocks each, got ",
                num_ctas_per_expert, ".");

    const std::uint32_t num_latent_splits =
        problem.use_latent ? num_latent_ctas / num_latent_tiles : 1u;
    const std::uint32_t num_shared_splits = num_shared_ctas / num_up_tiles;
    const std::uint32_t cluster_size = num_latent_splits > num_shared_splits
                                           ? num_latent_splits
                                           : num_shared_splits;

    moe::MoERuntime runtime(problem, num_router_ctas, num_latent_ctas,
                            num_shared_ctas);
    runtime.num_latent_splits = num_latent_splits;
    runtime.num_shared_splits = num_shared_splits;
    runtime.cluster_size = cluster_size;
    runtime.num_ctas_per_expert = num_ctas_per_expert;
    runtime.ctas_per_expert_divmod =
        cutlass::FastDivmod(static_cast<int>(num_ctas_per_expert));
    runtime.latent_splits_divmod =
        cutlass::FastDivmod(static_cast<int>(num_latent_splits));
    runtime.shared_splits_divmod =
        cutlass::FastDivmod(static_cast<int>(num_shared_splits));
    runtime.latent_tiles_divmod =
        cutlass::FastDivmod(static_cast<int>(std::max(1u, num_latent_tiles)));
    runtime.cluster_divmod =
        cutlass::FastDivmod(static_cast<int>(cluster_size));
    runtime.num_tail_ctas = TailGeometry::kNumCtas;
    return runtime;
}

// All CTAs must remain resident for the kernel lifetime.
template <typename MoEDims>
uint32_t CalculateGridSize(int device, cudaStream_t stream,
                           moe::MoERuntime const& runtime,
                           std::uint32_t grid_cap = 0u) {
    using Traits = moe::MoEKernelTraits<MoEDims>;
    using Mainloop = moe::CollectiveMainloopMoE<Traits>;

    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);

    uint32_t grid_size =
        std::min(static_cast<std::uint32_t>(sm_count), Traits::kMaxGrid);
    if (grid_cap != 0u) grid_size = std::min(grid_size, grid_cap);
    const std::uint32_t kClusterSize = runtime.cluster_size;
    if (kClusterSize > 1u) {
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
    TORCH_CHECK(grid_size >= runtime.get_min_grid(),
                "The kernel needs at least ", runtime.get_min_grid(),
                " co-resident blocks (router, latent and shared expert), but "
                "the launch grid has only ",
                grid_size,
                grid_cap > 0u ? " (grid_cap is set)" : " (the device limit).");
    return grid_size;
}

template <typename MoEDims>
void MoEImpl(
    const torch::Tensor& input, const torch::Tensor& router_weight,
    moe::MoEProblemSize const& problem, const torch::Tensor& expert_up_weights,
    const torch::Tensor& expert_up_weights_scale,
    const torch::Tensor& expert_down_weights,
    const torch::Tensor& expert_down_weights_scale,
    const torch::Tensor& shared_up_weight,
    const torch::Tensor& shared_up_weight_scale,
    const torch::Tensor& shared_down_weight,
    const torch::Tensor& shared_down_weight_scale,
    const torch::Tensor& latent_down_weight,
    const torch::Tensor& latent_down_weight_scale,
    const torch::Tensor& tail_fc2_weight, const torch::Tensor& tail_fc2_scale,
    const torch::Tensor& tail_norm_weight,
    const std::optional<torch::Tensor>& correction_bias,
    const std::optional<torch::Tensor>& per_expert_scale,
    const torch::Tensor& tail_epochs, const torch::Tensor& tail_uc_bases,
    const torch::Tensor& workspace, const torch::Tensor& tail_out,
    int64_t top_k, int64_t scoring_func, int64_t activation, int64_t tp_rank,
    int64_t tail_mc_base, int64_t tail_grid_cap, bool renormalize,
    double situ_beta, double situ_linear_beta, double tail_eps, bool use_pdl,
    bool routed_use_swizzle, bool shared_use_swizzle, bool latent_use_swizzle) {
    TORCH_CHECK(
        top_k >= 1 && top_k <= static_cast<std::int64_t>(MoEDims::kMaxTopk),
        "top_k must be between 1 and ", MoEDims::kMaxTopk, ".");

    CheckRouterWeightShape(router_weight, problem);
    CheckRoutedExpertShape<MoEDims>(expert_up_weights, expert_down_weights,
                                    problem);
    if (latent_down_weight.defined()) {
        CheckLatentDownShape(latent_down_weight, problem);
    }
    CheckSharedExpertShape(shared_up_weight, shared_down_weight, problem);
    CheckTailFc2Shape<MoEDims>(tail_fc2_weight, tail_fc2_scale, problem);
    CheckRoutingVectors(correction_bias, per_expert_scale, problem);

    using Traits = moe::MoEKernelTraits<MoEDims>;
    using Mainloop = moe::CollectiveMainloopMoE<Traits>;
    using TailGeometry = moe::TailGeometryT<MoEDims>;

    typename Mainloop::Arguments args{};
    args.problem_size = problem;
    args.layouts = moe::MoELayouts(routed_use_swizzle, shared_use_swizzle,
                                   latent_use_swizzle);
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
        problem.use_latent
            ? static_cast<
                  const moe::LatentDownGemmTraits<MoEDims>::MmaElement*>(
                  latent_down_weight.const_data_ptr())
            : nullptr;
    using LatentScale =
        typename moe::LatentDownGemmTraits<MoEDims>::ElementScale;
    args.latent_down_weight_scale =
        problem.use_latent ? static_cast<const LatentScale*>(
                                 latent_down_weight_scale.const_data_ptr())
                           : nullptr;

    const auto scratch_bytes = MoEWorkspace<MoEDims>::bytes(problem);
    TORCH_CHECK(workspace.nbytes() >= scratch_bytes, "MoE scratchpad needs ",
                scratch_bytes, " bytes, got ", workspace.nbytes(), ".");
    args.workspace.bind(workspace.data_ptr(), problem);

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

    args.runtime = BuildRuntimeConfig<MoEDims>(problem);

    const uint32_t grid_size =
        CalculateGridSize<MoEDims>(device, stream, args.runtime, grid_cap);

    const cudaError_t launch_status = moe::LaunchMoEKernel<MoEDims>(
        args, grid_size, args.runtime.cluster_size, stream, use_pdl);
    TORCH_CHECK(launch_status == cudaSuccess,
                "kernel launch failed: ", cudaGetErrorString(launch_status));
}

template <class Dims, class GemmTraits>
void PrintGemmInfo(const char* phase) {
    using Slots = GemmSlots<Dims, GemmTraits>;
    printf(
        "  %-12s TileShape=(%u,%u,%u) Slots=%u ScaleSlots=%u TmemAccSlots=%u "
        "TmemScaleSlots=%u %s\n",
        phase, GemmTraits::kTileShapeM, GemmTraits::kTileShapeN,
        GemmTraits::kTileShapeK, Slots::kNumSlots, Slots::kScaleSlots,
        Slots::kTmemAccSlots, Slots::kTmemScaleSlots,
        GemmTraits::MmaCore::kUseBlockscaled ? "blockscaled" : "blockwise");
}

template <class Dims>
void PrintDimsKernelInfo(const char* tag) {
    using Traits = MoEKernelTraits<Dims>;

    printf("%s: SMEM=%zu bytes\n", tag, get_moe_smem_size<Dims>());
    PrintGemmInfo<Dims, typename Traits::RouterGemm>("router");
    PrintGemmInfo<Dims, typename Traits::LatentDownGemm>("latent_down");
    PrintGemmInfo<Dims, typename Traits::RoutedUpGemm>("routed_up");
    PrintGemmInfo<Dims, typename Traits::RoutedDownGemm>("routed_down");
    PrintGemmInfo<Dims, typename Traits::SharedUpGemm>("shared_up");
    PrintGemmInfo<Dims, typename Traits::SharedDownGemm>("shared_down");
}

}  // namespace moe

struct MoEKernelDispatcher {
    using QuantBWFP8 = moe::QuantConfig<moe::QuantMethod::kBlockwiseFP8,
                                        moe::QuantMethod::kBlockwiseFP8,
                                        moe::QuantMethod::kBlockwiseFP8>;
    using QuantMXFP8 =
        moe::QuantConfig<moe::QuantMethod::kMXFP8, moe::QuantMethod::kMXFP8,
                         moe::QuantMethod::kMXFP8>;
    using QuantMXFP4 =
        moe::QuantConfig<moe::QuantMethod::kMXFP8, moe::QuantMethod::kMXFP8,
                         moe::QuantMethod::kMXFP4>;

    enum class QuantRecipe : std::uint32_t { kBWFP8, kMXFP8, kW4A8 };

    template <typename... Ts>
    struct TypeList {};

    template <typename... A, typename... B>
    static constexpr TypeList<A..., B...> concat(TypeList<A...>,
                                                 TypeList<B...>) {
        return {};
    }

    // The Dims one (TP world, token count) pair launches, one per recipe.
    template <int WorldSize, std::uint32_t Tokens>
    using DimsPerWorld =
        TypeList<moe::MoEDims_E512_N512_K2048<QuantBWFP8{}, Tokens, WorldSize>,
                 moe::MoEDims_E512_N512_K2048<QuantMXFP8{}, Tokens, WorldSize>,
                 moe::MoEDims_E512_N512_K2048<QuantMXFP4{}, Tokens, WorldSize>>;

    using DimsAll =
        decltype(concat(concat(DimsPerWorld<1, 8>{}, DimsPerWorld<4, 8>{}),
                        concat(DimsPerWorld<1, 16>{}, DimsPerWorld<4, 16>{})));

    template <typename Dims>
    static constexpr QuantRecipe recipe_of() {
        if constexpr (Dims::kQuantConfig.kExpertsQuant ==
                      moe::QuantMethod::kMXFP4) {
            return QuantRecipe::kW4A8;
        } else if constexpr (Dims::kQuantConfig.kLatentQuant ==
                             moe::QuantMethod::kMXFP8) {
            return QuantRecipe::kMXFP8;
        } else {
            return QuantRecipe::kBWFP8;
        }
    }

    template <typename Dims>
    static constexpr bool is_wide() {
        return Dims::kNumTokens > 8u;
    }

    template <typename Dims>
    static constexpr const char* tag() {
        if constexpr (recipe_of<Dims>() == QuantRecipe::kW4A8) {
            return is_wide<Dims>() ? "BS16-w4a8-mxfp4" : "BS8-w4a8-mxfp4";
        } else if constexpr (recipe_of<Dims>() == QuantRecipe::kMXFP8) {
            return is_wide<Dims>() ? "BS16-w8a8-mxfp8" : "BS8-w8a8-mxfp8";
        } else {
            return is_wide<Dims>() ? "BS16-w8a8-blockwise"
                                   : "BS8-w8a8-blockwise";
        }
    }

    template <typename... Dims>
    static int64_t workspace_size(TypeList<Dims...>,
                                  const moe::MoEProblemSize& problem) {
        std::size_t bytes = 0;
        auto include = [&](auto dims) {
            using D = decltype(dims);
            if (problem.num_tokens == 0u ||
                is_wide<D>() == (problem.num_tokens > 8u)) {
                bytes = std::max(bytes, moe::MoEWorkspace<D>::bytes(problem));
            }
        };
        (include(Dims{}), ...);
        return static_cast<int64_t>(bytes);
    }

    template <typename... Dims>
    static constexpr int64_t tail_bytes_of(TypeList<Dims...>, bool wide,
                                           int64_t tp_size) {
        int64_t bytes = 0;
        ((bytes = (is_wide<Dims>() == wide &&
                           Dims::KernelConfig::kTailWorld == tp_size &&
                           recipe_of<Dims>() == QuantRecipe::kBWFP8
                       ? moe::tail_region_bytes<Dims>()
                       : bytes)),
         ...);
        return bytes;
    }

    static int64_t tail_region_bytes(bool wide, int64_t tp_size) {
        return tail_bytes_of(DimsAll{}, wide, tp_size);
    }

    template <typename... Dims>
    static void print_matching(TypeList<Dims...>, const std::string& mode) {
        const auto wanted = [&mode](const char* name) {
            return mode.empty() ||
                   std::string(name).find(mode) != std::string::npos;
        };
        ((wanted(tag<Dims>()) && Dims::KernelConfig::kTailWorld == 1
              ? (moe::PrintDimsKernelInfo<Dims>(tag<Dims>()), 0)
              : 0),
         ...);
    }

    static void print(const std::string& quant_mode) {
        print_matching(DimsAll{}, quant_mode);
    }

    // Invoke `launch(Dims{})` for the one matching schedule.
    template <typename Launch, typename... Dims>
    static bool launch_matching(TypeList<Dims...>, int64_t tp_size, bool wide,
                                QuantRecipe recipe, Launch&& launch) {
        return ((is_wide<Dims>() == wide &&
                         Dims::KernelConfig::kTailWorld == tp_size &&
                         recipe_of<Dims>() == recipe
                     ? (launch(Dims{}), true)
                     : false) ||
                ...);
    }

    static QuantRecipe mode_to_recipe(const std::string& quant_mode) {
        if (quant_mode == "w4a8-mxfp4") return QuantRecipe::kW4A8;
        if (quant_mode == "w8a8-mxfp8") return QuantRecipe::kMXFP8;
        return QuantRecipe::kBWFP8;
    }

    template <typename Launch>
    static bool launch(int64_t tp_size, bool wide,
                       const std::string& quant_mode, Launch&& launch) {
        return launch_matching(DimsAll{}, tp_size, wide,
                               mode_to_recipe(quant_mode),
                               std::forward<Launch>(launch));
    }
};

int64_t GetWorkspaceSize(std::uint32_t max_tokens, std::uint32_t hidden_size,
                         std::uint32_t intermediate_size,
                         std::uint32_t latent_size, std::uint32_t num_experts,
                         std::uint32_t num_router_gates) {
    TORCH_CHECK(max_tokens <= 16u, "MoE workspace supports at most 16 tokens.");
    return MoEKernelDispatcher::workspace_size(
        MoEKernelDispatcher::DimsAll{},
        {max_tokens, hidden_size, intermediate_size, latent_size, num_experts,
         num_router_gates});
}

void PrintMoEKernelInfo(const std::string& quant_mode) {
    MoEKernelDispatcher::print(quant_mode);
    fflush(stdout);
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
               bool use_pdl, bool routed_use_swizzle, bool shared_use_swizzle,
               bool latent_use_swizzle) {
    TORCH_CHECK(
        quant_mode == "w8a8-mxfp8" || quant_mode == "w4a8-mxfp4" ||
            quant_mode == "w8a8-blockwise",
        "quant_mode must be w8a8-blockwise, w8a8-mxfp8 or w4a8-mxfp4, got ",
        quant_mode, ".");

    const moe::MoEProblemSize problem =
        moe::BuildProblem(input, router_weight, shared_down_weight,
                          latent_down_weight, num_router_gates);

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
            input, router_weight, problem, expert_up_weights,
            expert_up_weights_scale, expert_down_weights,
            expert_down_weights_scale, shared_up_weight, shared_up_weight_scale,
            shared_down_weight, shared_down_weight_scale, latent_down_weight,
            latent_down_weight_scale, tail_fc2_weight, tail_fc2_scale,
            tail_norm_weight, correction_bias, per_expert_scale, tail_epochs,
            tail_uc_bases, workspace, tail_out, top_k, scoring_type,
            activation_type, tp_rank, tail_mc_base, tail_grid_cap, renormalize,
            situ_beta, situ_linear_beta, tail_eps, use_pdl, routed_use_swizzle,
            shared_use_swizzle, latent_use_swizzle);
    };

    const bool launched = MoEKernelDispatcher::launch(
        tp_size, problem.num_tokens > 8, quant_mode,
        [&](auto dims_tag) { launch_func(dims_tag); });
    TORCH_CHECK(launched, "no schedule matches quant_mode=", quant_mode,
                ", tp_size=", tp_size, ".");
}

int64_t MoETailRegionBytes(bool wide, int64_t tp_size) {
    TORCH_CHECK(tp_size == 1 || tp_size == 4, "tp_size must be 1 or 4.");
    return MoEKernelDispatcher::tail_region_bytes(wide, tp_size);
}

}  // namespace halops
