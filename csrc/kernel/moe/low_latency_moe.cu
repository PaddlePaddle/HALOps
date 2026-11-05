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
#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <type_traits>

#include "common/cuda_utils.h"
#include "common/kernel_timer_host.hpp"
#include "kernel/moe/moe_kernel.cuh"

namespace halops {

namespace moe {

void CheckTensor(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.defined() && tensor.is_cuda() && tensor.is_contiguous(),
                "moe: ", name, " must be a contiguous CUDA tensor.");
}

void CheckTensorType(const torch::Tensor& tensor, at::ScalarType expected,
                     const char* name) {
    TORCH_CHECK(tensor.defined() && tensor.is_cuda() &&
                    tensor.is_contiguous() && tensor.scalar_type() == expected,
                "moe: ", name, " must be a contiguous CUDA tensor of type ",
                c10::toString(expected), ".");
}

#define MOE_CHECK_TENSOR(x) ::halops::moe::CheckTensor((x), #x)
#define MOE_CHECK_TENSOR_TYPE(x, expected) \
    ::halops::moe::CheckTensorType((x), (expected), #x)

template <typename MoEDims>
void CheckRouterShape(const torch::Tensor& router_weight) {
    using RouterGemm = moe::RouterGemmTraits<MoEDims>;
    constexpr std::uint32_t kNumBoxKs =
        RouterGemm::kNumTilesK * RouterGemm::kNumSwizzleBoxKs;
    constexpr std::uint32_t kPackedTiles =
        RouterGemm::kNumRouterGates * RouterGemm::kNumTilesM;
    TORCH_CHECK(
        router_weight.dim() == 4 && router_weight.size(0) == kPackedTiles &&
            router_weight.size(1) == kNumBoxKs &&
            router_weight.size(2) == RouterGemm::kTileShapeM &&
            router_weight.size(3) == RouterGemm::kSwizzleBoxK,
        "moe: router_weight must be the packed [", kPackedTiles, ", ",
        kNumBoxKs, ", ", RouterGemm::kTileShapeM, ", ",
        RouterGemm::kSwizzleBoxK, "] layout (see prepare_router_weight).");
    // Each tile is one bulk copy, which needs a 16-B aligned source.
    TORCH_CHECK(
        reinterpret_cast<std::uintptr_t>(router_weight.const_data_ptr()) %
                16u ==
            0u,
        "moe: router_weight must be 16-byte aligned.");
}

template <typename MoEDims>
void CheckLatentDownShape(const torch::Tensor& latent_down_weight,
                          const torch::Tensor& latent_down_weight_scale) {
    using Quant = moe::QuantSpec<MoEDims::kQuantConfig.kLatentQuant>;
    // Every MX recipe carries a one-byte UE8M0 scale; only block-wise
    // recipes use an fp32 table.
    constexpr bool kUint8Scale = Quant::kMethod != QuantMethod::kBlockwiseFP8;
    constexpr auto kScaleShape =
        Quant::template ScaleShape<MoEDims::kLatentSize,
                                   MoEDims::kHiddenSize>();
    constexpr std::int64_t kScaleRows =
        static_cast<std::int64_t>(cute::get<0>(kScaleShape));
    constexpr std::int64_t kScaleCols =
        static_cast<std::int64_t>(cute::get<1>(kScaleShape));
    TORCH_CHECK(latent_down_weight.dim() == 2 &&
                    latent_down_weight.size(0) == MoEDims::kLatentSize &&
                    latent_down_weight.size(1) == MoEDims::kHiddenSize,
                "moe: latent_down_weight must be [kLatentSize, kHiddenSize].");
    TORCH_CHECK(latent_down_weight_scale.dim() == 2 &&
                    latent_down_weight_scale.size(0) == kScaleRows &&
                    latent_down_weight_scale.size(1) == kScaleCols,
                "moe: latent_down_weight_scale must be [", kScaleRows, ", ",
                kScaleCols, "].");
    TORCH_CHECK(latent_down_weight_scale.scalar_type() ==
                    (kUint8Scale ? at::kByte : at::kFloat),
                "moe: latent_down_weight_scale has the wrong dtype.");
}

template <typename MoEDims>
void CheckRoutedExpertShape(const torch::Tensor& expert_up_weights,
                            const torch::Tensor& expert_up_weights_scale,
                            const torch::Tensor& expert_down_weights,
                            const torch::Tensor& expert_down_weights_scale) {
    constexpr std::int64_t kExperts = MoEDims::kNumExperts;
    constexpr std::int64_t kIntermediate = MoEDims::kIntermediateSize;
    constexpr std::int64_t kExpertK = moe::MoEInputSize<MoEDims>::value;
    constexpr std::int64_t kUpRows = 2 * kIntermediate;
    // A 4-bit weight packs two values per byte along K.
    constexpr std::int64_t kWeightBytes =
        MoEDims::kQuantConfig.kExpertsQuant == moe::QuantMethod::kMXFP4 ? 2 : 1;
    constexpr auto kUpScaleShape =
        moe::QuantSpec<MoEDims::kQuantConfig.kExpertsQuant>::
            template ScaleShape<kUpRows, kExpertK>();
    constexpr auto kDownScaleShape =
        moe::QuantSpec<MoEDims::kQuantConfig.kExpertsQuant>::
            template ScaleShape<kExpertK, kIntermediate>();
    constexpr std::int64_t kUpScaleRows = cute::get<0>(kUpScaleShape);
    constexpr std::int64_t kUpScaleCols = cute::get<1>(kUpScaleShape);
    constexpr std::int64_t kDownScaleRows = cute::get<0>(kDownScaleShape);
    constexpr std::int64_t kDownScaleCols = cute::get<1>(kDownScaleShape);
    TORCH_CHECK(
        expert_up_weights.dim() == 3 && expert_up_weights.size(0) == kExperts &&
            expert_up_weights.size(1) == kUpRows &&
            expert_up_weights.size(2) == kExpertK / kWeightBytes,
        "moe: expert_up_weights must be [E, 2N, K / ", kWeightBytes, "].");
    TORCH_CHECK(expert_up_weights_scale.dim() == 3 &&
                    expert_up_weights_scale.size(0) == kExperts &&
                    expert_up_weights_scale.size(1) == kUpScaleRows &&
                    expert_up_weights_scale.size(2) == kUpScaleCols,
                "moe: expert_up_weights_scale must be [E, ", kUpScaleRows, ", ",
                kUpScaleCols, "].");
    TORCH_CHECK(expert_down_weights.dim() == 3 &&
                    expert_down_weights.size(0) == kExperts &&
                    expert_down_weights.size(1) == kExpertK &&
                    expert_down_weights.size(2) == kIntermediate / kWeightBytes,
                "moe: expert_down_weights must be [E, K, N / ", kWeightBytes,
                "].");
    TORCH_CHECK(expert_down_weights_scale.dim() == 3 &&
                    expert_down_weights_scale.size(0) == kExperts &&
                    expert_down_weights_scale.size(1) == kDownScaleRows &&
                    expert_down_weights_scale.size(2) == kDownScaleCols,
                "moe: expert_down_weights_scale must be [E, ", kDownScaleRows,
                ", ", kDownScaleCols, "].");
}

template <typename MoEDims>
void CheckSharedExpertShape(const torch::Tensor& input,
                            const torch::Tensor& shared_up_weight,
                            const torch::Tensor& shared_up_weight_scale,
                            const torch::Tensor& shared_down_weight,
                            const torch::Tensor& shared_down_weight_scale,
                            const torch::Tensor& shared_out) {
    constexpr std::int64_t kUpRows = 2 * MoEDims::kIntermediateSize;
    // The scale table follows the group's recipe: 128x128 fp32 blocks, or
    // one UE8M0 byte per 32-K group.
    constexpr auto kUpScaleShape =
        moe::QuantSpec<MoEDims::kQuantConfig.kSharedQuant>::template ScaleShape<
            kUpRows, MoEDims::kHiddenSize>();
    constexpr auto kDownScaleShape =
        moe::QuantSpec<MoEDims::kQuantConfig.kSharedQuant>::template ScaleShape<
            MoEDims::kHiddenSize, MoEDims::kIntermediateSize>();
    constexpr std::int64_t kUpScaleRows = cute::get<0>(kUpScaleShape);
    constexpr std::int64_t kUpScaleCols = cute::get<1>(kUpScaleShape);
    constexpr std::int64_t kDownScaleRows = cute::get<0>(kDownScaleShape);
    constexpr std::int64_t kDownScaleCols = cute::get<1>(kDownScaleShape);
    TORCH_CHECK(shared_up_weight.dim() == 2 &&
                    shared_up_weight.size(0) == kUpRows &&
                    shared_up_weight.size(1) == MoEDims::kHiddenSize,
                "moe: shared_up_weight must be the pair-layout [2N, H].");
    TORCH_CHECK(shared_up_weight_scale.dim() == 2 &&
                    shared_up_weight_scale.size(0) == kUpScaleRows &&
                    shared_up_weight_scale.size(1) == kUpScaleCols,
                "moe: shared_up_scale must be [", kUpScaleRows, ", ",
                kUpScaleCols, "].");
    TORCH_CHECK(shared_down_weight.dim() == 2 &&
                    shared_down_weight.size(0) == MoEDims::kHiddenSize &&
                    shared_down_weight.size(1) == MoEDims::kIntermediateSize,
                "moe: shared_down_weight must be [H, N].");
    TORCH_CHECK(shared_down_weight_scale.dim() == 2 &&
                    shared_down_weight_scale.size(0) == kDownScaleRows &&
                    shared_down_weight_scale.size(1) == kDownScaleCols,
                "moe: shared_down_scale must be [", kDownScaleRows, ", ",
                kDownScaleCols, "].");
    TORCH_CHECK(shared_out.dim() == 2 && shared_out.size(0) == input.size(0) &&
                    shared_out.size(1) == MoEDims::kHiddenSize,
                "moe: shared_out must be [tokens, H].");
}

// One block per SM, with no slack: every block must stay scheduled for the
// kernel's full lifetime, so a single unavailable SM deadlocks.
template <typename MoEDims>
uint32_t CalculateGridSize(int device, cudaStream_t stream,
                           std::uint32_t grid_cap = 0u) {
    using Traits = moe::MoEKernelTraits<MoEDims>;
    using Mainloop = moe::CollectiveMainloopMoE<Traits>;

    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
    TORCH_CHECK(sm_count >= static_cast<int>(Traits::kMinGrid),
                "moe: needs at least kMinGrid (=", Traits::kMinGrid,
                ") SMs for the co-residency invariant; got ", sm_count);
    // The shared expert runs on the blocks the router and the projection
    // leave idle.
    constexpr std::uint32_t kSharedCtas =
        Traits::RouterGemm::kNumCtas + Traits::LatentDownGemm::kNumCtas +
        moe::SharedDownGemmTraits<MoEDims>::kNumCtas;
    TORCH_CHECK(static_cast<std::uint32_t>(sm_count) >= kSharedCtas,
                "moe: the shared expert needs the grid to reach its ",
                moe::SharedDownGemmTraits<MoEDims>::kNumCtas,
                " blocks, i.e. at least ", kSharedCtas, " SMs; got ", sm_count);
    uint32_t grid_size =
        std::min(static_cast<std::uint32_t>(sm_count), Traits::kMaxGrid);
    if (grid_cap != 0u) grid_size = std::min(grid_size, grid_cap);

    constexpr std::uint32_t kClusterSize = Traits::kClusterSize;
    if constexpr (kClusterSize > 1u) {
        // A grid past the clusters the driver can co-schedule would leave
        // blocks unscheduled, i.e. break co-residency.
        grid_size -= grid_size % kClusterSize;
        cudaLaunchAttribute cluster_attr{};
        cluster_attr.id = cudaLaunchAttributeClusterDimension;
        cluster_attr.val.clusterDim.x = kClusterSize;
        cluster_attr.val.clusterDim.y = 1u;
        cluster_attr.val.clusterDim.z = 1u;
        constexpr size_t smem_size = moe::get_moe_smem_size<MoEDims>();
        auto* kernel_ptr = &moe::fused_moe_kernel<Traits, Mainloop>;
        // The >48 KB carve-out has to be opted into before the driver can
        // size the block.
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
        TORCH_CHECK(grid_size >= Traits::kMinGrid,
                    "moe: needs at least kMinGrid (=", Traits::kMinGrid,
                    ") co-resident blocks, but the device's ", sm_count,
                    " SMs fit only ", grid_size, " as ", kClusterSize,
                    "-block clusters.");
    }
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
             const torch::Tensor& workspace, const torch::Tensor& shared_out,
             const torch::Tensor& tail_fc2_weight,
             const torch::Tensor& tail_fc2_scale,
             const torch::Tensor& tail_norm_weight,
             const torch::Tensor& tail_out, const torch::Tensor& tail_epochs,
             const torch::Tensor& tail_uc_bases, int64_t top_k,
             int64_t scoring_func, int64_t activation, int64_t tail_rank,
             int64_t tail_mc_base, int64_t tail_grid_cap, bool renormalize,
             double situ_beta, double situ_linear_beta, double tail_eps) {
    const uint32_t num_tokens = static_cast<uint32_t>(input.size(0));
    TORCH_CHECK(num_tokens >= 1 && num_tokens <= MoEDims::kNumTokens,
                "moe: token count M=", num_tokens,
                " out of range (this schedule serves 1 <= M <= ",
                MoEDims::kNumTokens, ").");

    constexpr std::int64_t kMaxTopk =
        static_cast<std::int64_t>(MoEDims::kMaxTopk);
    TORCH_CHECK(top_k >= 1 && top_k <= kMaxTopk,
                "moe: top_k must be between 1 and ", kMaxTopk, ".");

    CheckRouterShape<MoEDims>(router_weight);
    CheckRoutedExpertShape<MoEDims>(expert_up_weights, expert_up_weights_scale,
                                    expert_down_weights,
                                    expert_down_weights_scale);
    if constexpr (MoEDims::kUseLatent) {
        CheckLatentDownShape<MoEDims>(latent_down_weight,
                                      latent_down_weight_scale);
    }
    CheckSharedExpertShape<MoEDims>(input, shared_up_weight,
                                    shared_up_weight_scale, shared_down_weight,
                                    shared_down_weight_scale, shared_out);

    using Traits = moe::MoEKernelTraits<MoEDims>;
    using Mainloop = moe::CollectiveMainloopMoE<Traits>;

    typename Mainloop::Arguments args{};
    args.num_tokens = num_tokens;
    args.hidden_size = static_cast<uint32_t>(input.size(1));
    args.latent_size = MoEDims::kUseLatent
                           ? static_cast<uint32_t>(latent_down_weight.size(0))
                           : 0u;
    args.intermediate_size = static_cast<uint32_t>(shared_down_weight.size(1));
    args.top_k = static_cast<uint32_t>(top_k);
    args.scoring_func = static_cast<ScoringFunc>(scoring_func);
    args.renormalize = renormalize;
    args.activation = static_cast<Activation>(activation);
    args.situ_beta = static_cast<float>(situ_beta);
    args.situ_linear_beta = static_cast<float>(situ_linear_beta);

    args.input = static_cast<const DataType::ElementA*>(input.const_data_ptr());
    args.router_weight =
        static_cast<const DataType::ElementA*>(router_weight.const_data_ptr());

    args.expert_up_weights =
        static_cast<const moe::UpGemmTraits<MoEDims>::MmaElement*>(
            expert_up_weights.const_data_ptr());
    args.expert_up_weights_scale =
        static_cast<const moe::UpGemmTraits<MoEDims>::ElementScale*>(
            expert_up_weights_scale.const_data_ptr());
    args.expert_down_weights =
        static_cast<const moe::DownGemmTraits<MoEDims>::MmaElement*>(
            expert_down_weights.const_data_ptr());
    args.expert_down_weights_scale =
        static_cast<const moe::DownGemmTraits<MoEDims>::ElementScale*>(
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
    args.shared_out = static_cast<DataType::ElementOut*>(shared_out.data_ptr());

    const int device = input.get_device();
    int l2_cached_bytes = 0;
    cudaDeviceGetAttribute(&l2_cached_bytes, cudaDevAttrL2CacheSize, device);
    args.l2_cached_bytes =
        l2_cached_bytes > 0 ? static_cast<uint32_t>(l2_cached_bytes) : 0u;

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(device).stream();
    const std::uint32_t grid_cap = static_cast<std::uint32_t>(tail_grid_cap);
    auto& tp = args.tail;
    const int64_t* uc_bases = tail_uc_bases.const_data_ptr<int64_t>();
    for (int r = 0; r < MoEDims::KernelConfig::kTailWorld; ++r) {
        tp.uc_base[r] = reinterpret_cast<std::uint8_t*>(uc_bases[r]);
    }
    tp.mc_base = reinterpret_cast<std::uint8_t*>(
        static_cast<std::uintptr_t>(tail_mc_base));
    tp.epochs = static_cast<std::uint32_t*>(tail_epochs.data_ptr());
    tp.fc2_w8 =
        static_cast<const std::uint8_t*>(tail_fc2_weight.const_data_ptr());
    tp.fc2_scale = static_cast<const float*>(tail_fc2_scale.const_data_ptr());
    tp.norm_weight = static_cast<const decltype(tp.norm_weight)>(
        tail_norm_weight.const_data_ptr());
    tp.out = static_cast<decltype(tp.out)>(tail_out.data_ptr());
    tp.eps = static_cast<float>(tail_eps);
    tp.m = num_tokens;
    tp.rank = static_cast<std::uint32_t>(tail_rank);
    tp.valid_counts = nullptr;
    tp.slot_rows = 0;
    const uint32_t grid_size =
        CalculateGridSize<MoEDims>(device, stream, grid_cap);
    constexpr std::uint32_t kSharedEnd =
        Traits::RouterGemm::kNumCtas + Traits::LatentDownGemm::kNumCtas +
        moe::SharedDownGemmTraits<MoEDims>::kNumCtas;
    TORCH_CHECK(grid_size >= kSharedEnd,
                "moe: grid_cap leaves the shared expert's blocks outside "
                "the grid: need at least ",
                kSharedEnd, " CTAs; got ", grid_size, ".");

    // Device event timer: zero this launch's slice of the record pool, hand
    // its pointer to the kernel, and copy the records back afterwards.  It is
    // a debug path, so a CUDA-graph capture runs with the pool off.
#if defined(HALOPS_KERNEL_TIMER_ENABLED)
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(stream, &capture);
    const bool timer_on = capture == cudaStreamCaptureStatusNone;
    if (timer_on) {
        halops::TimerStage(static_cast<int>(grid_size), stream);
    }
    args.timer_events = timer_on ? halops::TimerDeviceBuffer() : nullptr;
#endif

    const cudaError_t launch_status =
        moe::LaunchMoEKernel<MoEDims>(args, grid_size, stream);
    TORCH_CHECK(launch_status == cudaSuccess, "moe: kernel launch failed: ",
                cudaGetErrorString(launch_status));
#if defined(HALOPS_KERNEL_TIMER_ENABLED)
    if (timer_on) {
        halops::TimerCollect(stream);
    }
#endif
}

}  // namespace moe

// The workspace is sized for the largest whitelisted recipe: the layout is
// recipe-dependent (the latent output turns into its bf16 variant when the
// experts run MXFP8), while the caller allocates without a recipe.
template <class Dims>
constexpr int64_t WorkspaceBytesFor() {
    return static_cast<int64_t>(sizeof(moe::MoEWorkspace<Dims>));
}

int64_t GetWorkspaceSize(const std::optional<torch::Tensor>& input) {
    using moe::QuantConfig;
    using moe::QuantMethod;
    constexpr auto kBW = QuantMethod::kBlockwiseFP8;
    constexpr auto kMX = QuantMethod::kMXFP8;
    constexpr auto kMX4 = QuantMethod::kMXFP4;
    constexpr int64_t kBs8Bytes =
        std::max({WorkspaceBytesFor<moe::Dims_BS8_E512_N512_K2048_SM100<
                      QuantConfig<kBW, kBW, kBW>{}>>(),
                  WorkspaceBytesFor<moe::Dims_BS8_E512_N512_K2048_SM100<
                      QuantConfig<kMX, kMX, kMX>{}>>(),
                  WorkspaceBytesFor<moe::Dims_BS8_E512_N512_K2048_SM100<
                      QuantConfig<kMX, kMX, kMX4>{}>>()});
    constexpr int64_t kBs16Bytes =
        std::max({WorkspaceBytesFor<moe::Dims_BS16_E512_N512_K2048_SM100<
                      QuantConfig<kBW, kBW, kBW>{}>>(),
                  WorkspaceBytesFor<moe::Dims_BS16_E512_N512_K2048_SM100<
                      QuantConfig<kMX, kMX, kMX>{}>>(),
                  WorkspaceBytesFor<moe::Dims_BS16_E512_N512_K2048_SM100<
                      QuantConfig<kMX, kMX, kMX4>{}>>()});
    if (!input.has_value()) {
        return kBs8Bytes > kBs16Bytes ? kBs8Bytes : kBs16Bytes;
    }
    constexpr int64_t kBs8Tokens =
        static_cast<int64_t>(moe::Dims_BS8_E512_N512_K2048_SM100<
                             QuantConfig<kBW, kBW, kBW>{}>::kNumTokens);
    return input->size(0) <= kBs8Tokens ? kBs8Bytes : kBs16Bytes;
}

void LowLatencyMoE(
    const torch::Tensor& input, const torch::Tensor& router_weight,
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
    const torch::Tensor& workspace, const torch::Tensor& shared_out,
    const torch::Tensor& tail_fc2_weight, const torch::Tensor& tail_fc2_scale,
    const torch::Tensor& tail_norm_weight, const torch::Tensor& tail_out,
    const torch::Tensor& tail_epochs, const torch::Tensor& tail_uc_bases,
    int64_t top_k, int64_t num_router_gates, int64_t tail_rank,
    int64_t tail_mc_base, int64_t tail_grid_cap,
    const std::string& scoring_func, const std::string& activation,
    const std::string& quant_mode, bool renormalize, double situ_beta,
    double situ_linear_beta, double tail_eps) {
    // One recipe per call: all block-wise, all MXFP8, or MXFP4 experts over
    // MXFP8 latent/shared weights (w4a8).
    const bool mxfp8 = quant_mode == "w8a8-mxfp8";
    const bool mxfp4 = quant_mode == "w4a8-mxfp4";
    TORCH_CHECK(mxfp8 || mxfp4 || quant_mode == "w8a8-blockwise",
                "moe: quant_mode must be 'w8a8-blockwise', 'w8a8-mxfp8' or "
                "'w4a8-mxfp4', got '",
                quant_mode, "'.");
    // The experts' scale image is a UE8M0 table under either MX recipe; their
    // latent and shared groups are MXFP8 unless the whole call is block-wise.
    const bool experts_mx = mxfp8 || mxfp4;
    const bool mxfp8_rest = mxfp8 || mxfp4;
    using moe::QuantConfig;
    constexpr auto kBW = moe::QuantMethod::kBlockwiseFP8;
    constexpr auto kMX = moe::QuantMethod::kMXFP8;
    constexpr auto kMX4 = moe::QuantMethod::kMXFP4;

    MOE_CHECK_TENSOR_TYPE(input, at::kBFloat16);
    MOE_CHECK_TENSOR_TYPE(router_weight, at::kBFloat16);
    MOE_CHECK_TENSOR_TYPE(expert_up_weights,
                          mxfp4 ? at::kByte : at::kFloat8_e4m3fn);
    MOE_CHECK_TENSOR_TYPE(expert_up_weights_scale,
                          experts_mx ? at::kByte : at::kFloat);
    MOE_CHECK_TENSOR_TYPE(expert_down_weights,
                          mxfp4 ? at::kByte : at::kFloat8_e4m3fn);
    MOE_CHECK_TENSOR_TYPE(expert_down_weights_scale,
                          experts_mx ? at::kByte : at::kFloat);
    MOE_CHECK_TENSOR_TYPE(shared_up_weight, at::kFloat8_e4m3fn);
    MOE_CHECK_TENSOR_TYPE(shared_up_weight_scale,
                          mxfp8_rest ? at::kByte : at::kFloat);
    MOE_CHECK_TENSOR_TYPE(shared_down_weight, at::kFloat8_e4m3fn);
    MOE_CHECK_TENSOR_TYPE(shared_down_weight_scale,
                          mxfp8_rest ? at::kByte : at::kFloat);
    MOE_CHECK_TENSOR_TYPE(shared_out, at::kBFloat16);
    MOE_CHECK_TENSOR_TYPE(latent_down_weight, at::kFloat8_e4m3fn);
    MOE_CHECK_TENSOR_TYPE(latent_down_weight_scale,
                          mxfp8_rest ? at::kByte : at::kFloat);

    constexpr int64_t kNumRouterGates = static_cast<int64_t>(
        moe::RouterGemmTraits<moe::Dims_BS8_E512_N512_K2048_SM100<
            QuantConfig<kBW, kBW, kBW>{}>>::kNumRouterGates);
    TORCH_CHECK(num_router_gates == kNumRouterGates, "moe: this build scores ",
                kNumRouterGates, " router gates, got ", num_router_gates, ".");

    TORCH_CHECK(input.dim() == 2, "moe: input must be [tokens, hidden].");
    const int64_t num_tokens = input.size(0);
    TORCH_CHECK(num_tokens >= 1 && num_tokens <= 16,
                "moe: num_tokens must be in [1, 16].");

    TORCH_CHECK(scoring_func == "sigmoid" || scoring_func == "softmax",
                "moe: scoring_func must be sigmoid or softmax.");
    TORCH_CHECK(activation == "silu" || activation == "situ",
                "moe: activation must be silu or situ.");
    if (activation == "situ") {
        // A non-finite beta turns `beta * tanh(x / beta)` into an `inf * 0`
        // NaN; beta <= 0 (including -inf) means "no cap".
        TORCH_CHECK(std::isfinite(situ_beta) && std::isfinite(situ_linear_beta),
                    "moe: situ_beta / situ_linear_beta must be finite (<= 0 "
                    "means no cap).");
    }

    const int64_t scoring_type = scoring_func == "sigmoid" ? 0 : 1;
    const int64_t activation_type = activation == "silu" ? 0 : 1;

    // One call shape for both token widths: the schedule is picked by M.
    const auto launch_func = [&](auto dims_tag) {
        using MoEDims = decltype(dims_tag);
        moe::MoEImpl<MoEDims>(
            input, router_weight, expert_up_weights, expert_up_weights_scale,
            expert_down_weights, expert_down_weights_scale, shared_up_weight,
            shared_up_weight_scale, shared_down_weight,
            shared_down_weight_scale, latent_down_weight,
            latent_down_weight_scale, workspace, shared_out, tail_fc2_weight,
            tail_fc2_scale, tail_norm_weight, tail_out, tail_epochs,
            tail_uc_bases, top_k, scoring_type, activation_type, tail_rank,
            tail_mc_base, tail_grid_cap, renormalize, situ_beta,
            situ_linear_beta, tail_eps);
    };
    // One instantiation per supported recipe and token width.
    if (num_tokens <= 8) {
        if (mxfp4) {
            launch_func(moe::Dims_BS8_E512_N512_K2048_SM100<
                        QuantConfig<kMX, kMX, kMX4>{}>{});
        } else if (mxfp8) {
            launch_func(moe::Dims_BS8_E512_N512_K2048_SM100<
                        QuantConfig<kMX, kMX, kMX>{}>{});
        } else {
            launch_func(moe::Dims_BS8_E512_N512_K2048_SM100<
                        QuantConfig<kBW, kBW, kBW>{}>{});
        }
    } else {
        if (mxfp4) {
            launch_func(moe::Dims_BS16_E512_N512_K2048_SM100<
                        QuantConfig<kMX, kMX, kMX4>{}>{});
        } else if (mxfp8) {
            launch_func(moe::Dims_BS16_E512_N512_K2048_SM100<
                        QuantConfig<kMX, kMX, kMX>{}>{});
        } else {
            launch_func(moe::Dims_BS16_E512_N512_K2048_SM100<
                        QuantConfig<kBW, kBW, kBW>{}>{});
        }
    }
}

int64_t MoETailRegionBytes(bool wide) {
    using Bs8 = moe::Dims_BS8_E512_N512_K2048_SM100<moe::QuantConfig<>{}>;
    using Bs16 = moe::Dims_BS16_E512_N512_K2048_SM100<moe::QuantConfig<>{}>;
    return wide ? moe::tail_region_bytes<Bs16>()
                : moe::tail_region_bytes<Bs8>();
}

int64_t RouterTileM(int64_t num_tokens = 0) {
    using Bs8 = moe::Dims_BS8_E512_N512_K2048_SM100<moe::QuantConfig<>{}>;
    using Bs16 = moe::Dims_BS16_E512_N512_K2048_SM100<moe::QuantConfig<>{}>;
    return (num_tokens > 8) ? moe::RouterGemmTraits<Bs16>::kTileShapeM
                            : moe::RouterGemmTraits<Bs8>::kTileShapeM;
}

#undef MOE_CHECK_TENSOR
#undef MOE_CHECK_TENSOR_TYPE

}  // namespace halops
