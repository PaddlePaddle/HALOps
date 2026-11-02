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

#include <cmath>
#include <cstdint>
#include <optional>
#include <string>

#include "common/cuda_utils.h"
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
    constexpr std::int64_t kBlockM = MoEDims::kQuantBlockM;
    constexpr std::int64_t kBlockK = MoEDims::kQuantBlockK;
    TORCH_CHECK(latent_down_weight.dim() == 2 &&
                    latent_down_weight.size(0) == MoEDims::kLatentSize &&
                    latent_down_weight.size(1) == MoEDims::kHiddenSize,
                "moe: latent_down_weight must be [kLatentSize, kHiddenSize].");
    TORCH_CHECK(
        latent_down_weight_scale.dim() == 2 &&
            latent_down_weight_scale.size(0) ==
                MoEDims::kLatentSize / kBlockM &&
            latent_down_weight_scale.size(1) == MoEDims::kHiddenSize / kBlockK,
        "moe: latent_down_weight_scale must be [L/", kBlockM, ", H/", kBlockK,
        "].");
}

template <typename MoEDims>
void CheckRoutedExpertShape(const torch::Tensor& expert_up_weights,
                            const torch::Tensor& expert_up_weights_scale,
                            const torch::Tensor& expert_down_weights,
                            const torch::Tensor& expert_down_weights_scale) {
    constexpr std::int64_t kExperts = MoEDims::kNumExperts;
    constexpr std::int64_t kIntermediate = MoEDims::kIntermediateSize;
    constexpr std::int64_t kExpertK = moe::MoEInputSize<MoEDims>::value;
    constexpr std::int64_t kBlockM = MoEDims::kQuantBlockM;
    constexpr std::int64_t kBlockK = MoEDims::kQuantBlockK;
    constexpr std::int64_t kUpRows = 2 * kIntermediate;
    TORCH_CHECK(expert_up_weights.dim() == 3 &&
                    expert_up_weights.size(0) == kExperts &&
                    expert_up_weights.size(1) == kUpRows &&
                    expert_up_weights.size(2) == kExpertK,
                "moe: expert_up_weights must be [E, 2N, K].");
    TORCH_CHECK(expert_up_weights_scale.dim() == 3 &&
                    expert_up_weights_scale.size(0) == kExperts &&
                    expert_up_weights_scale.size(1) == kUpRows / kBlockM &&
                    expert_up_weights_scale.size(2) == kExpertK / kBlockK,
                "moe: expert_up_weights_scale must be [E, 2N/", kBlockM, ", K/",
                kBlockK, "].");
    TORCH_CHECK(expert_down_weights.dim() == 3 &&
                    expert_down_weights.size(0) == kExperts &&
                    expert_down_weights.size(1) == kExpertK &&
                    expert_down_weights.size(2) == kIntermediate,
                "moe: expert_down_weights must be [E, K, N].");
    TORCH_CHECK(
        expert_down_weights_scale.dim() == 3 &&
            expert_down_weights_scale.size(0) == kExperts &&
            expert_down_weights_scale.size(1) == kExpertK / kBlockM &&
            expert_down_weights_scale.size(2) == kIntermediate / kBlockK,
        "moe: expert_down_weights_scale must be [E, K/", kBlockM, ", N/",
        kBlockK, "].");
}

template <typename MoEDims>
void CheckSharedExpertShape(const torch::Tensor& input,
                            const torch::Tensor& shared_up_weight,
                            const torch::Tensor& shared_up_weight_scale,
                            const torch::Tensor& shared_down_weight,
                            const torch::Tensor& shared_down_weight_scale,
                            const torch::Tensor& shared_out) {
    static_assert(moe::MoEKernelTraits<MoEDims>::kUseSharedExpert,
                  "this interface takes the shared expert's operands, so the "
                  "schedule it builds must run the shared expert.");
    constexpr std::int64_t kUpRows = 2 * MoEDims::kIntermediateSize;
    constexpr std::int64_t kBlockM = MoEDims::kQuantBlockM;
    constexpr std::int64_t kBlockK = MoEDims::kQuantBlockK;
    TORCH_CHECK(shared_up_weight.dim() == 2 &&
                    shared_up_weight.size(0) == kUpRows &&
                    shared_up_weight.size(1) == MoEDims::kHiddenSize,
                "moe: shared_up_weight must be the pair-layout [2N, H].");
    TORCH_CHECK(
        shared_up_weight_scale.dim() == 2 &&
            shared_up_weight_scale.size(0) == kUpRows / kBlockM &&
            shared_up_weight_scale.size(1) == MoEDims::kHiddenSize / kBlockK,
        "moe: shared_up_scale must be [2N/", kBlockM, ", H/", kBlockK, "].");
    TORCH_CHECK(shared_down_weight.dim() == 2 &&
                    shared_down_weight.size(0) == MoEDims::kHiddenSize &&
                    shared_down_weight.size(1) == MoEDims::kIntermediateSize,
                "moe: shared_down_weight must be [H, N].");
    TORCH_CHECK(shared_down_weight_scale.dim() == 2 &&
                    shared_down_weight_scale.size(0) ==
                        MoEDims::kHiddenSize / kBlockM &&
                    shared_down_weight_scale.size(1) ==
                        MoEDims::kIntermediateSize / kBlockK,
                "moe: shared_down_scale must be [H/", kBlockM, ", N/", kBlockK,
                "].");
    TORCH_CHECK(shared_out.dim() == 2 && shared_out.size(0) == input.size(0) &&
                    shared_out.size(1) == MoEDims::kHiddenSize,
                "moe: shared_out must be [tokens, H].");
}

// One block per SM, with no slack: every block must stay scheduled for the
// kernel's full lifetime, so a single unavailable SM deadlocks.
template <typename MoEDims>
uint32_t CalculateGridSize(int device, cudaStream_t stream) {
    using Traits = moe::MoEKernelTraits<MoEDims>;
    using Mainloop = moe::CollectiveMainloopMoE<Traits>;

    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
    TORCH_CHECK(sm_count >= static_cast<int>(Traits::kMinGrid),
                "moe: needs at least kMinGrid (=", Traits::kMinGrid,
                ") SMs for the co-residency invariant; got ", sm_count);
    if constexpr (Traits::kUseSharedExpert) {
        // It runs on the blocks the router and the projection leave idle.
        constexpr std::uint32_t kSharedCtas =
            Traits::RouterGemm::kNumCtas + Traits::LatentDownGemm::kNumCtas +
            moe::SharedDownGemmTraits<MoEDims>::kNumCtas;
        TORCH_CHECK(static_cast<std::uint32_t>(sm_count) >= kSharedCtas,
                    "moe: the shared expert needs the grid to reach its ",
                    moe::SharedDownGemmTraits<MoEDims>::kNumCtas,
                    " blocks, i.e. at least ", kSharedCtas, " SMs; got ",
                    sm_count);
    }
    uint32_t grid_size =
        std::min(static_cast<std::uint32_t>(sm_count), Traits::kMaxGrid);

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

// One 2-D SW128 TMA descriptor: `inner` is the contiguous extent, `outer`
// the row axis, and `row_bytes` the global stride between rows.
static CUtensorMap CreateTmaDesc2D(const void* ptr, CUtensorMapDataType type,
                                   const char* what, uint64_t inner,
                                   uint64_t outer, uint64_t row_bytes,
                                   uint32_t box_inner, uint32_t box_outer) {
    CUtensorMap desc{};
    uint64_t global_dim[2] = {inner, outer};
    uint64_t global_strides[1] = {row_bytes};
    uint32_t box_dim[2] = {box_inner, box_outer};
    uint32_t element_strides[2] = {1u, 1u};
    const CUresult res = cuTensorMapEncodeTiled(
        &desc, type, 2, const_cast<void*>(ptr), global_dim, global_strides,
        box_dim, element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(res == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed for ", what,
                ": CUresult=", static_cast<int>(res), " (inner=", inner,
                ", outer=", outer, ", box=(", box_inner, ", ", box_outer, "))");
    return desc;
}

CUtensorMap CreateUpWeightTmaDesc(const void* weights_ptr, uint32_t num_experts,
                                  uint32_t n, uint32_t k, uint32_t row_box) {
    return CreateTmaDesc2D(
        weights_ptr, CU_TENSOR_MAP_DATA_TYPE_UINT8, "up-projection weights",
        /*inner=*/k,
        /*outer=*/static_cast<uint64_t>(num_experts) * 2ULL * n,
        /*row_bytes=*/k, /*box_inner=*/128u, /*box_outer=*/row_box);
}

CUtensorMap CreateActivationsTmaDesc(const void* activations_ptr,
                                     uint32_t batch_size_cap,
                                     uint32_t block_rows, uint32_t K_hidden) {
    return CreateTmaDesc2D(activations_ptr, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                           "activations",
                           /*inner=*/K_hidden, /*outer=*/batch_size_cap,
                           /*row_bytes=*/static_cast<uint64_t>(K_hidden) * 2ULL,
                           /*box_inner=*/64u, /*box_outer=*/block_rows);
}

CUtensorMap CreateDownWeightTmaDesc(const void* weights_ptr,
                                    uint32_t num_experts, uint32_t k,
                                    uint32_t n, uint32_t row_box) {
    return CreateTmaDesc2D(
        weights_ptr, CU_TENSOR_MAP_DATA_TYPE_UINT8, "down-projection weights",
        /*inner=*/n, /*outer=*/static_cast<uint64_t>(num_experts) * k,
        /*row_bytes=*/n, /*box_inner=*/128u, /*box_outer=*/row_box);
}

CUtensorMap CreateLatentDownWeightTmaDesc(const void* weights_ptr, uint32_t m,
                                          uint32_t k, uint32_t row_box) {
    return CreateTmaDesc2D(
        weights_ptr, CU_TENSOR_MAP_DATA_TYPE_UINT8, "latent-projection weights",
        /*inner=*/k, /*outer=*/m,
        /*row_bytes=*/k, /*box_inner=*/128u, /*box_outer=*/row_box);
}

CUtensorMap CreateDownActivationTmaDesc(const void* activations_ptr,
                                        uint32_t temp_rows, uint32_t block_rows,
                                        uint32_t n) {
    return CreateTmaDesc2D(
        activations_ptr, CU_TENSOR_MAP_DATA_TYPE_UINT8,
        "down-projection activations", /*inner=*/n, /*outer=*/temp_rows,
        /*row_bytes=*/n, /*box_inner=*/128u, /*box_outer=*/block_rows);
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
             const torch::Tensor& workspace, const torch::Tensor& routed_out,
             const torch::Tensor& shared_out, int64_t top_k,
             int64_t scoring_func, bool renormalize, int64_t activation,
             double situ_beta, double situ_linear_beta) {
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
    args.top_k = static_cast<uint32_t>(top_k);
    args.scoring_func = static_cast<ScoringFunc>(scoring_func);
    args.renormalize = renormalize;
    args.activation = static_cast<Activation>(activation);
    args.situ_beta = static_cast<float>(situ_beta);
    args.situ_linear_beta = static_cast<float>(situ_linear_beta);

    args.input = static_cast<const DataType::ElementA*>(input.const_data_ptr());
    args.router_weight =
        static_cast<const DataType::ElementA*>(router_weight.const_data_ptr());

    args.expert_up_weights = static_cast<const DataType::ElementW*>(
        expert_up_weights.const_data_ptr());
    args.expert_up_weights_scale = static_cast<const DataType::ElementScale*>(
        expert_up_weights_scale.const_data_ptr());
    args.expert_down_weights = static_cast<const DataType::ElementW*>(
        expert_down_weights.const_data_ptr());
    args.expert_down_weights_scale = static_cast<const DataType::ElementScale*>(
        expert_down_weights_scale.const_data_ptr());

    args.shared_up_weight = static_cast<const moe::DataType::ElementW*>(
        shared_up_weight.const_data_ptr());
    args.shared_up_weight_scale =
        static_cast<const moe::DataType::ElementScale*>(
            shared_up_weight_scale.const_data_ptr());
    args.shared_down_weight = static_cast<const DataType::ElementW*>(
        shared_down_weight.const_data_ptr());
    args.shared_down_weight_scale = static_cast<const DataType::ElementScale*>(
        shared_down_weight_scale.const_data_ptr());

    args.latent_down_weight = MoEDims::kUseLatent
                                  ? static_cast<const DataType::ElementW*>(
                                        latent_down_weight.const_data_ptr())
                                  : nullptr;
    args.latent_down_weight_scale =
        MoEDims::kUseLatent ? static_cast<const DataType::ElementScale*>(
                                  latent_down_weight_scale.const_data_ptr())
                            : nullptr;

    args.routed_out = static_cast<DataType::ElementOut*>(routed_out.data_ptr());
    args.workspace = static_cast<char*>(workspace.data_ptr());
    args.shared_out = static_cast<DataType::ElementOut*>(shared_out.data_ptr());

    const int device = input.get_device();
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(device).stream();
    const uint32_t grid_size = CalculateGridSize<MoEDims>(device, stream);

    const cudaError_t launch_status =
        moe::LaunchMoEKernel<MoEDims>(args, grid_size, stream);
    TORCH_CHECK(launch_status == cudaSuccess, "moe: kernel launch failed: ",
                cudaGetErrorString(launch_status));
}

}  // namespace moe

int64_t GetWorkspaceSize(const std::optional<torch::Tensor>& input) {
    constexpr int64_t kBs8Bytes = static_cast<int64_t>(
        sizeof(moe::MoEWorkspace<moe::Dims_BS8_E512_N512_K2048_SM100<true>>));
    constexpr int64_t kBs16Bytes = static_cast<int64_t>(
        sizeof(moe::MoEWorkspace<moe::Dims_BS16_E512_N512_K2048_SM100<true>>));
    if (!input.has_value()) {
        return kBs8Bytes > kBs16Bytes ? kBs8Bytes : kBs16Bytes;
    }
    constexpr int64_t kBs8Tokens = static_cast<int64_t>(
        moe::Dims_BS8_E512_N512_K2048_SM100<true>::kNumTokens);
    return input->size(0) <= kBs8Tokens ? kBs8Bytes : kBs16Bytes;
}

torch::Tensor GetWorkspace(const torch::Tensor& input) {
    TORCH_CHECK(input.dim() == 2, "moe: input must be [tokens, hidden].");
    TORCH_CHECK(input.is_cuda(),
                "moe: the workspace lives on the input's device, so input "
                "must be a CUDA tensor.");
    return torch::zeros(
        {GetWorkspaceSize(input)},
        torch::TensorOptions().dtype(torch::kUInt8).device(input.device()));
}

void LowLatencyMoE(const torch::Tensor& input,
                   const torch::Tensor& router_weight,
                   const torch::Tensor& expert_up_weights,
                   const torch::Tensor& expert_up_weights_scale,
                   const torch::Tensor& expert_down_weights,
                   const torch::Tensor& expert_down_weights_scale,
                   const torch::Tensor& shared_up_weight,
                   const torch::Tensor& shared_up_weight_scale,
                   const torch::Tensor& shared_down_weight,
                   const torch::Tensor& shared_down_weight_scale,
                   const std::optional<torch::Tensor>& latent_down_weight,
                   const std::optional<torch::Tensor>& latent_down_weight_scale,
                   const torch::Tensor& workspace,
                   const torch::Tensor& routed_out,
                   const torch::Tensor& shared_out, int64_t top_k,
                   int64_t num_router_gates, const std::string& scoring_func,
                   bool renormalize, const std::string& activation,
                   double situ_beta, double situ_linear_beta) {
    MOE_CHECK_TENSOR_TYPE(input, at::kBFloat16);
    MOE_CHECK_TENSOR_TYPE(router_weight, at::kBFloat16);
    MOE_CHECK_TENSOR_TYPE(expert_up_weights, at::kFloat8_e4m3fn);
    MOE_CHECK_TENSOR_TYPE(expert_up_weights_scale, at::kFloat);
    MOE_CHECK_TENSOR_TYPE(expert_down_weights, at::kFloat8_e4m3fn);
    MOE_CHECK_TENSOR_TYPE(expert_down_weights_scale, at::kFloat);
    MOE_CHECK_TENSOR_TYPE(shared_up_weight, at::kFloat8_e4m3fn);
    MOE_CHECK_TENSOR_TYPE(shared_up_weight_scale, at::kFloat);
    MOE_CHECK_TENSOR_TYPE(shared_down_weight, at::kFloat8_e4m3fn);
    MOE_CHECK_TENSOR_TYPE(shared_down_weight_scale, at::kFloat);
    MOE_CHECK_TENSOR_TYPE(routed_out, at::kBFloat16);
    MOE_CHECK_TENSOR_TYPE(shared_out, at::kBFloat16);

    constexpr int64_t kNumRouterGates = static_cast<int64_t>(
        moe::RouterGemmTraits<
            moe::Dims_BS8_E512_N512_K2048_SM100<true>>::kNumRouterGates);
    TORCH_CHECK(num_router_gates == kNumRouterGates, "moe: this build scores ",
                kNumRouterGates, " router gates, got ", num_router_gates, ".");

    bool use_latent = false;
    if (latent_down_weight.has_value() &&
        latent_down_weight_scale.has_value()) {
        use_latent = true;
        MOE_CHECK_TENSOR_TYPE(*latent_down_weight, at::kFloat8_e4m3fn);
        MOE_CHECK_TENSOR_TYPE(*latent_down_weight_scale, at::kFloat);
    }

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

    const torch::Tensor absent;
    const torch::Tensor& latent_weight =
        latent_down_weight.has_value() ? *latent_down_weight : absent;
    const torch::Tensor& latent_scale = latent_down_weight_scale.has_value()
                                            ? *latent_down_weight_scale
                                            : absent;

    // One call shape for both token widths: the schedule is picked by M.
    const auto launch_func = [&](auto dims_tag) {
        using MoEDims = decltype(dims_tag);
        moe::MoEImpl<MoEDims>(
            input, router_weight, expert_up_weights, expert_up_weights_scale,
            expert_down_weights, expert_down_weights_scale, shared_up_weight,
            shared_up_weight_scale, shared_down_weight,
            shared_down_weight_scale, latent_weight, latent_scale, workspace,
            routed_out, shared_out, top_k, scoring_type, renormalize,
            activation_type, situ_beta, situ_linear_beta);
    };
    // The non-latent schedule does not fit the block's SMEM carve yet.
    TORCH_CHECK(use_latent,
                "moe: this build runs the latent stage; pass "
                "latent_down_weight and latent_down_weight_scale.");
    if (num_tokens <= 8) {
        launch_func(moe::Dims_BS8_E512_N512_K2048_SM100<true>{});
    } else {
        launch_func(moe::Dims_BS16_E512_N512_K2048_SM100<true>{});
    }
}

int64_t RouterTileM(int64_t num_tokens = 0) {
    return (num_tokens > 8)
               ? moe::RouterGemmTraits<
                     moe::Dims_BS16_E512_N512_K2048_SM100<true>>::kTileShapeM
               : moe::RouterGemmTraits<
                     moe::Dims_BS8_E512_N512_K2048_SM100<true>>::kTileShapeM;
}

#undef MOE_CHECK_TENSOR
#undef MOE_CHECK_TENSOR_TYPE

}  // namespace halops
