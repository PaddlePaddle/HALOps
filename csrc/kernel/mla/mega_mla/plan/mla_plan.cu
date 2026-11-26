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

#include "kernel/mla/mega_mla/output_proj/plan.cuh"
#include "kernel/mla/mega_mla/plan/kernel.cuh"

namespace halops {
namespace mla {

void mla_plan_check_inputs(const torch::Tensor& kv_indices,
                           const std::optional<torch::Tensor>& topk_lengths,
                           int64_t page_size, int64_t kv_run_step,
                           const std::optional<torch::Tensor>& c128_attn_len) {
    TORCH_CHECK(
        kv_indices.scalar_type() == at::kInt && kv_indices.dim() == 2 &&
            kv_indices.size(0) > 0 && kv_indices.size(1) > 0 &&
            kv_indices.stride(1) == 1,
        "mla_attention_plan: kv_indices must be int32 [T > 0, topk > 0] "
        "with unit column stride");
    const int64_t num_tokens = kv_indices.size(0);
    TORCH_CHECK((!topk_lengths || (topk_lengths->numel() == num_tokens &&
                                   topk_lengths->is_contiguous())) &&
                    (!c128_attn_len || c128_attn_len->numel() == num_tokens),
                "mla_attention_plan: T = ", num_tokens,
                " must match topk_lengths / c128_attn_len");
    TORCH_CHECK(page_size > 0 && (page_size & (page_size - 1)) == 0,
                "mla_attention_plan: page_size must be a power of two");
    TORCH_CHECK(kv_run_step == 1 || kv_run_step == -1,
                "mla_attention_plan: kv_run_step must be 1 or -1");
}

void mla_plan_launch_kv_runs(const torch::Tensor& kv_indices,
                             const std::optional<torch::Tensor>& topk_lengths,
                             int64_t page_size, int64_t kv_run_step,
                             int4* kv_runs) {
    const int topk = (int)kv_indices.size(1);
    const MlaPlanKvRunsArgs args{.kv_indices = kv_indices.data_ptr<int>(),
                                 .topk_length = optional_ptr<int>(topk_lengths),
                                 .row_stride = kv_indices.stride(0),
                                 .topk = topk,
                                 .step = (int)kv_run_step,
                                 .page_shift = page_shift_of(page_size),
                                 .kv_runs = kv_runs};
    const int threads = std::min(cute::round_up(topk, KernelBase::WarpSize),
                                 MlaPlanCfg::kThreads);
    launch_pdl(mla_plan_kv_runs_kernel<MlaPlanCfg>,
               {(int)kv_indices.size(0), threads}, args);
}

OutputProjShape mla_plan_check_workspace_weights(
    const torch::Tensor& wq_a_weight, const torch::Tensor& wkv_weight,
    const torch::Tensor& wo_a_weight, const torch::Tensor& wo_b_weight) {
    const int hidden =
        check_attention_weights(wq_a_weight, wkv_weight, "mla_workspace");
    const OutputProjShape shape =
        output_proj_shape(wo_a_weight, wo_b_weight, "mla_workspace");
    TORCH_CHECK(shape.hidden == hidden, "mla_workspace: wo_b has ",
                shape.hidden, " rows, wq_a has H = ", hidden);
    return shape;
}

}  // namespace mla

using namespace mla;

torch::Tensor MlaAttentionPlan(
    const torch::Tensor& kv_indices,
    const std::optional<torch::Tensor>& topk_lengths, int64_t page_size,
    int64_t kv_run_step, const std::optional<torch::Tensor>& c128_attn_len,
    const torch::Tensor& workspace) {
    const CudaDeviceGuard device_guard(workspace.device());
    mla_plan_check_inputs(kv_indices, topk_lengths, page_size, kv_run_step,
                          c128_attn_len);
    const int token_count = (int)kv_indices.size(0);
    const int grid_x = mla::grid_x();
    const int kv_splits = kv_splits_for(grid_x, token_count);
    const bool hca = c128_attn_len.has_value();
    const int n_groups = n_groups_for(token_count, !hca);

    const MlaWorkspace::Layout wsl = MlaWorkspace::layout(token_count, grid_x);
    TORCH_CHECK(workspace.numel() >= wsl.kv_split_off,
                "mla_attention_plan: workspace needs ", wsl.kv_split_off,
                " bytes, got ", workspace.numel());
    char* base = reinterpret_cast<char*>(workspace.data_ptr());
    const PlanView view =
        plan_ptrs(base, wsl.plan_off[hca], token_count, grid_x);
    mla_plan_launch_kv_runs(kv_indices, topk_lengths, page_size, kv_run_step,
                            const_cast<int4*>(view.kv_runs));

    const int items_kvg = kv_gate_items(n_groups, hca);
    using Split = AttnConfig::KvSplit;
    const MlaPlanArgs args{
        .kv_runs = view.kv_runs,
        .c128_attn_len = optional_ptr<int>(c128_attn_len),
        .num_tokens = token_count,
        .grid_x = grid_x,
        .kv_splits = kv_splits,
        .split_min_nb = Split::MinBlocksToCut,
        .split_gain = Split::Gain,
        .tail_cap = Split::TailCap,
        .kvg_ctas = kv_gate_ctas(token_count, grid_x, items_kvg),
        .hdr = const_cast<PlanHeader*>(view.hdr),
        .work = const_cast<CtaWork*>(view.work),
        .tok = const_cast<TokenSplit*>(view.tok),
        .partial_bar = reinterpret_cast<int*>(base + wsl.pbar_off),
        .pbar_words = (int)((wsl.barrier_end - wsl.pbar_off) / sizeof(int))};
    launch_pdl(mla_plan_attention_kernel<MlaPlanCfg>, {1, MlaPlanCfg::kThreads},
               args);

    return workspace
        .narrow(0, wsl.plan_off[hca] + wsl.plan.kv_runs_off,
                (int64_t)token_count * sizeof(int4))
        .view(at::kInt)
        .view({token_count, MlaPlanCfg::kRunFields});
}

std::vector<int64_t> MlaWorkspaceBytes(int64_t max_tokens,
                                       const torch::Tensor& wq_a_weight,
                                       const torch::Tensor& wkv_weight,
                                       const torch::Tensor& wo_a_weight,
                                       const torch::Tensor& wo_b_weight) {
    const CudaDeviceGuard device_guard(wq_a_weight.device());
    const OutputProjShape shape = mla_plan_check_workspace_weights(
        wq_a_weight, wkv_weight, wo_a_weight, wo_b_weight);
    const OutputProjPlan plan((int)max_tokens, shape, "mla_workspace");
    const MlaWorkspace::Layout wsl =
        MlaWorkspace::layout(plan.num_tokens, grid_x());
    return {wsl.output_proj_off(shape.hidden) + plan.bytes, wsl.barrier_end};
}
}  // namespace halops
