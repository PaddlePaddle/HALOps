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

#include <c10/cuda/CUDAGuard.h>

#include "kernel/mla/combine_with_rope_quant.cuh"
#include "kernel/mla/mla_output_proj_plan.cuh"

namespace halops {
using namespace mla;

void check_combine_with_rope_quant_inputs(
    const torch::Tensor& q_or_x, const torch::Tensor& freqs_real,
    const std::optional<torch::Tensor>& attn_sink,
    const torch::Tensor& workspace, int64_t workspace_bytes) {
    TORCH_CHECK(q_or_x.scalar_type() == at::kBFloat16 &&
                    q_or_x.is_contiguous() &&
                    q_or_x.size(1) == ModelDims::OTotal,
                "combine_with_rope_quant: q_or_x must be contiguous bf16 [T, ",
                ModelDims::OTotal, "]");
    check_rope_and_sink(freqs_real, attn_sink, "combine_with_rope_quant");
    TORCH_CHECK(workspace.numel() >= workspace_bytes,
                "combine_with_rope_quant: workspace needs ", workspace_bytes,
                " bytes, got ", workspace.numel());
}

void CombineWithRopeQuant(const torch::Tensor& q_or_x,
                          const torch::Tensor& positions,
                          const torch::Tensor& freqs_real,
                          const std::optional<torch::Tensor>& attn_sink,
                          const std::optional<torch::Tensor>& c128_attn_len,
                          const torch::Tensor& wo_a_weight,
                          const torch::Tensor& wo_b_weight,
                          const torch::Tensor& workspace) {
    const at::cuda::OptionalCUDAGuard device_guard(q_or_x.device());
    const int num_tokens = (int)q_or_x.size(0);
    const OutputProjShape shape =
        output_proj_shape(wo_a_weight, wo_b_weight, "combine_with_rope_quant");
    const OutputProjPlan plan(num_tokens, shape, "combine_with_rope_quant");
    const MlaWorkspace::Layout wsl = MlaWorkspace::layout(num_tokens, grid_x());
    const int64_t proj_off = wsl.output_proj_off(shape.hidden);
    check_combine_with_rope_quant_inputs(q_or_x, freqs_real, attn_sink,
                                         workspace, proj_off + plan.bytes);
    uint8_t* base = workspace.data_ptr<uint8_t>();
    uint8_t* proj = base + proj_off;
    uint8_t* flat_o = proj + plan.off_flat_o;
    uint8_t* flat_o_sf = proj + plan.off_flat_o_sf;
    const PlanView view = plan_ptrs(reinterpret_cast<char*>(base),
                                    wsl.plan_off[c128_attn_len.has_value()],
                                    num_tokens, grid_x());
    const bool slice = 2 * num_tokens <= grid_x();
    const int slices = num_tokens * ModelDims::OTotal / CombineCfg::SliceDims;
    const LaunchDims dims =
        slice ? LaunchDims{(slices + CombineCfg::SliceThreads - 1) /
                               CombineCfg::SliceThreads,
                           CombineCfg::SliceThreads}
              : LaunchDims{num_tokens * CombineCfg::RowsPerToken,
                           CombineCfg::RowThreads};
    dispatch_output_proj_tile(
        plan, "combine_with_rope_quant", [&](auto pair, auto tt, auto nh) {
            const auto layout_sf =
                output_proj_flat_o_sf_layout<decltype(pair)::value,
                                             decltype(tt)::value,
                                             decltype(nh)::value>(
                    shape, plan, flat_o, flat_o_sf);
            using LayoutSF = std::decay_t<decltype(layout_sf)>;
            const CombineParams<LayoutSF> params{
                .q_or_x =
                    reinterpret_cast<const __nv_bfloat16*>(q_or_x.data_ptr()),
                .o_partial = reinterpret_cast<const __nv_bfloat16*>(
                    base + wsl.kv_split_off),
                .lse_partial =
                    reinterpret_cast<const float*>(base + wsl.lse_off),
                .tok = view.tok,
                .positions = positions.data_ptr<int>(),
                .freqs_real = freqs_real.data_ptr<float>(),
                .attn_sink = optional_ptr<float>(attn_sink),
                .num_tokens = num_tokens,
                .pbar = reinterpret_cast<int*>(base + wsl.pbar_off),
                .n_pbar = MlaWorkspace::NumBarrierBanks * wsl.pbar_bank,
                .pbar_stride =
                    WorkspaceLayout::RendezvousSlotBytes / sizeof(int),
                .c128_flags = reinterpret_cast<int*>(base + wsl.c128_flags_off),
                .n_c128_flags = C128Layout::FlagsPerToken * num_tokens,
                .out_fp8 = flat_o,
                .out_sf = flat_o_sf,
                .layout_sf = layout_sf,
                .sf_group_tokens = plan.group_tokens / plan.n_parts,
                .sf_groups = plan.groups * plan.n_parts,
                .flags = reinterpret_cast<int*>(proj + plan.off_flags),
                .n_flags = shape.n_flags()};
            TORCH_CHECK(
                (int64_t)dims.grid * dims.block >=
                    std::max(
                        {params.n_pbar, params.n_c128_flags, params.n_flags}),
                "combine_with_rope_quant: grid too small to reset barriers");
            launch_pdl(slice ? combine_with_rope_quant_slice_kernel<LayoutSF>
                             : combine_with_rope_quant_row_kernel<LayoutSF>,
                       dims, params);
        });
}

}  // namespace halops
