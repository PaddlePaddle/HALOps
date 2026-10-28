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

#include "kernel/mla/mla_output_proj_plan.cuh"
#include "kernel/mla/mla_params.cuh"

namespace halops {
using namespace mla;

namespace {

template <class Gemm>
struct OutputProjParams {
    Gemm wo_a;
    Gemm wo_b;
    cutlass::float_e4m3_t* olora_fp8;
    uint8_t* olora_sf;
    cutlass::bfloat16_t* out;
    int* flags;
    int num_tokens;
    const uint8_t* wo_b_raw;
    OutputProjShape shape;
};

void check_output_proj_inputs(
    const torch::Tensor& wo_a_weight, const torch::Tensor& wo_a_sf,
    const torch::Tensor& wo_b_weight, const torch::Tensor& wo_b_sf,
    const torch::Tensor& out, const torch::Tensor& workspace,
    int64_t workspace_bytes, const OutputProjShape& shape) {
    check_sfatom_weight(wo_a_weight, wo_a_sf, "mla_output_proj");
    check_sfatom_weight(wo_b_weight, wo_b_sf, "mla_output_proj");
    TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.is_contiguous() &&
                    out.dim() == 2 && out.size(1) == shape.hidden,
                "mla_output_proj: out must be contiguous bf16 [T, ",
                shape.hidden, "]");
    TORCH_CHECK(workspace.numel() >= workspace_bytes,
                "mla_output_proj: workspace needs ", workspace_bytes,
                " bytes, got ", workspace.numel());
}

}  // namespace

void MlaOutputProj(const torch::Tensor& wo_a_weight,
                   const torch::Tensor& wo_a_sf,
                   const torch::Tensor& wo_b_weight,
                   const torch::Tensor& wo_b_sf, torch::Tensor& out,
                   const torch::Tensor& workspace) {
    const CudaDeviceGuard device_guard(out.device());
    const OutputProjShape shape =
        output_proj_shape(wo_a_weight, wo_b_weight, "mla_output_proj");
    const OutputProjPlan plan((int)out.size(0), shape, "mla_output_proj");
    const int64_t proj_off = MlaWorkspace::layout(plan.num_tokens, grid_x())
                                 .output_proj_off(shape.hidden);
    check_output_proj_inputs(wo_a_weight, wo_a_sf, wo_b_weight, wo_b_sf, out,
                             workspace, proj_off + plan.bytes, shape);
    uint8_t* proj = workspace.data_ptr<uint8_t>() + proj_off;
    uint8_t* flat_o = proj + plan.off_flat_o;
    uint8_t* flat_o_sf = proj + plan.off_flat_o_sf;
    uint8_t* olora = proj + plan.off_olora;
    uint8_t* olora_sf = proj + plan.off_olora_sf;
    auto make_params = [&](auto wo_a, auto wo_b) {
        return OutputProjParams<decltype(wo_a)>{
            .wo_a = wo_a,
            .wo_b = wo_b,
            .olora_fp8 = reinterpret_cast<cutlass::float_e4m3_t*>(olora),
            .olora_sf = olora_sf,
            .out = reinterpret_cast<cutlass::bfloat16_t*>(out.data_ptr()),
            .flags = reinterpret_cast<int*>(proj + plan.off_flags),
            .num_tokens = plan.num_tokens,
            .wo_b_raw =
                reinterpret_cast<const uint8_t*>(wo_b_weight.data_ptr()),
            .shape = shape};
    };
    dispatch_output_proj_tile(
        plan, "mla_output_proj", [&](auto pair, auto tt, auto nh) {
            constexpr int TT = decltype(tt)::value, NH = decltype(nh)::value;
            if constexpr (decltype(pair)::value) {
                using Sh = pair_umma::Shape<TT, NH>;
                auto mainloop = [&](const void* weight, const void* weight_sf,
                                    int rows, const void* act,
                                    const void* act_sf, int k_weight,
                                    int k_act) {
                    auto layout_a = cute::make_layout(
                        cute::make_shape(rows, k_weight, 1),
                        cute::make_stride(k_weight, cute::Int<1>{}, 0));
                    auto layout_b = cute::make_layout(
                        cute::make_shape(plan.t_pad, k_act, 1),
                        cute::make_stride(k_act, cute::Int<1>{}, 0));
                    auto layout_sfa =
                        Sh::Tr::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
                            cute::make_shape(rows, plan.t_pad, k_weight, 1));
                    return Sh::ML::to_underlying_arguments(
                        {static_cast<const cutlass::float_e4m3_t*>(weight),
                         static_cast<const cutlass::float_e4m3_t*>(act),
                         static_cast<const cutlass::float_ue8m0_t*>(weight_sf),
                         static_cast<const cutlass::float_ue8m0_t*>(act_sf),
                         layout_a, layout_b, layout_sfa,
                         pair_umma::act_sf_layout<TT, NH>(rows, plan.t_pad_sf,
                                                          k_act),
                         k_weight, rows, 1});
                };
                auto prm = make_params(
                    mainloop(wo_a_weight.data_ptr(), wo_a_sf.data_ptr(),
                             shape.olora, flat_o, flat_o_sf, shape.group_k,
                             ModelDims::OTotal),
                    mainloop(wo_b_weight.data_ptr(), wo_b_sf.data_ptr(),
                             shape.hidden, olora, olora_sf, shape.olora,
                             shape.olora));
                launch_pdl(
                    pair_umma::mla_output_proj_pair_kernel<TT, NH,
                                                           decltype(prm)>,
                    {shape.grid(), KernelBase::NumThreads,
                     pair_umma::kSmemLaunchBytes<TT, NH>,
                     OutputProjLayout::ClusterSize},
                    prm);
            } else {
                using Cfg = typename swapab::Shape<TT>::Cfg;
                auto prm = make_params(
                    swapab::Operands<Cfg>{
                        .weight = make_weight_atoms<Cfg>(
                            wo_a_weight.data_ptr(), wo_a_sf.data_ptr(),
                            shape.olora, shape.group_k),
                        .act = make_act_atoms<Cfg>(
                            flat_o, flat_o_sf, shape.olora, plan.t_pad,
                            ModelDims::OTotal, plan.t_pad_sf)},
                    swapab::Operands<Cfg>{
                        .weight = make_weight_atoms<Cfg>(
                            wo_b_weight.data_ptr(), wo_b_sf.data_ptr(),
                            shape.hidden, shape.olora),
                        .act = make_act_atoms<Cfg>(
                            olora, olora_sf, shape.hidden, plan.t_pad,
                            shape.olora, plan.t_pad_sf)});
                launch_pdl(
                    swapab::mla_output_proj_swapab_kernel<TT, decltype(prm)>,
                    {shape.grid(), KernelBase::NumThreads,
                     swapab::Smem<TT>::Total, OutputProjLayout::ClusterSize},
                    prm);
            }
        });
}

}  // namespace halops
