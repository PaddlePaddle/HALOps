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

#include "kernel/mla/mla_params.cuh"

namespace halops::mla {
namespace {

constexpr int kPrequantThreads = 256;

template <int TileTokens, class LayoutSF>
__global__ void __launch_bounds__(kPrequantThreads)
    mla_prequant_x_kernel(const __nv_bfloat16* __restrict__ x_input,
                          int x_stride,
                          cutlass::float_e4m3_t* __restrict__ ws_fp8,
                          cutlass::float_ue8m0_t* __restrict__ ws_sf,
                          LayoutSF layout_sf, int num_tokens, int hidden) {
    pdl_trigger();
    pdl_wait();
    const int per_row = hidden / QuantX::PerThread;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_tokens * per_row) return;
    const int token = idx / per_row, chunk = idx - token * per_row;
    __nv_bfloat16 vals[QuantX::PerThread];
    *reinterpret_cast<uint4*>(vals) = __ldg(reinterpret_cast<const uint4*>(
        x_input + (int64_t)token * x_stride + chunk * QuantX::PerThread));
    quantize_group32_from_regs8<TileTokens>(
        vals, ws_fp8, ws_sf, layout_sf, token, hidden,
        chunk / QuantX::ThreadsPerGrp, KernelBase::lane_idx());
}

template <class KT>
void check_prequant_x_inputs(const MlaAttentionProblem<KT>& prob) {
    constexpr int kVecBytes = sizeof(uint4);
    static_assert(QuantX::PerThread * sizeof(__nv_bfloat16) == kVecBytes,
                  "one 16 B load per thread");
    const auto x_addr = reinterpret_cast<uintptr_t>(prob.hidden_ptr);
    TORCH_CHECK(prob.num_tokens > 0 && prob.hidden % SfLayout::Gran == 0 &&
                    prob.hidden_row_stride % QuantX::PerThread == 0 &&
                    x_addr % kVecBytes == 0,
                "mla_prequant_x: needs T > 0, H a multiple of ", SfLayout::Gran,
                " and 16 B aligned hidden rows");
}
}  // namespace

template <class KT>
void launch_prequant_x(const MlaAttentionProblem<KT>& prob,
                       const ActAtoms<typename KT::GemmBase>& act_x) {
    check_prequant_x_inputs(prob);
    auto layout = sf_layout(act_x);
    launch_pdl(
        mla_prequant_x_kernel<KT::TileT, decltype(layout)>,
        {cute::ceil_div(prob.num_tokens * (prob.hidden / QuantX::PerThread),
                        kPrequantThreads),
         kPrequantThreads},
        reinterpret_cast<const __nv_bfloat16*>(prob.hidden_ptr),
        prob.hidden_row_stride, prob.ws_x_fp8, prob.ws_x_sf, layout,
        prob.num_tokens, prob.hidden);
}

template void launch_prequant_x(const MlaAttentionProblem<TraitsN16>&,
                                const ActAtoms<TraitsN16::GemmBase>&);
template void launch_prequant_x(const MlaAttentionProblem<TraitsN32>&,
                                const ActAtoms<TraitsN32::GemmBase>&);
template void launch_prequant_x(const MlaAttentionProblem<TraitsN64>&,
                                const ActAtoms<TraitsN64::GemmBase>&);
template void launch_prequant_x(const MlaAttentionProblem<TraitsN128>&,
                                const ActAtoms<TraitsN128::GemmBase>&);
}  // namespace halops::mla
