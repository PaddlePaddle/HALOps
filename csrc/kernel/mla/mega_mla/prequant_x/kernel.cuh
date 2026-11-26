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

#pragma once

#include "kernel/mla/mega_mla/mla_params.cuh"
#include "kernel/mla/mega_mla/mla_utils.cuh"
#include "kernel/mla/mega_mla/prequant_x/config.cuh"

namespace halops::mla {
/// Quantises the bf16 hidden rows to fp8 + ue8m0 ahead of the attention kernel.
template <int TileTokens, class LayoutSF>
__global__ void __launch_bounds__(MlaPrequantXConfig::kThreads)
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
}  // namespace halops::mla
