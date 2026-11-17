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

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <vector>

#include "common/activation.cuh"
#include "common/cuda_utils.h"
#include "common/mx_quant.cuh"
#include "common/pdl.cuh"

namespace halops {

namespace {

constexpr int kSgVec = sizeof(uint4) / sizeof(__nv_bfloat16);
constexpr int kSgBlock = 128;

__global__ void __launch_bounds__(kSgBlock)
    sigmoid_gate_mul_group_quant_kernel(const __nv_bfloat16* __restrict__ o,
                                        int64_t o_ld,
                                        const __nv_bfloat16* __restrict__ gate,
                                        int64_t gate_ld, uint2* __restrict__ q,
                                        uint8_t* __restrict__ sf, int num_vecs,
                                        int row_vecs) {
    constexpr int kGroupLanes = mxfp8::kGroup / kSgVec;
    pdl_trigger();
    pdl_wait();
    const int v = blockIdx.x * kSgBlock + threadIdx.x;
    if (v >= num_vecs) return;
    const int64_t row = v / row_vecs;
    const int col = v % row_vecs * kSgVec;
    const uint4 ov = *reinterpret_cast<const uint4*>(o + row * o_ld + col);
    const uint4 gv =
        *reinterpret_cast<const uint4*>(gate + row * gate_ld + col);
    const auto* op = reinterpret_cast<const __nv_bfloat162*>(&ov);
    const auto* gp = reinterpret_cast<const __nv_bfloat162*>(&gv);
    float2 y[kSgVec / 2];
    float amax = mxfp8::kAmaxFloor;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kSgVec / 2; ++i) {
        const float2 of = __bfloat1622float2(op[i]);
        const float2 gf = __bfloat1622float2(gp[i]);
        const float2 s = __bfloat1622float2(
            __floats2bfloat162_rn(sigmoid_rn(gf.x), sigmoid_rn(gf.y)));
        y[i] = __bfloat1622float2(
            __floats2bfloat162_rn(__fmul_rn(of.x, s.x), __fmul_rn(of.y, s.y)));
        amax = fmaxf(amax, fmaxf(fabsf(y[i].x), fabsf(y[i].y)));
    }
    const int lane = cutlass::canonical_lane_idx();
    const uint32_t group_mask = ((1u << kGroupLanes) - 1)
                                << (lane / kGroupLanes * kGroupLanes);
    CUTLASS_PRAGMA_UNROLL
    for (int m = kGroupLanes / 2; m > 0; m /= 2)
        amax = fmaxf(amax, __shfl_xor_sync(group_mask, amax, m));
    float inv;
    const uint8_t e =
        mxfp8::ue8m0_scale(__fmul_rn(amax, 1.0f / mxfp8::kE4m3Max), inv);
    const auto clamp = [](float x) {
        return fminf(fmaxf(x, -mxfp8::kE4m3Max), mxfp8::kE4m3Max);
    };
    uint2 codes;
    auto* cp = reinterpret_cast<__nv_fp8x2_storage_t*>(&codes);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kSgVec / 2; ++i) {
        cp[i] =
            __nv_cvt_float2_to_fp8x2(make_float2(clamp(__fmul_rn(y[i].x, inv)),
                                                 clamp(__fmul_rn(y[i].y, inv))),
                                     __NV_SATFINITE, __NV_E4M3);
    }
    q[v] = codes;
    if (v % kGroupLanes == 0) sf[v / kGroupLanes] = e;
}

}  // namespace

std::vector<torch::Tensor> SigmoidGateMulGroupQuant(const torch::Tensor& o,
                                                    const torch::Tensor& gate,
                                                    const bool use_pdl) {
    const CudaDeviceGuard device_guard(o.device());
    TORCH_CHECK(o.dim() == 2 && o.scalar_type() == torch::kBFloat16 &&
                    gate.scalar_type() == torch::kBFloat16 &&
                    gate.sizes().equals(o.sizes()),
                "o and gate must be bfloat16 tensors of one [T, hidden] shape");
    CheckRowsVecAligned(o, "o");
    CheckRowsVecAligned(gate, "gate");
    const int64_t num_tokens = o.size(0), hidden = o.size(1);
    TORCH_CHECK(hidden % mxfp8::kGroup == 0, "hidden must be a multiple of ",
                mxfp8::kGroup, ", got ", hidden);
    const int row_vecs = hidden / kSgVec;
    const int num_vecs = num_tokens * row_vecs;
    torch::Tensor q = torch::empty({num_tokens, hidden},
                                   o.options().dtype(torch::kFloat8_e4m3fn));
    torch::Tensor sf = torch::empty({num_tokens, hidden / mxfp8::kGroup},
                                    o.options().dtype(torch::kByte));
    if (num_vecs == 0) return {q, sf};
    HALOPS_CUDA_CHECK(LaunchKernelWithPdl(
        sigmoid_gate_mul_group_quant_kernel,
        dim3(cute::ceil_div(num_vecs, kSgBlock)), dim3(kSgBlock), kNoCluster, 0,
        GetCurrentCUDAStream(o), use_pdl,
        reinterpret_cast<const __nv_bfloat16*>(o.data_ptr()), o.stride(0),
        reinterpret_cast<const __nv_bfloat16*>(gate.data_ptr()), gate.stride(0),
        reinterpret_cast<uint2*>(q.data_ptr()), sf.data_ptr<uint8_t>(),
        num_vecs, row_vecs));
    return {q, sf};
}

}  // namespace halops
