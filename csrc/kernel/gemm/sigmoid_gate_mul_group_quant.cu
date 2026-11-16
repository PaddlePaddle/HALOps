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

#include <algorithm>
#include <vector>

#include "common/activation.cuh"
#include "common/cuda_utils.h"
#include "common/mx_quant.cuh"
#include "common/pdl.cuh"

namespace halops {

namespace {

constexpr int kSgVec = sizeof(uint4) / sizeof(__nv_bfloat16);
constexpr int kSgGroupThreads = mxfp8::kGroup / kSgVec;
constexpr int kSgMaxThreads = 1024;

__global__ void __launch_bounds__(kSgMaxThreads, 1)
    sigmoid_gate_mul_group_quant_kernel(const __nv_bfloat16* __restrict__ o,
                                        const __nv_bfloat16* __restrict__ gate,
                                        int64_t gate_ld, int hidden,
                                        uint8_t* __restrict__ q,
                                        uint8_t* __restrict__ sf) {
    const int64_t token = blockIdx.x;
    const int tid = threadIdx.x;
    const int num_vecs = hidden / kSgVec;
    const auto* o_row = reinterpret_cast<const uint4*>(o + token * hidden);
    const auto* g_row = reinterpret_cast<const uint4*>(gate + token * gate_ld);
    auto* q_row = reinterpret_cast<uint2*>(q + token * hidden);
    uint8_t* sf_row = sf + token * (hidden / mxfp8::kGroup);

    pdl_trigger();
    pdl_wait();
    for (int base = 0; base < num_vecs; base += blockDim.x) {
        const int v = base + tid;
        const bool active = v < num_vecs;
        uint4 ov{}, gv{};
        if (active) {
            ov = o_row[v];
            gv = g_row[v];
        }
        const auto* op = reinterpret_cast<const __nv_bfloat162*>(&ov);
        const auto* gp = reinterpret_cast<const __nv_bfloat162*>(&gv);
        float2 n[kSgVec / 2];
        float amax = 0.0f;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kSgVec / 2; ++i) {
            const float2 of = __bfloat1622float2(op[i]);
            const float2 gf = __bfloat1622float2(gp[i]);
            const float2 s = __bfloat1622float2(
                __floats2bfloat162_rn(sigmoid_rn(gf.x), sigmoid_rn(gf.y)));
            n[i] = __bfloat1622float2(__floats2bfloat162_rn(
                __fmul_rn(of.x, s.x), __fmul_rn(of.y, s.y)));
            amax = fmaxf(amax, fmaxf(fabsf(n[i].x), fabsf(n[i].y)));
        }
        CUTLASS_PRAGMA_UNROLL
        for (int m = kSgGroupThreads / 2; m > 0; m /= 2)
            amax = fmaxf(amax, __shfl_xor_sync(~0u, amax, m));
        float inv = 0.0f;
        const uint8_t e = mxfp8::ue8m0_scale(
            __fmul_rn(fmaxf(amax, mxfp8::kAmaxFloor), 1.0f / mxfp8::kE4m3Max),
            inv);
        if (!active) continue;
        uint2 out;
        auto* qp = reinterpret_cast<__nv_fp8x2_storage_t*>(&out);
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kSgVec / 2; ++i) {
            const float2 c = make_float2(
                fminf(fmaxf(__fmul_rn(n[i].x, inv), -mxfp8::kE4m3Max),
                      mxfp8::kE4m3Max),
                fminf(fmaxf(__fmul_rn(n[i].y, inv), -mxfp8::kE4m3Max),
                      mxfp8::kE4m3Max));
            qp[i] = __nv_cvt_float2_to_fp8x2(c, __NV_SATFINITE, __NV_E4M3);
        }
        q_row[v] = out;
        if (v % kSgGroupThreads == 0) sf_row[v / kSgGroupThreads] = e;
    }
}

}  // namespace

std::vector<torch::Tensor> SigmoidGateMulGroupQuant(const torch::Tensor& o,
                                                    const torch::Tensor& gate,
                                                    const bool use_pdl) {
    const CudaDeviceGuard device_guard(o.device());
    TORCH_CHECK(o.dim() == 2, "o must be [T, hidden]");
    const int64_t num_tokens = o.size(0), hidden = o.size(1);
    CheckContiguousTensor(o, "o", torch::kBFloat16, {num_tokens, hidden});
    TORCH_CHECK(hidden > 0 && hidden % mxfp8::kGroup == 0,
                "hidden must be a positive multiple of ", mxfp8::kGroup,
                ", got ", hidden);
    TORCH_CHECK(gate.is_cuda() && gate.scalar_type() == torch::kBFloat16 &&
                    gate.sizes().equals(o.sizes()),
                "gate must be a bfloat16 CUDA tensor of shape ", o.sizes());
    const int64_t gate_ld = num_tokens > 1 ? gate.stride(0) : hidden;
    TORCH_CHECK(
        gate.stride(1) == 1 && gate_ld % kSgVec == 0 &&
            reinterpret_cast<uintptr_t>(gate.data_ptr()) % sizeof(uint4) == 0,
        "gate rows must be contiguous and 16-byte aligned");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(o.data_ptr()) % sizeof(uint4) == 0,
                "o must be 16-byte aligned");
    torch::Tensor q = torch::empty({num_tokens, hidden},
                                   o.options().dtype(torch::kFloat8_e4m3fn));
    torch::Tensor sf = torch::empty({num_tokens, hidden / mxfp8::kGroup},
                                    o.options().dtype(torch::kByte));
    if (num_tokens == 0) return {q, sf};
    const int threads = std::min(
        cute::round_up(int(hidden / kSgVec), int{cutlass::NumThreadsPerWarp}),
        kSgMaxThreads);
    HALOPS_CUDA_CHECK(LaunchKernelWithPdl(
        sigmoid_gate_mul_group_quant_kernel, dim3(num_tokens), dim3(threads),
        kNoCluster, 0, GetCurrentCUDAStream(o), use_pdl,
        reinterpret_cast<const __nv_bfloat16*>(o.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(gate.data_ptr()), gate_ld,
        int(hidden), reinterpret_cast<uint8_t*>(q.data_ptr()),
        sf.data_ptr<uint8_t>()));
    return {q, sf};
}

}  // namespace halops
