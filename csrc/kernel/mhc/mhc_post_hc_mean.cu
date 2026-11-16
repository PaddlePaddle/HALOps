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
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>

#include "common/cuda_utils.h"
#include "common/pdl.cuh"

namespace halops {

namespace {

constexpr int kHcMeanThreads = 256;

template <int kHc>
__global__ void __launch_bounds__(kHcMeanThreads, 1)
    mhc_post_hc_mean_kernel(const __nv_bfloat16* __restrict__ x,
                            const __nv_bfloat16* __restrict__ residual,
                            const float* __restrict__ post,
                            const float* __restrict__ comb,
                            __nv_bfloat16* __restrict__ out, int hidden) {
    __shared__ float s_post[kHc];
    __shared__ float s_comb[kHc * kHc];
    const int64_t token = blockIdx.x;
    const int tid = threadIdx.x;

    pdl_trigger();
    pdl_wait();
    if (tid < kHc) s_post[tid] = post[token * kHc + tid];
    if (tid < kHc * kHc) s_comb[tid] = comb[token * kHc * kHc + tid];
    __syncthreads();

    const __nv_bfloat16* res = residual + token * kHc * hidden;
    for (int h = tid; h < hidden; h += blockDim.x) {
        const float xh = __bfloat162float(x[token * hidden + h]);
        float r[kHc];
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kHc; ++i)
            r[i] = __bfloat162float(res[int64_t(i) * hidden + h]);
        float mean = 0.0f;
        CUTLASS_PRAGMA_UNROLL
        for (int o = 0; o < kHc; ++o) {
            float acc = __fmul_rn(s_post[o], xh);
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < kHc; ++i)
                acc = __fmaf_rn(s_comb[i * kHc + o], r[i], acc);
            mean = __fadd_rn(mean, __bfloat162float(__float2bfloat16_rn(acc)));
        }
        out[token * hidden + h] =
            __float2bfloat16_rn(__fmul_rn(mean, 1.0f / kHc));
    }
}

using HcMeanKernel = void (*)(const __nv_bfloat16*, const __nv_bfloat16*,
                              const float*, const float*, __nv_bfloat16*, int);

HcMeanKernel SelectHcMeanKernel(int hc) {
    switch (hc) {
        case 2:
            return mhc_post_hc_mean_kernel<2>;
        case 3:
            return mhc_post_hc_mean_kernel<3>;
        case 4:
            return mhc_post_hc_mean_kernel<4>;
    }
    TORCH_CHECK(false, "hc must be 2, 3 or 4, got ", hc);
}

}  // namespace

torch::Tensor MhcPostHcMean(const torch::Tensor& x,
                            const torch::Tensor& residual,
                            const torch::Tensor& post,
                            const torch::Tensor& comb, const bool use_pdl) {
    const CudaDeviceGuard device_guard(x.device());
    TORCH_CHECK(residual.dim() == 3, "residual must be [T, hc, hidden]");
    const int64_t num_tokens = residual.size(0), hc = residual.size(1);
    const int64_t hidden = residual.size(2);
    CheckContiguousTensor(x, "x", torch::kBFloat16, {num_tokens, hidden});
    CheckContiguousTensor(residual, "residual", torch::kBFloat16,
                          {num_tokens, hc, hidden});
    CheckContiguousTensor(post, "post", torch::kFloat32, {num_tokens, hc});
    CheckContiguousTensor(comb, "comb", torch::kFloat32, {num_tokens, hc, hc});
    const HcMeanKernel kernel = SelectHcMeanKernel(hc);
    torch::Tensor out = torch::empty({num_tokens, hidden}, x.options());
    if (num_tokens == 0) return out;
    HALOPS_CUDA_CHECK(LaunchKernelWithPdl(
        kernel, dim3(num_tokens), dim3(kHcMeanThreads), kNoCluster, 0,
        GetCurrentCUDAStream(x), use_pdl,
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr()),
        post.const_data_ptr<float>(), comb.const_data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), int(hidden)));
    return out;
}

}  // namespace halops
