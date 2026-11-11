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

#include <vector>

#include "common/mx_quant.cuh"
#include "common/pdl.cuh"
#include "kernel/gemm/gemm_utils.h"

namespace halops {

namespace {

constexpr int kRmsMaxThreads = 256;
constexpr int kRmsVec = sizeof(uint4) / sizeof(__nv_bfloat16);

CUTE_DEVICE void load_group(const __nv_bfloat16* src,
                            float (&v)[mxfp8::kGroup]) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < mxfp8::kGroup / kRmsVec; ++i) {
        const uint4 u = reinterpret_cast<const uint4*>(src)[i];
        const auto* e = reinterpret_cast<const __nv_bfloat16*>(&u);
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < kRmsVec; ++j)
            v[i * kRmsVec + j] = __bfloat162float(e[j]);
    }
}

template <bool kSwizzle>
__global__ void __launch_bounds__(kRmsMaxThreads)
    rms_norm_quant_kernel(const __nv_bfloat16* __restrict__ x, int64_t ld,
                          const __nv_bfloat16* __restrict__ gamma, float eps,
                          int m, int k, uint8_t* __restrict__ q,
                          uint8_t* __restrict__ sf) {
    constexpr int kG = mxfp8::kGroup;
    __shared__ float warp_sums[kRmsMaxThreads / cutlass::NumThreadsPerWarp];

    const int row = blockIdx.x;
    const int g = threadIdx.x;
    const int kb_dim = k / kG;
    const bool active = g < kb_dim;
    const int64_t sf_off = kSwizzle ? mxfp8::sf_atom_offset(row, g, kb_dim)
                                    : int64_t(row) * kb_dim + g;
    const bool sf_lead = g % mxfp8::kAtomCols == 0;

    float w[kG];
    if (active) load_group(gamma + g * kG, w);

    pdl_wait();
    pdl_trigger();

    if (kSwizzle && row >= m) {
        if (active && sf_lead) *reinterpret_cast<uint32_t*>(sf + sf_off) = 0;
        return;
    }

    float v[kG];
    float sum_sq = 0.0f;
    if (active) {
        load_group(x + row * ld + g * kG, v);
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kG; ++i) sum_sq += v[i] * v[i];
    }
    CUTLASS_PRAGMA_UNROLL
    for (int s = cutlass::NumThreadsPerWarp / 2; s > 0; s /= 2) {
        sum_sq += __shfl_xor_sync(~0u, sum_sq, s);
    }
    const int warp = g / cutlass::NumThreadsPerWarp;
    const int lane = g % cutlass::NumThreadsPerWarp;
    const int warps = blockDim.x / cutlass::NumThreadsPerWarp;
    if (warps > 1) {
        if (lane == 0) warp_sums[warp] = sum_sq;
        __syncthreads();
        sum_sq = 0.0f;
        for (int i = 0; i < warps; ++i) sum_sq += warp_sums[i];
    }
    if (!active) return;

    const float rrms = rsqrtf(sum_sq / k + eps);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kG; ++i) {
        v[i] = __bfloat162float(__float2bfloat16_rn(v[i] * rrms * w[i]));
    }
    uint4 codes[2];
    const uint8_t e = mxfp8::quant_e4m3_group(v, codes);
    uint4* out = reinterpret_cast<uint4*>(q + int64_t(row) * k + g * kG);
    out[0] = codes[0];
    out[1] = codes[1];
    if constexpr (kSwizzle) {
        const uint32_t word = mxfp8::sf_quad_word(e);
        if (sf_lead) *reinterpret_cast<uint32_t*>(sf + sf_off) = word;
    } else {
        sf[sf_off] = e;
    }
}

}  // namespace

std::vector<torch::Tensor> RmsNormQuant(const torch::Tensor& input,
                                        const torch::Tensor& weight,
                                        const double eps,
                                        const bool use_swizzle,
                                        const bool use_pdl) {
    const CudaDeviceGuard device_guard(input.device());
    TORCH_CHECK(input.dim() == 2 && input.scalar_type() == torch::kBFloat16,
                "input must be a 2D bfloat16 tensor [M, K]");
    const int m = input.size(0);
    const int k = input.size(1);
    CheckRowsVecAligned(input, "input");
    const int kb_dim = k / mxfp8::kGroup;
    TORCH_CHECK(
        k % (mxfp8::kGroup * mxfp8::kAtomCols) == 0 && kb_dim <= kRmsMaxThreads,
        "K must be a multiple of 128 and at most ",
        kRmsMaxThreads * mxfp8::kGroup, ", got ", k);
    TORCH_CHECK(weight.dim() == 1 && weight.size(0) == k &&
                    weight.scalar_type() == torch::kBFloat16 &&
                    weight.is_contiguous(),
                "weight must be a contiguous bfloat16 [K] tensor");
    const int m_sf = use_swizzle ? AlignSfM(m) : m;
    torch::Tensor q =
        torch::empty({m, k}, input.options().dtype(torch::kFloat8_e4m3fn));
    torch::Tensor sf =
        torch::empty({m_sf, kb_dim}, input.options().dtype(torch::kByte));
    if (m_sf == 0) {
        return {q, sf};
    }
    const int threads = cute::round_up(kb_dim, int{cutlass::NumThreadsPerWarp});
    const auto* x = reinterpret_cast<const __nv_bfloat16*>(input.data_ptr());
    const auto* w = reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr());
    auto* q_ptr = reinterpret_cast<uint8_t*>(q.data_ptr());
    auto* sf_ptr = sf.data_ptr<uint8_t>();
    const cudaStream_t stream = GetCurrentCUDAStream(input);
    const auto kernel = use_swizzle ? rms_norm_quant_kernel<true>
                                    : rms_norm_quant_kernel<false>;
    HALOPS_CUDA_CHECK(LaunchKernelWithPdl(
        kernel, m_sf, threads, kNoCluster, 0, stream, use_pdl, x,
        input.stride(0), w, static_cast<float>(eps), m, k, q_ptr, sf_ptr));
    return {q, sf};
}

}  // namespace halops
