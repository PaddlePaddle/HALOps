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

#include <optional>
#include <vector>

#include "common/mx_quant.cuh"
#include "common/pdl.cuh"
#include "kernel/gemm/gemm_utils.h"

namespace halops {

namespace {

constexpr int kRmsMaxThreads = 256;
constexpr int kRmsVec = sizeof(uint4) / sizeof(__nv_bfloat16);

// The mxfp8::kGroup bf16 values at src as floats.
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

// One block per (SfAtom-padded) row, one thread per 32-element quantization
// group: the row is read once, normalized in registers, rounded to bf16 and
// quantized with the group_wise_quant recipe.
template <bool kWriteY>
__global__ void __launch_bounds__(kRmsMaxThreads)
    rms_norm_quant_kernel(const __nv_bfloat16* __restrict__ x, int64_t ld,
                          const __nv_bfloat16* __restrict__ gamma, float eps,
                          int m, int k, uint8_t* __restrict__ q,
                          uint8_t* __restrict__ sf,
                          __nv_bfloat16* __restrict__ y) {
    constexpr int kG = mxfp8::kGroup;
    __shared__ float warp_sums[kRmsMaxThreads / cutlass::NumThreadsPerWarp];

    const int row = blockIdx.x;
    const int g = threadIdx.x;
    const int kb_dim = k / kG;
    const bool active = g < kb_dim;
    const int64_t sf_off = mxfp8::sf_atom_offset(row, g, kb_dim);
    const bool sf_lead = g % mxfp8::kAtomCols == 0;

    float w[kG];
    if (active) load_group(gamma + g * kG, w);

    pdl_wait();
    pdl_trigger();

    if (row >= m) {
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
    if constexpr (kWriteY) {
        uint4 packed[kG / kRmsVec];
        auto* e = reinterpret_cast<__nv_bfloat16*>(packed);
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kG; ++i) e[i] = __float2bfloat16_rn(v[i]);
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kG / kRmsVec; ++i) {
            reinterpret_cast<uint4*>(y + int64_t(row) * k + g * kG)[i] =
                packed[i];
        }
    }
    uint4 codes[2];
    const uint32_t word =
        mxfp8::sf_quad_word(mxfp8::quant_e4m3_group(v, codes));
    uint4* out = reinterpret_cast<uint4*>(q + int64_t(row) * k + g * kG);
    out[0] = codes[0];
    out[1] = codes[1];
    if (sf_lead) *reinterpret_cast<uint32_t*>(sf + sf_off) = word;
}

}  // namespace

std::vector<torch::Tensor> RmsNormQuant(
    const torch::Tensor& input, const torch::Tensor& weight, const double eps,
    const bool use_pdl, const std::optional<torch::Tensor>& output_quant_opt,
    const std::optional<torch::Tensor>& output_sf_opt,
    const std::optional<torch::Tensor>& output_bf16) {
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
    const int m_sf = AlignSfM(m);
    torch::Tensor q =
        output_quant_opt.has_value()
            ? *output_quant_opt
            : torch::empty({m, k},
                           input.options().dtype(torch::kFloat8_e4m3fn));
    torch::Tensor sf =
        output_sf_opt.has_value()
            ? *output_sf_opt
            : torch::empty({m_sf, kb_dim}, input.options().dtype(torch::kByte));
    TORCH_CHECK(
        q.is_contiguous() && q.scalar_type() == torch::kFloat8_e4m3fn &&
            q.numel() == int64_t(m) * k,
        "output_quant must be a contiguous float8_e4m3fn [M, K] tensor");
    TORCH_CHECK(sf.is_contiguous() && sf.scalar_type() == torch::kByte &&
                    sf.numel() == int64_t(m_sf) * kb_dim,
                "output_sf must be a contiguous uint8 tensor of ", m_sf, " x ",
                kb_dim, " elements");
    TORCH_CHECK(!output_bf16.has_value() ||
                    (output_bf16->is_contiguous() &&
                     output_bf16->scalar_type() == torch::kBFloat16 &&
                     output_bf16->numel() == int64_t(m) * k),
                "output_bf16 must be a contiguous bfloat16 [M, K] tensor");
    if (m_sf == 0) {
        return {q, sf};
    }
    const int threads = cute::round_up(kb_dim, int{cutlass::NumThreadsPerWarp});
    const auto* x = reinterpret_cast<const __nv_bfloat16*>(input.data_ptr());
    const auto* w = reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr());
    auto* q_ptr = reinterpret_cast<uint8_t*>(q.data_ptr());
    auto* sf_ptr = sf.data_ptr<uint8_t>();
    const cudaStream_t stream = GetCurrentCUDAStream(input);
    if (output_bf16.has_value()) {
        HALOPS_CUDA_CHECK(LaunchKernelWithPdl(
            rms_norm_quant_kernel<true>, m_sf, threads, 0, stream, use_pdl, x,
            input.stride(0), w, static_cast<float>(eps), m, k, q_ptr, sf_ptr,
            reinterpret_cast<__nv_bfloat16*>(output_bf16->data_ptr())));
    } else {
        HALOPS_CUDA_CHECK(LaunchKernelWithPdl(
            rms_norm_quant_kernel<false>, m_sf, threads, 0, stream, use_pdl, x,
            input.stride(0), w, static_cast<float>(eps), m, k, q_ptr, sf_ptr,
            static_cast<__nv_bfloat16*>(nullptr)));
    }
    return {q, sf};
}

}  // namespace halops
