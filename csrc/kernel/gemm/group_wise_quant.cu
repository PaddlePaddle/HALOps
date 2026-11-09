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
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <optional>
#include <string>

#include "common/mx_quant.cuh"
#include "common/pdl.cuh"
#include "kernel/gemm/gemm_utils.h"

namespace halops {

constexpr float FP4_E2M1_MAX = 6.0f;

enum class QuantMode : int32_t {
    W8A8MXFP8 = 0,
    W4A4MXFP4 = 1,
    W4A4NVFP4 = 2,
};

constexpr bool IsFp4(QuantMode mode) { return mode != QuantMode::W8A8MXFP8; }

constexpr int GroupSize(QuantMode mode) {
    return mode == QuantMode::W4A4NVFP4 ? 16 : 32;
}

QuantMode ParseQuantMode(const std::string& mode) {
    TORCH_CHECK(
        mode == "w8a8-mxfp8" || mode == "w4a4-mxfp4" || mode == "w4a4-nvfp4",
        "Unsupported quantization mode: ", mode,
        ". Expected 'w8a8-mxfp8', 'w4a4-mxfp4' or 'w4a4-nvfp4'.");
    if (mode == "w8a8-mxfp8") {
        return QuantMode::W8A8MXFP8;
    }
    if (mode == "w4a4-mxfp4") {
        return QuantMode::W4A4MXFP4;
    }
    return QuantMode::W4A4NVFP4;
}

// Two FP4 E2M1 codes (x in the low nibble), rounded to nearest even with -0
// coded as +0.
__device__ __forceinline__ uint8_t quantize_to_fp4x2(float2 v) {
    v = __fadd2_rn(v, make_float2(0.0f, 0.0f));
    return __nv_cvt_float2_to_fp4x2(v, __NV_E2M1, cudaRoundNearest);
}

// UE4M3 scale: for non-negative values the __nv_fp8_e4m3 bit pattern equals
// UE4M3 (sign bit 0, bias 7).
__device__ __forceinline__ uint8_t float32_to_ue4m3(float val) {
    val = fmaxf(fminf(val, mxfp8::kE4m3Max), 0.0f);
    __nv_fp8_e4m3 fp8_val(val);
    uint8_t bits;
    memcpy(&bits, &fp8_val, sizeof(uint8_t));
    return bits;
}

__device__ __forceinline__ float ue4m3_to_float32(uint8_t bits) {
    __nv_fp8_e4m3 fp8_val;
    memcpy(&fp8_val, &bits, sizeof(uint8_t));
    return static_cast<float>(fp8_val);
}

template <typename T>
struct TypeTraits;

template <>
struct TypeTraits<__nv_bfloat16> {
    static __device__ __forceinline__ float to_float(const __nv_bfloat16& val) {
        return __bfloat162float(val);
    }
    static constexpr int elem_per_float4 = 8;
};

template <>
struct TypeTraits<__half> {
    static __device__ __forceinline__ float to_float(const __half& val) {
        return __half2float(val);
    }
    static constexpr int elem_per_float4 = 8;
};

// One thread per quantization group. With USE_SWIZZLE, kb_dim % 4 == 0, so the
// 4 groups of one SfAtom word sit in 4 adjacent lanes of one row and store
// their scales as one 32-bit word.
template <typename T, QuantMode MODE, bool USE_SWIZZLE, bool USE_PDL>
__global__ void group_wise_quant_kernel(
    const T* __restrict__ input, int64_t ld,
    void* __restrict__ output_quant,  // FP8 e4m3 or packed FP4 bytes
    void* __restrict__ output_sf, int m, int k, int kb_dim) {
    constexpr int GROUP_SIZE = GroupSize(MODE);
    static_assert(GROUP_SIZE == 16 || GROUP_SIZE == 32,
                  "Only GROUP_SIZE=16 or 32 is supported");
    using Traits = TypeTraits<T>;
    constexpr int NUM_LOAD_VEC = GROUP_SIZE / Traits::elem_per_float4;

    const int total_groups = (USE_SWIZZLE ? AlignSfM(m) : m) * kb_dim;
    uint8_t* const sf = reinterpret_cast<uint8_t*>(output_sf);

    // The loop below holds the first global access, so a PDL launch resolves
    // the grid dependency here: the index math above overlaps with the tail of
    // the producing kernel.
    if constexpr (USE_PDL) {
        pdl_wait();
        pdl_trigger();
    }

    for (int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
         group_idx < total_groups; group_idx += gridDim.x * blockDim.x) {
        const int row = group_idx / kb_dim;
        const int kb = group_idx % kb_dim;
        const int k_start = kb * GROUP_SIZE;
        const int64_t sf_off = USE_SWIZZLE
                                   ? mxfp8::sf_atom_offset(row, kb, kb_dim)
                                   : int64_t(row) * kb_dim + kb;
        const bool sf_lead = !USE_SWIZZLE || kb % mxfp8::kAtomCols == 0;
        if (row >= m) {
            if (sf_lead) {
                if constexpr (USE_SWIZZLE) {
                    *reinterpret_cast<uint32_t*>(sf + sf_off) = 0;
                } else {
                    sf[sf_off] = 0;
                }
            }
            continue;
        }

        const float4* input_vec =
            reinterpret_cast<const float4*>(input + row * ld + k_start);
        float vals[GROUP_SIZE];
#pragma unroll
        for (int i = 0; i < NUM_LOAD_VEC; ++i) {
            const float4 v = input_vec[i];
            const T* elem_ptr = reinterpret_cast<const T*>(&v);
#pragma unroll
            for (int j = 0; j < Traits::elem_per_float4; ++j) {
                vals[i * Traits::elem_per_float4 + j] =
                    Traits::to_float(elem_ptr[j]);
            }
        }

        uint8_t scale_byte;
        if constexpr (IsFp4(MODE)) {
            float amax = mxfp8::kAmaxFloor;
#pragma unroll
            for (int i = 0; i < GROUP_SIZE; ++i) {
                amax = fmaxf(amax, fabsf(vals[i]));
            }
            float inv_scale;
            if constexpr (MODE == QuantMode::W4A4NVFP4) {
                scale_byte = float32_to_ue4m3(amax / FP4_E2M1_MAX);
                // UE4M3 may round a tiny scale to 0; use the smallest positive
                // code (2^-9).
                if (scale_byte == 0 && amax > 0.0f) {
                    scale_byte = 0x01;
                }
                inv_scale = 1.0f / ue4m3_to_float32(scale_byte);
            } else {
                scale_byte = mxfp8::ue8m0_scale(amax / FP4_E2M1_MAX, inv_scale);
            }
            uint8_t fp4[GROUP_SIZE / 2];
#pragma unroll
            for (int i = 0; i < GROUP_SIZE; i += 2) {
                fp4[i / 2] = quantize_to_fp4x2(make_float2(
                    fminf(fmaxf(vals[i] * inv_scale, -FP4_E2M1_MAX),
                          FP4_E2M1_MAX),
                    fminf(fmaxf(vals[i + 1] * inv_scale, -FP4_E2M1_MAX),
                          FP4_E2M1_MAX)));
            }
            using Packed = cute::conditional_t<GROUP_SIZE == 32, uint4, uint2>;
            *reinterpret_cast<Packed*>(
                reinterpret_cast<uint8_t*>(output_quant) + row * (k / 2) +
                k_start / 2) = *reinterpret_cast<const Packed*>(fp4);
        } else {
            uint4 codes[2];
            scale_byte = mxfp8::quant_e4m3_group(vals, codes);
            uint4* out = reinterpret_cast<uint4*>(
                reinterpret_cast<uint8_t*>(output_quant) + row * k + k_start);
            out[0] = codes[0];
            out[1] = codes[1];
        }

        if constexpr (USE_SWIZZLE) {
            const uint32_t word = mxfp8::sf_quad_word(scale_byte);
            if (sf_lead) {
                *reinterpret_cast<uint32_t*>(sf + sf_off) = word;
            }
        } else {
            sf[sf_off] = scale_byte;
        }
    }
}

template <typename T, QuantMode MODE, bool USE_SWIZZLE, bool USE_PDL>
void LaunchQuantKernel(const T* input, int64_t ld, void* output_quant,
                       void* output_sf, int m, int k, int kb_dim, int grid_dim,
                       int block_dim, cudaStream_t stream) {
    LaunchKernelWithPdl(group_wise_quant_kernel<T, MODE, USE_SWIZZLE, USE_PDL>,
                        grid_dim, block_dim, kNoCluster, 0, stream, USE_PDL,
                        input, ld, output_quant, output_sf, m, k, kb_dim);
}

template <typename T, QuantMode MODE, bool USE_SWIZZLE>
void LaunchGroupWiseQuantForSwizzle(const T* input, int64_t ld,
                                    void* output_quant, void* output_sf, int m,
                                    int k, int kb_dim, int grid_dim,
                                    int block_dim, cudaStream_t stream,
                                    bool use_pdl) {
    if (use_pdl) {
        LaunchQuantKernel<T, MODE, USE_SWIZZLE, true>(
            input, ld, output_quant, output_sf, m, k, kb_dim, grid_dim,
            block_dim, stream);
    } else {
        LaunchQuantKernel<T, MODE, USE_SWIZZLE, false>(
            input, ld, output_quant, output_sf, m, k, kb_dim, grid_dim,
            block_dim, stream);
    }
}

template <typename T, QuantMode MODE>
void LaunchGroupWiseQuant(const T* input, int64_t ld, void* output_quant,
                          void* output_sf, int m, int k, int kb_dim,
                          bool use_swizzle, int grid_dim, int block_dim,
                          cudaStream_t stream, bool use_pdl) {
    if (use_swizzle) {
        LaunchGroupWiseQuantForSwizzle<T, MODE, true>(
            input, ld, output_quant, output_sf, m, k, kb_dim, grid_dim,
            block_dim, stream, use_pdl);
    } else {
        LaunchGroupWiseQuantForSwizzle<T, MODE, false>(
            input, ld, output_quant, output_sf, m, k, kb_dim, grid_dim,
            block_dim, stream, use_pdl);
    }
}

template <typename T>
void LaunchGroupWiseQuantForMode(const T* input, int64_t ld, void* output_quant,
                                 void* output_sf, int m, int k, int kb_dim,
                                 QuantMode mode, bool use_swizzle, int grid_dim,
                                 int block_dim, cudaStream_t stream,
                                 bool use_pdl) {
    if (mode == QuantMode::W8A8MXFP8) {
        LaunchGroupWiseQuant<T, QuantMode::W8A8MXFP8>(
            input, ld, output_quant, output_sf, m, k, kb_dim, use_swizzle,
            grid_dim, block_dim, stream, use_pdl);
    } else if (mode == QuantMode::W4A4MXFP4) {
        LaunchGroupWiseQuant<T, QuantMode::W4A4MXFP4>(
            input, ld, output_quant, output_sf, m, k, kb_dim, use_swizzle,
            grid_dim, block_dim, stream, use_pdl);
    } else {
        LaunchGroupWiseQuant<T, QuantMode::W4A4NVFP4>(
            input, ld, output_quant, output_sf, m, k, kb_dim, use_swizzle,
            grid_dim, block_dim, stream, use_pdl);
    }
}

std::vector<torch::Tensor> GroupWiseQuant(
    const torch::Tensor& input, const std::string& mode, const bool use_swizzle,
    const bool use_pdl, const std::optional<torch::Tensor>& output_quant_opt,
    const std::optional<torch::Tensor>& output_sf_opt) {
    const CudaDeviceGuard device_guard(input.device());
    TORCH_CHECK(input.dim() == 2, "input must be a 2D tensor of shape [m, k]");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 ||
                    input.scalar_type() == torch::kHalf,
                "input dtype must be bfloat16 or float16");
    CheckRowsVecAligned(input, "input");
    const QuantMode quant_mode = ParseQuantMode(mode);
    const int group_size = GroupSize(quant_mode);

    const int m = static_cast<int>(input.size(0));
    const int k = static_cast<int>(input.size(1));
    const int kb_dim = k / group_size;

    TORCH_CHECK(k % group_size == 0,
                "k must be divisible by the quantization group size (",
                group_size, ") for mode ", mode, ", got k=", k);
    if (use_swizzle) {
        // m_sf pads the scale rows to whole 128-row atoms, so any m (including
        // 0) is valid; only kb_dim is constrained.
        TORCH_CHECK(kb_dim % 4 == 0,
                    "scale kb_dim must be divisible by 4 for CUTLASS SfAtom "
                    "layout, got ",
                    kb_dim);
    }

    const int m_sf = use_swizzle ? AlignSfM(m) : m;

    // output_quant: (m, k) fp8, or (m, k/2) packed FP4.
    // output_sf: (m_sf, kb_dim) uint8 holding UE8M0 / UE4M3 bits;
    const auto quant_dtype =
        IsFp4(quant_mode) ? torch::kByte : torch::kFloat8_e4m3fn;
    const int quant_k = IsFp4(quant_mode) ? k / 2 : k;
    torch::Tensor output_quant =
        output_quant_opt.has_value()
            ? *output_quant_opt
            : torch::empty({m, quant_k}, input.options().dtype(quant_dtype));
    torch::Tensor output_sf =
        output_sf_opt.has_value()
            ? *output_sf_opt
            : torch::empty({m_sf, kb_dim}, input.options().dtype(torch::kByte));
    TORCH_CHECK(output_quant.is_contiguous() &&
                    output_quant.scalar_type() == quant_dtype &&
                    output_quant.numel() == int64_t(m) * quant_k,
                "output_quant must be a contiguous [", m, ", ", quant_k,
                "] tensor of the mode's dtype");
    TORCH_CHECK(output_sf.is_contiguous() &&
                    output_sf.scalar_type() == torch::kByte &&
                    output_sf.numel() == int64_t(m_sf) * kb_dim,
                "output_sf must be a contiguous uint8 tensor of ", m_sf, " x ",
                kb_dim, " elements");

    const int total_groups = m_sf * kb_dim;
    const int block_dim = 128;
    const int grid_dim = max((total_groups + block_dim - 1) / block_dim, 1);
    void* quant_data = output_quant.data_ptr();
    auto* sf_data = output_sf.data_ptr<uint8_t>();
    const int64_t ld = input.stride(0);

    const cudaStream_t stream = GetCurrentCUDAStream(input);
    if (input.scalar_type() == torch::kBFloat16) {
        const auto* input_ptr = reinterpret_cast<const __nv_bfloat16*>(
            input.mutable_data_ptr<c10::BFloat16>());
        LaunchGroupWiseQuantForMode<__nv_bfloat16>(
            input_ptr, ld, quant_data, sf_data, m, k, kb_dim, quant_mode,
            use_swizzle, grid_dim, block_dim, stream, use_pdl);
    } else {
        const auto* input_ptr = reinterpret_cast<const __half*>(
            input.mutable_data_ptr<c10::Half>());
        LaunchGroupWiseQuantForMode<__half>(
            input_ptr, ld, quant_data, sf_data, m, k, kb_dim, quant_mode,
            use_swizzle, grid_dim, block_dim, stream, use_pdl);
    }

    cudaError_t cuda_status = cudaGetLastError();
    TORCH_CHECK(cuda_status == cudaSuccess,
                "CUDA error: ", cudaGetErrorString(cuda_status));

    return {output_quant, output_sf};
}

}  // namespace halops
