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
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>

#include "kernel/gemm/gemm_utils.h"
#include "kernel/gemm/pdl.cuh"

namespace halops {

constexpr float FP8_E4M3_MAX = 448.0f;

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

// FP4 E2M1 magnitudes {0, 0.5, 1, 1.5, 2, 3, 4, 6}, rounded at the midpoints.
// Returns the 4-bit code [S M2 M1 M0].
__device__ __forceinline__ uint8_t quantize_to_fp4_code(float val) {
    uint8_t sign = (val < 0.0f) ? 0x8 : 0x0;
    float abs_val = fabsf(val);
    uint8_t mag;
    if (abs_val < 0.25f)
        mag = 0;
    else if (abs_val < 0.75f)
        mag = 1;
    else if (abs_val < 1.25f)
        mag = 2;
    else if (abs_val < 1.75f)
        mag = 3;
    else if (abs_val < 2.5f)
        mag = 4;
    else if (abs_val < 3.5f)
        mag = 5;
    else if (abs_val < 5.0f)
        mag = 6;
    else
        mag = 7;
    return sign | mag;
}

// UE4M3 scale: for non-negative values the __nv_fp8_e4m3 bit pattern equals
// UE4M3 (sign bit 0, bias 7).
__device__ __forceinline__ uint8_t float32_to_ue4m3(float val) {
    val = fmaxf(fminf(val, 448.0f), 0.0f);
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

__device__ __forceinline__ uint8_t float32_to_ue8m0(float scale,
                                                    float& inv_scale) {
    const float biased_exp =
        fminf(fmaxf(ceilf(log2f(scale)) + 127.0f, 0.0f), 254.0f);
    inv_scale = exp2f(127.0f - biased_exp);
    return static_cast<uint8_t>(biased_exp);
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

template <typename T, QuantMode MODE, bool USE_SWIZZLE, bool USE_PDL>
__global__ void group_wise_quant_kernel(
    const T* __restrict__ input,
    void* __restrict__ output_quant,  // FP8 e4m3 or packed FP4 bytes
    void* __restrict__ output_sf, int m, int k, int kb_dim) {
    constexpr int GROUP_SIZE = GroupSize(MODE);
    static_assert(GROUP_SIZE == 16 || GROUP_SIZE == 32,
                  "Only GROUP_SIZE=16 or 32 is supported");
    using Traits = TypeTraits<T>;

    constexpr int NUM_LOAD_VEC = GROUP_SIZE / Traits::elem_per_float4;

    const int num_k_tiles = kb_dim / 4;
    const int total_groups = m * kb_dim;

    // The loop below holds the first global access, so a PDL launch resolves
    // the grid dependency here: the index math above overlaps with the tail of
    // the producing kernel.
    if constexpr (USE_PDL) {
        PdlWaitForDependents();
    }

    for (int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
         group_idx < total_groups; group_idx += gridDim.x * blockDim.x) {
        const int row = group_idx / kb_dim;
        const int kb = group_idx % kb_dim;
        const int k_start = kb * GROUP_SIZE;

        const float4* input_vec =
            reinterpret_cast<const float4*>(input + row * k + k_start);

        float vals[GROUP_SIZE];
        float amax = 1e-10f;

#pragma unroll
        for (int i = 0; i < NUM_LOAD_VEC; ++i) {
            float4 v = input_vec[i];
            const T* elem_ptr = reinterpret_cast<const T*>(&v);

#pragma unroll
            for (int j = 0; j < Traits::elem_per_float4; ++j) {
                float val = Traits::to_float(elem_ptr[j]);
                vals[i * Traits::elem_per_float4 + j] = val;
                amax = fmaxf(amax, fabsf(val));
            }
        }

        uint8_t scale_byte;
        float inv_scale;

        if constexpr (MODE == QuantMode::W4A4NVFP4) {
            float scale = amax / FP4_E2M1_MAX;
            scale_byte = float32_to_ue4m3(scale);
            // UE4M3 may round a tiny scale to 0; use the smallest positive
            // code (2^-9).
            if (scale_byte == 0 && amax > 0.0f) {
                scale_byte = 0x01;
            }
            float scale_decoded = ue4m3_to_float32(scale_byte);
            inv_scale = 1.0f / scale_decoded;
        } else if constexpr (MODE == QuantMode::W4A4MXFP4) {
            scale_byte = float32_to_ue8m0(amax / FP4_E2M1_MAX, inv_scale);
        } else {
            scale_byte = float32_to_ue8m0(amax / FP8_E4M3_MAX, inv_scale);
        }

        if constexpr (IsFp4(MODE)) {
            uint8_t fp4_codes[GROUP_SIZE];

#pragma unroll
            for (int i = 0; i < GROUP_SIZE; ++i) {
                float scaled = vals[i] * inv_scale;
                scaled = fminf(fmaxf(scaled, -FP4_E2M1_MAX), FP4_E2M1_MAX);
                fp4_codes[i] = quantize_to_fp4_code(scaled);
            }

            // Pack: even index = low nibble, odd index = high nibble
            uint8_t* output_ptr = reinterpret_cast<uint8_t*>(output_quant) +
                                  row * (k / 2) + k_start / 2;
#pragma unroll
            for (int i = 0; i < GROUP_SIZE; i += 2) {
                output_ptr[i / 2] = (fp4_codes[i + 1] << 4) | fp4_codes[i];
            }
        } else {
            __nv_fp8_e4m3 fp8_vals[GROUP_SIZE];
            constexpr int NUM_STORE_VEC = GROUP_SIZE / 16;

#pragma unroll
            for (int i = 0; i < GROUP_SIZE; ++i) {
                float scaled = vals[i] * inv_scale;
                scaled = fminf(fmaxf(scaled, -FP8_E4M3_MAX), FP8_E4M3_MAX);
                fp8_vals[i] = __nv_fp8_e4m3(scaled);
            }

            float4* output_vec = reinterpret_cast<float4*>(
                reinterpret_cast<__nv_fp8_e4m3*>(output_quant) + row * k +
                k_start);
            const float4* fp8_vec = reinterpret_cast<const float4*>(fp8_vals);

#pragma unroll
            for (int i = 0; i < NUM_STORE_VEC; ++i) {
                output_vec[i] = fp8_vec[i];
            }
        }

        int sf_offset;
        if constexpr (USE_SWIZZLE) {
            // Pad and swizzle into CUTLASS SfAtom layout.
            const int n_tile = row / 128;
            const int n_local = row % 128;
            const int mn_i = n_local % 32;
            const int mn_j = n_local / 32;
            const int k_tile = kb / 4;
            const int sf_l = kb % 4;

            const int cutlass_byte = (n_tile * num_k_tiles + k_tile) * 512 +
                                     mn_i * 16 + mn_j * 4 + sf_l;
            const int n_phys = cutlass_byte / kb_dim;
            const int kb_phys = cutlass_byte % kb_dim;
            sf_offset = n_phys * kb_dim + kb_phys;
        } else {
            sf_offset = row * kb_dim + kb;
        }
        reinterpret_cast<uint8_t*>(output_sf)[sf_offset] = scale_byte;
    }

    // Every store of this block is out: publish it and let a PDL-dependent
    // grid (typically the GEMM consuming these buffers) start.
    if constexpr (USE_PDL) {
        PdlTriggerAfterBlockWrites();
    }
}

template <typename T, QuantMode MODE, bool USE_SWIZZLE, bool USE_PDL>
void LaunchQuantKernel(const T* input, void* output_quant, void* output_sf,
                       int m, int k, int kb_dim, int grid_dim, int block_dim,
                       cudaStream_t stream) {
    LaunchKernelWithPdl(group_wise_quant_kernel<T, MODE, USE_SWIZZLE, USE_PDL>,
                        grid_dim, block_dim, 0, stream, USE_PDL, input,
                        output_quant, output_sf, m, k, kb_dim);
}

template <typename T, QuantMode MODE, bool USE_SWIZZLE>
void LaunchGroupWiseQuantForSwizzle(const T* input, void* output_quant,
                                    void* output_sf, int m, int k, int kb_dim,
                                    int grid_dim, int block_dim,
                                    cudaStream_t stream, bool use_pdl) {
    if (use_pdl) {
        LaunchQuantKernel<T, MODE, USE_SWIZZLE, true>(
            input, output_quant, output_sf, m, k, kb_dim, grid_dim, block_dim,
            stream);
    } else {
        LaunchQuantKernel<T, MODE, USE_SWIZZLE, false>(
            input, output_quant, output_sf, m, k, kb_dim, grid_dim, block_dim,
            stream);
    }
}

template <typename T, QuantMode MODE>
void LaunchGroupWiseQuant(const T* input, void* output_quant, void* output_sf,
                          int m, int k, int kb_dim, bool use_swizzle,
                          int grid_dim, int block_dim, cudaStream_t stream,
                          bool use_pdl) {
    if (use_swizzle) {
        LaunchGroupWiseQuantForSwizzle<T, MODE, true>(
            input, output_quant, output_sf, m, k, kb_dim, grid_dim, block_dim,
            stream, use_pdl);
    } else {
        LaunchGroupWiseQuantForSwizzle<T, MODE, false>(
            input, output_quant, output_sf, m, k, kb_dim, grid_dim, block_dim,
            stream, use_pdl);
    }
}

template <typename T>
void LaunchGroupWiseQuantForMode(const T* input, void* output_quant,
                                 void* output_sf, int m, int k, int kb_dim,
                                 QuantMode mode, bool use_swizzle, int grid_dim,
                                 int block_dim, cudaStream_t stream,
                                 bool use_pdl) {
    if (mode == QuantMode::W8A8MXFP8) {
        LaunchGroupWiseQuant<T, QuantMode::W8A8MXFP8>(
            input, output_quant, output_sf, m, k, kb_dim, use_swizzle, grid_dim,
            block_dim, stream, use_pdl);
    } else if (mode == QuantMode::W4A4MXFP4) {
        LaunchGroupWiseQuant<T, QuantMode::W4A4MXFP4>(
            input, output_quant, output_sf, m, k, kb_dim, use_swizzle, grid_dim,
            block_dim, stream, use_pdl);
    } else {
        LaunchGroupWiseQuant<T, QuantMode::W4A4NVFP4>(
            input, output_quant, output_sf, m, k, kb_dim, use_swizzle, grid_dim,
            block_dim, stream, use_pdl);
    }
}

std::vector<torch::Tensor> GroupWiseQuant(const torch::Tensor& input,
                                          const std::string& mode,
                                          const bool use_swizzle,
                                          const bool use_pdl) {
    const CudaDeviceGuard device_guard(input.device());
    TORCH_CHECK(input.dim() == 2, "input must be a 2D tensor of shape [m, k]");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 ||
                    input.scalar_type() == torch::kHalf,
                "input dtype must be bfloat16 or float16");
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
    torch::Tensor output_quant;
    if (IsFp4(quant_mode)) {
        output_quant =
            torch::empty({m, k / 2}, input.options().dtype(torch::kByte));
    } else {
        output_quant =
            torch::empty({m, k}, input.options().dtype(torch::kFloat8_e4m3fn));
    }

    auto output_sf =
        torch::zeros({m_sf, kb_dim}, input.options().dtype(torch::kByte));

    const int total_groups = m * kb_dim;
    const int block_dim = 128;
    const int grid_dim = max((total_groups + block_dim - 1) / block_dim, 1);
    void* quant_data;
    if (IsFp4(quant_mode)) {
        quant_data = static_cast<void*>(output_quant.data_ptr<uint8_t>());
    } else {
        quant_data =
            static_cast<void*>(output_quant.data_ptr<c10::Float8_e4m3fn>());
    }
    auto* sf_data = output_sf.data_ptr<uint8_t>();

    const cudaStream_t stream = GetCurrentCUDAStream(input);
    if (input.scalar_type() == torch::kBFloat16) {
        const auto* input_ptr = reinterpret_cast<const __nv_bfloat16*>(
            input.mutable_data_ptr<c10::BFloat16>());
        LaunchGroupWiseQuantForMode<__nv_bfloat16>(
            input_ptr, quant_data, sf_data, m, k, kb_dim, quant_mode,
            use_swizzle, grid_dim, block_dim, stream, use_pdl);
    } else {
        const auto* input_ptr = reinterpret_cast<const __half*>(
            input.mutable_data_ptr<c10::Half>());
        LaunchGroupWiseQuantForMode<__half>(
            input_ptr, quant_data, sf_data, m, k, kb_dim, quant_mode,
            use_swizzle, grid_dim, block_dim, stream, use_pdl);
    }

    cudaError_t cuda_status = cudaGetLastError();
    TORCH_CHECK(cuda_status == cudaSuccess,
                "CUDA error: ", cudaGetErrorString(cuda_status));

    return {output_quant, output_sf};
}

}  // namespace halops
