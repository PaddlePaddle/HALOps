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
#include <stdint.h>

#include <algorithm>
#include <vector>

#include "kernel/gemm/gemm_utils.h"

namespace halops {

// Local AlignedVector / Load / Store: helper.h is not available in this target.
template <typename T, int Size>
struct alignas(sizeof(T) * Size) AlignedVector {
    T val[Size];
    __host__ __device__ T& operator[](int i) { return val[i]; }
    __host__ __device__ const T& operator[](int i) const { return val[i]; }
};

template <typename T, int Size>
__device__ __forceinline__ void Load(const T* addr,
                                     AlignedVector<T, Size>* vec) {
    *vec = *reinterpret_cast<const AlignedVector<T, Size>*>(addr);
}

template <typename T, int Size>
__device__ __forceinline__ void Store(const AlignedVector<T, Size>& vec,
                                      T* addr) {
    *reinterpret_cast<AlignedVector<T, Size>*>(addr) = vec;
}

constexpr float kEpsilon = 1e-10f;
constexpr float kFP8Max = 448.f;
constexpr float kFP4Max = 6.f;

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

__host__ __device__ __forceinline__ int64_t ceil_div_i64(int64_t x, int64_t y) {
    return (x + y - 1) / y;
}

// Fused SwiGLU + quantization with scales written directly in CUTLASS SfAtom
// layout (out_scale [E*M, kb_dim] uint8).
//   QUANT_GROUP_SIZE=16: FP4 E2M1, two values per byte, UE4M3 scale per 16.
//   QUANT_GROUP_SIZE=32: FP8 E4M3, UE8M0 scale per 32.
// input is [E*M, 2H] with gate = [:, :H], up = [:, H:]. Rows past
// token_nums_per_expert[e] within each expert are skipped.
template <typename T, int QUANT_GROUP_SIZE>
__global__ void FusedSwigluQuantKernel(
    const T* __restrict__ input, const int* __restrict__ token_nums_per_expert,
    void* __restrict__ out_quant, uint8_t* __restrict__ out_scale,
    int64_t group_num,
    int64_t group_size,  // rows per expert
    int64_t hidden_size, int64_t kb_dim) {
    // SfAtom tile: 4 quant groups per K tile
    constexpr int SFATOM_TILE = 4 * QUANT_GROUP_SIZE;  // 64 (FP4) or 128 (FP8)
    constexpr int VEC_BF16 = 8;                        // 128-bit bf16 load
    constexpr int VEC_FP8 = 16;                        // 128-bit fp8 store

    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int num_warps = blockDim.x >> 5;

    // Prefix sum of valid tokens per expert.
    extern __shared__ int smem[];
    int* smem_cumsum = smem;

    if (tid == 0) {
        smem_cumsum[0] = 0;
        for (int i = 0; i < group_num; ++i) {
            smem_cumsum[i + 1] = smem_cumsum[i] + token_nums_per_expert[i];
        }
    }
    __syncthreads();

    int total_tokens = smem_cumsum[group_num];

    using VecBF16 = AlignedVector<T, VEC_BF16>;
    using VecFP8_16 = AlignedVector<c10::Float8_e4m3fn, VEC_FP8>;
    using VecFP8_8 = AlignedVector<c10::Float8_e4m3fn, VEC_BF16>;

    // Cached expert mapping for grid-stride locality.
    int cached_expert = -1;
    int cached_cumsum_lo = 0;
    int cached_cumsum_hi = 0;

    for (int64_t block_id = static_cast<int64_t>(blockIdx.x);
         block_id < total_tokens; block_id += gridDim.x) {
        // Map block_id to (expert, token_in_expert).
        int64_t expert, token_in_expert;
        if (lane == 0) {
            int bid = static_cast<int>(block_id);
            if (bid >= cached_cumsum_lo && bid < cached_cumsum_hi) {
                expert = cached_expert;
                token_in_expert = bid - cached_cumsum_lo;
            } else {
                int lo = 0, hi = static_cast<int>(group_num) + 1;
                while (lo < hi) {
                    int mid = (lo + hi) >> 1;
                    if (smem_cumsum[mid] <= bid)
                        lo = mid + 1;
                    else
                        hi = mid;
                }
                expert = static_cast<int64_t>(lo - 1);
                token_in_expert =
                    bid - static_cast<int64_t>(smem_cumsum[lo - 1]);
                cached_expert = static_cast<int>(expert);
                cached_cumsum_lo = smem_cumsum[lo - 1];
                cached_cumsum_hi = smem_cumsum[lo];
            }
        }
        expert = __shfl_sync(0xffffffff, expert, 0);
        token_in_expert = __shfl_sync(0xffffffff, token_in_expert, 0);
        cached_expert = __shfl_sync(0xffffffff, cached_expert, 0);
        cached_cumsum_lo = __shfl_sync(0xffffffff, cached_cumsum_lo, 0);
        cached_cumsum_hi = __shfl_sync(0xffffffff, cached_cumsum_hi, 0);

        int64_t token = expert * group_size + token_in_expert;
        const T* in = input + token * hidden_size * 2;

        if constexpr (QUANT_GROUP_SIZE == 16) {
            // FP4: each thread handles one 16-element group.
            constexpr int VEC_8 = 8;
            using VecBF16_8 = AlignedVector<T, VEC_8>;

            int64_t num_groups = hidden_size / QUANT_GROUP_SIZE;
            int num_k_tiles = kb_dim / 4;

            // MN part of SfAtom offset
            int n_tile = static_cast<int>(token_in_expert) / 128;
            int pos = static_cast<int>(token_in_expert) % 128;
            int mn_i = pos % 32;
            int mn_j = pos / 32;
            int64_t sfatom_mn_base =
                expert * group_size * kb_dim +
                static_cast<int64_t>(n_tile) * num_k_tiles * 512 + mn_i * 16 +
                mn_j * 4;

            uint8_t* out_ptr = reinterpret_cast<uint8_t*>(out_quant) +
                               token * (hidden_size / 2);

            for (int64_t g = tid; g < num_groups; g += blockDim.x) {
                int64_t base = g * QUANT_GROUP_SIZE;

                VecBF16_8 x1_vec, x2_vec;
                Load(in + base, &x1_vec);
                Load(in + hidden_size + base, &x2_vec);

                float vals[QUANT_GROUP_SIZE];
                float amax = 0.f;
#pragma unroll
                for (int i = 0; i < VEC_8; ++i) {
                    float x1 = static_cast<float>(x1_vec[i]);
                    float x2 = static_cast<float>(x2_vec[i]);
                    float y = x2 * x1 * __frcp_rn(1.f + expf(-x1));
                    vals[i] = y;
                    amax = fmaxf(amax, fabsf(y));
                }

                Load(in + base + VEC_8, &x1_vec);
                Load(in + hidden_size + base + VEC_8, &x2_vec);
#pragma unroll
                for (int i = 0; i < VEC_8; ++i) {
                    float x1 = static_cast<float>(x1_vec[i]);
                    float x2 = static_cast<float>(x2_vec[i]);
                    float y = x2 * x1 * __frcp_rn(1.f + expf(-x1));
                    vals[VEC_8 + i] = y;
                    amax = fmaxf(amax, fabsf(y));
                }

                amax = fmaxf(amax, kEpsilon);

                float scale = amax / kFP4Max;
                uint8_t scale_byte = float32_to_ue4m3(scale);
                if (scale_byte == 0 && amax > 0.f) scale_byte = 0x01;
                float scale_decoded = ue4m3_to_float32(scale_byte);
                float inv_scale = 1.f / scale_decoded;

                uint8_t fp4_codes[QUANT_GROUP_SIZE];
#pragma unroll
                for (int i = 0; i < QUANT_GROUP_SIZE; ++i) {
                    float scaled = vals[i] * inv_scale;
                    scaled = fminf(fmaxf(scaled, -kFP4Max), kFP4Max);
                    fp4_codes[i] = quantize_to_fp4_code(scaled);
                }
#pragma unroll
                for (int i = 0; i < QUANT_GROUP_SIZE; i += 2) {
                    out_ptr[base / 2 + i / 2] =
                        (fp4_codes[i + 1] << 4) | fp4_codes[i];
                }

                int k_tile = g / 4;
                int sf_l = g % 4;
                int64_t sf_offset = sfatom_mn_base + k_tile * 512 + sf_l;
                out_scale[sf_offset] = scale_byte;
            }

        } else {
            auto* out = reinterpret_cast<c10::Float8_e4m3fn*>(out_quant) +
                        token * hidden_size;

            const float inv_fp8_max = 1.f / kFP8Max;
            const int64_t num_iters = hidden_size / SFATOM_TILE;
            const int64_t hidden_size_scale = num_iters;
            const int64_t n_scale = ceil_div_i64(hidden_size_scale, 4);
            const int64_t hs4_sfatom = n_scale * 4;

            int n_tile = static_cast<int>(token_in_expert) / 128;
            int pos = static_cast<int>(token_in_expert) % 128;
            int lane_sfatom = 4 * (pos % 32) + pos / 32;
            int64_t sfatom_token_base =
                expert * group_size * kb_dim +
                static_cast<int64_t>(n_tile) * hs4_sfatom * 512 +
                lane_sfatom * 4;

            // Each lane handles 16 elements; 8 lanes cover one 128-element
            // SfAtom tile, so a warp covers 4 tiles per iteration.
            for (int64_t iter_quad = warp; iter_quad < num_iters / 4;
                 iter_quad += num_warps) {
                int64_t iter0 = iter_quad * 4;
                int64_t base = iter0 * SFATOM_TILE + lane * VEC_FP8;

                VecBF16 x1_lo, x1_hi, x2_lo, x2_hi;
                Load(in + base, &x1_lo);
                Load(in + base + VEC_BF16, &x1_hi);
                Load(in + base + hidden_size, &x2_lo);
                Load(in + base + hidden_size + VEC_BF16, &x2_hi);

                float v[VEC_FP8];
                float my_amax = 0.f;

#pragma unroll
                for (int i = 0; i < VEC_BF16; ++i) {
                    float x1 = static_cast<float>(x1_lo[i]);
                    float x2 = static_cast<float>(x2_lo[i]);
                    float y = x2 * x1 / (1.f + expf(-x1));
                    v[i] = y;
                    my_amax = fmaxf(my_amax, fabsf(y));
                }
#pragma unroll
                for (int i = 0; i < VEC_BF16; ++i) {
                    float x1 = static_cast<float>(x1_hi[i]);
                    float x2 = static_cast<float>(x2_hi[i]);
                    float y = x2 * x1 * __frcp_rn(1.f + expf(-x1));
                    v[i + VEC_BF16] = y;
                    my_amax = fmaxf(my_amax, fabsf(y));
                }

                // Each lane pair forms one group-32.
                float pair_amax =
                    fmaxf(my_amax, __shfl_xor_sync(0xffffffff, my_amax, 1));
                pair_amax = fmaxf(pair_amax, kEpsilon);

                // UE8M0 scale per group-32.
                float raw_scale = fmaxf(pair_amax * inv_fp8_max, kEpsilon);
                uint32_t raw_bits = __float_as_int(raw_scale);
                int biased_exp = static_cast<int>((raw_bits >> 23) & 0xFFu);
                int has_frac = (raw_bits & 0x007FFFFF) ? 1 : 0;
                int ue8m0_exp = min(biased_exp + has_frac, 254);
                float inv_scale = __int_as_float((254 - ue8m0_exp) << 23);

                VecFP8_16 q_vec;
#pragma unroll
                for (int i = 0; i < VEC_FP8; ++i) {
                    q_vec[i] =
                        static_cast<c10::Float8_e4m3fn>(v[i] * inv_scale);
                }

                Store(q_vec, out + base);

                // Pack the 4 group-32 exponents of a tile into one uint32.
                // Shuffles stay outside the branch: all 32 lanes must
                // participate.
                int g1 = __shfl_sync(0xffffffff, ue8m0_exp, (lane & ~7) | 2);
                int g2 = __shfl_sync(0xffffffff, ue8m0_exp, (lane & ~7) | 4);
                int g3 = __shfl_sync(0xffffffff, ue8m0_exp, (lane & ~7) | 6);
                if ((lane & 7) == 0) {
                    int group_idx = lane >> 3;
                    uint32_t packed = static_cast<uint32_t>(ue8m0_exp) |
                                      (static_cast<uint32_t>(g1) << 8) |
                                      (static_cast<uint32_t>(g2) << 16) |
                                      (static_cast<uint32_t>(g3) << 24);
                    *reinterpret_cast<uint32_t*>(out_scale + sfatom_token_base +
                                                 (iter0 + group_idx) * 512) =
                        packed;
                }
            }

            // Remainder: 2 tiles when num_iters % 4 >= 2.
            int64_t quad_done = (num_iters / 4) * 4;
            int64_t rem_pair_start = quad_done / 2;
            for (int64_t iter_pair = rem_pair_start + warp;
                 iter_pair < num_iters / 2; iter_pair += num_warps) {
                int64_t iter0 = iter_pair * 2;
                int64_t base = iter0 * SFATOM_TILE + lane * VEC_BF16;

                VecBF16 x1_vec, x2_vec;
                Load(in + base, &x1_vec);
                Load(in + base + hidden_size, &x2_vec);

                float v2[VEC_BF16];
                float amax0 = 0.f;
                float amax1 = 0.f;

#pragma unroll
                for (int i = 0; i < VEC_BF16; ++i) {
                    float x1 = static_cast<float>(x1_vec[i]);
                    float x2 = static_cast<float>(x2_vec[i]);
                    float y = x2 * x1 * __frcp_rn(1.f + expf(-x1));
                    v2[i] = y;
                    if (i < 4)
                        amax0 = fmaxf(amax0, fabsf(y));
                    else
                        amax1 = fmaxf(amax1, fabsf(y));
                }

                float my_amax = fmaxf(amax0, amax1);
                // Group-32 amax: 4-lane reduction (4 lanes × 8 elements = 32)
                float p1 =
                    fmaxf(my_amax, __shfl_xor_sync(0xffffffff, my_amax, 1));
                float group32_amax =
                    fmaxf(p1, __shfl_xor_sync(0xffffffff, p1, 2));
                group32_amax = fmaxf(group32_amax, kEpsilon);

                float raw_s = fmaxf(group32_amax * inv_fp8_max, kEpsilon);
                uint32_t bits_s = __float_as_int(raw_s);
                int bexp_s = static_cast<int>((bits_s >> 23) & 0xFFu);
                int hfrac_s = (bits_s & 0x007FFFFF) ? 1 : 0;
                int ue_exp = min(bexp_s + hfrac_s, 254);
                float inv_s = __int_as_float((254 - ue_exp) << 23);

                VecFP8_8 q_vec2;
#pragma unroll
                for (int i = 0; i < VEC_BF16; ++i) {
                    q_vec2[i] = static_cast<c10::Float8_e4m3fn>(v2[i] * inv_s);
                }
                Store(q_vec2, out + base);

                // Lanes 0-15 hold iter0, lanes 16-31 hold iter0+1; 4 lanes
                // per group-32. Shuffles stay outside the branch: all 32 lanes
                // must participate.
                int half_base = (lane < 16) ? 0 : 16;
                int e1 = __shfl_sync(0xffffffff, ue_exp, half_base + 4);
                int e2 = __shfl_sync(0xffffffff, ue_exp, half_base + 8);
                int e3 = __shfl_sync(0xffffffff, ue_exp, half_base + 12);
                if ((lane & 0xF) == 0) {
                    uint32_t packed = static_cast<uint32_t>(ue_exp) |
                                      (static_cast<uint32_t>(e1) << 8) |
                                      (static_cast<uint32_t>(e2) << 16) |
                                      (static_cast<uint32_t>(e3) << 24);
                    int64_t my_iter = (lane < 16) ? iter0 : (iter0 + 1);
                    *reinterpret_cast<uint32_t*>(out_scale + sfatom_token_base +
                                                 my_iter * 512) = packed;
                }
            }

            // Remainder: last tile when num_iters is odd.
            if (num_iters & 1) {
                int64_t iter = num_iters - 1;
                if (warp == (num_iters / 2) % num_warps ||
                    num_iters / 2 < static_cast<int64_t>(num_warps)) {
                    using VecBF16_4 = AlignedVector<T, 4>;
                    using VecFP8_4 = AlignedVector<c10::Float8_e4m3fn, 4>;
                    VecBF16_4 rx1, rx2;
                    VecFP8_4 rq;

                    int64_t rbase = iter * SFATOM_TILE + lane * 4;
                    if (rbase < hidden_size) {
                        Load(in + rbase, &rx1);
                        Load(in + rbase + hidden_size, &rx2);

                        float rv[4];
                        float ramax = 0.f;
#pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            float x1 = static_cast<float>(rx1[i]);
                            float x2 = static_cast<float>(rx2[i]);
                            float y = x2 * x1 * __frcp_rn(1.f + expf(-x1));
                            rv[i] = y;
                            ramax = fmaxf(ramax, fabsf(y));
                        }

#pragma unroll
                        for (int offset = 4; offset > 0; offset >>= 1)
                            ramax = fmaxf(
                                ramax,
                                __shfl_xor_sync(0xffffffff, ramax, offset));
                        ramax = fmaxf(ramax, kEpsilon);

                        float raw_r = fmaxf(ramax * inv_fp8_max, kEpsilon);
                        uint32_t bits_r = __float_as_int(raw_r);
                        int bexp_r = static_cast<int>((bits_r >> 23) & 0xFFu);
                        int hfrac_r = (bits_r & 0x007FFFFF) ? 1 : 0;
                        int ue_exp_r = min(bexp_r + hfrac_r, 254);
                        float inv_sr = __int_as_float((254 - ue_exp_r) << 23);
#pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            rq[i] =
                                static_cast<c10::Float8_e4m3fn>(rv[i] * inv_sr);
                        }

                        // Lanes 0-7 / 8-15 / 16-23 / 24-31 hold group-32 0-3.
                        // Shuffles stay outside the branch: all 32 lanes must
                        // participate.
                        int s1 = __shfl_sync(0xffffffff, ue_exp_r, 8);
                        int s2 = __shfl_sync(0xffffffff, ue_exp_r, 16);
                        int s3 = __shfl_sync(0xffffffff, ue_exp_r, 24);
                        if (lane == 0) {
                            uint32_t packed =
                                static_cast<uint32_t>(ue_exp_r) |
                                (static_cast<uint32_t>(s1) << 8) |
                                (static_cast<uint32_t>(s2) << 16) |
                                (static_cast<uint32_t>(s3) << 24);
                            *reinterpret_cast<uint32_t*>(
                                out_scale + sfatom_token_base + iter * 512) =
                                packed;
                        }

                        Store(rq, out + rbase);
                    }
                }
            }
        }
    }
}

static inline int64_t compute_kb_dim(int64_t hidden_size,
                                     int quant_group_size) {
    int64_t sfatom_tile = 4 * quant_group_size;
    int64_t num_quant_groups = hidden_size / quant_group_size;
    if (quant_group_size == 16) {
        // FP4: one scale byte per group-16.
        return num_quant_groups;
    }
    int64_t hidden_size_scale = hidden_size / sfatom_tile;
    int64_t n_scale = ceil_div_i64(hidden_size_scale, 4);
    return n_scale * 16;
}

std::vector<torch::Tensor> FusedSwigluQuant(
    const torch::Tensor& input, const torch::Tensor& token_nums_per_expert,
    const bool is_fp4) {
    const CudaDeviceGuard device_guard(input.device());
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16,
                "input dtype must be bfloat16");
    TORCH_CHECK(token_nums_per_expert.scalar_type() == torch::kInt32,
                "token_nums_per_expert dtype must be int32, got ",
                token_nums_per_expert.scalar_type());
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(token_nums_per_expert.is_cuda(),
                "token_nums_per_expert must be a CUDA tensor");
    TORCH_CHECK(input.device() == token_nums_per_expert.device(),
                "input and token_nums_per_expert must be on the same device, "
                "got ",
                input.device(), " and ", token_nums_per_expert.device());
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(token_nums_per_expert.is_contiguous(),
                "token_nums_per_expert must be contiguous");

    const int quant_group_size = is_fp4 ? 16 : 32;
    const int64_t group_num = token_nums_per_expert.size(0);
    const int64_t total_rows = input.size(0);  // E * M
    TORCH_CHECK(input.dim() == 2,
                "input must be a 2D tensor of shape [E * M, 2 * hidden_size]");
    TORCH_CHECK(token_nums_per_expert.dim() == 1,
                "token_nums_per_expert must be a 1D tensor");
    TORCH_CHECK(group_num > 0, "token_nums_per_expert must not be empty");
    TORCH_CHECK(total_rows % group_num == 0,
                "input rows must be divisible by expert count, got rows=",
                total_rows, ", expert count=", group_num);
    TORCH_CHECK(input.size(1) % 2 == 0,
                "input hidden dimension must be even, got ", input.size(1));
    const int64_t expert_group_size = total_rows / group_num;
    const int64_t hidden_size = input.size(1) / 2;
    TORCH_CHECK(
        hidden_size % quant_group_size == 0,
        "hidden_size must be divisible by quant_group_size, got hidden_size=",
        hidden_size, ", quant_group_size=", quant_group_size);
    if (!is_fp4) {
        // FP8 kernel walks hidden in 128-element SfAtom tiles; a tail would
        // be skipped.
        TORCH_CHECK(hidden_size % 128 == 0,
                    "FP8 path requires hidden_size divisible by 128, got ",
                    hidden_size);
    }
    const int64_t kb_dim = compute_kb_dim(hidden_size, quant_group_size);
    TORCH_CHECK(kb_dim % 4 == 0,
                "out_scale kb_dim must be divisible by 4, got ", kb_dim);
    const int64_t token_num = group_num * expert_group_size;

    // Prefix-sum smem is (group_num + 1) ints; keep it within 48KB.
    const int64_t smem_required =
        static_cast<int64_t>(group_num + 1) * sizeof(int);
    TORCH_CHECK(smem_required <= 48 * 1024,
                "group_num too large: required smem=", smem_required,
                " bytes exceeds 48KB limit (group_num=", group_num, ")");

    const cudaStream_t stream = GetCurrentCUDAStream(input);

    torch::Tensor out_quant;
    if (quant_group_size == 16) {
        out_quant = torch::empty({total_rows, hidden_size / 2},
                                 input.options().dtype(torch::kByte));
    } else {
        out_quant = torch::empty({total_rows, hidden_size},
                                 input.options().dtype(torch::kFloat8_e4m3fn));
    }

    // out_scale: [E*M, kb_dim] uint8 in SfAtom layout.
    auto out_scale =
        torch::empty({total_rows, kb_dim}, input.options().dtype(torch::kByte));

    if (token_num == 0) {
        return {out_quant, out_scale};
    }

    int sm_count = GetSmCount();
    constexpr int BLOCKS_PER_SM = 4;
    int blockx = std::min(static_cast<int64_t>(256), hidden_size / 128 * 32);
    blockx = std::max(blockx, 32);
    int gridx =
        std::min(static_cast<int64_t>(sm_count * BLOCKS_PER_SM), token_num);
    int smem_bytes = (group_num + 1) * sizeof(int);

    using InputT = c10::BFloat16;

    void* quant_data;
    if (quant_group_size == 16) {
        quant_data = static_cast<void*>(out_quant.data_ptr<uint8_t>());
    } else {
        quant_data =
            static_cast<void*>(out_quant.data_ptr<c10::Float8_e4m3fn>());
    }

    if (quant_group_size == 16) {
        FusedSwigluQuantKernel<InputT, 16>
            <<<gridx, blockx, smem_bytes, stream>>>(
                input.data_ptr<InputT>(), token_nums_per_expert.data_ptr<int>(),
                quant_data, out_scale.data_ptr<uint8_t>(), group_num,
                expert_group_size, hidden_size, kb_dim);
    } else {
        FusedSwigluQuantKernel<InputT, 32>
            <<<gridx, blockx, smem_bytes, stream>>>(
                input.data_ptr<InputT>(), token_nums_per_expert.data_ptr<int>(),
                quant_data, out_scale.data_ptr<uint8_t>(), group_num,
                expert_group_size, hidden_size, kb_dim);
    }

    HALOPS_CUDA_CHECK(cudaGetLastError());
    return {out_quant, out_scale};
}

}  // namespace halops
