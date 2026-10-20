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

#include <cuda_fp16.h>
#include <stdint.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cstddef>
#include <fstream>
#include <iostream>
#include <utility>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#    include <cuda_bf16.h>
#endif

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include <cute/arch/cluster_sm90.hpp>  // For cute::elect_one_sync()
#include <cute/tensor.hpp>

namespace halops {

constexpr int kDequantScalePackSize = 4;
constexpr float kFP8Max = 448.0f;
constexpr float kFP8MaxRcp = 1.0f / kFP8Max;

template <int N, typename Func>
__forceinline__ __device__ void range_constexpr(Func &&f) {
    [&]<int... Is>(std::integer_sequence<int, Is...>) {
        (f(std::integral_constant<int, Is>{}), ...);
    }(std::make_integer_sequence<int, N>{});
}

// Whether this sample takes part in decode attention; samples with encoder
// tokens in a mixed batch are prefilling.
__forceinline__ __device__ bool is_decoder_batch(
    const int *__restrict__ seq_len_encoder,
    const int *__restrict__ seq_len_decoder, const int batch) {
    return (seq_len_encoder[batch] == 0) & (seq_len_decoder[batch] > 0);
}

template <int kHeadDim>
auto get_gmem_layout(int token_num, int head_num) {
    return cute::make_layout(
        cute::make_shape(token_num, kHeadDim, head_num),
        cute::make_stride(head_num * kHeadDim, cute::_1{}, kHeadDim));
}

template <int kHeadDim, int kBlockSize>
auto get_gmem_layout(const int batch_size, const int kv_head_num) {
    return cute::make_layout(
        cute::make_shape(kBlockSize, kHeadDim, kv_head_num, batch_size),
        cute::make_stride(kHeadDim, cute::_1{}, kBlockSize * kHeadDim,
                          kBlockSize * kHeadDim * kv_head_num));
}

template <int kHeadDim, int kBlockSize>
auto get_fp8_gmem_layout(const int batch_size, const int kv_head_num) {
    return cute::make_layout(
        cute::make_shape(int64_t(kBlockSize), int64_t(kHeadDim),
                         int64_t(kv_head_num), int64_t(batch_size)),
        cute::make_stride(int64_t(kHeadDim), cute::_1{},
                          int64_t(kBlockSize * kHeadDim),
                          int64_t(kBlockSize * kHeadDim * kv_head_num)));
}

template <bool zero_init = false, typename TiledMma, typename Tensor0,
          typename Tensor1, typename Tensor2>
__forceinline__ __device__ void gemm(TiledMma &tiled_mma, Tensor0 const &tCrA,
                                     Tensor1 const &tCrB, Tensor2 &tCrC) {
    if constexpr (zero_init) {
        tiled_mma.accumulate_ = cute::UMMA::ScaleOut::Zero;
    }
    constexpr int k_len = cute::size<2>(Tensor0{});
    range_constexpr<k_len>([&](auto k_block) {
        cute::gemm(tiled_mma, tCrA(cute::_, cute::_, k_block),
                   tCrB(cute::_, cute::_, k_block), tCrC);
        if constexpr (k_block == 0) {
            tiled_mma.accumulate_ = cute::UMMA::ScaleOut::One;
        }
    });
}

template <bool zero_init = false, typename TiledMma, typename Tensor0,
          typename Tensor1, typename Tensor2>
__forceinline__ __device__ void gemm(TiledMma &tiled_mma, Tensor0 const &tCrA,
                                     Tensor1 const &tCrB, Tensor2 &tCrC,
                                     cute::uint64_t &barrier, int &phase) {
    constexpr int pause = cute::size<2>(Tensor0{}) / 4 * 3;
    constexpr int copy_len = cute::size<2>(Tensor0{}) - pause;
    if constexpr (zero_init) {
        tiled_mma.accumulate_ = cute::UMMA::ScaleOut::Zero;
    }
    range_constexpr<pause>([&](auto k_block) {
        cute::gemm(tiled_mma, tCrA(cute::_, cute::_, k_block),
                   tCrB(cute::_, cute::_, k_block), tCrC);
        if constexpr (k_block == 0) {
            tiled_mma.accumulate_ = cute::UMMA::ScaleOut::One;
        }
    });
    cute::wait_barrier(barrier, phase);
    range_constexpr<copy_len>([&](auto k_block) {
        cute::gemm(tiled_mma, tCrA(cute::_, cute::_, k_block + pause),
                   tCrB(cute::_, cute::_, k_block + pause), tCrC);
    });
}

template <bool zero_init = false, typename TiledMma, typename Tensor0,
          typename Tensor1, typename Tensor2>
__forceinline__ __device__ void fp8_gemm(TiledMma &tiled_mma,
                                         Tensor0 const &tCrA,
                                         Tensor1 const &tCrB, Tensor2 &tCrC,
                                         cute::uint64_t &barrier1,
                                         const int phase) {
    if constexpr (zero_init) {
        tiled_mma.accumulate_ = cute::UMMA::ScaleOut::Zero;
    }
    cute::gemm(tiled_mma, tCrA(cute::_, cute::_, 0), tCrB(cute::_, cute::_, 0),
               tCrC);
    tiled_mma.accumulate_ = cute::UMMA::ScaleOut::One;
    cute::gemm(tiled_mma, tCrA(cute::_, cute::_, 1), tCrB(cute::_, cute::_, 1),
               tCrC);
    cute::wait_barrier(barrier1, phase);
    cute::gemm(tiled_mma, tCrA(cute::_, cute::_, 2), tCrB(cute::_, cute::_, 2),
               tCrC);
    cute::gemm(tiled_mma, tCrA(cute::_, cute::_, 3), tCrB(cute::_, cute::_, 3),
               tCrC);
}

template <class T>
CUTE_DEVICE T warp_prefix_sum(T val, int const lane_idx) {
    CUTLASS_PRAGMA_UNROLL
#pragma unroll
    for (int i = 1; i < cutlass::NumThreadsPerWarp; i <<= 1) {
        T partial_sum = __shfl_up_sync(0xffffffff, val, i);
        if (lane_idx >= i) {
            val += partial_sum;
        }
    }
    return val;
}

CUTE_DEVICE float2 add_packed_f32x2(float2 const &a, float2 const &b) {
    float2 c;
    asm volatile("add.rm.f32x2 %0, %1, %2;\n"
                 : "=l"(reinterpret_cast<uint64_t &>(c))
                 : "l"(reinterpret_cast<uint64_t const &>(a)),
                   "l"(reinterpret_cast<uint64_t const &>(b)));
    return c;
}

CUTE_DEVICE float2 sub_packed_f32x2(float2 const &a, float2 const &b) {
    float2 c;
    asm volatile("sub.rn.f32x2 %0, %1, %2;\n"
                 : "=l"(reinterpret_cast<uint64_t &>(c))
                 : "l"(reinterpret_cast<uint64_t const &>(a)),
                   "l"(reinterpret_cast<uint64_t const &>(b)));
    return c;
}

CUTE_DEVICE void mul_packed_f32x2(float2 const &a, float2 const &b, float2 &c) {
    asm volatile("mul.rn.f32x2 %0, %1, %2;\n"
                 : "=l"(reinterpret_cast<uint64_t &>(c))
                 : "l"(reinterpret_cast<uint64_t const &>(a)),
                   "l"(reinterpret_cast<uint64_t const &>(b)));
}

CUTE_DEVICE void fma_packed_f32x2(float2 const &a, float2 const &b,
                                  float2 const &c, float2 &d) {
    asm volatile("fma.rn.f32x2 %0, %1, %2, %3;\n"
                 : "=l"(reinterpret_cast<uint64_t &>(d))
                 : "l"(reinterpret_cast<uint64_t const &>(a)),
                   "l"(reinterpret_cast<uint64_t const &>(b)),
                   "l"(reinterpret_cast<uint64_t const &>(c)));
}

CUTE_DEVICE float fmax_3_input(float const a, float const b, float const c) {
    float d;
    asm volatile("max.f32 %0, %1, %2, %3;\n"
                 : "=f"(d)
                 : "f"(a), "f"(b), "f"(c));
    return d;
}

CUTE_DEVICE nv_bfloat162 bf16_abs_max(nv_bfloat162 const a,
                                      nv_bfloat162 const b) {
    nv_bfloat162 c;
    __asm__ volatile("max.xorsign.abs.bf16x2 %0, %1, %2;"
                     : "=r"(reinterpret_cast<int &>(c))
                     : "r"(reinterpret_cast<const int &>(a)),
                       "r"(reinterpret_cast<const int &>(b)));
    return c;
}

CUTE_DEVICE nv_bfloat162 bf16_abs_max_half_warp(nv_bfloat162 a) {
#pragma unroll
    for (int mask = 8; mask > 0; mask /= 2) {
        a = bf16_abs_max(a, __shfl_xor_sync(0xffffffff, a, mask));
    }
    return a;
}

CUTE_DEVICE nv_bfloat162 bf16_abs(nv_bfloat162 const a) {
    nv_bfloat162 b;
    __asm__ volatile("abs.bf16x2 %0, %1;"
                     : "=r"(reinterpret_cast<int &>(b))
                     : "r"(reinterpret_cast<const int &>(a)));
    return b;
}

CUTE_DEVICE void bf16_abs_max(const float4 &src, nv_bfloat162 &dst) {
    const nv_bfloat162 *src_ptr = reinterpret_cast<const nv_bfloat162 *>(&src);
    for (int i = 0; i < 4; i++) {
        __asm__ volatile("max.xorsign.abs.bf16x2 %0, %1, %2;"
                         : "=r"(reinterpret_cast<int &>(dst))
                         : "r"(reinterpret_cast<const int &>(dst)),
                           "r"(reinterpret_cast<const int &>(src_ptr[i])));
    }
}

struct Convert_fp8 {
    __forceinline__ __device__ uint32_t operator()(const uint32_t src1,
                                                   const uint32_t src2) {
        float2 res1 =
            __bfloat1622float2(reinterpret_cast<const nv_bfloat162 &>(src1));
        float2 res2 =
            __bfloat1622float2(reinterpret_cast<const nv_bfloat162 &>(src2));
        uint32_t dst;
        asm volatile(
            "{\n"
            ".reg .b16 lo;\n"
            ".reg .b16 hi;\n"
            "cvt.rn.satfinite.e4m3x2.f32   lo, %2, %1;\n"
            "cvt.rn.satfinite.e4m3x2.f32   hi, %4, %3;\n"
            "mov.b32 %0, {lo, hi};\n"
            "}"
            : "=r"(dst)
            : "f"(res1.x), "f"(res1.y), "f"(res2.x), "f"(res2.y));
        return dst;
    }
};

CUTE_DEVICE void quant_to_fp8(float4 src, const nv_bfloat162 scale,
                              uint2 &dst) {
    nv_bfloat162 *src_ptr = reinterpret_cast<nv_bfloat162 *>(&src);
    for (int i = 0; i < 4; i++) {
        src_ptr[i] *= scale;
    }
    dst.x = Convert_fp8()(reinterpret_cast<uint32_t *>(&src)[0],
                          reinterpret_cast<uint32_t *>(&src)[1]);
    dst.y = Convert_fp8()(reinterpret_cast<uint32_t *>(&src)[2],
                          reinterpret_cast<uint32_t *>(&src)[3]);
}

CUTE_DEVICE float combine_int_frac_ex2(float const x_rounded,
                                       float const frac_ex2) {
    float res;
    asm volatile(
        "{\n\t"
        ".reg .s32 x_rounded_i, frac_ex_i, x_rounded_e, out_i;\n\t"
        "mov.b32 x_rounded_i, %1;\n\t"
        "mov.b32 frac_ex_i, %2;\n\t"
        "shl.b32 x_rounded_e, x_rounded_i, 23;\n\t"
        "add.s32 out_i, x_rounded_e, frac_ex_i;\n\t"
        "mov.b32 %0, out_i;\n\t"
        "}\n"
        : "=f"(res)
        : "f"(x_rounded), "f"(frac_ex2));

    return res;
}

CUTE_DEVICE void ex2_emulation_2(float &x, float &y) {
    constexpr float poly_ex2_deg3[4] = {1.0f, 0.695146143436431884765625f,
                                        0.227564394474029541015625f,
                                        0.077119089663028717041015625f};
    constexpr float fp32_round_int = float((1 << 23) + (1 << 22));
    float2 const fp32_round_int2 = {fp32_round_int, fp32_round_int};
    x = fmax(x, -127.0f);
    y = fmax(y, -127.0f);
    float2 xy_clamped = {x, y};
    float2 xy_rounded = add_packed_f32x2(xy_clamped, fp32_round_int2);
    float2 xy_rounded_back = sub_packed_f32x2(xy_rounded, fp32_round_int2);
    float2 xy_frac = sub_packed_f32x2(xy_clamped, xy_rounded_back);
    float2 xy_frac_ex2 = {poly_ex2_deg3[3], poly_ex2_deg3[3]};
    range_constexpr<3>([&](auto index) {
        constexpr int i = 2 - index;
        fma_packed_f32x2(xy_frac_ex2, xy_frac,
                         {poly_ex2_deg3[i], poly_ex2_deg3[i]}, xy_frac_ex2);
    });
    x = combine_int_frac_ex2(xy_rounded.x, xy_frac_ex2.x);
    y = combine_int_frac_ex2(xy_rounded.y, xy_frac_ex2.y);
}

template <typename T>
struct PackedHalf;

template <>
struct PackedHalf<cutlass::half_t> {
    using Type = __half2;
};

template <>
struct PackedHalf<cutlass::bfloat16_t> {
    using Type = nv_bfloat162;
};

template <typename T, int PackSize>
inline __device__ void apply_rotary_embedding(float4 &src, const float *cos,
                                              const float *sin) {
    static_assert(PackSize == 8);
    T *vec = reinterpret_cast<T *>(&src);
#pragma unroll
    for (int i = 0; i < PackSize / 2; i++) {
        const float cos_inv_freq = cos[i];
        const float sin_inv_freq = sin[i];
        const float v1 = static_cast<float>(vec[2 * i]);
        const float v2 = static_cast<float>(vec[2 * i + 1]);
        vec[2 * i] = static_cast<T>(cos_inv_freq * v1 - sin_inv_freq * v2);
        vec[2 * i + 1] = static_cast<T>(sin_inv_freq * v1 + cos_inv_freq * v2);
    }
}

template <typename T, typename ReductionOp, int block_size>
__inline__ __device__ T BlockAllReduce(T val) {
    typedef cub::BlockReduce<T, block_size> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ T result_broadcast;
    T result = BlockReduce(temp_storage).Reduce(val, ReductionOp());
    if (threadIdx.x == 0) {
        result_broadcast = result;
    }
    __syncthreads();
    return result_broadcast;
}

template <typename T>
struct MaxOp {
    __device__ __forceinline__ T operator()(T const &x, T const &y) {
        return x > y ? x : y;
    }
};

template <typename T>
struct SumOp {
    __device__ __forceinline__ T operator()(T const &x, T const &y) {
        return x + y;
    }
};

template <typename CopyOp, std::size_t... Is>
CUTLASS_DEVICE static void tmem_load(uint32_t src_addr, uint32_t *arr,
                                     std::index_sequence<Is...>) {
    CopyOp::copy(src_addr, arr[Is]...);
}

template <typename CopyOp, std::size_t... Is>
CUTLASS_DEVICE static void tmem_store(uint32_t dst_addr, uint32_t *arr,
                                      std::index_sequence<Is...>) {
    CopyOp::copy(arr[Is]..., dst_addr);
}

__device__ __forceinline__ uint32_t floats4_to_fp8x4(const float x,
                                                     const float y,
                                                     const float z,
                                                     const float w) {
    uint32_t dst;
    asm volatile(
        "{\n"
        ".reg .b16 lo;\n"
        ".reg .b16 hi;\n"
        "cvt.rn.satfinite.e4m3x2.f32   lo, %2, %1;\n"
        "cvt.rn.satfinite.e4m3x2.f32   hi, %4, %3;\n"
        "mov.b32 %0, {lo, hi};\n"
        "}"
        : "=r"(dst)
        : "f"(x), "f"(y), "f"(z), "f"(w));
    return dst;
}

}  // namespace halops
