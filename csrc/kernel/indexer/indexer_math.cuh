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

#include <cuda_fp8.h>

#include "indexer_config.cuh"
#include "indexer_utils.cuh"

namespace halops::indexer {

CUTE_DEVICE uint8_t to_e4m3(float x) {
    return __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
}

CUTE_DEVICE uint32_t pack_e4m3x4(float a, float b, float c, float d,
                                 float inverse_scale) {
    const float2 inverse = make_float2(inverse_scale, inverse_scale);
    const uint32_t low = __nv_cvt_float2_to_fp8x2(
        __fmul2_rn(make_float2(a, b), inverse), __NV_SATFINITE, __NV_E4M3);
    const uint32_t high = __nv_cvt_float2_to_fp8x2(
        __fmul2_rn(make_float2(c, d), inverse), __NV_SATFINITE, __NV_E4M3);
    return low | (high << 16);
}

CUTE_DEVICE float fp8_row_scale(float amax) {
    return fmaxf(amax, IndexerDimensions::FP8_AMAX_FLOOR) /
           IndexerDimensions::FP8_MAX;
}

CUTE_DEVICE void warp_hadamard_128(float (&v)[4], int lane) {
    const float a = v[0] + v[1], b = v[0] - v[1];
    const float c = v[2] + v[3], d = v[2] - v[3];
    v[0] = a + c;
    v[1] = b + d;
    v[2] = a - c;
    v[3] = b - d;

    CUTLASS_PRAGMA_UNROLL
    for (int m = 1; m <= 16; m <<= 1) {
        const float sign = (lane & m) ? -1.f : 1.f;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < 4; i += 2) {
            const float2 peer =
                make_float2(__shfl_xor_sync(0xffffffffu, v[i], m),
                            __shfl_xor_sync(0xffffffffu, v[i + 1], m));
            const float2 result = __ffma2_rn(make_float2(sign, sign),
                                             make_float2(v[i], v[i + 1]), peer);
            v[i] = result.x;
            v[i + 1] = result.y;
        }
    }
}

template <int N, class V>
CUTE_DEVICE void thread_hadamard(V&& v) {
    static_assert(N > 1 && (N & (N - 1)) == 0, "N must be a power of two");
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; i += 2) {
        const float a = v[i], b = v[i + 1];
        v[i] = a + b;
        v[i + 1] = a - b;
    }
    CUTLASS_PRAGMA_UNROLL
    for (int d = 2; d < N; d <<= 1) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < N; i += 2) {
            if ((i & d) == 0) {
                const float2 a = make_float2(v[i], v[i + 1]);
                const float2 b = make_float2(v[i | d], v[(i | d) + 1]);
                const float2 sum = __fadd2_rn(a, b);
                const float2 diff = __fadd2_rn(a, make_float2(-b.x, -b.y));
                v[i] = sum.x;
                v[i + 1] = sum.y;
                v[i | d] = diff.x;
                v[(i | d) + 1] = diff.y;
            }
        }
    }
}

template <int COUNT, class V>
CUTE_DEVICE void scale_values(V&& values, float scale) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < COUNT; i += 2) {
        const float2 pair = __fmul2_rn(make_float2(values[i], values[i + 1]),
                                       make_float2(scale, scale));
        values[i] = pair.x;
        values[i + 1] = pair.y;
    }
}

template <bool ALIGNED, bool NEOX, class V>
CUTE_DEVICE void thread_rope_impl(V&& v, const float* __restrict__ cos_sin) {
    constexpr int HALF = IndexerDimensions::ROPE_DIM / 2;  // 32 pairs
    CUTLASS_PRAGMA_UNROLL
    for (int g = 0; g < HALF / 4; ++g) {
        float4 c4, s4;
        if constexpr (ALIGNED) {
            c4 = *reinterpret_cast<const float4*>(cos_sin + g * 4);
            s4 = *reinterpret_cast<const float4*>(cos_sin + HALF + g * 4);
        } else {
            const float2 c01 =
                *reinterpret_cast<const float2*>(cos_sin + g * 4);
            const float2 c23 =
                *reinterpret_cast<const float2*>(cos_sin + g * 4 + 2);
            const float2 s01 =
                *reinterpret_cast<const float2*>(cos_sin + HALF + g * 4);
            const float2 s23 =
                *reinterpret_cast<const float2*>(cos_sin + HALF + g * 4 + 2);
            c4 = make_float4(c01.x, c01.y, c23.x, c23.y);
            s4 = make_float4(s01.x, s01.y, s23.x, s23.y);
        }
        const float cosines[4] = {c4.x, c4.y, c4.z, c4.w};
        const float sines[4] = {s4.x, s4.y, s4.z, s4.w};
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < 4; j += 2) {
            const int pair = g * 4 + j;
            const int first = NEOX ? pair : pair * 2;
            const int second = NEOX ? pair + HALF : first + 1;
            constexpr int NEXT = NEOX ? 1 : 2;
            const float2 a = make_float2(v[first], v[first + NEXT]);
            const float2 b = make_float2(v[second], v[second + NEXT]);
            const float2 cosine = make_float2(cosines[j], cosines[j + 1]);
            const float2 sine = make_float2(sines[j], sines[j + 1]);
            const float2 left = __ffma2_rn(
                a, cosine, __fmul2_rn(make_float2(-b.x, -b.y), sine));
            const float2 right = __ffma2_rn(a, sine, __fmul2_rn(b, cosine));
            v[first] = left.x;
            v[first + NEXT] = left.y;
            v[second] = right.x;
            v[second + NEXT] = right.y;
        }
    }
}

template <bool ALIGNED = false, class V>
CUTE_DEVICE void thread_rope(V&& v, const float* __restrict__ cos_sin,
                             bool is_neox_style) {
    // Compile-time indices keep the accumulator in registers.
    if (is_neox_style)
        thread_rope_impl<ALIGNED, true>(v, cos_sin);
    else
        thread_rope_impl<ALIGNED, false>(v, cos_sin);
}

// A subgroup owns one head; NeoX partners cross lanes at dimension offset 32.
template <int LANES>
CUTE_DEVICE void group_rope(
    float (&values)[IndexerDimensions::HEAD_DIM / LANES], int group_lane,
    const float* __restrict__ cos_sin, bool is_neox_style) {
    constexpr int VALUES = IndexerDimensions::HEAD_DIM / LANES;
    constexpr int HALF = IndexerDimensions::ROPE_DIM / 2;
    const unsigned mask = __activemask();
    if (is_neox_style) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < VALUES; ++i) {
            // All participating lanes shuffle, including the non-rotary half.
            const float peer =
                __shfl_xor_sync(mask, values[i], HALF / VALUES, LANES);
            const int dim = group_lane * VALUES + i;
            if (dim < IndexerDimensions::ROPE_DIM) {
                const float cosine = cos_sin[dim % HALF];
                const float sine = cos_sin[HALF + dim % HALF];
                values[i] = dim < HALF ? values[i] * cosine - peer * sine
                                       : peer * sine + values[i] * cosine;
            }
        }
    } else if (group_lane * VALUES < IndexerDimensions::ROPE_DIM) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < VALUES; i += 2) {
            const int pair = (group_lane * VALUES + i) / 2;
            const float a = values[i], b = values[i + 1];
            values[i] = a * cos_sin[pair] - b * cos_sin[HALF + pair];
            values[i + 1] = a * cos_sin[HALF + pair] + b * cos_sin[pair];
        }
    }
}

// Match cvt.rn.e2m1: E2M1 midpoint ties select the even FP4 code.
CUTE_DEVICE uint8_t to_e2m1(float value) {
    const float magnitude = fabsf(value);
    const uint8_t code = (magnitude > 0.25f) + (magnitude >= 0.75f) +
                         (magnitude > 1.25f) + (magnitude >= 1.75f) +
                         (magnitude > 2.5f) + (magnitude >= 3.5f) +
                         (magnitude > 5.0f);
    return code | (value < 0.f ? 8 : 0);
}

CUTE_DEVICE uint8_t pack_e2m1x2(float first, float second) {
    // RN conversion handles midpoint ties; the additions canonicalize signed
    // zero. Even elements occupy the low nibble.
    uint16_t packed;
    asm("{ .reg .b8 packed_byte; .reg .f32 a, b;"
        "add.rn.f32 a, %1, 0f00000000;"
        "add.rn.f32 b, %2, 0f00000000;"
        "cvt.rn.satfinite.e2m1x2.f32 packed_byte, b, a;"
        "mov.b16 %0, {packed_byte, 0}; }"
        : "=h"(packed)
        : "f"(first), "f"(second));
    return static_cast<uint8_t>(packed);
}

CUTE_DEVICE uint8_t pack_scaled_e2m1x2(float first, float second,
                                       float inverse) {
    const float2 pair =
        __fmul2_rn(make_float2(first, second), make_float2(inverse, inverse));
    return pack_e2m1x2(pair.x, pair.y);
}

CUTE_DEVICE uint8_t mxfp4_scale(float amax, float& inverse_scale) {
    // Select the scale exponent exactly at power-of-two boundaries.
    const uint8_t exponent = quant_scale_exponent<2, 0x400000u>(amax);
    inverse_scale = quant_scale_inverse(exponent);
    return exponent;
}

CUTE_DEVICE uint8_t mxfp8_scale(float amax, float& inverse_scale) {
    const uint8_t exponent = quant_scale_exponent<8, 0x600000u>(amax);
    inverse_scale = quant_scale_inverse(exponent);
    return exponent;
}

// FP8 uses one row scale; MXFP4 and MXFP8 store group scales separately.
template <int LANES>
CUTE_DEVICE float group_quantize_output(
    float (&values)[IndexerDimensions::HEAD_DIM / LANES], int group_lane,
    uint8_t* __restrict__ data, uint8_t* __restrict__ scales, bool is_mx_sf,
    bool is_fp4) {
    constexpr int VALUES = IndexerDimensions::HEAD_DIM / LANES;
    constexpr int SCALE_LANES = IndexerDimensions::MX_GROUP_SIZE / VALUES;
    const unsigned mask = __activemask();
    float amax = 0.f;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < VALUES; ++i) {
        amax = fmaxf(amax, fabsf(values[i]));
    }
    if (is_fp4) {
        CUTLASS_PRAGMA_UNROLL
        for (int offset = 1; offset < SCALE_LANES; offset <<= 1)
            amax =
                fmaxf(amax, __shfl_xor_sync(mask, amax, offset, SCALE_LANES));
        float inverse_scale;
        const uint8_t scale = mxfp4_scale(amax, inverse_scale);
        if (group_lane % SCALE_LANES == 0)
            scales[group_lane / SCALE_LANES] = scale;
        if constexpr (VALUES == 4) {
            const uint16_t packed =
                pack_scaled_e2m1x2(values[0], values[1], inverse_scale) |
                (uint16_t(
                     pack_scaled_e2m1x2(values[2], values[3], inverse_scale))
                 << 8);
            *reinterpret_cast<uint16_t*>(data + group_lane * 2) = packed;
        } else {
            uint32_t words[VALUES / 8];
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < VALUES; i += 8) {
                uint32_t packed = 0;
                CUTLASS_PRAGMA_UNROLL
                for (int pair = 0; pair < 4; ++pair)
                    packed |= uint32_t(pack_scaled_e2m1x2(
                                  values[i + pair * 2],
                                  values[i + pair * 2 + 1], inverse_scale))
                              << (pair * 8);
                words[i / 8] = packed;
            }
            uint8_t* destination = data + group_lane * VALUES / 2;
            if constexpr (VALUES == 32)
                *reinterpret_cast<uint4*>(destination) =
                    make_uint4(words[0], words[1], words[2], words[3]);
            else if constexpr (VALUES == 16)
                *reinterpret_cast<uint2*>(destination) =
                    make_uint2(words[0], words[1]);
            else
                *reinterpret_cast<uint32_t*>(destination) = words[0];
        }
        return 1.f;
    }
    float scale = 1.f, inverse_scale;
    if (is_mx_sf) {
        CUTLASS_PRAGMA_UNROLL
        for (int offset = 1; offset < SCALE_LANES; offset <<= 1)
            amax =
                fmaxf(amax, __shfl_xor_sync(mask, amax, offset, SCALE_LANES));
        const uint8_t exponent = mxfp8_scale(amax, inverse_scale);
        if (group_lane % SCALE_LANES == 0)
            scales[group_lane / SCALE_LANES] = exponent;
    } else {
        CUTLASS_PRAGMA_UNROLL
        for (int offset = 1; offset < LANES; offset <<= 1)
            amax = fmaxf(amax, __shfl_xor_sync(mask, amax, offset, LANES));
        scale = fp8_row_scale(amax);
        inverse_scale = 1.f / scale;
    }
    if constexpr (VALUES >= 16) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < VALUES; i += 16) {
            uint32_t words[4];
            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < 4; ++j)
                words[j] = pack_e4m3x4(values[i + j * 4], values[i + j * 4 + 1],
                                       values[i + j * 4 + 2],
                                       values[i + j * 4 + 3], inverse_scale);
            *reinterpret_cast<uint4*>(data + group_lane * VALUES + i) =
                make_uint4(words[0], words[1], words[2], words[3]);
        }
    } else if constexpr (VALUES == 8) {
        *reinterpret_cast<uint2*>(data + group_lane * VALUES) =
            make_uint2(pack_e4m3x4(values[0], values[1], values[2], values[3],
                                   inverse_scale),
                       pack_e4m3x4(values[4], values[5], values[6], values[7],
                                   inverse_scale));
    } else {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < VALUES; i += 4)
            *reinterpret_cast<uint32_t*>(data + group_lane * VALUES + i) =
                pack_e4m3x4(values[i], values[i + 1], values[i + 2],
                            values[i + 3], inverse_scale);
    }
    if (!is_mx_sf && scales != nullptr && group_lane == 0)
        *reinterpret_cast<float*>(scales) = scale;
    return scale;
}

template <int COUNT, class V>
CUTE_DEVICE float thread_amax(const V& values, int begin = 0) {
    float maxima[4] = {};
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < COUNT; i += 4) {
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < 4; ++j)
            maxima[j] = fmaxf(maxima[j], fabsf(values[begin + i + j]));
    }
    return fmaxf(fmaxf(maxima[0], maxima[1]), fmaxf(maxima[2], maxima[3]));
}

CUTE_DEVICE void store_output_words8(uint8_t* data, const uint32_t (&words)[8],
                                     bool use_256bit_stores) {
    if (use_256bit_stores) {
        const uint64_t a = uint64_t(words[0]) | (uint64_t(words[1]) << 32);
        const uint64_t b = uint64_t(words[2]) | (uint64_t(words[3]) << 32);
        const uint64_t c = uint64_t(words[4]) | (uint64_t(words[5]) << 32);
        const uint64_t d = uint64_t(words[6]) | (uint64_t(words[7]) << 32);
        asm volatile("st.global.L1::no_allocate.v4.b64 [%0], {%1, %2, %3, %4};"
                     :
                     : "l"(data), "l"(a), "l"(b), "l"(c), "l"(d)
                     : "memory");
    } else {
        *reinterpret_cast<uint4*>(data) =
            make_uint4(words[0], words[1], words[2], words[3]);
        *reinterpret_cast<uint4*>(data + 16) =
            make_uint4(words[4], words[5], words[6], words[7]);
    }
}

CUTE_DEVICE void store_output_scales(uint8_t* scales, uint32_t packed) {
    if ((reinterpret_cast<uintptr_t>(scales) & 3) == 0) {
        *reinterpret_cast<uint32_t*>(scales) = packed;
    } else {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < 4; ++i) scales[i] = uint8_t(packed >> (i * 8));
    }
}

template <class V>
CUTE_DEVICE float thread_quantize_output(V&& values, uint8_t* data,
                                         uint8_t* scales, bool is_mx_sf,
                                         bool is_fp4, bool use_256bit_stores) {
    if (is_fp4) {
        uint32_t scale_word = 0;
        CUTLASS_PRAGMA_UNROLL
        for (int pair_group = 0; pair_group < 2; ++pair_group) {
            uint32_t words[8];
            CUTLASS_PRAGMA_UNROLL
            for (int half = 0; half < 2; ++half) {
                const int group = pair_group * 2 + half;
                const float amax =
                    thread_amax<IndexerDimensions::MX_GROUP_SIZE>(
                        values, group * IndexerDimensions::MX_GROUP_SIZE);
                float inverse_scale;
                scale_word |= uint32_t(mxfp4_scale(amax, inverse_scale))
                              << (group * 8);
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < IndexerDimensions::MX_GROUP_SIZE; i += 8) {
                    uint32_t packed = 0;
                    CUTLASS_PRAGMA_UNROLL
                    for (int pair = 0; pair < 4; ++pair)
                        packed |= uint32_t(pack_scaled_e2m1x2(
                                      values[group * 32 + i + pair * 2],
                                      values[group * 32 + i + pair * 2 + 1],
                                      inverse_scale))
                                  << (pair * 8);
                    words[half * 4 + i / 8] = packed;
                }
            }
            store_output_words8(data + pair_group * 32, words,
                                use_256bit_stores);
        }
        store_output_scales(scales, scale_word);
        return 1.f;
    }
    if (is_mx_sf) {
        uint32_t scale_word = 0;
        CUTLASS_PRAGMA_UNROLL
        for (int group = 0; group < 4; ++group) {
            const float amax = thread_amax<IndexerDimensions::MX_GROUP_SIZE>(
                values, group * IndexerDimensions::MX_GROUP_SIZE);
            float inverse_scale;
            scale_word |= uint32_t(mxfp8_scale(amax, inverse_scale))
                          << (group * 8);
            uint32_t words[8];
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < IndexerDimensions::MX_GROUP_SIZE; i += 4)
                words[i / 4] = pack_e4m3x4(
                    values[group * 32 + i], values[group * 32 + i + 1],
                    values[group * 32 + i + 2], values[group * 32 + i + 3],
                    inverse_scale);
            store_output_words8(data + group * 32, words, use_256bit_stores);
        }
        store_output_scales(scales, scale_word);
        return 1.f;
    }
    const float amax = thread_amax<IndexerDimensions::HEAD_DIM>(values);
    const float scale = fp8_row_scale(amax);
    const float inverse_scale = 1.f / scale;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < IndexerDimensions::HEAD_DIM; i += 32) {
        uint32_t packed[8];
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < 8; ++j)
            packed[j] = pack_e4m3x4(values[i + j * 4], values[i + j * 4 + 1],
                                    values[i + j * 4 + 2],
                                    values[i + j * 4 + 3], inverse_scale);
        store_output_words8(data + i, packed, use_256bit_stores);
    }
    return scale;
}

CUTE_DEVICE void wait_group_ready(const PartialBarrier& bar) {
    if (bar.try_wait_warp(0)) return;
    do {
        __nanosleep(64);
    } while (!bar.try_wait_warp(0));
}

// One active lane acquires the weights, then synchronizes the participating
// lanes.
CUTE_DEVICE void wait_group_ready_active(const PartialBarrier& bar) {
    const unsigned active = __activemask();
    if ((int)(threadIdx.x & 31) == __ffs(active) - 1)
        while (!bar.try_wait(0)) __nanosleep(64);
    __syncwarp(active);
}

struct KCacheSlot {
    uint8_t* __restrict__ data;
    uint8_t* __restrict__ scale;
};

CUTE_DEVICE KCacheSlot k_cache_slot(uint8_t* __restrict__ pool, int row_bytes,
                                    int page_size, int page_shift,
                                    int cache_slot, bool is_fp4) {
    const int page = cache_slot >> page_shift;
    const int slot = cache_slot & (page_size - 1);
    uint8_t* page_start = pool + (int64_t)page * row_bytes;
    const int data_bytes =
        is_fp4 ? IndexerDimensions::HEAD_DIM / 2 : IndexerDimensions::HEAD_DIM;
    return KCacheSlot{page_start + (int64_t)slot * data_bytes,
                      page_start + (int64_t)page_size * data_bytes +
                          (int64_t)slot * IndexerDimensions::CACHE_SCALE_BYTES};
}

}  // namespace halops::indexer
