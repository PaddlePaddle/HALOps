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

#include <cute/arch/copy_sm80.hpp>
#include <cute/arch/util.hpp>

#include "common/pdl.cuh"
#include "kernel/gemm/gemm_utils.h"
#include "kernel/mla/mega_mla/vha_postmix/config.cuh"

namespace halops::mla {

// 16-byte chunk c of tile row r, XOR-swizzled so that 8 rows of one chunk
// column (an ldmatrix 8x8 matrix) hit 8 different bank groups.
template <class C>
CUTE_DEVICE uint4& vha_chunk(uint4* tile, int r, int c) {
    return tile[r * C::kChunks + (c ^ (r % kMatRows))];
}

// Quarter of item (token item / 4): rotated by the token so that a block's
// items, a grid stride apart, do not all land on the rope quarter.
template <class C>
CUTE_DEVICE int vha_quarter(int item) {
    return (item + item / C::kQuarters) % C::kQuarters;
}

// Inverse RoPE of 4 interleaved (even, odd) pairs with their (cos, sin).
template <class C>
CUTE_DEVICE uint4 vha_inverse_rope(uint4 v, float4 f0, float4 f1) {
    const float cs[C::kElems] = {f0.x, f0.y, f0.z, f0.w,
                                 f1.x, f1.y, f1.z, f1.w};
    auto* e = reinterpret_cast<__nv_bfloat162*>(&v);
    CUTLASS_PRAGMA_UNROLL
    for (int p = 0; p < C::kElems / 2; ++p) {
        const float2 x = __bfloat1622float2(e[p]);
        const float c = cs[2 * p], s = cs[2 * p + 1];
        e[p] = __floats2bfloat162_rn(
            __fadd_rn(__fmul_rn(x.x, c), __fmul_rn(x.y, s)),
            __fsub_rn(__fmul_rn(x.y, c), __fmul_rn(x.x, s)));
    }
    return v;
}

// One item is (token t, 128-column quarter of every head): the [64, 128]
// tile of o (and, for the last quarter, the position's rope row) is loaded one
// item ahead into the other of two shared buffers (the last quarter is
// inverse-RoPEd in place), multiplied by the
// [64, 64] mix on the tensor cores with fp32 accumulation, rounded to bf16 and
// quantized like group_wise_quant straight from the accumulator fragments: a
// group's 32 columns are 4 n-tiles of one quad, transposed across the quad
// into 8-byte stores. Head h of group h / hpg writes
// row (group_major ? g * T + t : t * G + g) of the [G * T, D] output.
template <class C, typename Pos>
__global__ void __launch_bounds__(C::kThreads, C::kBlocksPerSm)
    vha_postmix_quant_kernel(const __nv_bfloat16* __restrict__ o,
                             const __nv_bfloat16* __restrict__ mix,
                             const float* __restrict__ freqs,
                             const Pos* __restrict__ positions, int tokens,
                             int groups, bool group_major,
                             uint8_t* __restrict__ q,
                             uint8_t* __restrict__ sf) {
    extern __shared__ uint4 tiles[];
    const int tid = threadIdx.x;
    const int warp = tid / cutlass::NumThreadsPerWarp;
    const int lane = tid % cutlass::NumThreadsPerWarp;
    const int heads_per_group = C::kHeads / groups;
    const int d = heads_per_group * C::kHeadDim;
    const int quad = lane % kQuad;
    int head_g[2];
    int head_col[2];
    CUTLASS_PRAGMA_UNROLL
    for (int half = 0; half < 2; ++half) {
        const int h = warp * C::kMmaM + lane / kQuad + half * kMatRows;
        head_g[half] = h / heads_per_group;
        head_col[half] = h % heads_per_group * C::kHeadDim;
    }
    const int kb_dim = d / mxfp8::kGroup;

    uint32_t a[C::kKSteps][C::kARegs];
    {
        const int r = warp * C::kMmaM + lane / kQuad;
        const int c = 2 * (lane % kQuad);
        const auto at = [&](int rr, int cc) {
            return *reinterpret_cast<const uint32_t*>(mix + rr * C::kHeads +
                                                      cc);
        };
        CUTLASS_PRAGMA_UNROLL
        for (int s = 0; s < C::kKSteps; ++s) {
            const int k0 = s * C::kMmaK + c;
            a[s][0] = at(r, k0);
            a[s][1] = at(r + kMatRows, k0);
            a[s][2] = at(r, k0 + kMatRows);
            a[s][C::kARegs - 1] = at(r + kMatRows, k0 + kMatRows);
        }
    }

    const int ldsm_row =
        (lane % kMatRows + kMatRows * (lane / kMatRows % 2)) * C::kTileRowBytes;
    int ldsm_chunk[C::kNTiles / 2];
    CUTLASS_PRAGMA_UNROLL
    for (int p = 0; p < C::kNTiles / 2; ++p) {
        ldsm_chunk[p] = ((2 * p + lane / (2 * kMatRows)) ^ (lane % kMatRows)) *
                        int(sizeof(uint4));
    }

    pdl_wait();
    pdl_trigger();

    const int items = tokens * C::kQuarters;
    const auto rope_pos = [&](int item) {
        return item < items && vha_quarter<C>(item) == C::kQuarters - 1
                   ? int64_t(positions[item / C::kQuarters])
                   : int64_t(0);
    };
    const auto load = [&](int item, uint4* buf, int64_t pos) {
        if (item < items) {
            const __nv_bfloat16* src =
                o + int64_t(item / C::kQuarters) * C::kHeads * C::kHeadDim +
                vha_quarter<C>(item) * C::kCols;
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < C::kLoads; ++i) {
                const int id = i * C::kThreads + tid;
                cute::SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>::copy(
                    *reinterpret_cast<const cute::uint128_t*>(
                        src + (id / C::kChunks) * C::kHeadDim +
                        (id % C::kChunks) * C::kElems),
                    reinterpret_cast<cute::uint128_t&>(
                        vha_chunk<C>(buf, id / C::kChunks, id % C::kChunks)));
            }
            if (vha_quarter<C>(item) == C::kQuarters - 1 &&
                tid < C::kFreqChunks) {
                cute::SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>::copy(
                    reinterpret_cast<const cute::uint128_t*>(
                        freqs + pos * C::kRopeDim)[tid],
                    reinterpret_cast<cute::uint128_t&>(
                        buf[C::kTileChunks + tid]));
            }
        }
        cute::cp_async_fence();
    };

    load(blockIdx.x, tiles, rope_pos(blockIdx.x));
    int64_t next_pos = rope_pos(blockIdx.x + gridDim.x);
    int stage = 0;
    for (int item = blockIdx.x; item < items; item += gridDim.x, stage ^= 1) {
        uint4* const tile = tiles + stage * C::kStageChunks;
        const int t = item / C::kQuarters;
        const int quarter = vha_quarter<C>(item);
        load(item + gridDim.x, tiles + (stage ^ 1) * C::kStageChunks, next_pos);
        next_pos = rope_pos(item + 2 * gridDim.x);
        cute::cp_async_wait<1>();
        __syncthreads();
        if (quarter == C::kQuarters - 1) {
            const int c = C::kRopeChunk + tid % C::kRopeChunks;
            const float4* f =
                reinterpret_cast<const float4*>(tile + C::kTileChunks) +
                (c - C::kRopeChunk) * 2;
            const float4 f0 = f[0], f1 = f[1];
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < C::kRopeLoads; ++i) {
                uint4& v = vha_chunk<C>(
                    tile,
                    i * (C::kThreads / C::kRopeChunks) + tid / C::kRopeChunks,
                    c);
                v = vha_inverse_rope<C>(v, f0, f1);
            }
            __syncthreads();
        }

        int64_t out[2];
        int64_t sf_off[2];
        CUTLASS_PRAGMA_UNROLL
        for (int half = 0; half < 2; ++half) {
            const int row = group_major ? head_g[half] * tokens + t
                                        : t * groups + head_g[half];
            const int col = head_col[half] + quarter * C::kCols;
            out[half] = int64_t(row) * d + col + quad * C::kMmaN;
            sf_off[half] =
                mxfp8::sf_atom_offset(row, col / mxfp8::kGroup, kb_dim);
        }
        const uint8_t* const tile_bytes =
            reinterpret_cast<const uint8_t*>(tile);
        uint32_t sf_word[2] = {};
        CUTLASS_PRAGMA_UNROLL
        for (int nh = 0; nh < C::kNHalves; ++nh) {
            float acc[C::kHalfTiles][C::kCRegs] = {};
            CUTLASS_PRAGMA_UNROLL
            for (int s = 0; s < C::kKSteps; ++s) {
                uint32_t b[C::kHalfTiles / 2][C::kLdsmRegs];
                CUTLASS_PRAGMA_UNROLL
                for (int p = 0; p < C::kHalfTiles / 2; ++p) {
                    copy_to_regs<typename C::Ldsm>(
                        *reinterpret_cast<const cute::uint128_t*>(
                            tile_bytes + s * C::kMmaK * C::kTileRowBytes +
                            ldsm_row + ldsm_chunk[nh * C::kHalfTiles / 2 + p]),
                        b[p]);
                }
                CUTLASS_PRAGMA_UNROLL
                for (int n = 0; n < C::kHalfTiles; ++n) {
                    float* r = acc[n];
                    cute::detail::explode(
                        cute::detail::CallFMA<typename C::Mma>{}, r,
                        cute::make_int_sequence<C::kCRegs>{}, a[s],
                        cute::make_int_sequence<C::kARegs>{},
                        b[n / 2] + 2 * (n % 2),
                        cute::make_int_sequence<C::kBRegs>{}, r,
                        cute::make_int_sequence<C::kCRegs>{});
                }
            }
            CUTLASS_PRAGMA_UNROLL
            for (int gh = 0; gh < C::kHalfGroups; ++gh) {
                const int gq = nh * C::kHalfGroups + gh;
                uint32_t codes[2][C::kGroupTiles / 2];
                CUTLASS_PRAGMA_UNROLL
                for (int half = 0; half < 2; ++half) {
                    __nv_bfloat162 v[C::kGroupTiles];
                    CUTLASS_PRAGMA_UNROLL
                    for (int x = 0; x < C::kGroupTiles; ++x) {
                        const float* r =
                            acc[gh * C::kGroupTiles + x] + 2 * half;
                        v[x] = __floats2bfloat162_rn(r[0], r[1]);
                    }
                    const __nv_bfloat162 m = mxfp8::abs_max<C::kGroupTiles>(v);
                    float amax = fmaxf(__low2float(m), __high2float(m));
                    CUTLASS_PRAGMA_UNROLL
                    for (int sh = 1; sh < kQuad; sh *= 2) {
                        amax = fmaxf(amax, __shfl_xor_sync(~0u, amax, sh));
                    }
                    float inv_scale;
                    const uint32_t e = mxfp8::ue8m0_scale(
                        fmaxf(amax, mxfp8::kAmaxFloor) / mxfp8::kE4m3Max,
                        inv_scale);
                    sf_word[half] |= e << (C::kScaleBits * gq);
                    const __nv_bfloat162 sc = __float2bfloat162_rn(inv_scale);
                    CUTLASS_PRAGMA_UNROLL
                    for (int x = 0; x < C::kGroupTiles; x += 2) {
                        codes[half][x / 2] = mxfp8::e4m3x4(v[x], v[x + 1], sc);
                    }
                }
                const uint4 w = make_uint4(
                    __byte_perm(codes[0][0], codes[1][0], C::kPermLo),
                    __byte_perm(codes[0][0], codes[1][0], C::kPermHi),
                    __byte_perm(codes[0][1], codes[1][1], C::kPermLo),
                    __byte_perm(codes[0][1], codes[1][1], C::kPermHi));
                uint32_t tw[kQuad];
                quad_transpose_4x4(w, lane, tw);
                const int col = gq * mxfp8::kGroup;
                uint32_t lo[2], hi[2];
                CUTLASS_PRAGMA_UNROLL
                for (int x = 0; x < 2; ++x) {
                    lo[x] = __byte_perm(tw[2 * x], tw[2 * x + 1], C::kPermLo);
                    hi[x] = __byte_perm(tw[2 * x], tw[2 * x + 1], C::kPermHi);
                }
                *reinterpret_cast<uint2*>(q + out[0] + col) =
                    make_uint2(lo[0], lo[1]);
                *reinterpret_cast<uint2*>(q + out[1] + col) =
                    make_uint2(hi[0], hi[1]);
            }
        }
        if (quad == 0) {
            *reinterpret_cast<uint32_t*>(sf + sf_off[0]) = sf_word[0];
            *reinterpret_cast<uint32_t*>(sf + sf_off[1]) = sf_word[1];
        }
        __syncthreads();
    }

    const int rows = groups * tokens;
    const int words_per_row = kb_dim / mxfp8::kAtomCols;
    const int pad_words = (AlignSfM(rows) - rows) * words_per_row;
    for (int i = blockIdx.x * C::kThreads + tid; i < pad_words;
         i += gridDim.x * C::kThreads) {
        *reinterpret_cast<uint32_t*>(
            sf + mxfp8::sf_atom_offset(rows + i / words_per_row,
                                       i % words_per_row * mxfp8::kAtomCols,
                                       kb_dim)) = 0;
    }
}

}  // namespace halops::mla
