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

#include <cute/arch/copy_sm75.hpp>
#include <cute/arch/copy_sm80.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <optional>
#include <vector>

#include "common/mx_quant.cuh"
#include "common/pdl.cuh"
#include "kernel/gemm/gemm_utils.h"

namespace halops {

namespace {

constexpr int kVhaHeads = 64;
constexpr int kVhaHeadDim = 512;
constexpr int kVhaRopeDim = 64;
constexpr int kVhaCols = 128;
constexpr int kVhaQuarters = kVhaHeadDim / kVhaCols;
constexpr int kVhaThreads = 128;
constexpr int kVhaWarps = kVhaThreads / cutlass::NumThreadsPerWarp;
constexpr int kVhaElems = sizeof(uint4) / sizeof(__nv_bfloat16);
constexpr int kVhaChunks = kVhaCols / kVhaElems;
constexpr int kVhaRopeChunk =
    (kVhaHeadDim - kVhaRopeDim) % kVhaCols / kVhaElems;
constexpr int kVhaLoads = kVhaHeads * kVhaChunks / kVhaThreads;
constexpr int kVhaRopeChunks = kVhaRopeDim / kVhaElems;
constexpr int kVhaRopeLoads = kVhaHeads * kVhaRopeChunks / kVhaThreads;
constexpr int kVhaTileChunks = kVhaHeads * kVhaChunks;
constexpr int kVhaMmaM = 16;
constexpr int kVhaMmaN = 8;
constexpr int kVhaMmaK = 16;
constexpr int kVhaKSteps = kVhaHeads / kVhaMmaK;
constexpr int kVhaNTiles = kVhaCols / kVhaMmaN;
constexpr int kVhaBlocksPerSm = 4;
constexpr int kVhaFreqChunks = kVhaRopeDim * sizeof(float) / sizeof(uint4);
constexpr int kVhaStageChunks = kVhaTileChunks + kVhaFreqChunks;
constexpr int kVhaSmemBytes = 2 * kVhaStageChunks * sizeof(uint4);
constexpr int kVhaGroupTiles = mxfp8::kGroup / kVhaMmaN;
constexpr int kVhaNHalves = 2;
constexpr int kVhaHalfTiles = kVhaNTiles / kVhaNHalves;
constexpr int kVhaHalfGroups = kVhaHalfTiles / kVhaGroupTiles;
constexpr int kVhaRowBytes = kVhaCols * sizeof(__nv_bfloat16);

static_assert(kVhaHeads == kVhaWarps * kVhaMmaM, "one 16-head band per warp");

using VhaMma = cute::SM80_16x8x16_F32BF16BF16F32_TN;

// 16-byte chunk c of tile row r, XOR-swizzled so that 8 rows of one chunk
// column (an ldmatrix 8x8 matrix) hit 8 different bank groups.
CUTE_DEVICE uint4& vha_chunk(uint4* tile, int r, int c) {
    return tile[r * kVhaChunks + (c ^ (r % kMatRows))];
}

// Quarter of item (token item / 4): rotated by the token so that a block's
// items, a grid stride apart, do not all land on the rope quarter.
CUTE_DEVICE int vha_quarter(int item) {
    return (item + item / kVhaQuarters) % kVhaQuarters;
}

// Inverse RoPE of 4 interleaved (even, odd) pairs with their (cos, sin).
CUTE_DEVICE uint4 inverse_rope(uint4 v, float4 f0, float4 f1) {
    const float cs[kVhaElems] = {f0.x, f0.y, f0.z, f0.w,
                                 f1.x, f1.y, f1.z, f1.w};
    auto* e = reinterpret_cast<__nv_bfloat162*>(&v);
    CUTLASS_PRAGMA_UNROLL
    for (int p = 0; p < kVhaElems / 2; ++p) {
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
template <typename Pos>
__global__ void __launch_bounds__(kVhaThreads, kVhaBlocksPerSm)
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
    const int heads_per_group = kVhaHeads / groups;
    const int d = heads_per_group * kVhaHeadDim;
    const int quad = lane % kQuad;
    int head_g[2];
    int head_col[2];
    CUTLASS_PRAGMA_UNROLL
    for (int half = 0; half < 2; ++half) {
        const int h = warp * kVhaMmaM + lane / kQuad + half * kMatRows;
        head_g[half] = h / heads_per_group;
        head_col[half] = h % heads_per_group * kVhaHeadDim;
    }
    const int kb_dim = d / mxfp8::kGroup;

    uint32_t a[kVhaKSteps][4];
    {
        const int r = warp * kVhaMmaM + lane / kQuad;
        const int c = 2 * (lane % kQuad);
        const auto at = [&](int rr, int cc) {
            return *reinterpret_cast<const uint32_t*>(mix + rr * kVhaHeads +
                                                      cc);
        };
        CUTLASS_PRAGMA_UNROLL
        for (int s = 0; s < kVhaKSteps; ++s) {
            const int k0 = s * kVhaMmaK + c;
            a[s][0] = at(r, k0);
            a[s][1] = at(r + kMatRows, k0);
            a[s][2] = at(r, k0 + kMatRows);
            a[s][3] = at(r + kMatRows, k0 + kMatRows);
        }
    }

    // ldmatrix.x4.trans rows: lanes 8i..8i+7 address the 8 k rows of matrix i
    // (k + 8 for odd i, n-tile + 1 for i >= 2).
    const int ldsm_row =
        (lane % kMatRows + kMatRows * (lane / kMatRows % 2)) * kVhaRowBytes;
    int ldsm_chunk[kVhaNTiles / 2];
    CUTLASS_PRAGMA_UNROLL
    for (int p = 0; p < kVhaNTiles / 2; ++p) {
        ldsm_chunk[p] = ((2 * p + lane / (2 * kMatRows)) ^ (lane % kMatRows)) *
                        int(sizeof(uint4));
    }

    pdl_wait();
    pdl_trigger();

    const int items = tokens * kVhaQuarters;
    const auto rope_pos = [&](int item) {
        return item < items && vha_quarter(item) == kVhaQuarters - 1
                   ? int64_t(positions[item / kVhaQuarters])
                   : int64_t(0);
    };
    const auto load = [&](int item, uint4* buf, int64_t pos) {
        if (item < items) {
            const __nv_bfloat16* src =
                o + int64_t(item / kVhaQuarters) * kVhaHeads * kVhaHeadDim +
                vha_quarter(item) * kVhaCols;
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < kVhaLoads; ++i) {
                const int id = i * kVhaThreads + tid;
                cute::SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>::copy(
                    *reinterpret_cast<const cute::uint128_t*>(
                        src + (id / kVhaChunks) * kVhaHeadDim +
                        (id % kVhaChunks) * kVhaElems),
                    reinterpret_cast<cute::uint128_t&>(
                        vha_chunk(buf, id / kVhaChunks, id % kVhaChunks)));
            }
            if (vha_quarter(item) == kVhaQuarters - 1 && tid < kVhaFreqChunks) {
                cute::SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>::copy(
                    reinterpret_cast<const cute::uint128_t*>(
                        freqs + pos * kVhaRopeDim)[tid],
                    reinterpret_cast<cute::uint128_t&>(
                        buf[kVhaTileChunks + tid]));
            }
        }
        cute::cp_async_fence();
    };

    load(blockIdx.x, tiles, rope_pos(blockIdx.x));
    int64_t next_pos = rope_pos(blockIdx.x + gridDim.x);
    int stage = 0;
    for (int item = blockIdx.x; item < items; item += gridDim.x, stage ^= 1) {
        uint4* const tile = tiles + stage * kVhaStageChunks;
        const int t = item / kVhaQuarters;
        const int quarter = vha_quarter(item);
        load(item + gridDim.x, tiles + (stage ^ 1) * kVhaStageChunks, next_pos);
        next_pos = rope_pos(item + 2 * gridDim.x);
        cute::cp_async_wait<1>();
        __syncthreads();
        if (quarter == kVhaQuarters - 1) {
            const int c = kVhaRopeChunk + tid % kVhaRopeChunks;
            const float4* f =
                reinterpret_cast<const float4*>(tile + kVhaTileChunks) +
                (c - kVhaRopeChunk) * 2;
            const float4 f0 = f[0], f1 = f[1];
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < kVhaRopeLoads; ++i) {
                uint4& v = vha_chunk(
                    tile,
                    i * (kVhaThreads / kVhaRopeChunks) + tid / kVhaRopeChunks,
                    c);
                v = inverse_rope(v, f0, f1);
            }
            __syncthreads();
        }

        int64_t out[2];
        int64_t sf_off[2];
        CUTLASS_PRAGMA_UNROLL
        for (int half = 0; half < 2; ++half) {
            const int row = group_major ? head_g[half] * tokens + t
                                        : t * groups + head_g[half];
            const int col = head_col[half] + quarter * kVhaCols;
            out[half] = int64_t(row) * d + col + quad * kVhaMmaN;
            sf_off[half] =
                mxfp8::sf_atom_offset(row, col / mxfp8::kGroup, kb_dim);
        }
        const uint8_t* const tile_bytes =
            reinterpret_cast<const uint8_t*>(tile);
        uint32_t sf_word[2] = {};
        CUTLASS_PRAGMA_UNROLL
        for (int nh = 0; nh < kVhaNHalves; ++nh) {
            float acc[kVhaHalfTiles][4] = {};
            CUTLASS_PRAGMA_UNROLL
            for (int s = 0; s < kVhaKSteps; ++s) {
                uint32_t b[kVhaHalfTiles / 2][4];
                CUTLASS_PRAGMA_UNROLL
                for (int p = 0; p < kVhaHalfTiles / 2; ++p) {
                    cute::SM75_U16x8_LDSM_T::copy(
                        *reinterpret_cast<const cute::uint128_t*>(
                            tile_bytes + s * kVhaMmaK * kVhaRowBytes +
                            ldsm_row + ldsm_chunk[nh * kVhaHalfTiles / 2 + p]),
                        b[p][0], b[p][1], b[p][2], b[p][3]);
                }
                CUTLASS_PRAGMA_UNROLL
                for (int n = 0; n < kVhaHalfTiles; ++n) {
                    float* r = acc[n];
                    VhaMma::fma(r[0], r[1], r[2], r[3], a[s][0], a[s][1],
                                a[s][2], a[s][3], b[n / 2][2 * (n % 2)],
                                b[n / 2][2 * (n % 2) + 1], r[0], r[1], r[2],
                                r[3]);
                }
            }
            CUTLASS_PRAGMA_UNROLL
            for (int gh = 0; gh < kVhaHalfGroups; ++gh) {
                const int gq = nh * kVhaHalfGroups + gh;
                uint32_t codes[2][kVhaGroupTiles / 2];
                CUTLASS_PRAGMA_UNROLL
                for (int half = 0; half < 2; ++half) {
                    __nv_bfloat162 v[kVhaGroupTiles];
                    CUTLASS_PRAGMA_UNROLL
                    for (int x = 0; x < kVhaGroupTiles; ++x) {
                        const float* r =
                            acc[gh * kVhaGroupTiles + x] + 2 * half;
                        v[x] = __floats2bfloat162_rn(r[0], r[1]);
                    }
                    const __nv_bfloat162 m = mxfp8::abs_max<kVhaGroupTiles>(v);
                    float amax = fmaxf(__low2float(m), __high2float(m));
                    CUTLASS_PRAGMA_UNROLL
                    for (int sh = 1; sh < kQuad; sh *= 2) {
                        amax = fmaxf(amax, __shfl_xor_sync(~0u, amax, sh));
                    }
                    float inv_scale;
                    const uint32_t e = mxfp8::ue8m0_scale(
                        fmaxf(amax, mxfp8::kAmaxFloor) / mxfp8::kE4m3Max,
                        inv_scale);
                    sf_word[half] |= e << (8 * gq);
                    const __nv_bfloat162 sc = __float2bfloat162_rn(inv_scale);
                    CUTLASS_PRAGMA_UNROLL
                    for (int x = 0; x < kVhaGroupTiles; x += 2) {
                        codes[half][x / 2] = mxfp8::e4m3x4(v[x], v[x + 1], sc);
                    }
                }
                // Word x: n-tile (4 gq + x) codes of row h (low) and h + 8
                // (high); after the transpose word j holds lane j's codes of
                // n-tile (4 gq + quad), i.e. bytes 2j.. of its 8 columns.
                const uint4 w =
                    make_uint4(__byte_perm(codes[0][0], codes[1][0], 0x5410),
                               __byte_perm(codes[0][0], codes[1][0], 0x7632),
                               __byte_perm(codes[0][1], codes[1][1], 0x5410),
                               __byte_perm(codes[0][1], codes[1][1], 0x7632));
                uint32_t tw[kQuad];
                quad_transpose_4x4(w, lane, tw);
                const int col = gq * mxfp8::kGroup;
                *reinterpret_cast<uint2*>(q + out[0] + col) =
                    make_uint2(__byte_perm(tw[0], tw[1], 0x5410),
                               __byte_perm(tw[2], tw[3], 0x5410));
                *reinterpret_cast<uint2*>(q + out[1] + col) =
                    make_uint2(__byte_perm(tw[0], tw[1], 0x7632),
                               __byte_perm(tw[2], tw[3], 0x7632));
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
    for (int i = blockIdx.x * kVhaThreads + tid; i < pad_words;
         i += gridDim.x * kVhaThreads) {
        *reinterpret_cast<uint32_t*>(
            sf + mxfp8::sf_atom_offset(rows + i / words_per_row,
                                       i % words_per_row * mxfp8::kAtomCols,
                                       kb_dim)) = 0;
    }
}

template <typename Pos>
void LaunchVhaPostmixQuant(const torch::Tensor& o, const torch::Tensor& mix,
                           const torch::Tensor& freqs,
                           const torch::Tensor& positions, int groups,
                           bool group_major, torch::Tensor& q,
                           torch::Tensor& sf, bool use_pdl) {
    const int tokens = o.size(0);
    const int grid = std::max(
        std::min(tokens * kVhaQuarters, GetSmCount() * kVhaBlocksPerSm), 1);
    const auto kernel = vha_postmix_quant_kernel<Pos>;
    HALOPS_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kVhaSmemBytes));
    LaunchKernelWithPdl(kernel, grid, kVhaThreads, kNoCluster, kVhaSmemBytes,
                        GetCurrentCUDAStream(o), use_pdl,
                        reinterpret_cast<const __nv_bfloat16*>(o.data_ptr()),
                        reinterpret_cast<const __nv_bfloat16*>(mix.data_ptr()),
                        freqs.data_ptr<float>(), positions.data_ptr<Pos>(),
                        tokens, groups, group_major,
                        reinterpret_cast<uint8_t*>(q.data_ptr()),
                        sf.data_ptr<uint8_t>());
}

}  // namespace

std::vector<torch::Tensor> VhaPostmixQuant(
    const torch::Tensor& o, const torch::Tensor& mix,
    const torch::Tensor& freqs, const torch::Tensor& positions,
    const int64_t groups, const bool group_major, const bool use_pdl,
    const std::optional<torch::Tensor>& output_quant_opt,
    const std::optional<torch::Tensor>& output_sf_opt) {
    const CudaDeviceGuard device_guard(o.device());
    TORCH_CHECK(o.dim() == 3 && o.size(1) == kVhaHeads &&
                    o.size(2) == kVhaHeadDim &&
                    o.scalar_type() == torch::kBFloat16 && o.is_contiguous(),
                "o must be a contiguous bfloat16 [T, ", kVhaHeads, ", ",
                kVhaHeadDim, "] tensor");
    TORCH_CHECK(mix.sizes() == torch::IntArrayRef({kVhaHeads, kVhaHeads}) &&
                    mix.scalar_type() == torch::kBFloat16 &&
                    mix.is_contiguous(),
                "mix must be a contiguous bfloat16 [", kVhaHeads, ", ",
                kVhaHeads, "] tensor");
    TORCH_CHECK(freqs.dim() == 2 && freqs.size(1) == kVhaRopeDim &&
                    freqs.scalar_type() == torch::kFloat &&
                    freqs.is_contiguous(),
                "freqs must be a contiguous float32 [max_pos, ", kVhaRopeDim,
                "] (cos, sin) table");
    const int tokens = o.size(0);
    TORCH_CHECK(positions.dim() == 1 && positions.size(0) == tokens &&
                    (positions.scalar_type() == torch::kInt ||
                     positions.scalar_type() == torch::kLong),
                "positions must be an int32 / int64 [T] tensor");
    TORCH_CHECK(groups > 0 && kVhaHeads % groups == 0, "groups must divide ",
                kVhaHeads, ", got ", groups);
    const int d = kVhaHeads / groups * kVhaHeadDim;
    const int rows = groups * tokens;
    const int m_sf = AlignSfM(rows);
    const std::vector<int64_t> q_shape =
        group_major ? std::vector<int64_t>{groups, tokens, d}
                    : std::vector<int64_t>{tokens, groups, d};
    torch::Tensor q =
        output_quant_opt.has_value()
            ? *output_quant_opt
            : torch::empty(q_shape, o.options().dtype(torch::kFloat8_e4m3fn));
    torch::Tensor sf = output_sf_opt.has_value()
                           ? *output_sf_opt
                           : torch::empty({m_sf, d / mxfp8::kGroup},
                                          o.options().dtype(torch::kByte));
    TORCH_CHECK(q.is_contiguous() && q.scalar_type() == torch::kFloat8_e4m3fn &&
                    q.numel() == int64_t(rows) * d,
                "output_quant must be a contiguous float8_e4m3fn tensor of ",
                rows, " x ", d, " elements");
    TORCH_CHECK(sf.is_contiguous() && sf.scalar_type() == torch::kByte &&
                    sf.numel() == int64_t(m_sf) * (d / mxfp8::kGroup),
                "output_sf must be a contiguous uint8 tensor of ", m_sf, " x ",
                d / mxfp8::kGroup, " elements");
    if (m_sf == 0) {
        return {q, sf};
    }
    if (positions.scalar_type() == torch::kInt) {
        LaunchVhaPostmixQuant<int32_t>(o, mix, freqs, positions, groups,
                                       group_major, q, sf, use_pdl);
    } else {
        LaunchVhaPostmixQuant<int64_t>(o, mix, freqs, positions, groups,
                                       group_major, q, sf, use_pdl);
    }
    HALOPS_CUDA_CHECK(cudaGetLastError());
    return {q, sf};
}

}  // namespace halops
