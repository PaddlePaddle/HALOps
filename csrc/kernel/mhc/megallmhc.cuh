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

#include <cuda.h>
#include <cuda_bf16.h>
#include <cutlass/arch/barrier.h>

#include <cstddef>
#include <cstdint>
#include <cute/arch/copy_sm75.hpp>
#include <cute/arch/copy_sm90.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/mma_sm80.hpp>

#include "common/mx_quant.cuh"
#include "common/pdl.cuh"
#include "common/ptx.cuh"
#include "kernel/mhc/megallmhc_utils.cuh"

namespace halops::mhc {

template <int Hidden, int Hc = 4, int Ring = 4, int FinCols = 16>
struct MhcKernelTraits {
    static constexpr int kHidden = Hidden;
    static constexpr int kHc = Hc;
    static constexpr int kRing = Ring;
    static constexpr int kFinCols = FinCols;
    static constexpr int kFinNumRegs = 56;

    static constexpr int kMixDim = kHc * 2 + kHc * kHc;
    static constexpr int kPC = kHc + kHc * kHc;
    static constexpr int kScaleGroups = 3;
    static constexpr int kSqSlot = kMixDim;
    static constexpr int kSfCols = kHidden / mxfp8::kGroup;

    static constexpr int kMTile = 16;
    static constexpr int kNTile = 8;
    static constexpr int kNTiles = kMixDim / kNTile;
    static constexpr int kSwizzleRowBytes = 128;
    static constexpr int kSwizzleAtomBytes = 8 * kSwizzleRowBytes;
    static constexpr int kCtaCols = kSwizzleRowBytes / sizeof(__nv_bfloat16);
    static constexpr int kWarpCols = 16;
    static constexpr int kChunks = kHidden / kCtaCols;
    static constexpr int kPartComputeWarps = kCtaCols / kWarpCols;
    static constexpr int kPartLoadWarpId = kPartComputeWarps;
    static constexpr int kPartNumWarp = kPartLoadWarpId + 1;
    static constexpr int kPartComputeThreads = 32 * kPartComputeWarps;
    static constexpr int kPartThreads = 32 * kPartNumWarp;
    static constexpr int kPartW = 32;
    static constexpr int kPartVecs = kPartW * sizeof(float) / sizeof(uint4);

    static constexpr int kInSlots = 2;
    static constexpr int kQBytes = kMTile * kSwizzleRowBytes;
    static constexpr int kRedStride = 40;
    static constexpr int kPcOff = (kInSlots + kHc) * kQBytes;
    static constexpr int kCombOff = kMTile * kHc;
    static constexpr int kPcBytes = kMTile * kPC * sizeof(float);
    static constexpr int kStageBytes =
        (kPcOff + kPcBytes + kSwizzleAtomBytes - 1) / kSwizzleAtomBytes *
        kSwizzleAtomBytes;
    static constexpr int kRedBytes =
        kPartComputeWarps * kMTile * kRedStride * sizeof(float);
    static constexpr int kBarBytes = (2 * kRing + 2) * sizeof(uint64_t);
    static constexpr size_t kPartSmem =
        static_cast<size_t>(kRing) * kStageBytes + 2 * kRedBytes + kBarBytes +
        kSwizzleAtomBytes;

    static constexpr int kFinComputeWarps = kHidden / kFinCols / 32;
    static constexpr int kFinSinkhornWarpId = kFinComputeWarps;
    static constexpr int kFinNumWarp = kFinSinkhornWarpId + 1;
    static constexpr int kFinComputeThreads = 32 * kFinComputeWarps;
    static constexpr int kFinThreads = 32 * kFinNumWarp;
    static constexpr int kSmRegs = 65536;
    static constexpr int kSmThreads = 2048;
    static constexpr int kFinBlocksByRegs =
        kSmRegs / (kFinThreads * kFinNumRegs);
    static constexpr int kFinBlocksByThreads = kSmThreads / kFinThreads;
    static constexpr int kFinMinBlocks = kFinBlocksByRegs < kFinBlocksByThreads
                                             ? kFinBlocksByRegs
                                             : kFinBlocksByThreads;
    static constexpr int kBf16PerVec = sizeof(uint4) / sizeof(__nv_bfloat16);
    static constexpr int kFinVecs = kFinCols / kBf16PerVec;
    static constexpr int kFinQVecs = kFinCols / sizeof(uint4);
    static constexpr int kGroupThreads = mxfp8::kGroup / kFinCols;

    static_assert(kRing >= 2);
    static_assert(kHc * sizeof(float) % sizeof(uint4) == 0);
    static_assert(kMixDim % kNTile == 0);
    static_assert(kPartW >= kMixDim + 1 && 32 % kPartVecs == 0);
    static_assert(kRedStride >= kMixDim && kRedStride % 4 == 0);
    static_assert(kHidden % kCtaCols == 0);
    static_assert(kFinCols % sizeof(uint4) == 0 &&
                  mxfp8::kGroup % kFinCols == 0);
    static_assert(kHidden % (32 * kFinCols) == 0);
    static_assert(kFinThreads <= 1024);
    static_assert(kFinMinBlocks >= 1);
    static_assert((kChunks * kPartVecs) % kFinComputeThreads == 0);
};

struct Params {
    const __nv_bfloat16* __restrict__ x;
    const __nv_bfloat16* __restrict__ shared;
    const __nv_bfloat16* __restrict__ residual;
    const float* __restrict__ post;
    const float* __restrict__ comb;
    const float* __restrict__ fn;
    const float* __restrict__ hc_scale;
    const float* __restrict__ hc_base;
    const __nv_bfloat16* __restrict__ norm_weight;
    __nv_bfloat16* __restrict__ residual_out;
    __nv_bfloat16* __restrict__ layer_out;
    float* __restrict__ post_out;
    float* __restrict__ comb_out;
    uint8_t* __restrict__ q_out;
    uint8_t* __restrict__ sf_out;
    int num_tokens;
    int sf_swizzle;
    int sinkhorn_repeat;
    float rms_eps;
    float hc_pre_eps;
    float hc_sinkhorn_eps;
    float hc_post_mult;
    float norm_eps;
};

struct TmaMaps {
    CUtensorMap x, shared, res, out;
};

template <typename Traits, bool kAlignFleet>
CUTE_DEVICE float mix_rms(float sqrsum, float eps) {
    constexpr float kN = static_cast<float>(Traits::kHc * Traits::kHidden);
    if constexpr (kAlignFleet)
        return 1.0f / (sqrtf(sqrsum / kN) + eps);
    else
        return rsqrtf(sqrsum / kN + eps);
}

template <typename Traits>
CUTE_DEVICE cute::uint128_t& frag(uint8_t* slot, int lane) {
    const int c = 2 * static_cast<int>(threadIdx.x / 32) + (lane >> 4);
    return *reinterpret_cast<cute::uint128_t*>(
        slot + (lane & 15) * Traits::kSwizzleRowBytes +
        ((c ^ (lane & 7)) << 4));
}

template <typename Traits, bool kHasShared>
CUTE_DEVICE void mix_tile(
    uint8_t* stage, const float* pcs,
    const uint32_t (&wb)[Traits::kNTiles][Traits::kHc][2][2], int lane,
    float* red) {
    using Ldsm = cute::SM75_U32x4_LDSM_N;
    constexpr int kHc = Traits::kHc, kNTiles = Traits::kNTiles;
    constexpr int kRedStride = Traits::kRedStride;
    const int g = lane >> 2, t = lane & 3;
    uint32_t xa[4], sa[4], ra[kHc][4];
    copy_to_regs<Ldsm>(frag<Traits>(stage, lane), xa);
    if constexpr (kHasShared)
        copy_to_regs<Ldsm>(frag<Traits>(stage + Traits::kQBytes, lane), sa);
#pragma unroll
    for (int j = 0; j < kHc; ++j)
        copy_to_regs<Ldsm>(
            frag<Traits>(stage + (Traits::kInSlots + j) * Traits::kQBytes,
                         lane),
            ra[j]);
    uint32_t a[kHc][4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const float* pp = pcs + (g + 8 * (k & 1)) * kHc;
        const float* cp =
            pcs + Traits::kCombOff + (g + 8 * (k & 1)) * kHc * kHc - kHc;
        float2 hin = bf16x2_to_float2(xa[k]);
        if constexpr (kHasShared)
            hin = __fadd2_rn(hin, bf16x2_to_float2(sa[k]));
        float2 r[kHc];
#pragma unroll
        for (int i = 0; i < kHc; ++i) r[i] = bf16x2_to_float2(ra[i][k]);
#pragma unroll
        for (int j = 0; j < kHc; ++j) {
            float2 acc = __ffma2_rn(splat(pp[j]), hin,
                                    __fmul2_rn(splat(cp[kHc + j]), r[0]));
#pragma unroll
            for (int i = 1; i < kHc; ++i)
                acc = __ffma2_rn(splat(cp[kHc + i * kHc + j]), r[i], acc);
            a[j][k] = pack_bf16_rn(acc.x, acc.y);
        }
    }
    __syncwarp();
#pragma unroll
    for (int j = 0; j < kHc; ++j)
        cute::SM90_U32x4_STSM_N::copy(
            a[j][0], a[j][1], a[j][2], a[j][3],
            frag<Traits>(stage + (Traits::kInSlots + j) * Traits::kQBytes,
                         lane));
    float cc[kNTiles][2][4];
#pragma unroll
    for (int nt = 0; nt < kNTiles; ++nt)
#pragma unroll
        for (int h = 0; h < 2; ++h)
            cc[nt][h][0] = cc[nt][h][1] = cc[nt][h][2] = cc[nt][h][3] = 0.0f;
#pragma unroll
    for (int s = 1; s >= 0; --s)
#pragma unroll
        for (int j = 0; j < kHc; ++j)
#pragma unroll
            for (int nt = 0; nt < kNTiles; ++nt) {
                float(&d)[4] = cc[nt][j & 1];
                cute::SM80_16x8x16_F32BF16BF16F32_TN::fma(
                    d[0], d[1], d[2], d[3], a[j][0], a[j][1], a[j][2], a[j][3],
                    wb[nt][j][s][0], wb[nt][j][s][1], d[0], d[1], d[2], d[3]);
            }
    float c[kNTiles][4];
#pragma unroll
    for (int nt = 0; nt < kNTiles; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e) c[nt][e] = cc[nt][0][e] + cc[nt][1][e];
    __syncwarp();
#pragma unroll
    for (int nt = 0; nt < kNTiles; ++nt) {
        *reinterpret_cast<float2*>(
            &red[g * kRedStride + Traits::kNTile * nt + 2 * t]) =
            make_float2(c[nt][0], c[nt][1]);
        *reinterpret_cast<float2*>(
            &red[(g + 8) * kRedStride + Traits::kNTile * nt + 2 * t]) =
            make_float2(c[nt][2], c[nt][3]);
    }
}

template <typename Traits>
CUTE_DEVICE void store_part(const float* red, float* part, int64_t tok0, int n,
                            int chunk, int lane) {
    constexpr int kMixDim = Traits::kMixDim, kRedStride = Traits::kRedStride;
    for (int i = lane; i < Traits::kMTile * kMixDim / 4; i += 32) {
        const int tk = i / (kMixDim / 4), q4 = i % (kMixDim / 4);
        if (tk >= n) continue;
        float4 s = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
        for (int w = 0; w < Traits::kPartComputeWarps; ++w) {
            const float4 v = *reinterpret_cast<const float4*>(
                &red[(w * Traits::kMTile + tk) * kRedStride + 4 * q4]);
            s.x += v.x;
            s.y += v.y;
            s.z += v.z;
            s.w += v.w;
        }
        *reinterpret_cast<float4*>(
            &part[((tok0 + tk) * Traits::kChunks + chunk) * Traits::kPartW +
                  4 * q4]) = s;
    }
}

template <typename Traits, bool kHasShared>
CUTE_DEVICE void issue_stage(const Params& p, const TmaMaps& tm, uint8_t* stage,
                             uint64_t* bar, int col0, int tok0, int n) {
    constexpr int kHc = Traits::kHc;
    constexpr int kStreams = (kHasShared ? 2 : 1) + kHc;
    constexpr auto kHint =
        static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL);
    cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(
        bar, kStreams * Traits::kQBytes + n * Traits::kPC * sizeof(float));
    cute::SM90_TMA_LOAD_2D::copy(&tm.x, bar, kHint, stage, col0, tok0);
    if constexpr (kHasShared)
        cute::SM90_TMA_LOAD_2D::copy(&tm.shared, bar, kHint,
                                     stage + Traits::kQBytes, col0, tok0);
#pragma unroll
    for (int j = 0; j < kHc; ++j)
        cute::SM90_TMA_LOAD_3D::copy(
            &tm.res, bar, kHint,
            stage + (Traits::kInSlots + j) * Traits::kQBytes, col0, j, tok0);
    cute::SM90_BULK_COPY_G2S::copy(p.post + static_cast<int64_t>(tok0) * kHc,
                                   bar, stage + Traits::kPcOff,
                                   n * kHc * sizeof(float));
    cute::SM90_BULK_COPY_G2S::copy(
        p.comb + static_cast<int64_t>(tok0) * kHc * kHc, bar,
        stage + Traits::kPcOff + Traits::kCombOff * sizeof(float),
        n * kHc * kHc * sizeof(float));
}

template <typename Traits>
CUTE_DEVICE void store_residual(const TmaMaps& tm, uint8_t* stage, int col0,
                                int tok0) {
#pragma unroll
    for (int j = 0; j < Traits::kHc; ++j)
        cute::SM90_TMA_STORE_3D::copy(
            &tm.out, stage + (Traits::kInSlots + j) * Traits::kQBytes, col0, j,
            tok0);
    cute::tma_store_arrive();
}

template <typename Traits, bool kHasShared>
__global__ void __launch_bounds__(Traits::kPartThreads)
    megallmhc_partial_kernel(const __grid_constant__ Params p,
                             float* __restrict__ part, int tokens_per_tile,
                             const __grid_constant__ TmaMaps tm) {
    using Barrier = cutlass::arch::ClusterBarrier;
    constexpr int kHc = Traits::kHc, kRing = Traits::kRing;
    constexpr int kStageBytes = Traits::kStageBytes,
                  kRedBytes = Traits::kRedBytes;
    extern __shared__ __align__(Traits::kSwizzleRowBytes) uint8_t smem_raw[];
    const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    const int g = lane >> 2, t = lane & 3;
    const int col0 = blockIdx.x * Traits::kCtaCols;
    const int colw = col0 + warp * Traits::kWarpCols;
    const int tok0 = static_cast<int>(blockIdx.y) * tokens_per_tile;
    const int n = max(0, min(p.num_tokens - tok0, tokens_per_tile));
    const int nmt = (n + Traits::kMTile - 1) / Traits::kMTile;
    constexpr uint32_t kAlignMask = Traits::kSwizzleAtomBytes - 1;
    uint8_t* const smem =
        smem_raw + ((Traits::kSwizzleAtomBytes -
                     (cute::cast_smem_ptr_to_uint(smem_raw) & kAlignMask)) &
                    kAlignMask);
    float* red = reinterpret_cast<float*>(smem + kRing * kStageBytes);
    constexpr int kRedFloats = kRedBytes / sizeof(float);
    uint64_t* const full =
        reinterpret_cast<uint64_t*>(smem + kRing * kStageBytes + 2 * kRedBytes);
    uint64_t* const cdone = full + kRing;
    uint64_t* const redfree = cdone + kRing;
    if (tid == 0) {
        for (int s = 0; s < kRing; ++s) {
            Barrier::init(full + s, 1);
            Barrier::init(cdone + s, Traits::kPartComputeThreads);
        }
        Barrier::init(redfree, 1);
        Barrier::init(redfree + 1, 1);
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();
    if (warp == Traits::kPartLoadWarpId) {
        pdl_wait();
        if (lane == 0)
            for (int s = 0; s < kRing && s < nmt; ++s)
                issue_stage<Traits, kHasShared>(
                    p, tm, smem + s * kStageBytes, full + s, col0,
                    tok0 + s * Traits::kMTile,
                    min(Traits::kMTile, n - s * Traits::kMTile));
        for (int mt = 0; mt < nmt; ++mt) {
            const int slot = mt % kRing;
            uint8_t* const st = smem + slot * kStageBytes;
            const int tm0 = tok0 + mt * Traits::kMTile;
            Barrier::wait(cdone + slot, (mt / kRing) & 1);
            if (mt == nmt - 1) pdl_trigger();
            if (lane == 0) store_residual<Traits>(tm, st, col0, tm0);
            store_part<Traits>(red + (mt & 1) * kRedFloats, part, tm0,
                               min(Traits::kMTile, n - mt * Traits::kMTile),
                               blockIdx.x, lane);
            __syncwarp();
            if (lane == 0) Barrier::arrive(redfree + (mt & 1));
            const int nx = mt + kRing;
            if (nx < nmt && lane == 0) {
                cute::tma_store_wait<0>();
                issue_stage<Traits, kHasShared>(
                    p, tm, st, full + slot, col0, tok0 + nx * Traits::kMTile,
                    min(Traits::kMTile, n - nx * Traits::kMTile));
            }
        }
        bulk_store_wait_all();
        return;
    }
    uint32_t wb[Traits::kNTiles][kHc][2][2];
#pragma unroll
    for (int nt = 0; nt < Traits::kNTiles; ++nt)
#pragma unroll
        for (int j = 0; j < kHc; ++j) {
            const float* src =
                p.fn + ((Traits::kNTile * nt + g) * kHc + j) * Traits::kHidden +
                colw + 2 * t;
            uint32_t lo[2], hi[2];
            split_bf16x2(__ldg(reinterpret_cast<const float2*>(src)), lo);
            split_bf16x2(__ldg(reinterpret_cast<const float2*>(src + 8)), hi);
#pragma unroll
            for (int s = 0; s < 2; ++s) {
                wb[nt][j][s][0] = lo[s];
                wb[nt][j][s][1] = hi[s];
            }
        }
    for (int mt = 0; mt < nmt; ++mt) {
        const int slot = mt % kRing;
        Barrier::wait(full + slot, (mt / kRing) & 1);
        if (mt >= 2) Barrier::wait(redfree + (mt & 1), ((mt >> 1) & 1) ^ 1);
        uint8_t* const st = smem + slot * kStageBytes;
        mix_tile<Traits, kHasShared>(
            st, reinterpret_cast<const float*>(st + Traits::kPcOff), wb, lane,
            red + (mt & 1) * kRedFloats +
                warp * Traits::kMTile * Traits::kRedStride);
        cutlass::arch::fence_view_async_shared();
        Barrier::arrive(cdone + slot);
    }
    pdl_trigger();
}

template <typename Traits>
CUTE_DEVICE void store_sf(uint8_t* sf, int tok, int kb, uint8_t e,
                          int num_tokens, bool swizzle) {
    constexpr int kSfCols = Traits::kSfCols;
    if (!swizzle) {
        sf[static_cast<int64_t>(tok) * kSfCols + kb] = e;
        return;
    }
    sf[mxfp8::sf_atom_offset(tok, kb, kSfCols)] = e;
    // Padding rows all lie in the last atom: add its base once, then index
    // rows inside it.
    const int last = (num_tokens - 1) / mxfp8::kAtomRows * mxfp8::kAtomRows;
    uint8_t* pad = sf + mxfp8::sf_atom_offset(last, kb, kSfCols);
    for (uint32_t r = tok + num_tokens - last; r < mxfp8::kAtomRows;
         r += num_tokens)
        pad[mxfp8::sf_atom_offset(r, 0, mxfp8::kAtomCols)] = 0;
}

template <typename Traits, bool kAlignFleet>
__global__ void __launch_bounds__(Traits::kFinThreads, Traits::kFinMinBlocks)
    megallmhc_finish_kernel(const __grid_constant__ Params p,
                            const float* __restrict__ part) {
    constexpr int kHidden = Traits::kHidden, kHc = Traits::kHc,
                  kMixDim = Traits::kMixDim, kFinCols = Traits::kFinCols;
    constexpr int kChunks = Traits::kChunks,
                  kFinComputeThreads = Traits::kFinComputeThreads,
                  kFinComputeWarps = Traits::kFinComputeWarps,
                  kFinVecs = Traits::kFinVecs;
    __shared__ alignas(16) float red[kFinComputeWarps][Traits::kPartW];
    __shared__ float pre_s[kHc];
    __shared__ float ysq_w[kFinComputeWarps];
    const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    const int64_t tok = blockIdx.x;
    const bool compute = tid < kFinComputeThreads;
    const int h0 = tid * kFinCols;
    uint4 wv[kFinVecs];
    if (compute)
#pragma unroll
        for (int k = 0; k < kFinVecs; ++k)
            wv[k] =
                __ldg(reinterpret_cast<const uint4*>(p.norm_weight + h0) + k);
    const int e = lane;
    float hcs = 0.0f, hcb = 0.0f;
    if ((warp == 0 || !compute) && e < kMixDim) {
        hcs = __ldg(p.hc_scale + (e < kHc ? 0 : e < 2 * kHc ? 1 : 2));
        hcb = __ldg(p.hc_base + e);
    }
    pdl_wait();
    pdl_trigger();
    uint4 rv[kHc][kFinVecs];
    if (compute) {
#pragma unroll
        for (int j = 0; j < kHc; ++j)
#pragma unroll
            for (int k = 0; k < kFinVecs; ++k)
                rv[j][k] = reinterpret_cast<const uint4*>(
                    p.residual_out + (tok * kHc + j) * kHidden + h0)[k];
        const float4* src = reinterpret_cast<const float4*>(
            part + tok * kChunks * Traits::kPartW);
        float4 s = src[tid];
#pragma unroll
        for (int u = 1; u < kChunks * Traits::kPartVecs / kFinComputeThreads;
             ++u) {
            const float4 v = src[tid + u * kFinComputeThreads];
            s.x += v.x;
            s.y += v.y;
            s.z += v.z;
            s.w += v.w;
        }
#pragma unroll
        for (int o = Traits::kPartVecs; o < 32; o <<= 1) {
            s.x += __shfl_xor_sync(0xffffffffu, s.x, o);
            s.y += __shfl_xor_sync(0xffffffffu, s.y, o);
            s.z += __shfl_xor_sync(0xffffffffu, s.z, o);
            s.w += __shfl_xor_sync(0xffffffffu, s.w, o);
        }
        float2 sq2[kFinVecs];
#pragma unroll
        for (int k = 0; k < kFinVecs; ++k) sq2[k] = make_float2(0.0f, 0.0f);
#pragma unroll
        for (int j = 0; j < kHc; ++j)
#pragma unroll
            for (int k = 0; k < kFinVecs; ++k) {
                const uint32_t* u =
                    reinterpret_cast<const uint32_t*>(&rv[j][k]);
#pragma unroll
                for (int q = 0; q < Traits::kBf16PerVec / 2; ++q) {
                    const float2 v = bf16x2_to_float2_opaque(u[q]);
                    sq2[k] = __ffma2_rn(v, v, sq2[k]);
                }
            }
#pragma unroll
        for (int k = 1; k < kFinVecs; ++k) sq2[0] = __fadd2_rn(sq2[0], sq2[k]);
        float sq = sq2[0].x + sq2[0].y;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            sq += __shfl_xor_sync(0xffffffffu, sq, o);
        if (lane == Traits::kSqSlot / 4) s.x = sq;
        if (lane < Traits::kPartVecs)
            *reinterpret_cast<float4*>(&red[warp][4 * lane]) = s;
    }
    __syncthreads();
    if (warp == 0 || !compute) {
        float t = red[0][lane];
#pragma unroll
        for (int r = 1; r < kFinComputeWarps; ++r) t += red[r][lane];
        const float sq = __shfl_sync(0xffffffffu, t, Traits::kSqSlot);
        const float rms = mix_rms<Traits, kAlignFleet>(sq, p.rms_eps);
        const float z = fmaf(t * rms, hcs, hcb);
        if (!compute) {
            float c[kHc * kHc];
#pragma unroll
            for (int k = 0; k < kHc * kHc; ++k)
                c[k] = __shfl_sync(0xffffffffu, z, 2 * kHc + k);
            if (lane == 0) {
                sinkhorn<kHc, kAlignFleet>(c, p.hc_sinkhorn_eps,
                                           p.sinkhorn_repeat);
                float4* dst =
                    reinterpret_cast<float4*>(p.comb_out + tok * kHc * kHc);
#pragma unroll
                for (int k = 0; k < kHc * kHc / 4; ++k)
                    dst[k] = make_float4(c[4 * k], c[4 * k + 1], c[4 * k + 2],
                                         c[4 * k + 3]);
            }
            return;
        }
        if (e < kHc)
            pre_s[e] = sigmoid(z) + p.hc_pre_eps;
        else if (e < 2 * kHc)
            p.post_out[tok * kHc + e - kHc] = sigmoid(z) * p.hc_post_mult;
    }
    cutlass::arch::NamedBarrier::sync(
        kFinComputeThreads,
        cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
    float pre[kHc];
#pragma unroll
    for (int j = 0; j < kHc; ++j) pre[j] = pre_s[j];
    float y[kFinCols];
    float ysq = 0.0f;
#pragma unroll
    for (int c = 0; c < kFinCols; ++c) {
        float rf[kHc];
#pragma unroll
        for (int j = 0; j < kHc; ++j)
            rf[j] = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(
                &rv[j][c / Traits::kBf16PerVec])[c % Traits::kBf16PerVec]);
        float v = pre[0] * rf[0];
#pragma unroll
        for (int j = 1; j < kHc; ++j) v = fmaf(pre[j], rf[j], v);
        y[c] = v;
        ysq = fmaf(v, v, ysq);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        ysq += __shfl_xor_sync(0xffffffffu, ysq, o);
    if (lane == 0) ysq_w[warp] = ysq;
    cutlass::arch::NamedBarrier::sync(
        kFinComputeThreads,
        cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
    float ytot = ysq_w[0];
#pragma unroll
    for (int w = 1; w < kFinComputeWarps; ++w) ytot += ysq_w[w];
    const float rn = rsqrtf(ytot / static_cast<float>(kHidden) + p.norm_eps);
    uint4 ov[kFinVecs];
    __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(ov);
    float amax = mxfp8::kAmaxFloor;
#pragma unroll
    for (int c = 0; c < kFinCols; ++c) {
        const float yb = __bfloat162float(__float2bfloat16_rn(y[c]));
        const float wn =
            __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(wv)[c]);
        ob[c] = __float2bfloat16_rn((yb * rn) * wn);
        amax = fmaxf(amax, fabsf(__bfloat162float(ob[c])));
    }
    constexpr int kGroupThreads = Traits::kGroupThreads;
#pragma unroll
    for (int o = 1; o < kGroupThreads; o <<= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    float inv;
    const uint8_t be = mxfp8::ue8m0_scale(amax / mxfp8::kE4m3Max, inv);
    const auto clamp = [](float v) {
        return fminf(fmaxf(v, -mxfp8::kE4m3Max), mxfp8::kE4m3Max);
    };
    uint4 qv[Traits::kFinQVecs];
    uint16_t* qp = reinterpret_cast<uint16_t*>(qv);
#pragma unroll
    for (int c = 0; c < kFinCols; c += 2)
        qp[c / 2] = __nv_cvt_float2_to_fp8x2(
            make_float2(clamp(__bfloat162float(ob[c]) * inv),
                        clamp(__bfloat162float(ob[c + 1]) * inv)),
            __NV_SATFINITE, __NV_E4M3);
    uint4* lo = reinterpret_cast<uint4*>(p.layer_out + tok * kHidden + h0);
#pragma unroll
    for (int k = 0; k < kFinVecs; ++k) lo[k] = ov[k];
    uint4* qo = reinterpret_cast<uint4*>(p.q_out + tok * kHidden + h0);
#pragma unroll
    for (int k = 0; k < Traits::kFinQVecs; ++k) qo[k] = qv[k];
    if ((tid & (kGroupThreads - 1)) == 0)
        store_sf<Traits>(p.sf_out, tok, h0 / mxfp8::kGroup, be, p.num_tokens,
                         p.sf_swizzle);
}

}  // namespace halops::mhc
