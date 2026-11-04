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

#include <cstdint>

#include "common/device_timer.hpp"
#include "moe_tail_prims.cuh"

namespace halops::moe {
namespace tail {
template <int ROWS, int THREADS>
struct SharedWeightGemvCfg {
    static constexpr int kWarps = THREADS / 32;
    static constexpr int kTiles = ROWS / 16;
    static constexpr int kKSplit = kTiles >= kWarps ? 1 : kWarps / kTiles;
    static constexpr int kItemsPerWarp = kTiles * kKSplit / kWarps;
    static_assert(ROWS % 16 == 0,
                  "rows per CTA must be a multiple of the mma M tile");
    static_assert(kTiles * kKSplit % kWarps == 0,
                  "work items must tile over the warps");
};

template <int ROWS, int L, int PAD>
constexpr int stage_smem_bytes() {
    return ROWS * (L + PAD);
}

template <int L, int TPG, int ROWS, int BLKB, int THREADS, int SCALE_BLOCK,
          bool STAGED = false>
__device__ __forceinline__ void shared_weight_mma_gemv_fp8(
    const uint8_t* __restrict__ w, const float* __restrict__ scale,
    const uint16_t* s_x, int x_stride, float* s_part, uint32_t warp,
    uint32_t lane, const uint8_t* s_w = nullptr, int w_stride = 0) {
    using Cfg = SharedWeightGemvCfg<ROWS, THREADS>;
    constexpr int KR = L / Cfg::kKSplit;
    constexpr int kNSB = KR / 64;
    constexpr int kNG = KR / SCALE_BLOCK;
    constexpr int kNC = kNSB / BLKB;
    static_assert(TPG >= 1 && TPG <= 8, "mma N is 8");
    static_assert(SCALE_BLOCK == 128, "scale group = two 64-wide super-blocks");
    static_assert(ROWS <= SCALE_BLOCK, "CTA rows must fit in one scale block");
    static_assert(KR % SCALE_BLOCK == 0 && kNSB % BLKB == 0 && BLKB % 2 == 0,
                  "K blocks must tile into batches");
    const uint32_t grp = lane >> 2, tig = lane & 3;
    const bool has_x = grp < uint32_t(TPG);
#pragma unroll
    for (int i = 0; i < Cfg::kItemsPerWarp; ++i) {
        const int item = int(warp) + i * Cfg::kWarps;
        const int tile = item % Cfg::kTiles, ks = item / Cfg::kTiles;
        const uint8_t* wa =
            STAGED
                ? s_w + (tile * 16 + int(grp)) * w_stride + ks * KR + tig * 16
                : w + int64_t(tile * 16 + grp) * L + ks * KR + tig * 16;
        const uint8_t* wb =
            wa + (STAGED ? int64_t(8) * w_stride : int64_t(8) * L);
        const uint16_t* xs =
            s_x + (has_x ? grp : 0) * x_stride + ks * KR + tig * 16;
        float sc[kNG];
#pragma unroll
        for (int g = 0; g < kNG; ++g) sc[g] = scale[ks * kNG + g];
        float4 total = make_float4(0.f, 0.f, 0.f, 0.f);
        float4 g0 = total, g1 = total;
        auto load_chunk = [&](int c, vec_t(&da)[BLKB], vec_t(&db)[BLKB]) {
#pragma unroll
            for (int k = 0; k < BLKB; ++k) {
                if constexpr (STAGED) {
                    *reinterpret_cast<uint4*>(&da[k]) =
                        *reinterpret_cast<const uint4*>(wa +
                                                        (c * BLKB + k) * 64);
                    *reinterpret_cast<uint4*>(&db[k]) =
                        *reinterpret_cast<const uint4*>(wb +
                                                        (c * BLKB + k) * 64);
                } else {
                    ld_global_16B(da[k], wa + (c * BLKB + k) * 64, 0);
                    ld_global_16B(db[k], wb + (c * BLKB + k) * 64, 0);
                }
            }
        };
        vec_t ca[BLKB], cb[BLKB];
        load_chunk(0, ca, cb);
#pragma unroll
        for (int c = 0; c < kNC; ++c) {
            vec_t na[BLKB], nb[BLKB];
            if (c + 1 < kNC) load_chunk(c + 1, na, nb);
#pragma unroll
            for (int k = 0; k < BLKB; ++k) {
                const int sb = c * BLKB + k;
                uint4 xlo = make_uint4(0u, 0u, 0u, 0u), xhi = xlo;
                if (has_x) {
                    xlo = *reinterpret_cast<const uint4*>(xs + sb * 64);
                    xhi = *reinterpret_cast<const uint4*>(xs + sb * 64 + 8);
                }
                const uint32_t xb[8] = {xlo.x, xlo.y, xlo.z, xlo.w,
                                        xhi.x, xhi.y, xhi.z, xhi.w};
                const uint4& A = *reinterpret_cast<const uint4*>(&ca[k]);
                const uint4& B = *reinterpret_cast<const uint4*>(&cb[k]);
                const uint32_t aw[4] = {A.x, A.y, A.z, A.w},
                               bw[4] = {B.x, B.y, B.z, B.w};
#pragma unroll
                for (int mm = 0; mm < 4; ++mm) {
                    uint32_t a0, a1, a2, a3;
                    e4m3x4_to_f16x2x2(aw[mm], a0, a2);
                    e4m3x4_to_f16x2x2(bw[mm], a1, a3);
                    mma_f16_m16n8k16((mm & 1) ? g1 : g0, a0, a1, a2, a3,
                                     xb[2 * mm], xb[2 * mm + 1]);
                }
                if (sb & 1) {
                    const float s = sc[sb >> 1];
                    total.x += s * (g0.x + g1.x);
                    total.y += s * (g0.y + g1.y);
                    total.z += s * (g0.z + g1.z);
                    total.w += s * (g0.w + g1.w);
                    g0 = make_float4(0.f, 0.f, 0.f, 0.f);
                    g1 = g0;
                }
            }
            if (c + 1 < kNC) {
#pragma unroll
                for (int k = 0; k < BLKB; ++k) {
                    ca[k] = na[k];
                    cb[k] = nb[k];
                }
            }
        }
        float* dst = s_part + ks * (TPG * ROWS);
        const int r = tile * 16 + int(grp);
        if (2 * int(tig) < TPG) {
            dst[(2 * tig) * ROWS + r] = total.x;
            dst[(2 * tig) * ROWS + r + 8] = total.z;
        }
        if (2 * int(tig) + 1 < TPG) {
            dst[(2 * tig + 1) * ROWS + r] = total.y;
            dst[(2 * tig + 1) * ROWS + r + 8] = total.w;
        }
    }
}

template <int WORLD, int L, int H, int MAXM, class TC>
struct TailSmem {
    static constexpr int TPG = TC::kTpg, GSIZE = TC::kGsize,
                         THREADS = TC::kThreads, STAGE = TC::kStage;
    static constexpr uint32_t kWarps = THREADS / 32;
    static constexpr int kShard = H / WORLD;
    static constexpr int kRows = kShard / GSIZE;
    static constexpr int kXStride = L + TC::kXPad;
    static constexpr uint32_t align_up(uint32_t v, uint32_t a) {
        return (v + a - 1) / a * a;
    }
    static constexpr uint32_t kWStage = 0;
    static constexpr uint32_t kWStageBytes =
        STAGE == 0 ? 0u
                   : uint32_t(stage_smem_bytes<kRows, L, TC::kWStagePad>());
    static constexpr uint32_t kWBar = align_up(kWStage + kWStageBytes, 16);
    static constexpr uint32_t kX = align_up(kWBar + 8, 16);
    static constexpr uint32_t kWarp =
        align_up(kX + uint32_t(TPG * kXStride) * 2, 16);
    static constexpr uint32_t kPart =
        align_up(kWarp + uint32_t(kWarps) * 4, 16);
    static constexpr uint32_t kCol =
        align_up(kPart + uint32_t(SharedWeightGemvCfg<kRows, THREADS>::kKSplit *
                                  TPG * kRows) *
                             4,
                 16);
    static constexpr uint32_t kBytes =
        align_up(kCol + uint32_t(TPG * kRows) * 4, 128);
};

template <int WORLD, int L, int H, int MAXM, class TC, class Sync>
struct TailGroup {
    using P = Params<WORLD, L, H, MAXM>;
    using Smem = TailSmem<WORLD, L, H, MAXM, TC>;
    static constexpr int TPG = TC::kTpg, GSIZE = TC::kGsize,
                         THREADS = TC::kThreads, STAGE = TC::kStage,
                         BLKB = TC::kBlkB, SCALE_BLOCK = TC::kScaleBlock;
    static constexpr uint32_t kVec = 8;  // bf16 per 16B vector
    static constexpr uint32_t kThreads = THREADS;
    static constexpr uint32_t kWarps = kThreads / 32;

    static constexpr int kShard = H / WORLD;
    static constexpr int kRows = kShard / GSIZE;
    using Cfg = SharedWeightGemvCfg<kRows, THREADS>;
    static constexpr int kXStride = L + TC::kXPad;
    static constexpr uint32_t kNumCtas = uint32_t((MAXM / TPG) * GSIZE);
    static constexpr int kXBufBytes = MAXM * L * 2;
    static constexpr int kSliceVec = kXBufBytes / 16 / int(kNumCtas);
    static constexpr int kHV = H / kVec;
    static constexpr int kShardHV = kShard / kVec;
    static constexpr int kOwnHV = kRows / kVec;
    static constexpr int kNonOwnHV = (kHV - kShardHV) / GSIZE;
    static constexpr int kPer = kOwnHV + kNonOwnHV;
    static constexpr bool kPrefetchC = TPG * kPer <= int(kThreads);

    static_assert(kVec * sizeof(__nv_bfloat16) == sizeof(vec_t),
                  "kVec bf16 fill exactly one 16B vector");
    static_assert(L % (kThreads * kVec) == 0 && L / (kThreads * kVec) == 1,
                  "one CTA covers one L-row");
    static_assert(kShard % GSIZE == 0 && MAXM % TPG == 0, "group tiling");
    static_assert(TPG <= GSIZE && TPG <= 8,
                  "round A needs one CTA per token; mma N = 8");
    static_assert(kShard % SCALE_BLOCK == 0 && kRows <= SCALE_BLOCK,
                  "FP8 weights: rows must sit inside one scale block");
    static_assert(STAGE == 0 || (STAGE == 1 &&
                                 stage_smem_bytes<kRows, L, TC::kWStagePad>() <=
                                     TC::kMaxStageSmemBytes),
                  "weight staging: a CTA's rows must fit in shared memory");
    static_assert(kXBufBytes / 16 % int(kNumCtas) == 0 &&
                      kSliceVec <= int(kThreads),
                  "xbuf clear slice");
    static_assert(kRows % kVec == 0 && (kHV - kShardHV) % GSIZE == 0,
                  "round-C split");

    struct Ctx {
        const P& p;
        Sync sync;
        uint32_t cta, tid;
        uint32_t grank;
        uint32_t t0;
        int parityA, parityC;
        uint8_t* xbuf;
        uint8_t* s_wstage;
        uint64_t* s_wbar;
        __nv_bfloat16* s_x;
        float* s_warp;
        float* s_part;
        float* s_col;
    };

    __device__ static __forceinline__ Ctx make_ctx(const P& p, uint32_t cta,
                                                   uint32_t tid,
                                                   uint8_t* scratch,
                                                   Sync sync) {
        return Ctx{p,
                   sync,
                   cta,
                   tid,
                   cta % GSIZE,
                   (cta / GSIZE) * TPG,
                   0,
                   0,
                   p.uc_base[p.rank] + int64_t(2) * 2 * WORLD * MAXM * H * 2,
                   scratch + Smem::kWStage,
                   reinterpret_cast<uint64_t*>(scratch + Smem::kWBar),
                   reinterpret_cast<__nv_bfloat16*>(scratch + Smem::kX),
                   reinterpret_cast<float*>(scratch + Smem::kWarp),
                   reinterpret_cast<float*>(scratch + Smem::kPart),
                   reinterpret_cast<float*>(scratch + Smem::kCol)};
    }

    __device__ static __forceinline__ bool group_active(const Ctx& c) {
        bool active = false;
#pragma unroll
        for (int r = 0; r < TPG; ++r) active |= row_valid(c.p, c.t0 + r);
        return active;
    }

    __device__ static __forceinline__ void stage_weights(const Ctx& c) {
        if constexpr (STAGE != 0) {
            const uint8_t* wrow0 =
                c.p.fc2_w8 +
                (int64_t(c.p.rank) * kShard + int64_t(c.grank) * kRows) * L;
            if (c.tid == 0) mbar_init_1(c.s_wbar);
            c.sync();
            if (c.tid == 0) mbar_expect_tx(c.s_wbar, uint32_t(kRows) * L);
            c.sync();
            for (int r = int(c.tid); r < kRows; r += int(kThreads))
                bulk_g2s(c.s_wstage + r * (L + TC::kWStagePad),
                         wrow0 + int64_t(r) * L, L, c.s_wbar);
        }
    }

    __device__ static __forceinline__ void clear_xbuf(const Ctx& c) {
        if (int(c.tid) < kSliceVec) {
            vec_t z;
            z.fill(__nv_bfloat162{get_pos_zero<__nv_bfloat16>(),
                                  get_pos_zero<__nv_bfloat16>()});
            st_global_16B(z, c.xbuf + int64_t(c.parityA ^ 1) * kXBufBytes,
                          c.cta * kSliceVec + c.tid);
        }
    }

    __device__ static __forceinline__ void zero_inactive_group(const Ctx& c) {
        constexpr int kGroupVecs = TPG * (H / int(kVec));
        for (int v = int(c.grank * kThreads + c.tid); v < kGroupVecs;
             v += GSIZE * int(kThreads)) {
            const uint32_t t = c.t0 + uint32_t(v / (H / int(kVec)));
            if (t >= c.p.m) break;
            vec_t z;
            z.fill(__nv_bfloat162{get_pos_zero<__nv_bfloat16>(),
                                  get_pos_zero<__nv_bfloat16>()});
            st_global_16B(z, c.p.out + int64_t(t) * H,
                          uint32_t(v % (H / int(kVec))));
        }
    }

    __device__ static __forceinline__ void round_a_produce(const Ctx& c) {
        const P& p = c.p;
        const uint32_t tid = c.tid;
        uint4 xrow = make_uint4(0u, 0u, 0u, 0u);
        const uint32_t t = c.t0 + c.grank;
        if (row_valid(p, t)) {
            vec_t lat = poll_reduce<WORLD, L, H, MAXM>(p, /*round=*/0,
                                                       c.parityA, t, L, tid);
            float sq = 0.f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const auto f = cast<float2>(lat[j]);
                const float a = float(__nv_bfloat16(f.x)),
                            b = float(__nv_bfloat16(f.y));
                sq += a * a + b * b;
            }
            sq = warp_reduce_sum(sq);
            if ((tid & 31) == 0) c.s_warp[tid >> 5] = sq;
            c.sync();
            float total = 0.f;
#pragma unroll
            for (int w = 0; w < kWarps; ++w) total += c.s_warp[w];
            const float inv = rsqrtf(total / float(L) + p.eps);
            vec_t nw;
            ld_global_16B(nw, p.norm_weight, tid);
            vec_t normed;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const auto x = cast<float2>(lat[j]);
                const auto g = cast<float2>(nw[j]);
                normed[j] = cast<__nv_bfloat162>(
                    float2{x.x * inv * g.x, x.y * inv * g.y});
            }
            {
                uint4 h;
                uint32_t* hw = reinterpret_cast<uint32_t*>(&h);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const auto f = cast<float2>(normed[j]);
                    const __half2 h2 = __floats2half2_rn(f.x, f.y);
                    hw[j] = *reinterpret_cast<const uint32_t*>(&h2);
                }
                xrow = h;
            }
        }
        *reinterpret_cast<uint4*>(
            reinterpret_cast<uint8_t*>(&c.s_x[c.grank * kXStride]) + tid * 16) =
            xrow;
        vec_t pv = *reinterpret_cast<const vec_t*>(&xrow);
        sanitize(pv);
        if (row_valid(p, t))
            st_global_16B(
                pv, c.xbuf + (int64_t(c.parityA) * MAXM + t) * (L * 2), tid);
    }

    __device__ static __forceinline__ void round_a_gather(const Ctx& c) {
#pragma unroll
        for (int r = 0; r < TPG; ++r) {
            if (c.grank == uint32_t(r)) continue;
            uint4 row =
                make_uint4(0u, 0u, 0u, 0u);  // padding token -> zero latent
            if (row_valid(c.p, c.t0 + r)) {
                vec_t pv;
                do {
                    ld_relaxed_16B(
                        pv,
                        c.xbuf +
                            (int64_t(c.parityA) * MAXM + c.t0 + r) * (L * 2),
                        c.tid);
                    const auto b = *reinterpret_cast<const uint4*>(&pv);
                    if ((b.x != 0) & (b.y != 0) & (b.z != 0) & (b.w != 0))
                        break;
                } while (true);
                row = *reinterpret_cast<const uint4*>(&pv);
            }
            *reinterpret_cast<uint4*>(
                reinterpret_cast<uint8_t*>(&c.s_x[r * kXStride]) + c.tid * 16) =
                row;
        }
    }

    __device__ static __forceinline__ uint32_t round_c_vec(const Ctx& c,
                                                           int j) {
        if (j < kOwnHV) return c.p.rank * kShardHV + c.grank * kOwnHV + j;
        const uint32_t q = c.grank * kNonOwnHV + (j - kOwnHV);
        return q < c.p.rank * kShardHV ? q : q + kShardHV;
    }

    // Round-C input prefetch
    __device__ static __forceinline__ vec_t prefetch_round_c(const Ctx& c) {
        vec_t vpre;
        if constexpr (kPrefetchC) {
            const int tt = int(c.tid) / kPer, j = int(c.tid) % kPer;
            if (int(c.tid) < TPG * kPer && row_valid(c.p, c.t0 + tt)) {
                ld_global_16B(vpre, c.p.shared + int64_t(c.t0 + tt) * H,
                              round_c_vec(c, j));
            }
        }
        return vpre;
    }

    __device__ static __forceinline__ void round_b(const Ctx& c) {
        const P& p = c.p;
        const int64_t row0 =
            int64_t(p.rank) * kShard + int64_t(c.grank) * kRows;
        if constexpr (STAGE != 0) {
            mbar_wait_phase0(c.s_wbar);
            shared_weight_mma_gemv_fp8<L, TPG, kRows, BLKB, THREADS,
                                       SCALE_BLOCK, true>(
                nullptr, p.fc2_scale + (row0 / SCALE_BLOCK) * (L / SCALE_BLOCK),
                reinterpret_cast<const uint16_t*>(c.s_x), kXStride, c.s_part,
                c.tid >> 5, c.tid & 31, c.s_wstage, L + TC::kWStagePad);
        } else {
            shared_weight_mma_gemv_fp8<L, TPG, kRows, BLKB, THREADS,
                                       SCALE_BLOCK>(
                p.fc2_w8 + row0 * L,
                p.fc2_scale + (row0 / SCALE_BLOCK) * (L / SCALE_BLOCK),
                reinterpret_cast<const uint16_t*>(c.s_x), kXStride, c.s_part,
                c.tid >> 5, c.tid & 31);
        }
        c.sync();
        for (int i = c.tid; i < TPG * kRows; i += kThreads) {
            float a = 0.f;
#pragma unroll
            for (int ks = 0; ks < Cfg::kKSplit; ++ks)
                a += c.s_part[ks * (TPG * kRows) + i];
            c.s_col[i] = a;
        }
        c.sync();
    }

    // Round C: distributed combine (shard result + shared partial) + AllReduce,
    __device__ static __forceinline__ void round_c(const Ctx& c,
                                                   const vec_t& vpre) {
        const P& p = c.p;
        for (int i = int(c.tid); i < TPG * kPer; i += int(kThreads)) {
            const int tt = i / kPer, j = i % kPer;
            const uint32_t t = c.t0 + tt;
            if (t >= p.m) continue;
            const bool owned = j < kOwnHV;
            const uint32_t hv = round_c_vec(c, j);
            vec_t v;
            if (!row_valid(p, t)) {
                v.fill(__nv_bfloat162{get_pos_zero<__nv_bfloat16>(),
                                      get_pos_zero<__nv_bfloat16>()});
                st_global_16B(v, p.out + int64_t(t) * H, hv);
                continue;
            }
            if constexpr (kPrefetchC) {
                v = vpre;
            } else {
                ld_global_16B(v, p.shared + int64_t(t) * H, hv);
            }
            if (owned) {
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const auto a = cast<float2>(v[e]);
                    v[e] = cast<__nv_bfloat162>(float2{
                        a.x + c.s_col[tt * kRows + j * kVec + 2 * e],
                        a.y + c.s_col[tt * kRows + j * kVec + 2 * e + 1]});
                }
            }
            v = push_reduce<WORLD, L, H, MAXM>(p, v, /*round=*/1, c.parityC, t,
                                               H, hv);
            st_global_16B(v, p.out + int64_t(t) * H, hv);
        }
    }

    template <class Timer>
    __device__ static __forceinline__ void run(const P& p, uint32_t cta,
                                               uint32_t tid, uint8_t* scratch,
                                               Sync sync,
                                               uint32_t launch_parity,
                                               Timer timer) {
        Ctx c = make_ctx(p, cta, tid, scratch, sync);
        const bool active = group_active(c);
        if (active) stage_weights(c);
        c.parityA = c.parityC = int(launch_parity & 1u);
        clear_xbuf(c);
        if (!active) {
            zero_inactive_group(c);
            return;
        }

        const auto span_a = KERNEL_TIMER_BEGIN(timer, "tail_round_a");
        if (c.grank < TPG) round_a_produce(c);
        round_a_gather(c);
        sync();
        KERNEL_TIMER_END(timer, span_a);
        const vec_t vpre = prefetch_round_c(c);
        const auto span_b = KERNEL_TIMER_BEGIN(timer, "tail_round_b");
        round_b(c);
        KERNEL_TIMER_END(timer, span_b);
        const auto span_c = KERNEL_TIMER_BEGIN(timer, "tail_round_c");
        round_c(c, vpre);
        KERNEL_TIMER_END(timer, span_c);
    }
};

template <int WORLD, int L, int H, int MAXM, class TC, class Sync, class Timer>
__device__ __forceinline__ void eb5_tail_group_body(
    const Params<WORLD, L, H, MAXM>& p, uint32_t cta, uint32_t tid,
    uint8_t* scratch, Sync sync, uint32_t launch_parity, Timer timer) {
    TailGroup<WORLD, L, H, MAXM, TC, Sync>::run(p, cta, tid, scratch, sync,
                                                launch_parity, timer);
}

}  // namespace tail
}  // namespace halops::moe
