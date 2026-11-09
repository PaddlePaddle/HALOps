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

#include "moe_tail_prims.cuh"

namespace halops::moe {
namespace tail {
template <int ROWS, int L, int PAD>
constexpr int stage_smem_bytes() {
    return ROWS * (L + PAD);
}

template <class TC, int L>
struct Fp8Block128 {
    using Scale = float;
    using Frag = uint4;
    static constexpr int kTigStride = 16;
    static constexpr int SCALE_BLOCK = TC::kScaleBlock;
    static constexpr int KR = L / TC::kKSplit;
    static constexpr int kNG = KR / SCALE_BLOCK;
    static_assert(SCALE_BLOCK == 128, "scale group = two 64-wide super-blocks");
    static_assert(TC::kRows <= SCALE_BLOCK,
                  "CTA rows must fit in one scale block");
    static_assert(KR % SCALE_BLOCK == 0 && (KR / 64) % TC::kBlkB == 0 &&
                      TC::kBlkB % 2 == 0,
                  "K blocks must tile into batches");

    struct State {
        float sc[kNG];
        float4 g0, g1;
    };

    __device__ static State init(const float* scale, int /*row*/, int ks) {
        State s;
#pragma unroll
        for (int g = 0; g < kNG; ++g) s.sc[g] = scale[ks * kNG + g];
        s.g0 = s.g1 = make_float4(0.f, 0.f, 0.f, 0.f);
        return s;
    }

    template <bool STAGED>
    __device__ static Frag load_w(const uint8_t* p, int sb) {
        Frag f;
        if constexpr (STAGED) {
            f = *reinterpret_cast<const uint4*>(p + sb * 64);
        } else {
            ld_global_16B(f, p + sb * 64, 0);
        }
        return f;
    }

    __device__ static void load_x(const uint16_t* xs, int sb, bool has_x,
                                  uint32_t (&xb)[8]) {
        uint4 xlo = make_uint4(0u, 0u, 0u, 0u), xhi = xlo;
        if (has_x) {
            xlo = *reinterpret_cast<const uint4*>(xs + sb * 64);
            xhi = *reinterpret_cast<const uint4*>(xs + sb * 64 + 8);
        }
        xb[0] = xlo.x, xb[1] = xlo.y, xb[2] = xlo.z, xb[3] = xlo.w;
        xb[4] = xhi.x, xb[5] = xhi.y, xb[6] = xhi.z, xb[7] = xhi.w;
    }

    __device__ static void step(State& s, float4& total, int sb, const Frag& A,
                                const Frag& B, const uint32_t (&xb)[8]) {
        const uint32_t aw[4] = {A.x, A.y, A.z, A.w},
                       bw[4] = {B.x, B.y, B.z, B.w};
#pragma unroll
        for (int mm = 0; mm < 4; ++mm) {
            uint32_t a0, a1, a2, a3;
            e4m3x4_to_f16x2x2(aw[mm], a0, a2);
            e4m3x4_to_f16x2x2(bw[mm], a1, a3);
            mma_f16_m16n8k16((mm & 1) ? s.g1 : s.g0, a0, a1, a2, a3, xb[2 * mm],
                             xb[2 * mm + 1]);
        }
        if (sb & 1) {
            const float sc = s.sc[sb >> 1];
            total.x += sc * (s.g0.x + s.g1.x);
            total.y += sc * (s.g0.y + s.g1.y);
            total.z += sc * (s.g0.z + s.g1.z);
            total.w += sc * (s.g0.w + s.g1.w);
            s.g0 = s.g1 = make_float4(0.f, 0.f, 0.f, 0.f);
        }
    }
};

template <class TC, int L>
struct Mxfp8 {
    using Scale = uint8_t;
    using Frag = uint4;
    static constexpr int kTigStride = 8;
    static constexpr int KR = L / TC::kKSplit;
    static constexpr int kNG = KR / 32;
    static constexpr int kSfVec = kNG / 16;
    static_assert(KR % 64 == 0 && (KR / 64) % TC::kBlkB == 0,
                  "K blocks must tile into batches");
    static_assert(kNG % 16 == 0, "scales are read as 16-byte vectors");

    struct State {
        uint4 sfa[kSfVec], sfb[kSfVec];
    };

    __device__ static State init(const uint8_t* sf, int row, int ks) {
        State s;
#pragma unroll
        for (int v = 0; v < kSfVec; ++v) {
            s.sfa[v] = __ldg(reinterpret_cast<const uint4*>(
                                 sf + int64_t(row) * (L / 32) + ks * kNG) +
                             v);
            s.sfb[v] = __ldg(reinterpret_cast<const uint4*>(
                                 sf + int64_t(row + 8) * (L / 32) + ks * kNG) +
                             v);
        }
        return s;
    }

    template <bool STAGED>
    __device__ static Frag load_w(const uint8_t* p, int sb) {
        const uint2 lo = *reinterpret_cast<const uint2*>(p + sb * 64);
        const uint2 hi = *reinterpret_cast<const uint2*>(p + sb * 64 + 32);
        return make_uint4(lo.x, lo.y, hi.x, hi.y);
    }

    __device__ static void load_x(const uint16_t* xs, int sb, bool has_x,
                                  uint32_t (&xb)[8]) {
        uint4 xlo = make_uint4(0u, 0u, 0u, 0u), xhi = xlo;
        if (has_x) {
            xlo = *reinterpret_cast<const uint4*>(xs + sb * 64);
            xhi = *reinterpret_cast<const uint4*>(xs + sb * 64 + 32);
        }
        xb[0] = xlo.x, xb[1] = xlo.y, xb[2] = xlo.z, xb[3] = xlo.w;
        xb[4] = xhi.x, xb[5] = xhi.y, xb[6] = xhi.z, xb[7] = xhi.w;
    }

    __device__ static float scale_of(const uint4 (&v)[kSfVec], int g) {
        const uint32_t* u = reinterpret_cast<const uint32_t*>(v);
        return __uint_as_float(((u[g >> 2] >> ((g & 3) * 8)) & 0xffu) << 23);
    }

    __device__ static void step(State& s, float4& total, int sb, const Frag& A,
                                const Frag& B, const uint32_t (&xb)[8]) {
        const uint32_t aw[4] = {A.x, A.y, A.z, A.w},
                       bw[4] = {B.x, B.y, B.z, B.w};
        float4 g[4];
#pragma unroll
        for (int mm = 0; mm < 4; ++mm) {
            g[mm] = make_float4(0.f, 0.f, 0.f, 0.f);
            uint32_t a0, a1, a2, a3;
            e4m3x4_to_f16x2x2(aw[mm], a0, a2);
            e4m3x4_to_f16x2x2(bw[mm], a1, a3);
            mma_f16_m16n8k16(g[mm], a0, a1, a2, a3, xb[2 * mm], xb[2 * mm + 1]);
        }
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const float sA = scale_of(s.sfa, 2 * sb + h);
            const float sB = scale_of(s.sfb, 2 * sb + h);
            total.x += sA * (g[2 * h].x + g[2 * h + 1].x);
            total.y += sA * (g[2 * h].y + g[2 * h + 1].y);
            total.z += sB * (g[2 * h].z + g[2 * h + 1].z);
            total.w += sB * (g[2 * h].w + g[2 * h + 1].w);
        }
    }
};

template <template <class, int> class Recipe, class TC, int L,
          bool STAGED = false>
__device__ __forceinline__ void shared_weight_mma_gemv(
    const uint8_t* __restrict__ w,
    const typename Recipe<TC, L>::Scale* __restrict__ sf, const uint16_t* s_x,
    int x_stride, float* s_part, uint32_t warp, uint32_t lane,
    const uint8_t* s_w = nullptr, int w_stride = 0) {
    using R = Recipe<TC, L>;
    constexpr int TPG = TC::kTpg, ROWS = TC::kRows, BLKB = TC::kBlkB;
    constexpr int KR = L / TC::kKSplit;
    constexpr int kNSB = KR / 64;
    constexpr int kNC = kNSB / BLKB;
    static_assert(TPG >= 1 && TPG <= 8, "mma N is 8");
    const uint32_t grp = lane >> 2, tig = lane & 3;
    const bool has_x = grp < uint32_t(TPG);
#pragma unroll
    for (int i = 0; i < TC::kItemsPerWarp; ++i) {
        const int item = int(warp) + i * int(TC::kWarps);
        const int tile = item % TC::kTiles, ks = item / TC::kTiles;
        const int row = tile * 16 + int(grp);
        const uint8_t* wa = (STAGED ? s_w + row * w_stride + ks * KR
                                    : w + int64_t(row) * L + ks * KR) +
                            tig * R::kTigStride;
        const uint8_t* wb =
            wa + (STAGED ? int64_t(8) * w_stride : int64_t(8) * L);
        const uint16_t* xs =
            s_x + (has_x ? grp : 0) * x_stride + ks * KR + tig * R::kTigStride;
        typename R::State state = R::init(sf, row, ks);
        float4 total = make_float4(0.f, 0.f, 0.f, 0.f);
        auto load_chunk = [&](int c, typename R::Frag(&da)[BLKB],
                              typename R::Frag(&db)[BLKB]) {
#pragma unroll
            for (int k = 0; k < BLKB; ++k) {
                da[k] = R::template load_w<STAGED>(wa, c * BLKB + k);
                db[k] = R::template load_w<STAGED>(wb, c * BLKB + k);
            }
        };
        typename R::Frag ca[BLKB], cb[BLKB];
        load_chunk(0, ca, cb);
#pragma unroll
        for (int c = 0; c < kNC; ++c) {
            typename R::Frag na[BLKB], nb[BLKB];
            if (c + 1 < kNC) load_chunk(c + 1, na, nb);
#pragma unroll
            for (int k = 0; k < BLKB; ++k) {
                const int sb = c * BLKB + k;
                uint32_t xb[8];
                R::load_x(xs, sb, has_x, xb);
                R::step(state, total, sb, ca[k], cb[k], xb);
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

template <int WORLD, int L, int H, int MAXM, class TC, class P, class Sync>
struct TailGroup {
    static constexpr int TPG = TC::kTpg, GSIZE = TC::kGsize, STAGE = TC::kStage,
                         SCALE_BLOCK = TC::kScaleBlock;
    static constexpr uint32_t kVec = 8;
    static constexpr uint32_t kThreads = TC::kThreads;
    static constexpr uint32_t kWarps = TC::kWarps;

    static constexpr int kShard = TC::kShard;
    static constexpr int kRows = TC::kRows;
    static constexpr int kXStride = TC::kXStride;
    static constexpr uint32_t kNumCtas = TC::kNumCtas;
    static constexpr int kXBufBytes = MAXM * L * 2;
    static constexpr int kSliceVec = kXBufBytes / 16 / int(kNumCtas);
    static constexpr int kHV = H / kVec;
    static constexpr int kShardHV = kShard / kVec;
    static constexpr int kOwnHV = kRows / kVec;
    static constexpr int kNonOwnHV = (kHV - kShardHV) / GSIZE;
    static constexpr int kPer = kOwnHV + kNonOwnHV;
    static constexpr bool kPrefetchC = TPG * kPer <= int(kThreads);

    static_assert(kShard == H / WORLD && kXStride == L + TC::kXPad,
                  "TC geometry must match the WORLD/L/H template arguments");
    static_assert(kVec * sizeof(__nv_bfloat16) == sizeof(Vec16),
                  "kVec bf16 fill exactly one 16B vector");
    static_assert(L % (kThreads * kVec) == 0 && L / (kThreads * kVec) == 1,
                  "one CTA covers one L-row");
    static_assert(kShard % GSIZE == 0 && MAXM % TPG == 0, "group tiling");
    static_assert(TPG <= GSIZE && TPG <= 8,
                  "round A needs one CTA per token; mma N = 8");
    static_assert(TC::kMxfp8 ||
                      (kShard % SCALE_BLOCK == 0 && kRows <= SCALE_BLOCK),
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

    // Pointers (inputs, then global and shared-memory scratch) first, then the
    // scalars. make_ctx aggregate-initializes it in this order.
    struct Ctx {
        const P& p;
        Sync sync;
        const DataType::ElementOut* shared;
        uint8_t* xbuf;
        uint8_t* s_wstage;
        uint64_t* s_wbar;
        __nv_bfloat16* s_x;
        float* s_warp;
        float* s_part;
        float* s_col;
        uint32_t m;
        uint32_t cta;
        uint32_t tid;
        uint32_t grank;
        uint32_t t0;
        int parityA;
        int parityC;
    };

    __device__ static __forceinline__ Ctx make_ctx(const P& p, uint32_t cta,
                                                   uint32_t tid,
                                                   uint8_t* scratch,
                                                   Sync sync) {
        return Ctx{p,
                   sync,
                   p.shared_out,
                   p.tail_uc_base[p.tail_rank] + TC::kSlotRegionBytes,
                   scratch + TC::kSmemWStage,
                   reinterpret_cast<uint64_t*>(scratch + TC::kSmemWBar),
                   reinterpret_cast<__nv_bfloat16*>(scratch + TC::kSmemX),
                   reinterpret_cast<float*>(scratch + TC::kSmemWarp),
                   reinterpret_cast<float*>(scratch + TC::kSmemPart),
                   reinterpret_cast<float*>(scratch + TC::kSmemCol),
                   p.num_tokens,
                   cta,
                   tid,
                   cta % GSIZE,
                   (cta / GSIZE) * TPG,
                   0,
                   0};
    }

    __device__ static __forceinline__ bool group_active(const Ctx& c) {
        bool active = false;
#pragma unroll
        for (int r = 0; r < TPG; ++r) active |= c.t0 + r < c.m;
        return active;
    }

    __device__ static __forceinline__ void stage_weights(const Ctx& c) {
        if constexpr (STAGE != 0) {
            const uint8_t* wrow0 =
                reinterpret_cast<const uint8_t*>(c.p.tail_fc2_w8) +
                (int64_t(c.p.tail_rank) * kShard + int64_t(c.grank) * kRows) *
                    L;
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
            Vec16 z;
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
            if (t >= c.m) break;
            Vec16 z;
            z.fill(__nv_bfloat162{get_pos_zero<__nv_bfloat16>(),
                                  get_pos_zero<__nv_bfloat16>()});
            st_global_16B(z, c.p.tail_out + int64_t(t) * H,
                          uint32_t(v % (H / int(kVec))));
        }
    }

    __device__ static __forceinline__ void round_a_produce(const Ctx& c) {
        const P& p = c.p;
        const uint32_t tid = c.tid;
        uint4 xrow = make_uint4(0u, 0u, 0u, 0u);
        const uint32_t t = c.t0 + c.grank;
        if (t < c.m) {
            Vec16 lat = poll_reduce<WORLD, L, H, MAXM>(p, /*round=*/0,
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
            const float inv = rsqrtf(total / float(L) + p.tail_eps);
            Vec16 nw;
            ld_global_16B(nw, p.tail_norm_weight, tid);
            Vec16 normed;
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
        Vec16 pv = *reinterpret_cast<const Vec16*>(&xrow);
        sanitize(pv);
        if (t < c.m)
            st_global_16B(
                pv, c.xbuf + (int64_t(c.parityA) * MAXM + t) * (L * 2), tid);
    }

    __device__ static __forceinline__ void round_a_gather(const Ctx& c) {
#pragma unroll
        for (int r = 0; r < TPG; ++r) {
            if (c.grank == uint32_t(r)) continue;
            uint4 row =
                make_uint4(0u, 0u, 0u, 0u);  // padding token -> zero latent
            if (c.t0 + r < c.m) {
                Vec16 pv;
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
        if (j < kOwnHV) return c.p.tail_rank * kShardHV + c.grank * kOwnHV + j;
        const uint32_t q = c.grank * kNonOwnHV + (j - kOwnHV);
        return q < c.p.tail_rank * kShardHV ? q : q + kShardHV;
    }

    // Round-C input prefetch
    __device__ static __forceinline__ Vec16 prefetch_round_c(const Ctx& c) {
        Vec16 vpre;
        if constexpr (kPrefetchC) {
            const int tt = int(c.tid) / kPer, j = int(c.tid) % kPer;
            if (int(c.tid) < TPG * kPer && c.t0 + tt < c.m) {
                ld_global_16B(vpre, c.shared + int64_t(c.t0 + tt) * H,
                              round_c_vec(c, j));
            }
        }
        return vpre;
    }

    __device__ static __forceinline__ void round_b(const Ctx& c) {
        const P& p = c.p;
        const int64_t row0 =
            int64_t(p.tail_rank) * kShard + int64_t(c.grank) * kRows;
        // The fc2 weight format is a compile-time property of the schedule.
        if constexpr (STAGE != 0) mbar_wait_phase0(c.s_wbar);
        const uint16_t* s_x = reinterpret_cast<const uint16_t*>(c.s_x);
        const uint8_t* w_global =
            reinterpret_cast<const uint8_t*>(p.tail_fc2_w8) + row0 * L;
        if constexpr (TC::kMxfp8) {
            const uint8_t* sf = p.tail_fc2_scale + row0 * (L / 32);
            if constexpr (STAGE != 0) {
                shared_weight_mma_gemv<Mxfp8, TC, L, true>(
                    nullptr, sf, s_x, kXStride, c.s_part, c.tid >> 5,
                    c.tid & 31, c.s_wstage, L + TC::kWStagePad);
            } else {
                shared_weight_mma_gemv<Mxfp8, TC, L>(w_global, sf, s_x,
                                                     kXStride, c.s_part,
                                                     c.tid >> 5, c.tid & 31);
            }
        } else {
            const float* sf =
                p.tail_fc2_scale + (row0 / SCALE_BLOCK) * (L / SCALE_BLOCK);
            if constexpr (STAGE != 0) {
                shared_weight_mma_gemv<Fp8Block128, TC, L, true>(
                    nullptr, sf, s_x, kXStride, c.s_part, c.tid >> 5,
                    c.tid & 31, c.s_wstage, L + TC::kWStagePad);
            } else {
                shared_weight_mma_gemv<Fp8Block128, TC, L>(
                    w_global, sf, s_x, kXStride, c.s_part, c.tid >> 5,
                    c.tid & 31);
            }
        }
        c.sync();
        for (int i = c.tid; i < TPG * kRows; i += kThreads) {
            float a = 0.f;
#pragma unroll
            for (int ks = 0; ks < TC::kKSplit; ++ks)
                a += c.s_part[ks * (TPG * kRows) + i];
            c.s_col[i] = a;
        }
        c.sync();
    }

    // Round C: distributed combine (shard result + shared partial) + AllReduce,
    __device__ static __forceinline__ void round_c(const Ctx& c,
                                                   const Vec16& vpre) {
        const P& p = c.p;
        for (int i = int(c.tid); i < TPG * kPer; i += int(kThreads)) {
            const int tt = i / kPer, j = i % kPer;
            const uint32_t t = c.t0 + tt;
            if (t >= c.m) continue;
            const bool owned = j < kOwnHV;
            const uint32_t hv = round_c_vec(c, j);
            Vec16 v;
            if constexpr (kPrefetchC) {
                v = vpre;
            } else {
                ld_global_16B(v, c.shared + int64_t(t) * H, hv);
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
            st_global_16B(v, p.tail_out + int64_t(t) * H, hv);
        }
    }

    // The fused tail runs three rounds over the latent exchange buffer:
    //   A  AllReduce the WORLD partials of each token's routed latent,
    //      RMSNorm the sum, and exchange the normalized rows among the TPG
    //      CTAs that share this token group.
    //   B  fc2 GEMV: this CTA's kRows output rows of the hidden shard.
    //   C  Add the shared-expert partial, AllReduce the full hidden band
    //      across WORLD, and store `tail_out`.
    __device__ static __forceinline__ void run(const P& p, uint32_t cta,
                                               uint32_t tid, uint8_t* scratch,
                                               Sync sync,
                                               uint32_t launch_parity) {
        Ctx c = make_ctx(p, cta, tid, scratch, sync);
        const bool active = group_active(c);
        if (active) stage_weights(c);
        c.parityA = c.parityC = int(launch_parity & 1u);
        clear_xbuf(c);
        if (!active) {
            zero_inactive_group(c);
            return;
        }

        if (c.grank < TPG) round_a_produce(c);
        round_a_gather(c);
        sync();
        const Vec16 vpre = prefetch_round_c(c);
        round_b(c);
        round_c(c, vpre);
    }
};

template <int WORLD, int L, int H, int MAXM, class TC, class P, class Sync>
__device__ __forceinline__ void eb5_tail_group_body(const P& p, uint32_t cta,
                                                    uint32_t tid,
                                                    uint8_t* scratch, Sync sync,
                                                    uint32_t launch_parity) {
    TailGroup<WORLD, L, H, MAXM, TC, P, Sync>::run(p, cta, tid, scratch, sync,
                                                   launch_parity);
}

}  // namespace tail
}  // namespace halops::moe
