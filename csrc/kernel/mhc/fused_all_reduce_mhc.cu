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
#include <cutlass/arch/barrier.h>
#include <cutlass/kernel_hardware_info.h>

#include <algorithm>
#include <cute/arch/cluster_sm90.hpp>
#include <cutlass/cluster_launch.hpp>
#include <optional>
#include <vector>

#include "common/activation.cuh"
#include "common/cuda_utils.h"
#include "common/mx_quant.cuh"
#include "common/pdl.cuh"
#include "common/ptx.cuh"
#include "kernel/mhc/megallmhc_utils.cuh"
#include "kernel/mla/mla_intrinsics.cuh"
#include "kernel/moe/moe_tail_prims.cuh"
#include "kernel/moe/ptx_utils.cuh"

namespace halops {

namespace {

using moe::tail::Vec16;

constexpr int kMaxWorld = 8;
constexpr int kMaxTokensPerCluster = 4;
constexpr int kMaxPrefetchRanges = 8;

template <int kHc>
struct MhcShape {
    static constexpr int kMix = kHc * (kHc + 2);
    static constexpr int kGram = kHc * (kHc + 1) / 2;
    static constexpr int kPartials = kMix + kGram;
    static constexpr int kPartialsPad = (kPartials + 3) / 4 * 4;
    static constexpr int kPV = kPartialsPad / 4;
    static constexpr int kPc = kHc + kHc * kHc;
    static constexpr int kComb = kHc * kHc;
};

template <int kHidden>
struct MhcTile {
    static constexpr int kClusterSize = 16;
    static constexpr int kCols = kHidden / kClusterSize;
    static constexpr int kWarps = kCols / cutlass::NumThreadsPerWarp;
    static constexpr int kVec = sizeof(Vec16) / sizeof(__nv_bfloat16);
    static constexpr int kRowVecs = kCols / kVec;
    static constexpr int kBlock = kCols + cutlass::NumThreadsPerWarp;
    static constexpr float kInvHidden = 1.0f / kHidden;
};

struct LamportPlane {
    uint8_t* peers[kMaxWorld];
    uint32_t* counters;
    int64_t slot_bytes;
    int world;
    int rank;
    int num_counters;
};

struct MhcParams {
    const __nv_bfloat16* __restrict__ x;
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
    float* __restrict__ mix_out;
    uint8_t* __restrict__ q_out;
    uint8_t* __restrict__ sf_out;
    int num_tokens;
    int sinkhorn_repeat;
    bool write_logits;
    float rms_eps;
    float hc_pre_eps;
    float hc_sinkhorn_eps;
    float hc_post_mult;
    float norm_eps;
    LamportPlane plane;
};

template <int kHc, int kHidden, int kTPC>
struct MhcSmem {
    using S = MhcShape<kHc>;
    using L = MhcTile<kHidden>;
    alignas(16) __nv_bfloat16 xs[kTPC][L::kCols];
    alignas(16) __nv_bfloat16 rs[kTPC][kHc][L::kCols];
    alignas(16) float pc[kTPC][S::kPc];
    alignas(16) float red[L::kWarps][kTPC][S::kPartialsPad];
    alignas(16) float cl[L::kClusterSize][kTPC][S::kPartialsPad];
    float tot[kTPC][S::kPartials];
    float hcs[3];
    float hcb[S::kMix];
    alignas(8) uint64_t bar;
    uint64_t tot_bar;
};

struct L2PrefetchArgs {
    uint64_t addr[kMaxPrefetchRanges];
    uint32_t bytes[kMaxPrefetchRanges];
};

template <int kHc, int kHidden>
CUTE_DEVICE float token_rms(const float* tot, float eps) {
    constexpr float kInvN = 1.0f / (kHc * kHidden);
    float d[kHc];
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < kHc; ++j)
        d[j] = tot[MhcShape<kHc>::kMix + j * kHc - j * (j - 1) / 2];
    CUTLASS_PRAGMA_UNROLL
    for (int n = kHc; n > 1; n = (n + 1) / 2) {
        CUTLASS_PRAGMA_UNROLL
        for (int k = 0; k < n / 2; ++k)
            d[k] = __fadd_rn(d[2 * k], d[2 * k + 1]);
        if (n % 2) d[n / 2] = d[n - 1];
    }
    return __frcp_rn(__fadd_rn(__fsqrt_rn(__fmul_rn(d[0], kInvN)), eps));
}

template <int kN, int kMask>
struct ReduceScatter {
    static constexpr int kHalf = (kN + 1) / 2;
    using Next = ReduceScatter<kHalf, kMask / 2>;
    static constexpr int kOut = Next::kOut;

    CUTE_DEVICE static void run(const float (&in)[kN], float (&out)[kOut],
                                int lane) {
        const bool hi = lane & kMask;
        float half[kHalf];
        CUTLASS_PRAGMA_UNROLL
        for (int k = 0; k < kHalf; ++k) {
            const float lo_v = in[k];
            const float hi_v = kHalf + k < kN ? in[kHalf + k] : 0.0f;
            const float send = hi ? lo_v : hi_v;
            const float keep = hi ? hi_v : lo_v;
            half[k] = __fadd_rn(keep, __shfl_xor_sync(~0u, send, kMask));
        }
        Next::run(half, out, lane);
    }

    CUTE_DEVICE static int index(int k, int lane) {
        const int p = Next::index(k, lane);
        if (p < 0) return -1;
        const int q = (lane & kMask ? kHalf : 0) + p;
        return q < kN ? q : -1;
    }
};

template <int kN>
struct ReduceScatter<kN, 0> {
    static constexpr int kOut = kN;

    CUTE_DEVICE static void run(const float (&in)[kN], float (&out)[kOut],
                                int) {
        CUTLASS_PRAGMA_UNROLL
        for (int k = 0; k < kN; ++k) out[k] = in[k];
    }

    CUTE_DEVICE static int index(int k, int) { return k; }
};

template <int kHidden, int kHc>
CUTE_DEVICE void load_weights(const MhcParams& p, int h,
                              float (&w)[MhcShape<kHc>::kMix][kHc],
                              __nv_bfloat16& wn) {
    constexpr int kStride = kHc * kHidden;
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < kHc; ++j) {
        const float* fn = p.fn + j * kHidden + h;
        CUTLASS_PRAGMA_UNROLL
        for (int m = 0; m < MhcShape<kHc>::kMix; ++m, fn += kStride)
            w[m][j] = __ldg(fn);
    }
    wn = p.norm_weight[h];
}

template <bool kWithX, bool kCombLogits, int kHidden, int kHc, int kTPC>
CUTE_DEVICE void load_inputs(const MhcParams& p,
                             MhcSmem<kHc, kHidden, kTPC>& sm, int tok0, int nt,
                             int col0, int lt, int nthr) {
    using S = MhcShape<kHc>;
    constexpr int kRowVecs = MhcTile<kHidden>::kRowVecs;
    constexpr int kFirst = kWithX ? 0 : 1;
    constexpr int kRows = 1 + kHc - kFirst;
    for (int i = lt; i < nt * kRows * kRowVecs; i += nthr) {
        const int q = i / kRowVecs, k = i % kRowVecs;
        const int t = q / kRows, r = kFirst + q % kRows;
        const int64_t tok = tok0 + t;
        const __nv_bfloat16* src =
            r == 0 ? p.x + tok * kHidden
                   : p.residual + (tok * kHc + r - 1) * kHidden;
        __nv_bfloat16* dst = r == 0 ? sm.xs[t] : sm.rs[t][r - 1];
        reinterpret_cast<uint4*>(dst)[k] =
            reinterpret_cast<const uint4*>(src + col0)[k];
    }
    constexpr int kPcLoad = kCombLogits ? kHc : S::kPc;
    for (int i = lt; i < nt * kPcLoad; i += nthr) {
        const int t = i / kPcLoad, e = i % kPcLoad;
        const int64_t tok = tok0 + t;
        sm.pc[t][e] =
            e < kHc ? p.post[tok * kHc + e] : p.comb[tok * S::kComb + e - kHc];
    }
}

template <int kHidden, int kHc, int kTPC>
CUTE_DEVICE void post_residual_and_reduce(
    const MhcParams& p, MhcSmem<kHc, kHidden, kTPC>& sm,
    const float (&w)[MhcShape<kHc>::kMix][kHc], int t, int64_t tok, int h,
    int tid, int lane, int warp, float (&rf)[kHc]) {
    using S = MhcShape<kHc>;
    const float* pc = sm.pc[t];
    const float xh = __bfloat162float(sm.xs[t][tid]);
    const float2 hin = mhc::splat(xh);
    float2 r[kHc];
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kHc; ++i) {
        r[i] = mhc::splat(__bfloat162float(sm.rs[t][i][tid]));
    }
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < kHc; j += 2) {
        const int j1 = min(j + 1, kHc - 1);
        float2 acc = __ffma2_rn(
            make_float2(pc[j], pc[j1]), hin,
            __fmul2_rn(make_float2(pc[kHc + j], pc[kHc + j1]), r[0]));
        CUTLASS_PRAGMA_UNROLL
        for (int i = 1; i < kHc; ++i) {
            const float* ci = pc + kHc + i * kHc;
            acc = __ffma2_rn(make_float2(ci[j], ci[j1]), r[i], acc);
        }
        const __nv_bfloat162 rb = __float22bfloat162_rn(acc);
        p.residual_out[(tok * kHc + j) * kHidden + h] = rb.x;
        rf[j] = __bfloat162float(rb.x);
        if (j + 1 < kHc) {
            p.residual_out[(tok * kHc + j + 1) * kHidden + h] = rb.y;
            rf[j + 1] = __bfloat162float(rb.y);
        }
    }
    float pv[S::kPartials];
    CUTLASS_PRAGMA_UNROLL
    for (int m = 0; m < S::kMix; m += 2) {
        const int m1 = min(m + 1, S::kMix - 1);
        float2 a =
            __fmul2_rn(make_float2(w[m][0], w[m1][0]), mhc::splat(rf[0]));
        CUTLASS_PRAGMA_UNROLL
        for (int j = 1; j < kHc; ++j) {
            a = __ffma2_rn(make_float2(w[m][j], w[m1][j]), mhc::splat(rf[j]),
                           a);
        }
        pv[m] = a.x;
        if (m + 1 < S::kMix) pv[m + 1] = a.y;
    }
    float ga[S::kGram], gb[S::kGram];
    int g_idx = 0;
    CUTLASS_PRAGMA_UNROLL
    for (int a = 0; a < kHc; ++a) {
        CUTLASS_PRAGMA_UNROLL
        for (int b = a; b < kHc; ++b) {
            ga[g_idx] = rf[a];
            gb[g_idx++] = rf[b];
        }
    }
    CUTLASS_PRAGMA_UNROLL
    for (int n = 0; n < S::kGram; n += 2) {
        const int n1 = min(n + 1, S::kGram - 1);
        const float2 g =
            __fmul2_rn(make_float2(ga[n], ga[n1]), make_float2(gb[n], gb[n1]));
        pv[S::kMix + n] = g.x;
        if (n + 1 < S::kGram) pv[S::kMix + n + 1] = g.y;
    }
    using RS = ReduceScatter<S::kPartials, cutlass::NumThreadsPerWarp / 2>;
    float s[RS::kOut];
    RS::run(pv, s, lane);
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < RS::kOut; ++k) {
        const int idx = RS::index(k, lane);
        if (idx >= 0) sm.red[warp][t][idx] = s[k];
    }
}

template <int kHidden, int kHc, int kTPC>
CUTE_DEVICE void token_scalars(const MhcParams& p,
                               const MhcSmem<kHc, kHidden, kTPC>& sm, int tok0,
                               int nt, int lane, bool store,
                               float (&pre_out)[kTPC][kHc],
                               float (&rn_out)[kTPC]) {
    using S = MhcShape<kHc>;
    constexpr int kLanes = 2 * kHc;
    static_assert(kLanes * kTPC <= cutlass::NumThreadsPerWarp);
    const int t = min(lane / kLanes, nt - 1), q = lane % kLanes, j = q % kHc;
    const bool is_post = q >= kHc;
    const float* tot = sm.tot[t];
    const int e = is_post ? kHc + j : j;
    const float rms = token_rms<kHc, kHidden>(tot, p.rms_eps);
    const float s =
        sigmoid_rn(fmaf(tot[e] * rms, sm.hcs[is_post ? 1 : 0], sm.hcb[e]));
    const float pre_j = s + p.hc_pre_eps;
    float pre[kHc];
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < kHc; ++k)
        pre[k] = __shfl_sync(~0u, pre_j, t * kLanes + k);
    float ysq = 0.0f;
    CUTLASS_PRAGMA_UNROLL
    for (int a = 0; a < kHc; ++a) {
        CUTLASS_PRAGMA_UNROLL
        for (int b = 0; b < kHc; ++b) {
            const int lo = min(a, b), hi = max(a, b);
            const float g =
                tot[S::kMix + lo * kHc - lo * (lo - 1) / 2 + hi - lo];
            ysq = fmaf(pre[a] * g, pre[b], ysq);
        }
    }
    const float rn = rsqrtf(
        __fadd_rn(__fmul_rn(ysq, MhcTile<kHidden>::kInvHidden), p.norm_eps));
    CUTLASS_PRAGMA_UNROLL
    for (int u = 0; u < kTPC; ++u) {
        CUTLASS_PRAGMA_UNROLL
        for (int k = 0; k < kHc; ++k) {
            pre_out[u][k] =
                kTPC == 1 ? pre[k] : __shfl_sync(~0u, pre_j, kLanes * u + k);
        }
        rn_out[u] = kTPC == 1 ? rn : __shfl_sync(~0u, rn, kLanes * u);
    }
    if (!store || lane / kLanes >= nt) return;
    const int64_t tok = tok0 + t;
    if (is_post) p.post_out[tok * kHc + j] = s * p.hc_post_mult;
    if (p.write_logits) {
        for (int k = q; k < kHc * kHc; k += kLanes) {
            p.mix_out[tok * kHc * kHc + k] =
                fmaf(tot[2 * kHc + k] * rms, sm.hcs[2], sm.hcb[2 * kHc + k]);
        }
    }
}

template <int kHidden, int kHc>
CUTE_DEVICE void norm_quant_out(const MhcParams& p, int64_t tok, int h,
                                int lane, __nv_bfloat16 wn,
                                const float (&rf)[kHc], const float (&pre)[kHc],
                                float rn) {
    float y = __fmul_rn(pre[0], rf[0]);
    CUTLASS_PRAGMA_UNROLL
    for (int j = 1; j < kHc; ++j) y = fmaf(pre[j], rf[j], y);
    const float yb = __bfloat162float(__float2bfloat16_rn(y));
    const __nv_bfloat16 ob =
        __float2bfloat16_rn(__fmul_rn(__fmul_rn(yb, rn), __bfloat162float(wn)));
    p.layer_out[tok * kHidden + h] = ob;
    const float o = __bfloat162float(ob);
    float amax = fabsf(o);
    CUTLASS_PRAGMA_UNROLL
    for (int m = cutlass::NumThreadsPerWarp / 2; m > 0; m /= 2)
        amax = fmaxf(amax, __shfl_xor_sync(~0u, amax, m));
    const uint32_t e = mxfp8::scale_of_exp8(
        mxfp8::exp8_pair(__float2bfloat162_rn(amax)) & 0xffu);
    const float inv = __uint_as_float(mxfp8::multiplier_bits(e) << 16);
    const float v = fminf(fmaxf(o * inv, -mxfp8::kE4m3Max), mxfp8::kE4m3Max);
    p.q_out[tok * kHidden + h] =
        __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
    if (lane == 0)
        p.sf_out[tok * (kHidden / mxfp8::kGroup) + h / mxfp8::kGroup] = e;
}

template <int kHc, int kHidden, int kTPC, bool kFuseAR, bool kCombLogits>
__global__ void __launch_bounds__(MhcTile<kHidden>::kBlock, 1)
    fused_all_reduce_mhc_kernel(const MhcParams __grid_constant__ p) {
    using S = MhcShape<kHc>;
    using L = MhcTile<kHidden>;
    using cutlass::arch::NamedBarrier;
    extern __shared__ __align__(16) uint8_t smem_raw[];
    auto& sm = *reinterpret_cast<MhcSmem<kHc, kHidden, kTPC>*>(smem_raw);
    const int tid = threadIdx.x, lane = tid % cutlass::NumThreadsPerWarp;
    const int warp = tid / cutlass::NumThreadsPerWarp;
    const bool compute = tid < L::kCols;
    const int rank = cute::block_rank_in_cluster();
    const int col0 = rank * L::kCols, h = col0 + tid;
    const int blk = blockIdx.y * L::kClusterSize + rank;
    const int tok0 = blockIdx.y * kTPC;
    const int nt = min(kTPC, p.num_tokens - tok0);
    const uint32_t bar = cute::cast_smem_ptr_to_uint(&sm.bar);
    const LamportPlane& pl = p.plane;
    int64_t phase_off = 0;

    float w[S::kMix][kHc];
    __nv_bfloat16 wn{};
    if constexpr (!kFuseAR) {
        if (compute) load_weights<kHidden>(p, h, w, wn);
        for (int i = tid; i < 3 + S::kMix; i += L::kBlock) {
            if (i < 3) {
                sm.hcs[i] = p.hc_scale[i];
            } else {
                sm.hcb[i - 3] = p.hc_base[i - 3];
            }
        }
    }
    constexpr int kPad = S::kPartialsPad - S::kPartials;
    if (tid < L::kWarps * kTPC * kPad)
        sm.red[tid / (kTPC * kPad)][tid / kPad % kTPC]
              [S::kPartials + tid % kPad] = 0;
    if (tid == 0) {
        cutlass::arch::ClusterTransactionBarrier::init(&sm.bar, 1);
        cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(
            &sm.bar, L::kClusterSize * nt * S::kPartialsPad * sizeof(float));
        cutlass::arch::ClusterBarrier::init(&sm.tot_bar, 1);
        cutlass::arch::fence_barrier_init();
    }
    cute::cluster_arrive_relaxed();
    pdl_wait();

    if (compute) {
        if constexpr (kFuseAR) {
            const int active = nt * L::kRowVecs;
            const int pushers =
                cute::round_up(active, int{cutlass::NumThreadsPerWarp});
            if (tid < pushers) {
                phase_off =
                    int64_t(pl.counters[blk] & 1) * pl.world * pl.slot_bytes;
                const int q = tid / L::kRowVecs, k = tid % L::kRowVecs;
                const int64_t vid = int64_t(tok0 + q) * (kHidden / L::kVec) +
                                    col0 / L::kVec + k;
                if (tid < active) {
                    Vec16 v;
                    moe::ld_global_16B(v, p.x, vid);
                    moe::tail::sanitize(v);
                    const int64_t src_off = pl.rank * pl.slot_bytes + phase_off;
                    CUTLASS_PRAGMA_UNROLL
                    for (int d = 0; d < kMaxWorld; ++d) {
                        if (d < pl.world)
                            st_relaxed_16B(v, pl.peers[d] + src_off, vid);
                    }
                }
                pdl_trigger();
                NamedBarrier::arrive(L::kCols, 0);
                load_weights<kHidden>(p, h, w, wn);
                if (tid < active) {
                    const uint8_t* slots = pl.peers[pl.rank] + phase_off;
                    Vec16 v[kMaxWorld];
                    bool pending;
                    do {
                        pending = false;
                        CUTLASS_PRAGMA_UNROLL
                        for (int s = 0; s < kMaxWorld; ++s) {
                            if (s < pl.world) {
                                moe::ld_relaxed_16B(
                                    v[s], slots + s * pl.slot_bytes, vid);
                            }
                        }
                        CUTLASS_PRAGMA_UNROLL
                        for (int s = 0; s < kMaxWorld; ++s) {
                            const uint4 b =
                                reinterpret_cast<const uint4&>(v[s]);
                            if (s < pl.world)
                                pending |= !(b.x && b.y && b.z && b.w);
                        }
                    } while (pending);
                    Vec16 sum;
                    CUTLASS_PRAGMA_UNROLL
                    for (int j = 0; j < 4; ++j) {
                        float2 acc = __bfloat1622float2(v[0][j]);
                        CUTLASS_PRAGMA_UNROLL
                        for (int s = 1; s < kMaxWorld; ++s) {
                            if (s < pl.world) {
                                acc = __fadd2_rn(acc,
                                                 __bfloat1622float2(v[s][j]));
                            }
                        }
                        sum[j] = __float22bfloat162_rn(acc);
                    }
                    reinterpret_cast<Vec16*>(sm.xs[q])[k] = sum;
                }
            } else {
                NamedBarrier::sync(L::kCols, 0);
                const int lt = tid - pushers, nl = L::kCols - pushers;
                if (blk == 0) {
                    for (int i = L::kClusterSize * gridDim.y + lt;
                         i < pl.num_counters; i += nl)
                        atomicAdd(pl.counters + i, 1u);
                }
                load_weights<kHidden>(p, h, w, wn);
                load_inputs<false, kCombLogits, kHidden>(p, sm, tok0, nt, col0,
                                                         lt, nl);
                for (int i = lt; i < 3 + S::kMix; i += nl) {
                    if (i < 3) {
                        sm.hcs[i] = p.hc_scale[i];
                    } else {
                        sm.hcb[i - 3] = p.hc_base[i - 3];
                    }
                }
            }
        } else {
            load_inputs<true, kCombLogits, kHidden>(p, sm, tok0, nt, col0, tid,
                                                    L::kCols);
        }
    } else {
        if constexpr (kFuseAR) {
            phase_off =
                int64_t(pl.counters[blk] & 1) * pl.world * pl.slot_bytes;
        }
        if constexpr (kCombLogits) {
            if (lane < nt) {
                const float* src = p.comb + int64_t(tok0 + lane) * S::kComb;
                float c[S::kComb];
                CUTLASS_PRAGMA_UNROLL
                for (int k = 0; k < S::kComb; ++k) c[k] = src[k];
                mhc::sinkhorn<kHc, true>(c, p.hc_sinkhorn_eps,
                                         p.sinkhorn_repeat);
                CUTLASS_PRAGMA_UNROLL
                for (int k = 0; k < S::kComb; ++k) sm.pc[lane][kHc + k] = c[k];
            }
        }
    }
    __syncthreads();

    if (compute) {
        float rf[kTPC][kHc] = {};
        CUTLASS_PRAGMA_UNROLL
        for (int t = 0; t < kTPC; ++t) {
            if (t < nt)
                post_residual_and_reduce<kHidden>(p, sm, w, t, tok0 + t, h, tid,
                                                  lane, warp, rf[t]);
        }
        NamedBarrier::sync(L::kCols, 0);
        cute::cluster_wait();
        for (int i = tid; i < L::kClusterSize * nt * S::kPV; i += L::kCols) {
            const int dst = i % L::kClusterSize, v = i / L::kClusterSize;
            const int t = v / S::kPV, k = v % S::kPV;
            float4 r = reinterpret_cast<const float4&>(sm.red[0][t][4 * k]);
            float2 lo = make_float2(r.x, r.y), hi = make_float2(r.z, r.w);
            CUTLASS_PRAGMA_UNROLL
            for (int wi = 1; wi < L::kWarps; ++wi) {
                r = reinterpret_cast<const float4&>(sm.red[wi][t][4 * k]);
                lo = __fadd2_rn(lo, make_float2(r.x, r.y));
                hi = __fadd2_rn(hi, make_float2(r.z, r.w));
            }
            st_async_v4(
                cute::set_block_rank(
                    cute::cast_smem_ptr_to_uint(&sm.cl[rank][t][4 * k]), dst),
                make_float4(lo.x, lo.y, hi.x, hi.y),
                cute::set_block_rank(bar, dst));
        }
        mla::mbar_wait_cluster(bar, 0);
        for (int i = tid; i < nt * S::kPartials; i += L::kCols) {
            const int t = i / S::kPartials, v = i % S::kPartials;
            float s = sm.cl[0][t][v];
            CUTLASS_PRAGMA_UNROLL
            for (int r = 1; r < L::kClusterSize; ++r)
                s = __fadd_rn(s, sm.cl[r][t][v]);
            sm.tot[t][v] = s;
        }
        cute::cluster_arrive_relaxed();
        NamedBarrier::sync(L::kCols, 0);
        if (tid == 0) cutlass::arch::ClusterBarrier::arrive(&sm.tot_bar);
        if constexpr (!kFuseAR) pdl_trigger();
        float pre[kTPC][kHc], rn[kTPC];
        token_scalars<kHidden>(p, sm, tok0, nt, lane, warp == 0 && rank == 0,
                               pre, rn);
        CUTLASS_PRAGMA_UNROLL
        for (int t = 0; t < kTPC; ++t) {
            if (t < nt)
                norm_quant_out<kHidden>(p, tok0 + t, h, lane, wn, rf[t], pre[t],
                                        rn[t]);
        }
    } else {
        if constexpr (kFuseAR) {
            if (lane == 0) atomicAdd(pl.counters + blk, 1u);
            const uint4 empty{};
            for (int i = lane; i < nt * L::kRowVecs;
                 i += cutlass::NumThreadsPerWarp) {
                const int q = i / L::kRowVecs, k = i % L::kRowVecs;
                const int64_t vid = int64_t(tok0 + q) * (kHidden / L::kVec) +
                                    col0 / L::kVec + k;
                uint8_t* slots = pl.peers[pl.rank] + phase_off;
                CUTLASS_PRAGMA_UNROLL
                for (int s = 0; s < kMaxWorld; ++s) {
                    if (s < pl.world)
                        moe::st_global_16B(empty, slots + s * pl.slot_bytes,
                                           vid);
                }
            }
        }
        cute::cluster_wait();
        cute::cluster_arrive_relaxed();
        mla::mbar_wait_cluster(cute::cast_smem_ptr_to_uint(&sm.tot_bar), 0);
        if (lane == 0 && !p.write_logits) {
            for (int t = rank; t < nt; t += L::kClusterSize) {
                const float rms = token_rms<kHc, kHidden>(sm.tot[t], p.rms_eps);
                float c[S::kComb];
                CUTLASS_PRAGMA_UNROLL
                for (int k = 0; k < S::kComb; ++k) {
                    c[k] = fmaf(sm.tot[t][2 * kHc + k] * rms, sm.hcs[2],
                                sm.hcb[2 * kHc + k]);
                }
                mhc::sinkhorn<kHc, true>(c, p.hc_sinkhorn_eps,
                                         p.sinkhorn_repeat);
                CUTLASS_PRAGMA_UNROLL
                for (int k = 0; k < S::kComb; ++k)
                    p.mix_out[int64_t(tok0 + t) * S::kComb + k] = c[k];
            }
        }
    }
    cute::cluster_wait();
}

template <int kHc>
__global__ void __launch_bounds__(cutlass::NumThreadsPerWarp)
    mhc_sinkhorn_kernel(const float* __restrict__ logits,
                        float* __restrict__ out, int num_tokens, float eps,
                        int repeat, const L2PrefetchArgs __grid_constant__ pf) {
    constexpr int kComb = MhcShape<kHc>::kComb;
    constexpr uint32_t kChunk = 64 * 1024;
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t nw = gridDim.x * blockDim.x;
    uint32_t first = 0;
#pragma unroll 1
    for (int r = 0; r < kMaxPrefetchRanges; ++r) {
        const uint32_t bytes = pf.bytes[r];
        const uint32_t n = (bytes + kChunk - 1) / kChunk;
        for (uint32_t c = (t + nw - first % nw) % nw; c < n; c += nw) {
            const uint32_t off = c * kChunk;
            moe::prefetch_l2_bulk(
                reinterpret_cast<const void*>(pf.addr[r] + off),
                min(kChunk, bytes - off));
        }
        first += n;
    }
    if (t >= num_tokens) return;
    float c[kComb];
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < kComb; ++k) c[k] = logits[int64_t(t) * kComb + k];
    mhc::sinkhorn<kHc, true>(c, eps, repeat);
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < kComb; ++k) out[int64_t(t) * kComb + k] = c[k];
}

using MhcKernel = void (*)(MhcParams);

struct MhcLaunch {
    MhcKernel kernel;
    size_t smem;
    int cluster_size;
    int block;
};

template <int kHc, int kHidden, int kTPC = 1>
MhcLaunch SelectMhcKernel(int tpc, bool fuse_ar, bool comb_logits) {
    if constexpr (kTPC < kMaxTokensPerCluster) {
        if (tpc > kTPC) {
            return SelectMhcKernel<kHc, kHidden, kTPC + 1>(tpc, fuse_ar,
                                                           comb_logits);
        }
    }
    using L = MhcTile<kHidden>;
    const MhcKernel kernels[2][2] = {
        {fused_all_reduce_mhc_kernel<kHc, kHidden, kTPC, false, false>,
         fused_all_reduce_mhc_kernel<kHc, kHidden, kTPC, false, true>},
        {fused_all_reduce_mhc_kernel<kHc, kHidden, kTPC, true, false>,
         fused_all_reduce_mhc_kernel<kHc, kHidden, kTPC, true, true>}};
    const MhcKernel kernel = kernels[fuse_ar][comb_logits];
    TORCH_CHECK(cutlass::ClusterLauncher::init(reinterpret_cast<const void*>(
                    kernel)) == cutlass::Status::kSuccess,
                "failed to allow a cluster of ", L::kClusterSize, " CTAs");
    return {kernel, sizeof(MhcSmem<kHc, kHidden, kTPC>), L::kClusterSize,
            L::kBlock};
}

template <int kHc>
MhcLaunch SelectMhcHidden(int hidden, int tpc, bool fuse_ar, bool comb_logits) {
    switch (hidden) {
        case 4096:
            return SelectMhcKernel<kHc, 4096>(tpc, fuse_ar, comb_logits);
    }
    TORCH_CHECK(false, "hidden support 4096 only, got ", hidden);
}

MhcLaunch SelectMhcKernel(int hc, int hidden, int tpc, bool fuse_ar,
                          bool comb_logits) {
    switch (hc) {
        case 2:
            return SelectMhcHidden<2>(hidden, tpc, fuse_ar, comb_logits);
        case 4:
            return SelectMhcHidden<4>(hidden, tpc, fuse_ar, comb_logits);
    }
    TORCH_CHECK(false, "hc must be 2 or 4, got ", hc);
}

int MaxActiveClusters(const MhcLaunch& launch) {
    const int clusters =
        cutlass::KernelHardwareInfo::query_device_max_active_clusters(
            dim3(launch.cluster_size), launch.block,
            reinterpret_cast<const void*>(launch.kernel));
    TORCH_CHECK(clusters > 0, "no cluster of ", launch.cluster_size,
                " CTAs is schedulable");
    return clusters;
}

L2PrefetchArgs MakeL2PrefetchArgs(const std::vector<torch::Tensor>& tensors) {
    TORCH_CHECK(tensors.size() <= kMaxPrefetchRanges, "at most ",
                kMaxPrefetchRanges, " prefetch tensors, got ", tensors.size());
    L2PrefetchArgs args{};
    for (size_t i = 0; i < tensors.size(); ++i) {
        const auto ptr = reinterpret_cast<uint64_t>(tensors[i].data_ptr());
        const uint64_t start = ptr & ~uint64_t{15};
        const uint64_t end = (ptr + tensors[i].nbytes() + 15) & ~uint64_t{15};
        args.addr[i] = start;
        args.bytes[i] = static_cast<uint32_t>(end - start);
    }
    return args;
}

using SinkhornKernel = void (*)(const float*, float*, int, float, int,
                                L2PrefetchArgs);

SinkhornKernel SelectSinkhornKernel(int hc) {
    switch (hc) {
        case 2:
            return mhc_sinkhorn_kernel<2>;
        case 4:
            return mhc_sinkhorn_kernel<4>;
    }
    TORCH_CHECK(false, "hc must be 2 or 4, got ", hc);
}

}  // namespace

std::vector<torch::Tensor> FusedAllReduceMhc(
    const torch::Tensor& x, const torch::Tensor& residual,
    const torch::Tensor& post, const torch::Tensor& comb,
    const torch::Tensor& fn, const torch::Tensor& hc_scale,
    const torch::Tensor& hc_base, const torch::Tensor& norm_weight,
    const double rms_eps, const double hc_pre_eps, const double hc_sinkhorn_eps,
    const double hc_post_mult, const int64_t sinkhorn_repeat,
    const double norm_eps, const bool comb_is_logits, const bool return_logits,
    const std::optional<torch::Tensor>& peer_ptrs,
    const std::optional<torch::Tensor>& counters, const int64_t slot_bytes,
    const int64_t tp_rank, const bool use_pdl) {
    const CudaDeviceGuard device_guard(x.device());
    TORCH_CHECK(residual.dim() == 3, "residual must be [T, hc, hidden]");
    const int64_t num_tokens = residual.size(0), hc = residual.size(1),
                  hidden = residual.size(2);
    const bool fuse_ar = peer_ptrs.has_value();
    const MhcLaunch probe =
        SelectMhcKernel(hc, hidden, 1, fuse_ar, comb_is_logits);
    const int cs = probe.cluster_size;

    const auto opts = x.options();
    torch::Tensor residual_out = torch::empty({num_tokens, hc, hidden}, opts);
    torch::Tensor layer_out = torch::empty({num_tokens, hidden}, opts);
    torch::Tensor post_out =
        torch::empty({num_tokens, hc}, opts.dtype(torch::kFloat32));
    torch::Tensor mix_out =
        torch::empty({num_tokens, hc, hc}, opts.dtype(torch::kFloat32));
    torch::Tensor q =
        torch::empty({num_tokens, hidden}, opts.dtype(torch::kFloat8_e4m3fn));
    torch::Tensor sf = torch::empty({num_tokens, hidden / mxfp8::kGroup},
                                    opts.dtype(torch::kByte));
    std::vector<torch::Tensor> outs{residual_out, layer_out, post_out,
                                    mix_out,      q,         sf};
    if (num_tokens == 0) return outs;

    const int max_clusters = MaxActiveClusters(probe);
    int tpc = cute::ceil_div(int(num_tokens), max_clusters);
    TORCH_CHECK(!fuse_ar || tpc <= kMaxTokensPerCluster, "at most ",
                kMaxTokensPerCluster * max_clusters,
                " tokens are supported with the fused all-reduce, got ",
                num_tokens);
    tpc = std::min(tpc, kMaxTokensPerCluster);
    const MhcLaunch launch =
        SelectMhcKernel(hc, hidden, tpc, fuse_ar, comb_is_logits);
    const int num_clusters = cute::ceil_div(int(num_tokens), tpc);

    LamportPlane plane{};
    if (fuse_ar) {
        const int world = peer_ptrs->numel();
        TORCH_CHECK(world <= kMaxWorld &&
                        counters->numel() >= int64_t{cs} * num_clusters &&
                        num_tokens * hidden * int64_t{sizeof(__nv_bfloat16)} <=
                            slot_bytes,
                    "workspace for world ", world, " cannot hold ", num_tokens,
                    " tokens");
        const auto* addrs = peer_ptrs->const_data_ptr<int64_t>();
        for (int r = 0; r < world; ++r) {
            plane.peers[r] = reinterpret_cast<uint8_t*>(addrs[r]);
        }
        plane.counters = reinterpret_cast<uint32_t*>(counters->data_ptr());
        plane.slot_bytes = slot_bytes;
        plane.world = world;
        plane.rank = tp_rank;
        plane.num_counters = counters->numel();
    }

    const MhcParams params{
        .x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        .residual = reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr()),
        .post = post.const_data_ptr<float>(),
        .comb = comb.const_data_ptr<float>(),
        .fn = fn.const_data_ptr<float>(),
        .hc_scale = hc_scale.const_data_ptr<float>(),
        .hc_base = hc_base.const_data_ptr<float>(),
        .norm_weight =
            reinterpret_cast<const __nv_bfloat16*>(norm_weight.data_ptr()),
        .residual_out =
            reinterpret_cast<__nv_bfloat16*>(residual_out.data_ptr()),
        .layer_out = reinterpret_cast<__nv_bfloat16*>(layer_out.data_ptr()),
        .post_out = post_out.data_ptr<float>(),
        .mix_out = mix_out.data_ptr<float>(),
        .q_out = reinterpret_cast<uint8_t*>(q.data_ptr()),
        .sf_out = sf.data_ptr<uint8_t>(),
        .num_tokens = int(num_tokens),
        .sinkhorn_repeat = int(sinkhorn_repeat),
        .write_logits = return_logits,
        .rms_eps = float(rms_eps),
        .hc_pre_eps = float(hc_pre_eps),
        .hc_sinkhorn_eps = float(hc_sinkhorn_eps),
        .hc_post_mult = float(hc_post_mult),
        .norm_eps = float(norm_eps),
        .plane = plane,
    };
    HALOPS_CUDA_CHECK(LaunchKernelWithPdl(
        launch.kernel, dim3(cs, num_clusters), dim3(launch.block),
        dim3(cs, 1, 1), launch.smem, GetCurrentCUDAStream(x), use_pdl, params));
    return outs;
}

void MhcSinkhorn(const torch::Tensor& logits, torch::Tensor& out,
                 const double eps, const int64_t repeat,
                 const std::vector<torch::Tensor>& prefetch) {
    const CudaDeviceGuard device_guard(logits.device());
    TORCH_CHECK(logits.dim() == 3 && logits.size(1) == logits.size(2),
                "logits must be [T, hc, hc]");
    const int64_t num_tokens = logits.size(0), hc = logits.size(1);
    const SinkhornKernel kernel = SelectSinkhornKernel(hc);
    const L2PrefetchArgs pf = MakeL2PrefetchArgs(prefetch);
    if (num_tokens == 0) return;
    HALOPS_CUDA_CHECK(LaunchKernelWithPdl(
        kernel,
        dim3(cute::ceil_div(int(num_tokens), cutlass::NumThreadsPerWarp)),
        dim3(cutlass::NumThreadsPerWarp), kNoCluster, 0,
        GetCurrentCUDAStream(logits), false, logits.const_data_ptr<float>(),
        out.data_ptr<float>(), int(num_tokens), float(eps), int(repeat), pf));
}

}  // namespace halops
