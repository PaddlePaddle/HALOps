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

#include "kernel/gemm/low_latency_utils.cuh"
#include "kernel/gemm/pdl.cuh"

namespace halops {

template <int kBlockN, int kWarpsN, int kWarpsK, int kRowsA, int kStages,
          int kAccStages>
struct LlTraits {
    static constexpr int kGroupK = kLlGroupK;
    static constexpr int kOuts = kRowsA == 16 ? 4 : 2;
    static constexpr int kAAtoms = kRowsA / 8;
    static constexpr int kAtomBytes = 8 * 128;
    static constexpr int kATileBytes = kAAtoms * kAtomBytes;
    static constexpr int kBTileBytes = (kLlBRows<kWarpsN>() / 8) * kAtomBytes;
    static constexpr int kBPhaseGroups = kLlBPhaseGroups;
    static constexpr int kBPhaseBytes = kLlBPhaseBytes<kWarpsN>();
    static constexpr int kConsumerWarps = kLlConsumerWarps<kWarpsN, kWarpsK>();
    static constexpr int kProducerWarp = kLlProducerWarp<kWarpsN, kWarpsK>();
    static constexpr int kEpilogueWarp0 = kLlEpilogueWarp0<kWarpsN, kWarpsK>();
    static constexpr int kBRings = kLlBRings<kWarpsN, kWarpsK>();
    static constexpr int kRingSlots = kBRings * kStages;
    static constexpr int kAccCols = kAccStages * kWarpsN;
    static constexpr int kAccSlot = kWarpsK * 32 * kOuts;
    static constexpr int kSfbStages = kLlSfbStages;
    static constexpr int kSfaRows = kLlSfaRows;
};

enum class LlWarpRole { kConsumer, kProducer, kEpilogue };

struct LlSmemView {
    uint8_t* a_res;
    uint8_t* b_ring;
    int32_t* sfa_res;
    int32_t* sfb_ring;
    Barrier* a_full;
    Barrier* sfa_bar;
    Barrier* sfb_full;
    Barrier* sfb_empty;
    Barrier* full_bar;
    Barrier* empty_bar;
    Barrier* acc_full;
    Barrier* acc_empty;
    float* acc_buf;
};

struct LlCtx {
    const LowLatencyMmaGemmParams& params;
    LlScaleLayout sl;
    LlSmemView v;
    int lane, gid, tig, n, m_base, n_tiles, swz;
    int n_iter, n_phase, k1_groups;
    bool dual_a, dual_sfa;
};

template <int kWarpsN, int kWarpsK>
__device__ __forceinline__ LlWarpRole ll_warp_role(int warp) {
    if (warp == kLlProducerWarp<kWarpsN, kWarpsK>())
        return LlWarpRole::kProducer;
    if (warp >= kLlEpilogueWarp0<kWarpsN, kWarpsK>())
        return LlWarpRole::kEpilogue;
    return LlWarpRole::kConsumer;
}

template <int kWarpsN, int kWarpsK, int kStages, int kAccStages>
__device__ __forceinline__ LlSmemView ll_smem_view(uint8_t* smem,
                                                   const LlSmemPlan& plan,
                                                   int n_groups) {
    constexpr int kRingSlots = kLlBRings<kWarpsN, kWarpsK>() * kStages;
    constexpr int kAccCols = kAccStages * kWarpsN;
    constexpr int kSfbStages = kLlSfbStages;
    LlSmemView v;
    v.a_res = smem;
    v.b_ring = smem + plan.b_off;
    v.sfa_res = reinterpret_cast<int32_t*>(smem + plan.sfa_off);
    v.sfb_ring = reinterpret_cast<int32_t*>(smem + plan.sfb_off);
    v.a_full = reinterpret_cast<Barrier*>(smem + plan.bars_off);
    v.sfa_bar = v.a_full + n_groups;
    v.sfb_full = v.sfa_bar + 1;
    v.sfb_empty = v.sfb_full + kSfbStages;
    v.full_bar = v.sfb_empty + kSfbStages;
    v.empty_bar = v.full_bar + kRingSlots;
    v.acc_full = v.empty_bar + kStages;
    v.acc_empty = v.acc_full + kAccCols;
    v.acc_buf = reinterpret_cast<float*>(smem + plan.acc_off);
    return v;
}

template <int kWarpsN, int kWarpsK, int kStages, int kAccStages>
__device__ __forceinline__ void ll_init_barriers(const LlSmemView& v,
                                                 int n_groups, int lane) {
    constexpr int kRingSlots = kLlBRings<kWarpsN, kWarpsK>() * kStages;
    constexpr int kAccCols = kAccStages * kWarpsN;
    constexpr int kSfbStages = kLlSfbStages;
    constexpr int kConsumerWarps = kWarpsN * kWarpsK;
    for (int i = lane; i < n_groups; i += 32) (v.a_full + i)->init(1);
    if (lane == 0) v.sfa_bar->init(1);
    for (int i = lane; i < kSfbStages; i += 32) (v.sfb_full + i)->init(1);
    for (int i = lane; i < kSfbStages; i += 32)
        (v.sfb_empty + i)->init(kConsumerWarps);
    for (int i = lane; i < kRingSlots; i += 32) (v.full_bar + i)->init(1);
    for (int i = lane; i < kStages; i += 32)
        (v.empty_bar + i)->init(kConsumerWarps);
    for (int i = lane; i < kAccCols; i += 32) {
        (v.acc_full + i)->init(kWarpsK);
        (v.acc_empty + i)->init(1);
    }
    cutlass::arch::fence_view_async_shared();
}

template <int kBlockN, int kWarpsN, int kWarpsK, int kRowsA, int kStages,
          int kAccStages>
__device__ __forceinline__ void ll_producer_warp(const LlCtx& c) {
    using T = LlTraits<kBlockN, kWarpsN, kWarpsK, kRowsA, kStages, kAccStages>;
    const bool act = c.lane < T::kBRings;
    Barrier* const my_full = c.v.full_bar + c.lane * kStages;
    uint8_t* const my_b = c.v.b_ring + c.lane * kStages * T::kBPhaseBytes;

    auto issue_sfa = [&]() {
        const uint32_t row_bytes =
            static_cast<uint32_t>(ll_sfa_g32_row_bytes(c.params.k));
        const int rows = min(c.params.m - c.m_base, T::kSfaRows);
        const uint32_t b1 =
            c.dual_sfa ? static_cast<uint32_t>(c.params.k1 / 32) : 0u;
        const uint32_t b2 =
            c.dual_sfa ? static_cast<uint32_t>(
                             ll_sfa_g32_row_bytes(c.params.k - c.params.k1))
                       : 0u;
        const uint32_t per_row = c.dual_sfa ? b1 + b2 : row_bytes;
        if (c.lane == 0)
            c.v.sfa_bar->arrive_and_expect_tx(per_row *
                                              static_cast<uint32_t>(rows));
        __syncwarp();
        const uint8_t* const src =
            reinterpret_cast<const uint8_t*>(c.params.ptr_sfa);
        if (c.dual_sfa) {
            const uint8_t* const src2 =
                reinterpret_cast<const uint8_t*>(c.params.ptr_sfa2);
            for (int r = c.lane; r < rows; r += 32) {
                cute::SM90_BULK_COPY_G2S::copy(
                    src + static_cast<int64_t>(c.m_base + r) *
                              c.params.sfa_stride_mn,
                    reinterpret_cast<uint64_t*>(c.v.sfa_bar),
                    c.v.sfa_res + r * c.sl.sfa_pitch, static_cast<int32_t>(b1));
                cute::SM90_BULK_COPY_G2S::copy(
                    src2 + static_cast<int64_t>(c.m_base + r) *
                               c.params.sfa2_stride_mn,
                    reinterpret_cast<uint64_t*>(c.v.sfa_bar),
                    c.v.sfa_res + r * c.sl.sfa_pitch + c.k1_groups,
                    static_cast<int32_t>(b2));
            }
        } else {
            for (int r = c.lane; r < rows; r += 32)
                cute::SM90_BULK_COPY_G2S::copy(
                    src + static_cast<int64_t>(c.m_base + r) *
                              c.params.sfa_stride_mn,
                    reinterpret_cast<uint64_t*>(c.v.sfa_bar),
                    c.v.sfa_res + r * c.sl.sfa_pitch,
                    static_cast<int32_t>(row_bytes));
        }
    };
    if (!c.params.pdl) issue_sfa();

    int st = 0;
    uint32_t spass = 0;
    bool sprimed = false;

    LlTileWalk w =
        ll_tile_walk(c.n_tiles, c.swz, c.params.walk_q, c.params.walk_r);
    int buf = 0;
    uint32_t pass = 0;
    bool primed = false;
    bool first = true;
    bool pdl_synced = false;
    auto pdl_sync_once = [&]() {
        if (c.params.pdl && !pdl_synced) {
            PdlWaitForDependents();
            issue_sfa();
            pdl_synced = true;
        }
    };
    while (w.is_valid()) {
        const int tile_n0 = w.tile_id() * kBlockN;
        if (sprimed) (c.v.sfb_empty + st)->wait((spass - 1) & 1);
        if (c.lane == 0) {
            Barrier* const slot_full = c.v.sfb_full + st;
            slot_full->arrive_and_expect_tx(
                static_cast<uint32_t>(c.sl.sfb_tx_bytes));
            cute::SM90_TMA_LOAD_2D::copy(
                &c.params.sfb_desc, reinterpret_cast<uint64_t*>(slot_full),
                kTmaCacheHint, c.v.sfb_ring + st * c.sl.sfb_slot_ints, 0,
                tile_n0);
        }
        for (int phase = 0; phase < c.n_phase; ++phase) {
            const int j0 = phase * T::kBPhaseGroups;
            auto issue_a_phase = [&]() {
                if (!first || c.lane >= kWarpsK) return;
                for (int q = 0; q < T::kBPhaseGroups; ++q) {
                    const int j = j0 + q;
                    if (j >= c.n_iter) break;
                    const int g = ll_kb<kWarpsK>(c.lane, j, c.n_iter);
                    const bool a2 = c.dual_a && g >= c.k1_groups;
                    const CUtensorMap* const adesc =
                        a2 ? &c.params.a_desc2 : &c.params.a_desc;
                    const int acrd0 = (a2 ? g - c.k1_groups : g) * T::kGroupK;
                    (c.v.a_full + g)
                        ->arrive_and_expect_tx(
                            static_cast<uint32_t>(T::kATileBytes));
#pragma unroll
                    for (int t = 0; t < T::kAAtoms; ++t)
                        cute::SM90_TMA_LOAD_2D::copy(
                            adesc, reinterpret_cast<uint64_t*>(c.v.a_full + g),
                            kTmaCacheHint,
                            c.v.a_res + g * T::kATileBytes + t * T::kAtomBytes,
                            acrd0, c.m_base + t * 8);
                }
            };
            if (primed) (c.v.empty_bar + buf)->wait((pass - 1) & 1);
            if (act) {
                Barrier* const fb = my_full + buf;
                const int live_groups = c.n_iter - j0 < T::kBPhaseGroups
                                            ? c.n_iter - j0
                                            : T::kBPhaseGroups;
                fb->arrive_and_expect_tx(
                    static_cast<uint32_t>(live_groups * T::kBTileBytes));
                cute::SM90_TMA_LOAD_3D::copy(
                    &c.params.b_desc, reinterpret_cast<uint64_t*>(fb),
                    kTmaCacheHint, my_b + buf * T::kBPhaseBytes, 0, tile_n0,
                    ll_kb<kWarpsK>(c.lane, j0, c.n_iter));
            }
            // Under PDL, B is issued before the dependency wait; A and sfa
            // come from the upstream kernel and are issued after it.
            pdl_sync_once();
            issue_a_phase();
            if (++buf == kStages) {
                buf = 0;
                ++pass;
                primed = true;
            }
        }
        first = false;
        if (++st == T::kSfbStages) {
            st = 0;
            ++spass;
            sprimed = true;
        }
        w.advance();
    }
}

template <int kBlockN, int kWarpsN, int kWarpsK, int kRowsA, int kStages,
          int kAccStages>
__device__ __forceinline__ void ll_epilogue_warp(const LlCtx& c) {
    using T = LlTraits<kBlockN, kWarpsN, kWarpsK, kRowsA, kStages, kAccStages>;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int warp_n = warp - T::kEpilogueWarp0;
    const int row_c0 = c.m_base + c.gid;
    const int row_c1 = row_c0 + 8;
    __nv_bfloat16* const ptr_d = c.params.ptr_d;
    __nv_bfloat16* const ptr_d2 = c.params.ptr_d2;
    const int n_split = c.params.n_split;
    LlTileWalk w =
        ll_tile_walk(c.n_tiles, c.swz, c.params.walk_q, c.params.walk_r);
    int as = 0;
    uint32_t apar = 0;
    while (w.is_valid()) {
        (c.v.acc_full + as * kWarpsN + warp_n)->wait(apar);
        const float* const slot = c.v.acc_buf +
                                  (as * kWarpsN + warp_n) * T::kAccSlot +
                                  c.lane * T::kOuts;
        float out[T::kOuts];
#pragma unroll
        for (int i = 0; i < T::kOuts; ++i) out[i] = slot[i];
#pragma unroll
        for (int wk = 1; wk < kWarpsK; ++wk) {
            const float* const o = slot + wk * 32 * T::kOuts;
#pragma unroll
            for (int i = 0; i < T::kOuts; ++i) out[i] += o[i];
        }
        const int col_c = w.tile_id() * kBlockN + warp_n * 8 + 2 * c.tig;
        const bool seg2 = ptr_d2 != nullptr && col_c >= n_split;
        __nv_bfloat16* const dst = seg2 ? ptr_d2 : ptr_d;
        const int seg_n =
            ptr_d2 != nullptr ? (seg2 ? c.n - n_split : n_split) : c.n;
        const int seg_col = seg2 ? col_c - n_split : col_c;
        if (seg_col + 1 < seg_n) {
            if (row_c0 < c.params.m)
                *reinterpret_cast<__nv_bfloat162*>(
                    dst + static_cast<int64_t>(row_c0) * seg_n + seg_col) =
                    __floats2bfloat162_rn(out[0], out[1]);
            if constexpr (kRowsA == 16) {
                if (row_c1 < c.params.m)
                    *reinterpret_cast<__nv_bfloat162*>(
                        dst + static_cast<int64_t>(row_c1) * seg_n + seg_col) =
                        __floats2bfloat162_rn(out[2], out[3]);
            }
        }
        __syncwarp();
        if (c.lane == 0) (c.v.acc_empty + as * kWarpsN + warp_n)->arrive();
        if (++as == kAccStages) {
            as = 0;
            apar ^= 1;
        }
        w.advance();
    }
}

template <int kBlockN, int kWarpsN, int kWarpsK, int kRowsA, int kStages,
          int kAccStages>
__device__ __forceinline__ void ll_consumer_warp(const LlCtx& c) {
    using T = LlTraits<kBlockN, kWarpsN, kWarpsK, kRowsA, kStages, kAccStages>;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int warp_n = warp % kWarpsN;
    const int warp_k = warp / kWarpsN;
    Barrier* const my_full = c.v.full_bar + warp_k * kStages;
    const uint8_t* const my_b = c.v.b_ring + warp_k * kStages * T::kBPhaseBytes;
    const int row_c0 = c.m_base + c.gid;
    const int row_c1 = row_c0 + 8;
    const bool has_a0 = row_c0 < c.params.m;
    const bool has_a1 = kRowsA == 16 && row_c1 < c.params.m;

    const int sfb_col = warp_n * 8 + 2 * c.tig;

    LlTileWalk w =
        ll_tile_walk(c.n_tiles, c.swz, c.params.walk_q, c.params.walk_r);
    if (!w.is_valid()) return;
    int buf = 0;
    uint32_t par = 0;
    int as = 0;
    uint32_t apass = 0;
    bool aprimed = false;
    bool first_tile = true;
    int st = 0;
    uint32_t sparity = 0;

    while (w.is_valid()) {
        float out[T::kOuts] = {};
        const int32_t* const my_sfb = c.v.sfb_ring + st * c.sl.sfb_slot_ints;

        (c.v.sfb_full + st)->wait(sparity);
        for (int phase = 0; phase < c.n_phase; ++phase) {
            const int j0 = phase * T::kBPhaseGroups;
            // A stays resident for every tile: its barriers complete once.
            if (first_tile) {
                for (int q = 0; q < T::kBPhaseGroups; ++q) {
                    const int j = j0 + q;
                    if (j >= c.n_iter) break;
                    (c.v.a_full + ll_kb<kWarpsK>(warp_k, j, c.n_iter))->wait(0);
                }
            }
            (my_full + buf)->wait(par);
            __syncwarp();

            for (int q = 0; q < T::kBPhaseGroups; ++q) {
                const int j = j0 + q;
                if (j >= c.n_iter) break;
                const uint8_t* const tile_a =
                    c.v.a_res +
                    ll_kb<kWarpsK>(warp_k, j, c.n_iter) * T::kATileBytes;
                const uint8_t* const tile_b = my_b + buf * T::kBPhaseBytes +
                                              q * T::kBTileBytes +
                                              warp_n * T::kAtomBytes;

                // acc[s] holds k-group s of the 128-wide k-block, scaled by
                // byte s of the packed sfa/sfb words below.
                float acc[4][4] = {};
#pragma unroll
                for (int s = 0; s < 4; ++s) {
                    uint32_t frag_a[4];
                    uint32_t frag_b[2];
                    const uint8_t* ra = tile_a + c.tig * 4;
                    const uint8_t* rb = tile_b + c.tig * 4;
                    frag_a[0] = lds_u32(ra + swz128(c.gid, 2 * s));
                    frag_a[2] = lds_u32(ra + swz128(c.gid, 2 * s + 1));
                    if constexpr (kRowsA == 16) {
                        frag_a[1] =
                            lds_u32(ra + T::kAtomBytes + swz128(c.gid, 2 * s));
                        frag_a[3] = lds_u32(ra + T::kAtomBytes +
                                            swz128(c.gid, 2 * s + 1));
                    } else {
                        frag_a[1] = 0u;
                        frag_a[3] = 0u;
                    }
                    frag_b[0] = lds_u32(rb + swz128(c.gid, 2 * s));
                    frag_b[1] = lds_u32(rb + swz128(c.gid, 2 * s + 1));
                    mma_m16n8k32_e4m3(acc[s], frag_a, frag_b);
                }

                const int kb = ll_kb<kWarpsK>(warp_k, j, c.n_iter);
                if (first_tile && j == 0) c.v.sfa_bar->wait(0);
                const int32_t wb0 = my_sfb[sfb_col * c.sl.sfb_pitch + kb];
                const int32_t wb1 = my_sfb[(sfb_col + 1) * c.sl.sfb_pitch + kb];
                const int32_t wa0 =
                    has_a0 ? c.v.sfa_res[c.gid * c.sl.sfa_pitch + kb] : 0;
                const int32_t wa1 =
                    has_a1 ? c.v.sfa_res[(c.gid + 8) * c.sl.sfa_pitch + kb] : 0;
#pragma unroll
                for (int s = 0; s < 4; ++s) {
                    const float sb0 = decode_packed_ue8m0_word(wb0, s);
                    const float sb1 = decode_packed_ue8m0_word(wb1, s);
                    const float sa0 = decode_packed_ue8m0_word(wa0, s);
                    out[0] += acc[s][0] * (sa0 * sb0);
                    out[1] += acc[s][1] * (sa0 * sb1);
                    if constexpr (kRowsA == 16) {
                        const float sa1 = decode_packed_ue8m0_word(wa1, s);
                        out[2] += acc[s][2] * (sa1 * sb0);
                        out[3] += acc[s][3] * (sa1 * sb1);
                    }
                }
            }
            __syncwarp();
            if (c.lane == 0) (c.v.empty_bar + buf)->arrive();
            if (++buf == kStages) {
                buf = 0;
                par ^= 1;
            }
        }
        __syncwarp();
        if (c.lane == 0) (c.v.sfb_empty + st)->arrive();
        if (++st == T::kSfbStages) {
            st = 0;
            sparity ^= 1;
        }

        if (aprimed)
            (c.v.acc_empty + as * kWarpsN + warp_n)->wait((apass - 1) & 1);
        float* const slot = c.v.acc_buf +
                            (as * kWarpsN + warp_n) * T::kAccSlot +
                            (warp_k * 32 + c.lane) * T::kOuts;
        if constexpr (T::kOuts == 4) {
            *reinterpret_cast<float4*>(slot) =
                make_float4(out[0], out[1], out[2], out[3]);
        } else {
            *reinterpret_cast<float2*>(slot) = make_float2(out[0], out[1]);
        }
        __syncwarp();
        if (c.lane == 0) (c.v.acc_full + as * kWarpsN + warp_n)->arrive();
        if (++as == kAccStages) {
            as = 0;
            ++apass;
            aprimed = true;
        }
        first_tile = false;
        w.advance();
    }
}

template <int kBlockN, int kWarpsN, int kWarpsK, int kRowsA, int kStages,
          int kAccStages = 3>
__global__ void __launch_bounds__(32 * (kWarpsN * kWarpsK + 1 + kWarpsN), 1)
    low_latency_mma_gemm_kernel(
        const __grid_constant__ LowLatencyMmaGemmParams params) {
    static_assert(kBlockN == kWarpsN * 8, "one warp owns 8 columns of N");
    static_assert(kStages >= 1 && kStages <= kLlMaxStages,
                  "kStages must be 1..kLlMaxStages");
    static_assert(kAccStages >= 1 && kAccStages <= 4,
                  "kAccStages must be 1..4");
    static_assert(kRowsA == 8 || kRowsA == 16, "kRowsA is the M tier: 8 or 16");
    using T = LlTraits<kBlockN, kWarpsN, kWarpsK, kRowsA, kStages, kAccStages>;

    extern __shared__ __align__(16) uint8_t smem[];
    const LlScaleLayout sl = ll_scale_layout(params.k, kBlockN);
    const LlSmemPlan plan =
        ll_smem_plan<kWarpsN, kWarpsK, kRowsA, kAccStages>(params.k, kStages);
    const int lane = threadIdx.x & 31;
    const int n_groups = params.k / T::kGroupK;
    const LlSmemView v = ll_smem_view<kWarpsN, kWarpsK, kStages, kAccStages>(
        smem, plan, n_groups);
    if (threadIdx.x < 32)
        ll_init_barriers<kWarpsN, kWarpsK, kStages, kAccStages>(v, n_groups,
                                                                lane);
    __syncthreads();

    const int n_iter = n_groups / kWarpsK;
    const LlCtx c{params,
                  sl,
                  v,
                  lane,
                  lane >> 2,
                  lane & 3,
                  params.n,
                  static_cast<int>(blockIdx.y) * 16,
                  (params.n + kBlockN - 1) / kBlockN,
                  params.tile_swz,
                  n_iter,
                  (n_iter + T::kBPhaseGroups - 1) / T::kBPhaseGroups,
                  params.k1 / T::kGroupK,
                  params.ptr_a2 != nullptr && params.k1 > 0,
                  params.ptr_a2 != nullptr && params.ptr_sfa2 != nullptr};

    const int warp = static_cast<int>(threadIdx.x) >> 5;
    switch (ll_warp_role<kWarpsN, kWarpsK>(warp)) {
        case LlWarpRole::kProducer:
            ll_producer_warp<kBlockN, kWarpsN, kWarpsK, kRowsA, kStages,
                             kAccStages>(c);
            break;
        case LlWarpRole::kEpilogue:
            ll_epilogue_warp<kBlockN, kWarpsN, kWarpsK, kRowsA, kStages,
                             kAccStages>(c);
            break;
        case LlWarpRole::kConsumer:
            ll_consumer_warp<kBlockN, kWarpsN, kWarpsK, kRowsA, kStages,
                             kAccStages>(c);
            break;
    }
}

struct LlDescKey {
    const void* ptr = nullptr;
    int kind = 0;
    int v0 = 0, v1 = 0, v2 = 0, v3 = 0;
};

inline bool LlCachedDesc(CUtensorMap* out, const LlDescKey& key, int rank,
                         CUtensorMapDataType dtype, CUtensorMapSwizzle swizzle,
                         const uint64_t* gdim, const uint64_t* gstride,
                         const uint32_t* bdim) {
    struct Slot {
        LlDescKey key;
        bool ok = false;
        CUtensorMap map = {};
    };
    constexpr int kSlots = 16;
    static thread_local Slot slots[kSlots];
    static thread_local int next = 0;
    for (int i = 0; i < kSlots; ++i) {
        const LlDescKey& k = slots[i].key;
        if (k.ptr == key.ptr && k.kind == key.kind && k.v0 == key.v0 &&
            k.v1 == key.v1 && k.v2 == key.v2 && k.v3 == key.v3) {
            *out = slots[i].map;
            return slots[i].ok;
        }
    }
    const uint32_t estride[3] = {1u, 1u, 1u};
    Slot s;
    s.key = key;
    s.ok = cuTensorMapEncodeTiled(
               &s.map, dtype, rank, const_cast<void*>(key.ptr), gdim, gstride,
               bdim, estride, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
               CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
    slots[next] = s;
    next = (next + 1) & (kSlots - 1);
    *out = s.map;
    return s.ok;
}

inline bool LowLatencyMmaBDesc3D(CUtensorMap* out, const void* ptr_b, int n,
                                 int k, int rows, int groups) {
    const uint64_t gdim[3] = {128ull, static_cast<uint64_t>(n),
                              static_cast<uint64_t>(k / 128)};
    const uint64_t gstride[2] = {static_cast<uint64_t>(k), 128ull};
    const uint32_t bdim[3] = {128u, static_cast<uint32_t>(rows),
                              static_cast<uint32_t>(groups)};
    return LlCachedDesc(out, {ptr_b, 0, n, k, rows, groups}, 3,
                        CU_TENSOR_MAP_DATA_TYPE_UINT8,
                        CU_TENSOR_MAP_SWIZZLE_128B, gdim, gstride, bdim);
}

inline bool LowLatencyMmaSfbDesc(CUtensorMap* out, const void* ptr_sfb, int n,
                                 int k, int stride, int cols) {
    const uint64_t gdim[2] = {static_cast<uint64_t>(k / 128),
                              static_cast<uint64_t>(n)};
    const uint64_t gstride[1] = {static_cast<uint64_t>(stride)};
    const uint32_t bdim[2] = {static_cast<uint32_t>(ll_scale_pitch(k / 128)),
                              static_cast<uint32_t>(cols)};
    return LlCachedDesc(out, {ptr_sfb, 1, n, k, stride, cols}, 2,
                        CU_TENSOR_MAP_DATA_TYPE_INT32,
                        CU_TENSOR_MAP_SWIZZLE_NONE, gdim, gstride, bdim);
}

inline bool LowLatencyMmaADesc(CUtensorMap* out, const void* ptr_a, int m,
                               int k) {
    const uint64_t gdim[2] = {static_cast<uint64_t>(k),
                              static_cast<uint64_t>(m)};
    const uint64_t gstride[1] = {static_cast<uint64_t>(k)};
    const uint32_t bdim[2] = {128u, 8u};
    return LlCachedDesc(out, {ptr_a, 2, m, k}, 2, CU_TENSOR_MAP_DATA_TYPE_UINT8,
                        CU_TENSOR_MAP_SWIZZLE_128B, gdim, gstride, bdim);
}

template <int kWarpsN, int kWarpsK, int kRowsA, int kAccStages>
int LowLatencyMmaFitStages(int k) {
    const LlSmemPlan plan =
        ll_smem_plan<kWarpsN, kWarpsK, kRowsA, kAccStages>(k, 0);
    const int budget = kLlSmemBudgetBytes - plan.fixed_bytes;
    if (budget <= 0) return 0;
    const int fit = budget / plan.per_stage_bytes;
    return fit > kLlMaxStages ? kLlMaxStages : fit;
}

template <int kWarpsN, int kWarpsK>
constexpr int LowLatencyMmaThreads() {
    return 32 * (kWarpsN * kWarpsK + 1 + kWarpsN);
}

template <int kBlockN, int kWarpsN, int kWarpsK, int kRowsA, int kStages,
          int kAccStages>
void LowLatencyMmaSetSmem(int smem_bytes) {
    constexpr int kMaxDevices = 16;
    static thread_local int cap[kMaxDevices] = {};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) dev = -1;
    const bool cached = dev >= 0 && dev < kMaxDevices;
    if (cached && smem_bytes <= cap[dev]) return;
    const void* kernel = reinterpret_cast<const void*>(
        &low_latency_mma_gemm_kernel<kBlockN, kWarpsN, kWarpsK, kRowsA, kStages,
                                     kAccStages>);
    if (smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    }
    cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxShared);
    if (cached) cap[dev] = smem_bytes;
}

template <int kBlockN, int kWarpsN, int kWarpsK, int kRowsA, int kStages,
          int kAccStages = 3>
bool LaunchLowLatencyMmaGemm(const LowLatencyMmaGemmParams& params,
                             int tile_swz, int sm_count, bool use_pdl,
                             cudaStream_t stream) {
    const LlSmemPlan plan =
        ll_smem_plan<kWarpsN, kWarpsK, kRowsA, kAccStages>(params.k, kStages);
    const int smem_bytes = plan.total_bytes;
    if (smem_bytes > kLlSmemBudgetBytes) return false;
    const int num_k_groups = params.k / 128;
    if (num_k_groups <= 0 || num_k_groups % kWarpsK != 0) return false;
    const bool dual_a = params.ptr_a2 != nullptr;
    const bool dual_sfa = dual_a && params.ptr_sfa2 != nullptr;
    if (dual_a) {
        if (params.k1 <= 0 || params.k1 >= params.k || params.k1 % 128 != 0)
            return false;
        if (dual_sfa &&
            (params.k1 % 512 != 0 ||
             reinterpret_cast<uintptr_t>(params.ptr_sfa2) % 16 != 0))
            return false;
    }
    const int sfa_k = dual_sfa ? params.k1 : params.k;
    if (params.sfa_stride_k != 1 || params.sfa_stride_mn % 16 != 0 ||
        params.sfa_stride_mn < ll_sfa_g32_row_bytes(sfa_k) ||
        reinterpret_cast<uintptr_t>(params.ptr_sfa) % 16 != 0)
        return false;
    if (dual_sfa &&
        (params.sfa2_stride_mn % 16 != 0 ||
         params.sfa2_stride_mn < ll_sfa_g32_row_bytes(params.k - params.k1)))
        return false;
    if (params.ptr_sfb_packed == nullptr || params.sfb_stride_k != 1 ||
        params.sfb_stride_mn % 16 != 0 || params.n % kBlockN != 0 ||
        ll_scale_pitch(params.k / 128) > 256)
        return false;
    LowLatencyMmaSetSmem<kBlockN, kWarpsN, kWarpsK, kRowsA, kStages,
                         kAccStages>(smem_bytes);
    const int m_tiles = (params.m + kRowsA - 1) / kRowsA;
    const int gx = sm_count / m_tiles > 0 ? sm_count / m_tiles : 1;
    LowLatencyMmaGemmParams p = params;
    p.tile_swz = tile_swz;
    p.pdl = use_pdl && !params.pdl_force_off ? 1 : 0;
    {
        const int n_tiles = (params.n + kBlockN - 1) / kBlockN;
        p.walk_q = n_tiles / gx;
        p.walk_r = n_tiles % gx;
    }
    {
        const int n_iter = (params.k / 128) / kWarpsK;
        if (n_iter % kLlBPhaseGroups != 0) return false;
        if (!LowLatencyMmaBDesc3D(&p.b_desc, params.ptr_b, params.n, params.k,
                                  kLlBRows<kWarpsN>(), kLlBPhaseGroups))
            return false;
    }
    if (!LowLatencyMmaADesc(&p.a_desc, params.ptr_a, params.m,
                            dual_a ? params.k1 : params.k))
        return false;
    if (dual_a && !LowLatencyMmaADesc(&p.a_desc2, params.ptr_a2, params.m,
                                      params.k - params.k1))
        return false;
    if (params.ptr_sfb_packed != nullptr &&
        !LowLatencyMmaSfbDesc(&p.sfb_desc, params.ptr_sfb_packed, params.n,
                              params.k, params.sfb_stride_mn, kBlockN))
        return false;
    LaunchKernelWithPdl(
        low_latency_mma_gemm_kernel<kBlockN, kWarpsN, kWarpsK, kRowsA, kStages,
                                    kAccStages>,
        dim3(gx, m_tiles), dim3(LowLatencyMmaThreads<kWarpsN, kWarpsK>()),
        static_cast<std::size_t>(smem_bytes), stream, p.pdl != 0, p);
    return cudaGetLastError() == cudaSuccess;
}

template <int kBlockN, int kWarpsN, int kWarpsK, int kRowsA, int kAccStages = 3>
bool LaunchLowLatencyMmaGemmAuto(const LowLatencyMmaGemmParams& params,
                                 int want, int tile_swz, int sm_count,
                                 bool use_pdl, cudaStream_t stream) {
    const int fit =
        LowLatencyMmaFitStages<kWarpsN, kWarpsK, kRowsA, kAccStages>(params.k);
    if (fit < 1) return false;
    int s = (want > 0 && want < fit) ? want : fit;
    s = s >= 32 ? 32 : (s >= 16 ? 16 : (s >= 8 ? 8 : s));
#define HALOPS_LL_LAUNCH_STAGES(S)                                           \
    LaunchLowLatencyMmaGemm<kBlockN, kWarpsN, kWarpsK, kRowsA, (S),          \
                            kAccStages>(params, tile_swz, sm_count, use_pdl, \
                                        stream)
    switch (s) {
        case 1:
            return HALOPS_LL_LAUNCH_STAGES(1);
        case 2:
            return HALOPS_LL_LAUNCH_STAGES(2);
        case 3:
            return HALOPS_LL_LAUNCH_STAGES(3);
        case 8:
            return HALOPS_LL_LAUNCH_STAGES(8);
        case 16:
            return HALOPS_LL_LAUNCH_STAGES(16);
        case 32:
            return HALOPS_LL_LAUNCH_STAGES(32);
        default:
            return HALOPS_LL_LAUNCH_STAGES(4);
    }
#undef HALOPS_LL_LAUNCH_STAGES
}

}  // namespace halops
