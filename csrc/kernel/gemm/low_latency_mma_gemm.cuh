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

#include <algorithm>
#include <array>
#include <cstddef>
#include <iterator>
#include <utility>

#include "kernel/gemm/gemm_utils.h"
#include "kernel/gemm/low_latency_utils.cuh"
#include "kernel/gemm/pdl.cuh"

namespace halops {

// One kernel instance, like GemmTile: BlockN columns per CTA tile, WarpsK K
// splits, RowsA resident A rows (the M tier) and a Stages-deep B ring.
template <int BlockN, int WarpsK, int RowsA, int Stages>
struct LlTile : LlGemmConfig {
    static_assert(BlockN % KMmaN == 0, "one warp owns one mma N tile");
    static_assert(Stages >= 1 && Stages <= KMaxStages,
                  "Stages must be 1..KMaxStages");
    static_assert(RowsA == 8 || RowsA == 16, "RowsA is the M tier: 8 or 16");
    static constexpr int KBlockN = BlockN;
    static constexpr int KWarpsN = BlockN / KMmaN;
    static constexpr int KWarpsK = WarpsK;
    static constexpr int KRowsA = RowsA;
    static constexpr int KStages = Stages;
    static constexpr int KOuts = RowsA == 16 ? 4 : 2;
    static constexpr int KAAtoms = RowsA / 8;
    static constexpr int KATileBytes = KAAtoms * KAtomBytes;
    static constexpr int KBTileBytes = KWarpsN * KAtomBytes;
    static constexpr int KBPhaseBytes = KBPhaseGroups * KBTileBytes;
    static constexpr int KAccCols = KAccStages * KWarpsN;
    static constexpr int KAccSlot = WarpsK * 32 * KOuts;
    // Warp layout: consumer warps, then one producer warp, then KWarpsN
    // epilogue warps.
    static constexpr int KConsumerWarps = KWarpsN * WarpsK;
    static constexpr int KThreads = 32 * (KConsumerWarps + 1 + KWarpsN);
};

// Per-CTA state shared by the three warp roles: the shared-memory regions,
// their barriers and the loop bounds.
struct LlCtx {
    const LowLatencyMmaGemmParams& params;
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
    int sf_pitch, sfb_slot_ints;
    int lane, gid, tig, m_base;
    int n_tiles, n_groups, n_iter, n_phase, k1_groups;
    bool dual_a;
};

template <class Tile>
__device__ __forceinline__ LlCtx
ll_make_ctx(const LowLatencyMmaGemmParams& params, uint8_t* smem) {
    const LlSmemPlan plan = ll_smem_plan(Tile::KWarpsN, Tile::KWarpsK,
                                         Tile::KRowsA, params.k, Tile::KStages);
    LlCtx c{.params = params};
    c.n_groups = params.k / Tile::KGroupK;
    c.a_res = smem;
    c.b_ring = smem + plan.b_off;
    c.sfa_res = reinterpret_cast<int32_t*>(smem + plan.sfa_off);
    c.sfb_ring = reinterpret_cast<int32_t*>(smem + plan.sfb_off);
    c.a_full = reinterpret_cast<Barrier*>(smem + plan.bars_off);
    c.sfa_bar = c.a_full + c.n_groups;
    c.sfb_full = c.sfa_bar + 1;
    c.sfb_empty = c.sfb_full + Tile::KSfbStages;
    c.full_bar = c.sfb_empty + Tile::KSfbStages;
    c.empty_bar = c.full_bar + Tile::KWarpsK * Tile::KStages;
    c.acc_full = c.empty_bar + Tile::KStages;
    c.acc_empty = c.acc_full + Tile::KAccCols;
    c.acc_buf = reinterpret_cast<float*>(smem + plan.acc_off);
    c.sf_pitch = ll_scale_pitch(c.n_groups);
    c.sfb_slot_ints = ll_sfb_slot_ints(Tile::KBlockN, params.k);
    c.lane = threadIdx.x & 31;
    c.gid = c.lane >> 2;
    c.tig = c.lane & 3;
    c.m_base = blockIdx.y * Tile::KRowsA;
    c.n_tiles = params.n / Tile::KBlockN;
    c.n_iter = c.n_groups / Tile::KWarpsK;
    c.n_phase = (c.n_iter + Tile::KBPhaseGroups - 1) / Tile::KBPhaseGroups;
    c.k1_groups = params.k1 / Tile::KGroupK;
    c.dual_a = params.ptr_a2 != nullptr;
    return c;
}

template <class Tile>
__device__ __forceinline__ void ll_init_barriers(const LlCtx& c) {
    for (int i = c.lane; i < c.n_groups; i += 32) (c.a_full + i)->init(1);
    if (c.lane == 0) c.sfa_bar->init(1);
    for (int i = c.lane; i < Tile::KSfbStages; i += 32) {
        (c.sfb_full + i)->init(1);
        (c.sfb_empty + i)->init(Tile::KConsumerWarps);
    }
    for (int i = c.lane; i < Tile::KWarpsK * Tile::KStages; i += 32)
        (c.full_bar + i)->init(1);
    for (int i = c.lane; i < Tile::KStages; i += 32)
        (c.empty_bar + i)->init(Tile::KConsumerWarps);
    for (int i = c.lane; i < Tile::KAccCols; i += 32) {
        (c.acc_full + i)->init(Tile::KWarpsK);
        (c.acc_empty + i)->init(1);
    }
    cutlass::arch::fence_view_async_shared();
}

template <class Tile, bool UsePdl>
__device__ __forceinline__ void ll_producer_warp(const LlCtx& c) {
    // Lane r of the producer feeds the B ring of K-split warp r.
    const bool act = c.lane < Tile::KWarpsK;
    Barrier* const my_full = c.full_bar + c.lane * Tile::KStages;
    uint8_t* const my_b =
        c.b_ring + c.lane * Tile::KStages * Tile::KBPhaseBytes;

    auto issue_sfa = [&]() {
        const uint32_t row_bytes = ll_sfa_g32_row_bytes(c.params.k);
        const int rows = min(c.params.m - c.m_base, Tile::KSfaRows);
        const uint32_t b1 = c.dual_a ? c.params.k1 / Tile::KScaleGroupK : 0;
        const uint32_t b2 =
            c.dual_a ? ll_sfa_g32_row_bytes(c.params.k - c.params.k1) : 0;
        const uint32_t per_row = c.dual_a ? b1 + b2 : row_bytes;
        if (c.lane == 0) c.sfa_bar->arrive_and_expect_tx(per_row * rows);
        __syncwarp();
        if (c.dual_a) {
            for (int r = c.lane; r < rows; r += 32) {
                cute::SM90_BULK_COPY_G2S::copy(
                    c.params.ptr_sfa + (c.m_base + r) * c.params.sfa_stride_mn,
                    reinterpret_cast<uint64_t*>(c.sfa_bar),
                    c.sfa_res + r * c.sf_pitch, b1);
                cute::SM90_BULK_COPY_G2S::copy(
                    c.params.ptr_sfa2 +
                        (c.m_base + r) * c.params.sfa2_stride_mn,
                    reinterpret_cast<uint64_t*>(c.sfa_bar),
                    c.sfa_res + r * c.sf_pitch + c.k1_groups, b2);
            }
        } else {
            for (int r = c.lane; r < rows; r += 32)
                cute::SM90_BULK_COPY_G2S::copy(
                    c.params.ptr_sfa + (c.m_base + r) * c.params.sfa_stride_mn,
                    reinterpret_cast<uint64_t*>(c.sfa_bar),
                    c.sfa_res + r * c.sf_pitch, row_bytes);
        }
    };
    if constexpr (!UsePdl) issue_sfa();

    int st = 0;
    uint32_t spass = 0;
    bool sprimed = false;
    int buf = 0;
    uint32_t pass = 0;
    bool primed = false;
    bool first = true;
    for (int tile = blockIdx.x; tile < c.n_tiles; tile += gridDim.x) {
        const int tile_n0 = tile * Tile::KBlockN;
        if (sprimed) (c.sfb_empty + st)->wait((spass - 1) & 1);
        if (c.lane == 0) {
            Barrier* const slot_full = c.sfb_full + st;
            slot_full->arrive_and_expect_tx(4 * Tile::KBlockN * c.sf_pitch);
            cute::SM90_TMA_LOAD_2D::copy(
                &c.params.sfb_desc, reinterpret_cast<uint64_t*>(slot_full),
                Tile::KTmaCacheHint, c.sfb_ring + st * c.sfb_slot_ints, 0,
                tile_n0);
        }
        for (int phase = 0; phase < c.n_phase; ++phase) {
            const int j0 = phase * Tile::KBPhaseGroups;
            auto issue_a_phase = [&]() {
                if (!first || c.lane >= Tile::KWarpsK) return;
                for (int q = 0; q < Tile::KBPhaseGroups; ++q) {
                    const int j = j0 + q;
                    if (j >= c.n_iter) break;
                    const int g = ll_kb(c.lane, j, c.n_iter);
                    const bool a2 = c.dual_a && g >= c.k1_groups;
                    const CUtensorMap* const adesc =
                        a2 ? &c.params.a_desc2 : &c.params.a_desc;
                    const int acrd0 =
                        (a2 ? g - c.k1_groups : g) * Tile::KGroupK;
                    (c.a_full + g)->arrive_and_expect_tx(Tile::KATileBytes);
#pragma unroll
                    for (int t = 0; t < Tile::KAAtoms; ++t)
                        cute::SM90_TMA_LOAD_2D::copy(
                            adesc, reinterpret_cast<uint64_t*>(c.a_full + g),
                            Tile::KTmaCacheHint,
                            c.a_res + g * Tile::KATileBytes +
                                t * Tile::KAtomBytes,
                            acrd0, c.m_base + t * 8);
                }
            };
            if (primed) (c.empty_bar + buf)->wait((pass - 1) & 1);
            if (act) {
                Barrier* const fb = my_full + buf;
                const int live_groups = c.n_iter - j0 < Tile::KBPhaseGroups
                                            ? c.n_iter - j0
                                            : Tile::KBPhaseGroups;
                fb->arrive_and_expect_tx(live_groups * Tile::KBTileBytes);
                cute::SM90_TMA_LOAD_3D::copy(
                    &c.params.b_desc, reinterpret_cast<uint64_t*>(fb),
                    Tile::KTmaCacheHint, my_b + buf * Tile::KBPhaseBytes, 0,
                    tile_n0, ll_kb(c.lane, j0, c.n_iter));
            }
            // Under PDL, B and sfb are issued before the dependency wait; A
            // and sfa come from the upstream kernel and are issued after it.
            if constexpr (UsePdl) {
                if (first && phase == 0) {
                    PdlWaitForDependents();
                    issue_sfa();
                }
            }
            issue_a_phase();
            if (++buf == Tile::KStages) {
                buf = 0;
                ++pass;
                primed = true;
            }
        }
        first = false;
        if (++st == Tile::KSfbStages) {
            st = 0;
            ++spass;
            sprimed = true;
        }
    }
}

template <class Tile>
__device__ __forceinline__ void ll_epilogue_warp(const LlCtx& c) {
    const int warp = threadIdx.x >> 5;
    const int warp_n = warp - Tile::KConsumerWarps - 1;
    const int row_c0 = c.m_base + c.gid;
    const int row_c1 = row_c0 + 8;
    c10::BFloat16* const ptr_d = c.params.ptr_d;
    c10::BFloat16* const ptr_d2 = c.params.ptr_d2;
    const int n_split = c.params.n_split;
    int as = 0;
    uint32_t apar = 0;
    for (int tile = blockIdx.x; tile < c.n_tiles; tile += gridDim.x) {
        (c.acc_full + as * Tile::KWarpsN + warp_n)->wait(apar);
        const float* const slot =
            c.acc_buf + (as * Tile::KWarpsN + warp_n) * Tile::KAccSlot +
            c.lane * Tile::KOuts;
        float out[Tile::KOuts];
#pragma unroll
        for (int i = 0; i < Tile::KOuts; ++i) out[i] = slot[i];
#pragma unroll
        for (int wk = 1; wk < Tile::KWarpsK; ++wk) {
            const float* const o = slot + wk * 32 * Tile::KOuts;
#pragma unroll
            for (int i = 0; i < Tile::KOuts; ++i) out[i] += o[i];
        }
        const int col_c =
            tile * Tile::KBlockN + warp_n * Tile::KMmaN + 2 * c.tig;
        const bool seg2 = ptr_d2 != nullptr && col_c >= n_split;
        c10::BFloat16* const dst = seg2 ? ptr_d2 : ptr_d;
        const int seg_n = ptr_d2 != nullptr
                              ? (seg2 ? c.params.n - n_split : n_split)
                              : c.params.n;
        const int seg_col = seg2 ? col_c - n_split : col_c;
        if (seg_col + 1 < seg_n) {
            if (row_c0 < c.params.m)
                *reinterpret_cast<__nv_bfloat162*>(dst + row_c0 * seg_n +
                                                   seg_col) =
                    __floats2bfloat162_rn(out[0], out[1]);
            if constexpr (Tile::KRowsA == 16) {
                if (row_c1 < c.params.m)
                    *reinterpret_cast<__nv_bfloat162*>(dst + row_c1 * seg_n +
                                                       seg_col) =
                        __floats2bfloat162_rn(out[2], out[3]);
            }
        }
        __syncwarp();
        if (c.lane == 0) (c.acc_empty + as * Tile::KWarpsN + warp_n)->arrive();
        if (++as == Tile::KAccStages) {
            as = 0;
            apar ^= 1;
        }
    }
}

template <class Tile>
__device__ __forceinline__ void ll_consumer_warp(const LlCtx& c) {
    const int warp = threadIdx.x >> 5;
    const int warp_n = warp % Tile::KWarpsN;
    const int warp_k = warp / Tile::KWarpsN;
    Barrier* const my_full = c.full_bar + warp_k * Tile::KStages;
    const uint8_t* const my_b =
        c.b_ring + warp_k * Tile::KStages * Tile::KBPhaseBytes;
    const int row_c0 = c.m_base + c.gid;
    const int row_c1 = row_c0 + 8;
    const bool has_a0 = row_c0 < c.params.m;
    const bool has_a1 = Tile::KRowsA == 16 && row_c1 < c.params.m;

    const int sfb_col = warp_n * Tile::KMmaN + 2 * c.tig;
    int buf = 0;
    uint32_t par = 0;
    int as = 0;
    uint32_t apass = 0;
    bool aprimed = false;
    bool first_tile = true;
    int st = 0;
    uint32_t sparity = 0;

    for (int tile = blockIdx.x; tile < c.n_tiles; tile += gridDim.x) {
        float out[Tile::KOuts] = {};
        const int32_t* const my_sfb = c.sfb_ring + st * c.sfb_slot_ints;

        (c.sfb_full + st)->wait(sparity);
        for (int phase = 0; phase < c.n_phase; ++phase) {
            const int j0 = phase * Tile::KBPhaseGroups;
            // A stays resident for every tile: its barriers complete once.
            if (first_tile) {
                for (int q = 0; q < Tile::KBPhaseGroups; ++q) {
                    const int j = j0 + q;
                    if (j >= c.n_iter) break;
                    (c.a_full + ll_kb(warp_k, j, c.n_iter))->wait(0);
                }
            }
            (my_full + buf)->wait(par);
            __syncwarp();

            for (int q = 0; q < Tile::KBPhaseGroups; ++q) {
                const int j = j0 + q;
                if (j >= c.n_iter) break;
                const uint8_t* const tile_a =
                    c.a_res + ll_kb(warp_k, j, c.n_iter) * Tile::KATileBytes;
                const uint8_t* const tile_b = my_b + buf * Tile::KBPhaseBytes +
                                              q * Tile::KBTileBytes +
                                              warp_n * Tile::KAtomBytes;

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
                    if constexpr (Tile::KRowsA == 16) {
                        frag_a[1] = lds_u32(ra + Tile::KAtomBytes +
                                            swz128(c.gid, 2 * s));
                        frag_a[3] = lds_u32(ra + Tile::KAtomBytes +
                                            swz128(c.gid, 2 * s + 1));
                    } else {
                        frag_a[1] = 0u;
                        frag_a[3] = 0u;
                    }
                    frag_b[0] = lds_u32(rb + swz128(c.gid, 2 * s));
                    frag_b[1] = lds_u32(rb + swz128(c.gid, 2 * s + 1));
                    mma_m16n8k32_e4m3(acc[s], frag_a, frag_b);
                }

                const int kb = ll_kb(warp_k, j, c.n_iter);
                if (first_tile && j == 0) c.sfa_bar->wait(0);
                const int32_t wb0 = my_sfb[sfb_col * c.sf_pitch + kb];
                const int32_t wb1 = my_sfb[(sfb_col + 1) * c.sf_pitch + kb];
                const int32_t wa0 =
                    has_a0 ? c.sfa_res[c.gid * c.sf_pitch + kb] : 0;
                const int32_t wa1 =
                    has_a1 ? c.sfa_res[(c.gid + 8) * c.sf_pitch + kb] : 0;
#pragma unroll
                for (int s = 0; s < 4; ++s) {
                    const float sb0 = decode_packed_ue8m0_word(wb0, s);
                    const float sb1 = decode_packed_ue8m0_word(wb1, s);
                    const float sa0 = decode_packed_ue8m0_word(wa0, s);
                    out[0] += acc[s][0] * (sa0 * sb0);
                    out[1] += acc[s][1] * (sa0 * sb1);
                    if constexpr (Tile::KRowsA == 16) {
                        const float sa1 = decode_packed_ue8m0_word(wa1, s);
                        out[2] += acc[s][2] * (sa1 * sb0);
                        out[3] += acc[s][3] * (sa1 * sb1);
                    }
                }
            }
            __syncwarp();
            if (c.lane == 0) (c.empty_bar + buf)->arrive();
            if (++buf == Tile::KStages) {
                buf = 0;
                par ^= 1;
            }
        }
        __syncwarp();
        if (c.lane == 0) (c.sfb_empty + st)->arrive();
        if (++st == Tile::KSfbStages) {
            st = 0;
            sparity ^= 1;
        }

        if (aprimed)
            (c.acc_empty + as * Tile::KWarpsN + warp_n)->wait((apass - 1) & 1);
        float* const slot = c.acc_buf +
                            (as * Tile::KWarpsN + warp_n) * Tile::KAccSlot +
                            (warp_k * 32 + c.lane) * Tile::KOuts;
        if constexpr (Tile::KOuts == 4) {
            *reinterpret_cast<float4*>(slot) =
                make_float4(out[0], out[1], out[2], out[3]);
        } else {
            *reinterpret_cast<float2*>(slot) = make_float2(out[0], out[1]);
        }
        __syncwarp();
        if (c.lane == 0) (c.acc_full + as * Tile::KWarpsN + warp_n)->arrive();
        if (++as == Tile::KAccStages) {
            as = 0;
            ++apass;
            aprimed = true;
        }
        first_tile = false;
    }
}

template <class Tile, bool UsePdl>
__global__ void __launch_bounds__(Tile::KThreads, 1)
    low_latency_mma_gemm_kernel(
        const __grid_constant__ LowLatencyMmaGemmParams params) {
    extern __shared__ __align__(16) uint8_t smem[];
    const LlCtx c = ll_make_ctx<Tile>(params, smem);
    if (threadIdx.x < 32) ll_init_barriers<Tile>(c);
    __syncthreads();
    // Only scheduling: a dependent grid's griddepcontrol.wait still waits for
    // this grid to complete, so its prologue may overlap the whole kernel.
    if constexpr (UsePdl) cudaTriggerProgrammaticLaunchCompletion();

    const int warp = threadIdx.x / 32;
    if (warp < Tile::KConsumerWarps) {
        ll_consumer_warp<Tile>(c);
    } else if (warp == Tile::KConsumerWarps) {
        ll_producer_warp<Tile, UsePdl>(c);
    } else {
        ll_epilogue_warp<Tile>(c);
    }
}

inline bool LlEncodeDesc(CUtensorMap* out, const void* ptr, int rank,
                         CUtensorMapDataType dtype, CUtensorMapSwizzle swizzle,
                         const uint64_t* gdim, const uint64_t* gstride,
                         const uint32_t* bdim) {
    const uint32_t estride[3] = {1u, 1u, 1u};
    return cuTensorMapEncodeTiled(
               out, dtype, rank, const_cast<void*>(ptr), gdim, gstride, bdim,
               estride, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
               CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}

inline bool LowLatencyMmaBDesc3D(CUtensorMap* out, const void* ptr_b,
                                 uint64_t n, uint64_t k, uint32_t rows,
                                 uint32_t groups) {
    const uint64_t gdim[3] = {LlGemmConfig::KGroupK, n,
                              k / LlGemmConfig::KGroupK};
    const uint64_t gstride[2] = {k, LlGemmConfig::KGroupK};
    const uint32_t bdim[3] = {LlGemmConfig::KGroupK, rows, groups};
    return LlEncodeDesc(out, ptr_b, 3, CU_TENSOR_MAP_DATA_TYPE_UINT8,
                        CU_TENSOR_MAP_SWIZZLE_128B, gdim, gstride, bdim);
}

inline bool LowLatencyMmaSfbDesc(CUtensorMap* out, const void* ptr_sfb,
                                 uint64_t n, uint64_t k, uint64_t stride,
                                 uint32_t cols) {
    const uint32_t pitch = ll_scale_pitch(k / LlGemmConfig::KGroupK);
    const uint64_t gdim[2] = {k / LlGemmConfig::KGroupK, n};
    const uint64_t gstride[1] = {stride};
    const uint32_t bdim[2] = {pitch, cols};
    return LlEncodeDesc(out, ptr_sfb, 2, CU_TENSOR_MAP_DATA_TYPE_INT32,
                        CU_TENSOR_MAP_SWIZZLE_NONE, gdim, gstride, bdim);
}

// One TMA box is one 128B-swizzled atom: 8 rows of a 128-wide k-block.
inline bool LowLatencyMmaADesc(CUtensorMap* out, const void* ptr_a, uint64_t m,
                               uint64_t k) {
    const uint64_t gdim[2] = {k, m};
    const uint64_t gstride[1] = {k};
    const uint32_t bdim[2] = {LlGemmConfig::KGroupK,
                              LlGemmConfig::KAtomBytes / LlGemmConfig::KGroupK};
    return LlEncodeDesc(out, ptr_a, 2, CU_TENSOR_MAP_DATA_TYPE_UINT8,
                        CU_TENSOR_MAP_SWIZZLE_128B, gdim, gstride, bdim);
}

// The host entry has already validated shapes, scales and the shared memory
// fit; what is left to fail here is descriptor encoding.
template <class Tile>
void LaunchLowLatencyMmaGemm(const LowLatencyMmaGemmParams& params,
                             int sm_count, bool use_pdl, cudaStream_t stream) {
    const int smem_bytes = ll_smem_plan(Tile::KWarpsN, Tile::KWarpsK,
                                        Tile::KRowsA, params.k, Tile::KStages)
                               .total_bytes;
    const auto kernel = use_pdl ? &low_latency_mma_gemm_kernel<Tile, true>
                                : &low_latency_mma_gemm_kernel<Tile, false>;
    HALOPS_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
    HALOPS_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
        cudaSharedmemCarveoutMaxShared));

    const int m_tiles = (params.m + Tile::KRowsA - 1) / Tile::KRowsA;
    const int n_tiles = params.n / Tile::KBlockN;
    // Every CTA owns at least one tile: an idle CTA would exit with its sfa
    // bulk copy still in flight.
    const int gx = std::clamp(sm_count / m_tiles, 1, n_tiles);
    LowLatencyMmaGemmParams p = params;
    TORCH_CHECK(
        LowLatencyMmaBDesc3D(&p.b_desc, params.ptr_b, params.n, params.k,
                             Tile::KBlockN, Tile::KBPhaseGroups),
        "low_latency_mma_gemm: the weight TMA descriptor was rejected "
        "(weight must be 16-byte aligned)");
    TORCH_CHECK(LowLatencyMmaADesc(&p.a_desc, params.ptr_a, params.m,
                                   params.ptr_a2 ? params.k1 : params.k),
                "low_latency_mma_gemm: the input TMA descriptor was rejected "
                "(input must be 16-byte aligned)");
    TORCH_CHECK(
        !params.ptr_a2 || LowLatencyMmaADesc(&p.a_desc2, params.ptr_a2,
                                             params.m, params.k - params.k1),
        "low_latency_mma_gemm: the input2 TMA descriptor was rejected "
        "(input2 must be 16-byte aligned)");
    TORCH_CHECK(
        LowLatencyMmaSfbDesc(&p.sfb_desc, params.ptr_sfb, params.n, params.k,
                             params.sfb_stride_mn, Tile::KBlockN),
        "low_latency_mma_gemm: the sfb TMA descriptor was rejected for K=",
        params.k);
    LaunchKernelWithPdl(kernel, dim3(gx, m_tiles), dim3(Tile::KThreads),
                        smem_bytes, stream, use_pdl, p);
}

using LlLauncher = void (*)(const LowLatencyMmaGemmParams&, int, bool,
                            cudaStream_t);

template <std::size_t I>
constexpr LlLauncher LlLauncherAt() {
    constexpr std::size_t kStages = std::size(kLlStageTiers);
    constexpr std::size_t kRows = std::size(kLlRowsTiers);
    constexpr LlTileSpec kTile = kLlTiles[I / (kStages * kRows)];
    return &LaunchLowLatencyMmaGemm<
        LlTile<kTile.block_n, kTile.warps_k, kLlRowsTiers[I / kStages % kRows],
               kLlStageTiers[I % kStages]>>;
}

template <std::size_t... I>
constexpr std::array<LlLauncher, sizeof...(I)> MakeLlLaunchers(
    std::index_sequence<I...>) {
    return {LlLauncherAt<I>()...};
}

inline constexpr auto kLlLaunchers = MakeLlLaunchers(
    std::make_index_sequence<std::size(kLlTiles) * std::size(kLlRowsTiers) *
                             std::size(kLlStageTiers)>{});

}  // namespace halops
