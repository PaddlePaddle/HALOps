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
#include <cutlass/arch/barrier.h>

#include <cute/arch/copy_sm80.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/mma_sm100_desc.hpp>

#include "kernel/mla/common/intrinsics.cuh"
#include "kernel/mla/pair_mla_prefill/config.cuh"
#include "kernel/mla/pair_mla_prefill/scheduler_range.cuh"

namespace halops::mla {

// SW128 bf16 UMMA descriptor: K-major (QK^T) or MN-major (PV) operand.
template <class C, cute::UMMA::Major kMajor>
CUTE_DEVICE uint64_t prefill_sw128_desc(const void* p) {
    using namespace cute;
    constexpr int kRowElems = C::kRowPitch / sizeof(cutlass::bfloat16_t);
    const auto ptr = make_smem_ptr(static_cast<const cutlass::bfloat16_t*>(p));
    if constexpr (kMajor == UMMA::Major::K) {
        return UMMA::make_umma_desc<UMMA::Major::K>(make_tensor(
            ptr, tile_to_shape(UMMA::Layout_K_SW128_Atom<cutlass::bfloat16_t>{},
                               Shape<Int<C::kTile>, Int<kRowElems>>{})));
    } else {
        return UMMA::make_umma_desc<UMMA::Major::MN>(make_tensor(
            ptr,
            tile_to_shape(UMMA::Layout_MN_SW128_Atom<cutlass::bfloat16_t>{},
                          Shape<Int<C::kPvN / 2>, Int<C::kTile>>{},
                          Step<_2, _1>{})));
    }
}

// Instruction descriptor of the 2-CTA bf16 UMMA with N columns.
template <class C, int N, cute::UMMA::Major BMajor>
CUTE_DEVICE uint64_t prefill_instr_desc() {
    return cute::UMMA::make_runtime_instr_desc<
        cutlass::bfloat16_t, cutlass::bfloat16_t, float, 2 * C::kHeads, N,
        cute::UMMA::Major::K, BMajor>();
}

// Row r and its swizzled 16-byte chunk c in a SW128 tile of 8-row atoms.
template <class C>
CUTE_DEVICE int prefill_row_offset(int r) {
    return (r / C::kAtomRows) * C::kAtomBytes +
           (r % C::kAtomRows) * C::kRowPitch;
}
template <class C>
CUTE_DEVICE int prefill_sw128_offset(int r, int c) {
    return prefill_row_offset<C>(r) + (c ^ (r % C::kAtomRows)) * C::kChunk;
}

// Eight fp8 e4m3 values times the e8 block scale, as bf16.
template <class C>
CUTE_DEVICE uint4 prefill_dequant8(uint32_t lo, uint32_t hi, uint8_t e8) {
    const uint32_t raw[2] = {lo, hi};
    const auto* f8 = reinterpret_cast<const __nv_fp8x2_e4m3*>(raw);
    const nv_bfloat16 scale =
        __ushort_as_bfloat16(uint16_t(e8) << C::kBf16Mant);
    uint4 out;
    auto* o = reinterpret_cast<nv_bfloat162*>(&out);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < C::kElemsPerChunk / 2; ++i) {
        o[i] = fp8x2_to_bf16x2_with_scale(f8[i], scale);
    }
    return out;
}

// Index warp w resolves keys [32w, 32w + 32) of each tile and gathers their
// rows into the KV stage once PV(i - kKvStages) frees it; c128-only tiles
// load the bf16 copy by TMA when both tokens' lists are prefixes of it.
template <class C>
CUTE_DEVICE void prefill_index_warp(const PrefillParams& p,
                                    const PrefillMaps& maps,
                                    typename C::Smem& sm, int cta, int w,
                                    int lane) {
    const int j = w * C::kWarp + lane;
    PrefillUnitCursor ahead = prefill_start_cursor<C>(p, sm);
    PrefillUnit au;
    int a_left = 0, a_tile = 0, a_bf = -1;
    const auto fetch = [&](PrefillKeyRaw& r, int slot) {
        if (a_left == 0 && ahead.next<C>(p, sm.unit_lens, au)) {
            a_left = au.n;
            a_tile = au.off;
            a_bf = -1;
            if constexpr (!C::kFp8) {
                const int ba = __ldg(p.c128_base + au.g.a);
                const int bb =
                    au.g.has_b ? __ldg(p.c128_base + au.g.a + 1) : ba;
                a_bf = ba >= 0 && bb >= 0 ? ba : -1;
            }
        }
        r = {false, false, 1, -1};
        if (a_left > 0) {
            const int pos = a_tile * C::kTile + j;
            const int seg = pos < au.g.n_c4 ? 0 : 1;
            const int ub = seg == 0 ? pos : pos - au.g.n_c4,
                      ua = ub - seg * p.swa_shift;
            const int* row = p.idx[seg] + int64_t(au.g.a) * p.ld[seg];
            r = {ua >= 0 && ua < au.g.len[seg][0], ub < au.g.len[seg][1], seg,
                 a_bf >= 0 && (a_tile + 1) * C::kTile <= au.g.n_c4
                     ? a_bf + a_tile * C::kTile
                     : -1};
            using Cp4 = cute::SM80_CP_ASYNC_CACHEALWAYS<int>;
            if (r.in_a) Cp4::copy(row[ua], sm.idx_raw[w][slot][lane][0]);
            if (r.in_b) {
                Cp4::copy(row[p.ld[seg] + ub], sm.idx_raw[w][slot][lane][1]);
            }
            ++a_tile;
            --a_left;
        }
        cute::cp_async_fence();
    };
    const int n = sm.start[2];
    PrefillKeyRaw ring[C::kIndexDepth];
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < C::kIndexDepth; ++k) fetch(ring[k], k);
    const auto step = [&](int i, PrefillKeyRaw& r, int slot) {
        cute::cp_async_wait<C::kIndexDepth - 1>();
        __syncwarp();
        const int ka = r.in_a ? sm.idx_raw[w][slot][lane][0] : -1;
        const int kb = r.in_b ? sm.idx_raw[w][slot][lane][1] : -1;
        const int seg = r.seg;
        const bool okb = r.in_b && kb >= 0 && kb < p.pool_tokens[seg];
        const bool oka = r.in_a && ka >= 0 && ka < p.pool_tokens[seg];
        const int key = okb ? kb : ka;
        const bool va = oka && ka == key, any = okb || va;
        const int page = key >> p.page_shift[seg];
        const int row = key & ((1 << p.page_shift[seg]) - 1);
        const int coord = any ? page * p.page_rows[seg] + row : -1;
        const uint32_t valid = __ballot_sync(~0u, cta == 0 ? va : okb);
        const uint32_t has = __ballot_sync(~0u, any);
        const int is = i % C::kIdxStages;
        cutlass::arch::ClusterBarrier::wait(&sm.idx_empty[is],
                                            ((i / C::kIdxStages) & 1) ^ 1);
        if (lane == 0) {
            sm.valid[is][w] = valid;
            sm.any[is][w] = has;
            if (w == 0) sm.bf[is] = !C::kFp8 && r.bf >= 0;
            cutlass::arch::ClusterBarrier::arrive(&sm.idx_full[is]);
        }
        const int s = i % C::kKvStages;
        cutlass::arch::ClusterBarrier::wait(&sm.kv_empty[s],
                                            ((i / C::kKvStages) & 1) ^ 1);
        if (!C::kFp8 && r.bf >= 0) {
            cutlass::arch::cpasync_barrier_arrive_noinc(&sm.raw_full[s]);
            if (lane == 0) {
                cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(
                    &sm.raw_full[s],
                    C::kWarp * C::kHalfDims * int(sizeof(__nv_bfloat16)));
                CUTLASS_PRAGMA_UNROLL
                for (int b = 0; b < C::kLocalBlocks; ++b) {
                    cute::SM90_TMA_LOAD_2D::copy(
                        &maps.bf16, &sm.raw_full[s], C::kBf16L2Hint,
                        sm.kv[s] + b * C::kBlockBytes +
                            prefill_row_offset<C>(w * C::kWarp),
                        cta * C::kHalfDims + b * C::kBlockDims,
                        r.bf + w * C::kWarp);
                }
            }
            __syncwarp();
        } else {
            const uint8_t* const sf =
                p.pool[seg] + int64_t(page) * p.pool_stride[seg] +
                (int64_t(1) << p.page_shift[seg]) * C::kRowBytes +
                row * C::kSfBytes;
            if (any) {
                cute::SM80_CP_ASYNC_CACHEALWAYS<uint64_t>::copy(
                    *reinterpret_cast<const uint64_t*>(sf),
                    *reinterpret_cast<uint64_t*>(sm.sf[s][j]));
            }
            cutlass::arch::cpasync_barrier_arrive_noinc(&sm.raw_full[s]);
            const int src = lane % C::kGroupsPerWarp * C::kGather;
            const int r0 = __shfl_sync(~0u, coord, src);
            const int r1 = __shfl_sync(~0u, coord, src + 1);
            const int r2 = __shfl_sync(~0u, coord, src + 2);
            const int r3 = __shfl_sync(~0u, coord, src + (C::kGather - 1));
            const int gseg = __shfl_sync(~0u, seg, src);
            if (lane == 0) {
                cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(
                    &sm.raw_full[s], C::kWarp * C::raw_pitch(cta));
            }
            __syncwarp();
            if (lane < C::kGroupsPerWarp) {
                const CUtensorMap* const map =
                    cta == 0 ? (gseg == 0 ? &maps.raw[0][0] : &maps.raw[0][1])
                             : (gseg == 0 ? &maps.raw[1][0] : &maps.raw[1][1]);
                tma_gather4_cluster(
                    map, &sm.raw_full[s],
                    sm.kv[s] + (w * C::kWarp + src) * C::raw_pitch(cta),
                    cta * C::kHalfDims / 2, make_int4(r0, r1, r2, r3),
                    C::kKvL2Hint);
            }
            __syncwarp();
        }
        fetch(r, slot);
    };
#pragma unroll 1
    for (int i = 0; i < n; ++i) {
        step(i, ring[0], i % C::kIndexDepth);
        const PrefillKeyRaw r = ring[0];
        CUTLASS_PRAGMA_UNROLL
        for (int k = 0; k + 1 < C::kIndexDepth; ++k) ring[k] = ring[k + 1];
        ring[C::kIndexDepth - 1] = r;
    }
}

// s + x * x for the low (H = 0) or high bf16 of w, as one exact FFMA.
template <int H>
CUTE_DEVICE float prefill_square_add(uint32_t w, float s) {
    float d;
    if constexpr (H == 0) {
        asm("{\n\t.reg .b16 l, h;\n\tmov.b32 {l, h}, %1;\n\t"
            "fma.rn.f32.bf16 %0, l, l, %2;\n\t}"
            : "=f"(d)
            : "r"(w), "f"(s));
    } else {
        asm("{\n\t.reg .b16 l, h;\n\tmov.b32 {l, h}, %1;\n\t"
            "fma.rn.f32.bf16 %0, h, h, %2;\n\t}"
            : "=f"(d)
            : "r"(w), "f"(s));
    }
    return d;
}

// s plus the squares of the eight bf16 of c, in element order.
template <class C>
CUTE_DEVICE float prefill_square_chain(const uint4& c, float s) {
    const uint32_t w[C::kVecLen] = {c.x, c.y, c.z, c.w};
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < C::kVecLen; ++k)
        s = prefill_square_add<1>(w[k], prefill_square_add<0>(w[k], s));
    return s;
}

// (xr, xi) * (fr, fi), rounded as the reference.
CUTE_DEVICE void prefill_rope_pair(uint32_t& w, float fr, float fi) {
    const float xr = bf16_lo(w), xi = bf16_hi(w);
    float re;
    asm("{\n\t.reg .f32 m;\n\tmul.rn.f32 m, %1, %2;\n\tneg.f32 m, m;\n\t"
        "fma.rn.f32 %0, %3, %4, m;\n\t}"
        : "=f"(re)
        : "f"(xi), "f"(fi), "f"(xr), "f"(fr));
    w = pack_bf16_rn(re, fma_rn_f32(xr, fi, mul_rn_f32(xi, fr)));
}

// Dequantizes each KV stage in place (PrefillDequantPlan); bf16-copy
// tiles only pass the stage on.
template <int kCta, class C>
CUTE_DEVICE void prefill_dequant_wg(typename C::Smem& sm, int n, int t) {
    using Plan = PrefillDequantPlan<C, kCta>;
    constexpr int kPitch = C::raw_pitch(kCta);
    for (int i = 0; i < n; ++i) {
        const int s = i % C::kKvStages, is = i % C::kIdxStages;
        cutlass::arch::ClusterBarrier::wait(&sm.idx_full[is],
                                            (i / C::kIdxStages) & 1);
        const uint32_t any[2] = {sm.any[is][0], sm.any[is][1]};
        const bool bf = !C::kFp8 && sm.bf[is];
        cutlass::arch::ClusterBarrier::wait(&sm.raw_full[s],
                                            (i / C::kKvStages) & 1);
        if (bf) {
            __syncwarp();
            if (t % C::kWarp == 0)
                cutlass::arch::ClusterBarrier::arrive(&sm.idx_empty[is]);
            if (t % C::kWarp == 0)
                cutlass::arch::ClusterBarrier::arrive(&sm.kv_full[s], 0, 1);
            continue;
        }
        uint8_t* const stage = sm.kv[s];
        uint4 raw[Plan::kFp8Loads];
        uint8_t e8[Plan::kFp8Loads];
        CUTLASS_PRAGMA_UNROLL
        for (int k = 0; k < Plan::kFp8Loads; ++k) {
            const int e = t + k * C::kWarpGroup;
            const int row = e / Plan::kFp8Chunks, c = e % Plan::kFp8Chunks;
            raw[k] = *reinterpret_cast<const uint4*>(stage + row * kPitch +
                                                     c * C::kChunk);
            const uint8_t sf = sm.sf[s][row][kCta * C::kLocalBlocks +
                                             c / C::kFp8ChunksPerBlock];
            e8[k] = (any[row / C::kWarp] >> (row % C::kWarp)) & 1 ? sf : 0;
        }
        constexpr int kRopeChunks = cute::max(Plan::kRopeChunks, 1);
        uint4 rope[cute::max(Plan::kRopeLoads, 1)];
        CUTLASS_PRAGMA_UNROLL
        for (int k = 0; k < Plan::kRopeLoads; ++k) {
            const int e = t + k * C::kWarpGroup;
            const int row = e / kRopeChunks, c = e % kRopeChunks;
            rope[k] = *reinterpret_cast<const uint4*>(
                stage + row * kPitch + C::kNopeBytes1 + c * C::kChunk);
        }
        __syncwarp();
        if (t % C::kWarp == 0)
            cutlass::arch::ClusterBarrier::arrive(&sm.idx_empty[is]);
        cutlass::arch::NamedBarrier::sync(C::kWarpGroup, C::kDequantBar);
        CUTLASS_PRAGMA_UNROLL
        for (int k = 0; k < Plan::kFp8Loads; ++k) {
            const int e = t + k * C::kWarpGroup;
            const int row = e / Plan::kFp8Chunks, c = e % Plan::kFp8Chunks;
            const bool odd = (c / C::kFp8ChunksPerBlock) & 1;
            const int q0 = c % C::kFp8ChunksPerBlock * 2;
            uint8_t* const block =
                stage + c / C::kFp8ChunksPerBlock * C::kBlockBytes;
            const uint4 first = prefill_dequant8<C>(
                odd ? raw[k].z : raw[k].x, odd ? raw[k].w : raw[k].y, e8[k]);
            const uint4 second = prefill_dequant8<C>(
                odd ? raw[k].x : raw[k].z, odd ? raw[k].y : raw[k].w, e8[k]);
            *reinterpret_cast<uint4*>(
                block + prefill_sw128_offset<C>(row, q0 + odd)) = first;
            *reinterpret_cast<uint4*>(
                block + prefill_sw128_offset<C>(row, q0 + !odd)) = second;
        }
        CUTLASS_PRAGMA_UNROLL
        for (int k = 0; k < Plan::kRopeLoads; ++k) {
            const int e = t + k * C::kWarpGroup;
            const int row = e / kRopeChunks, c = e % kRopeChunks;
            *reinterpret_cast<uint4*>(
                stage + (C::kLocalBlocks - 1) * C::kBlockBytes +
                prefill_sw128_offset<C>(row, c)) = rope[k];
        }
        cutlass::arch::fence_view_async_shared();
        __syncwarp();
        if (t % C::kWarp == 0)
            cutlass::arch::ClusterBarrier::arrive(&sm.kv_full[s], 0, 1);
    }
}

// QK^T(i) into S buffer i & 1 once stage i is dequantized and S(i - 2) read.
template <class C>
CUTE_DEVICE void prefill_qk_warp(const PrefillParams& p, typename C::Smem& sm,
                                 uint32_t tmem) {
    using Mma = PrefillUmmaTs<C, 2 * C::kTile, cute::UMMA::Major::K>;
    const uint64_t idesc =
        prefill_instr_desc<C, 2 * C::kTile, cute::UMMA::Major::K>();
    PrefillUnitCursor cur = prefill_start_cursor<C>(p, sm);
    PrefillUnit u;
    for (int i = 0, unit = 0; cur.next<C>(p, sm.unit_lens, u); ++unit) {
        cutlass::arch::ClusterBarrier::wait(&sm.q_full, unit & 1);
        for (int k = 0; k < u.n; ++k, ++i) {
            const int s = i % C::kKvStages, sb = i % 2;
            cutlass::arch::ClusterBarrier::wait(&sm.kv_full[s],
                                                (i / C::kKvStages) & 1);
            cutlass::arch::ClusterBarrier::wait(&sm.s_empty[sb],
                                                ((i / 2) & 1) ^ 1);
            tcgen05_after_thread_sync();
            CUTLASS_PRAGMA_UNROLL
            for (int ks = 0; ks < C::kQkSteps; ++ks) {
                Mma::fma(
                    tmem + C::kTmemQ + ks * C::kUmmaK / 2,
                    prefill_sw128_desc<C, cute::UMMA::Major::K>(
                        sm.kv[s] + ks / C::kKStepsPerBlock * C::kBlockBytes +
                        ks % C::kKStepsPerBlock * C::kKStepBytes),
                    tmem + C::kTmemS + sb * C::kTmemSCols, ks > 0, idesc);
            }
            cutlass::arch::umma_arrive_multicast_2x1SM(&sm.s_full[sb],
                                                       C::kBothCtas);
            if (k == u.n - 1) {
                cutlass::arch::umma_arrive_multicast_2x1SM(&sm.q_empty,
                                                           C::kBothCtas);
            }
        }
    }
}

// PV(i) once P(i) is written; it frees the KV stage and the P buffer.
template <class C>
CUTE_DEVICE void prefill_pv_warp(const PrefillParams& p, typename C::Smem& sm,
                                 uint32_t tmem) {
    using Mma = PrefillUmmaSs<C, C::kPvN, cute::UMMA::Major::MN>;
    const uint64_t idesc =
        prefill_instr_desc<C, C::kPvN, cute::UMMA::Major::MN>();
    PrefillUnitCursor cur = prefill_start_cursor<C>(p, sm);
    PrefillUnit u;
    for (int i = 0, unit = 0; cur.next<C>(p, sm.unit_lens, u); ++unit) {
        for (int k = 0; k < u.n; ++k, ++i) {
            const int s = i % C::kKvStages, pb = i % 2;
            cutlass::arch::ClusterBarrier::wait(&sm.p_full[pb], (i / 2) & 1);
            CUTLASS_PRAGMA_UNROLL
            for (int m = 0; m < C::kPvMmas; ++m) {
                if (k == 0 && unit > 0)
                    cutlass::arch::ClusterBarrier::wait(&sm.o_empty[m],
                                                        (unit - 1) & 1);
                tcgen05_after_thread_sync();
                CUTLASS_PRAGMA_UNROLL
                for (int ks = 0; ks < C::kPvSteps; ++ks) {
                    Mma::fma(
                        prefill_sw128_desc<C, cute::UMMA::Major::K>(
                            sm.p[pb] + ks * C::kKStepBytes),
                        prefill_sw128_desc<C, cute::UMMA::Major::MN>(
                            sm.kv[s] + m * C::kPvBlocksPerMma * C::kBlockBytes +
                            ks * C::kUmmaK * C::kRowPitch),
                        tmem + C::kTmemO + m * C::kPvN / 2, k > 0 || ks > 0,
                        idesc);
                }
                __syncwarp();
            }
            cutlass::arch::umma_arrive_multicast_2x1SM(&sm.pv_done[pb],
                                                       C::kBothCtas);
            cutlass::arch::umma_arrive_multicast_2x1SM(&sm.kv_empty[s],
                                                       C::kBothCtas);
            if (k == u.n - 1) {
                cutlass::arch::umma_arrive_multicast_2x1SM(&sm.o_full,
                                                           C::kBothCtas);
            }
        }
    }
}

// Softmax thread t owns TMEM lane t (head t % 64, dims half t / 64).
template <class C>
CUTE_DEVICE void prefill_pair_sync(int t) {
    cutlass::arch::NamedBarrier::sync(2 * C::kWarp,
                                      C::kPairBar + t / C::kWarp % 2);
}

// Waits for PV(j).
template <class C>
CUTE_DEVICE void prefill_wait_pv(typename C::Smem& sm, int j) {
    cutlass::arch::ClusterBarrier::wait(&sm.pv_done[j % 2], (j / 2) & 1);
}

// Online softmax of tile i, adding the partner lane's partial S (exchanged
// through bank-swizzled xchg rows). kQnr scales by the head's 1 / rms. O is
// rescaled only after PV(i - 1), and P buffer i % 2 is reused after PV(i - 2).
template <class C>
CUTE_DEVICE void prefill_softmax_tile(const PrefillParams& p,
                                      typename C::Smem& sm, int i, int k,
                                      uint32_t lane_base, int t, int b,
                                      float& m, float& l) {
    const int head = t % C::kHeads, half = t / C::kHeads;
    const float scale_log2 = C::kQnr ? sm.rs[b][head] : p.scale_log2;
    const int is = i % C::kIdxStages, sb = i % 2;
    constexpr int kQuads = C::kHalf / C::kVecLen;
    const uint32_t ph_idx = (i / C::kIdxStages) & 1, ph_s = (i / 2) & 1;
    const bool idx_ok =
        cutlass::arch::ClusterBarrier::test_wait(&sm.idx_full[is], ph_idx, 1);
    const bool s_ok =
        cutlass::arch::ClusterBarrier::test_wait(&sm.s_full[sb], ph_s, 1);
    const bool pv_ok = cutlass::arch::ClusterBarrier::test_wait(
        &sm.pv_done[sb], ph_s ^ 1, i > 1);
    if (!idx_ok) cutlass::arch::ClusterBarrier::wait(&sm.idx_full[is], ph_idx);
    static_assert(C::kHalf == C::kWarp, "one valid word per half");
    const uint32_t valid = sm.valid[is][half];
    if (!s_ok) cutlass::arch::ClusterBarrier::wait(&sm.s_full[sb], ph_s);
    tcgen05_after_thread_sync();
    float4* const peer = reinterpret_cast<float4*>(sm.xchg[t ^ C::kHeads]);
    float own[C::kHalf], other[C::kHalf];
    tmem_ld_32dp32bNx<C::kHalf>(
        lane_base + C::kTmemS + sb * C::kTmemSCols + (half ^ 1) * C::kHalf,
        other);
    tmem_ld_32dp32bNx<C::kHalf>(
        lane_base + C::kTmemS + sb * C::kTmemSCols + half * C::kHalf, own);
    cutlass::arch::fence_view_async_tmem_load();
    CUTLASS_PRAGMA_UNROLL
    for (int q = 0; q < kQuads; ++q) {
        peer[q ^ (t % kQuads)] = make_float4(
            other[C::kVecLen * q], other[C::kVecLen * q + 1],
            other[C::kVecLen * q + 2], other[C::kVecLen * q + C::kVecW]);
    }
    tcgen05_before_thread_sync();
    __syncwarp();
    if (t % C::kWarp == 0) {
        cutlass::arch::ClusterBarrier::arrive(&sm.s_empty[sb], 0, 1);
        cutlass::arch::ClusterBarrier::arrive(&sm.idx_empty[is]);
    }
    prefill_pair_sync<C>(t);
    const float4* const got = reinterpret_cast<const float4*>(sm.xchg[t]);
    float2* const own2 = reinterpret_cast<float2*>(own);
    CUTLASS_PRAGMA_UNROLL
    for (int q = 0; q < kQuads; ++q) {
        const float4 v = got[q ^ ((t ^ C::kHeads) % kQuads)];
        own2[2 * q] = __fadd2_rn(own2[2 * q], make_float2(v.x, v.y));
        own2[2 * q + 1] = __fadd2_rn(own2[2 * q + 1], make_float2(v.z, v.w));
    }
    if (valid != ~0u) {
        CUTLASS_PRAGMA_UNROLL
        for (int c = 0; c < C::kHalf; ++c) {
            if (!((valid >> c) & 1)) own[c] = -CUDART_INF_F;
        }
    }
    float mx[C::kHalf / 2];
    CUTLASS_PRAGMA_UNROLL
    for (int c = 0; c < C::kHalf / 2; ++c)
        mx[c] = fmaxf(own[2 * c], own[2 * c + 1]);
    CUTLASS_PRAGMA_UNROLL
    for (int w = C::kHalf / 2 / 2; w >= 1; w /= 2) {
        CUTLASS_PRAGMA_UNROLL
        for (int c = 0; c < w; ++c) mx[c] = fmaxf(mx[c], mx[c + w]);
    }
    float local = mx[0];
    local *= scale_log2;
    sm.red[t] = local;
    prefill_pair_sync<C>(t);
    const float cur = fmaxf(local, sm.red[t ^ C::kHeads]);
    float factor = 1.0f;
    if (__any_sync(~0u, cur - m > C::kRescaleLog2)) {
        const float m_new = fmaxf(m, cur);
        factor = exp2f(m - m_new);
        m = m_new;
    }
    float2 sum = make_float2(0.0f, 0.0f);
    uint32_t pk[C::kHalf / 2];
    const float2 scale2 = make_float2(scale_log2, scale_log2),
                 neg_m = make_float2(-m, -m);
    CUTLASS_PRAGMA_UNROLL
    for (int c = 0; c < C::kHalf; c += 2) {
        const float2 x = __ffma2_rn(own2[c / 2], scale2, neg_m);
        const float2 e = make_float2(exp2f(x.x), exp2f(x.y));
        sum = __fadd2_rn(sum, e);
        const nv_bfloat162 b2 = __floats2bfloat162_rn(e.x, e.y);
        pk[c / 2] = reinterpret_cast<const uint32_t&>(b2);
    }
    l = l * factor + (sum.x + sum.y);
    if (k > 0 && __any_sync(~0u, factor != 1.0f)) {
        prefill_wait_pv<C>(sm, i - 1);
        tcgen05_after_thread_sync();
        CUTLASS_PRAGMA_UNROLL
        for (int c0 = 0; c0 < C::kHalfDims; c0 += C::kTmemLd) {
            float o[C::kTmemLd];
            tmem_ld_32dp32bNx_wait<C::kTmemLd>(lane_base + C::kTmemO + c0, o);
            CUTLASS_PRAGMA_UNROLL
            for (int c = 0; c < C::kTmemLd; ++c) o[c] *= factor;
            tmem_st_32dp32bNx<C::kTmemLd>(lane_base + C::kTmemO + c0, o);
        }
        cutlass::arch::fence_view_async_tmem_store();
    }
    if (i > 1 && !pv_ok) prefill_wait_pv<C>(sm, i - 2);
    CUTLASS_PRAGMA_UNROLL
    for (int q = 0; q < C::kHalf / C::kElemsPerChunk; ++q) {
        *reinterpret_cast<uint4*>(
            sm.p[sb] + prefill_sw128_offset<C>(
                           head, half * C::kHalf / C::kElemsPerChunk + q)) =
            reinterpret_cast<const uint4*>(pk)[q];
    }
    cutlass::arch::fence_view_async_shared();
    tcgen05_before_thread_sync();
    __syncwarp();
    if (t % C::kWarp == 0)
        cutlass::arch::ClusterBarrier::arrive(&sm.p_full[sb], 0, 1);
}

// Per unit: online softmax over its tiles, row max and sum to the Q/O WG.
template <class C>
CUTE_DEVICE void prefill_softmax_wg(const PrefillParams& p,
                                    typename C::Smem& sm, uint32_t tmem,
                                    int t) {
    const uint32_t lane_base =
        tmem + ((t / C::kWarp * C::kWarp) << C::kTmemLaneShift);
    PrefillUnitCursor cur = prefill_start_cursor<C>(p, sm);
    PrefillUnit u;
    for (int i = 0, unit = 0; cur.next<C>(p, sm.unit_lens, u); ++unit) {
        const int b = unit % 2;
        if constexpr (C::kQnr)
            cutlass::arch::ClusterBarrier::wait(&sm.rs_full[b], (unit / 2) & 1);
        float m = C::kMaxInit, l = 0.0f;
        for (int k = 0; k < u.n; ++k, ++i) {
            prefill_softmax_tile<C>(p, sm, i, k, lane_base, t, b, m, l);
        }
        cutlass::arch::ClusterBarrier::wait(&sm.ml_empty[b],
                                            ((unit / 2) & 1) ^ 1);
        sm.ml[b][t] = make_float2(m, l);
        __syncwarp();
        if (t % C::kWarp == 0)
            cutlass::arch::ClusterBarrier::arrive(&sm.ml_full[b]);
    }
}

}  // namespace halops::mla
