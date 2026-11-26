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

#include <math_constants.h>

#include <cute/arch/copy_sm80.hpp>

#include "kernel/mla/pair_mla/decode/scheduler_range.cuh"

namespace halops::mla {

// Index warp w resolves keys [32w, 32w + 32) of every tile, one per lane:
// the union key, this CTA's token validity, the TMA row coordinate; then it
// gathers its 8 four-row groups into the tile's KV stage and copies the 8
// scale bytes of each key with cp.async, both completing on raw_full.  The
// raw indices of kIndexDepth tiles ahead are in flight meanwhile.
template <class C>
CUTE_DEVICE void decode_index_warp(const DecodeParams& p,
                                   const DecodeMaps& maps, typename C::Smem& sm,
                                   int cta, int w, int lane, int cluster) {
    const int j = w * kWarp + lane;
    DecodeUnitCursor<C> ahead = decode_start_cursor<C>(p, sm, cluster);
    DecodeUnit au;
    int a_left = 0, a_tile = 0;
    const auto fetch = [&](DecodeKeyRaw& r, int slot) {
        if (a_left == 0 && ahead.next(p, sm.unit_lens, au)) {
            a_left = au.n;
            a_tile = au.off;
        }
        r = {false, false, 1};
        if (a_left > 0) {
            const int pos = a_tile * C::kTile + j;
            const int seg = pos < au.g.n_c4 ? 0 : 1;
            const int ub = seg == 0 ? pos : pos - au.g.n_c4, ua = ub - seg;
            const int* row = p.idx[seg] + int64_t(au.g.a) * p.ld[seg];
            r = {ua >= 0 && ua < au.g.len[seg][0], ub < au.g.len[seg][1], seg};
            if (r.in_a)
                decode_cp_async4(&sm.idx_raw[w][slot][lane][0], row + ua);
            if (r.in_b) {
                decode_cp_async4(&sm.idx_raw[w][slot][lane][1],
                                 row + p.ld[seg] + ub);
            }
            ++a_tile;
            --a_left;
        }
        cute::cp_async_fence();
    };
    const int n = sm.start[2];
    if constexpr (C::kDsa) {
        if (cta == 1) {
            CUTLASS_PRAGMA_UNROLL
            for (int s = 0; s < C::kStages; ++s) {
                CUTLASS_PRAGMA_UNROLL
                for (int c = 0; c < C::kDsaRopeBytes / kChunk; ++c) {
                    *reinterpret_cast<uint4*>(sm.rope_kv[s] +
                                              pair_sw128_offset(j, c)) =
                        uint4{0, 0, 0, 0};
                }
            }
            cutlass::arch::fence_view_async_shared();
        }
    }
    DecodeKeyRaw ring[C::kIndexDepth];
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < C::kIndexDepth; ++k) fetch(ring[k], k);
    const auto step = [&](int i, DecodeKeyRaw& r, int slot) {
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
        DecodeBar::wait(&sm.idx_empty[is], ((i / C::kIdxStages) & 1) ^ 1);
        if (lane == 0) {
            sm.valid[is][w] = valid;
            sm.any[is][w] = has;
            DecodeBar::arrive(&sm.idx_full[is]);
        }
        const int s = i % C::kStages;
        DecodeBar::wait(&sm.kv_empty[s], ((i / C::kStages) & 1) ^ 1);
        if constexpr (C::kDsa) {
            if (any) {
                const uint8_t* const base = p.pool[seg] +
                                            int64_t(page) * p.pool_stride[seg] +
                                            int64_t(row) * C::kDsaRowBytes;
                decode_cp_async8(
                    sm.dsa_sc[s][j],
                    base + C::kDsaScaleOff +
                        cta * (C::kDsaScales / 2) * int(sizeof(float)));
                if (cta == 0) {
                    CUTLASS_PRAGMA_UNROLL
                    for (int c = 0; c < C::kDsaRopeBytes / kChunk; ++c) {
                        decode_cp_async16(
                            sm.rope_kv[s] + pair_sw128_offset(j, c),
                            base + C::kDsaRopeOff + c * kChunk);
                    }
                }
            }
            decode_cp_async_arrive_noinc(&sm.raw_full[s]);
            const int src = lane % C::kGroupsPerWarp * kTmaGatherRows;
            const int r0 = __shfl_sync(~0u, coord, src);
            const int r1 = __shfl_sync(~0u, coord, src + 1);
            const int r2 = __shfl_sync(~0u, coord, src + 2);
            const int r3 = __shfl_sync(~0u, coord, src + kTmaGatherRows - 1);
            if (lane == 0) {
                cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(
                    &sm.raw_full[s], kWarp * C::kDsaNopeHalf);
            }
            __syncwarp();
            if (lane < C::kGroupsPerWarp) {
                decode_tma_gather4(
                    &maps.raw[cta][0], &sm.raw_full[s],
                    sm.kv[s] + (w * kWarp + src) * C::kDsaNopeHalf,
                    cta * C::kHalfDims / 2, r0, r1, r2, r3);
            }
            __syncwarp();
        } else {
            const uint8_t* const sf =
                p.pool[seg] + int64_t(page) * p.pool_stride[seg] +
                (int64_t(1) << p.page_shift[seg]) * C::kRowBytes +
                row * C::kSfBytes;
            if (any) decode_cp_async8(sm.sf[s][j], sf);
            decode_cp_async_arrive_noinc(&sm.raw_full[s]);
            const int src = lane % C::kGroupsPerWarp * kTmaGatherRows;
            const int r0 = __shfl_sync(~0u, coord, src);
            const int r1 = __shfl_sync(~0u, coord, src + 1);
            const int r2 = __shfl_sync(~0u, coord, src + 2);
            const int r3 = __shfl_sync(~0u, coord, src + kTmaGatherRows - 1);
            const int gseg = __shfl_sync(~0u, seg, src);
            if (lane == 0) {
                cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(
                    &sm.raw_full[s], kWarp * decode_raw_pitch<C>(cta));
            }
            __syncwarp();
            if (lane < C::kGroupsPerWarp) {
                const CUtensorMap* const map =
                    cta == 0 ? (gseg == 0 ? &maps.raw[0][0] : &maps.raw[0][1])
                             : (gseg == 0 ? &maps.raw[1][0] : &maps.raw[1][1]);
                decode_tma_gather4(
                    map, &sm.raw_full[s],
                    sm.kv[s] + (w * kWarp + src) * decode_raw_pitch<C>(cta),
                    cta * C::kHalfDims / 2, r0, r1, r2, r3);
            }
            __syncwarp();
        }
        fetch(r, slot);
    };
    for (int i = 0; i < n; i += C::kIndexDepth) {
        CUTLASS_PRAGMA_UNROLL
        for (int k = 0; k < C::kIndexDepth; ++k) {
            if (i + k < n) step(i + k, ring[k], k);
        }
    }
}

// Thread t of the dequant warp group converts the 16-byte chunks e = t + 128k
// of the gathered rows in place: all reads, a warp-group barrier, then the
// bf16 writes.  A wavefront (8 lanes) reads 8 consecutive chunks; chunks of an
// odd 64-dim block store their two bf16 halves in the other order, so its
// stores also hit 8 distinct bank groups.
template <int kCta, class C>
CUTE_DEVICE void decode_dequant_wg(typename C::Smem& sm, int n, int t) {
    if constexpr (C::kDsa) {
        constexpr int kChunks = C::kHalfDims / kChunk;
        constexpr int kLoads = C::kTile * kChunks / kWarpGroup;
        constexpr int kPitch = C::kDsaNopeHalf;
        constexpr int kGroupChunks = C::kDsaScaleGroup / kChunk;
        for (int i = 0; i < n; ++i) {
            const int s = i % C::kStages, is = i % C::kIdxStages;
            DecodeBar::wait(&sm.idx_full[is], (i / C::kIdxStages) & 1);
            uint32_t any[2];
            any[0] = sm.any[is][0];
            if constexpr (C::kIndexWarps > 1)
                any[1] = sm.any[is][1];
            else
                any[1] = 0u;
            DecodeBar::wait(&sm.raw_full[s], (i / C::kStages) & 1);
            uint8_t* const stage = sm.kv[s];
            uint4 raw[kLoads];
            float sc[kLoads];
            CUTLASS_PRAGMA_UNROLL
            for (int k = 0; k < kLoads; ++k) {
                const int e = t + k * kWarpGroup;
                const int row = e / kChunks, c = e % kChunks;
                raw[k] = *reinterpret_cast<const uint4*>(stage + row * kPitch +
                                                         c * kChunk);
                const float s0 = sm.dsa_sc[s][row][c / kGroupChunks];
                sc[k] = (any[row / kWarp] >> (row % kWarp)) & 1 ? s0 : 0.0f;
            }
            __syncwarp();
            if (t % kWarp == 0) DecodeBar::arrive(&sm.idx_empty[is]);
            DecodeNamedBar::sync(kWarpGroup, C::kDequantBar);
            CUTLASS_PRAGMA_UNROLL
            for (int k = 0; k < kLoads; ++k) {
                const int e = t + k * kWarpGroup;
                const int row = e / kChunks, c = e % kChunks;
                const bool odd = (c / C::kFp8BlockChunks) & 1;
                const int q0 = c % C::kFp8BlockChunks * 2;
                uint8_t* const block =
                    stage + c / C::kFp8BlockChunks * C::kBlockBytes;
                const uint4 first =
                    decode_dequant8_f32(odd ? raw[k].z : raw[k].x,
                                        odd ? raw[k].w : raw[k].y, sc[k]);
                const uint4 second =
                    decode_dequant8_f32(odd ? raw[k].x : raw[k].z,
                                        odd ? raw[k].y : raw[k].w, sc[k]);
                *reinterpret_cast<uint4*>(
                    block + pair_sw128_offset(row, q0 + odd)) = first;
                *reinterpret_cast<uint4*>(
                    block + pair_sw128_offset(row, q0 + !odd)) = second;
            }
            cutlass::arch::fence_view_async_shared();
            __syncwarp();
            if (t % kWarp == 0) DecodeBar::arrive(&sm.kv_full[s], 0, 1);
        }
    } else {
        using Plan = PairDequantPlan<kCta, C>;
        constexpr int kPitch = decode_raw_pitch<C>(kCta);
        for (int i = 0; i < n; ++i) {
            const int s = i % C::kStages, is = i % C::kIdxStages;
            DecodeBar::wait(&sm.idx_full[is], (i / C::kIdxStages) & 1);
            const uint32_t any[2] = {sm.any[is][0], sm.any[is][1]};
            DecodeBar::wait(&sm.raw_full[s], (i / C::kStages) & 1);
            uint8_t* const stage = sm.kv[s];
            uint4 raw[Plan::kFp8Loads];
            uint8_t e8[Plan::kFp8Loads];
            CUTLASS_PRAGMA_UNROLL
            for (int k = 0; k < Plan::kFp8Loads; ++k) {
                const int e = t + k * kWarpGroup;
                const int row = e / Plan::kFp8Chunks, c = e % Plan::kFp8Chunks;
                raw[k] = *reinterpret_cast<const uint4*>(stage + row * kPitch +
                                                         c * kChunk);
                const uint8_t sf = sm.sf[s][row][kCta * C::kLocalBlocks +
                                                 c / C::kFp8BlockChunks];
                e8[k] = (any[row / kWarp] >> (row % kWarp)) & 1 ? sf : 0;
            }
            constexpr int kRopeChunks = cute::max(Plan::kRopeChunks, 1);
            uint4 rope[cute::max(Plan::kRopeLoads, 1)];
            CUTLASS_PRAGMA_UNROLL
            for (int k = 0; k < Plan::kRopeLoads; ++k) {
                const int e = t + k * kWarpGroup;
                const int row = e / kRopeChunks, c = e % kRopeChunks;
                rope[k] = *reinterpret_cast<const uint4*>(
                    stage + row * kPitch + C::kNopeBytes1 + c * kChunk);
            }
            __syncwarp();
            if (t % kWarp == 0) DecodeBar::arrive(&sm.idx_empty[is]);
            DecodeNamedBar::sync(kWarpGroup, C::kDequantBar);
            CUTLASS_PRAGMA_UNROLL
            for (int k = 0; k < Plan::kFp8Loads; ++k) {
                const int e = t + k * kWarpGroup;
                const int row = e / Plan::kFp8Chunks, c = e % Plan::kFp8Chunks;
                const bool odd = (c / C::kFp8BlockChunks) & 1;
                const int q0 = c % C::kFp8BlockChunks * 2;
                uint8_t* const block =
                    stage + c / C::kFp8BlockChunks * C::kBlockBytes;
                const uint4 first =
                    pair_dequant8(odd ? raw[k].z : raw[k].x,
                                  odd ? raw[k].w : raw[k].y, e8[k]);
                const uint4 second =
                    pair_dequant8(odd ? raw[k].x : raw[k].z,
                                  odd ? raw[k].y : raw[k].w, e8[k]);
                *reinterpret_cast<uint4*>(
                    block + pair_sw128_offset(row, q0 + odd)) = first;
                *reinterpret_cast<uint4*>(
                    block + pair_sw128_offset(row, q0 + !odd)) = second;
            }
            CUTLASS_PRAGMA_UNROLL
            for (int k = 0; k < Plan::kRopeLoads; ++k) {
                const int e = t + k * kWarpGroup;
                const int row = e / kRopeChunks, c = e % kRopeChunks;
                *reinterpret_cast<uint4*>(
                    stage + (C::kLocalBlocks - 1) * C::kBlockBytes +
                    pair_sw128_offset(row, c)) = rope[k];
            }
            cutlass::arch::fence_view_async_shared();
            __syncwarp();
            if (t % kWarp == 0) DecodeBar::arrive(&sm.kv_full[s], 0, 1);
        }
    }
}

// The QK^T and PV warps of CTA 0 issue independently, so a late KV stage
// never holds back the PV of a tile whose P is ready, nor a late P the next
// QK^T.  QK^T(i) goes to S buffer i % 2 once its stage is dequantized and the
// softmax has read S(i - 2); a unit's first QK^T also waits for its Q.
template <class C>
CUTE_DEVICE void decode_qk_warp(const DecodeParams& p, typename C::Smem& sm,
                                uint32_t tmem, int cluster) {
    const uint32_t idesc =
        decode_instr_desc<C, 2 * C::kTile, cute::UMMA::Major::K>();
    DecodeUnitCursor<C> cur = decode_start_cursor<C>(p, sm, cluster);
    DecodeUnit u;
    for (int i = 0, unit = 0; cur.next(p, sm.unit_lens, u); ++unit) {
        DecodeBar::wait(&sm.q_full, unit & 1);
        for (int k = 0; k < u.n; ++k, ++i) {
            const int s = i % C::kStages, sb = i % 2;
            DecodeBar::wait(&sm.kv_full[s], (i / C::kStages) & 1);
            DecodeBar::wait(&sm.s_empty[sb], ((i / 2) & 1) ^ 1);
            tcgen05_after_thread_sync();
            if (cute::elect_one_sync()) {
                CUTLASS_PRAGMA_UNROLL
                for (int ks = 0; ks < C::kQkSteps; ++ks) {
                    decode_umma_ts(
                        tmem + C::kTmemS + sb * C::kTmemSCols,
                        tmem + C::kTmemQ + ks * kUmmaK / 2,
                        decode_sw128_desc(
                            sm.kv[s] +
                                ks / C::kKStepsPerBlock * C::kBlockBytes +
                                ks % C::kKStepsPerBlock * kUmmaKBytes,
                            kChunk),
                        idesc, ks > 0);
                }
                if constexpr (C::kDsa) {
                    CUTLASS_PRAGMA_UNROLL
                    for (int ks = 0; ks < C::kDsaRopeSteps; ++ks) {
                        decode_umma_ss(
                            tmem + C::kTmemS + sb * C::kTmemSCols,
                            decode_sw128_desc(sm.q_rope + ks * kUmmaKBytes,
                                              kChunk),
                            decode_sw128_desc(sm.rope_kv[s] + ks * kUmmaKBytes,
                                              kChunk),
                            idesc, true);
                    }
                }
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

// PV(i) once P(i) is written; it releases the KV stage and P buffer.  The
// first PV of a unit overwrites O, each half (128 columns) once the previous
// unit's epilogue has read it.
template <class C>
CUTE_DEVICE void decode_pv_warp(const DecodeParams& p, typename C::Smem& sm,
                                uint32_t tmem, int cluster) {
    const uint32_t idesc =
        decode_instr_desc<C, C::kPvN, cute::UMMA::Major::MN>();
    DecodeUnitCursor<C> cur = decode_start_cursor<C>(p, sm, cluster);
    DecodeUnit u;
    for (int i = 0, unit = 0; cur.next(p, sm.unit_lens, u); ++unit) {
        for (int k = 0; k < u.n; ++k, ++i) {
            const int s = i % C::kStages, pb = i % 2;
            DecodeBar::wait(&sm.p_full[pb], (i / 2) & 1);
            CUTLASS_PRAGMA_UNROLL
            for (int m = 0; m < C::kPvMmas; ++m) {
                if (k == 0 && unit > 0)
                    DecodeBar::wait(&sm.o_empty[m], (unit - 1) & 1);
                tcgen05_after_thread_sync();
                if (cute::elect_one_sync()) {
                    CUTLASS_PRAGMA_UNROLL
                    for (int ks = 0; ks < C::kPvSteps; ++ks) {
                        decode_umma_ss(
                            tmem + C::kTmemO + m * C::kPvN / 2,
                            decode_sw128_desc(sm.p[pb] + ks * kUmmaKBytes,
                                              kChunk),
                            decode_sw128_desc(
                                sm.kv[s] +
                                    m * C::kPvBlocksPerMma * C::kBlockBytes +
                                    ks * kUmmaK * kSw128AtomRowBytes,
                                C::kBlockBytes),
                            idesc, k > 0 || ks > 0);
                    }
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

// The online softmax of tile i (k-th of its unit): this lane's partial S of
// the partner's keys goes to the partner first, then its own keys are read;
// the masked sums give the row max (with the partner's), P and the row sum.
// O is rescaled when the row max grows by more than 2^6.
template <class C>
CUTE_DEVICE void decode_softmax_tile(const DecodeParams& p,
                                     typename C::Smem& sm, int i, int k,
                                     uint32_t lane_base, int t, float& m,
                                     float& l) {
    const int head = t % C::kHeads, half = t / C::kHeads;
    const int is = i % C::kIdxStages, sb = i % 2;
    constexpr int kHalf = C::kHalf;
    constexpr int kTmemSCols = C::kTmemSCols;
    constexpr int kQuads = kHalf / kQuad;
    DecodeBar::wait(&sm.idx_full[is], (i / C::kIdxStages) & 1);
    const int vbit = (half * kHalf) & (kWarp - 1);
    const uint32_t valid = sm.valid[is][half * kHalf / kWarp];
    DecodeBar::wait(&sm.s_full[sb], (i / 2) & 1);
    tcgen05_after_thread_sync();
    float4* const peer = reinterpret_cast<float4*>(sm.xchg[t ^ C::kHeads]);
    float own[kHalf];
    decode_tmem_ld(lane_base + C::kTmemS + sb * kTmemSCols + (half ^ 1) * kHalf,
                   own);
    CUTLASS_PRAGMA_UNROLL
    for (int q = 0; q < kQuads; ++q) {
        peer[q ^ (t % kQuads)] =
            make_float4(own[kQuad * q], own[kQuad * q + 1], own[kQuad * q + 2],
                        own[kQuad * q + kQuad - 1]);
    }
    decode_tmem_ld(lane_base + C::kTmemS + sb * kTmemSCols + half * kHalf, own);
    tcgen05_before_thread_sync();
    __syncwarp();
    if (t % kWarp == 0) {
        DecodeBar::arrive(&sm.s_empty[sb], 0, 1);
        DecodeBar::arrive(&sm.idx_empty[is]);
    }
    pair_lane_sync<C>(t);
    const float4* const got = reinterpret_cast<const float4*>(sm.xchg[t]);
    float local = -CUDART_INF_F;
    CUTLASS_PRAGMA_UNROLL
    for (int q = 0; q < kQuads; ++q) {
        const float4 v = got[q ^ ((t ^ C::kHeads) % kQuads)];
        const float add[kQuad] = {v.x, v.y, v.z, v.w};
        CUTLASS_PRAGMA_UNROLL
        for (int e = 0; e < kQuad; ++e) {
            const int c = kQuad * q + e;
            own[c] =
                (valid >> (vbit + c)) & 1 ? own[c] + add[e] : -CUDART_INF_F;
            local = fmaxf(local, own[c]);
        }
    }
    local *= p.scale_log2;
    sm.red[t] = local;
    pair_lane_sync<C>(t);
    const float cur = fmaxf(local, sm.red[t ^ C::kHeads]);
    float factor = 1.0f;
    if (cur - m > C::kRescaleLog2) {
        const float m_new = fmaxf(m, cur);
        factor = exp2f(m - m_new);
        m = m_new;
    }
    float sum = 0.0f;
    uint32_t pk[kHalf / 2];
    CUTLASS_PRAGMA_UNROLL
    for (int c = 0; c < kHalf; c += 2) {
        const float p0 = exp2f(fmaf(own[c], p.scale_log2, -m));
        const float p1 = exp2f(fmaf(own[c + 1], p.scale_log2, -m));
        sum += p0 + p1;
        const nv_bfloat162 b2 = __floats2bfloat162_rn(p0, p1);
        pk[c / 2] = reinterpret_cast<const uint32_t&>(b2);
    }
    l = l * factor + sum;
    if (k > 0 && __any_sync(~0u, factor != 1.0f)) {
        pair_wait_pv<C>(sm, i - 1);
        tcgen05_after_thread_sync();
        CUTLASS_PRAGMA_UNROLL
        for (int c0 = 0; c0 < C::kHalfDims; c0 += kTmemLd) {
            float o[kTmemLd];
            decode_tmem_ld(lane_base + C::kTmemO + c0, o);
            CUTLASS_PRAGMA_UNROLL
            for (int c = 0; c < kTmemLd; ++c) o[c] *= factor;
            tmem_st_32dp32bNx<kTmemLd>(lane_base + C::kTmemO + c0, o);
        }
        cutlass::arch::fence_view_async_tmem_store();
    }
    if (i > 1) pair_wait_pv<C>(sm, i - 2);
    CUTLASS_PRAGMA_UNROLL
    for (int q = 0; q < kHalf / kElemsPerChunk; ++q) {
        *reinterpret_cast<uint4*>(
            sm.p[sb] +
            pair_sw128_offset(head, half * kHalf / kElemsPerChunk + q)) =
            reinterpret_cast<const uint4*>(pk)[q];
    }
    cutlass::arch::fence_view_async_shared();
    tcgen05_before_thread_sync();
    __syncwarp();
    if (t % kWarp == 0) DecodeBar::arrive(&sm.p_full[sb], 0, 1);
}

// Per unit: the online softmax over its tiles, then the row max and partial
// sum go to the Q/O warp group.
template <class C>
CUTE_DEVICE void decode_softmax_wg(const DecodeParams& p, typename C::Smem& sm,
                                   uint32_t tmem, int t, int cluster) {
    const uint32_t lane_base =
        tmem + ((t / kWarp * kWarp) << DecodeIsa::kTmemLaneShift);
    DecodeUnitCursor<C> cur = decode_start_cursor<C>(p, sm, cluster);
    DecodeUnit u;
    for (int i = 0, unit = 0; cur.next(p, sm.unit_lens, u); ++unit) {
        float m = C::kMaxInit, l = 0.0f;
        for (int k = 0; k < u.n; ++k, ++i) {
            decode_softmax_tile<C>(p, sm, i, k, lane_base, t, m, l);
        }
        const int b = unit % 2;
        DecodeBar::wait(&sm.ml_empty[b], ((unit / 2) & 1) ^ 1);
        sm.ml[b][t] = make_float2(m, l);
        __syncwarp();
        if (t % kWarp == 0) DecodeBar::arrive(&sm.ml_full[b]);
    }
}

}  // namespace halops::mla
