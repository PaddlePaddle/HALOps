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
#include <cutlass/pipeline/sm90_pipeline.hpp>
#include <iterator>
#include <utility>

#include "common/pdl.cuh"
#include "kernel/gemm/gemm_utils.h"
#include "kernel/gemm/low_latency_utils.cuh"

namespace halops {

template <int BlockN, int WarpsK, int RowsA, int Stages>
struct LlTile : LlGemmConfig {
    static constexpr int KBlockN = BlockN;
    static constexpr int KWarpsN = BlockN / KMmaN;
    static constexpr int KWarpsK = WarpsK;
    static constexpr int KRowsA = RowsA;
    static constexpr int KStages = Stages;
    static constexpr int KAAtoms = RowsA / KAtomRows;
    static constexpr int KPair = ll_warp_pair(KWarpsN);
    static constexpr bool KSwapAB = RowsA == KAtomRows && KPair > 1;
    // Accumulator floats per lane.
    static constexpr int KOuts = 2 * KAAtoms;
    static constexpr int KATileBytes = KAAtoms * KAtomBytes;
    static constexpr int KBPhaseBytes = KBPhaseGroups * KWarpsN * KAtomBytes;
    static constexpr int KAccCols = KAccStages * KWarpsN;
    static constexpr int KAccSplits = WarpsK * KPair;
    static constexpr int KAccSlot = KAccSplits * 32 * KOuts;
    // Warps: consumers, one producer, then KWarpsN epilogue warps.
    static constexpr int KConsumerWarps = KWarpsN * WarpsK;
    static constexpr int KThreads = 32 * (KConsumerWarps + 1 + KWarpsN);
};

struct LlSmem {
    uint8_t* a_resident;
    uint8_t* b_rings;
    int32_t* sfa_resident;
    int32_t* sfb_ring;
    float* acc_slots;
    Barrier* a_full;
    Barrier* sfa_full;
    Barrier* sfb_full;
    Barrier* sfb_empty;
    Barrier* b_full;
    Barrier* b_empty;
    Barrier* acc_full;
    Barrier* acc_empty;
    int sfa_pitch;
    int sfb_pitch;
    int sfb2_pitch;
    int sfb_slot_ints;
};

template <class Tile>
__device__ __forceinline__ LlSmem ll_smem(uint8_t* smem_buf,
                                          const LlGemmParams& params) {
    const int k = params.k;
    const LlSmemPlan plan =
        ll_smem_plan(Tile::KWarpsN, Tile::KWarpsK, Tile::KRowsA, k, params.k1,
                     Tile::KStages);
    LlSmem smem;
    smem.a_resident = smem_buf;
    smem.b_rings = smem_buf + plan.b_off;
    smem.sfa_resident = reinterpret_cast<int32_t*>(smem_buf + plan.sfa_off);
    smem.sfb_ring = reinterpret_cast<int32_t*>(smem_buf + plan.sfb_off);
    smem.acc_slots = reinterpret_cast<float*>(smem_buf + plan.acc_off);
    smem.a_full = reinterpret_cast<Barrier*>(smem_buf + plan.bars_off);
    smem.sfa_full = smem.a_full + k / Tile::KGroupK;
    smem.sfb_full = smem.sfa_full + 1;
    smem.sfb_empty = smem.sfb_full + Tile::KSfbStages;
    smem.b_full = smem.sfb_empty + Tile::KSfbStages;
    smem.b_empty = smem.b_full + Tile::KWarpsK * Tile::KStages;
    smem.acc_full = smem.b_empty + Tile::KStages;
    smem.acc_empty = smem.acc_full + Tile::KAccCols;
    smem.sfa_pitch = plan.sfa_pitch;
    smem.sfb_pitch = plan.sfb_pitch;
    smem.sfb2_pitch = plan.sfb2_pitch;
    smem.sfb_slot_ints = plan.sfb_slot_ints;
    return smem;
}

template <class Tile>
__device__ __forceinline__ void ll_init_barriers(const LlSmem& smem, int k) {
    const int lane = threadIdx.x & 31;
    for (int i = lane; i < k / Tile::KGroupK; i += 32)
        (smem.a_full + i)->init(1);
    if (lane == 0) smem.sfa_full->init(1);
    for (int i = lane; i < Tile::KSfbStages; i += 32) {
        (smem.sfb_full + i)->init(1);
        (smem.sfb_empty + i)->init(Tile::KConsumerWarps);
    }
    for (int i = lane; i < Tile::KWarpsK * Tile::KStages; i += 32)
        (smem.b_full + i)->init(1);
    for (int i = lane; i < Tile::KStages; i += 32)
        (smem.b_empty + i)->init(Tile::KConsumerWarps);
    for (int i = lane; i < Tile::KAccCols; i += 32) {
        (smem.acc_full + i)->init(Tile::KAccSplits);
        (smem.acc_empty + i)->init(1);
    }
    cutlass::arch::fence_view_async_shared();
}

// sfa2 goes behind sfa in each shared-memory row.
__device__ __forceinline__ void ll_load_sfa(const LlGemmParams& params,
                                            const LlSmem& smem, int lane) {
    const uint32_t sfa_bytes = ll_scale_row_bytes(params.k1);
    const uint32_t sfa2_bytes =
        params.ptr_sfa2 ? ll_scale_row_bytes(params.k - params.k1) : 0;
    if (lane == 0)
        smem.sfa_full->arrive_and_expect_tx((sfa_bytes + sfa2_bytes) *
                                            params.m);
    __syncwarp();
    uint64_t* const bar = reinterpret_cast<uint64_t*>(smem.sfa_full);
    for (int row = lane; row < params.m; row += 32) {
        int32_t* const dst = smem.sfa_resident + row * smem.sfa_pitch;
        cute::SM90_BULK_COPY_G2S::copy(
            params.ptr_sfa + row * params.sfa_stride_mn, bar, dst, sfa_bytes);
        if (params.ptr_sfa2)
            cute::SM90_BULK_COPY_G2S::copy(
                params.ptr_sfa2 + row * params.sfa2_stride_mn, bar,
                dst + params.k1 / LlGemmConfig::KGroupK, sfa2_bytes);
    }
}

// k-blocks past K1 come from input2.
template <class Tile>
__device__ __forceinline__ void ll_load_a_phase(const LlGemmParams& params,
                                                const LlSmem& smem,
                                                int first_block) {
    const int k1_blocks = params.k1 / Tile::KGroupK;
    for (int block = first_block; block < first_block + Tile::KBPhaseGroups;
         ++block) {
        const bool from_a2 = block >= k1_blocks;
        const CUtensorMap* const desc =
            from_a2 ? &params.a_desc2 : &params.a_desc;
        const int k_offset =
            (from_a2 ? block - k1_blocks : block) * Tile::KGroupK;
        Barrier* const bar = smem.a_full + block;
        bar->arrive_and_expect_tx(Tile::KATileBytes);
#pragma unroll
        for (int atom = 0; atom < Tile::KAAtoms; ++atom)
            cute::SM90_TMA_LOAD_2D::copy(
                desc, reinterpret_cast<uint64_t*>(bar), Tile::KTmaCacheHint,
                smem.a_resident + block * Tile::KATileBytes +
                    atom * Tile::KAtomBytes,
                k_offset, atom * Tile::KAtomRows);
    }
}

template <class Tile, bool UsePdl, bool ConcatK>
__device__ __forceinline__ void ll_producer_warp(const LlGemmParams& params,
                                                 const LlSmem& smem) {
    const int lane = threadIdx.x & 31;
    const int n_tiles = params.n / Tile::KBlockN;
    const int split_blocks = params.k / Tile::KGroupK / Tile::KWarpsK;
    const int k1_blocks = params.k1 / Tile::KGroupK;
    // Lane r feeds the B ring of K split r.
    const bool feeds_ring = lane < Tile::KWarpsK;
    Barrier* const b_full = smem.b_full + lane * Tile::KStages;
    uint8_t* const b_ring =
        smem.b_rings + lane * Tile::KStages * Tile::KBPhaseBytes;
    // Producers start at phase 1: every slot is free on the first pass.
    cutlass::PipelineState<Tile::KStages> b_state{0, 1, 0};
    cutlass::PipelineState<Tile::KSfbStages> sfb_state{0, 1, 0};

    // Under PDL only A and sfa wait for the upstream kernel.
    if constexpr (!UsePdl) ll_load_sfa(params, smem, lane);
    bool first_tile = true;
    for (int tile = blockIdx.x; tile < n_tiles; tile += gridDim.x) {
        const int tile_n0 = tile * Tile::KBlockN;
        // Tiles from n_split on read weight2 / sfb2.
        const bool tile_in_b2 = tile_n0 >= params.n_split;
        const int b_row = tile_in_b2 ? tile_n0 - params.n_split : tile_n0;
        (smem.sfb_empty + sfb_state.index())->wait(sfb_state.phase());
        if (lane == 0) {
            Barrier* const bar = smem.sfb_full + sfb_state.index();
            int32_t* const dst =
                smem.sfb_ring + sfb_state.index() * smem.sfb_slot_ints;
            bar->arrive_and_expect_tx(4 * Tile::KBlockN *
                                      (smem.sfb_pitch + smem.sfb2_pitch));
            cute::SM90_TMA_LOAD_2D::copy(
                tile_in_b2 ? &params.sfb_desc2 : &params.sfb_desc,
                reinterpret_cast<uint64_t*>(bar), Tile::KTmaCacheHint, dst, 0,
                b_row);
            if constexpr (ConcatK)
                cute::SM90_TMA_LOAD_2D::copy(
                    &params.sfb_desc2, reinterpret_cast<uint64_t*>(bar),
                    Tile::KTmaCacheHint, dst + Tile::KBlockN * smem.sfb_pitch,
                    0, tile_n0);
        }
        ++sfb_state;

        for (int phase_block = 0; phase_block < split_blocks;
             phase_block += Tile::KBPhaseGroups) {
            (smem.b_empty + b_state.index())->wait(b_state.phase());
            if (feeds_ring) {
                const int block = lane * split_blocks + phase_block;
                Barrier* const bar = b_full + b_state.index();
                uint8_t* const dst =
                    b_ring + b_state.index() * Tile::KBPhaseBytes;
                bar->arrive_and_expect_tx(Tile::KBPhaseBytes);
                const CUtensorMap* desc =
                    tile_in_b2 ? &params.b_desc2 : &params.b_desc;
                int b_block = block;
                // k-blocks from K1 on read weight2.
                if constexpr (ConcatK) {
                    if (block >= k1_blocks) {
                        desc = &params.b_desc2;
                        b_block = block - k1_blocks;
                    }
                }
                cute::SM90_TMA_LOAD_3D::copy(
                    desc, reinterpret_cast<uint64_t*>(bar), Tile::KTmaCacheHint,
                    dst, 0, b_row, b_block);
            }
            ++b_state;
            if constexpr (UsePdl) {
                if (first_tile && phase_block == 0) {
                    pdl_wait();
                    ll_load_sfa(params, smem, lane);
                }
            }
            if (first_tile && feeds_ring)
                ll_load_a_phase<Tile>(params, smem,
                                      lane * split_blocks + phase_block);
        }
        first_tile = false;
    }
}

template <class Tile, bool ConcatK>
__device__ __forceinline__ void ll_consumer_warp(const LlGemmParams& params,
                                                 const LlSmem& smem) {
    const int lane = threadIdx.x & 31;
    // Lane position in the mma fragment.
    const int row = lane >> 2;
    const int col = lane & 3;
    const int warp = threadIdx.x >> 5;
    const int warp_n = warp % Tile::KWarpsN;
    const int warp_k = warp / Tile::KWarpsN;
    const int n_tiles = params.n / Tile::KBlockN;
    const int split_blocks = params.k / Tile::KGroupK / Tile::KWarpsK;
    const int split_block0 = warp_k * split_blocks;
    const int k1_blocks = params.k1 / Tile::KGroupK;
    Barrier* const b_full = smem.b_full + warp_k * Tile::KStages;
    const uint8_t* const b_ring =
        smem.b_rings + warp_k * Tile::KStages * Tile::KBPhaseBytes;
    constexpr bool KSwap = Tile::KSwapAB;
    constexpr int KPair = Tile::KPair;
    constexpr int KMmas = KSwap ? 1 : KPair;
    const int pair_rank = warp_n % KPair;
    const int pair_warp_n = warp_n - pair_rank;
    const int sfb_col = pair_warp_n * Tile::KMmaN + 2 * col;
    const int swap_n = pair_warp_n * Tile::KMmaN + row;

    cutlass::PipelineState<Tile::KStages> b_state;
    cutlass::PipelineState<Tile::KSfbStages> sfb_state;
    cutlass::PipelineState<Tile::KAccStages> acc_state{0, 1, 0};
    bool first_tile = true;
    for (int tile = blockIdx.x; tile < n_tiles; tile += gridDim.x) {
        float out[KPair * Tile::KOuts] = {};
        const int32_t* const sfb =
            smem.sfb_ring + sfb_state.index() * smem.sfb_slot_ints;
        (smem.sfb_full + sfb_state.index())->wait(sfb_state.phase());

        for (int phase_block = 0; phase_block < split_blocks;
             phase_block += Tile::KBPhaseGroups) {
            int sfb_offset = 0;
            int sfb_pitch = smem.sfb_pitch;
            // sfb2 rows hold the k-blocks from K1 on.
            if constexpr (ConcatK) {
                if (split_block0 + phase_block >= k1_blocks) {
                    sfb_offset = Tile::KBlockN * smem.sfb_pitch - k1_blocks;
                    sfb_pitch = smem.sfb2_pitch;
                }
            }
            // A is resident: wait for it on the first tile only.
            if (first_tile) {
                for (int sub_block = 0; sub_block < Tile::KBPhaseGroups;
                     ++sub_block)
                    (smem.a_full + split_block0 + phase_block + sub_block)
                        ->wait(0);
            }
            (b_full + b_state.index())->wait(b_state.phase());
            __syncwarp();

#pragma unroll
            for (int i = 0; i < Tile::KBPhaseGroups / KPair; ++i) {
                const int sub_block = i * KPair + pair_rank;
                const int block = split_block0 + phase_block + sub_block;
                const uint8_t* const tile_a =
                    smem.a_resident + block * Tile::KATileBytes;
                const uint8_t* const tile_b =
                    b_ring + b_state.index() * Tile::KBPhaseBytes +
                    (sub_block * Tile::KWarpsN + pair_warp_n) *
                        Tile::KAtomBytes;
                const uint8_t* const mma_a = KSwap ? tile_b : tile_a;
                const uint8_t* const mma_b = KSwap ? tile_a : tile_b;

                float acc[Tile::KScaleGroups][KMmas][4] = {};
#pragma unroll
                for (int group = 0; group < Tile::KScaleGroups; ++group) {
                    const uint2 a_top = ll_lds_atom(mma_a, row, col, group);
                    uint2 a_bottom = make_uint2(0u, 0u);
                    if constexpr (KSwap || Tile::KRowsA == 16)
                        a_bottom = ll_lds_atom(mma_a + Tile::KAtomBytes, row,
                                               col, group);
                    const uint32_t frag_a[4] = {a_top.x, a_bottom.x, a_top.y,
                                                a_bottom.y};
#pragma unroll
                    for (int j = 0; j < KMmas; ++j) {
                        const uint2 b_words = ll_lds_atom(
                            mma_b + j * Tile::KAtomBytes, row, col, group);
                        const uint32_t frag_b[2] = {b_words.x, b_words.y};
                        mma_m16n8k32_e4m3(acc[group][j], frag_a, frag_b);
                    }
                }

                if (first_tile && block == split_block0 + pair_rank)
                    smem.sfa_full->wait(0);
                const int32_t* const sfa_row = smem.sfa_resident + block;
                const int32_t* const sfb_row = sfb + sfb_offset + block;
                const uint32_t row_word_top =
                    KSwap ? sfb_row[swap_n * sfb_pitch]
                          : sfa_row[row * smem.sfa_pitch];
                const uint32_t row_word_bottom =
                    KSwap ? sfb_row[(swap_n + Tile::KMmaN) * sfb_pitch]
                          : sfa_row[(row + Tile::KAtomRows) * smem.sfa_pitch];
#pragma unroll
                for (int j = 0; j < KMmas; ++j) {
                    const int n = sfb_col + j * Tile::KMmaN;
                    const uint32_t col_word0 =
                        KSwap ? sfa_row[2 * col * smem.sfa_pitch]
                              : sfb_row[n * sfb_pitch];
                    const uint32_t col_word1 =
                        KSwap ? sfa_row[(2 * col + 1) * smem.sfa_pitch]
                              : sfb_row[(n + 1) * sfb_pitch];
                    float* const o = out + j * Tile::KOuts;
#pragma unroll
                    for (int group = 0; group < Tile::KScaleGroups; ++group) {
                        const float col_scale0 =
                            decode_packed_ue8m0_word(col_word0, group);
                        const float col_scale1 =
                            decode_packed_ue8m0_word(col_word1, group);
                        const float row_scale_top =
                            decode_packed_ue8m0_word(row_word_top, group);
                        o[0] += acc[group][j][0] * (row_scale_top * col_scale0);
                        o[1] += acc[group][j][1] * (row_scale_top * col_scale1);
                        if constexpr (KSwap || Tile::KRowsA == 16) {
                            const float row_scale_bottom =
                                decode_packed_ue8m0_word(row_word_bottom,
                                                         group);
                            o[2] += acc[group][j][2] *
                                    (row_scale_bottom * col_scale0);
                            o[3] += acc[group][j][3] *
                                    (row_scale_bottom * col_scale1);
                        }
                    }
                }
            }
            __syncwarp();
            if (lane == 0) (smem.b_empty + b_state.index())->arrive();
            ++b_state;
        }
        __syncwarp();
        if (lane == 0) (smem.sfb_empty + sfb_state.index())->arrive();
        ++sfb_state;

#pragma unroll
        for (int half = 0; half < KPair; ++half) {
            const int acc_col =
                acc_state.index() * Tile::KWarpsN + pair_warp_n + half;
            (smem.acc_empty + acc_col)->wait(acc_state.phase());
            float* const slot = smem.acc_slots + acc_col * Tile::KAccSlot;
            const int split = warp_k * KPair + pair_rank;
            const float* const o = out + half * Tile::KOuts;
            if constexpr (KSwap) {
#pragma unroll
                for (int t = 0; t < Tile::KOuts; ++t)
                    slot[split * 32 * Tile::KOuts +
                         (2 * col + t) * Tile::KMmaN + row] = o[t];
            } else if constexpr (Tile::KOuts == 4) {
                *reinterpret_cast<float4*>(slot + (split * 32 + lane) * 4) =
                    make_float4(o[0], o[1], o[2], o[3]);
            } else {
                *reinterpret_cast<float2*>(slot + (split * 32 + lane) * 2) =
                    make_float2(o[0], o[1]);
            }
            __syncwarp();
            if (lane == 0) (smem.acc_full + acc_col)->arrive();
        }
        ++acc_state;
        first_tile = false;
    }
}

__device__ __forceinline__ float ll_sigmoid(float acc) {
    const float x = __bfloat162float(__float2bfloat16_rn(acc));
    constexpr float kScale = 0x1p-64f;
    float s;
    asm("mul.rn.f32 %0, %1, %2;"
        : "=f"(s)
        : "f"(__fdividef(1.0f, (1.0f + __expf(-x)) * kScale)), "f"(kScale));
    return s;
}

template <class Tile, bool Sigmoid>
__device__ __forceinline__ void ll_epilogue_warp(const LlGemmParams& params,
                                                 const LlSmem& smem) {
    const int lane = threadIdx.x & 31;
    const int warp_n = (threadIdx.x >> 5) - Tile::KConsumerWarps - 1;
    const int n_tiles = params.n / Tile::KBlockN;
    const int row_top = lane >> 2;
    const int row_bottom = row_top + Tile::KAtomRows;
    cutlass::PipelineState<Tile::KAccStages> acc_state;
    for (int tile = blockIdx.x; tile < n_tiles; tile += gridDim.x) {
        const int acc_col = acc_state.index() * Tile::KWarpsN + warp_n;
        (smem.acc_full + acc_col)->wait(acc_state.phase());
        const float* const slot =
            smem.acc_slots + acc_col * Tile::KAccSlot + lane * Tile::KOuts;
        float out[Tile::KOuts] = {};
#pragma unroll
        for (int split = 0; split < Tile::KAccSplits; ++split) {
#pragma unroll
            for (int i = 0; i < Tile::KOuts; ++i)
                out[i] += slot[split * 32 * Tile::KOuts + i];
        }
        if constexpr (Sigmoid) {
#pragma unroll
            for (int i = 0; i < Tile::KOuts; ++i) out[i] = ll_sigmoid(out[i]);
        }

        int out_col =
            tile * Tile::KBlockN + warp_n * Tile::KMmaN + 2 * (lane & 3);
        c10::BFloat16* dst = params.ptr_d;
        int row_stride = params.n_split;
        if (out_col >= params.n_split) {
            dst = params.ptr_d2;
            out_col -= params.n_split;
            row_stride = params.n - params.n_split;
        }
        if (row_top < params.m)
            *reinterpret_cast<__nv_bfloat162*>(dst + row_top * row_stride +
                                               out_col) =
                __floats2bfloat162_rn(out[0], out[1]);
        if constexpr (Tile::KRowsA == 16) {
            if (row_bottom < params.m)
                *reinterpret_cast<__nv_bfloat162*>(
                    dst + row_bottom * row_stride + out_col) =
                    __floats2bfloat162_rn(out[2], out[3]);
        }
        __syncwarp();
        if (lane == 0) (smem.acc_empty + acc_col)->arrive();
        ++acc_state;
    }
}

template <class Tile, bool UsePdl, bool ConcatK, bool Sigmoid = false>
__global__ void __launch_bounds__(Tile::KThreads, 1)
    low_latency_mma_gemm_kernel(const __grid_constant__ LlGemmParams params) {
    extern __shared__ __align__(16) uint8_t smem_buf[];
    const LlSmem smem = ll_smem<Tile>(smem_buf, params);
    if (threadIdx.x < 32) ll_init_barriers<Tile>(smem, params.k);
    __syncthreads();
    if constexpr (UsePdl) pdl_trigger();

    const int warp = threadIdx.x >> 5;
    if (warp < Tile::KConsumerWarps) {
        ll_consumer_warp<Tile, ConcatK>(params, smem);
    } else if (warp == Tile::KConsumerWarps) {
        ll_producer_warp<Tile, UsePdl, ConcatK>(params, smem);
    } else {
        ll_epilogue_warp<Tile, Sigmoid>(params, smem);
    }
}

inline void EncodeLlTmaDescs(LlGemmParams& params, int block_n) {
    const uint64_t m = params.m;
    const uint64_t n = params.n;
    const uint64_t k = params.k;
    const uint64_t k1 = params.k1;
    const uint64_t n_split = params.n_split;
    const uint32_t elem_strides[3] = {1, 1, 1};

    const auto encode_a = [&](CUtensorMap* desc, const void* ptr,
                              uint64_t cols) {
        const uint64_t dims[2] = {cols, m};
        const uint32_t box[2] = {LlGemmConfig::KGroupK,
                                 LlGemmConfig::KAtomRows};
        TORCH_CHECK(
            cuTensorMapEncodeTiled(
                desc, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(ptr),
                dims, &cols, box, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
                CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS,
            "low_latency_mma_gemm: input TMA descriptor encoding failed");
    };
    // Weight [rows, cols] is viewed as [cols / 128, rows, 128].
    const auto encode_b = [&](CUtensorMap* desc, const void* ptr, uint64_t rows,
                              uint64_t cols) {
        const uint64_t dims[3] = {LlGemmConfig::KGroupK, rows,
                                  cols / LlGemmConfig::KGroupK};
        const uint64_t strides[2] = {cols, LlGemmConfig::KGroupK};
        const uint32_t box[3] = {LlGemmConfig::KGroupK,
                                 static_cast<uint32_t>(block_n),
                                 LlGemmConfig::KBPhaseGroups};
        TORCH_CHECK(
            cuTensorMapEncodeTiled(
                desc, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, const_cast<void*>(ptr),
                dims, strides, box, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
                CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS,
            "low_latency_mma_gemm: weight TMA descriptor encoding failed");
    };
    // sfb is read as one int32 word per k-block.
    const auto encode_sfb = [&](CUtensorMap* desc, const void* ptr,
                                uint64_t rows, uint64_t stride, uint64_t cols) {
        const uint64_t dims[2] = {cols / LlGemmConfig::KGroupK, rows};
        const uint32_t box[2] = {static_cast<uint32_t>(ll_scale_pitch(cols)),
                                 static_cast<uint32_t>(block_n)};
        TORCH_CHECK(
            cuTensorMapEncodeTiled(
                desc, CU_TENSOR_MAP_DATA_TYPE_INT32, 2, const_cast<void*>(ptr),
                dims, &stride, box, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
                CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS,
            "low_latency_mma_gemm: sfb TMA descriptor encoding failed");
    };

    encode_a(&params.a_desc, params.ptr_a, k1);
    encode_b(&params.b_desc, params.ptr_b, n_split, k1);
    encode_sfb(&params.sfb_desc, params.ptr_sfb, n_split, params.sfb_stride_mn,
               k1);
    if (params.ptr_a2) encode_a(&params.a_desc2, params.ptr_a2, k - k1);
    if (params.ptr_b2) {
        // weight2 is [N, K - K1] with input2, [N - n_split, K] with output2.
        const uint64_t rows2 = params.ptr_a2 ? n : n - n_split;
        const uint64_t cols2 = params.ptr_a2 ? k - k1 : k;
        encode_b(&params.b_desc2, params.ptr_b2, rows2, cols2);
        encode_sfb(&params.sfb_desc2, params.ptr_sfb2, rows2,
                   params.sfb2_stride_mn, cols2);
    }
}

// Launches the kernel instance of kLlTiles x KRowsTiers x KStageTiers that
// matches tile and rows_a.
inline void LaunchLlKernel(LlGemmParams params, const LlTileSpec& tile,
                           int rows_a, int num_sms, bool use_pdl,
                           bool use_sigmoid, cudaStream_t stream) {
    EncodeLlTmaDescs(params, tile.block_n);
    // An idle CTA would exit with its sfa bulk copy in flight.
    const int n_tiles = params.n / tile.block_n;
    const int waves = (n_tiles + num_sms - 1) / num_sms;
    const int grid = (n_tiles + waves - 1) / waves;
    const int smem_bytes =
        ll_smem_plan(tile.block_n / LlGemmConfig::KMmaN, tile.warps_k, rows_a,
                     params.k, params.k1, tile.stages)
            .total_bytes;

    void (*kernel)(LlGemmParams) = nullptr;
    int threads = 0;
    constexpr int num_rows_tiers = std::size(LlGemmConfig::KRowsTiers);
    constexpr int num_stage_tiers = std::size(LlGemmConfig::KStageTiers);
    const auto pick_if_selected = [&](auto combo_index) {
        constexpr int combo = decltype(combo_index)::value;
        constexpr LlTileSpec spec =
            kLlTiles[combo / (num_rows_tiers * num_stage_tiers)];
        constexpr int rows =
            LlGemmConfig::KRowsTiers[combo / num_stage_tiers % num_rows_tiers];
        constexpr int stages =
            LlGemmConfig::KStageTiers[combo % num_stage_tiers];
        if (spec.block_n != tile.block_n || spec.warps_k != tile.warps_k ||
            rows != rows_a || stages != tile.stages)
            return false;
        using Tile = LlTile<spec.block_n, spec.warps_k, rows, stages>;
        kernel =
            use_sigmoid
                ? (use_pdl
                       ? &low_latency_mma_gemm_kernel<Tile, true, false, true>
                       : &low_latency_mma_gemm_kernel<Tile, false, false, true>)
            : params.ptr_a2
                ? (use_pdl ? &low_latency_mma_gemm_kernel<Tile, true, true>
                           : &low_latency_mma_gemm_kernel<Tile, false, true>)
                : (use_pdl ? &low_latency_mma_gemm_kernel<Tile, true, false>
                           : &low_latency_mma_gemm_kernel<Tile, false, false>);
        threads = Tile::KThreads;
        return true;
    };
    [&]<int... Combos>(std::integer_sequence<int, Combos...>) {
        (pick_if_selected(std::integral_constant<int, Combos>{}) || ...);
    }(std::make_integer_sequence<int, std::size(kLlTiles) * num_rows_tiers *
                                          num_stage_tiers>{});

    HALOPS_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
    HALOPS_CUDA_CHECK(LaunchKernelWithPdl(kernel, dim3(grid), dim3(threads),
                                          kNoCluster, smem_bytes, stream,
                                          use_pdl, params));
}

}  // namespace halops
