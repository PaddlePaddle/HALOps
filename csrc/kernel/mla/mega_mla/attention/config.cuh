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

#include <cute/arch/tmem_allocator_sm100.hpp>

#include "kernel/mla/mega_mla/mla_params.cuh"

namespace halops::mla {
/// Tuning and bit-layout constants of the mega MLA attention kernel.
struct MlaAttnConfig {
    static constexpr unsigned kFullMask = 0xffffffffu;
    static constexpr int kXQuantBatch = 4;
    static constexpr int kKvPerThread = 4;
    static constexpr float kKvAmaxFloor = 1e-4f;
    static constexpr int kF32MantBits = 23;
    static constexpr int kF32ExpMask = 0xFF;
    static constexpr int kF32MantMask = 0x7FFFFF;
    static constexpr float kMinWeightSum = 1e-38f;
    static constexpr int kF32PerVec4 = sizeof(float4) / sizeof(float);
    static constexpr int kFp8x2PerWord =
        sizeof(uint64_t) / sizeof(__nv_fp8x2_e4m3);
    static constexpr int kRolledKvGateMinTile = 64;
    static constexpr int kRescaleCols = 64;
    static constexpr int kHalfKeys = AttnConfig::KvBlock / 2;
    static constexpr int kUtccpBits = 256;
    static constexpr int kTmemColBits = 32;
    static constexpr int kSmemDescUnit = 16;
    static constexpr int kRopeCopyBatch = 4;
};

template <class KT>
struct SharedMemoryPlan {
    using Attn = typename KT::Attn;
    using Dims = typename KT::Dims;
    using Arena = GemmArena;

    union {
        struct {
            cute::array_aligned<
                cutlass::bfloat16_t,
                cute::cosize_v<typename Attn::SmemLayoutQ_SW128>>
                head;
            cute::array_aligned<cutlass::bfloat16_t,
                                cute::cosize_v<typename Attn::SmemLayoutOBuf>>
                o_buf;
        } qo;
        struct {
            struct {
                cute::array_aligned<cutlass::bfloat16_t,
                                    Dims::Heads * Dims::DimNope>
                    nope;
                cute::array_aligned<cutlass::bfloat16_t,
                                    Dims::Heads * Dims::DimRope>
                    rope;
            } dequant[Attn::NumBufs];
            cute::array_aligned<cutlass::float_e4m3_t,
                                Dims::Heads * Dims::DimNope>
                raw_nope[Attn::NumBufs];

            float4 p_exchange_buf[2][KT::WarpSize * (Attn::KvBlock / 2) /
                                     MlaAttnConfig::kF32PerVec4];
            cute::array_aligned<cutlass::bfloat16_t,
                                cute::cosize_v<typename Attn::SmemLayoutS>>
                s[Attn::NumBufs];

            cute::array_aligned<cutlass::bfloat16_t,
                                Dims::Heads * Dims::DimRope>
                rope_stage;
        } kv;

        char gemm[Arena::Total];
    } bufs;

    CUTE_ALIGNAS(16)
    cutlass::bfloat16_t rope_stage2[Dims::Heads * Dims::DimRope];

    CUTE_ALIGNAS(16) float wg_exchange[KT::ThreadsPerWarpGroup];

    CUTE_ALIGNAS(16) float c128_warp_sq[KT::Roles::VecWarps];
    int c128_last_slice;
    CUTE_ALIGNAS(16) float kv_write_sq[KT::Roles::VecWarps];

    union {
        CUTE_ALIGNAS(16) __nv_bfloat16 c128_row[Dims::DimV];
        CUTE_ALIGNAS(16) __nv_bfloat16 kv_write_row[Dims::DimV];
    };

    cute::array_aligned<uint32_t, 1> tmem_start_addr;
    int q_head_ctr;
    cutlass::arch::ClusterTransactionBarrier bar_q_tma, bar_q_utccp;
    cutlass::arch::ClusterTransactionBarrier bar_rope_ready[Attn::NumBufs];
    cutlass::arch::ClusterTransactionBarrier bar_rope_staged[Attn::NumBufs];
    cutlass::arch::ClusterTransactionBarrier bar_nope_ready[Attn::NumBufs];
    cutlass::arch::ClusterTransactionBarrier bar_raw_ready[Attn::NumBufs],
        bar_raw_free[Attn::NumBufs];
    cutlass::arch::ClusterTransactionBarrier bar_raw_ready2[Attn::NumBufs];
    cutlass::arch::ClusterTransactionBarrier bar_qk_done[Attn::NumBufs],
        bar_so_ready[Attn::NumBufs], bar_sv_done[Attn::NumBufs];

    cutlass::arch::ClusterTransactionBarrier
        bar_gemm_tma_a[Arena::MaxTmaStages];
    cutlass::arch::ClusterTransactionBarrier
        bar_gemm_tma_b[Arena::MaxTmaStages];
    cutlass::arch::ClusterTransactionBarrier
        bar_gemm_smem_free[Arena::MaxTmaStages];
    cutlass::arch::ClusterTransactionBarrier
        bar_gemm_acc_full[Arena::MaxAccStages];
    cutlass::arch::ClusterTransactionBarrier
        bar_gemm_acc_empty[Arena::MaxAccStages];
    cutlass::arch::ClusterTransactionBarrier
        bar_gemm_acc_win_lo[Arena::MaxAccStages];
    cutlass::arch::ClusterTransactionBarrier
        bar_gemm_acc_win_hi[Arena::MaxAccStages];

    CUTE_DEVICE GemmPipelineBars gemm_bars() {
        return {bar_gemm_tma_a,     bar_gemm_tma_b,     bar_gemm_smem_free,
                bar_gemm_acc_full,  bar_gemm_acc_empty, bar_gemm_acc_win_lo,
                bar_gemm_acc_win_hi};
    }

    CUTE_DEVICE char* gemm_smem() {
        return reinterpret_cast<char*>(bufs.kv.dequant[0].nope.data());
    }

    CUTE_DEVICE void init_gemm_bars() {
        CUTE_UNROLL
        for (int i = 0; i < Arena::MaxTmaStages; ++i) {
            bar_gemm_tma_a[i].init(1);
            bar_gemm_tma_b[i].init(1);
            bar_gemm_smem_free[i].init(1);
        }
        CUTE_UNROLL
        for (int i = 0; i < Arena::MaxAccStages; ++i) {
            bar_gemm_acc_full[i].init(1);
            bar_gemm_acc_empty[i].init(KT::Roles::DrainWarps * KT::WarpSize);
            bar_gemm_acc_win_lo[i].init(KT::Roles::DrainWarps * KT::WarpSize);
            bar_gemm_acc_win_hi[i].init(KT::Roles::DrainWarps * KT::WarpSize);
        }
    }

    CUTE_DEVICE void init_bars() {
        q_head_ctr = 0;
        bar_q_tma.init(1);
        bar_q_utccp.init(1);
        CUTE_UNROLL
        for (int i = 0; i < Attn::NumBufs; ++i) {
            bar_rope_ready[i].init(1);
            bar_rope_staged[i].init(1);
            bar_nope_ready[i].init(KT::ThreadsPerWarpGroup);
            bar_raw_ready[i].init(1);
            bar_raw_ready2[i].init(1);
            bar_raw_free[i].init(KT::ThreadsPerWarpGroup);
            bar_qk_done[i].init(1);
            bar_so_ready[i].init(KT::ThreadsPerWarpGroup);
            bar_sv_done[i].init(1);
        }
        init_gemm_bars();
        cutlass::arch::fence_barrier_init();
    }

    CUTE_DEVICE void acquire_tmem() {
        cute::TMEM::Allocator1Sm().allocate(Attn::TmemCols::Total,
                                            tmem_start_addr.data());
        cute::TMEM::Allocator1Sm().release_allocation_lock();
    }

    CUTE_DEVICE void release_tmem() {
        cute::TMEM::Allocator1Sm().free(0, Attn::TmemCols::Total);
    }
};
}  // namespace halops::mla
