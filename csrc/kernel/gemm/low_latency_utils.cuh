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
#include <cuda_runtime.h>
#include <cutlass/arch/barrier.h>

#include <cstdint>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm90_tma.hpp>

namespace halops {

struct LowLatencyMmaGemmParams {
    const uint8_t* __restrict__ ptr_a = nullptr;
    const uint8_t* __restrict__ ptr_a2 = nullptr;
    const uint8_t* __restrict__ ptr_b = nullptr;
    const int32_t* __restrict__ ptr_sfa = nullptr;
    const int32_t* __restrict__ ptr_sfa2 = nullptr;
    const int32_t* __restrict__ ptr_sfb_packed = nullptr;
    __nv_bfloat16* __restrict__ ptr_d = nullptr;
    __nv_bfloat16* __restrict__ ptr_d2 = nullptr;
    int n_split = 0;
    int m = 0;
    int n = 0;
    int k = 0;
    int k1 = 0;
    int sfa_stride_mn = 0;
    int sfa_stride_k = 0;
    int sfa2_stride_mn = 0;
    int sfb_stride_mn = 0;
    int sfb_stride_k = 0;
    int tile_swz = 0;
    int walk_q = -1;
    int walk_r = 0;
    int pdl = 0;
    int pdl_force_off = 0;
    CUtensorMap b_desc = {};
    CUtensorMap a_desc = {};
    CUtensorMap a_desc2 = {};
    CUtensorMap sfb_desc = {};
};

__device__ __forceinline__ void mma_m16n8k32_e4m3(float (&c)[4],
                                                  const uint32_t (&a)[4],
                                                  const uint32_t (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Byte idx of a packed scale word is the UE8M0 exponent of 32-wide k-group
// idx inside one 128-wide k-block.
__device__ __forceinline__ float decode_packed_ue8m0_word(int32_t packed,
                                                          int idx) {
    const uint32_t exponent =
        (static_cast<uint32_t>(packed) >> ((idx & 3) * 8)) & 0xffu;
    return __int_as_float(exponent << 23);
}

__device__ __forceinline__ uint32_t lds_u32(const uint8_t* smem_ptr) {
    return *reinterpret_cast<const uint32_t*>(smem_ptr);
}

using Barrier = cutlass::arch::ClusterTransactionBarrier;

constexpr uint64_t kTmaCacheHint =
    static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL);

__device__ __forceinline__ int swz128(int row, int chunk) {
    return row * 128 + ((chunk ^ row) & 7) * 16;
}

struct LlTileWalk {
    int cur = 0;
    int end = 0;
    int base = 0;
    int step = 1;
    __device__ __forceinline__ bool is_valid() const { return cur < end; }
    __device__ __forceinline__ int tile_id() const { return base + cur * step; }
    __device__ __forceinline__ void advance() { ++cur; }
};

__device__ __forceinline__ LlTileWalk ll_tile_walk(int n_tiles, int swz,
                                                   int walk_q = -1,
                                                   int walk_r = 0) {
    const int bx = static_cast<int>(blockIdx.x);
    const int gx = static_cast<int>(gridDim.x);
    LlTileWalk w;
    w.cur = 0;
    if (swz == 0) {
        w.base = bx;
        w.step = gx;
        if (walk_q >= 0) {
            w.end = walk_q + (bx < walk_r ? 1 : 0);
        } else {
            const int left = n_tiles - bx;
            w.end = left > 0 ? (left + gx - 1) / gx : 0;
        }
    } else {
        const int per = (n_tiles + gx - 1) / gx;
        const int left = n_tiles - bx * per;
        w.base = bx * per;
        w.step = 1;
        w.end = left <= 0 ? 0 : (left < per ? left : per);
    }
    return w;
}

constexpr int kLlBPhaseGroups = 2;
constexpr int kLlSfaRows = 16;
constexpr int kLlSfbStages = 4;
constexpr int kLlGroupK = 128;
constexpr int kLlScaleGroupK = 32;
constexpr int kLlMaxStages = 32;
constexpr int kLlSmemBudgetBytes = 227 * 1024;
constexpr int kLlAlignBytes = 16;
constexpr int kLlSfbAlignBytes = 128;
constexpr int kLlASlabAlignBytes = 1024;

__host__ __device__ __forceinline__ int ll_scale_pitch(int words) {
    const int p = (words + 3) & ~3;
    return (p & 4) ? p : p + 4;
}

struct LlScaleLayout {
    int sfa_pitch;
    int sfa_ints;
    int sfb_pitch;
    int sfb_slot_ints;
    int sfb_tx_bytes;
};

__host__ __device__ __forceinline__ LlScaleLayout ll_scale_layout(int k,
                                                                  int block_n) {
    const int pitch = ll_scale_pitch(k / 128);
    LlScaleLayout l;
    l.sfa_pitch = pitch;
    l.sfa_ints = kLlSfaRows * pitch;
    l.sfb_pitch = pitch;
    l.sfb_slot_ints = (block_n * pitch + 31) & ~31;
    l.sfb_tx_bytes = 4 * block_n * pitch;
    return l;
}

__host__ __device__ __forceinline__ int ll_sfa_g32_row_bytes(int k) {
    return (k / kLlScaleGroupK + kLlAlignBytes - 1) & ~(kLlAlignBytes - 1);
}

template <int kWarpsK>
__device__ __forceinline__ int ll_kb(int warp_k, int j, int n_iter) {
    return warp_k * n_iter + j;
}

template <int kWarpsN>
constexpr int kLlBRows() {
    return 8 * kWarpsN;
}
template <int kWarpsN, int kWarpsK>
constexpr int kLlBRings() {
    return kWarpsK;
}

template <int kWarpsN>
constexpr int kLlBTileBytes() {
    return (kLlBRows<kWarpsN>() / 8) * 8 * 128;
}
template <int kWarpsN>
constexpr int kLlBPhaseBytes() {
    return kLlBPhaseGroups * kLlBTileBytes<kWarpsN>();
}

template <int kWarpsN, int kWarpsK>
constexpr int kLlConsumerWarps() {
    return kWarpsN * kWarpsK;
}
template <int kWarpsN, int kWarpsK>
constexpr int kLlProducerWarp() {
    return kLlConsumerWarps<kWarpsN, kWarpsK>();
}
template <int kWarpsN, int kWarpsK>
constexpr int kLlEpilogueWarp0() {
    return kLlConsumerWarps<kWarpsN, kWarpsK>() + 1;
}

struct LlSmemPlan {
    int b_off;
    int sfa_off;
    int sfb_off;
    int bars_off;
    int acc_off;
    int total_bytes;
    int fixed_bytes;
    int per_stage_bytes;
};

template <int kWarpsN, int kWarpsK, int kRowsA, int kAccStages>
__host__ __device__ __forceinline__ LlSmemPlan ll_smem_plan(int k, int stages) {
    constexpr int kOuts = kRowsA == 16 ? 4 : 2;
    constexpr int kBRingsCount = kLlBRings<kWarpsN, kWarpsK>();
    constexpr int kAccCols = kAccStages * kWarpsN;
    constexpr int kStageBytes = kBRingsCount * kLlBPhaseBytes<kWarpsN>();
    constexpr int kBarPerStage = 8 * (kBRingsCount + 1);

    const LlScaleLayout sl = ll_scale_layout(k, kWarpsN * 8);
    const int a_bytes =
        (kRowsA * k + kLlASlabAlignBytes - 1) & ~(kLlASlabAlignBytes - 1);
    const int sfa_bytes = 4 * sl.sfa_ints;
    const int sfb_bytes = 4 * kLlSfbStages * sl.sfb_slot_ints;
    const int bar_fixed =
        8 * (k / kLlGroupK + 1 + 2 * kLlSfbStages + 2 * kAccCols);
    const int acc_bytes = kAccCols * kWarpsK * 32 * kOuts * sizeof(float);

    LlSmemPlan p;
    p.fixed_bytes = a_bytes + sfa_bytes + (kLlSfbAlignBytes - 1) + sfb_bytes +
                    bar_fixed + acc_bytes + (kLlAlignBytes - 1);
    p.per_stage_bytes = kStageBytes + kBarPerStage;
    p.b_off = a_bytes;
    p.sfa_off = p.b_off + kStageBytes * stages;
    p.sfb_off = (p.sfa_off + sfa_bytes + kLlSfbAlignBytes - 1) &
                ~(kLlSfbAlignBytes - 1);
    p.bars_off = p.sfb_off + sfb_bytes;
    p.acc_off =
        (p.bars_off + bar_fixed + kBarPerStage * stages + kLlAlignBytes - 1) &
        ~(kLlAlignBytes - 1);
    p.total_bytes = p.acc_off + acc_bytes;
    return p;
}

}  // namespace halops
