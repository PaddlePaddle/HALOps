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
#include <cutlass/arch/arch.h>
#include <cutlass/arch/barrier.h>
#include <torch/extension.h>

#include <cstdint>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <optional>

namespace halops {

struct LlGemmParams {
    const c10::Float8_e4m3fn* __restrict__ ptr_a = nullptr;
    const c10::Float8_e4m3fn* __restrict__ ptr_a2 = nullptr;
    const c10::Float8_e4m3fn* __restrict__ ptr_b = nullptr;
    const c10::Float8_e4m3fn* __restrict__ ptr_b2 = nullptr;
    const uint8_t* __restrict__ ptr_sfa = nullptr;
    const uint8_t* __restrict__ ptr_sfa2 = nullptr;
    const uint8_t* __restrict__ ptr_sfb = nullptr;
    const uint8_t* __restrict__ ptr_sfb2 = nullptr;
    c10::BFloat16* __restrict__ ptr_d = nullptr;
    c10::BFloat16* __restrict__ ptr_d2 = nullptr;
    int n_split = 0;
    int m = 0;
    int n = 0;
    int k = 0;
    int k1 = 0;
    int sfa_stride_mn = 0;
    int sfa2_stride_mn = 0;
    int sfb_stride_mn = 0;
    int sfb2_stride_mn = 0;
    CUtensorMap b_desc = {};
    CUtensorMap b_desc2 = {};
    CUtensorMap a_desc = {};
    CUtensorMap a_desc2 = {};
    CUtensorMap sfb_desc = {};
    CUtensorMap sfb_desc2 = {};
};

struct LlGemmConfig {
    static constexpr int KMaxM = 16;
    // K-blocks per B ring stage.
    static constexpr int KBPhaseGroups = 2;
    static constexpr int KSfbStages = 4;
    static constexpr int KGroupK = 128;
    static constexpr int KScaleGroupK = 32;
    static constexpr int KScaleGroups = KGroupK / KScaleGroupK;
    static constexpr int KAccStages = 3;
    // N of mma.sync.m16n8k32.
    static constexpr int KMmaN = 8;
    // One SWIZZLE_128B atom.
    static constexpr int KAtomRows = 8;
    static constexpr int KAtomBytes = KAtomRows * 128;
    static constexpr int KSmemBudgetBytes =
        cutlass::arch::sm100_smem_capacity_bytes;
    static constexpr int KAlignBytes = 16;
    static constexpr int KSfbAlignBytes = 128;
    static constexpr int KASlabAlignBytes = 1024;
    static constexpr uint64_t KTmaCacheHint =
        static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL);
    static constexpr int KRowsTiers[] = {KAtomRows, KMaxM};
    static constexpr int KStageTiers[] = {1, 2, 4, 8};
    // SelectLlTile cost model, fitted on GB200.
    static constexpr double KCostIssue = 2.0;
    static constexpr double KCostNarrowTile = 1.15;
    static constexpr double KCostSingleStage = 2.0;
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

// Byte `group` of a packed scale word is the UE8M0 exponent of that k-group.
__device__ __forceinline__ float decode_packed_ue8m0_word(uint32_t packed,
                                                          int group) {
    return __int_as_float(((packed >> (group * 8)) & 0xffu) << 23);
}

using Barrier = cutlass::arch::ClusterTransactionBarrier;

// One lane's mma fragment words of k-group `group` in a SWIZZLE_128B atom.
__device__ __forceinline__ uint2 ll_lds_atom(const uint8_t* atom, int row,
                                             int col, int group) {
    const uint8_t* const row_ptr = atom + row * 128 + col * 4;
    const int chunk0 = ((2 * group) ^ row) & 7;
    const int chunk1 = ((2 * group + 1) ^ row) & 7;
    return make_uint2(
        *reinterpret_cast<const uint32_t*>(row_ptr + chunk0 * 16),
        *reinterpret_cast<const uint32_t*>(row_ptr + chunk1 * 16));
}

// Scale row pitch in int32 words; an odd number of 16-byte chunks avoids
// bank conflicts.
__host__ __device__ __forceinline__ int ll_scale_pitch(int k) {
    const int words = (k / LlGemmConfig::KGroupK + 3) & ~3;
    return (words & 4) ? words : words + 4;
}

__host__ __device__ __forceinline__ int ll_scale_row_bytes(int k) {
    return (k / LlGemmConfig::KScaleGroupK + LlGemmConfig::KAlignBytes - 1) &
           ~(LlGemmConfig::KAlignBytes - 1);
}

struct LlSmemPlan {
    int sfa_pitch;
    int sfb_pitch;
    int sfb2_pitch;
    int sfb_slot_ints;
    int b_off;
    int sfa_off;
    int sfb_off;
    int bars_off;
    int acc_off;
    int total_bytes;
    int fixed_bytes;
    int per_stage_bytes;
};

// With concat-K (k1 < k), each sfb slot holds the sfb rows and then the sfb2
// rows.
__host__ __device__ __forceinline__ LlSmemPlan
ll_smem_plan(int warps_n, int warps_k, int rows_a, int k, int k1, int stages) {
    constexpr int kBarBytes = sizeof(Barrier);
    const int outs = 2 * (rows_a / LlGemmConfig::KAtomRows);
    const int acc_cols = LlGemmConfig::KAccStages * warps_n;
    const int stage_bytes = warps_k * LlGemmConfig::KBPhaseGroups * warps_n *
                            LlGemmConfig::KAtomBytes;
    const int bar_per_stage = kBarBytes * (warps_k + 1);
    const int a_bytes = (rows_a * k + LlGemmConfig::KASlabAlignBytes - 1) &
                        ~(LlGemmConfig::KASlabAlignBytes - 1);
    constexpr int kSlotAlign = LlGemmConfig::KSfbAlignBytes / sizeof(int32_t);
    const int sfa_pitch = ll_scale_pitch(k);
    const int sfb_pitch = ll_scale_pitch(k1);
    const int sfb2_pitch = k1 < k ? ll_scale_pitch(k - k1) : 0;
    const int sfb_slot_ints =
        (warps_n * LlGemmConfig::KMmaN * (sfb_pitch + sfb2_pitch) + kSlotAlign -
         1) &
        ~(kSlotAlign - 1);
    const int sfa_bytes = 4 * LlGemmConfig::KMaxM * sfa_pitch;
    const int sfb_bytes = 4 * LlGemmConfig::KSfbStages * sfb_slot_ints;
    const int bar_fixed =
        kBarBytes * (k / LlGemmConfig::KGroupK + 1 +
                     2 * LlGemmConfig::KSfbStages + 2 * acc_cols);
    const int acc_bytes = acc_cols * warps_k * 32 * outs * sizeof(float);

    LlSmemPlan plan;
    plan.sfa_pitch = sfa_pitch;
    plan.sfb_pitch = sfb_pitch;
    plan.sfb2_pitch = sfb2_pitch;
    plan.sfb_slot_ints = sfb_slot_ints;
    plan.fixed_bytes = a_bytes + sfa_bytes +
                       (LlGemmConfig::KSfbAlignBytes - 1) + sfb_bytes +
                       bar_fixed + acc_bytes + (LlGemmConfig::KAlignBytes - 1);
    plan.per_stage_bytes = stage_bytes + bar_per_stage;
    plan.b_off = a_bytes;
    plan.sfa_off = plan.b_off + stage_bytes * stages;
    plan.sfb_off =
        (plan.sfa_off + sfa_bytes + LlGemmConfig::KSfbAlignBytes - 1) &
        ~(LlGemmConfig::KSfbAlignBytes - 1);
    plan.bars_off = plan.sfb_off + sfb_bytes;
    plan.acc_off = (plan.bars_off + bar_fixed + bar_per_stage * stages +
                    LlGemmConfig::KAlignBytes - 1) &
                   ~(LlGemmConfig::KAlignBytes - 1);
    plan.total_bytes = plan.acc_off + acc_bytes;
    return plan;
}

struct LlTileSpec {
    int block_n = 0;
    int warps_k = 0;
    int stages = 0;
};

// block_n * warps_k <= 128 caps a CTA at 16 consumer warps.
inline constexpr LlTileSpec kLlTiles[] = {
    {8, 1},  {8, 2},  {8, 4},  {8, 8},  {8, 16}, {16, 1},
    {16, 2}, {16, 4}, {16, 8}, {32, 1}, {32, 2}, {32, 4},
};

inline int LlFitStages(int warps_n, int warps_k, int rows_a, int k, int k1) {
    const LlSmemPlan plan = ll_smem_plan(warps_n, warps_k, rows_a, k, k1, 0);
    const int budget = LlGemmConfig::KSmemBudgetBytes - plan.fixed_bytes;
    int fit = 0;
    for (const int stages : LlGemmConfig::KStageTiers) {
        if (stages * plan.per_stage_bytes <= budget) fit = stages;
    }
    return fit;
}

inline void ResolveLlShapes(const torch::Tensor& input,
                            const torch::Tensor& weight,
                            const torch::Tensor& sfa, const torch::Tensor& sfb,
                            const std::optional<torch::Tensor>& input2,
                            const std::optional<torch::Tensor>& weight2,
                            const std::optional<torch::Tensor>& sfa2,
                            const std::optional<torch::Tensor>& sfb2,
                            const std::optional<torch::Tensor>& output2) {
    const int m = input.size(0);
    const int n = weight.size(0);
    const int n2 = output2.has_value() ? weight2->size(0) : 0;
    const int k = weight.size(1) + (input2.has_value() ? weight2->size(1) : 0);
    TORCH_CHECK(
        input2.has_value() == sfa2.has_value() &&
            weight2.has_value() == sfb2.has_value() &&
            weight2.has_value() == (input2.has_value() != output2.has_value()),
        "weight2 / sfb2 must come with either input2 / sfa2 or "
        "output2");
    TORCH_CHECK(m <= LlGemmConfig::KMaxM,
                "low_latency_mma_gemm supports M <= ", LlGemmConfig::KMaxM,
                ", got M=", m);
    TORCH_CHECK(n % LlGemmConfig::KMmaN == 0 && n2 % LlGemmConfig::KMmaN == 0,
                "low_latency_mma_gemm weight rows must be divisible by ",
                LlGemmConfig::KMmaN, ", got ", n, " and ", n2);
    TORCH_CHECK(k % (LlGemmConfig::KGroupK * LlGemmConfig::KBPhaseGroups) == 0,
                "low_latency_mma_gemm K must be divisible by ",
                LlGemmConfig::KGroupK * LlGemmConfig::KBPhaseGroups,
                ", got K=", k);
    // sfa2 starts 16-byte aligned in each shared-memory sfa row.
    TORCH_CHECK(
        !input2.has_value() || input.size(1) % (LlGemmConfig::KScaleGroupK *
                                                LlGemmConfig::KAlignBytes) ==
                                   0,
        "concat-K K1 must be a multiple of ",
        LlGemmConfig::KScaleGroupK * LlGemmConfig::KAlignBytes,
        ", got K1=", input.size(1));

    // An SfAtom sfb with N % 128 == 0 cannot be told apart from a plain one.
    const auto scale_ok = [](const torch::Tensor& scale, int rows) {
        return scale.size(0) == rows &&
               scale.stride(0) % LlGemmConfig::KAlignBytes == 0;
    };
    TORCH_CHECK(
        scale_ok(sfa, m) && scale_ok(sfb, n) &&
            (!sfa2.has_value() || scale_ok(*sfa2, m)) &&
            (!sfb2.has_value() || scale_ok(*sfb2, weight2->size(0))),
        "sfa / sfb / sfa2 / sfb2 must be plain row-major [rows, K / 32] "
        "scales (group_wise_quant(..., use_swizzle=False)) with a row stride "
        "that is a multiple of 16 bytes; got sfa ",
        sfa.sizes(), " strides ", sfa.strides(), ", sfb ", sfb.sizes(),
        " strides ", sfb.strides());
}

// stages == 0 when no tile fits.
inline LlTileSpec SelectLlTile(int n, int k, int k1, int rows_a, int n_split,
                               int sm_count) {
    LlTileSpec best;
    double best_cost = 0.0;
    int best_warps = 0;
    for (const LlTileSpec& tile : kLlTiles) {
        if (n % tile.block_n != 0 || n_split % tile.block_n != 0 ||
            (k / LlGemmConfig::KGroupK) %
                    (LlGemmConfig::KBPhaseGroups * tile.warps_k) !=
                0)
            continue;
        const int warps_n = tile.block_n / LlGemmConfig::KMmaN;
        const int stages = LlFitStages(warps_n, tile.warps_k, rows_a, k, k1);
        if (stages == 0) continue;
        const int warps = warps_n * tile.warps_k;
        const int waves = (n / tile.block_n + sm_count - 1) / sm_count;
        const double cost =
            waves * tile.block_n * (1.0 + LlGemmConfig::KCostIssue / warps) *
            (warps_n == 1 ? LlGemmConfig::KCostNarrowTile : 1.0) *
            (stages == 1 ? LlGemmConfig::KCostSingleStage : 1.0);
        // Ties go to more warps, then to the wider tile.
        const bool better =
            best.stages == 0 || cost < best_cost ||
            (cost == best_cost &&
             (warps > best_warps ||
              (warps == best_warps && tile.block_n > best.block_n)));
        if (better) {
            best = {tile.block_n, tile.warps_k, stages};
            best_cost = cost;
            best_warps = warps;
        }
    }
    return best;
}

}  // namespace halops
