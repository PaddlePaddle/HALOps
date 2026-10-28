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

#include <algorithm>
#include <cstdint>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <iterator>
#include <optional>

namespace halops {

struct LowLatencyMmaGemmParams {
    const c10::Float8_e4m3fn* __restrict__ ptr_a = nullptr;
    const c10::Float8_e4m3fn* __restrict__ ptr_a2 = nullptr;
    const c10::Float8_e4m3fn* __restrict__ ptr_b = nullptr;
    const uint8_t* __restrict__ ptr_sfa = nullptr;
    const uint8_t* __restrict__ ptr_sfa2 = nullptr;
    const uint8_t* __restrict__ ptr_sfb = nullptr;
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
    CUtensorMap b_desc = {};
    CUtensorMap a_desc = {};
    CUtensorMap a_desc2 = {};
    CUtensorMap sfb_desc = {};
};

// Configuration shared by every tile, like GemmKernelConfig.
struct LlGemmConfig {
    // K-blocks per B ring stage.
    static constexpr int KBPhaseGroups = 2;
    // sfa rows resident per CTA, the largest M tier.
    static constexpr int KSfaRows = 16;
    static constexpr int KSfbStages = 4;
    static constexpr int KGroupK = 128;
    static constexpr int KScaleGroupK = 32;
    // Cap on B ring stages: deeper rings measured no faster, the B stream is
    // already bandwidth bound.
    static constexpr int KMaxStages = 8;
    // Accumulator slots between the consumer and epilogue warps.
    static constexpr int KAccStages = 3;
    // N of one mma.sync.m16n8k32: each consumer warp owns 8 output columns.
    static constexpr int KMmaN = 8;
    // One SWIZZLE_128B atom: 8 rows of 128 bytes.
    static constexpr int KAtomBytes = 8 * 128;
    static constexpr int KSmemBudgetBytes =
        cutlass::arch::sm100_smem_capacity_bytes;
    static constexpr int KAlignBytes = 16;
    static constexpr int KSfbAlignBytes = 128;
    static constexpr int KASlabAlignBytes = 1024;
    static constexpr uint64_t KTmaCacheHint =
        static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL);
    // Tile cost model, per unit of K: a CTA streams waves * block_n bytes of
    // B (the bandwidth term) and splits the MMAs over block_n / 8 * warps_k
    // consumer warps (the issue term, KCostIssue / warps). Fitted on GB200
    // (152 SMs) to an exhaustive tile sweep of 48 decode shapes (M 1/8/16,
    // N 1K-32K, K 1K-8K, cold L2); the pick is within 5% of the fastest tile
    // on all of them.
    static constexpr double KCostIssue = 2.0;
    // A one-warp-wide tile moves a single atom per B load.
    static constexpr double KCostNarrowTile = 1.15;
    // One B stage cannot overlap loads with MMAs.
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

// Byte idx of a packed scale word is the UE8M0 exponent of 32-wide k-group
// idx inside one 128-wide k-block.
__device__ __forceinline__ float decode_packed_ue8m0_word(uint32_t packed,
                                                          int idx) {
    return __int_as_float(((packed >> ((idx & 3) * 8)) & 0xffu) << 23);
}

__device__ __forceinline__ uint32_t lds_u32(const uint8_t* smem_ptr) {
    return *reinterpret_cast<const uint32_t*>(smem_ptr);
}

using Barrier = cutlass::arch::ClusterTransactionBarrier;

__device__ __forceinline__ int swz128(int row, int chunk) {
    return row * 128 + ((chunk ^ row) & 7) * 16;
}

// sfa / sfb shared-memory row pitch, in int32 words, for `words` packed
// 128-wide k-blocks.
__host__ __device__ __forceinline__ int ll_scale_pitch(int words) {
    const int p = (words + 3) & ~3;
    return (p & 4) ? p : p + 4;
}

__host__ __device__ __forceinline__ int ll_sfa_g32_row_bytes(int k) {
    return (k / LlGemmConfig::KScaleGroupK + LlGemmConfig::KAlignBytes - 1) &
           ~(LlGemmConfig::KAlignBytes - 1);
}

// int32 words of one sfb ring slot of a block_n-wide tile.
__host__ __device__ __forceinline__ int ll_sfb_slot_ints(int block_n, int k) {
    return (block_n * ll_scale_pitch(k / LlGemmConfig::KGroupK) + 31) & ~31;
}

// Warp warp_k owns the contiguous k-block range [warp_k * n_iter, +n_iter).
__device__ __forceinline__ int ll_kb(int warp_k, int j, int n_iter) {
    return warp_k * n_iter + j;
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

// Shared memory layout of one CTA. Each B stage holds one phase per K-split
// ring (warps_k rings) plus a full and an empty barrier per ring and stage.
__host__ __device__ __forceinline__ LlSmemPlan ll_smem_plan(int warps_n,
                                                            int warps_k,
                                                            int rows_a, int k,
                                                            int stages) {
    const int outs = rows_a == 16 ? 4 : 2;
    const int acc_cols = LlGemmConfig::KAccStages * warps_n;
    const int stage_bytes = warps_k * LlGemmConfig::KBPhaseGroups * warps_n *
                            LlGemmConfig::KAtomBytes;
    const int bar_per_stage = 8 * (warps_k + 1);
    const int a_bytes = (rows_a * k + LlGemmConfig::KASlabAlignBytes - 1) &
                        ~(LlGemmConfig::KASlabAlignBytes - 1);
    const int sfa_bytes =
        4 * LlGemmConfig::KSfaRows * ll_scale_pitch(k / LlGemmConfig::KGroupK);
    const int sfb_bytes = 4 * LlGemmConfig::KSfbStages *
                          ll_sfb_slot_ints(warps_n * LlGemmConfig::KMmaN, k);
    const int bar_fixed = 8 * (k / LlGemmConfig::KGroupK + 1 +
                               2 * LlGemmConfig::KSfbStages + 2 * acc_cols);
    const int acc_bytes = acc_cols * warps_k * 32 * outs * sizeof(float);

    LlSmemPlan p;
    p.fixed_bytes = a_bytes + sfa_bytes + (LlGemmConfig::KSfbAlignBytes - 1) +
                    sfb_bytes + bar_fixed + acc_bytes +
                    (LlGemmConfig::KAlignBytes - 1);
    p.per_stage_bytes = stage_bytes + bar_per_stage;
    p.b_off = a_bytes;
    p.sfa_off = p.b_off + stage_bytes * stages;
    p.sfb_off = (p.sfa_off + sfa_bytes + LlGemmConfig::KSfbAlignBytes - 1) &
                ~(LlGemmConfig::KSfbAlignBytes - 1);
    p.bars_off = p.sfb_off + sfb_bytes;
    p.acc_off = (p.bars_off + bar_fixed + bar_per_stage * stages +
                 LlGemmConfig::KAlignBytes - 1) &
                ~(LlGemmConfig::KAlignBytes - 1);
    p.total_bytes = p.acc_off + acc_bytes;
    return p;
}

// B ring stages that fit next to the resident A, capped at KMaxStages.
inline int LowLatencyMmaFitStages(int warps_n, int warps_k, int rows_a, int k) {
    const LlSmemPlan plan = ll_smem_plan(warps_n, warps_k, rows_a, k, 0);
    const int budget = LlGemmConfig::KSmemBudgetBytes - plan.fixed_bytes;
    if (budget <= 0) return 0;
    return std::min(budget / plan.per_stage_bytes, LlGemmConfig::KMaxStages);
}

// Tile table: every (block_n, warps_k) x M tier x B ring depth is one kernel
// instance.
struct LlTileSpec {
    int block_n;
    int warps_k;
};

// block_n * warps_k <= 128 caps a CTA at 16 consumer warps.
inline constexpr LlTileSpec kLlTiles[] = {
    {8, 1},  {8, 2},  {8, 4},  {8, 8},  {8, 16}, {16, 1},
    {16, 2}, {16, 4}, {16, 8}, {32, 1}, {32, 2}, {32, 4},
};
inline constexpr int kLlRowsTiers[] = {8, 16};
inline constexpr int kLlStageTiers[] = {1, 2, 4, LlGemmConfig::KMaxStages};

// Deepest stage tier that fits, or -1 when not even one stage fits.
inline int LlStageTier(int fit_stages) {
    for (int i = std::size(kLlStageTiers) - 1; i >= 0; --i) {
        if (kLlStageTiers[i] <= fit_stages) return i;
    }
    return -1;
}

constexpr int LlLauncherIndex(int tile, int rows_tier, int stage_tier) {
    return (tile * std::size(kLlRowsTiers) + rows_tier) *
               std::size(kLlStageTiers) +
           stage_tier;
}

// Rejects shapes and scale layouts the kernel would compute wrongly.
inline void ResolveLlShapes(const torch::Tensor& input,
                            const torch::Tensor& weight,
                            const torch::Tensor& sfa, const torch::Tensor& sfb,
                            const torch::Tensor& output,
                            const std::optional<torch::Tensor>& output2,
                            int n_split,
                            const std::optional<torch::Tensor>& input2,
                            const std::optional<torch::Tensor>& sfa2) {
    const int m = input.size(0);
    const int n = weight.size(0);
    const int k = weight.size(1);
    const int k1 = input.size(1);
    TORCH_CHECK(!input2.has_value() || sfa2.has_value(),
                "dual-A (input2) also needs sfa2");
    TORCH_CHECK(!(input2.has_value() && output2.has_value()),
                "dual-A concat-K and dual-output are mutually exclusive");
    TORCH_CHECK(sfa.scalar_type() == torch::kUInt8 &&
                    sfb.scalar_type() == torch::kUInt8,
                "low_latency_mma_gemm supports group-32 scales only (uint8 "
                "sfa/sfb), got sfa=",
                sfa.scalar_type(), ", sfb=", sfb.scalar_type());
    TORCH_CHECK(m > 0 && n > 0 && k > 0,
                "M, N and K must be positive, got m=", m, ", n=", n, ", k=", k);
    TORCH_CHECK(m <= 16,
                "low_latency_mma_gemm is the M <= 16 decode kernel; got m=", m);
    TORCH_CHECK(n % LlGemmConfig::KMmaN == 0,
                "low_latency_mma_gemm N must be divisible by ",
                LlGemmConfig::KMmaN, ", got N=", n);

    if (input2.has_value()) {
        const int k2 = input2->size(1);
        TORCH_CHECK(sfa2->scalar_type() == sfa.scalar_type(),
                    "sfa2 must have sfa's dtype, got sfa=", sfa.scalar_type(),
                    ", sfa2=", sfa2->scalar_type());
        TORCH_CHECK(k1 + k2 == k,
                    "dual-A input K1 + input2 K2 must equal weight K, got K1=",
                    k1, ", K2=", k2, ", weight K=", k);
        TORCH_CHECK(input2->size(0) == m, "input2 M must match input M, got ",
                    input2->size(0), " vs ", m);
        TORCH_CHECK(
            k1 % LlGemmConfig::KGroupK == 0 && k2 % LlGemmConfig::KGroupK == 0,
            "group-32 dual-A K1 and K2 must be multiples of ",
            LlGemmConfig::KGroupK, ", got K1=", k1, ", K2=", k2);
        // sfa2 lands at byte K1 / 32 of each shared-memory sfa row.
        TORCH_CHECK(
            k1 % (LlGemmConfig::KScaleGroupK * LlGemmConfig::KAlignBytes) == 0,
            "group-32 dual-A K1 must be a multiple of ",
            LlGemmConfig::KScaleGroupK * LlGemmConfig::KAlignBytes,
            ", got K1=", k1,
            "; concatenate input / input2 and their scales along K for other "
            "splits");
    } else {
        TORCH_CHECK(k1 == k, "input K must match weight K, got input K=", k1,
                    ", weight K=", k);
    }
    TORCH_CHECK(k % LlGemmConfig::KGroupK == 0,
                "low_latency_mma_gemm K must be divisible by ",
                LlGemmConfig::KGroupK, ", got ", k);
    // Every K split walks whole two-k-block B phases.
    TORCH_CHECK((k / LlGemmConfig::KGroupK) % LlGemmConfig::KBPhaseGroups == 0,
                "low_latency_mma_gemm K must be divisible by ",
                LlGemmConfig::KGroupK * LlGemmConfig::KBPhaseGroups, ", got ",
                k);

    // Plain row-major [rows, K / 32] group-32 scales, copied as whole
    // 16-byte-rounded rows. An SfAtom sfa is rejected by its padded row count;
    // an SfAtom sfb with N % 128 == 0 has the same shape and cannot be told
    // apart.
    const auto scale_ok = [](const torch::Tensor& sf, int rows, int sf_k) {
        return sf.size(0) == rows &&
               sf.size(1) == sf_k / LlGemmConfig::KScaleGroupK &&
               sf.stride(1) == 1 &&
               sf.stride(0) % LlGemmConfig::KAlignBytes == 0;
    };
    TORCH_CHECK(
        scale_ok(sfa, m, k1) && scale_ok(sfb, n, k) &&
            (!input2.has_value() || scale_ok(*sfa2, m, k - k1)),
        "sfa / sfb / sfa2 must be plain row-major [rows, K / 32] group-32 "
        "scales (group_wise_quant(..., use_swizzle=False)) whose row stride "
        "is a multiple of 16 bytes; got sfa ",
        sfa.sizes(), " strides ", sfa.strides(), ", sfb ", sfb.sizes(),
        " strides ", sfb.strides());

    if (output2.has_value()) {
        TORCH_CHECK(n_split > 0 && n_split < n, "n_split must be in (0, n)");
        TORCH_CHECK(n_split % LlGemmConfig::KMmaN == 0,
                    "low_latency_mma_gemm n_split must be divisible by ",
                    LlGemmConfig::KMmaN, ", got n_split=", n_split);
        TORCH_CHECK(output.size(0) == m && output.size(1) == n_split,
                    "output (segment 0) must be [M, n_split], got [",
                    output.size(0), ", ", output.size(1), "], expected [", m,
                    ", ", n_split, "]");
        TORCH_CHECK(output2->size(0) == m && output2->size(1) == n - n_split,
                    "output2 (segment 1) must be [M, n - n_split], got [",
                    output2->size(0), ", ", output2->size(1), "], expected [",
                    m, ", ", n - n_split, "]");
    } else {
        TORCH_CHECK(output.size(0) == m && output.size(1) == n,
                    "output must be [M, N], got [", output.size(0), ", ",
                    output.size(1), "], expected [", m, ", ", n, "]");
    }
}

// Launcher index of the cheapest tile under the LlGemmConfig cost model, or
// -1 when the resident A leaves no room for a B stage.
inline int SelectLlTile(int n, int k, int rows_tier, int n_split, int grid_x) {
    int best = -1;
    double best_cost = 0.0;
    int best_warps = 0;
    int best_block_n = 0;
    for (std::size_t i = 0; i < std::size(kLlTiles); ++i) {
        const LlTileSpec& t = kLlTiles[i];
        if (n % t.block_n != 0 || n_split % t.block_n != 0 ||
            (k / LlGemmConfig::KGroupK) %
                    (LlGemmConfig::KBPhaseGroups * t.warps_k) !=
                0)
            continue;
        const int warps_n = t.block_n / LlGemmConfig::KMmaN;
        const int fit = LowLatencyMmaFitStages(warps_n, t.warps_k,
                                               kLlRowsTiers[rows_tier], k);
        const int tier = LlStageTier(fit);
        if (tier < 0) continue;
        const int warps = warps_n * t.warps_k;
        const int waves = (n / t.block_n + grid_x - 1) / grid_x;
        const double cost =
            waves * t.block_n * (1.0 + LlGemmConfig::KCostIssue / warps) *
            (warps_n == 1 ? LlGemmConfig::KCostNarrowTile : 1.0) *
            (fit == 1 ? LlGemmConfig::KCostSingleStage : 1.0);
        // Ties go to more warps, then to the wider tile.
        const bool better =
            best < 0 || cost < best_cost ||
            (cost == best_cost &&
             (warps > best_warps ||
              (warps == best_warps && t.block_n > best_block_n)));
        if (better) {
            best = LlLauncherIndex(i, rows_tier, tier);
            best_cost = cost;
            best_warps = warps;
            best_block_n = t.block_n;
        }
    }
    return best;
}

}  // namespace halops
