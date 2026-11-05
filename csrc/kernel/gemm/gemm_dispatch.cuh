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
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <span>
#include <utility>

#include "kernel/gemm/gemm_kernel.cuh"
#include "kernel/gemm/gemm_tile.cuh"
#include "kernel/gemm/gemm_utils.h"

namespace halops {

// Shape of one launch: m is the rows one expert contributes (what a tile's M
// has to divide), m_grid the rows the grid covers.
struct GemmShape {
    int m;
    int n;
    int k;
    int m_grid;
    int sm_count;
    bool has_bias;
    EpilogueOp op = EpilogueOp::Identity;
};

constexpr int CeilDiv(int a, int b) { return (a + b - 1) / b; }

constexpr const char* ModeName(KernelMode mode) {
    switch (mode) {
        case KernelMode::Dense:
            return "dense";
        case KernelMode::Masked:
            return "masked";
        case KernelMode::Contiguous:
            return "contiguous";
        case KernelMode::Batch:
            return "batch";
    }
    return "unknown";
}

constexpr bool TileAllowedFor(const TileSpec& t, KernelMode mode) {
    return !t.dense_only || mode == KernelMode::Dense ||
           (t.batch && mode == KernelMode::Batch);
}

constexpr bool TileFitsShape(const TileSpec& t, const GemmShape& s,
                             KernelMode mode) {
    // A small tile only pays off when the grid would otherwise be too small.
    if (IsQuantOp(s.op) && !t.quantizes()) {
        return false;
    }
    if (t.swap_ab && !t.pair()) {
        if (s.m > 2 * t.act_m()) {
            return false;
        }
    } else if (t.act_n() <= 64 && !t.pair() && s.m > 4 * t.act_m()) {
        return false;
    }
    // An activation N that is not 128-aligned overhangs N through the scale
    // run, so N only has to be aligned to the atomic 128 columns.
    const bool n_ok = (t.act_n() > 128 && t.act_n() % 128 != 0)
                          ? (s.n % 128 == 0 && s.n >= 384)
                          : s.n % (t.act_n() * t.cluster_n) == 0;
    if (!n_ok || s.k % t.block_k != 0) {
        return false;
    }
    // Dense may pad a short activation; Batch runs a tile straddling two
    // batches once per batch and drops the rows past the grid; the other
    // modes want whole tiles.
    if (s.m % t.act_m() == 0 || mode == KernelMode::Batch) {
        return true;
    }
    return !t.pair() && mode == KernelMode::Dense && s.m <= t.act_m();
}

// Cost model
namespace cost {
inline constexpr double kWeight = 40.0;     // per weight byte read from L2
inline constexpr double kStage = 5.0e5;     // per K stage of pipeline fill
inline constexpr double kPairFill = 5.5e6;  // a pair MMA's extra fill
inline constexpr double kOverlap = 1.05;    // TMEM overlap costs a little
inline constexpr double kActN224 = 1.01;    // 224-wide reads 3 atoms/2
inline constexpr double kSwapPair = 1.03;   // SwapAB pair epilogue + SF run
inline constexpr double kTieBand = 3e-3;
inline constexpr double kQuantTail = 0.3;  // quant epilogue tail per tile MAC
inline constexpr double kQuantRowShift = 1.01;  // 224 SwapAB quant SFB shift
// Dense Fp8 prices an M below one legacy 128-row tile as 128 rows.
inline constexpr int kMFloorFp8 = 128;
}  // namespace cost

// When the shape matches, skip the cost model and use this
// tile.
struct TileOverride {
    bool (*matches)(const GemmShape&);
    TileSpec tile;
};

inline constexpr TileOverride kDenseFp8Overrides[] = {
    {[](const GemmShape& s) {
         return s.sm_count == 152 && s.m == 1024 && s.n == 4096 &&
                s.k >= 1024 && s.k <= 2048;
     },
     kFp8_128x224},
    {[](const GemmShape& s) {
         return s.m == 8192 && s.n == 4096 && s.k == 1536;
     },
     kFp8_256x224_pair_reg160},
    {[](const GemmShape& s) { return s.m <= 16 && s.n >= 8192 && s.k >= 4096; },
     kFp8_128x16x256_swap},
    {[](const GemmShape& s) { return s.m <= 16 && s.n >= 16384; },
     kFp8_128x16x256_swap},
    {[](const GemmShape& s) { return s.m > 16 && s.m <= 64 && s.n >= 16384; },
     kFp8_128x64x256_swap},
};

// A 16-row swap tile reads one SfAtom column, so it serves a grid of at most
// 16 rows.
inline constexpr TileOverride kBatchFp8Overrides[] = {
    {[](const GemmShape& s) { return s.m_grid <= 16; }, kFp8_128x16x256_swap},
};

inline constexpr TileOverride kDenseFp4Overrides[] = {
    {[](const GemmShape& s) { return s.m == 128 && s.n >= 16384; },
     kFp4_128x256},
    // Very large M on a narrow N with a short grid: halve N.
    {[](const GemmShape& s) {
         if (s.m < 4096 || s.n > 4096) {
             return false;
         }
         const int units = s.sm_count / 2;
         const int grid_tiles = CeilDiv(s.m_grid, 256) * CeilDiv(s.n, 128);
         return units > 0 && CeilDiv(grid_tiles, units) <= 8;
     },
     kFp4_256x128_pair},
    // Small M: SwapAB (the activation becomes the MMA's N).
    {[](const GemmShape& s) { return s.m <= 16; }, kFp4_128x16x256_swap_reg160},
    {[](const GemmShape& s) { return s.m > 16 && s.m <= 32; },
     kFp4_128x32x256_swap},
    {[](const GemmShape& s) { return s.m > 32 && s.m <= 64; },
     kFp4_128x64x512_swap_reg160},
    // 64 < m <= 128.
    {[](const GemmShape& s) { return s.m > 64 && s.m <= 128 && s.m % 64 != 0; },
     kFp4_128x128x256_swap},
    {[](const GemmShape& s) { return s.m > 64 && s.m <= 128 && s.n >= 8192; },
     kFp4_128x64x512_swap_reg160},
    {[](const GemmShape& s) { return s.m > 64 && s.m <= 128; },
     kFp4_128x64x512_swap},
    // 128 < m <= 2048.
    {[](const GemmShape& s) {
         return s.m > 128 && s.m <= 2048 && s.n >= 1536 && s.n <= 4096;
     },
     kFp4_256x128_pair},
    {[](const GemmShape& s) { return s.m > 128 && s.m <= 2048 && s.n <= 8192; },
     kFp4_128x256_reg160},
    {[](const GemmShape& s) { return s.m > 128 && s.m <= 2048; },
     kFp4_256x256_pair_reg160},
    // m > 2048.
    {[](const GemmShape& s) { return s.m > 2048; }, kFp4_256x256_pair_reg160},
};

inline std::span<const TileOverride> Overrides(MmaKind kind, KernelMode mode) {
    if (mode == KernelMode::Batch && kind == MmaKind::MXFP8) {
        return kBatchFp8Overrides;
    }
    if (mode != KernelMode::Dense) {
        return {};
    }
    switch (kind) {
        case MmaKind::MXFP8:
            return kDenseFp8Overrides;
        case MmaKind::NVFP4:
        case MmaKind::MXFP4:
            return kDenseFp4Overrides;
        case MmaKind::BF16:
            break;
    }
    return {};
}

// Activation tiles a launch runs: Batch runs a tile straddling two batches
// once per batch.
inline int ActTiles(const GemmShape& s, const TileSpec& t, KernelMode mode) {
    const int act_m = t.act_m();
    if (mode != KernelMode::Batch || s.m % act_m == 0) {
        return CeilDiv(s.m_grid, act_m);
    }
    if (t.swap_ab && t.pair()) {
        const int align = std::gcd(act_m, kSfAtomM);
        return s.m_grid / s.m *
               CeilDiv(s.m + align - std::gcd(s.m, align), act_m);
    }
    int tiles = 0;
    for (int b = 0; b < s.m_grid / s.m; ++b) {
        tiles += ((b + 1) * s.m - 1) / act_m - b * s.m / act_m + 1;
    }
    return tiles;
}

// PickByCostModel: the heuristic.
inline int PickByCostModel(const TileSpec* tiles, int count, MmaKind kind,
                           const GemmShape& s, KernelMode mode) {
    // Dense Fp8 prices an M below one legacy 128-row tile as 128 rows.
    GemmShape c = s;
    if (mode == KernelMode::Dense && kind == MmaKind::MXFP8) {
        c.m = std::max(c.m, cost::kMFloorFp8);
        c.m_grid = c.m;
    }
    int best = -1;
    double best_cost = 0.0;
    double best_area = 0.0;
    int best_ctas = 0;
    for (int i = 0; i < count; ++i) {
        const TileSpec& t = tiles[i];
        if (t.bias || !t.searchable || !TileAllowedFor(t, mode) ||
            !TileFitsShape(t, c, mode)) {
            continue;
        }
        const int ctas = t.pair() ? 2 : 1;
        const int units = c.sm_count / ctas;
        if (units <= 0) {
            continue;
        }
        const int grid_tiles = ActTiles(c, t, mode) * CeilDiv(c.n, t.act_n());
        const int waves = CeilDiv(grid_tiles, units);
        const double area = static_cast<double>(t.act_m() / ctas) * t.act_n();
        // A pair tile multicasts its weight, so only half comes from L2.
        const double weight_bytes = static_cast<double>(t.act_n()) * c.k / ctas;
        const double tile_cost =
            area * c.k + weight_bytes * cost::kWeight +
            static_cast<double>(c.k / t.block_k) * cost::kStage +
            (ctas == 2 ? cost::kPairFill : 0.0);
        const double pair_eff = (ctas == 2 && waves > 1) ? 0.95 : 1.0;
        double total = waves * tile_cost * pair_eff *
                       (t.tmem_overlap ? cost::kOverlap : 1.0) *
                       (t.act_n() == 224 ? cost::kActN224 : 1.0) *
                       (t.swap_ab && t.pair() ? cost::kSwapPair : 1.0);
        if (IsQuantOp(s.op)) {
            total += cost::kQuantTail * area * c.k;
            if (t.swap_ab && t.pair() && t.act_m() % kSfbRowShiftAlign != 0) {
                total *= cost::kQuantRowShift;
            }
        }
        const bool better =
            best < 0 || total < best_cost * (1.0 - cost::kTieBand) ||
            (total <= best_cost * (1.0 + cost::kTieBand) &&
             (area > best_area || (area == best_area && ctas > best_ctas)));
        if (better) {
            best = i;
            best_cost = total;
            best_area = area;
            best_ctas = ctas;
        }
    }
    return best;
}

inline int PickTile(const TileSpec* tiles, int count, MmaKind kind,
                    const GemmShape& s, KernelMode mode) {
    if (s.has_bias) {
        for (int i = 0; i < count; ++i) {
            if (tiles[i].bias) {
                const bool ok = TileAllowedFor(tiles[i], mode) &&
                                TileFitsShape(tiles[i], s, mode);
                return ok ? i : -1;
            }
        }
        return -1;
    }
    // A matching override decides the tile: no heuristic search.
    for (const TileOverride& r : Overrides(kind, mode)) {
        if (!r.matches(s) || !TileAllowedFor(r.tile, mode) ||
            !TileFitsShape(r.tile, s, mode)) {
            continue;
        }

        for (int i = 0; i < count; ++i) {
            if (tiles[i] == r.tile) {
                return i;
            }
        }
        return -1;
    }
    return PickByCostModel(tiles, count, kind, s, mode);
}

// Dispatch
using TileLauncher = void (*)(GemmParams&, const GemmShape&, cudaStream_t);

// Candidates is any type exposing `static constexpr tiles`:
// CandidateTiles<Kind> for the cost-model tables, or the per-mode BF16 tables
// below.
template <MmaKind Kind, KernelMode Mode, class Candidates, std::size_t I,
          EpilogueOp Op>
void LaunchTile(GemmParams& params, const GemmShape& shape,
                cudaStream_t stream) {
    constexpr TileSpec S = Candidates::tiles[I];
    if constexpr (TileAllowedFor(S, Mode) &&
                  (!IsQuantOp(Op) ||
                   (Kind == MmaKind::MXFP8 && S.quantizes()))) {
        using T = MmaTraits<Kind>;
        using TypeD = cute::conditional_t<IsQuantOp(Op), cutlass::float_e4m3_t,
                                          typename T::Output>;

        using OpA = cute::conditional_t<S.swap_ab, typename T::OperandB,
                                        typename T::OperandA>;
        using OpB = cute::conditional_t<S.swap_ab, typename T::OperandA,
                                        typename T::OperandB>;
        gemm_impl<OpA, OpB, typename T::Output, TypeD, typename T::Scale, float,
                  MakeTile<Kind, Mode, S, Op>>(params, shape.sm_count, stream);
    }
}

template <MmaKind Kind, KernelMode Mode, EpilogueOp Op, class Candidates,
          std::size_t... I>
constexpr std::array<TileLauncher, sizeof...(I)> MakeTileLaunchers(
    std::index_sequence<I...>) {
    return {&LaunchTile<Kind, Mode, Candidates, I, Op>...};
}

template <MmaKind Kind, KernelMode Mode, EpilogueOp Op,
          class Candidates = CandidateTiles<Kind>>
inline constexpr auto kTileLaunchers =
    MakeTileLaunchers<Kind, Mode, Op, Candidates>(
        std::make_index_sequence<Candidates::tiles.size()>{});

inline bool GemmTraceEnabled() {
    static const bool on = std::getenv("IR_GEMM_TRACE") != nullptr;
    return on;
}

template <KernelMode Mode, MmaKind Kind, EpilogueOp Op = EpilogueOp::Identity>
void LaunchCfg(GemmParams& params, GemmShape shape, cudaStream_t stream) {
    shape.op = Op;
    constexpr auto& candidates = CandidateTiles<Kind>::tiles;
    const int pick =
        PickTile(candidates.data(), static_cast<int>(candidates.size()), Kind,
                 shape, Mode);
    TORCH_CHECK(pick >= 0, "No ", ModeName(Mode), " ", MmaTraits<Kind>::kName,
                " tile for M=", shape.m, " (grid M=", shape.m_grid,
                "), N=", shape.n, ", K=", shape.k,
                shape.has_bias ? ", with bias" : "");
    if (GemmTraceEnabled()) {
        const TileSpec& t = candidates[pick];
        std::fprintf(stderr,
                     "[gemm] mode=%s kind=%s m=%d n=%d k=%d grid=%d sm=%d "
                     "bias=%d op=%d -> %dx%d/%d pair=%d swap=%d regs=%d "
                     "cfg=%d\n",
                     ModeName(Mode), MmaTraits<Kind>::kName, shape.m, shape.n,
                     shape.k, shape.m_grid, shape.sm_count,
                     shape.has_bias ? 1 : 0, static_cast<int>(Op), t.block_m,
                     t.block_n, t.block_k, t.pair() ? 1 : 0, t.swap_ab ? 1 : 0,
                     t.epi_regs, pick);
    }
    kTileLaunchers<Kind, Mode, Op>[pick](params, shape, stream);
}

// ===========================================================================
// BF16 dispatch. Rule-based rather than cost-model based: the ladder below
// encodes measured launch-shape bands (pair-UMMA gates, wave-fill judgements,
// K-tile depth gates). A selector returns an index into the per-mode tile
// tables in gemm_tile.cuh, or -1 when no tile supports the shape; the launch
// helpers validate the selected tile and run it.
// ===========================================================================
namespace bf16 {

// KBlockK of every BF16 tile.
inline constexpr int kBlockK = 64;

// Deep-pipeline tiles (14/18 stages) only pay off from this many K-tiles.
inline constexpr int kDeepPipelineMinKTiles = 32;

// The 64x96 / 128x64-pair deep tiles need at least this many K-tiles.
inline constexpr int kMidMDeepMinKTiles = 54;

// 256x128 re-reads A twice as often as 256x256; beyond this many K-tiles the
// re-read outweighs any wave-packing gain.
inline constexpr int kNarrowPairMaxKTiles = 64;

// A 64-row tile has no multicast, so every extra N-tile re-reads A; cap the
// re-read volume (k_tiles * n_tiles) before stepping down to one.
inline constexpr int kRereadsCap = 2048;

// True if the deep (14/18-stage) pipeline variant pays off at this K.
inline bool UseDeepPipeline(int k) {
    return k / kBlockK >= kDeepPipelineMinKTiles;
}

// CTAs the 128x64 pair-tile family yields (2 CTAs per cluster tile).
inline int64_t Count128x64PairCtas(int m, int n) {
    return 2 * int64_t(CeilDiv(m, 128)) * CeilDiv(n, 64);
}

// For m < 128: a plain (non-SwapAB) 64-row tile beats SwapAB when even
// SwapAB's smallest tile leaves SMs idle.
inline bool SkipSwapForThinM(int m, int n, int k, int sm_count) {
    if (m % 64 != 0 || k % 64 != 0) {
        return false;
    }
    const int target = ResolveSmCount(sm_count);
    const int mt = CeilDiv(m, 64);
    const bool narrow_ok = (n % 32 == 0) && (CeilDiv(n, 32) * mt <= target);
    const bool square_ok = (n % 64 == 0);
    if (!narrow_ok && !square_ok) {
        return false;
    }
    return CeilDiv(n, 128) * CeilDiv(m, 32) < target;
}

// True if the fine grid keeps a larger fraction of SMs busy (ctas / waves).
inline bool FillsLastWaveBetter(int64_t coarse_ctas, int64_t fine_ctas,
                                int64_t target) {
    const int64_t coarse_waves = (coarse_ctas + target - 1) / target;
    const int64_t fine_waves = (fine_ctas + target - 1) / target;
    return fine_ctas * coarse_waves > coarse_ctas * fine_waves;
}

// 256x128 instead of 256x256 when K is short, the grid is not too deep, and
// 256x128 keeps more SMs busy.
inline bool UseNarrowPairTile(int n, int k, int64_t coarse_ctas,
                              int64_t fine_ctas, int64_t target) {
    return (n % 128 == 0) && (k / kBlockK <= kNarrowPairMaxKTiles) &&
           (coarse_ctas <= 4 * target) &&
           FillsLastWaveBetter(coarse_ctas, fine_ctas, target);
}

// ---- Small-N (N <= 128) tiles ----------------------------------------------

inline constexpr int kSmallNMaxN = 128;
inline constexpr int kSmallNAlignN = 64;

enum class SmallNTile { kNone, kPairM256N128, kM64N128, kPairM256N64, kM64N64 };

struct SmallNRung {
    SmallNTile tile;
    int block_m;
    int block_n;
    int ctas_per_cluster;
};

inline bool FillsOneWave(int64_t ctas, int64_t target) {
    return ctas <= target && 2 * ctas > target;
}

// N fits one tile, so all CTAs come from splitting M: pick the largest tile
// whose CTA count lands in (0.5, 1.0] of one wave.
inline SmallNTile PickSmallNTile(int m, int n, int64_t target) {
    if (n > kSmallNMaxN || n % kSmallNAlignN != 0) {
        return SmallNTile::kNone;
    }
    static constexpr SmallNRung kRungs[] = {
        {SmallNTile::kPairM256N128, 256, 128, 2},
        {SmallNTile::kM64N128, 64, 128, 1},
        {SmallNTile::kPairM256N64, 256, 64, 2},
        {SmallNTile::kM64N64, 64, 64, 1},
    };
    for (const SmallNRung& r : kRungs) {
        const int64_t ctas = int64_t(r.ctas_per_cluster) *
                             CeilDiv(m, r.block_m) * CeilDiv(n, r.block_n);
        if (FillsOneWave(ctas, target)) {
            return r.tile;
        }
    }
    return SmallNTile::kNone;
}

// ---- Dense -----------------------------------------------------------------
// Ids into kBF16DenseTiles (gemm_tile.cuh); the enum order mirrors the table.
enum DenseTile : int {
    kPair256x256,  // 256x256 pair, 6 stages
    kPair256x256P7,
    kPair256x128,  // 8 stages
    kPair256x128Ntail,
    kPair256x128P9,
    kPair256x128P9Ntail,
    kPair256x192,
    kPair256x64,  // 10 stages
    kM128x256,
    kM128x128,
    kM128x64,        // 8 stages
    kM128x32,        // 10 stages
    kPair128x64P18,  // 18 stages, 1 epilogue warp group
    kM64x128,
    kM64x96P11,  // 11 stages, 1 epilogue warp group
    kM64x224,    // 1 epilogue warp group
    kM64x64,     // 2 epilogue warp groups, 10 stages
    kM64x64_1wg,
    kM64x64_1wgP14,
    kM64x32_1wg,
    kM64x32_1wgP18,
    kSwap128x64,
    kSwap128x32,
    kSwap256x64Pair,
    kSwap128x64K128,  // SwapAB, KBlockK=128 (k >= 8192)
};
static_assert(kSwap128x64K128 + 1 == static_cast<int>(kBF16DenseTiles.size()),
              "BF16 dense tile table and DenseTile enum drifted apart");

// m >= 4096: pair tiles, with a small-N ladder for narrow N.
inline int SelectLargeMTile(int m, int n, int k, int target) {
    switch (PickSmallNTile(m, n, target)) {
        case SmallNTile::kM64N128:
            return kM64x128;
        case SmallNTile::kM64N64:
            return UseDeepPipeline(k) ? kM64x64_1wgP14 : kM64x64;
        case SmallNTile::kPairM256N64:
            return kPair256x64;
        case SmallNTile::kPairM256N128:
        case SmallNTile::kNone:
            break;
    }
    const int64_t tiles_256 = int64_t(CeilDiv(m, 256)) * CeilDiv(n, 256);
    // 256x128 (deeper pipeline) when it keeps more SMs busy on a short K.
    if (UseNarrowPairTile(n, k, 2 * tiles_256,
                          2 * int64_t(CeilDiv(m, 256)) * CeilDiv(n, 128),
                          target)) {
        return kPair256x128P9;
    }
    if (n % 256 == 0) {
        return kPair256x256P7;
    }
    if (n % 128 == 0) {
        return kPair256x128;
    }
    // n % 128 != 0 (e.g. 1600): predicated-tail 128-wide tile, else 256x64.
    if (n % 64 == 0) {
        return kPair256x128Ntail;
    }
    return kPair256x64;
}

// Halve the 128-row tile's N (256 -> 128 -> 64 -> 32) while the current width
// leaves SMs idle. Returns the chosen N width.
inline int Pick128xNWidth(int m, int n, int target) {
    auto tiles = [&](int tm, int tn) {
        return CeilDiv(m, tm) * CeilDiv(n, tn);
    };
    int pick = (n % 256 == 0) ? 0 : (n % 128 == 0) ? 1 : (n % 64 == 0) ? 2 : 3;
    // Keep 128x256 when 128x128 would spill into a second wave.
    const bool overshoots_128 =
        m > 128 && tiles(128, 128) > target && 2 * tiles(128, 256) > target;
    if (pick == 0 && tiles(128, 256) < target && (n % 128 == 0) &&
        !overshoots_128) {
        pick = 1;
    }
    if (pick == 1 && tiles(128, 128) < target && (n % 64 == 0)) {
        pick = 2;
    }
    if (pick == 2 && tiles(128, 64) < target && (n % 32 == 0)) {
        pick = 3;
    }
    return pick == 0 ? 256 : (pick == 1 ? 128 : (pick == 2 ? 64 : 32));
}

// 256xN pair-UMMA ladder for m % 256 == 0; returns -1 when its gates reject
// the shape and the 128xN ladder should decide instead.
inline int SelectPair256xNTile(int m, int n, int k, int target) {
    if (n % 64 != 0) {
        return -1;
    }
    auto pair_ctas = [&](int tn) {
        return int64_t(m / 256) * CeilDiv(n, tn) * 2;
    };
    // Halve N only while the grid is under one wave, and never from >= half
    // a wave into more than one wave.
    auto step_ok = [&](int coarse, int fine) {
        return pair_ctas(coarse) < target &&
               !(pair_ctas(fine) > target && 2 * pair_ctas(coarse) > target);
    };
    int pair_tn = (n % 256 == 0) ? 256 : ((n % 128 == 0) ? 128 : 64);
    // n % 128 != 0 (e.g. 1600): the 128-wide predicated-tail tile instead of
    // 64 if it fills at least half a wave.
    bool pair_n_tail = false;
    if (pair_tn == 64 && (n % 128 != 0) && 2 * pair_ctas(128) >= target) {
        pair_tn = 128;
        pair_n_tail = true;
    }
    // Also 128 when the narrow-pair rule says so (multi-wave).
    if (pair_tn == 256 && (n % 128 == 0) &&
        (step_ok(256, 128) ||
         UseNarrowPairTile(n, k, pair_ctas(256), pair_ctas(128), target))) {
        pair_tn = 128;
    }
    // 192 if it fits one wave and keeps more SMs busy than both 256 and 128.
    if (pair_tn != 64 && (n % 32 == 0) && pair_ctas(192) <= target &&
        FillsLastWaveBetter(pair_ctas(256), pair_ctas(192), target) &&
        FillsLastWaveBetter(pair_ctas(128), pair_ctas(192), target)) {
        pair_tn = 192;
        pair_n_tail = false;
    }
    if (!pair_n_tail && pair_tn == 128 && (n % 64 == 0) && step_ok(128, 64)) {
        pair_tn = 64;
    }
    const int64_t cta_count = pair_ctas(pair_tn);
    const int k_tiles = k / kBlockK;
    // Prologue amortization: enough K-tiles to fill three pipelines, or a
    // full wave of tiles.
    const int stages =
        pair_tn == 256 ? 6 : (pair_tn == 192 ? 8 : (pair_tn == 128 ? 8 : 10));
    const bool amortizes = k_tiles >= 3 * stages || cta_count >= target;
    // The 64-wide arm needs at least 0.8 of a wave.
    const bool narrow_arm_fills = pair_tn != 64 || 5 * cta_count >= 4 * target;
    // If n % 128 != 0 forced 256x64 and it spills past one wave, defer to the
    // 64x96 / 64x224 tiles below.
    const int ctas96 = CeilDiv(m, 64) * CeilDiv(n, 96);
    const int ctas224 = CeilDiv(m, 64) * CeilDiv(n, 224);
    const bool narrow_arm_overshoots =
        pair_tn == 64 && cta_count > target && (m % 64 == 0) && (n % 32 == 0) &&
        (!(k_tiles >= kMidMDeepMinKTiles) ||
         FillsLastWaveBetter(ctas224, ctas96, target));
    if (!amortizes || !narrow_arm_fills || narrow_arm_overshoots) {
        return -1;
    }
    switch (pair_tn) {
        case 256:
            return kPair256x256;
        case 192:
            return kPair256x192;
        case 128:
            return pair_n_tail ? kPair256x128P9Ntail : kPair256x128P9;
        default:
            // With <= 32 K-tiles the 1-CTA 128x64 tile (same CTA count) is
            // faster than the 256x64 pair.
            return k_tiles <= 32 ? kM128x64 : kPair256x64;
    }
}

// 128 <= m < 4096.
inline int SelectMidMTile(int m, int n, int k, int target) {
    if (m % 128 != 0 || n % 32 != 0 || k % 64 != 0) {
        return -1;
    }

    // ---- 64-row tiles (64x64 / 64x32) ----------------------------------
    // Smaller tiles add CTAs but, with no multicast, every extra N-tile
    // re-reads A; use them only on a one-wave grid with bounded re-reads.
    if (m % 64 == 0 && n % 64 == 0) {
        const int tiles64 = CeilDiv(m, 64) * CeilDiv(n, 64);
        const int64_t rereads = int64_t(k / kBlockK) * CeilDiv(n, 64);
        if (tiles64 <= target && rereads <= kRereadsCap) {
            // 1 epilogue warp group: required for 64x32, and it skips the
            // inter-warp-group barrier on 64x64.
            if (2 * tiles64 <= target && n % 32 == 0) {
                return UseDeepPipeline(k) ? kM64x32_1wgP18 : kM64x32_1wg;
            }
            return UseDeepPipeline(k) ? kM64x64_1wgP14 : kM64x64_1wg;
        }
    }

    // ---- 256xN pair-UMMA tiles (m % 256 == 0) --------------------------
    if (m % 256 == 0) {
        const int pair = SelectPair256xNTile(m, n, k, target);
        if (pair >= 0) {
            return pair;
        }
    }

    // ---- 128xN width ----------------------------------------------------
    const int pick_tn = Pick128xNWidth(m, n, target);
    auto tiles = [&](int tm, int tn) {
        return CeilDiv(m, tm) * CeilDiv(n, tn);
    };

    // ---- 64x96 / 64x224 fill tiles --------------------------------------
    // Power-of-2 N widths can miss one wave entirely (n=7168: 112 or 224 CTAs
    // on 152 SMs); 96/224 land in between.
    const int ctas96 = CeilDiv(m, 64) * CeilDiv(n, 96);
    const int ctas224 = CeilDiv(m, 64) * CeilDiv(n, 224);
    // The incumbent is the 128xN choice, adjusted for the 64x128 wave fix
    // below when that applies.
    const int incumbent_ctas = (n % 128 == 0 && tiles(128, pick_tn) > target &&
                                tiles(64, 128) <= target)
                                   ? tiles(64, 128)
                                   : tiles(128, pick_tn);
    auto fills_better = [&](int64_t c) {
        return c <= target && 2 * c > target &&
               (incumbent_ctas > target || c > incumbent_ctas);
    };
    // The 256x64 pair arm deferred here in favour of 64x96; take it even
    // though it exceeds one wave.
    const bool pair_deferred_to_96 =
        (m % 256 == 0) && (n % 64 == 0) && (n % 128 != 0) &&
        (int64_t(m / 256) * CeilDiv(n, 64) * 2 > target);
    if ((m % 64 == 0) && (n % 32 == 0) &&
        (fills_better(ctas96) || pair_deferred_to_96) &&
        (k / kBlockK) >= kMidMDeepMinKTiles) {
        return kM64x96P11;
    }
    if ((m % 64 == 0) && (n % 32 == 0) && fills_better(ctas224)) {
        return kM64x224;
    }

    // ---- 64x128 when the 128xN tile spills past one wave -----------------
    if ((m % 64 == 0) && (n % 128 == 0) && tiles(128, pick_tn) > target &&
        tiles(64, 128) <= target) {
        return kM64x128;
    }

    // ---- 128x64 pair, 18 stages ------------------------------------------
    // One wave with enough K-tiles, or the 128x128 incumbent (only 4 stages).
    if ((m % 128 == 0) && (n % 128 == 0)) {
        const int64_t pair_ctas = Count128x64PairCtas(m, n);
        const int k_tiles = k / kBlockK;
        const bool one_wave = pair_ctas <= target && 2 * pair_ctas > target;
        const bool deep_enough = k_tiles >= kMidMDeepMinKTiles;
        const bool shallow_incumbent = (pick_tn == 128);
        if ((one_wave && deep_enough) || shallow_incumbent) {
            // With <= 64 K-tiles the 64x64 1WG deep tile is faster.
            if (one_wave && deep_enough && k_tiles <= 64 && (m % 64 == 0) &&
                (n % 64 == 0)) {
                return kM64x64_1wgP14;
            }
            return kPair128x64P18;
        }
    }

    // ---- 128xN base ------------------------------------------------------
    return pick_tn == 256
               ? kM128x256
               : (pick_tn == 128 ? kM128x128
                                 : (pick_tn == 64 ? kM128x64 : kM128x32));
}

// m < 128 without SwapAB (SkipSwapForThinM): KBlockM=64 tiles give ceil(n/32)
// CTAs; 64x32 if it fits one wave, else 64x64.
inline int SelectThinMTile(int m, int n, int k, int target) {
    const int nt32 = CeilDiv(n, 32);
    const int mt = CeilDiv(m, 64);
    if (n % 32 == 0 && nt32 * mt <= target) {
        return UseDeepPipeline(k) ? kM64x32_1wgP18 : kM64x32_1wg;
    }
    return UseDeepPipeline(k) ? kM64x64_1wgP14 : kM64x64_1wg;
}

// m < 128 with SwapAB: the weight (N x K) becomes the MMA's A (M-dim 128),
// the activation (M x K) the MMA's B (N-dim).
inline int SelectSwapABTile(int m, int n, int k, int target) {
    if (n % 128 != 0 || m % 64 != 0 || k % 64 != 0) {
        return -1;
    }
    // KBlockK=128 from k >= 8192.
    if (k % 128 == 0 && k >= 8192) {
        return kSwap128x64K128;
    }
    // 256-wide pair tile when the 128-wide grid exceeds 2 waves.
    const int tiles128 = CeilDiv(n, 128) * CeilDiv(m, 64);
    if ((n % 256 == 0) && tiles128 > 2 * target) {
        return kSwap256x64Pair;
    }
    // KBlockN=32 if that still fits one wave.
    if (CeilDiv(n, 128) * CeilDiv(m, 32) <= target) {
        return kSwap128x32;
    }
    return kSwap128x64;
}

// The dense ladder; -1 when no tile supports the shape.
inline int SelectBF16Cfg(int m, int n, int k, int sm_count) {
    const int target = ResolveSmCount(sm_count);
    if (m >= 4096) {
        return SelectLargeMTile(m, n, k, target);
    }
    if (m >= 128) {
        return SelectMidMTile(m, n, k, target);
    }
    if (SkipSwapForThinM(m, n, k, sm_count)) {
        return SelectThinMTile(m, n, k, target);
    }
    return SelectSwapABTile(m, n, k, target);
}

// ---- Grouped / batch -------------------------------------------------------

// Ids into kBF16ContiguousTiles (gemm_tile.cuh).
enum ContiguousTile : int {
    kContPair256x256,  // 256x256 pair, 7 stages
    kContPair256x128,  // 256x128 pair, 9 stages
    kContM128x256,     // 128x256, 16 overlap cols
    kContM128x128,     // 128x128, 6 stages
    kContM128x64,      // 128x64, 16 overlap cols
    kContSwap128x64,
};
static_assert(kContSwap128x64 + 1 ==
                  static_cast<int>(kBF16ContiguousTiles.size()),
              "BF16 contiguous tile table and enum drifted apart");

// m is the total token count.
inline int SelectBF16ContiguousCfg(int m, int n, int k, int sm_count) {
    if (m >= 4096) {
        return kContPair256x256;
    }
    if (m >= 128) {
        // M stays 128 (the 1-CTA MMA requires M == 128); step finer on N
        // only while the coarser tile underfills the grid.
        const int target = ResolveSmCount(sm_count);
        auto tiles = [&](int tm, int tn) {
            return CeilDiv(m, tm) * CeilDiv(n, tn);
        };
        // 2-CTA pair-UMMA first: weight TMA-multicast across the pair. Each
        // cluster spans 2 SMs, so require clusters * 2 >= target.
        const int clusters = CeilDiv(m, 256) * CeilDiv(n, 256);
        if (m % 256 == 0 && n % 256 == 0 && k % 64 == 0 &&
            clusters * 2 >= target) {
            // 256x128 frees a deeper pipeline; it pays only when 256x256 is
            // lightly oversubscribed (<= ~2 waves).
            const bool square_like = (n % 128 == 0) && (clusters <= 2 * target);
            return square_like ? kContPair256x128 : kContPair256x256;
        }
        // 128x64 only on a <= 1-wave fit: halving N doubles the M*K panel
        // reloads (costly on DRAM-bound mid-M).
        if (tiles(128, 256) < target && (n % 128 == 0)) {
            if (tiles(128, 64) <= target && (n % 64 == 0)) {
                return kContM128x64;
            }
            return kContM128x128;
        }
        return kContM128x256;
    }
    return kContSwap128x64;
}

// Ids into kBF16MaskedTiles (gemm_tile.cuh).
enum MaskedTile : int {
    kMaskM128x256,  // 128x256, 16 overlap cols
    kMaskSwap128x64,
    kMaskM128x128,     // 128x128, 6 stages
    kMaskM128x64,      // 128x64, 16 overlap cols
    kMaskPair256x256,  // 256x256 pair, 7 stages
    kMaskPair256x128,  // 256x128 pair, 9 stages
};
static_assert(kMaskPair256x128 + 1 == static_cast<int>(kBF16MaskedTiles.size()),
              "BF16 masked tile table and enum drifted apart");

// Masked pads every expert to max_m_per_expert, so the grid scales with
// num_experts * ceil(max_m / tm) * ceil(n / tn).
inline int SelectBF16MaskedCfg(int num_experts, int mm, int n, int k,
                               int sm_count) {
    if (mm >= 4096) {
        return kMaskPair256x256;
    }
    if (mm >= 128) {
        const int target = ResolveSmCount(sm_count);
        auto tiles = [&](int tm, int tn) {
            return num_experts * CeilDiv(mm, tm) * CeilDiv(n, tn);
        };
        const int clusters = num_experts * CeilDiv(mm, 256) * CeilDiv(n, 256);
        if (mm % 256 == 0 && n % 256 == 0 && k % 64 == 0 &&
            clusters * 2 >= target) {
            const bool square_like = (n % 128 == 0) && (clusters <= 2 * target);
            return square_like ? kMaskPair256x128 : kMaskPair256x256;
        }
        // Keep 128x256 (best B-reuse) unless it underfills the grid; a
        // blanket 128x128 preference regresses oversubscribed mid-M.
        if (tiles(128, 256) < target && (n % 128 == 0)) {
            if (tiles(128, 64) <= target && (n % 64 == 0)) {
                return kMaskM128x64;
            }
            return kMaskM128x128;
        }
        return kMaskM128x256;
    }
    return kMaskSwap128x64;
}

// Ids into kBF16BatchTiles (gemm_tile.cuh).
enum BatchTile : int {
    kBatchPair256x256,  // 256x256 pair, 6 stages
    kBatchM128x256,     // 128x256, 16 overlap cols
    kBatchSwap128x64,
};
static_assert(kBatchSwap128x64 + 1 == static_cast<int>(kBF16BatchTiles.size()),
              "BF16 batch tile table and enum drifted apart");

// Pure alignment ladder (no runtime SM reasoning); m is one item's rows.
inline int SelectBF16BatchCfg(int m, int n, int k) {
    if (m % 256 == 0 && n % 256 == 0) {
        return kBatchPair256x256;
    }
    if (m % 128 == 0 && n % 256 == 0) {
        return kBatchM128x256;
    }
    if (m % 64 == 0 && n % 128 == 0) {
        return kBatchSwap128x64;
    }
    return -1;
}

// ---- Tile tables as launcher policies --------------------------------------

struct DenseTiles {
    static constexpr const auto& tiles = kBF16DenseTiles;
};
struct ContiguousTiles {
    static constexpr const auto& tiles = kBF16ContiguousTiles;
};
struct MaskedTiles {
    static constexpr const auto& tiles = kBF16MaskedTiles;
};
struct BatchTiles {
    static constexpr const auto& tiles = kBF16BatchTiles;
};

// Divisibility contract of a selected tile: the output M/N tile multiples
// (N relaxes to %64 for predicated tails and %32 for non-power-of-2 widths),
// K a KBlockK multiple. SwapAB swaps the roles: KBlockM tiles the weight
// (output N), KBlockN the activation (output M).
inline void CheckBF16TileShape(const char* what, const TileSpec& t, int m,
                               int n, int k) {
    const int n_mod =
        t.force_ntail ? 64
                      : (((t.block_n & (t.block_n - 1)) != 0) ? 32 : t.block_n);
    const int m_div = t.swap_ab ? t.block_n : t.block_m;
    const int n_div = t.swap_ab ? t.block_m : n_mod;
    TORCH_CHECK(m % m_div == 0, what, " requires M to be divisible by ", m_div,
                ", got ", m);
    TORCH_CHECK(n % n_div == 0, what, " requires N to be divisible by ", n_div,
                ", got ", n);
    TORCH_CHECK(k % t.block_k == 0, what, " requires K to be divisible by ",
                t.block_k, ", got ", k);
}

template <class Tiles, KernelMode Mode>
inline void LaunchSelected(int cfg, const char* what, int m_check,
                           GemmParams& params, int sm_count,
                           cudaStream_t stream) {
    CheckBF16TileShape(what, Tiles::tiles[cfg], m_check, params.n, params.k);
    const GemmShape shape = {.m = params.m,
                             .n = params.n,
                             .k = params.k,
                             .m_grid = params.m,
                             .sm_count = sm_count,
                             .has_bias = false};
    kTileLaunchers<MmaKind::BF16, Mode, EpilogueOp::Identity, Tiles>[cfg](
        params, shape, stream);
}

// Selects, validates and launches. The .cu mode branches call these.
inline void LaunchDense(GemmParams& params, int sm_count, cudaStream_t stream) {
    const int cfg = SelectBF16Cfg(params.m, params.n, params.k, sm_count);
    TORCH_CHECK(cfg >= 0, "No BF16 dense GEMM tile supports M=", params.m,
                ", N=", params.n, ", K=", params.k);
    LaunchSelected<DenseTiles, KernelMode::Dense>(
        cfg, "BF16 dense GEMM", params.m, params, sm_count, stream);
}

inline void LaunchContiguous(GemmParams& params, int sm_count,
                             cudaStream_t stream) {
    const int cfg =
        SelectBF16ContiguousCfg(params.m, params.n, params.k, sm_count);
    TORCH_CHECK(cfg >= 0,
                "bf16 contiguous GEMM: unsupported shape M=", params.m,
                " N=", params.n, " K=", params.k);
    LaunchSelected<ContiguousTiles, KernelMode::Contiguous>(
        cfg, "bf16 contiguous GEMM", params.m, params, sm_count, stream);
}

inline void LaunchMasked(GemmParams& params, int sm_count,
                         cudaStream_t stream) {
    const int cfg =
        SelectBF16MaskedCfg(params.num_experts, params.max_m_per_expert,
                            params.n, params.k, sm_count);
    TORCH_CHECK(cfg >= 0,
                "bf16 masked GEMM: unsupported shape E=", params.num_experts,
                " max_m_per_expert=", params.max_m_per_expert, " N=", params.n,
                " K=", params.k);
    LaunchSelected<MaskedTiles, KernelMode::Masked>(cfg, "bf16 masked GEMM",
                                                    params.max_m_per_expert,
                                                    params, sm_count, stream);
}

inline void LaunchBatch(GemmParams& params, int sm_count, cudaStream_t stream) {
    const int cfg =
        SelectBF16BatchCfg(params.max_m_per_expert, params.n, params.k);
    TORCH_CHECK(
        cfg >= 0, "bf16 batch GEMM: unsupported shape B=", params.num_experts,
        " M=", params.max_m_per_expert, " N=", params.n, " K=", params.k,
        ". Need one of: (M%256==0 && N%256==0), "
        "(M%128==0 && N%256==0), or (M%64==0 && N%128==0).");
    LaunchSelected<BatchTiles, KernelMode::Batch>(cfg, "bf16 batch GEMM",
                                                  params.max_m_per_expert,
                                                  params, sm_count, stream);
}

}  // namespace bf16
}  // namespace halops
