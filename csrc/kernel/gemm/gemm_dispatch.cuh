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
#include <span>
#include <utility>

#include "kernel/gemm/gemm_kernel.cuh"
#include "kernel/gemm/gemm_tile.cuh"

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
    return !t.dense_only || mode == KernelMode::Dense;
}

constexpr bool TileFitsShape(const TileSpec& t, const GemmShape& s,
                             KernelMode mode) {
    // A small tile only pays off when the grid would otherwise be too small.
    if (t.swap_ab) {
        if (s.m > 2 * t.act_m()) {
            return false;
        }
    } else if (t.act_n() <= 64 && !t.pair() && s.m > 4 * t.act_m()) {
        return false;
    }
    // An activation N that is not 128-aligned overhangs N through the scale
    // run, so N only has to be aligned to the atomic 128 columns.
    const bool n_ok = (t.act_n() % 128 != 0)
                          ? (s.n % 128 == 0 && s.n >= 384)
                          : s.n % (t.act_n() * t.cluster_n) == 0;
    if (!n_ok || s.k % t.block_k != 0) {
        return false;
    }
    // Dense may pad a short activation; every other mode wants whole tiles, and
    // a pair tile always wants whole M tiles.
    if (s.m % t.act_m() == 0) {
        return true;
    }
    if (t.pair()) {
        return false;
    }
    return mode == KernelMode::Dense && s.m <= t.act_m();
}

// Cost model
namespace cost {
inline constexpr double kWeight = 40.0;     // per weight byte read from L2
inline constexpr double kStage = 5.0e5;     // per K stage of pipeline fill
inline constexpr double kPairFill = 5.5e6;  // a pair MMA's extra fill
inline constexpr double kOverlap = 1.05;    // TMEM overlap costs a little
inline constexpr double kActN224 = 1.01;    // 224-wide reads 3 atoms/2
inline constexpr double kTieBand = 3e-3;
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
         return s.m == 4096 && s.n == 4096 && s.k == 4096;
     },
     kFp8_256x192_pair_reg160},
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
        const int grid_tiles =
            CeilDiv(c.m_grid, t.act_m()) * CeilDiv(c.n, t.act_n());
        const int waves = CeilDiv(grid_tiles, units);
        const double area = static_cast<double>(t.act_m() / ctas) * t.act_n();
        // A pair tile multicasts its weight, so only half comes from L2.
        const double weight_bytes = static_cast<double>(t.act_n()) * c.k / ctas;
        const double tile_cost =
            area * c.k + weight_bytes * cost::kWeight +
            static_cast<double>(c.k / t.block_k) * cost::kStage +
            (ctas == 2 ? cost::kPairFill : 0.0);
        const double pair_eff = (ctas == 2 && waves > 1) ? 0.95 : 1.0;
        const double total = waves * tile_cost * pair_eff *
                             (t.tmem_overlap ? cost::kOverlap : 1.0) *
                             (t.act_n() == 224 ? cost::kActN224 : 1.0);
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

template <MmaKind Kind, KernelMode Mode, std::size_t I>
void LaunchTile(GemmParams& params, const GemmShape& shape,
                cudaStream_t stream) {
    if constexpr (TileAllowedFor(CandidateTiles<Kind>::tiles[I], Mode)) {
        using T = MmaTraits<Kind>;
        gemm_impl<typename T::Operand, typename T::Operand, typename T::Output,
                  typename T::Output, typename T::Scale, float,
                  MakeTile<Kind, Mode, CandidateTiles<Kind>::tiles[I]>>(
            params, shape.sm_count, stream);
    }
}

template <MmaKind Kind, KernelMode Mode, std::size_t... I>
constexpr std::array<TileLauncher, sizeof...(I)> MakeTileLaunchers(
    std::index_sequence<I...>) {
    return {&LaunchTile<Kind, Mode, I>...};
}

template <MmaKind Kind, KernelMode Mode>
inline constexpr auto kTileLaunchers = MakeTileLaunchers<Kind, Mode>(
    std::make_index_sequence<CandidateTiles<Kind>::tiles.size()>{});

inline bool GemmTraceEnabled() {
    static const bool on = std::getenv("IR_GEMM_TRACE") != nullptr;
    return on;
}

template <KernelMode Mode, MmaKind Kind>
void LaunchCfg(GemmParams& params, const GemmShape& shape,
               cudaStream_t stream) {
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
                     "bias=%d -> %dx%d/%d pair=%d swap=%d regs=%d cfg=%d\n",
                     ModeName(Mode), MmaTraits<Kind>::kName, shape.m, shape.n,
                     shape.k, shape.m_grid, shape.sm_count,
                     shape.has_bias ? 1 : 0, t.block_m, t.block_n, t.block_k,
                     t.pair() ? 1 : 0, t.swap_ab ? 1 : 0, t.epi_regs, pick);
    }
    kTileLaunchers<Kind, Mode>[pick](params, shape, stream);
}

}  // namespace halops
