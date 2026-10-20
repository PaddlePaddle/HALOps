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

#include <array>
#include <utility>

#include "kernel/gemm/gemm_tile.cuh"
#include "kernel/gemm/gemm_utils.h"  // TORCH_CHECK

namespace halops {

namespace {

template <class Tile, bool Heuristic, bool Packed = false>
struct W8A8Tile : Tile {
    using Config = Tile;
    static constexpr bool kHeuristic = Heuristic;
    static constexpr bool kHeuristicPacked = Packed;
};

using W8A8Tiles = cute::tuple<
    W8A8Tile<GemmKernelConfigNaiveNoBias128x256x128_1CTA, true>,
    W8A8Tile<GemmKernelConfigNaiveNoBias256x256x128_2CTA, true>,
    W8A8Tile<GemmKernelConfigNaiveNoBias256x128x128_2CTA, true>,
    W8A8Tile<GemmKernelConfigNaiveNoBias128x128x128_1CTA, true>,
    W8A8Tile<GemmKernelConfigNaiveNoBias256x192x128_2CTA, true>,
    W8A8Tile<GemmKernelConfigNaiveNoBias128x64x256_SwapAB_1CTA, true>,
    W8A8Tile<GemmKernelConfigNaiveNoBias256x192x128_2CTA_Reg160, false>,
    W8A8Tile<
        GemmKernelConfigNaiveNoBias256x224x128_2CTA_PackedSfb_Reg160_Stage7,
        false, true>,
    W8A8Tile<GemmKernelConfig128x224x128_1CTA_PackedSfb, false, true>,
    W8A8Tile<GemmKernelConfigNaiveNoBias128x64x128_1CTA, true>,
    W8A8Tile<GemmKernelConfigNaiveNoBias256x64x128_2CTA, true>,
    W8A8Tile<
        GemmKernelConfigNaiveNoBias256x224x128_2CTA_PackedSfb_Reg160_Stage7_Swizzle,
        false, true>>;
constexpr int kCfgCount = static_cast<int>(cute::tuple_size_v<W8A8Tiles>);

// Index of a tile in the catalog; the dispatch names tiles by type so that
// reordering the list cannot silently re-point an id.
template <class T, int I = 0>
constexpr int W8A8TileIndex() {
    if constexpr (I < kCfgCount) {
        if constexpr (cute::is_same<
                          typename cute::tuple_element_t<I, W8A8Tiles>::Config,
                          T>::value) {
            return I;
        } else {
            return W8A8TileIndex<T, I + 1>();
        }
    } else {
        static_assert(I < 0, "tile is not in W8A8Tiles");
        return -1;
    }
}

struct DenseCfgDesc {
    int bm;  // activation-side tile extent
    int bn;  // weight-side tile extent
    int bk;
    int pair;      // 1 for pair-UMMA (2 CTAs per MMA tile)
    int cn;        // MMA tiles per cluster along N (A is multicast across them)
    bool overlap;  // accumulator stages share TMEM columns (KTmemOverlapCols)
    bool heuristic;  // eligible for SelectDenseCfg (else probe-only)
};

template <class T>
constexpr DenseCfgDesc MakeDenseDesc() {
    // SwapAB puts the activation in the UMMA N dim, so bm/bn follow operand
    // roles, not KBlockM/KBlockN.
    return {T::SwapAB ? T::KBlockN : T::KBlockM,
            T::SwapAB ? T::KBlockM : T::KBlockN,
            T::KBlockK,
            T::IsPairUmma ? 1 : 0,
            T::KClusterN,
            T::KTmemOverlapCols != 0,
            T::kHeuristic};
}

template <class Tuple, std::size_t... I>
constexpr std::array<DenseCfgDesc, sizeof...(I)> MakeDenseTable(
    std::index_sequence<I...>) {
    return {{MakeDenseDesc<cute::tuple_element_t<I, Tuple>>()...}};
}

// Tiles the heuristic may pick only when the caller supplies packed SFB.
template <class Tuple, std::size_t... I>
constexpr std::array<bool, sizeof...(I)> MakePackedFlags(
    std::index_sequence<I...>) {
    return {{cute::tuple_element_t<I, Tuple>::kHeuristicPacked...}};
}

constexpr auto kDenseCfgs =
    MakeDenseTable<W8A8Tiles>(std::make_index_sequence<kCfgCount>{});
constexpr auto kDenseCfgHeuristicPacked =
    MakePackedFlags<W8A8Tiles>(std::make_index_sequence<kCfgCount>{});

static_assert(
    kDenseCfgs
            [W8A8TileIndex<GemmKernelConfigNaiveNoBias128x64x256_SwapAB_1CTA>()]
                .bm == 64 &&
    kDenseCfgs
            [W8A8TileIndex<GemmKernelConfigNaiveNoBias128x64x256_SwapAB_1CTA>()]
                .bk == 256);
static_assert(
    kDenseCfgs
        [W8A8TileIndex<
             GemmKernelConfigNaiveNoBias256x224x128_2CTA_PackedSfb_Reg160_Stage7>()]
            .bn == 224);

// W4A4 dense tile table.
using W4A4Nvfp4Tiles =
    cute::tuple<GemmKernelConfigW4A4NaiveNoBias128x256x256_1CTA,          // 0
                GemmKernelConfigW4A4NaiveNoBias128x256x256_1CTA_Reg160,   // 1
                GemmKernelConfigW4A4NaiveNoBias128x64x512_1CTA,           // 2
                GemmKernelConfigW4A4NaiveNoBias128x64x512_1CTA_Reg160,    // 3
                GemmKernelConfigW4A4NaiveNoBiasSwA128x32x256_S9,          // 4
                GemmKernelConfigW4A4NaiveNoBiasSwA128x16x256_S10_Reg160,  // 5
                GemmKernelConfigW4A4NaiveNoBias256x128x256_2CTA_S8_E3,    // 6
                GemmKernelConfigW4A4NaiveNoBiasSwA128x128x256_S6,         // 7
                GemmKernelConfigW4A4NaiveNoBias256x256x256_2CTA_Reg160>;  // 8

// The MXFP4 family: same geometry and pipeline, SFVecSize 32 (one UE8M0 scale
// per 32 elements) and no per-tensor scale.
template <class T>
struct Mxfp4Twin : T {
    static constexpr int KSfVecSize = 32;
    static constexpr bool HasAlpha = false;
};
template <class... Ts>
auto W4A4TwinsOf(cute::tuple<Ts...>*) -> cute::tuple<Mxfp4Twin<Ts>...>;
template <class... A, class... B>
auto W4A4Cat(cute::tuple<A...>*, cute::tuple<B...>*) -> cute::tuple<A..., B...>;

using W4A4Mxfp4Tiles =
    decltype(W4A4TwinsOf(static_cast<W4A4Nvfp4Tiles*>(nullptr)));
using W4A4Tiles = decltype(W4A4Cat(static_cast<W4A4Nvfp4Tiles*>(nullptr),
                                   static_cast<W4A4Mxfp4Tiles*>(nullptr)));
constexpr int kW4A4CfgCount = static_cast<int>(cute::tuple_size_v<W4A4Tiles>);
static_assert(kW4A4CfgCount % 2 == 0,
              "the two W4A4 families must be the same length");

// Index of a tile in the NVFP4 list (by type, as for W8A8TileIndex).
template <class T, class Tuple = W4A4Nvfp4Tiles, int I = 0>
constexpr int W4A4TileIndex() {
    if constexpr (I < static_cast<int>(cute::tuple_size_v<Tuple>)) {
        if constexpr (cute::is_same<cute::tuple_element_t<I, Tuple>,
                                    T>::value) {
            return I;
        } else {
            return W4A4TileIndex<T, Tuple, I + 1>();
        }
    } else {
        static_assert(I < 0, "tile is not in W4A4Nvfp4Tiles");
        return -1;
    }
}

// W4A4 descriptor table, generated from W4A4Tiles.
struct W4A4CfgDesc {
    int act_m;
    int wt_n;
    int bk;
    int ctas;
    bool swap;
    bool mxfp4;
};

template <class T>
constexpr W4A4CfgDesc MakeW4A4Desc() {
    return {T::SwapAB ? T::KBlockN : T::KBlockM,
            T::SwapAB ? T::KBlockM : T::KBlockN,
            T::KBlockK,
            T::IsPairUmma ? 2 : 1,
            T::SwapAB,
            T::KSfVecSize == 32};
}

template <class Tuple, std::size_t... I>
constexpr std::array<W4A4CfgDesc, sizeof...(I)> MakeW4A4Table(
    std::index_sequence<I...>) {
    return {{MakeW4A4Desc<cute::tuple_element_t<I, Tuple>>()...}};
}

constexpr auto kW4A4Cfgs =
    MakeW4A4Table<W4A4Tiles>(std::make_index_sequence<kW4A4CfgCount>{});
bool W4A4CfgDivides(const W4A4CfgDesc& c, int m, int n, int k) {
    // A SwapAB tile may also cover an activation shorter than its extent.
    const bool m_ok =
        c.swap ? (m <= c.act_m || m % c.act_m == 0) : (m % c.act_m == 0);
    return m_ok && n % c.wt_n == 0 && k % c.bk == 0;
}

constexpr int CfgCtasPerTile(const DenseCfgDesc& c) { return c.pair ? 2 : 1; }

// Tiles whose N is not a multiple of 128 (e.g. 192) overhang N on the last
// tile.
bool CfgPartialN(const DenseCfgDesc& c) { return c.bn % 128 != 0; }

bool CfgDivides(const DenseCfgDesc& c, int m, int n, int k) {
    const bool n_ok =
        CfgPartialN(c) ? (n % 128 == 0 && n >= 384) : n % (c.bn * c.cn) == 0;

    const bool m_ok = (m % c.bm == 0) || (!c.pair && m <= c.bm);
    return m_ok && n_ok && k % c.bk == 0;
}

int CfgTilesN(const DenseCfgDesc& c, int n) { return (n + c.bn - 1) / c.bn; }

constexpr double kTileWeightMacs = 40.0;
constexpr double kTileStageMacs = 5.0e5;
constexpr double kPairTileMacs = 5.5e6;
constexpr double kTmemOverlapPenalty = 1.05;
constexpr double kPackedSfbPenalty = 1.01;

double PairUmmaEfficiency(const DenseCfgDesc& c, int waves) {
    if (!c.pair) {
        return 1.0;
    }
    return waves <= 1 ? 1.0 : 0.95;
}

int SelectDenseCfg(int m, int n, int k, int sm_count, bool has_packed_sfb) {
    int best = -1;
    double best_cost = 0.0;
    double best_area = 0.0;
    int best_ctas = 0;
    for (int i = 0; i < kCfgCount; ++i) {
        const DenseCfgDesc& c = kDenseCfgs[i];
        const bool packed = kDenseCfgHeuristicPacked[i];

        const bool swap_ab =
            i ==
            W8A8TileIndex<GemmKernelConfigNaiveNoBias128x64x256_SwapAB_1CTA>();
        if (swap_ab && m > 2 * c.bm) {
            continue;
        }

        const bool narrow_weight =
            i == W8A8TileIndex<GemmKernelConfigNaiveNoBias128x64x128_1CTA>();
        if (narrow_weight && m > 4 * c.bm) {
            continue;
        }
        if (!(c.heuristic || (packed && has_packed_sfb)) ||
            !CfgDivides(c, m, n, k)) {
            continue;
        }
        const int ctas = CfgCtasPerTile(c);
        const int units = sm_count / ctas;
        if (units <= 0) {
            continue;
        }
        const int tiles = ((m + c.bm - 1) / c.bm) * CfgTilesN(c, n);
        const int waves = (tiles + units - 1) / units;
        const double area = static_cast<double>(c.bm / ctas) * c.bn;
        // Pair tiles multicast the weight, so each CTA pulls half from L2.
        const double weight_bytes = static_cast<double>(c.bn) * k / ctas;
        const double tile_cost =
            area * k + weight_bytes * kTileWeightMacs +
            static_cast<double>(k / c.bk) * kTileStageMacs +
            (c.pair ? kPairTileMacs : 0.0);
        const double cost = waves * tile_cost * PairUmmaEfficiency(c, waves) *
                            (c.overlap ? kTmemOverlapPenalty : 1.0) *
                            (packed ? kPackedSfbPenalty : 1.0);

        constexpr double kCostTieBand = 3e-3;
        const bool better =
            best < 0 || cost < best_cost * (1.0 - kCostTieBand) ||
            (cost <= best_cost * (1.0 + kCostTieBand) &&
             (area > best_area || (area == best_area && ctas > best_ctas)));
        if (better) {
            best = i;
            best_cost = cost;
            best_area = area;
            best_ctas = ctas;
        }
    }
    return best;
}

int SelectW4A4Cfg(int m, int n, int k, int sm_count, bool mxfp4) {
    constexpr int kFamily = kW4A4CfgCount / 2;
    const int off = mxfp4 ? kFamily : 0;
    auto pick = [&](int nvfp4_id) { return nvfp4_id + off; };

    if (m > 64 && m <= 128 && n >= 16384 && m % 128 == 0) {
        return pick(
            W4A4TileIndex<GemmKernelConfigW4A4NaiveNoBias128x256x256_1CTA>());
    }

    if (m >= 4096 && n <= 4096) {
        const int units = sm_count / 2;
        const int tiles_256 = (m / 256) * ((n + 255) / 256);
        if (units <= 0 || tiles_256 <= 4 * units) {
            return pick(
                W4A4TileIndex<
                    GemmKernelConfigW4A4NaiveNoBias256x128x256_2CTA_S8_E3>());
        }
    }
    if (m <= 16) {
        return pick(W4A4TileIndex<
                    GemmKernelConfigW4A4NaiveNoBiasSwA128x16x256_S10_Reg160>());
    }
    if (m <= 32) {
        return pick(
            W4A4TileIndex<GemmKernelConfigW4A4NaiveNoBiasSwA128x32x256_S9>());
    }
    if (m <= 64) {
        return pick(W4A4TileIndex<
                    GemmKernelConfigW4A4NaiveNoBias128x64x512_1CTA_Reg160>());
    }
    if (m <= 128) {
        if (m % 64 == 0) {
            return pick(
                n >= 8192
                    ? W4A4TileIndex<
                          GemmKernelConfigW4A4NaiveNoBias128x64x512_1CTA_Reg160>()
                    : W4A4TileIndex<
                          GemmKernelConfigW4A4NaiveNoBias128x64x512_1CTA>());
        }
        return pick(
            W4A4TileIndex<GemmKernelConfigW4A4NaiveNoBiasSwA128x128x256_S6>());
    }
    if (m <= 2048) {
        if (n >= 1536 && n <= 4096) {
            return pick(
                W4A4TileIndex<
                    GemmKernelConfigW4A4NaiveNoBias256x128x256_2CTA_S8_E3>());
        }
        return pick(
            n <= 8192
                ? W4A4TileIndex<
                      GemmKernelConfigW4A4NaiveNoBias128x256x256_1CTA_Reg160>()
                : W4A4TileIndex<
                      GemmKernelConfigW4A4NaiveNoBias256x256x256_2CTA_Reg160>());
    }
    return pick(W4A4TileIndex<
                GemmKernelConfigW4A4NaiveNoBias256x256x256_2CTA_Reg160>());
}

template <class Entry>
struct CatalogTile {
    using type = Entry;
};
template <class Tile, bool Heuristic, bool Packed>
struct CatalogTile<W8A8Tile<Tile, Heuristic, Packed>> {
    using type = Tile;
};
template <class Entry>
using CatalogTileT = typename CatalogTile<Entry>::type;

template <class Catalog, class LaunchTile>
void DispatchCfgAt(int cfg, const char* what, LaunchTile&& launch) {
    constexpr int kCount = static_cast<int>(cute::tuple_size_v<Catalog>);
    // One branch per catalog entry, generated: no case labels to keep in sync
    // with the table.
    const bool found = [&]<int... Is>(std::integer_sequence<int, Is...>) {
        return ((cfg == Is
                     ? (launch.template operator()<
                            CatalogTileT<cute::tuple_element_t<Is, Catalog>>>(),
                        true)
                     : false) ||
                ...);
    }(std::make_integer_sequence<int, kCount>{});
    TORCH_CHECK(found, "unknown ", what, " cfg id ", cfg);
}

template <class LaunchTile>
void DispatchW8A8Cfg(int cfg, LaunchTile&& launch) {
    DispatchCfgAt<W8A8Tiles>(cfg, "w8a8", launch);
}

template <class LaunchTile>
void DispatchW4A4Cfg(int cfg, LaunchTile&& launch) {
    DispatchCfgAt<W4A4Tiles>(cfg, "w4a4", launch);
}

}  // namespace

}  // namespace halops
