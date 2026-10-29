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

// ===========================================================================
// BF16 dense/grouped/batch dispatch: the shape-aware ladder moved verbatim
// from the DenseGemm/GroupContiguousGemm/GroupMaskedGemm/BatchGemm host
// functions. Unlike the W8A8 cost model (SelectDenseCfg above), the BF16
// ladder is rule-based (pair-UMMA CTA bands, stream-K, k-tile depth gates,
// wave-fill judgements) -- the same shape as SelectW4A4Cfg, just longer.
// Selectors return a catalog id; DispatchBF16*Cfg maps ids to instantiations.
// ===========================================================================

#pragma once

#include <array>
#include <cstdlib>
#include <string>
#include <utility>

#include "kernel/gemm/gemm_tile.cuh"
#include "kernel/gemm/gemm_utils.h"

namespace halops {
namespace bf16_dispatch {

// ---------------------------------------------------------------------------
// Shared helpers (verbatim from the old dense_gemm.cu anonymous namespace).
// ---------------------------------------------------------------------------

constexpr int64_t kPairDeepMinCtas = 40;
constexpr int64_t kPairDeepMaxCtas = 152;

constexpr int64_t CeilDiv(int64_t x, int64_t d) { return (x + d - 1) / d; }

// KBlockK of every BF16 config below.
constexpr int64_t kBF16BlockK = 64;

// Deep-prologue configs (14/18 stages) only help with at least this many
// K-tiles.
constexpr int64_t kDeepPrologueMinKTiles = 32;

inline bool DeepProloguePays(int k) {
    return k / kBF16BlockK >= kDeepPrologueMinKTiles;
}

inline int64_t PairDeepCtaCount(int m, int n) {
    return 2 * CeilDiv(m, 128) * CeilDiv(n, 64);
}

inline void CheckPairDeepCtaBand(int m, int n, int cluster4_arm) {
    // HALOPS_PAIR_UNSAFE=1 lifts the guard (debugging only).
    static const bool unsafe_ok = [] {
        const char* e = std::getenv("HALOPS_PAIR_UNSAFE");
        return e != nullptr && e[0] == '1';
    }();
    if (unsafe_ok) {
        return;
    }
    const int64_t ctas = PairDeepCtaCount(m, n);
    TORCH_CHECK(ctas >= kPairDeepMinCtas && ctas <= kPairDeepMaxCtas,
                "HALOPS_CLUSTER4=", cluster4_arm,
                " selects the 2SM/KBlockM=128 pair-UMMA family, which is still "
                "numerically WRONG outside a CTA count of [",
                kPairDeepMinCtas, ", ", kPairDeepMaxCtas,
                "]. This shape (m=", m, ", n=", n, ") gives ", ctas,
                " CTAs; refusing rather than "
                "returning silently wrong results.");
}

// For m < 128: use a non-SwapAB KBlockM=64 tile when even SwapAB's smallest
// tile leaves SMs idle. HALOPS_M64_NOSWAP=0 always uses SwapAB.
inline bool ThinMNoSwapOk(int m, int n, int k, int sm_count) {
    static const bool disabled = [] {
        const char* e = std::getenv("HALOPS_M64_NOSWAP");
        return e != nullptr && e[0] == '0';
    }();
    if (disabled) {
        return false;
    }
    if (m % 64 != 0 || k % 64 != 0) {
        return false;
    }
    const int target = ResolveSmCount(sm_count);
    const int64_t nt32 = CeilDiv(n, 32);
    const int64_t mt = CeilDiv(m, 64);
    const bool narrow_ok = (n % 32 == 0) && (nt32 * mt <= target);
    const bool square_ok = (n % 64 == 0);
    if (!narrow_ok && !square_ok) {
        return false;
    }
    const int64_t swapab_tiles = CeilDiv(n, 128) * CeilDiv(m, 32);
    return swapab_tiles < target;
}

// Above this many K-tiles keep 256x256: 256x128 re-reads A twice as often,
// which outweighs any wave-packing gain.
constexpr int64_t kPairFineNMaxKTiles = 64;

// Minimum K-tiles for the deep-prologue mid-M configs (64x96 P11, 128x64 pair
// P18).
constexpr int64_t kMidMDeepPrologueMinKTiles = 54;

// True if fine_ctas keeps a larger fraction of SMs busy than coarse_ctas,
// where busy fraction = ctas / (target * number_of_waves).
inline bool PacksTailTighter(int64_t coarse_ctas, int64_t fine_ctas,
                             int64_t target) {
    const int64_t coarse_waves = (coarse_ctas + target - 1) / target;
    const int64_t fine_waves = (fine_ctas + target - 1) / target;
    return fine_ctas * coarse_waves > coarse_ctas * fine_waves;
}

// HALOPS_PAIR_FINE_N=0 disables PreferFinePairN at both call sites.
inline bool PairFineNDisabled() {
    static const bool disabled = [] {
        const char* e = std::getenv("HALOPS_PAIR_FINE_N");
        return e != nullptr && e[0] == '0';
    }();
    return disabled;
}

// Use 256x128 instead of 256x256 for the pair-UMMA tile when K is short and
// 256x128 keeps more SMs busy; beyond 4 waves the gain is negligible.
inline bool PreferFinePairN(int n, int k, int64_t coarse_ctas,
                            int64_t fine_ctas, int64_t target) {
    return (n % 128 == 0) && (k / 64 <= kPairFineNMaxKTiles) &&
           (coarse_ctas <= 4 * target) &&
           PacksTailTighter(coarse_ctas, fine_ctas, target);
}

// ---- Small-N tiles (N <= 128) ----------------------------------------------
// N fits in a single tile, so all CTAs must come from splitting M. Pick the
// largest tile whose CTA count is within (0.5, 1.0] of one wave.
// HALOPS_SMALL_N=0 disables this.
inline bool SmallNLadderDisabled() {
    static const bool disabled = [] {
        const char* e = std::getenv("HALOPS_SMALL_N");
        return e != nullptr && e[0] == '0';
    }();
    return disabled;
}

enum class SmallNTile {
    kNone,
    kPairM256N128,
    kM64N128,
    kPairM256N64,
    kM64N64,
};

constexpr int64_t kSmallNMaxN = 128;
constexpr int64_t kSmallNAlignN = 64;

inline bool FillsOneWave(int64_t ctas, int64_t target) {
    return ctas <= target && 2 * ctas > target;
}

struct SmallNRung {
    SmallNTile tile;
    int64_t block_m;
    int64_t block_n;
    int64_t ctas_per_cluster;
};

inline SmallNTile PickSmallNTile(int m, int n, int64_t target) {
    if (SmallNLadderDisabled() || n > kSmallNMaxN || n % kSmallNAlignN != 0) {
        return SmallNTile::kNone;
    }
    static constexpr SmallNRung kRungs[] = {
        {SmallNTile::kPairM256N128, 256, 128, 2},
        {SmallNTile::kM64N128, 64, 128, 1},
        {SmallNTile::kPairM256N64, 256, 64, 2},
        {SmallNTile::kM64N64, 64, 64, 1},
    };
    for (const SmallNRung& r : kRungs) {
        const int64_t ctas =
            r.ctas_per_cluster * CeilDiv(m, r.block_m) * CeilDiv(n, r.block_n);
        if (FillsOneWave(ctas, target)) {
            return r.tile;
        }
    }
    // No tile fits; use the regular large-M dispatch.
    return SmallNTile::kNone;
}

// ---------------------------------------------------------------------------
// Environment-variable knobs, read once per process (static locals).
// ---------------------------------------------------------------------------

inline bool EnvIs(const char* name, const char* value) {
    const char* e = std::getenv(name);
    return e != nullptr && e[0] == value[0];
}

inline bool EnvNot0(const char* name) {
    const char* e = std::getenv(name);
    return e == nullptr || e[0] != '0';
}

inline int EnvIntOr(const char* name, int fallback) {
    const char* e = std::getenv(name);
    return e != nullptr ? atoi(e) : fallback;
}

inline int64_t EnvInt64Or(const char* name, int64_t fallback) {
    const char* e = std::getenv(name);
    return e != nullptr ? static_cast<int64_t>(atoll(e)) : fallback;
}

// ---------------------------------------------------------------------------
// Dense (Naive) catalog. Order defines the id; ids are only meaningful
// together with DispatchBF16Cfg below (both index the same tuple).
// ---------------------------------------------------------------------------

using BF16Tiles =
    cute::tuple<GemmKernelConfigBF16NaiveNoBias256x256x64_2CTA,           // 0
                GemmKernelConfigBF16NaiveNoBias256x256x64_2CTA_P7,        // 1
                GemmKernelConfigBF16NaiveNoBias256x128x64_2CTA,           // 2
                GemmKernelConfigBF16NaiveNoBias256x128x64_2CTA_Ntail,     // 3
                GemmKernelConfigBF16NaiveNoBias256x128x64_2CTA_P9,        // 4
                GemmKernelConfigBF16NaiveNoBias256x128x64_2CTA_P9_Ntail,  // 5
                GemmKernelConfigBF16NaiveNoBias256x192x64_2CTA,           // 6
                GemmKernelConfigBF16NaiveNoBias256x64x64_2CTA,            // 7
                GemmKernelConfigBF16NaiveNoBias128x256x64_1CTA,           // 8
                GemmKernelConfigBF16NaiveNoBias128x128x64_1CTA,           // 9
                GemmKernelConfigBF16NaiveNoBias128x64x64_1CTA,            // 10
                GemmKernelConfigBF16NaiveNoBias128x32x64_1CTA,            // 11
                GemmKernelConfigBF16NaiveNoBias128x32x32_1CTA,            // 12
                GemmKernelConfigBF16NaiveNoBias128x32x64_P4_1CTA,         // 13
                GemmKernelConfigBF16NaiveNoBias128x32x64_P6_1CTA,         // 14
                GemmKernelConfigBF16NaiveNoBias128x64x64_2CTA_P18,        // 15
                GemmKernelConfigBF16NaiveNoBias128x64x64_2x1CTA,          // 16
                GemmKernelConfigBF16NaiveNoBias128x64x64_2x2CTA,          // 17
                GemmKernelConfigBF16NaiveNoBias128x64x64_2x2CTA_P16,      // 18
                GemmKernelConfigBF16NaiveNoBias128x64x64_2x2CTA_P18,      // 19
                GemmKernelConfigBF16NaiveNoBias64x128x64_1CTA,            // 20
                GemmKernelConfigBF16NaiveNoBias64x96x64_1CTA_P11,         // 21
                GemmKernelConfigBF16NaiveNoBias64x224x64_1CTA_P6,         // 22
                GemmKernelConfigBF16NaiveNoBias64x64x64_1CTA,             // 23
                GemmKernelConfigBF16NaiveNoBias64x64x64_P12_1CTA,         // 24
                GemmKernelConfigBF16NaiveNoBias64x64x64_P14_1CTA,         // 25
                GemmKernelConfigBF16NaiveNoBias64x64x64_1WG_1CTA,         // 26
                GemmKernelConfigBF16NaiveNoBias64x64x64_1WG_P14_1CTA,     // 27
                GemmKernelConfigBF16NaiveNoBias64x32x64_1WG_1CTA,         // 28
                GemmKernelConfigBF16NaiveNoBias64x32x64_1WG_P14_1CTA,     // 29
                GemmKernelConfigBF16NaiveNoBias64x32x64_1WG_P18_1CTA,     // 30
                GemmKernelConfigBF16NaiveNoBias128x64x64_SwapAB_1CTA,     // 31
                GemmKernelConfigBF16NaiveNoBias128x32x64_SwapAB_1CTA,     // 32
                GemmKernelConfigBF16NaiveNoBias256x64x64_SwapAB_2CTA,     // 33
                GemmKernelConfigBF16NaiveNoBias128x64x128_SwapAB_1CTA>;   // 34
constexpr int kBF16CfgCount = static_cast<int>(cute::tuple_size_v<BF16Tiles>);

// Index of a tile in the catalog (dispatch names tiles by type; reordering
// the list cannot silently re-point an id).
template <class T, int I = 0>
constexpr int BF16TileIndex() {
    if constexpr (I < kBF16CfgCount) {
        if constexpr (cute::is_same<cute::tuple_element_t<I, BF16Tiles>,
                                    T>::value) {
            return I;
        } else {
            return BF16TileIndex<T, I + 1>();
        }
    } else {
        static_assert(I < 0, "tile is not in BF16Tiles");
        return -1;
    }
}

// Shape descriptor for host-side divisibility checks after selection.
struct BF16CfgDesc {
    int bm;      // KBlockM (for SwapAB: the weight M)
    int bn;      // KBlockN (for SwapAB: the activation extent)
    int bk;      // KBlockK
    bool swap;   // SwapAB
    bool ntail;  // KForceEpiNPredicate: N only needs % 64 (or % 32 for
                 // non-power-of-2 KBlockN)
};

template <class T>
constexpr BF16CfgDesc MakeBF16Desc() {
    return {T::KBlockM, T::KBlockN, T::KBlockK, T::SwapAB,
            T::KForceEpiNPredicate};
}

template <class Tuple, std::size_t... I>
constexpr std::array<BF16CfgDesc, sizeof...(I)> MakeBF16Table(
    std::index_sequence<I...>) {
    return {{MakeBF16Desc<cute::tuple_element_t<I, Tuple>>()...}};
}

constexpr auto kBF16Cfgs =
    MakeBF16Table<BF16Tiles>(std::make_index_sequence<kBF16CfgCount>{});

// ---------------------------------------------------------------------------
// SelectBF16Cfg: the dense ladder, moved verbatim from dense_gemm.cu.
// Returns a BF16Tiles catalog id or -1 when no rung supports the shape.
// The caller validates divisibility against kBF16Cfgs[id] and launches via
// DispatchBF16Cfg.
// ---------------------------------------------------------------------------

namespace dense {

using bf16_dispatch::CeilDiv;
using bf16_dispatch::CheckPairDeepCtaBand;
using bf16_dispatch::DeepProloguePays;
using bf16_dispatch::EnvInt64Or;
using bf16_dispatch::EnvIntOr;
using bf16_dispatch::EnvIs;
using bf16_dispatch::EnvNot0;
using bf16_dispatch::kBF16BlockK;
using bf16_dispatch::kMidMDeepPrologueMinKTiles;
using bf16_dispatch::PacksTailTighter;
using bf16_dispatch::PairDeepCtaCount;
using bf16_dispatch::PairFineNDisabled;
using bf16_dispatch::PickSmallNTile;
using bf16_dispatch::PreferFinePairN;
using bf16_dispatch::SmallNTile;
using bf16_dispatch::ThinMNoSwapOk;

inline int SelectBF16Cfg(int m, int n, int k, int sm_count) {
    if (m >= 4096) {
        // Large-M: 256x256 pair tile by default, or 256x128 (deeper
        // pipeline) when PreferFinePairN says so. If n % 256 != 0, fall
        // back to 256x128, then _Ntail / 256x64; n % 64 != 0 is rejected.
        const int target_sm = ResolveSmCount(sm_count);
        switch (PickSmallNTile(m, n, target_sm)) {
            case SmallNTile::kM64N128:
                return bf16_dispatch::BF16TileIndex<
                    GemmKernelConfigBF16NaiveNoBias64x128x64_1CTA>();
            case SmallNTile::kM64N64: {
                // HALOPS_SMALLN_1WG_DEEP=0 restores 2WG/10 stages.
                static const bool smalln_1wg_deep =
                    EnvNot0("HALOPS_SMALLN_1WG_DEEP");
                if (smalln_1wg_deep && DeepProloguePays(k)) {
                    return bf16_dispatch::BF16TileIndex<
                        GemmKernelConfigBF16NaiveNoBias64x64x64_1WG_P14_1CTA>();
                }
                return bf16_dispatch::BF16TileIndex<
                    GemmKernelConfigBF16NaiveNoBias64x64x64_1CTA>();
            }
            case SmallNTile::kPairM256N64:
                return bf16_dispatch::BF16TileIndex<
                    GemmKernelConfigBF16NaiveNoBias256x64x64_2CTA>();
            // Handled by the default dispatch below.
            case SmallNTile::kPairM256N128:
            case SmallNTile::kNone:
                break;
        }
        const int64_t tiles_256 = CeilDiv(m, 256) * CeilDiv(n, 256);
        const int64_t mt_256 = CeilDiv(m, 256);
        const bool square_like =
            PairFineNDisabled()
                ? ((n % 128 == 0) && (tiles_256 <= 2 * target_sm))
                : PreferFinePairN(n, k, 2 * tiles_256,
                                  2 * mt_256 * CeilDiv(n, 128), target_sm);
        // HALOPS_LARGEM_DEEP=0 restores the shallower P6/P8 configs.
        static const bool largem_deep = EnvNot0("HALOPS_LARGEM_DEEP");
        if (square_like) {
            return largem_deep
                       ? bf16_dispatch::BF16TileIndex<
                             GemmKernelConfigBF16NaiveNoBias256x128x64_2CTA_P9>()
                       : bf16_dispatch::BF16TileIndex<
                             GemmKernelConfigBF16NaiveNoBias256x128x64_2CTA>();
        }
        if (n % 256 == 0) {
            return largem_deep
                       ? bf16_dispatch::BF16TileIndex<
                             GemmKernelConfigBF16NaiveNoBias256x256x64_2CTA_P7>()
                       : bf16_dispatch::BF16TileIndex<
                             GemmKernelConfigBF16NaiveNoBias256x256x64_2CTA>();
        }
        if (n % 128 == 0) {
            return bf16_dispatch::BF16TileIndex<
                GemmKernelConfigBF16NaiveNoBias256x128x64_2CTA>();
        }
        // n % 128 != 0 (e.g. 1600): 128-wide _Ntail tile, which masks
        // the store of the partial last tile. HALOPS_LARGEM_NTAIL=0
        // uses 256x64 instead.
        static const bool largem_ntail_disabled =
            EnvIs("HALOPS_LARGEM_NTAIL", "0");
        if (!largem_ntail_disabled && (n % 64 == 0)) {
            return bf16_dispatch::BF16TileIndex<
                GemmKernelConfigBF16NaiveNoBias256x128x64_2CTA_Ntail>();
        }
        return bf16_dispatch::BF16TileIndex<
            GemmKernelConfigBF16NaiveNoBias256x64x64_2CTA>();
    }

    if (m >= 128) {
        // The narrowest mid-M tile is 128x32, so only n % 32 is needed.
        if (m % 128 != 0 || n % 32 != 0 || k % 64 != 0) {
            return -1;
        }

        // HALOPS_CLUSTER4=1..5 forces a clustered 128x64 pair variant
        // (debugging only; checked against the trusted CTA band).
        static const int cluster4 = EnvIntOr("HALOPS_CLUSTER4", 0);
        if (cluster4 != 0 && (m % 128 == 0) && (n % 128 == 0) &&
            (k % 64 == 0)) {
            CheckPairDeepCtaBand(m, n, cluster4);
            switch (cluster4) {
                case 1:
                    return bf16_dispatch::BF16TileIndex<
                        GemmKernelConfigBF16NaiveNoBias128x64x64_2x2CTA>();
                case 3:
                    return bf16_dispatch::BF16TileIndex<
                        GemmKernelConfigBF16NaiveNoBias128x64x64_2x2CTA_P16>();
                case 4:
                    return bf16_dispatch::BF16TileIndex<
                        GemmKernelConfigBF16NaiveNoBias128x64x64_2x2CTA_P18>();
                case 5:
                    return bf16_dispatch::BF16TileIndex<
                        GemmKernelConfigBF16NaiveNoBias128x64x64_2CTA_P18>();
                default:
                    return bf16_dispatch::BF16TileIndex<
                        GemmKernelConfigBF16NaiveNoBias128x64x64_2x1CTA>();
            }
        }

        // ---- KBlockM=64 tiles (64x64 / 64x32) --------------------------
        // Smaller tiles add CTAs, but with no multicast each extra N-tile
        // reloads A. Use them only if the grid fits one wave and the A
        // reload volume k_tiles * n_tiles is within rereads_cap.
        // HALOPS_TILE_M64=0 disables them.
        static const bool m64_disabled = EnvIs("HALOPS_TILE_M64", "0");
        if (!m64_disabled && (m % 64 == 0) && (n % 64 == 0) && (k % 64 == 0)) {
            const int sm = ResolveSmCount(sm_count);
            const int64_t tiles64 = CeilDiv(m, 64) * CeilDiv(n, 64);
            const int64_t rereads = (k / kBF16BlockK) * CeilDiv(n, 64);
            // HALOPS_REREADS_CAP=<n> overrides the re-read cap.
            static const int64_t rereads_cap =
                EnvInt64Or("HALOPS_REREADS_CAP", 2048);
            // HALOPS_FORCE_M64_N32=1 uses 64x32 regardless of both
            // limits.
            static const bool n32_force = EnvIs("HALOPS_FORCE_M64_N32", "1");
            if (n32_force || (tiles64 <= sm && rereads <= rereads_cap)) {
                // 1-epilogue-WG configs: required for 64x32 (KEpiComputeN
                // must be >= 32) and free of the inter-WG barrier. Use
                // 64x32 if twice the 64x64 tile count still fits one wave,
                // else 64x64. HALOPS_M64_1WG=0 uses the 2WG 64x64.
                static const bool wg1_disabled = EnvIs("HALOPS_M64_1WG", "0");
                if (!wg1_disabled) {
                    if ((n32_force || 2 * tiles64 <= sm) && (n % 32 == 0)) {
                        // HALOPS_N32_STAGES=10|14|18 forces the stage
                        // count.
                        static const int n32_stages =
                            EnvIntOr("HALOPS_N32_STAGES", 0);
                        const bool deep = n32_stages == 0 ? DeepProloguePays(k)
                                                          : n32_stages == 18;
                        if (n32_stages == 14) {
                            return bf16_dispatch::BF16TileIndex<
                                GemmKernelConfigBF16NaiveNoBias64x32x64_1WG_P14_1CTA>();
                        }
                        if (deep) {
                            return bf16_dispatch::BF16TileIndex<
                                GemmKernelConfigBF16NaiveNoBias64x32x64_1WG_P18_1CTA>();
                        }
                        return bf16_dispatch::BF16TileIndex<
                            GemmKernelConfigBF16NaiveNoBias64x32x64_1WG_1CTA>();
                    }
                    // HALOPS_M64_1WG_DEEP=0 restores 10 stages.
                    static const bool m64_1wg_deep =
                        EnvNot0("HALOPS_M64_1WG_DEEP");
                    if (m64_1wg_deep && DeepProloguePays(k)) {
                        return bf16_dispatch::BF16TileIndex<
                            GemmKernelConfigBF16NaiveNoBias64x64x64_1WG_P14_1CTA>();
                    }
                    return bf16_dispatch::BF16TileIndex<
                        GemmKernelConfigBF16NaiveNoBias64x64x64_1WG_1CTA>();
                }
                // HALOPS_M64_STAGES=12|14 uses a deeper 2WG 64x64.
                static const int m64_stages = EnvIntOr("HALOPS_M64_STAGES", 0);
                if (m64_stages == 12) {
                    return bf16_dispatch::BF16TileIndex<
                        GemmKernelConfigBF16NaiveNoBias64x64x64_P12_1CTA>();
                }
                if (m64_stages == 14) {
                    return bf16_dispatch::BF16TileIndex<
                        GemmKernelConfigBF16NaiveNoBias64x64x64_P14_1CTA>();
                }
                return bf16_dispatch::BF16TileIndex<
                    GemmKernelConfigBF16NaiveNoBias64x64x64_1CTA>();
            }
        }

        // ---- 128xN tiles ------------------------------------------------
        // Halve N (256 -> 128 -> 64 -> 32) while the current tile leaves
        // SMs idle.
        const int target = ResolveSmCount(sm_count);
        auto tiles = [&](int tm, int tn) {
            return ((m + tm - 1) / tm) * ((n + tn - 1) / tn);
        };
        // HALOPS_DISABLE_TILE_N32=1 stops at 128x64.
        static const bool n32_disabled = EnvIs("HALOPS_DISABLE_TILE_N32", "1");
        // pick: 0=128x256, 1=128x128, 2=128x64, 3=128x32.
        int pick;
        if (n % 256 == 0) {
            pick = 0;
        } else if (n % 128 == 0) {
            pick = 1;
        } else if (n % 64 == 0) {
            pick = 2;
        } else {
            pick = 3;
        }
        // Keep 128x256 if it already fills half a wave and 128x128 would
        // spill into a second wave.
        const bool overshoots_128 =
            m > 128 && tiles(128, 128) > target && 2 * tiles(128, 256) > target;
        if (pick == 0 && tiles(128, 256) < target && (n % 128 == 0) &&
            !overshoots_128) {
            pick = 1;  // 128x128
        }
        if (pick == 1 && tiles(128, 128) < target && (n % 64 == 0)) {
            pick = 2;  // 128x64
        }
        if (pick == 2 && !n32_disabled && tiles(128, 64) < target &&
            (n % 32 == 0)) {
            pick = 3;  // 128x32
        }

        // ---- 256xN pair-UMMA tiles (m % 256 == 0) ----------------------
        // Same CTA count as 128xN, but B is loaded half as often. Used
        // only if:
        //   1) k_tiles >= 3 * KPrologueStages, or the grid is >= 1 wave;
        //   2) for 256x64, the grid is >= 0.8 wave.
        // HALOPS_MIDM_PAIR=0 disables them.
        static const bool midm_pair = EnvNot0("HALOPS_MIDM_PAIR");
        if (midm_pair && (m % 256 == 0) && (n % 64 == 0) && (k % 64 == 0)) {
            auto pair_ctas = [&](int64_t tn) {
                return static_cast<int64_t>(m / 256) * ((n + tn - 1) / tn) * 2;
            };
            // Halve N only while the grid is under one wave, and never
            // from >= half a wave into more than one wave.
            auto step_ok = [&](int64_t coarse, int64_t fine) {
                return pair_ctas(coarse) < target &&
                       !(pair_ctas(fine) > target &&
                         2 * pair_ctas(coarse) > target);
            };
            int pair_tn = (n % 256 == 0) ? 256 : ((n % 128 == 0) ? 128 : 64);
            // n % 128 != 0 (e.g. 1600): use the 128-wide _Ntail tile
            // instead of 64 if it fills at least half a wave.
            // HALOPS_MIDM_NTAIL=0 disables it.
            static const bool midm_ntail = EnvNot0("HALOPS_MIDM_NTAIL");
            bool pair_n_tail = false;
            if (midm_ntail && pair_tn == 64 && (n % 128 != 0) &&
                2 * pair_ctas(128) >= target) {
                pair_tn = 128;
                pair_n_tail = true;
            }
            // Also use 128 when PreferFinePairN says so (multi-wave).
            if (pair_tn == 256 && (n % 128 == 0) &&
                (step_ok(256, 128) ||
                 (!PairFineNDisabled() &&
                  PreferFinePairN(n, k, pair_ctas(256), pair_ctas(128),
                                  target)))) {
                pair_tn = 128;
            }
            // Use 192 if it fits one wave and keeps more SMs busy than
            // both 256 and 128. HALOPS_MIDM_PAIR192=0 disables it.
            static const bool midm_pair192 = EnvNot0("HALOPS_MIDM_PAIR192");
            if (midm_pair192 && pair_tn != 64 && (n % 32 == 0) &&
                pair_ctas(192) <= target &&
                PacksTailTighter(pair_ctas(256), pair_ctas(192), target) &&
                PacksTailTighter(pair_ctas(128), pair_ctas(192), target)) {
                pair_tn = 192;
                pair_n_tail = false;
            }
            if (!pair_n_tail && pair_tn == 128 && (n % 64 == 0) &&
                step_ok(128, 64)) {
                pair_tn = 64;
            }
            const int64_t pair_cta_count = pair_ctas(pair_tn);
            const int64_t pair_k_tiles = k / 64;
            const int pair_stages =
                (pair_tn == 256)
                    ? GemmKernelConfigBF16NaiveNoBias256x256x64_2CTA::
                          KPrologueStages
                    : ((pair_tn == 192)
                           ? GemmKernelConfigBF16NaiveNoBias256x192x64_2CTA::
                                 KPrologueStages
                           : ((pair_tn == 128)
                                  ? GemmKernelConfigBF16NaiveNoBias256x128x64_2CTA::
                                        KPrologueStages
                                  : GemmKernelConfigBF16NaiveNoBias256x64x64_2CTA::
                                        KPrologueStages));
            const bool amortizes_prologue =
                pair_k_tiles >= 3 * pair_stages || pair_cta_count >= target;
            const bool narrow_arm_fills =
                pair_tn != 64 || 5 * pair_cta_count >= 4 * target;
            // If n % 128 != 0 forced 256x64 and it spills past one wave,
            // skip the pair tiles and let 64x96 / 64x224 below handle it.
            const int64_t narrow_ctas96 = CeilDiv(m, 64) * CeilDiv(n, 96);
            const int64_t narrow_ctas224 = CeilDiv(m, 64) * CeilDiv(n, 224);
            const bool ninety_six_legal =
                (m % 64 == 0) && (n % 32 == 0) &&
                (k / kBF16BlockK) >= kMidMDeepPrologueMinKTiles;
            const bool two_two_four_legal = (m % 64 == 0) && (n % 32 == 0);
            const bool narrow_arm_overshoots =
                pair_tn == 64 && pair_cta_count > target &&
                two_two_four_legal &&
                (!ninety_six_legal ||
                 PacksTailTighter(narrow_ctas224, narrow_ctas96, target));
            if (amortizes_prologue && narrow_arm_fills &&
                !narrow_arm_overshoots) {
                if (pair_tn == 256) {
                    return bf16_dispatch::BF16TileIndex<
                        GemmKernelConfigBF16NaiveNoBias256x256x64_2CTA>();
                }
                if (pair_tn == 192) {
                    return bf16_dispatch::BF16TileIndex<
                        GemmKernelConfigBF16NaiveNoBias256x192x64_2CTA>();
                }
                if (pair_tn == 128) {
                    // HALOPS_MIDM_PAIR_DEEP=0 restores P8.
                    static const bool midm_pair_deep =
                        EnvNot0("HALOPS_MIDM_PAIR_DEEP");
                    if (pair_n_tail) {
                        return bf16_dispatch::BF16TileIndex<
                            GemmKernelConfigBF16NaiveNoBias256x128x64_2CTA_P9_Ntail>();
                    }
                    return bf16_dispatch::BF16TileIndex<
                        GemmKernelConfigBF16NaiveNoBias256x128x64_2CTA_P9>();
                }
                // With <= 32 K-tiles, 128x64x64_1CTA (same CTA count) is
                // faster than 256x64 pair. HALOPS_MIDM_PAIR64_1CTA=0
                // disables it.
                static const bool midm_pair64_1cta =
                    EnvNot0("HALOPS_MIDM_PAIR64_1CTA");
                if (midm_pair64_1cta && pair_k_tiles <= 32) {
                    return bf16_dispatch::BF16TileIndex<
                        GemmKernelConfigBF16NaiveNoBias128x64x64_1CTA>();
                }
                return bf16_dispatch::BF16TileIndex<
                    GemmKernelConfigBF16NaiveNoBias256x64x64_2CTA>();
            }
        }

        // If the 128xN tile spills past one wave, use 64x128 when that
        // fits in one wave. HALOPS_DISABLE_WAVE64=1 disables it;
        // HALOPS_WAVE64_ANYCTA=1 drops the one-wave condition.
        static const bool wave64_disabled = EnvIs("HALOPS_DISABLE_WAVE64", "1");
        static const bool wave64_anycta = EnvIs("HALOPS_WAVE64_ANYCTA", "1");
        const int64_t pick_tn =
            pick == 0 ? 256 : (pick == 1 ? 128 : (pick == 2 ? 64 : 32));
        // ---- 64x96 / 64x224 tiles --------------------------------------
        // Power-of-2 N widths can miss one wave entirely (n=7168: 112 or
        // 224 CTAs on 152 SMs); 96/224 land in between.
        // HALOPS_DISABLE_TILE96 / _TILE224 disable them.
        static const bool tile96_disabled = EnvIs("HALOPS_DISABLE_TILE96", "1");
        static const bool tile224_disabled =
            EnvIs("HALOPS_DISABLE_TILE224", "1");
        auto ctas_at = [&](int64_t tn) {
            return CeilDiv(m, 64) * CeilDiv(n, tn);
        };
        const int64_t ctas96 = ctas_at(96);
        const int64_t ctas224 = ctas_at(224);
        const int64_t incumbent_ctas =
            (n % 128 == 0 && tiles(128, pick_tn) > target &&
             tiles(64, 128) <= target)
                ? tiles(64, 128)
                : tiles(128, pick_tn);
        // c fits in one wave at > 50%, and either the current choice
        // needs more than one wave or c gives more CTAs.
        auto fills_better = [&](int64_t c) {
            return c <= target && 2 * c > target &&
                   (incumbent_ctas > target || c > incumbent_ctas);
        };
        // The 256x64 pair tile above was skipped in favour of 64x96;
        // take 64x96 even though it exceeds one wave.
        const bool pair_deferred_to_96 =
            (m % 256 == 0) && (n % 64 == 0) && (n % 128 != 0) &&
            (k % 64 == 0) &&
            (static_cast<int64_t>(m / 256) * CeilDiv(n, 64) * 2 > target);
        if (!tile96_disabled && (m % 64 == 0) && (n % 32 == 0) &&
            (fills_better(ctas96) || pair_deferred_to_96) &&
            (k / kBF16BlockK) >= kMidMDeepPrologueMinKTiles) {
            return bf16_dispatch::BF16TileIndex<
                GemmKernelConfigBF16NaiveNoBias64x96x64_1CTA_P11>();
        }
        if (!tile224_disabled && (m % 64 == 0) && (n % 32 == 0) &&
            fills_better(ctas224)) {
            return bf16_dispatch::BF16TileIndex<
                GemmKernelConfigBF16NaiveNoBias64x224x64_1CTA_P6>();
        }

        if (!wave64_disabled && (m % 64 == 0) && (n % 128 == 0) &&
            tiles(128, pick_tn) > target &&
            (wave64_anycta || tiles(64, 128) <= target)) {
            return bf16_dispatch::BF16TileIndex<
                GemmKernelConfigBF16NaiveNoBias64x128x64_1CTA>();
        }

        // ---- 128x64 pair-UMMA, 18 stages --------------------------------
        // Used when the grid is one wave with enough K-tiles, or when the
        // 128xN choice is 128x128 (only 4 stages). Its CTA count stays in
        // the correct band. HALOPS_DISABLE_PAIR_DEEP=1 disables it.
        static const bool pairdeep_disabled =
            EnvIs("HALOPS_DISABLE_PAIR_DEEP", "1");
        if (!pairdeep_disabled && (m % 128 == 0) && (n % 128 == 0) &&
            (k % 64 == 0)) {
            const int pd_target = ResolveSmCount(sm_count);
            const int64_t pair_ctas = PairDeepCtaCount(m, n);
            const int64_t pair_k_tiles = k / kBF16BlockK;
            const bool one_wave =
                pair_ctas <= pd_target && 2 * pair_ctas > pd_target;
            const bool deep_enough = pair_k_tiles >= kMidMDeepPrologueMinKTiles;
            const bool shallow_incumbent = (pick == 1);
            if ((one_wave && deep_enough) || shallow_incumbent) {
                // With <= 64 K-tiles, 64x64x64_1WG_P14 is faster.
                // HALOPS_PAIRDEEP_SHORTK=0 disables it.
                static const bool pairdeep_shortk =
                    EnvNot0("HALOPS_PAIRDEEP_SHORTK");
                if (pairdeep_shortk && one_wave && deep_enough &&
                    pair_k_tiles <= 64 && (m % 64 == 0) && (n % 64 == 0)) {
                    return bf16_dispatch::BF16TileIndex<
                        GemmKernelConfigBF16NaiveNoBias64x64x64_1WG_P14_1CTA>();
                }
                return bf16_dispatch::BF16TileIndex<
                    GemmKernelConfigBF16NaiveNoBias128x64x64_2CTA_P18>();
            }
        }
        if (pick == 3) {
            // HALOPS_TILE_K32=1 swaps in 128x32x32 (16 stages).
            static const bool k32_enabled = EnvIs("HALOPS_TILE_K32", "1");
            if (k32_enabled && (k % 32 == 0)) {
                return bf16_dispatch::BF16TileIndex<
                    GemmKernelConfigBF16NaiveNoBias128x32x32_1CTA>();
            }
            // HALOPS_THINM_STAGES=4|6 forces a shallower prologue.
            static const int thinm_stages = EnvIntOr("HALOPS_THINM_STAGES", 0);
            if (thinm_stages == 4) {
                return bf16_dispatch::BF16TileIndex<
                    GemmKernelConfigBF16NaiveNoBias128x32x64_P4_1CTA>();
            }
            if (thinm_stages == 6) {
                return bf16_dispatch::BF16TileIndex<
                    GemmKernelConfigBF16NaiveNoBias128x32x64_P6_1CTA>();
            }
            return bf16_dispatch::BF16TileIndex<
                GemmKernelConfigBF16NaiveNoBias128x32x64_1CTA>();
        }
        if (pick == 2) {
            return bf16_dispatch::BF16TileIndex<
                GemmKernelConfigBF16NaiveNoBias128x64x64_1CTA>();
        }
        if (pick == 1) {
            return bf16_dispatch::BF16TileIndex<
                GemmKernelConfigBF16NaiveNoBias128x128x64_1CTA>();
        }
        return bf16_dispatch::BF16TileIndex<
            GemmKernelConfigBF16NaiveNoBias128x256x64_1CTA>();
    }

    if (ThinMNoSwapOk(m, n, k, sm_count)) {
        // ---- m < 128 without SwapAB ------------------------------------
        // At m=64 SwapAB yields too few CTAs; KBlockM=64 tiles give
        // ceil(n/32). Use 64x32 if it fits one wave, else 64x64.
        const int target = ResolveSmCount(sm_count);
        const int64_t mt = CeilDiv(m, 64);
        const int64_t nt32 = CeilDiv(n, 32);
        if (nt32 * mt <= target && (n % 32 == 0)) {
            if (DeepProloguePays(k)) {
                return bf16_dispatch::BF16TileIndex<
                    GemmKernelConfigBF16NaiveNoBias64x32x64_1WG_P18_1CTA>();
            }
            return bf16_dispatch::BF16TileIndex<
                GemmKernelConfigBF16NaiveNoBias64x32x64_1WG_1CTA>();
        }
        // HALOPS_M64_1WG_DEEP=0 restores the 10-stage config.
        static const bool thin_1wg_deep = EnvNot0("HALOPS_M64_1WG_DEEP");
        if (thin_1wg_deep && DeepProloguePays(k)) {
            return bf16_dispatch::BF16TileIndex<
                GemmKernelConfigBF16NaiveNoBias64x64x64_1WG_P14_1CTA>();
        }
        return bf16_dispatch::BF16TileIndex<
            GemmKernelConfigBF16NaiveNoBias64x64x64_1WG_1CTA>();
    }

    // Small-N SwapAB: weight (N×K) → UMMA A (M-dim=128), activation
    // (M×K) → UMMA B (N-dim=64).
    if (n % 128 != 0 || m % 64 != 0 || k % 64 != 0) {
        return -1;
    }
    // Use KBlockK=128 for k >= 8192. HALOPS_DISABLE_SWAPAB_K128=1
    // disables it.
    static const bool swapab_k128_disabled =
        EnvIs("HALOPS_DISABLE_SWAPAB_K128", "1");
    if (!swapab_k128_disabled && (k % 128 == 0) && k >= 8192) {
        return bf16_dispatch::BF16TileIndex<
            GemmKernelConfigBF16NaiveNoBias128x64x128_SwapAB_1CTA>();
    }
    // Use the 256-wide pair tile when the 128-wide grid exceeds 2
    // waves. HALOPS_DISABLE_SWAPAB_N256=1 disables it.
    static const bool swapab_n256_disabled =
        EnvIs("HALOPS_DISABLE_SWAPAB_N256", "1");
    const int swapab_target = ResolveSmCount(sm_count);
    const int64_t swapab_tiles128 = CeilDiv(n, 128) * CeilDiv(m, 64);
    if (!swapab_n256_disabled && (n % 256 == 0) &&
        swapab_tiles128 > 2 * swapab_target) {
        return bf16_dispatch::BF16TileIndex<
            GemmKernelConfigBF16NaiveNoBias256x64x64_SwapAB_2CTA>();
    }
    // Use KBlockN=32 if that still fits one wave.
    // HALOPS_DISABLE_SWAPAB_N32=1 disables it.
    static const bool swapab_n32_disabled =
        EnvIs("HALOPS_DISABLE_SWAPAB_N32", "1");
    const int64_t swapab_tiles64 = CeilDiv(n, 128) * CeilDiv(m, 32);
    if (!swapab_n32_disabled && swapab_tiles64 <= swapab_target) {
        return bf16_dispatch::BF16TileIndex<
            GemmKernelConfigBF16NaiveNoBias128x32x64_SwapAB_1CTA>();
    }
    return bf16_dispatch::BF16TileIndex<
        GemmKernelConfigBF16NaiveNoBias128x64x64_SwapAB_1CTA>();
}

}  // namespace dense

// ---------------------------------------------------------------------------
// Grouped/batch catalogs and selectors (moved verbatim from their .cu
// host functions).
// ---------------------------------------------------------------------------

using BF16ContiguousTiles = cute::tuple<
    GemmKernelConfigBF16ContiguousNoBias256x256x64_2CTA,         // 0
    GemmKernelConfigBF16ContiguousNoBias256x128x64_2CTA,         // 1
    GemmKernelConfigBF16ContiguousNoBias128x256x64_1CTA,         // 2
    GemmKernelConfigBF16ContiguousNoBias128x128x64_1CTA,         // 3
    GemmKernelConfigBF16ContiguousNoBias128x64x64_1CTA,          // 4
    GemmKernelConfigBF16ContiguousNoBias128x64x64_SwapAB_1CTA>;  // 5
constexpr int kBF16ContiguousCount =
    static_cast<int>(cute::tuple_size_v<BF16ContiguousTiles>);

using BF16MaskedTiles =
    cute::tuple<GemmKernelConfigBF16MaskedNoBias128x256x64_1CTA,        // 0
                GemmKernelConfigBF16MaskedNoBias128x64x64_SwapAB_1CTA,  // 1
                GemmKernelConfigBF16MaskedNoBias128x128x64_1CTA,        // 2
                GemmKernelConfigBF16MaskedNoBias128x64x64_1CTA,         // 3
                GemmKernelConfigBF16MaskedNoBias256x256x64_2CTA,        // 4
                GemmKernelConfigBF16MaskedNoBias256x128x64_2CTA>;       // 5
constexpr int kBF16MaskedCount =
    static_cast<int>(cute::tuple_size_v<BF16MaskedTiles>);

using BF16BatchTiles =
    cute::tuple<GemmKernelConfigBF16BatchNoBias256x256x64_2CTA,         // 0
                GemmKernelConfigBF16BatchNoBias128x256x64_1CTA,         // 1
                GemmKernelConfigBF16BatchNoBias128x64x64_SwapAB_1CTA>;  // 2
constexpr int kBF16BatchCount =
    static_cast<int>(cute::tuple_size_v<BF16BatchTiles>);

// group_contiguous: m is the total token count.
inline int SelectBF16ContiguousCfg(int m, int n, int k, int sm_count) {
    if (m >= 4096) {
        return 0;
    }
    if (m >= 128) {
        // Step finer on N only while the coarser tile underfills the SM
        // grid. M stays 128 (BF16 dummy-SF 1-CTA MMA requires M==128).
        const int target_sm = ResolveSmCount(sm_count);
        auto tiles = [&](int tm, int tn) {
            return ((m + tm - 1) / tm) * ((n + tn - 1) / tn);
        };
        // Step 0: 2-CTA pair-UMMA 256x256 (weight TMA-multicast across the
        // pair, much higher per-SM MMA throughput than 1-CTA 128x256).
        // Each cluster spans 2 SMs: require clusters*2 >= target_sm.
        // Needs m%256, n%256, k%64.
        const int clusters_2cta = ((m + 255) / 256) * ((n + 255) / 256);
        if (m % 256 == 0 && n % 256 == 0 && k % 64 == 0 &&
            clusters_2cta * 2 >= target_sm) {
            // 256x128 frees a deeper (8- vs 6-stage) pipeline; pays only
            // when 256x256 is lightly oversubscribed (<= ~2 waves).
            const bool square_like =
                (n % 128 == 0) && (clusters_2cta <= 2 * target_sm);
            return square_like ? 1 : 0;
        }
        // N-tile ladder: 128x128 when 128x256 underfills; 128x64 only on a
        // <=1-wave fit, since halving N doubles M*K panel reloads (costly
        // on DRAM-bound mid-M).
        int pick = 2;
        if (tiles(128, 256) < target_sm && (n % 128 == 0)) {
            pick = 3;
            if (tiles(128, 64) <= target_sm && (n % 64 == 0)) {
                pick = 4;
            }
        }
        return pick;
    }
    return 5;  // SwapAB
}

// group_masked: tiles scale with num_experts; mm is max_m_per_expert.
inline int SelectBF16MaskedCfg(int num_experts, int mm, int n, int k,
                               int sm_count) {
    if (mm >= 4096) {
        return 4;
    }
    if (mm >= 128) {
        // Masked pads every expert to max_m_per_expert, so tiles =
        // num_experts * ceil(max_m/tm) * ceil(n/tn). M stays 128.
        const int target_sm = ResolveSmCount(sm_count);
        auto tiles = [&](int tm, int tn) {
            return num_experts * ((mm + tm - 1) / tm) * ((n + tn - 1) / tn);
        };
        const int clusters_2cta =
            num_experts * ((mm + 255) / 256) * ((n + 255) / 256);
        if (mm % 256 == 0 && n % 256 == 0 && k % 64 == 0 &&
            clusters_2cta * 2 >= target_sm) {
            const bool square_like =
                (n % 128 == 0) && (clusters_2cta <= 2 * target_sm);
            return square_like ? 5 : 4;
        }
        // N-tile ladder: keep 128x256 (best B-reuse) unless it underfills
        // the grid; a blanket 128x128 preference regresses oversubscribed
        // mid-M.
        int pick = 0;
        if (tiles(128, 256) < target_sm && (n % 128 == 0)) {
            pick = 2;
            // 128x64 only on a <=1-wave fit: halving N doubles M*K panel
            // reloads (costly on DRAM-bound mid-M).
            if (tiles(128, 64) <= target_sm && (n % 64 == 0)) {
                pick = 3;
            }
        }
        return pick;
    }
    return 1;  // SwapAB
}

// batch: pure alignment ladder (no runtime SM reasoning).
inline int SelectBF16BatchCfg(int m_per_item, int n, int k) {
    if (m_per_item % 256 == 0 && n % 256 == 0) {
        return 0;
    }
    if (m_per_item % 128 == 0 && n % 256 == 0) {
        return 1;
    }
    if (m_per_item % 64 == 0 && n % 128 == 0) {
        return 2;
    }
    return -1;
}

template <class Tuple, std::size_t... I>
constexpr std::array<BF16CfgDesc, sizeof...(I)> MakeBF16DescTable(
    std::index_sequence<I...>) {
    return {{MakeBF16Desc<cute::tuple_element_t<I, Tuple>>()...}};
}

constexpr auto kBF16ContiguousCfgs = MakeBF16DescTable<BF16ContiguousTiles>(
    std::make_index_sequence<kBF16ContiguousCount>{});
constexpr auto kBF16MaskedCfgs = MakeBF16DescTable<BF16MaskedTiles>(
    std::make_index_sequence<kBF16MaskedCount>{});
constexpr auto kBF16BatchCfgs = MakeBF16DescTable<BF16BatchTiles>(
    std::make_index_sequence<kBF16BatchCount>{});

// Divisibility validation for a selected grouped/batch cfg, with the same
// per-rung contract the .cu ladders enforced (N relaxes to %32 for
// non-power-of-2 widths, %64 for predicated tails).
inline void CheckBF16CfgShape(const char* what, const BF16CfgDesc& d, int m,
                              int n, int k) {
    const int n_mod = d.ntail ? 64 : (((d.bn & (d.bn - 1)) != 0) ? 32 : d.bn);
    // SwapAB: KBlockM tiles the weight (output N), KBlockN the activation
    // (output M).
    const int m_div = d.swap ? d.bn : d.bm;
    const int n_div = d.swap ? d.bm : n_mod;
    TORCH_CHECK(m % m_div == 0, what, " requires M to be divisible by ", m_div,
                ", got ", m);
    TORCH_CHECK(n % n_div == 0, what, " requires N to be divisible by ", n_div,
                ", got ", n);
    TORCH_CHECK(k % d.bk == 0, what, " requires K to be divisible by ", d.bk,
                ", got ", k);
}

// Generic id->tile dispatcher for the small catalogs above.
template <class Catalog, int Count, class LaunchTile>
void DispatchBF16SmallCfg(int cfg, const char* what, LaunchTile&& launch) {
    const bool found = [&]<int... Is>(std::integer_sequence<int, Is...>) {
        return ((cfg == Is ? (launch.template
                              operator()<cute::tuple_element_t<Is, Catalog>>(),
                              true)
                           : false) ||
                ...);
    }(std::make_integer_sequence<int, Count>{});
    TORCH_CHECK(found, "unknown ", what, " cfg id ", cfg);
}

// Launch dispatch: maps a catalog id to the tile instantiation. One branch
// per catalog entry, generated from the tuple (gemm_tile.cuh's DispatchCfgAt
// lives in an anonymous namespace, so this is a local twin).
template <class LaunchTile>
void DispatchBF16Cfg(int cfg, LaunchTile&& launch) {
    const bool found = [&]<int... Is>(std::integer_sequence<int, Is...>) {
        return ((cfg == Is
                     ? (launch.template
                        operator()<cute::tuple_element_t<Is, BF16Tiles>>(),
                        true)
                     : false) ||
                ...);
    }(std::make_integer_sequence<int, kBF16CfgCount>{});
    TORCH_CHECK(found, "unknown bf16 cfg id ", cfg);
}

}  // namespace bf16_dispatch
}  // namespace halops
