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

#include <cutlass/arch/barrier.h>
#include <cutlass/numeric_types.h>

#include "mla_intrinsics.cuh"

namespace mega_mla {

using e4m3 = cutlass::float_e4m3_t;
using ue8m0 = cutlass::float_ue8m0_t;
using bf16 = cutlass::bfloat16_t;

CUTE_HOST_DEVICE constexpr int cfg_max4(int a, int b, int c, int d) {
    int m = a;
    if (b > m) m = b;
    if (c > m) m = c;
    if (d > m) m = d;
    return m;
}

// Switch models by pointing ModelDims at another struct; only the first block
// is free, everything below it is derived.
struct Dsv4Dims {
    static constexpr int B_H = 64;
    static constexpr int D_Q = 512;
    static constexpr int D_V = 512;
    static constexpr int D_NOPE = 448;
    static constexpr int D_ROPE = 64;

    static constexpr int HIDDEN = 4096;
    static constexpr int QLORA_RANK = 1024;
    static constexpr int KVLORA_RANK = 512;
    static constexpr int O_GROUPS = 8;
    static constexpr int O_LORA_RANK = 1024;

    static constexpr int C128_RATIO = 128;

    static constexpr int Q_TOTAL = B_H * D_Q;
    static constexpr int O_TOTAL = B_H * D_V;
    static constexpr int D_Q_SW128 = D_Q;
    static constexpr int K_ROPE_SW = 2 * D_ROPE;
    static constexpr int TMA_K_STRIDE = D_NOPE + 2 * D_ROPE;

    static_assert(D_Q == D_V && D_Q == D_NOPE + D_ROPE,
                  "the NoPE and RoPE halves must partition the one 512-wide "
                  "latent that Q, K and V all share");
    static_assert(O_TOTAL % O_GROUPS == 0,
                  "flat_o must split evenly into O_GROUPS groups");
    static_assert(C128_RATIO > 0 && (C128_RATIO & (C128_RATIO - 1)) == 0,
                  "the HCA compression ratio is used as a mask, so it must be "
                  "a power of two");
};

using ModelDims = Dsv4Dims;

#ifndef MEGA_MLA_OLORA_GPP
#    define MEGA_MLA_OLORA_GPP 4
#endif

// The ONE home of every number that is not a model dimension.
struct KernelTrait {
    static constexpr int NumThreads = 384;
    static constexpr int WarpSize = 32;
    static constexpr int NumWarps = NumThreads / WarpSize;
    static constexpr int ThreadsPerWarpGroup = 128;
    static constexpr int NumWarpGroups = NumThreads / ThreadsPerWarpGroup;
    static constexpr int WarpsPerWarpGroup = ThreadsPerWarpGroup / WarpSize;

    // The instantiated tile widths, in ascending order: each one is a separate
    // cubin, so a width MegaMla's switch lacks silently falls to the default.
    struct Tile {
        static constexpr int Widths[4] = {16, 32, 64, 128};
        static constexpr int Limits[3] = {16, 64, 192};
        static constexpr int Smallest = Widths[0];
        static constexpr int Widest = Widths[3];
    };

    static constexpr int DeviceCacheSlots = 16;

    struct Trace {
        static constexpr int CtaSlots = 256;
        static constexpr int WordsPerWarp = 4;
        static constexpr int WarpsPerCta = NumWarps;
    };

    static CUTE_DEVICE int warp_idx() { return (int)(threadIdx.x / WarpSize); }
    static CUTE_DEVICE int lane_idx() {
        return (int)(threadIdx.x & (WarpSize - 1));
    }
    static CUTE_DEVICE int warpgroup_idx() {
        return (int)(threadIdx.x / ThreadsPerWarpGroup);
    }
    static CUTE_DEVICE int idx_in_warpgroup() {
        return (int)(threadIdx.x % ThreadsPerWarpGroup);
    }

    // Never add a Bar overload taking the width separately: two sites passing
    // different counts for one id is a silent hang nothing can catch.
    struct Bars {
        template <uint32_t Id, int Width>
        struct Bar {
            static constexpr uint32_t id = Id;
            static constexpr int width = Width;
            static_assert(Width % WarpSize == 0,
                          "a NamedBarrier's width must be a whole number of "
                          "warps: bar.sync requires all 32 lanes of a warp to "
                          "arrive together, so a partial warp can never "
                          "satisfy it");
            static_assert(Width <= NumThreads,
                          "width exceeds the CTA: this barrier can never be "
                          "satisfied");
        };

        using Wg0Sync = Bar<1, ThreadsPerWarpGroup>;
        using Wg1Warp02Sync = Bar<2, 2 * WarpSize>;
        using Wg1Warp13Sync = Bar<3, 2 * WarpSize>;
        using EveryoneSync = Bar<4, NumThreads>;
        using Wg1Sync = Bar<5, ThreadsPerWarpGroup>;
        using Wg2Sync = Bar<6, ThreadsPerWarpGroup>;
        using Wg01Sync = Bar<7, 2 * ThreadsPerWarpGroup>;

        static_assert(Wg1Warp13Sync::id == Wg1Warp02Sync::id + 1,
                      "attention picks between these two at run time with "
                      "`Wg1Warp02Sync::id + (local_warp & 1)`, so they must be "
                      "adjacent");
    };

    // Two roles matching one warp silently drops work, none hangs.  These stay
    // bare __device__: __forceinline__ here costs 204->276 B of spill loads.
    struct Roles {
        static constexpr int MmaWarp = 0;
        static constexpr int Tma1Warp = 1;
        static constexpr int Tma2Warp = 2;
        static constexpr int Tma3Warp = 3;
        static constexpr int DrainWarpBegin = 4;
        static constexpr int DrainWarps = WarpsPerWarpGroup;
        static constexpr int VecWarpBegin = 8;
        static constexpr int VecWarps = WarpsPerWarpGroup;

        __device__ static constexpr bool is_mma(int w) { return w == MmaWarp; }
        __device__ static constexpr bool is_tma1(int w) {
            return w == Tma1Warp;
        }
        __device__ static constexpr bool is_tma2(int w) {
            return w == Tma2Warp;
        }
        __device__ static constexpr bool is_tma3(int w) {
            return w == Tma3Warp;
        }
        __device__ static constexpr bool is_drain(int w) {
            return w >= DrainWarpBegin && w < DrainWarpBegin + DrainWarps;
        }
        __device__ static constexpr bool is_vec(int w) {
            return w >= VecWarpBegin && w < VecWarpBegin + VecWarps;
        }
        __device__ static constexpr int drain_local_warp(int w) {
            return w - DrainWarpBegin;
        }

        static constexpr int roles_matching(int w) {
            return (int)(w == MmaWarp) + (int)(w == Tma1Warp) +
                   (int)(w == Tma2Warp) + (int)(w == Tma3Warp) +
                   (int)(w >= DrainWarpBegin &&
                         w < DrainWarpBegin + DrainWarps) +
                   (int)(w >= VecWarpBegin && w < VecWarpBegin + VecWarps);
        }

        static constexpr bool partitions_all_warps() {
            for (int w = 0; w < NumWarps; ++w)
                if (roles_matching(w) != 1) return false;
            return true;
        }

        static_assert(
            DrainWarpBegin % WarpsPerWarpGroup == 0 &&
                VecWarpBegin % WarpsPerWarpGroup == 0,
            "the drain and vec groups each run a warpgroup-collective "
            "setmaxnreg and a 128-wide NamedBarrier, so each must be "
            "exactly one aligned warpgroup; the drain additionally "
            "needs one warp per TMEM datapath quadrant (warp % 4)");
        static_assert(MmaWarp < WarpsPerWarpGroup &&
                          Tma1Warp < WarpsPerWarpGroup &&
                          Tma2Warp < WarpsPerWarpGroup &&
                          Tma3Warp < WarpsPerWarpGroup,
                      "the mma + three TMA producers share one setmaxnreg, so "
                      "all four must sit in warpgroup 0");
    };

    template <class Dims>
    struct AttnTrait {
        static constexpr int B_TOPK = 64;
        static constexpr int NUM_BUFS = 2;
        static constexpr int MaxKvSegsPerToken = 2;
        static constexpr int KvDenseLevels = 4;

        struct KvSplit {
            static constexpr int SegmentCapDefault = 8;
            static constexpr int MinBlocksToCutDefault = 8;
            static constexpr int GainDefault = 4;
        };

        static constexpr float MAX_INIT_VAL = -1e30f;
        static constexpr int O_TMA_BOX = 64;
        static constexpr int O_MMA_N = 256;

        struct tmem_cols {
            static constexpr int O = 0;
            static constexpr int Q = 256;
            static constexpr int P = 384;
            static constexpr int P_STRIDE = 64;
            static constexpr int TOTAL = 512;
        };
        static_assert(tmem_cols::P + NUM_BUFS * tmem_cols::P_STRIDE <=
                          tmem_cols::TOTAL,
                      "P double buffer must fit in the 512 TMEM columns");

        using SmemLayoutQ_SW128 = decltype(cute::coalesce(
            cute::tile_to_shape(
                cute::UMMA::Layout_K_SW128_Atom<cutlass::bfloat16_t>{},
                cute::Shape<cute::Int<Dims::B_H>, cute::Int<Dims::D_Q_SW128>>{},
                cute::Step<cute::_1, cute::_2>{}),
            cute::Shape<cute::_1, cute::_1>{}));

        using SmemLayoutOBuf = decltype(cute::tile_to_shape(
            cute::UMMA::Layout_K_SW128_Atom<cutlass::bfloat16_t>{},
            cute::Shape<cute::Int<Dims::B_H>, cute::Int<Dims::D_V>>{}));

        using SmemLayoutOBuf_TMA = decltype(cute::tile_to_shape(
            cute::UMMA::Layout_K_SW128_Atom<cutlass::bfloat16_t>{},
            cute::Shape<cute::Int<Dims::B_H>, cute::Int<O_TMA_BOX>>{}));

        using SmemLayoutS = decltype(cute::tile_to_shape(
            cute::UMMA::Layout_K_INTER_Atom<cutlass::bfloat16_t>{},
            cute::Shape<cute::Int<Dims::B_H>, cute::Int<B_TOPK>>{},
            cute::Step<cute::_1, cute::_2>{}));

        template <int NUM_TILES>
        using SmemLayoutKTiles_SW128 = decltype(cute::coalesce(
            cute::tile_to_shape(
                cute::UMMA::Layout_K_SW128_Atom<cutlass::bfloat16_t>{},
                cute::Shape<cute::Int<Dims::B_H>,
                            cute::Int<B_TOPK * NUM_TILES>>{},
                cute::Step<cute::_1, cute::_2>{}),
            cute::Shape<cute::_1, cute::_1>{}));

        template <int NUM_TILES>
        using SmemLayoutKTiles_DualGemm_SW128 = decltype(cute::coalesce(
            cute::tile_to_shape(
                cute::UMMA::Layout_K_SW128_Atom<cutlass::bfloat16_t>{},
                cute::Shape<cute::Int<Dims::B_H * 2>,
                            cute::Int<B_TOPK * NUM_TILES>>{},
                cute::Step<cute::_1, cute::_2>{}),
            cute::Shape<cute::_1, cute::_1>{}));

        template <int NUM_TILES>
        using SmemLayoutKTilesTransposed_SW128 = decltype(cute::composition(
            SmemLayoutKTiles_SW128<NUM_TILES>{},
            cute::Layout<
                cute::Shape<cute::Int<B_TOPK * NUM_TILES>, cute::Int<B_TOPK>>,
                cute::Stride<cute::Int<B_TOPK>, cute::_1>>{}));

        using TiledMMA_P = decltype(cute::make_tiled_mma(
            cute::SM100_MMA_F16BF16_WS_TS_NOELECT<
                cutlass::bfloat16_t, cutlass::bfloat16_t, float, Dims::B_H,
                B_TOPK * 2, cute::UMMA::Major::K, cute::UMMA::Major::K>{}));

        using TiledMMA_O = decltype(cute::make_tiled_mma(
            cute::SM100_MMA_F16BF16_WS_SS_NOELECT<
                cutlass::bfloat16_t, cutlass::bfloat16_t, float, Dims::B_H,
                O_MMA_N, cute::UMMA::Major::K, cute::UMMA::Major::MN>{}));
    };

    static_assert(NumWarps == 12 && NumWarpGroups == 3);
};

static_assert(KernelTrait::Roles::partitions_all_warps(),
              "Roles must partition all 12 warps exactly once");

template <class Cfg>
struct ValidateGemmCfg {
    static_assert(Cfg::BM == 128,
                  "BM is pinned: the SF atom is 128 MN rows and the TMEM drain "
                  "needs one warp per datapath quadrant of an M=128 acc");
    static_assert(Cfg::BN >= KernelTrait::Tile::Smallest &&
                      Cfg::BN <= Cfg::SfMnPerTile &&
                      (Cfg::BN & (Cfg::BN - 1)) == 0,
                  "BN must be a power of two in 16..128: the bf16 kv_gate UMMA "
                  "needs N % 16 == 0 at M=128, and 256 collides with the "
                  "attention TMEM map");
    static_assert(Cfg::BK > 0 && Cfg::KSplit > 0 && Cfg::TmaStages >= 1 &&
                      Cfg::AccStages >= 1,
                  "tile and stage counts must be positive; how they divide M "
                  "and K is checked by make_gemm_shape on the host");
    static_assert(Cfg::KSplit <= Cfg::KSplitMax,
                  "a K-split above this GEMM's ceiling overruns its partials");
    static_assert(Cfg::SwapAB || (Cfg::BN == Cfg::BM),
                  "the non-SwapAB orientation is implemented only where the "
                  "tile is square: below that the N side carries CTA_N_SF "
                  "sub-slot machinery the M side has no counterpart for");
    static constexpr bool ok = true;
};

template <class Cfg, int KS>
struct WithKSplit : Cfg {
    static constexpr int KSplit = KS;
    static_assert(KS > 0, "a K-split must be positive");
};

// One config per GEMM, all beside Attn: BM/BN/BK, the two pipeline depths, the
// K-split and its own workspace ceiling, and the operand orientation.  SwapAB
// puts the WEIGHT in the A slot; nothing else in the kernel is named for a
// tensor, so a_side()/b_side() are the only bridge from tensor to slot.
template <class Dims_, int TileT_>
struct KernelTraits : KernelTrait {
    using Dims = Dims_;
    static constexpr int TileT = TileT_;
    using Attn = KernelTrait::AttnTrait<Dims_>;

    struct GemmBase {
        // The two MMA axes by MEANING, before either is called M or N.
        static constexpr int TokenTile = TileT_;
        static constexpr int WeightTile = 128;
        static constexpr int BK = 256;
        static constexpr int KItersPerBlock = 8;

        static constexpr int SfGran = 32;
        static constexpr int SfMnPerCol = 32;
        static constexpr int SfMnPerTile = 4 * SfMnPerCol;
        static constexpr int SfPackedMinTile = 64;
        static constexpr bool SfTilePerGroup = TokenTile < SfPackedMinTile;

        // SwapAB puts the WEIGHT in the A slot, so M is weight rows and the
        // token axis is N.  From NoSwapMinTile up the ACTIVATION is A and the
        // token axis becomes M -- which is why BM/BN are derived, never
        // spelled.
        static constexpr int NoSwapMinTile = 128;
        static constexpr bool SwapAB = TokenTile < NoSwapMinTile;
        static constexpr int BM = SwapAB ? WeightTile : TokenTile;
        static constexpr int BN = SwapAB ? TokenTile : WeightTile;

        static constexpr int TmaStages = 2;
        static constexpr int AccStages = 1;
        static constexpr int KSplit = 1;
        static constexpr int KSplitMax = 1;
        using PartialT = bf16;
    };

    struct Qkv : GemmBase {
        static constexpr int DeepPipelineMaxTile = 64;
        static constexpr int TmaStages =
            GemmBase::TokenTile <= DeepPipelineMaxTile ? 4 : 3;
        static constexpr int AccStages = 2;
        static constexpr int KSplit = 4;
        static constexpr int KSplitMax = 4;
        using PartialT = float;
        static_assert(GemmBase::TokenTile <= DeepPipelineMaxTile ||
                          TmaStages == 3,
                      "the widest token tile must stay at 3 stages: its "
                      "per-stage cost is 67584 B, so a 4th wants 270336 B of a "
                      "232448 B plan");
    };

    struct WqB : GemmBase {
        static constexpr int AccStages = 2;
    };

    struct WoA : GemmBase {
        static constexpr int TmaStages = 3;
        static constexpr int AccStages = 2;
        static constexpr int KSplit = 2;
        static constexpr int KSplitMax = 2;
        static constexpr int OloraGroupsPerPass = MEGA_MLA_OLORA_GPP;
    };

    struct WoB : GemmBase {
        static constexpr int TmaStages = 3;
        static constexpr int AccStages = 2;
        static constexpr int KSplit = 4;
        static constexpr int KSplitMax = 4;
        static constexpr int Variants = 3;
    };

    struct KvGate : GemmBase {
        static constexpr int BK = 128;
        static constexpr int TmaStages = 3;
        static constexpr int AccStages = 2;
        using PartialT = float;
    };

    // Everything about the gmem workspace's shape.  The three byte counts are
    // independent 128s: the first block holds the one grid-barrier counter,
    // each rendezvous counter owns a cache line so their red.add traffic does
    // not collide, and every block's offset must clear a TMA descriptor's 128 B
    // base requirement (a 32 B-aligned ws_o_partial emitted nan with no
    // assert). TokenPadRows is in TOKENS, not bytes: it pads the per-token
    // blocks to a whole scale-factor tile, so it IS the widest SfMnPerTile.
    struct Workspace {
        static constexpr int GridBarrierBytes = 128;
        static constexpr int RendezvousSlotBytes = 128;
        static constexpr int BlockAlignBytes = 128;
        static constexpr int TokenPadRows =
            cfg_max4(Qkv::SfMnPerTile, WqB::SfMnPerTile, WoA::SfMnPerTile,
                     WoB::SfMnPerTile);
        static constexpr int SfPadRows = TokenPadRows / Tile::Smallest;
    };

    static constexpr int MaxTmaStages = cfg_max4(
        Qkv::TmaStages, WqB::TmaStages, WoA::TmaStages, WoB::TmaStages);
    static constexpr int MaxAccStages = cfg_max4(
        Qkv::AccStages, WqB::AccStages, WoA::AccStages, WoB::AccStages);

    static constexpr bool validated =
        ValidateGemmCfg<Qkv>::ok && ValidateGemmCfg<WqB>::ok &&
        ValidateGemmCfg<WoA>::ok && ValidateGemmCfg<WoB>::ok;
    static_assert(Qkv::SwapAB == WqB::SwapAB && Qkv::SwapAB == WoA::SwapAB &&
                      Qkv::SwapAB == WoB::SwapAB &&
                      Qkv::SwapAB == KvGate::SwapAB,
                  "all five GEMMs overlay ONE smem A/B arena and one set of "
                  "per-stage barriers, so they must agree on which operand is "
                  "A; a lone disagreeing GEMM reads the other's half");
    static_assert(KvGate::TmaStages <= MaxTmaStages &&
                      KvGate::AccStages <= MaxAccStages,
                  "kv_gate shares bar_gemm_*, sized from the fp8 GEMMs' max; a "
                  "deeper depth indexes past them and hangs");
};

using AttnConfig = KernelTrait::AttnTrait<ModelDims>;

struct AttnRing {
    int buf_idx = 0;
    bool bar_phase = 0;
    CUTE_DEVICE void update() {
        bar_phase ^= (buf_idx == AttnConfig::NUM_BUFS - 1);
        buf_idx = (buf_idx + 1) % AttnConfig::NUM_BUFS;
    }
};

using TraitsN16 = KernelTraits<ModelDims, 16>;
using TraitsN32 = KernelTraits<ModelDims, 32>;
using TraitsN64 = KernelTraits<ModelDims, 64>;
using TraitsN128 = KernelTraits<ModelDims, 128>;

// Two spellings for the tile-independent halves of the config, so code that is
// not templated on a tile width does not have to invent one.
using GemmShapeDefaults = TraitsN128::GemmBase;
using WorkspaceLayout = TraitsN128::Workspace;
static_assert(TraitsN16::validated && TraitsN32::validated &&
                  TraitsN64::validated && TraitsN128::validated,
              "every instantiated tile width must validate");

}  // namespace mega_mla
