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

#include "kernel/mla/mla_gemm.cuh"

namespace halops::mla::swapab {

template <int TT>
constexpr int exchange_tokens() {
    return TT > 64 ? 32 : TT % 32 == 0 ? 32 : 16;
}

template <int TT>
constexpr int k_block() {
    return TT <= 64 ? 256 : 128;
}

template <int TT, int Stages>
struct GemmGeom {
    using Geom = GemmGeom;
    struct OwnerTraits {
        static constexpr int MaxTmaStages = Stages;
    };
    static constexpr int TokenTile = TT;
    static constexpr bool SfTilePerGroup = TT < SfLayout::PackedMinTile;
    static constexpr bool WeightInA = true;
    static constexpr int BK = k_block<TT>();
    static constexpr int KItersPerBlock = BK / SfLayout::Gran;
    static constexpr int WeightTile = OutputProjLayout::RowTile;
    static constexpr int BM = OutputProjLayout::RowTile;
    static constexpr int BN = TT;
    static constexpr int CtaNSf = cute::round_up(BN, SfLayout::MnPerTile);
    static constexpr int TmaStages = Stages;
    static constexpr int SfCols =
        BK / SfLayout::Gran + (CtaNSf / SfLayout::MnPerCol) * (BK / 128);
    static constexpr int AccStages =
        2 * TT + SfCols <= Sm100::TmemColumns ? 2 : 1;
    using PartialT = cutlass::bfloat16_t;
    static constexpr int DrainNOverride =
        exchange_tokens<TT>() < TT ? exchange_tokens<TT>() : 0;
    static constexpr int ChainBN = TT;
    static constexpr int ChainCtaNSf = CtaNSf;
    static constexpr int AccStride = TT;
    static constexpr int AccSharedCols = 0;
    static constexpr int SfBaseCols = AccStages * TT;
};

template <int TT>
struct Shape {
    static constexpr int BK = k_block<TT>();
    static constexpr int Exchange = exchange_tokens<TT>();
    static constexpr int Slots = TT / Exchange;
    static constexpr int Ranks = OutputProjLayout::ClusterSize;
    static constexpr int RankRows = OutputProjLayout::RowTile / Ranks;
    static constexpr int SfRows = cute::round_up(TT, SfLayout::MnPerTile);
    static constexpr int StageBytes =
        OutputProjLayout::RowTile * BK + TT * BK +
        (OutputProjLayout::RowTile + SfRows) * BK / SfLayout::Gran;
    static constexpr int RxFloats = 2 * OutputProjLayout::RowTile * Exchange;
    static constexpr int ExchangeBytes = 2 * RxFloats * (int)sizeof(float);
    static constexpr int BarrierReserve = 3 * 1024;
    static constexpr int MaxStages = 6;
    static constexpr int FitStages =
        (Sm100::SmemPerCtaMax - ExchangeBytes - BarrierReserve) / StageBytes;
    static constexpr int Stages = FitStages < MaxStages ? FitStages : MaxStages;
    using Cfg = GemmGeom<TT, Stages>;
    static constexpr int TilesPerKBlock = BK / OutputProjLayout::RowTile;
    static constexpr int TileArrivals = 2;
};

template <int TT>
CUTE_HOST_DEVICE int wo_a_k_tiles(const OutputProjShape& shape) {
    return shape.group_k / Shape<TT>::BK / 2;
}
template <int TT>
CUTE_HOST_DEVICE int wo_b_k_tiles(const OutputProjShape& shape) {
    return shape.olora / Shape<TT>::BK / Shape<TT>::Ranks;
}

template <class Cfg>
struct Operands {
    WeightAtoms<Cfg> weight;
    ActAtoms<Cfg> act;
};

template <int TT>
struct Smem {
    using Sh = Shape<TT>;
    using Inst = GemmInst<typename Sh::Cfg>;
    static constexpr int Arena =
        cute::round_up(Inst::SmemTotal, Sm100::Sw128AtomBytes);
    static constexpr int RxOff = Arena;
    static constexpr int StagingOff = RxOff + Sh::RxFloats * (int)sizeof(float);
    static constexpr int BarOff =
        StagingOff + Sh::RxFloats * (int)sizeof(float);
    struct Bars {
        cutlass::arch::ClusterTransactionBarrier tma_a[Sh::Stages],
            tma_b[Sh::Stages], smem_free[Sh::Stages];
        cutlass::arch::ClusterTransactionBarrier acc_full[2], acc_empty[2];
        uint64_t rx_full[2], rx_free[2];
        uint32_t tmem_slot;
        float amax[KernelBase::ThreadsPerWarpGroup];
    };
    static constexpr int Total = BarOff + (int)sizeof(Bars);
};

template <int TT>
CUTE_DEVICE GemmWorkItem gemm_item(int tile, int rows, int num_tokens) {
    return {.n_row = 0,
            .w_tile = tile,
            .w_out = rows,
            .k_tile_begin = 0,
            .a_k_tile_begin = 0,
            .out_row_stride = rows,
            .n_valid = min(TT, num_tokens),
            .w_valid = OutputProjLayout::RowTile};
}

template <int TT>
CUTE_DEVICE void prefetch_wo_b(const uint8_t* wo_b, int olora, int cluster,
                               int rank, int lane) {
    using Sh = Shape<TT>;
    static_assert(Sh::RankRows == KernelBase::WarpSize,
                  "one wo_b row per lane");
    constexpr int kSkip = Sh::Ranks * Sh::Stages * Sh::BK;
    const uint8_t* row = wo_b + (int64_t)(cluster * OutputProjLayout::RowTile +
                                          Sh::RankRows * rank + lane) *
                                    olora;
    if (olora > kSkip) prefetch_l2_bulk(row + kSkip, olora - kSkip);
}

struct Coord {
    int cluster, rank;
    int tile_a;
    int half;

    CUTE_DEVICE explicit Coord(const OutputProjShape& shape) {
        rank = cute::block_rank_in_cluster();
        cluster = blockIdx.x / OutputProjLayout::ClusterSize;
        tile_a = cluster + shape.clusters() * (rank >> 1);
        half = rank & 1;
    }
};

template <class Op>
CUTE_DEVICE void mma_item(Op& op, const GemmWorkItem& item, int k_tiles) {
    op.begin_item(item);
    CUTLASS_PRAGMA_NO_UNROLL
    for (int k = 0; k < k_tiles; ++k) {
        op.wait_tma(k);
        op.run_k_tile(k);
        op.commit(k);
    }
    op.end_item();
}

template <int TT, class Prm, class Res>
CUTE_DEVICE void mma_role(const Prm& params, const Res& res_a, const Res& res_b,
                          const Coord& coord, int lane) {
    using Cfg = typename Shape<TT>::Cfg;
    const int k_a = wo_a_k_tiles<TT>(params.shape);
    const int k_b = wo_b_k_tiles<TT>(params.shape);
    auto op_a = make_mma_op<Cfg>(res_a, 0, 0, k_a);
    mma_item(op_a,
             gemm_item<TT>(coord.tile_a, params.shape.olora, params.num_tokens),
             k_a);
    prefetch_wo_b<TT>(params.wo_b_raw, params.shape.olora, coord.cluster,
                      coord.rank, lane);
    auto op_b = make_mma_op<Cfg>(res_b, k_a, 1, k_b);
    mma_item(
        op_b,
        gemm_item<TT>(coord.cluster, params.shape.hidden, params.num_tokens),
        k_b);
}

template <int TT, class Prm, class Res>
CUTE_DEVICE void weight_role(const Prm& params, const Res& res_a,
                             const Res& res_b, const GemmPipelineBars& bars,
                             const Coord& coord) {
    using Sh = Shape<TT>;
    using Cfg = typename Sh::Cfg;
    const int k_a = wo_a_k_tiles<TT>(params.shape);
    const int k_b = wo_b_k_tiles<TT>(params.shape);
    auto op_a = make_tma_op_a<Cfg>(res_a, 0, k_a, NoGate{});
    op_a.begin_item(
        params.wo_a.weight,
        gemm_item<TT>(coord.tile_a, params.shape.olora, params.num_tokens));
    for (int k = 0; k < k_a; ++k) {
        wait_smem_stage_free<Sh::Stages>(bars, k);
        if (op_a.elected)
            dispatch_stage<Sh::Stages>(k % Sh::Stages, op_a.issuer(),
                                       coord.half * k_a + k);
    }
    auto op_b = make_tma_op_a<Cfg>(res_b, k_a, k_b, NoGate{});
    op_b.begin_item(
        params.wo_b.weight,
        gemm_item<TT>(coord.cluster, params.shape.hidden, params.num_tokens));
    for (int k = 0; k < k_b; ++k) {
        const int seq = k_a + k;
        wait_smem_stage_free<Sh::Stages>(bars, seq);
        if (op_b.elected)
            dispatch_stage<Sh::Stages>(seq % Sh::Stages, op_b.issuer(),
                                       Sh::Ranks * k + coord.rank);
    }
}

template <int TT, class Prm, class Res>
CUTE_DEVICE void act_role(const Prm& params, const Res& res_a, const Res& res_b,
                          const GemmPipelineBars& bars, const Coord& coord,
                          int lane) {
    using Sh = Shape<TT>;
    using Cfg = typename Sh::Cfg;
    const int k_a = wo_a_k_tiles<TT>(params.shape);
    const int k_b = wo_b_k_tiles<TT>(params.shape);
    const int tiles_per_group =
        params.shape.group_rows / OutputProjLayout::RowTile;
    pdl_wait();
    auto op_a = make_tma_op_b<Cfg>(res_a, 0, k_a, NoGate{});
    const int out_group = coord.tile_a / tiles_per_group;
    op_a.begin_item(
        params.wo_a.weight,
        gemm_item<TT>(coord.tile_a, params.shape.olora, params.num_tokens));
    for (int k = 0; k < k_a; ++k) {
        wait_smem_stage_free<Sh::Stages>(bars, k);
        if (op_a.elected)
            dispatch_stage<Sh::Stages>(
                k % Sh::Stages, op_a.issuer(),
                out_group * (2 * k_a) + coord.half * k_a + k);
    }
    auto op_b = make_tma_op_b<Cfg>(res_b, k_a, k_b, NoGate{});
    op_b.begin_item(
        params.wo_b.weight,
        gemm_item<TT>(coord.cluster, params.shape.hidden, params.num_tokens));
    uint32_t ready = 0;
    for (int k = 0; k < k_b; ++k) {
        const int seq = k_a + k;
        while (!((ready >> k) & 1)) {
            bool done = true;
            if (lane < k_b) {
                const int k_block = Sh::Ranks * lane + coord.rank;
                CUTLASS_PRAGMA_UNROLL
                for (int tile = 0; tile < Sh::TilesPerKBlock; ++tile)
                    done =
                        done && gmem_ld_acquire(params.flags +
                                                Sh::TilesPerKBlock * k_block +
                                                tile) >= Sh::TileArrivals;
            }
            ready = __ballot_sync(0xffffffffu, done && lane < k_b);
        }
        __syncwarp();
        fence_proxy_async_global();
        wait_smem_stage_free<Sh::Stages>(bars, seq);
        if (op_b.elected)
            dispatch_stage<Sh::Stages>(seq % Sh::Stages, op_b.issuer(),
                                       Sh::Ranks * k + coord.rank);
    }
}

template <int TT, class Prm, class BarsT>
struct Reduce {
    using Sh = Shape<TT>;
    using Sync = KernelBase::Bars;
    static constexpr int kHalfRows = OutputProjLayout::RowTile / 2;
    static constexpr int kTaskRows = SfLayout::Gran / 2;
    static constexpr int kTasks = OutputProjLayout::RowTile / Sh::Exchange;
    static constexpr int kRowsPerTask = Sh::RankRows / kTasks;
    static constexpr int kBufFloats = OutputProjLayout::RowTile * Sh::Exchange;
    static constexpr uint32_t kRowBytes = Sh::Exchange * sizeof(float);
    static constexpr uint32_t kSlotBytes =
        OutputProjLayout::RowTile * kRowBytes;
    static_assert(kTaskRows / 4 == sizeof(uint4) / sizeof(uint32_t),
                  "a wo_a task stores one 16-B fp8 vector");

    const Prm& params;
    BarsT& bars;
    const Coord& coord;
    const float* rx;
    float* staging;
    uint32_t rx_smem, staging_smem;
    int tid;

    CUTE_DEVICE static int at(int row, int token) {
        return swizzled_token_index<Sh::Exchange>(row, token);
    }
    CUTE_DEVICE static uint32_t bar_addr(const uint64_t* bar) {
        return cute::cast_smem_ptr_to_uint(bar);
    }

    template <class Op>
    CUTE_DEVICE void stage(Op& op, int slot, bool dry) {
        const int part = slot % Sh::Slots;
        float* buf = staging + (slot & 1) * kBufFloats;
        if (slot >= 2) {
            if (tid == 0) cute::tma_store_wait<1>();
            bar_sync<typename Sync::Wg1Sync>();
        }
        if (!dry) {
            if (part == 0) op.wait_acc_full();
            op.load_slice(part);
        }
        const int a_row = cute::get<0>(op.coord(0));
        CUTLASS_PRAGMA_UNROLL
        for (int c4 = 0; c4 < Sh::Exchange / 4; ++c4)
            *reinterpret_cast<float4*>(buf + at(a_row, 4 * c4)) =
                make_float4(op.thr_racc(4 * c4), op.thr_racc(4 * c4 + 1),
                            op.thr_racc(4 * c4 + 2), op.thr_racc(4 * c4 + 3));
        if (part == Sh::Slots - 1 && !dry) op.end_item();
        cutlass::arch::fence_view_async_shared();
        bar_sync<typename Sync::Wg1Sync>();
    }

    CUTE_DEVICE void copy_rows(int buf, uint32_t dst, int dst_row, int src_row,
                               int rows) const {
        bulk_copy_to_peer(
            cute::set_block_rank(
                rx_smem +
                    (uint32_t)(buf * OutputProjLayout::RowTile + dst_row) *
                        kRowBytes,
                dst),
            staging_smem +
                (uint32_t)(buf * OutputProjLayout::RowTile + src_row) *
                    kRowBytes,
            (uint32_t)rows * kRowBytes,
            cute::set_block_rank(bar_addr(&bars.rx_full[buf]), dst));
    }

    CUTE_DEVICE void copy_to_owners(int slot) const {
        const int buf = slot & 1;
        if (slot >= 2)
            mbar_wait_cluster(bar_addr(&bars.rx_free[buf]),
                              (uint32_t)(((slot >> 1) - 1) & 1));
        if (slot < Sh::Slots) {
            for (int dst_half = 0; dst_half < 2; ++dst_half)
                copy_rows(buf, (uint32_t)((coord.rank & ~1) | dst_half),
                          coord.half * kHalfRows, kHalfRows * dst_half,
                          kHalfRows);
        } else {
            for (int dst = 0; dst < Sh::Ranks; ++dst)
                copy_rows(buf, (uint32_t)dst, coord.rank * Sh::RankRows,
                          Sh::RankRows * dst, Sh::RankRows);
        }
        cute::tma_store_arrive();
    }

    template <class Op>
    CUTE_DEVICE void send(Op& op, int slot, bool dry) {
        stage(op, slot, dry);
        if (tid == 0 && !dry) copy_to_owners(slot);
    }

    CUTE_DEVICE void expect(int slot, bool dry) const {
        if (tid == 0 && !dry)
            cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(
                &bars.rx_full[slot & 1], kSlotBytes);
    }

    template <int Src, int SrcRows, int N>
    CUTE_DEVICE void gather(int slot, int token, int row0, float (&vals)[N],
                            bool dry) const {
        const int buf = slot & 1;
        if (!dry)
            mbar_wait_cluster(bar_addr(&bars.rx_full[buf]),
                              (uint32_t)((slot >> 1) & 1));
        const float* block = rx + buf * kBufFloats;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < N; ++i) vals[i] = block[at(row0 + i, token)];
        CUTLASS_PRAGMA_UNROLL
        for (int src = 1; src < Src; ++src) {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < N; ++i)
                vals[i] += block[at(src * SrcRows + row0 + i, token)];
        }
    }

    CUTE_DEVICE void release_rx(int slot, bool dry) const {
        bar_sync<typename Sync::Wg2Sync>();
        if (tid < Sh::Ranks && !dry)
            mbar_arrive_cluster_relaxed(cute::set_block_rank(
                bar_addr(&bars.rx_free[slot & 1]), (uint32_t)tid));
    }

    CUTE_DEVICE int slot_valid(int slot) const {
        return max(0,
                   min(Sh::Exchange, params.num_tokens - slot * Sh::Exchange));
    }

    CUTE_DEVICE void store_olora(int row, int sub,
                                 const float (&vals)[kTaskRows],
                                 const Ue8m0Scale& scale) const {
        const int col = coord.tile_a * OutputProjLayout::RowTile +
                        kHalfRows * coord.half + kTaskRows * sub;
        uint32_t words[kTaskRows / 4];
        pack_e4m3(vals, scale.multiplier, words);
        *reinterpret_cast<uint4*>(reinterpret_cast<uint8_t*>(params.olora_fp8) +
                                  (int64_t)row * params.shape.olora + col) =
            make_uint4(words[0], words[1], words[2], words[3]);
        if ((sub & 1) == 0) {
            cute::Tensor sf =
                cute::make_tensor(cute::make_gmem_ptr(params.olora_sf),
                                  sf_layout(params.wo_b.act));
            sf(row, col, cute::_0{}) = (uint8_t)scale.biased_exp;
        }
    }

    CUTE_DEVICE void finish_a(int slot, bool dry) {
        const int token = tid % Sh::Exchange, sub = tid / Sh::Exchange;
        const bool active = sub < kHalfRows / kTaskRows;
        expect(slot, dry);
        float vals[kTaskRows];
        gather<2, kHalfRows>(slot, token, active ? kTaskRows * sub : 0, vals,
                             dry);
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kTaskRows; ++i)
            vals[i] = __bfloat162float(__float2bfloat16(vals[i]));
        float amax = abs_max(vals);
        bars.amax[tid] = amax;
        bar_sync<typename Sync::Wg2Sync>();
        if (active)
            amax = fmaxf(amax, bars.amax[(sub ^ 1) * Sh::Exchange + token]);
        const Ue8m0Scale scale = ue8m0_scale(amax);
        if (active && token < slot_valid(slot) && !dry)
            store_olora(slot * Sh::Exchange + token, sub, vals, scale);
        if (slot == Sh::Slots - 1 && !dry) {
            __threadfence();
            fence_proxy_async_global();
        }
        release_rx(slot, dry);
        if (tid == KernelBase::WarpSize && slot == Sh::Slots - 1 && !dry)
            gmem_red_add1_relaxed(params.flags + coord.tile_a);
    }

    CUTE_DEVICE void store_out(int row, int chunk,
                               const float (&vals)[kRowsPerTask]) const {
        const int col = coord.cluster * OutputProjLayout::RowTile +
                        Sh::RankRows * coord.rank + kRowsPerTask * chunk;
        uint32_t words[kRowsPerTask / 2];
        pack_bf16(vals, words);
        cutlass::bfloat16_t* dst =
            params.out + (int64_t)row * params.shape.hidden + col;
        if constexpr (kRowsPerTask >= 8) {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < kRowsPerTask / 8; ++i)
                reinterpret_cast<uint4*>(dst)[i] =
                    make_uint4(words[4 * i], words[4 * i + 1], words[4 * i + 2],
                               words[4 * i + 3]);
        } else {
            *reinterpret_cast<uint2*>(dst) = make_uint2(words[0], words[1]);
        }
    }

    CUTE_DEVICE void finish_b(int slot, bool dry) {
        const int token = tid % Sh::Exchange, chunk = tid / Sh::Exchange;
        expect(slot, dry);
        float vals[kRowsPerTask];
        gather<Sh::Ranks, Sh::RankRows>(slot, token, kRowsPerTask * chunk, vals,
                                        dry);
        if (token < slot_valid(slot - Sh::Slots) && !dry)
            store_out((slot - Sh::Slots) * Sh::Exchange + token, chunk, vals);
        release_rx(slot, dry);
    }

    template <class Res>
    CUTE_DEVICE void run_sender(const Res& res) {
        auto op = make_drain_op<typename Sh::Cfg>(res, 0);
#pragma unroll 1
        for (int slot = -1; slot < 2 * Sh::Slots; ++slot)
            send(op, slot < 0 ? 0 : slot, slot < 0);
        if (tid == 0) bulk_wait_all();
    }

    CUTE_DEVICE void run_gatherer() {
#pragma unroll 1
        for (int slot = -2; slot < 2 * Sh::Slots; ++slot) {
            const bool dry = slot < 0;
            if (slot == -1 || (slot >= 0 && slot < Sh::Slots))
                finish_a(dry ? 0 : slot, dry);
            if (slot == -2 || slot >= Sh::Slots)
                finish_b(dry ? Sh::Slots : slot, dry);
        }
    }
};

template <int TT, class Prm, class Res, class BarsT>
CUTE_DEVICE void reduce_role(const Prm& params, const Res& res_a, BarsT& bars,
                             char* smem, const Coord& coord, int warp) {
    using Roles = KernelBase::Roles;
    using Sm = Smem<TT>;
    pdl_wait();
    const bool sender = Roles::is_drain(warp);
    float* rx = reinterpret_cast<float*>(smem + Sm::RxOff);
    float* staging = reinterpret_cast<float*>(smem + Sm::StagingOff);
    Reduce<TT, Prm, BarsT> reduce{
        .params = params,
        .bars = bars,
        .coord = coord,
        .rx = rx,
        .staging = staging,
        .rx_smem = cute::cast_smem_ptr_to_uint(rx),
        .staging_smem = cute::cast_smem_ptr_to_uint(staging),
        .tid = (int)(threadIdx.x -
                     (sender ? Roles::DrainWarpBegin : Roles::VecWarpBegin) *
                         KernelBase::WarpSize)};
    if (sender)
        reduce.run_sender(res_a);
    else
        reduce.run_gatherer();
}

template <int TT, class Prm>
__global__ void __launch_bounds__(KernelBase::NumThreads, 1)
    mla_output_proj_swapab_kernel(__grid_constant__ const Prm params) {
    using Sh = Shape<TT>;
    using Cfg = typename Sh::Cfg;
    using Sm = Smem<TT>;
    using Roles = KernelBase::Roles;
    extern __shared__ __align__(1024) char smem[];
    auto& bars = *reinterpret_cast<typename Sm::Bars*>(smem + Sm::BarOff);
    const int warp = KernelBase::warp_idx();
    const int lane = KernelBase::lane_idx();
    const Coord coord(params.shape);

    if (warp == 0) {
        if (cute::elect_one_sync()) {
            for (int i = 0; i < Sh::Stages; ++i) {
                bars.tma_a[i].init(1);
                bars.tma_b[i].init(1);
                bars.smem_free[i].init(1);
            }
            for (int i = 0; i < 2; ++i) {
                bars.acc_full[i].init(1);
                bars.acc_empty[i].init(Roles::DrainWarps *
                                       KernelBase::WarpSize);
                cutlass::arch::ClusterBarrier::init(&bars.rx_full[i], 1);
                cutlass::arch::ClusterBarrier::init(&bars.rx_free[i],
                                                    Sh::Ranks);
            }
            cutlass::arch::fence_barrier_init();
        }
        __syncwarp();
        cute::TMEM::Allocator1Sm().allocate(Sm100::TmemColumns,
                                            &bars.tmem_slot);
        cute::TMEM::Allocator1Sm().release_allocation_lock();
    }
    cute::cluster_sync();
    pdl_trigger();

    const GemmPipelineBars pipeline{
        bars.tma_a,     bars.tma_b,    bars.smem_free, bars.acc_full,
        bars.acc_empty, bars.acc_full, bars.acc_empty};
    auto tmem = make_gemm_tmem_for<Cfg>(0u);
    using Res = GemmResources<typename Cfg::Geom, decltype(tmem)>;
    const Res res_a{.weight = params.wo_a.weight,
                    .act = params.wo_a.act,
                    .gemm_smem = smem,
                    .tmem = tmem,
                    .bars = pipeline};
    const Res res_b{.weight = params.wo_b.weight,
                    .act = params.wo_b.act,
                    .gemm_smem = smem,
                    .tmem = tmem,
                    .bars = pipeline};

    if (warp == Roles::MmaWarp) {
        mma_role<TT>(params, res_a, res_b, coord, lane);
    } else if (warp == Roles::Tma1Warp) {
        weight_role<TT>(params, res_a, res_b, pipeline, coord);
    } else if (warp == Roles::Tma2Warp) {
        act_role<TT>(params, res_a, res_b, pipeline, coord, lane);
    } else if (Roles::is_drain(warp) || Roles::is_vec(warp)) {
        reduce_role<TT>(params, res_a, bars, smem, coord, warp);
    }

    __syncwarp();
    cute::cluster_sync();
    if (warp == 0) cute::TMEM::Allocator1Sm().free(0, Sm100::TmemColumns);
}

}  // namespace halops::mla::swapab
