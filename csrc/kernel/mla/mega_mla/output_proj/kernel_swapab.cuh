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

#include "kernel/mla/mega_mla/output_proj/params.cuh"

namespace halops::mla {

template <class C>
CUTE_DEVICE int oproj_swapab_wo_a_k_tiles(const OutputProjShape& shape) {
    return shape.group_k / C::BK / 2;
}
template <class C>
CUTE_DEVICE int oproj_swapab_wo_b_k_tiles(const OutputProjShape& shape) {
    return shape.olora / C::BK / C::Ranks;
}

template <class C>
CUTE_DEVICE GemmWorkItem oproj_swapab_item(int tile, int rows, int num_tokens) {
    return {.n_row = 0,
            .w_tile = tile,
            .w_out = rows,
            .k_tile_begin = 0,
            .a_k_tile_begin = 0,
            .out_row_stride = rows,
            .n_valid = min(C::TT, num_tokens),
            .w_valid = OutputProjLayout::RowTile};
}

template <class C>
CUTE_DEVICE void oproj_swapab_prefetch_wo_b(const uint8_t* wo_b, int olora,
                                            int cluster, int rank, int lane) {
    const uint8_t* row = wo_b + (int64_t)(cluster * OutputProjLayout::RowTile +
                                          C::RankRows * rank + lane) *
                                    olora;
    if (olora > C::kPrefetchSkip)
        prefetch_l2_bulk(row + C::kPrefetchSkip, olora - C::kPrefetchSkip);
}

template <class Op>
CUTE_DEVICE void oproj_swapab_mma_item(Op& op, const GemmWorkItem& item,
                                       int k_tiles) {
    op.begin_item(item);
    CUTLASS_PRAGMA_NO_UNROLL
    for (int k = 0; k < k_tiles; ++k) {
        op.wait_tma(k);
        op.run_k_tile(k);
        op.commit(k);
    }
    op.end_item();
}

template <class C, class Prm, class Res>
CUTE_DEVICE void oproj_swapab_mma_role(const Prm& params, const Res& res_a,
                                       const Res& res_b,
                                       const OprojSwapabCoord& coord,
                                       int lane) {
    using Gemm = typename C::Gemm;
    const int k_a = oproj_swapab_wo_a_k_tiles<C>(params.shape);
    const int k_b = oproj_swapab_wo_b_k_tiles<C>(params.shape);
    auto op_a = make_mma_op<Gemm>(res_a, 0, 0, k_a);
    oproj_swapab_mma_item(op_a,
                          oproj_swapab_item<C>(coord.tile_a, params.shape.olora,
                                               params.num_tokens),
                          k_a);
    oproj_swapab_prefetch_wo_b<C>(params.wo_b_raw, params.shape.olora,
                                  coord.cluster, coord.rank, lane);
    auto op_b = make_mma_op<Gemm>(res_b, k_a, 1, k_b);
    oproj_swapab_mma_item(
        op_b,
        oproj_swapab_item<C>(coord.cluster, params.shape.hidden,
                             params.num_tokens),
        k_b);
}

template <class C, class Prm, class Res>
CUTE_DEVICE void oproj_swapab_weight_role(const Prm& params, const Res& res_a,
                                          const Res& res_b,
                                          const GemmPipelineBars& bars,
                                          const OprojSwapabCoord& coord) {
    using Gemm = typename C::Gemm;
    const int k_a = oproj_swapab_wo_a_k_tiles<C>(params.shape);
    const int k_b = oproj_swapab_wo_b_k_tiles<C>(params.shape);
    auto op_a = make_tma_op_a<Gemm>(res_a, 0, k_a, NoGate{});
    op_a.begin_item(params.wo_a.weight,
                    oproj_swapab_item<C>(coord.tile_a, params.shape.olora,
                                         params.num_tokens));
    for (int k = 0; k < k_a; ++k) {
        wait_smem_stage_free<C::Stages>(bars, k);
        if (op_a.elected)
            dispatch_stage<C::Stages>(k % C::Stages, op_a.issuer(),
                                      coord.half * k_a + k);
    }
    auto op_b = make_tma_op_a<Gemm>(res_b, k_a, k_b, NoGate{});
    op_b.begin_item(params.wo_b.weight,
                    oproj_swapab_item<C>(coord.cluster, params.shape.hidden,
                                         params.num_tokens));
    for (int k = 0; k < k_b; ++k) {
        const int seq = k_a + k;
        wait_smem_stage_free<C::Stages>(bars, seq);
        if (op_b.elected)
            dispatch_stage<C::Stages>(seq % C::Stages, op_b.issuer(),
                                      C::Ranks * k + coord.rank);
    }
}

template <class C, class Prm, class Res>
CUTE_DEVICE void oproj_swapab_act_role(const Prm& params, const Res& res_a,
                                       const Res& res_b,
                                       const GemmPipelineBars& bars,
                                       const OprojSwapabCoord& coord,
                                       int lane) {
    using Gemm = typename C::Gemm;
    const int k_a = oproj_swapab_wo_a_k_tiles<C>(params.shape);
    const int k_b = oproj_swapab_wo_b_k_tiles<C>(params.shape);
    const int tiles_per_group =
        params.shape.group_rows / OutputProjLayout::RowTile;
    pdl_wait();
    auto op_a = make_tma_op_b<Gemm>(res_a, 0, k_a, NoGate{});
    const int out_group = coord.tile_a / tiles_per_group;
    op_a.begin_item(params.wo_a.weight,
                    oproj_swapab_item<C>(coord.tile_a, params.shape.olora,
                                         params.num_tokens));
    for (int k = 0; k < k_a; ++k) {
        wait_smem_stage_free<C::Stages>(bars, k);
        if (op_a.elected)
            dispatch_stage<C::Stages>(
                k % C::Stages, op_a.issuer(),
                out_group * (2 * k_a) + coord.half * k_a + k);
    }
    auto op_b = make_tma_op_b<Gemm>(res_b, k_a, k_b, NoGate{});
    op_b.begin_item(params.wo_b.weight,
                    oproj_swapab_item<C>(coord.cluster, params.shape.hidden,
                                         params.num_tokens));
    uint32_t ready = 0;
    for (int k = 0; k < k_b; ++k) {
        const int seq = k_a + k;
        while (!((ready >> k) & 1)) {
            bool done = true;
            if (lane < k_b) {
                const int k_block = C::Ranks * lane + coord.rank;
                CUTLASS_PRAGMA_UNROLL
                for (int tile = 0; tile < C::TilesPerKBlock; ++tile)
                    done = done && gmem_ld_acquire(params.flags +
                                                   C::TilesPerKBlock * k_block +
                                                   tile) >= C::TileArrivals;
            }
            ready = __ballot_sync(~0u, done && lane < k_b);
        }
        __syncwarp();
        fence_proxy_async_global();
        wait_smem_stage_free<C::Stages>(bars, seq);
        if (op_b.elected)
            dispatch_stage<C::Stages>(seq % C::Stages, op_b.issuer(),
                                      C::Ranks * k + coord.rank);
    }
}

template <class C>
CUTE_DEVICE int oproj_swapab_at(int row, int token) {
    return swizzled_token_index<C::Exchange>(row, token);
}
CUTE_DEVICE uint32_t oproj_swapab_bar_addr(const uint64_t* bar) {
    return cute::cast_smem_ptr_to_uint(bar);
}

template <class C, class Prm, class Op>
CUTE_DEVICE void oproj_swapab_stage(const OprojSwapabReduce<C, Prm>& r, Op& op,
                                    int slot, bool dry) {
    using Sync = KernelBase::Bars;
    constexpr int kV = C::kVecFloats;
    const int part = slot % C::Slots;
    float* buf = r.staging + (slot & 1) * C::kBufFloats;
    if (slot >= 2) {
        if (r.tid == 0) cute::tma_store_wait<1>();
        bar_sync<typename Sync::Wg1Sync>();
    }
    if (!dry) {
        if (part == 0) op.wait_acc_full();
        op.load_slice(part);
    }
    const int a_row = cute::get<0>(op.coord(0));
    CUTLASS_PRAGMA_UNROLL
    for (int c4 = 0; c4 < C::Exchange / kV; ++c4)
        *reinterpret_cast<float4*>(buf + oproj_swapab_at<C>(a_row, kV * c4)) =
            make_float4(op.thr_racc(kV * c4), op.thr_racc(kV * c4 + 1),
                        op.thr_racc(kV * c4 + 2),
                        op.thr_racc(kV * c4 + kV - 1));
    if (part == C::Slots - 1 && !dry) op.end_item();
    cutlass::arch::fence_view_async_shared();
    bar_sync<typename Sync::Wg1Sync>();
}

template <class C, class Prm>
CUTE_DEVICE void oproj_swapab_copy_rows(const OprojSwapabReduce<C, Prm>& r,
                                        int buf, uint32_t dst, int dst_row,
                                        int src_row, int rows) {
    bulk_copy_to_peer(
        cute::set_block_rank(
            r.rx_smem + (uint32_t)(buf * OutputProjLayout::RowTile + dst_row) *
                            C::kRowBytes,
            dst),
        r.staging_smem + (uint32_t)(buf * OutputProjLayout::RowTile + src_row) *
                             C::kRowBytes,
        (uint32_t)rows * C::kRowBytes,
        cute::set_block_rank(oproj_swapab_bar_addr(&r.bars.rx_full[buf]), dst));
}

template <class C, class Prm>
CUTE_DEVICE void oproj_swapab_copy_to_owners(const OprojSwapabReduce<C, Prm>& r,
                                             int slot) {
    const int buf = slot & 1;
    if (slot >= 2)
        mbar_wait_cluster(oproj_swapab_bar_addr(&r.bars.rx_free[buf]),
                          (uint32_t)(((slot >> 1) - 1) & 1));
    if (slot < C::Slots) {
        for (int dst_half = 0; dst_half < 2; ++dst_half)
            oproj_swapab_copy_rows(r, buf,
                                   (uint32_t)((r.coord.rank & ~1) | dst_half),
                                   r.coord.half * C::kHalfRows,
                                   C::kHalfRows * dst_half, C::kHalfRows);
    } else {
        for (int dst = 0; dst < C::Ranks; ++dst)
            oproj_swapab_copy_rows(r, buf, (uint32_t)dst,
                                   r.coord.rank * C::RankRows,
                                   C::RankRows * dst, C::RankRows);
    }
    cute::tma_store_arrive();
}

template <class C, class Prm, class Op>
CUTE_DEVICE void oproj_swapab_send(const OprojSwapabReduce<C, Prm>& r, Op& op,
                                   int slot, bool dry) {
    oproj_swapab_stage(r, op, slot, dry);
    if (r.tid == 0 && !dry) oproj_swapab_copy_to_owners(r, slot);
}

template <class C, class Prm>
CUTE_DEVICE void oproj_swapab_expect(const OprojSwapabReduce<C, Prm>& r,
                                     int slot, bool dry) {
    if (r.tid == 0 && !dry)
        cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(
            &r.bars.rx_full[slot & 1], C::kSlotBytes);
}

template <int Src, int SrcRows, class C, class Prm, int N>
CUTE_DEVICE void oproj_swapab_gather(const OprojSwapabReduce<C, Prm>& r,
                                     int slot, int token, int row0,
                                     float (&vals)[N], bool dry) {
    const int buf = slot & 1;
    if (!dry)
        mbar_wait_cluster(oproj_swapab_bar_addr(&r.bars.rx_full[buf]),
                          (uint32_t)((slot >> 1) & 1));
    const float* block = r.rx + buf * C::kBufFloats;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i)
        vals[i] = block[oproj_swapab_at<C>(row0 + i, token)];
    CUTLASS_PRAGMA_UNROLL
    for (int src = 1; src < Src; ++src) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < N; ++i)
            vals[i] +=
                block[oproj_swapab_at<C>(src * SrcRows + row0 + i, token)];
    }
}

template <class C, class Prm>
CUTE_DEVICE void oproj_swapab_release_rx(const OprojSwapabReduce<C, Prm>& r,
                                         int slot, bool dry) {
    bar_sync<typename KernelBase::Bars::Wg2Sync>();
    if (r.tid < C::Ranks && !dry)
        mbar_arrive_cluster_relaxed(cute::set_block_rank(
            oproj_swapab_bar_addr(&r.bars.rx_free[slot & 1]), (uint32_t)r.tid));
}

template <class C, class Prm>
CUTE_DEVICE int oproj_swapab_slot_valid(const OprojSwapabReduce<C, Prm>& r,
                                        int slot) {
    return max(0, min(C::Exchange, r.params.num_tokens - slot * C::Exchange));
}

template <class C, class Prm>
CUTE_DEVICE void oproj_swapab_store_olora(const OprojSwapabReduce<C, Prm>& r,
                                          int row, int sub,
                                          const float (&vals)[C::kTaskRows],
                                          const Ue8m0Scale& scale) {
    const int col = r.coord.tile_a * OutputProjLayout::RowTile +
                    C::kHalfRows * r.coord.half + C::kTaskRows * sub;
    uint32_t words[C::kTaskRows / C::kCodesPerWord];
    pack_e4m3(vals, scale.multiplier, words);
    *reinterpret_cast<uint4*>(reinterpret_cast<uint8_t*>(r.params.olora_fp8) +
                              (int64_t)row * r.params.shape.olora + col) =
        make_uint4(words[0], words[1], words[2], words[C::kVecWords - 1]);
    if ((sub & 1) == 0) {
        cute::Tensor sf =
            cute::make_tensor(cute::make_gmem_ptr(r.params.olora_sf),
                              sf_layout(r.params.wo_b.act));
        sf(row, col, cute::Int<0>{}) = (uint8_t)scale.biased_exp;
    }
}

template <class C, class Prm>
CUTE_DEVICE void oproj_swapab_finish_a(const OprojSwapabReduce<C, Prm>& r,
                                       int slot, bool dry) {
    const int token = r.tid % C::Exchange, sub = r.tid / C::Exchange;
    const bool active = sub < C::kHalfRows / C::kTaskRows;
    oproj_swapab_expect(r, slot, dry);
    float vals[C::kTaskRows];
    oproj_swapab_gather<2, C::kHalfRows>(
        r, slot, token, active ? C::kTaskRows * sub : 0, vals, dry);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < C::kTaskRows; ++i)
        vals[i] = __bfloat162float(__float2bfloat16(vals[i]));
    float amax = abs_max(vals);
    r.bars.amax[r.tid] = amax;
    bar_sync<typename KernelBase::Bars::Wg2Sync>();
    if (active)
        amax = fmaxf(amax, r.bars.amax[(sub ^ 1) * C::Exchange + token]);
    const Ue8m0Scale scale = ue8m0_scale(amax);
    if (active && token < oproj_swapab_slot_valid(r, slot) && !dry)
        oproj_swapab_store_olora(r, slot * C::Exchange + token, sub, vals,
                                 scale);
    if (slot == C::Slots - 1 && !dry) {
        __threadfence();
        fence_proxy_async_global();
    }
    oproj_swapab_release_rx(r, slot, dry);
    if (r.tid == KernelBase::WarpSize && slot == C::Slots - 1 && !dry)
        gmem_red_add1_relaxed(r.params.flags + r.coord.tile_a);
}

template <class C, class Prm>
CUTE_DEVICE void oproj_swapab_store_out(const OprojSwapabReduce<C, Prm>& r,
                                        int row, int chunk,
                                        const float (&vals)[C::kRowsPerTask]) {
    const int col = r.coord.cluster * OutputProjLayout::RowTile +
                    C::RankRows * r.coord.rank + C::kRowsPerTask * chunk;
    uint32_t words[C::kRowsPerTask / 2];
    pack_bf16(vals, words);
    cutlass::bfloat16_t* dst =
        r.params.out + (int64_t)row * r.params.shape.hidden + col;
    if constexpr (C::kRowsPerTask >= C::kBfPerVec) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < C::kRowsPerTask / C::kBfPerVec; ++i)
            reinterpret_cast<uint4*>(dst)[i] =
                make_uint4(words[C::kVecWords * i], words[C::kVecWords * i + 1],
                           words[C::kVecWords * i + 2],
                           words[C::kVecWords * i + C::kVecWords - 1]);
    } else {
        *reinterpret_cast<uint2*>(dst) = make_uint2(words[0], words[1]);
    }
}

template <class C, class Prm>
CUTE_DEVICE void oproj_swapab_finish_b(const OprojSwapabReduce<C, Prm>& r,
                                       int slot, bool dry) {
    const int token = r.tid % C::Exchange, chunk = r.tid / C::Exchange;
    oproj_swapab_expect(r, slot, dry);
    float vals[C::kRowsPerTask];
    oproj_swapab_gather<C::Ranks, C::RankRows>(
        r, slot, token, C::kRowsPerTask * chunk, vals, dry);
    if (token < oproj_swapab_slot_valid(r, slot - C::Slots) && !dry)
        oproj_swapab_store_out(r, (slot - C::Slots) * C::Exchange + token,
                               chunk, vals);
    oproj_swapab_release_rx(r, slot, dry);
}

template <class C, class Prm, class Res>
CUTE_DEVICE void oproj_swapab_run_sender(const OprojSwapabReduce<C, Prm>& r,
                                         const Res& res) {
    auto op = make_drain_op<typename C::Gemm>(res, 0);
#pragma unroll 1
    for (int slot = -1; slot < 2 * C::Slots; ++slot)
        oproj_swapab_send(r, op, slot < 0 ? 0 : slot, slot < 0);
    if (r.tid == 0) bulk_wait_all();
}

template <class C, class Prm>
CUTE_DEVICE void oproj_swapab_run_gatherer(const OprojSwapabReduce<C, Prm>& r) {
#pragma unroll 1
    for (int slot = -2; slot < 2 * C::Slots; ++slot) {
        const bool dry = slot < 0;
        if (slot == -1 || (slot >= 0 && slot < C::Slots))
            oproj_swapab_finish_a(r, dry ? 0 : slot, dry);
        if (slot == -2 || slot >= C::Slots)
            oproj_swapab_finish_b(r, dry ? C::Slots : slot, dry);
    }
}

template <class C, class Prm, class Res>
CUTE_DEVICE void oproj_swapab_reduce_role(const Prm& params, const Res& res_a,
                                          typename C::Bars& bars, char* smem,
                                          const OprojSwapabCoord& coord,
                                          int warp) {
    using Roles = KernelBase::Roles;
    pdl_wait();
    const bool sender = Roles::is_drain(warp);
    float* rx = reinterpret_cast<float*>(smem + C::RxOff);
    float* staging = reinterpret_cast<float*>(smem + C::StagingOff);
    const OprojSwapabReduce<C, Prm> reduce{
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
        oproj_swapab_run_sender(reduce, res_a);
    else
        oproj_swapab_run_gatherer(reduce);
}

/// Swap-AB output projection for at most 128 tokens: each CTA runs the wo_a
/// GEMM of half of one 128-row tile's K and the wo_b GEMM of one K quarter
/// with the weight in UMMA A, and the four CTAs of a cluster reduce their
/// partials through DSMEM before quantizing olora and storing the output.
template <class C, class Prm>
__global__ void __launch_bounds__(KernelBase::NumThreads, 1)
    mla_output_proj_swapab_kernel(__grid_constant__ const Prm params) {
    using Gemm = typename C::Gemm;
    using Roles = KernelBase::Roles;
    extern __shared__ __align__(Sm100::Sw128AtomBytes) char smem[];
    auto& bars = *reinterpret_cast<typename C::Bars*>(smem + C::BarOff);
    const int warp = KernelBase::warp_idx();
    const int lane = KernelBase::lane_idx();
    const OprojSwapabCoord coord(params.shape);

    if (warp == 0) {
        if (cute::elect_one_sync()) {
            for (int i = 0; i < C::Stages; ++i) {
                bars.tma_a[i].init(1);
                bars.tma_b[i].init(1);
                bars.smem_free[i].init(1);
            }
            for (int i = 0; i < 2; ++i) {
                bars.acc_full[i].init(1);
                bars.acc_empty[i].init(Roles::DrainWarps *
                                       KernelBase::WarpSize);
                cutlass::arch::ClusterBarrier::init(&bars.rx_full[i], 1);
                cutlass::arch::ClusterBarrier::init(&bars.rx_free[i], C::Ranks);
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
    auto tmem = make_gemm_tmem_for<Gemm>(0u);
    const GemmResources<typename Gemm::Geom, decltype(tmem)> res_a{
        .weight = params.wo_a.weight,
        .act = params.wo_a.act,
        .gemm_smem = smem,
        .tmem = tmem,
        .bars = pipeline};
    const GemmResources<typename Gemm::Geom, decltype(tmem)> res_b{
        .weight = params.wo_b.weight,
        .act = params.wo_b.act,
        .gemm_smem = smem,
        .tmem = tmem,
        .bars = pipeline};

    if (warp == Roles::MmaWarp) {
        oproj_swapab_mma_role<C>(params, res_a, res_b, coord, lane);
    } else if (warp == Roles::Tma1Warp) {
        oproj_swapab_weight_role<C>(params, res_a, res_b, pipeline, coord);
    } else if (warp == Roles::Tma2Warp) {
        oproj_swapab_act_role<C>(params, res_a, res_b, pipeline, coord, lane);
    } else if (Roles::is_drain(warp) || Roles::is_vec(warp)) {
        oproj_swapab_reduce_role<C>(params, res_a, bars, smem, coord, warp);
    }

    __syncwarp();
    cute::cluster_sync();
    if (warp == 0) cute::TMEM::Allocator1Sm().free(0, Sm100::TmemColumns);
}

}  // namespace halops::mla
