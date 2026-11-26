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

#include "kernel/mla/mega_mla/gemm/bf16.cuh"

namespace halops::mla {
template <class Cfg, class Res>
struct Bf16TmaOpA : TmaStageWalk<Bf16TmaOpA<Cfg, Res>, Cfg::TmaStages> {
    using Types = Bf16GemmTypes<Cfg>;

    const Res& res;
    int w_tile;

    CUTE_DEVICE void begin_item(const Bf16GemmItem& it) { w_tile = it.w_tile; }

    CUTE_DEVICE auto issuer() {
        return [this](auto St, int k) {
            constexpr int stage = decltype(St)::value;
            res.bars.tma_a[stage].arrive_and_expect_tx(Types::SmemA);
            uint64_t& mbar = reinterpret_cast<uint64_t&>(res.bars.tma_a[stage]);
            auto mA = res.tma_w.get_tma_tensor(
                cute::make_shape(res.w_out, res.k_dim));
            auto thr_w = res.tma_w.get_slice(cute::_0{});
            auto tA = cute::local_tile(mA, typename Types::TileA{},
                                       cute::make_coord(w_tile, k));
            cute::copy(
                res.tma_w.with(mbar, 0), thr_w.partition_S(tA),
                thr_w.partition_D(bf16_smem_a<Cfg, stage>(res.gemm_smem)));
        };
    }
};

template <class Cfg, class Res>
struct Bf16TmaOpB : TmaStageWalk<Bf16TmaOpB<Cfg, Res>, Cfg::TmaStages> {
    using Types = Bf16GemmTypes<Cfg>;

    const Res& res;
    int n_tile;

    CUTE_DEVICE void begin_item(const Bf16GemmItem& it) { n_tile = it.n_tile; }

    CUTE_DEVICE auto issuer() {
        return [this](auto St, int k) {
            constexpr int stage = decltype(St)::value;
            res.bars.tma_b[stage].arrive_and_expect_tx(Types::SmemB);
            uint64_t& mbar = reinterpret_cast<uint64_t&>(res.bars.tma_b[stage]);
            auto mB = res.tma_act.get_tma_tensor(
                cute::make_shape(res.act_rows, res.k_dim));
            auto thr_act = res.tma_act.get_slice(cute::_0{});
            auto tB = cute::local_tile(mB, typename Types::TileB{},
                                       cute::make_coord(n_tile, k));
            cute::copy(
                res.tma_act.with(mbar, 0), thr_act.partition_S(tB),
                thr_act.partition_D(bf16_smem_b<Cfg, stage>(res.gemm_smem)));
        };
    }
};

template <class Cfg, class Res>
struct Bf16MmaOp : MmaStageWalk<Cfg::TmaStages, Cfg::AccStages,
                                Bf16GemmTypes<Cfg>::Overlap> {
    using Types = Bf16GemmTypes<Cfg>;

    const Res& res;
    typename Types::TiledMma tiled_mma;

    CUTE_DEVICE void begin_item() { this->acquire_acc(); }

    CUTE_DEVICE void run_k_tile(int k) {
        tcgen05_after_thread_sync();
        if (!this->elected) return;
        dispatch_stage<Cfg::TmaStages>(this->smem_stage(k), stager(), k);
    }

    CUTE_DEVICE auto stager() {
        return [this](auto St, int k) {
            constexpr int stage = decltype(St)::value;
            cute::Tensor tmem_frag = cute::partition_fragment_C(
                tiled_mma,
                cute::Shape<cute::Int<Cfg::BM>, cute::Int<Cfg::BN>>{});
            tmem_frag.data().get() =
                res.tmem_base + (uint32_t)(this->acc_stage * Cfg::AccStride);
            umma_ss(tiled_mma, bf16_smem_a<Cfg, stage>(res.gemm_smem),
                    bf16_smem_b<Cfg, stage>(res.gemm_smem), tmem_frag, k == 0);
        };
    }
};

template <class Cfg, class Res, class TiledT2R, class ThrTacc, class Racc,
          class Coord>
struct Bf16DrainOp : DrainStageWalk<Cfg::AccStages> {
    using Types = Bf16GemmTypes<Cfg>;
    using Walk = DrainStageWalk<Cfg::AccStages>;

    const Res& res;
    TiledT2R tiled_t2r;
    ThrTacc thr_tacc;
    Racc thr_racc;
    Coord coord;
    uint32_t acc_addr;

    CUTE_DEVICE void wait_acc_full() {
        Walk::wait_acc_full();
        acc_addr = thr_tacc.data().get() +
                   (uint32_t)(this->acc_stage * Cfg::AccStride);
    }

    CUTE_DEVICE void store_ring(const Bf16GemmItem& it, int i,
                                float value) const {
        const int row = cute::get<0>(coord(i));
        const int act_row = cute::get<1>(coord(i));
        if (act_row < it.n_valid) {
            const int ring_row = res.ring_rows[it.n_tile * Cfg::BN + act_row];
            res.ring[(int64_t)ring_row * res.w_out + it.w_tile * Cfg::BM +
                     row] = value;
        }
    }

    CUTE_DEVICE void release_windows() const {
        arrive_acc_window_lo<Types::Overlap>(this->bars, this->acc_stage);
        arrive_acc_window_hi<Types::Overlap>(this->bars, this->acc_stage);
    }

    CUTE_DEVICE void store_unrolled(const Bf16GemmItem& it) {
        ThrTacc src = thr_tacc;
        src.data() = acc_addr;
        cute::copy(tiled_t2r, src, thr_racc);
        cutlass::arch::fence_view_async_tmem_load();
        release_windows();
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < cute::size(thr_racc); ++i)
            store_ring(it, i, thr_racc(i));
    }

    CUTE_DEVICE void store_rolled(const Bf16GemmItem& it) const {
        constexpr int kChunk =
            cute::size(Racc{}) < kTmemLd ? (int)cute::size(Racc{}) : kTmemLd;
        constexpr int kNumChunks = cute::size(Racc{}) / kChunk;
        float cur[kChunk], nxt[kChunk];
        tmem_ld_32dp32bNx<kChunk>(acc_addr, nxt);
        cutlass::arch::fence_view_async_tmem_load();
        CUTE_NO_UNROLL
        for (int ch = 0; ch < kNumChunks; ++ch) {
            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < kChunk; ++j) cur[j] = nxt[j];
            if (ch + 1 < kNumChunks)
                tmem_ld_32dp32bNx<kChunk>(acc_addr + (ch + 1) * kChunk, nxt);
            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < kChunk; ++j)
                store_ring(it, ch * kChunk + j, cur[j]);
            cutlass::arch::fence_view_async_tmem_load();
        }
        release_windows();
    }
};

template <class Cfg, class Res>
CUTE_DEVICE auto make_bf16_weight_tma_op(const Res& res, int base_tile) {
    return Bf16TmaOpA<Cfg, Res>{
        {res.bars, cute::elect_one_sync() != 0, base_tile, 0, res.k_tiles},
        res,
        0};
}

template <class Cfg, class Res>
CUTE_DEVICE auto make_bf16_act_tma_op(const Res& res, int base_tile) {
    return Bf16TmaOpB<Cfg, Res>{
        {res.bars, cute::elect_one_sync() != 0, base_tile, 0, res.k_tiles},
        res,
        0};
}

template <class Cfg, class Res>
CUTE_DEVICE auto make_bf16_mma_op(const Res& res, int base_tile, int base_acc) {
    return Bf16MmaOp<Cfg, Res>{{res.bars, cute::elect_one_sync() != 0,
                                base_tile, 0, res.k_tiles, base_acc, 0, 0},
                               res,
                               typename Bf16GemmTypes<Cfg>::TiledMma{}};
}

template <class Cfg, class Res>
CUTE_DEVICE auto make_bf16_drain_op(const Res& res, int base_acc) {
    using Types = Bf16GemmTypes<Cfg>;
    using Roles = KernelBase::Roles;

    typename Types::TiledMma tiled_mma;

    cute::Tensor frag0 = cute::partition_fragment_C(
        tiled_mma, cute::Shape<cute::Int<Cfg::BM>, cute::Int<Cfg::BN>>{});
    frag0.data().get() = res.tmem_base;

    cute::Tensor acc0 =
        frag0(cute::make_coord(cute::_, cute::_), cute::_0{}, cute::_0{});
    cute::TiledCopy tiled_t2r =
        cute::make_tmem_copy(typename Types::CopyOpT2R{}, acc0);

    const int warp = KernelBase::warp_idx();
    const int lane = KernelBase::lane_idx();
    cute::ThrCopy thr_t2r = tiled_t2r.get_slice(
        (warp % Roles::DrainWarps) * KernelBase::WarpSize + lane);
    cute::Tensor t2r_dst = thr_t2r.partition_D(acc0);
    cute::Tensor thr_racc = cute::make_tensor<float>(cute::shape(t2r_dst));
    cute::Tensor coord =
        thr_t2r.partition_D(cute::make_identity_tensor(cute::shape(acc0)));

    using Op = Bf16DrainOp<Cfg, Res, decltype(tiled_t2r),
                           decltype(thr_t2r.partition_S(acc0)),
                           decltype(thr_racc), decltype(coord)>;

    return Op{{res.bars, base_acc, 0, 0},
              res,
              tiled_t2r,
              thr_t2r.partition_S(acc0),
              thr_racc,
              coord,
              0};
}

template <class Op, class Begin>
CUTE_DEVICE void issue_items(Op& op, int first, int end, int step, int k_tiles,
                             Begin&& begin) {
    for (int item = first; item < end; item += step) {
        begin(item);
        CUTLASS_PRAGMA_NO_UNROLL
        for (int k = 0; k < k_tiles; ++k) op.issue_tile(k);
        op.end_item();
    }
}

template <class Op, class Begin>
CUTE_DEVICE void mma_items(Op& op, int first, int end, int step, int k_tiles,
                           Begin&& begin) {
    for (int item = first; item < end; item += step) {
        begin(item);
        CUTLASS_PRAGMA_NO_UNROLL
        for (int k = 0; k < k_tiles; ++k) {
            op.wait_tma(k);
            op.run_k_tile(k);
            op.commit(k);
        }
        op.end_item();
    }
}

template <class Op, class Store, class Publish>
CUTE_DEVICE void drain_items(Op& op, int first, int end, int step,
                             Store&& store, Publish&& publish) {
    for (int item = first; item < end; item += step) {
        op.wait_acc_full();
        const auto stored = store(item);
        op.end_item();
        publish(item, stored);
    }
}
}  // namespace halops::mla
