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

#include "kernel/mla/mega_mla/gemm/atoms.cuh"

namespace halops::mla {
template <class Acc, class SFA, class SFB>
struct GemmTmemStorage {
    Acc tmem_accumulators;
    SFA tmem_SFA;
    SFB tmem_SFB;
};

template <class Cfg, int AccStages = Cfg::AccStages>
CUTE_DEVICE auto make_gemm_tmem_for(uint32_t tmem_base) {
    using Inst = GemmInst<Cfg, Cfg::TmaStages, AccStages>;
    using Mainloop = typename Inst::Mainloop;
    auto tmem_storage = Mainloop::init_tmem_tensors();
    Mainloop::set_tmem_offsets(tmem_storage, tmem_base);
    tmem_storage.tmem_SFA.data() = tmem_base + (uint32_t)Cfg::SfBaseCols;
    tmem_storage.tmem_SFB.data() =
        tmem_storage.tmem_SFA.data().get() +
        cutlass::detail::find_tmem_tensor_col_offset(tmem_storage.tmem_SFA);
    if constexpr (Inst::OverlapCols == 0) {
        return GemmTmemStorage<decltype(tmem_storage.tmem_accumulators),
                               decltype(tmem_storage.tmem_SFA),
                               decltype(tmem_storage.tmem_SFB)>{
            tmem_storage.tmem_accumulators, tmem_storage.tmem_SFA,
            tmem_storage.tmem_SFB};
    } else {
        auto const& acc = tmem_storage.tmem_accumulators;
        auto acc_ring = cute::make_tensor(
            acc.data(),
            cute::make_layout(cute::replace<CommonGemmModes::StageMode>(
                                  acc.layout().shape(), cute::Int<AccStages>{}),
                              acc.layout().stride()));
        return GemmTmemStorage<decltype(acc_ring),
                               decltype(tmem_storage.tmem_SFA),
                               decltype(tmem_storage.tmem_SFB)>{
            acc_ring, tmem_storage.tmem_SFA, tmem_storage.tmem_SFB};
    }
}

struct GemmPipelineBars {
    cutlass::arch::ClusterTransactionBarrier* tma_a;
    cutlass::arch::ClusterTransactionBarrier* tma_b;
    cutlass::arch::ClusterTransactionBarrier* smem_free;
    cutlass::arch::ClusterTransactionBarrier* acc_full;
    cutlass::arch::ClusterTransactionBarrier* acc_empty;
    cutlass::arch::ClusterTransactionBarrier* acc_win_lo;
    cutlass::arch::ClusterTransactionBarrier* acc_win_hi;
};

CUTE_DEVICE int chain_tile(int base_tile, int seq, int k_tps, int k_tile) {
    return base_tile + seq * k_tps + k_tile;
}

template <int Stages>
CUTE_DEVICE void wait_smem_stage_free(const GemmPipelineBars& bars, int seq) {
    if (seq < Stages) return;
    __syncwarp();
    bars.smem_free[seq % Stages].wait((uint32_t)((seq / Stages - 1) & 1));
}

template <int Stages>
CUTE_DEVICE void wait_smem_stage_filled(const GemmPipelineBars& bars, int seq) {
    const int smem_stage = seq % Stages;
    const uint32_t phase = (uint32_t)((seq / Stages) & 1);
    __syncwarp();
    bars.tma_a[smem_stage].wait(phase);
    bars.tma_b[smem_stage].wait(phase);
}

template <int AccStages>
CUTE_DEVICE int wait_acc_stage_free(const GemmPipelineBars& bars, int base_acc,
                                    int acc_seq) {
    const int acc_global = base_acc + acc_seq;
    const int acc_stage = acc_global % AccStages;
    if (acc_global >= AccStages) {
        __syncwarp();
        bars.acc_empty[acc_stage].wait(
            (uint32_t)((acc_global / AccStages - 1) & 1));
    }
    return acc_stage;
}

template <int AccStages>
CUTE_DEVICE int wait_acc_stage_filled(const GemmPipelineBars& bars,
                                      int base_acc, int acc_seq) {
    const int acc_global = base_acc + acc_seq;
    const int acc_stage = acc_global % AccStages;
    bars.acc_full[acc_stage].wait((uint32_t)((acc_global / AccStages) & 1));
    return acc_stage;
}

template <int AccStages, bool Overlap>
CUTE_DEVICE void wait_acc_windows_free(const GemmPipelineBars& bars,
                                       int base_acc, int acc_seq) {
    if constexpr (Overlap) {
        const int acc_global = base_acc + acc_seq;
        const int stage = acc_global % AccStages;
        const int prev_acc = acc_global - 1;
        if (stage > 0 && prev_acc >= 0) {
            __syncwarp();
            bars.acc_win_hi[prev_acc % AccStages].wait(
                (uint32_t)((prev_acc / AccStages) & 1));
        }
        const int next_acc = acc_global - (AccStages - 1);
        if (stage < AccStages - 1 && next_acc >= 0) {
            __syncwarp();
            bars.acc_win_lo[next_acc % AccStages].wait(
                (uint32_t)((next_acc / AccStages) & 1));
        }
    }
}

template <bool Overlap>
CUTE_DEVICE void arrive_acc_window_lo(const GemmPipelineBars& bars,
                                      int acc_stage) {
    if constexpr (Overlap) bars.acc_win_lo[acc_stage].arrive();
}

template <bool Overlap>
CUTE_DEVICE void arrive_acc_window_hi(const GemmPipelineBars& bars,
                                      int acc_stage) {
    if constexpr (Overlap) bars.acc_win_hi[acc_stage].arrive();
}

template <int Stages, int Stage = 0, class Fn>
CUTE_DEVICE void dispatch_stage(int stage, Fn&& fn, int k) {
    if constexpr (Stage + 1 == Stages)
        fn(cute::Int<Stage>{}, k);
    else if (stage == Stage)
        fn(cute::Int<Stage>{}, k);
    else
        dispatch_stage<Stages, Stage + 1>(stage, fn, k);
}
}  // namespace halops::mla
