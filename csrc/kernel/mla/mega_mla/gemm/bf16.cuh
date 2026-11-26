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

#include "kernel/mla/mega_mla/gemm/mma_drain.cuh"

namespace halops::mla {
template <class Cfg>
struct Bf16GemmTypes {
    static constexpr int BM = Cfg::BM;
    static constexpr int BN = Cfg::BN;
    static constexpr int BK = Cfg::BK;
    static constexpr int Stages = Cfg::TmaStages;

    using SmemLayoutA1 =
        decltype(make_umma_canonical_k_major_layout<BM, BK, 128,
                                                    cutlass::bfloat16_t>());
    using SmemLayoutB1 =
        decltype(make_umma_canonical_k_major_layout<BN, BK, 128,
                                                    cutlass::bfloat16_t>());

    static constexpr int SmemA =
        cute::cosize_v<SmemLayoutA1> * (int)sizeof(cutlass::bfloat16_t);
    static constexpr int SmemB =
        cute::cosize_v<SmemLayoutB1> * (int)sizeof(cutlass::bfloat16_t);
    static constexpr int ArenaStages = Cfg::OwnerTraits::MaxTmaStages;

    using Fp8Slot = GemmInst<typename Cfg::OwnerTraits::Qkv, Stages>;
    static constexpr int SlotA = Fp8Slot::SmemA;
    static constexpr int SlotB = Fp8Slot::SlotB;
    static constexpr bool Overlap = Cfg::AccSharedCols > 0;
    static constexpr int OffA = 0;
    static constexpr int OffB = ArenaStages * SlotA;

    using MmaAtom =
        SM100_MMA_F16BF16_SS_NOELECT<cutlass::bfloat16_t, cutlass::bfloat16_t,
                                     float, BM, BN, cute::UMMA::Major::K,
                                     cute::UMMA::Major::K>;
    using TiledMma = decltype(cute::make_tiled_mma(MmaAtom{}));

    using TileA = cute::Shape<cute::Int<BM>, cute::Int<BK>>;
    using TileB = cute::Shape<cute::Int<BN>, cute::Int<BK>>;

    using CopyOpT2R = TmemLoad32dp32b<BN>;
};

template <class Cfg>
CUTE_HOST auto make_bf16_gemm_tma_w(const void* weight, int w_out, int k_dim) {
    using Types = Bf16GemmTypes<Cfg>;
    auto layout = cute::make_layout(cute::make_shape(w_out, k_dim),
                                    cute::make_stride(k_dim, cute::_1{}));
    return cute::make_tma_copy(
        cute::SM90_TMA_LOAD{},
        cute::make_tensor(cute::make_gmem_ptr(
                              static_cast<const cutlass::bfloat16_t*>(weight)),
                          layout),
        typename Types::SmemLayoutA1{});
}

template <class Cfg>
CUTE_HOST auto make_bf16_gemm_tma_act(const void* act, int num_rows,
                                      int row_stride, int k_dim) {
    using Types = Bf16GemmTypes<Cfg>;
    auto layout = cute::make_layout(cute::make_shape(num_rows, k_dim),
                                    cute::make_stride(row_stride, cute::_1{}));
    return cute::make_tma_copy(
        cute::SM90_TMA_LOAD{},
        cute::make_tensor(
            cute::make_gmem_ptr(static_cast<const cutlass::bfloat16_t*>(act)),
            layout),
        typename Types::SmemLayoutB1{});
}

struct Bf16GemmItem {
    int w_tile;
    int n_tile;
    int n_valid;
};

template <class Cfg>
CUTE_DEVICE Bf16GemmItem bf16_gemm_item_at(int it, int n_groups,
                                           int num_tokens) {
    const int tg = it % n_groups;
    const int mt = it / n_groups;
    return Bf16GemmItem{mt, tg, min(Cfg::BN, num_tokens - tg * Cfg::BN)};
}

template <class Cfg, class TmaW, class TmaAct>
struct Bf16GemmResources {
    const TmaW& tma_w;
    const TmaAct& tma_act;
    char* __restrict__ gemm_smem;
    GemmPipelineBars bars;
    uint32_t tmem_base;
    int act_rows;
    int w_out;
    int k_dim;
    int k_tiles;
    float* __restrict__ ring;
    const int* __restrict__ ring_rows;
};

template <class Cfg, int St>
CUTE_DEVICE auto bf16_smem_a(char* smem) {
    using Types = Bf16GemmTypes<Cfg>;
    return cute::make_tensor(
        cute::make_smem_ptr(reinterpret_cast<cutlass::bfloat16_t*>(
            smem + Types::OffA + St * Types::SlotA)),
        typename Types::SmemLayoutA1{});
}
template <class Cfg, int St>
CUTE_DEVICE auto bf16_smem_b(char* smem) {
    using Types = Bf16GemmTypes<Cfg>;
    return cute::make_tensor(
        cute::make_smem_ptr(reinterpret_cast<cutlass::bfloat16_t*>(
            smem + Types::OffB + St * Types::SlotB)),
        typename Types::SmemLayoutB1{});
}
}  // namespace halops::mla
