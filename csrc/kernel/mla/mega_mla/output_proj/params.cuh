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

#include "kernel/mla/mega_mla/output_proj/config_pair.cuh"
#include "kernel/mla/mega_mla/output_proj/config_swapab.cuh"

namespace halops::mla {

/// Kernel arguments shared by both output projection paths.
template <class Operands>
struct OprojParams {
    Operands wo_a;
    Operands wo_b;
    cutlass::float_e4m3_t* olora_fp8;
    uint8_t* olora_sf;
    cutlass::bfloat16_t* out;
    int* flags;
    int num_tokens;
    const uint8_t* wo_b_raw;
    OutputProjShape shape;
};

/// Weight and activation TMA atoms of one swap-AB GEMM.
template <class Gemm>
struct OprojSwapabOperands {
    WeightAtoms<Gemm> weight;
    ActAtoms<Gemm> act;
};

/// Cluster position and wo_a / wo_b tiles of one pair-path CTA.
template <class C>
struct OprojPairCoord {
    int rank, cluster, pair, half;
    int group_a, tile_a;
    int group_b, tile_b;
    CUTE_DEVICE explicit OprojPairCoord(const OutputProjShape& shape) {
        rank = cute::block_rank_in_cluster();
        cluster = blockIdx.x / OutputProjLayout::ClusterSize;
        pair = rank >> 1;
        half = rank & 1;
        const int item = 2 * cluster + pair;
        const int tiles_a = shape.olora / C::kPairRows;
        const int tiles_b = shape.hidden / C::kPairRows;
        group_a = item / tiles_a;
        tile_a = item % tiles_a;
        group_b = cluster / tiles_b;
        tile_b = cluster % tiles_b;
    }
    CUTE_DEVICE int quant_tile() const { return 2 * tile_a + half; }
};

/// Work of one swap-AB CTA: its cluster and rank and the wo_a tile it owns.
struct OprojSwapabCoord {
    int cluster, rank;
    int tile_a;
    int half;

    CUTE_DEVICE explicit OprojSwapabCoord(const OutputProjShape& shape) {
        rank = cute::block_rank_in_cluster();
        cluster = blockIdx.x / OutputProjLayout::ClusterSize;
        tile_a = cluster + shape.clusters() * (rank >> 1);
        half = rank & 1;
    }
};

/// State of one pair-path drain thread.
template <class C>
struct OprojPairDrain {
    typename C::Smem& smem;
    const OprojPairCoord<C>& coord;
    int wg, tid;
    uint32_t tmem_base;
    __nv_bfloat16* transpose;
};

/// State of one swap-AB reduction thread.
template <class C, class Prm>
struct OprojSwapabReduce {
    const Prm& params;
    typename C::Bars& bars;
    const OprojSwapabCoord& coord;
    const float* rx;
    float* staging;
    uint32_t rx_smem, staging_smem;
    int tid;
};

}  // namespace halops::mla
