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

#include <cuda_bf16.h>

#include <cstdint>

#include "common/sm100_arch.cuh"

namespace halops::detail {

// Shared memory of an HCA build: KV stages, the per-key UE8M0 scales, P, the
// Q/O staging boxes, the index ring and the barriers.
template <class T>
struct HcaSmem {
    alignas(kSw128AtomBytes) uint8_t kv[T::kStages][T::kStageBytes];
    alignas(kSw128AtomBytes) uint8_t p[2][T::kHeads * kSw128AtomRowBytes];
    alignas(kSw128AtomBytes) uint8_t qo_stage[2][T::kQBoxBytes];
    alignas(kChunk) float xchg[kWarpGroup][T::kHalf];
    alignas(kChunk) uint8_t sf[T::kStages][T::kTile][T::kSfBytes];
    uint32_t valid[T::kIdxStages][T::kIndexWarps];
    uint32_t any[T::kIdxStages][T::kIndexWarps];
    int idx_raw[T::kIndexWarps][T::kIndexDepth][kWarp][2];
    float red[kWarpGroup];
    float2 ml[2][kWarpGroup];
    int start[3];
    int4 unit_lens[T::kTableUnits];
    int total;
    int wstart;
    int cap;
    int sched;
    int comb;
    int n_mj;
    int4 mj[2];
    uint64_t q_full, q_empty, o_full, o_empty[2], merge_full[T::kMergeBuf];
    uint64_t qs_full;
    uint64_t ml_full[2], ml_empty[2];
    uint64_t idx_full[T::kIdxStages], idx_empty[T::kIdxStages];
    uint64_t raw_full[T::kStages], kv_full[T::kStages], kv_empty[T::kStages];
    uint64_t s_full[2], s_empty[2], p_full[2], pv_done[2];
    uint32_t tmem_base;
};

// Shared memory of a DSA build.  Mirrors HcaSmem's field names so the shared
// softmax, scheduling, PV and merge code reuses them; it drops the separate
// scale block (DSA scales ride inline in the dequant input) and adds a rope
// KV tile per stage and the rope-Q tile for the second QK^T.
template <class T>
struct DsaSmem {
    alignas(kSw128AtomBytes) uint8_t kv[T::kStages][T::kStageBytes];
    alignas(kSw128AtomBytes) uint8_t
        rope_kv[T::kStages][T::kTile * T::kDsaRope * sizeof(__nv_bfloat16)];
    alignas(kSw128AtomBytes) uint8_t p[2][T::kHeads * kSw128AtomRowBytes];
    alignas(kSw128AtomBytes) uint8_t qo_stage[2][T::kQBoxBytes];
    alignas(kSw128AtomBytes) uint8_t q_rope[T::kDsaQRopeBytes];
    alignas(kChunk) float xchg[kWarpGroup][T::kHalf];
    alignas(kChunk) float dsa_sc[T::kStages][T::kTile][T::kDsaScales / 2];
    uint32_t valid[T::kIdxStages][T::kIndexWarps];
    uint32_t any[T::kIdxStages][T::kIndexWarps];
    int idx_raw[T::kIndexWarps][T::kIndexDepth][kWarp][2];
    float red[kWarpGroup];
    float2 ml[2][kWarpGroup];
    int start[3];
    int4 unit_lens[T::kTableUnits];
    int total;
    int wstart;
    int cap;
    int sched;
    int comb;
    int n_mj;
    int4 mj[2];
    uint64_t q_full, q_empty, o_full, o_empty[2], merge_full[T::kMergeBuf];
    uint64_t qs_full;
    uint64_t ml_full[2], ml_empty[2];
    uint64_t idx_full[T::kIdxStages], idx_empty[T::kIdxStages];
    uint64_t raw_full[T::kStages], kv_full[T::kStages], kv_empty[T::kStages];
    uint64_t s_full[2], s_empty[2], p_full[2], pv_done[2];
    uint32_t tmem_base;
};

}  // namespace halops::detail
