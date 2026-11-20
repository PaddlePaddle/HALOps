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

#include <cuda.h>
#include <cuda_bf16.h>

#include <cstdint>

#include "kernel/mla/prefill_pair_mla_traits.cuh"

namespace halops::mla::prefill_pair {

// kQst: a unit's four Q rounds alternate between two buffers.
template <class T, bool kQst>
struct QBufs {};
template <class T>
struct QBufs<T, true> {
    alignas(T::kAtomBytes)
        uint8_t q_stage[2][2][T::kQBoxBytes];  // [round % 2][dims half]
    uint64_t qb_full[2];
};

template <class T, bool kQst>
struct PairMlaSmemT {
    alignas(T::kAtomBytes) uint8_t kv[T::kKvStages][T::kStageBytes];
    alignas(T::kAtomBytes) uint8_t p[2][T::kHeads * T::kRowPitch];
    alignas(T::kAtomBytes) uint8_t qo_stage[2][T::kQBoxBytes];  // [dims half]
    QBufs<T, kQst> qb;
    alignas(T::kChunk) float xchg[T::kWarpGroup][T::kHalf];
    alignas(T::kChunk) float qfreq[T::kQRopeDims];  // kQnr: the unit's RoPE
    float rs[2][T::kHeads];  // kQnr: scale_log2 / rms per head and unit parity
    alignas(T::kChunk) uint8_t sf[T::kKvStages][T::kTile][T::kSfBytes];
    // bit j of word w: key 32w + j is valid for this CTA's token / has a row
    uint32_t valid[T::kIdxStages][T::kIndexWarps];
    uint32_t any[T::kIdxStages][T::kIndexWarps];
    int bf[T::kIdxStages];  // the tile was gathered from the bf16 c128 copy
    int idx_raw[T::kIndexWarps][T::kIndexDepth][T::kWarp][2];
    float red[T::kWarpGroup];
    float2 ml[2][T::kWarpGroup];  // per unit parity: row max, partial row sum
    int start[3];                 // first pair, its first tile, tile count
    // lengths of the range's pairs from start[0]
    int4 unit_lens[PairSchedTraits::kTableUnits];
    int wstart;  // stream position of the range's first pair's first tile
    int sched;   // clusters actually scheduled this launch (<= p.clusters)
    int dense;   // every scheduled cluster's range is non-empty
    // cluster c: tiles [bound[c], bound[c + 1])
    int bound[PairSchedTraits::kMaxClusters + 1];
    uint64_t q_full, q_empty, o_full, o_empty[2], merge_full[T::kMergeBuf];
    uint64_t qs_full;  // a Q round landed in qo_stage (not used by kQst)
    uint64_t ml_full[2], ml_empty[2];
    uint64_t rs_full[2];  // kQnr: rs[b] written
    uint64_t idx_full[T::kIdxStages], idx_empty[T::kIdxStages];
    uint64_t raw_full[T::kKvStages], kv_full[T::kKvStages],
        kv_empty[T::kKvStages];
    uint64_t s_full[2], s_empty[2], p_full[2], pv_done[2];
    uint32_t tmem_base;
};

// kQst: Q buffers for short pairs; kFp8: no bf16 c128 copy; kQnr: fused
// q_norm_rope.
template <bool kQstV, bool kFp8V, bool kQnrV, class Dims = ModelDims>
struct Cfg : PairMlaTraits<Dims> {
    using Traits = PairMlaTraits<Dims>;
    static constexpr bool kQst = kQstV;
    static constexpr bool kFp8 = kFp8V;
    static constexpr bool kQnr = kQnrV;
    using Smem = PairMlaSmemT<Traits, kQstV>;
    static constexpr int kSmemBytes = sizeof(Smem) + Traits::kAtomBytes;
    static_assert(kSmemBytes <= Sm100::SmemPerCtaMax);
};

struct PairMlaMaps {
    CUtensorMap raw[2][2];  // [cta][seg] 16-bit view of the rows, raw_pitch box
    CUtensorMap bf16;       // bf16 c128 copy, boxes of 64 dims x 32 rows
    CUtensorMap q;    // q rows (token x head), 64-dim swizzled boxes of 64 rows
    CUtensorMap out;  // out rows, boxes as for q
};

// A split pair for the combine kernel: first token, has b, stream tiles (wn
// 0: none).
struct MergeJob {
    int a, has_b, w0, wn;
};

struct PairMlaParams {
    const int* idx[2];  // [0] c128, [1] SWA: [T, ld] pool token indices
    const int* len[2];
    int ld[2];
    const uint8_t* pool[2];
    int64_t pool_stride[2];
    int page_shift[2];  // log2 of the tokens per page
    int page_rows[2];   // TMA rows (576 B) per page
    int64_t pool_tokens[2];
    // Pre-dequant: c128_base[t] is token t's first row in c128_bf, or -1.
    int c128_rows;  // rows per request: ld[0] rounded up to a tile
    __nv_bfloat16* c128_bf;
    int* c128_base;  // [T]
    const __nv_bfloat16* q;
    const float* sink;
    __nv_bfloat16* out;
    float* o_part;    // [2 * clusters][2][kPlanes][128][4] normalized partial O
    float* lse_part;  // [2 * clusters][2][64] their log2 sums
    int* counter;     // [clusters][2] parts published of a cluster's split pair
    int tokens;
    int tokens_per_req;  // t; tokens are paired within a request, not globally
    // Varlen: [R + 1] token and pair offsets of the requests.
    const int* cu_seqlens;
    const int* cu_pairs;
    int n_req;
    // Position gap of a pair's tokens: a's SWA list is b's shifted by it.
    int swa_shift;
    int clusters;  // clusters launched
    // Few pairs: every part writes its partial, the combine kernel merges.
    int combine;
    MergeJob* jobs;  // [clusters] by first cluster when combine
    int* total_g;    // the stream's tile total, for cluster_of in combine
    int* sched_g;    // clusters scheduled (sched), for the combine kernel
    float scale_log2;
    // kQnr: raw q projection, token stride q_stride.
    int64_t q_stride;
    const void* q_pos;  // [T] int64 (pos64) or int32
    int pos64;
    const float* q_freqs;  // [max_pos, 64] fp32, (re, im) interleaved
    float q_eps;
};

}  // namespace halops::mla::prefill_pair
