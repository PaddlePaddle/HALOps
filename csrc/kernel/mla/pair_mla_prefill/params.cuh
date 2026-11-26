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

#include "kernel/mla/pair_mla_prefill/config.cuh"

namespace halops::mla {

// TMA maps: raw[cta][seg] KV rows, the bf16 c128 copy, q and out rows.
struct PrefillMaps {
    CUtensorMap raw[2][2];
    CUtensorMap bf16;
    CUtensorMap q;
    CUtensorMap out;
};

// A split pair for the combine kernel (wn 0: no job).
struct PrefillMergeJob {
    int a, has_b, w0, wn;
};

// Kernel arguments; segment 0 is c128 and 1 is SWA, c128_base[t] is token t's
// first c128_bf row or -1, and a's SWA list is b's shifted by swa_shift.
struct PrefillParams {
    const int* idx[2];
    const int* len[2];
    int ld[2];
    const uint8_t* pool[2];
    int64_t pool_stride[2];
    int page_shift[2];
    int page_rows[2];
    int64_t pool_tokens[2];
    int c128_rows;
    __nv_bfloat16* c128_bf;
    int* c128_base;
    const __nv_bfloat16* q;
    const float* sink;
    __nv_bfloat16* out;
    float* o_part;
    float* lse_part;
    int* counter;
    int tokens;
    const int* cu_seqlens;
    const int* cu_pairs;
    int n_req;
    int swa_shift;
    int clusters;
    int combine;
    PrefillMergeJob* jobs;
    int* total_g;
    int* sched_g;
    float scale_log2;
    int64_t q_stride;
    const void* q_pos;
    int pos64;
    const float* q_freqs;
    float q_eps;
};

// Key stream of pair (a, b): the c128 rows, then b's SWA window and a's oldest
// swa_shift rows; len is [seg][token of pair], no b for a lone token.
struct PrefillPairGeom {
    int a, len[2][2], n_c4, n_s, n_tiles;
    bool has_b;
};

struct PrefillPairLens {
    int v[2][2];
};

struct PrefillPairMap {
    int a;
    bool has_b;
};

// Tiles [off, off + n) of a pair whose stream tiles are [w0, w0 + wn).
struct PrefillUnit {
    PrefillPairGeom g;
    int off, n;
    int w0, wn;
    bool split;
};

// An index warp's key: lists, segment and bf16 copy row (all-c128 tile) or -1.
struct PrefillKeyRaw {
    bool in_a, in_b;
    int seg;
    int bf;
};

}  // namespace halops::mla
