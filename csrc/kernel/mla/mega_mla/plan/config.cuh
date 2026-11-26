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

#include "kernel/mla/mega_mla/mla_params.cuh"

namespace halops::mla {

/// Launch shape of the attention planner kernels and the kv_runs row layout.
struct MlaPlanCfg {
    static constexpr int kThreads = 256;
    static constexpr int kWarps = kThreads / KernelBase::WarpSize;
    static constexpr int kRunFields = sizeof(int4) / sizeof(int);
};

/// Arguments of the per-token kv_runs kernel.
struct MlaPlanKvRunsArgs {
    const int* __restrict__ kv_indices;
    const int* __restrict__ topk_length;
    int64_t row_stride;
    int topk, step, page_shift;
    int4* __restrict__ kv_runs;
};

/// Arguments of the single-CTA attention work planner.
struct MlaPlanArgs {
    const int4* __restrict__ kv_runs;
    const int* __restrict__ c128_attn_len;
    int num_tokens, grid_x;
    int kv_splits;
    int split_min_nb, split_gain, tail_cap;
    int kvg_ctas;
    PlanHeader* __restrict__ hdr;
    CtaWork* __restrict__ work;
    TokenSplit* __restrict__ tok;
    int* __restrict__ partial_bar;
    int pbar_words;
};

}  // namespace halops::mla
