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

#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>

#include "mla_traits.cuh"

namespace mega_mla {

struct alignas(16) CtaWork {
    int token;
    int blk_lo;
    int blk_hi;
    int slot;
};

struct alignas(16) TokenSplit {
    int nb;
    int nseg;
    int slot0;
    int pad_;
};

struct PlanHeader {
    int any_split;
    int slot_step;
};

struct PlanView {
    const PlanHeader* __restrict__ hdr;
    const int* __restrict__ seg_in_group;
    const CtaWork* __restrict__ work;
    const TokenSplit* __restrict__ tok;
};

struct PlanLayout {
    int64_t bytes;
    int n_work;
};

inline int64_t align128(int64_t n) {
    constexpr int64_t A = WorkspaceLayout::BlockAlignBytes;
    return (n + A - 1) / A * A;
}

inline PlanLayout plan_layout(int T, int grid_x, int n_groups) {
    PlanLayout l{};
    l.n_work = ((T + grid_x - 1) / grid_x) * grid_x;
    l.bytes = align128(sizeof(PlanHeader)) +
              align128((int64_t)n_groups * sizeof(int)) +
              align128((int64_t)l.n_work * sizeof(CtaWork)) +
              align128((int64_t)T * sizeof(TokenSplit));
    return l;
}

inline PlanView plan_ptrs(char* ws_base, int64_t header_prefix, int T,
                          int grid_x, int n_groups) {
    const PlanLayout l = plan_layout(T, grid_x, n_groups);
    char* p = ws_base + header_prefix;
    PlanView v{};
    v.hdr = reinterpret_cast<PlanHeader*>(p);
    p += align128(sizeof(PlanHeader));
    v.seg_in_group = reinterpret_cast<int*>(p);
    p += align128((int64_t)n_groups * sizeof(int));
    v.work = reinterpret_cast<CtaWork*>(p);
    p += align128((int64_t)l.n_work * sizeof(CtaWork));
    v.tok = reinterpret_cast<TokenSplit*>(p);
    return v;
}

inline int env_int(const char* name, int def) {
    const char* e = std::getenv(name);
    return e ? atoi(e) : def;
}

inline int grid_x_for(int T) {
    constexpr int MAX_DEV = KernelTrait::DeviceCacheSlots;
    const int dev = c10::cuda::current_device();
    static int sm_cache[MAX_DEV] = {0};
    int num_sm = (dev >= 0 && dev < MAX_DEV) ? sm_cache[dev] : 0;
    if (num_sm == 0) {
        cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, dev);
        if (dev >= 0 && dev < MAX_DEV) sm_cache[dev] = num_sm;
    }
    const int auto_x = num_sm > 0 ? num_sm : T;
    static const int forced = env_int("MEGA_MLA_GRID", 0);
    return forced ? std::max(1, std::min(auto_x, forced)) : auto_x;
}

inline int kv_splits_for(int grid_x, int T) {
    static const int cap = std::max(
        1,
        env_int("MEGA_MLA_KV_SPLIT", AttnConfig::KvSplit::SegmentCapDefault));
    return std::max(1, std::min(grid_x / T, cap));
}

inline int select_tile_t(int T) {
    using Tile = KernelTrait::Tile;
    static const int f = env_int("MEGA_MLA_BN", 0);
    for (int i = 0; i < 4; ++i)
        if (f == Tile::Widths[i]) return f;
    for (int i = 0; i < 3; ++i)
        if (T <= Tile::Limits[i]) return Tile::Widths[i];
    return Tile::Widths[3];
}

inline int n_groups_for(int T) {
    const int tile_t = select_tile_t(T);
    return (T + tile_t - 1) / tile_t;
}

inline int split_min_nb() {
    static const int v =
        std::max(1, env_int("MEGA_MLA_SPLIT_MIN_NB",
                            AttnConfig::KvSplit::MinBlocksToCutDefault));
    return v;
}

inline int split_gain() {
    static const int v = std::max(
        0, env_int("MEGA_MLA_SPLIT_GAIN", AttnConfig::KvSplit::GainDefault));
    return v;
}

inline bool plan_verbose() {
    static const bool v = std::getenv("MEGA_MLA_PLAN_VERBOSE") != nullptr;
    return v;
}

}  // namespace mega_mla
