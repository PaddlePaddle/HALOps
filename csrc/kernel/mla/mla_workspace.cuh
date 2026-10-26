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

#include <algorithm>
#include <iterator>

#include "kernel/mla/mla_host_utils.h"
#include "kernel/mla/mla_traits.cuh"

namespace halops::mla {
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
};

struct PlanHeader {
    int iters;
};

__host__ __device__ inline int work_rounds(int num_tokens, int grid_x) {
    return (num_tokens + grid_x - 1) / grid_x;
}
__host__ __device__ inline int split_slots(int num_tokens, int grid_x) {
    return num_tokens + grid_x;
}

struct PlanView {
    const PlanHeader* __restrict__ hdr;
    const CtaWork* __restrict__ work;
    const TokenSplit* __restrict__ tok;
    const int4* __restrict__ kv_runs;
};

struct PlanLayout {
    int64_t bytes;
    int n_work;
    int64_t work_off, tok_off, kv_runs_off;
};

inline int64_t align_block(int64_t bytes) {
    return cute::round_up((int64_t)(bytes),
                          (int64_t)WorkspaceLayout::BlockAlignBytes);
}

inline PlanLayout plan_layout(int num_tokens, int grid_x) {
    PlanLayout layout{};
    layout.n_work = work_rounds(num_tokens, grid_x) * grid_x;
    layout.work_off = align_block(sizeof(PlanHeader));
    layout.tok_off =
        layout.work_off + align_block((int64_t)layout.n_work * sizeof(CtaWork));
    layout.kv_runs_off =
        layout.tok_off + align_block((int64_t)num_tokens * sizeof(TokenSplit));
    layout.bytes =
        layout.kv_runs_off + align_block((int64_t)num_tokens * sizeof(int4));
    return layout;
}

inline PlanView plan_ptrs(char* ws_base, int64_t header_prefix, int num_tokens,
                          int grid_x) {
    const PlanLayout layout = plan_layout(num_tokens, grid_x);
    char* region = ws_base + header_prefix;
    PlanView view{};
    view.hdr = reinterpret_cast<PlanHeader*>(region);
    view.work = reinterpret_cast<CtaWork*>(region + layout.work_off);
    view.tok = reinterpret_cast<TokenSplit*>(region + layout.tok_off);
    view.kv_runs = reinterpret_cast<int4*>(region + layout.kv_runs_off);
    return view;
}

inline int kv_splits_for(int grid_x, int num_tokens) {
    return std::max(
        1, std::min(grid_x / num_tokens, AttnConfig::KvSplit::SegmentCap));
}

inline int select_tile_t(int num_tokens, bool cheap_attn = false) {
    using Tile = KernelBase::Tile;
    if (cheap_attn) return Tile::Widest;
    if (num_tokens > Tile::WidestFrom && num_tokens <= Tile::Limits[1])
        return Tile::Widest;
    for (int i = 0; i < (int)std::size(Tile::Limits); ++i)
        if (num_tokens <= Tile::Limits[i]) return Tile::Widths[i];
    return Tile::Widest;
}

inline int n_groups_for(int num_tokens, bool cheap_attn = false) {
    const int tile_t = select_tile_t(num_tokens, cheap_attn);
    return (num_tokens + tile_t - 1) / tile_t;
}

inline int max_groups_upto(int num_tokens) {
    using Tile = KernelBase::Tile;
    int groups = n_groups_for(num_tokens);
    for (int step :
         {Tile::Limits[0], Tile::WidestFrom, Tile::Limits[1], Tile::Limits[2]})
        if (step < num_tokens) groups = std::max(groups, n_groups_for(step));
    return groups;
}

inline int kv_gate_items(int n_groups, bool hca) {
    return hca ? n_groups * ((2 * ModelDims::DimV +
                              TraitsN128::KvGate::WeightTile - 1) /
                             TraitsN128::KvGate::WeightTile)
               : 0;
}

inline int kv_gate_ctas(int num_tokens, int grid_x, int items_kv_gate) {
    if (items_kv_gate == 0 || num_tokens > 2 * grid_x) return 0;
    return std::min(items_kv_gate, grid_x / AttnConfig::KvGateGridDiv);
}

struct MlaWorkspace {
    using Qkv = TraitsN128::Qkv;

    static constexpr int C128TokenBytes =
        ModelDims::DimV * sizeof(float) + C128Layout::Slices * sizeof(float) +
        ModelDims::DimV * sizeof(cutlass::bfloat16_t);
    static constexpr int SplitSlotBytes =
        ModelDims::OTotal * sizeof(cutlass::bfloat16_t) +
        ModelDims::Heads * 2 * sizeof(float);

    enum : int {
        BankXReady,
        BankQloraReady,
        BankQqReady,
        BankRingReady,
        BankKvReady,
        BankFlatQReady,
        BankKvCacheReady,
        BankQnReady
    };
    static constexpr int NumBarrierBanks = BankQnReady + 1;

    static int bytes_per_token(int hidden) {
        constexpr int kPartBytes = (int)sizeof(Qkv::PartialT);
        return hidden + WorkspaceLayout::SfPadRows * (hidden / SfLayout::Gran) +
               ModelDims::QLoraRank +
               WorkspaceLayout::SfPadRows *
                   (ModelDims::QLoraRank / SfLayout::Gran) +
               ModelDims::QLoraRank * kPartBytes * Qkv::KSplit +
               ModelDims::QTotal * sizeof(cutlass::bfloat16_t) +
               ModelDims::DimV * kPartBytes * Qkv::KSplit + C128TokenBytes;
    }

    static int token_pad(int num_tokens) {
        return cute::round_up(num_tokens, WorkspaceLayout::TokenPadRows);
    }

    struct Layout {
        int t_pad;
        int pbar_bank;
        int64_t pbar_off;
        int64_t c128_flags_off;
        int64_t barrier_end;
        PlanLayout plan;
        int64_t plan_off[2];
        int64_t kv_split_off;
        int64_t lse_off;
        int64_t token_off;

        int64_t attention_end(int hidden) const {
            return token_off + (int64_t)t_pad * bytes_per_token(hidden);
        }
        int64_t output_proj_off(int hidden) const {
            return cute::round_up(
                attention_end(hidden),
                (int64_t)WorkspaceLayout::OutputProjAlignBytes);
        }
    };

    static Layout layout(int num_tokens, int grid_x) {
        Layout result{};
        result.t_pad = token_pad(num_tokens);
        result.pbar_bank = max_groups_upto(num_tokens);
        result.pbar_off = WorkspaceLayout::GridBarrierBytes;
        result.c128_flags_off =
            result.pbar_off + (int64_t)NumBarrierBanks * result.pbar_bank *
                                  WorkspaceLayout::RendezvousSlotBytes;
        result.barrier_end = result.c128_flags_off +
                             align_block((int64_t)C128Layout::FlagsPerToken *
                                         num_tokens * sizeof(int));
        result.plan = plan_layout(num_tokens, grid_x);
        result.plan_off[0] = result.barrier_end;
        result.plan_off[1] = result.plan_off[0] + result.plan.bytes;
        result.kv_split_off = result.plan_off[1] + result.plan.bytes;
        const int64_t slots = split_slots(num_tokens, grid_x);
        result.lse_off =
            result.kv_split_off +
            slots * ModelDims::OTotal * (int64_t)sizeof(cutlass::bfloat16_t);
        result.token_off =
            align_block(result.kv_split_off + slots * SplitSlotBytes);
        return result;
    }
};
}  // namespace halops::mla
