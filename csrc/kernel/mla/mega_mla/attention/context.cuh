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

#include "kernel/mla/mega_mla/attention/config.cuh"
#include "kernel/mla/mega_mla/mla_utils.cuh"

namespace halops::mla {
/// Per-CTA state shared by every warp role of the attention kernel.
template <class KT_, class MP, class EP, class Plan, class Bars, class Sched,
          class PreludeRanges>
struct KernelPipelines {
    using KT = KT_;
    static constexpr int TileT = KT::TileT;

    const MP& params;
    const EP& ep;
    Plan& plan;
    AttnRing& rs;
    uint32_t& q_tma_phase;
    uint32_t& q_utccp_phase;
    int warp_idx;
    int lane_idx;
    int idx_in_warpgroup;
    int warpgroup_idx;

    int token_iters;
    const Sched& sched;
    PreludeRanges prelude;
    Bars bars;
};

/// The CTA state plus the KV geometry of the token it attends in one iteration.
template <class Ctx>
struct PerToken {
    using KT = typename Ctx::KT;

    decltype(Ctx::params)& params;
    decltype(Ctx::ep)& ep;
    decltype(Ctx::plan)& plan;
    decltype(Ctx::rs)& rs;
    decltype(Ctx::q_tma_phase)& q_tma_phase;
    decltype(Ctx::q_utccp_phase)& q_utccp_phase;
    int warp_idx;
    int lane_idx;
    int idx_in_warpgroup;

    int token_idx;
    bool token_active;
    int topk_length;
    int swa_blocks;
    int c128_len;
    const int* c128_pt;
    KvRuns kv_runs;
    KvSegGeom kv_seg_swa;
    KvSegGeom kv_seg_c128;
    int blk_lo;
    int blk_hi;
    int slot;
    int q_buf;

    CUTE_DEVICE KvRuns kv_tile_at(int block_idx) const {
        if (block_idx < swa_blocks)
            return kv_tile_slice(kv_runs, block_idx * AttnConfig::KvBlock,
                                 AttnConfig::KvBlock);
        return kv_tile_page_table(
            c128_pt, c128_len, (block_idx - swa_blocks) * AttnConfig::KvBlock,
            AttnConfig::KvBlock, params.c128_page_shift,
            params.c128_rows_per_page);
    }
    CUTE_DEVICE const KvSegGeom& kv_seg_at(int block_idx) const {
        return block_idx < swa_blocks ? kv_seg_swa : kv_seg_c128;
    }

    CUTE_DEVICE int kv_tile_valid(int block_idx) const {
        return block_idx < swa_blocks
                   ? topk_length - block_idx * AttnConfig::KvBlock
                   : c128_len - (block_idx - swa_blocks) * AttnConfig::KvBlock;
    }

    CUTE_DEVICE auto kv_nope_desc(int block_idx) const {
        return block_idx < swa_blocks ? params.kv_nope_dense
                                      : params.c128_nope_dense;
    }
    CUTE_DEVICE const CUtensorMap* kv_nope_gather(int block_idx) const {
        return block_idx < swa_blocks ? &params.tensor_map_kv_nope
                                      : &params.c128_nope;
    }
    CUTE_DEVICE auto kv_rope_desc(int block_idx) const {
        return block_idx < swa_blocks ? params.kv_rope_dense
                                      : params.c128_rope_dense;
    }
    CUTE_DEVICE const CUtensorMap* kv_rope_gather(int block_idx) const {
        return block_idx < swa_blocks ? &params.tensor_map_kv_rope
                                      : &params.c128_rope;
    }
};

template <class Ctx>
CUTE_DEVICE PerToken<Ctx> make_per_token(Ctx const& k, int token_it) {
    const CtaWork work = k.params.wplan.work[token_it * gridDim.x + blockIdx.x];
    const bool token_active = work.blk_hi > work.blk_lo;
    const int token_idx = work.token;

    const int topk_length = token_active
                                ? __ldg(&k.params.wplan.kv_runs[token_idx].w)
                                : AttnConfig::KvBlock;
    const int total_topk_padded =
        max(cute::round_up(topk_length, int{AttnConfig::KvBlock}),
            int{AttnConfig::KvBlock});
    const int swa_blocks = total_topk_padded / AttnConfig::KvBlock;
    const int c128_len = (k.params.c128_attn_len && token_active)
                             ? __ldg(k.params.c128_attn_len + token_idx)
                             : 0;
    const int* c128_pt =
        (k.params.c128_page_table && token_active)
            ? k.params.c128_page_table +
                  (int64_t)token_idx * k.params.c128_page_table_stride
            : nullptr;

    const KvRuns kv_runs = make_kv_runs(
        __ldg(k.params.wplan.kv_runs + (token_active ? token_idx : 0)),
        topk_length, k.params.page_shift, k.params.rows_per_page);

    const KvSegGeom kv_seg_swa{reinterpret_cast<const uint8_t*>(k.params.kv),
                               k.params.stride_kv_block};
    const KvSegGeom kv_seg_c128{k.params.c128_pool,
                                k.params.c128_stride_kv_block};

    return PerToken<Ctx>{k.params,
                         k.ep,
                         k.plan,
                         k.rs,
                         k.q_tma_phase,
                         k.q_utccp_phase,
                         k.warp_idx,
                         k.lane_idx,
                         k.idx_in_warpgroup,
                         token_idx,
                         token_active,
                         topk_length,
                         swa_blocks,
                         c128_len,
                         c128_pt,
                         kv_runs,
                         kv_seg_swa,
                         kv_seg_c128,
                         work.blk_lo,
                         work.blk_hi,
                         work.slot,
                         k.rs.buf_idx ^ 1};
}

/// Online-softmax running max, running sum and the unclamped max of one row.
struct MlaAttnSoftmaxState {
    float mi;
    float li;
    float real_mi;
};
}  // namespace halops::mla
