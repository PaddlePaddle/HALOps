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

#include "cutlass/arch/barrier.h"
#include "cutlass/fast_math.h"
#include "kernel/attention/utils.cuh"

namespace halops {

struct TileSchedulerArguments {
    int const num_block;
    int const num_head;
    int const num_batch;
    int const seqlen_k;
    int const headdim;
    int const total_q;
    int const tile_shape_mn[2] = {256, 128};
    int const cluster_shape_mn[2] = {1, 1};
    int const *cu_seq_q = nullptr;
    int const *cu_seq_k = nullptr;
    int const element_size = 2;
    bool const is_persistent = false;
    bool const is_causal = false;
    bool const lpt = false;
    bool const head_swizzle = false;
};

class SingleTileVarlenScheduler {
    int const NumThreadsPerWarp = cutlass::NumThreadsPerWarp;

  public:
    bool is_first = true;

    struct Params {
        int const num_head;
        int const num_batch;
        int const total_q;
        int const max_kvblock_in_l2;
        int const tile_shape_mn[2];
        int const *cu_seq_q;
        int const *cu_seq_k;
        bool const lpt;
        bool const head_swizzle;
    };

    struct WorkTileInfo {
        int block_idx;
        int bidh;
        int bidb;
        bool is_valid;
    };

    static Params to_underlying_arguments(TileSchedulerArguments const &args) {
        int const size_l2 = 50 * 1024 * 1024;  // 50 MB for K & V
        int const max_kvblock_in_l2 =
            size_l2 /
            (2 * args.headdim * args.element_size * args.tile_shape_mn[1]);
        return {args.num_head,
                args.num_batch,
                args.total_q,
                max_kvblock_in_l2,
                {args.tile_shape_mn[0], args.tile_shape_mn[1]},
                args.cu_seq_q,
                args.cu_seq_k,
                args.lpt,
                args.head_swizzle};
    }

    static dim3 get_grid_shape(Params const &params) {
        int const total_blocks_max =
            (params.total_q +
             params.num_batch * (params.tile_shape_mn[0] - 1)) /
            params.tile_shape_mn[0];
        return {total_blocks_max * params.num_head, 1, 1};
    }

    __forceinline__ __device__ int get_num_m_blocks(Params const &params,
                                                    int const lane_idx,
                                                    int const bidb_start = 0) {
        int const batch_idx = lane_idx + bidb_start;
        int cur_cu_seqlen = 0;
        if (batch_idx <= params.num_batch) {
            cur_cu_seqlen = params.cu_seq_q[batch_idx];
        }
        int const next_cu_seqlen =
            __shfl_down_sync(0xffffffff, cur_cu_seqlen, 1);
        int seqlen = next_cu_seqlen - cur_cu_seqlen;
        int num_m_blocks = 0;
        if (batch_idx < params.num_batch && lane_idx < NumThreadsPerWarp - 1) {
            num_m_blocks = cute::ceil_div(seqlen, params.tile_shape_mn[0]);
        }
        return num_m_blocks;
    }

    CUTLASS_DEVICE
    WorkTileInfo get_current_work(Params const &params, int const lane_idx) {
        int num_m_blocks = get_num_m_blocks(params, lane_idx);
        int num_m_blocks_cumulative = warp_prefix_sum(num_m_blocks, lane_idx);
        int m_blocks_in_group = __shfl_sync(0xffffffff, num_m_blocks_cumulative,
                                            cutlass::NumThreadsPerWarp - 1);
        int group_end_tile = m_blocks_in_group * params.num_head;
        int block = 0;
        int head_idx = 0;
        int batch_idx = 0;
        int next_tile_idx = blockIdx.x;
        while (group_end_tile <= next_tile_idx) {
            batch_idx += NumThreadsPerWarp - 1;
            if (batch_idx >= params.num_batch) {
                batch_idx = params.num_batch;
                group_end_tile = next_tile_idx + 1;
            } else {
                num_m_blocks = get_num_m_blocks(params, lane_idx, batch_idx);
                num_m_blocks_cumulative =
                    warp_prefix_sum(num_m_blocks, lane_idx);
                m_blocks_in_group = __shfl_sync(
                    0xffffffff, num_m_blocks_cumulative, NumThreadsPerWarp - 1);
                group_end_tile += m_blocks_in_group * params.num_head;
            }
        }

        bool is_valid = false;
        if (batch_idx >= params.num_batch) {
            block = 0;
            head_idx = 0;
            batch_idx = params.num_batch;
        } else {
            int group_start_tile =
                group_end_tile - m_blocks_in_group * params.num_head;
            int batch_idx_in_group = __popc(__ballot_sync(
                0xffffffff,
                group_start_tile + num_m_blocks_cumulative * params.num_head <=
                    next_tile_idx));
            batch_idx += batch_idx_in_group;
            int num_m_blocks_prev_lane = 0;
            if (batch_idx_in_group != 0) {
                num_m_blocks_prev_lane =
                    __shfl_sync(0xffffffff, num_m_blocks_cumulative,
                                batch_idx_in_group - 1);
            }
            num_m_blocks =
                __shfl_sync(0xffffffff, num_m_blocks, batch_idx_in_group);

            int const mh_block = next_tile_idx - group_start_tile -
                                 num_m_blocks_prev_lane * params.num_head;

            if (params.lpt || params.head_swizzle) {
                int const num_n_blocks = cute::ceil_div(
                    params.cu_seq_k[batch_idx + 1] - params.cu_seq_k[batch_idx],
                    params.tile_shape_mn[1]);
                int nheads_in_l2 = 1;
                if (num_n_blocks * 16 <= params.max_kvblock_in_l2) {
                    nheads_in_l2 = 16;
                } else if (num_n_blocks * 8 <= params.max_kvblock_in_l2) {
                    nheads_in_l2 = 8;
                } else if (num_n_blocks * 4 <= params.max_kvblock_in_l2) {
                    nheads_in_l2 = 4;
                } else if (num_n_blocks * 2 <= params.max_kvblock_in_l2) {
                    nheads_in_l2 = 2;
                }
                nheads_in_l2 = min(nheads_in_l2, params.num_head);
                int const mh_in_l2 = nheads_in_l2 * num_m_blocks;
                int const section_idx = mh_block / mh_in_l2;
                int const l2_mod = mh_block - section_idx * mh_in_l2;
                int nheads_in_this_section;
                if (nheads_in_l2 * (section_idx + 1) <= params.num_head) {
                    nheads_in_this_section = nheads_in_l2;
                } else {
                    nheads_in_this_section =
                        params.num_head - section_idx * nheads_in_l2;
                }
                int block = l2_mod / nheads_in_this_section;
                int const head_idx_residual =
                    l2_mod - block * nheads_in_this_section;
                head_idx = section_idx * nheads_in_l2 + head_idx_residual;
                if (params.lpt) {
                    block = num_m_blocks - 1 - block;
                }
            } else {
                head_idx = mh_block / num_m_blocks;
                block = mh_block - head_idx * num_m_blocks;
            }
            is_valid = is_first && batch_idx < params.num_batch;
        }

        return WorkTileInfo{block, head_idx, batch_idx, is_valid};
    }

    CUTLASS_DEVICE
    WorkTileInfo initial_work_tile_info(Params const &params,
                                        int const lane_idx) {
        return get_current_work(params, lane_idx);
    }

    CUTLASS_DEVICE
    void advance_to_next_work() { is_first = false; }
};

}  // namespace halops
