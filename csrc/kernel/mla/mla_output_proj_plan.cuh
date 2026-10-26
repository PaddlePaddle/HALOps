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

#include <torch/extension.h>

#include <algorithm>
#include <type_traits>

#include "kernel/mla/output_proj_pair.cuh"
#include "kernel/mla/output_proj_swapab.cuh"

namespace halops::mla {

constexpr int kPairMaxGroupTokens =
    OutputProjLayout::MaxTokens / OutputProjLayout::MaxTokenGroups;

constexpr int swapab_tile(int tokens) {
    for (int width : KernelBase::Tile::Widths)
        if (tokens <= width) return width;
    return KernelBase::Tile::Widest;
}

inline OutputProjShape output_proj_shape(const torch::Tensor& wo_a_weight,
                                         const torch::Tensor& wo_b_weight,
                                         const char* op) {
    constexpr int kMaxBK = swapab::k_block<KernelBase::Tile::Smallest>();
    static_assert((OutputProjLayout::ClusterSize * kMaxBK) %
                          (4 * OutputProjLayout::RowTile) ==
                      0,
                  "each cluster owns RowTile wo_b rows and two wo_a tiles");
    TORCH_CHECK(wo_a_weight.dim() == 2 && wo_b_weight.dim() == 2, op,
                ": weights must be 2-D");
    OutputProjShape shape{};
    shape.hidden = (int)wo_b_weight.size(0);
    shape.olora = (int)wo_a_weight.size(0);
    shape.group_k = (int)wo_a_weight.size(1);
    TORCH_CHECK(shape.group_k > 0 && ModelDims::OTotal % shape.group_k == 0 &&
                    shape.group_k % (2 * kMaxBK) == 0,
                op, ": wo_a K = ", shape.group_k,
                " must split OTotal = ", ModelDims::OTotal,
                " into output groups of a multiple of ", 2 * kMaxBK);
    shape.group_rows = shape.olora / (ModelDims::OTotal / shape.group_k);
    TORCH_CHECK(
        shape.olora == 2 * shape.hidden && wo_b_weight.size(1) == shape.olora &&
            shape.olora % (OutputProjLayout::ClusterSize * kMaxBK) == 0 &&
            shape.group_rows % (2 * OutputProjLayout::RowTile) == 0,
        op, ": needs wo_a [2 H, K], wo_b [H, 2 H] with 2 H a multiple of ",
        OutputProjLayout::ClusterSize * kMaxBK, " and of ",
        2 * OutputProjLayout::RowTile * (ModelDims::OTotal / shape.group_k),
        ", got wo_a [", shape.olora, ", ", shape.group_k, "], wo_b [",
        shape.hidden, ", ", wo_b_weight.size(1), "]");
    TORCH_CHECK(shape.wo_a_tiles() / 2 <= KernelBase::WarpSize, op,
                ": wo_a has more than ", 2 * KernelBase::WarpSize, " ",
                OutputProjLayout::RowTile, "-row tiles");
    return shape;
}

struct OutputProjPlan {
    int num_tokens;
    bool pair;
    int group_tokens, groups, n_parts;
    int t_pad, t_pad_sf;
    int64_t off_flat_o, off_flat_o_sf, off_olora, off_olora_sf, off_flags,
        bytes;

    OutputProjPlan(int tokens, const OutputProjShape& shape, const char* op)
        : num_tokens(tokens) {
        TORCH_CHECK(tokens > 0 && tokens <= OutputProjLayout::MaxTokens, op,
                    ": T = ", tokens, " not in [1, ",
                    OutputProjLayout::MaxTokens, "]");
        pair = tokens > KernelBase::Tile::Widest;
        if (pair) {
            groups = OutputProjLayout::MaxTokenGroups;
            const int per_group = (tokens + groups - 1) / groups;
            n_parts = per_group > Sm100::UmmaMaxN ? 2 : 1;
            group_tokens = std::max(
                SfLayout::MnPerTile,
                cute::round_up(per_group, n_parts * pair_umma::kSlice));
        } else {
            groups = 1;
            n_parts = 1;
            group_tokens = swapab_tile(tokens);
        }
        t_pad = cute::round_up(groups * group_tokens, SfLayout::MnPerTile);
        t_pad_sf = groups * n_parts *
                   cute::round_up(group_tokens / n_parts, SfLayout::MnPerTile);
        int64_t offset = 0;
        auto take = [&](int64_t size) {
            const int64_t at = offset;
            offset = cute::round_up(
                offset + size, (int64_t)WorkspaceLayout::OutputProjAlignBytes);
            return at;
        };
        off_flat_o = take((int64_t)t_pad * ModelDims::OTotal);
        off_flat_o_sf =
            take((int64_t)t_pad_sf * ModelDims::OTotal / SfLayout::Gran);
        off_olora = take((int64_t)t_pad * shape.olora);
        off_olora_sf = take((int64_t)t_pad_sf * shape.olora / SfLayout::Gran);
        off_flags = take((int64_t)shape.n_flags() * sizeof(int));
        bytes = offset;
    }
};

template <bool Pair, int TT, int NH, class Fn>
void call_output_proj_tile(Fn& fn) {
    fn(std::integral_constant<bool, Pair>{}, std::integral_constant<int, TT>{},
       std::integral_constant<int, NH>{});
}

template <class Fn>
void dispatch_output_proj_tile(const OutputProjPlan& plan, const char* op,
                               Fn&& fn) {
    if (plan.pair) {
        switch (plan.group_tokens) {
            case 128:
                return call_output_proj_tile<true, 128, 1>(fn);
            case 160:
                return call_output_proj_tile<true, 160, 1>(fn);
            case 192:
                return call_output_proj_tile<true, 192, 1>(fn);
            case 224:
                return call_output_proj_tile<true, 224, 1>(fn);
            case 256:
                return call_output_proj_tile<true, 256, 1>(fn);
            case kPairMaxGroupTokens:
                return call_output_proj_tile<true, kPairMaxGroupTokens, 2>(fn);
            default:
                TORCH_CHECK(false, op, ": no pair tile for ", plan.group_tokens,
                            " tokens per group");
        }
    } else {
        switch (plan.group_tokens) {
            case KernelBase::Tile::Widths[0]:
                return call_output_proj_tile<false, KernelBase::Tile::Widths[0],
                                             1>(fn);
            case KernelBase::Tile::Widths[1]:
                return call_output_proj_tile<false, KernelBase::Tile::Widths[1],
                                             1>(fn);
            case KernelBase::Tile::Widths[2]:
                return call_output_proj_tile<false, KernelBase::Tile::Widths[2],
                                             1>(fn);
            case KernelBase::Tile::Widths[3]:
                return call_output_proj_tile<false, KernelBase::Tile::Widths[3],
                                             1>(fn);
            default:
                TORCH_CHECK(false, op, ": no swap-AB tile for ",
                            plan.group_tokens, " tokens");
        }
    }
}

template <bool Pair, int TT, int NH>
auto output_proj_flat_o_sf_layout(const OutputProjShape& shape,
                                  const OutputProjPlan& plan, uint8_t* flat_o,
                                  uint8_t* flat_o_sf) {
    if constexpr (Pair) {
        return pair_umma::act_sf_layout<TT, NH>(shape.olora, plan.t_pad_sf,
                                                ModelDims::OTotal);
    } else {
        using Cfg = typename swapab::Shape<TT>::Cfg;
        return sf_layout(make_act_atoms<Cfg>(flat_o, flat_o_sf, shape.olora,
                                             plan.t_pad, ModelDims::OTotal,
                                             plan.t_pad_sf));
    }
}

}  // namespace halops::mla
