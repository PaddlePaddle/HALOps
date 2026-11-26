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

#include <optional>
#include <type_traits>

#include "kernel/mla/common/traits.cuh"
#include "kernel/mla/mega_mla/mla_gemm.cuh"
#include "kernel/mla/mega_mla/mla_workspace.cuh"

namespace halops::mla {
struct MlaAttentionArguments {
    torch::Tensor q_or_x;
    torch::Tensor kv_cache;
    int64_t page_size;
    torch::Tensor positions;
    torch::Tensor freqs_real;
    double sm_scale;
    torch::Tensor wq_a_weight;
    torch::Tensor q_norm_weight;
    torch::Tensor wq_b_weight;
    double q_eps;
    torch::Tensor wkv_weight;
    torch::Tensor kv_norm_weight;
    double kv_eps;
    torch::Tensor swa_loc;
    std::optional<torch::Tensor> attn_sink;
    torch::Tensor wq_a_sf;
    torch::Tensor wq_b_sf;
    torch::Tensor wkv_sf;
    torch::Tensor workspace;
    std::optional<torch::Tensor> wkv_gate_weight;
    std::optional<torch::Tensor> kv_score_ring;
    std::optional<torch::Tensor> kv_gate_ring_row;
    std::optional<torch::Tensor> c128_ape;
    std::optional<torch::Tensor> c128_norm_weight;
    std::optional<torch::Tensor> c128_freqs_real;
    std::optional<torch::Tensor> c128_out_loc;
    std::optional<torch::Tensor> c128_pool;
    int64_t c128_stride_kv_block;
    int64_t c128_page_block;
    std::optional<torch::Tensor> c128_page_table;
    std::optional<torch::Tensor> c128_attn_len;
    int64_t draft_tokens;
    torch::Tensor hidden_states;
    bool reset_barriers;
};

template <class T = void>
T* optional_ptr(const std::optional<torch::Tensor>& tensor) {
    if (!tensor.has_value()) return nullptr;
    if constexpr (std::is_void_v<T>)
        return tensor->data_ptr();
    else
        return tensor->data_ptr<T>();
}

inline void check_sfatom_weight(const torch::Tensor& weight,
                                const torch::Tensor& sf, const char* op) {
    TORCH_CHECK(weight.scalar_type() == at::kFloat8_e4m3fn &&
                    sf.scalar_type() == at::kByte &&
                    sf.numel() == weight.numel() / SfLayout::Gran,
                op,
                ": weights must be float8_e4m3fn with uint8 SfAtom scales "
                "[rows, K / ",
                SfLayout::Gran, "]");
}

inline void check_rope_and_sink(const torch::Tensor& freqs_real,
                                const std::optional<torch::Tensor>& attn_sink,
                                const char* op) {
    using Dims = ModelDims;
    TORCH_CHECK(
        freqs_real.scalar_type() == at::kFloat && freqs_real.dim() == 2 &&
            freqs_real.size(1) == Dims::DimRope && freqs_real.is_contiguous(),
        op, ": freqs_real must be contiguous float32 [max_pos, ", Dims::DimRope,
        "]");
    TORCH_CHECK(
        !attn_sink.has_value() || (attn_sink->scalar_type() == at::kFloat &&
                                   attn_sink->numel() == Dims::Heads),
        op, ": attn_sink must be float32 [", Dims::Heads, "]");
}

template <class Fn>
decltype(auto) dispatch_tile_t(int num_tokens, bool cheap_attn, Fn&& fn) {
    using Tile = KernelBase::Tile;
    switch (select_tile_t(num_tokens, cheap_attn)) {
        case Tile::Widths[0]:
            return fn(KernelTraits<ModelDims, Tile::Widths[0]>{});
        case Tile::Widths[1]:
            return fn(KernelTraits<ModelDims, Tile::Widths[1]>{});
        case Tile::Widths[2]:
            return fn(KernelTraits<ModelDims, Tile::Widths[2]>{});
        default:
            return fn(KernelTraits<ModelDims, Tile::Widths[3]>{});
    }
}

template <class KT>
struct MlaAttentionProblem {
    using Dims = ModelDims;
    using WS = MlaWorkspace;
    using CfgQkv = typename KT::Qkv;
    static constexpr int TileT = KT::TileT;
    static_assert(
        CfgQkv::KSplit == WS::Qkv::KSplit &&
            std::is_same_v<typename CfgQkv::PartialT, WS::Qkv::PartialT>,
        "every token tile shares the workspace layout");

    int num_tokens, t_pad, t_pad_sf, grid_x, n_groups;
    int hidden, qlora_rank, kvlora_rank;

    WS::Layout wsl;

    char* base;
    cutlass::bfloat16_t* q_ptr;
    const cutlass::bfloat16_t* hidden_ptr;
    int hidden_row_stride;
    cutlass::bfloat16_t* o_partial;

    cutlass::float_e4m3_t* ws_x_fp8;
    cutlass::float_ue8m0_t* ws_x_sf;
    typename CfgQkv::PartialT* ws_qlora_part;
    cutlass::float_e4m3_t* ws_qlora_fp8;
    cutlass::float_ue8m0_t* ws_qlora_sf;
    typename CfgQkv::PartialT* ws_kv_part;
    cutlass::bfloat16_t* ws_flat_q_bf16;
    char* ws_c128_tok;
    int* c128_flags;

    static int w_rows(const torch::Tensor& weight) {
        return (int)(weight.numel() / weight.size(weight.dim() - 1));
    }
    static int w_cols(const torch::Tensor& weight) {
        return (int)weight.size(weight.dim() - 1);
    }

    static MlaAttentionProblem from(const MlaAttentionArguments& args) {
        MlaAttentionProblem params{};
        params.num_tokens = (int)args.q_or_x.size(0);
        const int num_tokens = params.num_tokens;

        params.hidden = w_cols(args.wq_a_weight);
        params.qlora_rank = w_rows(args.wq_a_weight);
        params.kvlora_rank = w_rows(args.wkv_weight);

        params.grid_x = mla::grid_x();
        params.n_groups = cute::ceil_div(num_tokens, TileT);
        params.wsl = WS::layout(num_tokens, params.grid_x);
        params.t_pad = params.wsl.t_pad;
        params.t_pad_sf = CfgQkv::SfTilePerGroup
                              ? WorkspaceLayout::SfPadRows * params.t_pad
                              : params.t_pad;

        params.base = reinterpret_cast<char*>(args.workspace.data_ptr());
        params.q_ptr =
            reinterpret_cast<cutlass::bfloat16_t*>(args.q_or_x.data_ptr());
        params.hidden_ptr = reinterpret_cast<const cutlass::bfloat16_t*>(
            args.hidden_states.data_ptr());
        params.hidden_row_stride = (int)args.hidden_states.stride(0);
        params.o_partial = reinterpret_cast<cutlass::bfloat16_t*>(
            params.base + params.wsl.kv_split_off);

        char* cur = params.base + params.wsl.token_off;
        const int t_pad = params.t_pad;
        auto take = [&cur, t_pad](int per_token) {
            char* got = cur;
            cur += (int64_t)t_pad * per_token;
            return got;
        };
        auto take_sf = [&take](int per_token) {
            return take(WorkspaceLayout::SfPadRows * per_token);
        };
        constexpr int kPartBytes = (int)sizeof(typename CfgQkv::PartialT);
        params.ws_x_fp8 =
            reinterpret_cast<cutlass::float_e4m3_t*>(take(params.hidden));
        params.ws_x_sf = reinterpret_cast<cutlass::float_ue8m0_t*>(
            take_sf(params.hidden / SfLayout::Gran));
        params.ws_qlora_part = reinterpret_cast<typename CfgQkv::PartialT*>(
            take(Dims::QLoraRank * kPartBytes * CfgQkv::KSplit));
        params.ws_qlora_fp8 =
            reinterpret_cast<cutlass::float_e4m3_t*>(take(Dims::QLoraRank));
        params.ws_qlora_sf = reinterpret_cast<cutlass::float_ue8m0_t*>(
            take_sf(Dims::QLoraRank / SfLayout::Gran));
        params.ws_kv_part = reinterpret_cast<typename CfgQkv::PartialT*>(
            take(Dims::DimV * kPartBytes * CfgQkv::KSplit));
        params.ws_flat_q_bf16 = reinterpret_cast<cutlass::bfloat16_t*>(
            take(Dims::QTotal * sizeof(cutlass::bfloat16_t)));
        params.ws_c128_tok = take(WS::C128TokenBytes);
        params.c128_flags =
            reinterpret_cast<int*>(params.base + params.wsl.c128_flags_off);
        return params;
    }
};

inline int check_attention_weights(const torch::Tensor& wq_a_weight,
                                   const torch::Tensor& wkv_weight,
                                   const char* op) {
    using Problem = MlaAttentionProblem<TraitsN128>;
    const int hidden = Problem::w_cols(wq_a_weight);
    TORCH_CHECK(Problem::w_rows(wq_a_weight) == ModelDims::QLoraRank &&
                    Problem::w_rows(wkv_weight) == ModelDims::DimV &&
                    Problem::w_cols(wkv_weight) == hidden &&
                    hidden % SfLayout::Gran == 0,
                op, ": needs wq_a [", ModelDims::QLoraRank, ", H], wkv [",
                ModelDims::DimV, ", H] with H a multiple of ", SfLayout::Gran);
    return hidden;
}
}  // namespace halops::mla
