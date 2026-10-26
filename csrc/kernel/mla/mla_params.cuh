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

#include "kernel/mla/mla_gemm.cuh"
#include "kernel/mla/mla_workspace.cuh"

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

template <class KT>
struct MlaAttentionProblem;

template <class KT>
struct CommonParams {
    int num_tokens;
    int draft_tokens;
    int reset_barriers;
    int prequant_x;
    int qnorm_dist;

    cutlass::bfloat16_t* __restrict__ q;
    cutlass::bfloat16_t* __restrict__ ws_flat_q_bf16;

    PlanView wplan;
    int* __restrict__ grid_bar;
    int* __restrict__ partial_bar;
    int pbar_bank;

    static CommonParams from(const MlaAttentionArguments& args,
                             const MlaAttentionProblem<KT>& prob) {
        CommonParams params{};
        params.num_tokens = prob.num_tokens;
        params.draft_tokens = (int)args.draft_tokens;
        params.reset_barriers = args.reset_barriers;
        params.prequant_x = prob.num_tokens <= KT::Attn::PrequantXMaxT;
        params.qnorm_dist = prob.num_tokens <= KT::Attn::QnormDistMaxT;
        params.q = prob.q_ptr;
        params.grid_bar = reinterpret_cast<int*>(prob.base);
        params.partial_bar =
            reinterpret_cast<int*>(prob.base + prob.wsl.pbar_off);
        params.pbar_bank = prob.wsl.pbar_bank;
        const bool hca = args.c128_attn_len.has_value();
        params.wplan = plan_ptrs(prob.base, prob.wsl.plan_off[hca],
                                 prob.num_tokens, prob.grid_x);
        params.ws_flat_q_bf16 = prob.ws_flat_q_bf16;
        return params;
    }
};

template <class KT, class TmaQ, class TmaKvGateW, class TmaKvGateAct>
struct MainloopParams : CommonParams<KT> {
    using Attn = typename KT::Attn;

    float sm_scale_div_log2;
    int page_block_size;
    cutlass::bfloat16_t* __restrict__ kv;
    float* __restrict__ attn_sink;
    int stride_kv_block;

    int page_shift;
    int rows_per_page;

    GemmShape shape_wq_a;
    GemmShape shape_wkv;
    GemmShape shape_wq_b;
    GemmShape shape_kv_gate;

    WeightAtoms<typename KT::Qkv> gemm_wq_a;
    WeightAtoms<typename KT::GemmBase> gemm_wq_b;
    WeightAtoms<typename KT::Qkv> gemm_wkv;
    ActAtoms<typename KT::GemmBase> act_x;
    ActAtoms<typename KT::GemmBase> act_qlora;

    const float* __restrict__ freqs_real;
    const int* __restrict__ positions;

    const cutlass::bfloat16_t* __restrict__ x_input;
    int hidden_row_stride;
    const cutlass::bfloat16_t* __restrict__ q_norm_weight;
    float q_eps;
    const cutlass::bfloat16_t* __restrict__ kv_norm_weight;
    float kv_eps;

    const int* __restrict__ swa_loc;

    int use_kv_gate;
    float* __restrict__ kv_score_ring;
    const int* __restrict__ kv_gate_ring_row;

    int use_c128;
    CUTE_HOST_DEVICE bool c128_compress() const {
        return use_kv_gate && use_c128;
    }
    const float* __restrict__ c128_ape;
    const cutlass::bfloat16_t* __restrict__ c128_norm_weight;
    const float* __restrict__ c128_freqs_real;
    const int* __restrict__ c128_out_loc;
    uint8_t* __restrict__ c128_pool;
    int c128_stride_kv_block;
    int c128_page_block;
    int c128_page_shift;
    int c128_rows_per_page;

    const int* __restrict__ c128_page_table;
    int c128_page_table_stride;
    const int* __restrict__ c128_attn_len;
    int use_c128_attn;
    char* __restrict__ ws_c128_tok;
    int* __restrict__ c128_flags;

    cutlass::float_e4m3_t* __restrict__ ws_x_fp8;
    cutlass::float_ue8m0_t* __restrict__ ws_x_sf;
    typename KT::Qkv::PartialT* __restrict__ ws_qlora_part;
    cutlass::float_e4m3_t* __restrict__ ws_qlora_fp8;
    cutlass::float_ue8m0_t* __restrict__ ws_qlora_sf;
    typename KT::Qkv::PartialT* __restrict__ ws_kv_part;

    cute::tuple<int, int, uint64_t> shape_Q;
    TmaQ tma_Q;
    TmaKvGateW tma_kv_gate_w;
    TmaKvGateAct tma_kv_gate_act;
    int kv_gate_act_rows;

    CUtensorMap tensor_map_kv_nope;
    CUtensorMap tensor_map_kv_rope;
    CUtensorMap kv_nope_dense[Attn::KvDenseLevels];
    CUtensorMap kv_rope_dense[Attn::KvDenseLevels];
    CUtensorMap c128_nope;
    CUtensorMap c128_rope;
    CUtensorMap c128_nope_dense[Attn::KvDenseLevels];
    CUtensorMap c128_rope_dense[Attn::KvDenseLevels];

    CUTE_DEVICE void prefetch_descriptors() const {
        cute::prefetch_tma_descriptor(tma_Q.get_tma_descriptor());
        cute::prefetch_tma_descriptor(&tensor_map_kv_nope);
        cute::prefetch_tma_descriptor(&tensor_map_kv_rope);
        CUTE_UNROLL
        for (int k = 0; k < Attn::KvDenseLevels; ++k) {
            cute::prefetch_tma_descriptor(&kv_nope_dense[k]);
            cute::prefetch_tma_descriptor(&kv_rope_dense[k]);
        }
        if (use_c128_attn) {
            cute::prefetch_tma_descriptor(&c128_nope);
            cute::prefetch_tma_descriptor(&c128_rope);
            CUTE_UNROLL
            for (int k = 0; k < Attn::KvDenseLevels; ++k) {
                cute::prefetch_tma_descriptor(&c128_nope_dense[k]);
                cute::prefetch_tma_descriptor(&c128_rope_dense[k]);
            }
        }
    }
};

template <class KT, class TmaO>
struct EpilogueParams : CommonParams<KT> {
    float* __restrict__ ws_lse_partial;

    cute::tuple<int, int, uint64_t> shape_O;
    TmaO tma_O;
    cute::tuple<int, int, uint64_t> shape_O_partial;
    TmaO tma_O_partial;
};

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

template <class KT, class MP>
struct KernelBars {
    using Dims = typename KT::Dims;
    using WS = MlaWorkspace;
    using Bar = PartialBarrier;
    static constexpr int TileT = KT::TileT;

    const MP& mp;

    CUTE_DEVICE int nv(int group) const {
        return min(TileT, mp.num_tokens - group * TileT);
    }
    CUTE_DEVICE Bar of(int bank, int group, int count) const {
        return Bar(mp.partial_bar, bank * mp.pbar_bank + group, count);
    }
    CUTE_DEVICE Bar per_token_cta(int bank, int group) const {
        return of(bank, group, nv(group));
    }

    CUTE_DEVICE Bar x_ready(int group) const {
        const int items = nv(group) * (mp.shape_wq_a.k_dim / SfLayout::Gran);
        return of(WS::BankXReady, group, QuantX::ctas(items, gridDim.x));
    }
    CUTE_DEVICE Bar qq_ready(int group) const {
        return per_token_cta(WS::BankQqReady, group);
    }
    CUTE_DEVICE Bar kvcache_ready(int group) const {
        return per_token_cta(WS::BankKvCacheReady, group);
    }
    CUTE_DEVICE Bar qlora_ready(int group) const {
        return of(WS::BankQloraReady, group,
                  mp.shape_wq_a.w_tiles * KT::Qkv::KSplit);
    }
    CUTE_DEVICE Bar kv_ready(int group) const {
        return of(WS::BankKvReady, group,
                  mp.shape_wkv.w_tiles * KT::Qkv::KSplit);
    }
    CUTE_DEVICE Bar ring_ready(int group) const {
        return of(WS::BankRingReady, group, mp.shape_kv_gate.w_tiles);
    }
    CUTE_DEVICE Bar flat_q_ready(int group) const {
        return of(WS::BankFlatQReady, group,
                  Dims::QTotal / KT::WqB::WeightTile);
    }
    CUTE_DEVICE static int qn_slice(int num_tokens) {
        const int n_heads = num_tokens * Dims::Heads;
        return (n_heads + gridDim.x - 1) / gridDim.x;
    }
    CUTE_DEVICE Bar qn_ready(int group) const {
        const int heads_per_cta = qn_slice(mp.num_tokens);
        const int lo = group * TileT * Dims::Heads;
        const int hi = min((group + 1) * TileT, mp.num_tokens) * Dims::Heads;
        return of(WS::BankQnReady, group,
                  (hi - 1) / heads_per_cta - lo / heads_per_cta + 1);
    }

    CUTE_DEVICE void reset_after_grid() const {
        int prev = 0;
        cta_sync();
        if (threadIdx.x == 0) prev = gmem_atom_add1_acq_rel(mp.grid_bar);
        if (__syncthreads_or(threadIdx.x == 0 && prev == (int)gridDim.x - 1)) {
            reset_all();
            if (threadIdx.x == 0) gmem_st_zero_relaxed(mp.grid_bar);
        }
    }

    CUTE_DEVICE void reset_all() const {
        const int n_bars = WS::NumBarrierBanks * mp.pbar_bank;
        for (int i = threadIdx.x; i < n_bars; i += blockDim.x)
            Bar(mp.partial_bar, i, 1).reset_relaxed();
        for (int i = threadIdx.x; i < C128Layout::FlagsPerToken * mp.num_tokens;
             i += blockDim.x)
            gmem_st_zero_relaxed(mp.c128_flags + i);
    }
};
}  // namespace halops::mla
