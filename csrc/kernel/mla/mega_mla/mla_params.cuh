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

#include "kernel/mla/common/attention_problem.cuh"
#include "kernel/mla/mega_mla/mla_gemm.cuh"
#include "kernel/mla/mega_mla/mla_workspace.cuh"

namespace halops::mla {
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
