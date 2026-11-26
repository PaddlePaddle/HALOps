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

#include <math_constants.h>

#include "kernel/mla/mega_mla/attention/config.cuh"
#include "kernel/mla/mega_mla/mla_gemm.cuh"
#include "kernel/mla/mega_mla/mla_params.cuh"
#include "kernel/mla/mega_mla/mla_utils.cuh"

namespace halops::mla {
/// Host-side builder of the mainloop kernel parameters.
template <class KT>
struct AttentionMainloop {
    using Dims = ModelDims;
    using CfgQkv = typename KT::Qkv;
    using CfgWqB = typename KT::WqB;
    using CfgKvGate = typename KT::KvGate;

    template <class Params>
    static void set_c128_page_geometry(Params& params,
                                       const torch::Tensor& c128_pool,
                                       int64_t stride_kv_block,
                                       int64_t page_block) {
        params.c128_pool = reinterpret_cast<uint8_t*>(c128_pool.data_ptr());
        params.c128_stride_kv_block = (int)stride_kv_block;
        params.c128_page_block = (int)page_block;
        params.c128_page_shift = page_shift_of(page_block);
        params.c128_rows_per_page =
            params.c128_stride_kv_block / Dims::KvRowBytes;
    }

    template <int Levels>
    static void make_kv_maps(void* pool, int64_t rows, CUtensorMap& nope,
                             CUtensorMap (&nope_dense)[Levels],
                             CUtensorMap& rope,
                             CUtensorMap (&rope_dense)[Levels]) {
        constexpr int kNopeWords = Dims::DimNope / sizeof(int64_t);
        auto nope_map = [&](uint32_t box_rows) {
            return make_tensor_map(
                {kNopeWords, (uint64_t)rows}, {Dims::KvRowBytes},
                {kNopeWords, box_rows}, pool,
                CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_INT64,
                CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
                CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_128B);
        };
        auto rope_map = [&](uint32_t box_rows) {
            return make_tensor_map(
                {Dims::DimRope, (uint64_t)rows}, {Dims::KvRowBytes},
                {Dims::DimRope, box_rows}, (uint8_t*)pool + Dims::DimNope,
                CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
                CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_128B);
        };
        nope = nope_map(1);
        rope = rope_map(1);
        for (int k = 0; k < Levels; ++k) {
            nope_dense[k] = nope_map((uint32_t)(Sm100::Gather4Rows << k));
            rope_dense[k] = rope_map((uint32_t)(Sm100::Gather4Rows << k));
        }
    }

    static auto to_underlying_arguments(const MlaAttentionArguments& args,
                                        const MlaAttentionProblem<KT>& prob) {
        const int num_tokens = prob.num_tokens;

        auto shape_Q =
            cute::make_shape(Dims::Heads, Dims::DimQ, (uint64_t)num_tokens);
        auto tma_Q = cute::make_tma_copy(
            cute::SM90_TMA_LOAD{},
            cute::make_tensor(
                cute::make_gmem_ptr(prob.ws_flat_q_bf16),
                cute::make_layout(shape_Q,
                                  cute::make_stride(Dims::DimQ, cute::_1{},
                                                    (int64_t)Dims::QTotal))),
            AttnConfig::SmemLayoutQ_SW128{});
        const GemmShape shape_kv_gate =
            make_gemm_shape<CfgKvGate::WeightTile, CfgKvGate::BK,
                            CfgKvGate::KSplit>(2 * prob.kvlora_rank,
                                               prob.hidden);
        const bool use_kv_gate = args.wkv_gate_weight.has_value();
        auto tma_kv_gate_w = make_bf16_gemm_tma_w<CfgKvGate>(
            use_kv_gate ? args.wkv_gate_weight->data_ptr() : (void*)prob.q_ptr,
            shape_kv_gate.w_out, shape_kv_gate.k_dim);
        auto tma_kv_gate_act = make_bf16_gemm_tma_act<CfgKvGate>(
            (const void*)prob.hidden_ptr, num_tokens, prob.hidden_row_stride,
            shape_kv_gate.k_dim);
        MainloopParams<KT, decltype(tma_Q), decltype(tma_kv_gate_w),
                       decltype(tma_kv_gate_act)>
            params{};
        static_cast<CommonParams<KT>&>(params) =
            CommonParams<KT>::from(args, prob);
        params.shape_Q = shape_Q;
        params.tma_Q = tma_Q;
        params.tma_kv_gate_w = tma_kv_gate_w;
        params.tma_kv_gate_act = tma_kv_gate_act;
        params.kv_gate_act_rows = num_tokens;
        params.shape_kv_gate = shape_kv_gate;
        set_attention_args(params, args, prob);
        set_prelude_args(params, args, prob);
        return params;
    }

    template <class Params>
    static void set_attention_args(Params& params,
                                   const MlaAttentionArguments& args,
                                   const MlaAttentionProblem<KT>& prob) {
        params.sm_scale_div_log2 = (float)args.sm_scale * CUDART_L2E_F;
        params.page_block_size = (int)args.page_size;
        params.page_shift = page_shift_of(args.page_size);
        params.attn_sink = optional_ptr<float>(args.attn_sink);
        params.kv =
            reinterpret_cast<cutlass::bfloat16_t*>(args.kv_cache.data_ptr());
        params.stride_kv_block = (int)args.kv_cache.stride(0);
        params.rows_per_page = params.stride_kv_block / Dims::KvRowBytes;
        params.freqs_real = args.freqs_real.data_ptr<float>();
        params.positions = args.positions.data_ptr<int>();
        params.use_c128 = args.c128_ape.has_value() ? 1 : 0;
        if (params.use_c128) {
            params.c128_ape = args.c128_ape->data_ptr<float>();
            params.c128_norm_weight =
                reinterpret_cast<const cutlass::bfloat16_t*>(
                    args.c128_norm_weight->data_ptr());
            params.c128_freqs_real = args.c128_freqs_real->data_ptr<float>();
            params.c128_out_loc = args.c128_out_loc->data_ptr<int>();
            set_c128_page_geometry(params, *args.c128_pool,
                                   args.c128_stride_kv_block,
                                   args.c128_page_block);
        }
        params.use_c128_attn =
            (args.c128_page_table.has_value() && args.c128_attn_len.has_value())
                ? 1
                : 0;
        if (params.use_c128_attn) {
            if (!params.use_c128)
                set_c128_page_geometry(params, *args.c128_pool,
                                       args.c128_stride_kv_block,
                                       args.c128_page_block);
            params.c128_page_table_stride =
                (int)args.c128_page_table->stride(0);
            params.c128_page_table = args.c128_page_table->data_ptr<int>();
            params.c128_attn_len = args.c128_attn_len->data_ptr<int>();
        }
        const int64_t kv_tma_rows = (int64_t)args.kv_cache.size(0) *
                                    (params.stride_kv_block / Dims::KvRowBytes);
        make_kv_maps<KT::Attn::KvDenseLevels>(
            args.kv_cache.data_ptr(), kv_tma_rows, params.tensor_map_kv_nope,
            params.kv_nope_dense, params.tensor_map_kv_rope,
            params.kv_rope_dense);
        const int64_t c128_page_bytes = params.use_c128_attn
                                            ? params.c128_stride_kv_block
                                            : params.stride_kv_block;
        make_kv_maps<KT::Attn::KvDenseLevels>(
            params.use_c128_attn ? args.c128_pool->data_ptr()
                                 : args.kv_cache.data_ptr(),
            params.use_c128_attn
                ? (int64_t)(args.c128_pool->numel() / c128_page_bytes) *
                      (c128_page_bytes / Dims::KvRowBytes)
                : kv_tma_rows,
            params.c128_nope, params.c128_nope_dense, params.c128_rope,
            params.c128_rope_dense);
    }

    template <class Params>
    static void set_prelude_args(Params& params,
                                 const MlaAttentionArguments& args,
                                 const MlaAttentionProblem<KT>& prob) {
        using GemmGeomT = typename KT::GemmBase;
        const bool use_kv_gate = args.wkv_gate_weight.has_value();
        params.q_eps = (float)args.q_eps;
        params.kv_eps = (float)args.kv_eps;
        params.shape_wq_a =
            make_gemm_shape<CfgQkv::WeightTile, CfgQkv::BK, CfgQkv::KSplit>(
                prob.qlora_rank, prob.hidden);
        params.shape_wkv =
            make_gemm_shape<CfgQkv::WeightTile, CfgQkv::BK, CfgQkv::KSplit>(
                prob.kvlora_rank, prob.hidden);
        params.shape_wq_b =
            make_gemm_shape<CfgWqB::WeightTile, CfgWqB::BK, CfgWqB::KSplit>(
                Dims::QTotal, prob.qlora_rank);
        params.x_input = prob.hidden_ptr;
        params.hidden_row_stride = prob.hidden_row_stride;
        params.q_norm_weight = reinterpret_cast<const cutlass::bfloat16_t*>(
            args.q_norm_weight.data_ptr());
        params.kv_norm_weight = reinterpret_cast<const cutlass::bfloat16_t*>(
            args.kv_norm_weight.data_ptr());
        params.swa_loc = args.swa_loc.data_ptr<int>();
        params.use_kv_gate = use_kv_gate ? 1 : 0;
        if (use_kv_gate) {
            params.kv_score_ring =
                reinterpret_cast<float*>(args.kv_score_ring->data_ptr());
            params.kv_gate_ring_row =
                reinterpret_cast<const int*>(args.kv_gate_ring_row->data_ptr());
        }
        params.gemm_wq_a = make_weight_atoms<typename KT::Qkv>(
            args.wq_a_weight.data_ptr(), args.wq_a_sf.data_ptr(),
            params.shape_wq_a.w_out, params.shape_wq_a.k_dim);
        params.gemm_wq_b = make_weight_atoms<GemmGeomT>(
            args.wq_b_weight.data_ptr(), args.wq_b_sf.data_ptr(),
            params.shape_wq_b.w_out, params.shape_wq_b.k_dim);
        params.gemm_wkv = make_weight_atoms<typename KT::Qkv>(
            args.wkv_weight.data_ptr(), args.wkv_sf.data_ptr(),
            params.shape_wkv.w_out, params.shape_wkv.k_dim);
        params.ws_x_fp8 = prob.ws_x_fp8;
        params.ws_x_sf = prob.ws_x_sf;
        params.ws_qlora_part = prob.ws_qlora_part;
        params.ws_qlora_fp8 = prob.ws_qlora_fp8;
        params.ws_qlora_sf = prob.ws_qlora_sf;
        params.ws_kv_part = prob.ws_kv_part;
        params.ws_c128_tok = prob.ws_c128_tok;
        params.c128_flags = prob.c128_flags;
        params.act_x = make_act_atoms<GemmGeomT>(
            params.ws_x_fp8, params.ws_x_sf, params.shape_wq_a.w_out,
            prob.t_pad, prob.hidden, prob.t_pad_sf);
        params.act_qlora = make_act_atoms<GemmGeomT>(
            params.ws_qlora_fp8, params.ws_qlora_sf, params.shape_wq_b.w_out,
            prob.t_pad, ModelDims::QLoraRank, prob.t_pad_sf);
    }
};

/// Host-side builder of the epilogue kernel parameters.
template <class KT>
struct AttentionEpilogue {
    using Dims = typename KT::Dims;

    template <class Op>
    static auto make_o_like_tma(cutlass::bfloat16_t* base, int64_t pages) {
        return cute::make_tma_copy(
            Op{},
            cute::make_tensor(
                cute::make_gmem_ptr(base),
                cute::make_layout(
                    cute::make_shape(Dims::Heads, Dims::DimV, (uint64_t)pages),
                    cute::make_stride(Dims::DimV, cute::_1{},
                                      (int64_t)(Dims::DimV * Dims::Heads)))),
            AttnConfig::SmemLayoutOBuf_TMA{});
    }

    static auto to_underlying_arguments(const MlaAttentionArguments& args,
                                        const MlaAttentionProblem<KT>& prob) {
        const int num_tokens = prob.num_tokens;
        const int slots = split_slots(num_tokens, prob.grid_x);

        auto shape_O =
            cute::make_shape(Dims::Heads, Dims::DimV, (uint64_t)num_tokens);
        auto shape_O_partial =
            cute::make_shape(Dims::Heads, Dims::DimV, (uint64_t)slots);
        auto tma_O =
            make_o_like_tma<cute::SM90_TMA_STORE>(prob.q_ptr, num_tokens);
        auto tma_O_partial =
            make_o_like_tma<cute::SM90_TMA_STORE>(prob.o_partial, slots);
        EpilogueParams<KT, decltype(tma_O)> params{};
        static_cast<CommonParams<KT>&>(params) =
            CommonParams<KT>::from(args, prob);
        params.shape_O = shape_O;
        params.tma_O = tma_O;
        params.shape_O_partial = shape_O_partial;
        params.tma_O_partial = tma_O_partial;
        params.ws_lse_partial =
            reinterpret_cast<float*>(prob.base + prob.wsl.lse_off);
        return params;
    }
};
}  // namespace halops::mla
