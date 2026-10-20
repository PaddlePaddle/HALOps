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

#include "mla_gemm.cuh"
#include "mla_params.cuh"
#include "mla_plan.cuh"
#include "mla_traits.cuh"
#include "mla_utils.cuh"

namespace mega_mla {

using namespace cute;

using e8m0 = __nv_fp8_e8m0;

#ifndef MEGA_MLA_OPOST_BATCH
#    define MEGA_MLA_OPOST_BATCH 4
#endif

template <class Ctx>
struct PerToken {
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
    int c128_slot0;
    KvRuns kv_runs;
    KvSegGeom kv_seg_swa;
    KvSegGeom kv_seg_c128;
    int blk_lo;
    int blk_hi;
    int slot;

    CUTE_DEVICE KvRuns kv_tile_at(int block_idx) const {
        if (block_idx < swa_blocks)
            return kv_tile_slice(kv_runs, block_idx * AttnConfig::B_TOPK,
                                 AttnConfig::B_TOPK);
        return kv_tile_ascending(c128_slot0, c128_len,
                                 (block_idx - swa_blocks) * AttnConfig::B_TOPK,
                                 AttnConfig::B_TOPK, params.c128_page_shift,
                                 params.c128_rows_per_page);
    }
    CUTE_DEVICE const KvSegGeom& kv_seg_at(int block_idx) const {
        return block_idx < swa_blocks ? kv_seg_swa : kv_seg_c128;
    }

    CUTE_DEVICE int kv_tile_valid(int block_idx) const {
        return block_idx < swa_blocks
                   ? topk_length - block_idx * AttnConfig::B_TOPK
                   : c128_len - (block_idx - swa_blocks) * AttnConfig::B_TOPK;
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
    const mega_mla::CtaWork w =
        k.params.wplan.work[token_it * (int)gridDim.x + (int)blockIdx.x];
    const bool token_active = w.blk_hi > w.blk_lo;
    const int token_idx = w.token;

    const int topk_length =
        !token_active
            ? (int)AttnConfig::B_TOPK
            : (k.params.topk_length ? __ldg(k.params.topk_length + token_idx)
                                    : k.params.topk);
    const int total_topk_padded =
        max(ku::ceil(topk_length, (int)AttnConfig::B_TOPK),
            (int)AttnConfig::B_TOPK);
    const int swa_blocks = total_topk_padded / AttnConfig::B_TOPK;
    const int c128_len = (k.params.c128_attn_len && token_active)
                             ? __ldg(k.params.c128_attn_len + token_idx)
                             : 0;
    const int c128_slot0 = (k.params.c128_attn_slot0 && token_active)
                               ? __ldg(k.params.c128_attn_slot0 + token_idx)
                               : 0;

    const KvRuns kv_runs = load_kv_runs(
        k.params.kv_runs, token_active ? token_idx : 0, topk_length,
        k.params.kv_run_step, k.params.page_shift, k.params.rows_per_page);

    const KvSegGeom kv_seg_swa{reinterpret_cast<const uint8_t*>(k.params.kv),
                               k.params.stride_kv_block};
    const KvSegGeom kv_seg_c128{k.params.c128_pool,
                                k.params.c128_stride_kv_block};

    return PerToken<Ctx>{k.params,   k.ep,          k.plan,
                         k.rs,       k.q_tma_phase, k.q_utccp_phase,
                         k.warp_idx, k.lane_idx,    k.idx_in_warpgroup,
                         token_idx,  token_active,  topk_length,
                         swa_blocks, c128_len,      c128_slot0,
                         kv_runs,    kv_seg_swa,    kv_seg_c128,
                         w.blk_lo,   w.blk_hi,      w.slot};
}

template <class Ctx, class Tok>
CUTE_DEVICE void phase1_prepare_q(Ctx const& c, Tok const& t) {
    if (c.params.use_phase1 && c.warpgroup_idx < 2) {
        if (c.params.use_phase1) {
            if (c.warp_idx == 0)
                c.bars.flat_q_ready(t.token_idx / c.tile_t)
                    .wait_warp(0, mega_mla::TraceProbe::WS_PBAR_FLAT_Q,
                               t.token_idx / c.tile_t);
            mega_mla::bar_sync<mega_mla::KernelTrait::Bars::Wg01Sync>();
        }

        const int cta_tid = threadIdx.x;

        Tensor sQ = make_tensor(make_smem_ptr(c.plan.u.qo.q.data()),
                                AttnConfig::SmemLayoutQ_SW128{});

        constexpr int NUM_WARPS = 2 * mega_mla::KernelTrait::WarpsPerWarpGroup;
        static_assert(mega_mla::KernelTrait::Bars::Wg01Sync::width ==
                          NUM_WARPS * mega_mla::KernelTrait::WarpSize,
                      "the head walk's stride and the barrier this "
                      "region joins on must name the same warp set");
        static_assert(ModelDims::B_H % NUM_WARPS == 0,
                      "otherwise the last pass has idle warps whose "
                      "lanes still reach the shfl reduction");
        const int lane = cta_tid & 31, warp = cta_tid >> 5;
        const int pos = c.params.positions[t.token_idx];
        const float* fr = c.params.freqs_real + pos * ModelDims::D_ROPE;
        const auto* gq =
            reinterpret_cast<const __nv_bfloat16*>(c.params.ws_flat_q_bf16) +
            (int64_t)t.token_idx * ModelDims::Q_TOTAL;
        for (int h = warp; h < ModelDims::B_H; h += NUM_WARPS) {
            const __nv_bfloat16* grow = gq + h * ModelDims::D_Q;
            __nv_bfloat16 v[16];
            *reinterpret_cast<uint4*>(v) =
                *reinterpret_cast<const uint4*>(grow + lane * 8);
            *reinterpret_cast<uint4*>(v + 8) =
                *reinterpret_cast<const uint4*>(grow + (lane + 32) * 8);
            float sq = 0.0f;
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < 16; ++i) {
                const float x = __bfloat162float(v[i]);
                sq += x * x;
            }
            CUTLASS_PRAGMA_UNROLL
            for (int o = 16; o > 0; o >>= 1)
                sq += __shfl_xor_sync(0xffffffff, sq, o);
            const float inv = rsqrtf(sq / ModelDims::D_Q + c.params.q_eps);

            CUTLASS_PRAGMA_UNROLL
            for (int half = 0; half < 2; ++half) {
                const int base = (half == 0 ? lane : lane + 32) * 8;
                __nv_bfloat16 o8[8];
                float f[8];
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < 8; ++i)
                    f[i] = __bfloat162float(v[half * 8 + i]) * inv;

                if (base >= ModelDims::D_NOPE) {
                    CUTLASS_PRAGMA_UNROLL
                    for (int k = 0; k < 4; ++k) {
                        const int pi = (base - ModelDims::D_NOPE) / 2 + k;
                        const float c = fr[pi * 2], s_v = fr[pi * 2 + 1];
                        const float a = f[k * 2], b = f[k * 2 + 1];
                        f[k * 2] = a * c - b * s_v;
                        f[k * 2 + 1] = a * s_v + b * c;
                    }
                }
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < 8; ++i) o8[i] = __float2bfloat16(f[i]);

                *reinterpret_cast<uint4*>(&sQ(h, base)) =
                    *reinterpret_cast<const uint4*>(o8);
            }
        }

        cutlass::arch::fence_view_async_shared();

        mega_mla::bar_sync<mega_mla::KernelTrait::Bars::Wg01Sync>();
    }
}

template <class KT>
struct CollectiveMainloop {
    using CfgQkv = typename KT::Qkv;
    using CfgWqB = typename KT::WqB;
    using CfgKvGate = typename KT::KvGate;
    using CfgWoA = typename KT::WoA;
    using CfgWoB = typename KT::WoB;
    static constexpr int BN = CfgQkv::BN;
    static constexpr int N_PAD = KT::TileT;

    static int page_shift_of(int64_t page) {
        int s = 0;
        while ((1LL << s) < page) ++s;
        return s;
    }

    template <class P>
    static void set_c128_page_geometry(P& p, const torch::Tensor& c128_pool,
                                       int64_t stride_kv_block,
                                       int64_t page_block) {
        TORCH_CHECK((stride_kv_block & 15) == 0 && stride_kv_block > 0,
                    "c128_stride_kv_block must be a positive multiple of 16");
        TORCH_CHECK(page_block > 0 && (page_block & (page_block - 1)) == 0,
                    "c128_page_block must be a power of two");
        p.c128_pool = reinterpret_cast<uint8_t*>(c128_pool.data_ptr());
        p.c128_stride_kv_block = (int)stride_kv_block;
        p.c128_page_block = (int)page_block;
        p.c128_page_shift = page_shift_of(page_block);
        p.c128_rows_per_page = p.c128_stride_kv_block / ModelDims::TMA_K_STRIDE;
        TORCH_CHECK(p.c128_rows_per_page >= p.c128_page_block,
                    "each c128 page must hold at least c128_page_block TMA "
                    "rows, got ",
                    p.c128_rows_per_page, " < ", p.c128_page_block);
    }

    template <int Levels>
    static void make_kv_maps(void* pool, int64_t rows, CUtensorMap& nope,
                             CUtensorMap (&nope_dense)[Levels],
                             CUtensorMap& rope,
                             CUtensorMap (&rope_dense)[Levels]) {
        using D = ModelDims;
        auto nope_map = [&](uint32_t box_rows) {
            return ku::make_tensor_map(
                {D::D_NOPE / 8, (uint64_t)rows}, {D::TMA_K_STRIDE},
                {D::D_NOPE / 8, box_rows}, pool,
                CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_INT64,
                CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
                CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_128B);
        };
        auto rope_map = [&](uint32_t box_rows) {
            return ku::make_tensor_map(
                {D::D_ROPE, (uint64_t)rows}, {D::TMA_K_STRIDE},
                {D::K_ROPE_SW / 2, box_rows}, (uint8_t*)pool + D::D_NOPE,
                CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                D::K_ROPE_SW == 64
                    ? CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_64B
                    : CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
                CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_128B);
        };
        nope = nope_map(1);
        rope = rope_map(1);
        for (int k = 0; k < Levels; ++k) {
            nope_dense[k] = nope_map((uint32_t)(4u << k));
            rope_dense[k] = rope_map((uint32_t)(4u << k));
        }
    }

    static auto to_underlying_arguments(
        const halops::MegaMlaArguments& args,
        const halops::MegaMlaProblem<KT>& prob) {
        using D = ModelDims;
        using Attn = typename KT::Attn;
        using CfgA = typename KT::Qkv;
        using CfgB = typename KT::WqB;
        using CfgKvG = typename KT::KvGate;
        constexpr int TileT = KT::TileT;
        constexpr int LV = Attn::KvDenseLevels;
        const int T = prob.num_tokens;

        auto shape_Q = cute::make_shape(D::B_H, D::D_Q, (uint64_t)T);
        auto tma_Q = cute::make_tma_copy(
            cute::SM90_TMA_LOAD{},
            cute::make_tensor(
                cute::make_gmem_ptr(prob.q_ptr),
                cute::make_layout(shape_Q,
                                  cute::make_stride(D::D_Q, cute::_1{},
                                                    (int64_t)D::Q_TOTAL))),
            mega_mla::AttnConfig::SmemLayoutQ_SW128{});
        const mega_mla::GemmShape shape_kv_gate =
            mega_mla::make_gemm_shape<CfgKvG::BM, CfgKvG::BK, CfgKvG::KSplit,
                                      CfgKvG::TmaStages>(2 * prob.kvlora_rank,
                                                         prob.hidden);
        const bool use_kv_gate = args.wkv_gate_weight.numel() > 0;
        auto tma_kv_gate_w = mega_mla::make_bf16_gemm_tma_w<CfgKvG>(
            use_kv_gate ? args.wkv_gate_weight.data_ptr() : (void*)prob.q_ptr,
            shape_kv_gate.m_total, shape_kv_gate.k_dim);
        auto tma_kv_gate_act = mega_mla::make_bf16_gemm_tma_act<CfgKvG>(
            (const void*)prob.q_ptr, T, D::Q_TOTAL, shape_kv_gate.k_dim);
        halops::MainloopParams<KT, decltype(tma_Q), decltype(tma_kv_gate_w),
                               decltype(tma_kv_gate_act)>
            p{};
        static_cast<halops::CommonParams<KT>&>(p) =
            halops::CommonParams<KT>::from(args, prob);
        p.shape_Q = shape_Q;
        p.tma_Q = tma_Q;
        p.tma_kv_gate_w = tma_kv_gate_w;
        p.tma_kv_gate_act = tma_kv_gate_act;
        p.kv_gate_act_rows = T;
        p.shape_kv_gate = shape_kv_gate;
        p.sm_scale_div_log2 = (float)args.sm_scale * CUDART_L2E_F;
        p.num_blocks = (int)((args.topk + Attn::B_TOPK - 1) / Attn::B_TOPK);
        p.page_block_size = (int)args.page_size;
        p.page_shift = page_shift_of(args.page_size);
        p.kv_run_step = (int)args.kv_run_step;
        p.topk = (int)args.topk;
        p.topk_length = args.topk_lengths.data_ptr<int>();
        p.attn_sink = args.attn_sink.numel() > 0
                          ? args.attn_sink.data_ptr<float>()
                          : nullptr;
        p.kv = reinterpret_cast<mega_mla::bf16*>(args.kv_cache.data_ptr());
        p.kv_runs = reinterpret_cast<const int4*>(args.kv_runs.data_ptr<int>());
        p.stride_kv_block = (int)args.kv_cache.stride(0);
        p.rows_per_page = p.stride_kv_block / D::TMA_K_STRIDE;
        TORCH_CHECK(p.rows_per_page >= p.page_block_size,
                    "each page must hold at least page_size TMA rows, got ",
                    p.rows_per_page, " < ", p.page_block_size);
        p.freqs_real = args.freqs_real.data_ptr<float>();
        p.positions = args.positions.data_ptr<int>();
        p.q_eps = (float)args.q_eps;
        p.kv_eps = (float)args.kv_eps;
        p.shape_wq_a = mega_mla::make_gemm_shape<CfgA::BM, CfgA::BK,
                                                 CfgA::KSplit, CfgA::TmaStages>(
            prob.qlora_rank, prob.hidden);
        p.shape_wkv = mega_mla::make_gemm_shape<CfgA::BM, CfgA::BK,
                                                CfgA::KSplit, CfgA::TmaStages>(
            prob.kvlora_rank, prob.hidden);
        p.shape_wq_b = mega_mla::make_gemm_shape<CfgB::BM, CfgB::BK,
                                                 CfgB::KSplit, CfgB::TmaStages>(
            D::Q_TOTAL, prob.qlora_rank);
        p.use_phase1 = args.use_phase1 ? 1 : 0;
        if (args.use_phase1) {
            p.x_input = reinterpret_cast<const mega_mla::bf16*>(prob.q_ptr);
            TORCH_CHECK(
                reinterpret_cast<uintptr_t>(args.q_norm_weight.data_ptr()) %
                        8 ==
                    0,
                "q_norm_weight must be 8 B aligned (its RMSNorm reads it as "
                "uint2)");
            p.q_norm_weight = reinterpret_cast<const mega_mla::bf16*>(
                args.q_norm_weight.data_ptr());
        }
        p.use_kv_write = args.use_kv_write ? 1 : 0;
        if (args.use_kv_write) {
            TORCH_CHECK(
                reinterpret_cast<uintptr_t>(args.kv_norm_weight.data_ptr()) %
                        8 ==
                    0,
                "kv_norm_weight must be 8 B aligned (kv_write_line reads it as "
                "uint2)");
            p.kv_norm_weight = reinterpret_cast<const mega_mla::bf16*>(
                args.kv_norm_weight.data_ptr());
            p.swa_loc = args.swa_loc.data_ptr<int>();
        }
        p.use_kv_gate = use_kv_gate ? 1 : 0;
        if (use_kv_gate) {
            TORCH_CHECK(args.wkv_gate_weight.scalar_type() == torch::kBFloat16,
                        "wkv_gate_weight must be bf16 -- the upstream op is "
                        "linear_bf16_fp32 and quantizing it would change the "
                        "numerics");
            TORCH_CHECK(
                args.wkv_gate_weight.dim() == 2 &&
                    args.wkv_gate_weight.size(0) == prob.wsd.kv_gate_k &&
                    args.wkv_gate_weight.size(1) == prob.wsd.x_k,
                "wkv_gate_weight must be [", prob.wsd.kv_gate_k, ", ",
                prob.wsd.x_k, "], got ", args.wkv_gate_weight.sizes());
            TORCH_CHECK(args.wkv_gate_weight.stride(1) == 1 &&
                            args.wkv_gate_weight.stride(0) == prob.wsd.x_k,
                        "wkv_gate_weight must be contiguous K-major");
            p.wkv_gate_weight = reinterpret_cast<const mega_mla::bf16*>(
                args.wkv_gate_weight.data_ptr());
            TORCH_CHECK(
                args.kv_score_ring.scalar_type() == torch::kFloat,
                "kv_score_ring must be fp32; the epilogue stores the fp32 "
                "accumulator without rounding");
            TORCH_CHECK(args.kv_score_ring.dim() == 2 &&
                            args.kv_score_ring.size(1) == prob.wsd.kv_gate_k,
                        "kv_score_ring must be [slots, ", prob.wsd.kv_gate_k,
                        "] = | kv 512 | score 512 |, got ",
                        args.kv_score_ring.sizes());
            TORCH_CHECK(
                args.kv_score_ring.stride(1) == 1 &&
                    args.kv_score_ring.stride(0) == prob.wsd.kv_gate_k,
                "kv_score_ring must be contiguous: the epilogue indexes it "
                "as ring_row * KV_GATE_K + col");
            TORCH_CHECK(args.kv_gate_ring_row.scalar_type() == torch::kInt &&
                            args.kv_gate_ring_row.numel() == T,
                        "kv_gate_ring_row must be int32 [T]=", T);
            p.kv_score_ring =
                reinterpret_cast<float*>(args.kv_score_ring.data_ptr());
            p.kv_gate_ring_row =
                reinterpret_cast<const int*>(args.kv_gate_ring_row.data_ptr());
        }
        p.use_c128 = args.c128_ape.numel() > 0 ? 1 : 0;
        if (p.use_c128) {
            TORCH_CHECK(
                p.use_kv_gate && args.kv_gate_ring_row.numel() > 0,
                "c128 consumes the kv_score_ring: pass wkv_gate_weight and "
                "kv_score_ring / kv_gate_ring_row");
            TORCH_CHECK(args.c128_ape.scalar_type() == torch::kFloat &&
                            args.c128_ape.dim() == 2 &&
                            args.c128_ape.size(0) == D::C128_RATIO &&
                            args.c128_ape.size(1) == D::D_V,
                        "c128_ape must be fp32 [", (int)D::C128_RATIO, ", ",
                        (int)D::D_V, "] (per-ring-slot score bias)");
            TORCH_CHECK(args.c128_ape.stride(1) == 1 &&
                            args.c128_ape.stride(0) == D::D_V,
                        "c128_ape must be contiguous");
            TORCH_CHECK(
                args.c128_norm_weight.scalar_type() == torch::kBFloat16 &&
                    args.c128_norm_weight.numel() == D::D_V,
                "c128_norm_weight must be bf16 [", (int)D::D_V, "]");
            TORCH_CHECK(args.c128_freqs_real.scalar_type() == torch::kFloat &&
                            args.c128_freqs_real.dim() == 2 &&
                            args.c128_freqs_real.size(1) == D::D_ROPE,
                        "c128_freqs_real must be fp32 [max_pos, ",
                        (int)D::D_ROPE, "] (theta=compress_rope_theta table)");
            TORCH_CHECK(args.c128_out_loc.scalar_type() == torch::kInt &&
                            args.c128_out_loc.numel() == T,
                        "c128_out_loc must be int32 [T]=", T);
            TORCH_CHECK(args.c128_row_out.numel() == 0 ||
                            (args.c128_row_out.scalar_type() == torch::kFloat &&
                             args.c128_row_out.numel() == (int64_t)T * D::D_V),
                        "c128_row_out must be fp32 [T, ", (int)D::D_V,
                        "] or empty (probe gate A)");
            p.c128_ape = args.c128_ape.data_ptr<float>();
            p.c128_norm_weight = reinterpret_cast<const mega_mla::bf16*>(
                args.c128_norm_weight.data_ptr());
            p.c128_freqs_real = args.c128_freqs_real.data_ptr<float>();
            p.c128_out_loc = args.c128_out_loc.data_ptr<int>();
            p.c128_row_out =
                args.c128_row_out.numel()
                    ? reinterpret_cast<float*>(args.c128_row_out.data_ptr())
                    : nullptr;
            set_c128_page_geometry(p, args.c128_pool, args.c128_stride_kv_block,
                                   args.c128_page_block);
        }
        p.use_c128_attn =
            (args.c128_attn_slot0.numel() > 0 && args.c128_attn_len.numel() > 0)
                ? 1
                : 0;
        if (p.use_c128_attn) {
            TORCH_CHECK(args.c128_pool.numel() > 0,
                        "the c128 attention segment needs c128_pool");
            TORCH_CHECK(args.c128_attn_slot0.scalar_type() == torch::kInt &&
                            args.c128_attn_slot0.numel() == T,
                        "c128_attn_slot0 must be int32 [T]=", T);
            TORCH_CHECK(args.c128_attn_len.scalar_type() == torch::kInt &&
                            args.c128_attn_len.numel() == T,
                        "c128_attn_len must be int32 [T]=", T);
            if (!p.use_c128)
                set_c128_page_geometry(p, args.c128_pool,
                                       args.c128_stride_kv_block,
                                       args.c128_page_block);
            TORCH_CHECK(
                p.c128_page_block >= Attn::B_TOPK,
                "the c128 attention segment needs c128_page_block >= "
                "B_TOPK (",
                (int)Attn::B_TOPK,
                ") so a KV tile crosses at most one page boundary; got ",
                p.c128_page_block,
                ". This is why the c128 pool is allocated with SWA's page "
                "geometry (128) instead of the page_size/128 the full-pool "
                "page table implies.");
            p.c128_attn_slot0 = args.c128_attn_slot0.data_ptr<int>();
            p.c128_attn_len = args.c128_attn_len.data_ptr<int>();
        }
        const int64_t kv_tma_rows = (int64_t)args.kv_cache.size(0) *
                                    (p.stride_kv_block / D::TMA_K_STRIDE);
        make_kv_maps<LV>(args.kv_cache.data_ptr(), kv_tma_rows,
                         p.tensor_map_kv_nope, p.kv_nope_dense,
                         p.tensor_map_kv_rope, p.kv_rope_dense);
        const int64_t c128_page_bytes =
            p.use_c128_attn ? p.c128_stride_kv_block : p.stride_kv_block;
        make_kv_maps<LV>(
            p.use_c128_attn ? args.c128_pool.data_ptr()
                            : args.kv_cache.data_ptr(),
            p.use_c128_attn
                ? (int64_t)(args.c128_pool.numel() / c128_page_bytes) *
                      (c128_page_bytes / D::TMA_K_STRIDE)
                : kv_tma_rows,
            p.c128_nope, p.c128_nope_dense, p.c128_rope, p.c128_rope_dense);
        if (args.use_phase1) {
            p.gemm_wq_a = mega_mla::make_weight_atoms<TileT>(
                args.wq_a_weight.data_ptr(), args.wq_a_sf_g32.data_ptr(),
                p.shape_wq_a.m_total, p.shape_wq_a.k_dim);
            p.gemm_wq_b = mega_mla::make_weight_atoms<TileT>(
                args.wq_b_weight.data_ptr(), args.wq_b_sf_g32.data_ptr(),
                p.shape_wq_b.m_total, p.shape_wq_b.k_dim);
        }
        if (args.use_kv_write)
            p.gemm_wkv = mega_mla::make_weight_atoms<TileT>(
                args.wkv_weight.data_ptr(), args.wkv_sf_g32.data_ptr(),
                p.shape_wkv.m_total, p.shape_wkv.k_dim);
        p.ws_x_fp8 = prob.ws_x_fp8;
        p.ws_x_sf = prob.ws_x_sf;
        p.ws_qlora_part = prob.ws_qlora_part;
        p.ws_qlora_fp8 = prob.ws_qlora_fp8;
        p.ws_qlora_sf = prob.ws_qlora_sf;
        p.ws_kv_part = prob.ws_kv_part;
        p.act_x = mega_mla::make_act_atoms<TileT>(
            p.ws_x_fp8, p.ws_x_sf, p.shape_wq_a.m_total, prob.t_pad,
            prob.wsd.x_k, prob.t_pad_sf);
        p.act_qlora = mega_mla::make_act_atoms<TileT>(
            p.ws_qlora_fp8, p.ws_qlora_sf, p.shape_wq_b.m_total, prob.t_pad,
            prob.wsd.qlora_k, prob.t_pad_sf);
        return p;
    }

    template <class Ctx>
    CUTE_DEVICE void mma(Ctx const& c) const {
        {
            const RangeBarrierLedger led = c.sched.ledger_before_kv_gate();
            auto op = make_bf16_mma_op<CfgKvGate>(
                c.r.res_kv_gate, led.base_bits(), led.base_acc());
            for (int item = (int)blockIdx.x; item < c.sched.gemm.items_kv_gate;
                 item += gridDim.x) {
                op.begin_item(TraceProbe::WS_KVG_ACC_EMPTY, item);
                CUTLASS_PRAGMA_NO_UNROLL
                for (int k = 0; k < c.sched.k_tiles_kvg; ++k) {
                    op.wait_tma(k, TraceProbe::WS_KVG_TMA_W,
                                TraceProbe::WS_KVG_TMA_ACT, item);
                    op.run_k_tile(k);
                    op.commit(k);
                }
                op.end_item();
            }
        }
        {
            const RangeBarrierLedger led = c.sched.ledger_before_qkv();
            auto op = make_mma_op<CfgQkv>(c.r.res_qkv, led.base_bits(),
                                          led.base_acc(), c.sched.k_tps_qkv);
            for (int item = (int)blockIdx.x; item < c.sched.gemm.items_qkv;
                 item += gridDim.x) {
                op.begin_item(c.r.qkv_item(c.sched, item).wi,
                              TraceProbe::WS_QKV_ACC_EMPTY, item);
                CUTLASS_PRAGMA_NO_UNROLL
                for (int k = 0; k < c.sched.k_tps_qkv; ++k) {
                    op.wait_tma(k, TraceProbe::WS_QKV_TMA_W,
                                TraceProbe::WS_QKV_TMA_ACT, item);
                    op.run_k_tile(k);
                    op.commit(k);
                }
                op.end_item();
            }
        }
        {
            const RangeBarrierLedger led = c.sched.ledger_before_wq_b();
            auto op = make_mma_op<CfgWqB>(c.r.res_wqb, led.base_bits(),
                                          led.base_acc(), c.sched.k_tps_wqb);
            for (int item = (int)blockIdx.x; item < c.sched.gemm.items_wq_b;
                 item += gridDim.x) {
                op.begin_item(c.sched.wqb_item(item),
                              TraceProbe::WS_WQB_ACC_EMPTY, item);
                CUTLASS_PRAGMA_NO_UNROLL
                for (int k = 0; k < c.sched.k_tps_wqb; ++k) {
                    op.wait_tma(k, TraceProbe::WS_WQB_TMA_W,
                                TraceProbe::WS_WQB_TMA_ACT, item);
                    op.run_k_tile(k);
                    op.commit(k);
                }
                op.end_item();
            }
        }

        __syncwarp();
        mega_mla::bar_sync<mega_mla::KernelTrait::Bars::EveryoneSync>();

        if (elect_one_sync()) c.plan.init_gemm_bars();
        cutlass::arch::fence_barrier_init();
        mega_mla::bar_sync<mega_mla::KernelTrait::Bars::EveryoneSync>();
    }

    template <class Ctx>
    CUTE_DEVICE void tma_a(Ctx const& c) const {
        {
            const RangeBarrierLedger led = c.sched.ledger_before_kv_gate();
            auto op = make_bf16_weight_tma_op<CfgKvGate>(
                c.r.res_kv_gate, led.base_bits(), led.base_zero_bits());
            for (int item = (int)blockIdx.x; item < c.sched.gemm.items_kv_gate;
                 item += gridDim.x) {
                op.begin_item(bf16_gemm_item_at<CfgKvGate>(
                    item, c.sched.gemm.n_groups, c.sched.num_tokens()));
                CUTLASS_PRAGMA_NO_UNROLL
                for (int k = 0; k < c.sched.k_tiles_kvg; ++k)
                    op.issue_tile(k, TraceProbe::WS_KVG_BACKPRESSURE_W, item);
                op.end_item();
            }
        }
        {
            const RangeBarrierLedger led = c.sched.ledger_before_qkv();
            auto gx = [&](int tg) {
                c.bars.x_ready(tg).wait_warp(0, TraceProbe::WS_PBAR_X, tg);
            };
            auto op = make_tma_op_a<CfgQkv>(c.r.res_qkv, led.base_bits(),
                                            led.base_zero_bits(),
                                            c.sched.k_tps_qkv, gx);
            for (int item = (int)blockIdx.x; item < c.sched.gemm.items_qkv;
                 item += gridDim.x) {
                const QkvItem<BN> it = c.r.qkv_item(c.sched, item);
                op.begin_item(*it.weight, it.wi);
                CUTLASS_PRAGMA_NO_UNROLL
                for (int k = 0; k < c.sched.k_tps_qkv; ++k)
                    op.issue_tile(k, TraceProbe::WS_QKV_BACKPRESSURE_W, item);
                op.end_item();
            }
        }
        {
            const RangeBarrierLedger led = c.sched.ledger_before_wq_b();
            auto gq = [&](int tg) {
                c.bars.qq_ready(tg).wait_warp(0, TraceProbe::WS_PBAR_QQ, tg);
            };
            auto op = make_tma_op_a<CfgWqB>(c.r.res_wqb, led.base_bits(),
                                            led.base_zero_bits(),
                                            c.sched.k_tps_wqb, gq);
            for (int item = (int)blockIdx.x; item < c.sched.gemm.items_wq_b;
                 item += gridDim.x) {
                op.begin_item(c.r.res_wqb.weight, c.sched.wqb_item(item));
                CUTLASS_PRAGMA_NO_UNROLL
                for (int k = 0; k < c.sched.k_tps_wqb; ++k)
                    op.issue_tile(k, TraceProbe::WS_WQB_BACKPRESSURE_W, item);
                op.end_item();
            }
        }

        __syncwarp();
        mega_mla::bar_sync<mega_mla::KernelTrait::Bars::EveryoneSync>();
        cutlass::arch::fence_barrier_init();
        mega_mla::bar_sync<mega_mla::KernelTrait::Bars::EveryoneSync>();
    }

    template <class Ctx>
    CUTE_DEVICE void tma_b(Ctx const& c) const {
        {
            const RangeBarrierLedger led = c.sched.ledger_before_kv_gate();
            auto op = make_bf16_act_tma_op<CfgKvGate>(
                c.r.res_kv_gate, led.base_bits(), led.base_zero_bits());
            for (int item = (int)blockIdx.x; item < c.sched.gemm.items_kv_gate;
                 item += gridDim.x) {
                op.begin_item(bf16_gemm_item_at<CfgKvGate>(
                    item, c.sched.gemm.n_groups, c.sched.num_tokens()));
                CUTLASS_PRAGMA_NO_UNROLL
                for (int k = 0; k < c.sched.k_tiles_kvg; ++k)
                    op.issue_tile(k, TraceProbe::WS_KVG_BACKPRESSURE_ACT, item);
                op.end_item();
            }
        }
        {
            const RangeBarrierLedger led = c.sched.ledger_before_qkv();
            auto gx = [&](int tg) {
                c.bars.x_ready(tg).wait_warp(0, TraceProbe::WS_PBAR_X, tg);
            };
            auto op = make_tma_op_b<CfgQkv>(c.r.res_qkv, led.base_bits(),
                                            led.base_zero_bits(),
                                            c.sched.k_tps_qkv, gx);
            for (int item = (int)blockIdx.x; item < c.sched.gemm.items_qkv;
                 item += gridDim.x) {
                const QkvItem<BN> it = c.r.qkv_item(c.sched, item);
                op.begin_item(*it.weight, it.wi);
                CUTLASS_PRAGMA_NO_UNROLL
                for (int k = 0; k < c.sched.k_tps_qkv; ++k)
                    op.issue_tile(k, TraceProbe::WS_QKV_BACKPRESSURE_ACT, item);
                op.end_item();
            }
        }
        {
            const RangeBarrierLedger led = c.sched.ledger_before_wq_b();
            auto gq = [&](int tg) {
                c.bars.qq_ready(tg).wait_warp(0, TraceProbe::WS_PBAR_QQ, tg);
            };
            auto op = make_tma_op_b<CfgWqB>(c.r.res_wqb, led.base_bits(),
                                            led.base_zero_bits(),
                                            c.sched.k_tps_wqb, gq);
            for (int item = (int)blockIdx.x; item < c.sched.gemm.items_wq_b;
                 item += gridDim.x) {
                op.begin_item(c.r.res_wqb.weight, c.sched.wqb_item(item));
                CUTLASS_PRAGMA_NO_UNROLL
                for (int k = 0; k < c.sched.k_tps_wqb; ++k)
                    op.issue_tile(k, TraceProbe::WS_WQB_BACKPRESSURE_ACT, item);
                op.end_item();
            }
        }

        __syncwarp();
        mega_mla::bar_sync<mega_mla::KernelTrait::Bars::EveryoneSync>();
        cutlass::arch::fence_barrier_init();
        mega_mla::bar_sync<mega_mla::KernelTrait::Bars::EveryoneSync>();
    }

    template <class Ctx>
    CUTE_DEVICE void idle_ph1(Ctx const& c) const {
        __syncwarp();
        mega_mla::bar_sync<mega_mla::KernelTrait::Bars::EveryoneSync>();
        cutlass::arch::fence_barrier_init();
        mega_mla::bar_sync<mega_mla::KernelTrait::Bars::EveryoneSync>();
    }

    template <class Ctx>
    CUTE_DEVICE void mma_wo_a(Ctx const& c) const {
        const RangeBarrierLedger led = c.sched.ledger_before_wo_a();
        auto op = make_mma_op<CfgWoA>(c.ro.res_wo_a, led.base_bits(),
                                      led.base_acc(), c.sched.k_tps_woa);
        for (int item = (int)blockIdx.x; item < c.sched.out_proj.items_wo_a;
             item += gridDim.x) {
            op.begin_item(c.sched.wo_a_item(item), TraceProbe::WS_WOA_ACC_EMPTY,
                          item);
            CUTLASS_PRAGMA_NO_UNROLL
            for (int k = 0; k < c.sched.k_tps_woa; ++k) {
                op.wait_tma(k, TraceProbe::WS_WOA_TMA_W,
                            TraceProbe::WS_WOA_TMA_ACT, item);
                op.run_k_tile(k);
                op.commit(k);
            }
            op.end_item();
        }
    }

    template <int KS, class Ctx>
    CUTE_DEVICE void mma_wo_b(Ctx const& c) const {
        const RangeBarrierLedger led = c.sched.ledger_before_wo_b();
        auto op = make_mma_op<WithKSplit<CfgWoB, KS>>(
            c.ro.res_wo_b, led.base_bits(), led.base_acc(), c.sched.k_tps_wob);
        for (int item = (int)blockIdx.x; item < c.sched.out_proj.items_wo_b;
             item += gridDim.x) {
            op.begin_item(c.sched.template wo_b_item<KS>(item),
                          TraceProbe::WS_WOB_ACC_EMPTY, item);
            CUTLASS_PRAGMA_NO_UNROLL
            for (int k = 0; k < c.sched.k_tps_wob; ++k) {
                op.wait_tma(k, TraceProbe::WS_WOB_TMA_W,
                            TraceProbe::WS_WOB_TMA_ACT, item);
                op.run_k_tile(k);
                op.commit(k);
            }
            op.end_item();
        }
    }

    template <class Ctx>
    CUTE_DEVICE void tma_a_wo_a(Ctx const& c) const {
        auto gate_flat_o = [&](int g) {
            c.bars.flat_o(g).wait_warp(0, TraceProbe::WS_PBAR_FLAT_O, g);
        };
        const RangeBarrierLedger led = c.sched.ledger_before_wo_a();
        auto op = make_tma_op_a<CfgWoA>(c.ro.res_wo_a, led.base_bits(),
                                        led.base_zero_bits(), c.sched.k_tps_woa,
                                        gate_flat_o);
        for (int item = (int)blockIdx.x; item < c.sched.out_proj.items_wo_a;
             item += gridDim.x) {
            op.begin_item(c.ro.res_wo_a.weight, c.sched.wo_a_item(item));
            CUTLASS_PRAGMA_NO_UNROLL
            for (int k = 0; k < c.sched.k_tps_woa; ++k)
                op.issue_tile(k, TraceProbe::WS_WOA_BACKPRESSURE_W, item);
            op.end_item();
        }
    }
    template <int KS, class Ctx>
    CUTE_DEVICE void tma_a_wo_b(Ctx const& c) const {
        const RangeBarrierLedger led = c.sched.ledger_before_wo_b();
        auto op = make_tma_op_a<WithKSplit<CfgWoB, KS>>(
            c.ro.res_wo_b, led.base_bits(), led.base_zero_bits(),
            c.sched.k_tps_wob, NoGate{});
        for (int item = (int)blockIdx.x; item < c.sched.out_proj.items_wo_b;
             item += gridDim.x) {
            c.bars.oq_ready(c.sched.wo_b_tg(item))
                .wait_warp(0, TraceProbe::WS_PBAR_OQ, c.sched.wo_b_tg(item));
            op.begin_item(c.ro.res_wo_b.weight,
                          c.sched.template wo_b_item<KS>(item));
            CUTLASS_PRAGMA_NO_UNROLL
            for (int k = 0; k < c.sched.k_tps_wob; ++k)
                op.issue_tile(k, TraceProbe::WS_WOB_BACKPRESSURE_W, item);
            op.end_item();
        }
    }

    template <class Ctx>
    CUTE_DEVICE void tma_b_wo_a(Ctx const& c) const {
        auto gate_flat_o = [&](int g) {
            c.bars.flat_o(g).wait_warp(0, TraceProbe::WS_PBAR_FLAT_O, g);
        };
        const RangeBarrierLedger led = c.sched.ledger_before_wo_a();
        auto op = make_tma_op_b<CfgWoA>(c.ro.res_wo_a, led.base_bits(),
                                        led.base_zero_bits(), c.sched.k_tps_woa,
                                        gate_flat_o);
        for (int item = (int)blockIdx.x; item < c.sched.out_proj.items_wo_a;
             item += gridDim.x) {
            op.begin_item(c.ro.res_wo_a.weight, c.sched.wo_a_item(item));
            CUTLASS_PRAGMA_NO_UNROLL
            for (int k = 0; k < c.sched.k_tps_woa; ++k)
                op.issue_tile(k, TraceProbe::WS_WOA_BACKPRESSURE_ACT, item);
            op.end_item();
        }
    }
    template <int KS, class Ctx>
    CUTE_DEVICE void tma_b_wo_b(Ctx const& c) const {
        const RangeBarrierLedger led = c.sched.ledger_before_wo_b();
        auto op = make_tma_op_b<WithKSplit<CfgWoB, KS>>(
            c.ro.res_wo_b, led.base_bits(), led.base_zero_bits(),
            c.sched.k_tps_wob, NoGate{});
        for (int item = (int)blockIdx.x; item < c.sched.out_proj.items_wo_b;
             item += gridDim.x) {
            c.bars.oq_ready(c.sched.wo_b_tg(item))
                .wait_warp(0, TraceProbe::WS_PBAR_OQ, c.sched.wo_b_tg(item));
            op.begin_item(c.ro.res_wo_b.weight,
                          c.sched.template wo_b_item<KS>(item));
            CUTLASS_PRAGMA_NO_UNROLL
            for (int k = 0; k < c.sched.k_tps_wob; ++k)
                op.issue_tile(k, TraceProbe::WS_WOB_BACKPRESSURE_ACT, item);
            op.end_item();
        }
    }

    template <class Ctx>
    CUTE_DEVICE void attn_mma(Ctx const& c) const {
        if (!c.params.use_phase1) {
            Tensor gQ = c.params.tma_Q.get_tma_tensor(c.params.shape_Q)(
                _, _, c.token_idx);
            Tensor sQ = make_tensor(make_smem_ptr(c.plan.u.qo.q.data()),
                                    AttnConfig::SmemLayoutQ_SW128{});
            ku::launch_tma_copy(c.params.tma_Q, gQ, sQ, c.plan.bar_q_tma,
                                TMA::CacheHintSm90::EVICT_FIRST);
            c.plan.bar_q_tma.arrive_and_expect_tx(
                ModelDims::B_H * ModelDims::D_Q * sizeof(bf16));
            c.plan.bar_q_tma.wait(c.q_tma_phase);
            c.q_tma_phase ^= 1;
            ku::tcgen05_after_thread_sync();
        }

        UMMA::SmemDescriptor sQ_desc =
            UMMA::make_umma_desc<UMMA::Major::K>(make_tensor(
                make_smem_ptr(c.plan.u.qo.q.data()),
                tile_to_shape(UMMA::Layout_K_SW128_Atom<bf16>{},
                              Shape<Int<ModelDims::B_H * 2>, Int<64>>{})));
        CUTE_UNROLL
        for (int tile_idx = 0; tile_idx < ModelDims::D_Q_SW128 / 128;
             ++tile_idx) {
            CUTE_UNROLL
            for (int sub = 0; sub < 64 / 16; ++sub) {
                SM100_UTCCP_128dp256bit_1cta::copy(
                    sQ_desc +
                        (tile_idx * (ModelDims::B_H * 128) + sub * 16) * 2 / 16,
                    AttnConfig::tmem_cols::Q + tile_idx * 32 + sub * 8);
            }
        }
        ku::umma_arrive_noelect(c.plan.bar_q_utccp);

        TiledMMA tiled_mma_P = AttnConfig::TiledMMA_P{};
        TiledMMA tiled_mma_O = AttnConfig::TiledMMA_O{};
        Tensor tP = partition_fragment_C(tiled_mma_P,
                                         Shape<Int<ModelDims::B_H>, _128>{});
        Tensor tO = partition_fragment_C(
            tiled_mma_O, Shape<Int<ModelDims::B_H>, Int<ModelDims::D_V>>{});
        tP.data().get() = AttnConfig::tmem_cols::P;
        tO.data().get() = AttnConfig::tmem_cols::O;

        c.plan.bar_q_utccp.wait(c.q_utccp_phase);
        c.q_utccp_phase ^= 1;
        ku::tcgen05_after_thread_sync();

        auto rq = c.rs;
        auto issue_qk = [&]() {
            c.plan.bar_rope_ready[rq.buf_idx].wait(rq.bar_phase);
            c.plan.bar_nope_ready[rq.buf_idx].wait(rq.bar_phase);
            ku::tcgen05_after_thread_sync();

            Tensor tQ =
                tiled_mma_P.get_slice(_0{}).make_fragment_A(partition_shape_A(
                    tiled_mma_P,
                    Shape<Int<ModelDims::B_H>, Int<ModelDims::D_Q / 2>>{}));
            tQ.data().get() = AttnConfig::tmem_cols::Q;
            Tensor sK = make_tensor(
                make_smem_ptr(c.plan.u.kv.dequant[rq.buf_idx].nope.data()),
                AttnConfig::SmemLayoutKTiles_DualGemm_SW128<512 / 64 / 2>{});
            tP.data().get() = AttnConfig::tmem_cols::P +
                              rq.buf_idx * AttnConfig::tmem_cols::P_STRIDE;
            ku::utcmma_ts(tiled_mma_P, tQ, sK, tP, true);
            ku::umma_arrive_noelect(c.plan.bar_qk_done[rq.buf_idx]);
            rq.update();
        };

        issue_qk();
        CUTE_NO_UNROLL
        for (int block_idx = c.blk_lo; block_idx < c.blk_hi; ++block_idx) {
            if (block_idx + 1 < c.blk_hi) issue_qk();

            c.plan.bar_so_ready[c.rs.buf_idx].wait(c.rs.bar_phase);
            ku::tcgen05_after_thread_sync();
            Tensor sS =
                make_tensor(make_smem_ptr(c.plan.u.kv.s[c.rs.buf_idx].data()),
                            AttnConfig::SmemLayoutS{});
            Tensor sV = make_tensor(
                make_smem_ptr(c.plan.u.kv.dequant[c.rs.buf_idx].nope.data()),
                AttnConfig::SmemLayoutKTilesTransposed_SW128<ModelDims::D_V /
                                                             64>{});
            ku::utcmma_ss(tiled_mma_O, sS, sV, tO, block_idx == c.blk_lo);
            ku::umma_arrive_noelect(c.plan.bar_sv_done[c.rs.buf_idx]);

            c.rs.update();
        }
    }

    template <class Ctx>
    CUTE_DEVICE void attn_tma1(Ctx const& c) const {
        c.plan.bar_q_utccp.wait(c.q_utccp_phase);
        c.q_utccp_phase ^= 1;
        c.plan.bar_last_store_done.wait(0);
        CUTE_NO_UNROLL
        for (int block_idx = c.blk_lo; block_idx < c.blk_hi; ++block_idx) {
            c.plan.bar_raw_free[c.rs.buf_idx].wait(c.rs.bar_phase ^ 1);
            mega_mla::issue_kv_span<AttnConfig::B_TOPK / 2>(
                c.kv_nope_desc(block_idx), c.kv_nope_gather(block_idx),
                c.plan.bar_raw_ready[c.rs.buf_idx],
                (uint8_t*)c.plan.u.kv.raw_nope[c.rs.buf_idx].data(),
                ModelDims::D_NOPE * sizeof(e4m3), c.kv_tile_at(block_idx), 0,
                0);

            c.plan.bar_raw_ready[c.rs.buf_idx].arrive_and_expect_tx(
                AttnConfig::B_TOPK / 2 * ModelDims::D_NOPE * sizeof(e4m3));
            c.rs.update();
        }
    }

    template <class Ctx>
    CUTE_DEVICE void attn_tma3(Ctx const& c) const {
        c.plan.bar_q_utccp.wait(c.q_utccp_phase);
        c.q_utccp_phase ^= 1;
        c.plan.bar_last_store_done.wait(0);
        CUTE_NO_UNROLL
        for (int block_idx = c.blk_lo; block_idx < c.blk_hi; ++block_idx) {
            c.plan.bar_sv_done[c.rs.buf_idx].wait(c.rs.bar_phase ^ 1);
            mega_mla::issue_kv_span<AttnConfig::B_TOPK>(
                c.kv_rope_desc(block_idx), c.kv_rope_gather(block_idx),
                c.plan.bar_rope_ready[c.rs.buf_idx],
                (uint8_t*)c.plan.u.kv.dequant[c.rs.buf_idx].rope.data(),
                ModelDims::D_ROPE * sizeof(bf16), c.kv_tile_at(block_idx), 0,
                0);
            c.plan.bar_rope_ready[c.rs.buf_idx].arrive_and_expect_tx(
                ModelDims::B_H * ModelDims::D_ROPE * sizeof(bf16));
            c.rs.update();
        }
    }

    template <class Ctx>
    CUTE_DEVICE void attn_tma2(Ctx const& c) const {
        c.plan.bar_q_utccp.wait(c.q_utccp_phase);
        c.q_utccp_phase ^= 1;
        c.plan.bar_last_store_done.wait(0);
        CUTE_NO_UNROLL
        for (int block_idx = c.blk_lo; block_idx < c.blk_hi; ++block_idx) {
            c.plan.bar_raw_free[c.rs.buf_idx].wait(c.rs.bar_phase ^ 1);
            mega_mla::issue_kv_span<AttnConfig::B_TOPK / 2>(
                c.kv_nope_desc(block_idx), c.kv_nope_gather(block_idx),
                c.plan.bar_raw_ready2[c.rs.buf_idx],
                (uint8_t*)c.plan.u.kv.raw_nope[c.rs.buf_idx].data(),
                ModelDims::D_NOPE * sizeof(e4m3), c.kv_tile_at(block_idx), 0,
                AttnConfig::B_TOPK / 2);
            c.plan.bar_raw_ready2[c.rs.buf_idx].arrive_and_expect_tx(
                AttnConfig::B_TOPK / 2 * ModelDims::D_NOPE * sizeof(e4m3));
            c.rs.update();
        }
    }

    template <class Ctx, class Tok>
    CUTE_DEVICE void kv_cache_gate(Ctx const& c, Tok const& t) const {
        if (c.params.use_kv_write ||
            (c.params.use_kv_gate && c.params.use_c128)) {
            if (c.warp_idx == 0) {
                const int d = c.params.draft_tokens;
                const int g_lo =
                    d > 0 ? max(0, t.token_idx - (d - 1)) / c.tile_t : 0;
                const int g_hi = d > 0 ? min(c.params.num_tokens - 1,
                                             t.token_idx + (d - 1)) /
                                             c.tile_t
                                       : c.sched.n_groups() - 1;
                for (int g = g_lo; g <= g_hi; ++g) {
                    if (c.params.use_kv_write)
                        c.bars.kvcache_ready(g).wait_warp(
                            0, TraceProbe::WS_PBAR_KVCACHE, g);
                    if (c.params.use_kv_gate && c.params.use_c128)
                        c.bars.c128_ready(g).wait_warp(
                            0, mega_mla::TraceProbe::WS_PBAR_C128, g);
                }
            }
            mega_mla::bar_sync<mega_mla::KernelTrait::Bars::Wg0Sync>();
        }
    }

    template <class Ctx>
    CUTE_DEVICE void token_loop_mma(Ctx const& c) const {
        for (int token_it = 0; token_it < c.token_iters; ++token_it) {
            const PerToken<Ctx> t = make_per_token(c, token_it);

            if (t.token_active) {
                phase1_prepare_q(c, t);

                c.bars.flatq_free(t.token_idx / c.tile_t).arrive_warp();

                kv_cache_gate(c, t);
                if (elect_one_sync()) attn_mma(t);

                mega_mla::bar_sync<mega_mla::KernelTrait::Bars::Wg0Sync>();
            }

            __syncwarp();
            mega_mla::bar_sync<mega_mla::KernelTrait::Bars::EveryoneSync>();

            if (c.sched.o_post_flat && t.token_active)
                c.bars.o_stored(t.token_idx / c.tile_t).arrive_warp();
            if (token_it + 1 == c.token_iters) {
                __syncwarp();
                mega_mla::bar_sync<mega_mla::KernelTrait::Bars::EveryoneSync>();
            }
        }
    }

    template <class Ctx>
    CUTE_DEVICE void out_proj_mma(Ctx const& c) const {
        mma_wo_a(c);
        __threadfence();
        auto run_wo_b = [&](auto KS_C) {
            constexpr int KS = decltype(KS_C)::value;
            mma_wo_b<KS>(c);
            __threadfence();
        };
        if (c.sched.ks_wo_b == 4)
            run_wo_b(cute::Int<4>{});
        else if (c.sched.ks_wo_b == 2)
            run_wo_b(cute::Int<2>{});
        else
            run_wo_b(cute::Int<1>{});
    }

    template <class Ctx>
    CUTE_DEVICE void role_mma(Ctx const& c) const {
        mma(c);
        token_loop_mma(c);
        out_proj_mma(c);
    }

    template <class Ctx>
    CUTE_DEVICE void token_loop_tma1(Ctx const& c) const {
        for (int token_it = 0; token_it < c.token_iters; ++token_it) {
            const PerToken<Ctx> t = make_per_token(c, token_it);

            if (t.token_active) {
                phase1_prepare_q(c, t);

                kv_cache_gate(c, t);
                if (elect_one_sync()) attn_tma1(t);

                mega_mla::bar_sync<mega_mla::KernelTrait::Bars::Wg0Sync>();
            }

            __syncwarp();
            mega_mla::bar_sync<mega_mla::KernelTrait::Bars::EveryoneSync>();
            if (token_it + 1 == c.token_iters) {
                __syncwarp();
                mega_mla::bar_sync<mega_mla::KernelTrait::Bars::EveryoneSync>();
            }
        }
    }

    template <class Ctx>
    CUTE_DEVICE void out_proj_tma1(Ctx const& c) const {
        tma_a_wo_a(c);
        __threadfence();
        auto run_wo_b = [&](auto KS_C) {
            constexpr int KS = decltype(KS_C)::value;
            tma_a_wo_b<KS>(c);
            __threadfence();
        };
        if (c.sched.ks_wo_b == 4)
            run_wo_b(cute::Int<4>{});
        else if (c.sched.ks_wo_b == 2)
            run_wo_b(cute::Int<2>{});
        else
            run_wo_b(cute::Int<1>{});
    }

    template <class Ctx>
    CUTE_DEVICE void role_tma1(Ctx const& c) const {
        tma_a(c);
        token_loop_tma1(c);
        out_proj_tma1(c);
    }

    template <class Ctx>
    CUTE_DEVICE void token_loop_tma2(Ctx const& c) const {
        for (int token_it = 0; token_it < c.token_iters; ++token_it) {
            const PerToken<Ctx> t = make_per_token(c, token_it);

            if (t.token_active) {
                phase1_prepare_q(c, t);

                kv_cache_gate(c, t);
                if (elect_one_sync()) attn_tma2(t);

                mega_mla::bar_sync<mega_mla::KernelTrait::Bars::Wg0Sync>();
            }

            __syncwarp();
            mega_mla::bar_sync<mega_mla::KernelTrait::Bars::EveryoneSync>();
            if (token_it + 1 == c.token_iters) {
                __syncwarp();
                mega_mla::bar_sync<mega_mla::KernelTrait::Bars::EveryoneSync>();
            }
        }
    }

    template <class Ctx>
    CUTE_DEVICE void out_proj_tma2(Ctx const& c) const {
        tma_b_wo_a(c);
        __threadfence();
        auto run_wo_b = [&](auto KS_C) {
            constexpr int KS = decltype(KS_C)::value;
            tma_b_wo_b<KS>(c);
            __threadfence();
        };
        if (c.sched.ks_wo_b == 4)
            run_wo_b(cute::Int<4>{});
        else if (c.sched.ks_wo_b == 2)
            run_wo_b(cute::Int<2>{});
        else
            run_wo_b(cute::Int<1>{});
    }

    template <class Ctx>
    CUTE_DEVICE void role_tma2(Ctx const& c) const {
        tma_b(c);
        token_loop_tma2(c);
        out_proj_tma2(c);
    }

    template <class Ctx>
    CUTE_DEVICE void token_loop_tma3(Ctx const& c) const {
        for (int token_it = 0; token_it < c.token_iters; ++token_it) {
            const PerToken<Ctx> t = make_per_token(c, token_it);

            if (t.token_active) {
                phase1_prepare_q(c, t);

                kv_cache_gate(c, t);
                if (elect_one_sync()) attn_tma3(t);

                mega_mla::bar_sync<mega_mla::KernelTrait::Bars::Wg0Sync>();
            }

            __syncwarp();
            mega_mla::bar_sync<mega_mla::KernelTrait::Bars::EveryoneSync>();
            if (token_it + 1 == c.token_iters) {
                __syncwarp();
                mega_mla::bar_sync<mega_mla::KernelTrait::Bars::EveryoneSync>();
            }
        }
    }

    CUTE_DEVICE void out_proj_tma3() const { __threadfence(); }

    template <class Ctx>
    CUTE_DEVICE void role_tma3(Ctx const& c) const {
        idle_ph1(c);
        token_loop_tma3(c);
        out_proj_tma3();
    }
};

}  // namespace mega_mla
