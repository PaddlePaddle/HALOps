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
#include "mla_plan.cuh"
#include "mla_traits.cuh"
#include "mla_utils.cuh"
#include "umma_gemm.cuh"

namespace halops {

template <class KT>
struct SharedMemoryPlan {
    using Attn = typename KT::Attn;
    using Dims = typename KT::Dims;
    using Arena = mega_mla::GemmArena;

    union {
        struct {
            cute::array_aligned<
                mega_mla::bf16,
                cute::cosize_v<typename Attn::SmemLayoutQ_SW128>>
                q;
            union {
                cute::array_aligned<
                    mega_mla::bf16,
                    cute::cosize_v<typename Attn::SmemLayoutOBuf>>
                    o_buf;
            } o;
        } qo;
        struct {
            struct {
                cute::array_aligned<mega_mla::bf16, Dims::B_H * Dims::D_NOPE>
                    nope;
                cute::array_aligned<mega_mla::bf16, Dims::B_H * Dims::D_ROPE>
                    rope;
            } dequant[Attn::NUM_BUFS];
            cute::array_aligned<mega_mla::e4m3, Dims::B_H * Dims::D_NOPE>
                raw_nope[Attn::NUM_BUFS];

            float4 p_exchange_buf[4][16 * Attn::B_TOPK / 4];
            cute::array_aligned<mega_mla::bf16,
                                cute::cosize_v<typename Attn::SmemLayoutS>>
                s[Attn::NUM_BUFS];
        } kv;

        char gemm[Arena::Total];
    } u;

    CUTE_ALIGNAS(16) float rowwise_max_buf[KT::ThreadsPerWarpGroup];

    CUTE_ALIGNAS(16) float c128_stats[8];
    CUTE_ALIGNAS(16) __nv_bfloat16 c128_row[Dims::D_V];

    CUTE_ALIGNAS(16) float kv_write_sq[12];
    CUTE_ALIGNAS(16) __nv_bfloat16 kv_write_row[Dims::D_V];

    // Indexed by token-within-group, so the bound is the WIDEST tile the launch
    // switch instantiates -- compile-time because it is smem.
    CUTE_ALIGNAS(16) int2 o_split[KT::Tile::Widest];

    cute::array_aligned<uint32_t, 1> tmem_start_addr;
    transac_bar_t bar_last_store_done;
    transac_bar_t bar_q_tma, bar_q_utccp;
    transac_bar_t bar_rope_ready[Attn::NUM_BUFS];
    transac_bar_t bar_nope_ready[Attn::NUM_BUFS];
    transac_bar_t bar_raw_ready[Attn::NUM_BUFS], bar_raw_free[Attn::NUM_BUFS];
    transac_bar_t bar_raw_ready2[Attn::NUM_BUFS];
    transac_bar_t bar_qk_done[Attn::NUM_BUFS], bar_so_ready[Attn::NUM_BUFS],
        bar_sv_done[Attn::NUM_BUFS];

    transac_bar_t bar_gemm_tma_a[Arena::MaxTmaStages];
    transac_bar_t bar_gemm_tma_b[Arena::MaxTmaStages];
    transac_bar_t bar_gemm_smem_free[Arena::MaxTmaStages];
    transac_bar_t bar_gemm_acc_full[Arena::MaxAccStages];
    transac_bar_t bar_gemm_acc_empty[Arena::MaxAccStages];

    static_assert(Arena::Total <= (int)sizeof(u),
                  "the GEMM smem arena does not fit the `u` union");
    static_assert(Arena::MaxTmaStages >= KT::MaxTmaStages,
                  "bar_gemm_tma_a must cover this tile's deepest GEMM");

    CUTE_DEVICE mega_mla::GemmPipelineBars gemm_bars() {
        return {bar_gemm_tma_a, bar_gemm_tma_b, bar_gemm_smem_free,
                bar_gemm_acc_full, bar_gemm_acc_empty};
    }

    CUTE_DEVICE char* gemm_smem() {
        return reinterpret_cast<char*>(u.kv.dequant[0].nope.data());
    }

    CUTE_DEVICE void init_gemm_bars() {
        CUTE_UNROLL
        for (int i = 0; i < Arena::MaxTmaStages; ++i) {
            bar_gemm_tma_a[i].init(1);
            bar_gemm_tma_b[i].init(1);
            bar_gemm_smem_free[i].init(1);
        }
        CUTE_UNROLL
        for (int i = 0; i < Arena::MaxAccStages; ++i) {
            bar_gemm_acc_full[i].init(1);
            bar_gemm_acc_empty[i].init(KT::Roles::DrainWarps * KT::WarpSize);
        }
    }

    // Every mbarrier the CTA owns, armed by warp 0 before the role split.
    CUTE_DEVICE void init_bars() {
        bar_last_store_done.init(KT::ThreadsPerWarpGroup);
        bar_q_tma.init(1);
        bar_q_utccp.init(1);
        CUTE_UNROLL
        for (int i = 0; i < Attn::NUM_BUFS; ++i) {
            bar_rope_ready[i].init(1);
            bar_nope_ready[i].init(KT::ThreadsPerWarpGroup);
            bar_raw_ready[i].init(1);
            bar_raw_ready2[i].init(1);
            bar_raw_free[i].init(KT::ThreadsPerWarpGroup);
            bar_qk_done[i].init(1);
            bar_so_ready[i].init(KT::ThreadsPerWarpGroup);
            bar_sv_done[i].init(1);
        }
        init_gemm_bars();
        cutlass::arch::fence_barrier_init();
    }

    CUTE_DEVICE void acquire_tmem() {
        cute::TMEM::Allocator1Sm().allocate(Attn::tmem_cols::TOTAL,
                                            tmem_start_addr.data());
        KU_TRAP_ONLY_DEVICE_ASSERT(tmem_start_addr.data()[0] == 0);
        cute::TMEM::Allocator1Sm().release_allocation_lock();
    }

    CUTE_DEVICE void release_tmem() {
        cute::TMEM::Allocator1Sm().free(0, Attn::tmem_cols::TOTAL);
    }
};

// The launch inputs exactly as the torch binding hands them over.  Both
// collectives and the scheduler derive their Params from this one object, so a
// new input is added in ONE place instead of in three signatures.
struct MegaMlaArguments {
    torch::Tensor q_or_x;
    torch::Tensor kv_cache;
    int64_t page_size;
    torch::Tensor kv_runs;
    int64_t topk;
    int64_t kv_run_step;
    torch::Tensor topk_lengths;
    torch::Tensor positions;
    torch::Tensor freqs_real;
    torch::Tensor wo_a_weight;
    torch::Tensor wo_a_scale_inv;
    torch::Tensor wo_b_weight;
    torch::Tensor out;
    double sm_scale;
    torch::Tensor wq_a_weight;
    torch::Tensor q_norm_weight;
    torch::Tensor wq_b_weight;
    double q_eps;
    bool use_phase1;
    torch::Tensor wkv_weight;
    torch::Tensor kv_norm_weight;
    double kv_eps;
    bool use_kv_write;
    torch::Tensor swa_loc;
    torch::Tensor attn_sink;
    torch::Tensor wq_a_sf_g32;
    torch::Tensor wq_b_sf_g32;
    torch::Tensor wkv_sf_g32;
    torch::Tensor wo_b_sf_g32;
    torch::Tensor wo_a_sf_g32;
    torch::Tensor workspace;
    torch::Tensor wkv_gate_weight;
    torch::Tensor kv_score_ring;
    torch::Tensor kv_gate_ring_row;
    torch::Tensor c128_ape;
    torch::Tensor c128_norm_weight;
    torch::Tensor c128_freqs_real;
    torch::Tensor c128_out_loc;
    torch::Tensor c128_row_out;
    torch::Tensor c128_pool;
    int64_t c128_stride_kv_block;
    int64_t c128_page_block;
    torch::Tensor c128_attn_slot0;
    torch::Tensor c128_attn_len;
    int64_t draft_tokens;
};

template <class KT>
struct MegaMlaProblem;

// What BOTH halves read.  `q` is the O staging area (the drain stores into it,
// the epilogue reads it back) and `ws_flat_q_bf16` is flat_q first and the wo_b
// partials afterwards, so neither belongs to one side.
template <class KT>
struct CommonParams {
    int num_tokens;
    int draft_tokens;

    mega_mla::bf16* __restrict__ q;
    mega_mla::bf16* __restrict__ ws_flat_q_bf16;

    mega_mla::PlanView wplan;
    int* __restrict__ grid_bar;
    int* __restrict__ partial_bar;
    int pbar_bank;

    static CommonParams from(const MegaMlaArguments& args,
                             const MegaMlaProblem<KT>& prob) {
        CommonParams p{};
        p.num_tokens = prob.num_tokens;
        p.draft_tokens = (int)args.draft_tokens;
        p.q = prob.q_ptr;
        p.grid_bar = reinterpret_cast<int*>(prob.base);
        p.partial_bar = reinterpret_cast<int*>(prob.base + prob.wsg.pbar_off);
        p.pbar_bank = prob.wsg.pbar_bank;
        p.wplan =
            mega_mla::plan_ptrs(prob.base, prob.wsg.plan_off, prob.num_tokens,
                                prob.grid_x, prob.n_groups);
        p.ws_flat_q_bf16 = prob.ws_flat_q_bf16;
        return p;
    }
};

template <class KT, class TmaQ, class TmaKvGateW, class TmaKvGateAct>
struct MainloopParams : CommonParams<KT> {
    using Attn = typename KT::Attn;

    float sm_scale_div_log2;
    int num_blocks, page_block_size, topk;
    mega_mla::bf16* __restrict__ kv;
    int* __restrict__ topk_length;
    float* __restrict__ attn_sink;
    int stride_kv_block;

    const int4* __restrict__ kv_runs;
    int kv_run_step;
    int page_shift;
    int rows_per_page;

    mega_mla::GemmShape shape_wq_a;
    mega_mla::GemmShape shape_wkv;
    mega_mla::GemmShape shape_wq_b;
    mega_mla::GemmShape shape_kv_gate;

    mega_mla::WeightAtoms<KT::TileT> gemm_wq_a;
    mega_mla::WeightAtoms<KT::TileT> gemm_wq_b;
    mega_mla::WeightAtoms<KT::TileT> gemm_wkv;
    mega_mla::ActAtoms<KT::TileT> act_x;
    mega_mla::ActAtoms<KT::TileT> act_qlora;

    const float* __restrict__ freqs_real;
    const int* __restrict__ positions;

    int use_phase1;
    const mega_mla::bf16* __restrict__ x_input;
    const mega_mla::bf16* __restrict__ q_norm_weight;
    float q_eps;
    const mega_mla::bf16* __restrict__ kv_norm_weight;
    float kv_eps;

    int use_kv_write;
    const int* __restrict__ swa_loc;

    const mega_mla::bf16* __restrict__ wkv_gate_weight;
    int use_kv_gate;
    float* __restrict__ kv_score_ring;
    const int* __restrict__ kv_gate_ring_row;

    int use_c128;
    const float* __restrict__ c128_ape;
    const mega_mla::bf16* __restrict__ c128_norm_weight;
    const float* __restrict__ c128_freqs_real;
    const int* __restrict__ c128_out_loc;
    float* __restrict__ c128_row_out;
    uint8_t* __restrict__ c128_pool;
    int c128_stride_kv_block;
    int c128_page_block;
    int c128_page_shift;
    int c128_rows_per_page;

    const int* __restrict__ c128_attn_slot0;
    const int* __restrict__ c128_attn_len;
    int use_c128_attn;

    mega_mla::e4m3* __restrict__ ws_x_fp8;
    mega_mla::ue8m0* __restrict__ ws_x_sf;
    float* __restrict__ ws_qlora_part;
    mega_mla::e4m3* __restrict__ ws_qlora_fp8;
    mega_mla::ue8m0* __restrict__ ws_qlora_sf;
    float* __restrict__ ws_kv_part;

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
    mega_mla::WeightAtoms<KT::TileT> gemm_wo_a;
    mega_mla::WeightAtoms<KT::TileT> gemm_wo_b;
    mega_mla::ActAtoms<KT::TileT> act_flat_o;
    mega_mla::ActAtoms<KT::TileT> act_olora;

    mega_mla::GemmShape shape_wo_a;
    mega_mla::GemmShape shape_wo_b;
    mega_mla::OutProjDims opdims;

    mega_mla::bf16* __restrict__ final_out;
    mega_mla::e4m3* __restrict__ ws_flat_o_fp8;
    mega_mla::ue8m0* __restrict__ ws_flat_o_sf;
    mega_mla::bf16* __restrict__ ws_olora_bf16;
    mega_mla::e4m3* __restrict__ ws_olora_fp8;
    mega_mla::ue8m0* __restrict__ ws_olora_sf;

    // Largest T that still takes the FLAT O post-processing path; host-set from
    // MEGA_MLA_OPOST_MAX so one binary can sweep it.
    int opost_flat_max;
    mega_mla::bf16* __restrict__ ws_o_partial;
    float* __restrict__ ws_lse_partial;

    cute::tuple<int, int, uint64_t> shape_O;
    TmaO tma_O;
    // Same descriptor shape as tma_O but over ws_o_partial, indexed by CTA
    // instead of by token.  Only used when the plan split something.
    cute::tuple<int, int, uint64_t> shape_O_partial;
    TmaO tma_O_partial;
};

// THE ONE PLACE `olora_rounds` IS DERIVED FROM A WEIGHT, because three entry
// points need it before OutProjDims exists.  wo_a may be 3-D [G, R, d], so the
// group width is the LAST dim, not size(1).
inline int olora_rounds_from_wo_a(const torch::Tensor& wo_a_weight) {
    using Dims = mega_mla::ModelDims;
    const int d_per_group = (int)wo_a_weight.size(wo_a_weight.dim() - 1);
    TORCH_CHECK(
        d_per_group > 0 && (int64_t)Dims::B_H * Dims::D_V % d_per_group == 0,
        "wo_a's group width ", d_per_group, " must tile flat_o (",
        (int64_t)Dims::B_H * Dims::D_V, ") exactly");
    const int o_groups = (int)((int64_t)Dims::B_H * Dims::D_V / d_per_group);
    const int r = mega_mla::olora_rounds_for(o_groups);
    TORCH_CHECK(r > 0, "wo_a's ", o_groups,
                " groups give zero o_lora rendezvous rounds");
    return r;
}

template <class KT>
struct MegaMlaWorkspace {
    using Dims = typename KT::Dims;
    using Ws = typename KT::Workspace;

    static constexpr int FLAT_O_K = Dims::O_TOTAL;
    static constexpr int FLAT_Q_K = Dims::Q_TOTAL;

    enum : int {
        PB_X_READY,
        PB_QLORA_READY,
        PB_QQ_READY,
        PB_RING_READY,
        PB_KV_READY,
        PB_OLORA_READY,
        PB_OQ_READY,
        PB_WOB_READY,
        PB_FLAT_O_READY,
        PB_FLATQ_FREE,
        PB_O_STORED,
        PB_FLAT_Q_READY,
        PB_KVCACHE_READY,
        PB_C128_READY,
        PB_O_COEF
    };
    static constexpr int PARTIAL_BARRIER_BANKS = PB_O_COEF + 1;

    // Sized from the LAUNCH, never from a compile-time cap -- a MAX_TOKENS here
    // is a model that cannot run.  Two rendezvous may not share a bank inside
    // one launch (the second finds the count already satisfied), and because
    // this region's offset AND size move with T it cannot be zeroed once at
    // allocation: a T whose banks land on a longer T's bytes starts satisfied
    // and the reader gets garbage.  MegaMlaPlan zeroes it per shape.
    struct Geom {
        int pbar_bank;
        int64_t plan_off;
        int64_t pbar_off;
        int64_t pbar_bytes;
        int64_t kv_split_off;
        int64_t token_off;
        mega_mla::PlanLayout plan;
    };

    static Geom geom(int T, int grid_x, int n_groups, int olora_rounds) {
        Geom g{};
        g.pbar_bank = std::max(1, olora_rounds) * std::max(1, n_groups);
        g.plan = mega_mla::plan_layout(T, grid_x, n_groups);
        g.plan_off = Ws::GridBarrierBytes;
        g.pbar_off = g.plan_off + g.plan.bytes;
        g.pbar_bytes = (int64_t)PARTIAL_BARRIER_BANKS * g.pbar_bank *
                       Ws::RendezvousSlotBytes;
        g.kv_split_off = mega_mla::align128(g.pbar_off + g.pbar_bytes);
        g.token_off = mega_mla::align128(
            g.kv_split_off +
            (int64_t)grid_x * (Dims::B_H * Dims::D_V * 2 + Dims::B_H * 2 * 4));
        return g;
    }

    static constexpr int64_t PLAN_OFFSET = Ws::GridBarrierBytes;

    struct Dimensions {
        int x_k;
        int qlora_k;
        int olora_k;
        int kv_k;
        int kv_gate_k;
        int bytes_per_token;

        int64_t bytes(const Geom& g, int t_pad) const {
            return g.token_off + (int64_t)t_pad * bytes_per_token;
        }
    };

    static Dimensions dims(int hidden, int qlora, int kvlora, int olora_total) {
        constexpr int SF = KT::Qkv::SfGran;
        constexpr int SFP = Ws::SfPadRows;
        constexpr int KS_QKV = KT::Qkv::KSplit;
        Dimensions d{};
        d.x_k = hidden;
        d.qlora_k = qlora;
        d.olora_k = olora_total;
        d.kv_k = kvlora;
        d.kv_gate_k = 2 * kvlora;
        TORCH_CHECK(hidden * 2 * KT::WoB::KSplitMax <= FLAT_Q_K * 2,
                    "wo_b partials must fit inside the dead flat_q buffer: "
                    "need ",
                    hidden * 2 * KT::WoB::KSplitMax, " have ", FLAT_Q_K * 2);
        d.bytes_per_token =
            d.x_k + SFP * (d.x_k / SF) + d.qlora_k + SFP * (d.qlora_k / SF) +
            FLAT_O_K + SFP * (FLAT_O_K / SF) + d.olora_k +
            SFP * (d.olora_k / SF) + d.qlora_k * 4 * KS_QKV + FLAT_Q_K * 2 +
            d.olora_k * 2 * KT::WoA::KSplitMax + d.kv_k * 4 * KS_QKV;
        return d;
    }

    static int token_pad(int T) {
        constexpr int A = Ws::TokenPadRows;
        return (T + A - 1) / A * A;
    }
};

// Everything derived from the launch inputs before either half is built: the
// weight-shape checks, the sizes the workspace layout needs, and the one linear
// carve of the token region -- one cursor, so it cannot be done twice.
template <class KT>
struct MegaMlaProblem {
    using D = mega_mla::ModelDims;
    using WS = MegaMlaWorkspace<KT>;
    using CfgA = typename KT::Qkv;
    using CfgOA = typename KT::WoA;
    static constexpr int TileT = KT::TileT;

    int num_tokens, t_pad, t_pad_sf, grid_x, n_groups;
    int hidden, qlora_rank, kvlora_rank;
    int d_per_group, o_groups, o_lora_rank, olora_total;
    int ks_wo_b;

    typename WS::Dimensions wsd;
    typename WS::Geom wsg;
    mega_mla::OutProjDims opdims;

    char* base;
    mega_mla::bf16* q_ptr;
    mega_mla::bf16* o_partial;

    mega_mla::e4m3* ws_x_fp8;
    mega_mla::ue8m0* ws_x_sf;
    float* ws_qlora_part;
    mega_mla::e4m3* ws_qlora_fp8;
    mega_mla::ue8m0* ws_qlora_sf;
    float* ws_kv_part;
    mega_mla::bf16* ws_flat_q_bf16;
    mega_mla::e4m3* ws_flat_o_fp8;
    mega_mla::ue8m0* ws_flat_o_sf;
    mega_mla::bf16* ws_olora_bf16;
    mega_mla::e4m3* ws_olora_fp8;
    mega_mla::ue8m0* ws_olora_sf;

    static int w_rows(const torch::Tensor& w) {
        return (int)(w.numel() / w.size(w.dim() - 1));
    }
    static int w_cols(const torch::Tensor& w) {
        return (int)w.size(w.dim() - 1);
    }

    static void check_fp8_k_major(const char* name, const torch::Tensor& w) {
        TORCH_CHECK(w.scalar_type() == torch::kFloat8_e4m3fn, name,
                    " must be fp8_e4m3fn, got ", w.scalar_type());
        TORCH_CHECK(w.stride(w.dim() - 1) == 1, name,
                    " must be K-major (innermost stride 1), got ", w.strides());
    }

    // The one place a weight set is matched against a compiled-in model.  Only
    // QLORA_RANK is load-bearing at compile time; widening this rejects models
    // that would have run.
    static void check_model_dims(int qlora) {
        TORCH_CHECK(qlora == D::QLORA_RANK, "unsupported q-lora rank ", qlora,
                    ": this build dispatches ", (int)D::QLORA_RANK,
                    " only, because q_reduce_norm_quant sizes a register array "
                    "from it (PASSES = rank/32/16, and VecWarps*PASSES == 8 is "
                    "what makes the reduction bit-identical)");
    }

    static MegaMlaProblem from(const MegaMlaArguments& args) {
        MegaMlaProblem p{};
        TORCH_CHECK(args.q_or_x.is_cuda(), "q_or_x must be CUDA");
        TORCH_CHECK(
            reinterpret_cast<uintptr_t>(args.q_or_x.data_ptr()) % 16 == 0,
            "q_or_x must be 16 B aligned (the O read-back loads it as uint4); "
            "pass the whole tensor, not a byte-offset slice");
        TORCH_CHECK(args.kv_runs.dim() == 2 && args.kv_runs.size(1) == 4,
                    "kv_runs must be [T, 4] (slot0, len0, slot1, _)");
        TORCH_CHECK(args.kv_runs.scalar_type() == torch::kInt,
                    "kv_runs must be int32");
        TORCH_CHECK(args.kv_runs.stride(0) == 4 && args.kv_runs.stride(1) == 1,
                    "kv_runs must be contiguous (read as one int4 per token)");
        TORCH_CHECK(args.topk_lengths.scalar_type() == torch::kInt,
                    "topk_lengths must be int32");
        p.num_tokens = (int)args.kv_runs.size(0);
        const int T = p.num_tokens;
        TORCH_CHECK(args.topk_lengths.size(0) == T,
                    "topk_lengths must be [T]=", T, ", got ",
                    args.topk_lengths.size(0));
        TORCH_CHECK(args.draft_tokens >= 0 && args.draft_tokens <= T,
                    "draft_tokens must be in [0, T]; 0 means unknown, got ",
                    args.draft_tokens, " with T=", T);
        TORCH_CHECK(
            args.page_size > 0 && (args.page_size & (args.page_size - 1)) == 0,
            "page_size must be a power of two, got ", args.page_size);
        TORCH_CHECK(args.kv_run_step == 1 || args.kv_run_step == -1,
                    "kv_run_step must be +1 or -1, got ", args.kv_run_step);
        TORCH_CHECK(
            mega_mla::select_tile_t(T) == TileT,
            "mega_mla: the tile ladder and the instantiated tile disagree (",
            mega_mla::select_tile_t(T), " vs ", TileT,
            "); MegaMlaPlan grouped the tokens by the former");

        check_fp8_k_major("wo_a_weight", args.wo_a_weight);
        check_fp8_k_major("wo_b_weight", args.wo_b_weight);
        const bool wo_a_sf_packed =
            args.wo_a_scale_inv.scalar_type() != torch::kByte;
        if (wo_a_sf_packed) {
            TORCH_CHECK(args.wo_a_scale_inv.scalar_type() == torch::kInt,
                        "wo_a_scale_inv must be uint8 (group-32 ue8m0) or "
                        "int32 (128-granularity packed ue8m0), got ",
                        args.wo_a_scale_inv.scalar_type());
            TORCH_CHECK(args.wo_a_scale_inv.dim() == 3,
                        "a packed int32 wo_a_scale_inv must be 3D [G, R, k]");
            TORCH_CHECK(args.wo_a_scale_inv.stride(1) == 1,
                        "a packed int32 wo_a_scale_inv must be MN-major, got ",
                        args.wo_a_scale_inv.stride(1));
        }

        p.d_per_group = w_cols(args.wo_a_weight);
        TORCH_CHECK(
            p.d_per_group > 0 && (int64_t)D::O_TOTAL % p.d_per_group == 0,
            "wo_a's groups must tile flat_o exactly: ", (int)D::O_TOTAL,
            " is not a multiple of ", p.d_per_group);
        p.o_groups = (int)((int64_t)D::O_TOTAL / p.d_per_group);
        TORCH_CHECK(w_rows(args.wo_a_weight) % p.o_groups == 0,
                    "wo_a's row count ", w_rows(args.wo_a_weight),
                    " must divide by its group count ", p.o_groups);
        p.o_lora_rank = w_rows(args.wo_a_weight) / p.o_groups;
        p.olora_total = p.o_groups * p.o_lora_rank;
        p.hidden = w_rows(args.wo_b_weight);
        TORCH_CHECK(w_cols(args.wo_b_weight) == p.olora_total,
                    "wo_b reduces over o_lora, so its K must be G*R = ",
                    p.olora_total, ", got ", w_cols(args.wo_b_weight));
        if (wo_a_sf_packed)
            TORCH_CHECK((int)args.wo_a_scale_inv.size(0) == p.o_groups,
                        "a packed wo_a_scale_inv's leading dim must be G = ",
                        p.o_groups, ", got ", args.wo_a_scale_inv.size(0));

        p.qlora_rank = w_rows(args.wq_a_weight);
        if (args.use_phase1) {
            check_fp8_k_major("wq_a_weight", args.wq_a_weight);
            check_fp8_k_major("wq_b_weight", args.wq_b_weight);
            TORCH_CHECK(w_cols(args.wq_a_weight) == p.hidden,
                        "wq_a reduces over hidden ", p.hidden, ", got ",
                        w_cols(args.wq_a_weight));
            TORCH_CHECK(w_rows(args.wq_b_weight) == D::Q_TOTAL,
                        "wq_b must produce B_H*D_Q = ", (int)D::Q_TOTAL,
                        ", got ", w_rows(args.wq_b_weight));
            TORCH_CHECK(w_cols(args.wq_b_weight) == p.qlora_rank,
                        "wq_b reduces over the q-lora rank ", p.qlora_rank,
                        ", got ", w_cols(args.wq_b_weight));
        }
        p.kvlora_rank = w_rows(args.wkv_weight);
        if (args.use_kv_write) {
            check_fp8_k_major("wkv_weight", args.wkv_weight);
            TORCH_CHECK(w_cols(args.wkv_weight) == p.hidden,
                        "wkv reduces over hidden ", p.hidden, ", got ",
                        w_cols(args.wkv_weight));
        }
        check_model_dims(p.qlora_rank);

        p.grid_x = mega_mla::grid_x_for(T);
        p.n_groups = (T + TileT - 1) / TileT;
        TORCH_CHECK(p.grid_x >= TileT, "mega_mla: grid (", p.grid_x,
                    ") must be >= the token tile (", TileT,
                    ") so each token of a group lands on a distinct CTA");
        p.wsd = WS::dims(p.hidden, p.qlora_rank, p.kvlora_rank, p.olora_total);
        p.opdims = mega_mla::make_out_proj_dims(p.o_groups, p.o_lora_rank,
                                                p.hidden, CfgOA::KSplit);
        p.wsg = WS::geom(T, p.grid_x, p.n_groups, p.opdims.olora_rounds);
        const int wo_b_items = p.n_groups * p.opdims.m_tiles_all_wob;
        p.ks_wo_b = (wo_b_items * 4 <= p.grid_x)   ? 4
                    : (wo_b_items * 2 <= p.grid_x) ? 2
                                                   : 1;
        p.t_pad = WS::token_pad(T);
        TORCH_CHECK(args.workspace.scalar_type() == torch::kUInt8,
                    "workspace must be uint8");
        TORCH_CHECK(args.workspace.numel() >= p.wsd.bytes(p.wsg, p.t_pad),
                    "workspace too small: need ", p.wsd.bytes(p.wsg, p.t_pad),
                    " bytes for t_pad=", p.t_pad, ", got ",
                    args.workspace.numel());
        p.t_pad_sf = CfgA::SfTilePerGroup
                         ? mega_mla::WorkspaceLayout::SfPadRows * p.t_pad
                         : p.t_pad;
        TORCH_CHECK(!CfgA::SfTilePerGroup ||
                        p.n_groups * CfgA::SfMnPerTile <= p.t_pad_sf,
                    "mega_mla: the padded scale-factor region holds ",
                    p.t_pad_sf, " rows but ", p.n_groups,
                    " token groups at tile ", TileT, " need ",
                    p.n_groups * CfgA::SfMnPerTile);

        p.base = reinterpret_cast<char*>(args.workspace.data_ptr());
        p.q_ptr = reinterpret_cast<mega_mla::bf16*>(args.q_or_x.data_ptr());
        p.o_partial =
            reinterpret_cast<mega_mla::bf16*>(p.base + p.wsg.kv_split_off);

        char* cur = p.base + p.wsg.token_off;
        TORCH_CHECK(
            (reinterpret_cast<uintptr_t>(cur) & 31) == 0,
            "mega_mla: the workspace header leaves the token region at a ",
            (int)(reinterpret_cast<uintptr_t>(cur) & 31),
            "-byte offset; every header block must be a multiple of 32");
        const int t_pad = p.t_pad;
        auto take = [&cur, t_pad](int per_token) {
            char* got = cur;
            cur += (int64_t)t_pad * per_token;
            return got;
        };
        auto take_sf = [&take](int per_token) {
            return take(mega_mla::WorkspaceLayout::SfPadRows * per_token);
        };
        constexpr int SF = CfgA::SfGran;
        p.ws_x_fp8 = reinterpret_cast<mega_mla::e4m3*>(take(p.wsd.x_k));
        p.ws_x_sf = reinterpret_cast<mega_mla::ue8m0*>(take_sf(p.wsd.x_k / SF));
        p.ws_qlora_part =
            reinterpret_cast<float*>(take(p.wsd.qlora_k * 4 * CfgA::KSplit));
        p.ws_qlora_fp8 = reinterpret_cast<mega_mla::e4m3*>(take(p.wsd.qlora_k));
        p.ws_qlora_sf =
            reinterpret_cast<mega_mla::ue8m0*>(take_sf(p.wsd.qlora_k / SF));
        p.ws_kv_part =
            reinterpret_cast<float*>(take(p.wsd.kv_k * 4 * CfgA::KSplit));
        p.ws_flat_q_bf16 =
            reinterpret_cast<mega_mla::bf16*>(take(WS::FLAT_Q_K * 2));
        p.ws_flat_o_fp8 = reinterpret_cast<mega_mla::e4m3*>(take(WS::FLAT_O_K));
        p.ws_flat_o_sf =
            reinterpret_cast<mega_mla::ue8m0*>(take_sf(WS::FLAT_O_K / SF));
        p.ws_olora_bf16 = reinterpret_cast<mega_mla::bf16*>(
            take(p.wsd.olora_k * 2 * CfgOA::KSplitMax));
        p.ws_olora_fp8 = reinterpret_cast<mega_mla::e4m3*>(take(p.wsd.olora_k));
        p.ws_olora_sf =
            reinterpret_cast<mega_mla::ue8m0*>(take_sf(p.wsd.olora_k / SF));
        return p;
    }
};

// Every rendezvous the launch owns, as one object: a bank id plus the arrival
// count its participants agree on.  Two sites naming one bank with different
// counts is a silent hang, so no site may build a PartialBarrier itself.
template <class KT, class MP, class EP>
struct KernelBars {
    using Dims = typename KT::Dims;
    using WS = MegaMlaWorkspace<KT>;
    using Bar = mega_mla::PartialBarrier;
    using Roles = typename KT::Roles;
    static constexpr int TileT = KT::TileT;

    const MP& mp;
    const EP& ep;
    bool any_split;
    bool o_post_flat;
    int n_groups;
    int ks_wo_b;

    CUTE_DEVICE int nv(int g) const {
        return min(TileT, mp.num_tokens - g * TileT);
    }
    CUTE_DEVICE Bar of(int bank, int g, int count) const {
        return Bar(mp.partial_bar, bank * mp.pbar_bank + g, count);
    }
    CUTE_DEVICE Bar per_token(int bank, int g) const {
        return of(bank, g, nv(g) * Roles::VecWarps);
    }
    CUTE_DEVICE Bar per_item(int bank, int items, int g) const {
        return of(bank, g, items * Roles::DrainWarps);
    }
    CUTE_DEVICE Bar per_cta(int bank, int g) const {
        return of(bank, g, (int)gridDim.x * Roles::VecWarps);
    }

    CUTE_DEVICE Bar x_ready(int g) const {
        return per_token(WS::PB_X_READY, g);
    }
    CUTE_DEVICE Bar qq_ready(int g) const {
        return per_token(WS::PB_QQ_READY, g);
    }
    CUTE_DEVICE Bar oq_ready(int g) const {
        return per_token(WS::PB_OQ_READY, g);
    }
    CUTE_DEVICE Bar kvcache_ready(int g) const {
        return per_token(WS::PB_KVCACHE_READY, g);
    }
    CUTE_DEVICE Bar c128_ready(int g) const {
        return per_token(WS::PB_C128_READY, g);
    }
    CUTE_DEVICE Bar qlora_ready(int g) const {
        return per_item(WS::PB_QLORA_READY,
                        mp.shape_wq_a.m_tiles * KT::Qkv::KSplit, g);
    }
    CUTE_DEVICE Bar kv_ready(int g) const {
        return per_item(WS::PB_KV_READY, mp.shape_wkv.m_tiles * KT::Qkv::KSplit,
                        g);
    }
    CUTE_DEVICE Bar ring_ready(int g) const {
        return per_item(WS::PB_RING_READY, mp.shape_kv_gate.m_tiles, g);
    }
    CUTE_DEVICE Bar flat_q_ready(int g) const {
        return per_item(WS::PB_FLAT_Q_READY,
                        Dims::Q_TOTAL / mega_mla::GemmShapeDefaults::WeightTile,
                        g);
    }
    CUTE_DEVICE Bar olora_ready(int rnd, int tg) const {
        return per_item(WS::PB_OLORA_READY, ep.opdims.olora_items_per_slot,
                        rnd * n_groups + tg);
    }
    CUTE_DEVICE Bar wob_ready(int tg) const {
        return per_item(WS::PB_WOB_READY, ep.opdims.m_tiles_all_wob * ks_wo_b,
                        tg);
    }
    CUTE_DEVICE Bar flat_o(int g) const {
        return o_post_flat ? per_cta(WS::PB_FLAT_O_READY, g)
                           : per_token(WS::PB_FLAT_O_READY, g);
    }
    CUTE_DEVICE Bar o_coef(int g) const { return per_cta(WS::PB_O_COEF, g); }
    CUTE_DEVICE Bar flatq_free(int g) const {
        return of(WS::PB_FLATQ_FREE, g, nv(g));
    }
    CUTE_DEVICE Bar o_stored(int g) const {
        return of(WS::PB_O_STORED, g,
                  any_split ? __ldg(&mp.wplan.seg_in_group[g]) : nv(g));
    }

    CUTE_DEVICE void reset_all() const {
        const int n = WS::PARTIAL_BARRIER_BANKS * mp.pbar_bank;
        for (int i = (int)threadIdx.x; i < n; i += (int)blockDim.x)
            Bar(mp.partial_bar, i, 1).reset_relaxed();
    }
};

}  // namespace halops
