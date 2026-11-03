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

#include <cutlass/arch/barrier.h>
#include <cutlass/fast_math.h>

#include <cute/arch/copy_sm90_desc.hpp>

#include "kernel/moe/moe_configs.cuh"
#include "kernel/moe/moe_cta_comm.cuh"
#include "kernel/moe/moe_mainloop_down.cuh"
#include "kernel/moe/moe_mainloop_latent.cuh"
#include "kernel/moe/moe_mainloop_router.cuh"
#include "kernel/moe/moe_mainloop_shared.cuh"
#include "kernel/moe/moe_mainloop_up.cuh"
#include "kernel/moe/moe_scheduler.cuh"
#include "kernel/moe/moe_shared_storage.cuh"
#include "kernel/moe/moe_tma.cuh"
#include "kernel/moe/moe_traits.cuh"
#include "kernel/moe/moe_workspace.cuh"
#include "kernel/moe/quant_utils.cuh"

namespace halops::moe {

template <typename KernelTraits>
struct CollectiveMainloopMoE {
    using Traits = KernelTraits;
    using Dims = typename Traits::MoEDims;
    using SharedStorage = MoESharedStorage<Dims>;
    using Workspace = MoEWorkspace<Dims>;
    using Router = RouterProjection<KernelTraits, Traits::RouterGemm::kUseUmma>;
    using Latent = LatentDownProjection<KernelTraits>;

    // The routing tables as the 8-B words they travel between blocks in.
    static constexpr std::uint32_t kRoutingTableWords =
        static_cast<std::uint32_t>(sizeof(MoERoutingTables<Dims>) /
                                   sizeof(uint2));
    static_assert(sizeof(MoERoutingTables<Dims>) % sizeof(uint2) == 0u,
                  "the routing tables travel as whole 8-B words");

    // Host-side launch arguments: one member per runtime tensor plus the launch
    // scalars.
    struct Arguments {
        std::uint32_t num_tokens;
        std::uint32_t hidden_size;
        std::uint32_t latent_size;
        std::uint32_t intermediate_size;
        const DataType::ElementA* input;
        const typename Traits::RouterGemm::MmaElement* router_weight;
        const typename UpGemmTraits<Dims>::MmaElement* expert_up_weights;
        const typename UpGemmTraits<Dims>::ElementScale*
            expert_up_weights_scale;
        const typename DownGemmTraits<Dims>::MmaElement* expert_down_weights;
        const typename DownGemmTraits<Dims>::ElementScale*
            expert_down_weights_scale;
        const typename SharedUpGemmTraits<Dims>::MmaElement* shared_up_weight;
        const typename SharedUpGemmTraits<Dims>::ElementScale*
            shared_up_weight_scale;
        const typename SharedDownGemmTraits<Dims>::MmaElement*
            shared_down_weight;
        const typename SharedDownGemmTraits<Dims>::ElementScale*
            shared_down_weight_scale;
        const typename Traits::LatentDownGemm::MmaElement* latent_down_weight;
        const typename Traits::LatentDownGemm::ElementScale*
            latent_down_weight_scale;
        DataType::ElementOut* routed_out;
        DataType::ElementOut* shared_out;
        void* workspace;
        // Device event timer: the per-CTA record pool (null = off) and
        // whether the per-block detail probes should fire.
        std::uint64_t* timer_events;
        std::uint32_t top_k;
        ScoringFunc scoring_func;
        bool renormalize;
        Activation activation;
        float situ_beta;
        float situ_linear_beta;
        std::uint32_t l2_cached_bytes;
    };

    // Device-side launch parameters.
    struct Params {
        std::uint32_t num_tokens;
        std::uint32_t hidden_size;
        std::uint32_t latent_size;
        std::uint32_t intermediate_size;
        const DataType::ElementA* __restrict__ input;
        const typename Traits::RouterGemm::
            MmaElement* __restrict__ router_weight;
        const typename UpGemmTraits<
            Dims>::MmaElement* __restrict__ expert_weights_up;
        const typename UpGemmTraits<
            Dims>::ElementScale* __restrict__ expert_scales_up;
        const typename DownGemmTraits<
            Dims>::MmaElement* __restrict__ expert_weights_down;
        const typename DownGemmTraits<
            Dims>::ElementScale* __restrict__ expert_scales_down;
        DataType::ElementOut* __restrict__ activations_out;
        const typename Traits::LatentDownGemm::
            MmaElement* __restrict__ latent_down_weight;
        const typename Traits::LatentDownGemm::
            ElementScale* __restrict__ latent_down_weight_scale;
        const typename SharedUpGemmTraits<
            Dims>::MmaElement* __restrict__ shared_up_weight;
        const typename SharedUpGemmTraits<
            Dims>::ElementScale* __restrict__ shared_up_scale;
        const typename SharedDownGemmTraits<
            Dims>::MmaElement* __restrict__ shared_down_weight;
        const typename SharedDownGemmTraits<
            Dims>::ElementScale* __restrict__ shared_down_scale;
        DataType::ElementOut* __restrict__ shared_out;
        void* __restrict__ scratchpad;
        std::uint64_t* timer_events;
        std::uint32_t top_k;
        cutlass::FastDivmod topk_divmod;
        ScoringFunc scoring_func;
        bool renormalize;
        Activation activation;
        float situ_beta;
        float situ_linear_beta;
        // How many experts' weights fit in L2: the eviction hint flips to
        // evict-first above it.
        std::uint32_t l2_expert_capacity;
        alignas(128) CUtensorMap tma_up_weights_desc;
        alignas(128) CUtensorMap tma_input_desc;
        alignas(128) CUtensorMap tma_latent_out_desc;
        alignas(128) CUtensorMap tma_down_weights_desc;
        alignas(128) CUtensorMap tma_down_activations_desc;
        alignas(128) CUtensorMap tma_latent_down_weight_desc;
        alignas(128) CUtensorMap tma_shared_up_desc;
        alignas(128) CUtensorMap tma_shared_down_desc;
    };

    // Builds the device parameters, including the four TMA descriptors.
    static Params to_underlying_arguments(Arguments const& args) {
        Params params{};
        params.num_tokens = args.num_tokens;
        params.hidden_size = args.hidden_size;
        params.latent_size = args.latent_size;
        params.intermediate_size = args.intermediate_size;
        params.input = args.input;
        params.router_weight = args.router_weight;
        params.expert_weights_up = args.expert_up_weights;
        params.expert_scales_up = args.expert_up_weights_scale;
        params.expert_weights_down = args.expert_down_weights;
        params.expert_scales_down = args.expert_down_weights_scale;
        params.activations_out = args.routed_out;
        params.latent_down_weight = args.latent_down_weight;
        params.latent_down_weight_scale = args.latent_down_weight_scale;
        params.shared_up_weight = args.shared_up_weight;
        params.shared_up_scale = args.shared_up_weight_scale;
        params.shared_down_weight = args.shared_down_weight;
        params.shared_down_scale = args.shared_down_weight_scale;
        params.shared_out = args.shared_out;
        params.scratchpad = args.workspace;
        params.timer_events = args.timer_events;
        params.top_k = args.top_k;
        params.topk_divmod = cutlass::FastDivmod(static_cast<int>(args.top_k));
        params.scoring_func = args.scoring_func;
        params.renormalize = args.renormalize;
        params.activation = args.activation;
        params.situ_beta = args.situ_beta;
        params.situ_linear_beta = args.situ_linear_beta;

        const std::uint64_t expert_width =
            args.latent_size > 0u ? args.latent_size : args.hidden_size;
        const std::uint64_t expert_weight_bytes =
            (2ull * args.intermediate_size * expert_width *
                 Traits::UpGemm::MmaCore::kElementABits +
             static_cast<std::uint64_t>(args.intermediate_size) * expert_width *
                 Traits::DownGemm::MmaCore::kElementABits) /
            8u;
        params.l2_expert_capacity =
            args.l2_cached_bytes > 0u && expert_weight_bytes > 0u
                ? static_cast<std::uint32_t>(args.l2_cached_bytes /
                                             expert_weight_bytes)
                : 0xFFFFFFFFu;

        params.tma_input_desc = CreateTmaDesc2D(
            "activations", args.input, args.num_tokens, Dims::kHiddenSize,
            /*box_rows=*/8u, /*box_cols=*/64u);
        if constexpr (Traits::kUseLatent && !Traits::kLatentQuantizeOutput) {
            params.tma_latent_out_desc = CreateTmaDesc2D(
                "activations",
                reinterpret_cast<Workspace*>(args.workspace)
                    ->latent_down_out.data_ptr(),
                Dims::kNumTokens, Dims::kLatentSize, /*box_rows=*/8u,
                /*box_cols=*/64u);
        }

        params.tma_up_weights_desc = CreateTmaDesc2D(
            "up-projection weights", args.expert_up_weights,
            /*rows=*/Dims::kNumExperts * 2u * Dims::kIntermediateSize,
            /*cols=*/MoEInputSize<Dims>::value, /*box_rows=*/128u,
            /*box_cols=*/128u);
        params.tma_down_weights_desc = CreateTmaDesc2D(
            "down-projection weights", args.expert_down_weights,
            /*rows=*/Dims::kNumExperts * MoEInputSize<Dims>::value,
            /*cols=*/Dims::kIntermediateSize, /*box_rows=*/128u,
            /*box_cols=*/128u);
        params.tma_down_activations_desc =
            CreateTmaDesc2D("down-projection activations",
                            reinterpret_cast<const DataType::ElementQA*>(
                                static_cast<const char*>(args.workspace) +
                                Workspace::kUpOutFp8Offset),
                            Workspace::kMaxRoutedRows, Dims::kIntermediateSize,
                            /*box_rows=*/Dims::kNumTokens, /*box_cols=*/128u);

        params.tma_shared_up_desc =
            CreateTmaDesc2D("up-projection weights", args.shared_up_weight,
                            /*rows=*/Dims::kIntermediateSize * 2u,
                            /*cols=*/Dims::kHiddenSize,
                            /*box_rows=*/SharedUpGemmTraits<Dims>::kTileShapeM,
                            /*box_cols=*/128u);
        params.tma_shared_down_desc = CreateTmaDesc2D(
            "down-projection weights", args.shared_down_weight,
            /*rows=*/Dims::kHiddenSize,
            /*cols=*/Dims::kIntermediateSize,
            /*box_rows=*/SharedDownGemmTraits<Dims>::kTileShapeM,
            /*box_cols=*/128u);
        if (args.latent_down_weight != nullptr) {
            params.tma_latent_down_weight_desc = CreateTmaDesc2D(
                "latent-projection weights", args.latent_down_weight,
                Dims::kLatentSize, Dims::kHiddenSize,
                /*box_rows=*/Traits::LatentDownGemm::kTileShapeM,
                /*box_cols=*/128u);
        }
        return params;
    }

    // Phase 0a: the latent down-projection, on the blocks the router leaves
    // idle.
    __device__ void run_latent_down(Params const& params,
                                    SharedStorage* __restrict__ smem) {
        if constexpr (Traits::kUseLatent) {
            Latent::run(params.tma_latent_down_weight_desc,
                        params.tma_input_desc, params.latent_down_weight_scale,
                        reinterpret_cast<Workspace*>(params.scratchpad), smem,
                        static_cast<std::uint32_t>(params.num_tokens));
        }
    }

    // Phase 0b: the shared expert, on the blocks neither the router nor the
    // latent down-projection occupies.
    __device__ void run_shared(Params const& params,
                               SharedStorage* __restrict__ smem) {
        SharedExpertProjection<Traits>::run(
            params.tma_shared_up_desc, params.tma_shared_down_desc,
            params.tma_input_desc, params.shared_up_scale,
            params.shared_down_scale, params.shared_out,
            reinterpret_cast<Workspace*>(params.scratchpad), smem,
            static_cast<std::uint32_t>(params.num_tokens), params.activation,
            GatedActParams{params.situ_beta, params.situ_linear_beta});
    }

    // Phase 0c: the router Linear on this block's expert tiles, then this
    // block's routing tables into shared memory.
    __device__ static void run_router_topk(Params const& params,
                                           Workspace* __restrict__ workspace,
                                           SharedStorage* __restrict__ smem,
                                           std::uint32_t lane_idx) {
        constexpr std::uint32_t kExpertsPerThread = Dims::kNumExperts / 32u;

        // One warp per token, unless the launch has more tokens than the calc
        // warps have: then a cluster pair splits them, one token each.
        const bool coop_topk = Traits::kClusterSize >= 2u &&
                               params.num_tokens > WarpRole::kCalcWarpCount;
        if (threadIdx.x == 0u) {
            // One task per router CTA, whatever its M tile or K slice.
            const auto logits_span =
                KERNEL_TIMER_BEGIN(smem->timer(), "router_wait_logits");
            workspace->cta_comm.wait_router_tasks(smem->launch_parity,
                                                  Traits::RouterGemm::kNumCtas);
            KERNEL_TIMER_END(smem->timer(), logits_span);
        }
        WarpRole::sync_calc_threads();
        const std::uint32_t calc_warp_idx = WarpRole::get_calc_warp_idx();
        // The coop cluster's blocks take one token half each, one
        // warp per token; a single-block selection walks them all.
        constexpr std::uint32_t kTokensPerCta = (Dims::kNumTokens + 1u) / 2u;
        static_assert(kTokensPerCta <= WarpRole::kCalcWarpCount,
                      "one warp per token must cover a split block");
        const std::uint32_t tok_base =
            coop_topk ? (blockIdx.x % 2u) * kTokensPerCta : 0u;
        if (coop_topk) {
            // One warp per token, and this block runs only its half.
            const std::uint32_t tok = tok_base + calc_warp_idx;
            if (calc_warp_idx < kTokensPerCta && tok < params.num_tokens) {
                const auto topk_span =
                    KERNEL_TIMER_BEGIN(smem->timer(), "topk_one_token");
                topk_one_token<Dims, kExpertsPerThread>(
                    tok, params.top_k, params.scoring_func, params.renormalize,
                    workspace, smem);
                KERNEL_TIMER_END(smem->timer(), topk_span);
            }
        } else {
            constexpr std::uint32_t kTokensPerWarp =
                Dims::kNumTokens / WarpRole::kCalcWarpCount;
            static_assert(Dims::kNumTokens % WarpRole::kCalcWarpCount == 0u,
                          "the tokens must divide over the calc warps");
#pragma unroll 1
            for (std::uint32_t g = 0; g < kTokensPerWarp; ++g) {
                const std::uint32_t tok =
                    calc_warp_idx + g * WarpRole::kCalcWarpCount;
                if (tok >= params.num_tokens) break;
                const auto topk_span =
                    KERNEL_TIMER_BEGIN(smem->timer(), "topk_one_token");
                topk_one_token<Dims, kExpertsPerThread>(
                    tok, params.top_k, params.scoring_func, params.renormalize,
                    workspace, smem);
                KERNEL_TIMER_END(smem->timer(), topk_span);
            }
        }
        WarpRole::sync_calc_threads();
        if (coop_topk) {
            // My half of the selection is in my own scratch; the pair
            // swaps the other half through the cluster's shared
            // memory, so both blocks end up with the whole table.
            if (calc_warp_idx == 0u) {
                using Coop = decltype(smem->barriers.router_coop);
                const std::uint32_t peer_rank =
                    (blockIdx.x % Traits::kClusterSize) ^ 1u;
                if (lane_idx == 0u) {
                    const auto coop_span = KERNEL_TIMER_BEGIN(
                        smem->timer(), "router_coop_exchange");
                    smem->barriers.router_coop.arrive();
                    Coop::arrive_remote(
                        smem->barriers.router_coop.partner(peer_rank));
                    KERNEL_TIMER_END(smem->timer(), coop_span);
                }
                smem->barriers.router_coop.wait();
                // The partner's half as 16-B words, its ids and then its
                // weights: one word per lane, so the copy is one round trip.
                constexpr std::uint32_t kPairs = kTokensPerCta * Dims::kMaxTopk;
                static_assert(
                    kPairs * sizeof(std::uint16_t) % sizeof(uint4) == 0u,
                    "a half's ids and weights must be whole words");
                constexpr std::uint32_t kIdWords =
                    kPairs * sizeof(std::uint16_t) / sizeof(uint4);
                constexpr std::uint32_t kWords =
                    kIdWords + kPairs * sizeof(float) / sizeof(uint4);
                static_assert(kWords <= WarpRole::kThreadsPerWarp,
                              "one word per lane");
                const std::uint32_t peer_base =
                    (tok_base + kTokensPerCta) % (kTokensPerCta * 2u);
                const std::uint32_t at = peer_base * Dims::kMaxTopk;
                auto* ids = reinterpret_cast<uint4*>(
                    &smem->routing_scratch.topk_ids[at]);
                auto* weights = reinterpret_cast<uint4*>(
                    &smem->routing_scratch.topk_weights[at]);
                if (lane_idx < kWords) {
                    uint4* word = lane_idx < kIdWords
                                      ? ids + lane_idx
                                      : weights + (lane_idx - kIdWords);
                    *word = cluster_load_v4(
                        cluster_map_shared_rank(word, peer_rank));
                }
                __syncwarp();
            }
        }
    }

    __device__ void run_router(Params const& params,
                               SharedStorage* __restrict__ smem) {
        Workspace* __restrict__ workspace =
            reinterpret_cast<Workspace*>(params.scratchpad);
        if (Router::is_router_cta()) {
            Router::run(params.router_weight, params.tma_input_desc, workspace,
                        smem, params.scoring_func);
        }

        // Only the router's own blocks run the selection and build the routing
        // tables; every other block copies the tables block 0 publishes.
        const bool reads_published_tables = !Router::is_router_cta();
        const std::uint32_t lane_idx = WarpRole::get_lane_idx();
        if (!reads_published_tables) {
            if (WarpRole::is_calc_warp(WarpRole::get_warp_idx())) {
                run_router_topk(params, workspace, smem, lane_idx);
            }
            if (WarpRole::get_calc_warp_idx() == 0u) {
                // Every router block holds the whole selection by now (a
                // coop pair has swapped halves), so it builds its own.
                const auto tables_span =
                    KERNEL_TIMER_BEGIN(smem->timer(), "prepare_routing_tables");
                prepare_routing_tables<Dims>(params.num_tokens, params.top_k,
                                             params.topk_divmod, smem);
                KERNEL_TIMER_END(smem->timer(), tables_span);
                // Block 0 hands its finished tables to the other blocks.
                if (blockIdx.x == 0u) {
                    publish_routing_tables(workspace, smem, lane_idx);
                }
            }
        } else if (WarpRole::is_calc_warp(WarpRole::get_warp_idx()) &&
                   WarpRole::get_calc_warp_idx() == 0u) {
            // After the projection: one warp copies block 0's tables, and it
            // does not contend with the projection's own warps.
            if (lane_idx == 0u) {
                const auto wait_span =
                    KERNEL_TIMER_BEGIN(smem->timer(), "router_wait_tables");
                workspace->cta_comm.wait_router_tables(smem->launch_parity);
                KERNEL_TIMER_END(smem->timer(), wait_span);
            }
            __syncwarp();
            const auto load_span =
                KERNEL_TIMER_BEGIN(smem->timer(), "load_routing_tables");
            load_routing_tables(workspace, smem, lane_idx);
            KERNEL_TIMER_END(smem->timer(), load_span);
        }
    }

    // Block 0's calc warp 0: its tables into the workspace, then the flag.
    __device__ static __forceinline__ void publish_routing_tables(
        Workspace* __restrict__ workspace, SharedStorage* __restrict__ smem,
        std::uint32_t lane_idx) {
        const uint2* src = reinterpret_cast<const uint2*>(&smem->routing);
        uint2* dst = reinterpret_cast<uint2*>(&workspace->router_tables);
        // Every lane's build stores land before the copy reads them.
        __syncwarp();
        for (std::uint32_t i = lane_idx; i < kRoutingTableWords;
             i += WarpRole::kThreadsPerWarp) {
            dst[i] = src[i];
        }
        __threadfence();
        __syncwarp();
        if (lane_idx == 0u) {
            workspace->cta_comm.publish_router_tables(smem->launch_parity);
        }
    }

    // A non-router block's calc warp 0: block 0's tables into its own.  Every
    // load is issued before any store, so the copy is one round trip.
    __device__ static __forceinline__ void load_routing_tables(
        Workspace const* __restrict__ workspace,
        SharedStorage* __restrict__ smem, std::uint32_t lane_idx) {
        constexpr std::uint32_t kPerLane =
            (kRoutingTableWords + WarpRole::kThreadsPerWarp - 1u) /
            WarpRole::kThreadsPerWarp;
        const uint2* src =
            reinterpret_cast<const uint2*>(&workspace->router_tables);
        uint2* dst = reinterpret_cast<uint2*>(&smem->routing);
        uint2 words[kPerLane];
#pragma unroll
        for (std::uint32_t j = 0; j < kPerLane; ++j) {
            const std::uint32_t i = lane_idx + j * WarpRole::kThreadsPerWarp;
            if (i < kRoutingTableWords) words[j] = src[i];
        }
#pragma unroll
        for (std::uint32_t j = 0; j < kPerLane; ++j) {
            const std::uint32_t i = lane_idx + j * WarpRole::kThreadsPerWarp;
            if (i < kRoutingTableWords) dst[i] = words[j];
        }
        __syncwarp();
    }

    // The up phase's staging loader: the activation through the input ring,
    // one K tile per slot, in the order the quantize reads them.
    __device__ static void run_load_up_input_k_loop(
        CUtensorMap const& tma_desc, std::uint32_t num_tiles,
        SharedStorage* __restrict__ smem) {
        // This phase's own streaming tile, never the router's geometry.
        constexpr std::uint32_t kBytes =
            Traits::UpGemm::kTileShapeK * Dims::kNumTokens *
            static_cast<std::uint32_t>(sizeof(DataType::ElementA));
#pragma unroll 1
        for (std::uint32_t tile = 0; tile < num_tiles; ++tile) {
            const std::uint32_t slot = tile % Traits::kUpInputSlots;
            if (tile >= Traits::kUpInputSlots) {
                smem->barriers.up_input_slot_empty.wait(tile -
                                                        Traits::kUpInputSlots);
            }
            smem->barriers.up_input_slot_ready.arrive(tile, kBytes);
            load_input_tile<Dims::kNumTokens>(
                tma_desc, tile, /*token_start=*/0u,
                smem->routed_experts.input.data[slot],
                smem->barriers.up_input_slot_ready.raw(slot));
        }
    }

    // Calc warps: the whole activation in one pass, for the configs whose
    // input ring holds the projection output outright.
    __device__ void quantize_up_input_whole(SharedStorage* __restrict__ smem,
                                            int batch_size) {
        const auto quantize_span =
            KERNEL_TIMER_BEGIN(smem->timer(), "up_quantize_input");
        if constexpr (Traits::UpGemm::MmaCore::kUseBlockscaled) {
            quantize_mxfp8_groupwise_1x32<
                /*Staged=*/false, Dims::kNumTokens, Dims::kLatentSize,
                Dims::kQuantConfig.kBlockSize, /*SwizzleBoxK=*/1u,
                Traits::kQuantBlocksPerTile>(
                reinterpret_cast<const DataType::ElementA*>(
                    &smem->routed_experts.input.data[0][0][0]),
                /*k_base=*/0u, &smem->routed_experts.quantized_input.data[0],
                &smem->routed_experts.quantized_input.scale[0][0],
                /*staging_slot_base=*/0u, /*src_block_base=*/0u, batch_size,
                /*row_start=*/0u, WarpRole::get_calc_warp_idx(),
                WarpRole::kCalcWarpCount);
        } else {
            quantize_fp8_groupwise_128<
                /*Staged=*/false, Dims::kNumTokens, Dims::kLatentSize,
                Traits::UpGemm::kGemmN, Dims::kQuantConfig.kBlockSize,
                /*SwizzleBoxK=*/1u,
                /*BlocksPerStage=*/Traits::kQuantBlocksPerTile>(
                reinterpret_cast<const DataType::ElementA*>(
                    &smem->routed_experts.input.data[0][0][0]),
                /*k_base=*/0u, &smem->routed_experts.quantized_input.data[0],
                &smem->routed_experts.quantized_input.scale[0][0],
                /*staging_block_base=*/0u, /*scale_block_base=*/0u,
                /*src_block_base=*/0u, batch_size,
                /*row_start=*/0u, WarpRole::get_calc_warp_idx(),
                WarpRole::kCalcWarpCount);
        }
        // One pass, every tile at once: the issuer's waits all pass straight
        // through.
        KERNEL_TIMER_END(smem->timer(), quantize_span);
        WarpRole::sync_calc_threads();
        if (WarpRole::get_calc_warp_idx() == 0u &&
            WarpRole::get_lane_idx() == 0u) {
            for (std::uint32_t tile = 0; tile < Traits::UpGemm::kNumTilesK;
                 ++tile) {
                smem->barriers.up_staging_ready.arrive(tile);
            }
        }
    }

    // Calc warps: quantize the up phase's bf16 activation out of the input
    // ring, one K tile at a time, freeing each slot as it lands.
    __device__ void quantize_up_input(std::uint32_t num_tiles,
                                      SharedStorage* __restrict__ smem,
                                      int batch_size) {
        const auto quantize_span =
            KERNEL_TIMER_BEGIN(smem->timer(), "up_quantize_input");
        constexpr std::uint32_t kInputTileK = Traits::UpGemm::kTileShapeK;
        constexpr std::uint32_t kBlocksPerTile =
            kInputTileK / Dims::kQuantConfig.kBlockSize;
        const std::uint32_t warp_idx = WarpRole::get_calc_warp_idx();
#pragma unroll 1
        for (std::uint32_t tile = 0; tile < num_tiles; ++tile) {
            const std::uint32_t slot = tile % Traits::kUpInputSlots;
            smem->barriers.up_input_slot_ready.wait(tile);
            if constexpr (Traits::UpGemm::MmaCore::kUseBlockscaled) {
                quantize_mxfp8_groupwise_1x32<
                    /*Staged=*/true, Dims::kNumTokens, kInputTileK,
                    Dims::kQuantConfig.kBlockSize,
                    SwizzleBoxK<DataType::ElementA>::value, kBlocksPerTile>(
                    &smem->routed_experts.input.data[slot][0][0][0],
                    tile * kInputTileK,
                    &smem->routed_experts.quantized_input.data[0],
                    &smem->routed_experts.quantized_input.scale[0][0], tile,
                    tile * kBlocksPerTile, batch_size, /*row_start=*/0u,
                    warp_idx, WarpRole::kCalcWarpCount);
            } else {
                quantize_fp8_groupwise_128<
                    /*Staged=*/true, Dims::kNumTokens, kInputTileK,
                    Traits::UpGemm::kGemmN, Dims::kQuantConfig.kBlockSize,
                    SwizzleBoxK<DataType::ElementA>::value, kBlocksPerTile>(
                    &smem->routed_experts.input.data[slot][0][0][0],
                    tile * kInputTileK,
                    &smem->routed_experts.quantized_input.data[0],
                    &smem->routed_experts.quantized_input.scale[0][0],
                    /*staging_slot_base=*/tile, tile * kBlocksPerTile,
                    tile * kBlocksPerTile, batch_size,
                    /*row_start=*/0u, warp_idx, WarpRole::kCalcWarpCount);
            }
            __syncwarp();
            if (WarpRole::get_lane_idx() == 0u) {
                smem->barriers.up_input_slot_empty.arrive(tile);
            }
            // This tile's fp8 staging is complete once every calc warp has
            // stored: the issuer may consume it while the next one quantizes.
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            WarpRole::sync_calc_threads();
            if (warp_idx == 0u && WarpRole::get_lane_idx() == 0u) {
                smem->barriers.up_staging_ready.arrive(tile);
            }
        }
        KERNEL_TIMER_END(smem->timer(), quantize_span);
    }

    // The up entry's staging when the projection quantized its output: both
    // tensors are already the B operand's image, so this is one bulk copy each
    // and the issuer can read the staging as soon as it lands.
    __device__ static void load_quantized_activation(
        Workspace* __restrict__ workspace, SharedStorage* __restrict__ smem) {
        using Quantized = decltype(smem->routed_experts.quantized_input);
        constexpr std::uint32_t kFp8Bytes =
            static_cast<std::uint32_t>(sizeof(Quantized::data));
        constexpr std::uint32_t kScaleBytes =
            static_cast<std::uint32_t>(sizeof(Quantized::scale));
        static_assert(kFp8Bytes == Workspace::LatentOut::kDataBytes &&
                          kScaleBytes == Workspace::LatentOut::kScaleBytes,
                      "the projection's output must be exactly one up "
                      "staging tile.");
        smem->barriers.latent_copy_ready.arrive(0u, kFp8Bytes + kScaleBytes);
        tma_load_1d(workspace->latent_down_out.data_ptr(), kFp8Bytes,
                    &smem->routed_experts.quantized_input.data[0][0][0][0],
                    smem->barriers.latent_copy_ready.raw(0u));
        tma_load_1d(workspace->latent_down_out.scale_ptr(), kScaleBytes,
                    &smem->routed_experts.quantized_input.scale[0][0],
                    smem->barriers.latent_copy_ready.raw(0u));
        smem->barriers.latent_copy_ready.wait(0u);
        // Every K tile of the staging is complete at once.
        for (std::uint32_t tile = 0; tile < Traits::UpGemm::kNumTilesK;
             ++tile) {
            smem->barriers.up_staging_ready.arrive(tile);
        }
    }

    // Phase 2: up-projection over this block's tasks, as enumerated by `sched`.
    __device__ void run_up(Params const& params,
                           SharedStorage* __restrict__ smem,
                           TaskScheduler<KernelTraits>& sched) {
        Workspace* __restrict__ workspace =
            reinterpret_cast<Workspace*>(params.scratchpad);
        // The up phase's entry: the fp8 B operand.
        constexpr bool kWholeLatentFits =
            Dims::kLatentSize <=
            Traits::kUpInputSlots * Traits::UpGemm::kTileShapeK;
        if (WarpRole::is_other_loader_thread()) {
            if constexpr (Traits::kUseLatent) {
                const auto latent_span =
                    KERNEL_TIMER_BEGIN(smem->timer(), "up_wait_latent");
                workspace->cta_comm.wait_latent_tasks(
                    smem->launch_parity, Traits::LatentDownGemm::kNumCtas);
                KERNEL_TIMER_END(smem->timer(), latent_span);
            }
            const auto activation_span =
                KERNEL_TIMER_BEGIN(smem->timer(), "up_load_activation");
            if constexpr (Traits::kLatentQuantizeOutput) {
                load_quantized_activation(workspace, smem);
            } else if constexpr (Traits::kUseLatent && kWholeLatentFits) {
                // Only the launch's rows: the padding rows are zero-filled by
                // the quantize, which never reads their source.
                const std::uint32_t kBytes =
                    params.num_tokens * Dims::kLatentSize *
                    static_cast<std::uint32_t>(sizeof(DataType::ElementA));
                smem->barriers.latent_copy_ready.arrive(0u, kBytes);
                tma_load_1d(workspace->latent_down_out.data_ptr(), kBytes,
                            reinterpret_cast<DataType::ElementA*>(
                                &smem->routed_experts.input.data[0][0][0]),
                            smem->barriers.latent_copy_ready.raw(0u));
            } else if constexpr (Traits::kUseLatent) {
                run_load_up_input_k_loop(params.tma_latent_out_desc,
                                         Traits::UpGemm::kNumTilesK, smem);
            } else {
                run_load_up_input_k_loop(params.tma_input_desc,
                                         Traits::UpGemm::kNumTilesK, smem);
            }
            KERNEL_TIMER_END(smem->timer(), activation_span);
        }
        if constexpr (!Traits::kLatentQuantizeOutput) {
            if (WarpRole::is_calc_warp(WarpRole::get_warp_idx())) {
                if constexpr (Traits::kUseLatent && kWholeLatentFits) {
                    smem->barriers.latent_copy_ready.wait(0u);
                    quantize_up_input_whole(
                        smem, static_cast<int>(params.num_tokens));
                } else {
                    quantize_up_input(Traits::UpGemm::kNumTilesK, smem,
                                      static_cast<int>(params.num_tokens));
                }
                // The UMMA reads these stores through the async proxy.
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                WarpRole::sync_calc_threads();
                if (WarpRole::get_lane_idx() == 0u) {
                    smem->barriers.up_input_ready.arrive(0u);
                }
            }
        }

        UpProjection<KernelTraits>::run(
            params.expert_scales_up, params.num_tokens, params.activation,
            GatedActParams{params.situ_beta, params.situ_linear_beta},
            workspace, smem, params.tma_up_weights_desc, sched,
            get_tma_cache_hint(smem->routing.active_expert_count >
                               params.l2_expert_capacity));
    }

    // Phase 3: down-projection over this block's tasks, as enumerated by
    // `sched`.
    __device__ void run_down(Params const& params,
                             SharedStorage* __restrict__ smem,
                             TaskScheduler<KernelTraits>& sched) {
        DownProjection<KernelTraits>::run(
            params.expert_scales_down,
            reinterpret_cast<Workspace*>(params.scratchpad), smem,
            params.tma_down_weights_desc, params.tma_down_activations_desc,
            sched,
            get_tma_cache_hint(smem->routing.active_expert_count >
                               params.l2_expert_capacity));
    }

    // Pulls the TMA descriptors into the descriptor cache, so the first TMA of
    // each phase does not wait on a fetch of its descriptor from the kernel
    // parameters.
    __device__ static void prefetch_tma_descriptors(Params const& params) {
        cute::prefetch_tma_descriptor(&params.tma_up_weights_desc);
        cute::prefetch_tma_descriptor(&params.tma_input_desc);
        cute::prefetch_tma_descriptor(&params.tma_down_weights_desc);
        cute::prefetch_tma_descriptor(&params.tma_down_activations_desc);
    }

    // Per-block setup: mbarrier and TMEM setup, the input-activation TMA and
    // the per-block launch parity.
    __device__ void initialize(Params const& params,
                               SharedStorage* __restrict__ smem) {
        Workspace* __restrict__ workspace =
            reinterpret_cast<Workspace*>(params.scratchpad);
        const std::uint32_t warp_idx = WarpRole::get_warp_idx();
        const std::uint32_t lane_idx = WarpRole::get_lane_idx();

        if (WarpRole::is_tmem_warp(warp_idx)) {
            // TMEM allocation issued by ONE fully-active warp.
            tmem_alloc(Traits::kTmemColumns, &smem->tmem_ptr);
        } else if (warp_idx == 1 && lane_idx == 0) {
            smem->barriers.init_router_ring();
            smem->barriers.init_latent_ring();
            smem->barriers.init_expert_ring();
            smem->barriers.init_shared_ring();
            cutlass::arch::fence_barrier_init();
        } else if (warp_idx == 2 && lane_idx == 0) {
            // A warp with no other setup role; one thread covers the block.
            prefetch_tma_descriptors(params);
        } else if (warp_idx == 3 && lane_idx == 0) {
            // Per-CTA private launch counter.
            smem->launch_parity = workspace->cta_comm.claim_parity();
            // The down task queue starts empty.
            smem->down_tasks_claimed = 0u;
            smem->down_task_stores = 0u;
        }
        __syncthreads();

        if (warp_idx >= 4 && warp_idx < 8) {
            // Reset the readiness state for the NEXT launch.
            constexpr std::uint32_t kZeroThreads =
                4u * WarpRole::kThreadsPerWarp;
            // The ranks are the grid's zeroing threads in order, so the
            // launch covers the rows as long as the smallest grid still
            // reaches them.
            static_assert(
                Traits::kMinGrid * kZeroThreads >= Workspace::kMaxRoutedRows,
                "the grid's zeroing threads must cover the per-row "
                "up_out_ready words that reset_inactive zeroes by "
                "`rank`.");

            std::uint32_t launch_parity = smem->launch_parity;
            std::uint32_t zero_tid =
                threadIdx.x - 4u * WarpRole::kThreadsPerWarp;
            // One pass may not reach every row: stride until the table is
            // covered.
            for (std::uint32_t rank = blockIdx.x * kZeroThreads + zero_tid;
                 rank < Workspace::kMaxRoutedRows; rank += kZeroThreads) {
                workspace->cta_comm.reset_inactive(launch_parity, rank);
            }
        }
    }

    // Phase 4: waits for an m-block to be complete and reduces its per-group
    // partials into the output tensor.
    __device__ void combine(Params const& params,
                            SharedStorage* __restrict__ smem,
                            CtaGroup<Traits::kCtasPerExpert> const& cta_group) {
        Workspace* __restrict__ workspace =
            reinterpret_cast<Workspace*>(params.scratchpad);
        DataType::ElementOut* __restrict__ activations_out =
            params.activations_out;
        const std::uint32_t batch_size = params.num_tokens;
        constexpr std::uint32_t kDownOutputColsPerCta =
            Traits::DownGemm::MmaCore::kTileShapeM;
        // The block combines m-block `cta_idx_in_group`, whose cells the
        // num_groups blocks holding that index split by `group_idx`.
        if (!cta_group.in_whole_group()) return;
        const std::uint32_t m_block = cta_group.cta_idx_in_group;
        const std::uint32_t base_col = m_block * kDownOutputColsPerCta;
        const std::uint32_t first_cell =
            cta_group.group_idx * blockDim.x + threadIdx.x;
        const std::uint32_t cell_stride = cta_group.num_groups * blockDim.x;
        const std::uint32_t num_cells = batch_size * kDownOutputColsPerCta;
        const std::uint32_t top_k = params.top_k;

        if constexpr (Traits::kCombineWaitPerToken) {
            // A token's cells fill whole warps, so each wait is warp-uniform
            // and every lane acquires the wait for itself.
            static_assert(
                kDownOutputColsPerCta % WarpRole::kThreadsPerWarp == 0u,
                "a warp's combine cells must share one token");
            __syncwarp();
            for (std::uint32_t flat = first_cell; flat < num_cells;
                 flat += cell_stride) {
                const std::uint32_t tok = flat / kDownOutputColsPerCta;
                const auto wait_span =
                    KERNEL_TIMER_BEGIN(smem->timer(), "wait_down_out", tok);
                workspace->cta_comm.wait_down_out_field(
                    smem->launch_parity, m_block,
                    tok * PackedTokenCounts<Dims>::kBits,
                    PackedTokenCounts<Dims>::kMask, top_k);
                KERNEL_TIMER_END(smem->timer(), wait_span);
                const auto gather_span = KERNEL_TIMER_BEGIN(
                    smem->timer(), "gather_and_combine", tok);
                gather_cell(workspace, activations_out, tok,
                            base_col + flat % kDownOutputColsPerCta, top_k);
                KERNEL_TIMER_END(smem->timer(), gather_span);
            }
            return;
        }

        // Readiness wait: this block's m-block completes once every active
        // expert's task of it has been published.
        const auto m_block_wait =
            KERNEL_TIMER_BEGIN(smem->timer(), "wait_down_out");
        if (threadIdx.x == 0) {
            workspace->cta_comm.wait_down_out(
                smem->launch_parity, m_block,
                smem->routing.active_expert_count);
        }
        __syncthreads();
        KERNEL_TIMER_END(smem->timer(), m_block_wait);

        const auto gather_span =
            KERNEL_TIMER_BEGIN(smem->timer(), "gather_and_combine");
        for (std::uint32_t flat = first_cell; flat < num_cells;
             flat += cell_stride) {
            gather_cell(workspace, activations_out,
                        flat / kDownOutputColsPerCta,
                        base_col + flat % kDownOutputColsPerCta, top_k);
        }
        KERNEL_TIMER_END(smem->timer(), gather_span);
    }

    // One output cell: sums token `tok`'s top_k combine slots at column `col`
    // (slot = token*kMaxTopk + k, written by the block that ran the token's
    // k-th expert), in fixed k order and accumulating in FP32.
    __device__ static __forceinline__ void gather_cell(
        Workspace* __restrict__ workspace,
        DataType::ElementOut* __restrict__ activations_out, std::uint32_t tok,
        std::uint32_t col, std::uint32_t top_k) {
        // Token `tok`'s top-k slot run, at this thread's column.  Consecutive
        // threads walk consecutive `col`, so each k-step is a coalesced load.
        DataType::ElementOut slot_v[Dims::kMaxTopk];
#pragma unroll
        for (std::uint32_t k = 0; k < Dims::kMaxTopk; ++k) {
            slot_v[k] =
                workspace->routed_down_out.row(tok * Dims::kMaxTopk + k)[col];
        }
        float v = 0.f;
#pragma unroll
        for (std::uint32_t k = 0; k < Dims::kMaxTopk; ++k) {
            if (k < top_k) v += static_cast<float>(slot_v[k]);
        }
        activations_out[tok * Workspace::RoutedDownOut::kNumCols + col] =
            static_cast<DataType::ElementOut>(v);
    }
};

}  // namespace halops::moe
