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
#include "kernel/moe/moe_mainloop_latent_down.cuh"
#include "kernel/moe/moe_mainloop_routed_down.cuh"
#include "kernel/moe/moe_mainloop_routed_up.cuh"
#include "kernel/moe/moe_mainloop_router.cuh"
#include "kernel/moe/moe_mainloop_shared.cuh"
#include "kernel/moe/moe_mainloop_tail.cuh"
#include "kernel/moe/moe_scheduler.cuh"
#include "kernel/moe/moe_shared_storage.cuh"
#include "kernel/moe/moe_tail.cuh"
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
    using Tail = TailPhase<KernelTraits>;
    using TailGeometry = TailGeometryT<Dims>;

    // Routing tables are copied as 4-byte words.
    static constexpr std::uint32_t kRoutingTableWords =
        static_cast<std::uint32_t>(sizeof(MoERoutingTables<Dims>) /
                                   sizeof(std::uint32_t));
    static_assert(sizeof(MoERoutingTables<Dims>) % sizeof(std::uint32_t) == 0u,
                  "the routing tables travel as whole 4-B words");
    static_assert(Workspace::CtaComm::kRouterTableSlices ==
                      WarpRole::kThreadsPerWarp,
                  "one table slice per lane of the copying warp");

    // Runtime tensors and launch scalars.
    struct Arguments {
        std::uint32_t num_tokens;
        std::uint32_t hidden_size;
        std::uint32_t latent_size;
        std::uint32_t intermediate_size;
        const DataType::ElementA* input;
        const typename Traits::RouterGemm::MmaElement* router_weight;
        const typename Traits::RoutedUpGemm::MmaElement* expert_up_weights;
        const typename Traits::RoutedUpGemm::ElementScale*
            expert_up_weights_scale;
        const typename Traits::RoutedDownGemm::MmaElement* expert_down_weights;
        const typename Traits::RoutedDownGemm::ElementScale*
            expert_down_weights_scale;
        const typename Traits::SharedUpGemm::MmaElement* shared_up_weight;
        const typename Traits::SharedUpGemm::ElementScale*
            shared_up_weight_scale;
        const typename Traits::SharedDownGemm::MmaElement* shared_down_weight;
        const typename Traits::SharedDownGemm::ElementScale*
            shared_down_weight_scale;
        const typename Traits::LatentDownGemm::MmaElement* latent_down_weight;
        const typename Traits::LatentDownGemm::ElementScale*
            latent_down_weight_scale;
        const typename TailGeometry::ElementWeight* tail_fc2_w8;
        const typename TailGeometry::ElementScale* tail_fc2_scale;
        const DataType::ElementA* tail_norm_weight;
        std::uint8_t* tail_uc_base[TailGeometryT<Dims>::kWorld];
        std::uint8_t* tail_mc_base;
        std::uint32_t* tail_epochs;
        DataType::ElementOut* tail_out;
        void* workspace;
        std::uint64_t* timer_events;
        std::uint32_t top_k;
        std::uint32_t tail_rank;
        ScoringFunc scoring_func;
        bool renormalize;
        Activation activation;
        float situ_beta;
        float situ_linear_beta;
        std::uint32_t l2_cached_bytes;
        float tail_eps;
    };

    // Device-side launch parameters.
    struct Params {
        std::uint32_t num_tokens;
        std::uint32_t hidden_size;
        std::uint32_t latent_size;
        std::uint32_t intermediate_size;
        const DataType::ElementA* __restrict__ input;
        const typename Traits::RoutedUpGemm::
            MmaElement* __restrict__ expert_weights_up;
        const typename Traits::RoutedUpGemm::
            ElementScale* __restrict__ expert_scales_up;
        const typename Traits::RoutedDownGemm::
            MmaElement* __restrict__ expert_weights_down;
        const typename Traits::RoutedDownGemm::
            ElementScale* __restrict__ expert_scales_down;
        const typename Traits::LatentDownGemm::
            MmaElement* __restrict__ latent_down_weight;
        const typename Traits::LatentDownGemm::
            ElementScale* __restrict__ latent_down_weight_scale;
        const typename Traits::SharedUpGemm::
            MmaElement* __restrict__ shared_up_weight;
        const typename Traits::SharedUpGemm::
            ElementScale* __restrict__ shared_up_scale;
        const typename Traits::SharedDownGemm::
            MmaElement* __restrict__ shared_down_weight;
        const typename Traits::SharedDownGemm::
            ElementScale* __restrict__ shared_down_scale;
        const typename TailGeometry::ElementWeight* tail_fc2_w8;
        const typename TailGeometry::ElementScale* tail_fc2_scale;
        const DataType::ElementA* tail_norm_weight;
        std::uint8_t* tail_uc_base[TailGeometryT<Dims>::kWorld];
        std::uint8_t* tail_mc_base;
        std::uint32_t* tail_epochs;
        DataType::ElementOut* __restrict__ shared_out;
        DataType::ElementOut* tail_out;
        void* __restrict__ scratchpad;
        std::uint64_t* timer_events;
        std::uint32_t top_k;
        ScoringFunc scoring_func;
        bool renormalize;
        Activation activation;
        float situ_beta;
        float situ_linear_beta;
        // Evict weights first when their working set exceeds L2 capacity.
        std::uint32_t l2_expert_capacity;
        float tail_eps;
        std::uint32_t tail_shared_ctas;
        std::uint32_t tail_rank;
        alignas(128) CUtensorMap tma_up_weights_desc;
        alignas(128) CUtensorMap tma_input_desc;
        alignas(128) CUtensorMap tma_input_chunks_desc;
        // The router's gate-major [G, E, H] weight, in 128-B chunks.
        alignas(128) CUtensorMap tma_router_weights_desc;
        alignas(128) CUtensorMap tma_down_weights_desc;
        alignas(128) CUtensorMap tma_down_activations_desc;
        alignas(128) CUtensorMap tma_latent_down_weight_desc;
        alignas(128) CUtensorMap tma_shared_up_desc;
        alignas(128) CUtensorMap tma_shared_down_desc;
    };

    // Builds the device parameters, including the TMA descriptors.
    static Params to_underlying_arguments(Arguments const& args) {
        Params params{};
        params.num_tokens = args.num_tokens;
        params.hidden_size = args.hidden_size;
        params.latent_size = args.latent_size;
        params.intermediate_size = args.intermediate_size;
        params.input = args.input;
        params.expert_weights_up = args.expert_up_weights;
        params.expert_scales_up = args.expert_up_weights_scale;
        params.expert_weights_down = args.expert_down_weights;
        params.expert_scales_down = args.expert_down_weights_scale;
        params.latent_down_weight = args.latent_down_weight;
        params.latent_down_weight_scale = args.latent_down_weight_scale;
        params.shared_up_weight = args.shared_up_weight;
        params.shared_up_scale = args.shared_up_weight_scale;
        params.shared_down_weight = args.shared_down_weight;
        params.shared_down_scale = args.shared_down_weight_scale;
        params.tail_fc2_w8 = args.tail_fc2_w8;
        params.tail_fc2_scale = args.tail_fc2_scale;
        params.tail_norm_weight = args.tail_norm_weight;
        // The shared expert's partial output lives in the workspace.
        params.shared_out = reinterpret_cast<DataType::ElementOut*>(
            static_cast<char*>(args.workspace) +
            offsetof(Workspace, shared_down_out));
        params.tail_out = args.tail_out;
        params.scratchpad = args.workspace;
        params.timer_events = args.timer_events;
        for (int r = 0; r < TailGeometryT<Dims>::kWorld; ++r) {
            params.tail_uc_base[r] = args.tail_uc_base[r];
        }
        params.tail_mc_base = args.tail_mc_base;
        params.tail_epochs = args.tail_epochs;
        params.top_k = args.top_k;
        params.scoring_func = args.scoring_func;
        params.renormalize = args.renormalize;
        params.activation = args.activation;
        params.situ_beta = args.situ_beta;
        params.situ_linear_beta = args.situ_linear_beta;
        params.tail_eps = args.tail_eps;
        params.tail_shared_ctas = SharedDownGemmTraits<Dims>::kNumCtas;
        params.tail_rank = args.tail_rank;

        const std::uint64_t expert_width =
            args.latent_size > 0u ? args.latent_size : args.hidden_size;
        const std::uint64_t expert_weight_bytes =
            (2ull * args.intermediate_size * expert_width *
                 Traits::RoutedUpGemm::MmaCore::kElementABits +
             static_cast<std::uint64_t>(args.intermediate_size) * expert_width *
                 Traits::RoutedDownGemm::MmaCore::kElementABits) /
            8u;
        params.l2_expert_capacity =
            args.l2_cached_bytes > 0u && expert_weight_bytes > 0u
                ? static_cast<std::uint32_t>(args.l2_cached_bytes /
                                             expert_weight_bytes)
                : 0xFFFFFFFFu;

        constexpr std::uint64_t kInputBytes = sizeof(DataType::ElementA);
        constexpr std::uint32_t kChunkElems =
            128u / static_cast<std::uint32_t>(kInputBytes);
        params.tma_input_desc = CreateTmaDesc<2>(
            "activations", args.input,
            /*shape=*/{Dims::kHiddenSize, args.num_tokens},
            /*strides=*/{std::uint64_t{Dims::kHiddenSize} * kInputBytes},
            /*box=*/{64u, 8u});
        params.tma_input_chunks_desc = CreateTmaDesc<3>(
            "activations (K tiles)", args.input,
            /*shape=*/
            {kChunkElems, args.num_tokens, Dims::kHiddenSize / kChunkElems},
            /*strides=*/{std::uint64_t{Dims::kHiddenSize} * kInputBytes, 128u},
            /*box=*/{kChunkElems, 8u, 4u});
        params.tma_router_weights_desc = CreateTmaDesc<3>(
            "router weights", args.router_weight,
            /*shape=*/
            {kChunkElems, Dims::kNumRouterGates * Dims::kNumExperts,
             Dims::kHiddenSize / kChunkElems},
            /*strides=*/{std::uint64_t{Dims::kHiddenSize} * kInputBytes, 128u},
            /*box=*/
            {kChunkElems, Traits::RouterGemm::kTileShapeM,
             Traits::RouterGemm::MmaCore::kNumSwizzleBoxKs});
        if constexpr (Traits::RoutedUpGemm::kUsePairLayout) {
            constexpr std::uint64_t kUpBytes =
                std::uint64_t{MoEInputSize<Dims>::value} *
                cute::sizeof_bits_v<typename Traits::RoutedUpGemm::MmaElement> /
                8u;
            params.tma_up_weights_desc = CreateTmaDesc<2>(
                "up-projection weights", args.expert_up_weights,
                /*shape=*/
                {MoEInputSize<Dims>::value,
                 Dims::kNumExperts * 2u * Dims::kIntermediateSize},
                /*strides=*/{kUpBytes}, /*box=*/{128u, 128u});
        } else {
            // The natural [E, 2, N, K] weight: one box interleaves each
            // expert's gate/up 8-row groups.
            using UpGemm = typename Traits::RoutedUpGemm;
            constexpr std::uint64_t kElemBits =
                cute::sizeof_bits_v<typename UpGemm::MmaElement>;
            constexpr std::uint32_t kGran = UpGemm::kPairGran;
            constexpr std::uint32_t kK = MoEInputSize<Dims>::value;
            constexpr std::uint32_t kRows = Dims::kIntermediateSize;
            params.tma_up_weights_desc = CreateTmaDesc<5>(
                "up-projection weights", args.expert_up_weights,
                /*shape=*/{kK, kGran, 2u, kRows / kGran, Dims::kNumExperts},
                /*strides=*/
                {kK * kElemBits / 8, std::uint64_t{kRows} * kK * kElemBits / 8,
                 std::uint64_t{kGran} * kK * kElemBits / 8,
                 2ull * kRows * kK * kElemBits / 8},
                /*box=*/
                {UpGemm::MmaCore::kSwizzleBoxK, kGran, 2u,
                 UpGemm::kTileShapeM / (2u * kGran), 1u});
        }
        constexpr std::uint64_t kDownBytes =
            std::uint64_t{Dims::kIntermediateSize} *
            cute::sizeof_bits_v<typename Traits::RoutedDownGemm::MmaElement> /
            8u;
        params.tma_down_weights_desc = CreateTmaDesc<2>(
            "down-projection weights", args.expert_down_weights,
            /*shape=*/
            {Dims::kIntermediateSize,
             Dims::kNumExperts * MoEInputSize<Dims>::value},
            /*strides=*/{kDownBytes}, /*box=*/{128u, 128u});
        params.tma_down_activations_desc = CreateTmaDesc<2>(
            "down-projection activations",
            reinterpret_cast<const DataType::ElementQA*>(
                static_cast<const char*>(args.workspace) +
                Workspace::kUpOutFp8Offset),
            /*shape=*/{Dims::kIntermediateSize, Workspace::kMaxRoutedRows},
            /*strides=*/{std::uint64_t{Dims::kIntermediateSize}},
            /*box=*/{128u, Dims::kNumTokens});
        if constexpr (Traits::SharedUpGemm::kUsePairLayout) {
            params.tma_shared_up_desc = CreateTmaDesc<2>(
                "up-projection weights", args.shared_up_weight,
                /*shape=*/{Dims::kHiddenSize, Dims::kIntermediateSize * 2u},
                /*strides=*/{std::uint64_t{Dims::kHiddenSize}},
                /*box=*/{128u, Traits::SharedUpGemm::kTileShapeM});
        } else {
            // The natural [2, N, H] weight: the box interleaves the gate/up
            // 8-row groups.
            using UpGemm = typename Traits::SharedUpGemm;
            constexpr std::uint64_t kElemBytes =
                sizeof(typename UpGemm::MmaElement);
            constexpr std::uint32_t kGran = UpGemm::kPairGran;
            params.tma_shared_up_desc = CreateTmaDesc<4>(
                "up-projection weights", args.shared_up_weight,
                /*shape=*/
                {Dims::kHiddenSize, kGran, 2u, Dims::kIntermediateSize / kGran},
                /*strides=*/
                {std::uint64_t{Dims::kHiddenSize} * kElemBytes,
                 std::uint64_t{Dims::kIntermediateSize} * Dims::kHiddenSize *
                     kElemBytes,
                 std::uint64_t{kGran} * Dims::kHiddenSize * kElemBytes},
                /*box=*/
                {UpGemm::MmaCore::kSwizzleBoxK, kGran, 2u,
                 UpGemm::kTileShapeM / (2u * kGran)});
        }
        params.tma_shared_down_desc = CreateTmaDesc<2>(
            "down-projection weights", args.shared_down_weight,
            /*shape=*/{Dims::kIntermediateSize, Dims::kHiddenSize},
            /*strides=*/{std::uint64_t{Dims::kIntermediateSize}},
            /*box=*/{128u, Traits::SharedDownGemm::kTileShapeM});
        if (args.latent_down_weight != nullptr) {
            constexpr std::uint64_t kLatentBytes =
                std::uint64_t{Dims::kHiddenSize} *
                cute::sizeof_bits_v<
                    typename Traits::LatentDownGemm::MmaElement> /
                8u;
            params.tma_latent_down_weight_desc = CreateTmaDesc<2>(
                "latent-projection weights", args.latent_down_weight,
                /*shape=*/{Dims::kHiddenSize, Dims::kLatentSize},
                /*strides=*/{kLatentBytes},
                /*box=*/{128u, Traits::LatentDownGemm::kTileShapeM});
        }
        return params;
    }

    // Run latent projection on its assigned CTAs.
    __device__ void run_latent_down(Params const& params,
                                    SharedStorage* __restrict__ smem) {
        if constexpr (Traits::kUseLatent) {
            Latent::run(params.tma_latent_down_weight_desc,
                        params.tma_input_desc, params.latent_down_weight_scale,
                        reinterpret_cast<Workspace*>(params.scratchpad), smem,
                        static_cast<std::uint32_t>(params.num_tokens));
        }
    }

    // Run the shared expert on its assigned CTAs.
    __device__ void run_shared(Params const& params,
                               SharedStorage* __restrict__ smem) {
        using Shared = SharedUpProjectionBase<Traits>;
        using Up = SharedUpProjection<Traits>;
        using Down = SharedDownProjection<Traits>;

        if (!Shared::is_shared_cta()) return;

        auto* workspace = reinterpret_cast<Workspace*>(params.scratchpad);
        const std::uint32_t batch_size =
            static_cast<std::uint32_t>(params.num_tokens);
        if (Shared::cta_idx() < Shared::kNumUpCtas) {
            Up::run(params.tma_shared_up_desc, params.tma_input_desc,
                    params.shared_up_scale, workspace, smem, batch_size,
                    params.activation, params.situ_beta,
                    params.situ_linear_beta);
        }
        Down::run(params.tma_shared_down_desc, params.shared_down_scale,
                  params.shared_out, workspace, smem, batch_size);
        __syncthreads();
        if (threadIdx.x == 0u) {
            __threadfence();
            workspace->cta_comm.publish_shared_down_done(smem->launch_parity,
                                                         1u);
        }
    }

    // Compute router logits and build routing tables.
    __device__ static void run_router_topk(Params const& params,
                                           Workspace* __restrict__ workspace,
                                           SharedStorage* __restrict__ smem,
                                           std::uint32_t lane_idx) {
        constexpr std::uint32_t kExpertsPerThread = Dims::kNumExperts / 32u;

        // Use one warp per token; cluster pairs divide wider token batches.
        const bool coop_topk = Traits::kClusterSize >= 2u &&
                               params.num_tokens > WarpRole::kCalcWarpCount;
        if (threadIdx.x == 0u) {
            // One task per router CTA, whatever its M tile or K slice.
            const auto logits_span =
                KERNEL_TIMER_BEGIN(smem->timer.get(), "router_wait_logits");
            workspace->cta_comm.wait_router_tasks(smem->launch_parity,
                                                  Traits::RouterGemm::kNumCtas);
            KERNEL_TIMER_END(smem->timer.get(), logits_span);
        }
        WarpRole::sync_calc_threads();
        const std::uint32_t calc_warp_idx = WarpRole::get_calc_warp_idx();
        // Cluster pairs select one token half per CTA.
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
                    KERNEL_TIMER_BEGIN(smem->timer.get(), "topk_one_token");
                topk_one_token<Dims, kExpertsPerThread>(
                    tok, params.top_k, params.scoring_func, params.renormalize,
                    workspace, smem);
                KERNEL_TIMER_END(smem->timer.get(), topk_span);
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
                    KERNEL_TIMER_BEGIN(smem->timer.get(), "topk_one_token");
                topk_one_token<Dims, kExpertsPerThread>(
                    tok, params.top_k, params.scoring_func, params.renormalize,
                    workspace, smem);
                KERNEL_TIMER_END(smem->timer.get(), topk_span);
            }
        }
        WarpRole::sync_calc_threads();
        if (coop_topk) {
            // Exchange token halves through cluster shared memory.
            if (calc_warp_idx == 0u) {
                using Coop = decltype(smem->barriers.router_coop);
                const std::uint32_t peer_rank =
                    (blockIdx.x % Traits::kClusterSize) ^ 1u;
                if (lane_idx == 0u) {
                    const auto coop_span = KERNEL_TIMER_BEGIN(
                        smem->timer.get(), "router_coop_exchange");
                    smem->barriers.router_coop.arrive();
                    Coop::arrive_remote(
                        smem->barriers.router_coop.partner(peer_rank));
                    KERNEL_TIMER_END(smem->timer.get(), coop_span);
                }
                smem->barriers.router_coop.wait();
                // Copy the partner IDs and weights as 16-byte words.
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
            // The other calc warps read the swapped half too.
            WarpRole::sync_calc_threads();
        }
    }

    __device__ void run_router(Params const& params,
                               SharedStorage* __restrict__ smem) {
        Workspace* __restrict__ workspace =
            reinterpret_cast<Workspace*>(params.scratchpad);
        if (Router::is_router_cta()) {
            Router::run(params.tma_router_weights_desc,
                        params.tma_input_chunks_desc, workspace, smem,
                        params.scoring_func);
        }

        // Router CTAs build their tables; other CTAs copy the tables published
        // by CTA 0.
        const bool reads_published_tables = !Router::is_router_cta();
        const std::uint32_t lane_idx = WarpRole::get_lane_idx();
        if (!reads_published_tables) {
            if (WarpRole::is_calc_warp(WarpRole::get_warp_idx())) {
                run_router_topk(params, workspace, smem, lane_idx);
                // Each router CTA has the complete selection.
                const auto tables_span = KERNEL_TIMER_BEGIN(
                    smem->timer.get(), "prepare_routing_tables");
                prepare_routing_tables<Dims>(params.num_tokens, params.top_k,
                                             smem);
                KERNEL_TIMER_END(smem->timer.get(), tables_span);
                // Block 0 hands its finished tables to the other blocks.
                if (blockIdx.x == 0u && WarpRole::get_calc_warp_idx() == 0u) {
                    publish_routing_tables(workspace, smem, lane_idx);
                }
            }
        } else if (WarpRole::is_calc_warp(WarpRole::get_warp_idx()) &&
                   WarpRole::get_calc_warp_idx() == 0u) {
            // Copy routing tables after the latent projection finishes.
            const auto load_span =
                KERNEL_TIMER_BEGIN(smem->timer.get(), "load_routing_tables");
            load_routing_tables(workspace, smem, lane_idx);
            KERNEL_TIMER_END(smem->timer.get(), load_span);
        }
    }

    // CTA 0 publishes routing tables in lane-strided word slices.
    __device__ static __forceinline__ void publish_routing_tables(
        Workspace* __restrict__ workspace, SharedStorage* __restrict__ smem,
        std::uint32_t lane_idx) {
        constexpr std::uint32_t kPerLane =
            (kRoutingTableWords + WarpRole::kThreadsPerWarp - 1u) /
            WarpRole::kThreadsPerWarp;
        const std::uint32_t* src =
            reinterpret_cast<const std::uint32_t*>(&smem->routing);
        std::uint32_t* dst =
            reinterpret_cast<std::uint32_t*>(&workspace->router_tables);
        // Every lane's build stores land before the copy reads them.
        __syncwarp();
        std::uint32_t words[kPerLane];
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
        workspace->cta_comm.publish_router_table_slice(smem->launch_parity,
                                                       lane_idx);
    }

    // Each lane acquires and copies its routing-table slice before storing it.
    __device__ static __forceinline__ void load_routing_tables(
        Workspace const* __restrict__ workspace,
        SharedStorage* __restrict__ smem, std::uint32_t lane_idx) {
        constexpr std::uint32_t kPerLane =
            (kRoutingTableWords + WarpRole::kThreadsPerWarp - 1u) /
            WarpRole::kThreadsPerWarp;
        workspace->cta_comm.wait_router_table_slice(smem->launch_parity,
                                                    lane_idx);
        const std::uint32_t* src =
            reinterpret_cast<const std::uint32_t*>(&workspace->router_tables);
        std::uint32_t* dst = reinterpret_cast<std::uint32_t*>(&smem->routing);
        std::uint32_t words[kPerLane];
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

    // Copy quantized latent output and scales into routed-up staging.
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
        smem->barriers.routed.up_input_ready.arrive(0u,
                                                    kFp8Bytes + kScaleBytes);
        tma_load_1d(workspace->latent_down_out.data_ptr(), kFp8Bytes,
                    &smem->routed_experts.quantized_input.data[0][0][0][0],
                    smem->barriers.routed.up_input_ready.raw(0u));
        tma_load_1d(workspace->latent_down_out.scale_ptr(), kScaleBytes,
                    &smem->routed_experts.quantized_input.scale[0][0],
                    smem->barriers.routed.up_input_ready.raw(0u));
        smem->barriers.routed.up_input_ready.wait(0u);
    }

    // Phase 2: up-projection over this block's tasks, as enumerated by `sched`.
    __device__ void run_up(Params const& params,
                           SharedStorage* __restrict__ smem,
                           TaskScheduler<KernelTraits>& sched) {
        Workspace* __restrict__ workspace =
            reinterpret_cast<Workspace*>(params.scratchpad);
        if (WarpRole::is_other_loader_thread()) {
            if constexpr (Traits::kUseLatent) {
                const auto latent_span =
                    KERNEL_TIMER_BEGIN(smem->timer.get(), "up_wait_latent");
                workspace->cta_comm.wait_latent_tasks(
                    smem->launch_parity, Traits::LatentDownGemm::kNumCtas);
                // Order cross-CTA stores before TMA reads.
                fence_proxy_async_global();
                KERNEL_TIMER_END(smem->timer.get(), latent_span);
            }
            const auto activation_span =
                KERNEL_TIMER_BEGIN(smem->timer.get(), "up_load_activation");
            load_quantized_activation(workspace, smem);
            KERNEL_TIMER_END(smem->timer.get(), activation_span);
        }
        RoutedUpProjection<KernelTraits>::run(
            params.expert_scales_up, params.num_tokens, params.activation,
            params.situ_beta, params.situ_linear_beta, workspace, smem,
            params.tma_up_weights_desc, sched,
            get_tma_cache_hint(smem->routing.active_expert_count >
                               params.l2_expert_capacity));
    }

    // Run scheduled down-projection tasks.
    __device__ void run_down(Params const& params,
                             SharedStorage* __restrict__ smem,
                             TaskScheduler<KernelTraits>& sched) {
        RoutedDownProjection<KernelTraits>::run(
            params.expert_scales_down,
            reinterpret_cast<Workspace*>(params.scratchpad), smem,
            params.tma_down_weights_desc, params.tma_down_activations_desc,
            sched,
            get_tma_cache_hint(smem->routing.active_expert_count >
                               params.l2_expert_capacity));
    }

    // Prefetch the TMA descriptors.
    __device__ static void prefetch_tma_descriptors(Params const& params) {
        cute::prefetch_tma_descriptor(&params.tma_up_weights_desc);
        cute::prefetch_tma_descriptor(&params.tma_input_desc);
        cute::prefetch_tma_descriptor(&params.tma_input_chunks_desc);
        cute::prefetch_tma_descriptor(&params.tma_router_weights_desc);
        cute::prefetch_tma_descriptor(&params.tma_down_weights_desc);
        cute::prefetch_tma_descriptor(&params.tma_down_activations_desc);
    }

    // Initialize barriers, TMEM, descriptor cache and launch parity.
    __device__ void initialize(Params const& params,
                               SharedStorage* __restrict__ smem) {
        Workspace* __restrict__ workspace =
            reinterpret_cast<Workspace*>(params.scratchpad);
        const std::uint32_t warp_idx = WarpRole::get_warp_idx();
        const std::uint32_t lane_idx = WarpRole::get_lane_idx();

        if (warp_idx == 3 && lane_idx == 0) {
            // Per-CTA private launch counter.
            smem->launch_parity = workspace->cta_comm.claim_parity();
            // The down task queue starts empty.
            smem->down_tasks_claimed = 0u;
            smem->down_task_stores = 0u;
        } else if (WarpRole::is_tmem_warp(warp_idx)) {
            // TMEM allocation issued by ONE fully-active warp.
            tmem_alloc(Traits::kTmemColumns, &smem->tmem_ptr);
        } else if (warp_idx == 1 && lane_idx == 0) {
            // Only this block's prologue role's barriers; a router block's
            // loader warp initializes the router ring itself.
            if (Latent::is_latent_cta()) {
                smem->barriers.latent_down.init(
                    static_cast<int>(WarpRole::kCalcWarpCount));
            } else if (SharedUpProjectionBase<Traits>::is_shared_cta()) {
                smem->barriers.init_shared_ring();
            }
            cutlass::arch::fence_barrier_init();
        } else if (warp_idx == 2 && lane_idx == 0) {
            prefetch_tma_descriptors(params);
            // Every block runs the expert phases.
            smem->barriers.routed.init();
            cutlass::arch::fence_barrier_init();
        } else if (WarpRole::is_weight_loader_warp(warp_idx) &&
                   Router::is_router_cta()) {
            // The router is the prologue's critical path: start its loads now.
            Router::start_streams(params.tma_router_weights_desc,
                                  params.tma_input_chunks_desc, smem);
        }
        __syncthreads();

        // CTA 0 clears the next launch's readiness state.
        if (blockIdx.x == 0u && WarpRole::is_publish_warp(warp_idx)) {
            const std::uint32_t launch_parity = smem->launch_parity;
            for (std::uint32_t rank = lane_idx;
                 rank < Workspace::kMaxRoutedRows;
                 rank += WarpRole::kThreadsPerWarp) {
                workspace->cta_comm.reset_inactive(launch_parity, rank);
            }
        }
    }

    // Wait for completed m-blocks and combine their partial outputs.
    __device__ void combine(Params const& params,
                            SharedStorage* __restrict__ smem,
                            CtaGroup<Traits::kCtasPerExpert> const& cta_group) {
        Workspace* __restrict__ workspace =
            reinterpret_cast<Workspace*>(params.scratchpad);
        const std::uint32_t batch_size = params.num_tokens;
        constexpr std::uint32_t kDownOutputColsPerCta =
            Traits::RoutedDownGemm::MmaCore::kTileShapeM;
        // Groups partition the cells of m-block cta_idx_in_group.
        if (!cta_group.in_whole_group()) return;
        const std::uint32_t m_block = cta_group.cta_idx_in_group;
        const std::uint32_t base_col = m_block * kDownOutputColsPerCta;
        const std::uint32_t first_cell =
            cta_group.group_idx * blockDim.x + threadIdx.x;
        const std::uint32_t cell_stride = cta_group.num_groups * blockDim.x;
        const std::uint32_t num_cells = batch_size * kDownOutputColsPerCta;
        const std::uint32_t top_k = params.top_k;

        // Every lane acquires readiness before pushing its output to the tail.
        static_assert(kDownOutputColsPerCta % WarpRole::kThreadsPerWarp == 0u,
                      "a warp's combine cells must share one token");
        __syncwarp();
        for (std::uint32_t flat = first_cell; flat < num_cells;
             flat += cell_stride) {
            const std::uint32_t tok = flat / kDownOutputColsPerCta;
            const auto wait_span =
                KERNEL_TIMER_BEGIN(smem->timer.get(), "wait_down_out", tok);
            workspace->cta_comm.wait_down_out_field(
                smem->launch_parity, m_block,
                tok * PackedTokenCounts<Dims>::kBits,
                PackedTokenCounts<Dims>::kMask, top_k);
            KERNEL_TIMER_END(smem->timer.get(), wait_span);
            const auto gather_span = KERNEL_TIMER_BEGIN(
                smem->timer.get(), "gather_and_combine", tok);
            emit_cell(params, workspace, tok,
                      base_col + flat % kDownOutputColsPerCta, top_k,
                      smem->tail_parity);
            KERNEL_TIMER_END(smem->timer.get(), gather_span);
        }
    }

    __device__ __forceinline__ void prefetch_tail_weights(
        Params const& params) {
        Tail::prefetch_weights(params);
    }

    __device__ void run_tail(
        Params const& params, SharedStorage* __restrict__ smem,
        CtaGroup<Traits::kCtasPerExpert> const& cta_group) {
        Tail::run(params, reinterpret_cast<Workspace*>(params.scratchpad), smem,
                  cta_group);
    }

    // Sum a token's top-k output slots in rank order using FP32.
    __device__ static __forceinline__ float gather_value(
        Workspace* __restrict__ workspace, std::uint32_t tok, std::uint32_t col,
        std::uint32_t top_k) {
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
        return v;
    }

    __device__ __forceinline__ void emit_cell(Params const& params,
                                              Workspace* __restrict__ workspace,
                                              std::uint32_t tok,
                                              std::uint32_t col,
                                              std::uint32_t top_k,
                                              std::uint32_t tail_parity) {
        const DataType::ElementOut b = static_cast<DataType::ElementOut>(
            gather_value(workspace, tok, col, top_k));
        Tail::push_cell(params, tok, col, b, tail_parity);
    }
};

}  // namespace halops::moe
