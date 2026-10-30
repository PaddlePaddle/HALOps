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

// Phase 0: the router Linear, logits = input x weight.

#include <cuda_bf16.h>
#include <cutlass/arch/barrier.h>

#include <cstdint>
#include <cute/tensor.hpp>

#include "kernel/moe/mma_core.cuh"
#include "kernel/moe/moe_configs.cuh"
#include "kernel/moe/moe_routing.cuh"
#include "kernel/moe/moe_routing_topk.cuh"
#include "kernel/moe/moe_shared_storage.cuh"
#include "kernel/moe/moe_tma.cuh"
#include "kernel/moe/moe_traits.cuh"
#include "kernel/moe/moe_workspace.cuh"
#include "kernel/moe/ptx_utils.cuh"

namespace halops::moe {

template <typename KernelTraits>
struct RouterProjectionBase {
    using Traits = KernelTraits;
    using Dims = typename Traits::MoEDims;
    using RouterGemm = typename Traits::RouterGemm;
    using SharedStorage = MoESharedStorage<Dims>;
    using Workspace = MoEWorkspace<Dims>;
    using MmaCore = typename RouterGemm::MmaCore;
    using Element = typename RouterGemm::MmaElement;

    static constexpr std::uint32_t kTileShapeM = RouterGemm::kTileShapeM;

    // The gate this block scores; each gate owns a CTA range.
    __device__ static std::uint32_t router_gate() {
        return blockIdx.x / RouterGemm::kNumCtasForRouter;
    }

    // Block b's task within its gate: an (M tile, token window).
    __device__ static std::uint32_t router_tile() {
        return (blockIdx.x % RouterGemm::kNumCtasForRouter) /
               RouterGemm::kNumTilesN;
    }

    // First token of this block's window.
    __device__ static std::uint32_t token_begin() {
        return ((blockIdx.x % RouterGemm::kNumCtasForRouter) %
                RouterGemm::kNumTilesN) *
               RouterGemm::kTileShapeN;
    }

    // This block's gate weight: the packed blobs are gate-major.
    __device__ static const Element* gate_weight(
        const Element* __restrict__ router_weight) {
        return router_weight + static_cast<std::size_t>(router_gate()) *
                                   RouterGemm::kGemmM * RouterGemm::kGemmK;
    }

    // The router runs on the first kNumCtas blocks, one task each.
    __device__ static bool is_router_cta() {
        return blockIdx.x < RouterGemm::kNumCtas;
    }

    // Staging thread: the launch's bf16 input through the ring, one K tile
    // per slot.
    __device__ static void run_load_input_k_loop(
        CUtensorMap const& tma_input_desc, SharedStorage* __restrict__ smem) {
        constexpr std::uint32_t kInputTileBytes =
            RouterGemm::kNumSwizzleBoxKs * RouterGemm::kTileShapeN *
            RouterGemm::kSwizzleBoxK *
            static_cast<std::uint32_t>(sizeof(DataType::ElementA));

#pragma unroll 1
        for (std::uint32_t k_iter_id = 0; k_iter_id < RouterGemm::kNumTilesK;
             ++k_iter_id) {
            const std::uint32_t slot = k_iter_id % Traits::kRouterInputSlots;
            // The slot about to be overwritten must have been consumed.
            smem->barriers.wait_router_input_empty(k_iter_id);
            smem->barriers.router_input_ready.arrive(k_iter_id,
                                                     kInputTileBytes);
            load_input_tile<RouterGemm::kTileShapeN>(
                tma_input_desc, k_iter_id, token_begin(),
                smem->router.input.data[slot],
                smem->barriers.router_input_ready.raw(slot));
        }
    }

    // TMA launcher thread: the block's slice of its router M tile, one bulk
    // copy per K tile into ring slot `j % kNumRouterWeightSlots`.
    __device__ static void run_load_weight_k_loop(
        const Element* __restrict__ router_weight,
        SharedStorage* __restrict__ smem, std::uint32_t tile) {
#pragma unroll 1
        for (std::uint32_t j = 0; j < RouterGemm::kNumTilesK; ++j) {
            const std::uint32_t slot = j % KernelTraits::kNumRouterWeightSlots;
            if (j >= KernelTraits::kNumRouterWeightSlots) {
                smem->barriers.wait_router_weight_empty(j);
            }
            smem->barriers.router_weight_ready.arrive(
                j, RouterGemm::MmaCore::kWeightTileBytes);
            tma_load_router_weight_tile(
                router_weight, tile, j, RouterGemm::kNumTilesK,
                RouterGemm::MmaCore::kWeightTileBytes,
                &smem->router.w_router[slot][0],
                smem->barriers.router_weight_ready.raw(slot));
        }
    }

    // A score when a two-gate router publishes scores, else the bf16 logit.
    static constexpr bool kStoreScores =
        decltype(std::declval<Workspace>().router_logits)::kStoreScores;

    __device__ static float to_stored(float logit, ScoringFunc scoring) {
        if constexpr (kStoreScores) {
            if (scoring == ScoringFunc::kSigmoid) {
                return 1.0f / (1.0f + __expf(-logit));
            }
        }
        return logit;
    }

    __device__ static void store_logits(Workspace* __restrict__ workspace,
                                        std::uint32_t col,
                                        const float* __restrict__ v,
                                        ScoringFunc scoring) {
        const std::uint32_t t0 = token_begin();
        const std::uint32_t gate = router_gate();
#pragma unroll
        for (std::uint32_t t = 0; t < RouterGemm::kTileShapeN; ++t) {
            const float stored = to_stored(v[t], scoring);
            if constexpr (kStoreScores) {
                workspace->router_logits.row(gate, t0 + t)[col] = stored;
            } else {
                workspace->router_logits.row(gate, t0 + t)[col] =
                    __float2bfloat16_rn(stored);
            }
        }
    }
};

template <typename KernelTraits, bool UseUmma>
struct RouterProjection;

// UMMA on the MMA issuer warp; the calc warps stay free.
template <typename KernelTraits>
struct RouterProjection<KernelTraits, true>
    : RouterProjectionBase<KernelTraits> {
    using Base = RouterProjectionBase<KernelTraits>;
    using Base::kTileShapeM;
    using typename Base::Element;
    using typename Base::MmaCore;
    using typename Base::RouterGemm;
    using typename Base::SharedStorage;
    using typename Base::Workspace;

    // The B (input) operand of one K tile: its 8-token x 128-B SW128 boxes.
    using SBLayout = decltype(cute::UMMA::tile_to_mma_shape(
        cute::UMMA::Layout_K_SW128_Atom<typename RouterGemm::MmaElement>{},
        cute::partition_shape_B(
            typename MmaCore::TiledMma{},
            cute::make_shape(cute::Int<RouterGemm::kTileShapeN>{},
                             cute::Int<RouterGemm::kTileShapeK>{}))));

    // Every thread of the block: the loader streams, the issuer consumes.
    __device__ static void run(const Element* __restrict__ router_weight,
                               CUtensorMap const& tma_input_desc,
                               Workspace* __restrict__ workspace,
                               SharedStorage* __restrict__ smem,
                               ScoringFunc scoring) {
        const std::uint32_t warp_idx = WarpRole::get_warp_idx();
        if (!Base::is_router_cta()) {
            return;
        }

        // The router Linear's B operand, in both modes.
        if (WarpRole::is_other_loader_thread()) {
            run_load_input_k_loop(tma_input_desc, smem);
            return;
        }

        // ONE thread issues the ring's TMAs, each armed with its byte count.
        if (WarpRole::is_weight_loader_thread()) {
            run_load_weight_k_loop(Base::gate_weight(router_weight), smem,
                                   Base::router_tile());
            return;
        }
        if (WarpRole::is_mma_issuer_warp(warp_idx)) {
            run_mma_k_loop(smem, smem->tmem_ptr);
            return;
        }
        // The calc warps read their accumulators back and publish the logits.
        if (WarpRole::is_calc_warp(warp_idx)) {
            store_and_publish_logits(workspace, smem, Base::router_tile(),
                                     scoring);
        }
    }

    // One UMMA tile per SW128 box; warp-collective issue and commit.
    __device__ static void run_mma_k_loop(SharedStorage* __restrict__ smem,
                                          std::uint32_t tmem_base) {
        typename MmaCore::TiledMma tiled_mma{};
        auto cta_mma = tiled_mma.get_slice(cute::_0{});
        // Shape-only proxy: make_fragment_C only reads its argument's layout.
        cute::Tensor tC_proxy = cta_mma.partition_C(cute::make_tensor(
            cute::make_gmem_ptr(static_cast<float*>(nullptr)),
            cute::make_layout(
                cute::make_shape(cute::Int<MmaCore::kAtomShapeM>{},
                                 cute::Int<RouterGemm::kTileShapeN>{}),
                cute::GenRowMajor{})));
        cute::Tensor tCt = cta_mma.make_fragment_C(tC_proxy);

        // The streamed input holds one tile at a time; a block reads all K.
        constexpr std::uint32_t kInputSlots = KernelTraits::kRouterInputSlots;
        for (std::uint32_t j = 0; j < RouterGemm::kNumTilesK; ++j) {
            const std::uint32_t k_iter_id = j;
            const std::uint32_t slot = k_iter_id % kInputSlots;
            smem->barriers.router_input_ready.wait(k_iter_id);
            smem->barriers.router_weight_ready.wait(j);
            cute::Tensor tCrB = cta_mma.make_fragment_B(cute::make_tensor(
                cute::make_smem_ptr(
                    reinterpret_cast<typename MmaCore::TiledMma::ValTypeB*>(
                        &smem->router.input.data[slot][0][0][0])),
                SBLayout{}));
            MmaCore::run(
                tiled_mma, cta_mma,
                &smem->router
                     .w_router[j % KernelTraits::kNumRouterWeightSlots][0],
                tCrB, tCt, tmem_addr(tmem_base, 0u, 0u),
                /*accumulate=*/j != 0u);
            // The commit frees the slot kNumRouterWeightSlots K tiles later.
            cutlass::arch::umma_arrive(
                smem->barriers.router_weight_empty.raw(j));
            if (WarpRole::get_lane_idx() == 0u) {
                smem->barriers.arrive_router_input_empty(k_iter_id);
            }
        }
        // The commit covers every UMMA issued before it.
        cutlass::arch::umma_arrive(
            smem->barriers.router_accumulator_ready.raw(0u));
    }

    // Reads this block's M tile back, stores its logits and publishes them.
    __device__ static void store_and_publish_logits(
        Workspace* __restrict__ workspace, SharedStorage* __restrict__ smem,
        std::uint32_t tile, ScoringFunc scoring) {
        static_assert(RouterGemm::kTileShapeN % 8u == 0u,
                      "the token columns are whole 8-wide TMEM reads");
        static_assert(kTileShapeM <= 128u,
                      "the calc warps cover at most 8 lane blocks");
        const std::uint32_t warp_idx = WarpRole::get_warp_idx();
        const std::uint32_t lane_idx = WarpRole::get_lane_idx();
        // An M=64 UMMA accumulator keeps its 16-row groups in the low lanes
        // of each 32-lane window; warp w reads window w.
        constexpr std::uint32_t kRowsPerRead = 16u;
        const std::uint32_t row_base = kRowsPerRead * warp_idx;
        tcgen05_fence_after_thread_sync();
        if (row_base < kTileShapeM) {
            smem->barriers.router_accumulator_ready.wait(0u);
            float v[RouterGemm::kTileShapeN];
            tmem_ld_tokens(tmem_addr(smem->tmem_ptr, 32u * warp_idx, 0u), v);
            cutlass::arch::fence_view_async_tmem_load();
            // Only the tile's own rows are experts.
            if (lane_idx < kRowsPerRead && row_base + lane_idx < kTileShapeM) {
                store_logits(workspace,
                             tile * kTileShapeM + row_base + lane_idx, v,
                             scoring);
            }
        }
        // The up phase's UMMAs reuse the columns once every warp is done.
        tcgen05_fence_before_thread_sync();
        WarpRole::sync_calc_threads();
        tcgen05_fence_after_thread_sync();
        if (warp_idx == 0u && lane_idx == 0u) {
            workspace->cta_comm.publish_router_tasks(smem->launch_parity, 1u);
        }
    }

    // The weight tiles were read through the async proxy only: no fence.
    __device__ static void release_weight_smem() {}
};

// HMMA on the calc warps: warp w runs K tiles w, w + kCalcWarpCount, ...
template <typename KernelTraits>
struct RouterProjection<KernelTraits, false>
    : RouterProjectionBase<KernelTraits> {
    using Base = RouterProjectionBase<KernelTraits>;
    using Base::kTileShapeM;
    using typename Base::Element;
    using typename Base::MmaCore;
    using typename Base::RouterGemm;
    using typename Base::SharedStorage;
    using typename Base::Workspace;

    // The B (input) operand of one K tile: its N rows of SW128 boxes.
    using SBLayout = decltype(cute::tile_to_shape(
        cute::composition(
            cute::Swizzle<3, 3, 3>{},
            cute::Layout<
                cute::Shape<cute::_8, cute::Int<RouterGemm::kSwizzleBoxK>>,
                cute::Stride<cute::Int<RouterGemm::kSwizzleBoxK>, cute::_1>>{}),
        cute::Shape<cute::Int<RouterGemm::kTileShapeN>,
                    cute::Int<RouterGemm::kTileShapeK>>{}));

    // The HMMA reader's release of a weight slot: `ldmatrix` is synchronous,
    // so the reads have landed; every lane fences, one returns the slot.
    __device__ static void arrive_router_weight_empty(
        SharedStorage* __restrict__ smem, std::uint32_t k_tile) {
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        if (WarpRole::get_lane_idx() == 0u) {
            smem->barriers.router_weight_empty.arrive(k_tile);
        }
    }

    // Every thread of the block: the loader streams, the calc warps consume.
    __device__ static void run(const Element* __restrict__ router_weight,
                               CUtensorMap const& tma_input_desc,
                               Workspace* __restrict__ workspace,
                               SharedStorage* __restrict__ smem,
                               ScoringFunc scoring) {
        // Split-K is UMMA-only, so every block owns one whole M tile here.
        if (!Base::is_router_cta()) {
            return;
        }

        // The router Linear's B operand, in both modes.
        if (WarpRole::is_other_loader_thread()) {
            run_load_input_k_loop(tma_input_desc, smem);
            return;
        }

        if (WarpRole::is_weight_loader_thread()) {
            run_load_weight_k_loop(Base::gate_weight(router_weight), smem,
                                   Base::router_tile());
            return;
        }

        if (WarpRole::is_calc_thread()) {
            run_hmma_k_loop(smem);

            // The calc warps read their partials back and publish the logits.
            store_and_publish_logits(workspace, smem, Base::router_tile(),
                                     scoring);
        }
    }

    // Warp w sums its K tiles into mma_out[w], as (token, expert).
    __device__ static void run_hmma_k_loop(SharedStorage* __restrict__ smem) {
        const std::uint32_t warp_idx = WarpRole::get_calc_warp_idx();
        const std::uint32_t lane_idx = WarpRole::get_lane_idx();
        float acc[MmaCore::kNumAtomMs][MmaCore::kAtomElemsC] = {};
        constexpr std::uint32_t kInputSlots = KernelTraits::kRouterInputSlots;
#pragma unroll 1
        for (std::uint32_t j = 0; j < RouterGemm::kNumTilesKPerWarp; ++j) {
            const std::uint32_t k_iter_id =
                j * WarpRole::kCalcWarpCount + warp_idx;
            if (k_iter_id >= RouterGemm::kNumTilesK) break;
            const std::uint32_t slot = k_iter_id % kInputSlots;
            smem->barriers.router_input_ready.wait(k_iter_id);
            smem->barriers.router_weight_ready.wait(k_iter_id);
            MmaCore::run(
                cute::make_tensor(
                    cute::make_smem_ptr(
                        &smem->router
                             .w_router[k_iter_id %
                                       KernelTraits::kNumRouterWeightSlots][0]),
                    typename MmaCore::WeightTileLayout{}),
                cute::make_tensor(cute::make_smem_ptr(
                                      &smem->router.input.data[slot][0][0][0]),
                                  SBLayout{}),
                acc);
            // `ldmatrix` is synchronous: the K tile is read, the slot can
            // return to the ring.
            arrive_router_weight_empty(smem, k_iter_id);
            // This warp is the tile's only reader, so it frees the input
            // slot; the fence above already ordered both operands' reads.
            if (WarpRole::get_lane_idx() == 0u) {
                smem->barriers.arrive_router_input_empty(k_iter_id);
            }
        }
        // The m16n8 fragment: this lane holds rows local and local+8 of
        // every atom.
        const std::uint32_t local = lane_idx / 4u;
        const std::uint32_t col = 2u * (lane_idx & 3u);
        float* const cell = &smem->router.mma_out[warp_idx][0][0];
#pragma unroll
        for (std::uint32_t m = 0; m < MmaCore::kNumAtomMs; ++m) {
            const std::uint32_t row0 = m * MmaCore::kAtomShapeM + local;
            const std::uint32_t row1 = row0 + MmaCore::kAtomShapeM / 2u;
            cell[col * kTileShapeM + row0] = acc[m][0];
            cell[(col + 1u) * kTileShapeM + row0] = acc[m][1];
            cell[col * kTileShapeM + row1] = acc[m][2];
            cell[(col + 1u) * kTileShapeM + row1] = acc[m][3];
        }
    }

    // Reads this block's M tile back, stores its logits and publishes them.
    __device__ static void store_and_publish_logits(
        Workspace* __restrict__ workspace, SharedStorage* __restrict__ smem,
        std::uint32_t tile, ScoringFunc scoring) {
        // One thread per (token, 4 experts), summing the warps in order.
        WarpRole::sync_calc_threads();
        constexpr std::uint32_t kTokens = RouterGemm::kTileShapeN;
        constexpr std::uint32_t kQuads = kTileShapeM / 4u;
        static_assert(kTokens * kQuads <=
                          WarpRole::kCalcWarpCount * WarpRole::kThreadsPerWarp,
                      "the calc warps must cover every (token, quad) "
                      "cell");
        const std::uint32_t tok = threadIdx.x / kQuads % kTokens;
        const std::uint32_t q = threadIdx.x % kQuads;
        if (threadIdx.x < kTokens * kQuads) {
            float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
            for (std::uint32_t w = 0; w < WarpRole::kCalcWarpCount; ++w) {
                const float4 p = *reinterpret_cast<const float4*>(
                    &smem->router.mma_out[w][tok][4u * q]);
                sum.x += p.x;
                sum.y += p.y;
                sum.z += p.z;
                sum.w += p.w;
            }
            if constexpr (Base::kStoreScores) {
                *reinterpret_cast<float4*>(
                    workspace->router_logits.row(Base::router_gate(),
                                                 Base::token_begin() + tok) +
                    tile * kTileShapeM + 4u * q) =
                    make_float4(Base::to_stored(sum.x, scoring),
                                Base::to_stored(sum.y, scoring),
                                Base::to_stored(sum.z, scoring),
                                Base::to_stored(sum.w, scoring));
            } else {
                const __nv_bfloat162 lo = __floats2bfloat162_rn(sum.x, sum.y);
                const __nv_bfloat162 hi = __floats2bfloat162_rn(sum.z, sum.w);
                *reinterpret_cast<uint2*>(
                    workspace->router_logits.row(Base::router_gate(),
                                                 Base::token_begin() + tok) +
                    tile * kTileShapeM + 4u * q) =
                    make_uint2(*reinterpret_cast<const std::uint32_t*>(&lo),
                               *reinterpret_cast<const std::uint32_t*>(&hi));
            }
        }
        // Join the writers, then count the block's tiles in.
        WarpRole::sync_calc_threads();
        if (threadIdx.x == 0u) {
            workspace->cta_comm.publish_router_tasks(smem->launch_parity, 1u);
        }
    }

    // TMA launcher thread, after the block barrier that ends routing: the
    // tile reads are fenced per slot, this one covers the logits buffer.
    __device__ static void release_weight_smem() {
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }
};

}  // namespace halops::moe
