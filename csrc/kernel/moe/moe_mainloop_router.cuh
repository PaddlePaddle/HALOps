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

    static constexpr std::uint32_t kTileShapeM = RouterGemm::kTileShapeM;
    // TMA byte count for the staged BF16 input tile.
    static constexpr std::uint32_t kInputTileTxBytes =
        RouterGemm::kTileShapeN * RouterGemm::kTileShapeK *
        static_cast<std::uint32_t>(sizeof(DataType::ElementA));

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

    // This block's first weight row in the gate-stacked [G * E, H] tensor.
    __device__ static std::uint32_t weight_row_begin() {
        return router_gate() * RouterGemm::kGemmM + router_tile() * kTileShapeM;
    }

    // One K tile is kNumSwizzleBoxKs whole 64-K SW128 boxes.
    static constexpr std::uint32_t kWeightTileChunks =
        MmaCore::kNumSwizzleBoxKs;

    // The router runs on the first kNumCtas blocks, one task each.
    __device__ static bool is_router_cta() {
        return blockIdx.x < RouterGemm::kNumCtas;
    }

    // Every slot's first tile, which can be in flight before any slot drains.
    static constexpr std::uint32_t kNumFirstWeightTiles =
        RouterGemm::kNumWeightSlots;
    static constexpr std::uint32_t kNumFirstInputTiles =
        RouterGemm::kNumInputSlots < RouterGemm::kNumTilesK
            ? RouterGemm::kNumInputSlots
            : RouterGemm::kNumTilesK;
    // Weight tiles that wait for an earlier tile to leave their slot.
    static constexpr std::uint32_t kNumTailWeightTiles =
        RouterGemm::kNumTilesK - kNumFirstWeightTiles;
    static_assert(kNumFirstWeightTiles <= WarpRole::kThreadsPerWarp,
                  "one loader lane refills each weight slot");

    // Arm a weight tile's slot, then load the tile into it.
    __device__ static void load_weight_tile(
        CUtensorMap const& tma_router_weights_desc,
        SharedStorage* __restrict__ smem, std::uint32_t j) {
        const std::uint32_t slot = j % RouterGemm::kNumWeightSlots;
        smem->barriers.router.weight_ready.arrive(j, MmaCore::kWeightTileBytes);
        tma_load_3d(tma_router_weights_desc, /*coord0=*/0u,
                    /*row=*/weight_row_begin(),
                    /*chunk=*/j * kWeightTileChunks,
                    &smem->router.w_router[slot][0],
                    smem->barriers.router.weight_ready.raw(slot));
    }

    // Arm an input tile's slot, then load the tile into it.
    __device__ static void load_input_tile(CUtensorMap const& tma_input_desc,
                                           SharedStorage* __restrict__ smem,
                                           std::uint32_t k_iter_id) {
        const std::uint32_t slot = k_iter_id % RouterGemm::kNumInputSlots;
        smem->barriers.router.input_ready.arrive(k_iter_id, kInputTileTxBytes);
        load_input_tile_chunks<RouterGemm::kTileShapeN>(
            tma_input_desc, k_iter_id, token_begin(),
            smem->router.input.data[slot],
            smem->barriers.router.input_ready.raw(slot));
    }

    // The weight-loader warp at launch: initialize the router barriers and
    // issue every weight slot's first tile, before the rest of the prologue
    // runs. The weights are parameters, so this may precede pdl_wait().
    __device__ static void prefetch_first_weight_tiles(
        CUtensorMap const& tma_router_weights_desc,
        SharedStorage* __restrict__ smem) {
        const std::uint32_t lane_idx = WarpRole::get_lane_idx();
        if (lane_idx == 0u) {
            smem->barriers.init_router_ring();
            cutlass::arch::fence_barrier_init();
        }
        __syncwarp();
        for (std::uint32_t j = lane_idx; j < kNumFirstWeightTiles;
             j += WarpRole::kThreadsPerWarp) {
            load_weight_tile(tma_router_weights_desc, smem, j);
        }
        if constexpr (kNumTailWeightTiles > 0u) {
            // The slot-waiting tiles then load from L2; both token windows of
            // an M tile read them, so one block prefetches.
            if (lane_idx == 0u && token_begin() == 0u) {
#pragma unroll
                for (std::uint32_t j = kNumFirstWeightTiles;
                     j < RouterGemm::kNumTilesK; ++j) {
                    tma_prefetch_l2_3d(tma_router_weights_desc, /*coord0=*/0u,
                                       /*row=*/weight_row_begin(),
                                       /*chunk=*/j * kWeightTileChunks);
                }
            }
        }
    }

    // The same warp once pdl_wait() returns: the input is the previous
    // kernel's output.
    __device__ static void prefetch_first_input_tiles(
        CUtensorMap const& tma_input_desc, SharedStorage* __restrict__ smem) {
        for (std::uint32_t k = WarpRole::get_lane_idx();
             k < kNumFirstInputTiles; k += WarpRole::kThreadsPerWarp) {
            load_input_tile(tma_input_desc, smem, k);
        }
    }

    // Stream the input K tiles past the first lap through the ring.
    __device__ static void run_load_input_k_loop(
        CUtensorMap const& tma_input_desc, SharedStorage* __restrict__ smem) {
#pragma unroll 1
        for (std::uint32_t k_iter_id = kNumFirstInputTiles;
             k_iter_id < RouterGemm::kNumTilesK; ++k_iter_id) {
            smem->barriers.router.wait_input_empty(k_iter_id);
            load_input_tile(tma_input_desc, smem, k_iter_id);
        }
    }

    // Lane l refills slot l, a lap at a time, as each of its tiles drains.
    __device__ static void run_load_weight_tail(
        CUtensorMap const& tma_router_weights_desc,
        SharedStorage* __restrict__ smem) {
        if constexpr (kNumTailWeightTiles > 0u) {
            const std::uint32_t lane_idx = WarpRole::get_lane_idx();
            std::uint32_t j = kNumFirstWeightTiles + lane_idx;
            bool pending =
                lane_idx < kNumFirstWeightTiles && j < RouterGemm::kNumTilesK;
            // Slot 0 drains first; poll the slots from then on.
            smem->barriers.router.weight_empty.wait(0u);
            while (__any_sync(0xffffffffu, pending)) {
                if (pending && smem->barriers.router.weight_empty.test(
                                   j - kNumFirstWeightTiles)) {
                    load_weight_tile(tma_router_weights_desc, smem, j);
                    j += kNumFirstWeightTiles;
                    pending = j < RouterGemm::kNumTilesK;
                }
            }
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
    __device__ static void run(CUtensorMap const& tma_router_weights_desc,
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

        // The rings' first lap left at launch; the whole warp loads the rest.
        if (WarpRole::is_weight_loader_warp(warp_idx)) {
            Base::run_load_weight_tail(tma_router_weights_desc, smem);
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
        constexpr std::uint32_t kInputSlots = RouterGemm::kNumInputSlots;
        for (std::uint32_t j = 0; j < RouterGemm::kNumTilesK; ++j) {
            const std::uint32_t k_iter_id = j;
            const std::uint32_t slot = k_iter_id % kInputSlots;
            smem->barriers.router.wait_mma_ready(k_iter_id);
            cute::Tensor tCrB = cta_mma.make_fragment_B(cute::make_tensor(
                cute::make_smem_ptr(
                    reinterpret_cast<typename MmaCore::TiledMma::ValTypeB*>(
                        &smem->router.input.data[slot][0][0][0])),
                SBLayout{}));
            MmaCore::run(
                tiled_mma, cta_mma,
                &smem->router.w_router[j % RouterGemm::kNumWeightSlots][0],
                tCrB, tCt, tmem_addr(tmem_base, 0u, 0u),
                /*accumulate=*/j != 0u);
            // Release the weight slot after UMMA completes.
            cutlass::arch::umma_arrive(
                smem->barriers.router.weight_empty.raw(j));
            // UMMA must finish reading the input before its slot is reused.
            cutlass::arch::umma_arrive(
                smem->barriers.router.input_empty.raw(k_iter_id));
        }
        // The commit covers every UMMA issued before it.
        cutlass::arch::umma_arrive(
            smem->barriers.router.accumulator_ready.raw(0u));
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
        // M=64 accumulators use the low 16 rows of each warp window.
        constexpr std::uint32_t kRowsPerRead = 16u;
        const std::uint32_t row_base = kRowsPerRead * warp_idx;
        if (row_base < kTileShapeM) {
            smem->barriers.router.accumulator_ready.wait(0u);
            tcgen05_fence_after_thread_sync();
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
    using typename Base::MmaCore;
    using typename Base::RouterGemm;
    using typename Base::SharedStorage;
    using typename Base::Workspace;

    // A warp owns this many K tiles; the last warps may own fewer.
    static constexpr std::uint32_t kNumTilesKPerWarp =
        (RouterGemm::kNumTilesK + WarpRole::kCalcWarpCount - 1u) /
        WarpRole::kCalcWarpCount;

    // The B (input) operand of one K tile: its N rows of SW128 boxes.
    using SBLayout = decltype(cute::tile_to_shape(
        cute::composition(
            cute::Swizzle<3, 3, 3>{},
            cute::Layout<
                cute::Shape<cute::_8,
                            cute::Int<RouterGemm::MmaCore::kSwizzleBoxK>>,
                cute::Stride<cute::Int<RouterGemm::MmaCore::kSwizzleBoxK>,
                             cute::_1>>{}),
        cute::Shape<cute::Int<RouterGemm::kTileShapeN>,
                    cute::Int<RouterGemm::kTileShapeK>>{}));

    // Every thread of the block: the loader streams, the calc warps consume.
    __device__ static void run(CUtensorMap const& tma_router_weights_desc,
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

        // The rings' first lap left at launch; the whole warp loads the rest.
        if (WarpRole::is_weight_loader_warp(WarpRole::get_warp_idx())) {
            Base::run_load_weight_tail(tma_router_weights_desc, smem);
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
        constexpr std::uint32_t kInputSlots = RouterGemm::kNumInputSlots;
#pragma unroll 1
        for (std::uint32_t j = 0; j < kNumTilesKPerWarp; ++j) {
            const std::uint32_t k_iter_id =
                j * WarpRole::kCalcWarpCount + warp_idx;
            if (k_iter_id >= RouterGemm::kNumTilesK) break;
            const std::uint32_t slot = k_iter_id % kInputSlots;
            smem->barriers.router.wait_mma_ready(k_iter_id);
            MmaCore::run(
                cute::make_tensor(
                    cute::make_smem_ptr(
                        &smem->router.w_router[k_iter_id %
                                               RouterGemm::kNumWeightSlots][0]),
                    typename MmaCore::WeightTileLayout{}),
                cute::make_tensor(cute::make_smem_ptr(
                                      &smem->router.input.data[slot][0][0][0]),
                                  SBLayout{}),
                acc);
            // Every lane fences operand reads before lane 0 releases the
            // slots.
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            if (WarpRole::get_lane_idx() == 0u) {
                smem->barriers.router.weight_empty.arrive(k_iter_id);
                smem->barriers.router.input_empty.arrive(k_iter_id);
            }
        }
        // Each lane holds rows local and local+8 of every atom.
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

    // Fence logits-buffer reads after the routing block barrier.
    __device__ static void release_weight_smem() {
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }
};

}  // namespace halops::moe
