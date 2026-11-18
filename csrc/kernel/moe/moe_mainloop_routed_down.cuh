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

#include <cuda.h>
#include <cuda_fp8.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/array.h>

#include <cute/algorithm/gemm.hpp>
#include <cute/atom/mma_traits_sm100.hpp>
#include <cute/tensor.hpp>

#include "kernel/moe/mma_core.cuh"
#include "kernel/moe/moe_configs.cuh"
#include "kernel/moe/moe_cta_comm.cuh"
#include "kernel/moe/moe_mainloop_expert_base.cuh"
#include "kernel/moe/moe_scheduler.cuh"
#include "kernel/moe/moe_shared_storage.cuh"
#include "kernel/moe/moe_tma.cuh"
#include "kernel/moe/moe_traits.cuh"
#include "kernel/moe/moe_workspace.cuh"
#include "kernel/moe/ptx_utils.cuh"

namespace halops::moe {

template <typename KernelTraits>
struct RoutedDownProjectionBase
    : DownProjectionBase<KernelTraits, typename KernelTraits::RoutedDownGemm> {
    using Base =
        DownProjectionBase<KernelTraits, typename KernelTraits::RoutedDownGemm>;
    using typename Base::Accumulator;
    using typename Base::Dims;
    using typename Base::DownGemm;
    using typename Base::SharedStorage;
    using typename Base::Traits;
    using typename Base::Workspace;
    using RoutingTables = MoERoutingTables<Dims>;

    __device__ static constexpr std::uint32_t down_act_tx_bytes(
        std::uint32_t gemm_n) {
        return DownGemm::kTileShapeK * gemm_n *
               static_cast<std::uint32_t>(
                   sizeof(typename DownGemm::InputQuant::ElementType));
    }

    static constexpr std::uint64_t kBLbo = 16ULL;
    static constexpr std::uint64_t kBSbo = 128ULL;

    // The up phase's weight ring depth, which the down ring aliases at 2:1.
    static constexpr std::uint32_t kNumUpSlots =
        GemmSlots<Dims, typename Traits::RoutedUpGemm>::kNumSlots;

    static_assert(Traits::kUpSlotsPerDownSlot >= 1u,
                  "A w_down slot must cover at least one w_up slot: the first "
                  "lap waits the aliased up slots per down slot.");
    static_assert(Traits::kUpSlotsPerDownSlot * Base::kNumSlots <= kNumUpSlots,
                  "The down weight ring must alias whole groups of up weight "
                  "slots.");

    static_assert(DownGemm::kGemmN <= 16, "SM100 down-proj supports BS<=16.");

    // B uses SW128 slabs of tile-N rows per N substep.
    using SBLayout = decltype(cute::UMMA::tile_to_mma_shape(
        cute::UMMA::Layout_K_SW128_Atom<typename DownGemm::ActElement>{},
        cute::partition_shape_B(
            typename DownGemm::TiledMma{},
            cute::make_shape(cute::Int<DownGemm::kTileShapeN>{},
                             cute::Int<DownGemm::kTileShapeK>{}))));
    static_assert(
        cute::stride<0, 0>(cute::get_nonswizzle_portion(SBLayout{})) == kBSbo,
        "B: row/8-row-group stride must match kBSbo.");
    static_assert(cute::stride<0, 1>(cute::get_nonswizzle_portion(SBLayout{})) *
                          16 ==
                      kBLbo,
                  "B: K must be contiguous; 16 B = one LBO chunk.");
    static_assert(cute::stride<2>(cute::get_nonswizzle_portion(SBLayout{})) ==
                      32,
                  "B: K=32 step must be 32 B.");

    // Publisher lane 0 fences each task's stores and increments its completion
    // count.
    __device__ static void publish_down_tasks(
        MoEWorkspace<Dims>* __restrict__ workspace,
        MoESharedStorage<Dims>* __restrict__ smem,
        TaskScheduler<Traits>& sched) {
        TaskInfo task;
        while (sched.get_next_task(task)) {
            const std::uint64_t done_inc = PackedTokenCounts<Dims>::increment(
                smem->routing.token_mask(task.local_expert_idx));
            while (shared_load_acquire_u32(&smem->down_task_stores) <
                   WarpRole::kCalcWarpCount * (task.task_id + 1u)) {
            }
            workspace->cta_comm.publish_down_task(smem->launch_parity,
                                                  task.m_block_idx, done_inc);
        }
    }

    // Store each routed row in its combine slot, then notify the publisher.
    __device__ static void epilogue(Workspace* __restrict__ workspace,
                                    SharedStorage* __restrict__ smem,
                                    const Accumulator& acc_out,
                                    std::uint32_t base_m,
                                    std::uint32_t expert_start,
                                    std::uint32_t routed_count,
                                    unsigned m_tile_idx, unsigned lane_base) {
        const unsigned lane_idx = WarpRole::get_lane_idx();
        const std::uint32_t out_col =
            base_m + m_tile_idx * 128u + lane_base + lane_idx;
#pragma unroll
        for (std::uint32_t j = 0; j < DownGemm::kTileShapeN; ++j) {
            if (j >= routed_count) break;
            const std::uint32_t slot =
                smem->routing.expert_row_slot[expert_start + j];
            workspace->routed_down_out.row(slot)[out_col] =
                static_cast<DataType::ElementOut>(acc_out[j]);
        }
        __syncwarp();
        if (lane_idx == 0u) {
            shared_red_add_release_u32(&smem->down_task_stores, 1u);
        }
    }

    // Load one weight tile into an already-armed weight barrier.
    __device__ static void load_weight_tile(
        SharedStorage* __restrict__ smem,
        const CUtensorMap& tma_down_weights_desc, std::uint32_t expert_id,
        std::uint32_t base_m, std::uint32_t slot, std::uint32_t k_iter_id,
        std::uint32_t gemm_m, std::uint64_t weight_cache_hint) {
        Base::template load_weight_tile<DownGemm::kTileShapeM>(
            tma_down_weights_desc,
            reinterpret_cast<typename Base::WeightElement*>(
                &smem->routed_experts.w_down[slot][0][0]),
            smem->barriers.routed.down_weight_ready.raw(slot),
            expert_id * gemm_m + base_m, k_iter_id, weight_cache_hint);
    }

    // Arm and load one activation tile; an empty expert arms nothing.
    __device__ static void load_input_tile(
        SharedStorage* __restrict__ smem,
        const CUtensorMap& tma_down_activations_desc,
        std::uint32_t expert_start, std::uint32_t routed_count,
        std::uint32_t gemm_n, std::uint32_t step_local,
        std::uint32_t k_iter_id) {
        if (routed_count == 0u) return;
        const std::uint32_t slot = step_local % Base::kNumSlots;
        smem->barriers.routed.down_input_ready.arrive(
            slot, down_act_tx_bytes(gemm_n));
        tma_load_2d(tma_down_activations_desc, /*coord0=*/k_iter_id * 128u,
                    /*coord1=*/expert_start,
                    &smem->routed_experts.act_down_input[slot][0],
                    smem->barriers.routed.down_input_ready.raw(slot));
    }
};

template <typename KernelTraits,
          bool UseBlockscaled =
              KernelTraits::RoutedDownGemm::MmaCore::kUseBlockscaled>
struct RoutedDownProjection;

// MXFP8/MXFP4: scales are applied by the blockscaled MMA.
template <typename KernelTraits>
struct RoutedDownProjection<KernelTraits, true>
    : RoutedDownProjectionBase<KernelTraits> {
    using Base = RoutedDownProjectionBase<KernelTraits>;
    using typename Base::Accumulator;
    using typename Base::Dims;
    using typename Base::DownGemm;
    using typename Base::SharedStorage;
    using typename Base::Traits;
    using typename Base::Workspace;

    static_assert(DownGemm::MmaCore::kUseBlockscaled,
                  "RoutedDownProjection<true> requires a blockscaled MMA core");
    static_assert(Traits::RoutedUpGemm::MmaCore::kUseBlockscaled,
                  "blockscaled down continues blockscaled up's weight and "
                  "accumulator barriers");
    static_assert(RoutedBarrier<Dims>::kNumDownSlots == Base::kNumSlots,
                  "down input and weight slots share the MMA completion ring");
    // Each weight slot has an SFA/SFB set beside the fixed accumulator.
    using typename Base::DownScaleTraits;
    static constexpr std::uint32_t kScaleSlotColumns =
        DownScaleTraits::kSfaTmemColumns + DownScaleTraits::kSfbTmemColumns;
    static constexpr std::uint32_t kDownScaleTmemBase = Base::kAccColumns;
    static constexpr std::uint32_t kNumScaleSlots =
        GemmSlots<Dims, DownGemm>::kTmemScaleSlots;

    static constexpr std::uint32_t down_sfa_tmem(std::uint32_t step_local) {
        return kDownScaleTmemBase +
               (step_local % kNumScaleSlots) * kScaleSlotColumns;
    }

    static constexpr std::uint32_t down_sfb_tmem(std::uint32_t step_local) {
        return down_sfa_tmem(step_local) + DownScaleTraits::kSfaTmemColumns;
    }

    // Run down-projection over this block's dynamically claimed tasks.
    __device__ static void run(
        const typename DownGemm::ElementScale* __restrict__ expert_scales_down,
        Workspace* __restrict__ workspace, SharedStorage* __restrict__ smem,
        ScaleLayout down_scale_layout, CUtensorMap const& tma_down_weights_desc,
        CUtensorMap const& tma_down_scale_desc,
        CUtensorMap const& tma_down_activations_desc,
        TaskScheduler<Traits>& sched, MoEProblemSize const& problem_size,
        std::uint64_t weight_cache_hint) {
        const unsigned warp_idx = WarpRole::get_warp_idx();
        const unsigned lane_idx = WarpRole::get_lane_idx();
        const unsigned m_tile_idx = warp_idx / 4u;  // WG0 -> m-tile 0, WG1 -> 1
        const unsigned lane_base = (warp_idx & 3u) * 32u;
        const std::uint32_t gemm_m = problem_size.input_size();
        const std::uint32_t gemm_n = problem_size.num_tokens;
        const std::uint32_t gemm_k = problem_size.intermediate_size;
        // The blockscaled accumulator is shared with the preceding Up phase.
        const std::uint32_t acc_base = sched.up_task_count();
        const std::uint32_t up_k_tiles =
            gemm_m / Traits::RoutedUpGemm::kTileShapeK;
        const std::uint32_t num_k_tiles = gemm_k / DownGemm::kTileShapeK;
        const std::uint32_t num_up_tiles = acc_base * up_k_tiles;

        // All CTAs participate in the down phase.
        if (WarpRole::is_publish_warp(warp_idx)) {
            if (lane_idx == 0u) {
                Base::publish_down_tasks(workspace, smem, sched);
            }
            return;
        }
        // One thread loads weights; the input warp cooperates on scale loads.
        if (WarpRole::is_weight_loader_warp(warp_idx) && lane_idx != 0u) {
            return;
        }

        const std::uint32_t tmem_base = smem->tmem_ptr;

        TaskInfo task;
        while (true) {
            if (WarpRole::is_weight_loader_thread()) {
                // Claim the next task after issuing all current tiles; loads
                // still wait for free slots.
                sched.claim_next_task();
            }
            if (!sched.get_next_task(task)) {
                break;
            }

            const std::uint32_t task_id = task.task_id;
            const std::uint32_t acc_task_id = acc_base + task_id;
            const std::uint32_t local_id = task_id * num_k_tiles;
            const std::uint32_t e = task.local_expert_idx;
            const std::uint32_t base_m = task.m_block_idx * Base::kTileShapeM;
            const std::uint32_t expert_id = smem->routing.active_experts[e];

            const std::uint32_t routed_count = smem->routing.slab_count(e);
            const std::uint32_t expert_start = smem->routing.slab_begin(e);

            if (WarpRole::is_weight_loader_thread()) {
                run_load_weight_k_loop(smem, tma_down_weights_desc,
                                       tma_down_scale_desc, expert_scales_down,
                                       num_up_tiles, local_id, expert_id,
                                       base_m, down_scale_layout, num_k_tiles,
                                       gemm_m, weight_cache_hint);
            } else if (WarpRole::is_other_loader_warp(warp_idx)) {
                run_load_input_k_loop(
                    smem, workspace, tma_down_activations_desc, local_id,
                    expert_id, expert_start, routed_count, gemm_n, num_k_tiles);
            } else if (WarpRole::is_mma_issuer_warp(warp_idx)) {
                run_mma_k_loop(smem, tmem_base, task_id, acc_task_id,
                               routed_count, expert_scales_down, expert_id,
                               base_m, down_scale_layout, num_k_tiles, gemm_m);
            } else if (WarpRole::is_calc_warp(warp_idx)) {
                WarpRole::sync_calc_threads();

                Accumulator acc_out;
                load_accumulators(smem, acc_out, m_tile_idx, lane_base,
                                  tmem_base, acc_task_id);
                Base::epilogue(workspace, smem, acc_out, base_m, expert_start,
                               routed_count, m_tile_idx, lane_base);
            }
        }
    }

    // Stream weight tiles and their scale atoms through the ring.
    __device__ static void run_load_weight_k_loop(
        SharedStorage* __restrict__ smem,
        const CUtensorMap& tma_down_weights_desc,
        const CUtensorMap& tma_down_scale_desc,
        const typename DownGemm::ElementScale* __restrict__ down_scale,
        std::uint32_t num_up_tiles, std::uint32_t local_id,
        std::uint32_t expert_id, std::uint32_t base_m,
        ScaleLayout down_scale_layout, std::uint32_t num_k_tiles,
        std::uint32_t gemm_m, std::uint64_t weight_cache_hint) {
        const std::uint32_t tx_bytes =
            Base::down_weight_tx_bytes(down_scale_layout);
#pragma unroll 1
        for (std::uint32_t k_iter_id = 0; k_iter_id < num_k_tiles;
             ++k_iter_id) {
            const std::uint32_t slot = (local_id + k_iter_id) % Base::kNumSlots;
            wait_down_weight_empty(smem, num_up_tiles, local_id + k_iter_id);
            smem->barriers.routed.down_weight_ready.arrive(slot, tx_bytes);
            Base::load_weight_tile(smem, tma_down_weights_desc, expert_id,
                                   base_m, slot, k_iter_id, gemm_m,
                                   weight_cache_hint);
            load_weight_scale_tile(smem, down_scale, tma_down_scale_desc,
                                   expert_id, base_m, slot, k_iter_id,
                                   down_scale_layout, num_k_tiles, gemm_m);
        }
    }

    // Acquire up's output, then stream activation tiles and their scales.
    __device__ static void run_load_input_k_loop(
        SharedStorage* __restrict__ smem, Workspace* __restrict__ workspace,
        const CUtensorMap& tma_down_activations_desc, std::uint32_t local_id,
        std::uint32_t expert_id, std::uint32_t expert_start,
        std::uint32_t routed_count, std::uint32_t gemm_n,
        std::uint32_t num_k_tiles) {
        const unsigned lane_idx = WarpRole::get_lane_idx();
        workspace->cta_comm.wait_up_out(smem->launch_parity, expert_start,
                                        routed_count,
                                        Workspace::kUpOutReadyFull);
        // Order cross-CTA stores before TMA reads.
        if (lane_idx == 0u) {
            fence_proxy_async_global();
        }

#pragma unroll 1
        for (std::uint32_t k_iter_id = 0; k_iter_id < num_k_tiles;
             ++k_iter_id) {
            const std::uint32_t step_local = local_id + k_iter_id;
            wait_input_empty(smem, step_local);
            load_input_scale_tile(smem, workspace, expert_start, routed_count,
                                  step_local, k_iter_id);
            if (lane_idx == 0u) {
                Base::load_input_tile(smem, tma_down_activations_desc,
                                      expert_start, routed_count, gemm_n,
                                      step_local, k_iter_id);
            }
        }
    }

    // Accumulate all K tiles in TMEM and release each consumed input tile.
    __device__ static void run_mma_k_loop(
        SharedStorage* __restrict__ smem, std::uint32_t tmem_base,
        std::uint32_t task_id, std::uint32_t acc_task_id,
        std::uint32_t routed_count,
        const typename DownGemm::ElementScale* __restrict__ down_scale,
        std::uint32_t expert_id, std::uint32_t base_m,
        ScaleLayout down_scale_layout, std::uint32_t num_k_tiles,
        std::uint32_t gemm_m) {
        typename DownGemm::TiledMma tiled_mma{};
        auto cta_mma = tiled_mma.get_slice(cute::_0{});
        typename Base::SBLayout sB_layout{};
        auto tCt = Base::make_accumulator_fragment(cta_mma);

        const std::uint32_t local_id = task_id * num_k_tiles;
        if (acc_task_id > 0u) {
            smem->barriers.routed.accumulator_empty.wait(acc_task_id - 1u);
        }

#pragma unroll 1
        for (std::uint32_t k_iter_id = 0; k_iter_id < num_k_tiles;
             ++k_iter_id) {
            const std::uint32_t step_local = local_id + k_iter_id;
            const std::uint32_t w_slot = step_local % Base::kNumSlots;
            smem->barriers.routed.wait_down_mma_ready(step_local);
            if (down_scale_layout == ScaleLayout::kPlain) {
                Base::build_plain_scale(
                    smem->routed_experts.w_down_scale[w_slot],
                    reinterpret_cast<const std::uint8_t*>(down_scale),
                    expert_id * gemm_m + base_m, k_iter_id,
                    num_k_tiles * (DownGemm::kTileShapeK / 32u));
            }
            const auto sfa_tmem =
                tmem_addr(tmem_base, 0u, down_sfa_tmem(step_local));
            const auto sfb_tmem =
                tmem_addr(tmem_base, 0u, down_sfb_tmem(step_local));
            Base::mma(tiled_mma, cta_mma, sB_layout, tCt,
                      reinterpret_cast<typename Base::WeightElement*>(
                          &smem->routed_experts.w_down[w_slot][0][0]),
                      reinterpret_cast<typename Base::ActElement*>(
                          &smem->routed_experts.act_down_input[w_slot][0]),
                      tmem_addr(tmem_base, 0u, 0u),
                      smem->routed_experts.w_down_scale[w_slot],
                      smem->routed_experts.act_down_scale[w_slot], sfa_tmem,
                      sfb_tmem, /*accumulate=*/k_iter_id != 0u);
            cutlass::arch::umma_arrive(
                smem->barriers.routed.down_weight_empty.raw(w_slot));
        }
        cutlass::arch::umma_arrive(
            smem->barriers.routed.accumulator_ready.raw(acc_task_id));
    }

    // Read the completed task once, then release its fixed accumulator.
    __device__ static void load_accumulators(SharedStorage* __restrict__ smem,
                                             Accumulator& acc_out,
                                             unsigned m_tile_idx,
                                             unsigned lane_base,
                                             std::uint32_t tmem_base,
                                             std::uint32_t acc_task_id) {
        smem->barriers.routed.accumulator_ready.wait(acc_task_id);

        float values[1][DownGemm::kTileShapeN];
        Base::load_accumulator_fragment(tmem_base, 0u, m_tile_idx, lane_base,
                                        values);
        __syncwarp();
        if (WarpRole::get_lane_idx() == 0u) {
            smem->barriers.routed.accumulator_empty.arrive(acc_task_id);
        }
#pragma unroll
        for (std::uint32_t r = 0; r < DownGemm::kTileShapeN; ++r) {
            acc_out[r] = values[0][r];
        }
    }

    // Load the weight scale atoms into the tile's already-armed barrier.
    __device__ static void load_weight_scale_tile(
        SharedStorage* __restrict__ smem,
        const typename DownGemm::ElementScale* __restrict__ down_scale,
        const CUtensorMap& tma_down_scale_desc, std::uint32_t expert_id,
        std::uint32_t base_m, std::uint32_t slot, std::uint32_t k_iter_id,
        ScaleLayout down_scale_layout, std::uint32_t num_k_tiles,
        std::uint32_t gemm_m) {
        const std::uint32_t row_atom =
            expert_id * (gemm_m / 128u) + base_m / 128u;
        Base::load_weight_scale_tile(
            down_scale, row_atom, k_iter_id,
            smem->routed_experts.w_down_scale[slot],
            smem->barriers.routed.down_weight_ready.raw(slot),
            down_scale_layout, num_k_tiles, &tma_down_scale_desc);
    }

    // One lane per token loads its scale word and writes it in SfAtom layout.
    __device__ static void load_input_scale_tile(
        SharedStorage* __restrict__ smem, Workspace* __restrict__ workspace,
        std::uint32_t expert_start, std::uint32_t routed_count,
        std::uint32_t step_local, std::uint32_t k_iter_id) {
        if (routed_count == 0u) return;
        constexpr std::uint32_t kGroups = DownGemm::kTileShapeK / 32u;
        static_assert(
            kGroups == 4u && DownGemm::kTileShapeN <= WarpRole::kThreadsPerWarp,
            "one lane loads a token's K-tile scales as one word");
        const std::uint32_t row = WarpRole::get_lane_idx();
        if (row < DownGemm::kTileShapeN) {
            const std::uint32_t word =
                row < routed_count ? *reinterpret_cast<const std::uint32_t*>(
                                         workspace->routed_up_out.mx_scale_row(
                                             expert_start + row) +
                                         k_iter_id * kGroups)
                                   : 0u;
            const std::uint32_t slot = step_local % Base::kNumSlots;
            std::uint8_t* const sf = smem->routed_experts.act_down_scale[slot];
            *reinterpret_cast<std::uint32_t*>(sf + sf_atom_offset(row, 0u)) =
                word;
            // Each writer publishes its SMEM stores to the async proxy.
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        }
        // Lane 0 arms the input TMA only after every scale writer finishes.
        __syncwarp();
    }

    // Both operand slots are reusable once their previous K tile's MMA ends.
    __device__ static void wait_input_empty(SharedStorage* __restrict__ smem,
                                            std::uint32_t arm_id) {
        if (arm_id < Base::kNumSlots) return;
        smem->barriers.routed.down_weight_empty.wait(arm_id - Base::kNumSlots);
    }

    // A weight slot is free once its previous down MMA ends; on the first lap,
    // once the up MMAs that last read its aliased up slots end.
    __device__ static void wait_down_weight_empty(
        SharedStorage* __restrict__ smem, std::uint32_t num_up_tiles,
        std::uint32_t arm_id) {
        if (arm_id >= Base::kNumSlots) {
            smem->barriers.routed.down_weight_empty.wait(arm_id -
                                                         Base::kNumSlots);
            return;
        }
        if (num_up_tiles == 0u) return;
#pragma unroll
        for (std::uint32_t h = 0; h < Traits::kUpSlotsPerDownSlot; ++h) {
            const std::uint32_t up_slot =
                Traits::kUpSlotsPerDownSlot * arm_id + h;
            if (up_slot < num_up_tiles) {
                smem->barriers.routed.up_weight_empty.wait(
                    num_up_tiles - 1u -
                    (num_up_tiles - 1u - up_slot) % Base::kNumUpSlots);
            }
        }
        // The mbarrier wait does not order SMEM reads against TMA writes.
        asm volatile("fence.proxy.async;" ::: "memory");
    }
};

// Blockwise FP8: calc warps apply FP32 weight and activation scales.
template <typename KernelTraits>
struct RoutedDownProjection<KernelTraits, false>
    : RoutedDownProjectionBase<KernelTraits> {
    using Base = RoutedDownProjectionBase<KernelTraits>;
    using typename Base::Accumulator;
    using typename Base::Dims;
    using typename Base::DownGemm;
    using typename Base::SharedStorage;
    using typename Base::Traits;
    using typename Base::Workspace;

    static_assert(
        !DownGemm::MmaCore::kUseBlockscaled,
        "RoutedDownProjection<false> requires a non-blockscaled MMA core");
    static_assert(Base::kTileShapeM % DownGemm::WeightQuant::kBlockSize == 0u &&
                      128u % DownGemm::WeightQuant::kBlockSize == 0u,
                  "The down-proj m-tile base must be kQuantBlock-aligned for "
                  "the per-m-tile weight-scale load.");

    static constexpr std::uint32_t kTmemAccSlots =
        GemmSlots<Dims, DownGemm>::kTmemAccSlots;
    static_assert(
        Base::kNumSlots <= kTmemAccSlots,
        "The weight ring must not outrun the accumulator ring, or the "
        "re-arm's wait_accumulator_ready(ring_iter_id - Base::kNumSlots) "
        "can read a phase the ring has already wrapped past.");

    // One sub-accumulator per activation quantization block in each K tile.
    static constexpr std::uint32_t kBlocksPerTile =
        DownGemm::kTileShapeK / DownGemm::InputQuant::kBlockSize;
    static_assert(kBlocksPerTile == DownGemm::MmaCore::kNumQuantBlocksB,
                  "a tile's activation scale rows must match the dequant's "
                  "sub-accumulators");
    static_assert(kBlocksPerTile * DownGemm::kTileShapeN <=
                      WarpRole::kThreadsPerWarp,
                  "one warp stages a K tile's activation scale rows");
    // Run down-projection over this block's dynamically claimed tasks.
    __device__ static void run(
        const typename DownGemm::ElementScale* __restrict__ expert_scales_down,
        Workspace* __restrict__ workspace, SharedStorage* __restrict__ smem,
        [[maybe_unused]] ScaleLayout down_scale_layout,
        CUtensorMap const& tma_down_weights_desc,
        CUtensorMap const& tma_down_scale_desc,
        CUtensorMap const& tma_down_activations_desc,
        TaskScheduler<Traits>& sched, MoEProblemSize const& problem_size,
        std::uint64_t weight_cache_hint) {
        const unsigned warp_idx = WarpRole::get_warp_idx();
        const unsigned lane_idx = WarpRole::get_lane_idx();
        const unsigned m_tile_idx = warp_idx / 4u;  // WG0 -> m-tile 0, WG1 -> 1
        const unsigned lane_base = (warp_idx & 3u) * 32u;
        const std::uint32_t gemm_n = problem_size.num_tokens;

        // All CTAs participate in the down phase.
        if (WarpRole::is_publish_warp(warp_idx)) {
            if (lane_idx == 0u) {
                Base::publish_down_tasks(workspace, smem, sched);
            }
            return;
        }
        // The weight loader runs lane 0 only; the input warp stages the tile's
        // activation scales warp-wide.
        if (WarpRole::is_weight_loader_warp(warp_idx) && lane_idx != 0u) {
            return;
        }

        const std::uint32_t gemm_m = problem_size.input_size();
        const std::uint32_t gemm_k = problem_size.intermediate_size;
        const std::uint32_t num_k_tiles = gemm_k / DownGemm::kTileShapeK;
        const std::uint32_t up_k_tiles =
            gemm_m / Traits::RoutedUpGemm::kTileShapeK;
        // Blockwise up and down share an accumulator ring and continue its
        // iteration count.
        const std::uint32_t id_base =
            sched.up_task_count() *
            (Traits::RoutedUpGemm::MmaCore::kUseBlockscaled ? 0u : up_k_tiles);
        const std::uint32_t tmem_base = smem->tmem_ptr;

        TaskInfo task;
        while (true) {
            if (WarpRole::is_weight_loader_thread()) {
                // Claim the next task after the current task's first MMA reads
                // its weights.
                const std::uint32_t next = sched.next_task_id();
                if (next > 0u) {
                    const std::uint32_t next_local = next * num_k_tiles;
                    wait_down_weight_empty(smem, id_base, id_base + next_local,
                                           next_local % Base::kNumSlots);
                }
                sched.claim_next_task();
            }
            if (!sched.get_next_task(task)) {
                break;
            }

            const std::uint32_t task_id = task.task_id;
            const std::uint32_t local_id = task_id * num_k_tiles;
            const std::uint32_t ring_iter_id = id_base + local_id;
            const std::uint32_t e = task.local_expert_idx;
            const std::uint32_t base_m = task.m_block_idx * Base::kTileShapeM;
            const std::uint32_t expert_id = smem->routing.active_experts[e];

            const std::uint32_t routed_count = smem->routing.slab_count(e);
            const std::uint32_t expert_start = smem->routing.slab_begin(e);

            if (WarpRole::is_weight_loader_thread()) {
                run_load_weight_k_loop(smem, tma_down_weights_desc, id_base,
                                       local_id, expert_id, base_m,
                                       weight_cache_hint, num_k_tiles, gemm_m);
            } else if (WarpRole::is_other_loader_warp(warp_idx)) {
                run_load_input_k_loop(
                    smem, workspace, tma_down_activations_desc, id_base,
                    local_id, expert_start, routed_count, gemm_n, num_k_tiles);
            } else if (WarpRole::is_mma_issuer_warp(warp_idx)) {
                run_mma_k_loop(smem, tmem_base, ring_iter_id, local_id,
                               routed_count, num_k_tiles, gemm_m);
            } else if (WarpRole::is_calc_warp(warp_idx)) {
                Accumulator acc_out{};
                run_dequant_k_loop(smem, acc_out, expert_scales_down, expert_id,
                                   base_m, m_tile_idx, lane_base, tmem_base,
                                   ring_iter_id, local_id, num_k_tiles, gemm_m);
                Base::epilogue(workspace, smem, acc_out, base_m, expert_start,
                               routed_count, m_tile_idx, lane_base);
            }
        }
    }

    // This warp's m-tile weight scale for one K tile.
    __device__ static float weight_scale(
        const typename DownGemm::ElementScale* __restrict__ down_scale,
        std::uint32_t expert_id, std::uint32_t base_m, unsigned m_tile,
        std::uint32_t nb, std::uint32_t num_k_tiles, std::uint32_t gemm_m) {
        const std::uint32_t rb =
            (base_m + m_tile * DownGemm::MmaCore::kAtomShapeM) /
            DownGemm::WeightQuant::kBlockSize;
        const std::uint32_t scale_blocks_m =
            gemm_m / DownGemm::WeightQuant::kBlockSize;
        return down_scale[expert_id * scale_blocks_m * num_k_tiles +
                          rb * num_k_tiles + nb];
    }

    // Scale one K tile's sub-accumulators and accumulate into the output.
    __device__ static void dequant(SharedStorage* __restrict__ smem,
                                   std::uint32_t step_id, std::uint32_t slot,
                                   float ws, Accumulator& acc_out,
                                   unsigned m_tile_idx, unsigned lane_base,
                                   std::uint32_t tmem_base) {
        float values[DownGemm::MmaCore::kNumQuantBlocksB]
                    [DownGemm::kTileShapeN];
        smem->barriers.routed.accumulator_ready.wait(step_id);

        Base::load_accumulator_fragment(
            tmem_base, (step_id % kTmemAccSlots) * Base::kAccColumns,
            m_tile_idx, lane_base, values);
        Base::dequant(values, &smem->routed_experts.act_down_scale[slot][0], ws,
                      acc_out);
        // Every lane's read of the slot's scale rows has landed.
        __syncwarp();
        if (WarpRole::get_lane_idx() == 0u) {
            smem->barriers.routed.accumulator_empty.arrive(step_id);
        }
    }

    // Stream weight tiles through the ring.
    __device__ static void run_load_weight_k_loop(
        SharedStorage* __restrict__ smem,
        const CUtensorMap& tma_down_weights_desc, std::uint32_t id_base,
        std::uint32_t local_id, std::uint32_t expert_id, std::uint32_t base_m,
        std::uint64_t weight_cache_hint, std::uint32_t num_k_tiles,
        std::uint32_t gemm_m) {
        const std::uint32_t ring_iter_id = id_base + local_id;

#pragma unroll 1
        for (std::uint32_t k_iter_id = 0; k_iter_id < num_k_tiles;
             ++k_iter_id) {
            const std::uint32_t slot = (local_id + k_iter_id) % Base::kNumSlots;
            wait_down_weight_empty(smem, id_base, ring_iter_id + k_iter_id,
                                   slot);
            smem->barriers.routed.down_weight_ready.arrive(
                slot, DownGemm::MmaCore::kWeightTileTxBytes);
            Base::load_weight_tile(smem, tma_down_weights_desc, expert_id,
                                   base_m, slot, k_iter_id, gemm_m,
                                   weight_cache_hint);
        }
    }

    // Acquire up's output, then stage each tile's scales and activation tile.
    __device__ static void run_load_input_k_loop(
        SharedStorage* __restrict__ smem, Workspace* __restrict__ workspace,
        const CUtensorMap& tma_down_activations_desc, std::uint32_t id_base,
        std::uint32_t local_id, std::uint32_t expert_start,
        std::uint32_t routed_count, std::uint32_t gemm_n,
        std::uint32_t num_k_tiles) {
        const unsigned lane_idx = WarpRole::get_lane_idx();
        workspace->cta_comm.wait_up_out(smem->launch_parity, expert_start,
                                        routed_count,
                                        Workspace::kUpOutReadyFull);
        // Order cross-CTA stores before TMA reads.
        if (lane_idx == 0u) {
            fence_proxy_async_global();
        }

#pragma unroll 1
        for (std::uint32_t k_iter_id = 0; k_iter_id < num_k_tiles;
             ++k_iter_id) {
            const std::uint32_t step_local = local_id + k_iter_id;
            wait_input_empty(smem, id_base + step_local);
            load_input_scale_tile(smem, workspace, expert_start, routed_count,
                                  step_local, k_iter_id);
            if (lane_idx == 0u) {
                Base::load_input_tile(smem, tma_down_activations_desc,
                                      expert_start, routed_count, gemm_n,
                                      step_local, k_iter_id);
            }
        }
    }

    // Issue one K tile's unscaled sub-accumulators per ring slot.
    __device__ static void run_mma_k_loop(SharedStorage* __restrict__ smem,
                                          std::uint32_t tmem_base,
                                          std::uint32_t ring_iter_id,
                                          std::uint32_t local_id,
                                          std::uint32_t routed_count,
                                          std::uint32_t num_k_tiles,
                                          std::uint32_t gemm_m) {
        typename DownGemm::TiledMma tiled_mma{};
        auto cta_mma = tiled_mma.get_slice(cute::_0{});
        typename Base::SBLayout sB_layout{};
        auto tCt = Base::make_accumulator_fragment(cta_mma);

#pragma unroll 1
        for (std::uint32_t k_iter_id = 0; k_iter_id < num_k_tiles;
             ++k_iter_id) {
            const std::uint32_t step_id = ring_iter_id + k_iter_id;
            const std::uint32_t step_local = local_id + k_iter_id;
            const std::uint32_t acc_slot = step_id % kTmemAccSlots;
            const std::uint32_t w_slot = step_local % Base::kNumSlots;
            smem->barriers.routed.wait_down_mma_ready(step_local, step_id);
            Base::mma(tiled_mma, cta_mma, sB_layout, tCt,
                      reinterpret_cast<typename Base::WeightElement*>(
                          &smem->routed_experts.w_down[w_slot][0][0]),
                      reinterpret_cast<typename Base::ActElement*>(
                          &smem->routed_experts.act_down_input[w_slot][0]),
                      tmem_addr(tmem_base, 0u, acc_slot * Base::kAccColumns));
            // The commit covers every async tcgen05 op issued before it.
            cutlass::arch::umma_arrive(
                smem->barriers.routed.accumulator_ready.raw(acc_slot));
        }
    }

    // Read every K tile after its MMA, including the task's final tile.
    __device__ static void run_dequant_k_loop(
        SharedStorage* __restrict__ smem, Accumulator& acc_out,
        const typename DownGemm::ElementScale* __restrict__ down_scale,
        std::uint32_t expert_id, std::uint32_t base_m, unsigned m_tile_idx,
        unsigned lane_base, std::uint32_t tmem_base, std::uint32_t ring_iter_id,
        std::uint32_t local_id, std::uint32_t num_k_tiles,
        std::uint32_t gemm_m) {
#pragma unroll 1
        for (std::uint32_t k_iter_id = 1; k_iter_id <= num_k_tiles;
             ++k_iter_id) {
            const std::uint32_t nb = k_iter_id - 1u;
            // Issued before the wait, so it overlaps this tile's MMA.
            const float ws = weight_scale(down_scale, expert_id, base_m,
                                          m_tile_idx, nb, num_k_tiles, gemm_m);
            dequant(smem, ring_iter_id + nb, (local_id + nb) % Base::kNumSlots,
                    ws, acc_out, m_tile_idx, lane_base, tmem_base);
        }
    }

    // One lane per (row, K block) stages the tile's activation scale rows.
    __device__ static void load_input_scale_tile(
        SharedStorage* __restrict__ smem, Workspace* __restrict__ workspace,
        std::uint32_t expert_start, std::uint32_t routed_count,
        std::uint32_t step_local, std::uint32_t nb) {
        if (routed_count == 0u) return;
        const std::uint32_t lane = WarpRole::get_lane_idx();
        const std::uint32_t row = lane % DownGemm::kTileShapeN;
        const std::uint32_t sub = lane / DownGemm::kTileShapeN;
        if (sub < kBlocksPerTile) {
            const std::uint32_t slot = step_local % Base::kNumSlots;
            const std::uint32_t block = nb * kBlocksPerTile + sub;
            float scale = 0.f;
            if (row < routed_count) {
                const float* const rows =
                    workspace->routed_up_out.scale_row(expert_start + row);
                scale = rows[block];
            }
            smem->routed_experts.act_down_scale[slot][sub][row] = scale;
        }
        // Lane 0 arms the input TMA only after every scale writer finishes.
        __syncwarp();
    }

    // Hold an input slot until its MMA and its scale rows' readers finish.
    __device__ static void wait_input_empty(SharedStorage* __restrict__ smem,
                                            std::uint32_t arm_id) {
        constexpr std::uint32_t kNumSlots = RoutedBarrier<Dims>::kNumDownSlots;
        if (arm_id < kNumSlots) return;
        smem->barriers.routed.accumulator_ready.wait(arm_id - kNumSlots);
        smem->barriers.routed.accumulator_empty.wait(arm_id - kNumSlots);
    }

    // Wait for the previous down MMA or the first lap's aliased up MMAs.
    __device__ static void wait_down_weight_empty(
        SharedStorage* __restrict__ smem, std::uint32_t id_base,
        std::uint32_t ring_iter_id, std::uint32_t slot) {
        if (ring_iter_id - id_base >= Base::kNumSlots) {
            smem->barriers.routed.accumulator_ready.wait(ring_iter_id -
                                                         Base::kNumSlots);
            return;
        }
        // A block with no up task has no previous weight occupant.
        if (id_base == 0u) return;
#pragma unroll
        for (std::uint32_t h = 0; h < Traits::kUpSlotsPerDownSlot; ++h) {
            smem->barriers.routed.accumulator_ready.wait(
                id_base - Base::kNumUpSlots +
                Traits::kUpSlotsPerDownSlot * slot + h);
        }
        // The mbarrier wait does not order SMEM reads against TMA writes.
        asm volatile("fence.proxy.async;" ::: "memory");
    }
};

}  // namespace halops::moe
