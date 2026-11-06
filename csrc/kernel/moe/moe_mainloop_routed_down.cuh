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
    : DownProjectionBase<KernelTraits, typename KernelTraits::DownGemm> {
    using Base =
        DownProjectionBase<KernelTraits, typename KernelTraits::DownGemm>;
    using typename Base::Accumulator;
    using typename Base::Dims;
    using typename Base::DownGemm;
    using typename Base::SharedStorage;
    using typename Base::Traits;
    using typename Base::Workspace;
    using RoutingTables = MoERoutingTables<Dims>;

    static constexpr std::uint32_t kDownActTxBytes =
        DownGemm::kTileShapeN * DownGemm::kTileShapeK *
        static_cast<std::uint32_t>(
            sizeof(typename DownGemm::InputQuant::ElementType));
    static constexpr std::uint64_t kBLbo = 16ULL;
    static constexpr std::uint64_t kBSbo = 128ULL;

    // The up phase's weight ring depth, which the down ring aliases at 2:1.
    static constexpr std::uint32_t kNumUpWeightSlots =
        Traits::UpGemm::kNumWeightSlots;

    static_assert(Traits::kUpSlotsPerDownSlot >= 1u,
                  "A w_down slot must cover at least one w_up slot: the first "
                  "lap waits the aliased up slots per down slot.");
    static_assert(Traits::kUpSlotsPerDownSlot * Base::kNumWeightSlots <=
                      kNumUpWeightSlots,
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
        std::uint64_t weight_cache_hint) {
        Base::load_weight_tile(
            tma_down_weights_desc,
            reinterpret_cast<typename Base::WeightElement*>(
                &smem->routed_experts.w_down[slot][0][0]),
            smem->barriers.routed.down_weight_ready.raw(slot),
            expert_id * DownGemm::kGemmM + base_m, k_iter_id,
            weight_cache_hint);
    }

    // Arm and load one activation tile; an empty expert arms nothing.
    __device__ static void load_input_tile(
        SharedStorage* __restrict__ smem,
        const CUtensorMap& tma_down_activations_desc,
        std::uint32_t expert_start, std::uint32_t routed_count,
        std::uint32_t step_local, std::uint32_t k_iter_id) {
        if (routed_count == 0u) return;
        const std::uint32_t slot = step_local % Base::kNumWeightSlots;
        smem->barriers.routed.down_input_ready.arrive(slot, kDownActTxBytes);
        tma_load_2d(tma_down_activations_desc, /*coord0=*/k_iter_id * 128u,
                    /*coord1=*/expert_start,
                    &smem->routed_experts.act_down_input[slot][0],
                    smem->barriers.routed.down_input_ready.raw(slot));
    }
};

template <typename KernelTraits,
          bool UseBlockscaled =
              KernelTraits::DownGemm::MmaCore::kUseBlockscaled>
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
    static_assert(Traits::UpGemm::MmaCore::kUseBlockscaled,
                  "blockscaled down requires up to drain its weights and "
                  "accumulator before the down phase");
    static_assert(RoutedBarrier<Dims>::kNumDownWeightSlots ==
                      Base::kNumWeightSlots,
                  "down input and weight slots share the MMA completion ring");
    // Each weight slot has an SFA/SFB set beside the fixed accumulator.
    using typename Base::DownScaleTraits;
    static constexpr std::uint32_t kScaleSlotColumns =
        DownScaleTraits::kSfaTmemColumns + DownScaleTraits::kSfbTmemColumns;
    static constexpr std::uint32_t kDownScaleTmemBase = Base::kAccColumns;
    static constexpr std::uint32_t kNumScaleSlots = DownGemm::kWeightScaleSlots;

    static constexpr std::uint32_t down_sfa_tmem(std::uint32_t step_local) {
        return kDownScaleTmemBase +
               (step_local % kNumScaleSlots) * kScaleSlotColumns;
    }

    static constexpr std::uint32_t down_sfb_tmem(std::uint32_t step_local) {
        return down_sfa_tmem(step_local) + DownScaleTraits::kSfaTmemColumns;
    }

    using Base::kDownWeightTxBytes;

    // Run down-projection over this block's dynamically claimed tasks.
    __device__ static void run(
        const typename DownGemm::ElementScale* __restrict__ expert_scales_down,
        Workspace* __restrict__ workspace, SharedStorage* __restrict__ smem,
        CUtensorMap const& tma_down_weights_desc,
        CUtensorMap const& tma_down_activations_desc,
        TaskScheduler<Traits>& sched, std::uint64_t weight_cache_hint) {
        const unsigned warp_idx = WarpRole::get_warp_idx();
        const unsigned lane_idx = WarpRole::get_lane_idx();
        const unsigned m_tile_idx = warp_idx / 4u;  // WG0 -> m-tile 0, WG1 -> 1
        const unsigned lane_base = (warp_idx & 3u) * 32u;
        // The blockscaled accumulator is shared with the preceding Up phase.
        const std::uint32_t acc_base = sched.up_task_count();

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
            const std::uint32_t local_id = task_id * Base::kNumTilesK;
            const std::uint32_t e = task.local_expert_idx;
            const std::uint32_t base_m = task.m_block_idx * Base::kTileShapeM;
            const std::uint32_t expert_id = smem->routing.active_experts[e];

            const std::uint32_t routed_count = smem->routing.slab_count(e);
            const std::uint32_t expert_start = smem->routing.slab_begin(e);

            if (WarpRole::is_weight_loader_thread()) {
                run_load_weight_k_loop(smem, tma_down_weights_desc,
                                       expert_scales_down, local_id, expert_id,
                                       base_m, weight_cache_hint);
            } else if (WarpRole::is_other_loader_warp(warp_idx)) {
                run_load_input_k_loop(smem, workspace,
                                      tma_down_activations_desc, local_id,
                                      expert_id, expert_start, routed_count);
            } else if (WarpRole::is_mma_issuer_warp(warp_idx)) {
                run_mma_k_loop(smem, tmem_base, task_id, acc_task_id,
                               routed_count);
            } else if (WarpRole::is_calc_warp(warp_idx)) {
                WarpRole::sync_calc_threads();

                Accumulator acc_out;
                load_accumulators(smem, acc_out, m_tile_idx, lane_base,
                                  tmem_base, acc_task_id);

                // Write each slab pair's own combine slot, then hand off.
                Base::epilogue(workspace, smem, acc_out, base_m, expert_start,
                               routed_count, m_tile_idx, lane_base);
            }
        }
    }

    // Stream weight tiles and their scale atoms through the ring.
    __device__ static void run_load_weight_k_loop(
        SharedStorage* __restrict__ smem,
        const CUtensorMap& tma_down_weights_desc,
        const typename DownGemm::ElementScale* __restrict__ down_scale,
        std::uint32_t local_id, std::uint32_t expert_id, std::uint32_t base_m,
        std::uint64_t weight_cache_hint) {
#pragma unroll 1
        for (std::uint32_t k_iter_id = 0; k_iter_id < Base::kNumTilesK;
             ++k_iter_id) {
            const std::uint32_t slot =
                (local_id + k_iter_id) % Base::kNumWeightSlots;
            wait_down_weight_empty(smem, local_id + k_iter_id);
            smem->barriers.routed.down_weight_ready.arrive(slot,
                                                           kDownWeightTxBytes);
            Base::load_weight_tile(smem, tma_down_weights_desc, expert_id,
                                   base_m, slot, k_iter_id, weight_cache_hint);
            load_weight_scale_tile(smem, down_scale, expert_id, base_m, slot,
                                   k_iter_id);
        }
    }

    // Acquire up's output, then stream activation tiles and their scales.
    __device__ static void run_load_input_k_loop(
        SharedStorage* __restrict__ smem, Workspace* __restrict__ workspace,
        const CUtensorMap& tma_down_activations_desc, std::uint32_t local_id,
        std::uint32_t expert_id, std::uint32_t expert_start,
        std::uint32_t routed_count) {
        const unsigned lane_idx = WarpRole::get_lane_idx();
        // Every lane acquires before reading its token's scales from up.
        workspace->cta_comm.wait_up_out(smem->launch_parity, expert_start,
                                        routed_count,
                                        Workspace::kUpOutReadyFull);
        // Order cross-CTA stores before TMA reads.
        if (lane_idx == 0u) {
            fence_proxy_async_global();
        }

#pragma unroll 1
        for (std::uint32_t k_iter_id = 0; k_iter_id < Base::kNumTilesK;
             ++k_iter_id) {
            const std::uint32_t step_local = local_id + k_iter_id;
            wait_input_empty(smem, step_local);
            load_input_scale_tile(smem, workspace, expert_start, routed_count,
                                  step_local, k_iter_id);
            if (lane_idx == 0u) {
                Base::load_input_tile(smem, tma_down_activations_desc,
                                      expert_start, routed_count, step_local,
                                      k_iter_id);
            }
        }
    }

    // Accumulate all K tiles in TMEM and release each consumed input tile.
    __device__ static void run_mma_k_loop(SharedStorage* __restrict__ smem,
                                          std::uint32_t tmem_base,
                                          std::uint32_t task_id,
                                          std::uint32_t acc_task_id,
                                          std::uint32_t routed_count) {
        typename DownGemm::TiledMma tiled_mma{};
        auto cta_mma = tiled_mma.get_slice(cute::_0{});
        typename Base::SBLayout sB_layout{};
        auto tCt = Base::make_accumulator_fragment(cta_mma);

        const std::uint32_t local_id = task_id * Base::kNumTilesK;
        if (acc_task_id > 0u) {
            smem->barriers.routed.accumulator_empty.wait(acc_task_id - 1u);
        }

#pragma unroll 1
        for (std::uint32_t k_iter_id = 0; k_iter_id < Base::kNumTilesK;
             ++k_iter_id) {
            const std::uint32_t step_local = local_id + k_iter_id;
            const std::uint32_t w_slot = step_local % Base::kNumWeightSlots;
            smem->barriers.routed.wait_down_mma_ready(step_local);
            const auto sfa_tmem =
                tmem_addr(tmem_base, 0u, down_sfa_tmem(step_local));
            const auto sfb_tmem =
                tmem_addr(tmem_base, 0u, down_sfb_tmem(step_local));
            Base::copy_scales_to_tmem(
                smem->routed_experts.w_down_scale[w_slot],
                smem->routed_experts.act_down_scale[w_slot], sfa_tmem,
                sfb_tmem);
            Base::mma(tiled_mma, cta_mma, sB_layout, tCt,
                      reinterpret_cast<typename Base::WeightElement*>(
                          &smem->routed_experts.w_down[w_slot][0][0]),
                      reinterpret_cast<typename Base::ActElement*>(
                          &smem->routed_experts.act_down_input[w_slot][0]),
                      tmem_addr(tmem_base, 0u, 0u), sfa_tmem, sfb_tmem,
                      /*accumulate=*/k_iter_id != 0u);
            // The commit covers every async tcgen05 op issued before it.
            cutlass::arch::umma_arrive(
                smem->barriers.routed.down_weight_empty.raw(w_slot));
        }
        cutlass::arch::umma_arrive(
            smem->barriers.routed.accumulator_ready.raw(0u));
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
        std::uint32_t expert_id, std::uint32_t base_m, std::uint32_t slot,
        std::uint32_t k_iter_id) {
        const std::uint32_t row_atom =
            expert_id * (DownGemm::kGemmM / 128u) + base_m / 128u;
        Base::load_weight_scale_tile(
            down_scale, row_atom, k_iter_id,
            smem->routed_experts.w_down_scale[slot],
            smem->barriers.routed.down_weight_ready.raw(slot));
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
            const std::uint32_t slot = step_local % Base::kNumWeightSlots;
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
        if (arm_id < Base::kNumWeightSlots) return;
        smem->barriers.routed.down_weight_empty.wait(arm_id -
                                                     Base::kNumWeightSlots);
    }

    __device__ static void wait_down_weight_empty(
        SharedStorage* __restrict__ smem, std::uint32_t arm_id) {
        // Up drained the aliased weights before entering down's first lap.
        if (arm_id < Base::kNumWeightSlots) return;
        smem->barriers.routed.down_weight_empty.wait(arm_id -
                                                     Base::kNumWeightSlots);
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

    static constexpr std::uint32_t kNumAccSlots = DownGemm::kNumAccSlots;
    static_assert(
        Base::kNumWeightSlots <= kNumAccSlots,
        "The weight ring must not outrun the accumulator ring, or the "
        "re-arm's wait_accumulator_ready(ring_iter_id - Base::kNumWeightSlots) "
        "can read a phase the ring has already wrapped past.");

    // One sub-accumulator per activation quantization block in each K tile.
    static constexpr std::uint32_t kNumDownWeightScaleBlocksK =
        DownGemm::kGemmK / DownGemm::WeightQuant::kBlockSize;
    static constexpr std::uint32_t kNumDownWeightScaleBlocksM =
        DownGemm::kGemmM / DownGemm::WeightQuant::kBlockSize;

    // Run down-projection over this block's dynamically claimed tasks.
    __device__ static void run(
        const typename DownGemm::ElementScale* __restrict__ expert_scales_down,
        Workspace* __restrict__ workspace, SharedStorage* __restrict__ smem,
        CUtensorMap const& tma_down_weights_desc,
        CUtensorMap const& tma_down_activations_desc,
        TaskScheduler<Traits>& sched, std::uint64_t weight_cache_hint) {
        const unsigned warp_idx = WarpRole::get_warp_idx();
        const unsigned lane_idx = WarpRole::get_lane_idx();
        const unsigned m_tile_idx = warp_idx / 4u;  // WG0 -> m-tile 0, WG1 -> 1
        const unsigned lane_base = (warp_idx & 3u) * 32u;

        // All CTAs participate in the down phase.
        if (WarpRole::is_publish_warp(warp_idx)) {
            if (lane_idx == 0u) {
                Base::publish_down_tasks(workspace, smem, sched);
            }
            return;
        }
        // Only lane 0 of each loader warp participates.
        if ((WarpRole::is_weight_loader_warp(warp_idx) ||
             WarpRole::is_other_loader_warp(warp_idx)) &&
            lane_idx != 0u) {
            return;
        }

        // Blockwise up and down share an accumulator ring and continue its
        // iteration count.
        const std::uint32_t id_base =
            sched.up_task_count() * (Traits::UpGemm::MmaCore::kUseBlockscaled
                                         ? 0u
                                         : Traits::UpGemm::kNumTilesK);
        const std::uint32_t tmem_base = smem->tmem_ptr;

        TaskInfo task;
        while (true) {
            if (WarpRole::is_weight_loader_thread()) {
                // Claim the next task after the current task's first MMA reads
                // its weights.
                const std::uint32_t next = sched.next_task_id();
                if (next > 0u) {
                    const std::uint32_t next_local = next * Base::kNumTilesK;
                    wait_down_weight_empty(smem, id_base, id_base + next_local,
                                           next_local % Base::kNumWeightSlots);
                }
                sched.claim_next_task();
            }
            if (!sched.get_next_task(task)) {
                break;
            }

            const std::uint32_t task_id = task.task_id;
            const std::uint32_t local_id = task_id * Base::kNumTilesK;
            const std::uint32_t ring_iter_id = id_base + local_id;
            const std::uint32_t e = task.local_expert_idx;
            const std::uint32_t base_m = task.m_block_idx * Base::kTileShapeM;
            const std::uint32_t expert_id = smem->routing.active_experts[e];

            const std::uint32_t routed_count = smem->routing.slab_count(e);
            const std::uint32_t expert_start = smem->routing.slab_begin(e);

            if (WarpRole::is_weight_loader_thread()) {
                run_load_weight_k_loop(smem, tma_down_weights_desc, id_base,
                                       local_id, expert_id, base_m,
                                       weight_cache_hint);
            } else if (WarpRole::is_other_loader_warp(warp_idx)) {
                run_load_input_k_loop(
                    smem, workspace, tma_down_activations_desc, id_base,
                    local_id, expert_id, expert_start, routed_count);
            } else if (WarpRole::is_mma_issuer_warp(warp_idx)) {
                run_mma_k_loop(smem, tmem_base, ring_iter_id, local_id,
                               routed_count);
            } else if (WarpRole::is_calc_warp(warp_idx)) {
                WarpRole::sync_calc_threads();
                load_weight_and_act_scale(smem, workspace, expert_scales_down,
                                          expert_id, base_m, expert_start,
                                          routed_count);

                Accumulator acc_out{};
                run_dequant_k_loop(smem, acc_out, m_tile_idx, lane_base,
                                   tmem_base, ring_iter_id);

                // Write each slab pair's own combine slot, then hand off.
                Base::epilogue(workspace, smem, acc_out, base_m, expert_start,
                               routed_count, m_tile_idx, lane_base);
            }
        }
    }

    // Warp 0 loads weight scales; warp 1 acquires up's output and its scales.
    __device__ static void load_weight_and_act_scale(
        SharedStorage* __restrict__ smem, Workspace* __restrict__ workspace,
        const typename DownGemm::ElementScale* __restrict__ expert_scales_down,
        std::uint32_t expert_id, std::uint32_t base_m,
        std::uint32_t expert_start, std::uint32_t routed_count) {
        const unsigned warp_idx = WarpRole::get_warp_idx();
        if (warp_idx == 0u) {
            load_weight_scale(expert_scales_down, expert_id, base_m,
                              smem->routed_experts.w_down_scale[0]);
        } else if (warp_idx == 1u) {
            // Every lane acquires: each lane loads up_out_scale itself.
            workspace->cta_comm.wait_up_out(smem->launch_parity, expert_start,
                                            routed_count,
                                            Workspace::kUpOutReadyFull);
            load_down_input_scale(workspace, expert_start, routed_count,
                                  smem->routed_experts.act_down_scale);
        }
    }

    // Scale one K tile's sub-accumulators and accumulate into the output.
    __device__ static void dequant(SharedStorage* __restrict__ smem,
                                   std::uint32_t ring_iter_id,
                                   Accumulator& acc_out, unsigned m_tile_idx,
                                   unsigned lane_base, std::uint32_t k_iter_id,
                                   std::uint32_t tmem_base) {
        float values[DownGemm::MmaCore::kNumQuantBlocksB]
                    [DownGemm::kTileShapeN];
        const std::uint32_t step_id = ring_iter_id - 1u;
        smem->barriers.routed.accumulator_ready.wait(step_id);

        Base::load_accumulator_fragment(
            tmem_base, (step_id % kNumAccSlots) * Base::kAccColumns, m_tile_idx,
            lane_base, values);
        // This also completes the weight and input scale loads.
        WarpRole::sync_calc_threads();
        if (WarpRole::get_lane_idx() == 0u) {
            smem->barriers.routed.accumulator_empty.arrive(step_id);
        }

        const std::uint32_t nb = k_iter_id - 1u;
        const std::uint32_t a_blk =
            (nb * 128u) / Traits::UpGemm::kOutputQuantBlock;
        const std::uint32_t scale_idx =
            m_tile_idx * kNumDownWeightScaleBlocksK +
            ((kNumDownWeightScaleBlocksK > 1u) ? nb : 0u);
        const float ws = smem->routed_experts.w_down_scale[0][scale_idx];
        Base::dequant(values, &smem->routed_experts.act_down_scale[a_blk], ws,
                      acc_out);
    }

    // Stream weight tiles through the ring.
    __device__ static void run_load_weight_k_loop(
        SharedStorage* __restrict__ smem,
        const CUtensorMap& tma_down_weights_desc, std::uint32_t id_base,
        std::uint32_t local_id, std::uint32_t expert_id, std::uint32_t base_m,
        std::uint64_t weight_cache_hint) {
        const std::uint32_t ring_iter_id = id_base + local_id;

#pragma unroll 1
        for (std::uint32_t k_iter_id = 0; k_iter_id < Base::kNumTilesK;
             ++k_iter_id) {
            const std::uint32_t slot =
                (local_id + k_iter_id) % Base::kNumWeightSlots;
            wait_down_weight_empty(smem, id_base, ring_iter_id + k_iter_id,
                                   slot);
            smem->barriers.routed.down_weight_ready.arrive(
                slot, DownGemm::MmaCore::kWeightTileTxBytes);
            Base::load_weight_tile(smem, tma_down_weights_desc, expert_id,
                                   base_m, slot, k_iter_id, weight_cache_hint);
        }
    }

    // Acquire up's output, then stream its activation tiles.
    __device__ static void run_load_input_k_loop(
        SharedStorage* __restrict__ smem, Workspace* __restrict__ workspace,
        const CUtensorMap& tma_down_activations_desc, std::uint32_t id_base,
        std::uint32_t local_id, std::uint32_t expert_id,
        std::uint32_t expert_start, std::uint32_t routed_count) {
        // Acquire the routed_up_out readiness bitmask.
        workspace->cta_comm.wait_up_out(smem->launch_parity, expert_start,
                                        routed_count,
                                        Workspace::kUpOutReadyFull);
        // Order cross-CTA stores before TMA reads.
        fence_proxy_async_global();

#pragma unroll 1
        for (std::uint32_t k_iter_id = 0; k_iter_id < Base::kNumTilesK;
             ++k_iter_id) {
            wait_input_empty(smem, id_base + local_id + k_iter_id);
            Base::load_input_tile(smem, tma_down_activations_desc, expert_start,
                                  routed_count, local_id + k_iter_id,
                                  k_iter_id);
        }
    }

    // Issue one K tile's unscaled sub-accumulators per ring slot.
    __device__ static void run_mma_k_loop(SharedStorage* __restrict__ smem,
                                          std::uint32_t tmem_base,
                                          std::uint32_t ring_iter_id,
                                          std::uint32_t local_id,
                                          std::uint32_t routed_count) {
        typename DownGemm::TiledMma tiled_mma{};
        auto cta_mma = tiled_mma.get_slice(cute::_0{});
        typename Base::SBLayout sB_layout{};
        auto tCt = Base::make_accumulator_fragment(cta_mma);

#pragma unroll 1
        for (std::uint32_t k_iter_id = 0; k_iter_id < Base::kNumTilesK;
             ++k_iter_id) {
            const std::uint32_t step_id = ring_iter_id + k_iter_id;
            const std::uint32_t step_local = local_id + k_iter_id;
            const std::uint32_t acc_slot = step_id % kNumAccSlots;
            const std::uint32_t w_slot = step_local % Base::kNumWeightSlots;
            // The previous slot occupant may be an up tile or an earlier down
            // task.
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
    __device__ static void run_dequant_k_loop(SharedStorage* __restrict__ smem,
                                              Accumulator& acc_out,
                                              unsigned m_tile_idx,
                                              unsigned lane_base,
                                              std::uint32_t tmem_base,
                                              std::uint32_t ring_iter_id) {
#pragma unroll 1
        for (std::uint32_t k_iter_id = 1; k_iter_id <= Base::kNumTilesK;
             ++k_iter_id) {
            dequant(smem, ring_iter_id + k_iter_id, acc_out, m_tile_idx,
                    lane_base, k_iter_id, tmem_base);
        }
    }

    // Load the expert activation-scale table.
    __device__ static void load_down_input_scale(
        const Workspace* __restrict__ workspace, std::uint32_t expert_start,
        std::uint32_t routed_count,
        ActivationScaleStorage<DownGemm>& dest_scale) {
        constexpr unsigned kScaleKBlocks =
            ActivationScaleStorage<DownGemm>::kNumScaleBlocksK;
        const unsigned lane_idx = WarpRole::get_lane_idx();

        const float* __restrict__ src =
            workspace->routed_up_out.scale.blockwise;
        constexpr unsigned kTotal = DownGemm::kTileShapeN * kScaleKBlocks;
        for (unsigned i = lane_idx; i < kTotal; i += 32u) {
            const unsigned slot_row = i / kScaleKBlocks;
            const unsigned block = i % kScaleKBlocks;
            if (slot_row < routed_count) {
                const uint32_t source_row = expert_start + slot_row;
                dest_scale[block][slot_row] =
                    src[source_row * kScaleKBlocks + block];
            } else {
                dest_scale[block][slot_row] = 0.f;
            }
        }
    }

    // Per-expert weight-scale loader: one lane per (m-tile, K-block).
    __device__ static void load_weight_scale(
        const float* __restrict__ expert_scales_down, std::uint32_t expert_id,
        std::uint32_t base_m,
        float (&dest)[DownGemm::MmaCore::kNumAtomMs *
                      kNumDownWeightScaleBlocksK]) {
        constexpr uint32_t kNumScaleBlocksM = kNumDownWeightScaleBlocksM;
        constexpr uint32_t kNumScaleBlocksK = kNumDownWeightScaleBlocksK;
        constexpr uint32_t kTotal =
            DownGemm::MmaCore::kNumAtomMs * kNumScaleBlocksK;
        static_assert(kTotal <= WarpRole::kThreadsPerWarp,
                      "one lane per (m-tile, K-block) weight scale");

        const unsigned lane_idx = WarpRole::get_lane_idx();

        if (lane_idx < kTotal) {
            const unsigned m_tile = lane_idx / kNumScaleBlocksK;
            const unsigned k_block = lane_idx % kNumScaleBlocksK;
            const uint32_t rb =
                (base_m + m_tile * DownGemm::MmaCore::kAtomShapeM) /
                DownGemm::WeightQuant::kBlockSize;
            dest[lane_idx] =
                expert_scales_down[expert_id * kNumScaleBlocksM *
                                       kNumScaleBlocksK +
                                   rb * kNumScaleBlocksK + k_block];
        }
    }

    // Hold an input slot until its previous MMA completes.
    __device__ static void wait_input_empty(SharedStorage* __restrict__ smem,
                                            std::uint32_t arm_id) {
        constexpr std::uint32_t kSlots =
            RoutedBarrier<Dims>::kNumDownWeightSlots;
        if (arm_id < kSlots) return;
        smem->barriers.routed.accumulator_ready.wait(arm_id - kSlots);
    }

    // Wait for the previous down MMA or the first lap's aliased up MMAs.
    __device__ static void wait_down_weight_empty(
        SharedStorage* __restrict__ smem, std::uint32_t id_base,
        std::uint32_t ring_iter_id, std::uint32_t slot) {
        if (ring_iter_id - id_base >= Base::kNumWeightSlots) {
            smem->barriers.routed.accumulator_ready.wait(ring_iter_id -
                                                         Base::kNumWeightSlots);
            return;
        }
        // A block with no up task has no previous weight occupant.
        if (id_base == 0u) return;
#pragma unroll
        for (std::uint32_t h = 0; h < Traits::kUpSlotsPerDownSlot; ++h) {
            smem->barriers.routed.accumulator_ready.wait(
                id_base - Base::kNumUpWeightSlots +
                Traits::kUpSlotsPerDownSlot * slot + h);
        }
        // The mbarrier wait does not order SMEM reads against TMA writes.
        asm volatile("fence.proxy.async;" ::: "memory");
    }
};

}  // namespace halops::moe
