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
#include "kernel/moe/quant_utils.cuh"

namespace halops::moe {

template <typename KernelTraits>
struct RoutedUpProjectionBase
    : UpProjectionBase<KernelTraits, typename KernelTraits::RoutedUpGemm> {
    using Base =
        UpProjectionBase<KernelTraits, typename KernelTraits::RoutedUpGemm>;
    using Base::kAccColsPerThread;
    using Base::kNumTilesK;
    using Base::kNumWeightSlots;
    using Base::kQuantGroups;
    using typename Base::Accumulator;
    using typename Base::Dims;
    using typename Base::SharedStorage;
    using typename Base::Traits;
    using typename Base::UpGemm;
    using typename Base::Workspace;

    static_assert(UpGemm::kTileShapeN % WarpRole::kCalcWarpCount == 0u,
                  "every calc warp must publish the same number of tokens");
    static_assert(UpGemm::kTileShapeM / 2u * UpGemm::kTileShapeN ==
                      WarpRole::kCalcWarpCount * 32u * kAccColsPerThread,
                  "the calc warps must cover every routed gate/up pair");

    // The slab row this token's (expert, rank) pair writes, if it has one.
    __device__ static __forceinline__ bool output_row(
        SharedStorage* __restrict__ smem, std::uint32_t e, std::uint32_t tok,
        std::uint32_t batch_size, std::uint32_t& dest_row) {
        if (tok >= batch_size) return false;
        // `rank_of` returns 0xFF when the token misses this expert.
        const std::uint8_t rank = smem->routing.rank_of(e, tok);
        if (rank == 0xFFu) return false;
        dest_row = smem->routing.slab_begin(e) + rank;
        return true;
    }

    // Apply the gated activation, then quantize and publish each token.
    __device__ static void epilogue(
        Workspace* __restrict__ workspace, SharedStorage* __restrict__ smem,
        std::uint32_t e, std::uint32_t batch_size, std::uint32_t m_block_idx,
        Activation activation, float situ_beta, float situ_linear_beta,
        Accumulator& acc_gate, Accumulator& acc_up) {
        [[maybe_unused]] const std::uint32_t expert_id =
            smem->routing.active_experts[e];

        const std::uint32_t base_feature =
            m_block_idx * UpGemm::kTileShapeM / 2u;

        const auto act_span = KERNEL_TIMER_BEGIN(
            smem->timer.get(), "up_combine_gated_activation", expert_id);
        const auto& routing = smem->routing;
        const std::uint32_t token_mask =
            routing.token_mask(e) & ((1u << batch_size) - 1u);
        const float* const expert_weights =
            routing.expert_row_weight + routing.slab_begin(e);
        Accumulator weights{};
#pragma unroll
        for (std::uint32_t i = 0; i < kAccColsPerThread; ++i) {
            const std::uint32_t token =
                (i / 2u) * 8u + 2u * (WarpRole::get_lane_idx() & 3u) + i % 2u;
            if (((token_mask >> token) & 1u) != 0u) {
                weights[i] =
                    expert_weights[__popc(token_mask & ((1u << token) - 1u))];
            }
        }
        Base::combine_gated_activation(nullptr, smem->routed_experts.act_up_out,
                                       acc_gate, acc_up, activation, situ_beta,
                                       situ_linear_beta, &weights, token_mask);
        KERNEL_TIMER_END(smem->timer.get(), act_span);
        WarpRole::sync_calc_threads();

        const auto publish_span = KERNEL_TIMER_BEGIN(
            smem->timer.get(), "up_quantize_publish_rows", expert_id);

        constexpr std::uint32_t kTokensPerWarp =
            UpGemm::kTileShapeN / WarpRole::kCalcWarpCount;
#pragma unroll 1
        for (std::uint32_t g = 0; g < kTokensPerWarp; ++g) {
            quantize_publish_rows(
                workspace, smem, e, batch_size, base_feature, m_block_idx,
                WarpRole::get_warp_idx() + g * WarpRole::kCalcWarpCount);
        }
        KERNEL_TIMER_END(smem->timer.get(), publish_span);
        WarpRole::sync_calc_threads();
    }

    // Quantize and publish one routed output row.
    __device__ static void quantize_publish_rows(
        Workspace* __restrict__ workspace, SharedStorage* __restrict__ smem,
        std::uint32_t e, std::uint32_t batch_size, std::uint32_t base_feature,
        std::uint32_t m_block_idx, std::uint32_t tok) {
        using Quant = typename Traits::RoutedDownGemm::InputQuant;
        using Quantizer = GroupWiseQuantizer<Quant, UpGemm::kTileShapeM / 2u>;
        const std::uint32_t lane_idx = WarpRole::get_lane_idx();
        std::uint32_t dest_row = 0u;
        if (!output_row(smem, e, tok, batch_size, dest_row)) {
            return;
        }

        typename Quant::ElementType quantized[kQuantGroups];
        typename Quantizer::ScaleArray scales;
        Base::quantize_output_row(smem->routed_experts.act_up_out, tok,
                                  quantized, scales);
        auto* const out = workspace->routed_up_out.row(dest_row) + base_feature;
        typename Quantizer::ScaleType* scale;
        if constexpr (Quant::kMethod == QuantMethod::kMXFP8) {
            scale = workspace->routed_up_out.mx_scale_row(dest_row) +
                    base_feature / Quant::kBlockSize;
        } else {
            scale = workspace->routed_up_out.scale_row(dest_row) + m_block_idx;
        }
#pragma unroll
        for (std::uint32_t i = 0; i < kQuantGroups; ++i) {
            out[i * 32u + lane_idx] = quantized[i];
        }
        if (lane_idx == 0u) {
#pragma unroll
            for (std::uint32_t i = 0; i < Quantizer::kNumGroups; ++i) {
                scale[i] = scales[i];
            }
        }
        // Store quantized data and scales before publishing readiness.
        workspace->cta_comm.publish_up_out(smem->launch_parity, dest_row,
                                           m_block_idx, lane_idx);
        KERNEL_TIMER_MARK(smem->timer.get(), "up_handoff_publish",
                          smem->routing.active_experts[e]);
    }
};

template <typename KernelTraits,
          bool UseBlockscaled =
              KernelTraits::RoutedUpGemm::MmaCore::kUseBlockscaled>
struct RoutedUpProjection;

// MXFP8/MXFP4: scales are applied by the blockscaled MMA.
template <typename KernelTraits>
struct RoutedUpProjection<KernelTraits, true>
    : RoutedUpProjectionBase<KernelTraits> {
    using Base = RoutedUpProjectionBase<KernelTraits>;
    using typename Base::Accumulator;
    using typename Base::Dims;
    using typename Base::SharedStorage;
    using typename Base::Traits;
    using typename Base::UpGemm;
    using typename Base::Workspace;

    static_assert(UpGemm::MmaCore::kUseBlockscaled,
                  "RoutedUpProjection<true> requires a blockscaled MMA core");

    using Base::kNumAccSlots;
    using Base::kUpWeightTxBytes;

    // Run up-projection over this block's expert sequence.
    __device__ static void run(
        const typename UpGemm::ElementScale* __restrict__ expert_scales_up,
        std::uint32_t batch_size, Activation activation, float situ_beta,
        float situ_linear_beta, Workspace* __restrict__ workspace,
        SharedStorage* __restrict__ smem,
        CUtensorMap const& tma_up_weights_desc,
        CUtensorMap const& tma_up_scale_desc, TaskScheduler<Traits>& sched,
        std::uint64_t weight_cache_hint) {
        const unsigned warp_idx = WarpRole::get_warp_idx();

        const std::uint32_t m_block_idx = sched.cta_group().cta_idx_in_group;
        const std::uint32_t tmem_base = smem->tmem_ptr;

        TaskInfo task;
        while (sched.get_next_task(task)) {
            const std::uint32_t expert_id =
                smem->routing.active_experts[task.local_expert_idx];
            const std::uint32_t task_id = task.task_id;

            KERNEL_TIMER_SCOPE(smem->timer.get(), "up_expert_task", expert_id);
            if (WarpRole::is_weight_loader_thread()) {
                run_load_k_loop(smem, tma_up_weights_desc, tma_up_scale_desc,
                                expert_scales_up, m_block_idx, task_id,
                                expert_id, weight_cache_hint);
            } else if (WarpRole::is_mma_issuer_warp(warp_idx)) {
                run_mma_k_loop(smem, tmem_base, task_id, m_block_idx);
            } else if (WarpRole::is_calc_warp(warp_idx)) {
                Accumulator acc_gate_out;
                Accumulator acc_up_out;
                load_accumulators(smem, acc_gate_out, acc_up_out, tmem_base,
                                  task_id, expert_id);
                KERNEL_TIMER_SCOPE(smem->timer.get(), "up_epilogue", expert_id);
                Base::epilogue(workspace, smem, task.local_expert_idx,
                               batch_size, m_block_idx, activation, situ_beta,
                               situ_linear_beta, acc_gate_out, acc_up_out);
            }
        }

        if (sched.next_task_id() > 0u) {
            if (WarpRole::is_weight_loader_thread()) {
                const std::uint32_t last_tile =
                    sched.next_task_id() * Base::kNumTilesK - 1u;
                KERNEL_TIMER_SCOPE(smem->timer.get(), "up_wait_weight_empty",
                                   last_tile);
                smem->barriers.routed.up_weight_empty.wait(last_tile);
                // Order completed SMEM reads before down's TMA writes.
                asm volatile("fence.proxy.async;" ::: "memory");
            } else if (WarpRole::is_mma_issuer_warp(warp_idx)) {
                KERNEL_TIMER_SCOPE(smem->timer.get(),
                                   "up_wait_accumulator_empty",
                                   sched.next_task_id() - 1u);
                smem->barriers.routed.accumulator_empty.wait(
                    sched.next_task_id() - 1u);
            }
        }
    }

    // Read the final sum, then release this task's accumulator slot.
    __device__ static void load_accumulators(SharedStorage* __restrict__ smem,
                                             Accumulator& acc_gate_out,
                                             Accumulator& acc_up_out,
                                             std::uint32_t tmem_base,
                                             std::uint32_t task_id,
                                             std::uint32_t expert_id) {
        const auto acc_ready_span = KERNEL_TIMER_BEGIN(
            smem->timer.get(), "up_wait_accumulator_ready", expert_id);
        smem->barriers.routed.accumulator_ready.wait(task_id);
        KERNEL_TIMER_END(smem->timer.get(), acc_ready_span);

        Base::load_accumulators(tmem_base, acc_gate_out, acc_up_out,
                                task_id % kNumAccSlots);
        cutlass::arch::fence_view_async_tmem_load();
        __syncwarp();
        if (WarpRole::get_lane_idx() == 0u) {
            smem->barriers.routed.accumulator_empty.arrive(task_id);
        }
        WarpRole::sync_calc_threads();
    }

    // Accumulate one task's K tiles in its slot; release each consumed weight.
    __device__ static void run_mma_k_loop(SharedStorage* __restrict__ smem,
                                          std::uint32_t tmem_base,
                                          std::uint32_t task_id,
                                          std::uint32_t m_block_idx) {
        typename UpGemm::TiledMma tiled_mma{};
        auto cta_mma = tiled_mma.get_slice(cute::_0{});
        auto tCt = Base::make_accumulator_fragment(cta_mma);
        const std::uint32_t acc_slot = task_id % kNumAccSlots;

        if (task_id >= kNumAccSlots) {
            KERNEL_TIMER_SCOPE(smem->timer.get(), "up_wait_accumulator_empty",
                               task_id);
            smem->barriers.routed.accumulator_empty.wait(task_id -
                                                         kNumAccSlots);
        }

#pragma unroll 1
        for (std::uint32_t k_iter_id = 0; k_iter_id < Base::kNumTilesK;
             ++k_iter_id) {
            const std::uint32_t ring_iter_id =
                task_id * Base::kNumTilesK + k_iter_id;
            const std::uint32_t weight_slot =
                ring_iter_id % Base::kNumWeightSlots;
            const auto mma_ready_span = KERNEL_TIMER_BEGIN(
                smem->timer.get(), "up_wait_mma_ready", ring_iter_id);
            smem->barriers.routed.wait_up_mma_ready(ring_iter_id);
            KERNEL_TIMER_END(smem->timer.get(), mma_ready_span);
            if constexpr (UpGemm::kScaleLayout == ScaleLayout::kNaturalAtom) {
                KERNEL_TIMER_SCOPE(smem->timer.get(), "permute_up_scale",
                                   ring_iter_id);
                Base::permute_up_scale(
                    smem->routed_experts.w_up_scale[weight_slot], m_block_idx);
            }
            Base::mma(
                tiled_mma, cta_mma, tCt,
                reinterpret_cast<typename Base::WeightElement*>(
                    &smem->routed_experts.w_up[weight_slot][0]),
                &smem->routed_experts.quantized_input.data[k_iter_id][0][0][0],
                tmem_base, acc_slot,
                smem->routed_experts.w_up_scale[weight_slot],
                &smem->routed_experts.quantized_input.scale[k_iter_id][0],
                /*accumulate=*/k_iter_id != 0u);
            // The commit covers every async tcgen05 op issued before it.
            cutlass::arch::umma_arrive(
                smem->barriers.routed.up_weight_empty.raw(weight_slot));
        }
        cutlass::arch::umma_arrive(
            smem->barriers.routed.accumulator_ready.raw(acc_slot));
    }

    // A weight slot is reusable when its previous K tile's MMA completes.
    __device__ static void wait_weight_empty(SharedStorage* __restrict__ smem,
                                             std::uint32_t arm_id) {
        if (arm_id < Base::kNumWeightSlots) {
            return;
        }
        KERNEL_TIMER_SCOPE(smem->timer.get(), "up_wait_weight_empty", arm_id);
        smem->barriers.routed.up_weight_empty.wait(arm_id -
                                                   Base::kNumWeightSlots);
    }

    // Stream weight tiles and their scale atoms through the ring.
    __device__ static void run_load_k_loop(
        SharedStorage* __restrict__ smem,
        const CUtensorMap& tma_up_weights_desc,
        const CUtensorMap& tma_up_scale_desc,
        const typename UpGemm::ElementScale* __restrict__ up_scale,
        std::uint32_t m_block_idx, std::uint32_t task_id,
        std::uint32_t expert_id, std::uint64_t weight_cache_hint) {
        const std::uint32_t weight_row_base = m_block_idx * UpGemm::kTileShapeM;

#pragma unroll 1
        for (std::uint32_t k_iter_id = 0; k_iter_id < Base::kNumTilesK;
             ++k_iter_id) {
            const std::uint32_t ring_iter_id =
                task_id * Base::kNumTilesK + k_iter_id;
            wait_weight_empty(smem, ring_iter_id);
            const std::uint32_t slot = ring_iter_id % Base::kNumWeightSlots;
            smem->barriers.routed.up_weight_ready.arrive(slot,
                                                         kUpWeightTxBytes);
            Base::load_weight_tile(
                tma_up_weights_desc,
                reinterpret_cast<typename Base::WeightElement*>(
                    &smem->routed_experts.w_up[slot][0]),
                smem->barriers.routed.up_weight_ready.raw(slot),
                expert_id * UpGemm::kGemmM + weight_row_base, k_iter_id,
                weight_cache_hint);
            load_up_weight_scale(smem, tma_up_scale_desc, up_scale,
                                 ring_iter_id, expert_id, weight_row_base);
            KERNEL_TIMER_MARK(smem->timer.get(), "up_load_weight_tile",
                              k_iter_id);
        }
    }

    // Load this K tile's SFA into the weight slot's already-armed barrier.
    __device__ static void load_up_weight_scale(
        SharedStorage* __restrict__ smem, const CUtensorMap& tma_up_scale_desc,
        const typename UpGemm::ElementScale* __restrict__ up_scale,
        std::uint32_t ring_iter_id, std::uint32_t expert_id,
        std::uint32_t weight_row_base) {
        const std::uint32_t slot = ring_iter_id % Base::kNumWeightSlots;
        const std::uint32_t k_tile = ring_iter_id % Base::kNumTilesK;
        const std::uint32_t row_atom =
            expert_id * (UpGemm::kGemmM / 128u) + weight_row_base / 128u;
        Base::load_weight_scale_tile(
            up_scale, row_atom, k_tile, smem->routed_experts.w_up_scale[slot],
            smem->barriers.routed.up_weight_ready.raw(slot), tma_up_scale_desc);
    }
};

// Blockwise FP8: the calc warps apply FP32 weight and activation scales.
template <typename KernelTraits>
struct RoutedUpProjection<KernelTraits, false>
    : RoutedUpProjectionBase<KernelTraits> {
    using Base = RoutedUpProjectionBase<KernelTraits>;
    using typename Base::Accumulator;
    using typename Base::Dims;
    using typename Base::SharedStorage;
    using typename Base::Traits;
    using typename Base::UpGemm;
    using typename Base::Workspace;

    static_assert(
        !UpGemm::MmaCore::kUseBlockscaled,
        "RoutedUpProjection<false> requires a non-blockscaled MMA core");
    using Base::kNumAccSlots;
    static_assert(
        Base::kNumWeightSlots <= kNumAccSlots,
        "The weight ring must not outrun the accumulator ring, or the "
        "re-arm's wait_accumulator_ready(arm_id - kNumWeightSlots) "
        "can read a phase the ring has already wrapped past.");
    static constexpr std::uint32_t kNumUpScaleBlocksK =
        UpGemm::kGemmK / UpGemm::WeightQuant::kBlockSize;
    static constexpr std::uint32_t kScaleSlots = UpGemm::kWeightScaleSlots;

    // Run up-projection over this block's expert sequence.
    __device__ static void run(
        const typename UpGemm::ElementScale* __restrict__ expert_scales_up,
        std::uint32_t batch_size, Activation activation, float situ_beta,
        float situ_linear_beta, Workspace* __restrict__ workspace,
        SharedStorage* __restrict__ smem,
        CUtensorMap const& tma_up_weights_desc,
        [[maybe_unused]] CUtensorMap const& tma_up_scale_desc,
        TaskScheduler<Traits>& sched, std::uint64_t weight_cache_hint) {
        const unsigned warp_idx = WarpRole::get_warp_idx();

        const std::uint32_t m_block = sched.cta_group().cta_idx_in_group;
        const std::uint32_t base_feature = m_block * UpGemm::kTileShapeM / 2u;
        // Row base in the interleaved [E, 2N, K] tensor, one tile per K-step.
        const std::uint32_t weight_row_base = m_block * UpGemm::kTileShapeM;
        const std::uint32_t tmem_base = smem->tmem_ptr;

        const std::uint32_t first_e = sched.first_expert_idx();

        if (WarpRole::is_other_loader_warp(warp_idx) && sched.is_active()) {
            load_up_weight_scale(expert_scales_up,
                                 smem->routing.active_experts[first_e],
                                 base_feature, 0u, smem);
        }

        TaskInfo task;
        while (sched.get_next_task(task)) {
            const std::uint32_t expert_id =
                smem->routing.active_experts[task.local_expert_idx];
            const std::uint32_t task_id = task.task_id;

            // Task t reads slot t % kScaleSlots while the next one is filled.
            const std::uint32_t cur_scale_slot = task_id % kScaleSlots;

            const bool has_next_e = (task.next_expert_idx != kNoExpert);
            const auto task_span = KERNEL_TIMER_BEGIN(
                smem->timer.get(), "up_expert_task", expert_id);

            if (WarpRole::is_weight_loader_thread()) {
                run_load_k_loop(smem, tma_up_weights_desc, weight_row_base,
                                task_id, expert_id, weight_cache_hint);
            } else if (WarpRole::is_mma_issuer_warp(warp_idx)) {
                run_mma_k_loop(smem, tmem_base, task_id);
            } else if (WarpRole::is_calc_warp(warp_idx)) {
                Accumulator acc_gate_out{};
                Accumulator acc_up_out{};
                run_dequant_k_loop(smem, acc_gate_out, acc_up_out,
                                   cur_scale_slot, tmem_base, task_id,
                                   expert_id);
                KERNEL_TIMER_SCOPE(smem->timer.get(), "up_epilogue", expert_id);
                Base::epilogue(workspace, smem, task.local_expert_idx,
                               batch_size, m_block, activation, situ_beta,
                               situ_linear_beta, acc_gate_out, acc_up_out);
            } else if (WarpRole::is_other_loader_warp(warp_idx) && has_next_e) {
                const std::uint32_t next_expert_id =
                    smem->routing.active_experts[task.next_expert_idx];
                load_up_weight_scale(expert_scales_up, next_expert_id,
                                     base_feature, task_id + 1u, smem);
            }
            KERNEL_TIMER_END(smem->timer.get(), task_span);
        }
    }

    // The previous tile's accumulator readiness also releases its weights.
    __device__ static void wait_weight_empty(SharedStorage* __restrict__ smem,
                                             std::uint32_t arm_id) {
        if (arm_id < Base::kNumWeightSlots) {
            return;
        }
        KERNEL_TIMER_SCOPE(smem->timer.get(), "up_wait_weight_empty", arm_id);
        smem->barriers.routed.accumulator_ready.wait(arm_id -
                                                     Base::kNumWeightSlots);
    }

    // Stream weight tiles through the ring without scale atoms.
    __device__ static void run_load_k_loop(
        SharedStorage* __restrict__ smem,
        const CUtensorMap& tma_up_weights_desc, std::uint32_t weight_row_base,
        std::uint32_t task_id, std::uint32_t expert_id,
        std::uint64_t weight_cache_hint) {
#pragma unroll 1
        for (std::uint32_t k_iter_id = 0; k_iter_id < Base::kNumTilesK;
             ++k_iter_id) {
            const std::uint32_t ring_iter_id =
                task_id * Base::kNumTilesK + k_iter_id;
            wait_weight_empty(smem, ring_iter_id);
            const std::uint32_t slot = ring_iter_id % Base::kNumWeightSlots;
            smem->barriers.routed.up_weight_ready.arrive(
                slot, UpGemm::MmaCore::kWeightTileTxBytes);
            Base::load_weight_tile(
                tma_up_weights_desc,
                reinterpret_cast<typename Base::WeightElement*>(
                    &smem->routed_experts.w_up[slot][0]),
                smem->barriers.routed.up_weight_ready.raw(slot),
                expert_id * UpGemm::kGemmM + weight_row_base, k_iter_id,
                weight_cache_hint);
        }
    }

    // Read each completed accumulator slot and drain the last K tile.
    __device__ static void run_dequant_k_loop(SharedStorage* __restrict__ smem,
                                              Accumulator& acc_gate_out,
                                              Accumulator& acc_up_out,
                                              std::uint32_t cur_scale_slot,
                                              std::uint32_t tmem_base,
                                              std::uint32_t task_id,
                                              std::uint32_t expert_id) {
        const auto scale_span = KERNEL_TIMER_BEGIN(
            smem->timer.get(), "up_wait_weight_scale_ready", expert_id);
        smem->barriers.routed.wait_weight_scale_ready(task_id, cur_scale_slot);
        KERNEL_TIMER_END(smem->timer.get(), scale_span);

#pragma unroll 1
        for (std::uint32_t k_iter_id = 1; k_iter_id <= Base::kNumTilesK;
             ++k_iter_id) {
            dequant(smem, acc_gate_out, acc_up_out, cur_scale_slot,
                    task_id * Base::kNumTilesK + k_iter_id, k_iter_id,
                    tmem_base);
        }

        WarpRole::sync_calc_threads();
        if (WarpRole::get_lane_idx() == 0u) {
            smem->barriers.routed.weight_scale_empty.arrive(cur_scale_slot);
        }
    }

    // Wait for each input and weight tile, issue MMA, then commit the slot.
    __device__ static void run_mma_k_loop(SharedStorage* __restrict__ smem,
                                          std::uint32_t tmem_base,
                                          std::uint32_t task_id) {
        typename UpGemm::TiledMma tiled_mma{};
        auto cta_mma = tiled_mma.get_slice(cute::_0{});
        auto tCt = Base::make_accumulator_fragment(cta_mma);

#pragma unroll 1
        for (std::uint32_t k_iter_id = 0; k_iter_id < Base::kNumTilesK;
             ++k_iter_id) {
            const std::uint32_t ring_iter_id =
                task_id * Base::kNumTilesK + k_iter_id;
            const std::uint32_t acc_slot = ring_iter_id % kNumAccSlots;
            const std::uint32_t weight_slot =
                ring_iter_id % Base::kNumWeightSlots;
            // Each K tile needs its own accumulator for software scaling.
            const auto mma_ready_span = KERNEL_TIMER_BEGIN(
                smem->timer.get(), "up_wait_mma_ready", ring_iter_id);
            smem->barriers.routed.wait_up_mma_ready(ring_iter_id);
            KERNEL_TIMER_END(smem->timer.get(), mma_ready_span);
            Base::mma(
                tiled_mma, cta_mma, tCt,
                reinterpret_cast<typename Base::WeightElement*>(
                    &smem->routed_experts.w_up[weight_slot][0]),
                &smem->routed_experts.quantized_input.data[k_iter_id][0][0][0],
                tmem_base, acc_slot);
            // The commit covers every async tcgen05 op issued before it.
            cutlass::arch::umma_arrive(
                smem->barriers.routed.accumulator_ready.raw(acc_slot));
        }
    }

    // Release accumulator iteration ring_iter_id - 1 after TMEM reads.
    __device__ static __forceinline__ void release_accumulator_slot(
        SharedStorage* __restrict__ smem, std::uint32_t ring_iter_id) {
        cutlass::arch::fence_view_async_tmem_load();
        __syncwarp();
        if (WarpRole::get_lane_idx() == 0u) {
            smem->barriers.routed.accumulator_empty.arrive(ring_iter_id - 1u);
        }
    }

    __device__ static void dequant(SharedStorage* __restrict__ smem,
                                   Accumulator& acc_gate, Accumulator& acc_up,
                                   std::uint32_t cur_scale_slot,
                                   std::uint32_t ring_iter_id, std::uint32_t s,
                                   std::uint32_t tmem_base) {
        // Dequant reads tile ring_iter_id - 1, one iteration behind the UMMA.
        const auto acc_ready_span = KERNEL_TIMER_BEGIN(
            smem->timer.get(), "up_wait_accumulator_ready", ring_iter_id);
        smem->barriers.routed.accumulator_ready.wait(ring_iter_id - 1u);
        KERNEL_TIMER_END(smem->timer.get(), acc_ready_span);

        const std::uint32_t prev_slot = (ring_iter_id - 1u) % kNumAccSlots;

        const float* ws_buf = smem->routed_experts.w_up_scale[cur_scale_slot];
#pragma unroll
        for (std::uint32_t sub = 0; sub < UpGemm::MmaCore::kNumQuantBlocksB;
             ++sub) {
            // kblk indexes this activation block in both scale tables.
            const std::uint32_t kblk =
                (s - 1u) * UpGemm::MmaCore::kNumQuantBlocksB + sub;
            const std::uint32_t acc_col =
                prev_slot * Base::kAccColumns + sub * UpGemm::kTileShapeN;
            Base::dequant(tmem_base, acc_col,
                          smem->routed_experts.quantized_input.scale[kblk],
                          ws_buf[kblk], ws_buf[kblk + kNumUpScaleBlocksK],
                          acc_gate, acc_up);
        }
        release_accumulator_slot(smem, ring_iter_id);
    }

    // Async (cp.async) per-expert up-projection block-scale prefetch.
    __device__ static inline void load_up_weight_scale_per_expert_async(
        const float* __restrict__ expert_scales_up, std::uint32_t expert_id,
        unsigned base_row_up, float* __restrict__ dest) {
        const unsigned lane_idx = WarpRole::get_lane_idx();
        constexpr uint32_t kNumScaleBlocksK = kNumUpScaleBlocksK;
        constexpr uint32_t kNumScaleBlocksM =
            UpGemm::kGemmM / UpGemm::WeightQuant::kBlockSize;

        for (uint32_t i = lane_idx; i < 2 * kNumScaleBlocksK; i += 32u) {
            // 0 → gate half, 1 → up half
            uint32_t rb_local = i / kNumScaleBlocksK;
            uint32_t kb = i % kNumScaleBlocksK;
            uint32_t row = base_row_up + rb_local * (UpGemm::kGemmM / 2u);
            uint32_t rb_global = row / UpGemm::WeightQuant::kBlockSize;
            const float* src =
                &expert_scales_up[expert_id * kNumScaleBlocksM *
                                      kNumScaleBlocksK +
                                  rb_global * kNumScaleBlocksK + kb];
            cp_async_4bytes(&dest[i], src);
        }
        // Commit from the issuing warp; other warps' empty commits add a group.
        cp_async_commit_group();
    }

    // Load one expert's block-scale tile into `slot` and publish it.
    __device__ static void load_up_weight_scale(
        const float* __restrict__ expert_scales_up, std::uint32_t expert_id,
        unsigned base_row_up, std::uint32_t task_id,
        SharedStorage* __restrict__ smem) {
        const std::uint32_t slot = task_id % kScaleSlots;
        if (task_id >= kScaleSlots) {
            KERNEL_TIMER_SCOPE(smem->timer.get(), "up_wait_weight_scale_empty",
                               expert_id);
            smem->barriers.routed.wait_weight_scale_empty(task_id - kScaleSlots,
                                                          slot);
        }

        KERNEL_TIMER_SCOPE(smem->timer.get(), "up_load_weight_scale",
                           expert_id);
        load_up_weight_scale_per_expert_async(
            expert_scales_up, expert_id, base_row_up,
            smem->routed_experts.w_up_scale[slot]);
        cp_async_wait_group<0>();
        __syncwarp();
        if (WarpRole::get_lane_idx() == 0u) {
            smem->barriers.routed.weight_scale_ready.arrive(slot);
        }
    }
};

}  // namespace halops::moe
