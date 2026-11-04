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
#include "kernel/moe/moe_scheduler.cuh"
#include "kernel/moe/moe_shared_storage.cuh"
#include "kernel/moe/moe_tma.cuh"
#include "kernel/moe/moe_traits.cuh"
#include "kernel/moe/moe_workspace.cuh"
#include "kernel/moe/ptx_utils.cuh"

namespace halops::moe {

// Per-expert activation-scale loader (down-projection), bitmask variant.
template <typename Dims, std::size_t ScaleBlocks, std::size_t ScaleTok>
__device__ inline void load_down_act_scale(
    const MoEWorkspace<Dims>* __restrict__ workspace,
    std::uint32_t expert_start, std::uint32_t routed_count,
    float (&dest_scale)[ScaleBlocks][ScaleTok]) {
    using Traits = MoEKernelTraits<Dims>;
    static_assert(ScaleTok == Traits::DownGemm::kTileShapeN,
                  "down-proj activation scale tile must have the tile's N "
                  "= 8 tokens");
    static_assert(ScaleBlocks ==
                      Traits::DownGemm::kGemmK / Traits::UpGemm::kActQuantBlock,
                  "down-proj activation scale tile must have one block per "
                  "up-block across the full reduction dim");

    constexpr unsigned kScaleKBlocks =
        Traits::DownGemm::kGemmK / Traits::UpGemm::kActQuantBlock;
    const unsigned lane_idx = WarpRole::get_lane_idx();

    const float* __restrict__ src = workspace->routed_up_out.scale.blockwise;

    constexpr unsigned kTotal = static_cast<unsigned>(ScaleTok * ScaleBlocks);
    for (unsigned i = lane_idx; i < kTotal; i += 32u) {
        const unsigned slot_row = i / kScaleKBlocks;
        const unsigned block = i % kScaleKBlocks;
        if (slot_row < routed_count) {
            const uint32_t source_row = expert_start + slot_row;
            dest_scale[block][slot_row] =
                src[source_row * kScaleKBlocks + block];
        } else {
            // Unused rank; zeroed for cleanliness.
            dest_scale[block][slot_row] = 0.f;
        }
    }
}

// Per-expert weight-scale loader: one lane per (m-tile, K-block).
template <typename Dims, std::size_t TilesM, std::size_t BlocksK>
__device__ inline void load_down_weight_scale_per_expert(
    const float* __restrict__ expert_scales_down, std::uint32_t expert_id,
    std::uint32_t base_m, float (&dest)[TilesM][BlocksK]) {
    constexpr uint32_t kNumScaleBlocksM =
        MoEKernelTraits<Dims>::DownGemm::kNumScaleBlocksM;
    constexpr uint32_t kNumScaleBlocksK =
        MoEKernelTraits<Dims>::DownGemm::kNumScaleBlocksK;
    static_assert(BlocksK == kNumScaleBlocksK,
                  "down-proj weight scale tile must span the full K-block "
                  "range");
    static_assert(TilesM * BlocksK <= WarpRole::kThreadsPerWarp,
                  "one lane per (m-tile, K-block) weight scale");
    const unsigned lane_idx = WarpRole::get_lane_idx();

    if (lane_idx < TilesM * BlocksK) {
        const unsigned m_tile = lane_idx / BlocksK;
        const unsigned k_block = lane_idx % BlocksK;
        const uint32_t rb =
            (base_m +
             m_tile * MoEKernelTraits<Dims>::DownGemm::MmaCore::kAtomShapeM) /
            Dims::kQuantConfig.kBlockSize;
        dest[m_tile][k_block] =
            expert_scales_down[expert_id * kNumScaleBlocksM * kNumScaleBlocksK +
                               rb * kNumScaleBlocksK + k_block];
    }
}

template <typename KernelTraits>
struct DownProjection {
    using Traits = KernelTraits;
    using Dims = typename Traits::MoEDims;
    using SharedStorage = MoESharedStorage<Dims>;
    using Workspace = MoEWorkspace<Dims>;
    using RoutingTables = MoERoutingTables<Dims>;
    using DownGemm = typename Traits::DownGemm;

    static constexpr std::uint32_t kTileShapeM = DownGemm::kTileShapeM;
    static constexpr std::uint32_t kNumTilesK = DownGemm::kNumTilesK;
    static constexpr std::uint32_t kNumAccSlots = Traits::kNumAccSlots;
    // TMEM columns one accumulator-ring slot occupies.
    static constexpr std::uint32_t kAccColumns = DownGemm::kAccColumns;
    // The blockscaled recipe stages one SFA/SFB set per weight slot, past the
    // accumulator ring; the loaders fill a slot and the MMA reads it.
    using DownScaleTraits = typename DownGemm::ScaleTraits;
    static constexpr std::uint32_t kScaleSlotColumns =
        DownScaleTraits::kSfaTmemColumns + DownScaleTraits::kSfbTmemColumns;
    static constexpr std::uint32_t kDownScaleTmemBase =
        kNumAccSlots * kAccColumns;
    static constexpr std::uint32_t kNumScaleSlots = Traits::kDownScaleSlots;
    static constexpr std::uint32_t down_sfa_tmem(std::uint32_t step_local) {
        return kDownScaleTmemBase +
               (step_local % kNumScaleSlots) * kScaleSlotColumns;
    }
    static constexpr std::uint32_t down_sfb_tmem(std::uint32_t step_local) {
        return down_sfa_tmem(step_local) + DownScaleTraits::kSfaTmemColumns;
    }
    static_assert(!DownGemm::MmaCore::kUseBlockscaled ||
                      kDownScaleTmemBase +
                              Traits::kDownScaleSlots * kScaleSlotColumns <=
                          Traits::kTmemColumns,
                  "the down's scale ring must fit the TMEM allocation");

    // Sub-accumulators per (K-step, m-tile), one per activation quant block.
    static constexpr std::uint32_t kNumDownSubBlocks = DownGemm::kNumSubBlocks;
    static constexpr std::uint32_t kNumDownWeightScaleBlocksK =
        DownGemm::kNumScaleBlocksK;
    static constexpr std::uint32_t kDownWeightTxBytes =
        DownGemm::MmaCore::kWeightTileTxBytes +
        (DownGemm::MmaCore::kUseBlockscaled
             ? DownGemm::ScaleTraits::kSfaSmemBytes
             : 0u);
    static constexpr std::uint32_t kDownActTxBytes =
        DownGemm::kTileShapeN * Traits::kFp8ActChunksPerBlock *
        Traits::kLboChunkK;  // 1 KB
    static constexpr std::uint64_t kBLbo = 16ULL;
    static constexpr std::uint64_t kBSbo = 128ULL;

    static constexpr std::uint32_t kNumWeightSlots = DownGemm::kNumWeightSlots;
    // The up phase's weight ring depth, which the down ring aliases at 2:1.
    static constexpr std::uint32_t kNumUpWeightSlots =
        Traits::UpGemm::kNumWeightSlots;

    static_assert(Traits::kUpSlotsPerDownSlot >= 1u,
                  "A w_down slot must cover at least one w_up slot: the first "
                  "lap waits the aliased up slots per down slot.");
    static_assert(Traits::kUpSlotsPerDownSlot * kNumWeightSlots <=
                      kNumUpWeightSlots,
                  "The down weight ring must alias whole groups of up weight "
                  "slots.");

    static_assert(kTileShapeM % Dims::kQuantConfig.kBlockSize == 0u &&
                      128u % Dims::kQuantConfig.kBlockSize == 0u,
                  "The down-proj m-tile base must be kQuantBlock-aligned for "
                  "the per-m-tile weight-scale load.");
    static_assert(Dims::kNumTokens <= 16, "SM100 down-proj supports BS<=16.");

    static_assert(
        kNumWeightSlots <= kNumAccSlots,
        "The weight ring must not outrun the accumulator ring, or the "
        "re-arm's wait_accumulator_ready(ring_iter_id - kNumWeightSlots) can "
        "read a phase the ring has already wrapped past.");

    // B operand: one tile-N-row x 128-B SW128 slab per N-substep, built from
    // the CUTLASS SWZ128 atom.
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

    __device__ static void run(
        const typename DownGemm::ElementScale* __restrict__ expert_scales_down,
        MoEWorkspace<Dims>* __restrict__ workspace,
        MoESharedStorage<Dims>* __restrict__ smem,
        CUtensorMap const& tma_down_weights_desc,
        CUtensorMap const& tma_down_activations_desc,
        TaskScheduler<Traits>& sched, std::uint64_t weight_cache_hint) {
        const unsigned warp_idx = WarpRole::get_warp_idx();
        const unsigned lane_idx = WarpRole::get_lane_idx();
        const unsigned m_tile_idx = warp_idx / 4u;  // WG0 -> m-tile 0, WG1 -> 1
        const unsigned lane_base = (warp_idx & 3u) * 32u;

        // Every block takes part in the down phase, whatever its up schedule
        // was.
        if (WarpRole::is_publish_warp(warp_idx)) {
            if (lane_idx == 0u) {
                publish_down_tasks(workspace, smem, sched);
            }
            return;
        }
        // The other TMA-warp lanes have no down role and would slow lane 0.
        if ((WarpRole::is_weight_loader_warp(warp_idx) ||
             WarpRole::is_other_loader_warp(warp_idx)) &&
            lane_idx != 0u) {
            return;
        }

        // The ring id continues the up phase's count.
        const std::uint32_t id_base =
            sched.up_task_count() * Traits::UpGemm::kNumTilesK;
        const std::uint32_t tmem_base = smem->tmem_ptr;

        TaskInfo task;
        while (true) {
            if (WarpRole::is_weight_loader_thread()) {
                // Claim one task ahead, after its predecessor's first UMMA
                // read the weights.
                const std::uint32_t next = sched.next_task_id();
                if (next > 0u) {
                    const std::uint32_t next_local = next * kNumTilesK;
                    wait_down_weight_empty(smem, id_base, id_base + next_local,
                                           next_local % kNumWeightSlots);
                }
                sched.claim_next_task();
            }
            if (!sched.get_next_task(task)) {
                break;
            }

            const std::uint32_t task_id = task.task_id;
            const std::uint32_t local_id = task_id * kNumTilesK;
            const std::uint32_t ring_iter_id = id_base + local_id;
            const std::uint32_t e = task.local_expert_idx;
            const std::uint32_t base_m = task.m_block_idx * kTileShapeM;
            const std::uint32_t expert_id = smem->routing.active_experts[e];
            const auto task_span = KERNEL_TIMER_BEGIN(
                smem->timer(), "down_expert_task", expert_id);

            const std::uint32_t routed_count = smem->routing.slab_count(e);
            const std::uint32_t expert_start = smem->routing.slab_begin(e);

            if (WarpRole::is_weight_loader_thread()) {
                run_load_weights_k_loop(smem, tma_down_weights_desc,
                                        expert_scales_down, id_base, local_id,
                                        expert_id, base_m, weight_cache_hint);
            } else if (WarpRole::is_other_loader_warp(warp_idx)) {
                run_load_activations_k_loop(
                    smem, workspace, tma_down_activations_desc, id_base,
                    local_id, expert_start, routed_count);
            } else if (WarpRole::is_mma_issuer_warp(warp_idx)) {
                run_mma_k_loop(smem, tmem_base, ring_iter_id, local_id,
                               routed_count);
            } else if (WarpRole::is_calc_warp(warp_idx)) {
                WarpRole::sync_calc_threads();
                load_weight_and_act_scale(smem, workspace, expert_scales_down,
                                          expert_id, base_m, expert_start,
                                          routed_count);

                cutlass::Array<float, DownGemm::kTileShapeN> acc_out{};
                run_dequant_k_loop(smem, acc_out, m_tile_idx, lane_base,
                                   tmem_base, ring_iter_id, local_id);

                // Write each slab pair's own combine slot, then hand off.
                const auto epilogue_span = KERNEL_TIMER_BEGIN(
                    smem->timer(), "down_epilogue", expert_id);
                const std::uint32_t out_col =
                    base_m + m_tile_idx * 128u + lane_base + lane_idx;
                const std::uint32_t count = smem->routing.slab_count(e);
                // Compile-time trip count keeps `acc_out`'s rows in registers.
#pragma unroll
                for (std::uint32_t j = 0; j < DownGemm::kTileShapeN; ++j) {
                    if (j >= count) break;
                    const std::uint32_t slot =
                        smem->routing.expert_row_slot[expert_start + j];
                    workspace->routed_down_out.row(slot)[out_col] =
                        static_cast<DataType::ElementOut>(acc_out[j]);
                }
                __syncwarp();
                if (lane_idx == 0u) {
                    shared_red_add_release_u32(&smem->down_task_stores, 1u);
                }
                KERNEL_TIMER_END(smem->timer(), epilogue_span);
            }
            KERNEL_TIMER_END(smem->timer(), task_span);
        }
    }

    // Publish warp, lane 0: fence each task's stores and count it into its
    // m-block.
    __device__ static void publish_down_tasks(
        MoEWorkspace<Dims>* __restrict__ workspace,
        MoESharedStorage<Dims>* __restrict__ smem,
        TaskScheduler<Traits>& sched) {
        TaskInfo task;
        while (sched.get_next_task(task)) {
            std::uint64_t done_inc = 1u;
            if constexpr (Traits::kCombineWaitPerToken) {
                done_inc = PackedTokenCounts<Dims>::increment(
                    smem->routing.token_mask(task.local_expert_idx));
            }
            // Polled at full rate: a backoff would delay the publish it
            // precedes.
            while (shared_load_acquire_u32(&smem->down_task_stores) <
                   WarpRole::kCalcWarpCount * (task.task_id + 1u)) {
            }
            const auto publish_span = KERNEL_TIMER_BEGIN(
                smem->timer(), "down_publish_task", task.local_expert_idx);
            workspace->cta_comm.publish_down_task(smem->launch_parity,
                                                  task.m_block_idx, done_inc);
            KERNEL_TIMER_END(smem->timer(), publish_span);
        }
    }

    // Per-expert scale loads into SMEM: warp 0 loads the weight scales, warp 1
    // the activation scales.
    __device__ static void load_weight_and_act_scale(
        SharedStorage* __restrict__ smem, Workspace* __restrict__ workspace,
        const typename DownGemm::ElementScale* __restrict__ expert_scales_down,
        std::uint32_t expert_id, std::uint32_t base_m,
        std::uint32_t expert_start, std::uint32_t routed_count) {
        const unsigned warp_idx = WarpRole::get_warp_idx();
        if constexpr (!DownGemm::MmaCore::kUseBlockscaled) {
            if (warp_idx == 0u) {
                const auto weight_scale_span = KERNEL_TIMER_BEGIN(
                    smem->timer(), "down_load_weight_scale", expert_id);
                load_down_weight_scale_per_expert<Dims>(
                    expert_scales_down, expert_id, base_m,
                    smem->routed_experts.w_down_scale);
                KERNEL_TIMER_END(smem->timer(), weight_scale_span);
            } else if (warp_idx == 1u) {
                // Every lane acquires: each lane loads up_out_scale itself.
                const auto up_out_span = KERNEL_TIMER_BEGIN(
                    smem->timer(), "down_wait_up_out", expert_id);
                workspace->cta_comm.wait_up_out(smem->launch_parity,
                                                expert_start, routed_count,
                                                Workspace::kUpOutReadyFull);
                KERNEL_TIMER_END(smem->timer(), up_out_span);
                const auto act_scale_span = KERNEL_TIMER_BEGIN(
                    smem->timer(), "down_load_act_scale", expert_id);
                load_down_act_scale<
                    Dims, DownGemm::kGemmK / Traits::UpGemm::kActQuantBlock,
                    DownGemm::kTileShapeN>(workspace, expert_start,
                                           routed_count,
                                           smem->routed_experts.act_down_scale);
                KERNEL_TIMER_END(smem->timer(), act_scale_span);
            }
        }
    }

    // Readout of one accumulator-ring slot into `acc_out`, scaled by the
    // 128-N substep's weight and activation scales.
    __device__ static void dequant(
        SharedStorage* __restrict__ smem, std::uint32_t ring_iter_id,
        cutlass::Array<float, DownGemm::kTileShapeN>& acc_out,
        unsigned m_tile_idx, unsigned lane_base, std::uint32_t k_iter_id,
        std::uint32_t tmem_base) {
        // Tile ring_iter_id - 1: the dequant lags the UMMA by one iteration.
        const std::uint32_t prev = (ring_iter_id - 1u) % kNumAccSlots;

        const auto acc_ready_span =
            KERNEL_TIMER_BEGIN(smem->timer(), "down_wait_accumulator_ready");
        smem->barriers.accumulator_ready.wait(ring_iter_id - 1u);
        KERNEL_TIMER_END(smem->timer(), acc_ready_span);

        if constexpr (DownGemm::MmaCore::kUseBlockscaled) {
            float v[DownGemm::kTileShapeN];
            tmem_ld_tokens(tmem_addr(tmem_base, lane_base,
                                     prev * kAccColumns +
                                         m_tile_idx * DownGemm::kTileShapeN),
                           v);
            cutlass::arch::fence_view_async_tmem_load();
            WarpRole::sync_calc_threads();
            if (WarpRole::get_lane_idx() == 0u) {
                smem->barriers.accumulator_empty.arrive(ring_iter_id - 1u);
            }
#pragma unroll
            for (std::uint32_t r = 0; r < DownGemm::kTileShapeN; ++r) {
                acc_out[r] += v[r];
            }
        } else {
            float v[kNumDownSubBlocks][DownGemm::kTileShapeN];
#pragma unroll
            for (std::uint32_t sb = 0; sb < kNumDownSubBlocks; ++sb) {
                tmem_ld_tokens(
                    tmem_addr(tmem_base, lane_base,
                              prev * kAccColumns +
                                  (m_tile_idx * kNumDownSubBlocks + sb) *
                                      DownGemm::kTileShapeN),
                    v[sb]);
            }
            cutlass::arch::fence_view_async_tmem_load();

            // Sync the scale load.
            WarpRole::sync_calc_threads();

            if (WarpRole::get_lane_idx() == 0u) {
                smem->barriers.accumulator_empty.arrive(ring_iter_id - 1u);
            }

            const std::uint32_t nb = k_iter_id - 1u;
            // Substep nb maps to activation block nb*128 / kActQuantBlock.
            const std::uint32_t a_blk =
                (nb * 128u) / Traits::UpGemm::kActQuantBlock;
            // One weight scale per m-tile; the whole 128-row tile shares a
            // block.
            const float ws =
                smem->routed_experts
                    .w_down_scale[m_tile_idx]
                                 [(kNumDownWeightScaleBlocksK > 1u) ? nb : 0u];
#pragma unroll
            for (std::uint32_t r = 0; r < DownGemm::kTileShapeN; ++r) {
                float acc = 0.f;
#pragma unroll
                for (std::uint32_t sb = 0; sb < kNumDownSubBlocks; ++sb) {
                    acc += v[sb][r] *
                           smem->routed_experts.act_down_scale[a_blk + sb][r];
                }
                acc_out[r] += acc * ws;
            }
        }
    }

    // Arm the weight ring: one TMA per K-step, gated on its slot's occupant.
    __device__ static void run_load_weights_k_loop(
        SharedStorage* __restrict__ smem,
        const CUtensorMap& tma_down_weights_desc,
        const typename DownGemm::ElementScale* __restrict__ down_scale,
        std::uint32_t id_base, std::uint32_t local_id, std::uint32_t expert_id,
        std::uint32_t base_m, std::uint64_t weight_cache_hint) {
        const std::uint32_t ring_iter_id = id_base + local_id;

        // Rolled, like the other down K loops (see run_mma_k_loop).
#pragma unroll 1
        for (std::uint32_t k_iter_id = 0; k_iter_id < kNumTilesK; ++k_iter_id) {
            const std::uint32_t slot = (local_id + k_iter_id) % kNumWeightSlots;
            wait_down_weight_empty(smem, id_base, ring_iter_id + k_iter_id,
                                   slot);
            prefetch_down_weights(smem, tma_down_weights_desc, down_scale,
                                  expert_id, base_m, slot, k_iter_id,
                                  weight_cache_hint);
        }
    }

    // UTCCP one K-step's SFA and SFB into their ring slot.  The caller has
    // already waited for the step's weight and activation arrivals, so this
    // pays only the copies -- and it runs under the previous step's MMA.
    __device__ static void stage_down_scales(SharedStorage* __restrict__ smem,
                                             std::uint32_t step_local) {
        // A whole warp under `elect_one_sync()`, not a lane test: behind one
        // the compiler wraps each UTCCP in a per-lane operand shuffle.
        if (!cute::elect_one_sync()) return;
        const std::uint32_t w_slot = step_local % kNumWeightSlots;
        std::uint8_t* const sfa = smem->routed_experts.down_weight_sfa[w_slot];
#pragma unroll
        for (std::uint32_t atom = 0; atom < DownScaleTraits::kNumSfaAtoms;
             ++atom) {
            DownScaleTraits::copy_sfa_to_tmem(
                sfa + atom * DownScaleTraits::kSfaAtomBytes,
                down_sfa_tmem(step_local) +
                    atom * DownScaleTraits::kSfaAtomColumns);
        }
        DownScaleTraits::copy_sfb_to_tmem(
            smem->routed_experts.act_down_sfb[w_slot],
            down_sfb_tmem(step_local));
    }

    // Arm the activation ring on its own warp, one TMA per K-step, gated on
    // its slot's previous occupant.
    __device__ static void run_load_activations_k_loop(
        SharedStorage* __restrict__ smem, Workspace* __restrict__ workspace,
        const CUtensorMap& tma_down_activations_desc, std::uint32_t id_base,
        std::uint32_t local_id, std::uint32_t expert_start,
        std::uint32_t routed_count) {
        // Acquire the routed_up_out readiness bitmask.
        const auto up_out_span =
            KERNEL_TIMER_BEGIN(smem->timer(), "down_wait_up_out", expert_start);
        workspace->cta_comm.wait_up_out(smem->launch_parity, expert_start,
                                        routed_count,
                                        Workspace::kUpOutReadyFull);
        KERNEL_TIMER_END(smem->timer(), up_out_span);

        // Rolled, like the other down K loops (see run_mma_k_loop).
#pragma unroll 1
        for (std::uint32_t k_iter_id = 0; k_iter_id < kNumTilesK; ++k_iter_id) {
            wait_activation_empty(smem, id_base + local_id + k_iter_id);
            prefetch_down_activation(smem, workspace, tma_down_activations_desc,
                                     expert_start, routed_count,
                                     local_id + k_iter_id, k_iter_id);
        }
    }

    // The MMA K-loop: one tcgen05.mma per 128-N substep.
    __device__ static void run_mma_k_loop(SharedStorage* __restrict__ smem,
                                          std::uint32_t tmem_base,
                                          std::uint32_t ring_iter_id,
                                          std::uint32_t local_id,
                                          std::uint32_t routed_count) {
        typename DownGemm::TiledMma tiled_mma{};
        auto cta_mma = tiled_mma.get_slice(cute::_0{});
        SBLayout sB_layout{};
        // Shape-only proxy: make_fragment_C only reads the layout.
        cute::Tensor tC_proxy = cta_mma.partition_C(cute::make_tensor(
            cute::make_gmem_ptr(static_cast<float*>(nullptr)),
            cute::make_layout(cute::make_shape(cute::_128{}, cute::_8{}),
                              cute::GenRowMajor{})));
        cute::Tensor tCt = cta_mma.make_fragment_C(tC_proxy);

        // Rolled: unrolled, each K step is its own copy of the scale staging
        // and UMMA issue, with every slot's descriptors held in registers, and
        // a launch with a cold L2 fetches all of that code from DRAM.
#pragma unroll 1
        for (std::uint32_t k_iter_id = 0; k_iter_id < kNumTilesK; ++k_iter_id) {
            const std::uint32_t step_id = ring_iter_id + k_iter_id;
            const std::uint32_t step_local = local_id + k_iter_id;
            const std::uint32_t acc_slot = step_id % kNumAccSlots;
            const std::uint32_t w_slot = step_local % kNumWeightSlots;
            // Previous occupant is step_id - kNumAccSlots: the previous
            // task's same step, or an up phase iteration.
            if (step_id >= kNumAccSlots) {
                const auto acc_empty_span = KERNEL_TIMER_BEGIN(
                    smem->timer(), "down_wait_accumulator_empty");
                smem->barriers.wait_accumulator_empty(step_id);
                KERNEL_TIMER_END(smem->timer(), acc_empty_span);
            }
            const auto weight_span =
                KERNEL_TIMER_BEGIN(smem->timer(), "wait_weight");
            smem->barriers.down_weight_ready.wait(step_local);
            KERNEL_TIMER_END(smem->timer(), weight_span);
            if (routed_count > 0u) {
                const auto act_span =
                    KERNEL_TIMER_BEGIN(smem->timer(), "wait_activation");
                smem->barriers.activation_ready.wait(step_local);
                KERNEL_TIMER_END(smem->timer(), act_span);
            }
            if constexpr (DownGemm::MmaCore::kUseBlockscaled) {
                stage_down_scales(smem, step_local);
            }
            mma(smem, tiled_mma, cta_mma, sB_layout, tCt, acc_slot, w_slot,
                step_local, tmem_base);
            // The commit covers every async tcgen05 op issued before it.
            cutlass::arch::umma_arrive(
                smem->barriers.accumulator_ready.raw(acc_slot));
        }
    }

    // The dequant lags the UMMA by one iteration, so run one past the K-steps
    // to drain the expert's last substep.
    __device__ static void run_dequant_k_loop(
        SharedStorage* __restrict__ smem,
        cutlass::Array<float, DownGemm::kTileShapeN>& acc_out,
        unsigned m_tile_idx, unsigned lane_base, std::uint32_t tmem_base,
        std::uint32_t ring_iter_id, std::uint32_t local_id) {
#pragma unroll 1
        for (std::uint32_t k_iter_id = 1; k_iter_id <= kNumTilesK;
             ++k_iter_id) {
            dequant(smem, ring_iter_id + k_iter_id, acc_out, m_tile_idx,
                    lane_base, k_iter_id, tmem_base);
        }
    }

    // UMMA issue for one 128-N substep into accumulator-ring slot `acc`.
    template <typename TiledMma, typename CtaMma, typename SBLayout,
              typename TCt>
    __device__ static void mma(SharedStorage* __restrict__ smem,
                               TiledMma& tiled_mma, CtaMma& cta_mma,
                               SBLayout const& sB_layout, TCt& tCt,
                               std::uint32_t acc, std::uint32_t cur_w_slot,
                               std::uint32_t w_slot_iter,
                               std::uint32_t tmem_base) {
        using WeightElement = typename DownGemm::WeightSmemElement;
        using ActElement = typename DownGemm::ActElement;

        auto tCsB =
            cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<ActElement*>(
                                  &smem->routed_experts.up_down_act
                                       .act_down_input[cur_w_slot][0])),
                              sB_layout);
        cute::Tensor tCrB = cta_mma.make_fragment_B(tCsB);
        if constexpr (DownGemm::MmaCore::kUseBlockscaled) {
            // The issuer staged this step's scales one step ago; tcgen05 ops
            // from one warp are ordered, so the MMA reads them directly.
            DownGemm::MmaCore::run(
                tiled_mma, cta_mma,
                reinterpret_cast<WeightElement*>(
                    &smem->routed_experts.w_down[cur_w_slot][0][0]),
                tCrB, tCt, tmem_addr(tmem_base, 0u, acc * kAccColumns),
                tmem_addr(tmem_base, 0u, down_sfa_tmem(w_slot_iter)),
                tmem_addr(tmem_base, 0u, down_sfb_tmem(w_slot_iter)));
        } else {
            // One sub-accumulator per activation quantization block, scaled
            // in `dequant`.
            DownGemm::MmaCore::run<kNumDownSubBlocks>(
                tiled_mma, cta_mma,
                reinterpret_cast<WeightElement*>(
                    &smem->routed_experts.w_down[cur_w_slot][0][0]),
                tCrB, tCt, tmem_addr(tmem_base, 0u, acc * kAccColumns));
        }
    }

    // Issue the weight TMAs for one (expert, N-step): one TMA per M instruction
    // (kNumAtomMs 128-row boxes), all armed on the step's weight mbarrier.
    __device__ static void prefetch_down_weights(
        SharedStorage* __restrict__ smem,
        const CUtensorMap& tma_down_weights_desc,
        const typename DownGemm::ElementScale* __restrict__ down_scale,
        std::uint32_t expert_id, std::uint32_t base_m, std::uint32_t slot,
        std::uint32_t k_iter_id, std::uint64_t weight_cache_hint) {
        smem->barriers.down_weight_ready.arrive(slot, kDownWeightTxBytes);
        // A 4-bit weight's coordinates count values too: the box is the same
        // 128 N (64 B of packed gmem), landing as the same SW128 box.
#pragma unroll
        for (std::uint32_t h = 0; h < DownGemm::MmaCore::kNumAtomMs; ++h) {
            constexpr std::uint32_t kAtomM = DownGemm::MmaCore::kAtomShapeM;
            // coord0 = n, coord1 = row of `[E, k, n]`.
            const std::uint32_t global_row =
                expert_id * DownGemm::kGemmM + base_m + h * kAtomM;
            tma_load_2d(tma_down_weights_desc,
                        /*coord0=*/k_iter_id * 128u,
                        /*coord1=*/global_row,
                        &smem->routed_experts.w_down[slot][h * kAtomM][0],
                        smem->barriers.down_weight_ready.raw(slot),
                        weight_cache_hint);
        }
        if constexpr (DownGemm::MmaCore::kUseBlockscaled) {
            // One 512-B SfAtom per 128 weight rows, spread across the host
            // table's row atoms.
            constexpr std::uint32_t kSfAtomBytes = 512u;
            constexpr std::uint32_t kSfKBlocksPerRow = DownGemm::kGemmK / 128u;
            constexpr std::uint32_t kSfAtomsPerTile =
                DownGemm::kTileShapeM / 128u;
            const std::uint32_t row_atom =
                expert_id * (DownGemm::kGemmM / 128u) + base_m / 128u;
#pragma unroll
            for (std::uint32_t atom = 0; atom < kSfAtomsPerTile; ++atom) {
                const std::uint8_t* const sfa =
                    reinterpret_cast<const std::uint8_t*>(down_scale) +
                    ((row_atom + atom) * kSfKBlocksPerRow + k_iter_id) *
                        kSfAtomBytes;
                tma_load_1d(sfa, kSfAtomBytes,
                            smem->routed_experts.down_weight_sfa[slot] +
                                atom * kSfAtomBytes,
                            smem->barriers.down_weight_ready.raw(slot));
            }
        }
    }

    // Issue the bulk activation TMA for one N-step; a zero-row expert arms
    // nothing.  A blockscaled step also stages its SFB, which this warp packs.
    __device__ static void prefetch_down_activation(
        SharedStorage* __restrict__ smem, Workspace* __restrict__ workspace,
        const CUtensorMap& tma_down_activations_desc,
        std::uint32_t expert_start, std::uint32_t routed_count,
        std::uint32_t step_local, std::uint32_t k_iter_id) {
        const std::uint32_t slot = step_local % kNumWeightSlots;
        if (routed_count > 0u) {
            if constexpr (DownGemm::MmaCore::kUseBlockscaled) {
                // Pack this K tile's SfAtom slab from the plain UE8M0 image.
                // A token's four 32-K groups are one word in both, and the MMA
                // reads only the tile's N rows: N word copies (padded rows
                // zero), every load issued before a store, since this gates
                // the step's activation TMA.
                constexpr std::uint32_t kGroups = DownGemm::kTileShapeK / 32u;
                static_assert(kGroups == 4u && DownGemm::kTileShapeN <= 32u,
                              "a token's K-tile scales are one SfAtom word");
                std::uint8_t* const sf =
                    smem->routed_experts.act_down_sfb[slot];
                std::uint32_t words[DownGemm::kTileShapeN];
#pragma unroll
                for (std::uint32_t row = 0; row < DownGemm::kTileShapeN;
                     ++row) {
                    words[row] =
                        row < routed_count
                            ? *reinterpret_cast<const std::uint32_t*>(
                                  workspace->routed_up_out.mx_scale_row(
                                      expert_start + row) +
                                  k_iter_id * kGroups)
                            : 0u;
                }
#pragma unroll
                for (std::uint32_t row = 0; row < DownGemm::kTileShapeN;
                     ++row) {
                    *reinterpret_cast<std::uint32_t*>(
                        sf + sf_atom_offset(row, 0u)) = words[row];
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            }
            smem->barriers.activation_ready.arrive(slot, kDownActTxBytes);
            tma_load_2d(
                tma_down_activations_desc, /*coord0=*/k_iter_id * 128u,
                /*coord1=*/expert_start,
                &smem->routed_experts.up_down_act.act_down_input[slot][0],
                smem->barriers.activation_ready.raw(slot));
        }
    }

    // Hold arm `arm_id` until its previous occupant, arm_id -
    // kNumActivationSlots, has been read by its UMMA; no first-lap case.
    __device__ static void wait_activation_empty(
        SharedStorage* __restrict__ smem, std::uint32_t arm_id) {
        constexpr std::uint32_t kSlots = MoEBarriers<Dims>::kNumActivationSlots;
        if (arm_id < kSlots) return;
        const auto act_empty_span = KERNEL_TIMER_BEGIN(
            smem->timer(), "down_wait_activation_empty", arm_id);
        smem->barriers.accumulator_ready.wait(arm_id - kSlots);
        KERNEL_TIMER_END(smem->timer(), act_empty_span);
    }

    // Hold a down weight slot until its previous occupant is read: the down
    // step a ring back, or in the first lap the aliased up steps.
    __device__ static void wait_down_weight_empty(
        SharedStorage* __restrict__ smem, std::uint32_t id_base,
        std::uint32_t ring_iter_id, std::uint32_t slot) {
        if (ring_iter_id - id_base >= kNumWeightSlots) {
            const auto weight_empty_span = KERNEL_TIMER_BEGIN(
                smem->timer(), "down_wait_weight_empty", ring_iter_id);
            smem->barriers.accumulator_ready.wait(ring_iter_id -
                                                  kNumWeightSlots);
            KERNEL_TIMER_END(smem->timer(), weight_empty_span);
            return;
        }
        // A block with no up task never had its w_up read.
        if (id_base == 0u) return;
        const auto up_umma_span =
            KERNEL_TIMER_BEGIN(smem->timer(), "down_wait_up_umma_done", slot);
#pragma unroll
        for (std::uint32_t h = 0; h < Traits::kUpSlotsPerDownSlot; ++h) {
            smem->barriers.accumulator_ready.wait(
                id_base - kNumUpWeightSlots +
                Traits::kUpSlotsPerDownSlot * slot + h);
        }
        KERNEL_TIMER_END(smem->timer(), up_umma_span);
        // The mbarrier wait does not order SMEM reads against TMA writes.
        asm volatile("fence.proxy.async;" ::: "memory");
    }
};

}  // namespace halops::moe
