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

#include <cute/algorithm/gemm.hpp>
#include <cute/atom/mma_traits_sm100.hpp>
#include <cute/tensor.hpp>

#include "kernel/moe/mma_core.cuh"
#include "kernel/moe/moe_configs.cuh"
#include "kernel/moe/moe_cta_comm.cuh"
#include "kernel/moe/moe_input.cuh"
#include "kernel/moe/moe_scheduler.cuh"
#include "kernel/moe/moe_shared_storage.cuh"
#include "kernel/moe/moe_tma.cuh"
#include "kernel/moe/moe_traits.cuh"
#include "kernel/moe/moe_workspace.cuh"
#include "kernel/moe/ptx_utils.cuh"

namespace halops::moe {

// Soft-cap parameters of the up-projection expert activation.
struct GatedActParams {
    float beta;
    float inv_beta;
    float linear_beta;
    float inv_linear_beta;

    __host__ __device__ explicit GatedActParams(float gate_scale,
                                                float up_scale) {
        beta = gate_scale;
        inv_beta = (gate_scale > 0.0f) ? (1.0f / gate_scale) : 0.0f;
        linear_beta = up_scale;
        inv_linear_beta = (up_scale > 0.0f) ? (1.0f / up_scale) : 0.0f;
    }
};

// tanh(z) = 1 - 2/(1 + e^{2z}), safe at both saturating ends without a branch.
__device__ __forceinline__ float moe_tanh_fast(float z) {
    return 1.0f - 2.0f / (1.0f + __expf(2.0f * z));
}

// Gated expert activation: combine one (gate, up) pair with the routing weight.
template <bool kSitu>
__device__ __forceinline__ float moe_gated_act(float gate, float up, float rw,
                                               GatedActParams const& p) {
    if constexpr (!kSitu) {
        return rw * up * gate / (1.0f + __expf(-gate));
    } else {
        const float sigmoid_gate = 1.0f / (1.0f + __expf(-gate));
        const float capped_gate =
            (p.beta > 0.0f)
                ? p.beta * moe_tanh_fast(gate * p.inv_beta) * sigmoid_gate
                : gate * sigmoid_gate;
        const float capped_up =
            (p.linear_beta > 0.0f)
                ? p.linear_beta * moe_tanh_fast(up * p.inv_linear_beta)
                : up;
        return rw * capped_gate * capped_up;
    }
}

// Async (cp.async) per-expert up-projection block-scale prefetch.
template <typename Dims>
__device__ inline void load_up_weight_scale_per_expert_async(
    const DataType::ElementScale* __restrict__ expert_scales_up,
    std::uint32_t expert_id, unsigned base_row_up,
    DataType::ElementScale* __restrict__ dest) {
    const unsigned lane_idx = WarpRole::get_lane_idx();
    constexpr uint32_t kNumScaleBlocksK =
        MoEKernelTraits<Dims>::UpGemm::kNumScaleBlocksK;
    constexpr uint32_t kNumScaleBlocksM =
        MoEKernelTraits<Dims>::UpGemm::kNumScaleBlocksM;

    for (uint32_t i = lane_idx; i < 2 * kNumScaleBlocksK; i += 32u) {
        // 0 → gate half, 1 → up half
        uint32_t rb_local = i / kNumScaleBlocksK;
        uint32_t kb = i % kNumScaleBlocksK;
        uint32_t row = base_row_up + rb_local * Dims::kIntermediateSize;
        uint32_t rb_global = row / Dims::kQuantBlockM;
        const DataType::ElementScale* src =
            &expert_scales_up[expert_id * kNumScaleBlocksM * kNumScaleBlocksK +
                              rb_global * kNumScaleBlocksK + kb];
        cp_async_4bytes(&dest[i], src);
    }
    // Commit from the issuing warp; other warps' empty commits add a group.
    cp_async_commit_group();
}

// Up-projection for the SM100 (tcgen05 / UMMA) backend.
template <typename KernelTraits>
struct UpProjection {
    using Traits = KernelTraits;
    using Dims = typename Traits::MoEDims;
    using SharedStorage = MoESharedStorage<Dims>;
    using Workspace = MoEWorkspace<Dims>;
    using RoutingTables = MoERoutingTables<Dims>;
    using UpGemm = typename Traits::UpGemm;

    static constexpr std::uint32_t kNumTilesK = UpGemm::kNumTilesK;
    static constexpr std::uint32_t kNumWeightSlots = UpGemm::kNumWeightSlots;
    static constexpr std::uint32_t kNumAccSlots = Traits::kNumAccSlots;
    // TMEM columns one accumulator-ring slot occupies.
    static constexpr std::uint32_t kAccColumns = UpGemm::kAccColumns;
    static constexpr std::uint32_t kNumUpSubBlocks = UpGemm::kNumSubBlocks;
    static constexpr std::uint32_t kNumUpScaleBlocksK =
        UpGemm::kNumScaleBlocksK;
    static constexpr std::uint32_t kPairLayoutGranularity =
        Traits::kPairLayoutGranularity;
    // Register accumulators of one dequant thread, as [feature][token].
    static constexpr std::uint32_t kAccFeatures = 1u;
    static constexpr std::uint32_t kAccTokens = UpGemm::kTileShapeN / 4u;
    static constexpr std::uint32_t kQuantGroups = UpGemm::kActQuantBlock / 32u;
    static_assert(UpGemm::kTileShapeN % WarpRole::kCalcWarpCount == 0u,
                  "every calc warp must publish the same number of tokens.");
    // B-operand strides: SWIZZLE_NONE, 9 token rows (144 B) per 16-B chunk.
    static constexpr std::uint64_t kBLbo =
        static_cast<std::uint64_t>(UpGemm::kTileShapeN + 1u) *
        static_cast<std::uint64_t>(Traits::kLboChunkK);
    static constexpr std::uint32_t kBKStride =
        2u * static_cast<std::uint32_t>(kBLbo);
    static constexpr std::uint32_t kUpWeightTxBytes =
        UpGemm::MmaCore::kWeightTileBytes;

    // B operand: the padded `quantized_input` staging tile, [8 chunks][N + 1
    // token rows][16 B] with SWIZZLE_NONE, written out explicitly.
    using SBLayout = decltype(cute::make_layout(
        cute::make_shape(
            cute::make_shape(cute::Int<UpGemm::kTileShapeN>{},
                             cute::make_shape(cute::_16{}, cute::_2{})),
            cute::_1{}, cute::Int<UpGemm::MmaCore::kNumAtomKs>{}),
        cute::make_stride(
            cute::make_stride(
                cute::_16{}, cute::make_stride(cute::_1{}, cute::Int<kBLbo>{})),
            cute::_0{}, cute::Int<kBKStride>{})));
    static_assert(cute::stride<0, 0>(SBLayout{}) == 16,
                  "B: row stride must be 16 B.");
    static_assert(cute::get<1>(cute::stride<0, 1>(SBLayout{})) == kBLbo,
                  "B: LBO mismatch.");
    static_assert(cute::stride<2>(SBLayout{}) == kBKStride,
                  "B: K=32 step mismatch.");

    static_assert(Dims::kNumTokens <= 16, "SM100 up-proj supports BS<=16.");
    static_assert(kPairLayoutGranularity == 8u,
                  "The 16x256b readout pairs gate/up rows eight rows apart.");
    // This warp's first TMEM row; a warp reads only its own 32-lane block.
    __device__ static __forceinline__ std::uint32_t dequant_row_base() {
        const std::uint32_t warp_idx = WarpRole::get_warp_idx();
        return 32u * (warp_idx & 3u) + 16u * (warp_idx >> 2);
    }
    // First feature of that 16-row group; a group holds 8 features.
    __device__ static __forceinline__ std::uint32_t dequant_feature_base() {
        return 8u * (dequant_row_base() >> 4);
    }
    static_assert(UpGemm::kActQuantBlock * Dims::kNumTokens ==
                      WarpRole::kCalcWarpCount * 32u * kAccFeatures *
                          kAccTokens,
                  "The pair-layout epilogue's register file must cover the "
                  "block's features x kNumTokens exactly.");
    static_assert(
        kNumWeightSlots <= kNumAccSlots,
        "The weight ring must not outrun the accumulator ring, or the "
        "re-arm's wait_accumulator_ready(arm_id - kNumWeightSlots) "
        "can read a phase the ring has already wrapped past.");

    // Run up-projection over this block's expert sequence.
    __device__ static void run(
        const DataType::ElementScale* __restrict__ expert_scales_up,
        std::uint32_t batch_size, Activation activation,
        GatedActParams const& act_params,
        MoEWorkspace<Dims>* __restrict__ workspace,
        MoESharedStorage<Dims>* __restrict__ smem,
        CUtensorMap const& tma_up_weights_desc, TaskScheduler<Traits>& sched,
        std::uint64_t weight_cache_hint) {
        const unsigned warp_idx = WarpRole::get_warp_idx();
        const unsigned lane_idx = WarpRole::get_lane_idx();

        const std::uint32_t m_block = sched.cta_group().cta_idx_in_group;
        const std::uint32_t base_feature = m_block * UpGemm::kActQuantBlock;
        // Row base in the interleaved [E, 2N, K] tensor, one tile per K-step.
        const std::uint32_t weight_row_base = m_block * UpGemm::kTileShapeM;
        const std::uint32_t tmem_base = smem->tmem_ptr;

        const std::uint32_t first_e = sched.first_expert_idx();

        constexpr std::uint32_t kScaleSlots =
            RoutedExpertsSmem<Dims>::kNumUpScaleSlots;

        if (WarpRole::is_other_loader_warp(warp_idx) && sched.is_active()) {
            prefetch_up_weight_scale(expert_scales_up,
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

            if (WarpRole::is_weight_loader_thread()) {
                run_load_k_loop(smem, tma_up_weights_desc, weight_row_base,
                                task_id, expert_id, weight_cache_hint);
            } else if (WarpRole::is_mma_issuer_warp(warp_idx)) {
                run_mma_k_loop(smem, tmem_base, task_id);
            } else if (WarpRole::is_calc_warp(warp_idx)) {
                ZeroedFragment<kAccFeatures, kAccTokens> acc_gate_out;
                ZeroedFragment<kAccFeatures, kAccTokens> acc_up_out;
                run_dequant_k_loop(smem, acc_gate_out, acc_up_out,
                                   cur_scale_slot, tmem_base, task_id,
                                   expert_id);

                epilogue(workspace, smem, task.local_expert_idx, batch_size,
                         base_feature, m_block, activation, act_params,
                         acc_gate_out, acc_up_out);
            } else if (WarpRole::is_other_loader_warp(warp_idx) && has_next_e) {
                const std::uint32_t next_scale_slot =
                    (task_id + 1u) % kScaleSlots;
                const std::uint32_t next_expert_id =
                    has_next_e
                        ? smem->routing.active_experts[task.next_expert_idx]
                        : 0u;
                if (task_id + 1u >= kScaleSlots) {
                    // Throttle edge: wait for calc warps to release the slot.
                    smem->barriers.wait_weight_scale_empty(
                        task_id + 1u - kScaleSlots, next_scale_slot);
                }
                prefetch_up_weight_scale(expert_scales_up, next_expert_id,
                                         base_feature, next_scale_slot, smem);
            }
        }
    }

    // Epilogue: gated activation (register-local) -> act_up_out.
    template <bool kSitu>
    __device__ static void combine_gated_activation(
        SharedStorage* __restrict__ smem, std::uint32_t e,
        std::uint32_t batch_size,
        ZeroedFragment<kAccFeatures, kAccTokens>& acc_gate,
        ZeroedFragment<kAccFeatures, kAccTokens>& acc_up,
        GatedActParams const& act_params) {
        const std::uint32_t lane_idx = WarpRole::get_lane_idx();
        // Pair layout: `t0` is the column pair, `t1` the row in a group.
        const std::uint32_t t0 = lane_idx & 3u;
        const std::uint32_t t1 = lane_idx >> 2;
        const RoutingTables& tables = smem->routing;
        const std::uint32_t slab = tables.slab_begin(e);
#pragma unroll
        for (std::uint32_t f = 0; f < kAccFeatures; ++f) {
#pragma unroll
            for (std::uint32_t ti = 0; ti < kAccTokens; ++ti) {
                const std::uint32_t tok = 8u * (ti / 2u) + 2u * t0 + ti % 2u;
                if (tok < batch_size) {
                    // `rank_of` returns 0xFF when the token misses this expert.
                    const std::uint8_t rank = tables.rank_of(e, tok);
                    if (rank != 0xFFu) {
                        const float rw = tables.expert_row_weight[slab + rank];
                        // Rows t1 and t1 + 8 are one feature's gate and up.
                        const std::uint32_t frow =
                            dequant_feature_base() + t1 + 8u * f;
                        smem->routed_experts.up_down_act.act_up_out[frow][tok] =
                            moe_gated_act<kSitu>(acc_gate[f][ti], acc_up[f][ti],
                                                 rw, act_params);
                    }
                }
            }
        }
    }

    // Reduce |.| over the block's features, quantize and publish one row.
    __device__ static void quantize_publish_rows(
        Workspace* __restrict__ workspace, SharedStorage* __restrict__ smem,
        std::uint32_t e, std::uint32_t batch_size, std::uint32_t base_feature,
        std::uint32_t m_block_idx, std::uint32_t tok) {
        const std::uint32_t lane_idx = WarpRole::get_lane_idx();
        bool store = false;
        std::uint32_t dest_row = 0u;
        if (tok < batch_size) {
            const std::uint8_t rank = smem->routing.rank_of(e, tok);
            if (rank != 0xFFu) {
                store = true;
                dest_row = smem->routing.slab_begin(e) + rank;
            }
        }

        if (!store) {
            return;
        }

        float v[kQuantGroups];
        float local_max = 0.f;
#pragma unroll
        for (std::uint32_t i = 0; i < kQuantGroups; ++i) {
            v[i] = smem->routed_experts.up_down_act
                       .act_up_out[lane_idx + i * 32u][tok];
            local_max = fmaxf(local_max, fabsf(v[i]));
        }
        // local_max is finite and >= 0, so its bit pattern orders like the
        // value: one integer warp max replaces the fmaxf shuffle tree.
        float block_max = __uint_as_float(
            __reduce_max_sync(0xffffffffu, __float_as_uint(local_max)));
        // Eps clamp 1e-10 (vLLM's): a tiny block_max overflows the fp8 cast.
        block_max = fmaxf(block_max, 1e-10f);
        constexpr float kFp8Max = 448.0f;
        const float block_scale = block_max * (1.0f / kFp8Max);
        const float inv_scale = kFp8Max / block_max;
#pragma unroll
        for (std::uint32_t i = 0; i < kQuantGroups; ++i) {
            const std::uint32_t col = base_feature + lane_idx + i * 32u;
            workspace->routed_up_out.row(dest_row)[col] =
                static_cast<DataType::ElementQA>(v[i] * inv_scale);
        }
        // Scale store (data), then release the readiness bit (sync).
        if (lane_idx == 0u) {
            workspace->routed_up_out.scale_row(dest_row)[m_block_idx] =
                block_scale;
        }
        workspace->cta_comm.publish_up_out(smem->launch_parity, dest_row,
                                           m_block_idx, lane_idx);
    }

    // Epilogue: the gated activation into act_up_out, then one warp per token
    // reducing and publishing its quantization block.
    __device__ static void epilogue(
        Workspace* __restrict__ workspace, SharedStorage* __restrict__ smem,
        std::uint32_t e, std::uint32_t batch_size, std::uint32_t base_feature,
        std::uint32_t m_block_idx, Activation activation,
        GatedActParams const& act_params,
        ZeroedFragment<kAccFeatures, kAccTokens>& acc_gate,
        ZeroedFragment<kAccFeatures, kAccTokens>& acc_up) {
        [[maybe_unused]] const std::uint32_t expert_id =
            smem->routing.active_experts[e];

        if (activation == Activation::kSitu) {
            combine_gated_activation<true>(smem, e, batch_size, acc_gate,
                                           acc_up, act_params);
        } else {
            combine_gated_activation<false>(smem, e, batch_size, acc_gate,
                                            acc_up, act_params);
        }
        WarpRole::sync_calc_threads();

        constexpr std::uint32_t kTokensPerWarp =
            UpGemm::kTileShapeN / WarpRole::kCalcWarpCount;
#pragma unroll 1
        for (std::uint32_t g = 0; g < kTokensPerWarp; ++g) {
            quantize_publish_rows(
                workspace, smem, e, batch_size, base_feature, m_block_idx,
                WarpRole::get_warp_idx() + g * WarpRole::kCalcWarpCount);
        }
        WarpRole::sync_calc_threads();
    }

    // The launcher: arm ring id r with K-iteration r's tile, one ring ahead.
    __device__ static void run_load_k_loop(
        SharedStorage* __restrict__ smem,
        const CUtensorMap& tma_up_weights_desc, std::uint32_t weight_row_base,
        std::uint32_t task_id, std::uint32_t expert_id,
        std::uint64_t weight_cache_hint) {
#pragma unroll
        for (std::uint32_t k_iter_id = 0; k_iter_id < kNumTilesK; ++k_iter_id) {
            const std::uint32_t ring_iter_id = task_id * kNumTilesK + k_iter_id;
            wait_weight_empty(smem, ring_iter_id);
            prefetch_up_weights(smem, tma_up_weights_desc, ring_iter_id,
                                expert_id, k_iter_id, weight_row_base,
                                weight_cache_hint);
        }
    }

    // The issuer: one tcgen05.mma per K-iteration, on this warp only.
    __device__ static void run_mma_k_loop(SharedStorage* __restrict__ smem,
                                          std::uint32_t tmem_base,
                                          std::uint32_t task_id) {
        typename UpGemm::TiledMma tiled_mma{};
        auto cta_mma = tiled_mma.get_slice(cute::_0{});
        SBLayout sB_layout{};
        // Shape-only proxy: make_fragment_C only reads the layout.
        cute::Tensor tC_proxy = cta_mma.partition_C(cute::make_tensor(
            cute::make_gmem_ptr(static_cast<float*>(nullptr)),
            cute::make_layout(cute::make_shape(cute::_128{}, cute::_8{}),
                              cute::GenRowMajor{})));
        cute::Tensor tCt_gate = cta_mma.make_fragment_C(tC_proxy);
        cute::Tensor tCt_up = cta_mma.make_fragment_C(tC_proxy);

#pragma unroll 1
        for (std::uint32_t k_iter_id = 0; k_iter_id < kNumTilesK; ++k_iter_id) {
            const std::uint32_t ring_iter_id = task_id * kNumTilesK + k_iter_id;
            const std::uint32_t acc_slot = ring_iter_id % kNumAccSlots;
            const std::uint32_t weight_slot = ring_iter_id % kNumWeightSlots;
            // Previous occupant is ring_iter_id - kNumAccSlots; none on the
            // launch's first lap.
            if (ring_iter_id >= kNumAccSlots) {
                smem->barriers.wait_accumulator_empty(ring_iter_id);
            }
            smem->barriers.up_weight_ready.wait(ring_iter_id);
            if (task_id == 0u) {
                // Staging lands one K tile at a time; later tasks are ready.
                smem->barriers.up_staging_ready.wait(k_iter_id);
            }
            mma(smem, tiled_mma, cta_mma, sB_layout, tCt_gate, tCt_up,
                weight_slot, acc_slot, k_iter_id, tmem_base);
            // The commit covers every async tcgen05 op issued before it.
            cutlass::arch::umma_arrive(
                smem->barriers.accumulator_ready.raw(acc_slot));
        }
    }

    // The calc warps' loop lags the UMMA by one iteration and drains one past
    // the K-steps.
    __device__ static void run_dequant_k_loop(
        SharedStorage* __restrict__ smem,
        ZeroedFragment<kAccFeatures, kAccTokens>& acc_gate_out,
        ZeroedFragment<kAccFeatures, kAccTokens>& acc_up_out,
        std::uint32_t cur_scale_slot, std::uint32_t tmem_base,
        std::uint32_t task_id, std::uint32_t expert_id) {
        smem->barriers.wait_weight_scale_ready(task_id, cur_scale_slot);

#pragma unroll 1
        for (std::uint32_t k_iter_id = 1; k_iter_id <= kNumTilesK;
             ++k_iter_id) {
            dequant(smem, acc_gate_out, acc_up_out, cur_scale_slot,
                    task_id * kNumTilesK + k_iter_id, k_iter_id, tmem_base);
        }

        WarpRole::sync_calc_threads();
        if (WarpRole::get_lane_idx() == 0u) {
            smem->barriers.weight_scale_empty.arrive(cur_scale_slot);
        }
    }

    // UMMA issue for one K-iteration into the slot's TMEM columns.
    template <typename TiledMma, typename CtaMma, typename SBLayout,
              typename TCt>
    __device__ static void mma(SharedStorage* __restrict__ smem,
                               TiledMma& tiled_mma, CtaMma& cta_mma,
                               SBLayout const& sB_layout, TCt& tCt_gate,
                               TCt& tCt_up, std::uint32_t weight_slot,
                               std::uint32_t acc_slot, std::uint32_t s,
                               std::uint32_t tmem_base) {
        using MmaElement = typename UpGemm::MmaElement;

        auto tCsB = cute::make_tensor(
            cute::make_smem_ptr(reinterpret_cast<MmaElement*>(
                &smem->routed_experts.quantized_input.data[s][0][0][0])),
            sB_layout);
        cute::Tensor tCrB = cta_mma.make_fragment_B(tCsB);
        // One 128-row slab of interleaved gate/up rows; one sub-accumulator
        // per activation quantization block.
        UpGemm::MmaCore::run<kNumUpSubBlocks>(
            tiled_mma, cta_mma,
            reinterpret_cast<MmaElement*>(
                &smem->routed_experts.w_up[weight_slot][0]),
            tCrB, tCt_gate, tmem_addr(tmem_base, 0u, acc_slot * kAccColumns));
    }

    __device__ static void dequant(
        SharedStorage* __restrict__ smem,
        ZeroedFragment<kAccFeatures, kAccTokens>& acc_gate,
        ZeroedFragment<kAccFeatures, kAccTokens>& acc_up,
        std::uint32_t cur_scale_slot, std::uint32_t ring_iter_id,
        std::uint32_t s, std::uint32_t tmem_base) {
        const std::uint32_t lane_idx = WarpRole::get_lane_idx();
        const std::uint32_t prev_slot = (ring_iter_id - 1u) % kNumAccSlots;

        // Tile ring_iter_id - 1: the dequant lags the UMMA by one iteration.
        smem->barriers.accumulator_ready.wait(ring_iter_id - 1u);

        const auto release_slot = [&]() {
            cutlass::arch::fence_view_async_tmem_load();
            __syncwarp();
            if (lane_idx == 0u) {
                smem->barriers.accumulator_empty.arrive(ring_iter_id - 1u);
            }
        };

        const DataType::ElementScale* ws_buf =
            smem->routed_experts.w_up_scale[cur_scale_slot];

        // One 16-row group per warp; rows t1 and t1 + 8 are gate and up.
        const std::uint32_t t0 = lane_idx & 3u;
        const std::uint32_t lane_base = dequant_row_base();
#pragma unroll
        for (std::uint32_t sub = 0; sub < kNumUpSubBlocks; ++sub) {
            // Each sub-accumulator is one activation quant block; kblk is its
            // global index into both scale tables.
            const std::uint32_t kblk = (s - 1u) * kNumUpSubBlocks + sub;
#pragma unroll
            for (std::uint32_t g = 0; g < kAccTokens / 2u; ++g) {
                float v[4];
                tmem_ld_16x256b1x(
                    tmem_addr(tmem_base, lane_base,
                              prev_slot * kAccColumns +
                                  sub * UpGemm::kTileShapeN + g * 8u),
                    v);
#pragma unroll
                for (std::uint32_t ti = 0; ti < 2u; ++ti) {
                    // v[0..1] is row t1 (gate), v[2..3] is row t1+8 (up).
                    const float as = smem->routed_experts.quantized_input
                                         .scale[kblk][g * 8u + 2u * t0 + ti];
                    acc_gate[0][g * 2u + ti] += v[ti] * ws_buf[kblk] * as;
                    acc_up[0][g * 2u + ti] +=
                        v[ti + 2u] * ws_buf[kblk + kNumUpScaleBlocksK] * as;
                }
            }
            if (sub + 1u == kNumUpSubBlocks) release_slot();
        }
    }

    // Fetch one expert's block-scale tile into `slot` and publish it.
    __device__ static void prefetch_up_weight_scale(
        const DataType::ElementScale* __restrict__ expert_scales_up,
        std::uint32_t expert_id, unsigned base_row_up, std::uint32_t slot,
        SharedStorage* __restrict__ smem) {
        load_up_weight_scale_per_expert_async<Dims>(
            expert_scales_up, expert_id, base_row_up,
            smem->routed_experts.w_up_scale[slot]);
        cp_async_wait_group<0>();
        __syncwarp();
        if (WarpRole::get_lane_idx() == 0u) {
            smem->barriers.weight_scale_ready.arrive(slot);
        }
    }

    // Issue the weight TMA(s) for one (expert, K-tile) into its ring slot.
    __device__ static void prefetch_up_weights(
        SharedStorage* __restrict__ smem,
        const CUtensorMap& tma_up_weights_desc, std::uint32_t ring_iter_id,
        std::uint32_t expert_id, std::uint32_t k_tile,
        std::uint32_t weight_row_base, std::uint64_t weight_cache_hint) {
        const std::uint32_t slot = ring_iter_id % kNumWeightSlots;
        smem->barriers.up_weight_ready.arrive(slot, kUpWeightTxBytes);
        // SWIZZLE_128B caps a box at 128 B of K; a wider tile is many boxes.
        using MmaCore = typename UpGemm::MmaCore;
        const std::uint32_t k_start = k_tile * MmaCore::kTileShapeK;
#pragma unroll 1
        for (std::uint32_t h = 0; h < MmaCore::kNumSwizzleBoxKs; ++h) {
            tma_load_up_tcgen05_pair_tile(
                tma_up_weights_desc, expert_id, Dims::kIntermediateSize,
                weight_row_base, k_start + h * MmaCore::kSwizzleBoxK,
                &smem->routed_experts.w_up[slot][h][0],
                smem->barriers.up_weight_ready.raw(slot), weight_cache_hint);
        }
    }

    // Hold arm `arm_id` until its previous occupant, arm_id - kNumWeightSlots,
    // has been read by its UMMA; the first lap has no previous occupant.
    __device__ static void wait_weight_empty(SharedStorage* __restrict__ smem,
                                             std::uint32_t arm_id) {
        if (arm_id < kNumWeightSlots) {
            return;
        }
        smem->barriers.accumulator_ready.wait(arm_id - kNumWeightSlots);
    }
};

}  // namespace halops::moe
