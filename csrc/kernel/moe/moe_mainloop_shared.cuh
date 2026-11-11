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
#include "kernel/moe/moe_mainloop_expert_base.cuh"
#include "kernel/moe/moe_shared_storage.cuh"
#include "kernel/moe/moe_tma.cuh"
#include "kernel/moe/moe_traits.cuh"
#include "kernel/moe/moe_workspace.cuh"
#include "kernel/moe/ptx_utils.cuh"
#include "kernel/moe/quant_utils.cuh"

namespace halops::moe {

template <typename KernelTraits>
struct SharedUpProjectionBase
    : UpProjectionBase<KernelTraits,
                       SharedUpGemmTraits<typename KernelTraits::MoEDims>> {
    using Base =
        UpProjectionBase<KernelTraits,
                         SharedUpGemmTraits<typename KernelTraits::MoEDims>>;
    using typename Base::Accumulator;
    using typename Base::Dims;
    using typename Base::SBLayout;
    using typename Base::SharedStorage;
    using typename Base::Traits;
    using typename Base::Workspace;
    using SharedUpGemm = typename Base::UpGemm;
    using SharedDownGemm = SharedDownGemmTraits<Dims>;

    static constexpr std::uint32_t kNumDownCtas = SharedDownGemm::kNumCtas;
    static constexpr std::uint32_t kNumUpCtas = SharedUpGemm::kNumCtas;
    static constexpr std::uint32_t kFirstCtaIdx =
        Traits::RouterGemm::kNumCtas + Traits::LatentDownGemm::kNumCtas;
    static constexpr std::uint32_t kNumUpPublishers =
        kNumUpCtas / Base::kNumSplits;
    static constexpr std::uint32_t kCluster = Traits::kClusterSize;

    static constexpr std::uint32_t kUpTilesK =
        SharedUpGemm::kNumTilesK / Base::kNumSplits;
    static constexpr std::uint32_t kQuantBlocksPerCta =
        SharedUpGemm::kGemmK / Base::kNumSplits / 128u;
    static constexpr std::uint32_t kLboChunk =
        Workspace::SharedUpOut::kLboChunk;
    static constexpr std::uint32_t kNumUpSlots = Base::kNumWeightSlots;
    static constexpr std::uint32_t kInputSwizzleBoxK =
        decltype(SharedExpertSmem<SharedUpGemm,
                                  SharedDownGemm>::input)::kSwizzleBoxK;
    // TMA byte count for the staged BF16 input tile.
    static constexpr std::uint32_t kInputTileTxBytes =
        SharedUpGemm::kTileShapeN * SharedUpGemm::kTileShapeK *
        static_cast<std::uint32_t>(sizeof(DataType::ElementA));
    static_assert(kUpTilesK * SharedUpGemm::kNumStagingBlocks ==
                      kQuantBlocksPerCta,
                  "the block's K tiles must be its quantization blocks");
    static_assert(kNumUpSlots <= kUpTilesK,
                  "the weight ring must fit the block's K tiles");

    __device__ static std::uint32_t cta_idx() {
        return blockIdx.x - kFirstCtaIdx;
    }

    __device__ static bool is_shared_cta() {
        return blockIdx.x >= kFirstCtaIdx &&
               blockIdx.x < kFirstCtaIdx + kNumDownCtas;
    }

    __device__ static std::uint32_t slab_idx() {
        return cta_idx() / Base::kNumSplits;
    }

    __device__ static std::uint32_t split_idx() {
        return cta_idx() % Base::kNumSplits;
    }

    __device__ static void publish_up_output(Workspace* __restrict__ workspace,
                                             SharedStorage* __restrict__ smem) {
        KERNEL_TIMER_SCOPE(smem->timer.get(), "shared_publish_up_out");
        workspace->cta_comm.publish_shared_up_out(smem->launch_parity, 1u);
    }

    __device__ static void combine_gated_activation(
        SharedStorage* __restrict__ smem, Activation activation,
        float situ_beta, float situ_linear_beta, const Accumulator& gate,
        const Accumulator& up) {
        auto& partial = smem->shared_expert.splited_mma_out;
        Base::join_up_split_k(partial, gate, up);
        if (split_idx() == 0u &&
            WarpRole::is_calc_warp(WarpRole::get_warp_idx())) {
            KERNEL_TIMER_SCOPE(smem->timer.get(),
                               "shared_combine_gate_activation");
            Base::combine_gated_activation(
                partial, smem->shared_expert.act_up_out, gate, up, activation,
                situ_beta, situ_linear_beta);
            WarpRole::sync_calc_threads();
        }
    }

    __device__ static void run_load_input_k_loop(
        CUtensorMap const& tma_input_desc, SharedStorage* __restrict__ smem) {
        const std::uint32_t k_tile_base = split_idx() * kUpTilesK;

#pragma unroll 1
        for (std::uint32_t k_iter_id = 0; k_iter_id < kUpTilesK; ++k_iter_id) {
            const std::uint32_t slot = k_iter_id % kNumUpSlots;
            const auto empty_span = KERNEL_TIMER_BEGIN(
                smem->timer.get(), "shared_wait_up_input_empty", k_iter_id);
            smem->barriers.shared_up.wait_input_empty(k_iter_id);
            KERNEL_TIMER_END(smem->timer.get(), empty_span);
            smem->barriers.shared_up.input_ready.arrive(k_iter_id,
                                                        kInputTileTxBytes);
            load_input_tile_chunks<SharedUpGemm::kTileShapeN>(
                tma_input_desc, k_tile_base + k_iter_id,
                /*token_start=*/0u, smem->shared_expert.input.data[slot],
                smem->barriers.shared_up.input_ready.raw(slot));
        }
    }

    // Load one weight tile; the caller includes any scale bytes in the barrier.
    __device__ static void load_up_weight_tile(
        CUtensorMap const& tma_up_desc, SharedStorage* __restrict__ smem,
        std::uint32_t k_iter_id, std::uint32_t transaction_bytes) {
        const std::uint32_t slot = k_iter_id % kNumUpSlots;
        const auto empty_span = KERNEL_TIMER_BEGIN(
            smem->timer.get(), "shared_wait_up_weight_empty", k_iter_id);
        smem->barriers.shared_up.wait_weight_empty(k_iter_id);
        KERNEL_TIMER_END(smem->timer.get(), empty_span);
        const std::uint32_t k_tile = split_idx() * kUpTilesK + k_iter_id;
        smem->barriers.shared_up.weight_ready.arrive(k_iter_id,
                                                     transaction_bytes);
        Base::load_weight_tile(tma_up_desc,
                               &smem->shared_expert.w_up[slot][0][0][0],
                               smem->barriers.shared_up.weight_ready.raw(slot),
                               slab_idx() * SharedUpGemm::kTileShapeM, k_tile);
    }
};

template <typename KernelTraits>
struct SharedDownProjectionBase
    : DownProjectionBase<KernelTraits,
                         SharedDownGemmTraits<typename KernelTraits::MoEDims>> {
    using Base = DownProjectionBase<
        KernelTraits, SharedDownGemmTraits<typename KernelTraits::MoEDims>>;
    using typename Base::Dims;
    using typename Base::SharedStorage;
    using typename Base::Traits;
    using typename Base::Workspace;
    using SharedDownGemm = typename Base::DownGemm;
    using Shared = SharedUpProjectionBase<KernelTraits>;
    using SharedUpGemm = typename Shared::SharedUpGemm;

    static constexpr std::uint32_t kNumUpPublishers = Shared::kNumUpPublishers;
    static constexpr std::uint32_t kDownTilesK = SharedDownGemm::kNumTilesK;
    static constexpr std::uint32_t kDownActChunks =
        SharedDownGemm::kTileShapeK / Workspace::SharedUpOut::kLboChunk;
    static constexpr std::uint32_t kNumDownSlots =
        SharedDownGemm::kNumWeightSlots;

    using SBLayout = decltype(cute::make_layout(
        cute::make_shape(
            cute::make_shape(cute::Int<SharedDownGemm::kTileShapeN>{},
                             cute::make_shape(cute::_16{}, cute::_2{})),
            cute::_1{}, cute::Int<SharedDownGemm::kTileShapeK / 32u>{}),
        cute::make_stride(
            cute::make_stride(
                cute::_16{},
                cute::make_stride(
                    cute::_1{},
                    cute::Int<(SharedDownGemm::kTileShapeN + 1u) * 16u>{})),
            cute::_0{},
            cute::Int<2u * (SharedDownGemm::kTileShapeN + 1u) * 16u>{})));

    __device__ static std::uint32_t cta_idx() { return Shared::cta_idx(); }

    __device__ static void run_load_input_k_loop(
        Workspace* __restrict__ workspace, SharedStorage* __restrict__ smem,
        const void* __restrict__ scales) {
        const auto up_out_span =
            KERNEL_TIMER_BEGIN(smem->timer.get(), "wait_shared_up_out");
        workspace->cta_comm.wait_shared_up_out(smem->launch_parity,
                                               kNumUpPublishers);
        // Order cross-CTA stores before TMA reads.
        fence_proxy_async_global();
        KERNEL_TIMER_END(smem->timer.get(), up_out_span);

        // Blockwise uses a full activation-scale table; blockscaled stages one
        // SFB per input slot.
        constexpr std::uint32_t kScaleBytes =
            SharedDownGemm::MmaCore::kUseBlockscaled
                ? Workspace::SharedUpOut::kSfAtomBytes
                : static_cast<std::uint32_t>(
                      sizeof(smem->shared_expert.act_down_input.scale));
        constexpr std::uint32_t kInputBytes = static_cast<std::uint32_t>(
            sizeof(decltype(smem->shared_expert.act_down_input.data[0])));
#pragma unroll 1
        for (std::uint32_t k_iter_id = 0; k_iter_id < kDownTilesK;
             ++k_iter_id) {
            const std::uint32_t ring = k_iter_id % kNumDownSlots;
            smem->barriers.shared_down.wait_weight_empty(k_iter_id);
            const bool load_scale =
                SharedDownGemm::MmaCore::kUseBlockscaled || k_iter_id == 0u;
            const std::uint32_t kTotalBytes =
                kInputBytes + (load_scale ? kScaleBytes : 0u);
            smem->barriers.shared_down.input_ready.arrive(k_iter_id,
                                                          kTotalBytes);
            tma_load_1d(&workspace->shared_up_out
                             .data[k_iter_id * kDownActChunks][0][0],
                        kInputBytes,
                        &smem->shared_expert.act_down_input.data[ring][0][0][0],
                        smem->barriers.shared_down.input_ready.raw(ring));
            if (load_scale) {
                const auto* scale_src =
                    reinterpret_cast<const std::uint8_t*>(scales) +
                    (SharedDownGemm::MmaCore::kUseBlockscaled
                         ? k_iter_id * kScaleBytes
                         : 0u);
                auto* scale_dst = reinterpret_cast<std::uint8_t*>(
                    &smem->shared_expert.act_down_input
                         .scale[SharedDownGemm::MmaCore::kUseBlockscaled ? ring
                                                                         : 0u]
                               [0]);
                tma_load_1d(scale_src, kScaleBytes, scale_dst,
                            smem->barriers.shared_down.input_ready.raw(ring));
            }
        }
    }

    __device__ static void load_down_weight_tile(
        CUtensorMap const& tma_down_desc, SharedStorage* __restrict__ smem,
        std::uint32_t k_iter_id, std::uint32_t transaction_bytes) {
        const std::uint32_t slot = k_iter_id % kNumDownSlots;
        const auto empty_span = KERNEL_TIMER_BEGIN(
            smem->timer.get(), "shared_wait_down_weight_empty", k_iter_id);
        smem->barriers.shared_down.wait_weight_empty(k_iter_id);
        KERNEL_TIMER_END(smem->timer.get(), empty_span);
        smem->barriers.shared_down.weight_ready.arrive(k_iter_id,
                                                       transaction_bytes);
        Base::template load_weight_tile<SharedDownGemm::kTileShapeM>(
            tma_down_desc, &smem->shared_expert.w_down[slot][0][0],
            smem->barriers.shared_down.weight_ready.raw(slot), row_base(),
            k_iter_id);
    }

    __device__ static std::uint32_t row_base() {
        return cta_idx() * SharedDownGemm::kTileShapeM;
    }

    // Store one warp fragment of the completed down projection.
    __device__ static void store_down_output(
        DataType::ElementOut* __restrict__ out, std::uint32_t batch_size,
        std::uint32_t atom, std::uint32_t group, const float (&values)[4]) {
        const std::uint32_t warp_idx = WarpRole::get_warp_idx();
        const std::uint32_t lane_idx = WarpRole::get_lane_idx();
        const std::uint32_t lane_base =
            32u * (warp_idx & 3u) + 16u * (warp_idx >> 2);
        const std::uint32_t t0 = lane_idx & 3u;
        const std::uint32_t t1 = lane_idx >> 2;
        for (std::uint32_t r = 0; r < 2u; ++r) {
            const std::uint32_t col =
                row_base() + atom * 128u + lane_base + t1 + 8u * r;
            for (std::uint32_t i = 0; i < 2u; ++i) {
                const std::uint32_t tok = group * 8u + 2u * t0 + i;
                if (tok >= batch_size) continue;
                out[tok * SharedDownGemm::kGemmM + col] =
                    __float2bfloat16_rn(values[2u * r + i]);
            }
        }
    }
};

template <typename KernelTraits,
          bool UseBlockscaled = SharedUpGemmTraits<
              typename KernelTraits::MoEDims>::MmaCore::kUseBlockscaled>
struct SharedUpProjection;

template <typename KernelTraits,
          bool UseBlockscaled = SharedDownGemmTraits<
              typename KernelTraits::MoEDims>::MmaCore::kUseBlockscaled>
struct SharedDownProjection;

template <typename KernelTraits>
struct SharedUpProjection<KernelTraits, true>
    : SharedUpProjectionBase<KernelTraits> {
    using Base = SharedUpProjectionBase<KernelTraits>;
    using typename Base::Accumulator;
    using typename Base::Dims;
    using typename Base::SharedStorage;
    using typename Base::SharedUpGemm;
    using typename Base::Traits;
    using typename Base::Workspace;

    static_assert(SharedUpGemm::MmaCore::kUseBlockscaled,
                  "SharedUpProjection<true> requires a blockscaled MMA core");

    // Fill valid SFB rows and zero the padding for UTCCP.
    __device__ static void zero_sf_slots(SharedStorage* __restrict__ smem) {
        // Whole 32-bit words: the staging is a word-aligned byte blob.
        constexpr std::uint32_t kSfWords = static_cast<std::uint32_t>(
            sizeof(smem->shared_expert.quantized_input.scale) / 4u);
        auto* const sf = reinterpret_cast<std::uint32_t*>(
            &smem->shared_expert.quantized_input.scale[0][0]);
        for (std::uint32_t i = threadIdx.x; i < kSfWords;
             i += static_cast<std::uint32_t>(blockDim.x)) {
            sf[i] = 0u;
        }
        __syncthreads();
    }
    __device__ static void run(
        CUtensorMap const& tma_up_desc, CUtensorMap const& tma_input_desc,
        const typename SharedUpGemm::ElementScale* __restrict__ up_scale,
        Workspace* __restrict__ workspace, SharedStorage* __restrict__ smem,
        std::uint32_t batch_size, Activation activation, float situ_beta,
        float situ_linear_beta) {
        KERNEL_TIMER_SCOPE(smem->timer.get(), "shared_expert_up");
        zero_sf_slots(smem);

        Accumulator gate;
        Accumulator up;

        const std::uint32_t warp_idx = WarpRole::get_warp_idx();
        if (WarpRole::is_other_loader_thread()) {
            Base::run_load_input_k_loop(tma_input_desc, smem);
        } else if (WarpRole::is_weight_loader_thread()) {
            run_load_weight_k_loop(tma_up_desc, up_scale, smem);
        } else if (WarpRole::is_mma_issuer_warp(warp_idx)) {
            run_mma_k_loop(smem, up_scale);
        } else if (WarpRole::is_calc_warp(warp_idx)) {
#pragma unroll 1
            for (std::uint32_t k_iter_id = 0; k_iter_id < Base::kUpTilesK;
                 ++k_iter_id) {
                quantize_input_tile(smem, batch_size, k_iter_id);
            }
            load_accumulators(smem, gate, up);
        }
        Base::combine_gated_activation(smem, activation, situ_beta,
                                       situ_linear_beta, gate, up);
        if (Base::split_idx() == 0u && WarpRole::is_calc_warp(warp_idx)) {
            quantize_output(workspace, smem, batch_size);
        }
        // All split readers and publishers finish before the next phase.
        __syncthreads();
        cluster_arrive_and_wait();
        if (Base::split_idx() == 0u && threadIdx.x == 0u) {
            Base::publish_up_output(workspace, smem);
        }
    }

    __device__ static void run_load_weight_k_loop(
        CUtensorMap const& tma_up_desc,
        const typename SharedUpGemm::ElementScale* __restrict__ up_scale,
        SharedStorage* __restrict__ smem) {
#pragma unroll 1
        for (std::uint32_t k_iter_id = 0; k_iter_id < Base::kUpTilesK;
             ++k_iter_id) {
            Base::load_up_weight_tile(tma_up_desc, smem, k_iter_id,
                                      Base::kUpWeightTxBytes);
            const std::uint32_t slot = k_iter_id % Base::kNumUpSlots;
            const std::uint32_t k_tile =
                Base::split_idx() * Base::kUpTilesK + k_iter_id;
            Base::load_weight_scale_tile(
                up_scale, Base::slab_idx(), k_tile,
                smem->shared_expert.w_up_scale[slot],
                smem->barriers.shared_up.weight_ready.raw(slot));
        }
    }

    __device__ static void run_mma_k_loop(
        SharedStorage* __restrict__ smem,
        const typename SharedUpGemm::ElementScale* __restrict__ up_scale) {
        typename SharedUpGemm::TiledMma tiled_mma{};
        auto cta_mma = tiled_mma.get_slice(cute::_0{});
        auto tCt = Base::make_accumulator_fragment(cta_mma);
#pragma unroll 1
        for (std::uint32_t k_iter_id = 0; k_iter_id < Base::kUpTilesK;
             ++k_iter_id) {
            const std::uint32_t ring = k_iter_id % Base::kNumUpSlots;
            const auto mma_ready_span = KERNEL_TIMER_BEGIN(
                smem->timer.get(), "shared_up_wait_mma_ready", k_iter_id);
            smem->barriers.shared_up.template wait_mma_ready<true, false>(
                k_iter_id);
            KERNEL_TIMER_END(smem->timer.get(), mma_ready_span);
            if constexpr (SharedUpGemm::kScaleLayout ==
                          ScaleLayout::kPlainRows) {
                Base::build_plain_scale</*Paired=*/true>(
                    smem->shared_expert.w_up_scale[ring],
                    reinterpret_cast<const std::uint8_t*>(up_scale),
                    static_cast<std::uint32_t>(Base::slab_idx()) *
                        (SharedUpGemm::kTileShapeM / 2u),
                    Base::split_idx() * Base::kUpTilesK + k_iter_id,
                    SharedUpGemm::kGemmK / 32u, SharedUpGemm::kGemmM / 2u);
            }
            Base::mma(tiled_mma, cta_mma, tCt,
                      &smem->shared_expert.w_up[ring][0][0][0],
                      &smem->shared_expert.quantized_input.data[ring][0][0][0],
                      smem->tmem_ptr,
                      /*acc_slot=*/0u, smem->shared_expert.w_up_scale[ring],
                      &smem->shared_expert.quantized_input.scale[ring][0],
                      /*accumulate=*/k_iter_id != 0u);
            cutlass::arch::umma_arrive(
                smem->barriers.shared_up.weight_empty.raw(ring));
        }
        // Hand off the completed split once; its weight slots are now drained.
        cutlass::arch::umma_arrive(
            smem->barriers.shared_up.accumulator_ready.raw(0u));
    }

    __device__ static void quantize_input_tile(SharedStorage* __restrict__ smem,
                                               std::uint32_t batch_size,
                                               std::uint32_t k_iter_id) {
        const std::uint32_t warp_idx = WarpRole::get_calc_warp_idx();
        const std::uint32_t lane_idx = WarpRole::get_lane_idx();
        constexpr std::uint32_t kBlocksPerTile =
            SharedUpGemm::kNumStagingBlocks;
        const std::uint32_t slot = k_iter_id % Base::kNumUpSlots;
        const auto weight_empty_span = KERNEL_TIMER_BEGIN(
            smem->timer.get(), "shared_wait_up_weight_empty", k_iter_id);
        smem->barriers.shared_up.wait_weight_empty(k_iter_id);
        KERNEL_TIMER_END(smem->timer.get(), weight_empty_span);
        const auto input_ready_span = KERNEL_TIMER_BEGIN(
            smem->timer.get(), "shared_wait_up_input_ready", k_iter_id);
        smem->barriers.shared_up.input_ready.wait(k_iter_id);
        KERNEL_TIMER_END(smem->timer.get(), input_ready_span);
        const std::uint32_t k_base =
            (Base::split_idx() * Base::kUpTilesK + k_iter_id) *
            SharedUpGemm::kTileShapeK;
        const std::uint32_t blocks_base =
            Base::split_idx() * Base::kQuantBlocksPerCta +
            k_iter_id * kBlocksPerTile;
        const QuantizeTile tile(slot, blocks_base, /*row_start=*/0u,
                                batch_size);
        const auto* const act = reinterpret_cast<const DataType::ElementA*>(
            &smem->shared_expert.input.data[slot][0][0][0]);
        quantize_mxfp8_groupwise_1x32<
            /*Staged=*/true, SharedUpGemm::kTileShapeN,
            SharedUpGemm::kTileShapeK, 128u, Base::kInputSwizzleBoxK,
            kBlocksPerTile>(act, k_base,
                            &smem->shared_expert.quantized_input.data[0],
                            &smem->shared_expert.quantized_input.scale[0][0],
                            tile, warp_idx, WarpRole::kCalcWarpCount);
        cutlass::arch::fence_view_async_shared();
        WarpRole::sync_calc_threads();
        if (warp_idx == 0u && lane_idx == 0u) {
            smem->barriers.shared_up.input_empty.arrive(k_iter_id);
        }
    }

    // Read the completed split accumulator.
    __device__ static void load_accumulators(SharedStorage* __restrict__ smem,
                                             Accumulator& gate,
                                             Accumulator& up) {
        KERNEL_TIMER_SCOPE(smem->timer.get(), "shared_up_wait_accumulator");
        smem->barriers.shared_up.accumulator_ready.wait(0u);
        tcgen05_fence_after_thread_sync();
        Base::load_accumulators(smem->tmem_ptr, gate, up);
    }

    __device__ static void quantize_output(Workspace* __restrict__ workspace,
                                           SharedStorage* __restrict__ smem,
                                           std::uint32_t batch_size) {
        const std::uint32_t warp_idx = WarpRole::get_warp_idx();
        const std::uint32_t lane_idx = WarpRole::get_lane_idx();

        constexpr std::uint32_t kQuantGroups = Base::kOutputFeatureBlock / 32u;
        // Store one UE8M0 byte per 32 features in the down-input SfAtom
        // layout.
        const std::uint32_t tile =
            Base::slab_idx() / (128u / Base::kOutputFeatureBlock);
        const std::uint32_t group_base =
            (Base::slab_idx() * Base::kOutputFeatureBlock % 128u) / 32u;
        std::uint8_t* const sf = workspace->shared_up_out.sf_slab(tile);
#pragma unroll 1
        for (std::uint32_t tok = warp_idx; tok < batch_size;
             tok += WarpRole::kCalcWarpCount) {
            DataType::ElementQA quantized[kQuantGroups];
            typename Base::OutputQuant::ScaleType scales[kQuantGroups];
            Base::quantize_output_row(smem->shared_expert.act_up_out, tok,
                                      quantized, scales);
            if (lane_idx == 0u) {
#pragma unroll
                for (std::uint32_t i = 0; i < kQuantGroups; ++i) {
                    sf[sf_atom_offset(tok, group_base + i)] = scales[i];
                }
            }
#pragma unroll
            for (std::uint32_t i = 0; i < kQuantGroups; ++i) {
                const std::uint32_t feat = lane_idx + i * 32u;
                const std::uint32_t feature =
                    Base::slab_idx() * Base::kOutputFeatureBlock + feat;
                workspace->shared_up_out.cell(feature / Base::kLboChunk,
                                              tok)[feat % Base::kLboChunk] =
                    quantized[i];
            }
        }
        // The padded rows of the tile's slab stay zero for UTCCP.
        if (lane_idx == 0u) {
            for (std::uint32_t tok = batch_size + warp_idx; tok < 128u;
                 tok += WarpRole::kCalcWarpCount) {
#pragma unroll
                for (std::uint32_t i = 0; i < kQuantGroups; ++i) {
                    sf[sf_atom_offset(tok, group_base + i)] = 0u;
                }
            }
        }
    }
};

template <typename KernelTraits>
struct SharedUpProjection<KernelTraits, false>
    : SharedUpProjectionBase<KernelTraits> {
    using Base = SharedUpProjectionBase<KernelTraits>;
    using typename Base::Accumulator;
    using typename Base::Dims;
    using typename Base::SharedStorage;
    using typename Base::SharedUpGemm;
    using typename Base::Traits;
    using typename Base::Workspace;

    static_assert(
        !SharedUpGemm::MmaCore::kUseBlockscaled,
        "SharedUpProjection<false> requires a non-blockscaled MMA core");

    using Base::kNumAccSlots;

    __device__ static void run(
        CUtensorMap const& tma_up_desc, CUtensorMap const& tma_input_desc,
        const typename SharedUpGemm::ElementScale* __restrict__ up_scale,
        Workspace* __restrict__ workspace, SharedStorage* __restrict__ smem,
        std::uint32_t batch_size, Activation activation, float situ_beta,
        float situ_linear_beta) {
        KERNEL_TIMER_SCOPE(smem->timer.get(), "shared_expert_up");
        const std::uint32_t warp_idx = WarpRole::get_warp_idx();
        Accumulator gate{};
        Accumulator up{};
        if (WarpRole::is_other_loader_thread()) {
            Base::run_load_input_k_loop(tma_input_desc, smem);
        } else if (WarpRole::is_weight_loader_thread()) {
            run_load_weight_k_loop(tma_up_desc, smem);
        } else if (WarpRole::is_mma_issuer_warp(warp_idx)) {
            run_mma_k_loop(smem);
        } else if (WarpRole::is_calc_warp(warp_idx)) {
            run_quantize_and_dequant_k_loop(up_scale, smem, batch_size, gate,
                                            up);
        }
        Base::combine_gated_activation(smem, activation, situ_beta,
                                       situ_linear_beta, gate, up);
        if (Base::split_idx() == 0u && WarpRole::is_calc_warp(warp_idx)) {
            quantize_output(workspace, smem, batch_size);
        }
        // All split readers and publishers finish before the next phase.
        __syncthreads();
        cluster_arrive_and_wait();
        if (Base::split_idx() == 0u && threadIdx.x == 0u) {
            Base::publish_up_output(workspace, smem);
        }
    }

    __device__ static void run_load_weight_k_loop(
        CUtensorMap const& tma_up_desc, SharedStorage* __restrict__ smem) {
#pragma unroll 1
        for (std::uint32_t k_iter_id = 0; k_iter_id < Base::kUpTilesK;
             ++k_iter_id) {
            Base::load_up_weight_tile(
                tma_up_desc, smem, k_iter_id,
                SharedUpGemm::MmaCore::kWeightTileTxBytes);
        }
    }

    __device__ static void run_mma_k_loop(SharedStorage* __restrict__ smem) {
        auto& sh = smem->shared_expert;
        typename SharedUpGemm::TiledMma tiled_mma{};
        auto cta_mma = tiled_mma.get_slice(cute::_0{});
        auto tCt = Base::make_accumulator_fragment(cta_mma);
#pragma unroll 1
        for (std::uint32_t k_iter_id = 0; k_iter_id < Base::kUpTilesK;
             ++k_iter_id) {
            const std::uint32_t ring = k_iter_id % Base::kNumUpSlots;
            const std::uint32_t acc_slot = k_iter_id % kNumAccSlots;
            const auto mma_ready_span = KERNEL_TIMER_BEGIN(
                smem->timer.get(), "shared_up_wait_mma_ready", k_iter_id);
            smem->barriers.shared_up.template wait_mma_ready<true, true>(
                k_iter_id);
            KERNEL_TIMER_END(smem->timer.get(), mma_ready_span);
            Base::mma(tiled_mma, cta_mma, tCt, &sh.w_up[ring][0][0][0],
                      &sh.quantized_input.data[ring][0][0][0], smem->tmem_ptr,
                      acc_slot);
            cutlass::arch::umma_arrive(
                smem->barriers.shared_up.weight_empty.raw(ring));
            cutlass::arch::umma_arrive(
                smem->barriers.shared_up.accumulator_ready.raw(acc_slot));
        }
    }

    __device__ static void run_quantize_and_dequant_k_loop(
        const typename SharedUpGemm::ElementScale* __restrict__ up_scale,
        SharedStorage* __restrict__ smem, std::uint32_t batch_size,
        Accumulator& gate, Accumulator& up) {
        quantize_input_tile(smem, batch_size, 0u);
#pragma unroll 1
        for (std::uint32_t k_iter_id = 1; k_iter_id < Base::kUpTilesK;
             ++k_iter_id) {
            quantize_input_tile(smem, batch_size, k_iter_id);
            dequant(up_scale, smem, k_iter_id - 1u, gate, up);
        }
        dequant(up_scale, smem, Base::kUpTilesK - 1u, gate, up);
    }

    __device__ static void quantize_input_tile(SharedStorage* __restrict__ smem,
                                               std::uint32_t batch_size,
                                               std::uint32_t k_iter_id) {
        const auto quantize_span = KERNEL_TIMER_BEGIN(
            smem->timer.get(), "shared_quantize_input_tile", k_iter_id);
        const std::uint32_t warp_idx = WarpRole::get_calc_warp_idx();
        const std::uint32_t lane_idx = WarpRole::get_lane_idx();
        constexpr std::uint32_t kBlocksPerTile =
            SharedUpGemm::kNumStagingBlocks;
        const std::uint32_t slot = k_iter_id % Base::kNumUpSlots;
        const auto weight_empty_span = KERNEL_TIMER_BEGIN(
            smem->timer.get(), "shared_wait_up_weight_empty", k_iter_id);
        smem->barriers.shared_up.wait_weight_empty(k_iter_id);
        KERNEL_TIMER_END(smem->timer.get(), weight_empty_span);
        const auto input_ready_span = KERNEL_TIMER_BEGIN(
            smem->timer.get(), "shared_wait_up_input_ready", k_iter_id);
        smem->barriers.shared_up.input_ready.wait(k_iter_id);
        KERNEL_TIMER_END(smem->timer.get(), input_ready_span);
        const std::uint32_t k_base =
            (Base::split_idx() * Base::kUpTilesK + k_iter_id) *
            SharedUpGemm::kTileShapeK;
        const std::uint32_t blocks_base =
            Base::split_idx() * Base::kQuantBlocksPerCta +
            k_iter_id * kBlocksPerTile;
        const QuantizeTile tile(/*staging_slot_base=*/0u, blocks_base,
                                /*row_start=*/0u, batch_size);
        quantize_fp8_groupwise_128<
            /*Staged=*/true, SharedUpGemm::kTileShapeN, SharedUpGemm::kGemmK,
            SharedUpGemm::kTileShapeN, SharedUpGemm::InputQuant::kBlockSize,
            Base::kInputSwizzleBoxK, kBlocksPerTile>(
            reinterpret_cast<const DataType::ElementA*>(
                &smem->shared_expert.input.data[slot][0][0][0]),
            k_base, &smem->shared_expert.quantized_input.data[slot],
            &smem->shared_expert.quantized_input.scale[0][0], tile, warp_idx,
            WarpRole::kCalcWarpCount);
        cutlass::arch::fence_view_async_shared();
        WarpRole::sync_calc_threads();
        if (warp_idx == 0u && lane_idx == 0u) {
            smem->barriers.shared_up.input_empty.arrive(k_iter_id);
        }
        KERNEL_TIMER_END(smem->timer.get(), quantize_span);
    }

    __device__ static void dequant(
        const typename SharedUpGemm::ElementScale* __restrict__ up_scale,
        SharedStorage* __restrict__ smem, std::uint32_t k_iter_id,
        Accumulator& gate, Accumulator& up) {
        const std::uint32_t warp_idx = WarpRole::get_warp_idx();
        const std::uint32_t lane_idx = WarpRole::get_lane_idx();
        const std::uint32_t acc_slot = k_iter_id % kNumAccSlots;

        const auto acc_ready_span = KERNEL_TIMER_BEGIN(
            smem->timer.get(), "shared_up_wait_accumulator", k_iter_id);
        smem->barriers.shared_up.accumulator_ready.wait(k_iter_id);
        tcgen05_fence_after_thread_sync();
        KERNEL_TIMER_END(smem->timer.get(), acc_ready_span);

        const bool reads_tile = Base::owns_up_rows(warp_idx);
        const std::uint32_t gate_row = Base::slab_idx() *
                                       Base::kOutputFeatureBlock /
                                       SharedUpGemm::WeightQuant::kBlockSize;
        const std::uint32_t up_row =
            (SharedUpGemm::kGemmM / SharedUpGemm::WeightQuant::kBlockSize) /
                2u +
            gate_row;
        constexpr std::uint32_t kScaleBlocksK =
            SharedUpGemm::kGemmK / SharedUpGemm::WeightQuant::kBlockSize;
        const std::uint32_t kblk_base =
            Base::split_idx() * Base::kQuantBlocksPerCta + k_iter_id * 2u;
#pragma unroll 1
        for (std::uint32_t s = 0; reads_tile && s < 2u; ++s) {
            const std::uint32_t kblk = kblk_base + s;
            const float ws_gate = up_scale[gate_row * kScaleBlocksK + kblk];
            const float ws_up = up_scale[up_row * kScaleBlocksK + kblk];
            Base::dequant(smem->tmem_ptr,
                          acc_slot * get_gemm_acc_columns<SharedUpGemm>() +
                              s * SharedUpGemm::kTileShapeN,
                          smem->shared_expert.quantized_input.scale[kblk],
                          ws_gate, ws_up, gate, up);
        }
        cutlass::arch::fence_view_async_tmem_load();
        WarpRole::sync_calc_threads();
        if (warp_idx == 0u && lane_idx == 0u) {
            smem->barriers.shared_up.accumulator_empty.arrive(k_iter_id);
        }
    }

    __device__ static void quantize_output(Workspace* __restrict__ workspace,
                                           SharedStorage* __restrict__ smem,
                                           std::uint32_t batch_size) {
        const std::uint32_t warp_idx = WarpRole::get_warp_idx();
        const std::uint32_t lane_idx = WarpRole::get_lane_idx();

        constexpr std::uint32_t kQuantGroups = Base::kOutputFeatureBlock / 32u;
#pragma unroll 1
        for (std::uint32_t tok = warp_idx; tok < batch_size;
             tok += WarpRole::kCalcWarpCount) {
            DataType::ElementQA quantized[kQuantGroups];
            typename Base::OutputQuantizer::ScaleArray scales;
            Base::quantize_output_row(smem->shared_expert.act_up_out, tok,
                                      quantized, scales);
            if (lane_idx == 0u) {
                workspace->shared_up_out.scale_row(Base::slab_idx())[tok] =
                    scales[0];
            }
#pragma unroll
            for (std::uint32_t i = 0; i < kQuantGroups; ++i) {
                const std::uint32_t feat = lane_idx + i * 32u;
                const std::uint32_t feature =
                    Base::slab_idx() * Base::kOutputFeatureBlock + feat;
                workspace->shared_up_out.cell(feature / Base::kLboChunk,
                                              tok)[feat % Base::kLboChunk] =
                    quantized[i];
            }
        }
    }
};

template <typename KernelTraits>
struct SharedDownProjection<KernelTraits, true>
    : SharedDownProjectionBase<KernelTraits> {
    using Base = SharedDownProjectionBase<KernelTraits>;
    using typename Base::Dims;
    using typename Base::SBLayout;
    using typename Base::SharedDownGemm;
    using typename Base::SharedStorage;
    using typename Base::Traits;
    using typename Base::Workspace;

    static_assert(SharedDownGemm::MmaCore::kUseBlockscaled,
                  "SharedDownProjection<true> requires a blockscaled MMA core");

    using typename Base::DownScaleTraits;
    static constexpr std::uint32_t kDownSfaTmemOffset =
        get_gemm_acc_columns<SharedDownGemm>();
    static constexpr std::uint32_t kDownSfbTmemOffset =
        kDownSfaTmemOffset + DownScaleTraits::kSfaTmemColumns;

    __device__ static void run(
        CUtensorMap const& tma_down_desc,
        const typename SharedDownGemm::ElementScale* __restrict__ down_scale,
        DataType::ElementOut* __restrict__ out,
        Workspace* __restrict__ workspace, SharedStorage* __restrict__ smem,
        std::uint32_t batch_size) {
        KERNEL_TIMER_SCOPE(smem->timer.get(), "shared_expert_down");
        const std::uint32_t warp_idx = WarpRole::get_warp_idx();
        if (WarpRole::is_other_loader_thread()) {
            Base::run_load_input_k_loop(workspace, smem,
                                        workspace->shared_up_out.scale_bytes());
        } else if (WarpRole::is_weight_loader_thread()) {
            run_load_weight_k_loop(tma_down_desc, down_scale, smem);
        } else if (WarpRole::is_mma_issuer_warp(warp_idx)) {
            run_mma_k_loop(smem, down_scale);
        } else if (WarpRole::is_calc_warp(warp_idx)) {
            epilogue(out, smem, batch_size);
        }
    }

    __device__ static void run_load_weight_k_loop(
        CUtensorMap const& tma_down_desc,
        const typename SharedDownGemm::ElementScale* __restrict__ down_scale,
        SharedStorage* __restrict__ smem) {
        const std::uint32_t row_atom = Base::row_base() / 128u;
        for (std::uint32_t k_iter_id = 0; k_iter_id < Base::kDownTilesK;
             ++k_iter_id) {
            Base::load_down_weight_tile(tma_down_desc, smem, k_iter_id,
                                        Base::kDownWeightTxBytes);
            const std::uint32_t slot = k_iter_id % Base::kNumDownSlots;
            Base::load_weight_scale_tile(
                down_scale, row_atom, k_iter_id,
                smem->shared_expert.w_down_scale[slot],
                smem->barriers.shared_down.weight_ready.raw(slot));
        }
    }

    __device__ static void run_mma_k_loop(
        SharedStorage* __restrict__ smem,
        const typename SharedDownGemm::ElementScale* __restrict__ down_scale) {
        typename SharedDownGemm::TiledMma tiled_mma{};
        auto cta_mma = tiled_mma.get_slice(cute::_0{});
        auto tCt = Base::make_accumulator_fragment(cta_mma);
#pragma unroll 1
        for (std::uint32_t k_iter_id = 0; k_iter_id < Base::kDownTilesK;
             ++k_iter_id) {
            const std::uint32_t slot = k_iter_id % Base::kNumDownSlots;
            smem->barriers.shared_down.wait_mma_ready(k_iter_id);
            if constexpr (SharedDownGemm::kScaleLayout ==
                          ScaleLayout::kPlainRows) {
                Base::build_plain_scale(
                    smem->shared_expert.w_down_scale[slot],
                    reinterpret_cast<const std::uint8_t*>(down_scale),
                    Base::row_base(), k_iter_id, SharedDownGemm::kGemmK / 32u);
            }
            const auto sfa_tmem =
                tmem_addr(smem->tmem_ptr, 0u, kDownSfaTmemOffset);
            const auto sfb_tmem =
                tmem_addr(smem->tmem_ptr, 0u, kDownSfbTmemOffset);
            // Reuse the fixed sum and scale staging in issuer order.
            Base::mma(tiled_mma, cta_mma, SBLayout{}, tCt,
                      &smem->shared_expert.w_down[slot][0][0],
                      &smem->shared_expert.act_down_input.data[slot][0][0][0],
                      tmem_addr(smem->tmem_ptr, 0u, 0u),
                      smem->shared_expert.w_down_scale[slot],
                      &smem->shared_expert.act_down_input.scale[slot][0],
                      sfa_tmem, sfb_tmem, /*accumulate=*/k_iter_id != 0u);
            cutlass::arch::umma_arrive(
                smem->barriers.shared_down.weight_empty.raw(slot));
        }
        cutlass::arch::umma_arrive(
            smem->barriers.shared_down.accumulator_ready.raw(0u));
    }

    // One readout of the final TMEM sum, followed by the output conversion.
    __device__ static void epilogue(DataType::ElementOut* __restrict__ out,
                                    SharedStorage* __restrict__ smem,
                                    std::uint32_t batch_size) {
        const std::uint32_t warp_idx = WarpRole::get_warp_idx();
        const std::uint32_t lane_base =
            32u * (warp_idx & 3u) + 16u * (warp_idx >> 2);
        const auto done_span =
            KERNEL_TIMER_BEGIN(smem->timer.get(), "shared_down_wait_done");
        smem->barriers.shared_down.accumulator_ready.wait(0u);
        KERNEL_TIMER_END(smem->timer.get(), done_span);
        tcgen05_fence_after_thread_sync();

#pragma unroll
        for (std::uint32_t atom = 0; atom < SharedDownGemm::MmaCore::kNumAtomMs;
             ++atom) {
#pragma unroll
            for (std::uint32_t group = 0;
                 group < SharedDownGemm::kTileShapeN / 8u; ++group) {
                float values[4];
                Base::load_accumulator_fragment(
                    smem->tmem_ptr,
                    atom * SharedDownGemm::kTileShapeN + group * 8u, lane_base,
                    values);
                Base::store_down_output(out, batch_size, atom, group, values);
            }
        }
    }
};

template <typename KernelTraits>
struct SharedDownProjection<KernelTraits, false>
    : SharedDownProjectionBase<KernelTraits> {
    using Base = SharedDownProjectionBase<KernelTraits>;
    using typename Base::Dims;
    using typename Base::SBLayout;
    using typename Base::SharedDownGemm;
    using typename Base::SharedStorage;
    using typename Base::Traits;
    using typename Base::Workspace;

    static_assert(
        !SharedDownGemm::MmaCore::kUseBlockscaled,
        "SharedDownProjection<false> requires a non-blockscaled MMA core");
    static_assert(SharedDownGemm::MmaCore::kAtomShapeM ==
                      SharedDownGemm::WeightQuant::kBlockSize,
                  "each M atom reads one row of weight-scale blocks");

    __device__ static void run(
        CUtensorMap const& tma_down_desc,
        const typename SharedDownGemm::ElementScale* __restrict__ down_scale,
        DataType::ElementOut* __restrict__ out,
        Workspace* __restrict__ workspace, SharedStorage* __restrict__ smem,
        std::uint32_t batch_size) {
        KERNEL_TIMER_SCOPE(smem->timer.get(), "shared_expert_down");
        const std::uint32_t warp_idx = WarpRole::get_warp_idx();
        if (WarpRole::is_other_loader_thread()) {
            Base::run_load_input_k_loop(workspace, smem,
                                        workspace->shared_up_out.scale_ptr());
        } else if (WarpRole::is_weight_loader_thread()) {
            run_load_weight_k_loop(tma_down_desc, smem);
        } else if (WarpRole::is_mma_issuer_warp(warp_idx)) {
            run_mma_k_loop(smem);
        } else if (WarpRole::is_calc_warp(warp_idx)) {
            epilogue(down_scale, out, smem, batch_size);
        }
    }

    __device__ static void run_load_weight_k_loop(
        CUtensorMap const& tma_down_desc, SharedStorage* __restrict__ smem) {
        for (std::uint32_t k_iter_id = 0; k_iter_id < Base::kDownTilesK;
             ++k_iter_id) {
            Base::load_down_weight_tile(
                tma_down_desc, smem, k_iter_id,
                SharedDownGemm::MmaCore::kWeightTileTxBytes);
        }
    }

    __device__ static void run_mma_k_loop(SharedStorage* __restrict__ smem) {
        typename SharedDownGemm::TiledMma tiled_mma{};
        auto cta_mma = tiled_mma.get_slice(cute::_0{});
        auto tCt = Base::make_accumulator_fragment(cta_mma);
#pragma unroll 1
        for (std::uint32_t k_iter_id = 0; k_iter_id < Base::kDownTilesK;
             ++k_iter_id) {
            const std::uint32_t slot = k_iter_id % Base::kNumDownSlots;
            smem->barriers.shared_down.wait_mma_ready(k_iter_id);
            Base::mma(
                tiled_mma, cta_mma, SBLayout{}, tCt,
                &smem->shared_expert.w_down[slot][0][0],
                &smem->shared_expert.act_down_input.data[slot][0][0][0],
                tmem_addr(smem->tmem_ptr, 0u,
                          k_iter_id * get_gemm_acc_columns<SharedDownGemm>()));
            cutlass::arch::umma_arrive(
                smem->barriers.shared_down.weight_empty.raw(slot));
        }
        cutlass::arch::umma_arrive(
            smem->barriers.shared_down.accumulator_ready.raw(0u));
    }

    __device__ static void epilogue(
        const typename SharedDownGemm::ElementScale* __restrict__ down_scale,
        DataType::ElementOut* __restrict__ out,
        SharedStorage* __restrict__ smem, std::uint32_t batch_size) {
        const std::uint32_t warp_idx = WarpRole::get_warp_idx();
        const std::uint32_t lane_idx = WarpRole::get_lane_idx();
        // The routed down's readout: warp w covers lanes 32*(w&3)+16*(w>>2).
        const std::uint32_t lane_base =
            32u * (warp_idx & 3u) + 16u * (warp_idx >> 2);
        const std::uint32_t t0 = lane_idx & 3u;

        const auto done_span =
            KERNEL_TIMER_BEGIN(smem->timer.get(), "shared_down_wait_done");
        smem->barriers.shared_down.accumulator_ready.wait(0u);
        KERNEL_TIMER_END(smem->timer.get(), done_span);
#pragma unroll
        for (std::uint32_t atom = 0; atom < SharedDownGemm::MmaCore::kNumAtomMs;
             ++atom) {
#pragma unroll
            for (std::uint32_t group = 0;
                 group < SharedDownGemm::kTileShapeN / 8u; ++group) {
                float acc[4] = {};
#pragma unroll 1
                for (std::uint32_t t = 0; t < Base::kDownTilesK; ++t) {
#pragma unroll
                    for (std::uint32_t s = 0;
                         s < SharedDownGemm::MmaCore::kNumQuantBlocksB; ++s) {
                        float v[4];
                        Base::load_accumulator_fragment(
                            smem->tmem_ptr,
                            t * get_gemm_acc_columns<SharedDownGemm>() +
                                atom *
                                    SharedDownGemm::MmaCore::kNumQuantBlocksB *
                                    SharedDownGemm::kTileShapeN +
                                s * SharedDownGemm::kTileShapeN + group * 8u,
                            lane_base, v);
                        const float ws = down_scale
                            [(Base::cta_idx() *
                                  SharedDownGemm::MmaCore::kNumAtomMs +
                              atom) *
                                 Base::kDownTilesK +
                             t];
                        Base::dequant(v,
                                      &smem->shared_expert.act_down_input
                                           .scale[t * SharedDownGemm::MmaCore::
                                                          kNumQuantBlocksB +
                                                  s][group * 8u + 2u * t0],
                                      ws, acc);
                    }
                }
                Base::store_down_output(out, batch_size, atom, group, acc);
            }
        }
    }
};

}  // namespace halops::moe
