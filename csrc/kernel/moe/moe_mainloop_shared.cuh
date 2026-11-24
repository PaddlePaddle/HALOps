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

    static_assert(SharedDownGemm::kTileShapeM == SharedUpGemm::kTileShapeM,
                  "the shared up and down passes share the phase's CTA range");

    static constexpr std::uint32_t kNumUpSlots = Base::kNumSlots;
    static constexpr std::uint32_t kInputSwizzleBoxK =
        decltype(SharedExpertSmem<Dims, SharedUpGemm,
                                  SharedDownGemm>::input)::kSwizzleBoxK;
    // TMA byte count for the staged BF16 input tile.
    static constexpr std::uint32_t kInputTileTxBytes =
        SharedUpGemm::kTileShapeN * SharedUpGemm::kTileShapeK *
        static_cast<std::uint32_t>(sizeof(DataType::ElementA));

    __device__ static std::uint32_t cta_idx(MoERuntime const& runtime) {
        return blockIdx.x - runtime.first_shared_cta;
    }

    // The M slab whose splits this CTA's block belongs to.
    __device__ static std::uint32_t slab_idx(MoERuntime const& runtime) {
        return static_cast<std::uint32_t>(runtime.shared_splits_divmod.div(
            static_cast<int>(cta_idx(runtime))));
    }

    __device__ static std::uint32_t split_idx(MoERuntime const& runtime) {
        const int rank =
            runtime.cluster_divmod.rem(static_cast<int>(blockIdx.x));
        return static_cast<std::uint32_t>(
            runtime.shared_splits_divmod.rem(rank));
    }

    // Up-pass slabs: one CTA publishes each of them.
    __device__ static std::uint32_t num_up_publishers(
        MoERuntime const& runtime) {
        return runtime.num_shared_ctas / runtime.num_shared_splits;
    }

    __device__ static std::uint32_t k_tiles_per_split(
        MoEProblemSize const& problem_size, MoERuntime const& runtime) {
        return problem_size.hidden_size / SharedUpGemm::kTileShapeK /
               runtime.num_shared_splits;
    }

    __device__ static void publish_up_output(
        Workspace const* __restrict__ workspace,
        SharedStorage* __restrict__ smem) {
        workspace->cta_comm->publish_shared_up_tasks(smem->launch_parity, 1u);
    }

    __device__ static void combine_gated_activation(
        SharedStorage* __restrict__ smem, Activation activation,
        float situ_beta, float situ_linear_beta, const Accumulator& gate,
        const Accumulator& up, MoERuntime const& runtime) {
        auto& partial = smem->shared_expert.splited_mma_out;
        Base::join_up_split_k(partial, gate, up, runtime.num_shared_splits);
        if (split_idx(runtime) == 0u &&
            WarpRole::is_calc_warp(WarpRole::get_warp_idx())) {
            Base::combine_gated_activation(
                partial, smem->shared_expert.act_up_out, gate, up, activation,
                situ_beta, situ_linear_beta, runtime.num_shared_splits,
                runtime.cluster_size);
            WarpRole::sync_calc_threads();
        }
    }

    __device__ static void run_load_input_k_loop(
        CUtensorMap const& tma_input_desc, SharedStorage* __restrict__ smem,
        std::uint32_t num_k_tiles, std::uint32_t k_tile_base) {
#pragma unroll 1
        for (std::uint32_t k_iter_id = 0; k_iter_id < num_k_tiles;
             ++k_iter_id) {
            const std::uint32_t slot = k_iter_id % kNumUpSlots;
            smem->barriers.shared_up.wait_input_empty(k_iter_id);
            smem->barriers.shared_up.input_ready.arrive(k_iter_id,
                                                        kInputTileTxBytes);
            load_input_tile_chunks<SharedUpGemm::kTileShapeN>(
                tma_input_desc, k_tile_base + k_iter_id,
                /*token_start=*/0u, smem->shared_expert.input.data[slot],
                smem->barriers.shared_up.input_ready.raw(slot));
        }
    }

    // Load one weight tile; the caller includes any scale bytes in the barrier.
    __device__ static void load_up_weight_tile(CUtensorMap const& tma_up_desc,
                                               SharedStorage* __restrict__ smem,
                                               std::uint32_t k_iter_id,
                                               std::uint32_t transaction_bytes,
                                               std::uint32_t k_tile_base,
                                               std::uint32_t slab) {
        const std::uint32_t slot = k_iter_id % kNumUpSlots;
        smem->barriers.shared_up.wait_weight_empty(k_iter_id);
        const std::uint32_t k_tile = k_tile_base + k_iter_id;
        smem->barriers.shared_up.weight_ready.arrive(k_iter_id,
                                                     transaction_bytes);
        Base::load_weight_tile(tma_up_desc,
                               &smem->shared_expert.w_up[slot][0][0][0],
                               smem->barriers.shared_up.weight_ready.raw(slot),
                               slab * SharedUpGemm::kTileShapeM, k_tile);
    }

    // Quantize one staged input K tile and hand its slot back to the loader.
    __device__ static void quantize_input_tile(SharedStorage* __restrict__ smem,
                                               std::uint32_t gemm_n,
                                               std::uint32_t k_iter_id,
                                               std::uint32_t k_tile_base) {
        constexpr std::uint32_t kBlocksPerTile =
            SharedUpGemm::MmaCore::kNumSwizzleBoxKs;
        const std::uint32_t warp_idx = WarpRole::get_calc_warp_idx();
        const std::uint32_t lane_idx = WarpRole::get_lane_idx();
        const std::uint32_t slot = k_iter_id % kNumUpSlots;
        smem->barriers.shared_up.wait_weight_empty(k_iter_id);
        smem->barriers.shared_up.input_ready.wait(k_iter_id);
        const std::uint32_t k_tile = k_tile_base + k_iter_id;
        const std::uint32_t blocks_base = k_tile * kBlocksPerTile;
        // A blockwise tile's scale rows live in its staging slot.
        const SmemTileQuantizer<typename SharedUpGemm::InputQuant,
                                SharedUpGemm::kTileShapeN, 128u,
                                kInputSwizzleBoxK, kBlocksPerTile>
            tile(slot, blocks_base, /*scale_row_base=*/slot * kBlocksPerTile,
                 /*row_start=*/0u, gemm_n);
        const auto* const act = reinterpret_cast<const DataType::ElementA*>(
            &smem->shared_expert.input.data[slot][0][0][0]);
#pragma unroll 1
        for (std::uint32_t i = warp_idx;
             i < SharedUpGemm::kTileShapeN * kBlocksPerTile;
             i += WarpRole::kCalcWarpCount) {
            tile.quantize(i / kBlocksPerTile, i % kBlocksPerTile, act,
                          &smem->shared_expert.quantized_input.data[0],
                          smem->shared_expert.quantized_input.scale_ptr(),
                          k_tile * SharedUpGemm::kTileShapeK);
        }
        cutlass::arch::fence_view_async_shared();
        WarpRole::sync_calc_threads();
        if (warp_idx == 0u && lane_idx == 0u) {
            smem->barriers.shared_up.input_empty.arrive(k_iter_id);
        }
    }

    // Quantize the slab's gated activations into the down's staging layout.
    __device__ static void quantize_output(
        Workspace const* __restrict__ workspace,
        SharedStorage* __restrict__ smem, std::uint32_t gemm_n,
        std::uint32_t slab) {
        const std::uint32_t warp_idx = WarpRole::get_warp_idx();
        const std::uint32_t lane_idx = WarpRole::get_lane_idx();
        constexpr std::uint32_t kQuantGroups = Base::kOutputFeatureBlock / 32u;
        constexpr auto kGroupsPerSlab = Base::OutputQuantizer::kNumGroups;
        const std::uint32_t first_group = slab * kGroupsPerSlab;
#pragma unroll 1
        for (std::uint32_t tok = warp_idx; tok < gemm_n;
             tok += WarpRole::kCalcWarpCount) {
            DataType::ElementQA quantized[kQuantGroups];
            typename Base::OutputQuantizer::ScaleArray scales;
            typename Base::OutputQuantizer::Values values;
#pragma unroll
            for (std::uint32_t i = 0; i < kQuantGroups; ++i) {
                values[i] =
                    smem->shared_expert.act_up_out[lane_idx + i * 32u][tok];
            }
            Base::OutputQuantizer::quantize(values, quantized, scales);
            if (lane_idx == 0u) {
#pragma unroll
                for (std::uint32_t i = 0; i < kGroupsPerSlab; ++i) {
                    workspace->shared_up_out.scale(tok, first_group + i) =
                        scales[i];
                }
            }
#pragma unroll
            for (std::uint32_t i = 0; i < kQuantGroups; ++i) {
                const std::uint32_t feat = lane_idx + i * 32u;
                const std::uint32_t feature =
                    slab * Base::kOutputFeatureBlock + feat;
                workspace->shared_up_out.value(tok, feature) = quantized[i];
            }
        }
        if constexpr (SharedUpGemm::MmaCore::kUseBlockscaled) {
            // The padded rows of the tile's slab stay zero for UTCCP.
            if (lane_idx == 0u) {
                for (std::uint32_t tok = gemm_n + warp_idx; tok < 128u;
                     tok += WarpRole::kCalcWarpCount) {
#pragma unroll
                    for (std::uint32_t i = 0; i < kGroupsPerSlab; ++i) {
                        workspace->shared_up_out.scale(tok, first_group + i) =
                            0u;
                    }
                }
            }
        }
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

    static constexpr std::uint32_t kNumDownSlots =
        GemmSlots<Dims, SharedDownGemm>::kNumSlots;

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

    __device__ static std::uint32_t cta_idx(MoERuntime const& runtime) {
        return Shared::cta_idx(runtime);
    }

    __device__ static void run_load_input_k_loop(
        Workspace const* __restrict__ workspace,
        SharedStorage* __restrict__ smem, std::uint32_t num_k_tiles,
        MoERuntime const& runtime) {
        workspace->cta_comm->wait_shared_up_tasks(
            smem->launch_parity, Shared::num_up_publishers(runtime));
        // Order cross-CTA stores before TMA reads.
        fence_proxy_async_global();

        // Blockwise uses a full activation-scale table; blockscaled stages one
        // SFB per input slot.
        const std::uint32_t scale_bytes =
            SharedDownGemm::MmaCore::kUseBlockscaled
                ? decltype(workspace->shared_up_out)::kSfAtomBytes
                : static_cast<std::uint32_t>(
                      num_k_tiles *
                      sizeof(smem->shared_expert.act_down_input.scale[0]));
        constexpr std::uint32_t kInputBytes = static_cast<std::uint32_t>(
            sizeof(decltype(smem->shared_expert.act_down_input.data[0])));
#pragma unroll 1
        for (std::uint32_t k_iter_id = 0; k_iter_id < num_k_tiles;
             ++k_iter_id) {
            const std::uint32_t ring = k_iter_id % kNumDownSlots;
            smem->barriers.shared_down.wait_weight_empty(k_iter_id);
            const bool load_scale =
                SharedDownGemm::MmaCore::kUseBlockscaled || k_iter_id == 0u;
            const std::uint32_t total_bytes =
                kInputBytes + (load_scale ? scale_bytes : 0u);
            smem->barriers.shared_down.input_ready.arrive(k_iter_id,
                                                          total_bytes);
            const std::uint32_t first_feature =
                k_iter_id * SharedDownGemm::kTileShapeK;
            tma_load_1d(&workspace->shared_up_out.value(0u, first_feature),
                        kInputBytes,
                        &smem->shared_expert.act_down_input.data[ring][0][0][0],
                        smem->barriers.shared_down.input_ready.raw(ring));
            if (load_scale) {
                const auto* scale_src = &workspace->shared_up_out.scale(
                    0u, first_feature / SharedDownGemm::InputQuant::kBlockSize);
                auto* scale_dst = reinterpret_cast<std::uint8_t*>(
                    &smem->shared_expert.act_down_input
                         .scale[SharedDownGemm::MmaCore::kUseBlockscaled ? ring
                                                                         : 0u]
                               [0]);
                tma_load_1d(scale_src, scale_bytes, scale_dst,
                            smem->barriers.shared_down.input_ready.raw(ring));
            }
        }
    }

    __device__ static void load_down_weight_tile(
        CUtensorMap const& tma_down_desc, SharedStorage* __restrict__ smem,
        std::uint32_t k_iter_id, std::uint32_t transaction_bytes,
        std::uint32_t row_base) {
        const std::uint32_t slot = k_iter_id % kNumDownSlots;
        smem->barriers.shared_down.wait_weight_empty(k_iter_id);
        smem->barriers.shared_down.weight_ready.arrive(k_iter_id,
                                                       transaction_bytes);
        Base::template load_weight_tile<SharedDownGemm::kTileShapeM>(
            tma_down_desc, &smem->shared_expert.w_down[slot][0][0],
            smem->barriers.shared_down.weight_ready.raw(slot), row_base,
            k_iter_id);
    }

    __device__ static std::uint32_t row_base(MoERuntime const& runtime) {
        return cta_idx(runtime) * SharedDownGemm::kTileShapeM;
    }

    // Store one warp fragment of the completed down projection.
    __device__ static void store_down_output(
        const WorkspaceTensor<DataType::ElementOut, 2>& out,
        std::uint32_t gemm_n, std::uint32_t atom, std::uint32_t group,
        const float (&values)[4], std::uint32_t row_base) {
        const std::uint32_t warp_idx = WarpRole::get_warp_idx();
        const std::uint32_t lane_idx = WarpRole::get_lane_idx();
        const std::uint32_t lane_base =
            32u * (warp_idx & 3u) + 16u * (warp_idx >> 2);
        const std::uint32_t t0 = lane_idx & 3u;
        const std::uint32_t t1 = lane_idx >> 2;
        for (std::uint32_t r = 0; r < 2u; ++r) {
            const std::uint32_t col =
                row_base + atom * 128u + lane_base + t1 + 8u * r;
            for (std::uint32_t i = 0; i < 2u; ++i) {
                const std::uint32_t tok = group * 8u + 2u * t0 + i;
                if (tok >= gemm_n) continue;
                out(tok, col) = __float2bfloat16_rn(values[2u * r + i]);
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
        Workspace const* __restrict__ workspace,
        SharedStorage* __restrict__ smem, MoEProblemSize const& problem_size,
        Activation activation, float situ_beta, float situ_linear_beta,
        MoERuntime const& runtime) {
        const std::uint32_t gemm_n = problem_size.num_tokens;
        zero_sf_slots(smem);

        Accumulator gate;
        Accumulator up;

        const std::uint32_t num_k_tiles =
            Base::k_tiles_per_split(problem_size, runtime);
        const std::uint32_t split = Base::split_idx(runtime);
        const std::uint32_t slab = Base::slab_idx(runtime);
        const std::uint32_t k_tile_base = split * num_k_tiles;
        const std::uint32_t warp_idx = WarpRole::get_warp_idx();
        if (WarpRole::is_other_loader_thread()) {
            Base::run_load_input_k_loop(tma_input_desc, smem, num_k_tiles,
                                        k_tile_base);
        } else if (WarpRole::is_weight_loader_thread()) {
            run_load_weight_k_loop(tma_up_desc, up_scale, smem, num_k_tiles,
                                   k_tile_base, slab, runtime);
        } else if (WarpRole::is_mma_issuer_warp(warp_idx)) {
            run_mma_k_loop(smem, num_k_tiles);
        } else if (WarpRole::is_calc_warp(warp_idx)) {
#pragma unroll 1
            for (std::uint32_t k_iter_id = 0; k_iter_id < num_k_tiles;
                 ++k_iter_id) {
                quantize_input_tile(smem, gemm_n, k_iter_id, k_tile_base);
            }
            load_accumulators(smem, gate, up);
        }
        Base::combine_gated_activation(smem, activation, situ_beta,
                                       situ_linear_beta, gate, up, runtime);
        if (split == 0u && WarpRole::is_calc_warp(warp_idx)) {
            quantize_output(workspace, smem, gemm_n, slab);
        }
        // All split readers and publishers finish before the next phase.
        __syncthreads();
        ClusterSplitK::sync();
        if (split == 0u && threadIdx.x == 0u) {
            Base::publish_up_output(workspace, smem);
        }
    }

    __device__ static void run_load_weight_k_loop(
        CUtensorMap const& tma_up_desc,
        const typename SharedUpGemm::ElementScale* __restrict__ up_scale,
        SharedStorage* __restrict__ smem, std::uint32_t num_k_tiles,
        std::uint32_t k_tile_base, std::uint32_t slab,
        MoERuntime const& runtime) {
        const std::uint32_t scale_slabs =
            (num_k_tiles * runtime.num_shared_splits) *
            (SharedUpGemm::kTileShapeK / 128u);
#pragma unroll 1
        for (std::uint32_t k_iter_id = 0; k_iter_id < num_k_tiles;
             ++k_iter_id) {
            Base::load_up_weight_tile(tma_up_desc, smem, k_iter_id,
                                      Base::kUpWeightTxBytes, k_tile_base,
                                      slab);
            const std::uint32_t slot = k_iter_id % Base::kNumUpSlots;
            Base::load_weight_scale_tile(
                up_scale, slab, k_tile_base + k_iter_id,
                smem->shared_expert.w_up_scale[slot],
                smem->barriers.shared_up.weight_ready.raw(slot), scale_slabs);
        }
    }

    __device__ static void run_mma_k_loop(SharedStorage* __restrict__ smem,
                                          std::uint32_t num_k_tiles) {
        typename SharedUpGemm::TiledMma tiled_mma{};
        auto cta_mma = tiled_mma.get_slice(cute::_0{});
        auto tCt = Base::make_accumulator_fragment(cta_mma);
#pragma unroll 1
        for (std::uint32_t k_iter_id = 0; k_iter_id < num_k_tiles;
             ++k_iter_id) {
            const std::uint32_t ring = k_iter_id % Base::kNumUpSlots;
            smem->barriers.shared_up.template wait_mma_ready<true, false>(
                k_iter_id);
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

    // Read the completed split accumulator.
    __device__ static void load_accumulators(SharedStorage* __restrict__ smem,
                                             Accumulator& gate,
                                             Accumulator& up) {
        smem->barriers.shared_up.accumulator_ready.wait(0u);
        tcgen05_fence_after_thread_sync();
        Base::load_accumulators(smem->tmem_ptr, gate, up);
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

    using Base::kTmemAccSlots;

    __device__ static void run(
        CUtensorMap const& tma_up_desc, CUtensorMap const& tma_input_desc,
        const typename SharedUpGemm::ElementScale* __restrict__ up_scale,
        Workspace const* __restrict__ workspace,
        SharedStorage* __restrict__ smem, MoEProblemSize const& problem_size,
        Activation activation, float situ_beta, float situ_linear_beta,
        MoERuntime const& runtime) {
        const std::uint32_t gemm_m = 2u * problem_size.intermediate_size;
        const std::uint32_t gemm_n = problem_size.num_tokens;
        const std::uint32_t warp_idx = WarpRole::get_warp_idx();
        const std::uint32_t num_k_tiles =
            Base::k_tiles_per_split(problem_size, runtime);
        const std::uint32_t split = Base::split_idx(runtime);
        const std::uint32_t slab = Base::slab_idx(runtime);
        const std::uint32_t k_tile_base = split * num_k_tiles;
        Accumulator gate{};
        Accumulator up{};
        if (WarpRole::is_other_loader_thread()) {
            Base::run_load_input_k_loop(tma_input_desc, smem, num_k_tiles,
                                        k_tile_base);
        } else if (WarpRole::is_weight_loader_thread()) {
            run_load_weight_k_loop(tma_up_desc, smem, num_k_tiles, k_tile_base,
                                   slab);
        } else if (WarpRole::is_mma_issuer_warp(warp_idx)) {
            run_mma_k_loop(smem, num_k_tiles);
        } else if (WarpRole::is_calc_warp(warp_idx)) {
            run_quantize_and_dequant_k_loop(up_scale, smem, gemm_n, gate, up,
                                            num_k_tiles, gemm_m, slab,
                                            k_tile_base, runtime);
        }
        Base::combine_gated_activation(smem, activation, situ_beta,
                                       situ_linear_beta, gate, up, runtime);
        if (split == 0u && WarpRole::is_calc_warp(warp_idx)) {
            quantize_output(workspace, smem, gemm_n, slab);
        }
        // All split readers and publishers finish before the next phase.
        __syncthreads();
        ClusterSplitK::sync();
        if (split == 0u && threadIdx.x == 0u) {
            Base::publish_up_output(workspace, smem);
        }
    }

    __device__ static void run_load_weight_k_loop(
        CUtensorMap const& tma_up_desc, SharedStorage* __restrict__ smem,
        std::uint32_t num_k_tiles, std::uint32_t k_tile_base,
        std::uint32_t slab) {
#pragma unroll 1
        for (std::uint32_t k_iter_id = 0; k_iter_id < num_k_tiles;
             ++k_iter_id) {
            Base::load_up_weight_tile(tma_up_desc, smem, k_iter_id,
                                      Base::kUpWeightTxBytes, k_tile_base,
                                      slab);
        }
    }

    __device__ static void run_mma_k_loop(SharedStorage* __restrict__ smem,
                                          std::uint32_t num_k_tiles) {
        auto& sh = smem->shared_expert;
        typename SharedUpGemm::TiledMma tiled_mma{};
        auto cta_mma = tiled_mma.get_slice(cute::_0{});
        auto tCt = Base::make_accumulator_fragment(cta_mma);
#pragma unroll 1
        for (std::uint32_t k_iter_id = 0; k_iter_id < num_k_tiles;
             ++k_iter_id) {
            const std::uint32_t ring = k_iter_id % Base::kNumUpSlots;
            const std::uint32_t acc_slot = k_iter_id % kTmemAccSlots;
            smem->barriers.shared_up.template wait_mma_ready<true, true>(
                k_iter_id);
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
        SharedStorage* __restrict__ smem, std::uint32_t gemm_n,
        Accumulator& gate, Accumulator& up, std::uint32_t num_k_tiles,
        std::uint32_t gemm_m, std::uint32_t slab, std::uint32_t k_tile_base,
        MoERuntime const& runtime) {
        const std::uint32_t k_blocks_base =
            k_tile_base *
            (SharedUpGemm::kTileShapeK / SharedUpGemm::WeightQuant::kBlockSize);
        const std::uint32_t scale_blocks =
            num_k_tiles * runtime.num_shared_splits *
            (SharedUpGemm::kTileShapeK / SharedUpGemm::WeightQuant::kBlockSize);
        const std::uint32_t gate_row = slab * Base::kOutputFeatureBlock /
                                       SharedUpGemm::WeightQuant::kBlockSize;
        const std::uint32_t up_row =
            gemm_m / 2u / SharedUpGemm::WeightQuant::kBlockSize + gate_row;
        quantize_input_tile(smem, gemm_n, 0u, k_tile_base);
#pragma unroll 1
        for (std::uint32_t k_iter_id = 1; k_iter_id < num_k_tiles;
             ++k_iter_id) {
            quantize_input_tile(smem, gemm_n, k_iter_id, k_tile_base);
            dequant(up_scale, smem, k_iter_id - 1u, gate, up, gate_row, up_row,
                    k_blocks_base, scale_blocks);
        }
        dequant(up_scale, smem, num_k_tiles - 1u, gate, up, gate_row, up_row,
                k_blocks_base, scale_blocks);
    }

    __device__ static void dequant(
        const typename SharedUpGemm::ElementScale* __restrict__ up_scale,
        SharedStorage* __restrict__ smem, std::uint32_t k_iter_id,
        Accumulator& gate, Accumulator& up, std::uint32_t gate_row,
        std::uint32_t up_row, std::uint32_t k_blocks_base,
        std::uint32_t scale_blocks) {
        const std::uint32_t warp_idx = WarpRole::get_warp_idx();
        const std::uint32_t lane_idx = WarpRole::get_lane_idx();
        const std::uint32_t acc_slot = k_iter_id % kTmemAccSlots;
        smem->barriers.shared_up.accumulator_ready.wait(k_iter_id);
        tcgen05_fence_after_thread_sync();

        const bool reads_tile = Base::owns_up_rows(warp_idx);
        constexpr std::uint32_t kBlocksPerTile =
            SharedUpGemm::kTileShapeK / SharedUpGemm::WeightQuant::kBlockSize;
        const std::uint32_t kblk_base =
            k_blocks_base + k_iter_id * kBlocksPerTile;
        const std::uint32_t act_slot = k_iter_id % Base::kNumUpSlots;
#pragma unroll 1
        for (std::uint32_t s = 0; reads_tile && s < 2u; ++s) {
            const std::uint32_t kblk = kblk_base + s;
            const float ws_gate = up_scale[gate_row * scale_blocks + kblk];
            const float ws_up = up_scale[up_row * scale_blocks + kblk];
            Base::dequant(smem->tmem_ptr,
                          acc_slot * get_gemm_acc_columns<SharedUpGemm>() +
                              s * SharedUpGemm::kTileShapeN,
                          &smem->shared_expert.quantized_input
                               .scale[act_slot][s * SharedUpGemm::kTileShapeN],
                          ws_gate, ws_up, gate, up);
        }
        cutlass::arch::fence_view_async_tmem_load();
        WarpRole::sync_calc_threads();
        if (warp_idx == 0u && lane_idx == 0u) {
            smem->barriers.shared_up.accumulator_empty.arrive(k_iter_id);
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
        CUtensorMap const& tma_down_scale_desc,
        const typename SharedDownGemm::ElementScale* __restrict__ down_scale,
        Workspace const* __restrict__ workspace,
        SharedStorage* __restrict__ smem, ScaleLayout down_scale_layout,
        MoEProblemSize const& problem_size, MoERuntime const& runtime) {
        const std::uint32_t gemm_n = problem_size.num_tokens;
        const std::uint32_t gemm_k = problem_size.intermediate_size;
        const std::uint32_t warp_idx = WarpRole::get_warp_idx();
        const std::uint32_t num_k_tiles = gemm_k / SharedDownGemm::kTileShapeK;
        const std::uint32_t row_base = Base::row_base(runtime);
        if (WarpRole::is_other_loader_thread()) {
            Base::run_load_input_k_loop(workspace, smem, num_k_tiles, runtime);
        } else if (WarpRole::is_weight_loader_thread()) {
            run_load_weight_k_loop(tma_down_desc, tma_down_scale_desc,
                                   down_scale, smem, down_scale_layout,
                                   num_k_tiles, row_base);
        } else if (WarpRole::is_mma_issuer_warp(warp_idx)) {
            run_mma_k_loop(smem, down_scale, down_scale_layout, num_k_tiles,
                           row_base);
        } else if (WarpRole::is_calc_warp(warp_idx)) {
            epilogue(workspace->shared_down_out, smem, gemm_n, runtime);
        }
    }

    __device__ static void run_load_weight_k_loop(
        CUtensorMap const& tma_down_desc,
        CUtensorMap const& tma_down_scale_desc,
        const typename SharedDownGemm::ElementScale* __restrict__ down_scale,
        SharedStorage* __restrict__ smem, ScaleLayout down_scale_layout,
        std::uint32_t num_k_tiles, std::uint32_t row_base) {
        const std::uint32_t row_atom = row_base / 128u;
        const std::uint32_t tx_bytes =
            Base::down_weight_tx_bytes(down_scale_layout);
        const CUtensorMap* const scale_desc =
            down_scale_layout == ScaleLayout::kSfAtom ? &tma_down_scale_desc
                                                      : nullptr;
#pragma unroll 1
        for (std::uint32_t k_iter_id = 0; k_iter_id < num_k_tiles;
             ++k_iter_id) {
            Base::load_down_weight_tile(tma_down_desc, smem, k_iter_id,
                                        tx_bytes, row_base);
            const std::uint32_t slot = k_iter_id % Base::kNumDownSlots;
            Base::load_weight_scale_tile(
                down_scale, row_atom, k_iter_id,
                smem->shared_expert.w_down_scale[slot],
                smem->barriers.shared_down.weight_ready.raw(slot),
                down_scale_layout, num_k_tiles, scale_desc);
        }
    }

    __device__ static void run_mma_k_loop(
        SharedStorage* __restrict__ smem,
        const typename SharedDownGemm::ElementScale* __restrict__ down_scale,
        ScaleLayout down_scale_layout, std::uint32_t num_k_tiles,
        std::uint32_t row_base) {
        typename SharedDownGemm::TiledMma tiled_mma{};
        auto cta_mma = tiled_mma.get_slice(cute::_0{});
        auto tCt = Base::make_accumulator_fragment(cta_mma);
#pragma unroll 1
        for (std::uint32_t k_iter_id = 0; k_iter_id < num_k_tiles;
             ++k_iter_id) {
            const std::uint32_t slot = k_iter_id % Base::kNumDownSlots;
            smem->barriers.shared_down.wait_mma_ready(k_iter_id);
            if (down_scale_layout == ScaleLayout::kPlain) {
                Base::build_plain_scale(
                    smem->shared_expert.w_down_scale[slot],
                    reinterpret_cast<const std::uint8_t*>(down_scale), row_base,
                    k_iter_id,
                    num_k_tiles * (SharedDownGemm::kTileShapeK / 32u));
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
    __device__ static void epilogue(
        const WorkspaceTensor<DataType::ElementOut, 2>& out,
        SharedStorage* __restrict__ smem, std::uint32_t gemm_n,
        MoERuntime const& runtime) {
        const std::uint32_t row_base =
            Base::cta_idx(runtime) * SharedDownGemm::kTileShapeM;
        const std::uint32_t warp_idx = WarpRole::get_warp_idx();
        const std::uint32_t lane_base =
            32u * (warp_idx & 3u) + 16u * (warp_idx >> 2);
        smem->barriers.shared_down.accumulator_ready.wait(0u);
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
                Base::store_down_output(out, gemm_n, atom, group, values,
                                        row_base);
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
        [[maybe_unused]] CUtensorMap const& tma_down_scale_desc,
        const typename SharedDownGemm::ElementScale* __restrict__ down_scale,
        Workspace const* __restrict__ workspace,
        SharedStorage* __restrict__ smem,
        [[maybe_unused]] ScaleLayout down_scale_layout,
        MoEProblemSize const& problem_size, MoERuntime const& runtime) {
        const std::uint32_t gemm_n = problem_size.num_tokens;
        const std::uint32_t gemm_k = problem_size.intermediate_size;
        const std::uint32_t warp_idx = WarpRole::get_warp_idx();
        const std::uint32_t num_k_tiles = gemm_k / SharedDownGemm::kTileShapeK;
        const std::uint32_t row_base = Base::row_base(runtime);
        if (WarpRole::is_other_loader_thread()) {
            Base::run_load_input_k_loop(workspace, smem, num_k_tiles, runtime);
        } else if (WarpRole::is_weight_loader_thread()) {
            run_load_weight_k_loop(tma_down_desc, smem, num_k_tiles, row_base);
        } else if (WarpRole::is_mma_issuer_warp(warp_idx)) {
            run_mma_k_loop(smem, num_k_tiles);
        } else if (WarpRole::is_calc_warp(warp_idx)) {
            epilogue(down_scale, workspace->shared_down_out, smem, gemm_n,
                     num_k_tiles, runtime);
        }
    }

    __device__ static void run_load_weight_k_loop(
        CUtensorMap const& tma_down_desc, SharedStorage* __restrict__ smem,
        std::uint32_t num_k_tiles, std::uint32_t row_base) {
#pragma unroll 1
        for (std::uint32_t k_iter_id = 0; k_iter_id < num_k_tiles;
             ++k_iter_id) {
            Base::load_down_weight_tile(
                tma_down_desc, smem, k_iter_id,
                SharedDownGemm::MmaCore::kWeightTileTxBytes, row_base);
        }
    }

    __device__ static void run_mma_k_loop(SharedStorage* __restrict__ smem,
                                          std::uint32_t num_k_tiles) {
        typename SharedDownGemm::TiledMma tiled_mma{};
        auto cta_mma = tiled_mma.get_slice(cute::_0{});
        auto tCt = Base::make_accumulator_fragment(cta_mma);
#pragma unroll 1
        for (std::uint32_t k_iter_id = 0; k_iter_id < num_k_tiles;
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
        const WorkspaceTensor<DataType::ElementOut, 2>& out,
        SharedStorage* __restrict__ smem, std::uint32_t gemm_n,
        std::uint32_t num_k_tiles, MoERuntime const& runtime) {
        const std::uint32_t cta_idx = Base::cta_idx(runtime);
        const std::uint32_t row_base = cta_idx * SharedDownGemm::kTileShapeM;
        const std::uint32_t warp_idx = WarpRole::get_warp_idx();
        const std::uint32_t lane_idx = WarpRole::get_lane_idx();
        // The routed down's readout: warp w covers lanes 32*(w&3)+16*(w>>2).
        const std::uint32_t lane_base =
            32u * (warp_idx & 3u) + 16u * (warp_idx >> 2);
        const std::uint32_t t0 = lane_idx & 3u;
        smem->barriers.shared_down.accumulator_ready.wait(0u);
#pragma unroll
        for (std::uint32_t atom = 0; atom < SharedDownGemm::MmaCore::kNumAtomMs;
             ++atom) {
#pragma unroll
            for (std::uint32_t group = 0;
                 group < SharedDownGemm::kTileShapeN / 8u; ++group) {
                float acc[4] = {};
#pragma unroll 1
                for (std::uint32_t t = 0; t < num_k_tiles; ++t) {
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
                            [(cta_idx * SharedDownGemm::MmaCore::kNumAtomMs +
                              atom) *
                                 num_k_tiles +
                             t];
                        Base::dequant(
                            v,
                            &smem->shared_expert.act_down_input
                                 .scale[t][s * SharedDownGemm::kTileShapeN +
                                           group * 8u + 2u * t0],
                            ws, acc);
                    }
                }
                Base::store_down_output(out, gemm_n, atom, group, acc,
                                        row_base);
            }
        }
    }
};

}  // namespace halops::moe
