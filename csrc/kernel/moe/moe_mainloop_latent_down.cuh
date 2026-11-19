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
#include "kernel/moe/moe_cluster.cuh"
#include "kernel/moe/moe_configs.cuh"
#include "kernel/moe/moe_shared_storage.cuh"
#include "kernel/moe/moe_tma.cuh"
#include "kernel/moe/moe_traits.cuh"
#include "kernel/moe/moe_workspace.cuh"
#include "kernel/moe/ptx_utils.cuh"
#include "kernel/moe/quant_utils.cuh"

namespace halops::moe {

// Latent down-projection computation shared by quantization variants.
template <typename KernelTraits>
struct LatentDownProjectionBase {
    using Traits = KernelTraits;
    using Dims = typename Traits::MoEDims;
    using RouterGemm = typename Traits::RouterGemm;
    using LatentDownGemm = typename Traits::LatentDownGemm;
    using SharedStorage = MoESharedStorage<Dims>;
    using Workspace = MoEWorkspace<Dims>;
    using Element = typename LatentDownGemm::MmaElement;
    using MmaCore = typename LatentDownGemm::MmaCore;
    using ScaleTraits = typename MmaCore::ScaleTraits;

    static constexpr std::uint32_t kTileShapeM = LatentDownGemm::kTileShapeM;
    static constexpr std::uint32_t kTileShapeN = LatentDownGemm::kTileShapeN;
    // TMA byte count for the staged BF16 input tile.
    static constexpr std::uint32_t kInputTileTxBytes =
        kTileShapeN * LatentDownGemm::kTileShapeK *
        static_cast<std::uint32_t>(sizeof(DataType::ElementA));
    static_assert(kTileShapeM <= 128u,
                  "a CTA's rows must sit inside one 128-row weight block");

    // The weight, input and scale rings share one depth.
    static constexpr std::uint32_t kNumSlots =
        GemmSlots<Dims, LatentDownGemm>::kNumSlots;

    static constexpr std::uint32_t kOutChunkK =
        LatentDownSmem<Dims, LatentDownGemm>::kOutLboChunkK;
    static constexpr std::uint32_t kOutChunks = kTileShapeM / kOutChunkK;
    static constexpr std::uint32_t kGemmN = LatentDownGemm::kGemmN;

    // 8-token columns per `16x256b` read.
    static constexpr std::uint32_t kTokenWindows = kTileShapeN / 8u;
    // B uses a padded token row in each contiguous K chunk.
    static constexpr std::uint64_t kBLbo =
        static_cast<std::uint64_t>(kTileShapeN + 1u) *
        LboChunkK<typename LatentDownGemm::InputQuant::ElementType>::value;
    using SBLayout = decltype(cute::make_layout(
        cute::make_shape(
            cute::make_shape(cute::Int<kTileShapeN>{},
                             cute::make_shape(cute::_16{}, cute::_2{})),
            cute::_1{}, cute::Int<MmaCore::kNumAtomKs>{}),
        cute::make_stride(
            cute::make_stride(
                cute::_16{}, cute::make_stride(cute::_1{}, cute::Int<kBLbo>{})),
            cute::_0{}, cute::Int<2u * static_cast<std::uint32_t>(kBLbo)>{})));

    // A 128-row atom interleaves 16-row groups across warp sub-partitions.
    static constexpr bool kNarrowAtom = MmaCore::kAtomShapeM == 64u;

    __device__ static std::uint32_t cta_idx(MoERuntime const& runtime) {
        return blockIdx.x - runtime.num_router_ctas * runtime.num_router_gates;
    }

    __device__ static std::uint32_t k_tiles_per_split(
        MoEProblemSize const& problem_size, MoERuntime const& runtime) {
        return problem_size.hidden_size / LatentDownGemm::kTileShapeK /
               runtime.num_latent_splits;
    }

    // The per-task constants every K tile reuses: the runtime split count
    // makes its divisions and modulos too costly to repeat per tile.
    struct SplitView {
        std::uint32_t split;
        std::uint32_t k_tile_base;
        std::uint32_t row_base;
        std::uint32_t scale_block;
    };

    __device__ static SplitView make_view(MoERuntime const& runtime,
                                          std::uint32_t num_k_tiles) {
        const std::uint32_t cta =
            blockIdx.x - runtime.num_router_ctas * runtime.num_router_gates;
        int quotient = 0;
        int remainder = 0;
        runtime.latent_splits_divmod.fast_divmod(quotient, remainder,
                                                 static_cast<int>(cta));
        const std::uint32_t split = static_cast<std::uint32_t>(remainder);
        const std::uint32_t cta_per_split =
            static_cast<std::uint32_t>(quotient);
        const std::uint32_t row_base =
            static_cast<std::uint32_t>(runtime.latent_tiles_divmod.rem(
                static_cast<int>(cta_per_split))) *
            kTileShapeM;
        return SplitView{split, split * num_k_tiles, row_base, row_base / 128u};
    }

    __device__ static Element* input_slot(SharedStorage* smem,
                                          std::uint32_t k_iter_id) {
        return reinterpret_cast<Element*>(
            &smem->latent_down.quantized_input
                 .data[k_iter_id % kNumSlots][0][0][0]);
    }

    __device__ static void run_load_input_k_loop(
        CUtensorMap const& tma_input_desc, SharedStorage* __restrict__ smem,
        std::uint32_t num_k_tiles, SplitView const& view) {
#pragma unroll 1
        for (std::uint32_t k_iter_id = 0; k_iter_id < num_k_tiles;
             ++k_iter_id) {
            const std::uint32_t slot = k_iter_id % kNumSlots;
            smem->barriers.latent_down.wait_input_empty(k_iter_id);

            smem->barriers.latent_down.input_ready.arrive(k_iter_id,
                                                          kInputTileTxBytes);
            load_input_tile_chunks<kTileShapeN>(
                tma_input_desc, view.k_tile_base + k_iter_id,
                /*token_start=*/0u, smem->latent_down.input.data[slot],
                smem->barriers.latent_down.input_ready.raw(slot));
        }
    }

    __device__ static void load_weight_tile(
        CUtensorMap const& tma_latent_down_weight_desc,
        SharedStorage* __restrict__ smem, std::uint32_t k_iter_id,
        std::uint32_t transaction_bytes, SplitView const& view) {
        const std::uint32_t slot = k_iter_id % kNumSlots;
        smem->barriers.latent_down.weight_ready.arrive(k_iter_id,
                                                       transaction_bytes);
        constexpr std::uint32_t kChunkK = LatentDownGemm::MmaCore::kSwizzleBoxK;
        const std::uint32_t k_start =
            (view.k_tile_base + k_iter_id) * LatentDownGemm::kTileShapeK;
        tma_load_3d(tma_latent_down_weight_desc, /*coord0=*/0u,
                    /*coord1=*/view.row_base, /*coord2=*/k_start / kChunkK,
                    &smem->latent_down.w_latent_down[slot][0][0][0],
                    smem->barriers.latent_down.weight_ready.raw(slot));
    }

    template <typename CtaMma>
    __device__ static auto make_accumulator_fragment(CtaMma& cta_mma) {
        // Shape-only proxy: make_fragment_C only reads the shape.
        return cta_mma.make_fragment_C(
            cta_mma.partition_C(cute::make_identity_tensor(cute::make_shape(
                cute::Int<kTileShapeM>{}, cute::Int<kTileShapeN>{}))));
    }

    __device__ static std::uint32_t group_idx(std::uint32_t warp_idx) {
        return kNarrowAtom ? warp_idx : 2u * (warp_idx & 3u) + (warp_idx >> 2);
    }

    __device__ static std::uint32_t lane_base(std::uint32_t warp_idx) {
        return kNarrowAtom ? 32u * warp_idx : 16u * group_idx(warp_idx);
    }

    // Quantize one staged input K tile and hand its slot back to the loader.
    __device__ static void quantize_input_tile(SharedStorage* __restrict__ smem,
                                               std::uint32_t batch_size,
                                               std::uint32_t k_iter_id,
                                               SplitView const& view) {
        const std::uint32_t slot = k_iter_id % kNumSlots;
        const std::uint32_t warp_idx = WarpRole::get_calc_warp_idx();
        if constexpr (LatentDownGemm::InputQuant::kMethod ==
                      QuantMethod::kMXFP8) {
            // Input and weight slots share the previous MMA's completion.
            smem->barriers.latent_down.wait_weight_empty(k_iter_id);
        }
        smem->barriers.latent_down.input_ready.wait(k_iter_id);
        const std::uint32_t k_tile = view.k_tile_base + k_iter_id;
        const std::uint32_t block_base = k_tile * MmaCore::kNumSwizzleBoxKs;
        // The source offset follows the tile's global K block; a blockwise
        // tile's scale rows live in its staging slot.
        const SmemTileQuantizer<
            typename LatentDownGemm::InputQuant, kTileShapeN, 128u,
            SwizzleBoxK<DataType::ElementA>::value, MmaCore::kNumSwizzleBoxKs>
            tile(slot, block_base,
                 /*scale_row_base=*/slot * MmaCore::kNumSwizzleBoxKs,
                 /*row_start=*/0u, batch_size);
        const auto* const act = reinterpret_cast<const DataType::ElementA*>(
            &smem->latent_down.input.data[slot][0][0][0]);
#pragma unroll 1
        for (std::uint32_t i = warp_idx;
             i < kTileShapeN * MmaCore::kNumSwizzleBoxKs;
             i += WarpRole::kCalcWarpCount) {
            tile.quantize(i / MmaCore::kNumSwizzleBoxKs,
                          i % MmaCore::kNumSwizzleBoxKs, act,
                          &smem->latent_down.quantized_input.data[0],
                          smem->latent_down.quantized_input.scale_ptr(),
                          k_tile * LatentDownGemm::kTileShapeK);
        }
        // Make FP8 stores visible to the async proxy before signaling
        // readiness.
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");

        WarpRole::sync_calc_threads();
        if (warp_idx == 0u && WarpRole::get_lane_idx() == 0u) {
            smem->barriers.latent_down.input_empty.arrive(k_iter_id);
        }
    }

    // Quantize this block's staged latent output into the workspace.
    __device__ static void quantize_output(Workspace* __restrict__ workspace,
                                           SharedStorage* __restrict__ smem,
                                           std::uint32_t batch_size,
                                           SplitView const& view) {
        using OutputQuant = typename LatentDownGemm::OutputQuant;
        using Scale = QuantScale<OutputQuant>;
        using Quantizer = RegisterQuantizer<OutputQuant, kTileShapeM>;
        static_assert(kTileShapeM % OutputQuant::kBlockSize == 0u,
                      "a scale group must not straddle CTAs");
        if constexpr (OutputQuant::kMethod == QuantMethod::kMXFP8) {
            static_assert(Traits::RoutedUpGemm::kTileShapeK % kTileShapeM == 0u,
                          "an MXFP8 CTA's rows must sit inside one up K tile");
        }
        const std::uint32_t chunk_base = view.row_base / kOutChunkK;
        const std::uint32_t warp_idx = WarpRole::get_warp_idx();
        // One warp quantizes one token, lane `L` owning the rows `L + 32 * i`.
        if (WarpRole::is_calc_warp(warp_idx)) {
            const std::uint32_t lane_idx = WarpRole::get_lane_idx();
            for (std::uint32_t tok = warp_idx; tok < kTileShapeN;
                 tok += WarpRole::kCalcWarpCount) {
                typename Quantizer::ElementType
                    quantized[Quantizer::kValuesPerThread] = {};
                typename Quantizer::ScaleArray scales;
                if (tok < batch_size) {
                    typename Quantizer::Values values;
#pragma unroll
                    for (std::uint32_t i = 0; i < Quantizer::kValuesPerThread;
                         ++i) {
                        const std::uint32_t row =
                            i * WarpRole::kThreadsPerWarp + lane_idx;
                        const float v = __bfloat162float(
                            smem->latent_down.act_out[row / kOutChunkK][tok]
                                                     [row % kOutChunkK]);
                        values[i] = (v == v) ? v : 0.0f;
                    }
                    Quantizer::quantize(values, quantized, scales);
                } else {
                    // Padding tokens carry the padding scale.
#pragma unroll
                    for (std::uint32_t g = 0; g < Quantizer::kNumGroups; ++g) {
                        scales[g] = Scale::kPadding;
                    }
                }

                const std::uint32_t k_group_base =
                    view.row_base / OutputQuant::kBlockSize;
                if constexpr (OutputQuant::kMethod == QuantMethod::kMXFP8) {
                    static_assert(Quantizer::kNumGroups <= 8u,
                                  "the group scales must pack into one word");
                    std::uint64_t scale_bits = 0u;
#pragma unroll
                    for (std::uint32_t g = 0; g < Quantizer::kNumGroups; ++g) {
                        scale_bits |= static_cast<std::uint64_t>(scales[g])
                                      << (8u * g);
                    }
                    if (lane_idx < Quantizer::kNumGroups) {
                        const std::uint32_t k_group = k_group_base + lane_idx;
                        workspace->latent_down_out.sf_atom(
                            k_group)[sf_atom_offset(tok, k_group % 4u)] =
                            static_cast<std::uint8_t>(scale_bits >>
                                                      (8u * lane_idx));
                    }
                } else {
#pragma unroll
                    for (std::uint32_t g = 0; g < Quantizer::kNumGroups; ++g) {
                        if (lane_idx == g) {
                            workspace->latent_down_out.scale_row(
                                k_group_base + g)[tok] = scales[g];
                        }
                    }
                }
#pragma unroll
                for (std::uint32_t i = 0; i < Quantizer::kValuesPerThread;
                     ++i) {
                    const std::uint32_t row =
                        i * WarpRole::kThreadsPerWarp + lane_idx;
                    workspace->latent_down_out.cell(
                        chunk_base + row / kOutChunkK, tok)[row % kOutChunkK] =
                        quantized[i];
                }
            }
        }
        for (std::uint32_t i = threadIdx.x; i < kOutChunks;
             i += Traits::kThreads) {
            workspace->latent_down_out.clear_pad_row(chunk_base + i);
        }
        __syncthreads();
    }

    __device__ static void epilogue(Workspace* __restrict__ workspace,
                                    SharedStorage* __restrict__ smem,
                                    std::uint32_t batch_size,
                                    float const (&acc)[kTokenWindows][4],
                                    SplitView const& view,
                                    MoERuntime const& runtime) {
        if (WarpRole::is_calc_warp(WarpRole::get_warp_idx())) {
            if (runtime.num_latent_splits > 1u) {
                store_splited_mma_out(smem, acc);
            } else {
                writeback(workspace, smem, acc, view);
                WarpRole::sync_calc_threads();
            }
        }
        if (runtime.num_latent_splits > 1u) {
            join_split_k(workspace, smem, view, runtime);
        }
        if (view.split == 0u) {
            quantize_output(workspace, smem, batch_size, view);
            if (threadIdx.x == 0u) {
                workspace->cta_comm.publish_latent_tasks(smem->launch_parity,
                                                         1u);
            }
        }
        if (runtime.num_latent_splits > 1u) {
            if (view.split == 0u) ClusterSplitK::wait();
        }
    }

    __device__ static void writeback(Workspace* __restrict__ workspace,
                                     SharedStorage* __restrict__ smem,
                                     float const (&acc)[kTokenWindows][4],
                                     SplitView const& view) {
        const std::uint32_t warp_idx = WarpRole::get_warp_idx();
        const std::uint32_t lane_idx = WarpRole::get_lane_idx();
        constexpr std::uint32_t kWriterWarps = kTileShapeM / 16u;
        if (warp_idx >= kWriterWarps) return;
        const std::uint32_t base = view.row_base;
        const std::uint32_t row =
            base + 16u * group_idx(warp_idx) + lane_idx / 4u;
#pragma unroll
        for (std::uint32_t g = 0; g < kTokenWindows; ++g) {
            const std::uint32_t tok_local = 8u * g + 2u * (lane_idx & 3u);
#pragma unroll
            for (std::uint32_t i = 0; i < 4u; ++i) {
                const std::uint32_t row_local = row + 8u * (i / 2u) - base;
                smem->latent_down
                    .act_out[row_local / kOutChunkK][tok_local + i % 2u]
                            [row_local % kOutChunkK] =
                    __float2bfloat16_rn(acc[g][i]);
            }
        }
    }

    // Half of a split-K sum: the calc warps' partials, staged for the peer
    // reduction.
    __device__ static void store_splited_mma_out(
        SharedStorage* __restrict__ smem,
        float const (&acc)[kTokenWindows][4]) {
        const std::uint32_t warp_idx = WarpRole::get_warp_idx();
        const std::uint32_t lane_idx = WarpRole::get_lane_idx();
        if (warp_idx >= kTileShapeM / 16u) return;
        const std::uint32_t row = 16u * group_idx(warp_idx) + lane_idx / 4u;
        const std::uint32_t t0 = lane_idx & 3u;
#pragma unroll
        for (std::uint32_t g = 0; g < kTokenWindows; ++g) {
#pragma unroll
            for (std::uint32_t i = 0; i < 4u; ++i) {
                smem->latent_down.splited_mma_out[row + 8u * (i / 2u)]
                                                 [8u * g + 2u * t0 + i % 2u] =
                    acc[g][i];
            }
        }
    }

    // Split 0 reduces peer outputs and quantizes output.
    __device__ static void join_split_k(Workspace* __restrict__ workspace,
                                        SharedStorage* __restrict__ smem,
                                        SplitView const& view,
                                        MoERuntime const& runtime) {
        // Every thread joins: the cluster barrier is block-collective.
        __syncthreads();
        ClusterSplitK::sync();
        if (view.split == 0u) {
            const std::uint32_t warp_idx = WarpRole::get_warp_idx();
            if (WarpRole::is_calc_warp(warp_idx)) {
                if (warp_idx < kTileShapeM / 16u) {
                    const std::uint32_t lane_idx = WarpRole::get_lane_idx();
                    const std::uint32_t row =
                        16u * group_idx(warp_idx) + lane_idx / 4u;
                    const std::uint32_t t0 = lane_idx & 3u;
                    ClusterSplitK cluster(runtime.cluster_size,
                                          runtime.num_latent_splits);
                    cluster.map(&smem->latent_down.splited_mma_out[0][0]);
                    float acc[kTokenWindows][4] = {};
#pragma unroll
                    for (std::uint32_t g = 0; g < kTokenWindows; ++g) {
#pragma unroll
                        for (std::uint32_t i = 0; i < 4u; ++i) {
                            const std::uint32_t at =
                                (row + 8u * (i / 2u)) * kTileShapeN +
                                (8u * g + 2u * t0 + i % 2u);
                            acc[g][i] = cluster.split_sum(at);
                        }
                    }
                    writeback(workspace, smem, acc, view);
                }
                WarpRole::sync_calc_threads();
            }
            // Release peer reads now; wait after output quantization.
            ClusterSplitK::arrive();
        } else {
            ClusterSplitK::arrive();
            ClusterSplitK::wait();
        }
    }
};

template <typename KernelTraits,
          bool UseBlockscaled =
              KernelTraits::LatentDownGemm::MmaCore::kUseBlockscaled>
struct LatentDownProjection;

// MXFP8 uses blockscaled UMMA and one UE8M0 output scale per 32 rows.
template <typename KernelTraits>
struct LatentDownProjection<KernelTraits, true>
    : LatentDownProjectionBase<KernelTraits> {
    using Base = LatentDownProjectionBase<KernelTraits>;
    using typename Base::Dims;
    using typename Base::LatentDownGemm;
    using typename Base::MmaCore;
    using typename Base::SBLayout;
    using typename Base::ScaleTraits;
    using typename Base::SharedStorage;
    using typename Base::Traits;
    using typename Base::Workspace;

    static_assert(MmaCore::kUseBlockscaled,
                  "LatentDownProjection<true> requires a blockscaled MMA core");

    static constexpr std::uint32_t kAccColumns = Base::kTileShapeN;
    static constexpr std::uint32_t kSfaTmemOffset = kAccColumns;
    static constexpr std::uint32_t kSfbTmemOffset =
        kSfaTmemOffset + ScaleTraits::kSfaTmemColumns;

    // Fill valid SFB rows and zero the padding for UTCCP.
    __device__ static void zero_sf_slots(SharedStorage* __restrict__ smem) {
        constexpr std::uint32_t kSfWords =
            Base::kNumSlots * (128u * (LatentDownGemm::kTileShapeK / 32u)) / 4u;
        auto* const sf = reinterpret_cast<std::uint32_t*>(
            &smem->latent_down.quantized_input.scale[0][0]);
        for (std::uint32_t i = threadIdx.x; i < kSfWords;
             i += static_cast<std::uint32_t>(blockDim.x)) {
            sf[i] = 0u;
        }
        __syncthreads();
    }

    // 128 rows x 4 K groups, the K-tile granularity of the weight's SfAtom.
    static constexpr std::uint32_t kSfAtomBytes = 512u;
    static_assert(ScaleTraits::kSfaSmemBytes == 2u * kSfAtomBytes,
                  "a K tile stages two 4-group SfAtoms");

    // Build one K tile's scale SfAtom from the natural row-major table: the
    // tile owns groups [8k, 8k+8) of every row, two 512-B atoms.
    __device__ static void build_plain_scale_tile(
        const std::uint8_t* __restrict__ natural,
        SharedStorage* __restrict__ smem, std::uint32_t k_tile,
        std::uint32_t slot, std::uint32_t blocks_k, std::uint32_t scale_block) {
        const std::uint32_t lane = WarpRole::get_lane_idx();
        const std::uint8_t* const src =
            natural + static_cast<std::size_t>(scale_block) * 128u * blocks_k;
        std::uint8_t* const dst = smem->latent_down.w_scale[slot];
#pragma unroll
        for (std::uint32_t q = 0; q < 2u; ++q) {
            std::uint32_t words[4];
#pragma unroll
            for (std::uint32_t band = 0; band < 4u; ++band) {
                const std::uint32_t row = 32u * band + lane;
                words[band] = *reinterpret_cast<const std::uint32_t*>(
                    src + static_cast<std::size_t>(row) * blocks_k +
                    k_tile * 8u + 4u * q);
            }
            *reinterpret_cast<uint4*>(dst + q * kSfAtomBytes + lane * 16u) =
                make_uint4(words[0], words[1], words[2], words[3]);
        }
    }

    __device__ static void run(
        CUtensorMap const& tma_latent_down_weight_desc,
        CUtensorMap const& tma_input_desc,
        const typename LatentDownGemm::
            ElementScale* __restrict__ latent_down_weight_scale,
        Workspace* __restrict__ workspace, SharedStorage* __restrict__ smem,
        ScaleLayout latent_scale_layout, MoEProblemSize const& problem_size,
        MoERuntime const& runtime) {
        if (!runtime.is_latent_cta()) return;

        const std::uint32_t batch_size = problem_size.num_tokens;
        zero_sf_slots(smem);

        const std::uint32_t num_k_tiles =
            Base::k_tiles_per_split(problem_size, runtime);
        const auto view = Base::make_view(runtime, num_k_tiles);
        const std::uint32_t warp_idx = WarpRole::get_warp_idx();
        float acc[Base::kTokenWindows][4] = {};
        if (WarpRole::is_other_loader_thread()) {
            Base::run_load_input_k_loop(tma_input_desc, smem, num_k_tiles,
                                        view);
        } else if (WarpRole::is_weight_loader_warp(warp_idx)) {
            run_load_weight_k_loop(
                tma_latent_down_weight_desc,
                reinterpret_cast<const std::uint8_t*>(latent_down_weight_scale),
                smem, num_k_tiles, latent_scale_layout, view, runtime);
        } else if (WarpRole::is_mma_issuer_warp(warp_idx)) {
            run_mma_k_loop(smem, smem->tmem_ptr, num_k_tiles);
        } else if (WarpRole::is_calc_warp(warp_idx)) {
#pragma unroll 1
            for (std::uint32_t k_iter_id = 0; k_iter_id < num_k_tiles;
                 ++k_iter_id) {
                quantize_input_tile(smem, batch_size, k_iter_id, view);
            }
            load_accumulators(smem, acc);
        }
        epilogue(workspace, smem, batch_size, acc, view, runtime);
    }

    // Stream one weight tile and its scale SfAtom through the ring per K tile.
    __device__ static void run_load_weight_k_loop(
        CUtensorMap const& tma_latent_down_weight_desc,
        const std::uint8_t* __restrict__ scale,
        SharedStorage* __restrict__ smem, std::uint32_t num_k_tiles,
        ScaleLayout scale_layout, Base::SplitView const& view,
        MoERuntime const& runtime) {
        const std::uint32_t lane = WarpRole::get_lane_idx();
        const std::uint32_t k_tiles_total =
            num_k_tiles * runtime.num_latent_splits;
        const std::uint32_t blocks_k =
            k_tiles_total * (LatentDownGemm::kTileShapeK / 32u);
        // SfAtom rows hold one slab per K tile; the plain table counts 32-wide
        // K blocks instead.
        const std::size_t sfa_base =
            static_cast<std::size_t>(view.scale_block) * k_tiles_total;
#pragma unroll 1
        for (std::uint32_t k_iter_id = 0; k_iter_id < num_k_tiles;
             ++k_iter_id) {
            const std::uint32_t slot = k_iter_id % Base::kNumSlots;
            const std::uint32_t k_tile = view.k_tile_base + k_iter_id;
            smem->barriers.latent_down.wait_weight_empty(k_iter_id);
            if (scale_layout == ScaleLayout::kSfAtom) {
                if (lane != 0u) continue;
                Base::load_weight_tile(
                    tma_latent_down_weight_desc, smem, k_iter_id,
                    MmaCore::kWeightTileBytes + ScaleTraits::kSfaSmemBytes,
                    view);
                tma_load_1d(
                    scale + (sfa_base + k_tile) * ScaleTraits::kSfaSmemBytes,
                    ScaleTraits::kSfaSmemBytes, smem->latent_down.w_scale[slot],
                    smem->barriers.latent_down.weight_ready.raw(slot));
                continue;
            }
            build_plain_scale_tile(scale, smem, k_tile, slot, blocks_k,
                                   view.scale_block);
            // The MMA reads the tile through the async proxy.
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            __syncwarp();
            if (lane == 0u) {
                Base::load_weight_tile(tma_latent_down_weight_desc, smem,
                                       k_iter_id, MmaCore::kWeightTileBytes,
                                       view);
            }
        }
    }

    __device__ static void run_mma_k_loop(SharedStorage* __restrict__ smem,
                                          std::uint32_t tmem_base,
                                          std::uint32_t num_k_tiles) {
        typename MmaCore::TiledMma tiled_mma{};
        auto cta_mma = tiled_mma.get_slice(cute::_0{});
        cute::Tensor tCt = Base::make_accumulator_fragment(cta_mma);
#pragma unroll 1
        for (std::uint32_t k_iter_id = 0; k_iter_id < num_k_tiles;
             ++k_iter_id) {
            const std::uint32_t w_slot = k_iter_id % Base::kNumSlots;
            smem->barriers.latent_down.template wait_mma_ready<true, false>(
                k_iter_id);
            auto tCsB = cute::make_tensor(
                cute::make_smem_ptr(Base::input_slot(smem, k_iter_id)),
                SBLayout{});
            cute::Tensor tCrB = cta_mma.make_fragment_B(tCsB);
            // SFA and SFB both ring one SfAtom per K tile.
            std::uint8_t* const sfa = &smem->latent_down.w_scale[w_slot][0];
            std::uint8_t* const sfb =
                &smem->latent_down.quantized_input
                     .scale[k_iter_id % Base::kNumSlots][0];
            if (cute::elect_one_sync()) {
                ScaleTraits::copy_sfa_to_tmem(sfa, tmem_base + kSfaTmemOffset);
                ScaleTraits::copy_sfb_to_tmem(sfb, tmem_base + kSfbTmemOffset);
                MmaCore::run(tiled_mma, cta_mma,
                             &smem->latent_down.w_latent_down[w_slot][0][0][0],
                             tCrB, tCt, tmem_addr(tmem_base, 0u, 0u),
                             tmem_base + kSfaTmemOffset,
                             tmem_base + kSfbTmemOffset,
                             /*accumulate=*/k_iter_id != 0u);
            }

            cutlass::arch::umma_arrive(
                smem->barriers.latent_down.weight_empty.raw(k_iter_id));
        }
        // Publish the completed split once, after all K tiles.
        cutlass::arch::umma_arrive(
            smem->barriers.latent_down.accumulator_ready.raw(0u));
    }

    // Read the completed TMEM sum once for this split.
    __device__ static void load_accumulators(
        SharedStorage* __restrict__ smem,
        float (&acc)[Base::kTokenWindows][4]) {
        const std::uint32_t warp_idx = WarpRole::get_warp_idx();
        // Each warp reads its 16-row TMEM fragment.
        constexpr std::uint32_t kReaderWarps = Base::kTileShapeM / 16u;
        smem->barriers.latent_down.accumulator_ready.wait(0u);
        tcgen05_fence_after_thread_sync();
        if (warp_idx >= kReaderWarps) return;
#pragma unroll
        for (std::uint32_t g = 0; g < Base::kTokenWindows; ++g) {
            tmem_ld_16x256b1x(
                tmem_addr(smem->tmem_ptr, Base::lane_base(warp_idx), g * 8u),
                acc[g]);
            cutlass::arch::fence_view_async_tmem_load();
        }
    }
};

// Blockwise FP8 applies FP32 scales and emits one scale per 128-row token
// block.
template <typename KernelTraits>
struct LatentDownProjection<KernelTraits, false>
    : LatentDownProjectionBase<KernelTraits> {
    using Base = LatentDownProjectionBase<KernelTraits>;
    using typename Base::Dims;
    using typename Base::LatentDownGemm;
    using typename Base::MmaCore;
    using typename Base::SBLayout;
    using typename Base::SharedStorage;
    using typename Base::Traits;
    using typename Base::Workspace;

    static_assert(
        !MmaCore::kUseBlockscaled,
        "LatentDownProjection<false> requires a non-blockscaled MMA core");

    static constexpr std::uint32_t kTmemAccSlots =
        GemmSlots<Dims, LatentDownGemm>::kTmemAccSlots;
    static constexpr std::uint32_t kAccColumns =
        get_gemm_acc_columns<Traits::RoutedUpGemm>();

    __device__ static void run(
        CUtensorMap const& tma_latent_down_weight_desc,
        CUtensorMap const& tma_input_desc,
        const typename LatentDownGemm::
            ElementScale* __restrict__ latent_down_weight_scale,
        Workspace* __restrict__ workspace, SharedStorage* __restrict__ smem,
        [[maybe_unused]] ScaleLayout latent_scale_layout,
        MoEProblemSize const& problem_size, MoERuntime const& runtime) {
        if (!runtime.is_latent_cta()) return;

        const std::uint32_t batch_size = problem_size.num_tokens;
        const std::uint32_t num_k_tiles =
            Base::k_tiles_per_split(problem_size, runtime);
        const auto view = Base::make_view(runtime, num_k_tiles);
        const std::uint32_t warp_idx = WarpRole::get_warp_idx();
        float acc[Base::kTokenWindows][4] = {};
        if (WarpRole::is_other_loader_thread()) {
            Base::run_load_input_k_loop(tma_input_desc, smem, num_k_tiles,
                                        view);
        } else if (WarpRole::is_weight_loader_thread()) {
            run_load_weight_k_loop(tma_latent_down_weight_desc, smem,
                                   num_k_tiles, view);
        } else if (WarpRole::is_mma_issuer_warp(warp_idx)) {
            run_mma_k_loop(smem, smem->tmem_ptr, num_k_tiles);
        } else if (WarpRole::is_calc_warp(warp_idx)) {
            run_quantize_and_dequant_k_loop(smem, latent_down_weight_scale,
                                            batch_size, num_k_tiles, acc, view,
                                            runtime);
        }
        epilogue(workspace, smem, batch_size, acc, view, runtime);
    }

    __device__ static void run_load_weight_k_loop(
        CUtensorMap const& tma_latent_down_weight_desc,
        SharedStorage* __restrict__ smem, std::uint32_t num_k_tiles,
        Base::SplitView const& view) {
#pragma unroll 1
        for (std::uint32_t k_iter_id = 0; k_iter_id < num_k_tiles;
             ++k_iter_id) {
            smem->barriers.latent_down.wait_weight_empty(k_iter_id);
            Base::load_weight_tile(tma_latent_down_weight_desc, smem, k_iter_id,
                                   MmaCore::kWeightTileBytes, view);
        }
    }

    __device__ static void run_mma_k_loop(SharedStorage* __restrict__ smem,
                                          std::uint32_t tmem_base,
                                          std::uint32_t num_k_tiles) {
        typename MmaCore::TiledMma tiled_mma{};
        auto cta_mma = tiled_mma.get_slice(cute::_0{});
        cute::Tensor tCt = Base::make_accumulator_fragment(cta_mma);
#pragma unroll 1
        for (std::uint32_t k_iter_id = 0; k_iter_id < num_k_tiles;
             ++k_iter_id) {
            const std::uint32_t acc_slot = k_iter_id % kTmemAccSlots;
            const std::uint32_t w_slot = k_iter_id % Base::kNumSlots;
            smem->barriers.latent_down.template wait_mma_ready<true, true>(
                k_iter_id);
            auto tCsB = cute::make_tensor(
                cute::make_smem_ptr(Base::input_slot(smem, k_iter_id)),
                SBLayout{});
            cute::Tensor tCrB = cta_mma.make_fragment_B(tCsB);
            MmaCore::run<MmaCore::kNumSwizzleBoxKs>(
                tiled_mma, cta_mma,
                &smem->latent_down.w_latent_down[w_slot][0][0][0], tCrB, tCt,
                tmem_addr(tmem_base, 0u, acc_slot * kAccColumns));

            cutlass::arch::umma_arrive(
                smem->barriers.latent_down.weight_empty.raw(k_iter_id));
            cutlass::arch::umma_arrive(
                smem->barriers.latent_down.accumulator_ready.raw(acc_slot));
        }
    }

    __device__ static void run_quantize_and_dequant_k_loop(
        SharedStorage* __restrict__ smem,
        const typename LatentDownGemm::ElementScale* __restrict__ w_scale,
        std::uint32_t batch_size, std::uint32_t num_k_tiles,
        float (&acc)[Base::kTokenWindows][4], Base::SplitView const& view,
        MoERuntime const& runtime) {
        constexpr std::uint32_t kStageAhead = 1u;
        quantize_input_tile(smem, batch_size, 0u, view);
        // Each split owns an equal slice of the K blocks.
        const std::uint32_t blocks_per_split =
            num_k_tiles * (LatentDownGemm::kTileShapeK / 128u);
        const std::uint32_t scale_base =
            view.scale_block * blocks_per_split * runtime.num_latent_splits +
            view.split * blocks_per_split;

#pragma unroll 1
        for (std::uint32_t k_iter_id = 0; k_iter_id < num_k_tiles;
             ++k_iter_id) {
            if (k_iter_id + kStageAhead < num_k_tiles) {
                quantize_input_tile(smem, batch_size, k_iter_id + kStageAhead,
                                    view);
            }
            dequant(smem, w_scale, acc, k_iter_id, scale_base);
        }
    }

    __device__ static void dequant(
        SharedStorage* __restrict__ smem,
        const typename LatentDownGemm::ElementScale* __restrict__ w_scale,
        float (&acc)[Base::kTokenWindows][4], std::uint32_t k_iter_id,
        std::uint32_t scale_base) {
        const std::uint32_t warp_idx = WarpRole::get_warp_idx();
        const std::uint32_t lane_idx = WarpRole::get_lane_idx();
        // Each warp reads its 16-row TMEM fragment.
        constexpr std::uint32_t kReaderWarps = Base::kTileShapeM / 16u;
        const std::uint32_t t0 = lane_idx & 3u;
        smem->barriers.latent_down.accumulator_ready.wait(k_iter_id);
#pragma unroll
        for (std::uint32_t sub = 0; sub < MmaCore::kNumSwizzleBoxKs; ++sub) {
            const std::uint32_t ws_block =
                k_iter_id * MmaCore::kNumSwizzleBoxKs + sub;
            const float ws = w_scale[scale_base + ws_block];
            float as[2] = {1.0f, 1.0f};
#pragma unroll
            for (std::uint32_t g = 0; g < Base::kTokenWindows; ++g) {
                float v[4];
                // TMEM reads are warp-collective: a warp past the rows skips.
                if (warp_idx < kReaderWarps) {
                    tmem_ld_16x256b1x(
                        tmem_addr(smem->tmem_ptr, Base::lane_base(warp_idx),
                                  (k_iter_id % kTmemAccSlots) * kAccColumns +
                                      sub * MmaCore::kAtomShapeN + g * 8u),
                        v);
                    cutlass::arch::fence_view_async_tmem_load();
                }
                // Value i is (row t1 + 8*(i/2), token 8*g + 2*t0 + i%2).
                const std::uint32_t act_slot = k_iter_id % Base::kNumSlots;
                const std::uint32_t row =
                    sub * Base::kTileShapeN + 8u * g + 2u * t0;
                as[0] = smem->latent_down.quantized_input.scale[act_slot][row];
                as[1] =
                    smem->latent_down.quantized_input.scale[act_slot][row + 1u];
#pragma unroll
                for (std::uint32_t i = 0; i < 4u; ++i) {
                    acc[g][i] += v[i] * ws * as[i % 2u];
                }
            }
        }
        if (lane_idx == 0u) {
            smem->barriers.latent_down.accumulator_empty.arrive(k_iter_id);
        }
    }
};

}  // namespace halops::moe
