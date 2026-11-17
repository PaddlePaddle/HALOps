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

    static constexpr std::uint32_t kNumCtas = LatentDownGemm::kNumCtas;
    static constexpr std::uint32_t kNumTilesM = LatentDownGemm::kNumTilesM;

    static constexpr std::uint32_t kNumInputSlots =
        LatentDownGemm::kNumInputSlots;
    static constexpr std::uint32_t kNumWeightSlots =
        LatentDownGemm::kNumWeightSlots;

    static constexpr std::uint32_t kNumSplits = LatentDownGemm::kNumSplits;

    static constexpr float kFp8Max = 448.0f;
    static constexpr std::uint32_t kOutChunkK =
        LatentDownSmem<LatentDownGemm>::kOutLboChunkK;
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

    __device__ static bool is_latent_cta() {
        return blockIdx.x >= RouterGemm::kNumCtas &&
               blockIdx.x < RouterGemm::kNumCtas + kNumCtas;
    }

    __device__ static std::uint32_t cta_idx() {
        return blockIdx.x - RouterGemm::kNumCtas;
    }

    __device__ static std::uint32_t split_idx() {
        return cta_idx() % kNumSplits;
    }

    __device__ static std::uint32_t k_tile(std::uint32_t k_iter_id,
                                           std::uint32_t num_k_tiles) {
        return split_idx() * num_k_tiles + k_iter_id;
    }

    // K tiles of this block's split: the reduction is `hidden` wide.
    __device__ static std::uint32_t num_k_tiles_per_cta(
        MoEProblemSize const& problem_size) {
        return problem_size.hidden_size / LatentDownGemm::kTileShapeK /
               kNumSplits;
    }

    using Cluster = ClusterSplitK<Traits::kClusterSize, kNumSplits>;

    __device__ static std::uint32_t token_begin() {
        return (cta_idx() / (kNumSplits * kNumTilesM)) * kTileShapeN;
    }

    __device__ static std::uint32_t row_base() {
        return ((cta_idx() / kNumSplits) % kNumTilesM) * kTileShapeM;
    }

    __device__ static std::uint32_t tile_idx() { return row_base() / 128u; }

    __device__ static Element* staged_block(SharedStorage* smem,
                                            std::uint32_t k_iter_id) {
        return reinterpret_cast<Element*>(
            &smem->latent_down.quantized_input
                 .data[k_iter_id % kNumInputSlots][0][0][0]);
    }

    __device__ static void run_load_input_k_loop(
        CUtensorMap const& tma_input_desc, SharedStorage* __restrict__ smem,
        std::uint32_t num_k_tiles) {
#pragma unroll 1
        for (std::uint32_t k_iter_id = 0; k_iter_id < num_k_tiles;
             ++k_iter_id) {
            const std::uint32_t slot = k_iter_id % kNumInputSlots;
            smem->barriers.latent_down.wait_input_empty(k_iter_id);

            smem->barriers.latent_down.input_ready.arrive(k_iter_id,
                                                          kInputTileTxBytes);
            load_input_tile_chunks<kTileShapeN>(
                tma_input_desc, k_tile(k_iter_id, num_k_tiles), token_begin(),
                smem->latent_down.input.data[slot],
                smem->barriers.latent_down.input_ready.raw(slot));
        }
    }

    __device__ static void load_weight_tile(
        CUtensorMap const& tma_latent_down_weight_desc,
        SharedStorage* __restrict__ smem, std::uint32_t k_iter_id,
        std::uint32_t transaction_bytes, std::uint32_t num_k_tiles) {
        const std::uint32_t slot = k_iter_id % kNumWeightSlots;
        smem->barriers.latent_down.weight_ready.arrive(k_iter_id,
                                                       transaction_bytes);
        constexpr std::uint32_t kChunkK = LatentDownGemm::MmaCore::kSwizzleBoxK;
        const std::uint32_t k_start =
            k_tile(k_iter_id, num_k_tiles) * LatentDownGemm::kTileShapeK;
        tma_load_3d(tma_latent_down_weight_desc, /*coord0=*/0u,
                    /*coord1=*/row_base(), /*coord2=*/k_start / kChunkK,
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

    // Quantize one output cell to FP8, mapping NaNs to zero.
    __device__ static uint4 quantize_cell(SharedStorage* __restrict__ smem,
                                          std::uint32_t cell_chunk,
                                          std::uint32_t cell_tok,
                                          float inv_scale) {
        const DataType::ElementA* const row =
            &smem->latent_down.act_out[cell_chunk][cell_tok][0];
        uint4 quantized = {};
#pragma unroll
        for (std::uint32_t i = 0; i < kOutChunkK; ++i) {
            float v = __bfloat162float(row[i]);
            v = (v == v) ? v : 0.0f;
            reinterpret_cast<unsigned char*>(&quantized)[i] =
                static_cast<unsigned char>(__nv_cvt_float_to_fp8(
                    v * inv_scale, __NV_SATFINITE, __NV_E4M3));
        }
        return quantized;
    }

    __device__ static std::uint32_t group_idx(std::uint32_t warp_idx) {
        return kNarrowAtom ? warp_idx : 2u * (warp_idx & 3u) + (warp_idx >> 2);
    }

    __device__ static std::uint32_t lane_base(std::uint32_t warp_idx) {
        return kNarrowAtom ? 32u * warp_idx : 16u * group_idx(warp_idx);
    }

    __device__ static void writeback(Workspace* __restrict__ workspace,
                                     SharedStorage* __restrict__ smem,
                                     float const (&acc)[kTokenWindows][4]) {
        const std::uint32_t warp_idx = WarpRole::get_warp_idx();
        const std::uint32_t lane_idx = WarpRole::get_lane_idx();
        constexpr std::uint32_t kWriterWarps = kTileShapeM / 16u;
        if (warp_idx >= kWriterWarps) return;
        const std::uint32_t row =
            row_base() + 16u * group_idx(warp_idx) + lane_idx / 4u;
#pragma unroll
        for (std::uint32_t g = 0; g < kTokenWindows; ++g) {
            const std::uint32_t tok_local = 8u * g + 2u * (lane_idx & 3u);
            const std::uint32_t tok = token_begin() + tok_local;
#pragma unroll
            for (std::uint32_t i = 0; i < 4u; ++i) {
                const std::uint32_t row_local =
                    row + 8u * (i / 2u) - row_base();
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
                                        SharedStorage* __restrict__ smem) {
        // Every thread joins: the cluster barrier is block-collective.
        __syncthreads();
        Cluster::sync();
        if (split_idx() == 0u) {
            const std::uint32_t warp_idx = WarpRole::get_warp_idx();
            if (WarpRole::is_calc_warp(warp_idx)) {
                if (warp_idx < kTileShapeM / 16u) {
                    const std::uint32_t lane_idx = WarpRole::get_lane_idx();
                    const std::uint32_t row =
                        16u * group_idx(warp_idx) + lane_idx / 4u;
                    const std::uint32_t t0 = lane_idx & 3u;
                    Cluster cluster;
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
                    writeback(workspace, smem, acc);
                }
                WarpRole::sync_calc_threads();
            }
            // Release peer reads now; wait after output quantization.
            Cluster::arrive();
        } else {
            Cluster::arrive();
            Cluster::wait();
        }
    }

    // Split 0's pending arrive from `join_split_k`.
    __device__ static void release_split_partner() {
        if (split_idx() == 0u) Cluster::wait();
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
    using Cluster = typename Base::Cluster;
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
            Base::kNumInputSlots *
            (128u * (LatentDownGemm::kTileShapeK / 32u)) / 4u;
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
        std::uint32_t slot, std::uint32_t blocks_k) {
        const std::uint32_t lane = WarpRole::get_lane_idx();
        const std::uint8_t* const src =
            natural +
            static_cast<std::size_t>(Base::tile_idx()) * 128u * blocks_k;
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
        ScaleLayout latent_scale_layout, MoEProblemSize const& problem_size) {
        if (!Base::is_latent_cta()) return;

        const std::uint32_t batch_size = problem_size.num_tokens;
        zero_sf_slots(smem);

        const std::uint32_t num_k_tiles =
            Base::num_k_tiles_per_cta(problem_size);
        const std::uint32_t warp_idx = WarpRole::get_warp_idx();
        if (WarpRole::is_other_loader_thread()) {
            Base::run_load_input_k_loop(tma_input_desc, smem, num_k_tiles);
        } else if (WarpRole::is_weight_loader_warp(warp_idx)) {
            run_load_weight_k_loop(
                tma_latent_down_weight_desc,
                reinterpret_cast<const std::uint8_t*>(latent_down_weight_scale),
                smem, num_k_tiles, latent_scale_layout);
        } else if (WarpRole::is_mma_issuer_warp(warp_idx)) {
            run_mma_k_loop(smem, smem->tmem_ptr, num_k_tiles);
        } else if (WarpRole::is_calc_warp(warp_idx)) {
#pragma unroll 1
            for (std::uint32_t k_iter_id = 0; k_iter_id < num_k_tiles;
                 ++k_iter_id) {
                quantize_input_tile(smem, batch_size, k_iter_id, num_k_tiles);
            }
        }
        epilogue(workspace, smem, batch_size);
    }

    // Stream one weight tile and its scale SfAtom through the ring per K tile.
    __device__ static void run_load_weight_k_loop(
        CUtensorMap const& tma_latent_down_weight_desc,
        const std::uint8_t* __restrict__ scale,
        SharedStorage* __restrict__ smem, std::uint32_t num_k_tiles,
        ScaleLayout scale_layout) {
        const std::uint32_t lane = WarpRole::get_lane_idx();
        const std::uint32_t blocks_k = num_k_tiles * Base::kNumSplits *
                                       (LatentDownGemm::kTileShapeK / 32u);
#pragma unroll 1
        for (std::uint32_t k_iter_id = 0; k_iter_id < num_k_tiles;
             ++k_iter_id) {
            const std::uint32_t slot = k_iter_id % Base::kNumWeightSlots;
            const std::uint32_t k_tile = Base::k_tile(k_iter_id, num_k_tiles);
            smem->barriers.latent_down.wait_weight_empty(k_iter_id);
            if (scale_layout == ScaleLayout::kSfAtom) {
                if (lane != 0u) continue;
                Base::load_weight_tile(
                    tma_latent_down_weight_desc, smem, k_iter_id,
                    MmaCore::kWeightTileBytes + ScaleTraits::kSfaSmemBytes,
                    num_k_tiles);
                tma_load_1d(
                    scale + (static_cast<std::size_t>(Base::tile_idx()) *
                                 num_k_tiles * Base::kNumSplits +
                             k_tile) *
                                ScaleTraits::kSfaSmemBytes,
                    ScaleTraits::kSfaSmemBytes, smem->latent_down.w_scale[slot],
                    smem->barriers.latent_down.weight_ready.raw(slot));
                continue;
            }
            build_plain_scale_tile(scale, smem, k_tile, slot, blocks_k);
            // The MMA reads the tile through the async proxy.
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            __syncwarp();
            if (lane == 0u) {
                Base::load_weight_tile(tma_latent_down_weight_desc, smem,
                                       k_iter_id, MmaCore::kWeightTileBytes,
                                       num_k_tiles);
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
            const std::uint32_t w_slot = k_iter_id % Base::kNumWeightSlots;
            smem->barriers.latent_down.template wait_mma_ready<true, false>(
                k_iter_id);
            auto tCsB = cute::make_tensor(
                cute::make_smem_ptr(Base::staged_block(smem, k_iter_id)),
                SBLayout{});
            cute::Tensor tCrB = cta_mma.make_fragment_B(tCsB);
            // SFA and SFB both ring one SfAtom per K tile.
            std::uint8_t* const sfa = &smem->latent_down.w_scale[w_slot][0];
            std::uint8_t* const sfb =
                &smem->latent_down.quantized_input
                     .scale[k_iter_id % Base::kNumInputSlots][0];
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

    __device__ static void quantize_input_tile(SharedStorage* __restrict__ smem,
                                               std::uint32_t batch_size,
                                               std::uint32_t k_iter_id,
                                               std::uint32_t num_k_tiles) {
        const std::uint32_t slot = k_iter_id % Base::kNumInputSlots;
        const std::uint32_t warp_idx = WarpRole::get_calc_warp_idx();
        // Input and weight slots share the previous MMA's completion.
        smem->barriers.latent_down.wait_weight_empty(k_iter_id);
        smem->barriers.latent_down.input_ready.wait(k_iter_id);
        const std::uint32_t block_base =
            Base::k_tile(k_iter_id, num_k_tiles) * MmaCore::kNumSwizzleBoxKs;
        const QuantizeTile tile(slot, block_base, block_base,
                                Base::token_begin(), batch_size);
        quantize_mxfp8_groupwise_1x32<
            /*Staged=*/true, Base::kTileShapeN, LatentDownGemm::kTileShapeK,
            128u, SwizzleBoxK<DataType::ElementA>::value,
            MmaCore::kNumSwizzleBoxKs>(
            reinterpret_cast<const DataType::ElementA*>(
                &smem->latent_down.input.data[slot][0][0][0]),
            Base::k_tile(k_iter_id, num_k_tiles) * LatentDownGemm::kTileShapeK,
            &smem->latent_down.quantized_input.data[0],
            &smem->latent_down.quantized_input.scale[0][0], tile, warp_idx,
            WarpRole::kCalcWarpCount);
        // Make FP8 stores visible to the async proxy before signaling
        // readiness.
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");

        WarpRole::sync_calc_threads();
        if (warp_idx == 0u && WarpRole::get_lane_idx() == 0u) {
            smem->barriers.latent_down.input_empty.arrive(k_iter_id);
        }
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

    // Quantize this block's staged latent output into the workspace.
    __device__ static void quantize_output(Workspace* __restrict__ workspace,
                                           SharedStorage* __restrict__ smem,
                                           std::uint32_t batch_size) {
        constexpr std::uint32_t kCells = Base::kOutChunks * Base::kTileShapeN;
        static_assert(kCells <= Traits::kThreads,
                      "one output cell per thread keeps the quantized bytes "
                      "across the block barrier");
        constexpr std::uint32_t kGroupRows = 32u;
        constexpr std::uint32_t kChunksPerGroup = kGroupRows / Base::kOutChunkK;
        constexpr std::uint32_t kNumGroups = Base::kTileShapeM / kGroupRows;
        static_assert(Base::kTileShapeM % kGroupRows == 0u,
                      "an MXFP8 scale group must not straddle CTAs");
        static_assert(
            Traits::RoutedUpGemm::kTileShapeK % Base::kTileShapeM == 0u,
            "an MXFP8 CTA's rows must sit inside one up K tile");
        const std::uint32_t warp_idx = WarpRole::get_warp_idx();
        const std::uint32_t lane_idx = WarpRole::get_lane_idx();
        const std::uint32_t chunk_base = Base::row_base() / Base::kOutChunkK;

        for (std::uint32_t tok = warp_idx; tok < Base::kTileShapeN;
             tok += WarpRole::kCalcWarpCount) {
            // Each warp reduces one token over 32-feature groups.
#pragma unroll
            for (std::uint32_t g = 0; g < kNumGroups; ++g) {
                const std::uint32_t f = kGroupRows * g + lane_idx;
                const float v = __bfloat162float(
                    smem->latent_down.act_out[f / Base::kOutChunkK][tok]
                                             [f % Base::kOutChunkK]);
                const float mag = fabsf(v);
                const float amax = __uint_as_float(__reduce_max_sync(
                    0xffffffffu, __float_as_uint((mag == mag) ? mag : 0.0f)));
                if (lane_idx == 0u) {
                    smem->latent_down.token_amax[tok][g] = amax;
                }
            }
        }
        __syncthreads();

        const std::uint32_t cell_chunk = threadIdx.x / Base::kTileShapeN;
        const std::uint32_t cell_tok = threadIdx.x % Base::kTileShapeN;
        const std::uint32_t tok = Base::token_begin() + cell_tok;
        uint4 quantized = {};
        if (threadIdx.x < kCells && tok < batch_size) {
            std::uint32_t biased;
            const float inv_scale = mxfp8_ue8m0_scale(
                smem->latent_down
                    .token_amax[cell_tok][cell_chunk / kChunksPerGroup],
                biased);
            quantized =
                Base::quantize_cell(smem, cell_chunk, cell_tok, inv_scale);
        }
        __syncthreads();

        if (threadIdx.x < kCells) {
            workspace->latent_down_out.store_cell(chunk_base + cell_chunk, tok,
                                                  quantized);
        }
        // One thread per token packs four scale exponents into an SfAtom word.
        if (threadIdx.x < Base::kTileShapeN) {
            const std::uint32_t tok_global = Base::token_begin() + threadIdx.x;
            std::uint32_t packed = 0u;
#pragma unroll
            for (std::uint32_t g = 0; g < kNumGroups; ++g) {
                // Use the neutral exponent for padding tokens.
                std::uint32_t biased = 127u;
                if (tok_global < batch_size) {
                    mxfp8_ue8m0_scale(
                        smem->latent_down.token_amax[threadIdx.x][g], biased);
                }
                packed |= biased << (8u * g);
            }
            constexpr std::uint32_t kUpTileShapeK =
                Traits::RoutedUpGemm::kTileShapeK;
            std::uint8_t* const sf = workspace->latent_down_out.sf_slab(
                Base::row_base() / kUpTileShapeK);
            *reinterpret_cast<std::uint32_t*>(
                sf +
                sf_atom_offset(tok_global, (Base::row_base() % kUpTileShapeK) /
                                               kGroupRows)) = packed;
        }
        for (std::uint32_t i = threadIdx.x; i < Base::kOutChunks;
             i += Traits::kThreads) {
            workspace->latent_down_out.clear_pad_row(chunk_base + i);
        }
        __syncthreads();
    }

    __device__ static void epilogue(Workspace* __restrict__ workspace,
                                    SharedStorage* __restrict__ smem,
                                    std::uint32_t batch_size) {
        if (WarpRole::is_calc_warp(WarpRole::get_warp_idx())) {
            float acc[Base::kTokenWindows][4] = {};
            load_accumulators(smem, acc);
            if constexpr (Base::kNumSplits > 1u) {
                Base::store_splited_mma_out(smem, acc);
            } else {
                Base::writeback(workspace, smem, acc);
                WarpRole::sync_calc_threads();
            }
        }
        if constexpr (Base::kNumSplits > 1u) {
            Base::join_split_k(workspace, smem);
        }
        if (Base::split_idx() == 0u) {
            quantize_output(workspace, smem, batch_size);
            if (threadIdx.x == 0u) {
                workspace->cta_comm.publish_latent_tasks(smem->launch_parity,
                                                         1u);
            }
        }
        if constexpr (Base::kNumSplits > 1u) {
            Base::release_split_partner();
        }
    }
};

// Blockwise FP8 applies FP32 scales and emits one scale per 128-row token
// block.
template <typename KernelTraits>
struct LatentDownProjection<KernelTraits, false>
    : LatentDownProjectionBase<KernelTraits> {
    using Base = LatentDownProjectionBase<KernelTraits>;
    using Cluster = typename Base::Cluster;
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

    static constexpr std::uint32_t kNumAccSlots = LatentDownGemm::kNumAccSlots;
    static constexpr std::uint32_t kAccColumns =
        get_gemm_acc_columns<Traits::RoutedUpGemm>();
    static_assert(LatentDownGemm::OutputQuant::kBlockSize <=
                          Base::kTileShapeM ||
                      Base::kNumSplits == 1u,
                  "the output scale block must fit one CTA: only the slice "
                  "owner quantizes, so its peers' token_amax stay unfilled");

    __device__ static void run(
        CUtensorMap const& tma_latent_down_weight_desc,
        CUtensorMap const& tma_input_desc,
        const typename LatentDownGemm::
            ElementScale* __restrict__ latent_down_weight_scale,
        Workspace* __restrict__ workspace, SharedStorage* __restrict__ smem,
        [[maybe_unused]] ScaleLayout latent_scale_layout,
        MoEProblemSize const& problem_size) {
        if (!Base::is_latent_cta()) return;

        const std::uint32_t batch_size = problem_size.num_tokens;
        const std::uint32_t num_k_tiles =
            Base::num_k_tiles_per_cta(problem_size);
        const std::uint32_t warp_idx = WarpRole::get_warp_idx();
        float acc[Base::kTokenWindows][4] = {};
        if (WarpRole::is_other_loader_thread()) {
            Base::run_load_input_k_loop(tma_input_desc, smem, num_k_tiles);
        } else if (WarpRole::is_weight_loader_thread()) {
            run_load_weight_k_loop(tma_latent_down_weight_desc, smem,
                                   num_k_tiles);
        } else if (WarpRole::is_mma_issuer_warp(warp_idx)) {
            run_mma_k_loop(smem, smem->tmem_ptr, num_k_tiles);
        } else if (WarpRole::is_calc_warp(warp_idx)) {
            run_quantize_and_dequant_k_loop(smem, latent_down_weight_scale,
                                            batch_size, num_k_tiles, acc);
        }
        epilogue(workspace, smem, batch_size, acc);
    }

    __device__ static void run_load_weight_k_loop(
        CUtensorMap const& tma_latent_down_weight_desc,
        SharedStorage* __restrict__ smem, std::uint32_t num_k_tiles) {
#pragma unroll 1
        for (std::uint32_t k_iter_id = 0; k_iter_id < num_k_tiles;
             ++k_iter_id) {
            smem->barriers.latent_down.wait_weight_empty(k_iter_id);
            Base::load_weight_tile(tma_latent_down_weight_desc, smem, k_iter_id,
                                   MmaCore::kWeightTileBytes, num_k_tiles);
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
            const std::uint32_t acc_slot = k_iter_id % kNumAccSlots;
            const std::uint32_t w_slot = k_iter_id % Base::kNumWeightSlots;
            smem->barriers.latent_down.template wait_mma_ready<true, true>(
                k_iter_id);
            auto tCsB = cute::make_tensor(
                cute::make_smem_ptr(Base::staged_block(smem, k_iter_id)),
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

    __device__ static void quantize_input_tile(SharedStorage* __restrict__ smem,
                                               std::uint32_t batch_size,
                                               std::uint32_t k_iter_id,
                                               std::uint32_t num_k_tiles) {
        const std::uint32_t slot = k_iter_id % Base::kNumInputSlots;
        const std::uint32_t warp_idx = WarpRole::get_calc_warp_idx();
        smem->barriers.latent_down.input_ready.wait(k_iter_id);

        const std::uint32_t block_base =
            Base::k_tile(k_iter_id, num_k_tiles) * MmaCore::kNumSwizzleBoxKs;
        // The source offset follows the tile's global K block; its scale rows
        // live in the tile's staging slot.
        const QuantizeTile tile(
            slot, block_base,
            /*scale_row_base=*/slot * MmaCore::kNumSwizzleBoxKs,
            Base::token_begin(), batch_size);
        quantize_fp8_groupwise_128<
            /*Staged=*/true, Base::kTileShapeN, LatentDownGemm::kTileShapeK,
            Base::kGemmN, LatentDownGemm::InputQuant::kBlockSize,
            SwizzleBoxK<DataType::ElementA>::value, MmaCore::kNumSwizzleBoxKs>(
            reinterpret_cast<const DataType::ElementA*>(
                &smem->latent_down.input.data[slot][0][0][0]),
            Base::k_tile(k_iter_id, num_k_tiles) * LatentDownGemm::kTileShapeK,
            &smem->latent_down.quantized_input.data[0],
            smem->latent_down.quantized_input.scale_ptr(), tile, warp_idx,
            WarpRole::kCalcWarpCount);

        // Make FP8 stores visible to the async proxy before signaling
        // readiness.
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");

        WarpRole::sync_calc_threads();
        if (warp_idx == 0u && WarpRole::get_lane_idx() == 0u) {
            smem->barriers.latent_down.input_empty.arrive(k_iter_id);
        }
    }

    __device__ static void run_quantize_and_dequant_k_loop(
        SharedStorage* __restrict__ smem,
        const typename LatentDownGemm::ElementScale* __restrict__ w_scale,
        std::uint32_t batch_size, std::uint32_t num_k_tiles,
        float (&acc)[Base::kTokenWindows][4]) {
        constexpr std::uint32_t kStageAhead = 1u;
        quantize_input_tile(smem, batch_size, 0u, num_k_tiles);

#pragma unroll 1
        for (std::uint32_t k_iter_id = 0; k_iter_id < num_k_tiles;
             ++k_iter_id) {
            if (k_iter_id + kStageAhead < num_k_tiles) {
                quantize_input_tile(smem, batch_size, k_iter_id + kStageAhead,
                                    num_k_tiles);
            }
            dequant(smem, w_scale, acc, k_iter_id, num_k_tiles);
        }
    }

    __device__ static float token_block_max(SharedStorage* __restrict__ smem,
                                            std::uint32_t tok) {
        // A max near FLT_MIN would invert to +inf and NaN the block.
        return fmaxf(smem->latent_down.token_amax[tok][0], 1e-10f);
    }

    __device__ static void dequant(
        SharedStorage* __restrict__ smem,
        const typename LatentDownGemm::ElementScale* __restrict__ w_scale,
        float (&acc)[Base::kTokenWindows][4], std::uint32_t k_iter_id,
        std::uint32_t num_k_tiles) {
        const std::uint32_t warp_idx = WarpRole::get_warp_idx();
        const std::uint32_t lane_idx = WarpRole::get_lane_idx();
        // Each warp reads its 16-row TMEM fragment.
        constexpr std::uint32_t kReaderWarps = Base::kTileShapeM / 16u;
        const std::uint32_t t0 = lane_idx & 3u;
        const std::uint32_t token_base = Base::token_begin();
        smem->barriers.latent_down.accumulator_ready.wait(k_iter_id);
        // Each split owns an equal slice of the K blocks.
        const std::uint32_t blocks_per_split =
            num_k_tiles * (LatentDownGemm::kTileShapeK / 128u);
        const std::uint32_t total_blocks = blocks_per_split * Base::kNumSplits;
#pragma unroll
        for (std::uint32_t sub = 0; sub < MmaCore::kNumSwizzleBoxKs; ++sub) {
            const std::uint32_t ws_block =
                k_iter_id * MmaCore::kNumSwizzleBoxKs + sub;
            const float ws =
                w_scale[Base::tile_idx() * total_blocks +
                        Base::split_idx() * blocks_per_split + ws_block];
            float as[2] = {1.0f, 1.0f};
#pragma unroll
            for (std::uint32_t g = 0; g < Base::kTokenWindows; ++g) {
                float v[4];
                // TMEM reads are warp-collective: a warp past the rows skips.
                if (warp_idx < kReaderWarps) {
                    tmem_ld_16x256b1x(
                        tmem_addr(smem->tmem_ptr, Base::lane_base(warp_idx),
                                  (k_iter_id % kNumAccSlots) * kAccColumns +
                                      sub * MmaCore::kAtomShapeN + g * 8u),
                        v);
                    cutlass::arch::fence_view_async_tmem_load();
                }
                // Value i is (row t1 + 8*(i/2), token 8*g + 2*t0 + i%2).
                const std::uint32_t act_slot = k_iter_id % Base::kNumInputSlots;
                const std::uint32_t row =
                    sub * Base::kTileShapeN + token_base + 8u * g + 2u * t0;
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

    // Quantize this block's staged latent output into the workspace.
    __device__ static void quantize_output(Workspace* __restrict__ workspace,
                                           SharedStorage* __restrict__ smem,
                                           std::uint32_t batch_size) {
        constexpr std::uint32_t kCells = Base::kOutChunks * Base::kTileShapeN;
        static_assert(kCells <= Traits::kThreads,
                      "one output cell per thread keeps the quantized bytes "
                      "across the block barrier");
        const std::uint32_t warp_idx = WarpRole::get_warp_idx();
        const std::uint32_t lane_idx = WarpRole::get_lane_idx();
        const std::uint32_t chunk_base = Base::row_base() / Base::kOutChunkK;

        for (std::uint32_t tok = warp_idx; tok < Base::kTileShapeN;
             tok += WarpRole::kCalcWarpCount) {
            float amax = 0.0f;
            for (std::uint32_t i = lane_idx;
                 i < Base::kOutChunks * Base::kOutChunkK;
                 i += WarpRole::kThreadsPerWarp) {
                amax = fmaxf(
                    amax,
                    fabsf(__bfloat162float(
                        smem->latent_down.act_out[i / Base::kOutChunkK][tok]
                                                 [i % Base::kOutChunkK])));
            }
            // Nonnegative floats have the same ordering as their bit patterns.
            const float block_amax = __uint_as_float(
                __reduce_max_sync(0xffffffffu, __float_as_uint(amax)));
            if (lane_idx == 0u) {
                smem->latent_down.token_amax[tok][0] = block_amax;
            }
        }
        __syncthreads();

        const std::uint32_t cell_chunk = threadIdx.x / Base::kTileShapeN;
        const std::uint32_t cell_tok = threadIdx.x % Base::kTileShapeN;
        const std::uint32_t tok = Base::token_begin() + cell_tok;
        uint4 quantized = {};
        if (threadIdx.x < kCells && tok < batch_size) {
            quantized = Base::quantize_cell(
                smem, cell_chunk, cell_tok,
                Base::kFp8Max / token_block_max(smem, cell_tok));
        }
        // Read the scale before releasing the staged output.
        float row_scale = 0.0f;
        if (threadIdx.x < Base::kTileShapeN) {
            row_scale =
                token_block_max(smem, threadIdx.x) * (1.0f / Base::kFp8Max);
            // Use the neutral scale for padding tokens.
            if (Base::token_begin() + threadIdx.x >= batch_size) {
                row_scale = 1.0f;
            }
        }

        if (threadIdx.x < kCells) {
            workspace->latent_down_out.store_cell(chunk_base + cell_chunk, tok,
                                                  quantized);
        }
        if (threadIdx.x < Base::kTileShapeN) {
            workspace->latent_down_out.scale_row(
                Base::row_base() /
                LatentDownGemm::OutputQuant::kBlockSize)[Base::token_begin() +
                                                         threadIdx.x] =
                row_scale;
        }
        for (std::uint32_t i = threadIdx.x; i < Base::kOutChunks;
             i += Traits::kThreads) {
            workspace->latent_down_out.clear_pad_row(chunk_base + i);
        }
        __syncthreads();
    }

    // Output phase: reduce the split partials, then let the slice owner
    // quantize and publish.
    __device__ static void epilogue(
        Workspace* __restrict__ workspace, SharedStorage* __restrict__ smem,
        std::uint32_t batch_size, float const (&acc)[Base::kTokenWindows][4]) {
        if (WarpRole::is_calc_warp(WarpRole::get_warp_idx())) {
            if constexpr (Base::kNumSplits > 1u) {
                Base::store_splited_mma_out(smem, acc);
            } else {
                Base::writeback(workspace, smem, acc);
                WarpRole::sync_calc_threads();
            }
        }
        if constexpr (Base::kNumSplits > 1u) {
            Base::join_split_k(workspace, smem);
        }
        if (Base::split_idx() == 0u) {
            quantize_output(workspace, smem, batch_size);
            if (threadIdx.x == 0u) {
                workspace->cta_comm.publish_latent_tasks(smem->launch_parity,
                                                         1u);
            }
        }
        if constexpr (Base::kNumSplits > 1u) {
            Base::release_split_partner();
        }
    }
};

}  // namespace halops::moe
