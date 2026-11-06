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
    static constexpr std::uint32_t kNumTilesK = LatentDownGemm::kNumTilesK;

    static constexpr std::uint32_t kNumInputSlots =
        LatentDownGemm::kNumInputSlots;
    static constexpr std::uint32_t kNumWeightSlots =
        LatentDownGemm::kNumWeightSlots;
    static constexpr std::uint32_t kNumScaleBlocksK =
        LatentDownGemm::kGemmK / 128u;

    static constexpr std::uint32_t kNumSplits = LatentDownGemm::kNumSplits;
    static constexpr std::uint32_t kTilesKPerCta = kNumTilesK / kNumSplits;

    static constexpr float kFp8Max = 448.0f;
    static constexpr float kFp8MaxInv = 1.0f / 448.0f;
    static constexpr std::uint32_t kOutChunkK =
        LatentDownSmem<LatentDownGemm>::kOutChunkK;
    static constexpr std::uint32_t kOutChunks = kTileShapeM / kOutChunkK;
    static constexpr std::uint32_t kGemmN = LatentDownGemm::kGemmN;

    // 8-token columns per `16x256b` read.
    static constexpr std::uint32_t kTokenWindows = kTileShapeN / 8u;
    // B uses a padded token row in each contiguous K chunk.
    static constexpr std::uint64_t kBLbo =
        static_cast<std::uint64_t>(kTileShapeN + 1u) *
        LboChunkK<typename LatentDownGemm::InputQuant::ElementType>::value;
    static constexpr std::uint32_t kBKStride =
        2u * static_cast<std::uint32_t>(kBLbo);
    using SBLayout = decltype(cute::make_layout(
        cute::make_shape(
            cute::make_shape(cute::Int<kTileShapeN>{},
                             cute::make_shape(cute::_16{}, cute::_2{})),
            cute::_1{}, cute::Int<MmaCore::kNumAtomKs>{}),
        cute::make_stride(
            cute::make_stride(
                cute::_16{}, cute::make_stride(cute::_1{}, cute::Int<kBLbo>{})),
            cute::_0{}, cute::Int<kBKStride>{})));

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

    __device__ static std::uint32_t k_tile(std::uint32_t k_iter_id) {
        return split_idx() * kTilesKPerCta + k_iter_id;
    }

    __device__ static std::uint32_t cluster_rank() {
        return blockIdx.x % Traits::kClusterSize;
    }

    __device__ static std::uint32_t group_rank(std::uint32_t r) {
        return cluster_rank() - split_idx() + r;
    }

    __device__ static std::uint32_t m_tile_idx() {
        return (cta_idx() / kNumSplits) % kNumTilesM;
    }

    __device__ static std::uint32_t token_begin() {
        return (cta_idx() / (kNumSplits * kNumTilesM)) * kTileShapeN;
    }

    __device__ static std::uint32_t row_base() {
        return m_tile_idx() * kTileShapeM;
    }

    __device__ static std::uint32_t tile_idx() { return row_base() / 128u; }

    __device__ static Element* staged_block(SharedStorage* smem,
                                            std::uint32_t k_iter_id) {
        return reinterpret_cast<Element*>(
            &smem->latent_down.quantized_input
                 .data[k_iter_id % kNumInputSlots][0][0][0]);
    }

    __device__ static void run_load_input_k_loop(
        CUtensorMap const& tma_input_desc, SharedStorage* __restrict__ smem) {
#pragma unroll 1
        for (std::uint32_t k_iter_id = 0; k_iter_id < kTilesKPerCta;
             ++k_iter_id) {
            const std::uint32_t slot = k_iter_id % kNumInputSlots;
            smem->barriers.latent_down.wait_input_empty(k_iter_id);

            smem->barriers.latent_down.input_ready.arrive(k_iter_id,
                                                          kInputTileTxBytes);
            load_input_tile_chunks<kTileShapeN>(
                tma_input_desc, k_tile(k_iter_id), token_begin(),
                smem->latent_down.input.data[slot],
                smem->barriers.latent_down.input_ready.raw(slot));
        }
    }

    __device__ static void load_weight_tile(
        CUtensorMap const& tma_latent_down_weight_desc,
        SharedStorage* __restrict__ smem, std::uint32_t k_iter_id,
        std::uint32_t transaction_bytes) {
        const std::uint32_t slot = k_iter_id % kNumWeightSlots;
        smem->barriers.latent_down.weight_ready.arrive(k_iter_id,
                                                       transaction_bytes);
        const std::uint32_t k_start =
            k_tile(k_iter_id) * LatentDownGemm::kTileShapeK;
#pragma unroll 1
        for (std::uint32_t h = 0; h < LatentDownGemm::kNumStagingBlocks; ++h) {
            tma_load_2d(tma_latent_down_weight_desc,
                        /*coord0=*/k_start + h * 128u,
                        /*coord1=*/row_base(),
                        &smem->latent_down.w_latent_down[slot][h][0][0],
                        smem->barriers.latent_down.weight_ready.raw(slot));
        }
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

    __device__ static void store_split_share(
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

    __device__ static void gather_split_share(SharedStorage* __restrict__ smem,
                                              float (&acc)[kTokenWindows][4]) {
        const std::uint32_t warp_idx = WarpRole::get_warp_idx();
        const std::uint32_t lane_idx = WarpRole::get_lane_idx();
        if (warp_idx >= kTileShapeM / 16u) return;
        const std::uint32_t row = 16u * group_idx(warp_idx) + lane_idx / 4u;
        const std::uint32_t t0 = lane_idx & 3u;
        float* const local = &smem->latent_down.splited_mma_out[0][0];
        // Map peer splits while skipping the local split.
        std::uint32_t shares[kNumSplits > 1u ? kNumSplits - 1u : 1u];
#pragma unroll
        for (std::uint32_t n = 0; n + 1u < kNumSplits; ++n) {
            const std::uint32_t r = n + (n >= split_idx() ? 1u : 0u);
            shares[n] = cluster_map_shared_rank(
                &smem->latent_down.splited_mma_out[0][0], group_rank(r));
        }
#pragma unroll
        for (std::uint32_t g = 0; g < kTokenWindows; ++g) {
#pragma unroll
            for (std::uint32_t i = 0; i < 4u; ++i) {
                const std::uint32_t at = (row + 8u * (i / 2u)) * kTileShapeN +
                                         (8u * g + 2u * t0 + i % 2u);
                float v = local[at];
                if constexpr (kNumSplits > 1u) {
#pragma unroll
                    for (std::uint32_t n = 0; n < kNumSplits - 1u; ++n) {
                        v += cluster_load_f32(shares[n] + 4u * at);
                    }
                }
                acc[g][i] = v;
            }
        }
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

    // The calc warps' finished sums, to the split reduction or straight out.
    __device__ static void store_acc(Workspace* __restrict__ workspace,
                                     SharedStorage* __restrict__ smem,
                                     float const (&acc)[kTokenWindows][4]) {
        if constexpr (kNumSplits > 1u) {
            store_split_share(smem, acc);
        } else {
            writeback(workspace, smem, acc);
            WarpRole::sync_calc_threads();
        }
    }

    // Split 0 reduces peer outputs; its final wait follows output
    // quantization.
    __device__ static void join_split_k(Workspace* __restrict__ workspace,
                                        SharedStorage* __restrict__ smem) {
        // Every thread joins: the cluster barrier is block-collective.
        __syncthreads();
        cluster_arrive_and_wait();
        if (split_idx() == 0u) {
            if (WarpRole::is_calc_warp(WarpRole::get_warp_idx())) {
                float acc[kTokenWindows][4] = {};
                gather_split_share(smem, acc);
                writeback(workspace, smem, acc);
                WarpRole::sync_calc_threads();
            }
            // Release peer reads now; wait after output quantization.
            cluster_arrive();
        } else {
            cluster_arrive();
            cluster_wait();
        }
    }

    // Split 0's pending arrive from `join_split_k`.
    __device__ static void release_split_partner() {
        if (split_idx() == 0u) cluster_wait();
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

    __device__ static void run(
        CUtensorMap const& tma_latent_down_weight_desc,
        CUtensorMap const& tma_input_desc,
        const typename LatentDownGemm::
            ElementScale* __restrict__ latent_down_weight_scale,
        Workspace* __restrict__ workspace, SharedStorage* __restrict__ smem,
        std::uint32_t batch_size) {
        if (!Base::is_latent_cta()) return;

        zero_sf_slots(smem);

        const std::uint32_t warp_idx = WarpRole::get_warp_idx();
        if (WarpRole::is_other_loader_thread()) {
            Base::run_load_input_k_loop(tma_input_desc, smem);
        } else if (WarpRole::is_weight_loader_thread()) {
            run_load_weight_k_loop(tma_latent_down_weight_desc,
                                   latent_down_weight_scale, smem);
        } else if (WarpRole::is_mma_issuer_warp(warp_idx)) {
            run_mma_k_loop(smem, smem->tmem_ptr);
        } else if (WarpRole::is_calc_warp(warp_idx)) {
#pragma unroll 1
            for (std::uint32_t k_iter_id = 0; k_iter_id < Base::kTilesKPerCta;
                 ++k_iter_id) {
                quantize_input_tile(smem, batch_size, k_iter_id);
            }
            float acc[Base::kTokenWindows][4] = {};
            load_accumulators(smem, acc);
            Base::store_acc(workspace, smem, acc);
        }
        if constexpr (LatentDownGemm::kNumSplits > 1u) {
            Base::join_split_k(workspace, smem);
        }
        quantize_output(workspace, smem, batch_size);
        if constexpr (LatentDownGemm::kNumSplits > 1u) {
            Base::release_split_partner();
        }
    }

    __device__ static void run_load_weight_k_loop(
        CUtensorMap const& tma_latent_down_weight_desc,
        const typename LatentDownGemm::ElementScale* __restrict__ scale,
        SharedStorage* __restrict__ smem) {
        // Weight scales are staged as 512-byte SfAtoms.
        constexpr std::uint32_t kSfaBytes = ScaleTraits::kSfaSmemBytes;
        constexpr std::uint32_t kSfAtomBytes = 512u;
        constexpr std::uint32_t kSfKBlocksPerRow = Base::kNumScaleBlocksK;

#pragma unroll 1
        for (std::uint32_t k_iter_id = 0; k_iter_id < Base::kTilesKPerCta;
             ++k_iter_id) {
            // Wait for MMA to release the previous slot contents.
            smem->barriers.latent_down.wait_weight_empty(k_iter_id);
            Base::load_weight_tile(tma_latent_down_weight_desc, smem, k_iter_id,
                                   MmaCore::kWeightTileBytes + kSfaBytes);

            const std::uint32_t slot = k_iter_id % Base::kNumWeightSlots;
            const std::uint8_t* const src =
                reinterpret_cast<const std::uint8_t*>(scale) +
                (Base::tile_idx() * kSfKBlocksPerRow +
                 Base::k_tile(k_iter_id) * LatentDownGemm::kNumStagingBlocks) *
                    kSfAtomBytes;
            tma_load_1d(src, kSfaBytes,
                        smem->latent_down.w_latent_down_scale[slot],
                        smem->barriers.latent_down.weight_ready.raw(slot));
        }
    }

    __device__ static void run_mma_k_loop(SharedStorage* __restrict__ smem,
                                          std::uint32_t tmem_base) {
        typename MmaCore::TiledMma tiled_mma{};
        auto cta_mma = tiled_mma.get_slice(cute::_0{});
        cute::Tensor tCt = Base::make_accumulator_fragment(cta_mma);
#pragma unroll 1
        for (std::uint32_t k_iter_id = 0; k_iter_id < Base::kTilesKPerCta;
             ++k_iter_id) {
            const std::uint32_t w_slot = k_iter_id % Base::kNumWeightSlots;
            smem->barriers.latent_down.template wait_mma_ready<true, false>(
                k_iter_id);
            auto tCsB = cute::make_tensor(
                cute::make_smem_ptr(Base::staged_block(smem, k_iter_id)),
                SBLayout{});
            cute::Tensor tCrB = cta_mma.make_fragment_B(tCsB);
            // SFA and SFB are staged in SfAtom layout.
            std::uint8_t* const sfa =
                smem->latent_down.w_latent_down_scale[w_slot];
            std::uint8_t* const sfb =
                &smem->latent_down.quantized_input
                     .scale[k_iter_id % Base::kNumInputSlots][0];
            if (cute::elect_one_sync()) {
                ScaleTraits::copy_to_tmem(sfa, sfb, tmem_base + kSfaTmemOffset,
                                          tmem_base + kSfbTmemOffset);
            }
            MmaCore::run(tiled_mma, cta_mma,
                         &smem->latent_down.w_latent_down[w_slot][0][0][0],
                         tCrB, tCt, tmem_addr(tmem_base, 0u, 0u),
                         tmem_base + kSfaTmemOffset, tmem_base + kSfbTmemOffset,
                         /*accumulate=*/k_iter_id != 0u);

            cutlass::arch::umma_arrive(
                smem->barriers.latent_down.weight_empty.raw(k_iter_id));
        }
        // Publish the completed split once, after all K tiles.
        cutlass::arch::umma_arrive(
            smem->barriers.latent_down.accumulator_ready.raw(0u));
    }

    __device__ static void quantize_input_tile(SharedStorage* __restrict__ smem,
                                               std::uint32_t batch_size,
                                               std::uint32_t k_iter_id) {
        const std::uint32_t slot = k_iter_id % Base::kNumInputSlots;
        const std::uint32_t warp_idx = WarpRole::get_calc_warp_idx();
        // Input and weight slots share the previous MMA's completion.
        smem->barriers.latent_down.wait_weight_empty(k_iter_id);
        smem->barriers.latent_down.input_ready.wait(k_iter_id);
        const QuantizeTile tile(
            slot, Base::k_tile(k_iter_id) * LatentDownGemm::kNumStagingBlocks,
            Base::token_begin(), batch_size);
        quantize_mxfp8_groupwise_1x32<
            /*Staged=*/true, Base::kTileShapeN, LatentDownGemm::kTileShapeK,
            128u, SwizzleBoxK<DataType::ElementA>::value,
            LatentDownGemm::kNumStagingBlocks>(
            reinterpret_cast<const DataType::ElementA*>(
                &smem->latent_down.input.data[slot][0][0][0]),
            Base::k_tile(k_iter_id) * LatentDownGemm::kTileShapeK,
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

    // Write one UE8M0 output scale per 32-row group.
    __device__ static void quantize_output(Workspace* __restrict__ workspace,
                                           SharedStorage* __restrict__ smem,
                                           std::uint32_t batch_size) {
        constexpr std::uint32_t kCells = Base::kOutChunks * Base::kTileShapeN;
        constexpr std::uint32_t kGroupRows = 32u;
        constexpr std::uint32_t kChunksPerGroup = kGroupRows / Base::kOutChunkK;
        constexpr std::uint32_t kNumGroups = Base::kTileShapeM / kGroupRows;
        static_assert(Base::kTileShapeM % kGroupRows == 0u,
                      "an MXFP8 scale group must not straddle CTAs");
        static_assert(Traits::UpGemm::kTileShapeK % Base::kTileShapeM == 0u,
                      "an MXFP8 CTA's rows must sit inside one up K tile");
        constexpr bool kLocalScale = LatentDownGemm::kClusterSize == 1u;
        const bool owns_slice = Base::split_idx() == 0u;
        if (kLocalScale && !owns_slice) {
            __syncthreads();
            if (threadIdx.x == 0u) {
                workspace->cta_comm.publish_latent_tasks(smem->launch_parity,
                                                         1u);
            }
            return;
        }
        const std::uint32_t warp_idx = WarpRole::get_warp_idx();
        const std::uint32_t lane_idx = WarpRole::get_lane_idx();
        const std::uint32_t chunk_base = Base::row_base() / Base::kOutChunkK;

        if (warp_idx < Base::kTileShapeN) {
            // Each warp reduces one token over 32-feature groups.
#pragma unroll
            for (std::uint32_t g = 0; g < kNumGroups; ++g) {
                const std::uint32_t f = kGroupRows * g + lane_idx;
                const float v = __bfloat162float(
                    smem->latent_down.act_out[f / Base::kOutChunkK][warp_idx]
                                             [f % Base::kOutChunkK]);
                const float mag = fabsf(v);
                const float amax = __uint_as_float(__reduce_max_sync(
                    0xffffffffu, __float_as_uint((mag == mag) ? mag : 0.0f)));
                if (lane_idx == 0u) {
                    smem->latent_down.token_group_amax[warp_idx][g] = amax;
                }
            }
        }

        if constexpr (kLocalScale) {
            __syncthreads();
        } else {
            cluster_arrive_and_wait();
        }

        const bool is_cell = owns_slice && threadIdx.x < kCells;
        const std::uint32_t cell_chunk = threadIdx.x / Base::kTileShapeN;
        const std::uint32_t cell_tok = threadIdx.x % Base::kTileShapeN;
        const std::uint32_t tok = Base::token_begin() + cell_tok;
        uint4 quantized = {};
        if (is_cell && tok < batch_size) {
            std::uint32_t biased;
            const float inv_scale = mxfp8_ue8m0_scale(
                smem->latent_down
                    .token_group_amax[cell_tok][cell_chunk / kChunksPerGroup],
                biased);
            quantized =
                Base::quantize_cell(smem, cell_chunk, cell_tok, inv_scale);
        }

        if constexpr (!kLocalScale) {
            cluster_arrive_and_wait();
        }

        if (is_cell) {
            workspace->latent_down_out.store_cell(chunk_base + cell_chunk, tok,
                                                  quantized);
        }
        // One thread per token packs four scale exponents into an SfAtom word.
        const std::uint32_t rank =
            Base::cta_idx() % LatentDownGemm::kClusterSize;
        const bool owns_scale_row =
            owns_slice && rank == 0u && threadIdx.x < Base::kTileShapeN;
        if (owns_scale_row) {
            const std::uint32_t tok_global = Base::token_begin() + threadIdx.x;
            std::uint32_t packed = 0u;
#pragma unroll
            for (std::uint32_t g = 0; g < kNumGroups; ++g) {
                // Use the neutral exponent for padding tokens.
                std::uint32_t biased = 127u;
                if (tok_global < batch_size) {
                    mxfp8_ue8m0_scale(
                        smem->latent_down.token_group_amax[threadIdx.x][g],
                        biased);
                }
                packed |= biased << (8u * g);
            }
            constexpr std::uint32_t kUpTileShapeK = Traits::UpGemm::kTileShapeK;
            std::uint8_t* const sf = workspace->latent_down_out.sf_slab(
                Base::row_base() / kUpTileShapeK);
            *reinterpret_cast<std::uint32_t*>(
                sf +
                sf_atom_offset(tok_global, (Base::row_base() % kUpTileShapeK) /
                                               kGroupRows)) = packed;
        }
        if (owns_slice) {
            for (std::uint32_t i = threadIdx.x; i < Base::kOutChunks;
                 i += Traits::kThreads) {
                workspace->latent_down_out.clear_pad_row(chunk_base + i);
            }
        }
        __syncthreads();
        if (threadIdx.x == 0u) {
            workspace->cta_comm.publish_latent_tasks(smem->launch_parity, 1u);
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

    static constexpr std::uint32_t kNumAccSlots = LatentDownGemm::kNumAccSlots;
    static constexpr std::uint32_t kAccColumns =
        get_gemm_acc_columns<Traits::UpGemm>();
    static constexpr bool kLocalScale =
        LatentDownGemm::OutputQuant::kBlockSize <= Base::kTileShapeM;

    __device__ static void run(
        CUtensorMap const& tma_latent_down_weight_desc,
        CUtensorMap const& tma_input_desc,
        const typename LatentDownGemm::
            ElementScale* __restrict__ latent_down_weight_scale,
        Workspace* __restrict__ workspace, SharedStorage* __restrict__ smem,
        std::uint32_t batch_size) {
        if (!Base::is_latent_cta()) return;

        const std::uint32_t warp_idx = WarpRole::get_warp_idx();
        if (WarpRole::is_other_loader_thread()) {
            Base::run_load_input_k_loop(tma_input_desc, smem);
        } else if (WarpRole::is_weight_loader_thread()) {
            run_load_weight_k_loop(tma_latent_down_weight_desc, smem);
        } else if (WarpRole::is_mma_issuer_warp(warp_idx)) {
            run_mma_k_loop(smem, smem->tmem_ptr);
        } else if (WarpRole::is_calc_warp(warp_idx)) {
            run_quantize_and_dequant_k_loop(smem, latent_down_weight_scale,
                                            workspace, batch_size);
        }
        if constexpr (LatentDownGemm::kNumSplits > 1u) {
            Base::join_split_k(workspace, smem);
        }
        quantize_output(workspace, smem, batch_size);
        if constexpr (LatentDownGemm::kNumSplits > 1u) {
            Base::release_split_partner();
        }
    }

    __device__ static void run_load_weight_k_loop(
        CUtensorMap const& tma_latent_down_weight_desc,
        SharedStorage* __restrict__ smem) {
#pragma unroll 1
        for (std::uint32_t k_iter_id = 0; k_iter_id < Base::kTilesKPerCta;
             ++k_iter_id) {
            smem->barriers.latent_down.wait_weight_empty(k_iter_id);
            Base::load_weight_tile(tma_latent_down_weight_desc, smem, k_iter_id,
                                   MmaCore::kWeightTileBytes);
        }
    }

    __device__ static void run_mma_k_loop(SharedStorage* __restrict__ smem,
                                          std::uint32_t tmem_base) {
        typename MmaCore::TiledMma tiled_mma{};
        auto cta_mma = tiled_mma.get_slice(cute::_0{});
        cute::Tensor tCt = Base::make_accumulator_fragment(cta_mma);
#pragma unroll 1
        for (std::uint32_t k_iter_id = 0; k_iter_id < Base::kTilesKPerCta;
             ++k_iter_id) {
            const std::uint32_t acc_slot = k_iter_id % kNumAccSlots;
            const std::uint32_t w_slot = k_iter_id % Base::kNumWeightSlots;
            smem->barriers.latent_down.template wait_mma_ready<true, true>(
                k_iter_id);
            auto tCsB = cute::make_tensor(
                cute::make_smem_ptr(Base::staged_block(smem, k_iter_id)),
                SBLayout{});
            cute::Tensor tCrB = cta_mma.make_fragment_B(tCsB);
            MmaCore::run<LatentDownGemm::kNumStagingBlocks>(
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
                                               std::uint32_t k_iter_id) {
        const std::uint32_t slot = k_iter_id % Base::kNumInputSlots;
        const std::uint32_t warp_idx = WarpRole::get_calc_warp_idx();
        smem->barriers.latent_down.input_ready.wait(k_iter_id);

        const QuantizeTile tile(
            slot, Base::k_tile(k_iter_id) * LatentDownGemm::kNumStagingBlocks,
            Base::token_begin(), batch_size);
        quantize_fp8_groupwise_128<
            /*Staged=*/true, Base::kTileShapeN, LatentDownGemm::kTileShapeK,
            Base::kGemmN, LatentDownGemm::InputQuant::kBlockSize,
            SwizzleBoxK<DataType::ElementA>::value,
            LatentDownGemm::kNumStagingBlocks>(
            reinterpret_cast<const DataType::ElementA*>(
                &smem->latent_down.input.data[slot][0][0][0]),
            Base::k_tile(k_iter_id) * LatentDownGemm::kTileShapeK,
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

    __device__ static void run_quantize_and_dequant_k_loop(
        SharedStorage* __restrict__ smem,
        const typename LatentDownGemm::ElementScale* __restrict__ w_scale,
        Workspace* __restrict__ workspace, std::uint32_t batch_size) {
        constexpr std::uint32_t kStageAhead = 1u;
        quantize_input_tile(smem, batch_size, 0u);

        float acc[Base::kTokenWindows][4] = {};
#pragma unroll 1
        for (std::uint32_t k_iter_id = 0; k_iter_id < Base::kTilesKPerCta;
             ++k_iter_id) {
            if (k_iter_id + kStageAhead < Base::kTilesKPerCta) {
                quantize_input_tile(smem, batch_size, k_iter_id + kStageAhead);
            }
            dequant(smem, w_scale, acc, k_iter_id);
        }
        Base::store_acc(workspace, smem, acc);
    }

    // A block's amax over one token's window, gathered from the cluster.
    __device__ static std::uint32_t group_base_rank() {
        return blockIdx.x % Traits::kClusterSize -
               Base::cta_idx() % LatentDownGemm::kClusterSize;
    }

    // Read peer maxima between cluster barriers when a scale spans CTAs.
    __device__ static float cluster_token_max(SharedStorage* __restrict__ smem,
                                              std::uint32_t tok) {
        float amax = smem->latent_down.token_amax[tok];
        if constexpr (!kLocalScale) {
#pragma unroll
            for (std::uint32_t rank = 0; rank < LatentDownGemm::kClusterSize;
                 ++rank) {
                amax = fmaxf(amax, cluster_load_f32(cluster_map_shared_rank(
                                       &smem->latent_down.token_amax[tok],
                                       group_base_rank() + rank)));
            }
        }
        return amax;
    }

    __device__ static float token_block_max(SharedStorage* __restrict__ smem,
                                            std::uint32_t tok) {
        // A max near FLT_MIN would invert to +inf and NaN the block.
        return fmaxf(cluster_token_max(smem, tok), 1e-10f);
    }

    __device__ static void dequant(
        SharedStorage* __restrict__ smem,
        const typename LatentDownGemm::ElementScale* __restrict__ w_scale,
        float (&acc)[Base::kTokenWindows][4], std::uint32_t k_iter_id) {
        const std::uint32_t warp_idx = WarpRole::get_warp_idx();
        const std::uint32_t lane_idx = WarpRole::get_lane_idx();
        // Each warp reads its 16-row TMEM fragment.
        constexpr std::uint32_t kReaderWarps = Base::kTileShapeM / 16u;
        const std::uint32_t t0 = lane_idx & 3u;
        const std::uint32_t token_base = Base::token_begin();

        // Every calc warp waits for its MMA before signaling completion.
        smem->barriers.latent_down.accumulator_ready.wait(k_iter_id);
#pragma unroll
        for (std::uint32_t sub = 0; sub < LatentDownGemm::kNumStagingBlocks;
             ++sub) {
            // Each split owns kNumScaleBlocksK / kNumSplits scale columns.
            constexpr std::uint32_t kBlocksPerSplit =
                Base::kNumScaleBlocksK / Base::kNumSplits;
            const std::uint32_t ws_block =
                k_iter_id * LatentDownGemm::kNumStagingBlocks + sub;
            const float ws =
                w_scale[Base::tile_idx() * Base::kNumScaleBlocksK +
                        Base::split_idx() * kBlocksPerSplit + ws_block];
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
                const std::uint32_t block =
                    Base::k_tile(k_iter_id) *
                        LatentDownGemm::kNumStagingBlocks +
                    sub;
                as[0] = smem->latent_down.quantized_input
                            .scale[block][token_base + 8u * g + 2u * t0];
                as[1] = smem->latent_down.quantized_input
                            .scale[block][token_base + 8u * g + 2u * t0 + 1u];
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

    // Write one FP32 scale per 128-row token block.
    __device__ static void quantize_output(Workspace* __restrict__ workspace,
                                           SharedStorage* __restrict__ smem,
                                           std::uint32_t batch_size) {
        constexpr std::uint32_t kCells = Base::kOutChunks * Base::kTileShapeN;
        static_assert(kCells <= Traits::kThreads,
                      "one output cell per thread keeps the quantized bytes in "
                      "registers across the cluster barrier");
        const bool owns_slice = Base::split_idx() == 0u;
        if (kLocalScale && !owns_slice) {
            __syncthreads();
            if (threadIdx.x == 0u) {
                workspace->cta_comm.publish_latent_tasks(smem->launch_parity,
                                                         1u);
            }
            return;
        }
        const std::uint32_t warp_idx = WarpRole::get_warp_idx();
        const std::uint32_t lane_idx = WarpRole::get_lane_idx();
        const std::uint32_t chunk_base = Base::row_base() / Base::kOutChunkK;

        if (warp_idx < Base::kTileShapeN) {
            float amax = 0.0f;
            for (std::uint32_t i = lane_idx;
                 i < Base::kOutChunks * Base::kOutChunkK;
                 i += WarpRole::kThreadsPerWarp) {
                amax =
                    fmaxf(amax, fabsf(__bfloat162float(
                                    smem->latent_down
                                        .act_out[i / Base::kOutChunkK][warp_idx]
                                                [i % Base::kOutChunkK])));
            }
            // Nonnegative floats have the same ordering as their bit patterns.
            const float block_amax = __uint_as_float(
                __reduce_max_sync(0xffffffffu, __float_as_uint(amax)));
            if (lane_idx == 0u) {
                smem->latent_down.token_amax[warp_idx] = block_amax;
            }
        }

        if constexpr (kLocalScale) {
            __syncthreads();
        } else {
            cluster_arrive_and_wait();
        }

        const bool is_cell = owns_slice && threadIdx.x < kCells;
        const std::uint32_t cell_chunk = threadIdx.x / Base::kTileShapeN;
        const std::uint32_t cell_tok = threadIdx.x % Base::kTileShapeN;
        const std::uint32_t tok = Base::token_begin() + cell_tok;
        uint4 quantized = {};
        if (is_cell && tok < batch_size) {
            quantized = Base::quantize_cell(
                smem, cell_chunk, cell_tok,
                Base::kFp8Max / token_block_max(smem, cell_tok));
        }

        // Read the scale before releasing peer shared memory.
        float row_scale = 0.0f;
        const std::uint32_t rank =
            Base::cta_idx() % LatentDownGemm::kClusterSize;
        const bool owns_scale_row =
            owns_slice && rank == 0u && threadIdx.x < Base::kTileShapeN;
        if (owns_scale_row) {
            row_scale = token_block_max(smem, threadIdx.x) * Base::kFp8MaxInv;
            // Use the neutral scale for padding tokens.
            if (Base::token_begin() + threadIdx.x >= batch_size) {
                row_scale = 1.0f;
            }
        }

        if constexpr (!kLocalScale) {
            cluster_arrive_and_wait();
        }

        if (is_cell) {
            workspace->latent_down_out.store_cell(chunk_base + cell_chunk, tok,
                                                  quantized);
        }
        if (owns_scale_row) {
            workspace->latent_down_out.scale_row(
                Base::row_base() /
                LatentDownGemm::OutputQuant::kBlockSize)[Base::token_begin() +
                                                         threadIdx.x] =
                row_scale;
        }
        if (owns_slice) {
            for (std::uint32_t i = threadIdx.x; i < Base::kOutChunks;
                 i += Traits::kThreads) {
                workspace->latent_down_out.clear_pad_row(chunk_base + i);
            }
        }
        __syncthreads();
        if (threadIdx.x == 0u) {
            workspace->cta_comm.publish_latent_tasks(smem->launch_parity, 1u);
        }
    }
};

}  // namespace halops::moe
