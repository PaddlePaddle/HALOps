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

// Shared latent down-projection computation; callers own the scheduling and
// the phase hand-offs.  The recipe's own version selects the weight-scale
// staging, the input quantizer and the output's scale image.
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
    // What one input tile's TMA copy credits the mbarrier with: the staged
    // bf16 activation, not the quantized B operand the MMA reads.
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
    static constexpr std::uint32_t kOutChunks = kTileShapeM / kFp8ActChunkK;
    static constexpr std::uint32_t kGemmN = LatentDownGemm::kGemmN;

    // 8-token columns per `16x256b` read.
    static constexpr std::uint32_t kTokenWindows = kTileShapeN / 8u;
    // B operand: the padded staging layout, so the chunk stride is the
    // staging's row count (window tokens plus the pad row).
    static constexpr std::uint64_t kBLbo =
        static_cast<std::uint64_t>(kTileShapeN + 1u) *
        static_cast<std::uint64_t>(Traits::kLboChunkK);
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

    // A warp reaches only its own 32-lane sub-partition: a 128-row atom
    // interleaves the groups one per 16-lane window.
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
            const auto empty_span = KERNEL_TIMER_BEGIN(
                smem->timer(), "latent_wait_input_empty", k_iter_id);
            smem->barriers.latent_down.wait_input_empty(k_iter_id);
            KERNEL_TIMER_END(smem->timer(), empty_span);

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

    // One output cell: this thread's 16-B chunk of `cell_tok`'s bf16 row, in
    // fp8 under `inv_scale`.  An unused token is zeroed below; NaN drops to 0
    // as in the up phase's quantize.
    __device__ static uint4 quantize_cell(SharedStorage* __restrict__ smem,
                                          std::uint32_t cell_chunk,
                                          std::uint32_t cell_tok,
                                          float inv_scale) {
        const DataType::ElementA* const row =
            &smem->latent_down.act_out[cell_chunk][cell_tok][0];
        uint4 quantized = {};
#pragma unroll
        for (std::uint32_t i = 0; i < kFp8ActChunkK; ++i) {
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
        // The n-th peer split skips this block's own: indexing `shares` by
        // the compile-time `n` keeps it in registers, where a runtime cursor
        // (one that skips the own split) would put it in local memory.
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
        const auto store_span =
            KERNEL_TIMER_BEGIN(smem->timer(), "latent_reduce_and_store");
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
                    .act_out[row_local / kFp8ActChunkK][tok_local + i % 2u]
                            [row_local % kFp8ActChunkK] =
                    __float2bfloat16_rn(acc[g][i]);
            }
        }
        KERNEL_TIMER_END(smem->timer(), store_span);
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

    // Split 0 sums the shares the splits have stored and writes the output;
    // each thread's arrive releases its reads of them, and the matching wait
    // comes after the output quantization.
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
            // Each thread's arrive releases its reads of the shares; the
            // wait is deferred until the output is out.
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

// MXFP8: the scales are staged in their SfAtom layout and applied by the
// blockscaled UMMA, and the output carries one UE8M0 byte per 32-row group.
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

    // The quantizer fills N rows of each SFB slot; the padded rows stay zero
    // for UTCCP.
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
            const auto mma_span =
                KERNEL_TIMER_BEGIN(smem->timer(), "latent_down_umma");
            run_mma_k_loop(smem, smem->tmem_ptr);
            KERNEL_TIMER_END(smem->timer(), mma_span);
        } else if (WarpRole::is_calc_warp(warp_idx)) {
            const auto quant_span =
                KERNEL_TIMER_BEGIN(smem->timer(), "latent_quantize_input");
#pragma unroll 1
            for (std::uint32_t k_iter_id = 0; k_iter_id < Base::kTilesKPerCta;
                 ++k_iter_id) {
                quantize_input_tile(smem, batch_size, k_iter_id);
            }
            KERNEL_TIMER_END(smem->timer(), quant_span);
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
        // The weight scales ride along in their SfAtom staging: the tile's
        // K/32 groups are whole 512-B atoms.
        constexpr std::uint32_t kSfaBytes = ScaleTraits::kSfaSmemBytes;
        constexpr std::uint32_t kSfAtomBytes = 512u;
        constexpr std::uint32_t kSfKBlocksPerRow = Base::kNumScaleBlocksK;

#pragma unroll 1
        for (std::uint32_t k_iter_id = 0; k_iter_id < Base::kTilesKPerCta;
             ++k_iter_id) {
            // The ring slot a weight tile reuses must have been drained by
            // the MMA.
            const auto empty_span = KERNEL_TIMER_BEGIN(
                smem->timer(), "latent_wait_weight_empty", k_iter_id);
            smem->barriers.latent_down.wait_weight_empty(k_iter_id);
            KERNEL_TIMER_END(smem->timer(), empty_span);
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

            const auto mma_ready_span = KERNEL_TIMER_BEGIN(
                smem->timer(), "latent_wait_mma_ready", k_iter_id);
            smem->barriers.latent_down.template wait_mma_ready<true, false>(
                k_iter_id);
            KERNEL_TIMER_END(smem->timer(), mma_ready_span);
            auto tCsB = cute::make_tensor(
                cute::make_smem_ptr(Base::staged_block(smem, k_iter_id)),
                SBLayout{});
            cute::Tensor tCrB = cta_mma.make_fragment_B(tCsB);
            // SFA was copied in by the weight loader and SFB was written by
            // the activation quantizer, both in SfAtom layout.
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
        const auto ready_span = KERNEL_TIMER_BEGIN(
            smem->timer(), "latent_wait_quantize_ready", k_iter_id);
        // Input and weight slots share the previous MMA's completion.
        smem->barriers.latent_down.wait_weight_empty(k_iter_id);
        smem->barriers.latent_down.input_ready.wait(k_iter_id);
        KERNEL_TIMER_END(smem->timer(), ready_span);

        const auto quantize_span =
            KERNEL_TIMER_BEGIN(smem->timer(), "quantize_fp8_blockwise_128");
        // Where this tile's rows and blocks sit in the source and staging.
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
        KERNEL_TIMER_END(smem->timer(), quantize_span);
        // The fp8 stores must reach the async proxy before the tile is
        // signalled: the issuer reads them through tcgen05.
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
        // A warp reads its 16 rows via `tcgen05.ld.16x256b` out of the lanes
        // its own sub-partition owns.
        constexpr std::uint32_t kReaderWarps = Base::kTileShapeM / 16u;

        const auto acc_ready_span = KERNEL_TIMER_BEGIN(
            smem->timer(), "latent_wait_accumulator_ready", 0u);
        smem->barriers.latent_down.accumulator_ready.wait(0u);
        tcgen05_fence_after_thread_sync();
        KERNEL_TIMER_END(smem->timer(), acc_ready_span);
        if (warp_idx >= kReaderWarps) return;
#pragma unroll
        for (std::uint32_t g = 0; g < Base::kTokenWindows; ++g) {
            tmem_ld_16x256b1x(
                tmem_addr(smem->tmem_ptr, Base::lane_base(warp_idx), g * 8u),
                acc[g]);
            cutlass::arch::fence_view_async_tmem_load();
        }
    }

    // The output quantized by the MXFP8 recipe: one UE8M0 byte per 32-row
    // group.
    __device__ static void quantize_output(Workspace* __restrict__ workspace,
                                           SharedStorage* __restrict__ smem,
                                           std::uint32_t batch_size) {
        constexpr std::uint32_t kCells = Base::kOutChunks * Base::kTileShapeN;
        constexpr std::uint32_t kGroupRows = 32u;
        constexpr std::uint32_t kChunksPerGroup = kGroupRows / kFp8ActChunkK;
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
        const std::uint32_t chunk_base = Base::row_base() / kFp8ActChunkK;

        if (warp_idx < Base::kTileShapeN) {
            // Lane `l` owns feature 32 g + l, so each pass reduces one
            // 32-feature group of this warp's token.
#pragma unroll
            for (std::uint32_t g = 0; g < kNumGroups; ++g) {
                const std::uint32_t f = kGroupRows * g + lane_idx;
                const float v = __bfloat162float(
                    smem->latent_down.act_out[f / kFp8ActChunkK][warp_idx]
                                             [f % kFp8ActChunkK]);
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
        // One thread per token writes its four group exponents as one word of
        // the tile's SfAtom slab.
        const std::uint32_t rank =
            Base::cta_idx() % LatentDownGemm::kClusterSize;
        const bool owns_scale_row =
            owns_slice && rank == 0u && threadIdx.x < Base::kTileShapeN;
        if (owns_scale_row) {
            const std::uint32_t tok_global = Base::token_begin() + threadIdx.x;
            std::uint32_t packed = 0u;
#pragma unroll
            for (std::uint32_t g = 0; g < kNumGroups; ++g) {
                // An unused token gets the neutral exponent, as the up phase's
                // quantize leaves its row.
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

// Block-wise FP8: the weight-scale rows travel with the epilogue, and the
// output carries one fp32 scale per 128-row token block.
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
            const auto empty_span = KERNEL_TIMER_BEGIN(
                smem->timer(), "latent_wait_weight_empty", k_iter_id);
            smem->barriers.latent_down.wait_weight_empty(k_iter_id);
            KERNEL_TIMER_END(smem->timer(), empty_span);
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

            const auto mma_ready_span = KERNEL_TIMER_BEGIN(
                smem->timer(), "latent_wait_mma_ready", k_iter_id);
            smem->barriers.latent_down.template wait_mma_ready<true, true>(
                k_iter_id);
            KERNEL_TIMER_END(smem->timer(), mma_ready_span);
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
        const auto ready_span = KERNEL_TIMER_BEGIN(
            smem->timer(), "latent_wait_input_ready", k_iter_id);
        smem->barriers.latent_down.input_ready.wait(k_iter_id);
        KERNEL_TIMER_END(smem->timer(), ready_span);

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

        // The fp8 stores must reach the async proxy before the tile is
        // signalled: the issuer reads them through tcgen05.
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

    // Reads partner shared memory: it runs between the two cluster barriers.
    // A block that holds whole quantization blocks has no partner to read.
    __device__ static float cluster_token_max(SharedStorage* __restrict__ smem,
                                              std::uint32_t tok) {
        float amax = smem->latent_down.token_amax[tok];
        if constexpr (LatentDownGemm::kClusterSize > 1u) {
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
        // A warp reads its 16 rows via `tcgen05.ld.16x256b` out of the lanes
        // its own sub-partition owns.
        constexpr std::uint32_t kReaderWarps = Base::kTileShapeM / 16u;
        const std::uint32_t t0 = lane_idx & 3u;
        const std::uint32_t token_base = Base::token_begin();

        // Every calc warp waits for the slice's UMMA before arriving: a warp
        // that raced ahead would complete the phase its own k never filled.
        const auto acc_ready_span = KERNEL_TIMER_BEGIN(
            smem->timer(), "latent_wait_accumulator_ready", k_iter_id);
        smem->barriers.latent_down.accumulator_ready.wait(k_iter_id);
        KERNEL_TIMER_END(smem->timer(), acc_ready_span);
#pragma unroll
        for (std::uint32_t sub = 0; sub < LatentDownGemm::kNumStagingBlocks;
             ++sub) {
            // This tile's slice of the weight-scale row: the split owns
            // kNumScaleBlocksK / kNumSplits columns.
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

    // The output quantized by the block-wise recipe: one fp32 scale per
    // 128-row token block.
    __device__ static void quantize_output(Workspace* __restrict__ workspace,
                                           SharedStorage* __restrict__ smem,
                                           std::uint32_t batch_size) {
        constexpr std::uint32_t kCells = Base::kOutChunks * Base::kTileShapeN;
        static_assert(kCells <= Traits::kThreads,
                      "one output cell per thread keeps the quantized bytes in "
                      "registers across the cluster barrier");
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
        const std::uint32_t chunk_base = Base::row_base() / kFp8ActChunkK;

        if (warp_idx < Base::kTileShapeN) {
            float amax = 0.0f;
            for (std::uint32_t i = lane_idx;
                 i < Base::kOutChunks * kFp8ActChunkK;
                 i += WarpRole::kThreadsPerWarp) {
                amax = fmaxf(
                    amax,
                    fabsf(__bfloat162float(
                        smem->latent_down.act_out[i / kFp8ActChunkK][warp_idx]
                                                 [i % kFp8ActChunkK])));
            }
            // Bit-pattern max: non-negative floats order like their IEEE-754
            // bits.
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

        // This block's scale row, read before the barrier releases the
        // partners' shared memory.
        float row_scale = 0.0f;
        const std::uint32_t rank =
            Base::cta_idx() % LatentDownGemm::kClusterSize;
        const bool owns_scale_row =
            owns_slice && rank == 0u && threadIdx.x < Base::kTileShapeN;
        if (owns_scale_row) {
            row_scale = token_block_max(smem, threadIdx.x) * Base::kFp8MaxInv;
            // An unused token gets the neutral scale used for padding
            // columns.
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
