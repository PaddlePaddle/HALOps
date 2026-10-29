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

#include "mma_core.cuh"
#include "moe_configs.cuh"
#include "moe_input.cuh"
#include "moe_shared_storage.cuh"
#include "moe_tma.h"
#include "moe_traits.cuh"
#include "moe_workspace.cuh"
#include "ptx_utils.cuh"

namespace halops::moe {

template <typename KernelTraits>
struct LatentDownProjectionBase {
    using Traits = KernelTraits;
    using Dims = typename Traits::MoEDims;
    using RouterGemm = typename Traits::RouterGemm;
    using LatentDownGemm = typename Traits::LatentDownGemm;
    using SharedStorage = MoESharedStorage<Dims>;
    using Workspace = MoEWorkspace<Dims>;
    using Element = typename LatentDownGemm::MmaElement;

    static constexpr std::uint32_t kTileShapeM = LatentDownGemm::kTileShapeM;
    static constexpr std::uint32_t kTileShapeN = LatentDownGemm::kTileShapeN;
    static_assert(kTileShapeM <= Dims::kQuantBlockM,
                  "a CTA's rows must sit inside one 128-row weight block");

    static constexpr std::uint32_t kNumCtas = LatentDownGemm::kNumCtas;
    static constexpr std::uint32_t kNumTilesM = LatentDownGemm::kNumTilesM;
    static constexpr std::uint32_t kNumTilesK = LatentDownGemm::kNumTilesK;

    static constexpr std::uint32_t kNumWeightSlots =
        LatentDownGemm::kNumWeightSlots;
    static constexpr std::uint32_t kNumScaleBlocksK =
        LatentDownGemm::kNumScaleBlocksK;

    static constexpr std::uint32_t kQuantBlocksPerTile =
        LatentDownGemm::kQuantBlocksPerTile;
    __device__ static bool is_latent_cta() {
        return blockIdx.x >= RouterGemm::kNumCtas &&
               blockIdx.x < RouterGemm::kNumCtas + kNumCtas;
    }

    __device__ static std::uint32_t cta_idx() {
        return blockIdx.x - RouterGemm::kNumCtas;
    }

    __device__ static Element* staged_block(SharedStorage* smem,
                                            std::uint32_t tile) {
        return reinterpret_cast<Element*>(
            &smem->latent_down.quantized_input
                 .data[tile % Traits::kLatentInputSlots][0][0][0]);
    }

    __device__ static std::uint32_t tile_idx() {
        return row_base() / Dims::kQuantBlockM;
    }

    static constexpr std::uint32_t kNumSplits = LatentDownGemm::kNumSplits;
    static constexpr std::uint32_t kTilesKPerCta =
        LatentDownGemm::kTilesKPerCta;

    __device__ static std::uint32_t split_idx() {
        return cta_idx() % kNumSplits;
    }

    __device__ static std::uint32_t k_tile(std::uint32_t local) {
        return split_idx() * kTilesKPerCta + local;
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

    __device__ static void run_load_input_k_loop(
        CUtensorMap const& tma_input_desc, SharedStorage* __restrict__ smem) {
        constexpr std::uint32_t kInputSlots = Traits::kLatentInputSlots;
        constexpr std::uint32_t kInputTileBytes =
            LatentDownGemm::kTileShapeK * kTileShapeN *
            static_cast<std::uint32_t>(sizeof(DataType::ElementA));

#pragma unroll 1
        for (std::uint32_t local = 0; local < kTilesKPerCta; ++local) {
            const std::uint32_t slot = local % kInputSlots;
            if (local >= kInputSlots) {
                smem->barriers.latent_down_input_empty.wait(local -
                                                            kInputSlots);
            }

            smem->barriers.latent_down_input_ready.arrive(local,
                                                          kInputTileBytes);
            load_input_tile<kTileShapeN>(
                tma_input_desc, k_tile(local), token_begin(),
                smem->latent_down.input.data[slot],
                smem->barriers.latent_down_input_ready.raw(slot));
        }
    }

    __device__ static void run_load_weight_k_loop(
        CUtensorMap const& tma_latent_down_weight_desc,
        SharedStorage* __restrict__ smem) {
        constexpr std::uint32_t kWeightTileBytes =
            LatentDownGemm::MmaCore::kWeightTileBytes;

#pragma unroll 1
        for (std::uint32_t local = 0; local < kTilesKPerCta; ++local) {
            const std::uint32_t slot = local % kNumWeightSlots;
            if (local >= kNumWeightSlots) {
                smem->barriers.wait_latent_down_weight_empty(local);
            }

            smem->barriers.latent_down_weight_ready.arrive(local,
                                                           kWeightTileBytes);
            // One SW128 box per 128-K step, each its own TMA destination.
            const std::uint32_t k_start =
                k_tile(local) * LatentDownGemm::kTileShapeK;
#pragma unroll
            for (std::uint32_t h = 0; h < kQuantBlocksPerTile; ++h) {
                tma_load_latent_down_weight_tile(
                    tma_latent_down_weight_desc, row_base(),
                    k_start + h * Dims::kQuantBlockK,
                    &smem->latent_down.w_latent_down[slot][h][0][0],
                    smem->barriers.latent_down_weight_ready.raw(slot));
            }
        }
    }

    __device__ static void load_weight_scale(
        const DataType::ElementScale* __restrict__ latent_down_weight_scale,
        SharedStorage* __restrict__ smem) {
        if (WarpRole::get_warp_idx() != 0u) return;
        const float* const row =
            latent_down_weight_scale + tile_idx() * kNumScaleBlocksK;
        for (std::uint32_t kb = WarpRole::get_lane_idx(); kb < kNumScaleBlocksK;
             kb += WarpRole::kThreadsPerWarp) {
            smem->latent_down.w_latent_down_scale[kb] = row[kb];
        }
    }

    __device__ static void publish(Workspace* __restrict__ workspace,
                                   SharedStorage* __restrict__ smem) {
        WarpRole::sync_calc_threads();
        if (threadIdx.x == 0u) {
            __threadfence();
            workspace->handoff.publish_latent_tasks(smem->launch_parity, 1u);
        }
    }

    static constexpr float kFp8Max = 448.0f;
    static constexpr float kFp8MaxInv = 1.0f / 448.0f;
    static constexpr std::uint32_t kOutChunks = kTileShapeM / kFp8ActChunkK;
    static constexpr std::uint32_t kGemmN = LatentDownGemm::kGemmN;

    __device__ static void stash_output(SharedStorage* __restrict__ smem,
                                        std::uint32_t row, std::uint32_t tok,
                                        __nv_bfloat16 value) {
        smem->latent_down
            .out_bf16[row / kFp8ActChunkK][tok][row % kFp8ActChunkK] = value;
    }

    __device__ static std::uint32_t group_base_rank() {
        return blockIdx.x % Traits::kClusterSize -
               cta_idx() % LatentDownGemm::kClusterSize;
    }

    // Reads partner shared memory: it runs between the two cluster barriers.
    __device__ static float cluster_token_max(SharedStorage* __restrict__ smem,
                                              std::uint32_t tok) {
        float amax = smem->latent_down.token_amax[tok];
#pragma unroll
        for (std::uint32_t rank = 0; rank < LatentDownGemm::kClusterSize;
             ++rank) {
            amax = fmaxf(amax, cluster_load_f32(cluster_map_shared_rank(
                                   &smem->latent_down.token_amax[tok],
                                   group_base_rank() + rank)));
        }
        return amax;
    }

    __device__ static float token_block_max(SharedStorage* __restrict__ smem,
                                            std::uint32_t tok) {
        // A max near FLT_MIN would invert to +inf and NaN the block.
        return fmaxf(cluster_token_max(smem, tok), 1e-10f);
    }

    // Whole-block: the cluster barrier is CTA-collective, and two of them keep
    // a partner out of this shared memory until every read is done.
    __device__ static void finish_quantized_output(
        Workspace* __restrict__ workspace, SharedStorage* __restrict__ smem,
        std::uint32_t batch_size) {
        constexpr std::uint32_t kCells = kOutChunks * kTileShapeN;
        static_assert(kCells <= Traits::kThreads,
                      "one output cell per thread keeps the quantized bytes in "
                      "registers across the cluster barrier");
        const std::uint32_t warp_idx = WarpRole::get_warp_idx();
        const std::uint32_t lane_idx = WarpRole::get_lane_idx();
        const std::uint32_t chunk_base = row_base() / kFp8ActChunkK;

        if (warp_idx < kTileShapeN) {
            float amax = 0.0f;
            for (std::uint32_t i = lane_idx; i < kOutChunks * kFp8ActChunkK;
                 i += WarpRole::kThreadsPerWarp) {
                amax = fmaxf(
                    amax,
                    fabsf(__bfloat162float(
                        smem->latent_down.out_bf16[i / kFp8ActChunkK][warp_idx]
                                                  [i % kFp8ActChunkK])));
            }
            // Bit-pattern max: non-negative floats order like their IEEE-754
            // bits, and `redux.sync.max.f32` deadlocks before a cluster
            // barrier here.
            const float block_amax = __uint_as_float(
                __reduce_max_sync(0xffffffffu, __float_as_uint(amax)));
            if (lane_idx == 0u) {
                smem->latent_down.token_amax[warp_idx] = block_amax;
            }
        }

        cluster_arrive_and_wait();

        const bool owns_slice = split_idx() == 0u;
        const bool is_cell = owns_slice && threadIdx.x < kCells;
        const std::uint32_t cell_chunk = threadIdx.x / kTileShapeN;
        const std::uint32_t cell_tok = threadIdx.x % kTileShapeN;
        const std::uint32_t tok = token_begin() + cell_tok;
        uint4 quantized = {};
        if (is_cell && tok < batch_size) {
            const float inv_scale = kFp8Max / token_block_max(smem, cell_tok);
            const DataType::ElementA* const row =
                &smem->latent_down.out_bf16[cell_chunk][cell_tok][0];
#pragma unroll
            for (std::uint32_t i = 0; i < kFp8ActChunkK; ++i) {
                // An unused token is zeroed below; NaN drops to 0 as in the
                // up phase's quantize.
                float v = __bfloat162float(row[i]);
                v = (v == v) ? v : 0.0f;
                reinterpret_cast<unsigned char*>(&quantized)[i] =
                    static_cast<unsigned char>(__nv_cvt_float_to_fp8(
                        v * inv_scale, __NV_SATFINITE, __NV_E4M3));
            }
        }
        float row_scale = 0.0f;
        const std::uint32_t rank = cta_idx() % LatentDownGemm::kClusterSize;
        const bool owns_scale_row =
            owns_slice && rank == 0u && threadIdx.x < kTileShapeN;
        if (owns_scale_row) {
            row_scale = token_block_max(smem, threadIdx.x) * kFp8MaxInv;
            // An unused token gets the neutral scale used for padding columns.
            if (token_begin() + threadIdx.x >= batch_size) {
                row_scale = 1.0f;
            }
        }

        cluster_arrive_and_wait();

        if (is_cell) {
            workspace->latent_down_out.store_cell(chunk_base + cell_chunk, tok,
                                                  quantized);
        }
        if (owns_scale_row) {
            workspace->latent_down_out.scale_row(
                row_base() / Dims::kQuantBlockM)[token_begin() + threadIdx.x] =
                row_scale;
        }
        if (owns_slice) {
            for (std::uint32_t i = threadIdx.x; i < kOutChunks;
                 i += Traits::kThreads) {
                workspace->latent_down_out.clear_pad_row(chunk_base + i);
            }
        }
        publish_all(workspace, smem);
    }

    __device__ static void publish_all(Workspace* __restrict__ workspace,
                                       SharedStorage* __restrict__ smem) {
        __syncthreads();
        if (threadIdx.x == 0u) {
            __threadfence();
            workspace->handoff.publish_latent_tasks(smem->launch_parity, 1u);
        }
    }
};

template <typename KernelTraits, bool kUseUmma>
struct LatentDownProjection;

template <typename KernelTraits>
struct LatentDownProjection<KernelTraits, true>
    : LatentDownProjectionBase<KernelTraits> {
    using Base = LatentDownProjectionBase<KernelTraits>;
    using typename Base::Element;
    using typename Base::LatentDownGemm;
    using MmaCore = typename LatentDownGemm::MmaCore;
    using Base::kNumScaleBlocksK;
    using Base::kNumSplits;
    using Base::kNumTilesK;
    using Dims = typename KernelTraits::MoEDims;
    using Base::kNumWeightSlots;
    using Base::kQuantBlocksPerTile;
    using Base::kTileShapeM;
    using Base::kTileShapeN;
    using typename Base::SharedStorage;
    using typename Base::Workspace;

    using Traits = KernelTraits;
    static constexpr std::uint32_t kNumAccSlots = Traits::kNumAccSlots;
    static constexpr std::uint32_t kAccColumns = Traits::UpGemm::kAccColumns;
    static_assert(kAccColumns * kNumAccSlots <= Traits::kTmemColumns,
                  "the latent accumulator ring must fit the TMEM allocation");
    // 8-token columns per `16x256b` read.
    static constexpr std::uint32_t kTokenWindows = kTileShapeN / 8u;
    // The up phase restarts the ring at 0, so the K loop must leave every
    // slot's barriers at an even phase.
    static_assert(Base::kTilesKPerCta % (2u * kNumAccSlots) == 0u,
                  "the latent K loop must run whole even laps of the "
                  "accumulator ring");
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

    __device__ static void run(
        CUtensorMap const& tma_latent_down_weight_desc,
        CUtensorMap const& tma_input_desc,
        const DataType::ElementScale* __restrict__ latent_down_weight_scale,
        Workspace* __restrict__ workspace, SharedStorage* __restrict__ smem,
        std::uint32_t batch_size) {
        if (!Base::is_latent_cta()) return;

        run_role(tma_latent_down_weight_desc, tma_input_desc,
                 latent_down_weight_scale, workspace, smem, batch_size);
        if constexpr (LatentDownGemm::kNumSplits > 1u) {
            // Every thread joins: the cluster barrier is block-collective.
            __syncthreads();
            cluster_arrive_and_wait();
            if (WarpRole::is_calc_warp(WarpRole::get_warp_idx())) {
                float acc[kTokenWindows][4] = {};
                gather_split_share(smem, acc);
                if (Base::split_idx() == 0u) {
                    writeback(workspace, smem, acc);
                }
                if constexpr (Traits::kLatentQuantizeOutput) {
                    WarpRole::sync_calc_threads();
                } else if (Base::split_idx() == 0u) {
                    Base::publish(workspace, smem);
                }
            }
            // The rendezvous region is reusable once every block has read it.
            __syncthreads();
            cluster_arrive_and_wait();
        }
        if constexpr (Traits::kLatentQuantizeOutput) {
            Base::finish_quantized_output(workspace, smem, batch_size);
        }
    }

    __device__ static void run_role(
        CUtensorMap const& tma_latent_down_weight_desc,
        CUtensorMap const& tma_input_desc,
        const DataType::ElementScale* __restrict__ latent_down_weight_scale,
        Workspace* __restrict__ workspace, SharedStorage* __restrict__ smem,
        std::uint32_t batch_size) {
        if (WarpRole::is_other_loader_thread()) {
            Base::run_load_input_k_loop(tma_input_desc, smem);
            return;
        }
        if (WarpRole::is_weight_loader_thread()) {
            Base::run_load_weight_k_loop(tma_latent_down_weight_desc, smem);
            return;
        }
        const std::uint32_t warp_idx = WarpRole::get_warp_idx();
        if (WarpRole::is_mma_issuer_warp(warp_idx)) {
            run_mma_k_loop(smem, smem->tmem_ptr);
            return;
        }
        if (WarpRole::is_calc_warp(warp_idx)) {
            run_quantize_and_dequant_k_loop(smem, latent_down_weight_scale,
                                            workspace, batch_size);
        }
    }

    __device__ static void run_mma_k_loop(SharedStorage* __restrict__ smem,
                                          std::uint32_t tmem_base) {
        typename MmaCore::TiledMma tiled_mma{};
        auto cta_mma = tiled_mma.get_slice(cute::_0{});
        // Shape-only proxy: make_fragment_C only reads the layout.
        cute::Tensor tC_proxy = cta_mma.partition_C(cute::make_tensor(
            cute::make_gmem_ptr(static_cast<float*>(nullptr)),
            cute::make_layout(cute::make_shape(cute::Int<Base::kTileShapeM>{},
                                               cute::Int<kTileShapeN>{}),
                              cute::GenRowMajor{})));
        cute::Tensor tCt = cta_mma.make_fragment_C(tC_proxy);

#pragma unroll 1
        for (std::uint32_t local = 0; local < Base::kTilesKPerCta; ++local) {
            const std::uint32_t acc_slot = local % kNumAccSlots;
            const std::uint32_t w_slot = local % Base::kNumWeightSlots;

            if (local >= kNumAccSlots) {
                smem->barriers.wait_accumulator_empty(local);
            }
            smem->barriers.latent_down_input_empty.wait(local);
            smem->barriers.latent_down_weight_ready.wait(local);
            auto tCsB = cute::make_tensor(
                cute::make_smem_ptr(Base::staged_block(smem, local)),
                SBLayout{});
            cute::Tensor tCrB = cta_mma.make_fragment_B(tCsB);
            MmaCore::run<kQuantBlocksPerTile>(
                tiled_mma, cta_mma,
                &smem->latent_down.w_latent_down[w_slot][0][0][0], tCrB, tCt,
                tmem_addr(tmem_base, 0u, acc_slot * kAccColumns));

            cutlass::arch::umma_arrive(
                smem->barriers.latent_down_weight_empty.raw(local));
            cutlass::arch::umma_arrive(
                smem->barriers.accumulator_ready.raw(acc_slot));
        }
    }

    __device__ static void quantize_k_tile(SharedStorage* __restrict__ smem,
                                           std::uint32_t batch_size,
                                           std::uint32_t local) {
        constexpr std::uint32_t kBlocksPerTile =
            Traits::LatentDownGemm::kTileShapeK / Dims::kQuantBlockK;
        const std::uint32_t slot = local % Traits::kLatentInputSlots;
        const std::uint32_t k_iter_id = Base::k_tile(local);
        const std::uint32_t warp_idx = WarpRole::get_calc_warp_idx();
        smem->barriers.latent_down_input_ready.wait(local);
        quantize_fp8_blockwise_128<
            /*kStaged=*/true, kTileShapeN, LatentDownGemm::kGemmN,
            Traits::LatentDownGemm::kTileShapeK,
            SwizzleBoxK<DataType::ElementA>::value, Dims::kQuantBlockK,
            kBlocksPerTile,
            /*kBlocksPerSlot=*/kBlocksPerTile>(
            reinterpret_cast<const DataType::ElementA*>(
                &smem->latent_down.input.data[slot][0][0][0]),
            k_iter_id * Traits::LatentDownGemm::kTileShapeK,
            &smem->latent_down.quantized_input.data[0],
            &smem->latent_down.quantized_input.scale[0][0],
            local % Traits::kLatentInputSlots, k_iter_id * kBlocksPerTile,
            k_iter_id * kBlocksPerTile, batch_size, Base::token_begin(),
            warp_idx, WarpRole::kCalcWarpCount);
        // The fp8 stores must reach the async proxy before the tile is
        // signalled: the issuer reads them through tcgen05.
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");

        WarpRole::sync_calc_threads();
        if (warp_idx == 0u && WarpRole::get_lane_idx() == 0u) {
            smem->barriers.latent_down_input_empty.arrive(local);
        }
    }

    __device__ static void run_quantize_and_dequant_k_loop(
        SharedStorage* __restrict__ smem,
        const DataType::ElementScale* __restrict__ latent_down_weight_scale,
        Workspace* __restrict__ workspace, std::uint32_t batch_size) {
        constexpr std::uint32_t kStageAhead = 1u;
        static_assert(kStageAhead < kNumAccSlots,
                      "the drain must stay inside the accumulator ring");
        static_assert(kStageAhead < Traits::kLatentInputSlots,
                      "the staged tile must stay inside the staging ring");
        quantize_k_tile(smem, batch_size, 0u);

        // The pipeline's first tile sync orders this row before any drain.
        Base::load_weight_scale(latent_down_weight_scale, smem);

        float acc[kTokenWindows][4] = {};
#pragma unroll 1
        for (std::uint32_t local = 0; local < Base::kTilesKPerCta; ++local) {
            if (local + kStageAhead < Base::kTilesKPerCta) {
                quantize_k_tile(smem, batch_size, local + kStageAhead);
            }
            dequant_k_tile(smem, acc, local);
        }

        if constexpr (Base::kNumSplits > 1u) {
            store_split_share(smem, acc);
        } else {
            writeback(workspace, smem, acc);
            if constexpr (Traits::kLatentQuantizeOutput) {
                WarpRole::sync_calc_threads();
            } else {
                Base::publish(workspace, smem);
            }
        }
    }

    // A warp reaches only its own 32-lane sub-partition: a 128-row atom
    // interleaves the groups one per 16-lane window.
    static constexpr bool kNarrowAtom = MmaCore::kAtomShapeM == 64u;

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
        std::uint32_t shares[kNumSplits > 1u ? kNumSplits - 1u : 1u];
#pragma unroll
        for (std::uint32_t r = 0, n = 0u; r < kNumSplits; ++r) {
            if (r == Base::split_idx()) continue;
            shares[n++] = cluster_map_shared_rank(
                &smem->latent_down.splited_mma_out[0][0], Base::group_rank(r));
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

    __device__ static void dequant_k_tile(SharedStorage* __restrict__ smem,
                                          float (&acc)[kTokenWindows][4],
                                          std::uint32_t k_iter_id) {
        const std::uint32_t warp_idx = WarpRole::get_warp_idx();
        const std::uint32_t lane_idx = WarpRole::get_lane_idx();
        // A warp reads its 16 rows via `tcgen05.ld.16x256b` out of the lanes
        // its own sub-partition owns.
        constexpr std::uint32_t kReaderWarps = kTileShapeM / 16u;
        const std::uint32_t t0 = lane_idx & 3u;
        const std::uint32_t token_base = Base::token_begin();

        // Every calc warp waits for the slice's UMMA before arriving: a warp
        // that raced ahead would complete the phase its own k never filled.
        smem->barriers.accumulator_ready.wait(k_iter_id);
#pragma unroll
        for (std::uint32_t sub = 0; sub < kQuantBlocksPerTile; ++sub) {
            // Both scale tables are whole-activation, indexed by the global
            // block, not by the staging tile.
            const std::uint32_t block =
                Base::k_tile(k_iter_id) * kQuantBlocksPerTile + sub;
            const float ws = smem->latent_down.w_latent_down_scale[block];
#pragma unroll
            for (std::uint32_t g = 0; g < kTokenWindows; ++g) {
                float v[4];
                // TMEM reads are warp-collective: a warp past the rows skips.
                if (warp_idx < kReaderWarps) {
                    tmem_ld_16x256b1x(
                        tmem_addr(smem->tmem_ptr, lane_base(warp_idx),
                                  (k_iter_id % kNumAccSlots) * kAccColumns +
                                      sub * MmaCore::kAtomShapeN + g * 8u),
                        v);
                    cutlass::arch::fence_view_async_tmem_load();
                }
                // Value i is (row t1 + 8*(i/2), token 8*g + 2*t0 + i%2).
                const float as[2] = {
                    smem->latent_down.quantized_input
                        .scale[block][token_base + 8u * g + 2u * t0],
                    smem->latent_down.quantized_input
                        .scale[block][token_base + 8u * g + 2u * t0 + 1u]};
#pragma unroll
                for (std::uint32_t i = 0; i < 4u; ++i) {
                    acc[g][i] += v[i] * ws * as[i % 2u];
                }
            }
        }
        if (lane_idx == 0u) {
            smem->barriers.accumulator_empty.arrive(k_iter_id);
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
            Base::row_base() + 16u * group_idx(warp_idx) + lane_idx / 4u;
#pragma unroll
        for (std::uint32_t g = 0; g < kTokenWindows; ++g) {
            const std::uint32_t tok_local = 8u * g + 2u * (lane_idx & 3u);
            const std::uint32_t tok = Base::token_begin() + tok_local;
#pragma unroll
            for (std::uint32_t i = 0; i < 4u; ++i) {
                const __nv_bfloat16 value = __float2bfloat16_rn(acc[g][i]);
                const std::uint32_t row_local =
                    row + 8u * (i / 2u) - Base::row_base();
                if constexpr (Traits::kLatentQuantizeOutput) {
                    Base::stash_output(smem, row_local, tok_local + i % 2u,
                                       value);
                } else {
                    workspace->latent_down_out.store(
                        tok + i % 2u, Base::row_base() + row_local, value);
                }
            }
        }
    }
};

template <typename KernelTraits>
struct LatentDownProjection<KernelTraits, false>
    : LatentDownProjectionBase<KernelTraits> {
    using Base = LatentDownProjectionBase<KernelTraits>;
    using Base::kNumTilesK;
    using Base::kNumWeightSlots;
    using Base::kQuantBlocksPerTile;
    using Base::kTileShapeM;
    using Base::kTileShapeN;
    using typename Base::LatentDownGemm;
    using typename Base::SharedStorage;
    using typename Base::Workspace;
    using MmaCore = typename LatentDownGemm::MmaCore;
    using Dims = typename KernelTraits::MoEDims;
    using Traits = KernelTraits;

    static_assert(MmaCore::kQuantBlockK == Dims::kQuantBlockK,
                  "the instruction's scale blocks must be the input's");
    static constexpr std::uint32_t kNumTilesPerWarp =
        (Base::kNumTilesK + WarpRole::kCalcWarpCount - 1u) /
        WarpRole::kCalcWarpCount;

    __device__ static void run(
        CUtensorMap const& tma_latent_down_weight_desc,
        CUtensorMap const& tma_input_desc,
        const DataType::ElementScale* __restrict__ latent_down_weight_scale,
        Workspace* __restrict__ workspace, SharedStorage* __restrict__ smem,
        std::uint32_t batch_size) {
        if (!Base::is_latent_cta()) return;

        run_role(tma_latent_down_weight_desc, tma_input_desc,
                 latent_down_weight_scale, workspace, smem, batch_size);
        if constexpr (Traits::kLatentQuantizeOutput) {
            Base::finish_quantized_output(workspace, smem, batch_size);
        }
    }

    __device__ static void run_role(
        CUtensorMap const& tma_latent_down_weight_desc,
        CUtensorMap const& tma_input_desc,
        const DataType::ElementScale* __restrict__ latent_down_weight_scale,
        Workspace* __restrict__ workspace, SharedStorage* __restrict__ smem,
        std::uint32_t batch_size) {
        if (WarpRole::is_other_loader_thread()) {
            Base::run_load_input_k_loop(tma_input_desc, smem);
            return;
        }
        if (WarpRole::is_weight_loader_thread()) {
            Base::run_load_weight_k_loop(tma_latent_down_weight_desc, smem);
            return;
        }
        if (WarpRole::is_calc_warp(WarpRole::get_warp_idx())) {
            Base::load_weight_scale(latent_down_weight_scale, smem);

            run_hmma_k_loop(workspace, smem, batch_size);
        }
    }

    __device__ static void quantize_input_tile_per_warp(
        SharedStorage* __restrict__ smem, std::uint32_t batch_size,
        std::uint32_t k_iter_id) {
        constexpr std::uint32_t kInputTileK =
            Traits::LatentDownGemm::kTileShapeK;
        constexpr std::uint32_t kBlocksPerTile =
            kInputTileK / Dims::kQuantBlockK;
        static_assert(kBlocksPerTile == LatentDownGemm::kQuantBlocksPerTile,
                      "a K tile must be whole quantization blocks");

        const std::uint32_t slot = k_iter_id % Traits::kLatentInputSlots;
        const std::uint32_t staging_slot =
            k_iter_id % Traits::kLatentInputSlots;
        smem->barriers.latent_down_input_ready.wait(k_iter_id);
        quantize_fp8_blockwise_128</*kStaged=*/true, Base::kTileShapeN,
                                   LatentDownGemm::kGemmN, kInputTileK,
                                   SwizzleBoxK<DataType::ElementA>::value,
                                   Dims::kQuantBlockK, kBlocksPerTile,
                                   /*kBlocksPerSlot=*/kBlocksPerTile>(
            reinterpret_cast<const DataType::ElementA*>(
                &smem->latent_down.input.data[slot][0][0][0]),
            k_iter_id * kInputTileK, &smem->latent_down.quantized_input.data[0],
            &smem->latent_down.quantized_input.scale[0][0], staging_slot,
            k_iter_id * kBlocksPerTile, k_iter_id * kBlocksPerTile, batch_size,
            Base::token_begin(),
            /*warp_idx=*/0u, /*num_quant_warps=*/1u);
        // The MMAs read it from other lanes; refill uses the async proxy.
        __syncwarp();
        arrive_latent_down_input_empty(smem, k_iter_id);
    }

    __device__ static void arrive_latent_down_input_empty(
        SharedStorage* __restrict__ smem, std::uint32_t k_iter_id) {
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        if (WarpRole::get_lane_idx() == 0u) {
            smem->barriers.latent_down_input_empty.arrive(k_iter_id);
        }
    }

    // The SMEM reads are synchronous, but they must reach the async proxy
    // that refills the slot.
    __device__ static void arrive_latent_down_weight_empty(
        SharedStorage* __restrict__ smem, std::uint32_t k_iter_id) {
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        if (WarpRole::get_lane_idx() == 0u) {
            smem->barriers.latent_down_weight_empty.arrive(k_iter_id);
        }
    }

    __device__ static void run_hmma_k_loop(Workspace* __restrict__ workspace,
                                           SharedStorage* __restrict__ smem,
                                           std::uint32_t batch_size) {
        const std::uint32_t warp_idx = WarpRole::get_calc_warp_idx();

        float acc[MmaCore::kNumAtomMs][MmaCore::kAtomElemsC] = {};
#pragma unroll 1
        for (std::uint32_t t = 0; t < kNumTilesPerWarp; ++t) {
            const std::uint32_t k_iter_id =
                t * WarpRole::kCalcWarpCount + warp_idx;
            if (k_iter_id >= Base::kNumTilesK) break;

            quantize_input_tile_per_warp(smem, batch_size, k_iter_id);

            const std::uint32_t slot = k_iter_id % kNumWeightSlots;
            smem->barriers.latent_down_weight_ready.wait(k_iter_id);
            const std::uint32_t quant_block_k = k_iter_id * kQuantBlocksPerTile;
            const std::uint32_t staged = k_iter_id;
            MmaCore::template run<Dims::kNumTokens>(
                &smem->latent_down.w_latent_down[slot][0][0][0],
                Base::staged_block(smem, staged), acc,
                smem->latent_down.w_latent_down_scale + quant_block_k,
                &smem->latent_down.quantized_input
                     .scale[quant_block_k][Base::token_begin()]);
            arrive_latent_down_weight_empty(smem, k_iter_id);
        }

        reduce_and_store(workspace, smem, acc);
        if constexpr (Traits::kLatentQuantizeOutput) {
            WarpRole::sync_calc_threads();
        } else {
            Base::publish(workspace, smem);
        }
    }

    __device__ static void reduce_and_store(
        Workspace* __restrict__ workspace, SharedStorage* __restrict__ smem,
        float const (&acc)[MmaCore::kNumAtomMs][MmaCore::kAtomElemsC]) {
        const std::uint32_t warp_idx = WarpRole::get_calc_warp_idx();
        const std::uint32_t lane_idx = WarpRole::get_lane_idx();
        // m16n8k32 accumulator: a lane holds rows local and local+8 of each
        // 16-row window, at tokens col and col+1.
        const std::uint32_t local = lane_idx / 4u;
        const std::uint32_t col = 2u * (lane_idx & 3u);
        constexpr std::uint32_t kColPairs = kTileShapeN / 2u;
        // The partials are added in slot order, so the sum is the same every
        // run.
        constexpr std::uint32_t kUnits = MmaCore::kNumAtomMs * kColPairs;
        constexpr std::uint32_t kValuesPerSlot = kTileShapeM * kTileShapeN;
        float2* const cell = reinterpret_cast<float2*>(
            &smem->latent_down.mma_out[warp_idx][0][0]);
#pragma unroll
        for (std::uint32_t w = 0; w < MmaCore::kNumAtomMs; ++w) {
            const std::uint32_t row0 = w * MmaCore::kAtomShapeM + local;
            const std::uint32_t row1 = row0 + MmaCore::kAtomShapeM / 2u;
            cell[row0 * (kTileShapeN / 2u) + col / 2u] =
                make_float2(acc[w][0], acc[w][1]);
            cell[row1 * (kTileShapeN / 2u) + col / 2u] =
                make_float2(acc[w][2], acc[w][3]);
        }
        WarpRole::sync_calc_threads();

        float const* const partials = &smem->latent_down.mma_out[0][0][0];
#pragma unroll
        for (std::uint32_t u = warp_idx; u < kUnits;
             u += WarpRole::kCalcWarpCount) {
            const std::uint32_t row =
                (u / kColPairs) * MmaCore::kAtomShapeM + lane_idx / 2u;
            const std::uint32_t tok = 2u * (u % kColPairs) + lane_idx % 2u;
            float sum = 0.0f;
#pragma unroll
            for (std::uint32_t g = 0; g < WarpRole::kCalcWarpCount; ++g) {
                sum += partials[g * kValuesPerSlot + row * kTileShapeN + tok];
            }
            const __nv_bfloat16 value = __float2bfloat16_rn(sum);
            if constexpr (Traits::kLatentQuantizeOutput) {
                Base::stash_output(smem, row, tok, value);
            } else {
                workspace->latent_down_out.store(Base::token_begin() + tok,
                                                 Base::row_base() + row, value);
            }
        }
    }
};

}  // namespace halops::moe
