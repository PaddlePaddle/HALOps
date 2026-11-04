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
#include "kernel/moe/moe_mainloop_up.cuh"
#include "kernel/moe/moe_shared_storage.cuh"
#include "kernel/moe/moe_tma.cuh"
#include "kernel/moe/moe_traits.cuh"
#include "kernel/moe/moe_workspace.cuh"
#include "kernel/moe/ptx_utils.cuh"
#include "kernel/moe/quant_utils.cuh"

namespace halops::moe {

template <typename KernelTraits>
struct SharedExpertProjectionBase {
    using Traits = KernelTraits;
    using Dims = typename Traits::MoEDims;
    using SharedStorage = MoESharedStorage<Dims>;
    using Workspace = MoEWorkspace<Dims>;
    using SharedUpGemm = SharedUpGemmTraits<Dims>;
    using SharedDownGemm = SharedDownGemmTraits<Dims>;

    static constexpr std::uint32_t kNumDownCtas = SharedDownGemm::kNumCtas;
    static constexpr std::uint32_t kNumUpCtas = SharedUpGemm::kNumCtas;
    static constexpr std::uint32_t kFirstCtaIdx =
        Traits::RouterGemm::kNumCtas + Traits::LatentDownGemm::kNumCtas;
    static constexpr std::uint32_t kNumSplits = SharedUpGemm::kNumSplits;
    static constexpr std::uint32_t kNumUpPublishers = kNumUpCtas / kNumSplits;
    static constexpr std::uint32_t kCluster = Traits::kClusterSize;
    static_assert(kNumUpCtas % kNumSplits == 0u,
                  "the up's blocks must split evenly");
    static_assert(kNumUpCtas % kCluster == 0u,
                  "the up's blocks must fill whole clusters");
    static_assert(kFirstCtaIdx % kCluster == 0u &&
                      kNumDownCtas % kCluster == 0u &&
                      kCluster % kNumSplits == 0u,
                  "the split reduction needs each group inside one cluster: "
                  "the launch's cluster must be at least kNumSplits wide and "
                  "the shared blocks must fill whole clusters");

    static constexpr std::uint32_t kUpTilesK = SharedUpGemm::kTilesKPerCta;
    static constexpr std::uint32_t kDownTilesK = SharedDownGemm::kNumTilesK;
    static constexpr std::uint32_t kDownActChunks =
        SharedDownGemm::kTileShapeK / Workspace::SharedUpOut::kLboChunk;
    static constexpr std::uint32_t kActQuantBlock =
        SharedUpGemm::kActQuantBlock;
    static constexpr std::uint32_t kLboChunk =
        Workspace::SharedUpOut::kLboChunk;
    // One `16x256b` TMEM read is 8 columns.
    static constexpr std::uint32_t kTokenGroups = Dims::kNumTokens / 8u;
    static_assert(SharedUpGemm::kTileShapeM / 16u <= WarpRole::kCalcWarpCount,
                  "the up tile's 16-row groups must fit the calc warps");
    static constexpr std::uint32_t kNumUpSlots = SharedUpGemm::kNumWeightSlots;
    static constexpr std::uint32_t kInputSwizzleBoxK =
        decltype(SharedExpertSmem<Dims>::input)::kSwizzleBoxK;
    static constexpr std::uint32_t kNumDownSlots =
        SharedDownGemm::kNumWeightSlots;

    __device__ static bool is_shared_cta() {
        return blockIdx.x >= kFirstCtaIdx &&
               blockIdx.x < kFirstCtaIdx + kNumDownCtas;
    }

    __device__ static std::uint32_t cta_idx() {
        return blockIdx.x - kFirstCtaIdx;
    }

    __device__ static std::uint32_t slab_idx() {
        return cta_idx() / kNumSplits;
    }

    __device__ static std::uint32_t split_idx() {
        return cta_idx() % kNumSplits;
    }

    __device__ static std::uint32_t cluster_rank() {
        return blockIdx.x % kCluster;
    }

    __device__ static std::uint32_t group_rank(std::uint32_t r) {
        return cluster_rank() - split_idx() + r;
    }

    // A warp reaches only its own 32-lane sub-partition: a 128-row slab
    // interleaves its groups one per 16-lane window.
    static constexpr bool kNarrowSlab = SharedUpGemm::kTileShapeM <= 64u;
    static constexpr bool kAllWarpsRead =
        SharedUpGemm::kTileShapeM / 16u == WarpRole::kCalcWarpCount;

    __device__ static std::uint32_t up_group_idx(std::uint32_t warp_idx) {
        return kNarrowSlab ? warp_idx : 2u * (warp_idx & 3u) + (warp_idx >> 2);
    }

    __device__ static std::uint32_t up_lane_base(std::uint32_t warp_idx) {
        return kNarrowSlab ? 32u * warp_idx : 16u * up_group_idx(warp_idx);
    }

    __device__ static bool owns_up_rows(std::uint32_t warp_idx) {
        return kAllWarpsRead || warp_idx * 16u < SharedUpGemm::kTileShapeM;
    }

    template <std::uint32_t kTileShapeK>
    using SBLayout = decltype(cute::make_layout(
        cute::make_shape(
            cute::make_shape(cute::Int<Dims::kNumTokens>{},
                             cute::make_shape(cute::_16{}, cute::_2{})),
            cute::_1{}, cute::Int<kTileShapeK / 32u>{}),
        cute::make_stride(
            cute::make_stride(
                cute::_16{},
                cute::make_stride(cute::_1{},
                                  cute::Int<(Dims::kNumTokens + 1u) * 16u>{})),
            cute::_0{}, cute::Int<2u * (Dims::kNumTokens + 1u) * 16u>{})));

    using UpSBLayout = SBLayout<SharedUpGemm::kTileShapeK>;
    using DownSBLayout = SBLayout<SharedDownGemm::kTileShapeK>;

    // All threads join; the caller quantizes and completes the cluster handoff.
    __device__ static void join_up_split_k(SharedStorage* __restrict__ smem,
                                           Activation activation,
                                           float situ_beta,
                                           float situ_linear_beta) {
        __syncthreads();
        cluster_arrive_and_wait();
        if (split_idx() == 0u &&
            WarpRole::is_calc_warp(WarpRole::get_warp_idx())) {
            const auto join_span =
                KERNEL_TIMER_BEGIN(smem->timer(), "shared_join_up_split_k");
            combine_up_splited_out(smem, activation, situ_beta,
                                   situ_linear_beta);
            KERNEL_TIMER_END(smem->timer(), join_span);
            WarpRole::sync_calc_threads();
        }
    }

    __device__ static void combine_up_splited_out(
        SharedStorage* __restrict__ smem, Activation activation,
        float situ_beta, float situ_linear_beta) {
        if (activation == Activation::kSitu) {
            combine_up_rows(smem,
                            GatedActFunctor<true>{situ_beta, situ_linear_beta});
        } else {
            combine_up_rows(
                smem, GatedActFunctor<false>{situ_beta, situ_linear_beta});
        }
    }

    template <bool kSitu>
    __device__ static void combine_up_rows(
        SharedStorage* __restrict__ smem,
        GatedActFunctor<kSitu> const& act_functor) {
        const std::uint32_t warp_idx = WarpRole::get_warp_idx();
        const std::uint32_t lane_idx = WarpRole::get_lane_idx();
        const std::uint32_t t0 = lane_idx & 3u;
        const std::uint32_t t1 = lane_idx >> 2;
        const std::uint32_t feature_base = 8u * up_group_idx(warp_idx);
        if (!owns_up_rows(warp_idx)) return;
        constexpr std::uint32_t kPlaneElems =
            SharedUpGemm::kActQuantBlock * Dims::kNumTokens;
        float* const local = &smem->shared_expert.splited_mma_out[0][0][0];
        std::uint32_t shares[kNumSplits > 1u ? kNumSplits - 1u : 1u];
#pragma unroll
        for (std::uint32_t r = 0, n = 0u; r < kNumSplits; ++r) {
            if (r == split_idx()) continue;
            shares[n++] = cluster_map_shared_rank(
                &smem->shared_expert.splited_mma_out[0][0][0], group_rank(r));
        }
#pragma unroll
        for (std::uint32_t group = 0; group < kTokenGroups; ++group) {
#pragma unroll
            for (std::uint32_t i = 0; i < 2u; ++i) {
                const std::uint32_t tok = group * 8u + 2u * t0 + i;
                const std::uint32_t at =
                    (feature_base + t1) * Dims::kNumTokens + tok;
                float gate = local[at];
                float up = local[kPlaneElems + at];
                if constexpr (kNumSplits > 1u) {
#pragma unroll
                    for (std::uint32_t n = 0; n < kNumSplits - 1u; ++n) {
                        gate += cluster_load_f32(shares[n] + 4u * at);
                        up += cluster_load_f32(shares[n] +
                                               4u * (kPlaneElems + at));
                    }
                }
                smem->shared_expert.act_up_out[feature_base + t1][tok] =
                    act_functor(gate, up, 1.0f);
            }
        }
    }

    __device__ static void run_up_load_input_k_loop(
        CUtensorMap const& tma_input_desc, SharedStorage* __restrict__ smem) {
        constexpr std::uint32_t kInputBytes =
            SharedUpGemm::kTileShapeK * Dims::kNumTokens *
            static_cast<std::uint32_t>(sizeof(DataType::ElementA));
        const std::uint32_t k_tile_base = split_idx() * kUpTilesK;

#pragma unroll 1
        for (std::uint32_t k_iter_id = 0; k_iter_id < kUpTilesK; ++k_iter_id) {
            const std::uint32_t slot = k_iter_id % kNumUpSlots;
            if (k_iter_id >= kNumUpSlots) {
                const auto empty_span = KERNEL_TIMER_BEGIN(
                    smem->timer(), "shared_wait_up_input_empty", k_iter_id);
                smem->barriers.shared_up_input_empty.wait(k_iter_id -
                                                          kNumUpSlots);
                KERNEL_TIMER_END(smem->timer(), empty_span);
            }
            smem->barriers.shared_up_input_ready.arrive(k_iter_id, kInputBytes);
            load_input_tile<Dims::kNumTokens>(
                tma_input_desc, k_tile_base + k_iter_id,
                /*token_start=*/0u, smem->shared_expert.input.data[slot],
                smem->barriers.shared_up_input_ready.raw(slot));
        }
    }

    // Load one weight tile; the caller includes any scale bytes in the barrier.
    __device__ static void load_up_weight_tile(
        CUtensorMap const& tma_up_desc, SharedStorage* __restrict__ smem,
        std::uint32_t k_iter_id, std::uint32_t transaction_bytes) {
        const std::uint32_t slot = k_iter_id % kNumUpSlots;
        if (k_iter_id >= kNumUpSlots) {
            const auto empty_span = KERNEL_TIMER_BEGIN(
                smem->timer(), "shared_wait_up_weight_empty", k_iter_id);
            smem->barriers.shared_up_weight_empty.wait(k_iter_id - kNumUpSlots);
            KERNEL_TIMER_END(smem->timer(), empty_span);
        }
        const std::uint32_t k_tile = split_idx() * kUpTilesK + k_iter_id;
        smem->barriers.shared_up_weight_ready.arrive(k_iter_id,
                                                     transaction_bytes);
        for (std::uint32_t b = 0; b < 2u; ++b) {
            tma_load_2d(tma_up_desc, k_tile * 256u + b * 128u,
                        slab_idx() * SharedUpGemm::kTileShapeM,
                        &smem->shared_expert.w_up[slot][b][0][0],
                        smem->barriers.shared_up_weight_ready.raw(slot));
        }
    }

    __device__ static void store_up_partial_sums(
        SharedStorage* __restrict__ smem, const float (&gate)[kTokenGroups][2],
        const float (&up)[kTokenGroups][2]) {
        const std::uint32_t warp_idx = WarpRole::get_warp_idx();
        const std::uint32_t lane_idx = WarpRole::get_lane_idx();
        const std::uint32_t t0 = lane_idx & 3u;
        const std::uint32_t t1 = lane_idx >> 2;
        const std::uint32_t feature_base = 8u * up_group_idx(warp_idx);
        if (!owns_up_rows(warp_idx)) return;
#pragma unroll
        for (std::uint32_t group = 0; group < kTokenGroups; ++group) {
#pragma unroll
            for (std::uint32_t i = 0; i < 2u; ++i) {
                const std::uint32_t tok = group * 8u + 2u * t0 + i;
                const std::uint32_t feature = feature_base + t1;
                smem->shared_expert.splited_mma_out[0][feature][tok] =
                    gate[group][i];
                smem->shared_expert.splited_mma_out[1][feature][tok] =
                    up[group][i];
            }
        }
    }

    __device__ static void run_down_load_input_k_loop(
        Workspace* __restrict__ workspace, SharedStorage* __restrict__ smem,
        const void* __restrict__ scales) {
        SharedExpertSmem<Dims>& sh = smem->shared_expert;
        const auto up_out_span =
            KERNEL_TIMER_BEGIN(smem->timer(), "wait_shared_up_out");
        workspace->cta_comm.wait_shared_up_out(smem->launch_parity,
                                               kNumUpPublishers);
        // The activation was stored by other blocks; the ring reads it
        // through TMA.
        fence_proxy_async_global();
        KERNEL_TIMER_END(smem->timer(), up_out_span);
        // The ring's slots tile the published image exactly.
        static_assert(sizeof(decltype(workspace->shared_up_out.data)) ==
                          sizeof(decltype(sh.act_down_input.data)),
                      "the published activation must match the staging");
        // Both recipes bulk-copy their complete scale image with tile 0.
        constexpr std::uint32_t kScaleBytes =
            static_cast<std::uint32_t>(sizeof(sh.act_down_input.scale));
        static_assert(kNumDownSlots == kDownTilesK,
                      "the published image feeds one slot per K tile");
        // One bulk copy per K tile; the scale table rides tile 0's barrier.
        constexpr std::uint32_t kInputBytes = static_cast<std::uint32_t>(
            sizeof(decltype(sh.act_down_input.data[0])));
#pragma unroll 1
        for (std::uint32_t k_iter_id = 0; k_iter_id < kDownTilesK;
             ++k_iter_id) {
            const std::uint32_t ring = k_iter_id % kNumDownSlots;
            const std::uint32_t kTotalBytes =
                kInputBytes + (k_iter_id == 0u ? kScaleBytes : 0u);
            smem->barriers.shared_down_input_ready.arrive(k_iter_id,
                                                          kTotalBytes);
            tma_load_1d(&workspace->shared_up_out
                             .data[k_iter_id * kDownActChunks][0][0],
                        kInputBytes, &sh.act_down_input.data[ring][0][0][0],
                        smem->barriers.shared_down_input_ready.raw(ring));
            if (k_iter_id == 0u) {
                tma_load_1d(scales, kScaleBytes, &sh.act_down_input.scale[0][0],
                            smem->barriers.shared_down_input_ready.raw(ring));
            }
        }
    }

    __device__ static void load_down_weight_tile(
        CUtensorMap const& tma_down_desc, SharedStorage* __restrict__ smem,
        std::uint32_t k_iter_id, std::uint32_t transaction_bytes) {
        const std::uint32_t slot = k_iter_id % kNumDownSlots;
        if (k_iter_id >= kNumDownSlots) {
            const auto empty_span = KERNEL_TIMER_BEGIN(
                smem->timer(), "shared_wait_down_weight_empty", k_iter_id);
            smem->barriers.shared_down_weight_empty.wait(k_iter_id -
                                                         kNumDownSlots);
            KERNEL_TIMER_END(smem->timer(), empty_span);
        }
        smem->barriers.shared_down_weight_ready.arrive(k_iter_id,
                                                       transaction_bytes);
        tma_load_2d(tma_down_desc, k_iter_id * 128u, row_base(),
                    &smem->shared_expert.w_down[slot][0][0],
                    smem->barriers.shared_down_weight_ready.raw(slot));
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
                out[tok * Dims::kHiddenSize + col] =
                    __float2bfloat16_rn(values[2u * r + i]);
            }
        }
    }
};

template <typename KernelTraits,
          bool UseBlockscaled = SharedUpGemmTraits<
              typename KernelTraits::MoEDims>::MmaCore::kUseBlockscaled>
struct SharedExpertProjection;

// MXFP8: both GEMMs accumulate their K tiles directly in TMEM.
template <typename KernelTraits>
struct SharedExpertProjection<KernelTraits, true>
    : SharedExpertProjectionBase<KernelTraits> {
    using Base = SharedExpertProjectionBase<KernelTraits>;
    using typename Base::Dims;
    using typename Base::DownSBLayout;
    using typename Base::SharedDownGemm;
    using typename Base::SharedStorage;
    using typename Base::SharedUpGemm;
    using typename Base::Traits;
    using typename Base::UpSBLayout;
    using typename Base::Workspace;

    static_assert(SharedUpGemm::MmaCore::kUseBlockscaled &&
                      SharedDownGemm::MmaCore::kUseBlockscaled,
                  "both shared GEMMs must match the projection specialization");

    using UpScaleTraits = typename SharedUpGemm::ScaleTraits;
    using DownScaleTraits = typename SharedDownGemm::ScaleTraits;
    // Each GEMM has one accumulator followed by its current tile's scales.
    static constexpr std::uint32_t kUpSfaTmemOffset = SharedUpGemm::kAccColumns;
    static constexpr std::uint32_t kUpSfbTmemOffset =
        kUpSfaTmemOffset + UpScaleTraits::kSfaTmemColumns;
    static constexpr std::uint32_t kDownSfaTmemOffset =
        SharedDownGemm::kAccColumns;
    static constexpr std::uint32_t kDownSfbTmemOffset =
        kDownSfaTmemOffset + DownScaleTraits::kSfaTmemColumns;
    static_assert(kUpSfbTmemOffset + UpScaleTraits::kSfbTmemColumns <=
                          Traits::kTmemColumns &&
                      kDownSfbTmemOffset + DownScaleTraits::kSfbTmemColumns <=
                          Traits::kTmemColumns,
                  "the shared accumulators and scales must fit in TMEM");

    __device__ static void run(
        CUtensorMap const& tma_up_desc, CUtensorMap const& tma_down_desc,
        CUtensorMap const& tma_input_desc,
        const typename SharedUpGemm::ElementScale* __restrict__ up_scale,
        const typename SharedDownGemm::ElementScale* __restrict__ down_scale,
        DataType::ElementOut* __restrict__ out,
        Workspace* __restrict__ workspace, SharedStorage* __restrict__ smem,
        std::uint32_t batch_size, Activation activation, float situ_beta,
        float situ_linear_beta) {
        if (!Base::is_shared_cta()) return;
        if (Base::cta_idx() < Base::kNumUpCtas) {
            // The quantizer fills the tile's rows; the padded rows of each
            // SFB slot stay zero for UTCCP.
            constexpr std::uint32_t kSfWords =
                (SharedUpGemm::kNumWeightSlots *
                 (128u * (SharedUpGemm::kTileShapeK / 32u))) /
                4u;
            auto* const sf = reinterpret_cast<std::uint32_t*>(
                &smem->shared_expert.quantized_input.scale[0][0]);
            for (std::uint32_t i = threadIdx.x; i < kSfWords;
                 i += static_cast<std::uint32_t>(blockDim.x)) {
                sf[i] = 0u;
            }
            __syncthreads();
            run_up(tma_up_desc, tma_input_desc, up_scale, workspace, smem,
                   batch_size, activation, situ_beta, situ_linear_beta);
        }
        __syncthreads();
        if (Base::cta_idx() < Base::kNumUpCtas && Base::split_idx() == 0u &&
            threadIdx.x == 0u) {
            const auto publish_span =
                KERNEL_TIMER_BEGIN(smem->timer(), "shared_publish_up_out");
            workspace->cta_comm.publish_shared_up_out(smem->launch_parity, 1u);
            KERNEL_TIMER_END(smem->timer(), publish_span);
        }
        run_down(tma_down_desc, down_scale, out, workspace, smem, batch_size);
    }

    __device__ static void run_up(
        CUtensorMap const& tma_up_desc, CUtensorMap const& tma_input_desc,
        const typename SharedUpGemm::ElementScale* __restrict__ up_scale,
        Workspace* __restrict__ workspace, SharedStorage* __restrict__ smem,
        std::uint32_t batch_size, Activation activation, float situ_beta,
        float situ_linear_beta) {
        const auto up_span =
            KERNEL_TIMER_BEGIN(smem->timer(), "shared_expert_up");
        const std::uint32_t warp_idx = WarpRole::get_warp_idx();
        if (WarpRole::is_other_loader_thread()) {
            Base::run_up_load_input_k_loop(tma_input_desc, smem);
        } else if (WarpRole::is_weight_loader_thread()) {
            run_up_load_weights_k_loop(tma_up_desc, up_scale, smem);
        } else if (WarpRole::is_mma_issuer_warp(warp_idx)) {
            run_up_mma_k_loop(smem);
        } else if (WarpRole::is_calc_warp(warp_idx)) {
#pragma unroll 1
            for (std::uint32_t k_iter_id = 0; k_iter_id < Base::kUpTilesK;
                 ++k_iter_id) {
                quantize_up_input_tile(smem, batch_size, k_iter_id);
            }
            float gate[Base::kTokenGroups][2];
            float up[Base::kTokenGroups][2];
            load_up_accumulators(smem, gate, up);
            Base::store_up_partial_sums(smem, gate, up);
        }
        Base::join_up_split_k(smem, activation, situ_beta, situ_linear_beta);
        if (Base::split_idx() == 0u && WarpRole::is_calc_warp(warp_idx)) {
            quantize_up_output(workspace, smem, batch_size);
        }
        // All split readers and publishers finish before the next phase.
        __syncthreads();
        cluster_arrive_and_wait();
        KERNEL_TIMER_END(smem->timer(), up_span);
    }

    __device__ static void run_up_load_weights_k_loop(
        CUtensorMap const& tma_up_desc,
        const typename SharedUpGemm::ElementScale* __restrict__ up_scale,
        SharedStorage* __restrict__ smem) {
        constexpr std::uint32_t kSfAtomBytes = 512u;
        constexpr std::uint32_t kSfKBlocksPerRow = SharedUpGemm::kGemmK / 128u;
#pragma unroll 1
        for (std::uint32_t k_iter_id = 0; k_iter_id < Base::kUpTilesK;
             ++k_iter_id) {
            Base::load_up_weight_tile(
                tma_up_desc, smem, k_iter_id,
                SharedUpGemm::MmaCore::kWeightTileTxBytes +
                    UpScaleTraits::kSfaSmemBytes);
            const std::uint32_t slot = k_iter_id % Base::kNumUpSlots;
            const std::uint32_t k_tile =
                Base::split_idx() * Base::kUpTilesK + k_iter_id;
            const std::uint8_t* const sfa =
                reinterpret_cast<const std::uint8_t*>(up_scale) +
                (Base::slab_idx() * kSfKBlocksPerRow +
                 k_tile * (SharedUpGemm::kTileShapeK / 128u)) *
                    kSfAtomBytes;
            tma_load_1d(sfa, UpScaleTraits::kSfaSmemBytes,
                        smem->shared_expert.w_up_scale[slot],
                        smem->barriers.shared_up_weight_ready.raw(slot));
        }
    }

    __device__ static void run_up_mma_k_loop(SharedStorage* __restrict__ smem) {
        SharedExpertSmem<Dims>& sh = smem->shared_expert;
        typename SharedUpGemm::TiledMma tiled_mma{};
        auto cta_mma = tiled_mma.get_slice(cute::_0{});
        // Shape-only proxy: make_fragment_C only reads the layout.
        cute::Tensor tCt =
            cta_mma.make_fragment_C(cta_mma.partition_C(cute::make_tensor(
                cute::make_gmem_ptr(static_cast<float*>(nullptr)),
                cute::make_layout(
                    cute::make_shape(cute::Int<SharedUpGemm::kTileShapeM>{},
                                     cute::_8{}),
                    cute::GenRowMajor{}))));
#pragma unroll 1
        for (std::uint32_t k_iter_id = 0; k_iter_id < Base::kUpTilesK;
             ++k_iter_id) {
            const std::uint32_t ring = k_iter_id % Base::kNumUpSlots;
            const auto weight_ready_span = KERNEL_TIMER_BEGIN(
                smem->timer(), "shared_up_wait_weight_ready", k_iter_id);
            smem->barriers.shared_up_weight_ready.wait(k_iter_id);
            KERNEL_TIMER_END(smem->timer(), weight_ready_span);
            const auto quantize_ready_span = KERNEL_TIMER_BEGIN(
                smem->timer(), "shared_up_wait_quantize_ready", k_iter_id);
            smem->barriers.shared_up_quantize_ready.wait(k_iter_id);
            KERNEL_TIMER_END(smem->timer(), quantize_ready_span);
            {
                auto tCsB = cute::make_tensor(
                    cute::make_smem_ptr(
                        &sh.quantized_input.data[ring][0][0][0]),
                    UpSBLayout{});
                cute::Tensor tCrB = cta_mma.make_fragment_B(tCsB);
                // SFA was bulk-copied in by the weight loader, SFB was
                // written by the activation quantizer, both in SfAtom
                // layout; UTCCP moves them into TMEM for the instruction.
                std::uint8_t* const sfa = sh.w_up_scale[ring];
                std::uint8_t* const sfb = &sh.quantized_input.scale[ring][0];
                if (cute::elect_one_sync()) {
                    UpScaleTraits::copy_to_tmem(
                        sfa, sfb,
                        tmem_addr(smem->tmem_ptr, 0u, kUpSfaTmemOffset),
                        tmem_addr(smem->tmem_ptr, 0u, kUpSfbTmemOffset));
                }
                // Copies and MMAs from this issuer are ordered. The first
                // K tile clears the fixed accumulator; later tiles add to it.
                SharedUpGemm::MmaCore::run(
                    tiled_mma, cta_mma, &sh.w_up[ring][0][0][0], tCrB, tCt,
                    tmem_addr(smem->tmem_ptr, 0u, 0u),
                    tmem_addr(smem->tmem_ptr, 0u, kUpSfaTmemOffset),
                    tmem_addr(smem->tmem_ptr, 0u, kUpSfbTmemOffset),
                    /*accumulate=*/k_iter_id != 0u);
            }
            cutlass::arch::umma_arrive(
                smem->barriers.shared_up_weight_empty.raw(ring));
        }
        // Hand off the completed split once; its weight slots are now drained.
        cutlass::arch::umma_arrive(
            smem->barriers.shared_up_accumulator_ready.raw(0u));
    }

    __device__ static void quantize_up_input_tile(
        SharedStorage* __restrict__ smem, std::uint32_t batch_size,
        std::uint32_t k_iter_id) {
        const auto quantize_span = KERNEL_TIMER_BEGIN(
            smem->timer(), "shared_quantize_up_input_tile", k_iter_id);
        const std::uint32_t warp_idx = WarpRole::get_calc_warp_idx();
        const std::uint32_t lane_idx = WarpRole::get_lane_idx();
        constexpr std::uint32_t kBlocksPerTile =
            SharedUpGemm::kTileShapeK / Dims::kQuantConfig.kBlockSize;
        const std::uint32_t slot = k_iter_id % Base::kNumUpSlots;
        if (k_iter_id >= Base::kNumUpSlots) {
            const auto weight_empty_span = KERNEL_TIMER_BEGIN(
                smem->timer(), "shared_wait_up_weight_empty", k_iter_id);
            smem->barriers.shared_up_weight_empty.wait(k_iter_id -
                                                       Base::kNumUpSlots);
            KERNEL_TIMER_END(smem->timer(), weight_empty_span);
        }
        const auto input_ready_span = KERNEL_TIMER_BEGIN(
            smem->timer(), "shared_wait_up_input_ready", k_iter_id);
        smem->barriers.shared_up_input_ready.wait(k_iter_id);
        KERNEL_TIMER_END(smem->timer(), input_ready_span);
        const std::uint32_t k_base =
            (Base::split_idx() * Base::kUpTilesK + k_iter_id) *
            SharedUpGemm::kTileShapeK;
        const std::uint32_t blocks_base =
            Base::split_idx() * SharedUpGemm::kQuantBlocksPerCta +
            k_iter_id * kBlocksPerTile;
        quantize_mxfp8_groupwise_1x32<
            /*Staged=*/true, Dims::kNumTokens, SharedUpGemm::kTileShapeK,
            Dims::kQuantConfig.kBlockSize, Base::kInputSwizzleBoxK,
            kBlocksPerTile>(
            reinterpret_cast<const DataType::ElementA*>(
                &smem->shared_expert.input.data[slot][0][0][0]),
            k_base, &smem->shared_expert.quantized_input.data[0],
            &smem->shared_expert.quantized_input.scale[0][0], slot, blocks_base,
            batch_size, /*row_start=*/0u, warp_idx, WarpRole::kCalcWarpCount);
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        WarpRole::sync_calc_threads();
        if (warp_idx == 0u && lane_idx == 0u) {
            smem->barriers.shared_up_input_empty.arrive(k_iter_id);
            smem->barriers.shared_up_quantize_ready.arrive(k_iter_id);
        }
        KERNEL_TIMER_END(smem->timer(), quantize_span);
    }

    // Read the completed split once; there is no per-K readout or sum.
    __device__ static void load_up_accumulators(
        SharedStorage* __restrict__ smem, float (&gate)[Base::kTokenGroups][2],
        float (&up)[Base::kTokenGroups][2]) {
        const auto ready_span =
            KERNEL_TIMER_BEGIN(smem->timer(), "shared_up_wait_accumulator");
        smem->barriers.shared_up_accumulator_ready.wait(0u);
        KERNEL_TIMER_END(smem->timer(), ready_span);
        tcgen05_fence_after_thread_sync();
        const std::uint32_t warp_idx = WarpRole::get_warp_idx();
        if (!Base::owns_up_rows(warp_idx)) return;
#pragma unroll
        for (std::uint32_t group = 0; group < Base::kTokenGroups; ++group) {
            float values[4];
            tmem_ld_16x256b1x(
                tmem_addr(smem->tmem_ptr, Base::up_lane_base(warp_idx),
                          group * 8u),
                values);
            cutlass::arch::fence_view_async_tmem_load();
#pragma unroll
            for (std::uint32_t i = 0; i < 2u; ++i) {
                gate[group][i] = values[i];
                up[group][i] = values[i + 2u];
            }
        }
    }

    __device__ static void quantize_up_output(Workspace* __restrict__ workspace,
                                              SharedStorage* __restrict__ smem,
                                              std::uint32_t batch_size) {
        const auto quantize_span =
            KERNEL_TIMER_BEGIN(smem->timer(), "shared_quantize_up_output");
        const std::uint32_t warp_idx = WarpRole::get_warp_idx();
        const std::uint32_t lane_idx = WarpRole::get_lane_idx();

        constexpr std::uint32_t kQuantGroups = Base::kActQuantBlock / 32u;
        // One UE8M0 byte per 32-feature group of each token, written
        // straight into the down's SFB SfAtom slab for this slab's K tile.
        const std::uint32_t tile =
            Base::slab_idx() / (128u / Base::kActQuantBlock);
        const std::uint32_t group_base =
            (Base::slab_idx() * Base::kActQuantBlock % 128u) / 32u;
        std::uint8_t* const sf = workspace->shared_up_out.sf_slab(tile);
#pragma unroll 1
        for (std::uint32_t tok = warp_idx; tok < batch_size;
             tok += WarpRole::kCalcWarpCount) {
            float v[kQuantGroups];
            float inv[kQuantGroups];
#pragma unroll
            for (std::uint32_t i = 0; i < kQuantGroups; ++i) {
                v[i] = smem->shared_expert.act_up_out[lane_idx + i * 32u][tok];
                std::uint32_t biased;
                inv[i] = mxfp8_ue8m0_scale(warp_reduce_max_float(fabsf(v[i])),
                                           biased);
                if (lane_idx == 0u) {
                    sf[sf_atom_offset(tok, group_base + i)] =
                        static_cast<std::uint8_t>(biased);
                }
            }
#pragma unroll
            for (std::uint32_t i = 0; i < kQuantGroups; ++i) {
                const std::uint32_t feat = lane_idx + i * 32u;
                const std::uint32_t feature =
                    Base::slab_idx() * Base::kActQuantBlock + feat;
                workspace->shared_up_out.cell(feature / Base::kLboChunk,
                                              tok)[feat % Base::kLboChunk] =
                    static_cast<DataType::ElementQA>(v[i] * inv[i]);
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
        KERNEL_TIMER_END(smem->timer(), quantize_span);
    }

    __device__ static void run_down(
        CUtensorMap const& tma_down_desc,
        const typename SharedDownGemm::ElementScale* __restrict__ down_scale,
        DataType::ElementOut* __restrict__ out,
        Workspace* __restrict__ workspace, SharedStorage* __restrict__ smem,
        std::uint32_t batch_size) {
        const auto down_span =
            KERNEL_TIMER_BEGIN(smem->timer(), "shared_expert_down");
        const std::uint32_t warp_idx = WarpRole::get_warp_idx();
        if (WarpRole::is_other_loader_thread()) {
            static_assert(sizeof(workspace->shared_up_out.scale.mxfp8) ==
                              sizeof(smem->shared_expert.act_down_input.scale),
                          "the published scales must match the staging");
            Base::run_down_load_input_k_loop(
                workspace, smem, workspace->shared_up_out.scale_bytes());
        } else if (WarpRole::is_weight_loader_thread()) {
            run_down_load_weight_k_loop(tma_down_desc, down_scale, smem);
        } else if (WarpRole::is_mma_issuer_warp(warp_idx)) {
            run_down_mma_k_loop(smem);
        } else if (WarpRole::is_calc_warp(warp_idx)) {
            drain_down(out, smem, batch_size);
        }
        KERNEL_TIMER_END(smem->timer(), down_span);
    }

    __device__ static void run_down_load_weight_k_loop(
        CUtensorMap const& tma_down_desc,
        const typename SharedDownGemm::ElementScale* __restrict__ down_scale,
        SharedStorage* __restrict__ smem) {
        constexpr std::uint32_t kSfAtomBytes = DownScaleTraits::kSfaAtomBytes;
        constexpr std::uint32_t kSfKBlocksPerRow =
            SharedDownGemm::kGemmK / 128u;
        const std::uint32_t row_atom = Base::row_base() / 128u;
        for (std::uint32_t k_iter_id = 0; k_iter_id < Base::kDownTilesK;
             ++k_iter_id) {
            Base::load_down_weight_tile(
                tma_down_desc, smem, k_iter_id,
                SharedDownGemm::MmaCore::kWeightTileTxBytes +
                    DownScaleTraits::kSfaSmemBytes);
            const std::uint32_t slot = k_iter_id % Base::kNumDownSlots;
#pragma unroll
            for (std::uint32_t atom = 0; atom < DownScaleTraits::kNumSfaAtoms;
                 ++atom) {
                const std::uint8_t* const sfa =
                    reinterpret_cast<const std::uint8_t*>(down_scale) +
                    ((row_atom + atom) * kSfKBlocksPerRow + k_iter_id) *
                        kSfAtomBytes;
                tma_load_1d(sfa, kSfAtomBytes,
                            smem->shared_expert.w_down_scale[slot] +
                                atom * kSfAtomBytes,
                            smem->barriers.shared_down_weight_ready.raw(slot));
            }
        }
    }

    __device__ static void run_down_mma_k_loop(
        SharedStorage* __restrict__ smem) {
        const auto mma_span =
            KERNEL_TIMER_BEGIN(smem->timer(), "shared_down_mma");
        SharedExpertSmem<Dims>& sh = smem->shared_expert;
        typename SharedDownGemm::TiledMma tiled_mma{};
        auto cta_mma = tiled_mma.get_slice(cute::_0{});
        cute::Tensor tCt =
            cta_mma.make_fragment_C(cta_mma.partition_C(cute::make_tensor(
                cute::make_gmem_ptr(static_cast<float*>(nullptr)),
                cute::make_layout(cute::make_shape(cute::_128{}, cute::_8{}),
                                  cute::GenRowMajor{}))));
#pragma unroll 1
        for (std::uint32_t t = 0; t < Base::kDownTilesK; ++t) {
            const std::uint32_t ring = t % Base::kNumDownSlots;
            smem->barriers.shared_down_weight_ready.wait(t);
            smem->barriers.shared_down_input_ready.wait(t);
            auto tCsB = cute::make_tensor(
                cute::make_smem_ptr(&sh.act_down_input.data[ring][0][0][0]),
                DownSBLayout{});
            cute::Tensor tCrB = cta_mma.make_fragment_B(tCsB);
            std::uint8_t* const sfa = sh.w_down_scale[ring];
            std::uint8_t* const sfb = &sh.act_down_input.scale[ring][0];
            if (cute::elect_one_sync()) {
                // Each 128-row atom's scales sit in their own TMEM slab;
                // SFB is one slab every atom reads, so it moves once.
                for (std::uint32_t atom = 0;
                     atom < DownScaleTraits::kNumSfaAtoms; ++atom) {
                    DownScaleTraits::copy_sfa_to_tmem(
                        sfa + atom * DownScaleTraits::kSfaAtomBytes,
                        tmem_addr(smem->tmem_ptr, 0u,
                                  kDownSfaTmemOffset +
                                      atom * DownScaleTraits::kSfaAtomColumns));
                }
                DownScaleTraits::copy_sfb_to_tmem(
                    sfb, tmem_addr(smem->tmem_ptr, 0u, kDownSfbTmemOffset));
            }
            // Reuse the fixed sum and scale staging in issuer order.
            SharedDownGemm::MmaCore::run(
                tiled_mma, cta_mma, &sh.w_down[ring][0][0], tCrB, tCt,
                tmem_addr(smem->tmem_ptr, 0u, 0u),
                tmem_addr(smem->tmem_ptr, 0u, kDownSfaTmemOffset),
                tmem_addr(smem->tmem_ptr, 0u, kDownSfbTmemOffset),
                /*accumulate=*/t != 0u);
            cutlass::arch::umma_arrive(
                smem->barriers.shared_down_weight_empty.raw(ring));
        }
        cutlass::arch::umma_arrive(
            smem->barriers.shared_down_accumulator_ready.raw(0u));
        KERNEL_TIMER_END(smem->timer(), mma_span);
    }

    // One readout of the final TMEM sum, followed by the output conversion.
    __device__ static void drain_down(DataType::ElementOut* __restrict__ out,
                                      SharedStorage* __restrict__ smem,
                                      std::uint32_t batch_size) {
        const std::uint32_t warp_idx = WarpRole::get_warp_idx();
        const std::uint32_t lane_base =
            32u * (warp_idx & 3u) + 16u * (warp_idx >> 2);
        const auto done_span =
            KERNEL_TIMER_BEGIN(smem->timer(), "shared_down_wait_done");
        smem->barriers.shared_down_accumulator_ready.wait(0u);
        KERNEL_TIMER_END(smem->timer(), done_span);
        tcgen05_fence_after_thread_sync();
        const auto readout_span =
            KERNEL_TIMER_BEGIN(smem->timer(), "shared_down_readout");
#pragma unroll
        for (std::uint32_t atom = 0; atom < SharedDownGemm::MmaCore::kNumAtomMs;
             ++atom) {
#pragma unroll
            for (std::uint32_t group = 0; group < Base::kTokenGroups; ++group) {
                float values[4];
                tmem_ld_16x256b1x(
                    tmem_addr(smem->tmem_ptr, lane_base,
                              atom * SharedDownGemm::kTileShapeN + group * 8u),
                    values);
                cutlass::arch::fence_view_async_tmem_load();
                Base::store_down_output(out, batch_size, atom, group, values);
            }
        }
        KERNEL_TIMER_END(smem->timer(), readout_span);
    }
};

// Blockwise FP8: calc warps dequantize each K tile before summing.
template <typename KernelTraits>
struct SharedExpertProjection<KernelTraits, false>
    : SharedExpertProjectionBase<KernelTraits> {
    using Base = SharedExpertProjectionBase<KernelTraits>;
    using typename Base::Dims;
    using typename Base::DownSBLayout;
    using typename Base::SharedDownGemm;
    using typename Base::SharedStorage;
    using typename Base::SharedUpGemm;
    using typename Base::Traits;
    using typename Base::UpSBLayout;
    using typename Base::Workspace;

    static_assert(!SharedUpGemm::MmaCore::kUseBlockscaled &&
                      !SharedDownGemm::MmaCore::kUseBlockscaled,
                  "both shared GEMMs must match the projection specialization");

    static constexpr std::uint32_t kNumUpAccSlots = Traits::kNumAccSlots;
    static_assert(kNumUpAccSlots * SharedUpGemm::kAccColumns <=
                          Traits::kTmemColumns &&
                      Base::kDownTilesK * SharedDownGemm::kAccColumns <=
                          Traits::kTmemColumns,
                  "the shared accumulator tiles must fit in TMEM");

    __device__ static void run(
        CUtensorMap const& tma_up_desc, CUtensorMap const& tma_down_desc,
        CUtensorMap const& tma_input_desc,
        const typename SharedUpGemm::ElementScale* __restrict__ up_scale,
        const typename SharedDownGemm::ElementScale* __restrict__ down_scale,
        DataType::ElementOut* __restrict__ out,
        Workspace* __restrict__ workspace, SharedStorage* __restrict__ smem,
        std::uint32_t batch_size, Activation activation, float situ_beta,
        float situ_linear_beta) {
        if (!Base::is_shared_cta()) return;
        if (Base::cta_idx() < Base::kNumUpCtas) {
            run_up(tma_up_desc, tma_input_desc, up_scale, workspace, smem,
                   batch_size, activation, situ_beta, situ_linear_beta);
        }
        __syncthreads();
        if (Base::cta_idx() < Base::kNumUpCtas && Base::split_idx() == 0u &&
            threadIdx.x == 0u) {
            const auto publish_span =
                KERNEL_TIMER_BEGIN(smem->timer(), "shared_publish_up_out");
            workspace->cta_comm.publish_shared_up_out(smem->launch_parity, 1u);
            KERNEL_TIMER_END(smem->timer(), publish_span);
        }
        run_down(tma_down_desc, down_scale, out, workspace, smem, batch_size);
    }

    __device__ static void run_up(
        CUtensorMap const& tma_up_desc, CUtensorMap const& tma_input_desc,
        const typename SharedUpGemm::ElementScale* __restrict__ up_scale,
        Workspace* __restrict__ workspace, SharedStorage* __restrict__ smem,
        std::uint32_t batch_size, Activation activation, float situ_beta,
        float situ_linear_beta) {
        const auto up_span =
            KERNEL_TIMER_BEGIN(smem->timer(), "shared_expert_up");
        const std::uint32_t warp_idx = WarpRole::get_warp_idx();
        if (WarpRole::is_other_loader_thread()) {
            Base::run_up_load_input_k_loop(tma_input_desc, smem);
        } else if (WarpRole::is_weight_loader_thread()) {
            run_up_load_weights_k_loop(tma_up_desc, smem);
        } else if (WarpRole::is_mma_issuer_warp(warp_idx)) {
            run_up_mma_k_loop(smem);
        } else if (WarpRole::is_calc_warp(warp_idx)) {
            run_up_quantize_and_dequant_k_loop(up_scale, smem, batch_size);
        }
        Base::join_up_split_k(smem, activation, situ_beta, situ_linear_beta);
        if (Base::split_idx() == 0u && WarpRole::is_calc_warp(warp_idx)) {
            quantize_up_output(workspace, smem, batch_size);
        }
        // All split readers and publishers finish before the next phase.
        __syncthreads();
        cluster_arrive_and_wait();
        KERNEL_TIMER_END(smem->timer(), up_span);
    }

    __device__ static void run_up_load_weights_k_loop(
        CUtensorMap const& tma_up_desc, SharedStorage* __restrict__ smem) {
#pragma unroll 1
        for (std::uint32_t k_iter_id = 0; k_iter_id < Base::kUpTilesK;
             ++k_iter_id) {
            Base::load_up_weight_tile(
                tma_up_desc, smem, k_iter_id,
                SharedUpGemm::MmaCore::kWeightTileTxBytes);
        }
    }

    __device__ static void run_up_mma_k_loop(SharedStorage* __restrict__ smem) {
        SharedExpertSmem<Dims>& sh = smem->shared_expert;
        typename SharedUpGemm::TiledMma tiled_mma{};
        auto cta_mma = tiled_mma.get_slice(cute::_0{});
        // Shape-only proxy: make_fragment_C only reads the layout.
        cute::Tensor tCt =
            cta_mma.make_fragment_C(cta_mma.partition_C(cute::make_tensor(
                cute::make_gmem_ptr(static_cast<float*>(nullptr)),
                cute::make_layout(
                    cute::make_shape(cute::Int<SharedUpGemm::kTileShapeM>{},
                                     cute::_8{}),
                    cute::GenRowMajor{}))));
#pragma unroll 1
        for (std::uint32_t k_iter_id = 0; k_iter_id < Base::kUpTilesK;
             ++k_iter_id) {
            const std::uint32_t ring = k_iter_id % Base::kNumUpSlots;
            const std::uint32_t acc_slot = k_iter_id % kNumUpAccSlots;
            if (k_iter_id >= kNumUpAccSlots) {
                const auto acc_empty_span = KERNEL_TIMER_BEGIN(
                    smem->timer(), "shared_wait_up_accumulator_empty",
                    k_iter_id);
                smem->barriers.shared_up_accumulator_empty.wait(k_iter_id -
                                                                kNumUpAccSlots);
                KERNEL_TIMER_END(smem->timer(), acc_empty_span);
            }
            const auto weight_ready_span = KERNEL_TIMER_BEGIN(
                smem->timer(), "shared_up_wait_weight_ready", k_iter_id);
            smem->barriers.shared_up_weight_ready.wait(k_iter_id);
            KERNEL_TIMER_END(smem->timer(), weight_ready_span);
            const auto quantize_ready_span = KERNEL_TIMER_BEGIN(
                smem->timer(), "shared_up_wait_quantize_ready", k_iter_id);
            smem->barriers.shared_up_quantize_ready.wait(k_iter_id);
            KERNEL_TIMER_END(smem->timer(), quantize_ready_span);
            {
                auto tCsB = cute::make_tensor(
                    cute::make_smem_ptr(
                        &sh.quantized_input.data[ring][0][0][0]),
                    UpSBLayout{});
                cute::Tensor tCrB = cta_mma.make_fragment_B(tCsB);
                SharedUpGemm::MmaCore::template run<2u>(
                    tiled_mma, cta_mma, &sh.w_up[ring][0][0][0], tCrB, tCt,
                    acc_slot * SharedUpGemm::kAccColumns);
            }
            cutlass::arch::umma_arrive(
                smem->barriers.shared_up_weight_empty.raw(ring));
            cutlass::arch::umma_arrive(
                smem->barriers.shared_up_accumulator_ready.raw(acc_slot));
        }
    }

    __device__ static void run_up_quantize_and_dequant_k_loop(
        const typename SharedUpGemm::ElementScale* __restrict__ up_scale,
        SharedStorage* __restrict__ smem, std::uint32_t batch_size) {
        float gate[Base::kTokenGroups][2] = {};
        float up[Base::kTokenGroups][2] = {};
        quantize_up_input_tile(smem, batch_size, 0u);
#pragma unroll 1
        for (std::uint32_t k_iter_id = 1; k_iter_id < Base::kUpTilesK;
             ++k_iter_id) {
            quantize_up_input_tile(smem, batch_size, k_iter_id);
            dequant_up_tile(up_scale, smem, k_iter_id - 1u, gate, up);
        }
        dequant_up_tile(up_scale, smem, Base::kUpTilesK - 1u, gate, up);
        Base::store_up_partial_sums(smem, gate, up);
    }

    __device__ static void quantize_up_input_tile(
        SharedStorage* __restrict__ smem, std::uint32_t batch_size,
        std::uint32_t k_iter_id) {
        const auto quantize_span = KERNEL_TIMER_BEGIN(
            smem->timer(), "shared_quantize_up_input_tile", k_iter_id);
        const std::uint32_t warp_idx = WarpRole::get_calc_warp_idx();
        const std::uint32_t lane_idx = WarpRole::get_lane_idx();
        constexpr std::uint32_t kBlocksPerTile =
            SharedUpGemm::kTileShapeK / Dims::kQuantConfig.kBlockSize;
        const std::uint32_t slot = k_iter_id % Base::kNumUpSlots;
        if (k_iter_id >= Base::kNumUpSlots) {
            const auto weight_empty_span = KERNEL_TIMER_BEGIN(
                smem->timer(), "shared_wait_up_weight_empty", k_iter_id);
            smem->barriers.shared_up_weight_empty.wait(k_iter_id -
                                                       Base::kNumUpSlots);
            KERNEL_TIMER_END(smem->timer(), weight_empty_span);
        }
        const auto input_ready_span = KERNEL_TIMER_BEGIN(
            smem->timer(), "shared_wait_up_input_ready", k_iter_id);
        smem->barriers.shared_up_input_ready.wait(k_iter_id);
        KERNEL_TIMER_END(smem->timer(), input_ready_span);
        const std::uint32_t k_base =
            (Base::split_idx() * Base::kUpTilesK + k_iter_id) *
            SharedUpGemm::kTileShapeK;
        const std::uint32_t blocks_base =
            Base::split_idx() * SharedUpGemm::kQuantBlocksPerCta +
            k_iter_id * kBlocksPerTile;
        quantize_fp8_groupwise_128<
            /*Staged=*/true, Dims::kNumTokens, Dims::kHiddenSize,
            Dims::kNumTokens, Dims::kQuantConfig.kBlockSize,
            Base::kInputSwizzleBoxK, kBlocksPerTile>(
            reinterpret_cast<const DataType::ElementA*>(
                &smem->shared_expert.input.data[slot][0][0][0]),
            k_base, &smem->shared_expert.quantized_input.data[slot],
            &smem->shared_expert.quantized_input.scale[0][0],
            /*staging_slot_base=*/0u, blocks_base, blocks_base, batch_size,
            /*row_start=*/0u, warp_idx, WarpRole::kCalcWarpCount);
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        WarpRole::sync_calc_threads();
        if (warp_idx == 0u && lane_idx == 0u) {
            smem->barriers.shared_up_input_empty.arrive(k_iter_id);
            smem->barriers.shared_up_quantize_ready.arrive(k_iter_id);
        }
        KERNEL_TIMER_END(smem->timer(), quantize_span);
    }

    __device__ static void dequant_up_tile(
        const typename SharedUpGemm::ElementScale* __restrict__ up_scale,
        SharedStorage* __restrict__ smem, std::uint32_t k_iter_id,
        float (&gate)[Base::kTokenGroups][2],
        float (&up)[Base::kTokenGroups][2]) {
        const auto dequant_span = KERNEL_TIMER_BEGIN(
            smem->timer(), "shared_dequant_up_tile", k_iter_id);
        const std::uint32_t warp_idx = WarpRole::get_warp_idx();
        const std::uint32_t lane_idx = WarpRole::get_lane_idx();
        const std::uint32_t acc_slot = k_iter_id % kNumUpAccSlots;
        const auto acc_ready_span = KERNEL_TIMER_BEGIN(
            smem->timer(), "shared_up_wait_accumulator", k_iter_id);
        smem->barriers.shared_up_accumulator_ready.wait(k_iter_id);
        KERNEL_TIMER_END(smem->timer(), acc_ready_span);
        tcgen05_fence_after_thread_sync();
        const std::uint32_t t0 = lane_idx & 3u;
        const std::uint32_t lane_base = Base::up_lane_base(warp_idx);
        const bool reads_tile = Base::owns_up_rows(warp_idx);
        const std::uint32_t gate_row = Base::slab_idx() * Base::kActQuantBlock /
                                       Dims::kQuantConfig.kBlockSize;
        const std::uint32_t up_row =
            Dims::kIntermediateSize / Dims::kQuantConfig.kBlockSize + gate_row;
        constexpr std::uint32_t kScaleBlocksK =
            Dims::kHiddenSize / Dims::kQuantConfig.kBlockSize;
        const std::uint32_t kblk_base =
            Base::split_idx() * SharedUpGemm::kQuantBlocksPerCta +
            k_iter_id * 2u;
#pragma unroll
        for (std::uint32_t s = 0; reads_tile && s < 2u; ++s) {
            const std::uint32_t kblk = kblk_base + s;
            const float ws_gate = up_scale[gate_row * kScaleBlocksK + kblk];
            const float ws_up = up_scale[up_row * kScaleBlocksK + kblk];
#pragma unroll
            for (std::uint32_t group = 0; group < Base::kTokenGroups; ++group) {
                float v[4];
                tmem_ld_16x256b1x(
                    tmem_addr(smem->tmem_ptr, lane_base,
                              acc_slot * SharedUpGemm::kAccColumns +
                                  s * Dims::kNumTokens + group * 8u),
                    v);
                cutlass::arch::fence_view_async_tmem_load();
#pragma unroll
                for (std::uint32_t i = 0; i < 2u; ++i) {
                    const std::uint32_t tok = group * 8u + 2u * t0 + i;
                    const float as =
                        smem->shared_expert.quantized_input.scale[kblk][tok];
                    gate[group][i] += v[i] * ws_gate * as;
                    up[group][i] += v[i + 2u] * ws_up * as;
                }
            }
        }
        cutlass::arch::fence_view_async_tmem_load();
        WarpRole::sync_calc_threads();
        if (warp_idx == 0u && lane_idx == 0u) {
            smem->barriers.shared_up_accumulator_empty.arrive(k_iter_id);
        }
        KERNEL_TIMER_END(smem->timer(), dequant_span);
    }

    __device__ static void quantize_up_output(Workspace* __restrict__ workspace,
                                              SharedStorage* __restrict__ smem,
                                              std::uint32_t batch_size) {
        const auto quantize_span =
            KERNEL_TIMER_BEGIN(smem->timer(), "shared_quantize_up_output");
        const std::uint32_t warp_idx = WarpRole::get_warp_idx();
        const std::uint32_t lane_idx = WarpRole::get_lane_idx();

        constexpr std::uint32_t kQuantGroups = Base::kActQuantBlock / 32u;
#pragma unroll 1
        for (std::uint32_t tok = warp_idx; tok < batch_size;
             tok += WarpRole::kCalcWarpCount) {
            float v[kQuantGroups];
            float local_max = 0.0f;
#pragma unroll
            for (std::uint32_t i = 0; i < kQuantGroups; ++i) {
                v[i] = smem->shared_expert.act_up_out[lane_idx + i * 32u][tok];
                local_max = fmaxf(local_max, fabsf(v[i]));
            }
            const float clamped =
                fmaxf(warp_reduce_max_float(local_max), 1e-10f);
            if (lane_idx == 0u) {
                workspace->shared_up_out.scale_row(Base::slab_idx())[tok] =
                    clamped * (1.0f / 448.0f);
            }
            const float inv = 448.0f / clamped;
#pragma unroll
            for (std::uint32_t i = 0; i < kQuantGroups; ++i) {
                const std::uint32_t feat = lane_idx + i * 32u;
                const std::uint32_t feature =
                    Base::slab_idx() * Base::kActQuantBlock + feat;
                workspace->shared_up_out.cell(feature / Base::kLboChunk,
                                              tok)[feat % Base::kLboChunk] =
                    static_cast<DataType::ElementQA>(v[i] * inv);
            }
        }
        KERNEL_TIMER_END(smem->timer(), quantize_span);
    }

    __device__ static void run_down(
        CUtensorMap const& tma_down_desc,
        const typename SharedDownGemm::ElementScale* __restrict__ down_scale,
        DataType::ElementOut* __restrict__ out,
        Workspace* __restrict__ workspace, SharedStorage* __restrict__ smem,
        std::uint32_t batch_size) {
        const auto down_span =
            KERNEL_TIMER_BEGIN(smem->timer(), "shared_expert_down");
        const std::uint32_t warp_idx = WarpRole::get_warp_idx();
        if (WarpRole::is_other_loader_thread()) {
            static_assert(sizeof(workspace->shared_up_out.scale.blockwise) ==
                              sizeof(smem->shared_expert.act_down_input.scale),
                          "the published scales must match the staging");
            Base::run_down_load_input_k_loop(
                workspace, smem, workspace->shared_up_out.scale_ptr());
        } else if (WarpRole::is_weight_loader_thread()) {
            run_down_load_weight_k_loop(tma_down_desc, smem);
        } else if (WarpRole::is_mma_issuer_warp(warp_idx)) {
            run_down_mma_k_loop(smem);
        } else if (WarpRole::is_calc_warp(warp_idx)) {
            drain_down(down_scale, out, smem, batch_size);
        }
        KERNEL_TIMER_END(smem->timer(), down_span);
    }

    __device__ static void run_down_load_weight_k_loop(
        CUtensorMap const& tma_down_desc, SharedStorage* __restrict__ smem) {
        for (std::uint32_t k_iter_id = 0; k_iter_id < Base::kDownTilesK;
             ++k_iter_id) {
            Base::load_down_weight_tile(
                tma_down_desc, smem, k_iter_id,
                SharedDownGemm::MmaCore::kWeightTileTxBytes);
        }
    }

    __device__ static void run_down_mma_k_loop(
        SharedStorage* __restrict__ smem) {
        const auto mma_span =
            KERNEL_TIMER_BEGIN(smem->timer(), "shared_down_mma");
        SharedExpertSmem<Dims>& sh = smem->shared_expert;
        typename SharedDownGemm::TiledMma tiled_mma{};
        auto cta_mma = tiled_mma.get_slice(cute::_0{});
        cute::Tensor tCt =
            cta_mma.make_fragment_C(cta_mma.partition_C(cute::make_tensor(
                cute::make_gmem_ptr(static_cast<float*>(nullptr)),
                cute::make_layout(cute::make_shape(cute::_128{}, cute::_8{}),
                                  cute::GenRowMajor{}))));
#pragma unroll 1
        for (std::uint32_t t = 0; t < Base::kDownTilesK; ++t) {
            const std::uint32_t ring = t % Base::kNumDownSlots;
            smem->barriers.shared_down_weight_ready.wait(t);
            smem->barriers.shared_down_input_ready.wait(t);
            auto tCsB = cute::make_tensor(
                cute::make_smem_ptr(&sh.act_down_input.data[ring][0][0][0]),
                DownSBLayout{});
            cute::Tensor tCrB = cta_mma.make_fragment_B(tCsB);
            SharedDownGemm::MmaCore::template run<
                SharedDownGemm::kNumSubBlocks>(
                tiled_mma, cta_mma, &sh.w_down[ring][0][0], tCrB, tCt,
                t * SharedDownGemm::kAccColumns);
            cutlass::arch::umma_arrive(
                smem->barriers.shared_down_weight_empty.raw(ring));
        }
        cutlass::arch::umma_arrive(
            smem->barriers.shared_down_accumulator_ready.raw(0u));
        KERNEL_TIMER_END(smem->timer(), mma_span);
    }

    __device__ static void drain_down(
        const typename SharedDownGemm::ElementScale* __restrict__ down_scale,
        DataType::ElementOut* __restrict__ out,
        SharedStorage* __restrict__ smem, std::uint32_t batch_size) {
        SharedExpertSmem<Dims>& sh = smem->shared_expert;
        const std::uint32_t warp_idx = WarpRole::get_warp_idx();
        const std::uint32_t lane_idx = WarpRole::get_lane_idx();
        // The routed down's readout: warp w covers lanes 32*(w&3)+16*(w>>2).
        const std::uint32_t lane_base =
            32u * (warp_idx & 3u) + 16u * (warp_idx >> 2);
        const std::uint32_t t0 = lane_idx & 3u;

        const auto done_span =
            KERNEL_TIMER_BEGIN(smem->timer(), "shared_down_wait_done");
        smem->barriers.shared_down_accumulator_ready.wait(0u);
        KERNEL_TIMER_END(smem->timer(), done_span);

        const auto readout_span =
            KERNEL_TIMER_BEGIN(smem->timer(), "shared_down_readout");
#pragma unroll
        for (std::uint32_t atom = 0; atom < 2u; ++atom) {
#pragma unroll
            for (std::uint32_t group = 0; group < Base::kTokenGroups; ++group) {
                float acc[4] = {};
#pragma unroll 1
                for (std::uint32_t t = 0; t < Base::kDownTilesK; ++t) {
#pragma unroll
                    for (std::uint32_t s = 0; s < SharedDownGemm::kNumSubBlocks;
                         ++s) {
                        float v[4];
                        tmem_ld_16x256b1x(
                            tmem_addr(smem->tmem_ptr, lane_base,
                                      t * SharedDownGemm::kAccColumns +
                                          atom * SharedDownGemm::kNumSubBlocks *
                                              Dims::kNumTokens +
                                          s * Dims::kNumTokens + group * 8u),
                            v);
                        cutlass::arch::fence_view_async_tmem_load();
                        const float ws =
                            down_scale[(Base::cta_idx() * 2u + atom) *
                                           Base::kDownTilesK +
                                       t];
                        for (std::uint32_t i = 0; i < 2u; ++i) {
                            const std::uint32_t tok = group * 8u + 2u * t0 + i;
                            const float as =
                                sh.act_down_input
                                    .scale[t * SharedDownGemm::kNumSubBlocks +
                                           s][tok];
                            acc[i] += v[i] * ws * as;
                            acc[i + 2u] += v[i + 2u] * ws * as;
                        }
                    }
                }
                Base::store_down_output(out, batch_size, atom, group, acc);
            }
        }
        KERNEL_TIMER_END(smem->timer(), readout_span);
    }
};

}  // namespace halops::moe
