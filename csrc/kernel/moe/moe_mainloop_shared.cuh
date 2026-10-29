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

#include "mma_core.cuh"
#include "moe_configs.cuh"
#include "moe_input.cuh"
#include "moe_mainloop_up.cuh"
#include "moe_shared_storage.cuh"
#include "moe_tma.h"
#include "moe_traits.cuh"
#include "moe_workspace.cuh"
#include "ptx_utils.cuh"

namespace halops::moe {

template <typename KernelTraits>
struct SharedExpertProjection {
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
    static constexpr std::uint32_t kNumUpAccSlots = Traits::kNumAccSlots;
    static constexpr std::uint32_t kInputSwizzleBoxK =
        decltype(SharedExpertSmem<Dims>::input)::kSwizzleBoxK;
    static constexpr std::uint32_t kNumDownSlots =
        SharedDownGemm::kNumWeightSlots;
    static_assert(kNumUpAccSlots * SharedUpGemm::kAccColumns <=
                          Traits::kTmemColumns &&
                      kDownTilesK * SharedDownGemm::kAccColumns <=
                          Traits::kTmemColumns,
                  "each GEMM's column blocks must fit the TMEM allocation");

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

    __device__ static void run(
        CUtensorMap const& tma_up_desc, CUtensorMap const& tma_down_desc,
        CUtensorMap const& tma_input_desc,
        const DataType::ElementScale* __restrict__ up_scale,
        const DataType::ElementScale* __restrict__ down_scale,
        DataType::ElementOut* __restrict__ out,
        Workspace* __restrict__ workspace, SharedStorage* __restrict__ smem,
        std::uint32_t batch_size, Activation activation,
        GatedActParams const& act_params) {
        if (!is_shared_cta()) return;
        if (cta_idx() < kNumUpCtas) {
            run_up(tma_up_desc, tma_input_desc, up_scale, workspace, smem,
                   batch_size, activation, act_params);
        }
        __syncthreads();
        if (cta_idx() < kNumUpCtas && split_idx() == 0u && threadIdx.x == 0u) {
            __threadfence();
            workspace->handoff.publish_shared_up_out(smem->launch_parity, 1u);
        }
        run_down(tma_down_desc, down_scale, out, workspace, smem, batch_size);
    }

    __device__ static void run_up(
        CUtensorMap const& tma_up_desc, CUtensorMap const& tma_input_desc,
        const DataType::ElementScale* __restrict__ up_scale,
        Workspace* __restrict__ workspace, SharedStorage* __restrict__ smem,
        std::uint32_t batch_size, Activation activation,
        GatedActParams const& act_params) {
        const std::uint32_t warp_idx = WarpRole::get_warp_idx();
        if (WarpRole::is_other_loader_thread()) {
            run_up_load_input_k_loop(tma_input_desc, smem);
        } else if (WarpRole::is_weight_loader_thread()) {
            run_up_load_weights_k_loop(tma_up_desc, smem);
        } else if (WarpRole::is_mma_issuer_warp(warp_idx)) {
            run_up_mma_k_loop(smem);
        } else if (WarpRole::is_calc_warp(warp_idx)) {
            run_up_quantize_and_dequant_k_loop(up_scale, smem, batch_size);
        }
        join_up_split_k(workspace, smem, batch_size, activation, act_params);
    }

    // Every thread joins: the cluster barrier is block-collective.
    __device__ static void join_up_split_k(Workspace* __restrict__ workspace,
                                           SharedStorage* __restrict__ smem,
                                           std::uint32_t batch_size,
                                           Activation activation,
                                           GatedActParams const& act_params) {
        __syncthreads();
        cluster_arrive_and_wait();
        if (split_idx() == 0u &&
            WarpRole::is_calc_warp(WarpRole::get_warp_idx())) {
            combine_up_splited_out(smem, activation, act_params);
            WarpRole::sync_calc_threads();
            quantize_up_output(workspace, smem, batch_size);
        }
        // No block may move on before every block has read the shares.
        __syncthreads();
        cluster_arrive_and_wait();
    }

    __device__ static void combine_up_splited_out(
        SharedStorage* __restrict__ smem, Activation activation,
        GatedActParams const& act_params) {
        if (activation == Activation::kSitu) {
            combine_up_rows<true>(smem, act_params);
        } else {
            combine_up_rows<false>(smem, act_params);
        }
    }

    template <bool kSitu>
    __device__ static void combine_up_rows(SharedStorage* __restrict__ smem,
                                           GatedActParams const& act_params) {
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
                    moe_gated_act<kSitu>(gate, up, 1.0f, act_params);
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
                smem->barriers.shared_up_input_empty.wait(k_iter_id -
                                                          kNumUpSlots);
            }
            smem->barriers.shared_up_input_ready.arrive(k_iter_id, kInputBytes);
            load_input_tile<Dims::kNumTokens>(
                tma_input_desc, k_tile_base + k_iter_id,
                /*token_start=*/0u, smem->shared_expert.input.data[slot],
                smem->barriers.shared_up_input_ready.raw(slot));
        }
    }

    __device__ static void run_up_load_weights_k_loop(
        CUtensorMap const& tma_up_desc, SharedStorage* __restrict__ smem) {
        const std::uint32_t k_tile_base = split_idx() * kUpTilesK;
#pragma unroll 1
        for (std::uint32_t k_iter_id = 0; k_iter_id < kUpTilesK; ++k_iter_id) {
            const std::uint32_t slot = k_iter_id % kNumUpSlots;
            if (k_iter_id >= kNumUpSlots) {
                smem->barriers.shared_up_weight_empty.wait(k_iter_id -
                                                           kNumUpSlots);
            }
            smem->barriers.shared_up_weight_ready.arrive(
                k_iter_id, 2u * SharedUpGemm::kTileShapeM * 128u);
            for (std::uint32_t b = 0; b < 2u; ++b) {
                tma_load_2d(
                    tma_up_desc, (k_tile_base + k_iter_id) * 256u + b * 128u,
                    slab_idx() * SharedUpGemm::kTileShapeM,
                    &smem->shared_expert.w_up[slot][b][0][0],
                    smem->barriers.shared_up_weight_ready.raw(k_iter_id));
            }
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
        for (std::uint32_t k_iter_id = 0; k_iter_id < kUpTilesK; ++k_iter_id) {
            const std::uint32_t ring = k_iter_id % kNumUpSlots;
            const std::uint32_t acc_slot = k_iter_id % kNumUpAccSlots;
            if (k_iter_id >= kNumUpAccSlots) {
                smem->barriers.shared_up_accumulator_empty.wait(k_iter_id -
                                                                kNumUpAccSlots);
            }
            smem->barriers.shared_up_weight_ready.wait(k_iter_id);
            smem->barriers.shared_up_quantize_ready.wait(k_iter_id);
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
        const DataType::ElementScale* __restrict__ up_scale,
        SharedStorage* __restrict__ smem, std::uint32_t batch_size) {
        float gate[kTokenGroups][2] = {};
        float up[kTokenGroups][2] = {};
        quantize_up_input_tile(smem, batch_size, 0u);
#pragma unroll 1
        for (std::uint32_t k_iter_id = 1; k_iter_id < kUpTilesK; ++k_iter_id) {
            quantize_up_input_tile(smem, batch_size, k_iter_id);
            dequant_up_tile(up_scale, smem, k_iter_id - 1u, gate, up);
        }
        dequant_up_tile(up_scale, smem, kUpTilesK - 1u, gate, up);
        store_up_partial_sums(smem, gate, up);
    }

    __device__ static void quantize_up_input_tile(
        SharedStorage* __restrict__ smem, std::uint32_t batch_size,
        std::uint32_t k_iter_id) {
        const std::uint32_t warp_idx = WarpRole::get_calc_warp_idx();
        const std::uint32_t lane_idx = WarpRole::get_lane_idx();
        constexpr std::uint32_t kBlocksPerTile =
            SharedUpGemm::kTileShapeK / Dims::kQuantBlockK;
        const std::uint32_t slot = k_iter_id % kNumUpSlots;
        if (k_iter_id >= kNumUpSlots) {
            smem->barriers.shared_up_weight_empty.wait(k_iter_id - kNumUpSlots);
        }
        smem->barriers.shared_up_input_ready.wait(k_iter_id);
        const std::uint32_t k_base =
            (split_idx() * kUpTilesK + k_iter_id) * SharedUpGemm::kTileShapeK;
        const std::uint32_t blocks_base =
            split_idx() * SharedUpGemm::kQuantBlocksPerCta +
            k_iter_id * kBlocksPerTile;
        quantize_fp8_blockwise_128<
            /*kStaged=*/true, Dims::kNumTokens, Dims::kNumTokens,
            Dims::kHiddenSize, kInputSwizzleBoxK, Dims::kQuantBlockK,
            kBlocksPerTile, /*kBlocksPerSlot=*/kBlocksPerTile>(
            reinterpret_cast<const DataType::ElementA*>(
                &smem->shared_expert.input.data[slot][0][0][0]),
            k_base, &smem->shared_expert.quantized_input.data[slot],
            &smem->shared_expert.quantized_input.scale[0][0],
            /*staging_slot_base=*/0u, blocks_base, blocks_base, batch_size,
            /*token_begin=*/0u, warp_idx, WarpRole::kCalcWarpCount);
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        WarpRole::sync_calc_threads();
        if (warp_idx == 0u && lane_idx == 0u) {
            smem->barriers.shared_up_input_empty.arrive(k_iter_id);
            smem->barriers.shared_up_quantize_ready.arrive(k_iter_id);
        }
    }

    __device__ static void dequant_up_tile(
        const DataType::ElementScale* __restrict__ up_scale,
        SharedStorage* __restrict__ smem, std::uint32_t k_iter_id,
        float (&gate)[kTokenGroups][2], float (&up)[kTokenGroups][2]) {
        const std::uint32_t warp_idx = WarpRole::get_warp_idx();
        const std::uint32_t lane_idx = WarpRole::get_lane_idx();
        const std::uint32_t acc_slot = k_iter_id % kNumUpAccSlots;
        smem->barriers.shared_up_accumulator_ready.wait(k_iter_id);
        tcgen05_fence_after_thread_sync();
        const std::uint32_t t0 = lane_idx & 3u;
        const std::uint32_t lane_base = up_lane_base(warp_idx);
        const bool reads_tile = owns_up_rows(warp_idx);
        const std::uint32_t gate_row =
            slab_idx() * kActQuantBlock / Dims::kQuantBlockM;
        const std::uint32_t up_row =
            Dims::kIntermediateSize / Dims::kQuantBlockM + gate_row;
        constexpr std::uint32_t kScaleBlocksK =
            Dims::kHiddenSize / Dims::kQuantBlockK;
        const std::uint32_t kblk_base =
            split_idx() * SharedUpGemm::kQuantBlocksPerCta + k_iter_id * 2u;
#pragma unroll
        for (std::uint32_t s = 0; reads_tile && s < 2u; ++s) {
            const std::uint32_t kblk = kblk_base + s;
            const float ws_gate = up_scale[gate_row * kScaleBlocksK + kblk];
            const float ws_up = up_scale[up_row * kScaleBlocksK + kblk];
#pragma unroll
            for (std::uint32_t group = 0; group < kTokenGroups; ++group) {
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

    __device__ static void quantize_up_output(Workspace* __restrict__ workspace,
                                              SharedStorage* __restrict__ smem,
                                              std::uint32_t batch_size) {
        const std::uint32_t warp_idx = WarpRole::get_warp_idx();
        const std::uint32_t lane_idx = WarpRole::get_lane_idx();

        constexpr std::uint32_t kQuantGroups = kActQuantBlock / 32u;
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
                workspace->shared_up_out.scale_row(slab_idx())[tok] =
                    clamped * (1.0f / 448.0f);
            }
            const float inv = 448.0f / clamped;
#pragma unroll
            for (std::uint32_t i = 0; i < kQuantGroups; ++i) {
                const std::uint32_t feat = lane_idx + i * 32u;
                const std::uint32_t feature =
                    slab_idx() * kActQuantBlock + feat;
                workspace->shared_up_out.cell(feature / kLboChunk,
                                              tok)[feat % kLboChunk] =
                    static_cast<DataType::ElementQA>(v[i] * inv);
            }
        }
    }

    __device__ static void run_down(
        CUtensorMap const& tma_down_desc,
        const DataType::ElementScale* __restrict__ down_scale,
        DataType::ElementOut* __restrict__ out,
        Workspace* __restrict__ workspace, SharedStorage* __restrict__ smem,
        std::uint32_t batch_size) {
        const std::uint32_t warp_idx = WarpRole::get_warp_idx();
        if (WarpRole::is_other_loader_thread()) {
            run_down_load_input_k_loop(workspace, smem);
        } else if (WarpRole::is_weight_loader_thread()) {
            run_down_load_weight_k_loop(tma_down_desc, smem);
        } else if (WarpRole::is_mma_issuer_warp(warp_idx)) {
            run_down_mma_k_loop(smem);
        } else if (WarpRole::is_calc_warp(warp_idx)) {
            drain_down(down_scale, out, workspace, smem, batch_size);
        }
    }

    __device__ static void run_down_load_input_k_loop(
        Workspace* __restrict__ workspace, SharedStorage* __restrict__ smem) {
        SharedExpertSmem<Dims>& sh = smem->shared_expert;
        workspace->handoff.wait_shared_up_out(smem->launch_parity,
                                              kNumUpPublishers);
        // The ring's slots tile the published image exactly.
        static_assert(sizeof(decltype(workspace->shared_up_out.data)) ==
                          sizeof(decltype(sh.act_down_input.data)),
                      "the published activation must match the staging");
        static_assert(sizeof(decltype(workspace->shared_up_out.scale)) ==
                          sizeof(decltype(sh.act_down_input.scale)),
                      "the published scales must match the staging");
        // One bulk copy per K tile; the scale table rides tile 0's barrier.
        constexpr std::uint32_t kInputBytes = static_cast<std::uint32_t>(
            sizeof(decltype(sh.act_down_input.data[0])));
#pragma unroll 1
        for (std::uint32_t k_iter_id = 0; k_iter_id < kDownTilesK;
             ++k_iter_id) {
            const std::uint32_t ring = k_iter_id % kNumDownSlots;
            const std::uint32_t kTotalBytes =
                kInputBytes + (k_iter_id == 0u
                                   ? static_cast<std::uint32_t>(sizeof(
                                         decltype(sh.act_down_input.scale)))
                                   : 0u);
            smem->barriers.shared_down_input_ready.arrive(k_iter_id,
                                                          kTotalBytes);
            tma_load_bulk(&workspace->shared_up_out
                               .data[k_iter_id * kDownActChunks][0][0],
                          kInputBytes, &sh.act_down_input.data[ring][0][0][0],
                          smem->barriers.shared_down_input_ready.raw(ring));
            if (k_iter_id == 0u) {
                tma_load_bulk(workspace->shared_up_out.scale_ptr(),
                              static_cast<std::uint32_t>(
                                  sizeof(workspace->shared_up_out.scale)),
                              &sh.act_down_input.scale[0][0],
                              smem->barriers.shared_down_input_ready.raw(ring));
            }
        }
    }

    __device__ static void run_down_load_weight_k_loop(
        CUtensorMap const& tma_down_desc, SharedStorage* __restrict__ smem) {
        SharedExpertSmem<Dims>& sh = smem->shared_expert;
        for (std::uint32_t k_iter_id = 0; k_iter_id < kDownTilesK;
             ++k_iter_id) {
            const std::uint32_t ring = k_iter_id % kNumDownSlots;
            if (k_iter_id >= kNumDownSlots) {
                smem->barriers.shared_down_weight_empty.wait(k_iter_id -
                                                             kNumDownSlots);
            }
            smem->barriers.shared_down_weight_ready.arrive(
                k_iter_id, SharedDownGemm::kTileShapeM * 128u);
            tma_load_2d(tma_down_desc, k_iter_id * 128u, row_base(),
                        &sh.w_down[ring][0][0],
                        smem->barriers.shared_down_weight_ready.raw(ring));
        }
    }

    __device__ static void run_down_mma_k_loop(
        SharedStorage* __restrict__ smem) {
        SharedExpertSmem<Dims>& sh = smem->shared_expert;
        typename SharedDownGemm::TiledMma tiled_mma{};
        auto cta_mma = tiled_mma.get_slice(cute::_0{});
        cute::Tensor tCt =
            cta_mma.make_fragment_C(cta_mma.partition_C(cute::make_tensor(
                cute::make_gmem_ptr(static_cast<float*>(nullptr)),
                cute::make_layout(cute::make_shape(cute::_128{}, cute::_8{}),
                                  cute::GenRowMajor{}))));
#pragma unroll 1
        for (std::uint32_t t = 0; t < kDownTilesK; ++t) {
            const std::uint32_t ring = t % kNumDownSlots;
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
    }

    __device__ static std::uint32_t row_base() {
        return cta_idx() * SharedDownGemm::kTileShapeM;
    }

    __device__ static void drain_down(
        const DataType::ElementScale* __restrict__ down_scale,
        DataType::ElementOut* __restrict__ out,
        Workspace* __restrict__ workspace, SharedStorage* __restrict__ smem,
        std::uint32_t batch_size) {
        SharedExpertSmem<Dims>& sh = smem->shared_expert;
        const std::uint32_t warp_idx = WarpRole::get_warp_idx();
        const std::uint32_t lane_idx = WarpRole::get_lane_idx();
        // The routed down's readout: warp w covers lanes 32*(w&3)+16*(w>>2).
        const std::uint32_t lane_base =
            32u * (warp_idx & 3u) + 16u * (warp_idx >> 2);
        const std::uint32_t t0 = lane_idx & 3u;
        const std::uint32_t t1 = lane_idx >> 2;

        smem->barriers.shared_down_accumulator_ready.wait(0u);
#pragma unroll
        for (std::uint32_t atom = 0; atom < 2u; ++atom) {
#pragma unroll
            for (std::uint32_t group = 0; group < kTokenGroups; ++group) {
                float acc[4] = {};
#pragma unroll 1
                for (std::uint32_t t = 0; t < kDownTilesK; ++t) {
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
                            down_scale[(cta_idx() * 2u + atom) * kDownTilesK +
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
                for (std::uint32_t r = 0; r < 2u; ++r) {
                    const std::uint32_t col =
                        row_base() + atom * 128u + lane_base + t1 + 8u * r;
                    for (std::uint32_t i = 0; i < 2u; ++i) {
                        const std::uint32_t tok = group * 8u + 2u * t0 + i;
                        if (tok >= batch_size) continue;
                        out[tok * Dims::kHiddenSize + col] =
                            __float2bfloat16_rn(acc[2u * r + i]);
                    }
                }
            }
        }
    }
};

}  // namespace halops::moe
