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

#include <cutlass/array.h>

#include <cute/tensor.hpp>

#include "kernel/moe/moe_shared_storage.cuh"
#include "kernel/moe/moe_tma.cuh"
#include "kernel/moe/moe_traits.cuh"
#include "kernel/moe/moe_workspace.cuh"
#include "kernel/moe/ptx_utils.cuh"
#include "kernel/moe/quant_utils.cuh"

namespace halops::moe {

template <bool kSitu>
struct GatedActFunctor {
    float beta;
    float inv_beta;
    float linear_beta;
    float inv_linear_beta;

    __host__ __device__ explicit GatedActFunctor(float gate_scale,
                                                 float up_scale) {
        beta = gate_scale;
        inv_beta = (gate_scale > 0.0f) ? (1.0f / gate_scale) : 0.0f;
        linear_beta = up_scale;
        inv_linear_beta = (up_scale > 0.0f) ? (1.0f / up_scale) : 0.0f;
    }

    // tanh(z) = 1 - 2/(1 + e^{2z}), safe at both saturating ends.
    __device__ static __forceinline__ float fast_tanh(float z) {
        return 1.0f - 2.0f / (1.0f + __expf(2.0f * z));
    }

    // Combine one (gate, up) pair with the routing weight.
    __device__ __forceinline__ float operator()(float gate, float up,
                                                float rw) const {
        if constexpr (!kSitu) {
            return rw * up * gate / (1.0f + __expf(-gate));
        } else {
            const float sigmoid_gate = 1.0f / (1.0f + __expf(-gate));
            const float capped_gate =
                (beta > 0.0f) ? beta * fast_tanh(gate * inv_beta) * sigmoid_gate
                              : gate * sigmoid_gate;
            const float capped_up =
                (linear_beta > 0.0f)
                    ? linear_beta * fast_tanh(up * inv_linear_beta)
                    : up;
            return rw * capped_gate * capped_up;
        }
    }
};

// Shared up-projection computation; callers own scheduling and output layout.
template <typename KernelTraits, typename GemmTraits>
struct UpProjectionBase {
    using Traits = KernelTraits;
    using Dims = typename Traits::MoEDims;
    using SharedStorage = MoESharedStorage<Dims>;
    using Workspace = MoEWorkspace<Dims>;
    using UpGemm = GemmTraits;
    using MmaCore = typename UpGemm::MmaCore;
    using WeightElement = typename MmaCore::SmemElementA;
    using ActElement = typename UpGemm::InputQuant::ElementType;
    using TiledMma = typename UpGemm::TiledMma;
    using UpScaleTraits = typename UpGemm::ScaleTraits;

    static constexpr std::uint32_t kNumSplits = UpGemm::kNumSplits;
    static constexpr std::uint32_t kNumTilesK = UpGemm::kNumTilesK;
    static constexpr std::uint32_t kNumWeightSlots = UpGemm::kNumWeightSlots;
    static constexpr std::uint32_t kNumAccSlots = UpGemm::kNumAccSlots;
    static constexpr std::uint32_t kAccColumns = get_gemm_acc_columns<UpGemm>();
    static constexpr std::uint32_t kUpSfaTmemOffset =
        kNumAccSlots * kAccColumns;
    static constexpr std::uint32_t kUpSfbTmemOffset =
        kUpSfaTmemOffset + UpScaleTraits::kSfaTmemColumns;
    static constexpr std::uint32_t kUpWeightTxBytes =
        MmaCore::kWeightTileTxBytes +
        (MmaCore::kUseBlockscaled ? UpScaleTraits::kSfaSmemBytes : 0u);

    static constexpr std::uint32_t kOutputFeatureBlock =
        UpGemm::kTileShapeM / 2u;
    static constexpr std::uint32_t kOutputQuantBlock =
        UpGemm::kOutputQuantBlock;
    static constexpr std::uint32_t kTokenGroups = UpGemm::kTileShapeN / 8u;
    static constexpr std::uint32_t kAccColsPerThread = kTokenGroups * 2u;
    static constexpr std::uint32_t kQuantGroups = kOutputFeatureBlock / 32u;

    using Accumulator = cutlass::Array<float, kAccColsPerThread>;
    using OutputQuant =
        QuantSpec<UpGemm::InputQuant::kMethod, kOutputQuantBlock>;
    using OutputQuantizer =
        GroupWiseQuantizer<OutputQuant, kOutputFeatureBlock>;

    static_assert(kNumSplits > 0u && kNumTilesK % kNumSplits == 0u,
                  "the K tiles must divide evenly over the splits");
    static_assert(Traits::kClusterSize % kNumSplits == 0u,
                  "each split group must fit inside one aligned cluster");
    static_assert(Traits::kPairLayoutGranularity == 8u,
                  "the TMEM readout pairs gate/up rows eight rows apart");
    static_assert(UpGemm::kTileShapeM / 16u <= WarpRole::kCalcWarpCount,
                  "the tile's 16-row groups must fit the calc warps");
    static constexpr bool kNarrowSlab = UpGemm::kTileShapeM <= 64u;
    static constexpr bool kAllWarpsRead =
        UpGemm::kTileShapeM / 16u == WarpRole::kCalcWarpCount;

    static constexpr std::uint64_t kBLbo =
        (UpGemm::kTileShapeN + 1u) * std::uint64_t{kFp8ActChunkK};
    static constexpr std::uint32_t kBKStride = 2u * kBLbo;
    using SBLayout = decltype(cute::make_layout(
        cute::make_shape(
            cute::make_shape(cute::Int<UpGemm::kTileShapeN>{},
                             cute::make_shape(cute::_16{}, cute::_2{})),
            cute::_1{}, cute::Int<MmaCore::kNumAtomKs>{}),
        cute::make_stride(
            cute::make_stride(
                cute::_16{}, cute::make_stride(cute::_1{}, cute::Int<kBLbo>{})),
            cute::_0{}, cute::Int<kBKStride>{})));
    static_assert(cute::stride<0, 0>(SBLayout{}) == 16 &&
                      cute::get<1>(cute::stride<0, 1>(SBLayout{})) == kBLbo &&
                      cute::stride<2>(SBLayout{}) == kBKStride,
                  "the B operand must match the padded 16-byte chunks");

    __device__ static std::uint32_t up_group_idx(std::uint32_t warp_idx) {
        return kNarrowSlab ? warp_idx : 2u * (warp_idx & 3u) + (warp_idx >> 2);
    }

    __device__ static std::uint32_t up_lane_base(std::uint32_t warp_idx) {
        return kNarrowSlab ? 32u * warp_idx : 16u * up_group_idx(warp_idx);
    }

    __device__ static bool owns_up_rows(std::uint32_t warp_idx) {
        return kAllWarpsRead || warp_idx * 16u < UpGemm::kTileShapeM;
    }

    template <typename CtaMma>
    __device__ static auto make_accumulator_fragment(CtaMma& cta_mma) {
        // Shape-only proxy: make_fragment_C only reads the shape.
        return cta_mma.make_fragment_C(
            cta_mma.partition_C(cute::make_identity_tensor(cute::make_shape(
                cute::Int<UpGemm::kTileShapeM>{}, cute::_8{}))));
    }

    // Read this lane's fragment: values[0..1] are gate, values[2..3] are up.
    __device__ static __forceinline__ void load_accumulator_fragment(
        std::uint32_t tmem_base, std::uint32_t acc_col, std::uint32_t group,
        float (&values)[4]) {
        tmem_ld_16x256b1x(
            tmem_addr(tmem_base, up_lane_base(WarpRole::get_warp_idx()),
                      acc_col + group * 8u),
            values);
    }

    // The caller owns the ring slot and arms its transaction barrier.
    __device__ static void load_weight_tile(
        CUtensorMap const& desc, WeightElement* weight, std::uint64_t* barrier,
        std::uint32_t row, std::uint32_t k_tile,
        std::uint64_t cache_hint = static_cast<std::uint64_t>(
            cute::TMA::CacheHintSm90::EVICT_NORMAL)) {
#pragma unroll 1
        for (std::uint32_t h = 0; h < MmaCore::kNumSwizzleBoxKs; ++h) {
            tma_load_2d(
                desc, k_tile * UpGemm::kTileShapeK + h * MmaCore::kSwizzleBoxK,
                row, weight + h * UpGemm::kTileShapeM * MmaCore::kSwizzleBoxK,
                barrier, cache_hint);
        }
    }

    __device__ static void load_weight_scale_tile(
        const typename UpGemm::ElementScale* scale, std::uint32_t row_atom,
        std::uint32_t k_tile, std::uint8_t* dest, std::uint64_t* barrier) {
        static_assert(MmaCore::kUseBlockscaled,
                      "scale tiles are only loaded for blockscaled MMA");
        constexpr std::uint32_t kSfAtomBytes = 512u;
        constexpr std::uint32_t kSfKBlocksPerRow = UpGemm::kGemmK / 128u;
        const auto* sfa = reinterpret_cast<const std::uint8_t*>(scale) +
                          (row_atom * kSfKBlocksPerRow +
                           k_tile * (UpGemm::kTileShapeK / 128u)) *
                              kSfAtomBytes;
        tma_load_1d(sfa, UpScaleTraits::kSfaSmemBytes, dest, barrier);
    }

    template <typename CtaMma, typename TensorC>
    __device__ static void mma(TiledMma& tiled_mma, CtaMma& cta_mma,
                               TensorC& tCt, WeightElement* weight,
                               ActElement* input, std::uint32_t tmem_base,
                               std::uint32_t acc_slot,
                               std::uint8_t* sfa = nullptr,
                               std::uint8_t* sfb = nullptr,
                               bool accumulate = false) {
        auto tCsB = cute::make_tensor(cute::make_smem_ptr(input), SBLayout{});
        auto tCrB = cta_mma.make_fragment_B(tCsB);
        if constexpr (MmaCore::kUseBlockscaled) {
            if (cute::elect_one_sync()) {
                UpScaleTraits::copy_to_tmem(
                    sfa, sfb, tmem_addr(tmem_base, 0u, kUpSfaTmemOffset),
                    tmem_addr(tmem_base, 0u, kUpSfbTmemOffset));
            }
            MmaCore::run(tiled_mma, cta_mma, weight, tCrB, tCt,
                         tmem_addr(tmem_base, 0u, 0u),
                         tmem_addr(tmem_base, 0u, kUpSfaTmemOffset),
                         tmem_addr(tmem_base, 0u, kUpSfbTmemOffset),
                         accumulate);
        } else {
            MmaCore::template run<MmaCore::kNumQuantBlocksB>(
                tiled_mma, cta_mma, weight, tCrB, tCt,
                tmem_addr(tmem_base, 0u, acc_slot * kAccColumns));
        }
    }

    // Read the fixed accumulator after all K tiles have completed.
    __device__ static void load_accumulators(std::uint32_t tmem_base,
                                             Accumulator& gate,
                                             Accumulator& up) {
        static_assert(MmaCore::kUseBlockscaled,
                      "only blockscaled MMA accumulates all K tiles in TMEM");
        if (!owns_up_rows(WarpRole::get_warp_idx())) return;
#pragma unroll
        for (std::uint32_t group = 0; group < kTokenGroups; ++group) {
            float values[4];
            load_accumulator_fragment(tmem_base, 0u, group, values);
            cutlass::arch::fence_view_async_tmem_load();
#pragma unroll
            for (std::uint32_t i = 0; i < 2u; ++i) {
                gate[group * 2u + i] = values[i];
                up[group * 2u + i] = values[i + 2u];
            }
        }
    }

    // One activation scale group of a K tile, in gate/up pair layout.
    __device__ static void dequant(std::uint32_t tmem_base,
                                   std::uint32_t acc_col,
                                   const float* act_scale, float gate_scale,
                                   float up_scale, Accumulator& gate,
                                   Accumulator& up) {
        if constexpr (!MmaCore::kUseBlockscaled) {
            if (!owns_up_rows(WarpRole::get_warp_idx())) return;
            const std::uint32_t t0 = WarpRole::get_lane_idx() & 3u;
#pragma unroll
            for (std::uint32_t group = 0; group < kTokenGroups; ++group) {
                float values[4];
                load_accumulator_fragment(tmem_base, acc_col, group, values);
                cutlass::arch::fence_view_async_tmem_load();
#pragma unroll
                for (std::uint32_t i = 0; i < 2u; ++i) {
                    const float as = act_scale[group * 8u + 2u * t0 + i];
                    gate[group * 2u + i] += values[i] * gate_scale * as;
                    up[group * 2u + i] += values[i + 2u] * up_scale * as;
                }
            }
        }
    }

    __device__ static void quantize_output_row(
        const float (&act)[kOutputFeatureBlock][UpGemm::kTileShapeN + 1u],
        std::uint32_t token,
        typename OutputQuant::ElementType (&quantized)[kQuantGroups],
        typename OutputQuantizer::ScaleArray& scales) {
        typename OutputQuantizer::Values values;
#pragma unroll
        for (std::uint32_t i = 0; i < kQuantGroups; ++i) {
            values[i] = act[WarpRole::get_lane_idx() + i * 32u][token];
        }
        OutputQuantizer::quantize(values, quantized, scales);
    }

    __device__ static void store_up_partial_sums(
        float (*partial)[kOutputFeatureBlock][UpGemm::kTileShapeN],
        const Accumulator& gate, const Accumulator& up) {
        const std::uint32_t warp = WarpRole::get_warp_idx();
        if (!owns_up_rows(warp)) return;
        const std::uint32_t lane = WarpRole::get_lane_idx();
        const std::uint32_t feature = 8u * up_group_idx(warp) + (lane >> 2);
#pragma unroll
        for (std::uint32_t group = 0; group < kTokenGroups; ++group) {
#pragma unroll
            for (std::uint32_t i = 0; i < 2u; ++i) {
                const std::uint32_t token = group * 8u + 2u * (lane & 3u) + i;
                partial[0][feature][token] = gate[group * 2u + i];
                partial[1][feature][token] = up[group * 2u + i];
            }
        }
    }

    // Every CTA thread joins when partial sums are shared across splits.
    __device__ static void join_up_split_k(
        float (*partial)[kOutputFeatureBlock][UpGemm::kTileShapeN],
        const Accumulator& gate, const Accumulator& up) {
        if constexpr (kNumSplits > 1u) {
            if (WarpRole::is_calc_warp(WarpRole::get_warp_idx())) {
                store_up_partial_sums(partial, gate, up);
            }
            __syncthreads();
            cluster_arrive_and_wait();
        }
    }

    // Split zero's calc warps call after join_up_split_k. Without split-K,
    // partial may be null; null weights apply a unit weight to every token.
    __device__ static void combine_gated_activation(
        const float (*partial)[kOutputFeatureBlock][UpGemm::kTileShapeN],
        float (*act)[UpGemm::kTileShapeN + 1u], const Accumulator& gate,
        const Accumulator& up, Activation activation, float situ_beta,
        float situ_linear_beta, const Accumulator* weights = nullptr,
        std::uint32_t token_mask = 0xFFFFFFFFu) {
        if (activation == Activation::kSitu) {
            combine_gated_activation_impl(
                partial, act, gate, up, weights, token_mask,
                GatedActFunctor<true>{situ_beta, situ_linear_beta});
        } else {
            combine_gated_activation_impl(
                partial, act, gate, up, weights, token_mask,
                GatedActFunctor<false>{situ_beta, situ_linear_beta});
        }
    }

    template <typename ActFunctor>
    __device__ static void combine_gated_activation_impl(
        const float (*partial)[kOutputFeatureBlock][UpGemm::kTileShapeN],
        float (*act)[UpGemm::kTileShapeN + 1u], const Accumulator& acc_gate,
        const Accumulator& acc_up, const Accumulator* weights,
        std::uint32_t token_mask, const ActFunctor& act_functor) {
        const std::uint32_t warp = WarpRole::get_warp_idx();
        if (!owns_up_rows(warp)) return;
        const std::uint32_t lane = WarpRole::get_lane_idx();
        const std::uint32_t feature = 8u * up_group_idx(warp) + (lane >> 2);
        constexpr std::uint32_t kPlaneElems =
            kOutputFeatureBlock * UpGemm::kTileShapeN;
        std::uint32_t shares[kNumSplits > 1u ? kNumSplits - 1u : 1u];
        if constexpr (kNumSplits > 1u) {
            const std::uint32_t first_rank = blockIdx.x % Traits::kClusterSize;
#pragma unroll
            for (std::uint32_t r = 1u; r < kNumSplits; ++r) {
                shares[r - 1u] =
                    cluster_map_shared_rank(&partial[0][0][0], first_rank + r);
            }
        }
#pragma unroll
        for (std::uint32_t group = 0; group < kTokenGroups; ++group) {
#pragma unroll
            for (std::uint32_t i = 0; i < 2u; ++i) {
                const std::uint32_t token = group * 8u + 2u * (lane & 3u) + i;
                if (((token_mask >> token) & 1u) == 0u) continue;
                const float rw = weights ? (*weights)[group * 2u + i] : 1.0f;
                float gate;
                float up;
                if constexpr (kNumSplits > 1u) {
                    gate = partial[0][feature][token];
                    up = partial[1][feature][token];
                    const std::uint32_t at =
                        feature * UpGemm::kTileShapeN + token;
#pragma unroll
                    for (std::uint32_t n = 0; n < kNumSplits - 1u; ++n) {
                        gate += cluster_load_f32(shares[n] + 4u * at);
                        up += cluster_load_f32(shares[n] +
                                               4u * (kPlaneElems + at));
                    }
                } else {
                    gate = acc_gate[group * 2u + i];
                    up = acc_up[group * 2u + i];
                }
                act[feature][token] = act_functor(gate, up, rw);
            }
        }
    }
};

// Shared down-projection computation; callers select buffers, layouts and
// accumulator/scale slots, and synchronize their producers and consumers.
template <typename KernelTraits, typename GemmTraits>
struct DownProjectionBase {
    using Traits = KernelTraits;
    using Dims = typename Traits::MoEDims;
    using SharedStorage = MoESharedStorage<Dims>;
    using Workspace = MoEWorkspace<Dims>;
    using DownGemm = GemmTraits;
    using MmaCore = typename DownGemm::MmaCore;
    using TiledMma = typename DownGemm::TiledMma;
    using DownScaleTraits = typename DownGemm::ScaleTraits;
    using WeightElement = typename MmaCore::SmemElementA;
    using ActElement = typename DownGemm::InputQuant::ElementType;
    using Accumulator = cutlass::Array<float, DownGemm::kTileShapeN>;

    static constexpr std::uint32_t kTileShapeM = DownGemm::kTileShapeM;
    static constexpr std::uint32_t kNumTilesK = DownGemm::kNumTilesK;
    static constexpr std::uint32_t kNumWeightSlots = DownGemm::kNumWeightSlots;
    static constexpr std::uint32_t kAccColumns =
        get_gemm_acc_columns<DownGemm>();
    static constexpr std::uint32_t kTokenGroups = DownGemm::kTileShapeN / 8u;
    static constexpr std::uint32_t kDownWeightTxBytes =
        MmaCore::kWeightTileTxBytes +
        (MmaCore::kUseBlockscaled ? DownScaleTraits::kSfaSmemBytes : 0u);

    template <typename CtaMma>
    __device__ static auto make_accumulator_fragment(CtaMma& cta_mma) {
        // Shape-only proxy: make_fragment_C only reads the shape.
        return cta_mma.make_fragment_C(
            cta_mma.partition_C(cute::make_identity_tensor(cute::make_shape(
                cute::Int<MmaCore::kAtomShapeM>{}, cute::_8{}))));
    }

    // RowsPerBox must match the caller's tensor-map box height.
    template <std::uint32_t RowsPerBox = MmaCore::kAtomShapeM>
    __device__ static void load_weight_tile(
        CUtensorMap const& desc, WeightElement* weight, std::uint64_t* barrier,
        std::uint32_t row, std::uint32_t k_tile,
        std::uint64_t cache_hint = static_cast<std::uint64_t>(
            cute::TMA::CacheHintSm90::EVICT_NORMAL)) {
        static_assert(kTileShapeM % RowsPerBox == 0u,
                      "the weight tile must contain whole tensor-map boxes");
#pragma unroll
        for (std::uint32_t box = 0; box < kTileShapeM / RowsPerBox; ++box) {
            tma_load_2d(desc, k_tile * DownGemm::kTileShapeK,
                        row + box * RowsPerBox,
                        weight + box * RowsPerBox * DownGemm::kTileShapeK,
                        barrier, cache_hint);
        }
    }

    __device__ static void load_weight_scale_tile(
        const typename DownGemm::ElementScale* scale, std::uint32_t row_atom,
        std::uint32_t k_tile, std::uint8_t* dest, std::uint64_t* barrier) {
        static_assert(MmaCore::kUseBlockscaled,
                      "scale tiles are only loaded for blockscaled MMA");
        static_assert(DownGemm::kTileShapeK == 128u,
                      "each down K tile loads one SfAtom per row atom");
        constexpr std::uint32_t kSfAtomBytes = DownScaleTraits::kSfaAtomBytes;
        constexpr std::uint32_t kSfKBlocksPerRow = DownGemm::kGemmK / 128u;
#pragma unroll
        for (std::uint32_t atom = 0; atom < DownScaleTraits::kNumSfaAtoms;
             ++atom) {
            const auto* sfa =
                reinterpret_cast<const std::uint8_t*>(scale) +
                ((row_atom + atom) * kSfKBlocksPerRow + k_tile) * kSfAtomBytes;
            tma_load_1d(sfa, kSfAtomBytes, dest + atom * kSfAtomBytes, barrier);
        }
    }

    // Called by the issuer warp after the operand barriers complete.
    __device__ static void copy_scales_to_tmem(std::uint8_t* sfa,
                                               std::uint8_t* sfb,
                                               std::uint32_t sfa_tmem,
                                               std::uint32_t sfb_tmem) {
        static_assert(MmaCore::kUseBlockscaled,
                      "only blockscaled MMA stages scales in TMEM");
        if (!cute::elect_one_sync()) return;
#pragma unroll
        for (std::uint32_t atom = 0; atom < DownScaleTraits::kNumSfaAtoms;
             ++atom) {
            DownScaleTraits::copy_sfa_to_tmem(
                sfa + atom * DownScaleTraits::kSfaAtomBytes,
                sfa_tmem + atom * DownScaleTraits::kSfaAtomColumns);
        }
        DownScaleTraits::copy_sfb_to_tmem(sfb, sfb_tmem);
    }

    template <typename CtaMma, typename InputLayout, typename TensorC>
    __device__ static void mma(TiledMma& tiled_mma, CtaMma& cta_mma,
                               InputLayout const& input_layout, TensorC& tCt,
                               WeightElement* weight, ActElement* input,
                               std::uint32_t acc_tmem,
                               std::uint32_t sfa_tmem = 0u,
                               std::uint32_t sfb_tmem = 0u,
                               bool accumulate = false) {
        auto tCsB = cute::make_tensor(cute::make_smem_ptr(input), input_layout);
        auto tCrB = cta_mma.make_fragment_B(tCsB);
        if constexpr (MmaCore::kUseBlockscaled) {
            MmaCore::run(tiled_mma, cta_mma, weight, tCrB, tCt, acc_tmem,
                         sfa_tmem, sfb_tmem, accumulate);
        } else {
            MmaCore::template run<MmaCore::kNumQuantBlocksB>(
                tiled_mma, cta_mma, weight, tCrB, tCt, acc_tmem);
        }
    }

    // A lane owns one output row and all of the tile's token columns.
    template <std::size_t NumSubBlocks>
    __device__ static void load_accumulator_fragment(
        std::uint32_t tmem_base, std::uint32_t acc_col, std::uint32_t atom,
        std::uint32_t lane_base,
        float (&values)[NumSubBlocks][DownGemm::kTileShapeN]) {
#pragma unroll
        for (std::uint32_t sub = 0; sub < NumSubBlocks; ++sub) {
            tmem_ld_tokens(tmem_addr(tmem_base, lane_base,
                                     acc_col + (atom * NumSubBlocks + sub) *
                                                   DownGemm::kTileShapeN),
                           values[sub]);
        }
        cutlass::arch::fence_view_async_tmem_load();
    }

    // A lane owns two rows and two tokens in a 16-row x 8-token window.
    __device__ static void load_accumulator_fragment(std::uint32_t tmem_base,
                                                     std::uint32_t acc_col,
                                                     std::uint32_t lane_base,
                                                     float (&values)[4]) {
        tmem_ld_16x256b1x(tmem_addr(tmem_base, lane_base, acc_col), values);
        cutlass::arch::fence_view_async_tmem_load();
    }

    // Scale a full row: combine its activation blocks before the weight scale.
    __device__ static void dequant(
        const float (&values)[MmaCore::kNumQuantBlocksB][DownGemm::kTileShapeN],
        const float (*act_scale)[DownGemm::kTileShapeN], float weight_scale,
        Accumulator& acc) {
        static_assert(!MmaCore::kUseBlockscaled,
                      "only non-blockscaled MMA needs software dequantization");
#pragma unroll
        for (std::uint32_t token = 0; token < DownGemm::kTileShapeN; ++token) {
            float value = 0.0f;
#pragma unroll
            for (std::uint32_t sub = 0; sub < MmaCore::kNumQuantBlocksB;
                 ++sub) {
                value += values[sub][token] * act_scale[sub][token];
            }
            acc[token] += value * weight_scale;
        }
    }

    // Scale one two-row/two-token fragment of an activation block.
    __device__ static void dequant(const float (&values)[4],
                                   const float* act_scale, float weight_scale,
                                   float (&acc)[4]) {
        static_assert(!MmaCore::kUseBlockscaled,
                      "only non-blockscaled MMA needs software dequantization");
#pragma unroll
        for (std::uint32_t i = 0; i < 2u; ++i) {
            const float as = act_scale[i];
            acc[i] += values[i] * weight_scale * as;
            acc[i + 2u] += values[i + 2u] * weight_scale * as;
        }
    }
};

}  // namespace halops::moe
