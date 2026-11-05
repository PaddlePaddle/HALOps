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
#include <cstdint>

#include "moe_configs.cuh"
#include "moe_scheduler.cuh"
#include "moe_shared_storage.cuh"
#include "moe_tail.cuh"
#include "moe_traits.cuh"
#include "moe_workspace.cuh"

namespace halops::moe {
template <typename KernelTraits>
struct TailPhase {
    using Traits = KernelTraits;
    using Dims = typename Traits::MoEDims;
    using SharedStorage = MoESharedStorage<Dims>;
    using Workspace = MoEWorkspace<Dims>;
    using Geometry = TailGeometryT<Dims>;
    using Args = TailParams<Dims>;
    using Group = CtaGroup<Traits::kCtasPerExpert>;

    __device__ static void prefetch_weights(Args const& tail) {
        constexpr std::uint32_t kGsize = Geometry::kGsize;
        constexpr std::uint32_t kShard = Dims::kHiddenSize / Geometry::kWorld;
        constexpr std::uint32_t kRows = kShard / kGsize;
        constexpr std::uint32_t kBytes = kRows * Dims::kLatentSize;
        constexpr std::uint32_t kChunk = Geometry::kPrefetchChunk;
        constexpr std::uint32_t kScale = Geometry::kScaleBlock;
        if (blockIdx.x >= kGsize) return;
        if (threadIdx.x != WarpRole::kPublishWarpId * WarpRole::kThreadsPerWarp)
            return;
        const std::uint64_t row0 =
            static_cast<std::uint64_t>(tail.rank) * kShard +
            static_cast<std::uint64_t>(blockIdx.x) * kRows;
        const std::uint8_t* w = tail.fc2_w8 + row0 * Dims::kLatentSize;
#pragma unroll
        for (std::uint32_t off = 0; off < kBytes; off += kChunk) {
            prefetch_l2_bulk(w + off,
                             (kBytes - off) < kChunk ? (kBytes - off) : kChunk);
        }
        prefetch_l2_bulk(
            tail.fc2_scale + (row0 / kScale) * (Dims::kLatentSize / kScale),
            (Dims::kLatentSize / kScale) * sizeof(float));
    }

    __device__ static __forceinline__ void push_cell(
        Args const& tail, std::uint32_t tok, std::uint32_t col,
        DataType::ElementOut b, std::uint32_t tail_parity) {
        const std::uint32_t lane = threadIdx.x & 31u;
        const std::uint32_t grp = lane & ~7u;
        const std::uint32_t raw = *reinterpret_cast<const std::uint16_t*>(&b);
        std::uint32_t w[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const std::uint32_t lo = __shfl_sync(0xffffffffu, raw, grp + 2 * j);
            const std::uint32_t hi =
                __shfl_sync(0xffffffffu, raw, grp + 2 * j + 1);
            w[j] = lo | (hi << 16);
        }
        if ((lane & 7u) == 0u) {
            const uint4 u = make_uint4(w[0], w[1], w[2], w[3]);
            ::halops::moe::tail::push_only<Geometry::kWorld, Dims::kLatentSize,
                                           Dims::kHiddenSize, Geometry::kMaxM>(
                tail, *reinterpret_cast<const ::halops::moe::tail::Vec16*>(&u),
                /*round=*/0, static_cast<int>(tail_parity & 1u),
                static_cast<int>(tok), Dims::kLatentSize, col >> 3);
        }
    }

    // AllReduce + RMSNorm + fc2 + shared-combine
    __device__ static void run(Args const& tail, std::uint32_t shared_ctas,
                               Workspace* __restrict__ workspace,
                               SharedStorage* __restrict__ smem,
                               Group const& cta_group) {
        if (blockIdx.x >= Geometry::kNumCtas) return;
        if (!WarpRole::is_calc_warp(WarpRole::get_warp_idx())) return;
        static_assert(
            TailSmemLayout<Dims>::kBytes <= sizeof(RoutedExpertsSmem<Dims>),
            "the tail scratch must fit the weight-ring union");
        if (shared_ctas != 0u) {
            if (threadIdx.x == 0) {
                workspace->cta_comm.wait_shared_down_done(smem->launch_parity,
                                                          shared_ctas);
            }
            WarpRole::sync_calc_threads();
            __threadfence();
        }
#if defined(HALOPS_KERNEL_TIMER_ENABLED)
        const auto timer = smem->timer();
#else
        const int timer = 0;  // the probes expand to nothing
#endif
        ::halops::moe::tail::eb5_tail_group_body<
            Geometry::kWorld, Dims::kLatentSize, Dims::kHiddenSize,
            Geometry::kMaxM, Geometry>(
            tail, blockIdx.x, threadIdx.x,
            reinterpret_cast<std::uint8_t*>(&smem->routed_experts),
            CalcWarpSync{}, smem->tail_parity, timer);
        WarpRole::sync_calc_threads();
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            __threadfence();
            atomicAdd(tail.epochs, 1u);
        }
    }
};

}  // namespace halops::moe
