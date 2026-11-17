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

#include "kernel/moe/ptx_utils.cuh"

namespace halops::moe {

// Split-K groups inside a launch cluster: `NumSplits` CTAs that reduce one
// GEMM. The cluster can be wider than one group (the latent projection runs
// pairs inside a 4-wide cluster), so callers name their group by its first
// cluster rank.
template <std::uint32_t ClusterSize>
struct ClusterSplitK {
    // This CTA's rank inside the launch cluster.
    __device__ static std::uint32_t rank() {
        return static_cast<std::uint32_t>(blockIdx.x) % ClusterSize;
    }

    // A cluster-wide barrier: release on arrive, acquire on wait.
    __device__ static void sync() { cluster_arrive_and_wait(); }
    __device__ static void arrive() { cluster_arrive(); }
    __device__ static void wait() { cluster_wait(); }

    // The other splits' copies of `local`, skipping `split`.
    template <std::uint32_t NumSplits>
    __device__ static void map_peers(const void* local,
                                     std::uint32_t group_base,
                                     std::uint32_t split,
                                     std::uint32_t (&peers)[NumSplits - 1u]) {
        static_assert(NumSplits >= 2u, "a reduction needs two splits or more");
#pragma unroll
        for (std::uint32_t n = 0; n < NumSplits - 1u; ++n) {
            const std::uint32_t r = n + (n >= split ? 1u : 0u);
            peers[n] = cluster_map_shared_rank(local, group_base + r);
        }
    }

    // `local` plus the peer copies' f32 at `index`.
    template <std::uint32_t NumSplits>
    __device__ static float peer_sum(
        const std::uint32_t (&peers)[NumSplits - 1u], float local,
        std::uint32_t index) {
        float v = local;
#pragma unroll
        for (std::uint32_t n = 0; n < NumSplits - 1u; ++n) {
            v += cluster_load_f32(peers[n] + 4u * index);
        }
        return v;
    }

    // The max of the f32 at `index` across every split, this one included.
    template <std::uint32_t NumSplits>
    __device__ static float split_max(const void* local,
                                      std::uint32_t group_base,
                                      std::uint32_t index) {
        static_assert(NumSplits >= 1u);
        float v = cluster_load_f32(cluster_map_shared_rank(local, group_base) +
                                   4u * index);
#pragma unroll
        for (std::uint32_t r = 1; r < NumSplits; ++r) {
            v = fmaxf(v, cluster_load_f32(
                             cluster_map_shared_rank(local, group_base + r) +
                             4u * index));
        }
        return v;
    }
};

}  // namespace halops::moe
