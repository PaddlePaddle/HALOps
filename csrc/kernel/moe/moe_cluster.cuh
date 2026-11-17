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
// GEMM.
template <std::uint32_t ClusterSize, std::uint32_t NumSplits>
struct ClusterSplitK {
    static_assert(NumSplits >= 1u, "a group needs one split or more");
    static_assert(ClusterSize % NumSplits == 0u,
                  "a split group is a whole run of the launch cluster");

    // This CTA's rank inside the launch cluster.
    __device__ static std::uint32_t rank() {
        return static_cast<std::uint32_t>(blockIdx.x) % ClusterSize;
    }

    __device__ static std::uint32_t split_idx() { return rank() % NumSplits; }
    __device__ static std::uint32_t group_base() {
        return rank() - split_idx();
    }

    // A cluster-wide barrier: release on arrive, acquire on wait.
    __device__ static void sync() { cluster_arrive_and_wait(); }
    __device__ static void arrive() { cluster_arrive(); }
    __device__ static void wait() { cluster_wait(); }

    __device__ ClusterSplitK() : group_base_(group_base()) {}

    // Point the split addresses at `local`'s copies in every split.
    __device__ void map(const void* local) {
#pragma unroll
        for (std::uint32_t r = 0; r < NumSplits; ++r) {
            peers_[r] = cluster_map_shared_rank(local, group_base_ + r);
        }
    }

    // The f32 at `index` summed over every split of the group.
    __device__ float split_sum(std::uint32_t index) const {
        float v = 0.0f;
#pragma unroll
        for (std::uint32_t r = 0; r < NumSplits; ++r) {
            v += cluster_load_f32(peers_[r] + 4u * index);
        }
        return v;
    }

  private:
    std::uint32_t group_base_;
    std::uint32_t peers_[NumSplits];
};

}  // namespace halops::moe
