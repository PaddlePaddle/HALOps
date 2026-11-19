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

#include "kernel/moe/moe_configs.cuh"
#include "kernel/moe/ptx_utils.cuh"

namespace halops::moe {

// Split-K groups inside a launch cluster.
struct ClusterSplitK {
    __device__ ClusterSplitK(std::uint32_t cluster_size,
                             std::uint32_t num_splits)
        : cluster_size_(cluster_size), num_splits_(num_splits) {}

    // This CTA's rank inside its launch cluster.
    __device__ static std::uint32_t rank(std::uint32_t cluster_size) {
        return static_cast<std::uint32_t>(blockIdx.x) % cluster_size;
    }

    __device__ std::uint32_t split_idx() const {
        return rank(cluster_size_) % num_splits_;
    }
    __device__ std::uint32_t group_base() const {
        return rank(cluster_size_) - split_idx();
    }

    __device__ static void sync() { cluster_arrive_and_wait(); }
    __device__ static void arrive() { cluster_arrive(); }
    __device__ static void wait() { cluster_wait(); }

    __device__ void map(const void* local) {
        const std::uint32_t base = group_base();

        // Unrolled to keep `peers_` stays in registers.
#pragma unroll
        for (std::uint32_t r = 0; r < MoERuntime::kMaxClusterSize; ++r) {
            if (r < num_splits_) {
                peers_[r] = cluster_map_shared_rank(local, base + r);
            }
        }
    }

    // The f32 at `index` summed over every split of the group.
    __device__ float split_sum(std::uint32_t index) const {
        float v = 0.0f;
#pragma unroll
        for (std::uint32_t r = 0; r < MoERuntime::kMaxClusterSize; ++r) {
            if (r < num_splits_) {
                v += cluster_load_f32(peers_[r] + 4u * index);
            }
        }
        return v;
    }

  private:
    std::uint32_t cluster_size_;
    std::uint32_t num_splits_;
    std::uint32_t peers_[MoERuntime::kMaxClusterSize];
};

}  // namespace halops::moe
