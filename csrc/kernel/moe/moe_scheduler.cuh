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

// A task is a `TaskInfo` naming the (expert, m block) it covers.

#include <cstdint>

#include "moe_shared_storage.cuh"
#include "moe_traits.cuh"
#include "moe_workspace.cuh"

namespace halops::moe {

// Which projection the scheduler is enumerating.
enum class BlockPhase : std::uint32_t { kNone = 0u, kUp = 1u, kDown = 2u };

// No further expert for this block.
inline constexpr std::uint32_t kNoExpert = 0xffffffffu;

// One scheduled task: the unit of work a block consumes in one loop iteration.
struct TaskInfo {
    std::uint32_t local_expert_idx;
    std::uint32_t m_block_idx;
    std::uint32_t next_expert_idx;
    std::uint32_t task_id;
};

template <std::uint32_t kCtasPerExpert>
struct CtaGroup {
    std::uint32_t group_idx;
    std::uint32_t cta_idx_in_group;
    std::uint32_t num_groups;

    __device__ CtaGroup()
        : group_idx(blockIdx.x / kCtasPerExpert),
          cta_idx_in_group(blockIdx.x % kCtasPerExpert),
          num_groups(gridDim.x / kCtasPerExpert) {}

    __device__ bool in_whole_group() const { return group_idx < num_groups; }
};

template <typename KernelTraits>
class TaskScheduler {
  public:
    using Traits = KernelTraits;
    using Dims = typename Traits::MoEDims;
    using SharedStorage = MoESharedStorage<Dims>;
    using Workspace = MoEWorkspace<Dims>;

    static constexpr std::uint32_t kCtasPerExpert = Traits::kCtasPerExpert;
    using Group = CtaGroup<kCtasPerExpert>;
    // `down_grid_task` entry that ends a block's down task sequence.
    static constexpr std::uint32_t kNoTask = SharedStorage::kNoDownTask;

    __device__ TaskScheduler(SharedStorage* smem, Workspace* workspace)
        : smem_(smem), workspace_(workspace) {}

    // Open the sequence for `phase`; `expert_count` is a runtime upper bound.
    __device__ void begin_phase(BlockPhase phase, std::uint32_t expert_count) {
        phase_ = phase;
        expert_count_ = expert_count;
        task_count_ = 0u;
        if (phase == BlockPhase::kUp) next_ = first_up_expert();
    }

    // Down phase, TMA launcher thread only: claim the next grid task and
    // publish it to the block ahead of the `get_next_task` that consumes it.
    __device__ void claim_next_task() {
        const std::uint32_t num_tasks = expert_count_ * kCtasPerExpert;
        const std::uint32_t num_static_tasks =
            cta_group_.num_groups * kCtasPerExpert;
        std::uint32_t grid_task = kNoTask;
        if (task_count_ == 0u && cta_group_.in_whole_group()) {
            const std::uint32_t static_task =
                cta_group_.group_idx * kCtasPerExpert +
                cta_group_.cta_idx_in_group;
            if (static_task < num_tasks) grid_task = static_task;
        } else if (num_tasks > num_static_tasks) {
            const std::uint32_t claimed =
                workspace_->handoff.claim_down_task(smem_->launch_parity);
            if (claimed < num_tasks - num_static_tasks)
                grid_task = num_static_tasks + claimed;
        }
        smem_->down_grid_task[task_count_] =
            static_cast<std::uint16_t>(grid_task);
        shared_store_release_u32(&smem_->down_tasks_claimed, task_count_ + 1u);
    }

    // The next task of this block, or false once its sequence is exhausted.
    __device__ bool get_next_task(TaskInfo& task) {
        if (phase_ == BlockPhase::kDown) return get_next_down_task(task);
        if (next_ >= expert_count_) return false;
        const std::uint32_t after = next_ + cta_group_.num_groups;
        task.local_expert_idx = next_;
        task.m_block_idx = cta_group_.cta_idx_in_group;
        task.next_expert_idx = after < expert_count_ ? after : kNoExpert;
        task.task_id = task_count_++;
        next_ = after;
        return true;
    }

    // The id the next `get_next_task` will hand out.
    __device__ std::uint32_t next_task_id() const { return task_count_; }

    // Up phase only: set before the loop so the caller can prime the first
    // expert's work.
    __device__ bool is_active() const { return next_ < expert_count_; }
    __device__ std::uint32_t first_expert_idx() const { return next_; }
    __device__ Group const& cta_group() const { return cta_group_; }

    // This block's up task count.
    __device__ std::uint32_t up_task_count() const {
        const std::uint32_t first = first_up_expert();
        const std::uint32_t stride = cta_group_.num_groups;
        return (first >= expert_count_)
                   ? 0u
                   : (expert_count_ - first + stride - 1u) / stride;
    }

  private:
    // Group g owns up experts g, g + num_groups, ...; others get nothing.
    __device__ std::uint32_t first_up_expert() const {
        return cta_group_.in_whole_group() ? cta_group_.group_idx
                                           : expert_count_;
    }

    __device__ bool get_next_down_task(TaskInfo& task) {
        while (shared_load_acquire_u32(&smem_->down_tasks_claimed) <=
               task_count_) {
        }
        const std::uint32_t grid_task = smem_->down_grid_task[task_count_];
        if (grid_task == kNoTask) return false;
        task.local_expert_idx = grid_task / kCtasPerExpert;
        task.m_block_idx = grid_task % kCtasPerExpert;
        task.next_expert_idx = kNoExpert;
        task.task_id = task_count_++;
        return true;
    }

    Group cta_group_;
    SharedStorage* smem_ = nullptr;
    Workspace* workspace_ = nullptr;
    BlockPhase phase_ = BlockPhase::kNone;
    std::uint32_t next_ = 0u;
    std::uint32_t expert_count_ = 0u;
    std::uint32_t task_count_ = 0u;
};

}  // namespace halops::moe
