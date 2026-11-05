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

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cutlass/cluster_launch.hpp>

#include "kernel/moe/moe_configs.cuh"
#include "kernel/moe/moe_mainloop.cuh"
#include "kernel/moe/moe_mainloop_router.cuh"
#include "kernel/moe/moe_routing.cuh"
#include "kernel/moe/moe_routing_topk.cuh"
#include "kernel/moe/moe_scheduler.cuh"
#include "kernel/moe/moe_shared_storage.cuh"
#include "kernel/moe/moe_tma.cuh"
#include "kernel/moe/moe_traits.cuh"
#include "kernel/moe/moe_workspace.cuh"

namespace halops::moe {

template <typename KernelTraits, typename CollectiveMainloop>
__global__ __launch_bounds__(KernelTraits::kThreads, 1) void fused_moe_kernel(
    __grid_constant__
    typename CollectiveMainloop::Params const mainloop_params) {
    using Dims = typename KernelTraits::MoEDims;
    using Traits = KernelTraits;

    static_assert(sizeof(MoESharedStorage<Dims>) <= 228 * 1024,
                  "MoESharedStorage<Dims> exceeds the 228 KB per-block SHM "
                  "budget.");
    static_assert(Dims::kNumTokens <= 16,
                  "fused_moe_kernel only supports kNumTokens<=16.");

    assert(blockDim.x == Dims::KernelConfig::kBlockSize);
    assert(blockDim.y == 1 && blockDim.z == 1);

    assert(gridDim.x >= MoEKernelTraits<Dims>::kMinGrid);
    assert(gridDim.x <= MoEKernelTraits<Dims>::kMaxGrid);
    assert(gridDim.y == 1 && gridDim.z == 1);

    assert(mainloop_params.num_tokens <= Dims::kNumTokens);
    assert(mainloop_params.top_k >= 1 &&
           mainloop_params.top_k <= Dims::kMaxTopk);

    extern __shared__ char smem_buffer[];
    MoESharedStorage<Dims>* __restrict__ smem =
        reinterpret_cast<MoESharedStorage<Dims>*>(smem_buffer);

    using Traits = MoEKernelTraits<Dims>;

    CollectiveMainloop collective_mainloop;

    const std::uint32_t warp_idx = WarpRole::get_warp_idx();

    // Device event timer: park the per-CTA record pool in SMEM and zero the
    // per-warp write cursors before any probe.
#if defined(HALOPS_KERNEL_TIMER_ENABLED)
    if (threadIdx.x < halops::kTimerWarps) {
        smem->timer_next[threadIdx.x] = 0u;
    }
    if (threadIdx.x == 0u) {
        smem->timer_base = mainloop_params.timer_events;
    }
    __syncthreads();
#endif

    KERNEL_TIMER_CALIBRATE(smem->timer());
    const auto kernel_span =
        KERNEL_TIMER_BEGIN(smem->timer(), "fused_moe_kernel");

    using Workspace = MoEWorkspace<Dims>;
    Workspace* __restrict__ workspace =
        reinterpret_cast<Workspace*>(mainloop_params.scratchpad);

    // Initialize the launcher-thread states and publish to all warps.
    const auto init_span = KERNEL_TIMER_BEGIN(smem->timer(), "initialize");
    collective_mainloop.initialize(mainloop_params, smem);
    KERNEL_TIMER_END(smem->timer(), init_span);

    // Phase 1: prologue, including routing + latent_down + shared_expert,
    // overlapped on different CTAs.
    const auto prologue_span =
        KERNEL_TIMER_BEGIN(smem->timer(), "prologue_phase");
    collective_mainloop.run_latent_down(mainloop_params, smem);
    collective_mainloop.run_shared(mainloop_params, smem);
    collective_mainloop.run_router(mainloop_params, smem);
    // Phase boundary: publishes the routing tables and orders the router's
    // SMEM traffic before the next phase's TMA writes into those bytes.
    __syncthreads();
    if (WarpRole::is_weight_loader_thread()) {
        CollectiveMainloop::Router::release_weight_smem();
    }
    KERNEL_TIMER_END(smem->timer(), prologue_span);

    // Scheduler per thread drives both projections; `begin_phase` opens
    // the task sequence of the projection about to run.
    TaskScheduler<Traits> sched(smem, workspace);

    // Phase 2: up-projection — expert groups in parallel. The scheduler maps
    // this block to (expert group, feature block) statically.
    sched.begin_phase(BlockPhase::kUp, smem->routing.active_expert_count);
    const auto up_span = KERNEL_TIMER_BEGIN(smem->timer(), "routed_up_phase",
                                            smem->routing.active_expert_count);
    if (sched.is_active()) {
        collective_mainloop.run_up(mainloop_params, smem, sched);
    }
    KERNEL_TIMER_END(smem->timer(), up_span);

    // Phase 3: down-projection over (expert, m-block) tasks, each published
    // to its m-block's completion count.
    sched.begin_phase(BlockPhase::kDown, smem->routing.active_expert_count);
    const auto down_span = KERNEL_TIMER_BEGIN(
        smem->timer(), "routed_down_phase", smem->routing.active_expert_count);
    collective_mainloop.run_down(mainloop_params, smem, sched);
    KERNEL_TIMER_END(smem->timer(), down_span);

    // Phase 4: gather and combine each token's top_k combine slots.
    const auto combine_span =
        KERNEL_TIMER_BEGIN(smem->timer(), "combine_phase");
    collective_mainloop.combine(mainloop_params, smem, sched.cta_group());
    KERNEL_TIMER_END(smem->timer(), combine_span);

    __syncthreads();
    if (WarpRole::is_tmem_warp(warp_idx)) {
        tmem_free(smem->tmem_ptr, MoEKernelTraits<Dims>::kTmemColumns);
    }
    KERNEL_TIMER_END(smem->timer(), kernel_span);
}

template <typename MoEDims>
cudaError_t LaunchMoEKernel(
    typename CollectiveMainloopMoE<MoEKernelTraits<MoEDims>>::Arguments& args,
    std::uint32_t grid_size, cudaStream_t stream) {
    using Dims = MoEDims;
    using Traits = MoEKernelTraits<Dims>;
    using Mainloop = CollectiveMainloopMoE<Traits>;

    constexpr size_t smem_size = get_moe_smem_size<Dims>();
    const typename Mainloop::Params params =
        Mainloop::to_underlying_arguments(args);

    auto* kernel_ptr = &moe::fused_moe_kernel<Traits, Mainloop>;
    if (smem_size >= 48 * 1024) {
        const cudaError_t attribute = cudaFuncSetAttribute(
            kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem_size));
        if (attribute != cudaSuccess) return attribute;
    }
    constexpr std::uint32_t kClusterSize = Traits::kClusterSize;
    const cutlass::ClusterLaunchParams launch_params{
        dim3(grid_size, 1, 1), dim3(Traits::kThreads, 1, 1),
        dim3(kClusterSize, 1, 1), static_cast<int>(smem_size), stream};
    const cutlass::Status launch_status = cutlass::launch_kernel_on_cluster(
        launch_params, reinterpret_cast<void const*>(kernel_ptr), params);
    if (launch_status != cutlass::Status::kSuccess) {
        return cudaErrorLaunchFailure;
    }
    return cudaGetLastError();
}

}  // namespace halops::moe
