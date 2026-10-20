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

#include "cutlass/arch/reg_reconfig.h"
#include "kernel/attention/kernel_traits.cuh"
#include "kernel/attention/mainloop_fwd.cuh"
#include "kernel/attention/tile_scheduler.cuh"

namespace halops {

template <typename KernelTraits, typename CollectiveMainloop>
__global__ void __launch_bounds__(KernelTraits::kThreads, 1)
    compute_attn_ws_encoder(
        CUTE_GRID_CONSTANT
        typename CollectiveMainloop::Params const mainloop_params) {
    using MainloopPipeline = typename KernelTraits::MainloopPipeline;
    using PipelineParams = typename MainloopPipeline::Params;
    using PipelineState = typename MainloopPipeline::PipelineState;
    using TmemAllocator = cute::TMEM::Allocator1Sm;
    using ClusterShape = typename KernelTraits::ClusterShape;

    constexpr int kNumStagesQ = KernelTraits::kNumStagesQ;
    constexpr int kNumThreadsPerWarp = cutlass::NumThreadsPerWarp;
    constexpr bool CustomMask = KernelTraits::CustomMask;
    constexpr int kBlockM = KernelTraits::kBlockM;

    __align__(16) __shared__ int mask[kBlockM * kNumStagesQ];

    const int m_block = gridDim.x - 1 - blockIdx.x;
    const int bidh = blockIdx.y;
    const int bidb = blockIdx.z;
    const int thread_idx = threadIdx.x;
    const int warp_idx = cutlass::canonical_warp_idx_sync();
    const int lane_idx = thread_idx % 32;
    int n_block_max;

    const int seqlen_q =
        mainloop_params.cu_seq_q[bidb + 1] - mainloop_params.cu_seq_q[bidb];
    const int seqlen_k =
        mainloop_params.cu_seq_k[bidb + 1] - mainloop_params.cu_seq_k[bidb];

    if (m_block * (kBlockM * kNumStagesQ) >= seqlen_q) {
        return;
    }

    extern __shared__ char shared_memory[];
    auto &shared_storage =
        *reinterpret_cast<typename KernelTraits::SharedStorage *>(
            shared_memory);

    if (warp_idx == 0) {
        CollectiveMainloop::prefetch_tma_descriptors(mainloop_params);
    } else if (warp_idx == 1) {
        for (int i = 0; i < kNumStagesQ; ++i) {
            cute::initialize_barrier(shared_storage.mbar_load_q_full[i], 1);
        }
    } else if (warp_idx == 2) {
        for (int i = 0; i < kNumStagesQ; ++i) {
            cute::initialize_barrier(shared_storage.mbar_softmax_corr_empty[i],
                                     4 * kNumThreadsPerWarp);
            cute::initialize_barrier(shared_storage.mbar_softmax_corr_full[i],
                                     4 * kNumThreadsPerWarp);
        }
    } else if (warp_idx == 3) {
    } else if (warp_idx == 4) {
        for (int i = 0; i < kNumStagesQ; ++i) {
            cute::initialize_barrier(shared_storage.mbar_corr_epi_full[i],
                                     4 * kNumThreadsPerWarp);
        }
    } else if (warp_idx == 5) {
        for (int i = 0; i < kNumStagesQ; ++i) {
            cute::initialize_barrier(shared_storage.mbar_P_full_O_rescaled[i],
                                     8 * kNumThreadsPerWarp);
            cute::initialize_barrier(shared_storage.mbar_S_full[i], 1);
            cute::initialize_barrier(shared_storage.mbar_O_full[i], 1);
        }
    } else if (warp_idx == 6) {
        for (int i = 0; i < kNumStagesQ; ++i) {
            cute::initialize_barrier(shared_storage.mbar_P_full_2[i],
                                     4 * kNumThreadsPerWarp);
        }
    } else if (warp_idx == 7) {
        cute::initialize_barrier(shared_storage.mbar_tmem_dealloc,
                                 (4 + kNumStagesQ * 4) * kNumThreadsPerWarp);
    }

    if constexpr (CustomMask) {
        static_assert(KernelTraits::kEmptyWarpStartIds *
                          cutlass::NumThreadsPerWarp >=
                      kBlockM * kNumStagesQ / 4);
        const int *mask_this_batch = mainloop_params.mask +
                                     mainloop_params.cu_seq_q[bidb] +
                                     m_block * (kBlockM * kNumStagesQ);
        if (thread_idx < kBlockM * kNumStagesQ &&
            (m_block * (kBlockM * kNumStagesQ) + thread_idx) < seqlen_q) {
            mask[thread_idx] = mask_this_batch[thread_idx];
        }
    }
    PipelineParams pipeline_params;
    pipeline_params.transaction_bytes =
        CollectiveMainloop::TmaTransactionBytesK;
    pipeline_params.role = warp_idx == KernelTraits::kLoadWarpIds
                               ? MainloopPipeline::ThreadCategory::Producer
                               : MainloopPipeline::ThreadCategory::Consumer;
    pipeline_params.is_leader = lane_idx == 0;
    pipeline_params.num_consumers = kNumThreadsPerWarp;

    MainloopPipeline pipeline_k(shared_storage.pipeline_k, pipeline_params,
                                ClusterShape{});
    MainloopPipeline pipeline_v(shared_storage.pipeline_v, pipeline_params,
                                ClusterShape{});
    __syncthreads();

    CollectiveMainloop collective_mainloop;

    if (warp_idx <= KernelTraits::kLoadWarpIds) {
        if constexpr (CustomMask) {
            const int mask_seq_len =
                mask[min(kBlockM * kNumStagesQ - 1,
                         seqlen_q - m_block * (kBlockM * kNumStagesQ) - 1)];
            n_block_max = (mask_seq_len + KernelTraits::kBlockN - 1) /
                          KernelTraits::kBlockN;
        } else {
            n_block_max = collective_mainloop.get_n_block_max(
                seqlen_q, seqlen_k, m_block);
        }
        if (n_block_max == 0) return;
    }

    if (warp_idx >= KernelTraits::kEmptyWarpStartIds) {
        cutlass::arch::warpgroup_reg_dealloc<KernelTraits::num_regs_empty>();
    } else if (warp_idx == KernelTraits::kLoadWarpIds) {
        cutlass::arch::warpgroup_reg_dealloc<KernelTraits::num_regs_other>();
        collective_mainloop.load(mainloop_params, pipeline_k, pipeline_v,
                                 shared_storage, lane_idx, n_block_max, m_block,
                                 seqlen_q, seqlen_k, bidh, bidb);
    } else if (warp_idx == KernelTraits::kMmaWarpIds) {
        cutlass::arch::warpgroup_reg_dealloc<KernelTraits::num_regs_other>();
        TmemAllocator tmem_allocator{};
        tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns,
                                &shared_storage.tmem_holding_buf);
        __syncwarp();

        collective_mainloop.mma(mainloop_params, pipeline_k, pipeline_v,
                                shared_storage, lane_idx, n_block_max);

        cute::wait_barrier(shared_storage.mbar_tmem_dealloc, 0);
        tmem_allocator.release_allocation_lock();
        tmem_allocator.free(shared_storage.tmem_holding_buf,
                            TmemAllocator::Sm100TmemCapacityColumns);
    } else if (warp_idx < KernelTraits::kCorrection0WarpStartIds) {
        cutlass::arch::warpgroup_reg_alloc<KernelTraits::num_regs_softmax>();
        collective_mainloop.softmax(mainloop_params, shared_storage, mask,
                                    lane_idx, warp_idx, thread_idx, n_block_max,
                                    m_block, seqlen_q, seqlen_k, bidb, bidh);
        cute::arrive_barrier(shared_storage.mbar_tmem_dealloc);
    } else if (warp_idx < KernelTraits::kMmaWarpIds) {
        cutlass::arch::warpgroup_reg_dealloc<
            KernelTraits::num_regs_correction>();
        collective_mainloop.correction(mainloop_params, shared_storage,
                                       lane_idx, warp_idx, thread_idx, bidb,
                                       bidh, n_block_max, m_block, seqlen_q);
        cute::arrive_barrier(shared_storage.mbar_tmem_dealloc);
    }
}

template <typename KernelTraits, typename CollectiveMainloop>
__global__ void __launch_bounds__(KernelTraits::kThreads, 1)
    compute_attn_ws_decoder(
        CUTE_GRID_CONSTANT
        typename CollectiveMainloop::Params const mainloop_params) {
    using MainloopPipeline = typename KernelTraits::MainloopPipeline;
    using PipelineParams = typename MainloopPipeline::Params;
    using PipelineState = typename MainloopPipeline::PipelineState;
    using TmemAllocator = cute::TMEM::Allocator1Sm;
    using ClusterShape = typename KernelTraits::ClusterShape;

    constexpr int kNumStagesQ = KernelTraits::kNumStagesQ;
    constexpr int kNumThreadsPerWarp = cutlass::NumThreadsPerWarp;
    constexpr int kBlockM = KernelTraits::kBlockM;
    constexpr int kBlockN = KernelTraits::kBlockN;

    // Non-persistent: directly use 3D grid indices
    const int n_block = blockIdx.x;
    const int bidh = blockIdx.y;
    const int bidb = blockIdx.z;

    const int thread_idx = threadIdx.x;
    const int warp_idx = cutlass::canonical_warp_idx_sync();
    const int lane_idx = thread_idx % kNumThreadsPerWarp;

    __align__(16) __shared__ int mask[kBlockM];

    extern __shared__ char shared_memory[];
    auto &shared_storage =
        *reinterpret_cast<typename KernelTraits::SharedStorage *>(
            shared_memory);

    const int seqlen_k = mainloop_params.cu_seq_k[bidb];

    if (warp_idx == 0) {
        CollectiveMainloop::prefetch_tma_descriptors(mainloop_params);
    } else if (warp_idx == 1) {
        for (int i = 0; i < kNumStagesQ; ++i) {
            cute::initialize_barrier(shared_storage.mbar_load_q_full[i], 1);
        }
    } else if (warp_idx == 2) {
        for (int i = 0; i < 2; ++i) {
            cute::initialize_barrier(shared_storage.mbar_P_full_O_rescaled[i],
                                     4 * kNumThreadsPerWarp);
            cute::initialize_barrier(shared_storage.mbar_S_full[i], 1);
            cute::initialize_barrier(shared_storage.mbar_O_full[i], 1);
            cute::initialize_barrier(shared_storage.mbar_pv_done[i], 1);
        }
    } else if (warp_idx == 3) {
        cute::initialize_barrier(shared_storage.mbar_tmem_dealloc,
                                 4 * kNumThreadsPerWarp);
    }

    PipelineParams pipeline_params;
    pipeline_params.transaction_bytes =
        CollectiveMainloop::TmaTransactionBytesK;
    pipeline_params.role = warp_idx == KernelTraits::kLoadWarpIds
                               ? MainloopPipeline::ThreadCategory::Producer
                               : MainloopPipeline::ThreadCategory::Consumer;
    pipeline_params.is_leader = lane_idx == 0;
    pipeline_params.num_consumers = kNumThreadsPerWarp;

    MainloopPipeline pipeline_k(shared_storage.pipeline_k, pipeline_params,
                                ClusterShape{});
    MainloopPipeline pipeline_v(shared_storage.pipeline_v, pipeline_params,
                                ClusterShape{});

    __syncthreads();

    CollectiveMainloop collective_mainloop;

    const int num_kv_blocks_per_cta =
        CollectiveMainloop::get_num_kv_blocks_per_cta(mainloop_params,
                                                      seqlen_k);
    if (n_block * kBlockN * num_kv_blocks_per_cta >= seqlen_k) {
        return;
    }

    if (warp_idx == KernelTraits::kLoadWarpIds) {
        cutlass::arch::warpgroup_reg_dealloc<KernelTraits::num_regs_other>();
        const int n_block_max = collective_mainloop.get_n_block_max(
            seqlen_k, n_block, num_kv_blocks_per_cta);

        collective_mainloop.load(mainloop_params, pipeline_k, pipeline_v,
                                 shared_storage, lane_idx, n_block_max, n_block,
                                 num_kv_blocks_per_cta, bidh, bidb);
    } else if (warp_idx == KernelTraits::kMmaWarpIds) {
        cutlass::arch::warpgroup_reg_dealloc<KernelTraits::num_regs_other>();
        TmemAllocator tmem_allocator{};
        tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns,
                                &shared_storage.tmem_holding_buf);
        __syncwarp();

        const int n_block_max = collective_mainloop.get_n_block_max(
            seqlen_k, n_block, num_kv_blocks_per_cta);

        collective_mainloop.mma(mainloop_params, pipeline_k, pipeline_v,
                                shared_storage, lane_idx, n_block_max, 0);

        cute::wait_barrier(shared_storage.mbar_tmem_dealloc, 0);
        tmem_allocator.release_allocation_lock();
        tmem_allocator.free(shared_storage.tmem_holding_buf,
                            TmemAllocator::Sm100TmemCapacityColumns);
    } else {
        cutlass::arch::warpgroup_reg_alloc<KernelTraits::num_regs_softmax>();
        // Prefilling samples in a mixed batch belong to the encoder kernel;
        // zero query rows makes this kernel skip them.
        const int num_q_rows =
            mainloop_params.cu_seq_q[bidb + 1] - mainloop_params.cu_seq_q[bidb];
        const int seqlen_q =
            is_decoder_batch(mainloop_params.seq_len_encoder,
                             mainloop_params.seq_len_decoder, bidb)
                ? num_q_rows
                : 0;
        const int n_block_max = collective_mainloop.get_n_block_max(
            seqlen_k, n_block, num_kv_blocks_per_cta);

        collective_mainloop.softmax(mainloop_params, shared_storage, mask,
                                    lane_idx, warp_idx, thread_idx, n_block_max,
                                    n_block, seqlen_q, seqlen_k,
                                    num_kv_blocks_per_cta, bidb, bidh, 0);
    }
}

template <bool CustomMask, typename InputType>
void LaunchEncoderAttention(AttentionParams &params, cudaStream_t stream) {
    using Element = InputType;
    constexpr int kBlockM = 128;
    constexpr int kBlockN = 128;
    constexpr int kHeadDim = 128;
    constexpr int kNumStagesQ = 2;
    constexpr bool kSplitKV = false;
    using KernelTraits =
        AttentionKernelTraits<kBlockM, kBlockN, kHeadDim, kNumStagesQ, kSplitKV,
                              /*MinKvBlocksPerPartition=*/1, CustomMask,
                              Element>;

    const int num_m_block =
        cute::ceil_div(params.q_token_num, kBlockM * KernelTraits::kNumStagesQ);
    using CollectiveMainloop = CollectiveMainloopFwdEncoder<KernelTraits>;

    typename CollectiveMainloop::Params mainloop_params =
        CollectiveMainloop::to_underlying_arguments(
            {static_cast<Element const *>(params.q_ptr),
             get_gmem_layout<kHeadDim>(params.q_token_num, params.head_num),
             static_cast<Element const *>(params.k_ptr),
             get_gmem_layout<kHeadDim>(params.k_token_num, params.kv_head_num),
             static_cast<Element const *>(params.v_ptr),
             get_gmem_layout<kHeadDim>(params.k_token_num, params.kv_head_num),
             static_cast<Element *>(params.o_ptr), params.cu_seq_q,
             params.cu_seq_k, params.mask, params.head_num * kHeadDim,
             params.scale_softmax_log2});

    void *kernel;
    kernel = (void *)compute_attn_ws_encoder<KernelTraits, CollectiveMainloop>;
    int smem_size = sizeof(typename KernelTraits::SharedStorage);

    if (smem_size >= 48 * 1024) {
        cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    }

    dim3 block_dims(KernelTraits::kThreads);
    dim3 grid_dims = {static_cast<unsigned int>(num_m_block),
                      static_cast<unsigned int>(params.head_num),
                      static_cast<unsigned int>(params.batch_size)};

    using ClusterShape = typename KernelTraits::ClusterShape;
    dim3 cluster_dims(cute::size<0>(ClusterShape{}),
                      cute::size<1>(ClusterShape{}),
                      cute::size<2>(ClusterShape{}));
    cutlass::ClusterLaunchParams launch_params{grid_dims, block_dims,
                                               cluster_dims, smem_size, stream};
    cutlass::launch_kernel_on_cluster(launch_params, kernel, mainloop_params);
}

// The grid is sized from the shared params.max_num_partitions budget; the
// kernel derives its own width from it per batch.
template <typename InputType, int kBlockM, bool SplitKV,
          int MinKvBlocksPerPartition>
void LaunchDecoderAttention(AttentionParams &params, cudaStream_t stream) {
    using Element = InputType;
    constexpr int kBlockN = 128;
    constexpr int kHeadDim = 128;
    constexpr int kNumStagesQ = 1;
    constexpr bool CustomMask = false;
    using KernelTraits =
        AttentionKernelTraits<kBlockM, kBlockN, kHeadDim, kNumStagesQ, SplitKV,
                              MinKvBlocksPerPartition, CustomMask, Element>;

    const int num_n_block = params.max_num_partitions;
    const int max_num_block = params.max_num_block;

    const int max_compute_block =
        num_n_block * params.kv_head_num * params.batch_size;

    using CollectiveMainloop = CollectiveMainloopFwdDecoder<KernelTraits>;

    typename CollectiveMainloop::Params mainloop_params =
        CollectiveMainloop::to_underlying_arguments(
            {static_cast<Element const *>(params.q_ptr),
             get_gmem_layout<kHeadDim, kBlockM>(params.batch_size,
                                                params.kv_head_num),
             static_cast<Element const *>(params.k_ptr),
             get_gmem_layout<kHeadDim, kBlockN>(max_num_block,
                                                params.kv_head_num),
             static_cast<Element const *>(params.v_ptr),
             get_gmem_layout<kHeadDim, kBlockN>(max_num_block,
                                                params.kv_head_num),
             static_cast<Element *>(params.o_ptr),
             params.cu_seq_q,
             params.cu_seq_k,
             params.seq_len_encoder,
             params.seq_len_decoder,
             params.mask,
             static_cast<Element *>(params.partition_o_ptr),
             params.max_sum,
             params.scale_softmax_log2,
             params.block_tables,
             params.max_blocks_per_seq,
             params.max_num_partitions,
             params.head_num,
             params.kv_head_num,
             num_n_block,
             max_compute_block});

    void *kernel;
    kernel = (void *)compute_attn_ws_decoder<KernelTraits, CollectiveMainloop>;
    int smem_size = sizeof(typename KernelTraits::SharedStorage);

    if (smem_size >= 48 * 1024) {
        cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    }

    dim3 block_dims(KernelTraits::kThreads);
    // Non-persistent: 3D grid = {num_n_block, kv_head_num, batch_size}
    dim3 grid_dims = {static_cast<unsigned int>(num_n_block),
                      static_cast<unsigned int>(params.kv_head_num),
                      static_cast<unsigned int>(params.batch_size)};
    using ClusterShape = typename KernelTraits::ClusterShape;
    dim3 cluster_dims(cute::size<0>(ClusterShape{}),
                      cute::size<1>(ClusterShape{}),
                      cute::size<2>(ClusterShape{}));
    cutlass::ClusterLaunchParams launch_params{grid_dims, block_dims,
                                               cluster_dims, smem_size, stream};
    cutlass::launch_kernel_on_cluster(launch_params, kernel, mainloop_params);
}

struct Fp8PersistentBlockCoord {
    int bidb;
    int bidh;
    int n_block;
};

CUTLASS_DEVICE Fp8PersistentBlockCoord decode_fp8_persistent_logical_block(
    const int logical_block, const int kv_head_num,
    const int active_partitions) {
    const int nb_block = logical_block / kv_head_num;
    const int bidb = nb_block / active_partitions;
    return {bidb, logical_block - nb_block * kv_head_num,
            nb_block - bidb * active_partitions};
}

template <typename KernelTraits, typename CollectiveMainloop>
__global__ void __launch_bounds__(KernelTraits::kThreads, 1)
    compute_fp8_attn_ws_decoder(
        CUTE_GRID_CONSTANT
        typename CollectiveMainloop::Params const mainloop_params) {
    using MainloopPipeline = typename KernelTraits::MainloopPipeline;
    using PipelineParams = typename MainloopPipeline::Params;
    using TmemAllocator = cute::TMEM::Allocator1Sm;
    using ClusterShape = typename KernelTraits::ClusterShape;

    constexpr int kNumThreadsPerWarp = cutlass::NumThreadsPerWarp;
    constexpr int kBlockN = KernelTraits::kBlockN;
    constexpr int kStageBlock = KernelTraits::KStageBlock;
    constexpr int kTokensPerBlock = kBlockN * kStageBlock;

    const int thread_idx = threadIdx.x;
    const int warp_idx = cutlass::canonical_warp_idx_sync();
    const int lane_idx = thread_idx % kNumThreadsPerWarp;

    extern __shared__ char shared_memory[];
    auto &shared_storage =
        *reinterpret_cast<typename KernelTraits::SharedStorage *>(
            shared_memory);

    if (warp_idx == 0) {
        CollectiveMainloop::prefetch_tma_descriptors(mainloop_params);
    } else if (warp_idx == 1) {
        cute::initialize_barrier(shared_storage.mbar_load_q_full, 1);
        cute::initialize_barrier(shared_storage.mbar_load_q_empty, 32);
        cute::initialize_barrier(shared_storage.mbar_tmem_dealloc,
                                 4 * kNumThreadsPerWarp);
    } else if (warp_idx == 2) {
#pragma unroll
        for (int i = 0; i < kStageBlock; ++i) {
            cute::initialize_barrier(shared_storage.mbar_S_full[i], 1);
            cute::initialize_barrier(shared_storage.mbar_O_full[i], 1);
        }
    } else if (warp_idx == 3) {
#pragma unroll
        for (int i = 0; i < kStageBlock; ++i) {
            cute::initialize_barrier(shared_storage.mbar_P_full_O_rescaled[i],
                                     4 * kNumThreadsPerWarp);
            cute::initialize_barrier(shared_storage.mbar_P_full_2[i],
                                     4 * kNumThreadsPerWarp);
        }
    }
    __syncthreads();

    {
        PipelineParams pipeline_params;
        pipeline_params.transaction_bytes =
            CollectiveMainloop::TmaTransactionBytesK;
        pipeline_params.num_consumers = kNumThreadsPerWarp;
        pipeline_params.num_producers = 1;
        MainloopPipeline::init_barriers(shared_storage.pipeline_k,
                                        pipeline_params, ClusterShape{});
        MainloopPipeline::init_barriers(shared_storage.pipeline_v,
                                        pipeline_params, ClusterShape{});
    }
    __syncthreads();

    constexpr int kFixedLaunchBlocks = 148;

    int local_active_partitions = 0;
    int local_min_partitions = mainloop_params.max_num_partitions + 1;
    int local_max_seqlen_k = 0;
    int local_min_seqlen_k = INT_MAX;
    int local_max_seqlen_q = 0;
    int local_min_seqlen_q = INT_MAX;
    for (int bidb = thread_idx; bidb < mainloop_params.batch_size;
         bidb += KernelTraits::kThreads) {
        const int seqlen_k = mainloop_params.cu_seq_k[bidb];
        const int seqlen_q =
            mainloop_params.cu_seq_q[bidb + 1] - mainloop_params.cu_seq_q[bidb];
        const int num_partitions =
            seqlen_k > 0 ? (seqlen_k + kTokensPerBlock - 1) / kTokensPerBlock
                         : 0;
        local_active_partitions = local_active_partitions > num_partitions
                                      ? local_active_partitions
                                      : num_partitions;
        local_min_partitions = local_min_partitions < num_partitions
                                   ? local_min_partitions
                                   : num_partitions;
        local_max_seqlen_k =
            local_max_seqlen_k > seqlen_k ? local_max_seqlen_k : seqlen_k;
        local_min_seqlen_k =
            local_min_seqlen_k < seqlen_k ? local_min_seqlen_k : seqlen_k;
        local_max_seqlen_q =
            local_max_seqlen_q > seqlen_q ? local_max_seqlen_q : seqlen_q;
        local_min_seqlen_q =
            local_min_seqlen_q < seqlen_q ? local_min_seqlen_q : seqlen_q;
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        int other =
            __shfl_down_sync(0xffffffff, local_active_partitions, offset);
        local_active_partitions =
            local_active_partitions > other ? local_active_partitions : other;
        other = __shfl_down_sync(0xffffffff, local_min_partitions, offset);
        local_min_partitions =
            local_min_partitions < other ? local_min_partitions : other;
        other = __shfl_down_sync(0xffffffff, local_max_seqlen_k, offset);
        local_max_seqlen_k =
            local_max_seqlen_k > other ? local_max_seqlen_k : other;
        other = __shfl_down_sync(0xffffffff, local_min_seqlen_k, offset);
        local_min_seqlen_k =
            local_min_seqlen_k < other ? local_min_seqlen_k : other;
        other = __shfl_down_sync(0xffffffff, local_max_seqlen_q, offset);
        local_max_seqlen_q =
            local_max_seqlen_q > other ? local_max_seqlen_q : other;
        other = __shfl_down_sync(0xffffffff, local_min_seqlen_q, offset);
        local_min_seqlen_q =
            local_min_seqlen_q < other ? local_min_seqlen_q : other;
    }

    float *active_partitions_smem = shared_storage.local_max;
    constexpr int kNumWarps = KernelTraits::kThreads / 32;
    if (lane_idx == 0) {
        active_partitions_smem[warp_idx] =
            static_cast<float>(local_active_partitions);
        active_partitions_smem[kNumWarps + warp_idx] =
            static_cast<float>(local_min_partitions);
        active_partitions_smem[2 * kNumWarps + warp_idx] =
            static_cast<float>(local_max_seqlen_k);
        active_partitions_smem[3 * kNumWarps + warp_idx] =
            static_cast<float>(local_min_seqlen_k);
        active_partitions_smem[4 * kNumWarps + warp_idx] =
            static_cast<float>(local_max_seqlen_q);
        active_partitions_smem[5 * kNumWarps + warp_idx] =
            static_cast<float>(local_min_seqlen_q);
    }
    __syncthreads();

    if (warp_idx == 0) {
        int max_active_partitions =
            lane_idx < kNumWarps
                ? static_cast<int>(active_partitions_smem[lane_idx])
                : 0;
        int min_active_partitions =
            lane_idx < kNumWarps
                ? static_cast<int>(active_partitions_smem[kNumWarps + lane_idx])
                : mainloop_params.max_num_partitions + 1;
        int max_seqlen_k =
            lane_idx < kNumWarps
                ? static_cast<int>(
                      active_partitions_smem[2 * kNumWarps + lane_idx])
                : 0;
        int min_seqlen_k =
            lane_idx < kNumWarps
                ? static_cast<int>(
                      active_partitions_smem[3 * kNumWarps + lane_idx])
                : INT_MAX;
        int max_seqlen_q =
            lane_idx < kNumWarps
                ? static_cast<int>(
                      active_partitions_smem[4 * kNumWarps + lane_idx])
                : 0;
        int min_seqlen_q =
            lane_idx < kNumWarps
                ? static_cast<int>(
                      active_partitions_smem[5 * kNumWarps + lane_idx])
                : INT_MAX;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            int other =
                __shfl_down_sync(0xffffffff, max_active_partitions, offset);
            max_active_partitions =
                max_active_partitions > other ? max_active_partitions : other;
            other = __shfl_down_sync(0xffffffff, min_active_partitions, offset);
            min_active_partitions =
                min_active_partitions < other ? min_active_partitions : other;
            other = __shfl_down_sync(0xffffffff, max_seqlen_k, offset);
            max_seqlen_k = max_seqlen_k > other ? max_seqlen_k : other;
            other = __shfl_down_sync(0xffffffff, min_seqlen_k, offset);
            min_seqlen_k = min_seqlen_k < other ? min_seqlen_k : other;
            other = __shfl_down_sync(0xffffffff, max_seqlen_q, offset);
            max_seqlen_q = max_seqlen_q > other ? max_seqlen_q : other;
            other = __shfl_down_sync(0xffffffff, min_seqlen_q, offset);
            min_seqlen_q = min_seqlen_q < other ? min_seqlen_q : other;
        }
        if (lane_idx == 0) {
            if (max_active_partitions > mainloop_params.max_num_partitions) {
                max_active_partitions = mainloop_params.max_num_partitions;
            }
            if (min_active_partitions > max_active_partitions) {
                min_active_partitions = max_active_partitions;
            }
            active_partitions_smem[0] =
                static_cast<float>(max_active_partitions);
            active_partitions_smem[1] =
                static_cast<float>(min_active_partitions);
            active_partitions_smem[2] = static_cast<float>(max_seqlen_k);
            active_partitions_smem[3] = static_cast<float>(min_seqlen_k);
            active_partitions_smem[4] = static_cast<float>(max_seqlen_q);
            active_partitions_smem[5] = static_cast<float>(min_seqlen_q);
        }
    }
    __syncthreads();

    const int active_partitions = static_cast<int>(active_partitions_smem[0]);
    const bool uniform_partitions =
        active_partitions == static_cast<int>(active_partitions_smem[1]);
    const int uniform_seqlen_k = static_cast<int>(active_partitions_smem[2]);
    const int uniform_seqlen_q = static_cast<int>(active_partitions_smem[4]);
    const bool uniform_lengths =
        uniform_seqlen_k == static_cast<int>(active_partitions_smem[3]) &&
        uniform_seqlen_q == static_cast<int>(active_partitions_smem[5]);
    const int uniform_k_blocks =
        uniform_seqlen_k > 0 ? (uniform_seqlen_k + kBlockN - 1) / kBlockN : 0;
    const int max_logical_blocks = active_partitions *
                                   mainloop_params.kv_head_num *
                                   mainloop_params.batch_size;

    if (warp_idx == KernelTraits::kMmaWarpIds) {
        TmemAllocator tmem_allocator{};
        tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns,
                                &shared_storage.tmem_holding_buf);
        __syncwarp();
        tmem_allocator.release_allocation_lock();
    }
    __syncthreads();

    if (warp_idx < KernelTraits::kSoftmaxEndWarpIds) {
        cutlass::arch::warpgroup_reg_alloc<KernelTraits::num_regs_softmax>();
        CollectiveMainloop collective_mainloop;
        int phase = 0;
        for (int logical_block = blockIdx.x; logical_block < max_logical_blocks;
             logical_block += kFixedLaunchBlocks) {
            const Fp8PersistentBlockCoord block_coord =
                decode_fp8_persistent_logical_block(logical_block,
                                                    mainloop_params.kv_head_num,
                                                    active_partitions);
            const int bidb = block_coord.bidb;
            const int bidh = block_coord.bidh;
            const int n_block = block_coord.n_block;
            const int seqlen_k = uniform_lengths
                                     ? uniform_seqlen_k
                                     : mainloop_params.cu_seq_k[bidb];
            const int seqlen_q = uniform_lengths
                                     ? uniform_seqlen_q
                                     : mainloop_params.cu_seq_q[bidb + 1] -
                                           mainloop_params.cu_seq_q[bidb];
            if (!uniform_partitions) {
                const int num_partitions =
                    seqlen_k > 0
                        ? (seqlen_k + kTokensPerBlock - 1) / kTokensPerBlock
                        : 0;
                if (n_block >= num_partitions) continue;
            }
            const int n_block_max =
                uniform_lengths
                    ? min(uniform_k_blocks - n_block * kStageBlock, kStageBlock)
                    : collective_mainloop.get_n_block_max(seqlen_k, n_block);
            collective_mainloop.softmax(mainloop_params, shared_storage,
                                        thread_idx, n_block_max, n_block, bidh,
                                        bidb, seqlen_q, seqlen_k, phase);
            if (n_block_max < kStageBlock) {
                const int warp_group = thread_idx / (4 * kNumThreadsPerWarp);
#pragma unroll
                for (int stage = 0; stage < kStageBlock; ++stage) {
                    if (stage >= n_block_max && (stage & 1) == warp_group) {
                        cute::arrive_barrier(
                            shared_storage.mbar_P_full_O_rescaled[stage]);
                        cute::arrive_barrier(
                            shared_storage.mbar_P_full_2[stage]);
                    }
                }
            }
            __syncthreads();
            phase ^= 1;
        }
    } else if (warp_idx < KernelTraits::kMmaWarpIds) {
        cutlass::arch::warpgroup_reg_dealloc<
            KernelTraits::num_regs_correction>();
        CollectiveMainloop collective_mainloop;
        int phase = 0;
        for (int logical_block = blockIdx.x; logical_block < max_logical_blocks;
             logical_block += kFixedLaunchBlocks) {
            const Fp8PersistentBlockCoord block_coord =
                decode_fp8_persistent_logical_block(logical_block,
                                                    mainloop_params.kv_head_num,
                                                    active_partitions);
            const int bidb = block_coord.bidb;
            const int bidh = block_coord.bidh;
            const int n_block = block_coord.n_block;
            const int seqlen_k = uniform_lengths
                                     ? uniform_seqlen_k
                                     : mainloop_params.cu_seq_k[bidb];
            const int seqlen_q = uniform_lengths
                                     ? uniform_seqlen_q
                                     : mainloop_params.cu_seq_q[bidb + 1] -
                                           mainloop_params.cu_seq_q[bidb];
            if (!uniform_partitions) {
                const int num_partitions =
                    seqlen_k > 0
                        ? (seqlen_k + kTokensPerBlock - 1) / kTokensPerBlock
                        : 0;
                if (n_block >= num_partitions) continue;
            }
            const int n_block_max =
                uniform_lengths
                    ? min(uniform_k_blocks - n_block * kStageBlock, kStageBlock)
                    : collective_mainloop.get_n_block_max(seqlen_k, n_block);
            collective_mainloop.correction(
                mainloop_params, shared_storage, thread_idx, n_block_max,
                n_block, bidh, bidb, seqlen_q, seqlen_k, phase);
            cute::arrive_barrier(shared_storage.mbar_tmem_dealloc);
            __syncthreads();
            phase ^= 1;
        }
    } else if (warp_idx == KernelTraits::kMmaWarpIds) {
        cutlass::arch::warpgroup_reg_dealloc<KernelTraits::num_regs_other>();
        CollectiveMainloop collective_mainloop;
        PipelineParams pipeline_params;
        pipeline_params.transaction_bytes =
            CollectiveMainloop::TmaTransactionBytesK;
        pipeline_params.role = MainloopPipeline::ThreadCategory::Consumer;
        pipeline_params.is_leader = lane_idx == 0;
        pipeline_params.num_consumers = kNumThreadsPerWarp;
        MainloopPipeline pipeline_k(shared_storage.pipeline_k, pipeline_params,
                                    ClusterShape{}, cute::false_type{});
        MainloopPipeline pipeline_v(shared_storage.pipeline_v, pipeline_params,
                                    ClusterShape{}, cute::false_type{});
        int pipeline_advance = 0;
        int phase = 0;
        for (int logical_block = blockIdx.x; logical_block < max_logical_blocks;
             logical_block += kFixedLaunchBlocks) {
            const Fp8PersistentBlockCoord block_coord =
                decode_fp8_persistent_logical_block(logical_block,
                                                    mainloop_params.kv_head_num,
                                                    active_partitions);
            const int bidb = block_coord.bidb;
            const int n_block = block_coord.n_block;
            const int seqlen_k = uniform_lengths
                                     ? uniform_seqlen_k
                                     : mainloop_params.cu_seq_k[bidb];
            if (!uniform_partitions) {
                const int num_partitions =
                    seqlen_k > 0
                        ? (seqlen_k + kTokensPerBlock - 1) / kTokensPerBlock
                        : 0;
                if (n_block >= num_partitions) continue;
            }
            const int n_block_max =
                uniform_lengths
                    ? min(uniform_k_blocks - n_block * kStageBlock, kStageBlock)
                    : collective_mainloop.get_n_block_max(seqlen_k, n_block);
            collective_mainloop.mma(mainloop_params, pipeline_k, pipeline_v,
                                    shared_storage, n_block_max, phase,
                                    pipeline_advance);
            if (n_block_max < kStageBlock) {
#pragma unroll
                for (int stage = 0; stage < kStageBlock; ++stage) {
                    if (stage >= n_block_max) {
                        cutlass::arch::umma_arrive(
                            &shared_storage.mbar_S_full[stage]);
                        cutlass::arch::umma_arrive(
                            &shared_storage.mbar_O_full[stage]);
                    }
                }
            }
            cute::arrive_barrier(shared_storage.mbar_load_q_empty);
            cute::wait_barrier(shared_storage.mbar_tmem_dealloc, phase);
            __syncthreads();
            pipeline_advance += n_block_max;
            phase ^= 1;
        }
    } else if (warp_idx == KernelTraits::kLoadWarpIds) {
        cutlass::arch::warpgroup_reg_dealloc<KernelTraits::num_regs_other>();
        CollectiveMainloop collective_mainloop;
        PipelineParams pipeline_params;
        pipeline_params.transaction_bytes =
            CollectiveMainloop::TmaTransactionBytesK;
        pipeline_params.role = MainloopPipeline::ThreadCategory::Producer;
        pipeline_params.is_leader = lane_idx == 0;
        pipeline_params.num_consumers = kNumThreadsPerWarp;
        MainloopPipeline pipeline_k(shared_storage.pipeline_k, pipeline_params,
                                    ClusterShape{}, cute::false_type{});
        MainloopPipeline pipeline_v(shared_storage.pipeline_v, pipeline_params,
                                    ClusterShape{}, cute::false_type{});
        int pipeline_advance = 0;
        int q_empty_phase = 1;
        for (int logical_block = blockIdx.x; logical_block < max_logical_blocks;
             logical_block += kFixedLaunchBlocks) {
            const Fp8PersistentBlockCoord block_coord =
                decode_fp8_persistent_logical_block(logical_block,
                                                    mainloop_params.kv_head_num,
                                                    active_partitions);
            const int bidb = block_coord.bidb;
            const int bidh = block_coord.bidh;
            const int n_block = block_coord.n_block;
            const int seqlen_k = uniform_lengths
                                     ? uniform_seqlen_k
                                     : mainloop_params.cu_seq_k[bidb];
            if (!uniform_partitions) {
                const int num_partitions =
                    seqlen_k > 0
                        ? (seqlen_k + kTokensPerBlock - 1) / kTokensPerBlock
                        : 0;
                if (n_block >= num_partitions) continue;
            }
            const int n_block_max =
                uniform_lengths
                    ? min(uniform_k_blocks - n_block * kStageBlock, kStageBlock)
                    : collective_mainloop.get_n_block_max(seqlen_k, n_block);
            cute::wait_barrier(shared_storage.mbar_load_q_empty, q_empty_phase);
            collective_mainloop.load(mainloop_params, pipeline_k, pipeline_v,
                                     shared_storage, lane_idx, n_block_max,
                                     n_block, bidh, bidb, pipeline_advance);
            const int next_logical_block = logical_block + kFixedLaunchBlocks;
            if (next_logical_block < max_logical_blocks) {
                const Fp8PersistentBlockCoord next_block_coord =
                    decode_fp8_persistent_logical_block(
                        next_logical_block, mainloop_params.kv_head_num,
                        active_partitions);
                const int next_seqlen_k =
                    uniform_lengths
                        ? uniform_seqlen_k
                        : mainloop_params.cu_seq_k[next_block_coord.bidb];
                bool prefetch_valid = uniform_partitions;
                if (!prefetch_valid) {
                    const int next_num_partitions =
                        next_seqlen_k > 0
                            ? (next_seqlen_k + kTokensPerBlock - 1) /
                                  kTokensPerBlock
                            : 0;
                    prefetch_valid =
                        next_block_coord.n_block < next_num_partitions;
                }
                if (prefetch_valid) {
                    collective_mainloop.prefetch_next_partition_warp(
                        mainloop_params, lane_idx, next_block_coord.n_block,
                        next_block_coord.bidh, next_block_coord.bidb,
                        next_seqlen_k);
                }
            }
            __syncthreads();
            pipeline_advance += n_block_max;
            q_empty_phase ^= 1;
        }
    } else {
        cutlass::arch::warpgroup_reg_dealloc<KernelTraits::num_regs_empty>();
        if (uniform_partitions) {
            for (int logical_block = blockIdx.x;
                 logical_block < max_logical_blocks;
                 logical_block += kFixedLaunchBlocks) {
                __syncthreads();
            }
        } else {
            for (int logical_block = blockIdx.x;
                 logical_block < max_logical_blocks;
                 logical_block += kFixedLaunchBlocks) {
                const Fp8PersistentBlockCoord block_coord =
                    decode_fp8_persistent_logical_block(
                        logical_block, mainloop_params.kv_head_num,
                        active_partitions);
                const int bidb = block_coord.bidb;
                const int n_block = block_coord.n_block;
                const int seqlen_k = uniform_lengths
                                         ? uniform_seqlen_k
                                         : mainloop_params.cu_seq_k[bidb];
                const int num_partitions =
                    seqlen_k > 0
                        ? (seqlen_k + kTokensPerBlock - 1) / kTokensPerBlock
                        : 0;
                if (n_block >= num_partitions) continue;
                __syncthreads();
            }
        }
    }

    __syncthreads();
    if (warp_idx == KernelTraits::kMmaWarpIds) {
        TmemAllocator tmem_allocator{};
        tmem_allocator.free(shared_storage.tmem_holding_buf,
                            TmemAllocator::Sm100TmemCapacityColumns);
    }
}

template <int kDequantScalePackSize>
auto get_scale_layout(int max_num_block, int kv_head_num) {
    return cute::make_layout(
        cute::make_shape(cute::Int<kDequantScalePackSize>{}, kv_head_num,
                         max_num_block),
        cute::make_stride(cute::_1{}, cute::Int<kDequantScalePackSize>{},
                          kv_head_num * kDequantScalePackSize));
}

template <int kBlockM, int kBlockN, int KStageBlock>
void LaunchFp8DecoderAttention(AttentionParams &params, cudaStream_t stream) {
    using KernelTraits =
        AttentionFp8KernelTraits<kBlockM, kBlockN, KStageBlock>;
    using Element = typename KernelTraits::Element;
    constexpr int kHeadDim = KernelTraits::kHeadDim;

    const int max_num_block = params.max_num_block;
    using CollectiveMainloop = CollectiveMainloopFwdFP8Decoder<KernelTraits>;

    typename CollectiveMainloop::Params mainloop_params =
        CollectiveMainloop::to_underlying_arguments(
            {static_cast<Element const *>(params.q_ptr),
             get_fp8_gmem_layout<kHeadDim, kBlockM>(params.batch_size,
                                                    params.kv_head_num),
             static_cast<Element const *>(params.k_ptr),
             get_fp8_gmem_layout<kHeadDim, kBlockN>(max_num_block,
                                                    params.kv_head_num),
             static_cast<Element const *>(params.v_ptr),
             get_fp8_gmem_layout<kHeadDim, kBlockN>(max_num_block,
                                                    params.kv_head_num),
             get_scale_layout<kDequantScalePackSize>(max_num_block,
                                                     params.kv_head_num),
             params.cu_seq_q,
             params.cu_seq_k,
             static_cast<cutlass::bfloat16_t *>(params.partition_o_ptr),
             params.max_sum,
             params.q_dequant_scale,
             params.kv_dequant_scale,
             params.block_tables,
             params.n_block_infos,
             params.max_blocks_per_seq,
             params.head_num,
             params.kv_head_num,
             params.batch_size,
             params.max_num_partitions});

    void *kernel =
        (void *)compute_fp8_attn_ws_decoder<KernelTraits, CollectiveMainloop>;
    int smem_size = sizeof(typename KernelTraits::SharedStorage);

    if (smem_size >= 48 * 1024) {
        cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    }

    dim3 block_dims(KernelTraits::kThreads);
    dim3 grid_dims = {148, 1, 1};
    using ClusterShape = typename KernelTraits::ClusterShape;
    dim3 cluster_dims(cute::size<0>(ClusterShape{}),
                      cute::size<1>(ClusterShape{}),
                      cute::size<2>(ClusterShape{}));
    cutlass::ClusterLaunchParams launch_params{grid_dims, block_dims,
                                               cluster_dims, smem_size, stream};
    cutlass::launch_kernel_on_cluster(launch_params, kernel, mainloop_params);
}

}  // namespace halops
