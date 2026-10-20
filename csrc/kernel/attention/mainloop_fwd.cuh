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
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include "cute/tensor.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/pipeline/pipeline.hpp"
#include "kernel/attention/reduce_kernel.cuh"
#include "kernel/attention/softmax.cuh"
#include "kernel/attention/tile_scheduler.cuh"
#include "kernel/attention/utils.cuh"

namespace halops {

template <typename KernelTraits>
struct CollectiveMainloopFwdDecoder {
    static constexpr int kNumStagesKV = KernelTraits::kNumStagesKV;
    static constexpr int kNumStagesQ = KernelTraits::kNumStagesQ;
    static constexpr int kHeadDim = KernelTraits::kHeadDim;
    static constexpr int kBlockM = KernelTraits::kBlockM;
    static constexpr int kBlockN = KernelTraits::kBlockN;
    static constexpr bool CustomMask = KernelTraits::CustomMask;

    static constexpr bool kSplitKV = KernelTraits::kSplitKV;
    // Must match the host's partition budget.
    static constexpr int kMinKvBlocksPerPartition =
        KernelTraits::kMinKvBlocksPerPartition;
    // Split granularity under split-KV, unroll granularity under no-split.
    static constexpr int kNumUnrolledKvBlocks =
        KernelTraits::kNumUnrolledKvBlocks;
    static_assert(kNumUnrolledKvBlocks % (2 * kNumStagesKV) == 0);

    using Element = typename KernelTraits::Element;
    using ElementAcc = typename KernelTraits::ElementAcc;
    using TileShape_MNK = typename KernelTraits::TileShape_MNK;
    using ClusterShape = typename KernelTraits::ClusterShape;
    using MainloopPipeline = typename KernelTraits::MainloopPipeline;
    using PipelineState = typename MainloopPipeline::PipelineState;

    using GmemTiledCopyQ = cute::SM90_TMA_LOAD;
    using GmemTiledCopyKV = decltype(cutlass::gemm::collective::detail::
                                         sm90_cluster_shape_to_tma_atom(
                                             cute::shape<0>(ClusterShape{})));
    using GmemTiledCopyO = cute::SM90_TMA_STORE;

    using SmemLayoutQ = typename KernelTraits::SmemLayoutQ;
    using SmemLayoutK = typename KernelTraits::SmemLayoutK;
    using SmemLayoutV = typename KernelTraits::SmemLayoutV;
    using SmemLayoutVt = typename KernelTraits::SmemLayoutVt;
    using SmemLayoutS = typename KernelTraits::SmemLayoutS;
    using SmemLayoutO = typename KernelTraits::SmemLayoutO;
    using TiledMma0 = typename KernelTraits::TiledMma0;
    using TiledMma1 = typename KernelTraits::TiledMma1;

    using ShapeT = cute::Shape<int32_t, int32_t, int32_t, int32_t>;
    using StrideT = cute::Shape<int32_t, cute::_1, int32_t, int32_t>;
    using LayoutT = cute::Layout<ShapeT, StrideT>;

    using TMA_Q = decltype(cute::make_tma_copy(
        GmemTiledCopyQ{},
        cute::make_tensor(
            cute::make_gmem_ptr(static_cast<Element const *>(nullptr)),
            cute::repeat_like(StrideT{}, int32_t(0)), StrideT{}),
        SmemLayoutQ{}(cute::_, cute::_, cute::_0{}),
        cute::select<0, 2>(TileShape_MNK{}), cute::_1{}));

    using TMA_KV = decltype(cute::make_tma_copy(
        GmemTiledCopyKV{},
        cute::make_tensor(
            cute::make_gmem_ptr(static_cast<Element const *>(nullptr)),
            cute::repeat_like(StrideT{}, int32_t(0)), StrideT{}),
        cute::take<0, 2>(SmemLayoutK{}), cute::select<1, 2>(TileShape_MNK{}),
        cute::size<0>(ClusterShape{})));

    static constexpr uint32_t TmaTransactionBytesQ =
        static_cast<uint32_t>(cute::size(cute::take<0, 2>(SmemLayoutQ{})) *
                              cutlass::sizeof_bits_v<Element> / 8);
    static constexpr uint32_t TmaTransactionBytesK =
        static_cast<uint32_t>(cute::size(cute::take<0, 2>(SmemLayoutK{})) *
                              cutlass::sizeof_bits_v<Element> / 8);

    struct Arguments {
        Element const *ptr_Q;
        LayoutT layout_Q;
        Element const *ptr_K;
        LayoutT layout_K;
        Element const *ptr_V;
        LayoutT layout_V;
        Element *ptr_O;
        const int *cu_seq_q;
        const int *cu_seq_k;
        const int *seq_len_encoder;
        const int *seq_len_decoder;
        const int *mask;
        Element *partition_o_ptr;
        float2 *max_sum;
        float const softmax_scale_log2;
        const int *block_tables;
        const int max_blocks_per_seq;
        const int max_num_partitions;
        const int head_num;
        const int kv_head_num;
        const int num_n_block;
        const int max_compute_block;
    };

    struct Params {
        LayoutT layout_Q;
        LayoutT layout_K;
        LayoutT layout_V;
        TMA_Q tma_load_Q;
        TMA_KV tma_load_K, tma_load_V;
        Element *ptr_O;
        const int *cu_seq_q;
        const int *cu_seq_k;
        const int *seq_len_encoder;
        const int *seq_len_decoder;
        const int *mask;
        Element *partition_o_ptr;
        float2 *max_sum;
        float const softmax_scale_log2;
        const int *block_tables;
        int max_blocks_per_seq;
        int max_num_partitions;
        const int head_num;
        const int kv_head_num;
        const int max_compute_block;
        cutlass::FastDivmod qhead_per_khead_divmod;
        cutlass::FastDivmod khead_divmod;
        cutlass::FastDivmod kblock_divmod;
        Element const *k_raw_ptr;
        Element const *v_raw_ptr;
    };

    static Params to_underlying_arguments(Arguments const &args) {
        cute::Tensor mQ =
            cute::make_tensor(cute::make_gmem_ptr(args.ptr_Q), args.layout_Q);
        TMA_Q tma_load_Q = cute::make_tma_copy(
            GmemTiledCopyQ{}, mQ, SmemLayoutQ{}(cute::_, cute::_, cute::_0{}),
            cute::select<0, 2>(TileShape_MNK{}), cute::_1{});
        cute::Tensor mK =
            cute::make_tensor(cute::make_gmem_ptr(args.ptr_K), args.layout_K);
        TMA_KV tma_load_K = cute::make_tma_copy(
            GmemTiledCopyKV{}, mK, SmemLayoutK{}(cute::_, cute::_, cute::_0{}),
            cute::select<1, 2>(TileShape_MNK{}), cute::size<0>(ClusterShape{}));
        cute::Tensor mV =
            cute::make_tensor(cute::make_gmem_ptr(args.ptr_V), args.layout_V);
        TMA_KV tma_load_V = cute::make_tma_copy(
            GmemTiledCopyKV{}, mV, SmemLayoutV{}(cute::_, cute::_, cute::_0{}),
            cute::select<1, 2>(TileShape_MNK{}), cute::size<0>(ClusterShape{}));

        return {args.layout_Q,
                args.layout_K,
                args.layout_V,
                tma_load_Q,
                tma_load_K,
                tma_load_V,
                args.ptr_O,
                args.cu_seq_q,
                args.cu_seq_k,
                args.seq_len_encoder,
                args.seq_len_decoder,
                args.mask,
                args.partition_o_ptr,
                args.max_sum,
                args.softmax_scale_log2,
                args.block_tables,
                args.max_blocks_per_seq,
                args.max_num_partitions,
                args.head_num,
                args.kv_head_num,
                args.max_compute_block,
                cutlass::FastDivmod(
                    cute::ceil_div(args.head_num, args.kv_head_num)),
                cutlass::FastDivmod(args.kv_head_num),
                cutlass::FastDivmod(args.num_n_block),
                args.ptr_K,
                args.ptr_V};
    }

    CUTLASS_DEVICE void get_coordinates(Params const &mainloop_params,
                                        const int block_idx, int &n_block,
                                        int &bidb, int &bidh) {
        int nb_block;
        mainloop_params.khead_divmod(nb_block, bidh, block_idx);
        mainloop_params.kblock_divmod(bidb, n_block, nb_block);
    }

    // KV tiles one CTA walks, derived from this batch's own KV length.
    CUTLASS_DEVICE static int get_num_kv_blocks_per_cta(
        Params const &mainloop_params, const int seqlen_k) {
        const int num_kv_blocks = (seqlen_k + kBlockN - 1) / kBlockN;
        const int max_num_partitions = mainloop_params.max_num_partitions;
        return max(kMinKvBlocksPerPartition,
                   mainloop_params.kblock_divmod.divide(
                       num_kv_blocks + max_num_partitions - 1));
    }

    CUTLASS_DEVICE
    static void prefetch_tma_descriptors(Params const &mainloop_params) {
        cute::prefetch_tma_descriptor(
            mainloop_params.tma_load_Q.get_tma_descriptor());
        cute::prefetch_tma_descriptor(
            mainloop_params.tma_load_K.get_tma_descriptor());
        cute::prefetch_tma_descriptor(
            mainloop_params.tma_load_V.get_tma_descriptor());
    }

    // Warp-collaborative L2 prefetch: all 32 lanes share the work.
    CUTLASS_DEVICE void prefetch_next_partition_warp(
        Params const &mainloop_params, const int lane_idx,
        const int next_n_block, const int next_bidh, const int next_bidb) {
        const int seqlen_k = mainloop_params.cu_seq_k[next_bidb];
        const int num_kv_blocks_per_cta =
            get_num_kv_blocks_per_cta(mainloop_params, seqlen_k);
        if (next_n_block * kBlockN * num_kv_blocks_per_cta >= seqlen_k) {
            return;
        }

        const int n_block_max =
            get_n_block_max(seqlen_k, next_n_block, num_kv_blocks_per_cta);
        const int *block_tables =
            mainloop_params.block_tables +
            next_bidb * mainloop_params.max_blocks_per_seq +
            next_n_block * num_kv_blocks_per_cta;

        constexpr int kElemBytes = cutlass::sizeof_bits_v<Element> / 8;
        constexpr int kBlockBytes = kBlockN * kHeadDim * kElemBytes;
        const int kv_head_num = mainloop_params.kv_head_num;
        const char *k_base =
            reinterpret_cast<const char *>(mainloop_params.k_raw_ptr);
        const char *v_base =
            reinterpret_cast<const char *>(mainloop_params.v_raw_ptr);
        constexpr int kCacheLine = 128;
        constexpr int kLinesPerBlock = kBlockBytes / kCacheLine;
        constexpr int kLinesPerLane = kLinesPerBlock / 32;

        const int64_t block_stride =
            static_cast<int64_t>(kv_head_num) * kBlockBytes;
        const int64_t head_stride = kBlockBytes;

        // For BF16, only prefetch K (not V) to avoid oversaturating memory.
        for (int iter = 0; iter < n_block_max; ++iter) {
            const int phys_block = __ldg(block_tables + iter);
            const int64_t byte_offset =
                static_cast<int64_t>(phys_block) * block_stride +
                static_cast<int64_t>(next_bidh) * head_stride;
            const char *k_block = k_base + byte_offset;
#pragma unroll
            for (int l = 0; l < kLinesPerLane; ++l) {
                int line_idx = lane_idx + l * 32;
                asm volatile("prefetch.global.L2 [%0];" ::"l"(
                    k_block + line_idx * kCacheLine));
            }
        }
    }

    // KV tiles this partition has to walk.
    CUTLASS_DEVICE int get_n_block_max(const int seqlen_k, const int m_block,
                                       const int num_kv_blocks_per_cta) {
        return min((seqlen_k + kBlockN - 1) / kBlockN -
                       m_block * num_kv_blocks_per_cta,
                   num_kv_blocks_per_cta);
    }

    template <typename SharedStorage>
    CUTLASS_DEVICE void load(Params const &mainloop_params,
                             MainloopPipeline pipeline_k,
                             MainloopPipeline pipeline_v,
                             SharedStorage &shared_storage, const int lane_idx,
                             const int n_block_max, const int m_block,
                             const int num_kv_blocks_per_cta, const int bidh,
                             const int bidb) {
        cute::Tensor sQ =
            cute::make_tensor(cute::make_smem_ptr(shared_storage.smem_q.data()),
                              SmemLayoutQ{}(cute::_, cute::_, cute::_0{}));
        cute::Tensor sK = cute::make_tensor(
            cute::make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutK{});
        cute::Tensor sV = cute::make_tensor(
            cute::make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutV{});

        cute::Tensor mQ = mainloop_params.tma_load_Q.get_tma_tensor(
            mainloop_params.layout_Q.shape());
        cute::Tensor mK = mainloop_params.tma_load_K.get_tma_tensor(
            mainloop_params.layout_K.shape());
        cute::Tensor mV = mainloop_params.tma_load_V.get_tma_tensor(
            mainloop_params.layout_V.shape());

        PipelineState k_producer_state =
            cutlass::make_producer_start_state<MainloopPipeline>();
        PipelineState v_producer_state =
            cutlass::make_producer_start_state<MainloopPipeline>();

        cute::Tensor gQ =
            cute::local_tile(mQ(cute::_, cute::_, bidh, bidb),
                             cute::select<0, 2>(TileShape_MNK{}),
                             cute::make_coord(cute::_0{}, cute::_0{}));
        cute::Tensor gK = cute::local_tile(
            mK(cute::_, cute::_, bidh, cute::_),
            cute::select<0, 1>(
                cute::Shape<cute::Int<kBlockN>, cute::Int<kHeadDim>>{}),
            cute::make_coord(cute::_0{}, cute::_0{}));
        cute::Tensor gV = cute::local_tile(
            mV(cute::_, cute::_, bidh, cute::_),
            cute::select<0, 1>(
                cute::Shape<cute::Int<kBlockN>, cute::Int<kHeadDim>>{}),
            cute::make_coord(cute::_0{}, cute::_0{}));

        auto [tQgQ, tQsQ] = cute::tma_partition(
            mainloop_params.tma_load_Q, cute::_0{}, cute::Layout<cute::_1>{},
            cute::group_modes<0, 2>(sQ), cute::group_modes<0, 2>(gQ));
        auto [tKgK, tKsK] = cute::tma_partition(
            mainloop_params.tma_load_K, cute::_0{}, cute::Layout<cute::_1>{},
            cute::group_modes<0, 2>(sK), cute::group_modes<0, 2>(gK));
        auto [tVgV, tVsV] = cute::tma_partition(
            mainloop_params.tma_load_V, cute::_0{}, cute::Layout<cute::_1>{},
            cute::group_modes<0, 2>(sV), cute::group_modes<0, 2>(gV));

        auto load_Q = [&]() {
            cute::set_barrier_transaction_bytes(
                shared_storage.mbar_load_q_full[0], TmaTransactionBytesQ);
            cute::copy(mainloop_params.tma_load_Q.with(
                           shared_storage.mbar_load_q_full[0], 0),
                       tQgQ, tQsQ);
        };

        auto load_K = [&](const int src_idx) {
            pipeline_k.producer_acquire(k_producer_state);
            cute::copy(
                mainloop_params.tma_load_K.with(
                    *pipeline_k.producer_get_barrier(k_producer_state), 0),
                tKgK(cute::_, src_idx),
                tKsK(cute::_, k_producer_state.index()));
            ++k_producer_state;
        };

        auto load_V = [&](const int src_idx) {
            pipeline_v.producer_acquire(v_producer_state);
            cute::copy(
                mainloop_params.tma_load_V.with(
                    *pipeline_v.producer_get_barrier(v_producer_state), 0),
                tVgV(cute::_, src_idx),
                tVsV(cute::_, v_producer_state.index()));
            ++v_producer_state;
        };

        if (cute::elect_one_sync()) {
            load_Q();
            const int *block_tables =
                mainloop_params.block_tables +
                bidb * mainloop_params.max_blocks_per_seq +
                m_block * num_kv_blocks_per_cta;
            load_K(block_tables[n_block_max - 1]);
            // Runtime partition length: unroll by kNumUnrolledKvBlocks and
            // predicate off the tail.
            for (int base = 0; base < n_block_max - 1;
                 base += kNumUnrolledKvBlocks) {
                range_constexpr<kNumUnrolledKvBlocks>([&](auto k) {
                    const int iter = base + k;
                    if (iter < n_block_max - 1) {
                        load_K(block_tables[n_block_max - 2 - iter]);
                        load_V(block_tables[n_block_max - 1 - iter]);
                    }
                });
            }
            load_V(block_tables[0]);
        }
    }

    template <typename SharedStorage>
    CUTLASS_DEVICE void mma(Params const &mainloop_params,
                            MainloopPipeline pipeline_k,
                            MainloopPipeline pipeline_v,
                            SharedStorage &shared_storage, const int lane_idx,
                            const int n_block_max, const int iter_phase = 0) {
        cute::Tensor sQ = cute::make_tensor(
            cute::make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutQ{});
        cute::Tensor sK = cute::make_tensor(
            cute::make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutK{});
        cute::Tensor sVt = cute::make_tensor(
            cute::make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutVt{});
        cute::Tensor sS = cute::make_tensor(
            cute::make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutS{});

        TiledMma0 tiled_mma0;
        TiledMma1 tiled_mma1;

        cute::ThrMMA cta_mma0 = tiled_mma0.get_slice(0);
        cute::ThrMMA cta_mma1 = tiled_mma1.get_slice(0);

        cute::Tensor tSrQ = cta_mma0.partition_fragment_A(sQ);
        cute::Tensor tSrK = cta_mma0.partition_fragment_B(sK);
        cute::Tensor tSrV = cta_mma1.partition_fragment_B(sVt);
        cute::Tensor tPtP = cta_mma1.partition_fragment_A(sS);
        cute::Tensor tStS = cute::partition_fragment_C(
            cta_mma0, cute::select<0, 1>(TileShape_MNK{}));
        int data_offset = 0;
        tPtP.data() = data_offset * kBlockN;
        tStS.data() = data_offset * kBlockN;
        cute::Tensor tOtO = cute::partition_fragment_C(
            cta_mma1, cute::select<0, 2>(TileShape_MNK{}));

        tOtO.data() = 2 * kBlockN;
        tiled_mma1.accumulate_ = cute::UMMA::ScaleOut::Zero;

        PipelineState mma_k_consumer_state, mma_v_consumer_state;
        // P_full_O_rescaled alternates between two slots, so its phase flips
        // once per pair of KV tiles.
        int P_full_O_rescaled_phase = 0;
        int P_stage = 0;

        cute::wait_barrier(shared_storage.mbar_load_q_full[0], iter_phase);
        pipeline_k.consumer_wait(mma_k_consumer_state);
        gemm<true>(
            tiled_mma0, tSrQ(cute::_, cute::_, cute::_, 0),
            tSrK(cute::_, cute::_, cute::_, mma_k_consumer_state.index()),
            tStS);
        pipeline_k.consumer_release(mma_k_consumer_state);
        cutlass::arch::umma_arrive(&shared_storage.mbar_S_full[data_offset]);
        ++mma_k_consumer_state;
        // Runtime partition length: unroll by kNumUnrolledKvBlocks and
        // predicate off the tail.
        for (int base = 0; base < n_block_max - 1;
             base += kNumUnrolledKvBlocks) {
            range_constexpr<kNumUnrolledKvBlocks>([&](auto k) {
                if (base + k < n_block_max - 1) {
                    data_offset ^= 1;
                    tStS.data() = data_offset * kBlockN;
                    pipeline_k.consumer_wait(mma_k_consumer_state);
                    gemm<true>(tiled_mma0, tSrQ(cute::_, cute::_, cute::_, 0),
                               tSrK(cute::_, cute::_, cute::_,
                                    mma_k_consumer_state.index()),
                               tStS);
                    pipeline_k.consumer_release(mma_k_consumer_state);
                    cutlass::arch::umma_arrive(
                        &shared_storage.mbar_S_full[data_offset]);
                    ++mma_k_consumer_state;
                    pipeline_v.consumer_wait(mma_v_consumer_state);
                    cute::wait_barrier(
                        shared_storage.mbar_P_full_O_rescaled[P_stage],
                        P_full_O_rescaled_phase);
                    gemm(tiled_mma1, tPtP,
                         tSrV(cute::_, cute::_, cute::_,
                              mma_v_consumer_state.index()),
                         tOtO);
                    pipeline_v.consumer_release(mma_v_consumer_state);
                    cutlass::arch::umma_arrive(&shared_storage.mbar_O_full[0]);
                    tPtP.data() = data_offset * kBlockN;
                    ++mma_v_consumer_state;
                    if constexpr (k % 2 == 1) {
                        P_full_O_rescaled_phase ^= 1;
                    }
                    P_stage ^= 1;
                }
            });
        }
        pipeline_v.consumer_wait(mma_v_consumer_state);
        cute::wait_barrier(shared_storage.mbar_P_full_O_rescaled[P_stage],
                           P_full_O_rescaled_phase);
        gemm(tiled_mma1, tPtP,
             tSrV(cute::_, cute::_, cute::_, mma_v_consumer_state.index()),
             tOtO);
        pipeline_v.consumer_release(mma_v_consumer_state);
        cutlass::arch::umma_arrive(&shared_storage.mbar_O_full[0]);
    }

    template <bool IsFirst, bool NeedMask, typename SharedStorage>
    CUTLASS_DEVICE void softmax_step(Params const &mainloop_params,
                                     SharedStorage &shared_storage,
                                     const int *mask, Softmax &softmax,
                                     const int seqlen_k, const int col_base,
                                     const int lane_idx, const int warp_idx,
                                     const int thread_idx,
                                     const int mma_si_consumer_phase,
                                     const int si_stage, int &pv_done_phase) {
        TiledMma0 tiled_mma0;
        cute::ThrMMA cta_mma0 = tiled_mma0.get_slice(0);
        cute::Tensor tStS = cute::partition_fragment_C(
            cta_mma0, cute::select<0, 1>(TileShape_MNK{}));
        cute::Tensor tStScale =
            cute::composition(tStS, cute::make_layout(cute::Int<kBlockM>{}));
        cute::Tensor tScS = cta_mma0.partition_C(
            cute::make_identity_tensor(cute::select<0, 1>(TileShape_MNK{})));

        constexpr int tilePlikeFP32 =
            kBlockN / 32 * cute::sizeof_bits_v<Element>;
        cute::Tensor tStP = tStS.compose(cute::make_layout(cute::make_shape(
            cute::Int<kBlockM>{}, cute::Int<tilePlikeFP32>{})));

        cute::Tensor tScP = tScS.compose(cute::make_layout(cute::make_shape(
            cute::Int<kBlockM>{}, cute::Int<tilePlikeFP32>{})));

        tStS.data() = si_stage * kBlockN;
        tStP.data() = si_stage * kBlockN;

        using tmem_load_atom = cute::SM100_TMEM_LOAD_16dp64b32x;
        auto tiled_tmem_load = cute::make_tmem_copy(tmem_load_atom{}, tStS);
        auto thr_tmem_load = tiled_tmem_load.get_slice(thread_idx);
        cute::Tensor tStS_t2r = thr_tmem_load.partition_S(tStS);
        cute::Tensor tSrS_t2r = cute::make_tensor<ElementAcc>(
            cute::shape(thr_tmem_load.partition_D(tScS)));

        using tmem_store_atom = cute::SM100_TMEM_STORE_16dp64b16x;
        auto tiled_tmem_store = cute::make_tmem_copy(tmem_store_atom{}, tStP);
        auto thr_tmem_store = tiled_tmem_store.get_slice(thread_idx);
        cute::Tensor tStP_r2t = thr_tmem_store.partition_D(tStP);
        cute::Tensor tSrP_r2t_f32 = cute::make_tensor<float>(
            cute::shape(thr_tmem_store.partition_S(tScP)));
        cute::Tensor tSrP_r2t =
            cute::make_tensor(reinterpret_cast<Element *>(tSrP_r2t_f32.data()),
                              tSrS_t2r.layout());
        cute::wait_barrier(shared_storage.mbar_S_full[si_stage],
                           mma_si_consumer_phase);

        cute::copy(tiled_tmem_load, tStS_t2r, tSrS_t2r);
        if constexpr (NeedMask) {
            if (!CustomMask) {
                int mask_row_idx =
                    (lane_idx / 4) + 8 * (lane_idx % 2) + warp_idx * 16;
                const int col_offset = lane_idx % 4 / 2;
                mask_row_idx =
                    mainloop_params.qhead_per_khead_divmod.divide(mask_row_idx);
                const int col_limit = mask_row_idx + 1 + col_base - col_offset;
                for (int j = 0; j < cute::size(tSrS_t2r); ++j) {
                    if (2 * j >= col_limit) {
                        tSrS_t2r(j) = -FLT_MAX;
                    }
                }
            }
        }

        float const acc_scale = softmax.update_row_max<IsFirst, true>(tSrS_t2r);
        if constexpr (!IsFirst) {
            // The previous PV GEMM must finish writing TMEM 256-384 before
            // correction_rescale modifies it.
            cute::wait_barrier(shared_storage.mbar_O_full[0], pv_done_phase);
            bool const should_rescale =
                __ballot_sync(0xffffffff, acc_scale < 1.0f) != 0;
            if (should_rescale) {
                correction_rescale(thread_idx, acc_scale);
            }
            pv_done_phase ^= 1;
        }

        softmax.scale_subtract_rowmax(tSrS_t2r);
        softmax.apply_exp2_convert<Element>(tSrS_t2r, tSrP_r2t);

        cute::copy(tiled_tmem_store, tSrP_r2t_f32, tStP_r2t);
        cutlass::arch::fence_view_async_tmem_store();
        cute::arrive_barrier(shared_storage.mbar_P_full_O_rescaled[si_stage]);
        softmax.update_row_sum<IsFirst>(tSrS_t2r, acc_scale);
    }

    template <typename SharedStorage>
    CUTLASS_DEVICE void softmax(Params const &mainloop_params,
                                SharedStorage &shared_storage, const int *mask,
                                const int lane_idx, const int warp_idx,
                                const int thread_idx, const int n_block_max,
                                const int m_block, const int seqlen_q,
                                const int seqlen_k,
                                const int num_kv_blocks_per_cta, const int bidb,
                                const int bidh, const int iter_phase = 0) {
        Softmax softmax(mainloop_params.softmax_scale_log2);

        // S_full slots alternate, so the consumer phase flips once per pair of
        // KV tiles.
        int mma_si_consumer_phase = 0;
        int si_stage = 0;
        // pv_done: stage 0 for even PV GEMMs, stage 1 for odd ones.
        int pv_done_phase = 0;

        const int num_kv_tokens_per_cta = kBlockN * num_kv_blocks_per_cta;
        int col_base = -n_block_max * kBlockN + kBlockN -
                       m_block * num_kv_tokens_per_cta + seqlen_k - seqlen_q;

        softmax_step<true, true>(mainloop_params, shared_storage, mask, softmax,
                                 seqlen_k, col_base, lane_idx, warp_idx,
                                 thread_idx, mma_si_consumer_phase, si_stage,
                                 pv_done_phase);
        si_stage ^= 1;
        int n_block_mask = seqlen_q > 1 && n_block_max > 1;

        if (n_block_mask) {
            softmax_step<false, true>(
                mainloop_params, shared_storage, mask, softmax, seqlen_k,
                col_base + kBlockN, lane_idx, warp_idx, thread_idx,
                mma_si_consumer_phase, si_stage, pv_done_phase);
            si_stage ^= 1;
            mma_si_consumer_phase ^= 1;
        }

        // Unrolled like load() and mma(). The fold is a multiple of 4, so
        // si_stage and the consumer phase return to their entry values at every
        // fold boundary.
        const int num_steps = n_block_max - 1 - n_block_mask;
        for (int base = 0; base < num_steps; base += kNumUnrolledKvBlocks) {
            range_constexpr<kNumUnrolledKvBlocks>([&](auto k) {
                if (base + k < num_steps) {
                    softmax_step<false, false>(
                        mainloop_params, shared_storage, mask, softmax,
                        seqlen_k, col_base, lane_idx, warp_idx, thread_idx,
                        mma_si_consumer_phase, si_stage, pv_done_phase);
                    si_stage ^= 1;
                    // base is even, so iter = n_block_mask + base + k is even
                    // exactly when (k & 1) == n_block_mask.
                    if (n_block_mask == (k & 1)) {
                        mma_si_consumer_phase ^= 1;
                    }
                }
            });
        }

        softmax.row_sum += __shfl_xor_sync(0xffffffff, softmax.row_sum, 2);
        // Split-KV leaves normalization to the reduce kernel.
        const float output_scale =
            kSplitKV ? 1.0f : 1.0f / (softmax.row_sum + 0.0000001f);
        correction_epilogue(mainloop_params, shared_storage, thread_idx, bidb,
                            bidh, seqlen_q, m_block, n_block_max, iter_phase,
                            output_scale);

        if constexpr (kSplitKV) {
            const int max_sum_offset =
                (bidb * mainloop_params.max_num_partitions + m_block) *
                    mainloop_params.kv_head_num * kBlockM +
                bidh * kBlockM + thread_idx / 4 * 2 + thread_idx % 2;
            if (thread_idx % 4 < 2) {
                mainloop_params.max_sum[max_sum_offset] = make_float2(
                    softmax.row_max * mainloop_params.softmax_scale_log2,
                    softmax.row_sum);
            }
        }
    }

    CUTLASS_DEVICE void correction_rescale(const int tidx, float const scale) {
        TiledMma1 tiled_mma1;
        cute::ThrMMA cta_mma1 = tiled_mma1.get_slice(0);
        cute::Tensor tOtO = cute::partition_fragment_C(
            cta_mma1, cute::select<0, 2>(TileShape_MNK{}));
        tOtO.data() = 2 * kBlockN;
        cute::Tensor tOcO = cta_mma1.partition_C(
            cute::make_identity_tensor(cute::select<0, 2>(TileShape_MNK{})));
        constexpr int corr_tile_size = 32;
        using tmem_load_atom = cute::SM100_TMEM_LOAD_16dp64b16x;
        using tmem_store_atom = cute::SM100_TMEM_STORE_16dp64b16x;
        cute::Tensor tOtO_i = tOtO.compose(cute::make_layout(cute::make_shape(
            cute::Int<kBlockM>{}, cute::Int<corr_tile_size>{})));
        cute::Tensor tOcO_i = cute::composition(
            tOcO, cute::make_layout(cute::make_shape(
                      cute::Int<kBlockM>{}, cute::Int<corr_tile_size>{})));
        auto tiled_tmem_load = cute::make_tmem_copy(tmem_load_atom{}, tOtO_i);
        auto tiled_tmem_store = cute::make_tmem_copy(tmem_store_atom{}, tOtO_i);
        auto thr_tmem_load = tiled_tmem_load.get_slice(tidx);
        auto thr_tmem_store = tiled_tmem_store.get_slice(tidx);
        cute::Tensor tOtO_t2r = thr_tmem_load.partition_S(tOtO_i);
        auto tOrO_t2r_shape = cute::shape(thr_tmem_load.partition_D(tOcO_i));
        cute::Tensor tOtO_r2t = thr_tmem_store.partition_D(tOtO_i);
        constexpr int frg_count = kHeadDim / corr_tile_size;

        float2 const scale2 = {scale, scale};

#pragma unroll
        for (int i = 0; i < frg_count; ++i) {
            cute::Tensor tOrO_frg = cute::make_tensor<float>(tOrO_t2r_shape);
            cute::Tensor tOtO_t2r_i = cute::make_tensor(
                tOtO_t2r.data() + i * corr_tile_size, tOtO_t2r.layout());
            cute::copy(tiled_tmem_load, tOtO_t2r_i, tOrO_frg);
#pragma unroll
            for (int j = 0; j < cute::size(tOrO_frg); j += 2) {
                mul_packed_f32x2(*reinterpret_cast<float2 *>(&tOrO_frg(j)),
                                 scale2,
                                 *reinterpret_cast<float2 *>(&tOrO_frg(j)));
            }
            cute::Tensor tOtO_r2t_i = cute::make_tensor(
                tOtO_r2t.data() + i * corr_tile_size, tOtO_r2t.layout());
            cute::copy(tiled_tmem_store, tOrO_frg, tOtO_r2t_i);
            cutlass::arch::fence_view_async_tmem_store();
        }
    }

    template <typename SharedStorage>
    CUTLASS_DEVICE void epilogue_s2g(Params const &mainloop_params,
                                     SharedStorage &shared_storage,
                                     const int seqlen_q, const int bidb,
                                     const int bidh, const int tidx,
                                     const int n_block) {
        constexpr int kNumVecElem =
            cute::ceil_div(128, cute::sizeof_bits_v<Element>);
        constexpr int kOutputRowStride = kHeadDim / (128 / kBlockM);
        constexpr int kNumThreadsPerRow = kOutputRowStride / kNumVecElem;
        static_assert(cutlass::NumThreadsPerWarpGroup % kNumThreadsPerRow == 0);

        constexpr int kNumRows =
            cutlass::NumThreadsPerWarpGroup / kNumThreadsPerRow;

        Element *O_ptr =
            mainloop_params.partition_o_ptr +
            ((bidb * mainloop_params.max_num_partitions + n_block) *
                 mainloop_params.kv_head_num +
             bidh) *
                (kBlockM * kHeadDim);

        const int store_row_idx = tidx / kNumThreadsPerRow;
        const int store_col_idx = tidx % kNumThreadsPerRow;

        const int active_rows =
            seqlen_q * (mainloop_params.head_num / mainloop_params.kv_head_num);

        for (int mem_row = store_row_idx; mem_row < kBlockM * 2;
             mem_row += kNumRows) {
            const int real_row =
                (mem_row & -32) | ((mem_row & 31) >> 1) | ((mem_row & 1) << 4);
            if (real_row / 2 >= active_rows) break;
            auto *gmem_o_ptr = O_ptr + real_row * kOutputRowStride +
                               store_col_idx * kNumVecElem;
            auto *smem_o_ptr =
                shared_storage.smem_o.data() + mem_row * kOutputRowStride +
                ((store_col_idx + mem_row) % kNumThreadsPerRow) * kNumVecElem;
            *reinterpret_cast<float4 *>(gmem_o_ptr) =
                *reinterpret_cast<float4 *>(smem_o_ptr);
        }
    }

    template <typename SharedStorage>
    CUTLASS_DEVICE void epilogue_s2g_nosplit(Params const &mainloop_params,
                                             SharedStorage &shared_storage,
                                             const int seqlen_q, const int bidb,
                                             const int bidh,
                                             const int thread_idx) {
        static_assert(kBlockM == 64);
        constexpr int kNumVecElem =
            cute::ceil_div(128, cute::sizeof_bits_v<Element>);
        constexpr int kNumThreadsPerRow =
            kHeadDim / (128 / kBlockM) / kNumVecElem;
        static_assert(cutlass::NumThreadsPerWarpGroup % kNumThreadsPerRow == 0);
        constexpr int kNumRows =
            cutlass::NumThreadsPerWarpGroup / kNumThreadsPerRow;
        constexpr int kHeadDimStride = kHeadDim / (128 / kBlockM);

        auto const &gqa_divmod = mainloop_params.qhead_per_khead_divmod;
        const int gqa_ratio = gqa_divmod.divisor;
        const int store_row_idx = thread_idx / kNumThreadsPerRow;
        const int store_col_idx = thread_idx % kNumThreadsPerRow;

        for (int mem_row = store_row_idx; mem_row < 128; mem_row += kNumRows) {
            const int lane_idx = mem_row % 32;
            const int warp_idx = mem_row / 32;
            const int logical_row =
                (lane_idx / 4) + 8 * (lane_idx % 2) + warp_idx * 16;
            if (logical_row >= seqlen_q * gqa_ratio) break;

            const int token_idx = gqa_divmod.divide(logical_row);
            const int gqa_head = logical_row - token_idx * gqa_ratio;
            const int head_idx = bidh * gqa_ratio + gqa_head;
            Element *out_ptr = mainloop_params.ptr_O +
                               (mainloop_params.cu_seq_q[bidb] + token_idx) *
                                   mainloop_params.head_num * kHeadDim +
                               head_idx * kHeadDim;

            const int headim_parity = (mem_row / 2) % 2;
            const Element *src =
                shared_storage.smem_o.data() + mem_row * kHeadDimStride +
                ((store_col_idx + mem_row) % kNumThreadsPerRow) * kNumVecElem;
#pragma unroll
            for (int j = 0; j < kNumVecElem; ++j) {
                out_ptr[headim_parity + 2 * (store_col_idx * kNumVecElem + j)] =
                    src[j];
            }
        }
    }

    // Split-KV stores the unnormalized partition tile for the reduce kernel;
    // no-split scales by 1/row_sum and writes the final output.
    template <typename SharedStorage>
    CUTLASS_DEVICE void correction_epilogue(
        Params const &mainloop_params, SharedStorage &shared_storage,
        const int tidx, const int bidb, const int bidh, const int seqlen_q,
        const int n_block, const int n_block_max, const int iter_phase,
        const float output_scale) {
        using PackedHalf = typename PackedHalf<Element>::Type;
        constexpr int corr_tile_size = 64 * 8 / cutlass::sizeof_bits_v<Element>;
        cute::Tensor sO =
            cute::make_tensor(cute::make_smem_ptr(shared_storage.smem_o.data()),
                              SmemLayoutO{})(cute::_, cute::_, 0);
        TiledMma1 tiled_mma1;
        cute::ThrMMA cta_mma1 = tiled_mma1.get_slice(0);

        cute::Tensor tOtO = cute::partition_fragment_C(
            cta_mma1, cute::select<0, 2>(TileShape_MNK{}));
        cute::Tensor tOsO = cta_mma1.partition_C(sO);
        tOtO.data() = 2 * kBlockN;
        auto tOcO = cta_mma1.partition_C(
            cute::make_identity_tensor(cute::select<0, 2>(TileShape_MNK{})));

        auto tOtO_i = cute::logical_divide(
            tOtO, cute::make_layout(cute::make_shape(
                      cute::Int<kBlockM>{}, cute::Int<corr_tile_size>{})));
        auto tOcO_i = cute::logical_divide(
            tOcO, cute::make_layout(cute::make_shape(
                      cute::Int<kBlockM>{}, cute::Int<corr_tile_size>{})));
        auto tOsO_i = cute::logical_divide(
            tOsO, cute::make_layout(cute::make_shape(
                      cute::Int<kBlockM>{}, cute::Int<corr_tile_size>{})));

        using tmem_load_atom = cute::SM100_TMEM_LOAD_16dp64b16x;
        using smem_store_atom =
            cute::Copy_Atom<cute::UniversalCopy<Element>, Element>;
        auto tiled_tmem_load =
            cute::make_tmem_copy(tmem_load_atom{}, tOtO_i(cute::_, 0));
        auto tiled_smem_store =
            cute::make_tiled_copy_D(smem_store_atom{}, tiled_tmem_load);
        auto thr_tmem_load = tiled_tmem_load.get_slice(tidx);
        cute::Tensor tOtO_t2r = thr_tmem_load.partition_S(tOtO_i);
        cute::Tensor tOcO_t2r = thr_tmem_load.partition_D(tOcO_i);
        cute::Tensor tOsO_s2r = thr_tmem_load.partition_D(tOsO_i);

        static constexpr int kNumVecElem =
            cute::ceil_div(128, cute::sizeof_bits_v<Element>);
        static constexpr int kNumThreadsPerRow =
            kHeadDim / (128 / kBlockM) / kNumVecElem;
        static_assert(cutlass::NumThreadsPerWarpGroup % kNumThreadsPerRow == 0);
        cute::wait_barrier(shared_storage.mbar_O_full[0], 1 - n_block_max % 2);
#pragma unroll
        for (int i = 0; i < kHeadDim / corr_tile_size; ++i) {
            cute::Tensor tOrO_frg =
                cute::make_tensor<float>(cute::shape(tOcO_t2r(cute::_, 0, i)));
            cute::Tensor tOsO_r2s_i = tOsO_s2r(cute::_, 0, i);
            cute::Tensor tOrO_frg_cvt = cute::make_tensor<Element>(
                cute::shape(tOcO_t2r(cute::_, 0, i)));
            cute::copy(tiled_tmem_load, tOtO_t2r(cute::_, 0, i), tOrO_frg);
#pragma unroll
            for (int j = 0; j < cute::size(tOrO_frg); j += 2) {
                if constexpr (kSplitKV) {
                    tOrO_frg_cvt(j) = static_cast<Element>(tOrO_frg(j));
                    tOrO_frg_cvt(j + 1) = static_cast<Element>(tOrO_frg(j + 1));
                } else {
                    tOrO_frg_cvt(j) =
                        static_cast<Element>(tOrO_frg(j) * output_scale);
                    tOrO_frg_cvt(j + 1) =
                        static_cast<Element>(tOrO_frg(j + 1) * output_scale);
                }
            }
            *reinterpret_cast<float4 *>(shared_storage.smem_o.data() +
                                        tidx * (kHeadDim / (128 / kBlockM)) +
                                        (tidx + 2 * i) % kNumThreadsPerRow *
                                            kNumVecElem) =
                *reinterpret_cast<float4 *>(tOrO_frg_cvt.data());
            *reinterpret_cast<float4 *>(shared_storage.smem_o.data() +
                                        tidx * (kHeadDim / (128 / kBlockM)) +
                                        (tidx + 2 * i + 1) % kNumThreadsPerRow *
                                            kNumVecElem) =
                *reinterpret_cast<float4 *>(tOrO_frg_cvt.data() + kNumVecElem);
        }

        cutlass::arch::fence_view_async_shared();

        cutlass::arch::NamedBarrier::sync(cutlass::NumThreadsPerWarpGroup, 1);
        if constexpr (kSplitKV) {
            epilogue_s2g(mainloop_params, shared_storage, seqlen_q, bidb, bidh,
                         tidx, n_block);
        } else {
            epilogue_s2g_nosplit(mainloop_params, shared_storage, seqlen_q,
                                 bidb, bidh, tidx);
        }
        cute::arrive_barrier(shared_storage.mbar_tmem_dealloc);
    }
};

template <typename KernelTraits>
struct CollectiveMainloopFwdEncoder {
    static constexpr int kNumStagesKV = KernelTraits::kNumStagesKV;
    static constexpr int kNumStagesQ = KernelTraits::kNumStagesQ;
    static constexpr int kHeadDim = KernelTraits::kHeadDim;
    static constexpr int kBlockM = KernelTraits::kBlockM;
    static constexpr int kBlockN = KernelTraits::kBlockN;
    static constexpr bool CustomMask = KernelTraits::CustomMask;

    static_assert(kBlockN == 128);

    using Element = typename KernelTraits::Element;
    using ElementAcc = typename KernelTraits::ElementAcc;
    using TileShape_MNK = typename KernelTraits::TileShape_MNK;
    using ClusterShape = typename KernelTraits::ClusterShape;
    using MainloopPipeline = typename KernelTraits::MainloopPipeline;
    using PipelineState = typename MainloopPipeline::PipelineState;

    using GmemTiledCopyQ = cute::SM90_TMA_LOAD;
    using GmemTiledCopyKV = decltype(cutlass::gemm::collective::detail::
                                         sm90_cluster_shape_to_tma_atom(
                                             cute::shape<0>(ClusterShape{})));
    using GmemTiledCopyO = cute::SM90_TMA_STORE;

    using SmemLayoutQ = typename KernelTraits::SmemLayoutQ;
    using SmemLayoutK = typename KernelTraits::SmemLayoutK;
    using SmemLayoutV = typename KernelTraits::SmemLayoutV;
    using SmemLayoutVt = typename KernelTraits::SmemLayoutVt;
    using SmemLayoutS = typename KernelTraits::SmemLayoutS;
    using SmemLayoutO = typename KernelTraits::SmemLayoutO;
    using TiledMma0 = typename KernelTraits::TiledMma0;
    using TiledMma1 = typename KernelTraits::TiledMma1;

    using ShapeT = cute::Shape<int32_t, int32_t, int32_t>;
    using StrideT = cute::Shape<int32_t, cute::_1, int32_t>;
    using LayoutT = cute::Layout<ShapeT, StrideT>;

    using TMA_Q = decltype(cute::make_tma_copy(
        GmemTiledCopyQ{},
        cute::make_tensor(
            cute::make_gmem_ptr(static_cast<Element const *>(nullptr)),
            cute::repeat_like(StrideT{}, int32_t(0)), StrideT{}),
        SmemLayoutQ{}(cute::_, cute::_, cute::_0{}),
        cute::select<0, 2>(TileShape_MNK{}), cute::_1{}));

    using TMA_KV = decltype(cute::make_tma_copy(
        GmemTiledCopyKV{},
        cute::make_tensor(
            cute::make_gmem_ptr(static_cast<Element const *>(nullptr)),
            cute::repeat_like(StrideT{}, int32_t(0)), StrideT{}),
        cute::take<0, 2>(SmemLayoutK{}), cute::select<1, 2>(TileShape_MNK{}),
        cute::size<0>(ClusterShape{})));

    using TMA_O = decltype(cute::make_tma_copy(
        GmemTiledCopyO{},
        cute::make_tensor(
            cute::make_gmem_ptr(static_cast<Element const *>(nullptr)),
            cute::repeat_like(StrideT{}, int32_t(0)), StrideT{}),
        SmemLayoutO{}(cute::_, cute::_, cute::_0{}),
        cute::select<0, 2>(TileShape_MNK{}), cute::size<0>(ClusterShape{})));

    static constexpr uint32_t TmaTransactionBytesQ =
        static_cast<uint32_t>(cute::size(cute::take<0, 2>(SmemLayoutQ{})) *
                              cutlass::sizeof_bits_v<Element> / 8);
    static constexpr uint32_t TmaTransactionBytesK =
        static_cast<uint32_t>(cute::size(cute::take<0, 2>(SmemLayoutK{})) *
                              cutlass::sizeof_bits_v<Element> / 8);

    struct Arguments {
        Element const *ptr_Q;
        LayoutT layout_Q;
        Element const *ptr_K;
        LayoutT layout_K;
        Element const *ptr_V;
        LayoutT layout_V;
        Element *ptr_O;
        const int *cu_seq_q;
        const int *cu_seq_k;
        const int *mask;
        const int O_stride;
        float const softmax_scale_log2;
    };

    struct Params {
        LayoutT layout_Q;
        LayoutT layout_K;
        LayoutT layout_V;
        cutlass::FastDivmod qhead_per_khead_divmod;
        TMA_Q tma_load_Q;
        TMA_KV tma_load_K, tma_load_V;
        Element *ptr_O;
        const int *cu_seq_q;
        const int *cu_seq_k;
        const int *mask;
        const int O_stride;
        float const softmax_scale_log2;
    };

    static Params to_underlying_arguments(Arguments const &args) {
        cute::Tensor mQ =
            cute::make_tensor(cute::make_gmem_ptr(args.ptr_Q), args.layout_Q);
        TMA_Q tma_load_Q = cute::make_tma_copy(
            GmemTiledCopyQ{}, mQ, SmemLayoutQ{}(cute::_, cute::_, cute::_0{}),
            cute::select<0, 2>(TileShape_MNK{}), cute::_1{});
        cute::Tensor mK =
            cute::make_tensor(cute::make_gmem_ptr(args.ptr_K), args.layout_K);
        TMA_KV tma_load_K = cute::make_tma_copy(
            GmemTiledCopyKV{}, mK, SmemLayoutK{}(cute::_, cute::_, cute::_0{}),
            cute::select<1, 2>(TileShape_MNK{}), cute::size<0>(ClusterShape{}));
        cute::Tensor mV =
            cute::make_tensor(cute::make_gmem_ptr(args.ptr_V), args.layout_V);
        TMA_KV tma_load_V = cute::make_tma_copy(
            GmemTiledCopyKV{}, mV, SmemLayoutV{}(cute::_, cute::_, cute::_0{}),
            cute::select<1, 2>(TileShape_MNK{}), cute::size<0>(ClusterShape{}));

        return {args.layout_Q,
                args.layout_K,
                args.layout_V,
                cutlass::FastDivmod(
                    cute::ceil_div(cute::get<2>(args.layout_Q.shape()),
                                   cute::get<2>(args.layout_K.shape()))),
                tma_load_Q,
                tma_load_K,
                tma_load_V,
                args.ptr_O,
                args.cu_seq_q,
                args.cu_seq_k,
                args.mask,
                args.O_stride,
                args.softmax_scale_log2};
    }

    CUTLASS_DEVICE
    static void prefetch_tma_descriptors(Params const &mainloop_params) {
        cute::prefetch_tma_descriptor(
            mainloop_params.tma_load_Q.get_tma_descriptor());
        cute::prefetch_tma_descriptor(
            mainloop_params.tma_load_K.get_tma_descriptor());
        cute::prefetch_tma_descriptor(
            mainloop_params.tma_load_V.get_tma_descriptor());
    }

    CUTLASS_DEVICE int get_n_block_max(const int seqlen_q, const int seqlen_k,
                                       const int m_block) {
        const int n_block_max = min((seqlen_k + kBlockN - 1) / kBlockN,
                                    ((m_block + 1) * kBlockM * kNumStagesQ +
                                     seqlen_k - seqlen_q + kBlockN - 1) /
                                        kBlockN);
        return n_block_max;
    }

    template <typename MTensor, typename Shape>
    CUTLASS_DEVICE auto get_local_tile_tensor(MTensor const &m_tensor,
                                              Shape const &tile_shape,
                                              const int *cu_seq_len,
                                              const int bidh, const int bidb,
                                              const int actual_seq_len) const {
        auto g_offset =
            cute::local_tile(m_tensor(cute::_, cute::_, bidh),
                             cute::make_shape(1, cute::get<1>(tile_shape)),
                             cute::make_coord(cu_seq_len[bidb], cute::_0{}));
        auto g_sequence = cute::make_tensor(
            g_offset.data(),
            cute::make_layout(
                cute::make_shape(actual_seq_len, cute::get<1>(tile_shape)),
                g_offset.stride()));
        auto g_tensor = cute::local_tile(g_sequence, tile_shape,
                                         cute::make_coord(cute::_, cute::_0{}));
        return g_tensor;
    }

    template <typename SharedStorage>
    CUTLASS_DEVICE void load(Params const &mainloop_params,
                             MainloopPipeline pipeline_k,
                             MainloopPipeline pipeline_v,
                             SharedStorage &shared_storage, const int lane_idx,
                             const int n_block_max, const int m_block,
                             const int seqlen_q, const int seqlen_k,
                             const int bidh, const int bidb) {
        cute::Tensor sQ = cute::make_tensor(
            cute::make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutQ{});
        cute::Tensor sK = cute::make_tensor(
            cute::make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutK{});
        cute::Tensor sV = cute::make_tensor(
            cute::make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutV{});

        cute::Tensor mQ = mainloop_params.tma_load_Q.get_tma_tensor(
            mainloop_params.layout_Q.shape());
        cute::Tensor mK = mainloop_params.tma_load_K.get_tma_tensor(
            mainloop_params.layout_K.shape());
        cute::Tensor mV = mainloop_params.tma_load_V.get_tma_tensor(
            mainloop_params.layout_V.shape());

        PipelineState k_producer_state =
            cutlass::make_producer_start_state<MainloopPipeline>();
        PipelineState v_producer_state =
            cutlass::make_producer_start_state<MainloopPipeline>();

        const int bidh_kv = mainloop_params.qhead_per_khead_divmod.divide(bidh);

        cute::Tensor gQ = get_local_tile_tensor(
            mQ, cute::select<0, 2>(TileShape_MNK{}), mainloop_params.cu_seq_q,
            bidh, bidb, seqlen_q);
        cute::Tensor gK = get_local_tile_tensor(
            mK, cute::select<1, 2>(TileShape_MNK{}), mainloop_params.cu_seq_k,
            bidh_kv, bidb, seqlen_k);
        cute::Tensor gV = get_local_tile_tensor(
            mV, cute::select<1, 2>(TileShape_MNK{}), mainloop_params.cu_seq_k,
            bidh_kv, bidb, seqlen_k);

        auto [tQgQ, tQsQ] = cute::tma_partition(
            mainloop_params.tma_load_Q, cute::_0{}, cute::Layout<cute::_1>{},
            cute::group_modes<0, 2>(sQ), cute::group_modes<0, 2>(gQ));
        auto [tKgK, tKsK] = cute::tma_partition(
            mainloop_params.tma_load_K, cute::_0{}, cute::Layout<cute::_1>{},
            cute::group_modes<0, 2>(sK), cute::group_modes<0, 2>(gK));
        auto [tVgV, tVsV] = cute::tma_partition(
            mainloop_params.tma_load_V, cute::_0{}, cute::Layout<cute::_1>{},
            cute::group_modes<0, 2>(sV), cute::group_modes<0, 2>(gV));

        auto load_Q = [&](const int q_stage, const int src_idx) {
            cute::set_barrier_transaction_bytes(
                shared_storage.mbar_load_q_full[q_stage], TmaTransactionBytesQ);
            cute::copy(mainloop_params.tma_load_Q.with(
                           shared_storage.mbar_load_q_full[q_stage], 0),
                       tQgQ(cute::_, src_idx), tQsQ(cute::_, q_stage));
        };

        auto load_K = [&](const int src_idx) {
            pipeline_k.producer_acquire(k_producer_state);
            cute::copy(
                mainloop_params.tma_load_K.with(
                    *pipeline_k.producer_get_barrier(k_producer_state), 0),
                tKgK(cute::_, src_idx),
                tKsK(cute::_, k_producer_state.index()));
            ++k_producer_state;
        };

        auto load_V = [&](const int src_idx) {
            pipeline_v.producer_acquire(v_producer_state);
            cute::copy(
                mainloop_params.tma_load_V.with(
                    *pipeline_v.producer_get_barrier(v_producer_state), 0),
                tVgV(cute::_, src_idx),
                tVsV(cute::_, v_producer_state.index()));
            ++v_producer_state;
        };

        if (cute::elect_one_sync()) {
            load_Q(0, kNumStagesQ * m_block);
            load_K(n_block_max - 1);
            load_Q(1, kNumStagesQ * m_block + 1);
            load_V(n_block_max - 1);

#pragma unroll 1
            for (int n_block = 2; n_block <= n_block_max; ++n_block) {
                load_K(n_block_max - n_block);
                load_V(n_block_max - n_block);
            }
        }
    }

    template <typename SharedStorage>
    CUTLASS_DEVICE void mma(Params const &mainloop_params,
                            MainloopPipeline pipeline_k,
                            MainloopPipeline pipeline_v,
                            SharedStorage &shared_storage, const int lane_idx,
                            const int n_block_max) {
        cute::Tensor sQ = cute::make_tensor(
            cute::make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutQ{});
        cute::Tensor sK = cute::make_tensor(
            cute::make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutK{});
        cute::Tensor sVt = cute::make_tensor(
            cute::make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutVt{});
        cute::Tensor sS = cute::make_tensor(
            cute::make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutS{});

        TiledMma0 tiled_mma0;
        TiledMma1 tiled_mma1[kNumStagesQ];

        cute::ThrMMA cta_mma0 = tiled_mma0.get_slice(0);
        cute::ThrMMA cta_mma1 = tiled_mma1[0].get_slice(0);

        cute::Tensor tSrQ = cta_mma0.partition_fragment_A(sQ);
        cute::Tensor tSrK = cta_mma0.partition_fragment_B(sK);
        cute::Tensor tSrV = cta_mma1.partition_fragment_B(sVt);
        cute::Tensor tPtP = cta_mma1.partition_fragment_A(sS);

        cute::Tensor tStS = cute::partition_fragment_C(
            cta_mma0, cute::select<0, 1>(TileShape_MNK{}));
        cute::Tensor tOtO = cute::partition_fragment_C(
            cta_mma1, cute::select<0, 2>(TileShape_MNK{}));

        int mma_q_consumer_phase = 0;
        PipelineState mma_k_consumer_state, mma_v_consumer_state;
        int P_full_O_rescaled_phase = 0;

        range_constexpr<kNumStagesQ>([&](auto q_stage) {
            tiled_mma1[q_stage].accumulate_ = cute::UMMA::ScaleOut::Zero;
            cute::wait_barrier(shared_storage.mbar_load_q_full[q_stage],
                               mma_q_consumer_phase);
            if constexpr (q_stage == 0) {
                pipeline_k.consumer_wait(mma_k_consumer_state);
            }
            constexpr int data_offset = q_stage * kBlockN;
            tStS.data() = data_offset;
            gemm<true>(
                tiled_mma0, tSrQ(cute::_, cute::_, cute::_, q_stage),
                tSrK(cute::_, cute::_, cute::_, mma_k_consumer_state.index()),
                tStS);
            cutlass::arch::umma_arrive(&shared_storage.mbar_S_full[q_stage]);
        });
        mma_q_consumer_phase ^= 1;
        pipeline_k.consumer_release(mma_k_consumer_state);
        ++mma_k_consumer_state;
#pragma unroll 1
        for (int n_block = 0; n_block <= n_block_max - 2; ++n_block) {
            pipeline_v.consumer_wait(mma_v_consumer_state);
            range_constexpr<kNumStagesQ>([&](auto q_stage) {
                cute::wait_barrier(
                    shared_storage.mbar_P_full_O_rescaled[q_stage],
                    P_full_O_rescaled_phase);

                constexpr int tp_data_offset = q_stage * kBlockN;
                constexpr int to_data_offset =
                    kNumStagesQ * kBlockN + q_stage * kHeadDim;
                tPtP.data() = tp_data_offset;
                tOtO.data() = to_data_offset;

                gemm(tiled_mma1[q_stage], tPtP,
                     tSrV(cute::_, cute::_, cute::_,
                          mma_v_consumer_state.index()),
                     tOtO, shared_storage.mbar_P_full_2[q_stage],
                     P_full_O_rescaled_phase);

                if constexpr (q_stage == kNumStagesQ - 1) {
                    pipeline_v.consumer_release(mma_v_consumer_state);
                    ++mma_v_consumer_state;
                }

                if constexpr (q_stage == 0) {
                    pipeline_k.consumer_wait(mma_k_consumer_state);
                }

                tStS.data() = tp_data_offset;
                gemm<true>(tiled_mma0, tSrQ(cute::_, cute::_, cute::_, q_stage),
                           tSrK(cute::_, cute::_, cute::_,
                                mma_k_consumer_state.index()),
                           tStS);
                cutlass::arch::umma_arrive(
                    &shared_storage.mbar_S_full[q_stage]);
            });

            pipeline_k.consumer_release(mma_k_consumer_state);
            ++mma_k_consumer_state;
            P_full_O_rescaled_phase ^= 1;
        }

        pipeline_v.consumer_wait(mma_v_consumer_state);

        range_constexpr<kNumStagesQ>([&](auto q_stage) {
            cute::wait_barrier(shared_storage.mbar_P_full_O_rescaled[q_stage],
                               P_full_O_rescaled_phase);
            tPtP.data() = q_stage * kBlockN;
            tOtO.data() = kNumStagesQ * kBlockN + q_stage * kHeadDim;
            gemm(tiled_mma1[q_stage], tPtP,
                 tSrV(cute::_, cute::_, cute::_, mma_v_consumer_state.index()),
                 tOtO, shared_storage.mbar_P_full_2[q_stage],
                 P_full_O_rescaled_phase);
            cutlass::arch::umma_arrive(&shared_storage.mbar_O_full[q_stage]);
        });
        P_full_O_rescaled_phase ^= 1;
        pipeline_v.consumer_release(mma_v_consumer_state);
        ++mma_v_consumer_state;
    }

    template <bool IsFirst, bool NeedMask, typename SharedStorage>
    CUTLASS_DEVICE void softmax_step(
        Params const &mainloop_params, SharedStorage &shared_storage,
        const int *mask, Softmax &softmax, const int row_base,
        const int col_base, const int q_stage, const int warp_group_thread_idx,
        const int mma_si_consumer_phase, const int si_corr_producer_phase) {
        TiledMma0 tiled_mma0;
        cute::ThrMMA cta_mma0 = tiled_mma0.get_slice(0);
        cute::Tensor tStS = cute::partition_fragment_C(
            cta_mma0, cute::select<0, 1>(TileShape_MNK{}));
        tStS.data() = kBlockN * q_stage;
        cute::Tensor tStScale =
            cute::composition(tStS, cute::make_layout(cute::Int<kBlockM>{}));
        cute::Tensor tScS = cta_mma0.partition_C(
            cute::make_identity_tensor(cute::select<0, 1>(TileShape_MNK{})));

        constexpr int tilePlikeFP32 =
            kBlockN / 32 * cute::sizeof_bits_v<Element>;
        cute::Tensor tStP = tStS.compose(cute::make_layout(cute::make_shape(
            cute::Int<kBlockM>{}, cute::Int<tilePlikeFP32>{})));
        tStP.data() = kBlockN * q_stage;
        cute::Tensor tScP = tScS.compose(cute::make_layout(cute::make_shape(
            cute::Int<kBlockM>{}, cute::Int<tilePlikeFP32>{})));

        using tmem_load_atom = cute::SM100_TMEM_LOAD_32dp32b32x;
        auto tiled_tmem_load = cute::make_tmem_copy(tmem_load_atom{}, tStS);
        auto thr_tmem_load = tiled_tmem_load.get_slice(warp_group_thread_idx);
        cute::Tensor tStS_t2r = thr_tmem_load.partition_S(tStS);
        cute::Tensor tSrS_t2r = cute::make_tensor<ElementAcc>(
            cute::shape(thr_tmem_load.partition_D(tScS)));

        using tmem_store_atom = cute::SM100_TMEM_STORE_32dp32b16x;
        auto tiled_tmem_store = cute::make_tmem_copy(tmem_store_atom{}, tStP);
        auto thr_tmem_store = tiled_tmem_store.get_slice(warp_group_thread_idx);
        cute::Tensor tStP_r2t = thr_tmem_store.partition_D(tStP);
        cute::Tensor tSrP_r2t_f32 = cute::make_tensor<float>(
            cute::shape(thr_tmem_store.partition_S(tScP)));
        cute::Tensor tSrP_r2t =
            cute::make_tensor(reinterpret_cast<Element *>(tSrP_r2t_f32.data()),
                              tSrS_t2r.layout());

        cute::wait_barrier(shared_storage.mbar_S_full[q_stage],
                           mma_si_consumer_phase);

        cute::copy(tiled_tmem_load, tStS_t2r, tSrS_t2r);

        if constexpr (NeedMask) {
            if constexpr (CustomMask) {
                const int col_limit =
                    mask[warp_group_thread_idx] - col_base + q_stage * kBlockN;
                for (int j = 0; j < cute::size(tSrS_t2r); ++j) {
                    if (j >= col_limit) {
                        tSrS_t2r(j) = -FLT_MAX;
                    }
                }
            } else {
                const int col_limit =
                    row_base + warp_group_thread_idx - col_base;
                for (int j = 0; j < cute::size(tSrS_t2r); ++j) {
                    if (j > col_limit) {
                        tSrS_t2r(j) = -FLT_MAX;
                    }
                }
            }
        }

        float const acc_scale =
            softmax.update_row_max<IsFirst, false>(tSrS_t2r);
        if constexpr (!IsFirst) {
            shared_storage.sScale[warp_group_thread_idx + q_stage * kBlockM] =
                acc_scale;
        }
        cute::arrive_barrier(shared_storage.mbar_softmax_corr_full[q_stage]);

        softmax.scale_subtract_rowmax(tSrS_t2r);
        softmax.apply_exp2_convert<Element>(tSrS_t2r, tSrP_r2t);

        constexpr int phase = cute::size<2>(tStP_r2t) / 4 * 3;
        constexpr int copy_len = cute::size<2>(tStP_r2t) - phase;
        range_constexpr<phase>([&](auto index) {
            cute::copy(tiled_tmem_store, tSrP_r2t_f32(cute::_, cute::_, index),
                       tStP_r2t(cute::_, cute::_, index));
        });

        cutlass::arch::fence_view_async_tmem_store();
        cute::arrive_barrier(shared_storage.mbar_P_full_O_rescaled[q_stage]);

        range_constexpr<copy_len>([&](auto index) {
            cute::copy(tiled_tmem_store,
                       tSrP_r2t_f32(cute::_, cute::_, index + phase),
                       tStP_r2t(cute::_, cute::_, index + phase));
        });

        cutlass::arch::fence_view_async_tmem_store();
        cute::arrive_barrier(shared_storage.mbar_P_full_2[q_stage]);
        cute::wait_barrier(shared_storage.mbar_softmax_corr_empty[q_stage],
                           si_corr_producer_phase);
        softmax.update_row_sum<IsFirst>(tSrS_t2r, acc_scale);
    }

    template <typename SharedStorage>
    CUTLASS_DEVICE void softmax(Params const &mainloop_params,
                                SharedStorage &shared_storage, const int *mask,
                                const int lane_idx, const int warp_idx,
                                const int thread_idx, const int n_block_max,
                                const int m_block, const int seqlen_q,
                                const int seqlen_k, const int bidb,
                                const int bidh) {
        Softmax softmax(mainloop_params.softmax_scale_log2);

        const int q_stage = thread_idx / cutlass::NumThreadsPerWarpGroup;
        const int warp_group_thread_idx =
            thread_idx % cutlass::NumThreadsPerWarpGroup;

        int mma_si_consumer_phase = 0;
        int si_corr_producer_phase = 1;

        const int mask_col_base = seqlen_k - seqlen_q;
        const int row_base =
            mask_col_base + (m_block * kNumStagesQ + q_stage) * kBlockM;
        int col_base = n_block_max * kBlockN - kBlockN;

        cute::wait_barrier(shared_storage.mbar_softmax_corr_empty[q_stage],
                           si_corr_producer_phase);
        si_corr_producer_phase ^= 1;

        softmax_step<true, true>(mainloop_params, shared_storage, mask, softmax,
                                 row_base, col_base, q_stage,
                                 warp_group_thread_idx, mma_si_consumer_phase,
                                 si_corr_producer_phase);

        mma_si_consumer_phase ^= 1;
        si_corr_producer_phase ^= 1;
        int n_block_mask;
        if constexpr (CustomMask) {
            n_block_mask = n_block_max - mask[0] / kBlockN - 1;
            for (int n_block = 0; n_block < n_block_mask; ++n_block) {
                col_base -= kBlockN;
                softmax_step<false, true>(
                    mainloop_params, shared_storage, mask, softmax, row_base,
                    col_base, q_stage, warp_group_thread_idx,
                    mma_si_consumer_phase, si_corr_producer_phase);
                mma_si_consumer_phase ^= 1;
                si_corr_producer_phase ^= 1;
            }
        } else {
            if (n_block_max > 1) {
                softmax_step<false, true>(
                    mainloop_params, shared_storage, mask, softmax, row_base,
                    col_base - kBlockN, q_stage, warp_group_thread_idx,
                    mma_si_consumer_phase, si_corr_producer_phase);
                mma_si_consumer_phase ^= 1;
                si_corr_producer_phase ^= 1;
            }
        }

        if constexpr (CustomMask) {
#pragma unroll 1
            for (int n_block = 0; n_block < n_block_max - n_block_mask - 1;
                 ++n_block) {
                softmax_step<false, false>(
                    mainloop_params, shared_storage, mask, softmax, row_base,
                    col_base, q_stage, warp_group_thread_idx,
                    mma_si_consumer_phase, si_corr_producer_phase);
                mma_si_consumer_phase ^= 1;
                si_corr_producer_phase ^= 1;
            }
        } else {
#pragma unroll 1
            for (int n_block = 0; n_block < n_block_max - 2; ++n_block) {
                softmax_step<false, false>(
                    mainloop_params, shared_storage, mask, softmax, row_base,
                    col_base, q_stage, warp_group_thread_idx,
                    mma_si_consumer_phase, si_corr_producer_phase);
                mma_si_consumer_phase ^= 1;
                si_corr_producer_phase ^= 1;
            }
        }

        shared_storage.sScale[warp_group_thread_idx + q_stage * kBlockM] =
            softmax.row_sum;
        cute::arrive_barrier(shared_storage.mbar_softmax_corr_full[q_stage]);
    }

    CUTLASS_DEVICE void correction_rescale(const int q_stage, const int tidx,
                                           float const scale) {
        TiledMma1 tiled_mma1;
        cute::ThrMMA cta_mma1 = tiled_mma1.get_slice(0);
        cute::Tensor tOtO = cute::partition_fragment_C(
            cta_mma1, cute::select<0, 2>(TileShape_MNK{}));
        tOtO.data() = kNumStagesQ * kBlockN + q_stage * kHeadDim;
        cute::Tensor tOcO = cta_mma1.partition_C(
            cute::make_identity_tensor(cute::select<0, 2>(TileShape_MNK{})));
        constexpr int corr_tile_size = 16;

        using tmem_load_atom = cute::SM100_TMEM_LOAD_32dp32b16x;
        using tmem_store_atom = cute::SM100_TMEM_STORE_32dp32b16x;

        cute::Tensor tOtO_i = tOtO.compose(cute::make_layout(cute::make_shape(
            cute::Int<kBlockM>{}, cute::Int<corr_tile_size>{})));
        cute::Tensor tOcO_i = cute::composition(
            tOcO, cute::make_layout(cute::make_shape(
                      cute::Int<kBlockM>{}, cute::Int<corr_tile_size>{})));
        auto tiled_tmem_load = cute::make_tmem_copy(tmem_load_atom{}, tOtO_i);
        auto tiled_tmem_store = cute::make_tmem_copy(tmem_store_atom{}, tOtO_i);
        auto thr_tmem_load = tiled_tmem_load.get_slice(tidx);
        auto thr_tmem_store = tiled_tmem_store.get_slice(tidx);
        cute::Tensor tOtO_t2r = thr_tmem_load.partition_S(tOtO_i);
        auto tOrO_t2r_shape = cute::shape(thr_tmem_load.partition_D(tOcO_i));
        cute::Tensor tOtO_r2t = thr_tmem_store.partition_D(tOtO_i);
        constexpr int frg_count = kHeadDim / corr_tile_size;

        float2 const scale2 = {scale, scale};

#pragma unroll
        for (int i = 0; i < frg_count; ++i) {
            cute::Tensor tOrO_frg = cute::make_tensor<float>(tOrO_t2r_shape);
            cute::Tensor tOtO_t2r_i = cute::make_tensor(
                tOtO_t2r.data() + i * corr_tile_size, tOtO_t2r.layout());
            cute::copy(tiled_tmem_load, tOtO_t2r_i, tOrO_frg);
#pragma unroll
            for (int j = 0; j < cute::size(tOrO_frg); j += 2) {
                mul_packed_f32x2(*reinterpret_cast<float2 *>(&tOrO_frg(j)),
                                 scale2,
                                 *reinterpret_cast<float2 *>(&tOrO_frg(j)));
            }
            cute::Tensor tOtO_r2t_i = cute::make_tensor(
                tOtO_r2t.data() + i * corr_tile_size, tOtO_r2t.layout());
            cute::copy(tiled_tmem_store, tOrO_frg, tOtO_r2t_i);
            cutlass::arch::fence_view_async_tmem_store();
        }
    }

    template <typename SharedStorage>
    CUTLASS_DEVICE void correction_epilogue(SharedStorage &shared_storage,
                                            const int tidx, const int q_stage,
                                            float const scale, const int rest_m,
                                            const int O_stride,
                                            Element *O_ptr) {
        using PackedHalf = typename PackedHalf<Element>::Type;
        constexpr int corr_tile_size = 32 * 8 / cutlass::sizeof_bits_v<Element>;
        Element *sO =
            shared_storage.smem_o.data() + q_stage * kBlockM * kHeadDim;
        TiledMma1 tiled_mma1;
        cute::ThrMMA cta_mma1 = tiled_mma1.get_slice(0);

        cute::Tensor tOtO = cute::partition_fragment_C(
            cta_mma1, cute::select<0, 2>(TileShape_MNK{}));
        tOtO.data() = kNumStagesQ * kBlockN + q_stage * kHeadDim;
        auto tOcO = cta_mma1.partition_C(
            cute::make_identity_tensor(cute::select<0, 2>(TileShape_MNK{})));

        auto tOtO_i = cute::logical_divide(
            tOtO, cute::make_layout(cute::make_shape(
                      cute::Int<kBlockM>{}, cute::Int<corr_tile_size>{})));
        auto tOcO_i = cute::logical_divide(
            tOcO, cute::make_layout(cute::make_shape(
                      cute::Int<kBlockM>{}, cute::Int<corr_tile_size>{})));

        using tmem_load_atom = cute::SM100_TMEM_LOAD_32dp32b16x;
        auto tiled_tmem_load =
            cute::make_tmem_copy(tmem_load_atom{}, tOtO_i(cute::_, 0));

        auto thr_tmem_load = tiled_tmem_load.get_slice(tidx);
        cute::Tensor tOtO_t2r = thr_tmem_load.partition_S(tOtO_i);
        cute::Tensor tOcO_t2r = thr_tmem_load.partition_D(tOcO_i);

        float2 const scale2 = {scale, scale};

        static constexpr int kNumVecElem =
            cute::ceil_div(128, cute::sizeof_bits_v<Element>);
        static constexpr int kNumThreadsPerRow = kHeadDim / kNumVecElem;
        static_assert(cutlass::NumThreadsPerWarpGroup % kNumThreadsPerRow == 0);
        static constexpr int kNumRows =
            cutlass::NumThreadsPerWarpGroup / kNumThreadsPerRow;

#pragma unroll
        for (int i = 0; i < kHeadDim / corr_tile_size; ++i) {
            cute::Tensor tOrO_frg =
                cute::make_tensor<float>(cute::shape(tOcO_t2r(cute::_, 0, i)));
            cute::Tensor tOrO_frg_cvt = cute::make_tensor<Element>(
                cute::shape(tOcO_t2r(cute::_, 0, i)));
            cute::copy(tiled_tmem_load, tOtO_t2r(cute::_, 0, i), tOrO_frg);
#pragma unroll
            for (int j = 0; j < cute::size(tOrO_frg); j += 2) {
                mul_packed_f32x2(*reinterpret_cast<float2 *>(&tOrO_frg(j)),
                                 scale2,
                                 *reinterpret_cast<float2 *>(&tOrO_frg(j)));
                *reinterpret_cast<PackedHalf *>(&tOrO_frg_cvt(j)) =
                    PackedHalf(tOrO_frg(j), tOrO_frg(j + 1));
            }

            *reinterpret_cast<float4 *>(sO + tidx * kHeadDim +
                                        (tidx + 2 * i) % kNumThreadsPerRow *
                                            kNumVecElem) =
                *reinterpret_cast<float4 *>(tOrO_frg_cvt.data());
            *reinterpret_cast<float4 *>(sO + tidx * kHeadDim +
                                        (tidx + 2 * i + 1) % kNumThreadsPerRow *
                                            kNumVecElem) =
                *reinterpret_cast<float4 *>(tOrO_frg_cvt.data() + kNumVecElem);
        }

        cutlass::arch::fence_view_async_shared();

        cutlass::arch::NamedBarrier::sync(cutlass::NumThreadsPerWarpGroup, 1);

        const int store_row_idx = tidx / kNumThreadsPerRow;
        const int store_col_idx = tidx % kNumThreadsPerRow;

        for (int i = store_row_idx; i < rest_m; i += kNumRows) {
            *reinterpret_cast<float4 *>(O_ptr + i * O_stride +
                                        store_col_idx * kNumVecElem) =
                *reinterpret_cast<float4 *>(
                    sO + i * kHeadDim +
                    ((store_col_idx + i) % kNumThreadsPerRow) * kNumVecElem);
        }
    }

    template <typename SharedStorage>
    CUTLASS_DEVICE void correction(Params const &mainloop_params,
                                   SharedStorage &shared_storage,
                                   const int lane_idx, const int warp_idx,
                                   const int thread_idx, const int bidb,
                                   const int bidh, const int n_block_max,
                                   const int m_block, const int seqlen_q) {
        const int tidx = thread_idx % cutlass::NumThreadsPerWarpGroup;
        const int O_stride = mainloop_params.O_stride;

        range_constexpr<kNumStagesQ>([&](auto q_stage) {
            cute::arrive_barrier(
                shared_storage.mbar_P_full_O_rescaled[q_stage]);
        });

        int softmax_corr_consumer_phase = 0;
        int o_corr_consumer_phase = 0;
        int corr_epi_producer_phase = 1;

        Element *mO_ptr = reinterpret_cast<Element *>(mainloop_params.ptr_O) +
                          (mainloop_params.cu_seq_q[bidb] +
                           m_block * (kNumStagesQ * kBlockM)) *
                              O_stride +
                          bidh * kHeadDim;

        cute::wait_barrier(shared_storage.mbar_softmax_corr_full[0],
                           softmax_corr_consumer_phase);
        cute::arrive_barrier(shared_storage.mbar_softmax_corr_empty[0]);
        cute::wait_barrier(shared_storage.mbar_softmax_corr_full[1],
                           softmax_corr_consumer_phase);
        softmax_corr_consumer_phase ^= 1;

#pragma unroll 1
        for (int n_block = 0; n_block <= n_block_max - 2; ++n_block) {
            range_constexpr<kNumStagesQ>([&](auto q_stage) {
                cute::wait_barrier(
                    shared_storage.mbar_softmax_corr_full[q_stage],
                    softmax_corr_consumer_phase);
                float const scale =
                    shared_storage.sScale[tidx + q_stage * kBlockM];
                bool const should_rescale =
                    __ballot_sync(0xffffffff, scale < 1.0f) != 0;
                if (should_rescale) {
                    correction_rescale(q_stage, tidx, scale);
                }
                cute::arrive_barrier(
                    shared_storage.mbar_P_full_O_rescaled[q_stage]);
                cute::arrive_barrier(
                    shared_storage.mbar_softmax_corr_empty[1 - q_stage]);
            });
            softmax_corr_consumer_phase ^= 1;
        }

        cute::arrive_barrier(shared_storage.mbar_softmax_corr_empty[1]);

        range_constexpr<kNumStagesQ>([&](auto q_stage) {
            cute::wait_barrier(shared_storage.mbar_softmax_corr_full[q_stage],
                               softmax_corr_consumer_phase);
            float const row_sum =
                shared_storage.sScale[tidx + q_stage * kBlockM];
            cute::arrive_barrier(
                shared_storage.mbar_softmax_corr_empty[q_stage]);
            float scale;
            asm volatile("rcp.approx.ftz.f32 %0, %1;\n"
                         : "=f"(scale)
                         : "f"(row_sum));
            cute::wait_barrier(shared_storage.mbar_O_full[q_stage],
                               o_corr_consumer_phase);
            const int rest_m =
                min(seqlen_q - (m_block * kNumStagesQ + q_stage) * kBlockM,
                    kBlockM);
            if (rest_m > 0) {
                correction_epilogue(shared_storage, tidx, q_stage, scale,
                                    rest_m, O_stride,
                                    mO_ptr + q_stage * kBlockM * O_stride);
            }
            cute::arrive_barrier(
                shared_storage.mbar_P_full_O_rescaled[q_stage]);
        });
        o_corr_consumer_phase ^= 1;
        softmax_corr_consumer_phase ^= 1;
        corr_epi_producer_phase ^= 1;
    }
};

template <typename KernelTraits>
struct CollectiveMainloopFwdFP8Decoder {
    static constexpr int kNumStagesKV = KernelTraits::kNumStagesKV;
    static constexpr int kHeadDim = KernelTraits::kHeadDim;
    static constexpr int kBlockM = KernelTraits::kBlockM;
    static constexpr int kBlockN = KernelTraits::kBlockN;
    static constexpr int kStageBlock = KernelTraits::KStageBlock;

    using Element = typename KernelTraits::Element;
    using OutputType = typename KernelTraits::OutputType;
    using ElementAcc = typename KernelTraits::ElementAcc;
    using TileShape_MNK = typename KernelTraits::TileShape_MNK;
    using ClusterShape = typename KernelTraits::ClusterShape;
    using MainloopPipeline = typename KernelTraits::MainloopPipeline;
    using PipelineState = typename MainloopPipeline::PipelineState;
    using SmemLayoutScale = typename KernelTraits::SmemLayoutScale;

    static constexpr bool use_fp8 = cutlass::sizeof_bits_v<Element> == 8;

    using GmemTiledCopyQ = cute::SM90_TMA_LOAD;
    using GmemTiledCopyKV = decltype(cutlass::gemm::collective::detail::
                                         sm90_cluster_shape_to_tma_atom(
                                             cute::shape<0>(ClusterShape{})));
    using GmemTiledCopyO = cute::SM90_TMA_STORE;

    using SmemLayoutQ = typename KernelTraits::SmemLayoutQ;
    using SmemLayoutK = typename KernelTraits::SmemLayoutK;
    using SmemLayoutV = typename KernelTraits::SmemLayoutV;
    using SmemLayoutVt = typename KernelTraits::SmemLayoutVt;
    using SmemLayoutS = typename KernelTraits::SmemLayoutS;
    using TiledMma0 = typename KernelTraits::TiledMma0;
    using TiledMma1 = typename KernelTraits::TiledMma1;

    using ShapeT = cute::Shape<int64_t, int64_t, int64_t, int64_t>;
    using StrideT = cute::Shape<int64_t, cute::_1, int64_t, int64_t>;
    using LayoutT = cute::Layout<ShapeT, StrideT>;

    using ShapeTScale =
        cute::Shape<cute::Int<kDequantScalePackSize>, int32_t, int32_t>;
    using StrideTScale =
        cute::Shape<cute::_1, cute::Int<kDequantScalePackSize>, int32_t>;
    using LayoutTScale = cute::Layout<ShapeTScale, StrideTScale>;

    using TMA_Q = decltype(cute::make_tma_copy(
        GmemTiledCopyQ{},
        cute::make_tensor(
            cute::make_gmem_ptr(static_cast<Element const *>(nullptr)),
            cute::repeat_like(StrideT{}, int32_t(0)), StrideT{}),
        SmemLayoutQ{}, cute::select<0, 2>(TileShape_MNK{}), cute::_1{}));

    using TMA_KV = decltype(cute::make_tma_copy(
        GmemTiledCopyKV{},
        cute::make_tensor(
            cute::make_gmem_ptr(static_cast<Element const *>(nullptr)),
            cute::repeat_like(StrideT{}, int32_t(0)), StrideT{}),
        cute::take<0, 2>(SmemLayoutK{}), cute::select<1, 2>(TileShape_MNK{}),
        cute::size<0>(ClusterShape{})));

    static constexpr uint32_t TmaTransactionBytesQ =
        static_cast<uint32_t>(cute::size(cute::take<0, 2>(SmemLayoutQ{})) *
                              cutlass::sizeof_bits_v<Element> / 8);
    static constexpr uint32_t TmaTransactionBytesK =
        static_cast<uint32_t>(cute::size(cute::take<0, 2>(SmemLayoutK{})) *
                              cutlass::sizeof_bits_v<Element> / 8);

    struct Arguments {
        Element const *ptr_Q;
        LayoutT layout_Q;
        Element const *ptr_K;
        LayoutT layout_K;
        Element const *ptr_V;
        LayoutT layout_V;
        LayoutTScale layout_S;
        const int *cu_seq_q;
        const int *cu_seq_k;
        OutputType *partition_o_ptr;
        float2 *max_sum;
        const float *q_dequant_scale;
        const float *kv_dequant_scale;
        const int *block_tables;
        const int *n_block_infos;
        const int max_blocks_per_seq;
        const int head_num;
        const int kv_head_num;
        const int batch_size;
        const int max_num_partitions;
    };

    struct Params {
        LayoutT layout_Q;
        LayoutT layout_K;
        LayoutT layout_V;
        LayoutTScale layout_S;
        TMA_Q tma_load_Q;
        TMA_KV tma_load_K;
        TMA_KV tma_load_V;
        const int *cu_seq_q;
        const int *cu_seq_k;
        OutputType *partition_o_ptr;
        float2 *max_sum;
        const float *q_dequant_scale;
        const float *kv_dequant_scale;
        const int *block_tables;
        const int *n_block_infos;
        int max_blocks_per_seq;
        const int kv_head_num;
        const int batch_size;
        const int max_num_partitions;
        const int qhead_per_khead;
        cutlass::FastDivmod qhead_per_khead_divmod;
        cutlass::FastDivmod khead_divmod;
        const Element *k_raw_ptr;
        const Element *v_raw_ptr;
    };

    static Params to_underlying_arguments(Arguments const &args) {
        cute::Tensor mQ =
            cute::make_tensor(cute::make_gmem_ptr(args.ptr_Q), args.layout_Q);
        TMA_Q tma_load_Q = cute::make_tma_copy(
            GmemTiledCopyQ{}, mQ, SmemLayoutQ{},
            cute::select<0, 2>(TileShape_MNK{}), cute::_1{});
        cute::Tensor mK =
            cute::make_tensor(cute::make_gmem_ptr(args.ptr_K), args.layout_K);
        TMA_KV tma_load_K = cute::make_tma_copy(
            GmemTiledCopyKV{}, mK, SmemLayoutK{}(cute::_, cute::_, cute::_0{}),
            cute::select<1, 2>(TileShape_MNK{}), cute::size<0>(ClusterShape{}));
        cute::Tensor mV =
            cute::make_tensor(cute::make_gmem_ptr(args.ptr_V), args.layout_V);
        TMA_KV tma_load_V = cute::make_tma_copy(
            GmemTiledCopyKV{}, mV, SmemLayoutV{}(cute::_, cute::_, cute::_0{}),
            cute::select<1, 2>(TileShape_MNK{}), cute::size<0>(ClusterShape{}));
        return {
            args.layout_Q,
            args.layout_K,
            args.layout_V,
            args.layout_S,
            tma_load_Q,
            tma_load_K,
            tma_load_V,
            args.cu_seq_q,
            args.cu_seq_k,
            args.partition_o_ptr,
            args.max_sum,
            args.q_dequant_scale,
            args.kv_dequant_scale,
            args.block_tables,
            args.n_block_infos,
            args.max_blocks_per_seq,
            args.kv_head_num,
            args.batch_size,
            args.max_num_partitions,
            args.head_num / args.kv_head_num,
            cutlass::FastDivmod(args.head_num / args.kv_head_num),
            cutlass::FastDivmod(args.kv_head_num),
            args.ptr_K,
            args.ptr_V,
        };
    }

    CUTLASS_DEVICE
    static void prefetch_tma_descriptors(Params const &mainloop_params) {
        cute::prefetch_tma_descriptor(
            mainloop_params.tma_load_Q.get_tma_descriptor());
        cute::prefetch_tma_descriptor(
            mainloop_params.tma_load_K.get_tma_descriptor());
        cute::prefetch_tma_descriptor(
            mainloop_params.tma_load_V.get_tma_descriptor());
    }

    // Warp-cooperative L2 prefetch; each lane covers different K cache lines.
    CUTLASS_DEVICE void prefetch_next_partition_warp(
        Params const &mainloop_params, const int lane_idx,
        const int next_n_block, const int next_bidh, const int next_bidb,
        const int seqlen_k) {
        const int kTokensPerBlock = kBlockN * kStageBlock;
        if (next_n_block * kTokensPerBlock >= seqlen_k) return;

        const int n_block_max = get_n_block_max(seqlen_k, next_n_block);
        const int *block_tables =
            mainloop_params.block_tables +
            next_bidb * mainloop_params.max_blocks_per_seq +
            next_n_block * kStageBlock;

        constexpr int kBlockBytes = kBlockN * kHeadDim;
        const int kv_head_num = mainloop_params.kv_head_num;
        const char *k_base =
            reinterpret_cast<const char *>(mainloop_params.k_raw_ptr);
        const int64_t block_stride =
            static_cast<int64_t>(kv_head_num) * kBlockBytes;
        const int64_t head_stride = kBlockBytes;

        constexpr int kCacheLine = 128;
        constexpr int kLinesPerBlock = kBlockBytes / kCacheLine;
        constexpr int kLinesPerLane = kLinesPerBlock / 32;

#pragma unroll
        for (int iter = 0; iter < kStageBlock; ++iter) {
            if (iter >= n_block_max) break;
            const int phys_block = __ldg(block_tables + iter);
            const int64_t byte_offset =
                static_cast<int64_t>(phys_block) * block_stride +
                static_cast<int64_t>(next_bidh) * head_stride;
            const char *k_block = k_base + byte_offset;
#pragma unroll
            for (int l = 0; l < kLinesPerLane; ++l) {
                int line_idx = lane_idx + l * 32;
                asm volatile("prefetch.global.L2 [%0];" ::"l"(
                    k_block + line_idx * kCacheLine));
            }
        }
    }

    CUTLASS_DEVICE int get_n_block_max(const int seqlen_k, const int n_block) {
        const int n_block_max =
            min((seqlen_k + kBlockN - 1) / kBlockN - n_block * kStageBlock,
                kStageBlock);
        return n_block_max;
    }

    CUTLASS_DEVICE void get_coordinates(Params const &mainloop_params,
                                        const int multiplier,
                                        const int shift_right,
                                        const int divisor, const int block_idx,
                                        int &n_block, int &bidb, int &bidh) {
        int nb_block;
        mainloop_params.khead_divmod(nb_block, bidh, block_idx);
        bidb = (divisor != 1) ? __umulhi(nb_block, multiplier) >> shift_right
                              : nb_block;
        n_block = nb_block - (bidb * divisor);
    }

    template <typename SharedStorage>
    CUTLASS_DEVICE void load(Params const &mainloop_params,
                             MainloopPipeline pipeline_k,
                             MainloopPipeline pipeline_v,
                             SharedStorage &shared_storage, const int lane_idx,
                             const int n_block_max, const int n_block,
                             const int bidh, const int bidb,
                             const int pipeline_advance = 0) {
        cute::Tensor sQ = cute::make_tensor(
            cute::make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutQ{});
        cute::Tensor sK = cute::make_tensor(
            cute::make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutK{});
        cute::Tensor sV = cute::make_tensor(
            cute::make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutV{});
        cute::Tensor mQ = mainloop_params.tma_load_Q.get_tma_tensor(
            mainloop_params.layout_Q.shape());
        cute::Tensor mK = mainloop_params.tma_load_K.get_tma_tensor(
            mainloop_params.layout_K.shape());
        cute::Tensor mV = mainloop_params.tma_load_V.get_tma_tensor(
            mainloop_params.layout_V.shape());

        cute::Tensor gQ =
            cute::local_tile(mQ(cute::_, cute::_, bidh, bidb),
                             cute::select<0, 2>(TileShape_MNK{}),
                             cute::make_coord(cute::_0{}, cute::_0{}));
        cute::Tensor gK = cute::local_tile(
            mK(cute::_, cute::_, bidh, cute::_),
            cute::select<0, 1>(
                cute::Shape<cute::Int<kBlockN>, cute::Int<kHeadDim>>{}),
            cute::make_coord(cute::_0{}, cute::_0{}));
        cute::Tensor gV = cute::local_tile(
            mV(cute::_, cute::_, bidh, cute::_),
            cute::select<0, 1>(
                cute::Shape<cute::Int<kBlockN>, cute::Int<kHeadDim>>{}),
            cute::make_coord(cute::_0{}, cute::_0{}));
        auto [tQgQ, tQsQ] = cute::tma_partition(
            mainloop_params.tma_load_Q, cute::_0{}, cute::Layout<cute::_1>{},
            cute::group_modes<0, 2>(sQ), cute::group_modes<0, 2>(gQ));
        auto [tKgK, tKsK] = cute::tma_partition(
            mainloop_params.tma_load_K, cute::_0{}, cute::Layout<cute::_1>{},
            cute::group_modes<0, 2>(sK), cute::group_modes<0, 2>(gK));
        auto [tVgV, tVsV] = cute::tma_partition(
            mainloop_params.tma_load_V, cute::_0{}, cute::Layout<cute::_1>{},
            cute::group_modes<0, 2>(sV), cute::group_modes<0, 2>(gV));

        const int kv_head_num = mainloop_params.kv_head_num;
        auto load_Q = [&]() {
            cute::set_barrier_transaction_bytes(shared_storage.mbar_load_q_full,
                                                TmaTransactionBytesQ);
            cute::copy(mainloop_params.tma_load_Q.with(
                           shared_storage.mbar_load_q_full, 0),
                       tQgQ, tQsQ);
        };

        int k_stage = 0;
        int v_stage = 0;
        PipelineState k_producer_state =
            cutlass::make_producer_start_state<MainloopPipeline>();
        PipelineState v_producer_state =
            cutlass::make_producer_start_state<MainloopPipeline>();
        k_producer_state.advance(pipeline_advance);
        v_producer_state.advance(pipeline_advance);
        auto load_K = [&](const int src_idx) {
            pipeline_k.producer_acquire(k_producer_state);
            // The scale store must precede the K TMA issue: otherwise the TMA
            // can complete and unblock MMA/softmax before the plain smem store
            // is visible.
            float *scale_dst = shared_storage.kv_deqaunt_scale.data() +
                               k_stage * kDequantScalePackSize;
            const float *scale_src =
                mainloop_params.kv_dequant_scale +
                src_idx * kv_head_num * kDequantScalePackSize +
                bidh * kDequantScalePackSize;
            *reinterpret_cast<float4 *>(scale_dst) =
                __ldg(reinterpret_cast<const float4 *>(scale_src));
            cute::copy(
                mainloop_params.tma_load_K.with(
                    *pipeline_k.producer_get_barrier(k_producer_state), 0),
                tKgK(cute::_, src_idx),
                tKsK(cute::_, k_producer_state.index()));
            ++k_producer_state;
            ++k_stage;
        };

        auto load_V = [&](const int src_idx) {
            pipeline_v.producer_acquire(v_producer_state);
            cute::copy(
                mainloop_params.tma_load_V.with(
                    *pipeline_v.producer_get_barrier(v_producer_state), 0),
                tVgV(cute::_, src_idx),
                tVsV(cute::_, v_producer_state.index()));
            ++v_producer_state;
            ++v_stage;
        };

        // All lanes stage block-table entries in smem in parallel, avoiding a
        // serial __ldg chain in the elected lane.
        {
            const int *bt_base = mainloop_params.block_tables +
                                 bidb * mainloop_params.max_blocks_per_seq +
                                 n_block * kStageBlock;
            // The last partition may hold fewer than kStageBlock blocks.
            if (lane_idx < n_block_max) {
                shared_storage.smem_block_tables[lane_idx] =
                    __ldg(bt_base + lane_idx);
            }
            __syncwarp();
        }

        if (cute::elect_one_sync()) {
            load_Q();
            load_K(shared_storage.smem_block_tables[n_block_max - 1]);
#pragma unroll
            for (int iter = 1; iter < kStageBlock; ++iter) {
                if (iter < n_block_max) {
                    load_K(shared_storage
                               .smem_block_tables[n_block_max - iter - 1]);
                    load_V(
                        shared_storage.smem_block_tables[n_block_max - iter]);
                }
            }
            load_V(shared_storage.smem_block_tables[0]);
        }
    }

    template <typename SharedStorage>
    CUTLASS_DEVICE void mma(Params const &mainloop_params,
                            MainloopPipeline pipeline_k,
                            MainloopPipeline pipeline_v,
                            SharedStorage &shared_storage,
                            const int n_block_max, const int phase,
                            const int pipeline_advance = 0) {
        cute::Tensor sQ = cute::make_tensor(
            cute::make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutQ{});
        cute::Tensor sK = cute::make_tensor(
            cute::make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutK{});
        cute::Tensor sVt = cute::make_tensor(
            cute::make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutVt{});
        cute::Tensor sS = cute::make_tensor(
            cute::make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutS{});

        TiledMma0 tiled_mma0;
        TiledMma1 tiled_mma1;

        cute::ThrMMA cta_mma0 = tiled_mma0.get_slice(0);
        cute::ThrMMA cta_mma1 = tiled_mma1.get_slice(0);

        cute::Tensor tSrQ = cta_mma0.partition_fragment_A(sQ);
        cute::Tensor tSrK = cta_mma0.partition_fragment_B(sK);
        cute::Tensor tSrV = cta_mma1.partition_fragment_B(sVt);
        cute::Tensor tPtP = cta_mma1.partition_fragment_A(sS);
        cute::Tensor tStS = cute::partition_fragment_C(
            cta_mma0, cute::select<0, 1>(TileShape_MNK{}));
        cute::Tensor tOtO = cute::partition_fragment_C(
            cta_mma1, cute::select<0, 2>(TileShape_MNK{}));
        tPtP.data() = 0;
        tStS.data() = 0;
        tOtO.data() = 2 * kBlockN;

        PipelineState mma_k_consumer_state;
        PipelineState mma_v_consumer_state;
        mma_k_consumer_state.advance(pipeline_advance);
        mma_v_consumer_state.advance(pipeline_advance);

        cute::wait_barrier(shared_storage.mbar_load_q_full, phase);
        pipeline_k.consumer_wait(mma_k_consumer_state);
        gemm<true>(
            tiled_mma0, tSrQ,
            tSrK(cute::_, cute::_, cute::_, mma_k_consumer_state.index()),
            tStS);
        pipeline_k.consumer_release(mma_k_consumer_state);
        cutlass::arch::umma_arrive(&shared_storage.mbar_S_full[0]);
        ++mma_k_consumer_state;

#pragma unroll
        for (int iter = 1; iter < kStageBlock; ++iter) {
            if (iter >= n_block_max) {
                break;
            }
            tStS.data() = (iter % 2) * kBlockN;
            pipeline_k.consumer_wait(mma_k_consumer_state);
            gemm<true>(
                tiled_mma0, tSrQ,
                tSrK(cute::_, cute::_, cute::_, mma_k_consumer_state.index()),
                tStS);
            pipeline_k.consumer_release(mma_k_consumer_state);
            cutlass::arch::umma_arrive(&shared_storage.mbar_S_full[iter]);
            ++mma_k_consumer_state;
            pipeline_v.consumer_wait(mma_v_consumer_state);
            cute::wait_barrier(shared_storage.mbar_P_full_O_rescaled[iter - 1],
                               phase);
            fp8_gemm<true>(
                tiled_mma1, tPtP,
                tSrV(cute::_, cute::_, cute::_, mma_v_consumer_state.index()),
                tOtO, shared_storage.mbar_P_full_2[(iter - 1)], phase);
            pipeline_v.consumer_release(mma_v_consumer_state);
            cutlass::arch::umma_arrive(&shared_storage.mbar_O_full[iter - 1]);
            ++mma_v_consumer_state;
            tPtP.data() = (iter % 2) * kBlockN;
            tOtO.data() = (iter % 2) * kBlockN + 2 * kBlockN;
        }
        pipeline_v.consumer_wait(mma_v_consumer_state);
        cute::wait_barrier(
            shared_storage.mbar_P_full_O_rescaled[n_block_max - 1], phase);
        fp8_gemm<true>(
            tiled_mma1, tPtP,
            tSrV(cute::_, cute::_, cute::_, mma_v_consumer_state.index()), tOtO,
            shared_storage.mbar_P_full_2[n_block_max - 1], phase);
        pipeline_v.consumer_release(mma_v_consumer_state);
        cutlass::arch::umma_arrive(
            &shared_storage.mbar_O_full[n_block_max - 1]);
    }

    template <bool IsFirst, bool NeedMask, typename SharedStorage>
    CUTLASS_DEVICE void softmax_step(Params const &mainloop_params,
                                     SharedStorage &shared_storage,
                                     const int thread_idx, const int col_base,
                                     const int si_stage, uint32_t *acc_s,
                                     float &acc_sum,
                                     const float q_dequant_scale,
                                     const int phase) {
        constexpr int VecLen = kBlockN / 2;
        const int data_offset = si_stage % 2 * kBlockN;
        cute::wait_barrier(shared_storage.mbar_S_full[si_stage], phase);
        const float k_dequant_scale =
            shared_storage.kv_deqaunt_scale
                .data()[si_stage * kDequantScalePackSize];
        float *acc_f = reinterpret_cast<float *>(acc_s);

        tmem_load<cute::SM100_TMEM_LOAD_16dp64b64x>(
            data_offset, acc_s, std::make_index_sequence<VecLen>{});

        if constexpr (IsFirst) {
            // tidx resets per warpgroup; rebuild the CTA warp for the mask.
            const int lane_idx = thread_idx % cutlass::NumThreadsPerWarp;
            const int warp_idx = thread_idx / cutlass::NumThreadsPerWarp +
                                 si_stage * cutlass::NumWarpsPerWarpGroup;
            int mask_row_idx =
                (lane_idx / 4) + 8 * (lane_idx % 2) + warp_idx * 16;
            const int col_offset = lane_idx % 4 / 2;
            mask_row_idx /= mainloop_params.qhead_per_khead;
            const int col_limit = mask_row_idx + 1 + col_base - col_offset;
#pragma unroll
            for (int j = 0; j < VecLen; ++j) {
                if (2 * j >= col_limit) {
                    reinterpret_cast<float *>(acc_s)[j] = -FLT_MAX;
                }
            }
        }

        float cur_max = fmax_3_input(acc_f[0], acc_f[1], acc_f[2]);

        float2 cur_sum = {0.0f, 0.0f};

#pragma unroll
        for (int i = 3; i < VecLen; i += 2) {
            cur_max = fmax_3_input(cur_max, acc_f[i], acc_f[i + 1]);
        }
        const float neginhbor_max = __shfl_xor_sync(0xffffffff, cur_max, 2);
        cur_max = fmax_3_input(cur_max, acc_f[VecLen - 1], neginhbor_max);
        const float scale = q_dequant_scale * k_dequant_scale;
        const float bias = scale * cur_max;
        float2 scale2 = {scale, scale};
        float2 bias2 = {-bias, -bias};
#pragma unroll
        for (int i = 0; i < VecLen / 2; i += 2) {
            fma_packed_f32x2(*reinterpret_cast<float2 *>(&acc_f[i]), scale2,
                             bias2, *reinterpret_cast<float2 *>(&acc_f[i]));
        }
#pragma unroll
        for (int i = 0; i < VecLen / 2; i += 4) {
            acc_f[i] = exp2f(acc_f[i]);
            acc_f[i + 1] = exp2f(acc_f[i + 1]);
            acc_f[i + 2] = exp2f(acc_f[i + 2]);
            acc_f[i + 3] = exp2f(acc_f[i + 3]);
            cur_sum = add_packed_f32x2(cur_sum, {acc_f[i], acc_f[i + 1]});
            cur_sum = add_packed_f32x2(cur_sum, {acc_f[i + 2], acc_f[i + 3]});
            acc_s[i / 4] = floats4_to_fp8x4(acc_f[i], acc_f[i + 1],
                                            acc_f[i + 2], acc_f[i + 3]);
        }
        tmem_store<cute::SM100_TMEM_STORE_16dp64b8x>(
            data_offset, acc_s, std::make_index_sequence<VecLen / 8>{});
        cutlass::arch::fence_view_async_tmem_store();
        cute::arrive_barrier(shared_storage.mbar_P_full_O_rescaled[si_stage]);
#pragma unroll
        for (int i = VecLen / 2; i < VecLen; i += 2) {
            fma_packed_f32x2(*reinterpret_cast<float2 *>(&acc_f[i]), scale2,
                             bias2, *reinterpret_cast<float2 *>(&acc_f[i]));
        }

#pragma unroll
        for (int i = VecLen / 2; i < VecLen; i += 4) {
            acc_f[i] = exp2f(acc_f[i]);
            acc_f[i + 1] = exp2f(acc_f[i + 1]);
            acc_f[i + 2] = exp2f(acc_f[i + 2]);
            acc_f[i + 3] = exp2f(acc_f[i + 3]);
            cur_sum = add_packed_f32x2(cur_sum, {acc_f[i], acc_f[i + 1]});
            cur_sum = add_packed_f32x2(cur_sum, {acc_f[i + 2], acc_f[i + 3]});
            acc_s[i / 4] = floats4_to_fp8x4(acc_f[i], acc_f[i + 1],
                                            acc_f[i + 2], acc_f[i + 3]);
        }

        tmem_store<cute::SM100_TMEM_STORE_16dp64b8x>(
            data_offset + VecLen / 4, acc_s + VecLen / 8,
            std::make_index_sequence<VecLen / 8>{});

        cutlass::arch::fence_view_async_tmem_store();
        cur_max = cur_max * scale;
        shared_storage.local_max[si_stage * 128 + thread_idx] = cur_max;
        if (si_stage > 0) {
            // mbar_P_full_2[si_stage - 1] is arrived only after the previous
            // warpgroup wrote local_max and loacl_sum.
            cute::wait_barrier(shared_storage.mbar_P_full_2[si_stage - 1],
                               phase);
            float last_sum = shared_storage.loacl_sum[thread_idx];
            last_sum = last_sum *
                       exp2f(shared_storage
                                 .local_max[si_stage * 128 - 128 + thread_idx] -
                             cur_max);
            shared_storage.loacl_sum[thread_idx] =
                last_sum + cur_sum.x + cur_sum.y;
        } else {
            shared_storage.loacl_sum[thread_idx] = cur_sum.x + cur_sum.y;
        }
        // Must follow the local_max / loacl_sum writes: the next warpgroup
        // reads loacl_sum after mbar_S_full[si_stage + 1], which MMA arrives
        // only after this barrier.
        cute::arrive_barrier(shared_storage.mbar_P_full_2[si_stage]);
    }

    template <typename SharedStorage>
    CUTLASS_DEVICE void softmax(Params const &mainloop_params,
                                SharedStorage &shared_storage,
                                const int thread_idx, const int n_block_max,
                                const int n_block, const int bidh,
                                const int bidb, const int seqlen_q,
                                const int seqlen_k, const int phase) {
        constexpr int VecLen = kBlockN / 2;
        uint32_t acc_s[VecLen];
        const int warp_group = thread_idx / 128;
        int iter = warp_group;
        float acc_sum = 0.0f;
        const int tidx = thread_idx % 128;

        const int col_base = -n_block_max * kBlockN + kBlockN -
                             n_block * (kBlockN * kStageBlock) + seqlen_k -
                             seqlen_q + iter * kBlockN;

        const int kv_head_num = mainloop_params.kv_head_num;
        const float q_dequant_scale =
            mainloop_params.q_dequant_scale[bidb * kv_head_num + bidh];

        if (iter < n_block_max) {
            softmax_step<true, false>(mainloop_params, shared_storage, tidx,
                                      col_base, iter, acc_s, acc_sum,
                                      q_dequant_scale, phase);
        }

        iter += 2;

        for (; iter < kStageBlock; iter += 2) {
            if (iter >= n_block_max) {
                break;
            }
            softmax_step<false, false>(mainloop_params, shared_storage, tidx,
                                       col_base, iter, acc_s, acc_sum,
                                       q_dequant_scale, phase);
        }

        if (iter == n_block_max + 1) {
            const int max_sum_offset =
                (bidb * mainloop_params.max_num_partitions + n_block) *
                    mainloop_params.kv_head_num * kBlockM +
                bidh * kBlockM + tidx / 4 * 2 + tidx % 2;
            float cur_sum = shared_storage.loacl_sum[tidx];
            cur_sum += __shfl_xor_sync(0xffffffff, cur_sum, 2);
            if (tidx % 4 < 2) {
                mainloop_params.max_sum[max_sum_offset] = make_float2(
                    shared_storage.local_max[tidx + n_block_max * 128 - 128],
                    cur_sum);
            }
        }
    }

    template <typename SharedStorage>
    CUTLASS_DEVICE void correction(Params const &mainloop_params,
                                   SharedStorage &shared_storage,
                                   const int thread_idx, const int n_block_max,
                                   const int n_block, const int bidh,
                                   const int bidb, const int seqlen_q,
                                   const int seqlen_k, const int phase) {
        constexpr int VecLen = kBlockN / 2;
        uint32_t acc_o[VecLen];
        uint32_t acc_s[VecLen / 2];
        const int tidx = thread_idx % 128;
        int data_offset = 2 * kBlockN;
        cute::wait_barrier(shared_storage.mbar_O_full[0], phase);
        tmem_load<cute::SM100_TMEM_LOAD_16dp64b32x>(
            data_offset, acc_s, std::make_index_sequence<VecLen / 2>{});

        const float v_deqaunt_scale = shared_storage.kv_deqaunt_scale.data()[1];
        const float2 scale2 = {v_deqaunt_scale, v_deqaunt_scale};
#pragma unroll
        for (int i = 0; i < VecLen / 2; i += 2) {
            mul_packed_f32x2(*reinterpret_cast<float2 *>(&acc_s[i]), scale2,
                             *reinterpret_cast<float2 *>(&acc_o[i]));
        }

        tmem_load<cute::SM100_TMEM_LOAD_16dp64b32x>(
            data_offset + VecLen, acc_s,
            std::make_index_sequence<VecLen / 2>{});
#pragma unroll
        for (int i = VecLen / 2; i < VecLen; i += 2) {
            mul_packed_f32x2(
                *reinterpret_cast<float2 *>(&acc_s[i - VecLen / 2]), scale2,
                *reinterpret_cast<float2 *>(&acc_o[i]));
        }

        int iter = 1;
        for (; iter < kStageBlock; iter++) {
            if (iter >= n_block_max) {
                break;
            }
            data_offset = iter % 2 * kBlockN + 2 * kBlockN;
            cute::wait_barrier(shared_storage.mbar_O_full[iter], phase);
            // Arrived by softmax only after its local_max / loacl_sum writes.
            cute::wait_barrier(shared_storage.mbar_P_full_2[iter], phase);
            tmem_load<cute::SM100_TMEM_LOAD_16dp64b32x>(
                data_offset, acc_s, std::make_index_sequence<VecLen / 2>{});
            const float v_deqaunt_scale =
                shared_storage.kv_deqaunt_scale
                    .data()[iter * kDequantScalePackSize + 1];
            const float exp_scale =
                exp2f(shared_storage.local_max[iter * 128 - 128 + tidx] -
                      shared_storage.local_max[iter * 128 + tidx]);

            const float2 scale2 = {v_deqaunt_scale, v_deqaunt_scale};
            const float2 re_scale = {exp_scale, exp_scale};

#pragma unroll
            for (int i = 0; i < VecLen / 2; i += 2) {
                mul_packed_f32x2(*reinterpret_cast<float2 *>(&acc_s[i]), scale2,
                                 *reinterpret_cast<float2 *>(&acc_s[i]));
                fma_packed_f32x2(*reinterpret_cast<float2 *>(&acc_o[i]),
                                 re_scale,
                                 *reinterpret_cast<float2 *>(&acc_s[i]),
                                 *reinterpret_cast<float2 *>(&acc_o[i]));
            }

            tmem_load<cute::SM100_TMEM_LOAD_16dp64b32x>(
                data_offset + VecLen, acc_s,
                std::make_index_sequence<VecLen / 2>{});
#pragma unroll
            for (int i = VecLen / 2; i < VecLen; i += 2) {
                mul_packed_f32x2(
                    *reinterpret_cast<float2 *>(&acc_s[i - VecLen / 2]), scale2,
                    *reinterpret_cast<float2 *>(&acc_s[i - VecLen / 2]));
                fma_packed_f32x2(
                    *reinterpret_cast<float2 *>(&acc_o[i]), re_scale,
                    *reinterpret_cast<float2 *>(&acc_s[i - VecLen / 2]),
                    *reinterpret_cast<float2 *>(&acc_o[i]));
            }
        }

        const int tidx_lane = tidx % 32;
        const int logical_row =
            (tidx_lane / 4) + 8 * (tidx_lane % 2) + (tidx / 32) * 16;
        if (logical_row >= seqlen_q * mainloop_params.qhead_per_khead) return;

#pragma unroll
        for (int i = 0; i < VecLen; i += 2) {
            *reinterpret_cast<__nv_bfloat162 *>(&acc_o[i / 2]) =
                __nv_bfloat162(*reinterpret_cast<float *>(&acc_o[i]),
                               *reinterpret_cast<float *>(&acc_o[i + 1]));
        }

        float4 *out_ptr = reinterpret_cast<float4 *>(
            mainloop_params.partition_o_ptr +
            (bidb * mainloop_params.max_num_partitions *
                 mainloop_params.kv_head_num +
             n_block * mainloop_params.kv_head_num + bidh) *
                (kBlockM * kHeadDim));
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            out_ptr[tidx * 8 + i] = reinterpret_cast<float4 *>(&acc_o)[i];
        }
    }
};

}  // namespace halops
