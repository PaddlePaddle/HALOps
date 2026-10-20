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

#include "cute/atom/mma_atom.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/layout/layout.h"
#include "cutlass/numeric_types.h"
#include "cutlass/pipeline/pipeline.hpp"
#include "kernel/attention/utils.cuh"

namespace halops {

struct AttentionParams {
    void *__restrict__ q_ptr;
    void *__restrict__ k_ptr;
    void *__restrict__ v_ptr;
    void *__restrict__ o_ptr;
    void *__restrict__ partition_o_ptr;
    const int *__restrict__ cu_seq_q;
    const int *__restrict__ cu_seq_k;
    const int *__restrict__ seq_len_encoder;
    const int *__restrict__ seq_len_decoder;
    const int *__restrict__ n_block_infos;
    int *__restrict__ mask;
    int *__restrict__ block_tables;
    float2 *__restrict__ max_sum;
    float *__restrict__ q_dequant_scale;
    float *__restrict__ kv_dequant_scale;
    int head_num;
    int kv_head_num;
    int q_token_num;
    int k_token_num;
    int batch_size;
    float scale_softmax_log2;
    int max_blocks_per_seq;
    int max_num_block;
    int max_num_partitions;
};

template <int kNumStagesQ, int kNumStagesKV, int kBlockM, typename GemmType,
          typename SmemLayoutQ, typename SmemLayoutK, typename SmemLayoutO>
struct SharedStorage {
    union {
        cute::array_aligned<GemmType, cute::cosize_v<SmemLayoutQ>> smem_q;
        cute::array_aligned<GemmType, cute::cosize_v<SmemLayoutO>> smem_o;
    };
    cute::array_aligned<GemmType, cute::cosize_v<SmemLayoutK>> smem_k;
    cute::array_aligned<GemmType, cute::cosize_v<SmemLayoutK>> smem_v;
    struct {
        typename cutlass::PipelineTmaAsync<kNumStagesKV>::SharedStorage
            pipeline_k;
        typename cutlass::PipelineTmaAsync<kNumStagesKV>::SharedStorage
            pipeline_v;
        float sScale[kNumStagesQ * kBlockM * 2];
        alignas(16) uint32_t tmem_holding_buf;
        alignas(16) cute::uint64_t mbar_load_q_full[kNumStagesQ];
        alignas(16) cute::uint64_t mbar_P_full_O_rescaled[2];
        alignas(16) cute::uint64_t mbar_S_full[2];
        alignas(16) cute::uint64_t mbar_O_full[2];
        alignas(16) cute::uint64_t mbar_pv_done[2];
        alignas(16) cute::uint64_t mbar_softmax_corr_full[kNumStagesQ];
        alignas(16) cute::uint64_t mbar_softmax_corr_empty[kNumStagesQ];
        alignas(16) cute::uint64_t mbar_tmem_dealloc;
        alignas(16) cute::uint64_t mbar_P_full_2[kNumStagesQ];
        alignas(16) cute::uint64_t mbar_corr_epi_full[kNumStagesQ];
    };
};

template <int kNumStagesKV, int KStageBlock, int kBlockM, typename GemmType,
          typename SmemLayoutQ, typename SmemLayoutK, typename SmemLayoutScale>
struct SharedStorageFp8 {
    static constexpr int kNumBarrierSlots = KStageBlock;
    cute::array_aligned<GemmType, cute::cosize_v<SmemLayoutQ>> smem_q;
    cute::array_aligned<GemmType, cute::cosize_v<SmemLayoutK>> smem_k;
    cute::array_aligned<GemmType, cute::cosize_v<SmemLayoutK>> smem_v;
    alignas(128) cute::array_aligned<
        float, cute::cosize_v<SmemLayoutScale>> kv_deqaunt_scale;
    struct {
        typename cutlass::PipelineTmaAsync<kNumStagesKV>::SharedStorage
            pipeline_k;
        typename cutlass::PipelineTmaAsync<kNumStagesKV>::SharedStorage
            pipeline_v;
        float sScale[128];
        float local_max[128 * kNumBarrierSlots];
        float loacl_sum[128];
        alignas(16) uint32_t tmem_holding_buf;
        alignas(16) cute::uint64_t mbar_load_q_full;
        alignas(16) cute::uint64_t mbar_load_q_empty;
        alignas(16) cute::uint64_t mbar_tmem_dealloc;
        alignas(16) cute::uint64_t mbar_P_full_O_rescaled[kNumBarrierSlots];
        alignas(16) cute::uint64_t mbar_S_full[kNumBarrierSlots];
        alignas(16) cute::uint64_t mbar_O_full[kNumBarrierSlots];
        alignas(16) cute::uint64_t mbar_P_full_2[kNumBarrierSlots];
        alignas(16) int smem_block_tables[kNumBarrierSlots];
    };
};

template <int BlockM_, int BlockN_, int HeadDim_, int StageQ_, bool SplitKV_,
          int MinKvBlocksPerPartition_, bool CustomMask_,
          typename Element_ = cutlass::bfloat16_t>
struct AttentionKernelTraits {
    using Element = Element_;
    using ElementAcc = float;
    static constexpr int kBlockM = BlockM_;
    static constexpr int kBlockN = BlockN_;
    static constexpr int kHeadDim = HeadDim_;

    // Q smem stages: two for the encoder, one for the decoder.
    static constexpr int kNumStagesQ = StageQ_;
    // Number of K/V tile slots used by the asynchronous TMA pipelines.
    static constexpr int kNumStagesKV = 2;

    static constexpr bool kSplitKV = SplitKV_;
    // Smallest partition KV is cut into.
    static constexpr int kMinKvBlocksPerPartition = MinKvBlocksPerPartition_;

    static constexpr bool CustomMask = CustomMask_;

    // The unroll granularity of the KV loops.
    static constexpr int kNumUnrolledKvBlocks = 4;
    static_assert(kHeadDim == 128);

    // The encoder uses two Q stages and the wider warp layout; decoder uses
    // one Q stage and the compact layout.
    static constexpr bool kUseEncoderWarpLayout = kNumStagesQ > 1;
    static constexpr int kCorrection0WarpStartIds = 8;
    static constexpr int kEpilogueWarpStartIds = 8;
    static constexpr int kMmaWarpIds = kUseEncoderWarpLayout ? 12 : 4;
    static constexpr int kLoadWarpIds = kUseEncoderWarpLayout ? 13 : 5;
    static constexpr int kEmptyWarpStartIds = 14;

    static constexpr int kThreads = kUseEncoderWarpLayout
                                        ? 32 * (kEmptyWarpStartIds + 2)
                                        : 32 * (kLoadWarpIds + 1);
    static constexpr int num_regs_softmax = 200;
    static constexpr int num_regs_correction = 64;
    static constexpr int num_regs_other = 48;
    static constexpr int num_regs_empty = 24;

    using TileShape_MNK = cute::Shape<cute::Int<kBlockM>, cute::Int<kBlockN>,
                                      cute::Int<kHeadDim>>;
    using ClusterShape = cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>;

    using TiledMma0 = decltype(cute::make_tiled_mma(
        cute::SM100_MMA_F16BF16_SS<
            Element, Element, ElementAcc, cute::get<0>(TileShape_MNK{}),
            cute::get<1>(TileShape_MNK{}), cute::UMMA::Major::K,
            cute::UMMA::Major::K>{}));

    using TiledMma1 = decltype(cute::make_tiled_mma(
        cute::SM100_MMA_F16BF16_TS<
            Element, Element, ElementAcc, cute::get<0>(TileShape_MNK{}),
            cute::get<2>(TileShape_MNK{}), cute::UMMA::Major::K,
            cute::UMMA::Major::MN>{}));

    using SmemLayoutAtomQ =
        decltype(cutlass::gemm::collective::detail::ss_smem_selector<
                 cute::GMMA::Major::K, Element,
                 decltype(cute::get<0>(TileShape_MNK{})),
                 decltype(cute::get<2>(TileShape_MNK{}))>());
    using SmemLayoutQ = decltype(cute::tile_to_shape(
        SmemLayoutAtomQ{}, cute::make_shape(cute::shape<0>(TileShape_MNK{}),
                                            cute::shape<2>(TileShape_MNK{}),
                                            cute::Int<kNumStagesQ>{})));

    using SmemLayoutAtomK =
        decltype(cutlass::gemm::collective::detail::ss_smem_selector<
                 cute::GMMA::Major::K, Element,
                 decltype(cute::get<1>(TileShape_MNK{})),
                 decltype(cute::get<2>(TileShape_MNK{}))>());
    using SmemLayoutK = decltype(cute::tile_to_shape(
        SmemLayoutAtomK{}, cute::make_shape(cute::shape<1>(TileShape_MNK{}),
                                            cute::shape<2>(TileShape_MNK{}),
                                            cute::Int<kNumStagesKV>{})));

    using SmemLayoutAtomV =
        decltype(cutlass::gemm::collective::detail::rs_smem_selector<
                 cute::GMMA::Major::K, Element,
                 decltype(cute::get<1>(TileShape_MNK{})),
                 decltype(cute::get<2>(TileShape_MNK{}))>());
    using SmemLayoutV = decltype(cute::tile_to_shape(
        SmemLayoutAtomV{}, cute::make_shape(cute::shape<1>(TileShape_MNK{}),
                                            cute::shape<2>(TileShape_MNK{}),
                                            cute::Int<kNumStagesKV>{})));

    using SmemLayoutVt = decltype(cute::composition(
        SmemLayoutV{},
        cute::make_layout(
            cute::make_shape(cute::get<2>(TileShape_MNK{}),
                             cute::get<1>(TileShape_MNK{}),
                             cute::Int<kNumStagesKV>{}),
            cute::make_stride(cute::get<1>(TileShape_MNK{}), cute::_1{},
                              cute::Int<cute::size(SmemLayoutV{}(
                                  cute::_, cute::_, cute::_0{}))>{}))));

    using SmemLayoutAtomQK =
        decltype(cutlass::gemm::collective::detail::rs_smem_selector<
                 cute::GMMA::Major::K, Element,
                 decltype(cute::get<0>(TileShape_MNK{})),
                 decltype(cute::get<1>(TileShape_MNK{}))>());
    using SmemLayoutS = decltype(cute::tile_to_shape(
        SmemLayoutAtomQK{}, cute::make_shape(cute::shape<0>(TileShape_MNK{}),
                                             cute::shape<1>(TileShape_MNK{}))));

    using SmemLayoutAtomO =
        decltype(cutlass::gemm::collective::detail::ss_smem_selector<
                 cute::GMMA::Major::K, Element,
                 decltype(cute::get<0>(TileShape_MNK{})),
                 decltype(cute::get<2>(TileShape_MNK{}))>());
    using SmemLayoutO = decltype(cute::tile_to_shape(
        SmemLayoutAtomO{}, cute::make_shape(cute::shape<0>(TileShape_MNK{}),
                                            cute::shape<2>(TileShape_MNK{}),
                                            cute::Int<kNumStagesQ>{})));

    using SharedStorage =
        ::halops::SharedStorage<kNumStagesQ, kNumStagesKV, kBlockM, Element,
                                SmemLayoutQ, SmemLayoutK, SmemLayoutO>;

    using MainloopPipeline = typename cutlass::PipelineTmaAsync<kNumStagesKV>;
};

template <typename SmemSelector, typename Shape2D>
using SmemLayout2D = decltype(cute::tile_to_shape(SmemSelector{}, Shape2D{}));
template <typename SmemSelector, typename Shape3D>
using SmemLayout3D = decltype(cute::tile_to_shape(SmemSelector{}, Shape3D{}));
template <int kBlockM_, int kBlockN_, int KStageBlock_>
struct AttentionFp8KernelTraits {
    using Element = cutlass::float_e4m3_t;
    using ElementAcc = float;
    using OutputType = cutlass::bfloat16_t;

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = 128;
    // Number of K/V tile slots used by the asynchronous TMA pipelines.
    static constexpr int kNumStagesKV = 4;
    static constexpr int KStageBlock = KStageBlock_;

    static constexpr int kSoftmaxEndWarpIds = 8;
    static constexpr int kMmaWarpIds = 12;
    static constexpr int kLoadWarpIds = 13;
    static constexpr int kNumWarps = (kLoadWarpIds + 1 + 3) / 4 * 4;
    static constexpr int kThreads = 32 * kNumWarps;

    static constexpr int num_regs_softmax = 128;
    static constexpr int num_regs_correction = 128;
    static constexpr int num_regs_other = 48;
    static constexpr int num_regs_empty = 24;

    using TileShape_MNK = cute::Shape<cute::Int<kBlockM>, cute::Int<kBlockN>,
                                      cute::Int<kHeadDim>>;
    using ClusterShape = cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>;

    // MMA0: S = Q * K^T (SS mode, both operands in shared memory)
    using TiledMma0 = decltype(cute::make_tiled_mma(
        cute::MMA_Traits<
            cute::SM100_MMA_F8F6F4_SS, cutlass::float_e4m3_t,
            cutlass::float_e4m3_t, float, cute::C<kBlockM>, cute::C<kBlockN>,
            cute::integral_constant<cute::UMMA::Major, cute::UMMA::Major::K>,
            cute::integral_constant<cute::UMMA::Major, cute::UMMA::Major::K>,
            cute::integral_constant<cute::UMMA::ScaleIn,
                                    cute::UMMA::ScaleIn::One>,
            cute::integral_constant<cute::UMMA::ScaleIn,
                                    cute::UMMA::ScaleIn::One>>{}));

    // MMA1: O = P * V (TS mode, P in TMEM, V in shared memory)
    using TiledMma1 = decltype(cute::make_tiled_mma(
        cute::SM100_MMA_F8F6F4_TS<
            cutlass::float_e4m3_t, cutlass::float_e4m3_t, float, kBlockM,
            kHeadDim, cute::UMMA::Major::K, cute::UMMA::Major::MN>{}));

  private:
    using SmemAtomQ =
        decltype(cutlass::gemm::collective::detail::ss_smem_selector<
                 cute::GMMA::Major::K, Element, cute::Int<kBlockM>,
                 cute::Int<kHeadDim>>());
    using SmemAtomK =
        decltype(cutlass::gemm::collective::detail::ss_smem_selector<
                 cute::GMMA::Major::K, Element, cute::Int<kBlockN>,
                 cute::Int<kHeadDim>>());
    using SmemAtomV =
        decltype(cutlass::gemm::collective::detail::rs_smem_selector<
                 cute::GMMA::Major::K, Element, cute::Int<kBlockN>,
                 cute::Int<kHeadDim>>());
    using SmemAtomS =
        decltype(cutlass::gemm::collective::detail::rs_smem_selector<
                 cute::GMMA::Major::K, Element, cute::Int<kBlockM>,
                 cute::Int<kBlockN>>());

  public:
    using SmemLayoutQ =
        SmemLayout3D<SmemAtomQ,
                     cute::Shape<cute::Int<kBlockM>, cute::Int<kHeadDim>>>;
    using SmemLayoutK =
        SmemLayout3D<SmemAtomK,
                     cute::Shape<cute::Int<kBlockN>, cute::Int<kHeadDim>,
                                 cute::Int<kNumStagesKV>>>;
    using SmemLayoutV =
        SmemLayout3D<SmemAtomV,
                     cute::Shape<cute::Int<kBlockN>, cute::Int<kHeadDim>,
                                 cute::Int<kNumStagesKV>>>;
    using SmemLayoutS =
        SmemLayout2D<SmemAtomS,
                     cute::Shape<cute::Int<kBlockM>, cute::Int<kBlockN>>>;

    // V transposed layout (for TS-mode MMA1: P * V^T)
    using SmemLayoutVt = decltype(cute::composition(
        SmemLayoutV{},
        cute::make_layout(
            cute::make_shape(cute::Int<kHeadDim>{}, cute::Int<kBlockN>{},
                             cute::Int<kNumStagesKV>{}),
            cute::make_stride(cute::Int<kBlockN>{}, cute::_1{},
                              cute::Int<cute::size(SmemLayoutV{}(
                                  cute::_, cute::_, cute::_0{}))>{}))));

    using SmemLayoutScale = cute::Layout<
        cute::Shape<cute::Int<kDequantScalePackSize>, cute::Int<KStageBlock>>>;

    using SharedStorage =
        ::halops::SharedStorageFp8<kNumStagesKV, KStageBlock, kBlockM, Element,
                                   SmemLayoutQ, SmemLayoutK, SmemLayoutScale>;
    using MainloopPipeline = typename cutlass::PipelineTmaAsync<kNumStagesKV>;
};

}  // namespace halops
