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
#include "kernel/gemm/gemm_utils.h"
#include "kernel/gemm/kernel_traits.cuh"
#include "kernel/gemm/tile_scheduler.cuh"

namespace halops {

template <typename KernelTraits>
struct CollectiveMainloop {
    using GmemTiledLoad =
        cute::conditional_t<KernelTraits::KNumCTA == 1, cute::SM90_TMA_LOAD,
                            cute::SM100_TMA_2SM_LOAD_MULTICAST>;

    using TiledMma = typename KernelTraits::TiledMma;
    using TiledMma_SF = typename KernelTraits::TiledMma_SF;
    using TiledMmaForSF = typename KernelTraits::TiledMmaForSF;
    // BF16 (kind::f16) has no block-scale factors; SF paths compile out.
    static constexpr bool KIsBF16 = KernelTraits::KIsBF16;
    using AtomThrID = typename TiledMma::AtomThrID;
    using UtccpOp = cute::conditional_t<KernelTraits::KNumCTA == 1,
                                        cute::SM100_UTCCP_4x32dp128bit_1cta,
                                        cute::SM100_UTCCP_4x32dp128bit_2cta>;

    using TileShape = typename KernelTraits::TileShape;
    using TileShape_SF = typename KernelTraits::TileShape_SF;
    // Stand-in tile shape the dummy SF atom accepts (== TileShape for FP8/FP4;
    // see kernel_traits.h).
    using SFTileShape = typename KernelTraits::SFTileShape;
    using ClusterShape_MNK = typename KernelTraits::ClusterShape_MNK;
    using EpilogueTile = typename KernelTraits::EpilogueTile;
    using AtomThrShapeMNK =
        cute::Shape<decltype(cute::shape<0>(
                        typename TiledMma::ThrLayoutVMNK{})),
                    cute::_1, cute::_1>;

    using TypeA = typename KernelTraits::TypeA;
    using TypeB = typename KernelTraits::TypeB;
    using TypeSF = typename KernelTraits::TypeSF;

    using SmemLayoutAtomA = typename KernelTraits::SmemLayoutAtomA;
    using SmemLayoutAtomB = typename KernelTraits::SmemLayoutAtomB;
    using SmemLayoutA = typename KernelTraits::SmemLayoutA;
    using SmemLayoutB = typename KernelTraits::SmemLayoutB;
    using SmemLayoutAtomSFA = typename KernelTraits::SmemLayoutAtomSFA;
    using SmemLayoutAtomSFB = typename KernelTraits::SmemLayoutAtomSFB;
    using SmemLayoutSFA = typename KernelTraits::SmemLayoutSFA;
    using SmemLayoutSFB = typename KernelTraits::SmemLayoutSFB;

    using ShapeA = cute::Shape<int32_t, int32_t, int32_t>;
    using StrideA = cute::Shape<int32_t, cute::_1, int32_t>;
    using LayoutA = cute::Layout<ShapeA, StrideA>;

    using ShapeB = cute::Shape<int32_t, int32_t, int32_t>;
    using StrideB = cute::Shape<int32_t, cute::_1, int32_t>;
    using LayoutB = cute::Layout<ShapeB, StrideB>;

    using LayoutSFA = typename KernelTraits::LayoutSFA;
    using LayoutSFB = typename KernelTraits::LayoutSFB;

    using ClusterLayout_VMNK = decltype(cute::tiled_divide(
        cute::make_layout(ClusterShape_MNK{}),
        cute::make_tile(typename TiledMma::AtomThrID{})));

    static constexpr bool IsSfbUnaligned = KernelTraits::IsSfbUnaligned;
    static constexpr int KSfbAtoms = KernelTraits::KSfbAtoms;
    // Bytes of one SfAtom: 128 MN rows x four scale values.
    static constexpr int kSfbAtomBytes = 512;
    // 224-column tiles: half of them start on an odd TMEM column, so a warp in
    // every CTA rewrites the stage (a 32-row rotation) before UTCCP reads it.
    static constexpr bool IsSfbRowShift = KernelTraits::IsSfbRowShift;
    using GmemTiledLoadSfb =
        cute::conditional_t<KernelTraits::IsSfbUnaligned &&
                                KernelTraits::KNumCTA == 2,
                            cute::SM100_TMA_2SM_LOAD, GmemTiledLoad>;
    using ClusterShapeSfb =
        cute::conditional_t<KernelTraits::IsSfbUnaligned,
                            cute::Shape<cute::_1, cute::_1, cute::_1>,
                            ClusterShape_MNK>;
    using ClusterLayoutSfb_VMNK = decltype(cute::tiled_divide(
        cute::make_layout(ClusterShapeSfb{}),
        cute::make_tile(typename TiledMma_SF::AtomThrID{})));
    using TMALoad_A = decltype(cute::make_tma_atom_A_sm100(
        GmemTiledLoad{},
        cute::make_tensor(
            cute::make_gmem_ptr(
                static_cast<typename TiledMma::ValTypeA const*>(nullptr)),
            cute::repeat_like(StrideA{}, int32_t(0)), StrideA{}),
        SmemLayoutA{}(cute::_, cute::_, cute::_, cute::Int<0>{}), TileShape{},
        TiledMma{}, ClusterLayout_VMNK{}));
    using TMALoad_B = decltype(cute::make_tma_atom_B_sm100(
        GmemTiledLoad{},
        cute::make_tensor(
            cute::make_gmem_ptr(
                static_cast<typename TiledMma::ValTypeB const*>(nullptr)),
            cute::repeat_like(StrideB{}, int32_t(0)), StrideB{}),
        SmemLayoutB{}(cute::_, cute::_, cute::_, cute::Int<0>{}), TileShape{},
        TiledMma{}, ClusterLayout_VMNK{}));
    using TMALoad_SFA = decltype(cute::make_tma_atom_A_sm100<uint16_t>(
        GmemTiledLoad{},
        cute::make_tensor(static_cast<TypeSF const*>(nullptr), LayoutSFA{}),
        SmemLayoutSFA{}(cute::_, cute::_, cute::_, cute::Int<0>{}),
        SFTileShape{}, TiledMmaForSF{}, ClusterLayout_VMNK{}));

    // Unaligned-N SFB loads through `sfb_run_desc`.
    struct NoSfbTma {};
    static auto make_sfb_tma_atom() {
        if constexpr (IsSfbUnaligned) {
            return NoSfbTma{};
        } else {
            return cute::make_tma_atom_B_sm100<uint16_t>(
                GmemTiledLoadSfb{},
                cute::make_tensor(static_cast<TypeSF const*>(nullptr),
                                  LayoutSFB{}),
                SmemLayoutSFB{}(cute::_, cute::_, cute::_, cute::Int<0>{}),
                TileShape_SF{}, TiledMma_SF{}, ClusterLayoutSfb_VMNK{});
        }
    }
    using TMALoad_SFB = decltype(make_sfb_tma_atom());

    static constexpr bool SwapAB = KernelTraits::SwapAB;

    static constexpr bool IsCtaNSmall = KernelTraits::IsCtaNSmall;
    static constexpr int IsCtaN64 = KernelTraits::IsCtaN64;
    static constexpr int VSPerCtaN = IsCtaN64 ? 2 : 1;

    using SharedStorage = typename KernelTraits::SharedStorage;
    using MainloopPipeline = typename KernelTraits::MainloopPipeline;
    using MainloopPipelineState = typename KernelTraits::MainloopPipelineState;
    using AccumulatorPipeline = typename KernelTraits::AccumulatorPipeline;
    using AccumulatorPipelineState =
        typename KernelTraits::AccumulatorPipelineState;

    // With a narrow A box the pipeline must expect the bytes the TMA actually
    // delivers, not the full stage.
    static constexpr uint32_t TmaTransactionBytesAFull =
        cutlass::bits_to_bytes(static_cast<uint32_t>(
            cute::cosize(cute::take<0, 3>(SmemLayoutA{})) *
            cute::sizeof_bits_v<TypeA> * cute::size(AtomThrShapeMNK{})));
    static constexpr uint32_t TmaTransactionBytesA = TmaTransactionBytesAFull;
    // Bytes the A/B/SF TMA actually delivers for this launch.
    CUTLASS_DEVICE static uint32_t TmaTransactionBytesABFor(int m) {
        if constexpr (KernelTraits::KNarrowA) {
            if (m > 0 && m < static_cast<int>(KernelTraits::KBlockM)) {
                return TmaTransactionBytesAB - TmaTransactionBytesAFull +
                       static_cast<uint32_t>(m) * KernelTraits::KBlockK *
                           sizeof(TypeA);
            }
        }
        return TmaTransactionBytesAB;
    }
    static constexpr uint32_t TmaTransactionBytesB =
        cutlass::bits_to_bytes(static_cast<uint32_t>(
            cute::cosize(cute::take<0, 3>(SmemLayoutB{})) *
            cute::sizeof_bits_v<TypeB> * cute::size(AtomThrShapeMNK{})));
    static constexpr uint32_t TmaTransactionBytesSFA =
        cutlass::bits_to_bytes(static_cast<uint32_t>(
            cute::cosize(cute::take<0, 3>(SmemLayoutSFA{})) *
            cute::sizeof_bits_v<TypeSF> * cute::size(AtomThrShapeMNK{})));
    // Bytes one stage's SFB contributes to this CTA's own SMEM.
    static constexpr uint32_t TmaTransactionBytesSfbStage =
        cutlass::bits_to_bytes(static_cast<uint32_t>(
            cute::cosize(cute::take<0, 3>(SmemLayoutSFB{})) *
            cute::sizeof_bits_v<TypeSF>));

    static constexpr uint32_t TmaTransactionBytesSFB =
        TmaTransactionBytesSfbStage * cute::size(AtomThrShapeMNK{});
    // 224-column tiles: each CTA's rewriter signals its own copy (and rewrite)
    // with 16 transaction bytes, so the barrier's arrival count is unchanged.
    static constexpr uint32_t KSfbReadyTx = 16;
    static constexpr uint32_t TmaTransactionBytesAB =
        KIsBF16 ? (TmaTransactionBytesA + TmaTransactionBytesB)
                : (TmaTransactionBytesA + TmaTransactionBytesB +
                   TmaTransactionBytesSFA +
                   (IsSfbRowShift ? KSfbReadyTx * KernelTraits::KNumCTA
                                  : TmaTransactionBytesSFB));

    struct Arguments {
        TypeA const* ptr_a;
        TypeB const* ptr_b;
        TypeSF const* ptr_sfa;
        TypeSF const* ptr_sfb;

        LayoutA layout_A;
        LayoutB layout_B;
        LayoutSFA layout_SFA;
        LayoutSFB layout_SFB;
        int k;
        int m;
        int num_experts = 1;
    };

    struct Params {
        LayoutA layout_A;
        LayoutB layout_B;
        LayoutSFA layout_SFA;
        LayoutSFB layout_SFB;
        TMALoad_A tma_load_A;
        TMALoad_B tma_load_B;
        TMALoad_SFA tma_load_SFA;
        TMALoad_SFB tma_load_SFB;
        // IsSfbUnaligned only: SFB as (512-byte atom, k-block, n-atom), boxed
        // to a run of KSfbAtoms atoms.  Unused otherwise.
        cute::TmaDescriptor sfb_run_desc;
        // Hand-built A descriptor whose box M is min(m, KBlockM) rows.
        // Used when KernelTraits::KNarrowA and m < KBlockM.
        cute::TmaDescriptor a_run_desc;
        int m;
        int k;
        int num_experts;
    };

    CUTLASS_DEVICE static auto init_tmem_tensors() {
        TiledMma tiled_mma;
        auto acc_shape =
            cute::partition_shape_C(TiledMma{}, cute::take<0, 2>(TileShape{}));

        static constexpr int OverlapCols = KernelTraits::KTmemOverlapCols;
        static constexpr bool UseCustomOverlap =
            (OverlapCols > 0 && KernelTraits::KEpilogueStages > 1);
        static constexpr bool IsOverlappingAccum = UseCustomOverlap;

        // When using custom overlap, substitute N with KTmemOverlapCols.
        using OverlapTile = cute::conditional_t<
            UseCustomOverlap,
            cute::Shape<decltype(cute::size<0>(EpilogueTile{})),
                        cute::Int<OverlapCols>>,
            EpilogueTile>;

        cute::Tensor tmem_accumulators =
            cutlass::detail::make_sm100_accumulator<
                KernelTraits::KEpilogueStages, IsOverlappingAccum>(
                tiled_mma, acc_shape, OverlapTile{});

        cute::Tensor tmem_SFA =
            cute::make_tensor<typename KernelTraits::FrgTypeSFA>(
                cute::shape(SmemLayoutAtomSFA{}));
        cute::Tensor tmem_SFB =
            cute::make_tensor<typename KernelTraits::FrgTypeSFB>(
                cute::shape(SmemLayoutAtomSFB{}));

        TmemStorage<decltype(tmem_accumulators), decltype(tmem_SFA),
                    decltype(tmem_SFB)>
            tmem_storage;
        tmem_storage.tmem_accumulators = tmem_accumulators;
        tmem_storage.tmem_SFA = tmem_SFA;
        tmem_storage.tmem_SFB = tmem_SFB;
        return tmem_storage;
    }

    template <class TmemStorage>
    CUTLASS_DEVICE static void set_tmem_offsets(TmemStorage& tmem_storage,
                                                uint32_t tmem_base_addr) {
        tmem_storage.tmem_accumulators.data() = tmem_base_addr;
        tmem_storage.tmem_SFA.data() =
            tmem_storage.tmem_accumulators.data().get() +
            cutlass::detail::find_tmem_tensor_col_offset(
                tmem_storage.tmem_accumulators);
        tmem_storage.tmem_SFB.data() =
            tmem_storage.tmem_SFA.data().get() +
            cutlass::detail::find_tmem_tensor_col_offset(tmem_storage.tmem_SFA);
    }

    static Params to_underlying_arguments(Arguments const& args) {
        int k = args.k;
        cute::Layout cluster_layout_vmnk =
            cute::tiled_divide(cute::make_layout(ClusterShape_MNK{}),
                               cute::make_tile(typename TiledMma::AtomThrID{}));
        cute::Layout cluster_layout_sfb_vmnk = cute::tiled_divide(
            cute::make_layout(ClusterShapeSfb{}),
            cute::make_tile(typename TiledMma_SF::AtomThrID{}));

        cute::Tensor tensor_a = cute::make_tensor(
            cute::make_gmem_ptr(
                static_cast<typename TiledMma::ValTypeA const*>(args.ptr_a)),
            args.layout_A);

        cute::Tensor tensor_b = cute::make_tensor(
            cute::make_gmem_ptr(
                static_cast<typename TiledMma::ValTypeB const*>(args.ptr_b)),
            args.layout_B);

        TMALoad_A tma_load_A = cute::make_tma_atom_A_sm100(
            GmemTiledLoad{}, tensor_a,
            SmemLayoutA{}(cute::_, cute::_, cute::_, cute::Int<0>{}),
            TileShape{}, TiledMma{}, cluster_layout_vmnk);

        TMALoad_B tma_load_B = cute::make_tma_atom_B_sm100(
            GmemTiledLoad{}, tensor_b,
            SmemLayoutB{}(cute::_, cute::_, cute::_, cute::Int<0>{}),
            TileShape{}, TiledMma{}, cluster_layout_vmnk);

        // BF16: SF pointers are null (encoding a TMA atom from them would
        // fault on the host), so return default-constructed SF atoms.
        if constexpr (KIsBF16) {
            return {args.layout_A,         args.layout_B, args.layout_SFA,
                    args.layout_SFB,       tma_load_A,    tma_load_B,
                    TMALoad_SFA{},         TMALoad_SFB{}, cute::TmaDescriptor{},
                    cute::TmaDescriptor{}, args.m,        k,
                    args.num_experts};
        } else {
            cute::Tensor tensor_sfa =
                cute::make_tensor(args.ptr_sfa, args.layout_SFA);
            cute::Tensor tensor_sfb =
                cute::make_tensor(args.ptr_sfb, args.layout_SFB);

            TMALoad_SFA tma_load_sfa = cute::make_tma_atom_A_sm100<uint16_t>(
                GmemTiledLoad{}, tensor_sfa,
                SmemLayoutSFA{}(cute::_, cute::_, cute::_, cute::Int<0>{}),
                TileShape{}, TiledMmaForSF{}, cluster_layout_vmnk);

            TMALoad_SFB tma_load_sfb{};
            if constexpr (!IsSfbUnaligned) {
                tma_load_sfb = cute::make_tma_atom_B_sm100<uint16_t>(
                    GmemTiledLoadSfb{}, tensor_sfb,
                    SmemLayoutSFB{}(cute::_, cute::_, cute::_, cute::Int<0>{}),
                    TileShape_SF{}, TiledMma_SF{}, cluster_layout_sfb_vmnk);
            }

            cute::TmaDescriptor a_run_desc{};
            if constexpr (KernelTraits::KNarrowA) {
                const uint64_t a_k = static_cast<uint64_t>(k);
                const uint64_t a_m =
                    static_cast<uint64_t>(cute::size<0>(args.layout_A));
                const uint64_t a_ld =
                    static_cast<uint64_t>(cute::stride<0>(args.layout_A)) *
                    sizeof(TypeA);
                const uint64_t box_m = std::min<uint64_t>(
                    a_m, static_cast<uint64_t>(KernelTraits::KBlockM));
                cuuint64_t dims[2] = {a_k, a_m};
                cuuint64_t strides[1] = {a_ld};
                cuuint32_t box[2] = {
                    static_cast<cuuint32_t>(KernelTraits::KBlockK),
                    static_cast<cuuint32_t>(box_m)};
                cuuint32_t elem_strides[2] = {1, 1};
                CUresult r = cuTensorMapEncodeTiled(
                    reinterpret_cast<CUtensorMap*>(&a_run_desc),
                    CU_TENSOR_MAP_DATA_TYPE_UINT8, 2,
                    const_cast<typename TiledMma::ValTypeA*>(args.ptr_a), dims,
                    strides, box, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
                    CU_TENSOR_MAP_SWIZZLE_128B,
                    CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
                TORCH_CHECK(r == CUDA_SUCCESS,
                            "cuTensorMapEncodeTiled(narrow A) failed: ", r);
            }
            cute::TmaDescriptor sfb_run_desc{};
            if constexpr (IsSfbUnaligned) {
                // tile_atom_to_shape_SFB orders atoms K-block first, then N.
                const uint64_t k_blks = cute::size<1, 1>(args.layout_SFB);

                // Grouped modes lay the experts out after the N atoms, so they
                // fold into the same run dimension (one stride per atom).
                const uint64_t n_atoms = static_cast<uint64_t>(
                    cute::ceil_div(cute::size<0>(args.layout_B), 128));
                const uint64_t n_atoms_all =
                    n_atoms * static_cast<uint64_t>(args.num_experts);
                cuuint64_t dims[3] = {kSfbAtomBytes / 2, k_blks, n_atoms_all};
                cuuint64_t strides[2] = {kSfbAtomBytes, k_blks * kSfbAtomBytes};
                cuuint32_t box[3] = {kSfbAtomBytes / 2, 1, KSfbAtoms};
                cuuint32_t elem_strides[3] = {1, 1, 1};
                CUresult r = cuTensorMapEncodeTiled(
                    reinterpret_cast<CUtensorMap*>(&sfb_run_desc),
                    CU_TENSOR_MAP_DATA_TYPE_UINT16, 3,
                    const_cast<TypeSF*>(args.ptr_sfb), dims, strides, box,
                    elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
                    CU_TENSOR_MAP_SWIZZLE_NONE,
                    CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
                TORCH_CHECK(r == CUDA_SUCCESS,
                            "cuTensorMapEncodeTiled(SFB atom run) failed: ", r);
            }

            return {args.layout_A,   args.layout_B, args.layout_SFA,
                    args.layout_SFB, tma_load_A,    tma_load_B,
                    tma_load_sfa,    tma_load_sfb,  sfb_run_desc,
                    a_run_desc,      args.m,        k,
                    args.num_experts};
        }
    }

    CUTLASS_DEVICE void tma_loadAB(Params const& mainloop_params,
                                   MainloopPipeline& mainloop_pipeline,
                                   SharedStorage& shared_storage,
                                   TiledMma& tiled_mma,
                                   TileScheduler<KernelTraits>& scheduler) {
        tma_loadAB_impl(mainloop_params, mainloop_pipeline, mainloop_pipeline,
                        shared_storage, tiled_mma, scheduler);
    }

  private:
    template <typename ABPipeline>
    CUTLASS_DEVICE void tma_loadAB_impl(
        Params const& mainloop_params, ABPipeline& ab_pipeline, ABPipeline&,
        SharedStorage& shared_storage, TiledMma& tiled_mma,
        TileScheduler<KernelTraits>& scheduler) {
        bool select_thr = cute::elect_one_sync();

        // Unused: the per-work-item K range comes from the scheduler.
        [[maybe_unused]] uint32_t tile_num_k =
            cute::ceil_div(mainloop_params.k, cute::size<2>(TileShape{}));

        int block_rank_in_cluster = cute::block_rank_in_cluster();

        cute::Layout cluster_layout_vmnk =
            cute::tiled_divide(cute::make_layout(ClusterShape_MNK{}),
                               cute::make_tile(typename TiledMma::AtomThrID{}));
        auto cta_coord_vmnk =
            cluster_layout_vmnk.get_flat_coord(block_rank_in_cluster);

        cute::Layout cluster_layout_sfb_vmnk = cute::tiled_divide(
            cute::make_layout(ClusterShapeSfb{}),
            cute::make_tile(typename TiledMma_SF::AtomThrID{}));
        auto cta_coord_sfb_vmnk = cluster_layout_sfb_vmnk.get_flat_coord(
            IsSfbUnaligned ? 0 : block_rank_in_cluster);

        // Each warp only computes its own multicast mask
        const bool narrow_a =
            KernelTraits::KNarrowA &&
            mainloop_params.m < static_cast<int>(KernelTraits::KBlockM);
        uint16_t mcast_mask_a = cute::create_tma_multicast_mask<2>(
            cluster_layout_vmnk, cta_coord_vmnk);
        uint16_t mcast_mask_b = cute::create_tma_multicast_mask<1>(
            cluster_layout_vmnk, cta_coord_vmnk);
        uint16_t mcast_mask_sfa = cute::create_tma_multicast_mask<2>(
            cluster_layout_vmnk, cta_coord_vmnk);
        uint16_t mcast_mask_sfb = cute::create_tma_multicast_mask<1>(
            cluster_layout_sfb_vmnk, cta_coord_sfb_vmnk);

        // --- AB tensors ---
        cute::Tensor A = [&]() {
            if constexpr (SwapAB) {
                return mainloop_params.tma_load_A.get_tma_tensor(
                    mainloop_params.layout_A.shape());
            } else {
                return mainloop_params.tma_load_A.get_tma_tensor(
                    mainloop_params.layout_A.shape())(cute::_, cute::_,
                                                      cute::Int<0>{});
            }
        }();
        cute::Tensor B = [&]() {
            if constexpr (SwapAB) {
                return mainloop_params.tma_load_B.get_tma_tensor(
                    mainloop_params.layout_B.shape())(cute::_, cute::_,
                                                      cute::Int<0>{});
            } else {
                return mainloop_params.tma_load_B.get_tma_tensor(
                    mainloop_params.layout_B.shape());
            }
        }();
        cute::Tensor cta_sA = shared_storage.tensor_sA();
        cute::Tensor cta_sB = shared_storage.tensor_sB();
        // Slice by the atom's thread coordinate, not the flat cluster rank:
        // with a 1SM atom in a multi-CTA cluster the extra CTAs would index
        // past the atom's thread layout and get invalid A/B fragments.
        cute::ThrMMA cta_mma = tiled_mma.get_slice(
            block_rank_in_cluster % cute::size(typename TiledMma::AtomThrID{}));

        cute::Tensor gA = cute::local_tile(
            A, TileShape{}, cute::make_coord(cute::_, cute::_, cute::_),
            cute::Step<cute::_1, cute::X, cute::_1>{});
        cute::Tensor gB = cute::local_tile(
            B, TileShape{}, cute::make_coord(cute::_, cute::_, cute::_),
            cute::Step<cute::X, cute::_1, cute::_1>{});

        cute::Tensor cta_gA = cta_mma.partition_A(gA);
        cute::Tensor cta_gB = cta_mma.partition_B(gB);

        auto [tma_gA, tma_sA] = cute::tma_partition(
            mainloop_params.tma_load_A, cute::get<2>(cta_coord_vmnk),
            cute::make_layout(cute::size<2>(cluster_layout_vmnk)),
            cute::group_modes<0, 3>(cta_sA), cute::group_modes<0, 3>(cta_gA));
        auto [tma_gB, tma_sB] = cute::tma_partition(
            mainloop_params.tma_load_B, cute::get<1>(cta_coord_vmnk),
            cute::make_layout(cute::size<1>(cluster_layout_vmnk)),
            cute::group_modes<0, 3>(cta_sB), cute::group_modes<0, 3>(cta_gB));

        // --- SF tensors ---
        cute::Tensor SFA = [&]() {
            if constexpr (SwapAB) {
                return mainloop_params.tma_load_SFA.get_tma_tensor(
                    mainloop_params.layout_SFA.shape());
            } else {
                return mainloop_params.tma_load_SFA.get_tma_tensor(
                    mainloop_params.layout_SFA.shape())(cute::_, cute::_,
                                                        cute::Int<0>{});
            }
        }();
        // Unaligned-N SFB never goes through the CuTe atom.
        auto SFB = [&]() {
            if constexpr (IsSfbUnaligned) {
                return 0;
            } else if constexpr (SwapAB) {
                if constexpr (IsCtaN64) {
                    cute::Tensor mSFB_tmp =
                        mainloop_params.tma_load_SFB.get_tma_tensor(
                            mainloop_params.layout_SFB.shape())(
                            cute::_, cute::_, cute::Int<0>{});
                    auto new_shape = cute::make_shape(
                        cute::make_shape(
                            cute::shape<0, 0>(mSFB_tmp),
                            cute::make_shape(cute::Int<VSPerCtaN>{},
                                             cute::shape<0, 1>(mSFB_tmp))),
                        cute::shape<1>(mSFB_tmp));
                    auto new_stride = cute::make_stride(
                        cute::make_stride(
                            cute::stride<0, 0>(mSFB_tmp),
                            cute::make_stride(cute::_0{},
                                              cute::stride<0, 1>(mSFB_tmp))),
                        cute::stride<1>(mSFB_tmp));
                    return cute::make_tensor(
                        mSFB_tmp.data(),
                        cute::make_layout(new_shape, new_stride));
                } else {
                    return mainloop_params.tma_load_SFB.get_tma_tensor(
                        mainloop_params.layout_SFB.shape())(cute::_, cute::_,
                                                            cute::Int<0>{});
                }
            } else {
                if constexpr (IsCtaN64) {
                    cute::Tensor mSFB_tmp =
                        mainloop_params.tma_load_SFB.get_tma_tensor(
                            mainloop_params.layout_SFB.shape());
                    auto new_shape = cute::make_shape(
                        cute::make_shape(
                            cute::shape<0, 0>(mSFB_tmp),
                            cute::make_shape(cute::Int<VSPerCtaN>{},
                                             cute::shape<0, 1>(mSFB_tmp))),
                        cute::shape<1>(mSFB_tmp), cute::shape<2>(mSFB_tmp));
                    auto new_stride = cute::make_stride(
                        cute::make_stride(
                            cute::stride<0, 0>(mSFB_tmp),
                            cute::make_stride(cute::_0{},
                                              cute::stride<0, 1>(mSFB_tmp))),
                        cute::stride<1>(mSFB_tmp), cute::stride<2>(mSFB_tmp));
                    return cute::make_tensor(
                        mSFB_tmp.data(),
                        cute::make_layout(new_shape, new_stride));
                } else {
                    return mainloop_params.tma_load_SFB.get_tma_tensor(
                        mainloop_params.layout_SFB.shape());
                }
            }
        }();
        cute::Tensor cta_sSFA = shared_storage.tensor_sSFA();
        cute::Tensor cta_sSFB = shared_storage.tensor_sSFB();
        cute::ThrMMA cta_mma_sfb = TiledMma_SF{}.get_slice(
            block_rank_in_cluster %
            cute::size(typename TiledMma_SF::AtomThrID{}));

        // SF tiling must use SFTileShape + TiledMmaForSF consistently: for BF16
        // they are dummy stand-ins (kernel_traits.h) and mixing them with the
        // real ones breaks tma_partition's mode-0 sizes, even though no SF
        // load is issued. Identical to the real ones for FP8/FP4.
        cute::ThrMMA cta_mma_sfa = TiledMmaForSF{}.get_slice(
            block_rank_in_cluster %
            cute::size(typename TiledMmaForSF::AtomThrID{}));
        cute::Tensor gSFA = cute::local_tile(
            SFA, SFTileShape{}, cute::make_coord(cute::_, cute::_, cute::_),
            cute::Step<cute::_1, cute::X, cute::_1>{});
        cute::Tensor cta_gSFA = cta_mma_sfa.partition_A(gSFA);

        auto [tma_gSFA, tma_sSFA] = cute::tma_partition(
            mainloop_params.tma_load_SFA, cute::get<2>(cta_coord_vmnk),
            cute::make_layout(cute::size<2>(cluster_layout_vmnk)),
            cute::group_modes<0, 3>(cta_sSFA),
            cute::group_modes<0, 3>(cta_gSFA));
        // Unaligned-N SFB is loaded through sfb_run_desc; the CuTe atom serves
        // the aligned SfAtom path only.
        auto [tma_gSFB, tma_sSFB] = [&]() {
            if constexpr (IsSfbUnaligned) {
                return cute::make_tuple(0, 0);
            } else {
                cute::Tensor gSFB = cute::local_tile(
                    SFB, TileShape_SF{},
                    cute::make_coord(cute::_, cute::_, cute::_),
                    cute::Step<cute::X, cute::_1, cute::_1>{});
                cute::Tensor cta_gSFB = cta_mma_sfb.partition_B(gSFB);
                return cute::tma_partition(
                    mainloop_params.tma_load_SFB,
                    cute::get<1>(cta_coord_sfb_vmnk),
                    cute::make_layout(cute::size<1>(cluster_layout_sfb_vmnk)),
                    cute::group_modes<0, 3>(cta_sSFB),
                    cute::group_modes<0, 3>(cta_gSFB));
            }
        }();

        // --- Pipeline state ---
        using BarrierType = typename ABPipeline::ProducerBarrierType;

        using ActivePipeline = ABPipeline;
        using ActiveState = typename ABPipeline::PipelineState;
        ActiveState mainloop_pipe_producer_state =
            cutlass::make_producer_start_state<ABPipeline>();

        uint32_t prologue_stage_idx = 0;

        while (scheduler.is_valid()) {
            uint32_t m_idx = scheduler.m_idx();
            uint32_t n_idx = scheduler.n_idx();
            uint32_t expert_idx = scheduler.expert_idx();
            // Stream-K: a work item may cover only a K-slice. Producer and
            // mma() consumer both take the range from the scheduler, so stage
            // counts always match.
            const uint32_t k_lo = scheduler.k_begin();
            const uint32_t k_hi = scheduler.k_end();
            for (uint32_t k_tile_id = k_lo; k_tile_id < k_hi; ++k_tile_id) {
                ab_pipeline.producer_acquire(mainloop_pipe_producer_state);
                BarrierType* tma_barrier = ab_pipeline.producer_get_barrier(
                    mainloop_pipe_producer_state);
                prologue_stage_idx = mainloop_pipe_producer_state.index();

                // Issue A, B (+ SFA, SFB unless BF16) TMA copies on the same
                // barrier.
                if (select_thr) {
                    if constexpr (SwapAB) {
                        copy(mainloop_params.tma_load_A.with(*tma_barrier,
                                                             mcast_mask_a),
                             tma_gA(cute::_, m_idx, k_tile_id, expert_idx),
                             tma_sA(cute::_,
                                    (cute::Int<0>{}, prologue_stage_idx)));
                        copy(mainloop_params.tma_load_B.with(*tma_barrier,
                                                             mcast_mask_b),
                             tma_gB(cute::_, n_idx, k_tile_id),
                             tma_sB(cute::_,
                                    (cute::Int<0>{}, prologue_stage_idx)));
                        if constexpr (!KIsBF16) {
                            copy(
                                mainloop_params.tma_load_SFA.with(
                                    *tma_barrier, mcast_mask_sfa),
                                tma_gSFA(cute::_, m_idx, k_tile_id, expert_idx),
                                tma_sSFA(cute::_,
                                         (cute::Int<0>{}, prologue_stage_idx)));
                            copy(mainloop_params.tma_load_SFB.with(
                                     *tma_barrier, mcast_mask_sfb),
                                 tma_gSFB(cute::_, n_idx, k_tile_id),
                                 tma_sSFB(cute::_, (cute::Int<0>{},
                                                    prologue_stage_idx)));
                        }
                    } else {
                        // Per-CTA SFB copy on `sf_full` (a rewriter warp fixes
                        // it up); issued first to overlap the A/B transfer.
                        if constexpr (IsSfbRowShift) {
                            constexpr int kSfbStageElems = cute::size(
                                cute::filter_zeros(SmemLayoutAtomSFB{}));
                            cutlass::arch::ClusterTransactionBarrier::
                                arrive_and_expect_tx(
                                    &shared_storage.sf_full[prologue_stage_idx],
                                    TmaTransactionBytesSfbStage);
                            cute::SM90_TMA_LOAD_3D::copy(
                                &mainloop_params.sfb_run_desc,
                                reinterpret_cast<uint64_t*>(
                                    &shared_storage
                                         .sf_full[prologue_stage_idx]),
                                static_cast<uint64_t>(
                                    cute::TMA::CacheHintSm90::EVICT_NORMAL),
                                shared_storage.SFB.begin() +
                                    prologue_stage_idx * kSfbStageElems,
                                0, static_cast<int32_t>(k_tile_id),
                                static_cast<int32_t>(
                                    expert_idx *
                                        cute::ceil_div(
                                            cute::size<0>(
                                                mainloop_params.layout_B),
                                            128) +
                                    n_idx * cute::size<1>(TileShape{}) / 128));
                        }
                        // When NarrowA is used, the UMMA still
                        // reads KBlockM rows, so the tail rows are stale and
                        // their D rows are discarded by the epilogue.
                        auto issue_full_a = [&]() {
                            cute::copy(mainloop_params.tma_load_A.with(
                                           *tma_barrier, mcast_mask_a),
                                       tma_gA(cute::_, m_idx, k_tile_id),
                                       tma_sA(cute::_, (cute::Int<0>{},
                                                        prologue_stage_idx)));
                        };
                        if constexpr (KernelTraits::KNarrowA) {
                            if (narrow_a) {
                                cute::SM90_TMA_LOAD_2D::copy(
                                    &mainloop_params.a_run_desc,
                                    reinterpret_cast<uint64_t*>(tma_barrier),
                                    static_cast<uint64_t>(
                                        cute::TMA::CacheHintSm90::EVICT_NORMAL),
                                    shared_storage.A.begin() +
                                        prologue_stage_idx *
                                            (KernelTraits::KBlockM *
                                             KernelTraits::KBlockK),
                                    static_cast<int32_t>(k_tile_id *
                                                         KernelTraits::KBlockK),
                                    static_cast<int32_t>(
                                        m_idx * KernelTraits::KBlockM));
                            } else {
                                issue_full_a();
                            }
                        } else {
                            issue_full_a();
                        }
                        cute::copy(
                            mainloop_params.tma_load_B.with(*tma_barrier,
                                                            mcast_mask_b),
                            tma_gB(cute::_, n_idx, k_tile_id, expert_idx),
                            tma_sB(cute::_,
                                   (cute::Int<0>{}, prologue_stage_idx)));
                        if constexpr (!KIsBF16) {
                            cute::copy(mainloop_params.tma_load_SFA.with(
                                           *tma_barrier, mcast_mask_sfa),
                                       tma_gSFA(cute::_, m_idx, k_tile_id),
                                       tma_sSFA(cute::_, (cute::Int<0>{},
                                                          prologue_stage_idx)));
                        }
                        if constexpr (IsSfbRowShift) {
                            // Issued first, see above.
                        } else if constexpr (IsSfbUnaligned) {
                            // Atoms past N are out of bounds (zero-filled);
                            // they only feed overhang columns never stored.
                            constexpr int kSfbStageElems = cute::size(
                                cute::filter_zeros(SmemLayoutAtomSFB{}));
                            {
                                using SfbLoad3D = cute::conditional_t<
                                    KernelTraits::KNumCTA == 1,
                                    cute::SM90_TMA_LOAD_3D,
                                    cute::SM100_TMA_2SM_LOAD_3D>;
                                SfbLoad3D::copy(
                                    &mainloop_params.sfb_run_desc, tma_barrier,
                                    static_cast<uint64_t>(
                                        cute::TMA::CacheHintSm90::EVICT_NORMAL),
                                    shared_storage.SFB.begin() +
                                        prologue_stage_idx * kSfbStageElems,
                                    0, static_cast<int32_t>(k_tile_id),
                                    static_cast<int32_t>(
                                        n_idx * cute::size<1>(TileShape{}) /
                                        128));
                            }
                        } else if constexpr (!KIsBF16) {
                            cute::copy(
                                mainloop_params.tma_load_SFB.with(
                                    *tma_barrier, mcast_mask_sfb),
                                tma_gSFB(cute::_, n_idx, k_tile_id, expert_idx),
                                tma_sSFB(cute::_,
                                         (cute::Int<0>{}, prologue_stage_idx)));
                        }
                    }
                }
                ++mainloop_pipe_producer_state;
            }
            scheduler.advance();
        }
        ab_pipeline.producer_tail(mainloop_pipe_producer_state);
    }

  public:
    // Shift one SFB stage up by 32 rows (odd in-atom offset -> even): rotate
    // the atom's four 32-row words, the tail one coming from the next atom.
    CUTLASS_DEVICE static void sfb_shift_rows_stage(SharedStorage& storage,
                                                    uint32_t stage_idx) {
        const int lane = cutlass::canonical_lane_idx();
        char* base = reinterpret_cast<char*>(storage.SFB.begin()) +
                     stage_idx * (KSfbAtoms * kSfbAtomBytes);
        // Read every atom before writing any: one warp sync for the stage.
        uint4 v[KSfbAtoms];
        CUTLASS_PRAGMA_UNROLL
        for (int atom = 0; atom < KSfbAtoms; ++atom) {
            v[atom] = *reinterpret_cast<uint4 const*>(
                base + atom * kSfbAtomBytes + lane * 16);
        }
        __syncwarp();
        CUTLASS_PRAGMA_UNROLL
        for (int atom = 0; atom < KSfbAtoms; ++atom) {
            uint4 w;
            w.x = v[atom].y;
            w.y = v[atom].z;
            w.z = v[atom].w;
            w.w = (atom + 1 < KSfbAtoms) ? v[atom + 1].x : 0u;
            *reinterpret_cast<uint4*>(base + atom * kSfbAtomBytes + lane * 16) =
                w;
        }
    }

    // Rewriter warps split the stages: wait each owned one on `sf_full`, then
    // shift it on an odd starting column and complete `KSfbReadyTx`.
    CUTLASS_DEVICE void sfb_shift_rows(Params const& mainloop_params,
                                       SharedStorage& shared_storage,
                                       TileScheduler<KernelTraits>& scheduler,
                                       uint32_t xpose) {
        static_assert(IsSfbRowShift,
                      "only the 224-column tiles rewrite their SFB stage");
        constexpr uint32_t kWarps = KernelTraits::KSfbTransposeWarps;
        const uint32_t tile_num_k =
            cute::ceil_div(mainloop_params.k, cute::size<2>(TileShape{}));
        const uint32_t leader_rank =
            cute::block_rank_in_cluster() - scheduler.cta_v();
        MainloopPipelineState stage;
        for (auto walk = scheduler; walk.is_valid(); walk.advance()) {
            // The tile's start column inside the window, in 32-row columns.
            const bool odd =
                (((walk.n_idx() * cute::size<1>(TileShape{})) % 128) / 32) & 1;
            for (uint32_t k_tile_id = 0; k_tile_id < tile_num_k; ++k_tile_id) {
                const uint32_t s = stage.index();
                if (s % kWarps == xpose) {
                    cutlass::arch::ClusterTransactionBarrier::wait(
                        &shared_storage.sf_full[s], stage.phase());
                    if (odd) {
                        sfb_shift_rows_stage(shared_storage, s);
                        __syncwarp();
                        cutlass::arch::fence_view_async_shared();
                    }
                    if (cute::elect_one_sync()) {
                        cutlass::arch::ClusterTransactionBarrier::
                            complete_transaction(
                                reinterpret_cast<uint64_t const*>(
                                    &shared_storage.mainloop.full_barrier_[s]),
                                leader_rank, KSfbReadyTx);
                    }
                    __syncwarp();
                }
                ++stage;
            }
        }
    }

    template <class TmemStorage, bool KStreamK = false>
    CUTLASS_DEVICE void mma(Params const& mainloop_params,
                            MainloopPipeline& mainloop_pipeline,
                            AccumulatorPipeline& accumulator_pipeline,
                            SharedStorage& shared_storage,
                            TmemStorage& tmem_storage, TiledMma& tiled_mma,
                            TileScheduler<KernelTraits>& scheduler) {
        bool select_thr = cute::elect_one_sync();

        // Unused: the per-work-item K range comes from the scheduler.
        [[maybe_unused]] uint32_t tile_num_k =
            cute::ceil_div(mainloop_params.k, cute::size<2>(TileShape{}));
        uint32_t iter_k = KernelTraits::kItersK;

        int block_rank_in_cluster = cute::block_rank_in_cluster();
        // Same atom-rank rule as tma_loadAB_impl (see there).
        int cta_atom_rank =
            block_rank_in_cluster % cute::size(typename TiledMma::AtomThrID{});

        cute::ThrMMA cta_mma = tiled_mma.get_slice(cta_atom_rank);

        uint32_t prologue_stage_idx = 0;
        uint32_t epilogue_stage_idx = 0;
        // Phase parity for the cluster-scope overlap barrier (pair-UMMA only).
        // mbarrier.try_wait.parity uses this to detect phase completion.
        [[maybe_unused]] int overlap_phase = 0;
        cute::Tensor cta_sA = shared_storage.tensor_sA();
        cute::Tensor cta_sB = shared_storage.tensor_sB();

        cute::Tensor tile_rA = cta_mma.make_fragment_A(cta_sA);
        cute::Tensor tile_rB = cta_mma.make_fragment_B(cta_sB);

        cute::Tensor smem_SFA = shared_storage.tensor_sSFA();
        cute::Tensor smem_SFB = shared_storage.tensor_sSFB();
        auto smem_SFA_compact = cute::make_tensor(
            smem_SFA.data(), cute::filter_zeros(smem_SFA.layout()));
        auto smem_SFB_compact = cute::make_tensor(
            smem_SFB.data(), cute::filter_zeros(smem_SFB.layout()));

        cute::Tensor tmem_accumulators = tmem_storage.tmem_accumulators;
        cute::Tensor tmem_SFA = tmem_storage.tmem_SFA;
        cute::Tensor tmem_SFB = tmem_storage.tmem_SFB;
        auto tmem_SFA_compact = cute::make_tensor(
            tmem_SFA.data(), cute::filter_zeros(tmem_SFA.layout()));
        auto tmem_SFB_compact = cute::make_tensor(
            tmem_SFB.data(), cute::filter_zeros(tmem_SFB.layout()));
        using AtomThrID = typename TiledMma::AtomThrID;

        auto tiled_copy_s2t_SFA =
            cute::make_utccp_copy(UtccpOp{}, tmem_SFA_compact);
        auto tiled_copy_s2t_SFB =
            cute::make_utccp_copy(UtccpOp{}, tmem_SFB_compact);

        auto thr_copy_s2t_SFA = tiled_copy_s2t_SFA.get_slice(0);
        auto thr_sSFA_s2t_ = thr_copy_s2t_SFA.partition_S(smem_SFA_compact);
        auto thr_sSFA_s2t =
            cute::get_utccp_smem_desc_tensor<UtccpOp>(thr_sSFA_s2t_);
        auto thr_tSFA_s2t = thr_copy_s2t_SFA.partition_D(tmem_SFA_compact);

        auto thr_copy_s2t_SFB = tiled_copy_s2t_SFB.get_slice(0);
        auto thr_sSFB_s2t_ = thr_copy_s2t_SFB.partition_S(smem_SFB_compact);
        auto thr_sSFB_s2t =
            cute::get_utccp_smem_desc_tensor<UtccpOp>(thr_sSFB_s2t_);
        auto thr_tSFB_s2t = thr_copy_s2t_SFB.partition_D(tmem_SFB_compact);

        MainloopPipelineState mainloop_pipe_consumer_state;
        AccumulatorPipelineState accumulator_pipe_producer_state =
            cutlass::make_producer_start_state<AccumulatorPipeline>();

        // KStreamK=false only: the next tile's accumulator stage and first
        // k-stage are waited on right after the current tile's MMAs are
        // issued, before the TMEM overlap barrier, hiding the producer's
        // stage latency. With stream-K the K ranges of consecutive work
        // items differ, so the first k-stage of the next item is not the
        // next pipeline stage -- the prefetch is skipped and every k-tile
        // waits explicitly.
        [[maybe_unused]] bool next_tile_ready = false;

        while (scheduler.is_valid()) {
            if constexpr (KStreamK) {
                accumulator_pipeline.producer_acquire(
                    accumulator_pipe_producer_state);
            } else {
                if (!next_tile_ready) {
                    accumulator_pipeline.producer_acquire(
                        accumulator_pipe_producer_state);
                }
            }
            epilogue_stage_idx = accumulator_pipe_producer_state.index();

            int n_idx = scheduler.n_idx();
            [[maybe_unused]] uint32_t sfb_mma_col = 0;
            if constexpr (IsSfbUnaligned) {
                // The window lands from column 0 and the MMA starts at the
                // tile's own column (already moved to the even one by 224).
                const uint32_t sfb_row_col =
                    (n_idx * cute::size<1>(TileShape{}) % 128) / 32;
                sfb_mma_col = sfb_row_col & ~1u;
            }
            auto thr_tSFB_s2t_tile = thr_tSFB_s2t;
            auto tmem_SFB_mma = [&]() {
                // A 64-column SfAtom tile is half of a 128-column atom, so the
                // TMEM column depends on which half it is.
                if constexpr (IsCtaN64) {
                    auto tmem_SFB_tmp = tmem_SFB;
                    tmem_SFB_tmp.data() =
                        tmem_SFB_tmp.data().get() + (n_idx % VSPerCtaN) * 2;
                    return tmem_SFB_tmp;
                } else if constexpr (IsSfbUnaligned) {
                    auto tmem_SFB_tmp = tmem_SFB;
                    tmem_SFB_tmp.data() =
                        tmem_SFB_tmp.data().get() + sfb_mma_col;
                    return tmem_SFB_tmp;
                } else {
                    return tmem_SFB;
                }
            }();

            tiled_mma.accumulate_ = cute::UMMA::ScaleOut::Zero;

            // Stream-K: same K range the producer used (see tma_loadAB).
            const uint32_t k_lo = scheduler.k_begin();
            const uint32_t k_hi = scheduler.k_end();
            for (uint32_t k_tile_id = k_lo; k_tile_id < k_hi; ++k_tile_id) {
                prologue_stage_idx = mainloop_pipe_consumer_state.index();
                // Data-parallel: the first k-stage of a prefetched tile was
                // already waited on below (after the previous tile's MMAs).
                if constexpr (KStreamK) {
                    mainloop_pipeline.consumer_wait(
                        mainloop_pipe_consumer_state);
                } else {
                    if (k_tile_id != k_lo || !next_tile_ready) {
                        mainloop_pipeline.consumer_wait(
                            mainloop_pipe_consumer_state);
                    }
                }
                // UTCCP copy of scale factors into TMEM (block-scaled only).
                if constexpr (!KIsBF16) {
                    if (select_thr) {
                        copy(tiled_copy_s2t_SFA,
                             thr_sSFA_s2t(cute::_, cute::_, cute::_, cute::_,
                                          prologue_stage_idx),
                             thr_tSFA_s2t(cute::_, cute::_, cute::_, cute::_));
                        copy(tiled_copy_s2t_SFB,
                             thr_sSFB_s2t(cute::_, cute::_, cute::_, cute::_,
                                          prologue_stage_idx),
                             thr_tSFB_s2t_tile(cute::_, cute::_, cute::_,
                                               cute::_));
                    }
                }

                CUTLASS_PRAGMA_UNROLL
                for (uint32_t k_block_id = 0; k_block_id < iter_k;
                     ++k_block_id) {
                    if constexpr (KIsBF16) {
                        // kind::f16 MMA: no scale factors; accumulate flag
                        // comes from tiled_mma.accumulate_.
                        gemm(tiled_mma,
                             tile_rA(cute::_, cute::_, k_block_id,
                                     prologue_stage_idx),
                             tile_rB(cute::_, cute::_, k_block_id,
                                     prologue_stage_idx),
                             tmem_accumulators(cute::_, cute::_, cute::_,
                                               epilogue_stage_idx));
                    } else {
                        gemm(tiled_mma.with(
                                 tiled_mma.accumulate_,
                                 tmem_SFA(cute::_, cute::_, k_block_id),
                                 tmem_SFB_mma(cute::_, cute::_, k_block_id)),
                             tile_rA(cute::_, cute::_, k_block_id,
                                     prologue_stage_idx),
                             tile_rB(cute::_, cute::_, k_block_id,
                                     prologue_stage_idx),
                             tmem_accumulators(cute::_, cute::_, cute::_,
                                               epilogue_stage_idx));
                    }
                    tiled_mma.accumulate_ = cute::UMMA::ScaleOut::One;
                }

                mainloop_pipeline.consumer_release(
                    mainloop_pipe_consumer_state);

                ++mainloop_pipe_consumer_state;
            }

            accumulator_pipeline.producer_commit(
                accumulator_pipe_producer_state);

            ++accumulator_pipe_producer_state;
            scheduler.advance();

            if constexpr (!KStreamK) {
                // Prefetch: acquire the next tile's accumulator stage and
                // wait on its first mainloop stage now, so the next
                // iteration's MMA issue path starts with data ready.
                next_tile_ready = scheduler.is_valid();
                if (next_tile_ready) {
                    accumulator_pipeline.producer_acquire(
                        accumulator_pipe_producer_state);
                    mainloop_pipeline.consumer_wait(
                        mainloop_pipe_consumer_state);
                }
            }

            // TMEM overlap sync: wait for the overlap warp-group to finish
            // reading the shared TMEM columns before MMA zeroes the next stage.
            //
            // CTA-local part: 1 wg (128 thr) + MMA warp (32 thr) = 160 threads.
            // The overlap wg does bar.arrive (non-blocking) in the epilogue;
            // MMA does bar.sync here (blocking, released once the wg arrives).
            //
            // Cluster part (pair-UMMA only): additionally wait on the cluster-
            // scope mbarrier so the non-leader CTA's overlap wg is also done.
            if constexpr (KernelTraits::KTmemOverlapCols > 0) {
                constexpr int overlap_bar_threads = 4 * 32 + 32;
                cutlass::arch::NamedBarrier overlap_barrier(
                    overlap_bar_threads, 0 /*user barrier id*/);
                overlap_barrier.arrive_and_wait();

                if constexpr (KernelTraits::KNumCTA > 1) {
                    // Wait for all CTAs' overlap wgs to arrive.
                    // Phase parity alternates each tile iteration.
                    cutlass::arch::ClusterBarrier::wait(
                        &shared_storage.overlap_cluster_bar, overlap_phase);
                    overlap_phase ^= 1;
                }
            }
        }

        accumulator_pipeline.producer_tail(accumulator_pipe_producer_state);
    }
};

}  // namespace halops
