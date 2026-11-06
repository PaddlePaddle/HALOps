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

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/atom/copy_traits_sm100_tma.hpp>
#include <cute/tensor.hpp>
#include <cutlass/detail/sm100_blockscaled_layout.hpp>
#include <cutlass/detail/sm100_tmem_helper.hpp>

#include "indexer_config.cuh"

#ifndef HALOPS_INDEXER_DEVICE_ASSERT
#    define HALOPS_INDEXER_DEVICE_ASSERT(cond) \
        do {                                   \
            if (not(cond)) asm("trap;");       \
        } while (0)
#endif

namespace halops::indexer {
using transac_bar_t = cutlass::arch::ClusterTransactionBarrier;
}

namespace halops::indexer::intrinsics {

CUTE_DEVICE
void gmem_red_add1_release(int* p) {
    asm volatile("red.release.gpu.global.add.s32 [%0], 1;" ::"l"(p) : "memory");
}

CUTE_DEVICE
int gmem_ld_acquire(const int* p) {
    int v;
    asm volatile("ld.acquire.gpu.global.s32 %0, [%1];"
                 : "=r"(v)
                 : "l"(p)
                 : "memory");
    return v;
}

CUTE_DEVICE
int gmem_ld_relaxed(const int* p) {
    int v;
    asm volatile("ld.relaxed.gpu.global.s32 %0, [%1];"
                 : "=r"(v)
                 : "l"(p)
                 : "memory");
    return v;
}

CUTE_DEVICE
int gmem_atom_add1_acq_rel(int* p) {
    int prev;
    asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], 1;"
                 : "=r"(prev)
                 : "l"(p)
                 : "memory");
    return prev;
}

CUTE_DEVICE
void gmem_st_zero_relaxed(int* p) {
    asm volatile("st.global.s32 [%0], 0;" ::"l"(p) : "memory");
}

CUTE_DEVICE
void umma_arrive_noelect(transac_bar_t& bar) {
    uint32_t bar_intptr = cute::cast_smem_ptr_to_uint(&bar);
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 "
        "[%0];\n"
        :
        : "r"(bar_intptr));
}

CUTE_DEVICE void tcgen05_after_thread_sync() {
    asm volatile("tcgen05.fence::after_thread_sync;");
}

template <typename TiledMMA, typename TensorA, typename TensorB,
          typename TensorFragC>
CUTE_DEVICE void utcmma_ss(TiledMMA& tiled_mma, TensorA sA, TensorB sB,
                           TensorFragC tC_frag, bool clear_accum) {
    using namespace cute;
    tiled_mma.accumulate_ =
        clear_accum ? UMMA::ScaleOut::Zero : UMMA::ScaleOut::One;

    ThrMMA thr_mma = tiled_mma.get_slice(_0{});
    auto sA_frag = thr_mma.partition_fragment_A(sA);
    auto sB_frag = thr_mma.partition_fragment_B(sB);
    static_assert(size<2>(sA_frag) == size<2>(sB_frag));
    static_assert(size<1>(sA_frag) == size<1>(tC_frag));
    static_assert(size<1>(sB_frag) == size<2>(tC_frag));
    CUTE_UNROLL
    for (int k = 0; k < size<2>(sA_frag); ++k) {
        cute::gemm(tiled_mma, sA_frag(_, _, k), sB_frag(_, _, k), tC_frag);
        tiled_mma.accumulate_ = UMMA::ScaleOut::One;
    }
}

template <int MN, int K, typename T>
static constexpr auto make_umma_k_major_layout() {
    using namespace cute;
    return coalesce(tile_to_shape(UMMA::Layout_K_SW128_Atom<T>{},
                                  Shape<Int<MN>, Int<K>>{}, Step<_1, _2>{}),
                    Shape<_1, _1>{});
}

}  // namespace halops::indexer::intrinsics

namespace cute {

template <class a_type, class b_type, class c_type, int M, int N,
          UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg = UMMA::ScaleIn::One,
          UMMA::ScaleIn b_neg = UMMA::ScaleIn::One>
struct SM100_MMA_F16BF16_SS_NOELECT
    : SM100_MMA_F16BF16_SS<a_type, b_type, c_type, M, N, a_major, b_major,
                           a_neg, b_neg> {
    CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                     uint64_t const& desc_b,
                                     uint32_t const& tmem_c,
                                     uint32_t const& scaleC,
                                     uint64_t const& idescE) {
        uint32_t mask[4] = {0, 0, 0, 0};
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %4, 0;\n\t"
            "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, {%5, %6, %7, "
            "%8}, p; \n\t"
            "}\n"
            :
            : "r"(tmem_c), "l"(desc_a), "l"(desc_b),
              "r"(uint32_t(idescE >> 32)), "r"(scaleC), "r"(mask[0]),
              "r"(mask[1]), "r"(mask[2]), "r"(mask[3]));
    }
};

template <class a_type, class b_type, class c_type, int M, int N,
          UMMA::Major a_major, UMMA::Major b_major, UMMA::ScaleIn a_neg,
          UMMA::ScaleIn b_neg>
struct MMA_Traits<SM100_MMA_F16BF16_SS_NOELECT<a_type, b_type, c_type, M, N,
                                               a_major, b_major, a_neg, b_neg>>
    : MMA_Traits<SM100_MMA_F16BF16_SS<a_type, b_type, c_type, M, N, a_major,
                                      b_major, a_neg, b_neg>> {
    template <class TD, class DLayout, class TA, class ALayout_, class TB,
              class BLayout_, class TC, class CLayout_>
    CUTE_HOST_DEVICE constexpr friend void mma_unpack(
        MMA_Traits const& traits, Tensor<TD, DLayout>& D,
        Tensor<TA, ALayout_> const& A, Tensor<TB, BLayout_> const& B,
        Tensor<TC, CLayout_> const& C) {
        static_assert(is_tmem<TD>::value, "Expected tmem in MMA_Atom::call");
        static_assert(is_rmem<TA>::value,
                      "Expected desc registers in MMA_Atom::call");
        static_assert(is_rmem<TB>::value,
                      "Expected desc registers in MMA_Atom::call");
        static_assert(is_tmem<TC>::value, "Expected tmem in MMA_Atom::call");

        uint64_t desc_a = A[0];
        uint64_t desc_b = B[0];
        uint32_t tmem_c = raw_pointer_cast(D.data());
        uint64_t idesc = UMMA::make_runtime_instr_desc<>(traits.idesc_);
        SM100_MMA_F16BF16_SS_NOELECT<a_type, b_type, c_type, M, N, a_major,
                                     b_major, a_neg,
                                     b_neg>::fma(desc_a, desc_b, tmem_c,
                                                 uint32_t(traits.accumulate_),
                                                 idesc);
    }
};

}  // namespace cute

namespace halops::indexer {

template <class Bar>
CUTE_DEVICE void sync_named_barrier() {
    cutlass::arch::NamedBarrier::arrive_and_wait(Bar::width, Bar::id);
}

struct GemmPipelineBarriers {
    transac_bar_t* tma_a;
    transac_bar_t* tma_b;
    transac_bar_t* buffer_free;
    transac_bar_t* accumulator_ready;
    transac_bar_t* accumulator_free;
};

template <int STAGES>
CUTE_DEVICE int stage_use_count(int tiles_per_split, int stage) {
    return tiles_per_split / STAGES +
           (stage < tiles_per_split % STAGES ? 1 : 0);
}

template <int STAGES>
CUTE_DEVICE void wait_stage_free(const GemmPipelineBarriers& barriers,
                                 uint32_t previous_stage_parities,
                                 uint32_t unused_stage_mask, int item_sequence,
                                 int tiles_per_split, int tile) {
    const int stage = tile % STAGES;
    const bool stage_untouched = ((unused_stage_mask >> stage) & 1u) != 0u;
    const bool first_ever =
        stage_untouched && item_sequence == 0 && tile < STAGES;
    if (first_ever) return;
    const int need =
        (int)((previous_stage_parities >> stage) & 1u) +
        item_sequence * stage_use_count<STAGES>(tiles_per_split, stage) +
        tile / STAGES;
    __syncwarp();
    barriers.buffer_free[stage].wait((need - 1) & 1);
}

template <int STAGES>
CUTE_DEVICE void wait_stage_tma(const GemmPipelineBarriers& barriers,
                                uint32_t previous_stage_parities,
                                int item_sequence, int tiles_per_split,
                                int tile) {
    const int stage = tile % STAGES;
    const int need =
        (int)((previous_stage_parities >> stage) & 1u) +
        item_sequence * stage_use_count<STAGES>(tiles_per_split, stage) +
        tile / STAGES + 1;
    const uint32_t phase = (uint32_t)((need - 1) & 1);
    __syncwarp();
    barriers.tma_a[stage].wait(phase);
    barriers.tma_b[stage].wait(phase);
}

template <int ACC_STAGES>
CUTE_DEVICE int reserve_accumulator_slot(const GemmPipelineBarriers& barriers,
                                         int previous_accumulators,
                                         int accumulator_sequence) {
    const int total = previous_accumulators + accumulator_sequence;
    const int slot = total % ACC_STAGES;
    if (total >= ACC_STAGES) {
        const int lap = total / ACC_STAGES;
        __syncwarp();
        barriers.accumulator_free[slot].wait((lap - 1) & 1);
    }
    return slot;
}

template <int ACC_STAGES>
CUTE_DEVICE int wait_accumulator_slot(const GemmPipelineBarriers& barriers,
                                      int previous_accumulators,
                                      int accumulator_sequence) {
    const int total = previous_accumulators + accumulator_sequence;
    const int slot = total % ACC_STAGES;
    const int lap = total / ACC_STAGES;
    barriers.accumulator_ready[slot].wait(lap & 1);
    return slot;
}

template <int STAGES, class F>
CUTE_DEVICE void dispatch_stage(int stage, F&& visit_stage, int tile) {
    static_assert(STAGES == 1 || STAGES == 2);
    if (stage == 0)
        visit_stage(cute::Int<0>{}, tile);
    else if constexpr (STAGES == 2)
        visit_stage(cute::Int<1>{}, tile);
}

struct PartialBarrier {
    int* slot;
    int producer_count;

    CUTE_DEVICE PartialBarrier(int* base, int index, int producer_count)
        : slot(base + (int64_t)index * (IndexerWarpTraits::RendezvousSlotBytes /
                                        sizeof(int))),
          producer_count(producer_count) {}

    CUTE_DEVICE void arrive_warp() const {
        __syncwarp();
        if (cute::elect_one_sync())
            halops::indexer::intrinsics::gmem_red_add1_release(slot);
    }

    CUTE_DEVICE void wait(uint32_t epoch) const {
        const int target = (int)(epoch + 1) * producer_count;
        int seen;
        do {
            seen = halops::indexer::intrinsics::gmem_ld_relaxed(slot);
        } while (seen < target);
        seen = halops::indexer::intrinsics::gmem_ld_acquire(slot);
    }

    CUTE_DEVICE void wait_warp(uint32_t epoch) const {
        if (cute::elect_one_sync()) wait(epoch);
        __syncwarp();
    }

    CUTE_DEVICE bool try_wait(uint32_t epoch) const {
        const int target = (int)(epoch + 1) * producer_count;
        // Poll relaxed; acquire published data only after the producers finish.
        if (halops::indexer::intrinsics::gmem_ld_relaxed(slot) < target)
            return false;
        return halops::indexer::intrinsics::gmem_ld_acquire(slot) >= target;
    }

    CUTE_DEVICE bool try_wait_warp(uint32_t epoch) const {
        int ready_vote = 0;
        if (cute::elect_one_sync()) ready_vote = try_wait(epoch) ? 1 : 0;
        const bool ready = __any_sync(0xffffffffu, ready_vote) != 0;
        if (ready) __syncwarp();
        return ready;
    }

    CUTE_DEVICE void reset_relaxed() const {
        halops::indexer::intrinsics::gmem_st_zero_relaxed(slot);
    }
};

// BF16 maxima permit exact exponent selection for divisors 6 and 448.
template <int SHIFT, uint32_t MANTISSA>
CUTE_DEVICE uint8_t quant_scale_exponent(float amax) {
    const uint32_t bits = __float_as_uint(fmaxf(amax, 1e-10f));
    if (bits >= 0x7f800000u) return 254;
    return static_cast<uint8_t>(max(
        0, min(254,
               int(bits >> 23) - SHIFT + int((bits & 0x7fffffu) > MANTISSA))));
}

CUTE_DEVICE float quant_scale_inverse(uint8_t exponent) {
    return __uint_as_float(exponent == 254 ? 0x00400000u
                                           : uint32_t(254 - exponent) << 23);
}

template <int N_PAD>
CUTE_HOST_DEVICE constexpr int sf_row_of(int t) {
    static_assert(N_PAD >= IndexerWarpTraits::MinTokenTile &&
                      N_PAD <= GemmShapeDefaults::ScaleRowsPerTile &&
                      (N_PAD & (N_PAD - 1)) == 0,
                  "token tile must be a power of two within the SfAtom tile");
    if constexpr (N_PAD < GemmShapeDefaults::MinPackedScaleTile)
        return (t / N_PAD) * GemmShapeDefaults::ScaleRowsPerTile + (t % N_PAD);
    else
        return t;
}

template <int TokenTile_, int Stages = GemmShapeDefaults::TmaStages>
struct GemmInstance {
    using Config = IndexerGemmConfig<TokenTile_>;
    static constexpr int TokenTile = TokenTile_;
    static_assert(Stages >= 1 && Stages <= Config::TmaStages);

    using TileShape = cute::Shape<cute::Int<Config::BM>, cute::Int<TokenTile>,
                                  cute::Int<Config::BK>>;
    using TileShape_SF =
        cute::Shape<cute::Int<Config::BM>, cute::Int<Config::ScaleRowsPerTile>,
                    cute::Int<Config::BK>>;
    using EpiComputeTile =
        cute::Shape<cute::Int<Config::BM>, cute::Int<TokenTile>>;
    using TiledMma = decltype(cute::make_tiled_mma(
        cute::SM100_MMA_MXF8F6F4_SS<e4m3, e4m3, float, ue8m0, Config::BM,
                                    TokenTile, cute::UMMA::Major::K,
                                    cute::UMMA::Major::K>{}));
    using TiledMma_SF = cute::TiledMMA<
        cute::MMA_Atom<typename TiledMma::MMA_ScaleFactor>,
        cute::Layout<cute::Shape<cute::_1, cute::_1, cute::_1>>,
        cute::Tile<cute::Underscore, cute::Underscore, cute::Underscore>>;
    using ScaleConfig =
        cutlass::detail::Sm1xxBlockScaledConfig<Config::ScaleGroupSize>;
    using ScaleLayout = typename ScaleConfig::LayoutSF;
    using CopyOpT2R = cute::conditional_t<
        TokenTile == 128, cute::SM100_TMEM_LOAD_32dp32b128x,
        cute::conditional_t<
            TokenTile == 64, cute::SM100_TMEM_LOAD_32dp32b64x,
            cute::conditional_t<TokenTile == 32,
                                cute::SM100_TMEM_LOAD_32dp32b32x,
                                cute::SM100_TMEM_LOAD_32dp32b16x>>>;

    using MmaShapeA = decltype(cute::partition_shape_A(
        TiledMma{}, cute::select<0, 2>(TileShape{})));
    using MmaShapeB = decltype(cute::partition_shape_B(
        TiledMma{}, cute::select<1, 2>(TileShape{})));
    using SmemLayoutAStaged = decltype(cute::UMMA::tile_to_mma_shape(
        cute::UMMA::Layout_K_SW128_Atom<e4m3>{},
        cute::append(MmaShapeA{}, cute::Int<Stages>{}),
        cute::Step<cute::_1, cute::_2, cute::_3>{}));
    using SmemLayoutBStaged = decltype(cute::UMMA::tile_to_mma_shape(
        cute::UMMA::Layout_K_SW128_Atom<e4m3>{},
        cute::append(MmaShapeB{}, cute::Int<Stages>{}),
        cute::Step<cute::_1, cute::_2, cute::_3>{}));
    using SmemLayoutAtomSFA =
        decltype(ScaleConfig::deduce_smem_layoutSFA(TiledMma{}, TileShape{}));
    using SmemLayoutAtomSFB = decltype(ScaleConfig::deduce_smem_layoutSFB(
        TiledMma{}, TileShape_SF{}));

    template <class Layout>
    using StagedScaleLayout = decltype(cute::make_layout(
        cute::append(cute::shape(Layout{}), cute::Int<Stages>{}),
        cute::append(cute::stride(Layout{}),
                     cute::size(cute::filter_zeros(Layout{})))));
    using SmemLayoutSFAStaged = StagedScaleLayout<SmemLayoutAtomSFA>;
    using SmemLayoutSFBStaged = StagedScaleLayout<SmemLayoutAtomSFB>;

    static constexpr bool SEPARATE_SCALE_TILE_PER_GROUP =
        TokenTile < Config::MinPackedScaleTile;
    static constexpr int TOKEN_TILES_PER_SCALE_TILE =
        Config::ScaleRowsPerTile / TokenTile;
    static constexpr int SCALE_B_COLUMNS_PER_TOKEN_TILE =
        TokenTile / Config::ScaleRowsPerColumn;
    static_assert(SEPARATE_SCALE_TILE_PER_GROUP ||
                      SCALE_B_COLUMNS_PER_TOKEN_TILE % 2 == 0,
                  "Packed scale slots must start at even TMEM columns");

    static constexpr int OPERAND_A_STAGE_BYTES =
        cute::cosize_v<SmemLayoutAStaged> * sizeof(e4m3) / Stages;
    static constexpr int OPERAND_B_STAGE_BYTES =
        cute::cosize_v<SmemLayoutBStaged> * sizeof(e4m3) / Stages;
    static constexpr int SCALE_A_STAGE_BYTES =
        cute::cosize_v<SmemLayoutSFAStaged> * sizeof(ue8m0) / Stages;
    static constexpr int SCALE_B_STAGE_BYTES =
        cute::cosize_v<SmemLayoutSFBStaged> * sizeof(ue8m0) / Stages;
    static constexpr int OPERAND_A_OFFSET = 0;
    static constexpr int OPERAND_B_OFFSET =
        Config::TmaStages * OPERAND_A_STAGE_BYTES;
    static constexpr int SCALE_A_OFFSET =
        OPERAND_B_OFFSET + Config::TmaStages * OPERAND_B_STAGE_BYTES;
    static constexpr int SCALE_B_OFFSET =
        SCALE_A_OFFSET + Config::TmaStages * SCALE_A_STAGE_BYTES;
    static constexpr int SHARED_MEMORY_BYTES =
        SCALE_B_OFFSET + Config::TmaStages * SCALE_B_STAGE_BYTES;
    static constexpr uint32_t TMA_BYTES_A =
        OPERAND_A_STAGE_BYTES + SCALE_A_STAGE_BYTES;
    static constexpr uint32_t TMA_BYTES_B =
        OPERAND_B_STAGE_BYTES + SCALE_B_STAGE_BYTES;

    // The indexer uses one CTA and K-major MXFP8 operands throughout.
    using ClusterLayout = decltype(cute::tiled_divide(
        cute::make_layout(cute::Shape<cute::_1, cute::_1, cute::_1>{}),
        cute::make_tile(typename TiledMma::AtomThrID{})));

    CUTE_HOST static auto make_tma_a(const void* data, int rows, int k) {
        auto tensor = cute::make_tensor(
            cute::make_gmem_ptr(static_cast<const e4m3*>(data)),
            cute::make_layout(cute::make_shape(rows, k, 1),
                              cute::make_stride(k, cute::_1{}, 0)));
        return cute::make_tma_atom_A_sm100(
            cute::SM90_TMA_LOAD{}, tensor,
            SmemLayoutAStaged{}(cute::_, cute::_, cute::_, cute::_0{}),
            TileShape{}, TiledMma{}, ClusterLayout{});
    }
    CUTE_HOST static auto make_tma_b(const void* data, int rows, int k) {
        auto tensor = cute::make_tensor(
            cute::make_gmem_ptr(static_cast<const e4m3*>(data)),
            cute::make_layout(cute::make_shape(rows, k, 1),
                              cute::make_stride(k, cute::_1{}, 0)));
        return cute::make_tma_atom_B_sm100(
            cute::SM90_TMA_LOAD{}, tensor,
            SmemLayoutBStaged{}(cute::_, cute::_, cute::_, cute::_0{}),
            TileShape{}, TiledMma{}, ClusterLayout{});
    }
    CUTE_HOST static auto make_tma_sfa(const void* data,
                                       const ScaleLayout& layout) {
        return cute::make_tma_atom_A_sm100<uint16_t>(
            cute::SM90_TMA_LOAD{},
            cute::make_tensor(static_cast<const ue8m0*>(data), layout),
            SmemLayoutSFAStaged{}(cute::_, cute::_, cute::_, cute::_0{}),
            TileShape{}, TiledMma{}, ClusterLayout{});
    }
    CUTE_HOST static auto make_tma_sfb(const void* data,
                                       const ScaleLayout& layout) {
        return cute::make_tma_atom_B_sm100<uint16_t>(
            cute::SM90_TMA_LOAD{},
            cute::make_tensor(static_cast<const ue8m0*>(data), layout),
            SmemLayoutSFBStaged{}(cute::_, cute::_, cute::_, cute::_0{}),
            TileShape_SF{}, TiledMma_SF{}, ClusterLayout{});
    }
};

template <class GemmConfig>
using GemmInstanceFor = GemmInstance<GemmConfig::BN, GemmConfig::TmaStages>;

template <int TokenTile>
CUTE_HOST_DEVICE constexpr int pipeline_smem_bytes() {
    return GemmInstance<TokenTile>::SHARED_MEMORY_BYTES;
}

template <int TokenTile>
struct GemmOperandADescriptors {
    using Instance = GemmInstance<TokenTile>;
    decltype(Instance::make_tma_a(nullptr, 0, 0)) tma_a;
    decltype(Instance::make_tma_sfa(nullptr,
                                    typename Instance::ScaleLayout{})) tma_sfa;
    typename Instance::ScaleLayout layout_sfa;
    int m_total;
    int k_dim;
};

template <int TokenTile>
CUTE_HOST GemmOperandADescriptors<TokenTile> make_atoms_a(const void* data,
                                                          const void* scales,
                                                          int rows, int k,
                                                          int valid_rows = 0) {
    using Instance = GemmInstance<TokenTile>;
    auto scale_layout = Instance::ScaleConfig::tile_atom_to_shape_SFA(
        cute::make_shape(rows, TokenTile, k, 1));
    return {Instance::make_tma_a(data, valid_rows > 0 ? valid_rows : rows, k),
            Instance::make_tma_sfa(scales, scale_layout), scale_layout, rows,
            k};
}

template <int TokenTile>
struct GemmOperandBDescriptors {
    using Instance = GemmInstance<TokenTile>;
    decltype(Instance::make_tma_b(nullptr, 0, 0)) tma_b;
    decltype(Instance::make_tma_sfb(nullptr,
                                    typename Instance::ScaleLayout{})) tma_sfb;
    typename Instance::ScaleLayout layout_sfb;
    int n_pad;
    int k_dim;
};

template <int TokenTile>
CUTE_HOST GemmOperandBDescriptors<TokenTile> make_atoms_b(const void* data,
                                                          const void* scales,
                                                          int rows, int k,
                                                          int scale_rows) {
    using Instance = GemmInstance<TokenTile>;
    auto scale_layout = Instance::ScaleConfig::tile_atom_to_shape_SFB(
        cute::make_shape(Instance::Config::BM, scale_rows, k, 1));
    return {Instance::make_tma_b(data, rows, k),
            Instance::make_tma_sfb(scales, scale_layout), scale_layout, rows,
            k};
}

template <int BN>
CUTE_HOST_DEVICE constexpr bool bn_swap_ab() {
    return IndexerGemmConfig<BN>::SwapAB;
}

template <int BN>
using WeightDescriptors =
    cute::conditional_t<bn_swap_ab<BN>(), GemmOperandADescriptors<BN>,
                        GemmOperandBDescriptors<BN>>;
template <int BN>
using ActivationDescriptors =
    cute::conditional_t<bn_swap_ab<BN>(), GemmOperandBDescriptors<BN>,
                        GemmOperandADescriptors<BN>>;

template <int BN>
CUTE_HOST WeightDescriptors<BN> make_weight_descriptors(const void* w,
                                                        const void* w_sf,
                                                        int m_total, int k) {
    if constexpr (bn_swap_ab<BN>())
        return make_atoms_a<BN>(w, w_sf, m_total, k);
    else
        return make_atoms_b<BN>(w, w_sf, m_total, k, m_total);
}

template <int BN>
CUTE_HOST ActivationDescriptors<BN> make_activation_descriptors(
    const void* a, const void* a_sf, int padded_tokens, int k, int t_pad_sf,
    int rows_valid = 0) {
    if constexpr (bn_swap_ab<BN>())
        return make_atoms_b<BN>(
            a, a_sf, rows_valid > 0 ? rows_valid : padded_tokens, k, t_pad_sf);
    else
        return make_atoms_a<BN>(a, a_sf, padded_tokens, k, rows_valid);
}

template <int BN>
CUTE_HOST_DEVICE auto const& sf_layout(const GemmOperandADescriptors<BN>& at) {
    return at.layout_sfa;
}
template <int BN>
CUTE_HOST_DEVICE auto const& sf_layout(const GemmOperandBDescriptors<BN>& at) {
    return at.layout_sfb;
}

template <int N_PAD, class LayoutSFB>
CUTE_DEVICE void quantize_group32_from_regs(const __nv_bfloat16 (&v)[4],
                                            e4m3* __restrict__ ws_fp8,
                                            ue8m0* __restrict__ ws_sf,
                                            LayoutSFB const& layout_sfb,
                                            int n_row, int k_dim, int group,
                                            int lane) {
    const auto* v2 = reinterpret_cast<const __nv_bfloat162*>(v);
    const __nv_bfloat162 m2 = __hmax2(__habs2(v2[0]), __habs2(v2[1]));
    float amax = fmaxf(fmaxf(__bfloat162float(__low2bfloat16(m2)),
                             __bfloat162float(__high2bfloat16(m2))),
                       1e-10f);
    const unsigned mask = 0xffu << (lane & ~7);
    CUTLASS_PRAGMA_UNROLL
    for (int o = 1; o < 8; o <<= 1)
        amax = fmaxf(amax, __shfl_xor_sync(mask, amax, o));

    const uint8_t biased_exp = quant_scale_exponent<8, 0x600000u>(amax);
    const __nv_bfloat162 inv2 =
        __float2bfloat162_rn(quant_scale_inverse(biased_exp));

    const int base = group * GemmShapeDefaults::ScaleGroupSize;
    const int j = lane & 7;
    const __nv_bfloat162 p0 = __hmul2(v2[0], inv2);
    const __nv_bfloat162 p1 = __hmul2(v2[1], inv2);
    const uint32_t packed =
        (uint32_t)__nv_cvt_bfloat16raw2_to_fp8x2(
            *reinterpret_cast<const __nv_bfloat162_raw*>(&p0), __NV_SATFINITE,
            __NV_E4M3) |
        ((uint32_t)__nv_cvt_bfloat16raw2_to_fp8x2(
             *reinterpret_cast<const __nv_bfloat162_raw*>(&p1), __NV_SATFINITE,
             __NV_E4M3)
         << 16);
    *reinterpret_cast<uint32_t*>(ws_fp8 + (int64_t)n_row * k_dim + base +
                                 j * 4) = packed;

    if (j == 0) {
        cute::Tensor mSFB = cute::make_tensor(
            cute::make_gmem_ptr(reinterpret_cast<uint8_t*>(ws_sf)), layout_sfb);
        mSFB(sf_row_of<N_PAD>(n_row), base, cute::_0{}) = biased_exp;
    }
}

template <class Accumulators, class ScaleA, class ScaleB>
struct GemmTensorMemory {
    Accumulators tmem_accumulators;
    ScaleA tmem_SFA;
    ScaleB tmem_SFB;
};

template <int TokenTile, int Stages = GemmShapeDefaults::TmaStages,
          int AccumulatorStages = GemmShapeDefaults::AccStages>
CUTE_DEVICE auto make_gemm_tmem(uint32_t tmem_base) {
    using Instance = GemmInstance<TokenTile, Stages>;
    using TiledMma = typename Instance::TiledMma;
    const auto shape = cute::partition_shape_C(
        TiledMma{}, typename Instance::EpiComputeTile{});
    auto accumulators = TiledMma::make_fragment_C(
        cute::append(shape, cute::Int<AccumulatorStages>{}));
    auto scale_a = cute::make_tensor<typename TiledMma::FrgTypeSFA>(
        cute::shape(typename Instance::SmemLayoutAtomSFA{}));
    auto scale_b = cute::make_tensor<typename TiledMma::FrgTypeSFB>(
        cute::shape(typename Instance::SmemLayoutAtomSFB{}));
    constexpr int accumulator_columns =
        cutlass::detail::find_tmem_tensor_col_offset(decltype(accumulators){});
    constexpr int scale_a_columns =
        cutlass::detail::find_tmem_tensor_col_offset(decltype(scale_a){});
    constexpr int scale_b_columns =
        cutlass::detail::find_tmem_tensor_col_offset(decltype(scale_b){});
    static_assert(accumulator_columns + scale_a_columns + scale_b_columns <=
                  IndexerWarpTraits::TensorMemoryColumns);
    accumulators.data() = tmem_base;
    scale_a.data() = tmem_base + accumulator_columns;
    scale_b.data() = tmem_base + accumulator_columns + scale_a_columns;
    return GemmTensorMemory{accumulators, scale_a, scale_b};
}

}  // namespace halops::indexer
