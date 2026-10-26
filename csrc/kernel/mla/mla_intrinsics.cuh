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

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/bfloat16.h>

#include <cute/config.hpp>
#include <cute/tensor.hpp>

namespace halops::mla {
CUTE_DEVICE
nv_bfloat162 fp8x2_to_bf16x2_with_scale(__nv_fp8x2_e4m3 data,
                                        nv_bfloat16 scale) {
    uint32_t bf;
    asm("cvt.rn.bf16x2.e4m3x2 %0, %1;"
        : "=r"(bf)
        : "h"(*reinterpret_cast<const unsigned short*>(&data)));
    return __hmul2(*reinterpret_cast<const nv_bfloat162*>(&bf),
                   __bfloat162bfloat162(scale));
}

template <typename TMA, typename Tensor0, typename Tensor1>
CUTE_DEVICE void launch_tma_copy(const TMA& tma_copy, Tensor0 src, Tensor1 dst,
                                 cutlass::arch::ClusterTransactionBarrier& bar,
                                 const cute::TMA::CacheHintSm90& cache_hint =
                                     cute::TMA::CacheHintSm90::EVICT_NORMAL) {
    auto thr_tma = tma_copy.get_slice(cute::_0{});
    cute::copy(
        tma_copy.with(
            reinterpret_cast<
                typename cutlass::arch::ClusterTransactionBarrier::ValueType&>(
                bar),
            0, cache_hint),
        thr_tma.partition_S(src), thr_tma.partition_D(dst));
}

CUTE_DEVICE
void tma_gather4(const void* desc_ptr,
                 cutlass::arch::ClusterTransactionBarrier& mbar_ptr,
                 void* smem_ptr, int col_idx, int4 row_idxs,
                 int64_t cache_hint) {
    uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
    uint32_t mbar_addr = cute::cast_smem_ptr_to_uint(&mbar_ptr);
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::"
        "complete_tx::bytes.cta_group::1.L2::cache_hint [%0], [%1, {%2, %3, "
        "%4, %5, %6}], [%7], %8;\n"
        :
        : "r"(smem_addr), "l"(desc_ptr), "r"(col_idx), "r"(row_idxs.x),
          "r"(row_idxs.y), "r"(row_idxs.z), "r"(row_idxs.w), "r"(mbar_addr),
          "l"(cache_hint)
        : "memory");
}

CUTE_DEVICE
void tma_load_2d(const void* desc_ptr,
                 cutlass::arch::ClusterTransactionBarrier& mbar, void* smem_ptr,
                 int crd0, int crd1, int64_t cache_hint) {
    uint32_t smem_addr = cute::cast_smem_ptr_to_uint(smem_ptr);
    uint32_t mbar_addr = cute::cast_smem_ptr_to_uint(&mbar);
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::"
        "bytes.L2::cache_hint"
        " [%0], [%1, {%3, %4}], [%2], %5;\n"
        :
        : "r"(smem_addr), "l"(desc_ptr), "r"(mbar_addr), "r"(crd0), "r"(crd1),
          "l"(cache_hint)
        : "memory");
}

CUTE_DEVICE
void gmem_red_add1_release(int* ptr) {
    asm volatile("red.release.gpu.global.add.s32 [%0], 1;" ::"l"(ptr)
                 : "memory");
}

CUTE_DEVICE
int gmem_ld_acquire(const int* ptr) {
    int value;
    asm volatile("ld.acquire.gpu.global.s32 %0, [%1];"
                 : "=r"(value)
                 : "l"(ptr)
                 : "memory");
    return value;
}

CUTE_DEVICE
int gmem_ld_relaxed(const int* ptr) {
    int value;
    asm volatile("ld.relaxed.gpu.global.s32 %0, [%1];"
                 : "=r"(value)
                 : "l"(ptr)
                 : "memory");
    return value;
}

CUTE_DEVICE
int gmem_atom_add1_acq_rel(int* ptr) {
    int prev;
    asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], 1;"
                 : "=r"(prev)
                 : "l"(ptr)
                 : "memory");
    return prev;
}

CUTE_DEVICE
void gmem_red_add1_relaxed(int* ptr) {
    asm volatile("red.relaxed.gpu.global.add.s32 [%0], 1;" ::"l"(ptr)
                 : "memory");
}

CUTE_DEVICE
void ld_global_cg_v8(const void* ptr, uint4& lo, uint4& hi) {
    asm volatile("ld.global.cg.v8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                 : "=r"(lo.x), "=r"(lo.y), "=r"(lo.z), "=r"(lo.w), "=r"(hi.x),
                   "=r"(hi.y), "=r"(hi.z), "=r"(hi.w)
                 : "l"(ptr));
}

CUTE_DEVICE
void st_global_v8(void* ptr, const uint32_t (&words)[8]) {
    asm volatile("st.global.v8.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};" ::"l"(ptr),
                 "r"(words[0]), "r"(words[1]), "r"(words[2]), "r"(words[3]),
                 "r"(words[4]), "r"(words[5]), "r"(words[6]), "r"(words[7])
                 : "memory");
}

CUTE_DEVICE
void prefetch_l2_bulk(const void* ptr, uint32_t bytes) {
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(ptr),
                 "r"(bytes)
                 : "memory");
}

CUTE_DEVICE
void fence_proxy_async_global() {
    asm volatile("fence.proxy.async.global;" ::: "memory");
}

CUTE_DEVICE
void mbar_arrive_cluster(uint32_t bar_cluster) {
    asm volatile(
        "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" ::"r"(
            bar_cluster)
        : "memory");
}

CUTE_DEVICE
void mbar_arrive_cluster_relaxed(uint32_t bar_cluster) {
    asm volatile(
        "mbarrier.arrive.relaxed.cluster.shared::cluster.b64 _, [%0];" ::"r"(
            bar_cluster)
        : "memory");
}

CUTE_DEVICE
void mbar_wait_cluster(uint32_t bar, uint32_t phase) {
    asm volatile(
        "{\n.reg .pred p;\nWAIT_CL:\n"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], "
        "%1;\n"
        "@!p bra WAIT_CL;\n}\n" ::"r"(bar),
        "r"(phase)
        : "memory");
}

CUTE_DEVICE
void bulk_copy_to_peer(uint32_t dst_cluster, uint32_t src, uint32_t bytes,
                       uint32_t bar_cluster) {
    asm volatile(
        "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::"
        "bytes [%0], [%1], %2, [%3];" ::"r"(dst_cluster),
        "r"(src), "r"(bytes), "r"(bar_cluster)
        : "memory");
}

CUTE_DEVICE
void bulk_wait_all() {
    asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
}

CUTE_DEVICE
void gmem_st_zero_relaxed(int* ptr) {
    asm volatile("st.global.s32 [%0], 0;" ::"l"(ptr) : "memory");
}

CUTE_DEVICE
void pdl_wait() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

CUTE_DEVICE
void pdl_trigger() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.launch_dependents;" :::);
#endif
}

CUTE_DEVICE
void smem_st_128b(uint32_t smem_addr, __int128_t const& data) {
    asm volatile("st.weak.shared::cta.b128 [%0], %1;\n" ::"r"(smem_addr),
                 "q"(data));
}

CUTE_DEVICE
float2 add_rn_f32x2(const float2& lhs, const float2& rhs) {
    float2 result;
    asm volatile("add.rn.f32x2 %0, %1, %2;\n"
                 : "=l"(reinterpret_cast<uint64_t&>(result))
                 : "l"(reinterpret_cast<uint64_t const&>(lhs)),
                   "l"(reinterpret_cast<uint64_t const&>(rhs)));
    return result;
}
CUTE_DEVICE
float2 mul_rn_f32x2(const float2& lhs, const float2& rhs) {
    float2 result;
    asm volatile("mul.rn.f32x2 %0, %1, %2;\n"
                 : "=l"(reinterpret_cast<uint64_t&>(result))
                 : "l"(reinterpret_cast<uint64_t const&>(lhs)),
                   "l"(reinterpret_cast<uint64_t const&>(rhs)));
    return result;
}
CUTE_DEVICE
float2 fma_rn_f32x2(const float2& lhs, const float2& rhs,
                    const float2& addend) {
    float2 result;
    asm volatile("fma.rn.f32x2 %0, %1, %2, %3;\n"
                 : "=l"(reinterpret_cast<uint64_t&>(result))
                 : "l"(reinterpret_cast<uint64_t const&>(lhs)),
                   "l"(reinterpret_cast<uint64_t const&>(rhs)),
                   "l"(reinterpret_cast<uint64_t const&>(addend)));
    return result;
}

CUTE_DEVICE
void umma_arrive_noelect(cutlass::arch::ClusterTransactionBarrier& bar) {
    uint32_t bar_intptr = cute::cast_smem_ptr_to_uint(&bar);
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 "
        "[%0];\n"
        :
        : "r"(bar_intptr)
        : "memory");
}

CUTE_DEVICE void tcgen05_before_thread_sync() {
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}
CUTE_DEVICE void tcgen05_after_thread_sync() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

template <int N>
using TmemLoad32dp32b = cute::conditional_t<
    N == 1, cute::SM100_TMEM_LOAD_32dp32b1x,
    cute::conditional_t<
        N == 2, cute::SM100_TMEM_LOAD_32dp32b2x,
        cute::conditional_t<
            N == 4, cute::SM100_TMEM_LOAD_32dp32b4x,
            cute::conditional_t<
                N == 8, cute::SM100_TMEM_LOAD_32dp32b8x,
                cute::conditional_t<
                    N == 16, cute::SM100_TMEM_LOAD_32dp32b16x,
                    cute::conditional_t<
                        N == 32, cute::SM100_TMEM_LOAD_32dp32b32x,
                        cute::conditional_t<
                            N == 64, cute::SM100_TMEM_LOAD_32dp32b64x,
                            cute::SM100_TMEM_LOAD_32dp32b128x>>>>>>>;

CUTE_DEVICE float warp_sum(float value) {
    CUTLASS_PRAGMA_UNROLL
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_xor_sync(0xffffffffu, value, offset);
    return value;
}

CUTE_DEVICE void st_global_v8_na(void* ptr, const uint32_t (&words)[8]) {
    asm volatile(
        "st.global.L1::no_allocate.v8.b32 [%0], "
        "{%1,%2,%3,%4,%5,%6,%7,%8};\n" ::"l"(ptr),
        "r"(words[0]), "r"(words[1]), "r"(words[2]), "r"(words[3]),
        "r"(words[4]), "r"(words[5]), "r"(words[6]), "r"(words[7])
        : "memory");
}

template <int kNumElements>
CUTE_DEVICE void tmem_ld_32dp32bNx(uint32_t tmem_start, void* data_) {
    uint32_t* data = (uint32_t*)data_;
    [&]<size_t... Is>(cute::index_sequence<Is...>) {
        TmemLoad32dp32b<kNumElements>::copy(tmem_start, data[Is]...);
    }(cute::make_index_sequence<kNumElements>{});
}

template <int kNumElements>
CUTE_DEVICE void tmem_st_32dp32bNx(uint32_t tmem_start, void const* data_) {
    uint32_t const* data = (uint32_t const*)data_;
    [&]<size_t... Is>(cute::index_sequence<Is...>) {
        if constexpr (kNumElements == 1)
            cute::SM100_TMEM_STORE_32dp32b1x::copy(data[Is]..., tmem_start);
        else if constexpr (kNumElements == 2)
            cute::SM100_TMEM_STORE_32dp32b2x::copy(data[Is]..., tmem_start);
        else if constexpr (kNumElements == 4)
            cute::SM100_TMEM_STORE_32dp32b4x::copy(data[Is]..., tmem_start);
        else if constexpr (kNumElements == 8)
            cute::SM100_TMEM_STORE_32dp32b8x::copy(data[Is]..., tmem_start);
        else if constexpr (kNumElements == 16)
            cute::SM100_TMEM_STORE_32dp32b16x::copy(data[Is]..., tmem_start);
        else if constexpr (kNumElements == 32)
            cute::SM100_TMEM_STORE_32dp32b32x::copy(data[Is]..., tmem_start);
        else if constexpr (kNumElements == 64)
            cute::SM100_TMEM_STORE_32dp32b64x::copy(data[Is]..., tmem_start);
        else
            cute::SM100_TMEM_STORE_32dp32b128x::copy(data[Is]..., tmem_start);
    }(cute::make_index_sequence<kNumElements>{});
}

template <typename TiledMMA, typename TensorA, typename TensorB,
          typename TensorFragC>
CUTE_DEVICE void umma_ss(TiledMMA& tiled_mma, TensorA sA, TensorB sB,
                         TensorFragC tC_frag, bool clear_accum) {
    using namespace cute;
    tiled_mma.accumulate_ =
        clear_accum ? UMMA::ScaleOut::Zero : UMMA::ScaleOut::One;

    ThrMMA thr_mma = tiled_mma.get_slice(_0{});
    auto sA_frag = thr_mma.partition_fragment_A(sA);
    auto sB_frag = thr_mma.partition_fragment_B(sB);
    CUTE_UNROLL
    for (int k = 0; k < size<2>(sA_frag); ++k) {
        cute::gemm(tiled_mma, sA_frag(_, _, k), sB_frag(_, _, k), tC_frag);
        tiled_mma.accumulate_ = UMMA::ScaleOut::One;
    }
}

template <typename TiledMMA, typename TensorA, typename TensorB,
          typename TensorFragC>
CUTE_DEVICE void umma_ts(TiledMMA& tiled_mma, TensorA tA_frag, TensorB sB,
                         TensorFragC tC_frag, bool clear_accum) {
    using namespace cute;
    tiled_mma.accumulate_ =
        clear_accum ? UMMA::ScaleOut::Zero : UMMA::ScaleOut::One;
    ThrMMA thr_mma = tiled_mma.get_slice(_0{});
    auto sB_frag = thr_mma.partition_fragment_B(sB);
    CUTE_UNROLL
    for (int k = 0; k < size<2>(tA_frag); ++k) {
        cute::gemm(tiled_mma, tA_frag(_, _, k), sB_frag(_, _, k), tC_frag);
        tiled_mma.accumulate_ = UMMA::ScaleOut::One;
    }
}

template <int MN, int K, int Swizzle, typename Elem>
constexpr auto make_umma_canonical_k_major_layout() {
    using namespace cute;
    using base_atom_type = std::conditional_t<
        Swizzle == 0 || Swizzle == 16, UMMA::Layout_K_INTER_Atom<Elem>,
        std::conditional_t<
            Swizzle == 32, UMMA::Layout_K_SW32_Atom<Elem>,
            std::conditional_t<
                Swizzle == 64, UMMA::Layout_K_SW64_Atom<Elem>,
                std::conditional_t<Swizzle == 128,
                                   UMMA::Layout_K_SW128_Atom<Elem>, void>>>>;
    return coalesce(tile_to_shape(base_atom_type{}, Shape<Int<MN>, Int<K>>{},
                                  Step<_1, _2>{}),
                    Shape<_1, _1>{});
}
template <class a_type, class b_type, class c_type, int M, int N>
struct SM100_MMA_F16BF16_NOELECT_TraitsBase {
    using ValTypeD = c_type;
    using ValTypeA = a_type;
    using ValTypeB = b_type;
    using ValTypeC = c_type;

    static constexpr int K = 256 / cute::sizeof_bits<ValTypeA>::value;

    using Shape_MNK = cute::Shape<cute::Int<M>, cute::Int<N>, cute::Int<K>>;
    using ThrID = cute::Layout<cute::_1>;
    using ALayout = cute::Layout<
        cute::Shape<cute::_1, cute::Shape<cute::Int<M>, cute::Int<K>>>,
        cute::Stride<cute::_0, cute::Stride<cute::_1, cute::Int<M>>>>;
    using BLayout = cute::Layout<
        cute::Shape<cute::_1, cute::Shape<cute::Int<N>, cute::Int<K>>>,
        cute::Stride<cute::_0, cute::Stride<cute::_1, cute::Int<N>>>>;
    using CLayout = cute::Layout<
        cute::Shape<cute::_1, cute::Shape<cute::Int<M>, cute::Int<N>>>,
        cute::Stride<cute::_0, cute::Stride<cute::_1, cute::Int<M>>>>;
};

template <class a_type, class b_type, class c_type, int M, int N,
          cute::UMMA::Major a_major, cute::UMMA::Major b_major,
          cute::UMMA::ScaleIn a_neg = cute::UMMA::ScaleIn::One,
          cute::UMMA::ScaleIn b_neg = cute::UMMA::ScaleIn::One>
struct SM100_MMA_F16BF16_WS_TS_NOELECT {
    using DRegisters = void;
    using ARegisters = uint64_t[1];
    using BRegisters = uint64_t[1];
    using CRegisters = uint32_t[1];

    CUTE_HOST_DEVICE static void fma(uint32_t const& tmem_a,
                                     uint64_t const& desc_b,
                                     uint32_t const& tmem_c,
                                     uint32_t const& scaleC,
                                     uint64_t const& idescE) {
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %4, 0;\n\t"
            "tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [%1], %2, %3, p, 0; "
            "\n\t"
            "}\n"
            :
            : "r"(tmem_c), "r"(tmem_a), "l"(desc_b),
              "r"(uint32_t(idescE >> 32)), "r"(scaleC));
    }
};

template <class a_type, class b_type, class c_type, int M, int N,
          cute::UMMA::Major a_major, cute::UMMA::Major b_major,
          cute::UMMA::ScaleIn a_neg = cute::UMMA::ScaleIn::One,
          cute::UMMA::ScaleIn b_neg = cute::UMMA::ScaleIn::One>
struct SM100_MMA_F16BF16_WS_SS_NOELECT {
    using DRegisters = void;
    using ARegisters = uint64_t[1];
    using BRegisters = uint64_t[1];
    using CRegisters = uint32_t[1];

    CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a,
                                     uint64_t const& desc_b,
                                     uint32_t const& tmem_c,
                                     uint32_t const& scaleC,
                                     uint64_t const& idescE) {
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %4, 0;\n\t"
            "tcgen05.mma.ws.cta_group::1.kind::f16 [%0], %1, %2, %3, p, 0; \n\t"
            "}\n"
            :
            : "r"(tmem_c), "l"(desc_a), "l"(desc_b),
              "r"(uint32_t(idescE >> 32)), "r"(scaleC));
    }
};

template <class a_type, class b_type, class c_type, int M, int N,
          cute::UMMA::Major a_major, cute::UMMA::Major b_major,
          cute::UMMA::ScaleIn a_neg = cute::UMMA::ScaleIn::One,
          cute::UMMA::ScaleIn b_neg = cute::UMMA::ScaleIn::One>
struct SM100_MMA_F16BF16_SS_NOELECT {
    using DRegisters = void;
    using ARegisters = uint64_t[1];
    using BRegisters = uint64_t[1];
    using CRegisters = uint32_t[1];

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

}  // namespace halops::mla

namespace cute {
template <class a_type, class b_type, class c_type, int M, int N,
          UMMA::Major a_major, UMMA::Major b_major, UMMA::ScaleIn a_neg,
          UMMA::ScaleIn b_neg>
struct MMA_Traits<halops::mla::SM100_MMA_F16BF16_WS_TS_NOELECT<
    a_type, b_type, c_type, M, N, a_major, b_major, a_neg, b_neg>>
    : halops::mla::SM100_MMA_F16BF16_NOELECT_TraitsBase<a_type, b_type, c_type,
                                                        M, N> {
    using FrgTypeA =
        UMMA::tmem_frg_1sm<a_type, a_type, UMMA::TmemAllocMode::NonInterleaved>;
    using FrgTypeB = UMMA::smem_desc<b_major>;
    using FrgTypeC = UMMA::tmem_frg_ws_1sm<c_type>;

    UMMA::ScaleOut accumulate_ = UMMA::ScaleOut::One;
    UMMA::InstrDescriptor idesc_ =
        UMMA::make_instr_desc<a_type, b_type, c_type, M, N, a_major, b_major,
                              a_neg, b_neg>();

    template <class TD, class DLayout, class TA, class ALayout_, class TB,
              class BLayout_, class TC, class CLayout_>
    CUTE_HOST_DEVICE constexpr friend void mma_unpack(
        MMA_Traits const& traits, Tensor<TD, DLayout>& tensor_d,
        Tensor<TA, ALayout_> const& tensor_a,
        Tensor<TB, BLayout_> const& tensor_b,
        Tensor<TC, CLayout_> const& tensor_c) {
        uint32_t tmem_a = raw_pointer_cast(tensor_a.data());
        uint64_t desc_b = tensor_b[0];
        uint32_t tmem_c = raw_pointer_cast(tensor_d.data());
        uint64_t idesc = UMMA::make_runtime_instr_desc<>(traits.idesc_);
        halops::mla::SM100_MMA_F16BF16_WS_TS_NOELECT<
            a_type, b_type, c_type, M, N, a_major, b_major, a_neg,
            b_neg>::fma(tmem_a, desc_b, tmem_c, uint32_t(traits.accumulate_),
                        idesc);
    }
};

template <class a_type, class b_type, class c_type, int M, int N,
          UMMA::Major a_major, UMMA::Major b_major, UMMA::ScaleIn a_neg,
          UMMA::ScaleIn b_neg>
struct MMA_Traits<halops::mla::SM100_MMA_F16BF16_WS_SS_NOELECT<
    a_type, b_type, c_type, M, N, a_major, b_major, a_neg, b_neg>>
    : halops::mla::SM100_MMA_F16BF16_NOELECT_TraitsBase<a_type, b_type, c_type,
                                                        M, N> {
    using FrgTypeA = UMMA::smem_desc<a_major>;
    using FrgTypeB = UMMA::smem_desc<b_major>;
    using FrgTypeC = UMMA::tmem_frg_ws_1sm<c_type>;

    UMMA::InstrDescriptor idesc_ =
        UMMA::make_instr_desc<a_type, b_type, c_type, M, N, a_major, b_major,
                              a_neg, b_neg>();
    UMMA::ScaleOut accumulate_ = UMMA::ScaleOut::One;

    template <class TD, class DLayout, class TA, class ALayout_, class TB,
              class BLayout_, class TC, class CLayout_>
    CUTE_HOST_DEVICE constexpr friend void mma_unpack(
        MMA_Traits const& traits, Tensor<TD, DLayout>& tensor_d,
        Tensor<TA, ALayout_> const& tensor_a,
        Tensor<TB, BLayout_> const& tensor_b,
        Tensor<TC, CLayout_> const& tensor_c) {
        uint64_t desc_a = tensor_a[0];
        uint64_t desc_b = tensor_b[0];
        uint32_t tmem_c = raw_pointer_cast(tensor_d.data());
        uint64_t idesc = UMMA::make_runtime_instr_desc<>(traits.idesc_);
        halops::mla::SM100_MMA_F16BF16_WS_SS_NOELECT<
            a_type, b_type, c_type, M, N, a_major, b_major, a_neg,
            b_neg>::fma(desc_a, desc_b, tmem_c, uint32_t(traits.accumulate_),
                        idesc);
    }
};

template <class a_type, class b_type, class c_type, int M, int N,
          UMMA::Major a_major, UMMA::Major b_major, UMMA::ScaleIn a_neg,
          UMMA::ScaleIn b_neg>
struct MMA_Traits<halops::mla::SM100_MMA_F16BF16_SS_NOELECT<
    a_type, b_type, c_type, M, N, a_major, b_major, a_neg, b_neg>>
    : halops::mla::SM100_MMA_F16BF16_NOELECT_TraitsBase<a_type, b_type, c_type,
                                                        M, N> {
    using FrgTypeA = UMMA::smem_desc<a_major>;
    using FrgTypeB = UMMA::smem_desc<b_major>;
    using FrgTypeC = UMMA::tmem_frg_1sm<c_type>;

    UMMA::InstrDescriptor idesc_ =
        UMMA::make_instr_desc<a_type, b_type, c_type, M, N, a_major, b_major,
                              a_neg, b_neg>();

    UMMA::ScaleOut accumulate_ = UMMA::ScaleOut::One;

    template <class TD, class DLayout, class TA, class ALayout_, class TB,
              class BLayout_, class TC, class CLayout_>
    CUTE_HOST_DEVICE constexpr friend void mma_unpack(
        MMA_Traits const& traits, Tensor<TD, DLayout>& tensor_d,
        Tensor<TA, ALayout_> const& tensor_a,
        Tensor<TB, BLayout_> const& tensor_b,
        Tensor<TC, CLayout_> const& tensor_c) {
        uint64_t desc_a = tensor_a[0];
        uint64_t desc_b = tensor_b[0];
        uint32_t tmem_c = raw_pointer_cast(tensor_d.data());
        uint64_t idesc = UMMA::make_runtime_instr_desc<>(traits.idesc_);
        halops::mla::SM100_MMA_F16BF16_SS_NOELECT<
            a_type, b_type, c_type, M, N, a_major, b_major, a_neg,
            b_neg>::fma(desc_a, desc_b, tmem_c, uint32_t(traits.accumulate_),
                        idesc);
    }
};
}  // namespace cute
