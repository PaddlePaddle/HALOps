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
#include <cuda_runtime_api.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/bfloat16.h>
#include <torch/extension.h>

#include <cute/config.hpp>
#include <cute/tensor.hpp>
#include <cutlass/cuda_host_adapter.hpp>
#include <string>
#include <vector>

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000 && __CUDA_ARCH__ < 1200))
#    define KERUTILS_ENABLE_SM100A
#endif

#if (defined(__CLION_IDE__) || defined(__VSCODE_IDE__))
#    define KERUTILS_ENABLE_SM100A
#endif

#define MEGA_MLA_ARCH_GATE_CHECKED 1

#ifndef KU_TRAP_ONLY_DEVICE_ASSERT
#    define KU_TRAP_ONLY_DEVICE_ASSERT(cond) \
        do {                                 \
            if (not(cond)) asm("trap;");     \
        } while (0)
#endif

using transac_bar_t = cutlass::arch::ClusterTransactionBarrier;

namespace sm100 {

CUTE_DEVICE
nv_bfloat162 fp8x2_to_bf16x2_with_scale(__nv_fp8x2_e4m3 data,
                                        nv_bfloat16 scale) {
    float2 data_float2 = (float2)data;
    nv_bfloat162 data_bf16x2 = __float22bfloat162_rn(data_float2);
    return nv_bfloat162{data_bf16x2.x * scale, data_bf16x2.y * scale};
}

}  // namespace sm100

namespace ku {

using nvbf16 = __nv_bfloat16;
using nvbf16x2 = __nv_bfloat162;
using nve4m3x2 = __nv_fp8x2_e4m3;

template <typename T>
CUTE_HOST_DEVICE constexpr T ceil(const T& a, const T& b) {
    return (a + b - 1) / b * b;
}

static inline CUtensorMap make_tensor_map(
    const std::vector<uint64_t>& size, const std::vector<uint64_t>& strides,
    const std::vector<uint32_t>& box_size, void* global_ptr,
    CUtensorMapDataType data_type, CUtensorMapSwizzle swizzle_mode,
    CUtensorMapL2promotion l2_promotion,
    CUtensorMapInterleave interleave_mode =
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
    CUtensorMapFloatOOBfill oob_fill =
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
    const std::vector<uint32_t>& element_strides_ = {}) {
    const int dim = (int)size.size();
    TORCH_CHECK(dim >= 1, "make_tensor_map: empty size");

    std::vector<uint32_t> element_strides;
    if (element_strides_.empty())
        element_strides.assign(dim, 1u);
    else
        element_strides = element_strides_;
    TORCH_CHECK(
        strides.size() == (size_t)dim - 1 && box_size.size() == (size_t)dim &&
            element_strides.size() == (size_t)dim,
        "make_tensor_map: strides must have dim-1 entries, box_size and "
        "element_strides dim entries; got ",
        strides.size(), " / ", box_size.size(), " / ", element_strides.size(),
        " for dim ", dim);

    CUtensorMap result;
    const CUresult ret_code =
        CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
            &result, data_type, dim, global_ptr, size.data(), strides.data(),
            box_size.data(), element_strides.data(), interleave_mode,
            swizzle_mode, l2_promotion, oob_fill);
    if (ret_code != CUresult::CUDA_SUCCESS) {
        auto join = [](auto const& v) {
            std::string s;
            for (auto e : v) s += std::to_string(e) + " ";
            return s;
        };
        TORCH_CHECK(false, "cuTensorMapEncodeTiled failed (", (int)ret_code,
                    "): dim=", dim, " size=[", join(size), "] strides=[",
                    join(strides), "] box=[", join(box_size),
                    "] elem_strides=[", join(element_strides),
                    "] ptr=", global_ptr, " dtype=", (int)data_type,
                    " swizzle=", (int)swizzle_mode, " l2=", (int)l2_promotion,
                    " interleave=", (int)interleave_mode,
                    " oob=", (int)oob_fill);
    }
    return result;
}

template <typename TMA, typename Tensor0, typename Tensor1>
CUTE_DEVICE void launch_tma_copy(const TMA& tma_copy, Tensor0 src, Tensor1 dst,
                                 transac_bar_t& bar,
                                 const cute::TMA::CacheHintSm90& cache_hint =
                                     cute::TMA::CacheHintSm90::EVICT_NORMAL) {
    auto thr_tma = tma_copy.get_slice(cute::_0{});
    cute::copy(
        tma_copy.with(reinterpret_cast<typename transac_bar_t::ValueType&>(bar),
                      0, cache_hint),
        thr_tma.partition_S(src), thr_tma.partition_D(dst));
}

CUTE_DEVICE
void tma_gather4(const void* desc_ptr, transac_bar_t& mbar_ptr, void* smem_ptr,
                 int col_idx, int4 row_idxs, int64_t cache_hint) {
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
void tma_load_2d(const void* desc_ptr, transac_bar_t& mbar, void* smem_ptr,
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
void gmem_st_zero_release(int* p) {
    asm volatile("st.release.gpu.global.s32 [%0], 0;" ::"l"(p) : "memory");
}

CUTE_DEVICE
void gmem_st_zero_relaxed(int* p) {
    asm volatile("st.global.s32 [%0], 0;" ::"l"(p) : "memory");
}

CUTE_DEVICE
void smem_st_128b(uint32_t smem_addr, __int128_t const& data) {
    asm volatile("st.weak.shared::cta.b128 [%0], %1;\n" ::"r"(smem_addr),
                 "q"(data));
}

CUTE_DEVICE
float2 float2_add(const float2& a, const float2& b) {
    float2 c;
    asm volatile("add.f32x2 %0, %1, %2;\n"
                 : "=l"(reinterpret_cast<uint64_t&>(c))
                 : "l"(reinterpret_cast<uint64_t const&>(a)),
                   "l"(reinterpret_cast<uint64_t const&>(b)));
    return c;
}
CUTE_DEVICE
float2 float2_mul(const float2& a, const float2& b) {
    float2 c;
    asm volatile("mul.f32x2 %0, %1, %2;\n"
                 : "=l"(reinterpret_cast<uint64_t&>(c))
                 : "l"(reinterpret_cast<uint64_t const&>(a)),
                   "l"(reinterpret_cast<uint64_t const&>(b)));
    return c;
}
CUTE_DEVICE
float2 float2_fma(const float2& a, const float2& b, const float2& c) {
    float2 d;
    asm volatile("fma.rn.f32x2 %0, %1, %2, %3;\n"
                 : "=l"(reinterpret_cast<uint64_t&>(d))
                 : "l"(reinterpret_cast<uint64_t const&>(a)),
                   "l"(reinterpret_cast<uint64_t const&>(b)),
                   "l"(reinterpret_cast<uint64_t const&>(c)));
    return d;
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

CUTE_DEVICE void tcgen05_before_thread_sync() {
    asm volatile("tcgen05.fence::before_thread_sync;");
}
CUTE_DEVICE void tcgen05_after_thread_sync() {
    asm volatile("tcgen05.fence::after_thread_sync;");
}

template <int kNumElements>
CUTE_DEVICE void tmem_ld_32dp32bNx(uint32_t tmem_start, void* data_) {
    uint32_t* data = (uint32_t*)data_;
    static_assert(kNumElements == 1 || kNumElements == 2 || kNumElements == 4 ||
                      kNumElements == 8 || kNumElements == 16 ||
                      kNumElements == 32 || kNumElements == 64 ||
                      kNumElements == 128,
                  "Invalid kNumElements");
    [&]<size_t... Is>(cute::index_sequence<Is...>) {
        if constexpr (kNumElements == 1)
            cute::SM100_TMEM_LOAD_32dp32b1x::copy(tmem_start, data[Is]...);
        else if constexpr (kNumElements == 2)
            cute::SM100_TMEM_LOAD_32dp32b2x::copy(tmem_start, data[Is]...);
        else if constexpr (kNumElements == 4)
            cute::SM100_TMEM_LOAD_32dp32b4x::copy(tmem_start, data[Is]...);
        else if constexpr (kNumElements == 8)
            cute::SM100_TMEM_LOAD_32dp32b8x::copy(tmem_start, data[Is]...);
        else if constexpr (kNumElements == 16)
            cute::SM100_TMEM_LOAD_32dp32b16x::copy(tmem_start, data[Is]...);
        else if constexpr (kNumElements == 32)
            cute::SM100_TMEM_LOAD_32dp32b32x::copy(tmem_start, data[Is]...);
        else if constexpr (kNumElements == 64)
            cute::SM100_TMEM_LOAD_32dp32b64x::copy(tmem_start, data[Is]...);
        else
            cute::SM100_TMEM_LOAD_32dp32b128x::copy(tmem_start, data[Is]...);
    }(cute::make_index_sequence<kNumElements>{});
}

template <int kNumElements>
CUTE_DEVICE void tmem_st_32dp32bNx(uint32_t tmem_start, void const* data_) {
    uint32_t const* data = (uint32_t const*)data_;
    static_assert(kNumElements == 1 || kNumElements == 2 || kNumElements == 4 ||
                      kNumElements == 8 || kNumElements == 16 ||
                      kNumElements == 32 || kNumElements == 64 ||
                      kNumElements == 128,
                  "Invalid kNumElements");
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

template <typename TiledMMA, typename TensorA, typename TensorB,
          typename TensorFragC>
CUTE_DEVICE void utcmma_ts(TiledMMA& tiled_mma, TensorA tA_frag, TensorB sB,
                           TensorFragC tC_frag, bool clear_accum) {
    using namespace cute;
    tiled_mma.accumulate_ =
        clear_accum ? UMMA::ScaleOut::Zero : UMMA::ScaleOut::One;
    ThrMMA thr_mma = tiled_mma.get_slice(_0{});
    auto sB_frag = thr_mma.partition_fragment_B(sB);
    static_assert(size<2>(tA_frag) == size<2>(sB_frag));
    CUTE_UNROLL
    for (int k = 0; k < size<2>(tA_frag); ++k) {
        cute::gemm(tiled_mma, tA_frag(_, _, k), sB_frag(_, _, k), tC_frag);
        tiled_mma.accumulate_ = UMMA::ScaleOut::One;
    }
}

template <int MN, int K, int SWIZZLE, typename T>
static constexpr auto make_umma_canonical_k_major_layout() {
    using namespace cute;
    using base_atom_type = std::conditional_t<
        SWIZZLE == 0 || SWIZZLE == 16, UMMA::Layout_K_INTER_Atom<T>,
        std::conditional_t<
            SWIZZLE == 32, UMMA::Layout_K_SW32_Atom<T>,
            std::conditional_t<
                SWIZZLE == 64, UMMA::Layout_K_SW64_Atom<T>,
                std::conditional_t<SWIZZLE == 128, UMMA::Layout_K_SW128_Atom<T>,
                                   void>>>>;
    static_assert(!std::is_same_v<base_atom_type, void>,
                  "Invalid SWIZZLE value");
    return coalesce(tile_to_shape(base_atom_type{}, Shape<Int<MN>, Int<K>>{},
                                  Step<_1, _2>{}),
                    Shape<_1, _1>{});
}

}  // namespace ku

namespace cute {

template <class a_type, class b_type, class c_type, int M, int N>
struct SM100_MMA_F16BF16_NOELECT_TraitsBase {
    using ValTypeD = c_type;
    using ValTypeA = a_type;
    using ValTypeB = b_type;
    using ValTypeC = c_type;

    static constexpr int K = 256 / cute::sizeof_bits<ValTypeA>::value;

    using Shape_MNK = Shape<Int<M>, Int<N>, Int<K>>;
    using ThrID = Layout<_1>;
    using ALayout = Layout<Shape<_1, Shape<Int<M>, Int<K>>>,
                           Stride<_0, Stride<_1, Int<M>>>>;
    using BLayout = Layout<Shape<_1, Shape<Int<N>, Int<K>>>,
                           Stride<_0, Stride<_1, Int<N>>>>;
    using CLayout = Layout<Shape<_1, Shape<Int<M>, Int<N>>>,
                           Stride<_0, Stride<_1, Int<M>>>>;
};

template <class a_type, class b_type, class c_type, int M, int N,
          UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg = UMMA::ScaleIn::One,
          UMMA::ScaleIn b_neg = UMMA::ScaleIn::One>
struct SM100_MMA_F16BF16_WS_TS_NOELECT {
    static_assert(
        M == 32 || M == 64 || M == 128,
        "SM100_MMA_F16BF16_WS_TS_NOELECT M-mode size should be 32, 64 or 128 "
        "for 1 CTA cluster MMA.");
    static_assert(
        N == 64 || N == 128 || N == 256,
        "SM100_MMA_F16BF16_WS_TS_NOELECT N-mode size should be 64, 128 or 256");

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
          UMMA::Major a_major, UMMA::Major b_major, UMMA::ScaleIn a_neg,
          UMMA::ScaleIn b_neg>
struct MMA_Traits<SM100_MMA_F16BF16_WS_TS_NOELECT<
    a_type, b_type, c_type, M, N, a_major, b_major, a_neg, b_neg>>
    : SM100_MMA_F16BF16_NOELECT_TraitsBase<a_type, b_type, c_type, M, N> {
    static_assert(cute::sizeof_bits_v<a_type> == cute::sizeof_bits_v<b_type> &&
                      cute::sizeof_bits_v<b_type> == 16,
                  "SM100_MMA_F16BF16_WS_TS_NOELECT supports 16bit types");

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
        MMA_Traits const& traits, Tensor<TD, DLayout>& D,
        Tensor<TA, ALayout_> const& A, Tensor<TB, BLayout_> const& B,
        Tensor<TC, CLayout_> const& C) {
        static_assert(is_tmem<TD>::value, "Expected tmem in MMA_Atom::call");
        static_assert(is_tmem<TA>::value, "Expected tmem in MMA_Atom::call");
        static_assert(is_rmem<TB>::value,
                      "Expected desc registers in MMA_Atom::call");
        static_assert(is_tmem<TC>::value, "Expected tmem in MMA_Atom::call");

        uint32_t tmem_a = raw_pointer_cast(A.data());
        uint64_t desc_b = B[0];
        uint32_t tmem_c = raw_pointer_cast(D.data());
        uint64_t idesc = UMMA::make_runtime_instr_desc<>(traits.idesc_);
        SM100_MMA_F16BF16_WS_TS_NOELECT<
            a_type, b_type, c_type, M, N, a_major, b_major, a_neg,
            b_neg>::fma(tmem_a, desc_b, tmem_c, uint32_t(traits.accumulate_),
                        idesc);
    }
};

template <class a_type, class b_type, class c_type, int M, int N,
          UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg = UMMA::ScaleIn::One,
          UMMA::ScaleIn b_neg = UMMA::ScaleIn::One>
struct SM100_MMA_F16BF16_WS_SS_NOELECT {
    static_assert(
        M == 32 || M == 64 || M == 128,
        "SM100_MMA_F16BF16_WS_SS_NOELECT M-mode size should be 32, 64 or 128 "
        "for 1 CTA cluster MMA.");
    static_assert(
        N == 64 || N == 128 || N == 256,
        "SM100_MMA_F16BF16_WS_SS_NOELECT N-mode size should be 64, 128 or 256");

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
          UMMA::Major a_major, UMMA::Major b_major, UMMA::ScaleIn a_neg,
          UMMA::ScaleIn b_neg>
struct MMA_Traits<SM100_MMA_F16BF16_WS_SS_NOELECT<
    a_type, b_type, c_type, M, N, a_major, b_major, a_neg, b_neg>>
    : SM100_MMA_F16BF16_NOELECT_TraitsBase<a_type, b_type, c_type, M, N> {
    static_assert(cute::sizeof_bits_v<a_type> == cute::sizeof_bits_v<b_type> &&
                      cute::sizeof_bits_v<b_type> == 16,
                  "SM100_MMA_F16BF16_WS_SS_NOELECT supports 16bit types");

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
        SM100_MMA_F16BF16_WS_SS_NOELECT<
            a_type, b_type, c_type, M, N, a_major, b_major, a_neg,
            b_neg>::fma(desc_a, desc_b, tmem_c, uint32_t(traits.accumulate_),
                        idesc);
    }
};

template <class a_type, class b_type, class c_type, int M, int N,
          UMMA::Major a_major, UMMA::Major b_major,
          UMMA::ScaleIn a_neg = UMMA::ScaleIn::One,
          UMMA::ScaleIn b_neg = UMMA::ScaleIn::One>
struct SM100_MMA_F16BF16_SS_NOELECT {
    static_assert(
        M == 64 || M == 128,
        "SM100_MMA_F16BF16_SS_NOELECT M-mode size should be 64 or 128 for "
        "1 CTA cluster MMA.");
    static_assert(
        (M == 64 && (N % 8 == 0) && (8 <= N) && (N <= 256)) ||
            (M == 128 && (N % 16 == 0) && (16 <= N) && (N <= 256)),
        "SM100_MMA_F16BF16_SS_NOELECT N-mode size should be a multiple of 8 "
        "between 8 and 256 for M=64, or a multiple of 16 for M=128.");

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

template <class a_type, class b_type, class c_type, int M, int N,
          UMMA::Major a_major, UMMA::Major b_major, UMMA::ScaleIn a_neg,
          UMMA::ScaleIn b_neg>
struct MMA_Traits<SM100_MMA_F16BF16_SS_NOELECT<a_type, b_type, c_type, M, N,
                                               a_major, b_major, a_neg, b_neg>>
    : SM100_MMA_F16BF16_NOELECT_TraitsBase<a_type, b_type, c_type, M, N> {
    static_assert(cute::sizeof_bits_v<a_type> == cute::sizeof_bits_v<b_type> &&
                      cute::sizeof_bits_v<b_type> == 16,
                  "SM100_MMA_F16BF16_SS_NOELECT supports 16bit types");

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
