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

#include <cute/arch/mma_sm100_desc.hpp>

#include "common/sm100_arch.cuh"
#include "kernel/mla/common/intrinsics.cuh"
#include "kernel/mla/pair_mla_decode/config.cuh"

namespace halops::detail {

CUTE_DEVICE uint32_t smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// SW128 UMMA operand: K-major (lbo unused) or MN-major (lbo = next N atom).
CUTE_DEVICE uint64_t sw128_desc(const void* p, uint32_t lbo) {
    cute::UMMA::SmemDescriptor d;
    d.version_ = 1;
    d.layout_type_ = uint8_t(cute::UMMA::LayoutType::SWIZZLE_128B);
    d.start_address_ = smem_u32(p) >> HcaIsa::kDescShift;
    d.leading_byte_offset_ = lbo >> HcaIsa::kDescShift;
    d.stride_byte_offset_ = kSw128AtomBytes >> HcaIsa::kDescShift;
    return d;
}

template <class C, int N, cute::UMMA::Major BMajor>
CUTE_DEVICE uint32_t instr_desc() {
    return static_cast<uint32_t>(
        cute::UMMA::make_runtime_instr_desc<
            cutlass::bfloat16_t, cutlass::bfloat16_t, float, 2 * C::kHeads, N,
            cute::UMMA::Major::K, BMajor>() >>
        HcaIsa::kInstrDescShift);
}

CUTE_DEVICE void umma_ss(uint32_t d, uint64_t a, uint64_t b, uint32_t idesc,
                         bool accumulate) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n}\n" ::"r"(d),
        "l"(a), "l"(b), "r"(idesc), "r"(uint32_t(accumulate)));
}

CUTE_DEVICE void umma_ts(uint32_t d, uint32_t a, uint64_t b, uint32_t idesc,
                         bool accumulate) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], [%1], %2, %3, p;\n}\n" ::"r"(
            d),
        "r"(a), "l"(b), "r"(idesc), "r"(uint32_t(accumulate)));
}

CUTE_DEVICE void tma_gather4(const CUtensorMap* map, uint64_t* bar, void* dst,
                             int col, int r0, int r1, int r2, int r3) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4.mbarrier:"
        ":complete_tx::bytes [%0], [%1, {%2, %3, %4, %5, %6}], [%7];" ::"r"(
            smem_u32(dst)),
        "l"(map), "r"(col), "r"(r0), "r"(r1), "r"(r2), "r"(r3),
        "r"(smem_u32(bar))
        : "memory");
}

// Row r and its swizzled 16-byte chunk c in a SW128 tile of 8-row atoms.
CUTE_DEVICE int row_offset(int r) {
    return (r / kSw128AtomRows) * kSw128AtomBytes +
           (r % kSw128AtomRows) * kSw128AtomRowBytes;
}
CUTE_DEVICE int sw128_offset(int r, int c) {
    return row_offset(r) + (c ^ (r % kSw128AtomRows)) * kChunk;
}

CUTE_DEVICE uint4 dequant8(uint32_t lo, uint32_t hi, uint8_t e8) {
    const uint32_t raw[2] = {lo, hi};
    const auto* f8 = reinterpret_cast<const __nv_fp8x2_e4m3*>(raw);
    const nv_bfloat16 scale =
        __ushort_as_bfloat16(uint16_t(e8) << kBf16MantissaBits);
    uint4 out;
    auto* o = reinterpret_cast<nv_bfloat162*>(&out);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kElemsPerChunk / 2; ++i) {
        o[i] = fp8x2_to_bf16x2_with_scale(f8[i], scale);
    }
    return out;
}

// DSA variant: the scale is a direct fp32 multiplier (one per 128-dim group),
// not HCA's power-of-two UE8M0 code, so it is folded in as bf16 here (the P
// matmul rounds to bf16 anyway).  A disabled key passes scale 0.0f, which
// zeroes the dequantized row exactly as the HCA mask does.
CUTE_DEVICE uint4 dequant8_f32(uint32_t lo, uint32_t hi, float scale) {
    const uint32_t raw[2] = {lo, hi};
    const auto* f8 = reinterpret_cast<const __nv_fp8x2_e4m3*>(raw);
    const nv_bfloat16 s = __float2bfloat16(scale);
    uint4 out;
    auto* o = reinterpret_cast<nv_bfloat162*>(&out);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kElemsPerChunk / 2; ++i) {
        o[i] = fp8x2_to_bf16x2_with_scale(f8[i], s);
    }
    return out;
}

template <int N>
CUTE_DEVICE void tmem_ld(uint32_t addr, float (&v)[N]) {
    tmem_ld_32dp32bNx<N>(addr, v);
    cutlass::arch::fence_view_async_tmem_load();
}

CUTE_DEVICE void cp_async4(int* dst, const int* src) {
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(smem_u32(dst)),
        "l"(src)
        : "memory");
}

CUTE_DEVICE void cp_async8(void* dst, const void* src) {
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 8;" ::"r"(smem_u32(dst)),
        "l"(src)
        : "memory");
}

CUTE_DEVICE void cp_async16(void* dst, const void* src) {
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem_u32(dst)),
        "l"(src)
        : "memory");
}

// Arrives on bar once this thread's earlier cp.async copies have landed; the
// barrier's expected count includes the arrival.
CUTE_DEVICE void cp_async_arrive_noinc(uint64_t* bar) {
    asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(
                     smem_u32(bar))
                 : "memory");
}

CUTE_DEVICE void tma_load_2d(const CUtensorMap* map, uint64_t* bar, void* dst,
                             int col, int row) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::"
        "complete_tx::bytes [%0], [%1, {%2, %3}], [%4];" ::"r"(smem_u32(dst)),
        "l"(map), "r"(col), "r"(row), "r"(smem_u32(bar))
        : "memory");
}

CUTE_DEVICE void tma_store_2d(const CUtensorMap* map, const void* src, int col,
                              int row) {
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2, %3}], "
        "[%1];" ::"l"(map),
        "r"(smem_u32(src)), "r"(col), "r"(row)
        : "memory");
}

CUTE_DEVICE void bulk_commit() {
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}

// This thread's bulk stores have read their shared memory source.
CUTE_DEVICE void bulk_wait_read() {
    asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
}

// A lane's row of a 128-byte swizzled box: its address with the row's
// swizzle phase xor'ed in, so chunk Q is at row_x ^ 16Q.  The xor is done at
// each access: precomputed chunk addresses are 8 registers per box row.
CUTE_DEVICE uint32_t swizzled_row(const void* box, int row) {
    return (smem_u32(box) + row * kSw128AtomRowBytes) |
           (row % kSw128AtomRows) * kChunk;
}

template <int Q>
CUTE_DEVICE uint4 lds_chunk(uint32_t row_x) {
    uint4 v;
    asm volatile(
        "{\n\t.reg .b32 a;\n\txor.b32 a, %4, %5;\n\t"
        "ld.shared.v4.u32 {%0, %1, %2, %3}, [a];\n\t}"
        : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
        : "r"(row_x), "n"(Q * kChunk)
        : "memory");
    return v;
}

template <int Q>
CUTE_DEVICE void sts_chunk(uint32_t row_x, uint32_t a, uint32_t b, uint32_t c,
                           uint32_t d) {
    asm volatile(
        "{\n\t.reg .b32 a;\n\txor.b32 a, %0, %1;\n\t"
        "st.shared.v4.u32 [a], {%2, %3, %4, %5};\n\t}" ::"r"(row_x),
        "n"(Q * kChunk), "r"(a), "r"(b), "r"(c), "r"(d)
        : "memory");
}

// The Q/O warp group's named barrier, taking a value derived from data just
// loaded from shared memory: the barrier then waits for those loads, so a
// TMA write into their source issued after it cannot overtake loads still in
// flight (a plain bar.sync does not order them before the async proxy).
CUTE_DEVICE uint32_t xor_words(const uint4* v, int n) {
    uint32_t x = 0;
    for (int i = 0; i < n; ++i) x ^= v[i].x ^ v[i].w;
    return x;
}

template <class C>
CUTE_DEVICE void qo_sync_after(uint32_t x) {
    constexpr uint32_t kId =
        C::kQoBar +
        uint32_t(cutlass::arch::ReservedNamedBarriers::FirstUserBarrier);
    asm volatile(
        "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %0, 0;\n\t"
        "bar.red.or.pred p, %1, %2, p;\n\t}" ::"r"(x),
        "n"(kId), "n"(kWarpGroup)
        : "memory");
}

template <int N>
CUTE_DEVICE void store_out(__nv_bfloat16* dst, const float* v, float scale) {
    CUTLASS_PRAGMA_UNROLL
    for (int q = 0; q < N; q += HcaIsa::kStoreElems) {
        uint32_t w[HcaIsa::kStoreWords];
        CUTLASS_PRAGMA_UNROLL
        for (int e = 0; e < HcaIsa::kStoreWords; ++e) {
            const nv_bfloat162 b2 = __floats2bfloat162_rn(
                v[q + 2 * e] * scale, v[q + 2 * e + 1] * scale);
            w[e] = reinterpret_cast<const uint32_t&>(b2);
        }
        st_global_v8_na(dst + q, w);
    }
}

CUTE_DEVICE void red_release_add(int* ptr, int v) {
    asm volatile("red.release.gpu.global.add.s32 [%0], %1;" ::"l"(ptr), "r"(v)
                 : "memory");
}

CUTE_DEVICE int ld_acquire(const int* ptr) {
    int v;
    asm volatile("ld.acquire.gpu.global.b32 %0, [%1];"
                 : "=r"(v)
                 : "l"(ptr)
                 : "memory");
    return v;
}

CUTE_DEVICE void bulk_load(void* dst, const void* src, uint32_t bytes,
                           uint64_t* bar) {
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
        "[%0], [%1], %2, [%3];" ::"r"(smem_u32(dst)),
        "l"(src), "r"(bytes), "r"(smem_u32(bar))
        : "memory");
}

}  // namespace halops::detail
