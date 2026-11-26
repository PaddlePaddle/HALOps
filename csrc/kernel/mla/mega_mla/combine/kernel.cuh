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

#include <math_constants.h>

#include "kernel/mla/mega_mla/combine/config.cuh"

namespace halops::mla {

template <class Params>
CUTE_DEVICE void combine_reset_barriers(const Params& params, int idx) {
    if (idx < params.n_pbar) params.pbar[idx * params.pbar_stride] = 0;
    if (idx < params.n_c128_flags) params.c128_flags[idx] = 0;
    if (idx < params.n_flags) params.flags[idx] = 0;
}

template <class C>
CUTE_DEVICE void combine_lse_weights(const float2 (&ml)[C::MaxSplits],
                                     float sink, bool has_sink,
                                     float (&weights)[C::MaxSplits]) {
    float max_m = -CUDART_INF_F;
    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < C::MaxSplits; ++s)
        if (ml[s].y != 0.0f) max_m = fmaxf(max_m, ml[s].x);
    float denom = 0.0f;
    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < C::MaxSplits; ++s)
        if (ml[s].y != 0.0f)
            denom = fmaf(exp2f(ml[s].x - max_m), ml[s].y, denom);
    if (max_m != -CUDART_INF_F && has_sink) denom += exp2f(sink - max_m);
    const float inv = (denom == 0.0f || max_m == -CUDART_INF_F)
                          ? 0.0f
                          : __fdividef(1.0f, denom);
    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < C::MaxSplits; ++s)
        weights[s] = ml[s].y != 0.0f ? exp2f(ml[s].x - max_m) * inv : 0.0f;
}

template <class C>
CUTE_DEVICE float2 combine_load_ml(const float* __restrict__ lse_partial,
                                   const TokenSplit split, int seg, int head) {
    return seg < split.nseg
               ? __ldcg(reinterpret_cast<const float2*>(
                     lse_partial +
                     (int64_t)(split.slot0 + seg) * C::kSlotFloats + head * 2))
               : make_float2(-CUDART_INF_F, 0.0f);
}

template <class Params>
CUTE_DEVICE float combine_sink_log2(const Params& params, int head) {
    return params.attn_sink ? params.attn_sink[head] * CUDART_L2E_F
                            : -CUDART_INF_F;
}

template <class C, class Params>
CUTE_DEVICE const float* combine_rope_freqs(const Params& params, int token) {
    return params.freqs_real + (int64_t)params.positions[token] * C::kDimRope;
}

template <class C, class Params>
CUTE_DEVICE void combine_split_weights(const Params& params,
                                       const TokenSplit split, int head,
                                       float sink,
                                       float (&weights)[C::MaxSplits]) {
    float2 ml[C::MaxSplits];
    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < C::MaxSplits; ++s)
        ml[s] = combine_load_ml<C>(params.lse_partial, split, s, head);
    combine_lse_weights<C>(ml, sink, params.attn_sink != nullptr, weights);
}

template <class C, class Params>
CUTE_DEVICE void combine_merge_slice(const Params& params,
                                     const TokenSplit split, int head, int col,
                                     float sink, uint4& out) {
    uint4 raw[C::MaxSplits];
    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < C::MaxSplits; ++s)
        if (s < split.nseg)
            raw[s] = __ldcg(reinterpret_cast<const uint4*>(
                params.o_partial + (int64_t)(split.slot0 + s) * C::kOTotal +
                col));
    float weights[C::MaxSplits];
    combine_split_weights<C>(params, split, head, sink, weights);
    float acc[C::SliceDims] = {};
    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < C::MaxSplits; ++s)
        if (s < split.nseg && weights[s] != 0.0f)
            fma_bf16(weights[s], &raw[s], acc);
    pack_bf16(acc, &out);
}

template <class C>
CUTE_DEVICE void combine_merge_piece(
    const __nv_bfloat16* __restrict__ o_partial, const TokenSplit split,
    int col, const float (&weights)[C::MaxSplits], uint4 (&piece)[2]) {
    float acc[C::Piece] = {};
#pragma unroll 1
    for (int seg = 0; seg < split.nseg; ++seg) {
        float weight = 0.0f;
        CUTLASS_PRAGMA_UNROLL
        for (int s = 0; s < C::MaxSplits; ++s)
            if (s == seg) weight = weights[s];
        if (weight == 0.0f) continue;
        uint4 part[2];
        ld_global_cg_v8(
            o_partial + (int64_t)(split.slot0 + seg) * C::kOTotal + col,
            part[0], part[1]);
        fma_bf16(weight, part, acc);
    }
    pack_bf16(acc, piece);
}

template <class C, class LayoutSF>
__global__ void __launch_bounds__(C::SliceThreads)
    combine_with_rope_quant_slice_kernel(const CombineParams<LayoutSF> params) {
    pdl_trigger();
    const int item = blockIdx.x * C::SliceThreads + threadIdx.x;
    const bool live = item < params.num_tokens * C::kHeads * C::kSlicesPerHead;
    const int token = live ? item / (C::kHeads * C::kSlicesPerHead) : 0;
    const int head = (item / C::kSlicesPerHead) % C::kHeads;
    const int dim0 = (item % C::kSlicesPerHead) * C::SliceDims;
    const bool rope = dim0 >= C::kDimNope;
    float cs[C::SliceDims];
    if (live && rope)
        rope_slice(combine_rope_freqs<C>(params, token), dim0 - C::kDimNope,
                   cs);
    const float sink = live ? combine_sink_log2(params, head) : -CUDART_INF_F;
    pdl_wait();
    const TokenSplit split = live ? params.tok[token] : TokenSplit{};
    combine_reset_barriers(params, item);
    if (!live) return;

    const int col = head * C::kDimV + dim0;
    uint4 merged;
    if (split.nseg <= 1)
        merged = __ldcg(reinterpret_cast<const uint4*>(
            params.q_or_x + (int64_t)token * C::kOTotal + col));
    else
        combine_merge_slice<C>(params, split, head, col, sink, merged);
    float vals[C::SliceDims];
    unpack_bf16(&merged, vals);
    if (rope) inverse_rope_bf16(vals, cs);

    uint32_t words[C::kSliceWords];
    const Ue8m0Scale scale = quantize_group<C::SliceLanesPerGroup>(vals, words);
    *reinterpret_cast<uint2*>(params.out_fp8 + (int64_t)item * C::SliceDims) =
        make_uint2(words[0], words[1]);
    if (item % C::SliceLanesPerGroup == 0) {
        cute::Tensor sf = cute::make_tensor(cute::make_gmem_ptr(params.out_sf),
                                            params.layout_sf);
        sf(sf_row_grouped(token, params.sf_group_tokens, params.sf_groups), col,
           cute::_0{}) = (uint8_t)scale.biased_exp;
    }
}

template <class C, class Params, class SfTensor>
CUTE_DEVICE void combine_quantize_store_piece(const Params& params,
                                              SfTensor& sf, int sf_row,
                                              int token, int col, int tid,
                                              const float (&vals)[C::Piece]) {
    uint32_t words[C::kPieceWords];
    const Ue8m0Scale scale = quantize_group<C::RowLanesPerGroup>(vals, words);
    uint32_t packed[C::GroupWords];
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < C::kPieceWords; ++k) {
        packed[k] = words[k];
        packed[C::kPieceWords + k] = __shfl_down_sync(~0u, words[k], 1);
    }
    if (tid % C::RowLanesPerGroup == 0)
        st_global_v8(params.out_fp8 + (int64_t)token * C::kOTotal + col,
                     packed);
    const int exp_bits = (int)scale.biased_exp;
    uint32_t sf_word = (uint32_t)exp_bits;
    CUTLASS_PRAGMA_UNROLL
    for (int g = 1; g < C::GroupsPerSfWord; ++g)
        sf_word |=
            (uint32_t)__shfl_down_sync(~0u, exp_bits, g * C::RowLanesPerGroup)
            << (C::kSfBits * g);
    if (tid % (C::GroupsPerSfWord * C::RowLanesPerGroup) == 0)
        *reinterpret_cast<uint32_t*>(&sf(sf_row, col, cute::_0{})) = sf_word;
}

template <class C, class LayoutSF>
__global__ void __launch_bounds__(C::RowThreads)
    combine_with_rope_quant_row_kernel(const CombineParams<LayoutSF> params) {
    pdl_trigger();
    const int tid = threadIdx.x;
    const int token = blockIdx.x / C::RowsPerToken;
    const int chunk = blockIdx.x % C::RowsPerToken;
    const int dim0 = (C::Piece * tid) % C::kDimV;
    const bool rope = dim0 >= C::kDimNope;
    float cs[C::Piece];
    if (rope)
        rope_slice(combine_rope_freqs<C>(params, token), dim0 - C::kDimNope,
                   cs);
    pdl_wait();
    const TokenSplit split = params.tok[token];
    const int col0 = chunk * C::RowCols + C::Piece * tid;
    uint4 raw[C::PiecesPerThread][2];
    CUTLASS_PRAGMA_UNROLL
    for (int piece = 0; piece < C::PiecesPerThread; ++piece)
        ld_global_cg_v8(params.q_or_x + (int64_t)token * C::kOTotal + col0 +
                            piece * C::kPieceStride,
                        raw[piece][0], raw[piece][1]);
    combine_reset_barriers(params, blockIdx.x * C::RowThreads + tid);
    if (split.nseg > 1) {
        CUTLASS_PRAGMA_UNROLL
        for (int piece = 0; piece < C::PiecesPerThread; ++piece) {
            const int col = col0 + piece * C::kPieceStride,
                      head = col / C::kDimV;
            float weights[C::MaxSplits];
            combine_split_weights<C>(params, split, head,
                                     combine_sink_log2(params, head), weights);
            combine_merge_piece<C>(params.o_partial, split, col, weights,
                                   raw[piece]);
        }
    }
    cute::Tensor sf =
        cute::make_tensor(cute::make_gmem_ptr(params.out_sf), params.layout_sf);
    const int sf_row =
        sf_row_grouped(token, params.sf_group_tokens, params.sf_groups);
    CUTLASS_PRAGMA_UNROLL
    for (int piece = 0; piece < C::PiecesPerThread; ++piece) {
        float vals[C::Piece];
        unpack_bf16(raw[piece], vals);
        if (rope) inverse_rope_bf16(vals, cs);
        combine_quantize_store_piece<C>(params, sf, sf_row, token,
                                        col0 + piece * C::kPieceStride, tid,
                                        vals);
    }
}

}  // namespace halops::mla
