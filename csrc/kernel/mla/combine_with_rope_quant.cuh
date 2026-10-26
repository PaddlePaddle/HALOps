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

#include "kernel/mla/mla_params.cuh"

namespace halops::mla {

struct CombineCfg {
    static constexpr int MaxSplits = cute::max(AttnConfig::KvSplit::SegmentCap,
                                               AttnConfig::KvSplit::TailCap);
    static constexpr int SliceThreads = 256;
    static constexpr int SliceDims = sizeof(uint4) / sizeof(__nv_bfloat16);
    static constexpr int SliceLanesPerGroup = SfLayout::Gran / SliceDims;
    static constexpr int Piece = 16;
    static constexpr int PiecesPerThread = 2;
    static constexpr int RowThreads = 128;
    static constexpr int RowCols = RowThreads * Piece * PiecesPerThread;
    static constexpr int RowsPerToken = ModelDims::OTotal / RowCols;
    static constexpr int RowLanesPerGroup = SfLayout::Gran / Piece;
    static constexpr int GroupWords = SfLayout::Gran / sizeof(uint32_t);
    static constexpr int GroupsPerSfWord = sizeof(uint32_t);
    static_assert(ModelDims::OTotal % RowCols == 0, "rows cover whole tokens");
    static_assert((RowThreads * Piece) % ModelDims::DimV == 0,
                  "a thread's pieces sit at one offset in their head");
    static_assert(ModelDims::DimNope % Piece == 0 &&
                      ModelDims::DimNope % SliceDims == 0,
                  "a piece or slice is all nope or all rope");
};

template <class LayoutSF>
struct CombineParams {
    const __nv_bfloat16* q_or_x;
    const __nv_bfloat16* o_partial;
    const float* lse_partial;
    const TokenSplit* tok;
    const int* positions;
    const float* freqs_real;
    const float* attn_sink;
    int num_tokens;
    int* pbar;
    int n_pbar;
    int pbar_stride;
    int* c128_flags;
    int n_c128_flags;
    uint8_t* out_fp8;
    uint8_t* out_sf;
    LayoutSF layout_sf;
    int sf_group_tokens, sf_groups;
    int* flags;
    int n_flags;
};

template <class Params>
CUTE_DEVICE void reset_barriers(const Params& params, int idx) {
    if (idx < params.n_pbar) params.pbar[idx * params.pbar_stride] = 0;
    if (idx < params.n_c128_flags) params.c128_flags[idx] = 0;
    if (idx < params.n_flags) params.flags[idx] = 0;
}

CUTE_DEVICE void lse_weights(const float2 (&ml)[CombineCfg::MaxSplits],
                             float sink, bool has_sink,
                             float (&weights)[CombineCfg::MaxSplits]) {
    float max_m = -CUDART_INF_F;
    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < CombineCfg::MaxSplits; ++s)
        if (ml[s].y != 0.0f) max_m = fmaxf(max_m, ml[s].x);
    float denom = 0.0f;
    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < CombineCfg::MaxSplits; ++s)
        if (ml[s].y != 0.0f)
            denom = fmaf(exp2f(ml[s].x - max_m), ml[s].y, denom);
    if (max_m != -CUDART_INF_F && has_sink) denom += exp2f(sink - max_m);
    const float inv = (denom == 0.0f || max_m == -CUDART_INF_F)
                          ? 0.0f
                          : __fdividef(1.0f, denom);
    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < CombineCfg::MaxSplits; ++s)
        weights[s] = ml[s].y != 0.0f ? exp2f(ml[s].x - max_m) * inv : 0.0f;
}

CUTE_DEVICE float2 load_ml(const float* __restrict__ lse_partial,
                           const TokenSplit split, int seg, int head) {
    constexpr int kSlotFloats = ModelDims::Heads * 2;
    return seg < split.nseg
               ? __ldcg(reinterpret_cast<const float2*>(
                     lse_partial + (int64_t)(split.slot0 + seg) * kSlotFloats +
                     head * 2))
               : make_float2(-CUDART_INF_F, 0.0f);
}

template <class Params>
CUTE_DEVICE float sink_log2(const Params& params, int head) {
    return params.attn_sink ? params.attn_sink[head] * CUDART_L2E_F
                            : -CUDART_INF_F;
}

template <class Params>
CUTE_DEVICE const float* rope_freqs(const Params& params, int token) {
    return params.freqs_real +
           (int64_t)params.positions[token] * ModelDims::DimRope;
}

template <class Params>
CUTE_DEVICE void split_weights(const Params& params, const TokenSplit split,
                               int head, float sink,
                               float (&weights)[CombineCfg::MaxSplits]) {
    float2 ml[CombineCfg::MaxSplits];
    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < CombineCfg::MaxSplits; ++s)
        ml[s] = load_ml(params.lse_partial, split, s, head);
    lse_weights(ml, sink, params.attn_sink != nullptr, weights);
}

template <class Params>
CUTE_DEVICE void merge_slice(const Params& params, const TokenSplit split,
                             int head, int col, float sink, uint4& out) {
    uint4 raw[CombineCfg::MaxSplits];
    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < CombineCfg::MaxSplits; ++s)
        if (s < split.nseg)
            raw[s] = __ldcg(reinterpret_cast<const uint4*>(
                params.o_partial +
                (int64_t)(split.slot0 + s) * ModelDims::OTotal + col));
    float weights[CombineCfg::MaxSplits];
    split_weights(params, split, head, sink, weights);
    float acc[CombineCfg::SliceDims] = {};
    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < CombineCfg::MaxSplits; ++s)
        if (s < split.nseg && weights[s] != 0.0f)
            fma_bf16(weights[s], &raw[s], acc);
    pack_bf16(acc, &out);
}

CUTE_DEVICE void merge_piece(const __nv_bfloat16* __restrict__ o_partial,
                             const TokenSplit split, int col,
                             const float (&weights)[CombineCfg::MaxSplits],
                             uint4 (&piece)[2]) {
    float acc[CombineCfg::Piece] = {};
#pragma unroll 1
    for (int seg = 0; seg < split.nseg; ++seg) {
        float weight = 0.0f;
        CUTLASS_PRAGMA_UNROLL
        for (int s = 0; s < CombineCfg::MaxSplits; ++s)
            if (s == seg) weight = weights[s];
        if (weight == 0.0f) continue;
        uint4 part[2];
        ld_global_cg_v8(
            o_partial + (int64_t)(split.slot0 + seg) * ModelDims::OTotal + col,
            part[0], part[1]);
        fma_bf16(weight, part, acc);
    }
    pack_bf16(acc, piece);
}

template <class LayoutSF>
__global__ void __launch_bounds__(CombineCfg::SliceThreads)
    combine_with_rope_quant_slice_kernel(const CombineParams<LayoutSF> params) {
    using Cfg = CombineCfg;
    constexpr int kSlicesPerHead = ModelDims::DimV / Cfg::SliceDims;
    pdl_trigger();
    const int item = blockIdx.x * Cfg::SliceThreads + threadIdx.x;
    const bool live =
        item < params.num_tokens * ModelDims::Heads * kSlicesPerHead;
    const int token = live ? item / (ModelDims::Heads * kSlicesPerHead) : 0;
    const int head = (item / kSlicesPerHead) % ModelDims::Heads;
    const int dim0 = (item % kSlicesPerHead) * Cfg::SliceDims;
    const bool rope = dim0 >= ModelDims::DimNope;
    float cs[Cfg::SliceDims];
    if (live && rope)
        rope_slice(rope_freqs(params, token), dim0 - ModelDims::DimNope, cs);
    const float sink = live ? sink_log2(params, head) : -CUDART_INF_F;
    pdl_wait();
    const TokenSplit split = live ? params.tok[token] : TokenSplit{};
    reset_barriers(params, item);
    if (!live) return;

    const int col = head * ModelDims::DimV + dim0;
    uint4 merged;
    if (split.nseg <= 1)
        merged = __ldcg(reinterpret_cast<const uint4*>(
            params.q_or_x + (int64_t)token * ModelDims::OTotal + col));
    else
        merge_slice(params, split, head, col, sink, merged);
    float vals[Cfg::SliceDims];
    unpack_bf16(&merged, vals);
    if (rope) inverse_rope_bf16(vals, cs);

    uint32_t words[Cfg::SliceDims / 4];
    const Ue8m0Scale scale =
        quantize_group<Cfg::SliceLanesPerGroup>(vals, words);
    *reinterpret_cast<uint2*>(params.out_fp8 + (int64_t)item * Cfg::SliceDims) =
        make_uint2(words[0], words[1]);
    if (item % Cfg::SliceLanesPerGroup == 0) {
        cute::Tensor sf = cute::make_tensor(cute::make_gmem_ptr(params.out_sf),
                                            params.layout_sf);
        sf(sf_row_grouped(token, params.sf_group_tokens, params.sf_groups), col,
           cute::_0{}) = (uint8_t)scale.biased_exp;
    }
}

template <class Params, class SfTensor>
CUTE_DEVICE void quantize_store_piece(const Params& params, SfTensor& sf,
                                      int sf_row, int token, int col, int tid,
                                      const float (&vals)[CombineCfg::Piece]) {
    using Cfg = CombineCfg;
    constexpr int kPieceWords = Cfg::Piece / 4;
    static_assert(Cfg::RowLanesPerGroup * kPieceWords == Cfg::GroupWords,
                  "a group's lanes fill one 8-word store");
    uint32_t words[kPieceWords];
    const Ue8m0Scale scale = quantize_group<Cfg::RowLanesPerGroup>(vals, words);
    uint32_t packed[Cfg::GroupWords];
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < kPieceWords; ++k) {
        packed[k] = words[k];
        packed[kPieceWords + k] = __shfl_down_sync(0xffffffffu, words[k], 1);
    }
    if (tid % Cfg::RowLanesPerGroup == 0)
        st_global_v8(params.out_fp8 + (int64_t)token * ModelDims::OTotal + col,
                     packed);
    const int exp_bits = (int)scale.biased_exp;
    uint32_t sf_word = (uint32_t)exp_bits;
    CUTLASS_PRAGMA_UNROLL
    for (int g = 1; g < Cfg::GroupsPerSfWord; ++g)
        sf_word |= (uint32_t)__shfl_down_sync(0xffffffffu, exp_bits,
                                              g * Cfg::RowLanesPerGroup)
                   << (8 * g);
    if (tid % (Cfg::GroupsPerSfWord * Cfg::RowLanesPerGroup) == 0)
        *reinterpret_cast<uint32_t*>(&sf(sf_row, col, cute::_0{})) = sf_word;
}

template <class LayoutSF>
__global__ void __launch_bounds__(CombineCfg::RowThreads)
    combine_with_rope_quant_row_kernel(const CombineParams<LayoutSF> params) {
    using Cfg = CombineCfg;
    constexpr int kPieceStride = Cfg::RowThreads * Cfg::Piece;
    pdl_trigger();
    const int tid = threadIdx.x;
    const int token = blockIdx.x / Cfg::RowsPerToken;
    const int chunk = blockIdx.x % Cfg::RowsPerToken;
    const int dim0 = (Cfg::Piece * tid) % ModelDims::DimV;
    const bool rope = dim0 >= ModelDims::DimNope;
    float cs[Cfg::Piece];
    if (rope)
        rope_slice(rope_freqs(params, token), dim0 - ModelDims::DimNope, cs);
    pdl_wait();
    const TokenSplit split = params.tok[token];
    const int col0 = chunk * Cfg::RowCols + Cfg::Piece * tid;
    uint4 raw[Cfg::PiecesPerThread][2];
    CUTLASS_PRAGMA_UNROLL
    for (int piece = 0; piece < Cfg::PiecesPerThread; ++piece)
        ld_global_cg_v8(params.q_or_x + (int64_t)token * ModelDims::OTotal +
                            col0 + piece * kPieceStride,
                        raw[piece][0], raw[piece][1]);
    reset_barriers(params, blockIdx.x * Cfg::RowThreads + tid);
    if (split.nseg > 1) {
        CUTLASS_PRAGMA_UNROLL
        for (int piece = 0; piece < Cfg::PiecesPerThread; ++piece) {
            const int col = col0 + piece * kPieceStride,
                      head = col / ModelDims::DimV;
            float weights[Cfg::MaxSplits];
            split_weights(params, split, head, sink_log2(params, head),
                          weights);
            merge_piece(params.o_partial, split, col, weights, raw[piece]);
        }
    }
    cute::Tensor sf =
        cute::make_tensor(cute::make_gmem_ptr(params.out_sf), params.layout_sf);
    const int sf_row =
        sf_row_grouped(token, params.sf_group_tokens, params.sf_groups);
    CUTLASS_PRAGMA_UNROLL
    for (int piece = 0; piece < Cfg::PiecesPerThread; ++piece) {
        float vals[Cfg::Piece];
        unpack_bf16(raw[piece], vals);
        if (rope) inverse_rope_bf16(vals, cs);
        quantize_store_piece(params, sf, sf_row, token,
                             col0 + piece * kPieceStride, tid, vals);
    }
}

}  // namespace halops::mla
