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

#include "kernel/mla/mega_mla/mla_params.cuh"

namespace halops::mla {

/// Split-KV combine + inverse RoPE + MXFP8 quant: the slice kernel gives one
/// thread a 16-byte slice, the row kernel gives one CTA a run of 32-byte
/// pieces of a token row.
template <class Dims>
struct CombineCfg {
    static constexpr int kHeads = Dims::Heads;
    static constexpr int kDimV = Dims::DimV;
    static constexpr int kDimNope = Dims::DimNope;
    static constexpr int kDimRope = Dims::DimRope;
    static constexpr int kOTotal = Dims::OTotal;

    static constexpr int MaxSplits = cute::max(AttnConfig::KvSplit::SegmentCap,
                                               AttnConfig::KvSplit::TailCap);
    static constexpr int SliceThreads = 256;
    static constexpr int SliceDims = sizeof(uint4) / sizeof(__nv_bfloat16);
    static constexpr int SliceLanesPerGroup = SfLayout::Gran / SliceDims;
    static constexpr int Piece = 16;
    static constexpr int PiecesPerThread = 2;
    static constexpr int RowThreads = 128;
    static constexpr int RowCols = RowThreads * Piece * PiecesPerThread;
    static constexpr int RowsPerToken = kOTotal / RowCols;
    static constexpr int RowLanesPerGroup = SfLayout::Gran / Piece;
    static constexpr int GroupWords = SfLayout::Gran / sizeof(uint32_t);
    static constexpr int GroupsPerSfWord = sizeof(uint32_t);

    static constexpr int kCodesPerWord = sizeof(uint32_t);
    static constexpr int kSliceWords = SliceDims / kCodesPerWord;
    static constexpr int kPieceWords = Piece / kCodesPerWord;
    static constexpr int kSfBits = 8;
    static constexpr int kSlotFloats = kHeads * 2;
    static constexpr int kSlicesPerHead = kDimV / SliceDims;
    static constexpr int kPieceStride = RowThreads * Piece;

    static_assert(kOTotal % RowCols == 0, "rows cover whole tokens");
    static_assert((RowThreads * Piece) % kDimV == 0,
                  "a thread's pieces sit at one offset in their head");
    static_assert(kDimNope % Piece == 0 && kDimNope % SliceDims == 0,
                  "a piece or slice is all nope or all rope");
    static_assert(RowLanesPerGroup * kPieceWords == GroupWords,
                  "a group's lanes fill one 8-word store");
};

/// Device arguments of both combine kernels.
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

}  // namespace halops::mla
