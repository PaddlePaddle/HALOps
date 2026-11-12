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

#include <cutlass/numeric_types.h>

#include <algorithm>
#include <cute/config.hpp>

#include "common/cuda_utils.h"

namespace halops::indexer {

using e4m3 = cutlass::float_e4m3_t;
using ue8m0 = cutlass::float_ue8m0_t;
using bf16 = cutlass::bfloat16_t;

struct IndexerWarpTraits {
    static constexpr int NumThreads = 384;
    static constexpr int WarpSize = 32;
    static constexpr int NumWarps = NumThreads / WarpSize;
    static constexpr int ThreadsPerWarpGroup = 128;
    static constexpr int NumWarpGroups = NumThreads / ThreadsPerWarpGroup;
    static constexpr int WarpsPerWarpGroup = ThreadsPerWarpGroup / WarpSize;

    static constexpr int MinTokenTile = 16;

    static constexpr int RendezvousSlotBytes = 128;
    static constexpr int SingleTokenThreads = 256;
    static constexpr int TensorMemoryColumns = 512;
    static constexpr int SharedMemoryBudget = 226 * 1024;
    static constexpr int DefaultSharedMemoryLimit = 48 * 1024;

    struct Registers {
        static constexpr int Producer = 96;
        static constexpr int LocalQueryProducer = 160;
        static constexpr int AccumulatorConsumer = 232;
        static constexpr int EightWarpConsumer = 192;
        static constexpr int VectorConsumer = 176;
    };

    static CUTE_DEVICE int warp_idx() { return threadIdx.x / WarpSize; }
    static CUTE_DEVICE int lane_idx() { return threadIdx.x & (WarpSize - 1); }

    struct Barriers {
        template <uint32_t Id, int Width>
        struct Bar {
            static constexpr uint32_t id = Id;
            static constexpr int width = Width;
            static_assert(Width % WarpSize == 0,
                          "Named barriers require whole warps");
            static_assert(Width <= NumThreads,
                          "Barrier participants exceed the block size");
        };

        using QuantizerPair01Sync = Bar<2, 2 * WarpSize>;
        using QuantizerPair23Sync = Bar<3, 2 * WarpSize>;
        using EveryoneSync = Bar<4, NumThreads>;
        using AccumulatorGroupSync = Bar<5, ThreadsPerWarpGroup>;
        using VectorGroupSync = Bar<6, ThreadsPerWarpGroup>;
    };

    struct Roles {
        static constexpr int MmaWarp = 0;
        static constexpr int WeightLoaderWarp = 1;
        static constexpr int ActivationLoaderWarp = 2;
        static constexpr int PrefetchWarp = 3;
        static constexpr int DrainWarpBegin = 4;
        static constexpr int DrainWarps = WarpsPerWarpGroup;
        static constexpr int VectorWarpBegin = 8;
        static constexpr int VectorWarps = WarpsPerWarpGroup;

        __device__ static constexpr bool is_mma(int warp) {
            return warp == MmaWarp;
        }
        __device__ static constexpr bool is_weight_loader(int warp) {
            return warp == WeightLoaderWarp;
        }
        __device__ static constexpr bool is_activation_loader(int warp) {
            return warp == ActivationLoaderWarp;
        }
        __device__ static constexpr bool is_prefetch(int warp) {
            return warp == PrefetchWarp;
        }
        __device__ static constexpr bool is_drain(int warp) {
            return warp >= DrainWarpBegin && warp < DrainWarpBegin + DrainWarps;
        }
        __device__ static constexpr bool is_vector(int warp) {
            return warp >= VectorWarpBegin &&
                   warp < VectorWarpBegin + VectorWarps;
        }
        __device__ static constexpr int drain_local_warp(int warp) {
            return warp - DrainWarpBegin;
        }

        static constexpr int roles_matching(int warp) {
            return (warp == MmaWarp) + (warp == WeightLoaderWarp) +
                   (warp == ActivationLoaderWarp) + (warp == PrefetchWarp) +
                   (warp >= DrainWarpBegin &&
                    warp < DrainWarpBegin + DrainWarps) +
                   (warp >= VectorWarpBegin &&
                    warp < VectorWarpBegin + VectorWarps);
        }

        static constexpr bool partitions_all_warps() {
            for (int warp = 0; warp < NumWarps; ++warp)
                if (roles_matching(warp) != 1) return false;
            return true;
        }

        static_assert(
            DrainWarpBegin % WarpsPerWarpGroup == 0 &&
                VectorWarpBegin % WarpsPerWarpGroup == 0,
            "Accumulator and vector roles require aligned warp groups");
        static_assert(
            MmaWarp < WarpsPerWarpGroup &&
                WeightLoaderWarp < WarpsPerWarpGroup &&
                ActivationLoaderWarp < WarpsPerWarpGroup &&
                PrefetchWarp < WarpsPerWarpGroup,
            "Matrix and loader roles must share the first warp group");
    };

    static_assert(NumWarps == 12 && NumWarpGroups == 3);
};

template <class GemmConfig>
struct ValidateGemmConfig {
    static_assert(
        GemmConfig::BM == 128,
        "M must match the 128-row scale tile and TMEM quadrant layout");
    static_assert(GemmConfig::BN >= IndexerWarpTraits::MinTokenTile &&
                      GemmConfig::BN <= GemmConfig::ScaleRowsPerTile &&
                      (GemmConfig::BN & (GemmConfig::BN - 1)) == 0,
                  "N must be a power of two in [16, 128]");
    static_assert(GemmConfig::BK > 0 && GemmConfig::KSplit > 0 &&
                      GemmConfig::TmaStages >= 1 && GemmConfig::AccStages >= 1,
                  "Tile, split and pipeline stage counts must be positive");
    static_assert(GemmConfig::KSplit <= GemmConfig::KSplitMax,
                  "Split count exceeds partial-result capacity");
    static_assert(GemmConfig::SwapAB || (GemmConfig::BN == GemmConfig::BM),
                  "Unswapped operands require a square tile");
    static constexpr bool ok = true;
};

template <int TokenTile_>
struct IndexerGemmConfig {
    static constexpr int TokenTile = TokenTile_;
    static_assert(TokenTile == 16 || TokenTile == 32 || TokenTile == 64 ||
                  TokenTile == 128);
    static constexpr int BM = 128;
    static constexpr int BN = TokenTile;
    static constexpr int BK = 256;
    static constexpr bool SwapAB = TokenTile < BM;

    static constexpr int ScaleGroupSize = 32;
    static constexpr int ScaleRowsPerColumn = 32;
    static constexpr int ScaleRowsPerTile = 4 * ScaleRowsPerColumn;
    static constexpr int MinPackedScaleTile = 64;
    static constexpr int KItersPerBlock = BK / ScaleGroupSize;

    static constexpr int TmaStages = 2;
    static constexpr int AccStages = 2;
    static constexpr int KSplit = 1;
    static constexpr int KSplitMax = 1;
};

using GemmShapeDefaults = IndexerGemmConfig<128>;

struct IndexerDimensions {
    static constexpr int DEFAULT_HIDDEN = 4096;
    static constexpr int HEAD_DIM = 128;
    static constexpr int NUM_HEADS = 64;
    static constexpr int ROPE_DIM = 64;

    static constexpr int KEY_WEIGHT_ROWS = HEAD_DIM + NUM_HEADS;
    static constexpr int PADDED_KEY_WEIGHT_ROWS = 256;
    static constexpr int QUERY_ROWS = NUM_HEADS * HEAD_DIM;

    static constexpr int CACHE_SCALE_BYTES =
        4;  // One FP32 scale or four UE8M0 scales.
    static constexpr int FP8_CACHE_TOKEN_BYTES = HEAD_DIM + CACHE_SCALE_BYTES;
    static constexpr int MX_GROUP_SIZE = 32;
    static constexpr int MX_SCALE_BYTES = HEAD_DIM / MX_GROUP_SIZE;
    static constexpr int FP4_CACHE_TOKEN_BYTES = HEAD_DIM / 2 + MX_SCALE_BYTES;

    static constexpr float HADAMARD_SCALE =
        0.08838834764831845f;  // 1/sqrt(128)
    static constexpr float FP8_MAX = 448.0f;
    static constexpr float FP8_AMAX_FLOOR = 1e-4f;

    static_assert(HEAD_DIM == 2 * ROPE_DIM,
                  "RoPE covers the first half of head_dim");
    static_assert((HEAD_DIM & (HEAD_DIM - 1)) == 0,
                  "the 128-point Hadamard needs head_dim to be a power of two");
    static_assert(
        PADDED_KEY_WEIGHT_ROWS % 128 == 0 &&
            PADDED_KEY_WEIGHT_ROWS >= KEY_WEIGHT_ROWS,
        "Padded projection rows must cover K/weights in 128-row tiles");
    static_assert(HEAD_DIM % 32 == 0,
                  "the K epilogue is one warp per token with head_dim/32 "
                  "elements per lane");
};

template <int TokenTileSize, bool IsMxSf = false, bool IsFp4 = false,
          bool UseSharedRope = false, bool EightWarpQuery = false,
          bool NeoX = true, int RopeBuffers = 1, bool AlignedGlobalRope = false,
          bool TwoPassQuery = false>
struct IndexerTraitsFor : IndexerWarpTraits {
    using Dimensions = IndexerDimensions;
    static constexpr int TokenTile = TokenTileSize;
    static constexpr bool PrefetchRope = UseSharedRope;
    static constexpr bool is_mx_sf = IsMxSf;
    static constexpr bool is_fp4 = IsFp4;
    static_assert(!is_fp4 || is_mx_sf, "FP4 output requires MX scales");
    static constexpr bool EightWarpEpilogue = EightWarpQuery;
    static constexpr bool IsNeoX = NeoX;
    static constexpr int RopeStages = RopeBuffers;
    static constexpr bool RopeAligned16 = UseSharedRope || AlignedGlobalRope;
    static constexpr bool SplitQueryDrain = TwoPassQuery;
    static_assert(RopeStages == 1 || RopeStages == 2);

    using GemmBase = IndexerGemmConfig<TokenTile>;

    struct KeyWeight : GemmBase {
        static constexpr int BK = 128;  // bf16: 128 x 2 B == fp8's 256 x 1 B
        static constexpr int KSplit = 16;
        static constexpr int KSplitMax = 16;
    };

    using Query = GemmBase;

    static constexpr int MaxTmaStages = GemmBase::TmaStages;
    static constexpr int MaxAccStages = GemmBase::AccStages;

    static_assert(KeyWeight::BK * 2 == Query::BK * 1,
                  "the two ranges' per-stage A/B byte counts must match");

    struct Workspace {
        static constexpr int GridBarrierBytes = 128;
        static constexpr int RendezvousSlotBytes =
            IndexerWarpTraits::RendezvousSlotBytes;
        static constexpr int BlockAlignBytes = 128;
        static constexpr int TokenPadRows = GemmBase::ScaleRowsPerTile;
    };

    static constexpr bool validated =
        ValidateGemmConfig<KeyWeight>::ok && ValidateGemmConfig<Query>::ok;
};

using IndexerTraits = IndexerTraitsFor<128>;
static_assert(IndexerTraits::validated);
static_assert(IndexerTraitsFor<16>::validated);
static_assert(IndexerTraitsFor<32>::validated);
static_assert(IndexerTraits::Roles::partitions_all_warps(),
              "Roles must partition all 12 warps exactly once");

inline int select_token_tile(int num_tokens) {
    return num_tokens <= 16 ? 16 : (num_tokens <= 32 ? 32 : 128);
}

inline int64_t align128(int64_t bytes) {
    constexpr int64_t alignment = IndexerTraits::Workspace::BlockAlignBytes;
    return (bytes + alignment - 1) / alignment * alignment;
}

// Narrow batches occupy one token group.
inline int token_group_count(int num_tokens) {
    const int tile = select_token_tile(num_tokens);
    return (num_tokens + tile - 1) / tile;
}

struct RawWeightPrefetch {
    static constexpr int Disabled = 0;
    static constexpr int RopeWarp = 1;
    static constexpr int WeightLoaderWarp = 2;
};

inline int indexer_sm_count() {
    int major = 0;
    HALOPS_CUDA_CHECK(cudaDeviceGetAttribute(
        &major, cudaDevAttrComputeCapabilityMajor, CurrentDevice()));
    TORCH_CHECK(major == 10, "indexer_prepare requires an SM100-family GPU");
    const int sm_count = CurrentDeviceSmCount();
    TORCH_CHECK(sm_count >= 128,
                "indexer_prepare requires at least 128 available SMs, got ",
                sm_count);
    return sm_count;
}

inline int select_grid_blocks(int num_tokens, int available_sms,
                              bool fp8_schedule = false) {
    if (!fp8_schedule && num_tokens > 96 && num_tokens <= 128)
        return std::min(available_sms, 112);
    if (fp8_schedule && num_tokens >= 240 && num_tokens <= 256)
        return std::min(available_sms, 144);
    if (fp8_schedule && num_tokens > 384 && num_tokens <= 400)
        return available_sms;
    if (num_tokens == 1) return std::min(available_sms, 64);
    if (num_tokens <= 4) return std::min(available_sms, 80);
    if (num_tokens <= IndexerTraits::TokenTile)
        return std::min(available_sms, 96);
    if (num_tokens <= 2 * IndexerTraits::TokenTile)
        return std::min(available_sms, 128);
    if (num_tokens <= 1024) return std::min(available_sms, 144);
    return available_sms;
}

inline int select_key_weight_splits(int tokens, int grid_blocks,
                                    int key_weight_k_tiles,
                                    bool fp8_schedule = false) {
    auto fixed_split = [key_weight_k_tiles](int split) {
        return key_weight_k_tiles % split == 0 ? split : 0;
    };
    if (fp8_schedule && tokens >= 240 && tokens <= 256)
        if (const int split = fixed_split(4)) return split;
    if (tokens > 128 && tokens <= 256)
        if (const int split = fixed_split(8)) return split;
    if (tokens > 256 && tokens < 512 && tokens % 128 != 0)
        if (const int split = fixed_split(8)) return split;
    if (tokens > 256 && tokens <= 1024)
        if (const int split = fixed_split(4)) return split;
    constexpr int weight_tiles =
        IndexerDimensions::PADDED_KEY_WEIGHT_ROWS / IndexerTraits::GemmBase::BM;
    const int groups = token_group_count(tokens);
    int splits = 1;
    while (splits * 2 <= IndexerTraits::KeyWeight::KSplitMax &&
           key_weight_k_tiles % (splits * 2) == 0 &&
           (int64_t)weight_tiles * groups * (splits * 2) <= grid_blocks)
        splits *= 2;
    return splits;
}

inline int select_query_block_offset(int key_weight_work_items,
                                     int query_work_items, int grid_blocks) {
    if (key_weight_work_items == 0 || query_work_items == 0) return 0;
    // Prioritize CTAs with no KW work for the longest Q work lists.
    return std::min(grid_blocks - 1, key_weight_work_items);
}

template <class KernelTraits, class Parameters>
void configure_indexer_schedule(Parameters& params, int sm_count,
                                bool query_aligned_32) {
    using Dimensions = typename KernelTraits::Dimensions;
    using KeyWeightConfig = typename KernelTraits::KeyWeight;
    using QueryConfig = typename KernelTraits::Query;
    params.grid_blocks = select_grid_blocks(params.num_tokens, sm_count,
                                            !KernelTraits::is_mx_sf);
    params.token_groups = token_group_count(params.num_tokens);
    TORCH_CHECK(params.grid_blocks >= 1,
                "indexer_prepare: the grid must be positive");
    TORCH_CHECK(
        select_token_tile(params.num_tokens) == KernelTraits::TokenTile,
        "indexer_prepare: the tile ladder and the instantiated tile disagree (",
        select_token_tile(params.num_tokens), " vs ", KernelTraits::TokenTile,
        ")");
    TORCH_CHECK(params.input_hidden_size % KeyWeightConfig::BK == 0,
                "hidden_size must be divisible by the key projection tile ",
                KeyWeightConfig::BK, ", got ", params.input_hidden_size);
    const int key_weight_k_tiles =
        params.input_hidden_size / KeyWeightConfig::BK;
    params.key_weight_splits =
        select_key_weight_splits(params.num_tokens, params.grid_blocks,
                                 key_weight_k_tiles, !KernelTraits::is_mx_sf);
    TORCH_CHECK(key_weight_k_tiles % params.key_weight_splits == 0,
                "the kw K-split must divide the k-tile count; got ",
                key_weight_k_tiles,
                " tiles for KSplit=", params.key_weight_splits);
    const int key_weight_tiles_per_split =
        key_weight_k_tiles / params.key_weight_splits;
    TORCH_CHECK(key_weight_tiles_per_split >= KeyWeightConfig::TmaStages,
                "Each split needs at least one K tile per stage; got tiles=",
                key_weight_tiles_per_split,
                " stages=", KeyWeightConfig::TmaStages);

    static_assert(Dimensions::QUERY_ROWS % QueryConfig::BM == 0);
    TORCH_CHECK(
        params.query_lora_rank % (QueryConfig::BK * QueryConfig::KSplit) == 0,
        "q_lora_rank must be divisible by the query projection tile ",
        QueryConfig::BK * QueryConfig::KSplit, ", got ",
        params.query_lora_rank);
    const int query_tiles_per_split =
        params.query_lora_rank / (QueryConfig::BK * QueryConfig::KSplit);
    TORCH_CHECK(query_tiles_per_split >= QueryConfig::TmaStages,
                "q_lora_rank must provide at least ", QueryConfig::TmaStages,
                " query K tiles, got ", params.query_lora_rank);
    params.key_weight_tiles_per_split = key_weight_tiles_per_split;
    params.query_tiles_per_split = query_tiles_per_split;

    constexpr int WEIGHT_TILES =
        Dimensions::PADDED_KEY_WEIGHT_ROWS / KernelTraits::KeyWeight::BM;
    params.simt_projection_enabled =
        params.num_tokens == 1 && params.grid_blocks == Dimensions::NUM_HEADS;
    params.key_weight_items_per_group = WEIGHT_TILES * params.key_weight_splits;
    params.key_weight_work_items =
        params.key_weight_items_per_group * params.token_groups;
    if (params.uses_simt_projection()) params.key_weight_work_items = 0;
    constexpr int max_local_query_tokens = 32;
    params.local_query_enabled =
        params.num_tokens >= 2 && params.num_tokens <= max_local_query_tokens &&
        params.grid_blocks ==
            (params.num_tokens <= 4 ? 80 : 64 + 2 * params.key_weight_splits) &&
        (params.key_weight_splits == 16 ||
         (params.num_tokens > 4 && params.key_weight_splits >= 4));
    if (params.uses_local_query() && params.num_tokens <= 4) {
        params.key_weight_items_per_group = params.key_weight_splits;
        params.key_weight_work_items = params.key_weight_splits;
    }
    params.query_items_per_group = params.query_rows / KernelTraits::Query::BM;
    params.query_work_items =
        params.query_items_per_group * params.token_groups;
    params.use_256bit_stores = query_aligned_32;
    params.prefetch_small_rope =
        params.uses_local_query() && params.num_tokens > 4;
    params.raw_weight_prefetch_mode =
        KernelTraits::PrefetchRope && !KernelTraits::is_mx_sf
            ? (params.num_tokens <= 256 || params.num_tokens % 128 == 0
                   ? RawWeightPrefetch::RopeWarp
                   : RawWeightPrefetch::WeightLoaderWarp)
            : RawWeightPrefetch::Disabled;
    params.query_block_offset = halops::indexer::select_query_block_offset(
        params.key_weight_work_items, params.query_work_items,
        params.grid_blocks);
    // Eight-warp Q blocks use a separate range from K/weight blocks.
    params.key_weight_blocks =
        KernelTraits::EightWarpEpilogue
            ? std::clamp(
                  (KernelTraits::is_mx_sf && params.num_tokens > 96 ? 48 : 32),
                  1, params.grid_blocks - 1)
            : 0;
    // Long Q work lists need a loader independent of accumulator reuse.
    if (params.raw_weight_prefetch_mode ==
            RawWeightPrefetch::WeightLoaderWarp &&
        params.query_work_items >
            2 * (params.grid_blocks - params.key_weight_blocks))
        params.raw_weight_prefetch_mode = RawWeightPrefetch::RopeWarp;
    params.vector_blocks = params.key_weight_blocks ? params.key_weight_blocks
                                                    : params.grid_blocks;
    if (params.uses_local_query())
        params.vector_blocks = params.key_weight_work_items;
    params.independent_weights =
        KernelTraits::is_mx_sf &&
        (!KernelTraits::is_fp4 ||
         (!params.uses_simt_projection() &&
          (!params.uses_local_query() || params.num_tokens > 4)));
    if (params.uses_local_query())
        params.query_block_offset = params.key_weight_work_items;
}

enum class IndexerKernelKind {
    General,
    SingleToken,
    LocalQuery,
    ThreeWaySmallBatch
};

template <class KernelTraits, class Parameters>
IndexerKernelKind select_indexer_kernel(const Parameters& params) {
    if constexpr (KernelTraits::TokenTile == 16) {
        if (params.uses_local_query() && params.num_tokens >= 5)
            return IndexerKernelKind::ThreeWaySmallBatch;
        if (params.uses_simt_projection())
            return IndexerKernelKind::SingleToken;
    }
    if constexpr (KernelTraits::TokenTile <= 64)
        if (params.uses_local_query()) return IndexerKernelKind::LocalQuery;
    return IndexerKernelKind::General;
}

CUTE_HOST_DEVICE constexpr int query_epilogue_lanes(int tokens,
                                                    bool local_query) {
    if (tokens <= 4) return 32;
    if (!local_query) return 8;
    if (tokens <= 8) return 16;
    return tokens > 16 ? 4 : 8;
}

struct IndexerLaunchConfig {
    int token_tile;
    bool two_pass_query;
    bool eight_warp_query;
    bool shared_rope;
    int rope_stages;
};

inline IndexerLaunchConfig select_indexer_launch_config(int tokens,
                                                        bool is_mx_sf,
                                                        bool is_fp4,
                                                        int sm_count) {
    const bool two_pass_query =
        (!is_fp4 && tokens > 128 && tokens <= 144) ||
        (tokens > 256 && tokens < (is_mx_sf ? 512 : 384) && tokens % 128 > 0 &&
         tokens % 128 <= (is_mx_sf ? 48 : 16));
    const bool eight_warp_query = tokens > 32 && tokens <= 128 &&
                                  select_grid_blocks(tokens, sm_count) > 32;
    const bool shared_rope = !(is_mx_sf && (is_fp4 || !two_pass_query) &&
                               tokens > 128 && tokens <= 256);
    return {select_token_tile(tokens), two_pass_query, eight_warp_query,
            shared_rope, tokens > 256 && tokens % 128 != 0 ? 2 : 1};
}

}  // namespace halops::indexer
