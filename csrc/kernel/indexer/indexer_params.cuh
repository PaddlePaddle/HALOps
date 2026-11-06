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
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "indexer_config.cuh"
#include "indexer_gemm.cuh"
#include "indexer_utils.cuh"

namespace halops::indexer {

template <class KernelTraits>
struct IndexerSharedStorage {
    using Dimensions = typename KernelTraits::Dimensions;
    using Roles = typename KernelTraits::Roles;

    char gemm_storage[pipeline_smem_bytes<KernelTraits::TokenTile>()];

    static constexpr int QStageStride = Dimensions::HEAD_DIM + 4;
    union {
        float transpose[KernelTraits::Query::SwapAB ? KernelTraits::TokenTile
                                                    : 1][QStageStride];
        alignas(16) float rope[KernelTraits::RopeStages]
                              [KernelTraits::PrefetchRope
                                   ? KernelTraits::TokenTile
                                   : 1][Dimensions::ROPE_DIM + 4];
    } query;

    cute::array_aligned<uint32_t, 1> tmem_start_addr;

    transac_bar_t tma_a_ready[KernelTraits::MaxTmaStages];
    transac_bar_t tma_b_ready[KernelTraits::MaxTmaStages];
    transac_bar_t gemm_buffer_free[KernelTraits::MaxTmaStages];
    transac_bar_t accumulator_ready[KernelTraits::MaxAccStages];
    transac_bar_t accumulator_free[KernelTraits::MaxAccStages];
    transac_bar_t local_weight_ready;
    transac_bar_t norm_ready;
    transac_bar_t rope_ready[KernelTraits::RopeStages];
    transac_bar_t rope_free[KernelTraits::RopeStages];
    transac_bar_t query_weights_ready[2];
    transac_bar_t query_weights_free[2];
    float local_weight;
    float local_weights[64];
    alignas(16) float norm_weight[Dimensions::HEAD_DIM];
    alignas(16) float norm_bias[Dimensions::HEAD_DIM];
    alignas(16) float small_batch_rope[KernelTraits::TokenTile <= 32
                                           ? KernelTraits::TokenTile
                                           : 1][Dimensions::ROPE_DIM + 4];
    alignas(16) float query_weights[2][KernelTraits::PrefetchRope &&
                                               !KernelTraits::is_mx_sf
                                           ? KernelTraits::TokenTile
                                           : 1];

    static constexpr int TMEM_COLS = KernelTraits::TensorMemoryColumns;

    static_assert(sizeof(gemm_storage) + sizeof(query) <
                      KernelTraits::SharedMemoryBudget,
                  "GEMM and Q staging exceed the shared-memory budget");

    CUTE_DEVICE GemmPipelineBarriers gemm_barriers() {
        return {tma_a_ready, tma_b_ready, gemm_buffer_free, accumulator_ready,
                accumulator_free};
    }

    CUTE_DEVICE char* gemm_shared_memory() { return gemm_storage; }

    CUTE_DEVICE void initialize_gemm_barriers(int drain_threads) {
        CUTE_UNROLL
        for (int i = 0; i < KernelTraits::MaxTmaStages; ++i) {
            tma_a_ready[i].init(1);
            tma_b_ready[i].init(1);
            gemm_buffer_free[i].init(1);
        }
        CUTE_UNROLL
        for (int i = 0; i < KernelTraits::MaxAccStages; ++i) {
            accumulator_ready[i].init(1);
            accumulator_free[i].init(drain_threads);
        }
    }

    CUTE_DEVICE void initialize_barriers(int drain_threads = 128) {
        initialize_gemm_barriers(drain_threads);
        local_weight_ready.init(1);
        norm_ready.init(32);
        CUTE_UNROLL
        for (int stage = 0; stage < KernelTraits::RopeStages; ++stage) {
            rope_ready[stage].init(32);
            rope_free[stage].init(drain_threads);
        }
        if constexpr (KernelTraits::PrefetchRope && !KernelTraits::is_mx_sf) {
            CUTE_UNROLL
            for (int stage = 0; stage < 2; ++stage) {
                query_weights_ready[stage].init(32);
                query_weights_free[stage].init(drain_threads);
            }
        }
        cutlass::arch::fence_barrier_init();
    }

    CUTE_DEVICE void acquire_tmem() {
        cute::TMEM::Allocator1Sm().allocate(TMEM_COLS, tmem_start_addr.data());
        HALOPS_INDEXER_DEVICE_ASSERT(tmem_start_addr.data()[0] == 0);
        cute::TMEM::Allocator1Sm().release_allocation_lock();
    }

    CUTE_DEVICE void release_tmem() {
        cute::TMEM::Allocator1Sm().free(0, TMEM_COLS);
    }
};

// FP32 weights are rounded to the selected output dtype.
enum class IndexerWeightsDtype : int { Float32, BFloat16, Float16 };

struct IndexerHostParams {
    torch::Tensor x;
    torch::Tensor q_lora;
    torch::Tensor wk_w;
    torch::Tensor wq_b;
    torch::Tensor wq_b_sf;
    torch::Tensor k_norm_weight;
    torch::Tensor k_norm_bias;
    float k_norm_eps;
    torch::Tensor rope_cos_sin_cache;
    torch::Tensor positions;
    torch::Tensor paged_kv_cache_with_scale;
    torch::Tensor out_cache_loc;
    torch::Tensor out_q;
    std::optional<torch::Tensor> out_q_sf;
    torch::Tensor out_weights;
    float weight_scale;
    int64_t n_heads;
    int64_t head_dim;
    bool is_neox_style;
    bool apply_hadamard;
    bool use_pdl;
    bool is_mx_sf;
    bool is_fp4;

    IndexerHostParams(
        const torch::Tensor& x, const torch::Tensor& q_lora,
        const torch::Tensor& wk_w, const torch::Tensor& wq_b,
        const torch::Tensor& wq_b_sf, const torch::Tensor& k_norm_weight,
        const torch::Tensor& k_norm_bias, double k_norm_eps,
        const torch::Tensor& rope_cos_sin_cache, const torch::Tensor& positions,
        const torch::Tensor& paged_kv_cache_with_scale,
        const torch::Tensor& out_cache_loc, const torch::Tensor& out_q,
        const std::optional<torch::Tensor>& out_q_sf,
        const torch::Tensor& out_weights, double weight_scale, int64_t n_heads,
        int64_t head_dim, bool is_neox_style, bool apply_hadamard, bool use_pdl,
        bool is_mx_sf, bool is_fp4)
        : x(x),
          q_lora(q_lora),
          wk_w(wk_w),
          wq_b(wq_b),
          wq_b_sf(wq_b_sf),
          k_norm_weight(k_norm_weight),
          k_norm_bias(k_norm_bias),
          k_norm_eps(k_norm_eps),
          rope_cos_sin_cache(rope_cos_sin_cache),
          positions(positions),
          paged_kv_cache_with_scale(paged_kv_cache_with_scale),
          out_cache_loc(out_cache_loc),
          out_q(out_q),
          out_q_sf(out_q_sf),
          out_weights(out_weights),
          weight_scale(weight_scale),
          n_heads(n_heads),
          head_dim(head_dim),
          is_neox_style(is_neox_style),
          apply_hadamard(apply_hadamard),
          use_pdl(use_pdl),
          is_mx_sf(is_mx_sf),
          is_fp4(is_fp4) {
        validate();
    }

  private:
    static void check_indices(const char* name, const torch::Tensor& tensor,
                              int num_tokens) {
        TORCH_CHECK(tensor.scalar_type() == torch::kInt ||
                        tensor.scalar_type() == torch::kLong,
                    name, " must be int32 or int64, got ",
                    tensor.scalar_type());
        TORCH_CHECK(tensor.dim() == 1 && tensor.size(0) == num_tokens, name,
                    " must be [T]=", num_tokens, ", got ", tensor.sizes());
        TORCH_CHECK(tensor.stride(0) == 1, name, " must be contiguous");
    }

    static void check_storage(const char* name, const torch::Tensor& tensor,
                              const torch::Device& device, int alignment,
                              bool contiguous = true) {
        TORCH_CHECK(tensor.is_cuda() && tensor.device() == device, name,
                    " must be on the same CUDA device as x");
        TORCH_CHECK(!contiguous || tensor.is_contiguous(), name,
                    " must be contiguous");
        TORCH_CHECK(
            reinterpret_cast<uintptr_t>(tensor.data_ptr()) % alignment == 0,
            name, " must be ", alignment, " B aligned");
    }

    void validate() const {
        using Dimensions = IndexerDimensions;
        TORCH_CHECK(x.is_cuda(), "x must be CUDA");
        TORCH_CHECK(!is_fp4 || is_mx_sf, "is_fp4 requires is_mx_sf=True");
        const auto device = x.device();
        check_storage("x", x, device, 16, false);
        check_storage("q_lora", q_lora, device, 8, false);
        check_storage("wk_w", wk_w, device, 16);
        check_storage("wq_b", wq_b, device, 16);
        check_storage("wq_b_sf", wq_b_sf, device, 16);
        check_storage("k_norm_weight", k_norm_weight, device, 16);
        check_storage("k_norm_bias", k_norm_bias, device, 16);
        check_storage("rope_cos_sin_cache", rope_cos_sin_cache, device, 8);
        check_storage("positions", positions, device, positions.element_size());
        check_storage("out_cache_loc", out_cache_loc, device,
                      out_cache_loc.element_size());
        check_storage("paged_kv_cache_with_scale", paged_kv_cache_with_scale,
                      device, 4);
        check_storage("out_q", out_q, device, 16);
        if (is_mx_sf) {
            TORCH_CHECK(out_q_sf && out_q_sf->defined(),
                        "out_q_sf is required for MXFP4 or MXFP8 output");
            check_storage("out_q_sf", *out_q_sf, device, 1);
        } else {
            TORCH_CHECK(!out_q_sf, "out_q_sf must be None for FP8 output");
        }
        check_storage("out_weights", out_weights, device, 16);
        TORCH_CHECK(std::isfinite(k_norm_eps) && k_norm_eps > 0,
                    "k_norm_eps must be finite and positive in float32");
        TORCH_CHECK(std::isfinite(weight_scale),
                    "weight_scale must be finite in float32");
        TORCH_CHECK(x.dim() == 2 && x.scalar_type() == torch::kBFloat16,
                    "x must be a 2-D bfloat16 [num_tokens, hidden], got ",
                    x.sizes(), " ", x.scalar_type());
        TORCH_CHECK(x.stride(1) == 1,
                    "x's last dim must have stride 1 (the bf16 TMA reads it "
                    "K-major)");
        TORCH_CHECK(head_dim == Dimensions::HEAD_DIM &&
                        n_heads == Dimensions::NUM_HEADS,
                    "this build compiles head_dim=", Dimensions::HEAD_DIM,
                    ", n_heads=", Dimensions::NUM_HEADS, "; got ", head_dim,
                    ", ", n_heads);
        TORCH_CHECK(x.size(1) == Dimensions::HIDDEN,
                    "this build compiles hidden=", Dimensions::HIDDEN, ", got ",
                    x.size(1));

        TORCH_CHECK(
            x.size(0) > 0 &&
                x.size(0) <= std::numeric_limits<int>::max() -
                                 IndexerTraits::Workspace::TokenPadRows,
            "num_tokens must be positive and fit in int32 with padding");
        const int num_tokens = x.size(0);
        TORCH_CHECK(x.stride(0) >= Dimensions::HIDDEN &&
                        x.stride(0) <= std::numeric_limits<int>::max() &&
                        x.stride(0) % 8 == 0,
                    "x row stride must cover hidden and be 16 B aligned");

        TORCH_CHECK(wk_w.dim() == 2 && wk_w.scalar_type() == torch::kBFloat16,
                    "wk_w must be 2-D bfloat16");
        TORCH_CHECK(wk_w.size(0) == Dimensions::KEY_WEIGHT_ROWS,
                    "wk_w must be [head_dim + n_heads, hidden] = [",
                    Dimensions::KEY_WEIGHT_ROWS, ", ", Dimensions::HIDDEN,
                    "], got ", wk_w.sizes());
        TORCH_CHECK(wk_w.size(1) == Dimensions::HIDDEN && wk_w.stride(1) == 1 &&
                        wk_w.stride(0) == Dimensions::HIDDEN,
                    "wk_w must be contiguous K-major [",
                    Dimensions::KEY_WEIGHT_ROWS, ", ", Dimensions::HIDDEN,
                    "], got ", wk_w.sizes(), " strides ", wk_w.strides());

        TORCH_CHECK(q_lora.dim() == 2 && q_lora.size(0) == num_tokens &&
                        q_lora.size(1) == Dimensions::QUERY_LORA_RANK,
                    "q_lora must be [T, q_lora_rank] = [", num_tokens, ", ",
                    Dimensions::QUERY_LORA_RANK, "], got ", q_lora.sizes());
        TORCH_CHECK(q_lora.stride(1) == 1,
                    "q_lora's last dim must have stride 1");
        TORCH_CHECK(q_lora.scalar_type() == torch::kBFloat16,
                    "q_lora must be bfloat16");
        TORCH_CHECK(
            q_lora.stride(0) >= Dimensions::QUERY_LORA_RANK &&
                q_lora.stride(0) <= std::numeric_limits<int>::max() &&
                q_lora.stride(0) % 4 == 0,
            "q_lora row stride must cover q_lora_rank and be 8 B aligned");

        TORCH_CHECK(
            wq_b.dim() == 2 && wq_b.scalar_type() == torch::kFloat8_e4m3fn,
            "wq_b must be 2-D float8_e4m3fn, got ", wq_b.sizes(), " ",
            wq_b.scalar_type());
        TORCH_CHECK(wq_b.size(0) == Dimensions::QUERY_ROWS &&
                        wq_b.size(1) == Dimensions::QUERY_LORA_RANK &&
                        wq_b.stride(1) == 1 &&
                        wq_b.stride(0) == Dimensions::QUERY_LORA_RANK,
                    "wq_b must be contiguous K-major [", Dimensions::QUERY_ROWS,
                    ", ", Dimensions::QUERY_LORA_RANK, "], got ", wq_b.sizes());
        TORCH_CHECK(wq_b_sf.scalar_type() == torch::kByte &&
                        wq_b_sf.dim() == 2 &&
                        wq_b_sf.size(0) == Dimensions::QUERY_ROWS &&
                        wq_b_sf.size(1) == Dimensions::QUERY_LORA_RANK /
                                               Dimensions::MX_GROUP_SIZE,
                    "wq_b_sf must be the SfAtom-swizzled uint8 [N, K/32] = [",
                    Dimensions::QUERY_ROWS, ", ",
                    (Dimensions::QUERY_LORA_RANK / Dimensions::MX_GROUP_SIZE),
                    "] the block-scaled UMMA reads directly, got ",
                    wq_b_sf.sizes(), " ", wq_b_sf.scalar_type());

        TORCH_CHECK(
            k_norm_weight.scalar_type() == torch::kFloat &&
                k_norm_weight.dim() == 1 &&
                k_norm_weight.numel() == Dimensions::HEAD_DIM,
            "k_norm_weight must be float32 [head_dim]=", Dimensions::HEAD_DIM);
        const bool use_layernorm = k_norm_bias.numel() > 0;
        TORCH_CHECK(
            !use_layernorm || (k_norm_bias.scalar_type() == torch::kFloat &&
                               k_norm_bias.dim() == 1 &&
                               k_norm_bias.numel() == Dimensions::HEAD_DIM),
            "k_norm_bias must be float32 [head_dim] or empty (empty "
            "selects RMSNorm, which has no bias by definition)");

        TORCH_CHECK(rope_cos_sin_cache.scalar_type() == torch::kFloat &&
                        rope_cos_sin_cache.dim() == 2 &&
                        rope_cos_sin_cache.size(0) > 0 &&
                        rope_cos_sin_cache.size(1) == Dimensions::ROPE_DIM,
                    "rope_cos_sin_cache must be float32 [max_position, ",
                    Dimensions::ROPE_DIM,
                    "] in halves layout [cos | sin]; this build compiles "
                    "rope_dim=",
                    Dimensions::ROPE_DIM, ", got ", rope_cos_sin_cache.sizes());
        TORCH_CHECK(rope_cos_sin_cache.stride(1) == 1 &&
                        rope_cos_sin_cache.stride(0) == Dimensions::ROPE_DIM,
                    "rope_cos_sin_cache must be contiguous");
        check_indices("positions", positions, num_tokens);
        check_indices("out_cache_loc", out_cache_loc, num_tokens);

        const int token_bytes = is_fp4 ? Dimensions::FP4_CACHE_TOKEN_BYTES
                                       : Dimensions::FP8_CACHE_TOKEN_BYTES;
        const int q_row_bytes =
            is_fp4 ? Dimensions::HEAD_DIM / 2 : Dimensions::HEAD_DIM;
        TORCH_CHECK(paged_kv_cache_with_scale.scalar_type() == torch::kByte &&
                        paged_kv_cache_with_scale.dim() == 2,
                    "paged_kv_cache_with_scale must be 2-D uint8 [num_pages, "
                    "page_size * token_bytes] (132 for fp8/mxfp8, 68 for "
                    "mxfp4), got ",
                    paged_kv_cache_with_scale.sizes(), " ",
                    paged_kv_cache_with_scale.scalar_type());
        const int64_t row = paged_kv_cache_with_scale.size(1);
        TORCH_CHECK(row <= std::numeric_limits<int>::max() &&
                        paged_kv_cache_with_scale.size(0) > 0 &&
                        paged_kv_cache_with_scale.size(0) <=
                            std::numeric_limits<int>::max() /
                                std::max<int64_t>(1, row / token_bytes),
                    "K cache capacity must be positive and fit in int32 slots");
        TORCH_CHECK(row % token_bytes == 0,
                    "the K cache's row must be page_size * ", token_bytes,
                    " bytes, got ", row);
        const int page_size = row / token_bytes;
        TORCH_CHECK(page_size > 0 && (page_size & (page_size - 1)) == 0,
                    "the page_size implied by the cache shape must be a power "
                    "of two, got ",
                    page_size);
        TORCH_CHECK(paged_kv_cache_with_scale.stride(1) == 1,
                    "paged_kv_cache_with_scale must be row-contiguous");

        TORCH_CHECK(
            out_q.scalar_type() ==
                    (is_fp4 ? torch::kByte : torch::kFloat8_e4m3fn) &&
                out_q.dim() == 3 && out_q.size(0) == num_tokens &&
                out_q.size(1) == Dimensions::NUM_HEADS &&
                out_q.size(2) == q_row_bytes && out_q.is_contiguous(),
            "out_q must be contiguous ", is_fp4 ? "uint8" : "float8_e4m3fn",
            " [T, n_heads, row_bytes] = [", num_tokens, ", ",
            Dimensions::NUM_HEADS, ", ", q_row_bytes, "], got ", out_q.sizes());
        if (is_mx_sf) {
            TORCH_CHECK(
                out_q_sf->scalar_type() == torch::kByte &&
                    out_q_sf->dim() == 3 && out_q_sf->size(0) == num_tokens &&
                    out_q_sf->size(1) == Dimensions::NUM_HEADS &&
                    out_q_sf->size(2) == Dimensions::MX_SCALE_BYTES &&
                    out_q_sf->is_contiguous(),
                "out_q_sf must be contiguous uint8 [T, n_heads, head_dim/32]");
        }
        const auto weights_type = out_weights.scalar_type();
        TORCH_CHECK(
            (weights_type == torch::kFloat ||
             weights_type == torch::kBFloat16 ||
             weights_type == torch::kHalf) &&
                out_weights.dim() == 2 && out_weights.size(0) == num_tokens &&
                out_weights.size(1) == Dimensions::NUM_HEADS &&
                out_weights.is_contiguous(),
            "out_weights must be a contiguous float32, bfloat16 or float16 "
            "[T, n_heads] = [",
            num_tokens, ", ", Dimensions::NUM_HEADS, "], got ", weights_type,
            " ", out_weights.sizes());
    }
};

template <class KernelTraits>
struct IndexerPrepareWorkspace {
    using Dimensions = typename KernelTraits::Dimensions;
    using WorkspaceConfig = typename KernelTraits::Workspace;

    enum : int {
        QUERY_LORA_READY,
        KEY_WEIGHT_READY,
        WEIGHTS_READY,
        QUERY_SCALE_READY
    };
    static constexpr int COUNTER_BANKS = QUERY_SCALE_READY + 1;

    struct Layout {
        int counter_bank_stride;
        int activation_rows;
        int64_t counter_offset;
        int64_t scratch_offset;
        int64_t key_weight_offset;
        int64_t raw_weight_offset;
        int64_t query_lora_offset;
        int64_t query_scale_offset;
        int64_t total_bytes;
        bool has_query_lora;
    };

    static int padded_token_count(int num_tokens) {
        constexpr int alignment = WorkspaceConfig::TokenPadRows;
        return (num_tokens + alignment - 1) / alignment * alignment;
    }

    template <class Parameters>
    static Layout layout(const Parameters& params) {
        Layout result{};
        result.counter_bank_stride = params.token_groups;
        result.counter_offset = WorkspaceConfig::GridBarrierBytes;
        result.scratch_offset = result.counter_offset +
                                (int64_t)COUNTER_BANKS * params.token_groups *
                                    WorkspaceConfig::RendezvousSlotBytes;
        int64_t cursor = result.scratch_offset;
        auto reserve = [&cursor](int64_t bytes) {
            const int64_t offset = cursor;
            cursor += bytes;
            return offset;
        };
        // Partial stores and raw-weight loads are bounded by valid tokens.
        result.key_weight_offset = reserve(
            params.uses_simt_projection()
                ? (int64_t)Dimensions::HEAD_DIM * sizeof(float)
                : (int64_t)params.num_tokens * params.key_weight_splits *
                      Dimensions::KEY_WEIGHT_ROWS * sizeof(float));
        const bool has_raw_weights = !params.uses_simt_projection() &&
                                     !params.uses_local_query() &&
                                     !params.independent_weights;
        result.raw_weight_offset =
            reserve(has_raw_weights ? (int64_t)params.num_tokens *
                                          Dimensions::NUM_HEADS * sizeof(float)
                                    : 0);
        // Small local-Q paths quantize directly into shared memory.
        result.has_query_lora =
            !params.uses_simt_projection() &&
            (!params.uses_local_query() || params.num_tokens > 4);
        result.activation_rows = params.num_tokens;
        result.query_lora_offset =
            reserve(result.has_query_lora ? (int64_t)result.activation_rows *
                                                Dimensions::QUERY_LORA_RANK
                                          : 0);
        // SfAtom requires full 128-row scale tiles even for a partial tile.
        result.query_scale_offset =
            reserve(result.has_query_lora
                        ? (int64_t)padded_token_count(params.num_tokens) *
                              (Dimensions::QUERY_LORA_RANK /
                               KernelTraits::GemmBase::ScaleGroupSize)
                        : 0);
        result.total_bytes = cursor;
        return result;
    }
};

template <class KernelTraits, class Parameters>
struct IndexerBarriers {
    using Workspace = IndexerPrepareWorkspace<KernelTraits>;
    using Bar = PartialBarrier;
    using Roles = typename KernelTraits::Roles;
    static constexpr int TokenTile = KernelTraits::TokenTile;

    const Parameters& params;

    CUTE_DEVICE int valid_tokens(int group) const {
        return min(TokenTile, params.num_tokens - group * TokenTile);
    }
    CUTE_DEVICE Bar of(int bank, int group, int count) const {
        return Bar(params.group_counters,
                   bank * params.counter_bank_stride + group, count);
    }

    CUTE_DEVICE Bar query_lora_ready(int group) const {
        return of(Workspace::QUERY_LORA_READY, group, valid_tokens(group));
    }
    CUTE_DEVICE Bar key_weight_ready(int group) const {
        return of(Workspace::KEY_WEIGHT_READY, group,
                  params.uses_simt_projection()
                      ? IndexerDimensions::NUM_HEADS
                      : params.key_weight_items_per_group);
    }
    CUTE_DEVICE Bar query_scale_ready(int group) const {
        return of(Workspace::QUERY_SCALE_READY, group,
                  params.query_items_per_group);
    }
    CUTE_DEVICE Bar weights_ready(int group) const {
        return of(Workspace::WEIGHTS_READY, group, valid_tokens(group));
    }

    CUTE_DEVICE Bar key_ready() const {
        return of(Workspace::KEY_WEIGHT_READY, 0, params.key_weight_splits);
    }
    CUTE_DEVICE Bar projection_ready() const {
        return of(Workspace::QUERY_SCALE_READY, 0, params.key_weight_splits);
    }

    CUTE_DEVICE void reset_active_warp() const {
        // Inactive slots stay zero. Interleaving banks keeps division constant.
        constexpr int banks = Workspace::COUNTER_BANKS;
        for (int i = KernelTraits::lane_idx(); i < banks * params.token_groups;
             i += KernelTraits::WarpSize)
            Bar(params.group_counters,
                (i % banks) * params.counter_bank_stride + i / banks, 1)
                .reset_relaxed();
    }
};

template <class KernelTraits>
struct IndexerDeviceParams {
    using Dimensions = typename KernelTraits::Dimensions;
    using Workspace = IndexerPrepareWorkspace<KernelTraits>;
    using KeyWeightConfig = typename KernelTraits::KeyWeight;
    using KeyWeightTma =
        decltype(make_bf16_weight_tma<KeyWeightConfig>(nullptr, 0, 0));
    using InputTma =
        decltype(make_bf16_activation_tma<KeyWeightConfig>(nullptr, 0, 0, 0));
    static constexpr int TokenTile = KernelTraits::TokenTile;

    int num_tokens, padded_tokens, token_groups, grid_blocks;
    int key_weight_work_items, key_weight_items_per_group, query_work_items,
        query_items_per_group;
    int key_weight_tiles_per_split, query_tiles_per_split;
    int key_weight_splits;
    int padded_key_weight_rows, input_hidden_size, query_rows;
    int page_size, page_shift, cache_row_bytes;
    int x_row_stride, query_lora_row_stride;
    bool positions_are_int64, cache_slots_are_int64;
    bool use_layernorm, apply_hadamard;
    static constexpr bool is_neox_style = KernelTraits::IsNeoX;
    static constexpr bool is_fp4 = KernelTraits::is_fp4;
    static constexpr bool is_mx_sf = KernelTraits::is_mx_sf;
    static constexpr bool eight_warp_query = KernelTraits::EightWarpEpilogue;
    bool use_pdl, independent_weights, use_256bit_stores;
    bool prefetch_small_rope;
    int raw_weight_prefetch_mode;
    static constexpr bool prefetch_norm = true;
    static constexpr bool all_warps_process_keys =
        KernelTraits::EightWarpEpilogue;
    bool simt_projection_enabled;
    bool local_query_enabled;
    int query_block_offset;
    int key_weight_blocks;
    int vector_blocks;
    float k_norm_eps, weight_scale;
    IndexerWeightsDtype weights_dtype;

    CUTE_HOST_DEVICE bool uses_simt_projection() const {
        if constexpr (KernelTraits::TokenTile == 16)
            return simt_projection_enabled;
        return false;
    }
    CUTE_HOST_DEVICE bool uses_local_query() const {
        if constexpr (KernelTraits::TokenTile <= 32) return local_query_enabled;
        return false;
    }

    CUTE_DEVICE int query_work_start() const {
        if (key_weight_blocks)
            return blockIdx.x < key_weight_blocks
                       ? query_work_items
                       : (int)blockIdx.x - key_weight_blocks;
        const int b = (int)blockIdx.x + grid_blocks - query_block_offset;
        return b >= grid_blocks ? b - grid_blocks : b;
    }
    CUTE_DEVICE int key_weight_work_start() const {
        return key_weight_blocks && blockIdx.x >= key_weight_blocks
                   ? key_weight_work_items
                   : (int)blockIdx.x;
    }
    CUTE_DEVICE int key_weight_work_stride() const {
        return key_weight_blocks ? key_weight_blocks : grid_blocks;
    }
    CUTE_DEVICE int query_work_stride() const {
        return grid_blocks - key_weight_blocks;
    }

    int* __restrict__ grid_completion_counter;
    int* __restrict__ group_counters;
    int counter_bank_stride;

    float* __restrict__ key_weight_partials;
    float* __restrict__ raw_weights;
    e4m3* __restrict__ quantized_query_lora;
    ue8m0* __restrict__ query_lora_scales;

    const __nv_bfloat16* __restrict__ x;
    const __nv_bfloat16* __restrict__ key_weight_projection;
    const __nv_bfloat16* __restrict__ q_lora;
    const float* __restrict__ k_norm_weight;
    const float* __restrict__ k_norm_bias;
    const float* __restrict__ rope;
    const void* __restrict__ positions;
    const void* __restrict__ out_cache_loc;
    uint8_t* __restrict__ key_cache;
    uint8_t* __restrict__ q_output;
    uint8_t* __restrict__ q_scale;
    void* __restrict__ weights;

    KeyWeightTma key_weight_tma;
    InputTma input_tma;
    WeightDescriptors<KernelTraits::TokenTile> query_weight_descriptors;
    ActivationDescriptors<KernelTraits::TokenTile> query_lora_descriptors;

    CUTE_DEVICE void prefetch_descriptors() const {
        if (!uses_simt_projection() &&
            (!uses_local_query() || blockIdx.x < key_weight_work_items)) {
            cute::prefetch_tma_descriptor(key_weight_tma.get_tma_descriptor());
            cute::prefetch_tma_descriptor(input_tma.get_tma_descriptor());
        }
        if (uses_local_query() && blockIdx.x < key_weight_work_items) return;
        if constexpr (KernelTraits::Query::SwapAB) {
            cute::prefetch_tma_descriptor(
                query_weight_descriptors.tma_a.get_tma_descriptor());
            cute::prefetch_tma_descriptor(
                query_weight_descriptors.tma_sfa.get_tma_descriptor());
            if (!uses_simt_projection() &&
                (!uses_local_query() || num_tokens > 4)) {
                cute::prefetch_tma_descriptor(
                    query_lora_descriptors.tma_b.get_tma_descriptor());
                cute::prefetch_tma_descriptor(
                    query_lora_descriptors.tma_sfb.get_tma_descriptor());
            }
        } else {
            cute::prefetch_tma_descriptor(
                query_weight_descriptors.tma_b.get_tma_descriptor());
            cute::prefetch_tma_descriptor(
                query_weight_descriptors.tma_sfb.get_tma_descriptor());
            cute::prefetch_tma_descriptor(
                query_lora_descriptors.tma_a.get_tma_descriptor());
            cute::prefetch_tma_descriptor(
                query_lora_descriptors.tma_sfa.get_tma_descriptor());
        }
    }

    CUTE_DEVICE int position_of(int token) const {
        return positions_are_int64
                   ? (int)__ldg(static_cast<const int64_t*>(positions) + token)
                   : __ldg(static_cast<const int*>(positions) + token);
    }
    CUTE_DEVICE int cache_loc_of(int token) const {
        return cache_slots_are_int64
                   ? (int)__ldg(static_cast<const int64_t*>(out_cache_loc) +
                                token)
                   : __ldg(static_cast<const int*>(out_cache_loc) + token);
    }

    // Store consecutive heads with round-to-nearest-even conversion.
    CUTE_DEVICE void store_weights(int64_t offset, float value) const {
        if (weights_dtype == IndexerWeightsDtype::BFloat16)
            static_cast<__nv_bfloat16*>(weights)[offset] =
                __float2bfloat16_rn(value);
        else if (weights_dtype == IndexerWeightsDtype::Float16)
            static_cast<__half*>(weights)[offset] = __float2half_rn(value);
        else
            static_cast<float*>(weights)[offset] = value;
    }
    CUTE_DEVICE void store_weights(int64_t offset, float2 value) const {
        if (weights_dtype == IndexerWeightsDtype::Float32)
            *reinterpret_cast<float2*>(static_cast<float*>(weights) + offset) =
                value;
        else
            *reinterpret_cast<uint32_t*>(static_cast<uint16_t*>(weights) +
                                         offset) = pack_16bit_weights(value);
    }
    CUTE_DEVICE void store_weights(int64_t offset, float4 value) const {
        if (weights_dtype == IndexerWeightsDtype::Float32)
            *reinterpret_cast<float4*>(static_cast<float*>(weights) + offset) =
                value;
        else
            *reinterpret_cast<uint2*>(static_cast<uint16_t*>(weights) +
                                      offset) =
                make_uint2(pack_16bit_weights(make_float2(value.x, value.y)),
                           pack_16bit_weights(make_float2(value.z, value.w)));
    }
    // Two BF16 or FP16 weights; the first occupies the low half.
    CUTE_DEVICE uint32_t pack_16bit_weights(float2 value) const {
        if (weights_dtype == IndexerWeightsDtype::BFloat16) {
            const __nv_bfloat162_raw pair = __float22bfloat162_rn(value);
            return pair.x | (uint32_t)pair.y << 16;
        }
        const __half2_raw pair = __float22half2_rn(value);
        return pair.x | (uint32_t)pair.y << 16;
    }
};

struct WorkspaceCacheEntry {
    int device_index;
    cudaStream_t stream;
    unsigned long long capture_id;
    int64_t counter_prefix_bytes;
    torch::Tensor storage;
};

static __global__ void clear_indexer_counter_prefix_kernel(
    uint8_t* prefix, int64_t counter_prefix_bytes, bool use_pdl) {
    if (use_pdl) cudaGridDependencySynchronize();
    // The indexer may initialize private resources before waiting for this
    // grid.
    if (use_pdl && threadIdx.x == 0) cudaTriggerProgrammaticLaunchCompletion();
    for (int64_t offset = threadIdx.x; offset < counter_prefix_bytes;
         offset += blockDim.x)
        prefix[offset] = 0;
}

inline void clear_indexer_counter_prefix(uint8_t* prefix,
                                         int64_t counter_prefix_bytes,
                                         cudaStream_t stream, bool use_pdl) {
    constexpr int Threads = 256;
    if (use_pdl) {
        cudaLaunchAttribute attribute{};
        attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attribute.val.programmaticStreamSerializationAllowed = 1;
        cudaLaunchConfig_t config{};
        config.gridDim = dim3(1);
        config.blockDim = dim3(Threads);
        config.stream = stream;
        config.attrs = &attribute;
        config.numAttrs = 1;
        HALOPS_CUDA_CHECK(
            cudaLaunchKernelEx(&config, clear_indexer_counter_prefix_kernel,
                               prefix, counter_prefix_bytes, use_pdl));
    } else {
        clear_indexer_counter_prefix_kernel<<<1, Threads, 0, stream>>>(
            prefix, counter_prefix_bytes, use_pdl);
    }
    HALOPS_CUDA_CHECK(cudaGetLastError());
}

inline torch::Tensor acquire_workspace(const torch::Tensor& x,
                                       int64_t required_bytes,
                                       int64_t counter_prefix_bytes,
                                       bool use_pdl) {
    // Each host thread owns one entry per device/stream.
    thread_local std::vector<WorkspaceCacheEntry> workspace_cache;
    const int device_index = x.device().index();
    const cudaStream_t stream = GetCurrentCUDAStream(x);
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    unsigned long long capture_id = 0;
    if (stream != cudaStreamLegacy) {
        HALOPS_CUDA_CHECK(
            cudaStreamGetCaptureInfo(stream, &capture_status, &capture_id));
        TORCH_CHECK(capture_status != cudaStreamCaptureStatusInvalidated,
                    "indexer_prepare cannot use an invalidated CUDA capture");
        if (capture_status == cudaStreamCaptureStatusNone) capture_id = 0;
    }
    WorkspaceCacheEntry* cached = nullptr;
    for (auto& entry : workspace_cache) {
        if (entry.device_index != device_index || entry.stream != stream)
            continue;
        cached = &entry;
        if (entry.capture_id == capture_id &&
            entry.counter_prefix_bytes == counter_prefix_bytes &&
            entry.storage.numel() == required_bytes)
            return entry.storage;
        break;
    }
    // A new capture needs graph-pool storage and its own initialization node.
    auto storage =
        torch::empty({required_bytes}, x.options().dtype(torch::kUInt8));
    clear_indexer_counter_prefix(storage.data_ptr<uint8_t>(),
                                 counter_prefix_bytes, stream, use_pdl);
    WorkspaceCacheEntry entry{device_index, stream, capture_id,
                              counter_prefix_bytes, storage};
    if (cached)
        *cached = std::move(entry);
    else
        workspace_cache.push_back(std::move(entry));
    return storage;
}

template <class KernelTraits>
auto make_indexer_device_params(const IndexerHostParams& host, int sm_count,
                                torch::Tensor& workspace) {
    using Dimensions = typename KernelTraits::Dimensions;
    using KeyWeightConfig = typename KernelTraits::KeyWeight;
    constexpr int TokenTile = KernelTraits::TokenTile;

    IndexerDeviceParams<KernelTraits> params{};
    params.num_tokens = host.x.size(0);
    params.padded_tokens =
        IndexerPrepareWorkspace<KernelTraits>::padded_token_count(
            params.num_tokens);
    params.x_row_stride = host.x.stride(0);
    params.query_lora_row_stride = host.q_lora.stride(0);
    params.k_norm_eps = host.k_norm_eps;
    params.weight_scale = host.weight_scale;
    params.use_layernorm = host.k_norm_bias.numel() > 0;
    params.positions_are_int64 = host.positions.scalar_type() == torch::kLong;
    params.cache_slots_are_int64 =
        host.out_cache_loc.scalar_type() == torch::kLong;
    params.cache_row_bytes = host.paged_kv_cache_with_scale.size(1);
    const int token_bytes = host.is_fp4 ? Dimensions::FP4_CACHE_TOKEN_BYTES
                                        : Dimensions::FP8_CACHE_TOKEN_BYTES;
    params.page_size = params.cache_row_bytes / token_bytes;
    while ((1LL << params.page_shift) < params.page_size) ++params.page_shift;
    const auto weights_type = host.out_weights.scalar_type();
    params.weights_dtype =
        weights_type == torch::kBFloat16 ? IndexerWeightsDtype::BFloat16
        : weights_type == torch::kHalf   ? IndexerWeightsDtype::Float16
                                         : IndexerWeightsDtype::Float32;
    params.apply_hadamard = host.apply_hadamard;
    params.use_pdl = host.use_pdl;
    configure_indexer_schedule<KernelTraits>(
        params, sm_count,
        (reinterpret_cast<uintptr_t>(host.out_q.data_ptr()) & 31) == 0);
    params.padded_key_weight_rows = Dimensions::PADDED_KEY_WEIGHT_ROWS;
    params.input_hidden_size = Dimensions::HIDDEN;
    params.query_rows = Dimensions::QUERY_ROWS;

    using Workspace = IndexerPrepareWorkspace<KernelTraits>;
    const auto layout = Workspace::layout(params);
    // Establish this thread's CUDA context before driver TMA calls.
    workspace = acquire_workspace(host.x, layout.total_bytes,
                                  layout.scratch_offset, host.use_pdl);
    auto* base = static_cast<char*>(workspace.data_ptr());
    params.grid_completion_counter = reinterpret_cast<int*>(base);
    params.group_counters =
        reinterpret_cast<int*>(base + layout.counter_offset);
    params.counter_bank_stride = layout.counter_bank_stride;
    params.key_weight_partials =
        reinterpret_cast<float*>(base + layout.key_weight_offset);
    params.raw_weights =
        reinterpret_cast<float*>(base + layout.raw_weight_offset);
    params.quantized_query_lora =
        reinterpret_cast<e4m3*>(base + layout.query_lora_offset);
    params.query_lora_scales =
        reinterpret_cast<ue8m0*>(base + layout.query_scale_offset);

    params.key_weight_tma = make_bf16_weight_tma<KeyWeightConfig>(
        host.wk_w.data_ptr(), Dimensions::KEY_WEIGHT_ROWS, Dimensions::HIDDEN);
    params.input_tma = make_bf16_activation_tma<KeyWeightConfig>(
        host.x.data_ptr(), params.num_tokens, params.x_row_stride,
        Dimensions::HIDDEN);

    params.x = static_cast<const __nv_bfloat16*>(host.x.data_ptr());
    params.key_weight_projection =
        static_cast<const __nv_bfloat16*>(host.wk_w.data_ptr());
    params.q_lora = static_cast<const __nv_bfloat16*>(host.q_lora.data_ptr());
    params.k_norm_weight = host.k_norm_weight.data_ptr<float>();
    params.k_norm_bias = params.use_layernorm
                             ? host.k_norm_bias.data_ptr<float>()
                             : host.k_norm_weight.data_ptr<float>();
    params.rope = host.rope_cos_sin_cache.data_ptr<float>();
    params.positions = host.positions.data_ptr();
    params.out_cache_loc = host.out_cache_loc.data_ptr();
    params.key_cache = host.paged_kv_cache_with_scale.data_ptr<uint8_t>();
    params.q_output = static_cast<uint8_t*>(host.out_q.data_ptr());
    params.q_scale =
        host.is_mx_sf ? host.out_q_sf->data_ptr<uint8_t>() : nullptr;
    params.weights = host.out_weights.data_ptr();

    params.query_weight_descriptors = make_weight_descriptors<TokenTile>(
        host.wq_b.data_ptr(), host.wq_b_sf.data_ptr(), Dimensions::QUERY_ROWS,
        Dimensions::QUERY_LORA_RANK);
    if (layout.has_query_lora)
        params.query_lora_descriptors = make_activation_descriptors<TokenTile>(
            params.quantized_query_lora, params.query_lora_scales,
            params.padded_tokens, Dimensions::QUERY_LORA_RANK,
            params.padded_tokens, layout.activation_rows);
    return params;
}

template <class KernelTraits, class Parameters, class KeyWeightResources,
          class QueryResources, class Barriers>
struct IndexerContext {
    const Parameters& params;
    halops::indexer::IndexerSharedStorage<KernelTraits>& plan;
    const KeyWeightResources& key_weight_resources;
    const QueryResources& query_resources;
    const Barriers& barriers;
};

}  // namespace halops::indexer
