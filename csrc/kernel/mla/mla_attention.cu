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

#include "common/cuda_utils.h"
#include "kernel/mla/mla_attention_kernel.cuh"

namespace halops {
namespace mla {
template <class KT>
void launch_prequant_x(const MlaAttentionProblem<KT>& prob,
                       const ActAtoms<typename KT::GemmBase>& act_x);
}  // namespace mla

using namespace mla;

namespace {

using Problem = MlaAttentionProblem<TraitsN128>;

void check_attention_inputs(const MlaAttentionArguments& args) {
    const int hidden = check_attention_weights(
        args.wq_a_weight, args.wkv_weight, "mla_attention");
    TORCH_CHECK(Problem::w_rows(args.wq_b_weight) == ModelDims::QTotal &&
                    Problem::w_cols(args.wq_b_weight) == ModelDims::QLoraRank,
                "mla_attention: wq_b must be [", ModelDims::QTotal, ", ",
                ModelDims::QLoraRank, "]");
    check_sfatom_weight(args.wq_a_weight, args.wq_a_sf, "mla_attention");
    check_sfatom_weight(args.wq_b_weight, args.wq_b_sf, "mla_attention");
    check_sfatom_weight(args.wkv_weight, args.wkv_sf, "mla_attention");
    check_rope_and_sink(args.freqs_real, args.attn_sink, "mla_attention");
    const int64_t num_tokens = args.q_or_x.size(0);
    TORCH_CHECK(num_tokens > 0 && args.q_or_x.scalar_type() == at::kBFloat16 &&
                    args.q_or_x.is_contiguous() &&
                    args.q_or_x.size(1) == ModelDims::QTotal,
                "mla_attention: q_or_x must be contiguous bf16 [T > 0, ",
                ModelDims::QTotal, "]");
    const auto& x = args.hidden_states;
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.size(0) == num_tokens &&
                    x.size(1) == hidden && x.stride(1) == 1 &&
                    x.stride(0) % (16 / x.element_size()) == 0 &&
                    reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0,
                "mla_attention: hidden_states must be bf16 [T, H] with unit "
                "column stride and 16 B aligned rows");
    TORCH_CHECK(
        args.wkv_gate_weight.has_value() == args.c128_attn_len.has_value(),
        "mla_attention: wkv_gate_weight and c128_attn_len must be given "
        "together (HCA) or not at all (SWA)");
    TORCH_CHECK(
        !args.wkv_gate_weight || (args.kv_score_ring && args.kv_gate_ring_row),
        "mla_attention: wkv_gate_weight needs kv_score_ring and "
        "kv_gate_ring_row");
    TORCH_CHECK(
        !args.c128_ape || (args.c128_norm_weight && args.c128_freqs_real &&
                           args.c128_out_loc && args.c128_pool),
        "mla_attention: c128_ape needs c128_norm_weight, "
        "c128_freqs_real, c128_out_loc and c128_pool");
    TORCH_CHECK(!(args.c128_page_table && args.c128_attn_len) || args.c128_pool,
                "mla_attention: c128_page_table needs c128_pool");
}

template <class KT>
void mla_attention_launch(const MlaAttentionArguments& args) {
    const MlaAttentionProblem<KT> prob = MlaAttentionProblem<KT>::from(args);
    const int64_t bytes = prob.wsl.attention_end(prob.hidden);
    TORCH_CHECK(args.workspace.numel() >= bytes,
                "mla_attention: workspace needs ", bytes, " bytes, got ",
                args.workspace.numel());
    const auto mp = AttentionMainloop<KT>::to_underlying_arguments(args, prob);
    const auto ep = AttentionEpilogue<KT>::to_underlying_arguments(args, prob);
    const auto sched = MlaScheduler<KT>::to_underlying_arguments(prob, mp);

    static_assert(sizeof(SharedMemoryPlan<KT>) <= Sm100::SmemPerCtaMax,
                  "SharedMemoryPlan exceeds the opt-in smem");
    auto* kernel =
        mla_attention_kernel<KT, decltype(mp), decltype(ep), decltype(sched)>;
    if (mp.prequant_x) launch_prequant_x(prob, mp.act_x);
    launch_pdl(kernel,
               {prob.grid_x, KT::NumThreads, sizeof(SharedMemoryPlan<KT>)}, mp,
               ep, sched);
}
}  // namespace

void MlaAttention(const torch::Tensor& q_or_x, const torch::Tensor& kv_cache,
                  int64_t page_size, const torch::Tensor& positions,
                  const torch::Tensor& freqs_real, double sm_scale,
                  const torch::Tensor& wq_a_weight,
                  const torch::Tensor& q_norm_weight,
                  const torch::Tensor& wq_b_weight, double q_eps,
                  const torch::Tensor& wkv_weight,
                  const torch::Tensor& kv_norm_weight, double kv_eps,
                  const torch::Tensor& swa_loc,
                  const std::optional<torch::Tensor>& attn_sink,
                  const torch::Tensor& wq_a_sf, const torch::Tensor& wq_b_sf,
                  const torch::Tensor& wkv_sf, const torch::Tensor& workspace,
                  const std::optional<torch::Tensor>& wkv_gate_weight,
                  const std::optional<torch::Tensor>& kv_score_ring,
                  const std::optional<torch::Tensor>& kv_gate_ring_row,
                  const std::optional<torch::Tensor>& c128_ape,
                  const std::optional<torch::Tensor>& c128_norm_weight,
                  const std::optional<torch::Tensor>& c128_freqs_real,
                  const std::optional<torch::Tensor>& c128_out_loc,
                  const std::optional<torch::Tensor>& c128_pool,
                  int64_t c128_stride_kv_block, int64_t c128_page_block,
                  const std::optional<torch::Tensor>& c128_page_table,
                  const std::optional<torch::Tensor>& c128_attn_len,
                  int64_t draft_tokens, const torch::Tensor& hidden_states,
                  bool reset_barriers) {
    const CudaDeviceGuard device_guard(q_or_x.device());
    const MlaAttentionArguments args{
        .q_or_x = q_or_x,
        .kv_cache = kv_cache,
        .page_size = page_size,
        .positions = positions,
        .freqs_real = freqs_real,
        .sm_scale = sm_scale,
        .wq_a_weight = wq_a_weight,
        .q_norm_weight = q_norm_weight,
        .wq_b_weight = wq_b_weight,
        .q_eps = q_eps,
        .wkv_weight = wkv_weight,
        .kv_norm_weight = kv_norm_weight,
        .kv_eps = kv_eps,
        .swa_loc = swa_loc,
        .attn_sink = attn_sink,
        .wq_a_sf = wq_a_sf,
        .wq_b_sf = wq_b_sf,
        .wkv_sf = wkv_sf,
        .workspace = workspace,
        .wkv_gate_weight = wkv_gate_weight,
        .kv_score_ring = kv_score_ring,
        .kv_gate_ring_row = kv_gate_ring_row,
        .c128_ape = c128_ape,
        .c128_norm_weight = c128_norm_weight,
        .c128_freqs_real = c128_freqs_real,
        .c128_out_loc = c128_out_loc,
        .c128_pool = c128_pool,
        .c128_stride_kv_block = c128_stride_kv_block,
        .c128_page_block = c128_page_block,
        .c128_page_table = c128_page_table,
        .c128_attn_len = c128_attn_len,
        .draft_tokens = draft_tokens,
        .hidden_states = hidden_states,
        .reset_barriers = reset_barriers};
    check_attention_inputs(args);
    dispatch_tile_t(
        (int)args.q_or_x.size(0), !args.c128_attn_len.has_value(),
        [&](auto trait) { mla_attention_launch<decltype(trait)>(args); });
}

}  // namespace halops
