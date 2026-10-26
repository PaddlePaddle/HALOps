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

#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

namespace halops {

void WriteCacheKVEncoder(
    const torch::Tensor& qkv_input, const torch::Tensor& cu_seq_q,
    const torch::Tensor& cu_seq_k, const torch::Tensor& rotary_embs,
    const torch::Tensor& seq_len_encoder, const torch::Tensor& seq_len_decoder,
    const torch::Tensor& cache_k, const torch::Tensor& cache_v,
    const torch::Tensor& block_tables, const torch::Tensor& q_input,
    const torch::Tensor& k_input, const torch::Tensor& v_input,
    const std::optional<torch::Tensor>& kv_dequant_scale,
    const std::optional<torch::Tensor>& q_norm_gamma,
    const std::optional<torch::Tensor>& k_norm_gamma, const int head_num,
    const int kv_head_num, const int head_dim, const int max_seq_len_q,
    const int max_seq_len_k, const int max_input_len,
    const std::string& cache_quant_type_str);

std::vector<torch::Tensor> WriteCacheKVDecoder(
    const torch::Tensor& qkv_input, const torch::Tensor& cu_seq_q,
    const torch::Tensor& seq_len_encoder, const torch::Tensor& seq_len_decoder,
    const torch::Tensor& rotary_embs, const torch::Tensor& cache_k,
    const torch::Tensor& cache_v, const torch::Tensor& block_tables,
    const std::optional<torch::Tensor>& kv_dequant_scale,
    const std::optional<torch::Tensor>& q_norm_gamma,
    const std::optional<torch::Tensor>& k_norm_gamma, const int head_num,
    const int kv_head_num, const int head_dim, const int max_input_len,
    const std::string& cache_quant_type_str);

std::vector<torch::Tensor> GetQKToken(const torch::Tensor& seq_lens_encoder,
                                      const torch::Tensor& seq_lens_decoder,
                                      const torch::Tensor& cu_seq_q,
                                      const torch::Tensor& cu_seq_k,
                                      const torch::Tensor& actual_cu_seq_k,
                                      const torch::Tensor& seq_lens_kv,
                                      const int kv_head_num);

void EncoderAttention(const torch::Tensor& q_input,
                      const torch::Tensor& k_input,
                      const torch::Tensor& v_input,
                      const torch::Tensor& cu_seq_q,
                      const torch::Tensor& cu_seq_k,
                      const torch::Tensor& attn_out,
                      const std::optional<torch::Tensor>& mask);

void DecoderAttention(
    const torch::Tensor& q_input, const torch::Tensor& cu_seq_q,
    const torch::Tensor& seq_lens_kv, const torch::Tensor& seq_len_encoder,
    const torch::Tensor& seq_len_decoder, const torch::Tensor& cache_k,
    const torch::Tensor& cache_v, const torch::Tensor& block_tables,
    const torch::Tensor& attn_out,
    const std::optional<torch::Tensor>& q_dequant_scale,
    const std::optional<torch::Tensor>& kv_dequant_scale, const int head_num,
    const int kv_head_num, const int head_dim, const int max_seq_k,
    const int mtp_steps, const std::string& cache_quant_type_str);

void DenseGemm(const torch::Tensor& input, const torch::Tensor& weight,
               const torch::Tensor& sfa, const torch::Tensor& sfb,
               torch::Tensor& output, const std::optional<torch::Tensor>& bias,
               const std::optional<torch::Tensor>& alpha, const int sm_count,
               const std::string& mode);

void GroupMaskedGemm(const torch::Tensor& input, const torch::Tensor& weight,
                     const torch::Tensor& sfa, const torch::Tensor& sfb,
                     const torch::Tensor& masked_m, torch::Tensor& output,
                     const std::optional<torch::Tensor>& bias,
                     const std::optional<torch::Tensor>& alpha,
                     const int max_m_per_expert, const int sm_count,
                     const std::string& mode);

void GroupContiguousGemm(const torch::Tensor& input,
                         const torch::Tensor& weight, const torch::Tensor& sfa,
                         const torch::Tensor& sfb,
                         const torch::Tensor& expert_ids, torch::Tensor& output,
                         const std::optional<torch::Tensor>& bias,
                         const std::optional<torch::Tensor>& alpha,
                         const int sm_count, const std::string& mode);

void BatchGemm(const torch::Tensor& input, const torch::Tensor& weight,
               const torch::Tensor& sfa, const torch::Tensor& sfb,
               torch::Tensor& output, const std::optional<torch::Tensor>& bias,
               const std::optional<torch::Tensor>& alpha, const int sm_count,
               const std::string& mode);

std::vector<torch::Tensor> GroupWiseQuant(const torch::Tensor& input,
                                          const std::string& mode,
                                          const bool use_swizzle);

std::vector<torch::Tensor> PrefillPermuteToMaskedGemm(
    const torch::Tensor& x, const torch::Tensor& scale,
    const torch::Tensor& topk_ids, const int num_local_experts,
    const int max_token_num);

std::vector<torch::Tensor> FusedSwigluQuant(
    const torch::Tensor& input, const torch::Tensor& token_nums_per_expert,
    const bool is_fp4);

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
                  bool reset_barriers);

torch::Tensor MlaAttentionPlan(
    const torch::Tensor& kv_indices,
    const std::optional<torch::Tensor>& topk_lengths, int64_t page_size,
    int64_t kv_run_step, const std::optional<torch::Tensor>& c128_attn_len,
    const torch::Tensor& workspace);

std::vector<int64_t> MlaWorkspaceBytes(int64_t max_tokens,
                                       const torch::Tensor& wq_a_weight,
                                       const torch::Tensor& wkv_weight,
                                       const torch::Tensor& wo_a_weight,
                                       const torch::Tensor& wo_b_weight);

void CombineWithRopeQuant(const torch::Tensor& q_or_x,
                          const torch::Tensor& positions,
                          const torch::Tensor& freqs_real,
                          const std::optional<torch::Tensor>& attn_sink,
                          const std::optional<torch::Tensor>& c128_attn_len,
                          const torch::Tensor& wo_a_weight,
                          const torch::Tensor& wo_b_weight,
                          const torch::Tensor& workspace);

void MlaOutputProj(const torch::Tensor& wo_a_weight,
                   const torch::Tensor& wo_a_sf,
                   const torch::Tensor& wo_b_weight,
                   const torch::Tensor& wo_b_sf, torch::Tensor& out,
                   const torch::Tensor& workspace);

}  // namespace halops

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("write_cache_kv_encoder", &halops::WriteCacheKVEncoder);
    m.def("write_cache_kv_decoder", &halops::WriteCacheKVDecoder);
    m.def("get_qk_token", &halops::GetQKToken);
    m.def("encoder_attention", &halops::EncoderAttention);
    m.def("decoder_attention", &halops::DecoderAttention);

    m.def("dense_gemm", &halops::DenseGemm, py::arg("input"), py::arg("weight"),
          py::arg("sfa"), py::arg("sfb"), py::arg("output"), py::arg("bias"),
          py::arg("alpha"), py::arg("sm_count"), py::arg("mode"));
    m.def("group_masked_gemm", &halops::GroupMaskedGemm);
    m.def("group_contiguous_gemm", &halops::GroupContiguousGemm);
    m.def("batch_gemm", &halops::BatchGemm);
    m.def("group_wise_quant", &halops::GroupWiseQuant, py::arg("input"),
          py::arg("mode"), py::arg("use_swizzle") = true);
    m.def("prefill_permute_to_masked_gemm",
          &halops::PrefillPermuteToMaskedGemm);
    m.def("fused_swiglu_quant", &halops::FusedSwigluQuant);

    m.def("mla_workspace_bytes", &halops::MlaWorkspaceBytes);
    m.def("mla_attention_plan", &halops::MlaAttentionPlan);
    m.def("mla_attention", &halops::MlaAttention);
    m.def("combine_with_rope_quant", &halops::CombineWithRopeQuant);
    m.def("mla_output_proj", &halops::MlaOutputProj);
}
