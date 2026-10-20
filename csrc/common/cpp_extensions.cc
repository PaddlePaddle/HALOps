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
               const std::string& mode,
               const std::optional<torch::Tensor>& sfb_packed);

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
                                          const bool is_fp4,
                                          const bool use_swizzle);

std::vector<torch::Tensor> PrefillPermuteToMaskedGemm(
    const torch::Tensor& x, const torch::Tensor& scale,
    const torch::Tensor& topk_ids, const int num_local_experts,
    const int max_token_num);

std::vector<torch::Tensor> FusedSwigluQuant(
    const torch::Tensor& input, const torch::Tensor& token_nums_per_expert,
    const bool is_fp4);

void MegaMla(torch::Tensor q_or_x, torch::Tensor kv_cache, int64_t page_size,
             torch::Tensor kv_runs, int64_t topk, int64_t kv_run_step,
             torch::Tensor topk_lengths, torch::Tensor positions,
             torch::Tensor freqs_real, torch::Tensor wo_a_weight,
             torch::Tensor wo_a_scale_inv, torch::Tensor wo_b_weight,
             torch::Tensor out, double sm_scale, torch::Tensor wq_a_weight,
             torch::Tensor q_norm_weight, torch::Tensor wq_b_weight,
             double q_eps, bool use_phase1, torch::Tensor wkv_weight,
             torch::Tensor kv_norm_weight, double kv_eps, bool use_kv_write,
             torch::Tensor swa_loc, torch::Tensor attn_sink,
             torch::Tensor wq_a_sf_g32, torch::Tensor wq_b_sf_g32,
             torch::Tensor wkv_sf_g32, torch::Tensor wo_b_sf_g32,
             torch::Tensor wo_a_sf_g32, torch::Tensor workspace,
             torch::Tensor wkv_gate_weight, torch::Tensor kv_score_ring,
             torch::Tensor kv_gate_ring_row, torch::Tensor c128_ape,
             torch::Tensor c128_norm_weight, torch::Tensor c128_freqs_real,
             torch::Tensor c128_out_loc, torch::Tensor c128_row_out,
             torch::Tensor c128_pool, int64_t c128_stride_kv_block,
             int64_t c128_page_block, torch::Tensor c128_attn_slot0,
             torch::Tensor c128_attn_len, int64_t draft_tokens);

#ifdef MEGA_MLA_TRACE
torch::Tensor MegaMlaReadTrace();
void MegaMlaClearTrace();
#endif

void MegaMlaPlan(torch::Tensor topk_lengths, torch::Tensor c128_attn_len,
                 int64_t num_tokens, int64_t topk, torch::Tensor wo_a_weight,
                 torch::Tensor workspace);

std::vector<int64_t> MegaMlaWorkspaceBytes(int64_t num_tokens,
                                           torch::Tensor wo_a_weight,
                                           torch::Tensor wo_b_weight,
                                           torch::Tensor wq_a_weight,
                                           torch::Tensor wkv_weight);

}  // namespace halops

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("write_cache_kv_encoder", &halops::WriteCacheKVEncoder);
    m.def("write_cache_kv_decoder", &halops::WriteCacheKVDecoder);
    m.def("get_qk_token", &halops::GetQKToken);
    m.def("encoder_attention", &halops::EncoderAttention);
    m.def("decoder_attention", &halops::DecoderAttention);

    m.def("dense_gemm", &halops::DenseGemm, py::arg("input"), py::arg("weight"),
          py::arg("sfa"), py::arg("sfb"), py::arg("output"), py::arg("bias"),
          py::arg("alpha"), py::arg("sm_count"), py::arg("mode"),
          py::arg("sfb_packed") = py::none());
    m.def("group_masked_gemm", &halops::GroupMaskedGemm);
    m.def("group_contiguous_gemm", &halops::GroupContiguousGemm);
    m.def("batch_gemm", &halops::BatchGemm);
    m.def("group_wise_quant", &halops::GroupWiseQuant, py::arg("input"),
          py::arg("is_fp4") = false, py::arg("use_swizzle") = true);
    m.def("prefill_permute_to_masked_gemm",
          &halops::PrefillPermuteToMaskedGemm);
    m.def("fused_swiglu_quant", &halops::FusedSwigluQuant);

    m.def("mega_mla_workspace_bytes", &halops::MegaMlaWorkspaceBytes);
    m.def("mega_mla_plan", &halops::MegaMlaPlan);
    m.def("mega_mla", &halops::MegaMla);
#ifdef MEGA_MLA_TRACE
    m.def("mega_mla_read_trace", &halops::MegaMlaReadTrace,
          pybind11::call_guard<pybind11::gil_scoped_release>());
    m.def("mega_mla_clear_trace", &halops::MegaMlaClearTrace);
#endif
}
