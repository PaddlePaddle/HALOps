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

#include <torch/extension.h>

#include "kernel/gemm/epilogue_op.h"

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
               const std::string& mode, const bool use_pdl,
               const EpilogueOp epilogue,
               const std::optional<torch::Tensor>& output_sf,
               const double situ_beta,
               const std::optional<double> situ_linear_beta);

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
               const std::string& mode, const EpilogueOp epilogue,
               const std::optional<torch::Tensor>& output_sf,
               const std::optional<torch::Tensor>& gate, const bool use_pdl);

void LowLatencyMmaGemm(const torch::Tensor& input, const torch::Tensor& weight,
                       const torch::Tensor& sfa, const torch::Tensor& sfb,
                       torch::Tensor& output,
                       const std::optional<torch::Tensor>& input2,
                       const std::optional<torch::Tensor>& weight2,
                       const std::optional<torch::Tensor>& sfa2,
                       const std::optional<torch::Tensor>& sfb2,
                       const std::optional<torch::Tensor>& output2,
                       const int sm_count, const bool use_pdl,
                       const bool use_sigmoid);

std::vector<torch::Tensor> GroupWiseQuant(
    const torch::Tensor& input, const std::string& mode, const bool use_swizzle,
    const bool use_pdl, const std::optional<torch::Tensor>& output_quant,
    const std::optional<torch::Tensor>& output_sf);

std::vector<torch::Tensor> RmsNormQuant(
    const torch::Tensor& input, const torch::Tensor& weight, const double eps,
    const bool use_pdl, const std::optional<torch::Tensor>& output_quant,
    const std::optional<torch::Tensor>& output_sf,
    const std::optional<torch::Tensor>& output_bf16);

std::vector<torch::Tensor> VhaPostmixQuant(
    const torch::Tensor& o, const torch::Tensor& mix,
    const torch::Tensor& freqs, const torch::Tensor& positions,
    const int64_t groups, const bool group_major, const bool use_pdl,
    const std::optional<torch::Tensor>& output_quant,
    const std::optional<torch::Tensor>& output_sf);

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

void LowLatencyMoE(
    const torch::Tensor& input, const torch::Tensor& router_weight,
    const torch::Tensor& expert_up_weights,
    const torch::Tensor& expert_up_weights_scale,
    const torch::Tensor& expert_down_weights,
    const torch::Tensor& expert_down_weights_scale,
    const torch::Tensor& shared_up_weight,
    const torch::Tensor& shared_up_weight_scale,
    const torch::Tensor& shared_down_weight,
    const torch::Tensor& shared_down_weight_scale,
    const torch::Tensor& latent_down_weight,
    const torch::Tensor& latent_down_weight_scale,
    const torch::Tensor& tail_fc2_weight, const torch::Tensor& tail_fc2_scale,
    const torch::Tensor& tail_norm_weight, const torch::Tensor& tail_epochs,
    const torch::Tensor& tail_uc_bases, const torch::Tensor& workspace,
    const torch::Tensor& tail_out, int64_t top_k, int64_t num_router_gates,
    int64_t tail_rank, int64_t tail_mc_base, int64_t tail_grid_cap,
    const std::string& scoring_func, const std::string& activation,
    bool renormalize, double situ_beta, double situ_linear_beta,
    double tail_eps);

int64_t MoETailRegionBytes(bool wide);

int64_t GetWorkspaceSize(const std::optional<torch::Tensor>& input);

int64_t RouterTileM(int64_t num_tokens);

}  // namespace halops

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("write_cache_kv_encoder", &halops::WriteCacheKVEncoder);
    m.def("write_cache_kv_decoder", &halops::WriteCacheKVDecoder);
    m.def("get_qk_token", &halops::GetQKToken);
    m.def("encoder_attention", &halops::EncoderAttention);
    m.def("decoder_attention", &halops::DecoderAttention);

    py::enum_<halops::EpilogueOp>(m, "EpilogueOp")
        .value("Identity", halops::EpilogueOp::Identity)
        .value("Quant", halops::EpilogueOp::Quant)
        .value("GateMulQuant", halops::EpilogueOp::GateMulQuant)
        .value("Sigmoid", halops::EpilogueOp::Sigmoid)
        .value("SituQuant", halops::EpilogueOp::SituQuant)
        .value("SigmoidGateMulQuant", halops::EpilogueOp::SigmoidGateMulQuant);
    m.def("dense_gemm", &halops::DenseGemm, py::arg("input"), py::arg("weight"),
          py::arg("sfa"), py::arg("sfb"), py::arg("output"), py::arg("bias"),
          py::arg("alpha"), py::arg("sm_count"), py::arg("mode"),
          py::arg("use_pdl") = true,
          py::arg("epilogue") = halops::EpilogueOp::Identity,
          py::arg("output_sf") = py::none(), py::arg("situ_beta"),
          py::arg("situ_linear_beta"));
    m.def("group_masked_gemm", &halops::GroupMaskedGemm);
    m.def("group_contiguous_gemm", &halops::GroupContiguousGemm);
    m.def("batch_gemm", &halops::BatchGemm, py::arg("input"), py::arg("weight"),
          py::arg("sfa"), py::arg("sfb"), py::arg("output"), py::arg("bias"),
          py::arg("alpha"), py::arg("sm_count"), py::arg("mode"),
          py::arg("epilogue") = halops::EpilogueOp::Identity,
          py::arg("output_sf") = py::none(), py::arg("gate") = py::none(),
          py::arg("use_pdl") = true);
    m.def("low_latency_mma_gemm", &halops::LowLatencyMmaGemm, py::arg("input"),
          py::arg("weight"), py::arg("sfa"), py::arg("sfb"), py::arg("output"),
          py::arg("input2"), py::arg("weight2"), py::arg("sfa2"),
          py::arg("sfb2"), py::arg("output2"), py::arg("sm_count"),
          py::arg("use_pdl") = true, py::arg("use_sigmoid") = false);
    m.def("group_wise_quant", &halops::GroupWiseQuant, py::arg("input"),
          py::arg("mode"), py::arg("use_swizzle") = true,
          py::arg("use_pdl") = true, py::arg("output_quant") = py::none(),
          py::arg("output_sf") = py::none());
    m.def("rms_norm_quant", &halops::RmsNormQuant, py::arg("input"),
          py::arg("weight"), py::arg("eps"), py::arg("use_pdl") = true,
          py::arg("output_quant") = py::none(),
          py::arg("output_sf") = py::none(),
          py::arg("output_bf16") = py::none());
    m.def("prefill_permute_to_masked_gemm",
          &halops::PrefillPermuteToMaskedGemm);
    m.def("fused_swiglu_quant", &halops::FusedSwigluQuant);

    m.def("mla_workspace_bytes", &halops::MlaWorkspaceBytes);
    m.def("mla_attention_plan", &halops::MlaAttentionPlan);
    m.def("mla_attention", &halops::MlaAttention);
    m.def("combine_with_rope_quant", &halops::CombineWithRopeQuant);
    m.def("mla_output_proj", &halops::MlaOutputProj);
    m.def("vha_postmix_quant", &halops::VhaPostmixQuant, py::arg("o"),
          py::arg("mix"), py::arg("freqs"), py::arg("positions"),
          py::arg("groups"), py::arg("group_major") = true,
          py::arg("use_pdl") = true, py::arg("output_quant") = py::none(),
          py::arg("output_sf") = py::none());

    m.def("low_latency_moe", &halops::LowLatencyMoE);
    m.def("moe_tail_region_bytes", &halops::MoETailRegionBytes);
    m.def("moe_get_workspace_size", &halops::GetWorkspaceSize,
          py::arg("input") = py::none());
    m.def("moe_router_tile_m", &halops::RouterTileM, py::arg("num_tokens") = 0);
}
