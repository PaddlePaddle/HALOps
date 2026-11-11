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

#include "indexer_dispatch.cuh"

namespace halops {

void IndexerPrepare(const torch::Tensor& x, const torch::Tensor& q_lora,
                    const torch::Tensor& wk_w, const torch::Tensor& wq_b,
                    const torch::Tensor& wq_b_sf,
                    const torch::Tensor& k_norm_weight,
                    const std::optional<torch::Tensor>& k_norm_bias,
                    double k_norm_eps, const torch::Tensor& rope_cos_sin_cache,
                    const torch::Tensor& positions,
                    const torch::Tensor& paged_kv_cache_with_scale,
                    const torch::Tensor& kv_cache_loc,
                    const torch::Tensor& out_q,
                    const std::optional<torch::Tensor>& out_q_sf,
                    const torch::Tensor& out_weights, double weight_scale,
                    bool is_neox_style, bool apply_hadamard, bool use_pdl) {
    // The output buffers select the Q/K format on the host side.
    const indexer::IndexerHostParams host{x,
                                          q_lora,
                                          wk_w,
                                          wq_b,
                                          wq_b_sf,
                                          k_norm_weight,
                                          k_norm_bias,
                                          k_norm_eps,
                                          rope_cos_sin_cache,
                                          positions,
                                          paged_kv_cache_with_scale,
                                          kv_cache_loc,
                                          out_q,
                                          out_q_sf,
                                          out_weights,
                                          weight_scale,
                                          is_neox_style,
                                          apply_hadamard,
                                          use_pdl};
    const CudaDeviceGuard device_guard(host.x.device());
    indexer::dispatch_indexer(host, indexer::indexer_sm_count());
}

}  // namespace halops
