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

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include "kernel/attention/attention_kernel.cuh"
#include "kernel/attention/kernel_traits.cuh"

namespace halops {

// Variable-length prefill attention (bf16, head_dim 128, GQA via
// head_num / kv_head_num).
//   q_input / attn_out : [total_q_tokens, head_num, head_dim]
//   k_input / v_input  : [total_k_tokens, kv_head_num, head_dim]
//   cu_seq_q / cu_seq_k: [batch_size + 1] int32 prefix sums; sample b owns
//                        tokens [cu_seq[b], cu_seq[b + 1]).
//   mask               : optional int32 mask; selects the CustomMask kernel.
void EncoderAttention(const torch::Tensor& q_input,
                      const torch::Tensor& k_input,
                      const torch::Tensor& v_input,
                      const torch::Tensor& cu_seq_q,
                      const torch::Tensor& cu_seq_k,
                      const torch::Tensor& attn_out,
                      const std::optional<torch::Tensor>& mask) {
    const at::cuda::OptionalCUDAGuard device_guard(q_input.device());
    using T = c10::BFloat16;
    AttentionParams params;
    memset(&params, 0, sizeof(AttentionParams));

    params.q_ptr = const_cast<T*>(q_input.data_ptr<T>());
    params.k_ptr = const_cast<T*>(k_input.data_ptr<T>());
    params.v_ptr = const_cast<T*>(v_input.data_ptr<T>());
    params.cu_seq_q = cu_seq_q.data_ptr<int>();
    params.cu_seq_k = cu_seq_k.data_ptr<int>();

    params.head_num = q_input.size(1);
    params.kv_head_num = k_input.size(1);
    params.q_token_num = q_input.size(0);
    params.k_token_num = k_input.size(0);
    params.batch_size = cu_seq_q.size(0) - 1;
    constexpr int kHeadDim = 128;
    const int head_dim = q_input.size(2);
    TORCH_CHECK(
        head_dim == kHeadDim && params.kv_head_num > 0 &&
            params.head_num % params.kv_head_num == 0,
        "head_dim must be ", kHeadDim,
        " and head_num a multiple of kv_head_num > 0; got head_dim=", head_dim,
        ", head_num=", params.head_num, ", kv_head_num=", params.kv_head_num);

    // Softmax runs in the log2 domain.
    constexpr float kLog2e = 1.4426950408889634074;
    params.scale_softmax_log2 = 1.0f / std::sqrt((float)head_dim) * kLog2e;
    params.o_ptr = const_cast<T*>(attn_out.data_ptr<T>());

    auto stream =
        at::cuda::getCurrentCUDAStream(q_input.device().index()).stream();
    if (mask.has_value()) {
        params.mask = const_cast<int*>(mask.value().data_ptr<int>());
        LaunchEncoderAttention<true, cutlass::bfloat16_t>(params, stream);
    } else {
        LaunchEncoderAttention<false, cutlass::bfloat16_t>(params, stream);
    }
}

}  // namespace halops
