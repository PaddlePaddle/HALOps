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

#include <algorithm>
#include <optional>

#include "common/cuda_utils.h"
#include "kernel/attention/attention_kernel.cuh"
#include "kernel/attention/kernel_traits.cuh"
#include "kernel/attention/reduce_kernel.cuh"

namespace halops {

// Partition budget for split-KV decode: how many partitions each (batch, kv
// head) may be cut into.
template <int kBlockN, int MinKvBlocksPerPartition>
static int GetMaxNumPartitions(const int max_seq_len_kv, const int batch_size,
                               const int kv_head_num, const int sm_count) {
    constexpr int kMaxCtasPerSm = 1;

    const int max_kv_blocks = cute::ceil_div(max_seq_len_kv, kBlockN);
    const int max_partitions_by_grid =
        std::max(1, kMaxCtasPerSm * sm_count / (batch_size * kv_head_num));
    const int kv_blocks_per_partition =
        std::max(MinKvBlocksPerPartition,
                 cute::ceil_div(max_kv_blocks, max_partitions_by_grid));
    return cute::ceil_div(max_kv_blocks, kv_blocks_per_partition);
}

// Paged-KV GQA decode attention (BF16 or block-wise FP8 KV cache). The main
// kernel splits KV into partitions and AttentionReduce merges their partial
// softmax results into attn_out.
//   q_input     : [batch, kv_head_num, kBlockM, head_dim]; the GQA group and
//                 MTP tokens of a kv head are folded into kBlockM.
//   seq_lens_kv : [batch_size] total KV length per sample, used as cu_seq_k.
//   attn_out    : [q_token_num, head_num * head_dim] bf16, caller-allocated.
//   q_dequant_scale / kv_dequant_scale: required for the FP8 path.
void DecoderAttention(
    const torch::Tensor& q_input, const torch::Tensor& cu_seq_q,
    const torch::Tensor& seq_lens_kv, const torch::Tensor& seq_len_encoder,
    const torch::Tensor& seq_len_decoder, const torch::Tensor& cache_k,
    const torch::Tensor& cache_v, const torch::Tensor& block_tables,
    const torch::Tensor& attn_out,
    const std::optional<torch::Tensor>& q_dequant_scale,
    const std::optional<torch::Tensor>& kv_dequant_scale, const int head_num,
    const int kv_head_num, const int head_dim, const int max_seq_k,
    const int mtp_steps, const std::string& cache_quant_type_str) {
    const CudaDeviceGuard device_guard(q_input.device());
    using T = c10::BFloat16;
    using fp8 = c10::Float8_e4m3fn;

    AttentionParams params;
    memset(&params, 0, sizeof(AttentionParams));

    constexpr int kBlockM = 64;
    constexpr int kBlockN = 128;
    constexpr int kHeadDim = 128;
    constexpr int kMinKvBlocksPerPartition = 2;
    TORCH_CHECK(
        head_dim == kHeadDim && kv_head_num > 0 && head_num % kv_head_num == 0,
        "head_dim must be ", kHeadDim,
        " and head_num a multiple of kv_head_num > 0; got head_dim=", head_dim,
        ", head_num=", head_num, ", kv_head_num=", kv_head_num);
    const int batch_size = cu_seq_q.size(0) - 1;

    // GQA group and MTP tokens of one kv head must fit in one M tile.
    TORCH_CHECK((mtp_steps + 1) * (head_num / kv_head_num) <= kBlockM,
                "mtp_steps is too large and don't satisfy "
                "(mtp_steps + 1) * (head_num / kv_head_num) <= ",
                kBlockM);

    params.cu_seq_q = cu_seq_q.data_ptr<int>();
    params.cu_seq_k = seq_lens_kv.data_ptr<int>();
    params.seq_len_encoder = seq_len_encoder.data_ptr<int>();
    params.seq_len_decoder = seq_len_decoder.data_ptr<int>();
    params.block_tables = const_cast<int*>(block_tables.data_ptr<int>());
    params.max_blocks_per_seq = block_tables.size(1);
    params.max_num_block = cache_k.size(0);
    params.head_num = head_num;
    params.kv_head_num = kv_head_num;
    params.q_token_num = q_input.size(0);
    params.k_token_num = max_seq_k;
    params.batch_size = batch_size;

    // Softmax runs in the log2 domain.
    constexpr float kLog2e = 1.4426950408889634074;
    params.scale_softmax_log2 = 1.0f / std::sqrt((float)head_dim) * kLog2e;
    params.o_ptr = const_cast<T*>(attn_out.data_ptr<T>());

    auto stream = GetCurrentCUDAStream(q_input);
    auto device = q_input.device();

    if (cache_quant_type_str == "block_wise_fp8") {
        TORCH_CHECK(
            q_dequant_scale.has_value(),
            "q_dequant_scale must not be null when cache_quant_type_str is "
            "block_wise_fp8");
        TORCH_CHECK(
            kv_dequant_scale.has_value(),
            "kv_dequant_scale must not be null when cache_quant_type_str "
            "is block_wise_fp8");

        // Each partition covers KStageBlock kBlockN tiles.
        constexpr int KStageBlock = 8;
        constexpr int kNumKvTokensPerPartition = KStageBlock * kBlockN;
        const int max_num_partitions =
            (max_seq_k + kNumKvTokensPerPartition - 1) /
            kNumKvTokensPerPartition;
        auto opts_f32 =
            torch::TensorOptions().dtype(torch::kFloat32).device(device);
        auto opts_out =
            torch::TensorOptions().dtype(attn_out.scalar_type()).device(device);
        // One float2 (max, sum) per (row, partition).
        auto max_sums = torch::empty(
            {batch_size, max_num_partitions, kv_head_num, kBlockM, 2},
            opts_f32);
        auto partition_attn_out = torch::empty(
            {batch_size, max_num_partitions, kv_head_num, kBlockM, head_dim},
            opts_out);

        params.q_ptr = const_cast<fp8*>(q_input.data_ptr<fp8>());
        params.k_ptr = const_cast<uint8_t*>(cache_k.data_ptr<uint8_t>());
        params.v_ptr = const_cast<uint8_t*>(cache_v.data_ptr<uint8_t>());
        params.q_dequant_scale =
            const_cast<float*>(q_dequant_scale.value().data_ptr<float>());
        params.kv_dequant_scale =
            const_cast<float*>(kv_dequant_scale.value().data_ptr<float>());
        params.max_sum = reinterpret_cast<float2*>(max_sums.data_ptr<float>());
        params.partition_o_ptr =
            const_cast<T*>(partition_attn_out.data_ptr<T>());
        params.max_num_partitions = max_num_partitions;
        // n_block_infos lives in the 4-aligned region after seq_lens_kv.
        params.n_block_infos = params.cu_seq_k + (batch_size + 3) / 4 * 4;

        LaunchFp8DecoderAttention<kBlockM, kBlockN, KStageBlock>(params,
                                                                 stream);
        // The FP8 epilogue stores partition_o row-interleaved.
        AttentionReduce<kBlockM, kHeadDim, kBlockN, KStageBlock,
                        /*kRowInterleavedPartitionO=*/true,
                        /*kUseFixedPartitionWidth=*/true>(params, mtp_steps,
                                                          stream);
        return;
    }

    params.q_ptr = const_cast<T*>(q_input.data_ptr<T>());
    params.k_ptr = const_cast<T*>(cache_k.data_ptr<T>());
    params.v_ptr = const_cast<T*>(cache_v.data_ptr<T>());

    // Skip split-KV when one CTA per (batch, kv head) already fills most SMs;
    // the partition_o round trip and reduce kernel would be pure overhead.
    const int sm_count = CurrentDeviceSmCount();
    const bool enough_ctas_without_split =
        batch_size * kv_head_num * 5 >= sm_count * 4;  // >= 80% of the SMs

    if (enough_ctas_without_split ||
        max_seq_k <= kMinKvBlocksPerPartition * kBlockN) {
        params.max_num_partitions = 1;
        LaunchDecoderAttention<cutlass::bfloat16_t, kBlockM, /*SplitKV=*/false,
                               kMinKvBlocksPerPartition>(params, stream);
        return;
    }

    const int max_num_partitions =
        GetMaxNumPartitions<kBlockN, kMinKvBlocksPerPartition>(
            max_seq_k, batch_size, kv_head_num, sm_count);
    auto opts_f32 =
        torch::TensorOptions().dtype(torch::kFloat32).device(device);
    auto opts_inp =
        torch::TensorOptions().dtype(q_input.scalar_type()).device(device);
    // One float2 (max, sum) per (row, partition).
    auto max_sums = torch::empty(
        {batch_size, max_num_partitions, kv_head_num, kBlockM, 2}, opts_f32);
    auto partition_attn_out = torch::empty(
        {batch_size, max_num_partitions, kv_head_num, kBlockM, head_dim},
        opts_inp);

    params.max_sum = reinterpret_cast<float2*>(max_sums.data_ptr<float>());
    params.partition_o_ptr = const_cast<T*>(partition_attn_out.data_ptr<T>());
    params.max_num_partitions = max_num_partitions;

    LaunchDecoderAttention<cutlass::bfloat16_t, kBlockM, /*SplitKV=*/true,
                           kMinKvBlocksPerPartition>(params, stream);
    AttentionReduce<kBlockM, kHeadDim, kBlockN, kMinKvBlocksPerPartition,
                    /*kRowInterleavedPartitionO=*/false,
                    /*kUseFixedPartitionWidth=*/false>(params, mtp_steps,
                                                       stream);
}

}  // namespace halops
