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

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include <type_traits>
#include <vector>

#include "kernel/gemm/gemm_utils.h"

namespace halops {

constexpr int FUSED_BLOCK_THREADS = 256;
constexpr uintptr_t kFusedAlignment = 128;

static inline bool IsFusedAlignedTo(const void* ptr, uintptr_t alignment) {
    return (reinterpret_cast<uintptr_t>(ptr) & (alignment - 1)) == 0;
}

template <typename T, int TOP_K, bool BROADCAST_SCALE = true>
__global__ void __launch_bounds__(256, 5) PrefillPermuteToMaskedGemmKernel(
    T* __restrict__ permute_x, uint8_t* __restrict__ output_scale,
    int32_t* __restrict__ permuted_indice_map,
    int32_t* __restrict__ token_nums_per_expert, const T* __restrict__ x,
    const uint8_t* __restrict__ scale, const int64_t* __restrict__ topk_ids,
    const int num_tokens, const int hidden, const int hidden_scale,
    const int max_tokens_per_expert, const int kb_dim) {
    __shared__ int64_t smem_x_dst_base[TOP_K];
    __shared__ int64_t smem_scale_e_off[TOP_K];
    __shared__ int64_t smem_tile_base[TOP_K];
    __shared__ int smem_lane[TOP_K];

    const int tidx = threadIdx.x;
    const int x_num_vecs = hidden * static_cast<int>(sizeof(T)) / 16;

    for (int token_idx = blockIdx.x; token_idx < num_tokens;
         token_idx += gridDim.x) {
        if (tidx < TOP_K) {
            int slot = tidx;
            int64_t eidx = topk_ids[token_idx * TOP_K + slot];
            if (eidx != -1) {
                int32_t off = atomicAdd(&token_nums_per_expert[eidx], 1);
                permuted_indice_map[token_idx * TOP_K + slot] =
                    static_cast<int32_t>(eidx * max_tokens_per_expert + off);
                smem_x_dst_base[slot] = (eidx * max_tokens_per_expert + off) *
                                        static_cast<int64_t>(hidden);
                smem_scale_e_off[slot] =
                    eidx * static_cast<int64_t>(max_tokens_per_expert) * kb_dim;
                int n_tile = off / 128;
                int pos = off % 128;
                smem_tile_base[slot] =
                    static_cast<int64_t>(n_tile) * hidden_scale * 512;
                smem_lane[slot] = 4 * (pos % 32) + pos / 32;
            } else {
                smem_x_dst_base[slot] = -1;
            }
        }
        __syncthreads();

        int valid_mask = 0;
#pragma unroll
        for (int slot = 0; slot < TOP_K; slot++) {
            if (smem_x_dst_base[slot] != -1) {
                valid_mask |= (1 << slot);
            }
        }
        if (valid_mask == 0) continue;

        const uint4* src_x = reinterpret_cast<const uint4*>(
            x + static_cast<int64_t>(token_idx) * hidden);

        for (int v = tidx; v < x_num_vecs; v += FUSED_BLOCK_THREADS) {
            uint4 data = src_x[v];
#pragma unroll
            for (int slot = 0; slot < TOP_K; slot++) {
                if (valid_mask & (1 << slot)) {
                    reinterpret_cast<uint4*>(permute_x +
                                             smem_x_dst_base[slot])[v] = data;
                }
            }
        }

        // One uint32 (4 scale bytes) per op. BROADCAST_SCALE replicates one
        // source byte to all 4; otherwise 4 consecutive bytes are copied.
        const uint8_t* src_scale_base =
            scale + static_cast<int64_t>(token_idx) * hidden_scale *
                        (BROADCAST_SCALE ? 1 : 4);

        const int WRITES_PER_SLOT = 4;
        const int TOTAL_SCALE_OPS = hidden_scale;

        for (int idx = tidx; idx < TOTAL_SCALE_OPS * TOP_K;
             idx += FUSED_BLOCK_THREADS) {
            int slot = idx / TOTAL_SCALE_OPS;
            int rem = idx % TOTAL_SCALE_OPS;
            int s = rem / WRITES_PER_SLOT;
            int b = rem % WRITES_PER_SLOT;

            if (!(valid_mask & (1 << slot))) continue;

            uint32_t expanded;
            if constexpr (BROADCAST_SCALE) {
                uint8_t val = src_scale_base[rem];
                expanded = static_cast<uint32_t>(val) * 0x01010101u;
            } else {
                expanded =
                    reinterpret_cast<const uint32_t*>(src_scale_base)[rem];
            }

            int64_t s_offset = static_cast<int64_t>(s) << 11;
            int64_t block_base =
                smem_scale_e_off[slot] + smem_tile_base[slot] + s_offset;
            uint32_t* addr =
                reinterpret_cast<uint32_t*>(output_scale + block_base) +
                b * 128 + smem_lane[slot];
            asm volatile("st.global.cg.b32 [%0], %1;" ::"l"(addr),
                         "r"(expanded));
        }
        __syncthreads();  // smem is rewritten for the next token
    }
}

template <typename TorchT, typename CudaT, int TOP_K, bool BROADCAST_SCALE>
std::vector<torch::Tensor> PrefillPermuteToMaskedGemmDispatch(
    const torch::Tensor& x, const torch::Tensor& scale,
    const torch::Tensor& topk_ids, const int num_local_experts,
    const int max_token_num) {
    const int num_tokens = x.size(0);
    const int hidden = x.size(1);
    const int packed_hidden_scale = scale.size(1);
    const int topk = topk_ids.size(1);
    const int hidden_scale =
        BROADCAST_SCALE ? (packed_hidden_scale * 4) : (packed_hidden_scale / 4);
    TORCH_CHECK(hidden_scale > 0, "hidden_scale must be positive, got ",
                hidden_scale);
    const int kb_dim = hidden_scale * 4;

    const cudaStream_t stream = GetCurrentCUDAStream(x);

    // permute_x: [E, M, hidden], same dtype as x
    auto permute_x =
        torch::empty({num_local_experts * max_token_num, hidden}, x.options());

    // output_scale: [E, M, kb_dim] uint8 in SfAtom layout
    auto output_scale = torch::empty({num_local_experts, max_token_num, kb_dim},
                                     x.options().dtype(torch::kByte));

    auto permuted_indice_map =
        torch::empty({num_tokens, topk}, x.options().dtype(torch::kInt));
    auto token_nums_per_expert =
        torch::empty({num_local_experts, 1}, x.options().dtype(torch::kInt));

    // 0xFF fill makes unused permuted_indice_map entries -1.
    HALOPS_CUDA_CHECK(cudaMemsetAsync(token_nums_per_expert.data_ptr<int32_t>(),
                                      0, num_local_experts * sizeof(int32_t),
                                      stream));
    HALOPS_CUDA_CHECK(cudaMemsetAsync(permuted_indice_map.data_ptr<int32_t>(),
                                      0xFF, num_tokens * topk * sizeof(int32_t),
                                      stream));

    TORCH_CHECK(
        IsFusedAlignedTo(output_scale.data_ptr<uint8_t>(), kFusedAlignment),
        "output_scale buffer is not 128-byte aligned");

    if (num_tokens == 0) {
        return {permute_x, output_scale, permuted_indice_map,
                token_nums_per_expert};
    }

    int sm_count = GetSmCount();
    int max_blocks_per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm,
        PrefillPermuteToMaskedGemmKernel<CudaT, TOP_K, BROADCAST_SCALE>,
        FUSED_BLOCK_THREADS, 0);
    int num_blocks = min(num_tokens, max_blocks_per_sm * sm_count);

    PrefillPermuteToMaskedGemmKernel<CudaT, TOP_K, BROADCAST_SCALE>
        <<<num_blocks, FUSED_BLOCK_THREADS, 0, stream>>>(
            reinterpret_cast<CudaT*>(permute_x.mutable_data_ptr<TorchT>()),
            output_scale.data_ptr<uint8_t>(),
            permuted_indice_map.data_ptr<int32_t>(),
            token_nums_per_expert.data_ptr<int32_t>(),
            reinterpret_cast<const CudaT*>(x.mutable_data_ptr<TorchT>()),
            BROADCAST_SCALE ? reinterpret_cast<const uint8_t*>(
                                  scale.mutable_data_ptr<int32_t>())
                            : scale.data_ptr<uint8_t>(),
            topk_ids.data_ptr<int64_t>(), num_tokens, hidden, hidden_scale,
            max_token_num, kb_dim);

    return {permute_x, output_scale, permuted_indice_map,
            token_nums_per_expert};
}

std::vector<torch::Tensor> PrefillPermuteToMaskedGemm(
    const torch::Tensor& x, const torch::Tensor& scale,
    const torch::Tensor& topk_ids, const int num_local_experts,
    const int max_token_num) {
    const CudaDeviceGuard device_guard(x.device());
    const bool broadcast_scale = (scale.scalar_type() != torch::kByte);
    TORCH_CHECK(scale.scalar_type() == torch::kInt ||
                    scale.scalar_type() == torch::kByte,
                "scale dtype must be int32 or uint8, but got ",
                scale.scalar_type());
    TORCH_CHECK(max_token_num >= 128, "max_token_num must be >= 128, but got ",
                max_token_num);
    TORCH_CHECK(max_token_num % 128 == 0,
                "max_token_num must be divisible by 128, but got ",
                max_token_num);

    const int topk = topk_ids.size(1);

#define DISPATCH_TOPK(TORCH_T, CUDA_T, TOPK_VAL, BS)                         \
    case TOPK_VAL:                                                           \
        return PrefillPermuteToMaskedGemmDispatch<TORCH_T, CUDA_T, TOPK_VAL, \
                                                  BS>(                       \
            x, scale, topk_ids, num_local_experts, max_token_num);

    auto dispatch_dtype = [&](auto bs_const) {
        constexpr bool BS = decltype(bs_const)::value;
        switch (x.scalar_type()) {
            case torch::kFloat8_e4m3fn: {
                using TorchFP8 = c10::Float8_e4m3fn;
                switch (topk) {
                    DISPATCH_TOPK(TorchFP8, __nv_fp8_e4m3, 4, BS)
                    DISPATCH_TOPK(TorchFP8, __nv_fp8_e4m3, 8, BS)
                    default:
                        TORCH_CHECK(
                            false,
                            "Unsupported topk value for FP8, must be 4 or 8");
                }
            }
            case torch::kBFloat16: {
                using TorchBF16 = c10::BFloat16;
                switch (topk) {
                    DISPATCH_TOPK(TorchBF16, __nv_bfloat16, 4, BS)
                    DISPATCH_TOPK(TorchBF16, __nv_bfloat16, 6, BS)
                    DISPATCH_TOPK(TorchBF16, __nv_bfloat16, 8, BS)
                    default:
                        TORCH_CHECK(false,
                                    "Unsupported topk value for BF16, must be "
                                    "4, 6, or 8");
                }
            }
            default:
                TORCH_CHECK(
                    false,
                    "Unsupported x dtype, must be float8_e4m3fn or bfloat16");
        }
    };

    if (broadcast_scale) {
        return dispatch_dtype(std::true_type{});
    } else {
        return dispatch_dtype(std::false_type{});
    }

#undef DISPATCH_TOPK
}

}  // namespace halops
