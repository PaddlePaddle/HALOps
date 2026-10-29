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

#include <cuda.h>
#include <torch/extension.h>

#include "moe_tma.h"

namespace halops::moe {

CUtensorMap create_up_weight_tma_desc(const void* weights_ptr,
                                      uint32_t num_experts, uint32_t n,
                                      uint32_t k, uint32_t row_box) {
    // SWIZZLE_128B up-weight descriptor over the raw gate/up layout: rows
    // `[0, n)` of an expert hold gate and rows `[n, 2n)` hold up.
    CUtensorMap tma_up_weights_desc{};

    constexpr uint32_t kRank = 2;
    uint64_t global_dim[kRank] = {
        static_cast<uint64_t>(k),
        static_cast<uint64_t>(num_experts) * 2ULL * static_cast<uint64_t>(n),
    };
    uint64_t global_strides[kRank - 1] = {
        static_cast<uint64_t>(k),
    };
    TORCH_CHECK(k % row_box == 0, "create_up_weight_tma_desc: k=", k,
                " must be a multiple of row_box=", row_box);
    uint32_t box_dim[kRank] = {
        /*k-inner*/ 128u,
        /*rows   */ row_box,
    };
    uint32_t element_strides[kRank] = {1u, 1u};

    const CUresult res = cuTensorMapEncodeTiled(
        &tma_up_weights_desc, CU_TENSOR_MAP_DATA_TYPE_UINT8, kRank,
        const_cast<void*>(weights_ptr), global_dim, global_strides, box_dim,
        element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    TORCH_CHECK(
        res == CUDA_SUCCESS,
        "cuTensorMapEncodeTiled failed for up-projection weights: CUresult=",
        static_cast<int>(res), " (num_experts=", num_experts, ", n=", n,
        ", k=", k, ")");

    return tma_up_weights_desc;
}

CUtensorMap create_activations_tma_desc(const void* activations_ptr,
                                        uint32_t batch_size_cap,
                                        uint32_t block_rows,
                                        uint32_t K_hidden) {
    CUtensorMap tma_input_desc{};

    // `[BS, K_hidden]` row-major bf16: innermost = K_hidden, outer = row.
    constexpr uint32_t kRank = 2;
    uint64_t global_dim[kRank] = {
        static_cast<uint64_t>(K_hidden),
        static_cast<uint64_t>(batch_size_cap),
    };
    uint64_t global_strides[kRank - 1] = {
        static_cast<uint64_t>(K_hidden) * 2ULL,  // bf16 = 2 B/element
    };
    uint32_t box_dim[kRank] = {
        /*k-inner*/ 64u,
        /*tokens */ block_rows,
    };

    uint32_t element_strides[kRank] = {1u, 1u};

    // SW128: the box lands as the router UMMA's K-major B operand.
    const CUresult res = cuTensorMapEncodeTiled(
        &tma_input_desc, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, kRank,
        // The Driver API takes a non-const void*; it never writes through it.
        const_cast<void*>(activations_ptr), global_dim, global_strides, box_dim,
        element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    TORCH_CHECK(res == CUDA_SUCCESS,
                "cuTensorMapEncodeTiled failed for activations: CUresult=",
                static_cast<int>(res), " (batch_size_cap=", batch_size_cap,
                ", K_hidden=", K_hidden, ")");

    return tma_input_desc;
}

CUtensorMap create_down_weight_tma_desc(const void* weights_ptr,
                                        uint32_t num_experts, uint32_t k,
                                        uint32_t n, uint32_t row_box) {
    // SWIZZLE_128B down-weight descriptor over the raw row-major `[E, k, n]`
    // fp8 tensor; the TMA hardware applies the core-matrix XOR swizzle.
    TORCH_CHECK(row_box == 128u || row_box == 256u,
                "create_down_weight_tma_desc: row_box must be 128 or 256, got ",
                row_box);
    TORCH_CHECK(k % row_box == 0, "create_down_weight_tma_desc: k=", k,
                " must be a multiple of row_box=", row_box);

    CUtensorMap tma_down_weights_desc{};

    constexpr uint32_t kRank = 2;
    uint64_t global_dim[kRank] = {
        static_cast<uint64_t>(n),
        static_cast<uint64_t>(num_experts) * static_cast<uint64_t>(k),
    };
    uint64_t global_strides[kRank - 1] = {
        static_cast<uint64_t>(n),
    };
    uint32_t box_dim[kRank] = {
        /*n-inner*/ 128u,
        /*rows   */ row_box,
    };
    uint32_t element_strides[kRank] = {1u, 1u};

    const CUresult res = cuTensorMapEncodeTiled(
        &tma_down_weights_desc, CU_TENSOR_MAP_DATA_TYPE_UINT8, kRank,
        const_cast<void*>(weights_ptr), global_dim, global_strides, box_dim,
        element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    TORCH_CHECK(res == CUDA_SUCCESS,
                "cuTensorMapEncodeTiled failed for down-projection weights: "
                "CUresult=",
                static_cast<int>(res), " (num_experts=", num_experts, ", k=", k,
                ", n=", n, ", row_box=", row_box, ")");

    return tma_down_weights_desc;
}

CUtensorMap create_latent_down_weight_tma_desc(const void* weights_ptr,
                                               uint32_t m, uint32_t k,
                                               uint32_t row_box) {
    // SWIZZLE_128B descriptor over the shared latent weight `[m, k]`,
    // innermost = k (the reduction), outer = the m rows.
    TORCH_CHECK(m % row_box == 0u, "create_latent_down_weight_tma_desc: m=", m,
                " must be a multiple of row_box=", row_box);
    TORCH_CHECK(row_box % 8u == 0u,
                "create_latent_down_weight_tma_desc: row_box=", row_box,
                " must be a whole number of 8-row swizzle atoms");
    TORCH_CHECK(k % 128u == 0u, "create_latent_down_weight_tma_desc: k=", k,
                " must be a multiple of 128");

    CUtensorMap tma_latent_down_weights_desc{};
    constexpr uint32_t kRank = 2;
    uint64_t global_dim[kRank] = {static_cast<uint64_t>(k),
                                  static_cast<uint64_t>(m)};
    uint64_t global_strides[kRank - 1] = {static_cast<uint64_t>(k)};
    uint32_t box_dim[kRank] = {128u, row_box};
    uint32_t element_strides[kRank] = {1u, 1u};

    const CUresult res = cuTensorMapEncodeTiled(
        &tma_latent_down_weights_desc, CU_TENSOR_MAP_DATA_TYPE_UINT8, kRank,
        const_cast<void*>(weights_ptr), global_dim, global_strides, box_dim,
        element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(res == CUDA_SUCCESS,
                "cuTensorMapEncodeTiled failed for latent-projection weights: "
                "CUresult=",
                static_cast<int>(res), " (m=", m, ", k=", k, ")");
    return tma_latent_down_weights_desc;
}

CUtensorMap create_down_activation_tma_desc(const void* activations_ptr,
                                            uint32_t temp_rows,
                                            uint32_t block_rows, uint32_t n) {
    // SWIZZLE_128B down-activation descriptor over the token-major fp8 rows.
    CUtensorMap tma_down_activations_desc{};

    constexpr uint32_t kRank = 2;
    uint64_t global_dim[kRank] = {
        static_cast<uint64_t>(n),
        static_cast<uint64_t>(temp_rows),
    };
    uint64_t global_strides[kRank - 1] = {
        static_cast<uint64_t>(n),
    };
    uint32_t box_dim[kRank] = {
        /*k-inner*/ 128u,
        /*rows   */ block_rows,
    };
    uint32_t element_strides[kRank] = {1u, 1u};

    const CUresult res = cuTensorMapEncodeTiled(
        &tma_down_activations_desc, CU_TENSOR_MAP_DATA_TYPE_UINT8, kRank,
        const_cast<void*>(activations_ptr), global_dim, global_strides, box_dim,
        element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    TORCH_CHECK(
        res == CUDA_SUCCESS,
        "cuTensorMapEncodeTiled failed for down-projection activations: "
        "CUresult=",
        static_cast<int>(res), " (temp_rows=", temp_rows, ", n=", n, ")");

    return tma_down_activations_desc;
}

}  // namespace halops::moe
