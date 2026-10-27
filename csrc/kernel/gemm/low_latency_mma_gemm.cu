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

#include <optional>

#include "kernel/gemm/gemm_utils.h"
#include "kernel/gemm/low_latency_mma_gemm.cuh"

namespace halops {

namespace {

struct TileConfig {
    int block_n;
    int warps_k;
};

// Fitted on the GB200 decode shapes (M <= 16).
TileConfig SelectTileConfig(int n, int k) {
    int block_n = (n == 4096 && k >= 4096) ? 32 : 16;
    if (n % block_n != 0) {
        block_n = (n % 8 == 0) ? 8 : 0;
    }

    int warps_k;
    if (k <= 1024) {
        warps_k = 4;
    } else if (k <= 2048) {
        warps_k = n >= 4096 ? 8 : 4;
    } else {
        warps_k = (n <= 1536 || n == 4096) ? 4 : 8;
    }

    const int num_k_groups = k / 128;
    while (warps_k > 1 && num_k_groups % (2 * warps_k) != 0) {
        warps_k /= 2;
    }
    return {block_n, warps_k};
}

__global__ void ll_repack_bytes_kernel(const uint8_t* __restrict__ src,
                                       uint8_t* __restrict__ dst, int rows,
                                       int cols, int64_t stride_mn,
                                       int64_t stride_k, int pitch) {
    const int64_t idx =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= static_cast<int64_t>(rows) * cols) return;
    const int r = static_cast<int>(idx / cols);
    const int c = static_cast<int>(idx % cols);
    dst[static_cast<int64_t>(r) * pitch + c] =
        src[static_cast<int64_t>(r) * stride_mn + c * stride_k];
}

void LaunchLlRepackScale(const void* src, void* dst, int rows, int cols,
                         int64_t stride_mn, int64_t stride_k, int pitch,
                         cudaStream_t stream) {
    const int64_t total = static_cast<int64_t>(rows) * cols;
    if (total <= 0) return;
    const int threads = 256;
    const int64_t blocks64 = (total + threads - 1) / threads;
    TORCH_CHECK(blocks64 <= 0x7fffffffLL, "scale repack grid too large");
    ll_repack_bytes_kernel<<<static_cast<int>(blocks64), threads, 0, stream>>>(
        static_cast<const uint8_t*>(src), static_cast<uint8_t*>(dst), rows,
        cols, stride_mn, stride_k, pitch);
}

bool G32SlabCopyable(const torch::Tensor& slab, int rows, int k) {
    const int row_bytes = ll_sfa_g32_row_bytes(k);
    if (slab.stride(1) != 1 || slab.stride(0) % 16 != 0 ||
        slab.stride(0) < row_bytes ||
        reinterpret_cast<uintptr_t>(slab.const_data_ptr()) % 16 != 0)
        return false;
    const int64_t need = slab.storage_offset() +
                         static_cast<int64_t>(rows - 1) * slab.stride(0) +
                         row_bytes;
    return need <= static_cast<int64_t>(slab.storage().nbytes());
}

struct LlSlab {
    const int32_t* ptr = nullptr;
    int pitch = 0;
    bool repacked = false;
};

LlSlab CopyableSlab(const torch::Tensor& slab, int rows, int k,
                    torch::Tensor& tmp, cudaStream_t stream) {
    if (G32SlabCopyable(slab, rows, k)) {
        return {static_cast<const int32_t*>(slab.const_data_ptr()),
                static_cast<int>(slab.stride(0)), false};
    }
    const int pitch = ll_sfa_g32_row_bytes(k);
    tmp = torch::empty({(int64_t)rows, (int64_t)pitch}, slab.options());
    LaunchLlRepackScale(slab.const_data_ptr(), tmp.mutable_data_ptr(), rows,
                        k / 32, slab.stride(0), slab.stride(1), pitch, stream);
    return {static_cast<const int32_t*>(tmp.const_data_ptr()), pitch, true};
}

struct LlShapes {
    int m = 0;
    int n = 0;
    int k = 0;
    int k1 = 0;
    int k2 = 0;
};

LlShapes ResolveLlShapes(const torch::Tensor& input,
                         const torch::Tensor& weight, const torch::Tensor& sfa,
                         const torch::Tensor& sfb, torch::Tensor& output,
                         const std::optional<torch::Tensor>& output2,
                         int n_split,
                         const std::optional<torch::Tensor>& input2,
                         const std::optional<torch::Tensor>& sfa2) {
    const bool has_out2 = output2.has_value();
    const bool dual_a = input2.has_value();
    TORCH_CHECK(!dual_a || sfa2.has_value(), "dual-A (input2) also needs sfa2");
    TORCH_CHECK(!(dual_a && has_out2),
                "dual-A concat-K and dual-output are mutually exclusive");
    TORCH_CHECK(sfa.scalar_type() == torch::kUInt8 &&
                    sfb.scalar_type() == torch::kUInt8,
                "low_latency_mma_gemm supports group-32 scales only (uint8 "
                "sfa/sfb), got sfa=",
                sfa.scalar_type(), ", sfb=", sfb.scalar_type());

    LlShapes s;
    s.m = static_cast<int>(input.size(0));
    s.k = static_cast<int>(weight.size(1));
    s.n = static_cast<int>(weight.size(0));
    TORCH_CHECK(s.m > 0 && s.n > 0 && s.k > 0,
                "M, N and K must be positive, got m=", s.m, ", n=", s.n,
                ", k=", s.k);
    TORCH_CHECK(
        s.m <= 16,
        "low_latency_mma_gemm is the M <= 16 decode kernel; got m=", s.m);

    if (dual_a) {
        s.k1 = static_cast<int>(input.size(1));
        s.k2 = static_cast<int>(input2->size(1));
        TORCH_CHECK(sfa2->scalar_type() == sfa.scalar_type(),
                    "sfa2 must have sfa's dtype, got sfa=", sfa.scalar_type(),
                    ", sfa2=", sfa2->scalar_type());
        TORCH_CHECK(s.k1 + s.k2 == s.k,
                    "dual-A input K1 + input2 K2 must equal weight K, got K1=",
                    s.k1, ", K2=", s.k2, ", weight K=", s.k);
        TORCH_CHECK(input2->size(0) == s.m, "input2 M must match input M, got ",
                    input2->size(0), " vs ", s.m);
        TORCH_CHECK(s.k1 % 128 == 0 && s.k2 % 128 == 0,
                    "group-32 dual-A K1 and K2 must be multiples of 128, got "
                    "K1=",
                    s.k1, ", K2=", s.k2);
    } else {
        TORCH_CHECK(input.size(1) == s.k,
                    "input K must match weight K, got input K=", input.size(1),
                    ", weight K=", s.k);
    }
    TORCH_CHECK(s.k % 128 == 0,
                "low_latency_mma_gemm K must be divisible by 128, got ", s.k);

    const int sfa_k = dual_a ? s.k1 : s.k;
    TORCH_CHECK(sfa.size(0) >= s.m && sfa.size(1) >= sfa_k / 32,
                "sfa must be at least [M, K/32], got [", sfa.size(0), ", ",
                sfa.size(1), "]");
    if (dual_a) {
        TORCH_CHECK(sfa2->size(0) >= s.m && sfa2->size(1) >= s.k2 / 32,
                    "sfa2 must be at least [M, K2/32], got [", sfa2->size(0),
                    ", ", sfa2->size(1), "]");
    }
    TORCH_CHECK(sfb.size(0) >= s.n && sfb.size(1) >= s.k / 32,
                "sfb must be at least [N, K/32], got [", sfb.size(0), ", ",
                sfb.size(1), "]");

    if (has_out2) {
        TORCH_CHECK(n_split > 0 && n_split < s.n, "n_split must be in (0, n)");
        TORCH_CHECK(output.size(0) == s.m && output.size(1) == n_split,
                    "output (segment 0) must be [M, n_split], got [",
                    output.size(0), ", ", output.size(1), "], expected [", s.m,
                    ", ", n_split, "]");
        TORCH_CHECK(
            output2->size(0) == s.m && output2->size(1) == s.n - n_split,
            "output2 (segment 1) must be [M, n - n_split], got [",
            output2->size(0), ", ", output2->size(1), "], expected [", s.m,
            ", ", s.n - n_split, "]");
    } else {
        TORCH_CHECK(output.size(0) == s.m && output.size(1) == s.n,
                    "output must be [M, N], got [", output.size(0), ", ",
                    output.size(1), "], expected [", s.m, ", ", s.n, "]");
    }
    return s;
}

}  // namespace

void LowLatencyMmaGemm(const torch::Tensor& input, const torch::Tensor& weight,
                       const torch::Tensor& sfa, const torch::Tensor& sfb,
                       torch::Tensor& output, const int block_n,
                       const int warps_k, const int stage_smem,
                       const int persist, const int sm_count,
                       const std::optional<torch::Tensor>& output2,
                       const int n_split,
                       const std::optional<torch::Tensor>& input2,
                       const std::optional<torch::Tensor>& sfa2,
                       const bool use_pdl) {
    const bool has_out2 = output2.has_value();
    const bool dual_a = input2.has_value();

    const at::cuda::OptionalCUDAGuard device_guard(input.device());

    const LlShapes sh = ResolveLlShapes(input, weight, sfa, sfb, output,
                                        output2, n_split, input2, sfa2);
    const int m = sh.m;
    const int n = sh.n;
    const int k = sh.k;
    const int k1 = sh.k1;
    const int k2 = sh.k2;

    const int resolved_sms = sm_count > 0 ? sm_count : GetSmCount();
    TileConfig cfg = SelectTileConfig(n, k);
    if (block_n > 0) cfg.block_n = block_n;
    if (warps_k > 0) cfg.warps_k = warps_k;
    TORCH_CHECK(cfg.block_n > 0,
                "low_latency_mma_gemm N must be divisible by 8, got N=", n);
    TORCH_CHECK(n % cfg.block_n == 0,
                "low_latency_mma_gemm N must be divisible by block_n, got N=",
                n, ", block_n=", cfg.block_n);
    TORCH_CHECK(!has_out2 || n_split % cfg.block_n == 0,
                "low_latency_mma_gemm n_split must be divisible by block_n, "
                "got n_split=",
                n_split, ", block_n=", cfg.block_n);

    LowLatencyMmaGemmParams params;
    params.ptr_a = reinterpret_cast<const uint8_t*>(input.const_data_ptr());
    params.ptr_b = reinterpret_cast<const uint8_t*>(weight.const_data_ptr());
    params.ptr_d = reinterpret_cast<__nv_bfloat16*>(
        output.mutable_data_ptr<c10::BFloat16>());
    if (has_out2) {
        params.ptr_d2 = reinterpret_cast<__nv_bfloat16*>(
            output2->data_ptr<c10::BFloat16>());
        params.n_split = n_split;
    }
    params.m = m;
    params.n = n;
    params.k = k;
    params.sfa_stride_k = 1;
    if (dual_a) {
        params.ptr_a2 =
            reinterpret_cast<const uint8_t*>(input2->const_data_ptr());
        params.k1 = k1;
    }

    const cudaStream_t stream = GetCurrentCUDAStream(input);

    const int g32_pitch = ll_sfa_g32_row_bytes(k);
    torch::Tensor sfa_h;
    torch::Tensor sfa2_h;
    torch::Tensor sfb_h;

    const bool two_slabs = dual_a && k1 % 512 == 0;
    if (dual_a && !two_slabs) {
        sfa_h = torch::empty({(int64_t)m, (int64_t)g32_pitch}, sfa.options());
        sfa_h.narrow(1, 0, k1 / 32)
            .copy_(sfa.narrow(0, 0, m).narrow(1, 0, k1 / 32));
        sfa_h.narrow(1, k1 / 32, k2 / 32)
            .copy_(sfa2->narrow(0, 0, m).narrow(1, 0, k2 / 32));
        params.ptr_sfa = static_cast<const int32_t*>(sfa_h.const_data_ptr());
        params.sfa_stride_mn = g32_pitch;
        params.ptr_sfa2 = nullptr;
        params.sfa2_stride_mn = 0;
    } else {
        const LlSlab a =
            CopyableSlab(sfa, m, two_slabs ? k1 : k, sfa_h, stream);
        params.ptr_sfa = a.ptr;
        params.sfa_stride_mn = a.pitch;
        if (two_slabs) {
            const LlSlab a2 = CopyableSlab(*sfa2, m, k2, sfa2_h, stream);
            params.ptr_sfa2 = a2.ptr;
            params.sfa2_stride_mn = a2.pitch;
        }
    }

    const LlSlab b = CopyableSlab(sfb, n, k, sfb_h, stream);
    params.ptr_sfb_packed = b.ptr;
    params.sfb_stride_mn = b.pitch;
    params.sfb_stride_k = 1;
    if (b.repacked) params.pdl_force_off = 1;

#define HALOPS_LL_LAUNCH_ROWS(BLOCK_N, WARPS_N, WARPS_K, ROWS_A)            \
    LaunchLowLatencyMmaGemmAuto<(BLOCK_N), (WARPS_N), (WARPS_K), (ROWS_A)>( \
        params, stage_smem, persist == 2 ? 1 : 0, resolved_sms, use_pdl,    \
        stream)

#define HALOPS_LL_LAUNCH_TILE(BLOCK_N, WARPS_N, WARPS_K)          \
    (m <= 8 ? HALOPS_LL_LAUNCH_ROWS(BLOCK_N, WARPS_N, WARPS_K, 8) \
            : HALOPS_LL_LAUNCH_ROWS(BLOCK_N, WARPS_N, WARPS_K, 16))

#define HALOPS_LL_LAUNCH(BLOCK_N, WARPS_N, WARPS_K)                           \
    if (cfg.block_n == (BLOCK_N) && cfg.warps_k == (WARPS_K)) {               \
        TORCH_CHECK(HALOPS_LL_LAUNCH_TILE(BLOCK_N, WARPS_N, WARPS_K),         \
                    "low_latency_mma_gemm cannot serve m=", m, ", n=", n,     \
                    ", k=", k, " at block_n=", cfg.block_n,                   \
                    ", warps_k=", cfg.warps_k,                                \
                    ": a launch precondition failed (resident-A SMEM fit; "   \
                    "K/128 not divisible by 2*warps_k; a scale slab that is " \
                    "not mn-major or not 16B-aligned; or a rejected TMA "     \
                    "descriptor) -- use dense_gemm for this shape");          \
    } else

    HALOPS_LL_LAUNCH(8, 1, 1)
    HALOPS_LL_LAUNCH(8, 1, 2)
    HALOPS_LL_LAUNCH(8, 1, 4)
    HALOPS_LL_LAUNCH(8, 1, 8)
    HALOPS_LL_LAUNCH(8, 1, 16)
    HALOPS_LL_LAUNCH(16, 2, 1)
    HALOPS_LL_LAUNCH(16, 2, 2)
    HALOPS_LL_LAUNCH(16, 2, 4)
    HALOPS_LL_LAUNCH(16, 2, 8)
    HALOPS_LL_LAUNCH(32, 4, 1)
    HALOPS_LL_LAUNCH(32, 4, 2)
    HALOPS_LL_LAUNCH(32, 4, 4)
    HALOPS_LL_LAUNCH(64, 8, 1)
    HALOPS_LL_LAUNCH(64, 8, 2) {
        TORCH_CHECK(false, "unsupported (block_n, warps_k) = (", cfg.block_n,
                    ", ", cfg.warps_k,
                    "); supported: block_n in {8,16,32,64} with warps_k in "
                    "{1,2,4,8,16} / {1,2,4,8} / {1,2,4} / {1,2}");
    }
#undef HALOPS_LL_LAUNCH
#undef HALOPS_LL_LAUNCH_TILE
#undef HALOPS_LL_LAUNCH_ROWS

    C10_CUDA_CHECK(cudaGetLastError());
}

}  // namespace halops
