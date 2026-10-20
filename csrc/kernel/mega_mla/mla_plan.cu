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

#include "mla_intrinsics.cuh"
#include "mla_params.cuh"

namespace halops {

__device__ __forceinline__ int plan_blk_reduce(int v, int* s, bool take_max) {
    CUTE_UNROLL
    for (int o = 16; o; o >>= 1) {
        const int u = __shfl_down_sync(0xffffffffu, v, o);
        v = take_max ? max(v, u) : v + u;
    }
    const int w = (int)threadIdx.x >> 5, lane = (int)threadIdx.x & 31;
    if (lane == 0) s[w] = v;
    __syncthreads();
    if (threadIdx.x == 0) {
        int r = s[0];
        CUTE_UNROLL
        for (int i = 1; i < 8; ++i) r = take_max ? max(r, s[i]) : r + s[i];
        s[8] = r;
    }
    __syncthreads();
    const int out = s[8];
    __syncthreads();
    return out;
}

__global__ void __launch_bounds__(256)
    mega_mla_plan_kernel(const int* __restrict__ topk_length,
                         const int* __restrict__ c128_attn_len, int T,
                         int topk_uniform, int grid_x, int kv_splits, int n_pad,
                         int n_groups, int split_min_nb, int split_gain,
                         bool verbose, mega_mla::PlanHeader* __restrict__ hdr,
                         int* __restrict__ seg_in_group,
                         mega_mla::CtaWork* __restrict__ work,
                         mega_mla::TokenSplit* __restrict__ tok,
                         int* __restrict__ partial_bar, int pbar_words) {
    constexpr int B_TOPK = mega_mla::AttnConfig::B_TOPK;
    extern __shared__ int s_dyn[];
    int* s_nb = s_dyn;
    int* s_nseg = s_dyn + T;
    __shared__ int s_red[9];
    __shared__ int s_total, s_any;
    const int tid = (int)threadIdx.x, nthr = (int)blockDim.x;

    for (int i = tid; i < pbar_words; i += nthr) partial_bar[i] = 0;

    int my_max = 0;
    for (int t = tid; t < T; t += nthr) {
        const int tl = topk_length ? topk_length[t] : topk_uniform;
        const int swa = max((tl + B_TOPK - 1) / B_TOPK, 1);
        const int c128 = c128_attn_len ? c128_attn_len[t] : 0;
        const int nb = swa + (c128 + B_TOPK - 1) / B_TOPK;
        s_nb[t] = nb;
        my_max = max(my_max, nb);
    }
    __syncthreads();
    const int max_nb = plan_blk_reduce(my_max, s_red, true);

    int L = max_nb, cut_tokens = 0;
    bool packed = kv_splits > 1;
    if (packed) {
        int lo = 1, hi = max_nb;
        while (lo < hi) {
            const int mid = (lo + hi) >> 1;
            int s = 0;
            for (int t = tid; t < T; t += nthr)
                s += (s_nb[t] >= split_min_nb) ? (s_nb[t] + mid - 1) / mid : 1;
            if (plan_blk_reduce(s, s_red, false) <= grid_x)
                hi = mid;
            else
                lo = mid + 1;
        }
        L = lo;
        int c = 0;
        for (int t = tid; t < T; t += nthr)
            c += (s_nb[t] >= split_min_nb && s_nb[t] > L) ? 1 : 0;
        cut_tokens = plan_blk_reduce(c, s_red, false);
        packed = cut_tokens > 0 && (max_nb - L) * split_gain >= cut_tokens;
    }

    for (int t = tid; t < T; t += nthr)
        s_nseg[t] =
            (packed && s_nb[t] >= split_min_nb) ? (s_nb[t] + L - 1) / L : 1;
    __syncthreads();

    if (packed) {
        if (tid == 0) {
            int acc = 0, any = 0;
            for (int t = 0; t < T; ++t) {
                tok[t] = {s_nb[t], s_nseg[t], acc, 0};
                acc += s_nseg[t];
                any |= (s_nseg[t] > 1);
            }
            s_total = acc;
            s_any = any;
        }
    } else {
        for (int t = tid; t < T; t += nthr) tok[t] = {s_nb[t], 1, t, 0};
        if (tid == 0) {
            s_total = T;
            s_any = 0;
        }
    }
    __syncthreads();

    for (int g = tid; g < n_groups; g += nthr) {
        int s = 0;
        const int t1 = min(T, (g + 1) * n_pad);
        for (int t = g * n_pad; t < t1; ++t) s += s_nseg[t];
        seg_in_group[g] = s;
    }

    const int token_iters = (T + grid_x - 1) / grid_x;
    const int n_work = packed ? grid_x : token_iters * grid_x;
    if (packed) {
        for (int t = tid; t < T; t += nthr) {
            const int nb = s_nb[t], ns = s_nseg[t], base = tok[t].slot0;
            for (int s = 0; s < ns; ++s) {
                const int lo = (int)((int64_t)s * nb / ns);
                const int hi = (int)((int64_t)(s + 1) * nb / ns);
                work[base + s] = {t, lo, hi, ns > 1 ? base + s : -1};
            }
        }
        for (int i = s_total + tid; i < grid_x; i += nthr)
            work[i] = {0, 0, 0, -1};
    } else {
        for (int i = tid; i < n_work; i += nthr) {
            const int it = i / grid_x, bid = i - it * grid_x;
            const int token = bid + it * grid_x;
            work[i] = {token, 0, token < T ? s_nb[token] : 0, -1};
        }
    }

    if (tid == 0) {
        hdr->any_split = s_any;
        hdr->slot_step = 1;
        if (verbose)
            printf(
                "[mega_mla_plan] T=%d grid=%d kv_splits=%d min_nb=%d "
                "max_nb=%d -> L=%d cut=%d seg_total=%d (wave %.2f) "
                "any_split=%d\n",
                T, grid_x, kv_splits, split_min_nb, max_nb, L, cut_tokens,
                s_total, (double)s_total / grid_x, s_any);
    }
}

void MegaMlaPlan(torch::Tensor topk_lengths, torch::Tensor c128_attn_len,
                 int64_t num_tokens, int64_t topk, torch::Tensor wo_a_weight,
                 torch::Tensor workspace) {
    const int T = (int)num_tokens;
    TORCH_CHECK(T > 0, "mega_mla_plan: num_tokens must be positive, got ", T);
    TORCH_CHECK(workspace.scalar_type() == torch::kUInt8,
                "mega_mla_plan: workspace must be uint8");
    const int grid_x = mega_mla::grid_x_for(T);
    const int kv_splits = mega_mla::kv_splits_for(grid_x, T);
    const int n_pad = mega_mla::select_tile_t(T);
    const int n_groups = mega_mla::n_groups_for(T);

    const auto wsg = MegaMlaWorkspace<mega_mla::TraitsN128>::geom(
        T, grid_x, n_groups, olora_rounds_from_wo_a(wo_a_weight));
    TORCH_CHECK(workspace.numel() >= wsg.kv_split_off,
                "mega_mla_plan: workspace holds ", workspace.numel(),
                " bytes, the plan and barrier regions alone need ",
                wsg.kv_split_off);
    char* base = reinterpret_cast<char*>(workspace.data_ptr());
    const mega_mla::PlanView v =
        mega_mla::plan_ptrs(base, wsg.plan_off, T, grid_x, n_groups);

    auto stream = c10::cuda::getCurrentCUDAStream();
    mega_mla_plan_kernel<<<1, 256, 2 * T * sizeof(int), stream>>>(
        topk_lengths.numel() ? topk_lengths.data_ptr<int>() : nullptr,
        c128_attn_len.numel() ? c128_attn_len.data_ptr<int>() : nullptr, T,
        (int)topk, grid_x, kv_splits, n_pad, n_groups, mega_mla::split_min_nb(),
        mega_mla::split_gain(), mega_mla::plan_verbose(),
        const_cast<mega_mla::PlanHeader*>(v.hdr),
        const_cast<int*>(v.seg_in_group),
        const_cast<mega_mla::CtaWork*>(v.work),
        const_cast<mega_mla::TokenSplit*>(v.tok),
        reinterpret_cast<int*>(base + wsg.pbar_off),
        (int)(wsg.pbar_bytes / sizeof(int)));
}

}  // namespace halops
