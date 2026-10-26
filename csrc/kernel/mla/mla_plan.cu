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

#include <c10/cuda/CUDAGuard.h>

#include "kernel/mla/mla_output_proj_plan.cuh"
#include "kernel/mla/mla_params.cuh"

namespace halops {
using namespace mla;

namespace {

constexpr int kPlanThreads = 256;
constexpr int kPlanWarps = kPlanThreads / KernelBase::WarpSize;

struct KvRunsArgs {
    const int* __restrict__ kv_indices;
    const int* __restrict__ topk_length;
    int64_t row_stride;
    int topk, step, page_shift;
    int4* __restrict__ kv_runs;
};

struct PlanArgs {
    const int4* __restrict__ kv_runs;
    const int* __restrict__ c128_attn_len;
    int num_tokens, grid_x;
    int kv_splits;
    int split_min_nb, split_gain, tail_cap;
    int kvg_ctas;
    PlanHeader* __restrict__ hdr;
    CtaWork* __restrict__ work;
    TokenSplit* __restrict__ tok;
    int* __restrict__ partial_bar;
    int pbar_words;
};

template <bool TakeMax>
__device__ __forceinline__ int plan_blk_reduce(int value, int* smem) {
    if constexpr (TakeMax)
        value = __reduce_max_sync(0xffffffffu, value);
    else
        value = __reduce_add_sync(0xffffffffu, value);
    if (KernelBase::lane_idx() == 0) smem[KernelBase::warp_idx()] = value;
    __syncthreads();
    int result = smem[0];
    CUTE_UNROLL
    for (int i = 1; i < kPlanWarps; ++i)
        result = TakeMax ? max(result, smem[i]) : result + smem[i];
    __syncthreads();
    return result;
}

__device__ __forceinline__ CtaWork segment_work(int token, int nb, int ns,
                                                int slot0, int k) {
    return {token, (int)((int64_t)k * nb / ns),
            (int)((int64_t)(k + 1) * nb / ns), ns > 1 ? slot0 + k : -1};
}

__device__ bool plan_kv_split(const PlanArgs& args, int max_nb, int* s_red,
                              int& seg_len) {
    seg_len = max_nb;
    if (args.kv_splits <= 1) return false;
    int lo = max(1, (max_nb + args.kv_splits - 1) / args.kv_splits),
        hi = max(lo, max_nb);
    while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        int segs = 0;
        for (int token = threadIdx.x; token < args.num_tokens;
             token += kPlanThreads) {
            const int nb = args.tok[token].nb;
            segs += (nb >= args.split_min_nb) ? (nb + mid - 1) / mid : 1;
        }
        if (plan_blk_reduce<false>(segs, s_red) <= args.grid_x - args.kvg_ctas)
            hi = mid;
        else
            lo = mid + 1;
    }
    seg_len = lo;
    int cut = 0;
    for (int token = threadIdx.x; token < args.num_tokens;
         token += kPlanThreads) {
        const int nb = args.tok[token].nb;
        cut += (nb >= args.split_min_nb && nb > seg_len) ? 1 : 0;
    }
    const int cut_tokens = plan_blk_reduce<false>(cut, s_red);
    return cut_tokens > 0 && (max_nb - seg_len) * args.split_gain >= cut_tokens;
}

__device__ void plan_split_work(const PlanArgs& args, int seg_len) {
    __shared__ int s_total;
    const int tid = threadIdx.x;
    for (int token = tid; token < args.num_tokens; token += kPlanThreads) {
        const int nb = args.tok[token].nb;
        args.tok[token].nseg =
            nb >= args.split_min_nb ? (nb + seg_len - 1) / seg_len : 1;
    }
    __syncthreads();
    if (tid == 0) {
        int acc = 0;
        for (int token = 0; token < args.num_tokens; ++token) {
            args.tok[token].slot0 = acc;
            acc += args.tok[token].nseg;
        }
        s_total = acc;
    }
    __syncthreads();
    for (int token = tid; token < args.num_tokens; token += kPlanThreads) {
        const int nb = args.tok[token].nb, ns = args.tok[token].nseg,
                  slot0 = args.tok[token].slot0;
        for (int k = 0; k < ns; ++k)
            args.work[slot0 + k] = segment_work(token, nb, ns, slot0, k);
    }
    for (int i = s_total + tid; i < args.grid_x; i += kPlanThreads)
        args.work[i] = {0, 0, 0, -1};
}

__device__ void plan_wave_work(const PlanArgs& args, int token_iters) {
    __shared__ int s_tail;
    const int tid = threadIdx.x;
    for (int token = tid; token < args.num_tokens; token += kPlanThreads) {
        args.tok[token].nseg = 1;
        args.tok[token].slot0 = token;
    }
    __syncthreads();
    const int last = token_iters - 1, t0 = last * args.grid_x;
    if (tid == 0) {
        s_tail = 0;
        if (args.tail_cap > 1 && token_iters >= 2) {
            const int n_tail = args.num_tokens - t0;
            const int avail = args.grid_x - args.kvg_ctas;
            const int base = avail / n_tail,
                      extra = base >= args.tail_cap ? 0 : avail % n_tail;
            const int sp = min(base, args.tail_cap);
            if (sp >= 1 && (sp >= 2 || extra > 0)) {
                int slot = 0;
                for (int token = t0; token < args.num_tokens; ++token) {
                    const int want = sp + ((token - t0) < extra ? 1 : 0);
                    const int ns = args.tok[token].nb >= args.split_min_nb
                                       ? min(want, args.tok[token].nb)
                                       : 1;
                    args.tok[token].nseg = ns;
                    args.tok[token].slot0 = ns > 1 ? slot : token;
                    if (ns > 1) slot += ns;
                }
                s_tail = 1;
            }
        }
    }
    __syncthreads();
    const bool tail = s_tail != 0;
    const int n_work = token_iters * args.grid_x;
    for (int i = tid; i < n_work; i += kPlanThreads) {
        const int it = i / args.grid_x, bid = i - it * args.grid_x;
        if (tail && it == last) continue;
        const int token = bid + it * args.grid_x;
        args.work[i] = {token, 0,
                        token < args.num_tokens ? args.tok[token].nb : 0, -1};
    }
    if (tail && tid == 0) {
        int cta = 0;
        for (int token = t0; token < args.num_tokens; ++token) {
            const int nb = args.tok[token].nb, ns = args.tok[token].nseg,
                      slot0 = args.tok[token].slot0;
            for (int k = 0; k < ns; ++k, ++cta)
                args.work[t0 + cta] = segment_work(token, nb, ns, slot0, k);
        }
        for (; cta < args.grid_x; ++cta) args.work[t0 + cta] = {0, 0, 0, -1};
    }
}

__global__ void __launch_bounds__(kPlanThreads)
    mla_kv_runs_kernel(const __grid_constant__ KvRunsArgs args) {
    __shared__ int s_first_break, s_breaks;
    pdl_trigger();
    pdl_wait();
    const int token = blockIdx.x;
    const int* __restrict__ row = args.kv_indices + token * args.row_stride;
    const int rows = args.topk_length ? args.topk_length[token] : args.topk;
    if (threadIdx.x == 0) {
        s_first_break = rows;
        s_breaks = 0;
    }
    __syncthreads();
    for (int col = threadIdx.x + 1; col < min(rows, args.topk);
         col += blockDim.x) {
        if (kv_run_breaks(row[col - 1], row[col], args.step, args.page_shift)) {
            atomicMin(&s_first_break, col);
            atomicAdd(&s_breaks, 1);
        }
    }
    __syncthreads();
    if (threadIdx.x != 0) return;
    const bool cut = s_breaks > 1;
    const int len0 = cut ? 1 : s_first_break;
    args.kv_runs[token] = ascending_kv_runs(
        row[0], len0, row[min(len0, args.topk - 1)], cut ? 1 : rows, args.step);
}

__global__ void __launch_bounds__(kPlanThreads)
    mla_attention_plan_kernel(const __grid_constant__ PlanArgs args) {
    __shared__ int s_red[kPlanWarps];
    pdl_trigger();
    pdl_wait();
    for (int i = threadIdx.x; i < args.pbar_words; i += kPlanThreads)
        args.partial_bar[i] = 0;

    int max_nb = 0;
    for (int token = threadIdx.x; token < args.num_tokens;
         token += kPlanThreads) {
        const int tl = args.kv_runs[token].w;
        const int swa =
            max((tl + AttnConfig::KvBlock - 1) / AttnConfig::KvBlock, 1);
        const int c128 = args.c128_attn_len ? args.c128_attn_len[token] : 0;
        const int nb =
            swa + (c128 + AttnConfig::KvBlock - 1) / AttnConfig::KvBlock;
        args.tok[token].nb = nb;
        max_nb = max(max_nb, nb);
    }
    max_nb = plan_blk_reduce<true>(max_nb, s_red);

    int seg_len;
    if (plan_kv_split(args, max_nb, s_red, seg_len)) {
        plan_split_work(args, seg_len);
        if (threadIdx.x == 0) args.hdr->iters = 1;
    } else {
        const int token_iters = work_rounds(args.num_tokens, args.grid_x);
        plan_wave_work(args, token_iters);
        if (threadIdx.x == 0) args.hdr->iters = token_iters;
    }
}

void check_plan_inputs(const torch::Tensor& kv_indices,
                       const std::optional<torch::Tensor>& topk_lengths,
                       int64_t page_size, int64_t kv_run_step,
                       const std::optional<torch::Tensor>& c128_attn_len) {
    TORCH_CHECK(
        kv_indices.scalar_type() == at::kInt && kv_indices.dim() == 2 &&
            kv_indices.size(0) > 0 && kv_indices.size(1) > 0 &&
            kv_indices.stride(1) == 1,
        "mla_attention_plan: kv_indices must be int32 [T > 0, topk > 0] "
        "with unit column stride");
    const int64_t num_tokens = kv_indices.size(0);
    TORCH_CHECK((!topk_lengths || (topk_lengths->numel() == num_tokens &&
                                   topk_lengths->is_contiguous())) &&
                    (!c128_attn_len || c128_attn_len->numel() == num_tokens),
                "mla_attention_plan: T = ", num_tokens,
                " must match topk_lengths / c128_attn_len");
    TORCH_CHECK(page_size > 0 && (page_size & (page_size - 1)) == 0,
                "mla_attention_plan: page_size must be a power of two");
    TORCH_CHECK(kv_run_step == 1 || kv_run_step == -1,
                "mla_attention_plan: kv_run_step must be 1 or -1");
}

void launch_kv_runs(const torch::Tensor& kv_indices,
                    const std::optional<torch::Tensor>& topk_lengths,
                    int64_t page_size, int64_t kv_run_step, int4* kv_runs) {
    const int topk = (int)kv_indices.size(1);
    const KvRunsArgs args{.kv_indices = kv_indices.data_ptr<int>(),
                          .topk_length = optional_ptr<int>(topk_lengths),
                          .row_stride = kv_indices.stride(0),
                          .topk = topk,
                          .step = (int)kv_run_step,
                          .page_shift = page_shift_of(page_size),
                          .kv_runs = kv_runs};
    const int threads =
        std::min(cute::round_up(topk, KernelBase::WarpSize), kPlanThreads);
    launch_pdl(mla_kv_runs_kernel, {(int)kv_indices.size(0), threads}, args);
}

OutputProjShape check_workspace_weights(const torch::Tensor& wq_a_weight,
                                        const torch::Tensor& wkv_weight,
                                        const torch::Tensor& wo_a_weight,
                                        const torch::Tensor& wo_b_weight) {
    const int hidden =
        check_attention_weights(wq_a_weight, wkv_weight, "mla_workspace");
    const OutputProjShape shape =
        output_proj_shape(wo_a_weight, wo_b_weight, "mla_workspace");
    TORCH_CHECK(shape.hidden == hidden, "mla_workspace: wo_b has ",
                shape.hidden, " rows, wq_a has H = ", hidden);
    return shape;
}

}  // namespace

torch::Tensor MlaAttentionPlan(
    const torch::Tensor& kv_indices,
    const std::optional<torch::Tensor>& topk_lengths, int64_t page_size,
    int64_t kv_run_step, const std::optional<torch::Tensor>& c128_attn_len,
    const torch::Tensor& workspace) {
    const at::cuda::OptionalCUDAGuard device_guard(workspace.device());
    check_plan_inputs(kv_indices, topk_lengths, page_size, kv_run_step,
                      c128_attn_len);
    const int token_count = (int)kv_indices.size(0);
    const int grid_x = mla::grid_x();
    const int kv_splits = kv_splits_for(grid_x, token_count);
    const bool hca = c128_attn_len.has_value();
    const int n_groups = n_groups_for(token_count, !hca);

    const MlaWorkspace::Layout wsl = MlaWorkspace::layout(token_count, grid_x);
    TORCH_CHECK(workspace.numel() >= wsl.kv_split_off,
                "mla_attention_plan: workspace needs ", wsl.kv_split_off,
                " bytes, got ", workspace.numel());
    char* base = reinterpret_cast<char*>(workspace.data_ptr());
    const PlanView view =
        plan_ptrs(base, wsl.plan_off[hca], token_count, grid_x);
    launch_kv_runs(kv_indices, topk_lengths, page_size, kv_run_step,
                   const_cast<int4*>(view.kv_runs));

    const int items_kvg = kv_gate_items(n_groups, hca);
    using Split = AttnConfig::KvSplit;
    const PlanArgs args{
        .kv_runs = view.kv_runs,
        .c128_attn_len = optional_ptr<int>(c128_attn_len),
        .num_tokens = token_count,
        .grid_x = grid_x,
        .kv_splits = kv_splits,
        .split_min_nb = Split::MinBlocksToCut,
        .split_gain = Split::Gain,
        .tail_cap = Split::TailCap,
        .kvg_ctas = kv_gate_ctas(token_count, grid_x, items_kvg),
        .hdr = const_cast<PlanHeader*>(view.hdr),
        .work = const_cast<CtaWork*>(view.work),
        .tok = const_cast<TokenSplit*>(view.tok),
        .partial_bar = reinterpret_cast<int*>(base + wsl.pbar_off),
        .pbar_words = (int)((wsl.barrier_end - wsl.pbar_off) / sizeof(int))};
    launch_pdl(mla_attention_plan_kernel, {1, kPlanThreads}, args);

    constexpr int kRunFields = sizeof(int4) / sizeof(int);
    return workspace
        .narrow(0, wsl.plan_off[hca] + wsl.plan.kv_runs_off,
                (int64_t)token_count * sizeof(int4))
        .view(at::kInt)
        .view({token_count, kRunFields});
}

std::vector<int64_t> MlaWorkspaceBytes(int64_t max_tokens,
                                       const torch::Tensor& wq_a_weight,
                                       const torch::Tensor& wkv_weight,
                                       const torch::Tensor& wo_a_weight,
                                       const torch::Tensor& wo_b_weight) {
    const at::cuda::OptionalCUDAGuard device_guard(wq_a_weight.device());
    const OutputProjShape shape = check_workspace_weights(
        wq_a_weight, wkv_weight, wo_a_weight, wo_b_weight);
    const OutputProjPlan plan((int)max_tokens, shape, "mla_workspace");
    const MlaWorkspace::Layout wsl =
        MlaWorkspace::layout(plan.num_tokens, grid_x());
    return {wsl.output_proj_off(shape.hidden) + plan.bytes, wsl.barrier_end};
}
}  // namespace halops
