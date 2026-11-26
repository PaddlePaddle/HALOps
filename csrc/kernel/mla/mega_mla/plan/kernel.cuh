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

#pragma once

#include "kernel/mla/mega_mla/plan/config.cuh"

namespace halops::mla {

template <class C, bool TakeMax>
__device__ __forceinline__ int plan_blk_reduce(int value, int* smem) {
    if constexpr (TakeMax)
        value = __reduce_max_sync(~0u, value);
    else
        value = __reduce_add_sync(~0u, value);
    if (KernelBase::lane_idx() == 0) smem[KernelBase::warp_idx()] = value;
    __syncthreads();
    int result = smem[0];
    CUTE_UNROLL
    for (int i = 1; i < C::kWarps; ++i)
        result = TakeMax ? max(result, smem[i]) : result + smem[i];
    __syncthreads();
    return result;
}

__device__ __forceinline__ CtaWork segment_work(int token, int nb, int ns,
                                                int slot0, int k) {
    return {token, (int)((int64_t)k * nb / ns),
            (int)((int64_t)(k + 1) * nb / ns), ns > 1 ? slot0 + k : -1};
}

template <class C>
__device__ bool plan_kv_split(const MlaPlanArgs& args, int max_nb, int* s_red,
                              int& seg_len) {
    seg_len = max_nb;
    if (args.kv_splits <= 1) return false;
    int lo = max(1, (max_nb + args.kv_splits - 1) / args.kv_splits),
        hi = max(lo, max_nb);
    while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        int segs = 0;
        for (int token = threadIdx.x; token < args.num_tokens;
             token += C::kThreads) {
            const int nb = args.tok[token].nb;
            segs += (nb >= args.split_min_nb) ? (nb + mid - 1) / mid : 1;
        }
        if (plan_blk_reduce<C, false>(segs, s_red) <=
            args.grid_x - args.kvg_ctas)
            hi = mid;
        else
            lo = mid + 1;
    }
    seg_len = lo;
    int cut = 0;
    for (int token = threadIdx.x; token < args.num_tokens;
         token += C::kThreads) {
        const int nb = args.tok[token].nb;
        cut += (nb >= args.split_min_nb && nb > seg_len) ? 1 : 0;
    }
    const int cut_tokens = plan_blk_reduce<C, false>(cut, s_red);
    return cut_tokens > 0 && (max_nb - seg_len) * args.split_gain >= cut_tokens;
}

template <class C>
__device__ void plan_split_work(const MlaPlanArgs& args, int seg_len) {
    __shared__ int s_total;
    const int tid = threadIdx.x;
    for (int token = tid; token < args.num_tokens; token += C::kThreads) {
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
    for (int token = tid; token < args.num_tokens; token += C::kThreads) {
        const int nb = args.tok[token].nb, ns = args.tok[token].nseg,
                  slot0 = args.tok[token].slot0;
        for (int k = 0; k < ns; ++k)
            args.work[slot0 + k] = segment_work(token, nb, ns, slot0, k);
    }
    for (int i = s_total + tid; i < args.grid_x; i += C::kThreads)
        args.work[i] = {0, 0, 0, -1};
}

template <class C>
__device__ void plan_wave_work(const MlaPlanArgs& args, int token_iters) {
    __shared__ int s_tail;
    const int tid = threadIdx.x;
    for (int token = tid; token < args.num_tokens; token += C::kThreads) {
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
    for (int i = tid; i < n_work; i += C::kThreads) {
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

template <class C>
__global__ void __launch_bounds__(C::kThreads)
    mla_plan_kv_runs_kernel(const __grid_constant__ MlaPlanKvRunsArgs args) {
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

template <class C>
__global__ void __launch_bounds__(C::kThreads)
    mla_plan_attention_kernel(const __grid_constant__ MlaPlanArgs args) {
    __shared__ int s_red[C::kWarps];
    pdl_trigger();
    pdl_wait();
    for (int i = threadIdx.x; i < args.pbar_words; i += C::kThreads)
        args.partial_bar[i] = 0;

    int max_nb = 0;
    for (int token = threadIdx.x; token < args.num_tokens;
         token += C::kThreads) {
        const int tl = args.kv_runs[token].w;
        const int swa =
            max((tl + AttnConfig::KvBlock - 1) / AttnConfig::KvBlock, 1);
        const int c128 = args.c128_attn_len ? args.c128_attn_len[token] : 0;
        const int nb =
            swa + (c128 + AttnConfig::KvBlock - 1) / AttnConfig::KvBlock;
        args.tok[token].nb = nb;
        max_nb = max(max_nb, nb);
    }
    max_nb = plan_blk_reduce<C, true>(max_nb, s_red);

    int seg_len;
    if (plan_kv_split<C>(args, max_nb, s_red, seg_len)) {
        plan_split_work<C>(args, seg_len);
        if (threadIdx.x == 0) args.hdr->iters = 1;
    } else {
        const int token_iters = work_rounds(args.num_tokens, args.grid_x);
        plan_wave_work<C>(args, token_iters);
        if (threadIdx.x == 0) args.hdr->iters = token_iters;
    }
}

}  // namespace halops::mla
