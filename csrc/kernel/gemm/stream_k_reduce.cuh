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
#include <cuda_runtime.h>
#include <cutlass/array.h>

#include <cstdint>

// Reduction pass for tail-only stream-K (see tile_scheduler.h).
//
// The first `tail_base` cluster-tiles are data-parallel and written straight
// to D. Each of the remaining `rem` tiles is split into `splits` K-chunks, each
// writing a raw fp32 partial to its own workspace slot:
//
//   partial[ ((t * splits + s) * num_cta + v) ][ cta_m ][ block_n ]
//
// (v = CTA index inside the cluster). This kernel sums the `splits` partials of
// each tail tile and casts to TypeD in place in D; cost scales with
// rem*splits only. A shared fp32 buffer with atomicAdd is slower (address
// contention serializes the epilogue stores).
template <typename TypeD, int CtaM, int BlockN, int NumCta>
__global__ void stream_k_reduce_kernel(TypeD* __restrict__ ptr_d,
                                       float const* __restrict__ ptr_partial,
                                       int tail_base, int rem, int splits,
                                       int ctm, int ctn, bool m_fast, int out_n,
                                       int64_t total_vec) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total_vec) {
        return;
    }

    // Unflatten: [t][v][cta_m][block_n], 4 columns per thread (scalar is
    // issue-bound). BlockN % 4 == 0 and slots are CtaM*BlockN floats, so the
    // float4 loads and 4x TypeD store are naturally aligned.
    constexpr int kVec = 4;
    constexpr int kColVecs = BlockN / kVec;
    const int col = static_cast<int>(idx % kColVecs) * kVec;
    int64_t rest = idx / kColVecs;
    const int row = static_cast<int>(rest % CtaM);
    rest /= CtaM;
    const int v = static_cast<int>(rest % NumCta);
    const int t = static_cast<int>(rest / NumCta);

    // Sum the `splits` partials of these elements.
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    const int64_t slot_stride = static_cast<int64_t>(CtaM) * BlockN;
    const int64_t within = static_cast<int64_t>(row) * BlockN + col;
    for (int s = 0; s < splits; ++s) {
        const int64_t slot =
            (static_cast<int64_t>(t) * splits + s) * NumCta + v;
        const float4 p = *reinterpret_cast<float4 const*>(
            ptr_partial + slot * slot_stride + within);
        acc.x += p.x;
        acc.y += p.y;
        acc.z += p.z;
        acc.w += p.w;
    }

    // Map the tail work id to cluster coords as the scheduler's linear path
    // does.
    const int base_tile = tail_base + t;
    int cluster_m, cluster_n;
    if (m_fast) {
        cluster_m = base_tile % ctm;
        cluster_n = base_tile / ctm;
    } else {
        cluster_m = base_tile / ctn;
        cluster_n = base_tile % ctn;
    }

    const int64_t out_row =
        (static_cast<int64_t>(cluster_m) * NumCta + v) * CtaM + row;
    const int64_t out_col = static_cast<int64_t>(cluster_n) * BlockN + col;
    cutlass::AlignedArray<TypeD, kVec> out;
    out[0] = static_cast<TypeD>(acc.x);
    out[1] = static_cast<TypeD>(acc.y);
    out[2] = static_cast<TypeD>(acc.z);
    out[3] = static_cast<TypeD>(acc.w);
    *reinterpret_cast<cutlass::AlignedArray<TypeD, kVec>*>(
        &ptr_d[out_row * out_n + out_col]) = out;
}

template <typename TypeD, int CtaM, int BlockN, int NumCta>
inline void launch_stream_k_reduce(TypeD* ptr_d, float const* ptr_partial,
                                   int tail_base, int rem, int splits, int ctm,
                                   int ctn, bool m_fast, int out_n,
                                   cudaStream_t stream) {
    const int64_t total_vec =
        static_cast<int64_t>(rem) * NumCta * CtaM * (BlockN / 4);
    constexpr int kBlock = 256;
    const int64_t grid = (total_vec + kBlock - 1) / kBlock;
    stream_k_reduce_kernel<TypeD, CtaM, BlockN, NumCta>
        <<<static_cast<int>(grid), kBlock, 0, stream>>>(
            ptr_d, ptr_partial, tail_base, rem, splits, ctm, ctn, m_fast, out_n,
            total_vec);
}
