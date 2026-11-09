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

#include "common/pdl.cuh"
#include "indexer_params.cuh"
#include "indexer_prepare_kernel.cuh"
#include "indexer_small_kernel.cuh"

namespace halops::indexer {

template <class Kernel, class Parameters>
void launch_indexer_kernel(Kernel kernel, const IndexerHostParams& host,
                           const Parameters& params, int threads,
                           size_t shared_memory_bytes) {
    if (shared_memory_bytes > IndexerWarpTraits::DefaultSharedMemoryLimit)
        HALOPS_CUDA_CHECK(cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
            shared_memory_bytes));
    const auto stream = GetCurrentCUDAStream(host.x);
    HALOPS_CUDA_CHECK(LaunchKernelWithPdl(kernel, dim3(params.grid_blocks),
                                          dim3(threads), shared_memory_bytes,
                                          stream, host.use_pdl, params));
}

template <class KernelTraits>
void indexer_prepare_launch(const IndexerHostParams& host, int sm_count) {
    torch::Tensor workspace;
    const auto params =
        make_indexer_device_params<KernelTraits>(host, sm_count, workspace);

    const auto kernel_kind = select_indexer_kernel<KernelTraits>(params);
    const size_t shared_memory_bytes =
        sizeof(IndexerSharedStorage<KernelTraits>);
    if constexpr (KernelTraits::TokenTile == 16) {
        if (kernel_kind == IndexerKernelKind::ThreeWaySmallBatch) {
            auto* kernel =
                indexer_small_batch_kernel<KernelTraits, decltype(params), 8>;
            if (query_epilogue_lanes(params.num_tokens, true) == 16)
                kernel = indexer_small_batch_kernel<KernelTraits,
                                                    decltype(params), 16>;
            launch_indexer_kernel(kernel, host, params,
                                  KernelTraits::NumThreads,
                                  shared_memory_bytes);
            return;
        }
    }
    auto* kernel = indexer_prepare_kernel<KernelTraits, decltype(params)>;
    int threads = KernelTraits::NumThreads;
    if constexpr (KernelTraits::TokenTile <= 64) {
        if constexpr (KernelTraits::TokenTile == 16) {
            if (kernel_kind == IndexerKernelKind::SingleToken) {
                kernel = indexer_prepare_kernel<KernelTraits, decltype(params),
                                                true>;
                threads = KernelTraits::SingleTokenThreads;
            }
        }
        if (kernel_kind == IndexerKernelKind::LocalQuery) {
            kernel = indexer_prepare_kernel<KernelTraits, decltype(params),
                                            false, true>;
        }
    }
    launch_indexer_kernel(kernel, host, params, threads, shared_memory_bytes);
}

template <bool IsMxSf, bool IsFp4, bool NeoX, bool TwoPassQuery = false>
void dispatch_indexer_format(const IndexerHostParams& host, int sm_count,
                             const IndexerLaunchConfig& config) {
    if constexpr (!TwoPassQuery) {
        if (config.two_pass_query) {
            dispatch_indexer_format<IsMxSf, IsFp4, NeoX, true>(host, sm_count,
                                                               config);
            return;
        }
    }
    if (config.token_tile == 16)
        indexer_prepare_launch<
            IndexerTraitsFor<16, IsMxSf, IsFp4, false, false, NeoX>>(host,
                                                                     sm_count);
    else if (config.token_tile == 32)
        indexer_prepare_launch<
            IndexerTraitsFor<32, IsMxSf, IsFp4, false, false, NeoX>>(host,
                                                                     sm_count);
    else if (config.eight_warp_query)
        indexer_prepare_launch<
            IndexerTraitsFor<128, IsMxSf, IsFp4, true, true, NeoX>>(host,
                                                                    sm_count);
    else {
        if constexpr (IsMxSf && (IsFp4 || !TwoPassQuery)) {
            if (!config.shared_rope) {
                const bool rope_aligned_16 =
                    (reinterpret_cast<uintptr_t>(
                         host.rope_cos_sin_cache.data_ptr()) &
                     15) == 0;
                if (rope_aligned_16)
                    indexer_prepare_launch<
                        IndexerTraitsFor<128, IsMxSf, IsFp4, false, false, NeoX,
                                         1, true, TwoPassQuery>>(host,
                                                                 sm_count);
                else
                    indexer_prepare_launch<
                        IndexerTraitsFor<128, IsMxSf, IsFp4, false, false, NeoX,
                                         1, false, TwoPassQuery>>(host,
                                                                  sm_count);
                return;
            }
        }
        if (config.rope_stages == 2)
            indexer_prepare_launch<IndexerTraitsFor<
                128, IsMxSf, IsFp4, true, false, NeoX, 2, false, TwoPassQuery>>(
                host, sm_count);
        else
            indexer_prepare_launch<IndexerTraitsFor<
                128, IsMxSf, IsFp4, true, false, NeoX, 1, false, TwoPassQuery>>(
                host, sm_count);
    }
}

inline void dispatch_indexer(const IndexerHostParams& host, int sm_count) {
    const auto config = select_indexer_launch_config(
        host.x.size(0), host.is_mx_sf, host.is_fp4, sm_count);
    const auto dispatch_format = [&]<bool IsMxSf, bool IsFp4>() {
        if (host.is_neox_style)
            dispatch_indexer_format<IsMxSf, IsFp4, true>(host, sm_count,
                                                         config);
        else
            dispatch_indexer_format<IsMxSf, IsFp4, false>(host, sm_count,
                                                          config);
    };
    if (host.is_mx_sf) {
        if (host.is_fp4)
            dispatch_format.template operator()<true, true>();
        else
            dispatch_format.template operator()<true, false>();
    } else {
        dispatch_format.template operator()<false, false>();
    }
}

}  // namespace halops::indexer
