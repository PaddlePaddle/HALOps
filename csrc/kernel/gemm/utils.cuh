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
#include <cutlass/arch/barrier.h>
#include <cutlass/half.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <array>
#include <cstdio>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm100_tma.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/numeric/integral_constant.hpp>
#include <cute/tensor.hpp>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/util/print_error.hpp>
#include <iostream>
#include <utility>

#include "cutlass/detail/blockwise_scale_layout.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/detail/sm100_tmem_helper.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/builders/sm100_common.inl"
#include "cutlass/gemm/collective/builders/sm90_common.inl"

namespace halops {

enum class KernelMode : int32_t {
    Masked = 0,
    Contiguous = 1,
    Dense = 2,
    Batch = 3
};

struct GemmKernelConfig {
    static constexpr int KBlockM = 256;
    static constexpr int KBlockN = 128;
    static constexpr int KBlockK = 128;
    static constexpr bool IsPairUmma = true;  // 2-CTA pair-UMMA
    static constexpr int KClusterM = 2;
    static constexpr int KClusterN = 1;
    static constexpr int KPrologueStages = 2;
    static constexpr int KEpilogueStages = 2;
    static constexpr int KBiasStages = 2;
    static constexpr int KTmemOverlapCols = 0;
    static constexpr bool UseTileSwizzle = false;
    static constexpr KernelMode KernelType = KernelMode::Dense;
    static constexpr bool AddBias = false;
    static constexpr bool SwapAB = false;
    static constexpr bool KIsFP4 = false;
    static constexpr bool HasAlpha = false;
    // Scale-factor vector size: 16 = NVFP4 (UE4M3), 32 = MXFP4 (UE8M0). Fixes
    // the MMA and SF atom shapes, so it is a compile-time config.
    static constexpr int KSfVecSize = 16;
    static constexpr int KNumRegsEpilogue = 208;
    static constexpr bool SkipTmemFence = false;
};

static constexpr int get_num_warps(int load_c_warp_id, bool add_bias) {
    int last_warp = add_bias ? load_c_warp_id : (load_c_warp_id - 1);
    return ((last_warp + 4) / 4) * 4;
}

template <class GmemStrideType, class Element, class EpilogueTile_MN>
constexpr auto sm100_get_epilogue_smem_swizzle_layout_atom() {
    // Get the max contiguous tile usable by TMA
    [[maybe_unused]] auto tma_tile =
        cute::transform(EpilogueTile_MN{}, [](auto const& epi_tile) {
            // assumes get<0>(epi_tile) is coalesced and unit stride
            return cute::size<0>(cute::coalesce(cute::right_inverse(
                cute::make_layout(cute::get<0>(epi_tile)))));
        });

    // ColMajor C/D (M-major)
    if constexpr (cutlass::detail::is_major<0>(GmemStrideType{})) {
        return cutlass::gemm::collective::detail::sm100_smem_selector<
            cute::UMMA::Major::MN, Element, decltype(cute::get<0>(tma_tile)),
            decltype(cute::get<1>(tma_tile))>();
    }
    // RowMajor C/D (N-major)
    else if constexpr (cutlass::detail::is_major<1>(GmemStrideType{})) {
        return cutlass::gemm::collective::detail::sm100_smem_selector<
            cute::UMMA::Major::K, Element, decltype(cute::get<0>(tma_tile)),
            decltype(cute::get<1>(tma_tile))>();
    } else {
        static_assert(cutlass::detail::dependent_false<GmemStrideType>,
                      "Unsupported gmem layout.");
    }
}
}  // namespace halops
