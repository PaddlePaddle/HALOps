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

#include "kernel/mla/common/arch.cuh"
#include "kernel/mla/common/base_traits.cuh"
#include "kernel/mla/common/model.cuh"

namespace halops::mla {
struct WorkspaceLayout {
    static constexpr int GridBarrierBytes = 128;
    static constexpr int RendezvousSlotBytes = 128;
    static constexpr int BlockAlignBytes = 128;
    static constexpr int OutputProjAlignBytes = Sm100::Sw128AtomBytes;
    static constexpr int TokenPadRows = SfLayout::MnPerTile;
    static constexpr int SfPadRows = TokenPadRows / KernelBase::Tile::Smallest;
};

template <class Dims>
struct CommonC128Layout {
    static constexpr int DimsPerLane = 4;
    static constexpr int DimsPerWarp = 2 * DimsPerLane;
    static constexpr int RowGroups = KernelBase::WarpSize / 2;
    static constexpr int SliceDim = KernelBase::Roles::VecWarps * DimsPerWarp;
    static constexpr int Slices = Dims::DimV / SliceDim;
    static constexpr int FlagSlicesDone = 0;
    static constexpr int FlagRowWritten = 1;
    static constexpr int FlagsPerToken = 2;
};

using C128Layout = CommonC128Layout<ModelDims>;

struct OutputProjLayout {
    static constexpr int ClusterSize = 4;
    static constexpr int RowTile = 128;
    static constexpr int MaxTokenGroups = 2;
    static constexpr int MaxTokens = 640;
};

struct OutputProjShape {
    int hidden;
    int olora;
    int group_k;
    int group_rows;

    CUTE_HOST_DEVICE int clusters() const {
        return hidden / OutputProjLayout::RowTile;
    }
    CUTE_HOST_DEVICE int grid() const {
        return clusters() * OutputProjLayout::ClusterSize;
    }
    CUTE_HOST_DEVICE int wo_a_tiles() const {
        return olora / OutputProjLayout::RowTile;
    }
    CUTE_HOST_DEVICE int n_flags() const {
        return OutputProjLayout::MaxTokenGroups * wo_a_tiles();
    }
};

struct QuantX {
    static constexpr int PerThread = 8;
    static constexpr int ThreadsPerGrp = SfLayout::Gran / PerThread;
    static constexpr int GrpPerPass =
        KernelBase::Roles::VecWarps * KernelBase::WarpSize / ThreadsPerGrp;

    CUTE_HOST_DEVICE static bool active(int cta, int items) {
        return cta * GrpPerPass < items;
    }
    CUTE_HOST_DEVICE static int ctas(int items, int grid) {
        const int need = (items + GrpPerPass - 1) / GrpPerPass;
        return need < grid ? need : grid;
    }
};
}  // namespace halops::mla
