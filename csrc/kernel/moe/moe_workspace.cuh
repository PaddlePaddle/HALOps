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

#include <cstddef>
#include <cstdint>

#include "cute/container/array.hpp"
#include "cute/pointer.hpp"
#include "kernel/moe/mma_core.cuh"
#include "kernel/moe/moe_configs.cuh"
#include "kernel/moe/moe_cta_comm.cuh"
#include "kernel/moe/moe_shared_storage.cuh"
#include "kernel/moe/moe_traits.cuh"

namespace halops::moe {

template <typename Element>
__host__ __device__ constexpr std::size_t storage_bytes(std::size_t count) {
    if constexpr (cute::is_subbyte_v<Element>) {
        constexpr auto bits = cute::sizeof_bits_v<Element>;
        return (count / 8u) * bits +
               cute::bits_to_bytes<std::size_t>((count % 8u) * bits);
    } else {
        return count * sizeof(Element);
    }
}

// Contiguous view into workspace memory, with the last dimension at unit
// stride.
template <typename Element, std::size_t Rank>
struct WorkspaceTensor {
    Element* data;
    cute::array<std::uint32_t, Rank> shape;

    __host__ __device__ constexpr void init(
        Element* ptr, cute::array<std::uint32_t, Rank> extents) {
        data = ptr;
        shape = extents;
    }

    __host__ __device__ static constexpr std::size_t bytes(
        cute::array<std::uint32_t, Rank> extents) {
        std::size_t count = 1;
        for (std::size_t dim = 0; dim < Rank; ++dim) {
            count *= extents[dim];
        }
        return storage_bytes<Element>(count);
    }

    __host__ __device__ constexpr std::size_t bytes() const {
        return bytes(shape);
    }

    template <typename... Indices>
    __host__ __device__ decltype(auto) operator()(Indices... indices) const {
        static_assert(sizeof...(Indices) == Rank,
                      "expected one index per dimension");
        const std::size_t coord[] = {static_cast<std::size_t>(indices)...};
        std::size_t offset = coord[0];
#pragma unroll
        for (std::size_t dim = 1; dim < Rank; ++dim) {
            offset = offset * shape[dim] + coord[dim];
        }
        return cute::recast_ptr<Element>(data)[offset];
    }
};

// Row-major activation with one scale per group along the last axis.
template <typename Quant, std::size_t Rank = 2>
struct QuantActivation {
    using Element = typename Quant::ElementType;
    using Scale = typename Quant::ScaleType;

    WorkspaceTensor<Element, Rank> values;
    WorkspaceTensor<Scale, Rank> scales;

    __host__ __device__ static constexpr auto value_shape(
        cute::array<std::uint32_t, Rank> shape) {
        return shape;
    }

    __host__ __device__ static constexpr auto scale_shape(
        cute::array<std::uint32_t, Rank> shape) {
        const auto width = shape[Rank - 1];
        shape[Rank - 1] =
            width / Quant::kBlockSize + (width % Quant::kBlockSize != 0);
        return shape;
    }

    __host__ __device__ constexpr void init(
        Element* data, Scale* scale_data,
        cute::array<std::uint32_t, Rank> shape) {
        values.init(data, value_shape(shape));
        scales.init(scale_data, scale_shape(shape));
    }

    __host__ __device__ constexpr std::size_t bytes() const {
        return values.bytes() + scales.bytes();
    }
};

// Latent and shared producers both stage [K chunk][schedule row + pad][16 B].
template <typename Quant>
struct ChunkedQuantActivation {
    using Element = typename Quant::ElementType;
    using Scale = typename Quant::ScaleType;

    static constexpr std::uint32_t kLboChunk = LboChunkK<Element>::value;
    static constexpr std::uint32_t kSfAtomBytes = 512u;

    WorkspaceTensor<Element, 3> values;
    WorkspaceTensor<Scale, 2> scales;

    __host__ __device__ static constexpr cute::array<std::uint32_t, 3>
    value_shape(cute::array<std::uint32_t, 2> shape) {
        const std::size_t width = shape[1];
        const std::uint32_t chunks = (width + kLboChunk - 1u) / kLboChunk;
        return {chunks, shape[0] + 1u, kLboChunk};
    }

    __host__ __device__ static constexpr cute::array<std::uint32_t, 2>
    scale_shape(cute::array<std::uint32_t, 2> shape) {
        const std::size_t width = shape[1];
        if constexpr (Quant::kMethod == QuantMethod::kBlockwiseFP8) {
            const std::uint32_t blocks =
                (width + Quant::kBlockSize - 1u) / Quant::kBlockSize;
            return {blocks, shape[0]};
        } else {
            const std::uint32_t atoms =
                cute::ceil_div(width, 4u * Quant::kBlockSize);
            return {atoms, kSfAtomBytes};
        }
    }

    __host__ __device__ constexpr void init(
        Element* data, Scale* scale_data, cute::array<std::uint32_t, 2> shape) {
        values.init(data, value_shape(shape));
        scales.init(scale_data, scale_shape(shape));
    }

    __host__ __device__ constexpr std::size_t bytes() const {
        return values.bytes() + scales.bytes();
    }

    __host__ __device__ decltype(auto) value(std::uint32_t token,
                                             std::uint32_t feature) const {
        return values(feature / kLboChunk, token, feature % kLboChunk);
    }

    __host__ __device__ Scale& scale(std::uint32_t token,
                                     std::uint32_t group) const {
        if constexpr (Quant::kMethod == QuantMethod::kBlockwiseFP8) {
            return scales(group, token);
        } else {
            return scales(group / 4u, sf_atom_offset(token, group % 4u));
        }
    }

    __device__ void clear_pad_row(std::uint32_t chunk) const {
        *reinterpret_cast<uint4*>(&values(chunk, values.shape[1] - 1u, 0u)) =
            make_uint4(0u, 0u, 0u, 0u);
    }
};

template <typename Dims>
struct MoEWorkspace {
    static constexpr std::uint32_t kMaxRoutedRows =
        Dims::kNumTokens * Dims::kMaxTopk;
    using LatentDownQuant = typename LatentDownGemmTraits<Dims>::OutputQuant;
    using RoutedUpQuant = typename RoutedUpGemmTraits<Dims>::OutputQuant;
    using SharedUpQuant = typename SharedUpGemmTraits<Dims>::OutputQuant;

    using CtaComm = MoECtaComm<Dims::KernelConfig::kSpecMaxGrid, kMaxRoutedRows,
                               MoERuntime::kMaxCtasPerExpert>;

    CtaComm* cta_comm;
    MoERoutingTables<Dims>* router_tables;
    WorkspaceTensor<typename RouterGemmTraits<Dims>::LogitType, 3>
        router_logits;
    ChunkedQuantActivation<LatentDownQuant> latent_down_out;
    ChunkedQuantActivation<SharedUpQuant> shared_up_out;
    WorkspaceTensor<DataType::ElementOut, 2> shared_down_out;
    QuantActivation<RoutedUpQuant> routed_up_out;
    WorkspaceTensor<DataType::ElementOut, 2> routed_down_out;

    std::size_t bind(void* buffer, MoEProblemSize const& problem) {
        constexpr auto tokens = Dims::kNumTokens;
        const auto intermediate = problem.intermediate_size;
        auto* base = static_cast<std::uint8_t*>(buffer);
        std::size_t offset = 0;
        auto reserve = [&](std::size_t bytes) {
            auto* ptr = base ? base + offset : nullptr;
            offset += cute::round_up(bytes, std::size_t{16});
            return ptr;
        };
        auto bind_tensor = [&](auto& tensor, decltype(tensor.shape) shape) {
            auto* ptr = reserve(tensor.bytes(shape));
            if (ptr) {
                tensor.init(reinterpret_cast<decltype(tensor.data)>(ptr),
                            shape);
            }
        };
        auto bind_activation = [&]<typename Activation>(
                                   Activation& activation,
                                   cute::array<std::uint32_t, 2> shape,
                                   std::uint32_t extra_value_rows = 0) {
            auto value_shape = Activation::value_shape(shape);
            value_shape[0] += extra_value_rows;
            auto* values = reserve(activation.values.bytes(value_shape));
            auto* scales = reserve(
                activation.scales.bytes(Activation::scale_shape(shape)));
            if (values) {
                activation.init(
                    reinterpret_cast<typename Activation::Element*>(values),
                    reinterpret_cast<typename Activation::Scale*>(scales),
                    shape);
            }
        };

        cta_comm = reinterpret_cast<CtaComm*>(reserve(sizeof(*cta_comm)));
        router_tables = reinterpret_cast<MoERoutingTables<Dims>*>(
            reserve(sizeof(*router_tables)));
        bind_tensor(router_logits,
                    {problem.num_router_gates, tokens, problem.num_experts});
        bind_activation(latent_down_out, {tokens, problem.latent_size});
        bind_activation(shared_up_out, {tokens, intermediate});
        bind_tensor(shared_down_out, {tokens, problem.hidden_size});
        // Keep the eight padded rows outside the routed-up value view.
        bind_activation(routed_up_out, {kMaxRoutedRows, intermediate}, 8u);
        bind_tensor(routed_down_out, {kMaxRoutedRows, problem.input_size()});
        return offset;
    }

    static std::size_t bytes(MoEProblemSize const& problem) {
        MoEWorkspace view;
        return view.bind(nullptr, problem);
    }
};

}  // namespace halops::moe
