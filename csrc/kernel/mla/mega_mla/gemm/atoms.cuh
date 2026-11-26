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

#include "kernel/mla/mega_mla/gemm/config.cuh"

namespace halops::mla {
template <class Geom>
struct GemmAtomsA {
    using Inst = GemmInst<Geom>;
    typename Inst::Mainloop::TMALoad_A tma_a;
    typename Inst::Mainloop::TMALoad_SFA tma_sfa;
    typename Inst::Traits::LayoutSFA layout_sfa;
    int w_out;
    int k_dim;
};

template <class Geom>
CUTE_HOST GemmAtomsA<Geom> make_atoms_a(const void* weight,
                                        const void* weight_sf, int w_out,
                                        int k_dim) {
    using Traits = typename GemmInst<Geom>::Traits;
    using Mainloop = typename GemmInst<Geom>::Mainloop;

    auto layout_a =
        cute::make_layout(cute::make_shape(w_out, k_dim, 1),
                          cute::make_stride(k_dim, cute::Int<1>{}, 0));
    auto layout_sfa = Traits::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
        cute::make_shape(w_out, Geom::BN, k_dim, 1));

    cute::Tensor tensor_a = cute::make_tensor(
        cute::make_gmem_ptr(static_cast<const cutlass::float_e4m3_t*>(weight)),
        layout_a);
    cute::Tensor tensor_sfa = cute::make_tensor(
        static_cast<const cutlass::float_ue8m0_t*>(weight_sf), layout_sfa);

    auto cluster_layout_vmnk = cute::tiled_divide(
        cute::make_layout(typename Traits::ClusterShape_MNK{}),
        cute::make_tile(typename Traits::TiledMma::AtomThrID{}));

    GemmAtomsA<Geom> atoms{
        cute::make_tma_atom_A_sm100(
            typename Mainloop::GmemTiledLoad{}, tensor_a,
            typename Traits::SmemLayoutA{}(cute::_, cute::_, cute::_,
                                           cute::Int<0>{}),
            typename Traits::TileShape{}, typename Traits::TiledMma{},
            cluster_layout_vmnk),
        cute::make_tma_atom_A_sm100<uint16_t>(
            typename Mainloop::GmemTiledLoad{}, tensor_sfa,
            typename Traits::SmemLayoutSFA{}(cute::_, cute::_, cute::_,
                                             cute::Int<0>{}),
            typename Traits::TileShape{}, typename Traits::TiledMma{},
            cluster_layout_vmnk),
        layout_sfa,
        w_out,
        k_dim,
    };
    return atoms;
}

template <class Geom>
struct GemmAtomsB {
    using Inst = GemmInst<Geom>;
    typename Inst::Mainloop::TMALoad_B tma_b;
    typename Inst::Mainloop::TMALoad_SFB tma_sfb;
    typename Inst::Traits::LayoutSFB layout_sfb;
    int n_pad;
    int k_dim;
};

template <class Geom>
CUTE_HOST GemmAtomsB<Geom> make_atoms_b(const void* act_fp8, const void* act_sf,
                                        int w_out, int n_pad, int k_dim,
                                        int n_pad_sf) {
    using Traits = typename GemmInst<Geom>::Traits;
    using Mainloop = typename GemmInst<Geom>::Mainloop;

    auto layout_b =
        cute::make_layout(cute::make_shape(n_pad, k_dim, 1),
                          cute::make_stride(k_dim, cute::Int<1>{}, 0));
    auto layout_sfb = Traits::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
        cute::make_shape(w_out, n_pad_sf, k_dim, 1));

    cute::Tensor tensor_b = cute::make_tensor(
        cute::make_gmem_ptr(static_cast<const cutlass::float_e4m3_t*>(act_fp8)),
        layout_b);
    cute::Tensor tensor_sfb = cute::make_tensor(
        static_cast<const cutlass::float_ue8m0_t*>(act_sf), layout_sfb);

    auto cluster_layout_vmnk = cute::tiled_divide(
        cute::make_layout(typename Traits::ClusterShape_MNK{}),
        cute::make_tile(typename Traits::TiledMma::AtomThrID{}));
    auto cluster_layout_sfb_vmnk = cute::tiled_divide(
        cute::make_layout(typename Traits::ClusterShape_MNK{}),
        cute::make_tile(typename Traits::TiledMma_SF::AtomThrID{}));

    GemmAtomsB<Geom> atoms{
        cute::make_tma_atom_B_sm100(
            typename Mainloop::GmemTiledLoad{}, tensor_b,
            typename Traits::SmemLayoutB{}(cute::_, cute::_, cute::_,
                                           cute::Int<0>{}),
            typename Traits::TileShape{}, typename Traits::TiledMma{},
            cluster_layout_vmnk),
        cute::make_tma_atom_B_sm100<uint16_t>(
            typename Mainloop::GmemTiledLoad{}, tensor_sfb,
            typename Traits::SmemLayoutSFB{}(cute::_, cute::_, cute::_,
                                             cute::Int<0>{}),
            typename Traits::TileShape_SF{}, typename Traits::TiledMma_SF{},
            cluster_layout_sfb_vmnk),
        layout_sfb,
        n_pad,
        k_dim,
    };
    return atoms;
}

template <class Geom>
using WeightAtoms =
    cute::conditional_t<Geom::WeightInA, GemmAtomsA<Geom>, GemmAtomsB<Geom>>;
template <class Geom>
using ActAtoms =
    cute::conditional_t<Geom::WeightInA, GemmAtomsB<Geom>, GemmAtomsA<Geom>>;

template <class Geom>
CUTE_HOST WeightAtoms<Geom> make_weight_atoms(const void* weight,
                                              const void* w_sf, int w_out,
                                              int k) {
    if constexpr (Geom::WeightInA)
        return make_atoms_a<Geom>(weight, w_sf, w_out, k);
    else {
        const int w_tiles = (w_out + Geom::WeightTile - 1) / Geom::WeightTile;
        return make_atoms_b<Geom>(weight, w_sf, w_out, w_out, k,
                                  w_tiles * Geom::CtaNSf);
    }
}

template <class Geom>
CUTE_HOST ActAtoms<Geom> make_act_atoms(const void* act, const void* act_sf,
                                        int w_out, int t_pad, int k,
                                        int t_pad_sf) {
    if constexpr (Geom::WeightInA)
        return make_atoms_b<Geom>(act, act_sf, w_out, t_pad, k, t_pad_sf);
    else
        return make_atoms_a<Geom>(act, act_sf, t_pad, k);
}

template <class Geom>
CUTE_HOST_DEVICE auto const& sf_layout(const GemmAtomsA<Geom>& at) {
    return at.layout_sfa;
}
template <class Geom>
CUTE_HOST_DEVICE auto const& sf_layout(const GemmAtomsB<Geom>& at) {
    return at.layout_sfb;
}
}  // namespace halops::mla
