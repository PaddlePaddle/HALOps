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
#include <cutlass/fast_math.h>

#include "kernel/gemm/kernel_traits.cuh"

namespace halops {

// Persistent tile scheduler: clusters stride over a global tile space ordered
// expert by expert (non-Masked modes form a single logical expert).
// - UseTileSwizzle: S x S blocks, S in [2,8] dividing both expert dims if
//   possible, else the block-fast dim. Normal: blocks N-fast, tiles M-fast
//   within a block; SwapAB: transposed.
// - Otherwise S = 1, linear: N-fast for Normal, M-fast for SwapAB.
// - With S = 1 the tile space holds only valid (masked_m) tiles, computed on
//   device; expert lookup uses a forward-scan cache.
template <typename KernelTraits>
struct TileScheduler {
    using TiledMma = typename KernelTraits::TiledMma;
    using TiledMma_SF = typename KernelTraits::TiledMma_SF;
    using TileShape = typename KernelTraits::TileShape;
    using ClusterShape_MNK = typename KernelTraits::ClusterShape_MNK;

    using ClusterLayout_VMNK = decltype(cute::tiled_divide(
        cute::make_layout(ClusterShape_MNK{}),
        cute::make_tile(typename TiledMma::AtomThrID{})));

    static constexpr int kClusterMTiles = cute::size<1>(ClusterLayout_VMNK{});
    static constexpr int kClusterNTiles = cute::size<2>(ClusterLayout_VMNK{});
    static constexpr bool SwapAB = KernelTraits::SwapAB;
    static constexpr bool UseTileSwizzle = KernelTraits::UseTileSwizzle;
    static constexpr bool KDenseW8A8 = KernelTraits::KDenseW8A8;

    struct Params {
        int m;
        int n;
        int num_experts;
        int max_m_per_expert;
        int const* masked_m;
        int const* expert_ids;
        int swizzle_size;  // S
        int expert_ctm;    // per-expert cluster-tiles in M
        int expert_ctn;    // per-expert cluster-tiles in N
        int total_tiles;   // num_logical_experts * tiles_per_expert
        cutlass::FastDivmod divmod_tiles;   // ÷ tiles_per_expert
        cutlass::FastDivmod divmod_ss;      // ÷ S*S
        cutlass::FastDivmod divmod_blocks;  // ÷ blocks along the fast dim
        cutlass::FastDivmod divmod_s;       // ÷ S

        // Other compact state is derived on device (no D2H sync).
        bool use_compact_sched;
    };

    static Params to_underlying_arguments(int m, int n, int num_experts = 0,
                                          int max_m_per_expert = 0,
                                          int const* masked_m = nullptr,
                                          int const* expert_ids = nullptr) {
        constexpr int kTileM = cute::size<0>(TileShape{});
        constexpr int kTileN = cute::size<1>(TileShape{});

        int tile_num_m = (m + kTileM - 1) / kTileM;
        int tile_num_n = (n + kTileN - 1) / kTileN;
        int ctm = (tile_num_m + kClusterMTiles - 1) / kClusterMTiles;
        int ctn = (tile_num_n + kClusterNTiles - 1) / kClusterNTiles;

        // Experts are stacked along M (Normal) or N (SwapAB).
        int expert_ctm, expert_ctn;
        int num_logical_experts;

        if constexpr (KernelTraits::KernelType == KernelMode::Masked) {
            num_logical_experts = num_experts;
            if constexpr (SwapAB) {
                expert_ctm = ctm;
                expert_ctn = ctn / num_experts;
            } else {
                expert_ctm = ctm / num_experts;
                expert_ctn = ctn;
            }
        } else {
            expert_ctm = ctm;
            expert_ctn = ctn;
            num_logical_experts = 1;
        }

        int S = 1;
        if constexpr (UseTileSwizzle) {
            // Prefer S dividing both dims, else the block-fast dim.
            int fast_dim = SwapAB ? expert_ctm : expert_ctn;
            int min_dim = expert_ctm < expert_ctn ? expert_ctm : expert_ctn;
            int cap = min_dim < 8 ? min_dim : 8;

            for (int s = cap; s >= 2; s--) {
                if (expert_ctn % s == 0 && expert_ctm % s == 0) {
                    S = s;
                    break;
                }
            }
            if (S == 1) {
                for (int s = cap; s >= 2; s--) {
                    if (fast_dim % s == 0) {
                        S = s;
                        break;
                    }
                }
            }
        }

        int blocks_m = (expert_ctm + S - 1) / S;
        int blocks_n = (expert_ctn + S - 1) / S;
        int tiles_per_expert = blocks_m * blocks_n * S * S;
        int total_tiles = num_logical_experts * tiles_per_expert;

        int divmod_blocks_divisor = SwapAB ? blocks_m : blocks_n;
        cutlass::FastDivmod divmod_tiles(tiles_per_expert);
        cutlass::FastDivmod divmod_ss(S * S);
        cutlass::FastDivmod divmod_blocks(divmod_blocks_divisor);
        cutlass::FastDivmod divmod_s(S);

        bool use_compact_sched = false;
        if (S == 1) {
            use_compact_sched = true;
        }

        return {m,
                n,
                num_experts,
                max_m_per_expert,
                masked_m,
                expert_ids,
                S,
                expert_ctm,
                expert_ctn,
                total_tiles,
                divmod_tiles,
                divmod_ss,
                divmod_blocks,
                divmod_s,
                use_compact_sched};
    }

    uint32_t tile_num_m_;
    uint32_t tile_num_n_;
    uint32_t cluster_m_tiles_;
    uint32_t cluster_n_tiles_;
    uint32_t cta_m_offset_;
    uint32_t cta_n_offset_;
    uint32_t cta_v_;

    uint32_t cluster_num_;      // stride = number of clusters
    uint32_t current_tile_id_;  // current position in global tile space

    uint32_t expert_ctm_;
    uint32_t expert_ctn_;
    uint32_t total_tiles_;
    cutlass::FastDivmod divmod_tiles_;   // ÷ tiles_per_expert
    cutlass::FastDivmod divmod_ss_;      // ÷ S*S
    cutlass::FastDivmod divmod_blocks_;  // ÷ blocks_fast (mode-dependent)
    cutlass::FastDivmod divmod_s_;       // ÷ S

    uint32_t num_experts_;
    uint32_t max_m_per_expert_;
    int const* masked_m_;
    int const* expert_ids_;

    bool use_compact_sched_;
    uint32_t
        total_valid_tiles_;  // sum of valid cluster-tiles across all experts
    // tile ids only grow, so the expert cache is advanced by forward scan.
    uint32_t cached_expert_;        // last resolved expert index
    uint32_t cached_expert_start_;  // first compact tile_id of cached_expert_
    uint32_t
        cached_expert_end_;  // one past last compact tile_id of cached_expert_

    uint32_t m_idx_;
    uint32_t n_idx_;
    uint32_t expert_idx_;
    bool is_valid_;

    // Number of valid cluster-tiles of expert e given masked_m[e].
    CUTLASS_DEVICE uint32_t expert_valid_tiles(uint32_t e) const {
        constexpr int kExpertTileSize =
            SwapAB ? cute::size<1>(TileShape{}) : cute::size<0>(TileShape{});
        constexpr int kExpertClusterTiles =
            SwapAB ? kClusterNTiles : kClusterMTiles;
        int valid_m = __ldg(&masked_m_[e]);
        int valid_ct = (valid_m + kExpertTileSize * kExpertClusterTiles - 1) /
                       (kExpertTileSize * kExpertClusterTiles);
        if constexpr (SwapAB) {
            return expert_ctm_ * valid_ct;
        } else {
            return valid_ct * expert_ctn_;
        }
    }

    CUTLASS_DEVICE
    TileScheduler(Params const& params)
        : num_experts_(params.num_experts),
          max_m_per_expert_(params.max_m_per_expert),
          masked_m_(params.masked_m),
          expert_ids_(params.expert_ids),
          expert_ctm_(params.expert_ctm),
          expert_ctn_(params.expert_ctn),
          total_tiles_(params.total_tiles),
          divmod_tiles_(params.divmod_tiles),
          divmod_ss_(params.divmod_ss),
          divmod_blocks_(params.divmod_blocks),
          divmod_s_(params.divmod_s),
          use_compact_sched_(params.use_compact_sched) {
        tile_num_m_ = cute::ceil_div(params.m, cute::size<0>(TileShape{}));
        tile_num_n_ = cute::ceil_div(params.n, cute::size<1>(TileShape{}));

        cluster_num_ =
            cute::cluster_grid_dims().x * cute::cluster_grid_dims().y;
        int cluster_idx =
            cute::cluster_id_in_grid().x * cute::cluster_grid_dims().y +
            cute::cluster_id_in_grid().y;

        cute::Layout cluster_layout_vmnk =
            cute::tiled_divide(cute::make_layout(ClusterShape_MNK{}),
                               cute::make_tile(typename TiledMma::AtomThrID{}));
        int block_rank = cute::block_rank_in_cluster();
        auto cta_coord_vmnk = cluster_layout_vmnk.get_flat_coord(block_rank);

        cluster_m_tiles_ = cute::size<1>(cluster_layout_vmnk);
        cluster_n_tiles_ = cute::size<2>(cluster_layout_vmnk);
        cta_m_offset_ = cute::get<1>(cta_coord_vmnk);
        cta_n_offset_ = cute::get<2>(cta_coord_vmnk);
        cta_v_ = cute::get<0>(cta_coord_vmnk);

        if (use_compact_sched_) {
            if constexpr (KernelTraits::KernelType == KernelMode::Contiguous ||
                          KernelTraits::KernelType == KernelMode::Dense ||
                          KernelTraits::KernelType == KernelMode::Batch) {
                // No padding: every tile is valid.
                total_valid_tiles_ = total_tiles_;
                cached_expert_ = 0;
                cached_expert_start_ = 0;
                cached_expert_end_ = total_tiles_;
            } else {
                total_valid_tiles_ = 0;
                for (uint32_t e = 0; e < num_experts_; e++) {
                    total_valid_tiles_ += expert_valid_tiles(e);
                }
                cached_expert_ = 0;
                cached_expert_start_ = 0;
                cached_expert_end_ = expert_valid_tiles(0);
            }
        } else {
            total_valid_tiles_ = 0;
            cached_expert_ = 0;
            cached_expert_start_ = 0;
            cached_expert_end_ = 0;
        }

        current_tile_id_ = cluster_idx;
        find_next_valid_tile();
    }

    CUTLASS_DEVICE int cta_v() const { return cta_v_; }
    CUTLASS_DEVICE bool is_valid() const { return is_valid_; }
    CUTLASS_DEVICE uint32_t m_idx() const { return m_idx_; }
    CUTLASS_DEVICE uint32_t n_idx() const { return n_idx_; }
    CUTLASS_DEVICE uint32_t expert_idx() const { return expert_idx_; }

    CUTLASS_DEVICE void advance() {
        current_tile_id_ += cluster_num_;
        find_next_valid_tile();
    }

  private:
    // Map a compact tile_id (valid tiles only, contiguous per expert) to
    // (cluster_m, cluster_n, expert).
    CUTLASS_DEVICE bool tile_id_to_coords_compact(uint32_t tile_id,
                                                  uint32_t& cluster_m,
                                                  uint32_t& cluster_n,
                                                  uint32_t& expert) {
        if (tile_id >= cached_expert_start_ && tile_id < cached_expert_end_) {
            expert = cached_expert_;
        } else {
            uint32_t start = cached_expert_start_;
            uint32_t e = cached_expert_;
            // Defensive; tile_id never decreases.
            if (tile_id < start) {
                e = 0;
                start = 0;
            }
            while (e < num_experts_) {
                uint32_t tiles_e = expert_valid_tiles(e);
                uint32_t end = start + tiles_e;
                if (tile_id < end) {
                    expert = e;
                    cached_expert_ = e;
                    cached_expert_start_ = start;
                    cached_expert_end_ = end;
                    break;
                }
                start = end;
                e++;
            }
        }

        int local_tile = static_cast<int>(tile_id - cached_expert_start_);

        // Normal: q = local m, r = local n; SwapAB: swapped.
        int q, r;
        divmod_blocks_.fast_divmod(q, r, local_tile);

        if constexpr (SwapAB) {
            uint32_t local_cluster_m = r;
            uint32_t local_cluster_n = q;
            cluster_m = local_cluster_m;
            cluster_n = expert * expert_ctn_ + local_cluster_n;
        } else {
            uint32_t local_cluster_m = q;
            uint32_t local_cluster_n = r;
            cluster_m = expert * expert_ctm_ + local_cluster_m;
            cluster_n = local_cluster_n;
        }
        return true;
    }

    // Map a global tile_id to (cluster_m, cluster_n, expert). Returns false for
    // S x S padding tiles outside the expert's range.
    CUTLASS_DEVICE bool tile_id_to_coords(uint32_t tile_id, uint32_t& cluster_m,
                                          uint32_t& cluster_n,
                                          uint32_t& expert) const {
        if constexpr (UseTileSwizzle) {
            int i_expert, i_local_id;
            divmod_tiles_.fast_divmod(i_expert, i_local_id,
                                      static_cast<int>(tile_id));

            int i_block_idx, i_local;
            divmod_ss_.fast_divmod(i_block_idx, i_local, i_local_id);

            int bq, br;
            divmod_blocks_.fast_divmod(bq, br, i_block_idx);

            int lq, lr;
            divmod_s_.fast_divmod(lq, lr, i_local);

            // Remainder is fast. SwapAB: blocks M-fast, N-fast within a block.
            uint32_t block_m = SwapAB ? br : bq;
            uint32_t block_n = SwapAB ? bq : br;
            uint32_t local_m = SwapAB ? lq : lr;
            uint32_t local_n = SwapAB ? lr : lq;

            uint32_t S = divmod_s_.divisor;
            uint32_t expert_m = block_m * S + local_m;
            uint32_t expert_n = block_n * S + local_n;

            // Padded S x S blocks may exceed the expert's tile range.
            if (expert_m >= expert_ctm_ || expert_n >= expert_ctn_) {
                return false;
            }

            expert = i_expert;
            if constexpr (SwapAB) {
                cluster_m = expert_m;
                cluster_n = expert * expert_ctn_ + expert_n;
            } else {
                cluster_m = expert * expert_ctm_ + expert_m;
                cluster_n = expert_n;
            }
        } else {
            // Linear: Normal q = cluster_m, r = cluster_n; SwapAB swapped.
            int q, r;
            divmod_blocks_.fast_divmod(q, r, static_cast<int>(tile_id));

            if constexpr (SwapAB) {
                cluster_m = r;
                cluster_n = q;
            } else {
                cluster_m = q;
                cluster_n = r;
            }
            expert = 0;
        }
        return true;
    }

    // Scan forward from current_tile_id_ to find the next valid tile,
    // caching its coordinates in m_idx_/n_idx_/expert_idx_/is_valid_.
    CUTLASS_DEVICE void find_next_valid_tile() {
        if constexpr (KDenseW8A8 && !UseTileSwizzle) {
            while (current_tile_id_ < total_tiles_) {
                int q, r;
                divmod_blocks_.fast_divmod(q, r,
                                           static_cast<int>(current_tile_id_));
                if constexpr (SwapAB) {
                    m_idx_ = r * cluster_m_tiles_ + cta_m_offset_;
                    n_idx_ = q * cluster_n_tiles_ + cta_n_offset_;
                } else {
                    m_idx_ = q * cluster_m_tiles_ + cta_m_offset_;
                    n_idx_ = r * cluster_n_tiles_ + cta_n_offset_;
                }
                expert_idx_ = 0;
                is_valid_ = true;
                return;
            }
            is_valid_ = false;
            return;
        }
        if (use_compact_sched_) {
            while (current_tile_id_ < total_valid_tiles_) {
                uint32_t cluster_m, cluster_n, expert;
                uint32_t tile_id = current_tile_id_;
                tile_id_to_coords_compact(tile_id, cluster_m, cluster_n,
                                          expert);

                uint32_t m_idx = cluster_m * cluster_m_tiles_ + cta_m_offset_;
                uint32_t n_idx = cluster_n * cluster_n_tiles_ + cta_n_offset_;

                // Skip at cluster granularity: all CTAs of a pair-UMMA cluster
                // must make the same decision or the pipeline deadlocks.
                if constexpr (KernelTraits::KernelType == KernelMode::Masked) {
                    if constexpr (SwapAB) {
                        uint32_t local_m_cluster =
                            cluster_n - expert * expert_ctn_;
                        uint32_t local_m_start =
                            local_m_cluster * cluster_n_tiles_ *
                            cute::size<1>(TileShape{});
                        if (local_m_start >=
                            static_cast<uint32_t>(__ldg(&masked_m_[expert]))) {
                            current_tile_id_ += cluster_num_;
                            continue;
                        }
                    } else {
                        uint32_t cluster_m_start = cluster_m * cluster_m_tiles_;
                        if (cluster_m_start * cute::size<0>(TileShape{}) %
                                max_m_per_expert_ >=
                            static_cast<uint32_t>(__ldg(&masked_m_[expert]))) {
                            current_tile_id_ += cluster_num_;
                            continue;
                        }
                    }
                }
                m_idx_ = m_idx;
                n_idx_ = n_idx;
                if constexpr (KernelTraits::KernelType ==
                              KernelMode::Contiguous) {
                    if constexpr (SwapAB) {
                        uint32_t global_m_idx =
                            n_idx * cute::size<1>(TileShape{});
                        expert_idx_ = expert_ids_[global_m_idx];
                    } else {
                        uint32_t global_m_idx =
                            m_idx * cute::size<0>(TileShape{});
                        expert_idx_ = expert_ids_[global_m_idx];
                    }
                } else if constexpr (KernelTraits::KernelType ==
                                     KernelMode::Batch) {
                    if constexpr (SwapAB) {
                        uint32_t global_m_idx =
                            n_idx * cute::size<1>(TileShape{});
                        expert_idx_ = global_m_idx / max_m_per_expert_;
                    } else {
                        uint32_t global_m_idx =
                            m_idx * cute::size<0>(TileShape{});
                        expert_idx_ = global_m_idx / max_m_per_expert_;
                    }
                } else {
                    expert_idx_ = expert;
                }
                is_valid_ = true;
                return;
            }
            is_valid_ = false;
            return;
        }

        // Non-compact (swizzled) path.
        while (current_tile_id_ < total_tiles_) {
            uint32_t cluster_m, cluster_n, expert;

            if (!tile_id_to_coords(current_tile_id_, cluster_m, cluster_n,
                                   expert)) {
                current_tile_id_ += cluster_num_;
                continue;
            }

            uint32_t m_idx = cluster_m * cluster_m_tiles_ + cta_m_offset_;
            uint32_t n_idx = cluster_n * cluster_n_tiles_ + cta_n_offset_;

            // Checks use cluster-level coords so all CTAs agree (pair-UMMA).
            uint32_t m_idx_base = cluster_m * cluster_m_tiles_;
            uint32_t n_idx_base = cluster_n * cluster_n_tiles_;
            if (m_idx_base >= tile_num_m_ || n_idx_base >= tile_num_n_) {
                current_tile_id_ += cluster_num_;
                continue;
            }

            uint32_t expert_idx = 0;

            if constexpr (KernelTraits::KernelType == KernelMode::Masked) {
                if constexpr (SwapAB) {
                    expert_idx = m_idx_base * cute::size<1>(TileShape{}) /
                                 max_m_per_expert_;
                    bool valid = expert_idx < num_experts_ &&
                                 m_idx_base * cute::size<1>(TileShape{}) %
                                         max_m_per_expert_ <
                                     __ldg(&masked_m_[expert_idx]);
                    if (!valid) {
                        current_tile_id_ += cluster_num_;
                        continue;
                    }
                } else {
                    expert_idx = m_idx_base * cute::size<0>(TileShape{}) /
                                 max_m_per_expert_;
                    bool valid = expert_idx < num_experts_ &&
                                 m_idx_base * cute::size<0>(TileShape{}) %
                                         max_m_per_expert_ <
                                     __ldg(&masked_m_[expert_idx]);
                    if (!valid) {
                        current_tile_id_ += cluster_num_;
                        continue;
                    }
                }
            } else if constexpr (KernelTraits::KernelType ==
                                 KernelMode::Contiguous) {
                uint32_t global_m_idx = m_idx * cute::size<0>(TileShape{});
                expert_idx = expert_ids_[global_m_idx];
            } else if constexpr (KernelTraits::KernelType ==
                                 KernelMode::Batch) {
                uint32_t global_m_idx = m_idx * cute::size<0>(TileShape{});
                expert_idx = global_m_idx / max_m_per_expert_;
            }

            m_idx_ = m_idx;
            n_idx_ = n_idx;
            expert_idx_ = expert_idx;
            is_valid_ = true;
            return;
        }
        is_valid_ = false;
    }
};

}  // namespace halops
