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
    static constexpr bool KDenseW8A8MXFP8 = KernelTraits::KDenseW8A8MXFP8;

    // --- Preamble-cost specializations ---
    // The constructor runs in every warp of the CTA and is pure per-CTA
    // preamble; these constants let the compiler drop provably dead parts.
    //
    // kAlwaysCompact: host sets use_compact_sched = (S == 1), and S > 1 only
    // with UseTileSwizzle, so without swizzle only the compact path is
    // reachable (tile_num_m_/tile_num_n_ are then dead too).
    static constexpr bool kAlwaysCompact = !UseTileSwizzle;
    // kIsSingleCtaCluster: cluster == CTA, so cluster queries reduce to
    // gridDim/blockIdx (cheaper than CuTe's %%nclusterid/%%clusterid reads).
    static constexpr bool kIsSingleCtaCluster =
        (KernelTraits::KNumCTA == 1) && (cute::size(ClusterShape_MNK{}) == 1);

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
        // Host-precomputed fast divmod (avoid runtime integer division)
        cutlass::FastDivmod divmod_tiles;  // ÷ tiles_per_expert
        cutlass::FastDivmod divmod_ss;     // ÷ S*S
        cutlass::FastDivmod
            divmod_blocks;  // ÷ blocks_fast (mode-dependent, see below)
        cutlass::FastDivmod divmod_s;  // ÷ S

        // Compact scheduling flag (set by host, all other compact state
        // computed on device — no D2H sync needed).
        bool use_compact_sched;  // true for Masked mode with linear path (S=1)

        // Linear-path traversal orientation (Naive only; see
        // to_underlying_arguments).
        //   false → N-fast (tile_id ÷ expert_ctn): consecutive tiles sweep N.
        //   true  → M-fast (tile_id ÷ expert_ctm): consecutive tiles sweep M.
        bool traverse_m_fast;

        // Contiguous-range tile assignment instead of strided (Batch SwapAB
        // only). See to_underlying_arguments.
        bool chunked_assign;

        // --- Tail-only stream-K (Naive only; streamk_rem == 0 disables) ---
        // C clusters over W tiles: the first streamk_tail_base = (W/C)*C tiles
        // stay data-parallel (full K); each of the remaining streamk_rem tiles
        // is split into streamk_splits K-chunks on different clusters, so the
        // makespan drops from ceil(W/C) to W/C + 1/splits tile-times and only
        // the tail needs fp32 partials. Fused work-id space, one launch:
        //   [0, tail_base)                         whole tiles, full K range
        //   [tail_base, tail_base + rem * splits)  (tail tile, K-chunk) pairs
        int streamk_tail_base;
        int streamk_rem;
        int streamk_splits;
        int k_tiles;  // ceil_div(k, KBlockK); chunk bounds are derived from it
    };

    static Params to_underlying_arguments(int m, int n, int num_experts = 0,
                                          int max_m_per_expert = 0,
                                          int const* masked_m = nullptr,
                                          int const* expert_ids = nullptr,
                                          int num_clusters = 0,
                                          int k_tiles = 0) {
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

        // --- Linear-path traversal orientation (Naive only) ---
        // The operand indexed by the fast dimension stays L2-resident; the
        // other is streamed once. Keep the smaller one resident:
        //   N-fast → B (N×K) resident; right when N <= M.
        //   M-fast → A (M×K) resident; right when M <  N.
        // Naive non-SwapAB uses M-fast when m < n (N-fast re-reads a large B
        // from DRAM every wave). HALOPS_TRAVERSE_MFAST=0 forces N-fast.
        bool traverse_m_fast = false;
        if constexpr (KernelTraits::KernelType == KernelMode::Dense &&
                      !SwapAB) {
            static const bool mfast_enabled = [] {
                const char* e = std::getenv("HALOPS_TRAVERSE_MFAST");
                return e == nullptr || e[0] != '0';
            }();
            traverse_m_fast = mfast_enabled && (S == 1) && (m < n);
        }

        int divmod_blocks_divisor;
        if (S == 1 && traverse_m_fast) {
            divmod_blocks_divisor = expert_ctm;
        } else {
            divmod_blocks_divisor = SwapAB ? blocks_m : blocks_n;
        }
        cutlass::FastDivmod divmod_tiles(tiles_per_expert);
        cutlass::FastDivmod divmod_ss(S * S);
        cutlass::FastDivmod divmod_blocks(divmod_blocks_divisor);
        cutlass::FastDivmod divmod_s(S);

        // --- Tail-only stream-K decomposition (Naive only; see Params) ---
        // splits must divide k_tiles so every chunk has the same length.
        int streamk_tail_base = total_tiles;
        int streamk_rem = 0;
        int streamk_splits = 1;
        // KSupportsStreamK (kernel_traits.h) limits this to BF16 dense without
        // bias/alpha (they cannot deposit a partial) and without SwapAB.
        if constexpr (KernelTraits::KSupportsStreamK) {
            // Default ON; rejected shapes keep streamk_rem == 0 and run the
            // plain data-parallel path (no workspace, no reduce launch).
            // HALOPS_STREAMK=0 forces the data-parallel path.
            static const bool streamk_enabled = [] {
                const char* e = std::getenv("HALOPS_STREAMK");
                return e == nullptr || e[0] != '0';
            }();
            const int C = num_clusters;
            // Grid-underfill extension: total_tiles > 0 (not > C) also admits
            // whole == 0 (thin M, total_tiles < C), where every tile is split
            // so the existing tiles occupy more SMs. p_cap = C / rem keeps
            // near-full grids at p = 1 and bounds the workspace at <= C slots.
            // HALOPS_STREAMK_FILL=0 restores the tail-only gate.
            static const bool streamk_fill = [] {
                const char* e = std::getenv("HALOPS_STREAMK_FILL");
                return e == nullptr || e[0] != '0';
            }();
            const bool size_ok =
                streamk_fill ? (total_tiles > 0) : (total_tiles > C);
            if (streamk_enabled && S == 1 && C > 0 && k_tiles > 0 && size_ok) {
                const int whole = total_tiles / C;
                const int rem = total_tiles - whole * C;
                // Split only when the remainder idles >= 5% of total SM-time
                // (break-even for the rem*splits partial-tile reduction).
                const bool worth_it =
                    rem > 0 && (C - rem) * 20 >= (whole + 1) * C;
                if (worth_it) {
                    // Use the largest p <= min(C / rem, 8) dividing k_tiles:
                    // the overhead is fixed, not per-split.
                    // HALOPS_STREAMK_P=<p> overrides the cap.
                    int p_cap = C / rem < 8 ? C / rem : 8;
                    static const int p_forced = [] {
                        const char* e = std::getenv("HALOPS_STREAMK_P");
                        return e != nullptr ? atoi(e) : 0;
                    }();
                    if (p_forced > 0) {
                        p_cap = p_forced;
                    }
                    int p = 1;
                    for (int cand = p_cap; cand >= 2; --cand) {
                        if (k_tiles % cand == 0) {
                            p = cand;
                            break;
                        }
                    }
                    if (p >= 2 && (whole > 0 || p >= 4)) {
                        // Underfill gate: with whole == 0, only p >= 4 repays
                        // the fp32 workspace round-trip and reduce launch.

                        streamk_tail_base = whole * C;
                        streamk_rem = rem;
                        streamk_splits = p;
                        // Tail items become the (whole+1)-th item of clusters
                        // 0..rem*p-1, reusing the resident mainloop pipeline.
                        total_tiles = streamk_tail_base + rem * p;
                    }
                }
            }
        }

        // Compact scheduling: enabled for Masked and Contiguous modes with S=1
        // (linear path). All compact scheduling state (total_valid_tiles,
        // expert cache) is computed on the device — no D2H copy or host-side
        // masked_m access.
        bool use_compact_sched = false;
        if (S == 1) {
            use_compact_sched = true;
        }

        // --- Contiguous-range assignment (Batch SwapAB only) ---
        // SwapAB numbering keeps one item's N segments consecutive, but the
        // persistent stride spreads them over different CTAs, so the shared
        // activation's L2 reuse becomes a race. Chunked assignment gives CTA i
        // the range [i*chunk, (i+1)*chunk), so one CTA processes them
        // back-to-back. Only when expert_ctm > 1 (otherwise nothing is
        // shared). HALOPS_BATCH_CHUNK=0 restores the stride.
        bool chunked_assign = false;
        if constexpr (KernelTraits::KernelType == KernelMode::Batch && SwapAB) {
            static const bool chunk_enabled = [] {
                const char* e = std::getenv("HALOPS_BATCH_CHUNK");
                return e == nullptr || e[0] != '0';
            }();
            chunked_assign = chunk_enabled && S == 1 && expert_ctm > 1;
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
                use_compact_sched,
                traverse_m_fast,
                chunked_assign,
                streamk_tail_base,
                streamk_rem,
                streamk_splits,
                k_tiles};
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
    // Chunked assignment (Batch SwapAB): step is 1 and each CTA iterates a
    // contiguous range ending at tile_end_. Otherwise step = cluster_num_ and
    // the range ends at the total tile count.
    uint32_t advance_step_;
    uint32_t tile_end_;

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
    bool traverse_m_fast_;
    bool chunked_assign_;

    // Tail-only stream-K state (see Params). streamk_rem_ == 0 disables.
    uint32_t streamk_tail_base_;
    uint32_t streamk_rem_;
    uint32_t streamk_splits_;
    uint32_t k_tiles_;
    // Resolved for the current work item by find_next_valid_tile().
    uint32_t k_begin_;
    uint32_t k_end_;
    int partial_slot_;  // -1 = write straight to D; else index into the fp32 ws
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
          use_compact_sched_(params.use_compact_sched),
          traverse_m_fast_(params.traverse_m_fast),
          chunked_assign_(params.chunked_assign),
          streamk_tail_base_(params.streamk_tail_base),
          streamk_rem_(params.streamk_rem),
          streamk_splits_(params.streamk_splits),
          k_tiles_(params.k_tiles) {
        // tile_num_m_/tile_num_n_ only feed the non-compact bounds check, so
        // skip the two ceil_div's when that path is statically unreachable.
        if constexpr (!kAlwaysCompact) {
            tile_num_m_ = ceil_div(params.m, size<0>(TileShape{}));
            tile_num_n_ = ceil_div(params.n, size<1>(TileShape{}));
        }

        int cluster_idx;
        if constexpr (kIsSingleCtaCluster) {
            // cluster == CTA: read gridDim/blockIdx directly (cheaper than
            // CuTe's %%nclusterid / %%clusterid reads).
            cluster_num_ = gridDim.x * gridDim.y;
            cluster_idx = blockIdx.x * gridDim.y + blockIdx.y;
            // Single-CTA cluster: block_rank 0, flat coord (0,0,0).
            cluster_m_tiles_ = kClusterMTiles;
            cluster_n_tiles_ = kClusterNTiles;
            cta_m_offset_ = 0;
            cta_n_offset_ = 0;
            cta_v_ = 0;
        } else {
            cluster_num_ =
                cute::cluster_grid_dims().x * cute::cluster_grid_dims().y;
            cluster_idx =
                cute::cluster_id_in_grid().x * cute::cluster_grid_dims().y +
                cute::cluster_id_in_grid().y;

            cute::Layout cluster_layout_vmnk = cute::tiled_divide(
                cute::make_layout(ClusterShape_MNK{}),
                cute::make_tile(typename TiledMma::AtomThrID{}));
            int block_rank = cute::block_rank_in_cluster();
            auto cta_coord_vmnk =
                cluster_layout_vmnk.get_flat_coord(block_rank);

            cluster_m_tiles_ = cute::size<1>(cluster_layout_vmnk);
            cluster_n_tiles_ = cute::size<2>(cluster_layout_vmnk);
            cta_m_offset_ = cute::get<1>(cta_coord_vmnk);
            cta_n_offset_ = cute::get<2>(cta_coord_vmnk);
            cta_v_ = get<0>(cta_coord_vmnk);
        }

        // Compute compact scheduling state on device
        if (kAlwaysCompact || use_compact_sched_) {
            if constexpr (KernelTraits::KernelType == KernelMode::Contiguous ||
                          KernelTraits::KernelType == KernelMode::Dense ||
                          KernelTraits::KernelType == KernelMode::Batch) {
                // No padding: every tile is valid.
                total_valid_tiles_ = total_tiles_;
                cached_expert_ = 0;
                cached_expert_start_ = 0;
                // Single logical expert: the expert-cache fast path must
                // always hit (the slow path dereferences masked_m, nullptr
                // here), including for folded stream-K base tile ids.
                cached_expert_end_ = 0xFFFFFFFFu;
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

        // Defaults for the non-compact path (stream-K is compact/Naive-only).
        k_begin_ = 0;
        k_end_ = k_tiles_;
        partial_slot_ = -1;

        // Start at this cluster's first tile
        current_tile_id_ = cluster_idx;
        advance_step_ = cluster_num_;
        // Strided default; stream-K tail work ids lie inside
        // total_valid_tiles_ by construction.
        tile_end_ = total_valid_tiles_;
        if (chunked_assign_) {
            // Range [cluster_idx*chunk, +chunk), clamped to the tile count.
            uint32_t chunk =
                (total_valid_tiles_ + cluster_num_ - 1) / cluster_num_;
            uint32_t base = cluster_idx * chunk;
            current_tile_id_ = base;
            tile_end_ = base + chunk;
            if (tile_end_ > total_valid_tiles_ || chunk == 0) {
                tile_end_ = total_valid_tiles_;
            }
            if (current_tile_id_ > total_valid_tiles_) {
                current_tile_id_ = total_valid_tiles_;
            }
            advance_step_ = 1;
        }
        find_next_valid_tile();
    }

    CUTLASS_DEVICE int cta_v() const { return cta_v_; }
    CUTLASS_DEVICE bool is_valid() const { return is_valid_; }
    CUTLASS_DEVICE uint32_t m_idx() const { return m_idx_; }
    CUTLASS_DEVICE uint32_t n_idx() const { return n_idx_; }
    CUTLASS_DEVICE uint32_t expert_idx() const { return expert_idx_; }
    // K-tile range of the current work item. Producer and consumer MUST use
    // the same bounds or the mainloop pipeline deadlocks.
    CUTLASS_DEVICE uint32_t k_begin() const { return k_begin_; }
    CUTLASS_DEVICE uint32_t k_end() const { return k_end_; }
    // >= 0 when this work item computed only a K-slice and must deposit an
    // fp32 partial instead of the final bf16 result.
    CUTLASS_DEVICE int partial_slot() const { return partial_slot_; }

    CUTLASS_DEVICE void advance() {
        current_tile_id_ += advance_step_;
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
        } else if (traverse_m_fast_) {
            // M-fast: divisor was expert_ctm_, so r indexes M and q indexes N.
            cluster_m = expert * expert_ctm_ + static_cast<uint32_t>(r);
            cluster_n = static_cast<uint32_t>(q);
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
        // W8A8-dense fast path (quantized only): every tile is valid, so a
        // plain divmod resolves the tile without the compact-scan bookkeeping.
        // Guarded to !KIsBF16 (BF16 needs the stream-K fold and M-fast
        // traversal handled in the compact path below) and !traverse_m_fast_
        // (the divmod orientation below assumes N-fast).
        if constexpr (KDenseW8A8MXFP8 && !UseTileSwizzle &&
                      !KernelTraits::KIsBF16) {
            if (!traverse_m_fast_) {
                while (current_tile_id_ < total_tiles_) {
                    int q, r;
                    divmod_blocks_.fast_divmod(
                        q, r, static_cast<int>(current_tile_id_));
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
        }
        if (kAlwaysCompact || use_compact_sched_) {
            // Compact path: the tile space only contains valid cluster-tiles,
            // so we skip directly to the current tile without iterating through
            // large swaths of padding. Only need a lightweight per-CTA check
            // for the edge case where the last cluster row of an expert has
            // some CTAs with tiles beyond masked_m. With chunked_assign_ the
            // per-CTA range ends at tile_end_ instead of total_valid_tiles_.
            while (current_tile_id_ < tile_end_) {
                uint32_t cluster_m, cluster_n, expert;
                // Stream-K: ids >= streamk_tail_base_ are (tail tile, K-chunk)
                // pairs. Fold to the base tile id and resolve the K range and
                // partial slot here so all roles see identical values.
                uint32_t base_tile_id = current_tile_id_;
                k_begin_ = 0;
                k_end_ = k_tiles_;
                partial_slot_ = -1;
                if constexpr (KernelTraits::KernelType == KernelMode::Dense) {
                    if (streamk_rem_ > 0 &&
                        current_tile_id_ >= streamk_tail_base_) {
                        uint32_t local = current_tile_id_ - streamk_tail_base_;
                        uint32_t t = local / streamk_splits_;
                        uint32_t s = local - t * streamk_splits_;
                        base_tile_id = streamk_tail_base_ + t;
                        // streamk_splits_ divides k_tiles_ (host-enforced).
                        uint32_t chunk = k_tiles_ / streamk_splits_;
                        k_begin_ = s * chunk;
                        k_end_ = k_begin_ + chunk;
                        partial_slot_ = static_cast<int>(local);
                    }
                }
                tile_id_to_coords_compact(base_tile_id, cluster_m, cluster_n,
                                          expert);

                uint32_t m_idx = cluster_m * cluster_m_tiles_ + cta_m_offset_;
                uint32_t n_idx = cluster_n * cluster_n_tiles_ + cta_n_offset_;

                // Skip at cluster granularity: all CTAs of a pair-UMMA cluster
                // must make the same decision or the pipeline deadlocks.
                if constexpr (KernelTraits::KernelType == KernelMode::Masked) {
                    if constexpr (SwapAB) {
                        uint32_t local_m_cluster =
                            cluster_n - expert * expert_ctn_;
                        uint32_t local_m_start = local_m_cluster *
                                                 cluster_n_tiles_ *
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
                    expert_idx = n_idx_base * cute::size<1>(TileShape{}) /
                                 max_m_per_expert_;
                    bool valid = expert_idx < num_experts_ &&
                                 n_idx_base * cute::size<1>(TileShape{}) %
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
