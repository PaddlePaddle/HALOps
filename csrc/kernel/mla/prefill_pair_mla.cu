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

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <math_constants.h>

#include <algorithm>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm80.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/mma_sm100_desc.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <initializer_list>
#include <mutex>
#include <optional>
#include <unordered_map>

#include "common/pdl.cuh"
#include "kernel/gemm/gemm_utils.h"
#include "kernel/mla/mla_intrinsics.cuh"

namespace halops {

using namespace mla;

namespace {

// A 2-CTA cluster attends a token pair (2j, 2j+1) of a request over the union
// of their KV lists, so each KV tile is read and dequantized once for both. CTA
// c holds token 2j+c and dims [256c, 256c+256) of every key. The UMMAs are
// 2-CTA with M = 128:
// - QK^T: TMEM lanes 0-63 / 64-127 hold Q dims [0, 256) / [256, 512); B is
//   CTA0's keys x low dims and CTA1's keys x high dims (N = 128), so the two
//   lane halves get partial S sums, added by the softmax.
// - PV: B is the same tile read MN-major; CTA c gets its 256 output dims.
// A pair's keys form one stream: its c128 rows (padded to a multiple of 4 so a
// gather4 group never mixes the caches) then its SWA rows, in tiles of 64.
// Tiles are split over the clusters by pair_cost; a pair cut by a range
// boundary is merged in the kernel (store_o) or by the combine kernel.
// predequant_c128 may first write a bf16 copy of each request's c128 rows;
// tiles of only c128 keys then load from it with no dequant.
// Q and O move by TMA through shared memory: per-lane loads would fill L1.
constexpr int kHeads = 64;
constexpr int kDim = 512;
constexpr int kNope = 448;
constexpr int kRowBytes = 576;
constexpr int kSfBytes = 8;
constexpr int kBf16Mant = 7;
constexpr int kBlockDims = 64;
constexpr int kHalfDims = kDim / 2;
constexpr int kTile = 64;
constexpr int kHalf = kTile / 2;
constexpr int kLocalBlocks = kHalfDims / kBlockDims;
constexpr int kAtomRows = 8;
constexpr int kAtomBytes = 1024;
constexpr int kRowPitch = kAtomBytes / kAtomRows;
constexpr int kChunk = 16;
constexpr int kElemsPerChunk = kChunk / sizeof(__nv_bfloat16);
constexpr int kBlockBytes = kTile * kRowPitch;
constexpr int kStageBytes = kLocalBlocks * kBlockBytes;
constexpr int kUmmaK = 16;
constexpr int kKStepBytes = kUmmaK * sizeof(__nv_bfloat16);
constexpr int kKStepsPerBlock = kBlockDims / kUmmaK;
constexpr int kQkSteps = kHalfDims / kUmmaK;
constexpr int kPvSteps = kTile / kUmmaK;
constexpr int kPvN = 256;
constexpr int kPvMmas = kDim / kPvN;
constexpr int kPvBlocksPerMma = kLocalBlocks / kPvMmas;
// KV stages: 4 instead of 5 leaves more L1 and room for the kQst Q buffers.
constexpr int kKvStages = 4;
// Merge ring: a split pair's parts are combined box by box (32 KB, one KV
// stage), loading the next box while reducing the current one.
constexpr int kMergeBuf = 4;
static_assert(kMergeBuf <= kKvStages);
// Pair lengths of a cluster's range kept in shared memory; more are read from
// global memory.
constexpr int kTableUnits = 32;
constexpr int kMaxClusters = 128;  // scheduled clusters (SMs / 2) at most
constexpr int kIdxStages = 8;
constexpr int kIndexDepth = 4;
constexpr int kGather = 4;
constexpr int kWarp = 32;
constexpr int kWarpGroup = 128;
constexpr int kWarpsPerGroup = kWarpGroup / kWarp;
constexpr int kIndexWarps = kTile / kWarp;
constexpr int kGroupsPerWarp = kWarp / kGather;
constexpr int kThreads = 4 * kWarpGroup;
constexpr int kQkWarp = 12, kPvWarp = 13, kIndexWarp = 14, kAllocWarp = 15;
constexpr int kSoftmaxRegs = 152, kQoRegs = 200, kDequantRegs = 104,
              kAuxRegs = 56;
static_assert(kSoftmaxRegs + kQoRegs + kDequantRegs + kAuxRegs == 4 * 128);
constexpr int kTmemCols = 512;
constexpr int kTmemO = 0;
constexpr int kTmemQ = kHalfDims;
constexpr int kTmemS = kTmemQ + kHalfDims / 2;
constexpr int kTmemSCols = kTile;
constexpr int kTmemLd = 32;
constexpr uint16_t kBothCtas = 0b11;
// Named barriers: softmax partner warps (w, w ^ 2), the dequant and Q/O
// warp groups.
constexpr uint32_t kPairBar = 1, kDequantBar = 3, kQoBar = 4;
constexpr float kMaxInit = -1e30f;
// The softmax rescales O only when a row max grows by more than 2^6 (lazy
// rescale); a larger threshold loses accuracy at long contexts.
constexpr float kRescaleLog2 = 6.0f;

// Bytes per key row gathered by a CTA (CTA 0: fp8 dims [0, 256); CTA 1: fp8
// dims [256, 448) and bf16 rope [448, 512)), dequantized in place in the KV
// stage.
CUTE_HOST_DEVICE constexpr int raw_pitch(int cta) {
    return cta == 0 ? kHalfDims : kRowBytes - kHalfDims;
}
constexpr int kNopeBytes1 = kNope - kHalfDims;
// Q and O move between global memory and the Q/O warp group's registers in
// swizzled boxes of 64 dims x the 64 heads of a token.
constexpr int kQBoxDims = 64;
constexpr int kQBoxBytes = kHeads * kQBoxDims * sizeof(__nv_bfloat16);
constexpr int kChunksPerBox = kQBoxDims * sizeof(__nv_bfloat16) / kChunk;
constexpr int kQoRounds = kHalfDims / kQBoxDims;

// kQst Q buffers: a unit's four Q rounds alternate between two buffers, so the
// next unit's Q loads while this unit's O is stored.
template <bool kQst>
struct QBufs {};
template <>
struct QBufs<true> {
    alignas(kAtomBytes)
        uint8_t q_stage[2][2][kQBoxBytes];  // [round % 2][dims half]
    uint64_t qb_full[2];
};

// kQnr (fused q_norm_rope): q is the raw projection. The Q/O warp group applies
// RoPE in registers and computes each head's 1 / rms, which the softmax folds
// into the score scale (qo_norm_rope).
constexpr int kQRopeDims = kDim - kNope;

template <bool kQst>
struct PairMlaSmemT {
    alignas(kAtomBytes) uint8_t kv[kKvStages][kStageBytes];
    alignas(kAtomBytes) uint8_t p[2][kHeads * kRowPitch];
    alignas(kAtomBytes) uint8_t qo_stage[2][kQBoxBytes];  // [dims half] Q/O box
    QBufs<kQst> qb;
    alignas(kChunk) float xchg[kWarpGroup][kHalf];
    alignas(kChunk) float qfreq[kQRopeDims];  // kQnr: the unit's RoPE freqs
    float rs[2][kHeads];  // kQnr: scale_log2 / rms per head and unit parity
    alignas(kChunk) uint8_t sf[kKvStages][kTile][kSfBytes];
    // bit j of word w: key 32w + j is valid for this CTA's token / has a row
    uint32_t valid[kIdxStages][kIndexWarps];
    uint32_t any[kIdxStages][kIndexWarps];
    int bf[kIdxStages];  // the tile was gathered from the bf16 c128 copy
    int idx_raw[kIndexWarps][kIndexDepth][kWarp][2];
    float red[kWarpGroup];
    float2 ml[2][kWarpGroup];     // per unit parity: row max, partial row sum
    int start[3];                 // first pair, its first tile, tile count
    int4 unit_lens[kTableUnits];  // lengths of the range's pairs from start[0]
    int wstart;  // stream position of the range's first pair's first tile
    int sched;   // clusters actually scheduled this launch (<= p.clusters)
    int dense;   // every scheduled cluster's range is non-empty
    int bound[kMaxClusters + 1];  // cluster c: tiles [bound[c], bound[c + 1])
    uint64_t q_full, q_empty, o_full, o_empty[2], merge_full[kMergeBuf];
    uint64_t qs_full;  // a Q round landed in qo_stage (not used by kQst)
    uint64_t ml_full[2], ml_empty[2];
    uint64_t rs_full[2];  // kQnr: rs[b] written
    uint64_t idx_full[kIdxStages], idx_empty[kIdxStages];
    uint64_t raw_full[kKvStages], kv_full[kKvStages], kv_empty[kKvStages];
    uint64_t s_full[2], s_empty[2], p_full[2], pv_done[2];
    uint32_t tmem_base;
};
constexpr int kSmemBytes = sizeof(PairMlaSmemT<false>) + kAtomBytes;
constexpr int kQstSmemBytes = sizeof(PairMlaSmemT<true>) + kAtomBytes;
static_assert(kQstSmemBytes <= 227 * 1024);

// Kernel variants: kQst adds Q buffers for short pairs; kFp8 has no bf16 c128
// copy and compiles out its load path, which slows the index warps even when
// unused.
template <bool kQstV, bool kFp8V, bool kQnrV>
struct Cfg {
    static constexpr bool kQst = kQstV;
    static constexpr bool kFp8 = kFp8V;
    static constexpr bool kQnr = kQnrV;
    using Smem = PairMlaSmemT<kQstV>;
};

struct PairMlaMaps {
    CUtensorMap raw[2][2];  // [cta][seg] 16-bit view of the rows, raw_pitch box
    CUtensorMap bf16;       // bf16 c128 copy, boxes of 64 dims x 32 rows
    CUtensorMap q;    // q rows (token x head), 64-dim swizzled boxes of 64 rows
    CUtensorMap out;  // out rows, boxes as for q
};

// A split pair for the combine kernel: first token, whether it has a second,
// first stream tile and tile count (cluster_of derives its part clusters).
struct MergeJob {
    int a, has_b, w0, wn;
};

struct PairMlaParams {
    const int* idx[2];  // [0] c128, [1] SWA: [T, ld] pool token indices
    const int* len[2];
    int ld[2];
    const uint8_t* pool[2];
    int64_t pool_stride[2];
    int page_shift[2];  // log2 of the tokens per page
    int page_rows[2];   // TMA rows (576 B) per page
    int64_t pool_tokens[2];
    // Pre-dequant (c128_base != nullptr): request r's last token's c128 list is
    // dequantized to rows [r * c128_rows, ...) of c128_bf ([requests *
    // c128_rows, 512]). c128_base[t] is that first row if token t's list is a
    // prefix of it, else -1. nullptr: the kFp8 kernels run.
    int c128_rows;  // rows per request: ld[0] rounded up to a tile
    __nv_bfloat16* c128_bf;
    int* c128_base;  // [T]
    const __nv_bfloat16* q;
    const float* sink;
    __nv_bfloat16* out;
    float* o_part;    // [2 * clusters][2][kPlanes][128][4] normalized partial O
    float* lse_part;  // [2 * clusters][2][64] their log2 sums
    int* counter;     // [clusters][2] parts published of a cluster's split pair
    int tokens;
    int tokens_per_req;  // t; tokens are paired within a request, not globally
    // Varlen (cu_pairs != nullptr): request r owns tokens [cu_seqlens[r],
    // cu_seqlens[r + 1]) and ceil(len_r / 2) pairs from cu_pairs[r];
    // tokens_per_req is unused.
    const int* cu_seqlens;
    const int* cu_pairs;
    int n_req;
    // Position gap of a pair's tokens (1, or cp_size under CP round-robin): a's
    // SWA list is b's shifted by it, a's c128 list a prefix of b's.
    int swa_shift;
    int clusters;  // clusters launched
    // Small batches (2 * pairs <= clusters): every cluster launches with
    // ~kDefaultTpc tiles (sched_clusters); every part writes its partial, the
    // merger records a MergeJob and the combine kernel merges.
    int combine;
    MergeJob* jobs;  // [<=clusters] appended by mergers when combine
    int* n_jobs;     // atomic job count when combine
    int* total_g;    // the stream's tile total, for cluster_of in combine
    int* sched_g;    // clusters scheduled (sched), for the combine kernel
    float scale_log2;
    // kQnr: q is the raw projection (token stride q_stride), maps.q a 3D map
    // (dims, head, token).
    int64_t q_stride;
    const void* q_pos;  // [T] int64 (pos64) or int32
    int pos64;
    const float* q_freqs;  // [max_pos, 64] fp32, (re, im) interleaved
    float q_eps;
};

using Bar = cutlass::arch::ClusterBarrier;

using NamedBar = cutlass::arch::NamedBarrier;

CUTE_DEVICE uint32_t smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// SW128 UMMA operand: K-major (lbo unused) or MN-major (lbo = next N atom).
CUTE_DEVICE uint64_t sw128_desc(const void* p, uint32_t lbo) {
    cute::UMMA::SmemDescriptor d;
    d.version_ = 1;
    d.layout_type_ = uint8_t(cute::UMMA::LayoutType::SWIZZLE_128B);
    d.start_address_ = smem_u32(p) >> 4;
    d.leading_byte_offset_ = lbo >> 4;
    d.stride_byte_offset_ = kAtomBytes >> 4;
    return d;
}

template <int N, cute::UMMA::Major BMajor>
CUTE_DEVICE uint32_t instr_desc() {
    return static_cast<uint32_t>(
        cute::UMMA::make_runtime_instr_desc<
            cutlass::bfloat16_t, cutlass::bfloat16_t, float, 2 * kHeads, N,
            cute::UMMA::Major::K, BMajor>() >>
        32);
}

CUTE_DEVICE void umma_ss(uint32_t d, uint64_t a, uint64_t b, uint32_t idesc,
                         bool accumulate) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n}\n" ::"r"(d),
        "l"(a), "l"(b), "r"(idesc), "r"(uint32_t(accumulate)));
}

CUTE_DEVICE void umma_ts(uint32_t d, uint32_t a, uint64_t b, uint32_t idesc,
                         bool accumulate) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], [%1], %2, %3, p;\n}\n" ::"r"(
            d),
        "r"(a), "l"(b), "r"(idesc), "r"(uint32_t(accumulate)));
}

// KV gathers: evict-first, faster than evict-normal or evict-last.
constexpr uint64_t kKvL2Hint = 0x12F0000000000000ull;
// The bf16 c128 rows are read by every pair of a request: evict-normal.
constexpr uint64_t kBf16L2Hint = 0x1000000000000000ull;
CUTE_DEVICE void tma_gather4(const CUtensorMap* map, uint64_t* bar, void* dst,
                             int col, int r0, int r1, int r2, int r3) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4.mbarrier:"
        ":complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, %3, %4, %5, %6}], "
        "[%7], %8;" ::"r"(smem_u32(dst)),
        "l"(map), "r"(col), "r"(r0), "r"(r1), "r"(r2), "r"(r3),
        "r"(smem_u32(bar)), "l"(kKvL2Hint)
        : "memory");
}

// Row r and its swizzled 16-byte chunk c in a SW128 tile of 8-row atoms.
CUTE_DEVICE int row_offset(int r) {
    return (r / kAtomRows) * kAtomBytes + (r % kAtomRows) * kRowPitch;
}
CUTE_DEVICE int sw128_offset(int r, int c) {
    return row_offset(r) + (c ^ (r % kAtomRows)) * kChunk;
}

CUTE_DEVICE uint4 dequant8(uint32_t lo, uint32_t hi, uint8_t e8) {
    const uint32_t raw[2] = {lo, hi};
    const auto* f8 = reinterpret_cast<const __nv_fp8x2_e4m3*>(raw);
    const nv_bfloat16 scale = __ushort_as_bfloat16(uint16_t(e8) << kBf16Mant);
    uint4 out;
    auto* o = reinterpret_cast<nv_bfloat162*>(&out);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kElemsPerChunk / 2; ++i) {
        o[i] = fp8x2_to_bf16x2_with_scale(f8[i], scale);
    }
    return out;
}

template <int N>
CUTE_DEVICE void tmem_ld(uint32_t addr, float (&v)[N]) {
    tmem_ld_32dp32bNx<N>(addr, v);
    cutlass::arch::fence_view_async_tmem_load();
}

CUTE_DEVICE void cp_async4(int* dst, const int* src) {
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(smem_u32(dst)),
        "l"(src)
        : "memory");
}

CUTE_DEVICE void cp_async8(void* dst, const void* src) {
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 8;" ::"r"(smem_u32(dst)),
        "l"(src)
        : "memory");
}

// Arrives on bar once this thread's earlier cp.async copies have landed; the
// barrier's expected count includes the arrival.
CUTE_DEVICE void cp_async_arrive_noinc(uint64_t* bar) {
    asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(
                     smem_u32(bar))
                 : "memory");
}

// Key stream of a = 2j, b = 2j+1: c128 position u is b's row u (a's list is a
// prefix of b's) or a's where b has none, then b's SWA window and the oldest
// p.swa_shift rows of a. A row counts for a token only if its own list has it
// there, so other inputs are masked, never read out of bounds.
struct PairGeom {
    int a, len[2][2], n_c4, n_s, n_tiles;  // len[seg][token of pair]
    bool has_b;  // false for a lone (odd-request) token
};

// The lengths of a pair's tokens, [seg][token].
struct PairLens {
    int v[2][2];
};

// A pair's first token and whether it has a second. Pairs never straddle a
// request; an odd request's last token is a lone pair, whose CTA 1 has no
// token.
struct PairMap {
    int a;
    bool has_b;
};

CUTE_DEVICE PairMap pair_map(const PairMlaParams& p, int pair) {
    PairMap m;
    if (p.cu_pairs) {
        // Last request r with cu_pairs[r] <= pair (empty requests are skipped).
        int lo = 0, hi = p.n_req;
        while (hi - lo > 1) {
            const int mid = (lo + hi) / 2;
            if (__ldg(p.cu_pairs + mid) <= pair)
                lo = mid;
            else
                hi = mid;
        }
        m.a = __ldg(p.cu_seqlens + lo) + 2 * (pair - __ldg(p.cu_pairs + lo));
        m.has_b = m.a + 1 < __ldg(p.cu_seqlens + lo + 1);
        return m;
    }
    const int ppr = (p.tokens_per_req + 1) / 2;  // pairs per request
    const int r = pair / ppr, k = pair - r * ppr;
    m.a = r * p.tokens_per_req + 2 * k;
    m.has_b = 2 * k + 1 < p.tokens_per_req;
    return m;
}

CUTE_DEVICE int num_pairs(const PairMlaParams& p) {
    if (p.cu_pairs) return __ldg(p.cu_pairs + p.n_req);
    return (p.tokens / p.tokens_per_req) * ((p.tokens_per_req + 1) / 2);
}

CUTE_DEVICE PairLens load_lens(const PairMlaParams& p, int pair) {
    const PairMap m = pair_map(p, pair);
    PairLens l;
    CUTLASS_PRAGMA_UNROLL
    for (int seg = 0; seg < 2; ++seg) {
        l.v[seg][0] = __ldg(p.len[seg] + m.a);
        l.v[seg][1] = m.has_b ? __ldg(p.len[seg] + m.a + 1) : 0;
    }
    return l;
}

CUTE_DEVICE PairGeom pair_geom(const PairMlaParams& p, int pair,
                               const PairLens& l) {
    PairGeom g;
    const PairMap m = pair_map(p, pair);
    g.a = m.a;
    g.has_b = m.has_b;
    CUTLASS_PRAGMA_UNROLL
    for (int seg = 0; seg < 2; ++seg) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < 2; ++i) {
            g.len[seg][i] = max(min(l.v[seg][i], p.ld[seg]), 0);
        }
    }
    const int n_c = max(g.len[0][0], g.len[0][1]);
    g.n_c4 = (n_c + kGather - 1) / kGather * kGather;
    g.n_s = max(g.len[1][1], g.len[1][0] > 0 ? g.len[1][0] + p.swa_shift : 0);
    g.n_tiles = max((g.n_c4 + g.n_s + kTile - 1) / kTile, 1);
    return g;
}

// Persistent schedule: cluster c takes stream tiles [bound[c], bound[c + 1])
// (find_range). A unit is one pair's part in a range; the later parts of a
// split pair write partial O and log2 sums to their cluster's slot, merged by
// the first part (store_o) or the combine kernel.
CUTE_DEVICE int range_lo(int c, int total, int clusters) {
    return int(uint64_t(c) * uint64_t(total) / uint64_t(clusters));
}

constexpr int kDefaultTpc = 3;  // tiles per scheduled cluster when combine
constexpr int kQstMaxTiles =
    5;  // pairs of at most this many tiles run the kQst kernels

// Schedule cost of an n-tile pair, in quarter tiles: its tiles plus a fixed
// cost (Q load, pipeline fill and drain, O store). The kQst kernels pay it in
// full; else the KV stages hide it except below kPairFix / 3 tiles. A pair
// costing 4 n is long and may be cut by a range boundary.
constexpr int kPairFixQst = 24;
constexpr int kPairFix = 24;
constexpr int kTileCost = 4;
template <class C>
CUTE_DEVICE int pair_cost(int n) {
    return C::kQst ? kTileCost * n + kPairFixQst
                   : max(kTileCost * n, kPairFix + n);
}

// Clusters to schedule. With p.combine every cluster launches and ~kDefaultTpc
// tiles go to each: fewer tiles fragment pairs into sub-tile parts and slow
// short-KV batches. Clusters past the count get an empty range and exit.
CUTE_DEVICE int sched_clusters(const PairMlaParams& p, int total) {
    if (!p.combine) return p.clusters;
    const int s = (total + kDefaultTpc - 1) / kDefaultTpc;
    return max(1, min(s, p.clusters));
}

struct Unit {
    PairGeom g;
    int off, n;  // tiles [off, off + n) of the pair
    int w0, wn;  // the pair's first stream tile and its tile count
    bool split;
};

struct UnitCursor {
    int pair, off, left, wpair0, first_pair;
    CUTE_DEVICE bool next(const PairMlaParams& p, const int4* table, Unit& u) {
        if (left == 0) return false;
        PairLens lens;
        const int rel = pair - first_pair;
        if (rel < kTableUnits) {
            const int4 v = table[rel];
            lens = {{{v.x, v.y}, {v.z, v.w}}};
        } else {
            lens = load_lens(p, pair);
        }
        u.g = pair_geom(p, pair, lens);
        u.off = off;
        u.n = min(u.g.n_tiles - off, left);
        u.w0 = wpair0;
        u.wn = u.g.n_tiles;
        u.split = u.n != u.g.n_tiles;
        left -= u.n;
        off += u.n;
        if (off == u.g.n_tiles) {
            ++pair;
            off = 0;
            wpair0 += u.g.n_tiles;
        }
        return true;
    }
};

// All threads count and scan the tiles of every pair to find this cluster's
// first pair and tile.
template <class C>
CUTE_DEVICE void find_range(const PairMlaParams& p, typename C::Smem& sm,
                            int cluster) {
    __shared__ int part[kThreads / kWarp], cpart[kThreads / kWarp];
    const int pairs = num_pairs(p);
    const int per = (pairs + kThreads - 1) / kThreads;
    const int first = threadIdx.x * per;
    const int lane = threadIdx.x % kWarp, warp = threadIdx.x / kWarp;
    // The lengths of a thread's first kKeep pairs are kept for the table.
    constexpr int kKeep = 2;
    int4 keep[kKeep];
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < kKeep; ++j) keep[j] = make_int4(0, 0, 0, 0);

    // pair_cost is for the in-kernel merge only; combine uses the even split.
    bool by_cost = !p.combine;
    // Pass 1: the tile total and the cost of this thread's pairs.
    int sum = 0, csum = 0;
    for (int q = first, j = 0; q < min(first + per, pairs); ++q, ++j) {
        const PairLens l = load_lens(p, q);
        CUTLASS_PRAGMA_UNROLL
        for (int k = 0; k < kKeep; ++k) {
            if (j == k)
                keep[k] = make_int4(l.v[0][0], l.v[0][1], l.v[1][0], l.v[1][1]);
        }
        const int n = pair_geom(p, q, l).n_tiles;
        sum += n;
        csum += by_cost ? pair_cost<C>(n) : kTileCost * n;
    }
    {
        int v = sum, cv = csum;
        CUTLASS_PRAGMA_UNROLL
        for (int d = kWarp / 2; d > 0; d /= 2) {
            v += __shfl_xor_sync(~0u, v, d);
            cv += __shfl_xor_sync(~0u, cv, d);
        }
        if (lane == 0) {
            part[warp] = v;
            cpart[warp] = cv;
        }
    }
    __syncthreads();
    int tiles = 0, ctiles = 0;
    for (int w = 0; w < kThreads / kWarp; ++w) {
        tiles += part[w];
        ctiles += cpart[w];
    }
    // Every pair long (long contexts): the cost is the tiles.
    if (ctiles == kTileCost * tiles) by_cost = false;
    __syncthreads();
    const int sched = sched_clusters(p, tiles);
    if (threadIdx.x == 0) {
        sm.sched = sched;
        if (p.combine && cluster == 0) *p.sched_g = sched;
    }

    // Pass 2: scan this thread's pair tiles to their stream positions.
    int incl = sum;
    CUTLASS_PRAGMA_UNROLL
    for (int d = 1; d < kWarp; d *= 2) {
        const int v = __shfl_up_sync(~0u, incl, d);
        if (lane >= d) incl += v;
    }
    if (lane == kWarp - 1) part[warp] = incl;
    __syncthreads();
    int base = 0, total = 0;
    for (int w = 0; w < kThreads / kWarp; ++w) {
        base += w < warp ? part[w] : 0;
        total += part[w];
    }
    int pre = base + incl - sum;
    // Boundary c sits at cost c * cost_total / sched: at the start of a short
    // pair it falls in, else at the matching tile. Balancing tiles alone would
    // give the short first pairs of a request one range with several times the
    // pairs.
    if (!by_cost) {
        for (int c = threadIdx.x; c <= sched; c += kThreads) {
            sm.bound[c] = range_lo(c, total, sched);
        }
    } else {
        int cincl = csum;
        CUTLASS_PRAGMA_UNROLL
        for (int d = 1; d < kWarp; d *= 2) {
            const int v = __shfl_up_sync(~0u, cincl, d);
            if (lane >= d) cincl += v;
        }
        __syncthreads();
        if (lane == kWarp - 1) part[warp] = cincl;
        __syncthreads();
        int cbase = 0, ctotal = 0;
        for (int w = 0; w < kThreads / kWarp; ++w) {
            cbase += w < warp ? part[w] : 0;
            ctotal += part[w];
        }
        const auto x_of = [&](int c) {
            return int(uint64_t(c) * uint64_t(ctotal) / uint64_t(sched));
        };
        if (threadIdx.x == 0) {
            sm.bound[0] = 0;
            sm.bound[sched] = total;
        }
        int cs = cbase + cincl - csum, tp = pre;
        // First boundary at or past cost cs, walked only by the thread holding
        // it.
        int c = max(int((uint64_t(cs) * sched + ctotal - 1) / ctotal), 1);
        if (csum > 0 && c < sched && x_of(c) < cs + csum) {
            for (int q = first; q < min(first + per, pairs) && c < sched; ++q) {
                const int n = pair_geom(p, q, load_lens(p, q)).n_tiles;
                const int ce = cs + pair_cost<C>(n);
                const bool lng = ce - cs == kTileCost * n;
                for (; c < sched && x_of(c) < ce; ++c) {
                    sm.bound[c] = tp + (lng ? (x_of(c) - cs) / kTileCost : 0);
                }
                cs = ce;
                tp += n;
            }
        }
    }
    __syncthreads();
    // Clusters past sched get an empty range (lo == hi == total) and exit.
    const int lo = sm.bound[min(cluster, sched)];
    const int hi = sm.bound[min(cluster + 1, sched)];
    const int empty = int(threadIdx.x) < sched &&
                      sm.bound[threadIdx.x + 1] <= sm.bound[threadIdx.x];
    if (sum > 0 && pre <= lo && lo < pre + sum) {
        for (int q = first;; ++q) {
            const int w = pair_geom(p, q, load_lens(p, q)).n_tiles;
            if (lo < pre + w) {
                sm.start[0] = q;
                sm.wstart = pre;
                sm.start[1] = lo - pre;
                sm.start[2] = hi - lo;
                break;
            }
            pre += w;
        }
    }
    if (threadIdx.x == 0 && lo == hi) {
        sm.start[0] = 0;
        sm.start[2] = 0;
    }
    const int any_empty = __syncthreads_or(empty);
    if (threadIdx.x == 0) sm.dense = !any_empty;
    if (p.combine && cluster == 0 && threadIdx.x == 0) *p.total_g = total;
    __syncthreads();
    // Lengths of the range's first pairs for every role's cursor, from the
    // threads that loaded them (or reloaded past kKeep).
    if (sm.start[2] > 0) {
        if (per <= kKeep) {
            CUTLASS_PRAGMA_UNROLL
            for (int k = 0; k < kKeep; ++k) {
                const int q = first + k, rel = q - sm.start[0];
                if (k < per && q < pairs && rel >= 0 && rel < kTableUnits) {
                    sm.unit_lens[rel] = keep[k];
                }
            }
        } else if (threadIdx.x < kTableUnits &&
                   sm.start[0] + int(threadIdx.x) < pairs) {
            const int q = sm.start[0] + threadIdx.x;
            const PairLens l = load_lens(p, q);
            sm.unit_lens[threadIdx.x] =
                make_int4(l.v[0][0], l.v[0][1], l.v[1][0], l.v[1][1]);
        }
    }
    __syncthreads();
}

template <class C>
CUTE_DEVICE UnitCursor start_cursor(const PairMlaParams& p,
                                    const typename C::Smem& sm) {
    return {sm.start[0], sm.start[1], sm.start[2], sm.wstart, sm.start[0]};
}

// Index warp w resolves keys [32w, 32w + 32) of each tile, one per lane (union
// key, validity, TMA row), gathers 8 four-row groups into the KV stage and
// copies the scale bytes by cp.async, all completing on raw_full. A tile of
// only c128 keys loads its rows from the bf16 copy by TMA instead. Raw indices
// run kIndexDepth tiles ahead.
struct KeyRaw {
    bool in_a, in_b;
    int seg;
    int bf;  // first row in the bf16 copy (all-c128 tile), else -1
};

template <class C>
CUTE_DEVICE void index_warp(const PairMlaParams& p, const PairMlaMaps& maps,
                            typename C::Smem& sm, int cta, int w, int lane) {
    const int j = w * kWarp + lane;
    UnitCursor ahead = start_cursor<C>(p, sm);
    Unit au;
    int a_left = 0, a_tile = 0, a_bf = -1;
    const auto fetch = [&](KeyRaw& r, int slot) {
        if (a_left == 0 && ahead.next(p, sm.unit_lens, au)) {
            a_left = au.n;
            a_tile = au.off;
            // Both tokens' c128 lists must be prefixes of their request's copy.
            a_bf = -1;
            if constexpr (!C::kFp8) {
                const int ba = __ldg(p.c128_base + au.g.a);
                const int bb =
                    au.g.has_b ? __ldg(p.c128_base + au.g.a + 1) : ba;
                a_bf = ba >= 0 && bb >= 0 ? ba : -1;
            }
        }
        r = {false, false, 1, -1};
        if (a_left > 0) {
            const int pos = a_tile * kTile + j;
            const int seg = pos < au.g.n_c4 ? 0 : 1;
            const int ub = seg == 0 ? pos : pos - au.g.n_c4,
                      ua = ub - seg * p.swa_shift;
            const int* row = p.idx[seg] + int64_t(au.g.a) * p.ld[seg];
            r = {ua >= 0 && ua < au.g.len[seg][0], ub < au.g.len[seg][1], seg,
                 a_bf >= 0 && (a_tile + 1) * kTile <= au.g.n_c4
                     ? a_bf + a_tile * kTile
                     : -1};
            if (r.in_a) cp_async4(&sm.idx_raw[w][slot][lane][0], row + ua);
            if (r.in_b) {
                cp_async4(&sm.idx_raw[w][slot][lane][1], row + p.ld[seg] + ub);
            }
            ++a_tile;
            --a_left;
        }
        cute::cp_async_fence();
    };
    const int n = sm.start[2];
    KeyRaw ring[kIndexDepth];
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < kIndexDepth; ++k) fetch(ring[k], k);
    const auto step = [&](int i, KeyRaw& r, int slot) {
        cute::cp_async_wait<kIndexDepth - 1>();
        __syncwarp();
        const int ka = r.in_a ? sm.idx_raw[w][slot][lane][0] : -1;
        const int kb = r.in_b ? sm.idx_raw[w][slot][lane][1] : -1;
        const int seg = r.seg;
        const bool okb = r.in_b && kb >= 0 && kb < p.pool_tokens[seg];
        const bool oka = r.in_a && ka >= 0 && ka < p.pool_tokens[seg];
        const int key = okb ? kb : ka;
        const bool va = oka && ka == key, any = okb || va;
        const int page = key >> p.page_shift[seg];
        const int row = key & ((1 << p.page_shift[seg]) - 1);
        const int coord = any ? page * p.page_rows[seg] + row : -1;
        const uint32_t valid = __ballot_sync(~0u, cta == 0 ? va : okb);
        const uint32_t has = __ballot_sync(~0u, any);
        const int is = i % kIdxStages;
        Bar::wait(&sm.idx_empty[is], ((i / kIdxStages) & 1) ^ 1);
        if (lane == 0) {
            sm.valid[is][w] = valid;
            sm.any[is][w] = has;
            if (w == 0) sm.bf[is] = !C::kFp8 && r.bf >= 0;
            Bar::arrive(&sm.idx_full[is]);
        }
        // The tile's stage is free once PV(i - kKvStages) is done.
        const int s = i % kKvStages;
        Bar::wait(&sm.kv_empty[s], ((i / kKvStages) & 1) ^ 1);
        if (!C::kFp8 && r.bf >= 0) {
            // The request's c128 rows are contiguous in the copy: four swizzled
            // box loads per warp, already in the final layout.
            cp_async_arrive_noinc(&sm.raw_full[s]);
            if (lane == 0) {
                cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(
                    &sm.raw_full[s],
                    kWarp * kHalfDims * int(sizeof(__nv_bfloat16)));
                CUTLASS_PRAGMA_UNROLL
                for (int b = 0; b < kLocalBlocks; ++b) {
                    cute::SM90_TMA_LOAD_2D::copy(
                        &maps.bf16, &sm.raw_full[s], kBf16L2Hint,
                        sm.kv[s] + b * kBlockBytes + row_offset(w * kWarp),
                        cta * kHalfDims + b * kBlockDims, r.bf + w * kWarp);
                }
            }
            __syncwarp();
        } else {
            const uint8_t* const sf =
                p.pool[seg] + int64_t(page) * p.pool_stride[seg] +
                (int64_t(1) << p.page_shift[seg]) * kRowBytes + row * kSfBytes;
            if (any) cp_async8(sm.sf[s][j], sf);
            cp_async_arrive_noinc(&sm.raw_full[s]);
            const int src = lane % kGroupsPerWarp * kGather;
            const int r0 = __shfl_sync(~0u, coord, src);
            const int r1 = __shfl_sync(~0u, coord, src + 1);
            const int r2 = __shfl_sync(~0u, coord, src + 2);
            const int r3 = __shfl_sync(~0u, coord, src + 3);
            const int gseg = __shfl_sync(~0u, seg, src);
            if (lane == 0) {
                cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(
                    &sm.raw_full[s], kWarp * raw_pitch(cta));
            }
            __syncwarp();
            if (lane < kGroupsPerWarp) {
                const CUtensorMap* const map =
                    cta == 0 ? (gseg == 0 ? &maps.raw[0][0] : &maps.raw[0][1])
                             : (gseg == 0 ? &maps.raw[1][0] : &maps.raw[1][1]);
                tma_gather4(map, &sm.raw_full[s],
                            sm.kv[s] + (w * kWarp + src) * raw_pitch(cta),
                            cta * kHalfDims / 2, r0, r1, r2, r3);
            }
            __syncwarp();
        }
        fetch(r, slot);
    };
    // One copy of the loop body (hot code): the ring rotates, not unrolled.
#pragma unroll 1
    for (int i = 0; i < n; ++i) {
        step(i, ring[0], i % kIndexDepth);
        const KeyRaw r = ring[0];
        CUTLASS_PRAGMA_UNROLL
        for (int k = 0; k + 1 < kIndexDepth; ++k) ring[k] = ring[k + 1];
        ring[kIndexDepth - 1] = r;
    }
}

// ---- kQnr: q_norm_rope in the Q/O warp group -------------------------------

// q_norm_rope float ops in PTX: --use_fast_math would flush denormals and
// contract products, unlike the reference (sglang JIT, no fast math).
CUTE_DEVICE float fma_rn(float a, float b, float c) {
    float d;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(d) : "f"(a), "f"(b), "f"(c));
    return d;
}

CUTE_DEVICE float mul_rn(float a, float b) {
    float d;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
    return d;
}

CUTE_DEVICE float add_rn(float a, float b) {
    float d;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
    return d;
}

CUTE_DEVICE float bf16_lo(uint32_t w) { return __uint_as_float(w << 16); }
CUTE_DEVICE float bf16_hi(uint32_t w) {
    return __uint_as_float(w & 0xffff0000u);
}

CUTE_DEVICE uint32_t pack_bf16(float lo, float hi) {
    uint32_t d;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo));
    return d;
}

// s + x * x for the low (H = 0) or high bf16 of w, as one exact FFMA.
template <int H>
CUTE_DEVICE float square_add(uint32_t w, float s) {
    float d;
    if constexpr (H == 0) {
        asm("{\n\t.reg .b16 l, h;\n\tmov.b32 {l, h}, %1;\n\t"
            "fma.rn.f32.bf16 %0, l, l, %2;\n\t}"
            : "=f"(d)
            : "r"(w), "f"(s));
    } else {
        asm("{\n\t.reg .b16 l, h;\n\tmov.b32 {l, h}, %1;\n\t"
            "fma.rn.f32.bf16 %0, h, h, %2;\n\t}"
            : "=f"(d)
            : "r"(w), "f"(s));
    }
    return d;
}

// A 16-byte chunk's squares added onto s in dims order.
CUTE_DEVICE float square_chain(const uint4& c, float s) {
    const uint32_t w[4] = {c.x, c.y, c.z, c.w};
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < 4; ++k) s = square_add<1>(w[k], square_add<0>(w[k], s));
    return s;
}

// (xr, xi) * (fr, fi), rounded as the reference: {fma(xr, fr, -(xi * fi)),
// fma(xr, fi, xi * fr)}.
CUTE_DEVICE void rope_pair(uint32_t& w, float fr, float fi) {
    const float xr = bf16_lo(w), xi = bf16_hi(w);
    float re;
    asm("{\n\t.reg .f32 m;\n\tmul.rn.f32 m, %1, %2;\n\tneg.f32 m, m;\n\t"
        "fma.rn.f32 %0, %3, %4, m;\n\t}"
        : "=f"(re)
        : "f"(xi), "f"(fi), "f"(xr), "f"(fr));
    w = pack_bf16(re, fma_rn(xr, fi, mul_rn(xi, fr)));
}

// Thread t of the dequant warp group converts chunks t + 128k of the rows in
// place: all reads, a barrier, then the writes. Odd 64-dim blocks store their
// bf16 halves swapped, so stores hit 8 distinct bank groups.
template <int kCta>
struct DequantPlan {
    static constexpr int kFp8Chunks =
        (kCta == 0 ? kHalfDims : kNopeBytes1) / kChunk;
    static constexpr int kFp8Loads = kTile * kFp8Chunks / kWarpGroup;
    static constexpr int kRopeChunks =
        kCta == 0 ? 0 : (kRowBytes - kNope) / kChunk;
    static constexpr int kRopeLoads = kTile * kRopeChunks / kWarpGroup;
};

template <int kCta, class C>
CUTE_DEVICE void dequant_wg(typename C::Smem& sm, int n, int t) {
    using Plan = DequantPlan<kCta>;
    constexpr int kPitch = raw_pitch(kCta);
    for (int i = 0; i < n; ++i) {
        const int s = i % kKvStages, is = i % kIdxStages;
        Bar::wait(&sm.idx_full[is], (i / kIdxStages) & 1);
        const uint32_t any[2] = {sm.any[is][0], sm.any[is][1]};
        const bool bf = !C::kFp8 && sm.bf[is];
        Bar::wait(&sm.raw_full[s], (i / kKvStages) & 1);
        // A bf16 tile arrives in its final layout: only pass it on.
        if (bf) {
            __syncwarp();
            if (t % kWarp == 0) Bar::arrive(&sm.idx_empty[is]);
            if (t % kWarp == 0) Bar::arrive(&sm.kv_full[s], 0, 1);
            continue;
        }
        uint8_t* const stage = sm.kv[s];
        uint4 raw[Plan::kFp8Loads];
        uint8_t e8[Plan::kFp8Loads];
        CUTLASS_PRAGMA_UNROLL
        for (int k = 0; k < Plan::kFp8Loads; ++k) {
            const int e = t + k * kWarpGroup;
            const int row = e / Plan::kFp8Chunks, c = e % Plan::kFp8Chunks;
            raw[k] = *reinterpret_cast<const uint4*>(stage + row * kPitch +
                                                     c * kChunk);
            const uint8_t sf = sm.sf[s][row][kCta * kLocalBlocks + c / 4];
            e8[k] = (any[row / kWarp] >> (row % kWarp)) & 1 ? sf : 0;
        }
        // Only CTA 1 holds rope dims; the max() keeps CTA 0's dead code legal.
        constexpr int kRopeChunks = cute::max(Plan::kRopeChunks, 1);
        uint4 rope[cute::max(Plan::kRopeLoads, 1)];
        CUTLASS_PRAGMA_UNROLL
        for (int k = 0; k < Plan::kRopeLoads; ++k) {
            const int e = t + k * kWarpGroup;
            const int row = e / kRopeChunks, c = e % kRopeChunks;
            rope[k] = *reinterpret_cast<const uint4*>(stage + row * kPitch +
                                                      kNopeBytes1 + c * kChunk);
        }
        __syncwarp();
        if (t % kWarp == 0) Bar::arrive(&sm.idx_empty[is]);
        NamedBar::sync(kWarpGroup, kDequantBar);
        CUTLASS_PRAGMA_UNROLL
        for (int k = 0; k < Plan::kFp8Loads; ++k) {
            const int e = t + k * kWarpGroup;
            const int row = e / Plan::kFp8Chunks, c = e % Plan::kFp8Chunks;
            const bool odd = (c / 4) & 1;
            const int q0 = c % 4 * 2;
            uint8_t* const block = stage + c / 4 * kBlockBytes;
            const uint4 first = dequant8(odd ? raw[k].z : raw[k].x,
                                         odd ? raw[k].w : raw[k].y, e8[k]);
            const uint4 second = dequant8(odd ? raw[k].x : raw[k].z,
                                          odd ? raw[k].y : raw[k].w, e8[k]);
            *reinterpret_cast<uint4*>(block + sw128_offset(row, q0 + odd)) =
                first;
            *reinterpret_cast<uint4*>(block + sw128_offset(row, q0 + !odd)) =
                second;
        }
        CUTLASS_PRAGMA_UNROLL
        for (int k = 0; k < Plan::kRopeLoads; ++k) {
            const int e = t + k * kWarpGroup;
            const int row = e / kRopeChunks, c = e % kRopeChunks;
            *reinterpret_cast<uint4*>(stage + (kLocalBlocks - 1) * kBlockBytes +
                                      sw128_offset(row, c)) = rope[k];
        }
        cutlass::arch::fence_view_async_shared();
        __syncwarp();
        if (t % kWarp == 0) Bar::arrive(&sm.kv_full[s], 0, 1);
    }
}

// QK^T and PV issue from separate warps, so a late KV stage never delays a
// ready PV and a late P never delays the next QK^T. QK^T(i) writes S buffer i&1
// once its stage is dequantized and S(i - 2) is read.
template <class C>
CUTE_DEVICE void qk_warp(const PairMlaParams& p, typename C::Smem& sm,
                         uint32_t tmem) {
    const uint32_t idesc = instr_desc<2 * kTile, cute::UMMA::Major::K>();
    UnitCursor cur = start_cursor<C>(p, sm);
    Unit u;
    for (int i = 0, unit = 0; cur.next(p, sm.unit_lens, u); ++unit) {
        Bar::wait(&sm.q_full, unit & 1);
        for (int k = 0; k < u.n; ++k, ++i) {
            const int s = i % kKvStages, sb = i % 2;
            Bar::wait(&sm.kv_full[s], (i / kKvStages) & 1);
            Bar::wait(&sm.s_empty[sb], ((i / 2) & 1) ^ 1);
            tcgen05_after_thread_sync();
            if (cute::elect_one_sync()) {
                CUTLASS_PRAGMA_UNROLL
                for (int ks = 0; ks < kQkSteps; ++ks) {
                    umma_ts(tmem + kTmemS + sb * kTmemSCols,
                            tmem + kTmemQ + ks * kUmmaK / 2,
                            sw128_desc(sm.kv[s] +
                                           ks / kKStepsPerBlock * kBlockBytes +
                                           ks % kKStepsPerBlock * kKStepBytes,
                                       kChunk),
                            idesc, ks > 0);
                }
            }
            cutlass::arch::umma_arrive_multicast_2x1SM(&sm.s_full[sb],
                                                       kBothCtas);
            if (k == u.n - 1) {
                cutlass::arch::umma_arrive_multicast_2x1SM(&sm.q_empty,
                                                           kBothCtas);
            }
        }
    }
}

// PV(i) once P(i) is written; it frees the KV stage and P buffer. A unit's
// first PV overwrites O, each 128-column half once the previous unit has read
// it.
template <class C>
CUTE_DEVICE void pv_warp(const PairMlaParams& p, typename C::Smem& sm,
                         uint32_t tmem) {
    const uint32_t idesc = instr_desc<kPvN, cute::UMMA::Major::MN>();
    UnitCursor cur = start_cursor<C>(p, sm);
    Unit u;
    for (int i = 0, unit = 0; cur.next(p, sm.unit_lens, u); ++unit) {
        for (int k = 0; k < u.n; ++k, ++i) {
            const int s = i % kKvStages, pb = i % 2;
            Bar::wait(&sm.p_full[pb], (i / 2) & 1);
            CUTLASS_PRAGMA_UNROLL
            for (int m = 0; m < kPvMmas; ++m) {
                if (k == 0 && unit > 0)
                    Bar::wait(&sm.o_empty[m], (unit - 1) & 1);
                tcgen05_after_thread_sync();
                if (cute::elect_one_sync()) {
                    CUTLASS_PRAGMA_UNROLL
                    for (int ks = 0; ks < kPvSteps; ++ks) {
                        umma_ss(
                            tmem + kTmemO + m * kPvN / 2,
                            sw128_desc(sm.p[pb] + ks * kKStepBytes, kChunk),
                            sw128_desc(sm.kv[s] +
                                           m * kPvBlocksPerMma * kBlockBytes +
                                           ks * kUmmaK * kRowPitch,
                                       kBlockBytes),
                            idesc, k > 0 || ks > 0);
                    }
                }
                __syncwarp();
            }
            cutlass::arch::umma_arrive_multicast_2x1SM(&sm.pv_done[pb],
                                                       kBothCtas);
            cutlass::arch::umma_arrive_multicast_2x1SM(&sm.kv_empty[s],
                                                       kBothCtas);
            if (k == u.n - 1) {
                cutlass::arch::umma_arrive_multicast_2x1SM(&sm.o_full,
                                                           kBothCtas);
            }
        }
    }
}

// Softmax thread t owns TMEM lane t (head t % 64, dims half t / 64). Per tile
// it adds the partner's (t ^ 64, warp w ^ 2) partial S of its 32 keys, runs the
// online softmax and exchanges the row max.
CUTE_DEVICE void pair_sync(int t) {
    NamedBar::sync(2 * kWarp, kPairBar + t / kWarp % 2);
}

template <class C>
CUTE_DEVICE void wait_pv(typename C::Smem& sm, int j) {
    Bar::wait(&sm.pv_done[j % 2], (j / 2) & 1);
}

// Online softmax of tile i: send the partner's partial S, read own keys, then
// row max (with the partner's), P and row sum. O is rescaled when any lane's
// row max grows by more than 2^kRescaleLog2.
template <class C>
CUTE_DEVICE void softmax_tile(const PairMlaParams& p, typename C::Smem& sm,
                              int i, int k, uint32_t lane_base, int t, int b,
                              float& m, float& l) {
    const int head = t % kHeads, half = t / kHeads;
    // kQnr: scores also scale by the head's 1 / rms, read per tile (a register
    // across the unit spills).
    const float scale_log2 = C::kQnr ? sm.rs[b][head] : p.scale_log2;
    const int is = i % kIdxStages, sb = i % 2;
    constexpr int kQuads = kHalf / 4;
    // Test the index stage, S and PV(i - 2) back to back: each phase check is a
    // shared-memory round trip, nearly always satisfied.
    const uint32_t ph_idx = (i / kIdxStages) & 1, ph_s = (i / 2) & 1;
    const bool idx_ok = Bar::test_wait(&sm.idx_full[is], ph_idx, 1);
    const bool s_ok = Bar::test_wait(&sm.s_full[sb], ph_s, 1);
    const bool pv_ok = Bar::test_wait(&sm.pv_done[sb], ph_s ^ 1, i > 1);
    if (!idx_ok) Bar::wait(&sm.idx_full[is], ph_idx);
    // This thread owns keys [half*kHalf, +kHalf) of its lane; valid word `half`
    // holds them, bit c for key half*kHalf + c.
    static_assert(kHalf == kWarp, "one valid word per half");
    const uint32_t valid = sm.valid[is][half];
    if (!s_ok) Bar::wait(&sm.s_full[sb], ph_s);
    tcgen05_after_thread_sync();
    // xchg rows are swizzled by chunk to spread a warp's accesses over all
    // banks.
    float4* const peer = reinterpret_cast<float4*>(sm.xchg[t ^ kHeads]);
    // Both halves of the lane's S in one TMEM round trip.
    float own[kHalf], other[kHalf];
    tmem_ld_32dp32bNx<kHalf>(
        lane_base + kTmemS + sb * kTmemSCols + (half ^ 1) * kHalf, other);
    tmem_ld_32dp32bNx<kHalf>(
        lane_base + kTmemS + sb * kTmemSCols + half * kHalf, own);
    cutlass::arch::fence_view_async_tmem_load();
    CUTLASS_PRAGMA_UNROLL
    for (int q = 0; q < kQuads; ++q) {
        peer[q ^ (t % kQuads)] = make_float4(
            other[4 * q], other[4 * q + 1], other[4 * q + 2], other[4 * q + 3]);
    }
    tcgen05_before_thread_sync();
    __syncwarp();
    if (t % kWarp == 0) {
        Bar::arrive(&sm.s_empty[sb], 0, 1);
        Bar::arrive(&sm.idx_empty[is]);
    }
    pair_sync(t);
    const float4* const got = reinterpret_cast<const float4*>(sm.xchg[t]);
    // Partials are added in pairs (f32x2). valid is warp-uniform and nearly
    // always full, so the mask is then skipped.
    float2* const own2 = reinterpret_cast<float2*>(own);
    CUTLASS_PRAGMA_UNROLL
    for (int q = 0; q < kQuads; ++q) {
        const float4 v = got[q ^ ((t ^ kHeads) % kQuads)];
        own2[2 * q] = __fadd2_rn(own2[2 * q], make_float2(v.x, v.y));
        own2[2 * q + 1] = __fadd2_rn(own2[2 * q + 1], make_float2(v.z, v.w));
    }
    if (valid != ~0u) {
        CUTLASS_PRAGMA_UNROLL
        for (int c = 0; c < kHalf; ++c) {
            if (!((valid >> c) & 1)) own[c] = -CUDART_INF_F;
        }
    }
    // A max tree (max is exact in any order) instead of a serial chain.
    float mx[kHalf / 2];
    CUTLASS_PRAGMA_UNROLL
    for (int c = 0; c < kHalf / 2; ++c)
        mx[c] = fmaxf(own[2 * c], own[2 * c + 1]);
    CUTLASS_PRAGMA_UNROLL
    for (int w = kHalf / 4; w >= 1; w /= 2) {
        CUTLASS_PRAGMA_UNROLL
        for (int c = 0; c < w; ++c) mx[c] = fmaxf(mx[c], mx[c + w]);
    }
    float local = mx[0];
    // Scores are in log2 units after the scale; max commutes with it.
    local *= scale_log2;
    sm.red[t] = local;
    pair_sync(t);
    const float cur = fmaxf(local, sm.red[t ^ kHeads]);
    float factor = 1.0f;
    // Warp-wide rescale matches FlashMLA's error; per-row leaves stale maxima
    // and rounds the largest P.
    if (__any_sync(~0u, cur - m > kRescaleLog2)) {
        const float m_new = fmaxf(m, cur);
        factor = exp2f(m - m_new);
        m = m_new;
    }
    float2 sum = make_float2(0.0f, 0.0f);
    uint32_t pk[kHalf / 2];
    const float2 scale2 = make_float2(scale_log2, scale_log2),
                 neg_m = make_float2(-m, -m);
    CUTLASS_PRAGMA_UNROLL
    for (int c = 0; c < kHalf; c += 2) {
        const float2 x = __ffma2_rn(own2[c / 2], scale2, neg_m);
        const float2 e = make_float2(exp2f(x.x), exp2f(x.y));
        sum = __fadd2_rn(sum, e);
        const nv_bfloat162 b2 = __floats2bfloat162_rn(e.x, e.y);
        pk[c / 2] = reinterpret_cast<const uint32_t&>(b2);
    }
    l = l * factor + (sum.x + sum.y);
    // O may be rescaled only after PV(i - 1).
    if (k > 0 && __any_sync(~0u, factor != 1.0f)) {
        wait_pv<C>(sm, i - 1);
        tcgen05_after_thread_sync();
        CUTLASS_PRAGMA_UNROLL
        for (int c0 = 0; c0 < kHalfDims; c0 += kTmemLd) {
            float o[kTmemLd];
            tmem_ld(lane_base + kTmemO + c0, o);
            CUTLASS_PRAGMA_UNROLL
            for (int c = 0; c < kTmemLd; ++c) o[c] *= factor;
            tmem_st_32dp32bNx<kTmemLd>(lane_base + kTmemO + c0, o);
        }
        cutlass::arch::fence_view_async_tmem_store();
    }
    // P buffer i % 2 is free once PV(i - 2) is done.
    if (i > 1 && !pv_ok) wait_pv<C>(sm, i - 2);
    CUTLASS_PRAGMA_UNROLL
    for (int q = 0; q < kHalf / kElemsPerChunk; ++q) {
        *reinterpret_cast<uint4*>(
            sm.p[sb] + sw128_offset(head, half * kHalf / kElemsPerChunk + q)) =
            reinterpret_cast<const uint4*>(pk)[q];
    }
    cutlass::arch::fence_view_async_shared();
    tcgen05_before_thread_sync();
    __syncwarp();
    if (t % kWarp == 0) Bar::arrive(&sm.p_full[sb], 0, 1);
}

// Per unit: online softmax over its tiles, then row max and sum to the Q/O warp
// group.
template <class C>
CUTE_DEVICE void softmax_wg(const PairMlaParams& p, typename C::Smem& sm,
                            uint32_t tmem, int t) {
    const uint32_t lane_base = tmem + ((t / kWarp * kWarp) << 16);
    UnitCursor cur = start_cursor<C>(p, sm);
    Unit u;
    for (int i = 0, unit = 0; cur.next(p, sm.unit_lens, u); ++unit) {
        const int b = unit % 2;
        if constexpr (C::kQnr) Bar::wait(&sm.rs_full[b], (unit / 2) & 1);
        float m = kMaxInit, l = 0.0f;
        for (int k = 0; k < u.n; ++k, ++i) {
            softmax_tile<C>(p, sm, i, k, lane_base, t, b, m, l);
        }
        Bar::wait(&sm.ml_empty[b], ((unit / 2) & 1) ^ 1);
        sm.ml[b][t] = make_float2(m, l);
        __syncwarp();
        if (t % kWarp == 0) Bar::arrive(&sm.ml_full[b]);
    }
}

// Q/O thread t also owns TMEM lane t. It writes a unit's Q slice to TMEM once
// the previous unit's last QK^T is done, reads out the previous O, then loads
// the next Q into registers.
constexpr int kQVecs = kHalfDims * sizeof(__nv_bfloat16) / sizeof(uint4);

CUTE_DEVICE void tma_load_q_2d(const CUtensorMap* map, uint64_t* bar, void* dst,
                               int col, int row) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::"
        "complete_tx::bytes [%0], [%1, {%2, %3}], [%4];" ::"r"(smem_u32(dst)),
        "l"(map), "r"(col), "r"(row), "r"(smem_u32(bar))
        : "memory");
}

CUTE_DEVICE void tma_load_q_3d(const CUtensorMap* map, uint64_t* bar, void* dst,
                               int c0, int c1, int c2) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::"
        "complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];" ::"r"(
            smem_u32(dst)),
        "l"(map), "r"(c0), "r"(c1), "r"(c2), "r"(smem_u32(bar))
        : "memory");
}

// Box col of Q row row: token * 64 (contiguous q), or the token (kQnr 3D map).
template <class C>
CUTE_DEVICE void load_q_box(const PairMlaMaps& maps, uint64_t* bar, void* dst,
                            int col, int row) {
    if constexpr (C::kQnr) {
        tma_load_q_3d(&maps.q, bar, dst, col, 0, row);
    } else {
        tma_load_q_2d(&maps.q, bar, dst, col, row);
    }
}

// A lane's row address in a 128-byte swizzled box, swizzle xor'ed in: chunk Q
// is at row_x ^ 16Q. Xor per access; precomputed addresses cost 8 registers.
CUTE_DEVICE uint32_t swizzled_row(const void* box, int row) {
    return (smem_u32(box) + row * kRowPitch) | (row % kAtomRows) * kChunk;
}

template <int Q>
CUTE_DEVICE uint4 lds_chunk(uint32_t row_x) {
    uint4 v;
    asm volatile(
        "{\n\t.reg .b32 a;\n\txor.b32 a, %4, %5;\n\t"
        "ld.shared.v4.u32 {%0, %1, %2, %3}, [a];\n\t}"
        : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
        : "r"(row_x), "n"(Q * kChunk)
        : "memory");
    return v;
}

template <int Q>
CUTE_DEVICE void sts_chunk(uint32_t row_x, uint32_t a, uint32_t b, uint32_t c,
                           uint32_t d) {
    asm volatile(
        "{\n\t.reg .b32 a;\n\txor.b32 a, %0, %1;\n\t"
        "st.shared.v4.u32 [a], {%2, %3, %4, %5};\n\t}" ::"r"(row_x),
        "n"(Q * kChunk), "r"(a), "r"(b), "r"(c), "r"(d)
        : "memory");
}

// Q/O named barrier on a value just loaded from shared memory, so a later TMA
// write to that source cannot overtake the loads (bar.sync alone does not order
// them before the async proxy).
CUTE_DEVICE uint32_t xor_words(const uint4* v, int n) {
    uint32_t x = 0;
    for (int i = 0; i < n; ++i) x ^= v[i].x ^ v[i].w;
    return x;
}

CUTE_DEVICE void qo_sync_after(uint32_t x) {
    constexpr uint32_t kId =
        kQoBar +
        uint32_t(cutlass::arch::ReservedNamedBarriers::FirstUserBarrier);
    asm volatile(
        "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %0, 0;\n\t"
        "bar.red.or.pred p, %1, %2, p;\n\t}" ::"r"(x),
        "n"(kId), "n"(kWarpGroup)
        : "memory");
}

// Later units' Q come by TMA, one box per dims half: direct loads of strided
// rows fill L1 and stall the softmax.
template <class C>
CUTE_DEVICE void fetch_q(const PairMlaMaps& maps, typename C::Smem& sm, int row,
                         int t, uint4 (&v)[kQVecs]) {
    CUTLASS_PRAGMA_UNROLL
    for (int r = 0; r < kQoRounds; ++r) {
        if (t == 0) {
            cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(
                &sm.qs_full, 2 * kQBoxBytes);
            CUTLASS_PRAGMA_UNROLL
            for (int h = 0; h < 2; ++h) {
                load_q_box<C>(maps, &sm.qs_full, sm.qo_stage[h],
                              (h * kQoRounds + r) * kQBoxDims, row);
            }
        }
        Bar::wait(&sm.qs_full, r & 1);
        const uint32_t srow = swizzled_row(sm.qo_stage[t / kHeads], t % kHeads);
        cute::for_each(cute::make_int_sequence<kChunksPerBox>{}, [&](auto q) {
            v[r * kChunksPerBox + q] = lds_chunk<decltype(q)::value>(srow);
        });
        // Every lane has read the boxes before the next ones land.
        qo_sync_after(xor_words(v + r * kChunksPerBox, kChunksPerBox));
    }
}

// kQst: round r of a token's Q (two 64-dim boxes) into Q buffer r % 2.
template <class C>
CUTE_DEVICE void issue_q(const PairMlaMaps& maps, typename C::Smem& sm, int row,
                         int r) {
    uint64_t* const bar = &sm.qb.qb_full[r % 2];
    cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(
        bar, 2 * kQBoxBytes);
    CUTLASS_PRAGMA_UNROLL
    for (int h = 0; h < 2; ++h) {
        load_q_box<C>(maps, bar, sm.qb.q_stage[r % 2][h],
                      (h * kQoRounds + r) * kQBoxDims, row);
    }
}

// fetch_q from the Q buffers: rounds 0 and 1 were issued ahead; a buffer takes
// round r + 2 once round r is read, so round r waits parity (r / 2) & 1.
static_assert(kQoRounds == 4);
template <class C>
CUTE_DEVICE void fetch_q2(const PairMlaMaps& maps, typename C::Smem& sm,
                          int row, int t, uint4 (&v)[kQVecs]) {
    CUTLASS_PRAGMA_UNROLL
    for (int r = 0; r < kQoRounds; ++r) {
        Bar::wait(&sm.qb.qb_full[r % 2], (r / 2) & 1);
        const uint32_t srow =
            swizzled_row(sm.qb.q_stage[r % 2][t / kHeads], t % kHeads);
        cute::for_each(cute::make_int_sequence<kChunksPerBox>{}, [&](auto q) {
            v[r * kChunksPerBox + q] = lds_chunk<decltype(q)::value>(srow);
        });
        qo_sync_after(xor_words(v + r * kChunksPerBox, kChunksPerBox));
        if (r + 2 < kQoRounds && t == 0) issue_q<C>(maps, sm, row, r + 2);
    }
}

// kQnr: lanes 0..15 of warp 1 copy a unit's 64 RoPE floats into sm.qfreq
// (cp.async) with its Q; they wait for them before the norm's first barrier.
constexpr int kFreqLanes = kQRopeDims * sizeof(float) / kChunk;
constexpr int kFreqT0 = kWarp;

CUTE_DEVICE bool freq_lane(int t) {
    return t >= kFreqT0 && t < kFreqT0 + kFreqLanes;
}

CUTE_DEVICE int load_pos(const PairMlaParams& p, int token) {
    return p.pos64 ? int(static_cast<const int64_t*>(p.q_pos)[token])
                   : static_cast<const int*>(p.q_pos)[token];
}

CUTE_DEVICE void load_freqs(const PairMlaParams& p, float* qfreq, int pos,
                            int t) {
    const int i = (t - kFreqT0) * 4;
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n\tcp.async.commit_group;" ::
            "r"(smem_u32(qfreq + i)),
        "l"(p.q_freqs + int64_t(pos) * kQRopeDims + i)
        : "memory");
}

// sglang fused_q_norm_rope (deepseek_v4/main_norm_rope.cuh) with the scaling
// folded into the softmax: rope(x / rms) = rope(x) / rms, so Q gets only RoPE
// (dims 448..511) and 1 / rms scales the head's scores (sm.rs).
// The sum of squares follows the reference op for op (lane l chains chunks l
// and 32 + l, then a xor butterfly): thread t starts the chains of head t,
// thread t + 64 finishes them via qo_stage, sums them in butterfly order and
// applies RoPE. x / rms is not rounded to bf16.
// Unit parity b's scales go to sm.rs[b]; the softmax has read unit - 2's before
// store_o, which precedes this.
template <class C>
CUTE_DEVICE void qo_norm_rope(const PairMlaParams& p, typename C::Smem& sm,
                              int t, int b, uint4 (&v)[kQVecs]) {
    const int head = t % kHeads;
    auto* const part =
        reinterpret_cast<float4(*)[kHeads]>(sm.qo_stage);  // [8][64]
    static_assert(kQVecs / 4 * kHeads * sizeof(float4) <= sizeof(sm.qo_stage));
    if (t < kHeads) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kQVecs / 4; ++i) {
            part[i][head] = make_float4(square_chain(v[4 * i], 0.f),
                                        square_chain(v[4 * i + 1], 0.f),
                                        square_chain(v[4 * i + 2], 0.f),
                                        square_chain(v[4 * i + 3], 0.f));
        }
    }
    if (freq_lane(t)) asm volatile("cp.async.wait_all;" ::: "memory");
    NamedBar::sync(kWarpGroup, kQoBar);
    if (t >= kHeads) {
        float c[kQVecs];
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kQVecs / 4; ++i) {
            const float4 a = part[i][head];
            c[4 * i] = square_chain(v[4 * i], a.x);
            c[4 * i + 1] = square_chain(v[4 * i + 1], a.y);
            c[4 * i + 2] = square_chain(v[4 * i + 2], a.z);
            c[4 * i + 3] = square_chain(v[4 * i + 3], a.w);
        }
        CUTLASS_PRAGMA_UNROLL
        for (int d = kQVecs / 2; d >= 1; d /= 2) {
            CUTLASS_PRAGMA_UNROLL
            for (int l = 0; l < d; ++l) c[l] = add_rn(c[l], c[l + d]);
        }
        float n;
        asm("rsqrt.approx.f32 %0, %1;"
            : "=f"(n)
            : "f"(fma_rn(c[0], 1.f / kDim, p.q_eps)));
        sm.rs[b][head] = n * p.scale_log2;
        __syncwarp();
        if (t % kWarp == 0) Bar::arrive(&sm.rs_full[b]);
        constexpr int kRope0 = kQVecs - kQRopeDims / kElemsPerChunk;
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < kQRopeDims / kElemsPerChunk; ++j) {
            const float4 f0 = reinterpret_cast<const float4*>(sm.qfreq)[2 * j];
            const float4 f1 =
                reinterpret_cast<const float4*>(sm.qfreq)[2 * j + 1];
            uint4& w = v[kRope0 + j];
            rope_pair(w.x, f0.x, f0.y);
            rope_pair(w.y, f0.z, f0.w);
            rope_pair(w.z, f1.x, f1.y);
            rope_pair(w.w, f1.z, f1.w);
        }
    }
    // qo_stage and qfreq are read before the next writes into them land.
    cutlass::arch::fence_view_async_shared();
    qo_sync_after(xor_words(v + kQVecs - 1, 1));
}

template <class C>
CUTE_DEVICE void store_q(typename C::Smem& sm, uint32_t lane_base, int t,
                         const uint4 (&v)[kQVecs]) {
    // One store per 16-byte load: the registers are used as they are.
    CUTLASS_PRAGMA_UNROLL
    for (int q = 0; q < kQVecs; ++q) {
        tmem_st_32dp32bNx<4>(lane_base + kTmemQ + 4 * q, &v[q]);
    }
    cutlass::arch::fence_view_async_tmem_store();
    tcgen05_before_thread_sync();
    __syncwarp();
    if (t % kWarp == 0) Bar::arrive(&sm.q_full, 0, 1);
}

// Long Q/O waits: one warp polls, the others sleep on a named barrier, saving
// issue slots for the other roles.
CUTE_DEVICE void qo_wait(uint64_t* bar, int phase, int t) {
    if (t < kWarp) Bar::wait(bar, phase);
    NamedBar::sync(kWarpGroup, kQoBar);
}

// A continuation part's normalized O in o_part: plane p holds columns [4p, 4p +
// 4) of every lane, lanes contiguous, so stores are 512-byte runs and the
// merger's loads are conflict-free.
constexpr int kPlaneCols = 4;
constexpr int kPlanes = kHalfDims / kPlaneCols;
constexpr int kPlaneBytes = kWarpGroup * kPlaneCols * sizeof(float);
constexpr int kPartBoxPlanes = 16;  // planes per 32 KB bulk copy
constexpr int kPartBoxes = kPlanes / kPartBoxPlanes;

CUTE_DEVICE int64_t part_row(int slot, int cta, int head) {
    return (int64_t(slot) * 2 + cta) * kHeads + head;
}

// TMEM columns of O released to the next unit's first PV at a time.
constexpr int kOHalf = kPvN / 2;
constexpr int kMergeCols = 32;  // TMEM columns per step of the merge

// Cluster of stream tile pos: the last c with c * total < (pos + 1) * clusters.
// A split pair's parts start at the cluster of its first tile.
CUTE_DEVICE int cluster_of(int clusters, int total, int pos) {
    return int((uint64_t(pos + 1) * clusters - 1) / uint64_t(total));
}

// The same over the main kernel's bounds: the last c with bound[c] <= pos.
CUTE_DEVICE int cluster_of(const int* bound, int clusters, int pos) {
    int lo = 0, hi = clusters - 1;
    while (lo < hi) {
        const int mid = (lo + hi + 1) / 2;
        if (bound[mid] <= pos)
            lo = mid;
        else
            hi = mid - 1;
    }
    return lo;
}

// Reads O columns [m * 128, +128) of this lane, then frees them for the next
// PV.
template <class C>
CUTE_DEVICE void read_o_half(typename C::Smem& sm, uint32_t lane_base, int t,
                             int m, float (&o)[kOHalf]) {
    CUTLASS_PRAGMA_UNROLL
    for (int c = 0; c < kOHalf; c += kTmemLd) {
        tmem_ld_32dp32bNx<kTmemLd>(lane_base + kTmemO + m * kOHalf + c, o + c);
    }
    cutlass::arch::fence_view_async_tmem_load();
    tcgen05_before_thread_sync();
    __syncwarp();
    if (t % kWarp == 0) Bar::arrive(&sm.o_empty[m], 0, 1);
}

template <int N>
CUTE_DEVICE void store_out(__nv_bfloat16* dst, const float* v) {
    CUTLASS_PRAGMA_UNROLL
    for (int q = 0; q < N; q += 16) {
        uint32_t w[8];
        CUTLASS_PRAGMA_UNROLL
        for (int e = 0; e < 8; ++e) {
            const nv_bfloat162 b2 =
                __floats2bfloat162_rn(v[q + 2 * e], v[q + 2 * e + 1]);
            w[e] = reinterpret_cast<const uint32_t&>(b2);
        }
        st_global_v8_na(dst + q, w);
    }
}

CUTE_DEVICE int atom_acq_rel_add(int* ptr, int v) {
    int old;
    asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;"
                 : "=r"(old)
                 : "l"(ptr), "r"(v)
                 : "memory");
    return old;
}

// Added by a merger that finds a part missing: it publishes its own part and
// the last part to arrive merges, so no cluster waits.
constexpr int kDeferred = 1 << 16;

// O leaves through qo_stage by TMA: round r writes dims [64r, 64r + 64), a box
// per dims half. Thread 0 issues the stores and waits for the previous round to
// leave the stage.
constexpr int kRoundWords = kQBoxDims / 2;

template <class C>
CUTE_DEVICE void store_round(const PairMlaMaps& maps, typename C::Smem& sm,
                             int token, int t, int r, const uint32_t* w) {
    if (r > 0) {
        if (t == 0) cute::tma_store_wait<0>();
        NamedBar::sync(kWarpGroup, kQoBar);
    }
    const uint32_t row = swizzled_row(sm.qo_stage[t / kHeads], t % kHeads);
    cute::for_each(cute::make_int_sequence<kChunksPerBox>{}, [&](auto q) {
        constexpr int Q = decltype(q)::value;
        sts_chunk<Q>(row, w[4 * Q], w[4 * Q + 1], w[4 * Q + 2], w[4 * Q + 3]);
    });
    cutlass::arch::fence_view_async_shared();
    NamedBar::sync(kWarpGroup, kQoBar);
    if (t == 0) {
        CUTLASS_PRAGMA_UNROLL
        for (int h = 0; h < 2; ++h) {
            cute::SM90_TMA_STORE_2D::copy(&maps.out, sm.qo_stage[h],
                                          (h * kQoRounds + r) * kQBoxDims,
                                          token * kHeads);
        }
        cute::tma_store_arrive();
    }
}

// Reads half mh of this lane's O, scaled, as bf16 pairs, and frees it for the
// next PV.
template <class C>
CUTE_DEVICE void read_o_bf16(typename C::Smem& sm, uint32_t lane_base, int t,
                             int mh, float scale, uint32_t (&w)[kOHalf / 2]) {
    CUTLASS_PRAGMA_UNROLL
    for (int c = 0; c < kOHalf; c += kTmemLd) {
        float o[kTmemLd];
        tmem_ld(lane_base + kTmemO + mh * kOHalf + c, o);
        CUTLASS_PRAGMA_UNROLL
        for (int e = 0; e < kTmemLd / 2; ++e) {
            const nv_bfloat162 b2 =
                __floats2bfloat162_rn(o[2 * e] * scale, o[2 * e + 1] * scale);
            w[c / 2 + e] = reinterpret_cast<const uint32_t&>(b2);
        }
    }
    tcgen05_before_thread_sync();
    __syncwarp();
    if (t % kWarp == 0) Bar::arrive(&sm.o_empty[mh], 0, 1);
}

// bf16 output of an unsplit unit, a half at a time. `write` is false for a lone
// token's CTA 1: it drains O but stores nothing (the row belongs to the next
// request).
template <class C>
CUTE_DEVICE void store_out_rows(const PairMlaMaps& maps, typename C::Smem& sm,
                                int token, uint32_t lane_base, int t,
                                float scale, bool write) {
    CUTLASS_PRAGMA_UNROLL
    for (int mh = 0; mh < 2; ++mh) {
        uint32_t w[kOHalf / 2];
        read_o_bf16<C>(sm, lane_base, t, mh, scale, w);
        if (write) {
            store_round<C>(maps, sm, token, t, 2 * mh, w);
            store_round<C>(maps, sm, token, t, 2 * mh + 1, w + kRoundWords);
        }
    }
    // The stage is free for the next unit's Q.
    if (write && t == 0) cute::tma_store_wait<0>();
}

// f(c) for each part of a split pair in cluster order, skipping empty ranges.
template <class F>
CUTE_DEVICE void for_parts(int clusters, const Unit& u, int first,
                           const int* bound, F&& f) {
    const int end = u.w0 + u.wn;
    for (int c = first; c < clusters && bound[c] < end; ++c) {
        if (bound[c + 1] > bound[c]) f(c);
    }
}

// A part's planes are contiguous: plain bulk copies move them.
CUTE_DEVICE void bulk_load(void* dst, const void* src, uint32_t bytes,
                           uint64_t* bar) {
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
        "[%0], [%1], %2, [%3];" ::"r"(smem_u32(dst)),
        "l"(src), "r"(bytes), "r"(smem_u32(bar))
        : "memory");
}

CUTE_DEVICE const float* part_box(const PairMlaParams& p, int slot, int cta,
                                  int bx) {
    return p.o_part +
           (int64_t(slot * 2 + cta) * kPlanes + bx * kPartBoxPlanes) *
               (kPlaneBytes / sizeof(float));
}

// O of a finished unit: bf16 output with the sink in the denominator.
// A split pair is merged by the cluster of its first tile, staging the other
// parts in its idle KV stages. If one is missing, the last part to arrive
// merges from global memory instead. Both add the parts in the same order and
// ops, so the output is the same.
// Each half of O is freed for the next PV once read.
template <class C>
CUTE_DEVICE void store_o(const PairMlaParams& p, const PairMlaMaps& maps,
                         typename C::Smem& sm, const Unit& u, int unit, int cta,
                         int cluster, uint32_t lane_base, int t,
                         float sink_log2) {
    const int head = t % kHeads, half = t / kHeads;
    const int b = unit % 2;
    const int sched = sm.sched;
    // first and last are for split pairs only; unsplit units skip the searches.
    const int first = u.split ? cluster_of(sm.bound, sched, u.w0) : cluster;
    // Parts are clusters first .. last when every range is non-empty, else
    // empty ranges are skipped.
    const int last =
        u.split ? cluster_of(sm.bound, sched, u.w0 + u.wn - 1) : cluster;
    const bool dense = sm.dense;
    const auto part_cluster = [&](int j) {
        if (dense) return first + j;
        int c_j = first, n = 0;
        for_parts(sched, u, first, sm.bound, [&](int c) {
            if (n++ == j) c_j = c;
        });
        return c_j;
    };
    int n_parts = last - first + 1;
    if (u.split && !dense) {
        n_parts = 0;
        for_parts(sched, u, first, sm.bound, [&](int) { ++n_parts; });
    }
    const int others = n_parts - 1;
    // The merger prefetches the other parts into L2 while its part runs.
    if (u.split && first == cluster && t < kPartBoxes && !p.combine) {
        for (int k = 1; k <= others; ++k) {
            prefetch_l2_bulk(part_box(p, part_cluster(k), cta, t),
                             kPartBoxPlanes * kPlaneBytes);
        }
    }
    qo_wait(&sm.ml_full[b], (unit / 2) & 1, t);
    const float2 own = sm.ml[b][t], other = sm.ml[b][t ^ kHeads];
    __syncwarp();
    if (t % kWarp == 0) Bar::arrive(&sm.ml_empty[b]);
    const float m = own.x, sum = own.y + other.y;
    // A lone token's CTA 1 runs the pipeline for the shared KV tiles but writes
    // no output and skips the merge.
    const bool lone = cta == 1 && !u.g.has_b;
    if (!u.split) {
        const float scale =
            sum > 0.0f ? 1.0f / (sum + exp2f(sink_log2 - m)) : 0.0f;
        qo_wait(&sm.o_full, unit & 1, t);
        tcgen05_after_thread_sync();
        store_out_rows<C>(maps, sm, u.g.a + cta, lane_base, t, scale, !lone);
        return;
    }
    int* const ctr = p.counter + first * 2 + cta;  // parts published
    __nv_bfloat16* const out = p.out +
                               (int64_t(u.g.a + cta) * kHeads + head) * kDim +
                               half * kHalfDims;
    const float own_scale = sum > 0.0f ? 1.0f / sum : 0.0f;
    const float own_lse = sum > 0.0f ? m + log2f(sum) : -CUDART_INF_F;
    qo_wait(&sm.o_full, unit & 1, t);
    tcgen05_after_thread_sync();
    if (lone) {
        // A follower part drains O for the next PV; the merger (first ==
        // cluster) has no next unit and returns. CTA 1 writes no partial and
        // leaves the counter alone.
        if (first != cluster) {
            float o[kOHalf];
            read_o_half<C>(sm, lane_base, t, 0, o);
            read_o_half<C>(sm, lane_base, t, 1, o);
            NamedBar::sync(kWarpGroup, kQoBar);
        }
        return;
    }
    // A part's normalized O and log2 sum to partial slot sid.
    const auto write_part = [&](int sid) {
        if (half == 0) p.lse_part[part_row(sid, cta, head)] = own_lse;
        float* const dst =
            p.o_part +
            (int64_t(sid * 2 + cta) * kPlanes * kWarpGroup + t) * kPlaneCols;
        CUTLASS_PRAGMA_UNROLL
        for (int mh = 0; mh < 2; ++mh) {
            float o[kOHalf];
            read_o_half<C>(sm, lane_base, t, mh, o);
            CUTLASS_PRAGMA_UNROLL
            for (int q = 0; q < kOHalf / kPlaneCols; ++q) {
                const int pl = mh * kOHalf / kPlaneCols + q;
                *reinterpret_cast<float4*>(dst + pl * kWarpGroup * kPlaneCols) =
                    make_float4(o[4 * q] * own_scale, o[4 * q + 1] * own_scale,
                                o[4 * q + 2] * own_scale,
                                o[4 * q + 3] * own_scale);
            }
        }
        // Thread 0's release covers the group's stores, ordered by the barrier.
        NamedBar::sync(kWarpGroup, kQoBar);
    };
    __shared__ int merge_mode;
    const auto broadcast = [&](int v) {
        if (t == 0) merge_mode = v;
        NamedBar::sync(kWarpGroup, kQoBar);
        return merge_mode;
    };
    // The merger's own part goes to slot clusters + cluster: slot cluster may
    // hold another pair's part.
    const int merger_slot = p.clusters + first;
    // 0: nothing more to do, 1: merge here in TMEM, 2: merge from global.
    int mode;
    if (first != cluster || p.combine) {
        // combine: two slots per cluster, the owner part in slot 1,
        // continuations in slot 0.
        const int sid = p.combine ? cluster * 2 + (first == cluster) : cluster;
        write_part(sid);
        // One job per pair, recorded by CTA 0; the combine kernel runs after
        // this kernel, so no counter.
        if (p.combine) {
            if (first == cluster && cta == 0 && t == 0) {
                const int idx = atomicAdd(p.n_jobs, 1);
                p.jobs[idx] = MergeJob{u.g.a, u.g.has_b, u.w0, u.wn};
            }
            return;
        }
        int v = 0;
        if (t == 0)
            v = atom_acq_rel_add(ctr, 1) == kDeferred + others - 1 ? 2 : 0;
        mode = broadcast(v);
    } else {
        int v = 0;
        if (t == 0) v = gmem_ld_acquire(ctr) == others ? 1 : 0;
        mode = broadcast(v);
        if (mode == 0) {
            // Its TMEM keeps O: this is the cluster's last unit.
            write_part(merger_slot);
            if (t == 0) v = atom_acq_rel_add(ctr, kDeferred) == others ? 1 : 0;
            mode = broadcast(v);
        }
    }
    if (mode == 0) return;
    if (t == 0) {
        *ctr = 0;
        // The parts, written by generic stores, are read by bulk copies.
        if (mode == 1) fence_proxy_async_global();
    }
    NamedBar::sync(kWarpGroup, kQoBar);
    const float lse0 =
        mode == 1 ? own_lse
                  : __ldcg(p.lse_part + part_row(merger_slot, cta, head));
    // log2 sums of up to kLseRegs parts in registers; more (tiny batches) are
    // reloaded.
    constexpr int kLseRegs = 8;
    float lse_r[kLseRegs];
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < kLseRegs; ++j) {
        lse_r[j] =
            j >= n_parts ? -CUDART_INF_F
            : j == 0
                ? lse0
                : __ldcg(p.lse_part + part_row(part_cluster(j), cta, head));
    }
    const auto lse_at = [&](int j) {
        if (j >= kLseRegs)
            return __ldcg(p.lse_part + part_row(part_cluster(j), cta, head));
        float v = 0.0f;
        CUTLASS_PRAGMA_UNROLL
        for (int k = 0; k < kLseRegs; ++k) {
            if (k == j) v = lse_r[k];
        }
        return v;
    };
    float mx = sink_log2;
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < kLseRegs; ++j) mx = fmaxf(mx, lse_r[j]);
    for (int j = kLseRegs; j < n_parts; ++j) mx = fmaxf(mx, lse_at(j));
    float den = sink_log2 > -CUDART_INF_F ? exp2f(sink_log2 - mx) : 0.0f;
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < kLseRegs; ++j) {
        if (lse_r[j] > -CUDART_INF_F) den += exp2f(lse_r[j] - mx);
    }
    for (int j = kLseRegs; j < n_parts; ++j) {
        const float lse = lse_at(j);
        if (lse > -CUDART_INF_F) den += exp2f(lse - mx);
    }
    const float inv = den > 0.0f ? 1.0f / den : 0.0f;
    const float w_own = lse0 > -CUDART_INF_F ? exp2f(lse0 - mx) * inv : 0.0f;
    const uint32_t cols = lane_base + kTmemO;
    if (mode == 2) {
        // Every part from global memory, 16 columns at a time, in the same
        // order and ops as the TMEM merge.
        constexpr int kCols = 16;
        const auto plane = [&](int j, int pl) {
            const int slot = j == 0 ? merger_slot : part_cluster(j);
            return __ldcg(reinterpret_cast<const float4*>(
                p.o_part +
                ((int64_t(slot * 2 + cta) * kPlanes + pl) * kWarpGroup + t) *
                    kPlaneCols));
        };
        for (int db = 0; db < kHalfDims / kCols; ++db) {
            float x[kCols];
            for (int j = 0; j < n_parts; ++j) {
                const float lse = j == 0 ? lse0 : lse_at(j);
                const float w =
                    lse > -CUDART_INF_F ? exp2f(lse - mx) * inv : 0.0f;
                float4 v[kCols / kPlaneCols];
                CUTLASS_PRAGMA_UNROLL
                for (int q = 0; q < kCols / kPlaneCols; ++q)
                    v[q] = plane(j, db * (kCols / kPlaneCols) + q);
                CUTLASS_PRAGMA_UNROLL
                for (int q = 0; q < kCols / kPlaneCols; ++q) {
                    if (j == 0) {
                        x[4 * q] = v[q].x * w;
                        x[4 * q + 1] = v[q].y * w;
                        x[4 * q + 2] = v[q].z * w;
                        x[4 * q + 3] = v[q].w * w;
                    } else {
                        x[4 * q] += v[q].x * w;
                        x[4 * q + 1] += v[q].y * w;
                        x[4 * q + 2] += v[q].z * w;
                        x[4 * q + 3] += v[q].w * w;
                    }
                }
            }
            store_out<kCols>(out + db * kCols, x);
        }
    } else {
        // TMEM merge: own O is prescaled in TMEM, the other parts stream
        // through kMergeBuf boxes in the idle KV stages and are added; the last
        // part's pass writes the output.
        CUTLASS_PRAGMA_UNROLL
        for (int db = 0; db < kHalfDims / kMergeCols; ++db) {
            float x[kMergeCols];
            tmem_ld_32dp32bNx<kMergeCols>(cols + db * kMergeCols, x);
            cutlass::arch::fence_view_async_tmem_load();
            CUTLASS_PRAGMA_UNROLL
            // (O * own_scale) * w_own, as the published part is O * own_scale.
            for (int q = 0; q < kMergeCols; ++q)
                x[q] = x[q] * own_scale * w_own;
            tmem_st_32dp32bNx<kMergeCols>(cols + db * kMergeCols, x);
        }
        cutlass::arch::fence_view_async_tmem_store();
        NamedBar::sync(kWarpGroup, kQoBar);
        // One box (kPartBoxPlanes planes) into ring buffer buf in the KV
        // stages.
        constexpr int kSubPerBox = kPartBoxPlanes / (kMergeCols / kPlaneCols);
        constexpr int kMergeBoxBytes = kPartBoxPlanes * kPlaneBytes;
        static_assert(kMergeBuf * kMergeBoxBytes <= kKvStages * kStageBytes);
        const auto merge_buf = [&](int buf) {
            return sm.kv[0] + buf * kMergeBoxBytes;
        };
        const auto load_box = [&](int k, int bx, int buf) {
            if (t == 0) {
                cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(
                    &sm.merge_full[buf], kPartBoxPlanes * kPlaneBytes);
                bulk_load(merge_buf(buf), part_box(p, part_cluster(k), cta, bx),
                          kPartBoxPlanes * kPlaneBytes, &sm.merge_full[buf]);
            }
        };
        const int nbox = others * kPartBoxes;
        for (int g = 0; g < kMergeBuf && g < nbox; ++g)
            load_box(1 + g / kPartBoxes, g % kPartBoxes, g);
        float w = 0.0f;
        for (int g = 0; g < nbox; ++g) {
            const int k = 1 + g / kPartBoxes, bx = g % kPartBoxes;
            const int buf = g % kMergeBuf;
            if (bx == 0) {
                const float lse = lse_at(k);
                w = lse > -CUDART_INF_F ? exp2f(lse - mx) * inv : 0.0f;
            }
            Bar::wait(&sm.merge_full[buf], (g / kMergeBuf) & 1);
            CUTLASS_PRAGMA_UNROLL
            for (int sub = 0; sub < kSubPerBox; ++sub) {
                const int db = bx * kSubPerBox + sub;
                float x[kMergeCols];
                tmem_ld_32dp32bNx<kMergeCols>(cols + db * kMergeCols, x);
                cutlass::arch::fence_view_async_tmem_load();
                const uint8_t* const planes =
                    merge_buf(buf) +
                    sub * (kMergeCols / kPlaneCols) * kPlaneBytes +
                    t * kPlaneCols * sizeof(float);
                CUTLASS_PRAGMA_UNROLL
                for (int q = 0; q < kMergeCols / kPlaneCols; ++q) {
                    const float4 v = *reinterpret_cast<const float4*>(
                        planes + q * kPlaneBytes);
                    x[4 * q] += v.x * w;
                    x[4 * q + 1] += v.y * w;
                    x[4 * q + 2] += v.z * w;
                    x[4 * q + 3] += v.w * w;
                }
                if (k == others) {
                    store_out<kMergeCols>(out + db * kMergeCols, x);
                } else {
                    tmem_st_32dp32bNx<kMergeCols>(cols + db * kMergeCols, x);
                }
            }
            cutlass::arch::fence_view_async_tmem_store();
            NamedBar::sync(kWarpGroup, kQoBar);
            const int g2 = g + kMergeBuf;
            if (g2 < nbox)
                load_box(1 + g2 / kPartBoxes, g2 % kPartBoxes, g2 % kMergeBuf);
        }
    }
}

template <class C>
CUTE_DEVICE void qo_wg(const PairMlaParams& p, const PairMlaMaps& maps,
                       typename C::Smem& sm, uint32_t tmem, float sink_log2,
                       int cta, int t, int cluster) {
    const uint32_t lane_base = tmem + ((t / kWarp * kWarp) << 16);
    UnitCursor cur = start_cursor<C>(p, sm);
    Unit u, prev;
    bool more = cur.next(p, sm.unit_lens, u);
    int unit = 0;
    constexpr bool kQst = C::kQst;
    const auto q_tok = [&](const Unit& v) {
        return v.g.a + (cta == 1 && !v.g.has_b ? 0 : cta);
    };
    // The TMA row of a unit's Q (kQnr: the token, for the 3D map).
    const auto q_row = [&](const Unit& v) {
        return C::kQnr ? q_tok(v) : q_tok(v) * kHeads;
    };
    // kQnr: a unit's position loads once the unit is known, its frequencies
    // with its Q.
    int qpos = 0;
    const auto next_pos = [&](bool known, const Unit& v) {
        if constexpr (C::kQnr) {
            if (known && freq_lane(t)) qpos = load_pos(p, q_tok(v));
        }
    };
    next_pos(more, u);
    if constexpr (kQst) {
        if (more && t == 0) {
            const int row = q_row(u);
            issue_q<C>(maps, sm, row, 0);
            issue_q<C>(maps, sm, row, 1);
        }
    }
    for (; more; ++unit) {
        // Unit u's Q loads while the previous unit runs and goes to TMEM after
        // that unit's last QK^T, through qo_stage (or the kQst Q buffers), so
        // the KV stages run at full depth from the start.
        uint4 qv[kQVecs];
        if constexpr (C::kQnr) {
            if (freq_lane(t)) load_freqs(p, sm.qfreq, qpos, t);
        }
        // A lone token's CTA 1 loads the pair's valid token (an L2 hit) to keep
        // the Q barriers balanced.
        Unit nxt;
        bool more_n = false;
        if constexpr (kQst) {
            // Thread 0 issued rounds 0 and 1 from this row; it issues the rest.
            fetch_q2<C>(maps, sm, t == 0 ? q_row(u) : 0, t, qv);
            // The next unit's first two rounds load while this unit runs.
            more_n = cur.next(p, sm.unit_lens, nxt);
            if (more_n && t == 0) {
                const int nrow = q_row(nxt);
                issue_q<C>(maps, sm, nrow, 0);
                issue_q<C>(maps, sm, nrow, 1);
            }
            next_pos(more_n, nxt);
        } else {
            fetch_q<C>(maps, sm, t == 0 ? q_row(u) : 0, t, qv);
        }
        if constexpr (C::kQnr) qo_norm_rope<C>(p, sm, t, unit % 2, qv);
        if (unit > 0) {
            qo_wait(&sm.q_empty, (unit - 1) & 1, t);
            tcgen05_after_thread_sync();
        }
        store_q<C>(sm, lane_base, t, qv);
        if (unit > 0) {
            store_o<C>(p, maps, sm, prev, unit - 1, cta, cluster, lane_base, t,
                       sink_log2);
        }
        prev = u;
        if constexpr (kQst) {
            more = more_n;
            if (more) u = nxt;
        } else {
            more = cur.next(p, sm.unit_lens, u);
            next_pos(more, u);
        }
    }
    if (unit > 0) {
        store_o<C>(p, maps, sm, prev, unit - 1, cta, cluster, lane_base, t,
                   sink_log2);
    }
    // The output is written before the CTA exits.
    if (t == 0) bulk_wait_all();
}

template <class C>
CUTE_DEVICE void init_barriers(typename C::Smem& sm) {
    constexpr int kPairWarps = 2 * kWarpsPerGroup;  // a warp group of both CTAs
    Bar::init(&sm.q_full, kPairWarps);
    Bar::init(&sm.q_empty, 1);
    Bar::init(&sm.o_full, 1);
    for (int i = 0; i < kMergeBuf; ++i) Bar::init(&sm.merge_full[i], 1);
    Bar::init(&sm.qs_full, 1);
    if constexpr (C::kQst) {
        for (int b = 0; b < 2; ++b) Bar::init(&sm.qb.qb_full[b], 1);
    }
    for (int b = 0; b < 2; ++b) {
        Bar::init(&sm.o_empty[b], kPairWarps);
        Bar::init(&sm.ml_full[b], kWarpsPerGroup);
        Bar::init(&sm.ml_empty[b], kWarpsPerGroup);
        Bar::init(&sm.rs_full[b], kWarpsPerGroup / 2);
        Bar::init(&sm.s_full[b], 1);
        Bar::init(&sm.s_empty[b], kPairWarps);
        Bar::init(&sm.p_full[b], kPairWarps);
        Bar::init(&sm.pv_done[b], 1);
    }
    for (int s = 0; s < kIdxStages; ++s) {
        Bar::init(&sm.idx_full[s], kIndexWarps);
        Bar::init(&sm.idx_empty[s], 2 * kWarpsPerGroup);  // softmax + dequant
    }
    for (int s = 0; s < kKvStages; ++s) {
        // per index warp: the expect-tx arrival and one cp.async per lane
        Bar::init(&sm.raw_full[s], kIndexWarps * (1 + kWarp));
        Bar::init(&sm.kv_full[s], kPairWarps);
        Bar::init(&sm.kv_empty[s], 1);
    }
    cutlass::arch::fence_barrier_init();
}

// Persistent: cluster x walks its range of the tile stream. Warp groups: 0
// softmax, 1 Q/O, 2 dequant, 3 QK^T and PV warps (CTA 0) plus two index warps
// (one allocates TMEM).
template <class C>
CUTE_DEVICE void run_attention(const PairMlaParams& p,
                               const PairMlaMaps& maps) {
    extern __shared__ uint8_t smem_raw[];
    typename C::Smem& sm = *reinterpret_cast<typename C::Smem*>(
        smem_raw + (kAtomBytes - smem_u32(smem_raw) % kAtomBytes) % kAtomBytes);
    const int cta = cute::block_rank_in_cluster();
    const int cluster = blockIdx.x / 2;
    const int warp = __shfl_sync(~0u, threadIdx.x / kWarp, 0);
    const int lane = threadIdx.x % kWarp;
    const int wg = warp / kWarpsPerGroup;
    const int t = threadIdx.x % kWarpGroup;

    if (warp == kAllocWarp) {
        cute::TMEM::Allocator2Sm().allocate(kTmemCols, &sm.tmem_base);
        cute::TMEM::Allocator2Sm().release_allocation_lock();
    } else if (threadIdx.x == 0) {
        init_barriers<C>(sm);
    }
    // The sink is a weight, not a preceding kernel's output: read before
    // pdl_wait.
    const float sink_log2 =
        p.sink ? __ldg(p.sink + t % kHeads) * CUDART_L2E_F : -CUDART_INF_F;
    tcgen05_before_thread_sync();
    cute::cluster_sync();
    tcgen05_after_thread_sync();
    const uint32_t tmem = sm.tmem_base;

    pdl_wait();
    find_range<C>(p, sm, cluster);
    const int n = sm.start[2];
    if (wg == 0) {
        cutlass::arch::warpgroup_reg_alloc<kSoftmaxRegs>();
        softmax_wg<C>(p, sm, tmem, t);
    } else if (wg == 1) {
        cutlass::arch::warpgroup_reg_alloc<kQoRegs>();
        qo_wg<C>(p, maps, sm, tmem, sink_log2, cta, t, cluster);
    } else if (wg == 2) {
        cutlass::arch::warpgroup_reg_dealloc<kDequantRegs>();
        if (cta == 0) {
            dequant_wg<0, C>(sm, n, t);
        } else {
            dequant_wg<1, C>(sm, n, t);
        }
    } else {
        cutlass::arch::warpgroup_reg_dealloc<kAuxRegs>();
        if (warp == kQkWarp) {
            if (cta == 0) qk_warp<C>(p, sm, tmem);
        } else if (warp == kPvWarp) {
            if (cta == 0) pv_warp<C>(p, sm, tmem);
        } else {
            static_assert(kIndexWarp + kIndexWarps == kThreads / kWarp);
            index_warp<C>(p, maps, sm, cta, warp - kIndexWarp, lane);
        }
    }
    pdl_trigger();
    tcgen05_before_thread_sync();
    cute::cluster_sync();
    if (warp == kAllocWarp) {
        tcgen05_after_thread_sync();
        cute::TMEM::Allocator2Sm().free(tmem, kTmemCols);
    }
}

template <bool kQst, bool kFp8, bool kQnr>
__global__ void __launch_bounds__(kThreads, 1)
    pair_mla_kernel(const __grid_constant__ PairMlaParams p,
                    const __grid_constant__ PairMlaMaps maps) {
    run_attention<Cfg<kQst, kFp8, kQnr>>(p, maps);
}

// Combine kernel: one job per split pair, after the main kernel. grid =
// (kPlanes, 2 CTAs, jobs); a block reduces one plane of one token's output over
// the pair's parts. kCombSplit warp groups each take a strided subset of parts
// to hide load latency, then combine through shared memory. Deterministic, but
// not bitwise equal to the in-kernel merge (different order and rescaling).
constexpr int kCombSplit = 4;
__global__ void pair_mla_combine_kernel(
    const __grid_constant__ PairMlaParams p) {
    __shared__ float red[kCombSplit][kWarpGroup][kPlaneCols];
    __shared__ float redmd[kCombSplit][kWarpGroup][2];  // (running max, denom)
    // PDL: pdl_wait returns once the main kernel's writes are visible; only the
    // launch overlaps its tail.
    pdl_wait();
    const int job = blockIdx.z;
    if (job >= *p.n_jobs) return;
    const MergeJob j = p.jobs[job];
    const int cta = blockIdx.y;
    if (cta == 1 && !j.has_b) return;
    const int pl = blockIdx.x;                 // plane [0, kPlanes)
    const int grp = threadIdx.x / kWarpGroup;  // part-group [0, kCombSplit)
    const int tsi = threadIdx.x % kWarpGroup;  // storage thread index
    const int head = tsi % kHeads, half = tsi / kHeads;
    const int total = *p.total_g;
    const int sched = *p.sched_g;
    const int first = cluster_of(sched, total, j.w0);
    const int last = cluster_of(sched, total, j.w0 + j.wn - 1);
    const float sink_log2 =
        p.sink ? __ldg(p.sink + head) * CUDART_L2E_F : -CUDART_INF_F;
    // A cluster with a non-empty range has a part (as for_parts), in slot
    // cluster*2 + (cluster == first).
    const auto empty = [&](int c) {
        return range_lo(c + 1, total, sched) <= range_lo(c, total, sched);
    };
    const auto sid = [&](int c) { return c * 2 + (c == first); };
    // One online-softmax pass over this group's parts: each part is read once,
    // the running max rescales the fp32 accumulator.
    float m = -CUDART_INF_F, den = 0.0f;
    float acc[kPlaneCols] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int c = first + grp; c <= last; c += kCombSplit) {
        if (empty(c)) continue;
        const float lse = __ldg(p.lse_part + part_row(sid(c), cta, head));
        if (!(lse > -CUDART_INF_F)) continue;
        const float nm = fmaxf(m, lse);
        const float sc = exp2f(m - nm), e = exp2f(lse - nm);
        den = den * sc + e;
        const float4 v = *reinterpret_cast<const float4*>(
            p.o_part +
            (int64_t((sid(c) * 2 + cta) * kPlanes + pl) * kWarpGroup + tsi) *
                kPlaneCols);
        acc[0] = acc[0] * sc + e * v.x;
        acc[1] = acc[1] * sc + e * v.y;
        acc[2] = acc[2] * sc + e * v.z;
        acc[3] = acc[3] * sc + e * v.w;
        m = nm;
    }
    CUTLASS_PRAGMA_UNROLL
    for (int e = 0; e < kPlaneCols; ++e) red[grp][tsi][e] = acc[e];
    redmd[grp][tsi][0] = m;
    redmd[grp][tsi][1] = den;
    __syncthreads();
    if (grp != 0) return;
    // Merge the group partials with the sink (denominator only), then
    // normalize. A group with no part keeps max -inf and weight 0, which keeps
    // NaN out when there is no sink.
    float mx = sink_log2;
    CUTLASS_PRAGMA_UNROLL
    for (int g = 0; g < kCombSplit; ++g) mx = fmaxf(mx, redmd[g][tsi][0]);
    float dtot = sink_log2 > -CUDART_INF_F ? exp2f(sink_log2 - mx) : 0.0f;
    CUTLASS_PRAGMA_UNROLL
    for (int e = 0; e < kPlaneCols; ++e) acc[e] = 0.0f;
    CUTLASS_PRAGMA_UNROLL
    for (int g = 0; g < kCombSplit; ++g) {
        const float mg = redmd[g][tsi][0];
        const float sc = mg > -CUDART_INF_F ? exp2f(mg - mx) : 0.0f;
        dtot += redmd[g][tsi][1] * sc;
        CUTLASS_PRAGMA_UNROLL
        for (int e = 0; e < kPlaneCols; ++e) acc[e] += red[g][tsi][e] * sc;
    }
    const float inv = dtot > 0.0f ? 1.0f / dtot : 0.0f;
    CUTLASS_PRAGMA_UNROLL
    for (int e = 0; e < kPlaneCols; ++e) acc[e] *= inv;
    const int mh = pl / (kOHalf / kPlaneCols);
    const int q = pl % (kOHalf / kPlaneCols);
    const int dim = half * kHalfDims + mh * kOHalf + q * kPlaneCols;
    __nv_bfloat16* const out =
        p.out + (int64_t(j.a + cta) * kHeads + head) * kDim + dim;
    reinterpret_cast<nv_bfloat162*>(out)[0] =
        __floats2bfloat162_rn(acc[0], acc[1]);
    reinterpret_cast<nv_bfloat162*>(out)[1] =
        __floats2bfloat162_rn(acc[2], acc[3]);
    // Let a PDL dependent launch early; its pdl_wait still waits for this grid.
    pdl_trigger();
}

// Pre-dequant of the c128 rows, one launch. Copy row j of request r is key j of
// its last token's list (the reference).
// - Blocks [0, ceil(T / 8)): a warp per token sets c128_base = r * c128_rows if
//   its list is a prefix of the reference, else -1 (fp8 path).
// - Other blocks: a warp per copy row, dequant8 as in the kernel (bit-exact);
//   invalid keys give zero rows.
// Rows past the reference list are read only for masked keys and stay finite.
constexpr int kPredqThreads = 256, kPredqWarps = kPredqThreads / kWarp;

CUTE_DEVICE int c128_len(const PairMlaParams& p, int t) {
    return max(min(__ldg(p.len[0] + t), p.ld[0]), 0);
}

// Request r's tokens [first, last).
CUTE_DEVICE int2 request_tokens(const PairMlaParams& p, int r) {
    if (!p.cu_seqlens)
        return {r * p.tokens_per_req, (r + 1) * p.tokens_per_req};
    return {__ldg(p.cu_seqlens + r), __ldg(p.cu_seqlens + r + 1)};
}

CUTE_DEVICE int request_of(const PairMlaParams& p, int t) {
    if (!p.cu_seqlens) return t / p.tokens_per_req;
    int lo = 0, hi = p.n_req;  // last request r with cu_seqlens[r] <= t
    while (hi - lo > 1) {
        const int mid = (lo + hi) / 2;
        if (__ldg(p.cu_seqlens + mid) <= t)
            lo = mid;
        else
            hi = mid;
    }
    return lo;
}

__global__ void __launch_bounds__(kPredqThreads)
    predq_kernel(const __grid_constant__ PairMlaParams p) {
    pdl_wait();
    const int warp = threadIdx.x / kWarp, lane = threadIdx.x % kWarp;
    const int check_blocks = (p.tokens + kPredqWarps - 1) / kPredqWarps;
    if (int(blockIdx.x) < check_blocks) {
        const int t = blockIdx.x * kPredqWarps + warp;
        if (t >= p.tokens) return;
        const int r = request_of(p, t), ref = request_tokens(p, r).y - 1;
        const int len = c128_len(p, t);
        bool bad = len > c128_len(p, ref);
        if (!bad && t != ref) {
            const int* const mine = p.idx[0] + int64_t(t) * p.ld[0];
            const int* const theirs = p.idx[0] + int64_t(ref) * p.ld[0];
            // 16-byte loads kept in flight together (latency bound).
            int j = 0;
            if ((p.ld[0] & 3) == 0 &&
                (reinterpret_cast<uintptr_t>(p.idx[0]) & 15) == 0) {
                const int4* const m4 = reinterpret_cast<const int4*>(mine);
                const int4* const t4 = reinterpret_cast<const int4*>(theirs);
                const int n4 = len / 4;
#pragma unroll 4
                for (int k = lane; k < n4; k += kWarp) {
                    const int4 x = __ldg(m4 + k), y = __ldg(t4 + k);
                    bad |= x.x != y.x || x.y != y.y || x.z != y.z || x.w != y.w;
                }
                j = n4 * 4;
            }
            for (j += lane; j < len; j += kWarp) {
                bad |= __ldg(mine + j) != __ldg(theirs + j);
            }
        }
        bad = __any_sync(~0u, bad);
        if (lane == 0) p.c128_base[t] = bad ? -1 : r * p.c128_rows;
        return;
    }
    const int64_t g = int64_t(blockIdx.x - check_blocks) * kPredqWarps + warp;
    const int r = int(g / p.c128_rows), j = int(g % p.c128_rows);
    const int2 tok = request_tokens(p, r);
    if (tok.y <= tok.x || j >= c128_len(p, tok.y - 1)) return;
    const int key = __ldg(p.idx[0] + int64_t(tok.y - 1) * p.ld[0] + j);
    uint4* const dst = reinterpret_cast<uint4*>(p.c128_bf + g * kDim);
    if (key < 0 || key >= p.pool_tokens[0]) {
        dst[lane] = uint4{0, 0, 0, 0};
        dst[lane + kWarp] = uint4{0, 0, 0, 0};
        return;
    }
    const int64_t page = key >> p.page_shift[0];
    const int row = key & ((1 << p.page_shift[0]) - 1);
    const uint8_t* const base = p.pool[0] + page * p.pool_stride[0];
    const uint8_t* const src = base + int64_t(row) * kRowBytes;
    constexpr int kFp8Lanes = kNope / kChunk;
    if (lane < kFp8Lanes) {
        const uint4 raw = __ldg(reinterpret_cast<const uint4*>(src) + lane);
        const uint8_t e8 =
            __ldg(base + (int64_t(1) << p.page_shift[0]) * kRowBytes +
                  row * kSfBytes + lane * kChunk / kBlockDims);
        dst[2 * lane] = dequant8(raw.x, raw.y, e8);
        dst[2 * lane + 1] = dequant8(raw.z, raw.w, e8);
    } else {
        const uint4* const rope = reinterpret_cast<const uint4*>(src + kNope);
        const int c = 2 * (lane - kFp8Lanes);
        dst[2 * kFp8Lanes + c] = __ldg(rope + c);
        dst[2 * kFp8Lanes + c + 1] = __ldg(rope + c + 1);
    }
}

CUtensorMap make_map(const void* ptr, CUtensorMapDataType dtype,
                     const torch::Tensor& pool, uint64_t cols, uint32_t box,
                     CUtensorMapSwizzle swizzle) {
    CUtensorMap map;
    EncodeTmaDesc<2>(
        &map, dtype, ptr,
        {cols, uint64_t(pool.size(0) * (pool.stride(0) / kRowBytes))},
        {kRowBytes}, {box, 1}, swizzle, "prefill_pair_mla");
    return map;
}

void set_pool(PairMlaParams& p, PairMlaMaps& maps, int seg,
              const torch::Tensor& pool, const torch::Tensor& idx,
              const torch::Tensor& len, int64_t tokens) {
    TORCH_CHECK(pool.dim() == 4 && pool.scalar_type() == torch::kByte &&
                    pool.size(2) == 1 && pool.size(3) == kRowBytes + kSfBytes &&
                    pool.stride(0) % kRowBytes == 0 &&
                    pool.stride(0) >= pool.size(1) * (kRowBytes + kSfBytes),
                "k caches must be uint8 [pages, page_size, 1, 584] with a page "
                "stride that is a multiple of 576 and at least page_size * "
                "584");
    TORCH_CHECK(idx.scalar_type() == torch::kInt && idx.is_contiguous() &&
                    idx.dim() >= 2 && idx.numel() == tokens * idx.size(-1),
                "indices must be contiguous int32 [T, W] or [T, 1, W]");
    TORCH_CHECK(len.scalar_type() == torch::kInt && len.is_contiguous() &&
                    len.numel() == tokens,
                "lengths must be contiguous int32 [T]");
    p.idx[seg] = idx.data_ptr<int>();
    p.len[seg] = len.data_ptr<int>();
    p.ld[seg] = idx.size(-1);
    p.pool[seg] = pool.data_ptr<uint8_t>();
    p.pool_stride[seg] = pool.stride(0);
    const int page = pool.size(1);
    TORCH_CHECK((page & (page - 1)) == 0,
                "k cache page sizes must be powers of two, got ", page);
    p.page_shift[seg] = __builtin_ctz(page);
    p.page_rows[seg] = pool.stride(0) / kRowBytes;
    p.pool_tokens[seg] = pool.size(0) * pool.size(1);
    const uint8_t* base = pool.data_ptr<uint8_t>();
    for (int c = 0; c < 2; ++c) {
        maps.raw[c][seg] =
            make_map(base, CU_TENSOR_MAP_DATA_TYPE_UINT16, pool, kRowBytes / 2,
                     raw_pitch(c) / 2, CU_TENSOR_MAP_SWIZZLE_NONE);
    }
}

// Split-pair partials and counters per device; the merging part resets its
// counter.
struct Workspace {
    torch::Tensor o_part, lse_part, counter, jobs, n_jobs, total_g, sched_g;
};

Workspace& workspace(const torch::Tensor& q, int clusters) {
    static std::mutex mu;
    static std::unordered_map<int, Workspace> cache;
    const std::lock_guard<std::mutex> lock(mu);
    Workspace& ws = cache[q.get_device()];
    if (!ws.counter.defined() || ws.counter.size(0) != clusters) {
        const auto f32 = q.options().dtype(torch::kFloat);
        const auto i32 = q.options().dtype(torch::kInt);
        // Two slots per cluster (cluster*2 + which): leading continuation (0)
        // of one pair and trailing owner (1) of another.
        ws.o_part = torch::empty(
            {2 * clusters, 2, kPlanes, kWarpGroup, kPlaneCols}, f32);
        ws.lse_part = torch::empty({2 * clusters, 2, kHeads}, f32);
        ws.counter = torch::zeros({clusters, 2}, i32);
        ws.jobs = torch::empty({clusters, 4}, i32);  // MergeJob[clusters]
        ws.n_jobs = torch::zeros({1}, i32);
        ws.total_g = torch::zeros({1}, i32);
        ws.sched_g = torch::zeros({1}, i32);
    }
    return ws;
}

// The bf16 c128 copy (requests x ld[0] rows of 1 KB, grown and zeroed at
// allocation) and per-token bases, per device. Skipped (c128_base null, kFp8
// kernels) above kPredqMaxBytes, or when the extra launch costs more than it
// saves: ld >= kPredqMinLd needs kPredqMinTokens, ld >= kPredqShortLd needs
// kPredqShortTokens.
constexpr int64_t kPredqMaxBytes = int64_t(512) << 20;
constexpr int kPredqMinLd = 512, kPredqMinTokens = 256;
constexpr int kPredqShortLd = 256, kPredqShortTokens = 4096;
struct PredqWorkspace {
    torch::Tensor rows, base;
};

void predequant_c128(PairMlaParams& p, PairMlaMaps& maps,
                     const torch::Tensor& q, bool use_pdl) {
    const int n_req = p.cu_seqlens ? p.n_req : p.tokens / p.tokens_per_req;
    const int64_t per_req = (int64_t(p.ld[0]) + kTile - 1) / kTile * kTile;
    const int64_t rows = int64_t(n_req) * per_req;
    const bool worth =
        (p.ld[0] >= kPredqMinLd && p.tokens >= kPredqMinTokens) ||
        (p.ld[0] >= kPredqShortLd && p.tokens >= kPredqShortTokens);
    if (n_req == 0 || !worth ||
        rows * kDim * int64_t(sizeof(__nv_bfloat16)) > kPredqMaxBytes) {
        return;
    }
    static std::mutex mu;
    static std::unordered_map<int, PredqWorkspace> cache;
    {
        const std::lock_guard<std::mutex> lock(mu);
        PredqWorkspace& ws = cache[q.get_device()];
        if (!ws.rows.defined() || ws.rows.size(0) < rows) {
            ws.rows = torch::zeros({rows, kDim}, q.options());
        }
        if (!ws.base.defined() || ws.base.size(0) < p.tokens) {
            ws.base = torch::empty({p.tokens}, q.options().dtype(torch::kInt));
        }
        p.c128_bf = reinterpret_cast<__nv_bfloat16*>(ws.rows.data_ptr());
        p.c128_base = ws.base.data_ptr<int>();
    }
    p.c128_rows = int(per_req);
    EncodeTmaDesc<2>(&maps.bf16, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, p.c128_bf,
                     {uint64_t(kDim), uint64_t(rows)},
                     {uint64_t(kDim) * sizeof(__nv_bfloat16)},
                     {uint32_t(kBlockDims), uint32_t(kWarp)},
                     CU_TENSOR_MAP_SWIZZLE_128B, "prefill_pair_mla c128 bf16");
    cudaLaunchAttribute attr = PdlLaunchAttribute();
    cudaLaunchConfig_t cfg{};
    cfg.stream = GetCurrentCUDAStream(q);
    cfg.attrs = &attr;
    cfg.numAttrs = use_pdl ? 1 : 0;
    cfg.blockDim = dim3(kPredqThreads);
    cfg.gridDim = dim3((p.tokens + kPredqWarps - 1) / kPredqWarps +
                       (rows + kPredqWarps - 1) / kPredqWarps);
    HALOPS_CUDA_CHECK(cudaLaunchKernelEx(&cfg, predq_kernel, p));
}

// Launch width, workspace and kernel launches. Many pairs: one cluster per pair
// (up to SMs / 2), split pairs merged in the kernel. Few pairs (2 * pairs <=
// clusters): every cluster launches with ~kDefaultTpc tiles and the combine
// kernel merges.
void launch_attention(PairMlaParams& p, const PairMlaMaps& maps,
                      const torch::Tensor& q,
                      void (*kernel)(PairMlaParams, PairMlaMaps),
                      int smem_bytes, bool use_pdl) {
    const int max_clusters = GetSmCount() / 2;
    TORCH_CHECK(max_clusters <= kMaxClusters,
                "more SMs than the pair kernel schedules");
    // Varlen: the exact pair count is on the device; (T + R) / 2 bounds it for
    // the launch width.
    const int64_t pairs = p.cu_pairs ? (int64_t(p.tokens) + p.n_req) / 2
                                     : int64_t(p.tokens) / p.tokens_per_req *
                                           ((p.tokens_per_req + 1) / 2);
    p.combine = 2 * pairs <= max_clusters ? 1 : 0;
    p.clusters =
        p.combine ? max_clusters : int(std::min<int64_t>(pairs, max_clusters));
    Workspace& ws = workspace(q, max_clusters);
    p.o_part = ws.o_part.data_ptr<float>();
    p.lse_part = ws.lse_part.data_ptr<float>();
    p.counter = ws.counter.data_ptr<int>();
    p.jobs = reinterpret_cast<MergeJob*>(ws.jobs.data_ptr<int>());
    p.n_jobs = ws.n_jobs.data_ptr<int>();
    p.total_g = ws.total_g.data_ptr<int>();
    p.sched_g = ws.sched_g.data_ptr<int>();
    const cudaStream_t stream = GetCurrentCUDAStream(q);
    HALOPS_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
    if (p.combine)
        HALOPS_CUDA_CHECK(cudaMemsetAsync(p.n_jobs, 0, sizeof(int), stream));
    cudaLaunchAttribute attrs[2];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim = {2, 1, 1};
    attrs[1] = PdlLaunchAttribute();
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(2 * p.clusters);
    cfg.blockDim = dim3(kThreads);
    cfg.dynamicSmemBytes = smem_bytes;
    cfg.stream = stream;
    cfg.attrs = attrs;
    cfg.numAttrs = use_pdl ? 2 : 1;
    HALOPS_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, p, maps));
    if (p.combine) {
        // Same stream, launched as a PDL dependent: its launch overlaps the
        // main kernel's drain, griddepcontrol.wait gates the reads.
        cudaLaunchAttribute cattr = PdlLaunchAttribute();
        cudaLaunchConfig_t ccfg{};
        ccfg.gridDim = dim3(kPlanes, 2, p.clusters);
        ccfg.blockDim = dim3(kCombSplit * kWarpGroup);
        ccfg.dynamicSmemBytes = 0;
        ccfg.stream = stream;
        ccfg.attrs = &cattr;
        ccfg.numAttrs = use_pdl ? 1 : 0;
        HALOPS_CUDA_CHECK(
            cudaLaunchKernelEx(&ccfg, pair_mla_combine_kernel, p));
    }
}

}  // namespace

torch::Tensor PrefillPairMla(
    const torch::Tensor& q, const torch::Tensor& swa_k_cache,
    const torch::Tensor& swa_indices, const torch::Tensor& swa_lengths,
    const torch::Tensor& c128_k_cache, const torch::Tensor& c128_indices,
    const torch::Tensor& c128_lengths,
    const std::optional<torch::Tensor>& attn_sink, const double softmax_scale,
    const int64_t tokens_per_req, const std::optional<torch::Tensor>& out_opt,
    const bool use_pdl, const std::optional<torch::Tensor>& cu_seqlens,
    const std::optional<torch::Tensor>& cu_pairs, const int64_t swa_shift,
    const std::optional<torch::Tensor>& q_positions,
    const std::optional<torch::Tensor>& q_freqs, const double q_norm_eps) {
    const CudaDeviceGuard device_guard(q.device());
    TORCH_CHECK(swa_shift >= 1 && swa_shift <= 128,
                "swa_shift must be in [1, 128]");
    const int64_t tokens = q.size(0);
    // Fused q_norm_rope (q_positions given): q is the raw projection, a [T, 64,
    // 512] or [T, 1, 64, 512] view with contiguous heads (any token stride).
    const bool qnr = q_positions.has_value();
    TORCH_CHECK(qnr == q_freqs.has_value(),
                "q_positions and q_freqs go together");
    if (qnr) {
        const torch::Tensor q3 =
            q.dim() == 4 && q.size(1) == 1 ? q.squeeze(1) : q;
        TORCH_CHECK(
            q.is_cuda() && q.scalar_type() == torch::kBFloat16 &&
                q3.dim() == 3 && q3.size(1) == kHeads && q3.size(2) == kDim &&
                q3.stride(2) == 1 && q3.stride(1) == kDim &&
                q3.stride(0) % kElemsPerChunk == 0 &&
                reinterpret_cast<uintptr_t>(q.data_ptr()) % kChunk == 0,
            "q must be a bfloat16 [T, 64, 512] view with contiguous heads");
    } else {
        TORCH_CHECK(q.is_cuda() && q.scalar_type() == torch::kBFloat16 &&
                        q.is_contiguous() &&
                        q.numel() == tokens * kHeads * kDim,
                    "q must be a contiguous bfloat16 [T, 64, 512] tensor");
    }
    const bool varlen = cu_seqlens.has_value();
    TORCH_CHECK(varlen == cu_pairs.has_value(),
                "cu_seqlens and cu_pairs go together");
    if (varlen) {
        for (const torch::Tensor* x : {&*cu_seqlens, &*cu_pairs}) {
            TORCH_CHECK(
                x->device() == q.device() && x->scalar_type() == torch::kInt &&
                    x->is_contiguous() && x->dim() == 1 &&
                    x->numel() == cu_seqlens->numel() && x->numel() >= 2,
                "cu_seqlens / cu_pairs must be contiguous int32 [R + 1] on q's "
                "device");
        }
    } else {
        TORCH_CHECK(tokens_per_req >= 2 && tokens % tokens_per_req == 0,
                    "tokens_per_req must be >= 2 and divide T");
    }
    torch::Tensor out =
        out_opt.has_value() ? *out_opt : torch::empty(q.sizes(), q.options());
    TORCH_CHECK(out.scalar_type() == torch::kBFloat16 && out.is_contiguous() &&
                    out.numel() == q.numel(),
                "out must be a contiguous bfloat16 [T, 64, 512] tensor");
    for (const torch::Tensor* x : std::initializer_list<const torch::Tensor*>{
             &swa_k_cache, &swa_indices, &swa_lengths, &c128_k_cache,
             &c128_indices, &c128_lengths, &out}) {
        TORCH_CHECK(x->device() == q.device(),
                    "all tensors must be on q's device");
    }
    TORCH_CHECK(
        !attn_sink.has_value() ||
            (attn_sink->device() == q.device() &&
             attn_sink->scalar_type() == torch::kFloat &&
             attn_sink->is_contiguous() && attn_sink->numel() == kHeads),
        "attn_sink must be a contiguous float32 [64] tensor");
    if (tokens == 0) return out;

    PairMlaParams p{};
    PairMlaMaps maps{};
    set_pool(p, maps, 0, c128_k_cache, c128_indices, c128_lengths, tokens);
    set_pool(p, maps, 1, swa_k_cache, swa_indices, swa_lengths, tokens);
    p.q = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr());
    p.sink = attn_sink.has_value() ? attn_sink->data_ptr<float>() : nullptr;
    p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
    p.tokens = tokens;
    p.tokens_per_req = varlen ? 0 : int(tokens_per_req);
    p.swa_shift = int(swa_shift);
    if (varlen) {
        p.cu_seqlens = cu_seqlens->data_ptr<int>();
        p.cu_pairs = cu_pairs->data_ptr<int>();
        p.n_req = int(cu_seqlens->numel() - 1);
    }
    p.scale_log2 = float(softmax_scale) * CUDART_L2E_F;
    torch::Tensor freqs;
    if (qnr) {
        const torch::Tensor& pos = *q_positions;
        TORCH_CHECK(
            pos.device() == q.device() && pos.is_contiguous() &&
                pos.numel() == tokens &&
                (pos.scalar_type() == torch::kLong ||
                 pos.scalar_type() == torch::kInt),
            "q_positions must be a contiguous int64 / int32 [T] tensor");
        freqs = q_freqs->is_complex()
                    ? torch::view_as_real(*q_freqs).flatten(-2)
                    : *q_freqs;
        TORCH_CHECK(
            freqs.device() == q.device() &&
                freqs.scalar_type() == torch::kFloat && freqs.is_contiguous() &&
                freqs.dim() == 2 && freqs.size(1) == kQRopeDims,
            "q_freqs must be complex64 [P, 32] or contiguous float32 [P, 64]");
        p.q_stride = q.stride(0);
        p.q_pos = pos.data_ptr();
        p.pos64 = pos.scalar_type() == torch::kLong;
        p.q_freqs = freqs.data_ptr<float>();
        p.q_eps = float(q_norm_eps);
        // Same boxes as below, one token's 64 heads apart by q_stride.
        EncodeTmaDesc<3>(&maps.q, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, p.q,
                         {uint64_t(kDim), uint64_t(kHeads), uint64_t(tokens)},
                         {uint64_t(kDim) * sizeof(__nv_bfloat16),
                          uint64_t(p.q_stride) * sizeof(__nv_bfloat16)},
                         {uint32_t(kQBoxDims), uint32_t(kHeads), 1u},
                         CU_TENSOR_MAP_SWIZZLE_128B, "prefill_pair_mla raw q");
    } else {
        EncodeTmaDesc<2>(&maps.q, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, p.q,
                         {uint64_t(kDim), uint64_t(tokens * kHeads)},
                         {uint64_t(kDim) * sizeof(__nv_bfloat16)},
                         {uint32_t(kQBoxDims), uint32_t(kHeads)},
                         CU_TENSOR_MAP_SWIZZLE_128B, "prefill_pair_mla q");
    }
    EncodeTmaDesc<2>(&maps.out, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, p.out,
                     {uint64_t(kDim), uint64_t(tokens * kHeads)},
                     {uint64_t(kDim) * sizeof(__nv_bfloat16)},
                     {uint32_t(kQBoxDims), uint32_t(kHeads)},
                     CU_TENSOR_MAP_SWIZZLE_128B, "prefill_pair_mla out");
    if (p.pool_tokens[0] > 0) predequant_c128(p, maps, q, use_pdl);
    // Short pairs (<= kQstMaxTiles tiles): Q fetch + O store outlast compute,
    // so kQst overlaps the next Q with this O; its 32 KB of Q buffers cost L1
    // on longer pairs. ld[0] + ld[1] bounds a pair's keys.
    const bool qst = (p.ld[0] + p.ld[1] + kTile - 1) / kTile <= kQstMaxTiles;
    const bool fp8 = p.c128_base == nullptr;
    using Kernel = void (*)(PairMlaParams, PairMlaMaps);
    constexpr Kernel kKernels[2][2][2] = {
        {{pair_mla_kernel<false, false, false>,
          pair_mla_kernel<false, false, true>},
         {pair_mla_kernel<false, true, false>,
          pair_mla_kernel<false, true, true>}},
        {{pair_mla_kernel<true, false, false>,
          pair_mla_kernel<true, false, true>},
         {pair_mla_kernel<true, true, false>,
          pair_mla_kernel<true, true, true>}}};
    const Kernel kernel = kKernels[qst][fp8][qnr];
    launch_attention(p, maps, q, kernel, qst ? kQstSmemBytes : kSmemBytes,
                     use_pdl);
    return out;
}

}  // namespace halops
