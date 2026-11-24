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
#include <cutlass/barrier.h>
#include <math_constants.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm80.hpp>
#include <cute/arch/mma_sm100_desc.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <initializer_list>
#include <mutex>
#include <optional>
#include <unordered_map>

#include "common/pdl.cuh"
#include "common/ptx.cuh"
#include "kernel/gemm/gemm_utils.h"
#include "kernel/mla/mla_intrinsics.cuh"

namespace halops {

using namespace mla;

namespace detail {

// A 2-CTA cluster attends pairs of tokens (2j, 2j+1) of a request over the
// union of their KV lists, so each KV tile is read and dequantized once for
// both tokens.  CTA c holds token 2j+c and dims [256c, 256c+256) of every key
// of a tile.  The UMMAs are 2-CTA with M = 128:
// - QK^T "rails": Q sits in TMEM with dims [0, 256) of the CTA's token on
//   lanes 0-63 and dims [256, 512) on lanes 64-127, B is CTA0's keys x low
//   dims and CTA1's keys x high dims (N = 128), so lanes 0-63 / 64-127 get the
//   two partial sums of S, added by the softmax.
// - PV: B is the same KV tile read MN-major, the N split giving CTA c the
//   output dims [256c, 256c+256) on its lanes 64c.. of every head.
// The keys of a pair form one stream: its c128 rows (padded to a multiple of
// 4, one gather4 group never mixes the caches) followed by its SWA rows, cut
// into tiles of 64.  The tiles of all pairs, pair after pair, are split evenly
// over the clusters; a pair cut by a range boundary is merged by the cluster
// holding its first tile, at the end of its range.  Small batches use one or
// two clusters per pair instead (see HcaDecodeAttention).
// Q comes in and O goes out by TMA in swizzled boxes staged in shared memory:
// a lane's row is strided in global memory, and per-lane loads and stores of
// it would hold the L1 a row per lane, stalling the softmax's shared memory
// accesses.
constexpr int kHeads = 64;
constexpr int kDim = 512;
constexpr int kNope = 448;
constexpr int kRowBytes = 576;
constexpr int kSfBytes = 8;
constexpr int kBf16Mant = 7;
constexpr int kBlockDims = 64;
constexpr int kHalfDims = kDim / 2;
constexpr int kTile = 64;
constexpr int kHcaTile = kTile;  // Cfg alias (member kTile would shadow this)
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
constexpr int kKvStages = 5;
// Merge staging ring: a split pair's parts are combined box by box, loading
// the next box while the previous one is reduced.  Each box is one KV stage.
constexpr int kMergeBuf = 4;
constexpr int kMergeShared = 1 << 16;  // fused merge: counter flag, see store_o
static_assert(kMergeBuf <= kKvStages);
// Pairs of a cluster's range whose lengths are kept in shared memory; a
// range with more (tiny pairs) loads the rest from global memory.
constexpr int kTableUnits = 32;
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
constexpr float kRescaleLog2 = 6.0f;

// Raw bytes per key row gathered by a CTA: the fp8 dims [0, 256) for CTA 0,
// the fp8 dims [256, 448) and the bf16 rope dims [448, 512) for CTA 1.  The
// rows land at the start of the KV stage and are dequantized in place.
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

// ======================= DSA (full-MLA) variant ==========================
// DSA caches a 656 B inline row — 512 fp8 latent | 4 f32 group scales (one
// per 128 latent dims) | 64 bf16 rope — and contracts QK over all 576 dims
// (512 latent + 64 rope).  The 512-col TMEM is already fully packed by HCA
// (O 256 + Q 128 + S 128), so the 64 rope dims cannot widen the TMEM Q tile;
// they ride a SECOND umma_ss whose Q operand comes from SMEM and accumulates
// into the same S.  The 512 latent splits 256/CTA (== kHalfDims, so the nope
// QK is structurally identical to HCA's 512 contraction) and the 64 rope
// splits 32/CTA.  Scales are an inline f32 multiplier per 128-dim group (vs
// HCA's separate UE8M0 sf block); a disabled key passes scale 0 -> zero row.
constexpr int kDsaNope = 512;                // fp8 latent dims / row
constexpr int kDsaRope = 64;                 // bf16 rope dims / row
constexpr int kDsaQk = kDsaNope + kDsaRope;  // 576 contraction
constexpr int kDsaScaleGroup = 128;          // latent dims per f32 scale
constexpr int kDsaScales = kDsaNope / kDsaScaleGroup;                    // 4
constexpr int kDsaScaleBytes = kDsaScales * sizeof(float);               // 16
constexpr int kDsaRopeBytes = kDsaRope * sizeof(__nv_bfloat16);          // 128
constexpr int kDsaRowBytes = kDsaNope + kDsaScaleBytes + kDsaRopeBytes;  // 656
constexpr int kDsaNopeHalf = kDsaNope / 2;  // 256 fp8 nope dims / CTA
constexpr int kDsaScaleOff = kDsaNope;      // 512: first f32 scale byte
constexpr int kDsaRopeOff = kDsaNope + kDsaScaleBytes;  // 528: first rope byte
// Per-CTA raw gather: each CTA takes its 256 fp8 nope dims and its two f32
// scales.  The nope lands at the stage head (then dequantizes in place like
// HCA).  Rope is NOT split across CTAs: a 2-CTA umma_ss sources operand A
// (rope-Q) from CTA0's SMEM only, so a per-CTA rope-dim split would feed
// dims-half 1 the wrong A (CTA0's dims) and compute a cross term.  Instead the
// whole 64-dim rope dot rides dims-half 0: CTA0 stages the full 64 rope dims
// (Q and K), CTA1's rope_kv is zeroed once, so half0 = full rope, half1 = 0,
// and softmax's pair sum (half0 + half1) counts the 64-dim rope exactly once.
// rope-Q staged in SMEM for the ss-MMA: kHeads rows x 64 rope dims, bf16.
constexpr int kDsaQRopeBytes =
    kHeads * kDsaRope * sizeof(__nv_bfloat16);    // 8 KB
constexpr int kDsaRopeSteps = kDsaRope / kUmmaK;  // 64/16 = 4 K-steps
// SMEM budget: DSA stages a 256-dim nope KV tile, the full 64 rope dims/key
// and 64 rope-Q dims (8 KB total), and drops HCA's sf block (scales inline).
// The low-concurrency combine reuses the KV stages as an O-part scratch and
// needs kStages*kStageBytes >= 128 KB (one O-part = kPlanes*kPlaneBytes), which
// 4 x 32 KB meets exactly.  Everything the tile touches — kHalf, the S TMEM
// width, the PV K-steps, the index-warp count, the stage byte stride — is
// derived from C::kTile so HCA and DSA share one body and the DSA tile is a
// tunable knob (see the option-A note below).
// DSA KV-tile width is a Cfg trait so it can be tuned independently of HCA.
// A 32-key / 8-stage narrow tile ("option A": deeper prefetch ring at constant
// SMEM to attack the KV-load long_scoreboard stall) was built and MEASURED: it
// REGRESSED (T=64 1.10x vs 1.48x, T=512 0.66x vs 0.95x) because halving the
// tile doubles the per-tile barrier / online-softmax / MMA overhead, which
// outweighs the latency win on this latency-bound path.  Kept at 64/4, the
// proven-best config; the trait plumbing stays so the tile remains tunable.
constexpr int kDsaTile = 64;
constexpr int kDsaHalf = kDsaTile / 2;                         // 32
constexpr int kDsaIndexWarps = kDsaTile / kWarp;               // 2
constexpr int kDsaBlockBytes = kDsaTile * kRowPitch;           // 8 KB
constexpr int kDsaStageBytes = kLocalBlocks * kDsaBlockBytes;  // 32 KB
constexpr int kDsaKvStages = 4;
constexpr int kDsaRopeStageBytes =
    kDsaTile * kDsaRope * sizeof(__nv_bfloat16);  // 8 KB

struct HcaSmem {
    alignas(kAtomBytes) uint8_t kv[kKvStages][kStageBytes];
    alignas(kAtomBytes) uint8_t p[2][kHeads * kRowPitch];
    alignas(kAtomBytes) uint8_t qo_stage[2][kQBoxBytes];  // [dims half] Q/O box
    alignas(kChunk) float xchg[kWarpGroup][kHalf];
    alignas(kChunk) uint8_t sf[kKvStages][kTile][kSfBytes];
    // bit j of word w: key 32w + j is valid for this CTA's token / has a row
    uint32_t valid[kIdxStages][kIndexWarps];
    uint32_t any[kIdxStages][kIndexWarps];
    int idx_raw[kIndexWarps][kIndexDepth][kWarp][2];
    float red[kWarpGroup];
    float2 ml[2][kWarpGroup];     // per unit parity: row max, partial row sum
    int start[3];                 // first pair, its first tile, tile count
    int4 unit_lens[kTableUnits];  // lengths of the range's pairs from start[0]
    int total;   // sum of schedule weights (== total tiles when uncapped)
    int wstart;  // schedule-weight start of the range's first pair
    int cap;     // effective per-pair weight cap this launch (kNoCap = none)
    int sched;   // clusters actually scheduled this launch (<= p.clusters)
    int comb;    // split pairs merged by the combine kernel this launch
    int n_mj;    // fused merge: the split pairs this cluster holds a part of
    int4 mj[2];  // (first part cluster, last part cluster, token a, pair * 2 +
                 // has_b)
    uint64_t q_full, q_empty, o_full, o_empty[2], merge_full[kMergeBuf];
    uint64_t qs_full;  // every unit's Q boxes land in qo_stage
    uint64_t ml_full[2], ml_empty[2];
    uint64_t idx_full[kIdxStages], idx_empty[kIdxStages];
    uint64_t raw_full[kKvStages], kv_full[kKvStages], kv_empty[kKvStages];
    uint64_t s_full[2], s_empty[2], p_full[2], pv_done[2];
    uint32_t tmem_base;
};
constexpr int kSmemBytes = sizeof(HcaSmem) + kAtomBytes;

// DSA shared memory.  Mirrors HcaSmem's field NAMES so the shared softmax,
// scheduling, PV and merge code templates on the smem type and reuses this
// struct unchanged; the differences are all in the KV data region: one fewer
// KV stage (kDsaKvStages), HCA's separate sf block dropped (DSA scales ride
// inline in the dequant input), and two added buffers — a per-stage rope KV
// tile and the rope-Q tile — that feed the second (ss) QK matmul.
struct DsaSmem {
    alignas(kAtomBytes) uint8_t kv[kDsaKvStages][kDsaStageBytes];
    alignas(kAtomBytes) uint8_t rope_kv[kDsaKvStages][kDsaRopeStageBytes];
    alignas(kAtomBytes) uint8_t p[2][kHeads * kRowPitch];
    alignas(kAtomBytes) uint8_t qo_stage[2][kQBoxBytes];
    alignas(
        kAtomBytes) uint8_t q_rope[kDsaQRopeBytes];  // rope-Q for the ss-MMA
    alignas(kChunk) float xchg[kWarpGroup][kDsaHalf];
    // This CTA's two inline f32 group scales per key (vs HCA's sf block).
    alignas(kChunk) float dsa_sc[kDsaKvStages][kDsaTile][kDsaScales / 2];
    uint32_t valid[kIdxStages][kDsaIndexWarps];
    uint32_t any[kIdxStages][kDsaIndexWarps];
    int idx_raw[kDsaIndexWarps][kIndexDepth][kWarp][2];
    float red[kWarpGroup];
    float2 ml[2][kWarpGroup];
    int start[3];
    int4 unit_lens[kTableUnits];
    int total;
    int wstart;
    int cap;
    int sched;
    int comb;
    int n_mj;
    int4 mj[2];
    uint64_t q_full, q_empty, o_full, o_empty[2], merge_full[kMergeBuf];
    uint64_t qs_full;
    uint64_t ml_full[2], ml_empty[2];
    uint64_t idx_full[kIdxStages], idx_empty[kIdxStages];
    uint64_t raw_full[kDsaKvStages], kv_full[kDsaKvStages],
        kv_empty[kDsaKvStages];
    uint64_t s_full[2], s_empty[2], p_full[2], pv_done[2];
    uint32_t tmem_base;
};
constexpr int kDsaSmemBytes = sizeof(DsaSmem) + kAtomBytes;
// The full-rope KV stages push DSA just past HCA's footprint, but it still
// fits the SM100 opt-in per-block SMEM cap (227 KB).
static_assert(kDsaSmemBytes <= 227 * 1024);

// Compile-time variant selector.  kDsa == false reproduces HCA byte-for-byte
// (every DSA-only statement lives under `if constexpr (C::kDsa)`), kDsa ==
// true selects the DSA smem, the 656 B inline-row path and the narrow
// (32-key) KV tile with an 8-deep prefetch ring.  Every tile-derived constant
// is a Cfg trait so the shared softmax / scheduling / PV / index code uses
// C::kTile &c. uniformly; HcaCfg resolves them to HCA's 64-key values.
// kFuse compiles in the fused merge (see fused_merge); the launches that
// cannot use it run a build without it, whose main loop the merge code would
// otherwise slow down.
template <bool kDsaV, bool kFuseV = !kDsaV>
struct Cfg {
    static constexpr bool kDsa = kDsaV;
    static constexpr bool kFuse = kFuseV;
    using Smem = cute::conditional_t<kDsaV, DsaSmem, HcaSmem>;
    static constexpr int kStages = kDsaV ? kDsaKvStages : kKvStages;
    static constexpr int kTile = kDsaV ? kDsaTile : kHcaTile;
    static constexpr int kHalf = kTile / 2;
    static constexpr int kTmemSCols = kTile;
    static constexpr int kPvSteps = kTile / kUmmaK;
    static constexpr int kIndexWarps = kTile / kWarp;
    static constexpr int kBlockBytes = kTile * kRowPitch;
    static constexpr int kStageBytes = kLocalBlocks * kBlockBytes;
};
using HcaCfg = Cfg<false>;
using HcaPlainCfg = Cfg<false, false>;
using DsaCfg = Cfg<true>;

struct HcaMaps {
    CUtensorMap raw[2][2];  // [cta][seg] 16-bit view of the rows, raw_pitch box
    CUtensorMap q;    // q rows (token x head), 64-dim swizzled boxes of 64 rows
    CUtensorMap out;  // out rows, boxes as for q
};

// One split pair the separate combine kernel must finish: its first token,
// whether it has a second, and the schedule-weight span used to recompute the
// part clusters with cluster_of (identical to the main kernel's view).
struct MergeJob {
    int a, has_b, w0, wn;
};

struct HcaParams {
    const int* idx[2];  // [0] c128, [1] SWA: [T, ld] pool token indices
    const int* len[2];
    int ld[2];
    const uint8_t* pool[2];
    int64_t pool_stride[2];
    int page_shift[2];  // log2 of the tokens per page
    int page_rows[2];   // TMA rows (576 B) per page
    int64_t pool_tokens[2];
    const __nv_bfloat16* q;
    const float* sink;
    __nv_bfloat16* out;
    float* o_part;    // [clusters][2][kPlanes][128][4] normalized partial O
    float* lse_part;  // [clusters][2][64] their log2 sums
    int* counter;     // [clusters][2] parts published of a cluster's split pair
    int tokens;
    int tokens_per_req;  // t; tokens are paired within a request, not globally
    int clusters;
    int cap;   // manual per-pair weight cap override (tiles); kNoCap = use smax
    int smax;  // adaptive cap target: clusters a pair may span; 0 = no cap
    int nomerge;  // MEASUREMENT ONLY: skip the serial merge (wrong output) to
                  // time the main kernel's merge-free compute floor
    int pipe;     // 1 = pipelined box-ring merge; 0 = serial merge (fallback)
    int combine;  // 1 = split pairs are merged by the separate combine kernel:
                  // every part (merger included) writes its partial, the merger
                  // records a MergeJob, and no in-kernel merge runs.
    int fuse;     // with combine: the parts' clusters merge the pair themselves
                  // after their last unit (each a slice of planes), no combine
                  // kernel is launched
    MergeJob* jobs;  // [pairs] slot per pair when combine: the merger's job,
                     // wn == 0 for an unsplit pair (rewritten every launch)
    int* total_g;    // the capped tile total, for cluster_of in combine
    int eff;  // 1 = size the scheduled cluster count to the batch's tile work;
              // 2 = pick eff 1 or legacy per launch from the device lengths;
              // 3 = legacy, launched wide, one cluster per pair or all of them
    int tpc;  // work-aware target tiles per scheduled cluster (eff only)
    // eff 2 k-split (find_range): max parts per pair (0: off), its cost
    // slope b per part, 1.17 / b and 1 / P, precomputed from P on the host.
    int ks_kmax;
    float ks_b, ks_a, ks_rp;
    int* eff_g;  // clusters scheduled (sched), for the combine kernel;
                 // 0 = this launch merged inline, combine has no work
    float scale_log2;
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

CUTE_DEVICE void tma_gather4(const CUtensorMap* map, uint64_t* bar, void* dst,
                             int col, int r0, int r1, int r2, int r3) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4.mbarrier:"
        ":complete_tx::bytes [%0], [%1, {%2, %3, %4, %5, %6}], [%7];" ::"r"(
            smem_u32(dst)),
        "l"(map), "r"(col), "r"(r0), "r"(r1), "r"(r2), "r"(r3),
        "r"(smem_u32(bar))
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

// DSA variant: the scale is a direct fp32 multiplier (one per 128-dim group),
// not HCA's power-of-two UE8M0 code, so it is folded in as bf16 here (the P
// matmul rounds to bf16 anyway).  A disabled key passes scale 0.0f, which
// zeroes the dequantized row exactly as the HCA mask does.
CUTE_DEVICE uint4 dequant8_f32(uint32_t lo, uint32_t hi, float scale) {
    const uint32_t raw[2] = {lo, hi};
    const auto* f8 = reinterpret_cast<const __nv_fp8x2_e4m3*>(raw);
    const nv_bfloat16 s = __float2bfloat16(scale);
    uint4 out;
    auto* o = reinterpret_cast<nv_bfloat162*>(&out);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kElemsPerChunk / 2; ++i) {
        o[i] = fp8x2_to_bf16x2_with_scale(f8[i], s);
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

CUTE_DEVICE void cp_async16(void* dst, const void* src) {
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem_u32(dst)),
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

// The key stream of tokens a = 2j and b = 2j+1: position u of the c128
// segment is b's row u (a's list is a prefix of it) or a's row u where b has
// none, then the SWA window of b followed by the oldest row of a (a's window
// is b's shifted by one).  A row counts for a token only if its own list has
// it at that place, so inputs without this structure are masked instead of
// read out of bounds.
struct PairGeom {
    int a, len[2][2], n_c4, n_s, n_tiles;  // len[seg][token of pair]
    bool has_b;  // false for a lone (odd-request) token
};

// The lengths of a pair's tokens, [seg][token].
struct PairLens {
    int v[2][2];
};

// A pair's first token and whether it has a second.  Tokens are paired within
// a request (t tokens each, laid out [r*t, (r+1)*t)): a pair never straddles a
// request, and an odd request leaves its last token unpaired ("lone").  CTA 1
// of a lone pair has no token of its own.
struct PairMap {
    int a;
    bool has_b;
};

CUTE_DEVICE PairMap pair_map(const HcaParams& p, int pair) {
    const int ppr = (p.tokens_per_req + 1) / 2;  // pairs per request
    const int r = pair / ppr, k = pair - r * ppr;
    PairMap m;
    m.a = r * p.tokens_per_req + 2 * k;
    m.has_b = 2 * k + 1 < p.tokens_per_req;
    return m;
}

CUTE_DEVICE PairLens load_lens(const HcaParams& p, int pair) {
    const PairMap m = pair_map(p, pair);
    PairLens l;
    CUTLASS_PRAGMA_UNROLL
    for (int seg = 0; seg < 2; ++seg) {
        l.v[seg][0] = __ldg(p.len[seg] + m.a);
        l.v[seg][1] = m.has_b ? __ldg(p.len[seg] + m.a + 1) : 0;
    }
    return l;
}

CUTE_DEVICE PairGeom pair_geom(const HcaParams& p, int pair, const PairLens& l,
                               int tile) {
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
    g.n_s = max(g.len[1][1], g.len[1][0] > 0 ? g.len[1][0] + 1 : 0);
    g.n_tiles = max((g.n_c4 + g.n_s + tile - 1) / tile, 1);
    return g;
}

// Persistent schedule: cluster c takes tiles [c * total / C, (c + 1) * total
// / C) of the stream.  A unit is the part of one pair inside a cluster's
// range; the parts of a split pair after its first write normalized partial
// O and log2 sums to slot c of their cluster, merged by the first part.
CUTE_DEVICE int range_lo(int c, int total, int clusters) {
    return int(uint32_t(c) * uint32_t(total) / uint32_t(clusters));
}

// The cluster whose range holds stream tile pos: the last c with
// range_lo(c) <= pos, i.e. c * total < (pos + 1) * clusters.  A split pair's
// parts are in the clusters from the one holding its first tile on.
CUTE_DEVICE int cluster_of(int clusters, int total, int pos) {
    return int((uint64_t(pos + 1) * clusters - 1) / uint64_t(total));
}

constexpr int kNoCap = 0x7fffffff;
constexpr int kDefaultSmax =
    6;                          // adaptive cap target: clusters a pair may span
constexpr int kCapFloor = 8;    // never cap below this many tiles
constexpr int kDefaultTpc = 3;  // work-aware target tiles per scheduled cluster
// A pair's schedule weight is w = min(n_tiles, cap).  A part owning weight
// offset [a, b) of the pair owns real tiles [w_to_tile(a), w_to_tile(b)); so a
// long pair is spread over fewer clusters (bounded merge fan-in) while each of
// those clusters does proportionally more of its tiles.
CUTE_DEVICE int w_to_tile(int wofs, int n_tiles, int w) {
    return int(int64_t(wofs) * n_tiles / w);
}

// The effective per-pair weight cap for a launch.  A manual override (p.cap)
// wins; otherwise the adaptive policy keeps a long pair's split near p.smax
// parts: a cluster owns about total/clusters tiles, so capping a pair at
// smax * ceil(total / clusters) bounds its span (and its serial merge tail) to
// ~smax clusters, while short pairs (n_tiles below the cap) are left untouched.
CUTE_DEVICE int effective_cap(const HcaParams& p, int total_unc, int clusters) {
    if (p.cap != kNoCap) return p.cap;
    if (p.smax <= 0) return kNoCap;
    const int quota = (total_unc + clusters - 1) / clusters;
    return max(p.smax * quota, kCapFloor);
}

// Clusters to actually schedule this launch.  With p.eff the kernel is launched
// wide (p.clusters == max) and sizes itself to the batch: ~p.tpc tiles per
// cluster keeps every split part a couple of UMMA tiles (below that a pair
// fragments into sub-tile parts the combine must then re-merge, which the
// pole-length sweep showed regresses short-KV batches).  Clusters past the
// scheduled count see an empty range and exit, like the tiny-batch path.
// ceil(x / d) for 0 <= x < 2^20, d > 0: a float estimate and an integer
// correction, as integer division is a long dependent chain in find_range.
CUTE_DEVICE int ceil_div_f(int x, int d) {
    int q = __float2int_rz(__fdividef(float(x), float(d)));
    q += (q + 1) * d <= x;
    q -= q * d > x;  // floor
    return q + (q * d < x);
}

CUTE_DEVICE int sched_clusters(const HcaParams& p, int total_unc) {
    if (!p.eff || p.eff == 3) return p.clusters;
    const int tpc = p.tpc > 0 ? p.tpc : 3;
    int s = ceil_div_f(total_unc, tpc);
    return max(1, min(s, p.clusters));
}

struct Unit {
    PairGeom g;
    int off, n;  // real tiles [off, off + n) of the pair
    int w0, wn;  // pair's schedule-weight start and weight (min(n_tiles, cap))
    int pair;
    bool split;
};

struct UnitCursor {
    int pair, off, left, wpair0, cluster, first_pair, cap, tile;
    CUTE_DEVICE bool next(const HcaParams& p, const int4* table, Unit& u) {
        if (left == 0) return false;
        PairLens lens;
        const int rel = pair - first_pair;
        if (rel < kTableUnits) {
            const int4 v = table[rel];
            lens = {{{v.x, v.y}, {v.z, v.w}}};
        } else {
            lens = load_lens(p, pair);
        }
        u.g = pair_geom(p, pair, lens, tile);
        u.pair = pair;
        const int wn = min(u.g.n_tiles, cap);
        u.off = off;
        u.n = min(u.g.n_tiles - off, left);
        u.w0 = wpair0;
        u.wn = wn;
        u.split = u.n != u.g.n_tiles;
        left -= u.n;
        off += u.n;
        if (off == u.g.n_tiles) {
            ++pair;
            off = 0;
            wpair0 += wn;
        }
        return true;
    }
};

// All threads count the tiles of every pair and scan them; the result is
// the first pair and tile of this cluster's range.
template <class C>
CUTE_DEVICE void find_range(const HcaParams& p, typename C::Smem& sm,
                            int cluster) {
    __shared__ int part[kThreads / kWarp], part_max[kThreads / kWarp];
    __shared__ int
        part2[kThreads / kWarp];  // pass 2's, so pass 1's need no barrier
    const int pairs =
        (p.tokens / p.tokens_per_req) * ((p.tokens_per_req + 1) / 2);
    const int per = (pairs + kThreads - 1) / kThreads;
    const int first = threadIdx.x * per;
    const int lane = threadIdx.x % kWarp, warp = threadIdx.x / kWarp;
    // The lengths of a thread's first kKeep pairs are kept for the table.
    constexpr int kKeep = 2;
    int4 keep[kKeep];
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < kKeep; ++j) keep[j] = make_int4(0, 0, 0, 0);
    // A pair's lengths: kept if this thread loaded them, else loaded.  Each
    // reload is an L1/L2 round trip on the launch's critical path.
    const auto lens_of = [&](int q) {
        const int j = q - first;
        if (j >= 0 && j < min(per, kKeep)) {
            int4 v = keep[0];
            CUTLASS_PRAGMA_UNROLL
            for (int k = 1; k < kKeep; ++k) {
                if (j == k) v = keep[k];
            }
            return PairLens{{{v.x, v.y}, {v.z, v.w}}};
        }
        return load_lens(p, q);
    };

    // Pass 1: uncapped tile total, to size the adaptive cap.
    int sum_unc = 0, max_unc = 0;
    for (int q = first, j = 0; q < min(first + per, pairs); ++q, ++j) {
        const PairLens l = load_lens(p, q);
        CUTLASS_PRAGMA_UNROLL
        for (int k = 0; k < kKeep; ++k) {
            if (j == k)
                keep[k] = make_int4(l.v[0][0], l.v[0][1], l.v[1][0], l.v[1][1]);
        }
        const int n = pair_geom(p, q, l, C::kTile).n_tiles;
        sum_unc += n;
        max_unc = max(max_unc, n);
    }
    // Scanned here too: uncapped weights (no cap) need no second pass.
    int incl_unc = sum_unc;
    {
        int m = max_unc;
        CUTLASS_PRAGMA_UNROLL
        for (int d = 1; d < kWarp; d *= 2) {
            const int v = __shfl_up_sync(~0u, incl_unc, d);
            if (lane >= d) incl_unc += v;
            m = max(m, __shfl_xor_sync(~0u, m, d));
        }
        if (lane == kWarp - 1) {
            part[warp] = incl_unc;
            part_max[warp] = m;
        }
    }
    __syncthreads();
    int total_unc = 0, n_max = 0, base_unc = 0;
    for (int w = 0; w < kThreads / kWarp; ++w) {
        base_unc += w < warp ? part[w] : 0;
        total_unc += part[w];
        n_max = max(n_max, part_max[w]);
    }
    int sched = sched_clusters(p, total_unc);
    bool comb = p.combine;
    // One cluster per pair, ranges on pair boundaries: no pair splits, so no
    // merge at all, at the price of the longest pair's tiles.
    bool aligned = false;
    if (p.eff == 2) {
        bool spread = false;  // sched_clusters' count, legacy merge
        // Auto gate: the host cannot see the KV lengths (only the index
        // width), so pick per launch.  Each schedule's cost is a linear model
        // in its busiest cluster's tiles and the pair count P, fitted over KV
        // lengths 8K-512K and T2-76.  The legacy path (2 clusters per pair,
        // capped at smax, inline merge) has h0 = max(U / 2P, n_max / smax)
        // tiles; eff 1 (wide split, combine kernel) has h1 = U / sched.
        // Without the fused merge, eff 1 wins iff 4 (h0 - h1) > 3P - 8 (in
        // quarter tiles).  With the merge fused (no combine launch, see
        // fused_merge) the costs c0 (legacy), c1 (wide split) and cu (the
        // aligned schedule, no merge) below are compared and the cheapest
        // wins; 8K is always aligned.
        const int pp = max(pairs, 1);
        const int h0 = max(ceil_div_f(total_unc, 2 * pp),
                           (n_max + kDefaultSmax - 1) / kDefaultSmax);
        const int h1 = ceil_div_f(total_unc, sched);
        if (p.fuse) {
            const int cu = 1005 + 114 * n_max + 3 * pp;  // in 1/100 of a usec
            const int c0 = 1410 + 116 * h0 + 11 * pp;
            const int c1 = 1540 + 124 * h1 + 15 * pp;
            comb = p.combine && c1 < min(c0, cu);
            aligned = !comb && cu <= c0;
            if (float(p.clusters) * p.ks_rp < 3.f) {
                // P > clusters / 3 (no k-split): a second fit over KV lengths
                // 8K-1M, T16-72 and 8 tokens per request adds d3, the spread
                // schedule (eff 3's: every cluster, inline merge, capped),
                // which beats the fused merge's fan-in at long KV.
                const int d0 = 1362 + 115 * h0 + 14 * pp;
                const int d1 = 1813 + 116 * h1 + 14 * pp;
                const int d3 = 1417 + 119 * h1 + 8 * pp +
                               __float2int_rn(164.5f * sched * p.ks_rp);
                const int dm = min(min(d0, d1), d3);
                aligned = cu <= dm;
                spread = !aligned && d3 == dm;
                comb = p.combine && !aligned && !spread && d1 < d0;
            }
            // Equal pairs: k parts per pair, sched = k P, puts every range on
            // pair boundaries (range_lo(j k) = j n); c1's ranges straddle
            // pairs unless sched is a multiple of P.  The k-split cost is
            // 1.17 h + b k (in tiles), b = 0.08 + 0.06 P, h = ceil(n / k),
            // so the best k is near sqrt(1.17 n / b).  k = 2 stays on c0;
            // k <= n keeps every part non-empty.  Float math avoids the
            // integer divisions' dependent chains.
            const int kmax = min(n_max, p.ks_kmax);
            if (C::kFuse && !aligned && kmax >= 3 && n_max * pp == total_unc) {
                const float n = float(n_max), b = p.ks_b, fs = sched * p.ks_rp;
                // h near sqrt(n / a) = n / k, no smaller than ceil(n / kmax)
                const float hmin = ceilf(__fdividef(n, kmax) - 0.01f);
                const float hs = fmaxf(floorf(n * rsqrtf(n * p.ks_a)), hmin);
                float bc =
                    comb ? 1.17f * h1 + b * fs +
                               (fabsf(fs - rintf(fs)) > 1e-3f ? 1.8f : 0.f)
                         : 1.17f * ((n_max + 1) / 2) + 2.f * b;
                bc -= 0.3f;  // margin
                int best = 0;
                CUTLASS_PRAGMA_UNROLL
                for (int j = 0; j < 2; ++j) {
                    const float h = hs + j;
                    const float k =
                        ceilf(__fdividef(n, h) - 0.01f);  // fewest parts
                    const float c = 1.17f * h + b * k;
                    if (k >= 3.f && c < bc) {
                        bc = c;
                        best = int(k);
                    }
                }
                if (best) {
                    comb = true;
                    sched = best * pp;
                }
            }
        } else {
            comb = p.combine && 4 * (h0 - h1) > 3 * pp - 8;
        }
        if (!comb && !spread) sched = min((aligned ? 1 : 2) * pp, p.clusters);
    } else if (p.eff == 3) {
        // Legacy above the eff 1 region (P <= clusters < 2P): one cluster per
        // pair leaves the rest idle, which a long pair cannot afford.  Spread
        // over every cluster when that shortens the busiest one by >= 6 tiles,
        // enough to repay the split's merge; short KV keeps one per pair.
        const int wide = (total_unc + p.clusters - 1) / p.clusters;
        if (n_max - wide < 6) {
            sched = min(max(pairs, 1), p.clusters);
            aligned = pairs <= p.clusters;
        }
    }
    // eff 1 leaves the split uncapped; the legacy path keeps the adaptive cap.
    const int cap = (p.eff == 2 && comb) || aligned
                        ? kNoCap
                        : effective_cap(p, total_unc, sched);
    if (threadIdx.x == 0) {
        sm.cap = cap;
        sm.sched = sched;
        sm.comb = comb;
        if (p.combine && cluster == 0) *p.eff_g = comb ? sched : 0;
    }

    // Pass 2: capped schedule weights, scanned to this cluster's range.
    // A cap no pair reaches leaves the weights pass 1's tiles: reuse its scan.
    int sum = sum_unc, incl = incl_unc, base = base_unc, total = total_unc;
    if (cap < n_max) {
        sum = 0;
        for (int q = first; q < min(first + per, pairs); ++q)
            sum += min(pair_geom(p, q, lens_of(q), C::kTile).n_tiles, cap);
        incl = sum;
        CUTLASS_PRAGMA_UNROLL
        for (int d = 1; d < kWarp; d *= 2) {
            const int v = __shfl_up_sync(~0u, incl, d);
            if (lane >= d) incl += v;
        }
        if (lane == kWarp - 1) part2[warp] = incl;
        __syncthreads();
        base = 0;
        total = 0;
        for (int w = 0; w < kThreads / kWarp; ++w) {
            base += w < warp ? part2[w] : 0;
            total += part2[w];
        }
    }
    // Clamp to sched so clusters past the scheduled count get an empty range
    // (lo == hi == total) and drop out, exactly like the tiny-batch tail.
    const int lo = range_lo(min(cluster, sched), total, sched);
    const int hi = range_lo(min(cluster + 1, sched), total, sched);
    int pre = base + incl - sum;
    if (aligned) {
        if (first <= cluster && cluster < min(first + per, pairs)) {
            for (int q = first; q < cluster; ++q)
                pre += pair_geom(p, q, lens_of(q), C::kTile).n_tiles;
            sm.start[0] = cluster;
            sm.wstart = pre;
            sm.start[1] = 0;
            sm.start[2] =
                pair_geom(p, cluster, lens_of(cluster), C::kTile).n_tiles;
        } else if (threadIdx.x == 0 && cluster >= pairs) {
            sm.start[0] = 0;
            sm.start[2] = 0;
        }
    } else if (sum > 0 && pre <= lo && lo < pre + sum) {
        for (int q = first;; ++q) {
            const PairGeom g = pair_geom(p, q, lens_of(q), C::kTile);
            const int w = min(g.n_tiles, cap);
            if (lo < pre + w) {
                sm.start[0] = q;
                sm.wstart = pre;  // first pair's schedule-weight start
                if (cap == kNoCap) {
                    sm.start[1] = lo - pre;
                    sm.start[2] = hi - lo;
                } else {
                    // Weight boundaries map to real-tile boundaries per pair;
                    // sum the mapped tiles over the pairs this range touches.
                    sm.start[1] = w_to_tile(lo - pre, g.n_tiles, w);
                    int tiles =
                        w_to_tile(min(hi - pre, w), g.n_tiles, w) - sm.start[1];
                    int wpos = pre + w;
                    for (int qq = q + 1; wpos < hi; ++qq) {
                        const PairGeom g2 =
                            pair_geom(p, qq, lens_of(qq), C::kTile);
                        const int w2 = min(g2.n_tiles, cap);
                        tiles += w_to_tile(min(hi - wpos, w2), g2.n_tiles, w2);
                        wpos += w2;
                    }
                    sm.start[2] = tiles;
                }
                break;
            }
            pre += w;
        }
    }
    if (threadIdx.x == 0 && lo == hi && !aligned) {
        sm.start[0] = 0;
        sm.start[2] = 0;
    }
    if (threadIdx.x == 0) {
        sm.total = total;
        sm.n_mj = 0;
    }
    if (sm.comb && cluster == 0 && threadIdx.x == 0) *p.total_g = total;
    __syncthreads();
    if constexpr (C::kFuse) {
        // Fused merge: an unscheduled cluster helps merge pair (cluster -
        // sched) % P (see fused_merge), so a pair split over few clusters is
        // not merged by those alone.  The thread holding the pair finds its
        // part clusters.
        if (comb && p.fuse && cluster >= sched && pairs > 0) {
            const int q = (cluster - sched) % pairs;
            if (q >= first && q < min(first + per, pairs)) {
                int wq = base + incl - sum;
                for (int qq = first; qq < q; ++qq)
                    wq += min(pair_geom(p, qq, lens_of(qq), C::kTile).n_tiles,
                              cap);
                const PairGeom g = pair_geom(p, q, lens_of(q), C::kTile);
                const int f = cluster_of(sched, total, wq);
                const int l =
                    cluster_of(sched, total, wq + min(g.n_tiles, cap) - 1);
                if (l > f && (cute::block_rank_in_cluster() == 0 || g.has_b)) {
                    sm.mj[0] = make_int4(f, l, g.a, q * 2 + g.has_b);
                    sm.n_mj = 1;
                }
            }
            __syncthreads();
        }
    }
    // The lengths of the range's first pairs, read by every role's cursor:
    // from the threads that loaded them (or loaded again past kKeep pairs
    // per thread).
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
CUTE_DEVICE UnitCursor start_cursor(const HcaParams& p,
                                    const typename C::Smem& sm, int cluster) {
    return {sm.start[0], sm.start[1], sm.start[2], sm.wstart,
            cluster,     sm.start[0], sm.cap,      C::kTile};
}

// Index warp w resolves keys [32w, 32w + 32) of every tile, one per lane:
// the union key, this CTA's token validity, the TMA row coordinate; then it
// gathers its 8 four-row groups into the tile's KV stage and copies the 8
// scale bytes of each key with cp.async, both completing on raw_full.  The
// raw indices of kIndexDepth tiles ahead are in flight meanwhile.
struct KeyRaw {
    bool in_a, in_b;
    int seg;
};

template <class C>
CUTE_DEVICE void index_warp(const HcaParams& p, const HcaMaps& maps,
                            typename C::Smem& sm, int cta, int w, int lane,
                            int cluster) {
    const int j = w * kWarp + lane;
    UnitCursor ahead = start_cursor<C>(p, sm, cluster);
    Unit au;
    int a_left = 0, a_tile = 0;
    const auto fetch = [&](KeyRaw& r, int slot) {
        if (a_left == 0 && ahead.next(p, sm.unit_lens, au)) {
            a_left = au.n;
            a_tile = au.off;
        }
        r = {false, false, 1};
        if (a_left > 0) {
            const int pos = a_tile * C::kTile + j;
            const int seg = pos < au.g.n_c4 ? 0 : 1;
            const int ub = seg == 0 ? pos : pos - au.g.n_c4, ua = ub - seg;
            const int* row = p.idx[seg] + int64_t(au.g.a) * p.ld[seg];
            r = {ua >= 0 && ua < au.g.len[seg][0], ub < au.g.len[seg][1], seg};
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
    if constexpr (C::kDsa) {
        // CTA1 never loads rope from DRAM: the whole 64-dim rope dot rides
        // dims-half 0 against CTA0's rope_kv, so CTA1's rope_kv must be zero
        // for its dims-half to add nothing.  Zero it once here (it is never
        // rewritten), fenced for the ss-MMA's async-proxy read.  This halves
        // the rope DRAM/L2 traffic vs both CTAs staging the full rope.  CTA0
        // is the critical path, so staging the rope on both CTAs would only
        // add L2 traffic without shortening it.
        if (cta == 1) {
            CUTLASS_PRAGMA_UNROLL
            for (int s = 0; s < C::kStages; ++s) {
                CUTLASS_PRAGMA_UNROLL
                for (int c = 0; c < kDsaRopeBytes / kChunk; ++c) {
                    *reinterpret_cast<uint4*>(
                        sm.rope_kv[s] + sw128_offset(j, c)) = uint4{0, 0, 0, 0};
                }
            }
            cutlass::arch::fence_view_async_shared();
        }
    }
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
            Bar::arrive(&sm.idx_full[is]);
        }
        // The tile's stage is free once PV(i - C::kStages) is done.
        const int s = i % C::kStages;
        Bar::wait(&sm.kv_empty[s], ((i / C::kStages) & 1) ^ 1);
        if constexpr (C::kDsa) {
            // DSA: the inline f32 group scales (this CTA's half) and, on CTA0
            // only, the full 64 rope dims ride cp.async from the 656 B row onto
            // raw_full beside the TMA nope.  CTA1's rope_kv stays zero (zeroed
            // once at entry), so the full rope dot lands in dims-half 0.
            if (any) {
                const uint8_t* const base = p.pool[seg] +
                                            int64_t(page) * p.pool_stride[seg] +
                                            int64_t(row) * kDsaRowBytes;
                cp_async8(sm.dsa_sc[s][j],
                          base + kDsaScaleOff +
                              cta * (kDsaScales / 2) * int(sizeof(float)));
                if (cta == 0) {
                    CUTLASS_PRAGMA_UNROLL
                    for (int c = 0; c < kDsaRopeBytes / kChunk; ++c) {
                        cp_async16(sm.rope_kv[s] + sw128_offset(j, c),
                                   base + kDsaRopeOff + c * kChunk);
                    }
                }
            }
            cp_async_arrive_noinc(&sm.raw_full[s]);
            const int src = lane % kGroupsPerWarp * kGather;
            const int r0 = __shfl_sync(~0u, coord, src);
            const int r1 = __shfl_sync(~0u, coord, src + 1);
            const int r2 = __shfl_sync(~0u, coord, src + 2);
            const int r3 = __shfl_sync(~0u, coord, src + 3);
            if (lane == 0) {
                cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(
                    &sm.raw_full[s], kWarp * kDsaNopeHalf);
            }
            __syncwarp();
            if (lane < kGroupsPerWarp) {
                tma_gather4(&maps.raw[cta][0], &sm.raw_full[s],
                            sm.kv[s] + (w * kWarp + src) * kDsaNopeHalf,
                            cta * kHalfDims / 2, r0, r1, r2, r3);
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
    for (int i = 0; i < n; i += kIndexDepth) {
        CUTLASS_PRAGMA_UNROLL
        for (int k = 0; k < kIndexDepth; ++k) {
            if (i + k < n) step(i + k, ring[k], k);
        }
    }
}

// Thread t of the dequant warp group converts the 16-byte chunks e = t + 128k
// of the gathered rows in place: all reads, a warp-group barrier, then the
// bf16 writes.  A wavefront (8 lanes) reads 8 consecutive chunks; chunks of an
// odd 64-dim block store their two bf16 halves in the other order, so its
// stores also hit 8 distinct bank groups.
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
    if constexpr (C::kDsa) {
        // DSA: both CTAs hold 256 fp8 latent dims (== kHalfDims, HCA's CTA-0
        // layout), no rope in the nope stage, and an inline f32 scale per
        // 128-dim group.  A disabled key dequantizes with scale 0 -> zero row.
        constexpr int kChunks = kHalfDims / kChunk;  // 16 nope chunks/CTA
        constexpr int kLoads =
            C::kTile * kChunks / kWarpGroup;                   // 4 at kTile 32
        constexpr int kPitch = kDsaNopeHalf;                   // 256 B nope row
        constexpr int kGroupChunks = kDsaScaleGroup / kChunk;  // 8 chunks / f32
        for (int i = 0; i < n; ++i) {
            const int s = i % C::kStages, is = i % kIdxStages;
            Bar::wait(&sm.idx_full[is], (i / kIdxStages) & 1);
            // any[w] holds the 32 keys of index warp w; a tile narrower than 64
            // keys runs a single index warp, so any[is][1] does not exist and
            // row/kWarp is always 0 there.
            uint32_t any[2];
            any[0] = sm.any[is][0];
            if constexpr (C::kIndexWarps > 1)
                any[1] = sm.any[is][1];
            else
                any[1] = 0u;
            Bar::wait(&sm.raw_full[s], (i / C::kStages) & 1);
            uint8_t* const stage = sm.kv[s];
            uint4 raw[kLoads];
            float sc[kLoads];
            CUTLASS_PRAGMA_UNROLL
            for (int k = 0; k < kLoads; ++k) {
                const int e = t + k * kWarpGroup;
                const int row = e / kChunks, c = e % kChunks;
                raw[k] = *reinterpret_cast<const uint4*>(stage + row * kPitch +
                                                         c * kChunk);
                const float s0 = sm.dsa_sc[s][row][c / kGroupChunks];
                sc[k] = (any[row / kWarp] >> (row % kWarp)) & 1 ? s0 : 0.0f;
            }
            __syncwarp();
            if (t % kWarp == 0) Bar::arrive(&sm.idx_empty[is]);
            NamedBar::sync(kWarpGroup, kDequantBar);
            CUTLASS_PRAGMA_UNROLL
            for (int k = 0; k < kLoads; ++k) {
                const int e = t + k * kWarpGroup;
                const int row = e / kChunks, c = e % kChunks;
                const bool odd = (c / 4) & 1;
                const int q0 = c % 4 * 2;
                uint8_t* const block = stage + c / 4 * C::kBlockBytes;
                const uint4 first =
                    dequant8_f32(odd ? raw[k].z : raw[k].x,
                                 odd ? raw[k].w : raw[k].y, sc[k]);
                const uint4 second =
                    dequant8_f32(odd ? raw[k].x : raw[k].z,
                                 odd ? raw[k].y : raw[k].w, sc[k]);
                *reinterpret_cast<uint4*>(block + sw128_offset(row, q0 + odd)) =
                    first;
                *reinterpret_cast<uint4*>(
                    block + sw128_offset(row, q0 + !odd)) = second;
            }
            cutlass::arch::fence_view_async_shared();
            __syncwarp();
            if (t % kWarp == 0) Bar::arrive(&sm.kv_full[s], 0, 1);
        }
    } else {
        using Plan = DequantPlan<kCta>;
        constexpr int kPitch = raw_pitch(kCta);
        for (int i = 0; i < n; ++i) {
            const int s = i % C::kStages, is = i % kIdxStages;
            Bar::wait(&sm.idx_full[is], (i / kIdxStages) & 1);
            const uint32_t any[2] = {sm.any[is][0], sm.any[is][1]};
            Bar::wait(&sm.raw_full[s], (i / C::kStages) & 1);
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
            // Only CTA 1 holds rope dims; the max() keeps CTA 0's dead code
            // legal.
            constexpr int kRopeChunks = cute::max(Plan::kRopeChunks, 1);
            uint4 rope[cute::max(Plan::kRopeLoads, 1)];
            CUTLASS_PRAGMA_UNROLL
            for (int k = 0; k < Plan::kRopeLoads; ++k) {
                const int e = t + k * kWarpGroup;
                const int row = e / kRopeChunks, c = e % kRopeChunks;
                rope[k] = *reinterpret_cast<const uint4*>(
                    stage + row * kPitch + kNopeBytes1 + c * kChunk);
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
                *reinterpret_cast<uint4*>(
                    block + sw128_offset(row, q0 + !odd)) = second;
            }
            CUTLASS_PRAGMA_UNROLL
            for (int k = 0; k < Plan::kRopeLoads; ++k) {
                const int e = t + k * kWarpGroup;
                const int row = e / kRopeChunks, c = e % kRopeChunks;
                *reinterpret_cast<uint4*>(stage +
                                          (kLocalBlocks - 1) * kBlockBytes +
                                          sw128_offset(row, c)) = rope[k];
            }
            cutlass::arch::fence_view_async_shared();
            __syncwarp();
            if (t % kWarp == 0) Bar::arrive(&sm.kv_full[s], 0, 1);
        }
    }
}

// The QK^T and PV warps of CTA 0 issue independently, so a late KV stage
// never holds back the PV of a tile whose P is ready, nor a late P the next
// QK^T.  QK^T(i) goes to S buffer i % 2 once its stage is dequantized and the
// softmax has read S(i - 2); a unit's first QK^T also waits for its Q.
template <class C>
CUTE_DEVICE void qk_warp(const HcaParams& p, typename C::Smem& sm,
                         uint32_t tmem, int cluster) {
    const uint32_t idesc = instr_desc<2 * C::kTile, cute::UMMA::Major::K>();
    UnitCursor cur = start_cursor<C>(p, sm, cluster);
    Unit u;
    for (int i = 0, unit = 0; cur.next(p, sm.unit_lens, u); ++unit) {
        Bar::wait(&sm.q_full, unit & 1);
        for (int k = 0; k < u.n; ++k, ++i) {
            const int s = i % C::kStages, sb = i % 2;
            Bar::wait(&sm.kv_full[s], (i / C::kStages) & 1);
            Bar::wait(&sm.s_empty[sb], ((i / 2) & 1) ^ 1);
            tcgen05_after_thread_sync();
            if (cute::elect_one_sync()) {
                CUTLASS_PRAGMA_UNROLL
                for (int ks = 0; ks < kQkSteps; ++ks) {
                    umma_ts(
                        tmem + kTmemS + sb * C::kTmemSCols,
                        tmem + kTmemQ + ks * kUmmaK / 2,
                        sw128_desc(sm.kv[s] +
                                       ks / kKStepsPerBlock * C::kBlockBytes +
                                       ks % kKStepsPerBlock * kKStepBytes,
                                   kChunk),
                        idesc, ks > 0);
                }
                // DSA rope ride: the 64 rope dims didn't fit the TMEM Q tile,
                // so they contract from SMEM (q_rope x rope_kv) into the SAME
                // S, accumulating onto the 512-latent nope partial.  A 2-CTA
                // ss sources operand A (q_rope) from CTA0 only, so the full
                // 64-dim rope rides dims-half 0 (CTA0's rope_kv); CTA1's
                // rope_kv is zero, so half1 adds nothing and the softmax pair
                // sum lands the 64-dim rope once.
                if constexpr (C::kDsa) {
                    CUTLASS_PRAGMA_UNROLL
                    for (int ks = 0; ks < kDsaRopeSteps; ++ks) {
                        umma_ss(
                            tmem + kTmemS + sb * C::kTmemSCols,
                            sw128_desc(sm.q_rope + ks * kKStepBytes, kChunk),
                            sw128_desc(sm.rope_kv[s] + ks * kKStepBytes,
                                       kChunk),
                            idesc, true);
                    }
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

// PV(i) once P(i) is written; it releases the KV stage and P buffer.  The
// first PV of a unit overwrites O, each half (128 columns) once the previous
// unit's epilogue has read it.
template <class C>
CUTE_DEVICE void pv_warp(const HcaParams& p, typename C::Smem& sm,
                         uint32_t tmem, int cluster) {
    const uint32_t idesc = instr_desc<kPvN, cute::UMMA::Major::MN>();
    UnitCursor cur = start_cursor<C>(p, sm, cluster);
    Unit u;
    for (int i = 0, unit = 0; cur.next(p, sm.unit_lens, u); ++unit) {
        for (int k = 0; k < u.n; ++k, ++i) {
            const int s = i % C::kStages, pb = i % 2;
            Bar::wait(&sm.p_full[pb], (i / 2) & 1);
            CUTLASS_PRAGMA_UNROLL
            for (int m = 0; m < kPvMmas; ++m) {
                if (k == 0 && unit > 0)
                    Bar::wait(&sm.o_empty[m], (unit - 1) & 1);
                tcgen05_after_thread_sync();
                if (cute::elect_one_sync()) {
                    CUTLASS_PRAGMA_UNROLL
                    for (int ks = 0; ks < C::kPvSteps; ++ks) {
                        umma_ss(tmem + kTmemO + m * kPvN / 2,
                                sw128_desc(sm.p[pb] + ks * kKStepBytes, kChunk),
                                sw128_desc(
                                    sm.kv[s] +
                                        m * kPvBlocksPerMma * C::kBlockBytes +
                                        ks * kUmmaK * kRowPitch,
                                    C::kBlockBytes),
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

// Thread t of the softmax warp group owns TMEM lane t: head t % 64 of this
// CTA's token with the dims half t / 64.  Per tile it adds the partner lane's
// (t ^ 64, warp w ^ 2) partial S of its 32 keys and runs the online softmax
// on them; partners exchange the row max.
CUTE_DEVICE void pair_sync(int t) {
    NamedBar::sync(2 * kWarp, kPairBar + t / kWarp % 2);
}

template <class C>
CUTE_DEVICE void wait_pv(typename C::Smem& sm, int j) {
    Bar::wait(&sm.pv_done[j % 2], (j / 2) & 1);
}

// The online softmax of tile i (k-th of its unit): this lane's partial S of
// the partner's keys goes to the partner first, then its own keys are read;
// the masked sums give the row max (with the partner's), P and the row sum.
// O is rescaled when the row max grows by more than 2^6.
template <class C>
CUTE_DEVICE void softmax_tile(const HcaParams& p, typename C::Smem& sm, int i,
                              int k, uint32_t lane_base, int t, float& m,
                              float& l) {
    const int head = t % kHeads, half = t / kHeads;
    const int is = i % kIdxStages, sb = i % 2;
    // DSA narrows the KV tile (32 vs 64 keys), so the per-thread key span and
    // the S TMEM width shrink with it; shadow the globals with the Cfg values.
    constexpr int kHalf = C::kHalf;
    constexpr int kTmemSCols = C::kTmemSCols;
    constexpr int kQuads = kHalf / 4;
    Bar::wait(&sm.idx_full[is], (i / kIdxStages) & 1);
    // This thread owns S columns [half*kHalf, half*kHalf+kHalf) of its lane,
    // i.e. keys [half*kHalf, +kHalf) of the tile.  Each index warp ballots 32
    // keys into one valid word, so the owning word is key/kWarp and the bit is
    // key%kWarp.  HCA (kHalf==32): word==half, bit==c.  DSA (kHalf==16, one
    // index warp): word==0, bit==half*16+c.
    const int vbit = (half * kHalf) & (kWarp - 1);
    const uint32_t valid = sm.valid[is][half * kHalf / kWarp];
    Bar::wait(&sm.s_full[sb], (i / 2) & 1);
    tcgen05_after_thread_sync();
    // Rows of xchg are swizzled by 16-byte chunk so a warp's stores and loads
    // of one chunk index spread over all banks.
    float4* const peer = reinterpret_cast<float4*>(sm.xchg[t ^ kHeads]);
    float own[kHalf];
    tmem_ld(lane_base + kTmemS + sb * kTmemSCols + (half ^ 1) * kHalf, own);
    CUTLASS_PRAGMA_UNROLL
    for (int q = 0; q < kQuads; ++q) {
        peer[q ^ (t % kQuads)] = make_float4(own[4 * q], own[4 * q + 1],
                                             own[4 * q + 2], own[4 * q + 3]);
    }
    tmem_ld(lane_base + kTmemS + sb * kTmemSCols + half * kHalf, own);
    tcgen05_before_thread_sync();
    __syncwarp();
    if (t % kWarp == 0) {
        Bar::arrive(&sm.s_empty[sb], 0, 1);
        Bar::arrive(&sm.idx_empty[is]);
    }
    pair_sync(t);
    const float4* const got = reinterpret_cast<const float4*>(sm.xchg[t]);
    float local = -CUDART_INF_F;
    CUTLASS_PRAGMA_UNROLL
    for (int q = 0; q < kQuads; ++q) {
        const float4 v = got[q ^ ((t ^ kHeads) % kQuads)];
        const float add[4] = {v.x, v.y, v.z, v.w};
        CUTLASS_PRAGMA_UNROLL
        for (int e = 0; e < 4; ++e) {
            const int c = 4 * q + e;
            own[c] =
                (valid >> (vbit + c)) & 1 ? own[c] + add[e] : -CUDART_INF_F;
            local = fmaxf(local, own[c]);
        }
    }
    // Scores are in log2 units after the scale; max commutes with it.
    local *= p.scale_log2;
    sm.red[t] = local;
    pair_sync(t);
    const float cur = fmaxf(local, sm.red[t ^ kHeads]);
    float factor = 1.0f;
    if (cur - m > kRescaleLog2) {
        const float m_new = fmaxf(m, cur);
        factor = exp2f(m - m_new);
        m = m_new;
    }
    float sum = 0.0f;
    uint32_t pk[kHalf / 2];
    CUTLASS_PRAGMA_UNROLL
    for (int c = 0; c < kHalf; c += 2) {
        const float p0 = exp2f(fmaf(own[c], p.scale_log2, -m));
        const float p1 = exp2f(fmaf(own[c + 1], p.scale_log2, -m));
        sum += p0 + p1;
        const nv_bfloat162 b2 = __floats2bfloat162_rn(p0, p1);
        pk[c / 2] = reinterpret_cast<const uint32_t&>(b2);
    }
    l = l * factor + sum;
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
    if (i > 1) wait_pv<C>(sm, i - 2);
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

// Per unit: the online softmax over its tiles, then the row max and partial
// sum go to the Q/O warp group.
template <class C>
CUTE_DEVICE void softmax_wg(const HcaParams& p, typename C::Smem& sm,
                            uint32_t tmem, int t, int cluster) {
    const uint32_t lane_base = tmem + ((t / kWarp * kWarp) << 16);
    UnitCursor cur = start_cursor<C>(p, sm, cluster);
    Unit u;
    for (int i = 0, unit = 0; cur.next(p, sm.unit_lens, u); ++unit) {
        float m = kMaxInit, l = 0.0f;
        for (int k = 0; k < u.n; ++k, ++i) {
            softmax_tile<C>(p, sm, i, k, lane_base, t, m, l);
        }
        const int b = unit % 2;
        Bar::wait(&sm.ml_empty[b], ((unit / 2) & 1) ^ 1);
        sm.ml[b][t] = make_float2(m, l);
        __syncwarp();
        if (t % kWarp == 0) Bar::arrive(&sm.ml_full[b]);
    }
}

// Thread t of the Q/O warp group also owns TMEM lane t.  It writes a unit's Q
// slice (head t % 64, dims half t / 64) to TMEM once the previous unit's
// last QK^T is done, then reads out the previous unit's O; the next unit's Q
// is loaded into registers right after.
constexpr int kQVecs = kHalfDims * sizeof(__nv_bfloat16) / sizeof(uint4);

CUTE_DEVICE void tma_load_2d(const CUtensorMap* map, uint64_t* bar, void* dst,
                             int col, int row) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::"
        "complete_tx::bytes [%0], [%1, {%2, %3}], [%4];" ::"r"(smem_u32(dst)),
        "l"(map), "r"(col), "r"(row), "r"(smem_u32(bar))
        : "memory");
}

CUTE_DEVICE void tma_store_2d(const CUtensorMap* map, const void* src, int col,
                              int row) {
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2, %3}], "
        "[%1];" ::"l"(map),
        "r"(smem_u32(src)), "r"(col), "r"(row)
        : "memory");
}

CUTE_DEVICE void bulk_commit() {
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}

// This thread's bulk stores have read their shared memory source.
CUTE_DEVICE void bulk_wait_read() {
    asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
}

// A lane's row of a 128-byte swizzled box: its address with the row's
// swizzle phase xor'ed in, so chunk Q is at row_x ^ 16Q.  The xor is done at
// each access: precomputed chunk addresses are 8 registers per box row.
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

// The Q/O warp group's named barrier, taking a value derived from data just
// loaded from shared memory: the barrier then waits for those loads, so a
// TMA write into their source issued after it cannot overtake loads still in
// flight (a plain bar.sync does not order them before the async proxy).
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

// A later unit's Q comes by TMA, one box per dims half at a time: a lane's
// head row is strided, and direct loads of it occupy the L1 a row per lane,
// stalling the softmax's shared memory accesses meanwhile.
template <class C>
CUTE_DEVICE void fetch_q(const HcaMaps& maps, typename C::Smem& sm, int token,
                         int t, uint4 (&v)[kQVecs]) {
    CUTLASS_PRAGMA_UNROLL
    for (int r = 0; r < kQoRounds; ++r) {
        if (t == 0) {
            cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(
                &sm.qs_full, 2 * kQBoxBytes);
            CUTLASS_PRAGMA_UNROLL
            for (int h = 0; h < 2; ++h) {
                tma_load_2d(&maps.q, &sm.qs_full, sm.qo_stage[h],
                            (h * kQoRounds + r) * kQBoxDims, token * kHeads);
            }
        }
        Bar::wait(&sm.qs_full, r & 1);
        const uint32_t row = swizzled_row(sm.qo_stage[t / kHeads], t % kHeads);
        cute::for_each(cute::make_int_sequence<kChunksPerBox>{}, [&](auto q) {
            v[r * kChunksPerBox + q] = lds_chunk<decltype(q)::value>(row);
        });
        // Every lane has read the boxes before the next ones land.
        qo_sync_after(xor_words(v + r * kChunksPerBox, kChunksPerBox));
    }
}

template <class C>
CUTE_DEVICE void store_q(const HcaParams& p, typename C::Smem& sm,
                         uint32_t lane_base, int token, int cta, int t,
                         const uint4 (&v)[kQVecs]) {
    // One store per 16-byte load: a load's registers are a store's source as
    // they are, with no 32-register block to assemble.
    CUTLASS_PRAGMA_UNROLL
    for (int q = 0; q < kQVecs; ++q) {
        tmem_st_32dp32bNx<4>(lane_base + kTmemQ + 4 * q, &v[q]);
    }
    if constexpr (C::kDsa) {
        // The 64 rope dims of the token go to the SMEM rope-Q tile.  Measured
        // ground truth: a 2-CTA umma_ss sources A from CTA0's SMEM rows [0,64)
        // for BOTH dims-halves (CTA1's rows and the upper M rows [64,128) are
        // never read).  So the whole 64-dim rope dot rides dims-half 0 against
        // CTA0's rope_kv; CTA1's rope_kv is zero, so its dims-half adds nothing
        // and softmax's pair sum (half0 + half1) yields one full rope logit.
        // Each head row carries all 64 rope dims (sw128, 128 B/row).
        const int head = t % kHeads, g = t / kHeads;
        if (g == 0) {
            const __nv_bfloat16* src =
                p.q + (int64_t(token) * kHeads + head) * kDsaQk + kDsaNope;
            CUTLASS_PRAGMA_UNROLL
            for (int c = 0; c < kDsaRope / kElemsPerChunk; ++c) {
                *reinterpret_cast<uint4*>(sm.q_rope + sw128_offset(head, c)) =
                    *reinterpret_cast<const uint4*>(src + c * kElemsPerChunk);
            }
        }
        cutlass::arch::fence_view_async_shared();
    }
    cutlass::arch::fence_view_async_tmem_store();
    tcgen05_before_thread_sync();
    __syncwarp();
    if (t % kWarp == 0) Bar::arrive(&sm.q_full, 0, 1);
}

// The Q/O warp group's waits are long (a unit or more): one warp polls the
// barrier while the others sleep on a named barrier, so the polling does not
// take issue slots from the other roles.
CUTE_DEVICE void qo_wait(uint64_t* bar, int phase, int t) {
    if (t < kWarp) Bar::wait(bar, phase);
    NamedBar::sync(kWarpGroup, kQoBar);
}

// A continuation part's normalized O in o_part: plane p holds columns
// [4p, 4p + 4) of every TMEM lane's row, lanes contiguous, so a warp's
// 16-byte stores of a plane are 512 contiguous bytes and the merger's loads
// of the staged planes are free of bank conflicts.
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

// Reads O columns [m * 128, m * 128 + 128) of this lane, then frees them
// for the next unit's first PV.
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
CUTE_DEVICE void store_out(__nv_bfloat16* dst, const float* v, float scale) {
    CUTLASS_PRAGMA_UNROLL
    for (int q = 0; q < N; q += 16) {
        uint32_t w[8];
        CUTLASS_PRAGMA_UNROLL
        for (int e = 0; e < 8; ++e) {
            const nv_bfloat162 b2 = __floats2bfloat162_rn(
                v[q + 2 * e] * scale, v[q + 2 * e + 1] * scale);
            w[e] = reinterpret_cast<const uint32_t&>(b2);
        }
        st_global_v8_na(dst + q, w);
    }
}

CUTE_DEVICE void red_release_add(int* ptr, int v) {
    asm volatile("red.release.gpu.global.add.s32 [%0], %1;" ::"l"(ptr), "r"(v)
                 : "memory");
}

CUTE_DEVICE int ld_acquire(const int* ptr) {
    int v;
    asm volatile("ld.acquire.gpu.global.b32 %0, [%1];"
                 : "=r"(v)
                 : "l"(ptr)
                 : "memory");
    return v;
}

// O leaves through qo_stage by TMA in rounds: round r writes dims
// [64r, 64r + 64) of every lane's row (head t % 64, dims half t / 64), a box
// per dims half.  Thread 0 issues the stores and waits for the previous
// round's to have read the stage.
constexpr int kRoundWords = kQBoxDims / 2;

template <class C>
CUTE_DEVICE void store_round(const HcaMaps& maps, typename C::Smem& sm,
                             int token, int t, int r, const uint32_t* w) {
    if (r > 0) {
        if (t == 0) bulk_wait_read();
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
            tma_store_2d(&maps.out, sm.qo_stage[h],
                         (h * kQoRounds + r) * kQBoxDims, token * kHeads);
        }
        bulk_commit();
    }
}

// Reads half mh (128 columns) of this lane's O, scaled, as bf16 pairs, and
// releases it to the next unit's first PV.
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

// The bf16 output of an unsplit unit, a half at a time.  `write` is false for
// a lone token's CTA 1: it still drains O (releasing it to the next unit's PV)
// but stores nothing, as the row it would write belongs to the next request.
// The cluster's last unit stages all of O at once in the idle KV stages, so
// no round waits for the previous one's box to be read.
template <class C>
CUTE_DEVICE void store_out_rows(const HcaMaps& maps, typename C::Smem& sm,
                                int token, uint32_t lane_base, int t,
                                float scale, bool write, bool last) {
    static_assert(2 * kQoRounds * kQBoxBytes <= sizeof(sm.kv));
    if (!C::kDsa && last && write) {
        const auto box = [&](int h, int r) {
            return sm.kv[0] + (h * kQoRounds + r) * kQBoxBytes;
        };
        CUTLASS_PRAGMA_UNROLL
        for (int mh = 0; mh < 2; ++mh) {
            uint32_t w[kOHalf / 2];
            read_o_bf16<C>(sm, lane_base, t, mh, scale, w);
            CUTLASS_PRAGMA_UNROLL
            for (int k = 0; k < 2; ++k) {
                const uint32_t row =
                    swizzled_row(box(t / kHeads, 2 * mh + k), t % kHeads);
                const uint32_t* wr = w + k * kRoundWords;
                cute::for_each(cute::make_int_sequence<kChunksPerBox>{},
                               [&](auto q) {
                                   constexpr int Q = decltype(q)::value;
                                   sts_chunk<Q>(row, wr[4 * Q], wr[4 * Q + 1],
                                                wr[4 * Q + 2], wr[4 * Q + 3]);
                               });
            }
            cutlass::arch::fence_view_async_shared();
            NamedBar::sync(kWarpGroup, kQoBar);
            if (t == 0) {
                CUTLASS_PRAGMA_UNROLL
                for (int k = 0; k < 2; ++k) {
                    CUTLASS_PRAGMA_UNROLL
                    for (int h = 0; h < 2; ++h) {
                        const int r = 2 * mh + k;
                        tma_store_2d(&maps.out, box(h, r),
                                     (h * kQoRounds + r) * kQBoxDims,
                                     token * kHeads);
                    }
                }
                bulk_commit();
            }
        }
        // The stages are free again for the fused merge's staging.
        if (t == 0) bulk_wait_read();
        return;
    }
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
    if (write && t == 0) bulk_wait_read();
}

// Calls f(c) for the parts of a split pair in cluster order: clusters
// first.. up to the one holding the pair's last tile, skipping clusters with
// an empty range.
template <class F>
CUTE_DEVICE void for_parts(int clusters, const Unit& u, int first, int total,
                           F&& f) {
    const int end = u.w0 + u.wn;
    for (int c = first; c < clusters && range_lo(c, total, clusters) < end;
         ++c) {
        if (range_lo(c + 1, total, clusters) > range_lo(c, total, clusters)) {
            f(c);
        }
    }
}

// A part's planes are contiguous: plain bulk copies move them.
CUTE_DEVICE void bulk_prefetch(const void* src, uint32_t bytes) {
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(src),
                 "r"(bytes)
                 : "memory");
}

CUTE_DEVICE void bulk_load(void* dst, const void* src, uint32_t bytes,
                           uint64_t* bar) {
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
        "[%0], [%1], %2, [%3];" ::"r"(smem_u32(dst)),
        "l"(src), "r"(bytes), "r"(smem_u32(bar))
        : "memory");
}

CUTE_DEVICE const float* part_box(const HcaParams& p, int slot, int cta,
                                  int bx) {
    return p.o_part +
           (int64_t(slot * 2 + cta) * kPlanes + bx * kPartBoxPlanes) *
               (kPlaneBytes / sizeof(float));
}

// O of a finished unit: bf16 output with the sink in the denominator.  A
// split pair is merged by the cluster holding its first tile: that part is
// the cluster's last unit, so its KV stages are idle to stage the other
// parts, which are the first (or only) units of the following clusters and
// publish their normalized O and log2 sum by then.  The merge adds the parts
// in cluster order, in fp32 in TMEM, so it does not depend on timing.  Each
// half of O is released to the next unit's first PV once it is read.
template <class C>
CUTE_DEVICE void store_o(const HcaParams& p, const HcaMaps& maps,
                         typename C::Smem& sm, const Unit& u, int unit, int cta,
                         int cluster, uint32_t lane_base, int t,
                         float sink_log2, bool final_unit) {
    const int head = t % kHeads, half = t / kHeads;
    const int b = unit % 2;
    const int total = sm.total;
    const int sched = sm.sched;
    const int first = cluster_of(sched, total, u.w0);
    // Every cluster's range is non-empty when total >= sched: the parts of
    // a split pair are then clusters first .. last; else (tiny batches) empty
    // ones are skipped.
    const int last = cluster_of(sched, total, u.w0 + u.wn - 1);
    const bool dense = total >= sched;
    const auto part_cluster = [&](int j) {
        if (dense) return first + j;
        int c_j = first, n = 0;
        for_parts(sched, u, first, total, [&](int c) {
            if (n++ == j) c_j = c;
        });
        return c_j;
    };
    int n_parts = last - first + 1;
    if (u.split && !dense) {
        n_parts = 0;
        for_parts(sched, u, first, total, [&](int) { ++n_parts; });
    }
    const int others = n_parts - 1;
    // The merger warms L2 with the other parts while its own part runs: they
    // were written by the first units of the following clusters, possibly
    // long before.
    if (u.split && first == cluster && t < kPartBoxes && !sm.comb) {
        for (int k = 1; k <= others; ++k) {
            bulk_prefetch(part_box(p, part_cluster(k), cta, t),
                          kPartBoxPlanes * kPlaneBytes);
        }
    }
    qo_wait(&sm.ml_full[b], (unit / 2) & 1, t);
    const float2 own = sm.ml[b][t], other = sm.ml[b][t ^ kHeads];
    __syncwarp();
    if (t % kWarp == 0) Bar::arrive(&sm.ml_empty[b]);
    const float m = own.x, sum = own.y + other.y;
    // CTA 1 of a lone (odd-request) token has no token of its own: it still
    // runs the pipeline so the shared KV tiles flow, but writes no output and
    // takes no part in the split merge.
    const bool lone = cta == 1 && !u.g.has_b;
    if (!u.split) {
        // Mark the pair's job slot empty so combine skips it; with a slot per
        // pair, rewritten every launch, no job counter (and memset) is needed.
        if (sm.comb && !p.fuse && cta == 0 && t == 0) p.jobs[u.pair].wn = 0;
        const float scale =
            sum > 0.0f ? 1.0f / (sum + exp2f(sink_log2 - m)) : 0.0f;
        qo_wait(&sm.o_full, unit & 1, t);
        tcgen05_after_thread_sync();
        store_out_rows<C>(maps, sm, u.g.a + cta, lane_base, t, scale, !lone,
                          final_unit);
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
        // A follower part (the first unit of a later cluster) still drains O
        // for the next unit's PV; the merger (first == cluster) is this
        // cluster's last unit and has no next PV, so it just returns.  Either
        // way CTA 1 writes no partial, no log2 sum and never touches the
        // counter, so its merge lane stays empty.
        if (first != cluster) {
            float o[kOHalf];
            read_o_half<C>(sm, lane_base, t, 0, o);
            read_o_half<C>(sm, lane_base, t, 1, o);
            NamedBar::sync(kWarpGroup, kQoBar);
        }
        return;
    }
    // MEASUREMENT: nomerge makes the merger write its part like a follower and
    // skip the spin-wait + serial accumulate, so the pole spreads freely and we
    // time the merge-free compute floor (output is wrong on split pairs).
    if (first != cluster || p.nomerge || sm.comb) {
        // In combine mode a cluster owns two partial slots (which = first ==
        // cluster): the trailing owner part goes to slot 1, every leading /
        // middle continuation to slot 0, so the two never clash.
        const int sid = sm.comb ? cluster * 2 + (first == cluster) : cluster;
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
        // The merger records its pair's job (CTA 0 always owns the first
        // token); the combine kernel reads every part's partial after this
        // kernel completes, so no counter is needed.
        if (C::kFuse && sm.comb && p.fuse) {
            // Fused merge: publish the part on the pair's counter and note
            // the pair for this cluster's merge pass (fused_merge).
            if (t == 0) {
                // A first part in a cluster that already holds a part of an
                // earlier pair flags it: that cluster merges only the earlier
                // pair, so no cluster merges twice in a row.
                const bool shared = first == cluster && sm.n_mj > 0;
                red_release_add(ctr, 1 + (shared ? kMergeShared : 0));
                sm.mj[sm.n_mj++] =
                    make_int4(first, last, u.g.a, u.pair * 2 + u.g.has_b);
            }
        } else if (sm.comb) {
            if (first == cluster && cta == 0 && t == 0) {
                p.jobs[u.pair] = MergeJob{u.g.a, u.g.has_b, u.w0, u.wn};
            }
        } else if (t == 0) {
            red_release_add(ctr, 1);
        }
        return;
    }
    if (t == 0) {
        while (ld_acquire(ctr) != others) {
        }
        *ctr = 0;
        // The parts, written by generic stores, are read by bulk copies.
        asm volatile("fence.proxy.async.global;" ::: "memory");
    }
    NamedBar::sync(kWarpGroup, kQoBar);
    // Serial-merge fallback (p.pipe == 0): one part at a time into sm.kv[0].
    const auto stage_part = [&](int k) {
        if (t == 0) {
            cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(
                &sm.merge_full[0], kPlanes * kPlaneBytes);
        }
        NamedBar::sync(kWarpGroup, kQoBar);
        static_assert(kPartBoxes == kWarpsPerGroup);
        if (t % kWarp == 0) {
            const int bx = t / kWarp;
            bulk_load(sm.kv[0] + bx * kPartBoxPlanes * kPlaneBytes,
                      part_box(p, part_cluster(k + 1), cta, bx),
                      kPartBoxPlanes * kPlaneBytes, &sm.merge_full[0]);
        }
    };
    // The log2 sums of up to kLseRegs parts are loaded at once; parts beyond
    // (tiny batches only) are reloaded where needed.
    constexpr int kLseRegs = 8;
    float lse_r[kLseRegs];
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < kLseRegs; ++j) {
        lse_r[j] =
            j >= n_parts ? -CUDART_INF_F
            : j == 0
                ? own_lse
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
    const float w_own =
        own_lse > -CUDART_INF_F ? exp2f(own_lse - mx) * inv : 0.0f;
    // Each other part in turn is staged in the KV stages; the running sum
    // stays in TMEM, the last part's pass writes the output.
    static_assert(kPlanes * kPlaneBytes <= C::kStages * C::kStageBytes);
    const uint32_t cols = lane_base + kTmemO;
    if (p.pipe) {
        // Prescale own O (already in TMEM) by its normalization and merge
        // weight; the box ring then just adds the weighted other parts.
        CUTLASS_PRAGMA_UNROLL
        for (int db = 0; db < kHalfDims / kMergeCols; ++db) {
            float x[kMergeCols];
            tmem_ld_32dp32bNx<kMergeCols>(cols + db * kMergeCols, x);
            cutlass::arch::fence_view_async_tmem_load();
            CUTLASS_PRAGMA_UNROLL
            for (int q = 0; q < kMergeCols; ++q) x[q] *= own_scale * w_own;
            tmem_st_32dp32bNx<kMergeCols>(cols + db * kMergeCols, x);
        }
        cutlass::arch::fence_view_async_tmem_store();
        NamedBar::sync(kWarpGroup, kQoBar);
        // One box = kPartBoxPlanes planes = kSubPerBox dim blocks; load it into
        // ring buffer buf.  A box is kMergeBoxBytes; the KV stages are reused
        // as the scratch, so address the buffers as flat box-sized slabs of the
        // kv array (with a narrow tile a stage can be smaller than a box, so
        // one box may span >1 stage).
        constexpr int kSubPerBox = kPartBoxPlanes / (kMergeCols / kPlaneCols);
        constexpr int kMergeBoxBytes = kPartBoxPlanes * kPlaneBytes;
        static_assert(kMergeBuf * kMergeBoxBytes <=
                      C::kStages * C::kStageBytes);
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
                    store_out<kMergeCols>(out + db * kMergeCols, x, 1.0f);
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
    } else {
        stage_part(0);
        for (int k = 0; k < others; ++k) {
            const float lse = lse_at(k + 1);
            const float w = lse > -CUDART_INF_F ? exp2f(lse - mx) * inv : 0.0f;
            if (k > 0) stage_part(k);
            Bar::wait(&sm.merge_full[0], k & 1);
            // Two steps per TMEM round trip.
            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < kHalfDims / kMergeCols; j += 2) {
                float x[2][kMergeCols];
                tmem_ld_32dp32bNx<kMergeCols>(cols + j * kMergeCols, x[0]);
                tmem_ld_32dp32bNx<kMergeCols>(cols + (j + 1) * kMergeCols,
                                              x[1]);
                cutlass::arch::fence_view_async_tmem_load();
                CUTLASS_PRAGMA_UNROLL
                for (int jj = 0; jj < 2; ++jj) {
                    if (k == 0) {
                        CUTLASS_PRAGMA_UNROLL
                        for (int q = 0; q < kMergeCols; ++q)
                            x[jj][q] = x[jj][q] * own_scale * w_own;
                    }
                    const uint8_t* const planes =
                        sm.kv[0] +
                        (j + jj) * kMergeCols / kPlaneCols * kPlaneBytes +
                        t * kPlaneCols * sizeof(float);
                    CUTLASS_PRAGMA_UNROLL
                    for (int q = 0; q < kMergeCols / kPlaneCols; ++q) {
                        const float4 v = *reinterpret_cast<const float4*>(
                            planes + q * kPlaneBytes);
                        x[jj][4 * q] += v.x * w;
                        x[jj][4 * q + 1] += v.y * w;
                        x[jj][4 * q + 2] += v.z * w;
                        x[jj][4 * q + 3] += v.w * w;
                    }
                    if (k == others - 1) {
                        store_out<kMergeCols>(out + (j + jj) * kMergeCols,
                                              x[jj], 1.0f);
                    } else {
                        tmem_st_32dp32bNx<kMergeCols>(
                            cols + (j + jj) * kMergeCols, x[jj]);
                    }
                }
            }
            cutlass::arch::fence_view_async_tmem_store();
            // The staged part is read by every thread before the next one lands
            // in the same place.
            NamedBar::sync(kWarpGroup, kQoBar);
        }
    }
}

template <class C>
CUTE_DEVICE void qo_wg(const HcaParams& p, const HcaMaps& maps,
                       typename C::Smem& sm, uint32_t tmem, float sink_log2,
                       int cta, int t, int cluster) {
    const uint32_t lane_base = tmem + ((t / kWarp * kWarp) << 16);
    UnitCursor cur = start_cursor<C>(p, sm, cluster);
    Unit u, prev;
    bool more = cur.next(p, sm.unit_lens, u);
    int unit = 0;
    for (; more; ++unit) {
        // Unit u's Q is loaded while the previous unit runs, and goes to TMEM
        // once that unit's last QK^T is done.  Every unit's Q streams through
        // qo_stage, so the KV stages pipeline at full depth from the start.
        uint4 qv[kQVecs];
        // A lone token's CTA 1 has no token of its own; it loads the pair's
        // valid token (an L2 hit, never written back) instead of reading past
        // the request, keeping the Q pipeline barriers balanced.
        const int q_token = u.g.a + (cta == 1 && !u.g.has_b ? 0 : cta);
        fetch_q<C>(maps, sm, q_token, t, qv);
        if (unit > 0) {
            qo_wait(&sm.q_empty, (unit - 1) & 1, t);
            tcgen05_after_thread_sync();
        }
        store_q<C>(p, sm, lane_base, q_token, cta, t, qv);
        if (unit > 0) {
            store_o<C>(p, maps, sm, prev, unit - 1, cta, cluster, lane_base, t,
                       sink_log2, false);
        }
        prev = u;
        more = cur.next(p, sm.unit_lens, u);
    }
    if (unit > 0) {
        store_o<C>(p, maps, sm, prev, unit - 1, cta, cluster, lane_base, t,
                   sink_log2, true);
    }
    // The output is written before the CTA exits.
    if (t == 0) bulk_store_wait_all();
}

template <class C>
CUTE_DEVICE void init_barriers(typename C::Smem& sm) {
    constexpr int kPairWarps = 2 * kWarpsPerGroup;  // a warp group of both CTAs
    Bar::init(&sm.q_full, kPairWarps);
    Bar::init(&sm.q_empty, 1);
    Bar::init(&sm.o_full, 1);
    for (int i = 0; i < kMergeBuf; ++i) Bar::init(&sm.merge_full[i], 1);
    Bar::init(&sm.qs_full, 1);
    for (int b = 0; b < 2; ++b) {
        Bar::init(&sm.o_empty[b], kPairWarps);
        Bar::init(&sm.ml_full[b], kWarpsPerGroup);
        Bar::init(&sm.ml_empty[b], kWarpsPerGroup);
        Bar::init(&sm.s_full[b], 1);
        Bar::init(&sm.s_empty[b], kPairWarps);
        Bar::init(&sm.p_full[b], kPairWarps);
        Bar::init(&sm.pv_done[b], 1);
    }
    for (int s = 0; s < kIdxStages; ++s) {
        Bar::init(&sm.idx_full[s], C::kIndexWarps);
        Bar::init(&sm.idx_empty[s], 2 * kWarpsPerGroup);  // softmax + dequant
    }
    for (int s = 0; s < C::kStages; ++s) {
        // per index warp: the expect-tx arrival and one cp.async per lane
        Bar::init(&sm.raw_full[s], C::kIndexWarps * (1 + kWarp));
        Bar::init(&sm.kv_full[s], kPairWarps);
        Bar::init(&sm.kv_empty[s], 1);
    }
    cutlass::arch::fence_barrier_init();
}

// Fused merge (p.fuse with comb): after its last unit a cluster merges every
// split pair it holds a part of, each part cluster r of n taking planes
// [64 r / n, 64 (r + 1) / n) of the pair's two tokens (one per CTA), once all
// n parts are published on the pair's counter.  Parts are produced without
// waiting on anyone and all clusters are resident, so the waits cannot
// deadlock.  Comb mode is always dense (uncapped, >= 1 tile per cluster):
// part j is cluster first + j.  The counter is reset by the last of the n
// mergers.  Each CTA reads 128 KB of partials whatever n is, so the loop is
// latency bound: the per-head weights 2^(lse_k - M) / den (sink folded into
// den) are staged once, then every thread keeps 16 independent loads in
// flight over a (planes x parts) block shaped by n.
constexpr int kMergeWgs = 2;
constexpr uint32_t kMergeBar = 5;
constexpr int kMergeThreads = kMergeWgs * kWarpGroup;
CUTE_DEVICE void merge_sync() { NamedBar::sync(kMergeThreads, kMergeBar); }

// acc[u] += sum over parts k in [k0, k1) of w[k][head] * part k's plane
// pl0 + u * pstep (planes >= phi are skipped).
template <int UP, int UK>
CUTE_DEVICE void merge_block(const HcaParams& p, const float* w, int first,
                             int cta, int tsi, int head, int pl0, int pstep,
                             int phi, int k0, int k1, float4* acc) {
    for (int kb = k0; kb < k1; kb += UK) {
        float wk[UK];
        float4 v[UK][UP];
        CUTLASS_PRAGMA_UNROLL
        for (int b = 0; b < UK; ++b) {
            const int k = kb + b;
            wk[b] = k < k1 ? w[k * kHeads + head] : 0.0f;
            const int sid = (first + k) * 2 + (k == 0);
            const float4* const src = reinterpret_cast<const float4*>(
                p.o_part +
                (int64_t(sid * 2 + cta) * kPlanes * kWarpGroup + tsi) *
                    kPlaneCols);
            CUTLASS_PRAGMA_UNROLL
            for (int u = 0; u < UP; ++u) {
                const int pl = pl0 + u * pstep;
                v[b][u] = k < k1 && pl < phi
                              ? __ldcg(src + pl * kWarpGroup)
                              : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
        CUTLASS_PRAGMA_UNROLL
        for (int b = 0; b < UK; ++b) {
            // A zero weight (empty part) must not pick up its unscaled values.
            if (wk[b] == 0.0f) continue;
            CUTLASS_PRAGMA_UNROLL
            for (int u = 0; u < UP; ++u) {
                acc[u].x += wk[b] * v[b][u].x;
                acc[u].y += wk[b] * v[b][u].y;
                acc[u].z += wk[b] * v[b][u].z;
                acc[u].w += wk[b] * v[b][u].w;
            }
        }
    }
}

template <int UP, int UK>
CUTE_DEVICE void merge_slice(const HcaParams& p, const float* w, float4* red,
                             uint2* stage, int first, int n, int cta, int plo,
                             int phi, int token) {
    const int wg = threadIdx.x / kWarpGroup, tsi = threadIdx.x % kWarpGroup;
    const int head = tsi % kHeads, half = tsi / kHeads;
    // One plane: the two warp groups split the parts and sum through smem;
    // otherwise they take alternate planes.
    const bool ksplit = phi - plo < kMergeWgs;
    const int pstep = ksplit ? 1 : kMergeWgs;
    const int k0 = ksplit ? n * wg / kMergeWgs : 0;
    const int k1 = ksplit ? n * (wg + 1) / kMergeWgs : n;
    // The bf16 results go through smem, one row of ns planes per (half,
    // head), so the global stores are contiguous runs instead of 4 bytes per
    // 1 KB-strided row.  An odd pitch (in planes) keeps the writes conflict
    // free.
    const int ns = phi - plo, pitch = ns | 1;
    for (int pb = plo + (ksplit ? 0 : wg); pb < phi; pb += UP * pstep) {
        float4 acc[UP];
        CUTLASS_PRAGMA_UNROLL
        for (int u = 0; u < UP; ++u)
            acc[u] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        merge_block<UP, UK>(p, w, first, cta, tsi, head, pb, pstep, phi, k0, k1,
                            acc);
        if (ksplit) {
            if (wg == 1) red[tsi] = acc[0];
            merge_sync();
            if (wg == 1) continue;
            const float4 o = red[tsi];
            acc[0].x += o.x, acc[0].y += o.y, acc[0].z += o.z, acc[0].w += o.w;
        }
        CUTLASS_PRAGMA_UNROLL
        for (int u = 0; u < UP; ++u) {
            const int pl = pb + u * pstep;
            if (pl >= phi) break;
            const nv_bfloat162 lo = __floats2bfloat162_rn(acc[u].x, acc[u].y);
            const nv_bfloat162 hi = __floats2bfloat162_rn(acc[u].z, acc[u].w);
            stage[tsi * pitch + pl - plo] =
                make_uint2(*reinterpret_cast<const uint32_t*>(&lo),
                           *reinterpret_cast<const uint32_t*>(&hi));
        }
    }
    merge_sync();
    // Plane pl of (half, head) holds dims half * 256 + 4 pl .. + 3.
    static_assert(kHalfDims == kPlanes * kPlaneCols);
    for (int e = threadIdx.x; e < kWarpGroup * ns; e += kMergeThreads) {
        const int row = e / ns, c = e - row * ns;
        const int dim = (row / kHeads) * kHalfDims + (plo + c) * kPlaneCols;
        *reinterpret_cast<uint2*>(
            p.out + (int64_t(token + cta) * kHeads + row % kHeads) * kDim +
            dim) = stage[row * pitch + c];
    }
}

template <class C>
CUTE_DEVICE void fused_merge(const HcaParams& p, typename C::Smem& sm, int cta,
                             int cluster) {
    // The KV stages are idle once the last unit's PV is done (every merging
    // thread has seen o_full): weights in stage 0, the part-split sum in 1.
    constexpr int kMaxParts = sizeof(sm.kv[0]) / (kHeads * sizeof(float));
    static_assert(kMaxParts >= 128 &&
                  sizeof(sm.kv[1]) >= kWarpGroup * sizeof(float4));
    float* const w = reinterpret_cast<float*>(sm.kv[0]);
    float4* const red = reinterpret_cast<float4*>(sm.kv[1]);
    uint2* const stage = reinterpret_cast<uint2*>(sm.kv[2]);
    static_assert(sizeof(sm.kv) - 2 * sizeof(sm.kv[0]) >=
                  kWarpGroup * (kPlanes + 1) * sizeof(uint2));
    merge_sync();  // sm.mj from the Q/O group
    // Helpers: the clusters past sched, helper h on pair h % P (find_range),
    // at most enough that every merger keeps a plane.
    const int pairs =
        (p.tokens / p.tokens_per_req) * ((p.tokens_per_req + 1) / 2);
    const int n_help = p.clusters - sm.sched;
    for (int i = 0; i < sm.n_mj; ++i) {
        const int4 j = sm.mj[i];
        const int first = j.x, n = j.y - j.x + 1, q = j.w >> 1;
        int* const ctr = p.counter + first * 2 + cta;
        const int r = cluster - first;
        const bool helper = cluster >= sm.sched;
        const int hq = min(q < n_help ? (n_help - 1 - q) / pairs + 1 : 0,
                           max(kPlanes - n, 0));
        const int hi = (cluster - sm.sched) / max(pairs, 1);
        if (helper && hi >= hq) continue;
        // A shared first part (see store_o) sits this pair out: it neither
        // waits nor counts, and the other n - 1 parts split the planes.
        if (r == 0 && i > 0) continue;
        // Every merger with a share waits, even one with no planes (> 64
        // parts): its count must not let another merger past before all
        // parts are published.
        __shared__ int shared_first;
        if (threadIdx.x == 0) {
            int c;
            while (((c = ld_acquire(ctr)) & (kMergeShared - 1)) < n) {
            }
            shared_first = c >= kMergeShared;
        }
        merge_sync();
        const int sh = shared_first, nn = n - sh, nm = nn + hq;
        const int rr = helper ? nn + hi : r - sh;
        const int plo = kPlanes * rr / nm, phi = kPlanes * (rr + 1) / nm;
        if (phi > plo && n <= kMaxParts) {
            // Weights: kS threads per head each take every kS-th part with an
            // online (max, denom), combined through smem (red, free until
            // merge_slice) so the serial depth is n / kS, not 3 n.
            constexpr int kS = kMergeThreads / kHeads;
            float2* const md = reinterpret_cast<float2*>(red);
            const int h = threadIdx.x % kHeads, s = threadIdx.x / kHeads;
            float m = -CUDART_INF_F, den = 0.0f;
            for (int k = s; k < n; k += kS) {
                const float l = __ldcg(
                    p.lse_part + part_row((first + k) * 2 + (k == 0), cta, h));
                w[k * kHeads + h] = l;
                if (l > -CUDART_INF_F) {
                    const float nm = fmaxf(m, l);
                    den = den * exp2f(m - nm) + exp2f(l - nm);
                    m = nm;
                }
            }
            md[s * kHeads + h] = make_float2(m, den);
            merge_sync();
            const float sink =
                p.sink ? __ldg(p.sink + h) * CUDART_L2E_F : -CUDART_INF_F;
            float mx = sink;
            CUTLASS_PRAGMA_UNROLL
            for (int g = 0; g < kS; ++g) mx = fmaxf(mx, md[g * kHeads + h].x);
            float d = sink > -CUDART_INF_F ? exp2f(sink - mx) : 0.0f;
            CUTLASS_PRAGMA_UNROLL
            for (int g = 0; g < kS; ++g) {
                const float2 v = md[g * kHeads + h];
                if (v.y > 0.0f) d += v.y * exp2f(v.x - mx);
            }
            const float inv = d > 0.0f ? 1.0f / d : 0.0f;
            for (int k = s; k < n; k += kS) {
                const float l = w[k * kHeads + h];
                w[k * kHeads + h] =
                    l > -CUDART_INF_F ? exp2f(l - mx) * inv : 0.0f;
            }
            merge_sync();
            if (n <= 2) {
                merge_slice<8, 2>(p, w, red, stage, first, n, cta, plo, phi,
                                  j.z);
            } else if (n <= 4) {
                merge_slice<4, 4>(p, w, red, stage, first, n, cta, plo, phi,
                                  j.z);
            } else if (n <= 8) {
                merge_slice<2, 8>(p, w, red, stage, first, n, cta, plo, phi,
                                  j.z);
            } else {
                merge_slice<1, 16>(p, w, red, stage, first, n, cta, plo, phi,
                                   j.z);
            }
            merge_sync();  // w and red are rewritten by the next pair
        }
        // n parts and nm mergers count; the last one resets (flag included).
        if (threadIdx.x == 0 &&
            (atomicAdd(ctr, 1) & (kMergeShared - 1)) == n + nm - 1)
            *ctr = 0;
    }
}

// The softmax and Q/O warp groups (the ones with registers to spare) merge
// from inside their own branches, so the merge is compiled under their
// register budgets rather than the aux groups' 56; the others go straight
// to the teardown barrier.
template <class C>
CUTE_DEVICE void merge_tail(const HcaParams& p, typename C::Smem& sm, int cta,
                            int cluster) {
    if constexpr (C::kFuse) {
        if (sm.comb && p.fuse) {
            pdl_trigger();
            fused_merge<C>(p, sm, cta, cluster);
        }
    }
}

// Persistent: cluster x walks its range of the tile stream.  Warp groups:
// 0 softmax, 1 Q/O, 2 dequant, 3 the QK^T and PV warps (CTA 0) and the two
// index warps (one also allocates TMEM).
template <class C>
CUTE_DEVICE void run_decode(const HcaParams& p, const HcaMaps& maps) {
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
    const float sink_log2 =
        p.sink ? __ldg(p.sink + t % kHeads) * CUDART_L2E_F : -CUDART_INF_F;
    tcgen05_before_thread_sync();
    cute::cluster_sync();
    tcgen05_after_thread_sync();
    const uint32_t tmem = sm.tmem_base;

    pdl_wait();
    find_range<C>(p, sm, cluster);
    const int n = sm.start[2];
    // The fused build merges at one call site shared by both groups, as two
    // inlined copies slow the no-merge launches through code placement.
    if (C::kFuse && wg < 2) {
        if (wg == 0) {
            cutlass::arch::warpgroup_reg_alloc<kSoftmaxRegs>();
            softmax_wg<C>(p, sm, tmem, t, cluster);
        } else {
            cutlass::arch::warpgroup_reg_alloc<kQoRegs>();
            qo_wg<C>(p, maps, sm, tmem, sink_log2, cta, t, cluster);
        }
        merge_tail<C>(p, sm, cta, cluster);
    } else if (wg == 0) {
        cutlass::arch::warpgroup_reg_alloc<kSoftmaxRegs>();
        softmax_wg<C>(p, sm, tmem, t, cluster);
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
            if (cta == 0) qk_warp<C>(p, sm, tmem, cluster);
        } else if (warp == kPvWarp) {
            if (cta == 0) pv_warp<C>(p, sm, tmem, cluster);
        } else if (warp - kIndexWarp < C::kIndexWarps) {
            // DSA runs a single index warp (kIndexWarps==1); kAllocWarp (warp
            // 15) only allocates/frees TMEM and idles here.
            index_warp<C>(p, maps, sm, cta, warp - kIndexWarp, lane, cluster);
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

__global__ void __launch_bounds__(kThreads, 1)
    hca_decode_attention_kernel(const __grid_constant__ HcaParams p,
                                const __grid_constant__ HcaMaps maps) {
    run_decode<HcaCfg>(p, maps);
}

__global__ void __launch_bounds__(kThreads, 1)
    hca_plain_attention_kernel(const __grid_constant__ HcaParams p,
                               const __grid_constant__ HcaMaps maps) {
    run_decode<HcaPlainCfg>(p, maps);
}

__global__ void __launch_bounds__(kThreads, 1)
    dsa_decode_attention_kernel(const __grid_constant__ HcaParams p,
                                const __grid_constant__ HcaMaps maps) {
    run_decode<DsaCfg>(p, maps);
}

// Separate combine kernel: one job per split pair, launched after the main
// kernel so every part's normalized partial O and log2 sum are already in
// global.  grid = (kPlanes, 2 CTAs, <=clusters jobs); a block owns one plane of
// one token's output and reduces it over the pair's part clusters.  The block's
// kCombSplit warp groups each reduce a stride-interleaved subset of the parts
// (raising warps-in-flight so the latency of the plane-contiguous, coalesced
// loads is hidden even for a single wide-split pair), then combine their
// partial sums through shared memory.  The weighted fp32 sum matches the
// in-kernel merge bit for bit, so precision is unchanged.
constexpr int kCombSplit = 4;
constexpr int kCombJobs = 8;  // job-axis cap of the combine grid
__global__ void hca_combine_kernel(const __grid_constant__ HcaParams p) {
    __shared__ float red[kCombSplit][kWarpGroup][kPlaneCols];
    __shared__ float redmd[kCombSplit][kWarpGroup][2];  // (running max, denom)
    // PDL: wait on the main kernel's launch_dependents before touching any of
    // its global output (eff_g/jobs/o_part/lse_part).  Only the grid-launch
    // and scheduling latency overlaps the main kernel's drain; the reads below
    // still observe fully-written partials (the main kernel triggers only after
    // all warpgroups have written their parts).  Triggering on entry lets the
    // next launch's main kernel run its prologue meanwhile; its own pdl_wait
    // still waits for this grid to complete before it rewrites the partials.
    pdl_trigger();
    pdl_wait();
    const int sched = *p.eff_g;
    if (sched == 0) return;
    const int cta = blockIdx.y;
    const int pl = blockIdx.x;                 // plane [0, kPlanes)
    const int grp = threadIdx.x / kWarpGroup;  // part-group [0, kCombSplit)
    const int tsi = threadIdx.x % kWarpGroup;  // storage thread index
    const int head = tsi % kHeads, half = tsi / kHeads;
    const int total = *p.total_g;
    const float sink_log2 =
        p.sink ? __ldg(p.sink + head) * CUDART_L2E_F : -CUDART_INF_F;
    // A block walks the job slots z, z + gridDim.z, ...: the job axis is
    // capped (kCombJobs) so a launch whose split merged inline costs few
    // block-launch waves.  The skip conditions are block-uniform.
    const int pairs =
        (p.tokens / p.tokens_per_req) * ((p.tokens_per_req + 1) / 2);
    for (int z = blockIdx.z; z < pairs; z += gridDim.z) {
        const MergeJob j = p.jobs[z];
        if (j.wn == 0 || (cta == 1 && !j.has_b)) continue;
        const int first = cluster_of(sched, total, j.w0);
        const int last = cluster_of(sched, total, j.w0 + j.wn - 1);
        // A cluster contributes a part only if its range is non-empty (matches
        // for_parts); its part lives in slot cluster*2 + (cluster == first):
        // the owner (c == first) wrote the trailing slot, the rest the leading.
        const auto empty = [&](int c) {
            return range_lo(c + 1, total, sched) <= range_lo(c, total, sched);
        };
        const auto sid = [&](int c) { return c * 2 + (c == first); };
        // Single online-softmax pass over this group's stride-interleaved
        // subset: each part's lse and plane are read exactly once, the running
        // max rescales the fp32 accumulator on the fly.  Groups merge their
        // (max, denom, acc) partials below, so no part is rescanned.
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
                (int64_t((sid(c) * 2 + cta) * kPlanes + pl) * kWarpGroup +
                 tsi) *
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
        if (grp == 0) {
            // Flash-merge the group partials, folding in the attention sink
            // (denom only), then normalize.  Identical to the two-pass merge.
            float mx = sink_log2;
            CUTLASS_PRAGMA_UNROLL
            for (int g = 0; g < kCombSplit; ++g)
                mx = fmaxf(mx, redmd[g][tsi][0]);
            float dtot =
                sink_log2 > -CUDART_INF_F ? exp2f(sink_log2 - mx) : 0.0f;
            CUTLASS_PRAGMA_UNROLL
            for (int e = 0; e < kPlaneCols; ++e) acc[e] = 0.0f;
            CUTLASS_PRAGMA_UNROLL
            for (int g = 0; g < kCombSplit; ++g) {
                const float sc = exp2f(redmd[g][tsi][0] - mx);
                dtot += redmd[g][tsi][1] * sc;
                CUTLASS_PRAGMA_UNROLL
                for (int e = 0; e < kPlaneCols; ++e)
                    acc[e] += red[g][tsi][e] * sc;
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
        }
        __syncthreads();  // red / redmd are rewritten by the next job
    }
}

CUtensorMap make_map(const void* ptr, CUtensorMapDataType dtype,
                     const torch::Tensor& pool, uint64_t cols, uint32_t box,
                     CUtensorMapSwizzle swizzle) {
    CUtensorMap map;
    EncodeTmaDesc<2>(
        &map, dtype, ptr,
        {cols, uint64_t(pool.size(0) * (pool.stride(0) / kRowBytes))},
        {kRowBytes}, {box, 1}, swizzle, "hca_decode_attention");
    return map;
}

void set_pool(HcaParams& p, HcaMaps& maps, int seg, const torch::Tensor& pool,
              const torch::Tensor& idx, const torch::Tensor& len,
              int64_t tokens) {
    TORCH_CHECK(pool.dim() == 4 && pool.scalar_type() == torch::kByte &&
                    pool.size(2) == 1 && pool.size(3) == kRowBytes + kSfBytes &&
                    pool.stride(0) % kRowBytes == 0,
                "k caches must be uint8 [pages, page_size, 1, 584] with a page "
                "stride that is a multiple of 576");
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

// Split-pair partials and counters, kept per device: the counters start at
// zero and the last part of each split pair resets its own.
struct Workspace {
    torch::Tensor o_part, lse_part, counter, jobs, total_g, eff_g;
};

Workspace& workspace(const torch::Tensor& q, int clusters) {
    static std::mutex mu;
    static std::unordered_map<int, Workspace> cache;
    const std::lock_guard<std::mutex> lock(mu);
    Workspace& ws = cache[q.get_device()];
    if (!ws.counter.defined() || ws.counter.size(0) != clusters) {
        const auto f32 = q.options().dtype(torch::kFloat);
        const auto i32 = q.options().dtype(torch::kInt);
        // Two partial slots per cluster (index cluster*2 + which): a cluster
        // can be the leading continuation (which=0) of one split pair and the
        // trailing owner (which=1) of another, so one slot each avoids a clash.
        ws.o_part = torch::empty(
            {2 * clusters, 2, kPlanes, kWarpGroup, kPlaneCols}, f32);
        ws.lse_part = torch::empty({2 * clusters, 2, kHeads}, f32);
        ws.counter = torch::zeros({clusters, 2}, i32);
        ws.jobs =
            torch::empty({clusters, 4}, i32);  // MergeJob[pairs <= clusters]
        ws.total_g = torch::zeros({1}, i32);
        ws.eff_g = torch::zeros({1}, i32);
    }
    return ws;
}

// Scheduling, workspace sizing and the two kernel launches, shared by the HCA
// and DSA entry points.  The callers differ only in the K-cache pool layout
// (set_pool vs set_pool_dsa) and the Q tensor-map column count, all already in
// p / maps by the time they call here, so the launch tail itself is identical.
void launch_decode(HcaParams& p, const HcaMaps& maps, const torch::Tensor& q,
                   void (*kernel)(HcaParams, HcaMaps), int smem_bytes,
                   bool use_pdl) {
    // Work-aware sizing (p.eff): when few pairs, the legacy heuristic launches
    // < max_clusters and starves a long (pole) pair of split-KV parallelism, so
    // eff launches wide and sizes the scheduled clusters to the tile work; the
    // separate combine kernel merges the wide split.  When many pairs the
    // legacy path already fills the machine and inline-merges cheaply.
    const int max_clusters = GetSmCount() / 2;
    const int64_t pairs =
        int64_t(p.tokens) / p.tokens_per_req * ((p.tokens_per_req + 1) / 2);
    const int heuristic = int(2 * pairs <= max_clusters ? 2 * pairs
                              : pairs <= max_clusters   ? pairs
                                                        : max_clusters);
    // DSA: the wide-split + separate combine (eff=1) path only pays at very
    // low batch, where idle SMs far outnumber tokens and a single token's
    // 2048-KV latency dominates.  Above that the inline 2-way split-pair merge
    // (eff=0) wins, since it avoids the throughput-bound low-conc combine
    // kernel.
    // HCA's 2*pairs<=max_clusters region launches wide in auto mode (eff 2):
    // eff 1 only wins for long KV at small batch, the legacy path everywhere
    // else, and only the device knows the lengths, so find_range picks per
    // launch.
    const bool is_dsa = (kernel == dsa_decode_attention_kernel);
    const int eff_mul = is_dsa ? 8 : 2;
    p.eff = eff_mul * pairs <= max_clusters ? (is_dsa ? 1 : 2) : 0;
    // Between that and P <= max_clusters HCA launches wide and find_range
    // decides per launch whether to use the spare clusters (eff 3).
    if (!is_dsa && p.eff == 0 && pairs <= max_clusters) p.eff = 3;
    p.tpc = kDefaultTpc;
    p.clusters = p.eff ? max_clusters : heuristic;
    // eff leaves the split uncapped (work-aware sizing + cheap wide combine);
    // the legacy inline-merge path keeps the adaptive per-pair cap.
    p.smax = p.eff == 1 ? 0 : kDefaultSmax;
    p.cap = kNoCap;
    p.nomerge = 0;
    p.pipe = 1;
    p.combine = p.eff == 1 || p.eff == 2 ? 1 : 0;
    if (pairs > max_clusters) p.combine = 0;  // jobs has a slot per pair
    p.fuse = is_dsa ? 0 : 1;                  // DSA keeps the combine kernel
    p.ks_kmax = p.eff == 2 && p.combine && p.fuse && pairs > 0
                    ? p.clusters / int(pairs)
                    : 0;
    p.ks_b = 0.077f + 0.062f * float(pairs);
    p.ks_a = 1.17f / p.ks_b;
    p.ks_rp = pairs > 0 ? 1.f / float(pairs) : 0.f;
    if (kernel == hca_decode_attention_kernel && !(p.combine && p.fuse))
        kernel = hca_plain_attention_kernel;
    Workspace& ws = workspace(q, max_clusters);
    p.o_part = ws.o_part.data_ptr<float>();
    p.lse_part = ws.lse_part.data_ptr<float>();
    p.counter = ws.counter.data_ptr<int>();
    p.jobs = reinterpret_cast<MergeJob*>(ws.jobs.data_ptr<int>());
    p.total_g = ws.total_g.data_ptr<int>();
    p.eff_g = ws.eff_g.data_ptr<int>();
    const cudaStream_t stream = GetCurrentCUDAStream(q);
    HALOPS_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
    HALOPS_CUDA_CHECK(LaunchKernelWithPdl(
        kernel, dim3(2 * p.clusters), dim3(kThreads), dim3(2, 1, 1), smem_bytes,
        stream, use_pdl, p, maps));
    if (p.combine && !p.fuse) {
        // Same stream: the main kernel's partials are visible before combine.
        // PDL: launch combine as a programmatic dependent so its grid-launch
        // latency overlaps the main kernel's drain (griddepcontrol.wait inside
        // combine gates the actual reads).
        // One job slot per pair (pairs <= clusters here), the job axis capped
        // at kCombJobs: surplus blocks would only exit, yet still cost
        // block-launch waves, also when the device merged inline (eff 2).
        HALOPS_CUDA_CHECK(LaunchKernelWithPdl(
            hca_combine_kernel,
            dim3(kPlanes, 2, int(std::min<int64_t>(pairs, kCombJobs))),
            dim3(kCombSplit * kWarpGroup), kNoCluster, 0, stream, use_pdl, p));
    }
}

// DSA tensor map: UINT16 view of the 656 B inline rows, box = this CTA's 256 B
// fp8 nope half (128 uint16), gathered per key row like HCA's raw map.
CUtensorMap make_map_dsa(const void* ptr, const torch::Tensor& pool,
                         uint64_t cols, uint32_t box) {
    CUtensorMap map;
    EncodeTmaDesc<2>(
        &map, CU_TENSOR_MAP_DATA_TYPE_UINT16, ptr,
        {cols, uint64_t(pool.size(0) * (pool.stride(0) / kDsaRowBytes))},
        {kDsaRowBytes}, {box, 1}, CU_TENSOR_MAP_SWIZZLE_NONE,
        "dsa_decode_attention");
    return map;
}

// DSA uses a single c128-style pool (seg 0) of 656 B inline rows and has no SWA
// segment.  Seg 1 is cloned from seg 0 with ld == 0 so the shared scheduler's
// unconditional length reads touch valid memory yet yield zero SWA keys.
void set_pool_dsa(HcaParams& p, HcaMaps& maps, const torch::Tensor& pool,
                  const torch::Tensor& idx, const torch::Tensor& len,
                  int64_t tokens) {
    TORCH_CHECK(pool.dim() == 4 && pool.scalar_type() == torch::kByte &&
                    pool.size(2) == 1 && pool.size(3) == kDsaRowBytes &&
                    pool.stride(0) % kDsaRowBytes == 0,
                "DSA k cache must be uint8 [pages, page_size, 1, 656] with a "
                "page stride that is a multiple of 656");
    TORCH_CHECK(idx.scalar_type() == torch::kInt && idx.is_contiguous() &&
                    idx.dim() >= 2 && idx.numel() == tokens * idx.size(-1),
                "indices must be contiguous int32 [T, W] or [T, 1, W]");
    TORCH_CHECK(len.scalar_type() == torch::kInt && len.is_contiguous() &&
                    len.numel() == tokens,
                "lengths must be contiguous int32 [T]");
    p.idx[0] = idx.data_ptr<int>();
    p.len[0] = len.data_ptr<int>();
    p.ld[0] = idx.size(-1);
    p.pool[0] = pool.data_ptr<uint8_t>();
    p.pool_stride[0] = pool.stride(0);
    const int page = pool.size(1);
    TORCH_CHECK((page & (page - 1)) == 0,
                "k cache page sizes must be powers of two, got ", page);
    p.page_shift[0] = __builtin_ctz(page);
    p.page_rows[0] = pool.stride(0) / kDsaRowBytes;
    p.pool_tokens[0] = pool.size(0) * pool.size(1);
    const uint8_t* base = pool.data_ptr<uint8_t>();
    for (int c = 0; c < 2; ++c) {
        maps.raw[c][0] =
            make_map_dsa(base, pool, kDsaRowBytes / 2, kDsaNopeHalf / 2);
    }
    // Zero-length seg-1 clone: valid pointers, ld == 0 clamps every SWA length
    // to 0 (pair_geom: n_s == 0), so no seg-1 tile or gather is ever scheduled.
    p.idx[1] = p.idx[0];
    p.len[1] = p.len[0];
    p.ld[1] = 0;
    p.pool[1] = p.pool[0];
    p.pool_stride[1] = p.pool_stride[0];
    p.page_shift[1] = p.page_shift[0];
    p.page_rows[1] = p.page_rows[0];
    p.pool_tokens[1] = p.pool_tokens[0];
    maps.raw[0][1] = maps.raw[0][0];
    maps.raw[1][1] = maps.raw[1][0];
}

}  // namespace detail

using namespace detail;

torch::Tensor HcaDecodeAttention(
    const torch::Tensor& q, const torch::Tensor& swa_k_cache,
    const torch::Tensor& swa_indices, const torch::Tensor& swa_lengths,
    const torch::Tensor& c128_k_cache, const torch::Tensor& c128_indices,
    const torch::Tensor& c128_lengths,
    const std::optional<torch::Tensor>& attn_sink, const double softmax_scale,
    const int64_t tokens_per_req, const std::optional<torch::Tensor>& out_opt,
    const bool use_pdl) {
    const CudaDeviceGuard device_guard(q.device());
    const int64_t tokens = q.size(0);
    TORCH_CHECK(q.is_cuda() && q.scalar_type() == torch::kBFloat16 &&
                    q.is_contiguous() && q.numel() == tokens * kHeads * kDim,
                "q must be a contiguous bfloat16 [T, 64, 512] tensor");
    TORCH_CHECK(tokens_per_req >= 2 && tokens % tokens_per_req == 0,
                "tokens_per_req must be >= 2 and divide T");
    torch::Tensor out = out_opt.has_value() ? *out_opt : torch::empty_like(q);
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

    HcaParams p{};
    HcaMaps maps{};
    set_pool(p, maps, 0, c128_k_cache, c128_indices, c128_lengths, tokens);
    set_pool(p, maps, 1, swa_k_cache, swa_indices, swa_lengths, tokens);
    p.q = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr());
    p.sink = attn_sink.has_value() ? attn_sink->data_ptr<float>() : nullptr;
    p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
    p.tokens = tokens;
    p.tokens_per_req = int(tokens_per_req);
    p.scale_log2 = float(softmax_scale) * CUDART_L2E_F;
    EncodeTmaDesc<2>(&maps.q, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, p.q,
                     {uint64_t(kDim), uint64_t(tokens * kHeads)},
                     {uint64_t(kDim) * sizeof(__nv_bfloat16)},
                     {uint32_t(kQBoxDims), uint32_t(kHeads)},
                     CU_TENSOR_MAP_SWIZZLE_128B, "hca_decode_attention q");
    EncodeTmaDesc<2>(&maps.out, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, p.out,
                     {uint64_t(kDim), uint64_t(tokens * kHeads)},
                     {uint64_t(kDim) * sizeof(__nv_bfloat16)},
                     {uint32_t(kQBoxDims), uint32_t(kHeads)},
                     CU_TENSOR_MAP_SWIZZLE_128B, "hca_decode_attention out");
    launch_decode(p, maps, q, hca_decode_attention_kernel, kSmemBytes, use_pdl);
    return out;
}

// DSA (full-MLA) decode: q is [T, 64, 576] bf16 (512 latent + 64 rope), the
// single K cache holds 656 B inline fp8 rows, and the output is [T, 64, 512].
// There is no SWA segment.  Everything past the pool layout and the 576-wide Q
// tensor map is the shared launch / scheduling / combine path.
torch::Tensor DsaDecodeAttention(
    const torch::Tensor& q, const torch::Tensor& k_cache,
    const torch::Tensor& indices, const torch::Tensor& lengths,
    const std::optional<torch::Tensor>& attn_sink, const double softmax_scale,
    const int64_t tokens_per_req, const std::optional<torch::Tensor>& out_opt,
    const bool use_pdl) {
    const CudaDeviceGuard device_guard(q.device());
    const int64_t tokens = q.size(0);
    TORCH_CHECK(q.is_cuda() && q.scalar_type() == torch::kBFloat16 &&
                    q.is_contiguous() && q.numel() == tokens * kHeads * kDsaQk,
                "q must be a contiguous bfloat16 [T, 64, 576] tensor");
    TORCH_CHECK(tokens_per_req >= 1 && tokens % tokens_per_req == 0,
                "tokens_per_req must be >= 1 and divide T");
    torch::Tensor out = out_opt.has_value()
                            ? *out_opt
                            : torch::empty({tokens, kHeads, kDim}, q.options());
    TORCH_CHECK(out.scalar_type() == torch::kBFloat16 && out.is_contiguous() &&
                    out.numel() == tokens * kHeads * kDim,
                "out must be a contiguous bfloat16 [T, 64, 512] tensor");
    for (const torch::Tensor* x : std::initializer_list<const torch::Tensor*>{
             &k_cache, &indices, &lengths, &out}) {
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

    HcaParams p{};
    HcaMaps maps{};
    set_pool_dsa(p, maps, k_cache, indices, lengths, tokens);
    p.q = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr());
    p.sink = attn_sink.has_value() ? attn_sink->data_ptr<float>() : nullptr;
    p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
    p.tokens = tokens;
    p.tokens_per_req = int(tokens_per_req);
    p.scale_log2 = float(softmax_scale) * CUDART_L2E_F;
    // Q row stride is the full 576-dim contraction; the 64 rope dims live past
    // the 512 latent the tensor-map boxes read, loaded directly in store_q.
    EncodeTmaDesc<2>(&maps.q, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, p.q,
                     {uint64_t(kDsaQk), uint64_t(tokens * kHeads)},
                     {uint64_t(kDsaQk) * sizeof(__nv_bfloat16)},
                     {uint32_t(kQBoxDims), uint32_t(kHeads)},
                     CU_TENSOR_MAP_SWIZZLE_128B, "dsa_decode_attention q");
    EncodeTmaDesc<2>(&maps.out, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, p.out,
                     {uint64_t(kDim), uint64_t(tokens * kHeads)},
                     {uint64_t(kDim) * sizeof(__nv_bfloat16)},
                     {uint32_t(kQBoxDims), uint32_t(kHeads)},
                     CU_TENSOR_MAP_SWIZZLE_128B, "dsa_decode_attention out");
    launch_decode(p, maps, q, dsa_decode_attention_kernel, kDsaSmemBytes,
                  use_pdl);
    return out;
}

}  // namespace halops
