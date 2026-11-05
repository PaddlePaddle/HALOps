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

#include <cstdint>

#if defined(ENABLE_KERNEL_TIMER) || defined(INFERROCK_ENABLE_KERNEL_TIMER)
#    define HALOPS_KERNEL_TIMER_ENABLED
#endif

namespace halops {

// Two uint64 words per record, stored as one 16-byte `st.global.v2.u64`:
//
//   [0] = site_id | (dir << 32) | (warp << 40) | (arg << 48)
//   [1] = %globaltimer in ns; 0 marks an unused slot
//
// `dir` occupies bits 32..33 and `kTimerFlagArg` (bit 34) says the record
// carries a runtime `arg` in bits 48..63. The kernel uses it as the expert id,
// so a bar says WHICH expert it belongs to; trace.py just reports it as an
// opaque 16-bit integer. A span's `begin` owns the arg -- its `end` carries
// none.
//
// `site_id` identifies the CALL SITE, not an entry in a hand-maintained event
// table:
//
//   site_id = (fold16(fnv1a(basename(__FILE__))) << 16) | (__LINE__ & 0xffff)
//
// It is a compile-time constant (see device_timer.hpp), so a probe costs one
// `%globaltimer` read plus one store and carries no immediate-building code.
// `halops/moe/device_timer.py` recomputes the same hash over the repo's
// kernel sources to recover `file:line`, and takes the display name from the
// string literal written at that call site -- so neither the host nor python
// keeps an event enum, a name map, or a mirror of one.
inline constexpr std::uint32_t kTimerDirInstant = 0u;
inline constexpr std::uint32_t kTimerDirBegin = 1u;
inline constexpr std::uint32_t kTimerDirEnd = 2u;
// Written into a warp's LAST slot the first time that warp has to drop a
// record; `site_id` names the dropped call site, so an overflowing stream says
// where it overflowed instead of just ending early.
inline constexpr std::uint32_t kTimerDirTruncated = 3u;
// OR-ed into `dir` when bits 48..63 of the record hold a valid arg.
inline constexpr std::uint32_t kTimerFlagArg = 4u;

// Pool geometry. Both are powers of two and both are reported to python by the
// host export, so no decoder hard-codes them. The per-K-step mbarrier waits are
// probes too, which puts a warp at ~370 records for m=8/top-k=10; override at
// build time when a bigger shape still truncates.
#ifndef HALOPS_TIMER_WARPS
#    define HALOPS_TIMER_WARPS 16
#endif
#ifndef HALOPS_TIMER_CAP
#    define HALOPS_TIMER_CAP 1024
#endif

inline constexpr int kTimerWarps = HALOPS_TIMER_WARPS;
inline constexpr int kTimerCap = HALOPS_TIMER_CAP;
// Pool layout: [cta][warp][slot][word].
inline constexpr int kTimerCtaU64 = kTimerWarps * kTimerCap * 2;

inline constexpr int kTimerWords(const int num_ctas) {
    return num_ctas * kTimerCtaU64;
}

}  // namespace halops
