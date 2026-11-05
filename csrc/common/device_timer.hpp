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

#include "common/kernel_timer_types.hpp"
#include "cutlass/cutlass.h"

// Per-warp, per-call-site device event timer.
//
// A probe is identified by WHERE IT IS, not by an entry in an event enum:
// every `KERNEL_TIMER_*` macro folds `basename(__FILE__)` and `__LINE__` into
// a compile-time 32-bit id (see kernel_timer_types.hpp for the record layout),
// and trace.py recovers `file:line` plus the literal tag from the sources. Two
// probes that mean different things therefore cannot collide the way two
// `mark(kSameEvent)` call sites did, and adding one is a one-line edit with no
// table to keep in sync on the host or in python.
//
// Every warp that reaches a probe emits its own record, into its own slot
// range, with no atomics and no cross-warp contention. One warp's stream is
// therefore a totally ordered local trace, and the (cta, warp) pair is the
// timeline track.

namespace halops {

#define KERNEL_TIMER_CONSTEXPR __host__ __device__ constexpr

// The BASENAME of __FILE__ is hashed: the JIT copies the kernel tree into
// a per-run gen directory, so the full path changes between runs while the
// basename is stable, and the basename is what trace.py can resolve back to
// a file in the repo.
KERNEL_TIMER_CONSTEXPR const char* TimerBasename(const char* p,
                                                 const char* last) {
    return *p == '\0'
               ? last
               : TimerBasename(p + 1, (*p == '/' || *p == '\\') ? p + 1 : last);
}

KERNEL_TIMER_CONSTEXPR std::uint32_t TimerFnv1a(const char* s,
                                                std::uint32_t h) {
    return *s == '\0'
               ? h
               : TimerFnv1a(s + 1, (h ^ static_cast<std::uint32_t>(
                                            static_cast<unsigned char>(*s))) *
                                       16777619u);
}

KERNEL_TIMER_CONSTEXPR std::uint32_t TimerFold16(const std::uint32_t h) {
    return ((h >> 16) ^ h) & 0xffffu;
}

// `tag` is deliberately unused: the device only needs file + line, and trace.py
// reads the literal straight off the resolved source line. Taking it as a
// parameter still type-checks that the probe was given a string literal, which
// is what makes the trace readable.
KERNEL_TIMER_CONSTEXPR std::uint32_t TimerSiteId(const char* /*tag*/,
                                                 const char* path,
                                                 const int line) {
    return (TimerFold16(TimerFnv1a(TimerBasename(path, path), 2166136261u))
            << 16) |
           (static_cast<std::uint32_t>(line) & 0xffffu);
}

// Routing the id through a template argument forces it to a SASS immediate: a
// probe must not spend instructions computing its own identity.
template <std::uint32_t kId>
struct TimerSite {
    static constexpr std::uint32_t value = kId;
};

#define KERNEL_TIMER_SITE(tag)                                  \
    (::halops::TimerSite<::halops::TimerSiteId((tag), __FILE__, \
                                               __LINE__)>::value)

struct alignas(16) TimerRecord {
    std::uint64_t word;  // site_id | dir << 32 | warp << 40 | arg << 48
    std::uint64_t t;     // %globaltimer, ns (0 means "slot never written")
};

struct DeviceTimer {
    std::uint64_t* base;  // null when off (no pool was provided)
    std::uint32_t* next;  // shared memory: per-warp slot cursors
    int cta;
    int warp;

    CUTLASS_DEVICE static DeviceTimer make(std::uint64_t* b,
                                           std::uint32_t* smem_next,
                                           const int cta, const int warp) {
        DeviceTimer t;
        t.base = b;
        t.next = smem_next;
        t.cta = cta;
        t.warp = warp;
        return t;
    }

    CUTLASS_DEVICE bool on() const { return base != nullptr; }

    // An instant: this warp was HERE at this time.
    CUTLASS_DEVICE void mark(const std::uint32_t site) const {
        record(site, kTimerDirInstant, 0u);
    }

    // The `arg` overloads tag a record with a 16-bit runtime value (the
    // kernel passes the expert id). A span's `end` takes none: `begin` owns it.
    CUTLASS_DEVICE void mark(const std::uint32_t site,
                             const std::uint32_t arg) const {
        record(site, kTimerDirInstant | kTimerFlagArg, arg);
    }

    // A duration. `begin` hands back the id that `end` must be given, so a
    // pair stays matched even when the two calls sit on different lines or in
    // different functions, and so the trace knows the span's name from one
    // place. Prefer KERNEL_TIMER_SCOPE when the region is a lexical scope.
    CUTLASS_DEVICE std::uint32_t begin(const std::uint32_t site) const {
        record(site, kTimerDirBegin, 0u);
        return site;
    }

    CUTLASS_DEVICE std::uint32_t begin(const std::uint32_t site,
                                       const std::uint32_t arg) const {
        record(site, kTimerDirBegin | kTimerFlagArg, arg);
        return site;
    }

    CUTLASS_DEVICE void end(const std::uint32_t site) const {
        record(site, kTimerDirEnd, 0u);
    }

    CUTLASS_DEVICE std::uint64_t now() const {
        std::uint64_t t;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
        return t;
    }

    // Warp 0 of each CTA, once: two instants sharing one id (the cost of a
    // `%globaltimer` read) and one empty pair (the cost of a whole span).
    CUTLASS_DEVICE void calibrate() const {
        if (base == nullptr || warp != 0) {
            return;
        }
        const std::uint32_t read_cost = KERNEL_TIMER_SITE("calib_timer");
        mark(read_cost);
        mark(read_cost);
        end(begin(KERNEL_TIMER_SITE("calib_pair")));
    }

  private:
    CUTLASS_DEVICE void record(const std::uint32_t site,
                               const std::uint32_t dir,
                               const std::uint32_t arg) const {
        if (base == nullptr) {
            return;
        }
        // Lane 0 writes the warp's record.
        std::uint32_t lane;
        asm("mov.u32 %0, %%laneid;" : "=r"(lane));
        if (lane != 0u) {
            return;
        }
        // Read the clock BEFORE the slot bookkeeping: a record must date the
        // program point, not the probe's own shared-memory round trip.
        std::uint64_t t;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
        // The cursor is this warp's own uint32 in shared memory, so the stream
        // is ordered without a single atomic or barrier.
        const std::uint32_t slot = next[warp]++;
        if (slot >= static_cast<std::uint32_t>(kTimerCap) - 1u) {
            if (slot == static_cast<std::uint32_t>(kTimerCap) - 1u) {
                write(site, kTimerDirTruncated, 0u, kTimerCap - 1, t);
            }
            return;
        }
        write(site, dir, arg, slot, t);
    }

    CUTLASS_DEVICE void write(const std::uint32_t site, const std::uint32_t dir,
                              const std::uint32_t arg, const std::uint32_t slot,
                              const std::uint64_t t) const {
        TimerRecord* rec = reinterpret_cast<TimerRecord*>(
            base + static_cast<std::size_t>(cta) * kTimerCtaU64 +
            static_cast<std::size_t>(warp) * (kTimerCap * 2) + slot * 2);
        *rec = TimerRecord{
            static_cast<std::uint64_t>(site) |
                (static_cast<std::uint64_t>(dir) << 32) |
                (static_cast<std::uint64_t>(static_cast<std::uint32_t>(warp))
                 << 40) |
                (static_cast<std::uint64_t>(arg & 0xffffu) << 48),
            t};
    }
};

// A duration bounded by a lexical scope: begin on construction, end on
// destruction, so a region that `return`s or `break`s out still closes its
// span. It keeps a copy of the writer live for the whole region (~7
// registers); where that matters, use KERNEL_TIMER_BEGIN/END, which keeps only
// the 32-bit id alive and re-mints the writer from shared memory to close.
struct TimerScope {
    DeviceTimer timer;
    std::uint32_t site;

    CUTLASS_DEVICE TimerScope(const DeviceTimer t, const std::uint32_t s)
        : timer(t), site(s) {
        timer.begin(site);
    }

    CUTLASS_DEVICE TimerScope(const DeviceTimer t, const std::uint32_t s,
                              const std::uint32_t arg)
        : timer(t), site(s) {
        timer.begin(site, arg);
    }

    CUTLASS_DEVICE ~TimerScope() { timer.end(site); }

    TimerScope(const TimerScope&) = delete;
    TimerScope& operator=(const TimerScope&) = delete;
};

}  // namespace halops

// In a non-timer build every one of these expands to a void
// expression that does not even evaluate `timer`.
//
//   KERNEL_TIMER_MARK(t, "tag")        one record: this warp was here
//   KERNEL_TIMER_SCOPE(t, "tag")       a span over the enclosing scope
//   auto h = KERNEL_TIMER_BEGIN(t, "tag");  ... KERNEL_TIMER_END(t, h);
//   KERNEL_TIMER_CALIBRATE(t)          warp 0: instrumentation cost probes
//
// MARK / BEGIN / SCOPE take an optional last argument,
// `KERNEL_TIMER_SCOPE(t, "tag", expert_id)`: a 16-bit value stored in the
// record and shown beside the bar. It is evaluated only in a timer build, so
// it may read shared memory freely. `KERNEL_TIMER_END` takes none -- the
// `begin` owns the arg.
//
// A span must open and close in the SAME warp; to time an interval whose ends
// belong to different warps, mark both ends and difference them in analysis.
#define KERNEL_TIMER_CAT_IMPL(a, b) a##b
#define KERNEL_TIMER_CAT(a, b) KERNEL_TIMER_CAT_IMPL(a, b)

#if defined(HALOPS_KERNEL_TIMER_ENABLED)
#    define KERNEL_TIMER_MARK(timer, tag, ...) \
        (timer).mark(KERNEL_TIMER_SITE(tag) __VA_OPT__(, ) __VA_ARGS__)
#    define KERNEL_TIMER_BEGIN(timer, tag, ...) \
        (timer).begin(KERNEL_TIMER_SITE(tag) __VA_OPT__(, ) __VA_ARGS__)
#    define KERNEL_TIMER_END(timer, handle) (timer).end(handle)
#    define KERNEL_TIMER_CALIBRATE(timer) (timer).calibrate()
#    define KERNEL_TIMER_SCOPE(timer, tag, ...)                               \
        ::halops::TimerScope KERNEL_TIMER_CAT(kernel_timer_scope_, __LINE__)( \
            (timer), KERNEL_TIMER_SITE(tag) __VA_OPT__(, ) __VA_ARGS__)
#else
#    define KERNEL_TIMER_MARK(timer, tag, ...) ((void)0)
#    define KERNEL_TIMER_BEGIN(timer, tag, ...) (0u)
#    define KERNEL_TIMER_END(timer, handle) ((void)(handle))
#    define KERNEL_TIMER_CALIBRATE(timer) ((void)0)
#    define KERNEL_TIMER_SCOPE(timer, tag, ...) ((void)0)
#endif
