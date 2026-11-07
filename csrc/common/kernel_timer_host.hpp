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

#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>

#include "common/kernel_timer.hpp"

// Common host side of the kernel event timer: allocate the record pool, zero
// it per launch, copy it back, hand it to python.
//
// There is no event vocabulary to carry any more, a probe is identified by
// its own file:line and device_timer.py resolves that against
// the kernel sources, so this layer is kernel-independent. The pool itself
// is a process-wide singleton; the free functions below are its interface.
//
// A pool has one independent lane per concurrently timed kernel (or stream):
// lane 0 is the default and the plain overloads forward to it, so a single
// kernel needs no lane at all while several can share the pool without
// overwriting each other's records.
namespace halops {

#if defined(HALOPS_KERNEL_TIMER_ENABLED)

// (Re)allocate the device record pool for num_ctas CTAs and zero it on the
// stream. Only the CTAs of this launch are zeroed and copied back; the pool
// keeps its high-water-mark allocation across launches.
void TimerStage(const int num_ctas, cudaStream_t stream);
void TimerStage(const int lane, const int num_ctas, cudaStream_t stream);
uint64_t *TimerDeviceBuffer();
uint64_t *TimerDeviceBuffer(const int lane);

// Synchronize and copy the records back to a host tensor.
void TimerCollect(cudaStream_t stream);
void TimerCollect(const int lane, cudaStream_t stream);

class KernelTimerLaunch {
  public:
    // `lane` separates kernels that are timed at the same time; a lone kernel
    // keeps the default lane 0.
    KernelTimerLaunch(const int num_ctas, cudaStream_t stream,
                      const int lane = 0)
        : stream_(stream), lane_(lane) {
        cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
        cudaStreamIsCapturing(stream_, &capture);
        active_ = capture == cudaStreamCaptureStatusNone;
        if (active_) {
            TimerStage(lane_, num_ctas, stream_);
        }
    }

    uint64_t *events() const {
        return active_ ? TimerDeviceBuffer(lane_) : nullptr;
    }

    void collect() {
        if (active_) {
            TimerCollect(lane_, stream_);
            active_ = false;
        }
    }

  private:
    cudaStream_t stream_;
    int lane_{0};
    bool active_{false};
};

#else

inline void TimerStage(const int, cudaStream_t) {}
inline void TimerStage(const int, const int, cudaStream_t) {}
inline uint64_t *TimerDeviceBuffer() { return nullptr; }
inline uint64_t *TimerDeviceBuffer(const int) { return nullptr; }
inline void TimerCollect(cudaStream_t) {}
inline void TimerCollect(const int, cudaStream_t) {}

class KernelTimerLaunch {
  public:
    KernelTimerLaunch(const int, cudaStream_t, const int = 0) {}
    uint64_t *events() const { return nullptr; }
    void collect() {}
};

#endif

// The export is bound unconditionally by `cpp_extensions.cc`, so it is
// defined (as an empty dict) in a timer-less build too.
//
// Python-facing export: records plus the pool geometry device_timer.py needs
// to reshape them as [cta][warp][slot][word].
pybind11::dict GetKernelTimer(const int lane = 0);

}  // namespace halops
