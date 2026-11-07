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

// Extracted from InferRock csrc/kernel/attention/decoder_attention.cu.
// Compile this TU once alongside a PyTorch/pybind timer-enabled kernel.

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <vector>

#include "common/kernel_timer_host.hpp"

namespace halops {

#if defined(HALOPS_KERNEL_TIMER_ENABLED)
namespace {

// A single lane's record pool: one device buffer at the high-water mark of
// every launch's grid, plus the last pull.
struct TimerLane {
    std::uint64_t* buffer = nullptr;
    int capacity = 0;  // high-water mark: what buffer holds
    int used = 0;      // the last staged launch's grid
    bool zeroed = false;
    bool pending = false;
    cudaStream_t pending_stream = nullptr;
    torch::Tensor records;
};

// The process-wide timer pool. Each lane is an independent record buffer, so
// several kernels (or the same kernel on different streams) can be timed at
// the same time without overwriting each other's records. Lane 0 is the
// default and the free functions below stay lane-0 forwarders, keeping the
// existing callers and the python decoder unchanged.
class KernelTimerPool {
  public:
    static KernelTimerPool& instance() {
        static KernelTimerPool pool;
        return pool;
    }

    KernelTimerPool(const KernelTimerPool&) = delete;
    KernelTimerPool& operator=(const KernelTimerPool&) = delete;

    // (Re)allocate the lane's pool for num_ctas CTAs and zero this launch's
    // slice. Only the CTAs of this launch are zeroed and copied back; the
    // allocation keeps its high-water mark across launches.
    void stage(const int lane_id, const int num_ctas, cudaStream_t stream) {
        TimerLane& lane = get(lane_id);
        // Lazy reuse is valid only for repeated launches with the same grid: a
        // different grid writes a different set of slots.
        if (lane.used != num_ctas) {
            lane.zeroed = false;
        }
        if (lane.capacity < num_ctas) {
            if (lane.buffer != nullptr) {
                cudaFree(lane.buffer);
            }
            cudaMalloc(&lane.buffer,
                       static_cast<size_t>(kTimerWords(num_ctas)) * 8);
            lane.capacity = num_ctas;
            lane.zeroed = false;
        }
        // Zero and later copy back only the CTAs of this launch. The
        // allocation is kept at its high-water mark, but a launch smaller than
        // the one before it must not read the previous launch's records as its
        // own.
        lane.used = num_ctas;
        // Lazy staging: one zeroing per measurement burst. Every launch of a
        // given shape writes the same slots, so a pool zeroed before the burst
        // and re-zeroed after the next pull is as clean as a per-launch memset
        // -- and does not put 10MB of DRAM writes between two kernels.
        if (!lazy_ || !lane.zeroed) {
            cudaMemsetAsync(lane.buffer, 0,
                            static_cast<size_t>(kTimerWords(lane.used)) * 8,
                            stream);
            lane.zeroed = true;
        }
    }

    std::uint64_t* device_buffer(const int lane_id) {
        return get(lane_id).buffer;
    }

    // Synchronize and copy the records back to a host tensor. With lazy
    // staging the copy is deferred to the next export_records().
    void collect(const int lane_id, cudaStream_t stream) {
        TimerLane& lane = get(lane_id);
        if (lazy_) {
            lane.pending_stream = stream;
            lane.pending = true;
            return;
        }
        collect_now(lane, stream);
    }

    pybind11::dict export_records(const int lane_id) {
        TimerLane& lane = get(lane_id);
        if (lane.pending) {
            lane.pending = false;
            collect_now(lane, lane.pending_stream);
            lane.zeroed = false;
        }
        pybind11::dict d;
        d["records"] = lane.records;
        d["ctas"] = lane.used;
        d["warps"] = kTimerWarps;
        d["cap"] = kTimerCap;
        return d;
    }

  private:
    KernelTimerPool() : lazy_(lazy_requested()) {}

    static bool lazy_requested() {
        const char* env = std::getenv("INFERROCK_TIMER_LAZY");
        return env != nullptr && env[0] != '\0' && env[0] != '0';
    }

    // Lazy staging (INFERROCK_TIMER_LAZY=1). The per-launch copy-back is a
    // stream sync plus ~10MB of D2H, which serializes consecutive launches and
    // so hides exactly the back-to-back behaviour the coarse marks are here to
    // observe. With the env set, collect() only remembers the stream and the
    // copy happens on the first export_records() call, after the measurement
    // loop.
    static void collect_now(TimerLane& lane, cudaStream_t stream) {
        cudaStreamSynchronize(stream);
        const int64_t words = kTimerWords(lane.used);
        lane.records = torch::empty(
            {words},
            torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
        cudaMemcpy(lane.records.data_ptr(), lane.buffer,
                   static_cast<size_t>(words) * 8, cudaMemcpyDeviceToHost);
    }

    // Lanes are heap-allocated so a reference stays valid while the vector
    // grows; the mutex only guards the lookup, not the CUDA calls.
    TimerLane& get(const int lane_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (static_cast<size_t>(lane_id) >= lanes_.size()) {
            lanes_.resize(lane_id + 1);
        }
        if (lanes_[lane_id] == nullptr) {
            lanes_[lane_id] = std::make_unique<TimerLane>();
        }
        return *lanes_[lane_id];
    }

    // The buffers are deliberately not freed at exit: the pool outlives the
    // process's CUDA teardown ordering, and the OS reclaims them.
    std::vector<std::unique_ptr<TimerLane>> lanes_;
    std::mutex mutex_;
    bool lazy_ = false;
};

}  // namespace

void TimerStage(const int num_ctas, cudaStream_t stream) {
    KernelTimerPool::instance().stage(0, num_ctas, stream);
}

void TimerStage(const int lane, const int num_ctas, cudaStream_t stream) {
    KernelTimerPool::instance().stage(lane, num_ctas, stream);
}

uint64_t* TimerDeviceBuffer() {
    return KernelTimerPool::instance().device_buffer(0);
}

uint64_t* TimerDeviceBuffer(const int lane) {
    return KernelTimerPool::instance().device_buffer(lane);
}

void TimerCollect(cudaStream_t stream) {
    KernelTimerPool::instance().collect(0, stream);
}

void TimerCollect(const int lane, cudaStream_t stream) {
    KernelTimerPool::instance().collect(lane, stream);
}

pybind11::dict GetKernelTimer(const int lane) {
    return KernelTimerPool::instance().export_records(lane);
}

#else

// The binding in `cpp_extensions.cc` is unconditional; a timer-less build
// exports the geometry only, with no records.
pybind11::dict GetKernelTimer(const int) { return pybind11::dict(); }

#endif  // HALOPS_KERNEL_TIMER_ENABLED

}  // namespace halops
