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

#include <cstdlib>

#include "common/kernel_timer_host.hpp"

namespace halops {

#if defined(HALOPS_KERNEL_TIMER_ENABLED)
namespace {
uint64_t* g_timer_buf = nullptr;
int g_timer_ctas = 0;  // high-water mark: what g_timer_buf holds
cudaStream_t g_timer_pending = nullptr;
// The legacy default stream is 0, which is also a null pointer, so the
// stream alone cannot double as the "a pull is pending" flag.
bool g_timer_pending_valid = false;
bool g_timer_zeroed = false;
int g_timer_used = 0;  // the last staged launch's grid
torch::Tensor g_timer_records;
}  // namespace

bool TimerLazyRequested() {
    static const bool lazy = [] {
        const char* env = std::getenv("INFERROCK_TIMER_LAZY");
        return env != nullptr && env[0] != '\0' && env[0] != '0';
    }();
    return lazy;
}

void TimerStage(const int num_ctas, cudaStream_t stream) {
    // Lazy reuse is valid only for repeated launches with the same grid: a
    // different grid writes a different set of slots.
    if (g_timer_used != num_ctas) {
        g_timer_zeroed = false;
    }
    if (g_timer_ctas < num_ctas) {
        if (g_timer_buf) {
            cudaFree(g_timer_buf);
        }
        cudaMalloc(&g_timer_buf, (size_t)kTimerWords(num_ctas) * 8);
        g_timer_ctas = num_ctas;
        g_timer_zeroed = false;
    }
    // Zero and later copy back only the CTAs of this launch. The allocation is
    // kept at its high-water mark, but a launch smaller than the one before it
    // must not read the previous launch's records as its own.
    g_timer_used = num_ctas;
    // Lazy staging: one zeroing per measurement burst. Every launch of a
    // given shape writes the same slots, so a pool zeroed before the burst
    // and re-zeroed after the next pull is as clean as a per-launch memset
    // -- and does not put 10MB of DRAM writes between two kernels.
    if (!TimerLazyRequested() || !g_timer_zeroed) {
        cudaMemsetAsync(g_timer_buf, 0, (size_t)kTimerWords(g_timer_used) * 8,
                        stream);
        g_timer_zeroed = true;
    }
}

uint64_t* TimerDeviceBuffer() { return g_timer_buf; }

// Lazy staging (INFERROCK_TIMER_LAZY=1). The per-launch copy-back is a stream
// sync plus ~10MB of D2H, which serializes consecutive launches and so hides
// exactly the back-to-back behaviour the coarse marks are here to observe.
// With the env set, TimerCollect only remembers the stream and the copy
// happens on the first GetKernelTimer() call, after the measurement loop.
static void TimerCollectNow(cudaStream_t stream) {
    cudaStreamSynchronize(stream);
    const int64_t words = kTimerWords(g_timer_used);
    auto t = torch::empty(
        {words},
        torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
    cudaMemcpy(t.data_ptr(), g_timer_buf, (size_t)words * 8,
               cudaMemcpyDeviceToHost);
    g_timer_records = t;
}

void TimerCollect(cudaStream_t stream) {
    if (TimerLazyRequested()) {
        g_timer_pending = stream;
        g_timer_pending_valid = true;
        return;
    }
    TimerCollectNow(stream);
}

pybind11::dict GetKernelTimer() {
    if (g_timer_pending_valid) {
        cudaStream_t s = g_timer_pending;
        g_timer_pending_valid = false;
        TimerCollectNow(s);
        g_timer_zeroed = false;
    }
    pybind11::dict d;
    d["records"] = g_timer_records;
    d["ctas"] = g_timer_used;
    d["warps"] = kTimerWarps;
    d["cap"] = kTimerCap;
    return d;
}

#else

// The binding in `cpp_extensions.cc` is unconditional; a timer-less build
// exports the geometry only, with no records.
pybind11::dict GetKernelTimer() { return pybind11::dict(); }

#endif  // HALOPS_KERNEL_TIMER_ENABLED

}  // namespace halops
