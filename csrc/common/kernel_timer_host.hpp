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

#include "common/kernel_timer_types.hpp"

// Common host side of the kernel event timer: allocate the record pool, zero
// it per launch, copy it back, hand it to python.
//
// There is no event vocabulary to carry any more -- a probe is identified by
// its own file:line and halops/moe/device_timer.py resolves that against
// the kernel sources -- so this layer is kernel-independent.
//
// NOTE: this torch/pybind implementation is unused in this repo. The MoE
// module is TVM-FFI and cannot link against torch, so it carries the same
// allocate/zero/copy logic in `moe_timer_host` (moe_api.cuh). Kept for
// a pybind kernel that wants the common path.
namespace halops {

#if defined(HALOPS_KERNEL_TIMER_ENABLED)

// (Re)allocate the device record pool for num_ctas CTAs and zero it on the
// stream. Only the CTAs of this launch are zeroed and copied back; the pool
// keeps its high-water-mark allocation across launches.
void TimerStage(const int num_ctas, cudaStream_t stream);
uint64_t *TimerDeviceBuffer();

// Synchronize and copy the records back to a host tensor.
void TimerCollect(cudaStream_t stream);

#else

inline void TimerStage(const int, cudaStream_t) {}
inline uint64_t *TimerDeviceBuffer() { return nullptr; }
inline void TimerCollect(cudaStream_t) {}

#endif

// The export is bound unconditionally by `cpp_extensions.cc`, so it is
// defined (as an empty dict) in a timer-less build too.
//
// Python-facing export: records plus the pool geometry device_timer.py needs
// to reshape them as [cta][warp][slot][word].
pybind11::dict GetKernelTimer();

}  // namespace halops
