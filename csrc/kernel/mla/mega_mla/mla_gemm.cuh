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

#include <cutlass/arch/barrier.h>
#include <cutlass/numeric_types.h>

#include <cute/tensor.hpp>

#include "kernel/gemm/kernel_traits.cuh"
#include "kernel/gemm/mainloop_fwd.cuh"
#include "kernel/mla/common/intrinsics.cuh"
#include "kernel/mla/common/traits.cuh"
#include "kernel/mla/mega_mla/gemm/atoms.cuh"
#include "kernel/mla/mega_mla/gemm/bf16.cuh"
#include "kernel/mla/mega_mla/gemm/bf16_ops.cuh"
#include "kernel/mla/mega_mla/gemm/config.cuh"
#include "kernel/mla/mega_mla/gemm/geom.cuh"
#include "kernel/mla/mega_mla/gemm/mma_drain.cuh"
#include "kernel/mla/mega_mla/gemm/pipeline.cuh"
#include "kernel/mla/mega_mla/gemm/tma_ops.cuh"
#include "kernel/mla/mega_mla/mla_utils.cuh"
