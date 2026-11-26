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

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/bfloat16.h>

#include <cute/config.hpp>
#include <cute/tensor.hpp>

#include "common/pdl.cuh"
#include "common/ptx.cuh"
#include "kernel/mla/common/atomics.cuh"
#include "kernel/mla/common/fp_math.cuh"
#include "kernel/mla/common/ld_st.cuh"
#include "kernel/mla/common/tma.cuh"
#include "kernel/mla/common/tmem.cuh"
#include "kernel/mla/common/umma.cuh"
#include "kernel/mla/common/umma_traits.cuh"
