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

#include "kernel/mla/common/arch.cuh"
#include "kernel/mla/common/base_traits.cuh"
#include "kernel/mla/common/intrinsics.cuh"
#include "kernel/mla/common/layouts.cuh"
#include "kernel/mla/common/model.cuh"
#include "kernel/mla/common/tile_traits.cuh"
