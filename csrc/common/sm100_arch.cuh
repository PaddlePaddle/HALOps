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

#include <cuda_bf16.h>

namespace halops {

inline constexpr int kWarp = 32;
inline constexpr int kWarpGroup = 128;
inline constexpr int kWarpsPerGroup = kWarpGroup / kWarp;

inline constexpr int kUmmaK = 16;
inline constexpr int kUmmaKBytes = kUmmaK * sizeof(__nv_bfloat16);

inline constexpr int kTmemCols = 512;
inline constexpr int kTmemLd = 32;

inline constexpr int kSmemCapacity = 232448;

inline constexpr int kChunk = 16;
inline constexpr int kElemsPerChunk = kChunk / sizeof(__nv_bfloat16);

inline constexpr int kTmaGatherRows = 4;

inline constexpr int kSw128AtomRows = 8;
inline constexpr int kSw128AtomBytes = 1024;
inline constexpr int kSw128AtomRowBytes = kSw128AtomBytes / kSw128AtomRows;

inline constexpr int kBf16MantissaBits = 7;

}  // namespace halops
