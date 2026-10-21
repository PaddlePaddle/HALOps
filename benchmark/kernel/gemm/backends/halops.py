"""
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

from __future__ import annotations
from halops import dense_gemm
from halops import group_masked_gemm
from halops import group_contiguous_gemm


def run_dense_gemm(inp: dict):
    """Run halops dense GEMM (W8A8/FP8 or W4A4/FP4).

    inp keys:
      x        [M, K]  (w8a8) float8_e4m3fn  /  [M, K//2] (w4a4-nvfp4) uint8 packed
      w        [N, K]  (w8a8) float8_e4m3fn  /  [N, K//2] (w4a4-nvfp4) uint8 packed
      sfa      [M, K//32] (w8a8) / [M, K//16] (w4a4-nvfp4)  uint8 (CUTLASS layout)
      sfb      [N, K//32] (w8a8) / [N, K//16] (w4a4-nvfp4)  uint8
      out      [M, N]  bfloat16, pre-allocated output
      alpha    [1] float32 (w4a4-nvfp4) / None (w8a8)
      mode     "w8a8" | "w4a4-nvfp4"
      sm_count int
    """
    dense_gemm(
        inp["x"],
        inp["w"],
        inp["sfa"],
        inp["sfb"],
        inp["out"],
        None,  # bias
        inp.get("alpha"),  # w4a4-nvfp4 requires alpha; w8a8 passes None
        inp["sm_count"],
        inp.get("mode", "w8a8"),
    )
    return inp["out"]


def run_group_masked_gemm(inp: dict):
    """Run halops grouped masked GEMM (W8A8/FP8 or W4A4/FP4).

    inp keys:
      x        [E*max_m, K] (w8a8) float8_e4m3fn / [E*max_m, K//2] (w4a4-nvfp4) uint8
      w        [E, N, K]    (w8a8) float8_e4m3fn / [E, N, K//2]    (w4a4-nvfp4) uint8
      sfa      [E*max_m, K//32] (w8a8) / [E*max_m, K//16] (w4a4-nvfp4)  uint8
      sfb      [E, N, K//32]    (w8a8) / [E, N, K//16]    (w4a4-nvfp4)  uint8
      masked_m [E]             int32 (active rows per expert)
      out      [E*max_m, N]    bfloat16, pre-allocated
      alpha    [E] float32 (w4a4-nvfp4) / None (w8a8)
      max_m    int             max rows per expert
      mode     "w8a8" | "w4a4-nvfp4"
      sm_count int
    """
    group_masked_gemm(
        inp["x"],
        inp["w"],
        inp["sfa"],
        inp["sfb"],
        inp["masked_m"],
        inp["out"],
        None,  # bias
        inp.get("alpha"),  # w4a4-nvfp4 requires alpha; w8a8 passes None
        inp["max_m"],
        inp["sm_count"],
        inp.get("mode", "w8a8"),
    )
    return inp["out"]


def run_group_contiguous_gemm(inp: dict):
    """Run halops grouped contiguous GEMM (W8A8/FP8 or W4A4/FP4).

    inp keys:
      x          [M, K]        (w8a8) float8_e4m3fn / [M, K//2]    (w4a4-nvfp4) uint8
      w          [E, N, K]     (w8a8) float8_e4m3fn / [E, N, K//2] (w4a4-nvfp4) uint8
      sfa        [M, K//32]    (w8a8) / [M, K//16]    (w4a4-nvfp4)  uint8
      sfb        [E, N, K//32] (w8a8) / [E, N, K//16] (w4a4-nvfp4)  uint8
      expert_ids [M]           int32 (per-row expert assignment)
      out        [M, N]        bfloat16, pre-allocated
      alpha      [E] float32 (w4a4-nvfp4) / None (w8a8)
      mode       "w8a8" | "w4a4-nvfp4"
      sm_count   int
    """
    group_contiguous_gemm(
        inp["x"],
        inp["w"],
        inp["sfa"],
        inp["sfb"],
        inp["expert_ids"],
        inp["out"],
        None,  # bias
        inp.get("alpha"),  # w4a4-nvfp4 requires alpha; w8a8 passes None
        inp["sm_count"],
        inp.get("mode", "w8a8"),
    )
    return inp["out"]
