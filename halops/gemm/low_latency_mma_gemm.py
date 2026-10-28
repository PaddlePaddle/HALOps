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

from typing import Optional

import torch

import halops.halops_kernel as halops_kernel

__all__ = ["low_latency_mma_gemm", "prefer_low_latency_mma"]


def low_latency_mma_gemm(
    input: torch.Tensor,
    weight: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    output: torch.Tensor,
    block_n: int = -1,
    warps_k: int = -1,
    stage_smem: int = -1,
    persist: int = -1,
    sm_count: int = -1,
    output2: Optional[torch.Tensor] = None,
    n_split: int = 0,
    input2: Optional[torch.Tensor] = None,
    sfa2: Optional[torch.Tensor] = None,
    use_pdl: bool = True,
) -> None:
    halops_kernel.low_latency_mma_gemm(
        input,
        weight,
        sfa,
        sfb,
        output,
        block_n,
        warps_k,
        stage_smem,
        persist,
        sm_count,
        output2,
        n_split,
        input2,
        sfa2,
        use_pdl,
    )


def prefer_low_latency_mma(m: int, k: int) -> bool:
    return m <= 16 and k % 128 == 0 and (k // 128) % 2 == 0
