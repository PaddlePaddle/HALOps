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

from typing import Tuple

import torch

import halops.halops_kernel as halops_kernel

__all__ = ["fused_swiglu_quant"]


def fused_swiglu_quant(
    input: torch.Tensor,
    token_nums_per_expert: torch.Tensor,
    is_fp4: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused SwiGLU activation + FP8/FP4 quantization in a single pass.

    Computes SwiGLU: ``y = up * silu(gate)`` where input = [gate | up]
    (concatenated along dim-1), then quantizes the result to FP8 or FP4.
    Outputs quantized data and CUTLASS SfAtom-formatted scale factors.

    Args:
        input: [E * M, 2 * hidden_size] bfloat16.
            First half of dim-1 is gate, second half is up projection.
        token_nums_per_expert: [E] int32. Valid token count per expert.
        is_fp4: ``True`` for FP4 E2M1 output (group_size=16);
            ``False`` for FP8 E4M3 output (group_size=32).

    Returns:
        A tuple ``(out_quant, out_scale)`` where:
        - ``out_quant``:
            FP4: [E*M, hidden_size/2] uint8 (packed nibbles)
            FP8: [E*M, hidden_size] float8_e4m3fn
        - ``out_scale``: [E*M, kb_dim] uint8, CUTLASS SfAtom 128x4 interleaved layout.
            FP8 kb_dim = ceil(hidden_size/128/4) * 16
            FP4 kb_dim = hidden_size / 16

    Example:
        >>> from halops import fused_swiglu_quant
        >>>
        >>> e, m, hidden_size = 4, 128, 2048
        >>> gate_up = torch.randn(e * m, 2 * hidden_size,
        ...                       dtype=torch.bfloat16, device="cuda")
        >>> token_nums_per_expert = torch.full((e,), m,
        ...                                    dtype=torch.int32, device="cuda")
        >>> out_quant, out_scale = fused_swiglu_quant(
        ...     gate_up, token_nums_per_expert, is_fp4=False)
    """
    return halops_kernel.fused_swiglu_quant(
        input,
        token_nums_per_expert,
        is_fp4,
    )
