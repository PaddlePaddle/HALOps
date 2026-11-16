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

__all__ = ["sigmoid_gate_mul_group_quant"]


def sigmoid_gate_mul_group_quant(
    o: torch.Tensor,
    gate: torch.Tensor,
    use_pdl: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sigmoid gating + MXFP8 group-32 quantization in a single pass.

    Computes ``y = o * sigmoid(gate)`` with ``sigmoid(gate)`` and the product
    each rounded to bf16, then quantizes every 32 consecutive elements of a
    row to float8_e4m3fn with a UE8M0 scale ``2^ceil(log2(amax / 448))``.

    Args:
        o: [T, hidden] bfloat16 tensor with contiguous, 16-byte aligned rows;
            ``hidden`` is a multiple of 32 and the row stride may be larger
            than ``hidden``.
        gate: [T, hidden] bfloat16 tensor laid out like ``o``.
        use_pdl: Enable Programmatic Dependent Launch (on by default).

    Returns:
        A tuple ``(q, sf)`` where:

        - ``q``: [T, hidden] float8_e4m3fn codes.
        - ``sf``: [T, hidden / 32] uint8 UE8M0 scales in row-major layout.

    Example:
        >>> from halops import sigmoid_gate_mul_group_quant
        >>>
        >>> t, hidden = 4, 4096
        >>> o = torch.randn(t, hidden, dtype=torch.bfloat16, device="cuda")
        >>> qg = torch.randn(t, 2 * hidden, dtype=torch.bfloat16, device="cuda")
        >>> q, sf = sigmoid_gate_mul_group_quant(o, qg[:, hidden:])
    """
    q, sf = halops_kernel.sigmoid_gate_mul_group_quant(o, gate, use_pdl)
    return q, sf
