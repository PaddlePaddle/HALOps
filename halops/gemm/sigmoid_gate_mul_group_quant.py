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
    """Sigmoid gating fused with W8A8-MXFP8 quantization.

    Computes ``y = o * torch.sigmoid(gate)`` on the bfloat16 inputs and
    quantizes it exactly as ``group_wise_quant(y, use_swizzle=False)`` does.

    Args:
        o: [T, hidden] bfloat16 tensor; rows may be strided (``stride(1) ==
            1``, 16-byte aligned row starts). hidden must be a multiple of 32.
        gate: [T, hidden] bfloat16 tensor, same layout rules as ``o``.
        use_pdl: Enable Programmatic Dependent Launch (on by default).

    Returns:
        ``(q, sf)``: [T, hidden] float8_e4m3fn codes and [T, hidden / 32]
        uint8 UE8M0 scales in row-major layout.

    Example:
        >>> from halops import sigmoid_gate_mul_group_quant
        >>>
        >>> o = torch.randn(4, 4096, dtype=torch.bfloat16, device="cuda")
        >>> gate = torch.randn(4, 4096, dtype=torch.bfloat16, device="cuda")
        >>> q, sf = sigmoid_gate_mul_group_quant(o, gate)
    """
    q, sf = halops_kernel.sigmoid_gate_mul_group_quant(o, gate, use_pdl)
    return q, sf
