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

__all__ = ["group_wise_quant"]


def group_wise_quant(
    input: torch.Tensor,
    is_fp4: bool = False,
    use_swizzle: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Group-wise quantization from BF16 to FP8/FP4 with scale factors.

    Quantizes a 2D BF16 tensor per group (block_size=32 for FP8, block_size=16
    for FP4), producing both the quantized data and the scale factor tensor.

    Args:
        input: [m, k] CUDA tensor, dtype bfloat16. Source tensor.
        is_fp4: ``True`` for FP4 E2M1 output (group_size=16);
            ``False`` for FP8 E4M3 output (group_size=32).
        use_swizzle: ``True`` to output scale in CUTLASS SfAtom 128x4 interleaved
            layout; ``False`` for linear row-major scale layout.
    Returns:
        A tuple ``(output_quant, output_sf)`` where:
        - ``output_quant``:
            FP8: [m, k] float8_e4m3fn
            FP4: [m, k/2] uint8 (packed nibbles, even index = low nibble)
        - ``output_sf``: [ceil(m/128)*128, k/group_size] uint8
            FP8: UE8M0 encoding (power-of-2 biased exponent)
            FP4: UE4M3 encoding (fp8_e4m3fn bit pattern as unsigned byte)

    Example:
        >>> from halops import group_wise_quant
        >>>
        >>> m, k = 256, 512
        >>> x_bf16 = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
        >>> # FP8 (W8A8) quantization
        >>> x_fp8, sf = group_wise_quant(x_bf16, is_fp4=False)
        >>> # FP4 (W4A4) quantization
        >>> x_fp4, sf = group_wise_quant(x_bf16, is_fp4=True)
    """
    return halops_kernel.group_wise_quant(input, is_fp4, use_swizzle)
