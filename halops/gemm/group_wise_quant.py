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
    mode: str = "w8a8-mxfp8",
    use_swizzle: bool = True,
    use_pdl: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Group-wise quantization from BF16/FP16 to FP8/FP4 with scale factors.

    Quantizes a 2D BF16/FP16 tensor per group along K, producing both the
    quantized data and the block-scale tensor. Supports FP8 (W8A8-MXFP8) and
    FP4 (W4A4) quantization. The FP4 recipe is selected explicitly by
    ``mode``.

    The output pairs feed ``dense_gemm`` / ``batch_gemm`` /
    ``group_masked_gemm`` / ``group_contiguous_gemm`` of the same ``mode``.

    Args:
        input: [m, k] CUDA tensor, dtype bfloat16 or float16. Source tensor.
        mode: ``"w8a8-mxfp8"``, ``"w4a4-mxfp4"`` or ``"w4a4-nvfp4"``. Selects both
            the value encoding and the scale recipe.
        use_swizzle: ``True`` to output scales in the CUTLASS SfAtom 128x4
            interleaved layout consumed by the GEMM kernels; ``False`` for a
            plain row-major scale layout.
        use_pdl: Enable Programmatic Dependent Launch (on by default).

    Returns:
        A tuple ``(output_quant, output_sf)`` where:
        - ``output_quant``:
            FP8: [m, k] float8_e4m3fn
            FP4: [m, k/2] uint8 (packed nibbles, even index = low nibble)
        - ``output_sf``: [ceil(m/128)*128, k / group_size] uint8
            w8a8-mxfp8, w4a4-mxfp4: UE8M0 encoding (power-of-2 biased exponent)
            w4a4-nvfp4: UE4M3 encoding (fp8_e4m3fn bit pattern as unsigned byte)

    Example:
        >>> from halops import group_wise_quant
        >>>
        >>> m, k = 256, 512
        >>> x_bf16 = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
        >>> # FP8 (W8A8-MXFP8) quantization
        >>> x_fp8, sf = group_wise_quant(x_bf16, mode="w8a8-mxfp8")
        >>> # FP4 with UE8M0 scales, one per 32 elements (MXFP4)
        >>> x_mxfp4, sf = group_wise_quant(x_bf16, mode="w4a4-mxfp4")
        >>> # FP4 with UE4M3 scales, one per 16 elements (NVFP4)
        >>> x_nvfp4, sf = group_wise_quant(x_bf16, mode="w4a4-nvfp4")
    """
    return halops_kernel.group_wise_quant(input, mode, use_swizzle, use_pdl)
