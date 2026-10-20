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

__all__ = ["group_wise_quant", "pack_sfb_224"]


def pack_sfb_224(sfb: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """Prepare weight scales for the dense W8A8 224-column tiles.

    Each logical 224-column tile is stored as two aligned 128-row atoms.  The
    resulting tensor is a one-time weight format with physical shape
    ``(ceil(n/224)*256, k//32)``. Keep the ordinary SFB as the primary
    ``dense_gemm`` scale and pass this packed copy through ``sfb_packed`` so
    dispatch can choose the best format for each M. Repack if scales change.

    ``sfb`` must come from W8A8 ``group_wise_quant(..., use_swizzle=True)``;
    row-major scales and FP4 scales are incompatible. The returned tensor has
    the same device and uint8 dtype, owns its storage, and is contiguous.
    """
    if sfb.dtype != torch.uint8 or not sfb.is_contiguous():
        raise ValueError("sfb must be contiguous uint8")
    if n <= 0 or k <= 0 or n % 128 or k % 128:
        raise ValueError("n and k must be positive and divisible by 128")
    if sfb.ndim != 2 or tuple(sfb.shape) != (n, k // 32):
        raise ValueError("sfb must have shape (n, k//32) in W8A8 SfAtom layout")
    k_tiles = k // 128
    n_tiles = (n + 223) // 224
    src = sfb.view(n // 128, k_tiles, 32, 4, 4)
    logical = src.permute(0, 3, 2, 1, 4).contiguous().view(n, k // 32)
    rows = (
        torch.arange(n_tiles, device=sfb.device)[:, None] * 224
        + torch.arange(256, device=sfb.device)[None, :]
    )
    tile = logical[rows.clamp_max(n - 1)].clone()
    tile.masked_fill_(rows[:, :, None] >= n, 0)
    return (
        tile.view(n_tiles, 2, 4, 32, k_tiles, 4)
        .permute(0, 1, 4, 3, 2, 5)
        .contiguous()
        .view(n_tiles * 256, k // 32)
    )


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
