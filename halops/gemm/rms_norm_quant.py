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

__all__ = ["rms_norm_quant"]


def rms_norm_quant(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    use_swizzle: bool = True,
    use_pdl: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """RMSNorm fused with W8A8-MXFP8 quantization.

    Computes ``y = bf16(x * rsqrt(mean(x^2) + eps) * weight)`` in fp32 and
    quantizes it exactly as ``group_wise_quant(y, "w8a8-mxfp8", use_swizzle)``
    does, reading ``x`` once.

    Args:
        input: [M, K] bfloat16 CUDA tensor; rows may be strided
            (``stride(1) == 1``, 16-byte aligned row starts). K must be a
            multiple of 128 and at most 8192.
        weight: [K] bfloat16 RMSNorm gamma.
        eps: RMSNorm epsilon.
        use_swizzle: Write the scales in the CUTLASS SfAtom layout (default),
            or row-major as ``low_latency_mma_gemm`` reads them.
        use_pdl: Enable Programmatic Dependent Launch (on by default).

    Returns:
        ``(q, sf)``:

        - ``q``: [M, K] float8_e4m3fn
        - ``sf``: uint8 UE8M0 scales, [ceil(M / 128) * 128, K / 32] in
          the CUTLASS SfAtom layout (padding rows hold 0) with ``use_swizzle``,
          else row-major [M, K / 32]

    Example:
        >>> from halops import rms_norm_quant
        >>>
        >>> x = torch.randn(128, 1024, dtype=torch.bfloat16, device="cuda")
        >>> w = torch.ones(1024, dtype=torch.bfloat16, device="cuda")
        >>> x_fp8, sf = rms_norm_quant(x, w, 1e-6)
    """
    q, sf = halops_kernel.rms_norm_quant(input, weight, eps, use_swizzle, use_pdl)
    return q, sf
