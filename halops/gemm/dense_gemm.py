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
from halops.halops_kernel import EpilogueOp

__all__ = ["dense_gemm", "interleave_gate_up"]

_GLU_INTERLEAVE = 16


def interleave_gate_up(t: torch.Tensor) -> torch.Tensor:
    """Reorder ``[gate; up]`` rows for ``EpilogueOp.SituQuant``.

    Args:
        t: [2 * I, ...] weight, logical (unswizzled) weight scale or bias of a
            gate/up projection, gate rows first.

    Returns:
        [2 * I, ...] tensor with gate and up rows interleaved in blocks of 16:
        gate rows ``16i..16i+15``, then up rows ``16i..16i+15``.
    """
    gate, up = t.view(2, -1, _GLU_INTERLEAVE, *t.shape[1:]).unbind(0)
    return torch.stack([gate, up], 1).flatten(0, 2)


def dense_gemm(
    input: torch.Tensor,
    weight: torch.Tensor,
    sfa: Optional[torch.Tensor],
    sfb: Optional[torch.Tensor],
    output: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    alpha: Optional[torch.Tensor] = None,
    sm_count: int = -1,
    mode: str = "w8a8-mxfp8",
    use_pdl: bool = True,
    epilogue: EpilogueOp = EpilogueOp.Identity,
    output_sf: Optional[torch.Tensor] = None,
    situ_beta: float = 4.0,
    situ_linear_beta: Optional[float] = 25.0,
) -> None:
    """Dense GEMM on Blackwell SM100 tensor cores.

    Computes ``output = input @ weight^T + bias``. Supports FP8 (W8A8-MXFP8) and
    FP4 (W4A4) quantized GEMM via CUTLASS, where the FP4 recipe is selected
    explicitly by ``mode``.

    Args:
        input: [M, K] CUDA tensor.

            - ``w8a8-mxfp8``: dtype float8_e4m3fn.
            - ``w4a4-*``: dtype uint8 (packed FP4, physical shape [M, K // 2]).

        weight: [N, K] CUDA tensor. Same dtype and packing rules as input.
        sfa: Scale factor for A, dtype uint8, CUTLASS SfAtom 128x4 swizzled
            layout.  The layout covers whole 128-row atoms, so the shape is
            ``[ceil(M / 128) * 128, K // 32]`` (w8a8-mxfp8, w4a4-mxfp4) or
            ``[ceil(M / 128) * 128, K // 16]`` (w4a4-nvfp4) -- exactly what
            ``group_wise_quant`` returns.

            - ``w8a8-mxfp8``, ``w4a4-mxfp4``: UE8M0 encoding.
            - ``w4a4-nvfp4``: UE4M3 encoding.

        sfb: Scale factor for B, dtype uint8, in the same ordinary SfAtom
            layout as sfa. Shape [N, K // 32] (w8a8-mxfp8, w4a4-mxfp4) or
            [N, K // 16] (w4a4-nvfp4).
        output: [M, N] pre-allocated output tensor, dtype bfloat16. Written in-place.
        bias: Optional [N] bfloat16 bias vector. Only supported in w8a8-mxfp8 mode.
        alpha: Optional float32 scaling factor with shape [1]. Required for
            ``w4a4-nvfp4``; not accepted for ``w4a4-mxfp4``.
        sm_count: Number of SMs to use (-1 for all available).
        mode: ``"w8a8-mxfp8"``, ``"w4a4-mxfp4"`` or ``"w4a4-nvfp4"`` for
            quantized GEMM, or ``"bf16"`` for native BF16 GEMM (no
            quantization; pass ``sfa=None, sfb=None``).
        use_pdl: Enable Programmatic Dependent Launch (on by default).
        epilogue: ``EpilogueOp.Identity``, ``EpilogueOp.Sigmoid`` (bfloat16
            ``sigmoid(input @ weight^T + bias)``) or ``EpilogueOp.SituQuant``
            (``w8a8-mxfp8`` SiTU-GLU: with ``y = input @ weight^T + bias``
            and the ``[gate; up]`` rows of weight, sfb and bias reordered by
            ``interleave_gate_up``, output ``[M, N / 2]`` float8_e4m3fn is
            ``situ_beta * tanh(gate / situ_beta) * sigmoid(gate) *
            situ_linear_beta * tanh(up / situ_linear_beta)`` (``up`` as is
            when ``situ_linear_beta`` is None) with UE8M0 scales per 32
            columns in ``output_sf``).
        output_sf: SfAtom scales of a quantized output.
        situ_beta: SiTU gate softcap.
        situ_linear_beta: SiTU up softcap, or None for none.

    Returns:
        None. The result is written in-place to output.

    Example:
        >>> from halops import dense_gemm, group_wise_quant
        >>>
        >>> m = n = k = 256
        >>> x_bf16 = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
        >>> w_bf16 = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
        >>> x, sfa = group_wise_quant(x_bf16, mode="w8a8-mxfp8")
        >>> w, sfb = group_wise_quant(w_bf16, mode="w8a8-mxfp8")
        >>> out = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
        >>> dense_gemm(x, w, sfa, sfb, out, mode="w8a8-mxfp8")
    """
    if sfa is None:
        sfa = torch.empty(0, dtype=torch.uint8, device=input.device)
    if sfb is None:
        sfb = torch.empty(0, dtype=torch.uint8, device=input.device)

    halops_kernel.dense_gemm(
        input,
        weight,
        sfa,
        sfb,
        output,
        bias,
        alpha,
        sm_count,
        mode,
        use_pdl,
        epilogue,
        output_sf,
        situ_beta,
        situ_linear_beta,
    )
