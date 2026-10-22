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
import torch

# FP4 E2M1 magnitude lookup: index = 3-bit magnitude code, value = represented magnitude.
_FP4_E2M1_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def _ue8m0_to_scale(sf: torch.Tensor) -> torch.Tensor:
    """Decode a UE8M0 scale (uint8 biased exponent) to a float32 power-of-2 value."""
    return torch.pow(
        torch.tensor(2.0, dtype=torch.float32, device=sf.device),
        sf.to(torch.float32) - 127.0,
    )


def _ue4m3_to_scale(sf: torch.Tensor) -> torch.Tensor:
    """Decode a UE4M3 scale (float8_e4m3fn bit pattern stored as uint8) to float32."""
    return sf.view(torch.float8_e4m3fn).to(torch.float32)


def _dequant_to_bf16(src_bf16: torch.Tensor, mode: str) -> torch.Tensor:
    """Re-quantize a bf16 source to fp8/fp4 and dequantize it back to bf16.

    The kernels under test consume group-wise quantized fp8/fp4 operands, so
    their output already carries the quantization rounding error. Feeding the
    golden the *original* bf16 would make refcheck attribute that inherent
    quantization loss to the kernel. Instead we reproduce the exact quantized
    values the kernel sees -- quantization is deterministic and the quantized
    data is independent of the scale-factor swizzle layout -- and dequantize
    them, so the golden shares the same rounded inputs and only genuine compute
    error remains in the diff.

    ``use_swizzle=False`` returns the scale factors in linear [rows, k//group]
    layout, avoiding any CUTLASS SfAtom de-interleaving.
    """
    from halops.gemm.group_wise_quant import group_wise_quant

    if mode == "w8a8-mxfp8":
        group = 32
    elif mode == "w4a4-nvfp4":
        group = 16
    else:
        raise ValueError(
            f"golden backend does not support mode {mode!r}. "
            "Expected 'w8a8-mxfp8' or 'w4a4-nvfp4'."
        )
    is_nvfp4 = mode == "w4a4-nvfp4"
    rows, k = src_bf16.shape
    q, sf = group_wise_quant(src_bf16, mode=mode, use_swizzle=False)

    if is_nvfp4:
        # q: [rows, k//2] uint8, two 4-bit codes per byte (even idx = low nibble).
        codes = torch.empty(rows, k, dtype=torch.uint8, device=q.device)
        codes[:, 0::2] = q & 0x0F
        codes[:, 1::2] = (q >> 4) & 0x0F
        lut = torch.tensor(_FP4_E2M1_VALUES, dtype=torch.float32, device=q.device)
        sign = torch.where((codes & 0x8) > 0, -1.0, 1.0)
        vals = sign * lut[(codes & 0x7).long()]  # [rows, k] float32
        scale = _ue4m3_to_scale(sf)  # [rows, k//16]
    else:
        vals = q.to(torch.float32)  # [rows, k]
        scale = _ue8m0_to_scale(sf)  # [rows, k//32]

    deq = vals.reshape(rows, k // group, group) * scale.unsqueeze(-1)
    return deq.reshape(rows, k).to(torch.bfloat16)


def run_dense_golden(inp: dict) -> torch.Tensor:
    """Reference dense GEMM via torch.matmul on dequantized bf16 operands.

    The quantized kernels (fp8/fp4) approximate ``x @ w.T``. We dequantize the
    same quantized inputs the kernels see back to bf16 and matmul in bf16, so
    the golden is a neutral baseline that excludes quantization noise and shares
    the kernels' compute and output dtype.

    inp keys: x_bf16 [M, K], w_bf16 [N, K], mode ("w8a8-mxfp8" | "w4a4-nvfp4").
    """
    mode = inp.get("mode", "w8a8-mxfp8")
    x = _dequant_to_bf16(inp["x_bf16"], mode)
    w = _dequant_to_bf16(inp["w_bf16"], mode)
    return torch.matmul(x, w.t())


def run_grouped_golden(inp: dict) -> torch.Tensor:
    """Reference grouped GEMM via per-group torch.matmul on dequantized bf16.

    Each group g computes ``x[rows_g] @ w[g].T`` in bf16, where x and w are
    dequantized from the same fp8/fp4 quantization the kernels use (see
    ``_dequant_to_bf16``). Group row ranges come from ``m_indptr`` (prefix sum
    of per-group token counts), the same partitioning the kernels use.

    Args:
        inp: dict with keys
            x_bf16   [M, K]      unquantized activation source (bfloat16)
            w_bf16   [E, N, K]   unquantized per-expert weight source (bfloat16)
            m_indptr [E+1]       per-group row offsets (prefix sum)
            mode     str         "w8a8-mxfp8" (fp8) or "w4a4-nvfp4" (fp4)

    Returns:
        torch.Tensor [M, N] bfloat16 reference output.
    """
    mode = inp.get("mode", "w8a8-mxfp8")
    x = _dequant_to_bf16(inp["x_bf16"], mode)  # [M, K]
    w_src = inp["w_bf16"]  # [E, N, K]
    num_groups, n, k = w_src.shape
    w = _dequant_to_bf16(w_src.reshape(num_groups * n, k), mode).reshape(
        num_groups, n, k
    )
    # Materialize offsets on CPU once to avoid per-group GPU->CPU syncs.
    offsets = inp["m_indptr"].tolist()
    out = torch.empty(x.shape[0], n, dtype=torch.bfloat16, device=x.device)
    for g in range(num_groups):
        start, end = offsets[g], offsets[g + 1]
        if end <= start:
            continue
        out[start:end] = torch.matmul(x[start:end], w[g].t())
    return out
