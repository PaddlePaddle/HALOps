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
import importlib.util

import torch


def _check():
    """Gate on deep_gemm availability and required API.

    Raises ImportError when unavailable so the framework's per-backend
    ``except Exception`` path prints ``[ERROR]`` and skips deepgemm without
    affecting the other backends.
    """
    if importlib.util.find_spec("deep_gemm") is None:
        raise ImportError("deep_gemm not installed.")
    import deep_gemm as dg

    if not hasattr(dg, "fp8_gemm_nt"):
        raise ImportError("deep_gemm is missing fp8_gemm_nt API.")


# ---------------------------------------------------------------------------
# Input preparation helpers.
#
# These receive the SAME bf16 source tensors the framework built for the other
# backends (GemmInput.x_bf16 / w_bf16, GroupGemmInput.x_bf16 / w_bf16 and the
# existing expert layout) and re-quantize them with DeepGEMM's own convention
# (per-token / per-block / per-token-fp4 + UE8M0). Re-quantizing from the shared
# bf16 keeps the refcheck comparison fair, and matching the existing shapes (3D
# [E, max_m, K] for masked, reused expert_ids as m_indices for contiguous) keeps
# the flattened output numel aligned with the reference backend.
# ---------------------------------------------------------------------------


def make_dense_inputs(x_bf16: torch.Tensor, w_bf16: torch.Tensor, mode: str) -> dict:
    """Quantize shared dense bf16 inputs into DeepGEMM layout.

    x_bf16 [M, K], w_bf16 [N, K].
    w8a8: A/B=per-token FP8 gran_k=32.
    w4a4: A/B=per-token FP4 gran_k=32.

    Both scale factors are converted with `transform_sf_into_required_layout`
    here, outside the timed region: the GEMM call is then handed SFs already in
    DeepGEMM's required layout, so no scale-factor conversion kernel runs inside
    the measured launch.  gran_k=32 matches the group-32 scheme HALOps and
    FlashInfer use, so the three backends compare on the same quantization.
    """
    _check()
    import deep_gemm as dg

    m, k = x_bf16.shape
    n = w_bf16.shape[0]
    out = torch.empty(m, n, dtype=torch.bfloat16, device=x_bf16.device)

    cast = (
        dg.per_token_cast_to_fp4
        if mode.startswith("w4a4")
        else dg.per_token_cast_to_fp8
    )
    a_data, a_sf = cast(x_bf16, use_ue8m0=True, gran_k=32)
    b_data, b_sf = cast(w_bf16, use_ue8m0=True, gran_k=32)
    a_sf = dg.transform_sf_into_required_layout(a_sf, m, k, (1, 32))
    b_sf = dg.transform_sf_into_required_layout(b_sf, n, k, (1, 32))
    return {
        "a": (a_data, a_sf),
        "b": (b_data, b_sf),
        "out": out,
        "mode": mode,
    }


def make_grouped_masked_inputs(
    x_bf16: torch.Tensor,
    w_bf16: torch.Tensor,
    masked_m: torch.Tensor,
    max_m: int,
    mode: str,
) -> dict:
    """Quantize shared grouped bf16 inputs into DeepGEMM masked 3D layout.

    x_bf16 [E*max_m, K] -> reshaped to [E, max_m, K]; w_bf16 [E, N, K].
    w8a8: A=per-token FP8, B=per-block FP8.
    w4a4: A=per-token FP4 gran_k=32, B=per-token FP4 gran_k=32.
    Output is 3D [E, max_m, N]; flattened it matches the reference's
    [E*max_m, N] numel.

    Scale factors are pre-packed into UE8M0 INT format so that the timed
    kernel call skips the FP32->UE8M0 rshift conversion, reflecting the true
    inference cost where SFs are converted once at model load time.
    """
    _check()
    import deep_gemm as dg

    num_groups, n, k = w_bf16.shape
    dev = x_bf16.device
    a_bf16 = x_bf16.reshape(num_groups, max_m, k)
    masked_m = masked_m.to(device=dev, dtype=torch.int32)

    if mode.startswith("w4a4"):
        # A: FP4 per-token (gran_k=32), pre-pack SF to UE8M0 INT
        a_fp4 = torch.empty(num_groups, max_m, k // 2, dtype=torch.int8, device=dev)
        a_sf_list = []
        for i in range(num_groups):
            a_i, sf_i = dg.per_token_cast_to_fp4(a_bf16[i], use_ue8m0=True, gran_k=32)
            a_fp4[i] = a_i
            a_sf_list.append(sf_i)
        a_sf = torch.stack(a_sf_list)
        a_sf_packed = dg.get_mn_major_tma_aligned_packed_ue8m0_tensor(a_sf)
        # B: FP4 per-token (gran_k=32), pre-pack SF to UE8M0 INT
        b_fp4 = torch.empty(num_groups, n, k // 2, dtype=torch.int8, device=dev)
        b_sf_list = []
        for i in range(num_groups):
            b_i, sf_i = dg.per_token_cast_to_fp4(w_bf16[i], use_ue8m0=True, gran_k=32)
            b_fp4[i] = b_i
            b_sf_list.append(sf_i)
        b_sf = torch.stack(b_sf_list)
        b_sf_packed = dg.get_mn_major_tma_aligned_packed_ue8m0_tensor(b_sf)
        a, b = (a_fp4, a_sf_packed), (b_fp4, b_sf_packed)
    else:
        # A: FP8 per-token, pre-pack SF to UE8M0 INT (gran_mn=1, gran_k=128)
        a_fp8 = torch.empty(num_groups, max_m, k, dtype=torch.float8_e4m3fn, device=dev)
        a_sf_list = []
        for i in range(num_groups):
            a_i, sf_i = dg.per_token_cast_to_fp8(a_bf16[i], use_ue8m0=True)
            a_fp8[i] = a_i
            a_sf_list.append(sf_i)
        a_sf = torch.stack(a_sf_list)
        a_sf_packed = dg.transform_sf_into_required_layout(
            a_sf, max_m, k, (1, 128, 128), num_groups=num_groups, is_sfa=True
        )
        # B: FP8 per-block, pre-pack SF to UE8M0 INT (gran_mn=128, gran_k=128).
        b_fp8 = torch.empty(num_groups, n, k, dtype=torch.float8_e4m3fn, device=dev)
        b_sf_list = []
        for i in range(num_groups):
            b_i, sf_i = dg.per_block_cast_to_fp8(w_bf16[i], use_ue8m0=True)
            b_fp8[i] = b_i
            b_sf_list.append(sf_i)
        b_sf = torch.stack(b_sf_list)
        b_sf_packed = dg.transform_sf_into_required_layout(
            b_sf, n, k, (1, 128, 128), num_groups=num_groups, is_sfa=False
        )
        a, b = (a_fp8, a_sf_packed), (b_fp8, b_sf_packed)

    out = torch.empty(num_groups, max_m, n, dtype=torch.bfloat16, device=dev)
    return {
        "a": a,
        "b": b,
        "out": out,
        "masked_m": masked_m,
        "max_m": max_m,
        "mode": mode,
    }


def make_grouped_contiguous_inputs(
    x_bf16: torch.Tensor,
    w_bf16: torch.Tensor,
    m_indices: torch.Tensor,
    mode: str,
) -> dict:
    """Quantize shared grouped bf16 inputs into DeepGEMM contiguous layout.

    x_bf16 [M, K]; w_bf16 [E, N, K]. m_indices reuses the reference's
    expert_ids (already tile-aligned to a multiple of 128, satisfying
    DeepGEMM's contiguous alignment requirement). Output is 2D [M, N].
    w8a8: A=per-token FP8, B=per-block FP8.
    w4a4: A=per-token FP4 gran_k=32, B=per-token FP4 gran_k=32.

    Scale factors are pre-packed into UE8M0 INT format so that the timed
    kernel call skips the FP32->UE8M0 rshift conversion, reflecting the true
    inference cost where SFs are converted once at model load time.
    """
    _check()
    import deep_gemm as dg

    num_groups, n, k = w_bf16.shape
    m = x_bf16.shape[0]
    dev = x_bf16.device
    m_indices = m_indices.to(device=dev, dtype=torch.int32)

    if mode.startswith("w4a4"):
        a_fp4, a_sf = dg.per_token_cast_to_fp4(x_bf16, use_ue8m0=True, gran_k=32)
        a_sf_packed = dg.get_mn_major_tma_aligned_packed_ue8m0_tensor(a_sf)
        b_fp4 = torch.empty(num_groups, n, k // 2, dtype=torch.int8, device=dev)
        b_sf_list = []
        for i in range(num_groups):
            b_i, sf_i = dg.per_token_cast_to_fp4(w_bf16[i], use_ue8m0=True, gran_k=32)
            b_fp4[i] = b_i
            b_sf_list.append(sf_i)
        b_sf = torch.stack(b_sf_list)
        b_sf_packed = dg.get_mn_major_tma_aligned_packed_ue8m0_tensor(b_sf)
        a, b = (a_fp4, a_sf_packed), (b_fp4, b_sf_packed)
    else:
        # A: FP8 per-token, pre-pack SF to UE8M0 INT (gran_mn=1, gran_k=128)
        a_fp8, a_sf = dg.per_token_cast_to_fp8(x_bf16, use_ue8m0=True)
        a_sf_packed = dg.transform_sf_into_required_layout(a_sf, m, k, (1, 128))
        # B: FP8 per-block, pre-pack SF to UE8M0 INT (gran_mn=128, gran_k=128).
        b_fp8 = torch.empty(num_groups, n, k, dtype=torch.float8_e4m3fn, device=dev)
        b_sf_list = []
        for i in range(num_groups):
            b_i, sf_i = dg.per_block_cast_to_fp8(w_bf16[i], use_ue8m0=True)
            b_fp8[i] = b_i
            b_sf_list.append(sf_i)
        b_sf = torch.stack(b_sf_list)
        b_sf_packed = dg.transform_sf_into_required_layout(
            b_sf, n, k, (1, 128, 128), num_groups=num_groups, is_sfa=False
        )
        a, b = (a_fp8, a_sf_packed), (b_fp8, b_sf_packed)

    out = torch.empty(m, n, dtype=torch.bfloat16, device=dev)
    return {
        "a": a,
        "b": b,
        "out": out,
        "m_indices": m_indices,
        "mode": mode,
    }


# ---------------------------------------------------------------------------
# Run functions (signature: inp: dict -> Tensor), matching halops/flashinfer.
# ---------------------------------------------------------------------------


def run_dense_gemm(inp: dict):
    """Run DeepGEMM dense GEMM (w8a8 fp8_gemm_nt / w4a4 fp8_fp4_gemm_nt)."""
    _check()
    import deep_gemm as dg

    if inp.get("mode", "w8a8").startswith("w4a4"):
        dg.fp8_fp4_gemm_nt(
            inp["a"], inp["b"], inp["out"], recipe_a=(1, 32), recipe_b=(1, 32)
        )
    else:
        dg.fp8_gemm_nt(
            inp["a"], inp["b"], inp["out"], recipe_a=(1, 32), recipe_b=(1, 32)
        )
    return inp["out"]


def run_group_masked_gemm(inp: dict):
    """Run DeepGEMM grouped masked GEMM (w8a8 / w4a4)."""
    _check()
    import deep_gemm as dg

    if inp.get("mode", "w8a8").startswith("w4a4"):
        dg.m_grouped_fp8_fp4_gemm_nt_masked(
            inp["a"],
            inp["b"],
            inp["out"],
            inp["masked_m"],
            inp["max_m"],
            recipe_a=(1, 32),
            recipe_b=(1, 32),
        )
    else:
        dg.m_grouped_fp8_gemm_nt_masked(
            inp["a"], inp["b"], inp["out"], inp["masked_m"], inp["max_m"]
        )
    return inp["out"]


def run_group_contiguous_gemm(inp: dict):
    """Run DeepGEMM grouped contiguous GEMM (w8a8 / w4a4)."""
    _check()
    import deep_gemm as dg

    if inp.get("mode", "w8a8").startswith("w4a4"):
        dg.m_grouped_fp8_fp4_gemm_nt_contiguous(
            inp["a"],
            inp["b"],
            inp["out"],
            inp["m_indices"],
            recipe_a=(1, 32),
            recipe_b=(1, 32),
        )
    else:
        dg.m_grouped_fp8_gemm_nt_contiguous(
            inp["a"], inp["b"], inp["out"], inp["m_indices"]
        )
    return inp["out"]
