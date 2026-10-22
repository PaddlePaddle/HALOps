# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for group_masked_gemm (W8A8-MXFP8 FP8, W4A4 NVFP4 and W4A4 MXFP4)."""

import re
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from halops import group_masked_gemm
from tests.utils.fp4_utils import MXFP4_BLOCK_SIZE, NVFP4_BLOCK_SIZE

from tests.utils.gemm_utils import (
    W4A4_TOL,
    W8A8_MXFP8_TOL,
    per_block_cast_to_nvfp4,
    per_block_cast_to_fp8,
    per_block_cast_to_mxfp4,
    per_token_cast_to_nvfp4,
    per_token_cast_to_fp8,
    per_token_cast_to_mxfp4,
    reorder_sf_to_cutlass,
    ue8m0_to_float32,
)


# =====================================================================
# Scale padding: pad every 224 rows to 256 rows (only pad scale, not weight)
# =====================================================================


def pad_sfb_scale_224_to_256(sfb_logical, n_orig):
    """
    Pad sfb scale along N dimension from every 224 rows to 256 rows.
    Only pads scale, not the weight matrix.

    Example: N=7168, 7168/224=32 groups, each 224->256, padded N_padded=32*256=8192.
    Padded rows are filled with 0 (UE8M0/UE4M3 code=0 maps to tiny value, does not affect computation).

    Parameters:
      sfb_logical: (..., N, kb_dim), dtype=uint8
      n_orig: original N dimension size, must be divisible by 224
    Returns:
      padded sfb: (..., N_padded, kb_dim), dtype=uint8, torch.Tensor
      n_padded: N dimension size after padding
    """
    assert n_orig % 224 == 0, f"N({n_orig}) must be divisible by 224"
    sf_np = sfb_logical.cpu().numpy()
    orig_shape = sf_np.shape
    kb_dim = orig_shape[-1]
    num_groups = n_orig // 224
    n_padded = num_groups * 256

    # Reshape N dimension into (num_groups, 224)
    leading_shape = orig_shape[:-2]
    sf_grouped = sf_np.reshape(*leading_shape, num_groups, 224, kb_dim)

    # Pad each group from 224 to 256: append 32 zero rows after the 224 dimension
    pad_width = [(0, 0)] * len(leading_shape) + [(0, 0), (0, 32), (0, 0)]
    sf_padded = np.pad(sf_grouped, pad_width, mode="constant", constant_values=0)

    # Reshape back to (..., N_padded, kb_dim)
    new_shape = list(leading_shape) + [n_padded, kb_dim]
    sf_padded = sf_padded.reshape(new_shape)

    return torch.tensor(sf_padded, dtype=torch.uint8), n_padded


# =====================================================================
# Reference implementation
# =====================================================================


def ref_group_masked_gemm(
    x, w, sfa, sfb, masked_m, max_m_per_expert, num_experts, bias=None, block_size=32
):
    """
    Reference implementation of FP8 grouped masked GEMM.
    Dequantizes x and w using their respective UE8M0 scales, then performs matrix multiplication.

    Parameters:
      x:   (num_experts * max_m_per_expert, K), dtype=float8_e4m3fn
      w:   (num_experts, N, K), dtype=float8_e4m3fn
      sfa: (num_experts * max_m_per_expert, K // block_size), dtype=uint8 (UE8M0)
           from per_token_cast_to_fp8, k-dimension block-wise scale
      sfb: (num_experts, N, K // block_size), dtype=uint8 (UE8M0)
           from per_block_cast_to_fp8, n x k 2D block-wise scale (n dim already repeated)
      masked_m: (num_experts,), valid row count per expert
      max_m_per_expert: max row count per expert
      num_experts: number of experts
      bias: (num_experts, N), dtype=bfloat16, optional. If not None, added to matmul result
      block_size: quantization block size
    """
    N = w.shape[1]

    # Dequantize sfa/sfb: UE8M0 -> float32 scale
    sfa_scale = ue8m0_to_float32(sfa)  # (M_total, k_blocks)
    sfb_scale = ue8m0_to_float32(sfb)  # (num_experts, N, k_blocks)

    ref_out = torch.zeros([x.shape[0], N], dtype=torch.bfloat16, device=x.device)

    for expert_id in range(num_experts):
        m = masked_m[expert_id]
        if m == 0:
            continue
        start = expert_id * max_m_per_expert

        # Get current expert x slice and dequantize
        # x_slice: (m, K) fp8 -> float32
        x_slice = x[start : start + m, :].to(torch.float32)
        # sfa_slice: (m, k_blocks) -> expand to (m, K)
        sfa_slice = sfa_scale[start : start + m, :]  # (m, k_blocks)
        sfa_expanded = torch.repeat_interleave(sfa_slice, block_size, dim=1)  # (m, K)
        x_deq = x_slice * sfa_expanded  # (m, K)

        # Get current expert w slice and dequantize
        # w_slice: (N, K) fp8 -> float32
        w_slice = w[expert_id].to(torch.float32)  # (N, K)
        # sfb_slice: (N, k_blocks) -> expand to (N, K)
        sfb_slice = sfb_scale[expert_id]  # (N, k_blocks)
        sfb_expanded = torch.repeat_interleave(sfb_slice, block_size, dim=1)  # (N, K)
        w_deq = w_slice * sfb_expanded  # (N, K)

        # matmul: (m, K) @ (K, N) -> (m, N)
        result = torch.matmul(x_deq, w_deq.T).to(torch.bfloat16)
        if bias is not None:
            result = result + bias[expert_id].unsqueeze(0)
        ref_out[start : start + m, :] = result

    return ref_out


def ref_group_masked_gemm_w4a4(
    x_fp4_float, w_fp4_float, masked_m, max_m_per_expert, num_experts, alpha=None
):
    """
    Reference implementation of W4A4 grouped masked GEMM.
    Directly uses dequantized float32 values for matmul, so the NVFP4 and
    MXFP4 tests share it.

    Parameters:
      x_fp4_float: (M_total, K), float32 - dequantized activation
      w_fp4_float: (num_experts, N, K), float32 - dequantized weight
      masked_m: (num_experts,), int
      max_m_per_expert: int
      num_experts: int
      alpha: (num_experts,), float32. Each expert's result is multiplied by its corresponding alpha
    """
    N = w_fp4_float.shape[1]
    ref_out = torch.zeros(
        [x_fp4_float.shape[0], N], dtype=torch.bfloat16, device=x_fp4_float.device
    )

    for expert_id in range(num_experts):
        m = masked_m[expert_id]
        if m == 0:
            continue
        start = expert_id * max_m_per_expert

        x_slice = x_fp4_float[start : start + m, :]  # (m, K)
        w_slice = w_fp4_float[expert_id]  # (N, K)

        result = torch.matmul(x_slice, w_slice.T).to(torch.bfloat16)
        if alpha is not None:
            result = result * alpha[expert_id]
        ref_out[start : start + m, :] = result

    return ref_out


# =====================================================================
# Kernel invocation and input construction
# =====================================================================


def group_masked_gemm_wrapper(
    x,
    w,
    sfa,
    sfb,
    masked_m,
    max_m_per_expert,
    bias=None,
    alpha=None,
    sm_count=-1,
    mode="w8a8-mxfp8",
):
    output = torch.zeros(
        [x.shape[0], w.shape[1]], dtype=torch.bfloat16, device=x.device
    )
    group_masked_gemm(
        x, w, sfa, sfb, masked_m, output, bias, alpha, max_m_per_expert, sm_count, mode
    )
    return output


def make_masked_m(num_experts, max_m_per_expert, full_x):
    if full_x:
        return torch.full([num_experts], max_m_per_expert, dtype=torch.int32)
    return torch.randint(0, max_m_per_expert, [num_experts], dtype=torch.int32)


def make_w8a8_mxfp8_inputs(
    max_m_per_expert,
    num_experts,
    n,
    k,
    full_x=True,
    masked_m=None,
    add_bias=False,
    pad_n_224_to_256=False,
):
    """Build W8A8-MXFP8 inputs. ``masked_m`` (list) overrides the generated row counts."""
    masked_m_t = make_masked_m(num_experts, max_m_per_expert, full_x)
    x_origin = torch.randn([num_experts * max_m_per_expert, k], dtype=torch.bfloat16)
    w_origin = torch.randn([num_experts, n, k], dtype=torch.bfloat16)

    x, sfa_logical = per_token_cast_to_fp8(x_origin, 32)
    w, sfb_logical = per_block_cast_to_fp8(w_origin, 32)

    sfa = reorder_sf_to_cutlass(sfa_logical, x.shape[0], x.shape[1] // 32)
    if pad_n_224_to_256:
        # Only pad sfb scale, weight matrix is not padded
        sfb_padded, n_padded = pad_sfb_scale_224_to_256(sfb_logical, n)
        sfb = reorder_sf_to_cutlass(sfb_padded, n_padded, w.shape[2] // 32)
    else:
        sfb = reorder_sf_to_cutlass(sfb_logical, w.shape[1], w.shape[2] // 32)

    # bias: (num_experts, n), dtype=bfloat16
    bias = torch.rand([num_experts, n], dtype=torch.bfloat16) if add_bias else None

    if masked_m is not None:
        masked_m_t = torch.tensor(masked_m, dtype=torch.int32)

    return SimpleNamespace(
        max_m_per_expert=max_m_per_expert,
        num_experts=num_experts,
        masked_m=masked_m_t,
        x=x,
        w=w,
        sfa_logical=sfa_logical,
        sfb_logical=sfb_logical,
        sfa=sfa,
        sfb=sfb,
        bias=bias,
        mode="w8a8-mxfp8",
    )


def make_nvfp4_inputs(
    max_m_per_expert,
    num_experts,
    n,
    k,
    full_x=True,
    masked_m=None,
    alpha=None,
    pad_n_224_to_256=False,
):
    """
    Build W4A4 (NVFP4) inputs.
    Quantization: FP4 E2M1 (float_e2m1_t), one scale per 16 elements, scale stored as UE4M3.
    SF permute: CUTLASS SfAtom ((32,4),(16,4)) stride ((16,4),(0,1)),
                i.e., scale_vec::4X / block16 with K=64 mode.
    ``masked_m`` (list) overrides the generated row counts; ``alpha`` (list,
    per-expert fp32 epilogue scale) defaults to all 1.0.
    """
    masked_m_t = make_masked_m(num_experts, max_m_per_expert, full_x)

    M_total = num_experts * max_m_per_expert
    x_origin = torch.randn([M_total, k], dtype=torch.bfloat16)
    w_origin = torch.randn([num_experts, n, k], dtype=torch.bfloat16)

    # FP4 quantization: block_size=16, scale uses UE4M3 encoding
    x_packed, sfa_logical, x_fp4_float = per_token_cast_to_nvfp4(
        x_origin, NVFP4_BLOCK_SIZE
    )
    w_packed, sfb_logical, w_fp4_float = per_block_cast_to_nvfp4(
        w_origin, NVFP4_BLOCK_SIZE
    )

    # SF permute: CUTLASS SfAtom interleaved layout
    kb_dim = k // NVFP4_BLOCK_SIZE
    sfa = reorder_sf_to_cutlass(sfa_logical, M_total, kb_dim)
    if pad_n_224_to_256:
        # Only pad sfb scale, weight matrix is not padded
        sfb_padded, n_padded = pad_sfb_scale_224_to_256(sfb_logical, n)
        sfb = reorder_sf_to_cutlass(sfb_padded, n_padded, kb_dim)
    else:
        sfb = reorder_sf_to_cutlass(sfb_logical, n, kb_dim)

    # alpha: (num_experts,), dtype=float32 — W4A4 requires alpha
    if alpha is None:
        alpha = [1.0] * num_experts
    alpha_tensor = torch.tensor(alpha, dtype=torch.float32)

    if masked_m is not None:
        masked_m_t = torch.tensor(masked_m, dtype=torch.int32)

    return SimpleNamespace(
        max_m_per_expert=max_m_per_expert,
        num_experts=num_experts,
        masked_m=masked_m_t,
        x_packed=x_packed,
        w_packed=w_packed,
        x_fp4_float=x_fp4_float,
        w_fp4_float=w_fp4_float,
        sfa=sfa,
        sfb=sfb,
        alpha=alpha_tensor,
        sm_count=-1,
        mode="w4a4-nvfp4",
    )


def make_mxfp4_inputs(
    max_m_per_expert,
    num_experts,
    n,
    k,
    full_x=True,
    masked_m=None,
):
    """
    Build MXFP4 inputs: FP4 E2M1 with one UE8M0 scale per 32 elements.
    ``masked_m`` (list) overrides the generated row counts. There is no
    per-expert alpha: the UE8M0 block scales carry the full scale.
    """
    masked_m_t = make_masked_m(num_experts, max_m_per_expert, full_x)

    M_total = num_experts * max_m_per_expert
    x_origin = torch.randn([M_total, k], dtype=torch.bfloat16)
    w_origin = torch.randn([num_experts, n, k], dtype=torch.bfloat16)

    x_packed, sfa_logical, x_fp4_float = per_token_cast_to_mxfp4(
        x_origin, MXFP4_BLOCK_SIZE
    )
    w_packed, sfb_logical, w_fp4_float = per_block_cast_to_mxfp4(
        w_origin, MXFP4_BLOCK_SIZE
    )

    kb_dim = k // MXFP4_BLOCK_SIZE
    sfa = reorder_sf_to_cutlass(sfa_logical, M_total, kb_dim)
    sfb = reorder_sf_to_cutlass(sfb_logical, n, kb_dim)

    if masked_m is not None:
        masked_m_t = torch.tensor(masked_m, dtype=torch.int32)

    return SimpleNamespace(
        max_m_per_expert=max_m_per_expert,
        num_experts=num_experts,
        masked_m=masked_m_t,
        x_packed=x_packed,
        w_packed=w_packed,
        x_fp4_float=x_fp4_float,
        w_fp4_float=w_fp4_float,
        sfa=sfa,
        sfb=sfb,
        sm_count=-1,
        mode="w4a4-mxfp4",
    )


def assert_close_per_expert(output, ref_out, masked_m, max_m_per_expert, tol):
    """Compare the valid rows of every expert between kernel and reference."""
    for expert_id, m in enumerate(masked_m.tolist()):
        if m == 0:
            continue
        start = expert_id * max_m_per_expert
        torch.testing.assert_close(
            output[start : start + m, :].to(torch.float32),
            ref_out[start : start + m, :].to(torch.float32),
            **tol,
            msg=lambda msg, e=expert_id, rows=m: f"expert {e} (m={rows}): {msg}",
        )


# =====================================================================
# Accuracy tests
# =====================================================================

FULL_N, FULL_K = 7168, 3584


@pytest.mark.parametrize(
    "max_m_per_expert, num_experts, n, k, full_x, masked_m",
    [
        pytest.param(128, 4, FULL_N, FULL_K, True, None, id="max_m128_e4"),
        pytest.param(4096, 20, FULL_N, FULL_K, True, None, id="large_max_m4096_e20"),
        pytest.param(64, 20, FULL_N, FULL_K, True, None, id="small_max_m64_e20"),
        pytest.param(
            128,
            4,
            256,
            256,
            False,
            [17, 63, 95, 127],
            id="max_m128_e4_n256_k256_partial",
        ),
    ],
)
def test_group_masked_w8a8_mxfp8(max_m_per_expert, num_experts, n, k, full_x, masked_m):
    inp = make_w8a8_mxfp8_inputs(
        max_m_per_expert, num_experts, n, k, full_x=full_x, masked_m=masked_m
    )
    # ref uses logical layout scale, kernel uses CUTLASS reordered scale
    ref_out = ref_group_masked_gemm(
        inp.x,
        inp.w,
        inp.sfa_logical,
        inp.sfb_logical,
        inp.masked_m,
        inp.max_m_per_expert,
        inp.num_experts,
        bias=inp.bias,
    )
    output = group_masked_gemm_wrapper(
        inp.x,
        inp.w,
        inp.sfa,
        inp.sfb,
        inp.masked_m,
        inp.max_m_per_expert,
        bias=inp.bias,
        mode=inp.mode,
    )
    assert_close_per_expert(
        output, ref_out, inp.masked_m, inp.max_m_per_expert, W8A8_MXFP8_TOL
    )


@pytest.mark.parametrize(
    "max_m_per_expert, num_experts, n, k, full_x, masked_m, random_alpha",
    [
        pytest.param(128, 4, FULL_N, FULL_K, True, None, False, id="max_m128_e4"),
        pytest.param(
            4096, 20, FULL_N, FULL_K, True, None, False, id="large_max_m4096_e20"
        ),
        pytest.param(
            4096, 20, FULL_N, FULL_K, True, None, True, id="large_max_m4096_e20_alpha"
        ),
        pytest.param(
            128,
            4,
            128,
            512,
            False,
            [11, 37, 89, 128],
            False,
            id="max_m128_e4_n128_k512_partial",
        ),
    ],
)
def test_group_masked_nvfp4(
    max_m_per_expert, num_experts, n, k, full_x, masked_m, random_alpha
):
    alpha = None
    if random_alpha:
        # Per-expert alpha scaling from a local seeded generator.
        alpha = (
            np.random.RandomState(2026)
            .uniform(0.5, 2.0, size=num_experts)
            .astype(np.float32)
            .tolist()
        )
    inp = make_nvfp4_inputs(
        max_m_per_expert,
        num_experts,
        n,
        k,
        full_x=full_x,
        masked_m=masked_m,
        alpha=alpha,
    )
    # Reference: use dequantized float32 values for matmul
    ref_out = ref_group_masked_gemm_w4a4(
        inp.x_fp4_float,
        inp.w_fp4_float,
        inp.masked_m,
        inp.max_m_per_expert,
        inp.num_experts,
        alpha=inp.alpha,
    )
    output = group_masked_gemm_wrapper(
        inp.x_packed,
        inp.w_packed,
        inp.sfa,
        inp.sfb,
        inp.masked_m,
        inp.max_m_per_expert,
        alpha=inp.alpha,
        sm_count=inp.sm_count,
        mode=inp.mode,
    )
    assert_close_per_expert(
        output, ref_out, inp.masked_m, inp.max_m_per_expert, W4A4_TOL
    )


# =====================================================================
# MXFP4 accuracy tests
# =====================================================================


@pytest.mark.parametrize(
    "max_m_per_expert, num_experts, n, k, full_x, masked_m",
    [
        pytest.param(128, 4, FULL_N, FULL_K, True, None, id="max_m128_e4"),
        pytest.param(512, 8, FULL_N, FULL_K, True, None, id="max_m512_e8"),
        pytest.param(64, 20, FULL_N, FULL_K, True, None, id="small_max_m64_e20"),
        pytest.param(
            128,
            4,
            128,
            512,
            False,
            [11, 37, 89, 128],
            id="max_m128_e4_n128_k512_partial",
        ),
    ],
)
def test_group_masked_mxfp4(max_m_per_expert, num_experts, n, k, full_x, masked_m):
    inp = make_mxfp4_inputs(
        max_m_per_expert, num_experts, n, k, full_x=full_x, masked_m=masked_m
    )
    ref_out = ref_group_masked_gemm_w4a4(
        inp.x_fp4_float,
        inp.w_fp4_float,
        inp.masked_m,
        inp.max_m_per_expert,
        inp.num_experts,
    )
    output = group_masked_gemm_wrapper(
        inp.x_packed,
        inp.w_packed,
        inp.sfa,
        inp.sfb,
        inp.masked_m,
        inp.max_m_per_expert,
        sm_count=inp.sm_count,
        mode=inp.mode,
    )
    assert_close_per_expert(
        output, ref_out, inp.masked_m, inp.max_m_per_expert, W4A4_TOL
    )


# =====================================================================
# Unsupported-input checks
# =====================================================================


def run_mxfp4_alpha():
    num_experts, max_m, n, k = 2, 128, 256, 512
    x = torch.empty([num_experts * max_m, k // 2], dtype=torch.uint8)
    w = torch.empty([num_experts, n, k // 2], dtype=torch.uint8)
    sfa = torch.empty([num_experts * max_m, k // 32], dtype=torch.uint8)
    sfb = torch.empty([num_experts, n, k // 32], dtype=torch.uint8)
    masked_m = torch.full([num_experts], max_m, dtype=torch.int32)
    out = torch.empty([num_experts * max_m, n], dtype=torch.bfloat16)
    alpha = torch.ones([num_experts], dtype=torch.float32)
    group_masked_gemm(
        x, w, sfa, sfb, masked_m, out, None, alpha, max_m, -1, "w4a4-mxfp4"
    )


def run_mxfp4_bad_sf_k():
    num_experts, max_m, n, k = 2, 128, 256, 512
    x = torch.empty([num_experts * max_m, k // 2], dtype=torch.uint8)
    w = torch.empty([num_experts, n, k // 2], dtype=torch.uint8)
    # NVFP4-shaped scales (K // 16) must be rejected for MXFP4.
    sfa = torch.empty([num_experts * max_m, k // 16], dtype=torch.uint8)
    sfb = torch.empty([num_experts, n, k // 16], dtype=torch.uint8)
    masked_m = torch.full([num_experts], max_m, dtype=torch.int32)
    out = torch.empty([num_experts * max_m, n], dtype=torch.bfloat16)
    group_masked_gemm(
        x, w, sfa, sfb, masked_m, out, None, None, max_m, -1, "w4a4-mxfp4"
    )


@pytest.mark.parametrize(
    "launch, expected",
    [
        pytest.param(run_mxfp4_alpha, "does not take alpha", id="mxfp4_alpha"),
        pytest.param(
            run_mxfp4_bad_sf_k,
            "scale-factor shape does not match mode",
            id="mxfp4_bad_sf_k",
        ),
    ],
)
def test_group_masked_mxfp4_errors(launch, expected):
    with pytest.raises(RuntimeError, match=re.escape(expected)):
        launch()


def run_w8a8_mxfp8_bad_max_m():
    num_experts, max_m, n, k = 2, 129, 256, 128
    x = torch.empty([num_experts * max_m, k], dtype=torch.float8_e4m3fn)
    w = torch.empty([num_experts, n, k], dtype=torch.float8_e4m3fn)
    sfa = torch.empty([num_experts * max_m, k // 32], dtype=torch.uint8)
    sfb = torch.empty([num_experts, n, k // 32], dtype=torch.uint8)
    masked_m = torch.full([num_experts], max_m, dtype=torch.int32)
    out = torch.empty([num_experts * max_m, n], dtype=torch.bfloat16)
    group_masked_gemm(
        x, w, sfa, sfb, masked_m, out, None, None, max_m, -1, "w8a8-mxfp8"
    )


def run_nvfp4_bias():
    num_experts, max_m, n, k = 2, 128, 128, 512
    x = torch.empty([num_experts * max_m, k // 2], dtype=torch.uint8)
    w = torch.empty([num_experts, n, k // 2], dtype=torch.uint8)
    sfa = torch.empty([num_experts * max_m, k // 16], dtype=torch.uint8)
    sfb = torch.empty([num_experts, n, k // 16], dtype=torch.uint8)
    masked_m = torch.full([num_experts], max_m, dtype=torch.int32)
    out = torch.empty([num_experts * max_m, n], dtype=torch.bfloat16)
    bias = torch.empty([num_experts, n], dtype=torch.bfloat16)
    alpha = torch.ones([num_experts], dtype=torch.float32)
    group_masked_gemm(
        x, w, sfa, sfb, masked_m, out, bias, alpha, max_m, -1, "w4a4-nvfp4"
    )


def run_zero_max_m_per_expert():
    num_experts, max_m, n, k = 2, 0, 256, 128
    x = torch.empty([0, k], dtype=torch.float8_e4m3fn)
    w = torch.empty([num_experts, n, k], dtype=torch.float8_e4m3fn)
    sfa = torch.empty([0, k // 32], dtype=torch.uint8)
    sfb = torch.empty([num_experts, n, k // 32], dtype=torch.uint8)
    masked_m = torch.zeros([num_experts], dtype=torch.int32)
    out = torch.empty([0, n], dtype=torch.bfloat16)
    group_masked_gemm(
        x, w, sfa, sfb, masked_m, out, None, None, max_m, -1, "w8a8-mxfp8"
    )


def run_masked_m_int64():
    num_experts, max_m, n, k = 2, 128, 256, 128
    x = torch.empty([num_experts * max_m, k], dtype=torch.float8_e4m3fn)
    w = torch.empty([num_experts, n, k], dtype=torch.float8_e4m3fn)
    sfa = torch.empty([num_experts * max_m, k // 32], dtype=torch.uint8)
    sfb = torch.empty([num_experts, n, k // 32], dtype=torch.uint8)
    masked_m = torch.full([num_experts], max_m, dtype=torch.int64)
    out = torch.empty([num_experts * max_m, n], dtype=torch.bfloat16)
    group_masked_gemm(
        x, w, sfa, sfb, masked_m, out, None, None, max_m, -1, "w8a8-mxfp8"
    )


def run_bad_rank_output():
    num_experts, max_m, n, k = 2, 128, 256, 128
    x = torch.empty([num_experts * max_m, k], dtype=torch.float8_e4m3fn)
    w = torch.empty([num_experts, n, k], dtype=torch.float8_e4m3fn)
    sfa = torch.empty([num_experts * max_m, k // 32], dtype=torch.uint8)
    sfb = torch.empty([num_experts, n, k // 32], dtype=torch.uint8)
    masked_m = torch.full([num_experts], max_m, dtype=torch.int32)
    out = torch.empty([num_experts * max_m, n, 1], dtype=torch.bfloat16)
    group_masked_gemm(
        x, w, sfa, sfb, masked_m, out, None, None, max_m, -1, "w8a8-mxfp8"
    )


@pytest.mark.parametrize(
    "launch, expected",
    [
        pytest.param(
            run_w8a8_mxfp8_bad_max_m,
            "No masked w8a8-mxfp8 tile for M=129",
            id="w8a8_mxfp8_bad_max_m",
        ),
        pytest.param(
            run_nvfp4_bias, "W4A4 mode does not support bias yet", id="w4a4_bias"
        ),
        pytest.param(
            run_zero_max_m_per_expert,
            "max_m_per_expert must be greater than 0",
            id="zero_max_m_per_expert",
        ),
        pytest.param(
            run_masked_m_int64,
            "masked_m must have dtype torch.int32",
            id="masked_m_int64",
        ),
        pytest.param(
            run_bad_rank_output, "output must be a 2D tensor", id="bad_rank_output"
        ),
    ],
)
def test_group_masked_unsupported_shapes(launch, expected):
    with pytest.raises(RuntimeError, match=re.escape(expected)):
        launch()


# =====================================================================
# Determinism tests
# =====================================================================


def test_group_masked_w8a8_mxfp8_determinism():
    """Determinism: identical inputs must produce bitwise-identical outputs."""
    inp = make_w8a8_mxfp8_inputs(128, 4, 256, 256, full_x=True)
    out1, out2 = (
        group_masked_gemm_wrapper(
            inp.x,
            inp.w,
            inp.sfa,
            inp.sfb,
            inp.masked_m,
            inp.max_m_per_expert,
            bias=inp.bias,
            mode=inp.mode,
        )
        for _ in range(2)
    )
    assert torch.equal(
        out1, out2
    ), "W8A8-MXFP8 group_masked_gemm output is not deterministic"


def test_group_masked_mxfp4_determinism():
    """Determinism: identical inputs must produce bitwise-identical outputs."""
    inp = make_mxfp4_inputs(128, 4, 128, 512, full_x=True)
    out1, out2 = (
        group_masked_gemm_wrapper(
            inp.x_packed,
            inp.w_packed,
            inp.sfa,
            inp.sfb,
            inp.masked_m,
            inp.max_m_per_expert,
            sm_count=inp.sm_count,
            mode=inp.mode,
        )
        for _ in range(2)
    )
    assert torch.equal(
        out1, out2
    ), "MXFP4 group_masked_gemm output is not deterministic"


def test_group_masked_nvfp4_determinism():
    """Determinism: identical inputs must produce bitwise-identical outputs."""
    inp = make_nvfp4_inputs(128, 4, 128, 512, full_x=True)
    out1, out2 = (
        group_masked_gemm_wrapper(
            inp.x_packed,
            inp.w_packed,
            inp.sfa,
            inp.sfb,
            inp.masked_m,
            inp.max_m_per_expert,
            alpha=inp.alpha,
            sm_count=inp.sm_count,
            mode=inp.mode,
        )
        for _ in range(2)
    )
    assert torch.equal(out1, out2), "W4A4 group_masked_gemm output is not deterministic"


if __name__ == "__main__":
    pytest.main([__file__])
