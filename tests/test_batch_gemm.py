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

import re

import pytest
import torch

from halops import EpilogueOp, batch_gemm
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
    sfatom_offsets,
    sfatom_rows,
    ue8m0_to_float32,
)

FP8_BLOCK_SIZE = 32


# =====================================================================
# Reference implementation
# =====================================================================


def ref_batch_gemm_fp8(
    input_fp8, weight_fp8, sfa_logical, sfb_logical, bias=None, block_size=32
):
    """
    FP8 Batch GEMM reference: output[b] = dequant(input[b]) @ dequant(weight[b])^T + bias
    input_fp8: [B, M, K] float8_e4m3fn
    weight_fp8: [B, N, K] float8_e4m3fn
    sfa_logical: [B*M, K//block_size] uint8
    sfb_logical: [B, N, K//block_size] uint8 (repeated along N)
    """
    B, M, K = input_fp8.shape
    N = weight_fp8.shape[1]

    sfa_scale = ue8m0_to_float32(sfa_logical)  # [B*M, K//bs]
    sfb_scale = ue8m0_to_float32(sfb_logical)  # [B, N, K//bs]

    output = torch.zeros([B, M, N], dtype=torch.bfloat16)

    for b in range(B):
        # Dequantize input[b]
        x_slice = input_fp8[b].to(torch.float32)  # [M, K]
        sfa_slice = sfa_scale[b * M : (b + 1) * M, :]  # [M, K//bs]
        sfa_expanded = torch.repeat_interleave(sfa_slice, block_size, dim=1)
        x_deq = x_slice * sfa_expanded

        # Dequantize weight[b]
        w_slice = weight_fp8[b].to(torch.float32)  # [N, K]
        sfb_slice = sfb_scale[b]  # [N, K//bs]
        sfb_expanded = torch.repeat_interleave(sfb_slice, block_size, dim=1)
        w_deq = w_slice * sfb_expanded

        result = torch.matmul(x_deq, w_deq.T).to(torch.bfloat16)
        if bias is not None:
            result = result + bias[b].unsqueeze(0)
        output[b] = result

    return output


def ref_batch_gemm_fp4(x_fp4_float, w_fp4_float):
    """
    FP4 Batch GEMM reference: output[b] = x_fp4_float[b] @ w_fp4_float[b]^T
    x_fp4_float: [B, M, K] float32 (already dequantized)
    w_fp4_float: [B, N, K] float32 (already dequantized)
    """
    B = x_fp4_float.shape[0]
    M = x_fp4_float.shape[1]
    N = w_fp4_float.shape[1]
    output = torch.zeros([B, M, N], dtype=torch.bfloat16)
    for b in range(B):
        output[b] = torch.matmul(x_fp4_float[b], w_fp4_float[b].T).to(torch.bfloat16)
    return output


# =====================================================================
# Input builders
# =====================================================================


def make_w8a8_mxfp8_inputs(B, M, N, K, add_bias=False):
    """Quantize random BF16 tensors to FP8 and return kernel + reference inputs."""
    x_origin = torch.randn([B, M, K], dtype=torch.bfloat16)
    w_origin = torch.randn([B, N, K], dtype=torch.bfloat16)

    # Quantize input: reshape to [B*M, K] for per_token_cast_to_fp8
    x_flat = x_origin.reshape(B * M, K)
    x_fp8_flat, sfa_logical = per_token_cast_to_fp8(x_flat, FP8_BLOCK_SIZE)
    input_fp8 = x_fp8_flat.reshape(B, M, K)  # sfa_logical: [B*M, K//32]

    # Quantize weight: [B, N, K]
    weight_fp8, sfb_logical = per_block_cast_to_fp8(w_origin, FP8_BLOCK_SIZE)

    # Reorder scale factors to SfAtom layout
    kb = K // FP8_BLOCK_SIZE
    sfa = reorder_sf_to_cutlass(sfa_logical, B * M, kb)
    sfb = reorder_sf_to_cutlass(sfb_logical, N, kb)

    bias = torch.rand([B, N], dtype=torch.bfloat16) if add_bias else None
    return dict(
        input_fp8=input_fp8,
        weight_fp8=weight_fp8,
        sfa=sfa,
        sfb=sfb,
        sfa_logical=sfa_logical,
        sfb_logical=sfb_logical,
        bias=bias,
    )


def make_nvfp4_inputs(B, M, N, K):
    """Quantize random BF16 tensors to NVFP4 and return kernel + reference inputs."""
    x_origin = torch.randn([B, M, K], dtype=torch.bfloat16)
    w_origin = torch.randn([B, N, K], dtype=torch.bfloat16)

    # Quantize input per-token: reshape to [B*M, K]
    x_flat = x_origin.reshape(B * M, K)
    x_packed_flat, sfa_logical, x_deq_flat = per_token_cast_to_nvfp4(
        x_flat, NVFP4_BLOCK_SIZE
    )
    input_packed = x_packed_flat.reshape(B, M, K // 2)  # sfa_logical: [B*M, K//16]
    x_fp4_float = x_deq_flat.reshape(B, M, K)

    # Quantize weight per-block, one batch at a time
    w_packed_list = []
    sfb_logical_list = []
    w_deq_list = []
    for b in range(B):
        w_packed_b, sfb_logical_b, w_deq_b = per_block_cast_to_nvfp4(
            w_origin[b], NVFP4_BLOCK_SIZE
        )
        w_packed_list.append(w_packed_b)
        sfb_logical_list.append(sfb_logical_b)
        w_deq_list.append(w_deq_b)
    weight_packed = torch.stack(w_packed_list, dim=0)  # [B, N, K//2]
    sfb_logical = torch.stack(sfb_logical_list, dim=0)  # [B, N, K//16]
    w_fp4_float = torch.stack(w_deq_list, dim=0)  # [B, N, K]

    # Reorder scale factors to SfAtom layout
    kb = K // NVFP4_BLOCK_SIZE
    sfa = reorder_sf_to_cutlass(sfa_logical, B * M, kb)
    sfb = reorder_sf_to_cutlass(sfb_logical, N, kb)

    # W4A4 requires a per-expert (per-batch) scaling factor with shape [B]
    alpha = torch.ones([B], dtype=torch.float32)
    return dict(
        input_packed=input_packed,
        weight_packed=weight_packed,
        sfa=sfa,
        sfb=sfb,
        alpha=alpha,
        x_fp4_float=x_fp4_float,
        w_fp4_float=w_fp4_float,
    )


def make_mxfp4_inputs(B, M, N, K):
    """Quantize random BF16 tensors to MXFP4 and return kernel + reference inputs."""
    x_origin = torch.randn([B, M, K], dtype=torch.bfloat16)
    w_origin = torch.randn([B, N, K], dtype=torch.bfloat16)

    # Activation: per-token quantization over the flattened [B*M, K] rows.
    x_packed_flat, sfa_logical, x_deq_flat = per_token_cast_to_mxfp4(
        x_origin.reshape(B * M, K), MXFP4_BLOCK_SIZE
    )
    input_packed = x_packed_flat.reshape(B, M, K // 2)  # sfa: [B*M, K//32]
    x_fp4_float = x_deq_flat.reshape(B, M, K)

    # Weight: per-block quantization over the flattened [B*N, K] rows.
    w_packed, sfb_logical, w_deq = per_block_cast_to_mxfp4(
        w_origin.reshape(B * N, K), MXFP4_BLOCK_SIZE
    )
    weight_packed = w_packed.reshape(B, N, K // 2)  # sfb: [B*N, K//32]
    w_fp4_float = w_deq.reshape(B, N, K)

    # Reorder scale factors to SfAtom layout
    kb = K // MXFP4_BLOCK_SIZE
    sfa = reorder_sf_to_cutlass(sfa_logical, B * M, kb)
    sfb = reorder_sf_to_cutlass(sfb_logical, N, kb)

    # MXFP4 has no per-batch alpha: the UE8M0 block scales carry the full scale.
    return dict(
        input_packed=input_packed,
        weight_packed=weight_packed,
        sfa=sfa,
        sfb=sfb,
        x_fp4_float=x_fp4_float,
        w_fp4_float=w_fp4_float,
    )


def assert_close_per_batch(output, ref_out, tol, label):
    """Compare the [M, B * N] kernel output against the [B, M, N] reference
    batch by batch."""
    N = ref_out.shape[2]
    for b in range(ref_out.shape[0]):
        torch.testing.assert_close(
            output[:, b * N : (b + 1) * N].to(torch.float32),
            ref_out[b].to(torch.float32),
            **tol,
            msg=lambda s, b=b: f"{label} batch {b}: {s}",
        )


# =====================================================================
# W8A8-MXFP8 correctness tests
# =====================================================================


@pytest.mark.parametrize(
    "B, M, N, K, add_bias",
    [
        # Mid-M path: total_m >= 128, uses 128x256x128 tile.
        pytest.param(4, 128, 256, 128, False, id="mid_m"),
        # Large-M path: total_m >= 4096, uses 256x256x128 tile.
        pytest.param(4, 1024, 256, 128, False, id="large_m"),
        # Bias path: uses 256x128x128 tile.
        pytest.param(2, 256, 128, 128, True, id="with_bias"),
        # Boundary: total_m < 128 triggers the SwapAB path (needs N%128, M%64, K%256).
        pytest.param(1, 64, 128, 256, False, id="small_m_swapab"),
        # Boundary: single batch (B=1) mid-M path.
        pytest.param(1, 256, 256, 128, False, id="single_batch"),
        # Boundary: larger K dimension.
        pytest.param(2, 128, 256, 512, False, id="larger_k"),
        # M not a tile multiple: tiles straddle two batches.
        pytest.param(8, 1, 1024, 256, False, id="m1_swapab"),
        pytest.param(3, 100, 256, 256, False, id="m100_swapab"),
        pytest.param(3, 200, 256, 128, False, id="m200"),
        pytest.param(8, 1000, 1024, 256, False, id="m1000"),
        pytest.param(2, 640, 256, 128, False, id="m640"),
        pytest.param(2, 2176, 256, 128, False, id="m2176_pair"),
        pytest.param(2, 640, 128, 128, True, id="m640_bias_pair"),
        pytest.param(2, 1000, 256, 128, False, id="m1000_partial_pair"),
        pytest.param(3, 300, 128, 128, True, id="m300_bias_partial_pair"),
        pytest.param(2, 5, 256, 256, False, id="grid10_swap16"),
    ],
)
def test_batch_gemm_w8a8_mxfp8(B, M, N, K, add_bias):
    inp = make_w8a8_mxfp8_inputs(B, M, N, K, add_bias=add_bias)
    ref_out = ref_batch_gemm_fp8(
        inp["input_fp8"],
        inp["weight_fp8"],
        inp["sfa_logical"],
        inp["sfb_logical"],
        bias=inp["bias"],
        block_size=FP8_BLOCK_SIZE,
    )

    output = torch.zeros([M, B * N], dtype=torch.bfloat16)
    batch_gemm(
        inp["input_fp8"],
        inp["weight_fp8"],
        inp["sfa"],
        inp["sfb"],
        output,
        bias=inp["bias"],
        sm_count=-1,
        mode="w8a8-mxfp8",
    )

    assert_close_per_batch(
        output,
        ref_out,
        W8A8_MXFP8_TOL,
        f"W8A8-MXFP8 B={B}, M={M}, N={N}, K={K}, bias={add_bias}",
    )


# =====================================================================
# W8A8-MXFP8 error input tests
# =====================================================================

_W8A8_MXFP8_ERR_SHAPE = dict(B=2, M=128, N=256, K=128)


def make_empty_w8a8_mxfp8(B, M, N, K, x_shape=None, w_shape=None, out_shape=None):
    """Uninitialized W8A8-MXFP8 kernel inputs for argument-validation tests."""
    x = torch.empty(x_shape or [B, M, K], dtype=torch.float8_e4m3fn)
    w = torch.empty(w_shape or [B, N, K], dtype=torch.float8_e4m3fn)
    sfa = torch.empty([B * M, K // 32], dtype=torch.uint8)
    sfb = torch.empty([B, N, K // 32], dtype=torch.uint8)
    out = torch.empty(out_shape or [M, B * N], dtype=torch.bfloat16)
    return x, w, sfa, sfb, out


@pytest.mark.parametrize(
    "K, x_shape, w_shape, out_shape, mode, expected",
    [
        pytest.param(128, None, None, None, "w2a2", "Unsupported mode", id="bad_mode"),
        pytest.param(
            128,
            [256, 128],
            None,
            None,
            "w8a8-mxfp8",
            "input must be a 3D tensor",
            id="bad_input_rank",
        ),
        pytest.param(
            128,
            None,
            [3, 256, 128],
            None,
            "w8a8-mxfp8",
            "weight batch dim must match",
            id="bad_weight_batch",
        ),
        # The output is token-major [M, B * N]; [B, M, N] is rejected.
        pytest.param(
            128,
            None,
            None,
            [2, 128, 256],
            "w8a8-mxfp8",
            "output must be a 2D tensor",
            id="bad_output_rank",
        ),
        # K=96 is not divisible by 128 (mid-M tile K).
        pytest.param(
            96,
            None,
            None,
            None,
            "w8a8-mxfp8",
            "No batch w8a8-mxfp8 tile for M=128 (grid M=256), N=256, K=96",
            id="bad_k",
        ),
    ],
)
def test_batch_gemm_w8a8_mxfp8_errors(K, x_shape, w_shape, out_shape, mode, expected):
    shape = dict(_W8A8_MXFP8_ERR_SHAPE, K=K)
    x, w, sfa, sfb, out = make_empty_w8a8_mxfp8(
        **shape, x_shape=x_shape, w_shape=w_shape, out_shape=out_shape
    )
    with pytest.raises(RuntimeError, match=re.escape(expected)):
        batch_gemm(x, w, sfa, sfb, out, None, None, -1, mode)


# =====================================================================
# W8A8-MXFP8 determinism test
# =====================================================================


def test_batch_gemm_w8a8_mxfp8_determinism():
    """Determinism: identical inputs must produce bitwise-identical outputs."""
    B, M, N, K = 4, 128, 256, 128
    inp = make_w8a8_mxfp8_inputs(B, M, N, K)
    args = (inp["input_fp8"], inp["weight_fp8"], inp["sfa"], inp["sfb"])
    out1 = torch.zeros([M, B * N], dtype=torch.bfloat16)
    out2 = torch.zeros([M, B * N], dtype=torch.bfloat16)
    batch_gemm(*args, out1, None, None, -1, "w8a8-mxfp8")
    batch_gemm(*args, out2, None, None, -1, "w8a8-mxfp8")
    assert torch.equal(out1, out2), "W8A8-MXFP8 batch_gemm output is not deterministic"


# =====================================================================
# W8A8-MXFP8 with an e4m3 output (quantizing epilogue)
# =====================================================================


def ref_quant_e4m3(v):
    """Per-32-column MXFP8 reference: e4m3 codes and UE8M0 exponents."""
    g = v.float().unflatten(-1, (-1, FP8_BLOCK_SIZE))
    amax = g.abs().amax(-1).clamp_min(1e-10)
    mant, exp = torch.frexp(amax)
    e = (exp - 9 + (mant > 0.875).int() + 127).clamp(0, 254)
    q = (g * torch.exp2(127 - e.float())[..., None]).flatten(-2)
    return q.to(torch.float8_e4m3fn).view(torch.uint8), e.to(torch.uint8)


@pytest.mark.parametrize("with_gate", [False, True], ids=["quant", "gate_mul"])
@pytest.mark.parametrize(
    "B, M, N, K",
    [
        pytest.param(4, 128, 256, 128, id="mid_m"),
        pytest.param(2, 640, 256, 128, id="m640_pair"),
        pytest.param(3, 200, 256, 128, id="m200_straddle"),
        pytest.param(3, 100, 256, 256, id="m100_swapab"),
        pytest.param(8, 1000, 1024, 256, id="m1000_pair_swapab"),
        pytest.param(2, 5, 256, 256, id="grid10_swap16"),
        pytest.param(8, 2000, 1024, 512, id="m2000"),
    ],
)
def test_batch_gemm_w8a8_mxfp8_e4m3_out(B, M, N, K, with_gate):
    """The e4m3 output and its scales equal the bf16 output (times gate)
    quantized per 32 columns; scale padding is left untouched."""
    inp = make_w8a8_mxfp8_inputs(B, M, N, K)
    args = (inp["input_fp8"], inp["weight_fp8"], inp["sfa"], inp["sfb"])
    y = torch.empty([M, B * N], dtype=torch.bfloat16)
    batch_gemm(*args, y, None, None, -1, "w8a8-mxfp8")
    gate = (torch.rand([M, B * N]) * 2 - 0.5).to(torch.bfloat16) if with_gate else None
    ref_q, ref_sf = ref_quant_e4m3(y * gate if with_gate else y)

    sf_cols = B * N // FP8_BLOCK_SIZE
    sf_idx = sfatom_offsets(M, sf_cols)
    out = torch.empty([M, B * N], dtype=torch.float8_e4m3fn)
    out_sf = torch.zeros([sfatom_rows(M) * sf_cols], dtype=torch.uint8)
    op = EpilogueOp.GateMulQuant if with_gate else EpilogueOp.Quant
    batch_gemm(*args, out, None, None, -1, "w8a8-mxfp8", op, out_sf, gate)
    assert torch.equal(out.view(torch.uint8), ref_q)
    assert torch.equal(out_sf[sf_idx], ref_sf)
    pad = torch.ones_like(out_sf, dtype=torch.bool)
    pad[sf_idx] = False
    assert (out_sf[pad] == 0).all()

    out2 = torch.empty_like(out)
    out_sf2 = torch.zeros_like(out_sf)
    batch_gemm(*args, out2, None, None, -1, "w8a8-mxfp8", op, out_sf2, gate)
    assert torch.equal(out.view(torch.uint8), out2.view(torch.uint8))
    assert torch.equal(out_sf, out_sf2)


# =====================================================================
# NVFP4 (W4A4) correctness tests
# =====================================================================


@pytest.mark.parametrize(
    "B, M, N, K",
    [
        # Mid-M path: total_m > 128, uses 128x256x256 tile.
        pytest.param(4, 128, 256, 256, id="mid_m"),
        # Large-M path: total_m >= 4096, uses 256x256x256 tile.
        pytest.param(4, 1024, 256, 256, id="large_m"),
        # Boundary: total_m <= 128 triggers SwapAB path (needs N%128, M%64, K%512).
        pytest.param(1, 64, 128, 512, id="small_m_swapab"),
        # Boundary: single batch (B=1) mid-M path.
        pytest.param(1, 256, 256, 256, id="single_batch"),
    ],
)
def test_batch_gemm_nvfp4(B, M, N, K):
    inp = make_nvfp4_inputs(B, M, N, K)
    ref_out = ref_batch_gemm_fp4(inp["x_fp4_float"], inp["w_fp4_float"])

    output = torch.zeros([M, B * N], dtype=torch.bfloat16)
    batch_gemm(
        inp["input_packed"],
        inp["weight_packed"],
        inp["sfa"],
        inp["sfb"],
        output,
        alpha=inp["alpha"],
        sm_count=-1,
        mode="w4a4-nvfp4",
    )

    assert_close_per_batch(
        output, ref_out, W4A4_TOL, f"W4A4 B={B}, M={M}, N={N}, K={K}"
    )


# =====================================================================
# NVFP4 (W4A4) error input tests
# =====================================================================

_NVFP4_ERR_SHAPE = dict(B=2, M=128, N=256, K=256)


def make_empty_nvfp4(B, M, N, K, w_packed_k=None):
    """Uninitialized W4A4 kernel inputs for argument-validation tests."""
    x = torch.empty([B, M, K // 2], dtype=torch.uint8)
    w = torch.empty([B, N, w_packed_k or K // 2], dtype=torch.uint8)
    sfa = torch.empty([B * M, K // 16], dtype=torch.uint8)
    sfb = torch.empty([B, N, K // 16], dtype=torch.uint8)
    out = torch.empty([M, B * N], dtype=torch.bfloat16)
    alpha = torch.ones([1], dtype=torch.float32)
    return x, w, sfa, sfb, out, alpha


@pytest.mark.parametrize(
    "K, w_packed_k, pass_alpha, add_bias, expected",
    [
        pytest.param(
            256,
            None,
            False,
            False,
            "W4A4-nvfp4 mode requires alpha",
            id="missing_alpha",
        ),
        # K=32 fits no W4A4 tile.
        pytest.param(
            32,
            None,
            True,
            False,
            "No batch w4a4-nvfp4 tile for M=",
            id="bad_k",
        ),
        pytest.param(
            256,
            256 // 2 + 1,
            True,
            False,
            "packed input K/2 must match",
            id="mismatched_packed_k",
        ),
        pytest.param(
            256,
            None,
            True,
            True,
            "W4A4 batch GEMM does not support bias",
            id="bias_unsupported",
        ),
    ],
)
def test_batch_gemm_nvfp4_errors(K, w_packed_k, pass_alpha, add_bias, expected):
    shape = dict(_NVFP4_ERR_SHAPE, K=K)
    x, w, sfa, sfb, out, alpha = make_empty_nvfp4(**shape, w_packed_k=w_packed_k)
    bias = (
        torch.zeros([shape["B"], shape["N"]], dtype=torch.bfloat16)
        if add_bias
        else None
    )
    with pytest.raises(RuntimeError, match=re.escape(expected)):
        batch_gemm(
            x, w, sfa, sfb, out, bias, alpha if pass_alpha else None, -1, "w4a4-nvfp4"
        )


# =====================================================================
# NVFP4 (W4A4) determinism test
# =====================================================================


def test_batch_gemm_nvfp4_determinism():
    """Determinism: identical inputs must produce bitwise-identical outputs."""
    B, M, N, K = 4, 128, 256, 256
    inp = make_nvfp4_inputs(B, M, N, K)
    args = (inp["input_packed"], inp["weight_packed"], inp["sfa"], inp["sfb"])
    out1 = torch.zeros([M, B * N], dtype=torch.bfloat16)
    out2 = torch.zeros([M, B * N], dtype=torch.bfloat16)
    batch_gemm(*args, out1, None, inp["alpha"], -1, "w4a4-nvfp4")
    batch_gemm(*args, out2, None, inp["alpha"], -1, "w4a4-nvfp4")
    assert torch.equal(out1, out2), "W4A4 batch_gemm output is not deterministic"


# =====================================================================
# MXFP4 (W4A4 with UE8M0 scales) correctness tests
# =====================================================================


@pytest.mark.parametrize(
    "B, M, N, K",
    [
        pytest.param(4, 128, 256, 512, id="mid_m"),
        pytest.param(4, 1024, 256, 512, id="large_m"),
        pytest.param(1, 64, 128, 512, id="small_m_swapab"),
        pytest.param(1, 256, 256, 512, id="single_batch"),
    ],
)
def test_batch_gemm_mxfp4(B, M, N, K):
    inp = make_mxfp4_inputs(B, M, N, K)
    ref_out = ref_batch_gemm_fp4(inp["x_fp4_float"], inp["w_fp4_float"])

    output = torch.zeros([M, B * N], dtype=torch.bfloat16)
    batch_gemm(
        inp["input_packed"],
        inp["weight_packed"],
        inp["sfa"],
        inp["sfb"],
        output,
        sm_count=-1,
        mode="w4a4-mxfp4",
    )

    assert_close_per_batch(
        output, ref_out, W4A4_TOL, f"MXFP4 B={B}, M={M}, N={N}, K={K}"
    )


# =====================================================================
# MXFP4 error input tests
# =====================================================================

_MXFP4_ERR_SHAPE = dict(B=2, M=128, N=256, K=512)


def run_mxfp4_alpha():
    shape = _MXFP4_ERR_SHAPE
    x = torch.empty([shape["B"], shape["M"], shape["K"] // 2], dtype=torch.uint8)
    w = torch.empty([shape["B"], shape["N"], shape["K"] // 2], dtype=torch.uint8)
    sfa = torch.empty([shape["B"] * shape["M"], shape["K"] // 32], dtype=torch.uint8)
    sfb = torch.empty([shape["B"], shape["N"], shape["K"] // 32], dtype=torch.uint8)
    out = torch.empty([shape["M"], shape["B"] * shape["N"]], dtype=torch.bfloat16)
    alpha = torch.ones([shape["B"]], dtype=torch.float32)
    batch_gemm(x, w, sfa, sfb, out, None, alpha, -1, "w4a4-mxfp4")


def run_mxfp4_bias():
    shape = _MXFP4_ERR_SHAPE
    x = torch.empty([shape["B"], shape["M"], shape["K"] // 2], dtype=torch.uint8)
    w = torch.empty([shape["B"], shape["N"], shape["K"] // 2], dtype=torch.uint8)
    sfa = torch.empty([shape["B"] * shape["M"], shape["K"] // 32], dtype=torch.uint8)
    sfb = torch.empty([shape["B"], shape["N"], shape["K"] // 32], dtype=torch.uint8)
    out = torch.empty([shape["M"], shape["B"] * shape["N"]], dtype=torch.bfloat16)
    bias = torch.zeros([shape["B"], shape["N"]], dtype=torch.bfloat16)
    batch_gemm(x, w, sfa, sfb, out, bias, None, -1, "w4a4-mxfp4")


def run_mxfp4_bad_sf_k():
    shape = _MXFP4_ERR_SHAPE
    x = torch.empty([shape["B"], shape["M"], shape["K"] // 2], dtype=torch.uint8)
    w = torch.empty([shape["B"], shape["N"], shape["K"] // 2], dtype=torch.uint8)
    # NVFP4-shaped scales (K // 16) must be rejected for MXFP4.
    sfa = torch.empty([shape["B"] * shape["M"], shape["K"] // 16], dtype=torch.uint8)
    sfb = torch.empty([shape["B"], shape["N"], shape["K"] // 16], dtype=torch.uint8)
    out = torch.empty([shape["M"], shape["B"] * shape["N"]], dtype=torch.bfloat16)
    batch_gemm(x, w, sfa, sfb, out, None, None, -1, "w4a4-mxfp4")


@pytest.mark.parametrize(
    "launch, expected",
    [
        pytest.param(run_mxfp4_alpha, "does not take alpha", id="alpha_unsupported"),
        pytest.param(
            run_mxfp4_bias,
            "W4A4 batch GEMM does not support bias",
            id="bias_unsupported",
        ),
        pytest.param(
            run_mxfp4_bad_sf_k,
            "scale-factor shape does not match mode",
            id="bad_sf_k",
        ),
    ],
)
def test_batch_gemm_mxfp4_errors(launch, expected):
    with pytest.raises(RuntimeError, match=re.escape(expected)):
        launch()


# =====================================================================
# MXFP4 determinism test
# =====================================================================


def test_batch_gemm_mxfp4_determinism():
    """Determinism: identical inputs must produce bitwise-identical outputs."""
    B, M, N, K = 4, 128, 256, 512
    inp = make_mxfp4_inputs(B, M, N, K)
    args = (inp["input_packed"], inp["weight_packed"], inp["sfa"], inp["sfb"])
    out1 = torch.zeros([M, B * N], dtype=torch.bfloat16)
    out2 = torch.zeros([M, B * N], dtype=torch.bfloat16)
    batch_gemm(*args, out1, None, None, -1, "w4a4-mxfp4")
    batch_gemm(*args, out2, None, None, -1, "w4a4-mxfp4")
    assert torch.equal(out1, out2), "MXFP4 batch_gemm output is not deterministic"


if __name__ == "__main__":
    pytest.main([__file__])
