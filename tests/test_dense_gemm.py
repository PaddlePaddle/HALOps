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

import re

import pytest
import torch

from halops import dense_gemm
from tests.utils.fp4_utils import FP4_BLOCK_SIZE

from tests.utils.gemm_utils import (
    W4A4_TOL,
    W8A8_TOL,
    per_block_cast_to_fp4,
    per_block_cast_to_fp8,
    per_token_cast_to_fp4,
    per_token_cast_to_fp8,
    reorder_sf_to_cutlass,
    ue8m0_to_float32,
)

FP8_BLOCK_SIZE = 32


# =====================================================================
# Reference implementation
# =====================================================================


def ref_gemm(x, w, sfa, sfb, block_size=32):
    """
    Reference implementation of single-expert FP8 GEMM.
    x: (m, k) fp8, w: (n, k) fp8
    sfa: (m, k//block_size) ue8m0, sfb: (n, k//block_size) ue8m0
    """
    sfa_scale = ue8m0_to_float32(sfa)  # (m, k_blocks)
    sfb_scale = ue8m0_to_float32(sfb)  # (n, k_blocks)

    x_f32 = x.to(torch.float32)
    sfa_expanded = torch.repeat_interleave(sfa_scale, block_size, dim=1)
    x_deq = x_f32 * sfa_expanded

    w_f32 = w.to(torch.float32)
    sfb_expanded = torch.repeat_interleave(sfb_scale, block_size, dim=1)
    w_deq = w_f32 * sfb_expanded

    return torch.matmul(x_deq, w_deq.T).to(torch.bfloat16)


def ref_gemm_w4a4(x_fp4_float, w_fp4_float):
    """Reference implementation of single-expert W4A4 GEMM using dequantized float32 matmul."""
    return torch.matmul(x_fp4_float, w_fp4_float.T).to(torch.bfloat16)


# =====================================================================
# Kernel invocation and input builders
# =====================================================================


def gemm_wrapper(x, w, sfa, sfb, bias=None, alpha=None, sm_count=-1, mode="w8a8"):
    output = torch.zeros([x.shape[0], w.shape[0]], dtype=torch.bfloat16)
    dense_gemm(x, w, sfa, sfb, output, bias, alpha, sm_count, mode)
    return output


def make_w8a8_inputs(m, n, k):
    """Quantize random BF16 x/w to FP8 and return kernel + reference inputs."""
    x_origin = torch.randn([m, k], dtype=torch.bfloat16)
    w_origin = torch.randn([n, k], dtype=torch.bfloat16)

    x, sfa_logical = per_token_cast_to_fp8(x_origin, FP8_BLOCK_SIZE)
    w, sfb_logical = per_block_cast_to_fp8(w_origin, FP8_BLOCK_SIZE)

    sfa = reorder_sf_to_cutlass(sfa_logical, x.shape[0], x.shape[1] // FP8_BLOCK_SIZE)
    sfb = reorder_sf_to_cutlass(sfb_logical, w.shape[0], w.shape[1] // FP8_BLOCK_SIZE)
    return dict(
        x=x, w=w, sfa=sfa, sfb=sfb, sfa_logical=sfa_logical, sfb_logical=sfb_logical
    )


def make_w4a4_inputs(m, n, k):
    """Quantize random BF16 x/w to NVFP4 and return kernel + reference inputs."""
    x_origin = torch.randn([m, k], dtype=torch.bfloat16)
    w_origin = torch.randn([n, k], dtype=torch.bfloat16)

    x_packed, sfa_logical, x_fp4_float = per_token_cast_to_fp4(x_origin, FP4_BLOCK_SIZE)
    w_packed, sfb_logical, w_fp4_float = per_block_cast_to_fp4(w_origin, FP4_BLOCK_SIZE)

    kb_dim = k // FP4_BLOCK_SIZE
    sfa = reorder_sf_to_cutlass(sfa_logical, m, kb_dim)
    sfb = reorder_sf_to_cutlass(sfb_logical, n, kb_dim)

    # Single expert: alpha is [1]
    alpha = torch.ones([1], dtype=torch.float32)
    return dict(
        x_packed=x_packed,
        w_packed=w_packed,
        sfa=sfa,
        sfb=sfb,
        alpha=alpha,
        x_fp4_float=x_fp4_float,
        w_fp4_float=w_fp4_float,
    )


def make_empty_inputs(m, n, k, mode, x_shape=None, out_shape=None):
    """Uninitialized kernel inputs for argument-validation tests."""
    if mode == "w8a8":
        x = torch.empty(x_shape or [m, k], dtype=torch.float8_e4m3fn)
        w = torch.empty([n, k], dtype=torch.float8_e4m3fn)
        sfa = torch.empty([m, k // 32], dtype=torch.uint8)
        sfb = torch.empty([n, k // 32], dtype=torch.uint8)
        alpha = None
    else:
        x = torch.empty(x_shape or [m, k // 2], dtype=torch.uint8)
        w = torch.empty([n, k // 2], dtype=torch.uint8)
        sfa = torch.empty([m, k // 16], dtype=torch.uint8)
        sfb = torch.empty([n, k // 16], dtype=torch.uint8)
        alpha = torch.ones([1], dtype=torch.float32)
    out = torch.empty(out_shape or [m, n], dtype=torch.bfloat16)
    return x, w, sfa, sfb, out, alpha


# =====================================================================
# W8A8 (FP8) tests
# =====================================================================


@pytest.mark.parametrize(
    "m, n, k",
    [
        (4096, 7168, 3584),
        (64, 7168, 3584),
        (128, 256, 256),
        (64, 128, 256),
    ],
    ids=[
        "large_m-4096x7168x3584",
        "small_m-64x7168x3584",
        "aligned-128x256x256",
        "aligned-64x128x256",
    ],
)
def test_gemm_w8a8(m, n, k):
    inp = make_w8a8_inputs(m, n, k)
    ref_out = ref_gemm(inp["x"], inp["w"], inp["sfa_logical"], inp["sfb_logical"])
    output = gemm_wrapper(
        inp["x"], inp["w"], inp["sfa"], inp["sfb"], sm_count=-1, mode="w8a8"
    )
    torch.testing.assert_close(
        output.to(torch.float32),
        ref_out.to(torch.float32),
        **W8A8_TOL,
        msg=lambda s: f"W8A8 dense_gemm m={m}, n={n}, k={k}: {s}",
    )


# =====================================================================
# W4A4 (NVFP4) tests
# =====================================================================


@pytest.mark.parametrize(
    "m, n, k",
    [
        (4096, 7168, 3584),
        (128, 256, 512),
    ],
    ids=["large_m-4096x7168x3584", "aligned-128x256x512"],
)
def test_gemm_w4a4(m, n, k):
    inp = make_w4a4_inputs(m, n, k)
    ref_out = ref_gemm_w4a4(inp["x_fp4_float"], inp["w_fp4_float"])
    output = gemm_wrapper(
        inp["x_packed"],
        inp["w_packed"],
        inp["sfa"],
        inp["sfb"],
        alpha=inp["alpha"],
        sm_count=-1,
        mode="w4a4-nvfp4",
    )
    torch.testing.assert_close(
        output.to(torch.float32),
        ref_out.to(torch.float32),
        **W4A4_TOL,
        msg=lambda s: f"W4A4 dense_gemm m={m}, n={n}, k={k}: {s}",
    )


# =====================================================================
# Unsupported shapes
# =====================================================================


@pytest.mark.parametrize(
    "mode, m, n, k, x_shape, out_shape, expected",
    [
        pytest.param(
            "w8a8",
            128,
            256,
            96,
            None,
            None,
            "No dense w8a8 tile for M=128 (grid M=128), N=256, K=96",
            id="w8a8_bad_k",
        ),
        # Shape-aware dispatch includes N=128 tiles; N=255 fits none of them.
        pytest.param(
            "w8a8",
            128,
            255,
            128,
            None,
            None,
            "No dense w8a8 tile for M=128 (grid M=128), N=255, K=128",
            id="w8a8_bad_n",
        ),
        pytest.param(
            "w4a4-nvfp4",
            128,
            256,
            48,
            None,
            None,
            "No dense w4a4-nvfp4 tile for M=128 (grid M=128), N=256, K=48",
            id="w4a4_bad_k",
        ),
        pytest.param(
            "w8a8",
            128,
            256,
            128,
            [128, 1, 128],
            None,
            "input must be a 2D tensor",
            id="bad_rank_input",
        ),
        pytest.param(
            "w8a8",
            128,
            256,
            128,
            None,
            [128, 256, 1],
            "output must be a 2D tensor",
            id="bad_rank_output",
        ),
    ],
)
def test_gemm_unsupported_shapes(mode, m, n, k, x_shape, out_shape, expected):
    x, w, sfa, sfb, out, alpha = make_empty_inputs(m, n, k, mode, x_shape, out_shape)
    with pytest.raises(RuntimeError, match=re.escape(expected)):
        dense_gemm(x, w, sfa, sfb, out, None, alpha, -1, mode)


# =====================================================================
# Determinism
# =====================================================================


def test_gemm_w8a8_determinism():
    """Determinism: identical inputs must produce bitwise-identical outputs."""
    inp = make_w8a8_inputs(128, 256, 256)
    args = (inp["x"], inp["w"], inp["sfa"], inp["sfb"])
    out1 = gemm_wrapper(*args, sm_count=-1, mode="w8a8")
    out2 = gemm_wrapper(*args, sm_count=-1, mode="w8a8")
    assert torch.equal(out1, out2), "W8A8 dense_gemm output is not deterministic"


def test_gemm_w4a4_determinism():
    """Determinism: identical inputs must produce bitwise-identical outputs."""
    inp = make_w4a4_inputs(128, 256, 512)
    args = (inp["x_packed"], inp["w_packed"], inp["sfa"], inp["sfb"])
    out1 = gemm_wrapper(*args, alpha=inp["alpha"], sm_count=-1, mode="w4a4-nvfp4")
    out2 = gemm_wrapper(*args, alpha=inp["alpha"], sm_count=-1, mode="w4a4-nvfp4")
    assert torch.equal(out1, out2), "W4A4 dense_gemm output is not deterministic"


if __name__ == "__main__":
    pytest.main([__file__])
