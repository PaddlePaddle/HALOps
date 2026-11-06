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

from halops import group_wise_quant, rms_norm_quant
from tests.utils.gemm_utils import MXFP8_BLOCK_SIZE, sfatom_rows, ulp_diff

EPS = 1e-6


# =====================================================================
# Reference implementation
# =====================================================================


def ref_rms_norm(x, w):
    """fp32 RMSNorm rounded to bf16."""
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS) * w.float()
    return y.to(torch.bfloat16)


def make_inputs(m, k):
    x = torch.randn([m, k], dtype=torch.bfloat16) * 3
    w = torch.rand([k], dtype=torch.bfloat16) + 0.5
    return x, w


# =====================================================================
# Correctness tests
# =====================================================================


@pytest.mark.parametrize("k", [1024, 4096, 8192], ids=lambda k: f"k{k}")
@pytest.mark.parametrize(
    "m",
    [
        # Decode sizes: one partial SfAtom of scale rows.
        pytest.param(1, id="m1"),
        pytest.param(16, id="m16"),
        # Exactly one SfAtom.
        pytest.param(128, id="m128"),
        # Partial last SfAtom.
        pytest.param(200, id="m200"),
        # Prefill sizes.
        pytest.param(1536, id="m1536"),
        pytest.param(3072, id="m3072"),
    ],
)
def test_rms_norm_quant(m, k):
    """y is within 1 bf16 ulp of an fp32 RMSNorm (rsqrt and the summation
    order differ), and the fp8 codes and scales equal group_wise_quant(y)."""
    x, w = make_inputs(m, k)
    q, sf, y = rms_norm_quant(x, w, EPS, return_bf16=True)
    assert ulp_diff(y, ref_rms_norm(x, w)).max() <= 1
    ref_q, ref_sf = group_wise_quant(y, mode="w8a8-mxfp8")
    assert torch.equal(q.view(torch.uint8), ref_q.view(torch.uint8))
    assert torch.equal(sf, ref_sf)

    q2, sf2 = rms_norm_quant(x, w, EPS)
    assert torch.equal(q2.view(torch.uint8), q.view(torch.uint8))
    assert torch.equal(sf2, sf)


@pytest.mark.parametrize("m", [1, 200], ids=lambda m: f"m{m}")
def test_rms_norm_quant_strided_input_preallocated_outputs(m):
    """A column slice of a wider tensor quantizes like its contiguous copy, into
    the given buffers (scale padding rows are overwritten with 0)."""
    k = 1024
    x, w = make_inputs(m, 2 * k)
    x, w = x[:, k:], w[:k]
    ref_q, ref_sf = rms_norm_quant(x.contiguous(), w, EPS, use_pdl=False)
    out = torch.empty_like(ref_q)
    out_sf = torch.full_like(ref_sf, 0xFF)
    q, sf = rms_norm_quant(x, w, EPS, output_quant=out, output_sf=out_sf)
    assert q.data_ptr() == out.data_ptr() and sf.data_ptr() == out_sf.data_ptr()
    assert torch.equal(q.view(torch.uint8), ref_q.view(torch.uint8))
    assert torch.equal(sf, ref_sf)


def test_rms_norm_quant_empty_input():
    k = 1024
    x, w = make_inputs(0, k)
    q, sf = rms_norm_quant(x, w, EPS)
    assert q.shape == (0, k)
    assert sf.shape == (sfatom_rows(0), k // MXFP8_BLOCK_SIZE)


# =====================================================================
# Error input tests
# =====================================================================


@pytest.mark.parametrize(
    "k, expected",
    [
        pytest.param(1056, "K must be a multiple of 128", id="k_not_128_multiple"),
        pytest.param(
            8320, "K must be a multiple of 128 and at most 8192", id="k_too_large"
        ),
    ],
)
def test_rms_norm_quant_errors(k, expected):
    x, w = make_inputs(4, k)
    with pytest.raises(RuntimeError, match=re.escape(expected)):
        rms_norm_quant(x, w, EPS)


if __name__ == "__main__":
    pytest.main([__file__])
