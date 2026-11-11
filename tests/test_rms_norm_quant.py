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
from tests.utils.gemm_utils import (
    MXFP8_BLOCK_SIZE,
    sfatom_offsets,
    sfatom_rows,
    ulp_diff,
)

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


@pytest.mark.parametrize("use_swizzle", [True, False], ids=["sfatom", "rowmajor"])
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
def test_rms_norm_quant(m, k, use_swizzle):
    """The kernel's bf16 norm may differ from an fp32 RMSNorm by 1 ulp (rsqrt
    and the summation order differ). That moves a code by at most 1, or rarely
    a group's amax across a power of two and with it the scale by 1: against
    group_wise_quant of the fp32 norm, codes match within 1 where the scale
    matches, and at most 0.1% of the scales differ, each by 1."""
    x, w = make_inputs(m, k)
    q, sf = rms_norm_quant(x, w, EPS, use_swizzle=use_swizzle)
    ref_q, ref_sf = group_wise_quant(
        ref_rms_norm(x, w), mode="w8a8-mxfp8", use_swizzle=use_swizzle
    )
    sf_diff = (sf.int() - ref_sf.int()).abs()
    if use_swizzle:
        sf_diff = sf_diff.flatten()[sfatom_offsets(m, k // MXFP8_BLOCK_SIZE)]
    assert sf_diff.max() <= 1 and (sf_diff > 0).float().mean() <= 1e-3
    same_scale = (sf_diff == 0).repeat_interleave(MXFP8_BLOCK_SIZE, -1)
    assert ulp_diff(q, ref_q)[same_scale].max() <= 1


@pytest.mark.parametrize("m", [1, 200], ids=lambda m: f"m{m}")
def test_rms_norm_quant_strided_input(m):
    """A column slice of a wider tensor quantizes like its contiguous copy."""
    k = 1024
    x, w = make_inputs(m, 2 * k)
    x, w = x[:, k:], w[:k]
    ref_q, ref_sf = rms_norm_quant(x.contiguous(), w, EPS, use_pdl=False)
    q, sf = rms_norm_quant(x, w, EPS)
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
