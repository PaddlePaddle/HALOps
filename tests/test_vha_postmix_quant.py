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

import math
import re

import pytest
import torch

from halops import group_wise_quant, vha_postmix_quant
from tests.utils.gemm_utils import MXFP8_BLOCK_SIZE, ulp_diff

HEADS, HEAD_DIM, ROPE_DIM, GROUPS = 64, 512, 64, 8
MAX_POS = 4096

# Token counts: decode sizes, partial and whole SfAtoms of G * T scale rows,
# and prefill sizes.
T_CASES = [
    pytest.param(1, id="t1"),
    pytest.param(3, id="t3"),
    pytest.param(16, id="t16"),
    pytest.param(100, id="t100"),
    pytest.param(1536, id="t1536"),
    pytest.param(3072, id="t3072"),
]


# =====================================================================
# Reference implementation
# =====================================================================


def make_inputs(t, identity_mix):
    o = torch.randn([t, HEADS, HEAD_DIM], dtype=torch.bfloat16) * 2
    if identity_mix:
        mix = torch.eye(HEADS, dtype=torch.bfloat16)
    else:
        mix = (torch.eye(HEADS) + torch.randn(HEADS, HEADS) * 0.1).bfloat16()
    theta = torch.rand([MAX_POS, ROPE_DIM // 2]) * 2 * math.pi
    freqs = torch.stack([theta.cos(), theta.sin()], -1).flatten(-2).contiguous()
    positions = torch.randint(0, MAX_POS, [t], dtype=torch.int64)
    return o, mix, freqs, positions


def ref_mixed(o, mix, freqs, positions):
    """Inverse RoPE in fp32 rounded to bf16, then torch.matmul (bf16 out)."""
    f = freqs[positions].view(-1, 1, ROPE_DIM // 2, 2)
    rope = o[..., HEAD_DIM - ROPE_DIM :].float().view(-1, HEADS, ROPE_DIM // 2, 2)
    e, d = rope[..., 0], rope[..., 1]
    c, s = f[..., 0], f[..., 1]
    rot = torch.stack([e * c + d * s, d * c - e * s], -1).flatten(-2)
    o = torch.cat([o[..., : HEAD_DIM - ROPE_DIM], rot.to(torch.bfloat16)], -1)
    return torch.matmul(mix, o)


def ref_quant(y, group_major):
    """group_wise_quant of the [G * T, D] rows in the kernel's row order."""
    t = y.shape[0]
    y = y.view(t, GROUPS, -1)
    y = y.transpose(0, 1) if group_major else y
    return group_wise_quant(y.reshape(GROUPS * t, -1).contiguous(), mode="w8a8-mxfp8")


# =====================================================================
# Correctness tests
# =====================================================================


@pytest.mark.parametrize(
    "group_major", [True, False], ids=["group_major", "token_major"]
)
@pytest.mark.parametrize("t", T_CASES)
def test_vha_postmix_quant_identity_mix(t, group_major):
    """With mix = I each output has one non-zero term, so there is no reduction
    order and codes and scales equal inverse RoPE + group_wise_quant."""
    o, mix, freqs, positions = make_inputs(t, True)
    q, sf = vha_postmix_quant(o, mix, freqs, positions, GROUPS, group_major)
    ref_q, ref_sf = ref_quant(ref_mixed(o, mix, freqs, positions), group_major)
    assert torch.equal(q.view(torch.uint8).flatten(), ref_q.view(torch.uint8).flatten())
    assert torch.equal(sf, ref_sf)


@pytest.mark.parametrize("t", T_CASES)
def test_vha_postmix_quant_general_mix(t):
    """The tensor-core fp32 sum over 64 heads does not follow torch.matmul's
    order, so a bf16 mix result may move by 1 ulp. That moves a code by at most
    1, or rarely a group's amax across a power of two and with it the scale by
    1: codes match within 1 where the scale matches, and at most 0.1% of the
    scales differ, each by 1."""
    o, mix, freqs, positions = make_inputs(t, False)
    q, sf = vha_postmix_quant(o, mix, freqs, positions, GROUPS)
    ref_q, ref_sf = ref_quant(ref_mixed(o, mix, freqs, positions), True)
    sf_diff = (sf.int() - ref_sf.int()).abs()
    assert sf_diff.max() <= 1 and (sf_diff > 0).float().mean() <= 1e-3
    rows = GROUPS * t
    same_scale = (sf_diff == 0)[:rows].repeat_interleave(MXFP8_BLOCK_SIZE, -1)
    code_diff = ulp_diff(q.view(rows, -1), ref_q)
    assert code_diff[same_scale].max() <= 1


def test_vha_postmix_quant_preallocated_outputs_int32_positions():
    """int32 positions give the int64 result, written into the given buffers
    (scale padding rows are overwritten with 0)."""
    o, mix, freqs, positions = make_inputs(200, False)
    ref_q, ref_sf = vha_postmix_quant(o, mix, freqs, positions, use_pdl=False)
    out = torch.empty_like(ref_q)
    out_sf = torch.full_like(ref_sf, 0xFF)
    q, sf = vha_postmix_quant(
        o, mix, freqs, positions.int(), output_quant=out, output_sf=out_sf
    )
    assert q.data_ptr() == out.data_ptr() and sf.data_ptr() == out_sf.data_ptr()
    assert torch.equal(q.view(torch.uint8), ref_q.view(torch.uint8))
    assert torch.equal(sf, ref_sf)


# =====================================================================
# Error input tests
# =====================================================================


@pytest.mark.parametrize(
    "heads, groups, expected",
    [
        pytest.param(
            32, GROUPS, "o must be a contiguous bfloat16 [T, 64, 512]", id="bad_heads"
        ),
        pytest.param(HEADS, 3, "groups must divide 64, got 3", id="bad_groups"),
    ],
)
def test_vha_postmix_quant_errors(heads, groups, expected):
    o, mix, freqs, positions = make_inputs(4, True)
    with pytest.raises(RuntimeError, match=re.escape(expected)):
        vha_postmix_quant(o[:, :heads], mix, freqs, positions, groups)


if __name__ == "__main__":
    pytest.main([__file__])
