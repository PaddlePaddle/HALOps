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

from halops import group_wise_quant, sigmoid_gate_mul_group_quant


@pytest.mark.parametrize(
    "t, hidden, gate_stride",
    [
        pytest.param(1, 4096, 1, id="t1-h4096"),
        pytest.param(5, 8192, 2, id="t5-h8192-strided"),
        pytest.param(64, 4096, 2, id="t64-h4096-strided"),
    ],
)
def test_sigmoid_gate_mul_group_quant(t, hidden, gate_stride):
    o = torch.randn(t, hidden).bfloat16()
    gate = torch.randn(t, hidden * gate_stride).bfloat16()[:, :hidden]
    q, sf = sigmoid_gate_mul_group_quant(o, gate)
    y = (o.float() * torch.sigmoid(gate.float()).bfloat16().float()).bfloat16()
    ref_q, ref_sf = group_wise_quant(y, use_swizzle=False)
    assert torch.equal(q.view(torch.uint8), ref_q.view(torch.uint8))
    assert torch.equal(sf, ref_sf)


def test_sigmoid_gate_mul_group_quant_errors():
    o = torch.randn(2, 48).bfloat16()
    message = "hidden must be a positive multiple of 32, got 48"
    with pytest.raises(RuntimeError, match=re.escape(message)):
        sigmoid_gate_mul_group_quant(o, o)


if __name__ == "__main__":
    pytest.main([__file__])
