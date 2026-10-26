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

import pytest
import torch

from halops.mla.mla_attention import mla_attention
from tests.utils.mla_reference import (
    Case,
    Dims,
    Mode,
    build_inputs,
    check_attention,
    layer_workspace,
    params,
    plan_layer,
)

MODES = [Mode(name="swa", c128_len=0, fire=False)]


@pytest.mark.parametrize("attn_sink", [True, False], ids=["sink", "no_sink"])
@pytest.mark.parametrize("mode,case", params(MODES))
def test_accuracy(mode: Mode, case: Case, attn_sink: bool):
    """Every shape stays inside the fp8 noise band."""
    check_attention(mode, case, attn_sink=attn_sink)


Q_TOTAL = Dims().b_h * Dims().d_q


@pytest.fixture
def attention_args():
    """Valid SWA `mla_attention` arguments for T = 5, planned workspace
    included."""
    args, aux = build_inputs(bs=1, draft_tokens=5)
    args["workspace"] = layer_workspace(args, aux, 5)
    plan_layer(args, aux, args["workspace"])
    return args


def _misaligned_hidden(args: dict) -> None:
    x = args["hidden_states"]
    wide = torch.zeros((x.shape[0], x.shape[1] + 8), dtype=x.dtype)
    args["hidden_states"] = wide[:, 1 : 1 + x.shape[1]]


INVALID_ATTENTION = [
    pytest.param(
        lambda a: a.update(workspace=a["workspace"][:1]),
        "mla_attention: workspace needs",
        id="ws-small",
    ),
    pytest.param(
        lambda a: a.update(q_or_x=torch.zeros((0, Q_TOTAL), dtype=torch.bfloat16)),
        "q_or_x must be contiguous bf16 \\[T > 0",
        id="T0",
    ),
    pytest.param(
        lambda a: a.update(q_or_x=torch.zeros((5, Q_TOTAL // 2), dtype=torch.bfloat16)),
        "q_or_x must be contiguous bf16",
        id="q_or_x-width",
    ),
    pytest.param(
        lambda a: a.update(q_or_x=a["q_or_x"].float()),
        "q_or_x must be contiguous bf16",
        id="q_or_x-dtype",
    ),
    pytest.param(_misaligned_hidden, "16 B aligned rows", id="hidden-misaligned"),
    pytest.param(
        lambda a: a.update(hidden_states=a["hidden_states"][:, :-32]),
        "hidden_states must be bf16",
        id="hidden-width",
    ),
    pytest.param(
        lambda a: a.update(wq_a_weight=a["wq_a_weight"][:512]),
        "needs wq_a \\[1024, H\\], wkv \\[512, H\\]",
        id="wq_a-rows",
    ),
    pytest.param(
        lambda a: a.update(wkv_weight=a["wkv_weight"][:, :1024].contiguous()),
        "needs wq_a \\[1024, H\\], wkv \\[512, H\\]",
        id="wkv-width",
    ),
    pytest.param(
        lambda a: a.update(wq_b_weight=a["wq_b_weight"][:1024]),
        f"wq_b must be \\[{Q_TOTAL}, 1024\\]",
        id="wq_b-rows",
    ),
    pytest.param(
        lambda a: a.update(wq_a_sf=a["wq_a_sf"].view(torch.int8)),
        "weights must be float8_e4m3fn with uint8 SfAtom scales",
        id="wq_a_sf-dtype",
    ),
    pytest.param(
        lambda a: a.update(wq_b_sf=a["wq_b_sf"][:-1]),
        "weights must be float8_e4m3fn with uint8 SfAtom scales",
        id="wq_b_sf-numel",
    ),
    pytest.param(
        lambda a: a.update(wkv_weight=a["wkv_weight"].view(torch.uint8)),
        "weights must be float8_e4m3fn with uint8 SfAtom scales",
        id="wkv-dtype",
    ),
    pytest.param(
        lambda a: a.update(freqs_real=a["freqs_real"][:, :32].contiguous()),
        "freqs_real must be contiguous float32",
        id="freqs-width",
    ),
    pytest.param(
        lambda a: a.update(attn_sink=torch.zeros(32, dtype=torch.float32)),
        "attn_sink must be float32",
        id="sink-size",
    ),
    pytest.param(
        lambda a: a.update(
            wkv_gate_weight=torch.zeros(
                (1024, a["hidden_states"].shape[1]), dtype=torch.bfloat16
            )
        ),
        "wkv_gate_weight and c128_attn_len must be given together",
        id="kv_gate-without-c128",
    ),
    pytest.param(
        lambda a: a.update(
            wkv_gate_weight=torch.zeros(
                (1024, a["hidden_states"].shape[1]), dtype=torch.bfloat16
            ),
            c128_attn_len=torch.zeros(5, dtype=torch.int32),
        ),
        "wkv_gate_weight needs kv_score_ring and kv_gate_ring_row",
        id="kv_gate-without-ring",
    ),
    pytest.param(
        lambda a: a.update(c128_ape=torch.zeros((128, 512), dtype=torch.float32)),
        "c128_ape needs c128_norm_weight",
        id="c128_ape-alone",
    ),
]


@pytest.mark.quick
@pytest.mark.parametrize("mutate,match", INVALID_ATTENTION)
def test_invalid_inputs(attention_args: dict, mutate, match: str):
    """Inputs outside the host checks are rejected before the launch."""
    mutate(attention_args)
    with pytest.raises(RuntimeError, match=match):
        mla_attention(**attention_args)
