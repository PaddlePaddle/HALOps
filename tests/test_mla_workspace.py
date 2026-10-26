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

from tests.utils.mla_reference import (
    Dims,
    build_inputs,
    layer_workspace,
    plan_layer,
    run_layer,
)

# 448 SWA rows = 7 KV blocks of 64, one below the plan's split threshold
# (KvSplit::MinBlocksToCut in mla_traits.cuh): SWA tokens stay unsplit while
# HCA tokens (+ 2 c128 blocks) are KV-split, so each kind's combine needs its
# own TokenSplit table.  One page per window keeps it a single kv run.
SWA_LEN = 448
C128_LEN = 128
DIMS = Dims(page_size=512)
# the workspace is sized above the layers' T
MAX_TOKENS = 64


@pytest.mark.quick
def test_interleaved_kinds():
    """SWA, HCA, SWA, HCA layers on one workspace, after one plan per kind,
    give bit for bit the outputs of each layer on its own workspace."""
    layers = [
        build_inputs(
            bs=1,
            draft_tokens=5,
            swa_len=SWA_LEN,
            c128_len=c128_len,
            fire_compressor=c128_len > 0,
            dims=DIMS,
        )
        for c128_len in (0, C128_LEN, 0, C128_LEN)
    ]
    refs = [run_layer(args, aux)() for args, aux in layers]
    workspace = layer_workspace(*layers[0], MAX_TOKENS)
    for args, aux in layers[:2]:
        plan_layer(args, aux, workspace)
    for (args, aux), ref in zip(layers, refs):
        assert torch.equal(run_layer(args, aux, workspace)(), ref)
