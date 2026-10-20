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

The shape ladder and the tolerance gate shared by `test_mega_mla_swa.py` and
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pytest
import torch

from tests.mega_mla_reference import Dims, build_inputs, reference_forward

SM100 = (10, 0)


class Mode(NamedTuple):
    """One scenario: how much compressed KV a token attends over, and whether the"""

    name: str
    c128_len: int
    fire: bool
    swa_len: int = 128


class Case(NamedTuple):
    """One (bs, draft_tokens) shape.  Each is the smallest T exercising one thing."""

    bs: int
    draft_tokens: int
    tokens: int
    exercises: str
    quick: bool


CASES = [
    Case(1, 5, 5, "deep K-split, one wave; BN=16", quick=True),
    Case(12, 5, 60, "the largest T still on UMMA_N_PAD=32", quick=False),
    Case(13, 5, 65, "the n_groups 1->2 cliff AND the N_PAD 32->64 switch", quick=True),
    Case(24, 5, 120, "N_PAD=64 with a full second group", quick=False),
    Case(26, 5, 130, "the first T past 2 token groups", quick=True),
    Case(40, 5, 200, "N_PAD 64->128 AND the first T past gridDim.x=152", quick=True),
]


def skip_reason() -> str | None:
    if not torch.cuda.is_available():
        return "no CUDA device"
    cap = torch.cuda.get_device_capability()
    if cap != SM100:
        return f"mega_mla is sm100-only; this device is sm_{cap[0]}{cap[1]}"
    return None


def params(modes: list[Mode]):
    """The (mode, case) product for one suite; `quick` rides along as a marker."""
    return [
        pytest.param(
            mode,
            case,
            marks=[pytest.mark.quick] if case.quick else [],
            id=f"{mode.name}-T{case.tokens}",
        )
        for mode in modes
        for case in CASES
    ]


def check(
    mode: Mode, case: Case, *, assert_rel: float = 0.05, dims: Dims = Dims()
) -> float:
    """Build one fixture, run both sides, assert the tolerance.  Returns rel max_diff."""
    from halops.mla.mega_mla import mega_mla

    args, aux = build_inputs(
        bs=case.bs,
        draft_tokens=case.draft_tokens,
        swa_len=mode.swa_len,
        c128_len=mode.c128_len,
        fire_compressor=mode.fire,
        seed=0,
        dims=dims,
    )
    assert (
        args["out"].shape[0] == case.tokens
    ), f"built T={args['out'].shape[0]}, case declares {case.tokens}"

    mega_mla(**args)
    ref = reference_forward(args, aux).float().cpu().numpy()
    out = args["out"].float().cpu().numpy()

    blocks = -(-mode.swa_len // dims.b_topk) + -(-mode.c128_len // dims.b_topk)

    scale = max(float(np.abs(ref).max()), 1e-6)
    rel = float(np.abs(ref - out).max()) / scale
    print(
        f"[{mode.name}] swa={mode.swa_len} c128={mode.c128_len} blocks={blocks} "
        f"T={case.tokens} "
        f"rel max_diff = {rel:.6f}  ({case.exercises})"
    )

    assert np.isfinite(out).all(), "the megakernel produced non-finite values"
    np.testing.assert_allclose(out, ref, rtol=0, atol=assert_rel * scale)
    return rel
