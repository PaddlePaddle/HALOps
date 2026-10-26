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

from halops.mla.mla_attention_plan import mla_attention_plan
from halops.mla.mla_workspace import mla_workspace
from tests.utils.mla_reference import Dims, reference_kv_runs

# small pages and windows so that runs break on page boundaries; the long
# window spans more threads than one plan block holds
GEOMETRIES = [(16, 24), (512, 600)]  # (page_size, topk)
RANDOM_TOKENS = 200
# random windows: up to this many runs, starts below this slot
MAX_RUNS = 3
MAX_START = 4096


def _workspace(max_tokens: int) -> torch.Tensor:
    """A workspace for the DSV4 weight shapes (contents unused)."""
    d = Dims()

    def fp8(rows: int, cols: int) -> torch.Tensor:
        return torch.empty((rows, cols), dtype=torch.float8_e4m3fn)

    return mla_workspace(
        max_tokens,
        fp8(d.qlora_rank, d.hidden),
        fp8(d.kvlora_rank, d.hidden),
        fp8(d.o_groups * d.o_lora_rank, d.b_h * d.d_q // d.o_groups),
        fp8(d.hidden, d.o_groups * d.o_lora_rank),
    )


def _window(runs: list[tuple[int, int]], step: int, topk: int) -> list[int]:
    """Slots of consecutive (start, rows) runs scanned with `step`, -1 padded."""
    slots = [start + step * i for start, rows in runs for i in range(rows)]
    return (slots + [-1] * topk)[:topk]


def _edge_rows(page: int, topk: int, step: int):
    """(window, length) rows: one or two runs, page breaks, degenerate
    windows, -1 padding, lengths 0 and 1."""
    mid = 3 * page + page // 2  # mid page, scanning either way
    edge = 4 * page - 1 if step > 0 else 4 * page  # last row before a break
    return [
        (_window([(mid, 4)], step, topk), 4),
        (_window([(edge - 2 * step, 6)], step, topk), 6),
        (_window([(mid, 3), (9 * page, 5)], step, topk), 8),
        (_window([(mid, 3), (9 * page, 3), (20 * page, 2)], step, topk), 8),
        (_window([(mid, 3), (9 * page, 3), (20 * page, 2)], step, topk), 6),
        (_window([(edge - step, 2 * page + 2)], step, topk), topk),
        ([-1] * topk, 3),
        ([-1] * topk, 2),
        (_window([(mid, 5)], step, topk), 0),
        (_window([(mid, 5)], step, topk), 1),
        (_window([(mid, topk)], step, topk), topk),
    ]


def _random_rows(page: int, topk: int, step: int, count: int):
    """(window, length) rows of 1..MAX_RUNS random runs."""
    rows = []
    for _ in range(count):
        n_runs = int(torch.randint(1, MAX_RUNS + 1, ()))
        lens = torch.randint(1, max(topk // n_runs, 1) + 1, (n_runs,)).tolist()
        starts = torch.randint(page, MAX_START, (n_runs,)).tolist()
        window = _window(list(zip(starts, lens)), step, topk)
        rows.append((window, int(torch.randint(0, topk + 1, ()))))
    return rows


@pytest.mark.quick
@pytest.mark.parametrize("uniform", [False, True], ids=["lengths", "uniform"])
@pytest.mark.parametrize("hca", [False, True], ids=["swa", "hca"])
@pytest.mark.parametrize("step", [1, -1], ids=["up", "down"])
@pytest.mark.parametrize("page_size,topk", GEOMETRIES)
def test_kv_runs(page_size: int, topk: int, step: int, hca: bool, uniform: bool):
    """The plan's KV runs and effective lengths equal the torch reference."""
    rows = _edge_rows(page_size, topk, step)
    rows += _random_rows(page_size, topk, step, RANDOM_TOKENS)
    kv_indices = torch.tensor([w for w, _ in rows], dtype=torch.int32)
    lengths = None
    if not uniform:
        lengths = torch.tensor([n for _, n in rows], dtype=torch.int32)
    T = kv_indices.shape[0]
    c128_attn_len = torch.zeros(T, dtype=torch.int32) if hca else None
    runs = mla_attention_plan(
        kv_indices, lengths, page_size, step, c128_attn_len, _workspace(T)
    )
    ref = reference_kv_runs(kv_indices, lengths, page_size, step)
    assert torch.equal(runs, ref)


@pytest.fixture
def plan_args():
    """Valid `mla_attention_plan` arguments for T = 5."""
    T, topk = 5, 8
    return dict(
        kv_indices=torch.arange(T * topk, dtype=torch.int32).reshape(T, topk),
        topk_lengths=torch.full((T,), topk, dtype=torch.int32),
        page_size=16,
        kv_run_step=1,
        c128_attn_len=None,
        workspace=_workspace(T),
    )


INVALID_PLAN = [
    pytest.param(
        lambda a: a.update(workspace=a["workspace"][:1]),
        "mla_attention_plan: workspace needs",
        id="ws-small",
    ),
    pytest.param(
        lambda a: a.update(kv_indices=a["kv_indices"][:0]),
        "kv_indices must be int32 \\[T > 0",
        id="T0",
    ),
    pytest.param(
        lambda a: a.update(kv_indices=a["kv_indices"].long()),
        "kv_indices must be int32",
        id="kv_indices-dtype",
    ),
    pytest.param(
        lambda a: a.update(kv_indices=a["kv_indices"][:, ::2]),
        "with unit column stride",
        id="kv_indices-stride",
    ),
    pytest.param(
        lambda a: a.update(topk_lengths=a["topk_lengths"][:-1]),
        "match topk_lengths",
        id="topk_lengths-rows",
    ),
    pytest.param(
        lambda a: a.update(c128_attn_len=torch.zeros(4, dtype=torch.int32)),
        "match topk_lengths / c128_attn_len",
        id="c128_attn_len-rows",
    ),
    pytest.param(
        lambda a: a.update(page_size=24),
        "page_size must be a power of two",
        id="page_size",
    ),
    pytest.param(
        lambda a: a.update(kv_run_step=2),
        "kv_run_step must be 1 or -1",
        id="kv_run_step",
    ),
]


@pytest.mark.quick
@pytest.mark.parametrize("mutate,match", INVALID_PLAN)
def test_invalid_inputs(plan_args: dict, mutate, match: str):
    """Inputs outside the host checks are rejected before the launch."""
    mutate(plan_args)
    with pytest.raises(RuntimeError, match=match):
        mla_attention_plan(**plan_args)
