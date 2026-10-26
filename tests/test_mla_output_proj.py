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

from halops import halops_kernel
from halops.mla.mla_workspace import mla_workspace
from tests.utils.mla_reference import (
    LAYER_REL_TOL,
    OUTPUT_PROJ_REL_TOL,
    Dims,
    assert_rel_close,
    build_inputs,
    reference_attention,
    reference_output_proj,
    run_layer,
)

# (bs, draft): every projection tile, named by the T-only Plan of
# mla_output_proj.cu (swap-AB up to 128 tokens, pair UMMA tile per token group)
SHAPES = [
    pytest.param(1, 5, id="T5-swapab16", marks=pytest.mark.quick),
    pytest.param(4, 5, id="T20-swapab32"),
    pytest.param(10, 5, id="T50-swapab64"),
    pytest.param(13, 5, id="T65-swapab128", marks=pytest.mark.quick),
    pytest.param(16, 5, id="T80-swapab128"),
    pytest.param(43, 3, id="T129-pair128"),
    pytest.param(40, 5, id="T200-pair128", marks=pytest.mark.quick),
    pytest.param(60, 5, id="T300-pair160"),
    pytest.param(72, 5, id="T360-pair192"),
    pytest.param(80, 5, id="T400-pair224"),
    pytest.param(100, 5, id="T500-pair256", marks=pytest.mark.quick),
    pytest.param(119, 5, id="T595-pair320x2"),
    pytest.param(128, 5, id="T640-pair320x2"),
]
MODES = [
    pytest.param(0, False, id="swa"),
    pytest.param(128, True, id="hca"),
]

# hidden 1024 with 4 output groups: the smallest wo_b whose K is shorter than
# the swap-AB weight ring (no wo_b L2 prefetch left)
SMALL_DIMS = Dims(hidden=1024, o_groups=4, o_lora_rank=512)


# KV rows per token in these fixtures: 128 SWA + at most 128 compressed, i.e.
# at most 4 KV blocks of 64, below the plan's split threshold (KvSplit::
# MinBlocksToCut = 8 in mla_traits.cuh).  No token is KV-split, so q_or_x holds
# every token's O when mla_attention returns.
MAX_UNSPLIT_KV_ROWS = 256


def _layer(
    bs: int,
    draft: int,
    c128_len: int,
    fire: bool,
    *,
    attn_sink: bool = True,
    dims: Dims = Dims(),
):
    """Inputs of one layer and a closure running attention + projection."""
    args, aux = build_inputs(
        bs=bs,
        draft_tokens=draft,
        c128_len=c128_len,
        fire_compressor=fire,
        attn_sink=attn_sink,
        dims=dims,
    )
    return args, aux, run_layer(args, aux)


def _check_accuracy(args, aux, run) -> None:
    """The projection against the torch one on the kernel's own O (q_or_x, no
    token KV-split), and the layer output against the torch layer."""
    assert aux.swa_len + aux.c128_len <= MAX_UNSPLIT_KV_ROWS
    out = run()
    d = aux.dims
    o = args["q_or_x"].float().reshape(-1, d.b_h, d.d_q)
    assert_rel_close(out, reference_output_proj(o, args, aux), OUTPUT_PROJ_REL_TOL)
    ref = reference_output_proj(reference_attention(args, aux), args, aux)
    assert_rel_close(out, ref, LAYER_REL_TOL)


@pytest.mark.parametrize("c128_len,fire", MODES)
@pytest.mark.parametrize("bs,draft", SHAPES)
def test_accuracy(bs: int, draft: int, c128_len: int, fire: bool):
    """The projection and the layer output stay inside the fp8 noise band."""
    _check_accuracy(*_layer(bs, draft, c128_len, fire))


@pytest.mark.parametrize("c128_len,fire", MODES)
@pytest.mark.parametrize(
    "bs,draft", [pytest.param(1, 5, id="T5"), pytest.param(40, 5, id="T200")]
)
def test_no_sink(bs: int, draft: int, c128_len: int, fire: bool):
    """attn_sink=None leaves the sink out of the split-KV combine."""
    _check_accuracy(*_layer(bs, draft, c128_len, fire, attn_sink=False))


@pytest.mark.parametrize(
    "bs,draft", [pytest.param(1, 5, id="T5"), pytest.param(40, 5, id="T200")]
)
def test_small_hidden(bs: int, draft: int):
    """A wo_b K shorter than the weight ring still projects correctly."""
    _check_accuracy(*_layer(bs, draft, 128, True, dims=SMALL_DIMS))


def test_strided_hidden_states():
    """hidden_states as a column slice of a wider buffer (row stride > H) gives
    the output of its contiguous copy."""
    args, _, run = _layer(13, 5, 128, False)
    first = run()
    x = args["hidden_states"]
    wide = torch.zeros((x.shape[0], x.shape[1] + 64), dtype=x.dtype)
    wide[:, : x.shape[1]] = x
    args["hidden_states"] = wide[:, : x.shape[1]]
    assert torch.equal(run(), first)


@pytest.mark.parametrize(
    "bs,draft", [pytest.param(13, 5, id="T65"), pytest.param(100, 5, id="T500")]
)
def test_deterministic(bs: int, draft: int):
    """Back-to-back launches on one plan are bit-identical (the projection
    resets the attention's barrier banks)."""
    *_, run = _layer(bs, draft, 128, True)
    first = run()
    for _ in range(3):
        assert torch.equal(run(), first)


O_TOTAL = Dims().b_h * Dims().d_q


def _weights(hidden: int, group_k: int):
    """Uninitialized fp8 wo_a [2 H, group_k] / wo_b [H, 2 H] with their scales;
    the host checks only read shapes and dtypes."""
    wo_a = torch.empty((2 * hidden, group_k), dtype=torch.float8_e4m3fn)
    wo_b = torch.empty((hidden, 2 * hidden), dtype=torch.float8_e4m3fn)
    return dict(
        wo_a_weight=wo_a,
        wo_a_sf=torch.empty(wo_a.numel() // 32, dtype=torch.uint8),
        wo_b_weight=wo_b,
        wo_b_sf=torch.empty(wo_b.numel() // 32, dtype=torch.uint8),
    )


def _with_tokens(args: dict, tokens: int) -> None:
    hidden = args["out"].shape[1]
    args["q_or_x"] = torch.zeros((tokens, O_TOTAL), dtype=torch.bfloat16)
    args["positions"] = torch.zeros(tokens, dtype=torch.int32)
    args["out"] = torch.empty((tokens, hidden), dtype=torch.bfloat16)


def _with_weights(args: dict, hidden: int, group_k: int) -> None:
    args.update(_weights(hidden, group_k))
    args["out"] = torch.empty((args["out"].shape[0], hidden), dtype=torch.bfloat16)


def _workspace(tokens: int, weights: dict, hidden: int = 1024) -> torch.Tensor:
    """`mla_workspace` for uninitialized wq_a / wkv of hidden size H and the
    given wo_a / wo_b."""
    wq_a = torch.empty((1024, hidden), dtype=torch.float8_e4m3fn)
    wkv = torch.empty((512, hidden), dtype=torch.float8_e4m3fn)
    return mla_workspace(
        tokens, wq_a, wkv, weights["wo_a_weight"], weights["wo_b_weight"]
    )


@pytest.fixture
def proj_args():
    """`mla_output_proj` arguments for T = 200, H = 1024 and 4 output groups
    that pass every host check."""
    tokens, hidden, groups = 200, 1024, 4
    weights = _weights(hidden, O_TOTAL // groups)
    args = dict(
        freqs_real=torch.zeros((1, 64), dtype=torch.float32),
        workspace=_workspace(tokens, weights),
        attn_sink=None,
        c128_attn_len=None,
        out=torch.empty((tokens, hidden), dtype=torch.bfloat16),
        **weights,
    )
    _with_tokens(args, tokens)
    return args


def _small_workspace(args):
    args["workspace"] = _workspace(5, args)


INVALID_PROJ = [
    pytest.param(lambda a: _with_tokens(a, 0), r"T = 0 not in \[1, 640\]", id="T0"),
    pytest.param(
        lambda a: _with_tokens(a, 641), r"T = 641 not in \[1, 640\]", id="T641"
    ),
    pytest.param(_small_workspace, "workspace needs", id="ws-small"),
    pytest.param(
        lambda a: _with_weights(a, 1024, 0), "wo_a K = 0 must split", id="wo_a-K0"
    ),
    pytest.param(
        lambda a: _with_weights(a, 1024, 256),
        "wo_a K = 256 must split",
        id="wo_a-K256",
    ),
    pytest.param(
        lambda a: _with_weights(a, 1024, 3000),
        "wo_a K = 3000 must split",
        id="wo_a-K3000",
    ),
    pytest.param(
        lambda a: _with_weights(a, 256, 2 * O_TOTAL),
        f"wo_a K = {2 * O_TOTAL} must split",
        id="wo_a-K-gt-O",
    ),
    pytest.param(
        lambda a: a.update(wo_b_weight=a["wo_b_weight"][:768]),
        "needs wo_a \\[2 H, K\\], wo_b \\[H, 2 H\\]",
        id="wo_a-not-twice-wo_b",
    ),
    pytest.param(
        lambda a: _with_weights(a, 4608, O_TOTAL // 4),
        "wo_a has more than 64 128-row tiles",
        id="hidden-too-large",
    ),
    pytest.param(
        lambda a: a.update(wo_a_sf=a["wo_a_sf"].view(torch.int8)),
        "weights must be float8_e4m3fn with uint8 SfAtom scales",
        id="wo_a_sf-dtype",
    ),
    pytest.param(
        lambda a: a.update(wo_b_sf=a["wo_b_sf"][:-1]),
        "weights must be float8_e4m3fn with uint8 SfAtom scales",
        id="wo_b_sf-numel",
    ),
    pytest.param(
        lambda a: a.update(wo_b_weight=a["wo_b_weight"].view(torch.uint8)),
        "weights must be float8_e4m3fn with uint8 SfAtom scales",
        id="wo_b-dtype",
    ),
    pytest.param(
        lambda a: a.update(
            q_or_x=torch.zeros((200, O_TOTAL // 2), dtype=torch.bfloat16)
        ),
        "q_or_x must be contiguous bf16",
        id="q_or_x-width",
    ),
    pytest.param(
        lambda a: a.update(
            q_or_x=torch.zeros((O_TOTAL, 200), dtype=torch.bfloat16).t()
        ),
        "q_or_x must be contiguous bf16",
        id="q_or_x-strided",
    ),
    pytest.param(
        lambda a: a.update(q_or_x=a["q_or_x"].float()),
        "q_or_x must be contiguous bf16",
        id="q_or_x-dtype",
    ),
    pytest.param(
        lambda a: a.update(out=torch.empty((200, 512), dtype=torch.bfloat16)),
        "out must be contiguous bf16",
        id="out-width",
    ),
    pytest.param(
        lambda a: a.update(out=a["out"].float()),
        "out must be contiguous bf16",
        id="out-dtype",
    ),
    pytest.param(
        lambda a: a.update(freqs_real=torch.zeros((1, 32), dtype=torch.float32)),
        "freqs_real must be contiguous float32",
        id="freqs-width",
    ),
    pytest.param(
        lambda a: a.update(attn_sink=torch.zeros(32, dtype=torch.float32)),
        "attn_sink must be float32",
        id="sink-size",
    ),
]


# invalid inputs owned by one operator's host check (the rest are checked by both)
PROJECTION_ONLY = {
    "wo_a_sf-dtype",
    "wo_b_sf-numel",
    "wo_b-dtype",
    "out-width",
    "out-dtype",
}
COMBINE_ONLY = {
    "q_or_x-width",
    "q_or_x-strided",
    "q_or_x-dtype",
    "freqs-width",
    "sink-size",
}


def _combine(a: dict) -> None:
    halops_kernel.combine_with_rope_quant(
        a["q_or_x"],
        a["positions"],
        a["freqs_real"],
        a["attn_sink"],
        a["c128_attn_len"],
        a["wo_a_weight"],
        a["wo_b_weight"],
        a["workspace"],
    )


def _projection(a: dict) -> None:
    halops_kernel.mla_output_proj(
        a["wo_a_weight"],
        a["wo_a_sf"],
        a["wo_b_weight"],
        a["wo_b_sf"],
        a["out"],
        a["workspace"],
    )


@pytest.mark.quick
@pytest.mark.parametrize("mutate,match", INVALID_PROJ)
def test_invalid_inputs(request, proj_args: dict, mutate, match: str):
    """Each operator rejects the invalid inputs it owns in its host check,
    before its launch."""
    case = request.node.callspec.id
    mutate(proj_args)
    ops = []
    if case not in PROJECTION_ONLY:
        ops.append(_combine)
    if case not in COMBINE_ONLY:
        ops.append(_projection)
    for op in ops:
        with pytest.raises(RuntimeError, match=match):
            op(proj_args)


INVALID_WORKSPACE = [
    pytest.param(0, 1024, O_TOTAL // 4, r"T = 0 not in \[1, 640\]", id="T0"),
    pytest.param(641, 1024, O_TOTAL // 4, r"T = 641 not in \[1, 640\]", id="T641"),
    pytest.param(16, 1024, 0, "wo_a K = 0 must split", id="wo_a-K0"),
    pytest.param(16, 2048, O_TOTAL // 4, "wo_b has 2048 rows", id="hidden-mismatch"),
]


@pytest.mark.quick
@pytest.mark.parametrize("tokens,proj_hidden,group_k,match", INVALID_WORKSPACE)
def test_invalid_workspace(tokens: int, proj_hidden: int, group_k: int, match: str):
    """Workspace sizing rejects T outside [1, 640] (T = 0 instead of dividing
    by it), a bad wo_a and a wo_b whose H is not the attention's."""
    with pytest.raises(RuntimeError, match=match):
        _workspace(tokens, _weights(proj_hidden, group_k))
