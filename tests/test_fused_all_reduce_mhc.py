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

import os
import re
import subprocess
import sys

import pytest
import torch
import torch.distributed as dist

from halops import (
    fused_all_reduce_mhc,
    get_workspace_for_mhc,
    mhc_post_hc_mean,
    mhc_sinkhorn,
)
from tests.conftest import SEED
from tests.utils.gemm_utils import MXFP8_BLOCK_SIZE, ue8m0_to_float32

EPS = 1e-6
POST_MULT = 2.0
REPEAT = 20
FP8_E4M3_MAX = 448.0
MIX_TOL = dict(atol=1e-3, rtol=1e-3)
LAYER_TOL = dict(atol=2e-2, rtol=2e-2)
HIDDEN = 4096
TP_SIZE = 4
TP_CASES = [(1, 4), (16, 4), (5, 2)]
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_inputs(t, hc, hidden=HIDDEN):
    mix = hc * (hc + 2)
    return (
        torch.randn(t, hidden).bfloat16(),
        torch.randn(t, hc, hidden).bfloat16(),
        torch.rand(t, hc) * 2,
        torch.randn(t, hc, hc).softmax(-1),
        torch.randn(mix, hc * hidden) * 0.05,
        torch.rand(3) * 0.3 + 0.05,
        torch.randn(mix),
        (torch.rand(hidden) + 0.5).bfloat16(),
    )


def run_mhc(inputs, **kwargs):
    return fused_all_reduce_mhc(
        *inputs, EPS, EPS, EPS, POST_MULT, REPEAT, EPS, **kwargs
    )


def ref_post(x, residual, post, comb):
    mixed = torch.einsum("tij,tih->tjh", comb.double(), residual.double())
    return (post.double()[..., None] * x[:, None].double() + mixed).bfloat16()


def ref_sinkhorn(logits):
    x = logits.softmax(-1) + EPS
    x = x / (x.sum(-2, keepdim=True) + EPS)
    for _ in range(REPEAT - 1):
        x = x / (x.sum(-1, keepdim=True) + EPS)
        x = x / (x.sum(-2, keepdim=True) + EPS)
    return x


def ref_pre(residual, fn, hc_scale, hc_base, norm_weight):
    hc = residual.shape[1]
    r = residual.double()
    rms = 1 / (r.square().mean((1, 2)).sqrt() + EPS)
    mixes = (r.flatten(1) @ fn.double().T) * rms[:, None]
    pre, post, logits = mixes.split((hc, hc, hc * hc), -1)
    pre_base, post_base, comb_base = hc_base.double().split((hc, hc, hc * hc))
    pre = torch.sigmoid(pre * hc_scale[0] + pre_base) + EPS
    post = torch.sigmoid(post * hc_scale[1] + post_base) * POST_MULT
    logits = (logits * hc_scale[2] + comb_base).unflatten(-1, (hc, hc))
    y = (pre[..., None] * r).sum(1)
    rn = torch.rsqrt(y.square().mean(-1, keepdim=True) + EPS)
    layer = (y.bfloat16().double() * rn * norm_weight.double()).bfloat16()
    return post, logits, ref_sinkhorn(logits), layer


def ref_mxfp8(x):
    g = x.float().unflatten(-1, (-1, MXFP8_BLOCK_SIZE))
    m, e = torch.frexp(g.abs().amax(-1).clamp(min=1e-10) / FP8_E4M3_MAX)
    sf = (e + 126 + (m != 0.5)).to(torch.uint8)
    q = (g / ue8m0_to_float32(sf)[..., None]).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    return q.flatten(-2).to(torch.float8_e4m3fn), sf


def assert_equal_outputs(outs, refs, case=""):
    for i, (out, ref) in enumerate(zip(outs, refs)):
        same = torch.equal(out.view(torch.uint8), ref.view(torch.uint8))
        assert same, f"{case} output {i} differs"


@pytest.mark.parametrize(
    "t, hc",
    [
        pytest.param(1, 4, id="t1-hc4"),
        pytest.param(64, 4, id="t64-hc4"),
        pytest.param(3, 2, id="t3-hc2"),
    ],
)
def test_fused_all_reduce_mhc(t, hc):
    inputs = make_inputs(t, hc)
    residual_out, layer_out, post_out, comb_out, q, sf = run_mhc(inputs)
    residual = ref_post(*inputs[:4])
    post, _, comb, layer = ref_pre(residual, *inputs[4:])
    torch.testing.assert_close(residual_out, residual)
    torch.testing.assert_close(post_out, post.float(), **MIX_TOL)
    torch.testing.assert_close(comb_out, comb.float(), **MIX_TOL)
    torch.testing.assert_close(layer_out, layer, **LAYER_TOL)
    assert_equal_outputs((q, sf), ref_mxfp8(layer_out))


@pytest.mark.parametrize("hc", [2, 4])
def test_fused_all_reduce_mhc_logits(hc):
    inputs = make_inputs(4, hc)
    outs = run_mhc(inputs)
    logit_outs = run_mhc(inputs, return_logits=True)
    _, logits, _, _ = ref_pre(ref_post(*inputs[:4]), *inputs[4:])
    torch.testing.assert_close(logit_outs[3], logits.float(), **MIX_TOL)
    comb = mhc_sinkhorn(logit_outs[3], EPS, REPEAT, prefetch=inputs[4:])
    assert_equal_outputs(outs, logit_outs[:3] + (comb,) + logit_outs[4:])

    raw = torch.randn(4, hc, hc)
    fused = run_mhc(inputs[:3] + (raw,) + inputs[4:], comb_is_logits=True)
    ref = run_mhc(inputs[:3] + (mhc_sinkhorn(raw, EPS, REPEAT),) + inputs[4:])
    assert_equal_outputs(fused, ref)


def test_fused_all_reduce_mhc_workspace():
    inputs = make_inputs(16, 4)
    ref = run_mhc(inputs)
    workspace = get_workspace_for_mhc(16, HIDDEN, "", 0, 1)
    for _ in range(3):
        assert_equal_outputs(run_mhc(inputs, workspace=workspace), ref)


@pytest.mark.parametrize(
    "t, hc, hidden",
    [
        pytest.param(1, 4, 4096, id="t1-hc4-h4096"),
        pytest.param(16, 2, 1536, id="t16-hc2-h1536"),
    ],
)
def test_mhc_post_hc_mean(t, hc, hidden):
    x, residual, post, comb = make_inputs(t, hc, hidden)[:4]
    out = mhc_post_hc_mean(x, residual, post, comb)
    ref = ref_post(x, residual, post, comb).float().mean(1).bfloat16()
    torch.testing.assert_close(out, ref)


@pytest.mark.parametrize(
    "hc, hidden, message",
    [
        pytest.param(5, 4096, "hc must be 2 or 4, got 5", id="hc5"),
        pytest.param(4, 1024, "hidden support 4096 only, got 1024", id="h1024"),
    ],
)
def test_fused_all_reduce_mhc_errors(hc, hidden, message):
    with pytest.raises(RuntimeError, match=re.escape(message)):
        run_mhc(make_inputs(1, hc, hidden))


def test_fused_all_reduce_mhc_tp():
    if torch.cuda.device_count() < TP_SIZE:
        pytest.skip(f"needs {TP_SIZE} GPUs")
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={TP_SIZE}",
        os.path.abspath(__file__),
    ]
    env = dict(os.environ, PYTHONPATH=_ROOT)
    proc = subprocess.run(
        cmd, cwd=_ROOT, env=env, capture_output=True, text=True, timeout=600
    )
    assert proc.returncode == 0, (proc.stdout + proc.stderr)[-4000:]


def tp_worker():
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    torch.set_default_device(dev)
    dist.init_process_group("nccl", device_id=dev)
    group = dist.group.WORLD.group_name
    for t, hc in TP_CASES:
        torch.manual_seed(SEED)
        partials = torch.randn(world, t, HIDDEN).bfloat16()
        inputs = make_inputs(t, hc)
        x_sum = partials[0].float()
        for partial in partials[1:]:
            x_sum = x_sum + partial.float()
        ref = run_mhc((x_sum.bfloat16(),) + inputs[1:])
        workspace = get_workspace_for_mhc(t, HIDDEN, group, rank, world)
        for _ in range(3):
            outs = run_mhc((partials[rank],) + inputs[1:], workspace=workspace)
            assert_equal_outputs(outs, ref, f"rank {rank} t{t} hc{hc}")
    dist.destroy_process_group()


if __name__ == "__main__":
    if "WORLD_SIZE" in os.environ:
        tp_worker()
    else:
        pytest.main([__file__])
