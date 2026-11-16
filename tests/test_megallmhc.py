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

import pytest
import torch

from halops import group_wise_quant
from halops.mhc.megallmhc import megallmhc

HC = 4
HIDDEN = 4096
HC3 = HC * (2 + HC)
LOG2E = math.log2(math.e)
EPS = 1e-6
POST_MULT = 2.0
SINKHORN_REPEAT = 20
TOL = dict(post=(4e-7, 1e-6), comb=(5e-7, 1.5e-6), layer=(2.5e-4, 1.6e-2))
MIN_COS = 0.99999
TOKENS = [1, 2, 7, 8, 16, 28, 29, 64, 128, 160, 255, 256, 512]


def _kwargs(fleet):
    return dict(
        rms_eps=EPS,
        hc_pre_eps=EPS,
        hc_sinkhorn_eps=EPS,
        hc_post_mult_value=POST_MULT,
        sinkhorn_repeat=SINKHORN_REPEAT,
        norm_eps=EPS,
        mhc_align_fleet=fleet,
    )


def _round_tf32(x):
    u = x.contiguous().view(torch.int32)
    return ((u + 0x1000) & ~0x1FFF).view(torch.float32)


def _fma(a, b, c):
    return (a.double() * b.double() + c.double()).float()


def _sigmoid(z):
    return 1.0 / (1.0 + torch.exp(-z))


def _row_sum(c):
    return (c[..., 0:1] + c[..., 2:3]) + (c[..., 1:2] + c[..., 3:4])


def _col_sum(c):
    return (c[..., 0:1, :] + c[..., 2:3, :]) + (c[..., 1:2, :] + c[..., 3:4, :])


def _sinkhorn(logits, fleet):
    rmax = logits.amax(dim=-1, keepdim=True)
    if fleet:
        c = torch.exp2((logits - rmax) * LOG2E)
    else:
        c = torch.exp(logits - rmax)
    c = c / _row_sum(c) + EPS
    c = c / (_col_sum(c) + EPS)
    for _ in range(SINKHORN_REPEAT - 1):
        c = c / (_row_sum(c) + EPS)
        c = c / (_col_sum(c) + EPS)
    return c


def _reference(d, fleet):
    """Torch reference in fp32: ``|z|^2 = tr(G)``, ``|y|^2 = pre^T G pre``.

    ``G`` is the Gram of the bf16 ``residual_out`` per token. Returns
    ``(residual_out, post_out, comb_out, layer_out)``.
    """
    t = d["x"].shape[0]
    hin = d["x"].float()
    if d["shared"] is not None:
        hin = (d["x"].double() + d["shared"].double()).float()
    res = d["residual"].view(t, HC, HIDDEN).float()
    cmb = d["comb"].reshape(t, HC, HC).float()
    acc = _fma(
        d["post"].reshape(t, HC)[:, :, None],
        hin[:, None, :],
        cmb[:, 0, :, None] * res[:, 0, None, :],
    )
    for i in range(1, HC):
        acc = _fma(cmb[:, i, :, None], res[:, i, None, :], acc)
    residual_out = acc.bfloat16()
    cur = residual_out.float()
    w = _round_tf32(d["fn"]).view(HC3, HC, -1)
    wz = (cur[:, None] * w[None]).sum(dim=(2, 3))
    gram = (cur[:, :, None, :] * cur[:, None, :, :]).sum(dim=-1)
    ms = ((gram[:, 0, 0] + gram[:, 1, 1]) + (gram[:, 2, 2] + gram[:, 3, 3])) / (
        HC * HIDDEN
    )
    rms = 1.0 / (torch.sqrt(ms) + EPS) if fleet else torch.rsqrt(ms + EPS)
    m = wz * rms[:, None]
    s, b = d["hc_scale"], d["hc_base"]
    pre = _sigmoid(_fma(m[:, :HC], s[0], b[:HC])) + EPS
    post = _sigmoid(_fma(m[:, HC : 2 * HC], s[1], b[HC : 2 * HC])) * POST_MULT
    logits = _fma(m[:, 2 * HC :], s[2], b[2 * HC :]).view(-1, HC, HC)
    comb = _sinkhorn(logits, fleet)
    y = pre[:, 0, None] * cur[:, 0]
    for j in range(1, HC):
        y = _fma(pre[:, j, None], cur[:, j], y)
    y = y.bfloat16()
    y_sqsum = (pre[:, :, None] * gram * pre[:, None, :]).sum(dim=(1, 2))
    rn = torch.rsqrt(y_sqsum / HIDDEN + EPS)
    layer = ((y.float() * rn[:, None]) * d["norm_weight"].float()).bfloat16()
    return residual_out, post, comb, layer


def _error(out, ref):
    a, b = out.double().flatten(), ref.double().flatten()
    return ((a - b).norm() / b.norm()).item(), (a - b).abs().max().item()


def _cos(out, ref):
    a, b = out.double().flatten(), ref.double().flatten()
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def _inputs(num_tokens, has_shared, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    randn = lambda *s: torch.randn(*s, generator=g, device="cuda")  # noqa: E731
    rand = lambda *s: torch.rand(*s, generator=g, device="cuda")  # noqa: E731
    pre_base = randn(HC) * 0.2 + torch.tensor([3.0, -3.0, -3.0, -3.0], device="cuda")
    comb_base = torch.eye(HC, device="cuda") * 6.0 - 3.0
    hc_base = torch.cat(
        [pre_base, rand(HC) * -0.3, (comb_base + randn(HC, HC) * 0.3).flatten()]
    )
    lo = torch.tensor([0.05, 0.03, 0.1], device="cuda")
    hi = torch.tensor([0.2, 0.08, 0.4], device="cuda")
    comb_logits = comb_base + randn(num_tokens, HC, HC)
    return dict(
        x=randn(num_tokens, HIDDEN).bfloat16(),
        residual=randn(num_tokens, HC, HIDDEN).bfloat16(),
        post=2.0 * _sigmoid(randn(num_tokens, HC) * 0.3 - 0.2),
        comb=_sinkhorn(comb_logits, False),
        fn=randn(HC3, HC * HIDDEN) * 0.05,
        hc_scale=lo + (hi - lo) * rand(3),
        hc_base=hc_base,
        norm_weight=(randn(HIDDEN) * 0.02 + 0.4).bfloat16(),
        shared=(torch.randn(num_tokens, HIDDEN).bfloat16() if has_shared else None),
    )


def _run(d, fleet, use_pdl=True, swizzle=False):
    return megallmhc(
        d["x"], d["residual"], d["post"], d["comb"], d["fn"], d["hc_scale"],
        d["hc_base"], d["norm_weight"], shared=d["shared"], use_pdl=use_pdl,
        use_swizzle=swizzle, **_kwargs(fleet),
    )  # fmt: skip


@pytest.mark.parametrize("num_tokens", TOKENS)
@pytest.mark.parametrize("has_shared", [False, True])
@pytest.mark.parametrize("fleet", [True, False])
@pytest.mark.parametrize("swizzle", [False, True])
def test_megallmhc(num_tokens, has_shared, fleet, swizzle):
    d = _inputs(num_tokens, has_shared, seed=num_tokens)
    residual_out, layer_out, post_out, comb_out, q, sf = _run(d, fleet, swizzle=swizzle)
    ref_res, ref_post, ref_comb, ref_layer = _reference(d, fleet)
    cos = _cos(residual_out, ref_res)
    assert cos >= MIN_COS, ("residual", cos)
    for name, a, b in (
        ("post", post_out, ref_post),
        ("comb", comb_out, ref_comb),
        ("layer", layer_out, ref_layer),
    ):
        rel, max_abs = _error(a, b)
        rel_tol, abs_tol = TOL[name]
        assert rel <= rel_tol and max_abs <= abs_tol, (name, rel, max_abs)
    ref_q, ref_sf = group_wise_quant(layer_out, mode="w8a8-mxfp8", use_swizzle=swizzle)
    for name, a, b in (("q", q.float(), ref_q.float()), ("sf", sf, ref_sf)):
        cos = _cos(a, b)
        assert cos >= MIN_COS, (name, cos)


@pytest.mark.parametrize("num_tokens", [16, 100, 512])
def test_bitwise_deterministic(num_tokens):
    d = _inputs(num_tokens, True, seed=3)
    base = _run(d, True, use_pdl=False)
    for use_pdl in (False, True):
        for a, b in zip(_run(d, True, use_pdl=use_pdl), base):
            assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def test_fleet_off_by_default():
    d = _inputs(64, True, seed=5)
    kw = _kwargs(False)
    del kw["mhc_align_fleet"]
    out = megallmhc(
        d["x"], d["residual"], d["post"], d["comb"], d["fn"], d["hc_scale"],
        d["hc_base"], d["norm_weight"], shared=d["shared"], **kw,
    )  # fmt: skip
    for a, b in zip(out, _run(d, False)):
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))


@pytest.mark.parametrize("swizzle", [False, True])
def test_zero_tokens(swizzle):
    shapes = [(0, 4, 4096), (0, 4096), (0, 4), (0, 4, 4), (0, 4096), (0, 128)]
    out = _run(_inputs(0, False, seed=0), True, swizzle=swizzle)
    assert [tuple(t.shape) for t in out] == shapes


def _bad_x(d):
    d["x"] = d["x"][:, :2048].contiguous()


def _bad_dtype(d):
    d["residual"] = d["residual"].float()


def _non_contiguous(d):
    d["residual"] = d["residual"].transpose(0, 1).contiguous().transpose(0, 1)


def _bad_numel(d):
    d["post"] = d["post"][:-1]


def _bad_shared(d):
    d["shared"] = d["x"][:-1]


def _bad_device(d):
    d["norm_weight"] = d["norm_weight"].cpu()


@pytest.mark.parametrize(
    "mutate, error",
    [
        pytest.param(_bad_x, "unsupported hidden size 2048", id="x_hidden"),
        pytest.param(_bad_dtype, "residual must be a contiguous BFloat16", id="dtype"),
        pytest.param(_non_contiguous, "residual must be a contiguous", id="layout"),
        pytest.param(_bad_numel, "post must be .* of 32 elements", id="numel"),
        pytest.param(_bad_shared, "shared must be .* of 32768 elements", id="shared"),
        pytest.param(_bad_device, "norm_weight must be .* on cuda", id="device"),
    ],
)
def test_rejects(mutate, error):
    d = _inputs(8, False, seed=0)
    mutate(d)
    with pytest.raises(RuntimeError, match=error):
        _run(d, True)
