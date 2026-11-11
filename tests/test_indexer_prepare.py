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

from dataclasses import dataclass

import pytest
import torch

import halops
from halops import indexer_prepare
from tests.utils.gemm_utils import (
    per_token_cast_to_fp8,
    per_token_cast_to_mxfp4,
    ue8m0_to_float32,
)

FP8 = torch.float8_e4m3fn
HIDDEN, HEADS, HEAD_DIM, RANK, ROPE_DIM = 4096, 64, 128, 1024, 64
DEFAULT_WEIGHT_SCALE = HEAD_DIM**-0.5 * HEADS**-0.5
FP8_RMS_TOL, SCALE_RMS_TOL = 5e-3, 4e-3
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] != 10
    or torch.cuda.get_device_properties(0).multi_processor_count < 128,
    reason="indexer_prepare requires SM100 with at least 128 SMs",
)


def sfatom_scales(exponents):
    rows, groups = exponents.shape
    return (
        exponents.reshape(rows // 128, 4, 32, groups // 4, 4)
        .permute(0, 3, 2, 1, 4)
        .contiguous()
        .reshape(rows, groups)
    )


def hadamard(x):
    result = x.float()
    width = 1
    while width < result.shape[-1]:
        groups = result.reshape(*result.shape[:-1], -1, 2, width)
        left, right = groups.unbind(-2)
        result = torch.stack((left + right, left - right), -2).reshape(x.shape)
        width *= 2
    return result * x.shape[-1] ** -0.5


def rope(x, table, positions, is_neox_style=True):
    cs = table[positions.long()]
    shape = (x.shape[0],) + (1,) * (x.ndim - 2) + (ROPE_DIM // 2,)
    cos, sin = cs[:, : ROPE_DIM // 2].reshape(shape), cs[:, ROPE_DIM // 2 :].reshape(
        shape
    )
    result = x.clone()
    first, second = (
        (slice(0, ROPE_DIM // 2), slice(ROPE_DIM // 2, ROPE_DIM))
        if is_neox_style
        else (slice(0, ROPE_DIM, 2), slice(1, ROPE_DIM, 2))
    )
    left, right = x[..., first], x[..., second]
    result[..., first] = left * cos - right * sin
    result[..., second] = left * sin + right * cos
    return result


def quantize_rows(x):
    rounded = x.float()
    scale = rounded.abs().amax(-1).clamp_min(1e-4) / 448
    return (rounded / scale[..., None]).clamp(-448, 448).to(FP8), scale


def quantize_mx(x, *, fp4=False, group_size=32):
    """Quantize indexer values with the shared MX reference implementation."""
    quantize = per_token_cast_to_mxfp4 if fp4 else per_token_cast_to_fp8
    codes, scales = quantize(
        x.float().reshape(-1, x.shape[-1]),
        block_size=group_size,
        amax_floor=1e-10,
    )[:2]
    shape = (*x.shape[:-1], -1)
    return codes.to(x.device).reshape(shape), scales.to(x.device).reshape(shape)


def dequantize_mxfp8(codes, scales):
    return codes.float() * ue8m0_to_float32(scales).to(codes.device).repeat_interleave(
        32, -1
    )


def quantize_blocks(x, group_size):
    codes, exponents = quantize_mx(x, group_size=group_size)
    scale = ue8m0_to_float32(exponents).to(x.device)
    dequantized = (codes.float() * scale.repeat_interleave(group_size, -1)).reshape(
        x.shape
    )
    return codes, exponents, dequantized


def dequantize_mxfp4(packed, scales):
    if packed.dtype == torch.int8:
        packed = packed.view(torch.uint8)
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)
    values = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=packed.device)
    decoded = values[(codes & 7).long()] * torch.where(codes & 8 != 0, -1.0, 1.0)
    return decoded * torch.exp2(scales.float() - 127).repeat_interleave(32, dim=-1)


@dataclass
class Case:
    kwargs: dict
    wq_dequantized: torch.Tensor
    page_size: int

    @property
    def cache(self):
        return self.kwargs["paged_kv_cache_with_scale"]

    @property
    def is_fp4(self):
        return self.kwargs["out_q"].dtype == torch.int8

    @property
    def output_dtype(self):
        if self.is_fp4:
            return "mxfp4"
        return "mxfp8" if self.kwargs["out_q_sf"] is not None else "fp8"

    @property
    def data_bytes(self):
        return HEAD_DIM // 2 if self.is_fp4 else HEAD_DIM

    def read_cache(self):
        loc = self.kwargs["kv_cache_loc"].long()
        valid = loc >= 0
        loc = loc[valid]
        page, slot = loc // self.page_size, loc % self.page_size
        offsets = slot[:, None] * self.data_bytes + torch.arange(
            self.data_bytes, device=self.cache.device
        )
        codes = self.cache[page[:, None], offsets].contiguous()
        if not self.is_fp4:
            codes = codes.view(FP8)
        offsets = (
            self.page_size * self.data_bytes
            + slot[:, None] * 4
            + torch.arange(4, device=self.cache.device)
        )
        scales = self.cache[page[:, None], offsets].contiguous()
        if self.output_dtype == "fp8":
            scales = scales.view(torch.float32).flatten()
        return codes, scales, valid

    def written_bytes(self):
        loc = self.kwargs["kv_cache_loc"].long()
        loc = loc[loc >= 0]
        page, slot = loc // self.page_size, loc % self.page_size
        mask = torch.zeros_like(self.cache, dtype=torch.bool)
        columns = slot[:, None] * self.data_bytes + torch.arange(
            self.data_bytes, device=self.cache.device
        )
        mask[page[:, None], columns] = True
        columns = (
            self.page_size * self.data_bytes
            + slot[:, None] * 4
            + torch.arange(4, device=self.cache.device)
        )
        mask[page[:, None], columns] = True
        return mask


def make_case(
    tokens,
    *,
    page_size=64,
    bias=True,
    apply_hadamard=True,
    sentinels=False,
    shuffle=False,
    index_dtype=torch.int64,
    varied_groups=False,
    device="cuda",
    is_neox_style=True,
    output_dtype="fp8",
):
    def randn(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, dtype=dtype, device=device)

    x = randn(tokens, HIDDEN)
    q_lora = randn(tokens, RANK)
    wq = randn(HEADS * HEAD_DIM, RANK)
    if varied_groups:
        scales = torch.logspace(-2, 2, RANK // 32, device=device).repeat_interleave(32)
        q_lora = (q_lora.float() * scales).to(torch.bfloat16)
        wq = (wq.float() * scales.flip(0)).to(torch.bfloat16)
    wq, exponents, wq_dequantized = quantize_blocks(wq, 32)
    scale_input = sfatom_scales(exponents)
    pages = max(2, (tokens + page_size - 1) // page_size + 1)
    data_bytes = HEAD_DIM // 2 if output_dtype == "mxfp4" else HEAD_DIM
    cache = torch.full(
        (pages, page_size * (data_bytes + 4)), 0xA5, dtype=torch.uint8, device=device
    )
    loc = (
        torch.randperm(pages * page_size, device=device)[:tokens]
        if shuffle
        else torch.arange(tokens, device=device)
    ).to(index_dtype)
    if sentinels:
        loc[::3] = -7
    angles = randn(tokens + 17, ROPE_DIM // 2, dtype=torch.float32)
    kwargs = dict(
        x=x,
        q_lora=q_lora,
        wk_w=randn(HEAD_DIM + HEADS, HIDDEN),
        wq_b=wq,
        wq_b_sf=scale_input,
        k_norm_weight=randn(HEAD_DIM, dtype=torch.float32),
        k_norm_bias=randn(HEAD_DIM, dtype=torch.float32) if bias else None,
        k_norm_eps=1e-6,
        rope_cos_sin_cache=torch.cat((angles.cos(), angles.sin()), -1),
        positions=torch.randperm(tokens + 17, device=device)[:tokens].to(index_dtype),
        paged_kv_cache_with_scale=cache,
        kv_cache_loc=loc,
        out_q=torch.empty(
            (tokens, HEADS, HEAD_DIM // 2 if output_dtype == "mxfp4" else HEAD_DIM),
            dtype=torch.int8 if output_dtype == "mxfp4" else FP8,
            device=device,
        ),
        out_q_sf=(
            None
            if output_dtype == "fp8"
            else torch.empty(
                (tokens, HEADS, HEAD_DIM // 32), dtype=torch.uint8, device=device
            )
        ),
        out_weights=torch.empty((tokens, HEADS), dtype=torch.float32, device=device),
        weight_scale=DEFAULT_WEIGHT_SCALE,
        apply_hadamard=apply_hadamard,
        is_neox_style=is_neox_style,
    )
    return Case(kwargs, wq_dequantized, page_size)


def reference(case):
    inp = case.kwargs
    kw = inp["x"].float() @ inp["wk_w"].float().t()
    key, raw_weights = kw[:, :HEAD_DIM], kw[:, HEAD_DIM:]
    if inp["k_norm_bias"] is not None:
        key = key - key.mean(-1, keepdim=True)
    key = key * (key.square().mean(-1, keepdim=True) + inp["k_norm_eps"]).rsqrt()
    key = key * inp["k_norm_weight"]
    if inp["k_norm_bias"] is not None:
        key = key + inp["k_norm_bias"]
    _, _, q_lora = quantize_blocks(inp["q_lora"], 32)
    q = (q_lora @ case.wq_dequantized.t()).reshape(-1, HEADS, HEAD_DIM)
    key = rope(
        key, inp["rope_cos_sin_cache"], inp["positions"], inp.get("is_neox_style", True)
    )
    q = rope(
        q, inp["rope_cos_sin_cache"], inp["positions"], inp.get("is_neox_style", True)
    )
    if inp["apply_hadamard"]:
        key, q = hadamard(key), hadamard(q)
    weight_scale = inp.get("weight_scale")
    if weight_scale is None:
        weight_scale = DEFAULT_WEIGHT_SCALE
    output_dtype = (
        "mxfp4"
        if inp["out_q"].dtype == torch.int8
        else ("mxfp8" if inp["out_q_sf"] is not None else "fp8")
    )
    if output_dtype == "mxfp4":
        key_codes, key_scales = quantize_mx(key, fp4=True)
        q_codes, q_scales = quantize_mx(q, fp4=True)
        weights = raw_weights * weight_scale
    elif output_dtype == "mxfp8":
        key_codes, key_scales = quantize_mx(key)
        q_codes, q_scales = quantize_mx(q)
        weights = raw_weights * weight_scale
    else:
        key_codes, key_scales = quantize_rows(key)
        q_codes, q_scales = quantize_rows(q)
        weights = raw_weights * weight_scale * q_scales
    return dict(
        q=q_codes,
        q_scales=q_scales,
        weights=weights,
        key=key_codes,
        key_scales=key_scales,
        q_unquantized=q.to(torch.bfloat16),
        key_unquantized=key.to(torch.bfloat16),
    )


def relative_rms(actual, expected):
    diff = (actual.float() - expected.float()).square().mean().sqrt()
    norm = expected.float().square().mean().sqrt().clamp_min(1e-12)
    return (diff / norm).item()


def check_rms(actual, expected, tolerance):
    torch.testing.assert_close(
        relative_rms(actual, expected), 0.0, rtol=0, atol=tolerance
    )


def output_tensors(result):
    """Flatten the public result for byte comparisons, omitting FP8's None."""
    (query, scales), weights = result
    return (query, weights) if scales is None else (query, scales, weights)


def check_scale_exponents(actual, expected, unquantized, maximum=6):
    mismatch = actual != expected
    if not mismatch.any():
        return
    # Only BF16 neighbors straddling a scale boundary may select different scales.
    difference = (actual.to(torch.int16) - expected.to(torch.int16)).abs()
    assert (difference[mismatch] == 1).all()
    maxima = unquantized.float().reshape(*actual.shape, 32).abs().amax(-1)
    smaller_exponent = torch.minimum(actual, expected).float() - 127
    boundary = maximum * torch.exp2(smaller_exponent)
    bf16_step = torch.exp2(torch.floor(torch.log2(maxima.clamp_min(1e-10))) - 7)
    assert ((maxima - boundary).abs()[mismatch] <= bf16_step[mismatch]).all()


def native_call(case, query, scales, weights, **options):
    inp = case.kwargs
    halops.halops_kernel.indexer_prepare(
        inp["x"],
        inp["q_lora"],
        inp["wk_w"],
        inp["wq_b"],
        inp["wq_b_sf"],
        inp["k_norm_weight"],
        inp["k_norm_bias"],
        inp["k_norm_eps"],
        inp["rope_cos_sin_cache"],
        inp["positions"],
        case.cache,
        inp["kv_cache_loc"],
        out_q=query,
        out_q_sf=scales,
        out_weights=weights,
        weight_scale=inp["weight_scale"],
        apply_hadamard=inp["apply_hadamard"],
        **options,
    )


def check(case, result=None):
    expected = reference(case)
    if result is None:
        assert indexer_prepare(**case.kwargs) is None
        query, scales, weights = (
            case.kwargs["out_q"],
            case.kwargs["out_q_sf"],
            case.kwargs["out_weights"],
        )
    else:
        (query, scales), weights = result
    key, key_scales, valid = case.read_cache()
    key_ref, key_scales_ref = expected["key"][valid], expected["key_scales"][valid]
    output_dtype = case.output_dtype
    fp4 = output_dtype == "mxfp4"
    tolerance = 2e-2 if fp4 else FP8_RMS_TOL
    if output_dtype != "fp8":
        decode = dequantize_mxfp4 if fp4 else dequantize_mxfp8
        maximum = 6 if fp4 else 448
        assert query.dtype == (torch.int8 if fp4 else FP8)
        assert scales.dtype == torch.uint8
        assert query.shape == (
            case.kwargs["x"].shape[0],
            HEADS,
            HEAD_DIM // 2 if fp4 else HEAD_DIM,
        )
        assert scales.shape == (*query.shape[:-1], HEAD_DIM // 32)
        assert (key_scales != 255).all()
        check_scale_exponents(
            key_scales, key_scales_ref, expected["key_unquantized"][valid], maximum
        )
        key = decode(key, key_scales)
        key_ref = decode(key_ref, key_scales_ref)
        assert (scales != 255).all()
        check_scale_exponents(
            scales, expected["q_scales"], expected["q_unquantized"], maximum
        )
        query = decode(query, scales)
        query_ref = decode(expected["q"], expected["q_scales"])
    else:
        assert scales is None and query.dtype == FP8
        assert query.shape == (case.kwargs["x"].shape[0], HEADS, HEAD_DIM)
        if key.numel():
            check_rms(key_scales, key_scales_ref, SCALE_RMS_TOL)
        key, key_ref = key.float(), key_ref.float()
        query, query_ref = query.float(), expected["q"].float()
    if key.numel():
        check_rms(key, key_ref, tolerance)
    assert torch.isfinite(query).all() and torch.isfinite(weights).all()
    check_rms(query, query_ref, tolerance)
    check_rms(weights, expected["weights"], SCALE_RMS_TOL)
    if key.numel():
        if output_dtype == "fp8":
            key, key_ref = (
                key * key_scales[:, None],
                key_ref * key_scales_ref[:, None],
            )
        scores = torch.einsum("thd,kd->thk", query, key[-8:]).relu()
        scores_ref = torch.einsum("thd,kd->thk", query_ref, key_ref[-8:]).relu()
        check_rms(
            (scores * weights[..., None]).sum(1),
            (scores_ref * expected["weights"][..., None]).sum(1),
            tolerance,
        )
    assert (case.cache[~case.written_bytes()] == 0xA5).all()


def assert_same_outputs(actual, expected):
    for value, target in zip(
        output_tensors(actual), output_tensors(expected), strict=True
    ):
        assert torch.equal(value.view(torch.uint8), target.view(torch.uint8))


@pytest.mark.parametrize("output_dtype", ["fp8", "mxfp4", "mxfp8"])
@pytest.mark.parametrize("tokens", [1, 17, 128, 129, 512])
def test_formats_and_dispatch(tokens, output_dtype):
    """Cover every output format and the dispatch boundaries used in production."""
    case = make_case(
        tokens,
        output_dtype=output_dtype,
        shuffle=tokens > 1,
        sentinels=tokens > 1,
        is_neox_style=tokens % 2 == 0,
    )
    check(case)


@pytest.mark.parametrize(
    "options",
    [
        {"bias": False},
        {"apply_hadamard": False},
        {"page_size": 1, "shuffle": True, "sentinels": True},
        {"page_size": 32, "index_dtype": torch.int32},
        {"varied_groups": True},
    ],
)
def test_supported_options(options):
    check(make_case(17, **options))


@pytest.mark.parametrize("output_dtype", ["fp8", "mxfp4", "mxfp8"])
def test_cuda_graph_replay(output_dtype):
    case = make_case(129, output_dtype=output_dtype, shuffle=True, sentinels=True)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        assert indexer_prepare(**case.kwargs) is None
    result = (
        (case.kwargs["out_q"], case.kwargs["out_q_sf"]),
        case.kwargs["out_weights"],
    )
    for _ in range(2):
        case.kwargs["x"].normal_()
        case.kwargs["q_lora"].normal_()
        case.cache.fill_(0xA5)
        graph.replay()
        check(case, result)


def test_native_binding_matches_python_api():
    case = make_case(8, output_dtype="mxfp4", is_neox_style=False)
    expected_kwargs = {
        **case.kwargs,
        "paged_kv_cache_with_scale": case.cache.clone(),
        "out_q": torch.empty_like(case.kwargs["out_q"]),
        "out_q_sf": torch.empty_like(case.kwargs["out_q_sf"]),
        "out_weights": torch.empty_like(case.kwargs["out_weights"]),
    }
    assert indexer_prepare(**expected_kwargs) is None
    expected = (
        (expected_kwargs["out_q"], expected_kwargs["out_q_sf"]),
        expected_kwargs["out_weights"],
    )
    query = torch.empty_like(case.kwargs["out_q"])
    scales = torch.empty_like(case.kwargs["out_q_sf"])
    weights = torch.empty_like(case.kwargs["out_weights"])
    native_call(
        case,
        query,
        scales,
        weights,
        is_neox_style=False,
    )
    assert_same_outputs(((query, scales), weights), expected)


def test_invalid_output_buffers_are_rejected():
    case = make_case(4, output_dtype="mxfp8")
    with pytest.raises(RuntimeError, match="out_q"):
        indexer_prepare(
            **{
                **case.kwargs,
                "out_q": torch.empty((3, HEADS, HEAD_DIM), dtype=FP8, device="cuda"),
            }
        )
    with pytest.raises(RuntimeError, match="out_q_sf"):
        indexer_prepare(
            **{
                **case.kwargs,
                "out_q_sf": torch.empty(
                    (4, HEADS, 2), dtype=torch.uint8, device="cuda"
                ),
            }
        )


@pytest.mark.parametrize("fp4,zero_exponent", [(False, 85), (True, 92)])
def test_reference_rope_and_quantizer_contracts(fp4, zero_exponent):
    values = torch.arange(128, dtype=torch.float32, device="cuda").reshape(1, 128)
    table = torch.cat(
        (torch.zeros(1, 32, device="cuda"), torch.ones(1, 32, device="cuda")), -1
    )
    rotated = rope(values, table, torch.zeros(1, dtype=torch.long, device="cuda"))
    torch.testing.assert_close(rotated[:, :32], -values[:, 32:64])
    torch.testing.assert_close(rotated[:, 32:64], values[:, :32])

    x = torch.zeros((1, 64), dtype=torch.bfloat16, device="cuda")
    _, exponents, decoded = quantize_blocks(x, 32)
    assert (exponents == 85).all()
    assert torch.equal(decoded, x.float())

    # The shared helpers are 2-D/BF16; Q is 3-D/FP32 before quantization.
    values = torch.zeros((2, 3, 128), dtype=torch.float32, device="cuda")
    codes, scales = quantize_mx(values, fp4=fp4)
    assert codes.shape == (2, 3, 64 if fp4 else 128)
    assert scales.shape == (2, 3, 4)
    assert codes.device == scales.device == values.device
    assert (scales == zero_exponent).all()
    decode = dequantize_mxfp4 if fp4 else dequantize_mxfp8
    assert torch.equal(decode(codes, scales), values)
