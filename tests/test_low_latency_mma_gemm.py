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

import pytest
import torch

from halops import group_wise_quant, low_latency_mma_gemm
from tests.utils.gemm_utils import (
    MXFP8_BLOCK_SIZE,
    W8A8_MXFP8_TOL,
    ue8m0_to_float32,
)

# TMA and bulk copies need 16-byte aligned scale rows.
TMA_ALIGN_BYTES = 16


def spread_groups(values):
    """Scale every (row, 32-wide K group) by its own power of two in
    2^-2..2^2, so a scale read from the wrong row or k-group shows up as a
    factor-of-two error."""
    rows, cols = values.shape
    exponents = torch.randint(-2, 3, (rows, cols // MXFP8_BLOCK_SIZE, 1)).float()
    grouped = values.float().view(rows, cols // MXFP8_BLOCK_SIZE, MXFP8_BLOCK_SIZE)
    return (grouped * torch.exp2(exponents)).view(rows, cols).to(torch.bfloat16)


def make_bf16(rows, cols, magnitude):
    return spread_groups(torch.randn(rows, cols, dtype=torch.bfloat16) * magnitude)


def pad_scale_rows(scale, extra_bytes=0):
    """[rows, K / 32] view of ``scale`` in a buffer whose row pitch is the
    16-byte-rounded row plus ``extra_bytes``."""
    rows, cols = scale.shape
    pitch = -(-cols // TMA_ALIGN_BYTES) * TMA_ALIGN_BYTES
    buffer = torch.zeros(rows, pitch + extra_bytes, dtype=torch.uint8)
    buffer[:, :cols].copy_(scale)
    return buffer[:, :cols]


def quant(values):
    """W8A8-MXFP8 quantization with scale rows padded to 16 bytes."""
    values_q, scale = group_wise_quant(values, mode="w8a8-mxfp8", use_swizzle=False)
    if scale.stride(0) % TMA_ALIGN_BYTES:
        scale = pad_scale_rows(scale)
    return values_q, scale


def dequant(values_q, scale):
    block_size = values_q.shape[1] // scale.shape[1]
    expanded = ue8m0_to_float32(scale).repeat_interleave(block_size, dim=1)
    return values_q.float() * expanded


def reference(input_q, sfa, weight_q, sfb):
    return dequant(input_q, sfa) @ dequant(weight_q, sfb).T


def make_operands(m, n, k):
    input_q, sfa = quant(make_bf16(m, k, 0.1))
    weight_q, sfb = quant(make_bf16(n, k, k**-0.5))
    return input_q, sfa, weight_q, sfb


def run(input_q, weight_q, sfa, sfb, **kwargs):
    output = torch.empty(input_q.shape[0], weight_q.shape[0], dtype=torch.bfloat16)
    low_latency_mma_gemm(input_q, weight_q, sfa, sfb, output, **kwargs)
    return output


def check(output, expected):
    torch.testing.assert_close(output.float(), expected, **W8A8_MXFP8_TOL)


@pytest.mark.parametrize(
    "m, n, k",
    [
        pytest.param(1, 4096, 4096, id="m1-4096x4096"),
        pytest.param(5, 520, 7168, id="m5-n_8_aligned-k_blocks_not_pow2"),
        pytest.param(8, 2048, 1280, id="m8-padded_scale_rows"),
        pytest.param(16, 7168, 2048, id="m16-7168x2048"),
        pytest.param(1, 512, 23808, id="m1-max_resident_k"),
    ],
)
def test_gemm(m, n, k):
    input_q, sfa, weight_q, sfb = make_operands(m, n, k)
    check(run(input_q, weight_q, sfa, sfb), reference(input_q, sfa, weight_q, sfb))


# Mirrors kLlTiles. tile_shape(block_n, warps_k) is a shape SelectLlTile
# serves with exactly that tile on TILE_SM_COUNT SMs; recheck it when the
# tile table, cost model or shared memory plan changes.
TILES = [
    (block_n, warps_k)
    for block_n in (8, 16, 32)
    for warps_k in (1, 2, 4, 8, 16)
    if block_n * warps_k <= 128
]
TILE_SM_COUNT = 8


def tile_shape(block_n, warps_k):
    return 65 * block_n, 128 * warps_k * (2 if warps_k >= 8 else 6)


@pytest.mark.parametrize("m", [8, 16])
@pytest.mark.parametrize("block_n, warps_k", TILES)
def test_tiles(block_n, warps_k, m):
    n, k = tile_shape(block_n, warps_k)
    input_q, sfa, weight_q, sfb = make_operands(m, n, k)
    output = run(input_q, weight_q, sfa, sfb, sm_count=TILE_SM_COUNT)
    check(output, reference(input_q, sfa, weight_q, sfb))


def test_strided_scales_match_contiguous():
    input_q, sfa, weight_q, sfb = make_operands(8, 512, 4096)
    strided = run(
        input_q,
        weight_q,
        pad_scale_rows(sfa, TMA_ALIGN_BYTES),
        pad_scale_rows(sfb, TMA_ALIGN_BYTES),
    )
    assert torch.equal(strided, run(input_q, weight_q, sfa, sfb))


def test_sm_count_matches_full_device():
    # warps_k stays 4 at every grid size, so results are bitwise equal.
    input_q, sfa, weight_q, sfb = make_operands(8, 4096, 1024)
    full = run(input_q, weight_q, sfa, sfb)
    assert torch.equal(full, run(input_q, weight_q, sfa, sfb, sm_count=1))


@pytest.mark.parametrize("use_pdl", [True, False])
@pytest.mark.parametrize(
    "m, n, n2, k",
    [
        pytest.param(8, 4096, 1024, 2048, id="m8-wide_tiles"),
        pytest.param(16, 1544, 520, 1024, id="m16-n_8_aligned"),
    ],
)
def test_dual_output(m, n, n2, k, use_pdl):
    input_q, sfa, weight_q, sfb = make_operands(m, n, k)
    weight2_q, sfb2 = quant(make_bf16(n2, k, k**-0.5))
    output2 = torch.empty(m, n2, dtype=torch.bfloat16)
    output = run(
        input_q,
        weight_q,
        sfa,
        sfb,
        weight2=weight2_q,
        sfb2=sfb2,
        output2=output2,
        use_pdl=use_pdl,
    )
    check(output, reference(input_q, sfa, weight_q, sfb))
    check(output2, reference(input_q, sfa, weight2_q, sfb2))


# =====================================================================
# Concat-K: output = input @ weight^T + input2 @ weight2^T
# =====================================================================


@pytest.mark.parametrize("use_pdl", [True, False])
@pytest.mark.parametrize(
    "m, n, k1, k2, scale_pad",
    [
        pytest.param(8, 4096, 2048, 2048, 0, id="m8-aligned"),
        pytest.param(16, 2048, 1024, 768, 16, id="m16-k2_not_512-strided_scales"),
    ],
)
def test_concat_k(m, n, k1, k2, scale_pad, use_pdl):
    input_q, sfa, weight_q, sfb = make_operands(m, n, k1)
    input2_q, sfa2 = quant(make_bf16(m, k2, 0.1))
    weight2_q, sfb2 = quant(make_bf16(n, k2, k2**-0.5))
    if scale_pad:
        sfa, sfb, sfa2, sfb2 = (
            pad_scale_rows(scale, scale_pad) for scale in (sfa, sfb, sfa2, sfb2)
        )
    output = run(
        input_q,
        weight_q,
        sfa,
        sfb,
        input2=input2_q,
        weight2=weight2_q,
        sfa2=sfa2,
        sfb2=sfb2,
        use_pdl=use_pdl,
    )
    expected = reference(input_q, sfa, weight_q, sfb) + reference(
        input2_q, sfa2, weight2_q, sfb2
    )
    check(output, expected)


@pytest.mark.parametrize("m", [6, 16])
def test_sigmoid(m):
    input_q, sfa, weight_q, sfb = make_operands(m, 2048, 1024)
    output = run(input_q, weight_q, sfa, sfb, use_sigmoid=True)
    assert torch.equal(output, torch.sigmoid(run(input_q, weight_q, sfa, sfb)))


def gemm_args(input_q, sfa, weight_q, sfb, **kwargs):
    output = torch.empty(input_q.shape[0], weight_q.shape[0], dtype=torch.bfloat16)
    return dict(
        input=input_q, weight=weight_q, sfa=sfa, sfb=sfb, output=output, **kwargs
    )


def m_above_16():
    return gemm_args(*make_operands(17, 512, 4096))


def n_not_multiple_of_8():
    return gemm_args(*make_operands(8, 516, 256))


def k_not_multiple_of_256():
    return gemm_args(*make_operands(8, 512, 1152))


def resident_a_over_smem():
    return gemm_args(*make_operands(16, 512, 16384))


def unaligned_scale_rows():
    input_q, sfa, weight_q, sfb = make_operands(8, 512, 1024)
    return gemm_args(input_q, pad_scale_rows(sfa, 4), weight_q, sfb)


def swizzled_sfa():
    input_q, _, weight_q, sfb = make_operands(8, 4096, 4096)
    _, sfa = group_wise_quant(
        make_bf16(8, 4096, 0.1), mode="w8a8-mxfp8", use_swizzle=True
    )
    return gemm_args(input_q, sfa, weight_q, sfb)


def misaligned_weight():
    input_q, sfa, weight_q, sfb = make_operands(8, 512, 2048)
    buffer = torch.empty(weight_q.numel() + 1, dtype=torch.float8_e4m3fn)
    offset_weight = buffer[1:].view(weight_q.shape)
    offset_weight.copy_(weight_q)
    return gemm_args(input_q, sfa, offset_weight, sfb)


def weight2_without_input2_or_output2():
    input_q, sfa, weight_q, sfb = make_operands(8, 512, 2048)
    return gemm_args(input_q, sfa, weight_q, sfb, weight2=weight_q, sfb2=sfb)


def sigmoid_with_weight2():
    input_q, sfa, weight_q, sfb = make_operands(8, 512, 2048)
    output2 = torch.empty(8, 512, dtype=torch.bfloat16)
    return gemm_args(
        input_q,
        sfa,
        weight_q,
        sfb,
        weight2=weight_q,
        sfb2=sfb,
        output2=output2,
        use_sigmoid=True,
    )


def concat_k_k1_not_multiple_of_512():
    input_q, sfa, weight_q, sfb = make_operands(8, 512, 1920)
    input2_q, sfa2, weight2_q, sfb2 = make_operands(8, 512, 2176)
    return gemm_args(
        input_q,
        sfa,
        weight_q,
        sfb,
        input2=input2_q,
        weight2=weight2_q,
        sfa2=sfa2,
        sfb2=sfb2,
    )


@pytest.mark.parametrize(
    "make_args, error",
    [
        pytest.param(m_above_16, "supports M <= 16", id="m_above_16"),
        pytest.param(
            n_not_multiple_of_8,
            "weight rows must be divisible by 8",
            id="n_not_multiple_of_8",
        ),
        pytest.param(
            k_not_multiple_of_256,
            "K must be divisible by 256",
            id="k_not_multiple_of_256",
        ),
        pytest.param(
            resident_a_over_smem,
            "too large to keep A resident",
            id="resident_a_over_smem",
        ),
        pytest.param(
            unaligned_scale_rows, "plain row-major", id="unaligned_scale_rows"
        ),
        pytest.param(swizzled_sfa, "plain row-major", id="swizzled_sfa"),
        pytest.param(
            misaligned_weight,
            "TMA descriptor encoding failed",
            id="misaligned_weight",
        ),
        pytest.param(
            weight2_without_input2_or_output2,
            "weight2 / sfb2 must come with",
            id="weight2_without_input2_or_output2",
        ),
        pytest.param(
            concat_k_k1_not_multiple_of_512,
            "K1 must be a multiple of 512",
            id="concat_k_k1_not_multiple_of_512",
        ),
        pytest.param(
            sigmoid_with_weight2, "does not support", id="sigmoid_with_weight2"
        ),
    ],
)
def test_rejects(make_args, error):
    with pytest.raises(RuntimeError, match=error):
        low_latency_mma_gemm(**make_args())
