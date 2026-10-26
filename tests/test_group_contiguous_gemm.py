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
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from halops import group_contiguous_gemm
from tests.utils.fp4_utils import MXFP4_BLOCK_SIZE, NVFP4_BLOCK_SIZE

from tests.utils.gemm_utils import (
    W4A4_TOL,
    W8A8_MXFP8_TOL,
    per_block_cast_to_nvfp4,
    per_block_cast_to_fp8,
    per_block_cast_to_mxfp4,
    per_token_cast_to_nvfp4,
    per_token_cast_to_fp8,
    per_token_cast_to_mxfp4,
    reorder_sf_to_cutlass,
    ue8m0_to_float32,
)


# =====================================================================
# Reference implementation
# =====================================================================


def ref_group_contiguous_gemm(
    x, w, sfa, sfb, expert_ids, num_experts, token_nums=None, bias=None, block_size=32
):
    """
    Reference implementation of FP8 grouped contiguous GEMM.
    Tokens of different experts are laid out contiguously in the input.
    If token_nums is provided, slice directly by offset (more efficient and
    avoids the row-by-row write-back issue).
    """
    N = w.shape[1]

    sfa_scale = ue8m0_to_float32(sfa)  # (M_total, k_blocks)
    sfb_scale = ue8m0_to_float32(sfb)  # (num_experts, N, k_blocks)

    ref_out = torch.zeros([x.shape[0], N], dtype=torch.bfloat16)

    if token_nums is not None:
        # Contiguous layout: slice directly by offset
        offset = 0
        for expert_id in range(num_experts):
            m = int(token_nums[expert_id])
            if m == 0:
                continue
            x_slice = x[offset : offset + m, :].to(torch.float32)
            sfa_slice = sfa_scale[offset : offset + m, :]
            sfa_expanded = torch.repeat_interleave(sfa_slice, block_size, dim=1)
            x_deq = x_slice * sfa_expanded

            w_slice = w[expert_id].to(torch.float32)
            sfb_slice = sfb_scale[expert_id]
            sfb_expanded = torch.repeat_interleave(sfb_slice, block_size, dim=1)
            w_deq = w_slice * sfb_expanded

            result = torch.matmul(x_deq, w_deq.T).to(torch.bfloat16)
            if bias is not None:
                result = result + bias[expert_id].unsqueeze(0)
            ref_out[offset : offset + m, :] = result
            offset += m
    else:
        for expert_id in range(num_experts):
            mask = expert_ids == expert_id
            indices = torch.nonzero(mask).flatten()
            if indices.shape[0] == 0:
                continue
            x_slice = torch.index_select(x, 0, indices).to(torch.float32)
            sfa_slice = torch.index_select(sfa_scale, 0, indices)
            sfa_expanded = torch.repeat_interleave(sfa_slice, block_size, dim=1)
            x_deq = x_slice * sfa_expanded

            w_slice = w[expert_id].to(torch.float32)
            sfb_slice = sfb_scale[expert_id]
            sfb_expanded = torch.repeat_interleave(sfb_slice, block_size, dim=1)
            w_deq = w_slice * sfb_expanded

            result = torch.matmul(x_deq, w_deq.T).to(torch.bfloat16)
            if bias is not None:
                result = result + bias[expert_id].unsqueeze(0)
            for i, idx in enumerate(indices.cpu().numpy()):
                ref_out[idx] = result[i]

    return ref_out


def ref_group_contiguous_gemm_w4a4(x_fp4_float, w_fp4_float, expert_ids, num_experts):
    """
    Reference implementation of W4A4 grouped contiguous GEMM.
    Uses dequantized float32 values for matmul directly, so the NVFP4 and
    MXFP4 tests share it.
    """
    N = w_fp4_float.shape[1]
    ref_out = torch.zeros([x_fp4_float.shape[0], N], dtype=torch.bfloat16)

    for expert_id in range(num_experts):
        mask = expert_ids == expert_id
        indices = torch.nonzero(mask).flatten()

        if indices.shape[0] == 0:
            continue

        x_slice = torch.index_select(x_fp4_float, 0, indices)
        w_slice = w_fp4_float[expert_id]

        result = torch.matmul(x_slice, w_slice.T).to(torch.bfloat16)

        for i, idx in enumerate(indices.cpu().numpy()):
            ref_out[idx] = result[i]

    return ref_out


# =====================================================================
# Kernel invocation and input construction
# =====================================================================


def group_contiguous_gemm_wrapper(
    x, w, sfa, sfb, expert_ids, bias=None, alpha=None, sm_count=-1, mode="w8a8-mxfp8"
):
    output = torch.zeros([x.shape[0], w.shape[1]], dtype=torch.bfloat16)
    group_contiguous_gemm(
        x, w, sfa, sfb, expert_ids, output, bias, alpha, sm_count, mode
    )
    return output


def make_contiguous_expert_ids(m, num_experts, tile_m):
    """
    Generate contiguous expert_ids: each expert occupies a contiguous span of
    rows, and each span length aligns to tile_m.
    Returns: expert_ids [m], int32; token_nums_per_expert [num_experts], int32;
    actual number of experts used.
    """
    # Each expert gets at least tile_m rows (skip if insufficient)
    max_experts = m // tile_m
    actual_experts = min(num_experts, max_experts)
    # Even split: each expert gets m // actual_experts rows, aligned to tile_m
    per_expert = (m // actual_experts // tile_m) * tile_m
    assert (
        per_expert >= tile_m
    ), f"m={m} too small for {actual_experts} experts with tile_m={tile_m}"

    ids = []
    token_nums = []
    for e in range(actual_experts):
        count = (
            per_expert
            if e < actual_experts - 1
            else (m - per_expert * (actual_experts - 1))
        )
        # Ensure the last expert's token count is also aligned
        count = (count // tile_m) * tile_m
        if count <= 0:
            actual_experts = e
            break
        ids.append(np.full(count, e, dtype=np.int32))
        token_nums.append(count)

    expert_ids = torch.tensor(np.concatenate(ids), dtype=torch.int32)
    token_nums_per_expert = torch.tensor(token_nums, dtype=torch.int32)
    return expert_ids, token_nums_per_expert, actual_experts


def make_w8a8_mxfp8_inputs(m, n, k, num_experts, tile_m, add_bias=False):
    """Build W8A8-MXFP8 inputs; m/num_experts are adjusted to the tile-aligned layout."""
    expert_ids, token_nums, num_experts = make_contiguous_expert_ids(
        m, num_experts, tile_m
    )
    m = expert_ids.shape[0]

    x_origin = torch.randn([m, k], dtype=torch.bfloat16)
    w_origin = torch.randn([num_experts, n, k], dtype=torch.bfloat16)

    x, sfa_logical = per_token_cast_to_fp8(x_origin, 32)
    w, sfb_logical = per_block_cast_to_fp8(w_origin, 32)

    sfa = reorder_sf_to_cutlass(sfa_logical, x.shape[0], x.shape[1] // 32)
    sfb = reorder_sf_to_cutlass(sfb_logical, w.shape[1], w.shape[2] // 32)

    bias = torch.rand([num_experts, n], dtype=torch.bfloat16) if add_bias else None

    return SimpleNamespace(
        m=m,
        num_experts=num_experts,
        expert_ids=expert_ids,
        token_nums=token_nums,
        x=x,
        w=w,
        sfa_logical=sfa_logical,
        sfb_logical=sfb_logical,
        sfa=sfa,
        sfb=sfb,
        bias=bias,
        sm_count=-1,
        mode="w8a8-mxfp8",
    )


def make_nvfp4_inputs(m, n, k, num_experts, tile_m):
    """Build W4A4 inputs; m/num_experts are adjusted to the tile-aligned layout."""
    expert_ids, token_nums, num_experts = make_contiguous_expert_ids(
        m, num_experts, tile_m
    )
    m = expert_ids.shape[0]

    x_origin = torch.randn([m, k], dtype=torch.bfloat16)
    w_origin = torch.randn([num_experts, n, k], dtype=torch.bfloat16)

    x_packed, sfa_logical, x_fp4_float = per_token_cast_to_nvfp4(
        x_origin, NVFP4_BLOCK_SIZE
    )
    w_packed, sfb_logical, w_fp4_float = per_block_cast_to_nvfp4(
        w_origin, NVFP4_BLOCK_SIZE
    )

    kb_dim = k // NVFP4_BLOCK_SIZE
    sfa = reorder_sf_to_cutlass(sfa_logical, m, kb_dim)
    sfb = reorder_sf_to_cutlass(sfb_logical, n, kb_dim)

    # W4A4 requires per-expert alpha scaling factor [num_experts] (fp32)
    alpha = torch.ones([num_experts], dtype=torch.float32)

    return SimpleNamespace(
        m=m,
        num_experts=num_experts,
        expert_ids=expert_ids,
        token_nums=token_nums,
        x_packed=x_packed,
        w_packed=w_packed,
        x_fp4_float=x_fp4_float,
        w_fp4_float=w_fp4_float,
        sfa=sfa,
        sfb=sfb,
        alpha=alpha,
        sm_count=-1,
        mode="w4a4-nvfp4",
    )


def make_mxfp4_inputs(m, n, k, num_experts, tile_m):
    """Build MXFP4 inputs; m/num_experts are adjusted to the tile-aligned layout."""
    expert_ids, token_nums, num_experts = make_contiguous_expert_ids(
        m, num_experts, tile_m
    )
    m = expert_ids.shape[0]

    x_origin = torch.randn([m, k], dtype=torch.bfloat16)
    w_origin = torch.randn([num_experts, n, k], dtype=torch.bfloat16)

    x_packed, sfa_logical, x_fp4_float = per_token_cast_to_mxfp4(
        x_origin, MXFP4_BLOCK_SIZE
    )
    w_packed, sfb_logical, w_fp4_float = per_block_cast_to_mxfp4(
        w_origin, MXFP4_BLOCK_SIZE
    )

    kb_dim = k // MXFP4_BLOCK_SIZE
    sfa = reorder_sf_to_cutlass(sfa_logical, m, kb_dim)
    sfb = reorder_sf_to_cutlass(sfb_logical, n, kb_dim)

    # MXFP4 has no per-expert alpha: the UE8M0 block scales carry the full scale.
    return SimpleNamespace(
        m=m,
        num_experts=num_experts,
        expert_ids=expert_ids,
        token_nums=token_nums,
        x_packed=x_packed,
        w_packed=w_packed,
        x_fp4_float=x_fp4_float,
        w_fp4_float=w_fp4_float,
        sfa=sfa,
        sfb=sfb,
        sm_count=-1,
        mode="w4a4-mxfp4",
    )


def assert_close_per_expert(output, ref_out, token_nums, tol):
    """Compare kernel and reference outputs expert by expert (contiguous spans)."""
    offset = 0
    for expert_id, m in enumerate(token_nums.tolist()):
        if m == 0:
            continue
        torch.testing.assert_close(
            output[offset : offset + m, :].to(torch.float32),
            ref_out[offset : offset + m, :].to(torch.float32),
            **tol,
            msg=lambda msg, e=expert_id, rows=m: f"expert {e} (m={rows}): {msg}",
        )
        offset += m


# =====================================================================
# Accuracy tests
# =====================================================================

FULL_N, FULL_K = 7168, 3584


@pytest.mark.parametrize(
    "m, n, k, num_experts, tile_m",
    [
        pytest.param(512, FULL_N, FULL_K, 2, 256, id="m512_e2"),
        pytest.param(4096, FULL_N, FULL_K, 20, 256, id="large_m4096_e20"),
        pytest.param(64, FULL_N, FULL_K, 20, 64, id="small_m64_e20_swapab"),
        pytest.param(128, 256, 256, 1, 128, id="m128_n256_k256_e1"),
    ],
)
def test_group_contiguous_w8a8_mxfp8(m, n, k, num_experts, tile_m):
    inp = make_w8a8_mxfp8_inputs(m, n, k, num_experts, tile_m)
    ref_out = ref_group_contiguous_gemm(
        inp.x,
        inp.w,
        inp.sfa_logical,
        inp.sfb_logical,
        inp.expert_ids,
        inp.num_experts,
        token_nums=inp.token_nums,
        bias=inp.bias,
    )
    output = group_contiguous_gemm_wrapper(
        inp.x,
        inp.w,
        inp.sfa,
        inp.sfb,
        inp.expert_ids,
        bias=inp.bias,
        sm_count=inp.sm_count,
        mode=inp.mode,
    )
    assert_close_per_expert(output, ref_out, inp.token_nums, W8A8_MXFP8_TOL)


@pytest.mark.parametrize(
    "m, n, k, num_experts, tile_m",
    [
        pytest.param(512, FULL_N, FULL_K, 2, 256, id="m512_e2"),
        pytest.param(4096, FULL_N, FULL_K, 20, 256, id="large_m4096_e20"),
        pytest.param(128, 256, 512, 1, 128, id="m128_n256_k512_e1"),
    ],
)
def test_group_contiguous_nvfp4(m, n, k, num_experts, tile_m):
    inp = make_nvfp4_inputs(m, n, k, num_experts, tile_m)
    ref_out = ref_group_contiguous_gemm_w4a4(
        inp.x_fp4_float, inp.w_fp4_float, inp.expert_ids, inp.num_experts
    )
    output = group_contiguous_gemm_wrapper(
        inp.x_packed,
        inp.w_packed,
        inp.sfa,
        inp.sfb,
        inp.expert_ids,
        alpha=inp.alpha,
        sm_count=inp.sm_count,
        mode=inp.mode,
    )
    assert_close_per_expert(output, ref_out, inp.token_nums, W4A4_TOL)


# =====================================================================
# Unsupported-input checks
# =====================================================================


# =====================================================================
# MXFP4 accuracy tests
# =====================================================================


@pytest.mark.parametrize(
    "m, n, k, num_experts, tile_m",
    [
        pytest.param(512, FULL_N, FULL_K, 2, 256, id="m512_e2"),
        pytest.param(2048, FULL_N, FULL_K, 8, 256, id="large_m2048_e8"),
        pytest.param(128, 256, 512, 1, 128, id="m128_n256_k512_e1"),
    ],
)
def test_group_contiguous_mxfp4(m, n, k, num_experts, tile_m):
    inp = make_mxfp4_inputs(m, n, k, num_experts, tile_m)
    ref_out = ref_group_contiguous_gemm_w4a4(
        inp.x_fp4_float, inp.w_fp4_float, inp.expert_ids, inp.num_experts
    )
    output = group_contiguous_gemm_wrapper(
        inp.x_packed,
        inp.w_packed,
        inp.sfa,
        inp.sfb,
        inp.expert_ids,
        sm_count=inp.sm_count,
        mode=inp.mode,
    )
    assert_close_per_expert(output, ref_out, inp.token_nums, W4A4_TOL)


# =====================================================================
# Unsupported-input checks
# =====================================================================


def run_mxfp4_alpha():
    m, n, k, num_experts = 128, 256, 512, 1
    x = torch.empty([m, k // 2], dtype=torch.uint8)
    w = torch.empty([num_experts, n, k // 2], dtype=torch.uint8)
    sfa = torch.empty([m, k // 32], dtype=torch.uint8)
    sfb = torch.empty([num_experts, n, k // 32], dtype=torch.uint8)
    expert_ids = torch.zeros([m], dtype=torch.int32)
    out = torch.empty([m, n], dtype=torch.bfloat16)
    alpha = torch.ones([num_experts], dtype=torch.float32)
    group_contiguous_gemm(
        x, w, sfa, sfb, expert_ids, out, None, alpha, -1, "w4a4-mxfp4"
    )


def run_mxfp4_bad_sf_k():
    m, n, k, num_experts = 128, 256, 512, 1
    x = torch.empty([m, k // 2], dtype=torch.uint8)
    w = torch.empty([num_experts, n, k // 2], dtype=torch.uint8)
    # NVFP4-shaped scales (K // 16) must be rejected for MXFP4.
    sfa = torch.empty([m, k // 16], dtype=torch.uint8)
    sfb = torch.empty([num_experts, n, k // 16], dtype=torch.uint8)
    expert_ids = torch.zeros([m], dtype=torch.int32)
    out = torch.empty([m, n], dtype=torch.bfloat16)
    group_contiguous_gemm(x, w, sfa, sfb, expert_ids, out, None, None, -1, "w4a4-mxfp4")


@pytest.mark.parametrize(
    "launch, expected",
    [
        pytest.param(run_mxfp4_alpha, "does not take alpha", id="mxfp4_alpha"),
        pytest.param(
            run_mxfp4_bad_sf_k,
            "scale-factor shape does not match mode",
            id="mxfp4_bad_sf_k",
        ),
    ],
)
def test_group_contiguous_mxfp4_errors(launch, expected):
    with pytest.raises(RuntimeError, match=re.escape(expected)):
        launch()


def run_w8a8_mxfp8_bad_m():
    m, n, k, num_experts = 129, 256, 128, 1
    x = torch.empty([m, k], dtype=torch.float8_e4m3fn)
    w = torch.empty([num_experts, n, k], dtype=torch.float8_e4m3fn)
    sfa = torch.empty([m, k // 32], dtype=torch.uint8)
    sfb = torch.empty([num_experts, n, k // 32], dtype=torch.uint8)
    expert_ids = torch.zeros([m], dtype=torch.int32)
    out = torch.empty([m, n], dtype=torch.bfloat16)
    group_contiguous_gemm(x, w, sfa, sfb, expert_ids, out, None, None, -1, "w8a8-mxfp8")


def run_nvfp4_bias():
    m, n, k, num_experts = 128, 256, 512, 1
    x = torch.empty([m, k // 2], dtype=torch.uint8)
    w = torch.empty([num_experts, n, k // 2], dtype=torch.uint8)
    sfa = torch.empty([m, k // 16], dtype=torch.uint8)
    sfb = torch.empty([num_experts, n, k // 16], dtype=torch.uint8)
    expert_ids = torch.zeros([m], dtype=torch.int32)
    out = torch.empty([m, n], dtype=torch.bfloat16)
    bias = torch.empty([num_experts, n], dtype=torch.bfloat16)
    alpha = torch.ones([num_experts], dtype=torch.float32)
    group_contiguous_gemm(
        x, w, sfa, sfb, expert_ids, out, bias, alpha, -1, "w4a4-nvfp4"
    )


def run_expert_ids_int64():
    m, n, k, num_experts = 128, 256, 128, 1
    x = torch.empty([m, k], dtype=torch.float8_e4m3fn)
    w = torch.empty([num_experts, n, k], dtype=torch.float8_e4m3fn)
    sfa = torch.empty([m, k // 32], dtype=torch.uint8)
    sfb = torch.empty([num_experts, n, k // 32], dtype=torch.uint8)
    expert_ids = torch.zeros([m], dtype=torch.int64)
    out = torch.empty([m, n], dtype=torch.bfloat16)
    group_contiguous_gemm(x, w, sfa, sfb, expert_ids, out, None, None, -1, "w8a8-mxfp8")


def run_bad_rank_output():
    m, n, k, num_experts = 128, 256, 128, 1
    x = torch.empty([m, k], dtype=torch.float8_e4m3fn)
    w = torch.empty([num_experts, n, k], dtype=torch.float8_e4m3fn)
    sfa = torch.empty([m, k // 32], dtype=torch.uint8)
    sfb = torch.empty([num_experts, n, k // 32], dtype=torch.uint8)
    expert_ids = torch.zeros([m], dtype=torch.int32)
    out = torch.empty([m, n, 1], dtype=torch.bfloat16)
    group_contiguous_gemm(x, w, sfa, sfb, expert_ids, out, None, None, -1, "w8a8-mxfp8")


@pytest.mark.parametrize(
    "launch, expected",
    [
        pytest.param(
            run_w8a8_mxfp8_bad_m,
            "No contiguous w8a8-mxfp8 tile for M=129",
            id="w8a8_mxfp8_bad_m",
        ),
        pytest.param(
            run_nvfp4_bias, "W4A4 mode does not support bias yet", id="w4a4_bias"
        ),
        pytest.param(
            run_expert_ids_int64,
            "expert_ids must have dtype torch.int32",
            id="expert_ids_int64",
        ),
        pytest.param(
            run_bad_rank_output, "output must be a 2D tensor", id="bad_rank_output"
        ),
    ],
)
def test_group_contiguous_unsupported_shapes(launch, expected):
    with pytest.raises(RuntimeError, match=re.escape(expected)):
        launch()


# =====================================================================
# Determinism tests
# =====================================================================


def test_group_contiguous_w8a8_mxfp8_determinism():
    """Determinism: identical inputs must produce bitwise-identical outputs."""
    inp = make_w8a8_mxfp8_inputs(512, FULL_N, FULL_K, 2, 256)
    out1, out2 = (
        group_contiguous_gemm_wrapper(
            inp.x,
            inp.w,
            inp.sfa,
            inp.sfb,
            inp.expert_ids,
            bias=inp.bias,
            sm_count=inp.sm_count,
            mode=inp.mode,
        )
        for _ in range(2)
    )
    assert torch.equal(
        out1, out2
    ), "W8A8-MXFP8 group_contiguous_gemm output is not deterministic"


def test_group_contiguous_mxfp4_determinism():
    """Determinism: identical inputs must produce bitwise-identical outputs."""
    inp = make_mxfp4_inputs(512, FULL_N, FULL_K, 2, 256)
    out1, out2 = (
        group_contiguous_gemm_wrapper(
            inp.x_packed,
            inp.w_packed,
            inp.sfa,
            inp.sfb,
            inp.expert_ids,
            sm_count=inp.sm_count,
            mode=inp.mode,
        )
        for _ in range(2)
    )
    assert torch.equal(
        out1, out2
    ), "MXFP4 group_contiguous_gemm output is not deterministic"


def test_group_contiguous_nvfp4_determinism():
    """Determinism: identical inputs must produce bitwise-identical outputs."""
    inp = make_nvfp4_inputs(512, FULL_N, FULL_K, 2, 256)
    out1, out2 = (
        group_contiguous_gemm_wrapper(
            inp.x_packed,
            inp.w_packed,
            inp.sfa,
            inp.sfb,
            inp.expert_ids,
            alpha=inp.alpha,
            sm_count=inp.sm_count,
            mode=inp.mode,
        )
        for _ in range(2)
    )
    assert torch.equal(
        out1, out2
    ), "W4A4 group_contiguous_gemm output is not deterministic"


if __name__ == "__main__":
    pytest.main([__file__])
