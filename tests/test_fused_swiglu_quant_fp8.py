# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for fused_swiglu_quant on the FP8 (group-32, UE8M0 scale) path."""

import numpy as np
import pytest
import torch

from tests.utils.swiglu_utils import (
    get_sfatom_valid_byte_mask_fp8,
    get_valid_row_mask,
    run_single_case_fp8,
    swiglu_and_quant_fused,
)


# (group_num, group_size, H)
SHAPE_CASES = [
    (4, 128, 2048),
    (20, 128, 14336),
    (20, 256, 14336),
    (20, 1024, 14336),
    (8, 512, 7168),
]

# H=2*hidden; FP8 num_iters = hidden / 128
# num_iters % 4 == 2: hidden in {256, 768} -> H in {512, 1536}
# odd num_iters: hidden in {128, 384, 896} -> H in {256, 768, 1792}
REMAINDER_CASES = [
    (4, 128, 512),  # num_iters=2 (pair-remainder)
    (4, 128, 1536),  # num_iters=6 (one quad + one pair)
    (4, 128, 256),  # num_iters=1 (single-remainder only)
    (4, 128, 768),  # num_iters=3 (one pair + single)
    (4, 128, 1792),  # num_iters=7 (one quad + pair + single)
]

EDGE_TOKEN_NUMS = [
    [0, 0, 0, 0],
    [0, 128, 0, 64],
    [128, 128, 128, 128],
    [1, 1, 1, 1],
]

INVALID_INPUT_CASES = [
    "float32_input",
    "int64_token_nums",
    "non_contiguous_input",
    "cpu_tensors",
    "hidden_not_divisible_by_128",
]


def _shape_id(case):
    group_num, group_size, H = case
    return f"E{group_num}_M{group_size}_H{H}"


def _make_invalid_input(case):
    """Build (x, token_nums) violating one host-side TORCH_CHECK."""
    E, M, H = 4, 128, 2048
    base_x = torch.randn([E * M, H]).to(torch.bfloat16)
    base_tn = torch.tensor([M] * E, dtype=torch.int32)
    if case == "float32_input":
        return base_x.float(), base_tn
    if case == "int64_token_nums":
        return base_x, base_tn.long()
    if case == "non_contiguous_input":
        # Take every-other row to break row contiguity
        big_x = torch.randn([E * M * 2, H]).to(torch.bfloat16)
        nc = big_x[::2]
        assert not nc.is_contiguous()
        return nc, base_tn
    if case == "cpu_tensors":
        return base_x.cpu(), base_tn.cpu()
    if case == "hidden_not_divisible_by_128":
        # FP8 hidden not divisible by 128: hidden=64 -> H=128
        return torch.randn([E * M, 128]).to(torch.bfloat16), base_tn
    raise ValueError(f"unknown invalid-input case: {case}")


def test_sanity_all_valid():
    """All tokens valid sanity check."""
    run_single_case_fp8(4, 128, 2048, token_nums_list=[128, 128, 128, 128])


@pytest.mark.parametrize(
    "group_num,group_size,H",
    SHAPE_CASES,
    ids=[_shape_id(c) for c in SHAPE_CASES],
)
def test_cases(group_num, group_size, H):
    """Representative shape sweep with random token_nums."""
    run_single_case_fp8(group_num, group_size, H)


@pytest.mark.parametrize("all_valid", [True, False], ids=["all_valid", "random"])
@pytest.mark.parametrize(
    "E,M,H",
    REMAINDER_CASES,
    ids=[_shape_id(c) for c in REMAINDER_CASES],
)
def test_remainder_branches(E, M, H, all_valid):
    """Cover FP8 tail-remainder paths."""
    token_nums_list = [M] * E if all_valid else None
    run_single_case_fp8(E, M, H, token_nums_list=token_nums_list)


@pytest.mark.parametrize(
    "token_nums_list",
    EDGE_TOKEN_NUMS,
    ids=["-".join(map(str, tn)) for tn in EDGE_TOKEN_NUMS],
)
def test_token_nums_edge_cases(token_nums_list):
    """Edge token_nums distributions."""
    E, M, H = 4, 128, 2048
    run_single_case_fp8(E, M, H, token_nums_list=token_nums_list)


@pytest.mark.parametrize("E", [128, 256], ids=lambda e: f"E{e}")
def test_many_experts(E):
    """Stress binary-search and SMEM with large expert counts."""
    run_single_case_fp8(E, 128, 2048)


@pytest.mark.parametrize("case", INVALID_INPUT_CASES)
def test_input_validation(case):
    """Negative cases corresponding to host-side TORCH_CHECKs.

    Note: token_nums value-range (0 <= n <= group_size) is a caller
    precondition and intentionally not validated on the host to avoid a
    device sync, so there are no negative cases for it here.
    """
    x, token_nums = _make_invalid_input(case)
    with pytest.raises((RuntimeError, TypeError)):
        swiglu_and_quant_fused(x, token_nums, is_fp4=False)


def test_determinism():
    """Bit-exact repeatability for identical inputs (valid rows only).

    Invalid rows in out_quant/out_scale are uninitialized (torch.empty),
    so only compare positions that the kernel actually writes.
    """
    E, M, H = 4, 128, 2048
    hidden = H // 2
    x = torch.randn([E * M, H]).to(torch.bfloat16)
    tn_list = [np.random.randint(1, M + 1) for _ in range(E)]
    tn = torch.tensor(tn_list, dtype=torch.int32)
    tn_np = np.array(tn_list, dtype=np.int32)
    valid_row = get_valid_row_mask(tn_np, M)

    n_scale = ((hidden // 128) + 3) // 4
    kb_dim = n_scale * 16
    sf_mask = get_sfatom_valid_byte_mask_fp8(tn_np, M, kb_dim, hidden)

    q0, s0 = swiglu_and_quant_fused(x, tn, is_fp4=False)
    q0_v = q0.view(torch.uint8).cpu().numpy()[valid_row]
    s0_v = s0.cpu().numpy().reshape(-1)[sf_mask]
    for _ in range(2):
        q, s = swiglu_and_quant_fused(x, tn, is_fp4=False)
        q_v = q.view(torch.uint8).cpu().numpy()[valid_row]
        s_v = s.cpu().numpy().reshape(-1)[sf_mask]
        assert np.array_equal(q_v, q0_v), "non-deterministic quant output (fp8)"
        assert np.array_equal(s_v, s0_v), "non-deterministic scale (fp8)"


if __name__ == "__main__":
    pytest.main([__file__])
