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

import numpy as np
import pytest
import torch

from halops import prefill_permute_to_masked_gemm


HIDDEN = 7168
NUM_LOCAL_EXPERTS = 20
TOPK = 4

# (num_tokens, max_token_num)
CASES = [(64, 128), (128, 256)]
CASE_IDS = [f"{t}-{m}" for t, m in CASES]


def _make_inputs(num_tokens):
    """Build (x, scale, topk_ids) matching the kernel's expected layout."""
    hidden_scale = HIDDEN // 128 // 4
    x = torch.randn([num_tokens, HIDDEN], device="cuda").to(torch.float8_e4m3fn)
    scale = torch.randint(
        0, 256, [num_tokens, hidden_scale], dtype=torch.int32, device="cuda"
    )
    topk_ids = torch.randint(
        0, NUM_LOCAL_EXPERTS, [num_tokens, TOPK], dtype=torch.int64, device="cuda"
    )
    return x, scale, topk_ids


def _gather_via_indice_map(permute_x, indice_map, src_x):
    """Return (gathered, expected) float arrays for the valid (!= -1) slots.

    permute_x rows are traced back to their source token through indice_map,
    so this is invariant to the (non-deterministic) intra-expert placement
    order produced by atomicAdd.
    """
    indice_map_np = indice_map.cpu().numpy()
    valid = indice_map_np != -1
    token_idx, slot_idx = np.where(valid)
    flat = permute_x.reshape([-1, HIDDEN]).to(torch.float32).cpu().numpy()
    src = src_x.to(torch.float32).cpu().numpy()
    dst = indice_map_np[token_idx, slot_idx]
    return flat[dst], src[token_idx]


@pytest.mark.parametrize("num_tokens,max_token_num", CASES, ids=CASE_IDS)
def test_prefill_permute_determinism(num_tokens, max_token_num):
    """Two identical calls must agree on counts and gather-equivalent contents.

    Because tokens within one expert can be scattered to different offsets
    across runs (atomicAdd ordering is not fixed), bitwise equality of
    ``permute_x`` is not guaranteed. Instead we require:
      1. ``token_nums_per_expert`` identical across runs (bitwise), and
      2. each run's ``permute_x``, gathered through its own ``indice_map``,
         reproduces the source ``x`` exactly.
    """
    x, scale, topk_ids = _make_inputs(num_tokens)

    px1, _, im1, cnt1 = prefill_permute_to_masked_gemm(
        x, scale, topk_ids, NUM_LOCAL_EXPERTS, max_token_num
    )
    px2, _, im2, cnt2 = prefill_permute_to_masked_gemm(
        x, scale, topk_ids, NUM_LOCAL_EXPERTS, max_token_num
    )

    # 1. Per-expert token counts must be identical across the two runs.
    assert torch.equal(cnt1, cnt2), "token_nums_per_expert not deterministic"

    # 2. Each run must be internally consistent: permute_x gathered through its
    #    own indice_map reconstructs the source x.
    g1, e1 = _gather_via_indice_map(px1, im1, x)
    assert np.array_equal(g1, e1), "run 1: permute_x does not reproduce source x"
    g2, e2 = _gather_via_indice_map(px2, im2, x)
    assert np.array_equal(g2, e2), "run 2: permute_x does not reproduce source x"


@pytest.mark.parametrize("num_tokens,max_token_num", CASES, ids=CASE_IDS)
def test_prefill_permute_indice_map_invalid_slots(num_tokens, max_token_num):
    """topk_ids == -1 entries must map to -1 in the returned indice_map."""
    x, scale, topk_ids = _make_inputs(num_tokens)

    topk_np = topk_ids.cpu().numpy()
    num_invalid = max(1, (num_tokens * TOPK) // 4)
    flat_idx = np.random.default_rng(0).choice(
        num_tokens * TOPK, num_invalid, replace=False
    )
    topk_np.reshape(-1)[flat_idx] = -1
    topk_ids = torch.from_numpy(topk_np).cuda()

    _, _, indice_map, _ = prefill_permute_to_masked_gemm(
        x, scale, topk_ids, NUM_LOCAL_EXPERTS, max_token_num
    )

    indice_np = indice_map.cpu().numpy()
    invalid_mask = topk_np == -1
    assert np.all(indice_np[invalid_mask] == -1), "invalid slots must map to -1"


if __name__ == "__main__":
    pytest.main([__file__])
