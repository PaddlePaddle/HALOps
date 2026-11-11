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

from halops.indexer import mqa_logits_fused_topk
from tests.utils.gemm_utils import per_token_cast_to_mxfp4


TOP_K = 128
HEADS = 8
HEAD_DIM = 128
NUM_KEYS = 256


def _pack_ue8m0(scales):
    """Pack four UE8M0 bytes into the int32 layout consumed by the indexer."""
    scales = scales.to(torch.int32)
    return (
        scales[..., 0]
        | (scales[..., 1] << 8)
        | (scales[..., 2] << 16)
        | (scales[..., 3] << 24)
    )


def _reference_scores(q, kv, weights):
    """Compute the FP32 reference score used by the fused operator."""
    logits = torch.einsum("mhd,nd->mhn", q.float(), kv.float())
    return torch.relu(logits).mul(weights[:, :, None]).sum(dim=1)


def _assert_matches_reference(actual, q, kv, weights, ks, ke, offsets, *, top_k=TOP_K):
    scores = _reference_scores(q, kv, weights)
    for row in range(q.shape[0]):
        start = int(ks[row])
        end = int(ke[row])
        length = end - start
        offset = 0 if offsets is None else int(offsets.reshape(-1)[row])
        if length >= top_k:
            expected = torch.topk(
                scores[row, start:end], top_k, largest=True, sorted=False
            ).indices.to(torch.int64)
            expected += offset
        else:
            valid = torch.arange(length, device=actual.device, dtype=torch.int64)
            valid += offset
            padding = torch.full(
                (top_k - length,), -1, device=actual.device, dtype=torch.int64
            )
            expected = torch.cat((valid, padding))

        actual_row = torch.sort(actual[row].to(torch.int64)).values
        expected = torch.sort(expected).values
        torch.testing.assert_close(actual_row, expected)


def test_mqa_logits_fused_fp8_ragged_offsets():
    """Validate FP8 scoring, ragged bounds, short rows, and int64 offsets."""
    torch.manual_seed(2026)
    q = torch.randn((2, HEADS, HEAD_DIM), device="cuda").to(torch.float8_e4m3fn)
    kv = torch.randn((NUM_KEYS, HEAD_DIM), device="cuda").to(torch.float8_e4m3fn)
    weights = torch.linspace(0.5, 1.5, HEADS, device="cuda").repeat(2, 1)
    ks = torch.tensor([0, 32], device="cuda", dtype=torch.int32)
    # The second row has fewer than TOP_K legal keys and must be padded with -1.
    ke = torch.tensor([NUM_KEYS, 96], device="cuda", dtype=torch.int32)
    offsets = torch.tensor([[1000], [2000]], device="cuda", dtype=torch.int64)

    actual = mqa_logits_fused_topk(
        q,
        None,
        (kv, torch.ones(NUM_KEYS, device="cuda")),
        weights,
        ks,
        ke,
        offsets,
        TOP_K,
        scores_bf16=False,
        num_sms=1,
        kv_sample=(64, 4),
    )
    torch.cuda.synchronize()

    assert actual.shape == (2, TOP_K)
    _assert_matches_reference(actual, q, kv, weights, ks, ke, offsets)


def test_mqa_logits_fused_mxfp8_scales():
    """Validate the packed int32 query and key scale path for MXFP8."""
    torch.manual_seed(2027)
    q = torch.randn((1, HEADS, HEAD_DIM), device="cuda").to(torch.float8_e4m3fn)
    kv = torch.randn((NUM_KEYS, HEAD_DIM), device="cuda").to(torch.float8_e4m3fn)
    q_sf = torch.full((1, HEADS), 0x7F7F7F7F, device="cuda", dtype=torch.int32)
    kv_sf = torch.full((NUM_KEYS,), 0x7F7F7F7F, device="cuda", dtype=torch.int32)
    weights = torch.linspace(0.75, 1.25, HEADS, device="cuda").unsqueeze(0)
    ks = torch.zeros(1, device="cuda", dtype=torch.int32)
    ke = torch.full((1,), NUM_KEYS, device="cuda", dtype=torch.int32)
    offsets = torch.tensor([17], device="cuda", dtype=torch.int32)

    actual = mqa_logits_fused_topk(
        q,
        q_sf,
        (kv, kv_sf),
        weights,
        ks,
        ke,
        offsets,
        TOP_K,
        scores_bf16=False,
        num_sms=1,
        kv_sample=(64, 4),
    )
    torch.cuda.synchronize()

    assert actual.shape == (1, TOP_K)
    _assert_matches_reference(actual, q, kv, weights, ks, ke, offsets)


def test_mqa_logits_fused_mxfp4_scales():
    """Validate packed FP4 values and four-byte-per-head UE8M0 scales."""
    torch.manual_seed(2028)
    q_bf16 = torch.randn((HEADS, HEAD_DIM), dtype=torch.bfloat16)
    kv_bf16 = torch.randn((NUM_KEYS, HEAD_DIM), dtype=torch.bfloat16)
    q_packed, q_scale, q_dequant = per_token_cast_to_mxfp4(q_bf16)
    kv_packed, kv_scale, kv_dequant = per_token_cast_to_mxfp4(kv_bf16)

    q = q_packed.reshape(1, HEADS, HEAD_DIM // 2).cuda()
    q_sf = _pack_ue8m0(q_scale).reshape(1, HEADS).cuda()
    kv = kv_packed.cuda()
    kv_sf = _pack_ue8m0(kv_scale).cuda()
    weights = torch.linspace(0.5, 1.5, HEADS, device="cuda").unsqueeze(0)
    ks = torch.zeros(1, device="cuda", dtype=torch.int32)
    ke = torch.full((1,), NUM_KEYS, device="cuda", dtype=torch.int32)

    actual = mqa_logits_fused_topk(
        q,
        q_sf,
        (kv, kv_sf),
        weights,
        ks,
        ke,
        None,
        TOP_K,
        scores_bf16=False,
        num_sms=1,
        kv_sample=(64, 4),
    )
    torch.cuda.synchronize()

    q_reference = q_dequant.reshape(1, HEADS, HEAD_DIM).cuda()
    kv_reference = kv_dequant.cuda()
    assert actual.shape == (1, TOP_K)
    _assert_matches_reference(actual, q_reference, kv_reference, weights, ks, ke, None)


@pytest.mark.parametrize("top_k", [0, 127, 2049])
def test_mqa_logits_fused_rejects_invalid_top_k(top_k):
    q = torch.empty((1, HEADS, HEAD_DIM), dtype=torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="top_k"):
        mqa_logits_fused_topk(
            q,
            None,
            (
                torch.empty((NUM_KEYS, HEAD_DIM), dtype=torch.float8_e4m3fn),
                torch.empty(NUM_KEYS, dtype=torch.float32),
            ),
            torch.empty((1, HEADS), dtype=torch.float32),
            torch.zeros(1, dtype=torch.int32),
            torch.full((1,), NUM_KEYS, dtype=torch.int32),
            None,
            top_k,
            scores_bf16=False,
            num_sms=1,
        )
