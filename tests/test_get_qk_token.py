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

import pytest
import torch

from halops import get_qk_token


def reference_get_qk_token(enc, dec, q_tokens_per_sample, max_batch):
    """Pure-Python reference mirroring get_qk_token_kernel.

    Args:
        enc: list[int], seq_lens_encoder per sample.
        dec: list[int], seq_lens_decoder per sample.
        q_tokens_per_sample: list[int], Q tokens per sample (cu_seq_q deltas).
        max_batch: length of the seq_lens_kv output tensor.

    Returns:
        Tuple ``(summary, seq_lens_kv, actual_cu_seq_k)`` where ``summary`` is
        ``[max_enc, max_dec, total_q, total_k]``.
    """
    batch = len(enc)
    cu_seq_q = [0]
    for q in q_tokens_per_sample:
        cu_seq_q.append(cu_seq_q[-1] + q)

    seq_lens_kv = [0] * max_batch
    max_enc = 0
    for i in range(batch):
        max_enc = max(max_enc, enc[i])
        q_token = cu_seq_q[i + 1] - cu_seq_q[i]
        if enc[i] > 0 or dec[i] == 0 or q_token <= 0:
            seq_lens_kv[i] = 0
        else:
            seq_lens_kv[i] = dec[i] + q_token
    max_dec = max(seq_lens_kv[:batch]) if batch > 0 else 0

    actual_cu = [0] * (batch + 1)
    k_tokens = 0
    q_total = 0
    if max_enc > 0 or max_dec > 0:
        for i in range(batch):
            actual_cu[i] = k_tokens
            q_token = cu_seq_q[i + 1] - cu_seq_q[i]
            if enc[i] > 0:
                k_tokens += dec[i] + q_token
            else:
                k_tokens += q_token
            q_total += q_token
        actual_cu[batch] = k_tokens
    return [max_enc, max_dec, q_total, k_tokens], seq_lens_kv, actual_cu


def _call(enc, dec, q_tokens_per_sample, max_batch, kv_head_num=1):
    batch = len(enc)
    cu_seq_q_list = [0]
    for q in q_tokens_per_sample:
        cu_seq_q_list.append(cu_seq_q_list[-1] + q)

    enc_t = torch.tensor(enc, dtype=torch.int32, device="cuda")
    dec_t = torch.tensor(dec, dtype=torch.int32, device="cuda")
    cu_seq_q = torch.tensor(cu_seq_q_list, dtype=torch.int32, device="cuda")
    cu_seq_k = torch.zeros(batch + 1, dtype=torch.int32, device="cuda")
    actual_cu_seq_k = torch.zeros(batch + 1, dtype=torch.int32, device="cuda")
    seq_lens_kv = torch.zeros(max_batch, dtype=torch.int32, device="cuda")

    (summary,) = get_qk_token(
        enc_t, dec_t, cu_seq_q, cu_seq_k, actual_cu_seq_k, seq_lens_kv, kv_head_num
    )
    return summary.cpu(), seq_lens_kv.cpu(), actual_cu_seq_k.cpu()


def _i32(values):
    return torch.tensor(values, dtype=torch.int32, device="cpu")


@pytest.mark.parametrize(
    "enc,dec,qtok",
    [
        pytest.param([128], [0], [128], id="single_encoder"),
        pytest.param([0], [512], [1], id="single_decoder"),
        pytest.param([128, 0], [0, 512], [128, 1], id="mixed_enc_dec"),
        pytest.param(
            [64, 96, 0, 0], [0, 0, 200, 300], [64, 96, 1, 2], id="mixed_batch"
        ),
        pytest.param([0, 0], [0, 128], [1, 1], id="dec_zero_kv_zero"),
    ],
)
def test_get_qk_token_correctness(enc, dec, qtok):
    max_batch = len(enc) + 4
    summary, seq_lens_kv, actual_cu = _call(enc, dec, qtok, max_batch)
    ref_summary, ref_kv, ref_cu = reference_get_qk_token(enc, dec, qtok, max_batch)
    assert torch.equal(summary, _i32(ref_summary)), f"{summary=} {ref_summary=}"
    assert torch.equal(seq_lens_kv, _i32(ref_kv)), f"{seq_lens_kv=} {ref_kv=}"
    assert torch.equal(actual_cu, _i32(ref_cu)), f"{actual_cu=} {ref_cu=}"


def test_get_qk_token_padding_zeroed():
    # Entries of seq_lens_kv beyond batch_size must be zeroed by the kernel.
    enc, dec, qtok = [0, 0], [100, 200], [1, 1]
    max_batch = 8
    _, seq_lens_kv, _ = _call(enc, dec, qtok, max_batch)
    assert torch.all(seq_lens_kv[len(enc) :] == 0), f"{seq_lens_kv=}"


def test_get_qk_token_zero_q_token():
    # q_token <= 0 for a decoder sample forces seq_lens_kv = 0 for that sample.
    enc, dec, qtok = [0], [256], [0]
    max_batch = 4
    _, seq_lens_kv, _ = _call(enc, dec, qtok, max_batch)
    assert seq_lens_kv[0] == 0, f"{seq_lens_kv=}"


def test_get_qk_token_determinism():
    enc, dec, qtok = [128, 0, 64], [0, 512, 0], [128, 1, 64]
    max_batch = 8
    s1, kv1, cu1 = _call(enc, dec, qtok, max_batch)
    s2, kv2, cu2 = _call(enc, dec, qtok, max_batch)
    assert torch.equal(s1, s2)
    assert torch.equal(kv1, kv2)
    assert torch.equal(cu1, cu2)


def test_get_qk_token_wrong_dtype_raises():
    # data_ptr<int>() validates the scalar type, so a float input is rejected.
    batch = 2
    enc = torch.zeros(batch, dtype=torch.float32, device="cuda")
    dec = torch.zeros(batch, dtype=torch.int32, device="cuda")
    cu_seq_q = torch.tensor([0, 1, 2], dtype=torch.int32, device="cuda")
    cu_seq_k = torch.zeros(batch + 1, dtype=torch.int32, device="cuda")
    actual_cu_seq_k = torch.zeros(batch + 1, dtype=torch.int32, device="cuda")
    seq_lens_kv = torch.zeros(batch, dtype=torch.int32, device="cuda")
    with pytest.raises(RuntimeError, match=re.escape("expected scalar type Int")):
        get_qk_token(enc, dec, cu_seq_q, cu_seq_k, actual_cu_seq_k, seq_lens_kv, 1)


if __name__ == "__main__":
    pytest.main([__file__])
