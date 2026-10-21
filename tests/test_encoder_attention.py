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

from halops import encoder_attention
from tests.utils.attn_utils import HEAD_DIM

DTYPE = torch.bfloat16
HEAD_CASES = [(14, 1), (56, 4)]
HEAD_IDS = [f"h{h}-kv{kv}" for h, kv in HEAD_CASES]


def causal_reference(q_input, k_input, v_input, cu_seq_q, cu_seq_k):
    """Per-batch causal prefill attention reference (float32, GQA-aware).

    Computes the full score matrix per sequence, so it is only used for small
    sequence lengths in the correctness tests.
    """
    bsz = cu_seq_q.shape[0] - 1
    q_token_sum, num_head, head_dim = q_input.shape
    _, num_kv_head, _ = k_input.shape
    gqa = num_head // num_kv_head
    scale = 1.0 / np.sqrt(head_dim)
    out = np.zeros([q_token_sum, num_head, head_dim], dtype=np.float32)
    cu_q = cu_seq_q.cpu().numpy()
    cu_k = cu_seq_k.cpu().numpy()
    q_np = q_input.float().cpu().numpy()
    k_np = k_input.float().cpu().numpy()
    v_np = v_input.float().cpu().numpy()
    for bi in range(bsz):
        q = q_np[cu_q[bi] : cu_q[bi + 1]]  # [lq, num_head, hd]
        k = k_np[cu_k[bi] : cu_k[bi + 1]]  # [lk, num_kv_head, hd]
        v = v_np[cu_k[bi] : cu_k[bi + 1]]
        lq, lk = q.shape[0], k.shape[0]
        for h in range(num_head):
            kv_h = h // gqa
            scores = q[:, h, :] @ k[:, kv_h, :].T * scale  # [lq, lk]
            # bottom-right causal alignment: query i sees keys [0, lk - lq + i]
            row = np.arange(lq)[:, None]
            col = np.arange(lk)[None, :]
            valid = col <= (lk - lq + row)
            scores = np.where(valid, scores, -1e9)
            scores -= scores.max(axis=-1, keepdims=True)
            w = np.exp(scores)
            w /= w.sum(axis=-1, keepdims=True)
            out[cu_q[bi] : cu_q[bi + 1], h, :] = w @ v[:, kv_h, :]
    return out


def _run_encoder(seq_len_q, head_num, kv_head_num):
    cu_seq_q = torch.tensor([0, seq_len_q], dtype=torch.int32, device="cuda")
    cu_seq_k = torch.tensor([0, seq_len_q], dtype=torch.int32, device="cuda")
    q = torch.randn(seq_len_q, head_num, HEAD_DIM, dtype=DTYPE, device="cuda")
    k = torch.randn(seq_len_q, kv_head_num, HEAD_DIM, dtype=DTYPE, device="cuda")
    v = torch.randn(seq_len_q, kv_head_num, HEAD_DIM, dtype=DTYPE, device="cuda")
    # per-row valid length == causal mask (query i attends to keys [0, i]).
    mask = (torch.arange(seq_len_q, device="cuda") + 1).to(torch.int32)
    out = torch.empty(seq_len_q, head_num, HEAD_DIM, dtype=DTYPE, device="cuda")
    encoder_attention(q, k, v, cu_seq_q, cu_seq_k, out, mask)
    return q, k, v, cu_seq_q, cu_seq_k, out


def _seq_id(seq_len):
    return f"seq{seq_len}"


@pytest.mark.parametrize("seq_len_q", [128, 256, 512], ids=_seq_id)
@pytest.mark.parametrize("head_num,kv_head_num", HEAD_CASES, ids=HEAD_IDS)
def test_encoder_attention_correctness(seq_len_q, head_num, kv_head_num):
    q, k, v, cu_seq_q, cu_seq_k, out = _run_encoder(seq_len_q, head_num, kv_head_num)
    ref = causal_reference(q, k, v, cu_seq_q, cu_seq_k)
    torch.testing.assert_close(
        out.float().cpu(),
        torch.from_numpy(ref),
        rtol=2e-2,
        atol=2e-2,
    )


@pytest.mark.parametrize("seq_len_q", [8192, 32768], ids=_seq_id)
def test_encoder_attention_long_seq_smoke(seq_len_q):
    # Long sequences: only validate that the kernel runs and produces finite
    # output of the expected shape. A full O(S^2) reference is intentionally
    # avoided here.
    head_num, kv_head_num = 56, 4
    _, _, _, _, _, out = _run_encoder(seq_len_q, head_num, kv_head_num)
    assert out.shape == (seq_len_q, head_num, HEAD_DIM)
    assert torch.isfinite(out.float()).all()


@pytest.mark.parametrize("head_num,kv_head_num", HEAD_CASES, ids=HEAD_IDS)
@pytest.mark.parametrize("seq_len_q", [128, 512], ids=_seq_id)
def test_encoder_attention_determinism(seq_len_q, head_num, kv_head_num):
    # Same inputs must yield bit-identical outputs across runs. Build the
    # fixed q/k/v/mask once (cannot reuse _run_encoder, which re-randomizes),
    # then invoke the kernel twice into separate output buffers.
    cu_seq_q = torch.tensor([0, seq_len_q], dtype=torch.int32, device="cuda")
    cu_seq_k = torch.tensor([0, seq_len_q], dtype=torch.int32, device="cuda")
    q = torch.randn(seq_len_q, head_num, HEAD_DIM, dtype=DTYPE, device="cuda")
    k = torch.randn(seq_len_q, kv_head_num, HEAD_DIM, dtype=DTYPE, device="cuda")
    v = torch.randn(seq_len_q, kv_head_num, HEAD_DIM, dtype=DTYPE, device="cuda")
    mask = (torch.arange(seq_len_q, device="cuda") + 1).to(torch.int32)

    def _run():
        out = torch.empty(seq_len_q, head_num, HEAD_DIM, dtype=DTYPE, device="cuda")
        encoder_attention(q, k, v, cu_seq_q, cu_seq_k, out, mask)
        return out

    out1 = _run()
    out2 = _run()
    assert torch.equal(out1, out2)


if __name__ == "__main__":
    pytest.main([__file__])
