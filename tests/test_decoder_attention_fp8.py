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

import math

import numpy as np
import pytest
import torch

from halops import decoder_attention, write_cache_kv_decoder, write_cache_kv_encoder

from tests.utils.attn_utils import BLOCK_SIZE, HEAD_DIM, naive_attn, quant_fp8_q

MAX_INPUT_LEN = 131072
# Element-wise absolute tolerance vs the numpy FP8 reference.
ATOL = 0.1
K_PER_SEQ_CASES = [256, 512, 1024, 2048, 4096]


@pytest.fixture(scope="module")
def rotary_embs():
    """Identity RoPE table: cos=1, sin=0."""
    table = torch.zeros(2, MAX_INPUT_LEN, 64, dtype=torch.float32, device="cuda")
    table[0] = 1
    table[1] = 0
    return table


@pytest.fixture(scope="module")
def norm_gammas():
    """All-ones ``(q_norm_gamma, k_norm_gamma)`` RMSNorm weights."""
    q_norm_gamma = torch.ones(HEAD_DIM, dtype=torch.float32, device="cuda")
    k_norm_gamma = torch.ones(HEAD_DIM, dtype=torch.float32, device="cuda")
    return q_norm_gamma, k_norm_gamma


@pytest.mark.parametrize(
    "k_per_seq", K_PER_SEQ_CASES, ids=[f"k{k}" for k in K_PER_SEQ_CASES]
)
def test_accuracy(rotary_embs, norm_gammas, k_per_seq):
    q_norm_gamma, k_norm_gamma = norm_gammas
    head_num, kv_head_num = 1, 1
    mtp_steps = 0
    seq_lens_k = [k_per_seq]

    batch_size = len(seq_lens_k)
    q_per_seq = 1 + mtp_steps
    max_seq_k = max(seq_lens_k)
    max_blk = math.ceil(max_seq_k / BLOCK_SIZE) + 1
    num_blocks = batch_size * max_blk

    rng = np.random.default_rng(42)

    q_list = [
        rng.uniform(-2, 2, [q_per_seq, head_num, HEAD_DIM]).astype(np.float32)
        for _ in seq_lens_k
    ]
    k_list = [
        rng.uniform(-2, 2, [sl, kv_head_num, HEAD_DIM]).astype(np.float32)
        for sl in seq_lens_k
    ]
    v_list = [
        rng.uniform(-2, 2, [sl, kv_head_num, HEAD_DIM]).astype(np.float32)
        for sl in seq_lens_k
    ]

    q_stack = np.stack(q_list)
    decoder_q = torch.from_numpy(q_stack).squeeze(0).cuda()
    q_deq, _ = quant_fp8_q(q_stack, kv_head_num)

    ref_list = []
    for b in range(batch_size):
        for qi in range(q_per_seq):
            ref_list.append(
                naive_attn(q_deq[b][qi], k_list[b], v_list[b], head_num, kv_head_num)
            )
    ref = np.stack(ref_list)

    encoder_k = [x[0 : k_per_seq - 1 - mtp_steps] for x in k_list]
    decoder_k = [x[k_per_seq - mtp_steps - 1 :] for x in k_list]
    encoder_k = torch.from_numpy(np.concatenate(encoder_k)).cuda()
    decoder_k = torch.from_numpy(np.concatenate(decoder_k)).cuda()

    encoder_v = [x[0 : k_per_seq - 1 - mtp_steps] for x in v_list]
    decoder_v = [x[k_per_seq - mtp_steps - 1 :] for x in v_list]
    encoder_v = torch.from_numpy(np.concatenate(encoder_v)).cuda()
    decoder_v = torch.from_numpy(np.concatenate(decoder_v)).cuda()

    encoder_q = torch.randn(
        encoder_k.shape[0], head_num, HEAD_DIM, device="cuda", dtype=encoder_k.dtype
    )
    encoder_qkv = torch.cat([encoder_q, encoder_k, encoder_v], dim=1).to(torch.bfloat16)
    decoder_qkv = torch.cat([decoder_q, decoder_k, decoder_v], dim=1).to(torch.bfloat16)
    cu_seq_q = (torch.arange(batch_size + 1, device="cuda") * (k_per_seq - 1)).to(
        torch.int32
    )
    cu_seq_k = (torch.arange(batch_size + 1, device="cuda") * (k_per_seq - 1)).to(
        torch.int32
    )
    seq_len_encoder = torch.full(
        (batch_size,), k_per_seq - 1, dtype=torch.int32, device="cuda"
    )
    seq_len_decoder = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
    cachek = torch.zeros(
        num_blocks, kv_head_num, BLOCK_SIZE, HEAD_DIM, dtype=torch.uint8, device="cuda"
    )
    cachev = torch.zeros(
        num_blocks, kv_head_num, BLOCK_SIZE, HEAD_DIM, dtype=torch.uint8, device="cuda"
    )
    block_tables = (
        torch.arange(num_blocks, device="cuda")
        .reshape(batch_size, max_blk)
        .to(torch.int32)
    )
    q_input_encoder = torch.zeros(
        k_per_seq - 1, head_num, HEAD_DIM, dtype=torch.bfloat16, device="cuda"
    )
    k_input_encoder = torch.zeros(
        k_per_seq - 1, kv_head_num, HEAD_DIM, dtype=torch.bfloat16, device="cuda"
    )
    v_input_encoder = torch.zeros(
        k_per_seq - 1, kv_head_num, HEAD_DIM, dtype=torch.bfloat16, device="cuda"
    )
    kv_dequant_scale = torch.zeros(num_blocks, kv_head_num, 4, device="cuda")

    # FP8 cache write interface: encoder stage still takes BF16 qkv input; the kernel
    # internally quantizes K/V to FP8 via block-wise abs-max into cachek/cachev, and writes scales to kv_dequant_scale.
    write_cache_kv_encoder(
        encoder_qkv,
        cu_seq_q,
        cu_seq_k,
        rotary_embs,
        seq_len_encoder,
        seq_len_decoder,
        cachek,
        cachev,
        block_tables,
        q_input_encoder,
        k_input_encoder,
        v_input_encoder,
        kv_dequant_scale,
        q_norm_gamma,
        k_norm_gamma,
        int(head_num),
        int(kv_head_num),
        int(HEAD_DIM),
        int(k_per_seq - 1),
        int(k_per_seq - 1),
        int(MAX_INPUT_LEN),
        "block_wise_fp8",
    )

    cu_seq_q = (torch.arange(batch_size + 1, device="cuda") * (mtp_steps + 1)).to(
        torch.int32
    )
    seq_lens_kv = torch.full((batch_size,), k_per_seq, dtype=torch.int32, device="cuda")
    seq_len_encoder = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
    seq_len_decoder = torch.full(
        (batch_size,), k_per_seq - 1, dtype=torch.int32, device="cuda"
    )
    # FP8 decode cache write interface: returned q_output is in [batch, kv_head_num, 64, head_dim]
    # FP8 GQA-folded layout; q_dequant_scale is used by the subsequent FP8 attention kernel to recover Q numerical range.
    q_output, q_dequant_scale = write_cache_kv_decoder(
        decoder_qkv,
        cu_seq_q,
        seq_len_encoder,
        seq_len_decoder,
        rotary_embs,
        cachek,
        cachev,
        block_tables,
        kv_dequant_scale,
        q_norm_gamma,
        k_norm_gamma,
        int(head_num),
        int(kv_head_num),
        int(HEAD_DIM),
        int(MAX_INPUT_LEN),
        "block_wise_fp8",
    )

    out1 = torch.zeros(
        batch_size * q_per_seq, head_num * HEAD_DIM, dtype=torch.bfloat16, device="cuda"
    )
    out2 = torch.zeros(
        batch_size * q_per_seq, head_num * HEAD_DIM, dtype=torch.bfloat16, device="cuda"
    )

    for out in [out1, out2]:
        # FP8 decode attention forward: read FP8 q_output and FP8 paged KV cache, dequantize
        # using q_dequant_scale/kv_dequant_scale, compute attention, output BF16.
        decoder_attention(
            q_output,
            cu_seq_q,
            seq_lens_kv,
            seq_len_encoder,
            seq_len_decoder,
            cachek,
            cachev,
            block_tables,
            out,
            q_dequant_scale,
            kv_dequant_scale,
            int(head_num),
            int(kv_head_num),
            int(HEAD_DIM),
            int(max_seq_k),
            int(mtp_steps),
            "block_wise_fp8",
        )

    # Compare with reference
    out = out1.float().cpu().reshape(batch_size * q_per_seq, head_num, HEAD_DIM)
    torch.testing.assert_close(out, torch.from_numpy(ref), rtol=0.0, atol=ATOL)

    # Check for determinism within one decode call.
    assert torch.equal(out1, out2), "decoder_attention output is not deterministic"


if __name__ == "__main__":
    pytest.main([__file__])
