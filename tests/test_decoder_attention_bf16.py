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

from halops import decoder_attention, write_cache_kv_decoder, write_cache_kv_encoder

from tests.utils.attn_utils import BLOCK_SIZE, HEAD_DIM, reference_attention

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_SEQ_LEN = 131072
DTYPE = torch.bfloat16

# Correctness / determinism cases.
CORRECTNESS_CASES = [
    # batch_size, seq_len, head_num, kv_head_num, mtp_step
    (1, 128, 14, 1, 0),
    (4, 512, 14, 1, 0),
    (1, 1024, 14, 1, 1),
    (2, 256, 56, 4, 0),
]
DETERMINISM_CASES = [
    # batch_size, seq_len_encoder, head_num, kv_head_num
    (1, 256, 14, 1),
    (2, 512, 56, 4),
]


def _case_id(case):
    """Readable pytest id, e.g. ``bs1-seq128-h14-kv1-mtp0``."""
    names = ("bs", "seq", "h", "kv", "mtp")
    return "-".join(f"{n}{v}" for n, v in zip(names, case))


@pytest.fixture(scope="module")
def rotary_embs():
    """Identity RoPE table: cos=1, sin=0."""
    table = torch.zeros(
        2, MAX_SEQ_LEN, HEAD_DIM // 2, dtype=torch.float32, device="cuda"
    )
    table[0] = 1
    return table


def _get_tolerances(seq_len):
    """Return ``(atol, mean_max)`` for the correctness check.

    ``atol`` bounds the element-wise absolute error and ``mean_max`` bounds the
    mean absolute error.
    """
    atol = 0.1 if seq_len >= 32768 else 0.05
    mean_max = 0.01 if seq_len >= 32768 or seq_len <= 128 else 0.005
    return atol, mean_max


def _prepare_decode(
    rotary_embs, batch_size, head_num, kv_head_num, seq_len_encoder, mtp_step=0
):
    """Build the paged KV cache and folded decode query for one config.

    Runs the encoder (prefill) and decoder KV-cache writes, and returns
    everything needed to invoke ``decoder_attention`` plus the raw fused QKV
    inputs used to build the numpy reference.
    """
    num_q_per_batch = 1 + mtp_step
    total_kv_per_batch = seq_len_encoder + num_q_per_batch

    num_phys_blocks = batch_size * MAX_SEQ_LEN // BLOCK_SIZE
    cache_k = torch.zeros(
        num_phys_blocks, kv_head_num, BLOCK_SIZE, HEAD_DIM, dtype=DTYPE, device="cuda"
    )
    cache_v = torch.zeros_like(cache_k)

    blocks_per_seq = MAX_SEQ_LEN // BLOCK_SIZE
    block_tables = torch.arange(
        batch_size * blocks_per_seq, dtype=torch.int32, device="cuda"
    ).reshape(batch_size, blocks_per_seq)

    qkv_encoder_input = (
        torch.randn(
            seq_len_encoder * batch_size,
            head_num + 2 * kv_head_num,
            HEAD_DIM,
            device="cuda",
        )
        * 0.5
    ).to(DTYPE)
    qkv_decoder_input = (
        torch.randn(
            batch_size * num_q_per_batch,
            head_num + 2 * kv_head_num,
            HEAD_DIM,
            device="cuda",
        )
        * 0.5
    ).to(DTYPE)

    # Encoder/prefill stage: write Q/K/V and populate the KV cache.
    cu_seqlens_enc = (torch.arange(batch_size + 1, device="cuda") * seq_len_encoder).to(
        torch.int32
    )
    seq_len_enc_t = torch.full(
        (batch_size,), seq_len_encoder, dtype=torch.int32, device="cuda"
    )
    seq_len_dec_t = torch.zeros(batch_size, dtype=torch.int32, device="cuda")

    q_input_enc = torch.zeros(
        seq_len_encoder * batch_size, head_num, HEAD_DIM, dtype=DTYPE, device="cuda"
    )
    k_input_enc = torch.zeros(
        seq_len_encoder * batch_size, kv_head_num, HEAD_DIM, dtype=DTYPE, device="cuda"
    )
    v_input_enc = torch.zeros_like(k_input_enc)

    write_cache_kv_encoder(
        qkv_encoder_input,
        cu_seqlens_enc,
        cu_seqlens_enc,
        rotary_embs,
        seq_len_enc_t,
        seq_len_dec_t,
        cache_k,
        cache_v,
        block_tables,
        q_input_enc,
        k_input_enc,
        v_input_enc,
        None,
        None,
        None,
        int(head_num),
        int(kv_head_num),
        int(HEAD_DIM),
        int(seq_len_encoder),
        int(seq_len_encoder),
        int(MAX_SEQ_LEN),
        "none",
    )

    # Decode stage: append current token K/V and fold Q into the
    # [batch, kv_head_num, 64, head_dim] layout decode attention expects.
    cu_seqlens_q_dec = (
        torch.arange(batch_size + 1, device="cuda") * num_q_per_batch
    ).to(torch.int32)
    seq_len_enc_t2 = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
    seq_len_dec_t2 = torch.full(
        (batch_size,), seq_len_encoder, dtype=torch.int32, device="cuda"
    )

    q_output, _ = write_cache_kv_decoder(
        qkv_decoder_input,
        cu_seqlens_q_dec,
        seq_len_enc_t2,
        seq_len_dec_t2,
        rotary_embs,
        cache_k,
        cache_v,
        block_tables,
        None,
        None,
        None,
        int(head_num),
        int(kv_head_num),
        int(HEAD_DIM),
        int(MAX_SEQ_LEN),
        "none",
    )

    seq_lens_kv = torch.full(
        (batch_size,), total_kv_per_batch, dtype=torch.int32, device="cuda"
    )

    return {
        "q_output": q_output,
        "cache_k": cache_k,
        "cache_v": cache_v,
        "block_tables": block_tables,
        "cu_seqlens_q_dec": cu_seqlens_q_dec,
        "seq_lens_kv": seq_lens_kv,
        "seq_len_enc_t2": seq_len_enc_t2,
        "seq_len_dec_t2": seq_len_dec_t2,
        "qkv_encoder_input": qkv_encoder_input,
        "qkv_decoder_input": qkv_decoder_input,
        "num_q_per_batch": num_q_per_batch,
        "total_kv_per_batch": total_kv_per_batch,
    }


def _decode_attention(prep, batch_size, head_num, kv_head_num, mtp_step):
    """Run decode attention for a prepared config into a fresh output buffer."""
    num_q_per_batch = prep["num_q_per_batch"]
    attn_out = torch.zeros(
        batch_size * num_q_per_batch, head_num * HEAD_DIM, dtype=DTYPE, device="cuda"
    )
    decoder_attention(
        prep["q_output"],
        prep["cu_seqlens_q_dec"],
        prep["seq_lens_kv"],
        prep["seq_len_enc_t2"],
        prep["seq_len_dec_t2"],
        prep["cache_k"],
        prep["cache_v"],
        prep["block_tables"],
        attn_out,
        None,
        None,
        int(head_num),
        int(kv_head_num),
        int(HEAD_DIM),
        int(prep["total_kv_per_batch"]),
        int(mtp_step),
        "none",
    )
    return attn_out


def _to_np64(x):
    return x.float().cpu().numpy().astype(np.float64)


def _run_case(
    rotary_embs, batch_size, head_num, kv_head_num, seq_len_encoder, mtp_step
):
    """Run one decode config; return ``(kernel_out, ref_out)`` float64 tensors."""
    prep = _prepare_decode(
        rotary_embs, batch_size, head_num, kv_head_num, seq_len_encoder, mtp_step
    )
    attn_out = _decode_attention(prep, batch_size, head_num, kv_head_num, mtp_step)

    qkv_encoder_input = prep["qkv_encoder_input"]
    qkv_decoder_input = prep["qkv_decoder_input"]
    num_q_per_batch = prep["num_q_per_batch"]
    k_heads = slice(head_num, head_num + kv_head_num)
    v_heads = slice(head_num + kv_head_num, None)

    # Reference (per-batch)
    all_ref = []
    for b in range(batch_size):
        enc = slice(b * seq_len_encoder, (b + 1) * seq_len_encoder)
        dec = slice(b * num_q_per_batch, (b + 1) * num_q_per_batch)

        q_np = _to_np64(qkv_decoder_input[dec, :head_num, :])
        k_full = np.concatenate(
            [
                _to_np64(qkv_encoder_input[enc, k_heads, :]),
                _to_np64(qkv_decoder_input[dec, k_heads, :]),
            ],
            axis=0,
        )
        v_full = np.concatenate(
            [
                _to_np64(qkv_encoder_input[enc, v_heads, :]),
                _to_np64(qkv_decoder_input[dec, v_heads, :]),
            ],
            axis=0,
        )
        all_ref.append(reference_attention(q_np, k_full, v_full, head_num, kv_head_num))
    ref_out = torch.from_numpy(np.concatenate(all_ref, axis=0))

    kernel_out = torch.from_numpy(
        _to_np64(attn_out.reshape(batch_size * num_q_per_batch, head_num, HEAD_DIM))
    )
    return kernel_out, ref_out


@pytest.mark.parametrize(
    "batch_size,seq_len,head_num,kv_head_num,mtp_step",
    CORRECTNESS_CASES,
    ids=[_case_id(c) for c in CORRECTNESS_CASES],
)
def test_decoder_attention_correctness(
    rotary_embs, batch_size, seq_len, head_num, kv_head_num, mtp_step
):
    atol, mean_max = _get_tolerances(seq_len)
    kernel_out, ref_out = _run_case(
        rotary_embs, batch_size, head_num, kv_head_num, seq_len, mtp_step
    )

    torch.testing.assert_close(kernel_out, ref_out, rtol=0.0, atol=atol)
    mean_diff = (kernel_out - ref_out).abs().mean().item()
    assert mean_diff < mean_max, f"mean_diff={mean_diff:.6f} exceeds {mean_max}"


@pytest.mark.parametrize(
    "batch_size,seq_len_encoder,head_num,kv_head_num",
    DETERMINISM_CASES,
    ids=[_case_id(c) for c in DETERMINISM_CASES],
)
def test_decoder_attention_determinism(
    rotary_embs, batch_size, seq_len_encoder, head_num, kv_head_num
):
    """Two identical decode calls must produce bit-identical output."""
    mtp_step = 0
    prep = _prepare_decode(
        rotary_embs, batch_size, head_num, kv_head_num, seq_len_encoder, mtp_step
    )
    out1 = _decode_attention(prep, batch_size, head_num, kv_head_num, mtp_step)
    out2 = _decode_attention(prep, batch_size, head_num, kv_head_num, mtp_step)
    assert torch.equal(out1, out2), "decoder_attention output is not deterministic"


if __name__ == "__main__":
    pytest.main([__file__])
