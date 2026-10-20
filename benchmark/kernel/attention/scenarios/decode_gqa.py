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

from __future__ import annotations
import gc
import math
from dataclasses import dataclass

import torch

from core.runner import run_bench_for_backend, run_refcheck_cross_backends
from core.metrics import compute_metrics
from core.refcheck import run_refcheck
from kernel.quantization_util import tensor_wise_quantize
from core.output import (
    AttnRowResult,
    print_scenario_header,
    print_table_header,
    print_table_row,
    append_attention_csv,
)
from kernel.attention.backends.halops import (
    build_cache_kv,
    fold_dense_q_to_halops,
    halops_v_reorder_indices,
    read_cache_kv,
    recover_dense_q,
    unfold_halops_q,
    write_cache_kv,
)

# Constants matching the original bench_decode_c16.py / halops decode kernel.
BLOCK_SIZE = 128
KBLOCKM = 64


def _is_fp8_dtype(dtype: str) -> bool:
    return dtype in ("float8_e4m3fn", "e4m3", "fp8")


@dataclass
class AttnMeta:
    """Shape and execution metadata shared by decode attention backends."""

    batch_size: int
    head_num: int
    kv_head_num: int
    head_dim: int
    s_q: int
    s_kv: int
    total_kv: int
    mtp_steps: int
    softmax_scale: float
    causal: bool
    is_fp8: bool


@dataclass
class DecodeAttnInput:
    """Unified inputs for decode GQA attention benchmarks across all backends.

    Holds both the HALOps paged-attention layout (q_bw, block_tables, paged cache)
    and the dense layout (q_orig_bf16, k_orig_bf16, v_orig_bf16) used by FA2/FA4/cuDNN backends.
    FP8 quantization fields are None when running in BF16 mode.
    """

    # halops-specific
    q_bw: torch.Tensor  # [B, H_KV, KBLOCKM, D]
    cache_k: torch.Tensor  # [num_phys_blocks, H_KV, BLOCK_SIZE, D]
    cache_v: torch.Tensor  # [num_phys_blocks, H_KV, BLOCK_SIZE, D]
    block_tables: torch.Tensor  # [B, max_blocks]
    cu_seqlens_q: torch.Tensor  # [B+1]
    seq_lens_kv: torch.Tensor  # [B]
    seq_len_encoder: torch.Tensor  # [B]
    seq_len_decoder: torch.Tensor  # [B]
    attn_out_bw: torch.Tensor  # [B * s_q, H_Q * D]
    # Shared by other backends
    q_orig_bf16: torch.Tensor  # [B, s_q, H_Q, D]
    k_orig_bf16: torch.Tensor  # [B, s_kv, H_KV, D]
    v_orig_bf16: torch.Tensor  # [B, s_kv, H_KV, D]
    # FP8 quantization (None for BF16 path)
    q_dequant_scale: torch.Tensor | None  # [B, H_KV]
    kv_dequant_scale: torch.Tensor | None  # [num_phys_blocks, H_KV, 4]
    meta: AttnMeta
    use_write_cache: bool
    quant_mode: str  # "none" or "block_wise_fp8"


def _alloc_cache_and_block_tables(
    meta: AttnMeta,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    max_blk = meta.total_kv // BLOCK_SIZE + 1
    num_phys_blocks = meta.batch_size * max_blk
    block_tables = (
        torch.arange(num_phys_blocks, device="cuda")
        .reshape(meta.batch_size, max_blk)
        .to(torch.int32)
    )

    cache_dtype = torch.uint8 if meta.is_fp8 else torch.bfloat16
    cache_k = torch.zeros(
        num_phys_blocks,
        meta.kv_head_num,
        BLOCK_SIZE,
        meta.head_dim,
        dtype=cache_dtype,
        device="cuda",
    )
    cache_v = torch.zeros_like(cache_k)

    if meta.is_fp8:
        kv_dequant_scale = torch.zeros(
            num_phys_blocks, meta.kv_head_num, 4, device="cuda"
        )
        return cache_k, cache_v, block_tables, kv_dequant_scale

    return cache_k, cache_v, block_tables, None


def _fp8_q_scale_for_fi(inp: DecodeAttnInput) -> float:
    scale_softmax_log2 = math.log2(math.e) / math.sqrt(inp.meta.head_dim)
    return float((inp.q_dequant_scale[0, 0] / scale_softmax_log2).item())


def _build_shared_inputs(
    meta: AttnMeta,
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    block_tables: torch.Tensor,
    kv_dequant_scale: torch.Tensor | None,
    input_scale: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
]:
    B, H_Q, H_KV, D = (
        meta.batch_size,
        meta.head_num,
        meta.kv_head_num,
        meta.head_dim,
    )
    s_q = meta.s_q
    total_kv = meta.total_kv
    q_src = (
        torch.randn(B, s_q, H_Q, D, dtype=torch.bfloat16, device="cuda") * input_scale
    )
    k_src = (
        torch.randn(B, total_kv, H_KV, D, dtype=torch.bfloat16, device="cuda")
        * input_scale
    )
    v_src = (
        torch.randn(B, total_kv, H_KV, D, dtype=torch.bfloat16, device="cuda")
        * input_scale
    )

    if not meta.is_fp8:
        q_bw = fold_dense_q_to_halops(q_src, H_KV, KBLOCKM)
        build_cache_kv(
            cache_k,
            cache_v,
            block_tables,
            k_src,
            v_src,
            total_kv,
            is_fp8=False,
        )
        return (
            q_bw,
            None,
            q_src.contiguous(),
            k_src.contiguous(),
            v_src.contiguous(),
            total_kv,
        )

    q_fp8, q_scale = tensor_wise_quantize(q_src)
    k_fp8, k_scale = tensor_wise_quantize(k_src)
    v_fp8, v_scale = tensor_wise_quantize(v_src)

    q_bw = fold_dense_q_to_halops(q_fp8, H_KV, KBLOCKM)
    scale_softmax_log2 = math.log2(math.e) / math.sqrt(D)
    q_dequant_scale = torch.full(
        (B, H_KV),
        q_scale * scale_softmax_log2,
        dtype=torch.float32,
        device="cuda",
    )

    build_cache_kv(
        cache_k,
        cache_v,
        block_tables,
        k_fp8,
        v_fp8,
        total_kv,
        is_fp8=True,
        kv_dequant_scale=kv_dequant_scale,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    q_orig_bf16 = recover_dense_q(q_bw, B, s_q, H_Q, H_KV, D, q_dequant_scale)
    k_orig_bf16 = (k_fp8.float() * k_scale).to(torch.bfloat16).contiguous()
    v_orig_bf16 = (v_fp8.float() * v_scale).to(torch.bfloat16).contiguous()
    return q_bw, q_dequant_scale, q_orig_bf16, k_orig_bf16, v_orig_bf16, total_kv


def _build_write_cache_q(
    meta: AttnMeta,
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    block_tables: torch.Tensor,
    kv_dequant_scale: torch.Tensor | None,
    input_scale: float,
    quant_mode: str,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, int]:
    q_bw, q_dequant_scale = write_cache_kv(
        meta.batch_size,
        meta.s_kv,
        meta.s_q,
        meta.head_num,
        meta.kv_head_num,
        meta.head_dim,
        cache_k,
        cache_v,
        block_tables,
        kv_dequant_scale,
        input_scale,
        quant_mode,
    )
    q_orig_bf16 = recover_dense_q(
        q_bw,
        meta.batch_size,
        meta.s_q,
        meta.head_num,
        meta.kv_head_num,
        meta.head_dim,
        q_dequant_scale,
    )
    return q_bw, q_dequant_scale, q_orig_bf16, meta.total_kv


def _make_decode_metadata(
    meta: AttnMeta,
    dense_seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    attn_out = torch.zeros(
        meta.batch_size * meta.s_q,
        meta.head_num * meta.head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    cu_seq_q = (
        torch.arange(meta.batch_size + 1, dtype=torch.int32, device="cuda") * meta.s_q
    )
    seq_lens_kv = torch.full(
        (meta.batch_size,), dense_seq_len, dtype=torch.int32, device="cuda"
    )
    seq_len_encoder = torch.zeros(meta.batch_size, dtype=torch.int32, device="cuda")
    seq_len_decoder = torch.full(
        (meta.batch_size,), meta.s_kv, dtype=torch.int32, device="cuda"
    )
    return attn_out, cu_seq_q, seq_lens_kv, seq_len_encoder, seq_len_decoder


def build_inputs(cfg: dict) -> DecodeAttnInput:
    """Build DecodeAttnInput from a scenario config dict.

    Allocates paged KV cache, block_tables, and query tensors in both HALOps
    folded layout and standard dense layout. use_write_cache=True exercises the
    production write_cache kernels; use_write_cache=False builds shared BF16/FP8
    inputs directly so backends can consume aligned data.
    """
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    B = cfg["batch_size"]
    s_kv = cfg["s_kv"]
    H_Q = cfg["num_q_heads"]
    H_KV = cfg["num_kv_heads"]
    D = cfg["head_dim"]
    input_scale = cfg.get("input_scale", 2.0)
    mtp_steps = cfg.get("mtp_steps", 0)
    use_write_cache = cfg.get("use_write_cache", False)

    s_q = 1 + mtp_steps
    is_fp8 = _is_fp8_dtype(cfg["dtype"])
    quant_mode = cfg.get("cache_quant_type", "block_wise_fp8" if is_fp8 else "none")

    assert H_Q % H_KV == 0, f"H_Q ({H_Q}) must be divisible by H_KV ({H_KV})"
    gqa = H_Q // H_KV
    assert (
        s_q * gqa <= KBLOCKM
    ), f"(mtp_steps+1)*gqa={s_q * gqa} exceeds KBLOCKM={KBLOCKM}"
    total_kv = s_kv + s_q

    meta = AttnMeta(
        batch_size=B,
        head_num=H_Q,
        kv_head_num=H_KV,
        head_dim=D,
        s_q=s_q,
        s_kv=s_kv,
        total_kv=total_kv,
        mtp_steps=mtp_steps,
        softmax_scale=1.0 / math.sqrt(D),
        causal=True,
        is_fp8=is_fp8,
    )

    cache_k, cache_v, block_tables, kv_dequant_scale = _alloc_cache_and_block_tables(
        meta
    )

    if use_write_cache:
        q_bw, q_dequant_scale, q_orig_bf16, dense_seq_len = _build_write_cache_q(
            meta,
            cache_k,
            cache_v,
            block_tables,
            kv_dequant_scale,
            input_scale,
            quant_mode,
        )
        k_orig_bf16, v_orig_bf16 = read_cache_kv(
            cache_k,
            cache_v,
            block_tables,
            B,
            H_KV,
            D,
            dense_seq_len,
            is_fp8,
            kv_dequant_scale=kv_dequant_scale,
        )
    else:
        (
            q_bw,
            q_dequant_scale,
            q_orig_bf16,
            k_orig_bf16,
            v_orig_bf16,
            dense_seq_len,
        ) = _build_shared_inputs(
            meta,
            cache_k,
            cache_v,
            block_tables,
            kv_dequant_scale,
            input_scale,
        )

    attn_out, cu_seq_q, seq_lens_kv, seq_len_encoder, seq_len_decoder = (
        _make_decode_metadata(meta, dense_seq_len)
    )

    return DecodeAttnInput(
        q_bw=q_bw,
        cache_k=cache_k,
        cache_v=cache_v,
        block_tables=block_tables,
        cu_seqlens_q=cu_seq_q,
        seq_lens_kv=seq_lens_kv,
        seq_len_encoder=seq_len_encoder,
        seq_len_decoder=seq_len_decoder,
        attn_out_bw=attn_out,
        q_orig_bf16=q_orig_bf16,
        k_orig_bf16=k_orig_bf16,
        v_orig_bf16=v_orig_bf16,
        q_dequant_scale=q_dequant_scale,
        kv_dequant_scale=kv_dequant_scale,
        meta=meta,
        use_write_cache=use_write_cache,
        quant_mode=quant_mode,
    )


def adapt_halops(inp: DecodeAttnInput) -> dict:
    """Adapt decode input for HALOps."""
    return {
        "q": inp.q_bw,
        "cu_seq_q": inp.cu_seqlens_q,
        "seq_lens_kv": inp.seq_lens_kv,
        "seq_len_encoder": inp.seq_len_encoder,
        "seq_len_decoder": inp.seq_len_decoder,
        "cache_k": inp.cache_k,
        "cache_v": inp.cache_v,
        "block_tables": inp.block_tables,
        "attn_out": inp.attn_out_bw,
        "q_dequant_scale": inp.q_dequant_scale,
        "kv_dequant_scale": inp.kv_dequant_scale,
        "head_num": inp.meta.head_num,
        "kv_head_num": inp.meta.kv_head_num,
        "head_dim": inp.meta.head_dim,
        "max_seq_k": int(inp.seq_lens_kv.max().item()),
        "mtp_steps": inp.meta.mtp_steps,
        "quant_mode": inp.quant_mode,
    }


def adapt_cudnn(inp: DecodeAttnInput) -> dict:
    """Adapt DecodeAttnInput to cuDNN format (delegates to FA layout: [B, S, H, D])."""
    return adapt_fa(inp)


def _adapt_dense(inp: DecodeAttnInput) -> dict:
    """Build the dense [B, S, H, D] attention inputs shared by FA/cuDNN/golden.

    Uses dense BF16 Q/K/V tensors. For FP8 cache configs these tensors are the
    dequantized view of the quantized Q/K/V, not the original pre-quantized data.

    Disables causal masking when s_q < s_kv (decode case), since the query is at
    the end of the sequence and should attend to all prior KV tokens.
    """
    s_q = inp.meta.s_q
    # For decode (s_q << s_kv), the query is logically at the END of the sequence
    # and can attend to all prior KV tokens. is_causal=True with mismatched q/k lengths
    # would wrongly mask based on tensor positions, not absolute sequence positions.
    causal = False if s_q < inp.meta.s_kv else inp.meta.causal
    return {
        "q": inp.q_orig_bf16.reshape(
            inp.meta.batch_size, s_q, inp.meta.head_num, inp.meta.head_dim
        ),
        "k": inp.k_orig_bf16,
        "v": inp.v_orig_bf16,
        "softmax_scale": inp.meta.softmax_scale,
        "causal": causal,
        "num_splits": 0,
    }


def adapt_golden(inp: DecodeAttnInput) -> dict:
    """Adapt for the golden baseline (torch SDPA math).

    Same dense layout as FA, but without the FP8 guard: golden consumes the
    dequantized bf16 K/V history, so it serves as the high-precision reference
    for both BF16 and FP8 KV cache runs.
    """
    return _adapt_dense(inp)


def adapt_fa(inp: DecodeAttnInput) -> dict:
    """Adapt DecodeAttnInput to FA2/FA4 dense format [B, S, H, D]."""
    # fa2/fa4 expects [B, S, H, D]. s_q = 1 + mtp_steps.
    if inp.meta.is_fp8:
        raise RuntimeError("fa does not support FP8 KV cache")
    return _adapt_dense(inp)


def adapt_fa4_paged(inp: DecodeAttnInput) -> dict:
    """FA4 paged decode via flash_attn_varlen_func + page_table.

    Pages the dense (reorder-undone) K/V history over total_kv (=s_kv+s_q) via
    _cache_to_fi_paged_nhd, exactly like the fi backends, so fa4_paged sees the
    same logical K/V as golden/cudnn/fi. Feeding the raw paged cache_v directly
    would be wrong: the write_cache kernel stores V with an intra-block row
    reorder (undone in read_cache_kv), so an unshuffled cache_v scrambles the
    attention output. Mirrors golden's causal logic (causal disabled for the
    decode case s_q < s_kv, where the query attends to all KV positions).
    """
    if inp.meta.is_fp8:
        raise RuntimeError("fa does not support FP8 KV cache")

    B, H_Q, D = inp.meta.batch_size, inp.meta.head_num, inp.meta.head_dim
    s_q = inp.meta.s_q
    total_kv = inp.meta.total_kv
    page_size = BLOCK_SIZE
    pages_per_seq = math.ceil(total_kv / page_size)

    # [B*pages_per_seq, page_size, H_KV, D] logical pages + flat page indices.
    k_pages, v_pages, kv_indices, _, _ = _cache_to_fi_paged_nhd(inp, page_size)

    q = inp.q_orig_bf16.reshape(B * s_q, H_Q, D)
    page_table = kv_indices.reshape(B, pages_per_seq).to(torch.int32).contiguous()

    cu_seqlens_q = torch.arange(B + 1, dtype=torch.int32, device="cuda") * s_q
    seqused_k = torch.full((B,), total_kv, dtype=torch.int32, device="cuda")

    causal = False if s_q < inp.meta.s_kv else inp.meta.causal

    return {
        "q": q,
        "k_cache": k_pages,
        "v_cache": v_pages,
        "cu_seqlens_q": cu_seqlens_q,
        "max_seqlen_q": s_q,
        "max_seqlen_k": total_kv,
        "seqused_k": seqused_k,
        "page_table": page_table,
        "softmax_scale": inp.meta.softmax_scale,
        "causal": causal,
        "num_splits": 0,
    }


def _cache_to_fi_paged_nhd(
    inp: DecodeAttnInput,
    page_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float | None, float | None]:
    """Build FlashInfer NHD pages and their per-seq page indices.

    Returns (k_pages, v_pages, kv_indices, k_scale, v_scale) where
    kv_indices[b*pages_per_seq + i] is the physical page holding page i of
    sequence b. k_scale/v_scale are the FP8 dequant scales (None for bf16).

    BF16 pages the dense (reorder-undone) K/V history. FP8 shared-input mode
    reuses the exact HALOps cache bytes and scalar scales produced by
    build_inputs, only transforming layout to FlashInfer NHD.
    """
    B, H_KV, D = inp.meta.batch_size, inp.meta.kv_head_num, inp.meta.head_dim
    total_kv = inp.meta.total_kv
    pages_per_seq = math.ceil(total_kv / page_size)

    if inp.meta.is_fp8 and not inp.use_write_cache:
        if page_size != BLOCK_SIZE:
            raise ValueError("shared FP8 cache path requires page_size == BLOCK_SIZE")
        page_ids = inp.block_tables[:, :pages_per_seq].reshape(-1).long()
        k_pages = (
            inp.cache_k.index_select(0, page_ids)
            .contiguous()
            .view(torch.float8_e4m3fn)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        rows = halops_v_reorder_indices(BLOCK_SIZE, inp.cache_v.device, is_fp8=True)
        v_pages = (
            inp.cache_v.index_select(0, page_ids)[:, :, rows, :]
            .contiguous()
            .view(torch.float8_e4m3fn)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        scales = inp.kv_dequant_scale.index_select(0, page_ids)
        k_scale = float(scales[0, 0, 0].item())
        v_scale = float(scales[0, 0, 1].item())
        kv_indices = torch.arange(B * pages_per_seq, dtype=torch.int32, device="cuda")
        return k_pages, v_pages, kv_indices, k_scale, v_scale

    k_dense = inp.k_orig_bf16[:, :total_kv]
    v_dense = inp.v_orig_bf16[:, :total_kv]
    if inp.meta.is_fp8:
        k_dense, k_scale = tensor_wise_quantize(k_dense)
        v_dense, v_scale = tensor_wise_quantize(v_dense)
    else:
        k_scale, v_scale = None, None

    def _to_pages(t: torch.Tensor) -> torch.Tensor:
        # Pad to a whole number of pages based on the tensor's actual length.
        pad_len = pages_per_seq * page_size - t.shape[1]
        if pad_len > 0:
            t = torch.nn.functional.pad(t, (0, 0, 0, 0, 0, pad_len))
        # [B, total_kv_padded, H_KV, D] -> [B*pages_per_seq, page_size, H_KV, D]
        return t.reshape(B, pages_per_seq, page_size, H_KV, D).reshape(
            -1, page_size, H_KV, D
        )

    k_pages = _to_pages(k_dense)
    v_pages = _to_pages(v_dense)
    kv_indices = torch.arange(B * pages_per_seq, dtype=torch.int32, device="cuda")
    return k_pages, v_pages, kv_indices, k_scale, v_scale


def adapt_fi(
    inp: DecodeAttnInput, page_size: int = BLOCK_SIZE, backend: str = "auto"
) -> dict:
    """FlashInfer paged decode: adapt phase (create wrapper + plan).

    Repages dense K/V for BF16. In FP8 mode, the trtllm-gen backend uses
    FP8 Q/K/V (A8C8); other FlashInfer backends keep Q in BF16 (A16C8).
    """
    from kernel.attention.backends.flashinfer import plan_fi_decode_paged

    B, H_KV, D = inp.meta.batch_size, inp.meta.kv_head_num, inp.meta.head_dim
    s_q = inp.meta.s_q
    total_kv = inp.meta.total_kv

    # _cache_to_fi_paged_nhd repages the dense (reorder-undone) history at any
    # page_size, so no BLOCK_SIZE-specific splitting is needed.
    pages_per_seq = math.ceil(total_kv / page_size)
    k_pages, v_pages, kv_indices, k_scale, v_scale = _cache_to_fi_paged_nhd(
        inp, page_size
    )
    kv_data = torch.stack([k_pages, v_pages], dim=1)

    kv_indptr = torch.arange(B + 1, dtype=torch.int32, device="cuda") * pages_per_seq
    last_len = total_kv - (pages_per_seq - 1) * page_size
    kv_last_len = torch.full((B,), last_len, dtype=torch.int32, device="cuda")

    q_scale = None
    if inp.meta.is_fp8 and backend == "trtllm-gen":
        q = unfold_halops_q(
            inp.q_bw, s_q, inp.meta.head_num, inp.meta.kv_head_num, flatten=True
        )
        q_scale = _fp8_q_scale_for_fi(inp)
    else:
        q = inp.q_orig_bf16.reshape(B * s_q, inp.meta.head_num, inp.meta.head_dim)

    return plan_fi_decode_paged(
        q,
        kv_data,
        kv_indptr,
        kv_indices,
        kv_last_len,
        inp.meta.head_num,
        H_KV,
        D,
        page_size,
        backend=backend,
        q_len_per_req=s_q,
        kv_data_type=torch.float8_e4m3fn if inp.meta.is_fp8 else None,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
    )


def adapt_fi_trtllm(inp: DecodeAttnInput, page_size: int = BLOCK_SIZE) -> dict:
    """Adapt inputs for trtllm_batch_decode_with_kv_cache (NHD layout)."""
    B = inp.meta.batch_size
    s_q = inp.meta.s_q
    total_kv = inp.meta.total_kv

    # Page over total_kv (=s_kv+s_q) to match _cache_to_fi_paged_nhd: the decode
    # query attends over all s_kv history plus its own s_q tokens.
    pages_per_seq = math.ceil(total_kv / page_size)

    # [B*pages_per_seq, page_size, H_KV, D] (NHD) plus matching page indices.
    # k_scale/v_scale are the per-tensor FP8 dequant scales of the K/V cache.
    k_pages, v_pages, kv_indices, k_scale, v_scale = _cache_to_fi_paged_nhd(
        inp, page_size
    )
    # [num_pages, 2, page_size, H_KV, D]
    kv_cache = torch.stack([k_pages, v_pages], dim=1)

    # trtllm wants block_tables as [B, pages_per_seq]; kv_indices is its flat form.
    block_tables = kv_indices.reshape(B, pages_per_seq).contiguous()
    seq_lens = torch.full((B,), total_kv, dtype=torch.int32, device="cuda")

    # trtllm fuses dequant into the bmm scales:
    #   bmm1 = softmax_scale * q_dequant_scale * k_dequant_scale  (post-QK^T)
    #   bmm2 = v_dequant_scale                                    (post-PV)
    # For bf16 the dequant scales are 1, so only softmax_scale remains.
    bmm1_scale = inp.meta.softmax_scale
    bmm2_scale = 1.0
    if inp.meta.is_fp8:
        q = unfold_halops_q(
            inp.q_bw, s_q, inp.meta.head_num, inp.meta.kv_head_num, flatten=True
        )
        q_scale = _fp8_q_scale_for_fi(inp)
        bmm1_scale = inp.meta.softmax_scale * q_scale * k_scale
        bmm2_scale = v_scale
    else:
        q = inp.q_orig_bf16.reshape(B * s_q, inp.meta.head_num, inp.meta.head_dim)

    workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")

    # Pre-zeroed output buffer (see run_fi_decode_paged rationale): with
    # q_len_per_req>1 the kernel may leave output lanes unwritten; a pre-zeroed,
    # reused buffer keeps them at 0 and gives run() a fixed address.
    out = torch.zeros(q.shape, dtype=torch.bfloat16, device=q.device)

    return {
        "q": q,
        "kv_cache": kv_cache,
        "block_tables": block_tables,
        "seq_lens": seq_lens,
        "max_seq_len": total_kv,
        "workspace": workspace,
        "out": out,
        "bmm1_scale": bmm1_scale,
        "bmm2_scale": bmm2_scale,
        "q_len_per_req": s_q,
    }


def _refcheck_entry(
    cfg, ordered_names, refcheck_reference, refcheck_atol=1e-2, refcheck_rtol=1e-2
):
    """Subprocess entry for refcheck."""
    inp = build_inputs(cfg)
    backends = _get_backends(ordered_names)

    # Put refcheck_reference first so it becomes the comparison baseline
    refcheck_order = [refcheck_reference] + [
        n for n in ordered_names if n != refcheck_reference
    ]
    adapt_fns = {n: backends[n][0] for n in refcheck_order if n in backends}
    run_fns = {n: backends[n][1] for n in refcheck_order if n in backends}
    inp_dicts = {n: adapt_fns[n](inp) for n in adapt_fns}
    return run_refcheck(
        [n for n in refcheck_order if n in backends],
        run_fns,
        inp_dicts,
        atol=refcheck_atol,
        rtol=refcheck_rtol,
    )


def _bench_entry(
    cfg,
    name,
    dry_run_iters,
    repeat_iters,
    cold_l2_cache,
    use_cuda_graph,
    timing_method,
):
    """Subprocess entry: build inputs, adapt, and run benchmark for a single backend."""
    local_cfg = dict(cfg)
    inp = build_inputs(local_cfg)
    backends = _get_backends([name])
    if name not in backends:
        raise RuntimeError(f"backend '{name}' not found")

    adapt_fn, run_fn = backends[name]
    inp_dict = adapt_fn(inp)
    from core.timing import bench_gpu_time

    return bench_gpu_time(
        run_fn,
        (inp_dict,),
        dry_run_iters=dry_run_iters,
        repeat_iters=repeat_iters,
        cold_l2_cache=cold_l2_cache,
        use_cuda_graph=use_cuda_graph,
        method=timing_method,
    )


def _get_backends(backend_names: list[str]) -> dict[str, tuple]:
    from kernel.attention.backends.halops import run_halops_decode
    from kernel.attention.backends.flash_attention import (
        run_fa2_decode,
        run_fa4_decode,
        run_fa4_paged_decode,
    )
    from kernel.attention.backends.cudnn import run_cudnn
    from kernel.attention.backends.golden import run_golden
    from kernel.attention.backends.flashinfer import (
        run_fi_decode_paged,
        run_fi_trtllm_decode,
    )

    registry = {
        # Neutral correctness baseline (torch SDPA math). Uses the dense KV
        # layout (same as fa/cudnn) but via adapt_golden, which skips the FP8
        # guard so it can run on dequantized bf16 K/V for FP8 cache configs too.
        "golden": (adapt_golden, run_golden),
        "halops": (adapt_halops, run_halops_decode),
        "fa2": (adapt_fa, run_fa2_decode),
        "fa4": (adapt_fa, run_fa4_decode),
        "fa4_paged": (adapt_fa4_paged, run_fa4_paged_decode),
        "fi": (lambda inp: adapt_fi(inp, backend="auto"), run_fi_decode_paged),
        "fi(auto)": (lambda inp: adapt_fi(inp, backend="auto"), run_fi_decode_paged),
        "fi(fa2)": (lambda inp: adapt_fi(inp, backend="fa2"), run_fi_decode_paged),
        "fi(fa3)": (lambda inp: adapt_fi(inp, backend="fa3"), run_fi_decode_paged),
        "fi(trtllm-gen)": (
            lambda inp: adapt_fi(inp, backend="trtllm-gen"),
            run_fi_decode_paged,
        ),
        "fi_trtllm": (adapt_fi_trtllm, run_fi_trtllm_decode),
        "cudnn": (adapt_cudnn, run_cudnn),
    }

    return {name: registry[name] for name in backend_names if name in registry}


def run_scenario(cfg: dict) -> None:
    backend_names = cfg["backends"]
    dry_run_iters = cfg.get("dry_run_iters", 10)
    repeat_iters = cfg.get("repeat_iters", 50)
    cold_l2_cache = cfg.get("cold_l2_cache", True)
    use_cuda_graph = cfg.get("use_cuda_graph", False)
    timing_method = cfg.get("timing_method", "profiler")
    do_refcheck = cfg.get("refcheck", True)
    refcheck_reference = cfg.get("refcheck_reference", "golden")
    output_path = cfg.get("output_path", "results/decode_gqa.csv")
    case_tag = cfg.get("case_tag", "")
    use_subprocess = cfg.get("use_subprocess", True)
    timeout = cfg.get("timeout", 120)
    is_fp8 = _is_fp8_dtype(cfg.get("dtype", ""))

    def _move_to_front(names: list[str], target: str) -> list[str]:
        if target in names:
            return [target] + [n for n in names if n != target]
        return list(names)

    # benchmark / display order: halops first; it is the speedup baseline.
    speedup_baseline = "halops"
    ordered_names = _move_to_front(backend_names, speedup_baseline)

    for s_kv in cfg["s_kv_list"]:
        for mtp_steps in cfg.get("mtp_steps_list", [0]):
            s_q = 1 + mtp_steps
            local_cfg = {**cfg, "s_kv": s_kv, "mtp_steps": mtp_steps}

            refcheck_results: dict = {n: None for n in backend_names}
            if do_refcheck:
                refcheck_results = run_refcheck_cross_backends(
                    _refcheck_entry,
                    local_cfg,
                    ordered_names,
                    refcheck_reference,
                    refcheck_atol=cfg.get("refcheck_atol", 1e-2),
                    refcheck_rtol=cfg.get("refcheck_rtol", 1e-2),
                    use_subprocess=use_subprocess,
                    timeout=timeout,
                    probe_entry_func=_bench_entry,
                )

            baseline_us: float | None = None
            first = True
            for name in ordered_names:
                status, timing = run_bench_for_backend(
                    _bench_entry,
                    name,
                    local_cfg,
                    dry_run_iters,
                    repeat_iters,
                    cold_l2_cache,
                    use_cuda_graph,
                    timing_method,
                    use_subprocess=use_subprocess,
                    timeout=timeout,
                )

                if status != "OK":
                    from core.metrics import BenchmarkMetrics

                    m = BenchmarkMetrics()
                else:
                    m = compute_metrics(
                        timing,
                        batch_size=cfg["batch_size"],
                        s_q=s_q,
                        s_kv=s_kv,
                        num_q_heads=cfg["num_q_heads"],
                        num_kv_heads=cfg["num_kv_heads"],
                        head_dim=cfg["head_dim"],
                        causal=True,
                        dtype_bytes=1 if is_fp8 else 2,
                        kv_dtype_bytes=1 if is_fp8 else 2,
                        out_dtype_bytes=2,
                        gpu_type=cfg.get("gpu_type", "b200"),
                    )

                    if name == speedup_baseline:
                        baseline_us = m.median_us
                        m.speedup = 1.0
                    elif baseline_us:
                        m.speedup = m.median_us / baseline_us

                rc = refcheck_results.get(name)
                refcheck_pass = rc.passed if rc is not None else None
                mismatch_pct = rc.mismatch_pct if rc is not None else None
                row = AttnRowResult(
                    scenario="decode_gqa",
                    backend=name,
                    batch_size=cfg["batch_size"],
                    s_q=s_q,
                    s_kv=s_kv,
                    num_q_heads=cfg["num_q_heads"],
                    num_kv_heads=cfg["num_kv_heads"],
                    head_dim=cfg["head_dim"],
                    dtype=cfg["dtype"],
                    causal=True,
                    metrics=m,
                    refcheck_pass=refcheck_pass,
                    mismatch_pct=mismatch_pct,
                    case_tag=case_tag,
                    use_cuda_graph=use_cuda_graph,
                    status=status,
                )

                if first:
                    print_scenario_header(row)
                    print_table_header()
                    first = False
                print_table_row(row)
                append_attention_csv(row, output_path)
            gc.collect()
            torch.cuda.empty_cache()
