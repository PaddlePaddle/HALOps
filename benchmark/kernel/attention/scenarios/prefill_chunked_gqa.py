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
from typing import Optional

import torch

from core.metrics import compute_metrics
from core.refcheck import run_refcheck
from core.runner import run_bench_for_backend, run_refcheck_cross_backends
from core.output import (
    AttnRowResult,
    print_scenario_header,
    print_table_header,
    print_table_row,
    append_attention_csv,
)

BLOCK_SIZE = 128


@dataclass
class ChunkedPrefillInput:
    """Inputs for chunked-prefill scenario with prebuilt Q/K/V attention buffers."""

    qkv_input: torch.Tensor
    cu_seq_q: torch.Tensor
    cu_seq_k: torch.Tensor
    rotary_embs: torch.Tensor
    seq_len_encoder: torch.Tensor
    seq_len_decoder: torch.Tensor
    cache_k: torch.Tensor
    cache_v: torch.Tensor
    block_tables: torch.Tensor
    attn_out: torch.Tensor
    mask: torch.Tensor
    kv_dequant_scale: Optional[torch.Tensor]
    quant_mode: str
    batch_size: int
    head_num: int
    kv_head_num: int
    head_dim: int
    prefix_len: int
    chunk_len: int
    max_input_len: int
    max_seq_len_q: int
    max_seq_len_k: int
    softmax_scale: float


def build_inputs(cfg: dict) -> ChunkedPrefillInput:
    """Build inputs: prefill P tokens into cache, then prepare chunked-prefill buffers."""
    from kernel.attention.backends.halops import run_write_cache_kv_encoder

    B = cfg["batch_size"]
    P = cfg["prefix_len"]
    C = cfg["chunk_len"]
    H_Q = cfg["num_q_heads"]
    H_KV = cfg["num_kv_heads"]
    D = cfg["head_dim"]
    quant_mode = cfg.get("cache_quant_type", "none")
    is_fp8 = quant_mode == "block_wise_fp8"
    input_scale = cfg.get("input_scale", 1.0)

    total_kv = P + C
    max_blk = (total_kv + BLOCK_SIZE - 1) // BLOCK_SIZE + 1
    num_phys = B * max_blk
    max_input_len = total_kv

    # Identity RoPE
    rotary_embs = torch.zeros(
        2, max_input_len, D // 2, dtype=torch.float32, device="cuda"
    )
    rotary_embs[0] = 1.0

    # Allocate paged cache
    cache_dtype = torch.uint8 if is_fp8 else torch.bfloat16
    cache_k = torch.zeros(
        num_phys, H_KV, BLOCK_SIZE, D, dtype=cache_dtype, device="cuda"
    )
    cache_v = torch.zeros_like(cache_k)
    block_tables = torch.zeros(B, max_blk, dtype=torch.int32, device="cuda")
    for i in range(B):
        for j in range(max_blk):
            block_tables[i, j] = i * max_blk + j
    kv_dequant_scale = torch.zeros(num_phys, H_KV, 4, device="cuda") if is_fp8 else None

    # Step 1: fresh prefill P tokens into cache (seq_len_decoder=0)
    qkv_prefix = (
        torch.randn(B * P, H_Q + 2 * H_KV, D, device="cuda") * input_scale
    ).to(torch.bfloat16)
    cu_q_p = (torch.arange(B + 1, device="cuda") * P).to(torch.int32)
    cu_k_p = cu_q_p.clone()
    seq_len_enc_P = torch.full((B,), P, dtype=torch.int32, device="cuda")
    seq_len_dec_zero = torch.zeros(B, dtype=torch.int32, device="cuda")
    q_out_p = torch.zeros(B * P, H_Q, D, dtype=torch.bfloat16, device="cuda")
    k_out_p = torch.zeros(B * P, H_KV, D, dtype=torch.bfloat16, device="cuda")
    v_out_p = torch.zeros_like(k_out_p)

    run_write_cache_kv_encoder(
        {
            "qkv_input": qkv_prefix,
            "cu_seq_q": cu_q_p,
            "cu_seq_k": cu_k_p,
            "rotary_embs": rotary_embs,
            "seq_len_encoder": seq_len_enc_P,
            "seq_len_decoder": seq_len_dec_zero,
            "cache_k": cache_k,
            "cache_v": cache_v,
            "block_tables": block_tables,
            "q": q_out_p,
            "k": k_out_p,
            "v": v_out_p,
            "kv_dequant_scale": kv_dequant_scale,
            "head_num": H_Q,
            "kv_head_num": H_KV,
            "head_dim": D,
            "max_seq_len_q": P,
            "max_seq_len_k": P,
            "max_input_len": max_input_len,
            "quant_mode": quant_mode,
        }
    )

    # Step 2: prepare chunked-prefill inputs (seq_len_decoder=P > 0 → triggers read kernel)
    qkv_chunk = (torch.randn(B * C, H_Q + 2 * H_KV, D, device="cuda") * input_scale).to(
        torch.bfloat16
    )
    cu_seq_q = (torch.arange(B + 1, device="cuda") * C).to(torch.int32)
    cu_seq_k = (torch.arange(B + 1, device="cuda") * (C + P)).to(torch.int32)
    seq_len_encoder = torch.full((B,), C, dtype=torch.int32, device="cuda")
    seq_len_decoder = torch.full((B,), P, dtype=torch.int32, device="cuda")

    # encoder_attention output and causal mask
    attn_out = torch.zeros(B * C, H_Q, D, dtype=torch.bfloat16, device="cuda")
    # mask[i] = number of valid KV positions for query position i (causal within chunk,
    # plus full prefix visible). For chunked-prefill: position j in chunk sees P + j + 1 tokens.
    mask_per_batch = torch.arange(1, C + 1, device="cuda", dtype=torch.int32) + P
    mask = mask_per_batch.repeat(B)

    return ChunkedPrefillInput(
        qkv_input=qkv_chunk,
        cu_seq_q=cu_seq_q,
        cu_seq_k=cu_seq_k,
        rotary_embs=rotary_embs,
        seq_len_encoder=seq_len_encoder,
        seq_len_decoder=seq_len_decoder,
        cache_k=cache_k,
        cache_v=cache_v,
        block_tables=block_tables,
        attn_out=attn_out,
        mask=mask,
        kv_dequant_scale=kv_dequant_scale,
        quant_mode=quant_mode,
        batch_size=B,
        head_num=H_Q,
        kv_head_num=H_KV,
        head_dim=D,
        prefix_len=P,
        chunk_len=C,
        max_input_len=max_input_len,
        max_seq_len_q=C,
        max_seq_len_k=C + P,
        softmax_scale=1.0 / math.sqrt(D),
    )


def _materialize_qkv(inp: ChunkedPrefillInput) -> dict:
    """Run write_cache_kv_encoder to produce fresh dense q/k/v buffers for one backend.

    Each backend gets its own q/k/v so shared cache_k/cache_v state remains consistent
    (write_cache_kv_encoder appends the chunk into cache; running it multiple times on
    the SAME cache is idempotent because seq_len_decoder=P pins the write offset).
    """
    from kernel.attention.backends.halops import run_write_cache_kv_encoder

    B, C, P = inp.batch_size, inp.chunk_len, inp.prefix_len
    H_Q, H_KV, D = inp.head_num, inp.kv_head_num, inp.head_dim
    q = torch.zeros(B * C, H_Q, D, dtype=torch.bfloat16, device="cuda")
    # K/V output buffers larger than Q → triggers the read kernel to reconstruct prefix.
    k = torch.zeros(B * (C + P), H_KV, D, dtype=torch.bfloat16, device="cuda")
    v = torch.zeros(B * (C + P), H_KV, D, dtype=torch.bfloat16, device="cuda")
    run_write_cache_kv_encoder(
        {
            "qkv_input": inp.qkv_input,
            "cu_seq_q": inp.cu_seq_q,
            "cu_seq_k": inp.cu_seq_k,
            "rotary_embs": inp.rotary_embs,
            "seq_len_encoder": inp.seq_len_encoder,
            "seq_len_decoder": inp.seq_len_decoder,
            "cache_k": inp.cache_k,
            "cache_v": inp.cache_v,
            "block_tables": inp.block_tables,
            "q": q,
            "k": k,
            "v": v,
            "kv_dequant_scale": inp.kv_dequant_scale,
            "head_num": inp.head_num,
            "kv_head_num": inp.kv_head_num,
            "head_dim": inp.head_dim,
            "max_seq_len_q": inp.max_seq_len_q,
            "max_seq_len_k": inp.max_seq_len_k,
            "max_input_len": inp.max_input_len,
            "quant_mode": inp.quant_mode,
        }
    )
    return {"q": q, "k": k, "v": v}


def adapt_halops(inp: ChunkedPrefillInput) -> dict:
    """Adapt chunked prefill inputs for HALOps encoder_attention."""
    qkv = _materialize_qkv(inp)
    return {
        "q": qkv["q"],
        "k": qkv["k"],
        "v": qkv["v"],
        "cu_seq_q": inp.cu_seq_q,
        "cu_seq_k": inp.cu_seq_k,
        "attn_out": inp.attn_out,
        "mask": inp.mask,
    }


def adapt_fa_varlen(inp: ChunkedPrefillInput) -> dict:
    """Adapt chunked prefill inputs for FA2/FA4 varlen attention."""
    qkv = _materialize_qkv(inp)
    return {
        "q": qkv["q"],
        "k": qkv["k"],
        "v": qkv["v"],
        "cu_seq_q": inp.cu_seq_q,
        "cu_seq_k": inp.cu_seq_k,
        "max_seq_len_q": inp.max_seq_len_q,
        "max_seq_len_k": inp.max_seq_len_k,
        "softmax_scale": inp.softmax_scale,
        "causal": True,
    }


def adapt_fi(inp: ChunkedPrefillInput, backend: str = "auto") -> dict:
    """Adapt chunked prefill inputs for FlashInfer ragged prefill attention."""
    from flashinfer.prefill import BatchPrefillWithRaggedKVCacheWrapper

    qkv = _materialize_qkv(inp)
    workspace = torch.zeros(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    wrapper = BatchPrefillWithRaggedKVCacheWrapper(workspace, "NHD", backend=backend)
    qo_indptr = inp.cu_seq_q
    kv_indptr = inp.cu_seq_k
    plan_kwargs = {}
    if backend == "cudnn":
        qo_indptr = qo_indptr * (inp.head_num * inp.head_dim)
        kv_indptr = kv_indptr * (inp.kv_head_num * inp.head_dim)
        plan_kwargs.update(
            seq_lens_q=inp.seq_len_encoder,
            seq_lens=inp.seq_len_encoder + inp.seq_len_decoder,
            max_token_per_sequence=inp.chunk_len,
            max_sequence_kv=inp.prefix_len + inp.chunk_len,
        )
    wrapper.plan(
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr,
        num_qo_heads=inp.head_num,
        num_kv_heads=inp.kv_head_num,
        head_dim_qk=inp.head_dim,
        causal=True,
        sm_scale=inp.softmax_scale,
        q_data_type=inp.qkv_input.dtype,
        **plan_kwargs,
    )
    return {"q": qkv["q"], "k": qkv["k"], "v": qkv["v"], "wrapper": wrapper}


def adapt_chunked_golden(inp: ChunkedPrefillInput) -> dict:
    """Adapt chunked prefill inputs for the torch-SDPA reference (`run_golden_chunked`).

    Uses the same materialized varlen q/k/v as fa/fi backends, and passes along the
    shape metadata needed to unflatten them and construct the prefix+causal mask.
    """
    qkv = _materialize_qkv(inp)
    return {
        "q": qkv["q"],
        "k": qkv["k"],
        "v": qkv["v"],
        "batch_size": inp.batch_size,
        "chunk_len": inp.chunk_len,
        "prefix_len": inp.prefix_len,
        "num_q_heads": inp.head_num,
        "num_kv_heads": inp.kv_head_num,
        "head_dim": inp.head_dim,
        "softmax_scale": inp.softmax_scale,
    }


def adapt_fi_paged(
    inp: ChunkedPrefillInput, page_size: int = 128, backend: str = "trtllm-gen"
) -> dict:
    """Adapt chunked prefill inputs for FlashInfer paged prefill attention."""
    from kernel.attention.backends.flashinfer import plan_fi_prefill_paged

    qkv = _materialize_qkv(inp)
    B, C, P = inp.batch_size, inp.chunk_len, inp.prefix_len
    H_Q, H_KV, D = inp.head_num, inp.kv_head_num, inp.head_dim
    total_kv = P + C
    pages_per_seq = (total_kv + page_size - 1) // page_size
    num_pages = B * pages_per_seq

    k = qkv["k"].reshape(B, total_kv, H_KV, D)
    v = qkv["v"].reshape(B, total_kv, H_KV, D)
    kv_data = torch.zeros(
        num_pages, 2, page_size, H_KV, D, dtype=k.dtype, device="cuda"
    )
    for batch_idx in range(B):
        for page_idx in range(pages_per_seq):
            start = page_idx * page_size
            end = min(start + page_size, total_kv)
            length = end - start
            dst = batch_idx * pages_per_seq + page_idx
            kv_data[dst, 0, :length] = k[batch_idx, start:end]
            kv_data[dst, 1, :length] = v[batch_idx, start:end]

    q_stride = C * H_Q * D if backend == "cudnn" else C
    q_indptr = torch.arange(B + 1, dtype=torch.int32, device="cuda") * q_stride
    kv_indptr = torch.arange(B + 1, dtype=torch.int32, device="cuda") * pages_per_seq
    kv_indices = torch.arange(num_pages, dtype=torch.int32, device="cuda")
    last_page_len = torch.full(
        (B,),
        total_kv - (pages_per_seq - 1) * page_size,
        dtype=torch.int32,
        device="cuda",
    )
    workspace = torch.zeros(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    seq_lens_q = (
        torch.full((B,), C, dtype=torch.int32, device="cuda")
        if backend == "cudnn"
        else None
    )
    seq_lens_kv = (
        torch.full((B,), total_kv, dtype=torch.int32, device="cuda")
        if backend == "cudnn"
        else None
    )

    return plan_fi_prefill_paged(
        q=qkv["q"].reshape(B * C, H_Q, D).contiguous(),
        kv_data=kv_data,
        q_indptr=q_indptr,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        kv_last_page_len=last_page_len,
        num_q_heads=H_Q,
        num_kv_heads=H_KV,
        head_dim=D,
        page_size=page_size,
        causal=True,
        softmax_scale=inp.softmax_scale,
        workspace=workspace,
        backend=backend,
        seq_lens=seq_lens_kv,
        seq_lens_q=seq_lens_q,
        max_token_per_sequence=C if backend == "cudnn" else None,
        max_sequence_kv=total_kv if backend == "cudnn" else None,
    )


def _get_backends(backend_names: list[str]) -> dict[str, tuple]:
    from kernel.attention.backends.halops import run_halops_prefill
    from kernel.attention.backends.flash_attention import (
        run_fa2_prefill_varlen,
        run_fa4_prefill_varlen,
    )
    from kernel.attention.backends.flashinfer import (
        run_fi_prefill,
        run_fi_prefill_paged,
    )
    from kernel.attention.backends.golden import run_golden_chunked

    registry = {
        "halops": (adapt_halops, run_halops_prefill),
        "fa2": (adapt_fa_varlen, run_fa2_prefill_varlen),
        "fa4": (adapt_fa_varlen, run_fa4_prefill_varlen),
        "fi(auto)": (lambda inp: adapt_fi(inp, backend="auto"), run_fi_prefill),
        "fi(cudnn)": (
            lambda inp: adapt_fi(inp, backend="cudnn"),
            run_fi_prefill,
        ),
        "fi(trtllm-gen)": (
            lambda inp: adapt_fi_paged(inp, page_size=128, backend="trtllm-gen"),
            run_fi_prefill_paged,
        ),
        "golden": (adapt_chunked_golden, run_golden_chunked),
    }
    return {k: registry[k] for k in backend_names if k in registry}


def _refcheck_entry(
    cfg, ordered_names, refcheck_reference, refcheck_atol=1e-2, refcheck_rtol=1e-2
):
    """Subprocess entry for refcheck."""
    inp = build_inputs(cfg)
    backends = _get_backends(ordered_names)

    # Put refcheck_reference first so it becomes the comparison baseline.
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
    inp = build_inputs(dict(cfg))
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


def run_scenario(cfg: dict) -> None:
    """Bench chunked-prefill attention kernel; write_cache_kv_encoder runs in adapt phase."""
    backend_names = cfg.get("backends", ["halops"])
    dry_run_iters = cfg.get("dry_run_iters", 10)
    repeat_iters = cfg.get("repeat_iters", 50)
    cold_l2_cache = cfg.get("cold_l2_cache", True)
    use_cuda_graph = cfg.get("use_cuda_graph", False)
    timing_method = cfg.get("timing_method", "profiler")
    do_refcheck = cfg.get("refcheck", False)
    output_path = cfg.get("output_path", "results/prefill_chunked.csv")
    case_tag = cfg.get("case_tag", "")
    use_subprocess = cfg.get("use_subprocess", True)
    timeout = cfg.get("timeout", 120)

    def _move_to_front(names: list[str], target: str) -> list[str]:
        if target in names:
            return [target] + [n for n in names if n != target]
        return list(names)

    # benchmark / display order: halops first; it is the speedup baseline.
    speedup_baseline = "halops"
    ordered_names = _move_to_front(backend_names, speedup_baseline)

    batch_size = cfg["batch_size"]
    for prefix_len in cfg["prefix_len_list"]:
        for chunk_len in cfg["chunk_len_list"]:
            local_cfg = {
                **cfg,
                "prefix_len": prefix_len,
                "chunk_len": chunk_len,
            }

            refcheck_results: dict = {n: None for n in backend_names}
            if do_refcheck:
                refcheck_reference = cfg.get("refcheck_reference", "golden")
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
                    s_kv = prefix_len + chunk_len
                    m = compute_metrics(
                        timing,
                        batch_size=batch_size,
                        s_q=chunk_len,
                        s_kv=s_kv,
                        num_q_heads=cfg["num_q_heads"],
                        num_kv_heads=cfg["num_kv_heads"],
                        head_dim=cfg["head_dim"],
                        causal=True,
                        gpu_type=cfg.get("gpu_type", "b200"),
                    )
                    # speedup = candidate median / halops median; halops == 1.0.
                    # >1 means halops is faster than this backend.
                    if name == speedup_baseline:
                        baseline_us = m.median_us
                        m.speedup = 1.0
                    elif baseline_us:
                        m.speedup = m.median_us / baseline_us

                rc = refcheck_results.get(name)
                refcheck_pass = rc.passed if rc is not None else None
                mismatch_pct = rc.mismatch_pct if rc is not None else None

                row = AttnRowResult(
                    scenario="prefill_chunked_gqa",
                    backend=name,
                    batch_size=batch_size,
                    s_q=chunk_len,
                    s_kv=prefix_len + chunk_len,
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
