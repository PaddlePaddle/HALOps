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


@dataclass
class AttentionInput:
    q: torch.Tensor  # [B, S, H_Q, D]
    k: torch.Tensor  # [B, S, H_KV, D]
    v: torch.Tensor  # [B, S, H_KV, D]
    seq_lens: torch.Tensor  # [B]
    causal: bool
    softmax_scale: float


def build_inputs(cfg: dict) -> AttentionInput:
    B = cfg["batch_size"]
    S = cfg["s_kv"]  # prefill: s_q == s_kv
    H_Q = cfg["num_q_heads"]
    H_KV = cfg["num_kv_heads"]
    D = cfg["head_dim"]
    dtype = getattr(torch, cfg["dtype"])
    causal = cfg["causal"]
    input_scale = cfg.get("input_scale", 1.0)

    q = torch.randn(B, S, H_Q, D, dtype=dtype, device="cuda") * input_scale
    k = torch.randn(B, S, H_KV, D, dtype=dtype, device="cuda") * input_scale
    v = torch.randn(B, S, H_KV, D, dtype=dtype, device="cuda") * input_scale
    seq_lens = torch.full((B,), S, dtype=torch.int32, device="cuda")

    return AttentionInput(q, k, v, seq_lens, causal, 1.0 / math.sqrt(D))


def adapt_standard(inp: AttentionInput) -> dict:
    return {
        "q": inp.q.contiguous(),
        "k": inp.k.contiguous(),
        "v": inp.v.contiguous(),
        "softmax_scale": inp.softmax_scale,
        "causal": inp.causal,
    }


def adapt_fi_prefill(inp: AttentionInput) -> dict:
    from kernel.attention.backends.flashinfer import plan_fi_prefill

    B, S = inp.q.shape[:2]
    q_indptr = torch.arange(B + 1, dtype=torch.int32, device="cuda") * S
    kv_indptr = torch.arange(B + 1, dtype=torch.int32, device="cuda") * S
    workspace = torch.zeros(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    return plan_fi_prefill(
        q=inp.q.contiguous(),
        k=inp.k.contiguous(),
        v=inp.v.contiguous(),
        q_indptr=q_indptr,
        kv_indptr=kv_indptr,
        softmax_scale=inp.softmax_scale,
        causal=inp.causal,
        workspace=workspace,
    )


def adapt_fi_prefill_paged(
    inp: AttentionInput, page_size: int = 128, backend: str = "trtllm-gen"
) -> dict:
    """Adapt dense prefill inputs into paged KV format for FlashInfer BatchPrefillWithPagedKVCache."""
    from kernel.attention.backends.flashinfer import plan_fi_prefill_paged

    B, S, H_Q, D = inp.q.shape
    H_KV = inp.k.shape[2]
    pages_per_seq = (S + page_size - 1) // page_size
    num_pages = B * pages_per_seq

    # Build paged KV data [num_pages, 2, page_size, H_KV, D]
    kv_data = torch.zeros(
        num_pages, 2, page_size, H_KV, D, dtype=inp.k.dtype, device="cuda"
    )
    for i in range(B):
        for p in range(pages_per_seq):
            start = p * page_size
            end = min(start + page_size, S)
            length = end - start
            page_idx = i * pages_per_seq + p
            kv_data[page_idx, 0, :length] = inp.k[i, start:end]
            kv_data[page_idx, 1, :length] = inp.v[i, start:end]

    kv_indptr = torch.arange(B + 1, dtype=torch.int32, device="cuda") * pages_per_seq
    kv_indices = torch.arange(num_pages, dtype=torch.int32, device="cuda")
    last_page_len = torch.full(
        (B,), S - (pages_per_seq - 1) * page_size, dtype=torch.int32, device="cuda"
    )
    q_indptr = torch.arange(B + 1, dtype=torch.int32, device="cuda") * S
    workspace = torch.zeros(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")

    return plan_fi_prefill_paged(
        q=inp.q.reshape(B * S, H_Q, D).contiguous(),
        kv_data=kv_data,
        q_indptr=q_indptr,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        kv_last_page_len=last_page_len,
        num_q_heads=H_Q,
        num_kv_heads=H_KV,
        head_dim=D,
        page_size=page_size,
        causal=inp.causal,
        softmax_scale=inp.softmax_scale,
        workspace=workspace,
        backend=backend,
    )


def adapt_sglang_prefill(inp: AttentionInput) -> dict:
    B, S, H_Q, D = inp.q.shape
    _, _, H_KV, _ = inp.k.shape
    b_start_loc = torch.arange(B, dtype=torch.int32, device="cuda") * S
    b_seq_len = torch.full((B,), S, dtype=torch.int32, device="cuda")
    return {
        "q": inp.q.reshape(B * S, H_Q, D).contiguous(),
        "k": inp.k.reshape(B * S, H_KV, D).contiguous(),
        "v": inp.v.reshape(B * S, H_KV, D).contiguous(),
        "b_start_loc": b_start_loc,
        "b_seq_len": b_seq_len,
        "max_input_len": S,
        "softmax_scale": inp.softmax_scale,
        "is_causal": inp.causal,
    }


def adapt_halops_prefill(inp: AttentionInput) -> dict:
    """Flatten to varlen format, pre-allocate attn_out, build cu_seq."""
    B, S, H_Q, D = inp.q.shape
    _, _, H_KV, _ = inp.k.shape
    cu_seq = torch.arange(B + 1, dtype=torch.int32, device="cuda") * S
    return {
        "q": inp.q.reshape(B * S, H_Q, D).contiguous(),
        "k": inp.k.reshape(B * S, H_KV, D).contiguous(),
        "v": inp.v.reshape(B * S, H_KV, D).contiguous(),
        "cu_seq_q": cu_seq,
        "cu_seq_k": cu_seq,
        "attn_out": torch.empty(B * S, H_Q, D, dtype=inp.q.dtype, device="cuda"),
        "mask": None,
    }


def _get_backends(backend_names: list[str]) -> dict[str, tuple]:
    from kernel.attention.backends.flash_attention import (
        run_fa2_prefill,
        run_fa4_prefill,
    )
    from kernel.attention.backends.flashinfer import (
        run_fi_prefill,
        run_fi_prefill_paged,
    )  # plan_fi_prefill is imported locally inside adapt_fi_prefill
    from kernel.attention.backends.sglang_triton import run_sglang_triton_prefill
    from kernel.attention.backends.cudnn import run_cudnn
    from kernel.attention.backends.halops import run_halops_prefill
    from kernel.attention.backends.golden import run_golden

    registry = {
        "halops": (adapt_halops_prefill, run_halops_prefill),
        "fa2": (adapt_standard, run_fa2_prefill),
        "fa4": (adapt_standard, run_fa4_prefill),
        "fi(auto)": (adapt_fi_prefill, run_fi_prefill),
        "fi(trtllm-gen)": (
            lambda inp: adapt_fi_prefill_paged(
                inp, page_size=128, backend="trtllm-gen"
            ),
            run_fi_prefill_paged,
        ),
        "sglang_triton": (adapt_sglang_prefill, run_sglang_triton_prefill),
        "cudnn": (adapt_standard, run_cudnn),
        "golden": (adapt_standard, run_golden),
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
        print_detail=False,
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
    backend_names = cfg["backends"]
    dry_run_iters = cfg.get("dry_run_iters", 10)
    repeat_iters = cfg.get("repeat_iters", 50)
    cold_l2_cache = cfg.get("cold_l2_cache", True)
    use_cuda_graph = cfg.get("use_cuda_graph", False)
    timing_method = cfg.get("timing_method", "profiler")
    do_refcheck = cfg.get("refcheck", True)
    output_path = cfg.get("output_path", "results/out.csv")
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

    for s_kv in cfg["s_kv_list"]:
        local_cfg = {**cfg, "s_kv": s_kv}

        refcheck_results: dict[str, object] = {}
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
                m = compute_metrics(
                    timing,
                    batch_size=cfg["batch_size"],
                    s_q=s_kv,
                    s_kv=s_kv,
                    num_q_heads=cfg["num_q_heads"],
                    num_kv_heads=cfg["num_kv_heads"],
                    head_dim=cfg["head_dim"],
                    causal=cfg["causal"],
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
                scenario="prefill_dense",
                backend=name,
                batch_size=cfg["batch_size"],
                s_q=s_kv,
                s_kv=s_kv,
                num_q_heads=cfg["num_q_heads"],
                num_kv_heads=cfg["num_kv_heads"],
                head_dim=cfg["head_dim"],
                dtype=cfg["dtype"],
                causal=cfg["causal"],
                metrics=m,
                refcheck_pass=refcheck_pass,
                case_tag=case_tag,
                use_cuda_graph=use_cuda_graph,
                status=status,
                mismatch_pct=mismatch_pct,
            )

            if first:
                print_scenario_header(row)
                print_table_header()
                first = False
            print_table_row(row)
            append_attention_csv(row, output_path)
        gc.collect()
        torch.cuda.empty_cache()
