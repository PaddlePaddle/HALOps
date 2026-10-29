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

Batch GEMM benchmark scenario for BF16 on SM100.

Mirrors dense_gemm.py's structure. Shapes are [B, M, N, K] where B is the
batch (expert) count, and each batch item computes output[b] = input[b] @
weight[b]^T, i.e. [M, K] @ [K, N] -> [M, N].

The batch count B maps to GemmRowResult.num_experts so the existing output
infrastructure (header, table, CSV) renders it without modification.

Backends:
  halops     -- our native BF16 batch GEMM (batch_gemm.cu, 3-tier dispatch)
  fi         -- flashinfer.bmm_bf16, cuDNN backend (same default as the dense
                bf16 path)
  fi_cutlass -- flashinfer.bmm_bf16, CUTLASS backend, so flashinfer's own
                kernels and its cuDNN fallback can be told apart
  bmm        -- torch.bmm (cuBLAS batch GEMM), the reference the model
                currently uses in production via aten::bmm

deep_gemm has no batched BF16 entry point (dense/grouped only), so unlike the
dense scenario there is no deepgemm row here.
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from functools import partial

import torch

from core.timing import bench_gpu_time
from core.metrics import compute_gemm_metrics
from core.runner import run_bench_for_backend, run_refcheck_cross_backends
from core.output import (
    GemmRowResult,
    print_scenario_header,
    print_table_header,
    print_table_row,
    append_gemm_csv,
)

from halops import batch_gemm
from kernel.gemm.backends.flashinfer import run_fi_batch_gemm


@dataclass
class BatchGemmInput:
    """Batch GEMM inputs for BF16 mode."""

    x: torch.Tensor  # [B, M, K] bfloat16
    w: torch.Tensor  # [B, N, K] bfloat16
    out: torch.Tensor  # [B, M, N] bfloat16
    b: int
    m: int
    n: int
    k: int
    sm_count: int
    mode: str = "bf16"


def _build_inputs(
    b: int, m: int, n: int, k: int, sm_count: int, mode: str = "bf16"
) -> BatchGemmInput:
    """Build batch GEMM inputs on CUDA for BF16 mode."""
    dev = "cuda"
    assert mode == "bf16", f"batch scenario supports bf16 only, got {mode}"
    assert k % 64 == 0, f"k={k} must be divisible by 64 for bf16"

    x = torch.randn(b, m, k, dtype=torch.bfloat16, device=dev)
    w = torch.randn(b, n, k, dtype=torch.bfloat16, device=dev)
    out = torch.empty(b, m, n, dtype=torch.bfloat16, device=dev)

    return BatchGemmInput(
        x=x,
        w=w,
        out=out,
        b=b,
        m=m,
        n=n,
        k=k,
        sm_count=sm_count,
        mode=mode,
    )


def adapt_halops(inp: BatchGemmInput) -> dict:
    """Adapt inputs for the HALOps backend."""
    return {
        "x": inp.x,
        "w": inp.w,
        "out": inp.out,
        "sfa": None,
        "sfb": None,
        "bias": None,
        "alpha": None,
        "sm_count": inp.sm_count,
        "mode": "bf16",
    }


def adapt_bmm(inp: BatchGemmInput) -> dict:
    """Adapt inputs for the torch.bmm (cuBLAS) reference."""
    return {"x": inp.x, "w": inp.w, "out": inp.out}


def adapt_fi(inp: BatchGemmInput, backend: str = "cudnn") -> dict:
    """Inputs for flashinfer.bmm_bf16.

    It wants B as (b, k, n) column-major, which is exactly our (b, n, k) weight
    transposed -- no copy, just strides on the tensor we already have.
    """
    return {
        "x": inp.x,
        "bt": inp.w.transpose(-2, -1),
        "out": inp.out,
        "backend": backend,
    }


def run_halops_batch(inp: dict):
    """Run the HALOps batch GEMM kernel."""
    batch_gemm(
        inp["x"],
        inp["w"],
        inp["sfa"],
        inp["sfb"],
        inp["out"],
        bias=inp["bias"],
        alpha=inp["alpha"],
        sm_count=inp["sm_count"],
        mode=inp["mode"],
    )


def run_bmm_reference(inp: dict):
    """Reference: torch.bmm -> cuBLAS batch GEMM (what aten::bmm calls)."""
    torch.bmm(inp["x"], inp["w"].transpose(-2, -1), out=inp["out"])


def run_bmm_golden(inp: dict):
    """Golden reference: bf16 matmul in fp32 accumulation."""
    x_f = inp["x"].float()
    w_f = inp["w"].float().transpose(-2, -1)
    inp["out"].copy_(torch.bmm(x_f, w_f).to(torch.bfloat16))


def _get_backends(names: list[str]) -> dict:
    registry = {
        "golden": (adapt_bmm, run_bmm_golden),
        "halops": (adapt_halops, run_halops_batch),
        "fi": (partial(adapt_fi, backend="cudnn"), run_fi_batch_gemm),
        "fi_cutlass": (partial(adapt_fi, backend="cutlass"), run_fi_batch_gemm),
        "bmm": (adapt_bmm, run_bmm_reference),
    }
    backends = {}
    for name in names:
        if name in registry:
            backends[name] = registry[name]
        else:
            print(f"[WARN] unknown backend '{name}', skipped")
    return backends


def _bench_entry(
    cfg,
    name,
    dry_run_iters,
    repeat_iters,
    cold_l2_cache,
    use_cuda_graph,
    timing_method,
):
    """Subprocess entry: build inputs, adapt, bench a single backend."""
    b, m, n, k = cfg["b"], cfg["m"], cfg["n"], cfg["k"]
    sm_count = cfg.get("sm_count", -1)
    mode = cfg.get("mode", "bf16")
    raw_inp = _build_inputs(b, m, n, k, sm_count, mode=mode)
    backends = _get_backends([name])
    if name not in backends:
        raise RuntimeError(f"backend '{name}' not found")
    adapt_fn, run_fn = backends[name]
    inp = adapt_fn(raw_inp)

    return bench_gpu_time(
        run_fn,
        (inp,),
        dry_run_iters=dry_run_iters,
        repeat_iters=repeat_iters,
        cold_l2_cache=cold_l2_cache,
        method=timing_method,
    )


def _refcheck_entry(
    cfg, ordered_names, refcheck_reference, refcheck_atol=1e-2, refcheck_rtol=1e-2
):
    """Subprocess entry for refcheck: run all backends + golden, compare."""
    b, m, n, k = cfg["b"], cfg["m"], cfg["n"], cfg["k"]
    sm_count = cfg.get("sm_count", -1)
    mode = cfg.get("mode", "bf16")
    raw_inp = _build_inputs(b, m, n, k, sm_count, mode=mode)

    backends = _get_backends(ordered_names)
    golden_fn = run_bmm_golden
    golden_inp = adapt_bmm(raw_inp)
    golden_inp["out"].zero_()
    golden_fn(golden_inp)
    torch.cuda.synchronize()
    ref_out = golden_inp["out"].clone()

    results = {}
    for name in ordered_names:
        if name not in backends or name == refcheck_reference:
            continue
        adapt_fn, run_fn = backends[name]
        inp = adapt_fn(raw_inp)
        inp["out"].zero_()
        run_fn(inp)
        torch.cuda.synchronize()
        out = inp["out"]
        diff = (out.float() - ref_out.float()).abs()
        tol = refcheck_atol + refcheck_rtol * ref_out.float().abs()
        n_bad = (diff > tol).sum().item()
        mismatch = 100.0 * n_bad / ref_out.numel()
        from core.refcheck import RefcheckResult

        results[name] = RefcheckResult(passed=n_bad == 0, mismatch_pct=mismatch)
    return results


def run_scenario(cfg: dict) -> None:
    """Run the batch GEMM benchmark scenario."""
    mode = cfg.get("mode", "bf16")
    out_path = cfg.get("output_path", "results/batch_bf16.csv")
    case_tag = cfg.get("case_tag", "")
    dry_run_iters = cfg.get("dry_run_iters", 10)
    repeat_iters = cfg.get("repeat_iters", 50)
    cold_l2_cache = cfg.get("cold_l2_cache", True)
    timing_method = cfg.get("timing_method", "profiler")
    backend_names = cfg.get("backends", ["halops"])
    do_refcheck = cfg.get("refcheck", True)
    refcheck_reference = cfg.get("refcheck_reference", "golden")
    refcheck_atol = cfg.get("refcheck_atol", 1e-2)
    refcheck_rtol = cfg.get("refcheck_rtol", 1e-2)
    use_subprocess = cfg.get("use_subprocess", True)
    timeout = cfg.get("timeout", 120)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    dtype_bytes = 2
    dtype_str = "bf16"

    for shape in cfg["shapes"]:
        b, m, n, k = shape
        local_cfg = {**cfg, "b": b, "m": m, "n": n, "k": k}

        refcheck_results = {n_: None for n_ in backend_names}
        if do_refcheck:
            refcheck_results = run_refcheck_cross_backends(
                _refcheck_entry,
                local_cfg,
                backend_names,
                refcheck_reference,
                refcheck_atol=refcheck_atol,
                refcheck_rtol=refcheck_rtol,
                use_subprocess=use_subprocess,
                timeout=timeout,
                probe_entry_func=_bench_entry,
            )

        first = True
        baseline_us = None
        for name in backend_names:
            status, timing = run_bench_for_backend(
                _bench_entry,
                name,
                local_cfg,
                dry_run_iters,
                repeat_iters,
                cold_l2_cache,
                False,  # use_cuda_graph not used in GEMM
                timing_method,
                use_subprocess=use_subprocess,
                timeout=timeout,
            )
            if status != "OK":
                print(f"[{status}] {name} b={b},m={m},n={n},k={k}")
                continue

            metrics = compute_gemm_metrics(
                timing,
                b * m,  # total M for FLOPs calculation
                n,
                k,
                dtype_bytes=dtype_bytes,
                baseline_us=baseline_us,
                gpu_type=cfg.get("gpu_type", "b200"),
                num_experts=b,  # each expert loads [n, k] once
            )
            if name == "halops":
                baseline_us = metrics.median_us
            rc = refcheck_results.get(name)
            refcheck_pass = rc.passed if rc is not None else None
            mismatch_pct = rc.mismatch_pct if rc is not None else None
            row = GemmRowResult(
                scenario="batch_gemm",
                backend=name,
                m=m,  # per-batch-item M (total = b*m shown via num_experts)
                n=n,
                k=k,
                mode=mode,
                dtype=dtype_str,
                metrics=metrics,
                refcheck_pass=refcheck_pass,
                case_tag=case_tag,
                num_experts=b,
                mismatch_pct=mismatch_pct,
            )
            if first:
                print_scenario_header(row)
                print_table_header()
                first = False
            print_table_row(row)
            append_gemm_csv(row, out_path)

        if not first:
            print()
