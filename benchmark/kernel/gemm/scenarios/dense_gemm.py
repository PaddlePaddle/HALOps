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
import os
from dataclasses import dataclass

import torch

from core.timing import bench_gpu_time
from core.metrics import compute_gemm_metrics
from core.refcheck import run_refcheck
from core.runner import run_bench_for_backend, run_refcheck_cross_backends
from core.output import (
    GemmRowResult,
    print_scenario_header,
    print_table_header,
    print_table_row,
    append_gemm_csv,
)
from kernel.gemm.backends.halops import run_dense_gemm
from kernel.gemm.backends.flashinfer import run_fi_dense_gemm
from kernel.gemm.backends.golden import run_dense_golden

from kernel.gemm.backends.deepgemm import (
    run_dense_gemm as run_deepgemm_dense,
    make_dense_inputs as make_deepgemm_inputs,
)

# Scale block size: FP8 W8A8 uses k//32, FP4 W4A4 uses k//16.
FP8_SF_BLOCK = 32
FP4_SF_BLOCK = 16


@dataclass
class GemmInput:
    """Dense GEMM inputs for W8A8 or W4A4 modes."""

    x: torch.Tensor
    w: torch.Tensor
    sfa: torch.Tensor
    sfb: torch.Tensor
    out: torch.Tensor
    m: int
    n: int
    k: int
    sm_count: int
    mode: str = "w8a8"
    x_bf16: torch.Tensor | None = None
    w_bf16: torch.Tensor | None = None
    alpha: torch.Tensor | None = None


def _build_inputs(
    m: int, n: int, k: int, sm_count: int, mode: str = "w8a8"
) -> GemmInput:
    """Build dense GEMM inputs on CUDA for the given quantization mode."""
    from halops.gemm.group_wise_quant import group_wise_quant

    dev = "cuda"
    is_fp4 = mode == "w4a4"

    if is_fp4:
        assert k % (2 * FP4_SF_BLOCK) == 0, f"k={k} must be divisible by 32"
    else:
        assert k % FP8_SF_BLOCK == 0, f"k={k} must be divisible by {FP8_SF_BLOCK}"

    x_bf16 = torch.randn(m, k, dtype=torch.bfloat16, device=dev)
    w_bf16 = torch.randn(n, k, dtype=torch.bfloat16, device=dev)
    x, sfa = group_wise_quant(x_bf16, is_fp4=is_fp4)
    w, sfb = group_wise_quant(w_bf16, is_fp4=is_fp4)
    alpha = torch.ones((1,), dtype=torch.float32, device=dev) if is_fp4 else None

    out = torch.empty(m, n, dtype=torch.bfloat16, device=dev)
    return GemmInput(
        x=x,
        w=w,
        sfa=sfa,
        sfb=sfb,
        out=out,
        m=m,
        n=n,
        k=k,
        sm_count=sm_count,
        mode=mode,
        x_bf16=x_bf16,
        w_bf16=w_bf16,
        alpha=alpha,
    )


def adapt_halops(inp: GemmInput) -> dict:
    """Adapt inputs for HALOps backend."""
    d = {
        "x": inp.x,
        "w": inp.w,
        "sfa": inp.sfa,
        "sfb": inp.sfb,
        "out": inp.out,
        "sm_count": inp.sm_count,
        "mode": inp.mode,
    }
    if inp.alpha is not None:
        d["alpha"] = inp.alpha
    return d


def adapt_fi(inp: GemmInput) -> dict:
    """Adapt inputs for FlashInfer backend."""
    if inp.mode == "w4a4":
        return {
            "x": inp.x,
            "w": inp.w,
            "sfa": inp.sfa,
            "sfb": inp.sfb,
            "alpha": inp.alpha,
            "out": inp.out,
            "mode": "w4a4",
        }
    else:
        return {
            "x": inp.x,
            "w": inp.w,
            "a_descale": inp.sfa,
            "b_descale": inp.sfb.T.contiguous(),
            "out": inp.out,
            "mode": "w8a8",
        }


def adapt_golden(inp: GemmInput) -> dict:
    """Adapt inputs for the torch.matmul golden.

    Passes the BF16 source plus the quantization mode; the golden dequantizes
    the fp8/fp4 values (matching the kernels) before the matmul.
    """
    return {"x_bf16": inp.x_bf16, "w_bf16": inp.w_bf16, "mode": inp.mode}


def adapt_deepgemm(inp: GemmInput) -> dict:
    """Adapt inputs for DeepGEMM backend.

    Re-quantizes from the shared bf16 source (x_bf16 / w_bf16) using
    DeepGEMM's own per-token / per-block / per-token-fp4 + UE8M0 convention,
    so the refcheck comparison stays fair against fi/halops.
    """
    return make_deepgemm_inputs(inp.x_bf16, inp.w_bf16, inp.mode)


def _get_backends(names: list[str]) -> dict:
    """Return the subset of backends matching the requested names."""
    registry = {
        "golden": (adapt_golden, run_dense_golden),
        "halops": (adapt_halops, run_dense_gemm),
        "fi": (adapt_fi, run_fi_dense_gemm),
        "deepgemm": (adapt_deepgemm, run_deepgemm_dense),
    }
    backends = {}
    for name in names:
        if name in registry:
            backends[name] = registry[name]
        else:
            print(f"[WARN] unknown backend '{name}', skipped")
    return backends


def _refcheck_entry(
    cfg, ordered_names, refcheck_reference, refcheck_atol=1e-2, refcheck_rtol=1e-2
):
    """Subprocess entry for refcheck."""
    m, n, k = cfg["m"], cfg["n"], cfg["k"]
    sm_count = cfg.get("sm_count", -1)
    mode = cfg.get("mode", "w8a8")
    # Put reference backend first so it becomes the comparison baseline
    ordered = [refcheck_reference] + [
        n_ for n_ in ordered_names if n_ != refcheck_reference
    ]
    raw_inp = _build_inputs(m, n, k, sm_count, mode=mode)
    backends = _get_backends(ordered)
    run_fns = {n_: entry[1] for n_, entry in backends.items()}
    inp_dicts = {n_: entry[0](raw_inp) for n_, entry in backends.items()}
    return run_refcheck(
        [n_ for n_ in ordered if n_ in backends],
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
    """Subprocess entry: build inputs, adapt, bench a single backend."""
    m, n, k = cfg["m"], cfg["n"], cfg["k"]
    sm_count = cfg.get("sm_count", -1)
    mode = cfg.get("mode", "w8a8")
    raw_inp = _build_inputs(m, n, k, sm_count, mode=mode)
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


def run_scenario(cfg: dict) -> None:
    """Run the dense GEMM benchmark scenario."""
    mode = cfg.get("mode", "w8a8")
    out_path = cfg.get("output_path", "results/dense_fp8.csv")
    dry_run_iters = cfg.get("dry_run_iters", 10)
    repeat_iters = cfg.get("repeat_iters", 50)
    cold_l2_cache = cfg.get("cold_l2_cache", True)
    timing_method = cfg.get("timing_method", "profiler")
    backend_names = cfg.get("backends", ["halops"])
    case_tag = cfg.get("case_tag", "")
    do_refcheck = cfg.get("refcheck", True)
    refcheck_reference = cfg.get("refcheck_reference", "golden")
    refcheck_atol = cfg.get("refcheck_atol", 1e-2)
    refcheck_rtol = cfg.get("refcheck_rtol", 1e-2)
    use_subprocess = cfg.get("use_subprocess", True)
    timeout = cfg.get("timeout", 120)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # Bytes per operand element for the bandwidth metric: FP8 = 1, FP4 = 0.5.
    dtype_bytes = 0.5 if mode == "w4a4" else 1
    dtype_str = "fp4_e2m1" if mode == "w4a4" else "fp8_e4m3"

    for shape in cfg["shapes"]:
        m, n, k = shape
        local_cfg = {**cfg, "m": m, "n": n, "k": k}

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
                print(f"[{status}] {name} m={m},n={n},k={k}")
                continue

            metrics = compute_gemm_metrics(
                timing,
                m,
                n,
                k,
                dtype_bytes=dtype_bytes,
                baseline_us=baseline_us,
                gpu_type=cfg.get("gpu_type", "b200"),
            )
            if name == "halops":
                baseline_us = metrics.median_us
            rc = refcheck_results.get(name)
            refcheck_pass = rc.passed if rc is not None else None
            mismatch_pct = rc.mismatch_pct if rc is not None else None
            row = GemmRowResult(
                scenario="dense_gemm",
                backend=name,
                m=m,
                n=n,
                k=k,
                mode=mode,
                dtype=dtype_str,
                metrics=metrics,
                refcheck_pass=refcheck_pass,
                case_tag=case_tag,
                mismatch_pct=mismatch_pct,
            )
            if first:
                print_scenario_header(row)
                print_table_header()
                first = False
            print_table_row(row)
            append_gemm_csv(row, out_path)
