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
from kernel.gemm.backends.halops import (
    run_group_masked_gemm,
    run_group_contiguous_gemm,
)
from kernel.gemm.backends.flashinfer import (
    run_fi_group_masked_gemm,
    run_fi_group_contiguous_gemm,
)
from kernel.gemm.backends.golden import run_grouped_golden
from kernel.gemm.backends.deepgemm import (
    run_group_masked_gemm as run_deepgemm_masked,
    run_group_contiguous_gemm as run_deepgemm_contiguous,
    make_grouped_masked_inputs as make_deepgemm_masked_inputs,
    make_grouped_contiguous_inputs as make_deepgemm_contiguous_inputs,
)

# Scale block size: FP8 W8A8 uses k//32, FP4 NVFP4 uses k//16.
FP8_SF_BLOCK = 32
FP4_SF_BLOCK = 16
GROUPED_MODES = ("w8a8", "w4a4-nvfp4")


@dataclass
class GroupGemmInput:
    """Grouped GEMM inputs for masked or contiguous scenarios."""

    x: torch.Tensor
    w: torch.Tensor
    sfa: torch.Tensor
    sfb: torch.Tensor
    out: torch.Tensor
    num_experts: int
    max_m: int
    n: int
    k: int
    sm_count: int
    mode: str = "w8a8"
    x_bf16: torch.Tensor | None = None
    w_bf16: torch.Tensor | None = None
    alpha: torch.Tensor | None = None
    masked_m: torch.Tensor | None = None  # masked scenario
    expert_ids: torch.Tensor | None = None  # contiguous scenario
    token_counts: torch.Tensor | None = None  # contiguous scenario


def _make_grouped_tensors(num_experts: int, max_m: int, n: int, k: int, mode: str):
    """Build grouped GEMM inputs on CUDA for the given quantization mode.

    Activations/weights are generated as random BF16 and then quantized with
    group_wise_quant (group_size=32 for FP8, 16 for FP4, CUTLASS SfAtom scale
    layout). Keeping the BF16 source explicit lets comparison backends quantize
    from the same data so cross-backend comparisons stay fair.
      - w8a8: x/w float8_e4m3fn [.,K], scales [.,K//32], no alpha.
      - w4a4-nvfp4: x/w uint8-packed FP4 [.,K//2], scales [.,K//16], alpha=ones[E].
    """
    from halops.gemm.group_wise_quant import group_wise_quant

    dev = "cuda"
    m = num_experts * max_m
    if mode not in GROUPED_MODES:
        raise ValueError(
            f"Unsupported grouped benchmark mode: {mode!r}. "
            f"Expected one of {GROUPED_MODES}."
        )
    is_fp4 = mode == "w4a4-nvfp4"

    if is_fp4:
        assert k % (2 * FP4_SF_BLOCK) == 0, f"k={k} must be divisible by 32"
    else:
        assert k % FP8_SF_BLOCK == 0, f"k={k} must be divisible by {FP8_SF_BLOCK}"

    x_bf16 = torch.randn(m, k, dtype=torch.bfloat16, device=dev)
    x, sfa = group_wise_quant(x_bf16, is_fp4=is_fp4)

    # Per-expert weight: quantize each [n, k] slice from its own BF16 source.
    w_bf16_list, w_quant, w_sf = [], [], []
    for _ in range(num_experts):
        w_bf16 = torch.randn(n, k, dtype=torch.bfloat16, device=dev)
        w_e, sfb_e = group_wise_quant(w_bf16, is_fp4=is_fp4)
        w_bf16_list.append(w_bf16)
        w_quant.append(w_e)
        w_sf.append(sfb_e)
    w = torch.stack(w_quant, dim=0)
    sfb = torch.stack(w_sf, dim=0)
    w_bf16_stacked = torch.stack(w_bf16_list, dim=0)

    alpha = (
        torch.ones((num_experts,), dtype=torch.float32, device=dev) if is_fp4 else None
    )
    out = torch.empty(m, n, dtype=torch.bfloat16, device=dev)
    return x, w, sfa, sfb, out, alpha, x_bf16, w_bf16_stacked


def _make_contiguous_expert_ids(m: int, num_experts: int, tile_m: int):
    """Build per-row expert ids aligned to tile_m boundaries."""
    actual_experts = min(num_experts, max(1, m // tile_m))
    per_expert = (m // actual_experts // tile_m) * tile_m
    if per_expert <= 0:
        raise ValueError(f"m={m} is too small for tile_m={tile_m}")

    chunks = []
    token_counts = []
    used = 0
    for expert_id in range(actual_experts):
        count = per_expert if expert_id < actual_experts - 1 else m - used
        count = (count // tile_m) * tile_m
        if count <= 0:
            break
        chunks.append(torch.full((count,), expert_id, dtype=torch.int32))
        token_counts.append(count)
        used += count
    expert_ids = torch.cat(chunks)
    return expert_ids, torch.tensor(token_counts, dtype=torch.int32)


def _build_inputs(
    num_experts: int,
    max_m: int,
    n: int,
    k: int,
    sm_count: int,
    mode: str,
    scenario: str = "masked",
) -> GroupGemmInput:
    """Build grouped GEMM input for masked or contiguous scenario."""
    if scenario == "contiguous":
        m = num_experts * max_m
        tile_m = 256 if m >= 4096 else 128
        expert_ids, token_counts = _make_contiguous_expert_ids(m, num_experts, tile_m)
        actual_experts = token_counts.numel()
        m = expert_ids.numel()
        expert_ids = expert_ids.cuda()
        x, w, sfa, sfb, out, alpha, x_bf16, w_bf16 = _make_grouped_tensors(
            actual_experts, m // actual_experts, n, k, mode
        )
        return GroupGemmInput(
            x=x,
            w=w,
            sfa=sfa,
            sfb=sfb,
            out=out,
            num_experts=actual_experts,
            max_m=max_m,
            n=n,
            k=k,
            sm_count=sm_count,
            mode=mode,
            x_bf16=x_bf16,
            w_bf16=w_bf16,
            alpha=alpha,
            expert_ids=expert_ids,
            token_counts=token_counts,
        )
    else:
        x, w, sfa, sfb, out, alpha, x_bf16, w_bf16 = _make_grouped_tensors(
            num_experts, max_m, n, k, mode
        )
        masked_m = torch.full((num_experts,), max_m, dtype=torch.int32, device="cuda")
        return GroupGemmInput(
            x=x,
            w=w,
            sfa=sfa,
            sfb=sfb,
            out=out,
            num_experts=num_experts,
            max_m=max_m,
            n=n,
            k=k,
            sm_count=sm_count,
            mode=mode,
            x_bf16=x_bf16,
            w_bf16=w_bf16,
            alpha=alpha,
            masked_m=masked_m,
        )


def adapt_halops(inp: GroupGemmInput, scenario: str = "masked") -> dict:
    """Adapt GroupGemmInput to halops backend dict."""
    d = {
        "x": inp.x,
        "w": inp.w,
        "sfa": inp.sfa,
        "sfb": inp.sfb,
        "out": inp.out,
        "mode": inp.mode,
        "sm_count": inp.sm_count,
    }
    if inp.alpha is not None:
        d["alpha"] = inp.alpha
    if scenario == "contiguous":
        d["expert_ids"] = inp.expert_ids
    else:
        d["masked_m"] = inp.masked_m
        d["max_m"] = inp.max_m
    return d


def _build_m_indptr(inp: GroupGemmInput, scenario: str) -> torch.Tensor:
    """Per-group row offsets (prefix sum) shared by the fi and golden adapters."""
    if scenario == "contiguous":
        counts = inp.token_counts.cuda()
    else:
        counts = inp.masked_m
    m_indptr = torch.zeros(counts.numel() + 1, dtype=torch.int32, device=counts.device)
    m_indptr[1:] = torch.cumsum(counts, dim=0)
    return m_indptr


def adapt_fi(inp: GroupGemmInput, scenario: str = "masked") -> dict:
    """Adapt GroupGemmInput to flashinfer backend dict (m_indptr from masked_m or token_counts)."""
    m_indptr = _build_m_indptr(inp, scenario)
    return {
        "x": inp.x,
        "w": inp.w,
        "sfa": inp.sfa,
        "sfb": inp.sfb,
        "m_indptr": m_indptr,
        "mode": inp.mode,
    }


def adapt_golden(inp: GroupGemmInput, scenario: str = "masked") -> dict:
    """Adapt to the torch.matmul golden (BF16 source + mode + group offsets).

    The golden dequantizes the fp8/fp4 values (matching the kernels) before the
    per-group matmul.
    """
    m_indptr = _build_m_indptr(inp, scenario)
    return {
        "x_bf16": inp.x_bf16,
        "w_bf16": inp.w_bf16,
        "m_indptr": m_indptr,
        "mode": inp.mode,
    }


def adapt_deepgemm(inp: GroupGemmInput, scenario: str = "masked") -> dict:
    """Adapt GroupGemmInput to DeepGEMM backend dict.

    Re-quantizes from the shared bf16 source (x_bf16 / w_bf16) using
    DeepGEMM's own per-token / per-block / per-token-fp4 + UE8M0 convention.
    masked: reshape x_bf16 to [E, max_m, K], 3D output.
    contiguous: reuse inp.expert_ids as m_indices (already tile-aligned).
    """
    if scenario == "contiguous":
        return make_deepgemm_contiguous_inputs(
            inp.x_bf16, inp.w_bf16, inp.expert_ids, inp.mode
        )
    return make_deepgemm_masked_inputs(
        inp.x_bf16, inp.w_bf16, inp.masked_m, inp.max_m, inp.mode
    )


def _get_backends(scenario: str, names: list[str]) -> dict:
    """Return the subset of backends matching the requested names for given scenario."""
    if scenario == "masked":
        registry = {
            "golden": (adapt_golden, run_grouped_golden),
            "halops": (adapt_halops, run_group_masked_gemm),
            "fi": (adapt_fi, run_fi_group_masked_gemm),
            "deepgemm": (adapt_deepgemm, run_deepgemm_masked),
        }
    else:
        registry = {
            "golden": (adapt_golden, run_grouped_golden),
            "halops": (adapt_halops, run_group_contiguous_gemm),
            "fi": (adapt_fi, run_fi_group_contiguous_gemm),
            "deepgemm": (adapt_deepgemm, run_deepgemm_contiguous),
        }
    backends = {}
    for name in names:
        if name in registry:
            backends[name] = registry[name]
        else:
            print(f"[WARN] unknown backend '{name}', skipped")
    return backends


def _refcheck_entry(
    cfg,
    ordered_names,
    refcheck_reference,
    refcheck_atol=1e-2,
    refcheck_rtol=1e-2,
):
    """Subprocess entry for grouped GEMM refcheck."""
    num_experts = cfg["num_experts"]
    max_m = cfg["max_m"]
    n, k = cfg["n"], cfg["k"]
    sm_count = cfg.get("sm_count", -1)
    mode = cfg.get("mode", "w8a8")
    scenario = cfg.get("scenario", "masked")
    ordered = [refcheck_reference] + [
        n_ for n_ in ordered_names if n_ != refcheck_reference
    ]
    inp = _build_inputs(num_experts, max_m, n, k, sm_count, mode, scenario)
    backends = _get_backends(scenario, ordered)
    run_fns = {n_: entry[1] for n_, entry in backends.items()}
    inp_dicts = {n_: entry[0](inp, scenario) for n_, entry in backends.items()}
    return run_refcheck(
        ordered,
        run_fns,
        inp_dicts,
        atol=refcheck_atol,
        rtol=refcheck_rtol,
    )


def _bench_entry(
    cfg,
    backend_name,
    dry_run_iters,
    repeat_iters,
    cold_l2_cache,
    use_cuda_graph,
    timing_method,
):
    """Subprocess entry: build inputs, adapt, bench a single backend."""
    num_experts = cfg["num_experts"]
    max_m = cfg["max_m"]
    n, k = cfg["n"], cfg["k"]
    sm_count = cfg.get("sm_count", -1)
    mode = cfg.get("mode", "w8a8")
    scenario = cfg.get("scenario", "masked")
    inp = _build_inputs(num_experts, max_m, n, k, sm_count, mode, scenario)
    backends = _get_backends(scenario, [backend_name])
    if backend_name not in backends:
        raise RuntimeError(f"backend '{backend_name}' not found")

    adapt_fn, run_fn = backends[backend_name]
    run_inp = adapt_fn(inp, scenario)
    return bench_gpu_time(
        run_fn,
        (run_inp,),
        dry_run_iters=dry_run_iters,
        repeat_iters=repeat_iters,
        cold_l2_cache=cold_l2_cache,
    )


def run_group_masked(cfg: dict) -> None:
    """Run the grouped masked GEMM benchmark scenario."""
    mode = cfg.get("mode", "w8a8")
    out_path = cfg.get("output_path", "results/group_masked_fp8.csv")
    case_tag = cfg.get("case_tag", "")
    gpu_type = cfg.get("gpu_type", "b200")
    backend_names = cfg.get("backends", ["halops"])
    dry_run_iters = cfg.get("dry_run_iters", 10)
    repeat_iters = cfg.get("repeat_iters", 50)
    cold_l2_cache = cfg.get("cold_l2_cache", True)
    use_subprocess = cfg.get("use_subprocess", True)
    timeout = cfg.get("timeout", 120)
    do_refcheck = cfg.get("refcheck", True)
    refcheck_reference = cfg.get("refcheck_reference", "golden")
    refcheck_atol = cfg.get("refcheck_atol", 1e-2)
    refcheck_rtol = cfg.get("refcheck_rtol", 1e-2)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    dtype_bytes = 0.5 if mode == "w4a4-nvfp4" else 1
    dtype_str = "fp4_e2m1" if mode == "w4a4-nvfp4" else "fp8_e4m3"

    for num_experts, max_m, n, k in cfg["shapes"]:
        active_m = num_experts * max_m
        local_cfg = {
            **cfg,
            "num_experts": num_experts,
            "max_m": max_m,
            "n": n,
            "k": k,
            "scenario": "masked",
        }

        refcheck_results = {n_: None for n_ in backend_names}
        if do_refcheck:
            refcheck_results = run_refcheck_cross_backends(
                _refcheck_entry,
                local_cfg,
                backend_names,
                refcheck_reference,
                refcheck_atol,
                refcheck_rtol,
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
                False,
                "profiler",
                use_subprocess=use_subprocess,
                timeout=timeout,
            )
            if status != "OK":
                print(
                    f"[{status}] group_masked {name} e={num_experts},max_m={max_m},n={n},k={k}"
                )
                continue

            metrics = compute_gemm_metrics(
                timing,
                active_m,
                n,
                k,
                dtype_bytes=dtype_bytes,
                baseline_us=baseline_us,
                gpu_type=gpu_type,
                num_experts=num_experts,
            )
            if baseline_us is None:
                baseline_us = metrics.median_us
            rc = refcheck_results.get(name)
            refcheck_pass = rc.passed if rc is not None else None
            mismatch_pct = rc.mismatch_pct if rc is not None else None
            row = GemmRowResult(
                scenario="group_masked",
                backend=name,
                m=active_m,
                n=n,
                k=k,
                mode=mode,
                dtype=dtype_str,
                metrics=metrics,
                refcheck_pass=refcheck_pass,
                case_tag=case_tag,
                num_experts=num_experts,
                mismatch_pct=mismatch_pct,
            )
            if first:
                print_scenario_header(row)
                print_table_header()
                first = False
            print_table_row(row)
            append_gemm_csv(row, out_path)


def run_group_contiguous(cfg: dict) -> None:
    """Run the grouped contiguous GEMM benchmark scenario."""
    mode = cfg.get("mode", "w8a8")
    out_path = cfg.get("output_path", "results/group_contiguous_fp8.csv")
    case_tag = cfg.get("case_tag", "")
    gpu_type = cfg.get("gpu_type", "b200")
    backend_names = cfg.get("backends", ["halops"])
    dry_run_iters = cfg.get("dry_run_iters", 10)
    repeat_iters = cfg.get("repeat_iters", 50)
    cold_l2_cache = cfg.get("cold_l2_cache", True)
    use_subprocess = cfg.get("use_subprocess", True)
    timeout = cfg.get("timeout", 120)
    do_refcheck = cfg.get("refcheck", True)
    refcheck_reference = cfg.get("refcheck_reference", "golden")
    refcheck_atol = cfg.get("refcheck_atol", 1e-2)
    refcheck_rtol = cfg.get("refcheck_rtol", 1e-2)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    dtype_bytes = 0.5 if mode == "w4a4-nvfp4" else 1
    dtype_str = "fp4_e2m1" if mode == "w4a4-nvfp4" else "fp8_e4m3"

    for requested_experts, max_m, n, k in cfg["shapes"]:
        m = requested_experts * max_m
        local_cfg = {
            **cfg,
            "num_experts": requested_experts,
            "max_m": max_m,
            "n": n,
            "k": k,
            "scenario": "contiguous",
        }

        refcheck_results: dict[str, bool | None] = {n_: None for n_ in backend_names}
        if do_refcheck:
            refcheck_results = run_refcheck_cross_backends(
                _refcheck_entry,
                local_cfg,
                backend_names,
                refcheck_reference,
                refcheck_atol,
                refcheck_rtol,
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
                False,
                "profiler",
                use_subprocess=use_subprocess,
                timeout=timeout,
            )
            if status != "OK":
                print(
                    f"[{status}] group_contiguous {name} e={requested_experts},m={m},n={n},k={k}"
                )
                continue

            metrics = compute_gemm_metrics(
                timing,
                m,
                n,
                k,
                dtype_bytes=dtype_bytes,
                baseline_us=baseline_us,
                gpu_type=gpu_type,
                num_experts=requested_experts,
            )
            if baseline_us is None:
                baseline_us = metrics.median_us
            rc = refcheck_results.get(name)
            refcheck_pass = rc.passed if rc is not None else None
            mismatch_pct = rc.mismatch_pct if rc is not None else None
            row = GemmRowResult(
                scenario="group_contiguous",
                backend=name,
                m=m,
                n=n,
                k=k,
                mode=mode,
                dtype=dtype_str,
                metrics=metrics,
                refcheck_pass=refcheck_pass,
                case_tag=case_tag,
                num_experts=requested_experts,
                max_m=max_m,
                mismatch_pct=mismatch_pct,
            )
            if first:
                print_scenario_header(row)
                print_table_header()
                first = False
            print_table_row(row)
            append_gemm_csv(row, out_path)
