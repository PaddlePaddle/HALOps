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
from typing import Callable, NamedTuple
import re
import torch


class RefcheckResult(NamedTuple):
    """Per-backend correctness check result against a reference backend.

    Attributes:
        passed: Whether the output matches the reference within atol/rtol.
        max_atol: Maximum absolute difference vs reference.
        avg_atol: Mean absolute difference vs reference.
        max_rtol: Maximum relative difference vs reference.
        avg_rtol: Mean relative difference vs reference.
        mismatch_pct: Percentage of elements exceeding tolerance (from assert_close).
    """

    passed: bool
    max_atol: float | None = None
    avg_atol: float | None = None
    max_rtol: float | None = None
    avg_rtol: float | None = None
    mismatch_pct: float | None = None


def _run_and_flatten(
    name: str, run_fn: Callable, inp_dict: dict
) -> torch.Tensor | None:
    """Run a backend and return its flattened float-CPU output, None on error."""
    try:
        out = run_fn(inp_dict)
        torch.cuda.synchronize()
        return out.float().cpu().reshape(-1)
    except Exception as e:
        print(f"[REFCHECK] {name}: run failed -- {e}")
        return None


def _compare_to_reference(
    name: str,
    reference_name: str | None,
    out_flat: torch.Tensor,
    reference_out: torch.Tensor,
    atol: float,
    rtol: float,
) -> RefcheckResult:
    """Compare a candidate output against the reference into a RefcheckResult."""
    diff = (out_flat - reference_out).abs()
    max_atol = diff.max().item()
    avg_atol = diff.mean().item()
    rel_diff = diff / reference_out.abs().clamp(min=1e-12)
    max_rtol = rel_diff.max().item()
    avg_rtol = rel_diff.mean().item()

    try:
        torch.testing.assert_close(out_flat, reference_out, atol=atol, rtol=rtol)
        passed, mismatch_pct = True, 0.0
    except AssertionError as e:
        msg = str(e)
        m = re.search(r"Mismatched elements: \d+ / \d+ \(([\d.]+)%\)", msg)
        mismatch_pct = float(m.group(1)) if m else 100.0
        summary = " | ".join(
            ln.strip() for ln in msg.strip().splitlines() if ln.strip()
        )
        print(
            f"[WARN] {name} vs {reference_name}: max_atol={max_atol:.6f}, "
            f"avg_atol={avg_atol:.6f}, max_rtol={max_rtol:.6f}, "
            f"avg_rtol={avg_rtol:.6f}, {summary}"
        )
        passed = False
    return RefcheckResult(
        passed=passed,
        max_atol=max_atol,
        avg_atol=avg_atol,
        max_rtol=max_rtol,
        avg_rtol=avg_rtol,
        mismatch_pct=mismatch_pct,
    )


def run_refcheck(
    backend_names: list[str],
    run_fns: dict[str, Callable],
    inp_dicts: dict[str, dict],
    atol: float = 1e-2,
    rtol: float = 1e-2,
    print_detail: bool = False,
) -> dict[str, RefcheckResult]:
    """
    run_fns:   {backend_name: fn}  -- fn(inp_dict) -> Tensor
    inp_dicts: {backend_name: inp_dict}
    Returns {backend_name: RefcheckResult}.
    """
    reference_out: torch.Tensor | None = None
    reference_name: str | None = None
    results: dict[str, RefcheckResult] = {}

    for name in backend_names:
        out_flat = _run_and_flatten(name, run_fns[name], inp_dicts[name])
        if out_flat is None:
            results[name] = RefcheckResult(passed=False)
            continue

        if torch.isnan(out_flat).any() or torch.isinf(out_flat).any():
            print(f"[REFCHECK] {name}: output contains NaN/Inf -- fail")
            results[name] = RefcheckResult(passed=False)
            continue

        if reference_out is None:
            # Degenerate reference guard: if every element is within atol of
            # zero, the reference backend produced an all-zero / near-zero
            # tensor and any comparison against it is meaningless. Flag every
            # backend as failed with a sentinel mismatch of 200%.
            if out_flat.abs().max().item() < atol:
                print(
                    f"[REFCHECK] reference '{name}' output is degenerate "
                    f"(all |values| < atol={atol}); marking all backends failed"
                )
                results = {
                    n: RefcheckResult(passed=False, mismatch_pct=200.0)
                    for n in backend_names
                }
                break
            reference_out, reference_name = out_flat, name
            results[name] = RefcheckResult(passed=True)
            continue

        if out_flat.numel() != reference_out.numel():
            print(
                f"[WARN] {name} vs {reference_name}: shape mismatch, "
                f"numel {out_flat.numel()} vs {reference_out.numel()}"
            )
            results[name] = RefcheckResult(passed=False)
            continue

        results[name] = _compare_to_reference(
            name, reference_name, out_flat, reference_out, atol, rtol
        )

    if print_detail:
        print(f"refcheck results: {results}")
    return results
