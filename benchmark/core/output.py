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
import csv
import os
from dataclasses import dataclass
from core.metrics import BenchmarkMetrics


@dataclass
class AttnRowResult:
    """Single attention benchmark result row with scenario params and metrics."""

    scenario: str
    backend: str
    batch_size: int
    s_q: int
    s_kv: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    dtype: str
    causal: bool
    metrics: BenchmarkMetrics
    refcheck_pass: bool | None  # None = refcheck not performed
    case_tag: str = ""
    use_cuda_graph: bool = False
    status: str = "OK"
    mismatch_pct: float | None = None  # refcheck mismatch percentage


@dataclass
class GemmRowResult:
    """Single GEMM benchmark result row with matrix dimensions and metrics."""

    scenario: str
    backend: str
    m: int
    n: int
    k: int
    mode: str
    dtype: str
    metrics: BenchmarkMetrics
    refcheck_pass: bool | None  # None = refcheck not performed
    case_tag: str = ""
    # Grouped GEMM only: number of experts. None for dense GEMM.
    num_experts: int | None = None
    # Grouped GEMM only: per-expert m from config (shapes[*][1]). None for dense.
    max_m: int | None = None
    status: str = "OK"
    mismatch_pct: float | None = None  # refcheck mismatch percentage


_COMMON_FIELDS = [
    "median_us",
    "std_us",
    "speedup",
    "tflops",
    "tb_per_sec",
    "bw_pct",
    "arithmetic_intensity",
    "roofline_tflops",
    "attainable_pct",
    "host_us",
    "dispatch_us",
    "refcheck_pass",
    "mismatch_pct",
]
_CSV_FIELDS = [
    "scenario",
    "backend",
    "batch_size",
    "s_q",
    "s_kv",
    "num_q_heads",
    "num_kv_heads",
    "head_dim",
    "dtype",
    "causal",
    *_COMMON_FIELDS,
    "case_tag",
    "use_cuda_graph",
    "status",
]
_GEMM_CSV_FIELDS = [
    "scenario",
    "backend",
    "num_experts",
    "max_m",
    "m",
    "n",
    "k",
    "mode",
    *_COMMON_FIELDS,
    "case_tag",
    "status",
]


def attention_scenario_str(
    scenario: str,
    batch_size,
    s_q,
    s_kv,
    num_q_heads,
    num_kv_heads,
    head_dim,
    dtype,
    causal,
) -> str:
    """Format attention scenario descriptor for logs and report headers."""
    return (
        f"scenario: {scenario} | B={batch_size} | s_q={s_q} | s_kv={s_kv}"
        f" | H_q={num_q_heads} | H_kv={num_kv_heads} | D={head_dim}"
        f" | {dtype} | causal={causal}"
    )


def gemm_scenario_str(
    scenario: str,
    m,
    n,
    k,
    mode,
    num_experts=None,
    max_m=None,
) -> str:
    """Format GEMM scenario descriptor for logs and report headers.

    For grouped GEMM, both the per-expert config m (max_m) and the total
    computed m (m) are shown; dense GEMM omits num_experts/max_m.
    """
    experts = ""
    # Skip when None or NaN (NaN != NaN); grouped GEMM passes an int.
    if num_experts is not None and num_experts == num_experts:
        experts = f"num_experts={int(num_experts)} | "
    per_expert = ""
    if max_m is not None and max_m == max_m:
        per_expert = f"max_m={int(max_m)} | "
    return f"scenario: {scenario} | {experts}{per_expert}m={m} | n={n} | k={k} | mode={mode}"


def scenario_header_str(row: AttnRowResult | GemmRowResult) -> str:
    """Format scenario descriptor plus per-scenario arithmetic intensity."""
    if isinstance(row, GemmRowResult):
        s = gemm_scenario_str(
            row.scenario,
            row.m,
            row.n,
            row.k,
            row.mode,
            row.num_experts,
            row.max_m,
        )
    else:
        s = attention_scenario_str(
            row.scenario,
            row.batch_size,
            row.s_q,
            row.s_kv,
            row.num_q_heads,
            row.num_kv_heads,
            row.head_dim,
            row.dtype,
            row.causal,
        )
    return (
        f"{s}\n"
        f"arithmetic intensity: {row.metrics.arithmetic_intensity:.3f} FLOPs/byte"
        f"  roofline: {row.metrics.roofline_tflops:.2f} TFLOPS"
    )


def print_scenario_header(row: AttnRowResult | GemmRowResult) -> None:
    """Print scenario descriptor line, auto-dispatching by row type."""
    print(f"\n{scenario_header_str(row)}")


def table_header_str() -> str:
    """Return header + rule as a single string for the benchmark table."""
    _HEADER_FMT = (
        "{:<20} {:>10} {:>8} {:>8} {:>8} {:>8} {:>7} {:>12} {:>10} {:>10} {:>10} {:>8}"
    )
    h = _HEADER_FMT.format(
        "backend",
        "median(us)",
        "std(us)",
        "Speedup",
        "TFLOPS",
        "TB/s",
        "BW%",
        "roof_util%",
        "host(us)",
        "disp(us)",
        "mismatch%",
        "status",
    )
    return h + "\n" + "-" * len(h)


def print_table_header() -> None:
    """Print the benchmark table column header and separator line."""
    print(table_header_str())


def format_row_str(
    backend: str,
    median_us: float,
    std_us: float,
    speedup: float,
    tflops: float,
    tb_per_sec: float,
    bw_pct: float,
    attainable_pct: float,
    host_us: float,
    dispatch_us: float,
    refcheck_pass=None,
    status: str = "OK",
    mismatch_pct: float | None = None,
) -> str:
    """Format a single benchmark row as a fixed-width string."""
    tag = ""
    if refcheck_pass is False:
        tag = " [!]"
    if status != "OK":
        _FAIL_FMT = "{:<20} {:>10} {:>8} {:>8} {:>8} {:>8} {:>7} {:>12} {:>10} {:>10} {:>10} {:>8}"
        mismatch_str = f"{mismatch_pct:.1f}%" if mismatch_pct is not None else "N/A"
        return _FAIL_FMT.format(
            backend + tag,
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            mismatch_str,
            status,
        )
    _ROW_FMT = (
        "{:<20} {:>10.3f} {:>8.3f} {:>7.2f}x {:>8.2f} {:>8.3f} {:>6.1f}% "
        "{:>11.1f}% {:>10.3f} {:>10.3f} {:>10} {:>8}"
    )
    mismatch_str = f"{mismatch_pct:.1f}%" if mismatch_pct is not None else "N/A"
    return _ROW_FMT.format(
        backend + tag,
        median_us,
        std_us,
        speedup,
        tflops,
        tb_per_sec,
        bw_pct,
        attainable_pct,
        host_us,
        dispatch_us,
        mismatch_str,
        status,
    )


def print_table_row(row: AttnRowResult | GemmRowResult) -> None:
    """Print one benchmark result row; appends [!] if refcheck failed."""
    status = row.status if hasattr(row, "status") else "OK"
    mismatch_pct = row.mismatch_pct if hasattr(row, "mismatch_pct") else None
    m = row.metrics
    print(
        format_row_str(
            row.backend,
            m.median_us,
            m.std_us,
            m.speedup,
            m.tflops,
            m.tb_per_sec,
            m.bw_pct,
            m.attainable_pct,
            m.host_us,
            m.dispatch_us,
            refcheck_pass=row.refcheck_pass,
            status=status,
            mismatch_pct=mismatch_pct,
        )
    )


def append_attention_csv(row: AttnRowResult, output_path: str) -> None:
    """Append one attention benchmark result row to the CSV file."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    write_header = not os.path.exists(output_path)
    m = row.metrics
    with open(output_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "scenario": row.scenario,
                "backend": row.backend,
                "batch_size": row.batch_size,
                "s_q": row.s_q,
                "s_kv": row.s_kv,
                "num_q_heads": row.num_q_heads,
                "num_kv_heads": row.num_kv_heads,
                "head_dim": row.head_dim,
                "dtype": row.dtype,
                "causal": row.causal,
                "median_us": f"{m.median_us:.3f}",
                "std_us": f"{m.std_us:.3f}",
                "tflops": f"{m.tflops:.3f}",
                "tb_per_sec": f"{m.tb_per_sec:.4f}",
                "speedup": f"{m.speedup:.4f}",
                "bw_pct": f"{m.bw_pct:.2f}",
                "arithmetic_intensity": f"{m.arithmetic_intensity:.3f}",
                "roofline_tflops": f"{m.roofline_tflops:.3f}",
                "attainable_pct": f"{m.attainable_pct:.2f}",
                "host_us": f"{m.host_us:.3f}",
                "dispatch_us": f"{m.dispatch_us:.3f}",
                "refcheck_pass": row.refcheck_pass,
                "mismatch_pct": (
                    f"{row.mismatch_pct:.1f}" if row.mismatch_pct is not None else ""
                ),
                "case_tag": row.case_tag,
                "use_cuda_graph": row.use_cuda_graph,
                "status": row.status,
            }
        )


def append_gemm_csv(row: GemmRowResult, output_path: str) -> None:
    """Append one GEMM benchmark result row to the CSV file."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    write_header = not os.path.exists(output_path)
    m = row.metrics
    with open(output_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_GEMM_CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "scenario": row.scenario,
                "backend": row.backend,
                "num_experts": "" if row.num_experts is None else row.num_experts,
                "max_m": "" if row.max_m is None else row.max_m,
                "m": row.m,
                "n": row.n,
                "k": row.k,
                "mode": row.mode,
                "median_us": f"{m.median_us:.3f}",
                "std_us": f"{m.std_us:.3f}",
                "speedup": f"{m.speedup:.4f}",
                "tflops": f"{m.tflops:.3f}",
                "tb_per_sec": f"{m.tb_per_sec:.4f}",
                "bw_pct": f"{m.bw_pct:.2f}",
                "arithmetic_intensity": f"{m.arithmetic_intensity:.3f}",
                "roofline_tflops": f"{m.roofline_tflops:.3f}",
                "attainable_pct": f"{m.attainable_pct:.2f}",
                "host_us": f"{m.host_us:.3f}",
                "dispatch_us": f"{m.dispatch_us:.3f}",
                "refcheck_pass": row.refcheck_pass,
                "mismatch_pct": (
                    f"{row.mismatch_pct:.1f}"
                    if hasattr(row, "mismatch_pct") and row.mismatch_pct is not None
                    else ""
                ),
                "case_tag": row.case_tag,
                "status": row.status,
            }
        )
