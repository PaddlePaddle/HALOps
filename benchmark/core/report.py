#!/usr/bin/env python3
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
import argparse
import os
import sys
import pandas as pd

from core.output import (
    attention_scenario_str,
    gemm_scenario_str,
    table_header_str,
    format_row_str,
)

MISMATCH_THRESHOLD_PCT = 10.0


def _is_gemm_scenario(scenario):
    _GEMM_SCENARIOS = ("dense_gemm", "group_masked", "group_contiguous")
    return True if scenario in _GEMM_SCENARIOS else False


def _format_scenario_str(scenario: str, row: dict) -> str:
    if _is_gemm_scenario(scenario):
        return gemm_scenario_str(
            scenario,
            row["m"],
            row["n"],
            row["k"],
            row["mode"],
            row.get("num_experts"),
            row.get("max_m"),
        )
    else:
        return attention_scenario_str(
            scenario,
            row["batch_size"],
            row["s_q"],
            row["s_kv"],
            row["num_q_heads"],
            row["num_kv_heads"],
            row["head_dim"],
            row["dtype"],
            row["causal"],
        )


def _format_scenario_header(scenario_str: str, row: dict) -> str:
    arithmetic_intensity = float(row.get("arithmetic_intensity", 0.0) or 0.0)
    roofline_tflops = float(row.get("roofline_tflops", 0.0) or 0.0)
    return (
        f"{scenario_str}\n"
        f"arithmetic intensity: {arithmetic_intensity:.3f} FLOPs/byte"
        f"  roofline: {roofline_tflops:.2f} TFLOPS"
    )


def _format_table_row(rec: dict) -> str:
    refcheck_pass = rec.get("refcheck_pass")
    # Normalize string "True"/"False" from CSV to bool
    if isinstance(refcheck_pass, str):
        refcheck_pass = refcheck_pass.lower() == "true"
    elif refcheck_pass is not None and pd.notna(refcheck_pass):
        refcheck_pass = bool(refcheck_pass)
    else:
        refcheck_pass = None
    status = str(rec.get("status", "OK") or "OK")
    mp = rec.get("mismatch_pct")
    mismatch_pct = float(mp) if mp is not None and mp != "" and pd.notna(mp) else None
    return format_row_str(
        rec["backend"],
        float(rec["median_us"]),
        float(rec["std_us"]),
        float(rec.get("speedup", 0.0) or 0.0),
        float(rec["tflops"]),
        float(rec["tb_per_sec"]),
        float(rec["bw_pct"]),
        float(rec.get("attainable_pct", 0.0) or 0.0),
        float(rec.get("host_us", 0.0) or 0.0),
        float(rec.get("dispatch_us", 0.0) or 0.0),
        refcheck_pass=refcheck_pass,
        status=status,
        mismatch_pct=mismatch_pct,
    )


def _extract_log_header(log_path: str) -> str:
    """Extract the runtime-environment block at the top of the log
    (content between the two ===== lines, including the lines themselves)."""
    if not log_path or not os.path.exists(log_path):
        return ""
    lines: list[str] = []
    seen_first = False
    with open(log_path) as f:
        for line in f:
            stripped = line.rstrip("\n")
            if stripped.startswith("===="):
                lines.append(stripped)
                if seen_first:
                    break
                seen_first = True
                continue
            if seen_first:
                lines.append(stripped)
    return "\n".join(lines)


def _format_common_config(row: dict) -> str:
    """Compact per-config descriptor (without batch_size/s_kv) for the speedup summary."""
    if _is_gemm_scenario(row["scenario"]):
        ne = row.get("num_experts")
        mm = row.get("max_m")
        parts = []
        if ne is not None and pd.notna(ne) and str(ne) != "":
            parts.append(f"num_experts={int(ne)}")
        if mm is not None and pd.notna(mm) and str(mm) != "":
            parts.append(f"max_m={int(mm)}")
        parts.append(f"mode={row['mode']}")
        return ", ".join(parts)
    else:
        return (
            f"H_q={row['num_q_heads']}, H_kv={row['num_kv_heads']}, D={row['head_dim']}, "
            f"{row['dtype']}, causal={row['causal']}"
        )


def _speedup_summary(df: pd.DataFrame, sorted_keys: list[str]) -> list[str]:
    """Markdown table summarizing halops metrics and speedup across all configs."""
    if "speedup" not in df.columns:
        return []

    # Non-halops backends, preserve first-seen order.
    all_backends: list[str] = list(dict.fromkeys(df["backend"].tolist()))
    other_backends = [b for b in all_backends if b != "halops"]

    lines = ["## Speedup Summary", ""]
    lines.append(
        "Speedup = other_median_us / halops_median_us (>1 means halops faster)."
    )
    lines.append("")
    columns = [
        "config",
        *sorted_keys,
        "halops median(us)",
        "halops TFLOPS",
        "halops BW%",
        "halops host(us)",
        *other_backends,
    ]
    header = "|" + " | ".join(columns) + " |"
    rule = "|" + "---|" * len(columns)
    lines += [header, rule]

    grouped = df.groupby(sorted_keys, sort=True)
    for sorted_values, sub in grouped:
        config_str = _format_common_config(sub.iloc[0].to_dict())
        rows_by_backend = {r["backend"]: r for _, r in sub.iterrows()}

        # halops metrics
        ir = rows_by_backend.get("halops")
        if ir is not None:
            ir_median = f"{float(ir['median_us']):.3f}"
            ir_tflops = f"{float(ir['tflops']):.2f}"
            ir_bw = f"{float(ir['bw_pct']):.1f}%"
            ir_host = f"{float(ir.get('host_us', 0.0) or 0.0):.3f}"
        else:
            ir_median = ir_tflops = ir_bw = ir_host = "-"

        cells = [
            config_str,
            *[str(v) for v in sorted_values],
            ir_median,
            ir_tflops,
            ir_bw,
            ir_host,
        ]

        # speedup for other backends (with refcheck indicator)
        for bk in other_backends:
            r = rows_by_backend.get(bk)
            if r is not None:
                status = str(r.get("status", "OK") or "OK")
                speedup = r.get("speedup")
                if status != "OK":
                    cell = status
                elif speedup is not None and pd.notna(speedup):
                    cell = f"{float(speedup):.2f}x"
                else:
                    cell = "-"
                refcheck_pass = r.get("refcheck_pass")
                if (
                    refcheck_pass is not None
                    and pd.notna(refcheck_pass)
                    and not refcheck_pass
                ):
                    cell += " [!]"
                cells.append(cell)
            else:
                cells.append("-")

        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    return lines


def _to_float(value, default=None):
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _format_speedup(value: float) -> str:
    return "nan" if value != value else f"{value:.2f}x"


def _format_count_pct(count: int, total: int) -> str:
    pct = (count / total * 100.0) if total else 0.0
    return f"{count}({pct:.0f}%)"


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = (len(ordered) - 1) * p
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    if low == high:
        return ordered[low]
    weight = index - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _is_correct_row(row: pd.Series) -> bool:
    status = str(row.get("status", "OK") or "OK")
    if status != "OK":
        return False
    if _is_mismatch_row(row):
        return False
    median_us = _to_float(row.get("median_us"), None)
    return median_us is not None and median_us > 0.0


def _to_bool(value) -> bool | None:
    if value is None:
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "1", "yes"):
            return True
        if lowered in ("false", "0", "no"):
            return False
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return bool(value)


def _is_mismatch_row(row: pd.Series) -> bool:
    mismatch_pct = _to_float(row.get("mismatch_pct"), None)
    if mismatch_pct is not None:
        return mismatch_pct > MISMATCH_THRESHOLD_PCT
    return _to_bool(row.get("refcheck_pass")) is False


def _format_issue_summary(item: dict, total: int) -> str:
    parts = []
    for key, label in (
        ("mismatch", "mismatch"),
        ("timeout", "timeout"),
        ("oom", "oom"),
        ("failed", "failed"),
    ):
        count = item[key]
        if count:
            parts.append(f"{label}:{_format_count_pct(count, total)}")
    return ", ".join(parts) if parts else "-"


def _backend_summary(df: pd.DataFrame, sorted_keys: list[str]) -> list[str]:
    """Summarize speedup distribution by backend for this generated report."""
    if "backend" not in df.columns or "speedup" not in df.columns:
        return []

    all_backends: list[str] = list(dict.fromkeys(df["backend"].tolist()))
    stats = {
        backend: {
            "total": 0,
            "values": [],
            "lt1": 0,
            "ge1_lt2": 0,
            "ge2_lt3": 0,
            "ge3_le5": 0,
            "gt5": 0,
            "mismatch": 0,
            "timeout": 0,
            "oom": 0,
            "failed": 0,
        }
        for backend in all_backends
    }

    for _, sub in df.groupby(sorted_keys, sort=False):
        rows_by_backend = {r["backend"]: r for _, r in sub.iterrows()}
        halops = rows_by_backend.get("halops")
        halops_correct = _is_correct_row(halops) if halops is not None else False

        for backend in all_backends:
            row = rows_by_backend.get(backend)
            if row is None:
                continue

            stats[backend]["total"] += 1
            status = str(row.get("status", "OK") or "OK").upper()
            if status == "TIMEOUT":
                stats[backend]["timeout"] += 1
            elif status == "OOM":
                stats[backend]["oom"] += 1
            elif status != "OK":
                stats[backend]["failed"] += 1
            elif _is_mismatch_row(row):
                stats[backend]["mismatch"] += 1

            row_correct = _is_correct_row(row)
            if backend != "halops":
                row_correct = row_correct and halops_correct
            if not row_correct:
                continue

            speedup = (
                1.0 if backend == "halops" else _to_float(row.get("speedup"), None)
            )
            if speedup is None:
                continue

            stats[backend]["values"].append(speedup)
            if speedup < 1.0:
                stats[backend]["lt1"] += 1
            elif speedup < 2.0:
                stats[backend]["ge1_lt2"] += 1
            elif speedup < 3.0:
                stats[backend]["ge2_lt3"] += 1
            elif speedup <= 5.0:
                stats[backend]["ge3_le5"] += 1
            else:
                stats[backend]["gt5"] += 1

    lines = ["## Summary", ""]
    lines.append(
        "Speedup = backend_median_us / halops_median_us (>1 means halops faster). "
        f"Only OK samples with mismatch <= {MISMATCH_THRESHOLD_PCT:.0f}% are included in statistics."
    )
    lines.append("")
    columns = [
        "backend",
        "total",
        "valid",
        "mean",
        "P25/P50/P75/P90",
        "distribution",
        "issues",
    ]
    lines.append("|" + " | ".join(columns) + " |")
    lines.append("|---|---:|---:|---:|---|---|---|")

    for backend in all_backends:
        item = stats[backend]
        values = item["values"]
        total = item["total"]
        valid = len(values)
        mean = sum(values) / valid if values else float("nan")
        p25 = _percentile(values, 0.25)
        p50 = _percentile(values, 0.50)
        p75 = _percentile(values, 0.75)
        p90 = _percentile(values, 0.90)
        distribution = (
            f"<1:{item['lt1']}, 1-2:{item['ge1_lt2']}, "
            f"2-3:{item['ge2_lt3']}, 3-5:{item['ge3_le5']}, >5:{item['gt5']}"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    str(backend),
                    str(total),
                    _format_count_pct(valid, total),
                    _format_speedup(mean),
                    f"{_format_speedup(p25)}/{_format_speedup(p50)}/"
                    f"{_format_speedup(p75)}/{_format_speedup(p90)}",
                    distribution,
                    _format_issue_summary(item, total),
                ]
            )
            + " |"
        )

    lines.append("")
    return lines


def generate_report(
    df: pd.DataFrame,
    out_dir: str,
    tag: str,
    scenario: str,
    ts: str = "",
    log_header: str = "",
    name_suffix: str = "",
) -> str:
    lines: list[str] = [f"# {scenario} -- {tag}", ""]

    if log_header:
        lines += ["```", log_header, "```", ""]

    # Group within a group keep the original backend order.
    if _is_gemm_scenario(scenario):
        sorted_keys = ["m", "n", "k"]
    else:
        sorted_keys = ["batch_size", "s_kv", "s_q"]

    df = df.sort_values(sorted_keys, kind="stable")
    for _, sub in df.groupby(sorted_keys, sort=False):
        first_row = sub.iloc[0].to_dict()
        scenario_str = _format_scenario_header(
            _format_scenario_str(scenario, first_row), first_row
        )
        block = ["```", scenario_str, table_header_str()]
        for _, rec in sub.iterrows():
            block.append(_format_table_row(rec.to_dict()))
        block.append("```")
        lines.extend(block)
        lines.append("")

    # Summary tables after the detail tables. Keep these as the final report
    # section; per-row mismatch/status are already shown in the detail tables.
    lines += _speedup_summary(df, sorted_keys)
    lines += _backend_summary(df, sorted_keys)

    content = "\n".join(lines) + "\n"
    prefix = f"{ts}_" if ts else ""
    suffix = f"_{name_suffix}" if name_suffix else ""
    fname = f"{prefix}{scenario}_{tag}{suffix}.md".replace(" ", "_").replace("/", "-")
    path = os.path.join(out_dir, fname)
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    return path


def generate_reports_from_csv(
    csv_path: str, out_dir: str, ts: str, log_path: str = "", name_suffix: str = ""
) -> None:
    """Called after a benchmark run to produce md reports from the given CSV."""
    if not os.path.exists(csv_path):
        return
    df = pd.read_csv(csv_path)
    log_header = _extract_log_header(log_path)

    _required = {"scenario", "case_tag"}
    if not _required.issubset(df.columns):
        print(f"[report] skipped (unknown schema): {csv_path}")
        return
    for (scenario, tag), group in df.groupby(["scenario", "case_tag"]):
        path = generate_report(
            group,
            out_dir,
            str(tag),
            str(scenario),
            ts=ts,
            log_header=log_header,
            name_suffix=name_suffix,
        )
        print(f"[report] {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Markdown report from benchmark CSV"
    )
    parser.add_argument("csv", nargs="+", help="CSV file(s) to process")
    parser.add_argument(
        "--out", default="reports", help="Output directory (default: reports)"
    )
    parser.add_argument(
        "--log", default="", help="Optional log file to embed as header"
    )
    args = parser.parse_args()

    dfs = [pd.read_csv(p) for p in args.csv]
    df_all = pd.concat(dfs, ignore_index=True)
    log_header = _extract_log_header(args.log)

    generated = []
    for (scenario, tag), group in df_all.groupby(["scenario", "case_tag"]):
        path = generate_report(
            group, args.out, str(tag), str(scenario), log_header=log_header
        )
        generated.append(path)
        print(f"[report] {path}")
    print(f"\nGenerated {len(generated)} report(s) -> {args.out}/")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(__file__))
    main()
