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

HALOps CI benchmark tests.

Discovers all yaml configs under benchmark/ci_configs/, runs benchmark.py
for each, parses CSV results, and compares against baseline JSON files.
"""

import csv
import glob
import json
import os
import subprocess
import sys

import pytest

from constants import ATTENTION_KEY_COLS, GEMM_KEY_COLS, THRESHOLD


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def discover_configs(configs_dir):
    """Recursively discover all yaml config files under the given directory.

    Args:
        configs_dir (str): Root directory path for config files.

    Returns:
        list[str]: Sorted list of absolute paths to yaml/yml files.
    """
    yamls = []
    for root, _, files in os.walk(configs_dir):
        for f in sorted(files):
            if f.endswith((".yaml", ".yml")):
                yamls.append(os.path.join(root, f))
    return yamls


def get_op_from_config_path(config_path):
    """Extract operator type from config file path.

    Path format: .../ci_configs/<op>/<config>.yaml, extracts the <op> part.

    Args:
        config_path (str): Full path to the config file.

    Returns:
        str | None: Operator type string (e.g. "attention", "gemm"),
            or None if unrecognized.
    """
    parts = config_path.replace("\\", "/").split("/")
    for i, p in enumerate(parts):
        if p == "ci_configs" and i + 1 < len(parts):
            return parts[i + 1]
    return None


def get_config_name(config_path):
    """Get config file name without extension.

    Args:
        config_path (str): Full path to the config file.

    Returns:
        str: File name without .yaml/.yml suffix.
    """
    return os.path.splitext(os.path.basename(config_path))[0]


def parse_csv_results(csv_path, op):
    """Parse benchmark CSV result file, extracting key and median_us per row.

    Args:
        csv_path (str): Path to the CSV file.
        op (str): Operator type, "attention" or "gemm".

    Returns:
        list[tuple[dict, float]]: List of (key_dict, median_us) tuples.

    Raises:
        KeyError: If the CSV file is missing the "median_us" column.
        ValueError: If "median_us" value cannot be converted to float.
    """
    results = []
    key_cols = ATTENTION_KEY_COLS if op == "attention" else GEMM_KEY_COLS

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key_dict = {}
            for col in key_cols:
                val = row.get(col, "")
                # Preserve types for JSON comparison
                if val.lower() == "true":
                    val = True
                elif val.lower() == "false":
                    val = False
                else:
                    try:
                        val = int(val)
                    except (ValueError, TypeError):
                        try:
                            val = float(val)
                        except (ValueError, TypeError):
                            pass
                key_dict[col] = val

            median_us = float(row["median_us"])
            results.append((key_dict, median_us))
    return results


def load_baseline(baseline_dir, op, config_name):
    """Load baseline JSON file for a given config.

    Args:
        baseline_dir (str): Root path of baselines directory.
        op (str): Operator type, "attention" or "gemm".
        config_name (str): Config name (without extension).

    Returns:
        dict | None: Baseline data dict, or None if file does not exist.
    """
    baseline_path = os.path.join(baseline_dir, op, f"{config_name}.json")
    if not os.path.exists(baseline_path):
        return None
    with open(baseline_path, "r") as f:
        return json.load(f)


def key_to_str(key_dict):
    """Convert key dict to a human-readable string for log output.

    Args:
        key_dict (dict): Mapping from key column names to values.

    Returns:
        str: Formatted string like "k1=v1 | k2=v2 | ...".
    """
    parts = []
    for k, v in key_dict.items():
        parts.append(f"{k}={v}")
    return " | ".join(parts)


def find_baseline_entry(baseline_data, key_dict):
    """Find matching entry in baseline data by comparing key dicts.

    Args:
        baseline_data (dict): Baseline JSON data containing "entries" list.
        key_dict (dict): Key dict to search for.

    Returns:
        float | None: The median_us value of matched entry, or None if not found.
    """
    for entry in baseline_data.get("entries", []):
        if entry["key"] == key_dict:
            return entry["median_us"]
    return None


def find_csv_files(output_dir, config_name, backend=""):
    """Find CSV result files matching the given config name in output directory.

    Args:
        output_dir (str): Benchmark output root directory.
        config_name (str): Config name (without extension).
        backend (str): Backend name suffix. When non-empty, matches
            *_{config_name}_{backend}.csv; otherwise matches *_{config_name}.csv.

    Returns:
        list[str]: List of matching CSV file paths.
    """
    results_dir = os.path.join(output_dir, "results")
    if not os.path.isdir(results_dir):
        return []
    if backend:
        pattern = os.path.join(results_dir, f"*_{config_name}_{backend}.csv")
    else:
        pattern = os.path.join(results_dir, f"*_{config_name}.csv")
    return glob.glob(pattern)


# ---------------------------------------------------------------------------
# Test parametrization
# ---------------------------------------------------------------------------


def _get_all_configs():
    """Get all benchmark config file paths for pytest parametrization.

    Called during pytest collection phase. Auto-detects halops_root
    from current script location and locates benchmark/ci_configs/.

    Returns:
        list[str]: List of absolute paths to all yaml config files.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    halops_root = os.path.dirname(script_dir)
    configs_dir = os.path.join(halops_root, "benchmark", "ci_configs")
    if not os.path.isdir(configs_dir):
        return []
    return discover_configs(configs_dir)


ALL_CONFIGS = _get_all_configs()


@pytest.mark.parametrize(
    "config_path",
    ALL_CONFIGS,
    ids=[get_config_name(c) for c in ALL_CONFIGS],
)
def test_benchmark(config_path, benchmark_output_dir, gpu_id, baseline_dir, backend):
    """Run benchmark for a single config and compare against baseline.

    Args:
        config_path (str): Absolute path to the yaml config file.
        benchmark_output_dir (str): Benchmark output directory.
        gpu_id (str): GPU device ID.
        baseline_dir (str): Root path of baselines directory.
        backend (str): Backend name for benchmark (e.g. "halops").

    Raises:
        pytest.Failed: When benchmark execution fails, results are missing,
            or performance regression exceeds threshold.
    """
    config_name = get_config_name(config_path)
    op = get_op_from_config_path(config_path)
    assert op in ("attention", "gemm"), f"Unknown op type for config: {config_path}"

    # Resolve paths from script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    halops_root = os.path.dirname(script_dir)
    benchmark_py = os.path.join(halops_root, "benchmark", "benchmark.py")

    cmd = [
        sys.executable,
        "-u",
        benchmark_py,
        "--config",
        config_path,
        "--configs-dir",
        os.path.join(halops_root, "benchmark", "ci_configs"),
        "--output-dir",
        benchmark_output_dir,
        "--gpu",
        gpu_id,
        "--backend",
        backend,
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=os.path.join(halops_root, "benchmark"),
    )

    if result.returncode != 0:
        pytest.fail(
            f"benchmark.py failed for {config_name}:\n"
            f"stdout: {result.stdout[-2000:]}\n"
            f"stderr: {result.stderr[-2000:]}"
        )

    # Find and parse CSV results
    csv_files = find_csv_files(benchmark_output_dir, config_name, backend)
    if not csv_files:
        pytest.fail(
            f"No CSV result files found for {config_name} in {benchmark_output_dir}/results/"
        )

    all_results = []
    for csv_file in csv_files:
        all_results.extend(parse_csv_results(csv_file, op))

    if not all_results:
        pytest.fail(f"No results parsed from CSV for {config_name}")

    # Load baseline
    baseline_data = load_baseline(baseline_dir, op, config_name)
    baseline_path = os.path.join(baseline_dir, op, f"{config_name}.json")
    if baseline_data is None:
        pytest.fail(
            f"[无 BASELINE] 未找到 '{config_name}' 的 baseline 文件。\n"
            f"  请手动创建: {baseline_path}"
        )

    # Compare results against baseline
    threshold = baseline_data.get("threshold", THRESHOLD)
    failures = []
    for key_dict, current_median in all_results:
        baseline_median = find_baseline_entry(baseline_data, key_dict)
        if baseline_median is None:
            failures.append(
                f"  [缺失] {key_to_str(key_dict)}: baseline 中无对应条目\n"
                f"    → 请在 {baseline_path} 中添加该条目"
            )
            continue

        if baseline_median == 0:
            continue

        relative_error = abs(current_median - baseline_median) / baseline_median
        if relative_error > threshold:
            failures.append(
                f"  [REGRESSION] {key_to_str(key_dict)}:\n"
                f"    current={current_median:.3f}us, "
                f"baseline={baseline_median:.3f}us, "
                f"relative_error={relative_error*100:.2f}% (threshold={threshold*100:.1f}%)"
            )

    if failures:
        failure_msg = (
            f"检测到性能回归 ({config_name}):\n" + "\n".join(failures) + f"\n\n"
            f"  [提示] 如确认性能变化合理，请手动修改 baseline 文件:\n"
            f"    文件路径: {baseline_path}\n"
            f"    将对应条目的 'median_us' 值更新为当前测试结果。"
        )
        pytest.fail(failure_msg)
