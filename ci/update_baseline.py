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

Update baseline JSON files from benchmark CSV results.

Usage:
    python update_baseline.py \\
        --results-dir <benchmark_output>/results/ \\
        --baseline-dir ci/baselines/
"""

import argparse
import csv
import glob
import json
import os
import re
import sys

from constants import ATTENTION_KEY_COLS, GEMM_KEY_COLS


def detect_op_from_csv(csv_path, fieldnames):
    """Detect operator type from CSV column names.

    Args:
        csv_path (str): CSV file path (for logging purposes).
        fieldnames (list[str]): List of CSV column names.

    Returns:
        str | None: "attention" or "gemm", or None if unrecognized.
    """
    if "s_kv" in fieldnames and "num_q_heads" in fieldnames:
        return "attention"
    elif "num_experts" in fieldnames or "mode" in fieldnames:
        return "gemm"
    return None


def extract_config_name(csv_filename):
    """Extract config name from CSV filename.

    Filename format: 'YYYYMMDD_HHMMSS_<config_name>.csv', strips timestamp prefix.

    Args:
        csv_filename (str): CSV filename (without directory).

    Returns:
        str: Config name (without timestamp prefix and extension).
    """
    name = os.path.splitext(csv_filename)[0]
    match = re.match(r"^\d{8}_\d{6}_(.+)$", name)
    if match:
        return match.group(1)
    return name


def parse_value(val):
    """Parse string value to the appropriate Python type.

    Tries bool, int, float in order; falls back to original string.

    Args:
        val (str): String value to parse.

    Returns:
        bool | int | float | str: Parsed value.
    """
    if val.lower() == "true":
        return True
    elif val.lower() == "false":
        return False
    try:
        return int(val)
    except (ValueError, TypeError):
        try:
            return float(val)
        except (ValueError, TypeError):
            return val


def process_csv(csv_path):
    """Process a single CSV file, extracting config name, op type and entries.

    Args:
        csv_path (str): Absolute path to the CSV file.

    Returns:
        tuple[str | None, str | None, list[dict]]: (config_name, op, entries).
            Returns (None, None, []) when op type cannot be detected.

    Raises:
        KeyError: If the CSV file is missing the "median_us" column.
    """
    csv_filename = os.path.basename(csv_path)
    config_name = extract_config_name(csv_filename)

    entries = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        op = detect_op_from_csv(csv_path, fieldnames)
        if op is None:
            print(f"  WARNING: Cannot detect op type for {csv_filename}, skipping.")
            return None, None, []

        key_cols = ATTENTION_KEY_COLS if op == "attention" else GEMM_KEY_COLS

        for row in reader:
            key_dict = {}
            for col in key_cols:
                key_dict[col] = parse_value(row.get(col, ""))
            median_us = float(row["median_us"])
            entries.append({"key": key_dict, "median_us": median_us})

    return config_name, op, entries


def main():
    """Batch update baseline JSON files from benchmark CSV results.

    Parses --results-dir and --baseline-dir CLI arguments, scans CSV files
    in the results directory, and generates/updates corresponding baseline JSONs.
    Existing manually-set threshold values are preserved.

    Raises:
        SystemExit: When results-dir does not exist or contains no CSV files.
    """
    parser = argparse.ArgumentParser(description="Update HALOps benchmark baselines")
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Path to benchmark results directory containing CSV files",
    )
    parser.add_argument(
        "--baseline-dir",
        required=True,
        help="Path to baselines directory to write JSON files",
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    baseline_dir = args.baseline_dir

    if not os.path.isdir(results_dir):
        print(f"ERROR: Results directory not found: {results_dir}")
        sys.exit(1)

    csv_files = glob.glob(os.path.join(results_dir, "*.csv"))
    if not csv_files:
        print(f"ERROR: No CSV files found in {results_dir}")
        sys.exit(1)

    print(f"Found {len(csv_files)} CSV file(s) in {results_dir}")
    updated = []

    for csv_path in sorted(csv_files):
        config_name, op, entries = process_csv(csv_path)
        if config_name is None or not entries:
            continue

        # Ensure output directory exists
        op_dir = os.path.join(baseline_dir, op)
        os.makedirs(op_dir, exist_ok=True)

        # Preserve existing threshold if baseline already exists
        output_path = os.path.join(op_dir, f"{config_name}.json")
        threshold = 0.05
        if os.path.exists(output_path):
            with open(output_path, "r") as f:
                existing = json.load(f)
                threshold = existing.get("threshold", 0.05)

        baseline_data = {
            "config": config_name,
            "op": op,
            "threshold": threshold,
            "entries": entries,
        }

        with open(output_path, "w") as f:
            json.dump(baseline_data, f, indent=2)

        updated.append(f"  {op}/{config_name}.json ({len(entries)} entries)")
        print(f"  Updated: {output_path}")

    print(f"\nBaseline update complete. {len(updated)} file(s) updated:")
    for item in updated:
        print(item)


if __name__ == "__main__":
    main()
