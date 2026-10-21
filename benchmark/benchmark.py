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
import sys
import os
import datetime
import logging
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

import yaml  # noqa: E402

from core.log_util import tee_output  # noqa: E402
from core.runtime_info import (  # noqa: E402
    get_gpu_name,
    get_driver_version,
    print_backend_versions,
)
from kernel.attention.scenarios import SCENARIOS as _attn_scenarios  # noqa: E402
from kernel.gemm.scenarios import SCENARIOS as _gemm_scenarios  # noqa: E402

OP_REGISTRY: dict[str, dict] = {
    "attention": _attn_scenarios,
    "gemm": _gemm_scenarios,
}


def get_benchmark_root():
    """Return the benchmark package root used for default config/output paths."""
    return Path(__file__).parent


def resolve_config(path: str, configs_root: str | Path) -> tuple[Path, str]:
    """Resolve a user-supplied config to (absolute_path, op).

    Accepts:
      - absolute config path
      - '<op>/<file>.yaml' (resolved under configs_root)
      - bare '<file>.yaml' (uniquely matched across configs_root/<op>/)
    """
    p = Path(path).expanduser()
    configs_root = Path(configs_root).expanduser().resolve()
    # Bare filename: search across op directories for a unique match.
    if len(p.parts) == 1:
        hits = [op for op in OP_REGISTRY if (configs_root / op / p).is_file()]
        if len(hits) != 1:
            ops = list(OP_REGISTRY)
            reason = "not found" if not hits else f"ambiguous (found in {hits})"
            raise SystemExit(
                f"[ERROR] config '{path}' {reason} under "
                f"'{configs_root}/{{{','.join(ops)}}}/'. "
                f"Prefix with op, e.g. '{ops[0]}/{path}'."
            )
        op = hits[0]
        return configs_root / op / p, op

    abs_path = (p if p.is_absolute() else configs_root / p).resolve()
    op = abs_path.parent.name
    if op not in OP_REGISTRY:
        raise SystemExit(
            f"[ERROR] cannot infer op from '{path}'; parent dir '{op}' is not a known op. "
            f"available: {list(OP_REGISTRY)}"
        )
    return abs_path, op


def discover_configs(configs_root: str | Path) -> list[Path]:
    """Return all benchmark configs under known op directories."""
    configs_root = Path(configs_root).expanduser().resolve()
    configs = sorted(
        path
        for path in configs_root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".yaml", ".yml"}
        and path.parent.name in OP_REGISTRY
    )
    if not configs:
        raise SystemExit(
            f"[ERROR] no .yaml or .yml configs found under '{configs_root}'."
        )
    return configs


def _run_scenarios(op, cfg):
    """Run every batch size of a config; return False if any raised."""
    try:
        SCENARIOS = OP_REGISTRY[op]
        scenario = cfg.get("scenario")
        for batch_size in cfg.get("batch_sizes", [cfg.get("batch_size", 1)]):
            SCENARIOS[scenario]({**cfg, "batch_size": batch_size})
            print()
    except Exception as e:
        print(f"\n[FATAL] {e}", flush=True)
        import traceback

        traceback.print_exc()
        return False
    return True


def main(args):
    """Entry point: parse config files, set up environment, and run benchmarks."""
    configs_dir = Path(args.configs_dir).expanduser().resolve()
    raw_configs = args.config or discover_configs(configs_dir)

    if args.backend is not None and len(raw_configs) > 1:
        raise SystemExit(
            "[ERROR] --backend can only be used with a single --config file."
        )

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir).expanduser().resolve()
    results_dir = output_dir / "results"
    reports_dir = output_dir / "reports"
    logs_dir = output_dir / "logs"

    failed_configs = []
    for raw_config in raw_configs:
        config_path, op = resolve_config(str(raw_config), configs_dir)
        config_path = str(config_path)
        SCENARIOS = OP_REGISTRY[op]
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        scenario = cfg.get("scenario")
        if scenario not in SCENARIOS:
            print(
                f"[ERROR] unknown scenario '{scenario}' for op '{op}'. available: {list(SCENARIOS)}"
            )
            sys.exit(1)

        # Benchmark tuning parameters come exclusively from the CLI (single
        # source of truth); overwrite any stale values that may exist in YAML.
        cfg["timing_method"] = args.timing_method
        cfg["dry_run_iters"] = args.dry_run_iters
        cfg["repeat_iters"] = args.repeat_iters
        cfg["use_subprocess"] = not args.no_subprocess
        if args.refcheck_reference is not None:
            cfg["refcheck_reference"] = args.refcheck_reference

        if args.backend is not None:
            cfg_backends = list(cfg.get("backends", []))
            if args.backend not in cfg_backends:
                raise SystemExit(
                    f"[ERROR] --backend='{args.backend}' is not in the YAML "
                    f"'backends' list {cfg_backends} for config '{config_path}'."
                )
            # Benchmark only the requested backend, but keep refcheck on: the
            # scenario adds the neutral golden reference for comparison, so a
            # single-backend run is still validated against golden.
            cfg["backends"] = [args.backend]

        config_name = os.path.splitext(os.path.basename(config_path))[0]
        # case_tag always equals the config filename; any value in the YAML is
        # ignored so it never needs to be hand-maintained.
        cfg["case_tag"] = config_name
        # CSV and log are timestamped so each run produces fresh files. The
        # output_path is always generated here; any value in the YAML is ignored.
        suffix = f"_{args.backend}" if args.backend is not None else ""
        output_path = str(results_dir / f"{ts}_{config_name}{suffix}.csv")
        cfg["output_path"] = output_path
        log_path = str(logs_dir / f"{ts}_{op}_{config_name}{suffix}.log")

        with tee_output(
            log_path,
            intercept_native_stderr=args.log_interception,
        ) as tee:
            logging.basicConfig(
                stream=tee,
                level=logging.WARNING,
                force=True,
                format="[%(levelname)s] %(name)s: %(message)s",
            )

            # Print runtime environment & backend version info.
            print("=" * 60)
            print(
                f"Date       : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            print(f"Config     : {config_path}")
            print(f"GPU        : {get_gpu_name()}")
            print(f"Driver     : {get_driver_version()}")
            print(f"cuda_graph : {cfg.get('use_cuda_graph', False)}")
            print_backend_versions(cfg.get("backends", []))
            print("=" * 60)

            if not _run_scenarios(op, cfg):
                failed_configs.append(config_name)

        from core.report import generate_reports_from_csv

        generate_reports_from_csv(
            output_path,
            out_dir=str(reports_dir),
            ts=ts,
            log_path=log_path,
            name_suffix=args.backend or "",
        )

    if failed_configs:
        print(f"[FATAL] failed configs: {failed_configs}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kernel Benchmark")
    parser.add_argument(
        "--config",
        nargs="+",
        help="YAML config file(s). When omitted, run every .yaml/.yml file under "
        "<configs-dir>/<op>/. Otherwise, op is inferred from the parent directory, "
        "e.g. 'attention/decode_c16_gqa14.yaml' or just 'decode_c16_gqa14.yaml' "
        "(searched across <configs-dir>/<op>/)",
    )
    parser.add_argument("--gpu", default="0", help="CUDA device id(s), e.g. --gpu=0")
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        default=str(get_benchmark_root()),
        help="Root directory for reports/, results/, and logs/.",
    )
    parser.add_argument(
        "--configs-dir",
        "--configs_dir",
        dest="configs_dir",
        default=str(get_benchmark_root() / "configs"),
        help="Root directory containing <op>/*.yaml or *.yml benchmark configs.",
    )
    parser.add_argument(
        "--backend",
        default=None,
        help="Run only the specified backend (e.g. --backend=fa2). Must be present "
        "in the YAML's 'backends' list. When omitted, all backends in the YAML "
        "are run (default behavior). Requires exactly one --config.",
    )
    parser.add_argument(
        "--timing-method",
        choices=["cupti", "profiler"],
        default="profiler",
        help="GPU timing backend: 'cupti' (CUPTI hardware clock, supports CUDA "
        "graph) or 'profiler' (torch.cuda.Event + torch.profiler). Default: profiler.",
    )
    parser.add_argument(
        "--dry-run-iters",
        type=int,
        default=10,
        help="Warmup iterations before measurement. Default: 10.",
    )
    parser.add_argument(
        "--repeat-iters",
        type=int,
        default=50,
        help="Measured iterations. Default: 50.",
    )
    parser.add_argument(
        "--no-subprocess",
        action="store_true",
        default=False,
        help="Disable subprocess isolation (run benchmarks in-process). Default: use subprocess.",
    )
    parser.add_argument(
        "--log-interception",
        action="store_true",
        default=False,
        help="Intercept native C/CUDA stderr into the log. Default: disabled.",
    )
    parser.add_argument(
        "--refcheck-reference",
        default=None,
        help="Override YAML refcheck_reference, e.g. --refcheck-reference=golden.",
    )
    args = parser.parse_args()
    main(args)
