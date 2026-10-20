# HALOps CI Performance Regression Tests

This directory contains the orchestration scripts and baseline data for HALOps
performance regression testing. It uses pytest to invoke `benchmark.py`, parses
CSV results, and compares them against stored baselines. Tests fail when the
relative error exceeds the configured threshold.

## Directory Structure

```
ci/
├── conftest.py          # pytest config, registers CLI options and fixtures
├── constants.py         # shared constants (key columns, threshold)
├── test_benchmark.py    # main test file, runs benchmark and compares results
├── update_baseline.py   # standalone script to update baselines from CSV
├── baselines/           # baseline JSON data
│   ├── attention/       # attention operator baselines
│   └── gemm/            # gemm operator baselines
├── DockerFile.gpu       # CI Docker image definition
└── README.md            # this file
```

## Usage

### 1. Run Performance Regression Tests

Run directly from the HALOps project root:

```bash
cd /workspace/HALOps
pytest ci/test_benchmark.py -v
```

Options:

```bash
# Specify GPU device
pytest ci/test_benchmark.py -v --gpu 1

# Specify backend (default: halops)
pytest ci/test_benchmark.py -v --backend halops

# Specify benchmark output directory (default: auto-generated temp dir)
pytest ci/test_benchmark.py -v --benchmark-output-dir /path/to/output

# Run only a specific config
pytest ci/test_benchmark.py -v -k "decode_c16_gqa56-4_batch1"
```

### 2. Update Baselines

When baselines need to be refreshed (e.g. first run, hardware change, kernel
optimization), verify the results are correct, then run:

```bash
python ci/update_baseline.py \
    --results-dir <benchmark_output>/results/ \
    --baseline-dir ci/baselines/
```

`update_baseline.py` preserves any manually-set `threshold` values in existing JSONs.

### 3. Customize Threshold

Each baseline JSON file has a `threshold` field controlling the failure threshold:

```json
{
  "config": "decode_c16_gqa56-4_batch1",
  "op": "attention",
  "threshold": 0.05,
  "entries": [...]
}
```

- Default: `0.05` (5% relative error)
- Edit `threshold` in the corresponding JSON to customize per-config tolerance
- Formula: `abs(current - baseline) / baseline > threshold` triggers FAIL

## Test Behavior

| Scenario | Behavior |
|----------|----------|
| No baseline file | FAIL, log hints to create baseline |
| Within threshold | PASS |
| Exceeds threshold | FAIL, reports current value, baseline value, and error % |
| benchmark.py crashes | FAIL, outputs stdout/stderr |

## Test Coverage

Auto-discovers all yaml files under `benchmark/ci_configs/`, each yaml becomes
an independent pytest test case:

- `attention/` — decode/prefill scenarios, various batch_size, dtype, context lengths
- `gemm/` — dense/group_contiguous/group_masked scenarios, w4a4/w8a8 modes

## Baseline Environment

The baseline data stored in `baselines/` was collected under the following environment:

- **GPU**: NVIDIA B200
- **CUDA**: 13.0
- **OS**: Ubuntu 24.04
- **Python**: 3.12
- **PyTorch**: 2.13.0

When running on a different hardware/software environment, baselines should be
regenerated using `update_baseline.py` to reflect the actual performance
characteristics of the target platform.

## Path Resolution

The test script auto-detects the project root from its own location:

```
ci/test_benchmark.py  →  halops_root = parent directory (HALOps/)
                      →  benchmark.py   = HALOps/benchmark/benchmark.py
                      →  ci_configs     = HALOps/benchmark/ci_configs/
                      →  baselines      = HALOps/ci/baselines/
```

No need to copy files elsewhere — run pytest directly from the project root.
