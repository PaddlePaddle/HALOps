English | [简体中文](build_guide_CN.md)

# HALOps Build Guide

HALOps is currently distributed as source only. This guide covers environment setup, wheel and development builds, installation checks and common options. The CUDA/C++ extension is built by [`setup.py`](../setup.py) through PyTorch `CUDAExtension`.

[Back to README](../README_EN.md) | [Benchmark Guide](../benchmark/README.md#english-quick-reference)

## Quick Install

Prepare a CUDA build of PyTorch 2.13.0+ compatible with CUDA 13.0+, then run from the repository root:

```bash
git submodule update --init --recursive
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation --no-deps -v .
```

HALOps has no PyPI package or prebuilt wheel yet; see "Install from source" for what the options do.

## 1. Requirements

| Component | Requirement |
| --- | --- |
| Python | 3.10 or later |
| CUDA Toolkit | 13.0 or later; `nvcc` must support `compute_100f/sm_100f` |
| NVIDIA GPU | Blackwell GPU with compute capability compatible with `sm_100f` |
| PyTorch | 2.13.0 or later |
| C++ compiler | C++20 support |

## 2. Prepare the Build Environment

Run all commands below from the HALOps repository root.

### 2.1 Initialize CUTLASS

```bash
git submodule update --init --recursive
```

If `3rdparty/cutlass/include` is missing, `setup.py` performs a shallow initialization of `3rdparty/cutlass` automatically.

### 2.2 Install Python Dependencies

Activate a virtual or Conda environment, then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

PyTorch must be a CUDA build compatible with the local Toolkit.

### 2.3 Configure the CUDA Path

If CUDA is not installed in the default location, or several CUDA versions are installed, select the Toolkit explicitly:

```bash
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

`CUDA_HOME` should point to the CUDA Toolkit root that contains `bin/nvcc`, `include/` and `lib64/`.

### 2.4 Optional Checks

```bash
nvcc --version
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

Make sure `nvcc` is CUDA 13.0+ and PyTorch is built with a compatible CUDA version.

## 3. Build and Install

### 3.1 Install from Source

Build and install into the current environment:

```bash
python -m pip install --no-build-isolation --no-deps -v .
```

`--no-build-isolation` reuses the current PyTorch/CUDA toolchain, `--no-deps` leaves installed dependencies untouched, and `-v` prints the full compiler log.

### 3.2 Build and Install a Wheel

Use a wheel to validate the final artifact:

```bash
rm -rf build dist
python -m pip wheel --no-build-isolation --no-deps -v -w dist .
python -m pip install dist/halops-*.whl --force-reinstall --no-deps
```

Cleaning `dist/` first ensures the wildcard only matches the wheel from this build.

These steps are equivalent to `bash build.sh --release`.

### 3.3 Development Build

To make the current environment reference the source tree in editable mode, run:

```bash
python -m pip install --no-build-isolation --no-deps -e . -v
```

After changing CUDA/C++ code, run an incremental build to refresh the `.so` in the source tree:

```bash
bash build.sh
```

This is equivalent to `python setup.py build_ext --inplace`: it keeps the Ninja state in `build/`, recompiles only changed files, and copies the extension into `halops/` without building or installing a wheel. Use `PYTHON=/path/to/python bash build.sh` to choose the interpreter.

### 3.4 Control Build Parallelism

PyTorch extensions compile in parallel through Ninja by default. Reduce the number of parallel jobs if memory is limited:

```bash
MAX_JOBS=4 python -m pip install --no-build-isolation --no-deps -v .
```

## 4. Verify the Build

Check the installation with an import check and the operator tests.

### 4.1 Import Check

The check below only verifies that the Python package and the extension load; it does not launch any GPU kernel:

```bash
python - <<'PY'
import halops
import halops.halops_kernel as kernel

print("Package path:", halops.__file__)
print("Extension path:", kernel.__file__)
PY
```

### 4.2 Operator Tests

Operator tests require an NVIDIA Blackwell GPU compatible with `sm_100f`; the build does not run on other architectures.

A minimal operator test:

```bash
python tests/test_dense_gemm.py
```
