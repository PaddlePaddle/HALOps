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

Pytest conftest for HALOps CI benchmark tests.
"""

import os

import pytest


def pytest_addoption(parser):
    """Register custom CLI options for HALOps CI.

    Args:
        parser: pytest command-line argument parser.
    """
    parser.addoption(
        "--benchmark-output-dir",
        default=None,
        help="Override benchmark output directory (default: auto-generated timestamped dir)",
    )
    parser.addoption(
        "--gpu",
        default="0",
        help="GPU device id for benchmark (default: 0)",
    )
    parser.addoption(
        "--backend",
        default="halops",
        help="Backend name for benchmark (default: halops)",
    )


@pytest.fixture(scope="session")
def benchmark_output_dir(request, tmp_path_factory):
    """Get benchmark output directory.

    Uses --benchmark-output-dir if specified, otherwise creates a temp directory.

    Args:
        request: pytest request object.
        tmp_path_factory: pytest temporary directory factory.

    Returns:
        str: Absolute path of the benchmark output directory.
    """
    custom_dir = request.config.getoption("--benchmark-output-dir")
    if custom_dir:
        os.makedirs(custom_dir, exist_ok=True)
        return custom_dir
    d = tmp_path_factory.mktemp("benchmark_results")
    return str(d)


@pytest.fixture(scope="session")
def gpu_id(request):
    """Get GPU device ID.

    Args:
        request: pytest request object.

    Returns:
        str: GPU device ID string, defaults to "0".
    """
    return request.config.getoption("--gpu")


@pytest.fixture(scope="session")
def backend(request):
    """Get backend name for benchmark.

    Args:
        request: pytest request object.

    Returns:
        str: Backend name string, defaults to "halops".
    """
    return request.config.getoption("--backend")


@pytest.fixture(scope="session")
def baseline_dir():
    """Get baseline data directory path.

    Returns:
        str: Absolute path to the baselines directory (ci/baselines/).
    """
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "baselines")


@pytest.fixture(scope="session")
def halops_root():
    """Get HALOps project root directory path.

    Auto-detected from current file location: ci/conftest.py's parent is project root.

    Returns:
        str: Absolute path to the HALOps project root.
    """
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="session")
def configs_dir(halops_root):
    """Get CI config files directory path.

    Args:
        halops_root (str): HALOps project root directory.

    Returns:
        str: Absolute path to benchmark/ci_configs/ directory.
    """
    return os.path.join(halops_root, "benchmark", "ci_configs")
