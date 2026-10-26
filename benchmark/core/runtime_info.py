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

import re
import subprocess
from importlib.metadata import (
    PackageNotFoundError,
    packages_distributions,
    version as distribution_version,
)
from pathlib import Path


# Config backend-name prefix -> top-level Python import name.
_PACKAGE_IMPORT_NAMES = {
    "fi": "flashinfer",
    "fa": "flash_attn",
    "sglang": "sglang",
}


def get_package_version(import_name: str) -> str:
    """Return the installed distribution version for a Python import."""
    try:
        distributions = packages_distributions().get(import_name)
        distribution = distributions[0] if distributions else import_name
        return distribution_version(distribution)
    except PackageNotFoundError:
        return "not available"
    except Exception as error:
        return f"error ({error})"


def get_module_version(import_name: str) -> str:
    """Return a module version, falling back to distribution metadata."""
    dist_version = get_package_version(import_name)
    if dist_version != "not available":
        return dist_version
    try:
        module = __import__(import_name)
        version = getattr(module, "__version__", None)
        if version:
            return str(version)
    except Exception:
        pass
    return dist_version


def get_torch_version() -> str:
    """Return the imported PyTorch version."""
    try:
        import torch

        return torch.__version__
    except Exception as error:
        return f"error ({error})"


def get_halops_commit() -> str:
    """Return the short commit of the HALOps source tree."""
    try:
        project_root = Path(__file__).resolve().parents[2]
        return subprocess.check_output(
            ["git", "-C", str(project_root), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def get_driver_version() -> str:
    """Return the installed NVIDIA driver version."""
    try:
        versions = (
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version",
                    "--format=csv,noheader",
                ],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            .strip()
            .splitlines()
        )
        if versions:
            return versions[0].strip()
    except Exception:
        pass

    try:
        version_text = Path("/proc/driver/nvidia/version").read_text()
        match = re.search(
            r"NVRM version:.*?Kernel Module(?: for \S+)?\s+([0-9]+(?:\.[0-9]+)+)",
            version_text,
        )
        if match:
            return match.group(1)
    except Exception:
        pass
    return "unknown"


def get_gpu_name() -> str:
    """Return the current GPU name for benchmark reporting."""
    # Prefer GPU name from nvidia-smi (internal SKUs may differ from torch's name).
    try:
        gpu_name = (
            subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            .strip()
            .splitlines()[0]
        )
    except Exception:
        try:
            import torch

            gpu_name = (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else "unknown"
            )
        except Exception:
            gpu_name = "unknown"
    return gpu_name


def _format_cudnn_version(version: object) -> str | None:
    """Format cuDNN's encoded integer version as major.minor.patch."""
    try:
        encoded = int(version)
    except (TypeError, ValueError):
        return None
    if encoded <= 0:
        return None
    if encoded >= 90000:
        return f"{encoded // 10000}.{(encoded // 100) % 100}.{encoded % 100}"
    return f"{encoded // 1000}.{(encoded // 100) % 10}.{encoded % 100}"


def get_cudnn_version() -> str:
    """Return the loaded cuDNN runtime version."""
    try:
        import cudnn

        version = _format_cudnn_version(cudnn.backend_version())
        if version:
            return version
    except Exception:
        pass

    try:
        import torch
    except Exception as error:
        return f"error ({error})"

    try:
        version = _format_cudnn_version(torch.backends.cudnn.version())
        if version:
            return version
    except Exception:
        pass

    # PyTorch may initialize its cuDNN handle lazily. Trigger a tiny operation
    # only when the runtime version was unavailable without initialization.
    try:
        if torch.cuda.is_available():
            value = torch.randn(1, 1, 4, 4, device="cuda")
            weight = torch.randn(1, 1, 3, 3, device="cuda")
            torch.nn.functional.conv2d(value, weight)
            torch.cuda.synchronize()
            version = _format_cudnn_version(torch.backends.cudnn.version())
            if version:
                return version
    except Exception:
        pass
    return "not available"


def get_deep_gemm_version() -> str:
    """Return DeepGEMM's module version and differing dist version."""
    try:
        import deep_gemm

        module_version = getattr(deep_gemm, "__version__", None)
    except Exception:
        module_version = None

    dist_version = get_package_version("deep_gemm")
    if module_version and dist_version != module_version:
        return f"{module_version} (dist {dist_version})"
    return module_version or dist_version


def get_trtllm_gen_version() -> str:
    """Return the FlashInfer release carrying the TRT-LLM Gen backend."""
    # This backend is shipped inside FlashInfer and is unrelated to any
    # separately installed tensorrt_llm Python distribution.
    flashinfer_version = get_module_version("flashinfer")
    if flashinfer_version != "not available":
        return f"bundled with flashinfer {flashinfer_version}"
    return "not available"


def _backend_versions(name: str) -> list[tuple[str, str]]:
    """Resolve one configured backend to its displayed dependencies."""
    base, _, qualifier = name.partition("(")
    qualifier = qualifier.removesuffix(")")

    if base == "halops":
        return [("halops", f"commit {get_halops_commit()}")]
    if base == "cudnn":
        return [("cudnn", get_cudnn_version())]
    if base in {"deepgemm", "deep_gemm"}:
        return [("deep_gemm", get_deep_gemm_version())]

    versions: list[tuple[str, str]] = []
    for prefix, import_name in _PACKAGE_IMPORT_NAMES.items():
        if base == prefix or base.startswith(prefix):
            versions.append((import_name, get_module_version(import_name)))
            break

    if base == "fi" and qualifier == "trtllm-gen":
        versions.append(("trtllm-gen", get_trtllm_gen_version()))
    elif base == "fi" and qualifier == "cudnn":
        versions.append(("cudnn", get_cudnn_version()))
    return versions


def get_backend_versions(backend_names: list[str]) -> list[tuple[str, str]]:
    """Return deduplicated version entries for configured backends."""
    versions = [("torch", get_torch_version())]
    seen = {"torch"}
    for name in backend_names:
        resolved = _backend_versions(name)
        if not resolved:
            resolved = [(name, "(unknown backend)")]
        for label, version in resolved:
            if label not in seen:
                versions.append((label, version))
                seen.add(label)
    return versions


def print_backend_versions(backend_names: list[str]) -> None:
    """Print version information for the configured benchmark backends."""
    print("Backend versions:")
    for label, version in get_backend_versions(backend_names):
        print(f"  {label:<14}: {version}")
