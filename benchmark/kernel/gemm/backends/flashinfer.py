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
import importlib.util
import torch


_FI_AVAILABLE = importlib.util.find_spec("flashinfer") is not None


def _check():
    if not _FI_AVAILABLE:
        raise ImportError("flashinfer not installed.")


# fastest backend for flashinfer.
FI_DEFAULT_BACKEND = {"w8a8-mxfp8": "cudnn", "w4a4-nvfp4": "cute-dsl"}


def _fi_backend(inp: dict) -> str:
    """Resolve the FlashInfer kernel backend: explicit override, else per-mode default."""
    return inp.get("backend") or FI_DEFAULT_BACKEND.get(
        inp.get("mode", "w8a8-mxfp8"), "auto"
    )


def run_fi_dense_gemm(inp: dict):
    """Run FlashInfer dense GEMM (W8A8-MXFP8 via mm_mxfp8, W4A4 via mm_fp4)."""
    _check()
    import flashinfer

    # flashinfer >= 0.7.2 dense kernels only take 1D 128x4-swizzled block
    # scales (M_padded * K_padded,).
    mode = inp.get("mode", "w8a8-mxfp8")
    backend = _fi_backend(inp)
    if mode == "w4a4-mxfp4":
        raise ValueError("flashinfer backend does not support mode 'w4a4-mxfp4' yet")
    if mode == "w4a4-nvfp4":
        a_descale = inp["sfa"].view(torch.float8_e4m3fn).reshape(-1)
        b_descale = inp["sfb"].view(torch.float8_e4m3fn).reshape(-1)
        alpha_scalar = inp["alpha"].reshape(())
        return flashinfer.mm_fp4(
            inp["x"],
            inp["w"].T,
            a_descale,
            b_descale,
            alpha=alpha_scalar,
            out_dtype=torch.bfloat16,
            block_size=16,
            backend=backend,
            use_nvfp4=True,
        )
    else:
        return flashinfer.mm_mxfp8(
            inp["x"],
            inp["w"].T,
            inp["a_descale"].reshape(-1),
            inp["b_descale"].reshape(-1),
            out_dtype=torch.bfloat16,
            backend=backend,
        )


def run_fi_group_masked_gemm(inp: dict):
    """Run FlashInfer grouped masked GEMM (W8A8-MXFP8 via grouped_mm_mxfp8, W4A4 via grouped_mm_fp4)."""
    _check()
    import flashinfer

    mode = inp.get("mode", "w8a8-mxfp8")
    m_indptr = inp["m_indptr"]

    if mode == "w4a4-mxfp4":
        raise ValueError("flashinfer backend does not support mode 'w4a4-mxfp4' yet")
    if mode == "w4a4-nvfp4":
        return flashinfer.grouped_mm_fp4(
            inp["x"],
            inp["w"],
            inp["sfa"].view(torch.float8_e4m3fn),
            inp["sfb"].view(torch.float8_e4m3fn),
            m_indptr,
            alpha=None,
            out_dtype=torch.bfloat16,
            block_size=16,
            backend=inp.get("backend", "cudnn"),
        )
    else:
        return flashinfer.grouped_mm_mxfp8(
            inp["x"],
            inp["w"],
            inp["sfa"],
            inp["sfb"],
            m_indptr,
            out_dtype=torch.bfloat16,
            backend=inp.get("backend", "cudnn"),
        )


def run_fi_group_contiguous_gemm(inp: dict):
    """Run FlashInfer grouped contiguous GEMM (same API as masked, different m_indptr)."""
    return run_fi_group_masked_gemm(inp)
