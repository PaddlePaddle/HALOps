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
from dataclasses import dataclass
import numpy as np

from core.timing import TimingResult


@dataclass
class BenchmarkMetrics:
    median_us: float = 0.0
    std_us: float = 0.0
    tflops: float = 0.0
    tb_per_sec: float = 0.0
    bw_pct: float = 0.0
    # Roofline (Williams et al., CACM 2009): P = min(peak_tflops, peak_tbps * I),
    # I = arithmetic intensity = flops / bytes_io (FLOP/byte). roofline_tflops is
    # this case's theoretical ceiling on *this* GPU; attainable_pct = measured /
    # ceiling (how close the kernel is to its own roofline, not the bare HW peak).
    arithmetic_intensity: float = 0.0
    roofline_tflops: float = 0.0
    attainable_pct: float = 0.0
    # candidate device_us / halops device_us; halops itself = 1.0.
    # Set by the scenario loop (needs cross-backend context); 0.0 = unavailable.
    speedup: float = 0.0
    # Display-only; not used for TFLOPS/BW (those must use device_us).
    host_us: float = 0.0  # median of CPU launch API span
    dispatch_us: float = 0.0  # median of (CPU first launch -> GPU kernel start)


def _roofline(
    flops: int,
    bytes_io: int,
    tflops: float,
    gpu_peak_tflops: float,
    gpu_peak_tbps: float,
) -> tuple[float, float, float]:
    """Compute (arithmetic_intensity, roofline_tflops, attainable_pct).

    Roofline ceiling for this case: min(compute roof, memory roof). I is in
    FLOP/byte; memory roof (peak_tbps * I) and compute roof (peak_tflops) are
    both in TFLOPS since peak_tbps is TB/s and I is FLOP/byte.
    """
    arithmetic_intensity = flops / bytes_io if bytes_io else 0.0
    roofline_tflops = min(gpu_peak_tflops, gpu_peak_tbps * arithmetic_intensity)
    attainable_pct = tflops / roofline_tflops * 100 if roofline_tflops else 0.0
    return arithmetic_intensity, roofline_tflops, attainable_pct


# GPU peak specs (BF16, FP8, FP4 dense TFLOPS, HBM bandwidth TB/s).
# A precision is None when the GPU lacks that tensor-core support (A100 has no
# FP8/FP4; FP4 is Blackwell-only, so Hopper GPUs list None). Source: NVIDIA.
GPU_SPECS: dict[str, tuple[float, float | None, float | None, float]] = {
    # name        : (bf16_tflops, fp8_tflops, fp4_tflops, hbm_tbps)
    "a100": (312.0, None, None, 2.0),
    "h100": (1979.0, 3958.0, None, 3.35),
    "h100_pcie": (1513.0, 3026.0, None, 2.0),
    "h200": (1979.0, 3958.0, None, 4.8),
    "b200": (2250.0, 4500.0, 9000.0, 7.7),
}
_DEFAULT_GPU = "b200"


def get_gpu_peak(gpu_name: str, dtype_bytes: float) -> tuple[float, float]:
    """Return (peak_tflops, peak_tbps) for a GPU and operand precision.

    dtype_bytes: 0.5 -> FP4 (w4a4-nvfp4), 1 -> FP8, 2 -> BF16/FP16.
    Raises ValueError if the requested precision is unsupported on the GPU.
    """
    bf16_tflops, fp8_tflops, fp4_tflops, hbm_tbps = GPU_SPECS.get(
        gpu_name.lower(), GPU_SPECS[_DEFAULT_GPU]
    )
    if dtype_bytes >= 2:
        peak_tflops = bf16_tflops
    elif dtype_bytes == 1:
        peak_tflops = fp8_tflops
    else:
        peak_tflops = fp4_tflops
    if peak_tflops is None:
        raise ValueError(
            f"{gpu_name} has no tensor-core support for {dtype_bytes}-byte precision"
        )
    return peak_tflops, hbm_tbps


def compute_metrics(
    timing: TimingResult,
    batch_size: int,
    s_q: int,
    s_kv: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    causal: bool,
    dtype_bytes: int = 2,
    kv_dtype_bytes: int | None = None,
    out_dtype_bytes: int | None = None,
    baseline_us: float = 0.0,
    gpu_type: str = _DEFAULT_GPU,
) -> BenchmarkMetrics:
    gpu_peak_tflops, gpu_peak_tbps = get_gpu_peak(gpu_type, dtype_bytes)
    # TFLOPS / bandwidth must be based on device time (CUPTI GPU hardware clock).
    median_us = float(np.median(timing.device_us))
    std_us = float(np.std(timing.device_us))

    flops = 4 * batch_size * s_q * s_kv * num_q_heads * head_dim
    if causal:
        flops //= 2
    tflops = flops / (median_us * 1e-6) / 1e12

    kv_bytes = kv_dtype_bytes if kv_dtype_bytes is not None else dtype_bytes
    out_bytes = out_dtype_bytes if out_dtype_bytes is not None else dtype_bytes
    bytes_io = (
        dtype_bytes * batch_size * s_q * num_q_heads * head_dim  # Q
        + kv_bytes * batch_size * s_kv * num_kv_heads * head_dim * 2  # K + V
        + out_bytes * batch_size * s_q * num_q_heads * head_dim  # O
    )
    tb_per_sec = bytes_io / (median_us * 1e-6) / 1e12
    bw_pct = tb_per_sec / gpu_peak_tbps * 100

    arithmetic_intensity, roofline_tflops, attainable_pct = _roofline(
        flops, bytes_io, tflops, gpu_peak_tflops, gpu_peak_tbps
    )

    host_us = float(np.median(timing.host_us)) if timing.host_us else 0.0
    dispatch_us = float(np.median(timing.dispatch_us)) if timing.dispatch_us else 0.0
    return BenchmarkMetrics(
        median_us,
        std_us,
        tflops,
        tb_per_sec,
        bw_pct,
        arithmetic_intensity=arithmetic_intensity,
        roofline_tflops=roofline_tflops,
        attainable_pct=attainable_pct,
        speedup=median_us / baseline_us if baseline_us and baseline_us > 0 else 1.0,
        host_us=host_us,
        dispatch_us=dispatch_us,
    )


def compute_gemm_metrics(
    timing: TimingResult,
    m: int,
    n: int,
    k: int,
    dtype_bytes: float = 1,
    baseline_us: float = 0.0,
    gpu_type: str = _DEFAULT_GPU,
    num_experts: int = 1,
) -> BenchmarkMetrics:
    """Compute benchmark metrics for a GEMM kernel.

    Args:
        timing: GPU timing result from bench_gpu_time.
        m, n, k: GEMM dimensions (C[m,n] = A[m,k] x B[k,n]).
        dtype_bytes: Bytes per operand element (1 for FP8, 0.5 for FP4,
            2 for BF16/FP16).
        baseline_us: HALOps median_us for speedup calculation.
        gpu_type: GPU type for peak bandwidth lookup.
        num_experts: Number of expert weight matrices for grouped GEMM
            (each [n, k] is loaded once); 1 for dense GEMM. FLOPs are
            unchanged (each row only multiplies its own expert), but the
            B-matrix bandwidth scales with num_experts.
    """
    gpu_peak_tflops, gpu_peak_tbps = get_gpu_peak(gpu_type, dtype_bytes)
    median_us = float(np.median(timing.device_us))
    std_us = float(np.std(timing.device_us))

    flops = 2 * m * n * k
    tflops = flops / (median_us * 1e-6) / 1e12
    # A[m,k] + B[num_experts, n, k] + C[m,n]
    bytes_io = dtype_bytes * (m * k + num_experts * k * n + m * n)
    tb_per_sec = bytes_io / (median_us * 1e-6) / 1e12
    bw_pct = tb_per_sec / gpu_peak_tbps * 100

    arithmetic_intensity, roofline_tflops, attainable_pct = _roofline(
        flops, bytes_io, tflops, gpu_peak_tflops, gpu_peak_tbps
    )

    host_us = float(np.median(timing.host_us)) if timing.host_us else 0.0
    dispatch_us = float(np.median(timing.dispatch_us)) if timing.dispatch_us else 0.0

    return BenchmarkMetrics(
        median_us,
        std_us,
        tflops,
        tb_per_sec=tb_per_sec,
        bw_pct=bw_pct,
        arithmetic_intensity=arithmetic_intensity,
        roofline_tflops=roofline_tflops,
        attainable_pct=attainable_pct,
        speedup=median_us / baseline_us if baseline_us and baseline_us > 0 else 1.0,
        host_us=host_us,
        dispatch_us=dispatch_us,
    )
