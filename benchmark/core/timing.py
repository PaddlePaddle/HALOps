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
from typing import Callable
from dataclasses import dataclass, field
import bisect
import warnings
import torch
from core.cupti import cupti


_L2_FLUSH_SIZE_BYTES = 256 * 1024 * 1024


def _flush_l2_cache() -> None:
    """Evict the GPU L2 cache with a buffer larger than the target GB100 L2."""
    device = torch.cuda.current_device()
    buf = torch.empty(
        _L2_FLUSH_SIZE_BYTES // 4,
        dtype=torch.int32,
        device=torch.device("cuda", device),
    )
    buf.zero_()
    torch.cuda.synchronize(device)
    del buf


@dataclass
class TimingResult:
    """
    device_us   : pure GPU kernel time (CUPTI hardware clock, ns -> us)
    host_us     : total CPU launch span (RUNTIME activity span)
    dispatch_us : delay from first CPU launch to actual GPU kernel start
    """

    device_us: list[float]
    host_us: list[float] = field(default_factory=list)
    dispatch_us: list[float] = field(default_factory=list)


def _run_warmup(fn: Callable, input_args: tuple, dry_run_iters: int) -> None:
    for _ in range(dry_run_iters):
        fn(*input_args)
    torch.cuda.synchronize()


def _measure_with_event(
    runner: Callable,
    repeat_iters: int,
    cold_l2_cache: bool,
) -> list[float]:
    """Return per-iteration elapsed times (us) measured with torch.cuda.Event."""
    result: list[float] = []
    for _ in range(repeat_iters):
        if cold_l2_cache:
            _flush_l2_cache()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        runner()
        end.record()
        torch.cuda.synchronize()
        result.append(start.elapsed_time(end) * 1e3)
    return result


def bench_gpu_time_cupti(
    fn: Callable,
    input_args: tuple,
    dry_run_iters: int = 10,
    repeat_iters: int = 50,
    cold_l2_cache: bool = True,
    use_cuda_graph: bool = False,
) -> TimingResult:
    """Benchmark GPU kernel time using CUPTI activity tracing."""
    # CUPTI buffer callbacks
    launches: list[tuple] = []  # (start_ns, end_ns, corr_id, cbid, kind)
    kernels: list[tuple] = []  # (name, start_ns, end_ns, corr_id)

    def _buf_requested():
        return 8 * 1024 * 1024, 0

    def _buf_completed(activities):
        for act in activities:
            if act.kind == cupti.ActivityKind.CONCURRENT_KERNEL:
                kernels.append((act.name, act.start, act.end, act.correlation_id))
            elif act.kind in (cupti.ActivityKind.RUNTIME, cupti.ActivityKind.DRIVER):
                launches.append(
                    (act.start, act.end, act.correlation_id, act.cbid, act.kind)
                )

    _run_warmup(fn, input_args, dry_run_iters)

    # Optional CUDA graph capture.
    runner = lambda: fn(*input_args)  # noqa: E731
    if use_cuda_graph:
        torch.cuda.synchronize()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                fn(*input_args)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn(*input_args)
        runner = g.replay

    # Enable CUPTI.
    cupti.activity_enable(cupti.ActivityKind.CONCURRENT_KERNEL)
    cupti.activity_enable(cupti.ActivityKind.RUNTIME)
    cupti.activity_enable(cupti.ActivityKind.DRIVER)
    cupti.activity_register_callbacks(_buf_requested, _buf_completed)

    iter_timestamps: list[tuple[int, int]] = []
    for _ in range(repeat_iters):
        if cold_l2_cache:
            _flush_l2_cache()
        torch.cuda.synchronize()
        t0 = cupti.get_timestamp()
        runner()
        t1 = cupti.get_timestamp()
        torch.cuda.synchronize()
        iter_timestamps.append((t0, t1))

    cupti.activity_flush_all(0)
    cupti.activity_disable(cupti.ActivityKind.CONCURRENT_KERNEL)
    cupti.activity_disable(cupti.ActivityKind.RUNTIME)
    cupti.activity_disable(cupti.ActivityKind.DRIVER)
    cupti.finalize()

    # Post-hoc correlation: CPU launch -> GPU kernel.
    sorted_launches = sorted(launches, key=lambda x: x[0])
    launch_starts = [x[0] for x in sorted_launches]

    corr_to_kernels: dict[int, list] = {}
    for k in kernels:
        corr_to_kernels.setdefault(k[3], []).append(k)

    device_us, host_us, dispatch_us = [], [], []
    for idx, (t0, t1) in enumerate(iter_timestamps):
        li = bisect.bisect_left(launch_starts, t0)
        ri = bisect.bisect_right(launch_starts, t1)
        corr_ids = {sorted_launches[i][2] for i in range(li, ri)}

        iter_kernels = []
        for cid in corr_ids:
            if cid in corr_to_kernels:
                iter_kernels.extend(corr_to_kernels[cid])
        if not iter_kernels:
            raise ValueError(f"iter {idx}: no kernel activity found")

        gpu_start = min(k[1] for k in iter_kernels)
        gpu_end = max(k[2] for k in iter_kernels)
        device_us.append((gpu_end - gpu_start) / 1e3)

        iter_launches = [sorted_launches[i] for i in range(li, ri)]
        cpu_start = min(x[0] for x in iter_launches)
        cpu_end = max(x[1] for x in iter_launches)
        host_us.append((cpu_end - cpu_start) / 1e3)

        dispatch_us.append((gpu_start - cpu_start) / 1e3)

    return TimingResult(device_us=device_us, host_us=host_us, dispatch_us=dispatch_us)


def bench_gpu_time_event(
    fn: Callable,
    input_args: tuple,
    dry_run_iters: int = 10,
    repeat_iters: int = 50,
    cold_l2_cache: bool = True,
) -> TimingResult:
    """Benchmark GPU time using torch.cuda.Event only (no profiler overhead)."""
    _run_warmup(fn, input_args, dry_run_iters)
    runner = lambda: fn(*input_args)  # noqa: E731
    return TimingResult(
        device_us=_measure_with_event(runner, repeat_iters, cold_l2_cache)
    )


def bench_gpu_time_profiler(
    fn: Callable,
    input_args: tuple,
    dry_run_iters: int = 10,
    repeat_iters: int = 50,
    cold_l2_cache: bool = True,
) -> TimingResult:
    """Benchmark GPU kernel time using torch.profiler."""
    warnings.filterwarnings("ignore", message=".*Profiler clears events.*")

    _run_warmup(fn, input_args, dry_run_iters)
    runner = lambda: fn(*input_args)  # noqa: E731

    # Pass 1: torch.cuda.Event for end-to-end (host-perceived) latency.
    host_us = _measure_with_event(runner, repeat_iters, cold_l2_cache)

    # Pass 2: torch.profiler for device kernel time, one profile per iteration.
    device_us: list[float] = []
    for _ in range(repeat_iters):
        if cold_l2_cache:
            _flush_l2_cache()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
            with_stack=False,
        ) as prof:
            runner()
        torch.cuda.synchronize()
        kernel_us = sum(
            e.device_time_total for e in prof.key_averages() if e.device_time_total > 0
        )
        device_us.append(kernel_us)

    return TimingResult(device_us=device_us, host_us=host_us)


def bench_gpu_time(
    fn: Callable,
    input_args: tuple,
    dry_run_iters: int = 10,
    repeat_iters: int = 50,
    cold_l2_cache: bool = True,
    use_cuda_graph: bool = False,
    method: str = "profiler",
) -> TimingResult:
    """Dispatch to a timing backend selected by ``method``.

    "event"    -> bench_gpu_time_event    (torch.cuda.Event only; fast, low overhead)
    "cupti"    -> bench_gpu_time_cupti    (CUPTI hardware clock; supports CUDA graph)
    "profiler" -> bench_gpu_time_profiler (torch.cuda.Event + torch.profiler)
    """
    if method == "event":
        if use_cuda_graph:
            print(
                "[WARN] timing_method='event' does not support CUDA graph; ignoring use_cuda_graph."
            )
        return bench_gpu_time_event(
            fn,
            input_args,
            dry_run_iters=dry_run_iters,
            repeat_iters=repeat_iters,
            cold_l2_cache=cold_l2_cache,
        )
    if method == "cupti":
        return bench_gpu_time_cupti(
            fn,
            input_args,
            dry_run_iters=dry_run_iters,
            repeat_iters=repeat_iters,
            cold_l2_cache=cold_l2_cache,
            use_cuda_graph=use_cuda_graph,
        )
    if method == "profiler":
        if use_cuda_graph:
            print(
                "[WARN] timing_method='profiler' does not support CUDA graph; "
                "ignoring use_cuda_graph."
            )
        return bench_gpu_time_profiler(
            fn,
            input_args,
            dry_run_iters=dry_run_iters,
            repeat_iters=repeat_iters,
            cold_l2_cache=cold_l2_cache,
        )
    raise ValueError(
        f"unknown timing method {method!r}; expected 'event', 'cupti', or 'profiler'"
    )
