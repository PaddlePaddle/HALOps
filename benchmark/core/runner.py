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
import io
import sys
import multiprocessing as mp
from contextlib import contextmanager


def _subprocess_entry(conn, func, args, kwargs):
    """Generic subprocess worker. Captures stdout/stderr and sends back."""
    captured = io.StringIO()
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = captured
    try:
        result = func(*args, **kwargs)
        conn.send(("ok", result, captured.getvalue()))
    except Exception as e:
        conn.send(("err", str(e), captured.getvalue()))
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr
        conn.close()


@contextmanager
def bench_runner(use_subprocess: bool = False, timeout: float = 120):
    """Context manager yielding a `run(target, *args, **kwargs)` callable.

    use_subprocess=False: call target(*args, **kwargs) in current process.
    use_subprocess=True:  spawn daemon subprocess, poll with timeout, kill if stale.
    target must be pickle-able (module-level function) when use_subprocess=True.
    """

    def _run_direct(func, *args, **kwargs):
        return func(*args, **kwargs)

    def _run_subprocess(func, *args, **kwargs):
        ctx = mp.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe()
        p = ctx.Process(
            target=_subprocess_entry,
            args=(child_conn, func, args, kwargs),
            daemon=True,
        )
        p.start()
        if parent_conn.poll(timeout):
            status, payload, output = parent_conn.recv()
            p.join()
            if output:
                sys.stdout.write(output)
            if status == "ok":
                return payload
            raise RuntimeError(payload)
        else:
            p.kill()
            p.join()
            raise TimeoutError(f"exceeded {timeout}s")

    yield _run_subprocess if use_subprocess else _run_direct


def run_bench_for_backend(
    bench_entry_func,
    backend_name: str,
    cfg: dict,
    dry_run_iters: int,
    repeat_iters: int,
    cold_l2_cache: bool,
    use_cuda_graph: bool,
    timing_method: str,
    use_subprocess: bool = False,
    timeout: float = 120,
) -> tuple:
    """Run a single-backend benchmark entry and return (status, timing_or_None).

    Parameters
    ----------
    bench_entry_func:
        Module-level callable with signature
        bench_entry_func(cfg, backend_name, dry_run_iters, repeat_iters,
                         cold_l2_cache, use_cuda_graph, timing_method).
        Must be pickle-able when use_subprocess=True.
    backend_name:
        Name of the backend being benchmarked, used in error messages.
    cfg, dry_run_iters, repeat_iters, cold_l2_cache, use_cuda_graph, timing_method:
        Forwarded verbatim to bench_entry_func.
    use_subprocess:
        Spawn an isolated subprocess for each backend (GPU memory isolation).
    timeout:
        Seconds before the subprocess is killed and TimeoutError is raised.

    Returns
    -------
    status : str
        "OK", "TIMEOUT", "OOM", or "FAIL".
    timing : object or None
        Return value of bench_entry_func on success, None on failure.
    """
    status = "OK"
    timing = None
    try:
        with bench_runner(use_subprocess=use_subprocess, timeout=timeout) as runner:
            timing = runner(
                bench_entry_func,
                cfg,
                backend_name,
                dry_run_iters,
                repeat_iters,
                cold_l2_cache,
                use_cuda_graph,
                timing_method,
            )
    except TimeoutError:
        status = "TIMEOUT"
    except RuntimeError as e:
        status = "OOM" if "out of memory" in str(e).lower() else "FAIL"
        print(f"[ERROR] {backend_name}: {e}")
    except Exception as e:
        status = "FAIL"
        print(f"[ERROR] {backend_name}: {e}")
    return status, timing


def run_refcheck_cross_backends(
    refcheck_entry_func,
    cfg: dict,
    backend_names: list,
    refcheck_reference: str,
    refcheck_atol: float = 1e-2,
    refcheck_rtol: float = 1e-2,
    use_subprocess: bool = False,
    timeout: float = 120,
    probe_entry_func=None,
) -> dict:
    """Run refcheck for all backends and return a per-backend results dict.

    Builds the ordered check list (reference first), guards against too-few
    backends, runs the entry function, and handles errors — so the call site
    only needs a single line.

    If probe_entry_func is provided, every candidate (the preferred reference
    included) is probed in an isolated subprocess (dry_run=1, repeat=1) before
    the shared refcheck run. Candidates that fail the probe are dropped. The
    actual reference is the first surviving candidate, so an unusable preferred
    reference (e.g. golden not registered for this scenario or unsupported for
    this config) transparently falls back to the first viable backend.

    Parameters
    ----------
    refcheck_entry_func:
        Module-level callable with signature
        refcheck_entry_func(cfg, ordered_names, refcheck_reference,
                            refcheck_atol, refcheck_rtol) -> dict.
        Must be pickle-able when use_subprocess=True.
    cfg:
        Benchmark config forwarded verbatim to refcheck_entry_func.
    backend_names:
        All backend names that participate in this benchmark run.
    refcheck_reference:
        Name of the reference backend used as comparison baseline.
    refcheck_atol, refcheck_rtol:
        Absolute / relative tolerances forwarded to refcheck_entry_func.
    use_subprocess:
        Spawn an isolated subprocess (GPU memory isolation).
    timeout:
        Seconds before the subprocess is killed.
    probe_entry_func:
        Optional bench_entry_func-compatible callable used to probe each
        non-reference backend before refcheck.  When None, probing is skipped.

    Returns
    -------
    dict
        Mapping backend_name -> refcheck result, or {n: None} on failure/skip.
    """
    default = {n: None for n in backend_names}
    # Preferred reference goes first. The first candidate that actually runs
    # becomes the comparison baseline, so an unusable preferred reference (e.g.
    # golden not registered for this scenario, or unsupported for this config)
    # transparently falls back to the first viable backend.
    candidates = [refcheck_reference] + [
        n for n in backend_names if n != refcheck_reference
    ]
    if len(candidates) < 2:
        print(
            f"[REFCHECK] only '{candidates[0]}' to check against "
            f"reference '{refcheck_reference}'; nothing to compare, skipping"
        )
        return default

    # Probe every candidate (reference included); drop the ones that cannot run.
    if probe_entry_func is not None:
        viable = []
        for name in candidates:
            status, _ = run_bench_for_backend(
                probe_entry_func,
                name,
                cfg,
                dry_run_iters=1,
                repeat_iters=1,
                cold_l2_cache=False,
                use_cuda_graph=False,
                timing_method="event",
                use_subprocess=True,
                timeout=timeout,
            )
            if status == "OK":
                viable.append(name)
            else:
                tag = "reference " if name == refcheck_reference else ""
                print(f"[REFCHECK] probe {tag}'{name}' failed ({status}), excluded")
        candidates = viable

    if len(candidates) < 2:
        print("[REFCHECK] no viable backends after probing, skipping")
        return default

    # Resolve the actual reference: the first viable candidate. If the preferred
    # reference was dropped, fall back to the first backend that survived.
    resolved_reference = candidates[0]
    if resolved_reference != refcheck_reference:
        print(
            f"[REFCHECK] reference '{refcheck_reference}' unusable; "
            f"falling back to '{resolved_reference}' as reference"
        )

    try:
        with bench_runner(use_subprocess=use_subprocess, timeout=timeout) as runner:
            return runner(
                refcheck_entry_func,
                cfg,
                candidates,
                resolved_reference,
                refcheck_atol,
                refcheck_rtol,
            )
    except (TimeoutError, RuntimeError) as e:
        print(f"[REFCHECK] {e}, skipping")
        return default
