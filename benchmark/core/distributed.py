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

Single-node, multi-GPU (tensor-parallel) runs for the benchmark harness.

Usage in a scenario::

    def _rank_body(tp, mode, job):
        ...
        tp.sync(); timing = tp.worst_timing(timing)
        return row if tp.is_leader else None

    with tensor_parallel_context(cfg["tp_size"], __file__) as tp:
        row = tp.run("bench", job, _rank_body)

`tp` is a `RankContext`: with `tp_size == 1` the body runs in this process and
every cross-rank reduction is the identity, so one body serves both cases.
Larger sizes fan the body out over this node's GPUs through
`torch.distributed.run` (an entry runs in a daemonic subprocess, which may not
create multiprocessing children). The job travels as JSON in an environment
variable and rank 0 hands its result back through a file, so no tensor ever
crosses a process boundary.
"""

from __future__ import annotations

import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Callable

import torch

from core.refcheck import RefcheckResult
from core.timing import TimingResult

JOB_ENV = "BENCH_TP_JOB"


def visible_device_count() -> int:
    """GPUs this process may use, without initializing a CUDA context."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        return len([dev for dev in visible.split(",") if dev.strip()])
    return torch.cuda.device_count()


def _benchmark_root() -> str:
    """The `benchmark/` directory, on the workers' import path."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run_tp_job(
    worker_path: str, payload: dict, world: int, timeout: float = 1800.0
) -> dict:
    """Run `worker_path`'s rank handler on `world` GPUs of this node.

    `payload` reaches every rank through `JOB_ENV`; rank 0's return value comes
    back as the result. The caller's `payload` is not modified.
    """
    tmpdir = tempfile.mkdtemp(prefix="bench_tp_")
    out_path = os.path.join(tmpdir, "result.json")
    job = {**payload, "out": out_path}
    root = _benchmark_root()
    # The workers import the scenario through `benchmark/` and the op through
    # the repo root, so both must be visible.
    paths = [root, os.path.dirname(root)]
    if os.environ.get("PYTHONPATH"):
        paths.append(os.environ["PYTHONPATH"])
    env = dict(
        os.environ,
        **{JOB_ENV: json.dumps(job)},
        PYTHONPATH=os.pathsep.join(paths),
    )
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={world}",
        os.path.abspath(worker_path),
    ]
    try:
        proc = subprocess.run(
            cmd, cwd=root, env=env, capture_output=True, text=True, timeout=timeout
        )
        if proc.returncode != 0:
            tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-20:])
            raise RuntimeError(f"tp_size={world} workers failed:\n{tail}")
        with open(out_path) as handle:
            return json.load(handle)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@dataclass
class RankContext:
    """This rank's view of a tensor-parallel launch.

    A single-GPU run is a world of one: every cross-rank reduction is the
    identity, so the same rank body serves both cases.
    """

    rank: int
    world: int
    group_name: str | None

    @classmethod
    def local(cls) -> "RankContext":
        return cls(rank=0, world=1, group_name=None)

    @property
    def is_leader(self) -> bool:
        return self.rank == 0

    def sync(self) -> None:
        if self.world > 1:
            torch.distributed.barrier()

    def sum(self, tensor: torch.Tensor) -> torch.Tensor:
        return sum_across_ranks(tensor) if self.world > 1 else tensor

    def worst_timing(self, timing: TimingResult) -> TimingResult:
        return worst_rank_timing(timing) if self.world > 1 else timing

    def gather_timings(self, timing: TimingResult) -> list[TimingResult]:
        return gather_rank_timings(timing) if self.world > 1 else [timing]

    def worst_refcheck(self, result: RefcheckResult) -> RefcheckResult:
        return worst_rank_refcheck(result) if self.world > 1 else result


class TensorParallel:
    """A `tensor_parallel_context` launch: `run` dispatches one rank body."""

    def __init__(self, tp_size: int, worker_path: str):
        if tp_size < 1 or tp_size > visible_device_count():
            raise ValueError(
                f"tp_size={tp_size} needs that many visible GPUs, "
                f"got {visible_device_count()} (set --gpu accordingly)"
            )
        self.tp_size = tp_size
        self.worker_path = worker_path

    def __enter__(self) -> "TensorParallel":
        return self

    def __exit__(self, *exc_info) -> None:
        return None

    def run(
        self,
        mode: str,
        job: dict,
        dispatch: Callable[["RankContext", str, dict], dict | None],
    ) -> dict:
        """Rank 0's payload of `dispatch(rank, mode, job)`.

        A single-GPU launch runs the body here; otherwise one worker per rank
        of this node runs it under `torch.distributed.run`.
        """
        if self.tp_size == 1:
            return dispatch(RankContext.local(), mode, job)
        return _run_tp_job(self.worker_path, {"mode": mode, **job}, self.tp_size)


def tensor_parallel_context(tp_size: int, worker_path: str) -> TensorParallel:
    """`with tensor_parallel_context(tp_size, __file__) as tp:` then
    `tp.run(mode, job, dispatch)`; the same call works for one GPU or many."""
    return TensorParallel(tp_size, worker_path)


def sum_across_ranks(tensor: torch.Tensor) -> torch.Tensor:
    """All ranks: the elementwise sum of every rank's tensor, in its dtype."""
    total = tensor.clone()
    torch.distributed.all_reduce(total)
    return total


def gather_rank_timings(timing: TimingResult) -> list[TimingResult]:
    """All ranks: every rank's timing distribution, ordered by rank."""
    gathered = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(gathered, timing)
    return gathered


def worst_rank_timing(timing: TimingResult) -> TimingResult:
    """All ranks: the slowest rank's distribution, which the report shows."""
    return max(
        gather_rank_timings(timing),
        key=lambda item: statistics.median(item.device_us),
    )


def worst_rank_refcheck(result: RefcheckResult) -> RefcheckResult:
    """All ranks: pass only if every rank does, with the worst error stats."""
    stats = torch.tensor(
        [
            result.max_atol,
            result.avg_atol,
            result.max_rtol,
            result.avg_rtol,
            result.mismatch_pct,
        ],
        device="cuda",
        dtype=torch.float64,
    )
    torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.MAX)
    passed = torch.tensor([1.0 if result.passed else 0.0], device="cuda")
    torch.distributed.all_reduce(passed, op=torch.distributed.ReduceOp.MIN)
    return RefcheckResult(
        passed=bool(passed.item() > 0.5),
        max_atol=float(stats[0]),
        avg_atol=float(stats[1]),
        max_rtol=float(stats[2]),
        avg_rtol=float(stats[3]),
        mismatch_pct=float(stats[4]),
    )


def tp_worker_main(
    dispatch: Callable[[RankContext, str, dict], dict | None],
) -> None:
    """One rank of a `TensorParallel.run`, run under `torch.distributed.run`.

    `dispatch(rank_context, mode, job)` runs this rank and returns rank 0's
    payload (None on the other ranks)."""
    import torch.distributed as dist

    job = json.loads(os.environ[JOB_ENV])
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    try:
        ctx = RankContext(rank, world, dist.group.WORLD.group_name)
        result = dispatch(ctx, job["mode"], job)
        if rank == 0:
            with open(job["out"], "w") as handle:
                json.dump(result, handle)
    finally:
        dist.destroy_process_group()
