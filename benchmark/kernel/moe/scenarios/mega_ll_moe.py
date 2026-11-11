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

import math
import statistics
from dataclasses import dataclass
from pathlib import Path

import torch

from core.distributed import (
    RankContext,
    tensor_parallel_context,
    tp_worker_main,
    visible_device_count,
)
from core.metrics import BenchmarkMetrics, compute_moe_metrics
from core.output import (
    MoeRowResult,
    append_moe_csv,
    print_scenario_header,
    print_table_header,
    print_table_row,
)
from core.refcheck import RefcheckResult, _compare_to_reference
from core.runner import run_bench_for_backend, run_refcheck_cross_backends
from core.timing import TimingResult, bench_gpu_time, profile_kernel_timeline
from kernel.moe.backends.golden import (
    _dequant_weight,
    _route,
    routed_and_shared,
    run_mega_ll_moe_golden,
    tail,
)
from kernel.moe.backends.halops import run_mega_ll_moe


MOE_MODES = ("w8a8-blockwise", "w8a8-mxfp8", "w4a8-mxfp4")
MOE_SHAPE = dict(
    hidden_size=4096, intermediate_size=512, latent_size=2048, num_experts=512
)


@dataclass
class MoeInput:
    """Inputs with quantized weights and scales before layout conversion."""

    input: torch.Tensor
    weights: dict[str, torch.Tensor]
    options: dict
    routing: dict[str, torch.Tensor | None]


@dataclass
class MoeTiming:
    timing: TimingResult
    bytes_io: int
    active_experts: int | None = None
    # tp_size > 1: per-rank kernel time, ordered by rank.  The median is the
    # rank's typical launch and still contains the wait for a late peer; the
    # min is the same rank's best launch, i.e. its own kernel time.
    rank_median_us: list[float] | None = None
    rank_min_us: list[float] | None = None


def _validate_config(cfg: dict) -> dict:
    options = {
        **MOE_SHAPE,
        "batch_size": 1,
        "top_k": 8,
        "tp_size": 1,
        "quant_mode": "w8a8-blockwise",
        "scoring_func": "sigmoid",
        "renormalize": True,
        "activation": "silu",
        "situ_beta": 4.0,
        "situ_linear_beta": 25.0,
        "eps": 1e-5,
        "grid_cap": 0,
        "seed": 0,
        "routing": "bias+perexpert",
        **cfg,
    }
    for name, value in MOE_SHAPE.items():
        if options[name] != value:
            raise ValueError(f"MegaLLMoE requires {name}={value}")
    parts = set(filter(None, str(options["routing"]).split("+")))
    if not parts <= {"bias", "perexpert"}:
        raise ValueError(
            f"routing must combine 'bias' and 'perexpert', got {options['routing']!r}"
        )
    if options["tp_size"] not in (1, 4):
        raise ValueError(f"tp_size must be 1 or 4, got {options['tp_size']}")
    if options["tp_size"] > visible_device_count():
        raise ValueError(
            f"tp_size={options['tp_size']} needs that many visible GPUs, "
            f"got {visible_device_count()} (set --gpu accordingly)"
        )
    if (
        not isinstance(options["batch_size"], int)
        or not 1 <= options["batch_size"] <= 16
    ):
        raise ValueError("batch_size (tokens) must be an integer in [1, 16]")
    if not isinstance(options["top_k"], int) or not 1 <= options["top_k"] <= 10:
        raise ValueError("top_k must be an integer in [1, 10]")
    if options["quant_mode"] not in MOE_MODES:
        raise ValueError(f"quant_mode must be one of {MOE_MODES}")
    if options["scoring_func"] not in ("sigmoid", "softmax"):
        raise ValueError("scoring_func must be sigmoid or softmax")
    if options["activation"] not in ("silu", "situ"):
        raise ValueError("activation must be silu or situ")
    if not isinstance(options["renormalize"], bool):
        raise ValueError("renormalize must be a boolean")
    if not all(
        math.isfinite(options[key]) for key in ("eps", "situ_beta", "situ_linear_beta")
    ):
        raise ValueError("eps, situ_beta and situ_linear_beta must be finite")
    if options["eps"] <= 0 or options["grid_cap"] < 0:
        raise ValueError("eps must be positive and grid_cap must be nonnegative")
    return options


@torch.no_grad()
def _build_inputs(cfg: dict, tp_rank: int = 0) -> MoeInput:
    """The launch's tensors; with tp_size > 1 the expert and shared weights are
    the rank's own, while the router, latent and tail weights are shared."""
    options = _validate_config(cfg)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda")
    rank_gen = torch.Generator(device=device).manual_seed(options["seed"] + tp_rank)
    common_gen = (
        rank_gen
        if options["tp_size"] == 1
        else torch.Generator(device=device).manual_seed(options["seed"])
    )

    def normal(shape, scale=1.0, gen=None):
        return torch.randn(
            shape,
            device=device,
            dtype=torch.bfloat16,
            generator=rank_gen if gen is None else gen,
        ).mul_(scale)

    h, n, z, e = (options[key] for key in MOE_SHAPE)
    x = normal((options["batch_size"], h), gen=common_gen)
    weights = {
        "router_weight": normal((2, e, h), h**-0.5, common_gen),
        "expert_up_weights": normal((e, 2 * n, z), z**-0.5),
        "expert_down_weights": normal((e, z, n), n**-0.5),
        "shared_up_weight": normal((2 * n, h), h**-0.5),
        "shared_down_weight": normal((h, n), n**-0.5),
        "latent_down_weight": normal((z, h), h**-0.5, common_gen),
        "tail_fc2_weight": normal((h, z), z**-0.5, common_gen),
        "tail_norm_weight": normal((z,), 0.1, common_gen).add_(1.0),
    }
    # Router steering stays neutral: a zero bias keeps the selection (and the
    # active-expert count) unchanged, a unit scale keeps the combine weights.
    parts = set(filter(None, str(options["routing"]).split("+")))
    routing = {
        "correction_bias": (torch.zeros(e, device=device) if "bias" in parts else None),
        "per_expert_scale": (
            torch.ones(e, device=device) if "perexpert" in parts else None
        ),
    }
    return MoeInput(
        x, _quantize_weights(weights, options["quant_mode"]), options, routing
    )


def _quant_fp8_blockwise(weight: torch.Tensor, power_of_two=False):
    """Quantize 128x128 blocks, optionally rounding scales up to powers of two."""
    rows, cols = weight.shape[-2:]
    blocks = weight.float().reshape(
        *weight.shape[:-2], rows // 128, 128, cols // 128, 128
    )
    scale = blocks.abs().amax(dim=(-3, -1), keepdim=True).clamp_min(1e-12) / 448.0
    if power_of_two:
        scale = torch.exp2(torch.ceil(torch.log2(scale)))
    quant = (blocks / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return quant.reshape_as(weight), scale.squeeze(-3).squeeze(-1).contiguous()


@torch.no_grad()
def _quantize_weights(weights: dict, quant_mode: str) -> dict:
    from halops import group_wise_quant

    quantized = {}
    for name, weight in weights.items():
        if name in ("router_weight", "tail_norm_weight"):
            quantized[name] = weight
            continue
        if name == "tail_fc2_weight":
            # The tail's fc2 follows the latent recipe.
            if quant_mode == "w8a8-blockwise":
                quant, scale = _quant_fp8_blockwise(weight, power_of_two=True)
            else:
                quant, scale = group_wise_quant(
                    weight.flatten(0, -2), mode="w8a8-mxfp8", use_swizzle=False
                )
                quant = quant.reshape(*weight.shape[:-1], quant.shape[-1])
                scale = scale.reshape(*weight.shape[:-1], weight.shape[-1] // 32)
        elif quant_mode == "w8a8-blockwise":
            quant, scale = _quant_fp8_blockwise(weight)
        else:
            mode = (
                "w4a4-mxfp4"
                if name.startswith("expert_") and quant_mode == "w4a8-mxfp4"
                else "w8a8-mxfp8"
            )
            quant, scale = group_wise_quant(
                weight.flatten(0, -2),
                mode=mode,
                # The latent projection takes the natural scale rows.
                # The latent and shared scales stay plain row-major.
                use_swizzle=not (
                    name == "latent_down_weight" or name.startswith("shared_")
                ),
            )
            quant = quant.reshape(*weight.shape[:-1], quant.shape[-1])
            scale = scale.reshape(*weight.shape[:-1], weight.shape[-1] // 32)
        quantized[name] = quant
        quantized[
            "tail_fc2_scale" if name == "tail_fc2_weight" else name + "_scale"
        ] = scale
    return quantized


@torch.no_grad()
def adapt_halops(raw: MoeInput, tp: RankContext) -> dict:
    from halops import get_workspace_for_moe

    quantized = raw.weights
    return {
        **raw.options,
        **quantized,
        **raw.routing,
        "input": raw.input,
        "out": torch.empty_like(raw.input),
        "num_router_gates": 2,
        "workspace": get_workspace_for_moe(
            raw.input, tp.group_name, tp.rank, raw.options["tp_size"]
        ),
    }


@torch.no_grad()
def _active_experts(raw: MoeInput) -> int:
    return int(
        _route(
            {
                **raw.options,
                **raw.routing,
                "input_bf16": raw.input,
                "router_weight_fp32": raw.weights["router_weight"].float(),
            }
        )[1]
        .unique()
        .numel()
    )


@torch.no_grad()
def adapt_golden(raw: MoeInput, tp: RankContext) -> dict:
    weights = {}
    for name, weight in raw.weights.items():
        if name.endswith("_scale"):
            continue
        if name in ("router_weight", "tail_norm_weight"):
            weights[name] = weight
        else:
            scale_name = (
                "tail_fc2_scale" if name == "tail_fc2_weight" else name + "_scale"
            )
            weights[name] = _dequant_weight(
                weight,
                raw.weights[scale_name],
                sfatom=not (
                    name == "tail_fc2_weight"
                    or name == "latent_down_weight"
                    or name.startswith("shared_")
                ),
            )
    return {
        **raw.options,
        **raw.routing,
        **weights,
        "input_bf16": raw.input,
        "router_weight_fp32": raw.weights["router_weight"].float(),
    }


def _get_backends(names):
    registry = {
        "halops": (adapt_halops, run_mega_ll_moe),
        "golden": (adapt_golden, run_mega_ll_moe_golden),
    }
    unknown = set(names) - registry.keys()
    if unknown:
        raise ValueError(f"unknown MoE backends: {sorted(unknown)}")
    return {name: registry[name] for name in names}


def _rank_body(tp: RankContext, mode: str, job: dict) -> dict | None:
    """This rank's share of a launch; rank 0 returns the row."""
    if mode == "refcheck":
        return _refcheck_rank(tp, job["cfg"], job["kwargs"])
    if mode == "profile":
        return _profile_rank(tp, job["cfg"])
    return _bench_rank(tp, job["cfg"], job["name"], job["kwargs"])


@torch.no_grad()
def _bench_rank(
    tp: RankContext, cfg: dict, name: str, timing_kwargs: dict
) -> dict | None:
    raw = _build_inputs(cfg, tp.rank)
    adapt_fn, run_fn = _get_backends([name])[name]
    inputs = adapt_fn(raw, tp)
    unique_experts = _active_experts(raw)
    bytes_io = _bytes_io(inputs, cfg["num_experts"], name, unique_experts)
    del raw
    tp.sync()
    # Re-align the ranks before every measured iteration: without it the four
    # processes drift apart and the kernel's tail AllReduce turns that drift
    # into wait time inside the measured kernel duration.
    timing = bench_gpu_time(
        run_fn,
        (inputs,),
        before_iter=tp.sync if tp.world > 1 else None,
        **timing_kwargs,
    )
    tp.sync()
    # The row shows the worst rank; the per-rank medians make the spread (and
    # the launch-skew wait it carries) visible in the log.
    timings = tp.gather_timings(timing)
    timing = max(timings, key=lambda item: statistics.median(item.device_us))
    if not tp.is_leader:
        return None
    return {
        "device_us": [float(v) for v in timing.device_us],
        "host_us": [float(v) for v in timing.host_us],
        "dispatch_us": [float(v) for v in timing.dispatch_us],
        "bytes_io": int(bytes_io),
        "active_experts": int(unique_experts),
        "rank_median_us": [float(statistics.median(t.device_us)) for t in timings],
        "rank_min_us": [float(min(t.device_us)) for t in timings],
    }


@torch.no_grad()
def _refcheck_rank(tp: RankContext, cfg: dict, kwargs: dict) -> dict | None:
    atol, rtol = kwargs["atol"], kwargs["rtol"]
    raw = _build_inputs(cfg, tp.rank)
    golden_inputs = adapt_golden(raw, tp)
    halops_inputs = adapt_halops(raw, tp)
    out = run_mega_ll_moe(halops_inputs)
    del halops_inputs
    routed, shared = routed_and_shared(golden_inputs)
    reference = tail(
        golden_inputs,
        tp.sum(routed.float()),
        tp.sum(shared.float()),
    )
    if not torch.isfinite(reference).all() or reference.abs().max() < atol:
        raise RuntimeError("BF16 golden output is non-finite or degenerate")
    result = tp.worst_refcheck(
        _compare_to_reference(
            "halops",
            "golden",
            out.float().flatten(),
            reference.float().flatten(),
            atol,
            rtol,
        )
    )
    if not tp.is_leader:
        return None
    return {
        "halops": result._asdict(),
        "golden": RefcheckResult(passed=True)._asdict(),
    }


@torch.no_grad()
def _refcheck_entry(
    cfg, ordered_names, refcheck_reference, refcheck_atol, refcheck_rtol
):
    options = _validate_config(cfg)
    with tensor_parallel_context(options["tp_size"], __file__) as tp:
        payload = tp.run(
            "refcheck",
            {
                "cfg": {**cfg, **options},
                "kwargs": {"atol": refcheck_atol, "rtol": refcheck_rtol},
            },
            _rank_body,
        )
    return {name: RefcheckResult(**vals) for name, vals in payload.items()}


def _bytes_io(inputs: dict, num_experts: int, name: str, unique_experts: int) -> int:
    """Per-GPU bytes touched: the weights it reads plus the input and output."""
    bytes_io = inputs["input"].numel() * inputs["input"].element_size() * 2
    for key, value in inputs.items():
        if not isinstance(value, torch.Tensor) or not (
            "weight" in key or key == "tail_fc2_scale"
        ):
            continue
        if name == "golden" and key == "router_weight":
            continue
        count = value.numel()
        if key.startswith("expert_"):
            count = count // num_experts * unique_experts
        bytes_io += count * value.element_size()
    return bytes_io


@torch.no_grad()
def _bench_entry(
    cfg, name, dry_run_iters, repeat_iters, cold_l2_cache, use_cuda_graph, timing_method
):
    options = _validate_config(cfg)
    if options["tp_size"] > 1 and name != "halops":
        raise ValueError("tp_size > 1 benchmarks the halops backend only")
    with tensor_parallel_context(options["tp_size"], __file__) as tp:
        payload = tp.run(
            "bench",
            {
                "cfg": {**cfg, **options},
                "name": name,
                "kwargs": {
                    "dry_run_iters": dry_run_iters,
                    "repeat_iters": repeat_iters,
                    "cold_l2_cache": cold_l2_cache,
                    "use_cuda_graph": use_cuda_graph,
                    "method": timing_method,
                },
            },
            _rank_body,
        )
    timing = TimingResult(
        device_us=payload["device_us"],
        host_us=payload["host_us"],
        dispatch_us=payload["dispatch_us"],
    )
    return MoeTiming(
        timing,
        payload["bytes_io"],
        payload["active_experts"],
        payload.get("rank_median_us"),
        payload.get("rank_min_us"),
    )


def _rank_profile_path(profile_output: str, rank: int, world: int) -> str:
    """Where one rank writes its timer export.

    A single-GPU profile writes the requested file; a tensor-parallel one
    writes one file per rank next to it, which `_run_profile` then merges.
    """
    if world == 1:
        return profile_output
    path = Path(profile_output)
    return str(path.with_name(f"{path.stem}_r{rank}{path.suffix}"))


@torch.no_grad()
def _profile_rank(tp: RankContext, options: dict) -> dict | None:
    """This rank's launch under the timer; rank 0 returns the row."""
    raw = _build_inputs(options, tp.rank)
    inputs = adapt_halops(raw, tp)
    active_experts = _active_experts(raw)
    del raw
    # The first barrier lines the ranks up for the warmups -- the fused tail's
    # AllReduce is a rendezvous, so they then stay in lockstep; the second one
    # (before_launch, after each rank's own L2 flush) aligns the launch itself.
    tp.sync()
    profile_kernel_timeline(
        run_mega_ll_moe,
        (inputs,),
        _rank_profile_path(options["profile_output"], tp.rank, tp.world),
        warmup_iters=max(0, options.get("dry_run_iters", 5)),
        warp_roles={
            0: "calc tmem",
            1: "calc",
            2: "calc",
            3: "calc",
            4: "calc",
            5: "calc",
            6: "calc",
            7: "calc",
            8: "mma",
            9: "tma weight",
            10: "tma other",
            11: "idle",
        },
        cold_l2_cache=options.get("cold_l2_cache", True),
        before_launch=tp.sync,
        launch_metadata={
            "config": options.get("profile_config", ""),
            "cold_l2_cache": options.get("cold_l2_cache", True),
            "tokens": options["batch_size"],
            "top_k": options["top_k"],
            "experts": options["num_experts"],
            "active_experts": active_experts,
            "intermediate": options["intermediate_size"],
            "hidden": options["hidden_size"],
            "latent": options["latent_size"],
            "quant_mode": options["quant_mode"],
            "activation": options["activation"],
            "tp_size": options["tp_size"],
            "rank": tp.rank,
        },
    )
    if not tp.is_leader:
        return None
    return {"active_experts": active_experts}


@torch.no_grad()
def _run_profile(cfg: dict) -> None:
    """Profile one launch of a config.

    With tp_size=4 every rank exports its own timer pull and the ranks are
    merged into the requested file, so one timeline holds all four GPUs.
    """
    if cfg.get("use_cuda_graph", False):
        raise ValueError("MoE kernel timer profiling does not support CUDA graph")

    options = _validate_config(
        {
            **cfg,
            "batch_size": max(cfg.get("batch_sizes", [cfg.get("batch_size", 1)])),
            "top_k": cfg.get("top_k_list", [cfg.get("top_k", 8)])[0],
            # The rank workers run from benchmark/, so the path must be absolute.
            "profile_output": str(Path(cfg["profile_output"]).expanduser().resolve()),
        }
    )
    with tensor_parallel_context(options["tp_size"], __file__) as tp:
        tp.run("profile", {"cfg": options}, _rank_body)
    if options["tp_size"] == 1:
        return

    from halops.moe.device_timer import merge_traces

    rank_paths = [
        _rank_profile_path(options["profile_output"], rank, options["tp_size"])
        for rank in range(options["tp_size"])
    ]
    merge_traces(rank_paths, options["profile_output"])
    for path in rank_paths:
        Path(path).unlink(missing_ok=True)


def run_scenario(cfg: dict) -> None:
    """Benchmark the full fused operator on one GPU."""
    if cfg.get("profile", False):
        _run_profile(cfg)
        return

    backend_names = list(dict.fromkeys(cfg.get("backends", ["halops"])))
    _get_backends(backend_names)
    if not backend_names:
        raise ValueError("backends must not be empty")
    backend_names.sort(key=lambda name: name != "halops")
    if cfg.get("refcheck_reference", "golden") != "golden":
        raise ValueError("MoE refcheck_reference must be golden (BF16)")
    if (
        cfg.get("use_cuda_graph", False)
        and cfg.get("timing_method", "profiler") != "cupti"
    ):
        raise ValueError("use_cuda_graph requires --timing-method=cupti")
    if cfg.get("tp_size", 1) > 1 and cfg.get("use_cuda_graph", False):
        raise ValueError("tp_size > 1 does not support CUDA graph timing")
    if cfg.get("tp_size", 1) > 1 and cfg.get("timing_method", "profiler") == "profiler":
        print(
            "[WARN] tp_size>1 + timing_method='profiler': torch.profiler's "
            "per-iteration overhead perturbs the cross-rank launch cadence, so "
            "each rank's kernel duration picks up the wait for its peers. Use "
            "--timing-method=cupti for stable per-rank kernel times."
        )
    if cfg.get("dry_run_iters", 10) < 1 or cfg.get("repeat_iters", 50) < 1:
        raise ValueError("dry_run_iters and repeat_iters must be positive")
    failed = []
    for top_k in cfg.get("top_k_list", [cfg.get("top_k", 8)]):
        local = _validate_config({**cfg, "top_k": top_k})
        do_refcheck = local.get("refcheck", True)
        results = {}
        if do_refcheck and "halops" in backend_names:
            results = run_refcheck_cross_backends(
                _refcheck_entry,
                local,
                backend_names,
                "golden",
                refcheck_atol=local.get("refcheck_atol", 0.15),
                refcheck_rtol=local.get("refcheck_rtol", 0.15),
                use_subprocess=local.get("use_subprocess", True),
                timeout=local.get("timeout", 300),
            )
        baseline_us = 0.0
        for index, name in enumerate(backend_names):
            status, timing = run_bench_for_backend(
                _bench_entry,
                name,
                local,
                local.get("dry_run_iters", 10),
                local.get("repeat_iters", 50),
                local.get("cold_l2_cache", True),
                local.get("use_cuda_graph", False),
                local.get("timing_method", "profiler"),
                use_subprocess=local.get("use_subprocess", True),
                timeout=local.get("timeout", 300),
            )
            metrics = BenchmarkMetrics()
            if status == "OK":
                metrics = compute_moe_metrics(
                    timing.timing,
                    local["batch_size"],
                    top_k,
                    **MOE_SHAPE,
                    dtype_bytes=2 if name == "golden" else 1,
                    baseline_us=baseline_us,
                    gpu_type=local.get("gpu_type", "b200"),
                    bytes_io=timing.bytes_io,
                )
                if name == "halops":
                    baseline_us = metrics.median_us
            check = results.get(name)
            if status != "OK":
                failed.append(
                    f"tokens={local['batch_size']}, top_k={top_k}, {name}: {status}"
                )
            row = MoeRowResult(
                scenario="mega_ll_moe",
                backend=name,
                tokens=local["batch_size"],
                top_k=top_k,
                **MOE_SHAPE,
                active_experts=timing.active_experts if timing else None,
                tp_size=local["tp_size"],
                quant_mode=local["quant_mode"],
                dtype="bfloat16" if name == "golden" else local["quant_mode"],
                activation=local["activation"],
                scoring_func=local["scoring_func"],
                renormalize=local["renormalize"],
                grid_cap=local["grid_cap"],
                use_cuda_graph=local.get("use_cuda_graph", False),
                metrics=metrics,
                refcheck_pass=check.passed if check else None,
                case_tag=local.get("case_tag", ""),
                status=status,
                mismatch_pct=check.mismatch_pct if check else None,
                max_atol=check.max_atol if check else None,
                avg_atol=check.avg_atol if check else None,
                refcheck_atol=local.get("refcheck_atol", 0.15),
                refcheck_rtol=local.get("refcheck_rtol", 0.15),
            )
            if index == 0:
                print_scenario_header(row)
                print_table_header()
            print_table_row(row)
            if (
                status == "OK"
                and timing
                and local["tp_size"] > 1
                and timing.rank_median_us
            ):
                medians = " ".join(
                    f"r{rank}={us:.1f}" for rank, us in enumerate(timing.rank_median_us)
                )
                mins = " ".join(
                    f"r{rank}={us:.1f}" for rank, us in enumerate(timing.rank_min_us)
                )
                print(
                    f"    tp={local['tp_size']} kernel time per rank:"
                    f" median {medians} | min {mins}"
                    " (row above: worst median; min = no peer wait)"
                )
            append_moe_csv(row, local.get("output_path", "results/mega_ll_moe.csv"))
    if failed:
        raise RuntimeError("MoE benchmark failed: " + "; ".join(failed))


if __name__ == "__main__":
    tp_worker_main(_rank_body)
