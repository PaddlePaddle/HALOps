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
# Implemented with reference to the PR of flashinfer (https://github.com/flashinfer-ai/flashinfer/pull/7325)
"""

import functools

import cutlass
import cutlass.cute as cute
import torch
from cutlass import testing

from . import stages
from .config import (
    DEFAULT_KV_SAMPLE,
    PILOT_ARGUMENT_COUNT,
    PRODUCER_ARGUMENT_COUNT,
    LaunchArg,
    SamplingConfig,
    select_config,
    validate_argument_count,
    validate_query_shape,
    validate_topk,
)
from .launch import KernelLaunch, SampledTopKLaunch


@functools.cache
def _compile_stage(name, attributes, signatures):
    """Compile and cache an auxiliary stage with dynamic row extents.

    Args:
        name: Class name exported by the ``stages`` module.
        attributes: Hashable ``(name, value)`` pairs restoring the stage's compile-time
            state.
        signatures: Per-input ``(dtype, ndim, last_width, row_stride)`` tuples, or None
            for absent inputs.
    """
    cls = getattr(stages, name)
    obj = cls.__new__(cls)
    obj.__dict__.update(dict(attributes))
    dtypes = {
        torch.uint8: cutlass.Uint8,
        torch.int32: cutlass.Int32,
        torch.int64: cutlass.Int64,
        torch.bfloat16: cutlass.BFloat16,
        torch.float32: cutlass.Float32,
    }
    fake = []
    for signature in signatures:
        if signature is None:
            fake.append(None)
            continue
        dtype, ndim, last, row_stride = signature
        if row_stride:
            # The top-k columns can be a view of the wider [M, K + W] output.
            fake.append(
                cute.runtime.make_fake_tensor(
                    dtypes[dtype], (cute.sym_int(), last), stride=(row_stride, 1)
                )
            )
            continue
        shape = (
            (cute.sym_int(),)
            if ndim == 1
            else (cute.sym_int(), cute.sym_int() if last == 0 else last)
        )
        fake.append(
            cute.runtime.make_fake_compact_tensor(
                dtypes[dtype], shape, stride_order=tuple(reversed(range(ndim)))
            )
        )
    return cute.compile(
        obj,
        *fake,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def _stage(obj, *args):
    """Bind auxiliary-stage tensors to a cached compiled launch.

    Args:
        obj: Stage instance whose attributes determine the kernel specialization.
        args: Tensor inputs and outputs in launch order, including optional None values.
    """
    signature = []
    for tensor in args:
        if tensor is None:
            signature.append(None)
            continue
        last = tensor.shape[-1] if tensor.ndim > 1 else 0
        row_stride = (
            tensor.stride(0) if tensor.ndim == 2 and tensor.stride(0) != last else 0
        )
        signature.append((tensor.dtype, tensor.ndim, last, row_stride))
    fn = _compile_stage(
        type(obj).__name__, tuple(sorted(obj.__dict__.items())), tuple(signature)
    )
    return KernelLaunch(fn, testing.JitArguments(*args))


def _prepare_output(*, num_rows, width, device, offsets, topk_indices):
    """Validate or allocate output storage and flatten optional index offsets.

    Args:
        num_rows: Number M of live query rows.
        width: Output column count, equal to top-k plus sliding-window width.
        device: Q's CUDA device, which output storage and offsets must share.
        offsets: Optional contiguous int32/int64 ``[M]`` or ``[M, 1]`` output offsets.
        topk_indices: Optional int32 ``[>=M, width]`` storage with contiguous columns
            and nonoverlapping rows.
    """
    if offsets is not None:
        if (
            offsets.shape not in ((num_rows,), (num_rows, 1))
            or offsets.dtype not in (torch.int32, torch.int64)
            or offsets.device != device
            or not offsets.is_contiguous()
        ):
            raise ValueError(
                "topk_indices_offset must be contiguous int32/int64 [M] or [M, 1] on Q's device"
            )
        offsets = offsets.view(num_rows)
    if topk_indices is None:
        topk_indices = torch.empty((num_rows, width), device=device, dtype=torch.int32)
    elif (
        topk_indices.ndim != 2
        or topk_indices.shape[0] < num_rows
        or topk_indices.shape[1] != width
        or topk_indices.dtype != torch.int32
        or topk_indices.device != device
        or topk_indices.stride(1) != 1
        or topk_indices.stride(0) < width
    ):
        raise ValueError(
            "topk_indices must be int32 [>=M, K + W] on Q's device with contiguous, non-overlapping rows"
        )
    return topk_indices, offsets


def _prepare_pilot(*, inputs, sampling, top_k, threshold):
    """Prepare optional KV gathering and the pilot threshold launch.

    Args:
        inputs: Common packed-input, bounds and launch options passed to
            ``pilot.prepare_launch``.
        sampling: SamplingConfig selecting direct reads or a gathered dense sample.
        top_k: Final output width used to derive the sampled pilot rank.
        threshold: Shared score buffer ``[M, 1]`` written by the pilot and read by the
            filter.
    """
    from . import pilot

    kv, scales = inputs["kv"], inputs["kv_sf"]
    launches = []
    if not sampling.uses_direct_sampling:
        # A single 4-byte scale is below TMA's 16-byte minimum; gather take < 4.
        rows = sampling.padded_sampled_rows(kv.shape[0])
        sampled_kv = torch.empty((rows, kv.shape[1]), device=kv.device, dtype=kv.dtype)
        sampled_scales = torch.empty((rows,), device=scales.device, dtype=scales.dtype)
        launches.append(
            _stage(
                stages.SampleKV(kv.shape[1], sampling.group, sampling.take),
                kv,
                scales,
                sampled_kv,
                sampled_scales,
            )
        )
        kv, scales = sampled_kv, sampled_scales
    elif sampling.direct_sampled_rows(kv.shape[0]) == 0:
        # Zero-extent TMA descriptors are invalid; exact filter/repair still run.
        return [_stage(stages.EmptyPilot(), threshold)]
    launch = pilot.prepare_launch(
        **dict(inputs, kv=kv, kv_sf=scales),
        sample_group=sampling.group,
        sample_take=sampling.take,
        top_k=sampling.get_sampling_rank(top_k, num_heads=inputs["q"].shape[1]),
        threshold=threshold,
    )
    validate_argument_count(launch.arguments.args, PILOT_ARGUMENT_COUNT, "pilot")
    launches.append(launch)
    return launches


def _prepare_success(*, filtered, top_k, num_heads, num_sms, offsets):
    """Prepare exact selection of complete rows and the failed-group queue.

    Args:
        filtered: Prepared filter launch owning candidates, counts, bounds and output
            storage.
        top_k: Number of indices selected per query, from 128 through 2048.
        num_heads: Logical query head count: 8, 16, 32, or 64.
        num_sms: Number of SMs available to the persistent launch grid.
        offsets: Optional int32/int64 ``[M]`` offsets added only to valid output
            indices; None keeps row-local indices and padding remains -1.
    """
    args = filtered.arguments.args
    rows = args[LaunchArg.NUM_QUERY_ROWS]
    block_q = select_config(rows, num_sms, num_heads=num_heads, repair=True).block_q
    config = select_config(rows, num_sms, num_heads=num_heads)
    return _stage(
        stages.SuccessTopK(
            top_k,
            args[LaunchArg.CANDIDATES].shape[1],
            block_q,
            partitions=4 if config.block_q > config.math_groups else 0,
        ),
        args[LaunchArg.CANDIDATES],
        args[LaunchArg.CANDIDATE_COUNTS],
        offsets,
        args[LaunchArg.TOPK_INDICES],
        args[LaunchArg.KEY_START],
        args[LaunchArg.KEY_END],
    )


def prepare(
    *,
    q,
    q_sf,
    kv,
    kv_sf,
    weights,
    ks,
    ke,
    max_kv_tokens,
    qkv_dtype,
    top_k,
    scores_bf16,
    num_sms=148,
    sliding_window=0,
    topk_indices_offset=None,
    topk_indices=None,
    ordered_bounds=False,
    kv_sample=DEFAULT_KV_SAMPLE,
):
    """Prepare the bounded pilot, filter, success-selection and exact-repair chain.

    Args:
        q: Uint8 Q byte view ``[M, H, 128]`` for FP8/MXFP8 or ``[M, H, 64]`` for MXFP4.
        q_sf: Int32 ``[M, H]`` packed UE8M0 Q scales for MXFP4/MXFP8, or None for FP8.
        kv: Uint8 KV byte view ``[N, 128]`` for FP8/MXFP8 or ``[N, 64]`` for MXFP4.
        kv_sf: Per-key ``[N]`` scales: FP32 for FP8 or packed int32 UE8M0 for
            MXFP4/MXFP8.
        weights: Head weights ``[M, H]`` with softmax scaling folded in (and Q scaling
            for ordinary FP8); FP32 for FP32 scores, cast to BF16 for BF16 scores.
        ks: Nonnegative int32 ``[M]`` inclusive key starts in full-KV coordinates.
        ke: Int32 ``[M]`` exclusive absolute key ends in full-KV coordinates, between
            each row's start and the full KV row count. Each row is masked
            independently, including partial query tiles.
        max_kv_tokens: Host bound on the full key extent, at least twice ``top_k``.
        qkv_dtype: Packed input format, ``"fp8"``, ``"mxfp4"`` or ``"mxfp8"``.
        top_k: Number of indices selected per query, from 128 through 2048.
        scores_bf16: Whether score arithmetic uses BF16 instead of FP32.
        num_sms: Number of SMs available to the persistent launch grid.
        sliding_window: Number W of newest keys excluded from top-k and appended in
            order; each output section pads with -1 and bounds refresh on every replay.
            Use zero with pre-trimmed bounds when the caller appends SWA.
        topk_indices_offset: Optional contiguous int32/int64 ``[M]`` or ``[M, 1]``
            offsets added to valid row-local indices; padding remains -1.
        topk_indices: Optional int32 ``[>=M, top_k + W]`` output with contiguous
            columns; only the first M rows are written. None allocates ``[M, top_k +
            W]``; the tensor is exposed as ``launch.topk_indices``.
        ordered_bounds: Whether end bounds are nondecreasing within every query tile
            after window trimming, including on every graph replay. False permits
            arbitrary row order; this flag is part of the compile key.
        kv_sample: Pilot ``(group, take)`` pair: score the first ``take`` consecutive
            keys of each ``group`` keys. A take of at least four permits direct TMA
            reads.
    """
    if sliding_window < 0:
        raise ValueError("sliding_window must be non-negative")
    validate_topk(top_k)
    validate_query_shape(q.shape, qkv_dtype)
    sampling = SamplingConfig(*kv_sample)
    topk_indices, topk_indices_offset = _prepare_output(
        num_rows=q.shape[0],
        width=top_k + sliding_window,
        device=q.device,
        offsets=topk_indices_offset,
        topk_indices=topk_indices,
    )
    live_out = topk_indices[: q.shape[0]]
    # One producer / pilot for both score dtypes; epilogue.py holds the
    # per-dtype epilogue policy.
    from . import producer

    if scores_bf16:
        # Cast once so pilot, filter and repair share the same weights.
        weights = weights.to(torch.bfloat16)
    if qkv_dtype == "fp8":
        # Keep the shared launch ABI; FP8 compiles out Q block-scale reads.
        q_sf = torch.empty(q.shape[:2], device=q.device, dtype=torch.int32)
    if sliding_window:
        user_ke = ke
        ke = stages.topk_end(ks, user_ke, sliding_window)
    raw = dict(
        q=q,
        q_sf=q_sf,
        kv=kv,
        kv_sf=kv_sf,
        weights=weights,
        ks=ks,
        ke=ke,
        max_kv_tokens=max_kv_tokens,
        scores_bf16=scores_bf16,
        num_sms=num_sms,
        qkv_dtype=qkv_dtype,
        ordered_bounds=ordered_bounds,
    )

    prepare_producer = functools.partial(
        producer.prepare_launch,
        **raw,
        top_k=top_k,
        topk_indices_offset=topk_indices_offset,
    )
    filtered = prepare_producer(topk_indices=live_out[:, :top_k])
    filter_args = filtered.arguments.args
    validate_argument_count(filter_args, PRODUCER_ARGUMENT_COUNT, "filter")
    # Both producers read the same live bounds; repair reuses filter workspace.
    repaired = prepare_producer(repair=True, workspace=filtered)
    validate_argument_count(repaired.arguments.args, PRODUCER_ARGUMENT_COUNT, "repair")
    launches = _prepare_pilot(
        inputs=dict(
            raw,
            kv=filter_args[LaunchArg.KV],
            kv_sf=filter_args[LaunchArg.KV_SCALES],
        ),
        sampling=sampling,
        top_k=top_k,
        threshold=filter_args[LaunchArg.LOGITS],
    )
    selected = _prepare_success(
        filtered=filtered,
        top_k=top_k,
        num_heads=q.shape[1],
        num_sms=num_sms,
        offsets=topk_indices_offset,
    )
    launches.extend((filtered, selected, repaired))
    if sliding_window:
        launches.insert(
            0,
            stages.prep_launch(
                ks,
                user_ke,
                filter_args[LaunchArg.KEY_START],
                filter_args[LaunchArg.KEY_END],
                live_out,
                q.shape[0],
                top_k,
                sliding_window,
                topk_indices_offset=topk_indices_offset,
            ),
        )
    return SampledTopKLaunch(launches, topk_indices)
