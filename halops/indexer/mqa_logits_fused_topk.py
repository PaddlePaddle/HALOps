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

from typing import Optional, Tuple

import torch

from halops.indexer.kernels.mqa_logits_fused_topk.config import (
    DEFAULT_KV_SAMPLE,
    SUPPORTED_HEAD_DIM,
    SUPPORTED_QKV_DTYPES,
    SamplingConfig,
    validate_num_heads,
    validate_query_shape,
    validate_topk,
    warmup_row_multiples,
)


def mqa_logits_fused_topk(
    q: torch.Tensor,
    q_sf: Optional[torch.Tensor],
    kv: Tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    ks: torch.Tensor,
    ke: torch.Tensor,
    topk_indices_offset: Optional[torch.Tensor],
    top_k: int,
    *,
    scores_bf16: bool,
    num_sms: int,
    kv_sample: tuple[int, int] = DEFAULT_KV_SAMPLE,
    ordered_bounds: bool = False,
    topk_indices: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return ragged top-k indices for the ``M = ks.numel()`` live query rows.

    Args:
        q: FP8/MXFP8 ``[M, H, 128]`` e4m3, or MXFP4 packed ``[M, H, 64]`` int8/uint8,
            where H is 8, 16, 32, or 64.
        q_sf: MXFP4/MXFP8 UE8M0 Q scales ``[M, H]`` int32, packing four scales per head,
            one for each 32 elements. None selects ordinary FP8.
        kv: FP8 ``(k [N, 128] e4m3, k_scale [N] fp32)`` or MXFP4 ``(k [N, 64]
            int8/uint8, k_sf [N] int32)`` or MXFP8 ``(k [N, 128] e4m3, k_sf [N]
            int32)``, gathered in batch order. Both MX formats use the same four packed
            UE8M0 scales per key.
        weights: ``[M, H]`` head weights (BF16 with ``scores_bf16``, else FP32), softmax
            scaling folded in. Ordinary FP8 also folds Q scales into weights;
            MXFP4/MXFP8 apply both Q and KV block scales in MMA.
        ks: ``[M]`` int32 absolute start row of each query's keys in ``kv``.
        ke: ``[M]`` int32 exclusive end of each query's keys; the caller supplies any
            causal or sliding-window exclusion in these bounds.
        topk_indices_offset: Optional contiguous ``[M]`` or ``[M, 1]`` int32/int64
            output offsets; None leaves indices relative to each row's first key.
            Padding remains -1.
        top_k: Number of indices selected per query, from 128 through 2048.
        scores_bf16: Whether score accumulation and thresholds use BF16 instead of FP32.
        num_sms: Number of SMs available to the persistent launch grid.
        kv_sample: Pilot ``(group, take)`` pair: score the first ``take`` consecutive
            keys of each ``group`` keys. This is independent of text-generation sampling
            parameters.
        ordered_bounds: Whether end bounds are nondecreasing within every query tile
            after window trimming, including on every graph replay.
        topk_indices: Optional ``[>=M, top_k]`` int32 output with contiguous columns;
            rows past M are untouched. None allocates ``[M, top_k]``.
    """
    is_mx_sf = q_sf is not None
    is_fp4 = q.dtype != torch.float8_e4m3fn
    if q.dtype not in (torch.int8, torch.uint8, torch.float8_e4m3fn):
        raise ValueError("q must be packed int8/uint8 MXFP4 or float8_e4m3fn FP8/MXFP8")
    if is_fp4 and not is_mx_sf:
        raise ValueError("MXFP4 requires q_sf")
    qkv_dtype = "mxfp4" if is_fp4 else "mxfp8" if is_mx_sf else "fp8"
    SamplingConfig(*kv_sample)
    validate_topk(top_k)
    validate_query_shape(q.shape, qkv_dtype)
    import tvm_ffi

    from halops.indexer.kernels.mqa_logits_fused_topk.pipeline import prepare

    num_rows = ks.shape[0]
    weights = weights[:num_rows].contiguous()
    k, k_scale = kv
    kv_dtypes = (torch.int8, torch.uint8) if is_fp4 else (torch.float8_e4m3fn,)
    if k.dtype not in kv_dtypes or k.ndim != 2 or k.shape[1] != q.shape[2]:
        raise ValueError("Q and KV must use the same quantized dtype and head width")
    launch = prepare(
        q=q[:num_rows].contiguous().view(torch.uint8),
        q_sf=q_sf[:num_rows].contiguous() if is_mx_sf else None,
        kv=k.view(torch.uint8),
        kv_sf=k_scale.contiguous(),
        weights=weights,
        ks=ks,
        ke=ke,
        max_kv_tokens=k.shape[0],
        qkv_dtype=qkv_dtype,
        top_k=top_k,
        scores_bf16=scores_bf16,
        num_sms=num_sms,
        topk_indices_offset=topk_indices_offset,
        topk_indices=topk_indices,
        kv_sample=kv_sample,
        ordered_bounds=ordered_bounds,
    )
    with tvm_ffi.use_torch_stream():
        launch.run()

    return launch.topk_indices


def warmup(
    *,
    qkv_dtype: str,
    scores_bf16: bool,
    num_heads: int,
    top_k: int,
    num_sms: int,
    kv_sample: tuple[int, int] = DEFAULT_KV_SAMPLE,
    ordered_bounds: bool = False,
    max_kv_lens: int = 32768,
) -> None:
    """Warm all tile and offset variants for contiguous outputs on the current device.

    Args:
        qkv_dtype: Input format, ``"fp8"``, ``"mxfp4"`` or ``"mxfp8"``.
        scores_bf16: Whether score accumulation and thresholds use BF16 instead of FP32.
        num_heads: Logical query head count: 8, 16, 32, or 64.
        top_k: Number of indices selected per query, from 128 through 2048.
        num_sms: Number of SMs available to the persistent launch grid.
        kv_sample: Pilot ``(group, take)`` pair: score the first ``take`` consecutive
            keys of each ``group`` keys.
        ordered_bounds: Whether end bounds are nondecreasing within every query tile
            after window trimming, including on every graph replay.
        max_kv_lens: maximum length of kv used for warm_up, default 32768.

    Notes:
        Nonempty pilot samples reuse these kernels across serving row counts; empty
        direct samples and nonstandard output strides require separate warmup before
        graph capture.
    """
    validate_num_heads(num_heads)
    validate_topk(top_k)
    if qkv_dtype not in SUPPORTED_QKV_DTYPES:
        raise ValueError(f"Unsupported Q/KV dtype: {qkv_dtype}")
    sampling = SamplingConfig(*kv_sample)
    device_index = torch.cuda.current_device()
    sms = max(1, num_sms)
    row_multiples = warmup_row_multiples(num_heads)
    rows = max(row_multiples) * sms
    # Ensure a nonempty pilot even for sparse sampling configurations.
    kv_rows = max(max_kv_lens, rows, sampling.group * (128 // sampling.take))
    is_fp4 = qkv_dtype == "mxfp4"
    is_mx_sf = qkv_dtype != "fp8"
    dim = SUPPORTED_HEAD_DIM // 2 if is_fp4 else SUPPORTED_HEAD_DIM
    device = torch.device("cuda", device_index)
    dtype = torch.uint8 if is_fp4 else torch.float8_e4m3fn
    q = torch.zeros((rows, num_heads, dim), dtype=dtype, device=device)
    k = torch.zeros((kv_rows, dim), dtype=dtype, device=device)
    q_sf = (
        torch.full((rows, num_heads), 0x7F7F7F7F, dtype=torch.int32, device=device)
        if is_mx_sf
        else None
    )
    k_scale = torch.full(
        (kv_rows,),
        0x7F7F7F7F if is_mx_sf else 1.0,
        dtype=torch.int32 if is_mx_sf else torch.float32,
        device=device,
    )
    weights = torch.ones(
        (rows, num_heads),
        dtype=torch.bfloat16 if scores_bf16 else torch.float32,
        device=device,
    )
    ks = torch.zeros(rows, dtype=torch.int32, device=device)
    ke = torch.full((rows,), kv_rows, dtype=torch.int32, device=device)
    topk_indices = torch.empty((rows, top_k), dtype=torch.int32, device=device)
    offsets = (None, ks, ks.to(torch.int64))
    for multiple in row_multiples:
        count = multiple * sms
        for offset in offsets:
            mqa_logits_fused_topk(
                q[:count],
                q_sf[:count] if q_sf is not None else None,
                (k, k_scale),
                weights[:count],
                ks[:count],
                ke[:count],
                offset[:count] if offset is not None else None,
                top_k,
                scores_bf16=scores_bf16,
                num_sms=num_sms,
                topk_indices=topk_indices[:count],
                kv_sample=kv_sample,
                ordered_bounds=ordered_bounds,
            )
    torch.cuda.synchronize(device)
