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

from collections.abc import Callable
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
import torch
from cutlass import testing

from .config import validate_query_shape
from .epilogue import score_epilogue


def prepare_input_views(*, q, q_sf, kv, kv_sf, weights, ks, ke, qkv_dtype, epi_dtype):
    """Validate stage inputs and return unpadded flattened Q-side views.

    Args:
        q: Uint8 Q byte view ``[M, H, 128]`` for FP8/MXFP8 or ``[M, H, 64]`` for MXFP4.
        q_sf: Int32 ``[M, H]`` packed UE8M0 Q scales for MXFP4/MXFP8; an unused tensor
            placeholder for FP8.
        kv: Uint8 key bytes using the same packed head width as Q.
        kv_sf: Per-key ``[N]`` scales: FP32 for FP8 or packed int32 UE8M0 for
            MXFP4/MXFP8.
        weights: Head weights ``[M, H]`` with softmax scaling folded in (and Q scaling
            for ordinary FP8); FP32 for FP32 scores, cast to BF16 for BF16 scores.
        ks: Nonnegative int32 ``[M]`` inclusive key starts in full-KV coordinates.
        ke: Int32 ``[M]`` exclusive absolute key ends in full-KV coordinates, between
            each row's start and the full KV row count.
        qkv_dtype: Packed input format, ``"fp8"``, ``"mxfp4"`` or ``"mxfp8"``.
        epi_dtype: Score arithmetic type, ``cutlass.Float32`` or ``cutlass.BFloat16``.
    """
    validate_query_shape(q.shape, qkv_dtype)
    rows, heads, width = q.shape
    is_mx_sf = qkv_dtype != "fp8"
    if (
        q.dtype != torch.uint8
        or kv.dtype != torch.uint8
        or kv.ndim != 2
        or kv.shape[1] != width
        or not q.is_contiguous()
        or not kv.is_contiguous()
    ):
        raise ValueError(
            "Q and KV byte views must be contiguous with matching head width"
        )
    if (
        kv_sf is None
        or kv_sf.shape != (kv.shape[0],)
        or kv_sf.dtype != (torch.int32 if is_mx_sf else torch.float32)
        or not kv_sf.is_contiguous()
    ):
        raise ValueError(
            "kv_sf must be contiguous [N]: packed int32 UE8M0 for MXFP4/MXFP8, "
            "float32 for ordinary FP8"
        )
    if is_mx_sf and (
        q_sf is None
        or q_sf.shape != (rows, heads)
        or q_sf.dtype != torch.int32
        or not q_sf.is_contiguous()
    ):
        raise ValueError("MXFP4/MXFP8 require contiguous int32 q_sf of shape [M, H]")
    for name, bounds in (("ks", ks), ("ke", ke)):
        if (
            bounds.shape != (rows,)
            or bounds.dtype != torch.int32
            or not bounds.is_contiguous()
        ):
            raise ValueError(
                f"{name} must be int32 of shape ({rows},); "
                f"got {bounds.dtype} {tuple(bounds.shape)}"
            )
    if weights.shape != (rows, heads) or (
        epi_dtype == cutlass.Float32 and weights.dtype != torch.float32
    ):
        raise ValueError("weights must be [M, H], with float32 dtype for FP32 scores")
    for tensor in (q_sf, kv, kv_sf, weights, ks, ke):
        if tensor is not None and tensor.device != q.device:
            raise ValueError("All inputs must be on Q's device")
    # Bounds stay unpadded: device reads clamp partial query tiles to M - 1.
    q_sf_flat, w_flat = score_epilogue(epi_dtype).q_side_host(q_sf, weights)
    return q.reshape(rows * heads, width, 1), q_sf_flat, w_flat


def make_input_fakes(num_heads, head_dim, qkv_dtype, epi_dtype, num_rows):
    """Build the five common compilation inputs with symbolic row counts.

    Args:
        num_heads: Logical query head count: 8, 16, 32, or 64.
        head_dim: Unpacked dimension of each query/key head, fixed at 128.
        qkv_dtype: Packed input format, ``"fp8"``, ``"mxfp4"`` or ``"mxfp8"``.
        epi_dtype: Score arithmetic type, ``cutlass.Float32`` or ``cutlass.BFloat16``.
        num_rows: Symbolic number of live query rows shared by Q, Q scales and weights.
    """
    width = head_dim // 2 if qkv_dtype == "mxfp4" else head_dim
    num_keys = cute.sym_int()
    kv = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (num_keys, width), stride_order=(1, 0)
    )
    kv_sf = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32 if qkv_dtype == "fp8" else cutlass.Int32,
        (num_keys,),
        stride_order=(0,),
    )
    query = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (num_rows * num_heads, width, 1), stride_order=(1, 0, 2)
    )
    q_sf, weights = score_epilogue(epi_dtype).q_side_fakes(num_heads, num_rows)
    return kv, kv_sf, query, q_sf, weights


def make_bounds_fakes():
    """Build two int32 bound vectors sharing a symbolic query-row extent."""
    rows = cute.sym_int()
    return tuple(
        cute.runtime.make_fake_compact_tensor(cutlass.Int32, (rows,), stride_order=(0,))
        for _ in range(2)
    )


@dataclass
class KernelLaunch:
    """A prepared kernel launch reusable for dispatch and timing."""

    compiled_fn: Callable[..., object]
    arguments: testing.JitArguments

    def run(self) -> None:
        """Dispatch the prepared kernel once."""
        self.compiled_fn(*self.arguments.args, **self.arguments.kwargs)


class SampledTopKLaunch:
    """An ordered launch sequence with one caller-visible output tensor."""

    def __init__(self, launches, indices):
        """Store an ordered kernel launch chain and its output tensor.

        Args:
            launches: Prepared kernel launches in execution order.
            indices: Caller-visible int32 top-k output, including any sliding-window
                columns.
        """
        self._launches = tuple(launches)
        self.topk_indices = indices

    def run(self) -> None:
        """Dispatch each prepared kernel in execution order."""
        for launch in self._launches:
            launch.run()

    def launches(self):
        """Return the chain's kernel launches in execution order."""
        return list(self._launches)
