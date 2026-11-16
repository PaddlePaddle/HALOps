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

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch

import halops.halops_kernel as halops_kernel

__all__ = [
    "MhcWorkspace",
    "get_workspace_for_mhc",
    "fused_all_reduce_mhc",
    "mhc_sinkhorn",
]


@dataclass
class MhcWorkspace:
    """Lamport push buffers of the fused all-reduce in ``fused_all_reduce_mhc``.

    Every rank owns a buffer of two phases, each holding one slot of
    ``slot_bytes`` per source rank; ``peer_ptrs`` lists the buffers of all
    ranks in rank order and ``counters`` selects the phase of every CTA.
    """

    peer_ptrs: torch.Tensor
    counters: torch.Tensor
    slot_bytes: int
    tp_rank: int
    tp_size: int
    _keepalive: tuple


def get_workspace_for_mhc(
    max_tokens: int,
    hidden: int,
    group_name: str,
    tp_rank: int,
    tp_size: int,
    device: Optional[torch.device] = None,
) -> MhcWorkspace:
    """Allocate the workspace of the fused all-reduce in ``fused_all_reduce_mhc``.

    For tp_size > 1 the buffers are symmetric memory and the allocation is
    collective over ``group_name``. Allocate it once, before any CUDA-graph
    capture, and reuse it across calls and graph replays; it is returned
    zero-initialized.

    Args:
        max_tokens: Largest token count T of a call.
        hidden: Hidden size of ``x``.
        group_name: Name of the torch process group of the tensor-parallel
            ranks. It is unused for a single-rank launch.
        tp_rank: Rank in that group.
        tp_size: Size of that group, at most 8.
        device: CUDA device of the buffers (the current device by default).

    Returns:
        The workspace to pass to ``fused_all_reduce_mhc``.

    Example:
        >>> from halops import get_workspace_for_mhc
        >>>
        >>> workspace = get_workspace_for_mhc(16, 4096, "", 0, 1)
    """
    if not 1 <= tp_size <= 8:
        raise ValueError(f"tp_size must be in [1, 8], got {tp_size}")
    device = (
        torch.device("cuda", torch.cuda.current_device()) if device is None else device
    )
    slot_bytes = -(-max_tokens * hidden * 2 // 16) * 16
    nbytes = 2 * tp_size * slot_bytes
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    counters = torch.zeros(num_sms, dtype=torch.int32, device=device)

    if tp_size == 1:
        region = torch.zeros(nbytes, dtype=torch.uint8, device=device)
        return MhcWorkspace(
            peer_ptrs=torch.tensor(
                [region.data_ptr()], dtype=torch.int64, device="cpu"
            ),
            counters=counters,
            slot_bytes=slot_bytes,
            tp_rank=0,
            tp_size=1,
            _keepalive=(region,),
        )

    from torch._C._distributed_c10d import _SymmetricMemory

    with torch.inference_mode(False):
        region = _SymmetricMemory.empty_strided_p2p(
            (nbytes,), [1], torch.uint8, device, group_name
        )
    handle = _SymmetricMemory.rendezvous(region)
    if handle is None:
        raise RuntimeError("symmetric memory rendezvous failed")
    region.zero_()
    peers = [handle.get_buffer(r, [nbytes], torch.uint8) for r in range(tp_size)]
    peer_ptrs = torch.tensor(
        [b.data_ptr() for b in peers], dtype=torch.int64, device="cpu"
    )
    torch.cuda.synchronize()
    torch.distributed.barrier()
    return MhcWorkspace(
        peer_ptrs=peer_ptrs,
        counters=counters,
        slot_bytes=slot_bytes,
        tp_rank=tp_rank,
        tp_size=tp_size,
        _keepalive=(region, peers),
    )


def fused_all_reduce_mhc(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    norm_weight: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult: float,
    sinkhorn_repeat: int,
    norm_eps: float,
    workspace: Optional[MhcWorkspace] = None,
    comb_is_logits: bool = False,
    return_logits: bool = False,
    use_pdl: bool = True,
) -> Tuple[torch.Tensor, ...]:
    """mHC post mixing + all-reduce + mHC pre mixing + RMSNorm + MXFP8 quant.

    Joins the end of one mHC sub-layer to the start of the next in a single
    launch. With ``workspace``, ``x`` is first summed over the
    tensor-parallel ranks (fp32 in rank order, rounded to bf16). Then, per
    token and in fp32, with ``mix`` laid out as ``[pre (hc), post (hc),
    comb (hc * hc)]`` and ``g`` = 0, 1, 2 for these parts:

    - ``r_j = bf16(post[j] * x + sum_i comb[i, j] * residual[i])``;
    - ``z = r.flatten()``, ``rms = 1 / (sqrt(mean(z^2)) + rms_eps)`` and
      ``mix = (fn @ z) * rms * hc_scale[g] + hc_base``;
    - ``pre = sigmoid(mix_pre) + hc_pre_eps``,
      ``post_out = sigmoid(mix_post) * hc_post_mult`` and
      ``comb_out = sinkhorn(mix_comb)``: a row softmax plus
      ``hc_sinkhorn_eps``, a column normalization, then
      ``sinkhorn_repeat - 1`` rounds of row and column normalization, each
      dividing by ``sum + hc_sinkhorn_eps``;
    - ``y = bf16(sum_j pre[j] * r_j)`` and
      ``layer_out = bf16(y * rsqrt(mean(y^2) + norm_eps) * norm_weight)``,
      quantized to float8_e4m3fn with one UE8M0 scale per 32 elements.

    Args:
        x: [T, hidden] bfloat16 contiguous, 16-byte aligned layer output;
            this rank's partial sum when ``workspace`` is given. ``hidden``
            is 1024, 1536 or 4096.
        residual: [T, hc, hidden] bfloat16 contiguous, 16-byte aligned
            residual streams; ``hc`` is 2, 3 or 4.
        post: [T, hc] float32 contiguous post weights.
        comb: [T, hc, hc] float32 contiguous mixing matrices, or their
            logits with ``comb_is_logits``.
        fn: [hc * (hc + 2), hc * hidden] float32 contiguous mix weights.
        hc_scale: [3] float32 scales of the pre, post and comb mixes.
        hc_base: [hc * (hc + 2)] float32 mix biases.
        norm_weight: [hidden] bfloat16 RMSNorm weight.
        rms_eps: Epsilon of ``rms``.
        hc_pre_eps: Offset added to ``pre``.
        hc_sinkhorn_eps: Epsilon of the Sinkhorn normalization.
        hc_post_mult: Multiplier of ``post_out``.
        sinkhorn_repeat: Sinkhorn rounds, at least 1.
        norm_eps: Epsilon of the RMSNorm.
        workspace: Buffers from ``get_workspace_for_mhc``; ``None`` skips
            the all-reduce. With it, T is at most 4 times the number of
            co-resident clusters.
        comb_is_logits: ``comb`` holds logits; the Sinkhorn normalization
            runs in-kernel before mixing.
        return_logits: Return the comb logits instead of ``comb_out``; pass
            them to ``mhc_sinkhorn`` or back as ``comb`` with
            ``comb_is_logits``.
        use_pdl: Enable Programmatic Dependent Launch (on by default).

    Returns:
        A tuple ``(residual_out, layer_out, post_out, comb_out, q, sf)``:

        - ``residual_out``: [T, hc, hidden] bfloat16 streams ``r``.
        - ``layer_out``: [T, hidden] bfloat16.
        - ``post_out``: [T, hc] float32.
        - ``comb_out``: [T, hc, hc] float32 mixing matrices, or logits with
          ``return_logits``.
        - ``q``: [T, hidden] float8_e4m3fn codes of ``layer_out``.
        - ``sf``: [T, hidden / 32] uint8 UE8M0 scales in row-major layout.

    Example:
        >>> from halops import fused_all_reduce_mhc
        >>>
        >>> t, hc, hidden = 4, 4, 4096
        >>> mix = hc * (hc + 2)
        >>> x = torch.randn(t, hidden, dtype=torch.bfloat16, device="cuda")
        >>> residual = torch.randn(t, hc, hidden, dtype=torch.bfloat16,
        ...                        device="cuda")
        >>> post = torch.rand(t, hc, device="cuda")
        >>> comb = torch.rand(t, hc, hc, device="cuda").softmax(-1)
        >>> fn = torch.randn(mix, hc * hidden, device="cuda") * 1e-2
        >>> hc_scale = torch.ones(3, device="cuda")
        >>> hc_base = torch.zeros(mix, device="cuda")
        >>> w = torch.ones(hidden, dtype=torch.bfloat16, device="cuda")
        >>> outs = fused_all_reduce_mhc(x, residual, post, comb, fn, hc_scale,
        ...                             hc_base, w, 1e-6, 1e-6, 1e-6, 2.0, 20,
        ...                             1e-6)
    """
    peer_ptrs = counters = None
    slot_bytes = tp_rank = 0
    if workspace is not None:
        peer_ptrs, counters = workspace.peer_ptrs, workspace.counters
        slot_bytes, tp_rank = workspace.slot_bytes, workspace.tp_rank
    return tuple(
        halops_kernel.fused_all_reduce_mhc(
            x,
            residual,
            post,
            comb,
            fn,
            hc_scale,
            hc_base,
            norm_weight,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult,
            sinkhorn_repeat,
            norm_eps,
            comb_is_logits,
            return_logits,
            peer_ptrs,
            counters,
            slot_bytes,
            tp_rank,
            use_pdl,
        )
    )


def mhc_sinkhorn(
    logits: torch.Tensor,
    eps: float,
    repeat: int,
    out: Optional[torch.Tensor] = None,
    prefetch: Sequence[torch.Tensor] = (),
) -> torch.Tensor:
    """Sinkhorn normalization of mHC comb logits with optional L2 prefetch.

    Applies the ``comb_out`` normalization of ``fused_all_reduce_mhc`` to
    logits returned with ``return_logits``, one thread per token. While it
    runs, the grid prefetches ``prefetch`` into L2, e.g. the weights of the
    next ``fused_all_reduce_mhc`` when this kernel overlaps another one on a
    side stream.

    Args:
        logits: [T, hc, hc] float32 contiguous comb logits; ``hc`` is 2, 3
            or 4.
        eps: Epsilon of the normalization (``hc_sinkhorn_eps``).
        repeat: Sinkhorn rounds, at least 1.
        out: Optional pre-allocated [T, hc, hc] float32 contiguous output.
        prefetch: At most 8 contiguous CUDA tensors, each under 4 GiB.

    Returns:
        [T, hc, hc] float32 mixing matrices (``out`` when given).

    Example:
        >>> from halops import mhc_sinkhorn
        >>>
        >>> logits = torch.randn(4, 4, 4, device="cuda")
        >>> comb = mhc_sinkhorn(logits, 1e-6, 20)
    """
    if out is None:
        out = torch.empty_like(logits)
    halops_kernel.mhc_sinkhorn(logits, out, eps, repeat, list(prefetch))
    return out
