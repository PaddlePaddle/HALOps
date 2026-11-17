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
    """Workspace of ``fused_all_reduce_mhc``, from ``get_workspace_for_mhc``."""

    peer_ptrs: torch.Tensor
    counters: torch.Tensor
    slot_bytes: int
    tp_rank: int
    _keepalive: tuple


def get_workspace_for_mhc(
    max_tokens: int,
    hidden: int,
    group_name: str,
    tp_rank: int,
    tp_size: int,
) -> MhcWorkspace:
    """Allocate the all-reduce workspace of ``fused_all_reduce_mhc``.

    With ``tp_size > 1`` this is a collective call on every rank of
    ``group_name``. Allocate it once before any CUDA-graph capture and reuse
    it for all calls.

    Args:
        max_tokens: Largest T of a call.
        hidden: Hidden size of ``x``.
        group_name: Name of the tensor-parallel process group; unused when
            ``tp_size`` is 1.
        tp_rank: Rank in that group.
        tp_size: Size of that group, at most 8.

    Returns:
        The workspace to pass to ``fused_all_reduce_mhc``.
    """
    device = torch.device("cuda", torch.cuda.current_device())
    slot_bytes = max_tokens * hidden * 2
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
            _keepalive=(region,),
        )

    from torch._C._distributed_c10d import _SymmetricMemory

    with torch.inference_mode(False):
        region = _SymmetricMemory.empty_strided_p2p(
            (nbytes,), [1], torch.uint8, device, group_name
        )
    handle = _SymmetricMemory.rendezvous(region)
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
    """All-reduce + mHC post + mHC pre + RMSNorm + W8A8-MXFP8 quantization.

    Fuses the end of one mHC sub-layer with the start of the next. For every
    token: ``residual_out[j] = bf16(post[j] * x + sum_i comb[i, j] *
    residual[i])``; the mix ``fn @ residual_out`` with its RMS gives ``pre``,
    ``post_out`` (sigmoid) and ``comb_out`` (Sinkhorn); ``layer_out =
    RMSNorm(bf16(sum_j pre[j] * residual_out[j]))`` is quantized to MXFP8.
    With ``workspace``, ``x`` is first summed over the tensor-parallel ranks.
    All tensors must be contiguous.

    Args:
        x: [T, hidden] bfloat16 layer output, this rank's partial sum with
            ``workspace``. hidden must be 4096.
        residual: [T, hc, hidden] bfloat16 residual streams; hc is 2 or 4.
        post: [T, hc] float32 post weights.
        comb: [T, hc, hc] float32 comb matrices, [source, dest].
        fn: [hc * (hc + 2), hc * hidden] float32 mix weights.
        hc_scale: [3] float32 scales of the pre, post and comb mixes.
        hc_base: [hc * (hc + 2)] float32 mix biases.
        norm_weight: [hidden] bfloat16 RMSNorm weight.
        rms_eps: Epsilon of the mix RMS, ``1 / (sqrt(mean(z^2)) + rms_eps)``.
        hc_pre_eps: Added to ``pre``.
        hc_sinkhorn_eps: Sinkhorn epsilon.
        hc_post_mult: Scale of ``post_out``.
        sinkhorn_repeat: Sinkhorn iterations.
        norm_eps: Epsilon of the RMSNorm.
        workspace: From ``get_workspace_for_mhc``; ``None`` skips the
            all-reduce.
        comb_is_logits: ``comb`` holds logits, normalized in the kernel.
        return_logits: Return the comb logits as ``comb_out``; normalize
            them with ``mhc_sinkhorn``.
        use_pdl: Enable Programmatic Dependent Launch (on by default).

    Returns:
        ``(residual_out, layer_out, post_out, comb_out, q, sf)``: bfloat16
        [T, hc, hidden] and [T, hidden], float32 [T, hc] and [T, hc, hc],
        float8_e4m3fn [T, hidden] codes and uint8 [T, hidden / 32] UE8M0
        scales in row-major layout.
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
    """Sinkhorn normalization of the comb logits of ``fused_all_reduce_mhc``.

    Turns the ``comb_out`` returned with ``return_logits`` into the matrices
    returned without it, and prefetches ``prefetch`` into L2 meanwhile.

    Args:
        logits: [T, hc, hc] float32 contiguous logits; hc is 2 or 4.
        eps: Sinkhorn epsilon (``hc_sinkhorn_eps``).
        repeat: Sinkhorn iterations (``sinkhorn_repeat``).
        out: Optional pre-allocated output shaped like ``logits``.
        prefetch: Up to 8 CUDA tensors, each under 4 GiB.

    Returns:
        [T, hc, hc] float32 comb matrices (``out`` when given).

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
