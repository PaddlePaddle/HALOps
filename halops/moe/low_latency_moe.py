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
from typing import Optional, Sequence, Union

import torch

import halops.halops_kernel as halops_kernel
from halops.moe.low_latency_moe_utils import (
    _NARROW_TOKENS,
    _TAIL_ALIGN,
    _check_workspace,
    _interleave_up_for_pair_layout,
    _pack_router_weight,
    _prepare_latent_down_weight,
    _prepare_shared_expert_up_weight,
    _router_gates,
)

__all__ = [
    "MoEWorkspace",
    "low_latency_moe",
    "get_workspace_for_moe",
    "get_workspace_size_for_moe",
    "transform_weights_for_moe",
]


@dataclass
class MoEWorkspace:
    """Scratchpad and tail communication state of ``low_latency_moe``.

    The kernel has a narrow (1 <= T <= 8) and a wide (9 <= T <= 16)
    schedule; each has its own scratchpad and tail layout, so a workspace is
    bound to one of them.  ``buffer`` is the head of one NVLS multicast
    capable symmetric region whose tail part is the fused tail's mailbox.
    """

    buffer: torch.Tensor
    rank: int
    world: int
    wide: bool
    mc_base: int
    uc_bases: torch.Tensor
    epochs: torch.Tensor
    _keepalive: tuple


def _workspace_layout(input: Optional[torch.Tensor]):
    """Return (scratchpad bytes, tail offset, total bytes) of the symmetric region."""
    scratch = int(halops_kernel.moe_get_workspace_size(input))
    offset = -(-scratch // _TAIL_ALIGN) * _TAIL_ALIGN
    if input is None:
        tail = max(int(halops_kernel.moe_tail_region_bytes(w)) for w in (False, True))
    else:
        tail = int(halops_kernel.moe_tail_region_bytes(input.size(0) > _NARROW_TOKENS))
    return scratch, offset, offset + tail


def get_workspace_for_moe(
    input: torch.Tensor,
    group_name: str,
    rank: int,
    world: int,
) -> MoEWorkspace:
    """Allocate the workspace of ``low_latency_moe`` for ``input``'s token count.

    Collective over ``group_name``.  Allocate it once, before any CUDA-graph
    capture, and reuse it across calls and graph replays; it is returned
    zero-initialized.  A workspace must always be launched with the same
    grid.

    Args:
        input: [T, 4096] bfloat16 CUDA tensor; T selects the schedule.
        group_name: Name of the torch process group of the tensor-parallel
            ranks.
        rank: Rank in that group.
        world: Size of that group; this build requires 4.

    Returns:
        The workspace to pass to ``low_latency_moe``.
    """
    from torch._C._distributed_c10d import _SymmetricMemory

    wide = input.size(0) > _NARROW_TOKENS
    device = input.device
    scratch, offset, nbytes = _workspace_layout(input)
    with torch.inference_mode(False):
        region = _SymmetricMemory.empty_strided_p2p(
            (nbytes,), [1], torch.uint8, device, group_name
        )
    handle = _SymmetricMemory.rendezvous(region)
    if handle is None or handle.multicast_ptr == 0:
        raise RuntimeError("the fused tail needs an NVLS multicast-capable region")
    region.zero_()
    peers = [handle.get_buffer(r, [nbytes], torch.uint8) for r in range(world)]
    uc_bases = torch.tensor(
        [b.data_ptr() + offset for b in peers], dtype=torch.int64, device="cpu"
    )
    epochs = torch.zeros(1, dtype=torch.int32, device=device)
    torch.cuda.synchronize()
    torch.distributed.barrier()
    return MoEWorkspace(
        buffer=region[:scratch],
        rank=rank,
        world=world,
        wide=wide,
        mc_base=int(handle.multicast_ptr) + offset,
        uc_bases=uc_bases,
        epochs=epochs,
        _keepalive=(region, peers),
    )


def low_latency_moe(
    input: torch.Tensor,
    router_weight: torch.Tensor,
    expert_up_weights: torch.Tensor,
    expert_up_weights_scale: torch.Tensor,
    expert_down_weights: torch.Tensor,
    expert_down_weights_scale: torch.Tensor,
    shared_up_weight: torch.Tensor,
    shared_up_weight_scale: torch.Tensor,
    shared_down_weight: torch.Tensor,
    shared_down_weight_scale: torch.Tensor,
    latent_down_weight: torch.Tensor,
    latent_down_weight_scale: torch.Tensor,
    workspace: MoEWorkspace,
    shared_out: torch.Tensor,
    top_k: int,
    num_router_gates: int = 2,
    scoring_func: str = "sigmoid",
    renormalize: bool = True,
    activation: str = "silu",
    situ_beta: float = 4.0,
    situ_linear_beta: float = 25.0,
    *,
    fc2_weight: torch.Tensor,
    fc2_scale: torch.Tensor,
    norm_weight: torch.Tensor,
    out: torch.Tensor,
    eps: float = 1e-6,
    grid_cap: int = 0,
) -> None:
    """MoE forward pass on Blackwell SM100, support latent.

    Runs the router, the latent down-projection, the routed experts' fused
    up/down projections and the shared expert for 1..16 tokens, writing
    ``shared_out`` and ``out`` in place.  The shipped schedule is
    fixed to E=512 experts, hidden size 4096, intermediate size 512 and
    latent size 2048, with block-wise FP8 weights (128x128 blocks).

    Weights are block-wise quantized by the caller; the router, up-projection
    and latent-down operands must first be packed with
    ``transform_weights_for_moe``.  This build runs the latent stage, so
    ``latent_down_weight`` and ``latent_down_weight_scale`` are required.

    The fused tail always runs as the kernel's last phase (the same single
    launch)::

        out = fc2(RMSNorm(AllReduce(routed))) + AllReduce(shared)

    where ``routed`` is this rank's combined routed latent and ``shared`` its
    partial shared-expert output (``shared_out``, computed in-kernel).  Both
    AllReduces run over the tensor-parallel group of ``workspace`` (4 ranks, NVLS
    multicast).  A scratchpad must always be launched with the same grid, so
    keep ``grid_cap`` fixed per workspace; the tail needs the shared-expert
    blocks and its own CTAs inside the grid.

    Args:
        input: [T, 4096] bfloat16 activations, 1 <= T <= 16.
        router_weight: Packed router weight (bfloat16) from
            ``transform_weights_for_moe``.
        expert_up_weights: [512, 1024, 2048] float8_e4m3fn fused gate/up
            weights, in the pair layout returned by
            ``transform_weights_for_moe``.
        expert_up_weights_scale: [512, 8, 16] float32 block scales of
            ``expert_up_weights``.
        expert_down_weights: [512, 2048, 512] float8_e4m3fn down weights.
        expert_down_weights_scale: [512, 16, 4] float32 block scales of
            ``expert_down_weights``.
        shared_up_weight: [1024, 4096] float8_e4m3fn shared-expert gate/up
            weights, pair layout.
        shared_up_weight_scale: [8, 32] float32 block scales of
            ``shared_up_weight``.
        shared_down_weight: [4096, 512] float8_e4m3fn shared-expert down
            weights.
        shared_down_weight_scale: [32, 4] float32 block scales of
            ``shared_down_weight``.
        latent_down_weight: [2048, 4096] float8_e4m3fn latent
            down-projection weight, transposed by
            ``transform_weights_for_moe``.
        latent_down_weight_scale: [16, 32] float32 block scales of
            ``latent_down_weight``.
        workspace: Workspace from ``get_workspace_for_moe`` whose schedule
            width matches the token count (``T > 8`` needs the wide one).
        shared_out: [T, 4096] bfloat16 output of the shared expert (this rank's
            partial sum, which feeds the tail), written in place.
        top_k: Experts selected per token, in [1, 10].
        num_router_gates: Router gates merged into ``router_weight``; this
            build requires 2.
        scoring_func: Router scoring, ``"sigmoid"`` or ``"softmax"``.
        renormalize: Renormalize the top-k weights to sum to one.
        activation: Expert activation, ``"silu"`` or ``"situ"``.
        situ_beta: SiTU gate soft cap; <= 0 disables it.  Must be finite.
        situ_linear_beta: SiTU up soft cap; <= 0 disables it.  Must be
            finite.
        fc2_weight: [H, L] float8_e4m3fn, replicated on every rank.
        fc2_scale: [H/128, L/128] float32 block scales, powers of two.
        norm_weight: [L] bfloat16 RMSNorm weight.
        out: [T, H] bfloat16 result of the tail, identical on every rank.
        eps: RMSNorm epsilon.
        grid_cap: 0 uses one block per SM; otherwise at most this many
            blocks, leaving the other SMs to concurrent work.

    Returns:
        None. The outputs are written in place.
    """
    _check_workspace(workspace, input.size(0))
    halops_kernel.low_latency_moe(
        input,
        router_weight,
        expert_up_weights,
        expert_up_weights_scale,
        expert_down_weights,
        expert_down_weights_scale,
        shared_up_weight,
        shared_up_weight_scale,
        shared_down_weight,
        shared_down_weight_scale,
        latent_down_weight,
        latent_down_weight_scale,
        workspace.buffer,
        shared_out,
        fc2_weight,
        fc2_scale,
        norm_weight,
        out,
        workspace.epochs,
        workspace.uc_bases,
        top_k,
        num_router_gates,
        workspace.rank,
        workspace.mc_base,
        grid_cap,
        scoring_func,
        activation,
        renormalize,
        situ_beta,
        situ_linear_beta,
        eps,
    )


def get_workspace_size_for_moe(input: Optional[torch.Tensor] = None) -> int:
    """Return the size in bytes of the workspace's symmetric region for a token count.

    Args:
        input: Optional [T, 4096] bfloat16 CUDA tensor.  Without it, the
            larger of the supported schedules is returned.

    Returns:
        Size in bytes of the scratchpad plus the tail mailbox.
    """
    return _workspace_layout(input)[2]


def transform_weights_for_moe(
    router_weight: Optional[Union[torch.Tensor, Sequence[torch.Tensor]]],
    expert_up_weights: Optional[torch.Tensor],
    shared_up_weight: Optional[torch.Tensor],
    latent_down_weight: Optional[torch.Tensor] = None,
    num_tokens: int = 1,
    num_router_gates: int = 2,
) -> dict:
    """Pack the router, up-projection and latent-down weights for ``low_latency_moe``.

    The down-projection weights and all block scales already follow torch's
    ``[out, in]`` layout and are passed to ``low_latency_moe`` as-is.  This repacks the
    operands the kernel reads differently:

    * ``router_weight`` -- the router's SW128 swizzle, one packed blob per
      gate.
    * ``expert_up_weights`` / ``shared_up_weight`` -- the gate/up pair
      layout (pack after quantizing: the scales stay on the raw layout).
    * ``latent_down_weight`` -- transposed to ``[latent, hidden]``.

    ``num_tokens`` selects the router schedule and must match the ``low_latency_moe``
    call.  Every operand is optional: only the ones that are given are packed (and
    returned), e.g. to pack the experts as soon as they are loaded and the router once
    per schedule.

    Args:
        router_weight: Router weight ``[K, E]`` bfloat16, ``[G, K, E]`` or a
            sequence of G ``[K, E]`` tensors.  A single ``[K, E]`` weight is
            reused for every gate.  None to skip.
        expert_up_weights: [E, 2N, K] float8_e4m3fn routed weights; rows
            [0, N) are gate and [N, 2N) up.  None to skip.
        shared_up_weight: [2N, H] float8_e4m3fn shared-expert weights, same
            gate/up split.  None to skip.
        latent_down_weight: Optional [H, L] float8_e4m3fn latent
            down-projection weight.
        num_tokens: Token count of the ``low_latency_moe`` call, which picks the router
            tile M.
        num_router_gates: Gates to merge; this build requires 2.

    Returns:
        A dict with the keys ``"router_weight"``, ``"expert_up_weights"``,
        ``"shared_up_weight"`` and ``"latent_down_weight"`` of the operands
        that were given.
    """
    packed = {}
    if router_weight is not None:
        packed["router_weight"] = _pack_router_weight(
            _router_gates(router_weight, num_router_gates), num_tokens
        )
    if expert_up_weights is not None:
        packed["expert_up_weights"] = _interleave_up_for_pair_layout(expert_up_weights)
    if shared_up_weight is not None:
        packed["shared_up_weight"] = _prepare_shared_expert_up_weight(shared_up_weight)
    if latent_down_weight is not None:
        packed["latent_down_weight"] = _prepare_latent_down_weight(latent_down_weight)
    return packed
