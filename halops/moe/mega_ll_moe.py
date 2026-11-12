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
from typing import Optional

import torch

import halops.halops_kernel as halops_kernel
from halops.moe.mega_ll_moe_utils import (
    _NARROW_TOKENS,
    _TAIL_ALIGN,
    _check_workspace,
    _interleave_gate_up_weight,
    _interleave_gate_up_scale,
)

__all__ = [
    "MoEWorkspace",
    "mega_ll_moe",
    "transform_weights_for_moe",
    "get_workspace_for_moe",
    "get_workspace_size_for_moe",
]


@dataclass
class MoEWorkspace:
    """Scratchpad and tail communication state of ``mega_ll_moe``.

    The kernel has a narrow (1 <= T <= 8) and a wide (9 <= T <= 16)
    schedule; each has its own scratchpad and tail layout, so a workspace is
    bound to one of them.  ``buffer`` is the scratchpad head of the workspace
    region whose tail part is the fused tail's mailbox.
    """

    buffer: torch.Tensor
    uc_bases: torch.Tensor
    epochs: torch.Tensor
    tp_rank: int
    tp_size: int
    wide: bool
    mc_base: int
    _keepalive: tuple


def _workspace_layout(input: Optional[torch.Tensor], tp_size: int = 1):
    """Return (scratchpad bytes, tail offset, total bytes) of the workspace."""
    if tp_size not in (1, 4):
        raise ValueError(f"tp_size must be 1 or 4, got {tp_size}")
    scratch = int(halops_kernel.moe_get_workspace_size(input))
    offset = -(-scratch // _TAIL_ALIGN) * _TAIL_ALIGN
    if input is None:
        tail = max(
            int(halops_kernel.moe_tail_region_bytes(w, tp_size)) for w in (False, True)
        )
    else:
        tail = int(
            halops_kernel.moe_tail_region_bytes(input.size(0) > _NARROW_TOKENS, tp_size)
        )
    return scratch, offset, offset + tail


def get_workspace_for_moe(
    input: torch.Tensor,
    group_name: str,
    tp_rank: int,
    tp_size: int,
) -> MoEWorkspace:
    """Allocate the workspace of ``mega_ll_moe`` for ``input``'s token count.

    For tp_size 4, allocation is collective over ``group_name``. Allocate it
    once, before any CUDA-graph capture, and reuse it across calls and graph
    replays; it is returned zero-initialized. A workspace must always be
    launched with the same grid.

    Args:
        input: [T, 4096] bfloat16 CUDA tensor; T selects the schedule.
        group_name: Name of the torch process group of the tensor-parallel
            ranks. It is unused for a single-rank launch.
        tp_rank: Rank in that group.
        tp_size: Size of that group. Supported values are 1 and 4.

    Returns:
        The workspace to pass to ``mega_ll_moe``.
    """
    wide = input.size(0) > _NARROW_TOKENS
    device = input.device
    if tp_size not in (1, 4):
        raise ValueError(f"tp_size must be 1 or 4, got {tp_size}")
    scratch, offset, nbytes = _workspace_layout(input, tp_size)

    if tp_size == 1:
        region = torch.zeros(nbytes, dtype=torch.uint8, device=device)
        pointer = region.data_ptr() + offset
        uc_bases = torch.tensor([pointer], dtype=torch.int64, device="cpu")
        epochs = torch.zeros(1, dtype=torch.int32, device=device)
        return MoEWorkspace(
            buffer=region[:scratch],
            tp_rank=0,
            tp_size=1,
            wide=wide,
            mc_base=pointer,
            uc_bases=uc_bases,
            epochs=epochs,
            _keepalive=(region,),
        )

    from torch._C._distributed_c10d import _SymmetricMemory

    with torch.inference_mode(False):
        region = _SymmetricMemory.empty_strided_p2p(
            (nbytes,), [1], torch.uint8, device, group_name
        )
    handle = _SymmetricMemory.rendezvous(region)
    if handle is None or handle.multicast_ptr == 0:
        raise RuntimeError("the fused tail needs an NVLS multicast-capable region")
    region.zero_()
    peers = [handle.get_buffer(r, [nbytes], torch.uint8) for r in range(tp_size)]
    uc_bases = torch.tensor(
        [b.data_ptr() + offset for b in peers], dtype=torch.int64, device="cpu"
    )
    epochs = torch.zeros(1, dtype=torch.int32, device=device)
    torch.cuda.synchronize()
    torch.distributed.barrier()
    return MoEWorkspace(
        buffer=region[:scratch],
        tp_rank=tp_rank,
        tp_size=tp_size,
        wide=wide,
        mc_base=int(handle.multicast_ptr) + offset,
        uc_bases=uc_bases,
        epochs=epochs,
        _keepalive=(region, peers),
    )


def transform_weights_for_moe(
    expert_up_weights: torch.Tensor,
    expert_up_weights_scale: torch.Tensor,
    shared_up_weight: torch.Tensor,
    shared_up_weight_scale: torch.Tensor,
    routed_use_interleaved: bool = True,
    shared_use_interleaved: bool = True,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Repack the up projections' weights (and MX scale tables) for ``mega_ll_moe``.

    The up projections can interleave their gate/up rows in 8-row groups; the
    caller selects that layout per projection with ``routed_use_interleaved``
    and ``shared_use_interleaved`` on ``mega_ll_moe``.  This rewrites the
    natural ``[..., 2N, K]`` weights, and the ``uint8`` tables from
    ``group_wise_quant(use_swizzle=True)``, into that order.  Block-wise f32
    scale tables are returned unchanged.

    Args:
        expert_up_weights (Tensor): Fused gate/up weights of the routed
            experts, shape [E, 2N, K], in the natural [gate | up] row order.
        expert_up_weights_scale (Tensor): Scales of ``expert_up_weights``: the
            uint8 table in the natural SfAtom layout, or the block-wise f32
            table.
        shared_up_weight (Tensor): Fused gate/up weight of the shared expert,
            shape [2N, K], in the natural row order.
        shared_up_weight_scale (Tensor): Scales of ``shared_up_weight``, same
            layout as the routed ones.
        routed_use_interleaved (bool, optional): Repack the routed rows.
            Default: True.
        shared_use_interleaved (bool, optional): Repack the shared rows.
            Default: True.

    Returns:
        (Tensor, Tensor, Tensor, Tensor): the routed weight/scale and the
        shared weight/scale to pass to ``mega_ll_moe``.  Interleaved rows keep
        the weight's shape; the scale table stays in its SfAtom layout.
    """

    def pack(weight, scale, interleaved, name):
        if not interleaved:
            return weight, scale
        if scale is None:
            raise ValueError(
                f"interleaving {name} needs its scale table: the weight and "
                "its scale rows move together."
            )
        return (
            _interleave_gate_up_weight(weight).reshape(weight.shape),
            _interleave_gate_up_scale(scale),
        )

    expert_up_weights, expert_up_weights_scale = pack(
        expert_up_weights,
        expert_up_weights_scale,
        routed_use_interleaved,
        "expert_up_weights",
    )
    shared_up_weight, shared_up_weight_scale = pack(
        shared_up_weight,
        shared_up_weight_scale,
        shared_use_interleaved,
        "shared_up_weight",
    )
    return (
        expert_up_weights,
        expert_up_weights_scale,
        shared_up_weight,
        shared_up_weight_scale,
    )


def mega_ll_moe(
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
    latent_up_weight: torch.Tensor,
    latent_up_weight_scale: torch.Tensor,
    norm_weight: torch.Tensor,
    correction_bias: Optional[torch.Tensor],
    per_expert_scale: Optional[torch.Tensor],
    out: torch.Tensor,
    workspace: MoEWorkspace,
    top_k: int,
    num_router_gates: int = 2,
    scoring_func: str = "sigmoid",
    renormalize: bool = True,
    activation: str = "silu",
    situ_beta: float = 4.0,
    situ_linear_beta: float = 25.0,
    eps: float = 1e-6,
    quant_mode: str = "w4a8-mxfp4",
    grid_cap: int = 0,
    use_pdl: bool = True,
    routed_use_interleaved: bool = True,
    shared_use_interleaved: bool = True,
    routed_use_swizzle: bool = True,
    shared_use_swizzle: bool = True,
    latent_use_swizzle: bool = False,
) -> None:
    """MoE forward pass on Blackwell SM100, support latent.

    Runs the router, the latent down-projection, the routed experts' fused
    up/down projections and the shared expert for 1..16 tokens. The shipped
    schedule is fixed to E=512 experts, hidden size 4096, intermediate size 512
    and latent size 2048 now.

    Weights are quantized by the caller (``group_wise_quant`` for MX
    recipes); the up-projection gate/up rows are either passed pre-interleaved
    or interleaved by the TMA while loading, and the scale tables are either
    the caller's SfAtom tables or plain rows the kernel converts.  The layout
    flags below select which form each operand is in.

    The whole forward pass runs in a single kernel launch::

        (w, e) = renormalize(topk(sum_g sigmoid(x @ W_g.T)))    # router
        routed = sum_k w_k * (act(x @ W_latent.T @ W_up[e_k].T) @ W_down[e_k].T)
        shared = act(x @ W_sup.T) @ W_sdown.T
        out    = fc2(RMSNorm(AllReduce(routed))) + AllReduce(shared)

    where ``x`` is ``input``, ``g`` runs over the router gates, ``e_k``/``w_k``
    are each token's top-k experts and weights, ``act`` is the gated activation
    and ``fc2`` the tail projection.  ``routed`` is this rank's combined routed
    latent and ``shared`` its partial shared-expert output (kept in
    ``workspace``); both AllReduces run over the configured tensor-parallel
    group.

    Args:
        input (Tensor): Activations, shape [T, 4096], dtype bfloat16, with
            1 <= T <= 16.
        router_weight (Tensor): Router weight, dtype bfloat16, shape
            [G, E, H] = [2, 512, 4096], contiguous; gate ``g`` holds every
            expert's row.
        expert_up_weights (Tensor): Fused gate/up weights, dtype
            float8_e4m3fn, in the natural [512, 2 * 512, 2048] layout. Under
            "w4a8-mxfp4" the dtype is uint8 and K is halved: two 4-bit values
            per byte, as ``group_wise_quant(mode="w4a4-mxfp4")`` packs them.
            Pass the output of ``transform_weights_for_moe`` ([E, N/8, 2, 8,
            K]) instead to run the 8-row pair layout.
        expert_up_weights_scale (Tensor): Block scales of
            ``expert_up_weights``: shape [512, 1024, 64] and dtype uint8 in
            the CUTLASS SfAtom layout for "w8a8-mxfp8" and "w4a8-mxfp4", or
            shape [512, 8, 16] and dtype float32 for "w8a8-blockwise".  The
            MX scales are the natural SfAtom tables from
            ``group_wise_quant(use_swizzle=True)``; ``transform_weights_for_moe``
            moves them with the weight.
        expert_down_weights (Tensor): Down weights, shape [512, 2048, 512],
            dtype float8_e4m3fn; under "w4a8-mxfp4" uint8 of shape [512,
            2048, 256], packed the same way.
        expert_down_weights_scale (Tensor): Block scales of
            ``expert_down_weights``: shape [512, 2048, 16] and dtype uint8 in
            the SfAtom layout for "w8a8-mxfp8" and "w4a8-mxfp4", or shape
            [512, 16, 4] and dtype float32 for "w8a8-blockwise".
        shared_up_weight (Tensor): Shared-expert gate/up weights, dtype
            float8_e4m3fn, in the natural [2 * 512, 4096] layout.
        shared_up_weight_scale (Tensor): Scales of ``shared_up_weight``:
            shape [8, 32] and dtype float32 for "w8a8-blockwise", or the
            natural [1024, 128] uint8 rows from
            ``group_wise_quant(use_swizzle=False)`` for "w8a8-mxfp8" and
            "w4a8-mxfp4" (latent and shared stay MXFP8 in that recipe).
        shared_down_weight (Tensor): Shared-expert down weights, shape
            [4096, 512], dtype float8_e4m3fn.
        shared_down_weight_scale (Tensor): Scales of ``shared_down_weight``:
            shape [32, 4] and dtype float32 for "w8a8-blockwise", or the
            natural [4096, 16] uint8 rows from
            ``group_wise_quant(use_swizzle=False)`` for "w8a8-mxfp8" and
            "w4a8-mxfp4".
        latent_down_weight (Tensor): Latent down-projection weight in the
            kernel layout, shape [2048, 4096], dtype float8_e4m3fn.
            Quantize it in this layout and pass it directly.
        latent_down_weight_scale (Tensor): Scales of
            ``latent_down_weight``: shape [16, 32] and dtype float32 for
            "w8a8-blockwise", or the natural [2048, 128] uint8 rows (from
            ``group_wise_quant(use_swizzle=False)``) for "w8a8-mxfp8" and
            "w4a8-mxfp4"; the kernel rewrites them into the MMA's SfAtom
            order.
        latent_up_weight (Tensor): [4096, 2048] float8_e4m3fn latent up weight.
        latent_up_weight_scale (Tensor): [32, 16] float32 block scales, powers of two.
        latent_up_weight (Tensor): [4096, 2048] float8_e4m3fn latent up weight
            (the tail's fc2).
        latent_up_weight_scale (Tensor): Scales of ``latent_up_weight``,
            following the latent recipe of ``quant_mode``: shape [32, 16] and
            dtype float32 (powers of two) for "w8a8-blockwise"; shape
            [4096, 64] and dtype uint8 (UE8M0, one per row per 32 columns,
            plain row-major as from ``group_wise_quant(use_swizzle=False)``,
            not the SfAtom layout) for "w8a8-mxfp8" and "w4a8-mxfp4".
        norm_weight (Tensor): [2048] bfloat16 RMSNorm weight.
        correction_bias (Tensor or None): [512] float32 CUDA router bias
            (``e_score_correction_bias``). It is added to each expert's score
            only to choose the top-k; the weights and their renormalization
            use the bias-free scores. None: no bias.
        per_expert_scale (Tensor or None): [512] float32 CUDA routed scaling,
            multiplied into each selected weight after renormalization. None:
            no scaling.
        out (Tensor): [T, 4096] bfloat16 output, identical across ranks.
        workspace (MoEWorkspace): Scratchpad and tail communication state
            from ``get_workspace_for_moe``, matching the token schedule. It
            also fixes the tensor-parallel size and rank of the launch.
        top_k (int): Experts selected per token, in [1, 10].
        num_router_gates (int, optional): Gates in ``router_weight``'s leading
            dimension; this build requires 2. Default: 2.
        scoring_func (str, optional): Router scoring, either "sigmoid" or
            "softmax". Default: "sigmoid".
        renormalize (bool, optional): Renormalize the top-k weights to sum to
            one. Default: True.
        activation (str, optional): Expert activation, either "silu" or
            "situ". Default: "silu".
        situ_beta (float, optional): SiTU gate soft cap; <= 0 disables it.
            Must be finite. Default: 4.0.
        situ_linear_beta (float, optional): SiTU up soft cap; <= 0 disables
            it. Must be finite. Default: 25.0.
        eps (float, optional): RMSNorm epsilon. Default: 1e-6.
        quant_mode (str, optional): Quantization recipe: "w8a8-blockwise"
            (every group block-wise), "w8a8-mxfp8" (every group MXFP8) or
            "w4a8-mxfp4" (routed experts MXFP4, latent and shared MXFP8).
            Default: "w4a8-mxfp4".
        grid_cap (int, optional): Maximum resident CTAs; 0 uses the device
            limit. Keep this fixed for each workspace. Default: 0.
        use_pdl (bool, optional): Launch with Programmatic Dependent Launch, so
            the kernel's setup overlaps the previous kernel's tail. It waits for
            that kernel before touching shared state, and lets the next kernel
            start launching once the tail begins. Default: True.
        routed_use_interleaved (bool, optional): The routed experts' fused
            gate/up weights are in the 8-row interleaved order from
            ``transform_weights_for_moe``. Default: True.
        shared_use_interleaved (bool, optional): The shared expert's fused
            gate/up weight is in the 8-row interleaved order from
            ``transform_weights_for_moe``. Default: True.
        routed_use_swizzle (bool, optional): The routed up and down scale
            tables are the caller's SfAtom tables
            (``group_wise_quant(use_swizzle=True)``) rather than plain rows.
            Interleaved rows imply it. Default: True.
        shared_use_swizzle (bool, optional): The shared up and down scale
            tables are the caller's SfAtom tables rather than plain rows.
            Interleaved rows imply it. Default: True.
        latent_use_swizzle (bool, optional): The latent down-projection's
            scale table is the caller's SfAtom table rather than plain rows.
            Default: False.

    Returns:
        None. The outputs are written in place.
    """
    _check_workspace(workspace, input.size(0))
    halops_kernel.mega_ll_moe(
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
        latent_up_weight,
        latent_up_weight_scale,
        norm_weight,
        correction_bias,
        per_expert_scale,
        workspace.epochs,
        workspace.uc_bases,
        workspace.buffer,
        out,
        top_k,
        num_router_gates,
        workspace.tp_rank,
        workspace.mc_base,
        grid_cap,
        scoring_func,
        activation,
        renormalize,
        situ_beta,
        situ_linear_beta,
        eps,
        quant_mode,
        workspace.tp_size,
        use_pdl,
        routed_use_interleaved,
        shared_use_interleaved,
        routed_use_swizzle,
        shared_use_swizzle,
        latent_use_swizzle,
    )


def get_workspace_size_for_moe(
    input: Optional[torch.Tensor] = None, tp_size: int = 1
) -> int:
    """Return the workspace region size in bytes for a token count.

    Args:
        input (Tensor, optional): Activations, shape [T, 4096], dtype
            bfloat16. Without it, the larger of the supported schedules is
            returned. Default: None.
        tp_size (int, optional): Tensor-parallel size, either 1 or 4.
            Default: 1.

    Returns:
        Size in bytes of the scratchpad plus the tail mailbox.
    """
    return _workspace_layout(input, tp_size)[2]
