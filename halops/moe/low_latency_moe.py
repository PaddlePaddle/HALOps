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

from typing import Optional, Sequence, Union

import torch

import halops.halops_kernel as halops_kernel
from halops.moe.low_latency_moe_utils import (
    _interleave_up_for_pair_layout,
    _pack_router_weight,
    _prepare_latent_down_weight,
    _prepare_shared_expert_up_weight,
    _router_gates,
)

__all__ = [
    "low_latency_moe",
    "get_workspace_for_moe",
    "get_workspace_size_for_moe",
    "transform_weights_for_moe",
]


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
    latent_down_weight: Optional[torch.Tensor],
    latent_down_weight_scale: Optional[torch.Tensor],
    workspace: torch.Tensor,
    routed_out: torch.Tensor,
    shared_out: torch.Tensor,
    top_k: int,
    num_router_gates: int = 2,
    scoring_func: str = "sigmoid",
    renormalize: bool = True,
    activation: str = "silu",
    situ_beta: float = 4.0,
    situ_linear_beta: float = 25.0,
) -> None:
    """MoE forward pass on Blackwell SM100, support latent.

    Runs the router, the latent down-projection, the routed experts' fused
    up/down projections and the shared expert for 1..16 tokens, writing
    ``routed_out`` and ``shared_out`` in place.  The shipped schedule is
    fixed to E=512 experts, hidden size 4096, intermediate size 512 and
    latent size 2048, with block-wise FP8 weights (128x128 blocks).

    Weights are block-wise quantized by the caller; the router, up-projection
    and latent-down operands must first be packed with
    ``transform_weights_for_moe``.  This build runs the latent stage, so
    ``latent_down_weight`` and ``latent_down_weight_scale`` are required.

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
        workspace: uint8 workspace from ``get_workspace_for_moe``.
        routed_out: [T, 2048] bfloat16 output of the routed experts, written
            in place.
        shared_out: [T, 4096] bfloat16 output of the shared expert, written
            in place.
        top_k: Experts selected per token, in [1, 10].
        num_router_gates: Router gates merged into ``router_weight``; this
            build requires 2.
        scoring_func: Router scoring, ``"sigmoid"`` or ``"softmax"``.
        renormalize: Renormalize the top-k weights to sum to one.
        activation: Expert activation, ``"silu"`` or ``"situ"``.
        situ_beta: SiTU gate soft cap; <= 0 disables it.  Must be finite.
        situ_linear_beta: SiTU up soft cap; <= 0 disables it.  Must be
            finite.

    Returns:
        None. Both outputs are written in place.
    """
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
        workspace,
        routed_out,
        shared_out,
        top_k,
        num_router_gates,
        scoring_func,
        renormalize,
        activation,
        situ_beta,
        situ_linear_beta,
    )


def get_workspace_for_moe(input: torch.Tensor) -> torch.Tensor:
    """Allocate the uint8 workspace of ``low_latency_moe`` for ``input``'s token count.

    Allocate it once, before any CUDA-graph capture, and reuse it across
    calls and graph replays; it is returned zero-initialized.

    Args:
        input: [T, 4096] bfloat16 CUDA tensor; T selects the schedule.

    Returns:
        1-D uint8 CUDA tensor on ``input``'s device.
    """
    return halops_kernel.moe_get_workspace(input)


def get_workspace_size_for_moe(input: Optional[torch.Tensor] = None) -> int:
    """Return the workspace size in bytes for a token count.

    Args:
        input: Optional [T, 4096] bfloat16 CUDA tensor.  Without it, the
            larger of the supported schedules is returned.

    Returns:
        Workspace size in bytes.
    """
    return halops_kernel.moe_get_workspace_size(input)


def transform_weights_for_moe(
    router_weight: Union[torch.Tensor, Sequence[torch.Tensor]],
    expert_up_weights: torch.Tensor,
    shared_up_weight: torch.Tensor,
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
    call.

    Args:
        router_weight: Router weight ``[K, E]`` bfloat16, ``[G, K, E]`` or a
            sequence of G ``[K, E]`` tensors.  A single ``[K, E]`` weight is
            reused for every gate.
        expert_up_weights: [E, 2N, K] float8_e4m3fn routed weights; rows
            [0, N) are gate and [N, 2N) up.
        shared_up_weight: [2N, H] float8_e4m3fn shared-expert weights, same
            gate/up split.
        latent_down_weight: Optional [H, L] float8_e4m3fn latent
            down-projection weight.
        num_tokens: Token count of the ``low_latency_moe`` call, which picks the router
            tile M.
        num_router_gates: Gates to merge; this build requires 2.

    Returns:
        A dict with keys ``"router_weight"``, ``"expert_up_weights"``,
        ``"shared_up_weight"`` and, when given, ``"latent_down_weight"``.
    """
    packed = {}
    packed["router_weight"] = _pack_router_weight(
        _router_gates(router_weight, num_router_gates), num_tokens
    )
    packed["expert_up_weights"] = _interleave_up_for_pair_layout(expert_up_weights)
    packed["shared_up_weight"] = _prepare_shared_expert_up_weight(shared_up_weight)
    if latent_down_weight is not None:
        packed["latent_down_weight"] = _prepare_latent_down_weight(latent_down_weight)
    return packed
