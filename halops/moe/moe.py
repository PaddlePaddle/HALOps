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

import contextlib
from typing import Optional, Sequence, Union

import torch

import halops.halops_kernel as halops_kernel

__all__ = [
    "moe",
    "get_workspace_for_moe",
    "get_workspace_size_for_moe",
    "transform_weights_for_moe",
]


def moe(
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
    halops_kernel.moe(
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
    return halops_kernel.moe_get_workspace(input)


def get_workspace_size_for_moe(input: Optional[torch.Tensor] = None) -> int:
    return halops_kernel.moe_get_workspace_size(input)


# Gate/up interleave granularity: fixed by the TMEM read pattern, not a knob.
_PAIR_GRAN = 8

# K width of one router SW128 box.
_ROUTER_SWIZZLE_K = 64


def _interleave_up_for_pair_layout(w_fp8: torch.Tensor) -> torch.Tensor:
    """Repack ``[E, 2N, K]`` up/gate weights into the gran-8 pair layout; cached."""
    cached = getattr(w_fp8, "_tma_pair_layout_up", None)
    if cached is not None:
        return cached

    e, rows, k = w_fp8.shape
    if rows % 2 != 0:
        raise ValueError(f"expected rows = 2*N, got rows={rows}")
    half = rows // 2
    if half % _PAIR_GRAN != 0:
        raise ValueError(
            f"N (half of rows) must be a multiple of {_PAIR_GRAN}; got N={half}"
        )

    gate = w_fp8[:, :half].reshape(e, half // _PAIR_GRAN, _PAIR_GRAN, k)
    up = w_fp8[:, half:].reshape(e, half // _PAIR_GRAN, _PAIR_GRAN, k)
    result = torch.stack([gate, up], dim=2).reshape(e, rows, k).contiguous()

    with contextlib.suppress(AttributeError, RuntimeError):
        w_fp8._tma_pair_layout_up = result
        result._tma_pair_layout_up = result  # already packed: a second pass is a no-op
    return result


def _pack_router_gate(router_weight: torch.Tensor, tile_m: int) -> torch.Tensor:
    """Pack one gate's ``[K, E]`` weight into its SW128 blob."""
    k, e = (int(x) for x in router_weight.shape)
    if e % tile_m or k % _ROUTER_SWIZZLE_K:
        raise ValueError(
            f"router_weight [K={k}, E={e}] needs E % tile_m ({tile_m}) "
            f"== 0 and K % {_ROUTER_SWIZZLE_K} == 0"
        )
    t, b = e // tile_m, k // _ROUTER_SWIZZLE_K
    chunks = _ROUTER_SWIZZLE_K // 8  # 16-B chunks per box row
    # [t, b, m, c, kk]: logical (unswizzled) boxes.
    boxes = (
        router_weight.to(torch.bfloat16)
        .t()
        .reshape(t, tile_m, b, chunks, 8)
        .permute(0, 2, 1, 3, 4)
    )
    # Physical chunk c of row m holds logical chunk c ^ (m % 8).
    m = torch.arange(tile_m, device=router_weight.device)
    c = torch.arange(chunks, device=router_weight.device)
    src = (c[None, :] ^ (m[:, None] % 8)).view(1, 1, tile_m, chunks, 1)
    return (
        boxes.gather(3, src.expand(t, b, tile_m, chunks, 8))
        .reshape(t, b, tile_m, _ROUTER_SWIZZLE_K)
        .contiguous()
    )


def _prepare_latent_down_weight(
    latent_down_weight: torch.Tensor, device: Optional[torch.device] = None
) -> torch.Tensor:
    """Pack the fp8 latent down weight ``[H, L]`` into ``[L, H]``; cached."""
    if latent_down_weight.dim() != 2:
        raise ValueError(
            "latent_down_weight must be 2-D [hidden_size, latent_size], got "
            f"{tuple(latent_down_weight.shape)}."
        )
    packed = getattr(latent_down_weight, "_moe_latent_down_packed", None)
    if packed is None:
        packed = latent_down_weight.transpose(0, 1).contiguous()
        if device is not None:
            packed = packed.to(device)
        latent_down_weight._moe_latent_down_packed = packed
    return packed


def _pack_router_weight(
    router_weights: Union[torch.Tensor, Sequence[torch.Tensor]], num_tokens: int
) -> torch.Tensor:
    """Merge one router weight per gate into the kernel's packed blob."""
    tile_m = halops_kernel.moe_router_tile_m(num_tokens)
    if isinstance(router_weights, (list, tuple)):
        per_gate = list(router_weights)
    elif router_weights.dim() == 3:
        per_gate = list(router_weights.unbind(0))
    elif router_weights.dim() == 2:
        per_gate = [router_weights]
    else:
        raise ValueError(
            "router_weights must be [K, E], [G, K, E] or a sequence of G "
            f"[K, E] weights, got {tuple(router_weights.shape)}"
        )

    cached = getattr(per_gate[0], "_moe_router_packed", None)
    key = (tile_m, len(per_gate))
    if cached is not None and cached[0] == key:
        return cached[1]

    packed = torch.cat([_pack_router_gate(w, tile_m) for w in per_gate], dim=0)
    packed = packed.contiguous()
    with contextlib.suppress(AttributeError, RuntimeError):
        per_gate[0]._moe_router_packed = (key, packed)
    return packed


def _prepare_shared_expert_up_weight(shared_up_weight: torch.Tensor) -> torch.Tensor:
    """Pack the shared expert's up weight ``[2N, H]`` into the pair layout; cached.

    Pack after quantizing: the scales belong to the raw ``[gate | up]`` layout.
    """
    if shared_up_weight.dim() != 2:
        raise ValueError(
            "shared_up_weight must be 2-D [2*N, hidden_size], got "
            f"{tuple(shared_up_weight.shape)}."
        )
    cached = getattr(shared_up_weight, "_moe_shared_up_packed", None)
    if cached is not None:
        return cached

    rows, k = shared_up_weight.shape
    if rows % 2 != 0:
        raise ValueError(f"expected rows = 2*N, got rows={rows}")
    half = rows // 2
    if half % _PAIR_GRAN != 0:
        raise ValueError(
            f"N (half of rows) must be a multiple of {_PAIR_GRAN}; got N={half}"
        )
    gate = shared_up_weight[:half].reshape(half // _PAIR_GRAN, _PAIR_GRAN, k)
    up = shared_up_weight[half:].reshape(half // _PAIR_GRAN, _PAIR_GRAN, k)
    packed = torch.stack([gate, up], dim=1).reshape(rows, k).contiguous()

    with contextlib.suppress(AttributeError, RuntimeError):
        shared_up_weight._moe_shared_up_packed = packed
        packed._moe_shared_up_packed = (
            packed  # already packed: a second pass is a no-op
        )
    return packed


def _router_gates(
    router_weight: Union[torch.Tensor, Sequence[torch.Tensor]], num_router_gates: int
):
    """The per-gate weights: a `[K, E]` weight serves every gate."""
    if isinstance(router_weight, (list, tuple)):
        gates = list(router_weight)
    elif router_weight.dim() == 3:
        gates = list(router_weight.unbind(0))
    else:
        gates = [router_weight] * num_router_gates
    if len(gates) != num_router_gates:
        raise ValueError(
            f"expected {num_router_gates} router weights to merge, got " f"{len(gates)}"
        )
    return gates


def transform_weights_for_moe(
    router_weight: Union[torch.Tensor, Sequence[torch.Tensor]],
    expert_up_weights: torch.Tensor,
    shared_up_weight: torch.Tensor,
    latent_down_weight: Optional[torch.Tensor] = None,
    num_tokens: int = 1,
    num_router_gates: int = 2,
) -> dict:
    """Pack the router, up, and latent-down weights into the kernel's layouts.

    ``num_tokens`` selects the router schedule; results are cached on the inputs.
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
