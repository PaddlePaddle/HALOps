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

from typing import Sequence, Union

import torch

import halops.halops_kernel as halops_kernel

__all__ = []

# Gate/up interleave granularity: fixed by the TMEM read pattern, not a knob.
_PAIR_GRAN = 8

# One CUTLASS SfAtom: 128 weight rows x 4 K/32 blocks, 512 B in total.
_SF_ATOM_ROWS = 128
_SF_ATOM_COLS = 4
_SF_ATOM_BYTES = 512

# K width of one router SW128 box.
_ROUTER_SWIZZLE_K = 64

# Token capacity of the narrow schedule; larger token counts run the wide one.
_NARROW_TOKENS = 8
_MAX_TOKENS = 16

# Alignment of the tail mailbox inside the workspace's symmetric region.
_TAIL_ALIGN = 4096


def _interleave_up_for_pair_layout(w_fp8: torch.Tensor) -> torch.Tensor:
    """Repack ``[E, 2N, K]`` up/gate weights into the gran-8 pair layout."""
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
    return torch.stack([gate, up], dim=2).reshape(e, rows, k).contiguous()


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

    packed = torch.cat([_pack_router_gate(w, tile_m) for w in per_gate], dim=0)
    return packed.contiguous()


def _pair_layout_rows(w: torch.Tensor) -> torch.Tensor:
    """Repack ``[..., 2N, K]`` rows from ``[gate | up]`` into gran-8 pairs."""
    rows, k = int(w.shape[-2]), int(w.shape[-1])
    if rows % 2 != 0:
        raise ValueError(f"expected rows = 2*N, got rows={rows}")
    half = rows // 2
    if half % _PAIR_GRAN != 0:
        raise ValueError(
            f"N (half of rows) must be a multiple of {_PAIR_GRAN}; got N={half}"
        )
    lead = w.shape[:-2]
    gate = w[..., :half, :].reshape(*lead, half // _PAIR_GRAN, _PAIR_GRAN, k)
    up = w[..., half:, :].reshape(*lead, half // _PAIR_GRAN, _PAIR_GRAN, k)
    return torch.stack([gate, up], dim=-3).reshape(w.shape).contiguous()


def _prepare_shared_expert_up_weight(shared_up_weight: torch.Tensor) -> torch.Tensor:
    """Pack the shared expert's up weight ``[2N, H]`` into the pair layout.

    Pack after quantizing: the weight rows move, so an MXFP8 SfAtom scale
    table must be moved with them (``_prepare_shared_expert_up_scale``).
    """
    if shared_up_weight.dim() != 2:
        raise ValueError(
            "shared_up_weight must be 2-D [2*N, hidden_size], got "
            f"{tuple(shared_up_weight.shape)}."
        )
    return _pair_layout_rows(shared_up_weight)


def _sf_atom_offsets(rows: int, cols: int, device) -> torch.Tensor:
    """Flat offsets of a logical ``[rows, cols]`` table in a CUTLASS SfAtom.

    One 512-B atom holds 128 rows x 4 K/32 blocks, so ``rows`` and ``cols``
    must cover whole atoms.
    """
    m = torch.arange(rows, device=device)
    g = torch.arange(cols, device=device)
    return (
        (m[:, None] // _SF_ATOM_ROWS) * (cols // _SF_ATOM_COLS) * _SF_ATOM_BYTES
        + (g[None, :] // _SF_ATOM_COLS) * _SF_ATOM_BYTES
        + (m[:, None] % 32) * 16
        + ((m[:, None] % _SF_ATOM_ROWS) // 32) * _SF_ATOM_COLS
        + (g[None, :] % _SF_ATOM_COLS)
    )


def _prepare_up_scale(scale: torch.Tensor) -> torch.Tensor:
    """Move an MXFP8 up-weight SfAtom table onto the packed pair rows.

    ``scale`` is ``[..., 2N, K/32]``: one table per expert, or a single one
    for the shared expert.  The table is indexed by raw row, so the gate/up
    pairing that the weight packing applies has to be applied here as well.
    Block-wise f32 tables stay on the raw ``[gate | up]`` layout, the one
    every down-projection scale uses, and are returned untouched.
    """
    if scale.dtype != torch.uint8:
        return scale
    rows, cols = int(scale.shape[-2]), int(scale.shape[-1])
    if rows % _SF_ATOM_ROWS or cols % _SF_ATOM_COLS:
        raise ValueError(
            "an MXFP8 SfAtom table packs whole 128-rows-x-4-blocks atoms; got "
            f"{tuple(scale.shape)}."
        )
    offsets = _sf_atom_offsets(rows, cols, scale.device).reshape(-1)
    tables = scale.reshape(-1, rows * cols)
    plain = tables[:, offsets]
    packed = _pair_layout_rows(plain.reshape(*scale.shape[:-2], rows, cols))
    out = torch.empty_like(tables)
    out[:, offsets] = packed.reshape(-1, rows * cols)
    return out.reshape(scale.shape)


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


def _check_workspace(workspace, num_tokens: int) -> None:
    """Check that ``workspace`` is the one bound to the schedule that ``num_tokens`` runs."""
    if not 1 <= num_tokens <= _MAX_TOKENS:
        raise ValueError(f"num_tokens must be in [1, {_MAX_TOKENS}], got {num_tokens}")
    wide = num_tokens > _NARROW_TOKENS
    if workspace.wide != wide:
        raise ValueError(
            f"the workspace is bound to the {'wide' if workspace.wide else 'narrow'} "
            f"schedule but T={num_tokens} runs the {'wide' if wide else 'narrow'} one: "
            f"allocate it from an input with T {'>' if wide else '<='} {_NARROW_TOKENS}."
        )
