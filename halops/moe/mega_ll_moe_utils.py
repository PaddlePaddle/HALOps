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

import torch

__all__ = []

# One CUTLASS SfAtom: 128 weight rows x 4 K/32 blocks, 512 B in total.
_SF_ATOM_ROWS = 128
_SF_ATOM_COLS = 4
_SF_ATOM_BYTES = 512

# Token capacity of the narrow schedule; larger token counts run the wide one.
_NARROW_TOKENS = 8
_MAX_TOKENS = 16

# Alignment of the tail mailbox inside the workspace's symmetric region.
_TAIL_ALIGN = 4096


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


def _interleave_gate_up_weight(w: torch.Tensor, gran: int = 8) -> torch.Tensor:
    """Repack ``[..., 2N, K]`` rows from ``[gate | up]`` into ``gran``-row pairs.

    Returns ``[..., N/gran, 2, gran, K]``: the row group stays a dimension, so
    the row order reads as a gate group, then the matching up group.
    """
    rows, k = int(w.shape[-2]), int(w.shape[-1])
    if rows % 2 != 0:
        raise ValueError(f"expected rows = 2*N, got rows={rows}")
    half = rows // 2
    if half % gran != 0:
        raise ValueError(f"N (half of rows) must be a multiple of {gran}; got N={half}")
    lead = w.shape[:-2]
    gate = w[..., :half, :].reshape(*lead, half // gran, gran, k)
    up = w[..., half:, :].reshape(*lead, half // gran, gran, k)
    return torch.stack([gate, up], dim=-3).contiguous()


def _interleave_gate_up_scale(scale: torch.Tensor) -> torch.Tensor:
    """Move an MX up-weight SfAtom table onto the packed pair rows.

    ``scale`` is ``[..., 2N, K/32]``: one table per expert, in the natural
    row order.  The table is indexed by raw row, so the gate/up pairing the
    weight packing applies has to be applied here as well.  Block-wise f32
    tables stay on the raw ``[gate | up]`` layout and are returned untouched.
    """
    if scale.dtype != torch.uint8:
        return scale
    rows, cols = int(scale.shape[-2]), int(scale.shape[-1])
    if rows % _SF_ATOM_ROWS or cols % _SF_ATOM_COLS:
        raise ValueError(
            "an MX SfAtom table packs whole 128-rows-x-4-blocks atoms; got "
            f"{tuple(scale.shape)}."
        )
    offsets = _sf_atom_offsets(rows, cols, scale.device).reshape(-1)
    tables = scale.reshape(-1, rows * cols)
    plain = tables[:, offsets]
    packed = _interleave_gate_up_weight(plain.reshape(*scale.shape[:-2], rows, cols))
    out = torch.empty_like(tables)
    out[:, offsets] = packed.reshape(-1, rows * cols)
    return out.reshape(scale.shape)


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
