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

from typing import Optional

import torch

import halops.halops_kernel as halops_kernel

__all__ = ["mla_attention_plan"]


def mla_attention_plan(
    kv_indices: torch.Tensor,
    topk_lengths: Optional[torch.Tensor],
    page_size: int,
    kv_run_step: int,
    c128_attn_len: Optional[torch.Tensor],
    workspace: torch.Tensor,
) -> torch.Tensor:
    """Fill the plan slot of one layer kind in ``workspace`` and zero its
    barriers.

    ``workspace`` keeps one plan per layer kind: SWA (``c128_attn_len`` is
    None) and HCA.  Run it once per forward per kind, before that kind's
    ``mla_attention`` launches on the same stream with the same T; layers of
    one kind share the plan as long as every launch's barriers are zeroed after
    it (by ``mla_attention`` itself, or by ``mla_output_proj``).

    The plan compresses each token's SWA window into at most two runs of
    consecutive slots (a run ends where the next slot is not the previous one
    plus ``kv_run_step`` or lies on another page).  A window of more than two
    runs (a padding or not yet written window) is cut to its first row, as
    sglang pads such rows to length 1.  The block counts and the attention use
    these effective lengths.

    Args:
        kv_indices: [T, topk] int32 SWA ``kv_cache`` slots of each token's
            window, unit column stride, T > 0; columns past a token's length
            are ignored (e.g. -1 padding).
        topk_lengths: [T] int32 contiguous SWA rows each token attends over,
            including its own new row, or None if every token attends over
            ``topk`` rows.
        page_size: Rows per ``kv_cache`` page, a power of two.
        kv_run_step: Slot step between consecutive window rows, 1 or -1 (a
            property of the slot table's producer).
        c128_attn_len: [T] int32 compressed rows each token attends over, or
            None on SWA layers; selects the plan slot.
        workspace: uint8 workspace from ``mla_workspace``.

    Returns:
        [T, 4] int32 view of the plan's per-token KV runs in ``workspace``:
        (slot0, len0, slot1, rows), row g < len0 of the window on slot
        ``slot0 + g``, the others on ``slot1 + g - len0``, rows the effective
        length.  Valid until the next plan of this kind.
    """
    return halops_kernel.mla_attention_plan(
        kv_indices,
        topk_lengths,
        int(page_size),
        int(kv_run_step),
        c128_attn_len,
        workspace,
    )
