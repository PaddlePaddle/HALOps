# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""mega_mla's plan pass, as its own operator.

One tiny block (`mla_plan.cu`) turns the per-token KV lengths into the per-CTA
work list, the per-token split shape and the barrier targets that the megakernel
reads out of the same workspace.  It is a separate launch for two reasons:

  * the KV lengths are device tensors, so the host cannot derive any of this
    without a sync;
  * it is metadata work, so a benchmark can time it apart from the megakernel
    (`mega_mla(..., plan_ready=True)`).  In production it belongs inside the
    metadata kernel that already runs ahead of this layer -- `build_kv_runs`'s
    docstring says the same thing about the `kv_runs` descriptor.

`mega_mla()` calls this itself unless `plan_ready=True`, so ordinary callers do
not need to touch it.  ⚠️ A stale plan is a hang or a wrong result, not an error.
"""

import torch

from halops import halops_kernel

__all__ = ["mega_mla_plan"]


def mega_mla_plan(
    q_or_x: torch.Tensor,
    kv_runs: torch.Tensor,
    topk: int,
    topk_lengths: torch.Tensor,
    c128_attn_len: torch.Tensor,
    wo_a_weight: torch.Tensor,
    workspace: torch.Tensor,
) -> None:
    """Fill the plan region of `workspace`, which the following `mega_mla` reads.

    Must be followed by a `mega_mla` call with the SAME T / topk / KV lengths and
    the SAME workspace tensor, on the same stream and device.  ⚠️ The workspace is
    an argument rather than being fetched here: `_mega_mla_workspace` may grow and
    REPLACE the cached buffer, which would silently discard this plan.

    ⚠️ `wo_a_weight` is not read as data -- its shape gives the o_lora group count,
    which sizes the barrier banks this pass ZEROES.  That is also why this pass is
    mandatory before the first `mega_mla` of a given shape, not just an
    optimisation: without it those banks start dirty.
    """
    i = torch.empty(0, dtype=torch.int32, device=q_or_x.device)
    halops_kernel.mega_mla_plan(
        topk_lengths,
        c128_attn_len if c128_attn_len is not None else i,
        int(kv_runs.shape[0]),
        int(topk),
        wo_a_weight,
        workspace,
    )
