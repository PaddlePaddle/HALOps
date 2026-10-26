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

import halops.halops_kernel as halops_kernel

__all__ = ["mla_workspace"]


def mla_workspace(
    max_tokens: int,
    wq_a_weight: torch.Tensor,
    wkv_weight: torch.Tensor,
    wo_a_weight: torch.Tensor,
    wo_b_weight: torch.Tensor,
) -> torch.Tensor:
    """Allocate the gmem workspace of ``mla_attention_plan``,
    ``mla_attention`` and ``mla_output_proj``.

    One workspace serves every layer of the model (SWA and HCA alike, same
    weight shapes), every CUDA graph and any T up to ``max_tokens``.  It holds
    the barriers, one plan per layer kind, the attention partials and
    activations, and the output projection's buffers.  Its barrier prefix is
    zeroed here, once; the kernels keep it zeroed.

    Call order: ``mla_workspace`` once (before any CUDA-graph capture); per
    forward one ``mla_attention_plan`` per layer kind (required: the attention
    reads its KV runs and work list); per layer ``mla_attention`` then
    ``mla_output_proj``.

    Args:
        max_tokens: Largest T the workspace serves, in [1, 640].
        wq_a_weight: [1024, H] float8_e4m3fn wq_a weight; sets H.
        wkv_weight: [512, H] float8_e4m3fn wkv weight.
        wo_a_weight: [2 * H, 64 * 512 / G] float8_e4m3fn wo_a weight.
        wo_b_weight: [H, 2 * H] float8_e4m3fn wo_b weight.

    Returns:
        1-D uint8 workspace tensor on the device of ``wq_a_weight``.
    """
    total, zero_prefix = halops_kernel.mla_workspace_bytes(
        int(max_tokens), wq_a_weight, wkv_weight, wo_a_weight, wo_b_weight
    )
    workspace = torch.empty(total, dtype=torch.uint8, device=wq_a_weight.device)
    workspace[:zero_prefix].zero_()
    return workspace
