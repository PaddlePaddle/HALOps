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

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def _dequant_weight(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Decode blockwise FP8 or SfAtom-scaled MXFP8/MXFP4 weights to BF16."""
    if scale.dtype == torch.float32:
        rows, cols = weight.shape[-2:]
        blocks = weight.float().reshape(
            *weight.shape[:-2], rows // 128, 128, cols // 128, 128
        )
        return (blocks * scale[..., :, None, :, None]).reshape(weight.shape).bfloat16()

    from halops.moe.low_latency_moe_utils import _sf_atom_offsets

    rows, groups = scale.shape[-2:]
    offsets = _sf_atom_offsets(rows, groups, scale.device)
    plain = scale.reshape(*scale.shape[:-2], -1)[..., offsets]
    decoded_scale = torch.exp2(plain.float() - 127.0)
    if weight.dtype == torch.uint8:
        lut = torch.tensor(
            [
                0.0,
                0.5,
                1.0,
                1.5,
                2.0,
                3.0,
                4.0,
                6.0,
                -0.0,
                -0.5,
                -1.0,
                -1.5,
                -2.0,
                -3.0,
                -4.0,
                -6.0,
            ],
            device=weight.device,
            dtype=torch.float32,
        )
        codes = torch.stack((weight & 0xF, weight >> 4), dim=-1)
        values = lut[codes.long()]
    else:
        values = weight.float()
    blocks = values.reshape(*scale.shape, 32)
    return (
        (blocks * decoded_scale[..., None])
        .reshape(*weight.shape[:-1], groups * 32)
        .bfloat16()
    )


def _route(inp: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Select experts from the sum of both gates, breaking ties by expert ID.

    ``correction_bias`` steers the selection only; the renormalized combine
    weights keep the raw scores and are then scaled per selected expert.
    """
    x = inp["input_bf16"]
    # The dual router publishes FP32 scores without rounding logits to BF16.
    logits = torch.matmul(x.float(), inp["router_weight_fp32"])
    scores = (
        torch.sigmoid(logits)
        if inp["scoring_func"] == "sigmoid"
        else torch.softmax(logits, dim=-1)
    ).sum(dim=0)
    bias = inp.get("correction_bias")
    selection = scores + bias if bias is not None else scores
    ids = torch.argsort(selection, dim=-1, descending=True, stable=True)[
        :, : inp["top_k"]
    ]
    weights = scores.gather(1, ids)
    if inp["renormalize"]:
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    scale = inp.get("per_expert_scale")
    if scale is not None:
        weights = weights * scale[ids]
    return weights, ids


def _activate(up: torch.Tensor, inp: dict) -> torch.Tensor:
    gate, value = up.float().chunk(2, dim=-1)
    if inp["activation"] == "silu":
        return F.silu(gate) * value
    beta, linear_beta = inp["situ_beta"], inp["situ_linear_beta"]
    capped_gate = beta * torch.tanh(gate / beta) if beta > 0 else gate
    capped_value = (
        linear_beta * torch.tanh(value / linear_beta) if linear_beta > 0 else value
    )
    return capped_gate * torch.sigmoid(gate) * capped_value


@torch.no_grad()
def run_low_latency_moe_golden(inp: dict) -> torch.Tensor:
    """BF16 projections with FP32 routing, activation and normalization."""
    x = inp["input_bf16"]
    weights, ids = _route(inp)
    tokens, top_k = ids.shape
    latent = x @ inp["latent_down_weight"].T
    selected_up = inp["expert_up_weights"][ids.flatten()]
    repeated = latent.repeat_interleave(top_k, dim=0).unsqueeze(1)
    up = torch.bmm(
        repeated,
        selected_up.transpose(1, 2),
    ).squeeze(1)
    routed_act = (_activate(up, inp) * weights.reshape(-1, 1)).bfloat16()
    selected_down = inp["expert_down_weights"][ids.flatten()]
    routed_each = torch.bmm(
        routed_act.unsqueeze(1),
        selected_down.transpose(1, 2),
    ).squeeze(1)
    routed = routed_each.reshape(tokens, top_k, -1).float().sum(dim=1).bfloat16()
    shared_up = x @ inp["shared_up_weight"].T
    shared = _activate(shared_up, inp).bfloat16() @ inp["shared_down_weight"].T
    routed_float = routed.float()
    rms = torch.rsqrt(routed_float.square().mean(dim=-1, keepdim=True) + inp["eps"])
    normed = (routed_float * rms * inp["tail_norm_weight"].float()).bfloat16()
    tail = normed @ inp["tail_fc2_weight"].T
    return (tail.float() + shared.float()).bfloat16()
