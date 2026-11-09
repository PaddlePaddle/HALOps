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

from halops import low_latency_moe


def run_low_latency_moe(inp: dict):
    """Run the quantized HALOps latent MoE kernel."""
    low_latency_moe(
        inp["input"],
        inp["router_weight"],
        inp["expert_up_weights"],
        inp["expert_up_weights_scale"],
        inp["expert_down_weights"],
        inp["expert_down_weights_scale"],
        inp["shared_up_weight"],
        inp["shared_up_weight_scale"],
        inp["shared_down_weight"],
        inp["shared_down_weight_scale"],
        inp["latent_down_weight"],
        inp["latent_down_weight_scale"],
        inp["tail_fc2_weight"],
        inp["tail_fc2_scale"],
        inp["tail_norm_weight"],
        inp.get("correction_bias"),
        inp.get("per_expert_scale"),
        inp["out"],
        inp["workspace"],
        inp["top_k"],
        num_router_gates=inp["num_router_gates"],
        scoring_func=inp["scoring_func"],
        renormalize=inp["renormalize"],
        activation=inp["activation"],
        situ_beta=inp["situ_beta"],
        situ_linear_beta=inp["situ_linear_beta"],
        eps=inp["eps"],
        quant_mode=inp["quant_mode"],
        grid_cap=inp["grid_cap"],
        tp_size=inp["tp_size"],
    )
    return inp["out"]
