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

import pytest
import torch
import torch.nn.functional as F

from halops import (
    get_workspace_for_moe,
    low_latency_moe,
    transform_weights_for_moe,
)

_FP8_MAX = 448.0  # e4m3 dynamic range


def _quant_fp8_block_wise(w, block_row=128, block_col=128):
    """Block-wise FP8 quantization. w: [E, rows, cols] -> (fp8, scales)."""
    Ee, rows, cols = w.shape
    rb = (rows + block_row - 1) // block_row
    cb = (cols + block_col - 1) // block_col
    wf = w.float()
    scales = torch.zeros(Ee, rb, cb, device=w.device, dtype=torch.float32)
    w_fp8 = torch.zeros_like(wf)
    for ri in range(rb):
        r0, r1 = ri * block_row, min((ri + 1) * block_row, rows)
        for ci in range(cb):
            c0, c1 = ci * block_col, min((ci + 1) * block_col, cols)
            block = wf[:, r0:r1, c0:c1]
            amax = block.abs().amax(dim=(1, 2), keepdim=True).clamp(min=1e-12)
            s = amax / _FP8_MAX
            scales[:, ri, ci] = s.reshape(Ee)
            w_fp8[:, r0:r1, c0:c1] = (block / s).clamp(-_FP8_MAX, _FP8_MAX)
    return w_fp8.to(torch.float8_e4m3fn), scales


def _quant_fp8_group_wise(x, group_size=128):
    """Per-token-dynamic quant over the last dim; `x` is [..., K], bf16 or fp32."""
    x_float = x.float()
    shape = x_float.shape
    groups = shape[-1] // group_size
    blocks = x_float.reshape(*shape[:-1], groups, group_size)
    amax = blocks.abs().amax(dim=-1).clamp(min=1e-10)
    scales = amax * (1.0 / _FP8_MAX)
    q = (blocks / scales[..., None]).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
    return q.reshape(shape), scales


class MoEParams:
    """One `low_latency_moe` call: the knobs, plus the operands `init_weights` fills."""

    def __init__(self, use_latent=True, activation="silu"):
        self.hidden_size = 4096
        self.intermediate_size = 512
        self.latent_size = 2048
        self.num_experts = 512
        self.num_router_gates = 2
        self.use_latent = use_latent
        self.scoring_func = "sigmoid"
        self.renormalize = True
        self.activation = activation
        self.situ_beta = 4.0
        self.situ_linear_beta = 25.0
        self.init_weights()

    def init_weights(self, *, seed: int = 0, device=None):
        """Random weights and buffers for the launch this instance describes."""
        dev = torch.device(device or "cuda")

        e, h, n = self.num_experts, self.hidden_size, self.intermediate_size
        z = self.latent_size if self.use_latent else h
        g = torch.Generator(device=dev).manual_seed(seed)

        router = (torch.randn(h, e, device=dev, generator=g) / h**0.5).bfloat16()
        if self.num_router_gates == 2:
            router = torch.stack([router, router.flip(0)])

        w13, s13 = _quant_fp8_block_wise(
            torch.randn(e, 2 * n, z, device=dev, generator=g) * 0.1
        )
        w2, s2 = _quant_fp8_block_wise(
            torch.randn(e, z, n, device=dev, generator=g) * 0.1
        )
        latent_w = latent_s = None
        if self.use_latent:
            # The kernel reads [L, H], so the scales are transposed.
            lw, ls = _quant_fp8_block_wise(
                torch.randn(1, h, z, device=dev, generator=g) * 0.1
            )
            latent_w = lw[0]
            latent_s = ls[0].t().contiguous()

        self.router_weight = router
        self.expert_up_weights = w13
        self.expert_up_weights_scale = s13
        self.expert_down_weights = w2
        self.expert_down_weights_scale = s2
        self.latent_down_weight = latent_w
        self.latent_down_weight_scale = latent_s

        shared_up, shared_up_s = _quant_fp8_block_wise(
            torch.randn(1, 2 * n, h, device=dev, generator=g) * 0.1
        )
        shared_down, shared_down_s = _quant_fp8_block_wise(
            torch.randn(1, h, n, device=dev, generator=g) * 0.1
        )
        self.shared_up_weight = shared_up[0]
        self.shared_up_weight_scale = shared_up_s[0]
        self.shared_down_weight = shared_down[0]
        self.shared_down_weight_scale = shared_down_s[0]
        return self


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)


def _situ_glu(gate, up, beta, linear_beta):
    """SiTU: sigmoid(gate) times the soft-capped gate and up operands."""

    def _soft_cap(x, beta):
        return x if beta <= 0.0 else beta * torch.tanh(x / beta)

    return _soft_cap(gate, beta) * torch.sigmoid(gate) * _soft_cap(up, linear_beta)


class MoERef:
    """Reference implementation of the latent MoE, in fp32 torch ops."""

    def __init__(self, moe_params):
        self.moe_params = moe_params

    def _router_scores(self, input):
        """The router's selection key: sigmoid of each gate's logits, summed."""
        router_weight = self.moe_params.router_weight
        logits = (input.bfloat16() @ router_weight.bfloat16()).float()
        if router_weight.dim() == 2:
            # The kernel round-trips the logits through bf16 before scoring.
            logits = logits.bfloat16()
        scores = torch.sigmoid(logits.float())
        return scores.sum(0) if scores.dim() == 3 else scores

    def _routing_sigmoid_topk_from_scores(self, scores, top_k):
        """The greedy topk->renormalize over already-scored rows."""
        M = scores.shape[0]
        ids = torch.zeros(M, top_k, dtype=torch.int64, device=scores.device)
        weights = torch.zeros(M, top_k, dtype=torch.float32, device=scores.device)
        s = scores.clone()
        for k in range(top_k):
            v, idx = s.max(dim=-1)
            weights[:, k] = v
            ids[:, k] = idx
            s.scatter_(1, idx.unsqueeze(1), float("-inf"))
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        return weights, ids

    def matmul_fp8(self, x_fp8, y_fp8, x_scale, y_scale):
        """Return dequantized ``x @ y.T`` in FP32.

        Activations are [K] or [M, K], with one scale per 64/128 K values.
        Weights are [N, K], with one scale per 128x128 block.
        """
        assert x_fp8.ndim in (1, 2)
        assert y_fp8.ndim == 2 and x_fp8.shape[-1] == y_fp8.shape[-1]
        assert x_scale.ndim == x_fp8.ndim
        assert x_scale.shape[:-1] == x_fp8.shape[:-1]
        assert x_scale.shape[-1] > 0
        group_size, remainder = divmod(x_fp8.shape[-1], x_scale.shape[-1])
        assert remainder == 0 and group_size in (64, 128)

        def dequant(weight, scale):
            rows, cols = weight.shape
            assert scale.shape == ((rows + 127) // 128, (cols + 127) // 128)
            expanded = scale.float().repeat_interleave(128, 0).repeat_interleave(128, 1)
            return weight.float() * expanded[:rows, :cols]

        blocks = x_fp8.float().reshape(*x_scale.shape, group_size)
        activations = (blocks * x_scale.float().unsqueeze(-1)).reshape(x_fp8.shape)
        return activations @ dequant(y_fp8, y_scale).T

    def _shared_expert(self, input):
        """The shared expert: gate/up over the hidden space, then down."""
        n = self.moe_params.intermediate_size

        input_fp8, input_scale = _quant_fp8_group_wise(input, group_size=128)
        up_out = self.matmul_fp8(
            input_fp8,
            self.moe_params.shared_up_weight,
            input_scale,
            self.moe_params.shared_up_weight_scale,
        )
        act = self._gated_act(up_out[:, :n], up_out[:, n:], 1.0)

        act_fp8, act_scale = _quant_fp8_group_wise(act, group_size=64)
        return self.matmul_fp8(
            act_fp8,
            self.moe_params.shared_down_weight,
            act_scale,
            self.moe_params.shared_down_weight_scale,
        ).bfloat16()

    def _gated_act(self, gate, up, rw):
        """The kernel's epilogue combine, fp32; the knobs come from the params."""
        p = self.moe_params
        if p.activation == "silu":
            return rw * up * F.silu(gate)
        return rw * _situ_glu(gate, up, p.situ_beta, p.situ_linear_beta)

    def routed_experts(
        self,
        z,
        up_weights,
        up_scales,
        down_weights,
        down_scales,
        topk_w,
        topk_ids,
        n,
        k,
    ):
        """Block-FP8 expert stage over `z`, the activation the experts read."""
        out = torch.zeros(z.shape[0], k, device=z.device, dtype=torch.float32)
        for tok in range(z.shape[0]):
            xq, x_scales = _quant_fp8_group_wise(z[tok].float(), group_size=128)
            for ki in range(topk_ids.shape[1]):
                eid = int(topk_ids[tok, ki])
                weight = float(topk_w[tok, ki])
                raw = self.matmul_fp8(xq, up_weights[eid], x_scales, up_scales[eid])
                act = self._gated_act(raw[:n], raw[n:], weight)
                sq, act_scale = _quant_fp8_group_wise(act, group_size=64)
                out[tok] += self.matmul_fp8(
                    sq,
                    down_weights[eid],
                    act_scale,
                    down_scales[eid],
                )
        return out.bfloat16()

    def __call__(self, input, top_k):
        shared_out = self._shared_expert(input)

        scores = self._router_scores(input)
        topk_w, topk_ids = self._routing_sigmoid_topk_from_scores(scores, top_k)

        input_fp8, input_scale = _quant_fp8_group_wise(input, group_size=128)
        latent_out = self.matmul_fp8(
            input_fp8,
            self.moe_params.latent_down_weight.T,
            input_scale,
            self.moe_params.latent_down_weight_scale,
        ).bfloat16()

        routed_out = self.routed_experts(
            latent_out,
            self.moe_params.expert_up_weights,
            self.moe_params.expert_up_weights_scale,
            self.moe_params.expert_down_weights,
            self.moe_params.expert_down_weights_scale,
            topk_w,
            topk_ids,
            self.moe_params.intermediate_size,
            latent_out.shape[1],
        )
        return routed_out, shared_out


class MoEFused:
    """The prebuilt op behind one object: the weights are packed once."""

    def __init__(self, moe_params):
        self.moe_params = moe_params
        self.workspace = None
        self.packed_weights = None

    def __call__(self, input, top_k):
        num_tokens = input.size(0)
        if self.packed_weights is None:
            self.packed_weights = transform_weights_for_moe(
                self.moe_params.router_weight,
                self.moe_params.expert_up_weights,
                self.moe_params.shared_up_weight,
                self.moe_params.latent_down_weight,
                num_tokens,
                self.moe_params.num_router_gates,
            )

        if self.workspace is None:
            self.workspace = get_workspace_for_moe(input)

        routed_out = torch.empty(
            num_tokens,
            self.moe_params.latent_size,
            dtype=torch.bfloat16,
            device=input.device,
        )
        shared_out = torch.empty(
            num_tokens,
            self.moe_params.hidden_size,
            dtype=torch.bfloat16,
            device=input.device,
        )
        low_latency_moe(
            input,
            self.packed_weights["router_weight"],
            self.packed_weights["expert_up_weights"],
            self.moe_params.expert_up_weights_scale,
            self.moe_params.expert_down_weights,
            self.moe_params.expert_down_weights_scale,
            self.packed_weights["shared_up_weight"],
            self.moe_params.shared_up_weight_scale,
            self.moe_params.shared_down_weight,
            self.moe_params.shared_down_weight_scale,
            self.packed_weights.get("latent_down_weight"),
            self.moe_params.latent_down_weight_scale,
            self.workspace,
            routed_out,
            shared_out,
            top_k,
            self.moe_params.num_router_gates,
            self.moe_params.scoring_func,
            self.moe_params.renormalize,
            self.moe_params.activation,
            self.moe_params.situ_beta,
            self.moe_params.situ_linear_beta,
        )
        return routed_out, shared_out


def _cosine(a, b):
    """Flattened cosine similarity of two tensors."""
    return F.cosine_similarity(
        a.float().reshape(-1), b.float().reshape(-1), dim=0
    ).item()


def _run_and_compare(moe_params, num_tokens=1, top_k=8):
    """Run moe against the block-FP8 reference; return (cos routed, cos shared)."""
    dev = torch.device("cuda")
    input = torch.randn(
        num_tokens, moe_params.hidden_size, device=dev, dtype=torch.bfloat16
    )

    moe_ref = MoERef(moe_params)
    moe_fused = MoEFused(moe_params)

    routed_out_ref, shared_out_ref = moe_ref(input, top_k)
    routed_out, shared_out = moe_fused(input, top_k)

    cos_routed = _cosine(routed_out, routed_out_ref)
    cos_shared = _cosine(shared_out, shared_out_ref)
    return cos_routed, cos_shared


@pytest.mark.parametrize("num_tokens", [1, 2, 4, 8, 9, 16])
@pytest.mark.parametrize("top_k", [1, 8, 10])
@pytest.mark.parametrize("activation", ["silu", "situ"])
def test_moe_accuracy(num_tokens, top_k, activation):
    moe_params = MoEParams(use_latent=True, activation=activation)
    cos_routed, cos_shared = _run_and_compare(moe_params, num_tokens, top_k)
    print(
        f"\n[moe] num_experts={moe_params.num_experts} hidden_size={moe_params.hidden_size} intermediate_size={moe_params.intermediate_size} num_tokens={num_tokens} top_k={top_k}: cos_sim={cos_routed:.5f}, {cos_shared:.5f}"
    )
    assert cos_routed > 0.98, f"cosine similarity too low: {cos_routed:.5f}"
    assert cos_shared > 0.98, f"cosine similarity too low: {cos_shared:.5f}"


@pytest.mark.parametrize("num_tokens", [1, 2, 4, 8, 16])
def test_moe_cuda_graph_replay(num_tokens):
    """CUDA graph capture + multi-replay must produce correct results."""
    top_k = 8
    dev = torch.device("cuda")
    moe_params = MoEParams(use_latent=True)
    moe_params.init_weights()

    input = torch.randn(
        num_tokens, moe_params.hidden_size, device=dev, dtype=torch.bfloat16
    )

    moe_ref = MoERef(moe_params)
    moe_fused = MoEFused(moe_params)

    # Warmup: packs the weights and allocates the workspace outside the capture.
    for _ in range(3):
        moe_fused(input, top_k)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        routed_out, shared_out = moe_fused(input, top_k)
    torch.cuda.synchronize()

    for i in range(4):
        torch.manual_seed(200 + i)
        input.copy_(torch.randn_like(input))

        graph.replay()
        torch.cuda.synchronize()

        routed_ref, shared_ref = moe_ref(input, top_k)
        cos = _cosine(routed_out, routed_ref)
        cos_shared = _cosine(shared_out, shared_ref)
        print(
            f"\n[moe graph] m={num_tokens} replay={i}: cos={cos:.5f}, {cos_shared:.5f}"
        )
        assert cos > 0.98, f"graph replay {i}: routed cos too low: {cos:.5f}"
        assert (
            cos_shared > 0.98
        ), f"graph replay {i}: shared cos too low: {cos_shared:.5f}"
        assert torch.isfinite(
            routed_out.float()
        ).all(), f"graph replay {i}: NaN/Inf in output"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
