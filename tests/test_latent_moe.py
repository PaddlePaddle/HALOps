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

import halops.moe as op

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


def _quant_fp8_per_token(x, block_size=128):
    """Per-token-dynamic quant over the last dim; `x` is [..., K], bf16 or fp32."""
    x_float = x.float()
    shape = x_float.shape
    groups = shape[-1] // block_size
    blocks = x_float.reshape(*shape[:-1], groups, block_size)
    amax = blocks.abs().amax(dim=-1).clamp(min=1e-10)
    scales = amax * (1.0 / _FP8_MAX)
    q = (blocks / scales[..., None]).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
    return q.reshape(shape), scales


class MoEParams:
    """One `op.moe` call: the knobs, plus the operands `init_weights` fills."""

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


_SHARED_CACHE: dict = {}


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)


def _shared_operands(device, m: int):
    """Placeholder shared-expert operands (the tests ignore its output)."""
    key = (str(device), m)
    cached = _SHARED_CACHE.get(key)
    if cached is None:
        h, n = 4096, 512
        up = torch.zeros(2 * n, h, dtype=torch.float8_e4m3fn, device=device)
        up_s = torch.full((2 * n // 128, h // 128), 1.0, device=device)
        down = torch.zeros(h, n, dtype=torch.float8_e4m3fn, device=device)
        down_s = torch.full((h // 128, n // 128), 1.0, device=device)
        out = torch.empty(m, h, dtype=torch.bfloat16, device=device)
        cached = (up, up_s, down, down_s, out)
        _SHARED_CACHE[key] = cached
    return cached


def _block_wise_gemm(
    w_fp8,
    scales_bw,
    x_fp8,
    x_scales,
    block_row=128,
    block_col=128,
    act_group_size=128,
):
    """Dequantize each reduction block before accumulating FP32 outputs."""
    rows, cols = w_fp8.shape
    assert block_col % act_group_size == 0
    assert x_scales.numel() == cols // act_group_size
    wf, xf = w_fp8.bfloat16(), x_fp8.bfloat16()  # matmul in bf16
    rb, cb = scales_bw.shape
    assert rb == (rows + block_row - 1) // block_row
    assert cb == (cols + block_col - 1) // block_col
    result = torch.zeros(rows, device=w_fp8.device, dtype=torch.float32)
    for ri in range(rb):
        r0, r1 = ri * block_row, min((ri + 1) * block_row, rows)
        for ci in range(cb):
            c0, c1 = ci * block_col, min((ci + 1) * block_col, cols)
            partial = None
            for a0 in range(c0, c1, act_group_size):
                a1 = min(a0 + act_group_size, c1)
                subblock = (wf[r0:r1, a0:a1] @ xf[a0:a1]) * x_scales[
                    a0 // act_group_size
                ]
                partial = subblock if partial is None else partial + subblock
            result[r0:r1] += partial * scales_bw[ri, ci]
    return result


def _routing_sigmoid_topk_from_scores(scores, top_k):
    """The greedy topk->renormalize over already-scored rows."""
    M = scores.shape[0]
    ids = torch.zeros(M, top_k, dtype=torch.int64, device=scores.device)
    wts = torch.zeros(M, top_k, dtype=torch.float32, device=scores.device)
    s = scores.clone()
    for k in range(top_k):
        v, idx = s.max(dim=-1)
        wts[:, k] = v
        ids[:, k] = idx
        s.scatter_(1, idx.unsqueeze(1), float("-inf"))
    wts = wts / wts.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    return wts, ids


def _soft_cap(x, beta):
    """SiTU's scaled-tanh soft cap: beta * tanh(x / beta), identity when <= 0."""
    return x if beta <= 0.0 else beta * torch.tanh(x / beta)


def _router_logits(x, router_weight):
    """The router Linear, at the shapes the tests build (bf16 matmul, FP32)."""
    row = (x.bfloat16() @ router_weight.bfloat16()).float()
    return row.bfloat16() if router_weight.dim() == 2 else row


class MoERef:
    """Reference implementation of the latent MoE, in fp32 torch ops."""

    def __init__(self, moe_params):
        self.moe_params = moe_params

    def _router_logits(self, input):
        """The router Linear as the kernel computes it: bf16 operands, FP32 result."""
        router_weight = self.moe_params.router_weight
        row = (input.bfloat16() @ router_weight.bfloat16()).float()
        return row.bfloat16() if router_weight.dim() == 2 else row

    def _shared_expert(self, input):
        """The shared expert: gate/up over the hidden space, then down."""
        p = self.moe_params
        n = p.intermediate_size

        def dequant(weight, scale):
            rows, cols = weight.shape
            blocks = weight.float().reshape(rows // 128, 128, cols // 128, 128)
            return (blocks * scale[:, None, :, None]).reshape(rows, cols)

        up = dequant(p.shared_up_weight, p.shared_up_weight_scale)  # [2N, H]
        down = dequant(p.shared_down_weight, p.shared_down_weight_scale)  # [H, N]

        su = (input.bfloat16() @ up.bfloat16().T).float()
        act = self._gated_act(su[:, :n], su[:, n:], 1.0)

        m = act.shape[0]
        q, scale = _quant_fp8_per_token(act, block_size=64)
        dequant = (q.float().reshape(m, n // 64, 64) * scale[..., None]).reshape(m, n)
        return (dequant.bfloat16() @ down.bfloat16().T).bfloat16()

    def _gated_act(self, gate, up, rw):
        """The kernel's epilogue combine, fp32; the knobs come from the params."""
        p = self.moe_params
        activation, situ_beta, situ_linear_beta = (
            p.activation,
            p.situ_beta,
            p.situ_linear_beta,
        )
        if activation == "silu":
            return rw * (up * gate) / (1.0 + torch.exp(-gate))
        return (
            rw
            * _soft_cap(gate, situ_beta)
            * torch.sigmoid(gate)
            * _soft_cap(up, situ_linear_beta)
        )

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
            xq, x_scales = _quant_fp8_per_token(z[tok].float(), block_size=128)
            for ki in range(topk_ids.shape[1]):
                eid = int(topk_ids[tok, ki])
                weight = float(topk_w[tok, ki])
                raw = _block_wise_gemm(up_weights[eid], up_scales[eid], xq, x_scales)
                act = self._gated_act(raw[:n], raw[n:], weight)
                sq, act_scale = _quant_fp8_per_token(act, block_size=64)
                out[tok] += _block_wise_gemm(
                    down_weights[eid],
                    down_scales[eid],
                    sq,
                    act_scale,
                    act_group_size=64,
                )
        return out.bfloat16()

    def __call__(self, input, top_k):
        shared_out = self._shared_expert(input)

        scores = _router_scores(self._router_logits(input))
        topk_w, topk_ids = _routing_sigmoid_topk_from_scores(scores, top_k)

        # z = x @ W_latent^T; the [L/128, H/128] scales transpose onto [H, L].
        weight = self.moe_params.latent_down_weight.float()
        scale = self.moe_params.latent_down_weight_scale
        dequant = weight * scale.T.repeat_interleave(128, 0).repeat_interleave(128, 1)
        latent_out = (input.bfloat16() @ dequant.bfloat16()).bfloat16()

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
            self.packed_weights = op.transform_weights_for_moe(
                self.moe_params.router_weight,
                self.moe_params.expert_up_weights,
                self.moe_params.shared_up_weight,
                self.moe_params.latent_down_weight,
                num_tokens,
                self.moe_params.num_router_gates,
            )

        if self.workspace is None:
            self.workspace = op.get_workspace_for_moe(input)

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
        op.moe(
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


def _router_scores(logits):
    """The router's selection key: sigmoid of each gate's logits, summed."""
    scores = torch.sigmoid(logits.float())
    return scores.sum(0) if scores.dim() == 3 else scores


def _make_weights(dev, E, N, K, scale=0.1, seed=42):
    """Quantized block-FP8 up/down weights (fp8 tensors + scales) for a shape."""
    g = torch.Generator(device=dev).manual_seed(seed)
    w13_f = torch.randn(E, 2 * N, K, device=dev, generator=g) * scale
    w2_f = torch.randn(E, K, N, device=dev, generator=g) * scale
    w13_fp8, s13 = _quant_fp8_block_wise(w13_f)
    w2_fp8, s2 = _quant_fp8_block_wise(w2_f)
    return w13_fp8, s13, w2_fp8, s2


def _router_weight(dev, K, E, x_std=1.0, seed=0):
    """Random bf16 router Linear weight ``[K, E]`` giving unit-variance logits."""
    g = torch.Generator(device=dev).manual_seed(seed)
    w = torch.randn(K, E, device=dev, generator=g) / (x_std * K**0.5)
    return w.bfloat16()


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
