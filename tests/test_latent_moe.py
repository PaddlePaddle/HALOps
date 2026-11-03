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

import os
import re
import subprocess
import sys
import zlib

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from halops import (
    get_workspace_for_moe,
    low_latency_moe,
    transform_weights_for_moe,
)

_FP8_MAX = 448.0  # e4m3 dynamic range
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WORLD = 4
_ROW = re.compile(r"\[rank (\d+)\] (\S+): [^\n]*? -> (OK|FAIL)")
_TOL = 1.5e-1


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


def _pow2_block_quant(w):
    """[rows, cols] -> (fp8 e4m3, fp32 scales [rows/128, cols/128]), power-of-two scales."""
    r, c = w.shape
    blocks = w.float().reshape(r // 128, 128, c // 128, 128)
    amax = blocks.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-12)
    s = torch.exp2(torch.ceil(torch.log2(amax / _FP8_MAX)))
    q = (blocks / s).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
    return q.reshape(r, c).contiguous(), s.reshape(r // 128, c // 128).contiguous()


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

    def __init__(self, use_latent=True, activation="silu", **weight_kwargs):
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
        self.init_weights(**weight_kwargs)

    def init_weights(self, *, seed: int = 0, device=None, common_seed=None):
        """Random weights and buffers for the launch this instance describes.

        With ``common_seed`` the router and the latent-down weight come from their own
        stream, so instances that differ in ``seed`` (the ranks of a TP group, each with
        its own experts) still share them.
        """
        dev = torch.device(device or "cuda")

        e, h, n = self.num_experts, self.hidden_size, self.intermediate_size
        z = self.latent_size if self.use_latent else h
        g = torch.Generator(device=dev).manual_seed(seed)
        gc = (
            g
            if common_seed is None
            else torch.Generator(device=dev).manual_seed(common_seed)
        )

        router = (torch.randn(h, e, device=dev, generator=gc) / h**0.5).bfloat16()
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
                torch.randn(1, h, z, device=dev, generator=gc) * 0.1
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

    def init_tail_weights(self, *, seed: int = 1, device=None):
        """The fused tail's operands: fc2 (e4m3, power-of-two 128x128 scales) and the RMSNorm weight."""
        dev = torch.device(device or "cuda")
        g = torch.Generator(device=dev).manual_seed(seed)
        h, z = self.hidden_size, self.latent_size
        self.fc2_weight, self.fc2_scale = _pow2_block_quant(
            torch.randn(h, z, device=dev, generator=g) * 0.05
        )
        self.norm_weight = (
            1.0 + 0.1 * torch.randn(z, device=dev, generator=g)
        ).bfloat16()
        self.eps = 1e-5
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

    def tail(self, routed_outs, shared_outs):
        """The fused tail over the TP ranks' outputs, in fp32:
        fc2(RMSNorm(sum of routed)) + sum of shared."""
        p = self.moe_params
        routed = torch.stack([t.float() for t in routed_outs]).sum(0).bfloat16().float()
        rms = torch.rsqrt(routed.pow(2).mean(-1, keepdim=True) + p.eps)
        normed = (routed * rms * p.norm_weight.float()).bfloat16().float()
        fc2 = p.fc2_weight.float() * p.fc2_scale.repeat_interleave(
            128, 0
        ).repeat_interleave(128, 1)
        return normed @ fc2.t() + torch.stack([t.float() for t in shared_outs]).sum(0)

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

    def __init__(self, moe_params, tp, grid_cap=0):
        self.moe_params = moe_params
        self.tp = tp
        self.grid_cap = grid_cap
        self.workspace = None
        self.packed_weights = None

    def __call__(self, input, top_k):
        p = self.moe_params
        num_tokens = input.size(0)
        if self.packed_weights is None:
            self.packed_weights = transform_weights_for_moe(
                p.router_weight,
                p.expert_up_weights,
                p.shared_up_weight,
                p.latent_down_weight,
                num_tokens,
                p.num_router_gates,
            )
        if self.workspace is None:
            self.workspace = get_workspace_for_moe(input, *self.tp)

        def empty(cols):
            return torch.empty(
                num_tokens, cols, dtype=torch.bfloat16, device=input.device
            )

        out = empty(p.hidden_size)
        low_latency_moe(
            input,
            self.packed_weights["router_weight"],
            self.packed_weights["expert_up_weights"],
            p.expert_up_weights_scale,
            p.expert_down_weights,
            p.expert_down_weights_scale,
            self.packed_weights["shared_up_weight"],
            p.shared_up_weight_scale,
            p.shared_down_weight,
            p.shared_down_weight_scale,
            self.packed_weights["latent_down_weight"],
            p.latent_down_weight_scale,
            p.fc2_weight,
            p.fc2_scale,
            p.norm_weight,
            out,
            self.workspace,
            top_k,
            p.num_router_gates,
            p.scoring_func,
            p.renormalize,
            p.activation,
            p.situ_beta,
            p.situ_linear_beta,
            eps=p.eps,
            grid_cap=self.grid_cap,
        )
        return out


def _cosine(a, b):
    """Flattened cosine similarity of two tensors."""
    return F.cosine_similarity(
        a.float().reshape(-1), b.float().reshape(-1), dim=0
    ).item()


def test_transform_weights_partial():
    """Packing the operands one at a time gives the same tensors as packing them together."""
    p = MoEParams(use_latent=True)
    full = transform_weights_for_moe(
        p.router_weight,
        p.expert_up_weights,
        p.shared_up_weight,
        p.latent_down_weight,
        16,
    )
    assert set(full) == {
        "router_weight",
        "expert_up_weights",
        "shared_up_weight",
        "latent_down_weight",
    }
    parts = [
        transform_weights_for_moe(p.router_weight, None, None, num_tokens=16),
        transform_weights_for_moe(None, p.expert_up_weights, None),
        transform_weights_for_moe(None, None, p.shared_up_weight, p.latent_down_weight),
    ]
    assert [sorted(d) for d in parts] == [
        ["router_weight"],
        ["expert_up_weights"],
        ["latent_down_weight", "shared_up_weight"],
    ]
    for d in parts:
        for key, value in d.items():
            assert torch.equal(value, full[key]), key


def _all_gather(t, world):
    gathered = [torch.empty_like(t) for _ in range(world)]
    dist.all_gather(gathered, t)
    return gathered


def _check_outputs(ref, x, top_k, out):
    """(ok, text) for one launch: the end-to-end output against MoERef (all ranks' routed / shared, then the tail)."""
    expected = ref.tail(*(_all_gather(t, _WORLD) for t in ref(x, top_k)))
    err = (out.float() - expected).norm(dim=-1) / expected.norm(dim=-1)
    ok = bool((err < _TOL).all()) and bool(torch.isfinite(out).all())
    return ok, f"max_token_rel_l2={err.max().item():.2e}"


def _run_accuracy(p, tp, grid_cap, m, top_k, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, p.hidden_size, device="cuda", generator=g).bfloat16()
    out = MoEFused(p, tp, grid_cap)(x, top_k)
    torch.cuda.synchronize()
    return _check_outputs(MoERef(p), x, top_k, out)


def _run_graph(p, tp, grid_cap, m, top_k, seed):
    """CUDA graph capture + multi-replay must produce correct results."""
    ref = MoERef(p)
    x = torch.randn(m, p.hidden_size, device="cuda", dtype=torch.bfloat16)
    fused = MoEFused(p, tp, grid_cap)
    # Warmup: packs the weights and allocates the workspace outside the capture.
    for _ in range(3):
        fused(x, top_k)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fused(x, top_k)
    torch.cuda.synchronize()
    ok, texts = True, []
    for i in range(4):
        g = torch.Generator(device="cuda").manual_seed(seed + 200 + i)
        x.copy_(torch.randn(m, p.hidden_size, device="cuda", generator=g))
        graph.replay()
        torch.cuda.synchronize()
        good, text = _check_outputs(ref, x, top_k, out)
        ok &= good
        texts.append(text)
    return ok, f"replays: {' | '.join(texts)}"


def _tail_worker():
    """One rank of the 4-GPU run (launched by ``moe_run`` or directly under torchrun)."""
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    torch.set_default_device(dev)
    dist.init_process_group("nccl", device_id=dev)
    group = dist.group.WORLD.group_name
    tp = (group, rank, world)
    grid_cap = int(os.environ.get("TAIL_GRID_CAP", "0"))

    params = {}

    def get_params(activation):
        if activation not in params:
            p = MoEParams(
                use_latent=True,
                activation=activation,
                seed=100 + rank,
                common_seed=0,
                device=dev,
            )
            params[activation] = p.init_tail_weights(device=dev)
        return params[activation]

    ok = True
    for case in os.environ["MOE_CASES"].split(","):
        fields = case.split("-")
        m = int(fields[1][1:])
        if fields[0] == "accuracy":
            top_k, activation = int(fields[2][1:]), fields[3]
            run = _run_accuracy
        else:  # graph
            top_k, activation = 8, "silu"
            run = _run_graph
        seed = zlib.crc32(case.encode())
        good, text = run(get_params(activation), tp, grid_cap, m, top_k, seed)
        ok &= good
        print(f"[rank {rank}] {case}: {text} -> {'OK' if good else 'FAIL'}", flush=True)
    dist.barrier()
    if rank == 0:
        print("RESULT:", "ALL_OK" if ok else "FAILED", flush=True)
    dist.destroy_process_group()
    sys.exit(0 if ok else 1)


def _case_id(item):
    """The worker case of an accuracy / CUDA graph test item; None for the other tests."""
    callspec = getattr(item, "callspec", None)
    if callspec is None:
        return None
    p = callspec.params
    if item.originalname == "test_moe_accuracy":
        return f"accuracy-m{p['num_tokens']}-k{p['top_k']}-{p['activation']}"
    if item.originalname == "test_moe_cuda_graph_replay":
        return f"graph-m{p['num_tokens']}"
    return None


@pytest.fixture(scope="module")
def moe_run(request):
    """Run the worker once under torchrun for the cases of the selected tests; return its output by (case, rank)."""
    if torch.cuda.device_count() < _WORLD:
        pytest.skip(f"needs {_WORLD} GPUs")
    cases = [case for case in map(_case_id, request.session.items) if case]
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={_WORLD}",
        os.path.abspath(__file__),
    ]
    env = dict(os.environ, MOE_CASES=",".join(cases), PYTHONPATH=_ROOT)
    proc = subprocess.run(
        cmd, cwd=_ROOT, env=env, capture_output=True, text=True, timeout=3600
    )
    out = proc.stdout + proc.stderr
    if "NVLS multicast" in out:
        pytest.skip("the fused tail needs an NVLS multicast capable node")
    status = {}
    for rank, case, res in _ROW.findall(out):
        status[(case, int(rank))] = res == "OK"
    return status, out


def _check_case(moe_run, case):
    status, out = moe_run
    ranks = [status.get((case, r)) for r in range(_WORLD)]
    errors = "\n".join(line for line in out.splitlines() if "Error" in line)
    details = f"{errors[-1500:]}\n--- end of worker output ---\n{out[-1500:]}"
    assert None not in ranks, f"no result for {case} on every rank:\n{details}"
    assert all(ranks), f"{case} failed on some rank:\n{details}"


@pytest.mark.parametrize("num_tokens", [1, 2, 4, 8, 9, 16])
@pytest.mark.parametrize("top_k", [1, 8, 10])
@pytest.mark.parametrize("activation", ["silu", "situ"])
def test_moe_accuracy(request, moe_run, num_tokens, top_k, activation):
    _check_case(moe_run, _case_id(request.node))


@pytest.mark.parametrize("num_tokens", [1, 2, 4, 8, 16])
def test_moe_cuda_graph_replay(request, moe_run, num_tokens):
    _check_case(moe_run, _case_id(request.node))


if __name__ == "__main__":
    if "WORLD_SIZE" in os.environ:
        _tail_worker()
    else:
        pytest.main([__file__, "-v", "-s"])
