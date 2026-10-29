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

# Native BF16 (mode="bf16") accuracy tests for the batch / grouped-masked /
# grouped-contiguous GEMM entry points. These share the self-developed
# CollectiveMainloop path with W8A8/W4A4 but take the compile-time KIsBF16
# branch (no scale factors, no alpha). Golden reference is a plain fp32 matmul
# rounded to bf16, so tolerances are much tighter than the quantized paths.

import numpy as np
import torch

from halops import batch_gemm, group_contiguous_gemm, group_masked_gemm

torch.manual_seed(2026)
torch.cuda.manual_seed_all(2026)
torch.set_default_device("cuda")

# bf16 output is rounded once by the kernel and once by the reference; a couple
# of ulps of slack is enough. This is far tighter than fp8 (atol 2e-3 rtol 2e-2)
# in relative terms because there is no dequantization error.
RTOL = 1e-2
ATOL = 1e-2


def _assert_close(out, ref, tag):
    out_np = out.to(torch.float32).cpu().numpy()
    ref_np = ref.to(torch.float32).cpu().numpy()
    diff = np.abs(out_np - ref_np)
    print(f"  {tag}: max_diff={diff.max():.6f}, mean_diff={diff.mean():.6f}")
    np.testing.assert_allclose(out_np, ref_np, rtol=RTOL, atol=ATOL)


# =====================================================================
# Batch GEMM (BF16)
# =====================================================================


def _run_batch_bf16(B, M, N, K):
    print(f"[BF16 batch] B={B}, M={M}, N={N}, K={K}")
    x = torch.randn([B, M, K], dtype=torch.bfloat16)
    w = torch.randn([B, N, K], dtype=torch.bfloat16)
    out = torch.zeros([B, M, N], dtype=torch.bfloat16)

    # Golden: per-batch fp32 matmul, output[b] = x[b] @ w[b]^T
    ref = torch.matmul(x.to(torch.float32), w.to(torch.float32).transpose(1, 2)).to(
        torch.bfloat16
    )

    batch_gemm(x, w, None, None, out, mode="bf16")

    for b in range(B):
        _assert_close(out[b], ref[b], f"batch {b}")
    print(f"  [PASS] BF16 batch B={B}, M={M}, N={N}, K={K}")


def test_batch_bf16_mid_m():
    # total_m = 512 -> 128x256x64 mid path
    _run_batch_bf16(B=4, M=128, N=256, K=128)


def test_batch_bf16_large_m():
    # total_m = 4096 -> 256x256x64 large path
    _run_batch_bf16(B=4, M=1024, N=256, K=128)


# =====================================================================
# Grouped masked GEMM (BF16)
# =====================================================================


def _run_masked_bf16(num_experts, max_m_per_expert, N, K, masked_m):
    print(
        f"[BF16 masked] experts={num_experts}, max_m={max_m_per_expert}, "
        f"N={N}, K={K}, masked_m={masked_m}"
    )
    total_m = num_experts * max_m_per_expert
    x = torch.randn([total_m, K], dtype=torch.bfloat16)
    w = torch.randn([num_experts, N, K], dtype=torch.bfloat16)
    out = torch.zeros([total_m, N], dtype=torch.bfloat16)
    masked_m_t = torch.tensor(masked_m, dtype=torch.int32)

    group_masked_gemm(
        x,
        w,
        None,
        None,
        masked_m_t,
        out,
        max_m_per_expert=max_m_per_expert,
        mode="bf16",
    )

    # Golden + comparison per expert, valid rows only.
    for e in range(num_experts):
        m = masked_m[e]
        if m == 0:
            continue
        start = e * max_m_per_expert
        x_e = x[start : start + m, :].to(torch.float32)
        w_e = w[e].to(torch.float32).T
        ref_e = torch.matmul(x_e, w_e).to(torch.bfloat16)
        _assert_close(out[start : start + m, :], ref_e, f"expert {e} (m={m})")
    print(f"  [PASS] BF16 masked experts={num_experts}, max_m={max_m_per_expert}")


def test_masked_bf16_mid_m():
    # max_m_per_expert = 128 -> 128x256x64 mid path; masked_m < max exercises masking
    _run_masked_bf16(
        num_experts=2, max_m_per_expert=128, N=256, K=128, masked_m=[100, 128]
    )


# =====================================================================
# Grouped contiguous GEMM (BF16)
# =====================================================================


def _run_contiguous_bf16(token_nums, N, K):
    num_experts = len(token_nums)
    m = int(sum(token_nums))
    print(f"[BF16 contiguous] m={m}, N={N}, K={K}, token_nums={token_nums}")
    x = torch.randn([m, K], dtype=torch.bfloat16)
    w = torch.randn([num_experts, N, K], dtype=torch.bfloat16)
    out = torch.zeros([m, N], dtype=torch.bfloat16)

    ids = np.concatenate(
        [np.full(c, e, dtype=np.int32) for e, c in enumerate(token_nums)]
    )
    expert_ids = torch.tensor(ids, dtype=torch.int32)

    group_contiguous_gemm(x, w, None, None, expert_ids, out, mode="bf16")

    offset = 0
    for e, c in enumerate(token_nums):
        x_e = x[offset : offset + c, :].to(torch.float32)
        w_e = w[e].to(torch.float32).T
        ref_e = torch.matmul(x_e, w_e).to(torch.bfloat16)
        _assert_close(out[offset : offset + c, :], ref_e, f"expert {e} (m={c})")
        offset += c
    print(f"  [PASS] BF16 contiguous m={m}, experts={num_experts}")


def test_contiguous_bf16_mid_m():
    # Each expert's token count aligned to tile_m=128; m=512 -> 128x256x64 path.
    _run_contiguous_bf16(token_nums=[256, 256], N=256, K=128)


if __name__ == "__main__":
    print("=" * 60)
    print("BF16 batch / grouped GEMM tests")
    print("=" * 60)
    test_batch_bf16_mid_m()
    test_batch_bf16_large_m()
    test_masked_bf16_mid_m()
    test_contiguous_bf16_mid_m()
    print("\n" + "=" * 60)
    print("All BF16 batch/grouped GEMM tests passed!")
    print("=" * 60)
