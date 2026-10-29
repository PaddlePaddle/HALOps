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

import numpy as np
import torch

from halops import dense_gemm


torch.manual_seed(2026)
torch.cuda.manual_seed_all(2026)
torch.set_default_device("cuda")


def _run_bf16_gemm_case(m, n, k):
    print(f"[BF16 GEMM] m={m}, n={n}, k={k}")
    x = torch.randn([m, k], dtype=torch.bfloat16)
    w = torch.randn([n, k], dtype=torch.bfloat16)
    out = torch.empty([m, n], dtype=torch.bfloat16)

    ref = torch.matmul(x.to(torch.float32), w.to(torch.float32).T).to(torch.bfloat16)
    dense_gemm(x, w, None, None, out, mode="bf16")

    diff = (out.to(torch.float32) - ref.to(torch.float32)).abs()
    print(
        f"[BF16 GEMM] max_diff={diff.max().item():.6f}, "
        f"mean_diff={diff.mean().item():.6f}"
    )
    np.testing.assert_allclose(
        out.to(torch.float32).cpu().numpy(),
        ref.to(torch.float32).cpu().numpy(),
        rtol=2e-2,
        atol=2e-2,
    )


def _expect_runtime_error(fn, expected):
    try:
        fn()
    except RuntimeError as e:
        msg = str(e)
        assert expected in msg, f"expected '{expected}' in RuntimeError, got: {msg}"
        print(f"  [PASS] caught expected RuntimeError: {expected}")
    else:
        raise AssertionError(f"expected RuntimeError containing: {expected}")


def test_bf16_gemm_large_m():
    _run_bf16_gemm_case(4096, 256, 256)


def test_bf16_gemm_large_m_n1600():
    # Real production shape (MLA/DSA q_a_proj, bulk prefill): 1600 % 128 == 64,
    # so it fits neither the 256- nor the 128-wide full-tile large-M rung.
    # Covers 256x128x64_2CTA_Ntail end to end (predicated store at
    # KSubTileN=64 granularity, N % 64 == 0 launch check).
    _run_bf16_gemm_case(4096, 1600, 4096)


def test_bf16_gemm_large_m_n1600_ladder():
    # N=1600 tail (128-wide MMA, predicated store) across all production M
    # values in benchmark/configs/gemm/dense_bf16.yaml. M=1024/2048 are
    # mid-M and unaffected by the large-M Ntail change; included here to pin
    # the N=1600 correctness baseline across every M rung in one place.
    for m in (1024, 2048, 4096, 8192, 16384):
        _run_bf16_gemm_case(m, 1600, 4096)


def test_bf16_gemm_large_m_n128():
    # Real production shape (DSA indexer proj): n=128 leaves the KBlockM=256
    # family under half a wave (32 / 64 CTAs of 152), so large-M steps M down
    # instead of N -- covers the 64x64 and 64x128 rungs.
    _run_bf16_gemm_case(4096, 128, 4096)
    _run_bf16_gemm_case(8192, 128, 4096)


def test_bf16_gemm_mid_m():
    _run_bf16_gemm_case(128, 256, 256)


def test_bf16_gemm_small_m():
    # m < 128 -> 128x64x64 SwapAB small-M path.
    # SwapAB maps weight(N-dim) to tile_M=128 (N % 128) and activation(M-dim)
    # to tile_N=64 (M % 64), so use n=256 (multiple of 128) and m=64.
    _run_bf16_gemm_case(64, 256, 256)


def test_bf16_gemm_small_m_deep_k():
    # thin-M with k >= 8192 -> 128x64x128 SwapAB rung (KBlockK=128).
    _run_bf16_gemm_case(64, 4096, 8192)


def test_bf16_gemm_mid_m_odd_n():
    # N not a multiple of 128 -> mid-M 64x96 rung / predicated tail family.
    _run_bf16_gemm_case(1024, 1536, 4096)


def test_bf16_gemm_unsupported_shapes():
    def run_bad_k():
        m, n, k = 128, 256, 96
        x = torch.empty([m, k], dtype=torch.bfloat16)
        w = torch.empty([n, k], dtype=torch.bfloat16)
        out = torch.empty([m, n], dtype=torch.bfloat16)
        dense_gemm(x, w, None, None, out, mode="bf16")

    _expect_runtime_error(run_bad_k, "BF16 GEMM mid M K must be divisible by 64")

    def run_bad_dtype():
        m, n, k = 128, 256, 256
        x = torch.empty([m, k], dtype=torch.float16)
        w = torch.empty([n, k], dtype=torch.bfloat16)
        out = torch.empty([m, n], dtype=torch.bfloat16)
        dense_gemm(x, w, None, None, out, mode="bf16")

    _expect_runtime_error(run_bad_dtype, "input must be bfloat16")


if __name__ == "__main__":
    test_bf16_gemm_large_m()
    test_bf16_gemm_large_m_n1600()
    test_bf16_gemm_large_m_n1600_ladder()
    test_bf16_gemm_large_m_n128()
    test_bf16_gemm_mid_m()
    test_bf16_gemm_small_m()
    test_bf16_gemm_unsupported_shapes()
    print("All BF16 GEMM tests passed!")
