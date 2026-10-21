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

from halops import dense_gemm
import pytest
import torch
import numpy as np

from tests.utils.gemm_utils import (
    ue8m0_to_float32,
    per_token_cast_to_fp8,
    per_block_cast_to_fp8,
    per_token_cast_to_fp4,
    per_block_cast_to_fp4,
    reorder_sf_to_cutlass,
    _expect_runtime_error,
)

torch.manual_seed(2026)
torch.cuda.manual_seed_all(2026)
torch.set_default_device("cuda")


# =====================================================================
# Reference implementation
# =====================================================================


def ref_gemm(x, w, sfa, sfb, block_size=32):
    """
    Reference implementation of single-expert FP8 GEMM.
    x: (m, k) fp8, w: (n, k) fp8
    sfa: (m, k//block_size) ue8m0, sfb: (n, k//block_size) ue8m0
    """
    m, K = x.shape
    N = w.shape[0]  # noqa: F841
    k_blocks = K // block_size  # noqa: F841

    sfa_scale = ue8m0_to_float32(sfa)  # (m, k_blocks)
    sfb_scale = ue8m0_to_float32(sfb)  # (n, k_blocks)

    x_f32 = x.to(torch.float32)
    sfa_expanded = torch.repeat_interleave(sfa_scale, block_size, dim=1)
    x_deq = x_f32 * sfa_expanded

    w_f32 = w.to(torch.float32)
    sfb_expanded = torch.repeat_interleave(sfb_scale, block_size, dim=1)
    w_deq = w_f32 * sfb_expanded

    return torch.matmul(x_deq, w_deq.T).to(torch.bfloat16)


def ref_gemm_w4a4(x_fp4_float, w_fp4_float):
    """Reference implementation of single-expert W4A4 GEMM using dequantized float32 matmul."""
    return torch.matmul(x_fp4_float, w_fp4_float.T).to(torch.bfloat16)


# =====================================================================
# Kernel invocation
# =====================================================================


def gemm_wrapper(x, w, sfa, sfb, bias=None, alpha=None, sm_count=-1, mode="w8a8"):
    output = torch.zeros([x.shape[0], w.shape[0]], dtype=torch.bfloat16)
    dense_gemm(x, w, sfa, sfb, output, bias, alpha, sm_count, mode)
    return output


# =====================================================================
# W8A8 (FP8) tests
# =====================================================================


class TestGemmW8A8:
    """W8A8 single-expert GEMM tests."""

    def setup_method(self):
        self.m = 4096
        self.n = 7168
        self.k = 3584
        self.sm_count = -1

    def init_input(self):
        print(f"[init] m={self.m}, N={self.n}, K={self.k}")

        self.x_origin = torch.randn([self.m, self.k], dtype=torch.bfloat16)
        self.w_origin = torch.randn([self.n, self.k], dtype=torch.bfloat16)
        print(
            f"[init] x_origin shape={self.x_origin.shape}, w_origin shape={self.w_origin.shape}"
        )

        self.x, self.sfa_logical = per_token_cast_to_fp8(self.x_origin, 32)
        self.w, self.sfb_logical = per_block_cast_to_fp8(self.w_origin, 32)
        print(
            f"[init] x fp8 shape={self.x.shape}, sfa_logical shape={self.sfa_logical.shape}, "
            f"w fp8 shape={self.w.shape}, sfb_logical shape={self.sfb_logical.shape}"
        )

        self.sfa = reorder_sf_to_cutlass(
            self.sfa_logical, self.x.shape[0], self.x.shape[1] // 32
        )
        self.sfb = reorder_sf_to_cutlass(
            self.sfb_logical, self.w.shape[0], self.w.shape[1] // 32
        )

    def run(self):
        print("[test] computing reference output ...")
        ref_out = ref_gemm(self.x, self.w, self.sfa_logical, self.sfb_logical)
        print(
            f"[test] ref_out shape={ref_out.shape}, dtype={ref_out.dtype}, "
            f"min={ref_out.to(torch.float32).min().item():.4f}, max={ref_out.to(torch.float32).max().item():.4f}"
        )
        print("[test] computing kernel output ...")
        output = gemm_wrapper(
            self.x, self.w, self.sfa, self.sfb, sm_count=self.sm_count, mode="w8a8"
        )

        diff = (output.to(torch.float32) - ref_out.to(torch.float32)).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        print(f"[test] max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
        np.testing.assert_allclose(
            output.to(torch.float32).cpu().numpy(),
            ref_out.to(torch.float32).cpu().numpy(),
            rtol=2e-2,
            atol=2e-3,
        )
        print(f"  [PASS] W8A8 m={self.m}, n={self.n}, k={self.k}")


# =====================================================================
# W4A4 (NVFP4) tests
# =====================================================================

FP4_BLOCK_SIZE = 16


class TestGemmW4A4:
    """W4A4 single-expert GEMM tests."""

    def setup_method(self):
        self.m = 4096
        self.n = 7168
        self.k = 3584
        self.sm_count = -1

    def init_input(self):
        print(f"[init] m={self.m}, N={self.n}, K={self.k}")

        x_origin = torch.randn([self.m, self.k], dtype=torch.bfloat16)
        w_origin = torch.randn([self.n, self.k], dtype=torch.bfloat16)

        self.x_packed, self.sfa_logical, self.x_fp4_float = per_token_cast_to_fp4(
            x_origin, FP4_BLOCK_SIZE
        )
        self.w_packed, self.sfb_logical, self.w_fp4_float = per_block_cast_to_fp4(
            w_origin, FP4_BLOCK_SIZE
        )

        kb_dim = self.k // FP4_BLOCK_SIZE
        self.sfa = reorder_sf_to_cutlass(self.sfa_logical, self.m, kb_dim)
        self.sfb = reorder_sf_to_cutlass(self.sfb_logical, self.n, kb_dim)

        # Single expert: alpha is [1]
        self.alpha = torch.ones([1], dtype=torch.float32)

    def run(self):
        print("[test] computing reference output ...")
        ref_out = ref_gemm_w4a4(self.x_fp4_float, self.w_fp4_float)
        print(
            f"[test] ref_out shape={ref_out.shape}, dtype={ref_out.dtype}, "
            f"min={ref_out.to(torch.float32).min().item():.4f}, max={ref_out.to(torch.float32).max().item():.4f}"
        )
        print("[test] computing kernel output ...")
        for i in range(3):
            output = gemm_wrapper(
                self.x_packed,
                self.w_packed,
                self.sfa,
                self.sfb,
                alpha=self.alpha,
                sm_count=self.sm_count,
                mode="w4a4-nvfp4",
            )

        diff = (output.to(torch.float32) - ref_out.to(torch.float32)).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        print(f"[test] max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
        np.testing.assert_allclose(
            output.to(torch.float32).cpu().numpy(),
            ref_out.to(torch.float32).cpu().numpy(),
            rtol=5e-2,
            atol=5e-2,
        )
        print(f"  [PASS] W4A4 m={self.m}, n={self.n}, k={self.k}")


def test_gemm_w8a8_large_m():
    t = TestGemmW8A8()
    t.setup_method()
    t.init_input()
    t.run()


def test_gemm_w8a8_small_m():
    t = TestGemmW8A8()
    t.setup_method()
    t.m = 64
    t.init_input()
    t.run()


def test_gemm_w4a4_large_m():
    t = TestGemmW4A4()
    t.setup_method()
    t.init_input()
    t.run()


def test_gemm_additional_aligned_shapes():
    print("=" * 60)
    print("Additional GEMM aligned shape tests")
    print("=" * 60)

    test = TestGemmW8A8()
    test.setup_method()
    test.m = 128
    test.n = 256
    test.k = 256
    test.init_input()
    test.run()

    test = TestGemmW8A8()
    test.setup_method()
    test.m = 64
    test.n = 128
    test.k = 256
    test.init_input()
    test.run()

    t4 = TestGemmW4A4()
    t4.setup_method()
    t4.m = 128
    t4.n = 256
    t4.k = 512
    t4.init_input()
    t4.run()


def test_gemm_unsupported_shapes():
    print("=" * 60)
    print("GEMM unsupported shape checks")
    print("=" * 60)

    def run_w8a8_bad_k():
        m, n, k = 128, 256, 96
        x = torch.empty([m, k], dtype=torch.float8_e4m3fn)
        w = torch.empty([n, k], dtype=torch.float8_e4m3fn)
        sfa = torch.empty([m, k // 32], dtype=torch.uint8)
        sfb = torch.empty([n, k // 32], dtype=torch.uint8)
        out = torch.empty([m, n], dtype=torch.bfloat16)
        dense_gemm(x, w, sfa, sfb, out, None, None, -1, "w8a8")

    _expect_runtime_error(
        run_w8a8_bad_k,
        "No dense w8a8 tile for M=128 (grid M=128), N=256, K=96",
    )

    def run_w8a8_bad_n():
        m, n, k = 128, 255, 128
        x = torch.empty([m, k], dtype=torch.float8_e4m3fn)
        w = torch.empty([n, k], dtype=torch.float8_e4m3fn)
        sfa = torch.empty([m, k // 32], dtype=torch.uint8)
        sfb = torch.empty([n, k // 32], dtype=torch.uint8)
        out = torch.empty([m, n], dtype=torch.bfloat16)
        dense_gemm(x, w, sfa, sfb, out, None, None, -1, "w8a8")

    # Shape-aware dispatch includes N=128 tiles; N=255 fits none of them.
    _expect_runtime_error(
        run_w8a8_bad_n,
        "No dense w8a8 tile for M=128 (grid M=128), N=255, K=128",
    )

    def run_w4a4_bad_k():
        m, n, k = 128, 256, 48
        x = torch.empty([m, k // 2], dtype=torch.uint8)
        w = torch.empty([n, k // 2], dtype=torch.uint8)
        sfa = torch.empty([m, k // 16], dtype=torch.uint8)
        sfb = torch.empty([n, k // 16], dtype=torch.uint8)
        out = torch.empty([m, n], dtype=torch.bfloat16)
        alpha = torch.ones([1], dtype=torch.float32)
        dense_gemm(x, w, sfa, sfb, out, None, alpha, -1, "w4a4-nvfp4")

    _expect_runtime_error(run_w4a4_bad_k, "W4A4 GEMM K must be divisible by 64")

    def run_bad_rank_input():
        m, n, k = 128, 256, 128
        x = torch.empty([m, 1, k], dtype=torch.float8_e4m3fn)
        w = torch.empty([n, k], dtype=torch.float8_e4m3fn)
        sfa = torch.empty([m, k // 32], dtype=torch.uint8)
        sfb = torch.empty([n, k // 32], dtype=torch.uint8)
        out = torch.empty([m, n], dtype=torch.bfloat16)
        dense_gemm(x, w, sfa, sfb, out, None, None, -1, "w8a8")

    _expect_runtime_error(run_bad_rank_input, "input must be a 2D tensor")

    def run_bad_rank_output():
        m, n, k = 128, 256, 128
        x = torch.empty([m, k], dtype=torch.float8_e4m3fn)
        w = torch.empty([n, k], dtype=torch.float8_e4m3fn)
        sfa = torch.empty([m, k // 32], dtype=torch.uint8)
        sfb = torch.empty([n, k // 32], dtype=torch.uint8)
        out = torch.empty([m, n, 1], dtype=torch.bfloat16)
        dense_gemm(x, w, sfa, sfb, out, None, None, -1, "w8a8")

    _expect_runtime_error(run_bad_rank_output, "output must be a 2D tensor")


def test_gemm_w8a8_determinism():
    """Determinism: identical inputs must produce bitwise-identical outputs."""
    t = TestGemmW8A8()
    t.setup_method()
    t.m = 128
    t.n = 256
    t.k = 256
    t.init_input()
    out1 = gemm_wrapper(t.x, t.w, t.sfa, t.sfb, sm_count=t.sm_count, mode="w8a8")
    out2 = gemm_wrapper(t.x, t.w, t.sfa, t.sfb, sm_count=t.sm_count, mode="w8a8")
    assert torch.equal(out1, out2), "W8A8 dense_gemm output is not deterministic"


def test_gemm_w4a4_determinism():
    """Determinism: identical inputs must produce bitwise-identical outputs."""
    t = TestGemmW4A4()
    t.setup_method()
    t.m = 128
    t.n = 256
    t.k = 512
    t.init_input()
    out1 = gemm_wrapper(
        t.x_packed,
        t.w_packed,
        t.sfa,
        t.sfb,
        alpha=t.alpha,
        sm_count=t.sm_count,
        mode="w4a4-nvfp4",
    )
    out2 = gemm_wrapper(
        t.x_packed,
        t.w_packed,
        t.sfa,
        t.sfb,
        alpha=t.alpha,
        sm_count=t.sm_count,
        mode="w4a4-nvfp4",
    )
    assert torch.equal(out1, out2), "W4A4 dense_gemm output is not deterministic"


if __name__ == "__main__":
    pytest.main([__file__])
