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

from halops.gemm.batch_gemm import batch_gemm
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


def ref_batch_gemm_fp8(
    input_fp8, weight_fp8, sfa_logical, sfb_logical, bias=None, block_size=32
):
    """
    FP8 Batch GEMM reference: output[b] = dequant(input[b]) @ dequant(weight[b])^T + bias
    input_fp8: [B, M, K] float8_e4m3fn
    weight_fp8: [B, N, K] float8_e4m3fn
    sfa_logical: [B*M, K//block_size] uint8
    sfb_logical: [B, N, K//block_size] uint8 (repeated along N)
    """
    B, M, K = input_fp8.shape
    N = weight_fp8.shape[1]

    sfa_scale = ue8m0_to_float32(sfa_logical)  # [B*M, K//bs]
    sfb_scale = ue8m0_to_float32(sfb_logical)  # [B, N, K//bs]

    output = torch.zeros([B, M, N], dtype=torch.bfloat16)

    for b in range(B):
        # Dequantize input[b]
        x_slice = input_fp8[b].to(torch.float32)  # [M, K]
        sfa_slice = sfa_scale[b * M : (b + 1) * M, :]  # [M, K//bs]
        sfa_expanded = torch.repeat_interleave(sfa_slice, block_size, dim=1)
        x_deq = x_slice * sfa_expanded

        # Dequantize weight[b]
        w_slice = weight_fp8[b].to(torch.float32)  # [N, K]
        sfb_slice = sfb_scale[b]  # [N, K//bs]
        sfb_expanded = torch.repeat_interleave(sfb_slice, block_size, dim=1)
        w_deq = w_slice * sfb_expanded

        result = torch.matmul(x_deq, w_deq.T).to(torch.bfloat16)
        if bias is not None:
            result = result + bias[b].unsqueeze(0)
        output[b] = result

    return output


def ref_batch_gemm_fp4(x_fp4_float, w_fp4_float):
    """
    FP4 Batch GEMM reference: output[b] = x_fp4_float[b] @ w_fp4_float[b]^T
    x_fp4_float: [B, M, K] float32 (already dequantized)
    w_fp4_float: [B, N, K] float32 (already dequantized)
    """
    B = x_fp4_float.shape[0]
    M = x_fp4_float.shape[1]
    N = w_fp4_float.shape[1]
    output = torch.zeros([B, M, N], dtype=torch.bfloat16)
    for b in range(B):
        output[b] = torch.matmul(x_fp4_float[b], w_fp4_float[b].T).to(torch.bfloat16)
    return output


# =====================================================================
# Test class
# =====================================================================


class TestBatchGemmW8A8:
    def setup(self, B=4, M=256, N=256, K=128, add_bias=False):
        self.B = B
        self.M = M
        self.N = N
        self.K = K
        self.add_bias = add_bias
        self.block_size = 32

        # Generate random BF16 tensors
        x_origin = torch.randn([B, M, K], dtype=torch.bfloat16)
        w_origin = torch.randn([B, N, K], dtype=torch.bfloat16)

        # Quantize input: reshape to [B*M, K] for per_token_cast_to_fp8
        x_flat = x_origin.reshape(B * M, K)
        x_fp8_flat, sfa_logical_flat = per_token_cast_to_fp8(x_flat, self.block_size)
        self.input_fp8 = x_fp8_flat.reshape(B, M, K)
        self.sfa_logical = sfa_logical_flat  # [B*M, K//32]

        # Quantize weight: [B, N, K]
        self.weight_fp8, self.sfb_logical = per_block_cast_to_fp8(
            w_origin, self.block_size
        )

        # Reorder scale factors to SfAtom layout
        kb_a = K // self.block_size
        kb_b = K // self.block_size
        self.sfa = reorder_sf_to_cutlass(self.sfa_logical, B * M, kb_a)
        self.sfb = reorder_sf_to_cutlass(self.sfb_logical, N, kb_b)

        if add_bias:
            self.bias = torch.rand([B, N], dtype=torch.bfloat16)
        else:
            self.bias = None

    def run(self):
        print(
            f"[test] B={self.B}, M={self.M}, N={self.N}, K={self.K}, bias={self.add_bias}"
        )

        # Reference
        ref_out = ref_batch_gemm_fp8(
            self.input_fp8,
            self.weight_fp8,
            self.sfa_logical,
            self.sfb_logical,
            bias=self.bias,
            block_size=self.block_size,
        )

        # Kernel output
        output = torch.zeros(
            [self.B, self.M, self.N], dtype=torch.bfloat16, device="cuda"
        )
        batch_gemm(
            self.input_fp8,
            self.weight_fp8,
            self.sfa,
            self.sfb,
            output,
            bias=self.bias,
            sm_count=-1,
            mode="w8a8",
        )

        # Compare per batch
        for b in range(self.B):
            ref_b = ref_out[b].to(torch.float32).cpu().numpy()
            out_b = output[b].to(torch.float32).cpu().numpy()
            diff = np.abs(out_b - ref_b)
            max_diff = diff.max()
            mean_diff = diff.mean()
            print(f"  batch {b}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
            np.testing.assert_allclose(out_b, ref_b, rtol=2e-2, atol=2e-3)

        print(
            f"  [PASS] W8A8 Batch GEMM B={self.B}, M={self.M}, N={self.N}, K={self.K}"
        )


class TestBatchGemmW4A4:
    def setup(self, B=4, M=256, N=256, K=256):
        self.B = B
        self.M = M
        self.N = N
        self.K = K
        self.block_size = 16

        # Generate random BF16 tensors
        x_origin = torch.randn([B, M, K], dtype=torch.bfloat16)
        w_origin = torch.randn([B, N, K], dtype=torch.bfloat16)

        # Quantize input per-token: reshape to [B*M, K]
        x_flat = x_origin.reshape(B * M, K)
        x_packed_flat, sfa_logical_flat, x_deq_flat = per_token_cast_to_fp4(
            x_flat, self.block_size
        )
        self.input_packed = x_packed_flat.reshape(B, M, K // 2)
        self.sfa_logical = sfa_logical_flat  # [B*M, K//16]
        self.x_fp4_float = x_deq_flat.reshape(B, M, K)

        # Quantize weight per-block, one batch at a time
        w_packed_list = []
        sfb_logical_list = []
        w_deq_list = []
        for b in range(B):
            w_packed_b, sfb_logical_b, w_deq_b = per_block_cast_to_fp4(
                w_origin[b], self.block_size
            )
            w_packed_list.append(w_packed_b)
            sfb_logical_list.append(sfb_logical_b)
            w_deq_list.append(w_deq_b)
        self.weight_packed = torch.stack(w_packed_list, dim=0)  # [B, N, K//2]
        self.sfb_logical = torch.stack(sfb_logical_list, dim=0)  # [B, N, K//16]
        self.w_fp4_float = torch.stack(w_deq_list, dim=0)  # [B, N, K]

        # Reorder scale factors to SfAtom layout
        kb = K // self.block_size
        self.sfa = reorder_sf_to_cutlass(self.sfa_logical, B * M, kb)
        self.sfb = reorder_sf_to_cutlass(self.sfb_logical, N, kb)

        # W4A4 requires a per-expert (per-batch) scaling factor with shape [B]
        self.alpha = torch.ones([B], dtype=torch.float32)

    def run(self):
        print(f"[test] W4A4 B={self.B}, M={self.M}, N={self.N}, K={self.K}")

        # Reference
        ref_out = ref_batch_gemm_fp4(self.x_fp4_float, self.w_fp4_float)

        # Kernel output
        output = torch.zeros(
            [self.B, self.M, self.N], dtype=torch.bfloat16, device="cuda"
        )
        batch_gemm(
            self.input_packed,
            self.weight_packed,
            self.sfa,
            self.sfb,
            output,
            alpha=self.alpha,
            sm_count=-1,
            mode="w4a4-nvfp4",
        )

        # Compare per batch
        for b in range(self.B):
            ref_b = ref_out[b].to(torch.float32).cpu().numpy()
            out_b = output[b].to(torch.float32).cpu().numpy()
            np.testing.assert_allclose(out_b, ref_b, rtol=5e-2, atol=5e-2)

        print(
            f"  [PASS] W4A4 Batch GEMM B={self.B}, M={self.M}, N={self.N}, K={self.K}"
        )


# =====================================================================
# Main test entry
# =====================================================================


def test_batch_gemm_w8a8_mid_m():
    """Test mid-M path: total_m >= 128, uses 128x256x128 tile."""
    t = TestBatchGemmW8A8()
    t.setup(B=4, M=128, N=256, K=128)
    t.run()


def test_batch_gemm_w8a8_large_m():
    """Test large-M path: total_m >= 4096, uses 256x256x128 tile."""
    t = TestBatchGemmW8A8()
    t.setup(B=4, M=1024, N=256, K=128)
    t.run()


def test_batch_gemm_w8a8_with_bias():
    """Test W8A8 with bias path: uses 256x128x128 tile."""
    t = TestBatchGemmW8A8()
    t.setup(B=2, M=256, N=128, K=128, add_bias=True)
    t.run()


# ---- W8A8 boundary condition tests ----


def test_batch_gemm_w8a8_small_m_swapab():
    """Boundary: total_m < 128 triggers the SwapAB path (needs N%128, M%64, K%256)."""
    t = TestBatchGemmW8A8()
    t.setup(B=1, M=64, N=128, K=256)
    t.run()


def test_batch_gemm_w8a8_single_batch():
    """Boundary: single batch (B=1) mid-M path."""
    t = TestBatchGemmW8A8()
    t.setup(B=1, M=256, N=256, K=128)
    t.run()


def test_batch_gemm_w8a8_larger_k():
    """Boundary: larger K dimension."""
    t = TestBatchGemmW8A8()
    t.setup(B=2, M=128, N=256, K=512)
    t.run()


# ---- W8A8 error input tests ----


def test_batch_gemm_w8a8_errors():
    """Error inputs: invalid mode, wrong rank, mismatched shapes, bad divisibility."""
    B, M, N, K = 2, 128, 256, 128

    def make_w8a8():
        x = torch.empty([B, M, K], dtype=torch.float8_e4m3fn)
        w = torch.empty([B, N, K], dtype=torch.float8_e4m3fn)
        sfa = torch.empty([B * M, K // 32], dtype=torch.uint8)
        sfb = torch.empty([B, N, K // 32], dtype=torch.uint8)
        out = torch.empty([B, M, N], dtype=torch.bfloat16)
        return x, w, sfa, sfb, out

    def run_bad_mode():
        x, w, sfa, sfb, out = make_w8a8()
        batch_gemm(x, w, sfa, sfb, out, None, None, -1, "w2a2")

    _expect_runtime_error(run_bad_mode, "Unsupported mode")

    def run_bad_input_rank():
        _, w, sfa, sfb, out = make_w8a8()
        x = torch.empty([B * M, K], dtype=torch.float8_e4m3fn)
        batch_gemm(x, w, sfa, sfb, out, None, None, -1, "w8a8")

    _expect_runtime_error(run_bad_input_rank, "input must be a 3D tensor")

    def run_bad_weight_batch():
        x, _, sfa, sfb, out = make_w8a8()
        w = torch.empty([B + 1, N, K], dtype=torch.float8_e4m3fn)
        batch_gemm(x, w, sfa, sfb, out, None, None, -1, "w8a8")

    _expect_runtime_error(run_bad_weight_batch, "weight batch dim must match")

    def run_bad_k():
        bad_k = 96  # not divisible by 128 (mid-M tile K)
        x = torch.empty([B, M, bad_k], dtype=torch.float8_e4m3fn)
        w = torch.empty([B, N, bad_k], dtype=torch.float8_e4m3fn)
        sfa = torch.empty([B * M, bad_k // 32], dtype=torch.uint8)
        sfb = torch.empty([B, N, bad_k // 32], dtype=torch.uint8)
        out = torch.empty([B, M, N], dtype=torch.bfloat16)
        batch_gemm(x, w, sfa, sfb, out, None, None, -1, "w8a8")

    _expect_runtime_error(
        run_bad_k,
        "No batch w8a8 tile for M=128 (grid M=256), N=256, K=96",
    )


# ---- W8A8 determinism test ----


def test_batch_gemm_w8a8_determinism():
    """Determinism: identical inputs must produce bitwise-identical outputs."""
    t = TestBatchGemmW8A8()
    t.setup(B=4, M=128, N=256, K=128)
    out1 = torch.zeros([t.B, t.M, t.N], dtype=torch.bfloat16, device="cuda")
    out2 = torch.zeros([t.B, t.M, t.N], dtype=torch.bfloat16, device="cuda")
    batch_gemm(t.input_fp8, t.weight_fp8, t.sfa, t.sfb, out1, None, None, -1, "w8a8")
    batch_gemm(t.input_fp8, t.weight_fp8, t.sfa, t.sfb, out2, None, None, -1, "w8a8")
    assert torch.equal(out1, out2), "W8A8 batch_gemm output is not deterministic"


# ---- W4A4 correctness tests ----


def test_batch_gemm_w4a4_mid_m():
    """W4A4 mid-M path: total_m >= 128, uses 128x256x256 tile."""
    t = TestBatchGemmW4A4()
    t.setup(B=4, M=128, N=256, K=256)
    t.run()


def test_batch_gemm_w4a4_large_m():
    """W4A4 large-M path: total_m >= 4096, uses 256x256x256 tile."""
    t = TestBatchGemmW4A4()
    t.setup(B=4, M=1024, N=256, K=256)
    t.run()


# ---- W4A4 boundary condition tests ----


def test_batch_gemm_w4a4_small_m_swapab():
    """W4A4 boundary: total_m < 128 triggers SwapAB path (needs N%128, M%64, K%512)."""
    t = TestBatchGemmW4A4()
    t.setup(B=1, M=64, N=128, K=512)
    t.run()


def test_batch_gemm_w4a4_single_batch():
    """W4A4 boundary: single batch (B=1) mid-M path."""
    t = TestBatchGemmW4A4()
    t.setup(B=1, M=256, N=256, K=256)
    t.run()


# ---- W4A4 error input tests ----


def test_batch_gemm_w4a4_errors():
    """Error inputs: missing alpha, mismatched packed K, bad divisibility, bias unsupported."""
    B, M, N, K = 2, 128, 256, 256

    def make_w4a4():
        x = torch.empty([B, M, K // 2], dtype=torch.uint8)
        w = torch.empty([B, N, K // 2], dtype=torch.uint8)
        sfa = torch.empty([B * M, K // 16], dtype=torch.uint8)
        sfb = torch.empty([B, N, K // 16], dtype=torch.uint8)
        out = torch.empty([B, M, N], dtype=torch.bfloat16)
        alpha = torch.ones([1], dtype=torch.float32)
        return x, w, sfa, sfb, out, alpha

    def run_missing_alpha():
        x, w, sfa, sfb, out, _ = make_w4a4()
        batch_gemm(x, w, sfa, sfb, out, None, None, -1, "w4a4-nvfp4")

    _expect_runtime_error(run_missing_alpha, "W4A4 mode requires alpha")

    def run_bad_k():
        bad_k = 32  # k=32 not divisible by 64
        x = torch.empty([B, M, bad_k // 2], dtype=torch.uint8)
        w = torch.empty([B, N, bad_k // 2], dtype=torch.uint8)
        sfa = torch.empty([B * M, bad_k // 16], dtype=torch.uint8)
        sfb = torch.empty([B, N, bad_k // 16], dtype=torch.uint8)
        out = torch.empty([B, M, N], dtype=torch.bfloat16)
        alpha = torch.ones([1], dtype=torch.float32)
        batch_gemm(x, w, sfa, sfb, out, None, alpha, -1, "w4a4-nvfp4")

    _expect_runtime_error(run_bad_k, "W4A4 batch GEMM K must be divisible by 64")

    def run_mismatched_packed_k():
        x = torch.empty([B, M, K // 2], dtype=torch.uint8)
        w = torch.empty([B, N, K // 2 + 1], dtype=torch.uint8)
        sfa = torch.empty([B * M, K // 16], dtype=torch.uint8)
        sfb = torch.empty([B, N, K // 16], dtype=torch.uint8)
        out = torch.empty([B, M, N], dtype=torch.bfloat16)
        alpha = torch.ones([1], dtype=torch.float32)
        batch_gemm(x, w, sfa, sfb, out, None, alpha, -1, "w4a4-nvfp4")

    _expect_runtime_error(run_mismatched_packed_k, "packed input K/2 must match")

    def run_bias_unsupported():
        x, w, sfa, sfb, out, alpha = make_w4a4()
        bias = torch.zeros([B, N], dtype=torch.bfloat16)
        batch_gemm(x, w, sfa, sfb, out, bias, alpha, -1, "w4a4-nvfp4")

    _expect_runtime_error(run_bias_unsupported, "W4A4 batch GEMM does not support bias")


# ---- W4A4 determinism test ----


def test_batch_gemm_w4a4_determinism():
    """Determinism: identical inputs must produce bitwise-identical outputs."""
    t = TestBatchGemmW4A4()
    t.setup(B=4, M=128, N=256, K=256)
    out1 = torch.zeros([t.B, t.M, t.N], dtype=torch.bfloat16, device="cuda")
    out2 = torch.zeros([t.B, t.M, t.N], dtype=torch.bfloat16, device="cuda")
    batch_gemm(
        t.input_packed, t.weight_packed, t.sfa, t.sfb, out1, None, t.alpha, -1, "w4a4-nvfp4"
    )
    batch_gemm(
        t.input_packed, t.weight_packed, t.sfa, t.sfb, out2, None, t.alpha, -1, "w4a4-nvfp4"
    )
    assert torch.equal(out1, out2), "W4A4 batch_gemm output is not deterministic"


if __name__ == "__main__":
    pytest.main([__file__])
