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

from halops import group_contiguous_gemm
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


def transpose_sf(sf):
    ndim = sf.ndim
    perm = list(range(ndim - 2)) + [ndim - 1, ndim - 2]
    return sf.permute(perm)


# =====================================================================
# Reference implementation
# =====================================================================


def ref_group_contiguous_gemm(
    x, w, sfa, sfb, expert_ids, num_experts, token_nums=None, bias=None, block_size=32
):
    """
    Reference implementation of FP8 grouped contiguous GEMM.
    Tokens of different experts are laid out contiguously in the input.
    If token_nums is provided, slice directly by offset (more efficient and
    avoids the row-by-row write-back issue).
    """
    K = x.shape[1]
    N = w.shape[1]
    k_blocks = K // block_size  # noqa: F841

    sfa_scale = ue8m0_to_float32(sfa)  # (M_total, k_blocks)
    sfb_scale = ue8m0_to_float32(sfb)  # (num_experts, N, k_blocks)

    ref_out = torch.zeros([x.shape[0], N], dtype=torch.bfloat16)

    if token_nums is not None:
        # Contiguous layout: slice directly by offset
        offset = 0
        for expert_id in range(num_experts):
            m = int(token_nums[expert_id])
            if m == 0:
                continue
            x_slice = x[offset : offset + m, :].to(torch.float32)
            sfa_slice = sfa_scale[offset : offset + m, :]
            sfa_expanded = torch.repeat_interleave(sfa_slice, block_size, dim=1)
            x_deq = x_slice * sfa_expanded

            w_slice = w[expert_id].to(torch.float32)
            sfb_slice = sfb_scale[expert_id]
            sfb_expanded = torch.repeat_interleave(sfb_slice, block_size, dim=1)
            w_deq = w_slice * sfb_expanded

            result = torch.matmul(x_deq, w_deq.T).to(torch.bfloat16)
            if bias is not None:
                result = result + bias[expert_id].unsqueeze(0)
            ref_out[offset : offset + m, :] = result
            offset += m
    else:
        for expert_id in range(num_experts):
            mask = expert_ids == expert_id
            indices = torch.nonzero(mask).flatten()
            if indices.shape[0] == 0:
                continue
            x_slice = torch.index_select(x, 0, indices).to(torch.float32)
            sfa_slice = torch.index_select(sfa_scale, 0, indices)
            sfa_expanded = torch.repeat_interleave(sfa_slice, block_size, dim=1)
            x_deq = x_slice * sfa_expanded

            w_slice = w[expert_id].to(torch.float32)
            sfb_slice = sfb_scale[expert_id]
            sfb_expanded = torch.repeat_interleave(sfb_slice, block_size, dim=1)
            w_deq = w_slice * sfb_expanded

            result = torch.matmul(x_deq, w_deq.T).to(torch.bfloat16)
            if bias is not None:
                result = result + bias[expert_id].unsqueeze(0)
            for i, idx in enumerate(indices.cpu().numpy()):
                ref_out[idx] = result[i]

    return ref_out


def ref_group_contiguous_gemm_w4a4(x_fp4_float, w_fp4_float, expert_ids, num_experts):
    """
    Reference implementation of W4A4 grouped contiguous GEMM.
    Uses dequantized float32 values for matmul directly.
    """
    N = w_fp4_float.shape[1]
    ref_out = torch.zeros([x_fp4_float.shape[0], N], dtype=torch.bfloat16)

    for expert_id in range(num_experts):
        mask = expert_ids == expert_id
        indices = torch.nonzero(mask).flatten()

        if indices.shape[0] == 0:
            continue

        x_slice = torch.index_select(x_fp4_float, 0, indices)
        w_slice = w_fp4_float[expert_id]

        result = torch.matmul(x_slice, w_slice.T).to(torch.bfloat16)

        for i, idx in enumerate(indices.cpu().numpy()):
            ref_out[idx] = result[i]

    return ref_out


# =====================================================================
# Kernel invocation
# =====================================================================


def group_contiguous_gemm_wrapper(
    x, w, sfa, sfb, expert_ids, bias=None, alpha=None, sm_count=-1, mode="w8a8"
):
    output = torch.zeros([x.shape[0], w.shape[1]], dtype=torch.bfloat16)
    group_contiguous_gemm(
        x, w, sfa, sfb, expert_ids, output, bias, alpha, sm_count, mode
    )
    return output


# =====================================================================
# W8A8 (FP8) tests
# =====================================================================


class TestGroupedGemmContiguous:
    """
    W8A8 grouped contiguous GEMM tests.

    Differences from the Masked version:
    - No padding: tokens of different experts are packed contiguously in the input
    - expert_ids [M] marks the expert each token belongs to
    - No masked_m or max_m_per_expert parameters

    Note: the Contiguous kernel's tile scheduler only reads the expert_ids of the
    first row of each tile to decide which expert's weight the tile uses. Therefore:
    1. Tokens of the same expert must be laid out contiguously in the input
    2. Each expert's token count must align to a multiple of tile_m
       (large-M config tile_m=256, mid-M config tile_m=128)
    """

    def setup_method(self):
        self.m = 4096
        self.n = 7168
        self.k = 3584
        self.num_experts = 10
        self.add_bias = False
        self.sm_count = -1
        self.mode = "w8a8"
        self.tile_m = 256  # must align with the kernel config's KBlockM

    def _make_contiguous_expert_ids(self, m, num_experts, tile_m):
        """
        Generate contiguous expert_ids: each expert occupies a contiguous span of
        rows, and each span length aligns to tile_m.
        Returns: expert_ids [m], int32; token_nums_per_expert [num_experts], int32
        """
        # Each expert gets at least tile_m rows (skip if insufficient)
        max_experts = m // tile_m
        actual_experts = min(num_experts, max_experts)
        # Even split: each expert gets m // actual_experts rows, aligned to tile_m
        per_expert = (m // actual_experts // tile_m) * tile_m
        assert (
            per_expert >= tile_m
        ), f"m={m} too small for {actual_experts} experts with tile_m={tile_m}"

        ids = []
        token_nums = []
        for e in range(actual_experts):
            count = (
                per_expert
                if e < actual_experts - 1
                else (m - per_expert * (actual_experts - 1))
            )
            # Ensure the last expert's token count is also aligned
            count = (count // tile_m) * tile_m
            if count <= 0:
                actual_experts = e
                break
            ids.append(np.full(count, e, dtype=np.int32))
            token_nums.append(count)

        expert_ids = torch.tensor(np.concatenate(ids), dtype=torch.int32)
        token_nums_per_expert = torch.tensor(token_nums, dtype=torch.int32)
        return expert_ids, token_nums_per_expert, actual_experts

    def init_input(self):
        print(
            f"[init] m={self.m}, N={self.n}, K={self.k}, num_experts={self.num_experts}, "
            f"add_bias={self.add_bias}, sm_count={self.sm_count}, tile_m={self.tile_m}"
        )

        # Generate contiguous expert_ids (aligned to tile boundaries)
        self.expert_ids, self.token_nums, self.actual_experts = (
            self._make_contiguous_expert_ids(self.m, self.num_experts, self.tile_m)
        )
        self.num_experts = self.actual_experts  # update to the actual expert count
        # Adjust m to the actually used row count
        self.m = self.expert_ids.shape[0]
        print(
            f"[init] actual m={self.m}, num_experts={self.num_experts}, "
            f"token_nums={self.token_nums.cpu().numpy()}"
        )
        print(
            f"[init] expert_ids distribution: {np.bincount(self.expert_ids.cpu().numpy(), minlength=self.num_experts)}"
        )

        self.x_origin = torch.randn([self.m, self.k], dtype=torch.bfloat16)
        self.w_origin = torch.randn(
            [self.num_experts, self.n, self.k], dtype=torch.bfloat16
        )
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
            self.sfb_logical, self.w.shape[1], self.w.shape[2] // 32
        )

        if self.add_bias:
            self.bias = torch.rand([self.num_experts, self.n], dtype=torch.bfloat16)
        else:
            self.bias = None

    def run(self):
        print("[test] computing reference output ...")
        ref_out = ref_group_contiguous_gemm(
            self.x,
            self.w,
            self.sfa_logical,
            self.sfb_logical,
            self.expert_ids,
            self.num_experts,
            token_nums=self.token_nums,
            bias=self.bias,
        )
        print(
            f"[test] ref_out shape={ref_out.shape}, dtype={ref_out.dtype}, "
            f"min={ref_out.to(torch.float32).min().item():.4f}, max={ref_out.to(torch.float32).max().item():.4f}"
        )
        print("[test] computing kernel output ...")
        output = group_contiguous_gemm_wrapper(
            self.x,
            self.w,
            self.sfa,
            self.sfb,
            self.expert_ids,
            bias=self.bias,
            sm_count=self.sm_count,
            mode=self.mode,
        )

        # Check accuracy per expert — expert_ids are contiguous, slice by offset
        offset = 0
        for expert_id in range(self.num_experts):
            m = int(self.token_nums[expert_id])
            if m == 0:
                print(f"[test] expert {expert_id}: no tokens, skipped")
                continue

            ref_exp = ref_out[offset : offset + m, :]
            out_exp = output[offset : offset + m, :]
            diff = (out_exp.to(torch.float32) - ref_exp.to(torch.float32)).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            print(
                f"[test] expert {expert_id}: m={m}, max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}"
            )
            np.testing.assert_allclose(
                out_exp.to(torch.float32).cpu().numpy(),
                ref_exp.to(torch.float32).cpu().numpy(),
                rtol=2e-2,
                atol=2e-3,
            )
            offset += m
        print(
            f"  [PASS] W8A8 m={self.m}, n={self.n}, k={self.k}, experts={self.num_experts}"
        )


# =====================================================================
# W4A4 (NVFP4) tests
# =====================================================================

FP4_BLOCK_SIZE = 16


class TestGroupedGemmContiguousW4A4:
    """
    W4A4 (NVFP4) grouped contiguous GEMM tests.
    """

    def setup_method(self):
        self.m = 4096
        self.n = 7168
        self.k = 3584
        self.num_experts = 20
        self.sm_count = -1
        self.mode = "w4a4-nvfp4"
        self.tile_m = 256

    def init_input(self):
        print(
            f"[init] m={self.m}, N={self.n}, K={self.k}, num_experts={self.num_experts}"
        )

        # Generate contiguous expert_ids (aligned to tile boundaries)
        tmp = TestGroupedGemmContiguous()
        tmp.setup_method()
        self.expert_ids, self.token_nums, self.actual_experts = (
            tmp._make_contiguous_expert_ids(self.m, self.num_experts, self.tile_m)
        )
        self.num_experts = self.actual_experts
        self.m = self.expert_ids.shape[0]
        print(
            f"[init] actual m={self.m}, num_experts={self.num_experts}, "
            f"token_nums={self.token_nums.cpu().numpy()}"
        )

        x_origin = torch.randn([self.m, self.k], dtype=torch.bfloat16)
        w_origin = torch.randn([self.num_experts, self.n, self.k], dtype=torch.bfloat16)

        self.x_packed, self.sfa_logical, self.x_fp4_float = per_token_cast_to_fp4(
            x_origin, FP4_BLOCK_SIZE
        )
        self.w_packed, self.sfb_logical, self.w_fp4_float = per_block_cast_to_fp4(
            w_origin, FP4_BLOCK_SIZE
        )

        kb_dim = self.k // FP4_BLOCK_SIZE
        self.sfa = reorder_sf_to_cutlass(self.sfa_logical, self.m, kb_dim)
        self.sfb = reorder_sf_to_cutlass(self.sfb_logical, self.n, kb_dim)

        # W4A4 requires per-expert alpha scaling factor [num_experts] (fp32)
        self.alpha = torch.ones([self.num_experts], dtype=torch.float32)

    def run(self):
        ref_out = ref_group_contiguous_gemm_w4a4(
            self.x_fp4_float, self.w_fp4_float, self.expert_ids, self.num_experts
        )

        for i in range(3):
            output = group_contiguous_gemm_wrapper(
                self.x_packed,
                self.w_packed,
                self.sfa,
                self.sfb,
                self.expert_ids,
                alpha=self.alpha,
                sm_count=self.sm_count,
                mode=self.mode,
            )

        offset = 0
        for expert_id in range(self.num_experts):
            m = int(self.token_nums[expert_id])
            if m == 0:
                print(f"[test] expert {expert_id}: no tokens, skipped")
                continue

            ref_exp = ref_out[offset : offset + m, :]
            out_exp = output[offset : offset + m, :]

            np.testing.assert_allclose(
                out_exp.to(torch.float32).cpu().numpy(),
                ref_exp.to(torch.float32).cpu().numpy(),
                rtol=5e-2,
                atol=5e-2,
            )
            offset += m
        print(
            f"  [PASS] W4A4 m={self.m}, n={self.n}, k={self.k}, experts={self.num_experts}"
        )


def test_group_contiguous_w8a8():
    t = TestGroupedGemmContiguous()
    t.setup_method()
    t.m = 512
    t.num_experts = 2
    t.init_input()
    t.run()


def test_group_contiguous_w4a4():
    t = TestGroupedGemmContiguousW4A4()
    t.setup_method()
    t.m = 512
    t.num_experts = 2
    t.init_input()
    t.run()


def test_group_contiguous_w8a8_large_m():
    """W8A8 large-M path: m=4096 across many experts."""
    t = TestGroupedGemmContiguous()
    t.setup_method()
    t.m = 4096
    t.num_experts = 20
    t.init_input()
    t.run()


def test_group_contiguous_w8a8_small_m_swapab():
    """W8A8 small-M SwapAB path: m=64 with tile_m=64."""
    t = TestGroupedGemmContiguous()
    t.setup_method()
    t.m = 64
    t.num_experts = 20
    t.tile_m = 64
    t.init_input()
    t.run()


def test_group_contiguous_w4a4_large_m():
    """W4A4 large-M path: m=4096 across many experts."""
    t = TestGroupedGemmContiguousW4A4()
    t.setup_method()
    t.m = 4096
    t.num_experts = 20
    t.init_input()
    t.run()


def test_group_contiguous_additional_shapes():
    print("=" * 60)
    print("Additional GroupContiguous GEMM shape tests")
    print("=" * 60)

    test = TestGroupedGemmContiguous()
    test.setup_method()
    test.m = 128
    test.n = 256
    test.k = 256
    test.num_experts = 1
    test.tile_m = 128
    test.init_input()
    test.run()

    t4 = TestGroupedGemmContiguousW4A4()
    t4.setup_method()
    t4.m = 128
    t4.n = 256
    t4.k = 512
    t4.num_experts = 1
    t4.tile_m = 128
    t4.init_input()
    t4.run()


def test_group_contiguous_unsupported_shapes():
    print("=" * 60)
    print("GroupContiguous GEMM unsupported shape checks")
    print("=" * 60)

    def run_w8a8_bad_m():
        m, n, k, num_experts = 129, 256, 128, 1
        x = torch.empty([m, k], dtype=torch.float8_e4m3fn)
        w = torch.empty([num_experts, n, k], dtype=torch.float8_e4m3fn)
        sfa = torch.empty([m, k // 32], dtype=torch.uint8)
        sfb = torch.empty([num_experts, n, k // 32], dtype=torch.uint8)
        expert_ids = torch.zeros([m], dtype=torch.int32)
        out = torch.empty([m, n], dtype=torch.bfloat16)
        group_contiguous_gemm(x, w, sfa, sfb, expert_ids, out, None, None, -1, "w8a8")

    _expect_runtime_error(run_w8a8_bad_m, "requires M to be divisible by 128")

    def run_w4a4_bias():
        m, n, k, num_experts = 128, 256, 512, 1
        x = torch.empty([m, k // 2], dtype=torch.uint8)
        w = torch.empty([num_experts, n, k // 2], dtype=torch.uint8)
        sfa = torch.empty([m, k // 16], dtype=torch.uint8)
        sfb = torch.empty([num_experts, n, k // 16], dtype=torch.uint8)
        expert_ids = torch.zeros([m], dtype=torch.int32)
        out = torch.empty([m, n], dtype=torch.bfloat16)
        bias = torch.empty([num_experts, n], dtype=torch.bfloat16)
        alpha = torch.ones([num_experts], dtype=torch.float32)
        group_contiguous_gemm(x, w, sfa, sfb, expert_ids, out, bias, alpha, -1, "w4a4-nvfp4")

    _expect_runtime_error(run_w4a4_bias, "W4A4 mode does not support bias yet")

    def run_expert_ids_int64():
        m, n, k, num_experts = 128, 256, 128, 1
        x = torch.empty([m, k], dtype=torch.float8_e4m3fn)
        w = torch.empty([num_experts, n, k], dtype=torch.float8_e4m3fn)
        sfa = torch.empty([m, k // 32], dtype=torch.uint8)
        sfb = torch.empty([num_experts, n, k // 32], dtype=torch.uint8)
        expert_ids = torch.zeros([m], dtype=torch.int64)
        out = torch.empty([m, n], dtype=torch.bfloat16)
        group_contiguous_gemm(x, w, sfa, sfb, expert_ids, out, None, None, -1, "w8a8")

    _expect_runtime_error(
        run_expert_ids_int64, "expert_ids must have dtype torch.int32"
    )

    def run_bad_rank_output():
        m, n, k, num_experts = 128, 256, 128, 1
        x = torch.empty([m, k], dtype=torch.float8_e4m3fn)
        w = torch.empty([num_experts, n, k], dtype=torch.float8_e4m3fn)
        sfa = torch.empty([m, k // 32], dtype=torch.uint8)
        sfb = torch.empty([num_experts, n, k // 32], dtype=torch.uint8)
        expert_ids = torch.zeros([m], dtype=torch.int32)
        out = torch.empty([m, n, 1], dtype=torch.bfloat16)
        group_contiguous_gemm(x, w, sfa, sfb, expert_ids, out, None, None, -1, "w8a8")

    _expect_runtime_error(run_bad_rank_output, "output must be a 2D tensor")


def test_group_contiguous_w8a8_determinism():
    """Determinism: identical inputs must produce bitwise-identical outputs."""
    t = TestGroupedGemmContiguous()
    t.setup_method()
    t.m = 512
    t.num_experts = 2
    t.init_input()
    out1 = group_contiguous_gemm_wrapper(
        t.x,
        t.w,
        t.sfa,
        t.sfb,
        t.expert_ids,
        bias=t.bias,
        sm_count=t.sm_count,
        mode=t.mode,
    )
    out2 = group_contiguous_gemm_wrapper(
        t.x,
        t.w,
        t.sfa,
        t.sfb,
        t.expert_ids,
        bias=t.bias,
        sm_count=t.sm_count,
        mode=t.mode,
    )
    assert torch.equal(
        out1, out2
    ), "W8A8 group_contiguous_gemm output is not deterministic"


def test_group_contiguous_w4a4_determinism():
    """Determinism: identical inputs must produce bitwise-identical outputs."""
    t = TestGroupedGemmContiguousW4A4()
    t.setup_method()
    t.m = 512
    t.num_experts = 2
    t.init_input()
    out1 = group_contiguous_gemm_wrapper(
        t.x_packed,
        t.w_packed,
        t.sfa,
        t.sfb,
        t.expert_ids,
        alpha=t.alpha,
        sm_count=t.sm_count,
        mode=t.mode,
    )
    out2 = group_contiguous_gemm_wrapper(
        t.x_packed,
        t.w_packed,
        t.sfa,
        t.sfb,
        t.expert_ids,
        alpha=t.alpha,
        sm_count=t.sm_count,
        mode=t.mode,
    )
    assert torch.equal(
        out1, out2
    ), "W4A4 group_contiguous_gemm output is not deterministic"


if __name__ == "__main__":
    pytest.main([__file__])
