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

from halops import group_masked_gemm
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
# Scale padding: pad every 224 rows to 256 rows (only pad scale, not weight)
# =====================================================================


def pad_sfb_scale_224_to_256(sfb_logical, n_orig):
    """
    Pad sfb scale along N dimension from every 224 rows to 256 rows.
    Only pads scale, not the weight matrix.

    Example: N=7168, 7168/224=32 groups, each 224->256, padded N_padded=32*256=8192.
    Padded rows are filled with 0 (UE8M0/UE4M3 code=0 maps to tiny value, does not affect computation).

    Parameters:
      sfb_logical: (..., N, kb_dim), dtype=uint8
      n_orig: original N dimension size, must be divisible by 224
    Returns:
      padded sfb: (..., N_padded, kb_dim), dtype=uint8, torch.Tensor
      n_padded: N dimension size after padding
    """
    assert n_orig % 224 == 0, f"N({n_orig}) must be divisible by 224"
    sf_np = sfb_logical.cpu().numpy()
    orig_shape = sf_np.shape
    kb_dim = orig_shape[-1]
    num_groups = n_orig // 224
    n_padded = num_groups * 256

    # Reshape N dimension into (num_groups, 224)
    leading_shape = orig_shape[:-2]
    sf_grouped = sf_np.reshape(*leading_shape, num_groups, 224, kb_dim)

    # Pad each group from 224 to 256: append 32 zero rows after the 224 dimension
    pad_width = [(0, 0)] * len(leading_shape) + [(0, 0), (0, 32), (0, 0)]
    sf_padded = np.pad(sf_grouped, pad_width, mode="constant", constant_values=0)

    # Reshape back to (..., N_padded, kb_dim)
    new_shape = list(leading_shape) + [n_padded, kb_dim]
    sf_padded = sf_padded.reshape(new_shape)

    return torch.tensor(sf_padded, dtype=torch.uint8), n_padded


def transpose_sf(sf):
    """
    Transpose the last two dimensions of scale factor, dtype unchanged (uint8).

    Example: (10, 7168, 112) -> transpose -> (10, 112, 7168)
        (M, kb_dim) -> transpose -> (kb_dim, M)
    """
    ndim = sf.ndim
    perm = list(range(ndim - 2)) + [ndim - 1, ndim - 2]
    return sf.permute(perm)


def pack_sf_mn_major_u32(sf, mn_dim, k_sf_byte):
    """
    Pack uint8 SF data into MN-major uint32_t byte layout, for the !IsPrmtScale path.

    The kernel reinterprets the SF pointer as uint32_t, GMEM layout is (MN, K_sf_u32, L) stride (1, MN, MN*K_sf_u32).
    Each uint32_t contains 4 consecutive ue8m0/ue4m3 K values.
    MN dimension is stride-1 (contiguous), i.e., column-major form.

    Parameters:
      sf: (..., MN, K_sf_byte) uint8 torch tensor
      mn_dim: MN dimension size
      k_sf_byte: Number of K-dimension SF elements (uint8 count)
    Returns:
      Packed contiguous uint8 tensor, byte layout matches kernel-expected MN-major uint32_t format
    """
    sf_np = sf.cpu().numpy()
    # Separate leading dims (batch/expert) from (MN, K_sf_byte)
    sf_np = sf_np.reshape(-1, mn_dim, k_sf_byte)
    L = sf_np.shape[0]
    k_sf_u32 = k_sf_byte // 4

    # (L, MN, K_sf_byte) -> (L, MN, K_sf_u32, 4): group 4 consecutive K values
    sf_np = sf_np.reshape(L, mn_dim, k_sf_u32, 4)
    # (L, MN, K_sf_u32, 4) -> (L, K_sf_u32, MN, 4): MN-major (column-major per uint32_t group)
    sf_np = sf_np.transpose(0, 2, 1, 3)
    sf_np = np.ascontiguousarray(sf_np)

    return torch.tensor(sf_np, dtype=torch.uint8)


# =====================================================================
# Reference implementation
# =====================================================================


def ref_group_masked_gemm_bf16(x, w, masked_m, max_m_per_expert, num_experts):
    x_bf16 = x.to(torch.bfloat16)
    w_bf16 = w.to(torch.bfloat16)
    ref_out = torch.zeros(
        [x.shape[0], w.shape[1]], dtype=torch.bfloat16, device=x.device
    )
    for expert_id in range(num_experts):
        start_offset = expert_id * max_m_per_expert
        x_tmp = x_bf16[start_offset : start_offset + masked_m[expert_id], :]
        w_tmp = w_bf16[expert_id].T

        ref_out[start_offset : start_offset + masked_m[expert_id], :] = torch.matmul(
            x_tmp, w_tmp
        )
    return ref_out


def ref_group_masked_gemm(
    x, w, sfa, sfb, masked_m, max_m_per_expert, num_experts, bias=None, block_size=32
):
    """
    Reference implementation of FP8 grouped masked GEMM.
    Dequantizes x and w using their respective UE8M0 scales, then performs matrix multiplication.

    Parameters:
      x:   (num_experts * max_m_per_expert, K), dtype=float8_e4m3fn
      w:   (num_experts, N, K), dtype=float8_e4m3fn
      sfa: (num_experts * max_m_per_expert, K // block_size), dtype=uint8 (UE8M0)
           from per_token_cast_to_fp8, k-dimension block-wise scale
      sfb: (num_experts, N, K // block_size), dtype=uint8 (UE8M0)
           from per_block_cast_to_fp8, n x k 2D block-wise scale (n dim already repeated)
      masked_m: (num_experts,), valid row count per expert
      max_m_per_expert: max row count per expert
      num_experts: number of experts
      bias: (num_experts, N), dtype=bfloat16, optional. If not None, added to matmul result
      block_size: quantization block size
    """
    N = w.shape[1]

    # Dequantize sfa/sfb: UE8M0 -> float32 scale
    sfa_scale = ue8m0_to_float32(sfa)  # (M_total, k_blocks)
    sfb_scale = ue8m0_to_float32(sfb)  # (num_experts, N, k_blocks)

    ref_out = torch.zeros([x.shape[0], N], dtype=torch.bfloat16, device=x.device)

    for expert_id in range(num_experts):
        m = masked_m[expert_id]
        if m == 0:
            continue
        start = expert_id * max_m_per_expert

        # Get current expert x slice and dequantize
        # x_slice: (m, K) fp8 -> float32
        x_slice = x[start : start + m, :].to(torch.float32)
        # sfa_slice: (m, k_blocks) -> expand to (m, K)
        sfa_slice = sfa_scale[start : start + m, :]  # (m, k_blocks)
        sfa_expanded = torch.repeat_interleave(sfa_slice, block_size, dim=1)  # (m, K)
        x_deq = x_slice * sfa_expanded  # (m, K)

        # Get current expert w slice and dequantize
        # w_slice: (N, K) fp8 -> float32
        w_slice = w[expert_id].to(torch.float32)  # (N, K)
        # sfb_slice: (N, k_blocks) -> expand to (N, K)
        sfb_slice = sfb_scale[expert_id]  # (N, k_blocks)
        sfb_expanded = torch.repeat_interleave(sfb_slice, block_size, dim=1)  # (N, K)
        w_deq = w_slice * sfb_expanded  # (N, K)

        # matmul: (m, K) @ (K, N) -> (m, N)
        result = torch.matmul(x_deq, w_deq.T).to(torch.bfloat16)
        if bias is not None:
            result = result + bias[expert_id].unsqueeze(0)
        ref_out[start : start + m, :] = result

    return ref_out


def ref_group_masked_gemm_w4a4(
    x_fp4_float, w_fp4_float, masked_m, max_m_per_expert, num_experts, alpha=None
):
    """
    Reference implementation of W4A4 grouped masked GEMM.
    Directly uses dequantized float32 values for matmul.

    Parameters:
      x_fp4_float: (M_total, K), float32 - dequantized activation
      w_fp4_float: (num_experts, N, K), float32 - dequantized weight
      masked_m: (num_experts,), int
      max_m_per_expert: int
      num_experts: int
      alpha: (num_experts,), float32. Each expert's result is multiplied by its corresponding alpha
    """
    N = w_fp4_float.shape[1]
    ref_out = torch.zeros(
        [x_fp4_float.shape[0], N], dtype=torch.bfloat16, device=x_fp4_float.device
    )

    for expert_id in range(num_experts):
        m = masked_m[expert_id]
        if m == 0:
            continue
        start = expert_id * max_m_per_expert

        x_slice = x_fp4_float[start : start + m, :]  # (m, K)
        w_slice = w_fp4_float[expert_id]  # (N, K)

        result = torch.matmul(x_slice, w_slice.T).to(torch.bfloat16)
        if alpha is not None:
            result = result * alpha[expert_id]
        ref_out[start : start + m, :] = result

    return ref_out


# =====================================================================
# Kernel invocation
# =====================================================================


def group_masked_gemm_wrapper(
    x,
    w,
    sfa,
    sfb,
    masked_m,
    max_m_per_expert,
    bias=None,
    alpha=None,
    sm_count=-1,
    mode="w8a8",
):
    output = torch.zeros(
        [x.shape[0], w.shape[1]], dtype=torch.bfloat16, device=x.device
    )
    group_masked_gemm(
        x, w, sfa, sfb, masked_m, output, bias, alpha, max_m_per_expert, sm_count, mode
    )
    return output


# =====================================================================
# W8A8 (FP8) tests
# =====================================================================


class TestGroupedGemmMasked:
    def setup_method(self):
        self.max_m_per_expert = 4096
        self.n = 7168
        self.k = 3584
        self.num_experts = 10
        self.full_x = True
        self.add_bias = False
        self.sm_count = -1
        self.mode = "w8a8"
        self.pad_n_224_to_256 = (
            False  # whether to pad sfb scale every 224 rows to 256 rows
        )

    def init_input(self):
        print(
            f"[init] num_experts={self.num_experts}, max_m_per_expert={self.max_m_per_expert}, "
            f"N={self.n}, K={self.k}, full_x={self.full_x}, add_bias={self.add_bias}, "
            f"sm_count={self.sm_count}, pad_n_224_to_256={self.pad_n_224_to_256}"
        )
        if self.full_x:
            self.masked_m = torch.full(
                [self.num_experts], self.max_m_per_expert, dtype=torch.int32
            )
        else:
            self.masked_m = torch.randint(
                0, self.max_m_per_expert, [self.num_experts], dtype=torch.int32
            )
        print(f"[init] masked_m={self.masked_m.cpu().numpy()}")
        self.x_origin = torch.randn(
            [self.num_experts * self.max_m_per_expert, self.k], dtype=torch.bfloat16
        )
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

        if self.pad_n_224_to_256:
            # Only pad sfb scale, weight matrix is not padded
            sfb_padded, n_padded = pad_sfb_scale_224_to_256(self.sfb_logical, self.n)
            print(
                f"[init] sfb scale padded: N {self.n} -> {n_padded}, "
                f"sfb_padded shape={sfb_padded.shape}"
            )
            self.sfb = reorder_sf_to_cutlass(
                sfb_padded, n_padded, self.w.shape[2] // 32
            )
        else:
            self.sfb = reorder_sf_to_cutlass(
                self.sfb_logical, self.w.shape[1], self.w.shape[2] // 32
            )

        # bias: (num_experts, n), dtype=bfloat16
        if self.add_bias:
            self.bias = torch.rand([self.num_experts, self.n], dtype=torch.bfloat16)
        else:
            self.bias = None

    def run(self):
        # ref uses logical layout scale, kernel uses CUTLASS reordered scale
        print("[test] computing reference output ...")
        ref_out = ref_group_masked_gemm(
            self.x,
            self.w,
            self.sfa_logical,
            self.sfb_logical,
            self.masked_m,
            self.max_m_per_expert,
            self.num_experts,
            bias=self.bias,
        )
        print(
            f"[test] ref_out shape={ref_out.shape}, dtype={ref_out.dtype}, "
            f"min={ref_out.to(torch.float32).min().item():.4f}, max={ref_out.to(torch.float32).max().item():.4f}"
        )
        print("[test] computing kernel output ...")
        output = group_masked_gemm_wrapper(
            self.x,
            self.w,
            self.sfa,
            self.sfb,
            self.masked_m,
            self.max_m_per_expert,
            bias=self.bias,
            mode=self.mode,
        )
        for expert_id in range(self.num_experts):
            start_offset = expert_id * self.max_m_per_expert
            m = self.masked_m[expert_id].item()
            ref_out_this_exp = ref_out[
                start_offset : start_offset + self.masked_m[expert_id], :
            ]
            output_this_exp = output[
                start_offset : start_offset + self.masked_m[expert_id], :
            ]
            diff = (
                output_this_exp.to(torch.float32) - ref_out_this_exp.to(torch.float32)
            ).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            print(
                f"[test] expert {expert_id}: m={m}, max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}"
            )
            np.testing.assert_allclose(
                output_this_exp.to(torch.float32).cpu().numpy(),
                ref_out_this_exp.to(torch.float32).cpu().numpy(),
                rtol=2e-2,
                atol=2e-3,
            )
        print(
            f"  [PASS] W8A8 max_m={self.max_m_per_expert}, n={self.n}, "
            f"k={self.k}, experts={self.num_experts}, full_x={self.full_x}"
        )


# =====================================================================
# W4A4 (NVFP4) tests
# =====================================================================

FP4_BLOCK_SIZE = 16  # NVFP4 SFVecSize=16, one scale per 16 elements


class TestGroupedGemmMaskedW4A4:
    """
    W4A4 (NVFP4) grouped masked GEMM test.
    Quantization: FP4 E2M1 (float_e2m1_t), one scale per 16 elements, scale stored as UE4M3.
    SF permute: CUTLASS SfAtom ((32,4),(16,4)) stride ((16,4),(0,1)),
                i.e., scale_vec::4X / block16 with K=64 mode.
    """

    def setup_method(self):
        self.max_m_per_expert = 4096
        self.n = 7168
        self.k = 3584  # must be divisible by 16 (block_size) and 2 (packing)
        self.num_experts = 20
        self.full_x = True
        self.sm_count = -1
        self.mode = "w4a4-nvfp4"
        self.pad_n_224_to_256 = (
            False  # Whether to pad sfb scale from 224 rows to 256 rows
        )
        # W4A4 requires alpha: per-expert fp32 scaling factor applied in epilogue
        self.alpha = np.ones(
            1, dtype=np.float32
        ).tolist()  # default: all 1.0 (identity)

    def init_input(self):
        if self.full_x:
            self.masked_m = torch.full(
                [self.num_experts], self.max_m_per_expert, dtype=torch.int32
            )
        else:
            self.masked_m = torch.randint(
                0, self.max_m_per_expert, [self.num_experts], dtype=torch.int32
            )

        M_total = self.num_experts * self.max_m_per_expert
        x_origin = torch.randn([M_total, self.k], dtype=torch.bfloat16)
        w_origin = torch.randn([self.num_experts, self.n, self.k], dtype=torch.bfloat16)

        # FP4 quantization: block_size=16, scale uses UE4M3 encoding
        self.x_packed, self.sfa_logical, self.x_fp4_float = per_token_cast_to_fp4(
            x_origin, FP4_BLOCK_SIZE
        )
        self.w_packed, self.sfb_logical, self.w_fp4_float = per_block_cast_to_fp4(
            w_origin, FP4_BLOCK_SIZE
        )

        # SF permute: CUTLASS SfAtom interleaved layout
        kb_dim = self.k // FP4_BLOCK_SIZE
        self.sfa = reorder_sf_to_cutlass(self.sfa_logical, M_total, kb_dim)
        if self.pad_n_224_to_256:
            # Only pad sfb scale, weight matrix is not padded
            sfb_padded, n_padded = pad_sfb_scale_224_to_256(self.sfb_logical, self.n)
            print(
                f"[init] sfb scale padded: N {self.n} -> {n_padded}, "
                f"sfb_padded shape={sfb_padded.shape}"
            )
            self.sfb = reorder_sf_to_cutlass(sfb_padded, n_padded, kb_dim)
        else:
            self.sfb = reorder_sf_to_cutlass(self.sfb_logical, self.n, kb_dim)

        # alpha: (num_experts,), dtype=float32 — W4A4 requires alpha
        if self.alpha is None:
            self.alpha = np.ones(self.num_experts, dtype=np.float32).tolist()
        # Broadcast scalar alpha to per-expert array if needed
        if len(self.alpha) == 1:
            self.alpha = self.alpha * self.num_experts
        self.alpha_tensor = torch.tensor(self.alpha, dtype=torch.float32)

    def run(self):
        # Reference: use dequantized float32 values for matmul
        ref_out = ref_group_masked_gemm_w4a4(
            self.x_fp4_float,
            self.w_fp4_float,
            self.masked_m,
            self.max_m_per_expert,
            self.num_experts,
            alpha=self.alpha_tensor,
        )

        # Kernel: pass packed uint8 + CUTLASS reordered UE4M3 scale, mode="w4a4-nvfp4"
        for _ in range(3):
            output = group_masked_gemm_wrapper(
                self.x_packed,
                self.w_packed,
                self.sfa,
                self.sfb,
                self.masked_m,
                self.max_m_per_expert,
                alpha=self.alpha_tensor,
                sm_count=self.sm_count,
                mode=self.mode,
            )

        for expert_id in range(self.num_experts):
            start_offset = expert_id * self.max_m_per_expert
            m = int(self.masked_m[expert_id])
            if m == 0:
                continue
            ref_out_this_exp = ref_out[start_offset : start_offset + m, :]
            output_this_exp = output[start_offset : start_offset + m, :]

            np.testing.assert_allclose(
                output_this_exp.to(torch.float32).cpu().numpy(),
                ref_out_this_exp.to(torch.float32).cpu().numpy(),
                rtol=5e-2,
                atol=5e-2,
            )
        print(
            f"  [PASS] W4A4 max_m={self.max_m_per_expert}, n={self.n}, "
            f"k={self.k}, experts={self.num_experts}, full_x={self.full_x}"
        )


def test_group_masked_w8a8():
    t = TestGroupedGemmMasked()
    t.setup_method()
    t.max_m_per_expert = 128
    t.num_experts = 4
    t.full_x = True
    t.init_input()
    t.run()


def test_group_masked_w4a4():
    t = TestGroupedGemmMaskedW4A4()
    t.setup_method()
    t.max_m_per_expert = 128
    t.num_experts = 4
    t.full_x = True
    t.init_input()
    t.run()


def test_group_masked_w8a8_large_m():
    """W8A8 large path: max_m_per_expert=4096 across many experts."""
    t = TestGroupedGemmMasked()
    t.setup_method()
    t.max_m_per_expert = 4096
    t.num_experts = 20
    t.full_x = True
    t.pad_n_224_to_256 = False
    t.init_input()
    t.run()


def test_group_masked_w8a8_small_m():
    """W8A8 small path: max_m_per_expert=64 across many experts."""
    t = TestGroupedGemmMasked()
    t.setup_method()
    t.max_m_per_expert = 64
    t.num_experts = 20
    t.full_x = True
    t.pad_n_224_to_256 = False
    t.init_input()
    t.run()


def test_group_masked_w4a4_large_m():
    """W4A4 large path: max_m_per_expert=4096 across many experts."""
    t = TestGroupedGemmMaskedW4A4()
    t.setup_method()
    t.max_m_per_expert = 4096
    t.num_experts = 20
    t.init_input()
    t.run()


def test_group_masked_w4a4_with_alpha():
    """W4A4 large path with per-expert alpha scaling."""
    t = TestGroupedGemmMaskedW4A4()
    t.setup_method()
    t.max_m_per_expert = 4096
    t.num_experts = 20
    t.alpha = (
        np.random.RandomState(2026)
        .uniform(0.5, 2.0, size=t.num_experts)
        .astype(np.float32)
        .tolist()
    )
    t.init_input()
    t.run()


def test_group_masked_additional_shapes():
    print("=" * 60)
    print("Additional GroupMasked GEMM shape tests")
    print("=" * 60)

    test = TestGroupedGemmMasked()
    test.setup_method()
    test.max_m_per_expert = 128
    test.num_experts = 4
    test.n = 256
    test.k = 256
    test.full_x = False
    test.init_input()
    test.masked_m = torch.tensor([17, 63, 95, 127], dtype=torch.int32)
    test.run()

    t4 = TestGroupedGemmMaskedW4A4()
    t4.setup_method()
    t4.max_m_per_expert = 128
    t4.num_experts = 4
    t4.n = 128
    t4.k = 512
    t4.full_x = False
    t4.init_input()
    t4.masked_m = torch.tensor([11, 37, 89, 128], dtype=torch.int32)
    t4.run()


def test_group_masked_unsupported_shapes():
    print("=" * 60)
    print("GroupMasked GEMM unsupported shape checks")
    print("=" * 60)

    def run_w8a8_bad_max_m():
        num_experts, max_m, n, k = 2, 129, 256, 128
        x = torch.empty([num_experts * max_m, k], dtype=torch.float8_e4m3fn)
        w = torch.empty([num_experts, n, k], dtype=torch.float8_e4m3fn)
        sfa = torch.empty([num_experts * max_m, k // 32], dtype=torch.uint8)
        sfb = torch.empty([num_experts, n, k // 32], dtype=torch.uint8)
        masked_m = torch.full([num_experts], max_m, dtype=torch.int32)
        out = torch.empty([num_experts * max_m, n], dtype=torch.bfloat16)
        group_masked_gemm(x, w, sfa, sfb, masked_m, out, None, None, max_m, -1, "w8a8")

    _expect_runtime_error(run_w8a8_bad_max_m, "requires M to be divisible by 128")

    def run_w4a4_bias():
        num_experts, max_m, n, k = 2, 128, 128, 512
        x = torch.empty([num_experts * max_m, k // 2], dtype=torch.uint8)
        w = torch.empty([num_experts, n, k // 2], dtype=torch.uint8)
        sfa = torch.empty([num_experts * max_m, k // 16], dtype=torch.uint8)
        sfb = torch.empty([num_experts, n, k // 16], dtype=torch.uint8)
        masked_m = torch.full([num_experts], max_m, dtype=torch.int32)
        out = torch.empty([num_experts * max_m, n], dtype=torch.bfloat16)
        bias = torch.empty([num_experts, n], dtype=torch.bfloat16)
        alpha = torch.ones([num_experts], dtype=torch.float32)
        group_masked_gemm(x, w, sfa, sfb, masked_m, out, bias, alpha, max_m, -1, "w4a4-nvfp4")

    _expect_runtime_error(run_w4a4_bias, "W4A4 mode does not support bias yet")

    def run_zero_max_m_per_expert():
        num_experts, max_m, n, k = 2, 0, 256, 128
        x = torch.empty([0, k], dtype=torch.float8_e4m3fn)
        w = torch.empty([num_experts, n, k], dtype=torch.float8_e4m3fn)
        sfa = torch.empty([0, k // 32], dtype=torch.uint8)
        sfb = torch.empty([num_experts, n, k // 32], dtype=torch.uint8)
        masked_m = torch.zeros([num_experts], dtype=torch.int32)
        out = torch.empty([0, n], dtype=torch.bfloat16)
        group_masked_gemm(x, w, sfa, sfb, masked_m, out, None, None, max_m, -1, "w8a8")

    _expect_runtime_error(
        run_zero_max_m_per_expert, "max_m_per_expert must be greater than 0"
    )

    def run_masked_m_int64():
        num_experts, max_m, n, k = 2, 128, 256, 128
        x = torch.empty([num_experts * max_m, k], dtype=torch.float8_e4m3fn)
        w = torch.empty([num_experts, n, k], dtype=torch.float8_e4m3fn)
        sfa = torch.empty([num_experts * max_m, k // 32], dtype=torch.uint8)
        sfb = torch.empty([num_experts, n, k // 32], dtype=torch.uint8)
        masked_m = torch.full([num_experts], max_m, dtype=torch.int64)
        out = torch.empty([num_experts * max_m, n], dtype=torch.bfloat16)
        group_masked_gemm(x, w, sfa, sfb, masked_m, out, None, None, max_m, -1, "w8a8")

    _expect_runtime_error(run_masked_m_int64, "masked_m must have dtype torch.int32")

    def run_bad_rank_output():
        num_experts, max_m, n, k = 2, 128, 256, 128
        x = torch.empty([num_experts * max_m, k], dtype=torch.float8_e4m3fn)
        w = torch.empty([num_experts, n, k], dtype=torch.float8_e4m3fn)
        sfa = torch.empty([num_experts * max_m, k // 32], dtype=torch.uint8)
        sfb = torch.empty([num_experts, n, k // 32], dtype=torch.uint8)
        masked_m = torch.full([num_experts], max_m, dtype=torch.int32)
        out = torch.empty([num_experts * max_m, n, 1], dtype=torch.bfloat16)
        group_masked_gemm(x, w, sfa, sfb, masked_m, out, None, None, max_m, -1, "w8a8")

    _expect_runtime_error(run_bad_rank_output, "output must be a 2D tensor")


def test_group_masked_w8a8_determinism():
    """Determinism: identical inputs must produce bitwise-identical outputs."""
    t = TestGroupedGemmMasked()
    t.setup_method()
    t.max_m_per_expert = 128
    t.num_experts = 4
    t.n = 256
    t.k = 256
    t.full_x = True
    t.init_input()
    out1 = group_masked_gemm_wrapper(
        t.x,
        t.w,
        t.sfa,
        t.sfb,
        t.masked_m,
        t.max_m_per_expert,
        bias=t.bias,
        mode=t.mode,
    )
    out2 = group_masked_gemm_wrapper(
        t.x,
        t.w,
        t.sfa,
        t.sfb,
        t.masked_m,
        t.max_m_per_expert,
        bias=t.bias,
        mode=t.mode,
    )
    assert torch.equal(out1, out2), "W8A8 group_masked_gemm output is not deterministic"


def test_group_masked_w4a4_determinism():
    """Determinism: identical inputs must produce bitwise-identical outputs."""
    t = TestGroupedGemmMaskedW4A4()
    t.setup_method()
    t.max_m_per_expert = 128
    t.num_experts = 4
    t.n = 128
    t.k = 512
    t.full_x = True
    t.init_input()
    out1 = group_masked_gemm_wrapper(
        t.x_packed,
        t.w_packed,
        t.sfa,
        t.sfb,
        t.masked_m,
        t.max_m_per_expert,
        alpha=t.alpha_tensor,
        sm_count=t.sm_count,
        mode=t.mode,
    )
    out2 = group_masked_gemm_wrapper(
        t.x_packed,
        t.w_packed,
        t.sfa,
        t.sfb,
        t.masked_m,
        t.max_m_per_expert,
        alpha=t.alpha_tensor,
        sm_count=t.sm_count,
        mode=t.mode,
    )
    assert torch.equal(out1, out2), "W4A4 group_masked_gemm output is not deterministic"


if __name__ == "__main__":
    pytest.main([__file__])
