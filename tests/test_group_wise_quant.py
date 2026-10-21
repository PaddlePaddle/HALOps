# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re

import numpy as np
import pytest
import torch

from halops import group_wise_quant
from tests.utils.fp4_utils import (
    FP4_E2M1_MAX,
    FP4_E2M1_VALUES,
    float32_to_ue4m3_rne,
    fp4_abs_to_idx_round_half_up,
)
from tests.utils.gemm_utils import ue8m0_to_float32


# ============================================================================
# FP8 reference implementation
# ============================================================================


def float32_to_ue8m0(x):
    """Encode a float32 tensor into UE8M0 under the current CUDA fast-math logic."""
    x_f32 = torch.clip(x.to(torch.float32).abs(), min=1e-38)
    x_np = x_f32.cpu().numpy().astype(np.float32)
    log2_np = np.log2(x_np).astype(np.float32)

    raw_bits = x_np.view(np.uint32)
    exact_power_under_one = ((raw_bits & 0x007FFFFF) == 0) & (x_np < 1.0)
    log2_np[exact_power_under_one] = np.nextafter(
        log2_np[exact_power_under_one], np.float32(np.inf)
    )

    biased_exp = np.clip(np.ceil(log2_np) + 127.0, 0.0, 254.0).astype(np.uint8)
    return torch.tensor(biased_exp, dtype=torch.uint8, device=x.device)


def ref_per_token_cast_to_fp8(x, group_size=32):
    """
    FP8 reference implementation: quantize along k dimension with group_size granularity
    to float8_e4m3fn with UE8M0 scale.
    Returns: (fp8_tensor, scale_ue8m0)
    """
    assert x.dtype == torch.bfloat16
    m, k = x.shape
    assert k % group_size == 0

    num_groups = k // group_size
    x_grouped = x.reshape([m, num_groups, group_size]).to(torch.float32)
    amax = x_grouped.abs().amax(dim=-1).clip(min=1e-4)

    FP8_E4M3_MAX = 448.0
    scale = amax / FP8_E4M3_MAX
    scale_ue8m0 = float32_to_ue8m0(scale)
    scale_pow2 = ue8m0_to_float32(scale_ue8m0)

    x_scaled = x_grouped / scale_pow2.unsqueeze(-1)
    x_scaled = x_scaled.clamp(min=-FP8_E4M3_MAX, max=FP8_E4M3_MAX)
    x_fp8 = x_scaled.reshape([m, k]).to(torch.float8_e4m3fn)

    return x_fp8, scale_ue8m0


# ============================================================================
# FP4 reference implementation
# ============================================================================


def ue4m3_to_float32(scale_ue4m3):
    """Decode a UE4M3 (uint8) bit pattern into a float32 scale value."""
    if isinstance(scale_ue4m3, torch.Tensor):
        codes = scale_ue4m3.cpu().numpy()
    else:
        codes = np.asarray(scale_ue4m3, dtype=np.uint8)
    fp8_tensor = torch.tensor(codes, dtype=torch.uint8).view(torch.float8_e4m3fn)
    return fp8_tensor.to(torch.float32).cpu().numpy()


def ref_per_token_cast_to_fp4(x, group_size=16):
    """
    FP4 reference implementation: quantize along k dimension with group_size granularity
    to FP4 E2M1 with UE4M3 scale.
    Returns: (packed_uint8, scale_ue4m3, fp4_dequant_float32)
      - packed_uint8: (m, k // 2), dtype=uint8
      - scale_ue4m3: (m, k // group_size), dtype=uint8
      - fp4_dequant_float32: (m, k), dtype=float32
    """
    assert x.dtype == torch.bfloat16
    m, k = x.shape
    assert k % group_size == 0
    assert k % 2 == 0

    x_f32 = x.to(torch.float32).cpu().numpy()
    num_groups = k // group_size
    x_blocked = x_f32.reshape(m, num_groups, group_size)

    # per-group amax
    amax = np.abs(x_blocked).max(axis=-1)  # (m, num_groups)
    amax = np.clip(amax, 1e-12, None)

    scale = amax / FP4_E2M1_MAX  # (m, num_groups)

    # UE4M3 encoding: use torch float8_e4m3fn cast (RNE), consistent with CUDA __nv_fp8_e4m3
    scale_ue4m3 = float32_to_ue4m3_rne(scale)  # (m, num_groups), uint8
    # When scale is too small, UE4M3 may round to 0; use minimum positive value (consistent with CUDA kernel)
    zero_mask = (scale_ue4m3 == 0) & (amax > 0.0)
    scale_ue4m3[zero_mask] = 0x01

    # Decode scale (consistent with CUDA kernel's ue4m3_to_float32)
    scale_decoded = ue4m3_to_float32(scale_ue4m3)  # (m, num_groups), float32

    # Use multiplication (consistent with CUDA kernel: vals[i] * inv_scale)
    inv_scale = np.float32(1.0) / scale_decoded.astype(np.float32)
    x_scaled = x_blocked * inv_scale[..., np.newaxis]
    x_scaled = np.clip(x_scaled, -FP4_E2M1_MAX, FP4_E2M1_MAX)

    # Compute FP4 codes directly from x_scaled (consistent with CUDA quantize_to_fp4_code)
    flat = x_scaled.reshape(m, k)
    sign_bits = (flat < 0).astype(np.uint8) << 3
    mag_idx = fp4_abs_to_idx_round_half_up(np.abs(flat))
    fp4_codes = sign_bits | mag_idx  # (m, k), 4-bit codes

    # Dequantized values (for reference computation)
    fp4_float = (np.where(flat < 0, -1.0, 1.0) * FP4_E2M1_VALUES[mag_idx]).astype(
        np.float32
    )
    scale_expanded = np.repeat(scale_decoded, group_size, axis=-1)  # (m, k)
    fp4_dequant = fp4_float * scale_expanded

    # Pack to uint8: even index = low nibble, odd index = high nibble
    lo = fp4_codes[:, 0::2]
    hi = fp4_codes[:, 1::2]
    packed = (hi << 4) | lo  # (m, k // 2)

    return (
        torch.tensor(packed, dtype=torch.uint8),
        torch.tensor(scale_ue4m3, dtype=torch.uint8),
        torch.tensor(fp4_dequant, dtype=torch.float32),
    )


# ============================================================================
# CUTLASS SfAtom interleaved reorder (shared by W8A8 and W4A4)
# ============================================================================


def ref_reorder_sf_to_cutlass(sf_logical, mn_dim, kb_dim):
    """
    Reorder logical-layout scale factor (uint8) to CUTLASS SfAtom interleaved layout.
    Input must be uint8 type.
    """
    sf_np = (
        sf_logical.cpu().numpy()
        if isinstance(sf_logical, torch.Tensor)
        else np.asarray(sf_logical)
    )
    if sf_np.dtype != np.uint8:
        sf_np = sf_np.view(np.uint8)
    orig_shape = sf_np.shape
    flat = sf_np.reshape(-1, mn_dim, kb_dim)

    n_idx = np.arange(mn_dim)
    kb_idx = np.arange(kb_dim)
    n_grid, kb_grid = np.meshgrid(n_idx, kb_idx, indexing="ij")

    n_tile = n_grid // 128
    n_local = n_grid % 128
    mn_i = n_local % 32
    mn_j = n_local // 32
    k_tile = kb_grid // 4
    sf_l = kb_grid % 4
    num_k_tiles = kb_dim // 4

    cutlass_byte = (n_tile * num_k_tiles + k_tile) * 512 + mn_i * 16 + mn_j * 4 + sf_l
    n_phys = cutlass_byte // kb_dim
    kb_phys = cutlass_byte % kb_dim

    result = np.empty_like(flat)
    result[:, n_phys, kb_phys] = flat[:, n_grid, kb_grid]

    return torch.tensor(result.reshape(orig_shape), dtype=torch.uint8)


# ============================================================================
# Tests
# ============================================================================

# Bit-exact FP8 data comparison (as float32).
FP8_EXACT_TOL = dict(rtol=0, atol=0)
# FP4 dequantized-value comparison (UE4M3 rounding differences).
FP4_DEQUANT_TOL = dict(rtol=0.01, atol=0.01)

# (m, k) - various sizes
SWIZZLE_CASES = [
    (128, 1024),
    (256, 2048),
    (512, 3584),
    (1024, 4096),
    (2048, 7168),
    (4096, 3584),
    (8192, 7168),
    (32768, 3584),  # large scale test
]
# use_swizzle=False does not require m%128, add non-aligned sizes
NO_SWIZZLE_CASES = SWIZZLE_CASES + [(64, 1024), (192, 2048), (320, 3584)]

QUANT_MODES = [(False, False), (False, True), (True, False), (True, True)]


def _mk_id(case):
    m, k = case
    return f"m{m}_k{k}"


def _mode_id(mode):
    is_fp4, use_swizzle = mode
    return f"{'fp4' if is_fp4 else 'fp8'}_{'swizzle' if use_swizzle else 'linear'}"


def _unpack_fp4_values(packed, m, k):
    """Decode packed FP4 E2M1 bytes (m, k // 2) into float32 values (m, k)."""
    packed_np = packed.cpu().numpy()
    codes = np.zeros((m, k), dtype=np.uint8)
    codes[:, 0::2] = packed_np & 0x0F
    codes[:, 1::2] = (packed_np >> 4) & 0x0F
    signs = np.where(codes & 0x8, -1.0, 1.0)
    mags = FP4_E2M1_VALUES[codes & 0x7]
    return (signs * mags).astype(np.float32)


@pytest.mark.parametrize("m,k", SWIZZLE_CASES, ids=[_mk_id(c) for c in SWIZZLE_CASES])
def test_correctness_fp8(m, k):
    """Test is_fp4=False mode (UE8M0 scale, group_size=32)"""
    group_size = 32
    x = torch.randn([m, k], dtype=torch.bfloat16)

    ref_fp8, ref_sf_logical = ref_per_token_cast_to_fp8(x, group_size)
    ref_sf_cutlass = ref_reorder_sf_to_cutlass(ref_sf_logical, m, k // group_size)

    # CUDA operator: is_fp4=False, use_swizzle=True
    cuda_fp8, cuda_sf = group_wise_quant(x, False, True)

    torch.testing.assert_close(
        cuda_fp8.to(torch.float32),
        ref_fp8.to(torch.float32),
        **FP8_EXACT_TOL,
        msg=lambda m_: f"FP8 data mismatch for m={m}, k={k}: {m_}",
    )
    assert torch.equal(
        cuda_sf, ref_sf_cutlass
    ), f"FP8 swizzled scale mismatch for m={m}, k={k}"


@pytest.mark.parametrize("m,k", SWIZZLE_CASES, ids=[_mk_id(c) for c in SWIZZLE_CASES])
def test_correctness_fp4(m, k):
    """Test is_fp4=True mode (UE4M3 scale, group_size=16, packed uint8)"""
    group_size = 16
    x = torch.randn([m, k], dtype=torch.bfloat16)

    ref_packed, ref_sf_logical, ref_dequant = ref_per_token_cast_to_fp4(x, group_size)
    ref_sf_cutlass = ref_reorder_sf_to_cutlass(ref_sf_logical, m, k // group_size)

    # CUDA operator: is_fp4=True, use_swizzle=True
    cuda_packed, cuda_sf = group_wise_quant(x, True, True)

    # Packed FP4 data (uint8, byte-by-byte)
    assert torch.equal(
        cuda_packed, ref_packed
    ), f"FP4 packed data mismatch for m={m}, k={k}"
    assert torch.equal(
        cuda_sf, ref_sf_cutlass
    ), f"FP4 swizzled scale mismatch for m={m}, k={k}"

    # Dequantized values (allow small error from UE4M3 rounding differences).
    # ref_sf_logical stores raw UE4M3 bit patterns, so decode them to float32
    # before dividing ref_dequant back to plain FP4 values.
    cuda_fp4_vals = _unpack_fp4_values(cuda_packed, m, k)
    ref_scale_f32 = ue4m3_to_float32(ref_sf_logical).reshape(m, k // group_size, 1)
    ref_fp4_vals = (
        ref_dequant.cpu().numpy().reshape(m, k // group_size, group_size)
        / ref_scale_f32
    )
    torch.testing.assert_close(
        torch.from_numpy(cuda_fp4_vals.reshape(m, k // group_size, group_size)),
        torch.from_numpy(ref_fp4_vals),
        **FP4_DEQUANT_TOL,
        check_dtype=False,
        msg=lambda m_: f"FP4 dequant mismatch for m={m}, k={k}: {m_}",
    )


@pytest.mark.parametrize(
    "m,k", NO_SWIZZLE_CASES, ids=[_mk_id(c) for c in NO_SWIZZLE_CASES]
)
def test_correctness_fp8_no_swizzle(m, k):
    """Test is_fp4=False, use_swizzle=False mode (UE8M0 scale, linear layout)"""
    group_size = 32
    x = torch.randn([m, k], dtype=torch.bfloat16)

    # No swizzle: compare against the logical layout directly
    ref_fp8, ref_sf_logical = ref_per_token_cast_to_fp8(x, group_size)

    # CUDA operator: is_fp4=False, use_swizzle=False
    cuda_fp8, cuda_sf = group_wise_quant(x, False, False)

    torch.testing.assert_close(
        cuda_fp8.to(torch.float32),
        ref_fp8.to(torch.float32),
        **FP8_EXACT_TOL,
        msg=lambda m_: f"FP8 no-swizzle data mismatch for m={m}, k={k}: {m_}",
    )
    assert torch.equal(
        cuda_sf, ref_sf_logical
    ), f"FP8 no-swizzle scale mismatch for m={m}, k={k}"


@pytest.mark.parametrize(
    "m,k", NO_SWIZZLE_CASES, ids=[_mk_id(c) for c in NO_SWIZZLE_CASES]
)
def test_correctness_fp4_no_swizzle(m, k):
    """Test is_fp4=True, use_swizzle=False mode (UE4M3 scale, linear layout)"""
    group_size = 16
    x = torch.randn([m, k], dtype=torch.bfloat16)

    ref_packed, ref_sf_logical, _ = ref_per_token_cast_to_fp4(x, group_size)

    # CUDA operator: is_fp4=True, use_swizzle=False
    cuda_packed, cuda_sf = group_wise_quant(x, True, False)

    assert torch.equal(
        cuda_packed, ref_packed
    ), f"FP4 no-swizzle packed data mismatch for m={m}, k={k}"
    assert torch.equal(
        cuda_sf, ref_sf_logical
    ), f"FP4 no-swizzle scale mismatch for m={m}, k={k}"


@pytest.mark.parametrize("m", [1, 8, 64, 127], ids=lambda m: f"m{m}")
def test_small_m_swizzle_shapes(m):
    """Small m is legal under use_swizzle=True: the data keeps its own row
    count and only the scale tensor is rounded up to whole 128-row atoms.
    The dense W8A8 GEMM relies on that (it dispatches on the activation's
    row count and narrows its A TMA box to it)."""
    k = 1024
    x = torch.randn([m, k], dtype=torch.bfloat16)
    data, sf = group_wise_quant(x, False, True)
    assert list(data.shape) == [
        m,
        k,
    ], f"m={m}: data should keep its row count, got {list(data.shape)}"
    assert list(sf.shape) == [
        128,
        k // 32,
    ], f"m={m}: scale must cover one whole 128-row atom, got {list(sf.shape)}"


def test_unsupported_swizzle_shapes():
    """Test that unsupported SfAtom shapes under use_swizzle=True raise clearly."""
    x_bad_k = torch.randn([128, 96], dtype=torch.bfloat16)
    with pytest.raises(
        RuntimeError, match=re.escape("scale kb_dim must be divisible by 4")
    ):
        group_wise_quant(x_bad_k, False, True)


@pytest.mark.parametrize(
    "is_fp4,use_swizzle", QUANT_MODES, ids=[_mode_id(m) for m in QUANT_MODES]
)
def test_empty_input(is_fp4, use_swizzle):
    """Test m=0 empty input does not error"""
    k = 1024
    x = torch.empty([0, k], dtype=torch.bfloat16)
    cuda_out, cuda_sf = group_wise_quant(x, is_fp4, use_swizzle)
    group_size = 16 if is_fp4 else 32
    expected_quant_shape = [0, k // 2] if is_fp4 else [0, k]
    assert list(cuda_out.shape) == expected_quant_shape, (
        f"empty-input quant shape mismatch (is_fp4={is_fp4}, "
        f"use_swizzle={use_swizzle}): got {list(cuda_out.shape)}, "
        f"expected {expected_quant_shape}"
    )
    assert list(cuda_sf.shape) == [0, k // group_size], (
        f"empty-input sf shape mismatch (is_fp4={is_fp4}, "
        f"use_swizzle={use_swizzle}): got {list(cuda_sf.shape)}, "
        f"expected {[0, k // group_size]}"
    )


@pytest.mark.parametrize(
    "is_fp4,use_swizzle", QUANT_MODES, ids=[_mode_id(m) for m in QUANT_MODES]
)
def test_determinism(is_fp4, use_swizzle):
    """Determinism: identical inputs must produce bitwise-identical outputs."""
    m, k = 256, 2048
    x = torch.randn([m, k], dtype=torch.bfloat16)

    out1, sf1 = group_wise_quant(x, is_fp4, use_swizzle)
    out2, sf2 = group_wise_quant(x, is_fp4, use_swizzle)

    assert torch.equal(out1, out2), (
        f"group_wise_quant quant output is not deterministic "
        f"(is_fp4={is_fp4}, use_swizzle={use_swizzle})"
    )
    assert torch.equal(sf1, sf2), (
        f"group_wise_quant scale output is not deterministic "
        f"(is_fp4={is_fp4}, use_swizzle={use_swizzle})"
    )


if __name__ == "__main__":
    pytest.main([__file__])
