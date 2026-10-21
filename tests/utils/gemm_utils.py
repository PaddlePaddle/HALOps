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

Shared test helpers for GEMM quantization tests (FP8 / NVFP4).

These reference implementations and CUTLASS SfAtom reorder helpers are shared
across the dense / grouped / batched GEMM test files. Note that
test_group_wise_quant.py intentionally keeps its own ``ref_*`` variants because
those are bit-accurate to the CUDA kernel (RNE UE4M3 cast, reciprocal-multiply,
side='right' tie-break) and are not interchangeable with the generic references
below.
"""

import numpy as np
import torch

# =====================================================================
# FP8 (W8A8) quantization helpers
# =====================================================================


def float32_to_ue8m0(x):
    """Convert a float32 tensor to UE8M0 encoding (uint8)."""
    x_f32 = x.to(torch.float32).abs()
    x_f32 = torch.clip(x_f32, min=1e-38)  # avoid log(0)
    biased_exp = torch.clip(torch.ceil(torch.log2(x_f32)) + 127, 0, 254)
    return biased_exp.to(torch.uint8)


def ue8m0_to_float32(scale_ue8m0):
    """Decode UE8M0 (uint8) back to a float32 scale value: value = 2^(byte - 127)."""
    exp = scale_ue8m0.to(torch.float32) - 127.0
    return torch.pow(torch.tensor(2.0, dtype=torch.float32), exp)


def per_token_cast_to_fp8(x, block_size=32):
    """Quantize an (m, k) bfloat16 tensor to float8_e4m3fn at block_size granularity along k."""
    assert x.dtype == torch.bfloat16
    m, k = x.shape
    assert k % block_size == 0, f"k({k}) must be divisible by block_size({block_size})"

    num_blocks = k // block_size
    x_blocked = x.reshape([m, num_blocks, block_size]).to(torch.float32)
    amax = x_blocked.abs().amax(dim=-1).clip(min=1e-4)

    FP8_E4M3_MAX = 448.0
    scale = amax / FP8_E4M3_MAX

    scale_ue8m0 = float32_to_ue8m0(scale)
    scale_pow2 = ue8m0_to_float32(scale_ue8m0)

    x_scaled = x_blocked / scale_pow2.unsqueeze(-1)
    x_scaled = x_scaled.clip(min=-FP8_E4M3_MAX, max=FP8_E4M3_MAX)
    x_fp8 = x_scaled.reshape([m, k]).to(torch.float8_e4m3fn)

    return x_fp8, scale_ue8m0


def per_block_cast_to_fp8(x, block_size=32):
    """Quantize a bfloat16 tensor to float8_e4m3fn in n×k 2D blocks.

    Accepts either a 2D (n, k) tensor (single expert) or a 3D (b, n, k) tensor
    (batched / grouped experts); returns (fp8_tensor, scale_ue8m0) with the
    same rank as the input.
    """
    assert x.dtype == torch.bfloat16
    FP8_E4M3_MAX = 448.0

    if x.dim() == 3:
        b, n, k = x.shape
        assert (
            n % block_size == 0
        ), f"n({n}) must be divisible by block_size({block_size})"
        assert (
            k % block_size == 0
        ), f"k({k}) must be divisible by block_size({block_size})"

        n_blocks = n // block_size
        k_blocks = k // block_size

        x_blocked = x.reshape([b, n_blocks, block_size, k_blocks, block_size]).to(
            torch.float32
        )
        amax = x_blocked.abs().amax(dim=-1).amax(dim=2)

        scale = (amax / FP8_E4M3_MAX).clip(min=1e-12)
        scale_ue8m0_compact = float32_to_ue8m0(scale)
        scale_pow2 = ue8m0_to_float32(scale_ue8m0_compact)

        x_scaled = x_blocked / scale_pow2.unsqueeze(2).unsqueeze(-1)
        x_scaled = x_scaled.clip(min=-FP8_E4M3_MAX, max=FP8_E4M3_MAX)
        x_fp8 = x_scaled.reshape([b, n, k]).to(torch.float8_e4m3fn)

        scale_ue8m0 = torch.tensor(
            np.repeat(scale_ue8m0_compact.cpu().numpy(), block_size, axis=1),
            dtype=torch.uint8,
        )
        return x_fp8, scale_ue8m0

    n, k = x.shape
    assert n % block_size == 0, f"n({n}) must be divisible by block_size({block_size})"
    assert k % block_size == 0, f"k({k}) must be divisible by block_size({block_size})"

    n_blocks = n // block_size
    k_blocks = k // block_size

    x_blocked = x.reshape([n_blocks, block_size, k_blocks, block_size]).to(
        torch.float32
    )
    amax = x_blocked.abs().amax(dim=-1).amax(dim=1)

    scale = (amax / FP8_E4M3_MAX).clip(min=1e-12)
    scale_ue8m0_compact = float32_to_ue8m0(scale)
    scale_pow2 = ue8m0_to_float32(scale_ue8m0_compact)

    x_scaled = x_blocked / scale_pow2.unsqueeze(1).unsqueeze(-1)
    x_scaled = x_scaled.clip(min=-FP8_E4M3_MAX, max=FP8_E4M3_MAX)
    x_fp8 = x_scaled.reshape([n, k]).to(torch.float8_e4m3fn)

    scale_ue8m0 = torch.tensor(
        np.repeat(scale_ue8m0_compact.cpu().numpy(), block_size, axis=0),
        dtype=torch.uint8,
    )
    return x_fp8, scale_ue8m0


# =====================================================================
# NVFP4 (W4A4) quantization helpers
# =====================================================================

FP4_E2M1_MAX = 6.0
FP4_E2M1_VALUES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)

_FP4_MIDPOINTS = (FP4_E2M1_VALUES[:-1] + FP4_E2M1_VALUES[1:]) / 2.0


def _fp4_abs_to_idx(abs_x_flat):
    return np.searchsorted(_FP4_MIDPOINTS, abs_x_flat).astype(np.uint8)


def float_to_fp4_e2m1(x_f32):
    orig_shape = x_f32.shape
    flat = x_f32.ravel()
    sign = np.sign(flat)
    idx = _fp4_abs_to_idx(np.abs(flat))
    return (sign * FP4_E2M1_VALUES[idx]).reshape(orig_shape).astype(np.float32)


def encode_fp4_e2m1_array(vals):
    flat = vals.ravel()
    sign_bits = (flat < 0).astype(np.uint8) << 3
    codes = _fp4_abs_to_idx(np.abs(flat))
    return (sign_bits | codes).reshape(vals.shape)


def pack_fp4_to_uint8(fp4_float_array):
    shape = fp4_float_array.shape
    assert shape[-1] % 2 == 0, "K must be even for FP4 packing"
    flat = fp4_float_array.reshape(-1, shape[-1])
    encoded = encode_fp4_e2m1_array(flat)
    lo = encoded[..., 0::2]
    hi = encoded[..., 1::2]
    packed = (hi << 4) | lo
    out_shape = list(shape)
    out_shape[-1] = shape[-1] // 2
    return packed.reshape(out_shape)


_UE4M3_CODE2FLOAT = np.zeros(128, dtype=np.float64)
for _c in range(128):
    _E, _M = (_c >> 3) & 0xF, _c & 0x7
    if _E == 0:
        _UE4M3_CODE2FLOAT[_c] = (2.0 ** (-6)) * (_M / 8.0)
    elif _E == 15 and _M == 7:
        _UE4M3_CODE2FLOAT[_c] = np.nan
    else:
        _UE4M3_CODE2FLOAT[_c] = (2.0 ** (_E - 7)) * (1.0 + _M / 8.0)

_UE4M3_VALID = ~np.isnan(_UE4M3_CODE2FLOAT)
_UE4M3_VALID_CODES = np.arange(128, dtype=np.uint8)[_UE4M3_VALID]
_UE4M3_VALID_FLOATS = _UE4M3_CODE2FLOAT[_UE4M3_VALID].astype(np.float64)
_sort = np.argsort(_UE4M3_VALID_FLOATS)
_UE4M3_SORTED_VALS = _UE4M3_VALID_FLOATS[_sort]
_UE4M3_SORTED_CODES = _UE4M3_VALID_CODES[_sort]


def float32_to_ue4m3(x):
    x_f64 = np.abs(np.asarray(x, dtype=np.float64)).ravel()
    x_f64 = np.clip(x_f64, 0, 448.0)
    idx = np.searchsorted(_UE4M3_SORTED_VALS, x_f64)
    idx = np.clip(idx, 0, len(_UE4M3_SORTED_VALS) - 1)
    left = np.clip(idx - 1, 0, len(_UE4M3_SORTED_VALS) - 1)
    use_left = np.abs(x_f64 - _UE4M3_SORTED_VALS[left]) <= np.abs(
        x_f64 - _UE4M3_SORTED_VALS[idx]
    )
    best = np.where(use_left, left, idx)
    return _UE4M3_SORTED_CODES[best].reshape(np.asarray(x).shape).astype(np.uint8)


def ue4m3_to_float32(scale_ue4m3):
    if isinstance(scale_ue4m3, torch.Tensor):
        codes = scale_ue4m3.cpu().numpy()
    else:
        codes = np.asarray(scale_ue4m3)
    codes = codes.astype(np.int32) & 0x7F
    result = _UE4M3_CODE2FLOAT[codes].astype(np.float32)
    return torch.tensor(result)


def per_token_cast_to_fp4(x, block_size=16):
    assert x.dtype == torch.bfloat16
    m, k = x.shape
    assert k % block_size == 0
    assert k % 2 == 0

    x_f32 = x.to(torch.float32).cpu().numpy()
    num_blocks = k // block_size
    x_blocked = x_f32.reshape(m, num_blocks, block_size)

    amax = np.abs(x_blocked).max(axis=-1)
    amax = np.clip(amax, 1e-12, None)

    scale = amax / FP4_E2M1_MAX
    scale_ue4m3 = float32_to_ue4m3(scale)
    scale_decoded = ue4m3_to_float32(scale_ue4m3).cpu().numpy()

    x_scaled = x_blocked / scale_decoded[..., np.newaxis]
    x_scaled = np.clip(x_scaled, -FP4_E2M1_MAX, FP4_E2M1_MAX)
    fp4_rounded = float_to_fp4_e2m1(x_scaled)
    fp4_dequant = (fp4_rounded * scale_decoded[..., np.newaxis]).reshape(m, k)

    packed = pack_fp4_to_uint8(fp4_rounded.reshape(m, k))

    return (
        torch.tensor(packed, dtype=torch.uint8),
        torch.tensor(scale_ue4m3, dtype=torch.uint8),
        torch.tensor(fp4_dequant, dtype=torch.float32),
    )


def per_block_cast_to_fp4(x, block_size=16):
    """Quantize a bfloat16 tensor to NVFP4 in n×k 2D blocks.

    Accepts either a 2D (n, k) tensor (single expert) or a 3D (b, n, k) tensor
    (batched / grouped experts); returns (packed_uint8, scale_ue4m3,
    fp4_dequant_float32) with the packed/dequant tensors matching the input rank.
    """
    assert x.dtype == torch.bfloat16

    if x.dim() == 3:
        b, n, k = x.shape
        assert n % block_size == 0
        assert k % block_size == 0
        assert k % 2 == 0

        x_f32 = x.to(torch.float32).cpu().numpy()
        n_blocks = n // block_size
        k_blocks = k // block_size

        x_blocked = x_f32.reshape(b, n_blocks, block_size, k_blocks, block_size)
        amax = np.abs(x_blocked).max(axis=(2, 4))
        amax = np.clip(amax, 1e-12, None)
        scale = amax / FP4_E2M1_MAX
        scale_ue4m3_compact = float32_to_ue4m3(scale)
        scale_decoded = ue4m3_to_float32(scale_ue4m3_compact).cpu().numpy()

        inv_scale = 1.0 / scale_decoded[:, :, np.newaxis, :, np.newaxis]
        x_scaled = np.clip(x_blocked * inv_scale, -FP4_E2M1_MAX, FP4_E2M1_MAX)

        flat = x_scaled.ravel()
        sign_neg = flat < 0
        abs_flat = np.abs(flat)
        idx = _fp4_abs_to_idx(abs_flat)

        dequant_flat = np.where(sign_neg, -FP4_E2M1_VALUES[idx], FP4_E2M1_VALUES[idx])
        fp4_dequant = (
            dequant_flat.reshape(b, n_blocks, block_size, k_blocks, block_size)
            * scale_decoded[:, :, np.newaxis, :, np.newaxis]
        ).reshape(b, n, k)

        sign_bits = sign_neg.astype(np.uint8) << 3
        encoded = (sign_bits | idx).reshape(b, n, k)
        lo = encoded[..., 0::2]
        hi = encoded[..., 1::2]
        packed = (hi << 4) | lo

        scale_ue4m3 = np.repeat(scale_ue4m3_compact, block_size, axis=1)

        return (
            torch.tensor(packed, dtype=torch.uint8),
            torch.tensor(scale_ue4m3, dtype=torch.uint8),
            torch.tensor(fp4_dequant, dtype=torch.float32),
        )

    n, k = x.shape
    assert n % block_size == 0
    assert k % block_size == 0
    assert k % 2 == 0

    x_f32 = x.to(torch.float32).cpu().numpy()
    n_blocks = n // block_size
    k_blocks = k // block_size

    x_blocked = x_f32.reshape(n_blocks, block_size, k_blocks, block_size)
    amax = np.abs(x_blocked).max(axis=(1, 3))
    amax = np.clip(amax, 1e-12, None)
    scale = amax / FP4_E2M1_MAX
    scale_ue4m3_compact = float32_to_ue4m3(scale)
    scale_decoded = ue4m3_to_float32(scale_ue4m3_compact).cpu().numpy()

    inv_scale = 1.0 / scale_decoded[:, np.newaxis, :, np.newaxis]
    x_scaled = np.clip(x_blocked * inv_scale, -FP4_E2M1_MAX, FP4_E2M1_MAX)

    flat = x_scaled.ravel()
    sign_neg = flat < 0
    abs_flat = np.abs(flat)
    idx = _fp4_abs_to_idx(abs_flat)

    dequant_flat = np.where(sign_neg, -FP4_E2M1_VALUES[idx], FP4_E2M1_VALUES[idx])
    fp4_dequant = (
        dequant_flat.reshape(n_blocks, block_size, k_blocks, block_size)
        * scale_decoded[:, np.newaxis, :, np.newaxis]
    ).reshape(n, k)

    sign_bits = sign_neg.astype(np.uint8) << 3
    encoded = (sign_bits | idx).reshape(n, k)
    lo = encoded[..., 0::2]
    hi = encoded[..., 1::2]
    packed = (hi << 4) | lo

    scale_ue4m3 = np.repeat(scale_ue4m3_compact, block_size, axis=0)

    return (
        torch.tensor(packed, dtype=torch.uint8),
        torch.tensor(scale_ue4m3, dtype=torch.uint8),
        torch.tensor(fp4_dequant, dtype=torch.float32),
    )


# =====================================================================
# CUTLASS SfAtom interleaved reorder
# =====================================================================


def reorder_sf_to_cutlass(sf_logical, mn_dim, kb_dim):
    """Reorder scale factors to CUTLASS SfAtom interleaved layout.

    CUTLASS SfAtom has Blk_MN=128, so the physical layout is always 128-wide
    per atom tile.  When mn_dim < 128, we pad to 128 before reordering and
    keep the padded result — the kernel's TMA descriptor expects the full
    128-wide physical array.
    """
    sf_np = sf_logical.cpu().numpy()
    orig_shape = sf_np.shape

    mn_dim_padded = ((mn_dim + 127) // 128) * 128
    needs_pad = mn_dim_padded > mn_dim

    if needs_pad:
        flat = sf_np.reshape(-1, mn_dim, kb_dim)
        padded = np.zeros((flat.shape[0], mn_dim_padded, kb_dim), dtype=sf_np.dtype)
        padded[:, :mn_dim, :] = flat
        flat = padded
    else:
        flat = sf_np.reshape(-1, mn_dim_padded, kb_dim)

    n_idx = np.arange(mn_dim_padded)
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

    if needs_pad:
        return torch.tensor(result, dtype=torch.uint8)
    else:
        return torch.tensor(result.reshape(orig_shape), dtype=torch.uint8)


# =====================================================================
# Shared test assertion helper
# =====================================================================


def _expect_runtime_error(fn, expected):
    try:
        fn()
    except (RuntimeError, AssertionError) as e:
        msg = str(e)
        assert expected in msg, f"expected '{expected}' in RuntimeError, got: {msg}"
    else:
        raise AssertionError(f"expected RuntimeError containing: {expected}")
