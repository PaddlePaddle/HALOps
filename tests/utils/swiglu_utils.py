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

"""Shared references and checkers for the fused SwiGLU + FP8/FP4 quant tests."""

import numpy as np
import torch

from halops import fused_swiglu_quant
from tests.utils.fp4_utils import (
    FP4_E2M1_MAX,
    float32_to_ue4m3_rne,
    fp4_abs_to_idx_round_half_up,
)


# ============================================================================
# FP8 reference: group-32, UE8M0 scale, SfAtom layout
# ============================================================================


def swiglu_and_quant_group32_ref(x, token_nums_per_expert, group_num, group_size):
    """Reference for group-32 FP8 quantization with SfAtom scale.

    Computes SwiGLU activation, per-group-32 FP8 E4M3FN quantization with
    UE8M0 scales, then writes the scale bytes into the same SfAtom layout
    as the fused CUDA kernel.
    """
    hidden = x.shape[-1] // 2
    num_scale_groups = hidden // 128  # each group-128 still occupies 1 uint32 slot
    n_scale = (num_scale_groups + 3) // 4
    hs4_sfatom = n_scale * 4
    kb_dim = n_scale * 16
    total_tokens = group_num * group_size

    # --- SwiGLU in float32 ---
    x_f = x.to(torch.float32).cpu().numpy()
    x1 = x_f[:, :hidden]  # gate
    x2 = x_f[:, hidden:]  # up
    y = x2 * x1 / (1.0 + np.exp(-x1))

    # --- Group-32 FP8 quantization ---
    num_g32 = hidden // 32

    # amax per group-32
    y_reshaped = y.reshape(total_tokens, num_g32, 32)
    amax = np.max(np.abs(y_reshaped), axis=2)  # [total_tokens, num_g32]
    amax = np.maximum(amax, 1e-10)  # kEpsilon

    # UE8M0 exponent via IEEE-754 bit manipulation (matches kernel)
    fp8_max = 448.0
    raw_scale = np.maximum(amax / fp8_max, np.finfo(np.float32).tiny)
    raw_bits = raw_scale.view(np.uint32)
    biased_exp = ((raw_bits >> 23) & 0xFF).astype(np.int32)
    has_frac = ((raw_bits & 0x007FFFFF) != 0).astype(np.int32)
    ue_exp = np.minimum(biased_exp + has_frac, 254).astype(np.uint8)

    # Inverse scale per group-32
    inv_exp = 254 - ue_exp.astype(np.int32)
    inv_scale = np.left_shift(inv_exp, 23).view(np.float32)
    inv_scale_expanded = np.repeat(inv_scale, 32, axis=1)  # [total_tokens, hidden]

    # Quantize to FP8 E4M3FN
    y_scaled = y * inv_scale_expanded
    ref_fp8 = torch.tensor(y_scaled, dtype=torch.float32).to(torch.float8_e4m3fn)

    # --- Build SfAtom scale tensor ---
    token_nums_np = (
        token_nums_per_expert.cpu().numpy().astype(np.int64)
        if isinstance(token_nums_per_expert, torch.Tensor)
        else np.array(token_nums_per_expert, dtype=np.int64)
    )

    # Pack 4 group-32 exps per group-128
    ue_exp_128 = ue_exp.reshape(total_tokens, num_scale_groups, 4)

    ref_scale_flat = np.zeros(total_tokens * kb_dim, dtype=np.uint8)

    # Vectorized: build all valid (expert, token_in_expert) pairs
    expert_ids = np.repeat(np.arange(group_num, dtype=np.int64), token_nums_np)
    token_ids = np.concatenate([np.arange(n, dtype=np.int64) for n in token_nums_np])
    if token_ids.size > 0:
        expert_base = expert_ids * (group_size * kb_dim)
        n_tile = token_ids // 128
        pos = token_ids % 128
        lane_sfatom = 4 * (pos % 32) + pos // 32
        token_base = expert_base + n_tile * (hs4_sfatom * 512) + lane_sfatom * 4

        valid_row = expert_ids * group_size + token_ids
        valid_ue = ue_exp_128[valid_row]

        scale_offsets = np.arange(num_scale_groups, dtype=np.int64) * 512
        all_bases = token_base[:, None] + scale_offsets[None, :]
        byte_offsets = np.arange(4, dtype=np.int64)
        all_addrs = all_bases[:, :, None] + byte_offsets[None, None, :]

        ref_scale_flat[all_addrs.ravel()] = valid_ue.ravel()

    ref_scale = torch.tensor(
        ref_scale_flat.reshape(total_tokens, kb_dim), dtype=torch.uint8
    )
    return ref_fp8, ref_scale


# ============================================================================
# FP4 reference: group-16, UE4M3 scale, SfAtom layout
# ============================================================================


def _ue4m3_to_float32(scale_ue4m3):
    """Decode UE4M3 (uint8) bit pattern to float32 scale value."""
    codes = np.asarray(scale_ue4m3, dtype=np.uint8)
    fp8_tensor = torch.tensor(codes, dtype=torch.uint8).view(torch.float8_e4m3fn)
    return fp8_tensor.to(torch.float32).cpu().numpy()


def swiglu_and_quant_group16_ref(x, token_nums_per_expert, group_num, group_size):
    """NumPy/Paddle reference for group-16 FP4 quantization with SfAtom scale.

    Computes SwiGLU activation, per-group-16 FP4 E2M1 quantization with
    UE4M3 scales, then writes the scale bytes into the same SfAtom layout
    as the fused CUDA kernel.
    """
    hidden = x.shape[-1] // 2
    total_tokens = group_num * group_size
    kb_dim = hidden // 16  # FP4: one scale byte per group-16, no extra interleaving

    # --- SwiGLU in float32 ---
    x_f = x.to(torch.float32).cpu().numpy()
    x1 = x_f[:, :hidden]  # gate
    x2 = x_f[:, hidden:]  # up
    y = x2 * x1 / (1.0 + np.exp(-x1))

    # --- Group-16 FP4 quantization ---
    quant_group_size = 16
    num_groups = hidden // quant_group_size
    y_reshaped = y.reshape(total_tokens, num_groups, quant_group_size)

    # amax per group-16
    amax = np.max(np.abs(y_reshaped), axis=2)  # [total_tokens, num_groups]
    amax = np.maximum(amax, 1e-10)  # kEpsilon

    # UE4M3 scale encoding
    scale = amax / FP4_E2M1_MAX  # [total_tokens, num_groups]
    scale_byte = float32_to_ue4m3_rne(scale)
    # When scale is too small, UE4M3 may round to 0; use minimum positive value
    zero_mask = (scale_byte == 0) & (amax > 0.0)
    scale_byte[zero_mask] = 0x01
    scale_decoded = _ue4m3_to_float32(scale_byte)
    inv_scale = np.float32(1.0) / scale_decoded

    # FP4 quantize
    x_scaled = y_reshaped * inv_scale[..., np.newaxis]
    x_scaled = np.clip(x_scaled, -FP4_E2M1_MAX, FP4_E2M1_MAX)
    flat = x_scaled.reshape(total_tokens, hidden)
    sign_bits = (flat < 0).astype(np.uint8) << 3
    mag_idx = fp4_abs_to_idx_round_half_up(np.abs(flat))
    fp4_codes = sign_bits | mag_idx  # [total_tokens, hidden]

    # Pack: even index = low nibble, odd index = high nibble
    lo = fp4_codes[:, 0::2]
    hi = fp4_codes[:, 1::2]
    packed = (hi << 4) | lo  # [total_tokens, hidden // 2]
    ref_packed = torch.tensor(packed, dtype=torch.uint8)

    # --- Build SfAtom scale tensor ---
    token_nums_np = (
        token_nums_per_expert.cpu().numpy().astype(np.int64)
        if isinstance(token_nums_per_expert, torch.Tensor)
        else np.array(token_nums_per_expert, dtype=np.int64)
    )

    ref_scale_flat = np.zeros(total_tokens * kb_dim, dtype=np.uint8)

    expert_ids = np.repeat(np.arange(group_num, dtype=np.int64), token_nums_np)
    token_ids = np.concatenate([np.arange(n, dtype=np.int64) for n in token_nums_np])
    if token_ids.size > 0:
        num_k_tiles = kb_dim // 4

        # SfAtom address computation (same as kernel FP4 path)
        expert_base = expert_ids * (group_size * kb_dim)
        n_tile = token_ids // 128
        pos = token_ids % 128
        mn_i = pos % 32
        mn_j = pos // 32
        sfatom_mn_base = (
            expert_base + n_tile * (num_k_tiles * 512) + mn_i * 16 + mn_j * 4
        )

        valid_row = expert_ids * group_size + token_ids
        valid_scale = scale_byte[valid_row]  # [T, num_groups]

        # Each group-16 → one scale byte at k_tile = g/4, sf_l = g%4
        g_idx = np.arange(num_groups, dtype=np.int64)
        k_tile = g_idx // 4
        sf_l = g_idx % 4
        scale_offsets = k_tile * 512 + sf_l  # [num_groups]

        # All byte addresses: [T, num_groups]
        all_addrs = sfatom_mn_base[:, None] + scale_offsets[None, :]
        ref_scale_flat[all_addrs.ravel()] = valid_scale.ravel()

    ref_scale = torch.tensor(
        ref_scale_flat.reshape(total_tokens, kb_dim), dtype=torch.uint8
    )
    return ref_packed, ref_scale


# ============================================================================
# Fused kernel wrapper
# ============================================================================


def swiglu_and_quant_fused(x, token_nums_per_expert, is_fp4=False):
    """Fused: single kernel launch, 2D input/output."""
    act_out_quant, act_out_scale = fused_swiglu_quant(x, token_nums_per_expert, is_fp4)
    return act_out_quant, act_out_scale


# ============================================================================
# SfAtom valid byte mask helpers
# ============================================================================


def get_sfatom_valid_byte_mask_fp8(token_nums_np, group_size, kb_dim, hidden):
    """Build a flat boolean mask over the SfAtom scale tensor bytes (FP8 path).

    For each valid token, marks the 4-byte positions it writes as True.
    """
    group_num = len(token_nums_np)
    total_bytes = group_num * group_size * kb_dim
    mask = np.zeros(total_bytes, dtype=bool)

    num_scale_groups = hidden // 128
    n_scale = (num_scale_groups + 3) // 4
    hs4_sfatom = n_scale * 4

    expert_ids = np.repeat(np.arange(group_num, dtype=np.int64), token_nums_np)
    token_ids = np.concatenate([np.arange(n, dtype=np.int64) for n in token_nums_np])
    if token_ids.size == 0:
        return mask

    expert_base = expert_ids * group_size * kb_dim
    n_tile = token_ids // 128
    pos = token_ids % 128
    lane_sfatom = 4 * (pos % 32) + pos // 32
    token_base = expert_base + n_tile * (hs4_sfatom * 512) + lane_sfatom * 4

    scale_offsets = np.arange(num_scale_groups, dtype=np.int64) * 512
    all_bases = token_base[:, None] + scale_offsets[None, :]

    byte_offsets = np.arange(4, dtype=np.int64)
    all_addrs = all_bases[:, :, None] + byte_offsets[None, None, :]

    mask[all_addrs.ravel()] = True
    return mask


def get_sfatom_valid_byte_mask_fp4(token_nums_np, group_size, kb_dim, hidden):
    """Build a flat boolean mask over the SfAtom scale tensor bytes (FP4 path).

    For each valid token, marks the 1-byte positions it writes per group-16.
    """
    group_num = len(token_nums_np)
    total_bytes = group_num * group_size * kb_dim
    mask = np.zeros(total_bytes, dtype=bool)

    num_groups = hidden // 16
    num_k_tiles = kb_dim // 4

    expert_ids = np.repeat(np.arange(group_num, dtype=np.int64), token_nums_np)
    token_ids = np.concatenate([np.arange(n, dtype=np.int64) for n in token_nums_np])
    if token_ids.size == 0:
        return mask

    expert_base = expert_ids * group_size * kb_dim
    n_tile = token_ids // 128
    pos = token_ids % 128
    mn_i = pos % 32
    mn_j = pos // 32
    sfatom_mn_base = expert_base + n_tile * (num_k_tiles * 512) + mn_i * 16 + mn_j * 4

    g_idx = np.arange(num_groups, dtype=np.int64)
    k_tile = g_idx // 4
    sf_l = g_idx % 4
    scale_offsets = k_tile * 512 + sf_l  # [num_groups]

    all_addrs = sfatom_mn_base[:, None] + scale_offsets[None, :]
    mask[all_addrs.ravel()] = True
    return mask


def get_valid_row_mask(token_nums_np, group_size):
    """Build a boolean mask [E*M] that is True for valid token rows."""
    group_num = len(token_nums_np)
    mask = np.zeros(group_num * group_size, dtype=bool)
    for e in range(group_num):
        start = e * group_size
        end = start + token_nums_np[e]
        mask[start:end] = True
    return mask


# ============================================================================
# Verification (assert-based)
# ============================================================================


def verify_outputs_fp8(
    ref_fp8,
    fused_fp8,
    ref_scale,
    fused_scale,
    valid_mask,
    sfatom_byte_mask,
    kb_dim,
):
    """Assert FP8 and scale outputs match on valid rows only.

    Numerical tolerances (scale within +/-1 UE8M0 exponent; fp8 within 1 ULP,
    ~2x, or subnormal/zero) are intentional and accepted silently. Anything
    outside tolerance raises AssertionError with diagnostics.
    """
    # ---------- scale check ----------
    ref_s_flat = ref_scale.reshape([-1]).cpu().numpy().astype(np.int16)
    fused_s_flat = fused_scale.reshape([-1]).cpu().numpy().astype(np.int16)
    ref_s_valid = ref_s_flat[sfatom_byte_mask]
    fused_s_valid = fused_s_flat[sfatom_byte_mask]

    scale_diff = np.abs(ref_s_valid - fused_s_valid)
    scale_max_diff = int(scale_diff.max()) if scale_diff.size > 0 else 0
    bad_scale_idx = np.where(scale_diff > 1)[0]
    assert scale_max_diff <= 1, (
        f"[scale] {len(bad_scale_idx)} bytes differ by more than 1 "
        f"(max_diff={scale_max_diff}): "
        + ", ".join(
            f"byte[{idx}]: ref={ref_s_valid[idx]}, fused={fused_s_valid[idx]}"
            for idx in bad_scale_idx[:3]
        )
    )

    # ---------- fp8 check ----------
    ref_f = ref_fp8.to(torch.float32).cpu().numpy()
    fused_f = fused_fp8.to(torch.float32).cpu().numpy()
    ref_f_valid = ref_f[valid_mask]
    fused_f_valid = fused_f[valid_mask]

    if np.array_equal(ref_f_valid, fused_f_valid):
        return

    fp8_diff = np.abs(ref_f_valid - fused_f_valid)
    fp8_mismatch = fp8_diff > 0
    ref_m = ref_f_valid[fp8_mismatch]
    fused_m = fused_f_valid[fp8_mismatch]
    abs_diff = np.abs(ref_m - fused_m)
    max_mag = np.maximum(np.abs(ref_m), np.abs(fused_m))

    FP8_ULP_REL = 0.125
    FP8_ULP_ABS = 0.001953125
    FP8_SUBNORM = 0.0625

    within_1_ulp = abs_diff <= np.maximum(max_mag * FP8_ULP_REL, FP8_ULP_ABS)
    approx_2x = (
        np.abs(ref_m - 2.0 * fused_m)
        <= np.maximum(np.abs(fused_m) * FP8_ULP_REL * 2, FP8_ULP_ABS)
    ) | (
        np.abs(fused_m - 2.0 * ref_m)
        <= np.maximum(np.abs(ref_m) * FP8_ULP_REL * 2, FP8_ULP_ABS)
    )
    small = (np.abs(ref_m) < FP8_SUBNORM) | (np.abs(fused_m) < FP8_SUBNORM)
    zero = (ref_m == 0) | (fused_m == 0)
    ok_per_val = within_1_ulp | approx_2x | small | zero

    bad = ~ok_per_val
    bad_idx = np.where(fp8_mismatch)
    bad_r = bad_idx[0][bad]
    bad_c = bad_idx[1][bad]
    assert np.all(
        ok_per_val
    ), f"[fp8] {np.count_nonzero(bad)} genuine mismatches: " + ", ".join(
        f"row[{bad_r[i]}] col[{bad_c[i]}]: "
        f"ref={ref_f_valid[bad_r[i], bad_c[i]]:.6f}, "
        f"fused={fused_f_valid[bad_r[i], bad_c[i]]:.6f}"
        for i in range(min(5, len(bad_r)))
    )


def verify_outputs_fp4(
    ref_packed,
    fused_packed,
    ref_scale,
    fused_scale,
    valid_mask,
    sfatom_byte_mask,
    kb_dim,
):
    """Assert FP4 packed and scale outputs match on valid rows only.

    Tolerances (scale within +/-1 UE4M3 byte; packed bytes <5% mismatch from
    fast-math SwiGLU precision) are intentional and accepted silently.
    Anything outside tolerance raises AssertionError with diagnostics.
    """
    # ---------- scale check ----------
    ref_s_flat = ref_scale.reshape([-1]).cpu().numpy().astype(np.int16)
    fused_s_flat = fused_scale.reshape([-1]).cpu().numpy().astype(np.int16)
    ref_s_valid = ref_s_flat[sfatom_byte_mask]
    fused_s_valid = fused_s_flat[sfatom_byte_mask]

    scale_diff = np.abs(ref_s_valid - fused_s_valid)
    n_diff = np.count_nonzero(scale_diff > 0)
    max_diff = int(scale_diff.max()) if scale_diff.size > 0 else 0
    # Accept ±1 due to fast-math SwiGLU precision differences
    assert max_diff <= 1, f"[scale] {n_diff} bytes differ, max_diff={max_diff}"

    # ---------- packed FP4 check ----------
    ref_p = ref_packed.cpu().numpy()
    fused_p = fused_packed.cpu().numpy()
    ref_p_valid = ref_p[valid_mask]
    fused_p_valid = fused_p[valid_mask]

    diff_count = np.count_nonzero(ref_p_valid != fused_p_valid)
    total = ref_p_valid.size
    pct = diff_count / total * 100 if total > 0 else 0.0
    # Allow small mismatch due to fast-math SwiGLU causing different amax
    # → different scale
    assert pct < 5.0, f"[packed] {diff_count}/{total} bytes differ ({pct:.2f}%)"


# ============================================================================
# Single-case runners
# ============================================================================


def run_single_case_fp8(group_num, group_size, H, token_nums_list=None):
    """Run a single FP8 test case comparing fused vs reference."""
    hidden = H // 2
    hidden_size_scale = hidden // 128
    n_scale = (hidden_size_scale + 3) // 4
    kb_dim = n_scale * 16

    x = torch.randn([group_num * group_size, H]).to(torch.bfloat16)

    if token_nums_list is None:
        token_nums_list = [
            np.random.randint(1, group_size + 1) for _ in range(group_num)
        ]
    token_nums = torch.tensor(token_nums_list, dtype=torch.int32)
    token_nums_np = np.array(token_nums_list, dtype=np.int32)
    valid_mask = get_valid_row_mask(token_nums_np, group_size)
    sfatom_byte_mask = get_sfatom_valid_byte_mask_fp8(
        token_nums_np, group_size, kb_dim, hidden
    )

    ref_fp8, ref_scale = swiglu_and_quant_group32_ref(
        x, token_nums, group_num, group_size
    )
    fused_fp8, fused_scale = swiglu_and_quant_fused(x, token_nums, is_fp4=False)

    verify_outputs_fp8(
        ref_fp8,
        fused_fp8,
        ref_scale,
        fused_scale,
        valid_mask,
        sfatom_byte_mask,
        kb_dim,
    )


def run_single_case_fp4(group_num, group_size, H, token_nums_list=None):
    """Run a single FP4 test case comparing fused vs reference."""
    hidden = H // 2
    kb_dim = hidden // 16

    x = torch.randn([group_num * group_size, H]).to(torch.bfloat16)

    if token_nums_list is None:
        token_nums_list = [
            np.random.randint(1, group_size + 1) for _ in range(group_num)
        ]
    token_nums = torch.tensor(token_nums_list, dtype=torch.int32)
    token_nums_np = np.array(token_nums_list, dtype=np.int32)
    valid_mask = get_valid_row_mask(token_nums_np, group_size)
    sfatom_byte_mask = get_sfatom_valid_byte_mask_fp4(
        token_nums_np, group_size, kb_dim, hidden
    )

    ref_packed, ref_scale = swiglu_and_quant_group16_ref(
        x, token_nums, group_num, group_size
    )
    fused_packed, fused_scale = swiglu_and_quant_fused(x, token_nums, is_fp4=True)

    verify_outputs_fp4(
        ref_packed,
        fused_packed,
        ref_scale,
        fused_scale,
        valid_mask,
        sfatom_byte_mask,
        kb_dim,
    )
