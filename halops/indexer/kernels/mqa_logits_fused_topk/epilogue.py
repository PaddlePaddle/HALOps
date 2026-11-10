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
# Implemented with reference to the PR of flashinfer (https://github.com/flashinfer-ai/flashinfer/pull/7325)
"""

import inspect

import cutlass
import cutlass.cute as cute
import torch
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, nvvm, vector
from cutlass.cutlass_dsl import dsl_user_op


@dsl_user_op
def _cvt_relu_bf16x2(lo, hi, *, loc=None, ip=None):
    """Convert two FP32 values to packed BF16 with ReLU and round-to-nearest-even.

    Args:
        lo: FP32 input stored in the low 16 bits of the result.
        hi: FP32 input stored in the high 16 bits of the result.
        loc: Optional MLIR source location for emitted operations.
        ip: Optional MLIR insertion point for emitted operations.
    """
    # PTX puts the first input in the high half; reverse inputs to retain lane order.
    return cutlass.Uint32(
        llvm.inline_asm(
            cutlass.Uint32.mlir_type,
            [hi.ir_value(loc=loc, ip=ip), lo.ir_value(loc=loc, ip=ip)],
            "cvt.rn.relu.bf16x2.f32 $0, $1, $2;",
            "=r,f,f",
            has_side_effects=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def relu_bf16x4(values):
    """Convert four finite FP32 accumulators to BF16 and clamp negatives to zero.

    Args:
        values: Four FP32 accumulator values in lane order.
    """
    packed = cute.make_rmem_tensor((2,), cutlass.Uint32)
    for i in cutlass.range_constexpr(2):
        packed[i] = _cvt_relu_bf16x2(values[2 * i], values[2 * i + 1])
    return cute.recast_tensor(packed, cutlass.BFloat16).load()


# Packed FP32 arithmetic.

# 4.6.2: fma_packed_f32x2(res, src_a, src_b, src_c, *, rnd, ftz, ...)
# 4.8.x: fma_packed_f32x2(src_a, src_b, src_c, *, rnd, ftz, results, ...)
_FMA_PACKED_F32X2_PARAMETERS = inspect.signature(nvvm.fma_packed_f32x2).parameters
if "res" in _FMA_PACKED_F32X2_PARAMETERS:
    _FMA_PACKED_F32X2_NEEDS_RESULT_TYPE = True
elif "results" in _FMA_PACKED_F32X2_PARAMETERS:
    _FMA_PACKED_F32X2_NEEDS_RESULT_TYPE = False
else:
    raise RuntimeError(
        "Unsupported CUTLASS DSL fma_packed_f32x2 signature; "
        "expected 4.6.2 or 4.8.x API"
    )


def _check_operands(a, *others):
    """Require matching, even-width FP32 TensorSSA vectors.

    Args:
        a: Reference FP32 TensorSSA defining the flat vector type.
        others: Additional operands that must have the same vector type as ``a``.
    """
    for value in (a, *others):
        if not isinstance(value, cute.TensorSSA) or value.dtype != cutlass.Float32:
            raise TypeError("Packed FP32 operations require Float32 TensorSSA operands")
        # Logical shapes such as 4 and (4,) have the same flat vector type.
        if value.type != a.type:
            raise ValueError("Packed FP32 operands must have matching vector types")
    shape = ir.VectorType(a.type).shape
    if len(shape) != 1 or shape[0] % 2:
        raise ValueError("Packed FP32 operations require an even-width flat vector")


@dsl_user_op
def fma_packed_f32x2(a, b, c, *, loc=None, ip=None):
    """Compute elementwise FP32 fused multiply-add with round-to-nearest-even.

    Args:
        a: Even-width FP32 TensorSSA multiplicand.
        b: FP32 multiplier with the same vector type as ``a``.
        c: FP32 addend with the same vector type as ``a``.
        loc: Optional MLIR source location for emitted operations.
        ip: Optional MLIR insertion point for emitted operations.
    """
    _check_operands(a, b, c)
    args = (a.type, a, b, c) if _FMA_PACKED_F32X2_NEEDS_RESULT_TYPE else (a, b, c)
    result = nvvm.fma_packed_f32x2(*args, rnd=nvvm.FPRoundingMode.RN, loc=loc, ip=ip)
    return cute.TensorSSA(result, a.shape, a.dtype, loc=loc, ip=ip)


# Packed BF16 arithmetic.


def _width(a, *others):
    """Validate matching even-width vectors and return their element count.

    Args:
        a: Reference TensorSSA defining the flat vector type.
        others: Additional TensorSSA operands that must match ``a``.
    """
    for value in others:
        if not isinstance(value, cute.TensorSSA) or value.type != a.type:
            raise TypeError("Packed BF16 operands must have matching vector types")
    shape = ir.VectorType(a.type).shape
    if len(shape) != 1 or shape[0] % 2:
        raise ValueError("Packed BF16 operations require an even-width flat vector")
    return shape[0]


def _pair(value, pair_type, index, *, loc, ip):
    """Extract two consecutive elements from an MLIR vector.

    Args:
        value: Source MLIR vector value.
        pair_type: MLIR vector type for the two-element result.
        index: Starting element index of the extracted pair.
        loc: Optional MLIR source location for emitted operations.
        ip: Optional MLIR insertion point for emitted operations.
    """
    return vector.extract_strided_slice(
        pair_type, value, [index], [2], [1], loc=loc, ip=ip
    )


@dsl_user_op
def fma_packed_bf16x2(a, b, c, *, loc=None, ip=None):
    """Compute elementwise BF16 fused multiply-add with round-to-nearest-even.

    Args:
        a: Even-width BF16 TensorSSA multiplicand, or an operand for the generic FMA
            fallback.
        b: Multiplier matching ``a`` for packed BF16 operations.
        c: Addend matching ``a`` for packed BF16 operations.
        loc: Optional MLIR source location for emitted operations.
        ip: Optional MLIR insertion point for emitted operations.
    """
    if not isinstance(a, cute.TensorSSA) or a.dtype != cutlass.BFloat16:
        return cute.fma(a, b, c, loc=loc, ip=ip)
    pair_type = ir.VectorType.get([2], cutlass.BFloat16.mlir_type, loc=loc)
    i32 = ir.IntegerType.get_signless(32)
    values = []
    for i in range(0, _width(a, b, c), 2):
        operands = [
            llvm.bitcast(
                i32, _pair(value, pair_type, i, loc=loc, ip=ip), loc=loc, ip=ip
            )
            for value in (a, b, c)
        ]
        # fma.rn.bf16x2 is what cute.arch.fma_packed_bf16x2(rnd="rn") emits: one
        # round-to-nearest-even FMA per half, no .ftz and no saturation.
        bits = llvm.inline_asm(
            i32,
            operands,
            "fma.rn.bf16x2 $0, $1, $2, $3;",
            "=r,r,r,r",
            has_side_effects=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
        pair = llvm.bitcast(pair_type, bits, loc=loc, ip=ip)
        values.extend(vector.extract(pair, [], [j], loc=loc, ip=ip) for j in range(2))
    result = vector.from_elements(a.type, values, loc=loc, ip=ip)
    return cute.TensorSSA(result, a.shape, a.dtype, loc=loc, ip=ip)


# Score-dtype policy.


class _ScoreEpilogue:
    """Common Q-side preparation and accumulator initialization."""

    def zero(self):
        """Create four zero accumulators in the selected score dtype."""
        return cute.full((4,), 0.0, dtype=self.dtype)

    def q_side_fakes(self, num_heads, sym_nq):
        """Describe flattened Q scales and weights with the live symbolic extent.

        Args:
            num_heads: Logical query head count: 8, 16, 32, or 64.
            sym_nq: Symbolic number of query rows used to preserve partial-tile bounds.
        """
        # Preserve the real extent so TMA zero-fills a partial final query tile.
        shape = (sym_nq * num_heads,)
        q_sf = cute.runtime.make_fake_compact_tensor(
            cutlass.Int32, shape, stride_order=(0,)
        )
        weights = cute.runtime.make_fake_compact_tensor(
            self.dtype, shape, stride_order=(0,)
        )
        return q_sf, weights

    def q_side_host(self, q_sf, weights):
        """Flatten Q scales and dtype-adjusted head weights for kernel launch.

        Args:
            q_sf: Int32 ``[M, H]`` packed UE8M0 Q scales for MXFP4/MXFP8; an unused
                tensor placeholder for FP8.
            weights: Head weights ``[M, H]`` with softmax scaling folded in (and Q
                scaling for ordinary FP8); FP32 for FP32 scores, cast to BF16 for BF16
                scores.
        """
        return q_sf.reshape(-1), self._prepare_weights(weights).reshape(-1)

    def _prepare_weights(self, weights):
        """Return head weights unchanged for FP32 score arithmetic.

        Args:
            weights: FP32 head-weight tensor.
        """
        return weights


class Fp32ScoreEpilogue(_ScoreEpilogue):
    """FP32 scores: relu via ``x + |x|``, so the doubling folds into 0.5."""

    dtype = cutlass.Float32
    torch_dtype = torch.float32

    def relu(self, x4):
        """Compute twice the ReLU value for the FP32 epilogue.

        Args:
            x4: Four FP32 MMA accumulators.
        """
        # x + |x| == 2 * relu(x): no compare/select, and the 2 is undone below.
        return x4 + abs(x4)

    def fma(self, a4, w4, acc):
        """Accumulate four weighted head contributions using packed FP32 FMA.

        Args:
            a4: Four FP32 activations returned by this policy's ReLU.
            w4: Four FP32 head weights corresponding to ``a4``.
            acc: Four running FP32 score accumulators.
        """
        return fma_packed_f32x2(a4, w4, acc)

    def finish(self, acc, scale):
        """Reduce head accumulators and apply optional per-key scaling.

        Args:
            acc: Four FP32 accumulators whose doubled ReLU is corrected by 0.5.
            scale: FP32 per-key scale for FP8 inputs, or None for MXFP4/MXFP8.
        """
        value = (acc[0] + acc[1] + acc[2] + acc[3]) * cutlass.Float32(0.5)
        if scale is not None:
            value = value * scale
        return cutlass.Float32(value)

    def math_regs(self, three_wg):
        """Return the per-thread register budget for score computation.

        Args:
            three_wg: Whether the CTA uses three math warpgroups instead of two.
        """
        # FP32 keeps 4-byte weights and results in registers, so the two
        # warpgroup tile can afford a bigger budget.
        return 160 if three_wg else 240


class Bf16ScoreEpilogue(_ScoreEpilogue):
    """BF16 scores: ``cvt.rn.relu.bf16x2.f32`` is a true relu, so no 0.5."""

    dtype = cutlass.BFloat16
    torch_dtype = torch.bfloat16

    def relu(self, x4):
        """Convert four FP32 MMA accumulators to BF16 with ReLU.

        Args:
            x4: Four FP32 MMA accumulators.
        """
        return relu_bf16x4(x4)

    def fma(self, a4, w4, acc):
        """Accumulate four weighted head contributions using packed BF16 FMA.

        Args:
            a4: Four BF16 activations returned by this policy's ReLU.
            w4: Four BF16 head weights corresponding to ``a4``.
            acc: Four running BF16 score accumulators.
        """
        return fma_packed_bf16x2(a4, w4, acc)

    def finish(self, acc, scale):
        """Reduce head accumulators and apply optional per-key scaling.

        Args:
            acc: Four BF16 accumulators reduced in packed-lane order.
            scale: FP32 per-key scale for FP8 inputs, or None for MXFP4/MXFP8.
        """
        # Lanes 0,2 are the low halves of the two packed registers and 1,3 the
        # high halves, so this pairing is one packed add plus a scalar add.
        value = acc[0] + acc[2] + (acc[1] + acc[3])
        if scale is not None:
            return cutlass.BFloat16(cutlass.Float32(value) * scale)
        return cutlass.BFloat16(value)

    def _prepare_weights(self, weights):
        """Cast head weights to the BF16 score dtype.

        Args:
            weights: Head-weight tensor to convert.
        """
        return weights.to(self.torch_dtype)

    def math_regs(self, three_wg):
        """Return the per-thread register budget for score computation.

        Args:
            three_wg: Whether the CTA uses three math warpgroups; BF16 uses 160
                registers in either case.
        """
        return 160


def score_epilogue(epi_dtype):
    """Return the arithmetic and input-preparation policy for the score dtype.

    Args:
        epi_dtype: Score arithmetic type, ``cutlass.Float32`` or ``cutlass.BFloat16``.
    """
    if epi_dtype == cutlass.Float32:
        return Fp32ScoreEpilogue()
    if epi_dtype == cutlass.BFloat16:
        return Bf16ScoreEpilogue()
    raise ValueError(f"Fused scores must be FP32 or BF16; got {epi_dtype}")
