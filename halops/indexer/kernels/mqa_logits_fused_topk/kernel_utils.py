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

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op

# SMEM b32 load/store wrappers — `cute.arch.ld_shared / st_shared` are not
# exposed in the DSL surface, so we wrap raw PTX.


@dsl_user_op
def ld_shared_b32(smem_ptr, *, loc=None, ip=None) -> Int32:
    """Load a 32-bit word from shared memory.

    Args:
        smem_ptr: Shared-memory pointer to the word to read.
        loc: Optional MLIR source location for emitted operations.
        ip: Optional MLIR insertion point for emitted operations.
    """
    # `_Pointer.toint()` returns Int32 (SMEM addr fits in 32 bits).
    smem_addr = smem_ptr.toint(loc=loc, ip=ip)
    return Int32(
        llvm.inline_asm(
            Int32.mlir_type,
            [smem_addr.ir_value(loc=loc, ip=ip)],
            "ld.shared.b32 $0, [$1];",
            "=r,r",
            has_side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def st_shared_b32(smem_ptr, val: Int32, *, loc=None, ip=None) -> None:
    """Store a 32-bit word in shared memory.

    Args:
        smem_ptr: Shared-memory pointer to the destination word.
        val: Int32 value to store.
        loc: Optional MLIR source location for emitted operations.
        ip: Optional MLIR insertion point for emitted operations.
    """
    smem_addr = smem_ptr.toint(loc=loc, ip=ip)
    llvm.inline_asm(
        None,
        [smem_addr.ir_value(loc=loc, ip=ip), Int32(val).ir_value(loc=loc, ip=ip)],
        "st.shared.b32 [$0], $1;",
        "r,r",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )


@cute.jit
def warp_scan(val: cutlass.Int32, lane_id, num_threads_per_warp: cutlass.Constexpr):
    """Compute an inclusive prefix sum across participating warp lanes.

    Args:
        val: Int32 contribution from the calling lane.
        lane_id: Zero-based lane index within the participating prefix of the warp.
        num_threads_per_warp: Power-of-two count of participating lanes, at most 32.
    """
    mask_val = cutlass.const_expr(((1 << num_threads_per_warp) - 1) & 0xFFFFFFFF)
    mask_and_clamp_val = 0
    iteration = cute.arch.log2_of_pow2_int(cutlass.Int32(num_threads_per_warp))
    for i in cutlass.range(iteration, unroll_full=True):
        offset = 1 << i
        other = cute.arch.shuffle_sync_up(
            val, offset, mask=mask_val, mask_and_clamp=mask_and_clamp_val
        )
        if lane_id >= offset:
            val = val + other
    return val


@dsl_user_op
def _topk_red_add(dst_ptr, val, *, loc=None, ip=None):
    """Add to a shared histogram without returning the previous value.

    Args:
        dst_ptr: Shared-memory pointer to the counter updated at CTA scope.
        val: Integer increment applied to the counter.
        loc: Optional MLIR source location for emitted operations.
        ip: Optional MLIR insertion point for emitted operations.
    """
    smem_addr = dst_ptr.toint(loc=loc, ip=ip)
    llvm.inline_asm(
        None,
        [smem_addr.ir_value(loc=loc, ip=ip), val.ir_value(loc=loc, ip=ip)],
        "red.shared.add.u32 [$0], $1;",
        "r,r",
        has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _topk_atomic_add(dst_ptr, val, *, loc=None, ip=None):
    """Add to a shared Int32 counter and return its previous value.

    Args:
        dst_ptr: Shared-memory pointer to the counter updated at CTA scope.
        val: Integer increment applied to the counter.
        loc: Optional MLIR source location for emitted operations.
        ip: Optional MLIR insertion point for emitted operations.
    """
    return cute.arch.atomic_add(
        ptr=dst_ptr.llvm_ptr,
        val=val,
        sem="relaxed",
        scope="cta",
        loc=loc,
        ip=ip,
    )


def _subview(tensor, offset, count):
    """Load a contiguous tensor slice at an element offset.

    Args:
        tensor: Source CuTe tensor.
        offset: Starting offset in elements.
        count: Number of elements to load; no count alignment is required.
    """
    return cute.make_tensor(tensor.iterator + offset, cute.make_layout(count)).load()


@cute.jit
def utccp_required_smem_warp_transpose(smem_ptr) -> None:
    """Reshuffle one 128-int32 scale atom into the FP4 SMEM-to-TMEM layout.

    Args:
        smem_ptr: 512-byte-aligned shared-memory pointer to the scale atom, updated by
            all 32 lanes.

    Notes:
        The XOR permutation avoids shared-memory bank conflicts.
    """
    lane_idx = cute.arch.lane_idx()
    values = cute.make_rmem_tensor((4,), cutlass.Int32)
    for i in cutlass.range_constexpr(4):
        offset = (i ^ (lane_idx >> 3)) * 32 + lane_idx
        values[i] = ld_shared_b32(smem_ptr + offset)
    # Explicit -1 (all-lanes mask): the wrapper default 0xFFFFFFFF triggers a
    # noisy Int32-truncation DSLWarning on every compile; -1 is the same bits.
    cute.arch.sync_warp(-1)
    for i in cutlass.range_constexpr(4):
        offset = lane_idx * 4 + (i ^ (lane_idx >> 3))
        st_shared_b32(smem_ptr + offset, values[i])


@cute.jit
def _topk_prefix_pick_warp0(
    ltid: cutlass.Int32,  # thread id within the syncing group
    want: cutlass.Int32,  # boundary target: first bin with P_incl >= want
    s_hist: cute.Tensor,
    hbase: cutlass.Constexpr,
    s_scalars: cute.Tensor,
    sidx: cutlass.Constexpr,  # scalar slot receiving the picked bin
):
    """Use warp 0 to prefix-sum 256 bins and choose the boundary bin.

    Args:
        ltid: Thread index within the synchronizing group; only indices below 32 do
            work.
        want: One-based target rank for the inclusive histogram prefix.
        s_hist: Shared histogram updated in place with inclusive prefix counts.
        hbase: Element offset of this group's 256 bins in ``s_hist``.
        s_scalars: Shared scalar storage receiving the selected bin.
        sidx: Element index in ``s_scalars`` at which to store the bin.

    Notes:
        The caller must synchronize before publishing the histogram and after the pick.
    """
    if ltid < 32:
        lane = ltid
        seg = cutlass.Int32(0)
        vals = cute.make_rmem_tensor((8,), cutlass.Int32)
        for j in cutlass.range_constexpr(8):
            vals[j] = s_hist[hbase + lane * 8 + j]
            seg = seg + vals[j]
        seg_incl = warp_scan(seg, lane, num_threads_per_warp=32)
        run = seg_incl - seg  # exclusive segment prefix
        for j in cutlass.range_constexpr(8):
            p_prev = run
            run = run + vals[j]
            s_hist[hbase + lane * 8 + j] = run
            if p_prev < want and run >= want:
                s_scalars[sidx] = lane * 8 + j


def _apply_utccp_chunk_layout(sf_per_block: torch.Tensor) -> torch.Tensor:
    """Transpose each 128-token scale atom into the offline UTCCP layout.

    Args:
        sf_per_block: Packed scale tensor with a final dimension of 128, requiring
            ``block_kv=128``.

    Notes:
        The output maps ``out[m * 4 + k] = in[k * 32 + m]`` within each atom and is used
        with ``remove_online_sf_transpose``.
    """
    assert (
        sf_per_block.size(-1) == 128
    ), f"chunk layout requires atom_size=128 (block_kv=128); got {sf_per_block.size(-1)}"
    new_shape = sf_per_block.shape[:-1] + (128,)
    return (
        sf_per_block.reshape(*sf_per_block.shape[:-1], 4, 32)
        .transpose(-1, -2)
        .contiguous()
        .reshape(*new_shape)
    )


@dsl_user_op
def coarse_pair(lo, hi, *, loc=None, ip=None):
    """Encode two FP32 scores as packed descending round-toward-zero FP16 keys.

    Args:
        lo: FP32 score encoded in the low 16 bits.
        hi: FP32 score encoded in the high 16 bits.
        loc: Optional MLIR source location for emitted operations.
        ip: Optional MLIR insertion point for emitted operations.
    """
    # Encode descending RTZ-FP16 keys by reversing positive values' bits.
    return cutlass.Uint32(
        llvm.inline_asm(
            cutlass.Uint32.mlir_type,
            [hi.ir_value(loc=loc, ip=ip), lo.ir_value(loc=loc, ip=ip)],
            "{ .reg .b32 h, s; "
            "cvt.rz.f16x2.f32 h, $1, $2; "
            "prmt.b32 s, h, 0, 0xbb99; "
            "lop3.b32 $0, h, s, 0x7fff7fff, 0xd2; }",
            "=r,f,f",
            has_side_effects=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )
