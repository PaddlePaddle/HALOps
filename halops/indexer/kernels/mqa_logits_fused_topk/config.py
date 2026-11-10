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

import math
import re
from dataclasses import dataclass
from enum import IntEnum
from importlib.metadata import PackageNotFoundError, version

# The Blackwell blockscaled MMA path supports these logical Q/weight head
# counts.  ``num_heads`` is the logical score width; the SMEM/TMEM allocation
# may still round the physical tile up for alignment.
SUPPORTED_NUM_HEADS = (8, 16, 32, 64)
SUPPORTED_HEAD_DIM = 128
SUPPORTED_QKV_DTYPES = ("fp8", "mxfp4", "mxfp8")
MIN_TOPK, MAX_TOPK = 128, 2048
# All stages use tcgen05 and TMEM. Startup warmup remains the final probe for
# an SM100 minor the kernels have not been traced on.
SUPPORTED_ARCH_MAJOR = 10
MIN_CUTLASS_DSL = (4, 6, 2)


def validate_num_heads(num_heads: int) -> None:
    """Reject an unsupported logical query head count.

    Args:
        num_heads: Logical query head count: 8, 16, 32, or 64.
    """
    if type(num_heads) is not int or num_heads not in SUPPORTED_NUM_HEADS:
        supported = ", ".join(str(value) for value in SUPPORTED_NUM_HEADS)
        raise ValueError(f"num_heads must be one of ({supported}); got {num_heads!r}")


def validate_topk(top_k: int) -> None:
    """Reject a top-k width outside the supported integer range.

    Args:
        top_k: Number of indices selected per query, from 128 through 2048.
    """
    if type(top_k) is not int or not MIN_TOPK <= top_k <= MAX_TOPK:
        raise ValueError(
            f"top_k must be an integer in [{MIN_TOPK}, {MAX_TOPK}]; got {top_k!r}"
        )


def validate_query_shape(shape, qkv_dtype: str) -> None:
    """Validate the logical head count and packed Q width.

    Args:
        shape: Q shape ``[M, H, D]``, with byte width D=128 for FP8/MXFP8 or D=64 for
            MXFP4.
        qkv_dtype: Packed input format, ``"fp8"``, ``"mxfp4"`` or ``"mxfp8"``.
    """
    if qkv_dtype not in SUPPORTED_QKV_DTYPES:
        raise ValueError(f"Unsupported Q/KV dtype: {qkv_dtype}")
    packed_dim = SUPPORTED_HEAD_DIM // 2 if qkv_dtype == "mxfp4" else SUPPORTED_HEAD_DIM
    if len(shape) != 3:
        raise ValueError(f"Q must have shape [M, H, {packed_dim}]; got {tuple(shape)}")
    validate_num_heads(shape[1])
    if shape[2] != packed_dim:
        raise ValueError(
            f"Q must have packed width {packed_dim} for {qkv_dtype}; got {tuple(shape)}"
        )


def is_supported(num_heads: int | None, head_dim: int, top_k: int) -> bool:
    """Return whether the bounded chain supports the requested logical shape.

    Args:
        num_heads: Logical query head count, or None when unknown.
        head_dim: Unpacked query/key head dimension.
        top_k: Requested number of indices per query.
    """
    return (
        num_heads in SUPPORTED_NUM_HEADS
        and head_dim == SUPPORTED_HEAD_DIM
        and MIN_TOPK <= top_k <= MAX_TOPK
    )


def padded_num_heads(num_heads: int) -> int | None:
    """Return the smallest supported head count that fits, or None.

    Args:
        num_heads: Number of logical heads that the padded width must accommodate.
    """
    return next((heads for heads in SUPPORTED_NUM_HEADS if heads >= num_heads), None)


def cutlass_dsl_version() -> tuple[int, ...] | None:
    """Read the installed CuTe-DSL version without importing the DSL."""
    try:
        found = version("nvidia-cutlass-dsl")
    except PackageNotFoundError:
        return None
    # Tolerate pre-release suffixes such as "4.6.2a0".
    return tuple(
        int(re.match(r"\d*", part).group() or 0)
        for part in found.split(".")[: len(MIN_CUTLASS_DSL)]
    )


def unsupported_reason(
    num_heads: int | None,
    head_dim: int,
    top_k: int,
    *,
    qkv_dtype: str,
) -> str | None:
    """Return a support failure reason for the current CUDA device, or None.

    Args:
        num_heads: Logical query head count, or None when unknown.
        head_dim: Unpacked query/key head dimension.
        top_k: Requested number of indices per query.
        qkv_dtype: Packed input format, ``"fp8"``, ``"mxfp4"`` or ``"mxfp8"``.

    Notes:
        Use during initialization because this check queries CUDA; per-batch checks use
        host metadata.
    """
    import torch

    arch = torch.cuda.get_device_capability(torch.cuda.current_device())
    if arch[0] != SUPPORTED_ARCH_MAJOR:
        return (
            f"needs tcgen05 + TMEM (sm_{SUPPORTED_ARCH_MAJOR}x), "
            f"found sm_{arch[0]}{arch[1]}"
        )
    dsl = cutlass_dsl_version()
    want = ".".join(str(part) for part in MIN_CUTLASS_DSL)
    if dsl is None:
        return f"needs nvidia-cutlass-dsl >= {want}, which is not installed"
    if dsl < MIN_CUTLASS_DSL:
        found = ".".join(str(part) for part in dsl)
        return f"needs CuTe-DSL >= {want}, found {found}"
    if qkv_dtype not in SUPPORTED_QKV_DTYPES:
        return f"needs FP8, MXFP4 or MXFP8 Q/KV, found {qkv_dtype}"
    if not is_supported(num_heads, head_dim, top_k):
        supported = ", ".join(str(value) for value in SUPPORTED_NUM_HEADS)
        return (
            f"needs H in ({supported}), D={SUPPORTED_HEAD_DIM}, "
            f"{MIN_TOPK}<=top_k<={MAX_TOPK}, found H={num_heads}, "
            f"D={head_dim}, top_k={top_k}"
        )
    return None


def round_up(value: int, multiple: int) -> int:
    """Round a row or element count up to an allocation multiple.

    Args:
        value: Count to round up.
        multiple: Positive allocation alignment in the same units as ``value``.
    """
    return (value + multiple - 1) // multiple * multiple


def base_candidate_capacity(top_k: int) -> int:
    """Return the candidate target used to choose the pilot rank.

    Args:
        top_k: Number of indices selected per query, from 128 through 2048.
    """
    return round_up(max(2 * top_k, top_k + 1280), 256)


def candidate_capacity(top_k: int, qkv_dtype: str) -> int:
    """Return per-row candidate capacity within the input format's SMEM budget.

    Args:
        top_k: Number of indices selected per query, from 128 through 2048.
        qkv_dtype: Packed input format, ``"fp8"``, ``"mxfp4"`` or ``"mxfp8"``.
    """
    return min(
        6144 if qkv_dtype == "mxfp4" else 4352,
        base_candidate_capacity(top_k) + 512,
    )


@dataclass(frozen=True)
class HistogramConfig:
    """Compile-time query tiling, histogram layout and register policy."""

    block_q: int
    hist_splits: int = 1
    three_wg: bool = False
    single_acc: bool = False
    weights: int = 64
    ordered_bounds: bool = False
    pilot_bits: int = 13

    @property
    def math_groups(self) -> int:
        """Return the number of math warpgroups assigned to each CTA."""
        return 3 if self.three_wg else 2


def prefer_two_query_tiles(num_queries: int, num_sms: int) -> bool:
    """Return whether Q2 and Q3 require the same number of CTA waves.

    Args:
        num_queries: Number of live query rows.
        num_sms: Number of SMs available to the persistent launch grid.
    """
    sms = max(1, num_sms)
    q2_waves = (num_queries + 2 * sms - 1) // (2 * sms)
    q3_waves = (num_queries + 3 * sms - 1) // (3 * sms)
    return q2_waves == q3_waves


def warmup_row_multiples(num_heads: int) -> tuple[int, ...]:
    """Return query-row multiples of the SM count covering all tile choices.

    Args:
        num_heads: Logical query head count: 8, 16, 32, or 64.
    """
    if num_heads == 8:
        return (2, 3, 4, 6, 9, 12, 15)
    if num_heads == 16:
        return (2, 3, 4, 6, 9)
    if num_heads == 32:
        return (2, 3, 4, 6)
    return (4, 6)


def select_config(
    num_queries: int,
    num_sms: int,
    *,
    materialize_scores: bool = False,
    ordered_bounds: bool = False,
    num_heads: int = 64,
    repair: bool = False,
) -> HistogramConfig:
    """Choose query tiling and histogram policy for the requested launch.

    Args:
        num_queries: Number of live query rows.
        num_sms: Number of SMs available to the persistent launch grid.
        materialize_scores: Whether to select the full-score diagnostic path, which uses
            a Q3 tile.
        ordered_bounds: Whether end bounds are nondecreasing within every query tile
            after window trimming, including on every graph replay.
        num_heads: Logical query head count: 8, 16, 32, or 64.
        repair: Whether to select the exact-repair path, which retains Q2/Q3 tiles.
    """
    if num_heads in (16, 32) and not repair and not materialize_scores:
        # Relative scan cost per CTA wave, calibrated on FP8/MXFP4 inputs.
        costs = (
            ((2, 10), (3, 12), (4, 17), (6, 20), (9, 29))
            if num_heads == 16
            else ((2, 10), (3, 12), (4, 17))
        )
        sms = max(1, num_sms)
        block_q = min(
            costs,
            key=lambda item: (
                (num_queries + item[0] * sms - 1) // (item[0] * sms) * item[1],
                item[0],
            ),
        )[0]
        return HistogramConfig(
            block_q=block_q,
            three_wg=block_q % 3 == 0,
            pilot_bits=11 if num_heads == 16 else 12,
            ordered_bounds=ordered_bounds,
        )
    if (
        num_heads == 8
        and num_queries > 3 * max(1, num_sms)
        and not repair
        and not materialize_scores
    ):
        # Limit each math group to five queries to bound register pressure.
        sms = max(1, num_sms)
        waves = math.ceil(num_queries / (15 * sms))
        block_q = round_up(math.ceil(num_queries / (waves * sms)), 3)
        return HistogramConfig(
            block_q=block_q,
            three_wg=True,
            pilot_bits=10,
            ordered_bounds=ordered_bounds,
        )
    use_two = not materialize_scores and prefer_two_query_tiles(num_queries, num_sms)
    return HistogramConfig(
        block_q=2 if use_two else 3,
        three_wg=not use_two,
        ordered_bounds=ordered_bounds,
    )


BLOCK_KV_ROWS = 128
SPLIT_KV_ROWS = 2 * BLOCK_KV_ROWS
DEFAULT_KV_SAMPLE = (64, 4)


@dataclass(frozen=True)
class SamplingConfig:
    """Describe pilot sampling and the storage extent it requires.

    ``group`` keys form one sampling group and ``take`` consecutive keys from
    each group are scored by the pilot. Direct sampling reads the original KV;
    gather sampling allocates a dense buffer shared with the memory gate.
    """

    group: int = DEFAULT_KV_SAMPLE[0]
    take: int = DEFAULT_KV_SAMPLE[1]

    def __post_init__(self) -> None:
        """Validate that the sampling pair divides into supported 128-row tiles."""
        if (
            type(self.group) is not int
            or type(self.take) is not int
            or self.take < 1
            or self.group < self.take
            or self.group % self.take
        ):
            raise ValueError(
                "kv_sample must be integer (group, take) with 1 <= take <= group and "
                f"group % take == 0; got ({self.group}, {self.take})"
            )
        if self.take > BLOCK_KV_ROWS or BLOCK_KV_ROWS % self.take:
            raise ValueError(
                "kv_sample take must divide the 128-row pilot tile: "
                "use 1 or 2 for gather, or 4, 8, 16, 32, 64, 128 for direct TMA; "
                f"got ({self.group}, {self.take})"
            )

    @classmethod
    def from_string(cls, value: str) -> "SamplingConfig":
        """Parse the serving environment's ``group,take`` sampling pair.

        Args:
            value: Two comma-separated integers, for example ``"64,4"``.
        """
        try:
            group, take = value.split(",")
            return cls(int(group), int(take))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"kv_sample must be 'group,take' (for example '64,4'); got {value!r}"
            ) from exc

    @property
    def stride(self) -> int:
        """Return the number of full KV rows represented by one sampled row."""
        return self.group // self.take

    @property
    def uses_direct_sampling(self) -> bool:
        """Return whether each sampled scale run meets TMA's 16-byte minimum."""
        return self.take * 4 >= 16

    def sampled_rows(self, num_kv_rows: int) -> int:
        """Return the number of dense rows emitted by the gather sampler.

        Args:
            num_kv_rows: Nonnegative number of rows in the full KV buffer.
        """
        if num_kv_rows < 0:
            raise ValueError("num_kv_rows must be non-negative")
        return num_kv_rows // self.group * self.take + min(
            num_kv_rows % self.group, self.take
        )

    def padded_sampled_rows(self, num_kv_rows: int) -> int:
        """Return the gather sample size rounded up to 256 rows.

        Args:
            num_kv_rows: Nonnegative number of rows in the full KV buffer.
        """
        return round_up(self.sampled_rows(num_kv_rows), SPLIT_KV_ROWS)

    def direct_sampled_rows(self, num_kv_rows: int) -> int:
        """Return the safe sampled-row extent for a direct TMA descriptor.

        Args:
            num_kv_rows: Number of source KV rows containing the sampled runs.

        Notes:
            Only complete 128-row sampled tiles are declared; omitted tail samples can
            only lower the cutoff and retain more candidates.
        """
        if num_kv_rows < self.take:
            return 0
        runs = (num_kv_rows - self.take) // self.group + 1
        return runs // (BLOCK_KV_ROWS // self.take) * BLOCK_KV_ROWS

    def has_pilot_sample(self, num_kv_rows: int) -> bool:
        """Return whether the pilot has usable rows for threshold estimation.

        Args:
            num_kv_rows: Number of rows in the full KV buffer.

        Notes:
            An empty sample remains valid for the public operator through exact repair.
        """
        if self.uses_direct_sampling:
            return self.direct_sampled_rows(num_kv_rows) > 0
        return self.sampled_rows(num_kv_rows) > 0

    def get_sampling_rank(self, top_k: int, *, num_heads: int = 64) -> int:
        """Return the pilot rank with head-specific candidate overflow headroom.

        Args:
            top_k: Number of indices selected per query, from 128 through 2048.
            num_heads: Logical query head count: 8, 16, 32, or 64.
        """
        # Coarser low-head histograms need more candidate overflow headroom.
        numerator, denominator = (11, 24) if num_heads in (8, 16, 32) else (1, 2)
        return math.ceil(
            (top_k + base_candidate_capacity(top_k))
            * numerator
            / (denominator * self.stride)
        )


def bounded_workspace_bytes(
    num_rows: int,
    num_kv_rows: int,
    top_k: int,
    *,
    qkv_dtype: str,
    kv_sample: tuple[int, int] = DEFAULT_KV_SAMPLE,
) -> int:
    """Estimate bounded-chain workspace bytes, excluding transient casts.

    Args:
        num_rows: Number of live query rows.
        num_kv_rows: Number of full KV rows used to size any gathered sample.
        top_k: Number of indices selected per query, from 128 through 2048.
        qkv_dtype: Packed input format, ``"fp8"``, ``"mxfp4"`` or ``"mxfp8"``.
        kv_sample: Pilot ``(group, take)`` pair: score the first ``take`` consecutive
            keys of each ``group`` keys.

    Notes:
        The estimate includes candidates, output indices, 16 metadata bytes per query,
        and gathered KV/scales when direct sampling is unavailable.
    """
    cap = candidate_capacity(top_k, qkv_dtype)
    total = num_rows * (8 * cap + 4 * top_k + 16)
    sampling = SamplingConfig(*kv_sample)
    if not sampling.uses_direct_sampling:
        k_row_bytes = (
            SUPPORTED_HEAD_DIM // 2 if qkv_dtype == "mxfp4" else SUPPORTED_HEAD_DIM
        )
        total += sampling.padded_sampled_rows(num_kv_rows) * (k_row_bytes + 4)
    return total


def needs_repair(marker, window_length, top_k, candidate_capacity):
    """Return whether a filtered row needs an exact repair pass.

    Args:
        marker: Retained-candidate total, or ``-total - 1`` when a warp partition
            overflowed.
        window_length: Number of legal keys in the row's half-open ``[ks, ke)`` window.
        top_k: Number of selected indices required for a full output row.
        candidate_capacity: Maximum number of candidates stored for one row.

    Notes:
        A row retaining every legal key is complete even when shorter than top-k;
        bitwise operators support both Python and CuTe scalars.
    """
    incomplete = marker != window_length
    return (marker > candidate_capacity) | ((marker < top_k) & incomplete)


class LaunchArg(IntEnum):
    """Named positions in filter and repair launch argument lists."""

    KV = 0
    KV_SCALES = 1
    QUERY = 2
    QUERY_SCALES = 3
    HEAD_WEIGHTS = 4
    LOGITS = 5
    TOPK_INDICES = 6
    CANDIDATES = 7
    REPAIR_SCRATCH = 8
    CANDIDATE_COUNTS = 9
    KEY_START = 10
    KEY_END = 11
    NUM_QUERY_ROWS = 12
    NUM_KV_ROWS = 13
    TOPK_INDICES_OFFSET = 14


class PilotLaunchArg(IntEnum):
    """Pilot inputs and threshold output; no candidate or repair workspace."""

    KV = 0
    KV_SCALES = 1
    QUERY = 2
    QUERY_SCALES = 3
    HEAD_WEIGHTS = 4
    THRESHOLD = 5
    KEY_START = 6
    KEY_END = 7
    NUM_QUERY_ROWS = 8
    NUM_KV_ROWS = 9


PRODUCER_ARGUMENT_COUNT = LaunchArg.TOPK_INDICES_OFFSET + 1
PILOT_ARGUMENT_COUNT = PilotLaunchArg.NUM_KV_ROWS + 1


def validate_argument_count(args, expected: int, stage: str) -> None:
    """Reject a launch argument list that does not match its stage ABI.

    Args:
        args: Positional arguments prepared for the compiled kernel.
        expected: Required number of positional arguments.
        stage: Stage name included in an ABI mismatch error.
    """
    if len(args) != expected:
        raise RuntimeError(
            f"{stage} launch ABI has {len(args)} arguments; expected {expected}"
        )
