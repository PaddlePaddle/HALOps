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

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path


class ActivityKind(IntEnum):
    DRIVER = 4
    RUNTIME = 5
    CONCURRENT_KERNEL = 10


@dataclass(frozen=True)
class Activity:
    kind: ActivityKind | int
    start: int
    end: int
    correlation_id: int
    cbid: int = 0
    name: str = ""


CUPTI_SUCCESS = 0
CUPTI_ERROR_MAX_LIMIT_REACHED = 12
CUPTI_ACTIVITY_FLAG_FLUSH_FORCED = 1
_BUFFER_ALIGN_SIZE = 8
_DEFAULT_BUFFER_SIZE = 8 * 1024 * 1024

_BufferRequestedCallback = ctypes.CFUNCTYPE(
    None,
    ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)),
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.POINTER(ctypes.c_size_t),
)
_BufferCompletedCallback = ctypes.CFUNCTYPE(
    None,
    ctypes.c_void_p,
    ctypes.c_uint32,
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.c_size_t,
    ctypes.c_size_t,
)


class _ActivityBase(ctypes.Structure):
    _fields_ = [("kind", ctypes.c_int)]


class _ActivityAPI(ctypes.Structure):
    _fields_ = [
        ("kind", ctypes.c_int),
        ("cbid", ctypes.c_int),
        ("start", ctypes.c_uint64),
        ("end", ctypes.c_uint64),
        ("process_id", ctypes.c_uint32),
        ("thread_id", ctypes.c_uint32),
        ("correlation_id", ctypes.c_uint32),
        ("return_value", ctypes.c_uint32),
    ]


class _ActivityKernel9(ctypes.Structure):
    _fields_ = [
        ("kind", ctypes.c_int),
        ("cache_config", ctypes.c_uint8),
        ("shared_memory_config", ctypes.c_uint8),
        ("registers_per_thread", ctypes.c_uint16),
        ("partitioned_global_cache_requested", ctypes.c_int),
        ("partitioned_global_cache_executed", ctypes.c_int),
        ("start", ctypes.c_uint64),
        ("end", ctypes.c_uint64),
        ("completed", ctypes.c_uint64),
        ("device_id", ctypes.c_uint32),
        ("context_id", ctypes.c_uint32),
        ("stream_id", ctypes.c_uint32),
        ("grid_x", ctypes.c_int32),
        ("grid_y", ctypes.c_int32),
        ("grid_z", ctypes.c_int32),
        ("block_x", ctypes.c_int32),
        ("block_y", ctypes.c_int32),
        ("block_z", ctypes.c_int32),
        ("static_shared_memory", ctypes.c_int32),
        ("dynamic_shared_memory", ctypes.c_int32),
        ("local_memory_per_thread", ctypes.c_uint32),
        ("local_memory_total", ctypes.c_uint32),
        ("correlation_id", ctypes.c_uint32),
        ("grid_id", ctypes.c_int64),
        ("name", ctypes.c_char_p),
    ]


_buffers: list[ctypes.Array] = []
_activities: list[Activity] = []
_user_buffer_requested = None
_user_buffer_completed = None
_request_callback = None
_complete_callback = None


def _candidate_roots() -> list[Path]:
    roots = []
    for raw_path in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
        if raw_path:
            roots.append(Path(raw_path))
    here = Path(__file__).resolve()
    for parent in here.parents:
        site_packages = parent / "lib" / "python3.13" / "site-packages"
        if site_packages.exists():
            roots.extend(
                [
                    site_packages / "nvidia" / "cuda_cupti" / "lib",
                    site_packages / "nvidia" / "cu13" / "lib",
                    site_packages / "triton" / "backends" / "nvidia" / "lib" / "cupti",
                ]
            )
    roots.extend(
        [
            Path("/usr/local/cuda/extras/CUPTI/lib64"),
            Path("/usr/local/cuda/lib64"),
        ]
    )
    return roots


def _load_library() -> ctypes.CDLL:
    for name in ("libcupti.so", "libcupti.so.13", "libcupti.so.12"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            pass
    for root in _candidate_roots():
        for name in ("libcupti.so", "libcupti.so.13", "libcupti.so.12"):
            path = root / name
            if path.exists():
                return ctypes.CDLL(str(path))
    raise ImportError(
        "Cannot find libcupti.so; install nvidia-cuda-cupti or set LD_LIBRARY_PATH"
    )


_lib = _load_library()

_lib.cuptiActivityEnable.argtypes = [ctypes.c_int]
_lib.cuptiActivityEnable.restype = ctypes.c_int
_lib.cuptiActivityDisable.argtypes = [ctypes.c_int]
_lib.cuptiActivityDisable.restype = ctypes.c_int
_lib.cuptiActivityFlushAll.argtypes = [ctypes.c_uint32]
_lib.cuptiActivityFlushAll.restype = ctypes.c_int
_lib.cuptiActivityGetNextRecord.argtypes = [
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.POINTER(_ActivityBase)),
]
_lib.cuptiActivityGetNextRecord.restype = ctypes.c_int
_lib.cuptiActivityRegisterCallbacks.argtypes = [
    _BufferRequestedCallback,
    _BufferCompletedCallback,
]
_lib.cuptiActivityRegisterCallbacks.restype = ctypes.c_int
_lib.cuptiGetTimestamp.argtypes = [ctypes.POINTER(ctypes.c_uint64)]
_lib.cuptiGetTimestamp.restype = ctypes.c_int
_lib.cuptiGetResultString.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
_lib.cuptiGetResultString.restype = ctypes.c_int
if hasattr(_lib, "cuptiFinalize"):
    _lib.cuptiFinalize.argtypes = []
    _lib.cuptiFinalize.restype = ctypes.c_int


def _check(result: int, func_name: str) -> None:
    if result == CUPTI_SUCCESS:
        return
    message_ptr = ctypes.c_char_p()
    if (
        _lib.cuptiGetResultString(result, ctypes.byref(message_ptr)) == CUPTI_SUCCESS
        and message_ptr.value
    ):
        message = message_ptr.value.decode("utf-8", errors="replace")
    else:
        message = f"CUPTI error {result}"
    raise RuntimeError(f"{func_name} failed: {message}")


def _align_buffer(raw_size: int) -> ctypes.Array:
    raw = ctypes.create_string_buffer(raw_size + _BUFFER_ALIGN_SIZE)
    address = ctypes.addressof(raw)
    offset = (address + (_BUFFER_ALIGN_SIZE - 1)) & ~(_BUFFER_ALIGN_SIZE - 1)
    aligned_address = offset
    buffer_type = ctypes.c_uint8 * raw_size
    buffer = buffer_type.from_address(aligned_address)
    buffer._raw_buffer = raw  # type: ignore[attr-defined]
    return buffer


def _request_buffer(buffer_ptr, size_ptr, max_records_ptr) -> None:
    if _user_buffer_requested is not None:
        size, max_records = _user_buffer_requested()
    else:
        size, max_records = _DEFAULT_BUFFER_SIZE, 0
    buffer = _align_buffer(int(size))
    _buffers.append(buffer)
    buffer_ptr[0] = ctypes.cast(buffer, ctypes.POINTER(ctypes.c_uint8))
    size_ptr[0] = int(size)
    max_records_ptr[0] = int(max_records)


def _complete_buffer(_context, _stream_id, buffer, _size, valid_size) -> None:
    parsed: list[Activity] = []
    record = ctypes.POINTER(_ActivityBase)()
    while True:
        result = _lib.cuptiActivityGetNextRecord(
            buffer, valid_size, ctypes.byref(record)
        )
        if result == CUPTI_ERROR_MAX_LIMIT_REACHED:
            break
        _check(result, "cuptiActivityGetNextRecord")
        kind_value = int(record.contents.kind)
        if kind_value == ActivityKind.CONCURRENT_KERNEL:
            kernel = ctypes.cast(record, ctypes.POINTER(_ActivityKernel9)).contents
            parsed.append(
                Activity(
                    kind=ActivityKind.CONCURRENT_KERNEL,
                    start=int(kernel.start),
                    end=int(kernel.end),
                    correlation_id=int(kernel.correlation_id),
                    name=(
                        kernel.name.decode("utf-8", errors="replace")
                        if kernel.name
                        else ""
                    ),
                )
            )
        elif kind_value in (ActivityKind.RUNTIME, ActivityKind.DRIVER):
            api = ctypes.cast(record, ctypes.POINTER(_ActivityAPI)).contents
            parsed.append(
                Activity(
                    kind=ActivityKind(kind_value),
                    start=int(api.start),
                    end=int(api.end),
                    correlation_id=int(api.correlation_id),
                    cbid=int(api.cbid),
                )
            )
    _activities.extend(parsed)
    if _user_buffer_completed is not None:
        _user_buffer_completed(parsed)


def activity_register_callbacks(buffer_requested, buffer_completed) -> None:
    global _user_buffer_requested, _user_buffer_completed, _request_callback, _complete_callback
    _user_buffer_requested = buffer_requested
    _user_buffer_completed = buffer_completed
    _activities.clear()
    _buffers.clear()
    _request_callback = _BufferRequestedCallback(_request_buffer)
    _complete_callback = _BufferCompletedCallback(_complete_buffer)
    _check(
        _lib.cuptiActivityRegisterCallbacks(_request_callback, _complete_callback),
        "cuptiActivityRegisterCallbacks",
    )


def activity_enable(kind: ActivityKind | int) -> None:
    _check(_lib.cuptiActivityEnable(int(kind)), "cuptiActivityEnable")


def activity_disable(kind: ActivityKind | int) -> None:
    _check(_lib.cuptiActivityDisable(int(kind)), "cuptiActivityDisable")


def activity_flush_all(flag: int = 0) -> None:
    _check(_lib.cuptiActivityFlushAll(int(flag)), "cuptiActivityFlushAll")


def get_timestamp() -> int:
    timestamp = ctypes.c_uint64()
    _check(_lib.cuptiGetTimestamp(ctypes.byref(timestamp)), "cuptiGetTimestamp")
    return int(timestamp.value)


def finalize() -> None:
    global _user_buffer_requested, _user_buffer_completed, _request_callback, _complete_callback
    if hasattr(_lib, "cuptiFinalize"):
        _check(_lib.cuptiFinalize(), "cuptiFinalize")
    _user_buffer_requested = None
    _user_buffer_completed = None
    _request_callback = None
    _complete_callback = None
    _buffers.clear()


__all__ = [
    "Activity",
    "ActivityKind",
    "activity_disable",
    "activity_enable",
    "activity_flush_all",
    "activity_register_callbacks",
    "finalize",
    "get_timestamp",
]
