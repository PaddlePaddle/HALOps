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

import os
import re
import sys
import threading
from contextlib import contextmanager


class Tee:
    """Write Python output and native stderr to the console and a log file."""

    _SHUTDOWN_MARKER = b"\0halops-tee-shutdown\0"
    _IGNORED_NATIVE_STDERR = re.compile(
        rb"^USDT:.* SyncActivityProfilerHandler\.cpp:\d+\] "
        rb"profiler_(?:start|stop)$"
    )

    def __init__(self, log_path: str, intercept_native_stderr: bool = False):
        """Open the log and optionally route native stderr through a pipe."""
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        self._file = open(log_path, "w", buffering=1)
        self._stdout = sys.stdout
        self._lock = threading.Lock()
        self._closed = False
        self._intercept_native_stderr = intercept_native_stderr

        if not intercept_native_stderr:
            return

        # Python-level sys.stderr does not see writes from C++ libraries.
        self._saved_stderr_fd = os.dup(2)
        read_fd, write_fd = os.pipe()
        os.dup2(write_fd, 2)
        os.close(write_fd)
        self._native_stderr_thread = threading.Thread(
            target=self._forward_native_stderr,
            args=(read_fd,),
            name="benchmark-native-stderr",
            daemon=True,
        )
        self._native_stderr_thread.start()

    @staticmethod
    def _write_all(fd: int, data: bytes) -> None:
        """Write all bytes to a file descriptor, handling partial writes."""
        while data:
            written = os.write(fd, data)
            data = data[written:]

    def _forward_native_stderr(self, read_fd: int) -> None:
        """Forward native stderr to console and log, excluding known noise."""
        with os.fdopen(read_fd, "rb", buffering=0) as stream:
            for line in stream:
                marker = line.find(self._SHUTDOWN_MARKER)
                stopping = marker >= 0
                if stopping:
                    line = line[:marker]
                if line and not self._IGNORED_NATIVE_STDERR.fullmatch(
                    line.rstrip(b"\r\n")
                ):
                    try:
                        self._write_all(self._saved_stderr_fd, line)
                        with self._lock:
                            self._file.write(line.decode(errors="replace"))
                    except OSError:
                        break
                if stopping:
                    break

    def write(self, data: str):
        """Mirror Python-level output to the original console and log file."""
        self._stdout.write(data)
        with self._lock:
            self._file.write(data)

    def flush(self):
        """Flush both Python-level output destinations."""
        self._stdout.flush()
        with self._lock:
            self._file.flush()

    def close(self):
        """Restore native stderr, stop its forwarding thread, and close the log."""
        if self._closed:
            return
        self._closed = True
        if not self._intercept_native_stderr:
            with self._lock:
                self._file.close()
            return
        try:
            self._write_all(2, self._SHUTDOWN_MARKER + b"\n")
        except OSError:
            pass
        try:
            os.dup2(self._saved_stderr_fd, 2)
        finally:
            self._native_stderr_thread.join()
            os.close(self._saved_stderr_fd)
            with self._lock:
                self._file.close()

    def fileno(self):
        """Expose the original stdout descriptor for stream compatibility."""
        return self._stdout.fileno()

    def isatty(self):
        """Report a non-interactive stream because output is also logged."""
        return False


@contextmanager
def tee_output(log_path: str, intercept_native_stderr: bool = False):
    """Tee Python output and optionally intercept native stderr."""
    tee = Tee(log_path, intercept_native_stderr=intercept_native_stderr)
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = tee
    try:
        yield tee
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        tee.close()
