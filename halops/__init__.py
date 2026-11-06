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

from importlib.metadata import PackageNotFoundError, version

from halops import attention, gemm, indexer, mla

try:
    __version__ = version("halops")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"
del PackageNotFoundError, version

_SUBPACKAGES = (attention, gemm, indexer, mla)


def __dir__():
    _names = list(globals())
    for _pkg in _SUBPACKAGES:
        _names += [a for a in dir(_pkg) if not a.startswith("_")]
    return sorted(set(_names))


def __getattr__(name):
    for _pkg in _SUBPACKAGES:
        obj = getattr(_pkg, name, None)
        if obj is not None:
            globals()[name] = obj
            return obj
    raise AttributeError(f"module 'halops' has no attribute {name!r}")
