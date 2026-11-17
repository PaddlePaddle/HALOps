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

import importlib
import os
from pathlib import Path

_current_dir = str(Path(__file__).parent)

for _fname in os.listdir(_current_dir):
    if not _fname.endswith(".py") or _fname.startswith("_"):
        continue
    _mod_name = _fname[:-3]
    _mod = importlib.import_module(f".{_mod_name}", __package__)
    for _attr in _mod.__all__:
        if _attr.startswith("_"):
            continue
        _obj = getattr(_mod, _attr, None)
        if callable(_obj):
            globals()[_attr] = _obj

del _current_dir, _fname, _mod_name, _mod, _attr, _obj, importlib, os, Path
