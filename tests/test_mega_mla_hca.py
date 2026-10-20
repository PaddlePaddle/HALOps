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

Accuracy gate for `mega_mla` on the HCA path (compress_ratio 128).
"""

from __future__ import annotations

import pytest

from tests.mega_mla_gate import Case, Mode, check, params, skip_reason

pytestmark = pytest.mark.skipif(skip_reason() is not None, reason=skip_reason() or "")

MODES = [
    Mode(name="hca", c128_len=128, fire=True),
    Mode(name="hca_idle_long", c128_len=511, fire=False),
    Mode(name="hca_fired_long", c128_len=511, fire=True),
    Mode(name="hca_odd_blocks", c128_len=160, fire=True),
]


@pytest.mark.parametrize("mode,case", params(MODES))
def test_accuracy(mode: Mode, case: Case):
    """Tolerance gate: every (scenario, shape) stays inside the fp8 noise band."""
    check(mode, case)
