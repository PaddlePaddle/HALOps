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

from kernel.attention.scenarios.decode_gqa import run_scenario as run_decode_gqa
from kernel.attention.scenarios.prefill_dense import run_scenario as run_prefill_dense
from kernel.attention.scenarios.prefill_chunked_gqa import (
    run_scenario as run_prefill_chunked_gqa,
)

SCENARIOS = {
    "decode_gqa": run_decode_gqa,
    "prefill_dense": run_prefill_dense,
    "prefill_chunked_gqa": run_prefill_chunked_gqa,
}
