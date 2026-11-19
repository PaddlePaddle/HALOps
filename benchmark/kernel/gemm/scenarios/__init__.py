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

from kernel.gemm.scenarios.batch_gemm import run_scenario as run_batch_gemm
from kernel.gemm.scenarios.dense_gemm import run_scenario as run_dense_gemm
from kernel.gemm.scenarios.grouped_gemm import (
    run_group_masked,
    run_group_contiguous,
)

SCENARIOS = {
    "dense_gemm": run_dense_gemm,
    "batch_gemm": run_batch_gemm,
    "group_masked": run_group_masked,
    "group_contiguous": run_group_contiguous,
}
