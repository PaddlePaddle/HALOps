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

import numpy as np
import torch

NVFP4_BLOCK_SIZE = 16
MXFP4_BLOCK_SIZE = 32
FP4_E2M1_MAX = 6.0
FP4_E2M1_VALUES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
FP4_MIDPOINTS = (FP4_E2M1_VALUES[:-1] + FP4_E2M1_VALUES[1:]) / 2.0


def fp4_abs_to_idx_round_half_up(abs_x_flat):
    """FP4 magnitude index; a value on a midpoint takes the higher bin."""
    return np.searchsorted(FP4_MIDPOINTS, abs_x_flat, side="right").astype(np.uint8)


def float32_to_ue4m3_rne(x):
    """Encode non-negative float32 into a UE4M3 bit pattern (uint8), RNE rounding."""
    x_tensor = torch.tensor(np.asarray(x, dtype=np.float32), dtype=torch.float32)
    x_tensor = torch.clip(x_tensor, min=0.0, max=448.0)
    fp8_tensor = x_tensor.to(torch.float8_e4m3fn)
    return fp8_tensor.view(torch.uint8).cpu().numpy().reshape(np.asarray(x).shape)
