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

import torch

import halops.halops_kernel as halops_kernel

__all__ = ["mhc_post_hc_mean"]


def mhc_post_hc_mean(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    use_pdl: bool = True,
) -> torch.Tensor:
    """mHC post mixing after the last layer, averaged over the ``hc`` streams.

    Computes ``mean_j bf16(post[j] * x + sum_i comb[i, j] * residual[i])``
    per token in fp32, rounded to bf16. All tensors must be contiguous.

    Args:
        x: [T, hidden] bfloat16 layer output.
        residual: [T, hc, hidden] bfloat16 residual streams; hc is 2 or 4.
        post: [T, hc] float32 post weights.
        comb: [T, hc, hc] float32 comb matrices, [source, dest].
        use_pdl: Enable Programmatic Dependent Launch (on by default).

    Returns:
        [T, hidden] bfloat16 tensor.

    Example:
        >>> from halops import mhc_post_hc_mean
        >>>
        >>> x = torch.randn(4, 4096, dtype=torch.bfloat16, device="cuda")
        >>> residual = torch.randn(4, 4, 4096, dtype=torch.bfloat16, device="cuda")
        >>> post = torch.rand(4, 4, device="cuda")
        >>> comb = torch.rand(4, 4, 4, device="cuda")
        >>> out = mhc_post_hc_mean(x, residual, post, comb)
    """
    return halops_kernel.mhc_post_hc_mean(x, residual, post, comb, use_pdl)
