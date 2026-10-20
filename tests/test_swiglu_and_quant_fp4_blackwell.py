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
import pytest
import torch

from tests.swiglu_quant_test_common import (
    get_sfatom_valid_byte_mask_fp4,
    get_valid_row_mask,
    run_single_case_fp4,
    swiglu_and_quant_fused,
)


class TestFusedMaskSwigluQuantFP4:
    @pytest.fixture(autouse=True)
    def _seed(self):
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        np.random.seed(42)

    def test_sanity_all_valid(self):
        """All tokens valid sanity check."""
        run_single_case_fp4(4, 128, 2048, token_nums_list=[128, 128, 128, 128])

    def test_cases(self):
        """Representative shape sweep."""
        run_single_case_fp4(4, 128, 2048, token_nums_list=[128, 128, 128, 128])
        test_cases = [
            (4, 128, 2048),
            (20, 128, 14336),
            (20, 256, 14336),
            (20, 1024, 14336),
            (8, 512, 7168),
        ]
        for group_num, group_size, H in test_cases:
            run_single_case_fp4(group_num, group_size, H)

    def test_unsupported_kb_dim(self):
        """Reject FP4 scale layouts whose kb_dim is not divisible by 4."""
        group_num = 1
        group_size = 128
        hidden = 80
        H = hidden * 2
        x = torch.randn([group_num * group_size, H]).to(torch.bfloat16)
        token_nums = torch.tensor([group_size], dtype=torch.int32)

        msg = "out_scale kb_dim must be divisible by 4"
        with pytest.raises(RuntimeError, match=msg):
            swiglu_and_quant_fused(x, token_nums, is_fp4=True)

    def test_token_nums_edge_cases(self):
        """Edge token_nums distributions."""
        torch.manual_seed(123)
        torch.cuda.manual_seed_all(123)
        np.random.seed(123)

        E, M, H = 4, 128, 2048
        edge_lists = [
            [0, 0, 0, 0],
            [0, 128, 0, 64],
            [128, 128, 128, 128],
            [1, 1, 1, 1],
        ]
        for tn in edge_lists:
            run_single_case_fp4(E, M, H, token_nums_list=tn)

    def test_many_experts(self):
        """Stress binary-search and SMEM with large expert counts."""
        torch.manual_seed(7)
        torch.cuda.manual_seed_all(7)
        np.random.seed(7)

        for E in [128, 256]:
            run_single_case_fp4(E, 128, 2048)

    def test_determinism(self):
        """Bit-exact repeatability for identical inputs (valid rows only).

        Invalid rows in out_quant/out_scale are uninitialized (torch.empty),
        so only compare positions that the kernel actually writes.
        """
        torch.manual_seed(2026)
        torch.cuda.manual_seed_all(2026)
        np.random.seed(2026)

        E, M, H = 4, 128, 2048
        hidden = H // 2
        x = torch.randn([E * M, H]).to(torch.bfloat16)
        tn_list = [np.random.randint(1, M + 1) for _ in range(E)]
        tn = torch.tensor(tn_list, dtype=torch.int32)
        tn_np = np.array(tn_list, dtype=np.int32)
        valid_row = get_valid_row_mask(tn_np, M)

        kb_dim = hidden // 16
        sf_mask = get_sfatom_valid_byte_mask_fp4(tn_np, M, kb_dim, hidden)

        q0, s0 = swiglu_and_quant_fused(x, tn, is_fp4=True)
        q0_v = q0.view(torch.uint8).cpu().numpy()[valid_row]
        s0_v = s0.cpu().numpy().reshape(-1)[sf_mask]
        for _ in range(2):
            q, s = swiglu_and_quant_fused(x, tn, is_fp4=True)
            q_v = q.view(torch.uint8).cpu().numpy()[valid_row]
            s_v = s.cpu().numpy().reshape(-1)[sf_mask]
            assert np.array_equal(q_v, q0_v), "non-deterministic quant output (fp4)"
            assert np.array_equal(s_v, s0_v), "non-deterministic scale (fp4)"


if __name__ == "__main__":
    pytest.main([__file__])
