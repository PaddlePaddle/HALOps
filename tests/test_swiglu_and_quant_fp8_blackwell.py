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
    get_sfatom_valid_byte_mask_fp8,
    get_valid_row_mask,
    run_single_case_fp8,
    swiglu_and_quant_fused,
)


class TestFusedMaskSwigluQuantFP8:
    @pytest.fixture(autouse=True)
    def _seed(self):
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        np.random.seed(42)

    def test_sanity_all_valid(self):
        """All tokens valid sanity check."""
        run_single_case_fp8(4, 128, 2048, token_nums_list=[128, 128, 128, 128])

    def test_cases(self):
        """Representative shape sweep."""
        run_single_case_fp8(4, 128, 2048, token_nums_list=[128, 128, 128, 128])
        test_cases = [
            (4, 128, 2048),
            (20, 128, 14336),
            (20, 256, 14336),
            (20, 1024, 14336),
            (8, 512, 7168),
        ]
        for group_num, group_size, H in test_cases:
            run_single_case_fp8(group_num, group_size, H)

    def test_remainder_branches(self):
        """Cover FP8 tail-remainder paths."""
        # H=2*hidden; FP8 num_iters = hidden / 128
        # num_iters % 4 == 2: hidden in {256, 768} -> H in {512, 1536}
        # odd num_iters: hidden in {128, 384, 896} -> H in {256, 768, 1792}
        cases = [
            (4, 128, 512),  # num_iters=2 (pair-remainder)
            (4, 128, 1536),  # num_iters=6 (one quad + one pair)
            (4, 128, 256),  # num_iters=1 (single-remainder only)
            (4, 128, 768),  # num_iters=3 (one pair + single)
            (4, 128, 1792),  # num_iters=7 (one quad + pair + single)
        ]
        for E, M, H in cases:
            run_single_case_fp8(E, M, H, token_nums_list=[M] * E)
            run_single_case_fp8(E, M, H)

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
            run_single_case_fp8(E, M, H, token_nums_list=tn)

    def test_many_experts(self):
        """Stress binary-search and SMEM with large expert counts."""
        torch.manual_seed(7)
        torch.cuda.manual_seed_all(7)
        np.random.seed(7)

        for E in [128, 256]:
            run_single_case_fp8(E, 128, 2048)

    def test_input_validation(self):
        """Negative cases corresponding to host-side TORCH_CHECKs."""
        torch.manual_seed(0)
        E, M, H = 4, 128, 2048
        base_x = torch.randn([E * M, H]).to(torch.bfloat16)
        base_tn = torch.tensor([M] * E, dtype=torch.int32)

        # Wrong input dtype
        with pytest.raises((RuntimeError, TypeError)):
            swiglu_and_quant_fused(base_x.float(), base_tn, is_fp4=False)

        # Wrong token_nums dtype (int64)
        with pytest.raises((RuntimeError, TypeError)):
            swiglu_and_quant_fused(base_x, base_tn.long(), is_fp4=False)

        # Non-contiguous input: take every-other row to break row contiguity
        big_x = torch.randn([E * M * 2, H]).to(torch.bfloat16)
        nc = big_x[::2]
        assert not nc.is_contiguous()
        with pytest.raises((RuntimeError, TypeError)):
            swiglu_and_quant_fused(nc, base_tn, is_fp4=False)

        # CPU tensor
        with pytest.raises((RuntimeError, TypeError)):
            swiglu_and_quant_fused(base_x.cpu(), base_tn.cpu(), is_fp4=False)

        # Note: token_nums value-range (0 <= n <= group_size) is a caller
        # precondition and intentionally not validated on the host to avoid a
        # device sync, so there are no negative cases for it here.

        # FP8 hidden not divisible by 128: hidden=64 -> H=128
        bad_x = torch.randn([E * M, 128]).to(torch.bfloat16)
        with pytest.raises((RuntimeError, TypeError)):
            swiglu_and_quant_fused(bad_x, base_tn, is_fp4=False)

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

        n_scale = ((hidden // 128) + 3) // 4
        kb_dim = n_scale * 16
        sf_mask = get_sfatom_valid_byte_mask_fp8(tn_np, M, kb_dim, hidden)

        q0, s0 = swiglu_and_quant_fused(x, tn, is_fp4=False)
        q0_v = q0.view(torch.uint8).cpu().numpy()[valid_row]
        s0_v = s0.cpu().numpy().reshape(-1)[sf_mask]
        for _ in range(2):
            q, s = swiglu_and_quant_fused(x, tn, is_fp4=False)
            q_v = q.view(torch.uint8).cpu().numpy()[valid_row]
            s_v = s.cpu().numpy().reshape(-1)[sf_mask]
            assert np.array_equal(q_v, q0_v), "non-deterministic quant output (fp8)"
            assert np.array_equal(s_v, s0_v), "non-deterministic scale (fp8)"


if __name__ == "__main__":
    pytest.main([__file__])
