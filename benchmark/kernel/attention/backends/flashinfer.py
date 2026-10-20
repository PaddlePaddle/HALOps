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
import torch


try:
    import flashinfer
    from flashinfer.decode import (
        BatchDecodeWithPagedKVCacheWrapper,
        trtllm_batch_decode_with_kv_cache,
    )
    from flashinfer.prefill import (
        BatchPrefillWithRaggedKVCacheWrapper,
        BatchPrefillWithPagedKVCacheWrapper,
    )

    _FI_AVAILABLE = True
except ImportError:
    flashinfer = None
    _FI_AVAILABLE = False


def _check():
    if not _FI_AVAILABLE:
        raise ImportError("flashinfer not installed. pip install flashinfer")


def plan_fi_decode_paged(
    q: torch.Tensor,
    kv_data: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_last_page_len: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    backend: str = "auto",
    q_len_per_req: int = 1,
    kv_data_type: torch.dtype | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
) -> dict:
    """Create wrapper, plan, and return dict with keys: q, kv_data, wrapper."""
    _check()
    workspace = torch.zeros(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    wrapper = BatchDecodeWithPagedKVCacheWrapper(
        float_workspace_buffer=workspace,
        kv_layout="NHD",
        use_tensor_cores=True,
        backend=backend,
    )
    plan_kwargs = dict(
        indptr=kv_indptr,
        indices=kv_indices,
        last_page_len=kv_last_page_len,
        num_qo_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_size=page_size,
        q_data_type=q.dtype,
        q_len_per_req=q_len_per_req,
        kv_data_type=kv_data.dtype,
        o_data_type=torch.bfloat16,
    )
    wrapper.plan(**plan_kwargs)
    # Pre-allocate a zeroed output buffer here (adapt/plan phase) instead of in
    # run_fi_decode_paged. With q_len_per_req>1 the paged-decode kernel leaves
    # some output lanes unwritten; an auto-allocated buffer can then expose stale
    # garbage (NaN/Inf) from recycled allocator memory. A pre-zeroed, reused
    # buffer keeps those lanes at 0 and gives run() a fixed address (needed for
    # CUDA-graph capture / replay).
    out = torch.zeros(q.shape, dtype=torch.bfloat16, device=q.device)
    return {
        "q": q,
        "kv_data": kv_data,
        "wrapper": wrapper,
        "q_scale": q_scale,
        "k_scale": k_scale,
        "v_scale": v_scale,
        "out": out,
    }


def run_fi_decode_paged(inp: dict) -> torch.Tensor:
    """Run a planned decode (output of plan_fi_decode_paged).

    Uses the pre-zeroed out= buffer created in plan_fi_decode_paged instead of
    letting FlashInfer auto-allocate. With q_len_per_req>1 the paged-decode
    kernel leaves some output lanes unwritten; an auto-allocated buffer can then
    expose stale garbage (NaN/Inf) from recycled allocator memory. A pre-zeroed
    buffer keeps those lanes at 0.
    NOTE: this only stabilizes the output tensor; for q_len_per_req>1 the extra
    query tokens are still not computed by this fi kernel, so those rows are not
    a valid result.
    """
    kwargs = {}
    if inp.get("q_scale") is not None:
        kwargs["q_scale"] = inp["q_scale"]
    if inp.get("k_scale") is not None:
        kwargs["k_scale"] = inp["k_scale"]
    if inp.get("v_scale") is not None:
        kwargs["v_scale"] = inp["v_scale"]
    return inp["wrapper"].run(inp["q"], inp["kv_data"], out=inp["out"], **kwargs)


def run_fi_trtllm_decode(inp: dict) -> torch.Tensor:
    """Run trtllm_batch_decode_with_kv_cache.

    inp keys:
        q: [num_tokens, num_heads, head_dim]
        kv_cache: [num_pages, 2, page_size, num_kv_heads, head_dim] (NHD layout)
        block_tables: [B, max_pages_per_seq]
        seq_lens: [B]
        max_seq_len: int
        workspace: torch.Tensor
        out: pre-zeroed output buffer (created in adapt phase)
        bmm1_scale: optional float, fused scale post-QK^T
            (softmax_scale * q_dequant_scale * k_dequant_scale)
        bmm2_scale: optional float, fused scale post-PV (v_dequant_scale)
        q_len_per_req: optional int
    """
    _check()
    kwargs = {}
    if inp.get("bmm1_scale") is not None:
        kwargs["bmm1_scale"] = inp["bmm1_scale"]
    if inp.get("bmm2_scale") is not None:
        kwargs["bmm2_scale"] = inp["bmm2_scale"]
    return trtllm_batch_decode_with_kv_cache(
        query=inp["q"],
        kv_cache=inp["kv_cache"],
        workspace_buffer=inp["workspace"],
        block_tables=inp["block_tables"],
        seq_lens=inp["seq_lens"],
        max_seq_len=inp["max_seq_len"],
        out=inp["out"],
        kv_layout="NHD",
        q_len_per_req=inp.get("q_len_per_req", 1),
        **kwargs,
    )


def plan_fi_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    softmax_scale: float,
    causal: bool,
    workspace: torch.Tensor,
) -> dict:
    """Create wrapper, plan, and return dict with keys: q, k, v, wrapper."""
    _check()
    q_flat = q.flatten(0, 1)
    k_flat = k.flatten(0, 1)
    v_flat = v.flatten(0, 1)
    wrapper = BatchPrefillWithRaggedKVCacheWrapper(workspace, "NHD")
    wrapper.plan(
        qo_indptr=q_indptr,
        kv_indptr=kv_indptr,
        num_qo_heads=q.shape[2],
        num_kv_heads=k.shape[2],
        head_dim_qk=q.shape[3],
        causal=causal,
        sm_scale=softmax_scale,
        q_data_type=q.dtype,
    )
    return {"q": q_flat, "k": k_flat, "v": v_flat, "wrapper": wrapper}


def run_fi_prefill(inp: dict) -> torch.Tensor:
    """Run a planned prefill (output of plan_fi_prefill)."""
    out = inp["wrapper"].run(inp["q"], inp["k"], inp["v"])
    return out


def plan_fi_prefill_paged(
    q: torch.Tensor,
    kv_data: torch.Tensor,
    q_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_last_page_len: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    causal: bool,
    softmax_scale: float,
    workspace: torch.Tensor,
    backend: str = "auto",
    seq_lens: torch.Tensor | None = None,
    seq_lens_q: torch.Tensor | None = None,
    max_token_per_sequence: int | None = None,
    max_sequence_kv: int | None = None,
) -> dict:
    """Create wrapper, plan, and return dict with keys: q, kv_data, wrapper."""
    _check()
    wrapper = BatchPrefillWithPagedKVCacheWrapper(
        float_workspace_buffer=workspace,
        kv_layout="NHD",
        backend=backend,
    )
    wrapper.plan(
        qo_indptr=q_indptr,
        paged_kv_indptr=kv_indptr,
        paged_kv_indices=kv_indices,
        paged_kv_last_page_len=kv_last_page_len,
        num_qo_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        page_size=page_size,
        causal=causal,
        sm_scale=softmax_scale,
        q_data_type=q.dtype,
        seq_lens=seq_lens,
        seq_lens_q=seq_lens_q,
        max_token_per_sequence=max_token_per_sequence,
        max_sequence_kv=max_sequence_kv,
    )
    return {"q": q, "kv_data": kv_data, "wrapper": wrapper}


def run_fi_prefill_paged(inp: dict) -> torch.Tensor:
    """Run a planned prefill paged (output of plan_fi_prefill_paged)."""
    return inp["wrapper"].run(inp["q"], inp["kv_data"])
