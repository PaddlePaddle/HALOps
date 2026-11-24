"""DSA (full-MLA, topk=2048, no SWA) fp8 reference harness.

Validates the 656 B packed-pool layout and the DSA attention math against
FlashMLA's native fp8 sparse-decode kernel, BEFORE the HALOps DSA kernel
exists.  Pool row = [512 fp8 nope | 4 f32 scales (1/128-dim) | 64 bf16 rope],
fully inline (no separate scale block, unlike HCA).  Contraction K = 576
(512 latent + 64 rope); V = the 512 latent; output 512.

usage: python dsa_ref.py [T=256] [TOPK=2048]
"""
import sys
import torch

torch.set_default_device("cuda")
torch.cuda.set_device(0)
import sgl_kernel.flash_mla as fm

HEADS, LATENT, ROPE = 64, 512, 64
QK = LATENT + ROPE            # 576 contraction
GROUP = 128                   # one f32 scale per 128 nope dims
NGROUP = LATENT // GROUP      # 4
ROW = LATENT + NGROUP * 4 + ROPE * 2   # 512 + 16 + 128 = 656
PAGE = 64
SCALE = QK ** -0.5


def make_dsa_cache(pages, gen):
    """uint8 [pages, PAGE, 1, 656]: each token row is 656 B inline =
    512 fp8 nope | 4 f32 group scales | 64 bf16 rope."""
    buf = torch.zeros(pages, PAGE, ROW, dtype=torch.uint8)
    # per-group positive f32 scales ~ realistic amax/448 magnitudes
    scale = (torch.rand(pages, PAGE, NGROUP, generator=gen) * 0.5 + 0.25)
    nope_f = torch.randn(pages, PAGE, LATENT, generator=gen)
    nope_q = (nope_f / scale.repeat_interleave(GROUP, -1)).clamp(-448, 448)
    buf[..., :LATENT] = nope_q.to(torch.float8_e4m3fn).view(torch.uint8)
    buf[..., LATENT:LATENT + NGROUP * 4] = scale.view(torch.uint8).view(
        pages, PAGE, NGROUP * 4
    )
    rope = torch.randn(pages, PAGE, ROPE, generator=gen).bfloat16()
    buf[..., LATENT + NGROUP * 4:] = rope.view(torch.uint8).view(pages, PAGE, ROPE * 2)
    return buf.view(pages, PAGE, 1, ROW)


def dequant(cache, idx):
    """bf16 [n, 576] K rows (and [n,512] V = the latent) for token indices idx."""
    flat = cache.view(-1, ROW)[idx.long()]
    fp8 = flat[:, :LATENT].view(torch.float8_e4m3fn).float()
    scale = flat[:, LATENT:LATENT + NGROUP * 4].contiguous().view(torch.float32)
    nope = (fp8 * scale.repeat_interleave(GROUP, -1)).bfloat16()
    rope = flat[:, LATENT + NGROUP * 4:].contiguous().view(torch.bfloat16)
    return torch.cat([nope, rope], -1), nope  # K[576], V=nope[512]


def reference(q, cache, idx, length, sink):
    """fp32 attention over dequantized keys, sink in the denominator."""
    out = torch.zeros(q.shape[0], HEADS, LATENT, device=q.device, dtype=torch.bfloat16)
    for t in range(q.shape[0]):
        keep = idx[t][: int(length[t])]
        keep = keep[keep >= 0].long()
        if keep.numel() == 0:
            continue
        K, V = dequant(cache, keep)
        s = q[t].float() @ K.float().T * SCALE
        m = torch.maximum(s.amax(-1), sink)
        p = torch.exp(s - m[:, None])
        o = p @ V.float() / (p.sum(-1) + torch.exp(sink - m))[:, None]
        out[t] = o.bfloat16()
    return out


def make_batch(T, topk, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    pages = (T * topk) // PAGE // 4 + 2   # shared pool, random index lists
    cache = make_dsa_cache(pages, gen)
    n = pages * PAGE
    idx = torch.stack([torch.randperm(n, generator=gen)[:topk].int() for _ in range(T)])
    length = torch.full((T,), topk, dtype=torch.int32)
    q = torch.randn(T, HEADS, QK, generator=gen).bfloat16()
    sink = torch.randn(HEADS, generator=gen)
    return q, cache, idx, length, sink


def flash(q, cache, idx, length, sink):
    kv_paged = cache.view(-1, PAGE, 1, ROW)
    meta = fm.get_mla_metadata(is_fp8_kvcache=True, topk=idx.shape[-1])[0]
    o, _ = fm.flash_mla_with_kvcache(
        q=q[:, None], k_cache=kv_paged, block_table=None, cache_seqlens=None,
        head_dim_v=LATENT, tile_scheduler_metadata=meta, softmax_scale=SCALE,
        causal=False, is_fp8_kvcache=True, indices=idx[:, None],
        attn_sink=sink, topk_length=length,
    )
    return o[:, 0]


def halops(q, cache, idx, length, sink):
    from halops import dsa_decode_attention
    # Each token has its own independent index list, so no pairing: tpr == 1
    # makes every token a lone request (no union of neighbours).
    return dsa_decode_attention(
        q.contiguous(), cache.contiguous(), idx.int().contiguous(),
        length.int().contiguous(), sink.float().contiguous(), SCALE,
        tokens_per_req=1,
    )


def max_rel_err(out, ref):
    d = (out.float() - ref.float()).abs().flatten(1).amax(-1)
    scale = ref.float().abs().flatten(1).amax(-1).clamp_min(1e-3)
    return (d / scale).max().item()


if __name__ == "__main__":
    import os
    T = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    TOPK = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
    q, cache, idx, length, sink = make_batch(T, TOPK, seed=T + TOPK)
    if os.environ.get("ROPE_ZERO"):
        # Zero both q's rope dims and the cache's inline rope so only the 512
        # latent (nope) contributes — isolates the nope path from the ss rope.
        q[..., LATENT:] = 0
        cache.view(-1, ROW)[:, LATENT + NGROUP * 4:] = 0
    if os.environ.get("NOPE_ZERO"):
        # Zero the latent so only rope contributes — isolates the ss rope path.
        q[..., :LATENT] = 0
        cache.view(-1, ROW)[:, :LATENT] = 0
    if os.environ.get("QROPE_ZERO"):
        # Zero ONLY q's rope dims (keep cache rope). The ss-MMA still runs with
        # a zero A operand, so its logit contribution must be 0 and the output
        # must match nope-only. Isolates ss-MMA plumbing from the rope dot/data.
        q[..., LATENT:] = 0
    if os.environ.get("KROPE_ZERO"):
        # Zero ONLY the cache's inline rope (keep q rope). Mirror of QROPE_ZERO.
        cache.view(-1, ROW)[:, LATENT + NGROUP * 4:] = 0
    ref = reference(q, cache, idx, length, sink)
    fla = flash(q, cache, idx, length, sink)
    hal = halops(q, cache, idx, length, sink)
    torch.cuda.synchronize()
    print(f"DSA T={T} topk={TOPK} row={ROW}B  FlashMLA-fp8 vs fp32-ref "
          f"relerr = {max_rel_err(fla, ref):.3e}")
    print(f"DSA T={T} topk={TOPK} row={ROW}B  HALOps-fp8  vs fp32-ref "
          f"relerr = {max_rel_err(hal, ref):.3e}")
    print(f"DSA T={T} topk={TOPK} row={ROW}B  HALOps-fp8  vs FlashMLA "
          f"relerr = {max_rel_err(hal, fla):.3e}")
