"""Isolate the DSA HALOps kernel for ncu. usage: python dsa_prof.py [T] [TOPK]"""
import sys
import torch
import dsa_ref as D

T = int(sys.argv[1]) if len(sys.argv) > 1 else 256
TOPK = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
q, cache, idx, length, sink = D.make_batch(T, TOPK, seed=T + TOPK)
# warm up allocation / workspace before the profiled region
for _ in range(3):
    D.halops(q, cache, idx, length, sink)
torch.cuda.synchronize()
# profiled launches
for _ in range(2):
    D.halops(q, cache, idx, length, sink)
torch.cuda.synchronize()
