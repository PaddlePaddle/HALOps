# EB5.5 HALOps 需求：按请求合批的 HCA decode attention

整理人：lizhenyun（2026-11-05）。背景、测量方法和复现脚本都在 `/root/paddlejob/inference-public/lizhenyun/eb_1m`（下文简称 `eb_1m`）。量化、PDL、交付方式的公共约定和 `EB55_OP_REQUIREMENTS.md` §0 相同。

## 1. 为什么要做

EB5.5 decode 节点（DP4，开投机）里，HCA 层（43 层里有 37 层）的 attention 现在走 FlashMLA 的 `flash_fwd_splitkv_mla_fp8_sparse_kernel`（sm100 sparse head64）加 combine，是单个最大的 kernel：

| 场景（每 rank） | T（token） | 每请求 token t | attention + combine / 层 | 37 层合计 | 占 VERIFY step |
|---|---:|---:|---:|---:|---:|
| DSpark 3，C=768 | 768 | 4 | 157 µs | 5.8 ms | 18% |
| DSpark 7，C=128 | 256 | 8 | 63 µs | 2.3 ms | 12% |
| DSpark 7，C=768 | 1536 | 8 | 295 µs | 10.9 ms | — |

ncu（`--set full`，T=768、t=4，报告在 `eb_1m/profiler/ncu/hca_attn_T768_tpr4.ncu-rep`）的结论：

- **每个 token 独立读、反量化同一请求的 KV。** DRAM 读了 489 MB，正好等于"每 token 读一遍 ~987 条 KV × 584 B"，L2 命中率只有 7.5%。t=1 和 t=4 的耗时完全一样（157.2 和 157.7 µs）。
- **既不是带宽瓶颈，也不是算力瓶颈。** DRAM 吞吐 42%，tensor pipe 活跃 28%（约 630 TFLOPS）。
  - 每个 SM 只能放一个 CTA：168 寄存器 × 384 线程，smem 218 KB，每个 scheduler 只有 2.9 个 warp。
  - 62% 的 stall 是 long scoreboard，SASS 上集中在等 TMA gather 的 mbarrier。
  - 本质上是"每个 (token, 64 条 KV) 工作项约 1.9 µs"的延迟链，工作项数量是 T × 16。

同一请求的 t 个 verify token 看到的 c128 历史完全相同，SWA 窗口也只差 t−1 条，但现在被读了 t 次、反量化了 t 次。

## 2. 要做什么

新写一个 HCA decode attention，一个工作项 = **(请求, KV tile) × 这个请求的全部 t 个 token 的 64 个头**，KV tile 只 gather 和反量化一次。语义和现有调用完全一致：

```python
o = flash_mla_with_kvcache(
    q=q,                                   # [T, 1, 64, 512] bf16（448 nope + 64 rope）
    k_cache=swa_k_cache,                   # SWA 池
    head_dim_v=512, block_table=None, cache_seqlens=None,
    tile_scheduler_metadata=flashmla_metadata,
    softmax_scale=512 ** -0.5, is_fp8_kvcache=True,
    indices=swa_page_indices,              # [T, 1, 128] int32
    topk_length=swa_topk_lengths,          # [T] int32
    attn_sink=attn_sink,                   # [64] fp32
    extra_k_cache=c128_k_cache,            # c128 池
    extra_indices_in_kvcache=c128_page_indices,  # [T, 1, W] int32，W 补齐到 64 的倍数（现在是 8256）
    extra_topk_length=c128_topk_lengths,         # [T] int32
)[0]                                       # [T, 1, 64, 512] bf16
```

调用点：`sglang/python/sglang/srt/layers/attention/deepseek_v4_backend.py` 的 decode 分支（`compress_ratio == 128`）。

### 2.1 接口（放在 `halops.hca_decode_attention`）

```python
def hca_decode_attention(
    q,                  # [T, 64, 512] bf16（或 [T, 1, 64, 512]）
    swa_k_cache,        # uint8 [pages, 128, 1, 584]（每页 128 token，页 stride 补齐到 576 的倍数）
    swa_indices,        # int32 [T, 128]（或 [T, 1, 128]）
    swa_lengths,        # int32 [T]
    c128_k_cache,       # uint8 [pages, 2, 1, 584]（每页 2 条）
    c128_indices,       # int32 [T, W]
    c128_lengths,       # int32 [T]
    attn_sink,          # fp32 [64]
    softmax_scale: float,
    tokens_per_req: int,  # t；token 按请求排列，第 r 个请求是 [r*t, (r+1)*t)
    out=None,           # 可选预分配 [T, 64, 512] bf16
    use_pdl: bool = True,
) -> torch.Tensor
```

- KV 字节布局、indices 语义（池内 token 下标，-1 或越界表示无效）、sink 语义都和 FlashMLA 现在的 fp8 sparse decode 一样，同一份 cache 直接复用。
- 输出直接给 `halops.vha_postmix_quant`，要求 contiguous `[T, 64*512]`。
- kernel 数量：attention 一个，加最多一个 combine。不允许 host 侧 memset，也不引入其他小 kernel。
- `tokens_per_req == 1` 时可以直接报不支持，接入侧会走 FlashMLA。

### 2.2 能利用的输入结构（真实数据已验证）

用 `eb_1m/fixtures/hca_attn/*.pt` 验证过，这是服务里 dump 的真实下标和长度：

- **c128**：同一请求的 t 行下标前缀完全相同，长度单调不减，最多差 1（跨过 128 边界时，后面的 token 多看到本轮刚压出来的一条）。所以 c128 可以用该请求最后一行的下标做共享 tile，每行按自己的长度 mask。c128 占 KV 的 87%（110k 上下文时约 859 条）。
- **SWA**：每行 128 条，**从新到旧**排列，相邻 token 依次错一位：第 j+1 行的 `[1:]` 等于第 j 行的 `[:-1]`。所以一个请求的并集是 128+t−1 条。
- **padding 请求**：CUDA graph 补齐出来的假请求（`hca_T164_dev2.pt` 第 40 个请求）`seq_len=1`、positions 全为 0、下标是 0 或 -1、长度很短，不满足上面的结构。它的输出会被丢弃，但 kernel 不能越界访问，也不能挂。
- **建议分两步**：第一步只共享 c128，SWA 仍按行处理，结构假设最少，DRAM 流量已经从 489 MB 降到约 150 MB；第二步再共享 SWA 的并集。

### 2.3 设计约束（给出约束，具体方案由你定）

- t×64 行（t=4 是 256 行，t=8 是 512 行）。O 累加器 128 行 × 512 列 fp32 就会占满 256 KB TMEM，S 放不下。可选的做法：
  - 2-CTA cluster 按 dv 拆成两半，KV tile 用 TMA multicast；
  - 一个 CTA 只处理 2 个 token，同一请求的其余 token 放到相邻 CTA，靠 L2 去重；
  - dv 分两遍。
- split-KV 加 combine 的切分要保证 T=256 时 152 个 SM 也能占满：T=256 只有 32 个请求，每个请求要切成约 5 段。
- PDL：q、indices、lengths、c128 cache 都是同一 step 上游写的，必须在 `pdl_wait()` 之后读；attn_sink 可以提前读。写完再 `pdl_trigger()`，下游是 `vha_postmix_quant`。

## 3. 验收

- **正确性**：在相同输入下和 `flash_mla_with_kvcache` 比较（fp32 相对误差，按 bf16 + split 顺序给出容差）。
  - 覆盖 t ∈ {2, 4, 8}、T ∈ {8, 64, 128, 256, 768, 1536}、c128 长度在请求内 +1 的情况、padding 请求、全部无效下标的行。
  - compute-sanitizer memcheck 0 错误。
  - 某个 token 的 q 是 NaN 时，只能影响这个 token 自己的输出，不能扩散到同一请求的其他 token（它们共用 KV tile），也不能扩散到其他请求。
- **性能**：CUDA graph 回放，和下表的现状逐项对比（`eb_1m/scripts/halops/repro_hca_attn.py`，attention + combine 合计）。

| T | t | 现状 FlashMLA | 目标 |
|---:|---:|---:|---:|
| 768（真实 fixture `hca_T768_dev0.pt`） | 4 | 157.7 µs | ≤ 90 µs |
| 256 | 8 | 62.7 µs | ≤ 35 µs |
| 512 | 8 | 107.3 µs | ≤ 55 µs |
| 1536 | 8 | 295.3 µs | ≤ 150 µs |
| 128 | 4 / 8 | 30.9 / 30.6 µs | ≤ 22 µs |
| 64 | 4 | 25.0 µs | 不慢于现状 |

- 和前面几个算子一样，要给出 ncu 分析：DRAM 字节数、tensor pipe 利用率、主要 stall。

## 4. 复现和测试材料

- `eb_1m/scripts/halops/repro_hca_attn.py`：FlashMLA 基线计时。它可以直接改成新 kernel 的对比测试。
  - `python repro_hca_attn.py 768 4` 用合成下标；
  - `python repro_hca_attn.py <fixture.pt>` 用真实下标；
  - KV 内容是随机的，耗时和取值无关。
- `eb_1m/fixtures/hca_attn/hca_T768_dev0.pt`：DSpark 3、C=768、每 rank 192 个请求、110k 上下文。
- `eb_1m/fixtures/hca_attn/hca_T164_dev2.pt`：带 padding 请求。
- 环境：`/root/paddlejob/inference-public/lizhenyun/anaconda3/envs/eb55/bin/python`，里面有 sgl_kernel 的 FlashMLA 作为参考实现。

## 5. 预期收益

attention 降到目标值后，DSpark 3 C=768 每 step 省约 2.5 ms，OTPS 预计 +5%～+7%；DSpark 7 C=128 每 step 省约 1 ms，OTPS 预计 +3%～+4%。接入只改 `deepseek_v4_backend.py` 的 decode 调用：HCA 层、verify、t ≥ 2 时走新 kernel，其他情况保持 FlashMLA。
