# 指标与报告字段说明

benchmark 每次运行生成 CSV 结果、Markdown 报告和日志（输出文件布局见 [README 输出](../README.md#输出)）。本文说明 CSV 字段及其在报告中的展示方式。

## 通用指标

attention 和 GEMM 使用相同的指标字段：

| 字段 | 说明 | 单位 |
|---|---|---|
| `median_us` | 多次测量的 kernel 耗时中位数，性能主指标 | μs |
| `std_us` | 多次测量耗时的标准差，反映稳定性 | μs |
| `speedup` | 当前实现相对 halops 的耗时倍数，>1 表示 halops 更快 | 倍 |
| `tflops` | 计算吞吐 | TFLOPS |
| `tb_per_sec` | 访存吞吐 | TB/s |
| `bw_pct` | 访存带宽占 GPU 峰值带宽的比例 | % |
| `arithmetic_intensity` | 计算强度（每字节访存对应的计算量），用于判断计算受限还是访存受限 | FLOP/byte |
| `roofline_tflops` | 由计算强度与 GPU 峰值推出的该用例理论算力上限 | TFLOPS |
| `attainable_pct` | 实测吞吐占该用例 roofline 上限的比例 | % |
| `host_us` | 主机侧观测到的端到端调用耗时中位数，仅供参考 | μs |
| `dispatch_us` | 从 CPU 下发到 GPU kernel 开始执行的延迟，仅供参考 | μs |
| `refcheck_pass` | 与参考实现比对的正确性结论 | — |
| `mismatch_pct` | 输出中超出容差的元素占比 | % |

`speedup` 的基准实现固定为 `halops`。dense GEMM 和 attention 直接以 `halops` 为基准；grouped GEMM 以 `backends` 中首个成功运行的实现为基准，因此配置需将 `halops` 置于 `backends` 首位，否则基准会变成其他实现。

## 场景字段

### Attention

`scenario`、`backend`、`batch_size`、`s_q`、`s_kv`、`num_q_heads`、`num_kv_heads`、`head_dim`、`dtype`、`causal`、通用指标、`case_tag`（测试用例标签，值为 cfg 文件名）、`use_cuda_graph`、`status`。

### GEMM

`scenario`、`backend`、`num_experts`、`max_m`、`m`、`n`、`k`、`mode`、通用指标、`case_tag`、`status`。

`num_experts` 用于 grouped GEMM；`max_m` 在 `group_contiguous` 中有值，但当前 `group_masked` 未写入该字段。dense GEMM 中二者均为空。`status` 的取值为 `OK`、`TIMEOUT`、`OOM` 或 `FAIL`。部分 GEMM 场景在 backend 运行失败时不会写入结果行。

## 报告组成

一份报告由三部分组成：

1. **运行环境头**：来自日志，记录 GPU、驱动、各 backend 版本等，用于追溯结果的运行环境。
2. **按用例分组的明细表**：每个 case 一块，含 scenario 描述行与逐 backend 的指标行。列名为上文「通用指标」字段的简写（如 `median(us)`→`median_us`、`roof_util%`→`attainable_pct`）。backend 名后的 `[!]` 表示 refcheck 未通过；`status` 非 `OK` 时性能列显示 `-`。
3. **汇总**：`Speedup Summary` 按用例列出 halops 指标与各 backend 的加速比；`Summary` 按 backend 统计加速比的均值、分位数、区间分布与问题计数。

示例（节选）：

````
# dense_gemm -- dense_w8a8

```
============================================================
GPU        : <GPU 名>
Backend versions:
  torch         : ...
  halops        : commit ...
============================================================
```

```
scenario: dense_gemm | m=4096 | n=4096 | k=4096 | mode=w8a8-mxfp8
arithmetic intensity: 2730.667 FLOPs/byte  roofline: 4500.00 TFLOPS
backend         median(us)  std(us)  Speedup  TFLOPS   TB/s   BW%  roof_util%  host(us)  disp(us)  mismatch%  status
--------------------------------------------------------------------------------------------------------------------
halops          53.632    7.514    1.00x  2562.63  0.939 12.2%      57.0%    56.592     0.000      0.0%      OK
fi              61.200    8.100    1.14x  2245.10  0.822 10.7%      49.9%    64.100     0.000      0.0%      OK
```

## Speedup Summary
| config | m | n | k | halops median(us) | halops TFLOPS | halops BW% | halops host(us) | fi |
|---|---|---|---|---|---|---|---|---|
| mode=w8a8-mxfp8 | 4096 | 4096 | 4096 | 53.632 | 2562.63 | 12.2% | 56.592 | 1.14x |

## Summary
| backend | total | valid | mean | P25/P50/P75/P90 | distribution | issues |
|---|---|---|---|---|---|---|
| halops | 7 | 7(100%) | 1.00x | 1.00x/1.00x/1.00x/1.00x | <1:0, 1-2:7, 2-3:0, 3-5:0, >5:0 | - |
| fi | 7 | 7(100%) | 1.12x | 1.05x/1.14x/1.18x/1.22x | <1:0, 1-2:7, 2-3:0, 3-5:0, >5:0 | - |
````

GEMM 报告按 `m`、`n`、`k` 分组；不同配置得到相同 `m`、`n`、`k` 时会并入同一明细分组。

> 仅用已有 CSV 重新渲染报告的命令见 [README 输出](../README.md#输出)。直接生成的文件名为 `{scenario}_{case_tag}.md`，不保留原 CSV 文件名中的时间戳或 backend 后缀，同名报告会被覆盖。
