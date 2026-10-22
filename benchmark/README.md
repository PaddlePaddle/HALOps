# HALOps Benchmark 指南

[English Quick Reference](#english-quick-reference) | [简体中文](#环境准备)

## English Quick Reference

Kernel-level benchmark. Each run executes the same set of cases for every backend listed in the YAML config and writes a CSV, a Markdown report and a log. Results only hold for the stated hardware, software versions, config and measurement method; public performance claims must also pass the correctness check.

Install HALOps first ([Build Guide](../docs/build_guide_EN.md)), then the benchmark dependencies. Reference backends (FlashAttention, FlashInfer, DeepGEMM, SGLang, cuDNN) are not installed by this file; install only what the YAML `backends` list uses. Run all commands from `benchmark/`.

```bash
python -m pip install -r benchmark/requirements.txt

# All backends listed in the YAML
python benchmark.py --config=attention/prefill_chunked_c16_gqa14-1.yaml --gpu=0

# A single backend (must be in the YAML `backends` list; one --config only)
python benchmark.py --config=gemm/dense_w8a8.yaml --backend=halops --gpu=0
```

- `--config` takes a path relative to `configs/` (recommended), a file name unique across `configs/<op>/`, or an absolute path. Bundled configs live in `configs/attention/` and `configs/gemm/`.
- Common options: `--gpu` (default `0`, sets `CUDA_VISIBLE_DEVICES`), `--timing-method` (`profiler` or `cupti`, default `profiler`), `--dry-run-iters` (default `10`), `--repeat-iters` (default `50`), `--refcheck-reference`. See `python benchmark.py --help`.
- Outputs share one timestamp `{ts}`: `results/{ts}_{config}.csv`, `reports/{ts}_{scenario}_{case_tag}.md`, `logs/{ts}_{op}_{config}.log`.
- Speedup is `other_backend_median_us / halops_median_us`; above `1.00x` means HALOps is faster.

The sections below (in Chinese) cover config forms, timing methods, correctness rules and the scenario × backend matrix in detail.

Kernel 级 benchmark。每次运行会按 YAML 中声明的 backend 列表逐个执行同一组 case，并输出 CSV、Markdown 报告与日志。
Benchmark 结果只代表指定硬件、软件版本、配置与测量方法。公开性能结论还必须通过正确性校验。

## 环境准备

先按 [构建指南](../docs/build_guide_CN.md) 安装 HALOps，再安装 Benchmark 框架依赖：

```bash
python -m pip install -r benchmark/requirements.txt
```

FlashAttention、FlashInfer、DeepGEMM、SGLang 和 cuDNN 等参考 backend 不会由该 requirements 文件自动安装。只需安装目标 YAML 的 `backends` 列表实际使用的依赖；版本应在结果记录中保留。SGLang backend 通过 `SGLANG_PATH` 指向本地源码。

后续命令均从 `benchmark/` 目录执行。

## 快速运行

运行 YAML 中声明的全部 backend：

```bash
python benchmark.py --config=attention/prefill_chunked_c16_gqa14-1.yaml --gpu=0
```

一次运行多个配置：

```bash
python benchmark.py \
  --config=attention/prefill_chunked_c16_gqa14-1.yaml gemm/dense_w8a8.yaml \
  --gpu=0
```

只运行单个 backend：

```bash
python benchmark.py \
  --config=gemm/dense_w8a8.yaml \
  --backend=halops \
  --gpu=0
```

`--backend` 必须是 YAML `backends` 中的成员，且只能和一个 `--config` 一起使用。单 backend 模式仍会尝试使用场景内置的 neutral golden reference 做正确性校验；如果只剩一个可运行候选，则明确跳过校验，不应把该结果直接用于公开性能结论。

## 配置与场景

`--config` 支持以下形式：

- `attention/decode_c16_gqa14.yaml`：相对 `configs/` 的路径，推荐使用。
- `decode_c16_gqa14.yaml`：跨 `configs/<op>/` 唯一匹配的文件名。
- 配置文件绝对路径；其父目录名必须是已注册的 `attention` 或 `gemm`。

仓库内置配置位于：

| 目录 | 场景 | 主要配置 |
| --- | --- | --- |
| `configs/attention/` | `prefill_dense` | batch、sequence length、Q/KV head、dtype、causal |
| `configs/attention/` | `prefill_chunked_gqa` | prefix/chunk length、GQA、cache dtype |
| `configs/attention/` | `decode_gqa` | KV length、MTP steps、GQA、BF16/FP8 cache |
| `configs/gemm/` | `dense_gemm` | M/N/K shape、W8A8-MXFP8/W4A4、SM 数 |
| `configs/gemm/` | `group_masked` / `group_contiguous` | expert 数、per-expert M、N/K、精度模式 |

YAML 负责 workload、backend 和 cache/CUDA Graph 策略；命令行始终覆盖 `timing_method`、`dry_run_iters`、`repeat_iters` 和 subprocess 设置。输出路径与 `case_tag` 由 CLI 根据时间戳和配置文件名生成，YAML 中的同名字段会被忽略。

## 常用选项

| 选项 | 默认值 | 说明 |
| --- | --- | --- |
| `--gpu` | `0` | 写入 `CUDA_VISIBLE_DEVICES`；支持设备列表字符串 |
| `--timing-method` | `profiler` | `profiler` 或 `cupti`；当前 CLI 不暴露内部 `event` 方法 |
| `--dry-run-iters` | `10` | 测量前的 warmup 次数 |
| `--repeat-iters` | `50` | 正式测量次数 |
| `--backend` | 无 | 只测一个 YAML backend |
| `--no-subprocess` | 关闭 | 禁用默认的 per-backend 子进程隔离 |
| `--refcheck-reference` | YAML 值 | 覆盖正确性参考 backend，例如 `golden` |

查看当前 checkout 的完整参数：

```bash
python benchmark.py --help
```

`profiler` 方法先用 CUDA Event 收集 host-perceived latency，再以 `torch.profiler` 汇总每次调用内的 CUDA kernel duration，CSV 的 `median_us` 与 TFLOPS 使用后者。`cupti` 使用 CUPTI activity 时间戳，并支持 CUDA Graph。若 YAML 开启 CUDA Graph 但选择 `profiler`，工具会发出警告并忽略 Graph 设置。

## 正确性与数据口径

- 场景默认按 YAML 的 `refcheck` 和 `refcheck_reference` 执行跨 backend 比对；不可用的首选 reference 会退回到第一个可运行候选。
- `refcheck_atol`、`refcheck_rtol` 可在 YAML 中配置，默认均为 `1e-2`。
- 报告的 backend summary 只纳入 `status=OK` 且 mismatch 不超过 10% 的样本；`[!]` 表示正确性校验未通过。原始 CSV 仍保留失败或 mismatch 的行。
- 加速比为 `other_backend_median_us / halops_median_us`，大于 `1.00x` 表示 HALOps 更快。
- `cold_l2_cache=true` 时当前实现每轮写入 64 MiB flush buffer；这不保证完整驱逐所有 SM100 GPU 的 L2。
- Benchmark kernel latency 不包括输入构造、量化、backend plan/JIT 和 Python 端到端开销，除非对应场景把这些操作显式放进被测函数。

## 输出

每次运行使用同一时间戳 `{ts}` 命名文件，避免覆盖历史结果。指定 `--backend=<name>` 时会附加 backend 后缀：

| 路径 | 内容 |
| --- | --- |
| `results/{ts}_{config}.csv` | 原始逐 case 指标、状态与 refcheck 字段 |
| `reports/{ts}_{scenario}_{case_tag}.md` | 汇总、加速比分布和逐 case 表格 |
| `logs/{ts}_{op}_{config}.log` | 运行环境头、backend 版本和完整 stdout/stderr |

只有 CSV 时可以重新生成 Markdown 报告：

```bash
python core/report.py \
  results/{ts}_{config}.csv \
  --out reports \
  --log logs/{ts}_{op}_{config}.log
```

`benchmark/results/`、`reports/` 和 `logs/` 默认被 Git 忽略。需要在项目文档公开结果时，应只发布经过复核的摘要，并记录配置、commit、GPU、驱动、PyTorch/CUDA、backend 版本、测量选项和正确性状态。

## 场景 × Backend 支持矩阵

各场景当前注册（`_get_backends`）的对比 backend：

| 场景 | halops | golden | fa2 | fa4 | fi | deepgemm | cudnn | sglang_triton |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `decode_gqa` | ✅ | ✅ | ✅ | ✅¹ | ✅² | — | ✅ | — |
| `prefill_dense` | ✅ | ✅ | ✅ | ✅ | ✅³ | — | ✅ | ✅ |
| `prefill_chunked_gqa` | ✅ | ✅ | ✅ | ✅ | ✅⁴ | — | — | — |
| `dense_gemm` | ✅ | ✅ | — | — | ✅ | ✅ | — | — |
| `group_masked` | ✅ | ✅ | — | — | ✅ | ✅ | — | — |
| `group_contiguous` | ✅ | ✅ | — | — | ✅ | ✅ | — | — |

- `halops` 为默认基准，`golden` 为正确性参考。
- ¹ `decode_gqa` 还提供 `fa4_paged`。
- ² `decode_gqa` 的 `fi` 有多个后端变体：`fi(auto)`、`fi(fa2)`、`fi(fa3)`、`fi(trtllm-gen)`，以及独立的 `fi_trtllm`。
- ³ `prefill_dense` 的 `fi` 变体：`fi(auto)`、`fi(trtllm-gen)`。
- ⁴ `prefill_chunked_gqa` 的 `fi` 变体：`fi(auto)`、`fi(cudnn)`、`fi(trtllm-gen)`。
- 实际可用的 backend 还取决于运行环境是否安装了对应依赖。

## 相关文档

- [配置（YAML）字段说明](docs/config_schema.md)：各场景的遍历维度参数、普通参数与约束。
- [指标与报告字段说明](docs/report_guide.md)：CSV 指标字段、报告组成与重新生成方式。
- [正确性校验机制](docs/refcheck.md)：refcheck 的流程与比较规则。
- [新增测试指南](docs/add_test.md)：新增 backend / scenario / kernel 的扩展方式。
