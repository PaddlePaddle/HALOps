# 新增测试指南

benchmark 的测试按三级组织，与目录一一对应：

- **kernel（算子类别）**：一类算子，对应 `kernel/<op>/` 目录；`op` 是 `benchmark.py` 中 `OP_REGISTRY` 的键（当前为 `attention`、`gemm`）。
- **scenario（测试场景）**：某类算子下的一种测法，对应 `kernel/<op>/scenarios/__init__.py` 中 `SCENARIOS` 的键，即 YAML 的 `scenario`。
- **backend（对比实现）**：scenario 内 `_get_backends()` 注册的实现，即 YAML 的 `backends`。

新增内容时应优先复用现有层级。目录结构：

```text
benchmark/
├── benchmark.py              # OP_REGISTRY：注册所有 kernel（算子类别）
├── configs/<op>/*.yaml       # 各 kernel 的测试配置
└── kernel/<op>/
    ├── scenarios/
    │   ├── __init__.py       # SCENARIOS：注册该 kernel 的所有 scenario
    │   └── <scenario>.py     # run_scenario：输入构造 + 遍历 + 调用 runner
    └── backends/
        └── <backend>.py      # 各对比实现的运行函数
```

一个 kernel 测试由四部分组成：

```text
kernel 测试
├── scenario   定义输入构造、遍历维度、测试流程
├── backend    各对比实现（含 adapt_fn + run_fn）
├── config     YAML，指定遍历维度、backends 等
└── 注册项      OP_REGISTRY（kernel）/ SCENARIOS（scenario）/ _get_backends（backend）
```

## 新增 backend

一个 backend 由两个函数组成：

- **adapt_fn**：把 scenario 构造的统一输入，转换成该 backend 需要的入参。
- **run_fn**：接收转换后的入参，实际调用该实现并返回输出张量（供 refcheck 比较）。

步骤：

1. 在 `kernel/<op>/backends/` 中实现 `run_fn`。
2. 在 scenario 中实现 `adapt_fn`。
3. 在 `_get_backends()` 中注册 `"<name>": (adapt_fn, run_fn)`。
4. 将名称加入 YAML 的 `backends` 列表。

`golden` 通常作为正确性参考；`halops` 通常作为 speedup 基准，应置于 `backends` 首位以避免基准顺序影响结果。

## 新增 scenario + yaml 配置

1. 在 `kernel/<op>/scenarios/` 中新增模块，提供 `run_scenario(cfg)`：构造输入、注册 backend、遍历 case、调用 runner 跑测试、写入 CSV。
2. 每个 case 用 `core/runner.py` 封装好的函数跑两类测试，二者相互独立：
   - **性能测试**：`run_bench_for_backend()` 对单个 backend 计时，得到耗时等性能指标。
   - **正确性测试**：`run_refcheck_cross_backends()` 做 refcheck，得到是否通过与误差（流程见 [refcheck.md](refcheck.md)）。
3. 用指标函数与结果类型写入 CSV：attention 用 `compute_metrics()`、`AttnRowResult`、`append_attention_csv()`；GEMM 用 `compute_gemm_metrics()`、`GemmRowResult`、`append_gemm_csv()`。
4. 在 `kernel/<op>/scenarios/__init__.py` 的 `SCENARIOS` 中注册场景。
5. 在 `configs/<op>/` 中增加对应 YAML。

### 遍历机制

case 的遍历是 scenario 需要实现的核心逻辑。配置中的遍历维度参数以列表给出（如 `s_kv_list`、`mtp_steps_list`、`shapes`），`run_scenario` 对这些列表做笛卡尔积，每个组合生成一个 case，逐个跑性能测试和正确性测试。`run_scenario` 只需遍历自己的维度；`batch_sizes` 已由入口层展开为标量 `batch_size` 传入，不要在 scenario 内重复遍历（配置侧的填写方式见 [config_schema.md](config_schema.md)）。

新增 GEMM 场景时，还需在 `core/report.py` 的 `_is_gemm_scenario()` 中注册场景名称，否则报告会按 attention 字段解析。若新场景需要新的 CSV 字段或分组维度，应同步修改 `core/output.py` 和 `core/report.py`。

## 新增算子类别

1. 新建 `kernel/<op>/scenarios/` 和 `kernel/<op>/backends/`。
2. 在 scenario 包中导出 `SCENARIOS`。
3. 在 `benchmark.py` 中导入该注册表并加入 `OP_REGISTRY`。
4. 新建 `configs/<op>/`。目录名必须与 `OP_REGISTRY` 的键一致，因为入口程序根据配置文件的直接父目录名识别算子类别。
5. 为新算子增加结果类型、CSV 写入和报告渲染支持；现有报告仅区分 attention 与 GEMM。

## 验证

先使用较小配置验证单个 backend，再运行完整配置：

```bash
python benchmark.py --config=<op>/<config>.yaml --backend=<name> \
  --dry-run-iters=1 --repeat-iters=1
python benchmark.py --config=<op>/<config>.yaml
```

检查 CSV、Markdown 报告和日志是否生成，并确认 `status`、refcheck 和 speedup 符合预期。外部 backend 还需在运行环境中安装对应依赖。
