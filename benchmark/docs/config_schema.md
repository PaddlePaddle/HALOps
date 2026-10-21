# YAML 配置说明

配置文件通常位于 `configs/<op>/`，其中 `<op>` 必须是 `benchmark.py` 中已注册的算子类别。当前支持 `attention` 和 `gemm`。

## 加载规则

- `scenario` 必须在对应算子类别的 `SCENARIOS` 中注册。
- `batch_sizes` 由入口程序逐项展开为 `batch_size`；两者均未配置时使用 `1`。
- 自动发现使用递归扫描，但仅接收直接父目录名为已注册算子类别的 `.yaml` 或 `.yml` 文件。
- 裸文件名仅在 `<configs_dir>/<op>/` 的直接子目录中匹配；带目录的相对路径基于 `--configs-dir` 解析；绝对路径不受 `--configs-dir` 限制。所有显式路径均根据直接父目录名识别算子类别。

`timing_method`、`dry_run_iters`、`repeat_iters`、`use_subprocess`、`case_tag`、`output_path` 等字段由命令行或入口程序统一控制，YAML 中的同名字段会被忽略；具体见 [README 常用选项](../README.md#常用选项)。

## 通用字段

按用途分为四类。

**基本**

| 字段 | 必选 | 默认 | 说明 |
|---|---|---|---|
| `scenario` | 是 | — | 测试场景名称 |
| `backends` | 是 | — | 参与性能测试的 backend 列表；应包含 `halops` 并置于首位，以统一 speedup 口径 |

**正确性校验**

`refcheck`、`refcheck_reference`、`refcheck_atol`、`refcheck_rtol` 的取值与默认值见 [正确性校验机制](refcheck.md#配置)。

**测量控制**

| 字段 | 必选 | 默认 | 说明 |
|---|---|---|---|
| `cold_l2_cache` | 否 | `true` | 每次测量前清空 GPU L2 cache |
| `use_cuda_graph` | 否 | `false` | 仅 `cupti` 计时方式生效 |
| `timeout` | 否 | `120` | subprocess 超时时间，单位为秒 |

**环境与输入**

| 字段 | 必选 | 默认 | 说明 |
|---|---|---|---|
| `gpu_type` | 否 | `b200` | 选择 roofline 和带宽峰值参数，可选 `a100`/`h100`/`h100_pcie`/`h200`/`b200`；未知值回退 `b200`，不自动检测实际 GPU |
| `input_scale` | 否 | `2.0`/`1.0` | 随机输入的缩放系数（decode 默认 `2.0`，prefill 默认 `1.0`） |

## 场景字段

场景参数分两类：

- **遍历维度参数**：以列表填写。程序对各列表按组合（笛卡尔积）逐一展开，每个组合生成并运行一个 case；填的值越多，扫描的测试点越多。
- **普通参数**：填单个固定值，对本场景所有 case 共用。

例如 `decode_gqa` 配置 `batch_sizes: [1, 16]`、`s_kv_list: [1024, 4096]`、`mtp_steps_list: [0]`，会展开为 2 × 2 × 1 = 4 个 case：`(1, 1024, 0)`、`(1, 4096, 0)`、`(16, 1024, 0)`、`(16, 4096, 0)`；每个 case 再对 `backends` 里的所有实现各测一遍。想扫更多测试点，就往对应列表里多加值。

## Attention 场景

### `decode_gqa`

遍历维度参数：

- `batch_sizes`（默认 `[1]`）：batch size 列表。
- `s_kv_list`（必选）：KV 序列长度列表。
- `mtp_steps_list`（默认 `[0]`）：MTP 步数列表，`s_q = 1 + mtp_steps`。

普通参数：

- `num_q_heads`、`num_kv_heads`、`head_dim`、`dtype`（必选）。FP8 dtype 可写为 `float8_e4m3fn`、`e4m3` 或 `fp8`。
- `use_write_cache`（默认 `false`）：是否使用 write-cache kernel 构造输入。
- `cache_quant_type`：FP8 默认 `block_wise_fp8`，其他 dtype 默认 `none`。

约束：`num_q_heads % num_kv_heads == 0`，且 `(1 + max(mtp_steps_list)) × (num_q_heads / num_kv_heads) <= 64`。

### `prefill_dense`

遍历维度参数：

- `batch_sizes`（默认 `[1]`）：batch size 列表。
- `s_kv_list`（必选）：序列长度列表；该场景中 `s_q = s_kv`。

普通参数：

- `num_q_heads`、`num_kv_heads`、`head_dim`、`dtype`、`causal`（必选）。

### `prefill_chunked_gqa`

遍历维度参数：

- `batch_sizes`（默认 `[1]`）：batch size 列表。
- `prefix_len_list`、`chunk_len_list`（必选）：前缀长度和 chunk 长度列表。

普通参数：

- `num_q_heads`、`num_kv_heads`、`head_dim`、`dtype`（必选）。
- `cache_quant_type`（默认 `none`）。

该场景默认不执行 refcheck；需要校验时应显式设置 `refcheck: true`。

## GEMM 场景

### `dense_gemm`

遍历维度参数：

- `shapes`（必选）：`[M, N, K]` 列表，每项生成一个 case。

普通参数：

- `mode`（默认 `w8a8`）：支持 `w8a8` 和 `w4a4-nvfp4`（kernel 也支持 `w4a4-mxfp4`，但 benchmark 暂无对应量化输入生成）。
- `sm_count`（默认 `-1`）：`-1` 表示不限制 SM 数量。

### `group_masked` 和 `group_contiguous`

遍历维度参数：

- `shapes`（必选）：`[num_experts, max_m, N, K]` 列表，每项生成一个 case。

普通参数：

- `mode`（默认 `w8a8`）：支持 `w8a8` 和 `w4a4-nvfp4`（grouped kernel 暂不支持 `w4a4-mxfp4`）。
- `sm_count`（默认 `-1`）。

所有现有 GEMM 场景均要求 `K` 为 `32` 的倍数。
