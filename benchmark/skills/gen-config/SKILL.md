---
name: gen-config
description: Generate benchmark config YAML for the HALOps kernel benchmark (baidu/PaddlePaddle/HALOps, benchmark/configs/<op>/). Use this whenever the user wants to add, create, or scaffold a benchmark config for a scenario such as decode_gqa, prefill_dense, prefill_chunked_gqa, dense_gemm, group_masked, or group_contiguous — including phrases like "给 decode 加个 FP8 配置", "新建一个 gemm benchmark yaml", "add a benchmark config", "写个 benchmark 配置", or any request to produce a runnable .yaml under benchmark/configs. The skill reads the benchmark source as the source of truth, asks for the key fields, infers the rest, and checks constraints before writing.
---

# HALOps Benchmark Config Generator

生成 HALOps kernel benchmark 的配置 YAML。目标是产出一份**可直接运行、字段合法、满足 shape 约束**的配置，放到 `benchmark/configs/<op>/` 下。

代码是唯一事实来源。字段默认值、支持的 backend、shape 约束都可能随代码演进，因此每次生成前都要读当前代码确认，不要凭记忆或本文档的示例值直接下结论。

## 何时使用

用户希望为某个场景新增/生成 benchmark 配置时。用户通常会给出场景和意图（如“给 decode 加个 FP8 的配置”“加一组 group_masked 的 w4a4 shape”），细节可能不全，由本 skill 补齐。

## 交互原则（面向用户）

用户只需用自然语言表达想法，其余由 skill 完成。据此：

- 用户输入是意图，不是完整字段表。从中提取场景和已给出的参数，不要求用户按 schema 填写。
- 只在关键字段（决定测什么：遍历维度、`dtype`/`mode`、`backends`）缺失且无法安全推断时才追问，一次问清，避免连环提问。
- 次要字段自动推断并在 YAML 中用 `# inferred` 注释标注，供用户一眼核对。
- 读代码、约束核对等都是内部步骤，不需要用户了解或参与；最终只把可确认的 YAML 结果交给用户。

## 关键路径

先定位 benchmark 根目录（通常是 `<repo>/benchmark`）。`op` 即算子类别（文档中也称 kernel），对应 `kernel/<op>/` 目录。

代码事实来源（以此为准）：

- `benchmark.py`：`OP_REGISTRY`（合法的 op，如 `attention`、`gemm`）；`_run_scenarios`（`batch_sizes` 在此展开为标量 `batch_size`）。
- `kernel/<op>/scenarios/__init__.py`：`SCENARIOS`（合法的 scenario 名，即 YAML 的 `scenario`）。
- `kernel/<op>/scenarios/<scenario>.py`：
  - `run_scenario`（或 `run_*`）中的 `cfg[...]` 与 `cfg.get(...)`：该场景读取哪些字段、哪些必选、哪些有默认值、遍历哪些维度（如 `s_kv_list`、`shapes`）。
  - `_get_backends()` 的 registry：该场景**合法的 backend 名**（不同场景不同，不要跨场景套用）。
  - `adapt_*`：某 backend 是否支持所选 dtype（如 FP8 会不会 `raise`）。
  - `build_inputs` / `_build_inputs` / `_make_*`：shape 约束（`assert` 语句）。
- `core/metrics.py`：`GPU_SPECS`（合法的 `gpu_type`，及 FP4 只在 `b200` 有峰值）。

文档索引（人类可读，快速上手；以代码为准）：

- `benchmark/docs/config_schema.md`：各 scenario 的字段，以及**遍历维度参数 / 普通参数**的区分与约束。
- `benchmark/docs/report_guide.md`：指标与报告字段。
- `benchmark/docs/refcheck.md`：正确性校验流程。
- `benchmark/docs/add_test.md`：新增 backend / scenario / kernel 的扩展方式。

## 工作流程

### 1. 确定 op 与 scenario

从用户意图推断 op 和 scenario。读 `benchmark.py` 的 `OP_REGISTRY` 和 `kernel/<op>/scenarios/__init__.py` 的 `SCENARIOS` 确认名字合法。若无法唯一确定，向用户澄清。

### 2. 读代码，列出该场景的字段清单

打开该 scenario 源文件，据 `cfg[...]`（必选）与 `cfg.get(name, default)`（可选，带默认值）整理出字段清单，从 `build_inputs`/`_make_*` 的断言中提取 shape 约束，从 `_get_backends()` 读出合法 backend，并留意各 `adapt_*` 对 dtype 的限制。若 `build_inputs` 没有 `assert`（如 `prefill_dense`），则该场景没有静态 shape 约束，属正常情况，不必强凑。

只读 scenario 文件还不够：遍历维度中的 `batch_sizes` 由 `benchmark.py` 的 `_run_scenarios` 展开后再注入标量 `batch_size`，scenario 内只会看到 `batch_size`。所以列字段时也要读 `benchmark.py`，避免漏掉 `batch_sizes` 这类由入口层处理的字段。

不要跳过这一步：字段默认值、backend 集合、shape 约束都随代码演进，只有读当前代码才能保证准确。

### 3. 补齐字段

- **关键字段向用户追问**：直接决定测什么的字段必须问清，不要替用户假设。包括遍历维度（`s_kv_list`、`batch_sizes`、`shapes`、`prefix_len_list`/`chunk_len_list` 等）、`dtype`/`mode`、`backends`，以及决定 shape 的 `num_q_heads`/`num_kv_heads`/`head_dim`。若用户未给出 head 几何，可参考同族现有配置推断并标注 `# inferred`，但要在确认环节明确提示这是推断值。
- **次要字段自动推断并标注**：`refcheck`、`refcheck_reference`、`cold_l2_cache`、`use_cuda_graph`、`sm_count`、`mtp_steps_list`、`cache_quant_type` 等可参考同场景现有配置或代码默认值推断。生成时用 YAML 注释标注哪些是推断值（如 `# inferred`）。
- `gpu_type` 通常可省略（默认 `b200`），但当 `mode: w4a4` 或 dtype 为 FP4 时，roofline 指标只有 `b200` 有意义；若用户目标 GPU 非 `b200`，应提示该限制。
- **dtype 字符串写法**：多数场景用 `getattr(torch, cfg["dtype"])` 实例化，因此必须写 torch 属性名（如 `bfloat16`），写 `bf16` 会报错。`fp8`/`e4m3` 这类别名只在个别场景（如 `decode_gqa`）局部支持，勿跨场景套用；不确定时读该 scenario 的 `build_inputs` 确认。

参考同 op 目录下已有 `.yaml` 的写法，保持字段顺序与风格一致。若现有配置不带 `# inferred` 之类注释，在写入仓库前与用户确认是否去掉这些标注以保持风格统一。

### 4. 对照代码核对约束

不依赖外部脚本，直接读代码逐条核对（以当前代码为准，下面是常见项，非全集）：

- `scenario` 必须在该 op 的 `SCENARIOS` 中。
- `backends` 里每个名字都要在该场景 `_get_backends()` registry 中；`halops` 应置于首位。attention 场景 `run_scenario` 会用 `_move_to_front` 强制把它提前，顺序错也安全；grouped GEMM 则以首个成功 backend 为基准，`halops` 不在首位会导致基准静默变成别的 backend，务必置首。
- **dtype ↔ backend 兼容性**：backend 名合法不代表支持所选 dtype。典型：`decode_gqa` 用 FP8 时，`fa2`/`fa4`/`fa4_paged`/`cudnn` 的 `adapt_*` 会直接 `raise`，FP8 应改用 `halops`、`fi`/`fi(trtllm-gen)`/`fi_trtllm`、`golden`。生成 FP8 配置前回到对应 `adapt_*` 确认。
- GEMM：dense 的 `shapes` 为 `[M, N, K]`，grouped 为 `[num_experts, max_m, N, K]`；`K` 的对齐来自 `_make_*` 里按 `mode` 分支的断言（当前 W8A8/W4A4 都归结为 `K % 32 == 0`），以断言为准，不要把 32 当固定常量硬记。
- decode_gqa：`num_q_heads % num_kv_heads == 0`，且 `(1 + max(mtp_steps_list)) × (num_q_heads / num_kv_heads) <= 64`。
- attention：应设置 `batch_sizes`（或 `batch_size`），否则默认只跑 batch=1。
- `gpu_type` 应在 `GPU_SPECS` 中；未知值会静默回退到 `b200`。
- `refcheck_reference` 若指定，应是该场景合法 backend（`golden` 通常可用）。

部分约束只有运行期才暴露（如 grouped contiguous 在 `m` 过小时 `per_expert<=0` 抛错），静态核对无法完全覆盖，需靠下一步冒烟运行确认。

### 5. 展示 → 确认 → 写入

先把生成的 YAML（含 `# inferred` 注释）展示给用户确认。确认后写入 `benchmark/configs/<op>/<name>.yaml`。文件名建议体现关键参数（如 `decode_c16_gqa14_fp8.yaml`）。写入后回读一遍确认落盘成功。

### 6. 冒烟验证（可行时）

若当前环境可运行 benchmark（已构建 `halops`、有目标 GPU），跑单 backend 冒烟验证，确认能跑通、能生成产物：

```bash
python benchmark.py --config=<op>/<name>.yaml --backend=halops \
  --dry-run-iters=1 --repeat-iters=1
```

若环境不满足（缺 `halops`、无 GPU 等），明确说明跳过了运行时验证，只完成了静态核对，不要谎称已跑通。
