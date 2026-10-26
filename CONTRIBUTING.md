# 贡献指南 / Contributing to HALOps

[简体中文](#简体中文) | [English](#english)

## 简体中文

感谢参与 HALOps 开发。本文说明从准备环境到提交评审的最低要求,实现、文档、正确性测试和性能结论应使用一致的接口与硬件约束。

### 开始之前

提交较大的功能、接口或架构变更前，请先创建或关联议题，说明使用场景、设计方案、兼容性影响和验收方式，避免在实现完成后才讨论方向。缺陷修复应给出最小复现；性能问题应同时给出 workload、环境和测量方法。

开发环境准备：

```bash
git submodule update --init --recursive
python -m pip install -r requirements.txt
python -m pip install pre-commit
pre-commit install
```

CUDA、PyTorch、编译器和软硬件环境要求见 [构建指南](docs/build_guide_CN.md)。Python 规则位于 [`pyproject.toml`](pyproject.toml)，CUDA/C++ 格式位于 [`.clang-format`](.clang-format)，统一检查入口位于 [`.pre-commit-config.yaml`](.pre-commit-config.yaml)。

### 开发约定

CUDA/C++ 编码规范见 [代码规范](docs/code_standards.md)。

- 每个变更只解决一个明确问题；不要混入无关重构、机械格式化或生成文件。
- 优先复用现有目录结构、接口和调度模式。新增抽象应能消除实际复杂度或重复。
- Python wrapper、PyBind 声明与 CUDA/C++ 实现的参数顺序、默认值和语义必须一致。
- 新增公开 API 时，补充类型标注、docstring、顶层导出、README 算子表。
- 新增 kernel 时，覆盖正常 shape、边界 shape、非法输入及其支持的精度模式；错误信息应能定位违反的约束。
- 不提交 `.so`、wheel、profiling capture、日志或 Benchmark 自动输出目录。经过复核且用于项目文档的精选结果除外。

### 代码质量与测试

提交前只需在兼容 `sm_100f` 的 GPU 上运行与本次改动直接相关的单元测试，例如：

```bash
python tests/test_dense_gemm.py
```

格式和其他基础检查由 GitHub Actions（`.github/workflows/lint.yml`）覆盖：PR 只检查其改动的文件，合入 `main` 时执行 `pre-commit run --all-files` 以保持基线干净；GPU 单元测试和性能回归需要 `sm_100f` 机器，由维护者在内部 CI 运行。CI 报错时，根据日志对本 PR 修改的文件运行对应检查并修复；修复缺陷时应增加能在修复前复现问题的回归用例。没有 GPU 时，在变更说明中注明未运行的测试。

编写单元测试时遵循以下约定：

- 文件命名为 `tests/test_<api>.py`，公共参考实现和构造函数放在 `tests/utils/`。
- 使用模块级 `test_*` 函数；shape、配置和非法输入用 `@pytest.mark.parametrize` 展开，不在单个用例内循环。
- 随机种子、默认 CUDA 设备和无 `sm_100f` GPU 时的跳过由 `tests/conftest.py` 统一处理，测试文件中不再设置；不在 import 时分配 CUDA tensor，改用 fixture。
- 数值比较使用 `torch.testing.assert_close`，按位比较使用 `torch.equal`；容差定义为具名常量，GEMM 使用 `tests/utils/gemm_utils.py` 中的 `W8A8_MXFP8_TOL` / `W4A4_TOL`。
- 异常用 `pytest.raises(<类型>, match=...)` 断言；测试中不使用 `print`。

### 性能变更

涉及性能的变更应先通过正确性校验，再按 [Benchmark 指南](benchmark/README.md) 使用统一工具自测。变更前后应使用相同的 GPU、软件环境、YAML 和测量参数，并在 Pull Request 中说明测试命令、配置及结果。

```bash
cd benchmark
python benchmark.py --config=gemm/dense_w8a8.yaml --gpu=0
```

### 提交与评审

提交信息应简洁说明“做了什么”，并在正文解释“为什么”。标题建议使用以下格式：

```text
<type>[(<scope>)]: <short description>
```

- `type`：必填，取 `feat`、`fix`、`perf`、`docs`、`test`、`refactor`、`build` 等。
- `scope`：可选，为算子或模块名；改动不针对特定模块时省略，连同括号一起去掉。

例如：

```text
perf(dense_gemm): reduce epilogue smem traffic
docs: update build guide
```

创建 Pull Request 时请：

1. 关联对应议题或任务，并描述问题、方案、影响范围和兼容性。
2. 列出实际执行的检查与测试；未运行的项目需说明原因。
3. 填写 PR 模板中的精度影响与性能影响；性能变更附 Benchmark 命令、配置和自测结果。
4. 保持分支可评审；根据反馈追加提交，合入前再整理历史。

评审重点包括接口兼容性、正确性、边界条件、性能回退、可维护性和文档完整性。作者应回应所有未解决意见；重大设计变化应同步更新议题和文档。

### Pull Request 检查项

- [ ] 变更已关联议题或任务，且范围聚焦。
- [ ] Python、PyBind 与 CUDA/C++ 接口保持一致。
- [ ] 已运行与改动直接相关的单元测试，或明确说明未运行项。
- [ ] CI 报错时已根据日志完成检查和修复。
- [ ] 性能变更已完成 Benchmark 自测，并附命令、配置和结果。
- [ ] 新增或变更的公开行为已更新 README 和对应文档。
- [ ] 未包含生成产物、敏感信息或无关 profiling 文件。

## English

Thank you for contributing to HALOps. This guide describes the minimum requirements from setting up the environment to code review. Implementation, documentation, correctness tests and performance claims should share the same interfaces and hardware constraints.

### Before You Start

Before submitting a large feature, interface or architecture change, open or link an issue describing the use case, design, compatibility impact and acceptance criteria, so the direction is agreed on before implementation. Bug fixes should include a minimal reproduction; performance issues should include the workload, environment and measurement method.

Set up the development environment:

```bash
git submodule update --init --recursive
python -m pip install -r requirements.txt
python -m pip install pre-commit
pre-commit install
```

See the [Build Guide](docs/build_guide_EN.md) for CUDA, PyTorch, compiler and hardware requirements. Python rules live in [`pyproject.toml`](pyproject.toml), CUDA/C++ formatting in [`.clang-format`](.clang-format), and all checks run through [`.pre-commit-config.yaml`](.pre-commit-config.yaml).

### Development Conventions

CUDA/C++ coding rules are listed in the [code standards](docs/code_standards.md) (in Chinese).

- Each change addresses one clear problem; do not mix in unrelated refactoring, mechanical reformatting or generated files.
- Reuse the existing directory layout, interfaces and dispatch patterns. A new abstraction should remove real complexity or duplication.
- The Python wrapper, the PyBind declaration and the CUDA/C++ implementation must agree on argument order, defaults and semantics.
- A new public API needs type hints, a docstring, a top-level export and an entry in the README operator table.
- A new kernel needs tests for regular shapes, boundary shapes, invalid inputs and every supported precision mode; error messages should identify the violated constraint.
- Do not commit `.so` files, wheels, profiling captures, logs or benchmark output directories, except reviewed results used in project documentation.

### Code Quality and Tests

Before submitting, run the unit tests directly related to your change on a GPU compatible with `sm_100f`, for example:

```bash
python tests/test_dense_gemm.py
```

Formatting and basic checks run in GitHub Actions (`.github/workflows/lint.yml`): pull requests check only the files they change, while pushes to `main` run `pre-commit run --all-files` so the baseline stays clean. GPU unit tests and performance regression need an `sm_100f` machine and are run by maintainers in internal CI. When CI fails, run the corresponding check on the files changed by your PR and fix them. Bug fixes should add a regression test that reproduces the problem before the fix. If you have no GPU, list the tests you did not run in the PR description.

Unit test conventions:

- Name files `tests/test_<api>.py`; put shared reference implementations and input builders in `tests/utils/`.
- Use module-level `test_*` functions. Expand shapes, configs and invalid inputs with `@pytest.mark.parametrize` instead of looping inside one test.
- Seeding, the default CUDA device and skipping on machines without an `sm_100f` GPU are handled by `tests/conftest.py`; do not set them in test files. Do not allocate CUDA tensors at import time; use fixtures.
- Compare values with `torch.testing.assert_close` and bit-exact results with `torch.equal`. Define tolerances as named constants; GEMM tests use `W8A8_MXFP8_TOL` / `W4A4_TOL` from `tests/utils/gemm_utils.py`.
- Assert exceptions with `pytest.raises(<type>, match=...)`. Do not use `print` in tests.

### Performance Changes

A performance change must first pass correctness checks, then be measured with the shared tooling described in the [Benchmark Guide](benchmark/README.md#english-quick-reference). Use the same GPU, software environment, YAML and measurement options before and after the change, and include the command, config and results in the pull request.

```bash
cd benchmark
python benchmark.py --config=gemm/dense_w8a8.yaml --gpu=0
```

### Commits and Review

A commit message states what changed in the title and why in the body. Use this title format:

```text
<type>[(<scope>)]: <short description>
```

- `type`: required; one of `feat`, `fix`, `perf`, `docs`, `test`, `refactor`, `build`, etc.
- `scope`: optional; an operator or module name. Omit it, together with the parentheses, when the change is not specific to one module.

For example:

```text
perf(dense_gemm): reduce epilogue smem traffic
docs: update build guide
```

When opening a pull request:

1. Link the issue or task and describe the problem, solution, scope of impact and compatibility.
2. List the checks and tests you actually ran, and explain any you did not run.
3. Fill in the accuracy and performance impact sections of the PR template; performance changes include the benchmark command, config and results.
4. Keep the branch reviewable; push follow-up commits in response to review and clean up history before merging.

Review focuses on interface compatibility, correctness, edge cases, performance regressions, maintainability and documentation. Authors should respond to every open comment; significant design changes should be reflected in the issue and documentation.

### Pull Request Checklist

- [ ] The change is linked to an issue or task and is focused in scope.
- [ ] The Python, PyBind and CUDA/C++ interfaces are consistent.
- [ ] Unit tests directly related to the change were run, or the skipped ones are listed.
- [ ] CI failures were investigated and fixed.
- [ ] Performance changes include benchmark command, config and results.
- [ ] New or changed public behavior is reflected in the README and docs.
- [ ] No generated artifacts, secrets or unrelated profiling files are included.
