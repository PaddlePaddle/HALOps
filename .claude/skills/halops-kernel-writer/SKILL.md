---
name: halops-kernel-writer
description: 在 HALOps 里新增或修改 CUDA kernel、launcher 和 Python 接口。先保证正确性和性能；代码规范由 hook 和 tools/agent/gate.py 机械检查，写完交给格式刷处理剩余问题。用于"写一个 xx 算子""给 xx 加一个 kernel""优化 xx 的性能"等请求。
---

# HALOps kernel writer

规则不写在这里。规则在 `tools/lint/rules.py`，hook 和 gate 会在写代码的过程中和结束时检查。本文件只说明流程。

## 流程

1. **先找能复用的代码。** 开工前列出 `csrc/common/` 的头文件，PTX 封装、kernel launch、PDL、stream、`HALOPS_CUDA_CHECK`、量化和激活 helper 都先在这里找。lint 的 reuse-* 规则按函数名和 PTX 字符串比对，重复定义会被拦下。
2. **按现有结构放代码。** kernel 和 launcher 放在 `csrc/kernel/<模块>/`，namespace 用文件所在模块已有的 namespace，绑定只加在 `csrc/common/cpp_extensions.cc`，Python 接口放在 `halops/<模块>/`。参数检查放在 C++ 入口，Python 接口不再检查、不分配输出以外的张量。
3. **边写边看 hook。** 每次 Edit/Write 之后，PostToolUse hook 回报这个文件新增行上的 finding。error 级的当场改；改不了的记下原因，写进最后的评论。
4. **测试。** 用例放在 `tests/test_<算子>.py`。随机种子由 `tests/conftest.py` 统一设置，需要独立随机流时用局部 `torch.Generator`；容差用具名常量。
5. **性能。** 用 `benchmark/` 下的脚本测 before/after，记录 GPU、shape 和时间。
6. **结束。** Stop hook 会对本会话改过的 HALOps checkout 跑 lint，仍有 error 时要求继续处理一次。

## 交付

提交但不 push，然后运行 `python3 tools/agent/gate.py <base> --build --tests`。评论写：正确性结果、性能数据、gate 结果、剩余的 lint finding 和原因。剩余的规范问题交给格式刷（halops-format-brush）。

## 不做的事

- 不改 `tools/lint`、`tools/agent`、`.claude`、`.github`、`.pre-commit-config.yaml`、`tests/conftest.py`、`pyproject.toml`、`setup.py`、`requirements.txt`、`3rdparty/`。
- 不加 `halops-lint: allow(...)` 豁免，不改 `baseline.json`。
- 不新增 CUDAExtension，不新增绑定文件，不读取环境变量。
