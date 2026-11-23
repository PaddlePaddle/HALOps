---
name: halops-format-brush
description: 把 HALOps 里已经能跑通、性能达标的代码改到符合仓库硬规则（tools/lint）。只修 lint 报出的问题，不改算法和数值行为，验收以 tools/agent/gate.py 的结果为准。用于"格式刷""按 lint 修一遍""把这段代码改规范"等请求。
hooks:
  PreToolUse:
    - matcher: "Edit|Write|MultiEdit|NotebookEdit|Bash"
      hooks:
        - type: command
          command: python3 -I "${CLAUDE_PROJECT_DIR}/tools/agent/hooks.py" pre --role brush
---

# HALOps 格式刷

规则不写在这里。规则在 `tools/lint/rules.py`，入口是 `tools/lint/halops_lint.py`，验收是 `tools/agent/gate.py`。本文件只说明流程和每条规则的修法。

## 输入

- **base**：本次修改的起点提交。gate 的检查都相对它，并且使用 base 里的 `tools/lint`，所以改工作区里的 `tools/` 不起作用。
- **待修清单**：issue 指定的规则和文件，或者 `python3 tools/lint/halops_lint.py --diff <lint-base>` 的 error 级输出。

## 流程

1. 列出待修的 finding（文件:行、规则），按文件分组。
2. 按下面的修法逐条修。下面没有的修法不要自己发明，记为"无法修复"。
3. 每次 Edit/Write 之后，PostToolUse hook 会回报这个文件新增行上的 finding。回报清空后再改下一个文件。
4. 全部改完后提交（不 push），运行 issue 给出的 gate 命令，一般是：
   `python3 tools/agent/gate.py <base> --scope --build --sass <base 的 so 目录> --tests --junit-base <base 的 junit>`
5. gate 不通过就回到第 2 步。同一个问题连续三次不通过，或者属于"无法修复"，停下来写进评论。

## 修法

| 规则 | 修法 |
|---|---|
| anon-namespace | 删掉 `namespace {` 和配对的 `}`，内容留在文件已有的具名 namespace 里。删除后 gate 的 tree 步骤会报出同名定义（ns-collision）。 |
| ns-static | `.cu`/`.cc` 里直接删 `static`。`.cuh`/`.h` 里普通函数改成 `inline`，`static constexpr` 变量改成 `inline constexpr`，constexpr 函数和 `CUTLASS_DEVICE`/`__forceinline__` 函数直接删 `static`。头文件里不能留下非 inline 的函数定义，否则多个 TU 包含时会重复定义。 |
| ns-collision | 两个 TU 定义了同名同签名的函数。实现完全相同时，移到 `csrc/common/` 里合适的头文件（优先用已有头文件），两边改为 include。实现不同时列进评论，不要靠改名了事。 |
| reuse-* | 换成 lint 消息里点名的 common 接口。如果接口的语义覆盖不了（例如需要 cluster launch 属性而 `LaunchKernelWithPdl` 不支持），不要硬改，记为"无法修复"并写明缺什么。 |
| getenv | 删掉环境变量开关和它控制的分支，保留环境变量未设置时走的路径。 |
| debug-comment, comment-density | 删掉记录调试过程、实验数据、历史版本的注释，保留说明不变量和约束的注释。 |
| test-seed | 删掉全局 seed。需要固定随机数时改用局部 `torch.Generator(device=...).manual_seed(<原来的值>)`，并传给生成随机数的调用，让生成的数据和原来一致。 |
| new-namespace, binding-location, new-ext, new-extension | 涉及接口和目录结构，按 lint 消息改；改动会影响 Python 接口时先列进评论，不改。 |

## 不做的事

- 不改 `tools/lint`、`tools/agent`、`.claude`、`.github`、`.pre-commit-config.yaml`、`tests/conftest.py`、`pyproject.toml`、`setup.py`、`requirements.txt`、`3rdparty/`。PreToolUse hook 会拦截。
- 不加 `halops-lint: allow(...)` 豁免，不改 `baseline.json`。豁免由维护者决定。
- 不改测试的断言、容差、skip、parametrize。
- 不顺手重构，不改算法，不调性能参数。gate 的 sass 步骤报出指令变化的 kernel，要在评论里逐个说明原因。
- 不 push，不用 `--no-verify`。

## 交付

评论写：修了哪些 finding（规则和数量）、gate 每一步的结果、SASS 对比（identical / renamed / changed）、测试对比（regressed 必须为 0）、无法修复的项和原因、提交 SHA。
