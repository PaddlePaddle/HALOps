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

## kernel 配置分层

model-constexpr、ns-constexpr、local-constexpr、format-tuning、shape-template、smem-cap 的修法都是把常量放进下面四层之一（对应 FlashMLA 的 `KVCacheFormat<ModelType>` / `KernelTemplate<Config>` / `*Params` / `get_meta()`，DeepGEMM 的模型参数模板 / `GemmConfig` / 运行时 shape / heuristics）。换模型时只加一个 Format 别名和 launcher 里的一个分支，kernel 代码不动。

| 层 | 放什么 | 写法 |
|---|---|---|
| Format | 模型和数据格式参数（head 数，head/latent/nope/rope 维度，hidden，expert 数，top-k，page size，量化 group，行字节数）和只由它们推导的量；当前实现支持哪些取值的 `static_assert` | `template <int kHeads_, int kDim_, ...> struct XxxFormat`。不放 tile、MMA N、stage、box、buffer 大小（format-tuning）。每个模型一个别名 `using V4HcaFormat = HcaFormat<64, 512, 448, 576, 8>;`，模型数字只写在这一处 |
| Config | 调优量：tile、MMA N、stage 数、buffer 深度、warp 角色、寄存器数、barrier id，可以由 Format 推导；smem 结构体和 `kSmemBytes` | `template <class Fmt, bool kFuseV = ...> struct XxxCfg : Fmt { static constexpr int kTile = 64; ... };`。变体（是否融合、量化方式）用 bool/enum 模板参数，不复制一份 Cfg 或 Smem。smem 结构体写成 `template <class C> struct XxxSmem`，在 Cfg 里 `static_assert(kSmemBytes <= kSmemCapacity, ...)` |
| Params | 运行时 shape：batch、序列长度、token 数、topk 条数、page 数、split 数、stride、指针 | 运行时结构体或 kernel 参数，不做模板参数（shape-template）。launcher 选中实例后检查它们和 Config 的约束，例如 `TORCH_CHECK(topk % C::kTile == 0, ...)` |
| Plan | host 端调度和代价模型的数字（split 上限、每 SM 的 CTA 数、合并的分块数） | launcher 用的具名结构体 `struct XxxPlan { static constexpr int kXxx = ...; };`，调度结果写进 Params |

架构和 ISA 常量（warp 大小、UMMA K、swizzle atom、TMEM 列数、smem 容量 `kSmemCapacity`）不属于这四层，放在 `csrc/common/sm100_arch.cuh`，先查已有定义。kernel 和 device 函数写成 `template <class C>`，读 `C::kX`。launcher 按张量 shape 选实例：`if (heads == V4HcaFormat::kHeads && dim == V4HcaFormat::kDim) return run<HcaCfg<V4HcaFormat>>(...);`，没有实例时 `TORCH_CHECK(false, ...)` 写出支持的取值。不要在 kernel 里直接读某个模型别名（`V4HcaFormat::kHeads`），只通过 `C` 读。

## 修法

| 规则 | 修法 |
|---|---|
| anon-namespace | 删掉 `namespace {` 和配对的 `}`，内容留在文件已有的具名 namespace 里。删除后 gate 的 tree 步骤会报出同名定义（ns-collision）。 |
| ns-static | `.cu`/`.cc` 里直接删 `static`。`.cuh`/`.h` 里普通函数改成 `inline`，`static constexpr` 变量改成 `inline constexpr`，constexpr 函数和 `CUTLASS_DEVICE`/`__forceinline__` 函数直接删 `static`。头文件里不能留下非 inline 的函数定义，否则多个 TU 包含时会重复定义。 |
| ns-collision | 两个 TU 定义了同名同签名的函数。实现完全相同时，移到 `csrc/common/` 里合适的头文件（优先用已有头文件），两边改为 include。实现不同时列进评论，不要靠改名了事。 |
| reuse-* | 换成 lint 消息里点名的 common 接口。如果接口的语义覆盖不了（例如需要 cluster launch 属性而 `LaunchKernelWithPdl` 不支持），不要硬改，记为"无法修复"并写明缺什么。 |
| getenv | 删掉环境变量开关和它控制的分支，保留环境变量未设置时走的路径。 |
| comment, debug-comment, comment-density | 只保留 docstring：紧贴在函数、类型（struct/class/union/enum/template）、`using`/`typedef` 声明上方、中间没有空行的整行注释块，以及 Python 的 docstring 字符串。其余注释全部删除，包括函数体内注释、行尾注释、常量和成员上的注释、分隔横幅。工具指令（`halops-lint:`、`clang-format`、`NOLINT`、`noqa`、`type: ignore` 等）、license 头、`}  // namespace x` 和 `#endif  // X` 保留。不要把被删的注释挪到 docstring 里凑数，docstring 只写这个函数或类型做什么、输入输出和前置条件。必须保留的约束改写成名字或 `static_assert` 的消息。 |
| model-constexpr | 按"kernel 配置分层"把模型值改成 Format 的模板参数，推导量在 Format 里推导，依赖它们的 smem 结构体、Config、函数和 kernel 改成模板，launcher 按 shape 分发。所有值保持原样，生成的 SASS 只允许 identical 或 renamed。 |
| ns-constexpr | 文件作用域的常量和带数值的 `#define` 按"kernel 配置分层"归层：调优量进 Config 的 `static constexpr`（仓库已有写法见 `csrc/kernel/gemm/utils.cuh` 的 `GemmKernelConfig`、`csrc/kernel/mhc/megallmhc.cuh` 的 `MhcKernelTraits`），架构和 ISA 常量进 `csrc/common/sm100_arch.cuh`，host 调度数字进 Plan。值不变。 |
| local-constexpr | 函数里字面量初始化的 `constexpr`/`static const`/常量名局部量，改成所属 kernel 的 Config 的 `static constexpr` 成员，函数里读 `C::kX`；是 ISA 常量的改用 `csrc/common/sm100_arch.cuh` 的定义。由模板参数或类型推导出来的局部量（`constexpr int n = C::kTile / 4`）不报，不用动。 |
| format-tuning | 把这个成员从 Format 移到以该 Format 为参数的 Config（例如从 `HeadLayout` 移到 `HcaTraits`），Format 里只依赖它的推导量和用到它的 `static_assert` 条件一起移过去。如果有 kernel 直接拿这个 Format 当模板实参（例如 `hca_combine_kernel<typename C::Layout>`），把移走的成员写成 `template <class L> struct XxxPart`，主 Config 继承它，再给那个 kernel 加一个 `template <class L> struct XxxCombineCfg : L, XxxPart<L> {};`，实参改成它。值不变，这类 kernel 的 SASS 应为 renamed。 |
| shape-template | 运行时 shape 做模板参数会让 kernel 只能跑一种 batch 或长度。参数实际是编译期容量上限时，改名为 `kMaxXxx` 并在 launcher 加 `TORCH_CHECK(n <= C::kMaxXxx, ...)`。参数确实是运行时 shape 时，改成运行时参数会改变生成代码，格式刷不改，记为"无法修复"并写明位置。 |
| smem-cap | 字面量 smem 上限（232448、`227 * 1024`、`228 * 1024`）换成 `kSmemCapacity`，include `common/sm100_arch.cuh`。原来是 `228 * 1024` 的断言会变严格（SM100 每个 block 的 opt-in 上限是 232448 B），编译通过即可，编译失败记为"无法修复"。 |
| unused-alias | 删掉这个 `using`。如果它是为将来的变体准备的，同样删掉，需要时再加。 |
| file-too-long | 按 FlashMLA 的拆法只搬代码不改代码：`<op>_config.cuh` 放 Format、Config、Smem、Plan，`<op>_kernel.cuh` 放 device 函数和 `template <class C>` 的 `__global__` kernel，`<op>.cu` 放 launcher、shape 分发和显式实例化。新 `.cuh` 里的非模板函数加 `inline`，`setup.py` 的 sources 不变（仍只编译 `<op>.cu`）。 |
| test-seed | 删掉全局 seed。需要固定随机数时改用局部 `torch.Generator(device=...).manual_seed(<原来的值>)`，并传给生成随机数的调用，让生成的数据和原来一致。 |
| new-namespace, binding-location, new-ext, new-extension | 涉及接口和目录结构，按 lint 消息改；改动会影响 Python 接口时先列进评论，不改。 |

## 不做的事

- 不改 `tools/lint`、`tools/agent`、`.claude`、`.github`、`.pre-commit-config.yaml`、`tests/conftest.py`、`pyproject.toml`、`setup.py`、`requirements.txt`、`3rdparty/`。PreToolUse hook 会拦截。
- 不加 `halops-lint: allow(...)` 豁免，不改 `baseline.json`。豁免由维护者决定。
- 不改测试的断言、容差、skip、parametrize。
- 不顺手重构，不改算法，不调性能参数。model-constexpr、ns-constexpr、local-constexpr、format-tuning 要求的模板化和常量搬迁，以及 file-too-long 要求的文件拆分，属于修法本身，不算重构，但常量的值和代码路径都不能变。gate 的 sass 步骤报出指令变化的 kernel，要在评论里逐个说明原因。
- 不 push，不用 `--no-verify`。

## 交付

评论写：修了哪些 finding（规则和数量）、gate 每一步的结果、SASS 对比（identical / renamed / changed）、测试对比（regressed 必须为 0）、无法修复的项和原因、提交 SHA。
