# 正确性校验机制

refcheck 在性能测试前比较各 backend 的输出，相关实现位于 `core/runner.py` 和 `core/refcheck.py`。

## 配置

- `refcheck`：是否执行校验。除 `prefill_chunked_gqa` 默认关闭外，其他现有场景默认启用；设为 `false` 可关闭。
- `refcheck_reference`：首选参考 backend。取值优先级为 `--refcheck-reference`（CLI）> YAML `refcheck_reference` > 默认 `golden`。参考 backend 即使不在 `backends` 列表里，也会自动补入参与比对。
- `refcheck_atol`、`refcheck_rtol`：绝对误差和相对误差容限，默认均为 `1e-2`。

至少要有两个可比对象才会校验，否则跳过（例如只测一个 backend 且它就是参考）。

## 校验流程

每个测试用例只执行一次 refcheck：选定一个参考 backend，其余 backend 的输出统一与它比较。流程如下：

```text
        组成候选列表
             │
      探测并剔除不可用候选
             │
      可用候选 ≥ 2 ? ──否──▶ 跳过校验
             │ 是
       选定参考 backend
             │
      其余候选与参考比较
             │
       在容限内 ? ──是──▶ 通过
             │ 否
            失败
```

## 比较规则

每个 backend 的输出展平后与参考输出逐元素比较，判定规则如下：

- 执行失败、输出包含 NaN/Inf 或元素数量与参考不一致：判为失败。
- 参考输出的所有元素绝对值都小于 `atol`：视为退化参考（近似全零，无比较意义），所有候选均判为失败。
- 其余情况按 `atol` 和 `rtol` 容限比较：全部元素在容限内则通过，否则失败。

每个 backend 的校验返回两项结果：**是否通过**，以及**超出容限的元素比例**（通过时为 0）。
