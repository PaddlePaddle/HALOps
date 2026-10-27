## 变更说明 / Description

<!-- 做了什么、为什么；关联的 issue。 What and why; link the related issue. -->

## 精度影响 / Accuracy impact

<!-- 无 | 有（涉及的算子、精度类型，以及与改动前的误差对比）。 None | Changed (operators, dtypes and measured error vs. before). -->

## 性能影响 / Performance impact

<!-- 无 | 有（Benchmark 命令、配置、GPU 及改动前后数据）。 None | Changed (benchmark command, config, GPU, before/after numbers). -->

## 检查项 / Checklist

- [ ] 变更范围聚焦，已关联 issue 或任务 / Focused change, linked to an issue or task
- [ ] Python、PyBind 与 CUDA/C++ 接口保持一致 / Python, PyBind and CUDA/C++ interfaces are consistent
- [ ] 已运行相关单元测试，或说明未运行的原因 / Related unit tests pass (`pytest tests/...`), or skipped ones are listed
- [ ] 已填写上方精度与性能影响 / Accuracy and performance impact are filled in above
- [ ] 公开行为变更已更新 README 和文档 / README and docs updated for public behavior changes
