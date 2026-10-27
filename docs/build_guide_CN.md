[English](build_guide_EN.md) | 简体中文

# HALOps 构建指南

直接使用 HALOps 时安装预编译包即可，见 [README 快速开始](../README.md#快速开始)。本文面向源码构建，说明环境准备、wheel/开发态构建与安装验证；项目通过 [`setup.py`](../setup.py) 和 PyTorch `CUDAExtension` 编译 CUDA/C++ 扩展。

[返回 README](../README.md) | [Benchmark 指南](../benchmark/README.md)

## 1. 环境要求

| 组件 | 要求 |
| --- | --- |
| Python | 3.10 或更高版本 |
| CUDA Toolkit | 13.0 或更高版本；`nvcc` 必须支持 `compute_100f/sm_100f` |
| NVIDIA GPU | 计算能力与 `sm_100f` 兼容的 Blackwell GPU |
| NVIDIA Driver | 与 CUDA 13.x 兼容的 R580 或更新版本 |
| PyTorch | 2.13.0 或更高版本 |
| C++ 编译器 | 支持 C++20 |

## 2. 准备构建环境

以下命令均从 HALOps 仓库根目录执行。

### 2.1 初始化 CUTLASS

```bash
git submodule update --init --recursive
```

如果 `3rdparty/cutlass/include` 不存在，`setup.py` 会自动对 `3rdparty/cutlass` 执行浅层初始化。

### 2.2 安装 Python 依赖

激活虚拟环境或 Conda 环境后安装依赖：

```bash
python -m pip install -r requirements.txt
```

其中 PyTorch 必须为与本机 Toolkit 兼容的 CUDA 版本。

### 2.3 配置 CUDA 路径

当 CUDA 未安装在默认位置，或机器上存在多套 CUDA 时，显式指定要使用的 Toolkit：

```bash
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

`CUDA_HOME` 应指向包含 `bin/nvcc`、`include/` 和 `lib64/` 的 CUDA Toolkit 根目录。

### 2.4 可选检查

```bash
nvcc --version
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

确认 `nvcc` 为 CUDA 13.0+，且 PyTorch 启用了兼容的 CUDA 版本。

## 3. 构建和安装

### 3.1 从源码安装

在当前环境中直接构建并安装：

```bash
python -m pip install --no-build-isolation --no-deps -v .
```

`--no-build-isolation` 复用当前 PyTorch/CUDA 工具链，`--no-deps` 避免改动已安装依赖，`-v` 输出详细编译日志。

### 3.2 构建并安装 wheel

推荐使用 wheel 验证最终制品：

```bash
rm -rf build dist
python -m pip wheel --no-build-isolation --no-deps -v -w dist .
python -m pip install dist/halops-*.whl --force-reinstall --no-deps
```

先清理 `dist/`，保证通配符只匹配本次构建的 wheel。

以上步骤等价于 `bash build.sh --release`。

### 3.3 开发态扩展编译

需要让当前环境以 editable 方式引用源码树时，执行：

```bash
python -m pip install --no-build-isolation --no-deps -e . -v
```

之后修改 CUDA/C++ 代码时，执行增量编译刷新源码树中的 `.so`：

```bash
bash build.sh
```

### 3.4 控制编译并发

PyTorch 扩展默认通过 Ninja 并行编译。内存不足时可降低并行任务数：

```bash
MAX_JOBS=4 python -m pip install --no-build-isolation --no-deps -v .
```

## 4. 验证构建结果

HALOps 可通过导入验证和算子测试检查安装结果。

### 4.1 导入验证

下面的验证只检查 Python 包和动态扩展能否加载，不执行 GPU kernel：

```bash
python - <<'PY'
import halops
import halops.halops_kernel as kernel

print("Package path:", halops.__file__)
print("Extension path:", kernel.__file__)
PY
```

### 4.2 算子测试

算子测试需要兼容 `sm_100f` 的 NVIDIA Blackwell GPU；其他架构无法运行当前构建产物。

最小算子测试命令为：

```bash
python tests/test_dense_gemm.py
```
