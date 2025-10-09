# 测试目录结构

tests/ 目录下的每个文件夹对应 verl 中某个子命名空间的测试类别。例如：
- `tests/trainer` 用于测试 `verl/trainer` 相关功能
- `tests/models` 用于测试 `verl/models` 相关功能
- ...

有几个带有 `special_` 前缀的文件夹，用于特殊目的：
- `special_distributed`: 必须使用多 GPU 运行的单元测试
- `special_e2e`: 包含训练/生成脚本的端到端测试
- `special_npu`: NPU 测试
- `special_sanity`: 一套快速健全性测试
- `special_standalone`: 一组设计在专用环境中运行的测试

测试加速器
- 默认情况下，测试在 GPU 可用的情况下运行，除了 `special_npu` 下的测试，以及任何名称以 `on_cpu.py` 结尾的测试脚本。
- 名称后缀为 `on_cpu.py` 的测试脚本将在 Linux 环境的 CPU 资源上进行测试。

# Workflow 目录结构

所有 CI 测试都通过 `.github/workflows/` 中的 yaml 文件进行配置。以下是所有测试配置的概述：
1. 始终触发的 CPU 健全性测试列表：`check-pr-title.yml`、`secrets_scan.yml`、`check-pr-title,yml`、`pre-commit.yml`、`doc.yml`
2. 一些重量级多 GPU 单元测试，例如 `model.yml`、`vllm.yml`、`sgl.yml`
3. 端到端测试：`e2e_*.yml`
4. 单元测试
  - `cpu_unit_tests.yml`，对所有文件名模式为 `tests/**/test_*_on_cpu.py` 的脚本运行 pytest
  - `gpu_unit_tests.yml`，对所有不带 `on_cpu.py` 后缀的文件运行 pytest
  - 由于 cpu/gpu 单元测试默认运行 `tests` 下的所有测试，请确保在以下情况下手动排除测试：
    - 向 `.github/workflows` 添加新的 workflow yaml 文件时
    - 向第 2 点提到的 workflow 添加新测试时