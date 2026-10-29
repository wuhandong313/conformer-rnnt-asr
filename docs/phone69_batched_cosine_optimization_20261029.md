# phone69 自然峰谷批量实现（试验版）

此实现按 batch 计算 cosine_diff、与 SciPy 一致的峰谷和变长保留掩码，然后批量进行 five_frames / nine_frames 可学习融合。没有 keep_rate、配额或阈值；峰对应帧索引加一、谷不加一，平顶峰取中点，空候选保留第0帧。单帧绕过融合，与旧版一致。

## 启用方式

仅在独立试验配置中增加：

```yaml
model:
  compression:
    execution_backend: batched
```

缺省为 `reference`，现有配置及运行中的实验不变。当前只支持非流式、未经平滑的 cosine_diff、单个压缩层、five_frames 或 nine_frames 的 learned 融合，且不支持 neighborhood_radius。暂不支持 staged、其他探测器、其他融合方法；会明确报错，不静默更换算法。两种后端共享模型参数及 checkpoint 的计算定义；配置、代码和实际后端仍应通过实验配置与代码指纹记录。

GPU 实现使用排序和一次观测 batch 最大保留帧数来构建变长输出，同时保留输入/输出长度记录。仍有按 batch 的 CPU/GPU 同步；不是零同步实现。批量窗口可能增加临时显存，正式使用前必须在最大压力批次测量。

## 2026-10-29 验证状态

- 与 SciPy 对齐的随机序列、平顶、短序列、无候选、边界和 padding 测试通过。
- five_frames 与 nine_frames 的输出、输入梯度和融合 MLP 梯度在 CPU 上通过误差范围校验。
- 在服务器运行环境的临时隔离副本中，全部 74 项 CPU 回归测试通过，包括模型配置工厂、RNN-T 前后向和 checkpoint 定义兼容。
- 唯一 A100 正在训练九帧实验；GPU 端等价性、真实批次显存和速度尚未测量，不宣称已经提速或 GPU 完全等价。不能据此切换当前训练。

`scripts/server/benchmark_phone69_batched_cosine.py` 用完成后的九帧 checkpoint 和真实训练批次，在独占空闲 GPU 时比较原实现与批量实现的索引、输出、梯度、显存、压缩前后向耗时，并用少量真实样本比较模型前后向。时间使用同步后的 `perf_counter`，各模式轮流执行，报告中位数；模型比较不执行优化器更新，也不包括数据加载及验证。测试结果应保留原始数据，不能只报告最优 case。

已启动独立等待器 `scripts/server/wait_phone69_batched_gpu_check.py`：仅在原九帧训练完成、取得互斥锁并确认 GPU 空闲时，才运行隔离副本中的 GPU 单测与基准；等待期不初始化 CUDA。状态和结果记录在服务器 `artifacts/runs/phone69_batched_gpu_check_20261029/`。测试失败或没有加速时保留结果并报告，不自动启用新后端。
