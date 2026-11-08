# phone69 自然峰谷批量实现（试验版）

此实现按 batch 计算 cosine_diff、与 SciPy 一致的峰谷和变长保留掩码，然后批量进行 five_frames 可学习融合。没有 keep_rate、配额或阈值；峰对应帧索引加一、谷不加一，平顶峰取中点，空候选保留第0帧。单帧绕过融合，与旧版一致。

## 启用方式

仅在独立试验配置中增加：

```yaml
model:
  compression:
    execution_backend: batched
```

缺省为 `reference`。单压缩层支持非流式 cosine_diff 的 five_frames learned 融合；多阶段支持现有 placement 配方中的未经平滑的 cosine_diff、five_frames learned 融合。不支持 hard 融合、neighborhood_radius 或其他探测器的 batched 执行；不支持的组合会明确报错。

多阶段配置示例（合并到独立实验配方的 `model` 节中）：

```yaml
model:
  sequence_mode: offline
  num_layers: 10
  compression:
    execution_backend: batched
    stages:
      - after_layer: 0
        detector:
          type: cosine_diff
        mixer:
          type: learned
        window_size: 5
        weight_predictor_mode: five_frames
      - after_layer: 5
        detector:
          type: cosine_diff
        mixer:
          type: learned
        window_size: 5
        weight_predictor_mode: five_frames
      - after_layer: 8
        detector:
          type: cosine_diff
        mixer:
          type: learned
        window_size: 5
        weight_predictor_mode: five_frames
```

`execution_backend` 只放在 `compression` 顶层，应用到所有压缩位置，并在模型构造时确定。切换时用新配置构建模型后载入兼容 checkpoint；调度器的 `execution_backend` 是只读属性。每个阶段独立检测当前序列，并把自己的输出长度传给后续编码层和压缩阶段；Mel 端使用输入特征维度，后续阶段使用编码器维度。两种后端共享融合参数、state_dict 键和 checkpoint 的计算定义，不改变峰谷索引规则、五帧融合、层位置或保留帧策略。执行后端仍记录在配置与 Lightning 超参数中；切换后端需要记录新的代码和配置指纹。

GPU 实现使用排序和一次观测 batch 最大保留帧数来构建变长输出，同时保留输入/输出长度记录。仍有按 batch 的 CPU/GPU 同步；不是零同步实现。批量窗口可能增加临时显存，正式使用前必须在最大压力批次测量。

## 2026-11-07 多阶段接入

模型工厂、多阶段编码器和调度器已传递同一个后端选项；省略选项保持原来的 reference 构建与计算路径。多阶段回归检查覆盖每阶段索引和长度、padding、输出与梯度、RNN-T loss，以及双向 checkpoint 兼容。测试位于 `tests/test_staged_batched_cosine.py`，有本地 CUDA 时同时检查 CUDA 路径。

本地 CPU 完整回归 97 项通过，其中新增 8 项多阶段检查。正式 80/128 维、10 层 `[0,5,8]` 配方的参数量仍为 4,269,973；测试输入下的阶段索引和长度一致，最终输出最大绝对差约 `1.91e-6`，参数梯度整体相对 L2 差约 `4.41e-7`。正式深层梯度测试沿用 `rtol=5e-4`，近零元素使用 `atol=2e-5`，并额外要求每个参数张量的相对 L2 差不超过 `1e-5`；小模型测试维持原有容差。

测试要求每阶段索引与长度精确一致、浮点输出和梯度在误差范围内一致，不承诺逐位相同或后续训练轨迹完全一致。全为空序列或单帧的阶段会跳过融合网络，保持参数梯度为 `None`，避免 Adam 因零梯度继续更新已有动量。本机没有 CUDA，多阶段 CUDA 数值、峰值显存和速度仍需未来在获授权的独立实验中测量；已有单层基准不能替代多阶段验证。软件回归测试在本地运行；当前训练不作为测试场地。

## 2026-10-29 验证状态

- 与 SciPy 对齐的随机序列、平顶、短序列、无候选、边界和 padding 测试通过。
- five_frames 的输出、输入梯度和融合 MLP 梯度在 CPU 上通过误差范围校验。
- 在服务器运行环境的临时隔离副本中，全部 74 项 CPU 回归测试通过，包括模型配置工厂、RNN-T 前后向和 checkpoint 定义兼容。


九帧实验、专用 GPU 等待器和基准入口已清理。当前五帧与多阶段的实现由本地回归测试验证；性能指标以权威评测归档为准。
