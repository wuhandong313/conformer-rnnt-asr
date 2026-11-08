# 第二层特征先滤波，再计算余弦变化与峰谷

2026-11-02。已实现代码与候选配置，未启动正式训练或完整数据评测。

**运算顺序**

```text
Mel → 输入投影/拼帧 → 两层Conformer → 原始特征 H
                                     ├→ 时间滤波 H̃
                                     │   → L2归一化
                                     │   → 相邻余弦距离
                                     │   → 峰谷中心 I
                                     └→ 原始H与中心I → 原有局部可学习融合
```

滤波对象是128维编码特征序列，沿时间轴进行，所有通道共享一个固定核。
它发生在归一化、余弦计算和峰谷检测之前；融合窗口及权重预测器仍读取原始H。
本次没有在余弦距离曲线上叠加滤波，也没有更改五帧融合定义。

**滤波器**

内部三点核为：

`[alpha / 4, 1 - alpha / 2, alpha / 4]`。

等价于 `(1-alpha) * H + alpha * binomial3(H)`，其中二项式核为
`[1, 2, 1] / 4`。候选强度alpha=0.25，因此内部核为
`[0.0625, 0.875, 0.0625]`。

句首、句尾只使用真实存在的邻帧并将剩余权重重新归一化；batch补零不参与滤波，
不同句子不互相混合。长度0、1的序列保持相应空/单帧语义。
滤波本身保持帧数与通道数，后续峰谷选择决定实际保留数，没有目标率、配额或阈值。
不能保证每句话的候选数量减少，也不能据此预先声称PER或速度改善。

**配置**

新候选为：

`configs/server_phone69_460h_offline_native_stride1_cosd_learned_feature_smooth_a025_s42_e50.yaml`

```yaml
model:
  num_layers_block1: 2
  num_layers_block2: 8
  compression:
    detector:
      type: cosine_diff
      params:
        feature_smoothing_alpha: 0.25
    mixer:
      type: learned
    weight_predictor_mode: five_frames
```

未设置强度时默认0，既有配置保持原有计算。强度必须是[0,1]内的有限数。
该选项目前通过两块式编码器的detector.params配置；既有staged位置实验的严格配置
规则仍保持原状。特征滤波与`preprocess: savgol`曲线滤波同时启用会明确报错。

候选保持原stride1五帧配置的其他训练设置，采用独立输出目录和
`from_checkpoint: null`，没有覆盖既有实验。reference和batched后端均支持该滤波。

**参数与checkpoint**

滤波没有可训练参数，不改变state_dict键或参数形状。
参数形状兼容不代表计算定义相同：新detector配置进入现有
`offline_native_definition`，Lightning的原有严格检查应拒绝不同定义的checkpoint。
正式实验应使用匹配配置，不能将旧checkpoint加载失败绕过后当作滤波重训结果。

**验证范围**

回归覆盖：

- 固定核数值、有效边界、通道共享与坐标旋转一致性；
- 空/单帧、混合长度、padding污染隔离和梯度；
- 实际向量滤波先于归一化和余弦计算；
- 选点后仍返回/融合原始特征，原版右上下文不被滤波或删除；
- reference/batched五帧的索引、长度、输出、输入和参数梯度一致性；
- 原无滤波选择行为、新配置构建、参数量和state_dict形状一致，以及编码器前后向。
- checkpoint保存滤波强度、不同强度/无滤波定义拒绝混用，相同强度跨后端兼容。

验证结果：新增10项CPU测试、29项已有压缩/位置/融合相关测试通过，Python语法
编译检查通过。新增测试可用`python -m unittest discover -s tests -p test_feature_smoothing.py`
运行。测试使用本地已有Torch/TorchAudio 2.8.0与SciPy 1.17.1环境。

CPU测试通过不等同于CUDA一致性、真实语料PER或速度验证；这些结果尚未生成。
