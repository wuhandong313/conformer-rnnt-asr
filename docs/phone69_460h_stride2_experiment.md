# Phone69 460h：stride2 + 动态压缩对照

新增实验 `exp_phone69_460h_stride2_cosd_learned_s42_e50`，在现有 stride1/stride4 对照之间加入 2:1 帧拼接，再进行 CosDiff learned 动态压缩。

## 维度与参数审计

| 方案 | 输入投影 | stride | Conformer 内部宽度 | 输出维度 | 注意力头 | FFN | 模型总参数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stride4_none | 32 | 4 | 128 | 256 | 8 | 512 | 4,207,942 |
| stride1_cosd_learned | 128 | 1 | 128 | 256 | 8 | 512 | 4,215,947 |
| **stride2_cosd_learned** | **64** | **2** | **128** | **256** | **8** | **512** | **4,210,827** |
| stride4_cosd_learned | 32 | 4 | 128 | 256 | 8 | 512 | 4,208,267 |

所有方案的两个 Conformer block 合计参数均为 **3,846,400**，层数均为 2+8。三个动态压缩方案的 learned mixer 参数均为 **325**，predictor、joiner 和输出投影的结构相同。总参数量的少量差异来自输入投影宽度；维度公平保持了主干和输出的等宽。

新实验输入投影为 `80 → 64`，两帧拼接后为 `64 × 2 = 128`，因此不会因 stride2 将内部模型扩大到 256 维。输入右上下文为 4 帧，下采样后保留 2 帧，故 `model.compression.right_context_length=2`。

压缩设置沿用 `cosine_diff` detector、`learned` mixer、`keep_rate=0.27`、`weight_predictor_mode=diff5`。主体帧近似保留率为 `0.27 / 2 = 13.5%`；实际帧数还包含舍入和额外右上下文。

实际构建模型的审计及奇数/偶数输入长度的前后向、输出长度检查见 [dimension_audit.json](../artifacts/runs/phone69_460h_stride2/dimension_audit.json)。

## 训练设置

- 配置：[server_phone69_460h_stride2_cosd_learned_s42_e50.yaml](../configs/server_phone69_460h_stride2_cosd_learned_s42_e50.yaml)。
- 数据：固定 phone69，`train-clean-100 + train-clean-360` 共 460h；训练中验证集为 `dev-clean`。
- 从头训练，seed=42，总计 50 epoch，无早停。学习率 0.0005、warmup 10000，优化器、SpecAugment、dropout 和 checkpoint 策略沿用原配置。
- 单卡 A100 80 GiB。phone-token 预算 34,000，真实批次探针覆盖 train-clean-100、train-clean-360、dev-clean 的注意力、joint、句数和最长音频边界，并保留 allocator cache。完整探针结果位于 `artifacts/runs/phone69_460h_stride2/probe_b*.json`。
- checkpoint 使用现有训练器和安全选择逻辑；`last.ckpt` 包含模型、优化器和调度器状态。发生预算变更时，恢复必须满足已完成 epoch 的完整边界检查。

本轮执行训练及训练中的 dev-clean 验证。后续 test-clean 评测沿用固定 phone69、最佳 checkpoint、beam=20 和单调时钟 RTF 口径。

## 运行与检查

服务器原仓库 `/test21/conformer-rnnt-asr`：

```sh
tmux new-session -d -s phone69_460h_stride2 -c /test21/conformer-rnnt-asr \
  'env CUDA_VISIBLE_DEVICES=0 bash scripts/server/phone69_460h_stride2_pipeline.sh'
```

该独立入口共享原队列的训练锁。实验目录与日志使用新路径；原三组队列保持完成状态。

- 状态：`exp_phone69_460h_stride2_cosd_learned_s42_e50/run.json`
- 训练日志：`exp_phone69_460h_stride2_cosd_learned_s42_e50/logs/train.log`
- 显存记录：`exp_phone69_460h_stride2_cosd_learned_s42_e50/logs/gpu_usage.csv`
- 总日志：`artifacts/logs/phone69_460h_stride2_pipeline.log`
- 完成标记：`exp_phone69_460h_stride2_cosd_learned_s42_e50/TRAINING_COMPLETE`
- 启动记录：`artifacts/runs/phone69_460h_stride2/launch.json`

本地读取服务器状态：

```sh
python output/phone69_460h/remote.py scripts/server/status_phone69_460h_stride2.py
```

显存目标约为设备的 90%。reserved 峰值与 `nvidia-smi` 设备使用不同；不得把 GPU 计算利用率当作显存占用率。监控以服务器当前 UTC 和服务器文件时间比较，避免两端时钟差误报。
