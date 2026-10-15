# Phone69 clean-100，无压缩基线

配置：`configs/server_phone69_100h_none.yaml`。运行于服务器原仓库的 `main` 分支。

- 固定 69 个带重音 ARPABET 音素，编码时过滤 `spn`；blank 为 69，共 70 类输出。
- train-clean-100 / dev-clean / test-clean，源标签仍在 `/test21/librispeech`。
- Conformer 2+8 层、512 维、8 头、FFN 1024；LSTM predictor 2 层、128 维。
- 压缩 `none`；10 ms 帧移，时间降采样 stride 4，右上下文 4 帧。
- seed 42，固定 40 轮，不使用早停；前 3 轮之后加大 batch 并恢复训练。
- Adam，峰值学习率 5e-4，预热 10000 steps，RNN-T sum loss，梯度裁剪 10。
- 每批音素预算从 4000 提高到 10000；20 个读取进程，prefetch 2；单 GPU float32/TF32。
- 每 200 steps 验证，保留两个最优验证 checkpoint 和可恢复的 last checkpoint。
- 验证结束立即保存对应 checkpoint，以本次验证分数选优。
- 训练结束后选择最低验证损失的 checkpoint，自动进行 test-clean beam20 corpus PER 评估。

服务器启动入口：

```bash
CUDA_VISIBLE_DEVICES=0 /root/anaconda3/envs/whd/bin/python -u scripts/server/run_phone69_100h_none.py
```

使用独立目录 `/test21/conformer-rnnt-asr/exp_phone69_100h_none_s42_e40/`：

```text
run.json                  # 运行状态、启动提交、参数、进程及最终结果
logs/train.log
logs/test_clean_beam20.log
checkpoints/
lightning_logs/
evaluations/test_clean_beam20.csv
evaluations/metrics.json
```

已有 `run.json` 时，启动器要求显式指定完整的恢复 checkpoint，并检查旧进程已经退出。更改 batch 预算要求 checkpoint 处于轮次结束边界。每次启动记录保存到 `run_history/`，日志追加，保留旧记录。训练中的日志和大 checkpoint 留在实验目录；小型运行记录和日志快照保存到 `artifacts/` 后同步。运行记录保存每次启动时的提交和配置哈希。

当前配置的 `paths.from_checkpoint` 指向服务器上的轮次结束 checkpoint。若要重新从头训练，应将其置为 `null`，并使用全新的 `exp_dir`。

## 本次启动记录

已于 2026-10-15 16:27:29（Asia/Shanghai）在 tmux `phone69_100h_none_s42_e40` 中启动，初始训练代码提交为 `54008b1`。实际模型参数为 40,264,262，原预算下每轮 894 steps。真实音频训练/验证 smoke 和回归测试通过；正式训练已完成验证并保存可加载 checkpoint。训练和最终 PER 评估尚未完成。

## 加大 batch 后恢复

原进程收到 SIGINT 后正常退出，保留最新完整恢复点（epoch index 2，global step 2388）。先用原 4000 预算完成该轮，随后在 `Trainer.fit` 返回后保存 `epoch_boundary_before_batch_resize.ckpt`。从这个边界恢复时不手工改写 Lightning 的循环计数。

恢复时已经完成 3 轮、2682 steps。模型权重、Adam 动量、预热学习率与 global step 均恢复；总轮数仍为 40，剩余 37 轮，不启用早停。学习率和梯度累积设置保持原值。中途换 batch 会改变每轮更新次数及预热覆盖的样本量，因此这次运行应标记为混合 batch 预算实验。

`scripts/server/finish_phone69_epoch.py` 记录本次轮次收尾操作，`scripts/server/probe_phone69_batch.py` 从真实 FLAC 时长及标签长度选取注意力计算量、joiner 张量和句数最大的批次，测试前向、反向、梯度裁剪和 Adam 更新。探测不覆盖原 checkpoint。

| 每批音素预算 | 每轮 batch | 探测峰值已分配显存 | 结果 |
| --- | --- | --- | --- |
| 6000 | 593 | 36.01 GiB | 通过 |
| 8000 | 444 | 47.91 GiB | 通过 |
| 10000 | 355 | 57.16 GiB | 通过，正式采用 |

10000 预算的探测峰值保留显存为 59.18 GiB，所有 28,539 条训练语音保留。RNNT guard 将 354 个候选 batch 拆成 355 个，未丢弃样本。后续计划更新数为 `2682 + 37 × 355 = 15817`，不再使用最初的 35,760 steps 估算。以上是三个代表性极端 batch 的测试，不等同于遍历所有 batch；正式训练还需留意显存波动。探测 JSON、小日志及恢复后的资源采样见 `artifacts/runs/phone69_100h_none_s42_e40/` 和 `artifacts/logs/`。

2026-10-15 17:03:43（Asia/Shanghai）在同一 tmux 会话恢复，训练提交 `281e5e4`。本地 37 项回归测试通过，服务器上的 6 项恢复/保存检查通过。恢复后 41.64 秒窗口内完成 63 个 batch，约 1.51 batch/s（0.66 s/batch）；GPU 平均利用率 89.1%，采样显存为 62.8–77.9 GiB / 80 GiB。这里是 `nvidia-smi` 的进程占用，包含 PyTorch 缓存，与探测时的峰值已分配显存含义不同。该窗口不包含验证和保存开销，不能直接当作整轮平均速度。

恢复后已完成首次验证，并在 step 2882 保存了新的完整验证 checkpoint；后续训练正常继续。

启动时的小型日志快照位于 `artifacts/logs/phone69_100h_none_s42_e40.*.log`，状态和资源采样位于 `artifacts/runs/phone69_100h_none_s42_e40/`。这些是固定快照，实时状态以服务器实验目录的 `run.json`、`logs/train.log` 和 `lightning_logs/version_0/metrics.csv` 为准。

tmux 的 `monitor` 窗口上方显示训练日志，下方每 2 秒刷新 GPU 使用情况：

```bash
tmux attach -t phone69_100h_none_s42_e40
```
