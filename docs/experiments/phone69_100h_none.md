# Phone69 clean-100，无压缩基线

配置：`configs/server_phone69_100h_none.yaml`。运行于服务器原仓库的 `main` 分支。

- 固定 69 个带重音 ARPABET 音素，编码时过滤 `spn`；blank 为 69，共 70 类输出。
- train-clean-100 / dev-clean / test-clean，源标签仍在 `/test21/librispeech`。
- Conformer 2+8 层、512 维、8 头、FFN 1024；LSTM predictor 2 层、128 维。
- 压缩 `none`；10 ms 帧移，时间降采样 stride 4，右上下文 4 帧。
- seed 42，固定 40 轮，不使用早停，从头训练。
- Adam，峰值学习率 5e-4，预热 10000 steps，RNN-T sum loss，梯度裁剪 10。
- 每批最多 4000 个真实音素；20 个读取进程，prefetch 2；单 GPU float32/TF32。
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

启动器拒绝覆盖已有的 `run.json`。训练中的日志和大 checkpoint 留在实验目录；小型运行记录和日志快照可保存到 `artifacts/` 后同步。训练期间保持 main 的训练源码及本配置不变；运行记录保存启动时的提交和配置哈希。

## 本次启动记录

已于 2026-10-15 16:27:29（Asia/Shanghai）在 tmux `phone69_100h_none_s42_e40` 中启动，训练代码提交为 `54008b1`。实际模型参数为 40,264,262，每轮 894 steps，总计 35,760 steps。真实音频训练/验证 smoke 和 32 项回归测试通过；正式训练已完成验证并保存可加载 checkpoint。训练和最终 PER 评估尚未完成。

启动时的小型日志快照位于 `artifacts/logs/phone69_100h_none_s42_e40.*.log`，状态和资源采样位于 `artifacts/runs/phone69_100h_none_s42_e40/`。这些是固定快照，实时状态以服务器实验目录的 `run.json`、`logs/train.log` 和 `lightning_logs/version_0/metrics.csv` 为准。

tmux 的 `monitor` 窗口上方显示训练日志，下方每 2 秒刷新 GPU 使用情况：

```bash
tmux attach -t phone69_100h_none_s42_e40
```
