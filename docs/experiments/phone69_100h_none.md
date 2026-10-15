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
