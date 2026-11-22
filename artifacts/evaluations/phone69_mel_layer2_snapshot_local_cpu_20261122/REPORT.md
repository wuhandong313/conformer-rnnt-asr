# Mel＋L2 训练中快照 本机全量 Phone69 评测

服务器在训练第 36 轮、第 64,940 步时仍在运行。本地取得当时已保存的 dev-clean 验证损失最低检查点，完成接收与 SHA-256 核验的北京时间为 2026-11-22T21:31:55+08:00。训练快照和服务器原始时戳见 server_metadata.json。

检查点 `val-none-epoch=30-step=55150.ckpt`，全局步数 55,150，dev-clean RNN-T 损失 12.619652。训练源码提交 `5063a4e6376d49d932ebd07c567750bc49f0de3b`。
检查点 SHA-256 `aa01cd6f08d67a91f31116105abeebaeeb0b90fa8be8d1bc9ea64866547d6203`。

测试集为已冻结的全部 2,620 条 test-clean、189,023 个参考音素，采用 CPU／FP32／batch 1／beam 20、4 线程、interop 1、worker 0。

| 指标 | 结果 |
| --- | ---: |
| Corpus PER | 7.199124% |
| 音素错误数 | 13,608 |
| 固定 30 秒编码器及压缩 GMAC | 3.628389712 |
| 实际编码器及压缩 GMAC／音频秒 | 0.095849974 |
| 实际 predictor GMAC／音频秒 | 0.087315855 |
| 实际 joiner GMAC／音频秒 | 0.011068209 |
| 实际总 GMAC／音频秒 | 0.194234038 |

独立核验完成的北京时间为 2026-11-22T21:49:18+08:00。逐句复算编辑距离和完整语料统计，核对各阶段真实帧长、MAC 分项、固定长度公式及输入文件 SHA-256。
计数包括线性、投影、卷积、注意力、检测点积、融合和实际 beam 路径的 predictor、joiner 乘加。
特征提取、归一化、激活、softmax、排序、比较及访存均不计入。

[逐句预测及算量](mel_layer2/attempt_01/utterances.csv) · [完整结果](mel_layer2/attempt_01/result.json) · [固定时长算量](mel_layer2/fixed_duration_30s.json) · [冻结输入清单](mel_layer2/manifest.json) · [独立核验](verification.json)
