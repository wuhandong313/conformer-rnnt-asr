# 主干 432 维的 Stride 4 本机全量 Phone69 评测

该模型已完成 50 轮和 117,500 步训练，以原训练 dev-clean 验证损失最低检查点评测。

检查点 `val-none-epoch=14-step=34900.ckpt`，全局步数 34,900，dev-clean RNN-T 损失 10.572707。训练源码提交 `5ca625fb9038c08f80c12e2965c9716a1778d70b`。
检查点 SHA-256 `788ce66ad606aa40355e8f9efe4e88065c952c89b21024a1e8893bb8dd0698fc`。

测试集为已冻结的全部 2,620 条 test-clean、189,023 个参考音素，采用 CPU／FP32／batch 1／beam 20、4 线程、interop 1、worker 0。

| 指标 | 结果 |
| --- | ---: |
| Corpus PER | 5.425795% |
| 音素错误数 | 10,256 |
| 固定 30 秒编码器及压缩 GMAC | 21.537716832 |
| 实际编码器及压缩 GMAC／音频秒 | 0.616080849 |
| 实际 predictor GMAC／音频秒 | 0.111993491 |
| 实际 joiner GMAC／音频秒 | 0.015743953 |
| 实际总 GMAC／音频秒 | 0.743818293 |

独立核验完成的北京时间为 2026-11-22T21:54:57+08:00。逐句复算编辑距离和完整语料统计，核对各阶段真实帧长、MAC 分项、固定长度公式及输入文件 SHA-256。
计数包括线性、投影、卷积、注意力、检测点积、融合和实际 beam 路径的 predictor、joiner 乘加。
特征提取、归一化、激活、softmax、排序、比较及访存均不计入。

[逐句预测及算量](stride4_dim432/attempt_01/utterances.csv) · [完整结果](stride4_dim432/attempt_01/result.json) · [固定时长算量](stride4_dim432/fixed_duration_30s.json) · [冻结输入清单](stride4_dim432/manifest.json) · [独立核验](verification.json)
