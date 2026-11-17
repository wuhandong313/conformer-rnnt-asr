# Mel＋L3家中CPU全量评测

完整test-clean 2620句、189023个参考音素、19452.480625秒音频；CPU／FP32／batch1／beam20、4线程、interop1、worker0、预热10句。

| 选择规则 | checkpoint | Corpus PER | 音素错误数 | 编码器＋压缩GMAC／30秒 |
| --- | --- | ---: | ---: | ---: |
| dev-clean验证损失best | val-none-epoch=48-step=88800.ckpt | 6.020431% | 11380 | 4.174300 |

模型完成50轮、91250次更新，Mel与第3层后均采用learned五帧融合，stride1、主干128、共4249296个参数。checkpoint名称的epoch字段从0起算。

固定30秒输入为3001个Mel帧，各阶段条件保留率由本次完整逐层帧数校准，分别取该阶段输入作为分母，按统一固定长度公式推导；不是变长总MAC乘30。
主要MAC包括投影、线性、卷积、注意力矩阵乘积、检测点积、融合MLP与加权和；不含特征提取、归一化、softmax、激活、排序比较、访存和训练。

| 实际完整语料GMAC／音频秒 | 编码器与压缩 | Predictor | Joiner | 合计 |
| --- | ---: | ---: | ---: | ---: |
| beam20完整搜索 | 0.107929 | 0.089621 | 0.011174 | 0.208725 |

实际beam20解码端不并入30秒理论编码器主指标。2620句编辑距离和语料PER经独立复算，checkpoint、配置、冻结训练源码、输入清单、参考与CSV哈希通过核验。
首次bootstrap在scp下载认证阶段退出，未生成预测；原控制器日志保留在bootstrap_attempt_01。通过既有SSH通道传输并核验权重后，第二次bootstrap的attempt_01完成有效全量解码。

[完整结果](mel_layer3/attempt_01/result.json) · [逐句记录](mel_layer3/attempt_01/utterances.csv) · [30秒推导](mel_layer3/fixed_duration_30s.json) · [独立核验](verification.json) · [全部结果](../../research/phone69_all_results_20261117/REPORT.md)
