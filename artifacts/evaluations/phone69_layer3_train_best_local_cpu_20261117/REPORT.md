# L3训练损失best与验证损失best对照

同一次L3训练的两份checkpoint，均按完整test-clean 2620句、189023个参考音素、19452.480625秒音频，CPU／FP32／batch1／beam20、4线程、interop1、worker0、预热10句评测。

| 选择规则 | checkpoint | Corpus PER | 音素错误数 |
| --- | --- | ---: | ---: |
| 验证损失best | val-none-epoch=14-step=27324.ckpt | 6.393931% | 12086 |
| 已保存的训练回调best | train-none-epoch=30-step=57500.ckpt | 5.737397% | 10845 |

训练回调best的PER降低0.656534个百分点，少1241个错误；相对错误数降低10.268%。这是当前test-clean上的点估计，未新增显著性检验。

训练回调每500步检查一次，监控Losses/train_loss。所选step57500保存时的batch训练损失为9.764921188354492；训练日志的全局最小单步损失没有对应已保存权重。选择依据和最终完整边界中的回调状态见selection.json。

该结果是用户请求的选模规则诊断；两份checkpoint的test-clean PER排序与验证损失排序不一致。后续正式选模需在dev上确认与PER目标的一致性。

## 计算量

| 口径 | 编码器与压缩 | Predictor | Joiner | 合计 |
| --- | ---: | ---: | ---: | ---: |
| 实际完整语料GMAC／音频秒 | 0.343046 | 0.162770 | 0.024402 | 0.530217 |

固定30秒编码器＋压缩理论MAC：**16.347072 GMAC**，即0.544902 GMAC／音频秒。由本次完整逐层帧数校准，实际beam搜索端不加入理论编码器主指标。

逐句编辑距离和语料PER已独立复算，CSV、checkpoint、冻结源码、manifest及配置哈希通过核验。首次运行在解码前被运行库检查拦下，保留attempt_01日志；完整有效评测在attempt_02。

[完整结果](layer3_train_best/attempt_02/result.json) · [逐句记录](layer3_train_best/attempt_02/utterances.csv) · [独立核验](verification.json) · [选模依据](selection.json) · [30秒推导](layer3_train_best/fixed_duration_30s.json)
