# 固定 stride1／2 的全量本地评测

两组均完成 test-clean 全部2,620句、189,023个参考音素、19,452.480625秒音频。逐句CSV重新汇总的PER和主要MAC与结果一致，checkpoint、配置、冻结源码及参考清单SHA均核对通过。[独立核验记录](verification_stride1_stride2.json)

| 运行 | Corpus PER | 总主要 GMAC／音频秒 | 编码器／压缩 | Predictor | Joiner | 音素错误数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| stride1_proj128 | 4.985637% | 1.037971 | 0.667522 | 0.315452 | 0.054997 | 9,424 |
| stride2_proj128 | 4.898875% | 0.861297 | 0.641774 | 0.190072 | 0.029451 | 9,260 |

MAC各组成列的单位均为GMAC／输入音频秒；先汇总全语料乘加量，再除以真实总音频时长。编码器计数包含投影、10层Conformer及注意力矩阵乘积，解码端沿实际beam20搜索路径统计predictor和joiner。特征提取、归一化、softmax、激活、比较排序及访存不计入主要MAC。逐句编码器执行计数与独立固定stride公式一致。计数hook不改变张量；PER和MAC来自同一遍完整解码。

使用既有本地CPU协议：Python3.12.9、torch/torchaudio2.8.0、FP32、batch1、beam20、4线程／interop1、worker0、预热10句。全部音频、标签、词表与归一化统计匹配此前冻结清单；未生成RTF或重复计时指标。

| 运行 | 最佳 dev-clean checkpoint | 主干维度 | 参数量 | 训练音素预算 | 最终更新数 |
| --- | --- | ---: | ---: | ---: | ---: |
| stride1_proj128 | `val-none-epoch=43-step=136173.ckpt` | 128 | 4,215,622 | 5,300 | 155,550 |
| stride2_proj128 | `val-none-epoch=24-step=46360.ckpt` | 256 | 10,378,310 | 8,800 | 93,250 |

两组均为固定投影128、10层、seed42、从头训练50轮，但主干宽度、容量、批预算及实际更新数不同，不作为等训练暴露或等容量对照。单seed结果不用于推断稳定优劣。stride8仍在续训，其全量PER和解码端MAC尚未完成，暂不填值。

- stride1：[完整结果](stride1_proj128/attempt_01/result.json)、[逐句记录](stride1_proj128/attempt_01/utterances.csv)、[冻结输入](stride1_proj128/manifest.json)。北京时间2026-11-11 23:44完成。
- stride2：[完整结果](stride2_proj128/attempt_01/result.json)、[逐句记录](stride2_proj128/attempt_01/utterances.csv)、[冻结输入](stride2_proj128/manifest.json)。北京时间2026-11-12 00:05完成。

检查点在本地 `output/local_cpu_20261111/models/`，大文件不入Git。此目录独立于原E01–E10冻结归档。服务器恢复背景和已披露的滚动last归档例外见[恢复记录](../../../docs/phone69_stride_recovery_20261111.md)。
