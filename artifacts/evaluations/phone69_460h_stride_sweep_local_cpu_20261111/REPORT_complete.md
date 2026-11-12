# 固定 stride1／2／8 全量训练与本地评测完成

三组均完成50轮，服务器`completion.json`与实际run、最佳checkpoint及最终完整边界一致。各最终边界均已实际读取，确认完成50轮、优化器和scheduler状态齐全；29个原训练源码／配置SHA及续训配置核验通过。[服务器完成核验](server_completion_verification.json)、[本地独立核验](verification_complete.json)、[最终汇总](summary.json)、[文件校验清单](SHA256SUMS_complete)。

每组均覆盖冻结test-clean全部2,620句、189,023个参考音素、19,452.480625秒音频。逐句重新计算编辑距离、语料PER及主要MAC，结果与三份`result.json`一致；参考顺序、音频与标签、源码、配置、归一化统计、词表和本地checkpoint SHA均核对通过。最终本地完成标记为[EVALUATION_COMPLETE](EVALUATION_COMPLETE)。

| 运行 | Corpus PER | 总主要GMAC／音频秒 | 编码器／压缩 | Predictor | Joiner | 音素错误数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| stride1_proj128 | 4.985637% | 1.037971 | 0.667522 | 0.315452 | 0.054997 | 9,424 |
| stride2_proj128 | 4.898875% | 0.861297 | 0.641774 | 0.190072 | 0.029451 | 9,260 |
| stride8_proj128 | 6.678552% | 1.312000 | 1.230095 | 0.073011 | 0.008893 | 12,624 |

MAC各组成列单位均为GMAC／输入音频秒。先汇总全语料乘加量，再除以真实总音频时长。计数包含线性层、卷积、注意力矩阵乘积、压缩以及实际beam搜索路径的predictor／joiner矩阵；不含特征提取、归一化、softmax、激活、比较排序和访存。逐句编码器执行计数与独立固定stride公式一致，解码端计数与实际扩展token数及joiner行数一致。计数hook不改变张量，PER和MAC来自同一遍完整解码。

协议为既有本地CPU、FP32、batch1、beam20、4线程／interop1、worker0、预热10句；Python3.12.9、torch／torchaudio2.8.0。未生成RTF或重复计时指标，原E01–E10冻结档案保持原样。

| 运行 | 最佳dev-clean checkpoint | 主干维度 | 参数量 | 训练音素预算 | 最终更新数 |
| --- | --- | ---: | ---: | ---: | ---: |
| stride1_proj128 | `val-none-epoch=43-step=136173.ckpt` | 128 | 4,215,622 | 5,300 | 155,550 |
| stride2_proj128 | `val-none-epoch=24-step=46360.ckpt` | 256 | 10,378,310 | 8,800 | 93,250 |
| stride8_proj128 | `val-none-epoch=21-step=36553.ckpt` | 1024 | 95,523,398 | 9,700 | 84,650 |

三组固定投影128、10层、seed42。stride8在环境重启后从完整第8轮边界恢复优化器、scheduler和loop状态，未保存的第9轮部分重做；原预算和模型保持一致。实际恢复启动提交为`191e387c7b82aaa8eb51b02396b231478418ffc3`，后续归档提交不替代启动记录。

在这三次单seed运行中，stride2同时取得最低PER与最低主要MAC；各组主干宽度、容量、批预算及实际更新数不同，因此不能将差异归因于stride本身，也不构成等容量、等更新数或预算匹配的对照。

- stride1：[完整结果](stride1_proj128/attempt_01/result.json)、[逐句记录](stride1_proj128/attempt_01/utterances.csv)、[冻结输入](stride1_proj128/manifest.json)。最佳checkpoint SHA：`01409ecf07a8eb3d5536c56a552554bdd860801dc2ce790e011a580b46033c77`。
- stride2：[完整结果](stride2_proj128/attempt_01/result.json)、[逐句记录](stride2_proj128/attempt_01/utterances.csv)、[冻结输入](stride2_proj128/manifest.json)。最佳checkpoint SHA：`362f3aae0bf25a87e15d10e18fe57edacd735398d0f8efeda7f5d67bfe0d055d`。
- stride8：[完整结果](stride8_proj128/attempt_01/result.json)、[逐句记录](stride8_proj128/attempt_01/utterances.csv)、[冻结输入](stride8_proj128/manifest.json)。最佳checkpoint SHA：`b0d98ff359afee613f574b134f9111d39d737e0ec87a392b2d35e64afcb8ceef`。

checkpoint已保存在本地`output/local_cpu_20261111/models/stride{1,2,8}_proj128/`，大文件不入Git。服务器对应实验目录为`/test21/conformer-rnnt-asr/exp_phone69_460h_offline_native_stride{1,2,8}_none_proj128_s42_e50/`。本地stride8于北京时间2026-11-12 09:17完成，三组最终完整边界分别为step155,550／93,250／84,650。

最终控制器原始状态保留了缓存的`pending_strides=[8]`，但`status=complete`、三组`completed_models`、三份全量结果和完成标记一致。完训后只读状态接口因`active_stride=null`产生的配置误报已修正并重新核验。恢复过程及已披露的滚动last备份例外见[恢复记录](../../../docs/phone69_stride_recovery_20261111.md)；[早期两组报告](REPORT_stride1_stride2.md)保留为历史阶段记录。
