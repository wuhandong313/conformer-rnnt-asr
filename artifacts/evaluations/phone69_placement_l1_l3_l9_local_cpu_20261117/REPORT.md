# L1／L3／L9家中CPU全量评测

三组各覆盖test-clean全部2620句、189023个参考音素、19452.480625秒音频。
协议：CPU／FP32／batch1／beam20，4线程、interop1、worker0、预热10句。

| 插入位置 | Corpus PER | 编码器＋压缩GMAC／30秒 | GMAC／音频秒（30秒口径） | 音素错误数 |
| --- | ---: | ---: | ---: | ---: |
| layer1 | 5.452247% | 10.956676 | 0.365223 | 10306 |
| layer3 | 6.393931% | 16.311692 | 0.543723 | 12086 |
| layer9 | 4.921094% | 31.840138 | 1.061338 | 9302 |

PER来自完整原始音频，固定30秒理论MAC由本次完整逐层帧数校准；未将音频裁切到30秒，也未将变长全量MAC线性缩放。
独立重算每句编辑距离、语料PER、参考覆盖、MAC分项之和，并核验checkpoint、配置、冻结源码、manifest和CSV哈希。
实际beam20的predictor／joiner计数保留在各组result和verification中；不加入固定长度理论编码器主指标。
主要MAC包括投影、线性、卷积、注意力矩阵乘积、检测点积和融合；不包括特征提取、归一化、softmax、激活、比较排序和访存。
不同训练的单seed点估计，未新增配对区间，不据此声称统计显著。checkpoint文件名的epoch字段采用0起算。

[独立核验](verification.json) · [L1记录](layer1/attempt_01/result.json) · [L3记录](layer3/attempt_01/result.json) · [L9记录](layer9/attempt_01/result.json)
