# 固定 30 秒输入的理论主要 MAC

采用实测逐层保留率校准固定长度公式；不是对整集历史 MAC 作线性缩放。

输入为 3001 个80维Mel帧，batch 1。主指标为编码器＋压缩层。
单次乘加计1 MAC，GMAC为十亿次乘加；每秒列为固定长度MAC除以该时长。

| 原归档ID | 论文ID | 实验 | 主干宽度 | 保留率（位置:输出/输入） | 编码器＋压缩 GMAC | GMAC/音频秒 | 原始全量 PER |
|---|---|---|---:|---|---:|---:|---:|
| E01 | — | stride4_none | 128 | 无自适应压缩 | 4.335942 | 0.144531 | 6.1289% |
| E02 | E01 | stride1_cosd_learned_five_frames | 128 | L2:0.407122 | 13.774741 | 0.459158 | 5.5141% |
| E03 | — | stride2_cosd_learned_five_frames | 128 | L2:0.401124 | 4.920928 | 0.164031 | 6.7135% |
| E04 | — | stride4_cosd_learned_five_frames | 128 | L2:0.401338 | 1.987294 | 0.066243 | 7.6081% |
| E05 | E02 | placement_mel_stride1 | 128 | L0:0.392838 | 8.114835 | 0.270494 | 5.5782% |
| E06 | E03 | placement_layer5_stride1 | 128 | L5:0.394356 | 21.380416 | 0.712681 | 4.7809% |
| E07 | E04 | placement_layer7_stride1 | 128 | L7:0.376053 | 26.499048 | 0.883302 | 4.8603% |
| E08 | — | placement_layers2_5_n3_stride1 | 128 | L2:0.860761；L5:0.418638 | 18.615317 | 0.620511 | 5.4210% |
| E09 | E05 | dim512_stride4_none | 512 | 无自适应压缩 | 27.679450 | 0.922648 | 5.7427% |
| E10 | E06 | placement_mel_layer5_stride1 | 128 | L0:0.392838；L5:0.401862 | 5.267984 | 0.175599 | 5.6882% |
| S1 | — | stride1_proj128 | 128 | 无自适应压缩 | 34.612238 | 1.153741 | 4.9856% |
| S2 | — | stride2_proj128 | 256 | 无自适应压缩 | 26.539015 | 0.884634 | 4.8989% |
| S8 | — | stride8_proj128 | 1024 | 无自适应压缩 | 38.627846 | 1.287595 | 6.6786% |
| E11 | — | placement_mel_layer5_layer8_stride1 | 128 | L0:0.392838；L5:0.384232；L8:0.449261 | 4.924789 | 0.164160 | 7.0616% |

PER来自原始test-clean全部2620句的CPU/FP32/batch1/beam20评估，未裁切成固定时长重算。
L0代表Mel处；每层保留率以该层输入为分母。理论长度使用逐层四舍五入（半数向上）及stride向上取整。

## 解码参考与范围

汇总CSV另外列出规定单路径的predictor、joiner和合计，使用所有模型一致的目标音素数。
它不是beam20执行成本；主要结果不把该参考路径与实际beam搜索混为同一口径。
计数包含线性、卷积、注意力矩阵乘法、输入输出投影、融合MLP、相邻点积和融合加权和；
不含特征提取、归一化、softmax、激活、峰谷比较、排序、访存及训练。

## 待校准项

- supp_layer1 / layer1：没有对应checkpoint的完整逐层帧数，不填写数值。
- supp_layer3 / layer3：没有对应checkpoint的完整逐层帧数，不填写数值。
- supp_layer9 / layer9：没有对应checkpoint的完整逐层帧数，不填写数值。
- supp_mel_layer3 / mel_layer3：没有对应checkpoint的完整逐层帧数，不填写数值。
- supp_mel_layer7 / mel_layer7：没有对应checkpoint的完整逐层帧数，不填写数值。
- supp_layers2_7_n3 / layers2_7_n3：没有对应checkpoint的完整逐层帧数，不填写数值。

## 复现与来源

- [算法、公式与论文时长依据](../../../docs/phone69_fixed_duration_macs.md)
- [完整结果与逐层长度](results.json)；[所有时长汇总CSV](summary.csv)
- [Fast Conformer §3.1.1 / 表4](https://arxiv.org/html/2305.05084v4#S3.SS1.SSS1)：30秒、编码器GMAC。
- [Zipformer 表2](https://arxiv.org/html/2310.11230v4)：30秒、编码器GFLOPs；单位与GMAC分开。

所有已报告保留率来自完整校准集；只由平均保留率代入的理论MAC不等于随机输入MAC的期望。
旧实验档案、历史实测总GMAC和原论文曲线保留，本表是单独的固定时长结构比较。
