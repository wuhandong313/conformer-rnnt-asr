# ExtremaFuse · ACL 2027 长文初稿

[阅读英文初稿 PDF](main.pdf) · [LaTeX 源文件](main.tex) · [8 组结果主图](figures/quality_budget_30s.pdf)

本文研究离线语音识别中应按哪些声学变化合并帧，以及在什么深度合并可以兼顾识别误差与计算预算。用区分元音重音的语料级 PER 衡量识别表现。实验覆盖 460 小时 LibriSpeech 训练数据和 2,620 条完整 test-clean 录音。

## 近期 ACL 论文如何影响本稿

- Xiang 等发现，在大型语音语言模型深层进行池化后，最终只保留 14.3% 的音频 token，预填充计算量仍有原来的 92.0%。引言由此提出声学编码器中的位置问题：把压缩前移能节省多少计算，识别误差又会怎样变化？[Xiang 等，Findings of ACL 2026，表 1](https://aclanthology.org/2026.findings-acl.742.pdf)
- Ji 等研究视觉 token 在不同层的裁剪配置，按预算与实测前沿核对结果。本文对每个已训练的放置方案逐层计数。主图按算量直连融合组的五个非支配工作点，用独立方点标出三组 stride 4 参照。[Ji 等，ACL 2026](https://aclanthology.org/2026.acl-long.420/)
- Liao 等发现简单图像降采样在多项基准上能超过复杂压缩。对应到本文，128、256 和 512 维 stride 4 一起进入主图，并保留逐句输出长度完全一致的均匀池化对照。前者检验完整识别器，后者检验训练好的模型对汇聚规则有多敏感。[Liao 等，ACL 2026](https://aclanthology.org/2026.acl-long.195/)

## 摘要和引言之后的叙述方向

1. **方法先讲两个决定。** 相邻帧余弦距离的峰谷决定在哪里输出，五帧窗口的学习权重决定每个输出保留哪些邻帧的信息。用一张图沿 Mel 输入和编码器隐层走一遍，让读者看到重叠窗口以及各层实际接收的序列。[方法图](figures/extremafuse_network.pdf)
2. **实验先约定怎样比较。** 对同一批 test-clean 录音整句解码，报告语料级 PER。比较放置深度时，以同一个 30 秒参考输入逐层估算编码器和融合的主要 MAC。比较完整系统时，以实际音频时长归一化 beam-20 解码的编码器与融合、预测器、联合网络计数。区分参考输入的估算与全语料计数。[固定时长表](generated/operating_points.tex) · [解码分项](generated/actual_decode.tex)
3. **结果先解释位置，再对照预算。** 先用 Mel+L3 和 Mel+L5 说明第 4、5 层的输入长度怎样影响计算，两者观测 PER 如何变化。再以同为 128 维的固定堆叠检验可替代性，最后用 256、512 维参照展示其他预算，同时交代宽度和训练步数差异。[预算比较](generated/matched_points.tex) · [配对区间](generated/paired_per.tex)
4. **分析最后讨论原因。** 用 100 句样本的窗口覆盖解释未选为中心的帧如何参与融合。逐句等长的均匀池化对照只测冻结模型的参考转写似然。结论落在各放置方案实测的计算与识别误差上；峰谷选点或学习权重各自的贡献仍待独立对照确认。[机制诊断记录](../../artifacts/research/phone69_460h_summary_20261107/mel_layer5_interpretability_20261107.json)

## 支撑主张的对照

| 问题 | 实测结果 | 解释 |
| --- | --- | --- |
| 只替换前端能否工作？ | Mel 对 256 维 stride 4 为 5.578% 对 5.599% PER，8.115 对 10.394 GMAC。 | Mel 处理 1,179 步，stride 4 处理 751 步。两个完整模型的观测 PER 接近，Mel 的算量低 21.9%。 |
| 达到参照的实测 PER 需要多少算量？ | 128 维 stride 4 对应的最低算量融合点是 Mel+L3，低 3.7%。256 和 512 维 stride 4 对应的最低算量融合点均为 Mel。 | 低预算对照的主干同为 128 维。其余两组使用各参照的观测 PER 上限，主干宽度也有差异。[配对表](generated/matched_points.tex) |

固定组采用 stride 4／128、256、512 维，512 维对应第 31 轮。融合组包括单阶段 Mel、L1、L5，以及双阶段 Mel+L3、Mel+L5。[256 维全量评测](../../artifacts/evaluations/phone69_remaining_local_cpu_20261121/REPORT.md) · [512 维检查点对照](../../artifacts/evaluations/phone69_stride4_dim512_train_best_local_cpu_20261121/REPORT.md) · [60 秒成本表](compute_quality_curve_plan.md)

## 读数与复现

- **同输入长度核验。** 从 L1 的全量逐句归档读取原始 Mel 帧数（`input_mel_frames`），按 UID 对齐宽 stride 4 和 Mel-only 的首层输入步数。逐句核对宽 stride 4 的步数等于原始帧数除以 4 向上取整。示例选择出现次数最多的输入帧数，记录中保留了该组全部 11 个输出步数。[复核数据及统计](generated/analysis.json)
- **数据入口。** 当前生成器以 11 月 17 日清单为起点，补入窄 stride 4（E01）和 256 维配置，并采用 512 维第 31 轮的全量评测。主图、主表和附录共用这 8 个配置。生成器校验归档哈希、逐句参考转写、检查点标识和错误总数。[当前分析和入稿 ID](generated/analysis.json) · [11 月 17 日历史清单](../../artifacts/research/phone69_paper_results_table_20261117/data.json) · [完整归档](../../artifacts/research/phone69_all_results_20261117/REPORT.md)
- **固定输入：** 按[计数协议](../../docs/phone69_fixed_duration_macs.md)，对每个 checkpoint 的每个压缩位置，以全部 2,620 句的输出帧数之和除以该位置输入帧数之和；依网络顺序传播 3,001 帧，逐层计算算子。附录绝对值表的 GMAC/s 列将 30 秒成本除以 30。30 秒见 [Fast Conformer](https://arxiv.org/html/2305.05084v4) 和 [Zipformer](https://arxiv.org/html/2310.11230v4)；各论文的 MAC/FLOP 范围不相同。
- **真实解码：** 用整句 FP32、batch 1、beam 20 的实际计数，分别给 encoder＋fusion、predictor、joiner、总量。说明是每个真实音频秒的主要 MAC，不把它当固定输入理论值、设备耗时或速度倍数。
- **配对与分组：** [生成脚本](build_current_results.py)校验预测文件哈希、全部 UID、参考转写和错误数；按说话人配对重采样 10,000 次（种子 42）。同一脚本复用冻结语速十等分的成员，校验三组重新合并后复现整集错误数与实际 MAC。[配对区间及三组完整计数](generated/analysis.json)可独立核查。窗口与冻结替换数值来自[100 句 / 37 人机制诊断](../../artifacts/research/phone69_460h_summary_20261107/mel_layer5_interpretability_20261107.json)。
- 从项目根运行 `python papers/extremafuse_acl2027/build_current_results.py`（NumPy、Matplotlib），生成结果图、方法网络图的矢量 PDF 和高清 PNG，以及五份 LaTeX 表行。结果图按未舍入数值选点，方法图读取 Mel+L5 的 30 秒逐层帧数。编译入口为 `main.tex`。用 Tectonic 在本目录编译即能更新整稿 PDF。表 1 的 PNG 取自当前编译稿。[当前配对表预览](figures/matched_operating_points.png) · [11 月 12 日历史八点图](fixed_duration_curve_20261112/REPORT.md)

## 投稿前优先补的证据

1. 在同一 128 维主干、同一输入位置和相近的帧数与训练预算下，分别训练固定周期池化和内容自适应融合，对比整句 PER。这个控制可以分辨多少收益来自时间步规则。现有 100 句等长度对照测量冻结模型的参考序列似然。
2. 在独立的 Phone69 测试集上确认效果。先在开发集冻结配置、保留率和预算，再用多个随机种子估计不同检查点之间的变化。

稿件使用匿名 ACL 双栏格式，正文之后单列 Limitations，参考文献之后放附录。[ACL 2027 主会征稿](https://2027.aclweb.org/calls/main/)和 [ARR 征稿要求](https://aclrollingreview.org/cfp)给出长文正文上限及必需章节；8 页是上限，不是写满的目标。
