# ExtremaFuse · ACL 2027 长文初稿

[阅读英文初稿 PDF](main.pdf) · [LaTeX 源文件](main.tex) · [13 组结果主图](figures/quality_budget_30s.pdf)

本文研究离线音素识别中的一个设计选择：编码器能否根据声学变化决定下一步的位置，并在相同的误差要求下减少计算？实验覆盖 460 小时 LibriSpeech 训练数据和 2,620 条完整 test-clean 录音。

## 近期 ACL 论文如何影响本稿

- Feher 等从同一段文本在不同切分规则下会变成多少个 token 写起，再检验压短后能保留多少任务性能。引言采用同样可直接观察的切入点。11 条均为 272 帧的录音经 stride 4 都得到 68 步，经 Mel 融合得到 97–115 步。这个例子说明时间网格会随内容改变，识别质量交给后续实验检验。[Feher 等，ACL 2025](https://aclanthology.org/2025.acl-long.1444/)
- Ji 等研究视觉 token 在不同层的裁剪配置，按预算与实测前沿核对结果。本文对每个已训练的放置方案逐层计数，用两条阶梯回答给定 PER 上限时每一组最少花多少 GMAC。固定步长目前只有两个非支配点，图上的大台阶反映基线覆盖稀疏。[Ji 等，ACL 2026](https://aclanthology.org/2026.acl-long.420/)
- Liao 等发现简单图像降采样在多项基准上能超过复杂压缩。对应到本文，宽、窄 stride 4 都进主图，并保留逐句输出长度完全一致的均匀池化对照。前者检验完整识别器，后者检验训练好的模型对汇聚规则有多敏感。[Liao 等，ACL 2026](https://aclanthology.org/2026.acl-long.195/)

## 正文如何推进

1. **从固定堆叠提出问题。** 四帧拼接保留帧里的数值，但堆叠边界跟着步长走。在相邻帧余弦距离的峰谷取中心、学习五帧混合权重，同一模块就可以在 Mel 输入或 Conformer 隐层工作。方法紧接引言，网络图展示中心位置、重叠窗口及各层所见的序列长度。[方法图](figures/extremafuse_network.pdf)
2. **用同一把尺衡量替代效果。** 整句解码得到 PER，以实测保留率推算 30 秒参考输入的编码器与融合 GMAC。主图含四个固定配置和九个融合配置。表 1 在未舍入数据中取算量最近或 PER 最近的一对，再比较另一项指标。正文另问在给定上限时哪一个模型最省，两种选择规则可得到不同结果。[配对表](generated/matched_points.tex) · [全部绝对值](generated/operating_points.tex)
3. **先看真正有竞争力的固定基线。** 约 26.5 GMAC 时，L7 和 stride 2 的 PER 只差 0.039 个百分点。换成以 stride 2 的实测 PER 为上限，L5 达标并少用 19.4% 算量。宽 stride 4 与 L7 的差距更大，窄 stride 4 又把固定基线带到约 4.3 GMAC。主图同时保留这些点，读者能看清基线宽度改变了哪些结论。[结果曲线](figures/quality_budget_30s.pdf)
4. **用具体帧数解释位置和误差。** 第二处融合从 L5 提前到 L3，最终步数很接近，但第 4、5 层各从 1,179 步降至 483 步。再问压缩后的向量是否仍在利用输入。100 句冻结实验将第二阶段换成每句等长的均匀池化后，参考转写负对数似然差的区间跨零。它还不能确定峰谷选点本身贡献了多少 PER。[机制诊断记录](../../artifacts/research/phone69_460h_summary_20261107/mel_layer5_interpretability_20261107.json)
5. **把结论放回既有研究。** 相关工作交代固定网格、预设总压缩率的自适应步长和本文由极值数量确定步数的区别。结论落到离线 Phone69 这一任务，同时给出下一项能检验机制归因的训练对照。[TadaStride](https://doi.org/10.1016/j.csl.2024.101678) · [Stride-k](https://aclanthology.org/2026.emnlp-main.1608/)

## 支撑主张的对照

| 问题 | 实测结果 | 解释 |
| --- | --- | --- |
| 只替换前端能否工作？ | 同为 128 维时，Mel 对 stride 4 为 5.578% 对 6.129% PER，8.115 对 4.336 GMAC。将 stride 4 主干扩大到 512 维，得到 5.743% 和 27.679 GMAC。 | 两个 stride 4 都处理 751 步，Mel 处理 1,179 步。Mel 相对窄基线用更多算量降低 PER，相对宽基线节省 70.7% 的计算量。 |
| 面对最强固定步长呢？ | 用 stride 2 的实测 PER 作上限，L5 以 21.380 对 26.539 GMAC 达标，少 19.4%。 | PER 为 4.781% 对 4.899%。对照的区间 [−0.247, +0.004] 跨零。若把计算量也拉到几乎相同，L7 与 stride 2 只有 0.039 个百分点的 PER 差。 |
| 在相近实测 PER 下哪一侧省算量？ | 相对宽 stride 4，Mel+L5 的 GMAC 低 81.0%。相对窄 stride 4，Mel+L3 低 3.7%。与 stride 2 的 PER 最接近的是 L9，L9 的 GMAC 高 20.0%。 | 三行均按实测 PER 绝对差选择配对，方向由各对的实测成本决定。[十组配对区间](generated/paired_per.tex) |
| 在相同主干宽度下，压缩是否仍然可行？ | L5 对无压缩的 stride 1：4.781% 对 4.986% PER，21.380 对 34.612 GMAC。 | 两者都是 128 维，算量少 38.2%。这组对照回答保留所有时间步有没有必要。 |

四个固定系统和九个融合系统都按 dev-clean 验证损失选 checkpoint。单阶段 L3 使用第 27,324 步的原验证最佳。stride 4／256 维和 L3 重训尚无新一轮完整评估读数。[后续实验及 60 秒成本表](compute_quality_curve_plan.md)

## 读数与复现

- **同输入长度核验。** 用逐句 UID 对齐 stride 1、stride 4、Mel-only 的归档帧数，逐句核对 stride 4 的输出步数等于输入帧数除以 4 向上取整。示例选择出现次数最多的输入帧数，记录中保留了该组全部 11 个输出步数。[复核数据及统计](generated/analysis.json)
- **数据入口。** 主图和表格共同使用原清单的十二组及完整归档中的窄 stride 4（E01）。生成器校验归档哈希、逐句参考转写、验证选模和错误总数。[原十二组清单](../../artifacts/research/phone69_paper_results_table_20261117/data.json) · [完整归档](../../artifacts/research/phone69_all_results_20261117/REPORT.md)
- **固定输入：** 按[计数协议](../../docs/phone69_fixed_duration_macs.md)，对每个 checkpoint 的每个压缩位置，以全部 2,620 句的输出帧数之和除以该位置输入帧数之和；依网络顺序传播 3,001 帧，逐层计算算子。附录绝对值表的 GMAC/s 列将 30 秒成本除以 30。30 秒见 [Fast Conformer](https://arxiv.org/html/2305.05084v4) 和 [Zipformer](https://arxiv.org/html/2310.11230v4)；各论文的 MAC/FLOP 范围不相同。
- **真实解码：** 用整句 FP32、batch 1、beam 20 的实际计数，分别给 encoder＋fusion、predictor、joiner、总量。说明是每个真实音频秒的主要 MAC，不把它当固定输入理论值、设备耗时或速度倍数。
- **配对与分组：** [生成脚本](build_current_results.py)校验预测文件哈希、全部 UID、参考转写和错误数；按说话人配对重采样 10,000 次（种子 42）。同一脚本复用冻结语速十等分的成员，校验三组重新合并后复现整集错误数与实际 MAC。[配对区间及三组完整计数](generated/analysis.json)可独立核查。窗口与冻结替换数值来自[100 句 / 37 人机制诊断](../../artifacts/research/phone69_460h_summary_20261107/mel_layer5_interpretability_20261107.json)。
- 从项目根运行 `python papers/extremafuse_acl2027/build_current_results.py`（NumPy、Matplotlib），生成结果曲线、方法网络图的矢量 PDF 和高清 PNG，以及五份 LaTeX 表行。曲线与放大图使用相同的未舍入坐标，方法图读取 Mel+L5 的 30 秒逐层帧数。编译入口为 `main.tex`。用 Tectonic 在本目录编译即能更新整稿 PDF。表格 PNG 为 11 月 18 日导出。[11 月 12 日九点图](fixed_duration_curve_20261112/REPORT.md) · [历史导出记录](build_20261112.json)

## 投稿前优先补的证据

1. 在同一 128 维主干、同一输入位置和相近的帧数与训练预算下，分别训练固定周期池化和内容自适应融合，对比整句 PER。这个控制可以分辨多少收益来自时间步规则。现有 100 句等长度对照测量冻结模型的参考序列似然。
2. 在独立的 Phone69 测试集上确认效果。先在开发集冻结配置、保留率和预算，再用多个随机种子估计不同检查点之间的变化。

稿件使用匿名 ACL 双栏格式，正文之后单列 Limitations，参考文献之后放附录。[ACL 2027 主会征稿](https://2027.aclweb.org/calls/main/)和 [ARR 征稿要求](https://aclrollingreview.org/cfp)给出长文正文上限及必需章节；8 页是上限，不是写满的目标。
