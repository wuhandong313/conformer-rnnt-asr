# ExtremaFuse · ACL 2027 长文写作框架

**工作题目**　*ExtremaFuse for Adaptive Compression of Frame-Level Speech Representations*

**论文问题**　在相同解码协议下，按声学内容选择位置并融合邻近帧，能否用更少的主要乘加量达到固定步长系统的音素识别质量门槛？如果第二次压缩进一步缩短了序列，哪些输入仍参与输出、哪些识别错误随之改变？

英文稿写在 `main.tex`，用内置 LaTeX 编辑器编译并查看当前预览。`% FILL` 注释标明待补证据；磁盘上的 `main.pdf` 是历史导出，未包含本次删选与重编号。`references.bib` 保存实际引用的文献。实验来源仍以原始记录为准。[英文主文件](main.tex) · [引文库](references.bib) · [原编号权威指标](../phone69_460h_cpu_authoritative_20261108/summary.csv) · [逐项比较与写作笔记](../extremafuse_results_and_writing_20261108.md) · [补实验计划](../phone69_acl_2027_plan_20261107.md)

## 当前论文实验与编号

仅保留以下六组进入正文、附录和结果图。当前论文编号与原始归档编号分开；原始配置、checkpoint、评测记录与文件名保留归档编号。计算快照中的 `id` 为论文编号，`archive_id` 为溯源编号。

| 论文编号 | 论文名称 | 原归档编号 |
| --- | --- | --- |
| E01 | ExtremaFuse-L2 | E02 |
| E02 | ExtremaFuse-Mel | E05 |
| E03 | ExtremaFuse-L5 | E06 |
| E04 | ExtremaFuse-L7 | E07 |
| E05 | Stride4（主基线） | E09 |
| E06 | ExtremaFuse-Mel-L5（双阶段） | E10 |

原 E01、E03、E04、E08 从论文移除。原 E11 继续仅作内部参考。已完成的十组全量核验原件保留，论文只读取上述六组的快照。[计算量—性能曲线方案](compute_quality_curve_plan.md)

## 七节正文与版面预算

效率指标统一为按实际帧长和 beam 搜索路径计算的主要 GMAC／输入音频秒，分列编码器及压缩、预测器及 joiner。位置实验使用 E05 的计算量预算，论文不再报告 RTF 或安排重复计时。[计数方案](../phone69_460h_compute_eval_plan_20261108.md) · [全量计数快照与来源哈希](compute_snapshot.json)

下面的页数是写作分配，图表占用计入所属节。合计 7.75 页，另留 0.25 页给双栏浮动图表和换栏。

| 节 | 页数 | 读者应得到的答案 | 正文安排 |
| --- | ---: | --- | --- |
| 1 Introduction，含标题和摘要 | 1.10 | 为什么统一下采样可能损失有用的音素线索？如何判断自适应压缩值得采用？ | 从语音的时间分辨率切入，引出质量约束下的解码成本。结尾给出方法、系统比较和机制证据三项贡献；摘要最后写。 |
| 2 Related Work | 0.60 | 已有时间降采样和相似度合并如何选择、保留或汇总帧？ | 比较固定步长、Transducer 时间压缩、A-ToMe，以及从局部声学变化提取候选位置的工作。交代本研究把自然峰谷选点和五帧学习融合合为一个算子。 |
| 3 ExtremaFuse | 1.20 | 给定任意有序帧序列，如何得到更短、维度相同的表示？ | 图 1 与式 1。先定义输入、输出和长度，再给相邻帧余弦距离、峰谷到中心的映射、去重、窗口掩码和权重归一化。算子定义独立于 Conformer。 |
| 4 Experimental Setup | 1.35 | 各系统匹配了哪些条件，哪些量仍有差异？ | 460h 英语 Phone69、Conformer–RNN-T 验证载体、选模与训练预算、两处均匀池化、完整 MAC 计数范围及配对统计。列清输入投影和主干宽度。 |
| 5 Results | 1.40 | 达到怎样的识别质量，花了多少主要乘加量？ | 表 1 先比较 E06 与 E05，再看 E02、E03 及其他单阶段位置。图 2 给出六个实测工作点及配对区间。入选六组 test-clean 计数已完成，训练后的均匀池化和独立 test-other 留位。 |
| 6 Analysis | 1.80 | 二次压缩省下哪些时间步？哪些帧参与输出？收益和代价集中在哪类错误？ | 依次写 E02→E06、窗口覆盖与冻结替换、自然语速及替换／删除／插入。图 3 给出真实融合权重；语速表报告快慢差异的区间。 |
| 7 Conclusion | 0.30 | 在什么训练和评测条件下，观察到了什么，又能解释到哪一步？ | 用一段收束两个问题。质量门槛、池化对照和独立集结果到位后再定最终结论。 |

正文后依次放 `Limitations`、酌情保留的 `Ethical Considerations`、参考文献与附录。附录 A 追踪入选六组的训练配置、checkpoint、优化步数和计数范围；附录 B 展开十档语速、音素级错误与示例选取。读者仅凭正文就应能判断 E05 的容量差异、主比较的公平性和关键证据的力度。

## 每组证据填在哪里

| 位置 | 已有证据 | 定稿条件 |
| --- | --- | --- |
| §5 主比较，E06 对 E05 | E05 与 E06 输入投影均为 128。test-clean PER 分别为 5.7427% 和 5.6882%；ΔPER 为 −0.054 个百分点，95% 区间 [−0.192, +0.076]。E06 的仅编码器／合计为 0.13207／0.23252，E05 为 0.80207／0.92697 GMAC／音频秒。两项分别减少 83.53% 和 74.92%。 | 在 dev 阶段冻结允许的 PER 损失 δ，独立集检验质量门槛；报告独立集上的 PER 区间和计算量比。 |
| §5 单阶段工作点 | E02 作为直接前端替换，PER 为 5.5782%，计算量为 0.38207 GMAC／音频秒。E03 相对 E05 PER 低 0.962 个百分点，E03 为 0.62512 GMAC／音频秒。 | 列明各自匹配的条件和训练步数。E05 为 512 维、29.58M 参数；E03 的合计为 E05 的 67.44%，属于预算内高精度工作点。 |
| §5 独立训练的均匀池化 | 原验证计划新增的一组训练。与 E06 使用相同两处插入位置和 128 维主干，各阶段将有效序列分成连续、不重叠的段并取算术平均。曲线扩展采用固定 10 层、投影 128 的 stride1／2／8 新训练，另见专项方案。 | 在 dev-clean 校准并固定两阶段保留率，训练后同口径比较实际帧数、PER、主要 MAC 和训练预算。若结果相近，如实报告普通池化也能保留的性能。 |
| §6.1 第二次压缩，E02 到 E06 | test-clean 的 Mel 阶段均输出 764,950 步；E06 在第 5 层后剩 307,404 步，减少 59.81%。PER 从 5.5782% 到 5.6882%，差值为 +0.110 个百分点，区间 [−0.011, +0.232]。 | 明示 E02 和 E06 各自训练过。这里量化进一步压缩的代价，与 §5 的主基线比较回答不同问题。 |
| §6.2 输入覆盖和同长度干预 | 十个语速档各取 10 句，覆盖 37 位说话人。第 5 层窗口覆盖该阶段 98.24% 的输入位置，97.03% 至少一次获得 0.05 权重，每个输出的平均有效输入数为 3.89（权重平方和的倒数）。冻结 E06，仅把第 5 层换成逐句同长度的均匀池化，参考转写 NLL 从 0.1423 到 0.1435 nats／音素。 | 用权重图解释少量输出如何汇集邻近输入；冻结替换的 NLL 差值区间为 [−0.00241, +0.00499]。完整系统能否优于均匀池化由 §5 的独立训练对照回答。 |
| §6.3 语速和错误 | test-clean 相对 E05，E06 在慢、中、快三组分别多 13、多 48、少 164 个错误。快速组减少 504 个删除，同时增加 85 个替换与 255 个插入。快组相对慢组的 PER 变化差为 −0.273 个百分点，区间 [−0.697, +0.154]。 | 在 dev-clean 冻结自然语速分组边界，在独立集复核。语速按含停顿的整句时长计算；报告组内参考音素数、三类错误和快慢组差值，呈现一个受益例和一个失误例。 |

主表 PER 来自完整 test-clean 的 2,620 句；入选六组的主要 MAC 均来自同一批完整音频，计数及相对 E05 的比值已核验。40 位说话人的配对区间仅反映固定 checkpoint 的样本波动。正文披露单 seed、历史训练步数、容量差异、单一离线骨干及英语朗读任务，并列出未计入主要 MAC 的算子。[权威评测协议](../phone69_460h_cpu_authoritative_20261108/README.md) · [配置与优化历史](../../docs/phone69_460h_results.md) · [计数快照](compute_snapshot.json) · [机制数据](../../artifacts/research/phone69_460h_summary_20261107/mel_layer5_interpretability_20261107.json)

## 已录用论文怎样组织证据

| 论文与发表场次 | 原文的安排 | 本稿借用的写法 |
| --- | --- | --- |
| *Phonotomizer*，ACL 2025 主会 | §3 说明系统设计和数据流，§4 交代实验设置，§5 分开报告模型大小、对齐和聚类结果。 | §3 让读者先能从输入重建算子，§4 再给验证系统；主表把质量与工程成本并列。[ACL Anthology](https://aclanthology.org/2025.acl-long.592/) |
| *PRiSM*，ACL 2026 主会 | §3 定义音素实现评测框架，§5 报告结果，§6.1 单列音系规律与声学信号的诊断。 | §5 用 PER 回答识别任务问题；§6 再用错误类型和语速讨论差异来自哪里。[ACL Anthology](https://aclanthology.org/2026.acl-long.825/) |
| *Do We Need Distinct Representations for Every Speech Token?*，Findings of ACL 2026 | §3 研究语音表示冗余，§4 做受控干预，§5 同时报告下游任务和实际效率。 | 把冻结模型的同长度替换作为机制诊断；独立训练的池化与完整 MAC 分项各自给出结果。[ACL Anthology](https://aclanthology.org/2026.findings-acl.742/) |
| *Massive End-to-end Speech Recognition Models with Time Reduction*，NAACL 2024 主会 | §3 分析减少时间步与识别精度，§5.7 专门报告推理计时，§6 展开 LibriSpeech 实验。 | §6.1 同时给时间步、主要 MAC 和 PER，主表说清计数范围。[ACL Anthology](https://aclanthology.org/2024.naacl-long.344/) |
| *Stride-k Subsampling*，EMNLP 2026 主会 | §3 区分不同位置的压缩，§5 分析相邻表示，§6 再研究两处组合。 | 用 E02→E06 单独回答第二次压缩的增量问题，保留 E06 对 E05 作为系统主比较。[ACL Anthology](https://aclanthology.org/2026.emnlp-main.1608/) |

## ACL 格式与填写顺序

- 按 ACL 2027 主会长文准备，建议选 `Speech Recognition, Text-to-Speech and Spoken Language Understanding`。官方列出的最迟 ARR 周期截止时间是 2027 年 1 月 4 日 23:59 AoE，即北京时间 1 月 5 日 19:59。取得 ARR 评审及 meta-review 后再向 ACL commitment。[ACL 2027 主会征稿](https://2027.aclweb.org/calls/main/)
- 匿名审稿稿最多 8 页正文，含摘要、相关工作、正文图表和结论。摘要用纯文本，最多 200 个英文词。`Limitations` 必需，放在结论之后、参考文献之前，只讨论正文工作的局限。它及可选的伦理讨论与参考文献不计入正文 8 页；附录放在参考文献之后，审稿人可自行决定是否阅读。[ARR 征稿要求](https://aclrollingreview.org/cfp) · [ACLPUB 排版规范](https://acl-org.github.io/ACLPUB/formatting.html)
- ACL 提供 LaTeX 和 Word 模板，投稿文件为 PDF。本项目使用官方 LaTeX 模板，保留 `\usepackage[review]{acl}` 的双栏和行号。匿名稿隐去作者、单位和致谢，正文与附录所链接的材料一并匿名化。图表题注放在图表下方。模板文件取自官方提交 `d5adc823`。[官方模板版本](https://github.com/acl-org/acl-style-files/tree/d5adc823ff0f80f98c80405ca0ab66c68e684409) · [审稿稿格式](https://acl-org.github.io/ACLPUB/formatting.html) · [匿名要求](https://aclrollingreview.org/cfp)
- 提交时另填 Responsible NLP Checklist，最迟在投稿截止后 48 小时内完成；按 ARR 要求说明用于写作或编码的生成式工具及使用范围。[ARR 日程](https://aclrollingreview.org/dates) · [ARR 关于生成式工具的说明](https://aclrollingreview.org/cfp)

填写顺序从 §3 算子和 §4 可复现实验协议开始。在独立评测前固定 PER 门槛、保留率、语速边界和计算量预算，核实 test-other 的音频与 Phone69 标签逐句匹配，并确认其历史选模用途。随后补 §5 主表，再把 §6 的帧数、权重和错误落到图表上。最后回写引言、摘要、结论，使各处对优势的表述与同一组证据对应。

本次预览使用临时 TinyTeX 环境编译。日常可在 Overleaf 以 pdfLaTeX 编译，只需导入本目录的 `main.tex`、`references.bib`、`acl.sty`、`acl_natbib.bst` 和 `figures/`。若本地已安装带 `latexmk` 的 TeX Live 或 MacTeX，在本目录运行以下命令。

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

图 2 由 `compute_snapshot.json` 与 `plot_quality_compute.py` 生成：`python plot_quality_compute.py`。该快照保留全量计数文件和 PER／区间来源的 SHA-256；未完成模型用空值标记，图只绘制已完成全量计数的模型。全量计数补齐后，需同时更新快照、正文表格、当前结果入口及摘要，不混用 24 句试测值。
