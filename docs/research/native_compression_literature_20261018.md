# 自然峰谷压缩论文：文献核查与设计依据

检索日期：2026-10-18，Asia/Shanghai。只将论文作者原文、官方论文集和会议官方网站作为论证来源。下表区分正文核查与摘要核查；摘要核查不能冒充完整复现。这里记录的是论文设计依据，没有将文献中的结果当作本项目的实验结果。

主设计：[自然峰谷动态压缩研究规划](../dynamic_compression_paper_plan_20261018.md)。

## 1. 最近的识别系统工作

| ID | 核查的一手来源 | 元数据/阅读范围 | 对本研究设计的具体影响 |
|---|---|---|---|
| L01 | [Accelerating Transducers through Adjacent Token Merging](https://arxiv.org/html/2306.16009v1)；[ISCA正式条目](https://www.isca-archive.org/interspeech_2023/li23i_interspeech.html) | Yuang Li、Yu Wu、Jinyu Li、Shujie Liu；Interspeech 2023。正文方法、配置与实验已核查 | 邻接合并、Transducer效率及可变帧率已经存在。采用其适配为核心强基线，区分原始逐层合并与本项目单处极值/五帧融合，不直接比较跨模型绝对WER |
| L02 | [Accelerating RNN-T Training and Inference Using CTC guidance](https://arxiv.org/html/2210.16481v1)；[元数据](https://arxiv.org/abs/2210.16481) | Yongqiang Wang等；2022预印本，页面注明submitted to ICASSP 2023。核查正文/摘要；不从“submitted”推断发表状态 | 中间encoder帧裁减不是新概念。比较CTC-guided方案时，增加相同CTC监督但不裁减的控制，统计head和选择器成本 |
| L03 | [Key Frame Mechanism For Efficient Conformer Based End-to-end Speech Recognition](https://arxiv.org/html/2310.14954v1) | Peng Fan等；2023版本。正文KFSA/KFDS、辅助loss和实验设置已核查 | 已有两段Conformer及关键帧邻域信息。模块适配需要披露预热和辅助目标，不能宣称两段编码器或邻帧使用本身新颖 |
| L04 | [Efficient Conformer](https://arxiv.org/abs/2109.01163) | Maxime Burchi、Valentin Vielzeuf；arXiv元数据Journal reference为ASRU 2021。摘要和元数据已核查 | 渐进式下采样是强结构对照。不能只与不调优的硬抽帧相比；复现时需进一步核对完整配方 |
| L05 | [Squeezeformer: An Efficient Transformer for Automatic Speech Recognition](https://proceedings.neurips.cc/paper_files/paper/2022/hash/3ccf6da39eeb8fefc8bbb1b0124adbd1-Abstract.html) | Sehoon Kim等；NeurIPS 2022官方论文集。摘要及公开正文片段已核查 | 整体架构和微结构也能带来效率收益；完整模型对照与固定主干的下采样模块对照应分开标注 |

L01与L02比一般视觉token pruning更接近当前任务，应优先安排实现核查与受控比较。以其它论文94M、960h、wordpiece系统的绝对数字为4.2M、460h、phone69系统背书不成立。

## 2. 分段、非均匀表示与相邻特征变化

| ID | 一手来源 | 状态/阅读范围 | 对本研究设计的具体影响 |
|---|---|---|---|
| L06 | [Segmental Contrastive Predictive Coding for Unsupervised Word Segmentation](https://www.isca-archive.org/interspeech_2021/bhati21_interspeech.html) | Saurabhchand Bhati等；Interspeech 2021官方条目，摘要核查 | 分段表示和无监督边界学习有明确前史。当前固定极值规则与其可微边界目标不同，不能称分段/变长是首次 |
| L07 | [Variable-rate hierarchical CPC leads to acoustic unit discovery in speech](https://proceedings.neurips.cc/paper_files/paper/2022/hash/e2b0a30ea6a67cba58134e57348afb91-Abstract-Conference.html) | Santiago Cuervo等；NeurIPS 2022官方条目，摘要及补充材料开头核查 | 非均匀下采样和有意义声学单元的关系已有研究。对比的是选择机制、目标和识别效用，不把variable-rate本身当新贡献 |
| L08 | [Towards Audio Token Compression in Large Audio Language Models](https://arxiv.org/html/2511.20973v2) | Saurabhchand Bhati、Samuel Thomas、Hilde Kuehne、Rogerio Feris、James Glass；首版2025，核查v2（2026-08-20）正文§4、§5 | 原文使用相邻特征不相似度的峰值分段，并比较均匀平均/采样和学习压缩。不能把本地cosine detector注释当作原文完整复现，也不能假定学习模块必胜平均池化 |
| L09 | [FastAdaSP: Multitask-Adapted Efficient Inference for Large Speech Language Model](https://aclanthology.org/2024.emnlp-industry.33/) | Yichen Lu、Jiaqi Song、Chao-Han Huck Yang、Shinji Watanabe；EMNLP 2024 Industry Track正式条目及正文片段核查 | 加权token合并和针对任务的压缩已有先例；其训练自由/推理阶段与当前端到端训练不同，不混用效率主张 |

注意：L08原文以sim记相似度，并给出局部峰判据。本核查只将其概括为“相邻特征变化的峰值分段”，不凭代码注释将作者原文说成与本项目峰+谷、P+1/V定位和五帧条件融合完全相同。

## 3. 2026年补充工作及解释边界

| ID | 一手来源 | 状态/阅读范围 | 对本研究设计的具体影响 |
|---|---|---|---|
| L10 | [DTM-Codec: Dynamic Token Masking for VFR Speech Coding with Efficient Boundary Selection](https://www.isca-archive.org/interspeech_2026/sohn26_interspeech.html)；[算法正文](https://arxiv.org/html/2606.29480v1) | Hoyeol Sohn、Juhan Nam；Interspeech 2026官方条目确认；核查正文§3.2–3.4 | PLE为Path Length Equalization。其阈值/目标率控制与本项目自然极值定义不同。适配选择器时核对累计余量、单步跨阈值、端点与短序列；本地累计清零不能直接当作原版 |
| L11 | [Interpreting and Evaluating Dynamic-Rate Speech Codec Boundaries](https://arxiv.org/html/2609.36951v1) | Han Wang、Jiaqi Li、Yingda Shen、Yuxiang Wang、Zhizheng Wu；2026-09-29预印本，正文边界解释、匹配帧率实验及重建任务范围核查 | 表征深度改变边界含义，边界解释与效用需区分；本项目在RNN-T上直接检验识别错误，不移植其codec结论。仅凭“画边界对齐图”已不足以形成独有新意 |
| L12 | [Context-Driven Dynamic Pruning for Large Speech Foundation Models](https://arxiv.org/html/2505.18860v1) | Interspeech 2025相关版本，正文local gate predictor与模块/时间裁剪方式核查 | 表示动态计算研究的更广范围。其外部context指speaker/event等条件，与本项目已取消的右上下文保护段不是一回事；不列作只替换帧选择器的纯消融 |

近期预印本用于检查创新边界、改进实验，不把其投稿/预印本状态误写成已经被某顶会录取。codec选择器可作适配对照，但bpp/码率、波形重建与ASR实际时延是不同成本和任务。

## 4. 官方指南核查

| 来源 | 本次读取的要点 | 落实到设计 |
|---|---|---|
| [NeurIPS 2026 Reviewer Guidelines](https://neurips.cc/Conferences/2026/ReviewerGuidelines) | Quality、Clarity、Significance、Originality，主张需有合适支持 | 围绕可证伪问题、强对照、完整方法定义和贡献边界组织论文 |
| [NeurIPS 2026 Main Track Handbook](https://neurips.cc/Conferences/2026/MainTrackHandbook) | 主会checklist及复现/透明性要求 | 保存版本、配置、数据与预处理来源、计算资源和限制 |
| [ICML 2026 Reviewer Instructions](https://icml.cc/Conferences/2026/ReviewerInstructions) | Soundness、Presentation、Significance、Originality | 区分实用系统改进与机制知识，审查对照是否支撑主张 |
| [ICLR 2027 Reviewer Guidelines](https://www.iclr.cc/Conferences/2027/ReviewerGuidelines) | 清楚的问题/动机、科学严谨性、对社区的新知识；不强制以SOTA为唯一标准 | 不把小指标优化等同于顶会贡献，也不为了显得完整而堆无关实验 |

指南没有给出“达到某个PER/倍速、做够几组实验即可录取”的条件。统计重采样次数和执行顺序是本项目建议，不冒充官方门槛。按用户确定的当前范围，主设计固定seed42、phone69和当前主干，不安排多seed、BPE/WER或另一模型；相应结论范围与未知不确定性须明确。投稿前应再核对目标届次的格式与政策；本文件不提供未核实的未来截止时间。

## 5. 检索范围与复现入口

代表检索词：Adjacent Token Merging transducer；CTC guidance RNN-T；Key Frame Efficient Conformer；audio token compression segmentation pooling；variable-rate hierarchical CPC；dynamic-rate speech codec boundaries；NeurIPS 2026 Reviewer Guidelines；ICLR 2027 Reviewer Guidelines。

这不是系统综述或穷尽检索。进一步复现优先读L01/L02/L03/L08的作者代码与完整配方，固定具体版本；未实际执行作者代码前，不标为“复现成功”。

独立零假设分析可运行：

```bash
python scripts/diagnostics/analyze_native_extrema_null.py --out artifacts/research/native_compression_20261018/extrema_null_analysis.json
```

该分析只穷举四个距离值的排序并生成合成距离曲线，不读取GPU、训练配置、checkpoint、数据或凭据。它检验11/24的数学诊断值，不声称已验证真实语音中的语义适应性。
