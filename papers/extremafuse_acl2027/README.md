# ExtremaFuse · ACL 2027 长文

**[英文稿 · Replacing Fixed-Stride Frame Stacking with Content-Adaptive Fusion for Phone Recognition](main.tex)**

当前八页稿以英文源稿的编辑器预览为准。结果曲线、方法网络图分别在第 2、3 页，结论在第 6 页。目录中的历史整稿 PDF 导出于 11 月 12 日。[本轮编译核验](build_20261118.json)

**论文要回答的问题。** 语音编码器需要压短时间轴，输出步一定要按固定间隔产生吗？本文用 ExtremaFuse 替代固定步长的帧堆叠。输入仍是每 10 毫秒一帧，输出中心取自相邻帧余弦距离的局部峰谷。每个中心通过已学习的权重汇聚五帧。实验先检验单个输入端模块能否承担下采样，再检验不同计算预算下的替代效果。

## 借鉴的 ACL 写法

- **从旧规则的具体行为引出替代。** Sennrich 等用固定词表如何处理罕见词引出子词建模，又回到罕见词检验效果。我们的开头用 11 条同为 272 帧的真实录音说明固定堆叠怎样分组，并给出替代后实际产生的步数。对应的第一个实验就是输入端 Mel 融合对四帧堆叠。OpenAlex 于 2026-11-18 记录该文 10,063 次引用。[ACL 2016 原文](https://aclanthology.org/P16-1162/) · [OpenAlex](https://openalex.org/W1816313093)
- **让读者看到条件计算的整条曲线。** DeeBERT 先说明一个输入如何提前退出，再用图 2 展示不同退出阈值下的质量与成本。我们的曲线给出 12 个单独训练的工作点，报告整句 PER 与标准化的计算量。OpenAlex 同日记录 DeeBERT 693 次引用。[ACL 2020 原文，图 2](https://aclanthology.org/2020.acl-main.204.pdf#page=4) · [OpenAlex](https://openalex.org/W3035038672)
- **从曲线上按明确约束取点。** Kim 和 Cho 在图 2 和表 1 中比较准确率与成本，并选出达到标准模型准确率的低成本配置。本文分别问预算内最低 PER 是多少、达到基线实测 PER 时最低成本是多少。OpenAlex 同日记录该文 132 次引用。[ACL-IJCNLP 2021 原文](https://aclanthology.org/2021.acl-long.508/) · [OpenAlex](https://openalex.org/W3174708387)

## 正文如何推进

1. **引言**从输入帧数相同的 11 条录音写起。272 是 test-clean 中最常见的 Mel 长度，stride 4 一律给 68 步，输入端融合实际给出 97–115 步。第 3、4 帧间的变化落在同一个固定输出步，推迟一帧就跨越两步。由此提出能否让声学内容决定输出中心。
2. **相关工作和方法**交代已有固定步长、分层降采样与 A-ToMe 等内容自适应方法，再定义我们的输出规则。A-ToMe 按注意力键的相似度合并相邻 token。ExtremaFuse 在距离峰和谷选中心，汇聚可重叠的五帧窗口，原始 Mel 即可作为输入。图 2 给出单个算子和组成两阶段识别器的方式。[方法网络图](figures/extremafuse_network.pdf)
3. **评估**固定 460 小时训练语料、验证损失选模、整句 test-clean 和 beam 20。主比较用 30 秒输入的编码器及融合 GMAC。固定堆叠的主干为 128／256／512 维，融合主干为 128 维。真实解码的 predictor、joiner 和总 GMAC/音频秒另列于附录。
4. **结果**先核对所有具有相同 Mel 帧数的组。636 个被至少两条录音共有的输入长度中，629 个出现不同的融合输出步数。表 1 对 stride 4／128 维、stride 4／512 维和 stride 2／256 维分别查找算量最近与 PER 最近的融合系统。算量对齐时比较 PER，配对 95% 说话人区间排除零才加粗较低 PER。PER 对齐时加粗该对中较低的 GMAC。两项均在九个融合候选的未舍入读数上选取。另报同一预算或 PER 上限下的选择。[配对表](generated/matched_points.tex) · [表 1 预览](figures/matched_operating_points.png) · [结果曲线](figures/quality_budget_30s.pdf) · [13 个系统的绝对值](generated/operating_points.tex)
5. **分析**先说明哪些输入帧真正参与了输出。随后用等输出长度的均匀池化检验中心和权重的作用，再分析音素错误和插入位置。474 与 483 帧的例子放在这里，解释第二次融合提前后第 4、5 层各自少处理多少帧。
6. **结论**回到采用固定堆叠的读者所面临的选择。输入端即可替换，延后或增加融合给出不同预算下的工作点。当前系统级对照同时改变了时间分辨率和部分主干宽度。

## 支撑主张的对照

| 问题 | 实测结果 | 解释 |
| --- | --- | --- |
| 只替换前端能否工作？ | 同为 128 维时，Mel 对 stride 4 为 5.578% 对 6.129% PER，8.115 对 4.336 GMAC。将 stride 4 主干扩大到 512 维，得到 5.743% 和 27.679 GMAC。 | 两个 stride 4 都处理 751 步，Mel 处理 1,179 步。Mel 相对窄基线用更多算量降低 PER，相对宽基线节省 70.7% 的计算量。 |
| 面对最强固定步长呢？ | 用 stride 2 的实测 PER 作上限，L5 以 21.380 对 26.539 GMAC 达标，少 19.4%。 | PER 为 4.781% 对 4.899%。对照的区间 [−0.247, +0.004] 跨零。若把计算量也拉到几乎相同，L7 与 stride 2 只有 0.039 个百分点的 PER 差。 |
| 在相近实测 PER 下哪一侧省算量？ | 相对宽 stride 4，Mel+L5 的 GMAC 低 81.0%。相对窄 stride 4，Mel+L3 低 3.7%。与 stride 2 的 PER 最接近的是 L9，L9 的 GMAC 高 20.0%。 | 三行均按实测 PER 绝对差选择配对，方向由各对的实测成本决定。[十组配对区间](generated/paired_per.tex) |
| 在相同主干宽度下，压缩是否仍然可行？ | L5 对无压缩的 stride 1：4.781% 对 4.986% PER，21.380 对 34.612 GMAC。 | 两者都是 128 维，算量少 38.2%。这组对照回答保留所有时间步有没有必要。 |

全文包含 13 个按验证损失选 checkpoint 的系统。图 1 展示每帧投影均为 128 维的十二组，包括原投影配置下的 stride 1／2／4 和九种融合。配对表及附录另纳入每帧投影 32 维的 stride 4／128 维。L3 使用第 27,324 步的验证最佳 checkpoint。

## 读数与复现

- **同输入长度核验。** 用逐句 UID 对齐 stride 1、stride 4、Mel-only 的归档帧数，逐句核对 stride 4 的输出步数等于输入帧数除以 4 向上取整。示例选择出现次数最多的输入帧数，记录中保留了该组全部 11 个输出步数。[复核数据及统计](generated/analysis.json)
- **数据入口。** 图 1 的十二组读取原入选清单，表格另外读取完整归档中的窄 stride 4（归档 E01）。生成器校验归档哈希、逐句参考转写、验证选模和错误总数。[图 1 清单](../../artifacts/research/phone69_paper_results_table_20261117/data.json) · [完整归档](../../artifacts/research/phone69_all_results_20261117/REPORT.md)
- **固定输入：** 按[计数协议](../../docs/phone69_fixed_duration_macs.md)，对每个 checkpoint 的每个压缩位置，以全部 2,620 句的输出帧数之和除以该位置输入帧数之和；依网络顺序传播 3,001 帧，逐层计算算子。GMAC/30 s 除以 30 得主表最后一列，不能由变长解码成本外推。30 秒见 [Fast Conformer](https://arxiv.org/html/2305.05084v4) 和 [Zipformer](https://arxiv.org/html/2310.11230v4)；各论文的 MAC/FLOP 范围不相同。
- **真实解码：** 用整句 FP32、batch 1、beam 20 的实际计数，分别给 encoder＋fusion、predictor、joiner、总量。说明是每个真实音频秒的主要 MAC，不把它当固定输入理论值、设备耗时或速度倍数。
- **配对与分组：** [生成脚本](build_current_results.py)校验预测文件哈希、全部 UID、参考转写和错误数；按说话人配对重采样 10,000 次（种子 42）。同一脚本复用冻结语速十等分的成员，校验三组重新合并后复现整集错误数与实际 MAC。[配对区间及三组完整计数](generated/analysis.json)可独立核查。窗口与冻结替换数值来自[100 句 / 37 人机制诊断](../../artifacts/research/phone69_460h_summary_20261107/mel_layer5_interpretability_20261107.json)。
- 从项目根运行 `python papers/extremafuse_acl2027/build_current_results.py`（NumPy、Matplotlib），生成结果曲线、方法网络图的矢量 PDF 和高清 PNG，以及五份 LaTeX 表行。曲线与放大图使用相同的未舍入坐标，方法图读取 Mel+L5 的 30 秒逐层帧数。编译入口为 `main.tex`。表 1 的 PNG 从本次编译第 4 页裁出。[11 月 12 日九点图](fixed_duration_curve_20261112/REPORT.md) · [历史导出记录](build_20261112.json)

## 投稿前优先补的证据

1. 在同一 128 维主干、同一输入位置和相近的帧数与训练预算下，分别训练固定周期池化和内容自适应融合，对比整句 PER。这个控制可以分辨多少收益来自时间步规则。现有 100 句等长度对照测量冻结模型的参考序列似然。
2. 在独立的 Phone69 测试集上确认效果。先在开发集冻结配置、保留率和预算，再用多个随机种子估计不同检查点之间的变化。

稿件使用匿名 ACL 双栏格式，正文之后单列 Limitations，参考文献之后放附录。[ACL 2027 主会征稿](https://2027.aclweb.org/calls/main/)和 [ARR 征稿要求](https://aclrollingreview.org/cfp)给出长文正文上限及必需章节；8 页是上限，不是写满的目标。
