# 论文研究计划：基于帧间差分峰谷检测的动态序列压缩

> 版本：v1.1（2026-09-25）
> 现状基线：Conformer-RNN-T + DynamicCompression（`conformer_rnnt/compression.py`），LibriSpeech 音素，指标 PER/RTF
> 目标：达到语音顶会（Interspeech / ICASSP / ASRU）接收标准，并为冲刺 ML 顶会（NeurIPS / ICLR）保留升级路径
> v1.1 变更：**流式/因果性降级为未来工作**（不再作为投稿必需项）；**泛用性验证提升为核心支柱**（跨架构 / 跨任务 / 跨数据 / 跨特征 / 跨模态五个维度），并新增"冗余自适应假说"作为贯穿性科学问题

---

## 0. 方法的一句话重新定位

目前的实现是"插在两个 Conformer block 之间的帧选择模块"。写论文时建议把它上升为一个**通用的信号驱动动态序列压缩框架（signal-driven dynamic sequence compression）**：

> 给定 encoder 中间层激活序列，用帧间差分构造 novelty signal（新颖度信号），通过可配置的信号处理滤波器族检测变化点/稳定点，在检测点处用可学习的局部加权混合替代硬选择，从而在**不改变模型参数规模**的前提下，按内容自适应地缩短序列长度，降低后续所有层的计算量。

这个定位有三个好处：

1. **跳出"ASR 小技巧"的框架**：novelty signal + peak detection 是 MIR（音乐信息检索）onset detection、时间序列 change-point detection 的经典范式，你可以引用并对话一大支成熟文献，而不是孤立地发明一个模块。
2. **天然解释所有 filter 变体**：naive / hanning / fir / savgol / savgol_hanning 不是工程调参，而是"novelty signal 的去噪/平滑策略"这一设计空间的不同实例化——这是论文里的一节，而不是附录。
3. **为泛用性主张埋下伏笔**：任何序列模态、任何 encoder、任何任务的中间层都有帧间差分信号——"泛用性"不是事后补的实验，而是定位的自然推论。

---

## 1. 现状盘点与顶会差距分析

### 1.1 已有的资产（不要低估）

| 资产 | 价值 |
|---|---|
| 完整可跑的 Conformer-RNN-T 训练/评测流水线 | 大部分竞争者没有端到端 RNN-T 实验能力 |
| 6 种 filter 模式已实现并可配置 | 消融实验已经做了一半，只差系统地跑和记录 |
| 3 种 weight predictor 模式（single_frame / diff5 / five_frames） | 现成的"可学习混合"消融轴 |
| 音素级标注 | 可做**边界对齐分析**（选中帧 vs 音素边界），这是别人没有的分析维度 |
| eval 已输出 RTF | 效率故事的起点 |

### 1.2 与顶会接收标准的硬性差距

| 维度 | 现状 | 顶会要求 | 严重度 |
|---|---|---|---|
| 架构数量 | 1（Conformer-RNN-T） | ≥3 种主流范式（Transducer / CTC / AED） | ★★★ 致命 |
| 任务数量 | 1（ASR） | 泛用性主张需要 ≥3 种不同性质的任务 | ★★★ 致命 |
| 数据集 | 1（LibriSpeech train-100，音素） | ≥3 个，覆盖多语言/噪声/不同语体 | ★★★ 致命 |
| 评测标准 | PER（音素错误率，非主流） | 必须有 word-level WER（test-clean / test-other） | ★★★ 致命 |
| 外部基线 | 无 | ≥5 个等压缩率基线 + 1–2 个已发表序列压缩方法的对比 | ★★★ 致命 |
| 压缩率控制 | 不可控（数据驱动，峰有多少压多少） | 必须有压缩率–质量的 Pareto 曲线 | ★★☆ 严重 |
| 可微性 | scipy `find_peaks`，no_grad，不可微 | 至少讨论；最好有可微变体 | ★★☆ 严重 |
| 统计严谨性 | 单种子 | 3–5 种子，mean±std，显著性检验 | ★★☆ 严重 |
| 效率度量 | 仅 RTF | FLOPs、峰值显存、吞吐、延迟分解 | ★☆☆ 中等 |
| 可解释性分析 | 无 | 选中帧的可视化 + 与语言学边界对齐的定量分析 | ★☆☆ 中等 |
| 实现效率 | per-sample Python for-loop，CPU 上跑 scipy | 方法开销必须可忽略，否则效率故事不成立 | ★★☆ 严重 |

### 1.3 审稿人可能抓的技术问题（先自查）

1. **滤波器的前瞻帧数**。`filtfilt` 是双向零相位（看整个 utterance），居中 `savgol` 看未来 ±4 帧，`_hanning_smooth` 看 ±half_win。既然不宣称流式，这不再是致命伤，但**必须在论文里用一张小表如实标注每种 filter 的算法前瞻**，避免审稿人误以为你隐含实时性主张。因果化实现留作 future work 一句话。
2. **压缩率不可控且未测量**。对类随机游走的差分信号，朴素 find_peaks 约保留 2/3 的帧（局部极大+局部极小），滤波后可能降到 20–40%。**目前连实际的压缩率分布都没有**——这必须成为第一批要跑的统计。
3. **等压缩率下的简单基线**。审稿人必问："均匀下采样 / 随机丢帧 / top-k 差分，在和你的方法保留同样帧数时，指标差多少？"如果复杂滤波打不过 top-k-diff 这种三行代码的基线，论文故事就要重写（见 §8 风险登记）。

---

## 2. 压缩方法全景（除 LLM 压缩外的所有分支）

论文 Related Work 建议按下面五分支组织。你的方法属于 **2.2 序列级压缩中的"信号驱动选择"子类**，与其他分支是"区别 + 可叠加"的关系。

### 2.1 模型级压缩（权重静态压缩）
- **剪枝**：结构化/非结构化剪枝、lottery ticket
- **量化**：INT8/INT4 PTQ 与 QAT
- **低秩分解**：SVD、低秩 adapter 化
- **知识蒸馏**：大模型 → 小模型
- **参数共享**：ALBERT 式层间共享、Universal Transformer

**与你的关系**：正交互补。你的方法压的是**激活序列长度**（动态、随输入变化），它压的是**参数量/权重精度**（静态）。**创新点 I6（§5）= 两者叠加的复合加速**，这类"序列压缩 × 权重压缩"的联合研究在语音里很少见，是一个差异化卖点。

### 2.2 序列/Token 级压缩（你的方法所属分支）
按"选择依据"分四子类：

**(a) 固定下采样（无内容自适应）**
- CNN 前端 stride 2/4 子采样（所有现代 ASR 标配）
- Low Frame Rate（LFR）帧堆叠
- 平均/最大池化、Squeezeformer 的 Temporal U-Net 池化、Efficient Conformer 的渐进式分组下采样

**(b) 学习式选择/合并（内容自适应，可微）**
- **CIF（Continuous Integrate-and-Fire）**：语音里最有代表性的学习式帧压缩，按累积权重触发发射
- **Token pruning**：DynamicViT、EViT、A-ViT（用轻量预测头决定丢哪些 token，Gumbel-softmax 可微化）
- **Token merging**：ToMe（按相似度双向匹配合并）、TokenLearner、PatchMerger（学习加权求和）
- **边界预测**：MoChA、CTC spike 触发的分段、Align-and-Refine
- **自适应计算时间**：ACT、PonderNet（学习"思考多久"）

**(c) 信号驱动选择（内容自适应，无需学习的检测器）** ← **你在这里**
- VAD / 能量门限
- **Novelty function + peak picking**：onset detection 经典范式（spectral flux、能量差分、相位差分、complex domain）——你的"帧间 L1 差分 + find_peaks"本质就是能量型 novelty detection 在 encoder 激活空间的推广
- **Change-point detection**：PELT、binary segmentation、贝叶斯在线变点检测——可作为更强的检测器基线，也可作为你 filter zoo 的扩充

**(d) 分段/聚类式**
- 谱聚类分段、phone 边界强制对齐后池化

**你的差异化定位**：(b) 类方法完全可学习但丢掉了信号先验、训练不稳定、需要额外正则；(c) 类方法有信号先验但历史上是"检测完硬丢帧、不与任务联合训练"。**你的方法 = 信号先验检测 + 可学习局部混合（hanning/savgol_hanning 的 weight predictor）+ 端到端任务训练**，正好骑在 (b)(c) 的交界上，这是论文的核心 positioning 句子。

### 2.3 表示级压缩（码率意义）
- 神经音频编解码器：SoundStream / EnCodec 的 RVQ 离散 token
- bottleneck feature、PCA 特征降维

**关系**：压缩的是**比特率/存储**而非**计算量**，目标不同。Related Work 一段话带过，强调"我们压缩序列长度以降低 encoder 计算，与码率压缩正交"。

### 2.4 注意力近似（计算图内压缩）
- Linformer、Performer、稀疏/局部注意力、Emformer 的分块记忆

**关系**：不丢帧、只近似注意力，是你的**互补基线**之一（可以对比"丢帧 vs 近似注意力"在同等 FLOPs 下的质量），也可以叠加（丢帧后再稀疏注意力）。

### 2.5 训练/数据级
- 课程学习、子网络采样（MatFormer 思想的非 LLM 前身）、layer dropout

**关系**：一段话带过。

### 2.6 Related Work 写法建议
三段式：① ASR 序列压缩（CIF、Efficient Conformer、Squeezeformer pooling、MoChA）→ ② 通用 token 压缩（ToMe、DynamicViT、EViT、TokenLearner、ACT）→ ③ 信号处理视角（novelty detection、change-point detection）→ 收尾一句："与前人不同，我们将经典信号检测器嵌入深度 encoder 中层，检测不可微但混合可学习，兼顾信号先验与任务适配；并系统验证该范式跨架构、跨任务、跨模态的泛用性。"

**必引种子清单**（按上面分支各取 3–5 篇）：CIF（Dong & Xu, ICML 2020）；Efficient Conformer；Squeezeformer；MoChA；ToMe（ICLR 2023）；DynamicViT（NeurIPS 2021）；EViT；TokenLearner（NeurIPS 2021）；ACT（Graves 2016）；Bello et al. onset detection tutorial（TASLP 2005）；PELT（Killick et al. 2012）；Gumbel-softmax（Jang et al. 2017）；L0 正则（Louizos et al. 2018）；SoundStream/EnCodec（各一段）；AST（Audio Spectrogram Transformer，跨任务用）；ECAPA-TDNN（说话人验证用）。

---

## 3. 目标会议与投稿策略（分层）

| 层级 | 目标 | 需要的最低实验包 | 对应章节 |
|---|---|---|---|
| Tier 1（现实，先保这个） | **Interspeech / ICASSP / ASRU** | Phase A + B + C1 + D（架构≥3、数据集≥3、任务≥2）+ Phase E | §6 |
| Tier 2（扩展） | **TASLP 期刊** 或 SLT | Tier 1 + C2 + C4 + 泛用性全景（任务≥4）+ 更全消融 | §6 |
| Tier 3（野心） | **NeurIPS / ICLR** | Tier 1 + Phase F 跨模态 + I8 理论 + 可微版本成为主方法 | §5 I2/I7/I8/I9 |

时间锚点（按近年周期，投稿前请核实当年 CFP）：ICASSP 摘要/全文约在每年 9 月；Interspeech 约 2–3 月；ASRU 奇数年约 6–7 月；NeurIPS 约 5 月。**建议第一个完整版本瞄准下一个 Interspeech 或 ICASSP**，用 §10 的 14 周计划倒排。

---

## 4. 泛用性验证框架（本计划的核心支柱）

泛用性不是"多跑几个实验"，而是一个可以被检验、被证伪的科学主张。建议围绕一个**贯穿性假说**组织全部泛化实验：

### 4.0 冗余自适应假说（Redundancy-Adaptive Hypothesis）

> **动态压缩的收益与输入的时间冗余度正相关；质量退化与任务所需的时间精度负相关。**

- 输入冗余度可用免训练指标代理：静音/稳态段比例、novelty signal 的稀疏度（峰数量/总帧数）、谱通量均值、语速（音素/秒）。
- 任务时间精度：音素级 ASR（最高）> 词级 ASR > 关键词检测 > 情感识别 > 说话人验证（最低，全局信息）。
- **可证伪的预测**：① 同一模型内，per-utterance 压缩率应与冗余指标显著正相关；② 跨任务，同等压缩率下的相对退化应沿"时间精度"轴单调上升；③ 会议/自发言语（高冗余）上的收益应大于朗读语音（LibriSpeech）。
- 这个假说一旦被数据支持，论文贡献就从"一个模块"升级为"**一个可预测何时有效的压缩原则**"——这是泛用性主张的最强形态，也是 §5 创新点 I9 的基础。

### 4.1 五个泛化维度与实验设计

| 维度 | 具体项 | 假设 | 优先级 |
|---|---|---|---|
| **G1 跨训练范式/架构** | Conformer-RNN-T（已有）→ Conformer-CTC → Conformer-AED → wav2vec 2.0 / HuBERT fine-tune（模块插 SSL encoder 中层） | 对齐机制影响容忍度：AED 软注意力最鲁棒，CTC/RNN-T 的硬对齐更敏感——无论结果如何都是可写的发现 | P0（前两个）/ P1（SSL） |
| **G2 跨任务（语音内）** | ① 词级 ASR（本线）② 关键词检测 KWS（Speech Commands v2，小模型几小时一训，性价比最高）③ 说话人验证（VoxCeleb1 + ECAPA-TDNN，插帧压缩在统计池化前）④ 语音情感识别（IEMOCAP）⑤ 音频事件标注（AudioSet 子集 + AST/PANNs） | 退化幅度沿"任务时间精度"轴排序（验证假说 4.0②） | P0 选 2 个；P1 选 4 个 |
| **G3 跨数据/域** | LibriSpeech clean+other → TED-LIUM 3（自发言语）→ AISHELL-1（中文）→ 噪声套件（MUSAN 合成 SNR 0/5/10/15）→ （可选）会议长音频 AMI / 低资源 Common Voice | 高冗余域收益更大（验证假说 4.0③）；低资源 + 噪声下动态选择是否更脆弱 | P0（clean/other + 噪声）；P1（其余） |
| **G4 跨输入特征** | mel 100fps（当前）vs mel 200fps vs SSL 特征（wav2vec 50fps 层输出） | 帧率越高冗余越大、可压空间越大；SSL 特征的语义化程度更高，diff 信号更"干净" | P1 |
| **G5 跨模态** | ① 时间序列分类（UEA archive 3–5 个长序列集，改造成本最低）② 视频帧选择（Something-Something V2，替换 uniform 采样）③ 音乐 onset 检测（纯分析：你的检测器在 mel 上直接跑，与经典 novelty detection 比 F1） | 框架与模态无关；音乐 onset 实验顺带验证"我们的框架退化回经典方法仍有竞争力" | P2（Tier 3 必需） |

### 4.2 泛用性实验的统一协议（保证可比性）

每个维度上**固定三元组**：`none（不压缩）/ 最优 filter / top-k-diff 基线`，用 top-k 把三者压缩率钉在同一水平（±2%）。报告两个相对量：

- **相对退化** Δ = (metric_compressed − metric_none) / metric_none
- **相对加速** S = RTF_none / RTF_compressed（或 FLOPs 比）

最终汇总成一张 **generality heatmap**（行 = 架构/任务/数据/模态，列 = Δ 与 S）——论文主图候选。定义明确的泛用性判据（如 Δ ≤ 5% 且 S ≥ 1.3× 记为"泛化成立"），让审稿人看到你是在**检验**主张而不是在**展示**主张。

### 4.3 与架构迁移配套的工程要求

- 模块接口保持 `(B,T,F)+lengths → (B,T',F)+lengths'`，插进任何 encoder 只需一行——G1/G2 的迁移成本主要在数据管线和训练脚本，不在模块本身。
- 每个新任务优先**复用社区开源实现**（SpeechBrain 的 ECAPA-TDNN、HuggingFace 的 AST/wav2vec2），把模块当插件注入，不重写 backbone——既省时间又增强"即插即用"叙事。

---

## 5. 创新点设计（按优先级排序）

每个创新点给出：动机 / 核心假设 / 方法 / 关键实验 / 主要风险。

### I1. 可控压缩率与 Pareto 前沿（P0，必做）
- **动机**：当前压缩率由数据决定，无法控制。所有公平对比和部署故事都依赖"给定预算，质量如何"。
- **假设**：在同等平均压缩率下，动态选择的 Pareto 前沿严格优于均匀下采样。
- **方法**：三选一或都做——① `find_peaks` 加 `prominence`/`distance` 阈值扫参；② 按 diff 值做 top-k（k = 预算）；③ 可微版本（I2）里用 budget loss λ·(#frames) 调节。
- **实验**：压缩率 ∈ {10%, 20%, 30%, 50%, 70%} × 每种选择策略，画 PER/WER–压缩率曲线；同时画 RTF/FLOPs–压缩率曲线。
- **风险**：若动态选择不优于均匀，转向"在何种 regime 下有效"的分析型论文（这正是假说 4.0 的用武之地）。

### I2. 可微分动态帧选择器（P1，旗舰创新，Tier 3 的入场券）
- **动机**：`find_peaks` + `no_grad` 切断梯度，检测器无法被任务信号改进；这也是与 2.2(b) 类学习式方法正面对话的唯一方式。
- **假设**：以信号滤波器输出为初始化/归纳偏置的可学习门控，能在相同压缩率下超过固定 find_peaks 检测器。
- **方法**：小 Conv1d/MLP 门控网络 g(frame_t) → keep 概率；训练用 Gumbel-sigmoid 或 ST-estimator 采样硬选择，推理取阈值；加 budget loss（L0 式或平方惩罚 |实际率−目标率|）。**关键卖点：用 FIR/SG/Hanning 滤波后的 novelty 信号初始化或正则化门控**——"signal prior as initialization, task loss as refinement"。
- **实验**：find_peaks vs learned gate（同压缩率）；初始化消融（随机 vs 信号先验）；训练曲线稳定性对比。
- **风险**：训练不稳 → 课程学习（先固定 find_peaks 训 N epoch 再解冻门控）；退化为全保留 → 加强 budget 惩罚。

### I3. ~~因果/流式版本~~ → 降级为 Future Work
- 不再作为投稿必需。仅在论文中用一张小表如实报告各 filter 的算法前瞻帧数（§1.3-1），并在 Conclusion 写一句"causal variants for streaming deployment are future work"。

### I4. 音素边界对齐分析与边界监督（P1，你独有的分析维度）
- **动机**：你有音素转写——可以回答"被选中的帧到底是不是语言学上有意义的边界"，这是 CIF/ToMe 论文里都没有的分析。
- **假设**：峰谷检测选中的帧与音素边界的重合度显著高于随机/均匀基线；混合权重在边界处更尖锐。
- **实验**：选中帧位置 vs 强制对齐音素边界的 F1（容差 ±2/±3 帧）；不同 filter 的边界 F1 对比；可视化若干 utterance 的 diff 信号 + 选中帧 + 音素边界叠加图（论文主图候选）。
- **进阶**：边界 F1 变辅助 loss → 又一个小创新。
- **风险**：需要音素时间戳（用 Montreal Forced Aligner 跑对齐，半天工作量）。

### I5. 滤波器作为可学习前端（P1，把消融升级成方法）
- **动机**：现在 6 种 filter 是手工枚举；统一参数化为"一维卷积/多项式平滑层"，初始化为 FIR/SG/Hanning 系数再端到端微调——filter zoo 从"我们试了 6 种"升级为"我们提出了可学习检测滤波器族，并发现手工信号处理先验是最优初始化"。
- **实验**：冻结滤波器（=现有方法）vs 微调滤波器 vs 随机初始化滤波器，同压缩率对比。

### I6. 序列压缩 × 模型压缩的复合加速（P2，差异化卖点）
- **动机**：没人系统研究过"动态丢帧"与"INT8 量化/剪枝"是否可乘性叠加。
- **实验**：2×2 设计（基线 / +动态压缩 / +INT8 PTQ / +两者），测 RTF、FLOPs、显存、WER。若叠加无损或微损，论文多一节"deployment-ready compound compression"。

### I7. 跨模态泛化（P1→P2，Tier 3 必需，Tier 1 可选做时间序列一项当彩蛋）
- 见 §4.1 G5。时间序列分类改造成本最低（模块插 Transformer 中层 + UEA 数据集小），即使在语音会议上放一组时间序列结果也能显著强化"通用框架"叙事。

### I8. 理论视角（P2，加分项而非必需）
- 把帧间差分 + 峰谷检测形式化为 encoder 激活流形上的 novelty detection；讨论"变化点处信息密度高"的假设与 rate-distortion 直觉；给出"何种信号下动态选择严格优于均匀"的 toy 分析（如分段平稳信号）。
- 注意：语音会议不强求理论，这节写给 ML 会议版本。

### I9. 压缩收益预测器（P2，由假说 4.0 衍生的实用创新）
- **动机**：若冗余自适应假说成立，则可以**不训练就预测**"这条输入/这个任务能承受多大的压缩率"。
- **方法**：用免训练统计量（novelty 稀疏度、静音比、谱通量）回归每 utterance 的"安全压缩率"（退化 < 阈值的最大率）；或每任务层面预测期望退化。
- **价值**：部署时按输入自适应调预算——"dynamic"的第二层含义（不仅帧选择动态，压缩率本身也动态）。

### 创新点取舍建议
- **Interspeech/ICASSP 版本主线**：I1 + 泛用性框架（§4）+ I4，配充分基线。故事 = "信号驱动动态压缩是一个跨架构、跨任务、跨数据普遍有效的即插即用原则，且其有效性可被输入冗余度预测"。
- **NeurIPS/ICLR 版本主线**：I2 成为主方法（可微动态压缩），G5 跨模态，I8 理论，I5 统一视角，I9 收益预测。
- 两篇论文共享 Phase A/B/D 的实验基础设施，故事不同。

---

## 6. 实验规划（分 Phase）

### Phase A：地基工程（第 1–3 周）——不产出论文图，但决定一切上限

| # | 任务 | 细节 | 验收标准 |
|---|---|---|---|
| A1 | 向量化 `DynamicCompression.forward` | 消除 per-sample Python loop；scipy → torch 实现（local max ⇔ `x == F.max_pool1d(x, k=3)`；滤波器改 `conv1d`） | 单 batch 耗时下降 ≥10×，输出与原实现逐元素一致（写一次性对齐脚本验证） |
| A2 | word-level 评测 | 加标准 LibriSpeech word 转写读取 + WER（jiwer 或 sclite 格式）；保留 PER 双轨 | eval.py 同时输出 PER 和 WER |
| A3 | 种子×3 重训两个锚点 | `compression.filter = none`（无压缩）与当前最优 filter（savgol_hanning），各 3 种子 | 报告 mean±std；确认趋势在种子间稳定 |
| A4 | 等压缩率简单基线（**最高优先级实验**） | 实现 6 个压缩器：均匀下采样、随机丢帧、max-pool、avg-pool、top-k-diff（按差分值取前 k）、能量 top-k；k 匹配你的方法在同 utterance 上的实际保留数 | 一张"基线 × PER/WER × RTF"表。这张表决定论文故事走向 |
| A5 | 压缩率与冗余度统计 | 训练日志与 eval 记录每 utterance 压缩率；计算冗余代理指标（静音比、novelty 稀疏度、语速）；做相关分析 | 假说 4.0① 的第一份证据；论文 §Setup 基础数据 |

### Phase B：核心消融（第 4–6 周）

设计原则：**一次只动一个轴**，其余固定在 A3 确定的最优配置。每个配置 2–3 种子（主表 3 种子，次要扫描可 1–2 种子）。

| # | 消融轴 | 取值 | 科学问题 |
|---|---|---|---|
| B1 | filter zoo | none / naive / fir / savgol / hanning / savgol_hanning | 平滑 novelty 信号到底值不值？（已有代码，只差系统跑） |
| B2 | weight predictor | single_frame / diff5 / five_frames / （新增）无混合硬选择 | 可学习混合 vs 硬选择；什么特征最该喂给预测器 |
| B3 | 检测目标 | 仅峰 / 仅谷 / 峰+谷（当前） | "变化点"与"稳定点"哪个承载信息？仅谷 = 保留静止帧，是强对照 |
| B4 | 差分度量 | L1（当前）/ L2 / cosine / 每维归一化 L1 | 对特征尺度的敏感性 |
| B5 | 窗口/超参 | hanning_window_size ∈ {3,5,7,9}；SG window/polyorder；FIR taps/cutoff；find_peaks distance ∈ {1,2,3} | 敏感性分析（审稿人要求 robustness） |
| B6 | 插入位置 | block1 后（当前）/ 更浅（time reduction 后直接）/ 更深（block2 中拆） | 越早压越省算力但特征越低级——找甜点 |
| B7 | 多级压缩 | 单级（当前）/ 两级（block1 后 + block2 中各一次小压缩） | 渐进压缩 vs 一次压缩（呼应 Efficient Conformer 但动态） |

### Phase C：创新实验（第 6–12 周）

| # | 对应创新点 | 关键产出 | 优先级 |
|---|---|---|---|
| C1 | I1 可控压缩率 | PER/WER–压缩率 Pareto 图（论文主图 1）：动态选择 vs 均匀 vs top-k 三条曲线 | P0 |
| C2 | I2 可微门控 | learned gate vs find_peaks 同率对比表 + 初始化消融（Tier 3 主表） | P1 |
| C4 | I4 边界分析 | 选中帧–音素边界 F1 表 + 2–3 幅定性可视化（论文主图 2）；（进阶）边界 aux loss 版 | P1 |
| C5 | I5 可学习滤波器 | 冻结/微调/随机初始化三档对比 | P2 |
| C6 | I6 复合压缩 | 2×2（动态压缩 × INT8）效率–质量表 | P2 |
| C9 | I9 收益预测器 | 冗余统计量 → 安全压缩率的回归/预测精度；部署模拟 | P2 |

### Phase D：泛用性实验（第 6–14 周，本计划重心）

按 §4.2 统一协议执行。拆分如下（优先级即执行顺序）：

| # | 内容 | 成本估计 | 验证的假说 |
|---|---|---|---|
| D0 | LibriSpeech test-clean **+ test-other**（推理-only，用已有 ckpt） | 零训练 | 基线鲁棒性 |
| D1 | Conformer-CTC（共享 encoder 改 CTC 头，或用 A3 ckpt 初始化） | ~4 次训练 | G1 |
| D2 | 噪声套件：LibriSpeech + MUSAN，SNR ∈ {0,5,10,15}（先推理-only，必要时加噪重训 2–3 次） | 0–3 次训练 | G3，假说 4.0②的鲁棒性面 |
| D3 | KWS：Speech Commands v2 + 小 Conformer/Transformer，插模块 | ~3 次训练（每次仅数小时） | G2，假说 4.0② |
| D4 | 说话人验证：SpeechBrain ECAPA-TDNN 配方插模块，VoxCeleb1 | ~3 次训练 | G2（时间精度最低端） |
| D5 | Conformer-AED 或 wav2vec2 fine-tune（HuggingFace 注入） | ~4 次训练 | G1 补齐第三范式 |
| D6 | TED-LIUM 3 + AISHELL-1 | ~4 次训练 | G3（自发言语 + 跨语言） |
| D7 | （可选）IEMOCAP 情感 / AudioSet 子集事件标注 / AMI 长音频 / 低资源 Common Voice | 各 ~2 次训练 | G2/G3 全景 |
| D8 | Generality heatmap 汇总 + 退化–任务时间精度排序检验 | 分析-only | 假说 4.0②的最终裁决 |

**叙事要点**：D3–D4 是刻意选择的"任务时间精度"两端——KWS 需要局部触发信息，说话人验证只需要全局身份统计。如果同等压缩率下退化幅度呈现 ASR(音素) > KWS > 情感 > 说话人 的排序，假说 4.0② 即获强支持，这本身就是 headline 结果。

### Phase E：效率系统评估（第 12–14 周）

| 指标 | 工具/方法 | 注意 |
|---|---|---|
| FLOPs | fvcore / ptflops，报 encoder 总 FLOPs 与"压缩后节省比例" | 理论 FLOPs 和实测速度两套 |
| RTF | CPU（单线程/4线程）+ GPU，batch=1 和 batch 满两种 | 已部分有，补齐场景 |
| 延迟分解 | 每层计时：block1 / compression / block2 / joiner | 证明压缩模块自身开销 <5%（A1 向量化是前提） |
| 峰值显存 | `torch.cuda.max_memory_allocated` | 序列变短 → 注意力显存平方下降，这是卖点 |
| 吞吐 | utterances/sec 满 batch | |
| 能耗 | CodeCarbon 或 nvidia-smi 积分功耗 | 绿色 AI 叙事，一段话即可 |

### Phase F：跨模态（第 10–16 周，Tier 3 必需 / Tier 1 可选时间序列一项）

见 §4.1 G5。优先级：时间序列分类（改造成本最低）> 音乐 onset 分析（半天，顺带致敬经典文献）> 视频帧选择（算力贵）。

---

## 7. 实验矩阵总表与优先级

| 优先级 | 实验 | 预计成本（A100） | 卡住论文的哪个章节 |
|---|---|---|---|
| P0 | A1 向量化 | —（CPU 工程） | 效率章节合法性 |
| P0 | A2 word-level WER | — | 被接收的前提 |
| P0 | A3 锚点×3种子 | ~6 次训练 | 所有结论的统计基础 |
| P0 | A4 等率基线 | ~6–10 次训练 | 决定故事走向 |
| P0 | A5 冗余度统计 | 推理-only | 假说 4.0① |
| P0 | B1 filter zoo | ~6–12 次训练 | 主消融表 |
| P0 | C1 Pareto | ~8 次训练（阈值扫描部分推理-only） | 主图 1 |
| P0 | D0 + D1 + D2 | ~4–7 次训练 | 泛化章节骨干 |
| P0 | D3 或 D4（二选一先做） | ~3 次训练 | 跨任务第一证据 |
| P1 | B2–B7 消融 | ~15 次训练 | 消融章节 |
| P1 | C4 边界分析 | MFA 对齐 CPU 半天 + 推理-only | 主图 2 |
| P1 | D3+D4 补齐 + D5 + D6 | ~11 次训练 | 泛用性完整矩阵 |
| P1 | C2 可微门控 | ~8 次训练（含调参） | Tier 3 主线 |
| P1 | Phase E | 推理-only | 效率章节 |
| P2 | D7 任务/数据全景 | ~6 次训练 | 附录/期刊版 |
| P2 | C5 / C6 / C9 | ~8 次训练 | 加分节 |
| P2 | Phase F 跨模态 | 视数据集 | Tier 3 |

注：train-100h 单次成本以你们 server 上已有 run 的墙钟时间为准换算；大量"阈值扫描/推理-only"实验（C1 部分、D0、D2 前半、A5）**不需要重新训练**，优先榨干已有 checkpoint。G2 任务的模型都很小（KWS/ECAPA/IEMOCAP 单次训练以小时计），泛用性矩阵的边际成本比看起来低得多。

---

## 8. 风险登记与对策

| # | 风险 | 概率 | 对策 |
|---|---|---|---|
| R1 | **等压缩率下简单基线（尤其 top-k-diff / 均匀）打平你的方法** | 中 | ① 先跑 A4，早知道早调整；② 若打平，转向假说 4.0 的分析型故事："方法在何 regime 胜出"本身就是贡献；③ 强化 hanning 混合的独有优势（硬选择基线没有信息混合） |
| R2 | 可微门控训练不稳/退化为全保留或全丢弃 | 中 | 课程学习；budget loss 二次惩罚 + warm-up；信号先验初始化；硬性 min/max 率约束 |
| R3 | 音素标注被质疑非主流 | 高（必被问） | 双轨评测（A2），音素结果放附录作为"细粒度分析" |
| R4 | 算力不够跑完矩阵 | 中 | ① 推理-only 实验先行；② 次要扫描减种子；③ 砍 D7；④ CTC/AED 复用 RNN-T encoder 权重初始化 |
| R5 | **跨任务泛化失败**（如 KWS 上退化严重） | 低-中 | 这恰恰检验假说 4.0：若失败模式符合"时间精度排序"，写成发现；若完全无规律，缩小主张为"ASR 及其变体上有效"，论文仍成立 |
| R6 | 压缩率方差大，短句几乎不压缩 | 中 | I1 的 top-k 预算模式天然解决；论文里报"率分布"而非单点 |
| R7 | CTC/AED 上方法失效（RNN-T 特有） | 低-中 | 若失效，分析"为何 RNN-T 的 blank 机制容忍丢帧而 CTC 不容忍"——这也是可发表的发现（G1 的设计初衷） |
| R8 | 泛用性矩阵太大，结论被稀释 | 低 | 严格执行 §4.2 统一协议与判据；heatmap 一图收束，细节进附录 |

---

## 9. 论文故事线与章节大纲（两版）

### 版本 A：语音顶会版（Interspeech/ICASSP，4–8 页）
- **标题方向**：*Signal-Driven Dynamic Frame Compression: A Plug-and-Play Principle for Efficient Sequence Encoders*
- **Hook**：固定下采样对所有输入一视同仁，但语音（乃至一切序列信号）的信息密度天然不均匀。我们把经典 novelty detection 搬进 encoder 中层，让模型"在变化处多看，在平稳处少看"，并证明这一原则**跨训练范式、跨任务、跨语言普遍成立，且其有效性可由输入冗余度预测**。
- **大纲**：1 Intro（信息密度不均 + 三个贡献：信号驱动检测+可学习混合的统一框架 / 五维泛用性验证与冗余自适应假说 / 边界对齐的可解释性证据）→ 2 Related（§2.6 三段式）→ 3 Method（novelty signal 定义、filter zoo 统一记号、weight predictor、各 filter 前瞻帧数表、复杂度分析）→ 4 Setup → 5 Results（主表、Pareto 主图、generality heatmap 主图、消融、边界分析、效率表）→ 6 Analysis（假说 4.0 的检验：压缩率–冗余度相关、退化–时间精度排序）→ 7 Conclusion（含 causal variants 的 future work 一句）。

### 版本 B：ML 顶会版（NeurIPS/ICLR）
- **标题方向**：*Differentiable Signal-Prior Sequence Compression* 或 *Learning Where to Look: Signal-Prior-Initialized Dynamic Token Compression*
- **Hook**：token 压缩方法要么纯学习（丢先验、训练难）要么纯手工（不可微、不适配任务）。我们用信号处理先验初始化一个可微选择器，在语音/时间序列/视频三个模态验证，并给出何时优于均匀压缩的理论刻画。
- 主线实验 = I2 + G5 + I8 + I9，语音结果作为三模态之一。

---

## 10. 14 周时间线（可按会议 deadline 平移）

| 周 | 主线任务 | 里程碑 |
|---|---|---|
| 1–2 | A1 向量化 + A2 WER + A5 统计脚本 | 工程就绪，输出与原实现对齐验证 |
| 2–3 | A3 锚点×3 种子；A4 基线实现与开跑 | **Go/No-Go 决策点：等率基线结果** |
| 4–6 | B1–B7 消融（按 P0→P1）；C1 Pareto（推理-only 部分先行） | 主消融表 + Pareto 主图 |
| 6–8 | D0/D1（CTC）/D2（噪声）；D3（KWS）开跑 | 第二架构 + 第二任务的首批数据 |
| 8–10 | D4（说话人）/D5/D6；C4 MFA 对齐 + 边界分析；论文骨架开写 | 泛用性矩阵过半 + 主图 2 |
| 10–12 | D8 heatmap 与假说检验；Phase E 效率全套；显著性检验 | 完整结果表 + generality 主图 |
| 12–13 | 写稿 + 内审 + 图美化 | 完整初稿 |
| 14 | 按 checklist 终检（种子数/显著性/可复现性/附录） | 投稿 |

Tier 3 路线：在上述基础上 +4–6 周（C2 调参 + Phase F），目标下一个 NeurIPS/ICLR 周期。

---

## 11. 工程改造清单（代码级，直接对应本仓库）

1. `conformer_rnnt/compression.py`
   - [ ] 向量化峰检测：`F.max_pool1d` 等价实现 local-max/min，去掉 `for b in range(B)` 和 `.cpu().numpy()`
   - [ ] 新增 `rate_control: {none, prominence, topk}` 与 `target_rate / prominence` 字段
   - [ ] 新增 `learned_gate` 模式（I2）：Conv1d+MLP gate，Gumbel-sigmoid，budget loss 接口（loss 从模块传出，在 `lightning_module.py` 的 training_step 里加权）
   - [ ] 前向返回附加 dict：compression_rate、selected_indices（供 A5/C4 分析，eval 时落盘）
   - [ ] 模块接口文档化："任何 `(B,T,F)+lengths` 的 encoder 均可插入"——为 G1/G2 迁移做准备
   - [ ] （低优先，future work）`causal: bool` 配置
2. `configs/*.yaml`：新增上述字段；写个 10 行脚本批量产出实验矩阵各格子的 YAML
3. `eval.py`：加 WER（word 转写读取）、每 utterance 压缩率输出、（C4）选中帧落盘
4. `train.py` / `lightning_module.py`：`seed_everything`；TensorBoard 记录平均压缩率；budget loss 权重超参
5. 新文件 `baselines/fixed_compressors.py`：A4 的 6 个等率基线，与 `DynamicCompression` 同接口，配置切换
6. 新任务接入（G2）：fork SpeechBrain 的 KWS/ECAPA 配方 + HuggingFace AST，模块作为插件注入；各写 ≤50 行 glue code
7. 新脚本 `analysis/boundary_f1.py`（MFA TextGrid + 选中帧 → F1）；`analysis/redundancy_corr.py`（A5 相关分析）；`analysis/plot_heatmap.py`（generality 主图）
8. 可复现性：每个最终配置存档 YAML + git commit hash 进附录

---

## 12. 完成度自查清单（投稿前逐项打勾）

- [ ] word-level WER：test-clean / test-other 都有，且 ≥3 种子 mean±std
- [ ] 主对比均有显著性检验（bootstrap 或 paired t-test）
- [ ] 等压缩率基线表（A4）——审稿人第一问
- [ ] Pareto 图含至少 3 条方法曲线
- [ ] **泛用性**：≥3 架构（G1）× ≥2 额外语音任务（G2）× ≥3 数据集/域（G3），统一协议（§4.2）下的 heatmap 主图
- [ ] 冗余自适应假说的两项统计检验（压缩率–冗余相关；退化–时间精度排序）
- [ ] 噪声鲁棒性（≥3 个 SNR 档）
- [ ] 各 filter 算法前瞻帧数表（如实声明，streaming 留给 future work）
- [ ] 压缩模块自身开销 <5% 的实测证据
- [ ] 边界 F1 分析 + ≥2 幅定性可视化
- [ ] 效率四件套：FLOPs / RTF(CPU+GPU) / 显存 / 吞吐
- [ ] 失败案例分析 1 节（顶会论文的诚实度加分项；泛化失败的任务如实报告并解释）
- [ ] 代码与配置开源声明（或匿名仓库链接）

---

*本计划与代码现状（commit b0bf927）对齐；若方法实现演进，优先回看 §1.3 的技术自查与 §7 优先级表是否仍成立。*
