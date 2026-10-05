# 论文研究计划 v2.0：基于帧间差分峰谷检测的层间动态序列压缩

> 版本：v2.0（2026-09-26）
> 基线：Conformer-RNN-T + DynamicCompression（`conformer_rnnt/compression.py`），LibriSpeech 音素，指标 PER/RTF
> 目标：语音顶会（Interspeech / ICASSP / ASRU）接收标准，保留 ML 顶会（NeurIPS / ICLR）升级路径
> v2.0 变更：基于 10 篇 2025+ 顶会/前沿论文精读（见 §1 与 papers/ 目录）重构——**新增 scoop 风险分析与差异化定位**；**A4 基线池从 6 个扩到 10 个**（补入 2025+ 已发表方法作为必须对比的强基线）；**新增 Phase A0 诊断性 pilot study**；**消融改为"检测×混合"正交两组件**；**效率评测升级为 FastAdaSP 式 FLOPs 档位对齐**；**机制分析引入统计检验模板**。

---

## 0. 方法的一句话定位（v2 修订）

> 给定 encoder 中间层激活序列，用帧间差分构造 novelty signal，通过可配置的信号处理滤波器族检测变化点/稳定点，在检测点处用可学习的局部加权混合替代硬选择，从而在不改变模型参数规模的前提下，按内容自适应地缩短序列长度，降低后续所有层的计算量。

v2 补充定位坐标（来自 2025+ 文献，详见 §1）：

- 在 token 压缩 taxonomy（TMLR 2026 综述）中，本方法 = **similarity-based redundancy quantification 的时序特例**（相邻帧不相似 = novelty），与 attention-based（VisionZip/VScan）、diversity-based（DivPrune）、query-based（FastSLM/MMS-LLaMA）并列。论文 Related Work 按此四象限组织，我们占据"时序几何信号 + 无需前向注意力 + 天然适配流式"的格子。
- **结构同构论点**：2025+ 主流方法（VisionZip、VScan、FastAdaSP、DTM-Codec）收敛于"**选择 + 回收（select + recycle）**"两段式。本方法的"峰谷检测 + 类 Hanning 学习混合"正是该范式在信号先验下的实例——这不是缺陷而是定位：**把"检测器"与"混合器"解耦为两个正交组件**，分别消融（§5 I10），论文结构立刻与顶会话术对齐。

---

## 1. 2025+ 文献精读与重合度（scoop 风险）分析

> 全部 PDF 已下载至 `papers/`。按重合度排序。**粗体 = 必须引用且必须实验对比**。

### 1.1 高重合 —— 必须正面差异化

**[P1] DTM-Codec（KAIST，Interspeech 2026，papers/dtm-codec-vfr-boundary-interspeech26-2606.29480.pdf）**
可变帧率语音编解码器：**在两级 Transformer encoder 之间**用 PLE（累积相邻帧余弦距离，过阈值放边界）做动态帧选择，保留帧原样打包、遮蔽位填可学习 `<MASK>`；Robbins-Monro 控制器锁定 keep ratio；不可微检测 + 端到端训练；LibriSpeech 训练评测。
- **重合点**：层间位置、内容自适应帧选择、依据帧间特征变化、不可微+端到端、压缩率可控、RTF 指标、LibriSpeech——核心实验框架几乎重合。
- **不同点（我们的护城河）**：① 任务是 **ASR/PER**（理解任务）而非 codec 重建——重建任务天然偏好"保留原帧"，理解任务才能体现"软混合注入邻域信息"的价值；② 我们是**峰/谷双侧 novelty 检测**（PLE 是单侧累积阈值），且有滤波器族去噪；③ **可学习加权混合** vs 硬保留+MASK 填充；④ 无需边信息传输。
- **必须做的事**：引用并在 Related Work 明确区分；A4 基线池加入 **PLE 选择器**（复现其核心选择逻辑，≤30 行），同压缩率对比；借鉴其"等码率同架构 FFR 基线"与"选择器复杂度/耗时占比表"。

### 1.2 中重合 —— 必须引用，建议实验对比

**[P2] Audio Token Compression in LALMs（MIT/IBM，arXiv 2511.20973，2025-11）**
Whisper encoder 输出后、LLM 前的音频 token 压缩系统对比：**无监督分割（相邻帧余弦不相似度峰检测定边界 + 段内平均）** vs 均匀池化 vs 均匀采样，LoRA 微调对齐；结论 merging > sampling。
- 重合点：帧间不相似度 + 峰检测，与我们同源——**这是距离我们最近的"检测器"**，A4 必须加入此基线（cosine-diff 峰检测 + 段平均）。不同点：位置在 encoder 整体之后、段内硬平均（无学习权重）、LALM+LoRA 范式。
- 可借鉴：等压缩率下"平均 vs 丢弃"对照直接支撑我们"软混合优于硬丢弃"；训练/不训练消融量化端到端贡献。

**[P3] Context-Driven Dynamic Pruning（CMU/Amazon，Interspeech 2025）**
OWSM E-Branchformer **逐层逐帧**动态剪枝：外部上下文（说话人/事件/语言嵌入）门控，Straight-through Gumbel-Softmax 可微训练，30% 目标稀疏度；zero-pad 回原长（省模块计算不省注意力长度）。
- 重合点：encoder 内部逐帧动态决策、VAD 式剪枝模式。不同点：可微学习门控 vs 信号检测；序列长度不变；需外部上下文嵌入。
- 可借鉴：**机制分析写法**——剪枝模式可视化 + Mann-Whitney U / Welch's t 检验（p<0.01）；同时报 FLOPs 与墙钟时间的诚实做法。

**[P4] FastAdaSP（CMU/NVIDIA，arXiv 2410.03007）**
SpeechLM decoder **层间**免训练 token 合并：相邻 key 余弦相似度 top-k 成簇 + 注意力分数加权合并；Transfer Entropy 选层。
- 重合点：层间位置、局部信号驱动、**加权软合并**替代硬丢弃。不同点：LLM decoder、免训练推理期、相似度（去冗余）vs 我们的 novelty（变化检测）。
- 可借鉴：**等 FLOPs 档位（10/20/30/40/50%）对齐全部基线**（含 Random Merge/Evict 平凡基线）的表结构——Phase C1 直接照抄；系统指标（RTF/延迟/显存）与任务指标分表报告。

**[P5] MMS-LLaMA（KAIST，arXiv 2503.11315）**
AVSR 中按时长比例 + 语速预测动态分配 Q-Former query 数（3.5 tok/s）。
- 重合点：压缩量随输入内容动态分配。不同点：query 蒸馏机制、encoder 外部。
- 可借鉴：逐组件顺序集成表（每行同时列 tok/s、显存、FLOPs、WER）；baseline 复现时统一 backbone 排除混淆变量。

### 1.3 低重合 —— 泛化叙事与实验模板来源

**[P6] VisionZip（CVPR 2025）**：视觉 encoder 出口 [CLS] attention 选 dominant token + 其余按相似度合并回收；training-free。借鉴：pilot study 先观察冗余再提方法；**性能一律报"相对 upper bound 保持率 %"**，跨任务可比。
**[P7] DivPrune（CVPR 2025）**：token 剪枝形式化为 Max-Min 多样性问题（MMDP），无需任何重要性分数，training-free。借鉴：baseline 按"是否需训练/校准"分层组织；**极端压缩率（≤25% FLOPs）下的 graceful degradation 曲线**单列分析。与我们同为"只看表示几何、不学重要性分数"一族——A4 可加分段约束版 MMDP 选帧基线。
**[P8] VScan（TMLR 2026-01）**：先诊断"视觉注意力集中在中层、早层有位置偏置"再决定在哪剪；global+local 两段扫描。借鉴：**"诊断 study → 设计 → 消融验证"三段闭环**；（R₁, R₂）两段保留率网格消融 → 对应我们的"插入位置 × 压缩强度"二维消融。
**[P9] FastSLM（arXiv 2601.06199）**：层级 query 蒸馏把 50Hz 语音压到 1.67 tok/s 做长语音 LLM。借鉴：**Marginal Semantic Utility η = ΔWER/ΔFLOPs** 选压缩率工作点（信息瓶颈饱和拐点）。
**[P10] Token Compression 综述（TMLR 2026-01，42 页）**：模态×机制双轴 taxonomy；压缩三要素 = 重要性识别、冗余量化、合并/剪枝。借鉴：Related Work 组织框架；拆解三要素逐一对比的话术。

### 1.4 Scoop 风险总结

| 论文 | 重合度 | scoop 风险 | 应对 |
|---|---|---|---|
| P1 DTM-Codec | **高** | **实质**（层间动态选择+等率 FFR 框架被占先） | 差异化押注：理解任务、峰谷双侧检测、学习软混合、无边信息；A4 加 PLE 基线正面对比 |
| P2 Audio Token Compression | 中 | 中（检测器同源） | 必须引为最近邻基线；强调层间 vs encoder 后、学习混合 vs 段平均 |
| P3 Context-Driven Pruning | 中低 | 低 | 引用区分（可微门控 vs 信号检测） |
| P4 FastAdaSP | 中 | 低 | 引用区分（推理免训练 vs 训练期） |
| 其余 | 低 | 无 | 作为泛化叙事与实验模板 |

**结论：研究未被 scoop，但窗口在收窄**（2025–2026 语音 token 压缩爆发）。差异化三支柱 = ①信号先验检测器族（可解释、零学习成本）②可学习软混合（回收信息而非丢弃）③理解类任务上的端到端联合训练。建议按 §10 时间线尽快投出。

---

## 2. 与顶会接收标准的差距分析（v2 更新）

v1.1 §1.2 的差距表仍然成立（架构/任务/数据集/WER/种子），v2 新增/修订：

| 维度 | 现状 | 顶会要求 | 依据 |
|---|---|---|---|
| 2025+ 强基线 | 无 | 必须对比 P1（PLE 选择器）、P2（cosine-diff 分割）、A-ToMe 式相似度合并 | P1/P2 审稿人必引 |
| 诊断性动机实验 | 无 | pilot study：冗余分布统计/可视化先行，方法由观察驱动 | P6/P8 标准写法 |
| 性能报告口径 | 绝对 PER | 补"相对不压缩上界的保持率 %" | P6 |
| 压缩率工作点 | 无依据 | 用 η=Δmetric/ΔFLOPs 拐点论证 | P9 |
| 检测器开销 | 未测 | 报选择器自身 RTF/时间占比（须 <5%） | P1 |
| 机制分析统计 | 无 | 选中帧模式的组间差异做 U 检验/t 检验 | P3 |
| 极端压缩率行为 | 无 | ≤25% FLOPs 档的退化曲线 | P7 |
| 插入位置依据 | 拍脑袋（block1 后） | 诊断性层位分析（各层 novelty 信号性质） | P8 |

---

## 3. 压缩方法全景与 Related Work 组织（v2 微调）

沿用 v1.1 §2 的五分支（模型级/序列级/表示级/注意力近似/训练数据级），序列级内部改按 P10 综述的**机制四象限**组织，把 2025+ 工作嵌入：

- **Transformation-based**：池化/卷积下采样、Efficient Conformer、Squeezeformer、Zipformer
- **Similarity-based**：ToMe → A-ToMe（音频）、FastAdaSP、**P2 无监督分割**、**我们的帧间差分 novelty（时序特例）**
- **Attention-based**：DynamicViT、EViT、FastV、VisionZip、VScan、SpeechPrune
- **Query-based**：TokenLearner、Q-Former 系（MMS-LLaMA、FastSLM）
- 另立一小节讲**语音专用动态压缩**：CIF、MoChA、**P1 DTM-Codec（VFR codec）**、**P3 上下文剪枝**、Blank-regularized frame skipping

收尾 positioning 句（v2）：
> 与前人不同，我们将经典信号处理 novelty 检测器族嵌入 encoder 层间，检测不可微但混合可学习，零门控参数、天然因果可选；并在统一"检测×混合"解耦框架下系统对比 2025+ 各类选择依据（attention/diversity/query/signal），验证该范式跨架构、跨任务、跨数据的泛用性及其可被输入冗余度预测的边界。

---

## 4. 泛用性验证框架（保留 v1.1 §4，不改）

冗余自适应假说（Redundancy-Adaptive Hypothesis）与五维泛化（G1 架构 / G2 任务 / G3 数据 / G4 特征 / G5 模态）、统一三元组协议（none / 最优 filter / top-k 基线，压缩率钉 ±2%）、generality heatmap 主图——全部保留。v2 仅两点加强：

1. 三元组扩为**四元组**：`none / 最优信号filter / top-k-diff / P2 式 cosine-diff 分割`——第四个是 2025+ 文献里的最近邻方法，缺了会被审稿人直接拒。
2. 每个泛化格子除 Δ 与 S 外，补报**保持率 %**（compressed/none），跨任务汇总时以保持率为主口径（P6 做法）。

---

## 5. 创新点设计（v2 重排）

### I0（新增，P0）. 诊断性 pilot study：encoder 中层冗余的可视化证据
- **动机**：P6/P8 的标准开场——先证明"中间层激活确实存在大量时间冗余且分布不均"，方法才有正当性。
- **方法**：在 A3 训好的 none-ckpt 上做推理-only 分析：逐层帧间差分分布、novelty 稀疏度（峰占比）、相邻帧余弦相似度直方图、按音素/静音分段的冗余热力图、**层位诊断**（block 1..N 各层 novelty 信号与音素边界的重合度——直接决定 B6 插入位置）。
- **产出**：论文 §Motivation 的 2 幅图 + B6 的层位选择依据。

### I1. 可控压缩率与 Pareto 前沿（P0，保留）
- v2 修订：压缩率档位**改用 FLOPs 降低档 {10%, 20%, 30%, 40%, 50%}**（P4 口径），而非帧数比例——FLOPs 档天然把"帧数 ↓"换算成"算力 ↓"，与效率故事直接挂钩；同时报帧数率作对照。
- 加 **η = ΔPER/ΔFLOPs 拐点**标注最优工作点（P9）。
- 加 **≤25% FLOPs 极端档**的 graceful degradation 分析（P7）。

### I2. 可微分动态帧选择器（P1，保留，Tier 3 旗舰）
- 不变：信号先验初始化 + Gumbel-sigmoid + budget loss。
- v2 补充：与 P3 的 Gumbel 剪枝正面对话——我们的差异是"信号先验初始化"这一归纳偏置，实验里必须有"随机初始化 vs 信号初始化"的对照行。

### I3. 因果/流式 → Future Work（保留 v1.1 降级决定）

### I4. 音素边界对齐分析（P1，保留）
- v2 加强：边界 F1 之外，加 P3 式**统计检验**——选中帧距最近音素边界的距离分布 vs 随机/均匀基线的分布，Mann-Whitney U 检验报 p 值。可视化沿用"diff 信号 + 选中帧 + 音素边界叠加图"。

### I5. 滤波器作为可学习前端（P2，保留）

### I6. 序列压缩 × 模型压缩复合加速（P2，保留）

### I7. 跨模态泛化（P1→P2，保留）
- v2 补充：综述（P10）确认 ToMe→A-ToMe 的跨模态迁移已有先例——我们的 G5 时间序列/视频实验不是噱头而是跟随已验证的迁移路径；视频帧选择上"帧差峰 = 镜头/运动事件"是天然叙事。

### I8. 理论视角（P2，保留）

### I9. 压缩收益预测器（P2，保留）

### I10（新增，P1）. "检测 × 混合"正交解耦框架
- **动机**：P10 综述把压缩拆成三要素（识别/量化/合并）；P1/P2/P4 都是"检测器固定+混合固定"的单点方法。把两者解耦，我们的 6 种 filter × 4 种混合（无混合/top-k 硬选/段平均/学习权重）变成一张 **2D 设计空间矩阵**——消融即论文贡献本身："何种检测器配何种混合器最优"。
- **实验**：检测器 {find_peaks raw, FIR, SG, PLE（P1 复现）, cosine-diff（P2 复现）} × 混合器 {硬选择, 段平均, 学习 Hanning 权重}，等 FLOPs 档对比。
- **卖点**：这张矩阵表同时回答"为什么是我们的组合"，并把所有 2025+ 近邻方法收纳为矩阵中的格子。

### 创新点取舍（v2）
- **Interspeech/ICASSP 主线**：I0（诊断动机）+ I10（解耦框架）+ I1（Pareto）+ 泛用性（§4）+ I4（边界分析）。故事 = "信号驱动动态压缩是一个可解耦、可预测、可泛化的即插即用原则"。
- **NeurIPS/ICLR 主线**：I2 可微门控 + I10 设计空间 + G5 跨模态 + I8/I9。

---

## 6. 实验规划（Phase 重排）

### Phase A0：诊断性 pilot study（第 1 周，推理-only，零训练）—— 新增
| # | 任务 | 验收 |
|---|---|---|
| A0-1 | 逐层冗余统计：各 block 输出的帧间差分/novelty 稀疏度/相邻相似度分布 | 冗余随层加深的变化曲线图 |
| A0-2 | 层位诊断：各层 novelty 峰与音素边界的 F1（容差 ±2 帧） | 决定 B6 插入位置的数据表 |
| A0-3 | 压缩率现状分布：现有 find_peaks 在各 filter 下的保留率直方图 | §Setup 基础数据（v1.1 指出的盲区） |

### Phase A：地基工程（第 1–3 周，保留 v1.1）
A1 向量化（max_pool1d 等价 local-max）；A2 word-level WER；A3 锚点×3 种子；A5 冗余度统计。**A4 基线池 v2 扩充为 10 个**：

| 类别 | 基线 | 来源 |
|---|---|---|
| 平凡 | 均匀下采样、随机丢帧 | 必答审稿人 |
| 信号 | top-k-diff、能量 top-k | v1.1 已有 |
| 池化 | max-pool、avg-pool | v1.1 已有 |
| **2025+ 检测器** | **PLE 累积阈值边界（P1 复现）**、**cosine-diff 峰检测 + 段平均（P2 复现）** | 必须对比 |
| 几何 | 分段约束 Max-Min diversity 选帧（P7 思想） | 补强 |

全部基线与我们的方法在**相同 per-utterance 保留帧数**下对比；另按 FLOPs 档位对齐画汇总表（P4 结构）。

### Phase B：核心消融（第 4–6 周）—— v2 重构为两组件正交
- **B-矩阵（I10）**：检测器 {raw, FIR, SG, PLE, cosine-diff} × 混合器 {硬选, 段平均, 学习权重}，等 FLOPs 20% 档，2 种子 —— 论文主消融表。
- B3 检测目标（峰/谷/峰+谷）、B4 差分度量（L1/L2/cosine）、B5 超参敏感性：保留，降为单轴扫描。
- **B6 插入位置**：由 A0-2 诊断结果选 3 个候选层，不再盲扫（P8 做法）。
- **B7 两段压缩**：改（R₁, R₂）保留率网格 {单级, (10%,10%), (20%,10%), (10%,20%)}（P8 网格消融）。

### Phase C：创新实验（第 6–12 周）
| # | 内容 | v2 变化 |
|---|---|---|
| C1 | Pareto 主图 | FLOPs 档位口径 + η 拐点 + 极端档退化曲线；≥4 条曲线（我们/均匀/top-k/PLE 或 cosine-diff） |
| C2 | 可微门控 | 加"信号初始化 vs 随机初始化"对照行 |
| C4 | 边界分析 | 加 U 检验/t 检验统计列 |
| C5/C6/C9 | 保留 v1.1 | — |

### Phase D：泛用性实验（第 6–14 周，保留 v1.1 优先级）
D0–D8 不变，协议改四元组（§4），主口径改保持率 %。

### Phase E：效率系统评估（第 12–14 周）—— v2 升级
- **双表结构**（P4）：任务指标表（PER/WER/保持率）与系统指标表（RTF/延迟/显存/FLOPs/吞吐）分开。
- **检测器开销表**（P1）：各检测器自身 RTF 与时间占比，证明 find_peaks（向量化后）<5%。
- FLOPs 理论值 + 墙钟实测**并列**（P3 诚实做法），延迟分解到 block1/compression/block2/joiner。
- 显存：强调注意力显存随序列长平方下降。

### Phase F：跨模态（第 10–16 周，保留）
时间序列分类优先；视频帧选择叙事改用"帧差峰=运动事件"（§5 I7）。

---

## 7. 实验矩阵与优先级（v2 重排前 8 行）

| 优先级 | 实验 | 成本 | 卡点 |
|---|---|---|---|
| **P0** | **A0 诊断 pilot** | 推理-only 1 周 | 动机章节 + B6 层位 |
| P0 | A1 向量化 | CPU 工程 | 效率合法性 + 检测器开销 <5% |
| P0 | A2 WER 双轨 | — | 接收前提 |
| P0 | A3 锚点×3 种子 | ~6 次训练 | 统计基础 |
| **P0** | **A4 基线池（10 个，含 PLE + cosine-diff）** | ~10–14 次训练 | **决定故事走向 + scoop 防御** |
| P0 | B-矩阵（检测×混合） | ~15 次训练（部分可 1–2 种子） | 主消融表 |
| P0 | C1 Pareto（FLOPs 档） | 推理-only 为主 | 主图 1 |
| P0 | D0+D1+D2 | ~4–7 次训练 | 泛化骨干 |

（P1/P2 行同 v1.1，此处不重复；D3/D4 跨任务、C4 边界、Phase E、C2、D5/D6、Phase F 依序。）

---

## 8. 风险登记（v2 更新）

| # | 风险 | 概率 | 对策 |
|---|---|---|---|
| **R0（新）** | **被 P1 类工作进一步挤压**（层间动态选择论文增多） | 中 | ① 尽快投出（§10 倒排）；② 差异化三支柱写进 Intro 首段；③ A4 正面对比 PLE——若我们胜出是强结果，若打平转"检测×混合解耦 + 泛用性"故事 |
| R1 | 等压缩率简单基线打平 | 中 | 同 v1.1：早跑 A4；转向假说 4.0 分析型故事；强调软混合独有优势（硬选择基线无信息回收，P2 已证 merging>sampling） |
| R2–R8 | 同 v1.1 | — | 同 v1.1 |

---

## 9. 论文故事线（v2 更新）

### 版本 A：语音顶会版
- **标题方向**：*Signal-Driven Dynamic Frame Compression: A Plug-and-Play Principle for Efficient Sequence Encoders*（不变）
- **Hook（v2）**：2025+ 的 token 压缩研究在视觉与 LLM 侧爆发（VisionZip/DivPrune/VScan/FastAdaSP），语音侧刚起步（DTM-Codec/LALM 压缩）；这些工作收敛于"选择+回收"范式但选择依据各异。我们把经典 novelty detection 引入 encoder 层间，提出**检测×混合解耦框架**，系统对比五类检测器，并证明信号先验检测 + 学习混合在理解任务上跨架构/任务/数据普遍有效、且有效性可被输入冗余度预测。
- **大纲**：1 Intro → 2 Related（机制四象限 + 语音专项）→ 3 Motivation（A0 诊断图）→ 4 Method（解耦框架：novelty signal、filter zoo、学习混合器、前瞻帧数表、复杂度）→ 5 Setup → 6 Results（主表、Pareto、B-矩阵消融、generality heatmap、边界分析+统计检验、效率双表+检测器开销）→ 7 Analysis（假说检验）→ 8 Conclusion（causal variants 一句）。

### 版本 B：ML 顶会版（不变，主线 I2+I10+G5+I8/I9）

---

## 10. 时间线（v2，14 周不变，前置任务微调）

| 周 | 主线 | 里程碑 |
|---|---|---|
| 1 | **A0 诊断 pilot** + A1 向量化开工 | 动机图 + 层位决策 |
| 1–3 | A1/A2/A5 + A3 种子 + **A4 十基线（含 PLE/cosine-diff 复现）** | **Go/No-Go：等率对比（含 2025+ 方法）** |
| 4–6 | B-矩阵 + 单轴消融；C1 推理-only 部分 | 主消融表 + Pareto 主图 |
| 6–10 | D0–D4；C4 MFA+统计检验；论文骨架 | 泛化过半 + 主图 2 |
| 10–12 | D5–D8 heatmap；Phase E 双表 + 检测器开销表；显著性 | 完整结果 |
| 12–14 | 写稿 + 内审 + checklist 终检 | 投稿 |

---

## 11. 工程改造清单（v2 增量）

1. `compression.py`：保留 v1.1 全部条目；**新增 `detector: {find_peaks, ple, cosine_diff}` 与 `mixer: {hard, segment_avg, learned}` 两个正交配置轴**（替代单一 `compression_filter` 枚举，旧配置映射保留）；PLE 复现 = 累积余弦距离过阈值放边界（≤30 行）。
2. `baselines/fixed_compressors.py`：A4 十基线同接口实现。
3. `analysis/`：新增 `layer_redundancy.py`（A0-1）、`layer_boundary_f1.py`（A0-2）、`rate_distribution.py`（A0-3）；`boundary_f1.py` 加 scipy.stats 的 U 检验/t 检验输出。
4. eval：保持率 %、FLOPs 档对齐汇总表生成脚本。

---

## 12. 投稿前 checklist（v2 增补，新增项加粗）

- [ ] word-level WER test-clean/test-other，≥3 种子 mean±std
- [ ] 主对比显著性检验（bootstrap 或 paired t-test）
- [ ] **等压缩率基线表含 PLE（P1）与 cosine-diff 分割（P2）两个 2025+ 方法**
- [ ] Pareto 图 ≥4 条曲线，FLOPs 档口径，标注 η 拐点
- [ ] **检测×混合 2D 消融矩阵**
- [ ] 泛用性 ≥3 架构 × ≥2 额外任务 × ≥3 数据集，四元组协议 heatmap
- [ ] 冗余自适应假说两项统计检验
- [ ] 噪声鲁棒性 ≥3 SNR 档
- [ ] 各 filter 算法前瞻帧数表
- [ ] **检测器自身开销 <5% 实测表**
- [ ] **保持率 % 口径贯穿全文**
- [ ] **A0 诊断图（冗余分布 + 层位诊断）在 Motivation 节**
- [ ] **Related Work 覆盖全部 10 篇 papers/ 文献并明确差异化**

---

## 13. 顶会冲刺执行计划（v2.1，2026-10-05 增补）

> 原则：不绑定具体截稿日期，按"语音顶会（Interspeech/ICASSP 级）接收要求"组织为依赖排序的工作包；每个工作包有明确产出、工作量与 Go/No-Go 门。目标期刊/会议与投稿时机由进度质量决定，进度本身是唯一节拍器。

### 13.0 当前资产盘点（冲刺起点）

| 已有 | 状态 | 对应 checklist |
|---|---|---|
| B-matrix 2D 消融矩阵（5×3，单种子，PER） | ✅ 完整 | "检测×混合 2D 消融矩阵" |
| A0 诊断三件套（冗余分布/层位/oracle） | ✅ | "A0 诊断图进 Motivation" |
| 冠军 r=0.27 cosine_diff+learned（50ep） | ✅ | 主结果候选 |
| 不压缩基线 + 10 固定策略基线 | ✅ | 内部基线表 |
| beam 评测（3 头部模型） | ✅ | 解码口径补齐 |

| 缺口 | 严重度 |
|---|---|
| **word 级 WER 口径**（LibriSpeech 顶会通行 WER；当前目标是音素序列 + PER，reviewer 必问） | **P0，投稿硬门槛** |
| 多种子 + 显著性检验 | P0 |
| FLOPs 实测与延迟分解（RTF 陷阱已实证） | P0 |
| 外部已发表基线对比（P1/P2 至少一项） | P0 |
| Pareto 曲线（率扫） | P1 |
| 泛化四元组（≥3 架构 × ≥2 任务 × ≥3 数据集） | P1（可裁剪） |
| 噪声鲁棒、前瞻帧数表、检测器开销表 | P2 |

### 13.1 工作包（依赖序，非日历序）

**WP1 — 评测口径升级（P0，推理侧为主，1-2 周）**
- 产出 A：word-WER 解码管线。两条路线择一（决策门 D1）：
  - 路线 a（推荐）：word-BPE tokenizer 复刻冠军（重训 1-2 run，各 ~6h）——干净、口径无争议，作为投稿主配置；
  - 路线 b：音素→词还原（CMUdict 词典约束解码）——零重训但工程复杂、易被挑刺。
- 产出 B：test-clean + **test-other** 双测试集 WER 表（test-other 是 LibriSpeech 论文事实标配）。
- 产出 C：`eval.py` 架构一致性断言（strict=False 地雷，10-05 事故整改）。
- Go/No-Go D1：路线 a 复刻冠军 word-WER 与 PER 结论一致（soft>hard、learned 最优）→ 进入 WP2；若结论翻转，回炉归因（预期翻转概率低：机制与 tokenizer 无关）。

**WP2 — 统计严谨性（P0，4-6 GPU·h，1 周，依赖 WP1）**
- 冠军格（cosine_diff×learned）、次优（fir×seg_avg）、max-pool 基线各补 2 种子（共 6 run × ~6h）。
- 产出：mean±std 表 + paired bootstrap 显著性（冠军 vs max-pool，冠军 vs hard）。

**WP3 — 效率口径（P0，纯本地，3-5 天，可与 WP1 并行）**
- FLOPs 实测（thop/fvcore 双工具交叉验证）：不压缩 vs r=0.27 vs 率扫各点；detector 自身开销单列（checklist：<5%）。
- 延迟分解表：encoder / predictor-joiner / 解码循环（解释 RTF 陷阱）。
- 产出：论文效率表 + Pareto 图骨架。

**WP4 — Pareto 率扫（P1，推理-only，2-3 天，依赖 WP3）**
- 冠军 ckpt 推理时扫 keep_rate ∈ {0.15, 0.20, 0.27, 0.35, 0.50}（检测器分数是运行时旋钮，无需重训）。
- 产出：PER/FLOPs Pareto 主图（≥4 曲线：oracle/uniform/max-pool/ours）。

**WP5 — 泛化验证（P1，~4 GPU 天，依赖 WP1 口径）**
- 最小可投稿集（四元组裁剪版）：架构（Conformer→ squeeze/ECAPA 型前端 1 项）× 数据规模（100h→360h A6 锚点 1 run 15-18h）× 任务（ASR→命令词/说话人验证 1 项推理侧迁移）。≥3 数据集项可引 P2 的跨数据集结果补位。
- C2 可微门控（2-3 run）若赶不上投稿则移入扩展版，不阻塞主线。
- Go/No-Go D2：360h 锚点上冠军法仍 ≥ max-pool → 泛化故事成立；否则论文收缩为"100h 深度分析 + 机制发现"，投 workshop/期刊。

**WP6 — 外部基线对比（P0，4 周，风险最高，尽早启动）**
- P2（cosine-diff 分割，LALM 2025）：推理侧复现性高，优先。
- P1（PLE 边界+MASK，Interspeech 2026）：注意本仓库 PLE 复刻已崩坏——先做 P1 作者口径（变率+MASK 填充）核对是复刻错还是方法对 ASR 不适，结论无论哪边都可作为论文素材（复现性发现或负结果分析）。
- 止损：4 周内无法对齐任一方法 → 降级为 Related Work 定量引用 + 定位差异，主表不含外部基线（风险：reviewer 扣分项，需在 limitations 说明）。

**WP7 — 论文与复现包（P0，3-4 周，与 WP5/6 后期并行）**
- 故事线（§9 v2 微调）：以"检测×混合解耦发现 mixer 主导"为钩子，动机图用 A0 诊断，主表 = B-matrix 2D 热图 + WER 版基线对比，主图 = Pareto。
- 图表清单：架构图、2D 热图、Pareto（WP4）、A0 冗余分布、learned 权重可视化、延迟分解。
- 复现包：configs/exp + queue_runner + 清洗后 results.csv + 种子表。

### 13.2 关键决策门汇总

| 门 | 判据 | 通过 | 不通过 |
|---|---|---|---|
| D1（WP1 末） | word-WER 复刻结论与 PER 一致 | WP2 启动 | 回炉归因，暂缓 WP2 |
| D2（WP5 中） | 360h 锚点冠军 ≥ max-pool | 泛化故事完整版 | 收缩为 100h 机制论文 |
| D3（WP6 末） | ≥1 外部基线复现对齐 | 主表含外部对比 | limitations 声明 + 定位差异 |

### 13.3 GPU 预算汇总（粗估）

| 项 | runs | GPU·h |
|---|---|---|
| WP1 word-BPE 复刻 | 1-2 | 6-12 |
| WP2 种子 | 6 | ~36 |
| WP5 泛化 + A6 360h + C2 | 5-8 | 60-100 |
| 合计 | 12-16 | **~100-150**（1 GPU 约 6-9 周连续，与推理侧工作包并行无冲突） |

---

## 附录：papers/ 文献清单（2025+，均已下载）

| # | 文件 | 一句话 | venue |
|---|---|---|---|
| P1 | dtm-codec-vfr-boundary-interspeech26-2606.29480.pdf | 层间 PLE 边界选择 + MASK 填充的 VFR 语音 codec | Interspeech 2026 |
| P2 | audio-token-compression-lalm-2511.20973.pdf | LALM 音频 token 压缩对比：cosine-diff 分割/池化/采样 | arXiv 2025-11 |
| P3 | context-driven-dynamic-pruning-interspeech25-2505.18860.pdf | 上下文门控的逐层可微帧剪枝 | Interspeech 2025 |
| P4 | fastadasp-audio-token-merging-2410.03007.pdf | SpeechLM 层间免训练加权 token 合并 | arXiv 2024-10（最近邻，破格收录） |
| P5 | mms-llama-minimal-speech-tokens-2503.11315.pdf | 语速自适应动态分配 AV token 数 | arXiv 2025-03 |
| P6 | visionzip-cvpr25-2412.04467.pdf | attention 选 dominant + 相似度合并回收 | CVPR 2025 |
| P7 | divprune-cvpr25-2503.02175.pdf | Max-Min 多样性剪枝，training-free | CVPR 2025 |
| P8 | vscan-neurips25-2505.22654.pdf | 诊断驱动的 global+local 两段扫描剪枝 | TMLR 2026-01 |
| P9 | fastslm-hierarchical-temporal-2601.06199.pdf | 层级 query 蒸馏压长语音至 1.67 tok/s | arXiv 2026-01 |
| P10 | survey-token-compression-mllm-2507.20198.pdf | MLLM token 压缩首篇系统综述（42 页） | TMLR 2026-01 |
