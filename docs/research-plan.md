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
- **实验**：检测器 {find_peaks raw, SG（raw+savgol 平滑）, cosine-diff（P2 复现）, cosine-diff+savgol} × 混合器 {硬选择, 段平均, 学习 Hanning 权重}，等 FLOPs 档对比。FIR 平滑为 raw 的冗余变体，从叙事中去除（2026-10-05）；检测器集按"信号类型 ∈ {差分, 余弦} × 平滑 ∈ {无, SG}"2×2 网格完备化。
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
- **B-矩阵（I10）**：检测器 {raw, SG（raw+savgol）, cosine-diff, cosine-diff+savgol} × 混合器 {硬选, 段平均, 学习权重}，等 FLOPs 20% 档，2 种子 —— 论文主消融表。已跑 raw/savgol（=raw+SG）/cosine-diff 三格；**cos-diff+savgol 待补（3 runs × ~2.5-3h，100h/50ep，与既有 B-matrix 可比）**；FIR 去除。
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

## 9. 论文故事线（v3 更新，2026-10-05 按用户定位修订：**帧选择是最大优势，主线 = 低损耗的超高计算量缩减**）

### 核心论题（一句话）

> **ECHO 在流式 Conformer-RNNT 上实现 encoder FLOPs 3.7× 缩减（率扫最高冲 ~6×），精度代价 ≤1pt（beam 口径 +0.98pt）——这靠的是一个轻量、零学习参数、流式安全的帧选择器，把"算力花在内容变化的帧上"。**

### 叙事分工（谁主角谁配角）

- **主角 = 帧选择器**：信号处理式 novelty（帧间差分峰谷 / 余弦距离峰），零学习参数、开销 <5%、因果安全（右上下文受 chunk 约束）、内容自适应（对比 uniform/random 同码率基线拉开质量差）。**"选择"提供压缩本身。**
- **配角 = 学习混合器**：回收被丢帧信息，守住低损耗（相对 hard 平均 +1.87pt）——作为"低损耗"得以成立的机制保障，一段 + 一个消融表，不抢戏。
- **主图 = Pareto 率扫曲线**（r ∈ {0.15, 0.20, 0.27, 0.35, 0.50}，推理侧扫描无需重训）：给应用一张"压缩比 ↔ 代价"菜单——**2× 近无损 / 3.7× @ +1pt / 极限档 ~6×**。超高压缩的 claims 全部落在实测点上，不靠 extrapolation。

### 大纲（Interspeech 4+1 页）

1. **Intro**：端侧流式 ASR 编码器算力瓶颈 → 一句话抛出 3.7×@+0.98pt + Pareto 菜单图 → 三条贡献。
2. **Related**：token 压缩四象限（P2/P3/P4/P1 定位）；强调多数方法用昂贵的注意力打分或需训练的选择器，本方法用零参数信号先验。
3. **Motivation**：A0 帧间差异分布（冗余度的实证）+ RNN-T 逐帧对齐使丢帧可精确定义。
4. **Method**：选择器（novelty 信号 + 峰谷检测 + 等率 top-k，前瞻帧数表）+ 回收器（learned 边界混合，简述）+ 复杂度分析（检测器开销 <5%）。
5. **Results**：
   - **Pareto 主图**（率扫五档 × FLOPs 实测）——低损耗超高压缩的定量主证据；
   - 主表：冠军 8.79% beam vs 不压缩 7.81%（+0.98pt）+ uniform/random/max-pool 对照；
   - 消融表：5×3 矩阵作为**设计选择论证**——选择器只需"合理"即达峰值（鲁棒性 = 免调参、部署友好，是优势不是缺陷）；learned 回收守住质量；PLE 变率机制与等率冲突的教训。
6. **Generality**：跨架构（squeezeformer）/ 数据规模（100h→460h）/ 任务（KWS/SV 推理侧迁移）。
7. **Analysis**：为什么丢帧不掉分——音素去留统计（稳态元音高丢、边界低丢）+ learned 权重在边界自适应放宽的机制图（G7）。
8. **Conclusion**：Pareto 菜单 + 一句 limitation（RTF 陷阱 → 解码端另优化）。

### 贡献列表（Intro 尾部，3 条）

1. **ECHO 系统**：流式安全的零参数帧选择 + 可学习回收，encoder FLOPs 3.7×↓、精度代价 ≤1pt，率扫给出 2×-6× 全谱菜单。
2. **帧选择器实证**：简单信号处理检测器即可达到内容自适应选择的有效质量——零学习参数、<5% 开销、跨架构/数据/任务成立。
3. **解耦消融 + 机制解释**：选择/回收正交分解的 15 格框架 + 音素级去留分析，解释损耗来源与恢复机制。

### Reviewer 质疑预案（此叙事版）

| 质疑 | 回应 |
|---|---|
| 不压缩才 7.81%，压缩有什么用？ | Pareto 是菜单不是单点：r=0.50 档 ~2× 近无损；r=0.27 档 3.7× 换 1pt；端侧按预算选档。 |
| 检测器间差距小，选择不重要？ | 恰是优势：**选择器免调参、免训练、换架构即插即用**——工程价值；且 uniform/random 同码率显著更差，证明"内容自适应选择"本身有效。 |
| 与 P2（cosine-diff）区别？ | P2 是 LALM 离线分割；本方法给流式 ASR 全系统：等率机制 + 因果安全 + 回收器 + 率扫 Pareto + 跨架构泛化。 |
| 只有 PER？ | word-WER 由 WP1 承担（P0 硬门槛）。 |
| 速度没快？ | 主动披露 RTF 陷阱 + FLOPs/延迟分解表——解码端是独立问题。 |
| 单种子？ | 如实声明 + 率扫多点位 + 460h 锚点。 |

### 版本 B：ML 顶会版（保留，主线 I2+I10+G5+I8/I9）

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

## 13. 顶会冲刺执行计划（v2.2，2026-10-05 增补；同日修订：移除多种子工作包）

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
- Go/No-Go D1：路线 a 复刻冠军 word-WER 与 PER 结论一致（soft>hard、learned 最优）→ 进入 WP4（泛化验证）；若结论翻转，回炉归因（预期翻转概率低：机制与 tokenizer 无关）。

**WP2 — 效率口径（P0，纯本地，3-5 天，可与 WP1 并行）**
- FLOPs 实测（thop/fvcore 双工具交叉验证）：不压缩 vs r=0.27 vs 率扫各点；detector 自身开销单列（checklist：<5%）。
- 延迟分解表：encoder / predictor-joiner / 解码循环（解释 RTF 陷阱）。
- 产出：论文效率表 + Pareto 图骨架。

**WP3 — Pareto 率扫（P1，推理-only，2-3 天，依赖 WP2）**
- 冠军 ckpt 推理时扫 keep_rate ∈ {0.15, 0.20, 0.27, 0.35, 0.50}（检测器分数是运行时旋钮，无需重训）。
- 产出：PER/FLOPs Pareto 主图（≥4 曲线：oracle/uniform/max-pool/ours）。

**WP4 — 泛化验证（P1，~4 GPU 天，依赖 WP1 口径）**
- 最小可投稿集（四元组裁剪版）：架构（Conformer→ squeeze/ECAPA 型前端 1 项）× 数据规模（100h→360h A6 锚点 1 run 15-18h）× 任务（ASR→命令词/说话人验证 1 项推理侧迁移）。≥3 数据集项可引 P2 的跨数据集结果补位。
- C2 可微门控（2-3 run）若赶不上投稿则移入扩展版，不阻塞主线。
- Go/No-Go D2：360h 锚点上冠军法仍 ≥ max-pool → 泛化故事成立；否则论文收缩为"100h 深度分析 + 机制发现"，投 workshop/期刊。

**WP5 — 外部基线对比（P0，4 周，风险最高，尽早启动）**
- P2（cosine-diff 分割，LALM 2025）：推理侧复现性高，优先。
- P1（PLE 边界+MASK，Interspeech 2026）：注意本仓库 PLE 复刻已崩坏——先做 P1 作者口径（变率+MASK 填充）核对是复刻错还是方法对 ASR 不适，结论无论哪边都可作为论文素材（复现性发现或负结果分析）。
- 止损：4 周内无法对齐任一方法 → 降级为 Related Work 定量引用 + 定位差异，主表不含外部基线（风险：reviewer 扣分项，需在 limitations 说明）。

**WP6 — 论文与复现包（P0，3-4 周，与 WP4/5 后期并行）**
- 故事线（§9 v2 微调）：以"检测×混合解耦发现 mixer 主导"为钩子，动机图用 A0 诊断，主表 = B-matrix 2D 热图 + WER 版基线对比，主图 = Pareto。
- 图表清单：架构图、2D 热图、Pareto（WP3）、A0 冗余分布、learned 权重可视化、延迟分解。
- 复现包：configs/exp + queue_runner + 清洗后 results.csv + 种子表。

### 13.2 关键决策门汇总

| 门 | 判据 | 通过 | 不通过 |
|---|---|---|---|
| D1（WP1 末） | word-WER 复刻结论与 PER 一致 | WP4 启动 | 回炉归因，暂缓后续 GPU 工作包 |
| D2（WP4 中） | 360h 锚点冠军 ≥ max-pool | 泛化故事完整版 | 收缩为 100h 机制论文 |
| D3（WP5 末） | ≥1 外部基线复现对齐 | 主表含外部对比 | limitations 声明 + 定位差异 |

### 13.3 GPU 预算汇总（粗估）

| 项 | runs | GPU·h |
|---|---|---|
| WP1 word-BPE 复刻 | 1-2 | 6-12 |
| WP4 泛化 + A6 360h + C2 | 5-8 | 60-100 |
| 合计 | 6-10 | **~70-115**（1 GPU 约 5-7 周连续，与推理侧工作包并行无冲突） |

> 修订（v2.2，2026-10-05）：应用户决定移除原 WP2 多种子复跑工作包；learned vs seg_avg 等价性以单种子结果报告，论文中如实声明。

---

## 14. 泛化研究详细计划（v2.4，2026-10-05 增补；同日两次修订：加入 G6/G7，后续训练实验统一改用 460h 全量数据）

> 定位：对应 §13.1 WP4 的展开版。目标是在 B-matrix 已确立"mixer 主导、soft 制胜"之后，把核心结论从"Conformer-ASR-100h 单点"推向跨架构 / 跨数据规模 / 跨任务，形成顶会要求的泛化证据链。成本基准：100h/50ep 单 run 实测 ~2.5-3h（B-matrix 15 runs/37h）；**用户 10-05 决定后续训练实验统一 460h（train-clean-100+360 合并，~132,553 句），50ep 单 run 推算 ~12h（约 100h 的 4.6×）**。数据就绪：100h/360h 的 flac 与 -ph 音素标注均已验证；加载器当前仅支持单子集，需新增合并模式（train-clean-460）+ smoke 门禁。

### 14.1 研究问题与 reviewer 质疑映射

| RQ | 研究问题 | 回应的顶会质疑 | 优先级 |
|---|---|---|---|
| RQ1 | r=0.27 + "soft mixer 制胜"是否依赖 Conformer 特定架构？ | "只在 Conformer 上验证，换个 backbone 还有用吗？" | **P0** |
| RQ2 | 结论是否依赖 100h 小数据的冗余特性？ | "100h 太小，结论能 scale 吗？" | **P0** |
| RQ3 | 是否绑定 ASR 任务特性？ | "对其他语音任务有效吗？" | P1 |
| RQ4 | 压缩模块插入位置是否敏感？多次压缩（级联）能否优于单次？ | "为什么夹在 2+8 层中间？为什么只压一次？机制依据？" | P1 |
| RQ5 | top-k 等率选帧能否被可微门控超越？ | "为什么不用端到端可微剪枝（P3）？" | P1 |
| RQ6 | 被压缩掉的帧在语音学上是什么？为什么 soft 优于 hard？ | "方法像黑盒：压缩到底丢了什么信息？" | **P1** |

### 14.2 实验矩阵（G1-G5）

**G1 — 架构泛化（RQ1，P0）**
- `g1-squeeze-learned`：squeezeformer 式主干（替换 conformer block，保持 2+8 布局与压缩位），cosine_diff-learned 冠军口径，460h/50ep，~12h。
- `g1-squeeze-hard`：同主干 max-pool hard 对照，~12h。
- 产出：架构 × mixer 2×2 小矩阵——"soft>hard"若在新架构复现 → 结论升级为跨架构；否则限定 Conformer 并在 limitations 声明。
- 决策门 **D4**：G1 soft>hard 成立 → 泛化故事完整版；不成立 → 收缩为"Conformer 深度机制分析"。

**G2 — 数据规模泛化（RQ2，P0，用户决定直接上 460h 全量）**
- 规模轴 = 100h（B-matrix 已有结论）vs **460h**（train-clean-100+360 合并，~132,553 句；两段 flac 与 -ph 音素标注均已验证）。
- `a6-460-learned`：cosine_diff-learned，460h，50ep，~12h。
- `a6-460-maxpool`：max-pool hard 对照（D2 判据必需），~12h。
- `a6-460-baseline`（P1，预算允许则跑）：不压缩 460h 锚点，Pareto 图锚点，~12h。
- 前置：加载器新增 train-clean-460 合并模式（当前仅支持单子集）+ smoke 门禁。
- 免费附加：460h 冠军 ckpt 上 keep_rate ∈ {0.15,0.20,0.27,0.35,0.50} 推理侧率扫（与 WP3 Pareto 共享口径）。

**G3 — 任务迁移（RQ3，P1，推理为主，不占训练 GPU）**
- `g3-kws`：命令词/关键词检索（Speech Commands 或 LibriSpeech 内构 keyword 子集），现有 ckpt 零训练评测，~0.5 天工程。
- `g3-sv`：说话人验证（encoder 嵌入 + cosine scoring，VoxCeleb1-O），零训练，~1 天工程。
- `g3-sv-ft`（可选）：压缩 encoder 在 VoxCeleb 微调 1 run ~4h。
- 产出：压缩 vs 不压缩在下游任务的相对退化表（预期退化 ≤ ASR 主任务）。

**G4 — 插入位置消融（RQ4，P1）**
- `g4-pos-front` / `g4-pos-rear`：压缩模块移到 conformer1 前 / conformer2 后（现位 mid 已有），2 runs × ~12h（460h）。
- 依赖：A0 诊断层位统计（已有）。
- 注意口径：不同位置省的 FLOPs 不同（front 省最多——整个 encoder 都在压缩后帧率上跑），横向对比须用 WP2 实测 FLOPs 等价换算，不能只看 r。

**G5 — 可微门控 C2（RQ5，P1）**
- `c2-signal-init` / `c2-random-init`：门控 logit 用检测器分数初始化 vs 随机初始化，2-3 runs × ~12h（460h）。
- 产出：可微门控 vs top-k 等率对比；若超冠军 → 论文方法升级点；若平 → 反证等率 top-k 设计合理性。

**G6 — 多级（级联）压缩（RQ4，P1，用户 10-05 指定）**
- 动机：A0 诊断显示浅层帧间冗余更高——浅层粗压缩 + 深层细压缩有望在相同总预算下超越单点压缩；同时回应"为什么只压一次"。
- 布局：encoder 改 2+4+4，两个压缩点（block1 后 + block4 后），每点独立 detector+mixer（沿用冠军 cosine_diff+learned 口径）。
- 总预算约束 r1×r2 = 0.27，扫 split：
  - `g6-two-eq`：0.52 × 0.52（等比）
  - `g6-two-coarse-fine`：0.40 × 0.675（先粗后细，符合冗余随深度递减假设）
  - `g6-two-fine-coarse`：0.675 × 0.40（反向对照）
- 3 runs × ~12h = ~36h（460h）。
- 判据：总计算量与单点 r=0.27 等价（WP2 FLOPs 校准）下 PER 对比冠军 0.0915；若 two-stage 更优 → 论文方法升级点；机制产出 = "位置×次数" FLOPs-精度权衡图，配 A0 冗余分布解释。
- 注意：r 乘积 ≠ FLOPs 等价（各 block 帧率不同），等计算口径以实测为准。

**G7 — 可解释性分析（RQ6，P1，用户 10-05 指定，推理侧为主）**
- 前置：核实服务器 MFA 对齐工具状态（A0-2 遗留未验证项，G7a 依赖）。
- 产出（四件套，全部基于现有 ckpt，几乎零训练成本）：
  - **G7a 音素级去留统计**：MFA 对齐 → 音素类别 × 丢弃率（预期：元音稳态段高丢、辅音瞬变/边界低丢）+ 检测分数与音素边界的 Mann-Whitney U 检验（I4 承诺兑现）。
  - **G7b 检测分数 × 声学特征相关**：能量/短时过零/基频与各 detector 分数的 Pearson/Spearman 相关表——回答"检测器到底在看什么"。
  - **G7c learned mixer 权重可视化**：窗权重按语音上下文（稳态/瞬变/静音）分桶统计——直接解释 soft>hard 的机制（learned 在边界处自适应放宽混合 vs hard 硬切换）。
  - **G7d 表征与错误漂移**：压缩前后 encoder 输出 CKA 相似度（逐层）+ 不压缩 vs r=0.27 混淆矩阵差异（PER 增量集中在哪些音素对）。
- GPU·h：~1-2（表征/分数抽取），其余 CPU/本地分析，工程 ~3-4 天。
- 论文落点：Motivation 图（A0 冗余）+ Analysis 小节（G7a-c）+ 附录热图（G7d）；"mixer 主导"从经验发现升级为有机制解释的结论。

### 14.3 成本汇总

| 组 | 内容 | runs | GPU·h | 优先级 |
|---|---|---|---|---|
| G1 架构泛化 | squeezeformer 主干 × {learned, hard}（460h） | 2(+1) | ~24(+12) | P0 |
| G2 数据规模 | 460h × {冠军, max-pool, 不压缩锚点} + 免费率扫 | 2(+1) | ~24-36 | P0 |
| G3 任务迁移 | KWS/SV 推理侧零训练 | 0(+1) | 0(+4) | P1 |
| G4 插入位置 | 压缩移 conformer 前/后 vs 现位（460h） | 2 | ~24 | P1 |
| G5 可微门控 | 检测器分数初始化 vs 随机初始化（460h） | 2-3 | 24-36 | P1 |
| G6 多级压缩 | 2+4+4 双压缩点，r 拆分 3 种（460h） | 3 | ~36 | P1 |
| G7 可解释性 | 音素去留统计/分数-声学相关/权重可视化/CKA+混淆漂移 | 0 | 1-2 | P1 |
| **必选合计** | G1+G2 必选 | **4** | **~48** | |
| **全量合计** | | **13-16** | **~156-192**（约 7-8 GPU 天） | |

对照：B-matrix 15 runs 实测 37h——本计划必选部分 (~2 天) 远小于 B-matrix，全量 (~3-6 天) 可控。训练用统一 50ep（360h 用 40ep），评测口径与 B-matrix 一致（greedy PER 2620 句 test-clean；360h 结果另跑同集）。

### 14.4 执行顺序与基础设施

1. **前置改造**：加载器新增 train-clean-460 合并模式 + smoke 门禁。**G2 先行**（单 run 最长、承载 D2 判据）：写 stage4 queue 脚本（扩展 gen_configs 模板 g1/g2/g4/g6/c2 + smoke 门禁），`tmux new -s queue-g4` 提队。
2. G1 两 run 排队跟进（需先改 encoder 支持 squeezeformer block + 冒烟测试）。
3. G4/G5/G6 穿插填空（G6 需 encoder 支持 2+4+4 布局与双压缩点实例化；等 G1/G2 调试间隙的 GPU 空闲）。
3. G4/G5/G6 穿插填空（G6 需 encoder 支持 2+4+4 布局与双压缩点实例化；等 G1/G2 调试间隙的 GPU 空闲）。
4. G3/G7 推理侧随时并行（不占训练 GPU，与 WP2 效率口径共享工程；G7 前置核实 MFA 状态）。
5. 全部落库 results.csv（g1-/g2-/g3-/g4-/c2- 前缀），experiment-log 记分板续更。

### 14.5 顶会接收要求对照

| 接收要求 | 本计划覆盖 |
|---|---|
| 泛化性（常见底线：≥2 架构 或 ≥2 数据集 × ≥2 任务） | G1（Conformer+squeezeformer）× G2（100h→460h 规模轴）× G3（ASR+KWS/SV）三轴全覆盖 |
| 消融完整性 | G4 位置 + G5 可微对照 + G6 多级 + 已有 B-matrix 2D 主消融 |
| 机制可解释性（"为什么 work"） | G7 四件套（音素去留 / 分数-声学相关 / 权重可视化 / 表征漂移）+ A0 冗余分布 |
| 稳健性 | 单种子（用户 10-05 决定，论文如实声明）+ 推理侧率扫多点位 + test-clean 全量 |
| 口径一致 | 训练评测 PER；投稿主表 WER 由 WP1 冠军复刻承担，泛化实验不重训 WER 版 |

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
