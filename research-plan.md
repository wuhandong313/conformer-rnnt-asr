# 论文研究计划 v2.0：基于帧间差分峰谷检测的层间动态序列压缩

> 版本：v2.1（2026-09-30）
> 基线：Conformer-RNN-T + DynamicCompression（`conformer_rnnt/compression.py`），LibriSpeech 音素，指标 PER/RTF
> 目标：语音顶会（Interspeech / ICASSP / ASRU）接收标准，保留 ML 顶会（NeurIPS / ICLR）升级路径
> v2.0 变更：基于 10 篇 2025+ 顶会/前沿论文精读（见 §1 与 papers/ 目录）重构——**新增 scoop 风险分析与差异化定位**；**A4 基线池从 6 个扩到 10 个**（补入 2025+ 已发表方法作为必须对比的强基线）；**新增 Phase A0 诊断性 pilot study**；**消融改为"检测×混合"正交两组件**；**效率评测升级为 FastAdaSP 式 FLOPs 档位对齐**；**机制分析引入统计检验模板**。
> v2.1 变更：基于 §1.5 补充检索定向补强——**A4 加 DP 最优分段 oracle 天花板（P12）**；**C1 Pareto 加 oracle 曲线 + encoder/predictor 误差分解**（JSALT decoder–segmenter bias 启发）；**A0-2 边界 F1 改音素+词双基准**（P13"自然单元优于 MFA 音素"）；**B7 引 LACE 层位特异边界佐证**；**风险 R9（JSALT 组 learned segmenter 先发）**；**新增 A6 数据规模扩展（360h 音素标注已就位）**。不加新 Phase，时间线不变。

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

### 1.5 补充检索（2026-09-30，复检 6 个角度，无新增高重合）

针对"层间压缩 / RNN-T 帧跳过 / A-ToMe / 语音基础模型 token 压缩 / ICASSP 2026 / 动态降采样"再次检索，新增三条线索：

**[P11] LACE（SLT 2026，arXiv 2609.17509，CMU，已下载）**
codec 线延续：在 RVQ **每个量化层**独立做压缩分段——不同量化层的残余嵌入时间变化率不同，故各层拥有各自的分段边界；union alignment + boundary anchor 机制对齐层间时长供 TTS 使用；LibriTTS 重建任务，代码入 ESPnet3。**仍是 codec/重建线**，引用级。DTM→CodecSlime→LACE 半年三连，codec VFR 线在快速迭代。

**[P12] CodecSlime（ICASSP 2026，arXiv 2506.21074，已下载）**
即 DTM 引文 [17]：在 encoder 与量化器之间插入 ScheDFR 可调度降采样模块；给定目标率 R_S，用**动态规划**求分段 s* 最大化代理目标 J_h = −Σ‖h_t − h'_t‖²（段平均前后的特征 L2 距离），段平均下采样 + 每合并帧 ⌈log₂U⌉ bit 时长信息解耦内容与时长。重建任务，引用级。其 motivation（"FFR 在长元音/静音等平稳段浪费 token"）可直接引作 §Motivation 的领域前提证据。

**[P13] JSALT 2026 OmniEnc（工作组项目站 borrisonxiao.github.io/jsalt26-downsampling，暂无论文）⚠️ 盯住**
最接近的竞品动向：一整个 JSALT 工作组做 "learned dynamic downsampling for speech LLMs"——LibriSpeech ASR 锚点（10.3 Hz token，WER 4.89/9.43）、跨语言复制（GRPO 学分段器胜各语言最优 oracle）、情感/说话人/意图多任务。机制 = **强化学习（GRPO）学分段器** + order-aware pooling，非信号检测；目标是 token 率而非 encoder 层间算力。两个预警：① 其"自然书写/韵律单元当池化 oracle **优于 MFA 音素**"的发现，要求 A0-2 边界 F1 的基准解读留有余地；② 其论文挂出（大概率投 ICASSP/SLT）后 Related Work 必须新增对比。

**对结论的影响**：不变——"层间 + novelty 检测 + 学习混合 + 理解任务 + 真缩短序列"的精确组合仍无人占；但 codec 线半年三连 + JSALT 组进场，**窗口判断收紧至约 1 年**。A4/C1 优先级不变；新增监控项：JSALT 组产出与 Interspeech 2026 收稿。

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
| **压缩上界** | 无 oracle | **DP 最优分段作"天花板"曲线，量化检测器离上限多远** | P12 |
| **误差归因** | 只看总 PER | **分档分解 encoder 压缩误差 vs predictor/joiner 误差** | JSALT decoder–segmenter bias |

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
- **方法**：在无压缩锚点 ckpt（attn v2 `val-epoch=42-step=26700`）上做推理-only 分析：逐层帧间差分分布、novelty 稀疏度（峰占比）、相邻帧余弦相似度直方图、按音素/静音分段的冗余热力图、**层位诊断**（block 1..N 各层 novelty 信号与音素+词边界的重合度——直接决定 B6 插入位置）。
- **产出**：论文 §Motivation 的 2 幅图 + B6 的层位选择依据。

### I1. 可控压缩率与 Pareto 前沿（P0，保留）
- v2 修订：压缩率档位**改用 FLOPs 降低档 {10%, 20%, 30%, 40%, 50%}**（P4 口径），而非帧数比例——FLOPs 档天然把"帧数 ↓"换算成"算力 ↓"，与效率故事直接挂钩；同时报帧数率作对照。
- 加 **η = ΔPER/ΔFLOPs 拐点**标注最优工作点（P9）。
- 加 **≤25% FLOPs 极端档**的 graceful degradation 分析（P7）。
- **v2.1 加 oracle 天花板**（P12 DP 最优分段，全局非因果）：同图画出。检测器逼近上限 → 零参数方法是卖点；差距大 → 留给 I2 学习门控的空间，两种结果都有故事。
- **v2.1 加误差分解**：每个 FLOPs 档把 PER 增量归因到 encoder 压缩 vs predictor/joiner（固定解码器、只换分段方案，对比 oracle/我们的/随机分段下的误差转移），正面回应 JSALT 发现的 decoder–segmenter bias——压缩带来的错有多少能被 predictor/joiner 容忍。

### I2. 可微分动态帧选择器（P1，保留，Tier 3 旗舰）
- 不变：信号先验初始化 + Gumbel-sigmoid + budget loss。
- v2 补充：与 P3 的 Gumbel 剪枝正面对话——我们的差异是"信号先验初始化"这一归纳偏置，实验里必须有"随机初始化 vs 信号初始化"的对照行。

### I3. 因果/流式 → Future Work（保留 v1.1 降级决定）

### I4. 声学边界对齐分析（P1，v2.1 修订为双基准）
- 边界 F1 之外，加 P3 式**统计检验**——选中帧距最近边界的距离分布 vs 随机/均匀基线的分布，Mann-Whitney U 检验报 p 值。可视化沿用"diff 信号 + 选中帧 + 边界叠加图"。
- **v2.1：边界基准改为音素 + 词双轨**（MFA 对齐两者都出）。P13（JSALT）发现自然书写/韵律单元当池化 oracle 优于 MFA 音素——**不把音素当唯一真值**：论文表述为"novelty 峰与声学过渡段对齐"，F1 作为佐证而非命题本身，避免被"音素边界未必是最优压缩单位"反将一军。

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
| A0-2 | 层位诊断：各层 novelty 峰与**音素+词**边界的 F1（容差 ±2 帧，双基准） | 决定 B6 插入位置的数据表 |
| A0-3 | 压缩率现状分布：现有 find_peaks 在各 filter 下的保留率直方图 | §Setup 基础数据（v1.1 指出的盲区） |

### Phase A：地基工程（第 1–3 周，保留 v1.1）
A1 向量化（max_pool1d 等价 local-max）；A2 word-level WER；A5 冗余度统计。**A4 基线池 v2 扩充为 10 个**：

| 类别 | 基线 | 来源 |
|---|---|---|
| 平凡 | 均匀下采样、随机丢帧 | 必答审稿人 |
| 信号 | top-k-diff、能量 top-k | v1.1 已有 |
| 池化 | max-pool、avg-pool | v1.1 已有 |
| **2025+ 检测器** | **PLE 累积阈值边界（P1 复现）**、**cosine-diff 峰检测 + 段平均（P2 复现）** | 必须对比 |
| 几何 | 分段约束 Max-Min diversity 选帧（P7 思想） | 补强 |
| **oracle** | **DP 最优分段（P12 复现，全局/非因果——作天花板，不作对比基线）** | 量化检测器离上限距离 |

全部基线与我们的方法在**相同 per-utterance 保留帧数**下对比；另按 FLOPs 档位对齐画汇总表（P4 结构）。

**A6（v2.1 新增，P1）数据规模扩展**：train-clean-360 音素标注已就位（2026-09-30 生成，104,014 句）。主配置在 100h 验证成立后将锚点复训至 360h（约 3.6× 训练量），主表至少保留一行 360h 结果——这是顶刊版（TASLP 路线）的体量支撑，也回应"结论是否依赖小数据"的审稿质疑。

### Phase B：核心消融（第 4–6 周）—— v2 重构为两组件正交
- **B-矩阵（I10）**：检测器 {raw, FIR, SG, PLE, cosine-diff} × 混合器 {硬选, 段平均, 学习权重}，等 FLOPs 20% 档，2 种子 —— 论文主消融表。
- B3 检测目标（峰/谷/峰+谷）、B4 差分度量（L1/L2/cosine）、B5 超参敏感性：保留，降为单轴扫描。
- **B6 插入位置**：由 A0-2 诊断结果选 3 个候选层，不再盲扫（P8 做法）。
- **B7 两段压缩**：改（R₁, R₂）保留率网格 {单级, (10%,10%), (20%,10%), (10%,20%)}（P8 网格消融）。**v2.1 注**：LACE 已证实不同层位的表示变化率不同、各层应独立定边界——两段压缩不必同率，网格里的非对称档位（20%,10%）/（10%,20%）正是检验这一点。

### Phase C：创新实验（第 6–12 周）
| # | 内容 | v2 变化 |
|---|---|---|
| C1 | Pareto 主图 | FLOPs 档位口径 + η 拐点 + 极端档退化曲线 + **oracle 天花板曲线** + **误差分解子图**；≥5 条曲线（我们/均匀/top-k/PLE/cosine-diff/oracle） |
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
| P0 | ~~A3 锚点×3 种子~~ **已取消**：复用 attn v2 best ckpt（`exp/checkpoints/val-epoch=42-step=26700.ckpt`，greedy PER 11.12%）作无压缩锚点；多种子方差留到 B-矩阵关键格补 | 0 次新训练 | 统计基础 |
| **P0** | **A4 基线池（10 个，含 PLE + cosine-diff）** | ~10–14 次训练 | **决定故事走向 + scoop 防御** |
| P0 | B-矩阵（检测×混合） | ~15 次训练（部分可 1–2 种子） | 主消融表 |
| P0 | C1 Pareto（FLOPs 档） | 推理-only 为主 | 主图 1 |
| P0 | D0+D1+D2 | ~4–7 次训练 | 泛化骨干 |

（P1/P2 行同 v1.1，此处不重复；D3/D4 跨任务、C4 边界、Phase E、C2、D5/D6、Phase F 依序。**v2.1 新增 P1**：A6 360h 规模验证 ≈3.6× 单训——主表强度与顶刊版支撑。）

---

## 8. 风险登记（v2 更新）

| # | 风险 | 概率 | 对策 |
|---|---|---|---|
| **R0（新）** | **被 P1 类工作进一步挤压**（层间动态选择论文增多） | 中 | ① 尽快投出（§10 倒排）；② 差异化三支柱写进 Intro 首段；③ A4 正面对比 PLE——若我们胜出是强结果，若打平转"检测×混合解耦 + 泛用性"故事 |
| R1 | 等压缩率简单基线打平 | 中 | 同 v1.1：早跑 A4；转向假说 4.0 分析型故事；强调软混合独有优势（硬选择基线无信息回收，P2 已证 merging>sampling） |
| **R9（v2.1 新）** | **JSALT 组 learned segmenter 在"动态降采样+ASR"上先发论文** | 中 | ① 每月 arXiv 监控（§10 配套）；② 若先发：引为"学习派"对照，我们的零参数/流式/层间算力定位与之互补；③ C2 的"信号初始化 vs 随机初始化"对照即正面回应"先验 vs 学习" |
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
| 1–3 | A1/A2/A5 + ~~A3 种子~~（取消，用 attn v2 既有锚点）+ **A4 十基线（含 PLE/cosine-diff 复现）** | **Go/No-Go：等率对比（含 2025+ 方法）** |
| 4–6 | B-矩阵 + 单轴消融；C1 推理-only 部分 | 主消融表 + Pareto 主图 |
| 6–10 | D0–D4；C4 MFA+统计检验；论文骨架 | 泛化过半 + 主图 2 |
| 10–12 | D5–D8 heatmap；Phase E 双表 + 检测器开销表；显著性 | 完整结果 |
| 12–14 | 写稿 + 内审 + checklist 终检 | 投稿 |

> 全程配套：**每月一次 arXiv 文献监控**（codec VFR 线：DTM/CodecSlime/LACE 后续；JSALT 组产出；见 §1.5），命中新相关即更新 §1 与 A4 基线池。
> GPU 调度明细（队列/时长/基础设施约定）见 `docs/training-plan.md`。

---

## 11. 工程改造清单（v2 增量）

1. `compression.py`：保留 v1.1 全部条目；**新增 `detector: {find_peaks, ple, cosine_diff}` 与 `mixer: {hard, segment_avg, learned}` 两个正交配置轴**（替代单一 `compression_filter` 枚举，旧配置映射保留）；PLE 复现 = 累积余弦距离过阈值放边界（≤30 行）。
2. `baselines/fixed_compressors.py`：A4 十基线同接口实现。
3. `analysis/`：新增 `layer_redundancy.py`（A0-1）、`layer_boundary_f1.py`（A0-2）、`rate_distribution.py`（A0-3）；`boundary_f1.py` 加 scipy.stats 的 U 检验/t 检验输出。
4. eval：保持率 %、FLOPs 档对齐汇总表生成脚本。
5. `analysis/`（v2.1）：DP 最优分段 oracle（P12 复现，推理-only）与误差分解脚本；eval 增加**因果检测器变体**纯评估模式（savgol→因果 FIR，不换权重，量化流式惩罚——为"天然适配流式"的定位提供一行实测数字）。

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
- [ ] **Related Work 覆盖全部 papers/ 文献（含 P11–P13）并明确差异化**
- [ ] **oracle 天花板曲线（DP 最优分段）与差距分析**
- [ ] **边界 F1 音素+词双基准**
- [ ] **因果检测器变体的 eval-only 流式惩罚量化**
- [ ] **代码与配置开源包（YAML/种子/analysis 脚本）**
- [ ] **主表含 ≥1 行 360h 规模结果（A6）**

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
| P11 | lace-layerwise-dfr-codec-slt26-2609.17509.pdf | 每个 RVQ 量化层独立分段边界的动态帧率 codec（DTM 线延续） | SLT 2026 |
| P12 | codecslime-vfr-temporal-redundancy-icassp26-2506.21074.pdf | DP 求最优分段的 codec 时间冗余压缩（即 DTM[17]） | ICASSP 2026 |
| P13 | （无 PDF，项目站 borrisonxiao.github.io/jsalt26-downsampling） | GRPO 学习分段器做语音 LLM 动态降采样，LibriSpeech ASR 锚点 | JSALT 2026 工作组 |

> P11/P12/P13 为 2026-09-30 补充检索所得（§1.5），重合度均低于 P1，不改变差异化定位。
