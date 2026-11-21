# 计算量—识别性能曲线方案

## 2026-11-18 固定步长的后续实验

**固定 4∶1 的宽度对照。** 在原生拼接实现中，每帧投影 32、64、128 维给出 128、256、512 维主干。三者在 30 秒输入上都输出 751 步，计算量分别为 4.336、10.394、27.679 GMAC。11 月 21 日完成投影 64、主干 256 维的全量评测。该配置作为内部结果归档，论文采用下列 11 个工作点。原始 Conformer 的小、中、大模型都采用两层 stride-2 卷积获得 4∶1，宽度分别是 144、256、512，模型深度和头数等也有调整。本文的参照采用帧拼接。Fast Conformer 在输入端使用 8∶1。[Gulati 等，§3.2、表 1](https://www.isca-archive.org/interspeech_2020/gulati20_interspeech.pdf#page=3) · [Rekesh 等，§2](https://arxiv.org/html/2305.05084v4#S2)

当前入稿比较涵盖 11 个模型。固定组为 stride 2／256 维、stride 4／128 维和 stride 4／512 维。融合组为 Mel、L1、L2、L3、L5、L7、Mel+L3 和 Mel+L5。若研究问题转向同宽主干下步长本身的影响，再补 stride 8／256 维。相近预算或相近 PER 的描述性比较见正文配对表。每个参照独立按指定指标的绝对差选取融合模型，另一指标在配对内比较。新实验的预算和开发集选点规则见下文。[正文配对表](generated/matched_points.tex) · [全部 11 组读数](generated/operating_points.tex)

### 2026-11-19 追加 stride 4／256 维

新配方沿用已训练的 stride 2／256 维的 8,800 个目标音素批上限、460 小时数据、seed 42 和 50 轮，其余训练参数相同。每帧投影改为 64 维，拼接四帧得 256 维主干。队列入口等待前一批训练完成并释放 GPU，再持有共用 GPU 锁验证实际训练批，候选通过后从头训练。服务器状态写在 `output/phone69_460h_stride4_dim256_20261119/queue_state.json`，最终按 dev-clean 验证损失取 checkpoint。全量结果留在实验归档。[评测记录](../../artifacts/evaluations/phone69_remaining_local_cpu_20261121/REPORT.md) · [训练配方](../../configs/server_phone69_460h_offline_native_stride4_none_dim256_s42_e50.yaml) · [排队入口](../../scripts/server/phone69_stride4_dim256.py)

### stride 4／128 维的低预算对照

当前主图使用同一完整 test-clean 和 30 秒编码器与融合计数。单阶段 L3 使用原训练第 31 轮、step 57,500 的评测。两条阶梯分别表示各家族在给定 PER 上限内已观察到的最低成本。固定组有两个前沿点，即 stride 2／256 维（4.8989%，26.539 GMAC）和 stride 4／128 维（6.1289%，4.336 GMAC）。[当前结果图](figures/quality_budget_30s.pdf) · [11 月 18 日历史图](figures/stride4_128_comparison.pdf)

| 低预算模型 | 整句 PER | 编码器与融合 GMAC／30 秒 | 错误音素数 |
| --- | ---: | ---: | ---: |
| Mel+L3 | 6.0204% | 4.174 | 11,380 |
| stride 4／128 维 | 6.1289% | 4.336 | 11,585 |
| Mel+L5 | 5.6882% | 5.268 | 10,752 |

同为 128 维、同在 4.5 GMAC 上限内，Mel+L3 的计算量比窄 stride 4 低 3.73%，少 205 个错误。按 40 位说话人配对重采样 10,000 次（seed 42），stride 4 减 Mel+L3 的 PER 差值为 +0.108 个百分点，95% 区间为 [−0.034, +0.241]，包含零。若预算放宽到 5.5 GMAC，Mel+L5 比窄 stride 4 多花 0.932 GMAC，PER 低 0.441 个百分点，配对区间为 [0.303, 0.586]（同一减法方向）。预算上限与原始点都值得给出。[逐句评测来源及算量归档](../../artifacts/research/phone69_all_results_20261117/results.json)

横轴的 PER 上限从 6.10% 放宽到 6.13% 时，已测固定组的最小成本从 26.539 降至 4.336 GMAC，融合组均为 4.174 GMAC。两家族的成本比随之从 6.36 倍降到 1.04 倍。这一陡降由两个离散基线的间距决定，当时据此安排了两点之间的固定步长补测。

### 历史补测计划与算量

沿用离线原生拼接和 10 层 Conformer。表中 P 是每帧投影维度（`model.time_reduction_input_dim`），s 是步长（`model.time_reduction_stride`），D = s × P 是 Conformer 主干宽度。保持 8 头、FFN 512、卷积核 31、编码器输出 256 和相同的 predictor、joiner。第一轮锁定 D = 256，第二轮用 P 匹配论文关心的算量。下表保留当时的补测计划及固定输入理论算量。

| 轮次和用途 | s | P | D | 30 秒输出步 | 编码器 GMAC/30 秒 | GMAC/音频秒¹ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 已有，同宽参照 | 2 | 128 | 256 | 1,501 | 26.539 | 0.885 |
| 先跑，同宽中预算 | 4 | 64 | 256 | 751 | 10.394 | 0.346 |
| 先跑，同宽低预算 | 8 | 32 | 256 | 376 | 4.482 | 0.149 |
| 再跑，对照 Mel+L5 | 2 | 32 | 64 | 1,501 | 5.344 | 0.178 |
| 再跑，对照 Mel-only | 4 | 52 | 208 | 751 | 7.921 | 0.264 |
| 再跑，对照 L5 | 2 | 108 | 216 | 1,501 | 21.484 | 0.716 |

¹ 固定 30 秒成本除以 30。给定 3,001 个 Mel 帧，输出长度 L = ceil(3001 / s)，固定拼接模型的这些算量在训练前即可确定。按项目计数口径，MAC 总数为 `3001 × 80 × P + 10 × [L × (4 × D × 512 + 7 × D² + 31 × D) + 2 × D × L²] + L × D × 256`。三组对照当前归档的 Mel+L5、Mel-only、L5 算量分别相差 +1.44%、−2.39%、+0.49%，百分比以各自融合模型为分母。[固定时长计数器](../../scripts/evaluation/theoretical_phone69_macs.py) · [实际基线模型配置](../../configs/server_phone69_460h_offline_native_stride2_none_proj128_s42_e50.yaml)

第一轮只需两个新模型。stride 4／256 维的优先级最高。它在固定步长 4 下连接已有的 128 和 512 维两端（4.336 → 10.394 → 27.679 GMAC），也能与已有 stride 2／256 维组成同宽对照。再加 stride 8／256 维，同宽的三个步长便覆盖 4.482–26.539 GMAC，且在 4.5 GMAC 上限内有两种固定基线可比。原生拼接要求 P 随 s 调整，主干宽度和层数固定时，每帧投影容量仍有差异。历史 stride 8 的主干为 1024 维，成本 38.628 GMAC。[历史结果](../../artifacts/research/phone69_all_results_20261117/REPORT.md)

第二轮补三个预算对手。其中 stride 4／208 维与 Mel-only 都只在输入端压缩，适合检验原来相对宽主干 stride 4 的大幅节省，在固定组也按预算调宽后还能保留多少。若 5.5 GMAC 成为主打结论，可在查看新测试结果前加验 stride 8／P36／D288（5.285 GMAC）。它和 stride 2／P32／D64（5.344 GMAC）相差约 1.1%，直接检验相近预算下“大步长、宽主干”和“小步长、窄主干”哪个更适合固定组。基本方案共五个新配置，低预算加验为第六个。实施时给每组独立 YAML 和输出目录，通过 `train.py` 读取；原步长扫描脚本会校验每帧投影必须为 128。[旧扫描器的配方校验](../../scripts/server/phone69_stride_sweep.py)

### 训练、选点与读图

- **先对齐训练条件。** 使用 train-clean-100 + train-clean-360、相同 Phone69 标签、归一化、增强和数据顺序。初筛用 seed 42、50 轮、Adam 峰值学习率 0.0005、warm-up 10,000 次更新，每 200 次更新验证，按 dev-clean 验证损失选 checkpoint。原 stride 2 的有效配置为每批目标音素上限 8,800，验证最佳为第 46,360 次更新。新组优先复用该组批规则，核对实际每次更新的样本、音素和每轮更新数。若显存要求拆分，按同一有效批累计梯度，核对损失缩放与学习率进度。历史锚点与新条件对不上时先作历史参照，正式同训练条件的结论需补跑锚点。[冻结配置](../../artifacts/evaluations/phone69_460h_stride_sweep_local_cpu_20261111/stride2_proj128/training_config.yaml) · [完成记录](../../artifacts/evaluations/phone69_460h_stride_sweep_local_cpu_20261111/REPORT_complete.md)
- **看新 PER 前锁定预算。** 建议共同上限为 4.5、5.5、8.5、11、22、27 GMAC/30 秒。各家族在每个上限之内按最低 dev-clean 整句 PER 选模型，同分取较低算量，按未舍入成本判断资格。对旧融合模型也用同一 dev-clean 逐阶段保留率重新校准 30 秒 MAC，再冻结名单；上表所引用的历史融合算量只用于规划。近等算量的补充比较预设相差不超过融合模型成本的 3%，同时给出两个原始值。例如 21.484 比 21.380 高，二者可在 22 的共同上限内比较。
- **测量和作图。** 冻结后按已有本地整句 FP32、batch 1、beam 20 解码，报告 corpus PER 和逐说话人配对的 95% bootstrap 区间，同时单列实际解码的 predictor、joiner 及总 GMAC/音频秒。主图沿用横轴 PER、纵轴 30 秒编码器与融合 GMAC。以空心符号标出所有本轮预定的固定点，在图下注明 s、P、D。D=256 的三点可标为同宽系列；固定组的阶梯线取已测点在各 PER 上限下的最低成本。若某个新点被支配，仍展示该点。共同预算下的正式比较使用事先冻结的开发集胜者。这样即使固定组胜出或前沿仍只有一个点，也能如实回答问题。
- **确认核心结论。** 现有 test-clean 参与过研究设计。针对一个预先指定的主预算，把两侧选中的配置各用 seed 43、44 再训一次（共四次确认训练），分别报告种子间变化和同一测试集上的配对区间。有兼容 Phone69 标签的未用测试集时，用已经冻结的选择再作独立确认。

若论文要把收益进一步归因于**按内容选择中心**，可另做一个输入端机制对照。在同一 128 维主干和同一五帧可学习 mixer 上，只把中心规则改成均匀间隔，保持掩码、边界处理、训练配方和解码一致。固定规则按训练或开发集的全局保留率决定每句输出数，与 Mel-only 的平均帧预算对齐，分别从头训练并报告实际成本。这项对照回答中心选择的作用，256 维原生拼接系列回答固定步长系统自身的效率与性能取舍。[现有等长度冻结诊断](README.md) · [Fast Conformer 表 2 对步长与通道的逐项比较](https://arxiv.org/html/2305.05084v4#S2.T2)

### 固定输入时长的文献依据

| 研究 | 参考时长 | 论文实际统计的对象 |
| --- | ---: | --- |
| Rekesh 等，Fast Conformer，ASRU 2023 | 30 秒 | 用 DeepSpeed 在单条音频上估算编码器 MAC，与 LibriSpeech WER 并列。与本文主图的模型和计量范围最接近。[§3.1.1、表 4](https://arxiv.org/html/2305.05084v4#S3.SS1.SSS1) |
| Choi 等，On-device Streaming Discrete Speech Units，Interspeech 2025 | 60 秒 | 用 calflops 对每分钟输入计算语音到离散单元的前端 TFLOPs；图 1 与下游 LibriSpeech WER、ML-SUPERB CER 一起画质量与计算量曲线。[图 1、§2.3](https://www.isca-archive.org/interspeech_2025/choi25b_interspeech.pdf) |
| Li、Chen、Qian，Dual-Path Modeling With Memory Embedding Model for Continuous Speech Separation，IEEE/ACM TASLP 2022 | 60 秒 | §V-E 明确在 60 秒输入上统计连续语音分离模型的 MAC；表 I 同列分离后语音的 LibriCSS WER。这是直接使用 MAC 的语音前端先例。[§V-E、表 I，期刊第 1515 页](https://audiocc.sjtu.edu.cn/user/pages/03.research/02.publications/2022/Dual-Path%40Modeling%40With%40Memory%40Embedding%40Model%40for%40Continuous%40Speech%40Separation/paper.pdf#page=8) |
| Lee、Chun，FastSLM，Findings of EMNLP 2026 | 300 秒 | 对五分钟输入比较语音压缩后的下游 LLM FLOPs。平均池化 30.6 TFLOPs，HTA 2.15 TFLOPs。其统计范围到语言模型，是更长时长的旁证。[§4.3、表 2](https://aclanthology.org/2026.findings-emnlp.256.pdf#page=6) |

建议主表用 30 秒与 Fast Conformer 对齐，补充表统一报 60 秒并保留各论文原有的 MAC 或 FLOPs 单位。下表列出当前入稿的全部 11 个模型。读取各模型在完整归档中列出的 `config.yaml` 和 `stage_rates`，用 `projected_trace` 和 `encoder_macs` 将输入从 3,001 个 Mel 帧改为 6,001 个，累计 batch 1 整段编码器与融合的主要 MAC。全局自注意力的成本含长度的二次项。30 秒时 stride 2／256 维低于 stride 4／512 维，60 秒时二者互换。真实 test-clean 最长 34.955 秒，2,620 条中仅 9 条达到 30 秒。60 秒这列描述模型对输入长度的计算量敏感性，PER 仍来自原始整句。[计数与算子范围](../../docs/phone69_fixed_duration_macs.md) · [归档的真实时长](../phone69_460h_cpu_authoritative_20261108/supporting/original_archive/E01/attempt_01/utterances.csv)

| 模型 | 30 秒 GMAC | 60 秒 GMAC |
| --- | ---: | ---: |
| stride 2／256 维 | 26.539 | 76.108 |
| stride 4／512 维 | 27.679 | 66.850 |
| stride 4／128 维 | 4.336 | 11.548 |
| Mel | 8.115 | 23.331 |
| L1 | 10.957 | 33.134 |
| L2 | 13.775 | 42.868 |
| L3（第 31 轮，step 57,500） | 16.347 | 51.822 |
| L5 | 21.380 | 69.390 |
| L7 | 26.499 | 87.205 |
| Mel+L3 | 4.174 | 11.315 |
| Mel+L5 | 5.268 | 14.657 |

下文保留既往方案及当时的入选集合。当前入稿范围见[论文说明](README.md)。

## 2026-11-12 固定时长曲线（历史）

这份历史图保留八组配置，分别为固定 stride 1／2／4（主干 128／256／512）、单阶段 Mel／L2／L5／L7 和双阶段 Mel+L5。

核心曲线横轴为固定30秒、实测逐层保留率校准的编码器＋压缩理论GMAC／音频秒，纵轴为原始完整test-clean PER。入选集合的观察前沿为 Mel+L5 → Mel → L2 → L5。连线连接已有工作点。它是筛选后的探索性系统比较。当时五个新增位置尚待核验。

[历史图和八组完整表](fixed_duration_curve_20261112/REPORT.md) · [入选清单](fixed_duration_selection.json) · [固定时长算法](../../docs/phone69_fixed_duration_macs.md)。重建：`python papers/extremafuse_acl2027/plot_fixed_duration_quality.py`。原始评测和下述历史真实搜索计数不覆盖；实际predictor/joiner分项保持独立，不混入理论编码器曲线。

## 历史完整语料计数方案（保留来源）

当前六组按新论文编号 E01–E06 命名，原始归档编号见 [README](README.md#当前论文实验与编号)。固定stride1、2、8已完成50轮和本地全量PER／主要MAC评测，层数保持10、投影维度128，见[完成报告](../../artifacts/evaluations/phone69_460h_stride_sweep_local_cpu_20261111/REPORT_complete.md)。2026-11-12进一步授权[层位网格与自然语速补充](placement_rate_supplement_20261112.md)：已有位置复用，只训练五个缺少的位置，原六组快照保留。

### 历史六组结果

横轴为完整 test-clean 的总主要 GMAC／输入音频秒，包含编码器、压缩、实际 beam 20 搜索触发的 predictor 与 joiner；纵轴为 corpus PER。两轴都越低越好。辅助表继续分列编码器和解码端，避免只省了编码器却遗漏搜索成本。所有数值来自已核验的 [全量快照](compute_snapshot.json)。

| 论文编号 | 配置 | 总 GMAC／音频秒 | 相对 E05 | PER | 观察到的作用 |
| --- | --- | ---: | ---: | ---: | --- |
| E06 | ExtremaFuse-Mel-L5 | 0.23252 | 25.08% | 5.6882% | 最低计算量工作点 |
| E02 | ExtremaFuse-Mel | 0.38207 | 41.22% | 5.5782% | 前端单阶段替换 |
| E01 | ExtremaFuse-L2 | 0.48637 | 52.47% | 5.5141% | 早层单阶段工作点 |
| E03 | ExtremaFuse-L5 | 0.62512 | 67.44% | 4.7809% | 现有最高精度工作点 |
| E04 | ExtremaFuse-L7 | 0.70362 | 75.91% | 4.8603% | 点估计的质量和成本均未超过 E03 |
| E05 | Stride4 主基线 | 0.92697 | 100.00% | 5.7427% | 固定降采样参照 |

按已观察到的点估计，E06→E02→E01→E03 构成非支配前沿：在这六个候选中，没有另一个点同时具有更低或相同的计算量和 PER、且至少一项严格更低。E04、E05 仍显示在图中。连接线只帮助阅读已测点，不估计点间性能；这些分别训练的模型不构成同一个 checkpoint 的连续可调曲线。不能把 E06 与 E05 的小幅 PER 差写成已证实更准确，其配对区间仍跨零。

四个单阶段位置 Mel、L2、L5、L7 已覆盖位置与计算成本的变化。论文主问题仍是方法相对固定 stride 的质量—成本取舍。位置结果帮助提供多个工作点，不把“哪个层最好”改成方法贡献。

### 历史固定 stride 扫描的控制条件

按用户指定，保留每帧投影 128 和 10 个 Conformer 层，固定 FFN 512、8 个注意力头、卷积核 31、输出 256、predictor 与 joiner。新增无自适应压缩的 stride 1、2、8，原 E05（stride4）继续复用。取消以减少层数调预算的建议。

本项目 stride 为帧拼接，主干宽度等于 `128 × stride`。因此 stride 1／2／4／8 的宽度分别为 128／256／512／1024，参数约为 4.216／10.378／29.585／95.523 M。层数与投影匹配，主干宽度和参数量随这一降采样实现改变，需并列披露。不能预设 stride 越大，总 MAC 越少：序列变短和主干变宽同时影响计算。

相同完整test-clean长度上的编码器主要MAC已解析计算并通过mock及stride4历史逐句计数核对：stride1／2／4／8分别为0.66752／0.64177／0.80207／1.23010 GMAC／音频秒。新三组完整解码计数已经完成，stride1／2／8的PER分别为4.985637%／4.898875%／6.678552%，总主要GMAC／音频秒分别为1.037971／0.861297／1.312000；具体分项和checkpoint见完成报告。[范围、公式与核验记录](../../docs/phone69_460h_stride_sweep_20261108.md#编码器固定形状计算量)

三组从头训练 50 轮，使用同一 460h 数据、seed42、增强、Adam、学习率与 warm-up、裁剪、验证选模和 beam20。按用户要求，每组独立试探最大稳定 batch，记录 token 预算、实际每批样本数、优化器步数和学习率进度；不同 batch 不宣称完全相同的优化暴露量。

串行顺序为 stride1 → stride2 → stride8。每组先用可丢弃模型执行真实训练批的前向、反向、梯度裁剪与优化器更新，逐步增大预算，触及显存或 RNNT 元素边界后缩小区间，以 100 个目标音素为粒度细化。最终预算再经两轮压力批与固定随机批确认，保留约 4% 显存空间。之后启动该组正式训练；完成后自动校准并运行下一组。只修改运行目录中的有效 batch 配置，原基线和旧 checkpoint 保留。

### 历史扫描的读图与判定规则

三组已完成，可分别使用同一完整测试集上的PER和总主要GMAC／音频秒，以stride1／2／4／8的真实坐标更新固定降采样家族。方法侧沿用现有五个配置，并在新增层位全量结果核实后扩展，显示所有点和各自观察前沿。连接线只辅助阅读，不插值未知质量；当前主图快照尚未自动替换为扩展集合，避免把已完成补充与原六组来源混写。

先用 dev-clean 冻结模型选择、计算预算和允许的质量损失，再在独立 test-other 确认。同预算比较可采用共同上限：每类方法在上限以内选 dev PER 最低的候选，不强求两个整数 stride 的 MAC 恰好相等；若额外报告近似等算力配对，预设容差并展示真实差值。不能将测试点外推成相同算力，也不能用两条线的面积代替已测工作点证据。

固定位置、主干及训练配方的独立均匀池化仍用于归因自适应融合机制。自然峰谷数由内容决定；当前主线不新增目标保留率、固定 K 或阈值选择。beam 扫描可作为无需重新训练的搜索预算补充图，单列其含义。

参考：Fast Conformer 将识别错误率、GMAC 和结构配置并列展示；本项目的 stride 网格及串行最大 batch 校准由用户指定。[Rekesh et al., ASRU 2023, Table 2](https://arxiv.org/html/2305.05084v4#S2.T2)
