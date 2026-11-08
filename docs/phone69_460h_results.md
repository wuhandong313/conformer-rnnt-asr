# Phone69 460h 当前结果与训练配置

当前研究使用 corpus PER 与主要乘加量 GMAC／输入音频秒比较质量和计算成本。PER 来自 [2026-11-08 权威评测归档](../papers/phone69_460h_cpu_authoritative_20261108/README.md)，全量计数使用相同 test-clean 的 2,620 句、19,452.480625 秒音频、FP32、batch 1、beam 20。下表为 E01–E06 已完成计数的快照，E07–E10 待补；试测数不填入正式表。

| 编号 | 配置 | Corpus PER | 编码器与压缩 | 预测器与 joiner | 合计 |
| --- | --- | ---: | ---: | ---: | ---: |
| E01 | stride4 无动态压缩 | 6.1289% | 0.11424 | 0.12980 | 0.24404 |
| E02 | 第2层五帧，stride1 | 5.5141% | 0.29744 | 0.18893 | 0.48637 |
| E03 | 第2层五帧，stride2 | 6.7135% | 0.12418 | 0.11075 | 0.23493 |
| E04 | 第2层五帧，stride4 | 7.6081% | 0.05641 | 0.07657 | 0.13298 |
| E05 | Mel 五帧 | 5.5782% | 0.19552 | 0.18655 | 0.38207 |
| E06 | 第5层五帧 | 4.7809% | 0.43193 | 0.19319 | 0.62512 |
| E07 | 第7层五帧 | 4.8603% | — | — | — |
| E08 | 第2层邻域3帧＋第5层五帧 | 5.4210% | — | — | — |
| E09 | stride4、512维无动态压缩主基线 | 5.7427% | — | — | — |
| E10 | Mel＋第5层五帧 | 5.6882% | — | — | — |

后三列单位均为 GMAC／输入音频秒，按总乘加数除以真实音频总时长计算。范围包括线性层、卷积、注意力矩阵乘积、融合及真实搜索触发的预测器和 joiner 矩阵运算；特征提取、归一化、softmax、激活、比较排序和访存不计入主要 MAC。E09 全量计数完成后再计算相对预算。历史计时字段保留在归档，当前研究不使用 RTF 或重复计时作为比较指标。[已完成计数与来源哈希](../papers/extremafuse_acl2027/compute_snapshot.json) · [计数方案](../papers/phone69_460h_compute_eval_plan_20261108.md)

三阶段 Mel＋第5层＋第8层的后续评测另见 [独立报告](../artifacts/evaluations/phone69_460h_mel_layer5_layer8_local_cpu_20261108/REPORT.md)，未改写冻结 E01–E10 归档。训练预算、更新数和模型容量并非完全匹配；解释比较时须结合下述训练配置。

## 共同配置

| 项目 | 配置 |
| --- | --- |
| 训练集 | train-clean-100＋train-clean-360，约460小时 |
| 验证集与选模 | dev-clean；依据保存的验证损失选择checkpoint，E10跨训练阶段比较保留的验证checkpoint |
| 标签 | 固定69个带重音ARPABET音素；blank ID 69，joiner输出70类；过滤spn |
| 音频与特征 | 16 kHz，80维Mel，n_fft 400，hop 160，约100帧/秒 |
| 骨干 | 10层Conformer，8头，FFN 512，卷积核31，dropout 0.1；主干宽度见下表 |
| 编码器与预测器接口 | 编码器输出256；embedding 128，两层128隐维LSTM，LSTM dropout 0.3 |
| 优化 | Adam，学习率0.0005，betas 0.9/0.999，eps 1e-8，warmup 10,000步，梯度裁剪10 |
| 损失 | RNN-T reduction=sum，clamp=1.0，按batch分块 |
| 数据增强 | SpecAugment：频率mask 27，时间mask 100，time_mask_prob 0.2 |
| 训练预算 | seed 42，目标50轮，单张GPU，梯度累积1；实际完成状态见总表 |
| 验证频率 | 配置为每200步验证；训练checkpoint间隔500步 |
| 拼批约束 | phone-token预算可变；rnnt_numel_limit=1,900,000,000；frames_per_char估计见下表 |
| 自然选点 | cosine_diff自然峰谷，无目标keep_rate；五帧融合读取中心及左右各2帧特征 |
| 推理语义 | offline整句，尾部zero_pad_ceil；不能解释为流式结果 |

本轮评测实际使用的归一化统计文件SHA-256为`022a496264bafdba8ecb65f72d71a579c76f6f8c56e41866a600272fc3b1c457`，来源是训练服务器`/test21/librispeech/global_stats.json`的本地副本。它与仓库`assets/statistics/global_stats.json`的哈希不同，复现时应沿用各CPU配置指向的训练统计文件。词表SHA-256为`381f0fa6dd78e828ccf68c0f4b5d3988aabb3976753d4ae62ffea7f0359876e6`。

## 配置差异与文件索引

“stride × 投影 → 主干”表示拼帧后的Conformer宽度；编码器输出维度始终为256。批预算按音素token计数，并非固定句数。参数量取自已有模型审计或预检记录。

| 编号 | stride × 投影 → 主干 | 总参数量 | phone-token预算 | frames_per_char | 对应训练配置 |
| --- | --- | ---: | ---: | ---: | --- |
| E01 | 4 × 32 → 128 | 4,207,942 | 23,500 | 3.0 | [YAML](../configs/server_phone69_460h_offline_native_stride4_none_s42_e50.yaml) |
| E02 | 1 × 128 → 128 | 4,236,299 | 6,500 | 3.5 | [YAML](../configs/server_phone69_460h_offline_native_stride1_cosd_learned_s42_e50.yaml) |
| E03 | 2 × 64 → 128 | 4,231,179 | 16,000 | 3.5 | [YAML](../configs/server_phone69_460h_offline_native_stride2_cosd_learned_s42_e50.yaml) |
| E04 | 4 × 32 → 128 | 4,228,619 | 32,600 | 3.0 | [YAML](../configs/server_phone69_460h_offline_native_stride4_cosd_learned_s42_e50.yaml) |
| E05 | 1 × 128 → 128 | 4,228,619 | 6,500 | 3.5 | [YAML](../configs/server_phone69_460h_placement_mel_stride1_s42_e50.yaml) |
| E06 | 1 × 128 → 128 | 4,236,299 | 5,000 | 3.5 | [YAML](../configs/server_phone69_460h_placement_layer5_stride1_s42_e50.yaml) |
| E07 | 1 × 128 → 128 | 4,236,299 | 6,500 | 3.5 | [YAML](../configs/server_phone69_460h_placement_layer7_stride1_s42_e50.yaml) |
| E08 | 1 × 128 → 128 | 4,236,299 | 4,000→6,000 | 3.5 | [YAML](../configs/server_phone69_460h_placement_layers2_5_n3_stride1_s42_e50_batch6000_resume.yaml) |
| E09 | 4 × 128 → 512 | 29,584,966 | 7,000 | 3.0 | [YAML](../configs/server_phone69_460h_offline_native_stride4_none_dim512_s42_e50.yaml) |
| E10 | 1 × 128 → 128 | 4,249,296 | 6,500→11,050 | 3.5 | [YAML](../configs/server_phone69_460h_placement_mel_layer5_stride1_s42_resume_b11050_e50.yaml) |

压缩位置及特殊设置：

- E02、E03、E04采用2＋8层两块结构，在第2层后压缩。
- E05在Mel域、输入投影前压缩，融合器输入维度80；E06、E07分别在第5、7层后对128维特征融合。
- E08第2层只保留峰谷中心及左右各1帧的并集，第5层再作五帧融合；第一阶段并非可学习五帧融合。
- E09主干512维，FFN仍为512、输出仍为256；它是扩大主干容量的参照，参数量并不匹配其他组。
- E10的Mel、第5层均作五帧融合。三阶段实验的配置与选模记录见独立评测报告。


## 续训与复现

E08 使用前三轮预算4,000、之后预算6,000的边界续训。E10 的初始、桥接和预算11,050恢复配方属于同一训练链，保留用于解释训练历史；桥接发生于轮中，不能视为样本顺序完全无变化的恢复。

各模型的精确训练配置、checkpoint 哈希、逐句结果与冻结源码位于权威归档；当前训练辅助配方保留在 configs/。更严格的受控比较要求见 [严谨性审查](experiments/phone69_460h_rigor_review_20261107.md)。
