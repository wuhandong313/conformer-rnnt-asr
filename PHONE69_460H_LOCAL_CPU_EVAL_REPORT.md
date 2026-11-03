# Phone69 460h 模型本地 CPU 串行评测报告

评测日期：2026-10-29（Asia/Shanghai）  
数据集：LibriSpeech test-clean，完整评测 2620 条音频  
指标：phone69 corpus PER；解码 beam width=20  
执行方式：CPU 串行；PyTorch 线程数 4、interop 线程数 1；DataLoader workers=0  
结果目录：`artifacts/evaluations/phone69_460h_local_cpu_20261029_preparation/results/`

## 结果

PER 按所有样本 phone error 数总和除以参考音素总数计算。逐条 RTF 使用 `time.perf_counter()` 测量模型解码调用耗时与对应音频时长的比值；表中为 2620 条逐条 RTF 的算术平均。不同机器或设备上的 RTF 不作直接性能对比。

| 顺序 | 模型配置 | 参考配置文件 | phone errors / 189023 | Corpus PER | 平均逐条 RTF |
|---:|---|---|---:|---:|---:|
| 1 | stride4，无动态压缩 | `server_phone69_460h_offline_native_stride4_none_s42_e50.yaml` | 11585 | 6.1289% | 0.04848 |
| 2 | stride1，block1 后 cosine_diff + learned 五帧融合 | `server_phone69_460h_offline_native_stride1_cosd_learned_s42_e50.yaml` | 10423 | 5.5141% | 0.06333 |
| 3 | stride2，block1 后 cosine_diff + learned 五帧融合 | `server_phone69_460h_offline_native_stride2_cosd_learned_s42_e50.yaml` | 12690 | 6.7135% | 0.03206 |
| 4 | stride4，block1 后 cosine_diff + learned 五帧融合 | `server_phone69_460h_offline_native_stride4_cosd_learned_s42_e50.yaml` | 14381 | 7.6081% | 0.02137 |
| 5 | stride1，Mel 输入后（第0层前）cosine_diff + learned 五帧融合 | `server_phone69_460h_placement_mel_stride1_s42_e50.yaml` | 10544 | 5.5782% | 0.06134 |
| 6 | stride1，第5层后 cosine_diff + learned 五帧融合 | `server_phone69_460h_placement_layer5_stride1_s42_e50.yaml` | 9037 | 4.7809% | 0.07215 |
| 7 | stride1，第7层后 cosine_diff + learned 五帧融合 | `server_phone69_460h_placement_layer7_stride1_s42_e50.yaml` | 9187 | 4.8603% | 0.07100 |
| 8 | stride1，第2层后 hard/radius1、第5层后 learned 五帧融合 | `server_phone69_460h_placement_layers2_5_n3_stride1_s42_e50_batch6000_resume.yaml` | 10247 | 5.4210% | 0.05847 |

本次测试集上 PER 最低的是 **stride1，第5层后 cosine_diff + learned 五帧融合**，PER 为 **4.7809%**（9037/189023）。该结果来自单次评测和单个 test-clean 测试集，不代表多个随机种子上的平均。

## 配置与模型维数

所有模型均使用 80 维输入特征、128 维 Conformer 主干、256 维编码器输出、8 个注意力头、512 维 FFN 和 31 点 depthwise convolution。stride×投影维数分别为 stride1×128、stride2×64 或 stride4×32，堆叠后输入主干宽度均为 128。编码器为 2+8 层结构或 placement 的 10 层结构。

固定 phone69 词表包含 69 个 ARPABET 音素，blank ID=69，joiner 输出 70 类。预测器为 128 维 token embedding、两层 128 隐维 LSTM，预测器输出与编码器输出同为 256 维。动态融合配置使用 five_frames learned mixer；Mel-placement 在模型 Mel 特征上进行融合，layers2+5 配置还在第2层后使用 hard/radius1 阶段。

## 数据与验证

- 8 个模型均对同一 test-clean 样本顺序完成 2620 条识别，各模型参考序列 SHA-256 一致。
- 每个模型均有 189023 个参考音素；逐行重新校验 idx 连续、phone error 与 PER 计数一致、RTF 有限且非负。
- 全部 checkpoint 的训练配置、词表、压缩定义及本地 state_dict 结构已在评测前通过核对；传输文件与服务器审计清单哈希一致。
- 数据清单为 2620 个 FLAC 与 87 个章节级标签文件；额外异常副本已隔离，正式评测使用的音频集合为 2620/2620 且与清单匹配。
- `results/COMPLETE` 标记存在；本报告数值从逐样本 CSV 重算，并与运行 summary 交叉核对。

## 文件

- 汇总：`artifacts/evaluations/phone69_460h_local_cpu_20261029_preparation/results/summary.json`
- 逐条结果：`artifacts/evaluations/phone69_460h_local_cpu_20261029_preparation/results/full/`
- 配置：`artifacts/evaluations/phone69_460h_local_cpu_20261029_preparation/configs_cpu/`
- 服务器审计与传输计划：`artifacts/evaluations/phone69_460h_local_cpu_20261029_preparation/`
