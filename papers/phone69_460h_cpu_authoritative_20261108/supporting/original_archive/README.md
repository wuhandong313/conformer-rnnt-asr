# 460h Phone69 本地 CPU 串行评测权威结果（2026-11-08）

本目录按用户要求保存为本批实验的权威结果，供论文、图表和后续比较引用。权威范围为冻结清单中 E01–E10 的指定 checkpoint，在下述固定 CPU 协议下对完整 LibriSpeech test-clean 的评测。引用数据以 [summary.csv](summary.csv) 为入口，并以各模型的 `result.json` 和逐句 CSV 为依据。

## 完成与核验

- 批次完成时间：北京时间 2026-11-08 02:08:37；10/10 成功，全部为 `attempt_01`，失败和重试均为 0。
- 每个模型独立完成 2,620 句，189,023 个参考音素，音频总时长 19,452.480625 秒；E01–E10 串行执行。
- 原始核验于北京时间 2026-11-08 02:10:09 通过：逐句独立重算音素编辑距离、PER、RTF 与全部汇总一致；历史预测变化均为 0。
- 原始核验包括配置、checkpoint、代码、音频、章节标签、逐句 CSV 和汇总的哈希及一致性检查，详见 [verification.json](verification.json)。
- 归档时复核了原始结果记录的哈希、10 组配置与 checkpoint、统计量、词表和执行源码；复制后逐文件核验字节一致性。归档整体校验见 `SHA256SUMS`。

## 固定评测协议

- 设备：本地 CPU；float32；PyTorch/BLAS 4 线程，interop 1 线程，OpenMP 最大活动层级 1。
- batch 1，data workers 0，beam width 20；每个模型预热 10 句，再完整测量 2,620 句。
- RTF = 各句 `model(batch)` 的单调墙钟耗时总和 / FLAC 音频时长总和。计时范围为模型调用，预热、模型加载和数据准备位于计时范围外。
- PER = 全语料音素编辑距离总和 / 参考音素总数；词表为固定 69 个带重音 ARPABET 音素，blank ID 69。
- 主基线是 **E09：输入投影 128 维、stride4、无动态压缩**。`RTF_over_E09` = 当前模型 corpus RTF / E09 corpus RTF。
- Python 3.12.9，torch 2.8.0，torchaudio 2.8.0，arm64；完整环境与 OpenMP 库哈希见 `manifest.json`。

## 权威指标

下表为展示用舍入值，CSV 与 JSON 保留原始浮点精度。

| 实验 | 配置 | 输入投影维度 | Corpus RTF | RTF/E09 | Corpus PER |
|---|---|---:|---:|---:|---:|
| [E01](E01/result.json) | stride4 无动态压缩 | 32 | 0.031155520 | 0.784535 | 6.128884% |
| [E02](E02/result.json) | stride1 第2层五帧融合 | 128 | 0.049379514 | 1.243439 | 5.514144% |
| [E03](E03/result.json) | stride2 第2层五帧融合 | 64 | 0.026329756 | 0.663016 | 6.713469% |
| [E04](E04/result.json) | stride4 第2层五帧融合 | 32 | 0.017706603 | 0.445875 | 7.608069% |
| [E05](E05/result.json) | Mel 端五帧融合 | 128 | 0.047949579 | 1.207431 | 5.578157% |
| [E06](E06/result.json) | 第5层五帧融合 | 128 | 0.050619207 | 1.274656 | 4.780900% |
| [E07](E07/result.json) | 第7层五帧融合 | 128 | 0.050418336 | 1.269597 | 4.860255% |
| [E08](E08/result.json) | 第2层邻域3帧＋第5层五帧融合 | 128 | 0.046488417 | 1.170637 | 5.421033% |
| [E09](E09/result.json) | stride4 无动态压缩 512维主干（主基线） | 128 | 0.039712068 | 1.000000 | 5.742687% |
| [E10](E10/result.json) | Mel＋第5层双阶段五帧融合 | 128 | 0.024495745 | 0.616834 | 5.688197% |

## 文件与溯源

- [AUTHORITATIVE.json](AUTHORITATIVE.json)：权威范围、主基线、归档时间、原始核验摘要及复制文件清单。
- [summary.csv](summary.csv)、[summary.json](summary.json)：10 个模型的完整指标、相对 E09 比值与结果路径。
- [manifest.json](manifest.json)、[verification.json](verification.json)：原样保存的冻结输入清单和原始独立核验记录。
- `E01/` 至 `E10/`：各模型 `config.yaml`、`training_config.yaml`、`result.json`、完成状态，以及 `attempt_01/utterances.csv` 和 `attempt_01/run.log`。
- `model_source/`：冻结模型源码与当次执行脚本；`inputs/`：实验台账、固定词表和归一化统计量。
- [input_references.json](input_references.json)：原始 checkpoint、数据集、历史预测、统计量和词表的位置与哈希；配置保留当次运行的原始路径。
- `run_metadata/`：当次完成状态与历史定时启动记录。
- `SHA256SUMS`：本目录全部归档文件（除该校验清单自身）的 SHA-256。

在本目录运行以下命令可检查归档完整性：

```sh
shasum -a 256 -c SHA256SUMS
```

原始评测目录：`artifacts/evaluations/phone69_460h_cpu_serial_20261108`。归档时间：`2026-11-08T10:05:24+08:00`。
冻结 manifest SHA-256：`7bfe4ed442fb0cf7815a686448e5a6aae8de84abcc1673d43c7ed1bd69429acb`。

后续引用这批 CPU 评测时，以本目录保存的版本为准。新增实验按独立目录与协议记录，保留本次归档作为引用依据。
