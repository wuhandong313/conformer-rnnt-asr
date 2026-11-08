# 460h Phone69 本地 CPU 评测权威结果（2026-11-08）

本目录为用户指定的权威结果，供论文、图表和后续比较引用。权威范围为冻结清单中 E01–E10 的指定 checkpoint，在下述固定协议下对完整 LibriSpeech test-clean 的本地 CPU 评测。

## 引用入口与资料分层

- **权威指标**：[summary.csv](summary.csv)、[summary.json](summary.json)。顶层只保存权威说明、标记和汇总。
- **完整副本与原始核验依据**：[supporting/original_archive/](supporting/original_archive/)。该子文件夹原样保存此前 96 个归档文件，包括逐句 CSV、配置、源码、日志、冻结清单和原始核验报告。
- **本次整理的核验资料**：[supporting/verification/](supporting/verification/)，包含整理后的一致性核验和完整校验清单。
- **研究计划**：[ACL 2027 论文与补实验计划](supporting/research/phone69_acl_2027_plan_20261107.md)，保存 2026-11-07 的计划正文，已调整仓库内相对链接。计划中的历史 RTF 用于记录当时判断；当前指标以顶层权威汇总为准。
- 顶层汇总与原始汇总的全部指标一致，仅将 `config`、`training_config`、`csv`、`log` 路径更新为对应子文件夹。原始字节和校验清单保存在完整副本中。

## 批次与固定协议

- 完成时间：北京时间 2026-11-08 02:08:37；E01–E10 串行，10/10 成功，无失败或重试。
- 每模型：test-clean 全部 2,620 句、189,023 个参考音素、19,452.480625 秒音频；预热 10 句后完整计时。
- CPU、float32、PyTorch/BLAS 4 线程、interop 1 线程、OpenMP 最大活动层级 1；batch 1、data workers 0、beam 20。
- Corpus RTF = 各句 `model(batch)` 单调墙钟耗时总和 / FLAC 音频时长总和；模型加载、数据准备和预热在计时范围外。
- Corpus PER = 全语料音素编辑距离总和 / 参考音素总数；固定 69 个带重音 ARPABET 音素，blank ID 69。
- **E09 为主基线：输入投影 128 维、stride4、无动态压缩**。`RTF_over_E09` = 当前模型 RTF / E09 RTF。
- Python 3.12.9、torch/torchaudio 2.8.0、arm64；环境、OpenMP 库哈希和模型输入见完整副本中的冻结清单。

## 权威指标

下表为展示用舍入值；CSV/JSON 保留原始浮点精度。实验编号链接到原始模型结果。

| 实验 | 配置 | 输入投影维度 | Corpus RTF | RTF/E09 | Corpus PER |
|---|---|---:|---:|---:|---:|
| [E01](supporting/original_archive/E01/result.json) | stride4 无动态压缩 | 32 | 0.031155520 | 0.784535 | 6.128884% |
| [E02](supporting/original_archive/E02/result.json) | stride1 第2层五帧融合 | 128 | 0.049379514 | 1.243439 | 5.514144% |
| [E03](supporting/original_archive/E03/result.json) | stride2 第2层五帧融合 | 64 | 0.026329756 | 0.663016 | 6.713469% |
| [E04](supporting/original_archive/E04/result.json) | stride4 第2层五帧融合 | 32 | 0.017706603 | 0.445875 | 7.608069% |
| [E05](supporting/original_archive/E05/result.json) | Mel 端五帧融合 | 128 | 0.047949579 | 1.207431 | 5.578157% |
| [E06](supporting/original_archive/E06/result.json) | 第5层五帧融合 | 128 | 0.050619207 | 1.274656 | 4.780900% |
| [E07](supporting/original_archive/E07/result.json) | 第7层五帧融合 | 128 | 0.050418336 | 1.269597 | 4.860255% |
| [E08](supporting/original_archive/E08/result.json) | 第2层邻域3帧＋第5层五帧融合 | 128 | 0.046488417 | 1.170637 | 5.421033% |
| [E09](supporting/original_archive/E09/result.json) | stride4 无动态压缩 512维主干（主基线） | 128 | 0.039712068 | 1.000000 | 5.742687% |
| [E10](supporting/original_archive/E10/result.json) | Mel＋第5层双阶段五帧融合 | 128 | 0.024495745 | 0.616834 | 5.688197% |

## 核验与溯源

- [原始独立核验报告](supporting/original_archive/verification.json)：北京时间 2026-11-08 02:10:09 核验通过；逐句重算 PER/RTF 与模型结果、汇总一致，历史预测变化均为 0。
- [冻结清单](supporting/original_archive/manifest.json)：配置、checkpoint、音频、章节标签及源码的 SHA-256。
- [输入位置与哈希](supporting/original_archive/input_references.json)：checkpoint、数据集、历史预测、词表和统计量的原始路径。
- [完整副本文件说明](supporting/original_archive/README.md)：各模型 `E01/`–`E10/` 下的逐句 CSV、日志、配置和结果，以及 `model_source/`、`inputs/` 和 `run_metadata/`。
- [整理后核验](supporting/verification/archive_verification.json)：确认权威指标与原始指标完全一致、引用路径有效、完整副本原始校验通过。

在本目录检查全部归档文件：

```sh
shasum -a 256 -c supporting/verification/SHA256SUMS
```

冻结 manifest SHA-256：`7bfe4ed442fb0cf7815a686448e5a6aae8de84abcc1673d43c7ed1bd69429acb`。
原始评测目录：`artifacts/evaluations/phone69_460h_cpu_serial_20261108`。
首次归档：`2026-11-08T10:05:24+08:00`；分层整理：`2026-11-08T10:20:51+08:00`。

后续引用本批 CPU 指标时，以顶层权威汇总为准；副本与核验资料用于溯源。
