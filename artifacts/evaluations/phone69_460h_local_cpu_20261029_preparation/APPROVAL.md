# 本地 CPU 串行评测审批清单

状态：**等待用户审批**。核验时间：2026-10-29 14:33:32 北京时间。当前未下载 checkpoint、统计文件或测试数据，未运行 smoke 或全量 eval。

本轮按最近主线的 **8 组 phone69 / 460h / offline_native** 完成实验准备，每组选择训练时 dev-clean 验证选出的 best checkpoint。保留服务器文件，仅下载所需副本。历史 100h、word-BPE 和早期注意力实验另列库存，尚未纳入本次审批范围。

## 模型与顺序

| 顺序 | 实验 | best 文件中的 epoch / step | 大小 MiB | 训练配置 |
| --- | --- | --- | ---: | --- |
| 1 | stride4_none | 41 / 46858 | 48.59 | server_phone69_460h_offline_native_stride4_none_s42_e50.yaml |
| 2 | stride1_cosd_learned_five_frames | 27 / 69137 | 48.92 | server_phone69_460h_offline_native_stride1_cosd_learned_s42_e50.yaml |
| 3 | stride2_cosd_learned_five_frames | 20 / 24540 | 48.86 | server_phone69_460h_offline_native_stride2_cosd_learned_s42_e50.yaml |
| 4 | stride4_cosd_learned_five_frames | 44 / 43524 | 48.83 | server_phone69_460h_offline_native_stride4_cosd_learned_s42_e50.yaml |
| 5 | placement_mel_stride1 | 48 / 121688 | 48.83 | server_phone69_460h_placement_mel_stride1_s42_e50.yaml |
| 6 | placement_layer5_stride1 | 45 / 151700 | 48.92 | server_phone69_460h_placement_layer5_stride1_s42_e50.yaml |
| 7 | placement_layer7_stride1 | 46 / 118826 | 48.92 | server_phone69_460h_placement_layer7_stride1_s42_e50.yaml |
| 8 | placement_layers2_5_n3_stride1 | 25 / 73182 | 48.92 | server_phone69_460h_placement_layers2_5_n3_stride1_s42_e50_batch6000_resume.yaml |

epoch 使用 checkpoint 文件与框架的零起始编号；8 组训练均已到 50 个完整 epoch。best 的完整服务器路径、SHA-256、字节数、原始配置哈希和本地目标路径见 `plan.json`。两层实验使用实际续训的 `batch6000_resume` 配置。

## 已完成检查

- 服务器端：8 个 best checkpoint 可读，参数全为有限值；逐个配置通过 `on_load_checkpoint` 的词表/压缩定义检查和 `load_state_dict(strict=True)`。最后完整边界的 `ready == completed == 50`。
- 原始配置 SHA-256 与各自 `run.json` 一致，现存本地 provenance 配置副本与服务器哈希一致。
- 本地：两个隔离源码副本从已有 Git 历史与启动清单按文件 SHA-256 恢复，覆盖五帧与 placement 两组。8 个本地模型工厂的 state_dict 键/形状/dtype、offline_native 定义和词表指纹均与服务器核验结果一致。未加载本地 checkpoint，未做识别推理。
- 固定 phone69 69 类，blank=69；保留大小写、重音和空格，明确过滤 spn；采用每章节 `*-ph.trans.txt`，不使用 SentencePiece。
- 测试数据已在服务器逐文件 SHA-256 审计：2620 条 FLAC、87 个章节标签文件、189023 个参考音素；音频均为 16 kHz 单声道，共约 5.404 小时。
- CPU 环境可导入并构建所有模型：Python 3.12、torch 2.8.0、torchaudio 2.8.0、pytorch-lightning 2.6.1。通过进程专用 `DYLD_LIBRARY_PATH` 统一 OpenMP 库；不设置 `KMP_DUPLICATE_LIB_OK`，不修改已有 conda 环境。
- 准备脚本与隔离源码语法检查通过；runner dry run 通过；聚合、非有限 RTF 拒绝、索引顺序和下载文件损坏拒绝检查通过。

## 统计文件差异必须处理

服务器训练统计 SHA-256：`022a496264bafdba8ecb65f72d71a579c76f6f8c56e41866a600272fc3b1c457`。

仓库已有统计 SHA-256：`be2f80418ef66bad30dcd5556e194f95234d75c5d2b94a92d7a766d31900df87`。

二者不同，因此将服务器训练统计单独下载到 `output/local_cpu_20261029/assets/global_stats.json`，所有 CPU 配置只引用这份经哈希核验的文件。保留仓库现有统计文件。当前本地工厂检查仅验证结构，构造时临时使用现有统计文件；正式加载与评测会强制使用下载后的训练统计，缺失或哈希不符即停止。

## 审批后的传输范围

- 8 个 best checkpoint：409,763,536 字节，390.78 MiB。
- 仅 test-clean 音频与对应 phone69 标签：367,767,069 字节，350.73 MiB。
- 服务器训练统计：3888 字节。词表已有一致本地副本，无需重新下载。
- 合计 **777,534,493 字节 / 741.51 MiB**，约 0.78 GB；当前本地可用约 33 GiB。
- 不下载训练集、last/epoch-boundary checkpoint、优化器的额外副本或历史大日志。best 文件保持原样，不做剥离或转换。
- 本地目标根目录：`output/local_cpu_20261029/`；大文件均位于已忽略路径，不加入 Git。

## CPU 配置与执行流程

原始配置保存在 `configs_original/`，CPU 配置保存在 `configs_cpu/`。仅调整数据/统计/输出/词表路径、清空训练恢复路径、CPU accelerator/devices/strategy 与 data.num_workers=0；model、training、decoding 和其余 data 字段保持一致。

进程锁保证同一目标目录只有一个串行执行器。固定 torch 4 线程、interop 1、测试 DataLoader 0 worker、batch_size=1、beam width=20、reference 压缩实现。模型按表中顺序运行，逐项等待退出。

1. 下载后逐文件核验 SHA-256/字节数，检查8个 checkpoint 的词表、压缩定义、参数形状和有限性；任何不匹配立即停止。
2. 8 组各串行识别前2条样本，结果单独保存在 `results/smoke/`；全部通过后进入全量。
3. 8 组各完整评测 test-clean 2620 条，结果保存在 `results/full/`；输出每条 PER/RTF/参考/预测和汇总。
4. 汇总使用总 phone_errors / 总 reference_phones 的 corpus PER；逐条检查索引、参考序列一致性、189023 参考音素、PER 计数和非负有限 RTF。RTF 使用 `time.perf_counter`，报告每条 RTF 的算术平均；另记进程墙钟耗时。CPU RTF 与此前共享 A100 评测结果分开解释。
5. 出错即停止，并保留日志和已有结果。已有结果文件禁止覆盖，需检查失败原因后再决定重试。

当前未运行样本识别，暂不估计全量 CPU 耗时；下载后可用 smoke 的实测耗时做初步估计。

## 可审查文件与命令

- `server_preflight.json`：只读服务器审计和完整测试文件清单。
- `local_architecture_five_frames.json`、`local_architecture_placement.json`：本地结构对齐检查。
- `plan.json`：固定下载路径/哈希、8组配置、源码指纹和执行参数。
- `audit_server.py`、`prepare_plan.py`、`check_local.py`、`run_plan.py`：可复查实现。
- `run_plan.py` 不带执行开关只展示计划。以下命令仅在收到用户批准后运行：

```bash
/opt/anaconda3/envs/aligner/bin/python artifacts/evaluations/phone69_460h_local_cpu_20261029_preparation/run_plan.py --execute-after-approval --stage all
```

## 未纳入范围

- 九帧 stride1：本次初始在线库存时仍在训练，epoch=7 / step=16600；未纳入完成模型。
- 历史 phone69 100h：服务器有完成记录和 checkpoint；本轮8组460h对比之外，尚未重新做配置与 checkpoint 审计。
- 历史 word-BPE、注意力和更早模型：有 checkpoint 库存，但没有本轮可直接复用的训练完成/配置追溯审计，见 `all_existing_checkpoints_inventory.json`。
- 已作废的非 offline_native 460h 记录指向的 best 文件不存在。

第7层模型在严格审计时未见 TRAINING_COMPLETE，但 run 与完整50轮边界均通过；后续全目录库存中已出现 best_checkpoint.txt。完成状态以已验证的训练进度为依据，不把 marker 缺失当作未训练完成，也未补写服务器标记。
