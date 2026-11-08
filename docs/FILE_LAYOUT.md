# 当前文件布局

本分支保留 LibriSpeech 460h、固定 Phone69 的离线五帧融合与多阶段压缩实验。训练集为 train-clean-100 + train-clean-360；100h 独立实验、九帧、IPA、Word-BPE 和旧 keep-rate/diff5 实验已清理。

| 目录 | 内容 |
| --- | --- |
| conformer_rnnt/ | 编码器、预测器、Joiner、模型工厂和 Lightning 数据/训练模块 |
| conformer_rnnt/compression/ | 峰谷检测、五帧融合、平滑及多阶段压缩 |
| configs/ | 当前 460h 训练配方和固定 Phone69 词表 |
| scripts/server/ | 当前实验的训练、续训和状态辅助脚本 |
| scripts/evaluation/ | 本地 CPU 批量评测、计算量测量和核验 |
| scripts/diagnostics/ | 当前压缩方法的诊断工具 |
| tests/ | 当前方法、Phone69、指标与恢复逻辑的回归测试 |
| assets/statistics/ | 特征归一化统计量 |
| artifacts/evaluations/ | 逐句结果、日志、配置及执行快照 |
| artifacts/analysis/ | 帧保留率等分析结果 |
| artifacts/runs/ | 当前实验的启动、续训、完成及溯源记录 |
| artifacts/research/ | 计算量、机制分析与实验台账 |
| artifacts/recoveries/ | 当前实验重启恢复证据 |
| docs/ | 方法说明、实验历史和研究计划 |
| papers/ | 参考论文、稿件及权威评测归档 |
| speech_rate_analysis/ | test-clean 的原始语速及十档句子 ID |
| output/ | 本地模型、数据与运行工作文件，Git 忽略 |

训练入口为根目录 train.py，评估入口为 eval.py，共享工具为 common.py。train.py 接受一个 YAML 路径；当前配方显式设置 model.sequence_mode: offline、tokenizer.type: phone69。

当前结果入口见 [460h 结果](phone69_460h_results.md)。E01–E10 指标以 [权威归档](../papers/phone69_460h_cpu_authoritative_20261108/README.md) 为准。归档中的冻结源码和历史输入保持原字节，用于复现当次结果，不作为当前开发入口。

软件测试仅在本地运行。服务器代码与配置通过 Git 同步，数据集、checkpoint 和超过 10 MiB 的文件保留在 Git 外。
