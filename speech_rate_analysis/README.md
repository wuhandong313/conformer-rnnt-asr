# test-clean 原始十档语速清单

本目录保存 test-clean 全部 2,620 句的原始语速及十档音频 ID。数据复制自本地 Emformer 项目，分档成员保持原样。

PPM = 原始空格分隔音素数 / FLAC 音频秒数 × 60。按 PPM 从慢到快排序，分为 10 个等句数分位档，每档 262 句；相同 PPM 的边界沿用原始顺序。

- deciles/：十个音频 ID 清单。
- speech_rates_test_clean.csv、speech_rates_test_clean.txt：逐句语速。
- speech_rate_summary.json：总体语速统计。

旧 keep-rate/上下文实验的分档结果与旧汇总脚本已删除。当前模型按同一清单重新关联逐句结果，指标口径见 [权威评测协议](../papers/phone69_460h_cpu_authoritative_20261108/README.md)。
