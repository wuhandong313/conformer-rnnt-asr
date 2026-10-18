# test-clean 原始十档语速清单

复制自本地 Emformer 项目的 speech_rate_analysis。评测集合为同一份 test-clean（2620 句），与训练使用 100h 或 460h 无关。

PPM = 原始空格分隔音素数 / FLAC 音频秒数 × 60。按 PPM 从慢到快排序，分为 10 个等句数分位档，每档 262 句；相同 PPM 的边界沿用原始顺序。保留原始 CSV、TXT、JSON 和十个 ID 清单，不重新计算或改变分档成员。

历史 460h 三组统计位于 `eval_deciles_460h_作废_旧方法/`，同内容的归档位于 `artifacts/evaluations/phone69_460h_testclean_20261017/speech_rate_deciles_作废_旧方法/`。这三组属于已作废的旧 keep_rate/上下文实验；当前四组 `offline_native` 模型尚未完成全队列评测，不混用历史结果。

复核脚本：`scripts/diagnostics/aggregate_speech_rate_deciles.py`、`scripts/diagnostics/report_speech_rate_deciles.py`。统计保留大小写和重音完整 phone，PER 与 S/D/I 以参考音素数为分母，RTF 使用有效单调时钟复测值的逐句算术平均。
