# Mel＋第5层＋第8层三次压缩：本地CPU评测

2026年11月8日完成 LibriSpeech test-clean 全量评测。三次压缩的语料级 PER 为 **7.0616%**，corpus RTF 为 **0.01923582**。

| 模型 | test-clean corpus PER | corpus RTF |
| --- | ---: | ---: |
| Mel单阶段 E05 | 5.5782% | 0.04794958 |
| 第5层单阶段 E06 | 4.7809% | 0.05061921 |
| Mel＋第5层双阶段 E10 | 5.6882% | 0.02449575 |
| **Mel＋第5层＋第8层三阶段** | **7.0616%** | **0.01923582** |

相对双阶段，本次三阶段 PER 增加 **1.3734 个百分点**，RTF 降低 **21.47%**。本次测量显示解码耗时更少、音素错误更多；该比较不构成压缩层数的单变量消融。

## 评测协议

- 本地CPU；同最新十模型串行评测使用 `/opt/anaconda3/envs/aligner/bin/python`，Python 3.12.9、PyTorch/TorchAudio 2.8.0。
- CPU intra-op 4线程、inter-op 1线程，BLAS/OpenMP均4线程，单一OpenMP运行库，`OMP_MAX_ACTIVE_LEVELS=1`；float32、batch1、data workers0、beam20。
- 先预热前10句，再完整评测2,620句。189,023个参考音素，真实FLAC总时长 19452.480625 秒。全量音频与章节标签SHA、uid顺序、逐句参考均核对。
- corpus PER = 13,348 / 189,023 = 0.0706157452。它是语料级音素错误率。
- 解码时间只计 `model(batch)` 的 `perf_counter_ns()` 墙钟时间，排除模型加载、数据读取和特征准备。corpus RTF = 374.184321047 / 19452.480625000 = 0.0192358151。
- 另保留历史口径的逐句RTF均值（以Mel帧时长为分母）：0.0204054069，不与上表的corpus RTF混用。
- 对照取自 `../phone69_460h_cpu_serial_20261108/summary.json`，该批此前已完成；本次仅运行新三阶段模型。RTF是相同固定协议下的一次本地测量，仍受系统负载和温度等运行条件影响。

## 模型与来源

- 仅按训练阶段保存的dev-clean验证损失选模：原阶段最低34.73215866，续训阶段最低12.73441982；未使用test-clean选模。
- 最佳文件：`val-none-epoch=47-step=59949.ckpt`（epoch从0开始，训练第48轮）。服务器来源：`/test21/conformer-rnnt-asr/exp_phone69_460h_offline_native_placement_mel_layer5_layer8_stride1_s42_b11050_to_b25000_e50/checkpoints/val-none-epoch=47-step=59949.ckpt`。
- checkpoint SHA-256：`8cd027fec53d5863bf95d93005fc5dcc516b5d79bd074999ec8d77e9e8100978`。下载后核对服务器指纹；大文件存于 `output/local_cpu_20261108/models/mel_layer5_layer8/`，不纳入Git。
- 训练源码固定于提交 `eedae42d4c22fd11b643cc82e88adec31085cd86` 的本地快照；未采用之后新增的batched实现。模型字典与训练recipe一致，`on_load_checkpoint` 与严格权重加载通过。
- 主干128维、10层按[5,3,2]分段，压缩位置[0,5,8]，cosine_diff无预处理，五帧learned融合，reference后端、stride1；参数量 4,269,973。
- 完整来源与哈希见 `manifest.json`、`selection.json`、`invocation.json`；逐句结果见 `utterances.csv`，指标见 `result.json`。

## 比较限制

本实验仅seed42，前5轮phone-token预算11,050，第6—50轮25,000；双阶段实验前16轮6,500，之后11,050，且存在轮中恢复的历史限制。批预算安排不同，单次结果不能单独归因于新增第8层压缩，也不代表统计显著性。RTF属于本地CPU reference后端测量，不代表GPU或batched后端速度。

首次启动在解码前因防休眠程序转发丢失OpenMP库路径而被检查拒绝；改为直接启动Python后单OpenMP库检查通过。本次正式评测一次完整跑完，退出码0。首次日志保留在 `output/phone69_460h/mel_layer5_layer8_local_eval_20261108/attempt_01_runtime_failure/`。

完整CSV已独立只读复核：2,620句逐句音素编辑距离重算、参考与顺序一致，汇总指标与哈希全部通过，详见 `audit.json`。
