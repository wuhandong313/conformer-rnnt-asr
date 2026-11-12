# 固定 stride 队列重启恢复与本地评测

用户于 2026-11-11 明确要求重启环境后等待十分钟，重装环境、继续未完成训练，并将完成组拉到本地按既有协议评测。首次远端操作在客户端 UTC 14:51:20 之后进行。[启动、版本和传输记录](../artifacts/recoveries/phone69_stride_sweep_20261111/startup.json)

服务器持久化目录保留完好，原 `whd` Python 缺失。恢复为 Python 3.10.18、torch/torchaudio 2.6.0+cu124、Lightning 2.6.6，依赖使用此前记录的精确版本；`pip check` 通过、CUDA 可用。服务器未运行软件测试或额外显存校准。恢复控制脚本、评测脚本及本地安全检查均先通过 Git 同步后使用。

stride1、stride2 已完成50轮。恢复程序实际读取各组最终完整边界 checkpoint，确认 `resume_completed_epochs=50`、优化器及 scheduler 状态齐全、最佳checkpoint与标记一致。这两组不再训练；最佳checkpoint已下载到本地，并核对服务器与本地 SHA-256。

stride8 的重启前记录为第9轮、step14000，最近完整边界为第8轮、step13544。保持投影128、主干1024、10层、seed42、总50轮和预算9700，从该完整边界恢复；第9轮未保存部分重做。恢复配置只改变 `paths.from_checkpoint`，原实验路径继续使用，以保留最佳验证checkpoint选择状态。旧run.json、原13个不可变checkpoint和旧日志均保留；恢复日志位于独立目录。

滚动 `last.ckpt` 的备份最初也用了硬链接，随后随续训从原尺寸1,146,804,752字节更新为1,146,804,816字节、step14344，因此未保住它重启前的逐字节副本。已改为独立复制，并增加本地覆盖写入保护检查；原第8轮完整边界及第9轮step13744完整验证checkpoint均保持原样，后者包含模型、优化器和scheduler状态。此归档疏漏不影响所选第8轮边界恢复，但不将原滚动last文件声称为字节不变。

原控制器提交354baeb、stride2启动提交14c7b9、stride8原启动提交c4485c不同，但29个原训练源码与配置文件SHA全部一致。差异仅为后续文档、数据工具与恢复/评测工具更新，不再仅凭Git HEAD不同判训练源码改变。实际续训启动提交为 `191e387c7b82aaa8eb51b02396b231478418ffc3`；后续提交不替代该启动记录。

服务器恢复运行目录为 `/test21/conformer-rnnt-asr/output/phone69_460h_stride_sweep_recovery_20261111/`，`recovery_state.json`、各恢复日志及完成后生成的`completion.json`为状态依据。已启动控制器PID604、训练PID740只作为初始记录，监控核对实际状态。恢复脚本拒绝重复启动，遇失败保留现场，不自动降预算或重启。

本地目录为 `artifacts/evaluations/phone69_460h_stride_sweep_local_cpu_20261111/`，独立于冻结E01–E10归档。队列控制器串行评测stride1、stride2，并等待stride8完成后自动下载及评测。使用同一冻结test-clean的2,620句、189,023个参考音素、19,452.480625秒音频，FP32、CPU、batch1、beam20、4线程、interop1、worker0。一次解码同时统计corpus PER和主要MAC，分开报告编码器/压缩、predictor、joiner；不生成RTF或重复计时结果。只在完整语料核实通过后报告该模型的正式指标，尚未完成的组不填估计值。

本地恢复安全与指标聚合检查共9项通过；三个实际模型在同一真实音频上，执行计数与已独立审计的固定stride编码器公式一致。此项为软件预检，不是完整语料指标。运行中的本地进度见各组 `attempt_01/progress.json`，完整结果为`result.json`、`utterances.csv`及`summary.json`。

三组均已完成50轮训练及本地全量PER／主要MAC评测，见[最终三组报告](../artifacts/evaluations/phone69_460h_stride_sweep_local_cpu_20261111/REPORT_complete.md)。服务器stride8最终step84,650，实际读取最终边界确认已完成50轮且优化器、scheduler状态完整；最佳验证checkpoint为 `val-none-epoch=21-step=36553.ckpt`，下载SHA已核对。本地stride8于客户端UTC 2026-11-12 01:17:45完成，PER为6.678552%，总主要GMAC／音频秒为1.312000。三组逐句编辑距离、完整语料覆盖、编码器公式和实际解码路径矩阵MAC均已独立核验，`EVALUATION_COMPLETE`已生成。[较早两组报告](../artifacts/evaluations/phone69_460h_stride_sweep_local_cpu_20261111/REPORT_stride1_stride2.md)保留为当时的阶段记录。

完训后恢复状态将 `active_stride` 清空，而状态查询脚本仍用它查找run，导致一次 `run_config_mismatch`误报；已修正本地只读查询，在complete状态按唯一配置路径定位run。原始运行配置和续训配置SHA未变，29个原训练源码／配置SHA全部通过，实际续训YAML仍仅改变恢复checkpoint路径。最终本地控制器的`pending_strides=[8]`是上轮缓存字段；其`status=complete`、`completed_models=[1,2,8]`、三份完整结果和完成标记一致，归档保留原始状态字段，不据此重复评测。
