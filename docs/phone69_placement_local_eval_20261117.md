# L1／L3／L9家中CPU评测

2026-11-17用户授权在家中电脑按既有流程评测已完训的三个插入位置。使用各自dev-clean选出的最佳checkpoint，顺序L1→L3→L9；测试集为冻结test-clean全部2620句、189023个参考音素、19452.480625秒音频。协议为CPU、FP32、batch1、beam20、4线程、interop1、worker0、预热10句，与既有全量评测一致。

| 位置 | 最佳验证checkpoint | SHA-256 | 完训步数 |
| --- | --- | --- | ---: |
| L1 | val-none-epoch=44-step=88720.ckpt | bdb28193b75d348b6b3273140d182fe1b6401ec73f5d9cdeb7b7f00146900b75 | 99000 |
| L3 | val-none-epoch=14-step=27324.ckpt | 34f99288beef3f8e2b4318804297412f145e71875d41c99af427fecd74ac8e79 | 93300 |
| L9 | val-none-epoch=49-step=117150.ckpt | 5890e73762175447f7ba86ba6291893184a30febcae14101086b4ef61a35d2cc | 117500 |

三组均为stride1、投影／主干128、10层、单阶段learned五帧融合，位置分别为第1／3／9层输出之后。L3保留其完整第1轮边界恢复历史，评测模型配置严格使用实际训练配方。

入口为`scripts/evaluation/run_phone69_placement_cpu_batch.py --metadata <server_metadata.json>`，需在家中仓库使用`/opt/anaconda3/envs/aligner/bin/python`运行。服务器元数据由既有SSH跳板只读获取；仅下载checkpoint等运行资产，程序与配置通过Git同步。checkpoint放于忽略目录`output/local_cpu_20261117/models/`，训练源码按各自启动提交从Git冻结。

评测目录为`artifacts/evaluations/phone69_placement_l1_l3_l9_local_cpu_20261117/`。每组保存manifest、原始和CPU配置、源码哈希、逐句CSV、日志、进度及最终result；控制器串行执行并以`EVALUATION_COMPLETE`标记全部完成。完整结果核验前不报告PER或填入论文曲线。

一次完整解码同时记录逐层帧数、压缩输入／输出帧数和主要MAC，分列编码器／压缩、predictor、joiner。每句编码器六个分项与独立结构公式逐一核对；汇总核对UID顺序、参考音素总数、逐句错误总数与CSV落盘结果。计数不生成RTF、重复计时或设备加速比。保留逐层帧数以支持固定30秒理论MAC的后续校准，实际beam搜索成本与固定长度理论编码器成本继续使用各自明确的口径。

软件检查在本机进行，服务器只读获取完训记录、配置和权重，不执行软件测试或新增训练。
