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

## 完成结果

三组于2026-11-17北京时间11:54、12:10、12:24依次完成；独立核验于12:25通过。每组2620句的编辑距离、corpus PER、参考覆盖、逐层长度及MAC分项完成复算，输入和结果哈希核对通过。[固定30秒主指标及完整记录](../artifacts/evaluations/phone69_placement_l1_l3_l9_local_cpu_20261117/REPORT.md) · [独立核验与分项计数](../artifacts/evaluations/phone69_placement_l1_l3_l9_local_cpu_20261117/verification.json)。

| 插入位置 | Corpus PER | 编码器＋压缩GMAC／30秒 | 音素错误数 |
| --- | ---: | ---: | ---: |
| L1 | 5.452247% | 10.956676 | 10306 |
| L3 | 6.393931% | 16.311692 | 12086 |
| L9 | 4.921094% | 31.840138 | 9302 |

PER按完整原始test-clean统计；30秒理论MAC按本次完整逐层帧数校准，计数范围为编码器和压缩模块。每组`fixed_duration_30s.json`保存条件保留率、推导长度与分项MAC。实际beam20完整搜索的predictor／joiner成本另见下表，单位为完整语料总MAC除以真实音频秒；两种口径分别报告。

| 插入位置 | 编码器与压缩 | Predictor | Joiner | 实际搜索合计 |
| --- | ---: | ---: | ---: | ---: |
| L1 | 0.247019 | 0.163460 | 0.024207 | 0.434687 |
| L3 | 0.342296 | 0.162161 | 0.024304 | 0.528762 |
| L9 | 0.617352 | 0.149038 | 0.021987 | 0.788377 |

## 2026-11-19 L3 从头复跑

原 L3 在恢复第 1 轮完整边界后完成 50 轮，验证损失最佳在第 27,324 步。为检验 6.393931% 的 test-clean PER 能否重现，复跑沿用实际训练配方的 seed 42、8,800 音素批上限、单阶段 L3 五帧融合及 50 轮，从零初始化并使用独立目录。排在 stride 4／256 维之后，待 GPU 空闲再确认真实训练批。选模仍用 dev-clean 验证损失，后续用同一套 test-clean、FP32、batch 1、beam 20 协议对照两次结果。[复跑配方](../configs/server_phone69_460h_placement_layer3_stride1_s42_e50_rerun20261119.yaml) · [队列入口](../scripts/server/append_phone69_layer3_rerun.py)
