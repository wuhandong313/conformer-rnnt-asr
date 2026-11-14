# 将三组双阶段实验迁移到其他服务器

2026-11-14用户明确将Mel+L3、Mel+L7、L2+L7从当前服务器的后续队列中移出，改由其他服务器执行。原部署JSON与源码保持原字节，以保留运行指纹；不能只改JSON，因为原恢复控制器已经将后续项载入内存。

当前服务器由`scripts/server/continue_phone69_placement_local.py`接管：核对原队列父进程及L3训练进程身份，仅向旧队列父PID发送SIGTERM，不发送给训练PID或进程组；核实同一L3 worker仍存活后等待其严格完成。L3正常结束并通过50轮完整边界／best核验，才启动[仅L9清单](../configs/phone69_placement_current_server_layer9_20261114.json)。失败保留现场并停止。新的控制记录在服务器`output/phone69_460h_placement_local_handoff_20261114/`，L9原型校准及运行状态在`output/phone69_460h_placement_current_server_layer9_20261114/`。原状态文件保留为历史。

接管已执行并核实：新控制器PID1415，旧恢复控制器PID538已退出；L3原训练PID557保持不变，预算8800，已观察第2轮step2900继续推进。三个迁移项均尚未在当前服务器创建run，现行控制状态明确将它们列为`deferred_to_other_server`，本机剩余仅`layer9`。[接管核验记录](../artifacts/runs/phone69_placement_server_split_20261114/handoff.json)

其他服务器使用[仅三组清单](../configs/phone69_placement_remote_three_20261114.json)，顺序为Mel+L3→Mel+L7→L2+L7。三个action均为train，不会运行L1／L3／L9，也不要求复制旧四组实验的checkpoint。三个完整YAML、相应基线、队列程序、真实数据batch校准程序及环境锁定版本都由Git跟踪；数据、权重、凭据和环境本体不入Git。

配置采用原规范路径：仓库`/test21/conformer-rnnt-asr`、数据`/test21/librispeech`及其`global_stats.json`，词表`configs/phone69_vocab.json`。目标服务器可使用等价兼容路径；需要改路径时，应生成**新名称**的配置与相应基线、三组清单，先在本地提交并通过Git部署，不修改当前服务器运行指纹所引用的原YAML。仅适配路径，不改变模型、融合、训练、tokenizer或后端。

目标服务器准备好固定Phone69的train-clean-100／train-clean-360、dev-clean、外部逐章`*-ph.trans.txt`、同SHA归一化统计和词表。原环境依赖锁为`artifacts/recoveries/phone69_post_restart_20261021/requirements-restored.txt`，Python3.10.18、torch／torchaudio2.6.0+cu124、Lightning2.6.6。所有软件测试在本地运行；目标GPU只运行获授权的真实训练校准与50轮训练。新硬件上独立校准batch，不沿用当前服务器的8800预算。

路径与环境准备好后，服务器仓库根目录中的核心启动命令为：

```bash
nohup /root/anaconda3/envs/whd/bin/python -u \
  scripts/server/phone69_placement_supplement.py \
  configs/phone69_placement_remote_three_20261114.json --mode run \
  > output/phone69_placement_remote_three_20261114.launch.log 2>&1 < /dev/null &
```

应先确保`output/`存在，确认该清单的runtime目录及三个新实验目录尚未存在，也无相同队列／训练进程。不要在已有运行目录上重复执行。使用独占GPU0；该版GPU等待器和日志查询都按物理GPU0，不能只改CUDA_VISIBLE_DEVICES却不对齐GPU查询。控制器串行校准、至少24次真实优化器步骤确认、再从头训练50轮；训练失败不自动降预算或重启。

可复制给新服务器会话的prompt：

```text
请在我指定的目标服务器启动Phone69 460h三组补充实验，只跑Mel+L3、Mel+L7、L2+L7，串行执行。目标服务器连接方式和数据路径由本会话提供；凭据只由忽略的remote-access.md和已有包装器读取，不打印凭据、SSH配置或全量进程命令。

从Gitee wu3134/conformer-rnnt-asr（GitHub镜像wuhandong313/conformer-rnnt-asr）取得main分支并核对本次交接的完整commit。使用configs/phone69_placement_remote_three_20261114.json，不能用旧的十项总清单，也不要重跑L1、L3、L9或已完成参考实验。本次三组都是新训，不需要旧参考组checkpoint。

先检查GPU、驱动、磁盘、固定Phone69数据与标签、词表和归一化统计。默认仓库/test21/conformer-rnnt-asr，数据/test21/librispeech。若路径不同，创建新名称的配置、对应基线和三项清单，仅适配路径，在本地Git提交/推送后服务器fetch并快进部署；不直接复制部署代码或改动原始YAML。数据使用train-clean-100+360，验证dev-clean，Phone69外部逐章*-ph.trans.txt；不能混用train-other-500的IPA数据。

安装原锁定环境：Python3.10.18、torch/torchaudio2.6.0+cu124、Lightning2.6.6，按artifacts/recoveries/phone69_post_restart_20261021/requirements-restored.txt安装并pip check，确认CUDA可用。软件测试只在本地运行，不在GPU训练服务器跑测试。不要擅自升级驱动、切后端或改batch保护规则。

原模型固定stride1、投影/主干128、10层、seed42、50轮，保持各原YAML的模型和优化设置。Mel+L3/Mel+L7两阶段都是五帧learned融合；L2+L7首阶段hard三帧邻域(neighborhood_radius=1)，第二阶段五帧learned融合；后端保持reference。

确认没有相同队列/训练进程、GPU0可用，且新runtime/实验目录不已存在，再通过scripts/server/phone69_placement_supplement.py和这份三项清单启动一个可脱离SSH的串行控制器。每组在目标GPU上自动搜索最大稳定phone-token预算，按100 tokens细化，经真实数据前向、反向、裁剪、Adam及至少24步压力确认再从头训练；不要复制其他服务器的batch预算。仅使用独占物理GPU0，确保训练和GPU查询指向同一卡。

保留日志和全部checkpoint，记录commit、源码/配置SHA、GPU、PID、校准过程、选定预算和启动状态。训练失败/OOM保留现场并停止，不自动降预算或重启。每组完成必须核实trained、epoch50、best存在、实际完整第50轮边界和优化器/scheduler状态，再推进下一组。启动后确认日志真实推进再汇报。后续评测使用冻结test-clean全量corpus PER和主要MAC/音频秒，分编码器/压缩、predictor、joiner，FP32、batch1、beam20；不做RTF或重复计时。
```
