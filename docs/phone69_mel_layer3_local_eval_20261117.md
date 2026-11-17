# Mel＋L3家中CPU评测与全实验结果汇总

2026-11-17用户授权将已完训的Mel＋L3按既有家中CPU协议评测，并汇总所有实验的PER及固定30秒理论MAC。服务器仅只读获取训练记录、实际配置和checkpoint，不执行测试或新增训练。

本次模型完成50轮、91250次更新，按dev-clean验证损失选择`val-none-epoch=48-step=88800.ckpt`，SHA-256为`a905d1026801f3f68a750e6cc7179f8ea1fd2fa26982026e901a38127a56ddd2`。实际训练提交为`bbdbb44145d13508cbcdfd5583c322062e2fd63a`；在Git中冻结对应模型源码，严格加载权重，预期参数量4249296。

沿用test-clean全部2620句、189023个参考音素、19452.480625秒音频，CPU／FP32／batch1／beam20、4线程、interop1、worker0、预热10句。输入清单、标签、统计和词表使用同一冻结协议。运行目录为`artifacts/evaluations/phone69_mel_layer3_local_cpu_20261117/`，权重保留在忽略的`output/local_cpu_20261117/models/mel_layer3/`。

评测一次记录完整逐句PER、10层帧数、编码器与压缩六个MAC分项以及实际predictor／joiner计数。Mel与L3使用各自输入作为条件保留率的分母，再按既有公式推导3001帧（30秒）的编码器与压缩理论MAC，不将变长全量计数乘30，也不把实际beam20成本加入固定长度编码器指标。完成后独立重算2620句编辑距离并核验所有输入和结果哈希。

启动命令使用`scripts/evaluation/run_phone69_placement_cpu_batch.py --metadata <server_metadata.json> --names mel_layer3 --out artifacts/evaluations/phone69_mel_layer3_local_cpu_20261117`。新增显式多位置清单支持；原L1／L3／L9默认参数保持不变。直接启动Python，防休眠进程另行等待控制器PID，避免改变运行库环境。

总表保留所有已完成实验，并标记原论文入选／排除理由。L3训练回调best属于额外选模诊断，与原验证损失best分别列出，不覆盖历史结果；未完训和未核验的实验不填PER或30秒MAC。

本机验证：6项placement评测回归、9项理论MAC回归通过，含Mel＋L3两阶段宽度和压缩前后长度的独立计数；评测入口语法与Git whitespace检查通过。
