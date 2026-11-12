# 层位补充实验：串行训练与最大 batch 校准

2026-11-12用户授权补充单层L1／L3／L5／L7／L9，双层Mel+L5／Mel+L3／Mel+L7、L2+L5／L2+L7，并在准备中的论文加入实验和自然语速分组比较。已有训练和评测直接复用。[机器可读队列](../configs/phone69_placement_supplement_20261112.json)

| 布局 | 处理 | 对齐基线 |
| --- | --- | --- |
| L1、L3、L9 | 新训 | 原L5，档案E06 |
| L5、L7 | 复用 | 档案E06／E07 |
| Mel+L3、Mel+L7 | 新训 | 原Mel+L5，档案E10 |
| Mel+L5 | 复用 | 档案E10的完整恢复训练链 |
| L2+L7 | 新训 | 原L2+L5，档案E08 |
| L2+L5 | 复用 | 档案E08的完整恢复训练链 |

所有新组为stride1、投影128、主干128、10层、seed42、50轮，从头训练。相对各自模板只移动`after_layer`，另使用独立实验输出路径和用户授权的自动batch预算。数据、训练优化器、scheduler、SpecAugment、验证间隔、数值精度及reference执行后端均保持对应配置。L2+L7首阶段保持hard、`neighborhood_radius:1`，第二阶段保持五帧learned融合；Mel组合两阶段均为五帧learned。新单阶段及L2+L7参数4,236,299，Mel双阶段参数4,249,296。

控制器每组先以真实train-clean-100／360及dev-clean压力批次执行前向、反向、裁剪和Adam步，指数扩张后按100 phone tokens细化，再作至少24步重复压力确认。使用A100 80GiB，reserved显存上限为96%；token预算受原RNNT元素保护限制。所谓最大batch是该分批规则与压力样本下观测到的最大稳定预算，不是对所有可能语音组合的全局证明。保留每次测量，包括正常校准中的OOM；选定预算后正式训练不自动降预算、不自动重启。

队列使用全局串行锁；保留已存在的实验和旧日志，不覆盖。先核实四个复用组的trained／epoch50、best、完整第50轮边界及checkpoint SHA，再运行六个新组。每个新组完成后实际读取最终边界并检查完整状态，只有通过核验才推进到下一组。新组实验路径均带`supp20261112`，运行记录位于服务器`output/phone69_460h_placement_supplement_20261112/`。

原四组在服务器均已确认trained、epoch50、best存在；冻结配置、本地best checkpoint、全量逐句CSV哈希已核对。E08历史预算4000→6000，E10为6500→11050，保留这些训练历史，不声称与自动校准的新组严格batch或训练暴露匹配。

软件检查仅在本地进行：现有层位回归12项及队列安全检查8项通过，六个完整新模型的层位、参数量及有限前后向通过。服务器仅运行获授权的真实数据batch校准和正式训练，不运行软件测试。论文语速比较复用既有完整预测与MAC档案；新增组未完成前保持pending，不填写估计值。

队列已于2026-11-12启动，部署commit为`8ef8eef970362ecdaa184a58d61460ec3f4d1c3c`，控制器PID5031；四个复用组已实际通过完整边界与最佳checkpoint核验，首组L1正在进行batch校准。初始校准预算5000不是最终选定预算。后续PID及进展以`queue_state.json`、校准报告、selected配置和各实验run.json核对。[启动档案](../artifacts/runs/phone69_placement_supplement_20261112/startup.json)记录当时快照，后续只含论文或归档的HEAD变化不替代该实际启动commit。
