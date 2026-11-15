# 后三组恢复在当前服务器串行执行

2026-11-15用户明确将后三组实验重新安排在当前服务器。现行顺序为L3→L9→Mel+L3→Mel+L7→L2+L7。20261114的跨服务器清单与prompt保留为历史，本会话未在其他服务器实际启动过三组；本次按它们尚未在他机启动处理，不再安排迁移。

初始检查：L3训练PID557存活，第47轮、global_step87000、预算8800，原本机接管控制器PID1415仍运行；L9未启动，后三组在当前服务器均无实验目录或run。42个原冻结文件没有改变，GPU正常工作。

不停止现有控制器、训练进程，不改动它们冻结的清单或源码。新增`scripts/server/append_phone69_placement_queue.py`等待原L3/L9控制器正常完成并退出。只有在L3、L9均实际核实trained、epoch50、best及完整第50轮边界后，才执行[后三组当前服务器清单](../configs/phone69_placement_current_server_three_20261115.json)。前序失败、进程消失却未完整结束、配置或源码变化时停止，不自动恢复或降预算。

三组配方与此前Git跟踪的基础YAML、基线完全相同，reference后端、stride1、投影／主干128、10层、seed42、50轮保持原样。L2+L7仍为首阶段hard三帧邻域、第二阶段learned五帧融合。每组开跑前按原真实数据前后向／裁剪／Adam协议独立校准最大稳定phone-token预算，至少24步压力确认后再从头训练；不复制L3的8800预算。

等待控制状态位于服务器`output/phone69_460h_placement_append_20261115/append_state.json`，三组校准／训练状态位于`output/phone69_460h_placement_current_server_three_20261115/queue_state.json`。不同runtime目录保证保留历史记录，原实验输出路径未改动；已有同名实验目录时拒绝覆盖。注册后的实际PID与状态将记录在`artifacts/runs/phone69_placement_return_local_20261115/startup.json`。

本地4项安全检查通过，覆盖等待存活前序、拒绝失败／非完整退出、缺失L9完成记录不得读作成功、仅包含三项新训练且配方与原清单相同。服务器不运行软件测试。
