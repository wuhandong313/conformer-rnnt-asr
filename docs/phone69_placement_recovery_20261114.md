# 层位补充队列：环境重装与断点恢复

2026-11-14用户授权重新安装服务器环境并恢复任务。服务器持久化仓库、数据与checkpoint保留，`whd`训练Python缺失，无存活训练／队列进程，A100 GPU空闲。38个原部署源码和配置SHA全部一致；当前HEAD的后续论文归档变化不替代各实验实际启动commit。

| 串行次序 | 实验 | 恢复安排 |
| --- | --- | --- |
| 1 | L3 | 原预算8800 phone tokens；核实完整第1轮边界step1866后恢复优化器、scheduler和loop状态，继续至50轮 |
| 2 | L9 | 尚未启动，按原协议自动校准最大稳定batch，再从头50轮 |
| 3 | Mel+L3 | 尚未启动，独立校准后从头50轮 |
| 4 | Mel+L7 | 尚未启动，独立校准后从头50轮 |
| 5 | L2+L7 | 尚未启动，独立校准后从头50轮 |

L1已记录trained、epoch50、step99000及最终完整边界；恢复控制器还会实际读取其边界并核对best后跳过。L5、L7、Mel+L5、L2+L5继续复用原有完整实验，不重复训练。

环境沿用既有锁定版本：Python3.10.18、torch／torchaudio2.6.0+cu124、Lightning2.6.6和numpy2.2.6，依赖来自`artifacts/recoveries/phone69_post_restart_20261021/requirements-restored.txt`。环境重建、`pip check`及版本／CUDA可用性查询不运行软件测试；软件安全检查仅在本地运行。

L3原run记录第2轮、step2200，轮中完整验证／last checkpoint还记录到step2266。此次选择完整epoch边界以明确数据迭代边界，第2轮边界后的部分重新执行；较晚的轮中checkpoint仍完整保留，不称其为未保存。从原selected有效配置生成独立resume YAML，唯一变化为`paths.from_checkpoint`，预算8800、模型、reference后端及所有优化设置不变。

恢复前逐一独立复制L3全部旧checkpoint并校验SHA，不使用硬链接；同时备份旧run.json。原train.log和GPU日志不覆盖，新日志写入独立恢复目录。完成组检查trained／epoch50、best及标记、实际完整第50轮边界，未通过不推进。失败保留现场，不自动重启或降预算。

恢复入口为`scripts/server/recover_phone69_placement_supplement.py`，沿用原[队列清单](../configs/phone69_placement_supplement_20261112.json)。服务器独立目录为`output/phone69_460h_placement_recovery_20261114/`，以`recovery_state.json`、各组resume／selected记录、实际run与日志为准；原20261112队列状态保留为历史，不再据其旧PID判断进程。

[初始核对](../artifacts/recoveries/phone69_placement_supplement_20261114/initial_inspection.json)及[本地安全检查](../artifacts/recoveries/phone69_placement_supplement_20261114/local_validation.json)记录恢复依据。环境版本、选定断点、独立备份清单及实际恢复PID将在启动核实后归档。
