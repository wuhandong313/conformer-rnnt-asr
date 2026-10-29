# 串行 CPU 评测启动记录

启动记录时间：2026-10-29 16:40:25 CST。本记录为启动时快照；当前进度请查看 `results/full/` 中的各模型日志与 `results/summary.json`。

- 用户已授权在检查三端同步性后开始传输和评测。
- 本地 main、Gitee main 与 GitHub main 提交均为 `06740b72ca5d7a0c737c901c2fa776c56d4dabe6`，本地与两端远程全部 10 个分支提交一致。
- 服务器 main 为 `bae6d2c2170c1c1db9f4b1df4fa6743cc29b34f2`，落后本地 11 个提交；其余现存 8 个分支与本地对应分支一致，服务器缺少 `codex/aishell1-lexicon`。服务器有训练相关未提交修改，本次未重置或切换分支。
- 服务器8个 best checkpoint、配置和训练统计重新校验均与原计划一致；数据和模型下载后的 SHA-256 校验通过。
- CPU 本地8组 checkpoint 均通过严格加载、压缩定义与 phone69 词表检查；每组2条 smoke 全部通过。
- 8组完整 test-clean 评测已串行启动；运行目录 `output/local_cpu_20261029/`，结果位于本目录 `results/`。首次运行进程 PID 52860（进程号为启动快照）。
- 服务器没有 rsync，传输改为经 SSH 分批流式打包，失败后对已有文件进行 SHA-256 核验并续传。

完整运行会自动逐模型产生日志与结果，通过汇总验证后创建 `results/COMPLETE`。结果缺失或没有 COMPLETE 时不要视为全部完成。
