# 分支文件布局

训练、评估、推理入口及共享运行模块保留在根目录，内部网络模块在 `conformer_rnnt/`。

| 目录 | 内容 |
| --- | --- |
| artifacts/logs/ | 小型运行、训练、评估日志 |
| artifacts/evaluations/ | 逐句指标、评估 CSV/JSON 及可复现实验快照 |
| artifacts/training/ | 损失曲线数据 |
| artifacts/figures/ | 图表 |
| artifacts/checkpoints/ | 最优 checkpoint 的路径记录，不含大模型 |
| assets/tokenizers/ | 小型词表模型 |
| assets/statistics/ | 特征统计文件 |
| scripts/diagnostics/ | 性能探测与调试工具 |
| scripts/server/ | 服务器辅助脚本 |
| papers/ | 参考论文 |
| docs/ | 研究计划与报告 |
| tests/ | 验证脚本与测试 |

仅创建本分支实际需要的目录。服务器统一使用 `/test21/conformer-rnnt-asr`，旧工作树目录仅作兼容链接。运行命令以仓库根目录为工作目录。超过 10 MiB 的文件留在服务器，凭据和运行缓存不入库。

## 本次文件迁移

| 原路径 | 新路径 |
| --- | --- |
| `2012.10128v2.pdf` | `papers/references/2012.10128v2.pdf` |
| `iter8-architecture.html` | `docs/reports/iter8-architecture.html` |
