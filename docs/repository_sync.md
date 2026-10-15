# 原仓库分支与三端同步

2026-10-15 按用户要求收拢工作树，各实验保留为独立分支。默认工作分支为 `main`。

## 仓库位置

- 本地：`/Users/wuhandong/Desktop/python/conformer-rnnt-asr`
- 服务器：`/test21/conformer-rnnt-asr`
- Gitee：`wu3134/conformer-rnnt-asr`
- GitHub：`wuhandong313/conformer-rnnt-asr`，同步镜像。

三端同步以同名分支的完整提交 ID 和 Git 文件树一致为准。不同实验分支继续保留不同代码，切换实验时在原仓库选择对应分支。

九个分支均已整理文件布局，具体迁移表见各分支的 [FILE_LAYOUT.md](FILE_LAYOUT.md)。根目录保留运行入口和共享模块；小日志、评估结果、曲线、图表、checkpoint 路径记录归入 `artifacts/`，词表模型和特征统计归入 `assets/`，论文归入 `papers/`，辅助脚本归入 `scripts/diagnostics/` 与 `scripts/server/`，研究计划和报告归入 `docs/`。代码、配置与文档中的相关路径同时更新。

## 分支用途

| 分支 | 保留的内容 |
| --- | --- |
| main | 当前代码、严格 69 音素固定 tokenizer、IPA 支持、word-BPE，以及原服务器 main 的小日志、调试脚本、评估记录、词表文件 |
| ipa | 独立完整 Epitran IPA 方案及旧真实音频 smoke 记录；研究方向仍暂停 |
| attn-predictor | 注意力/压缩研究代码与实验记录，含服务器未提交的小日志、配置、清单和评估脚本 |
| attn-multihead | 原有注意力多头实验分支 |
| multihead | 原有多头实验分支 |
| streaming | 原有流式实验分支 |
| llm-compression | 服务器旧远程引用中保留的压缩实验提交 |
| master | 初始历史分支 |
| codex/server-original | 服务器原仓库曾经 detached HEAD 的历史实验状态，加上此前未入库的小日志、结果、配置和排队脚本 |

原服务器 main 的 WER 改进已包含于当前 main。合并时保留当前 phone69/IPA 的 PER、固定词表和 checkpoint 检查，避免用旧服务器评估代码覆盖这些功能。所有服务器原始提交均保留在分支历史中，没有重写历史。

## 文件范围

本次从四个服务器目录保存了 398 个此前未追踪的小文件，另保留已追踪文件的修改。日志、脚本、CSV、实验清单、配置、报告和小型词表文件均进入对应分支。

按单文件 **10 MiB（10,485,760 字节）** 为大文件界限，超过界限的 135 个文件留在服务器；清单见 [server_large_files.tsv](server_large_files.tsv)。登录凭据、虚拟环境、依赖下载缓存和 Python 运行缓存不入库。旧分支当前版本中的登录文档已取消追踪，本机登录文件继续保留。原先已追踪的大型论文 PDF 和 LLM tokenizer JSON 也已取消当前版本的追踪，副本保存在本地及服务器；历史提交不改写。

本地 `output/repository_sync/2026-10-15/` 中保存本次整理用的临时审计和传输材料，不属于项目源码。

## 服务器目录兼容与大文件

服务器仅保留原仓库作为 Git 工作目录。原 `-main`、`-ipa`、`-attn` 路径改为指向原仓库的兼容链接，保留旧脚本中的绝对路径可用性；它们不再代表不同的活动分支。运行实验前，必须在原仓库切换到相应分支。

从附加工作树移出的未入库文件保存在：

```text
/test21/conformer-rnnt-asr-worktree-archive-20261015/
  main/
  ipa/
  attn/
```

大日志按同样布局放在 `artifacts/logs/` 或原有实验结果目录，并以链接保留旧路径；不会将大文件推到远端。本来就在原仓库实验目录中的 checkpoint 和数据继续留在原位置。数据集 `/test21/librispeech` 是独立目录，本次不改动。

## 后续使用

在原仓库切换到目标分支后拉取该分支，再运行相应配置。日志即使命中旧的忽略规则，本次已经追踪的文件仍会正常同步；新小日志需要明确加入版本控制。超过 10 MiB 的文件保留在服务器，并记录路径。

切换分支前先保存当前修改。三端都以原仓库的分支管理实验，不再创建这些独立工作树。
