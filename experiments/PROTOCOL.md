# 实验管理规范（Experiment Protocol）

## 1. 分支策略

- `main`：只进**稳定的方法代码与正式实验配置**，不直接在 main 上跑新实验。
- 实验分支：`exp/<主题>`，**从 main 创建**（如 `exp/a3-baseline-seeds`）。一个主题一个分支，分支内所有实验 commit。
- 工程改造分支：`feat/<内容>`（如向量化 `feat/vectorize-compression`），验证输出与原实现逐元素一致后合回 main。
- 服务器只部署 `exp/*` 和 `main` 分支，靠 git 分支做"工作分区"。

## 2. 实验命名与目录

- 实验 ID：`<phase>-<tag>-s<seed>`，如 `a3-none-s42`、`b1-savgol-s42`。phase 对应 research-plan.md 的实验编号（A/B/C/D…）。
- 每个实验一个配置文件：`configs/exp/<实验ID>.yaml`（从 configs/server.yaml 复制后只改必要字段）。
- 每个实验一个独立输出目录：配置里 `paths.exp_dir: /test21/conformer-rnnt-asr/exp/<实验ID>/`——ckpt、日志互不污染，可随时整目录删除。

## 3. Checkpoint 纪律（不能存太多）

- `save_top_k: 1`（val 与 train 各最多 1 个，按监控指标自动替换旧的最差）。
- `val_checkpoint_interval: 500`、`train_checkpoint_interval: 2000`。
- `save_weights_only: true`（已默认，不含优化器状态）。
- 实验结束并回填 LOG 后，只保留 best ckpt；同配置已确认结论的旧实验目录可整目录删除（删除前在 LOG 备注保留日期）。

## 4. 启动与追踪流程

1. 本地：写 `configs/exp/<ID>.yaml` → 在 `experiments/LOG.md` 登记一行 → commit 到实验分支 → push gitee。
2. 服务器：`git pull` 实验分支 → `nohup python train.py configs/exp/<ID>.yaml > exp/<ID>/train.log 2>&1 &`（exp 目录先建好）→ 记下 PID 回填 LOG 备注。
3. 查看进度：`tail -f /test21/conformer-rnnt-asr/exp/<ID>/train.log`。
4. 结束：跑 eval → 把 val_loss / PER（以及 WER，如果已有）回填 LOG"关键结果"列 → 状态改为 `已完成`。
5. 失败：把报错最后一屏贴进 LOG 备注，状态 `失败`，修复后新 ID 重新登记（不覆盖旧行）。

## 5. 结果表约定

- 所有最终数字以 eval 输出为准，写论文时从 LOG.md 汇总，不凭记忆。
- 多种子实验：每个种子是独立一行，汇总时在论文草稿/分析表里合并。
