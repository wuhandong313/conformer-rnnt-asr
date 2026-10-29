# phone69 460h 训练证据索引（2026-10-29）

本目录是服务器只读快照，包含八个已完成实验及正在训练的九帧实验的运行记录、配置、分段训练指标、显存日志，以及启动、预检和恢复材料。原服务器路径、文件大小、SHA-256、是否随快照复制及本地位置见 `file_manifest.json`；checkpoint 的路径、SHA-256 和校验详情见 `checkpoint_audit.json`。

服务器根目录为 `/test21/conformer-rnnt-asr`。单文件超过 10 MiB 的训练日志、所有 checkpoint 和音频数据保留在服务器；不把它们纳入 Git。运行中的九帧日志、metrics.csv 和 run.json 仅为快照时间的前缀与状态，训练结束后需要补充最终记录。复制时的服务器时间见 manifest；`stable` 只表示读取时文件元数据未变化，不代表后续不会继续增长。

## 核查范围与结论

对八个完成实验逐一检查了配置内容与 run.json 中 SHA-256 一致；最佳 checkpoint 与最后完整 epoch 边界 checkpoint 可读取；模型参数有限，state_dict 与对应配置严格匹配，offline_native_definition 和 tokenizer 加载检查通过，并包含优化器、调度器。8 个最后边界的 epoch current.ready == current.completed == 50。两次续训的来源 checkpoint 也通过相同检查。九帧实验核验的是启动时的 epoch 0 边界，并非最终结果。

第 7 层实验 `run.json` 记为 trained、最后完整边界为 50；服务器未发现 `TRAINING_COMPLETE` 标记。此索引不创建或补写标记，也不把不存在的标记记为已存在。

## 续训和指标解释

- 两层压缩的恢复点是 epoch 3 / step 12414；第一次使用预算 4000，恢复后预算 6000。`version_0` 指标可能记录到恢复边界之后，`version_1` 从恢复点重新推进；不能把两段的全部行直接相加作为唯一训练步序列。
- 第 7 层恢复点是 epoch 29 / step 73399；`version_0` 部分指标已走到该点之后，恢复后 `version_1` 重新推进。应按恢复点分割两段并保留重叠部分作为审计记录。
- `metrics.csv` 的 epoch 列使用框架零起始计数；`run.json` 的训练进度显示为 1–50。多个版本和 step 0 的初始化指标均予保留。

## 代码追溯边界

`run.json` 中的 code_commit 记录启动时 Git HEAD，不必然包含当时工作区未提交改动。请同时参照各次 `prelaunch_validation.json` 的 `source_sha256`。本次比对四个有源码清单的启动门槛：先前五帧门槛与当前不同的六个文件都可从 `bae6d2c` 恢复；分层门槛与当前源码唯一不同的 `mixers.py` 也可从 `bae6d2c` 恢复；九帧门槛所列源码与快照时当前源码一致。逐文件哈希应在使用时再次校验；不能仅凭 Git HEAD 断言旧运行完全复现。

本次验证未重新训练，也未重新测评；checkpoint 校验不保证数据、数值环境或 GPU 的跨环境可复现性。作废旧方法的记录保留原来的作废标识，不混入当前结果。
