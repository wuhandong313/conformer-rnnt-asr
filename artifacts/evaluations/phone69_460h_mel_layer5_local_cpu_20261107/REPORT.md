# Mel ＋第 5 层压缩：本地 CPU 评测

## 结果

| 方法 | test-clean corpus PER | 平均逐句 RTF |
| --- | ---: | ---: |
| 原 Mel 单阶段 | 5.5782% | 0.06134 |
| Mel ＋第 5 层双阶段 | 5.6882% | 0.06778 |
| 第 5 层单阶段 | 4.7809% | 0.07215 |

本次双阶段相对原 Mel 单阶段 PER 高 0.1100 个百分点（相对增加约 1.97%）。三者使用相同 test-clean 参考顺序，但本次评测期间本机有其他负载；RTF 仅是本次平均逐句解码计时，不作为严格速度比较。

## 协议与选模

- LibriSpeech test-clean：2,620 句，189,023 个参考音素；Phone69，beam width 20，本地 Apple M4 CPU，4 线程，数据加载 worker 为 0。
- 语料级 PER 为总音素错误数 10,752 / 总参考音素数 189,023。全部参考文本与旧 Mel 基线逐句一致。
- 按各训练阶段保存的 dev-clean 验证损失选模：原阶段最低 13.01498，桥接阶段最低 13.06862，调批后阶段最低 10.21157；选择调批后 `epoch=49, step=89901` checkpoint。这里的 epoch 从 0 开始计数。
- checkpoint SHA-256：`6e88fdb52fe9ab06d4e2eea7e031c0cff19a4d4409da32aa1acca76e0a75c21c`；大文件留在 `output/`，不纳入 Git。
- 使用训练代码提交 `a05b80c3aacff5417b25bedca818c8054e1cb67d`，评测脚本快照见 `eval_cpu.py`，配置与完整指标见 `configs_cpu/mel_layer5.yaml`、`summary.json`。

## 比较限制

该实验为单 seed；第 16 轮后 phone-token 批预算由 6,500 调至 11,050，RNNT 损失使用 sum，训练过程并非与原 Mel 基线完全相同。桥接阶段从轮中 checkpoint 恢复时，Lightning 提示数据加载器不支持精确轮中恢复，可能重复或跳过部分样本。双阶段参数量为 4,249,296，原 Mel 单阶段为 4,228,619。以上结果不能单独归因于压缩位置，也不代表统计显著性。
